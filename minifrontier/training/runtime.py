"""Deterministic data cursors, distributed routing balance and atomic checkpoints."""

from __future__ import annotations

import os
import random
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import torch
import torch.distributed as dist

from minifrontier.storage import reserve_write, tensor_payload_bytes


def validation_indices(size, *, limit, seed, rank=0, world_size=1):
    """A stable uniform subset, with no global RNG changes or duplicated DDP samples.

    A zero limit selects the entire held-out set. A file prefix can be sorted by
    domain or identity examples, so it is not a representative validation sample.
    """
    if size < 1 or limit < 0 or not 0 <= rank < world_size:
        raise ValueError("invalid validation size, limit or rank")
    order = torch.randperm(size, generator=torch.Generator().manual_seed(seed)).tolist()
    return order[: min(size, limit) if limit else size][rank::world_size]


class BatchCursor:
    """A global sample stream sharded across ranks, independent of DataLoader workers."""

    def __init__(self, size, seed, rank=0, world_size=1, batch_size=1, offset=0):
        if min(size, world_size, batch_size) < 1:
            raise ValueError("dataset, world size and batch size must be positive")
        self.size, self.seed = size, seed
        self.rank, self.world_size, self.batch_size = rank, world_size, batch_size
        self.offset = offset
        self.epoch, self.order = -1, torch.empty(0, dtype=torch.int64)

    def next(self):
        result = []
        for j in range(self.batch_size):
            absolute = self.offset + self.rank * self.batch_size + j
            epoch, position = divmod(absolute, self.size)
            if epoch != self.epoch:
                self.order = torch.randperm(
                    self.size, generator=torch.Generator().manual_seed(self.seed + epoch)
                )
                self.epoch = epoch
            result.append(int(self.order[position]))
        self.offset += self.batch_size * self.world_size
        return result


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(v) for key, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with reserve_write(path, tensor_payload_bytes(value)):
        fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
        temporary = Path(name)
        try:
            with os.fdopen(fd, "wb") as f:
                torch.save(cpu_tree(value), f)
                f.flush()
                os.fsync(f.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def rng_state(device):
    return dict(
        python=random.getstate(),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    )


def restore_rng(state, device):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if device.type == "cuda" and state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)


class RouterBalance:
    """Auxiliary-loss-free correction updated once per optimizer step across all ranks.

    Hooks exist only around forward calls, so checkpoint replay cannot double-count.
    Rate is a local recipe value, not an undocumented official hyperparameter.
    """

    def __init__(self, model):
        from minifrontier.models.minideepseekv4.upstream_layers import Gate
        from minifrontier.models.minikimik3.upstream_layers import KimiMoEGate

        self.gates: list[tuple[Any, torch.Tensor, int]] = []
        for module in model.modules():
            if isinstance(module, KimiMoEGate):
                self.gates.append((module, module.e_score_correction_bias, 0))
            elif isinstance(module, Gate) and module.bias is not None:
                self.gates.append((module, module.bias, 1))
        for module in model.modules():
            if isinstance(module, Gate) and getattr(module, "bias_vl", None) is not None:
                self.gates.append((module, cast(torch.Tensor, module.bias_vl), 1))
        self.counts = [torch.zeros_like(bias) for _, bias, _ in self.gates]
        self.handles = []
        self.valid_mask = None

    @contextmanager
    def capture(self, valid_mask):
        self.valid_mask = valid_mask
        try:
            with self:
                yield self
        finally:
            self.valid_mask = None

    def __enter__(self):
        for (module, bias, index), counts in zip(self.gates, self.counts, strict=True):

            def collect(mod, args, output, index=index, counts=counts, bias=bias):
                with torch.no_grad():
                    selected = output[index]
                    valid = torch.ones(selected.shape[0], device=selected.device, dtype=torch.bool)
                    mask = getattr(mod, "valid_mask", None)
                    mask = self.valid_mask if mask is None else mask
                    if mask is not None:
                        valid &= mask.flatten().bool()
                    image = getattr(mod, "route_image_mask", None)
                    if image is not None:
                        valid &= image if bias is mod.bias_vl else ~image
                    selected = selected[valid]
                    counts.add_(torch.bincount(selected.flatten(), minlength=counts.numel()))

            self.handles.append(module.register_forward_hook(collect))
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    @torch.no_grad()
    def update(self, rate):
        for (_, bias, _), counts in zip(self.gates, self.counts, strict=True):
            if dist.is_initialized():
                dist.all_reduce(counts)
            bias.add_(rate * torch.sign(counts.mean() - counts))
            counts.zero_()

    def state_dict(self):
        return {"counts": self.counts}

    def load_state_dict(self, state):
        for counts, saved in zip(self.counts, state["counts"], strict=True):
            counts.copy_(saved)
