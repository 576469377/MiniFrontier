"""One actual-capacity two-GPU update; engineering acceptance, never TensorBoard.

Run via torchrun --standalone --nproc_per_node=2. No checkpoints are written.
This is not a formal recipe, a quality evaluation, or evidence of full stages.
"""

import argparse
import json
import os
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from minifrontier.models.miniqwen4 import MiniQwen4Config, MiniQwen4ForCausalLM
from minifrontier.training.miniqwen4_optim import MiniQwen4Optimizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/miniqwen4.json")
    parser.add_argument("--sequence-length", type=int, default=128)
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "0")) != 2:
        raise ValueError("requires exactly two torchrun ranks")
    if not 2 <= args.sequence_length <= 128:
        raise ValueError("acceptance sequence length must be in [2, 128]")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    torch.set_num_threads(2)
    free, _ = torch.cuda.mem_get_info(rank)
    if free < 20 * 2**30:
        raise RuntimeError("capacity acceptance requires at least 20 GiB free per selected GPU")
    dist.init_process_group("nccl", timeout=timedelta(minutes=10))
    try:
        values = json.loads(Path(args.config).read_text())
        values["ple_layer_ids"] = tuple(values["ple_layer_ids"])
        cfg = MiniQwen4Config(**values)
        if not cfg.gradient_checkpointing:
            raise ValueError("capacity acceptance requires activation checkpointing")
        torch.manual_seed(149)
        model = MiniQwen4ForCausalLM(cfg).cuda(rank)
        ddp = DistributedDataParallel(
            model, device_ids=[rank], broadcast_buffers=False, gradient_as_bucket_view=True
        )
        optimizer = MiniQwen4Optimizer(model, lr=0.001, adam_lr=0.0002)
        generator = torch.Generator(device=f"cuda:{rank}").manual_seed(811 + rank)
        ids = torch.randint(
            3, cfg.vocab_size, (1, args.sequence_length), generator=generator, device=f"cuda:{rank}"
        )
        before = model.lm_head.weight[:8].detach().clone()
        torch.cuda.synchronize()
        start = time.monotonic()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = ddp(ids, labels=ids)
        if not torch.isfinite(result.loss):
            raise RuntimeError("nonfinite loss")
        result.loss.backward()
        optimizer.step()
        if torch.equal(before, model.lm_head.weight[:8]):
            raise RuntimeError("LM head did not update")
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        # Compare all parameters exactly without allocating another full model.
        for parameter in model.parameters():
            for chunk in parameter.detach().reshape(-1).split(1024 * 1024):
                reference = chunk.clone()
                dist.broadcast(reference, src=0)
                if not torch.equal(chunk, reference):
                    raise RuntimeError("parameters diverged across ranks")
        print(
            json.dumps(
                dict(
                    kind="engineering_acceptance_not_formal_training",
                    rank=rank,
                    config=str(Path(args.config).resolve()),
                    sequence_length=args.sequence_length,
                    local_batch_size=1,
                    optimizer_updates=1,
                    parameters_without_mtp=sum(p.numel() for p in model.parameters()),
                    loss=result.loss.item(),
                    lm_loss=result.lm_loss.item(),
                    aux_loss=result.aux_loss.item(),
                    update_seconds=elapsed,
                    max_memory_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                    max_memory_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                    exact_rank_parameters=True,
                    checkpoint_written=False,
                    tensorboard_written=False,
                )
            ),
            flush=True,
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
