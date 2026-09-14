"""V4.1 §2.5/Algorithm 1: semantic Muon, embedding Sinkhorn and scalar AdamW.

FP32 optimizer states and matrix normalization are used on consumer GPUs. This
implements the reported update rules, not DeepSeek's distributed optimizer.
"""

import math
from collections import defaultdict
from itertools import pairwise

import torch

from .semantic_optim import newton_schulz


def sinkhorn_direction(direction, *, iterations=11, threshold=1e-3, eps=1e-20):
    """Algorithm 1; zero rows remain zero, final rows have unit RMS."""
    if direction.ndim != 2 or iterations < 1 or iterations % 2 != 1:
        raise ValueError("Sinkhorn requires a matrix and an odd positive iteration count")
    value = direction.float().clone()
    norms = value.norm(dim=1, keepdim=True)
    value.masked_fill_(norms <= threshold * norms.mean(), 0)
    for iteration in range(iterations):
        dimension = 1 if iteration % 2 == 0 else 0
        value.div_(value.norm(dim=dimension, keepdim=True).add_(eps))
    return value.mul_(math.sqrt(value.shape[1]))


class V41Optimizer(torch.optim.Optimizer):
    recipe = "v41-headwise-muon-sinkhorn-v1"
    coefficients = ((3.4445, -4.775, 2.0315),) * 8 + ((2.0, -1.5, 0.5),) * 2

    def __init__(self, model, *, lr, weight_decay=0.1, eps=1e-20, vision_lr=None):
        if not math.isfinite(lr) or lr <= 0 or eps <= 0 or weight_decay < 0:
            raise ValueError("invalid V4.1 optimizer settings")
        metadata = model.optimizer_metadata()
        groups = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            spec = metadata[name]
            kind = spec["kind"]
            if kind not in {"muon", "sinkhorn", "adamw"}:
                raise ValueError(f"unrecognized optimizer owner: {name}: {kind}")
            blocks = None
            if kind in {"muon", "sinkhorn"} and parameter.ndim != 2:
                raise ValueError(f"{kind} requires a matrix: {name}")
            if kind == "muon":
                heads = spec.get("heads", 1)
                if parameter.shape[0] % heads:
                    raise ValueError(f"head partition does not divide rows: {name}")
                width = parameter.shape[0] // heads
                blocks = tuple(
                    tuple(b)
                    for b in spec.get(
                        "blocks", [(i * width, (i + 1) * width) for i in range(heads)]
                    )
                )
                if not blocks or blocks[0][0] != 0 or blocks[-1][1] != parameter.shape[0]:
                    raise ValueError(f"incomplete matrix partition: {name}")
                if any(a >= b for a, b in blocks) or any(
                    left[1] != right[0] for left, right in pairwise(blocks)
                ):
                    raise ValueError(f"overlapping or unordered matrix partition: {name}")
            scale = float(spec.get("lr_scale", 1.0))
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError(f"invalid parameter learning rate scale: {name}")
            base = lr
            if (
                vision_lr is not None
                and name.startswith("vision.")
                and not name.startswith(("vision.aligner.", "vision.merger."))
            ):
                base = vision_lr
            # Norm weights decay; bias, learned scalar gains and embeddings do not.
            decay = (
                weight_decay
                if kind == "muon"
                or (
                    kind == "adamw"
                    and (
                        spec.get("weight_decay_norm", False)
                        or ("norm" in name.lower() and name.endswith("weight"))
                    )
                )
                else 0.0
            )
            groups.append(
                dict(
                    params=[parameter],
                    name=name,
                    kind=kind,
                    blocks=blocks,
                    lr=base * scale,
                    base_lr=base * scale,
                    lr_scale=scale,
                    weight_decay=decay,
                    recipe=self.recipe,
                    role=spec.get("role", kind),
                    coefficients=self.coefficients,
                    scaling=0.18,
                    sinkhorn_iterations=11,
                    sinkhorn_threshold=1e-3,
                )
            )
        super().__init__(groups, dict(lr=lr, momentum=0.95, betas=(0.9, 0.95), eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        active = [g for g in self.param_groups if g["params"][0].grad is not None]
        if any(g["params"][0].grad.is_sparse for g in active):
            raise ValueError("V4.1 optimizer requires dense gradient tensors")
        if (
            active
            and not torch.stack([torch.isfinite(g["params"][0].grad).all() for g in active]).all()
        ):
            raise FloatingPointError("optimizer requires finite gradients")
        buckets = defaultdict(list)
        for group in active:
            p = group["params"][0]
            state = self.state[p]
            grad = p.grad.float()
            state["step"] = state.get("step", 0) + 1
            if group["kind"] == "adamw":
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                b1, b2 = group["betas"]
                state["exp_avg"].lerp_(grad, 1 - b1)
                state["exp_avg_sq"].mul_(b2).addcmul_(grad, grad, value=1 - b2)
                mean = state["exp_avg"] / (1 - b1 ** state["step"])
                variance = state["exp_avg_sq"] / (1 - b2 ** state["step"])
                update = mean / (variance.sqrt() + group["eps"])
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.to(p.dtype), alpha=-group["lr"])
                continue
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.float32)
            beta = group["momentum"]
            state["momentum_buffer"].lerp_(grad, 1 - beta)
            direction = grad.mul(1 - beta).add_(state["momentum_buffer"], alpha=beta)
            if group["kind"] == "sinkhorn":
                update = sinkhorn_direction(
                    direction,
                    iterations=group["sinkhorn_iterations"],
                    threshold=group["sinkhorn_threshold"],
                    eps=group["eps"],
                )
                p.add_(update.to(p.dtype), alpha=-group["lr"] * group["scaling"])
            else:
                for start, end in group["blocks"]:
                    part = direction[start:end]
                    buckets[(part.shape, part.device)].append((p[start:end], part, group))
        for rows in buckets.values():
            for offset in range(0, len(rows), 32):
                chunk = rows[offset : offset + 32]
                updates = newton_schulz(torch.stack([row[1] for row in chunk]), self.coefficients)
                updates.mul_(0.18 * math.sqrt(max(updates.shape[-2:])))
                for (p, _, group), update in zip(chunk, updates, strict=True):
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.to(p.dtype), alpha=-group["lr"])
        return loss

    def manifest(self):
        return [
            {key: value for key, value in group.items() if key != "params"}
            for group in self.param_groups
        ]

    def load_state_dict(self, state_dict):
        for current, saved in zip(self.param_groups, state_dict["param_groups"], strict=True):
            for key in (
                "name",
                "kind",
                "blocks",
                "recipe",
                "coefficients",
                "scaling",
                "lr_scale",
                "weight_decay",
                "eps",
                "momentum",
                "betas",
                "sinkhorn_iterations",
                "sinkhorn_threshold",
            ):
                if current[key] != saved[key]:
                    raise ValueError(f"V4.1 optimizer contract differs: {key}")
        super().load_state_dict(state_dict)
        for current, saved in zip(self.param_groups, state_dict["param_groups"], strict=True):
            p = current["params"][0]
            original = state_dict["state"].get(saved["params"][0], {})
            for key in ("momentum_buffer", "exp_avg", "exp_avg_sq"):
                if key in original:
                    self.state[p][key] = (
                        original[key].to(device=p.device, dtype=torch.float32).clone()
                    )


def make_optimizer(model, *, lr, weight_decay=0.1, eps=1e-20, vision_lr=None):
    return V41Optimizer(model, lr=lr, weight_decay=weight_decay, eps=eps, vision_lr=vision_lr)
