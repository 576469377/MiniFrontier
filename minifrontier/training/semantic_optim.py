"""Small shared optimizer engine; each model supplies its own mathematical blocks.

Grouping, Newton-Schulz coefficients, scaling and learning rates are explicit.
Equal-shaped independent matrices are orthogonalized in bounded batches. This
engine deliberately does not alter the existing Qwen optimizer's conventions.
"""

from __future__ import annotations

import math
from collections import defaultdict

import torch


def newton_schulz(matrix, coefficients, eps=1e-14):
    if matrix.ndim < 2:
        raise ValueError("orthogonalization requires independent matrices")
    x = matrix.float()
    transpose = x.shape[-2] > x.shape[-1]
    if transpose:
        x = x.mT
    x = x / x.norm(dim=(-2, -1), keepdim=True).clamp_min(eps)
    # FP32 oracle/production first; TF32/BF16 kernel changes require their own tolerance audit.
    with torch.autocast(device_type=x.device.type, enabled=False):
        for a, b, c in coefficients:
            gram = x @ x.mT
            x = a * x + (b * gram + c * (gram @ gram)) @ x
    return x.mT if transpose else x


class SemanticOptimizer(torch.optim.Optimizer):
    """Blocks are row slices of 2-D independent matrices; others explicitly AdamW."""

    def __init__(
        self,
        model,
        specs,
        *,
        lr,
        adam_lr,
        coefficients,
        scaling,
        recipe,
        weight_decay=0.1,
        eps=1e-8,
        momentum=0.95,
    ):
        if not all(math.isfinite(v) and v > 0 for v in (lr, adam_lr, eps, scaling)):
            raise ValueError("invalid optimizer numerical settings")
        self.coefficients = tuple(coefficients)
        self.scaling = scaling
        self.recipe = recipe
        groups = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if id(parameter) not in specs:
                raise ValueError(f"unclassified {recipe} parameter: {name}")
            blocks, reason = specs[id(parameter)]
            if blocks is not None:
                if parameter.ndim != 2:
                    raise ValueError(f"Muon block must be a 2-D matrix: {name}")
                coverage = torch.zeros(parameter.shape[0], dtype=torch.int)
                for start, end in blocks:
                    coverage[start:end] += 1
                if not (coverage == 1).all():
                    raise ValueError(f"overlapping or missing semantic rows: {name}")
            groups.append(
                dict(
                    params=[parameter],
                    name=name,
                    blocks=blocks,
                    reason=reason,
                    recipe=recipe,
                    coefficients=self.coefficients,
                    scaling=scaling,
                    weight_decay=weight_decay if parameter.ndim >= 2 else 0.0,
                )
            )
        super().__init__(
            groups, dict(lr=lr, adam_lr=adam_lr, eps=eps, momentum=momentum, betas=(0.9, 0.95))
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            p = group["params"][0]
            if p.grad is not None and (p.grad.is_sparse or not torch.isfinite(p.grad).all()):
                raise FloatingPointError("optimizer requires finite dense gradients")
        buckets = defaultdict(list)
        for group in self.param_groups:
            p = group["params"][0]
            if p.grad is None:
                continue
            state = self.state[p]
            grad = p.grad.float()
            state["step"] = state.get("step", 0) + 1
            if group["blocks"] is None:
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                b1, b2 = group["betas"]
                state["exp_avg"].lerp_(grad, 1 - b1)
                state["exp_avg_sq"].mul_(b2).addcmul_(grad, grad, value=1 - b2)
                mean = state["exp_avg"] / (1 - b1 ** state["step"])
                var = state["exp_avg_sq"] / (1 - b2 ** state["step"])
                update = mean / (var.sqrt() + group["eps"])
                p.mul_(1 - group["adam_lr"] * group["weight_decay"])
                p.add_(update.to(p.dtype), alpha=-group["adam_lr"])
            else:
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.float32)
                state["momentum_buffer"].mul_(group["momentum"]).add_(grad)
                direction = grad + group["momentum"] * state["momentum_buffer"]
                for start, end in group["blocks"]:
                    part = direction[start:end]
                    buckets[(part.shape, part.device)].append((p[start:end], part, group))
        for rows in buckets.values():
            for offset in range(0, len(rows), 32):
                chunk = rows[offset : offset + 32]
                direction = torch.stack([item[1] for item in chunk])
                update = newton_schulz(direction, self.coefficients)
                update *= self.scaling * math.sqrt(max(update.shape[-2:]))
                for (p, _, group), u in zip(chunk, update, strict=True):
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(u.to(p.dtype), alpha=-group["lr"])
        return loss

    def load_state_dict(self, state_dict):
        for current, saved in zip(self.param_groups, state_dict["param_groups"], strict=True):
            for key in ("name", "blocks", "recipe", "coefficients", "scaling"):
                if current[key] != saved[key]:
                    raise ValueError(f"optimizer semantic contract differs: {key}")
        super().load_state_dict(state_dict)
        for current, saved in zip(self.param_groups, state_dict["param_groups"], strict=True):
            p = current["params"][0]
            original = state_dict["state"].get(saved["params"][0], {})
            for key in ("exp_avg", "exp_avg_sq", "momentum_buffer"):
                if key in original:
                    self.state[p][key] = (
                        original[key].to(device=p.device, dtype=torch.float32).clone()
                    )


def whole(parameter):
    return ((0, parameter.shape[0]),)


def head_rows(parameter, widths):
    result, offset = [], 0
    for width in widths:
        result.append((offset, offset + width))
        offset += width
    if offset != parameter.shape[0]:
        raise ValueError("head semantics do not cover matrix")
    return tuple(result)
