"""Qwen report §3.1 optimizer adapted to the official fused parameter layout.

Eight Polar Express iterations, per-head Q/K/V splits, gate/up expert splits,
AdamW output gates and GR/router/head/embeddings, zero decay on n-gram tables.
This is replicated single-device/DDP optimization, not Canzona/ZeRO/TP.
LR, Adam settings and weight decay remain explicit local experiment choices.
"""

from __future__ import annotations

import math
import weakref
from typing import Any, cast

import torch

from minifrontier.models.miniqwen4 import MiniQwen4ForCausalLM
from minifrontier.models.miniqwen4.upstream_decoder import Qwen4ExpTextDecoderLayer
from minifrontier.training.polar_express import polar_express_orthogonalize

# (expert index or -1 for a 2D tensor, row start, row end, algorithm)
Block = tuple[int, int, int, str]


def source_muon_blocks(model: MiniQwen4ForCausalLM) -> dict[int, tuple[Block, ...]]:
    specs: dict[int, tuple[Block, ...]] = {}

    def whole(linear):
        specs[id(linear.weight)] = ((-1, 0, linear.weight.shape[0], "muon"),)

    def heads(linear, widths):
        offset, blocks = 0, []
        for width in widths:
            blocks.append((-1, offset, offset + width, "muon"))
            offset += width
        if offset != linear.weight.shape[0]:
            raise ValueError("semantic head split does not cover projection")
        specs[id(linear.weight)] = tuple(blocks)

    for raw_layer in [*model.model.layers, *([model.mtp.block] if model.mtp is not None else [])]:
        layer = cast(Qwen4ExpTextDecoderLayer, raw_layer)
        if hasattr(layer, "linear_attn"):
            gdn = layer.linear_attn
            heads(
                gdn.in_proj_qkv,
                [gdn.head_k_dim] * (2 * gdn.num_k_heads) + [gdn.head_v_dim] * gdn.num_v_heads,
            )
            whole(gdn.out_proj)
        else:
            attn = layer.self_attn
            dim = attn.head_dim
            blocks = []
            for offset in range(0, attn.q_proj.weight.shape[0], 2 * dim):
                blocks.extend(
                    [
                        (-1, offset, offset + dim, "muon"),
                        (-1, offset + dim, offset + 2 * dim, "adamw"),
                    ]
                )
            specs[id(attn.q_proj.weight)] = tuple(blocks)
            heads(attn.k_proj, [dim] * (attn.k_proj.weight.shape[0] // dim))
            heads(attn.v_proj, [dim] * (attn.v_proj.weight.shape[0] // dim))
            whole(attn.o_proj)
        experts = layer.mlp.experts
        width = experts.intermediate_dim
        specs[id(experts.gate_up_proj)] = tuple(
            (e, start, start + width, "muon")
            for e in range(experts.num_experts)
            for start in (0, width)
        )
        specs[id(experts.down_proj)] = tuple(
            (e, 0, experts.hidden_dim, "muon") for e in range(experts.num_experts)
        )
        for name in ("gate_proj", "up_proj", "down_proj"):
            whole(getattr(layer.mlp.shared_expert, name))
        if layer.ple is not None:
            whole(layer.ple.key_proj)
            whole(layer.ple.value_proj)
    if model.mtp is not None:
        whole(model.mtp.fc_hidden)
        whole(model.mtp.fc_embedding)
    return specs


class MiniQwen4Optimizer(torch.optim.Optimizer):
    """Independent semantic blocks, including mixed Muon/Adam rows in fused Q/gate."""

    def __init__(
        self,
        model: MiniQwen4ForCausalLM,
        *,
        lr: float,
        adam_lr: float,
        weight_decay: float = 0.1,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
    ) -> None:
        for name, value in (("lr", lr), ("adam_lr", adam_lr), ("eps", eps)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not math.isfinite(weight_decay)
            or weight_decay < 0
            or len(betas) != 2
            or not all(math.isfinite(b) and 0 <= b < 1 for b in betas)
        ):
            raise ValueError("invalid decay or Adam betas")
        specs = source_muon_blocks(model)
        self._model = weakref.ref(model)
        self._phase = model.training_phase
        groups = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            blocks = specs.get(id(p))
            if blocks is not None:
                coverage = torch.zeros(p.shape[:-1], dtype=torch.int32)
                for expert, start, end, algorithm in blocks:
                    if algorithm not in {"muon", "adamw"}:
                        raise ValueError("invalid optimizer block algorithm")
                    if p.ndim == 3:
                        coverage[expert, start:end] += 1
                    else:
                        coverage[start:end] += 1
                if not bool((coverage == 1).all()):
                    raise ValueError(f"optimizer blocks must cover {name} exactly once")
            decay = 0.0 if p.ndim < 2 or "ngram_embedding.weight" in name else weight_decay
            groups.append(
                dict(params=[p], name=name, blocks=blocks, weight_decay=decay, phase=self._phase)
            )
        super().__init__(groups, dict(lr=lr, adam_lr=adam_lr, betas=betas, eps=eps, momentum=0.95))

    @torch.no_grad()
    def step(self, closure=None):
        model = self._model()
        if model is None or model.training_phase != self._phase:
            raise ValueError("rebuild the optimizer after a training phase transition")
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        # Reject nonfinite gradients before modifying any model/optimizer state.
        for group in self.param_groups:
            p = group["params"][0]
            if p.grad is not None and (p.grad.is_sparse or not bool(torch.isfinite(p.grad).all())):
                raise ValueError("source optimizer requires finite dense gradients")
        for group in self.param_groups:
            p = group["params"][0]
            if p.grad is None:
                continue
            gradient = p.grad.float()
            blocks = group["blocks"]
            algorithms = {b[3] for b in blocks} if blocks else {"adamw"}
            state = self.state[p]
            state["step"] = state.get("step", 0) + 1
            if "muon" in algorithms:
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.float32)
                momentum = state["momentum_buffer"]
                momentum.lerp_(gradient, 1 - group["momentum"])
                direction = gradient.lerp(momentum, group["momentum"])
            if "adamw" in algorithms:
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                b1, b2 = group["betas"]
                state["exp_avg"].lerp_(gradient, 1 - b1)
                state["exp_avg_sq"].mul_(b2).addcmul_(gradient, gradient, value=1 - b2)
                adam = state["exp_avg"] / (1 - b1 ** state["step"])
                adam = adam / (
                    (state["exp_avg_sq"] / (1 - b2 ** state["step"])).sqrt() + group["eps"]
                )
            if blocks is None:
                p.mul_(1 - group["adam_lr"] * group["weight_decay"])
                p.add_(adam.to(p.dtype), alpha=-group["adam_lr"])
                continue
            for expert, start, end, algorithm in blocks:
                index = (expert, slice(start, end)) if p.ndim == 3 else (slice(start, end),)
                part = p[index]
                if algorithm == "muon":
                    update = polar_express_orthogonalize(direction[index], steps=8, eps=1e-14)
                    update = update * (0.2 * math.sqrt(max(update.shape)))
                    rate = group["lr"]
                else:
                    update, rate = adam[index], group["adam_lr"]
                part.mul_(1 - rate * group["weight_decay"])
                part.add_(update.to(part.dtype), alpha=-rate)
        return loss

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        saved_groups = state_dict["param_groups"]
        if len(saved_groups) != len(self.param_groups):
            raise ValueError("source optimizer parameter topology changed")
        for current, saved in zip(self.param_groups, saved_groups, strict=True):
            if current["name"] != saved.get("name") or current["blocks"] != saved.get("blocks"):
                raise ValueError("source optimizer semantic parameter layout changed")
            if current["phase"] != saved.get("phase"):
                raise ValueError("optimizer checkpoint training phase differs")
        super().load_state_dict(state_dict)
        # Optimizer.load_state_dict normally casts state to parameter dtype.
        # Recover FP32 state from the original payload, even for BF16 parameters.
        for group, saved in zip(self.param_groups, saved_groups, strict=True):
            p = group["params"][0]
            original = state_dict["state"].get(saved["params"][0], {})
            for key in ("momentum_buffer", "exp_avg", "exp_avg_sq"):
                if key in original:
                    self.state[p][key] = (
                        original[key].to(device=p.device, dtype=torch.float32).clone()
                    )
