# Copyright 2026 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0; see THIRD_PARTY_NOTICES.md.
# mypy: disable-error-code="return-value"
"""MF-local grouped RMSNorm and gated residual from Transformers 4177486.

Source: 4177486a9f199bd7be520eff14431071d5d41ec5.
Only the GR computation and its normalization dependency are copied; parameter
names, registration order, equations and FP32 normalization match the source.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

Qwen4ExpTextConfig = Any


class Qwen4ExpTextRMSNorm(nn.Module):
    def __init__(self, dim: int, group_size: int | None = None, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))
        self.group_size = group_size
        if group_size is not None and dim % group_size != 0:
            raise ValueError(f"hidden_size ({dim}) must be divisible by group_size ({group_size}).")

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        if self.group_size is not None:
            x = x.reshape(*x.shape[:-1], -1, self.group_size)
        out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return out.flatten(-2) if self.group_size is not None else out

    def forward(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Qwen4ExpText is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class Qwen4ExpTextGatedResidual(nn.Module):
    def __init__(self, config: Qwen4ExpTextConfig, use_combine: bool = True):
        super().__init__()
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        hc_hidden_size = self.hc_count * self.hidden_size
        self.hc_norm = Qwen4ExpTextRMSNorm(
            hc_hidden_size, group_size=self.hidden_size, eps=config.rms_norm_eps
        )
        self.input_mix_weight_down = nn.Linear(hc_hidden_size, config.hc_lowrank, bias=False)
        self.input_mix_weight_up = nn.Linear(config.hc_lowrank, hc_hidden_size, bias=False)
        self.block_inject_weight = (
            nn.Linear(hc_hidden_size, self.hc_count, bias=False) if use_combine else None
        )

    def forward(
        self, hyper_input: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if hyper_input.shape[-1] != self.hc_count * self.hidden_size:
            raise ValueError(
                f"Expected {self.hc_count * self.hidden_size} hyper-connection features, got {hyper_input.shape[-1]}."
            )
        hyper_input_normed = self.hc_norm(hyper_input)
        input_mix_weight = F.silu(self.input_mix_weight_down(hyper_input_normed) / self.hc_count)
        input_mix_weight = torch.sigmoid(self.input_mix_weight_up(input_mix_weight))
        input_mix_weight = input_mix_weight.unflatten(-1, (self.hc_count, self.hidden_size))
        mixed_input = (
            input_mix_weight * hyper_input_normed.unflatten(-1, (self.hc_count, self.hidden_size))
        ).mean(dim=-2)
        if self.block_inject_weight is None:
            return mixed_input
        injection_weights = 2 * torch.sigmoid(
            self.block_inject_weight(hyper_input_normed) / self.hc_count
        )
        return mixed_input, hyper_input, injection_weights
