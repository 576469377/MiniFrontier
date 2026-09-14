# MF1.1-owned copy of the MF fusion implementation; retained component provenance below.
# Copyright 2025-2026 The Moonshot AI Team, DeepSeek-AI, HuggingFace Inc.
# Kimi K3 License
#
# Copyright (c) 2026 Moonshot AI
#
# Permission is hereby granted, free of charge, to any person (the "Licensee")
# obtaining a copy of this software — including the model weights, parameters,
# configuration files, inference and training code, and associated documentation
# (collectively, the "Software") — to deal in the Software without restriction.
# This includes, without limitation, the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software; to run,
# deploy, fine-tune, or otherwise modify the Software and create derivative works
# from it; and to permit persons to whom the Software is furnished to do so, in
# each case subject to the following conditions:
#
# 1. The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software. Licensee's use of the
# Software must comply with applicable laws and regulations.
#
# 2. "Model as a Service" means giving a third party access to language model
# inference or fine-tuning (e.g., via API) in a manner that allows such third
# party to exercise meaningful control over the inputs, parameters, or training
# data. This does not include (a) end-user products with model capabilities solely
# embedded within specific features or harnesses, or (b) mere relaying of requests
# to models hosted by others.
#
# If the Licensee or any of its affiliates operates a Model as a Service business,
# and the aggregate revenue of the Licensee and its affiliates exceeds 20 million
# US dollars (or the equivalent in other currencies) in total over any consecutive
# 12 months, the Licensee must enter into a separate agreement with Moonshot AI
# before using the Software or its derivative works for any commercial purpose.
#
# 3. If the Software (or any derivative works thereof) is used for any of the
# Licensee's commercial products or services that have more than 100 million
# monthly active users, or more than 20 million US dollars (or equivalent in other
# currencies) in monthly revenue, "Kimi K3" must be prominently displayed on the
# user interface of such product or service.
#
# 4. The requirements set forth in Sections 2 and 3 do not apply to: (a) internal
# use of the Software, defined as any use that does not make the Software, its
# outputs, or its underlying capabilities available to third parties; or (b) any
# use of the Software accessed through Moonshot AI's official products or
# certified inference partners.
#
# 5. THE SOFTWARE AND ANY OUTPUT AND RESULTS THEREFROM ARE PROVIDED ON AN “AS IS”
# BASIS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
# LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE
# AND NONINFRINGEMENT. IN NO EVENT SHALL MOONSHOT AI OR ITS AFFILIATES OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# For any questions regarding this license, please contact <license@moonshot.ai>.

"""MF-local KDA projections, convolution and differentiable recurrence.

Kimi K3 c5d1dd4 supplies KimiDeltaAttention.__init__; its unused forward and
other model layers are not copied. The kernel adapters match the local Kimi
training adaptation: FLA 0.5.2 on CUDA, explicit FP32 recurrence on CPU.
Parameter construction order, state layout and precision are preserved.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class ShortConvolution(nn.Conv1d):
    def __init__(self, hidden_size, kernel_size, activation="silu"):
        super().__init__(
            hidden_size,
            hidden_size,
            kernel_size,
            groups=hidden_size,
            padding=kernel_size - 1,
            bias=False,
        )
        self.activation = activation

    def forward(self, x, cache=None, output_final_state=False, cu_seqlens=None):
        if cu_seqlens is not None:
            raise ValueError("Kimi training adapter expects complete, separately batched sequences")
        if output_final_state or cache is not None:
            width = self.kernel_size[0] - 1
            prefix = x.new_zeros(x.shape[0], x.shape[2], width) if cache is None else cache
            if prefix.shape != (x.shape[0], x.shape[2], width):
                raise ValueError("convolution cache shape mismatch")
            combined = torch.cat((prefix, x.transpose(1, 2)), -1)
            value = F.conv1d(combined, self.weight, groups=self.groups).transpose(1, 2)
            return F.silu(value), combined[..., -width:].clone() if width else combined[
                ..., :0
            ].clone()
        value = F.conv1d(x.transpose(1, 2), self.weight, padding=self.padding, groups=self.groups)[
            ..., : x.shape[1]
        ].transpose(1, 2)
        return F.silu(value), None


class FusedRMSNormGated(nn.Module):
    def __init__(self, hidden_size, eps, activation="sigmoid"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x, gate):
        value = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return (value * self.weight * gate.float().sigmoid()).to(x.dtype)


def reference_kda(q, k, v, g, beta, A_log, dt_bias, lower_bound=-5.0, **kwargs):
    """K3 report eq. 5 and the delta recurrence, accumulating state in FP32."""
    dtype = v.dtype
    q, k = (F.normalize(t.float(), dim=-1, eps=1e-6) for t in (q, k))
    q = q * q.shape[-1] ** -0.5
    decay = lower_bound * torch.sigmoid(
        A_log.float().exp()[:, None] * (g.float() + dt_bias.float().view(g.shape[-2:]))
    )
    beta = beta.float().sigmoid()
    initial = kwargs.get("initial_state")
    state = (
        q.new_zeros(q.shape[0], q.shape[2], q.shape[3], v.shape[-1])
        if initial is None
        else initial.float().clone()
    )
    outputs = []
    for t in range(q.shape[1]):
        state = state * decay[:, t].exp().unsqueeze(-1)
        residual = v[:, t].float() - (state * k[:, t, :, :, None]).sum(-2)
        state = state + k[:, t, :, :, None] * (beta[:, t, :, None] * residual).unsqueeze(-2)
        outputs.append((state * q[:, t, :, :, None]).sum(-2))
    return torch.stack(outputs, dim=1).to(dtype), state


def chunk_kda(**kwargs):
    kwargs.pop("transpose_state_layout", None)
    if kwargs["q"].is_cuda:
        from fla.ops.kda import chunk_kda as fused  # type: ignore[import-untyped]

        return fused(**kwargs, state_v_first=True)
    return reference_kda(**kwargs)


def fused_recurrent_kda(**kwargs):
    kwargs.pop("transpose_state_layout", None)
    if kwargs["q"].is_cuda:
        from fla.ops.kda import fused_recurrent_kda as fused  # type: ignore[import-untyped]

        return fused(**kwargs, state_v_first=True)
    return reference_kda(**kwargs)


class KDAProjections(nn.Module):
    """Pinned parameter construction used by the MF boundary-aware adapter."""

    def __init__(self, config: Any, layer_idx: int):
        super().__init__()
        self.config = config
        self.mode = "chunk"

        self.hidden_size = config.hidden_size
        self.conv_size = config.linear_attn_config["short_conv_kernel_size"]
        self.head_dim = config.linear_attn_config["head_dim"]
        self.num_heads = config.linear_attn_config["num_heads"]
        self.head_k_dim = self.head_dim
        self.num_k_heads = self.num_heads

        self.layer_idx = layer_idx

        assert self.mode in ["chunk", "fused_recurrent"], f"Not supported mode `{self.mode}`."

        projection_k_size = self.head_k_dim * self.num_k_heads
        projection_size = self.head_dim * self.num_heads

        self.q_proj = nn.Linear(self.hidden_size, projection_k_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, projection_k_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, projection_size, bias=False)

        self.q_conv1d = ShortConvolution(
            hidden_size=projection_k_size,
            kernel_size=self.conv_size,
            activation="silu",
        )
        self.k_conv1d = ShortConvolution(
            hidden_size=projection_k_size,
            kernel_size=self.conv_size,
            activation="silu",
        )
        self.v_conv1d = ShortConvolution(
            hidden_size=projection_size,
            kernel_size=self.conv_size,
            activation="silu",
        )

        self.A_log = torch.nn.Parameter(
            torch.log(torch.empty(self.num_heads, dtype=torch.float32).uniform_(1, 16))
        )

        self.f_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)

        self.dt_bias = nn.Parameter(torch.empty(projection_size, dtype=torch.float32))

        self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)

        self.use_full_rank_gate = config.linear_attn_config.get("use_full_rank_gate", False)
        self.gate_lower_bound = config.linear_attn_config.get("gate_lower_bound", None)
        if self.use_full_rank_gate:
            self.g_proj = nn.Linear(self.hidden_size, projection_size, bias=False)
        else:
            self.g_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
            self.g_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)

        self.o_norm = FusedRMSNormGated(
            self.head_dim, eps=config.rms_norm_eps, activation="sigmoid"
        )
        self.o_proj = nn.Linear(projection_size, self.hidden_size, bias=False)
