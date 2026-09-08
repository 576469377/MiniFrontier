"""Autograd adapters for the pinned Kimi KDA call signature.

CUDA uses FLA 0.5.2 (the upstream dependency); CPU uses the explicit recurrence.
The old transpose_state_layout flag is mapped to state_v_first. Padding is
handled by the model as right padding, never by concatenating batch sequences.
"""

from __future__ import annotations

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
        from fla.ops.kda import chunk_kda as fused

        return fused(**kwargs, state_v_first=True)
    return reference_kda(**kwargs)


def fused_recurrent_kda(**kwargs):
    kwargs.pop("transpose_state_layout", None)
    if kwargs["q"].is_cuda:
        from fla.ops.kda import fused_recurrent_kda as fused

        return fused(**kwargs, state_v_first=True)
    return reference_kda(**kwargs)
