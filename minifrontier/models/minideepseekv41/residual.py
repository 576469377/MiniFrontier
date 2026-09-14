"""Single-Pass mHC from DeepSeek-V4.1 dba1be0 (MIT).

Reference: inference/model.py (Block), technical report Eq. 6.
This model owns its residual implementation; MF1.1 maintains a separate copy.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def single_pass_coefficients(
    stream: Tensor,
    projection: Tensor,
    scale: Tensor,
    base: Tensor,
    *,
    mult: int,
    norm_eps: float,
    eps: float,
    iters: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """FP32 coefficient projection and the official Sinkhorn split."""
    x = stream.flatten(-2).float()
    with torch.autocast(x.device.type, enabled=False):
        mixed = F.linear(x, projection.float()) * torch.rsqrt(
            x.square().mean(-1, keepdim=True) + norm_eps
        )
    scale, base = scale.float(), base.float()
    pre = (mixed[..., :mult] * scale[0] + base[:mult]).sigmoid() + eps
    post = 2 * (mixed[..., mult : 2 * mult] * scale[1] + base[mult : 2 * mult]).sigmoid()
    comb = (mixed[..., 2 * mult :] * scale[2] + base[2 * mult :]).unflatten(-1, (mult, mult))
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


class SinglePassHC(nn.Module):
    """Report Eq. 6: the incoming mixing coefficients belong to the prior sublayer."""

    def __init__(self, config: Any):
        super().__init__()
        self.mult = config.hc_mult
        self.eps, self.norm_eps = config.hc_eps, config.norm_eps
        self.iters = config.hc_sinkhorn_iters
        size = self.mult * (self.mult + 2)
        self.projection = nn.Parameter(torch.empty(size, self.mult * config.dim))
        self.base = nn.Parameter(torch.zeros(size))
        self.scale = nn.Parameter(torch.full((3,), 0.01))
        nn.init.normal_(self.projection, std=config.initializer_range)
        with torch.no_grad():
            self.base[2 * self.mult :].view(self.mult, self.mult).copy_(4 * torch.eye(self.mult))

    def coefficients(self, stream: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return single_pass_coefficients(
            stream,
            self.projection,
            self.scale,
            self.base,
            mult=self.mult,
            norm_eps=self.norm_eps,
            eps=self.eps,
            iters=self.iters,
        )

    @staticmethod
    def mix(stream: Tensor, incoming_pre: Tensor) -> Tensor:
        return (stream.float() * incoming_pre[..., None]).sum(-2).to(stream.dtype)

    @staticmethod
    def combine(output: Tensor, stream: Tensor, post: Tensor, comb: Tensor) -> Tensor:
        # Upstream comb indexes [source stream, destination stream].
        # Contract without materializing [B,T,hc,hc,dim] (512 MiB at 16x1024x4x512).
        # Autocast must not turn this FP32 residual reduction into BF16. GEMM
        # reduction order may differ from the reference multiply/sum by roundoff.
        with torch.autocast(stream.device.type, enabled=False):
            residual = torch.einsum("...ij,...id->...jd", comb.float(), stream.float())
        return (residual + post[..., None] * output.float().unsqueeze(-2)).to(output.dtype)
