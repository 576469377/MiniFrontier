"""MF1.1-owned single-pass mHC, copied from DeepSeek-V4.1 dba1be0 equations."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# MIT License
#
# Copyright (c) 2023 DeepSeek
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# Computation below derives from inference/model.py Block, dba1be0.


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


class SinglePassMHC(nn.Module):
    """V4.1 Eq. (6): this sublayer predicts the next sublayer's input mixer.

    Equations follow DeepSeek-V4.1 dba1be0 (MIT), inference/model.py Block.
    Coefficients, mixing and injection are copied locally from the V4.1 equations.
    This is the multi-kernel training implementation, not the Mega-mHC serving kernel.
    Initialization is local: small dynamic scales and an identity-biased residual map.
    """

    def __init__(self, config):
        super().__init__()
        self.streams = config.hc_count
        self.norm_eps = config.rms_norm_eps
        self.sinkhorn_iters = 20
        self.eps = 1e-6
        count = (2 + self.streams) * self.streams
        self.fn = nn.Parameter(torch.empty(count, self.streams * config.hidden_size))
        self.base = nn.Parameter(torch.zeros(count))
        self.scale = nn.Parameter(torch.full((3,), 0.01))
        nn.init.normal_(self.fn, std=config.initializer_range)
        with torch.no_grad():
            self.base[2 * self.streams :].view(self.streams, self.streams).copy_(
                torch.eye(self.streams) * 4
            )

    def coefficients(self, residual):
        return single_pass_coefficients(
            residual,
            self.fn,
            self.scale,
            self.base,
            mult=self.streams,
            norm_eps=self.norm_eps,
            eps=self.eps,
            iters=self.sinkhorn_iters,
        )

    @staticmethod
    def identity(residual):
        pre_mix = residual.new_zeros(residual.shape[:-1], dtype=torch.float32)
        pre_mix[..., 0] = 1
        return pre_mix

    @staticmethod
    def mix(residual, pre_mix):
        return (residual.float() * pre_mix[..., None]).sum(-2).to(residual.dtype)

    @staticmethod
    def inject(residual, update, post, combination):
        # The residual reduction stays FP32 under autocast, matching the source.
        with torch.autocast(residual.device.type, enabled=False):
            carried = torch.einsum("...ij,...id->...jd", combination.float(), residual.float())
        return (carried + post[..., None] * update.float().unsqueeze(-2)).to(update.dtype)
