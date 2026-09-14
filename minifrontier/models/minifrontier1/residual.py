# GR derives from Transformers 4177486 (Apache-2.0); see THIRD_PARTY_NOTICES.md.
"""Single owner for the pinned GR read and residual injection equations."""

import torch
from torch import nn

from minifrontier.models.deepseek_v41_layers import SinglePassHC, single_pass_coefficients
from minifrontier.models.miniqwen4.upstream_core import Qwen4ExpTextGatedResidual


class GatedResidual(nn.Module):
    def __init__(self, config, *, read_only=False):
        super().__init__()
        self.primitive = Qwen4ExpTextGatedResidual(config, use_combine=not read_only)
        self.read_only = read_only

    def read(self, residual):
        result = self.primitive(residual.flatten(-2))
        return (result, None) if self.read_only else (result[0], result[2])

    def inject(self, residual, update, weights):
        if self.read_only or weights is None:
            raise ValueError("final GR is read-only")
        return residual + weights.unsqueeze(-1) * update.unsqueeze(-2)

    def forward(self, residual):
        return self.read(residual)[0]


class SinglePassMHC(nn.Module):
    """V4.1 Eq. (6): this sublayer predicts the next sublayer's input mixer.

    Equations follow DeepSeek-V4.1 dba1be0 (MIT), inference/model.py Block.
    Coefficients, mixing and injection share the V4.1 mathematical implementation.
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
        return SinglePassHC.mix(residual, pre_mix)

    @staticmethod
    def inject(residual, update, post, combination):
        return SinglePassHC.combine(update, residual, post, combination)
