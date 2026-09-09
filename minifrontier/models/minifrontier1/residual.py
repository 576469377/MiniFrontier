# GR derives from Transformers 4177486 (Apache-2.0); see THIRD_PARTY_NOTICES.md.
"""Single owner for the pinned GR read and residual injection equations."""

from torch import nn

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
