"""Explicit FP32 routing for the new recipe; preserves the source sigmoid/top-k math."""

import torch

from .upstream_layers import KimiMoEGate


class FP32KimiGate(KimiMoEGate):
    force_fp32 = True

    def forward(self, hidden_states):
        # The source explicitly casts to float32, but enclosing AMP would still
        # downcast F.linear. The new recipe makes that precision boundary real.
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            return super().forward(hidden_states.float())
