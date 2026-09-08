"""Kimi routed experts only; calibration is required before formal posttraining."""

from .mx_quant import configure_experts


def configure(model):
    return dict(
        experts=configure_experts(model, activation_bits=8),
        weight_format="MXFP4-E2M1",
        activation_format="MXFP8-E4M3",
        block_size=32,
        scale="E8M0-ceil-amax",
        master="FP32",
        compute="BF16",
        rounding="nearest-ties-even",
        gradient="STE",
        scale_policy_status="local recipe; requires Kimi 5M SFT calibration",
        native_low_precision_acceleration=False,
    )
