"""MX format simulation with FP32 masters and straight-through derivatives.

E2M1/E4M3, 32-element blocks and E8M0 scales follow OCP MX v1.0. The
explicit ceil-amax scale policy follows DeepSeek 60d8d70/kernel.py; the OCP
format specification does not prescribe a unique scale-selection algorithm.
This is quantize/dequantize simulation, not a 3090 FP4 arithmetic kernel.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def fp4_codes(values):
    """E2M1 round-to-nearest, ties-to-even, saturating finite conversion."""
    magnitude = values.abs().clamp_max(6)
    boundaries = values.new_tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    code = torch.bucketize(magnitude.contiguous(), boundaries)
    tie = magnitude == boundaries[code.clamp_max(6)]
    code = code + (tie & (code % 2 == 1)).long()
    return code.to(torch.uint8) | (values.signbit().to(torch.uint8) << 3)


def fp4_values(codes):
    levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=codes.device)
    return levels[(codes & 7).long()] * torch.where(codes & 8 != 0, -1, 1)


def quantize_mx(x, *, bits=4, axis=-1):
    if bits not in {4, 8} or not torch.isfinite(x).all():
        raise ValueError("MX simulation accepts finite FP4/FP8 inputs")
    values = x.detach().float().movedim(axis, -1)
    if values.shape[-1] % 32:
        raise ValueError("MX group dimension must be divisible by 32")
    shape = values.shape
    blocks = values.reshape(*shape[:-1], shape[-1] // 32, 32)
    maximum = 6.0 if bits == 4 else 448.0
    # E8M0 exponent byte 255 is NaN. The source kernel uses minimum 2^-126.
    exponent = torch.ceil(torch.log2(blocks.abs().amax(-1).clamp_min(maximum * 2**-126) / maximum))
    exponent = exponent.clamp(-126, 127)
    scales = torch.exp2(exponent)
    normalized = (blocks / scales[..., None]).clamp(-maximum, maximum)
    if bits == 4:
        codes = fp4_codes(normalized)
        decoded = fp4_values(codes)
    else:
        converted = normalized.to(torch.float8_e4m3fn)
        codes, decoded = converted.view(torch.uint8), converted.float()
    dequantized = (decoded * scales[..., None]).reshape(shape).movedim(-1, axis).to(x.dtype)
    return dequantized, codes, (exponent + 127).to(torch.uint8)


def fake_mx(x, *, bits=4, axis=-1):
    dequantized, _, _ = quantize_mx(x, bits=bits, axis=axis)
    return x + (dequantized - x).detach()


def hadamard(x):
    width = x.shape[-1]
    if width < 1 or width & (width - 1):
        raise ValueError("normalized Hadamard needs a power-of-two head width")
    result = x.float()
    stride = 1
    while stride < width:
        blocks = result.reshape(*x.shape[:-1], -1, 2, stride)
        a, b = blocks.unbind(-2)
        result = torch.stack((a + b, a - b), -2).reshape(x.shape)
        stride *= 2
    return (result / math.sqrt(width)).to(x.dtype)


class MXLinear(nn.Linear):
    @classmethod
    def from_linear(cls, original, *, activation_bits=None):
        result = cls.__new__(cls)
        nn.Module.__init__(result)
        result.in_features, result.out_features = original.in_features, original.out_features
        result.weight, result.bias = original.weight, original.bias
        result.activation_bits = activation_bits
        result.train(original.training)
        return result

    def forward(self, x):
        if self.activation_bits is not None:
            x = fake_mx(x, bits=self.activation_bits)
        return F.linear(x, fake_mx(self.weight), self.bias)


def configure_experts(model, *, activation_bits=None):
    """Only routed expert w1/w2/w3; retain parameter identities and state keys."""
    import re

    replaced = []
    for name, module in list(model.named_modules()):
        if re.search(r"(?:^|\.)experts\.\d+\.(w1|w2|w3)$", name):
            if not isinstance(module, nn.Linear) or module.weight.shape[-1] % 32:
                raise ValueError(f"invalid native routed expert MX group layout: {name}")
            parent, key = name.rsplit(".", 1)
            setattr(
                model.get_submodule(parent),
                key,
                MXLinear.from_linear(module, activation_bits=activation_bits),
            )
            replaced.append(name)
    if not replaced:
        raise ValueError("QAT selected no native routed experts")
    return replaced
