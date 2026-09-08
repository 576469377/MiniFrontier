"""Differentiable FP32 counterpart of pinned kernel.py's mHC Sinkhorn.

DeepSeek-V4-Flash 60d8d70, MIT (see upstream_layers.py for license).
No quantization simulation: this adapter trains floating point master weights.
"""

import torch


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    hc = hc_mult
    mixes = mixes.float()
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
    comb = (mixes[..., 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]).unflatten(-1, (hc, hc))
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def rotary(x, freqs, inverse=False):
    pair = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs = freqs.view(1, freqs.shape[0], *([1] * (x.ndim - 3)), -1)
    return torch.view_as_real(pair * (freqs.conj() if inverse else freqs)).flatten(-2).to(x.dtype)
