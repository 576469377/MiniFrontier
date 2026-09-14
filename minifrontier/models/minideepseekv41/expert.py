# SPDX-License-Identifier: MIT
"""Local routing correction and sequence balance for the V4.1 training adapter.

Routing derives from DeepSeek V4/V4.1 (MIT); sequence loss is the existing
mini-training adaptation. Bias affects selection, not selected expert weights.
"""

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

import torch
import torch.nn.functional as F
from torch import nn

from .layers import Gate


def sequence_balance_loss(scores, valid_mask, top_k):
    """V3 Eq. 17-20 with V4 unbiased sqrtsoftplus scores; mean over real samples.

    Return the unweighted mean. Actual bias-adjusted dispatch statistics belong
    to the separate sign-bias controller. Hash layers are excluded by caller.
    """
    if scores.ndim != 3 or scores.shape[:2] != valid_mask.shape:
        raise ValueError("sequence balancing needs scores [B,T,E] and valid mask [B,T]")
    n = scores.shape[-1]
    if not 0 < top_k <= n:
        raise ValueError("invalid sequence balancing top-k")
    valid = valid_mask.bool()
    lengths = valid.sum(1).clamp_min(1)
    probabilities = scores.float() / scores.float().sum(-1, keepdim=True).clamp_min(1e-20)
    selected = scores.detach().topk(top_k, dim=-1).indices
    frequency = F.one_hot(selected, n).sum(-2) * valid[..., None]
    frequency = frequency.sum(1) * (n / top_k) / lengths[:, None]
    mean_probability = (probabilities * valid[..., None]).sum(1) / lengths[:, None]
    active = valid.any(1)
    return ((frequency * mean_probability).sum(-1) * active).sum() / active.sum().clamp_min(1)


class TrainingGate(Gate):
    """Original routing plus an explicitly normalized, differentiable sequence term."""

    valid_mask = None
    sequence_balance_enabled = False

    def __init__(self, layer_id, args):
        super().__init__(layer_id, args)
        self.vocab_size = args.vocab_size
        self.bias_vl = (
            nn.Parameter(torch.zeros(args.n_routed_experts), requires_grad=False)
            if getattr(args, "vision_config", None)
            else None
        )
        self.route_image_mask = None

    def forward(self, x, input_ids=None):
        self.route_image_mask = input_ids >= self.vocab_size if self.bias_vl is not None else None
        if self.bias_vl is None:
            result = super().forward(x, input_ids)
        else:
            scores = F.softplus(F.linear(x.float(), self.weight.float())).sqrt()
            image = self.route_image_mask
            assert image is not None
            correction = torch.where(image[:, None], self.bias_vl, self.bias)
            indices = (scores + correction).topk(self.topk, dim=-1).indices
            weights = scores.gather(1, indices.long())
            weights = weights / weights.sum(-1, keepdim=True) * self.route_scale
            result = weights, indices

        self.sequence_loss = x.new_zeros(())
        if self.sequence_balance_enabled and not self.hash:
            if self.valid_mask is None:
                raise ValueError("sequence balancing needs the real-token mask")
            scores = F.softplus(F.linear(x.float(), self.weight.float())).sqrt()
            self.sequence_loss = sequence_balance_loss(
                scores.view(*self.valid_mask.shape, -1), self.valid_mask, self.topk
            )
        return result
