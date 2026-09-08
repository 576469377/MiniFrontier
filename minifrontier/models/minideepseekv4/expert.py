"""Previously source-verified unquantized V4 expert; not a complete model.

Reference: DeepSeek-V4-Flash 60d8d70 (MIT; THIRD_PARTY_NOTICES.md).
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


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


from .upstream_layers import Gate  # noqa: E402


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
            # Vision-Exp Gate at 6821d6a: hash only text; content route image types.
            scores = F.softplus(F.linear(x.float(), self.weight.float())).sqrt()
            image = self.route_image_mask
            assert image is not None
            if self.hash:
                indices = self.tid2eid[torch.where(image, 0, input_ids)]
                visual = (scores + self.bias_vl).topk(self.topk, dim=-1).indices
                indices = torch.where(image[:, None], visual.to(indices.dtype), indices)
            else:
                assert self.bias is not None
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


def _positive_int(name: str, value: int) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _positive_float(name: str, value: float) -> None:
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


class ClampedSwiGLU(nn.Module):
    """V4 report: cap gate at +10, clamp linear branch to [-10, 10]."""

    def __init__(self, hidden_size: int, intermediate_size: int, limit: float = 10.0) -> None:
        super().__init__()
        _positive_int("hidden_size", hidden_size)
        _positive_int("intermediate_size", intermediate_size)
        _positive_float("limit", limit)
        self.limit = limit
        self.gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: Tensor, routing_weight: Tensor | None = None) -> Tensor:
        gate = self.gate(x).float().clamp(max=self.limit)
        up = self.up(x).float().clamp(-self.limit, self.limit)
        value = F.silu(gate) * up
        if routing_weight is not None:
            value = value * routing_weight.float()
        return self.down(value.to(x.dtype))
