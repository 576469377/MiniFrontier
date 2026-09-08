"""Kimi acceptance-overlap and DeepSpec DSpark distribution/confidence objectives."""

import torch
import torch.nn.functional as F


def kimi_lk(target_logits, draft_logits):
    log_p = target_logits.detach().float().log_softmax(-1)
    log_q = draft_logits.float().log_softmax(-1)
    return -torch.logsumexp(torch.minimum(log_p, log_q), -1)


def dspark_loss(target_logits, draft_logits, target_tokens, confidence_logits, valid, *, gamma=4.0):
    if gamma <= 0:
        raise ValueError("DSpark position decay must be positive")
    p = target_logits.detach().float().softmax(-1)
    q = draft_logits.float().softmax(-1)
    l1 = (p - q).abs().sum(-1)
    confidence_target = (1 - 0.5 * l1).detach().clamp(0, 1)
    ce = F.cross_entropy(
        draft_logits.float().flatten(0, 1),
        target_tokens.masked_fill(~valid, -100).flatten(),
        reduction="none",
    ).view_as(valid)
    confidence = F.binary_cross_entropy_with_logits(
        confidence_logits.float(), confidence_target, reduction="none"
    )
    weights = torch.exp(-torch.arange(valid.shape[-1], device=valid.device) / gamma) * valid
    loss = ((0.1 * ce + 0.9 * l1 + confidence) * weights).sum() / weights.sum().clamp_min(1)
    return loss, dict(
        positions=int(valid.sum()),
        normalizer=float(weights.sum()),
        overlap=confidence_target,
        l1=l1.detach(),
        ce=ce.detach(),
    )
