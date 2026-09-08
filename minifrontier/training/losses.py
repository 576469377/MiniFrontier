"""Model-independent language modeling losses."""

from torch import Tensor
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def causal_lm_loss(logits: Tensor, labels: Tensor, ignore_index: int = -100) -> Tensor:
    if logits.shape[:2] != labels.shape:
        raise ValueError(
            f"logits and labels disagree: logits={tuple(logits.shape)}, labels={tuple(labels.shape)}"
        )
    shifted_logits = logits[:, :-1].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    flat_labels = shifted_labels.view(-1)
    loss_sum = F.cross_entropy(
        shifted_logits.float().view(-1, logits.size(-1)),
        flat_labels,
        ignore_index=ignore_index,
        reduction="sum",
    )
    valid_targets = flat_labels.ne(ignore_index).sum().clamp_min(1)
    return loss_sum / valid_targets


def chunked_linear_ce(hidden, weight, labels, *, chunk_size=128, shift=True):
    """Full-vocabulary CE with position chunks and head recomputation in backward."""
    if chunk_size < 1 or hidden.shape[:2] != labels.shape:
        raise ValueError("invalid chunked head/label shape")
    features = hidden[:, :-1] if shift else hidden
    targets = labels[:, 1:] if shift else labels
    features, targets = features.reshape(-1, hidden.shape[-1]), targets.reshape(-1)
    valid = targets.ne(-100)
    features, targets = features[valid], targets[valid]
    total = hidden.sum() * 0 + weight.reshape(-1)[:1].sum() * 0

    def term(x, w, y):
        return F.cross_entropy(F.linear(x, w).float(), y, reduction="sum")

    for start in range(0, len(targets), chunk_size):
        total = total + checkpoint(
            term,
            features[start : start + chunk_size],
            weight,
            targets[start : start + chunk_size],
            use_reentrant=False,
        )
    return total / max(1, len(targets))
