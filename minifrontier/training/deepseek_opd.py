"""Exact full-vocabulary reverse KL on fixed student-generated states.

Chunk only the sequence positions: every chunk retains the full normalization
over the legal vocabulary. Teacher features and head are frozen. Checkpointing
recomputes head logits during backward instead of retaining all vocab activations.
This is a differentiable KL objective, distinct from Kimi's sampled token reward.
"""

import torch
from torch.utils.checkpoint import checkpoint

from .distributions import action_logits


class SharedHead(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        if isinstance(weight, torch.nn.Parameter):
            self.weight = weight
        else:
            self.register_buffer("weight", weight)

    def forward(self, hidden):
        return torch.nn.functional.linear(hidden, self.weight)


def trajectory_loss(features, weight, targets, mask, *, vocab_size, forbidden_ids):
    """Called inside the DDP model forward so head gradients are not marked unused."""
    total = features.sum() * 0
    positions = int(mask.sum())
    student_head = SharedHead(weight)
    covered = torch.zeros(mask.shape[0], device=features.device, dtype=torch.bool)
    for target in targets:
        indices = target["indices"].to(features.device)
        if covered[indices].any():
            raise ValueError("OPD teacher groups overlap")
        covered[indices] = True
        teacher_head = SharedHead(target["weight"].to(features.device))
        teacher_hidden = target["hidden"].to(features.device)
        subset = mask[indices]
        term = full_vocab_reverse_kl(
            features[indices, :-1],
            teacher_hidden[:, :-1],
            student_head,
            teacher_head,
            subset,
            vocab_size=vocab_size,
            forbidden_ids=forbidden_ids,
        )
        total = total + term * subset.sum()
    if not covered.all():
        raise ValueError("OPD teacher groups do not cover every student trajectory")
    return total / max(1, positions)


def reverse_kl_from_logits(
    student_logits, teacher_logits, *, vocab_size=None, forbidden_ids=(0, 1)
):
    width = student_logits.shape[-1] if vocab_size is None else vocab_size
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("OPD needs the same vocabulary and states")
    legal = torch.ones(width, device=student_logits.device, dtype=torch.bool)
    legal[list(set((0, 1, *forbidden_ids)) & set(range(width)))] = False
    student_logp = action_logits(student_logits, width, forbidden_ids).log_softmax(-1)[..., :width][
        ..., legal
    ]
    teacher_logp = action_logits(teacher_logits.detach(), width, forbidden_ids).log_softmax(-1)[
        ..., :width
    ][..., legal]
    return (student_logp.exp() * (student_logp - teacher_logp)).sum(-1)


def full_vocab_reverse_kl(
    student_hidden,
    teacher_hidden,
    student_head,
    teacher_head,
    response_mask,
    *,
    chunk_size=64,
    vocab_size=None,
    forbidden_ids=(0, 1),
):
    if chunk_size < 1 or student_hidden.shape[:2] != response_mask.shape:
        raise ValueError("invalid OPD positions or chunk size")
    if teacher_hidden.shape[:2] != response_mask.shape:
        raise ValueError("teacher must score the same student trajectory")
    if any(p.requires_grad for p in teacher_head.parameters()):
        raise ValueError("teacher head must be frozen")
    mask = response_mask.flatten().bool()
    student = student_hidden.flatten(0, 1)[mask]
    teacher = teacher_hidden.detach().flatten(0, 1)[mask]

    def term(s, t):
        with torch.no_grad():
            teacher_logits = teacher_head(t)
        return reverse_kl_from_logits(
            student_head(s), teacher_logits, vocab_size=vocab_size, forbidden_ids=forbidden_ids
        ).sum()

    total = student.sum() * 0
    for start in range(0, student.shape[0], chunk_size):
        total = total + checkpoint(
            term,
            student[start : start + chunk_size],
            teacher[start : start + chunk_size],
            use_reentrant=False,
        )
    return total / max(1, student.shape[0])
