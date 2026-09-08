"""Exact full-vocabulary reverse KL on fixed student-generated states.

Chunk only the sequence positions: every chunk retains the full normalization
over the legal vocabulary. Teacher features and head are frozen. Checkpointing
recomputes head logits during backward instead of retaining all vocab activations.
This is a differentiable KL objective, distinct from Kimi's sampled token reward.
"""

import torch
from torch.utils.checkpoint import checkpoint

from .distributions import action_logits


def reverse_kl_from_logits(student_logits, teacher_logits, *, vocab_size=None):
    width = student_logits.shape[-1] if vocab_size is None else vocab_size
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("OPD needs the same vocabulary and states")
    student_logp = action_logits(student_logits, width).log_softmax(-1)[..., 2:width]
    teacher_logp = action_logits(teacher_logits.detach(), width).log_softmax(-1)[..., 2:width]
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
        return reverse_kl_from_logits(student_head(s), teacher_logits, vocab_size=vocab_size).sum()

    total = student.sum() * 0
    for start in range(0, student.shape[0], chunk_size):
        total = total + checkpoint(
            term,
            student[start : start + chunk_size],
            teacher[start : start + chunk_size],
            use_reentrant=False,
        )
    return total / max(1, student.shape[0])
