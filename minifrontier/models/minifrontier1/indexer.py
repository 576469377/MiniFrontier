"""Real block membership, causal completion times and independent retrieval projections."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class Block:
    member_indices: tuple[int, ...]
    segment: int
    modality: int
    media_id: int
    complete_at: int | None

    @property
    def start(self):
        return self.member_indices[0]

    @property
    def end(self):
        return self.member_indices[-1]


def block_registry(segments, modalities, media_ids, *, offset=0, size=4):
    """A segment-tail block becomes visible when its boundary token arrives, never earlier."""
    if not (segments.shape == modalities.shape == media_ids.shape) or segments.ndim != 1:
        raise ValueError("registry metadata must be matching one-dimensional tensors")
    triples = list(zip(segments.tolist(), modalities.tolist(), media_ids.tolist(), strict=True))
    result = []
    left = 0
    while left < len(triples):
        key = triples[left]
        end = left + 1
        while end < len(triples) and triples[end] == key:
            end += 1
        if key[0] >= 0:
            for start in range(left, end, size):
                stop = min(start + size, end)
                complete = stop - 1 if stop - start == size else end if end < len(triples) else None
                result.append(
                    Block(
                        tuple(range(offset + start, offset + stop)),
                        *key,
                        None if complete is None else offset + complete,
                    )
                )
        left = end
    return result


def rope(x, positions, dim, theta=10000.0):
    """Rotate the last real dimensions in adjacent pairs; axes are supplied explicitly."""
    if not dim:
        return x
    freq = theta ** (-torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim)
    angle = positions.float()[..., None] * freq
    while angle.ndim < x.ndim:
        angle = angle.unsqueeze(-2)
    pairs = x[..., -dim:].float().unflatten(-1, (-1, 2))
    a, b = pairs.unbind(-1)
    rotated = torch.stack(
        (a * angle.cos() - b * angle.sin(), a * angle.sin() + b * angle.cos()), -1
    ).flatten(-2)
    return torch.cat((x[..., :-dim], rotated.to(x.dtype)), -1)


def mrope(x, positions, sections, theta):
    # One frequency table, split by axis; sections are numbers of frequency pairs.
    dim = 2 * sum(sections)
    freq = theta ** (-torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim)
    angles, cursor = [], 0
    for axis, width in enumerate(sections):
        angles.append(positions[axis].float()[..., None] * freq[cursor : cursor + width])
        cursor += width
    angle = torch.cat(angles, -1)
    while angle.ndim < x.ndim:
        angle = angle.unsqueeze(-2)
    a, b = x.float().unflatten(-1, (-1, 2)).unbind(-1)
    return (
        torch.stack((a * angle.cos() - b * angle.sin(), a * angle.sin() + b * angle.cos()), -1)
        .flatten(-2)
        .to(x.dtype)
    )


class BlockIndexer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads, self.dim = c.index_heads, c.index_dim
        self.rope_dim, self.theta = c.index_rope_dim, c.rope_theta
        self.q = nn.Linear(c.hidden_size, self.heads * self.dim, bias=False)
        self.key = nn.Linear(c.hidden_size, self.dim, bias=False)
        self.weight = nn.Linear(c.hidden_size, self.heads, bias=False)

    def queries(self, x, positions):
        return rope(
            self.q(x).unflatten(-1, (self.heads, self.dim)), positions, self.rope_dim, self.theta
        )

    def scores(self, x, positions, keys):
        q = self.queries(x, positions)
        dots = torch.einsum("qhd,kd->qhk", q.float(), keys.float()).relu()
        weights = self.weight(x).float() * (self.heads * self.dim) ** -0.5
        return (dots * weights[..., None]).sum(1)


def visible_blocks(blocks, query_positions, query_segments):
    complete = torch.tensor(
        [b.complete_at if b.complete_at is not None else 2**60 for b in blocks],
        device=query_positions.device,
    )
    segment = torch.tensor([b.segment for b in blocks], device=query_positions.device)
    return (
        (complete[None] <= query_positions[:, None])
        & (segment[None] == query_segments[:, None])
        & query_segments[:, None].ge(0)
    )


def select_blocks(scores, visible, budget):
    # Mask BEFORE sorting. Stable ties prefer the earlier real registry entry.
    indices = scores.masked_fill(~visible, float("-inf")).argsort(
        dim=-1, descending=True, stable=True
    )[..., :budget]
    return torch.zeros_like(visible).scatter(-1, indices, True) & visible


def block_members(blocks, device, *, offset=0):
    """Materialize the small CPU directory once, including short boundary blocks."""
    indices = torch.tensor(
        [list(b.member_indices) + [-1] * (4 - len(b.member_indices)) for b in blocks],
        device=device,
        dtype=torch.long,
    ).reshape(-1, 4)
    valid = indices.ge(0)
    return (indices - offset).clamp_min(0), valid


def gather_support(support):
    """One bounded gather per query chunk, preserving increasing key order.

    A single size synchronization replaces one nonzero synchronization per query.
    Padding is masked; no routed token is dropped or capacity-clipped.
    """
    size = max(1, int(support.sum(-1).max()))
    positions = torch.arange(support.shape[-1], device=support.device).expand_as(support)
    indices = (
        positions.masked_fill(~support, support.shape[-1])
        .topk(size, dim=-1, largest=False, sorted=True)
        .values
    )
    return indices.clamp_max(support.shape[-1] - 1), indices.lt(support.shape[-1])


def masked_probabilities(logits, support):
    """Fully masked/padded queries produce zero values and zero gradients."""
    valid = support.any(-1, keepdim=True)
    masked = logits.masked_fill(~support, float("-inf"))
    return masked.masked_fill(~valid, 0).softmax(-1) * valid


def sampled_queries(segments, limit):
    """Uniform real query positions; padding must not change an indexer's teacher sample."""
    valid = [i for i, segment in enumerate(segments.tolist()) if segment >= 0]
    if not valid:
        return []
    indices = torch.linspace(0, len(valid) - 1, min(limit, len(valid))).long().tolist()
    return [valid[i] for i in indices]


def index_kl(scores, target, visible):
    target = target.detach().float() * visible
    mass = target.sum(-1)
    valid = mass > 1e-8
    normalized = target / mass[:, None].clamp_min(1e-12)
    logp = scores.masked_fill(~visible, torch.finfo(scores.dtype).min).log_softmax(-1)
    losses = (normalized * (normalized.clamp_min(1e-12).log() - logp)).sum(-1)
    return losses[valid].sum(), int(valid.sum())
