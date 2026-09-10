"""One GR-aware MTP block, with an explicit two-step target mask and shared main head."""

import torch
from torch import nn

from .moe import LatentMoE, RMSNorm
from .qsa_mla import QSAMLA
from .residual import GatedResidual


def mtp_targets(ids, labels, metadata, c):
    shifted = torch.zeros_like(ids)
    shifted[:, :-1] = ids[:, 1:]
    targets = torch.full_like(labels, -100)
    seg = metadata["segment_ids"]
    text = metadata["modality"].eq(0)
    permitted = ids.ge(c.control_token_count) | ids.eq(c.eos_token_id)
    valid = (
        labels[:, 1:-1].ne(-100)
        & labels[:, 2:].ne(-100)
        & text[:, 1:-1]
        & text[:, 2:]
        & permitted[:, 1:-1]
        & permitted[:, 2:]
        & seg[:, :-2].eq(seg[:, 1:-1])
        & seg[:, :-2].eq(seg[:, 2:])
        & seg[:, :-2].ge(0)
        & ids[:, 1:-1].ne(c.eos_token_id)
    )
    targets[:, :-2] = labels[:, 2:].masked_fill(~valid, -100)
    return shifted, targets


class MF1MTPBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.stream_norm = RMSNorm(c.hc_count * c.hidden_size, c.rms_norm_eps)
        self.stream_width = c.hidden_size
        self.hidden_proj = nn.Linear(c.hidden_size, c.hidden_size, bias=False)
        self.embedding_proj = nn.Linear(c.hidden_size, c.hidden_size, bias=False)
        self.attention_gr = GatedResidual(c)
        self.attention = QSAMLA(c, dense_only=True)
        self.moe_gr = GatedResidual(c)
        self.moe = LatentMoE(c)
        self.final_gr = GatedResidual(c, read_only=True)

    def forward(self, residual, next_embedding, metadata):
        # Independent per-stream norms; the projection is shared across streams.
        normed = residual.float() * torch.rsqrt(
            residual.float().square().mean(-1, keepdim=True) + self.stream_norm.eps
        )
        normed = normed * self.stream_norm.weight.view(residual.shape[-2:])
        r = self.hidden_proj(normed.to(residual.dtype)) + self.embedding_proj(
            next_embedding
        ).unsqueeze(-2)
        h, weights = self.attention_gr.read(r)
        update, _, _, _ = self.attention(h, metadata, cache_output=False)
        r = self.attention_gr.inject(r, update, weights)
        h, weights = self.moe_gr.read(r)
        r = self.moe_gr.inject(r, self.moe(h, metadata.get("valid_token_indices")), weights)
        return self.final_gr(r)
