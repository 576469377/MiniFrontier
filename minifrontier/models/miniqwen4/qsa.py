"""QSA training objectives, report 6988587 §2.1.2 equations 17--20.

Indexer computations follow HF 4177486, including its 1/sqrt(head_dim) score
scale (absent from report equation 15). Teacher probabilities and indexer input
are stop-gradient: KL trains the indexer, while LM trains the sparse backbone.
This is an eager reference, not the report's fused long-context training kernel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from .upstream_decoder import Qwen4ExpTextQSAIndexer, apply_rotary_pos_emb


@dataclass
class MiniQwen4QSARow:
    batch: int
    query: int
    block_tokens: Tensor
    scores: Tensor
    selected_blocks: Tensor


def indexer_score_rows(
    indexer: Qwen4ExpTextQSAIndexer,
    hidden_states: Tensor,
    position_embeddings: tuple[Tensor, Tensor],
    attention_mask: Tensor,
) -> list[MiniQwen4QSARow]:
    """Expose differentiable scores with the original source's block ordering.

    Cached decoding is intentionally excluded from training. The mask must be
    the full causal/padding mask, before sparse selection is added.
    """
    batch_size, seq_length, _ = hidden_states.shape
    if attention_mask.shape != (batch_size, 1, seq_length, seq_length):
        raise ValueError("QSA training requires a square, uncached causal mask")
    full_cos, full_sin = position_embeddings
    hidden_shape = (batch_size, seq_length, -1, indexer.index_head_dim)
    qk = indexer.index_qk_proj(hidden_states)
    q, token_k = torch.split(
        qk,
        [
            indexer.index_n_heads * indexer.index_head_dim,
            indexer.index_kv_heads * indexer.index_head_dim,
        ],
        dim=-1,
    )
    q, raw_keys = q.reshape(*hidden_shape), token_k.reshape(*hidden_shape).squeeze(2)
    q = indexer.q_layernorm(q)
    q = apply_rotary_pos_emb(q, cos=full_cos, sin=full_sin, unsqueeze_dim=2)
    visible = attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0
    rows = []
    for batch_idx in range(batch_size):
        for query_idx in range(seq_length):
            indices = torch.nonzero(visible[batch_idx, 0, query_idx], as_tuple=False).flatten()
            count = indices.shape[-1] // indexer.compress_ratio
            if count == 0:
                continue
            block_tokens = indices[: count * indexer.compress_ratio].view(
                count, indexer.compress_ratio
            )
            key_groups = raw_keys[batch_idx].index_select(0, block_tokens.flatten())
            key_groups = key_groups.view(*block_tokens.shape, indexer.index_head_dim)
            pooled_keys = key_groups.float().mean(dim=1).to(raw_keys.dtype)
            pooled_keys = indexer.k_layernorm(pooled_keys)
            starts = block_tokens[:, 0]
            block_keys = apply_rotary_pos_emb(
                pooled_keys.unsqueeze(1),
                cos=full_cos[batch_idx].index_select(0, starts),
                sin=full_sin[batch_idx].index_select(0, starts),
            ).squeeze(1)
            scores = torch.matmul(
                q[batch_idx, query_idx].float(), block_keys.float().transpose(-1, -2)
            ).transpose(-1, -2)
            scores = torch.relu(scores).sum(dim=-1) / math.sqrt(indexer.index_head_dim)
            selected = scores.topk(min(indexer.block_topk, count), dim=0).indices
            rows.append(MiniQwen4QSARow(batch_idx, query_idx, block_tokens, scores, selected))
    return rows


def indexer_kl_loss(
    indexer: Qwen4ExpTextQSAIndexer,
    hidden_states: Tensor,
    position_embeddings: tuple[Tensor, Tensor],
    full_attention_mask: Tensor,
    teacher_attention: Tensor,
    query_valid: Tensor,
    *,
    selected_only: bool,
) -> Tensor:
    """Head sum/L1 -> complete-block max pool/L1 -> teacher||indexer KL.

    The denominator is ALL valid queries (report N), including early queries
    with no complete block, whose contribution is zero. Padded queries and tail
    tokens do not become distillation targets. Sparse teacher probabilities are
    equivalent to full teacher probabilities restricted/renormalized on the
    selected blocks, apart from finite-precision rounding.
    """
    if query_valid.shape != hidden_states.shape[:2] or not bool(query_valid.bool().any()):
        raise ValueError("QSA loss requires a nonempty valid-query mask")
    batch, length = query_valid.shape
    if (
        teacher_attention.ndim != 4
        or teacher_attention.shape[0] != batch
        or teacher_attention.shape[-2:] != (length, length)
    ):
        raise ValueError("teacher attention must be [batch, heads, sequence, sequence]")
    if not bool(torch.isfinite(teacher_attention).all()) or bool((teacher_attention < 0).any()):
        raise ValueError("teacher attention must be finite and nonnegative")
    rows = indexer_score_rows(
        indexer, hidden_states.detach(), position_embeddings, full_attention_mask
    )
    teacher = teacher_attention.detach().float().sum(dim=1)
    teacher = F.normalize(teacher, p=1, dim=-1)
    # Keep every indexer parameter connected even for short/empty-block batches.
    total = sum(
        (p.float().sum() * 0 for p in indexer.parameters()),
        hidden_states.new_zeros((), dtype=torch.float32),
    )
    for row in rows:
        if not bool(query_valid[row.batch, row.query]):
            continue
        probability = teacher[row.batch, row.query].index_select(0, row.block_tokens.flatten())
        probability = probability.view(row.block_tokens.shape).amax(dim=-1)
        scores = row.scores
        if selected_only:
            probability = probability.index_select(0, row.selected_blocks)
            scores = scores.index_select(0, row.selected_blocks)
        mass = probability.sum()
        if not bool(mass > 0):
            raise ValueError("teacher has zero mass on eligible complete blocks")
        probability = probability / mass
        total = total + F.kl_div(
            F.log_softmax(scores.float(), dim=-1), probability, reduction="sum"
        )
    return total / query_valid.bool().sum()
