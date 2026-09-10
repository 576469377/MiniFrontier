# Gated overlap pooling derives from DeepSeek 60d8d70 (MIT). Boundary handling is local.
"""CSA-4 reference with media-aware short-block flush and bounded raw/pending cache."""

from itertools import pairwise
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

from .indexer import (
    BlockIndexer,
    block_members,
    block_registry,
    gather_support,
    index_kl,
    masked_probabilities,
    prefill_directory,
    rope,
    sampled_queries,
    select_blocks,
    visible_blocks,
)
from .moe import RMSNorm


class Compressor(nn.Module):
    def __init__(self, width, dim, eps):
        super().__init__()
        self.dim = dim
        self.kv = nn.Linear(width, 2 * dim, bias=False)
        self.gate = nn.Linear(width, 2 * dim, bias=False)
        self.ape = nn.Parameter(torch.zeros(4, 2 * dim))
        self.norm = RMSNorm(dim, eps)

    def forward(self, x, blocks, offset):
        if not blocks:
            return x.new_empty((0, self.dim))
        with torch.autocast(device_type=x.device.type, enabled=False):
            kv = torch.nn.functional.linear(x.float(), self.kv.weight.float())
            gates = torch.nn.functional.linear(x.float(), self.gate.weight.float())
            ids, valid = block_members(blocks, x.device, offset=offset)
            previous = torch.arange(len(blocks), device=x.device).sub(1).clamp_min(0)
            overlaps = torch.tensor(
                [False]
                + [
                    (a.segment, a.modality, a.media_id) == (b.segment, b.modality, b.media_id)
                    for a, b in pairwise(blocks)
                ],
                device=x.device,
            )
            support = torch.cat((valid[previous] & overlaps[:, None], valid), 1)
            values = torch.cat((kv[ids[previous], : self.dim], kv[ids, self.dim :]), 1)
            scores = torch.cat(
                (
                    gates[ids[previous], : self.dim] + self.ape[None, :, : self.dim],
                    gates[ids, self.dim :] + self.ape[None, :, self.dim :],
                ),
                1,
            ).masked_fill(~support[..., None], float("-inf"))
            pooled = (values * scores.softmax(1)).sum(1)
        return self.norm(pooled)

    def batched(self, x, directory):
        ids, valid = directory["members"], directory["valid"]
        if not ids.shape[1]:
            return x.new_empty((len(x), 0, self.dim))
        with torch.autocast(device_type=x.device.type, enabled=False):
            kv = F.linear(x.float(), self.kv.weight.float())
            gates = F.linear(x.float(), self.gate.weight.float())
            batch = torch.arange(len(x), device=x.device)[:, None, None]
            values, scores = kv[batch, ids], gates[batch, ids] + self.ape
            previous = torch.arange(ids.shape[1], device=x.device).sub(1).clamp_min(0)
            support = torch.cat((valid[:, previous] & directory["overlap"][..., None], valid), -1)
            v = torch.cat((values[:, previous, :, : self.dim], values[..., self.dim :]), -2)
            g = torch.cat((scores[:, previous, :, : self.dim], scores[..., self.dim :]), -2)
            # Padded directory entries must have finite zero outputs/gradients.
            probabilities = masked_probabilities(g.transpose(-1, -2), support[..., None, :])
            pooled = (v * probabilities.transpose(-1, -2)).sum(-2)
        return self.norm(pooled)


class CSA(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.q_down = nn.Linear(c.hidden_size, c.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(c.q_lora_rank, c.rms_norm_eps)
        self.q_up = nn.Linear(c.q_lora_rank, c.num_attention_heads * c.csa_head_dim, bias=False)
        self.kv = nn.Linear(c.hidden_size, c.csa_head_dim, bias=False)
        self.kv_norm = RMSNorm(c.csa_head_dim, c.rms_norm_eps)
        self.out = nn.Linear(c.num_attention_heads * c.csa_head_dim, c.hidden_size, bias=False)
        self.compressor = Compressor(c.hidden_size, c.csa_head_dim, c.rms_norm_eps)
        self.indexer = BlockIndexer(c)
        # The CSA indexer has its own compressor. Its unused token-key projection is removed.
        del self.indexer.key
        self.indexer.compressor = Compressor(c.hidden_size, c.index_dim, c.rms_norm_eps)
        self.training_phase = "dense_pretrain"
        self.indexer_loss_enabled = True

    def forward(self, x, metadata, state=None, *, cache_output=True):
        if not cache_output and state is None and self.training_phase == "dense_pretrain":
            return self.dense_prefill(x, metadata), None, x.sum() * 0, 0
        c = self.config
        outputs, states, terms = [], [], []
        query_count = 0
        for b in range(len(x)):
            z, old = x[b], {} if state is None else state[b]
            offset = old.get("length", 0)
            qp = torch.arange(offset, offset + len(z), device=z.device)
            linear_positions = (
                metadata["linear_positions"][b] if "linear_positions" in metadata else qp
            )
            seg, mod, media = (metadata[k][b] for k in ("segment_ids", "modality", "media_ids"))
            q = self.q_up(self.q_norm(self.q_down(z))).unflatten(
                -1, (c.num_attention_heads, c.csa_head_dim)
            )
            q = rope(q, linear_positions, c.csa_rope_dim, c.rope_theta)
            raw = rope(self.kv_norm(self.kv(z)), linear_positions, c.csa_rope_dim, c.rope_theta)
            raw = torch.cat((old["raw"], raw)) if "raw" in old else raw
            raw_seg = torch.cat((old["raw_segment"], seg)) if "raw_segment" in old else seg
            raw_pos = torch.arange(offset + len(z) - len(raw), offset + len(z), device=z.device)
            tail = {
                "x": z,
                "segment_ids": seg,
                "modality": mod,
                "media_ids": media,
                "linear_positions": linear_positions,
            }
            tail = {
                k: torch.cat((old["tail"][k], v)) if "tail" in old else v for k, v in tail.items()
            }
            tail_offset = offset + len(z) - len(tail["x"])
            registry = block_registry(
                tail["segment_ids"], tail["modality"], tail["media_ids"], offset=tail_offset
            )
            complete = [block for block in registry if block.complete_at is not None]
            pooled = self.compressor(tail["x"], registry, tail_offset)[: len(complete)]
            index_keys = cast(Compressor, self.indexer.compressor)(
                tail["x"].detach(), registry, tail_offset
            )[: len(complete)]
            starts = tail["linear_positions"][[block.start - tail_offset for block in complete]]
            pooled = rope(pooled, starts, c.csa_rope_dim, c.rope_theta)
            index_keys = rope(index_keys, starts, c.index_rope_dim, c.rope_theta)
            # Already committed blocks are immutable; recomputing the retained overlap is not a new entry.
            existing = old.get("blocks", [])
            new_ids = [
                i
                for i, block in enumerate(complete)
                if not existing or block.start > existing[-1].start
            ]
            blocks = [*existing, *(complete[i] for i in new_ids)]
            compressed = (
                torch.cat((old["compressed"], pooled[new_ids])) if existing else pooled[new_ids]
            )
            keys = (
                torch.cat((old["index_keys"], index_keys[new_ids]))
                if existing
                else index_keys[new_ids]
            )
            kv = torch.cat((raw, compressed.to(raw.dtype)))
            rows = []
            sampled = (
                sampled_queries(seg, c.index_query_count)
                if self.indexer_loss_enabled and self.training_phase != "dense_pretrain"
                else []
            )
            for start in range(0, len(z), c.query_chunk_size):
                stop = min(start + c.query_chunk_size, len(z))
                local = (
                    (qp[start:stop, None] >= raw_pos)
                    & (qp[start:stop, None] - raw_pos < c.window_size)
                    & (seg[start:stop, None] == raw_seg)
                    & seg[start:stop, None].ge(0)
                )
                visible = visible_blocks(blocks, qp[start:stop], seg[start:stop])
                needs_indexer = self.training_phase == "sparse_cpt" or (
                    self.indexer_loss_enabled and self.training_phase != "dense_pretrain"
                )
                scores = (
                    self.indexer.scores(z[start:stop].detach(), linear_positions[start:stop], keys)
                    if needs_indexer
                    else z.new_empty((stop - start, 0))
                )
                selected = (
                    select_blocks(scores, visible, c.top_blocks)
                    if self.training_phase == "sparse_cpt"
                    else visible
                )
                support = torch.cat((local, selected), -1)
                indices, valid = gather_support(support)
                gathered = kv[indices]
                logits = (
                    torch.einsum("qhd,qkd->qhk", q[start:stop].float(), gathered.float())
                    * c.csa_head_dim**-0.5
                )
                rows.append(
                    torch.einsum(
                        "qhk,qkd->qhd",
                        masked_probabilities(logits, valid[:, None]).to(kv.dtype),
                        gathered,
                    )
                )
                chosen = [i - start for i in sorted(sampled) if start <= i < stop]
                if (
                    blocks
                    and self.indexer_loss_enabled
                    and self.training_phase != "dense_pretrain"
                    and chosen
                ):
                    with torch.no_grad():
                        dense = (
                            torch.einsum("qhd,kd->qhk", q[start:stop][chosen].float(), kv.float())
                            * c.csa_head_dim**-0.5
                        )
                        teacher = masked_probabilities(
                            dense, torch.cat((local[chosen], visible[chosen]), -1)[:, None]
                        ).mean(1)[:, len(raw) :]
                    term, count = index_kl(scores[chosen], teacher, visible[chosen])
                    terms.append(term)
                    query_count += count
            outputs.append(self.out(torch.cat(rows).flatten(-2)))
            # Keep the previous block plus pending/current block, enough for exact overlap replay.
            keep = registry[-2].start if len(registry) > 1 else tail_offset
            tail = {k: v[keep - tail_offset :].clone() for k, v in tail.items()}
            states.append(
                dict(
                    length=offset + len(z),
                    raw=raw[-(c.window_size - 1) :].clone(),
                    raw_segment=raw_seg[-(c.window_size - 1) :].clone(),
                    tail=tail,
                    blocks=blocks,
                    compressed=compressed,
                    index_keys=keys,
                )
            )
        return torch.stack(outputs), states, sum(terms, x.sum() * 0), query_count

    def dense_prefill(self, x, metadata):
        """Batched CSA with bounded SDPA masks; no inference cache or unused indexer."""
        c = self.config
        directory = metadata.get("prefill_directory")
        if directory is None:
            directory = prefill_directory(metadata)
        pos, seg = metadata["linear_positions"], metadata["segment_ids"]
        q = self.q_up(self.q_norm(self.q_down(x))).unflatten(
            -1, (c.num_attention_heads, c.csa_head_dim)
        )
        q = rope(q, pos, c.csa_rope_dim, c.rope_theta).transpose(1, 2)
        raw = rope(self.kv_norm(self.kv(x)), pos, c.csa_rope_dim, c.rope_theta)
        pooled = self.compressor.batched(x, directory)
        pooled = rope(pooled, pos.gather(1, directory["starts"]), c.csa_rope_dim, c.rope_theta)
        kv = torch.cat((raw, pooled.to(raw.dtype)), 1)[:, None].expand(
            -1, c.num_attention_heads, -1, -1
        )
        kp = torch.arange(x.shape[1], device=x.device)
        chunks = []
        for start in range(0, x.shape[1], c.query_chunk_size):
            stop = min(start + c.query_chunk_size, x.shape[1])
            qp = kp[start:stop, None]
            local = (
                (qp >= kp)
                & (qp - kp < c.window_size)
                & (seg[:, start:stop, None] == seg[:, None])
                & seg[:, start:stop, None].ge(0)
            )
            compressed = (
                (directory["complete"][:, None] <= qp)
                & (directory["segments"][:, None] == seg[:, start:stop, None])
                & seg[:, start:stop, None].ge(0)
            )
            support = torch.cat((local, compressed), -1)[:, None]
            chunks.append(
                F.scaled_dot_product_attention(
                    q[:, :, start:stop],
                    kv,
                    kv,
                    attn_mask=support,
                    dropout_p=0.0,
                )
            )
        return self.out(torch.cat(chunks, 2).transpose(1, 2).flatten(-2))
