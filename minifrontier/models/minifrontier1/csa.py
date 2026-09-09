# Gated overlap pooling derives from DeepSeek 60d8d70 (MIT). Boundary handling is local.
"""CSA-4 reference with media-aware short-block flush and bounded raw/pending cache."""

from typing import cast

import torch
from torch import nn

from .indexer import BlockIndexer, block_registry, index_kl, rope, select_blocks, visible_blocks
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
        with torch.autocast(device_type=x.device.type, enabled=False):
            kv = torch.nn.functional.linear(x.float(), self.kv.weight.float())
            gates = torch.nn.functional.linear(x.float(), self.gate.weight.float())
            result = []
            for i, block in enumerate(blocks):
                ids = torch.tensor(block.member_indices, device=x.device) - offset
                values = kv[ids, self.dim :]
                scores = gates[ids, self.dim :] + self.ape[: len(ids), self.dim :]
                if i and (
                    blocks[i - 1].segment,
                    blocks[i - 1].modality,
                    blocks[i - 1].media_id,
                ) == (block.segment, block.modality, block.media_id):
                    previous = torch.tensor(blocks[i - 1].member_indices, device=x.device) - offset
                    values = torch.cat((kv[previous, : self.dim], values))
                    scores = torch.cat(
                        (
                            gates[previous, : self.dim] + self.ape[: len(previous), : self.dim],
                            scores,
                        )
                    )
                # Invalid slots are absent (equivalent to -inf pooling masks).
                result.append((values * scores.softmax(0)).sum(0))
        return self.norm(torch.stack(result)) if result else x.new_empty((0, self.dim))


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

    def forward(self, x, metadata, state=None):
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
            sampled = set(
                torch.linspace(0, max(0, len(z) - 1), min(c.index_query_count, len(z)))
                .long()
                .tolist()
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
                scores = self.indexer.scores(
                    z[start:stop].detach(), linear_positions[start:stop], keys
                )
                selected = (
                    select_blocks(scores, visible, c.top_blocks)
                    if self.training_phase == "sparse_cpt"
                    else visible
                )
                support = torch.cat((local, selected), -1)
                for j in range(stop - start):
                    i = start + j
                    indices = support[j].nonzero().flatten()
                    if not indices.numel():
                        rows.append(z.new_zeros(c.num_attention_heads, c.csa_head_dim))
                        continue
                    logits = (
                        torch.einsum("hd,kd->hk", q[i].float(), kv[indices].float())
                        * c.csa_head_dim**-0.5
                    )
                    rows.append(
                        torch.einsum("hk,kd->hd", logits.softmax(-1).to(kv.dtype), kv[indices])
                    )
                    if (
                        blocks
                        and self.indexer_loss_enabled
                        and self.training_phase != "dense_pretrain"
                        and i in sampled
                    ):
                        with torch.no_grad():
                            dense = (
                                torch.einsum("hd,kd->hk", q[i].float(), kv.float())
                                * c.csa_head_dim**-0.5
                            )
                            teacher = (
                                dense.masked_fill(~torch.cat((local[j], visible[j])), float("-inf"))
                                .softmax(-1)
                                .mean(0)[len(raw) :]
                            )
                        term, count = index_kl(scores[j : j + 1], teacher[None], visible[j : j + 1])
                        terms.append(term)
                        query_count += count
            outputs.append(self.out(torch.stack(rows).flatten(-2)))
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
