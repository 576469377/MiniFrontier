# MLA originates in Kimi c5d1dd4 (Kimi K3 License); block routing is a local integration.
"""Token MLA with a sparse block directory; raw latent KV is never pooled away."""

import torch
from torch import nn

from .indexer import (
    BlockIndexer,
    block_members,
    block_registry,
    gather_support,
    index_kl,
    masked_probabilities,
    mrope,
    rope,
    sampled_queries,
    select_blocks,
    visible_blocks,
)
from .moe import RMSNorm


class QSAMLA(nn.Module):
    def __init__(self, c, *, dense_only=False):
        super().__init__()
        self.config = c
        h, heads = c.hidden_size, c.num_attention_heads
        self.q_down = nn.Linear(h, c.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(c.q_lora_rank, c.rms_norm_eps)
        self.q_up = nn.Linear(
            c.q_lora_rank, heads * (c.qk_nope_head_dim + c.qk_rope_head_dim), bias=False
        )
        self.kv_down = nn.Linear(h, c.kv_lora_rank + c.qk_rope_head_dim, bias=False)
        self.kv_norm = RMSNorm(c.kv_lora_rank, c.rms_norm_eps)
        self.kv_up = nn.Linear(
            c.kv_lora_rank, heads * (c.qk_nope_head_dim + c.v_head_dim), bias=False
        )
        self.gate = nn.Linear(h, heads * c.v_head_dim, bias=False)
        self.out = nn.Linear(heads * c.v_head_dim, h, bias=False)
        self.indexer = None if dense_only else BlockIndexer(c)
        self.training_phase = "dense_pretrain"
        self.indexer_loss_enabled = True

    def forward(self, x, metadata, state=None):
        c = self.config
        outputs, states, losses = [], [], []
        query_count = 0
        for batch in range(x.shape[0]):
            z = x[batch]
            old = {} if state is None else state[batch]
            offset = int(old.get("length", 0))
            seg, mod, media = (metadata[k][batch] for k in ("segment_ids", "modality", "media_ids"))
            pos = metadata["position_ids"][:, batch]
            qp = torch.arange(offset, offset + len(z), device=z.device)
            linear_positions = (
                metadata["linear_positions"][batch] if "linear_positions" in metadata else qp
            )
            q = self.q_up(self.q_norm(self.q_down(z))).unflatten(
                -1, (c.num_attention_heads, c.qk_nope_head_dim + c.qk_rope_head_dim)
            )
            qc, qr = q.split((c.qk_nope_head_dim, c.qk_rope_head_dim), -1)
            qr = mrope(qr, pos, c.mrope_sections, c.rope_theta)
            latent, kr = self.kv_down(z).split((c.kv_lora_rank, c.qk_rope_head_dim), -1)
            latent, kr = self.kv_norm(latent), mrope(kr, pos, c.mrope_sections, c.rope_theta)
            current = dict(
                latent=latent,
                rope=kr,
                segment_ids=seg,
                modality=mod,
                media_ids=media,
                linear_positions=linear_positions,
            )
            if self.indexer is not None:
                current["index_raw"] = self.indexer.key(z.detach())
            full = {
                key: torch.cat((old[key], value), 0) if key in old else value
                for key, value in current.items()
            }
            total = len(full["latent"])
            kp = torch.arange(total, device=z.device)
            blocks = block_registry(full["segment_ids"], full["modality"], full["media_ids"])
            blocks = [b for b in blocks if b.complete_at is not None]
            keys = z.new_empty((0, c.index_dim))
            members, member_valid = block_members(blocks, z.device)
            if self.indexer is not None and blocks:
                keys = (full["index_raw"][members] * member_valid[..., None]).sum(1)
                keys = keys / member_valid.sum(1, keepdim=True)
                keys = rope(
                    keys,
                    full["linear_positions"][[b.start for b in blocks]],
                    c.index_rope_dim,
                    c.rope_theta,
                )
            # Expanded projections are temporary, not persisted in the latent cache.
            kc, values = (
                self.kv_up(full["latent"])
                .unflatten(-1, (c.num_attention_heads, c.qk_nope_head_dim + c.v_head_dim))
                .split((c.qk_nope_head_dim, c.v_head_dim), -1)
            )
            chunks = []
            # Deterministic uniform query sampling, independent of global RNG/checkpoint replay.
            sampled = (
                sampled_queries(seg, c.index_query_count)
                if self.indexer_loss_enabled and self.training_phase != "dense_pretrain"
                else []
            )
            for start in range(0, len(z), c.query_chunk_size):
                stop = min(start + c.query_chunk_size, len(z))
                causal = (
                    (qp[start:stop, None] >= kp)
                    & (seg[start:stop, None] == full["segment_ids"][None])
                    & seg[start:stop, None].ge(0)
                )
                local = causal & (qp[start:stop, None] - kp < c.window_size)
                protected = (
                    causal & full["modality"][None].ne(0)
                    if c.protect_media
                    else torch.zeros_like(causal)
                )
                visible = visible_blocks(blocks, qp[start:stop], seg[start:stop])
                scores = (
                    self.indexer.scores(z[start:stop].detach(), linear_positions[start:stop], keys)
                    if self.indexer is not None
                    and blocks
                    and (
                        self.training_phase == "sparse_cpt"
                        or (self.indexer_loss_enabled and self.training_phase != "dense_pretrain")
                    )
                    else z.new_empty((stop - start, 0))
                )
                selected = (
                    select_blocks(scores, visible, c.top_blocks)
                    if blocks and self.indexer is not None and self.training_phase == "sparse_cpt"
                    else visible
                )
                support = causal
                if self.indexer is not None and self.training_phase == "sparse_cpt":
                    support = local | protected
                    if blocks:
                        directory = torch.full((total,), -1, device=z.device, dtype=torch.long)
                        directory[members[member_valid]] = torch.arange(
                            len(blocks), device=z.device
                        )[:, None].expand_as(members)[member_valid]
                        support |= selected[:, directory.clamp_min(0)] & directory.ge(0)
                    support &= causal
                    # Gather a bounded query chunk, rather than synchronizing once per token.
                    indices, valid = gather_support(support)
                    logits = (
                        torch.einsum("qhd,qkhd->qhk", qc[start:stop].float(), kc[indices].float())
                        + torch.einsum(
                            "qhd,qkd->qhk", qr[start:stop].float(), full["rope"][indices].float()
                        )
                    ) * (c.qk_nope_head_dim + c.qk_rope_head_dim) ** -0.5
                    chunks.append(
                        torch.einsum(
                            "qhk,qkhd->qhd",
                            masked_probabilities(logits, valid[:, None]).to(values.dtype),
                            values[indices],
                        )
                    )
                else:
                    logits = (
                        torch.einsum("qhd,khd->qhk", qc[start:stop].float(), kc.float())
                        + torch.einsum("qhd,kd->qhk", qr[start:stop].float(), full["rope"].float())
                    ) * (c.qk_nope_head_dim + c.qk_rope_head_dim) ** -0.5
                    chunks.append(
                        torch.einsum(
                            "qhk,khd->qhd",
                            masked_probabilities(logits, causal[:, None]).to(values.dtype),
                            values,
                        )
                    )
                chosen = [i - start for i in sorted(sampled) if start <= i < stop]
                if (
                    self.indexer is not None
                    and self.indexer_loss_enabled
                    and self.training_phase != "dense_pretrain"
                    and blocks
                    and chosen
                ):
                    with torch.no_grad():
                        dense = (
                            torch.einsum("qhd,khd->qhk", qc[start:stop][chosen].float(), kc.float())
                            + torch.einsum(
                                "qhd,kd->qhk", qr[start:stop][chosen].float(), full["rope"].float()
                            )
                        ) * (c.qk_nope_head_dim + c.qk_rope_head_dim) ** -0.5
                        teacher = masked_probabilities(dense, causal[chosen, None]).mean(1)
                        # Always-retained local/media tokens do not reward the indexer.
                        teacher = teacher * ~(local[chosen] | protected[chosen])
                        target = (teacher[:, members] * member_valid).sum(-1)
                    term, count = index_kl(scores[chosen], target, visible[chosen])
                    losses.append(term)
                    query_count += count
            out = torch.cat(chunks).flatten(-2)
            outputs.append(self.out(out * self.gate(z).sigmoid()))
            states.append(dict(full, length=total, blocks=blocks, index_keys=keys))
        loss = sum(losses, x.sum() * 0)
        return torch.stack(outputs), states, loss, query_count
