"""Differentiable CSA2 attention from DeepSeek-V4.1 dba1be0 (MIT).

Reference: inference/model.py (Attention, Indexer, Compressor).
The training adaptation uses explicit per-forward state, unquantized tensors and
query chunks. Indexer KL uses detached sparse attention probabilities, following
the existing V4 training adapter; the V4.1 inference release supplies no loss.
This implementation does not claim FP4 kernels or approximate bounded replay.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .layers import RMSNorm


@dataclass(frozen=True)
class CSA2State:
    kv: Tensor | None = None
    index_key: Tensor | None = None
    indices: Tensor | None = None
    candidates: Tensor | None = None
    valid: Tensor | None = None
    ratio: int = 0
    owner: int = -1


def rotary(x: Tensor, positions: Tensor, dim: int, theta: float, *, inverse=False) -> Tensor:
    """Rotate the final channels using adjacent real/imaginary pairs."""
    angles = positions.float()[..., None] * theta ** (
        -torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim
    )
    angles = angles.unsqueeze(0)
    while angles.ndim < x.ndim:
        angles = angles.unsqueeze(-2)
    real, imag = x[..., -dim:].float().unflatten(-1, (-1, 2)).unbind(-1)
    sine = angles.sin() * (-1 if inverse else 1)
    result = torch.stack(
        (real * angles.cos() - imag * sine, real * sine + imag * angles.cos()), -1
    ).flatten(-2)
    return torch.cat((x[..., :-dim], result.to(x.dtype)), -1)


class Compressor(nn.Module):
    def __init__(self, config: Any, ratio: int):
        super().__init__()
        self.ratio = ratio
        self.wkv = nn.Linear(config.dim, config.head_dim, bias=False)
        self.wgate = nn.Linear(config.dim, config.head_dim, bias=False) if ratio > 1 else None
        self.norm = RMSNorm(config.head_dim, config.norm_eps)

    def forward(self, x: Tensor) -> Tensor:
        count = x.shape[1] // self.ratio
        x = x[:, : count * self.ratio]
        if self.wgate is not None:
            with torch.autocast(x.device.type, enabled=False):
                values = F.linear(x.float(), self.wkv.weight.float())
                shape = (x.shape[0], count, self.ratio, values.shape[-1])
                weights = F.linear(x.float(), self.wgate.weight.float()).reshape(shape).softmax(-2)
                values = (values.reshape(shape) * weights).sum(-2).to(x.dtype)
        else:
            values = self.wkv(x)
        return self.norm(values)


class Indexer(nn.Module):
    def __init__(self, config: Any, *, owns_key: bool):
        super().__init__()
        self.heads = config.index_n_heads
        self.dim = config.index_head_dim
        self.wq_b = nn.Linear(config.q_lora_rank, self.heads * self.dim, bias=False)
        self.weights_proj = nn.Linear(config.dim, self.heads, bias=False)
        self.wk = nn.Linear(config.head_dim, self.dim, bias=False) if owns_key else None
        self.k_norm = RMSNorm(self.dim, config.norm_eps) if owns_key else None
        self.rope_dim = config.rope_head_dim
        self.theta = config.compress_rope_theta

    def keys(self, latent: Tensor, positions: Tensor) -> Tensor:
        assert self.wk is not None and self.k_norm is not None
        return rotary(self.k_norm(self.wk(latent.detach())), positions, self.rope_dim, self.theta)

    def forward(self, x: Tensor, qr: Tensor, key: Tensor, positions: Tensor) -> Tensor:
        q = self.wq_b(qr.detach()).unflatten(-1, (self.heads, self.dim))
        q = rotary(q, positions, self.rope_dim, self.theta)
        weights = self.weights_proj(x.detach()).float() * (self.heads * self.dim) ** -0.5
        with torch.autocast(x.device.type, enabled=False):
            return (
                torch.einsum("bthd,bcd->bthc", q.float(), key.float()).relu() * weights[..., None]
            ).sum(-2)


def candidate_mask(scores: Tensor, visible: Tensor, block_size: int, topk_blocks: int) -> Tensor:
    """Official block-max selection, pinning the newest causally reachable block."""
    count = scores.shape[-1]
    padded = F.pad(
        scores.detach().masked_fill(~visible, -torch.inf),
        (0, -count % block_size),
        value=-torch.inf,
    )
    blocks = padded.unflatten(-1, (-1, block_size)).amax(-1)
    # A partially filled latest block must remain eligible as new tokens arrive.
    last = visible.long().sum(-1).sub(1).clamp_min(0) // block_size
    blocks = blocks.scatter(-1, last[..., None], torch.inf)
    selected = blocks.argsort(dim=-1, descending=True, stable=True)[
        ..., : min(topk_blocks, blocks.shape[-1])
    ]
    mask = torch.zeros_like(blocks, dtype=torch.bool).scatter(-1, selected, True)
    return mask.repeat_interleave(block_size, -1)[..., :count] & visible


class CSA2Attention(nn.Module):
    def __init__(self, config: Any, layer_id: int):
        super().__init__()
        self.config, self.layer_id = config, layer_id
        self.ratio = config.compress_ratios[layer_id]
        self.is_full = layer_id in config.kv_source_layers
        self.mode = (
            "swa"
            if not self.ratio
            else "full"
            if self.is_full
            else "reindex"
            if layer_id in config.index_source_layers
            else "reuse"
        )
        self.wq_a = nn.Linear(config.dim, config.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(config.q_lora_rank, config.norm_eps)
        self.wq_b = nn.Linear(config.q_lora_rank, config.n_heads * config.head_dim, bias=False)
        self.wkv = nn.Linear(config.dim, config.head_dim, bias=False)
        self.kv_norm = RMSNorm(config.head_dim, config.norm_eps)
        self.attn_sink = nn.Parameter(torch.zeros(config.n_heads))
        self.wo_a = nn.Linear(
            config.n_heads * config.head_dim // config.o_groups,
            config.o_groups * config.o_lora_rank,
            bias=False,
        )
        self.wo_b = nn.Linear(config.o_groups * config.o_lora_rank, config.dim, bias=False)
        self.compressor = Compressor(config, self.ratio) if self.is_full else None
        self.indexer = (
            Indexer(config, owns_key=self.is_full)
            if layer_id in config.index_source_layers
            else None
        )
        self.indexer_loss_enabled = True

    def forward(
        self, x: Tensor, state: CSA2State, valid_mask: Tensor, encoder_hidden: Tensor | None = None
    ) -> tuple[Tensor, CSA2State, Tensor]:
        c = self.config
        batch, length, _ = x.shape
        positions = torch.arange(length, device=x.device)
        theta = c.compress_rope_theta if self.ratio else c.rope_theta
        qr = self.q_norm(self.wq_a(x))
        q = rotary(
            self.wq_b(qr).unflatten(-1, (c.n_heads, c.head_dim)), positions, c.rope_head_dim, theta
        )
        local = rotary(self.kv_norm(self.wkv(x)), positions, c.rope_head_dim, theta)
        if self.is_full:
            assert self.compressor is not None and self.indexer is not None
            source = encoder_hidden if self.layer_id >= c.n_encoder_layers else x
            if source is None:
                raise ValueError("decoder Full attention requires final encoder hidden states")
            latent = self.compressor(source)
            count = latent.shape[1]
            keys_pos = torch.arange(count, device=x.device) * self.ratio
            key_valid = (
                valid_mask[:, : count * self.ratio].reshape(batch, count, self.ratio).all(-1)
            )
            state = CSA2State(
                kv=rotary(latent, keys_pos, c.rope_head_dim, theta),
                index_key=self.indexer.keys(latent, keys_pos),
                valid=key_valid,
                ratio=self.ratio,
                owner=self.layer_id,
            )
        if self.ratio and (state.kv is None or state.ratio != self.ratio):
            raise ValueError("CSA2 consumer has no compatible preceding Full source")
        output, selections, candidates = [], [], []
        kl_sum = x.new_zeros((), dtype=torch.float32)
        window_offsets = torch.arange(c.window_size - 1, -1, -1, device=x.device)
        batch_ids = torch.arange(batch, device=x.device)[:, None, None]
        for start in range(0, length, c.attention_chunk_size):
            stop = min(start + c.attention_chunk_size, length)
            pos = positions[start:stop]
            wi = pos[:, None] - window_offsets
            wi = wi[None].expand(batch, -1, -1)
            window_valid = (wi >= 0) & valid_mask.gather(1, wi.clamp_min(0).flatten(1)).unflatten(
                1, wi.shape[1:]
            )
            values = local[batch_ids, wi.clamp_min(0)]
            mask = window_valid
            scores = support = None
            selected_count = 0
            if self.ratio and state.kv is not None and state.kv.shape[1]:
                count = state.kv.shape[1]
                assert state.valid is not None and state.index_key is not None
                visible = (
                    torch.arange(count, device=x.device)[None, None]
                    < (pos[None, :, None] + 1) // self.ratio
                ) & state.valid[:, None]
                if self.indexer is not None:
                    scores = self.indexer(x[:, start:stop], qr[:, start:stop], state.index_key, pos)
                    if c.hierarchical_indexing and self.layer_id == c.index_candidate_source_layer:
                        candidates.append(
                            candidate_mask(
                                scores,
                                visible,
                                c.index_candidate_block_size,
                                c.index_candidate_topk_blocks,
                            )
                        )
                    if c.hierarchical_indexing and self.layer_id > c.index_candidate_source_layer:
                        if state.candidates is None:
                            raise ValueError("hierarchical Reindex has no decoder candidate pool")
                        visible = visible & state.candidates[:, start:stop]
                    ranked = scores.detach().masked_fill(~visible, -torch.inf)
                    selected = ranked.argsort(dim=-1, descending=True, stable=True)[
                        ..., : min(c.index_topk, count)
                    ]
                    selection_valid = visible.gather(-1, selected)
                    selections.append(selected.masked_fill(~selection_valid, -1))
                else:
                    if state.indices is None:
                        raise ValueError("Reuse attention has no preceding sparse selection")
                    selected = state.indices[:, start:stop]
                    selection_valid = selected >= 0
                    selected = selected.clamp_min(0)
                selected_count = selected.shape[-1]
                values = torch.cat((values, state.kv[batch_ids, selected]), dim=-2)
                mask = torch.cat((mask, selection_valid), -1)
                if scores is not None:
                    scores = scores.gather(-1, selected)
                    support = selection_valid
            with torch.autocast(x.device.type, enabled=False):
                logits = (
                    torch.einsum("bthd,btkd->bthk", q[:, start:stop].float(), values.float())
                    * c.head_dim**-0.5
                )
            logits = logits.masked_fill(~mask[:, :, None], -torch.inf)
            sink = (
                self.attn_sink.float().view(1, 1, c.n_heads, 1).expand(batch, stop - start, -1, -1)
            )
            probs = torch.cat((logits, sink), -1).softmax(-1)[..., :-1]
            output.append(torch.einsum("bthk,btkd->bthd", probs.to(values.dtype), values))
            if scores is not None and support is not None and self.indexer_loss_enabled:
                teacher = probs[..., -selected_count:].detach().sum(-2) * support
                teacher = teacher / teacher.sum(-1, keepdim=True).clamp_min(1e-20)
                logp = scores.masked_fill(~support, torch.finfo(scores.dtype).min).log_softmax(-1)
                kl = (teacher * (teacher.clamp_min(1e-20).log() - logp)).sum(-1)
                kl_sum = kl_sum + (kl * valid_mask[:, start:stop]).sum()
        if selections:
            state = replace(state, indices=torch.cat(selections, 1))
        if candidates:
            state = replace(state, candidates=torch.cat(candidates, 1))
        result = rotary(torch.cat(output, 1), positions, c.rope_head_dim, theta, inverse=True)
        result = result.reshape(batch, length, c.o_groups, -1)
        result = torch.einsum(
            "btgd,grd->btgr", result, self.wo_a.weight.view(c.o_groups, c.o_lora_rank, -1)
        )
        return self.wo_b(result.flatten(2)), state, kl_sum / valid_mask.sum().clamp_min(1)
