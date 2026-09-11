"""Training adaptation of V4's sliding-window + CSA/HCA attention.

Parameter names and projection/pooling equations follow inference/model.py at
60d8d70. Mutable inference caches and FP8/FP4 simulation are replaced by functional
full-sequence tensors. This preserves gradients through compressed KV and RoPE.
The indexer KL is a documented local training integration, not released training code.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .kernels import rotary
from .upstream_layers import Linear, RMSNorm, precompute_freqs_cis


def _dense_attention(q, kv, mask, sink, *, recompute=False, chunk_size=128):
    """Keep complete key support while bounding score/probability temporaries."""

    def attend(queries, keys, allowed, sink_logits):
        logits = torch.einsum("bthd,bcd->bhtc", queries.float(), keys.float())
        logits = (logits * queries.shape[-1] ** -0.5).masked_fill(~allowed[:, None], float("-inf"))
        sinks = sink_logits.view(1, -1, 1, 1).expand(queries.shape[0], -1, queries.shape[1], 1)
        probs = torch.cat((logits, sinks), dim=-1).softmax(-1)[..., :-1]
        return torch.einsum("bhtc,bcd->bthd", probs.to(keys.dtype), keys)

    chunks = []
    for start in range(0, q.shape[1], chunk_size):
        args = (q[:, start : start + chunk_size], kv, mask[:, start : start + chunk_size], sink)
        chunks.append(
            checkpoint(attend, *args, use_reentrant=False)
            if recompute and q.shape[1] > chunk_size
            else attend(*args)
        )
    return torch.cat(chunks, dim=1)


class Compressor(nn.Module):
    def __init__(self, args, ratio, head_dim):
        super().__init__()
        self.ratio, self.head_dim = ratio, head_dim
        self.rope_head_dim = args.rope_head_dim
        self.overlap = ratio == 4
        width = head_dim * (1 + self.overlap)
        self.ape = nn.Parameter(torch.zeros(ratio, width))
        self.wkv = Linear(args.dim, width)
        self.wgate = Linear(args.dim, width)
        self.norm = RMSNorm(head_dim, args.norm_eps)

    def forward(self, x, freqs):
        b, length, _ = x.shape
        count = length // self.ratio
        if not count:
            return x.new_empty(b, 0, self.head_dim)
        # Pinned compressor explicitly computes projections and gated pooling in FP32.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x[:, : count * self.ratio].float()
            kv = self.wkv(x).unflatten(1, (count, self.ratio))
            score = self.wgate(x).unflatten(1, (count, self.ratio)) + self.ape
            if self.overlap:
                previous_kv = F.pad(kv[:, :-1, :, : self.head_dim], (0, 0, 0, 0, 1, 0))
                previous_score = F.pad(
                    score[:, :-1, :, : self.head_dim], (0, 0, 0, 0, 1, 0), value=float("-inf")
                )
                kv = torch.cat((previous_kv, kv[..., self.head_dim :]), dim=2)
                score = torch.cat((previous_score, score[..., self.head_dim :]), dim=2)
            pooled = (kv * score.softmax(dim=2)).sum(dim=2)
        pooled = self.norm(pooled)
        rd = self.rope_head_dim
        return torch.cat(
            (
                pooled[..., :-rd],
                rotary(pooled[..., -rd:], freqs[: count * self.ratio : self.ratio]),
            ),
            dim=-1,
        )


class Indexer(nn.Module):
    def __init__(self, args, ratio):
        super().__init__()
        self.n_heads, self.head_dim = args.index_n_heads, args.index_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.wq_b = Linear(args.q_lora_rank, self.n_heads * self.head_dim)
        self.weights_proj = Linear(args.dim, self.n_heads)
        self.compressor = Compressor(args, ratio, self.head_dim)
        self.qat_enabled = False

    def forward(self, x, qr, freqs):
        q = self.wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim))
        rd = self.rope_head_dim
        q = torch.cat((q[..., :-rd], rotary(q[..., -rd:], freqs)), dim=-1)
        k = self.compressor(x, freqs)
        weights = self.weights_proj(x).float() * (self.head_dim * self.n_heads) ** -0.5
        if getattr(self, "qat_enabled", False):
            from minifrontier.training.deepseek_qat import index_scores

            return index_scores(q, k, weights)
        # The common Hadamard rotation cancels in unquantized dot products.
        scores = torch.einsum("bthd,bcd->bthc", q.float(), k.float()).relu()
        return (scores * weights.unsqueeze(-1)).sum(2)


class Attention(nn.Module):
    def __init__(self, layer_id, args):
        super().__init__()
        self.n_heads, self.head_dim = args.n_heads, args.head_dim
        self.rope_head_dim, self.n_groups = args.rope_head_dim, args.o_groups
        self.o_lora_rank = args.o_lora_rank
        self.eps, self.window_size = args.norm_eps, args.window_size
        self.compress_ratio = args.compress_ratios[layer_id]
        self.index_topk = args.index_topk
        self.training_phase = "dense_pretrain"
        self.indexer_loss_enabled = True
        self.attn_sink = nn.Parameter(torch.zeros(args.n_heads))
        self.wq_a, self.q_norm = (
            Linear(args.dim, args.q_lora_rank),
            RMSNorm(args.q_lora_rank, args.norm_eps),
        )
        self.wq_b = Linear(args.q_lora_rank, args.n_heads * args.head_dim)
        self.wkv, self.kv_norm = (
            Linear(args.dim, args.head_dim),
            RMSNorm(args.head_dim, args.norm_eps),
        )
        self.wo_a = Linear(
            args.n_heads * args.head_dim // args.o_groups, args.o_groups * args.o_lora_rank
        )
        self.wo_b = Linear(args.o_groups * args.o_lora_rank, args.dim)
        self.indexer = None
        if self.compress_ratio:
            self.compressor = Compressor(args, self.compress_ratio, args.head_dim)
            if self.compress_ratio == 4:
                self.indexer = Indexer(args, 4)
        freqs = precompute_freqs_cis(
            args.rope_head_dim,
            args.max_seq_len,
            0,
            args.compress_rope_theta if self.compress_ratio else args.rope_theta,
            1,
            32,
            1,
        )
        self.register_buffer("freqs_cis", freqs, persistent=False)
        self.indexer_loss = torch.tensor(0.0)
        self.query_valid = None
        self.image_visible = None
        self.decode_state = None

    def forward(self, x, start_pos=0):
        if self.decode_state is not None and self.decode_state.get("length", 0):
            from .incremental import decode

            return decode(self, x, self.decode_state)
        if start_pos:
            raise ValueError(
                "V4 training adapter accepts full sequences; use generate for inference"
            )
        b, length, _ = x.shape
        freqs = cast(torch.Tensor, self.freqs_cis)[:length]
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim))
        q = (q.float() * torch.rsqrt(q.float().square().mean(-1, keepdim=True) + self.eps)).to(
            q.dtype
        )
        kv = self.kv_norm(self.wkv(x))
        rd = self.rope_head_dim
        q = torch.cat((q[..., :-rd], rotary(q[..., -rd:], freqs)), dim=-1)
        kv = torch.cat((kv[..., :-rd], rotary(kv[..., -rd:], freqs)), dim=-1)
        pos = torch.arange(length, device=x.device)
        window = (pos[:, None] >= pos) & (pos[:, None] - pos < self.window_size)
        mask = window.unsqueeze(0).expand(b, -1, -1)
        if self.image_visible is not None:
            left, right = self.image_visible
            begin = (
                pos[None] - (self.window_size - 1) - (left - (self.window_size - 1)).clamp_min(0)
            ).clamp_min(0)
            mask = (pos[None, None, :] >= begin[..., None]) & (
                pos[None, None, :] <= (pos[None] + right)[..., None]
            )
        self.indexer_loss = x.new_zeros((), dtype=torch.float32)
        scores = None
        compressed = None
        raw_kv = kv
        valid = None
        if self.compress_ratio and length >= self.compress_ratio:
            compressed = self.compressor(x, freqs).to(kv.dtype)
            count = compressed.shape[1]
            valid = (
                torch.arange(count, device=x.device)[None, :]
                < (pos[:, None] + 1) // self.compress_ratio
            )
            selected = valid.unsqueeze(0).expand(b, -1, -1)
            if self.indexer is not None and self.training_phase != "dense_pretrain":
                scores = self.indexer(x.detach(), qr.detach(), freqs)
                if self.training_phase == "sparse_cpt":
                    # Canonical lower-index tie breaking keeps prefix/full and
                    # incremental inference identical when ReLU index scores tie.
                    indices = scores.masked_fill(~selected, float("-inf")).argsort(
                        dim=-1, descending=True, stable=True
                    )[..., : min(self.index_topk, count)]
                    selected = torch.zeros_like(selected).scatter(-1, indices, True) & selected
            mask = torch.cat((mask, selected), dim=-1)
            kv = torch.cat((kv, compressed), dim=1)
        if self.decode_state is not None:
            from .incremental import initialize_state

            initialize_state(self, self.decode_state, x, raw_kv, compressed)
        if self.training_phase == "dense_pretrain":
            out = _dense_attention(
                q, kv, mask, self.attn_sink, recompute=self.training and torch.is_grad_enabled()
            )
            return self._output(out, freqs, b, length)
        logits = torch.einsum("bthd,bcd->bhtc", q.float(), kv.float()) * self.head_dim**-0.5
        logits = logits.masked_fill(~mask[:, None], float("-inf"))
        sinks = self.attn_sink.view(1, -1, 1, 1).expand(b, -1, length, 1)
        probs = torch.cat((logits, sinks), dim=-1).softmax(-1)[..., :-1]
        if scores is not None and self.indexer_loss_enabled:
            support = mask[..., length:]
            target = probs[..., length:].detach().sum(dim=1) * support
            target = target / target.sum(-1, keepdim=True).clamp_min(1e-12)
            # Finite logits avoid NaN for early queries with no complete block.
            logp = scores.masked_fill(~support, torch.finfo(scores.dtype).min).log_softmax(-1)
            kl = target * (target.clamp_min(1e-12).log() - logp)
            query_valid = (
                self.query_valid
                if self.query_valid is not None
                else torch.ones_like(x[..., 0], dtype=torch.bool)
            )
            self.indexer_loss = (kl.sum(-1) * query_valid).sum() / query_valid.sum().clamp_min(1)
        out = torch.einsum("bhtc,bcd->bthd", probs.to(kv.dtype), kv)
        return self._output(out, freqs, b, length)

    def _output(self, out, freqs, b, length):
        rd = self.rope_head_dim
        out = torch.cat((out[..., :-rd], rotary(out[..., -rd:], freqs, inverse=True)), dim=-1)
        out = out.reshape(b, length, self.n_groups, -1)
        weight = self.wo_a.weight.view(self.n_groups, self.o_lora_rank, -1)
        return self.wo_b(torch.einsum("btgd,grd->btgr", out, weight).flatten(2))
