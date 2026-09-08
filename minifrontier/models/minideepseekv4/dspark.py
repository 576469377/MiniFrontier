"""Differentiable mini adapter for Vision-Exp DSpark's three-stage/block5 path.

Source: DeepSeek-V4-Flash-Vision-Exp 6821d6a, MIT; license in upstream_layers.py.

Uses the pinned source mHC/MoE blocks and the source prefix-KV + bidirectional
noise-block attention relation. The three stages are independently copied from
the trained text MTP as an explicit local initialization; they share no trainable
weights with the frozen target. This module does not claim a measured speedup.
"""

import copy
import weakref
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import nn

from minifrontier.training.distributions import action_logits, forbidden_actions
from minifrontier.training.draft_losses import dspark_loss

from .kernels import rotary


class DSparkAttention(nn.Module):
    def __init__(self, attention):
        super().__init__()
        if attention.compress_ratio != 0:
            raise ValueError("DSpark stages use SWA, not the CSA/HCA indexer")
        self.base = attention

    def forward(self, x, main_x):
        a = self.base
        batch, block_size, _ = x.shape
        prefix = main_x.shape[1]
        if prefix + block_size > a.freqs_cis.shape[0]:
            raise ValueError("DSpark positions exceed the target context")
        rd = a.rope_head_dim
        current_freqs = a.freqs_cis[prefix : prefix + block_size]
        begin = max(0, prefix - a.window_size)
        main_freqs = a.freqs_cis[begin:prefix]
        q = a.wq_b(a.q_norm(a.wq_a(x))).unflatten(-1, (a.n_heads, a.head_dim))
        q = (q.float() * torch.rsqrt(q.float().square().mean(-1, keepdim=True) + a.eps)).to(q.dtype)
        q = torch.cat((q[..., :-rd], rotary(q[..., -rd:], current_freqs)), -1)
        main_kv, draft_kv = a.kv_norm(a.wkv(main_x[:, begin:])), a.kv_norm(a.wkv(x))
        main_kv = torch.cat((main_kv[..., :-rd], rotary(main_kv[..., -rd:], main_freqs)), -1)
        draft_kv = torch.cat((draft_kv[..., :-rd], rotary(draft_kv[..., -rd:], current_freqs)), -1)
        kv = torch.cat((main_kv, draft_kv), 1)
        scores = torch.einsum("bthd,bcd->bhtc", q.float(), kv.float()) * a.head_dim**-0.5
        sink = a.attn_sink.view(1, -1, 1, 1).expand(batch, -1, block_size, 1)
        probability = torch.cat((scores, sink), -1).softmax(-1)[..., :-1]
        value = torch.einsum("bhtc,bcd->bthd", probability.to(kv.dtype), kv)
        value = torch.cat(
            (value[..., :-rd], rotary(value[..., -rd:], current_freqs, inverse=True)), -1
        )
        value = value.reshape(batch, block_size, a.n_groups, -1)
        weight = a.wo_a.weight.view(a.n_groups, a.o_lora_rank, -1)
        return a.wo_b(torch.einsum("btgd,grd->btgr", value, weight).flatten(2))


class DSparkDraft(nn.Module):
    def __init__(self, target, *, block_size=5, markov_rank=64, noise_token_id=20):
        super().__init__()
        if target.mtp is None or target.config.n_layers < 3 or block_size != 5:
            raise ValueError("mini DSpark requires trained text MTP, three taps and block_size=5")
        if not 0 <= noise_token_id < target.config.vocab_size or markov_rank < 1:
            raise ValueError("invalid DSpark noise token or Markov rank")
        target.eval().requires_grad_(False)
        self._target = weakref.ref(target)
        self.block_size, self.noise_token_id = block_size, noise_token_id
        self.stages = nn.ModuleList([copy.deepcopy(target.mtp.block) for _ in range(3)])
        for raw_stage in self.stages:
            stage = cast(Any, raw_stage)
            stage.attn = DSparkAttention(stage.attn)
        self.main_proj = nn.Linear(3 * target.config.dim, target.config.dim, bias=False)
        self.main_norm, self.norm = copy.deepcopy(target.norm), copy.deepcopy(target.mtp.norm)
        self.markov_w1 = nn.Embedding(target.config.vocab_size, markov_rank)
        self.markov_w2 = nn.Linear(markov_rank, target.config.vocab_size, bias=False)
        self.confidence = nn.Linear(target.config.dim + markov_rank, 1)
        self.hc_head_fn = nn.Parameter(target.mtp.hc_head_fn.detach().clone())
        self.hc_head_scale = nn.Parameter(target.mtp.hc_head_scale.detach().clone())
        self.hc_head_base = nn.Parameter(target.mtp.hc_head_base.detach().clone())
        self.to(target.head.weight.device).requires_grad_(True)
        for name, parameter in self.named_parameters():
            if name.endswith(("gate.bias", "gate.bias_vl")):
                parameter.requires_grad_(False)
        nn.init.normal_(self.markov_w1.weight, std=0.02)
        nn.init.zeros_(self.markov_w2.weight)
        nn.init.zeros_(self.confidence.weight)
        nn.init.zeros_(self.confidence.bias)
        self.qat_scheme = target.config.qat_scheme
        self.train()

    def target(self):
        target = self._target()
        if target is None or target.training or any(p.requires_grad for p in target.parameters()):
            raise ValueError("DSpark target must stay frozen and in eval mode")
        return target

    def forward_features(self, taps, anchor_ids):
        target = self.target()
        main_x = self.main_norm(self.main_proj(torch.cat(taps, -1)))
        ids = anchor_ids.new_full((anchor_ids.shape[0], self.block_size), self.noise_token_id)
        ids[:, :1] = anchor_ids
        h = target.embed(ids).unsqueeze(2).expand(-1, -1, target.config.hc_mult, -1)
        for raw_stage in self.stages:
            stage = cast(Any, raw_stage)
            residual = h
            x, post, comb = stage.hc_pre(
                h, stage.hc_attn_fn, stage.hc_attn_scale, stage.hc_attn_base
            )
            x = stage.attn(stage.attn_norm(x), main_x)
            h = stage.hc_post(x, residual, post, comb)
            residual = h
            x, post, comb = stage.hc_pre(h, stage.hc_ffn_fn, stage.hc_ffn_scale, stage.hc_ffn_base)
            stage.ffn.gate.valid_mask = torch.ones_like(ids, dtype=torch.bool)
            stage.ffn.gate.sequence_balance_enabled = False
            x = stage.ffn(stage.ffn_norm(x), ids)
            h = stage.hc_post(x, residual, post, comb)
        return target.head.hc_head(h, self.hc_head_fn, self.hc_head_scale, self.hc_head_base)

    def forward(self, taps, anchor_ids, previous_tokens, *, vocab_size=None):
        if previous_tokens.shape != (anchor_ids.shape[0], self.block_size) or not torch.equal(
            previous_tokens[:, :1], anchor_ids
        ):
            raise ValueError("Markov previous tokens must begin with the causal anchor")
        target = self.target()
        hidden = self.forward_features(taps, anchor_ids)
        base = target.head.get_logits(self.norm(hidden))
        markov = self.markov_w1(previous_tokens)
        logits = action_logits(
            base + self.markov_w2(markov),
            vocab_size or target.config.vocab_size,
            forbidden_actions(target),
        )
        # Confidence is computed from pre-output-norm hidden, as in Vision-Exp.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            confidence = F.linear(
                torch.cat((hidden, markov), -1).float(),
                self.confidence.weight,
                self.confidence.bias,
            ).squeeze(-1)
        return logits, confidence, base

    def training_block(self, prefix, anchor_ids, target_tokens, *, media=None, vocab_size=None):
        target = self.target()
        if target_tokens.shape != (prefix.shape[0], self.block_size):
            raise ValueError("DSpark target block must have five positions")
        previous = torch.cat((anchor_ids, target_tokens[:, :-1]), 1)
        extras = dict(media=media) if media else {}
        with torch.no_grad():
            taps = target(
                prefix, return_taps=True, return_logits=False, **extras
            ).tapped_hidden_states
            trajectory = torch.cat((prefix, previous), 1)
            teacher = target(trajectory, **extras).logits[:, prefix.shape[1] :]
            teacher = action_logits(
                teacher, vocab_size or target.config.vocab_size, forbidden_actions(target)
            )
        logits, confidence, _ = self(taps, anchor_ids, previous, vocab_size=vocab_size)
        eos = target_tokens.eq(target.config.eos_token_id)
        valid = torch.cat((torch.ones_like(eos[:, :1]), eos[:, :-1].cumsum(-1).eq(0)), 1)
        valid &= anchor_ids.ne(target.config.eos_token_id)
        return dspark_loss(teacher, logits, target_tokens, confidence, valid)

    @torch.no_grad()
    def propose(self, context, *, steps=5, vocab_size=None):
        from minifrontier.speculative import Proposal, probability

        if not 0 <= steps <= self.block_size:
            raise ValueError("DSpark supports 0-5 extra proposals")
        target = self.target()
        token = torch.multinomial(context.next_p, 1)
        tokens, probabilities = [token], [context.next_p]
        # Evaluate all five noise positions once, then only the causal Markov head.
        if (
            steps
            and int(token) != target.config.eos_token_id
            and context.ids.shape[1] + self.block_size <= target.config.max_seq_len
        ):
            hidden = self.forward_features(context.taps, token)
            base = target.head.get_logits(self.norm(hidden))
            for index in range(steps):
                logits = base[:, index] + self.markov_w2(self.markov_w1(token[:, 0]))
                q = probability(logits, target, vocab_size)
                token = torch.multinomial(q, 1)
                tokens.append(token)
                probabilities.append(q)
                if int(token) == target.config.eos_token_id:
                    break
        return Proposal(torch.cat(tokens, 1), torch.stack(probabilities, 1), anchor_count=1)
