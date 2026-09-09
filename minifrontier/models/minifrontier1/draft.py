"""MTP-derived fixed-anchor draft; no unverified future target hidden states are read."""

import copy
import weakref

import torch
from torch import nn

from minifrontier.speculative import Proposal, probability
from minifrontier.training.deepseek_opd import reverse_kl_from_logits


class MF1Draft(nn.Module):
    def __init__(self, target):
        super().__init__()
        if target.mtp is None:
            raise ValueError("MF1 draft requires an MTP-bearing target")
        target.eval().requires_grad_(False)
        self._target = weakref.ref(target)
        self.mtp = copy.deepcopy(target.mtp).requires_grad_(True)

    def target(self):
        target = self._target()
        if target is None or target.training or any(p.requires_grad for p in target.parameters()):
            raise ValueError("draft target must stay resident, frozen and in eval mode")
        return target

    def logits(self, anchor, tokens, base):
        target = self.target()
        b, length = tokens.shape
        residual = anchor.expand(b, length, -1, -1)
        positions = torch.arange(base, base + length, device=tokens.device).expand(3, b, -1)
        metadata = dict(
            segment_ids=torch.zeros_like(tokens),
            modality=torch.zeros_like(tokens),
            media_ids=torch.full_like(tokens, -1),
            position_ids=positions,
        )
        features = self.mtp(residual, target.embed_tokens(tokens), metadata)
        return target.lm_head(features[:, -1])

    @torch.no_grad()
    def propose(self, context, *, steps=3, vocab_size=None, greedy=False):
        if not 0 <= steps <= 6:
            raise ValueError("MF1 draft supports at most six extra steps")
        token = (
            context.next_p.argmax(-1, keepdim=True)
            if greedy
            else torch.multinomial(context.next_p, 1)
        )
        tokens, probabilities = [token], [context.next_p]
        anchor = context.multistream[:, -1:]
        for _ in range(steps):
            if int(token) == self.target().config.eos_token_id:
                break
            q = probability(
                self.logits(anchor, torch.cat(tokens, 1), context.ids.shape[1] - 1),
                self.target(),
                vocab_size,
            )
            token = q.argmax(-1, keepdim=True) if greedy else torch.multinomial(q, 1)
            tokens.append(token)
            probabilities.append(q)
        return Proposal(torch.cat(tokens, 1), torch.stack(probabilities, 1), anchor_count=1)

    @torch.no_grad()
    def generate_greedy(self, ids, *, max_new_tokens=64, steps=4, media=None, vocab_size=None):
        from minifrontier.speculative import SpeculativeSession

        target = self.target()
        if self.training or not 1 <= steps <= 6 or max_new_tokens < 1:
            raise ValueError("greedy draft requires eval mode and positive bounded budgets")
        if ids.shape[1] + max_new_tokens > target.config.max_position_embeddings:
            raise ValueError("prompt and generation exceed trained context")
        with torch.autocast(ids.device.type, dtype=torch.bfloat16, enabled=ids.is_cuda):
            session = SpeculativeSession(target, ids, media=media, vocab_size=vocab_size)
            remaining = max_new_tokens
            while remaining:
                emitted = session.advance(
                    lambda context, maximum: self.propose(
                        context, steps=min(steps, maximum - 1), vocab_size=vocab_size, greedy=True
                    ),
                    remaining,
                    greedy=True,
                )
                remaining -= emitted.shape[1]
                if int(emitted[0, -1]) == target.config.eos_token_id:
                    break
        return session.context.ids, session.stats

    def unroll(self, prefix, *, steps=4, media=None, vocab_size=None):
        if not 2 <= steps <= 6:
            raise ValueError("rollout adaptation requires two to six self-fed steps")
        target = self.target()
        with torch.no_grad():
            initial = target(prefix, media=media, return_hidden=True)
            token = torch.multinomial(probability(initial.logits[:, -1], target, vocab_size), 1)
        anchor = initial.multistream_hidden[:, -1:].detach()
        tokens, current = token, torch.cat((prefix, token), 1)
        terms, count = [], 0
        for _ in range(steps):
            if int(tokens[0, -1]) == target.config.eos_token_id:
                break
            q_logits = self.logits(anchor, tokens, prefix.shape[1] - 1)
            with torch.no_grad():
                p_logits = target(current, media=media).logits[:, -1]
            term = reverse_kl_from_logits(
                q_logits,
                p_logits,
                vocab_size=vocab_size,
                forbidden_ids=target.config.forbidden_action_ids,
            ).mean()
            terms.append(term)
            count += len(prefix)
            token = torch.multinomial(probability(q_logits.detach(), target, vocab_size), 1)
            tokens = torch.cat((tokens, token), 1)
            current = torch.cat((current, token), 1)
        return sum(terms, self.mtp.hidden_proj.weight.sum() * 0) / max(1, len(terms)), dict(
            positions=count, rule="fixed verified target anchor + own sampled prefix"
        )
