"""Local K3 mini EAGLE-style conversion, with seven self-fed LK steps.

The copied MTP MLA/MoE/AttnRes block uses draft-local residuals after conversion;
the target's three completed block outputs enter only through the [0,0,I] fusion.
Future positions use the draft's own predicted features, never future target taps.
This explicit mini conversion needs trained-target acceptance/latency validation.
"""

import copy
import weakref

import torch
from torch import nn

from minifrontier.training.distributions import action_logits, forbidden_actions
from minifrontier.training.draft_losses import kimi_lk


class KimiDraft(nn.Module):
    def __init__(self, target):
        super().__init__()
        if (
            target.mtp is None
            or target.config.num_hidden_layers // target.config.attn_res_block_size != 3
        ):
            raise ValueError(
                "Kimi draft needs the trained MTP and exactly three mini AttnRes blocks"
            )
        target.eval().requires_grad_(False)
        self._target = weakref.ref(target)
        self.mtp = copy.deepcopy(target.mtp).requires_grad_(True).train()
        for name, parameter in self.mtp.named_parameters():
            if name.endswith("e_score_correction_bias"):
                parameter.requires_grad_(False)
        width = target.config.hidden_size
        self.tap_fusion = nn.Linear(3 * width, width, bias=False).to(target.lm_head.weight.device)
        with torch.no_grad():
            self.tap_fusion.weight.zero_()
            self.tap_fusion.weight[:, 2 * width :].copy_(
                torch.eye(width, device=self.tap_fusion.weight.device)
            )
        self.qat_scheme = target.config.qat_scheme

    def target(self):
        target = self._target()
        if target is None or target.training or any(p.requires_grad for p in target.parameters()):
            raise ValueError("draft target must remain resident, frozen and in eval mode")
        return target

    def forward(self, hidden_history, embedding_history, valid=None):
        batch, length, width = hidden_history.shape
        residuals = hidden_history.new_zeros(batch * length, 0, width)
        if valid is None:
            valid = torch.ones((batch, length), dtype=torch.bool, device=hidden_history.device)
        return self.mtp(hidden_history, embedding_history, residuals, valid)

    def embedding_history(self, ids, anchor):
        target = self.target()
        shifted = torch.cat((ids[:, 1:], anchor), 1)
        valid = ids.ne(target.config.image_token_id) & shifted.ne(target.config.image_token_id)
        # Visual content is carried by the target taps, never raw placeholder embeddings.
        return target.embed_tokens(shifted).masked_fill(~valid[..., None], 0), valid

    @torch.no_grad()
    def propose(self, context, *, steps=3, vocab_size=None):
        from minifrontier.speculative import Proposal, probability

        if not 0 <= steps <= 7:
            raise ValueError("Kimi draft supports 0-7 extra proposals")
        target = self.target()
        token = torch.multinomial(context.next_p, 1)
        tokens, probabilities = [token], [context.next_p]
        features = self.tap_fusion(torch.cat(context.taps, -1))
        embeddings, valid = self.embedding_history(context.ids, token)
        for _ in range(steps):
            if int(token) == target.config.eos_token_id:
                break
            predicted = self(features, embeddings, valid)
            q = probability(target.lm_head(predicted[:, -1]), target, vocab_size)
            token = torch.multinomial(q, 1)
            tokens.append(token)
            probabilities.append(q)
            features = torch.cat((features, predicted[:, -1:]), 1)
            embeddings = torch.cat((embeddings, target.embed_tokens(token)), 1)
            valid = torch.cat((valid, torch.ones_like(token, dtype=torch.bool)), 1)
        return Proposal(torch.cat(tokens, 1), torch.stack(probabilities, 1), anchor_count=1)

    def unroll(self, prefix_ids, *, media=None, anchor_ids=None, steps=7, vocab_size=None):
        if not 1 <= steps <= 7 or prefix_ids.ndim != 2 or prefix_ids.eq(0).any():
            raise ValueError("draft unroll needs an unpadded prefix and 1-7 steps")
        target = self.target()
        if prefix_ids.shape[1] + steps + 1 > target.config.max_position_embeddings:
            raise ValueError("draft unroll would exceed target context")
        vocab_size = vocab_size or target.config.vocab_size
        forbidden = forbidden_actions(target)
        extras = dict(media=media) if media else {}
        with torch.no_grad():
            initial = target(prefix_ids, return_taps=True, **extras)
            anchor_p = action_logits(initial.logits[:, -1], vocab_size, forbidden).softmax(-1)
            if anchor_ids is None:
                anchor_ids = torch.multinomial(anchor_p, 1)
            if anchor_ids.shape != (prefix_ids.shape[0], 1):
                raise ValueError("draft anchor shape mismatch")
            embeddings, valid = self.embedding_history(prefix_ids, anchor_ids)
        features = self.tap_fusion(torch.cat(initial.tapped_hidden_states, -1))
        current = torch.cat((prefix_ids, anchor_ids), 1)
        active = anchor_ids[:, 0].ne(target.config.eos_token_id)
        total = self.tap_fusion.weight.sum() * 0
        count, losses, tokens = 0, [], []
        for _ in range(steps):
            if not active.any():
                break
            predicted = self(features, embeddings, valid)
            draft_logits = action_logits(target.lm_head(predicted[:, -1]), vocab_size, forbidden)
            with torch.no_grad():
                teacher = action_logits(
                    target(current, **extras).logits[:, -1], vocab_size, forbidden
                )
            loss = kimi_lk(teacher, draft_logits)
            total = total + (loss * active).sum()
            count += int(active.sum())
            losses.append(loss.detach())
            sampled = torch.multinomial(draft_logits.detach().softmax(-1), 1)
            tokens.append(sampled)
            active = active & sampled[:, 0].ne(target.config.eos_token_id)
            current = torch.cat((current, sampled), 1)
            # Retain the differentiable self-fed feature path for all seven steps.
            features = torch.cat((features, predicted[:, -1:]), 1)
            embeddings = torch.cat((embeddings, target.embed_tokens(sampled)), 1)
            valid = torch.cat((valid, active[:, None]), 1)
        return total / max(1, count), dict(
            positions=count, steps=len(losses), sampled_tokens=tokens, lk=losses
        )
