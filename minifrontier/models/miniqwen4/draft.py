"""Target-frozen adaptation of the source-shaped four-stream MTP and its own head."""

import copy
import weakref

import torch
from torch import nn

from minifrontier.training.losses import chunked_linear_ce
from minifrontier.training.mtp import shifted_batch

from .processing import position_ids


class QwenDraft(nn.Module):
    def __init__(self, target):
        super().__init__()
        if target.mtp is None:
            raise ValueError("Qwen draft adaptation requires the trained multistream MTP")
        target.eval().requires_grad_(False)
        self._target = weakref.ref(target)
        self.mtp = copy.deepcopy(target.mtp).requires_grad_(True).train()
        self.mtp.block.self_attn.indexer.requires_grad_(False)

    def target(self):
        target = self._target()
        if target is None or target.training or any(p.requires_grad for p in target.parameters()):
            raise ValueError("Qwen draft target must stay resident, frozen and in eval mode")
        return target

    def forward(self, ids, labels, *, media=None):
        target = self.target()
        positions = position_ids(ids, media or [])
        valid_input = ids.ne(target.config.pad_token_id)
        with torch.no_grad():
            output = target(
                ids,
                attention_mask=valid_input,
                media=media,
                position_ids=positions,
                return_hidden=True,
                return_logits=False,
            )
            shifted, targets, valid = shifted_batch(
                ids,
                labels,
                vocab_size=target.config.vocab_size,
                image_token_id=target.config.image_token_id,
                attention_mask=valid_input,
            )
            embedding = target.model.embed_tokens(shifted)
        features, raw, _, _ = self.mtp(output.multistream_hidden, embedding, valid, positions + 1)
        loss = chunked_linear_ce(features, self.mtp.shared_head.head.weight, targets, shift=False)
        return loss, dict(positions=int(targets.ne(-100).sum()), multistream_hidden=raw)

    def histories(self, ids, anchor, media):
        target = self.target()
        shifted = torch.cat((ids[:, 1:], anchor), 1)
        valid = ids.ne(target.config.image_token_id) & shifted.ne(target.config.image_token_id)
        embeddings = target.model.embed_tokens(shifted).masked_fill(~valid[..., None], 0)
        return embeddings, valid, position_ids(ids, media or []) + 1

    @torch.no_grad()
    def propose(self, context, *, steps=3, vocab_size=None):
        from minifrontier.speculative import Proposal, probability

        if not 0 <= steps <= 3:
            raise ValueError("Qwen draft initially supports 0-3 extra proposals")
        target = self.target()
        token = torch.multinomial(context.next_p, 1)
        tokens, probabilities = [token], [context.next_p]
        embeddings, valid, positions = self.histories(context.ids, token, context.media)
        multi = context.multistream
        for _ in range(steps):
            if int(token) == target.config.eos_token_id:
                break
            features, raw, _, _ = self.mtp(multi, embeddings, valid, positions)
            q = probability(self.mtp.shared_head.head(features[:, -1]), target, vocab_size)
            token = torch.multinomial(q, 1)
            tokens.append(token)
            probabilities.append(q)
            multi = torch.cat((multi, raw[:, -1:]), 1)
            embeddings = torch.cat((embeddings, target.model.embed_tokens(token)), 1)
            valid = torch.cat((valid, torch.ones_like(token, dtype=torch.bool)), 1)
            positions = torch.cat((positions, positions[:, :, -1:] + 1), -1)
        return Proposal(torch.cat(tokens, 1), torch.stack(probabilities, 1), anchor_count=1)

    def unroll(self, prefix, *, media=None, steps=3, vocab_size=None):
        """CE on target-sampled next actions at the draft's own causal prefixes."""
        from minifrontier.speculative import probability
        from minifrontier.training.distributions import action_logits, forbidden_actions

        if not 1 <= steps <= 3 or prefix.ndim != 2 or prefix.eq(0).any():
            raise ValueError("Qwen short unroll needs an unpadded prefix and 1-3 steps")
        target = self.target()
        extras = dict(media=media) if media else {}
        with torch.no_grad():
            initial = target(prefix, return_hidden=True, **extras)
            anchor = torch.multinomial(probability(initial.logits[:, -1], target, vocab_size), 1)
        multi = initial.multistream_hidden
        embeddings, valid, positions = self.histories(prefix, anchor, media)
        active = anchor[:, 0].ne(target.config.eos_token_id)
        current = torch.cat((prefix, anchor), 1)
        total = self.mtp.fc_hidden.weight.sum() * 0
        count = 0
        for _ in range(steps):
            if not active.any():
                break
            features, raw, _, _ = self.mtp(multi, embeddings, valid, positions)
            q_logits = action_logits(
                self.mtp.shared_head.head(features[:, -1]), vocab_size, forbidden_actions(target)
            )
            q = q_logits.softmax(-1)
            with torch.no_grad():
                p = probability(target(current, **extras).logits[:, -1], target, vocab_size)
                teacher_action = torch.multinomial(p, 1)
            ce = -q_logits.log_softmax(-1).gather(-1, teacher_action)[:, 0]
            total = total + (ce * active).sum()
            count += int(active.sum())
            sampled = torch.multinomial(q.detach(), 1)
            active = active & sampled[:, 0].ne(target.config.eos_token_id)
            current = torch.cat((current, sampled), 1)
            multi = torch.cat((multi, raw[:, -1:]), 1)
            embeddings = torch.cat((embeddings, target.model.embed_tokens(sampled)), 1)
            valid = torch.cat((valid, active[:, None]), 1)
            positions = torch.cat((positions, positions[:, :, -1:] + 1), -1)
        return total / max(1, count), dict(positions=count)
