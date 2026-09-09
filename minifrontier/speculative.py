"""Temperature-one speculative sampling with transactional native target caches.

The correctness path keeps accepted target features and replays after rejection.
Drafts may recompute their own short block; no speedup is implied by this module.
Batch one and complete visual prefill are deliberate initial constraints.
"""

from dataclasses import dataclass
from typing import Any

import torch

from minifrontier.training.distributions import action_logits, forbidden_actions


def probability(logits, model, vocab_size=None):
    return action_logits(logits, vocab_size, forbidden_actions(model)).softmax(-1)


def residual_probability(p, q):
    """The exact rejection branch; zero overlap residual cannot be rejected."""
    value = (p - q).clamp_min(0)
    total = value.sum(-1, keepdim=True)
    if not torch.isfinite(total).all() or (total <= 0).any():
        raise ValueError("rejection has no positive target-minus-draft mass")
    return value / total


def accepts(p, q, token, uniform):
    denominator = q.gather(-1, token)
    if (denominator <= 0).any():
        raise ValueError("proposed token has zero draft probability")
    return bool((uniform < (p.gather(-1, token) / denominator).clamp(max=1)).all())


@dataclass
class Proposal:
    ids: torch.Tensor
    probabilities: torch.Tensor
    anchor_count: int = 0

    def validate(self, context, maximum):
        if self.anchor_count not in (0, 1):
            raise ValueError("proposal may include at most one target-sampled anchor")
        if self.ids.ndim == 2 and self.ids.shape[1] - self.anchor_count > 7:
            raise ValueError("initial sampler supports at most seven draft positions")
        if self.ids.ndim != 2 or self.ids.shape[0] != 1 or not 1 <= self.ids.shape[1] <= maximum:
            raise ValueError("draft must return a nonempty batch-one bounded proposal")
        if self.probabilities.shape != (*self.ids.shape, context.next_p.shape[-1]):
            raise ValueError("draft probability shape mismatch")
        if self.ids.device != context.ids.device or self.probabilities.device != context.ids.device:
            raise ValueError("draft and target device mismatch")
        q = self.probabilities
        if not torch.isfinite(q).all() or (q < 0).any():
            raise ValueError("draft distribution must be finite and nonnegative")
        if not torch.allclose(q.sum(-1), torch.ones_like(q[..., 0]), rtol=1e-5, atol=1e-6):
            raise ValueError("draft distribution is not normalized")
        if self.ids.dtype != torch.long or self.ids.min() < 0 or self.ids.max() >= q.shape[-1]:
            raise ValueError("invalid proposed action IDs")
        if (q.gather(-1, self.ids[..., None]) <= 0).any():
            raise ValueError("proposed token has zero probability")
        if self.anchor_count and not torch.equal(q[:, 0], context.next_p):
            raise ValueError("anchor must use the exact target distribution")


@dataclass
class DraftContext:
    ids: torch.Tensor
    next_p: torch.Tensor
    taps: tuple[torch.Tensor, ...] | None
    multistream: torch.Tensor | None
    media: Any = None

    def append(self, ids, output, model, vocab_size):
        taps = getattr(output, "tapped_hidden_states", None)
        if self.taps is not None:
            if taps is None:
                raise ValueError("target stopped returning the required draft taps")
            taps = tuple(torch.cat((old, new), 1) for old, new in zip(self.taps, taps, strict=True))
        multi = output.multistream_hidden
        if self.multistream is not None:
            multi = torch.cat((self.multistream, multi), 1)
        return DraftContext(
            torch.cat((self.ids, ids), 1),
            probability(output.logits[:, -1], model, vocab_size),
            taps,
            multi,
            self.media,
        )


class SpeculativeSession:
    def __init__(self, model, ids, *, media=None, vocab_size=None):
        if (
            model.training
            or ids.ndim != 2
            or ids.shape[0] != 1
            or ids.shape[1] == 0
            or ids.eq(0).any()
        ):
            raise ValueError("speculative inference requires eval and an unpadded batch-one prefix")
        from minifrontier.models.minideepseekv4 import MiniDeepSeekV4Cache
        from minifrontier.models.minifrontier1 import MiniFrontier1Cache
        from minifrontier.models.minikimik3 import MiniKimiK3Cache
        from minifrontier.models.miniqwen4 import MiniQwen4Cache

        caches = {
            "minifrontier1": MiniFrontier1Cache,
            "minikimik3": MiniKimiK3Cache,
            "miniqwen4": MiniQwen4Cache,
            "minideepseekv4": MiniDeepSeekV4Cache,
        }
        self.family = {
            "MiniFrontier1ForCausalLM": "minifrontier1",
            "MiniKimiK3ForCausalLM": "minikimik3",
            "MiniQwen4ForCausalLM": "miniqwen4",
            "MiniDeepSeekV4ForCausalLM": "minideepseekv4",
        }[type(model).__name__]
        self.cache: Any = caches[self.family]()
        self.model, self.vocab_size = model, vocab_size
        output = self.forward(ids, media=media)
        self.context = DraftContext(
            ids,
            probability(output.logits[:, -1], model, vocab_size),
            getattr(output, "tapped_hidden_states", None),
            output.multistream_hidden,
            media,
        )
        self.stats: dict[str, Any] = dict(
            proposed=0,
            accepted=0,
            rejections=0,
            target_forwards=1,
            replay_tokens=0,
            anchor_tokens=0,
            draft_proposed=0,
            draft_accepted=0,
            draft_position_attempts=[0] * 7,
            draft_position_accepts=[0] * 7,
        )

    @torch.no_grad()
    def forward(self, ids, *, media=None):
        kwargs = dict(cache=self.cache, media=media, return_hidden=True)
        if self.family not in {"miniqwen4", "minifrontier1"}:
            kwargs["return_taps"] = True
        return self.model(ids, **kwargs)

    @torch.no_grad()
    def advance(self, proposer, maximum, *, greedy=False):
        if maximum < 1:
            raise ValueError("remaining token budget must be positive")
        proposal = proposer(self.context, maximum)
        proposal.validate(self.context, maximum)
        ids, q = proposal.ids, proposal.probabilities
        snapshot = self.cache.snapshot()
        original = self.context
        from minifrontier.models.cache_utils import copy_state

        prior_stats = copy_state(self.stats)
        try:
            verified = self.forward(ids)
            p = torch.cat(
                (
                    original.next_p[:, None],
                    probability(verified.logits, self.model, self.vocab_size),
                ),
                1,
            )
            self.stats["target_forwards"] += 1
            self.stats["proposed"] += ids.shape[1]
            self.stats["anchor_tokens"] += proposal.anchor_count
            self.stats["draft_proposed"] += ids.shape[1] - proposal.anchor_count
            count, replacement = 0, None
            for index in range(ids.shape[1]):
                token = ids[:, index : index + 1]
                draft_index = index - proposal.anchor_count
                if draft_index >= 0:
                    self.stats["draft_position_attempts"][draft_index] += 1
                accepted_token = (
                    bool(token.eq(p[:, index].argmax(-1, keepdim=True)).all())
                    if greedy
                    else accepts(p[:, index], q[:, index], token, torch.rand((), device=ids.device))
                )
                if not accepted_token:
                    replacement = (
                        p[:, index].argmax(-1, keepdim=True)
                        if greedy
                        else torch.multinomial(residual_probability(p[:, index], q[:, index]), 1)
                    )
                    self.stats["rejections"] += 1
                    break
                count += 1
                if draft_index >= 0:
                    self.stats["draft_position_accepts"][draft_index] += 1
                    self.stats["draft_accepted"] += 1
                if int(token) == self.model.config.eos_token_id:
                    break
            self.stats["accepted"] += count
            accepted = ids[:, :count]
            ended = count > 0 and int(accepted[0, -1]) == self.model.config.eos_token_id
            if count == ids.shape[1]:
                self.context = original.append(ids, verified, self.model, self.vocab_size)
                if not ended and count < maximum:
                    replacement = (
                        p[:, -1].argmax(-1, keepdim=True)
                        if greedy
                        else torch.multinomial(p[:, -1], 1)
                    )
                    extra = self.forward(replacement)
                    self.stats["target_forwards"] += 1
                    self.context = self.context.append(
                        replacement, extra, self.model, self.vocab_size
                    )
                    accepted = torch.cat((accepted, replacement), 1)
            else:
                # This also discards speculative positions after an accepted EOS.
                self.cache.restore(snapshot)
                if replacement is not None:
                    accepted = torch.cat((accepted, replacement), 1)
                replay = self.forward(accepted)
                self.stats["target_forwards"] += 1
                self.stats["replay_tokens"] += accepted.shape[1]
                self.context = original.append(accepted, replay, self.model, self.vocab_size)
            return accepted
        except BaseException:
            self.cache.restore(snapshot)
            self.context = original
            self.stats = prior_stats
            raise


@torch.no_grad()
def generate_speculative(
    model, draft, ids, *, max_new_tokens=64, draft_steps=3, media=None, vocab_size=None
):
    if max_new_tokens < 1 or draft_steps < 1:
        raise ValueError("generation and draft budgets must be positive")
    limit = getattr(
        model.config, "max_position_embeddings", getattr(model.config, "max_seq_len", 0)
    )
    if ids.shape[1] + max_new_tokens > limit:
        raise ValueError("prompt and generation exceed trained context")
    if draft.target() is not model or draft.training:
        raise ValueError("the eval draft must belong to this exact frozen target")
    with torch.autocast(ids.device.type, dtype=torch.bfloat16, enabled=ids.is_cuda):
        session = SpeculativeSession(model, ids, media=media, vocab_size=vocab_size)
        remaining = max_new_tokens
        while remaining:
            emitted = session.advance(
                lambda context, maximum: draft.propose(
                    context, steps=min(draft_steps, maximum - 1), vocab_size=vocab_size
                ),
                remaining,
            )
            remaining -= emitted.shape[1]
            if int(emitted[0, -1]) == model.config.eos_token_id:
                break
        return session.context.ids, session.stats
