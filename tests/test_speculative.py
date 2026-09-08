"""Distribution identity and rejection at every step against full native targets."""

from unittest.mock import patch

import pytest
import torch
from test_miniqwen4 import tiny_config
from test_new_backbones import tiny_deepseek, tiny_kimi

from minifrontier.models.minideepseekv4 import MiniDeepSeekV4ForCausalLM
from minifrontier.models.minideepseekv4.dspark import DSparkDraft
from minifrontier.models.minikimik3 import MiniKimiK3ForCausalLM
from minifrontier.models.minikimik3.draft import KimiDraft
from minifrontier.models.miniqwen4 import MiniQwen4ForCausalLM
from minifrontier.models.miniqwen4.draft import QwenDraft
from minifrontier.speculative import (
    Proposal,
    SpeculativeSession,
    accepts,
    generate_speculative,
    probability,
    residual_probability,
)


def make_target(family):
    if family == "kimi":
        return MiniKimiK3ForCausalLM(tiny_kimi(num_hidden_layers=12, mtp_enabled=True)).eval()
    if family == "qwen":
        return MiniQwen4ForCausalLM(tiny_config(vocab_size=64, mtp_enabled=True)).eval()
    return MiniDeepSeekV4ForCausalLM(tiny_deepseek(mtp_enabled=True)).eval()


def test_rejection_sampling_exactly_preserves_target_mass():
    p = torch.tensor([[0.1, 0.2, 0.7]], dtype=torch.float64)
    q = torch.tensor([[0.5, 0.3, 0.2]], dtype=torch.float64)
    kept = torch.minimum(p, q)
    mass = kept + (1 - kept.sum()) * residual_probability(p, q)
    torch.testing.assert_close(mass, p, rtol=0, atol=1e-15)
    assert accepts(p, q, torch.tensor([[0]]), 0.19)
    assert not accepts(p, q, torch.tensor([[0]]), 0.21)
    with pytest.raises(ValueError, match="no positive"):
        residual_probability(p, p)


@pytest.mark.parametrize("family", ["kimi", "qwen", "deepseek"])
@pytest.mark.parametrize("reject_at", list(range(6)))
def test_rejection_replays_only_accepted_tokens_and_features(family, reject_at):
    torch.manual_seed(590)
    model = make_target(family)
    prefix = torch.tensor([[1, 23, 35, 44, 28]])
    with torch.no_grad():
        session = SpeculativeSession(model, prefix)

        def proposer(context, _maximum):
            # Oracle proposal: target distribution until one deliberately poor action.
            current = context.ids
            probs, tokens = [], []
            for index in range(6):
                p = probability(model(current).logits[:, -1], model)
                token = p.argmax(-1, keepdim=True)
                q = p.clone()
                if index == reject_at:
                    q.zero_().scatter_(-1, token, 1)
                tokens.append(token)
                probs.append(q)
                current = torch.cat((current, token), 1)
            return Proposal(torch.cat(tokens, 1), torch.stack(probs, 1))

        with patch("minifrontier.speculative.torch.rand", return_value=0.99):
            emitted = session.advance(proposer, 6)
        assert emitted.shape[1] == reject_at + 1
        assert session.stats["rejections"] == 1
        assert (
            session.cache.length == session.context.ids.shape[1] == prefix.shape[1] + reject_at + 1
        )
        extra = {"return_taps": True} if family != "qwen" else {}
        full = model(session.context.ids, return_hidden=True, **extra)
        torch.testing.assert_close(
            session.context.next_p, probability(full.logits[:, -1], model), rtol=3e-5, atol=3e-6
        )
        if family == "qwen":
            torch.testing.assert_close(
                session.context.multistream, full.multistream_hidden, rtol=3e-5, atol=3e-6
            )
        else:
            for cached, expected in zip(
                session.context.taps, full.tapped_hidden_states, strict=True
            ):
                torch.testing.assert_close(cached, expected, rtol=3e-5, atol=3e-6)
        following = torch.tensor([[31, 37]])
        cached = session.forward(following).logits
        expected = model(torch.cat((session.context.ids, following), 1)).logits[:, -2:]
        torch.testing.assert_close(cached, expected, rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize("family", ["kimi", "qwen", "deepseek"])
def test_all_accepted_bonus_and_eos_rollback(family):
    torch.manual_seed(817)
    model = make_target(family)
    prefix = torch.tensor([[1, 23, 29, 33, 47]])
    with torch.no_grad():
        session = SpeculativeSession(model, prefix)

        def proposer(context, _maximum):
            token = torch.tensor([[23, 25, 27]])
            q = torch.nn.functional.one_hot(token, model.config.vocab_size).float()
            return Proposal(token, q)

        # Exercise cache mechanics independently of the analytic acceptance test.
        with patch("minifrontier.speculative.accepts", return_value=True):
            emitted = session.advance(proposer, 4)
        assert emitted.shape[1] == 4 and session.cache.length == 9
        expected = probability(model(session.context.ids).logits[:, -1], model)
        torch.testing.assert_close(session.context.next_p, expected, rtol=3e-5, atol=3e-6)

        def eos_proposer(context, _maximum):
            token = torch.tensor([[23, 2, 27]])
            return Proposal(
                token, torch.nn.functional.one_hot(token, model.config.vocab_size).float()
            )

        with patch("minifrontier.speculative.accepts", return_value=True):
            emitted = session.advance(eos_proposer, 4)
        assert emitted.tolist() == [[23, 2]] and session.cache.length == 11
        expected = probability(model(session.context.ids).logits[:, -1], model)
        torch.testing.assert_close(session.context.next_p, expected, rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize("family", ["kimi", "qwen", "deepseek"])
def test_native_drafts_generate_without_future_target_feature_reads(family):
    torch.manual_seed(730)
    target = make_target(family)
    draft_type = {"kimi": KimiDraft, "qwen": QwenDraft, "deepseek": DSparkDraft}[family]
    draft = draft_type(target).eval()
    ids = torch.tensor([[1, 23, 31, 29, 48]])
    output, stats = generate_speculative(target, draft, ids, max_new_tokens=9, draft_steps=3)
    assert 1 <= output.shape[1] - ids.shape[1] <= 9
    assert output[:, ids.shape[1] :].ne(0).all() and output[:, ids.shape[1] :].ne(1).all()
    assert stats["proposed"] >= stats["accepted"] >= 1
    with torch.no_grad():
        context = SpeculativeSession(target, ids).context
        # A proposer is allowed the existing taps and q0, not any additional target forward.
        with patch.object(target, "forward", side_effect=AssertionError("future target read")):
            proposal = draft.propose(context, steps=3)
        proposal.validate(context, 4)
