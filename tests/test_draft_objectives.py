"""Independent overlap numbers and self-fed seven-step Kimi draft gradients."""

import math

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
from minifrontier.training.draft_losses import dspark_loss, kimi_lk


def test_overlap_losses_match_two_category_probabilities():
    teacher = torch.tensor([[[0.8, 0.2]]]).log().requires_grad_()
    student = torch.tensor([[[0.5, 0.5]]]).log().requires_grad_()
    confidence = torch.zeros(1, 1, requires_grad=True)
    lk = kimi_lk(teacher, student)
    torch.testing.assert_close(lk, torch.tensor([[-math.log(0.7)]]))
    loss, report = dspark_loss(
        teacher, student, torch.tensor([[0]]), confidence, torch.ones(1, 1, dtype=torch.bool)
    )
    torch.testing.assert_close(loss, torch.tensor(1.1 * math.log(2) + 0.9 * 0.6))
    torch.testing.assert_close(report["overlap"], torch.tensor([[0.7]]))
    (loss + lk.sum()).backward()
    assert teacher.grad is None and student.grad.abs().sum() > 0 and confidence.grad.abs().sum() > 0


def test_dspark_accumulation_uses_decayed_weight_sum_not_integer_position_count():
    torch.manual_seed(251)
    teacher = torch.randn(2, 5, 4)
    logits = torch.randn(2, 5, 4, requires_grad=True)
    confidence = torch.randn(2, 5, requires_grad=True)
    tokens = torch.tensor([[1, 2, 3, 0, 1], [2, 3, 0, 1, 2]])
    valid = torch.tensor([[True, False, False, False, False], [True] * 5])
    full, _ = dspark_loss(teacher, logits, tokens, confidence, valid)
    pieces = [
        dspark_loss(
            teacher[i : i + 1],
            logits[i : i + 1],
            tokens[i : i + 1],
            confidence[i : i + 1],
            valid[i : i + 1],
        )
        for i in range(2)
    ]
    combined = sum(loss * r["normalizer"] for loss, r in pieces) / sum(
        r["normalizer"] for _, r in pieces
    )
    wrong = sum(loss * r["positions"] for loss, r in pieces) / 6
    torch.testing.assert_close(combined, full)
    assert not torch.isclose(wrong, full)
    for expected, actual in zip(
        torch.autograd.grad(full, (logits, confidence), retain_graph=True),
        torch.autograd.grad(combined, (logits, confidence)),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)


def test_lk_has_gradient_when_overlap_is_too_small_for_probability_space():
    target = torch.tensor([[0.0, -300.0]])
    draft = torch.tensor([[-300.0, 0.0]], requires_grad=True)
    loss = kimi_lk(target, draft).sum()
    assert torch.isfinite(loss) and loss > 290
    loss.backward()
    assert torch.isfinite(draft.grad).all() and draft.grad.abs().sum() > 0.9


def test_kimi_block_taps_and_seven_step_draft_leave_target_unchanged(monkeypatch):
    torch.manual_seed(245)
    model = MiniKimiK3ForCausalLM(tiny_kimi(num_hidden_layers=12, mtp_enabled=True))
    ids = torch.tensor([[1, 21, 17, 33, 41]])
    observed = []
    handles = [
        model.layers[i].register_forward_hook(lambda _m, _a, result: observed.append(result[0]))
        for i in (3, 7, 11)
    ]
    with torch.no_grad():
        output = model.eval()(ids, return_taps=True)
    for expected, actual in zip(observed, output.tapped_hidden_states, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for handle in handles:
        handle.remove()
    draft = KimiDraft(model)
    torch.testing.assert_close(
        draft.tap_fusion(torch.cat(output.tapped_hidden_states, -1)), observed[-1]
    )
    original = {key: value.clone() for key, value in model.state_dict().items()}
    monkeypatch.setattr(
        torch, "multinomial", lambda p, n: torch.full((p.shape[0], n), 10, device=p.device)
    )
    loss, report = draft.unroll(ids, anchor_ids=torch.tensor([[20]]))
    assert report["steps"] == 7 and report["positions"] == 7 and torch.isfinite(loss)
    loss.backward()
    assert draft.tap_fusion.weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.count_nonzero() for p in draft.mtp.parameters())
    assert all(p.grad is None for p in model.parameters())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original[key], rtol=0, atol=0)


def test_dspark_backbone_cannot_read_future_tokens_but_markov_can_read_previous():
    torch.manual_seed(823)
    target = MiniDeepSeekV4ForCausalLM(tiny_deepseek(mtp_enabled=True))
    draft = DSparkDraft(target, markov_rank=8)
    prefix = torch.tensor([[1, 14, 22, 17, 33]])
    anchor = torch.tensor([[23]])
    tokens = torch.tensor([[25, 31, 17, 38, 2]])
    original = {key: value.clone() for key, value in target.state_dict().items()}
    with torch.no_grad():
        taps = target(prefix, return_taps=True).tapped_hidden_states
    previous = torch.cat((anchor, tokens[:, :-1]), 1)
    # Use a nonzero Markov head so this detects the permitted dependency too.
    torch.nn.init.normal_(draft.markov_w2.weight, std=0.02)
    logits, confidence, base = draft(taps, anchor, previous)
    other = previous.clone()
    other[:, 2] = 44
    changed, changed_confidence, changed_base = draft(taps, anchor, other)
    torch.testing.assert_close(base, changed_base, rtol=0, atol=0)
    torch.testing.assert_close(logits[:, :2], changed[:, :2], rtol=0, atol=0)
    torch.testing.assert_close(logits[:, 3:], changed[:, 3:], rtol=0, atol=0)
    assert not torch.equal(logits[:, 2], changed[:, 2])
    loss, report = draft.training_block(prefix, anchor, tokens)
    assert report["positions"] == 5 and confidence.shape == changed_confidence.shape == (1, 5)
    loss.backward()
    assert draft.main_proj.weight.grad.abs().sum() > 0
    assert all(
        any(p.grad is not None and p.grad.count_nonzero() for p in stage.parameters())
        for stage in draft.stages
    )
    assert all(p.grad is None for p in target.parameters())
    for key, value in target.state_dict().items():
        torch.testing.assert_close(value, original[key], rtol=0, atol=0)


def test_qwen_draft_ce_matches_copied_mtp_with_four_real_streams():
    target = MiniQwen4ForCausalLM(tiny_config(mtp_enabled=True, vocab_size=64)).eval()
    ids = torch.tensor([[1, 23, 31, 25, 38, 42, 2]])
    labels = ids.clone()
    labels[:, :3] = -100
    with torch.no_grad():
        expected = target(ids, labels=labels).mtp_loss
    draft = QwenDraft(target)
    loss, report = draft(ids, labels)
    torch.testing.assert_close(loss, expected)
    assert (
        report["multistream_hidden"].shape[-1] == target.config.hc_count * target.config.hidden_size
    )
    loss.backward()
    assert draft.mtp.shared_head.head.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in target.parameters())


def test_qwen_three_step_self_fed_ce_has_checkpointed_gradients(monkeypatch):
    target = MiniQwen4ForCausalLM(
        tiny_config(mtp_enabled=True, vocab_size=64, gradient_checkpointing=True)
    ).eval()
    draft = QwenDraft(target)
    monkeypatch.setattr(
        torch, "multinomial", lambda p, n: torch.full((p.shape[0], n), 23, device=p.device)
    )
    loss, report = draft.unroll(torch.tensor([[1, 23, 25, 29, 37]]))
    assert report["positions"] == 3
    loss.backward()
    assert draft.mtp.fc_hidden.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in target.parameters())


@pytest.mark.cuda
@pytest.mark.parametrize("family", ["kimi", "deepseek"])
def test_qat_draft_bf16_backward_preserves_target(family, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.manual_seed(317)
    if family == "kimi":
        target = MiniKimiK3ForCausalLM(
            tiny_kimi(
                num_hidden_layers=12,
                mtp_enabled=True,
                moe_intermediate_size=32,
                routed_expert_hidden_size=32,
                qat_scheme="mxfp4-mxfp8-v1",
            )
        ).cuda()
        draft = KimiDraft(target)
    else:
        target = MiniDeepSeekV4ForCausalLM(
            tiny_deepseek(
                mtp_enabled=True, moe_inter_dim=32, index_head_dim=32, qat_scheme="mxfp4-indexer-v1"
            )
        ).cuda()
        draft = DSparkDraft(target)
    original = {name: value.clone() for name, value in target.state_dict().items()}
    prefix = torch.tensor([[1, 21, 23, 31, 25]], device="cuda")
    monkeypatch.setattr(
        torch, "multinomial", lambda p, n: torch.full((p.shape[0], n), 23, device=p.device)
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if family == "kimi":
            loss, report = draft.unroll(prefix, anchor_ids=torch.tensor([[29]], device="cuda"))
            assert report["positions"] == 7
        else:
            loss, report = draft.training_block(
                prefix,
                torch.tensor([[29]], device="cuda"),
                torch.tensor([[31, 23, 25, 28, 2]], device="cuda"),
            )
            assert report["positions"] == 5
    loss.backward()
    assert torch.isfinite(loss)
    gradients = [p.grad for p in draft.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any(g.count_nonzero() for g in gradients)
    assert all(p.grad is None for p in target.parameters())
    for name, value in target.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)
