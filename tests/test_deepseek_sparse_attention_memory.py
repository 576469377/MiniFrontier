"""Sparse query tiling preserves the original attention and indexer teacher."""

import copy

import pytest
import torch
from test_new_backbones import tiny_deepseek

from minifrontier.models.minideepseekv4 import MiniDeepSeekV4ForCausalLM
from minifrontier.models.minideepseekv4.attention import _sparse_attention


def reference(q, kv, mask, sink, scores, valid):
    logits = torch.einsum("bthd,bcd->bhtc", q.float(), kv.float()) * q.shape[-1] ** -0.5
    logits = logits.masked_fill(~mask[:, None], float("-inf"))
    sinks = sink.view(1, -1, 1, 1).expand(q.shape[0], -1, q.shape[1], 1)
    probs = torch.cat((logits, sinks), dim=-1).softmax(-1)[..., :-1]
    support = mask[..., q.shape[1] :]
    target = probs[..., q.shape[1] :].detach().sum(dim=1) * support
    target = target / target.sum(-1, keepdim=True).clamp_min(1e-12)
    logp = scores.masked_fill(~support, torch.finfo(scores.dtype).min).log_softmax(-1)
    kl = target * (target.clamp_min(1e-12).log() - logp)
    loss = (kl.sum(-1) * valid).sum() / valid.sum().clamp_min(1)
    out = torch.einsum("bhtc,bcd->bthd", probs.to(kv.dtype), kv)
    return out, loss


@pytest.mark.parametrize("recompute", [False, True])
@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("window_size", [None, 64])
def test_sparse_chunks_match_full_teacher_outputs_and_gradients(recompute, autocast, window_size):
    torch.manual_seed(824)
    length, count = 137, 137 // 4
    q = torch.randn(2, length, 2, 16, requires_grad=True)
    kv = torch.randn(2, length + count, 16, requires_grad=True)
    sink = torch.randn(2, requires_grad=True)
    scores = torch.rand(2, length, count)
    scores[..., :8] = 0  # ReLU ties retain the original stable selection.
    scores.requires_grad_()
    pos = torch.arange(length)
    causal = (pos[:, None] >= pos) & (pos[:, None] - pos < 64)
    if window_size is None:
        causal[10:20, :24] = True  # Custom visibility retains complete raw support.
    support = (torch.arange(count) < (pos[:, None] + 1) // 4)[None].expand(2, -1, -1)
    indices = scores.masked_fill(~support, float("-inf")).argsort(
        dim=-1, descending=True, stable=True
    )[..., :8]
    selected = torch.zeros_like(support).scatter(-1, indices, True) & support
    mask = torch.cat((causal[None].expand(2, -1, -1), selected), dim=-1)
    mask[:, 0] = False  # Sink-only rows remain finite, with zero output and KL.
    valid = pos[None] < torch.tensor([103, length])[:, None]
    incoming = torch.randn_like(q)
    tensors = q, kv, sink, scores
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        expected, expected_kl = reference(*tensors[:2], mask, sink, scores, valid)
        actual, actual_kl = _sparse_attention(
            q,
            kv,
            mask,
            sink,
            scores,
            valid,
            recompute=recompute,
            chunk_size=32,
            window_size=window_size,
        )
    original_grads = torch.autograd.grad((expected * incoming).sum() + expected_kl, tensors)
    tiled_grads = torch.autograd.grad((actual * incoming).sum() + actual_kl, tensors)
    tolerance = dict(atol=3e-3, rtol=2e-2) if autocast else dict(atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(actual, expected, **tolerance)
    torch.testing.assert_close(actual_kl, expected_kl, **tolerance)
    assert actual[:, 0].count_nonzero() == 0
    assert torch.isfinite(actual).all() and torch.isfinite(actual_kl)
    if autocast:
        oracle, oracle_kl = reference(q, kv, mask, sink, scores, valid)
        fp32_grads = torch.autograd.grad((oracle * incoming).sum() + oracle_kl, tensors)
        for left, right, fp32 in zip(tiled_grads, original_grads, fp32_grads, strict=True):
            error = (left - fp32).square().mean().sqrt()
            baseline = (right - fp32).square().mean().sqrt()
            assert error <= baseline * 1.5 + 1e-6
            assert error <= fp32.square().mean().sqrt() * 0.02 + 1e-6
    else:
        for actual_grad, expected_grad in zip(tiled_grads, original_grads, strict=True):
            torch.testing.assert_close(actual_grad, expected_grad, **tolerance)


@pytest.mark.parametrize("checkpointing", [False, True])
def test_sparse_full_model_padding_mtp_and_parameter_gradients(checkpointing):
    torch.manual_seed(823)
    model = MiniDeepSeekV4ForCausalLM(
        tiny_deepseek(
            mtp_enabled=True, gradient_checkpointing=checkpointing, sequence_balance_coef=1e-4
        ),
        training_phase="sparse_cpt",
    )
    original = copy.deepcopy(model)
    model.set_sparse_attention_backend("chunked")
    ids = torch.randint(3, 64, (2, 137))
    ids[0, 103:] = 0
    options = dict(labels=ids, attention_mask=ids.ne(0), return_logits=False, return_hidden=True)
    actual, expected = model(ids, **options), original(ids, **options)
    for field in ("loss", "lm_loss", "indexer_loss", "mtp_loss", "aux_loss", "hidden_states"):
        torch.testing.assert_close(
            getattr(actual, field), getattr(expected, field), rtol=3e-5, atol=3e-6
        )
    actual.loss.backward()
    expected.loss.backward()
    for (name, parameter), (other, reference_parameter) in zip(
        model.named_parameters(), original.named_parameters(), strict=True
    ):
        assert name == other
        torch.testing.assert_close(
            parameter.grad, reference_parameter.grad, rtol=3e-4, atol=3e-6, msg=name
        )
    assert model.config == original.config
    assert model.state_dict().keys() == original.state_dict().keys()


def test_sparse_score_products_are_tiled_during_checkpoint_replay(monkeypatch):
    torch.manual_seed(825)
    model = MiniDeepSeekV4ForCausalLM(tiny_deepseek(), training_phase="sparse_cpt")
    model.set_sparse_attention_backend("chunked")
    original = torch.einsum
    lengths = []

    def observe(equation, *args, **kwargs):
        if equation == "bthd,bcd->bhtc":
            lengths.append(args[0].shape[1])
        return original(equation, *args, **kwargs)

    monkeypatch.setattr(torch, "einsum", observe)
    ids = torch.randint(3, 64, (2, 137))
    model(ids, labels=ids, return_logits=False).loss.backward()
    assert max(lengths) <= 128 and len(lengths) > 6
    with pytest.raises(ValueError, match="sparse attention backend"):
        model.set_sparse_attention_backend("unknown")
