"""Dense execution must preserve full key support without full-query logits."""

import pytest
import torch

from minifrontier.models.minideepseekv4.attention import _dense_attention


@pytest.mark.parametrize("recompute", [False, True])
@pytest.mark.parametrize("autocast", [False, True])
def test_query_chunks_preserve_sink_and_compressed_key_gradients(recompute, autocast):
    torch.manual_seed(413)
    q = torch.randn(2, 137, 2, 16, requires_grad=True)
    kv = torch.randn(2, 171, 16, requires_grad=True)
    sink = torch.randn(2, requires_grad=True)
    mask = torch.rand(2, 137, 171) > 0.4
    mask[:, 0] = False  # a sink-only query must remain finite
    incoming = torch.randn_like(q)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        scores = torch.einsum("bthd,bcd->bhtc", q.float(), kv.float()) * 16**-0.5
        scores = scores.masked_fill(~mask[:, None], float("-inf"))
        sinks = sink.view(1, -1, 1, 1).expand(2, -1, 137, 1)
        probabilities = torch.cat((scores, sinks), -1).softmax(-1)[..., :-1]
        expected = torch.einsum("bhtc,bcd->bthd", probabilities.to(kv.dtype), kv)
        actual = _dense_attention(q, kv, mask, sink, recompute=recompute)
    original = torch.autograd.grad((expected * incoming).sum(), (q, kv, sink))
    tiled = torch.autograd.grad((actual * incoming).sum(), (q, kv, sink))
    tolerance = dict(atol=2e-3, rtol=2e-2) if autocast else dict(atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(actual, expected, **tolerance)
    if autocast:
        # Query tiling changes BF16 KV-gradient reduction order. Compare both
        # versions to the same FP32 oracle instead of relative error near zero.
        fp_scores = torch.einsum("bthd,bcd->bhtc", q, kv) * 16**-0.5
        fp_scores = fp_scores.masked_fill(~mask[:, None], float("-inf"))
        fp_probs = torch.cat((fp_scores, sinks), -1).softmax(-1)[..., :-1]
        fp_out = torch.einsum("bhtc,bcd->bthd", fp_probs, kv)
        fp_grads = torch.autograd.grad((fp_out * incoming).sum(), (q, kv, sink))
        for left, right, oracle in zip(tiled, original, fp_grads, strict=True):
            error = (left - oracle).square().mean().sqrt()
            baseline = (right - oracle).square().mean().sqrt()
            assert error <= baseline * 1.5 + 1e-6
            assert error <= oracle.square().mean().sqrt() * 0.015 + 1e-6
    else:
        for left, right in zip(tiled, original, strict=True):
            torch.testing.assert_close(left, right, **tolerance)
    assert actual[:, 0].count_nonzero() == 0
    assert torch.isfinite(actual).all()


def test_forward_and_checkpoint_replay_bound_query_score_allocation(monkeypatch):
    original = torch.einsum
    lengths = []

    def observe(equation, *args, **kwargs):
        if equation == "bthd,bcd->bhtc":
            lengths.append(args[0].shape[1])
        return original(equation, *args, **kwargs)

    monkeypatch.setattr(torch, "einsum", observe)
    q = torch.randn(1, 513, 2, 8, requires_grad=True)
    kv = torch.randn(1, 641, 8, requires_grad=True)
    mask = torch.ones(1, 513, 641, dtype=torch.bool)
    sink = torch.zeros(2, requires_grad=True)
    _dense_attention(q, kv, mask, sink, recompute=True).square().sum().backward()
    assert max(lengths) == 128 and len(lengths) >= 10


def test_chunk_recomputation_preserves_full_model_mtp_and_layer_gradients(monkeypatch):
    import copy

    from test_new_backbones import tiny_deepseek

    from minifrontier.models.minideepseekv4 import MiniDeepSeekV4ForCausalLM, attention

    torch.manual_seed(83)
    model = MiniDeepSeekV4ForCausalLM(tiny_deepseek(mtp_enabled=True))
    reference = copy.deepcopy(model)
    ids = torch.randint(10, 60, (2, 137))
    original = attention._dense_attention
    actual = model(ids, labels=ids, return_logits=False)
    actual.loss.backward()
    monkeypatch.setattr(
        attention,
        "_dense_attention",
        lambda q, kv, mask, sink, **kw: original(q, kv, mask, sink, chunk_size=q.shape[1]),
    )
    expected = reference(ids, labels=ids, return_logits=False)
    expected.loss.backward()
    torch.testing.assert_close(actual.loss, expected.loss, rtol=2e-5, atol=2e-6)
    for (name, p), (other, r) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert name == other
        torch.testing.assert_close(p.grad, r.grad, rtol=3e-4, atol=3e-6, msg=name)


@pytest.mark.parametrize("ratio", [0, 4, 128])
@pytest.mark.parametrize("recompute", [False, True])
@pytest.mark.parametrize("autocast", [False, True])
def test_causal_key_bounds_preserve_outputs_and_gradients(ratio, recompute, autocast):
    torch.manual_seed(415)
    length, window = 269, 32
    compressed = length // ratio if ratio else 0
    q = torch.randn(2, length, 2, 16, requires_grad=True)
    kv = torch.randn(2, length + compressed, 16, requires_grad=True)
    sink = torch.randn(2, requires_grad=True)
    pos = torch.arange(length)
    visible = (pos[:, None] >= pos) & (pos[:, None] - pos < window)
    if ratio:
        visible = torch.cat(
            (visible, torch.arange(compressed)[None] < (pos[:, None] + 1) // ratio), -1
        )
    mask = visible[None].expand(2, -1, -1).clone()
    mask[:, 0] = False  # The learned sink remains the only visible entry.
    mask[1, :, -3:] = False  # Retain the original per-example mask too.
    incoming = torch.randn_like(q)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        expected = _dense_attention(q, kv, mask, sink, chunk_size=64)
        actual = _dense_attention(
            q,
            kv,
            mask,
            sink,
            chunk_size=64,
            recompute=recompute,
            window_size=window,
            compress_ratio=ratio,
        )
    original = torch.autograd.grad((expected * incoming).sum(), (q, kv, sink))
    bounded = torch.autograd.grad((actual * incoming).sum(), (q, kv, sink))
    tolerance = dict(atol=2e-3, rtol=2e-2) if autocast else dict(atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(actual, expected, **tolerance)
    if autocast:
        oracle_output = _dense_attention(q, kv, mask, sink, chunk_size=64)
        oracle_grads = torch.autograd.grad((oracle_output * incoming).sum(), (q, kv, sink))
        for actual_grad, old_grad, oracle in zip(bounded, original, oracle_grads, strict=True):
            error = (actual_grad - oracle).square().mean().sqrt()
            baseline = (old_grad - oracle).square().mean().sqrt()
            assert error <= baseline * 1.5 + 1e-6
            assert error <= oracle.square().mean().sqrt() * 0.015 + 1e-6
    else:
        for left, right in zip(bounded, original, strict=True):
            torch.testing.assert_close(left, right, **tolerance)
    assert actual[:, 0].count_nonzero() == 0
    if ratio:
        assert bounded[1][:, length:].count_nonzero() > 0


def test_causal_key_bounds_apply_during_checkpoint_replay(monkeypatch):
    original = torch.einsum
    products = []

    def observe(equation, *args, **kwargs):
        if equation == "bthd,bcd->bhtc":
            products.append((args[0].shape[1], args[1].shape[1]))
        return original(equation, *args, **kwargs)

    monkeypatch.setattr(torch, "einsum", observe)
    length, ratio, window = 513, 4, 128
    pos = torch.arange(length)
    raw = (pos[:, None] >= pos) & (pos[:, None] - pos < window)
    compressed = torch.arange(length // ratio)[None] < (pos[:, None] + 1) // ratio
    mask = torch.cat((raw, compressed), -1)[None]
    q = torch.randn(1, length, 2, 8, requires_grad=True)
    kv = torch.randn(1, mask.shape[-1], 8, requires_grad=True)
    sink = torch.zeros(2, requires_grad=True)
    _dense_attention(
        q, kv, mask, sink, recompute=True, window_size=window, compress_ratio=ratio
    ).square().sum().backward()
    assert len(products) == 10  # Five forward chunks and their backward recomputations.
    assert max(queries for queries, _ in products) == 128
    assert max(keys for _, keys in products) == 255 + 128
    assert all(keys < kv.shape[1] for _, keys in products)
