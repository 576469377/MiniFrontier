"""Fusion-specific oracles: old family tests cannot validate these new contracts."""

import copy
from dataclasses import asdict, replace

import pytest
import torch
from PIL import Image

from minifrontier.models.minifrontier1 import (
    MiniFrontier1Cache,
    MiniFrontier1Config,
    MiniFrontier1ForCausalLM,
)
from minifrontier.models.minifrontier1.csa import Compressor
from minifrontier.models.minifrontier1.indexer import (
    block_registry,
    mrope,
    select_blocks,
    visible_blocks,
)
from minifrontier.models.minifrontier1.kda import KDA
from minifrontier.models.minifrontier1.lookup import NgramLookup
from minifrontier.models.minifrontier1.moe import LatentMoE
from minifrontier.models.minifrontier1.mtp import mtp_targets
from minifrontier.models.minifrontier1.processing import process_frames, token_metadata
from minifrontier.models.minifrontier1.residual import GatedResidual
from minifrontier.training.minifrontier1_optim import (
    QuantileBalance,
    make_optimizer,
    parameter_report,
)
from minifrontier.training.minifrontier1_strategy import budget_report


@pytest.fixture
def config():
    torch.manual_seed(918)
    return MiniFrontier1Config.tiny()


def media_batch(c, *, video=False):
    images = [
        Image.new("RGB", (16, 16), color)
        for color in (["red", "blue", "green", "yellow"] if video else ["red"])
    ]
    sample = process_frames(
        images,
        patch_size=c.vision_config.patch_size,
        max_features=16,
        timestamps=[0.0, 0.2, 1.4, 2.0] if video else None,
    )
    sample.update(batch_index=0, start=3, resource_kind="video" if video else "image")
    ids = torch.tensor(
        [[30, 31, 9, *([7] * sample["feature_count"]), 10, 32, 33, 34, 35, 36, 37, 38, 39]]
    )
    return ids, [sample]


@pytest.mark.parametrize(
    "change",
    [
        dict(num_hidden_layers=3),
        dict(hc_count=2),
        dict(mrope_sections=(1, 1, 1)),
        dict(num_experts_per_token=5),
        dict(protected_media_tokens=512),
        dict(lookup_layer=0),
    ],
)
def test_config_rejects_invalid_contract(config, change):
    with pytest.raises(ValueError):
        MiniFrontier1Config(**(asdict(config) | change))


def test_full_parameter_and_budget_ledger():
    with torch.device("meta"):
        model = MiniFrontier1ForCausalLM(MiniFrontier1Config())
    report = parameter_report(model)
    assert 225_000_000 <= report["total"] <= 235_000_000
    assert report["total"] == sum(report["categories"].values())
    assert model.mtp is not None and not any(
        "lm_head" in n for n, _ in model.mtp.named_parameters()
    )
    assert budget_report()["main_ce_tokens"] == 3_000_000_000
    assert budget_report()["visual_ce_tokens"] == 630_000_000


def test_gr_exact_primitive_and_four_stream_gradients(config):
    gr = GatedResidual(config)
    x = torch.randn(2, 3, 4, 32, requires_grad=True)
    read, weights = gr.read(x)
    expected, original, injection = gr.primitive(x.flatten(-2))
    torch.testing.assert_close(read, expected)
    torch.testing.assert_close(
        gr.inject(x, read, weights),
        original.unflatten(-1, (4, 32)) + injection[..., None] * expected[..., None, :],
    )
    gr.inject(x, read, weights).square().sum().backward()
    assert x.grad.abs().sum((0, 1, 3)).gt(0).all()


def test_registry_completion_before_selection():
    s = torch.tensor([0] * 9 + [1] * 2)
    m = torch.tensor([0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    media = torch.where(m.bool(), 5, -1)
    blocks = block_registry(s, m, media)
    assert [(b.member_indices, b.complete_at) for b in blocks] == [
        ((0, 1), 2),
        ((2, 3, 4), 5),
        ((5, 6, 7, 8), 8),
        ((9, 10), None),
    ]
    visible = visible_blocks(blocks, torch.arange(11), s)
    selected = select_blocks(torch.tensor([[0.0, 0.0, 0.0, 1e9]]).expand(11, -1), visible, 1)
    assert not selected[:2].any() and not selected[:, -1].any()
    assert selected[2, 0] and not selected[9:].any()


def test_compressor_short_flush_and_no_cross_media_overlap():
    compressor = Compressor(2, 2, 1e-5)
    with torch.no_grad():
        compressor.kv.weight.copy_(torch.eye(2).repeat(2, 1))
        compressor.gate.weight.zero_()
    x = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 2.0], [0.0, 2.0], [0.0, 2.0], [5.0, 1.0]])
    seg, mod = torch.zeros(6, dtype=torch.long), torch.tensor([0, 0, 1, 1, 1, 0])
    blocks = block_registry(seg, mod, torch.where(mod.bool(), 0, -1))
    pooled = compressor(x, blocks, 0)
    torch.testing.assert_close(pooled[0], compressor.norm(torch.tensor([1.0, 0.0])))
    torch.testing.assert_close(pooled[1], compressor.norm(torch.tensor([0.0, 2.0])))
    assert torch.isfinite(pooled).all()


@pytest.mark.parametrize("phase", ["dense_pretrain", "sparse_cpt"])
@pytest.mark.parametrize("video", [False, True])
def test_all_paths_media_cache_every_boundary_and_rollback(config, phase, video):
    model = MiniFrontier1ForCausalLM(config, phase).eval()
    ids, media = media_batch(config, video=video)
    end = media[0]["start"] + media[0]["feature_count"]
    with torch.no_grad():
        full = model(ids, media=media).logits
        cache = MiniFrontier1Cache()
        prefix = model(ids[:, :end], media=media, cache=cache).logits
        snapshot = cache.snapshot()
        streamed = [prefix]
        for t in range(end, ids.shape[1]):
            streamed.append(model(ids[:, t : t + 1], cache=cache).logits)
        torch.testing.assert_close(torch.cat(streamed, 1), full, atol=2e-6, rtol=2e-5)
        assert cache.layers[-1][0]["latent"].shape[0] == ids.shape[1]
        assert len(cache.layers[1][0]["raw"]) <= config.window_size - 1
        assert len(cache.layers[1][0]["tail"]["x"]) <= 8
        cache.restore(snapshot)
        replay = model(ids[:, end:], cache=cache).logits
        torch.testing.assert_close(replay, full[:, end:], atol=2e-6, rtol=2e-5)


def test_cache_reorder_and_model_mutation(config):
    model = MiniFrontier1ForCausalLM(config).eval()
    ids = torch.randint(24, 100, (2, 12))
    with torch.no_grad():
        cache = MiniFrontier1Cache()
        model(ids[:, :8], cache=cache)
        cache.reorder(torch.tensor([1, 0, 1]))
        actual = model(ids[[1, 0, 1], 8:], cache=cache).logits
        expected = model(ids[[1, 0, 1]]).logits[:, 8:]
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        model.lm_head.weight.add_(0.1)
        with pytest.raises(ValueError, match="changed"):
            model(ids[:1, :1].expand(3, -1), cache=cache)


def test_no_future_or_packed_sample_leak(config):
    model = MiniFrontier1ForCausalLM(config, "sparse_cpt").eval()
    ids = torch.randint(24, 100, (1, 23))
    segments = torch.tensor([[0] * 11 + [1] * 12])
    with torch.no_grad():
        reference = model(ids, segment_ids=segments).logits
        changed = ids.clone()
        changed[:, :11] = torch.randint(100, 200, (1, 11))
        torch.testing.assert_close(
            reference[:, 11:],
            model(changed, segment_ids=segments).logits[:, 11:],
            atol=2e-6,
            rtol=2e-5,
        )
        changed = ids.clone()
        changed[:, 17:] = torch.randint(100, 200, (1, 6))
        torch.testing.assert_close(
            reference[:, :17],
            model(changed, segment_ids=segments).logits[:, :17],
            atol=2e-6,
            rtol=2e-5,
        )


def test_qsa_full_budget_dense_and_indexer_cannot_change_reference(config):
    model = MiniFrontier1ForCausalLM(replace(config, top_blocks=128)).eval()
    ids = torch.randint(24, 100, (1, 23))
    with torch.no_grad():
        dense = model(ids).logits
        for name, p in model.named_parameters():
            if ".indexer." in name:
                p.normal_(0, 10)
        torch.testing.assert_close(model(ids).logits, dense, atol=0, rtol=0)
        model.set_phase("sparse_cpt")
        torch.testing.assert_close(model(ids).logits, dense, atol=2e-6, rtol=2e-5)


def test_lookup_streaming_and_control_reset(config):
    lookup = NgramLookup(config)
    ids = torch.tensor([[40, 41, 42, 7, 43, 44, 5, 45, 46]])
    h = torch.randn(1, ids.shape[1], 32)
    seg, mod = torch.zeros_like(ids), ids.eq(7).long()
    full, _ = lookup(h, ids, seg, mod)
    state, chunks = None, []
    for i in range(ids.shape[1]):
        value, state = lookup(
            h[:, i : i + 1], ids[:, i : i + 1], seg[:, i : i + 1], mod[:, i : i + 1], state
        )
        chunks.append(value)
    torch.testing.assert_close(full, torch.cat(chunks, 1), atol=1e-7, rtol=1e-6)
    assert full[:, [3, 6]].eq(0).all()


def test_kda_reference_output_and_gradient_nonchunk_length(config):
    a = KDA(replace(config, kda_backend="reference"))
    b = KDA(replace(config, kda_backend="auto"))
    b.load_state_dict(a.state_dict())
    with torch.no_grad():
        a.core.A_log.zero_()
        b.core.A_log.zero_()
        a.core.dt_bias.zero_()
        b.core.dt_bias.zero_()
    x = torch.randn(2, 9, 32, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    seg = torch.tensor([[0] * 5 + [1] * 4, [0] * 7 + [-1] * 2])
    first = a(x, seg)[0]
    second = b(y, seg)[0]
    torch.testing.assert_close(first, second, atol=1e-5, rtol=1e-5)
    first.square().sum().backward()
    second.square().sum().backward()
    torch.testing.assert_close(x.grad, y.grad, atol=1e-5, rtol=1e-5)


def test_moe_matches_expert_oracle_and_bias_not_in_weights(config):
    moe = LatentMoE(config)
    with torch.no_grad():
        moe.router.weight.normal_(0, 0.1)
    x = torch.randn(2, 3, 32, requires_grad=True)
    flat = x.flatten(0, 1)
    selected, weights, scores = moe.router(flat)
    torch.testing.assert_close(
        weights, scores.gather(-1, selected) / scores.gather(-1, selected).sum(-1, keepdim=True)
    )
    latent = moe.down(flat)
    oracle = torch.stack(
        [
            sum(
                moe.experts[int(selected[t, j])](latent[t : t + 1])[0].float() * weights[t, j]
                for j in range(config.num_experts_per_token)
            )
            for t in range(len(flat))
        ]
    )
    oracle = moe.up(moe.norm(oracle)) + moe.shared(flat)
    torch.testing.assert_close(moe(x).flatten(0, 1), oracle, atol=1e-6, rtol=1e-5)


def test_visual_gradient_mtp_targets_and_optimizer_coverage(config):
    model = MiniFrontier1ForCausalLM(config)
    ids, media = media_batch(config)
    labels = ids.clone()
    labels[:, : media[0]["start"] + media[0]["feature_count"] + 1] = -100
    result = model(ids, labels=labels, media=media)
    result.loss.backward()
    assert model.vision.patch_embed.proj.weight.grad.abs().sum() > 0
    assert model.vision.merger[0].weight.grad.abs().sum() > 0
    meta = token_metadata(ids, config, media)
    _, targets = mtp_targets(ids, labels, meta, config)
    assert targets[:, : media[0]["start"] + media[0]["feature_count"]].eq(-100).all()
    assert result.mtp_tokens == int(targets.ne(-100).sum())
    opt = make_optimizer(model)
    assert sum(len(g["params"]) for g in opt.param_groups) == sum(
        p.requires_grad for p in model.parameters()
    )


def test_quantile_step_bound_and_restore(config):
    model = MiniFrontier1ForCausalLM(config)
    balance = QuantileBalance(model)
    ids = torch.randint(24, 100, (1, 12))
    with balance.capture(ids.ne(0)):
        model(ids, labels=ids)
    metrics = balance.update()
    assert metrics and max(v["max_bias_delta"] for v in metrics.values()) <= 1e-3
    copied = QuantileBalance(model)
    copied.load_state_dict(copy.deepcopy(balance.state_dict()))
    assert copied.updates == balance.updates


def test_mrope_axes_change_only_their_frequency_pairs(config):
    x = torch.ones(1, 2, 8)
    pos = torch.zeros(3, 1, dtype=torch.long)
    original = mrope(x, pos, config.mrope_sections, config.rope_theta)
    pos[1] = 3
    altered = mrope(x, pos, config.mrope_sections, config.rope_theta)
    assert torch.equal(original[..., :4], altered[..., :4])
    assert not torch.equal(original[..., 4:6], altered[..., 4:6])
    assert torch.equal(original[..., 6:], altered[..., 6:])
