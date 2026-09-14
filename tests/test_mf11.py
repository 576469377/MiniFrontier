"""MF1.1 version isolation, V4.1 residual equations and existing media/cache contracts."""

import copy
import hashlib
import json
from dataclasses import asdict, replace

import pytest
import torch
from PIL import Image
from torch import nn

from minifrontier.models.minifrontier1 import (
    MiniFrontier1Cache,
    MiniFrontier1Config,
    MiniFrontier1ForCausalLM,
    MiniFrontier11ForCausalLM,
)
from minifrontier.models.minifrontier1.configuration import MF1_VERSION, MF11_VERSION
from minifrontier.models.minifrontier1.modeling import MF11DecoderLayer
from minifrontier.models.minifrontier1.processing import process_frames
from minifrontier.models.minifrontier1.residual import SinglePassMHC
from minifrontier.training.minifrontier1_strategy import (
    MF11_PHASES,
    PHASES,
    budget_report,
    model_name_for,
    phases_for,
)


def tiny():
    return MiniFrontier1Config.tiny(model_version=MF11_VERSION)


def test_versioned_config_preserves_legacy_serialization_and_rejects_wrong_backbone():
    old = MiniFrontier1Config.tiny()
    encoded = json.dumps(asdict(old), sort_keys=True, separators=(",", ":")).encode()
    assert (
        hashlib.sha256(encoded).hexdigest()
        == "1d0caba7ccb253a4ed79f29950bb9cb76a73d8e63f348e954dce442fb93e0592"
    )
    new = MiniFrontier1Config.v11()
    assert new.model_version == MF11_VERSION and not new.mtp_enabled and new.mtp_loss_coef == 0
    assert asdict(new).keys() == asdict(MiniFrontier1Config()).keys()
    with pytest.raises(ValueError, match="mtp_enabled"):
        MiniFrontier1Config(model_version=MF11_VERSION)
    with pytest.raises(ValueError, match="unknown"):
        replace(old, model_version="1.2")
    with pytest.raises(ValueError, match="version differ"):
        MiniFrontier1ForCausalLM(tiny())
    with pytest.raises(ValueError, match="version differ"):
        MiniFrontier11ForCausalLM(old)
    model = MiniFrontier11ForCausalLM(tiny())
    with pytest.raises(ValueError, match="own complete"):
        model.load_state_dict(MiniFrontier1ForCausalLM(old).state_dict(), strict=False)
    model.load_state_dict(copy.deepcopy(model.state_dict()))


def _source_coefficients(x, fn, scale, base, eps):
    """Independent transcription of upstream Block + hc_split_sinkhorn equations."""
    streams = x.shape[-2]
    flat = x.flatten(-2).float()
    scores = (flat @ fn.T) / (flat.square().mean(-1, keepdim=True) + eps).sqrt()
    pre = (scores[..., :streams] * scale[0] + base[:streams]).sigmoid() + 1e-6
    post = (
        2 * (scores[..., streams : 2 * streams] * scale[1] + base[streams : 2 * streams]).sigmoid()
    )
    matrix = (scores[..., 2 * streams :] * scale[2] + base[2 * streams :]).reshape(
        *x.shape[:-2], streams, streams
    ).softmax(-1) + 1e-6
    matrix = matrix / (matrix.sum(-2, keepdim=True) + 1e-6)
    for _ in range(19):
        matrix = matrix / (matrix.sum(-1, keepdim=True) + 1e-6)
        matrix = matrix / (matrix.sum(-2, keepdim=True) + 1e-6)
    return pre, post, matrix


def test_single_pass_coefficients_equation_orientation_and_gradients():
    torch.manual_seed(18)
    mhc = SinglePassMHC(tiny())
    x = torch.randn(2, 3, 4, 32, requires_grad=True)
    update = torch.randn(2, 3, 32, requires_grad=True)
    pre, post, matrix = mhc.coefficients(x)
    expected = _source_coefficients(x, mhc.fn, mhc.scale, mhc.base, mhc.norm_eps)
    for actual, wanted in zip((pre, post, matrix), expected, strict=True):
        torch.testing.assert_close(actual, wanted, atol=2e-7, rtol=2e-6)
    torch.testing.assert_close(matrix.sum(-2), torch.ones_like(pre), atol=2e-6, rtol=0)
    # The published finite 20-iteration normalization is approximately doubly stochastic.
    torch.testing.assert_close(matrix.sum(-1), torch.ones_like(pre), atol=3e-5, rtol=0)
    wanted = (
        torch.einsum("bsij,bsid->bsjd", expected[2], x)
        + expected[1][..., None] * update[..., None, :]
    )
    actual = mhc.inject(x, update, post, matrix)
    torch.testing.assert_close(actual, wanted)
    parameters = (x, update, mhc.fn, mhc.base, mhc.scale)
    oracle_grad = torch.autograd.grad(wanted.square().sum(), parameters, retain_graph=True)
    actual_grad = torch.autograd.grad(actual.square().sum(), parameters)
    for actual, wanted in zip(actual_grad, oracle_grad, strict=True):
        torch.testing.assert_close(actual, wanted, atol=2e-5, rtol=2e-5)


def test_mixer_is_shifted_between_sublayers_and_the_next_layer():
    class CaptureAttention(nn.Module):
        def forward(self, x, metadata, state=None, *, cache_output=True):
            self.input = x
            return x * 0, state, x.sum() * 0, 0

    class CaptureMoE(nn.Module):
        def forward(self, x, indices=None):
            self.input = x
            return x * 0

    layer = MF11DecoderLayer(tiny(), "qsa_mla")
    layer.attention = CaptureAttention()
    layer.moe = CaptureMoE()
    x = torch.randn(1, 3, 4, 32)
    incoming = SinglePassMHC.identity(x)
    attn_pre, post, matrix = layer.attention_mhc.coefficients(x)
    after_attention = SinglePassMHC.inject(x, torch.zeros(1, 3, 32), post, matrix)
    expected_next = layer.moe_mhc.coefficients(after_attention)[0]
    _, next_pre, _, _, _ = layer(x, incoming, {})
    torch.testing.assert_close(layer.attention.input, layer.attention_norm(x[..., 0, :]))
    torch.testing.assert_close(
        layer.moe.input, layer.moe_norm(SinglePassMHC.mix(after_attention, attn_pre))
    )
    torch.testing.assert_close(next_pre, expected_next)
    assert not torch.allclose(
        layer.attention.input, layer.attention_norm(SinglePassMHC.mix(x, attn_pre))
    )


@pytest.mark.parametrize("video", [False, True])
@pytest.mark.parametrize("phase", ["dense_pretrain", "sparse_cpt"])
def test_native_media_gradients_cache_and_rollback(video, phase):
    c = tiny()
    frames = [
        Image.new("RGB", (16, 16), color) for color in (["red", "blue"] if video else ["red"])
    ]
    span = process_frames(
        frames,
        patch_size=c.vision_config.patch_size,
        max_features=8,
        timestamps=[0.0, 0.5] if video else None,
    )
    span.update(start=2, batch_index=0, resource_kind="video" if video else "image")
    end = 2 + span["feature_count"]
    ids = torch.tensor([[1, 9, *([7] * span["feature_count"]), 10, 30, 31, 32, 33]])
    labels = ids.clone()
    labels[:, : end + 1] = -100
    model = MiniFrontier11ForCausalLM(c, phase)
    output = model(ids, labels=labels, media=[span])
    output.loss.backward()
    assert model.mtp is None and output.mtp_loss is None and output.mtp_tokens == 0
    assert model.vision.patch_embed.proj.weight.grad.abs().sum() > 0
    assert model.layers[0].attention_mhc.fn.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.eval()
    with torch.no_grad():
        full = model(ids, media=[span]).logits
        cache = MiniFrontier1Cache()
        prefix = model(ids[:, :end], media=[span], cache=cache).logits
        saved = cache.snapshot()
        streamed = [prefix]
        for i in range(end, ids.shape[1]):
            streamed.append(model(ids[:, i : i + 1], cache=cache).logits)
        torch.testing.assert_close(torch.cat(streamed, 1), full, atol=2e-6, rtol=2e-5)
        cache.restore(saved)
        torch.testing.assert_close(
            model(ids[:, end:], cache=cache).logits, full[:, end:], atol=2e-6, rtol=2e-5
        )


def test_checkpointed_gradient_and_packed_causality():
    c = tiny()
    model = MiniFrontier11ForCausalLM(c)
    checked = MiniFrontier11ForCausalLM(replace(c, gradient_checkpointing=True))
    checked.load_state_dict(model.state_dict())
    ids = torch.tensor([[1, 30, 31, 32, 1, 33, 34, 35]])
    segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
    a = model(ids, labels=ids, segment_ids=segments)
    b = checked(ids, labels=ids, segment_ids=segments)
    a.loss.backward()
    b.loss.backward()
    torch.testing.assert_close(a.logits, b.logits)
    for (name, left), (_, right) in zip(
        model.named_parameters(), checked.named_parameters(), strict=True
    ):
        if left.grad is None:
            assert right.grad is None, name
        else:
            torch.testing.assert_close(left.grad, right.grad, atol=2e-6, rtol=2e-5, msg=name)
    model.eval()
    with torch.no_grad():
        changed = ids.clone()
        changed[:, :4] = torch.tensor([1, 40, 41, 42])
        torch.testing.assert_close(
            a.logits[:, 4:],
            model(changed, segment_ids=segments).logits[:, 4:],
            atol=2e-6,
            rtol=2e-5,
        )
        changed = ids.clone()
        changed[:, -2:] = torch.tensor([40, 41])
        torch.testing.assert_close(
            a.logits[:, :-2],
            model(changed, segment_ids=segments).logits[:, :-2],
            atol=2e-6,
            rtol=2e-5,
        )


def test_versioned_recipe_and_semantic_optimizer_ownership():
    assert phases_for(MF1_VERSION) is PHASES and phases_for(MF11_VERSION) is MF11_PHASES
    assert MF11_PHASES is not PHASES and MF11_PHASES["p0"] is not PHASES["p0"]
    assert budget_report(MF11_VERSION)["main_ce_tokens"] == 3_000_000_000
    assert MF11_PHASES["p0"]["budget"] == 200_000_000
    assert MF11_PHASES["p2"]["predecessor"] == ["indexer"]
    assert model_name_for(MF11_VERSION) == "minifrontier11"
    model = MiniFrontier11ForCausalLM(tiny())
    specs = model.optimizer_metadata()
    assert set(specs) == set(dict(model.named_parameters()))
    assert specs["embed_tokens.weight"]["kind"] == "sinkhorn"
    assert specs["layers.0.attention.core.q_proj.weight"]["heads"] == 2
    assert "heads" not in specs["layers.0.attention.core.v_proj.weight"]
    assert "heads" not in specs["layers.3.attention.kv_up.weight"]
    assert specs["vision.blocks.0.attn.qkv.weight"]["blocks"][-1] == (48, 72)
    from minifrontier.training.minifrontier1_optim import make_optimizer

    opt = make_optimizer(model, kind="v41_muon_sinkhorn")
    ids = torch.tensor([[1, 30, 31, 32]])
    model(ids, labels=ids).loss.backward()
    opt.step()
    assert torch.isfinite(model.lm_head.weight).all()
    assert "momentum_buffer" in opt.state[model.lm_head.weight]
    assert "exp_avg_sq" not in opt.state[model.lm_head.weight]


def test_cli_recipe_and_parameter_count_select_explicit_version(capsys):
    from minifrontier.commands.minifrontier1 import main

    main(["recipe", "--model-version", "1.1"])
    recipe = json.loads(capsys.readouterr().out)
    assert recipe["model_name"] == "minifrontier11"
    assert recipe["phases"]["p0"]["budget"] == 200_000_000
    assert recipe["draft_supported"] is False
    main(["params", "--model-version", "1.1"])
    parameters = json.loads(capsys.readouterr().out)
    assert parameters["config"]["model_version"] == MF11_VERSION
    assert parameters["categories"]["mtp"] == 0
    assert parameters["categories"]["mhc"] > 0


def test_cpu_training_resume_keeps_mhc_optimizer_router_and_rng(tmp_path, monkeypatch):
    from minifrontier.data.minifrontier1 import make_fixture
    from minifrontier.training import minifrontier1_strategy as strategy
    from minifrontier.training.minifrontier1 import train

    # This checks checkpoint continuity on a mechanism fixture, not formal admission.
    monkeypatch.setenv("MINIFRONTIER_MIN_FREE_GIB", "1")
    monkeypatch.setattr(strategy, "MF11_PLAN_PATH", strategy.PLAN_PATH)
    monkeypatch.setattr(strategy, "source_identity", lambda: dict(commit="unit-test", dirty=False))
    make_fixture(tmp_path / "data")
    args = dict(
        data=tmp_path / "data",
        config=asdict(tiny()),
        phase="pilot",
        steps=2,
        input_batch_tokens=32,
        save_every=2,
        eval_every=2,
    )
    train(**args, output=tmp_path / "continuous")
    train(**args, output=tmp_path / "resumed", stop_after_updates=1)
    train(**args, output=tmp_path / "resumed", resume=tmp_path / "resumed/checkpoint.pt")
    first, second = [
        torch.load(tmp_path / name / "checkpoint.pt", weights_only=True)
        for name in ("continuous", "resumed")
    ]

    def assert_same(a, b):
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                assert_same(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b)
            for x, y in zip(a, b, strict=True):
                assert_same(x, y)
        else:
            assert a == b

    for key in ("model", "optimizer", "sampler", "router_balance", "ledger", "rng"):
        assert_same(first[key], second[key])
    assert first["model_name"] == "minifrontier11"
    assert first["run_spec"]["optimizer_kind"] == "v41_muon_sinkhorn"
    assert first["config"]["model_version"] == MF11_VERSION
    assert first["ledger"]["mtp_target_tokens"] == 0
    with pytest.raises(ValueError, match="checkpoint model/config/tokenizer"):
        train(
            **dict(args, config=asdict(MiniFrontier1Config.tiny())),
            output=tmp_path / "wrong_version",
            init=tmp_path / "resumed/checkpoint.pt",
        )
    from minifrontier.inference.minifrontier1 import respond
    from minifrontier.inference.minifrontier1_export import export_checkpoint
    from minifrontier.inference.runtime import load_checkpoint
    from minifrontier.training.minifrontier1_posttrain import train_post

    checkpoint_path = tmp_path / "resumed/checkpoint.pt"
    generated = respond(checkpoint_path, "hello", max_new_tokens=1)
    assert len(generated["output_ids"]) == 1
    export_checkpoint(checkpoint_path, tmp_path / "export")
    restored, _, saved = load_checkpoint(tmp_path / "export/model.pt")
    assert type(restored) is MiniFrontier11ForCausalLM
    assert saved["model_name"] == "minifrontier11"
    for name, parameter in restored.state_dict().items():
        assert torch.equal(parameter, second["model"][name])
    with pytest.raises(ValueError, match="separately trained drafter"):
        train_post(
            phase="draft",
            checkpoint=checkpoint_path,
            data=tmp_path / "data",
            output=tmp_path / "unsupported_draft",
            steps=1,
        )
