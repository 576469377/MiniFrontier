"""Native vision shape, source math, checkpoint gradients and language insertion."""

import copy

import pytest
import torch
from test_miniqwen4 import tiny_config
from test_new_backbones import tiny_kimi

from minifrontier.models.minideepseekv4.vision import DeepSeekVision, DeepSeekVisionConfig
from minifrontier.models.minikimik3 import MiniKimiK3ForCausalLM
from minifrontier.models.minikimik3.upstream_vision import tpool_patch_merger
from minifrontier.models.minikimik3.vision import KimiVision, KimiVisionConfig
from minifrontier.models.miniqwen4 import MiniQwen4ForCausalLM
from minifrontier.models.miniqwen4.vision import QwenVision, QwenVisionConfig
from minifrontier.training.kimi_qk_clip import KimiQKClip
from minifrontier.training.losses import causal_lm_loss, chunked_linear_ce


@pytest.mark.parametrize("family", ["kimi", "qwen", "deepseek"])
def test_native_vision_checkpointing_and_parameter_gradients(family):
    torch.manual_seed(71)
    if family == "kimi":
        m = KimiVision(
            KimiVisionConfig(
                depth=2,
                hidden_size=32,
                qkv_hidden_size=48,
                num_heads=2,
                intermediate_size=64,
                output_size=32,
            )
        )
        args = (torch.randn(16, 3, 14, 14), torch.tensor([[1, 4, 4]]))
    elif family == "qwen":
        m = QwenVision(
            QwenVisionConfig(
                depth=2, hidden_size=32, num_heads=2, intermediate_size=64, output_size=32
            )
        )
        args = (torch.randn(16, 3 * 2 * 16 * 16), torch.tensor([[1, 4, 4]]))
    else:
        m = DeepSeekVision(
            DeepSeekVisionConfig(
                depth=2, hidden_size=32, num_heads=2, intermediate_size=64, output_size=32
            )
        )
        args = (torch.randn(16, 3, 14, 14), 4, 4)
    reference = copy.deepcopy(m)
    reference.config.gradient_checkpointing = False
    a, b = m(*args), reference(*args)
    if isinstance(a, (list, tuple)):
        a, b = torch.cat(tuple(a)), torch.cat(tuple(b))
    torch.testing.assert_close(a, b)
    a.square().mean().backward()
    b.square().mean().backward()
    for (name, p), q in zip(m.named_parameters(), reference.parameters(), strict=True):
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        torch.testing.assert_close(p.grad, q.grad)
    if family == "kimi":
        oracle = reference.merger(
            tpool_patch_merger(reference.encoder(reference.patch_embed(*args), args[1]), args[1])
        )
        torch.testing.assert_close(a, torch.cat(tuple(oracle)))
    elif family == "deepseek":
        torch.testing.assert_close(a, reference.aligner(reference.vit(*args), 4, 4))


@pytest.mark.parametrize("family", ["kimi", "qwen"])
def test_visual_language_inputs_are_causal_aligned_and_used(family):
    torch.manual_seed(5)
    if family == "kimi":
        vision = KimiVisionConfig(
            depth=1,
            hidden_size=32,
            qkv_hidden_size=48,
            num_heads=2,
            intermediate_size=64,
            output_size=32,
        )
        m = MiniKimiK3ForCausalLM(tiny_kimi(vision_config=vision))
        pixels = torch.randn(16, 3, 14, 14)
    else:
        vision = QwenVisionConfig(
            depth=1, hidden_size=32, num_heads=2, intermediate_size=64, output_size=32
        )
        m = MiniQwen4ForCausalLM(tiny_config(hidden_size=32, vision_config=vision))
        pixels = torch.randn(16, 3 * 2 * 16 * 16)
    ids = torch.full((1, 12), 15, dtype=torch.long)
    ids[:, 3:7] = 7
    labels = ids.clone()
    labels[:, :8] = -100
    media = [
        dict(
            batch_index=0,
            start=3,
            patches=pixels,
            grid_thw=torch.tensor([[1, 4, 4]]),
            feature_count=4,
        )
    ]
    m.eval()
    result = m(ids, labels=labels, media=media)
    other = m(ids, media=[dict(media[0], patches=-pixels)])
    torch.testing.assert_close(result.logits[:, :3], other.logits[:, :3])
    assert (result.logits[:, 8:] - other.logits[:, 8:]).abs().max() > 1e-5
    result.loss.backward()
    assert m.vision.patch_embed.proj.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match=r"missing|mismatch"):
        m(ids, media=[])
    with pytest.raises(ValueError, match="targets"):
        m(ids, labels=ids, media=media)


@pytest.mark.parametrize("shift", [True, False])
@pytest.mark.parametrize("chunk_size", [3, 128, 512])
def test_chunked_lm_head_has_full_ce_value_and_gradient(monkeypatch, shift, chunk_size):
    torch.manual_seed(16)
    h = torch.randn(2, 11, 8, requires_grad=True)
    w = torch.randn(21, 8, requires_grad=True)
    labels = torch.randint(0, 21, (2, 11))
    labels[0, :9] = -100
    dense = (
        causal_lm_loss(h @ w.T, labels)
        if shift
        else torch.nn.functional.cross_entropy((h @ w.T).reshape(-1, 21), labels.reshape(-1))
    )
    expected = torch.autograd.grad(dense, (h, w))
    monkeypatch.setenv("MINIFRONTIER_CE_CHUNK_SIZE", str(chunk_size))
    chunked = chunked_linear_ce(h, w, labels, shift=shift)
    actual = torch.autograd.grad(chunked, (h, w))
    torch.testing.assert_close(chunked, dense)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b)


def test_chunked_lm_head_explicit_size_overrides_environment(monkeypatch):
    monkeypatch.setenv("MINIFRONTIER_CE_CHUNK_SIZE", "0")
    h, w, labels = torch.randn(1, 3, 4), torch.randn(8, 4), torch.ones(1, 3, dtype=torch.long)
    with pytest.raises(ValueError, match="invalid"):
        chunked_linear_ce(h, w, labels)
    torch.testing.assert_close(
        chunked_linear_ce(h, w, labels, chunk_size=2), causal_lm_loss(h @ w.T, labels)
    )


def test_qk_clip_nonshared_rows_and_resume_state():
    m = MiniKimiK3ForCausalLM(tiny_kimi())
    clip = KimiQKClip(m)
    ids = torch.randint(10, 60, (1, 12))
    with clip.capture(torch.ones_like(ids, dtype=torch.bool)):
        loss = m(ids, labels=ids).loss
    maxima = copy.deepcopy(clip.state_dict())
    loss.backward()
    for name, value in clip.maxima.items():
        torch.testing.assert_close(value, maxima["maxima"][name])
    name, attn = clip.layers[0]
    before = copy.deepcopy(attn.state_dict())
    clip.maxima[name].fill_(400)
    clip.update()
    torch.testing.assert_close(attn.q_b_proj.weight, before["q_b_proj.weight"] * 0.5)
    torch.testing.assert_close(attn.kv_a_proj_with_mqa.weight, before["kv_a_proj_with_mqa.weight"])
    old = before["kv_b_proj.weight"].view(
        attn.num_heads, attn.qk_nope_head_dim + attn.v_head_dim, -1
    )
    new = attn.kv_b_proj.weight.view_as(old)
    torch.testing.assert_close(
        new[:, : attn.qk_nope_head_dim], old[:, : attn.qk_nope_head_dim] * 0.5
    )
    torch.testing.assert_close(new[:, attn.qk_nope_head_dim :], old[:, attn.qk_nope_head_dim :])
