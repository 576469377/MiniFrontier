"""Routing prepasses omit unused losses without changing the next training update."""

import copy

import pytest
import torch
from PIL import Image
from test_miniqwen4 import tiny_config

from minifrontier.models.miniqwen4 import MiniQwen4ForCausalLM
from minifrontier.models.miniqwen4.batched_experts import LoopQwenExperts
from minifrontier.models.miniqwen4.processing import process_frames
from minifrontier.models.miniqwen4.upstream_decoder import Qwen4ExpTextExperts
from minifrontier.models.miniqwen4.vision import QwenVisionConfig
from minifrontier.training.qwen_balance import QwenWindowBalance


@pytest.fixture(autouse=True)
def single_cpu_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def model_and_window(checkpointing, media_kind):
    vision = None
    if media_kind:
        vision = QwenVisionConfig(
            depth=1,
            hidden_size=16,
            num_heads=2,
            intermediate_size=32,
            patch_size=2,
            num_position_embeddings=16,
            output_size=16,
            gradient_checkpointing=checkpointing,
        )
    model = MiniQwen4ForCausalLM(
        tiny_config(
            vocab_size=64,
            mtp_enabled=True,
            gradient_checkpointing=checkpointing,
            vision_config=vision,
        )
    ).train()
    ids = torch.randint(10, 64, (2, 14))
    ids[0, 11] = 2
    ids[1, 9:] = 0
    labels = ids.masked_fill(ids.eq(0), -100)
    labels[:, :2] = -100
    media = None
    if media_kind:
        frames = [Image.new("RGB", (8, 8), "red")]
        timestamps = None
        if media_kind == "video":
            frames.append(Image.new("RGB", (8, 8), "blue"))
            timestamps = [0.0, 0.4]
        sample = process_frames(
            frames, patch_size=2, max_features=4, min_pixels=16, timestamps=timestamps
        )
        sample.update(batch_index=0, start=2)
        ids[0, 2 : 2 + sample["feature_count"]] = 7
        labels[0, 2 : 2 + sample["feature_count"]] = -100
        media = [sample]
    other = torch.tensor([[21, 22, 2, 31, 32, 33, 34, 35, 0]])
    return model, [(ids, labels, media), (other, other.masked_fill(other.eq(0), -100), None)]


def collect(model, window, *, fast, autocast):
    balance = QwenWindowBalance(model)
    predictions, routers, outputs = [], [], []
    handle = model.mtp.register_forward_hook(
        lambda _module, _args, output: predictions.append(output[0].detach().clone())
    )
    before_rng = torch.get_rng_state().clone()
    try:
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            for ids, labels, media in window:
                with balance.capture(ids.ne(0)):
                    out = model(
                        ids,
                        attention_mask=ids.ne(0),
                        labels=labels,
                        media=media,
                        return_logits=False,
                        router_prepass=fast,
                    )
                outputs.append(out)
                routers.extend(out.router_logits)
    finally:
        handle.remove()
    assert torch.equal(before_rng, torch.get_rng_state())
    balance.finalize()
    return balance, predictions, routers, outputs


@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("media_kind", [None, "image", "video"])
@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_prepass_keeps_main_mtp_routes_rng_and_following_gradients(
    checkpointing, autocast, media_kind, backend
):
    torch.manual_seed(941)
    original, window = model_and_window(checkpointing, media_kind)
    original.set_dense_attention_backend(backend)
    optimized = copy.deepcopy(original)
    old, old_predictions, old_routers, _ = collect(original, window, fast=False, autocast=autocast)
    new, new_predictions, new_routers, outputs = collect(
        optimized, window, fast=True, autocast=autocast
    )
    for key in old.totals:
        torch.testing.assert_close(old.totals[key], new.totals[key], atol=0, rtol=0)
    torch.testing.assert_close(
        original.router_window_frequency, optimized.router_window_frequency, atol=0, rtol=0
    )
    torch.testing.assert_close(
        original.mtp.router_window_frequency, optimized.mtp.router_window_frequency, atol=0, rtol=0
    )
    for actual, expected in zip(
        new_predictions + new_routers, old_predictions + old_routers, strict=True
    ):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert all(out.loss is None and out.mtp_loss is None and out.logits is None for out in outputs)
    for ids, labels, media in window:
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            expected = original(ids, ids.ne(0), labels, media=media, return_logits=False)
            actual = optimized(ids, ids.ne(0), labels, media=media, return_logits=False)
        for name in ("loss", "lm_loss", "aux_loss", "mtp_loss", "mtp_aux_loss"):
            torch.testing.assert_close(
                getattr(actual, name), getattr(expected, name), atol=0, rtol=0
            )
        (expected.loss * ids.ne(0).sum()).backward()
        (actual.loss * ids.ne(0).sum()).backward()
    old.clear()
    new.clear()
    for (name, parameter), (other, reference) in zip(
        optimized.named_parameters(), original.named_parameters(), strict=True
    ):
        assert name == other
        assert (parameter.grad is None) == (reference.grad is None), name
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, reference.grad, atol=0, rtol=0, msg=name)


@pytest.mark.parametrize("mtp_enabled, mtp_coef", [(False, 0.1), (True, 0.0), (True, 0.1)])
def test_router_prepass_never_evaluates_unused_losses(monkeypatch, mtp_enabled, mtp_coef):
    model = MiniQwen4ForCausalLM(tiny_config(mtp_enabled=mtp_enabled, mtp_loss_coef=mtp_coef))
    ids = torch.tensor([[11, 12, 2, 21, 22, 0]])
    labels = ids.masked_fill(ids.eq(0), -100)

    def forbidden(*args, **kwargs):
        raise AssertionError("routing prepass evaluated an unused loss")

    monkeypatch.setattr("minifrontier.training.losses.chunked_linear_ce", forbidden)
    monkeypatch.setattr("minifrontier.models.miniqwen4.modeling.normalized_router_loss", forbidden)
    with torch.no_grad():
        output = model(ids, ids.ne(0), labels, return_logits=False, router_prepass=True)
    assert output.loss is None and output.mtp_loss is None
    assert len(output.router_logits) == model.config.num_hidden_layers


def test_router_prepass_rejects_gradient_and_missing_label_calls():
    model = MiniQwen4ForCausalLM(tiny_config())
    ids = torch.tensor([[11, 12, 13]])
    with pytest.raises(ValueError, match="router prepass requires"):
        model(ids, labels=ids, return_logits=False, router_prepass=True)
    with torch.no_grad(), pytest.raises(ValueError, match="router prepass requires"):
        model(ids, return_logits=False, router_prepass=True)


@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("active_experts", [2, 4])
def test_loop_adapter_preserves_source_output_and_all_gradients(autocast, active_experts):
    torch.manual_seed(81)
    config = tiny_config().upstream_config()
    original = Qwen4ExpTextExperts(config)
    with torch.no_grad():
        for parameter in original.parameters():
            parameter.normal_(0, 0.1)
    optimized = LoopQwenExperts(config)
    optimized.load_state_dict(original.state_dict(), strict=True)
    ids = torch.stack(
        (torch.arange(17) % active_experts, (torch.arange(17) + 1) % active_experts), -1
    )
    x = torch.randn(17, config.hidden_size, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    weights = torch.randn(17, 2).softmax(-1).requires_grad_()
    reference_weights = weights.detach().clone().requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        actual = optimized(x, ids, weights)
        expected = original(reference_x, ids, reference_weights)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, reference_x.grad, atol=0, rtol=0)
    torch.testing.assert_close(weights.grad, reference_weights.grad, atol=0, rtol=0)
    for (name, parameter), (other, reference) in zip(
        optimized.named_parameters(), original.named_parameters(), strict=True
    ):
        assert name == other
        torch.testing.assert_close(parameter.grad, reference.grad, atol=0, rtol=0, msg=name)
