"""Invisible-KV removal against each model's unchanged dense attention oracle."""

import copy
import importlib
from contextlib import nullcontext
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F


def attention(package, kind):
    module = importlib.import_module(f"minifrontier.models.{package}")
    config_class = getattr(
        module, "MiniFrontier1Config" if package == "minifrontier1" else "MiniFrontier11Config"
    )
    config = replace(config_class.tiny(), query_chunk_size=5, window_size=7)
    implementation = importlib.import_module(f"minifrontier.models.{package}.{kind}")
    cls = implementation.CSA if kind == "csa" else implementation.QSAMLA
    return cls(config), config


def metadata(length, scenario):
    segments = torch.zeros(2, length, dtype=torch.long)
    if scenario in {"packed_media", "padding"}:
        segments[0, 9:22] = 1
        segments[0, 22:] = -1
        segments[1, 3:14] = 1
        segments[1, 14:] = 2
    if scenario == "padding":
        segments[0] = -1
        segments[1, 17:] = -1
    if scenario == "all_padding":
        segments[:] = -1
    modality = torch.zeros_like(segments)
    media = torch.full_like(segments, -1)
    if scenario == "packed_media":
        modality[0, 6:12] = 1
        media[0, 6:9], media[0, 9:12] = 0, 1
        modality[1, 2:10] = 2
        media[1, 2:6], media[1, 6:10] = 0, 1
    # RoPE coordinates differ from causal token positions at segment/media edges.
    linear = torch.arange(length).expand_as(segments).clone()
    for row in range(2):
        for segment in segments[row].unique().tolist():
            selected = segments[row] == segment
            linear[row, selected] -= linear[row, selected].min()
    positions = linear[None].repeat(3, 1, 1)
    positions[1][modality.ne(0)] %= 3
    positions[2][modality.ne(0)] //= 3
    return dict(
        segment_ids=segments,
        modality=modality,
        media_ids=media,
        linear_positions=linear,
        position_ids=positions,
        unpacked_prefill=False,
    )


def assert_gradients(actual, expected, *, bf16, name):
    assert torch.isfinite(actual).all(), name
    if bf16:
        # Slicing per chunk changes BF16 accumulation grouping at the shared
        # KV projection. Cancellation-near-zero elements need a norm-based bound.
        error = (actual - expected).float()
        expected = expected.float()
        assert error.norm() <= 0.01 * expected.norm() + 1e-7, name
        assert error.abs().max() <= 0.02 * expected.abs().max() + 1e-6, name
    else:
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6, msg=name)


@pytest.mark.parametrize("package", ["minifrontier1", "minifrontier11"])
@pytest.mark.parametrize("kind", ["csa", "qsa_mla"])
@pytest.mark.parametrize("scenario", ["text", "packed_media", "padding", "all_padding"])
@pytest.mark.parametrize("bf16", [False, True])
def test_trimmed_forward_and_all_gradients_match_reference(package, kind, scenario, bf16):
    torch.manual_seed(551)
    reference, config = attention(package, kind)
    reference.dense_prefill_backend = "reference"
    trimmed = copy.deepcopy(reference)
    trimmed.dense_prefill_backend = "trimmed"
    x = torch.randn(2, 29, config.hidden_size, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    info = metadata(29, scenario)
    multiplier = torch.randn_like(x)
    autocast = torch.autocast("cpu", dtype=torch.bfloat16) if bf16 else nullcontext()
    with autocast:
        expected = reference.dense_prefill(x, copy.deepcopy(info))
        actual = trimmed.dense_prefill(y, copy.deepcopy(info))
    (expected.float() * multiplier).sum().backward()
    (actual.float() * multiplier).sum().backward()
    tolerance = dict(rtol=0.01, atol=0.002) if bf16 else dict(rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(actual, expected, **tolerance)
    assert_gradients(y.grad, x.grad, bf16=bf16, name="input")
    for (name, first), (other, second) in zip(
        reference.named_parameters(), trimmed.named_parameters(), strict=True
    ):
        assert name == other
        if first.grad is None:
            assert second.grad is None
        else:
            assert_gradients(second.grad, first.grad, bf16=bf16, name=name)


@pytest.mark.parametrize("package", ["minifrontier1", "minifrontier11"])
@pytest.mark.parametrize("kind", ["csa", "qsa_mla"])
def test_trimmed_causality_and_packed_boundary_isolation(package, kind):
    torch.manual_seed(667)
    model, config = attention(package, kind)
    model.dense_prefill_backend = "trimmed"
    x = torch.randn(2, 29, config.hidden_size)
    info = metadata(29, "packed_media")
    with torch.no_grad():
        expected = model.dense_prefill(x, copy.deepcopy(info))
        future = x.clone()
        future[:, 18:] += 20 * torch.randn_like(future[:, 18:])
        actual = model.dense_prefill(future, copy.deepcopy(info))
        torch.testing.assert_close(actual[:, :18], expected[:, :18], rtol=1e-5, atol=1e-6)
        other_segment = x.clone()
        other_segment[0, :9] += 20 * torch.randn_like(other_segment[0, :9])
        actual = model.dense_prefill(other_segment, copy.deepcopy(info))
        torch.testing.assert_close(actual[0, 9:22], expected[0, 9:22], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("package", ["minifrontier1", "minifrontier11"])
@pytest.mark.parametrize("kind", ["csa", "qsa_mla"])
def test_default_reference_and_actual_key_reduction(package, kind, monkeypatch):
    monkeypatch.delenv("MINIFRONTIER_MF_DENSE_PREFILL", raising=False)
    model, config = attention(package, kind)
    assert model.dense_prefill_backend == "reference"
    calls = []
    original = F.scaled_dot_product_attention

    def record(q, k, v, *args, **kwargs):
        calls.append(k.shape[-2])
        return original(q, k, v, *args, **kwargs)

    monkeypatch.setattr(F, "scaled_dot_product_attention", record)
    x, info = torch.randn(2, 29, config.hidden_size), metadata(29, "packed_media")
    with torch.no_grad():
        model.dense_prefill(x, copy.deepcopy(info))
        reference = list(calls)
        calls.clear()
        model.dense_prefill_backend = "trimmed"
        model.dense_prefill(x, copy.deepcopy(info))
    assert len(calls) == len(reference)
    assert all(a <= b for a, b in zip(calls, reference, strict=True))
    assert sum(calls) < sum(reference)


@pytest.mark.parametrize("package", ["minifrontier1", "minifrontier11"])
def test_csa_completion_cache_handles_inference_and_metadata_changes(package):
    model, config = attention(package, "csa")
    model.dense_prefill_backend = "trimmed"
    build = importlib.import_module(f"minifrontier.models.{package}.indexer").prefill_directory
    with torch.inference_mode():
        info = metadata(29, "packed_media")
        info["prefill_directory"] = build(info)
        output = model.dense_prefill(torch.randn(2, 29, config.hidden_size), info)
        assert torch.isfinite(output).all()
    info = metadata(29, "packed_media")
    directory = build(info)
    first = model._dense_key_limits(info, directory, 29)
    assert first is model._dense_key_limits(info, directory, 29)
    directory["complete"].fill_(2**60)
    assert model._dense_key_limits(info, directory, 29) == [0] * len(first)


@pytest.mark.parametrize("package", ["minifrontier1", "minifrontier11"])
def test_qsa_unpacked_flash_path_is_unchanged(package):
    model, config = attention(package, "qsa_mla")
    x, info = torch.randn(2, 29, config.hidden_size), metadata(29, "text")
    info["unpacked_prefill"] = True
    with torch.no_grad():
        model.dense_prefill_backend = "reference"
        expected = model.dense_prefill(x, info)
        model.dense_prefill_backend = "trimmed"
        actual = model.dense_prefill(x, info)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
