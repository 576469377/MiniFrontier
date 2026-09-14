"""Equivalent model-local work reduction: unchanged routing, gradients and rotary arithmetic."""

import copy
import importlib

import pytest
import torch
from test_new_backbones import tiny_kimi

from minifrontier.models.minikimik3.upstream_layers import KimiSparseMoeBlock


def source_moe_infer(model, x, ids, weights):
    result = torch.zeros_like(x, dtype=torch.float32)
    for index, expert in enumerate(model.experts):
        token, slot = torch.where(ids == index)
        value = expert(x[token]).float() * weights[token, slot, None]
        result = result.index_add(0, token, value)
    return result.to(x.dtype)


@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("collapsed", [False, True])
def test_kimi_grouped_routing_keeps_exact_order_and_empty_expert_gradients(autocast, collapsed):
    torch.manual_seed(601)
    config = tiny_kimi(num_experts=8).upstream_config()
    model = KimiSparseMoeBlock(config)
    original = copy.deepcopy(model)
    ids = (
        torch.tensor([[1, 0]] * 13)
        if collapsed
        else torch.stack([torch.randperm(8)[:2] for _ in range(13)])
    )
    weights = torch.rand(13, 2).requires_grad_()
    reference_weights = weights.detach().clone().requires_grad_()
    x = torch.randn(13, config.routed_expert_hidden_size, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        actual = model.moe_infer(x, ids, weights)
        expected = source_moe_infer(original, reference_x, ids, reference_weights)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, reference_x.grad, rtol=0, atol=0)
    torch.testing.assert_close(weights.grad, reference_weights.grad, rtol=0, atol=0)
    for (name, value), (expected_name, reference) in zip(
        model.named_parameters(), original.named_parameters(), strict=True
    ):
        assert name == expected_name
        torch.testing.assert_close(value.grad, reference.grad, rtol=0, atol=0)
        if collapsed and name.startswith("experts.2."):
            assert value.grad is not None and torch.count_nonzero(value.grad) == 0


def test_kimi_routing_removes_per_expert_nonzero_calls():
    model = KimiSparseMoeBlock(tiny_kimi(num_experts=8).upstream_config())
    x, ids, weights = torch.randn(11, 16), torch.tensor([[1, 0]] * 11), torch.rand(11, 2)
    counts = []
    for run in (source_moe_infer, lambda model, *args: model.moe_infer(*args)):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            run(model, x, ids, weights)
        counts.append({item.key: item.count for item in profile.key_averages()})
    assert counts[0]["aten::nonzero"] == 8
    assert counts[1].get("aten::nonzero", 0) == 0
    assert counts[1].get("aten::bincount", 0) == 0


@pytest.mark.parametrize("family", ["minifrontier1", "minifrontier11"])
@pytest.mark.parametrize("multiaxis", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rotary_reused_trig_values_keep_forward_and_input_gradient(family, multiaxis, dtype):
    module = importlib.import_module("minifrontier.models." + family + ".indexer")
    torch.manual_seed(93)
    x = torch.randn(2, 7, 2, 8).to(dtype).requires_grad_()
    other = x.detach().clone().requires_grad_()
    theta = 10000.0
    if multiaxis:
        positions = torch.arange(7).expand(3, 2, 7).clone()
        positions[1] *= 2
        freq = theta ** (-torch.arange(0, 8, 2, dtype=torch.float32) / 8)
        angles = [
            positions[axis].float()[..., None] * freq[start:stop]
            for axis, start, stop in [(0, 0, 2), (1, 2, 3), (2, 3, 4)]
        ]
        angle = torch.cat(angles, -1).unsqueeze(-2)
        a, b = other.float().unflatten(-1, (-1, 2)).unbind(-1)
        expected = (
            torch.stack((a * angle.cos() - b * angle.sin(), a * angle.sin() + b * angle.cos()), -1)
            .flatten(-2)
            .to(dtype)
        )
        actual = module.mrope(x, positions, (2, 1, 1), theta)
    else:
        positions = torch.arange(7).expand(2, 7)
        angle = (
            positions.float()[..., None] * theta ** (-torch.arange(0, 4, 2).float() / 4)
        ).unsqueeze(-2)
        a, b = other[..., -4:].float().unflatten(-1, (-1, 2)).unbind(-1)
        rotated = torch.stack(
            (a * angle.cos() - b * angle.sin(), a * angle.sin() + b * angle.cos()), -1
        ).flatten(-2)
        expected = torch.cat((other[..., :-4], rotated.to(dtype)), -1)
        actual = module.rope(x, positions, 4, theta)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.float().square().sum().backward()
    expected.float().square().sum().backward()
    torch.testing.assert_close(x.grad, other.grad, rtol=0, atol=0)
