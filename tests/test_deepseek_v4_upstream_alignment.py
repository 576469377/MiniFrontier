"""Independent pinned-source oracle, limited to unquantized single-rank experts."""

import ast
import hashlib
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from minifrontier.models.minideepseekv4.expert import ClampedSwiGLU
from minifrontier.models.minideepseekv4.upstream_layers import Block


def upstream_expert():
    path = Path(__file__).resolve().parents[1] / "third_party/upstream/deepseek-v4-60d8d70/model.py"
    source = path.read_bytes()
    assert hashlib.sha256(source).hexdigest() == (
        "ce962f1face79d4f633d36436576214057a7e11443c9789935e1deb5c6cd1d71"
    )
    nodes = ast.parse("from __future__ import annotations").body
    nodes.extend(
        n
        for n in ast.parse(source).body
        if getattr(n, "name", None) in {"linear", "Linear", "Expert"}
    )
    namespace = {"torch": torch, "nn": nn, "F": F, "default_dtype": torch.float32}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["Expert"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("weighted", [False, True])
def test_expert_forward_and_all_gradients(dtype, weighted):
    torch.manual_seed(97)
    local = ClampedSwiGLU(16, 24, limit=10).to(dtype)
    official = upstream_expert()(16, 24, dtype=dtype, swiglu_limit=10)
    with torch.no_grad():
        official.w1.weight.copy_(local.gate.weight)
        official.w2.weight.copy_(local.down.weight)
        official.w3.weight.copy_(local.up.weight)
    # Large inputs exercise both sides of the up clamp and the gate upper cap.
    x = (torch.randn(7, 16, dtype=dtype) * 20).requires_grad_()
    y = x.detach().clone().requires_grad_()
    weight = torch.rand(7, 1, requires_grad=True) if weighted else None
    reference_weight = weight.detach().clone().requires_grad_() if weighted else None
    actual = local(x, weight)
    expected = official(y, reference_weight)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.float().square().mean().backward()
    expected.float().square().mean().backward()
    pairs = [(x.grad, y.grad)]
    pairs.extend(
        (a.weight.grad, b.weight.grad)
        for a, b in ((local.gate, official.w1), (local.down, official.w2), (local.up, official.w3))
    )
    if weighted:
        pairs.append((weight.grad, reference_weight.grad))
    for a, b in pairs:
        assert a is not None and torch.isfinite(a).all()
        torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.parametrize("streams", [2, 4])
@pytest.mark.parametrize("autocast", [False, True])
def test_hc_post_contraction_matches_official_forward_and_backward(streams, autocast):
    path = Path(__file__).resolve().parents[1] / "third_party/upstream/deepseek-v4-60d8d70/model.py"
    source = path.read_bytes()
    assert hashlib.sha256(source).hexdigest() == (
        "ce962f1face79d4f633d36436576214057a7e11443c9789935e1deb5c6cd1d71"
    )
    cls = next(node for node in ast.parse(source).body if getattr(node, "name", None) == "Block")
    method = next(node for node in cls.body if getattr(node, "name", None) == "hc_post")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    official = namespace["hc_post"]
    torch.manual_seed(491)
    dtype = torch.bfloat16 if autocast else torch.float32
    output = torch.randn(2, 5, 32, dtype=dtype, requires_grad=True)
    residual = torch.randn(2, 5, streams, 32, dtype=dtype, requires_grad=True)
    post = torch.randn(2, 5, streams, requires_grad=True)
    comb = torch.randn(2, 5, streams, streams, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        expected = official(None, output, residual, post, comb)
        actual = Block.hc_post(None, output, residual, post, comb)
    assert actual.dtype == dtype
    tolerance = dict(atol=1e-5, rtol=0.008) if autocast else dict(atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(actual, expected, **tolerance)
    inputs = (output, residual, post, comb)
    original_grads = torch.autograd.grad(expected.float().square().sum(), inputs)
    actual_grads = torch.autograd.grad(actual.float().square().sum(), inputs)
    grad_tolerance = dict(atol=2e-3, rtol=0.01) if autocast else dict(atol=2e-5, rtol=2e-5)
    for a, b in zip(actual_grads, original_grads, strict=True):
        torch.testing.assert_close(a, b, **grad_tolerance)
