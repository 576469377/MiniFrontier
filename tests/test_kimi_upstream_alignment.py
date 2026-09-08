"""Pinned-source component conformance; not whole-model conformance."""

import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from minifrontier.models.minikimik3.attnres import _apply_attn_res


def upstream_attnres():
    root = Path(__file__).resolve().parents[1]
    path = root / "third_party/upstream/kimi-k3-c5d1dd4/modeling_kimi_linear.py"
    source = path.read_bytes()
    assert hashlib.sha256(source).hexdigest() == (
        "9e3564c70ac21854ce5a090cc946c5dc76b70d1050ef50840449181a20fff44a"
    )
    node = next(n for n in ast.parse(source).body if getattr(n, "name", None) == "_apply_attn_res")
    port = ast.parse((root / "minifrontier/models/minikimik3/attnres.py").read_text())
    port_node = next(n for n in port.body if getattr(n, "name", None) == "_apply_attn_res")
    assert ast.dump(node) == ast.dump(port_node)
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    assert namespace["_apply_attn_res"] is not _apply_attn_res
    return namespace["_apply_attn_res"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("count", [1, 3, 7])
def test_attnres_forward_and_all_gradients(dtype, count):
    torch.manual_seed(71)
    query = torch.randn(1, 16, dtype=dtype, requires_grad=True)
    norm = torch.empty(16, dtype=dtype).uniform_(0.8, 1.2).requires_grad_()
    sources = [torch.randn(2, 5, 16, dtype=dtype, requires_grad=True) for _ in range(count)]
    reference = [x.detach().clone().requires_grad_() for x in sources]
    proj_weight = query.detach().clone().requires_grad_()
    norm_weight = norm.detach().clone().requires_grad_()
    values = torch.stack(reference, 2).reshape(10, count, 16)
    expected = upstream_attnres()(
        values[:, -1],
        values[:, :-1],
        SimpleNamespace(weight=proj_weight),
        SimpleNamespace(weight=norm_weight, variance_epsilon=1e-5),
    ).reshape(2, 5, 16)
    inputs = torch.stack(sources, 2).reshape(10, count, 16)
    actual = _apply_attn_res(
        inputs[:, -1],
        inputs[:, :-1],
        SimpleNamespace(weight=query),
        SimpleNamespace(weight=norm, variance_epsilon=1e-5),
    ).reshape(2, 5, 16)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.float().square().mean().backward()
    expected.float().square().mean().backward()
    for a, b in zip(sources, reference, strict=True):
        torch.testing.assert_close(a.grad, b.grad, atol=0, rtol=0)
    for a, b in (
        (query.grad, proj_weight.grad),
        (norm.grad, norm_weight.grad),
    ):
        assert a is not None and torch.isfinite(a).all()
        torch.testing.assert_close(a, b, atol=0, rtol=0)
