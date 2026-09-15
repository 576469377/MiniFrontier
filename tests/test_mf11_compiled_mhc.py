"""Optional MF1.1 coefficient fusion retains the full FP32 Sinkhorn contract."""

import copy
from dataclasses import asdict

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from minifrontier.models.minifrontier11 import (
    MiniFrontier11Config,
    MiniFrontier11ForCausalLM,
    residual,
)


@pytest.mark.parametrize("autocast", [False, True])
def test_compiled_opt_in_keeps_cpu_reference_and_state(autocast, monkeypatch):
    torch.manual_seed(1591)
    config = MiniFrontier11Config.tiny()
    actual = residual.SinglePassMHC(config)
    expected = copy.deepcopy(actual)
    actual.backend = "compiled"

    def forbidden():
        raise AssertionError("CPU must never initialize the CUDA compiled path")

    monkeypatch.setattr(residual, "_compiled_sinkhorn", forbidden)
    dtype = torch.bfloat16 if autocast else torch.float32
    x = torch.randn(2, 7, actual.streams, config.hidden_size, dtype=dtype, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        left, right = actual.coefficients(x), expected.coefficients(reference_x)
    for a, b in zip(left, right, strict=True):
        assert a.dtype == b.dtype == torch.float32
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    sum(t.square().sum() for t in left).backward()
    sum(t.square().sum() for t in right).backward()
    torch.testing.assert_close(x.grad, reference_x.grad, rtol=0, atol=0)
    for (name, parameter), (other, original) in zip(
        actual.named_parameters(), expected.named_parameters(), strict=True
    ):
        assert name == other
        torch.testing.assert_close(parameter.grad, original.grad, rtol=0, atol=0)
    assert actual.state_dict().keys() == expected.state_dict().keys()
    assert actual.sinkhorn_iters == 20


@pytest.mark.parametrize("recompute", [False, True])
def test_compilable_twenty_round_graph_matches_mhc_outputs_and_gradients(recompute):
    # AOT eager verifies graph capture and autograd on CPU without requiring CUDA
    # or a platform C++ compiler. The real Inductor kernel needs its own GPU check.
    torch.manual_seed(1592)
    compiled = torch.compile(
        residual._sinkhorn_coefficients, backend="aot_eager", fullgraph=True, dynamic=False
    )
    for batch, length in ((2, 7), (3, 11)):
        mixed = torch.randn(batch, length, 24, requires_grad=True)
        scale = torch.randn(3, requires_grad=True)
        base = torch.randn(24, requires_grad=True)
        inputs = mixed, scale, base
        originals = tuple(t.detach().clone().requires_grad_() for t in inputs)

        def run(mixed, scale, base):
            return compiled(mixed, scale, base, 4, 1e-6, 20)

        actual = checkpoint(run, *inputs, use_reentrant=False) if recompute else run(*inputs)
        expected = residual._sinkhorn_coefficients(*originals, 4, 1e-6, 20)
        incoming = tuple(torch.randn_like(t) for t in actual)
        for a, b in zip(actual, expected, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            assert a.dtype == torch.float32 and torch.isfinite(a).all()
        actual_grads = torch.autograd.grad(
            sum((t * g).sum() for t, g in zip(actual, incoming, strict=True)), inputs
        )
        expected_grads = torch.autograd.grad(
            sum((t * g).sum() for t, g in zip(expected, incoming, strict=True)), originals
        )
        for a, b in zip(actual_grads, expected_grads, strict=True):
            torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)
            assert torch.isfinite(a).all()


def test_runtime_selection_changes_no_config_or_parameter_contract():
    model = MiniFrontier11ForCausalLM(MiniFrontier11Config.tiny())
    config, parameters = asdict(model.config), list(model.state_dict())
    modules = [m for m in model.modules() if isinstance(m, residual.SinglePassMHC)]
    assert modules and all(m.backend == "reference" for m in modules)
    model.set_mhc_backend("compiled")
    assert all(m.backend == "compiled" for m in modules)
    assert asdict(model.config) == config and list(model.state_dict()) == parameters
    model.set_mhc_backend("reference")
    assert all(m.backend == "reference" for m in modules)
    with pytest.raises(ValueError, match="mHC backend"):
        model.set_mhc_backend("unknown")


def test_compile_is_lazy_cached_and_excludes_autotuning(monkeypatch):
    calls = []

    def capture(function, **kwargs):
        calls.append(kwargs)
        return function

    residual._compiled_sinkhorn.cache_clear()
    monkeypatch.setattr(torch, "compile", capture)
    try:
        assert residual._compiled_sinkhorn() is residual._compiled_sinkhorn()
        assert len(calls) == 1
        assert calls[0]["fullgraph"] and calls[0]["dynamic"] is False
        assert calls[0]["options"] == {
            "compile_threads": 1,
            "max_autotune": False,
            "triton.cudagraphs": False,
            "use_fast_math": False,
        }
    finally:
        residual._compiled_sinkhorn.cache_clear()


@pytest.mark.parametrize("shape", [(2, 7), (3, 701), (2, 2048)])
def test_fixed_tiles_preserve_tail_outputs_and_all_gradients(shape, monkeypatch):
    torch.manual_seed(1594)
    calls = []

    def eager_tile(mixed, *args):
        calls.append(tuple(mixed.shape))
        return residual._sinkhorn_coefficients(mixed, *args)

    monkeypatch.setattr(residual, "_compiled_sinkhorn", lambda: eager_tile)
    inputs = (
        torch.randn(*shape, 24, requires_grad=True),
        torch.randn(3, requires_grad=True),
        torch.randn(24, requires_grad=True),
    )
    originals = tuple(t.detach().clone().requires_grad_() for t in inputs)
    actual = residual._tiled_sinkhorn_coefficients(*inputs, 4, 1e-6, 20)
    expected = residual._sinkhorn_coefficients(*originals, 4, 1e-6, 20)
    incoming = tuple(torch.randn_like(t) / (shape[0] * shape[1]) for t in actual)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)
        assert (a - b).abs().max() < 1e-5
        assert a.dtype == torch.float32
    actual_grad = torch.autograd.grad(
        sum((t * g).sum() for t, g in zip(actual, incoming, strict=True)), inputs
    )
    expected_grad = torch.autograd.grad(
        sum((t * g).sum() for t, g in zip(expected, incoming, strict=True)), originals
    )
    for a, b in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)
        assert (a - b).abs().max() < 1e-5
        assert torch.isfinite(a).all()
    assert calls == [(2048, 24)] * ((shape[0] * shape[1] + 2047) // 2048)
