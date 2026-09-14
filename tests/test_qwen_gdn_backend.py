"""FLA boundary checks; direct execution benchmarks an explicitly allocated GPU."""

import copy
import json
import statistics
import time
from importlib.metadata import version

import pytest
import torch
from test_miniqwen4 import tiny_config
from test_qwen_sdpa import gradient_error, relative_error
from torch.utils.checkpoint import checkpoint

from minifrontier.models.miniqwen4 import (
    MiniQwen4Cache,
    MiniQwen4Config,
    MiniQwen4ForCausalLM,
    upstream_core,
)
from minifrontier.models.miniqwen4 import gdn as adapter


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def reference_chunk(q, k, v, **kwargs):
    return upstream_core.torch_chunk_gated_delta_rule(
        q, k, v, use_qk_l2norm_in_kernel=True, **kwargs
    )


def backward_pair(eager, optimized, hidden, mask, *, device, autocast, checkpointing):
    left = hidden.detach().clone().requires_grad_()
    right = hidden.detach().clone().requires_grad_()
    upstream_gradient = torch.randn_like(hidden)
    with torch.autocast(device, dtype=torch.bfloat16, enabled=autocast):
        if checkpointing:
            expected = checkpoint(eager, left, attention_mask=mask, use_reentrant=False)
            actual = checkpoint(optimized, right, attention_mask=mask, use_reentrant=False)
        else:
            expected = eager(left, attention_mask=mask)
            actual = optimized(right, attention_mask=mask)
    expected.backward(upstream_gradient)
    actual.backward(upstream_gradient)
    return expected, actual, left.grad, right.grad


@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("length", [17, 65])
def test_no_cache_adapter_preserves_source_operations(autocast, checkpointing, length):
    torch.manual_seed(564)
    eager = upstream_core.Qwen4ExpTextGatedDeltaNet(tiny_config().upstream_config(), 0)
    local = copy.deepcopy(eager)
    adapter.configure_gdn(local, "fla")
    hidden = torch.randn(2, length, eager.hidden_size)
    mask = torch.ones(2, length, dtype=torch.bool)
    mask[-1, length // 2 :] = False

    def optimized(x, attention_mask):
        return local._forward_chunk(x, attention_mask, reference_chunk)

    expected, actual, left, right = backward_pair(
        eager, optimized, hidden, mask, device="cpu", autocast=autocast, checkpointing=checkpointing
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(right, left, atol=0, rtol=0)
    assert gradient_error(local, eager) == 0


def test_fla_call_has_source_gate_and_state_contract(monkeypatch):
    captured = []

    def kernel(**kwargs):
        captured.append(kwargs)
        return kwargs["v"], kwargs["initial_state"]

    monkeypatch.setattr(adapter, "_load_fla_chunk", lambda: kernel)
    q, k, v = [torch.randn(2, 17, 4, 4).bfloat16() for _ in range(3)]
    g = -torch.rand(2, 17, 4)
    beta = torch.rand(2, 17, 4).bfloat16().sigmoid()
    state = torch.randn(2, 4, 4, 4).bfloat16()
    _, final = adapter.fla_chunk(
        q, k, v, g=g, beta=beta, initial_state=state, output_final_state=True
    )
    call = captured.pop()
    assert call["q"] is q and call["k"] is k and call["v"] is v
    assert call["g"] is g and call["beta"] is beta
    assert call["scale"] == 0.5 and call["chunk_size"] == 64
    assert call["use_qk_l2norm_in_kernel"] is True
    for flag in (
        "use_gate_in_kernel",
        "use_beta_sigmoid_in_kernel",
        "allow_neg_eigval",
        "state_v_first",
    ):
        assert call[flag] is False
    assert call["cu_seqlens"] is None
    assert final.dtype == torch.float32 and final.shape == (2, 4, 4, 4)
    torch.testing.assert_close(final, state.float(), atol=0, rtol=0)


@pytest.mark.parametrize("autocast", [False, True])
def test_opt_in_does_not_load_fla_on_cpu_or_change_parameters(monkeypatch, autocast):
    model = MiniQwen4ForCausalLM(tiny_config())
    original = copy.deepcopy(model)
    ids = {name: id(parameter) for name, parameter in model.named_parameters()}
    rng = torch.get_rng_state()

    def forbidden():
        raise AssertionError("CPU/cache path loaded FLA")

    monkeypatch.setattr(adapter, "_load_fla_chunk", forbidden)
    model.set_gdn_backend("fla")
    assert {name: id(p) for name, p in model.named_parameters()} == ids
    assert torch.equal(rng, torch.get_rng_state())
    assert model.state_dict().keys() == original.state_dict().keys()
    inputs = torch.tensor([[11, 12, 13, 14, 0], [11, 12, 13, 14, 15]])
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        expected = original(inputs, inputs.ne(0), inputs.masked_fill(inputs.eq(0), -100))
        actual = model(inputs, inputs.ne(0), inputs.masked_fill(inputs.eq(0), -100))
    torch.testing.assert_close(actual.loss, expected.loss, atol=0, rtol=0)
    expected.loss.backward()
    actual.loss.backward()
    assert gradient_error(model, original) == 0
    model.eval()
    original.eval()
    old_cache, new_cache = MiniQwen4Cache(), MiniQwen4Cache()
    with torch.no_grad():
        for inputs in (torch.tensor([[11, 12, 13]]), torch.tensor([[14]])):
            expected = original(inputs, cache=old_cache).logits
            actual = model(inputs, cache=new_cache).logits
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    with pytest.raises(ValueError, match="torch or fla"):
        model.set_gdn_backend("unknown")


def compare_states_cuda():
    """Include nonzero FP32 initial/final state and gradients through both."""
    q, k, v = [torch.randn(8, 2048, 12, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    decay = -torch.rand(8, 2048, 12, device="cuda") * 0.08
    beta = torch.rand(8, 2048, 12, device="cuda", dtype=torch.bfloat16)
    initial = torch.randn(8, 12, 64, 64, device="cuda") * 0.1
    inputs = (q, k, v, decay, beta, initial)
    left, right = [[t.detach().clone().requires_grad_() for t in inputs] for _ in range(2)]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected, expected_state = reference_chunk(
            *left[:3], g=left[3], beta=left[4], initial_state=left[5], output_final_state=True
        )
        actual, actual_state = adapter.fla_chunk(
            *right[:3], g=right[3], beta=right[4], initial_state=right[5], output_final_state=True
        )
    assert actual_state.dtype == expected_state.dtype == torch.float32
    assert actual_state.shape == expected_state.shape == initial.shape
    dout, dstate = torch.randn_like(expected), torch.randn_like(initial)
    torch.autograd.backward((expected, expected_state), (dout, dstate))
    torch.autograd.backward((actual, actual_state), (dout, dstate))
    metrics = {
        "output_relative_l2": relative_error(actual, expected),
        "final_state_relative_l2": relative_error(actual_state, expected_state),
        "gradient_relative_l2": {
            name: relative_error(a.grad, b.grad)
            for name, a, b in zip(
                ("q", "k", "v", "g", "beta", "initial_state"), right, left, strict=True
            )
        },
        "state_shape": list(actual_state.shape),
        "state_dtype": str(actual_state.dtype),
        "finite": bool(torch.isfinite(actual).all() and torch.isfinite(actual_state).all())
        and all(bool(torch.isfinite(t.grad).all()) for t in right),
    }
    return metrics


def benchmark_cuda():
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(9564)
    config = MiniQwen4Config()
    eager = upstream_core.Qwen4ExpTextGatedDeltaNet(config.upstream_config(), 0).cuda().train()
    fused = copy.deepcopy(eager)
    adapter.configure_gdn(fused, "fla")
    hidden = torch.randn(8, 2048, 512, device="cuda")
    valid = torch.ones(8, 2048, device="cuda", dtype=torch.bool)
    valid[-1, 1536:] = False
    results = {
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "fla_core_version": version("fla-core"),
        "seed": 9564,
        "shape": {
            "batch": 8,
            "sequence": 2048,
            "hidden": 512,
            "key_heads": 4,
            "value_heads": 12,
            "head_dim": 64,
        },
        "scope": "one uncached GDN layer with right padding; no optimizer or learning trial",
        "precision": "FP32 parameters/inputs with BF16 autocast; FP32 gate and recurrent state",
        "warmup_per_backend": 2,
        "timed_repeats_per_backend": 3,
        "timing_aggregation": "median",
        "comparisons": {},
        "timings": {},
    }
    for use_checkpoint in (False, True):
        eager.zero_grad(set_to_none=True)
        fused.zero_grad(set_to_none=True)
        expected, actual, left, right = backward_pair(
            eager, fused, hidden, valid, device="cuda", autocast=True, checkpointing=use_checkpoint
        )
        results["comparisons"][str(use_checkpoint)] = {
            "output_relative_l2": relative_error(actual, expected),
            "input_gradient_relative_l2": relative_error(right, left),
            "parameter_gradient_relative_l2": gradient_error(fused, eager),
            "per_parameter_gradient_relative_l2": {
                name: relative_error(p.grad, reference.grad)
                for (name, p), (_, reference) in zip(
                    fused.named_parameters(), eager.named_parameters(), strict=True
                )
            },
            "finite": bool(torch.isfinite(actual).all() and torch.isfinite(right).all()),
        }
        del expected, actual, left, right
    results["states"] = compare_states_cuda()
    for name, module in (("torch", eager), ("fla", fused)):
        timings = []
        module.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        for iteration in range(5):
            module.zero_grad(set_to_none=True)
            x = hidden.detach().requires_grad_()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = module(x, attention_mask=valid)
            output.backward(torch.ones_like(output))
            end.record()
            end.synchronize()
            if iteration >= 2:
                timings.append(start.elapsed_time(end))
            del output
        results["timings"][name] = {
            "forward_backward_ms": statistics.median(timings),
            "peak_above_baseline_mib": (torch.cuda.max_memory_allocated() - baseline) / 2**20,
        }
    results["elapsed_seconds_including_compilation"] = time.monotonic() - started
    results["passed"] = all(
        comparison["finite"]
        and max(
            comparison[key]
            for key in (
                "output_relative_l2",
                "input_gradient_relative_l2",
                "parameter_gradient_relative_l2",
            )
        )
        < 0.02
        and max(comparison["per_parameter_gradient_relative_l2"].values()) < 0.04
        for comparison in results["comparisons"].values()
    ) and (
        results["states"]["finite"]
        and results["states"]["output_relative_l2"] < 0.02
        and results["states"]["final_state_relative_l2"] < 0.02
        and max(results["states"]["gradient_relative_l2"].values()) < 0.04
    )
    print(json.dumps(results, indent=2))
    assert results["passed"], "FLA numerical comparison failed; do not deploy"


if __name__ == "__main__":
    benchmark_cuda()
