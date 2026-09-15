"""Slot-major expert grouping and fixed-frequency router loss retain exact gradients."""

import copy

import pytest
import torch
import torch.nn.functional as F
from test_miniqwen4 import tiny_config
from torch.utils.checkpoint import checkpoint

from minifrontier.models.miniqwen4.batched_experts import (
    BatchedQwenExperts,
    LoopQwenExperts,
    configure_experts,
)
from minifrontier.training.qwen_balance import normalized_router_loss


@pytest.fixture(autouse=True)
def single_cpu_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def expert_case(pattern):
    torch.manual_seed(1509)
    module = LoopQwenExperts(tiny_config(num_experts=8).upstream_config())
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(0, 0.1)
    if pattern == "collapsed":
        ids = torch.tensor([[5, 1, 7]] * 13)
    elif pattern == "duplicate":
        ids = torch.tensor([[3, 3, 1], [1, 3, 3], [3, 1, 3]] * 4)
    elif pattern == "single":
        ids = torch.tensor([[7, 0, 3]])
    else:
        ids = torch.stack([torch.randperm(8)[:3] for _ in range(13)])
    x = torch.randn(len(ids), module.hidden_dim, requires_grad=True)
    weights = torch.randn(ids.shape).softmax(-1).requires_grad_()
    return module, x, ids, weights


@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("pattern", ["random", "collapsed", "duplicate", "single"])
def test_stable_grouping_matches_reference_outputs_and_all_gradients(autocast, pattern):
    module, x, ids, weights = expert_case(pattern)
    reference = copy.deepcopy(module)
    old_x = x.detach().clone().requires_grad_()
    old_weights = weights.detach().clone().requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        actual = module(x, ids, weights)
        expected = reference.reference(old_x, ids, old_weights)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # Distinct signed upstream gradients expose row-order changes in parameter sums.
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    torch.testing.assert_close(x.grad, old_x.grad, rtol=0, atol=0)
    torch.testing.assert_close(weights.grad, old_weights.grad, rtol=0, atol=0)
    unused = torch.ones(module.num_experts, dtype=torch.bool)
    unused[ids.unique()] = False
    for (name, parameter), (old_name, old_parameter) in zip(
        module.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert name == old_name
        assert (parameter.grad is None) == (old_parameter.grad is None)
        torch.testing.assert_close(parameter.grad, old_parameter.grad, rtol=0, atol=0)
        # Expert weights share 3D parameters: inactive slices receive zero, not None.
        assert parameter.grad is not None
        assert not torch.count_nonzero(parameter.grad[unused])


def test_empty_routes_do_not_create_parameter_or_input_gradients():
    module, _, _, _ = expert_case("single")
    x = torch.empty(0, module.hidden_dim, requires_grad=True)
    ids = torch.empty(0, 3, dtype=torch.long)
    weights = torch.empty(0, 3, requires_grad=True)
    actual = module(x, ids, weights)
    expected = module.reference(x, ids, weights)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not actual.requires_grad and not expected.requires_grad
    assert x.grad is None and weights.grad is None
    assert all(parameter.grad is None for parameter in module.parameters())


@pytest.mark.parametrize("autocast", [False, True])
def test_unbound_expert_views_keep_gradients_during_checkpoint_replay(autocast):
    module, x, ids, weights = expert_case("collapsed")
    reference = copy.deepcopy(module)
    old_x = x.detach().clone().requires_grad_()
    old_weights = weights.detach().clone().requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        actual = checkpoint(module, x, ids, weights, use_reentrant=False)
        expected = checkpoint(reference.reference, old_x, ids, old_weights, use_reentrant=False)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    torch.testing.assert_close(x.grad, old_x.grad, rtol=0, atol=0)
    torch.testing.assert_close(weights.grad, old_weights.grad, rtol=0, atol=0)
    for parameter, old_parameter in zip(module.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(parameter.grad, old_parameter.grad, rtol=0, atol=0)


def test_expert_gradients_are_joined_once_per_parameter():
    calls = []
    for optimized in (False, True):
        module, x, ids, weights = expert_case("random")
        forward = module if optimized else module.reference
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            forward(x, ids, weights).sum().backward()
        calls.append({item.key: item.count for item in profile.key_averages()})
    assert calls[0]["aten::select_backward"] == 2 * ids.unique().numel()
    assert calls[1].get("aten::select_backward", 0) == 0
    assert calls[1]["UnbindBackward0"] == 2


def test_grouping_retains_expert_and_slot_major_projection_rows(monkeypatch):
    module, x, ids, weights = expert_case("random")
    rows = []
    original_linear = F.linear
    pointers = {module.gate_up_proj[i].data_ptr(): i for i in range(module.num_experts)}

    def capture(value, weight, bias=None):
        if weight.data_ptr() in pointers:
            rows.append((pointers[weight.data_ptr()], value.detach().clone()))
        return original_linear(value, weight, bias)

    monkeypatch.setattr(F, "linear", capture)
    module(x, ids, weights)
    assert [expert for expert, _ in rows] == sorted(ids.unique().tolist())
    for expert, actual in rows:
        _, tokens = torch.where(ids.transpose(0, 1).eq(expert))
        torch.testing.assert_close(actual, x[tokens], rtol=0, atol=0)


def test_stable_grouping_has_no_dynamic_nonzero_or_bincount():
    module, x, ids, weights = expert_case("random")
    calls = []
    for forward in (module.reference, module):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            forward(x, ids, weights)
        calls.append({item.key: item.count for item in profile.key_averages()})
    assert calls[0]["aten::nonzero"] == ids.unique().numel() + 1
    assert calls[1].get("aten::nonzero", 0) == 0
    assert calls[1].get("aten::bincount", 0) == 0
    assert calls[1].get("aten::one_hot", 0) == 0


@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("pattern", ["random", "collapsed", "single"])
def test_batched_experts_match_projection_precision_and_all_gradients(
    autocast, weight_dtype, pattern
):
    module, x, ids, weights = expert_case(pattern)
    reference = copy.deepcopy(module)
    configure_experts(module, "batched")
    old_x = x.detach().clone().requires_grad_()
    weights = weights.detach().to(weight_dtype).requires_grad_()
    old_weights = weights.detach().clone().requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        actual = module(x, ids, weights)
        expected = reference.reference(old_x, ids, old_weights)
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    pairs = [(actual, expected), (x.grad, old_x.grad), (weights.grad, old_weights.grad)]
    pairs.extend(
        (p.grad, old.grad)
        for p, old in zip(module.parameters(), reference.parameters(), strict=True)
    )
    for value, old in pairs:
        # CPU BF16 GEMMs retain these fixtures exactly. FP32 batched GEMMs have
        # small reduction-order differences, including padding in weight gradients.
        torch.testing.assert_close(
            value, old, rtol=0 if autocast else 1e-6, atol=0 if autocast else 2e-7
        )
    unused = torch.ones(module.num_experts, dtype=torch.bool)
    unused[ids.unique()] = False
    for parameter in module.parameters():
        assert parameter.grad is not None
        assert not torch.count_nonzero(parameter.grad[unused])


@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.bfloat16])
def test_batched_bf16_accumulator_and_checkpoint_gradients(weight_dtype):
    module, x, ids, weights = expert_case("random")
    reference = copy.deepcopy(module)
    configure_experts(module, "batched")
    x = x.detach().bfloat16().requires_grad_()
    old_x = x.detach().clone().requires_grad_()
    weights = weights.detach().to(weight_dtype).requires_grad_()
    old_weights = weights.detach().clone().requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = checkpoint(module, x, ids, weights, use_reentrant=False)
        expected = checkpoint(reference.reference, old_x, ids, old_weights, use_reentrant=False)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    # Packed input-gradient scatter changes the sum order for BF16 leaf inputs;
    # routing, forward accumulator and parameter-gradient fixtures remain exact.
    difference = (x.grad.float() - old_x.grad.float()).double()
    scale = old_x.grad.double()
    assert difference.norm() <= 0.006 * scale.norm()
    assert difference.abs().max() <= 0.015 * scale.abs().max()
    torch.testing.assert_close(weights.grad, old_weights.grad, rtol=0, atol=0)
    for parameter, old in zip(module.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(parameter.grad, old.grad, rtol=0, atol=0)


@pytest.mark.parametrize("reason", ["duplicate", "empty", "no_slots"])
def test_batched_unsafe_or_empty_routes_use_the_exact_loop(reason, monkeypatch):
    module, x, ids, weights = expert_case("duplicate" if reason == "duplicate" else "random")
    if reason == "empty":
        x = x.detach()[:0].requires_grad_()
        ids = ids[:0]
        weights = weights.detach()[:0].requires_grad_()
    elif reason == "no_slots":
        ids = ids[:, :0]
        weights = weights.detach()[:, :0].requires_grad_()
    reference = copy.deepcopy(module)
    configure_experts(module, "batched")
    old_x = x.detach().clone().requires_grad_()
    old_weights = weights.detach().clone().requires_grad_()

    def forbidden(*args, **kwargs):
        raise AssertionError("fallback must not allocate or run padded GEMMs")

    monkeypatch.setattr(torch, "bmm", forbidden)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = module(x, ids, weights)
        expected = reference.reference(old_x, ids, old_weights)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if actual.requires_grad:
        gradient = torch.randn_like(actual)
        actual.backward(gradient)
        expected.backward(gradient)
    for value, old in [
        (x, old_x),
        (weights, old_weights),
        *zip(module.parameters(), reference.parameters(), strict=True),
    ]:
        assert (value.grad is None) == (old.grad is None)
        torch.testing.assert_close(value.grad, old.grad, rtol=0, atol=0)


def test_batched_gemms_pack_source_slot_major_rows_without_changing_parameters(monkeypatch):
    module, x, ids, weights = expert_case("random")
    original = dict(module.named_parameters())
    configure_experts(module, "batched")
    assert isinstance(module, BatchedQwenExperts)
    assert all(parameter is original[name] for name, parameter in module.named_parameters())
    calls = []
    bmm = torch.bmm

    def capture(left, right):
        calls.append(left.detach().clone())
        return bmm(left, right)

    monkeypatch.setattr(torch, "bmm", capture)
    module.execution_stats_enabled = True
    module(x, ids, weights)
    counts = ids.flatten().bincount(minlength=module.num_experts).tolist()
    capacities = sorted({1 << (count - 1).bit_length() for count in counts if count})
    assert len(calls) == 2 * len(capacities)
    assert module.last_execution["backend"] == "bucketed"
    assert module.last_execution["packed_assignments"] < 2 * ids.numel()
    for packed, capacity in zip(calls[::2], capacities, strict=True):
        experts = [
            expert
            for expert, count in enumerate(counts)
            if count and 1 << (count - 1).bit_length() == capacity
        ]
        for local, expert in enumerate(experts):
            _, tokens = torch.where(ids.transpose(0, 1).eq(expert))
            torch.testing.assert_close(packed[local, : len(tokens)], x[tokens], rtol=0, atol=0)
            assert not torch.count_nonzero(packed[local, len(tokens) :])
    configure_experts(module, "loop")
    assert type(module) is LoopQwenExperts
    assert all(parameter is original[name] for name, parameter in module.named_parameters())


@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("pattern", ["single_expert", "padding_like"])
def test_bucketed_extreme_skew_uses_gemms_without_dropping_routes(autocast, pattern, monkeypatch):
    torch.manual_seed(1509)
    module = BatchedQwenExperts(tiny_config(num_experts=64).upstream_config())
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(0, 0.1)
    if pattern == "single_expert":
        ids = torch.zeros(257, 1, dtype=torch.long)
    else:
        ids = torch.tensor(
            [[60, 59, 58, 57]] * 240 + [torch.randperm(64)[:4].tolist() for _ in range(17)]
        )
    x = torch.randn(len(ids), module.hidden_dim, requires_grad=True)
    weights = torch.randn(ids.shape).softmax(-1)
    weights = weights.to(torch.bfloat16 if autocast else torch.float32).requires_grad_()
    reference = copy.deepcopy(module)
    old_x = x.detach().clone().requires_grad_()
    old_weights = weights.detach().clone().requires_grad_()
    module.execution_stats_enabled = True

    def forbidden(*args, **kwargs):
        raise AssertionError("skewed nonduplicate routes must execute bucketed GEMMs")

    monkeypatch.setattr(LoopQwenExperts, "forward", forbidden)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        actual = module(x, ids, weights)
        expected = reference.reference(old_x, ids, old_weights)
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    pairs = [(actual, expected), (x.grad, old_x.grad), (weights.grad, old_weights.grad)]
    pairs.extend(
        (p.grad, q.grad) for p, q in zip(module.parameters(), reference.parameters(), strict=True)
    )
    for value, old in pairs:
        torch.testing.assert_close(
            value, old, rtol=0 if autocast else 2e-5, atol=0 if autocast else 2e-6
        )
    stats = module.last_execution
    assert stats["backend"] == "bucketed"
    assert stats["routed_assignments"] == ids.numel()
    assert ids.numel() <= stats["packed_assignments"] < 2 * ids.numel()
    unused = torch.ones(module.num_experts, dtype=torch.bool)
    unused[ids.unique()] = False
    for parameter in module.parameters():
        assert parameter.grad is not None
        assert not torch.count_nonzero(parameter.grad[unused])


def reference_fixed_frequency_loss(routers, experts, top_k, valid_mask, frequency):
    """Previous implementation, including its unused assignment counts."""
    counts = torch.zeros(experts, device=routers[0].device, dtype=torch.float32)
    probabilities = torch.zeros_like(counts)
    denominator = counts.new_zeros(())
    for logits in routers:
        p = logits.softmax(-1)
        valid = (
            torch.ones(p.shape[0], device=p.device, dtype=torch.bool)
            if valid_mask is None
            else valid_mask.flatten().bool()
        )
        selected = p.detach().topk(top_k, dim=-1).indices[valid]
        counts += torch.bincount(selected.flatten(), minlength=experts)
        probabilities = probabilities + (p.float() * valid[:, None]).sum(0)
        denominator += valid.sum()
    mean_probability = probabilities / denominator.clamp_min(1)
    return (frequency * mean_probability.unsqueeze(0)).sum() * experts


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mask_kind", ["none", "padding", "all_masked"])
def test_fixed_frequency_loss_and_router_gradients_match_reference(dtype, mask_kind):
    torch.manual_seed(915)
    routers = [torch.randn(14, 8, dtype=dtype, requires_grad=True) for _ in range(3)]
    old = [router.detach().clone().requires_grad_() for router in routers]
    mask = None if mask_kind == "none" else torch.arange(14).reshape(2, 7).lt(9)
    if mask_kind == "all_masked":
        mask.zero_()
    frequency = torch.rand(8)
    actual = normalized_router_loss(routers, 8, 3, mask, frequency)
    expected = reference_fixed_frequency_loss(old, 8, 3, mask, frequency)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.backward()
    expected.backward()
    for router, old_router in zip(routers, old, strict=True):
        torch.testing.assert_close(router.grad, old_router.grad, rtol=0, atol=0)


def test_fixed_frequency_does_not_recompute_assignments(monkeypatch):
    routers = [torch.randn(5, 8, requires_grad=True)]

    def forbidden(*args, **kwargs):
        raise AssertionError("fixed frequency must not recompute assignments")

    monkeypatch.setattr(torch.Tensor, "topk", forbidden)
    monkeypatch.setattr(torch, "bincount", forbidden)
    normalized_router_loss(routers, 8, 3, frequency=torch.ones(8)).backward()


def test_missing_frequency_and_all_masked_input_keeps_zero_gradients():
    routers = [torch.randn(5, 8, requires_grad=True)]
    loss = normalized_router_loss(routers, 8, 3, torch.zeros(5, dtype=torch.bool))
    assert loss.item() == 0
    loss.backward()
    assert routers[0].grad is not None
    assert not torch.count_nonzero(routers[0].grad)
