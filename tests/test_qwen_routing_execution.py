"""Slot-major expert grouping and fixed-frequency router loss retain exact gradients."""

import copy

import pytest
import torch
import torch.nn.functional as F
from test_miniqwen4 import tiny_config

from minifrontier.models.miniqwen4.batched_experts import LoopQwenExperts
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
