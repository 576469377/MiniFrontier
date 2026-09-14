"""Relocation preserves weights, precision, gradients and routing ownership."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from minifrontier.models.minideepseekv4.batched_experts import (
    BatchedDeepSeekMoE as PreviousBatchedMoE,
)
from minifrontier.models.minideepseekv4.expert import TrainingGate as PreviousTrainingGate
from minifrontier.models.minideepseekv4.upstream_layers import (
    Expert as PreviousExpert,
)
from minifrontier.models.minideepseekv4.upstream_layers import (
    MoE as PreviousMoE,
)
from minifrontier.models.minideepseekv4.upstream_layers import (
    RMSNorm as PreviousRMSNorm,
)
from minifrontier.models.minideepseekv4.vision import (
    DeepSeekVision as PreviousVision,
)
from minifrontier.models.minideepseekv4.vision import (
    DeepSeekVisionConfig as PreviousVisionConfig,
)
from minifrontier.models.minideepseekv41.batched_experts import (
    BatchedDeepSeekMoE,
    configure_experts,
)
from minifrontier.models.minideepseekv41.expert import TrainingGate
from minifrontier.models.minideepseekv41.layers import Expert, MoE, RMSNorm
from minifrontier.models.minideepseekv41.vision import DeepSeekVision, DeepSeekVisionConfig


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def arguments():
    return SimpleNamespace(
        dim=16,
        n_routed_experts=4,
        n_activated_experts=2,
        n_shared_experts=1,
        moe_inter_dim=12,
        n_hash_layers=0,
        score_func="sqrtsoftplus",
        route_scale=1.5,
        expert_dtype=None,
        swiglu_limit=10.0,
        vocab_size=32,
        vision_config={},
    )


def copy_initialized_state(previous, local):
    torch.manual_seed(83)
    with torch.no_grad():
        for parameter in previous.parameters():
            if parameter.is_floating_point():
                parameter.normal_(std=0.1)
    assert previous.state_dict().keys() == local.state_dict().keys()
    local.load_state_dict(previous.state_dict(), strict=True)


def compare_backward(previous, local, inputs, call, *, bfloat16):
    a = inputs.clone().requires_grad_(True)
    b = inputs.clone().requires_grad_(True)
    with torch.autocast("cpu", enabled=bfloat16, dtype=torch.bfloat16):
        expected = call(previous, a)
        actual = call(local, b)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    expected.float().square().sum().backward()
    actual.float().square().sum().backward()
    torch.testing.assert_close(a.grad, b.grad, atol=0, rtol=0)
    for (name, p), (other, q) in zip(
        previous.named_parameters(), local.named_parameters(), strict=True
    ):
        assert name == other
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0, msg=name)


@pytest.mark.parametrize("bfloat16", [False, True])
@pytest.mark.parametrize("kind", ["norm", "expert", "moe", "batched_moe"])
def test_relocated_text_layers_preserve_state_forward_and_backward(kind, bfloat16):
    args = arguments()
    if kind == "norm":
        previous, local = PreviousRMSNorm(16, 1e-20), RMSNorm(16, 1e-20)
        inputs = torch.randn(2, 5, 16)

        def call(module, x):
            return module(x)
    elif kind == "expert":
        previous, local = PreviousExpert(16, 12, swiglu_limit=10), Expert(16, 12, swiglu_limit=10)
        inputs = torch.randn(10, 16)
        weights = torch.linspace(0.1, 1.0, 10)[:, None]

        def call(module, x):
            return module(x, weights)
    else:
        previous_cls = PreviousMoE if kind == "moe" else PreviousBatchedMoE
        local_cls = MoE if kind == "moe" else BatchedDeepSeekMoE
        previous, local = previous_cls(0, args), local_cls(0, args)
        ids = torch.arange(10).reshape(2, 5)
        inputs = torch.randn(2, 5, 16)

        def call(module, x):
            return module(x, ids)

    copy_initialized_state(previous, local)
    compare_backward(previous, local, inputs, call, bfloat16=bfloat16)


@pytest.mark.parametrize("visual", [False, True])
def test_relocated_training_gate_preserves_modality_and_sequence_balance(visual):
    args = arguments()
    args.vision_config = {"enabled": True} if visual else None
    previous, local = PreviousTrainingGate(0, args), TrainingGate(0, args)
    copy_initialized_state(previous, local)
    ids = torch.arange(10)
    if visual:
        ids[2:5] += args.vocab_size
    valid = torch.ones(2, 5, dtype=torch.bool)
    valid[-1, -1] = False
    for gate in (previous, local):
        gate.valid_mask = valid
        gate.sequence_balance_enabled = True
    inputs = torch.randn(10, 16)
    expected_weights, expected_ids = previous(inputs, ids)
    actual_weights, actual_ids = local(inputs, ids)
    torch.testing.assert_close(actual_ids, expected_ids, atol=0, rtol=0)
    torch.testing.assert_close(actual_weights, expected_weights, atol=0, rtol=0)
    if visual:
        torch.testing.assert_close(local.route_image_mask, previous.route_image_mask)

    def call(gate, x):
        weights, _ = gate(x, ids)
        return weights + gate.sequence_loss

    compare_backward(previous, local, inputs, call, bfloat16=True)


@pytest.mark.parametrize("bfloat16", [False, True])
def test_relocated_vision_preserves_pixels_and_weight_gradients(bfloat16):
    values = dict(
        depth=2,
        hidden_size=16,
        num_heads=2,
        intermediate_size=24,
        patch_size=2,
        output_size=32,
        gradient_checkpointing=True,
    )
    previous = PreviousVision(PreviousVisionConfig(**values))
    local = DeepSeekVision(DeepSeekVisionConfig(**values))
    copy_initialized_state(previous, local)
    compare_backward(
        previous,
        local,
        torch.randn(12, 3, 2, 2),
        lambda model, x: model(x, 3, 4),
        bfloat16=bfloat16,
    )


def test_v41_configuration_only_replaces_local_experts():
    model = nn.ModuleList([MoE(0, arguments()), PreviousMoE(0, arguments())])
    original_keys = tuple(model.state_dict())
    configure_experts(model, "batched")
    assert type(model[0]) is BatchedDeepSeekMoE
    assert type(model[1]) is PreviousMoE
    configure_experts(model, "loop")
    assert type(model[0]) is MoE and type(model[1]) is PreviousMoE
    assert tuple(model.state_dict()) == original_keys


def test_v41_production_code_has_no_cross_model_computation_imports():
    root = Path(__file__).parents[1] / "minifrontier/models/minideepseekv41"
    siblings = ("minikimik3", "miniqwen4", "minideepseekv4", "minifrontier1")
    for path in root.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            elif isinstance(node, ast.Import):
                modules = [name.name for name in node.names]
            else:
                continue
            for module in modules:
                assert not set(module.split(".")) & set(siblings), (path, module)
