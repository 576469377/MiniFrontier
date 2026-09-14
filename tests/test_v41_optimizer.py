"""Independent Algorithm 1 and optimizer partition/recovery checks."""

import copy

import pytest
import torch

from minifrontier.training.v41_optim import make_optimizer, sinkhorn_direction


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Linear(3, 4, bias=False)
        self.embedding = torch.nn.Embedding(5, 3)
        self.norm = torch.nn.LayerNorm(3)

    def optimizer_metadata(self):
        return {
            "query.weight": {"kind": "muon", "heads": 2},
            "embedding.weight": {"kind": "sinkhorn", "lr_scale": 5},
            "norm.weight": {"kind": "adamw"},
            "norm.bias": {"kind": "adamw"},
        }


def test_sinkhorn_matches_report_and_preserves_unvisited_rows():
    gradient = torch.tensor(
        [[3.0, 2.0, -1.0], [0.0, 0.0, 0.0], [1e-10, 0.0, 0.0], [2.0, -4.0, 1.0]]
    )
    expected = gradient.clone()
    norms = [sum(float(x) ** 2 for x in row) ** 0.5 for row in expected]
    for index, norm in enumerate(norms):
        if norm <= 1e-3 * sum(norms) / len(norms):
            expected[index] = 0
    for iteration in range(11):
        if iteration % 2 == 0:
            for row in expected:
                row /= sum(float(x) ** 2 for x in row) ** 0.5 + 1e-20
        else:
            for column in range(expected.shape[1]):
                expected[:, column] /= (
                    sum(float(x) ** 2 for x in expected[:, column]) ** 0.5 + 1e-20
                )
    expected *= 3**0.5
    torch.testing.assert_close(sinkhorn_direction(gradient), expected)
    assert torch.equal(sinkhorn_direction(torch.zeros(5, 3)), torch.zeros(5, 3))
    torch.testing.assert_close(gradient[0], torch.tensor([3.0, 2.0, -1.0]))


def test_optimizer_partition_update_and_exact_resume():
    torch.manual_seed(5)
    model = Model()
    optimizer = make_optimizer(model, lr=3e-4)
    groups = {g["name"]: g for g in optimizer.param_groups}
    assert groups["query.weight"]["blocks"] == ((0, 2), (2, 4))
    assert groups["embedding.weight"]["weight_decay"] == 0
    assert groups["embedding.weight"]["lr"] == 5 * 3e-4
    assert groups["norm.weight"]["weight_decay"] == 0.1
    for p in model.parameters():
        p.grad = torch.randn_like(p)
    before = model.embedding.weight.detach().clone()
    gradient = model.embedding.weight.grad.clone()
    optimizer.step()
    # The first Nesterov EMA direction is (1-beta)*(1+beta)*gradient.
    expected = before - 5 * 3e-4 * 0.18 * sinkhorn_direction(gradient * (1 - 0.95**2))
    torch.testing.assert_close(model.embedding.weight, expected)
    restored = copy.deepcopy(model)
    restored_optimizer = make_optimizer(restored, lr=3e-4)
    restored_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    for left, right in zip(model.parameters(), restored.parameters(), strict=True):
        left.grad = torch.randn_like(left)
        right.grad = left.grad.clone()
    optimizer.step()
    restored_optimizer.step()
    for left, right in zip(model.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    corrupted = copy.deepcopy(optimizer.state_dict())
    corrupted["param_groups"][0]["blocks"] = ((0, 4),)
    with pytest.raises(ValueError, match="contract differs: blocks"):
        restored_optimizer.load_state_dict(corrupted)


def test_nonfinite_gradient_cannot_partially_update_weights():
    model = Model()
    optimizer = make_optimizer(model, lr=3e-4)
    saved = copy.deepcopy(model.state_dict())
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    model.norm.bias.grad[0] = float("nan")
    with pytest.raises(FloatingPointError):
        optimizer.step()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, saved[name], rtol=0, atol=0)
