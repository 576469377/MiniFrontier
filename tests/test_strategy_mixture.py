"""Mixture weights count tokens, distribute disjoint global batches and resume exactly."""

from types import SimpleNamespace

import pytest

from minifrontier.training.mixture import TokenMixtureCursor


def test_variable_length_token_mixture_and_ddp_resume():
    data = SimpleNamespace(
        domains=["a"] * 100 + ["b"] * 100,
        ce_counts=[10] * 100 + [100] * 100,
        input_counts=[11] * 100 + [101] * 100,
    )
    recipe = dict(batch_size=4, world_size=2, seed=17)
    left = TokenMixtureCursor(data, {"a": 0.5, "b": 0.5}, rank=0, **recipe)
    right = TokenMixtureCursor(data, {"a": 0.5, "b": 0.5}, rank=1, **recipe)
    for _ in range(20):
        assert not set(left.next()) & set(right.next())
    assert abs(left.served["a"] - left.served["b"]) <= 100
    assert left.state_dict() == right.state_dict()
    resumed = TokenMixtureCursor(data, {"a": 0.5, "b": 0.5}, rank=0, **recipe)
    resumed.load_state_dict(left.state_dict())
    for _ in range(30):
        assert resumed.next() == left.next()
    with pytest.raises(ValueError, match="silently exclude"):
        TokenMixtureCursor(data, {"a": 1.0}, batch_size=1)


def test_token_mixture_respects_unique_epoch_limit():
    data = SimpleNamespace(domains=["a"], ce_counts=[3], input_counts=[4])
    cursor = TokenMixtureCursor(data, {"a": 1.0}, batch_size=1, max_epochs=1)
    assert cursor.next() == [0]
    with pytest.raises(ValueError, match="epoch limit"):
        cursor.next()
