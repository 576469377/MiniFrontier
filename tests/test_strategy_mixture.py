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


def test_media_quotas_use_images_and_visual_samples_not_ce_weights():
    from minifrontier.training.media_mixture import MediaMixtureCursor

    # Text domains differ by 10x in length; image tasks by 8x. Their intended
    # denominators are deliberately different and multiimage rows count twice.
    data = SimpleNamespace(
        domains=["zh"] * 100 + ["en"] * 100 + ["caption"] * 100 + ["multi"] * 100 + ["video"] * 100,
        ce_counts=[10] * 100 + [100] * 100 + [5] * 100 + [40] * 100 + [10] * 100,
        input_counts=[11] * 100 + [101] * 100 + [70] * 100 + [170] * 100 + [270] * 100,
        image_counts=[0] * 200 + [1] * 100 + [2] * 100 + [0] * 100,
        video_counts=[0] * 400 + [1] * 100,
    )
    recipe = dict(
        schema_version=1,
        ce_token_budget=10000,
        image_occurrences=60,
        video_examples=7,
        text_mixture_tokens={"zh": 0.5, "en": 0.5},
        image_mixture_samples={"caption": 0.5, "multi": 0.5},
        video_mixture_samples={"video": 1.0},
    )
    options = dict(batch_size=2, world_size=2, seed=914)
    left = MediaMixtureCursor(data, recipe, rank=0, **options)
    right = MediaMixtureCursor(data, recipe, rank=1, **options)
    for _ in range(25):
        assert not set(left.next()) & set(right.next())
    assert left.state_dict() == right.state_dict()
    restored = MediaMixtureCursor(data, recipe, rank=0, **options)
    restored.load_state_dict(left.state_dict())
    while left.ce_seen < 10000 or not left.quotas_met():
        assert restored.next() == left.next()
    assert abs(left.pools["text"].served["zh"] - left.pools["text"].served["en"]) <= 100
    assert abs(left.pools["image"].served["caption"] - left.pools["image"].served["multi"]) <= 1
    assert 60 <= left.seen["image"] <= 62 and left.seen["video"] == 7
