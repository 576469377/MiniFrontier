from types import SimpleNamespace

import pytest

from minifrontier.training.batching import input_target, microbatch_count, parse_schedule
from minifrontier.training.media_mixture import MediaMixtureCursor
from minifrontier.training.mixture import TokenMixtureCursor
from minifrontier.training.runtime import BatchCursor


def test_schedule_uses_consumed_ce_at_update_boundaries():
    stages = parse_schedule("0:8192,400000:16384,2000000:32768", 8192, 20000000)
    assert [input_target(8192, stages, t) for t in (0, 399999, 400001, 2000000)] == [
        8192,
        8192,
        16384,
        32768,
    ]
    for bad in (
        "0:8192,0:16384",
        "1:8192",
        "0:8192,1:0",
        "0:8192,20000000:16384",
        "0:8192,x",
        "0:8192,4:2",
        "0:8192:3",
    ):
        with pytest.raises(ValueError):
            parse_schedule(bad, 8192, 20000000)
    with pytest.raises(ValueError):
        parse_schedule("0:8192", 8192, None)


@pytest.mark.parametrize("kind", ["uniform", "tokens", "media"])
@pytest.mark.parametrize("world", [1, 2])
def test_windows_and_resumed_stream_do_not_depend_on_microbatch_grouping(kind, world):
    data = SimpleNamespace(
        domains=["text"] * 50 + ["image"] * 50,
        input_counts=[17 + (i * 37) % 239 for i in range(100)],
        ce_counts=[9 + (i * 13) % 173 for i in range(100)],
        image_counts=[0] * 50 + [1] * 50,
    )

    def make(cap, rank, saved=None):
        args = dict(batch_size=cap, rank=rank, world_size=world, seed=42)
        if kind == "uniform":
            return BatchCursor(100, **args, offset=saved.offset if saved else 0)
        if kind == "tokens":
            cursor = TokenMixtureCursor(data, {"text": 0.7, "image": 0.3}, **args)
        else:
            cursor = MediaMixtureCursor(
                data,
                dict(
                    schema_version=1,
                    ce_token_budget=20000,
                    image_occurrences=20,
                    text_mixture_tokens={"text": 1.0},
                    image_mixture_samples={"image": 1.0},
                ),
                **args,
            )
        if saved:
            cursor.load_state_dict(saved.state_dict())
        return cursor

    def windows(cap):
        cursors = [make(cap, rank) for rank in range(world)]
        result = []
        for target in [500, 1000, 1000, 2000, 2000]:
            consumed, samples = 0, []
            while consumed < target:
                count = microbatch_count(cap, target - consumed, 256, world)
                selected = [i for cursor in cursors for i in cursor.next(count)]
                consumed += sum(data.input_counts[i] for i in selected)
                samples.extend(selected)
            assert target <= consumed < target + 256 * world
            result.append(samples)
            # Exact checkpoint recovery after every window, including ramp changes.
            resumed = [make(cap, rank, cursor) for rank, cursor in enumerate(cursors)]
            assert [c.next(1) for c in resumed] == [c.next(1) for c in cursors]
            cursors = resumed
        return result

    assert windows(1) == windows(16) == windows(128)
