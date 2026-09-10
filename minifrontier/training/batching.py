"""Global input-token targets, independent of the per-device microbatch ceiling."""

from itertools import pairwise


def parse_schedule(value, initial, ce_budget):
    """Parse CE milestone:input target pairs; milestones apply before an update."""
    if not value:
        return ()
    try:
        stages = tuple(tuple(int(n) for n in pair.split(":")) for pair in value.split(","))
    except ValueError as error:
        raise ValueError(
            "batch schedule must be CE:input pairs, e.g. 0:8192,400000:16384"
        ) from error
    if (
        not initial
        or not ce_budget
        or any(len(pair) != 2 for pair in stages)
        or stages[0] != (0, initial)
        or any(not 0 <= ce < ce_budget or not 1 <= target <= 1048576 for ce, target in stages)
        or any(b[0] <= a[0] or b[1] < a[1] for a, b in pairwise(stages))
    ):
        raise ValueError(
            "batch ramp requires a CE budget, 0:initial first, increasing CE milestones "
            "inside the budget and nondecreasing input targets in 1..1048576"
        )
    return stages


def input_target(initial, stages, ce_seen):
    target = initial
    for milestone, value in stages:
        if ce_seen < milestone:
            break
        target = value
    return target


def microbatch_count(capacity, remaining_inputs, sequence_length, world_size=1):
    """Consume whole examples; overshoot is bounded by one example per rank.

    Every row must have at most sequence_length input positions. When a whole
    group fits, no sample can cross the target. Near the boundary each rank
    takes one row. Thus the global sample-window boundary is independent of the
    microbatch ceiling, without truncating labels/media or buffering cursor state.
    """
    if min(capacity, remaining_inputs, sequence_length, world_size) < 1:
        raise ValueError("batch capacities and remaining input target must be positive")
    return min(capacity, max(1, remaining_inputs // (sequence_length * world_size)))
