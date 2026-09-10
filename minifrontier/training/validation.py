"""Fixed, nonrepeating CE validation sets and main-training-token cadence."""

import hashlib
import json
from collections import defaultdict
from typing import Any

from minifrontier.training.runtime import validation_indices

PRETRAINING_VALIDATION = dict(
    periodic_ce_tokens=1_000_000,
    phase_end_ce_tokens=5_000_000,
    early_interval_ce=10_000_000,
    early_until_ce=100_000_000,
    later_interval_ce=50_000_000,
)


def validation_due(before, after, policy):
    """Detect crossed main-CE milestones without a separate resumable clock."""
    end = policy["early_until_ce"]
    early, later = policy["early_interval_ce"], policy["later_interval_ce"]
    return after > before and (
        min(before, end) // early < min(after, end) // early
        or (after > end and (max(before, end) - end) // later < (after - end) // later)
    )


class CEValidation:
    def __init__(self, dataset, *, seed, policy=None):
        self.policy = dict(PRETRAINING_VALIDATION if policy is None else policy)
        if any(v < 1 for v in self.policy.values()) or (
            self.policy["periodic_ce_tokens"] > self.policy["phase_end_ce_tokens"]
        ):
            raise ValueError("invalid pretraining validation policy")
        metadata = getattr(dataset, "documents", None)
        metadata = dataset if metadata is None else metadata
        if not hasattr(metadata, "ce_counts") or not hasattr(metadata, "domains"):
            raise ValueError("CE validation requires encoded CE/domain metadata")
        self.counts = [int(v) for v in metadata.ce_counts]
        self.domains = list(metadata.domains)
        if (
            len(self.counts) != len(dataset)
            or len(self.domains) != len(dataset)
            or any(v < 0 for v in self.counts)
        ):
            raise ValueError("validation needs aligned nonnegative CE/domain metadata")
        if sum(self.counts) < self.policy["phase_end_ce_tokens"]:
            raise ValueError("validation inventory is smaller than the phase-end CE budget")
        order = validation_indices(len(dataset), limit=0, seed=seed)
        self.selections = {}
        self.binding = dict(policy=self.policy, seed=seed, selection="fixed_without_replacement")
        for scope, key in (
            ("periodic", "periodic_ce_tokens"),
            ("phase_end", "phase_end_ce_tokens"),
        ):
            selected, count = [], 0
            domains: dict[str, int] = defaultdict(int)
            for index in order:
                if not self.counts[index]:
                    continue
                selected.append(index)
                count += self.counts[index]
                domains[self.domains[index]] += self.counts[index]
                if count >= self.policy[key]:
                    break
            self.selections[scope] = selected
            self.binding[scope] = dict(
                requested_ce_tokens=self.policy[key],
                actual_ce_tokens=count,
                examples=len(selected),
                domain_ce=dict(domains),
                indices_sha256=hashlib.sha256(json.dumps(selected).encode()).hexdigest(),
                overshoot_bound_ce=max(self.counts),
            )

    def batches(self, *, final, batch_size, rank=0, world_size=1):
        if batch_size < 1 or not 0 <= rank < world_size:
            raise ValueError("invalid validation microbatch or rank")
        scope = "phase_end" if final else "periodic"
        groups = defaultdict(list)
        for index in self.selections[scope][rank::world_size]:
            groups[self.domains[index]].append(index)
        for domain, indices in sorted(groups.items()):
            for start in range(0, len(indices), batch_size):
                yield domain, indices[start : start + batch_size]


class NativeCEValidation(CEValidation):
    """Index complete native answers and nonoverlapping document targets before selection."""

    def __init__(self, dataset, *, max_length, seed, policy=None):
        if max_length < 2:
            raise ValueError("native validation needs at least two token positions")
        self.dataset, self.max_length = dataset, max_length
        self.windows = []
        self.ce_counts, self.domains = [], []
        for index in range(len(dataset)):
            compact = hasattr(dataset, "ce_count_at")
            item: Any = None if compact else dataset[index]
            length = dataset.length_at(index) if compact else item["input_ids"].shape[1]
            domain = dataset.domain_at(index) if compact else item["domain"]
            windowable = hasattr(dataset, "windowable_at") and dataset.windowable_at(index)
            if not windowable and length > max_length:
                raise ValueError("complete validation record exceeds the declared context")
            starts = range(0, length - 1, max_length - 1) if windowable else (0,)
            for start in starts:
                self.windows.append((index, start, bool(windowable)))
                self.domains.append(domain)
                self.ce_counts.append(
                    dataset.ce_count_at(index, start, max_length)
                    if compact
                    else int(item["labels"][:, 1:].ne(-100).sum())
                )
        super().__init__(self, seed=seed, policy=policy)
        self.binding.update(
            context_length=max_length,
            window_descriptors_sha256=hashlib.sha256(json.dumps(self.windows).encode()).hexdigest(),
            units="complete native answers or text context windows with unique CE positions",
        )

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        record, start, windowable = self.windows[index]
        return (
            self.dataset.window_at(record, start, self.max_length)
            if windowable
            else self.dataset[record]
        )
