"""CE-weighted text domains, sample-weighted visual domains and separate media quotas.

Videos have their own phase occurrence budget: the corpus pool's video percentage
is not incorrectly treated as either CE tokens or image occurrences. Every rank
replays the same global decisions before taking its disjoint local slice.
"""

from types import SimpleNamespace

from .mixture import TokenMixtureCursor


class MediaMixtureCursor:
    def __init__(
        self, dataset, recipe, *, batch_size, rank=0, world_size=1, seed=42, max_epochs=None
    ):
        self.data = dataset if hasattr(dataset, "domains") else dataset.documents
        self.recipe = recipe
        if recipe.get("schema_version") != 1 or recipe.get("ce_token_budget", 0) <= 0:
            raise ValueError("media mixture needs version 1 and its actual CE budget")
        self.batch_size, self.rank, self.world_size = batch_size, rank, world_size
        self.images = getattr(self.data, "image_counts", [0] * len(self.data.domains))
        self.videos = getattr(self.data, "video_counts", [0] * len(self.data.domains))
        self.goals = {
            "image": recipe.get("image_occurrences", 0),
            "video": recipe.get("video_examples", 0),
        }
        if any(type(n) is not int or n < 0 for n in self.goals.values()):
            raise ValueError("media occurrence budgets must be nonnegative integers")
        disabled = set(recipe.get("disabled_domains", []))
        rows: dict[str, list[int]] = {"text": [], "image": [], "video": []}
        for i, domain in enumerate(self.data.domains):
            if domain in disabled or self.data.ce_counts[i] <= 0:
                continue
            kind = "video" if self.videos[i] else "image" if self.images[i] else "text"
            rows[kind].append(i)
        weights = {
            "text": recipe["text_mixture_tokens"],
            "image": recipe["image_mixture_samples"],
            "video": recipe.get("video_mixture_samples", {}),
        }
        self.rows, self.pools = rows, {}
        for kind, indices in rows.items():
            if not indices:
                if weights[kind] or self.goals.get(kind, 0):
                    raise ValueError(f"required {kind} pool has no eligible rows")
                continue
            if kind != "text" and self.goals[kind] == 0:
                raise ValueError(
                    f"{kind} rows require a phase quota or an explicit disabled domain"
                )
            view = SimpleNamespace(
                domains=[self.data.domains[i] for i in indices],
                ce_counts=[self.data.ce_counts[i] if kind == "text" else 1 for i in indices],
                input_counts=[self.data.input_counts[i] for i in indices],
            )
            self.pools[kind] = TokenMixtureCursor(
                view, weights[kind], batch_size=1, seed=seed, max_epochs=max_epochs
            )
        if not self.pools:
            raise ValueError("mixture has no learning data")
        self.offset, self.ce_seen = 0, 0
        self.seen = {"image": 0, "video": 0}

    def next(self, count=None):
        count = self.batch_size if count is None else count
        if not 1 <= count <= self.batch_size:
            raise ValueError("sample count must fit the microbatch capacity")
        selected = []
        for _ in range(count * self.world_size):
            progress = self.ce_seen / self.recipe["ce_token_budget"]
            behind = [
                kind
                for kind in self.goals
                if self.goals[kind] and self.seen[kind] / self.goals[kind] < min(1, progress)
            ]
            if behind:
                kind = min(behind, key=lambda k: self.seen[k] / self.goals[k])
            elif "text" in self.pools:
                kind = "text"
            else:
                kind = min(self.pools, key=lambda k: self.seen[k] / max(1, self.goals[k]))
            index = self.rows[kind][self.pools[kind].next()[0]]
            self.ce_seen += self.data.ce_counts[index]
            self.seen["image"] += self.images[index]
            self.seen["video"] += self.videos[index]
            self.offset += 1
            selected.append(index)
        start = self.rank * count
        return selected[start : start + count]

    def quotas_met(self):
        return all(self.seen[kind] >= goal for kind, goal in self.goals.items())

    def state_dict(self):
        return dict(
            kind="media-mixture-v1",
            recipe=self.recipe,
            batch_size=self.batch_size,
            world_size=self.world_size,
            offset=self.offset,
            ce_seen=self.ce_seen,
            seen=self.seen.copy(),
            pools={kind: pool.state_dict() for kind, pool in self.pools.items()},
        )

    def load_state_dict(self, state):
        if (
            state["recipe"] != self.recipe
            or state["batch_size"] != self.batch_size
            or state["world_size"] != self.world_size
        ):
            raise ValueError("media mixture changed during exact resume")
        for kind, pool in self.pools.items():
            pool.load_state_dict(state["pools"][kind])
        self.offset, self.ce_seen, self.seen = (
            state["offset"],
            state["ce_seen"],
            state["seen"].copy(),
        )
