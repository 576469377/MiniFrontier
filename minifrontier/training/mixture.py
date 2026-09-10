"""Deterministic domain sampling by actual tokens, with exactly resumable DDP cursors."""

import hashlib
import random


class TokenMixtureCursor:
    def __init__(
        self,
        dataset,
        proportions,
        *,
        batch_size,
        rank=0,
        world_size=1,
        seed=42,
        max_epochs=None,
        denominator="ce",
    ):
        if abs(sum(proportions.values()) - 1) > 1e-8 or any(p <= 0 for p in proportions.values()):
            raise ValueError("domain proportions must be positive and sum to one")
        data = dataset if hasattr(dataset, "domains") else getattr(dataset, "documents", None)
        if data is None or not hasattr(data, "domains"):
            raise ValueError("token mixtures require audited document/native v2 data")
        self.proportions = dict(sorted(proportions.items()))
        self.rows = {
            domain: [i for i, item in enumerate(data.domains) if item == domain]
            for domain in self.proportions
        }
        unknown = set(data.domains) - self.proportions.keys()
        if unknown:
            raise ValueError(f"mixture would silently exclude domains: {sorted(unknown)}")
        if any(not rows for rows in self.rows.values()):
            raise ValueError("a required mixture domain has no examples; rebuild the corpus")
        self.counts = data.ce_counts if denominator == "ce" else data.input_counts
        for domain, rows in self.rows.items():
            self.rows[domain] = [i for i in rows if self.counts[i] > 0]
            if not self.rows[domain]:
                raise ValueError("mixture domain has no effective positions")
        self.batch_size, self.rank, self.world_size = batch_size, rank, world_size
        self.seed, self.max_epochs, self.denominator = seed, max_epochs, denominator
        self.offset = 0
        self.served = {d: 0 for d in self.rows}
        self.epochs = {d: 0 for d in self.rows}
        self.cursors = {d: 0 for d in self.rows}
        self.orders = {d: self._order(d) for d in self.rows}

    def _order(self, domain):
        digest = hashlib.sha256(f"{self.seed}:{domain}:{self.epochs[domain]}".encode()).hexdigest()
        order = self.rows[domain].copy()
        random.Random(int(digest[:16], 16)).shuffle(order)
        return order

    def next(self, count=None):
        count = self.batch_size if count is None else count
        if not 1 <= count <= self.batch_size:
            raise ValueError("sample count must fit the microbatch capacity")
        selected = []
        for _ in range(count * self.world_size):
            domain = min(self.rows, key=lambda d: self.served[d] / self.proportions[d])
            cursor = self.cursors[domain]
            if cursor == len(self.orders[domain]):
                self.epochs[domain] += 1
                if self.max_epochs is not None and self.epochs[domain] >= self.max_epochs:
                    raise ValueError("domain exhausted its unique-data epoch limit")
                self.orders[domain] = self._order(domain)
                self.cursors[domain] = cursor = 0
            index = self.orders[domain][cursor]
            self.cursors[domain] += 1
            self.served[domain] += self.counts[index]
            self.offset += 1
            selected.append(index)
        start = self.rank * count
        return selected[start : start + count]

    def state_dict(self):
        return dict(
            kind="token-mixture-v1",
            offset=self.offset,
            served=self.served.copy(),
            epochs=self.epochs.copy(),
            cursors=self.cursors.copy(),
            proportions=self.proportions,
            seed=self.seed,
            denominator=self.denominator,
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            world_size=self.world_size,
        )

    def load_state_dict(self, state):
        if (
            state["proportions"] != self.proportions
            or state["seed"] != self.seed
            or state["denominator"] != self.denominator
            or state["max_epochs"] != self.max_epochs
            or state["batch_size"] != self.batch_size
            or state["world_size"] != self.world_size
        ):
            raise ValueError("mixture recipe changed during exact resume")
        self.offset = state["offset"]
        self.served, self.epochs, self.cursors = (
            state["served"].copy(),
            state["epochs"].copy(),
            state["cursors"].copy(),
        )
        self.orders = {d: self._order(d) for d in self.rows}
