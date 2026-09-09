# Conditional lookup design references Qwen/Engram; this bounded text-only adapter is local.
"""One shallow 2/3-gram, two-hash memory with explicit reset and streaming semantics."""

from typing import cast

import torch
from torch import nn
from torch.nn import functional as F


class NgramLookup(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.tables = nn.ModuleList(nn.Embedding(c.lookup_rows, c.lookup_dim) for _ in range(4))
        self.proj = nn.Linear(4 * c.lookup_dim, c.hidden_size, bias=False)
        self.gate = nn.Linear(c.hidden_size, c.hidden_size, bias=False)
        self.conv = nn.Conv1d(
            c.hidden_size, c.hidden_size, c.conv_size, groups=c.hidden_size, bias=False
        )
        self.eta = nn.Parameter(torch.tensor(0.01))

    def hash_ids(self, history, order, head):
        value = self.config.lookup_seed + 104729 * (order + 3 * head)
        for token in history[-order:]:
            value = (value * (1000003 + head * 2) + token + 1) % 2147483647
        return value % self.config.lookup_rows

    def forward(self, h, input_ids, segments, modality, state=None):
        outputs, states = [], []
        c = self.config
        for b in range(len(h)):
            old = {} if state is None else state[b]
            history = list(old.get("ids", []))
            previous = old.get("segment", -1)
            conv = old.get("conv", h.new_zeros(c.hidden_size, c.conv_size - 1))
            vectors = []
            for t, token in enumerate(input_ids[b].tolist()):
                segment = int(segments[b, t])
                text = segment >= 0 and int(modality[b, t]) == 0 and token >= c.control_token_count
                if segment != previous or not text:
                    history, conv = [], torch.zeros_like(conv)
                previous = segment
                if not text:
                    vectors.append(h[b, t] * 0)
                    continue
                history.append(token)
                entries = []
                for i, (order, head) in enumerate(((2, 0), (2, 1), (3, 0), (3, 1))):
                    idx = self.hash_ids(history, order, head)
                    # Incomplete n-grams do not query a fictitious padding prefix.
                    entries.append(
                        cast(nn.Embedding, self.tables[i]).weight[idx]
                        if len(history) >= order
                        else cast(nn.Embedding, self.tables[i]).weight[0] * 0
                    )
                value = self.proj(torch.cat(entries)) * self.gate(h[b, t]).sigmoid()
                joined = torch.cat((conv, value[:, None]), -1)
                filtered = F.silu((joined * self.conv.weight[:, 0]).sum(-1))
                vectors.append(self.eta * filtered)
                conv = joined[:, 1:]
                history = history[-2:]
            outputs.append(torch.stack(vectors))
            states.append(dict(ids=history, segment=previous, conv=conv))
        return torch.stack(outputs), states
