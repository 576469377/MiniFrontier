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
        if state is None and input_ids.shape[1] > 1:
            return self.prefill(h, input_ids, segments, modality)
        return self.forward_reference(h, input_ids, segments, modality, state)

    def prefill(self, h, input_ids, segments, modality):
        """Batch projections and depthwise windows; retain the streaming reset rules."""
        c = self.config
        indices, valid_orders, starts, text_masks, histories = [], [], [], [], []
        # One metadata transfer instead of two device synchronizations per token.
        for tokens, sample_segments, modalities in zip(
            input_ids.tolist(), segments.tolist(), modality.tolist(), strict=True
        ):
            row_indices, row_valid, row_starts, row_text = [], [], [], []
            history: list[int] = []
            previous, start = -1, 0
            for t, (token, segment, kind) in enumerate(
                zip(tokens, sample_segments, modalities, strict=True)
            ):
                text = segment >= 0 and kind == 0 and token >= c.control_token_count
                if segment != previous or not text:
                    history, start = [], t
                previous = segment
                if text:
                    history.append(token)
                specs = ((2, 0), (2, 1), (3, 0), (3, 1))
                row_indices.append(
                    [
                        self.hash_ids(history, order, head) if len(history) >= order else 0
                        for order, head in specs
                    ]
                )
                row_valid.append([text and len(history) >= order for order, _ in specs])
                row_starts.append(start)
                row_text.append(text)
                history = history[-2:]
            indices.append(row_indices)
            valid_orders.append(row_valid)
            starts.append(row_starts)
            text_masks.append(row_text)
            histories.append((history, previous))
        index = torch.tensor(indices, device=h.device)
        valid = torch.tensor(valid_orders, device=h.device)
        entries = [
            table(index[..., i]) * valid[..., i, None] for i, table in enumerate(self.tables)
        ]
        value = self.proj(torch.cat(entries, -1)) * self.gate(h).sigmoid()
        # Streaming concatenates with history initialized in h.dtype, including autocast.
        value = value.to(torch.promote_types(h.dtype, value.dtype))
        width = c.conv_size
        padded = F.pad(value, (0, 0, width - 1, 0))
        windows = padded.unfold(1, width, 1)
        run_start = torch.tensor(starts, device=h.device)
        is_text = torch.tensor(text_masks, device=h.device)
        positions = torch.arange(input_ids.shape[1], device=h.device)
        offsets = torch.arange(1 - width, 1, device=h.device)
        support = (positions[:, None] + offsets >= run_start[..., None]) & is_text[..., None]
        filtered = F.silu((windows * support[..., None, :] * self.conv.weight[:, 0]).sum(-1))
        result = self.eta * filtered
        tail = (
            padded[:, -(width - 1) :].transpose(1, 2)
            if width > 1
            else value.new_empty(value.shape[0], value.shape[2], 0)
        )
        tail_positions = torch.arange(
            input_ids.shape[1] - width + 1, input_ids.shape[1], device=h.device
        )
        tail_valid = (tail_positions >= run_start[:, -1, None]) & is_text[:, -1, None]
        tail = (tail * tail_valid[:, None]).clone()
        states = [
            dict(ids=history, segment=segment, conv=tail[b])
            for b, (history, segment) in enumerate(histories)
        ]
        return result, states

    def forward_reference(self, h, input_ids, segments, modality, state=None):
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
