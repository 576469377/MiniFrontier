"""Native packing and reproducible per-batch context curriculum; never truncate an answer."""

from typing import Any

import torch

from minifrontier.training.minifrontier1_strategy import PHASES


def context_length(phase, rng, phase_ce, maximum):
    schedule = PHASES[phase].get("lengths", {maximum: 1.0})
    lengths, weights = [], []
    for length, weight in schedule.items():
        if length <= maximum:
            if phase == "p3" and length == 8192:
                weight *= min(1.0, phase_ce / 10_000_000)
            lengths.append(length)
            weights.append(weight)
    if not lengths:
        raise ValueError("configured context cannot execute this phase's length curriculum")
    return rng.choices(lengths, weights=weights, k=1)[0]


def pack_records(items, capacity):
    """One packed row with complete media, independent samples and aligned CE masks."""
    if not items or sum(i["input_ids"].shape[1] for i in items) > capacity:
        raise ValueError("packing needs complete records that fit its capacity")
    ids, labels, segments = [], [], []
    spans: list[dict[str, Any]] = []
    offset = 0
    for segment, item in enumerate(items):
        length = item["input_ids"].shape[1]
        ids.append(item["input_ids"])
        labels.append(item["labels"])
        segments.append(torch.full_like(item["input_ids"], segment))
        spans.extend(
            dict(s, batch_index=0, start=s["start"] + offset) for s in item.get("media", [])
        )
        offset += length
    return dict(
        input_ids=torch.cat(ids, 1),
        labels=torch.cat(labels, 1),
        segment_ids=torch.cat(segments, 1),
        media=spans,
        sample_id=[i["sample_id"] for i in items],
        media_hashes=[h for i in items for h in i["media_hashes"]],
        media_exposures=sum(i.get("media_exposures", len(i["media"])) for i in items),
        packing_capacity=capacity,
        sample_count=len(items),
    )


def collate_records(items, pad_token_id):
    """Pad independent rows and relocate media without changing any sample's positions."""
    if len(items) == 1:
        return items[0]
    length = max(i["input_ids"].shape[1] for i in items)
    ids = items[0]["input_ids"].new_full((len(items), length), pad_token_id)
    labels, segments = torch.full_like(ids, -100), torch.full_like(ids, -1)
    spans: list[dict[str, Any]] = []
    for row, item in enumerate(items):
        n = item["input_ids"].shape[1]
        ids[row, :n], labels[row, :n] = item["input_ids"][0], item["labels"][0]
        segments[row, :n] = item.get("segment_ids", torch.zeros_like(item["input_ids"]))[0]
        spans.extend(dict(s, batch_index=row) for s in item.get("media", []))
    return dict(
        input_ids=ids,
        labels=labels,
        segment_ids=segments,
        media=spans,
        media_hashes=[h for i in items for h in i["media_hashes"]],
        media_exposures=sum(i.get("media_exposures", len(i.get("media", []))) for i in items),
        sample_count=sum(i.get("sample_count", 1) for i in items),
        attention_phase=items[0].get("attention_phase"),
    )


def microbatches(window, *, batch_size, max_padded_tokens, pad_token_id):
    """Group a preselected global window; capacity and P2 phase never mix rows incorrectly."""
    pending: list[dict[str, Any]] = []
    longest = 0
    for item in window:
        length = item["input_ids"].shape[1]
        if pending and (
            len(pending) == batch_size
            or max(longest, length) * (len(pending) + 1) > max_padded_tokens
            or item.get("attention_phase") != pending[0].get("attention_phase")
        ):
            yield collate_records(pending, pad_token_id)
            pending, longest = [], 0
        pending.append(item)
        longest = max(longest, length)
    if pending:
        yield collate_records(pending, pad_token_id)
