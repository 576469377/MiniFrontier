"""MF1 media budgets, aspect-preserving patches and a single packed-token contract."""

import hashlib
import json
from typing import Any

import torch

from minifrontier.models.miniqwen4.processing import process_frames as qwen_frames

PROCESSOR_VERSION = "mf1-native-v1"
CONTROL_VERSION = "mf1-control-v1"


def process_frames(frames, *, max_features=1024, patch_size=16, min_pixels=None, timestamps=None):
    groups = (len(frames) + 1) // 2
    per_group = max_features // max(1, groups)
    if per_group < 1:
        raise ValueError("frame count exceeds total media budget; reduce frames explicitly")
    minimum = (2 * patch_size) ** 2 if min_pixels is None else min_pixels
    sample = qwen_frames(
        frames,
        max_features=per_group,
        patch_size=patch_size,
        min_pixels=minimum,
        timestamps=timestamps,
    )
    if sample["feature_count"] > max_features:
        raise ValueError("processed media exceeds total feature budget")
    sample.update(
        processor=PROCESSOR_VERSION,
        source_size=list(frames[0].size),
        resized_size=[
            int(sample["grid_thw"][0, 2]) * patch_size,
            int(sample["grid_thw"][0, 1]) * patch_size,
        ],
        temporal_positions=[
            round(float(timestamps[min(2 * i, len(frames) - 1)]) * 100) for i in range(groups)
        ]
        if timestamps
        else [0],
    )
    return sample


def document_tiles(image, *, tile_size=448, max_tiles=4):
    if max_tiles not in range(1, 5) or tile_size < 32:
        raise ValueError("document mode supports one to four source-image tiles")
    width, height = image.size
    points = [
        (0, 0),
        (max(0, width - tile_size), 0),
        (0, max(0, height - tile_size)),
        (max(0, width - tile_size), max(0, height - tile_size)),
    ]
    result: list[dict[str, Any]] = []
    seen = set()
    for left, top in points:
        box = (left, top, min(width, left + tile_size), min(height, top + tile_size))
        if box not in seen:
            result.append(dict(image=image.crop(box), source_box=box, tile_id=len(result)))
            seen.add(box)
        if len(result) == max_tiles:
            break
    return result


def process_document(image, *, patch_size=16):
    """Global thumbnail and up to four crops made from original pixels, with source boxes."""
    global_view = process_frames([image], patch_size=patch_size, max_features=49)
    global_view.update(tile_id="global", source_box=[0, 0, *image.size])
    result = [global_view]
    for tile in document_tiles(image):
        view = process_frames([tile["image"]], patch_size=patch_size, max_features=196)
        view.update(
            tile_id=tile["tile_id"],
            source_box=list(tile["source_box"]),
            original_size=list(image.size),
        )
        result.append(view)
    return result


def token_metadata(
    input_ids, c, media=None, *, segment_ids=None, position_ids=None, offset=0, position_base=None
):
    b, length = input_ids.shape
    segments = torch.zeros_like(input_ids) if segment_ids is None else segment_ids.clone()
    if segments.shape != input_ids.shape:
        raise ValueError("packed segments must match expanded input positions")
    segments = segments.masked_fill(input_ids.eq(c.pad_token_id), -1)
    modality, media_ids = torch.zeros_like(input_ids), torch.full_like(input_ids, -1)
    positions = torch.zeros((3, b, length), dtype=torch.long, device=input_ids.device)
    linear_positions = (
        torch.arange(offset, offset + length, device=input_ids.device).expand(b, -1).clone()
    )
    for row in range(b):
        spans = sorted(
            (s for s in media or [] if s["batch_index"] == row), key=lambda s: s["start"]
        )
        media_budgets: dict[int, int] = {}
        for span in spans:
            if not 0 <= span["start"] < length:
                raise ValueError("media span start is outside the expanded sequence")
            segment = int(segments[row, span["start"]])
            media_budgets[segment] = media_budgets.get(segment, 0) + span["feature_count"]
        if any(count > c.protected_media_tokens for count in media_budgets.values()):
            raise ValueError(
                "request media exceeds protected-media budget; reduce resolution/frames before encoding"
            )
        cursor = 0
        base = offset if position_base is None else int(position_base[row])
        for number, span in enumerate(spans):
            start, count = span["start"], span["feature_count"]
            if (
                start < cursor
                or start + count > length
                or not input_ids[row, start : start + count].eq(c.image_token_id).all()
            ):
                raise ValueError("media overlaps or placeholder expansion is inconsistent")
            if (
                segments[row, start : start + count].unique().numel() != 1
                or segments[row, start] < 0
            ):
                raise ValueError("one media instance cannot cross a packed sample or padding")
            positions[:, row, cursor:start] = torch.arange(
                base, base + start - cursor, device=input_ids.device
            )
            base += start - cursor
            axes_list = []
            for t, h, w in span["grid_thw"].tolist():
                times = span.get("temporal_positions", list(range(t)))
                if len(times) != t:
                    raise ValueError("time coordinates disagree with temporal patch groups")
                axes_list.append(
                    torch.stack(
                        torch.meshgrid(
                            torch.tensor(times),
                            torch.arange(h // 2),
                            torch.arange(w // 2),
                            indexing="ij",
                        )
                    ).flatten(1)
                )
            axes = torch.cat(axes_list, 1).to(input_ids.device)
            if axes.shape[1] != count:
                raise ValueError("mRoPE axes and visual feature count differ")
            positions[:, row, start : start + count] = axes + base
            base += int(axes.max()) + 1
            modality[row, start : start + count] = 2 if span.get("resource_kind") == "video" else 1
            media_ids[row, start : start + count] = number
            cursor = start + count
        positions[:, row, cursor:] = torch.arange(
            base, base + length - cursor, device=input_ids.device
        )
        # Independent samples restart all three textual axes. Media offsets are translated with them.
        for start in (segments[row, 1:] != segments[row, :-1]).nonzero().flatten().add(1).tolist():
            if segments[row, start] >= 0:
                end = start + 1
                while end < length and segments[row, end] == segments[row, start]:
                    end += 1
                positions[:, row, start:end] -= positions[:, row, start : start + 1].clone()
                linear_positions[row, start:end] -= linear_positions[row, start].clone()
    if not torch.equal(input_ids.eq(c.image_token_id), modality.ne(0)):
        raise ValueError("every visual placeholder needs exactly one media feature")
    if position_ids is not None:
        if position_ids.shape != positions.shape or (position_ids < 0).any():
            raise ValueError("position_ids must be nonnegative [3,batch,length]")
        positions = position_ids
    return dict(
        segment_ids=segments,
        modality=modality,
        media_ids=media_ids,
        position_ids=positions,
        linear_positions=linear_positions,
    )


def prefix_key(
    *,
    model_sha256,
    tokenizer_sha256,
    processor_sha256,
    template,
    input_ids,
    media_hashes,
    position_ids,
    phase,
    adapter_sha256=None,
):
    value = dict(
        model=model_sha256,
        tokenizer=tokenizer_sha256,
        processor=processor_sha256,
        template=template,
        tokens=input_ids.tolist(),
        media=media_hashes,
        positions=position_ids.tolist(),
        phase=phase,
        adapter=adapter_sha256,
    )
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
