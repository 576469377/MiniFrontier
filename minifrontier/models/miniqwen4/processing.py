"""Native smart_resize, Conv3D patch ordering and temporal/spatial mRoPE positions."""

from itertools import pairwise

import numpy as np
import torch
import torch.nn.functional as F

from .upstream_processing import smart_resize


def process_frames(frames, *, patch_size=16, max_features=256, min_pixels=65536, timestamps=None):
    if not frames or max_features < 1 or min_pixels > max_features * (2 * patch_size) ** 2:
        raise ValueError("invalid Qwen pixel budget; explicitly lower min_pixels for tiny pilots")
    if len({im.size for im in frames}) != 1:
        raise ValueError("video resolution changes must start a new segment")
    if len(frames) > 1 and (
        timestamps is None
        or len(timestamps) != len(frames)
        or any(b <= a for a, b in pairwise(timestamps))
    ):
        raise ValueError("genuine videos require increasing source timestamps")
    height, width = frames[0].height, frames[0].width
    h, w = smart_resize(
        height,
        width,
        factor=patch_size * 2,
        min_pixels=min_pixels,
        max_pixels=max_features * (patch_size * 2) ** 2,
    )
    values = torch.stack(
        [torch.from_numpy(np.array(im.convert("RGB"))).permute(2, 0, 1) for im in frames]
    )
    # torchvision's tensor resize dispatches to this same uint8 bicubic operator.
    values = F.interpolate(values, (h, w), mode="bicubic", align_corners=False, antialias=True)
    values = (values.float() / 255 - 0.5) / 0.5
    original_frames = len(frames)
    if len(frames) % 2:
        values = torch.cat((values, values[-1:]), dim=0)
    t, gh, gw = values.shape[0] // 2, h // patch_size, w // patch_size
    # t, temporal, c, block_h, merge_h, p_h, block_w, merge_w, p_w
    patches = values.reshape(t, 2, 3, gh // 2, 2, patch_size, gw // 2, 2, patch_size)
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(-1, 3 * 2 * patch_size**2)
    return dict(
        patches=patches,
        grid_thw=torch.tensor([[t, gh, gw]]),
        feature_count=t * gh * gw // 4,
        timestamps=list(timestamps or []),
        original_frames=original_frames,
        padded_frames=values.shape[0] - original_frames,
        processor="qwen4-4177486-mini-budget-v1",
    )


def process_image(image, **kwargs):
    return process_frames([image], **kwargs)


def position_ids(input_ids, media):
    """Three axes for the *complete* sequence, preserving actual merged-grid order.

    The mini video convention uses integer group indices, with original seconds
    separately retained in metadata; it is not represented as calibrated seconds.
    """
    positions = torch.zeros((3, *input_ids.shape), device=input_ids.device, dtype=torch.long)
    for batch in range(input_ids.shape[0]):
        cursor, base = 0, 0
        for span in sorted(
            (s for s in media if s["batch_index"] == batch), key=lambda s: s["start"]
        ):
            start = span["start"]
            if start < cursor:
                raise ValueError("overlapping media spans")
            n = start - cursor
            positions[:, batch, cursor:start] = torch.arange(
                base, base + n, device=input_ids.device
            )
            base += n
            for t, h, w in span["grid_thw"].tolist():
                hh, ww = h // 2, w // 2
                axes = torch.stack(
                    torch.meshgrid(
                        torch.arange(t), torch.arange(hh), torch.arange(ww), indexing="ij"
                    )
                ).flatten(1)
                count = axes.shape[1]
                positions[:, batch, start : start + count] = axes.to(input_ids.device) + base
                base += max(t, hh, ww)
                start += count
            if start - span["start"] != span["feature_count"]:
                raise ValueError("mRoPE grid and feature count disagree")
            cursor = start
        positions[:, batch, cursor:] = torch.arange(
            base, base + input_ids.shape[1] - cursor, device=input_ids.device
        )
    return positions
