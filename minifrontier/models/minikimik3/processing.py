"""Pinned MoonViT image preprocessing with bounded mini patch budgets.

Videos arrive as decoded, timestamped frames. Grouping up to four actual frames
is a local loader adapter; no duplicate still images are counted as videos.
"""

from itertools import pairwise

import numpy as np
import torch

from .upstream_processing import (
    TransparentBgConfig,
    image_to_np,
    navit_patchify,
    navit_resize_image,
    normalize,
)


def process_frames(frames, *, patch_size=14, max_features=256, timestamps=None):
    if not 1 <= len(frames) <= 4 or max_features < 1:
        raise ValueError("one MoonViT temporal group contains 1-4 decoded frames")
    if timestamps is not None and (
        len(timestamps) != len(frames) or any(b <= a for a, b in pairwise(timestamps))
    ):
        raise ValueError("video frame timestamps must be strictly increasing")
    if len(frames) > 1 and timestamps is None:
        raise ValueError("genuine video groups require source timestamps")
    if len({im.size for im in frames}) != 1:
        raise ValueError("video resolution changes must start a new group")
    width, height = frames[0].size
    # Native resize rounds padded dimensions up. Reduce its area limit until the
    # resulting language feature count fits, instead of silently cropping features.
    limit = max_features * 4
    while True:
        resize = navit_resize_image(width, height, patch_size, 2, limit, 512, None)
        if resize["num_tokens"] <= max_features:
            break
        limit -= 1
        if limit < 1:
            raise ValueError("image aspect ratio cannot fit the requested feature budget")
    bg = TransparentBgConfig(
        pattern="chessboard", chessboard_square_size=8, chessboard_gray_value=180
    )
    pixels = []
    for frame in frames:
        value = image_to_np(
            frame,
            (resize["new_width"], resize["new_height"]),
            "resize",
            transparent_bg_config=bg,
            transparent_bg_fill_stage="after_resize",
        )
        pixels.append(np.pad(value, ((0, resize["pad_height"]), (0, resize["pad_width"]), (0, 0))))
    result = navit_patchify(
        normalize(np.stack(pixels), np.array([0.5] * 3), np.array([2.0] * 3)), patch_size
    )
    return dict(
        patches=torch.from_numpy(result["pixel_values"].copy()),
        grid_thw=torch.from_numpy(result["grid_thw"]).long()[None],
        feature_count=resize["num_tokens"],
        timestamps=list(timestamps or []),
        original_size=(width, height),
        processor="moonvit-c5d1dd4-mini-budget-v1",
    )


def process_image(image, **kwargs):
    return process_frames([image], **kwargs)
