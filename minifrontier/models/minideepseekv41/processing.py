"""Prepare local PIL images in the V4.1 row-major layout."""

from types import SimpleNamespace

import numpy as np
import torch
from PIL import ImageOps

from .upstream_processing import plan_image_grid


def process_image(image, *, start=0, patch_size=14, max_features=64, min_pixels=3136):
    if min(patch_size, max_features, min_pixels) < 1:
        raise ValueError("patch size and image budgets must be positive")
    # Bound the whole span. Each grid includes at least one newline and two delimiters.
    args = SimpleNamespace(
        vision_patch_size=patch_size,
        vision_downsample_ratio=3,
        vision_max_wh_ratio=None,
        vision_min_pixels=min_pixels,
        vision_max_n_token=max_features + 3,
    )
    rows, cols, height, width = plan_image_grid(image.width, image.height, args)
    rgb = ImageOps.pad(image.convert("RGB"), (width, height), color=(127, 127, 127))
    pixels = torch.from_numpy(np.asarray(rgb, dtype=np.float32)).permute(2, 0, 1) / 255
    pixels = ((pixels - 0.5) / 0.5).to(torch.bfloat16)
    h, w = height // patch_size, width // patch_size
    patches = pixels.reshape(3, h, patch_size, w, patch_size).permute(1, 3, 0, 2, 4)
    # Mini's start/feature/newline/end IDs are 0/2/3/4. Aligner rows stay in reading order.
    types = torch.tensor([0] + ([2] * cols + [3]) * rows + [4], dtype=torch.long)
    return dict(
        start=start,
        patches=patches.reshape(h * w, 3, patch_size, patch_size),
        n_vit_h=h,
        n_vit_w=w,
        types=types,
        feature_count=rows * cols,
        span_length=len(types),
    )
