"""Native Vision-Exp resize, patchification and N-layout, with two explicit budgets."""

import io
from types import SimpleNamespace

from .upstream_processing import build_image_block, load_image


def process_image(
    image, *, start=0, patch_size=14, max_features=384, min_pixels=3136, max_span=None
):
    if max_features < 1 or min_pixels < 1:
        raise ValueError("pixel and feature budgets must be positive")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    # Official max_n_token bounds the complete sentinel span. The strategy also
    # specifies a separate content-feature cap. Enforce both; never call them equal.
    span_budget = max_span or (max_features * 2 + 16)
    if span_budget < 12:
        raise ValueError("span cannot fit native alignment and boundary tokens")
    while span_budget >= 12:
        args = SimpleNamespace(
            vision_patch_size=patch_size,
            vision_downsample_ratio=3,
            vision_min_pixels=min_pixels,
            vision_max_wh_ratio=32,
            vision_max_n_token=span_budget,
        )
        patches, nh, nw, lh, lw = load_image(dict(data=buffer.getvalue()), args)
        if lh * lw <= max_features:
            break
        span_budget -= max(1, (lh * lw - max_features) // 2)
    else:
        raise ValueError("image cannot fit native feature/span constraints")
    types, perm = build_image_block(lh, lw, start)
    return dict(
        start=start,
        patches=patches,
        n_vit_h=nh,
        n_vit_w=nw,
        types=types,
        perm=perm,
        feature_count=lh * lw,
        span_length=len(types),
        min_pixels=min_pixels,
        max_features=max_features,
        max_span=span_budget,
        processor="deepseek-vision-6821d6a-mini-budget-v1",
    )
