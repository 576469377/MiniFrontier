"""Decoded RGB content hashes and deterministic DCT pHash for split grouping."""

import hashlib

import numpy as np
from PIL import Image

PHASH_BANDS = ((0, 10), (10, 9), (19, 9), (28, 9), (37, 9), (46, 9), (55, 9))


def decoded_hashes(image):
    rgb = image.convert("RGB")
    # Include dimensions: differently shaped arrays with the same bytes differ.
    payload = f"{rgb.width}x{rgb.height}:RGB\0".encode() + rgb.tobytes()
    luminance = np.asarray(
        rgb.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64
    )
    indices = np.arange(32, dtype=np.float64)
    basis = np.cos(np.pi / 32 * np.arange(8)[:, None] * (indices[None] + 0.5))
    low = basis @ luminance @ basis.T
    bits = low.flatten() > np.median(low.flatten()[1:])
    phash = int.from_bytes(np.packbits(bits).tobytes(), "big")
    return dict(
        rgb_sha256=hashlib.sha256(payload).hexdigest(),
        phash=f"{phash:016x}",
        width=rgb.width,
        height=rgb.height,
        phash_version="dct32-low8-median-v1",
    )
