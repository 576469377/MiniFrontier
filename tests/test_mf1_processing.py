"""Packed position contracts, including repeated segment IDs and video time axes."""

import pytest
import torch

from minifrontier.models.minifrontier1 import MiniFrontier1Config
from minifrontier.models.minifrontier1.processing import token_metadata

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]


@pytest.mark.parametrize("device", DEVICES)
def test_packed_text_positions_reset_contiguous_runs_and_preserve_padding(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    config = MiniFrontier1Config.tiny()
    segments = torch.tensor(
        [
            [0, 0, 0, 5, 5, 5, 5, 0, 0, 0, -1, -1],
            [-1, -1, 7, 7, 7, 7, -1, -1, 7, 7, 7, 7],
            [-1] * 12,
        ],
        device=device,
    )
    ids = torch.full_like(segments, 25).masked_fill(segments.lt(0), config.pad_token_id)
    before = segments.clone()
    actual = token_metadata(ids, config, segment_ids=segments, offset=5, position_base=[20, 40, 60])
    positions = torch.tensor(
        [
            [20, 21, 22, 0, 1, 2, 3, 0, 1, 2, 30, 31],
            [40, 41, 0, 1, 2, 3, 46, 47, 0, 1, 2, 3],
            list(range(60, 72)),
        ],
        device=device,
    )
    linear = torch.tensor(
        [
            [5, 6, 7, 0, 1, 2, 3, 0, 1, 2, 15, 16],
            [5, 6, 0, 1, 2, 3, 11, 12, 0, 1, 2, 3],
            list(range(5, 17)),
        ],
        device=device,
    )
    torch.testing.assert_close(actual["position_ids"], positions.expand(3, -1, -1))
    torch.testing.assert_close(actual["linear_positions"], linear)
    torch.testing.assert_close(segments, before)
    assert not actual["modality"].any()
    assert actual["media_ids"].eq(-1).all()
    override = positions.expand(3, -1, -1).clone().add_(100)
    explicit = token_metadata(ids, config, segment_ids=segments, position_ids=override)
    torch.testing.assert_close(explicit["position_ids"], override)


@pytest.mark.parametrize("device", DEVICES)
def test_packed_video_and_image_axes_match_independent_samples(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    config = MiniFrontier1Config.tiny()
    ids = torch.full((1, 20), 25, dtype=torch.long, device=device)
    ids[:, 3:11] = config.image_token_id
    ids[:, 15:17] = config.image_token_id
    segments = torch.zeros_like(ids)
    segments[:, 13:] = 4
    media = [
        dict(
            batch_index=0,
            start=3,
            feature_count=8,
            grid_thw=torch.tensor([[2, 4, 4]], device=device),
            temporal_positions=[0, 7],
            resource_kind="video",
        ),
        dict(
            batch_index=0,
            start=15,
            feature_count=2,
            grid_thw=torch.tensor([[1, 2, 4]], device=device),
            resource_kind="image",
        ),
    ]
    actual = token_metadata(ids, config, media, segment_ids=segments)
    video_axes = torch.tensor(
        [[3, 3, 3, 3, 10, 10, 10, 10], [3, 3, 4, 4, 3, 3, 4, 4], [3, 4, 3, 4, 3, 4, 3, 4]],
        device=device,
    )
    torch.testing.assert_close(actual["position_ids"][:, 0, 3:11], video_axes)
    torch.testing.assert_close(
        actual["position_ids"][:, 0, 11:13],
        torch.tensor([11, 12], device=device).expand(3, -1),
    )
    for start, end, span in [(0, 13, media[0]), (13, 20, media[1])]:
        independent = token_metadata(
            ids[:, start:end], config, [dict(span, start=span["start"] - start)]
        )
        for key in ("position_ids", "linear_positions", "modality"):
            torch.testing.assert_close(actual[key][..., start:end], independent[key])
    assert actual["modality"][:, 3:11].eq(2).all()
    assert actual["modality"][:, 15:17].eq(1).all()
    # Grid and sample-boundary validation must remain active after vectorization.
    with pytest.raises(ValueError, match="cannot cross"):
        invalid = segments.clone()
        invalid[:, 8:] = 5
        token_metadata(ids, config, media, segment_ids=invalid)
    with pytest.raises(ValueError, match="exactly one"):
        token_metadata(ids, config)
