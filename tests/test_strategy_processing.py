"""Processor/source correspondences, N-layout, image visibility and migration."""

from dataclasses import asdict

import numpy as np
import pytest
import torch
from PIL import Image
from test_new_backbones import tiny_deepseek

from minifrontier.models.minideepseekv4 import MiniDeepSeekV4ForCausalLM
from minifrontier.models.minideepseekv4.migration import configure_visual_warmup, migrate_text_state
from minifrontier.models.minideepseekv4.processing import process_image as ds_image
from minifrontier.models.minideepseekv4.upstream_visibility import (
    get_image_visible,
    get_window_topk_idxs_visible,
)
from minifrontier.models.minideepseekv4.vision import DeepSeekVisionConfig
from minifrontier.models.minikimik3.processing import process_frames as kimi_frames
from minifrontier.models.miniqwen4.processing import position_ids
from minifrontier.models.miniqwen4.processing import process_frames as qwen_frames
from minifrontier.training.runtime import RouterBalance


def color_grid(size=84):
    y, x = np.mgrid[:size, :size]
    return Image.fromarray(np.stack((x * 3 % 256, y * 3 % 256, (x + y) % 256), -1).astype(np.uint8))


def test_moonvit_pixel_order_transparency_and_temporal_groups():
    image = color_grid(56)
    sample = kimi_frames([image], max_features=4)
    expected = torch.from_numpy(np.array(image).copy()).float().permute(2, 0, 1) / 255 * 2 - 1
    rebuilt = sample["patches"].reshape(4, 4, 3, 14, 14).permute(2, 0, 3, 1, 4).reshape(3, 56, 56)
    torch.testing.assert_close(rebuilt, expected)
    group = kimi_frames(
        [image, image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)],
        max_features=4,
        timestamps=[0.0, 0.125],
    )
    assert group["grid_thw"].tolist() == [[2, 4, 4]] and group["feature_count"] == 4
    assert not torch.equal(group["patches"][:16], group["patches"][16:])
    transparent = kimi_frames([Image.new("RGBA", (28, 28), (255, 0, 0, 0))], max_features=1)
    assert transparent["patches"].max() == 1
    assert transparent["patches"].min() == pytest.approx(180 / 255 * 2 - 1)
    with pytest.raises(ValueError, match="timestamps"):
        kimi_frames([image, image])


def test_qwen_native_conv3d_patch_order_and_mrope():
    first, second = color_grid(64), Image.new("RGB", (64, 64), (0, 0, 255))
    sample = qwen_frames([first, second], max_features=4, min_pixels=4096, timestamps=[0.0, 0.5])
    # Invert block-major patch order and verify the two temporal channels differ.
    restored = (
        sample["patches"]
        .reshape(1, 2, 2, 2, 2, 3, 2, 16, 16)
        .permute(0, 6, 5, 1, 3, 7, 2, 4, 8)
        .reshape(2, 3, 64, 64)
    )
    expected = (
        torch.stack(
            [torch.from_numpy(np.array(x)).permute(2, 0, 1) for x in [first, second]]
        ).float()
        / 255
        * 2
        - 1
    )
    torch.testing.assert_close(restored, expected)
    ids = torch.tensor([[1, 3, 7, 7, 7, 7, 20, 2]])
    pos = position_ids(ids, [dict(sample, start=2, batch_index=0)])
    assert pos[:, 0, 2:6].tolist() == [[2, 2, 2, 2], [2, 2, 3, 3], [2, 3, 2, 3]]
    assert pos[:, 0, 6:].tolist() == [[4, 5]] * 3
    with pytest.raises(ValueError, match="min_pixels"):
        qwen_frames([first], max_features=4)


def vision_model():
    config = tiny_deepseek(
        max_seq_len=128,
        vision_config=DeepSeekVisionConfig(
            depth=1, hidden_size=32, num_heads=2, intermediate_size=64, output_size=32
        ),
    )
    return MiniDeepSeekV4ForCausalLM(config)


def test_deepseek_native_span_visibility_hash_routes_and_gradients():
    torch.manual_seed(52)
    model = vision_model()
    media = ds_image(color_grid(), start=3, max_features=4, min_pixels=1764)
    ids = torch.tensor(
        [[1, 15, 20, *(model.config.vocab_size + media["types"]).tolist(), 30, 31, 2]]
    )
    labels = ids.clone()
    labels[:, : 3 + media["span_length"]] = -100
    media["batch_index"] = 0
    mask = torch.ones_like(ids, dtype=torch.bool)
    assert media["feature_count"] <= 4 and media["span_length"] > media["feature_count"]
    left, right = get_image_visible(ids, model.config.vocab_size, media["span_length"])
    indices = get_window_topk_idxs_visible(
        model.config.window_size, ids.shape[1], left, right, media["span_length"]
    )
    legal = torch.zeros((1, ids.shape[1], ids.shape[1]), dtype=torch.bool)
    legal.scatter_(2, indices.long().clamp_min(0), indices >= 0)
    balance = RouterBalance(model)
    with balance.capture(mask):
        result = model(ids, labels=labels, media=[media], attention_mask=mask)
    counts = [x.clone() for x in balance.counts]
    result.loss.backward()
    assert model.vision.vit.patch_embed.proj.weight.grad.abs().sum() > 0
    for old, new in zip(counts, balance.counts, strict=True):
        torch.testing.assert_close(old, new)
    for (gate, bias, _), count in zip(balance.gates, counts, strict=True):
        expected = (
            media["span_length"] if bias is gate.bias_vl else ids.shape[1] - media["span_length"]
        )
        assert count.sum() == expected * gate.topk
    model.eval()
    with torch.no_grad():
        baseline = model(ids, media=[media]).logits
        other = model(ids, media=[dict(media, patches=-media["patches"])]).logits
        torch.testing.assert_close(baseline[:, :3], other[:, :3])
        assert (baseline[:, -3:] - other[:, -3:]).abs().max() > 1e-6
        future = ids.clone()
        future[:, -2] = 40
        altered = model(future, media=[media]).logits
        torch.testing.assert_close(baseline[:, :-2], altered[:, :-2])
    with pytest.raises(ValueError, match="missing"):
        model(ids)
    with pytest.raises(ValueError, match=r"targets|labels"):
        model(ids, labels=ids, media=[media])


def test_deepseek_visual_migration_preserves_text_and_freezes_only_learned_text():
    text = MiniDeepSeekV4ForCausalLM(tiny_deepseek(max_seq_len=128))
    visual = vision_model()
    report = migrate_text_state(visual, asdict(text.config), text.state_dict())
    assert any(row["action"] == "initialized" for row in report["keys"])
    ids = torch.randint(10, 60, (1, 20))
    text.eval()
    visual.eval()
    torch.testing.assert_close(text(ids).logits, visual(ids).logits, rtol=0, atol=0)
    configure_visual_warmup(visual)
    for name, p in visual.named_parameters():
        assert p.requires_grad == name.startswith(("vision.", "image_"))
    config = asdict(text.config)
    config["route_scale"] = 1.5
    with pytest.raises(ValueError, match="silently"):
        migrate_text_state(visual, config, text.state_dict())
