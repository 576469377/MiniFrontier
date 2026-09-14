"""MF1.1 shares immutable MF1 inputs only under an explicit processing contract."""

# Imported fixture names intentionally also name the test parameters below.
# ruff: noqa: F811

import json
from dataclasses import asdict

import pytest
import torch

from minifrontier.data import sha256
from minifrontier.data.minifrontier1 import digest
from minifrontier.data.minifrontier1_components import (
    ComponentDataset,
    assemble_components,
    create_model_compatibility_view,
)
from minifrontier.data.minifrontier1_encoding import encode_canonical_videos
from minifrontier.models.minifrontier1 import MiniFrontier1Config
from minifrontier.models.minifrontier1.processing import token_metadata
from tests.test_mf1_canonical_media import components, corpus  # noqa: F401
from tests.test_mf1_canonical_video import video_corpus  # noqa: F401


def target_config(config, **overrides):
    return MiniFrontier1Config(
        **dict(
            asdict(config),
            model_version="1.1-reference-v1",
            mtp_enabled=False,
            mtp_loss_coef=0.0,
            **overrides,
        )
    )


def compare_inputs(before, after, old_config, new_config):
    for key in ("input_ids", "labels"):
        torch.testing.assert_close(before[key], after[key], atol=0, rtol=0)
    for key in ("sample_id", "split_group", "domain", "media_exposures"):
        assert before[key] == after[key]
    original_positions = token_metadata(before["input_ids"], old_config, before["media"])
    reused_positions = token_metadata(after["input_ids"], new_config, after["media"])
    for key in original_positions:
        torch.testing.assert_close(original_positions[key], reused_positions[key], atol=0, rtol=0)
    for a, b in zip(before["media"], after["media"], strict=True):
        assert a.keys() == b.keys()
        for key in a:
            if isinstance(a[key], torch.Tensor):
                torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
            else:
                assert a[key] == b[key]


def test_reuse_preserves_text_images_windows_and_original_admission(components, tmp_path):
    roots, config = components
    source, output = tmp_path / "source", tmp_path / "reuse"
    manifest = assemble_components(roots, source, config)
    manifest.update(formal_admission=True, main_budget_eligible=True, human_review_completed=False)
    (source / "manifest.json").write_text(json.dumps(manifest))
    source_sha = sha256(source / "manifest.json")
    target = target_config(config)
    with pytest.raises(ValueError, match="composition config"):
        ComponentDataset(source, "train", target)
    reused = create_model_compatibility_view(source, output, asdict(config), target)
    assert sha256(source / "manifest.json") == source_sha
    assert reused["formal_admission"] is False and reused["main_budget_eligible"] is False
    binding = reused["model_compatibility"]
    assert binding["source_formal_admission"] is True
    assert binding["source_config_sha256"] == digest(asdict(config))
    assert binding["target_config_sha256"] == digest(asdict(target))
    assert digest(binding["target_config"]) == digest(asdict(target))
    assert binding["payload_files_copied"] == 0 and binding["raw_media_copied"] is False
    assert {p.name for p in output.iterdir()} == {"manifest.json", "tokenizer.json"}
    assert (source / "tokenizer.json").samefile(output / "tokenizer.json")
    for split in ("train", "val", "test"):
        old = ComponentDataset(source, split, config)
        new = ComponentDataset(output, split, target)
        assert len(new) == len(old)
        for index in range(len(old)):
            compare_inputs(old[index], new[index], config, target)
            assert old.ce_count_at(index) == new.ce_count_at(index)
            if old.windowable_at(index):
                compare_inputs(
                    old.window_at(index, 0, 128), new.window_at(index, 0, 128), config, target
                )


@pytest.mark.parametrize(
    "change",
    [
        dict(image_token_id=13),
        dict(max_position_embeddings=4096),
        dict(vision_config={"patch_size": 8}),
    ],
)
def test_reuse_rejects_changed_processing_fields_before_writing(components, tmp_path, change):
    roots, config = components
    source, output = tmp_path / "source", tmp_path / "reuse"
    assemble_components(roots, source, config)
    if "vision_config" in change:
        change = dict(vision_config=dict(asdict(config.vision_config), **change["vision_config"]))
    with pytest.raises(ValueError, match="processing contract differs"):
        create_model_compatibility_view(
            source, output, asdict(config), target_config(config, **change)
        )
    assert not output.exists()


def test_reuse_preserves_remote_media_cache_location_and_direct_policy(components, tmp_path):
    roots, config = components
    source, output = tmp_path / "source", tmp_path / "nested" / "reuse"
    manifest = assemble_components(roots, source, config)
    policy = dict(
        base_url="http://127.0.0.1:18390/",
        uri_prefix="images/",
        cache_dir="../shared-media-cache",
        max_bytes=1024**3,
        max_file_bytes=64 * 1024**2,
        reserve_bytes=80 * 1024**3,
    )
    manifest["components"][0]["media_access"] = policy
    (source / "manifest.json").write_text(json.dumps(manifest))
    reused = create_model_compatibility_view(source, output, asdict(config), target_config(config))
    actual = reused["components"][0]["media_access"]
    assert (source / policy["cache_dir"]).resolve() == (output / actual["cache_dir"]).resolve()
    assert {k: v for k, v in actual.items() if k != "cache_dir"} == {
        k: v for k, v in policy.items() if k != "cache_dir"
    }


@pytest.mark.parametrize(
    "fault", ["source_manifest", "source_config", "target_config", "child_manifest", "tokens"]
)
def test_reuse_keeps_source_config_component_and_payload_hash_checks(components, tmp_path, fault):
    roots, config = components
    source, output = tmp_path / "source", tmp_path / "reuse"
    assemble_components(roots, source, config)
    target = target_config(config)
    create_model_compatibility_view(source, output, asdict(config), target)
    if fault == "source_manifest":
        path = source / "manifest.json"
        path.write_text(path.read_text() + " ")
    elif fault in {"source_config", "target_config"}:
        path = output / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["model_compatibility"][fault]["max_position_embeddings"] += 1
        path.write_text(json.dumps(manifest))
    elif fault == "child_manifest":
        path = roots[0] / "manifest.json"
        path.write_text(path.read_text() + " ")
    else:
        child = json.loads((roots[0] / "manifest.json").read_text())
        path = roots[0] / child["splits"]["train"]["parts"][0]["files"]["tokens.bin"]["name"]
        value = path.read_bytes()
        path.write_bytes(bytes([value[0] ^ 1]) + value[1:])
    with pytest.raises(ValueError):
        dataset = ComponentDataset(output, "train", target)
        dataset[0]


def test_video_reuse_preserves_frame_pixels_and_timestamps(video_corpus, tmp_path):
    root, tokenizer, config, _ = video_corpus
    component, source, output = tmp_path / "video", tmp_path / "source", tmp_path / "reuse"
    encode_canonical_videos(root, tokenizer, component, config, max_features=16)
    assemble_components([component], source, config)
    target = target_config(config)
    create_model_compatibility_view(source, output, asdict(config), target)
    for split in ("train", "val", "test"):
        old = ComponentDataset(source, split, config)[0]
        new = ComponentDataset(output, split, target)[0]
        compare_inputs(old, new, config, target)
        assert new["media"][0]["timestamps"] == [0.0, 0.3, 1.2, 2.1]
