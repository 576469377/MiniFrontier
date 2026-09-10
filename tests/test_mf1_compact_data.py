"""Compact shards preserve model inputs, media gradients' pixel inputs and sampler order."""

from dataclasses import asdict

import pytest
import torch

from minifrontier.data.minifrontier1 import RecordDataset, make_fixture
from minifrontier.data.minifrontier1_encoding import CompactDataset, encode_dataset, open_dataset
from minifrontier.models.minifrontier1 import MiniFrontier1Config, MiniFrontier1ForCausalLM
from minifrontier.models.minifrontier1.processing import token_metadata
from minifrontier.training.minifrontier1 import Sampler
from minifrontier.training.minifrontier1_curriculum import collate_records


@pytest.fixture
def encoded(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIFRONTIER_MIN_FREE_GIB", "1")
    source, output = tmp_path / "source", tmp_path / "compact"
    make_fixture(source)
    config = MiniFrontier1Config.tiny()
    manifest = encode_dataset(source, output, config, compact=True, shard_tokens=512)
    return source, output, config, manifest


def test_compact_text_image_video_model_outputs_and_shard_boundaries(encoded):
    source, output, config, manifest = encoded
    original, compact = (
        RecordDataset(source, "train", config),
        open_dataset(output, "train", config),
    )
    assert isinstance(compact, CompactDataset) and len(compact) == len(original)
    assert len(manifest["splits"]["train"]["parts"]) > 1
    assert not manifest["formal_admission"]
    expected, actual = [], []
    for index in (0, 31, 32, 63, 64, len(original) - 1):
        a, b = original[index], compact[index]
        assert a.keys() == b.keys()
        for name in ("input_ids", "labels"):
            torch.testing.assert_close(a[name], b[name], atol=0, rtol=0)
        for name in ("domain", "sample_id", "split_group", "media_hashes", "media_exposures"):
            assert a[name] == b[name]
        for before, after in zip(a["media"], b["media"], strict=True):
            torch.testing.assert_close(before["patches"], after["patches"], atol=0, rtol=0)
        for name, value in token_metadata(a["input_ids"], config, a["media"]).items():
            torch.testing.assert_close(
                value, token_metadata(b["input_ids"], config, b["media"])[name], atol=0, rtol=0
            )
        expected.append(a)
        actual.append(b)
    torch.manual_seed(42)
    model = MiniFrontier1ForCausalLM(config).eval()
    with torch.no_grad():
        batches = [collate_records(items, config.pad_token_id) for items in (expected, actual)]
        results = [
            model(
                item["input_ids"],
                labels=item["labels"],
                media=item["media"],
                segment_ids=item["segment_ids"],
            )
            for item in batches
        ]
    torch.testing.assert_close(results[0].logits, results[1].logits, atol=0, rtol=0)
    torch.testing.assert_close(results[0].lm_loss, results[1].lm_loss, atol=0, rtol=0)


def test_compact_sampler_uses_index_without_retokenizing_or_decoding(encoded, monkeypatch):
    source, output, config, _ = encoded
    original, compact = (
        RecordDataset(source, "train", config),
        CompactDataset(output, "train", config),
    )
    weights = {"zh_general": 1 / 3, "caption": 1 / 3, "video": 1 / 3}
    # Use the fixture's actual domains rather than encoding assumptions.
    weights = dict.fromkeys({original.record(i)["domain"] for i in range(len(original))}, 1 / 3)
    before = Sampler(original, 9, weights, length_filter=True)

    def forbidden(*args, **kwargs):
        raise AssertionError("sampler initialization must use compact index only")

    monkeypatch.setattr(CompactDataset, "__getitem__", forbidden)
    after = Sampler(compact, 9, weights, length_filter=True)
    assert before.state_dict() == after.state_dict()
    assert before.lengths == after.lengths


def test_compact_loader_rejects_corruption_and_wrong_config(encoded):
    _, output, config, manifest = encoded
    changed = MiniFrontier1Config(
        **dict(asdict(config), max_position_embeddings=config.max_position_embeddings * 2)
    )
    with pytest.raises(ValueError, match="identity"):
        CompactDataset(output, "train", changed)
    entry = manifest["splits"]["train"]["parts"][0]["files"]["tokens.bin"]
    path = output / entry["name"]
    with path.open("r+b") as handle:
        handle.write(b"\xff\xff")
    with pytest.raises(ValueError, match="checksum"):
        CompactDataset(output, "train", config)[0]


def test_compact_ids_never_silently_wrap_uint16(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIFRONTIER_MIN_FREE_GIB", "1")
    source = tmp_path / "source"
    make_fixture(source)
    original = RecordDataset.__getitem__

    def large_id(self, index):
        item = original(self, index)
        item["input_ids"][0, 2] = 65536
        item["labels"][0, 2] = 65536
        return item

    monkeypatch.setattr(RecordDataset, "__getitem__", large_id)
    config = MiniFrontier1Config(**dict(asdict(MiniFrontier1Config.tiny()), vocab_size=65537))
    manifest = encode_dataset(source, tmp_path / "wide", config, compact=True)
    assert manifest["token_dtype"] == "<u4"
    assert CompactDataset(tmp_path / "wide", "train", config)[0]["input_ids"][0, 2] == 65536
    with pytest.raises(ValueError, match="truncation"):
        encode_dataset(source, tmp_path / "narrow", MiniFrontier1Config.tiny(), compact=True)
