"""Canonical source videos preserve frame identity, temporal inputs and answer CE."""

import copy
import json
import random
from dataclasses import asdict

import pytest
import torch
from PIL import Image

from minifrontier.data import sha256
from minifrontier.data.corpus import CorpusBuilder, train_tokenizer
from minifrontier.data.media_hash import decoded_hashes
from minifrontier.data.minifrontier1 import SPECIAL_TOKENS, digest, encode_record, validate_record
from minifrontier.data.minifrontier1_components import ComponentDataset, assemble_components
from minifrontier.data.minifrontier1_encoding import (
    CompactDataset,
    canonical_video_record,
    encode_canonical_videos,
)
from minifrontier.models.minifrontier1 import MiniFrontier1Config, MiniFrontier1ForCausalLM
from minifrontier.models.minifrontier1.processing import token_metadata


@pytest.fixture
def video_corpus(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIFRONTIER_MIN_FREE_GIB", "1")
    root = tmp_path / "videos"
    builder = CorpusBuilder(root)
    for number in range(3):
        frames, file_hashes, rgb_hashes = [], [], []
        for frame in range(4):
            path = root / f"clip-{number}-{frame}.png"
            image = Image.frombytes(
                "RGB", (32, 32), random.Random(10 * number + frame).randbytes(3072)
            )
            image.save(path)
            frames.append(path.name)
            file_hashes.append(sha256(path))
            rgb_hashes.append(decoded_hashes(image)["rgb_sha256"])
        caption = f"Clip {number} has a complete caption containing literal <|video_end|>."
        assert builder.add(
            dict(
                source="video-fixture",
                revision="pinned",
                item_id=str(number),
                group_id=str(number),
                license="CC0-1.0",
                task="video",
                lang="en",
                stage="pretrain",
                text=caption,
                turns=[
                    dict(role="user", content=f"Describe clip {number}."),
                    dict(role="assistant", content=caption),
                ],
                media=[
                    dict(
                        kind="video",
                        video_id=f"clip-{number}",
                        frames=frames,
                        frame_sha256=file_hashes,
                        frame_rgb_sha256=rgb_hashes,
                        sha256=digest(file_hashes),
                        rgb_sha256=digest(rgb_hashes),
                        timestamps=[0.0, 0.3, 1.2, 2.1],
                        width=32,
                        height=32,
                    )
                ],
            )
        )
    builder.finalize(
        split_locks={"source-group:video-fixture:1": "val", "source-group:video-fixture:2": "test"}
    )
    rows = [
        (s, json.loads(p), g)
        for s, p, g in builder.db.execute(
            "SELECT split,payload,group_root FROM samples ORDER BY id"
        )
    ]
    builder.db.close()
    (root / "source-audit.json").write_text(
        json.dumps(
            dict(status="candidate_slice_complete_pending_admission", formal_admission=False)
        )
    )
    tokenizer = tmp_path / "tokenizer.json"
    train_tokenizer(root, tokenizer, 350, special_tokens=SPECIAL_TOKENS)
    config = MiniFrontier1Config(
        **dict(asdict(MiniFrontier1Config.tiny(350)), max_position_embeddings=1024)
    )
    return root, tokenizer, config, rows


def test_canonical_video_roundtrip_preserves_temporal_inputs_and_logits(video_corpus, tmp_path):
    from tokenizers import Tokenizer

    root, tokenizer_path, config, rows = video_corpus
    output = tmp_path / "encoded"
    manifest = encode_canonical_videos(root, tokenizer_path, output, config, max_features=16)
    assert manifest["kind"] == "canonical_video_component"
    assert manifest["splits"]["train"]["counts"]["video_examples"] == 1
    assert manifest["splits"]["train"]["counts"]["frames"] == 4
    assert not manifest["raw_media_copied"] and not manifest["formal_admission"]
    composition = tmp_path / "composition"
    assembled = assemble_components([output], composition, config)
    assert assembled["kind"] == "canonical_text_image_video_composition"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    torch.manual_seed(42)
    model = MiniFrontier1ForCausalLM(config).eval()
    for split, row, group in rows:
        original = encode_record(
            canonical_video_record(row, group, max_features=16), tokenizer, config, root
        )
        compact = ComponentDataset(composition, split, config)[0]
        for key in ("input_ids", "labels"):
            torch.testing.assert_close(original[key], compact[key], atol=0, rtol=0)
        for key in token_metadata(original["input_ids"], config, original["media"]):
            torch.testing.assert_close(
                token_metadata(original["input_ids"], config, original["media"])[key],
                token_metadata(compact["input_ids"], config, compact["media"])[key],
                atol=0,
                rtol=0,
            )
        assert compact["media"][0]["timestamps"] == [0.0, 0.3, 1.2, 2.1]
        torch.testing.assert_close(
            original["media"][0]["patches"], compact["media"][0]["patches"], atol=0, rtol=0
        )
        with torch.no_grad():
            a = model(original["input_ids"], labels=original["labels"], media=original["media"])
            b = model(compact["input_ids"], labels=compact["labels"], media=compact["media"])
        torch.testing.assert_close(a.logits, b.logits, atol=0, rtol=0)
        torch.testing.assert_close(a.lm_loss, b.lm_loss, atol=0, rtol=0)


@pytest.mark.parametrize("fault", ["timestamps", "frame_count", "answer"])
def test_video_adapter_rejects_incomplete_or_unordered_sources(video_corpus, fault):
    root, _, _, rows = video_corpus
    row, group = copy.deepcopy(rows[0][1]), rows[0][2]
    if fault == "timestamps":
        row["media"][0]["timestamps"][1] = 0.0
    elif fault == "frame_count":
        row["media"][0]["frame_rgb_sha256"].pop()
    else:
        row["turns"][-1]["content"] = ""
    with pytest.raises(ValueError):
        validate_record(canonical_video_record(row, group, max_features=16), root)


def test_video_frame_mutation_is_rejected_by_compact_loader(video_corpus, tmp_path):
    root, tokenizer, config, rows = video_corpus
    output = tmp_path / "encoded"
    encode_canonical_videos(root, tokenizer, output, config, max_features=16)
    split, row, _ = rows[0]
    path = root / row["media"][0]["frames"][0]
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="source/hash"):
        CompactDataset(output, split, config)[0]
