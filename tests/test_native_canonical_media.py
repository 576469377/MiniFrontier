"""Canonical visual PT preserves literal text and shares audited text storage."""

import json
import random
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from test_miniqwen4 import tiny_config
from test_new_backbones import tiny_kimi
from tokenizers import Tokenizer

from minifrontier.data import StageDataset, sha256
from minifrontier.data.corpus import CorpusBuilder, encode_corpus, train_tokenizer
from minifrontier.data.media_hash import decoded_hashes
from minifrontier.data.native import audit_native_encoding, encode_native
from minifrontier.data.partitions import create_partition_view
from minifrontier.models.minikimik3 import MiniKimiK3ForCausalLM
from minifrontier.models.minikimik3.vision import KimiVisionConfig
from minifrontier.models.miniqwen4 import MiniQwen4ForCausalLM
from minifrontier.models.miniqwen4.vision import QwenVisionConfig
from minifrontier.multimodal import collate, prepare_record, pretraining_tokens


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIFRONTIER_MIN_FREE_GIB", "1")
    root = tmp_path / "canonical"
    builder = CorpusBuilder(root)
    for i, task in enumerate(("caption", "vqa", "ocr_document", "chart_table")):
        image = Image.frombytes("RGB", (32, 24), random.Random(i).randbytes(32 * 24 * 3))
        path = root / f"{i}.png"
        image.save(path)
        question = f"Describe image {i}. Is <|image|> printed?"
        answer = f"Complete answer {i}, including literal <|eos|>."
        assert builder.add(
            dict(
                source="fixture",
                revision="fixed",
                item_id=str(i),
                group_id=str(i),
                license="CC0-1.0",
                lang="en",
                stage="pretrain",
                task=task,
                text="<|image|>\nUser: " + question + "\nAssistant: " + answer,
                visual_question=question,
                visual_answer=answer,
                reference_tokens=30,
                media=[
                    dict(kind="image", path=path.name, sha256=sha256(path), **decoded_hashes(image))
                ],
            )
        )
    builder.finalize(
        split_locks={"source-group:fixture:2": "val", "source-group:fixture:3": "test"}
    )
    rows = [
        (s, json.loads(p), g)
        for s, p, g in builder.db.execute(
            "SELECT split,payload,group_root FROM samples ORDER BY id"
        )
    ]
    builder.db.close()
    tokenizer = tmp_path / "tokenizer.json"
    train_tokenizer(root, tokenizer, 350)
    return root, tokenizer, None, rows


@pytest.fixture
def inputs(corpus, tmp_path):
    root, tokenizer, _, rows = corpus
    text = CorpusBuilder(tmp_path / "canonical-text")
    sentences = (
        "The museum preserves rare books and maps for future generations.",
        "A spacecraft measures the magnetic field around a distant planet.",
        "Insects communicate through chemical signals produced by their glands.",
    )
    for i, sentence in enumerate(sentences):
        assert text.add(
            dict(
                source="text",
                revision="fixed",
                item_id=str(i),
                group_id=str(i),
                license="CC0-1.0",
                lang="en",
                stage="pretrain",
                task="en_edu",
                text=sentence,
            )
        )
    text.finalize(split_locks={"source-group:text:1": "val", "source-group:text:2": "test"})
    text.db.close()
    shared = tmp_path / "shared-text"
    encode_corpus(text.root, tokenizer, shared, max_length=512)
    return root, tokenizer, rows, shared


@pytest.mark.parametrize("family", ["minikimik3", "miniqwen4"])
def test_canonical_literals_stay_text_and_full_pt_targets_reach_vision(corpus, family):
    root, tokenizer_path, _, rows = corpus
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    row = rows[0][1]
    expected = "\nUser: " + row["visual_question"] + "\nAssistant: " + row["visual_answer"]
    assert "<|image|>" in expected and "<|eos|>" in expected
    before = tokenizer.encode_special_tokens
    batch = prepare_record(
        row,
        tokenizer,
        family,
        root=root,
        max_features=4,
        min_pixels=1024 if family == "miniqwen4" else None,
    )
    assert tokenizer.encode_special_tokens == before
    target = batch.labels[0][batch.labels[0].ne(-100)].tolist()
    assert tokenizer.decode(target, skip_special_tokens=True) == expected
    assert set(target).intersection(range(20)) == {2}
    assert batch.labels[0, -1] == 2 and batch.input_ids.eq(2).sum() == 1
    assert batch.input_ids.eq(7).sum() == batch.image_features
    if family == "minikimik3":
        vision = KimiVisionConfig(
            depth=1,
            hidden_size=32,
            qkv_hidden_size=48,
            num_heads=2,
            intermediate_size=64,
            output_size=32,
        )
        model = MiniKimiK3ForCausalLM(
            tiny_kimi(vocab_size=512, max_position_embeddings=512, vision_config=vision)
        )
    else:
        vision = QwenVisionConfig(
            depth=1, hidden_size=32, num_heads=2, intermediate_size=64, output_size=32
        )
        model = MiniQwen4ForCausalLM(
            tiny_config(
                vocab_size=512, hidden_size=32, max_position_embeddings=512, vision_config=vision
            )
        )
    result = model(batch.input_ids, labels=batch.labels, return_logits=False, **batch.extras)
    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.vision.parameters())
    with pytest.raises(ValueError, match="complete question/answer"):
        pretraining_tokens(dict(row, visual_answer=""), tokenizer)
    assert tokenizer.encode_special_tokens == before
    with pytest.raises(ValueError, match="raw media hash"):
        prepare_record(
            dict(row, media=[dict(row["media"][0], sha256="bad")]),
            tokenizer,
            family,
            root=root,
            max_features=4,
            min_pixels=1024 if family == "miniqwen4" else None,
        )


@pytest.mark.parametrize("family", ["minikimik3", "miniqwen4"])
def test_native_reuses_text_files_and_resolves_pixels_through_partition(
    inputs, tmp_path, family, monkeypatch
):
    monkeypatch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(free=900 * 1024**3))
    root, tokenizer, rows, shared = inputs
    original = sha256(root / "corpus.sqlite")
    (root / "source-audit.json").write_text(
        json.dumps(
            dict(status="candidate_slice_complete_pending_admission", formal_admission=False)
        )
    )
    integrity = root / "integrity-and-split-audit.json"
    integrity.write_text(json.dumps(dict(cross_split_groups=0)))
    group = next(g for split, _, g in rows if split == "train")
    proposal = tmp_path / "reservation.json"
    proposal.write_text(
        json.dumps(
            dict(
                corpus_manifest_sha256=sha256(root / "corpus-manifest.json"),
                integrity_audit_sha256=sha256(integrity),
                minimum_validation_group_fraction=0.005,
                groups={group: "val"},
            )
        )
    )
    view = tmp_path / "partition"
    create_partition_view(root, proposal, view)
    output = tmp_path / family
    manifest = encode_native(
        view,
        tokenizer,
        output,
        family,
        text_encoding=shared,
        max_length=512,
        max_features=4,
        min_pixels=1024 if family == "miniqwen4" else None,
        max_gib=0.01,
    )
    assert manifest["shared_text_files_copied"] == 0 and not manifest["formal_admission"]
    assert manifest["media_root"] == str(root)
    assert sha256(root / "corpus.sqlite") == original
    assert not list(output.glob("*.bin")) and not list(output.glob("pretrain.*.labels*"))
    assert all(p.suffix != ".png" for p in output.rglob("*"))
    assert (output / manifest["text_source"]["path"]).resolve() == shared
    audit = audit_native_encoding(view, output, output / "encoding-audit.json")
    assert audit["status"] == "mechanical_checks_passed_pending_quality_admission"
    assert sum(v["counts"]["records"] for v in audit["splits"].values()) == 4
    assert not audit["errors"] and not audit["formal_admission"]
    for split, media_examples in (("train", 1), ("val", 2), ("test", 1)):
        data = StageDataset(output, "pretrain", split, 512)
        text = StageDataset(shared, "pretrain", split, 512)
        assert len(data) == len(text) + media_examples
        batch = collate([data[i] for i in range(len(data))], "cpu")
        assert batch.image_count == media_examples
        assert int(batch.labels[:, 1:].ne(-100).sum()) == sum(data.documents.ce_counts)
        assert torch.equal(data[0].input_ids[0], text[0][0])
        serialized = [
            json.loads(line)
            for line in (output / f"pretrain.{split}.media.jsonl").read_text().splitlines()
        ]
        assert (group in {v["split_group"] for v in serialized}) == (split == "val")
    # A different shared manifest cannot be accepted merely because the files still exist.
    source_manifest = shared / "manifest.json"
    source_manifest.write_text(source_manifest.read_text() + "\n")
    with pytest.raises(ValueError, match="shared native text manifest"):
        StageDataset(output, "pretrain", "train", 512)


def test_native_disk_limit_never_publishes_partial_manifest_or_changes_shared_data(
    inputs, tmp_path
):
    root, tokenizer, _, shared = inputs
    checksum = sha256(shared / "manifest.json")
    output = tmp_path / "bounded"
    budget = (Path(tokenizer).stat().st_size + 128 * 1024 + 10) / 1024**3
    with pytest.raises(ValueError, match="disk budget"):
        encode_native(
            root,
            tokenizer,
            output,
            "minikimik3",
            text_encoding=shared,
            max_features=4,
            max_gib=budget,
        )
    assert not (output / "manifest.json").exists()
    assert sum(p.stat().st_size for p in output.iterdir() if p.is_file()) <= budget * 1024**3
    assert sha256(shared / "manifest.json") == checksum


@pytest.mark.parametrize("size", [(32, 24), (60, 20), (100, 200)])
def test_qwen_small_image_minimum_rounding_cannot_exceed_phase_pixel_cap(size):
    from minifrontier.models.miniqwen4.processing import process_frames
    from minifrontier.models.miniqwen4.upstream_processing import smart_resize

    h, w = smart_resize(size[1], size[0], factor=32, min_pixels=50176, max_pixels=50176)
    assert h * w // 1024 > 49  # The upstream ceil alone exceeds this mini budget.
    result = process_frames([Image.new("RGB", size)], max_features=49, min_pixels=50176)
    assert result["feature_count"] <= 49
    assert result["patches"].shape[0] == 4 * result["feature_count"]


def test_native_full_audit_rejects_rehashed_partial_text_supervision(inputs, tmp_path):
    root, tokenizer, _, shared = inputs
    output = tmp_path / "encoded"
    manifest = encode_native(
        root, tokenizer, output, "minikimik3", text_encoding=shared, max_features=4
    )
    meta = manifest["stages"]["pretrain"]["val"]["media"]
    path = output / meta["file"]
    row = json.loads(path.read_text())
    index = next(i for i, value in enumerate(row["expected_labels"]) if value > 23)
    row["expected_labels"][index] = -100
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n")
    index_path = output / meta["index_file"]
    offsets = np.load(index_path)
    offsets[0, 1] = path.stat().st_size
    np.save(index_path, offsets)
    meta.update(sha256=sha256(path), index_sha256=sha256(index_path))
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="ids/labels"):
        audit_native_encoding(root, output, output / "encoding-audit.json")
    report = json.loads((output / "encoding-audit.json").read_text())
    assert report["status"] == "failed" and not report["formal_admission"]
