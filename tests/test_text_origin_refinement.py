"""Later text partitions exclude entire old OCR train groups without moving holdouts."""

import contextlib
import json
import random
from dataclasses import asdict
from types import SimpleNamespace

import pytest
from PIL import Image

from minifrontier.data import sha256
from minifrontier.data.corpus import CorpusBuilder, train_tokenizer
from minifrontier.data.encoding_audit import audit_image_encoding
from minifrontier.data.encoding_filters import filter_media_encoding
from minifrontier.data.minifrontier1 import SPECIAL_TOKENS
from minifrontier.data.minifrontier1_encoding import CompactDataset, encode_canonical_images
from minifrontier.data.partitions import create_text_origin_exclusion_view, open_corpus
from minifrontier.models.minifrontier1 import MiniFrontier1Config


def _finish(builder, locks):
    root = builder.root
    builder.finalize(split_locks=locks)
    rows = [
        (i, s, g, json.loads(p))
        for i, s, g, p in builder.db.execute("SELECT id,split,group_root,payload FROM samples")
    ]
    builder.db.close()
    (root / "source-audit.json").write_text(
        json.dumps(
            dict(
                status="candidate_slice_complete_pending_admission",
                formal_admission=False,
            )
        )
    )
    (root / "review-samples.jsonl").write_text("")
    return rows


@pytest.mark.parametrize("bad_holdout", [False, True])
def test_origin_refinement_keeps_absence_distinct_from_holdout_and_preserves_tokens(
    tmp_path, monkeypatch, bad_holdout
):
    monkeypatch.setenv("MINIFRONTIER_MIN_FREE_GIB", "1")
    monkeypatch.setattr(
        "minifrontier.storage.shutil.disk_usage", lambda _: SimpleNamespace(free=900 * 1024**3)
    )
    text = tmp_path / "text"
    builder = CorpusBuilder(text, val_buckets=1, test_buckets=1)
    for i, content in enumerate(
        (
            "Glaciers preserve bubbles of ancient air under layers of solid ice.",
            "The chamber orchestra rehearses a symphony in a large concert hall.",
            "A spacecraft measures magnetic fields surrounding distant planets.",
        )
    ):
        assert builder.add(
            dict(
                source="text",
                revision="fixed",
                item_id=str(i),
                group_id=str(i),
                license="CC0-1.0",
                lang="en",
                task="natural",
                stage="pretrain",
                text=content,
                reference_tokens=20,
            )
        )
    text_rows = _finish(builder, {"source-group:text:1": "val", "source-group:text:2": "test"})
    by_split = {split: dict(sample_id=i, split=split, group_root=g) for i, split, g, _ in text_rows}
    assert set(by_split) == {"train", "val", "test"}
    media = tmp_path / "media"
    builder = CorpusBuilder(media, val_buckets=1, test_buckets=1)
    origins = [
        by_split["val"],
        by_split["train"],
        dict(sample_id="missing-text", split="train", group_root="missing-group"),
        by_split["train"],
        by_split["val"],
        by_split["test"],
    ]
    for i, origin in enumerate(origins):
        origin = dict(origin, split="train" if i < 4 else "val" if i == 4 else "test")
        if bad_holdout and i == 4:
            origin["sample_id"] = "missing-held-out-text"
        path = media / f"{i}.png"
        Image.frombytes("RGB", (32, 24), random.Random(i).randbytes(2304)).save(path)
        assert builder.add(
            dict(
                source="ocr",
                revision="fixed",
                item_id=str(i),
                group_id=str(0 if i < 2 else i),
                license="CC0-1.0",
                lang="en",
                task="ocr_document",
                stage="pretrain",
                text=f"Unique illustrated document {i} with its full transcription.",
                reference_tokens=20,
                answer_reference_tokens=10,
                visual_question=f"Read document {i}.",
                visual_answer=f"Printed answer {i}.",
                text_origin=origin,
                media=[
                    dict(
                        kind="image",
                        path=path.name,
                        width=32,
                        height=24,
                        sha256=sha256(path),
                        rgb_sha256=sha256(path),
                    )
                ],
            )
        )
    rows = _finish(builder, {"source-group:ocr:4": "val", "source-group:ocr:5": "test"})
    old_hash = sha256(media / "corpus.sqlite")
    view, report = tmp_path / "view", tmp_path / "origin-report.json"
    if bad_holdout:
        with pytest.raises(ValueError, match="old media holdout"):
            create_text_origin_exclusion_view(media, text, report, view)
        assert not view.exists()
        assert json.loads(report.read_text())["holdout_conflicts"]
        return
    tokenizer = tmp_path / "tokenizer.json"
    train_tokenizer(media, tokenizer, 350, special_tokens=SPECIAL_TOKENS)
    config = MiniFrontier1Config.tiny(350)
    parent = tmp_path / "encoded"
    encode_canonical_images(media, tokenizer, parent, config, max_features=4)
    audit_image_encoding(media, parent, parent / "encoding-audit.json", asdict(config))
    (parent / "source-audit.json").write_text(
        json.dumps(
            dict(
                status="mechanical_checks_passed_pending_quality_admission",
                producer_finished=True,
                manifest_sha256=sha256(parent / "manifest.json"),
                integrity_report="encoding-audit.json",
                integrity_report_sha256=sha256(parent / "encoding-audit.json"),
            )
        )
    )
    result = create_text_origin_exclusion_view(media, text, report, view)
    assert result["newly_excluded_records"] == 3  # Also exclude the clean sibling in group 0.
    assert result["splits"] == {"train": 1, "val": 1, "test": 1}
    assert sha256(media / "corpus.sqlite") == old_hash
    with contextlib.closing(open_corpus(view)) as db:
        kept = {i for (i,) in db.execute("SELECT id FROM samples")}
    assert {i for i, s, _, _ in rows if s != "train"} <= kept
    output = tmp_path / "filtered"
    with monkeypatch.context() as patch:
        patch.setattr(
            "minifrontier.data.minifrontier1_encoding.prepare_media",
            lambda *a, **k: pytest.fail("filter must not reprocess old pixels"),
        )
        filtered = filter_media_encoding(view, parent, output, config=asdict(config))
    assert filtered["splits"]["train"]["counts"]["records"] == 1
    for split in ("train", "val", "test"):
        old, new = CompactDataset(parent, split, config), CompactDataset(output, split, config)
        selected = [old[i] for i in range(len(old)) if old[i]["sample_id"] in kept]
        assert len(new) == len(selected)
        for i, before in enumerate(selected):
            assert new[i]["input_ids"].equal(before["input_ids"])
            assert new[i]["labels"].equal(before["labels"])
    proof = json.loads((output / "encoding-audit.json").read_text())
    assert proof["sealed_holdout_payloads_unchanged"]
