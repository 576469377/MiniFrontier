"""Shared-text updates preserve media tensors and never promote trained media to val."""

import json
import shutil
import sqlite3

import pytest
import torch
from test_native_canonical_media import _audited_native
from test_native_canonical_media import corpus as corpus
from test_native_canonical_media import inputs as inputs

from minifrontier.data import StageDataset, sha256
from minifrontier.data.corpus import encode_corpus
from minifrontier.data.encoding_audit import audit_text_encoding
from minifrontier.data.native_components import rebind_native_components


def finalize_changed_database(root):
    p = root / "corpus-manifest.json"
    m = json.loads(p.read_text())
    m["database_sha256"] = sha256(root / "corpus.sqlite")
    with sqlite3.connect(root / "corpus.sqlite") as db:
        m["splits"] = dict(db.execute("SELECT split,count(*) FROM samples GROUP BY split"))
    p.write_text(json.dumps(m))


def new_text(inputs, tmp_path, *, promote=False, remove_val=False):
    _, tokenizer, _, shared = inputs
    canonical = tmp_path / "new-canonical-text"
    shutil.copytree(shared.parent / "canonical-text", canonical)
    with sqlite3.connect(canonical / "corpus.sqlite") as db:
        if promote:
            db.execute("UPDATE samples SET split='val' WHERE split='train'")
        if remove_val:
            db.execute("DELETE FROM samples WHERE split='val'")
    finalize_changed_database(canonical)
    result = tmp_path / "new-shared"
    encode_corpus(canonical, tokenizer, result, max_length=1024)
    report = audit_text_encoding(canonical, result, result / "encoding-audit.json")
    (result / "source-audit.json").write_text(
        json.dumps(
            dict(
                status=report["status"],
                producer_finished=True,
                manifest_sha256=sha256(result / "manifest.json"),
                integrity_report="encoding-audit.json",
                integrity_report_sha256=sha256(result / "encoding-audit.json"),
            )
        )
    )
    return result


def add_origin(inputs, *, split, partner=False):
    root, _, _, shared = inputs
    with (shared / f"pretrain.{split}.jsonl").open() as stream:
        identity = json.loads(next(stream))["sample_id"]
    with sqlite3.connect(root / "corpus.sqlite") as db:
        rows = list(
            db.execute(
                "SELECT id,payload,group_root FROM samples WHERE split=? ORDER BY id", (split,)
            )
        )
        first_id, payload, group = rows[0]
        row = json.loads(payload)
        row["text_origin"] = dict(sample_id=identity, split=split, group_root=group)
        db.execute("UPDATE samples SET payload=? WHERE id=?", (json.dumps(row), first_id))
        if partner:
            db.execute("UPDATE samples SET group_root=? WHERE id=?", (group, rows[1][0]))
    finalize_changed_database(root)


@pytest.mark.parametrize("family", ["minikimik3", "miniqwen4"])
def test_rebind_preserves_media_bytes_pixels_and_processor_policy(inputs, tmp_path, family):
    root, tokenizer, _, shared = inputs
    parent = _audited_native(root, tokenizer, shared, tmp_path / "parent", family)
    old_hash = sha256(parent / "manifest.json")
    text = new_text(inputs, tmp_path)
    output = tmp_path / "rebound"
    result = rebind_native_components([parent], text, output)
    child = output / "components/000"
    assert sha256(parent / "manifest.json") == old_hash
    assert (child / "pretrain.train.media.jsonl").stat().st_ino == (
        parent / "pretrain.train.media.jsonl"
    ).stat().st_ino
    assert result["manifest"]["max_features"] == 4
    assert result["processor_and_resolution_unchanged"] and not result["full_next_phase_ready"]
    before = StageDataset(parent, "pretrain", "train", 512)
    after = StageDataset(output / "dataset", "pretrain", "train", 512)
    assert len(before) == len(after)
    for i in range(len(before)):
        torch.testing.assert_close(before[i].input_ids, after[i].input_ids, atol=0, rtol=0)
        torch.testing.assert_close(before[i].labels, after[i].labels, atol=0, rtol=0)


def test_changed_train_origin_excludes_its_complete_media_group(inputs, tmp_path):
    root, tokenizer, _, shared = inputs
    add_origin(inputs, split="train", partner=True)
    parent = _audited_native(root, tokenizer, shared, tmp_path / "parent", "minikimik3")
    text = new_text(inputs, tmp_path, promote=True)
    output = tmp_path / "rebound"
    result = rebind_native_components([parent], text, output)
    assert result["components"][0]["removed_records"] == 2
    assert result["manifest"]["stages"]["pretrain"]["train"]["media"]["examples"] == 0
    for split in ("val", "test"):
        assert (parent / f"pretrain.{split}.media.jsonl").read_bytes() == (
            output / f"components/000/pretrain.{split}.media.jsonl"
        ).read_bytes()


def test_old_validation_origin_conflict_requires_explicit_handling(inputs, tmp_path):
    root, tokenizer, _, shared = inputs
    add_origin(inputs, split="val")
    parent = _audited_native(root, tokenizer, shared, tmp_path / "parent", "minikimik3")
    text = new_text(inputs, tmp_path, remove_val=True)
    output = tmp_path / "rebound"
    with pytest.raises(ValueError, match="validation/test text origins"):
        rebind_native_components([parent], text, output)
    assert not output.exists()
