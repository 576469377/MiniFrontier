"""Incremental text encodings retain exact tokens and the merged corpus partition."""

import json
import random

import pytest
import torch
from test_mf1_canonical_text import encoded as encoded

from minifrontier.data import sha256
from minifrontier.data.corpus import CorpusBuilder
from minifrontier.data.encoding_audit import audit_text_encoding
from minifrontier.data.minifrontier1_encoding import (
    CompactDataset,
    encode_canonical_text,
    extend_canonical_text,
)
from minifrontier.data.partitions import open_corpus


def bound_parent(encoded, tmp_path):
    old, config, _ = encoded
    corpus = old.parent / "canonical"
    parent = tmp_path / "parent"
    encode_canonical_text(corpus, old / "tokenizer.json", parent, config, shard_tokens=100000)
    report = audit_text_encoding(corpus, parent, parent / "encoding-audit.json")
    (parent / "source-audit.json").write_text(
        json.dumps(
            dict(
                producer_finished=True,
                status=report["status"],
                manifest_sha256=sha256(parent / "manifest.json"),
                integrity_report="encoding-audit.json",
                integrity_report_sha256=sha256(parent / "encoding-audit.json"),
            )
        )
    )
    return corpus, parent, config


def merged(corpus, output, *, refine=False, downgrade=False):
    db = open_corpus(corpus)
    rows = [json.loads(p) for (p,) in db.execute("SELECT payload FROM samples ORDER BY id")]
    db.close()
    builder = CorpusBuilder(output)
    for row in rows:
        if refine and row["item_id"] == "0":
            continue
        assert builder.add(row)
    assert builder.add(
        dict(
            source="extension-fixture",
            revision="pinned",
            item_id="new",
            group_id="new",
            license="CC0-1.0",
            task="code",
            lang="en",
            stage="pretrain",
            text=" ".join(random.Random(929).choices("abcdefg", k=300)),
        )
    )
    locks = {"source-group:fixture:3": "test"}
    if not downgrade:
        locks["source-group:fixture:2"] = "val"
    if refine:
        locks["source-group:fixture:1"] = "val"
    builder.finalize(split_locks=locks)
    builder.db.close()
    (output / "source-audit.json").write_text(
        json.dumps(
            dict(status="candidate_slice_complete_pending_admission", formal_admission=False)
        )
    )


@pytest.mark.parametrize("refine", [False, True])
def test_extension_matches_full_encoding_without_tokenizing_old_documents(
    encoded, tmp_path, monkeypatch, refine
):
    from minifrontier.data import minifrontier1_encoding as module

    corpus, parent, config = bound_parent(encoded, tmp_path)
    target = tmp_path / "merged"
    merged(corpus, target, refine=refine)
    calls = []
    original = module.safe_text

    def observed(tokenizer, text):
        calls.append(text)
        return original(tokenizer, text)

    monkeypatch.setattr(module, "safe_text", observed)
    output = tmp_path / "incremental"
    manifest = extend_canonical_text(target, parent, output, config)
    assert len(calls) == 1
    assert manifest["incremental_reuse"]["new_documents_tokenized"] == 1
    assert manifest["incremental_reuse"]["parts_with_linked_tokens"] >= 2
    if refine:
        assert manifest["incremental_reuse"]["old_documents_excluded"] == 1
        assert manifest["incremental_reuse"]["parts_requiring_compaction"] == 1
    audit = audit_text_encoding(target, output, tmp_path / "audit.json")
    assert audit["status"] == "mechanical_checks_passed_pending_quality_admission"
    expected = tmp_path / "full"
    encode_canonical_text(target, parent / "tokenizer.json", expected, config)
    for split in ("train", "val", "test"):
        actual = CompactDataset(output, split, config)
        full = CompactDataset(expected, split, config)
        by_id = {full[i]["sample_id"]: full[i] for i in range(len(full))}
        assert len(actual) == len(by_id)
        for i in range(len(actual)):
            item = actual[i]
            match = by_id.pop(item["sample_id"])
            assert item["split_group"] == match["split_group"] and item["domain"] == match["domain"]
            torch.testing.assert_close(item["input_ids"], match["input_ids"], atol=0, rtol=0)
            torch.testing.assert_close(item["labels"], match["labels"], atol=0, rtol=0)
        for part in manifest["splits"][split]["parts"]:
            if "reused" in part["files"]["tokens.bin"]["name"]:
                assert (output / part["files"]["tokens.bin"]["name"]).stat().st_nlink == 2


def test_extension_rejects_reintroduced_old_holdout(encoded, tmp_path):
    corpus, parent, config = bound_parent(encoded, tmp_path)
    target = tmp_path / "merged"
    merged(corpus, target, downgrade=True)
    with pytest.raises(ValueError, match="held-out"):
        extend_canonical_text(target, parent, tmp_path / "bad", config)
