"""Reuse verified complete documents while rebuilding the new corpus partitions."""

import json
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from minifrontier.data import corpus as module
from minifrontier.data import sha256
from minifrontier.data.corpus import CorpusBuilder, encode_corpus, train_tokenizer
from minifrontier.data.encoding_audit import audit_text_encoding


@pytest.fixture
def encoding(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(free=900 * 1024**3))
    rows = [
        dict(
            item_id="a",
            group_id="a",
            text="A mountain observatory records faint light from distant galaxies.",
        ),
        dict(
            item_id="b",
            group_id="b",
            text="Library staff preserve historical maps inside climate controlled cabinets.",
        ),
        dict(
            item_id="c",
            group_id="c",
            text="This paragraph quotes <|image|> as literal text rather than a protocol marker.",
        ),
    ]
    for row in rows:
        row.update(
            source="fixture",
            revision="fixed",
            license="CC0",
            lang="en",
            task="zh_edu",
            stage="pretrain",
        )

    def build(name, records, locks=None):
        b = CorpusBuilder(tmp_path / name)
        for row in records:
            assert b.add(row)
        b.finalize(split_locks=locks)
        b.db.close()
        return b.root

    old = build("old", rows, {"source-group:fixture:c": "test"})
    tokenizer = tmp_path / "tokenizer.json"
    train_tokenizer(old, tokenizer, 300)
    encoded = tmp_path / "encoded"
    encode_corpus(old, tokenizer, encoded, max_length=8192)
    added = dict(
        rows[0],
        item_id="new",
        group_id="new",
        text="Wildlife researchers track migrating birds with lightweight radio transmitters.",
    )
    new = build(
        "new",
        [rows[0], rows[2], added],
        {"source-group:fixture:a": "val", "source-group:fixture:c": "test"},
    )
    return old, new, encoded, tokenizer, rows, build


def test_reuse_is_byte_identical_across_removal_and_split_moves(encoding, tmp_path, monkeypatch):
    old, new, source, tokenizer, _, _ = encoding
    before = {str(p): sha256(p) for r in (old, source) for p in r.iterdir() if p.is_file()}
    direct = tmp_path / "direct"
    expected = encode_corpus(new, tokenizer, direct, max_length=2048)
    calls = []
    original = module.Tokenizer

    class Spy:
        @staticmethod
        def from_file(path):
            class Wrapped:
                def __init__(self):
                    object.__setattr__(self, "tokenizer", original.from_file(path))

                def __getattr__(self, key):
                    return getattr(self.tokenizer, key)

                def __setattr__(self, key, value):
                    setattr(self.tokenizer, key, value)

                def encode(self, text, **kwargs):
                    calls.append(text)
                    return self.tokenizer.encode(text, **kwargs)

            return Wrapped()

    monkeypatch.setattr(module, "Tokenizer", Spy)
    output = tmp_path / "reused"
    actual = encode_corpus(
        new,
        tokenizer,
        output,
        max_length=2048,
        reuse_encoding=source,
        compact_metadata=True,
        max_gib=0.05,
    )
    assert len(calls) == 1
    assert sum(n["document_reuse"]["copied"] for n in actual["stages"]["pretrain"].values()) == 2
    for stage in ("pretrain", "sft"):
        for split in ("train", "val", "test"):
            node, reference = actual["stages"][stage][split], expected["stages"][stage][split]
            for key in ("file", "labels_file", "index_file"):
                assert (output / node[key]).read_bytes() == (direct / reference[key]).read_bytes()
            assert node["supervised_tokens"] == reference["supervised_tokens"]
            for line in (output / node["metadata_file"]).read_text().splitlines():
                row = json.loads(line)
                assert "text" not in row and "encoded_text_sha256" in row
    audit = audit_text_encoding(new, output, tmp_path / "audit.json")
    assert audit["errors"] == {}
    assert {str(p): sha256(p) for r in (old, source) for p in r.iterdir() if p.is_file()} == before
    calls.clear()
    encode_corpus(
        new, tokenizer, tmp_path / "second-reuse", reuse_encoding=output, compact_metadata=True
    )
    assert calls == []


def test_reuse_rejects_changed_provenance_for_the_same_document(encoding, tmp_path):
    _, _, source, tokenizer, rows, build = encoding
    changed = build("changed", [dict(rows[0], revision="different")])
    with pytest.raises(ValueError, match="text or provenance differs"):
        encode_corpus(changed, tokenizer, tmp_path / "invalid", reuse_encoding=source)
    assert not (tmp_path / "invalid/manifest.json").exists()


@pytest.mark.parametrize("mutation", ["tokenizer", "tokens", "protocol", "index"])
def test_reuse_rejects_mutated_identity_before_output(encoding, tmp_path, mutation):
    _, new, source, tokenizer, _, _ = encoding
    manifest = json.loads((source / "manifest.json").read_text())
    if mutation == "tokenizer":
        (source / "tokenizer.json").write_text("changed")
    elif mutation == "protocol":
        manifest["pretrain_text_encoding"] = "unverified-protocol"
    elif mutation == "tokens":
        with (source / manifest["stages"]["pretrain"]["test"]["file"]).open("ab") as h:
            h.write(b"changed")
    else:
        node = manifest["stages"]["pretrain"]["test"]
        path = source / node["index_file"]
        index = np.load(path)
        index[0, 0] += 1
        np.save(path, index)
        node["index_sha256"] = sha256(path)
    (source / "manifest.json").write_text(json.dumps(manifest))
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        encode_corpus(new, tokenizer, output, reuse_encoding=source)
    assert not output.exists()
