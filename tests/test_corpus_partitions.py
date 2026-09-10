"""Whole-group reservations must reach tokenizer and encoded training consumers."""

import contextlib
import hashlib
import json
import shutil
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer, models

from minifrontier.data import corpus as corpus_module
from minifrontier.data import sha256
from minifrontier.data.corpus import CorpusBuilder, encode_corpus, train_tokenizer
from minifrontier.data.minifrontier1 import SPECIAL_TOKENS as MF1_SPECIAL_TOKENS
from minifrontier.data.native import encode_native
from minifrontier.data.partitions import create_partition_view, open_corpus


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(free=900 * 1024**3))
    root = tmp_path / "corpus"
    builder = CorpusBuilder(root, val_buckets=1, test_buckets=1)
    rows = [
        ("train", "The museum preserves rare books and maps for future generations."),
        ("reserved", "A spacecraft measures the magnetic field around a distant planet."),
        ("reserved", "Insects communicate through chemical signals produced by their glands."),
        ("old-val", "Glaciers contain ancient bubbles of air trapped within solid ice."),
        ("old-test", "An orchestra rehearses a symphony in a large concert hall."),
    ]
    for i, (group, text) in enumerate(rows):
        assert builder.add(
            dict(
                source="fixture",
                revision="fixed",
                item_id=str(i),
                group_id=group,
                license="CC0-1.0",
                lang="en",
                task="natural",
                stage="pretrain",
                text=text,
                reference_tokens=len(text),
            )
        )
    builder.finalize(
        split_locks={"source-group:fixture:old-val": "val", "source-group:fixture:old-test": "test"}
    )
    selected, review_id = None, None
    for identity, group, payload in builder.db.execute("SELECT id,group_root,payload FROM samples"):
        row = json.loads(payload)
        if row["group_id"] == "reserved":
            selected = group
        elif row["group_id"] == "train":
            review_id = identity
    builder.db.close()
    (root / "source-audit.json").write_text(
        json.dumps(
            dict(status="candidate_slice_complete_pending_admission", formal_admission=False)
        )
    )
    (root / "integrity-and-split-audit.json").write_text(json.dumps(dict(cross_split_groups=0)))
    (root / "review-samples.jsonl").write_text(
        json.dumps(dict(record=dict(sample_id=review_id))) + "\n"
    )
    proposal = dict(
        corpus_manifest_sha256=sha256(root / "corpus-manifest.json"),
        integrity_audit_sha256=sha256(root / "integrity-and-split-audit.json"),
        minimum_validation_group_fraction=0.3,
        groups={selected: "val"},
    )
    reservation = root / "reservation.json"
    reservation.write_text(json.dumps(proposal))
    return root, reservation


@pytest.mark.parametrize("native", [False, True])
def test_reservation_reaches_tokenizer_and_encoded_splits_without_changing_base(
    corpus, tmp_path, native
):
    root, reservation = corpus
    before = sha256(root / "corpus.sqlite")
    view = tmp_path / "view"
    result = create_partition_view(root, reservation, view)
    assert result["splits"] == {"test": 1, "train": 1, "val": 3}
    assert not (view / "corpus.sqlite").exists()
    with contextlib.closing(open_corpus(view)) as db:
        expected = b"".join(
            t.encode()
            for (t,) in db.execute("SELECT text FROM samples WHERE split='train' ORDER BY id")
        )
        assert db.execute(
            "SELECT COUNT(*) FROM (SELECT group_root FROM samples GROUP BY group_root HAVING COUNT(DISTINCT split)>1)"
        ).fetchone() == (0,)
        with pytest.raises(sqlite3.OperationalError):
            db.execute("DELETE FROM main.samples")
    tokenizer_path = tmp_path / "tokenizer.json"
    # MF1 must retain its distinct control-token IDs even with shared text training.
    report = train_tokenizer(view, tokenizer_path, 350, special_tokens=MF1_SPECIAL_TOKENS)
    assert report["training_byte_sha256"] == hashlib.sha256(expected).hexdigest()
    tok = Tokenizer.from_file(str(tokenizer_path))
    assert [tok.token_to_id(s) for s in MF1_SPECIAL_TOKENS] == list(range(len(MF1_SPECIAL_TOKENS)))
    assert tok.decode(tok.encode("中文 code = 123\n").ids) == "中文 code = 123\n"
    output = tmp_path / "encoded"
    if native:
        encoded = encode_native(view, tokenizer_path, output, "minikimik3")
    else:
        encoded = encode_corpus(view, tokenizer_path, output)
    assert encoded["stages"]["pretrain"]["train"]["examples"] == 1
    assert encoded["stages"]["pretrain"]["val"]["examples"] == 3
    val_rows = [json.loads(line) for line in (output / "pretrain.val.jsonl").open()]
    assert sum(row["group_id"] == "reserved" for row in val_rows) == 2
    assert sha256(root / "corpus.sqlite") == before
    with contextlib.closing(open_corpus(root)) as db:
        assert db.execute("SELECT COUNT(*) FROM samples WHERE split='train'").fetchone() == (3,)


@pytest.mark.parametrize("fault", ["heldout", "missing", "review", "audit"])
def test_invalid_reservation_is_rejected_before_output(corpus, tmp_path, fault):
    root, reservation = corpus
    proposal = json.loads(reservation.read_text())
    with contextlib.closing(open_corpus(root)) as db:
        if fault == "heldout":
            proposal["groups"] = {
                db.execute("SELECT group_root FROM samples WHERE split='test'").fetchone()[0]: "val"
            }
        elif fault == "missing":
            proposal["groups"] = {"missing": "val"}
        elif fault == "review":
            review_id = json.loads((root / "review-samples.jsonl").read_text())["record"][
                "sample_id"
            ]
            proposal["groups"][
                db.execute("SELECT group_root FROM samples WHERE id=?", (review_id,)).fetchone()[0]
            ] = "val"
        else:
            proposal["integrity_audit_sha256"] = "0" * 64
    reservation.write_text(json.dumps(proposal))
    output = tmp_path / "view"
    with pytest.raises(ValueError):
        create_partition_view(root, reservation, output)
    assert not output.exists()


@pytest.mark.parametrize("target", ["split-overrides.json", "corpus.sqlite"])
def test_reader_rejects_changed_partition_or_original_database(corpus, tmp_path, target):
    root, reservation = corpus
    view = tmp_path / "view"
    create_partition_view(root, reservation, view)
    path = (root if target == "corpus.sqlite" else view) / target
    with path.open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match=r"changed|differs"):
        open_corpus(view)


def test_tokenizer_consumer_can_move_between_worker_threads(corpus, tmp_path, monkeypatch):
    root, reservation = corpus
    view = tmp_path / "view"
    create_partition_view(root, reservation, view)
    real = Tokenizer(models.BPE())

    class MigratingConsumer:
        def __init__(self, _model):
            pass

        def train_from_iterator(self, data, trainer):
            iterator = iter(data)
            end = object()
            # Two persistent workers guarantee different OS thread identities.
            # The first pulls the row; the second advances/closes the iterator.
            with ThreadPoolExecutor(1) as first, ThreadPoolExecutor(1) as second:
                rows = [first.submit(next, iterator, end).result()]
                assert second.submit(next, iterator, end).result() is end
            real.pre_tokenizer, real.decoder = self.pre_tokenizer, self.decoder
            real.train_from_iterator(rows, trainer)

        def get_vocab_size(self):
            return real.get_vocab_size()

        def save(self, path):
            real.save(path)

    monkeypatch.setattr(corpus_module, "Tokenizer", MigratingConsumer)
    report = train_tokenizer(view, tmp_path / "tokenizer.json", 350)
    assert report["training_bytes"] > 0


def test_encoding_rejects_insufficient_bound_before_copying_tokenizer(corpus, tmp_path):
    root, _ = corpus
    tokenizer = tmp_path / "tokenizer.json"
    train_tokenizer(root, tokenizer, 350)
    output = tmp_path / "encoded"
    with pytest.raises(ValueError, match="disk budget"):
        encode_corpus(root, tokenizer, output, max_gib=1e-6)
    assert not output.exists()
