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
from minifrontier.data.partitions import (
    HOLDOUT_FORMAT,
    create_partition_view,
    create_quality_exclusion_view,
    create_text_exclusion_view,
    open_corpus,
)


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


@pytest.mark.parametrize("fault", [None, "release", "binding", "review"])
def test_additional_reservation_preserves_quarantine_and_test_precedence(corpus, tmp_path, fault):
    root, reservation = corpus
    # Build two additional singleton groups before creating the initial view.
    with sqlite3.connect(root / "corpus.sqlite") as db:
        row = list(db.execute("SELECT * FROM samples LIMIT 1").fetchone())
        for name in ("additional", "quarantined"):
            record = json.loads(row[6])
            record.update(sample_id=name, group_id=name, item_id=name)
            new = [*row]
            new[0], new[6], new[8], new[9] = name, json.dumps(record), name, "train"
            db.execute("INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?,?)", new)
    manifest = json.loads((root / "corpus-manifest.json").read_text())
    manifest["database_sha256"] = sha256(root / "corpus.sqlite")
    manifest["splits"]["train"] += 2
    (root / "corpus-manifest.json").write_text(json.dumps(manifest))
    proposal = json.loads(reservation.read_text())
    proposal["corpus_manifest_sha256"] = sha256(root / "corpus-manifest.json")
    reservation.write_text(json.dumps(proposal))
    prior = tmp_path / "prior"
    create_partition_view(root, reservation, prior)
    # This synthetic prior has already quarantined one train group and moved an
    # original validation group to test. Those controls must survive the top-up.
    with contextlib.closing(open_corpus(root)) as db:
        promoted = db.execute("SELECT group_root FROM samples WHERE split='val'").fetchone()[0]
    controls = json.loads((prior / "split-overrides.json").read_text())
    controls.update(excluded_groups=["quarantined"], test_groups=[promoted])
    (prior / "split-overrides.json").write_text(json.dumps(controls))
    previous = json.loads((prior / "corpus-manifest.json").read_text())
    previous.update(format=HOLDOUT_FORMAT, partition_sha256=sha256(prior / "split-overrides.json"))
    (prior / "corpus-manifest.json").write_text(json.dumps(previous))
    with contextlib.closing(open_corpus(prior)) as db:
        held = dict(db.execute("SELECT id,split FROM samples WHERE split!='train'"))
    proposal.update(
        prior_partition=str(prior),
        prior_partition_manifest_sha256=sha256(prior / "corpus-manifest.json"),
        prior_partition_audit_sha256=sha256(prior / "source-audit.json"),
        groups={**controls["groups"], "additional": "val"},
    )
    if fault == "release":
        proposal["groups"] = {"additional": "val"}
    elif fault == "binding":
        proposal["prior_partition_manifest_sha256"] = "0" * 64
    elif fault == "review":
        review = json.loads((prior / "review-samples.jsonl").read_text())["record"]["sample_id"]
        with contextlib.closing(open_corpus(root)) as db:
            group = db.execute("SELECT group_root FROM samples WHERE id=?", (review,)).fetchone()[0]
        proposal["groups"][group] = "val"
    reservation.write_text(json.dumps(proposal))
    output = tmp_path / "updated"
    before = sha256(root / "corpus.sqlite")
    if fault:
        with pytest.raises(ValueError):
            create_partition_view(root, reservation, output)
        assert not output.exists()
        return
    result = create_partition_view(root, reservation, output)
    assert result["format"] == HOLDOUT_FORMAT
    assert result["splits"] == dict(train=1, val=3, test=2)
    with contextlib.closing(open_corpus(output)) as db:
        actual = dict(db.execute("SELECT id,split FROM samples"))
    assert "quarantined" not in actual and actual["additional"] == "val"
    assert all(actual[identity] == split for identity, split in held.items())
    assert sha256(root / "corpus.sqlite") == before


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


@pytest.mark.parametrize("nested", [False, True])
def test_text_exclusion_preserves_originals_and_all_holdouts(corpus, tmp_path, nested):
    original, reservation = corpus
    source = original
    if nested:
        source = tmp_path / "prior-view"
        create_partition_view(original, reservation, source)
    manifest = json.loads((source / "corpus-manifest.json").read_text())
    before = sha256(original / "corpus.sqlite")
    with contextlib.closing(open_corpus(source)) as db:
        holds = list(db.execute("SELECT id,split FROM samples WHERE split!='train' ORDER BY id"))
        group = db.execute("SELECT group_root FROM samples WHERE split='train' LIMIT 1").fetchone()[
            0
        ]
        count = db.execute("SELECT COUNT(*) FROM samples WHERE group_root=?", (group,)).fetchone()[
            0
        ]
    evidence = tmp_path / "text-grouping.json"
    evidence.write_text(
        json.dumps(
            dict(
                kind="cross_corpus_text_group_audit",
                inputs=dict(
                    text=dict(
                        corpus_manifest_sha256=sha256(source / "corpus-manifest.json"),
                        database_sha256=manifest["database_sha256"],
                    )
                ),
                split_conflicts=[
                    dict(
                        required_split="test",
                        members=[
                            dict(
                                inventory="text",
                                group=group,
                                split="train",
                                records=count,
                            )
                        ],
                    )
                ],
            )
        )
    )
    out = tmp_path / "text-excluded"
    result = create_text_exclusion_view(source, evidence, "text", out)
    assert result["newly_excluded_records"] == count
    assert not result["formal_admission"] and sha256(original / "corpus.sqlite") == before
    assert not (out / "corpus.sqlite").exists()
    with contextlib.closing(open_corpus(out)) as db:
        assert db.execute(
            "SELECT COUNT(*) FROM samples WHERE group_root=?", (group,)
        ).fetchone() == (0,)
        assert (
            list(db.execute("SELECT id,split FROM samples WHERE split!='train' ORDER BY id"))
            == holds
        )
    report = json.loads((out / "source-audit.json").read_text())
    assert report["excluded_training_groups"] == {group: count}


def test_reserved_validation_group_moves_whole_to_test_and_reaches_text_encoding(corpus, tmp_path):
    original, reservation = corpus
    prior = tmp_path / "prior"
    create_partition_view(original, reservation, prior)
    group = next(iter(json.loads(reservation.read_text())["groups"]))
    with contextlib.closing(open_corpus(prior)) as db:
        before = dict(db.execute("SELECT id,split FROM samples"))
        promoted = [r[0] for r in db.execute("SELECT id FROM samples WHERE group_root=?", (group,))]
        test_group = db.execute("SELECT group_root FROM samples WHERE split='test'").fetchone()[0]
        training_bytes = b"".join(
            r[0].encode()
            for r in db.execute("SELECT text FROM samples WHERE split='train' ORDER BY id")
        )
    report = tmp_path / "grouping.json"
    report.write_text(
        json.dumps(
            dict(
                kind="cross_corpus_text_group_audit",
                full_shared_text_cross_split_audit_complete=True,
                heldout_self_join_complete=True,
                inputs={
                    "text": dict(
                        corpus_manifest_sha256=sha256(prior / "corpus-manifest.json"),
                        database_sha256=sha256(original / "corpus.sqlite"),
                    )
                },
                split_conflicts=[
                    dict(
                        required_split="test",
                        members=[
                            dict(inventory="text", group=group, split="val", records=2),
                            dict(inventory="text", group=test_group, split="test", records=1),
                        ],
                    )
                ],
            )
        )
    )
    view = tmp_path / "resolved"
    result = create_text_exclusion_view(
        prior, report, "text", view, resolve_validation_conflicts=True
    )
    assert result["format"] == HOLDOUT_FORMAT
    assert result["newly_excluded_records"] == 0 and result["promoted_validation_records"] == 2
    assert result["splits"] == {"train": 1, "val": 1, "test": 3}
    with contextlib.closing(open_corpus(view)) as db:
        assert dict(db.execute("SELECT id,split FROM samples")) == {
            identity: ("test" if identity in promoted else split)
            for identity, split in before.items()
        }
    tokenizer = tmp_path / "tokenizer.json"
    tokens = train_tokenizer(view, tokenizer, 350)
    assert tokens["training_byte_sha256"] == hashlib.sha256(training_bytes).hexdigest()
    encoded = tmp_path / "encoded"
    manifest = encode_corpus(view, tokenizer, encoded)
    assert {
        split: record["examples"] for split, record in manifest["stages"]["pretrain"].items()
    } == result["splits"]
    assert {
        json.loads(line)["sample_id"]
        for line in (encoded / "pretrain.test.jsonl").read_text().splitlines()
    } == {identity for identity, split in before.items() if split == "test" or identity in promoted}
    # A consumer must not silently ignore test promotions as an older format.
    result["format"] = "corpus-partition-view-v1"
    (view / "corpus-manifest.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="versioned partition"):
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


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("fault", [None, "heldout", "count", "duplicate", "reason", "binding"])
def test_quality_exclusion_removes_whole_group_and_preserves_holdouts(
    corpus, tmp_path, nested, fault
):
    base, reservation = corpus
    source = tmp_path / "reserved" if nested else base
    if nested:
        create_partition_view(base, reservation, source)
    before_database = sha256(base / "corpus.sqlite")
    with contextlib.closing(open_corpus(source)) as db:
        before = list(db.execute("SELECT id,split,group_root,payload FROM samples ORDER BY id"))
        split = "val" if fault == "heldout" else "train"
        group, count = db.execute(
            "SELECT group_root,COUNT(*) n FROM samples WHERE split=? "
            "GROUP BY group_root ORDER BY n DESC LIMIT 1",
            (split,),
        ).fetchone()
    defect = dict(group=group, records=count, reason="Observed source template did not render")
    review = dict(
        kind="source_quality_exclusion_review",
        corpus_kind="text",
        status="targeted_defects_confirmed",
        review_method="model_assisted",
        inputs=dict(
            text=dict(
                corpus_manifest_sha256=sha256(source / "corpus-manifest.json"),
                source_audit_sha256=sha256(source / "source-audit.json"),
                database_sha256=before_database,
            )
        ),
        excluded_training_groups=[defect],
    )
    if fault == "count":
        defect["records"] += 1
    elif fault == "duplicate":
        review["excluded_training_groups"].append(defect)
    elif fault == "reason":
        defect["reason"] = " "
    elif fault == "binding":
        review["inputs"]["text"]["source_audit_sha256"] = "0" * 64
    evidence = tmp_path / "quality-review.json"
    evidence.write_text(json.dumps(review))
    output = tmp_path / "filtered"
    if fault:
        with pytest.raises(ValueError):
            create_quality_exclusion_view(source, evidence, "text", output)
        assert not output.exists()
        assert sha256(base / "corpus.sqlite") == before_database
        return
    result = create_quality_exclusion_view(source, evidence, "text", output)
    with contextlib.closing(open_corpus(output)) as db:
        after = list(db.execute("SELECT id,split,group_root,payload FROM samples ORDER BY id"))
    assert after == [row for row in before if row[2] != group]
    assert [r for r in after if r[1] != "train"] == [r for r in before if r[1] != "train"]
    assert result["newly_excluded_records"] == (1 if nested else 2)
    assert result["quality_review_sha256"] == sha256(evidence)
    assert "grouping_audit_sha256" not in result
    audit = json.loads((output / "source-audit.json").read_text())
    assert audit["operation"] == "exclude_source_quality_train_groups"
    assert audit["holdout_membership_sha256"] == audit["prior_holdout_membership_sha256"]
    assert not audit["formal_admission"] and not audit["main_budget_eligible"]
    assert sha256(base / "corpus.sqlite") == before_database
    assert not (output / "corpus.sqlite").exists()
