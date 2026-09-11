"""Lossless held-out matching must catch the construction-time Simhash gap."""

import contextlib
import itertools
import json
import random
import shutil
import sqlite3
import string
from types import SimpleNamespace

import pytest

from minifrontier.data import sha256
from minifrontier.data.corpus import simhash, text_shingles
from minifrontier.data.partitions import HOLDOUT_FORMAT, create_text_exclusion_view, open_corpus
from minifrontier.data.text_inventory import ExactTextIndex, audit_text_holdouts


def test_prefix_index_matches_exhaustive_sets_including_novel_query_tokens():
    universe = list("abcdefgh")
    references = [set(c) for n in range(1, 9) for c in itertools.combinations(universe, n)]
    # Include exact 17/20 boundary and unequal lengths; ordering depends on frequency.
    references.extend([set(range(20)), set(range(17))])
    references = [{str(t) for t in values} for values in references]
    random.Random(42).shuffle(references)
    queries = references + [s | {"novel"} for s in references]
    index = ExactTextIndex(references)
    for query in queries:
        expected = {
            key
            for key, values in enumerate(references)
            if 100 * len(query & values) >= 85 * len(query | values)
        }
        actual = {key for key, _, _ in index.neighbors(query)}
        assert actual == expected


def _missed_pair():
    rng = random.Random(20260911)
    base = "".join(rng.choices(string.ascii_lowercase + "中文训练测试", k=2000))
    other = base[:-21] + "".join(random.Random(21).choices(string.ascii_lowercase, k=21))
    return base, other


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(free=900 * 1024**3))
    root = tmp_path / "corpus"
    root.mkdir()
    base, other = _missed_pair()
    rows = [
        ("train-near", "train-group", "train", base),
        (
            "train-group-member",
            "train-group",
            "train",
            "An unrelated sentence in the same source group.",
        ),
        ("test-near", "test-group", "test", other),
        ("val-near", "val-group", "val", " \n".join(other)),
        ("train-code", "code-train", "train", "def calculate(value):\n    return value * 12345\n"),
        ("test-code", "code-test", "test", "def calculate(value):\n    return value * 12345\n"),
    ]
    with sqlite3.connect(root / "corpus.sqlite") as db:
        db.execute(
            "CREATE TABLE samples(id TEXT PRIMARY KEY,stage TEXT,source TEXT,task TEXT,"
            "first_question TEXT,text TEXT,payload TEXT,simhash TEXT,group_root TEXT,split TEXT)"
        )
        for name, group, split, text in rows:
            payload = dict(
                sample_id=name,
                text=text,
                source="fixture",
                revision="fixed",
                group_id=group,
                task="natural",
                stage="pretrain",
                reference_tokens=len(text),
                **(dict(text_format="source_code", syntax_sha256=name) if "code" in name else {}),
            )
            db.execute(
                "INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    name,
                    "pretrain",
                    "fixture",
                    "natural",
                    "",
                    text,
                    json.dumps(payload),
                    "0",
                    group,
                    split,
                ),
            )
    (root / "corpus-manifest.json").write_text(
        json.dumps(
            dict(
                database_sha256=sha256(root / "corpus.sqlite"),
                splits=dict(train=3, val=1, test=2),
                split_rule="fixed synthetic groups",
            )
        )
    )
    (root / "source-audit.json").write_text(
        json.dumps(
            dict(
                status="candidate_slice_complete_pending_admission",
                formal_admission=False,
            )
        )
    )
    (root / "review-samples.jsonl").write_text(
        json.dumps(dict(record=dict(sample_id="train-near"))) + "\n"
    )
    return root


def test_full_audit_catches_four_bit_simhash_gap_and_quarantines_entire_train_groups(
    corpus, tmp_path
):
    base, other = _missed_pair()
    left, right = simhash(base), simhash(other)
    assert (left ^ right).bit_count() == 4
    assert any((left >> (16 * i)) & 65535 == (right >> (16 * i)) & 65535 for i in range(4))
    a, b = text_shingles(base), text_shingles(other)
    assert len(a & b) / len(a | b) > 0.97
    before = sha256(corpus / "corpus.sqlite")
    output = tmp_path / "audit.json"
    states = []
    result = audit_text_holdouts(corpus, output, progress=lambda row: states.append(row["state"]))
    assert result["full_shared_text_cross_split_audit_complete"]
    assert result["formal_admission"] is False
    assert result["effective_records"] == dict(train=3, val=1, test=2)
    pairs = {frozenset(match["samples"]) for match in result["matches"]}
    assert pairs == {
        frozenset(("train-near", "test-near")),
        frozenset(("train-near", "val-near")),
        frozenset(("test-near", "val-near")),
        frozenset(("train-code", "test-code")),
    }
    assert states[-1] == "all_pairs_checked"
    assert len(result["split_conflicts"]) == 2
    view = tmp_path / "view"
    partition = create_text_exclusion_view(corpus, output, "shared-text", view)
    assert partition["newly_excluded_records"] == 3
    assert partition["splits"] == dict(val=1, test=2)
    with contextlib.closing(open_corpus(view)) as db:
        assert dict(db.execute("SELECT id,split FROM samples")) == {
            "test-near": "test",
            "val-near": "val",
            "test-code": "test",
        }
    assert sha256(corpus / "corpus.sqlite") == before
    # Removing training groups does not close any remaining val/test conflict.
    remaining = audit_text_holdouts(view, tmp_path / "remaining.json")
    assert remaining["status"] == "split_conflicts_require_partition_update"
    assert len(remaining["matches"]) == 1
    assert set(remaining["matches"][0]["splits"]) == {"val", "test"}
    # Text itself is never exported to the audit artifact.
    assert base not in output.read_text() and other not in output.read_text()


@pytest.mark.parametrize(
    "bound",
    ["max_reference_shingles", "max_candidates", "max_comparisons", "max_matches", "max_bytes"],
)
def test_audit_exceeded_bounds_never_publish_completed_evidence(corpus, tmp_path, bound):
    output = tmp_path / "incomplete.json"
    with pytest.raises(ValueError, match="budget exceeded"):
        audit_text_holdouts(corpus, output, **{bound: 1})
    assert not output.exists()


def test_changed_source_and_existing_output_are_rejected(corpus, tmp_path):
    output = tmp_path / "audit.json"
    output.write_text("existing evidence")
    with pytest.raises(FileExistsError):
        audit_text_holdouts(corpus, output)
    assert output.read_text() == "existing evidence"
    with sqlite3.connect(corpus / "corpus.sqlite") as db:
        db.execute("UPDATE samples SET text='changed source' WHERE id='train-near'")
    with pytest.raises(ValueError, match="differs from manifest"):
        audit_text_holdouts(corpus, tmp_path / "new.json")


def test_complete_text_conflicts_preserve_test_and_promote_whole_validation_groups(
    corpus, tmp_path
):
    output = tmp_path / "audit.json"
    audit_text_holdouts(corpus, output)
    view = tmp_path / "resolved"
    result = create_text_exclusion_view(
        corpus, output, "shared-text", view, resolve_validation_conflicts=True
    )
    assert result["format"] == HOLDOUT_FORMAT
    assert result["newly_excluded_records"] == 3
    assert result["promoted_validation_records"] == 1
    assert result["splits"] == dict(test=3)
    with contextlib.closing(open_corpus(view)) as db:
        assert dict(db.execute("SELECT id,split FROM samples")) == {
            "test-near": "test",
            "val-near": "test",
            "test-code": "test",
        }
    closed = audit_text_holdouts(view, tmp_path / "resolved-audit.json")
    assert closed["status"] == "cross_split_check_passed" and not closed["matches"]
    assert not closed["formal_admission"]


@pytest.mark.parametrize(
    "fault", ["incomplete", "no_test_member", "wrong_count", "wrong_split", "wrong_anchor"]
)
def test_validation_promotion_requires_complete_bound_membership(corpus, tmp_path, fault):
    output = tmp_path / "audit.json"
    result = audit_text_holdouts(corpus, output)
    if fault == "incomplete":
        result["full_shared_text_cross_split_audit_complete"] = False
    else:
        component = next(
            c for c in result["split_conflicts"] if any(m["split"] == "val" for m in c["members"])
        )
        if fault == "no_test_member":
            component["members"] = [m for m in component["members"] if m["split"] != "test"]
        elif fault == "wrong_anchor":
            next(m for m in component["members"] if m["split"] == "test")["group"] = (
                "missing-test-group"
            )
        else:
            member = next(m for m in component["members"] if m["split"] == "val")
            if fault == "wrong_count":
                member["records"] += 1
            else:
                member["group"] = "test-group"
    output.write_text(json.dumps(result))
    view = tmp_path / "invalid"
    with pytest.raises(ValueError):
        create_text_exclusion_view(
            corpus, output, "shared-text", view, resolve_validation_conflicts=True
        )
    assert not view.exists()
