"""Preview entry points must preserve data separation and installed-package scope."""

import sqlite3

import pytest

from minifrontier import provenance
from minifrontier.quickstart import prepare
from minifrontier.training.strategy_gate import check


def test_offline_example_splits_and_encoding(tmp_path):
    root = tmp_path / "data"
    info = prepare(root)
    assert 256 < info["actual_vocab"] <= 512
    with sqlite3.connect(root / "corpus.sqlite") as db:
        rows = db.execute("SELECT split, payload FROM samples").fetchall()
    import json

    groups = {}
    for split, payload in rows:
        row = json.loads(payload)
        group = row["group_id"]
        assert groups.setdefault(group, split) == split
        assert sum(row["verifier"]["operands"]) == row["verifier"]["answer"]
    assert len(rows) == 224
    assert (root / "encoded/manifest.json").is_file()


def test_wheel_rejects_formal_gate_before_reading_missing_documents(monkeypatch):
    monkeypatch.setattr(provenance, "checkout_root", lambda: None)
    assert provenance.source_identity()["commit"] is None
    with pytest.raises(ValueError, match="requires a Git checkout"):
        check(
            "missing-plan",
            "missing-phase",
            "missing-evidence",
            data="missing-data",
            config=None,
            output=None,
        )
