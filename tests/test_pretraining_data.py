"""Production candidate text filtering and leakage controls, without network IO."""

import hashlib
import json

import pytest

from minifrontier.data.corpus import CorpusBuilder
from minifrontier.data.pretraining import clean_record


def test_pretraining_dialogue_keeps_all_turns_in_continuation_text():
    turns = [
        dict(role="user", content="Please explain how a prism separates white light."),
        dict(
            role="assistant",
            content="Light changes direction when it enters glass. "
            "Different wavelengths change direction by different amounts.",
        ),
    ]
    row, reason = clean_record("ultrachat", dict(messages=turns, prompt_id="origin"), "row:7")
    assert reason is None and row["stage"] == "pretrain" and "turns" not in row
    assert all(turn["content"] in row["text"] for turn in turns)
    assert row["group_id"] == "origin"
    assert row["raw_text_sha256"] == hashlib.sha256(row["text"].encode()).hexdigest()


def test_pretraining_sources_reject_missing_origin_and_language_mismatch():
    text = "Explain the physical principles of reflection and refraction in glass. " * 3
    assert clean_record("openwebmath", dict(text=text), "row")[1] == "missing_origin_or_text"
    assert (
        clean_record("zh_edu", dict(text=text, score=0.8, source="CCI3"), "row")[1]
        == "language_script_mismatch"
    )
    row, _ = clean_record(
        "openwebmath", dict(text=text, url="https://example.org/math#section"), "row"
    )
    assert row["document_id"] == "https://example.org/math"
    assert row["source_metadata"]["url"].endswith("#section")
    assert "candidate-not-formally-admitted" in row["quality_flags"]
    assert (
        clean_record("zh_edu", dict(text=text, score=1.0126953125, source="WuDao"), "row")[1]
        == "score_outside_declared_0_1_range"
    )


def test_duplicate_alias_preserves_transitive_origin_groups(tmp_path):
    builder = CorpusBuilder(tmp_path, val_buckets=50, test_buckets=100)
    common = dict(revision="v1", license="CC0", lang="en", task="natural", stage="pretrain")
    a = dict(
        common,
        source="first",
        item_id="a",
        group_id="original",
        text="A prism bends different wavelengths of visible light by different angles.",
    )
    b = dict(a, source="second", item_id="b", group_id="mirror")
    c = dict(
        common,
        source="second",
        item_id="c",
        group_id="mirror",
        text="The northern forest contains old trees with thick layers of bark and green moss.",
    )
    assert builder.add(a) and not builder.add(b) and builder.add(c)
    manifest = builder.finalize()
    groups = {group for (group,) in builder.db.execute("SELECT group_root FROM samples")}
    assert len(groups) == 1
    assert manifest["split_rule"].endswith("val 0.5%, sealed test 1%")
    for _identity, group, split in builder.db.execute("SELECT id,group_root,split FROM samples"):
        bucket = int(hashlib.sha256(f"42:{group}".encode()).hexdigest()[:16], 16) % 10000
        assert split == ("val" if bucket < 50 else "test" if bucket < 150 else "train")
    assert json.loads((tmp_path / "corpus-manifest.json").read_text())["counts"]["accepted"] == 2
    builder.db.close()


def test_invalid_split_fractions_are_rejected_before_writing(tmp_path):
    with pytest.raises(ValueError, match="held-out"):
        CorpusBuilder(tmp_path / "invalid", val_buckets=5000, test_buckets=5000)
    assert not (tmp_path / "invalid").exists()
