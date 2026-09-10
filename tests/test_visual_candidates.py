"""Visual source ratings, complete answers and same-image split identity."""

import io
import json
import random
import shutil
from types import SimpleNamespace

import pytest
from PIL import Image
from tokenizers import Tokenizer, models, pre_tokenizers

from minifrontier.data import visual_sources


def row(seed=1):
    rng = random.Random(seed)
    image = Image.frombytes("RGB", (48, 48), rng.randbytes(48 * 48 * 3))
    binary = io.BytesIO()
    image.save(binary, format="PNG")
    item = dict(
        source="unit-test-source",
        images=[dict(bytes=binary.getvalue(), path="original/image.png")],
        texts=[
            dict(
                user="Describe the contents of this image.",
                assistant="The image contains a garden with several tall trees and a small stone path.",
            )
        ],
    )
    for name in (
        "image_correspondence_ratings",
        "visual_dependency_ratings",
        "formatting_ratings",
        "relevance_ratings",
    ):
        item[name] = [4]
    return item


def test_grounded_turn_filter_keeps_original_identity_and_complete_answer():
    item = row()
    selected, reason = visual_sources.candidate_turns("allava_laion", item)
    assert reason is None and selected[0]["index"] == 0 and selected[0]["task"] == "caption"
    assert selected[0]["answer"] == item["texts"][0]["assistant"]
    item["visual_dependency_ratings"] = [2]
    assert visual_sources.candidate_turns("allava_laion", item)[1] == "no_complete_grounded_turn"
    item["visual_dependency_ratings"] = []
    assert (
        visual_sources.candidate_turns("allava_laion", item)[1] == "missing_or_misaligned_ratings"
    )


def test_visual_candidate_bytes_groups_and_counts_are_distinct_from_training(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(free=900 * 1024**3))

    def rows(name, *, seed, audit, specification):
        audit["reads"] = [dict(file="fixture.parquet", row_group=0, rows=1)]
        yield row(seed), "fixture.parquet:rg0:row0"

    monkeypatch.setattr(visual_sources, "source_rows", rows)
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    targets = dict.fromkeys(visual_sources.VISUAL_CANDIDATES, 1)
    root = tmp_path / "candidate"
    report = visual_sources.build_visual_candidates(
        root, tmp_path / "tokenizer.json", targets=targets
    )
    assert report["unique_images"] == 3
    assert sum(report["split_independent_images"].values()) == 3
    assert report["status"] == "candidate_slice_complete_pending_admission"
    assert report["formal_admission"] is False and report["main_budget_eligible"] is False
    assert report["media_bytes"] == sum(
        p.stat().st_size for p in (root / "images").rglob("*.image")
    )
    manifest = json.loads((root / "corpus-manifest.json").read_text())
    assert manifest["split_rule"].endswith("val 0.5%, sealed test 1%")
    assert report["review"]["status"] == "awaiting_manual_review"
    for source in report["sources"].values():
        assert source["accepted_records"] == source["unique_images"] == 1
        assert source["accepted_reference_tokens"] > source["answer_reference_tokens"] > 0


def test_visual_budget_rejects_invalid_partition_before_creating_files(tmp_path):
    with pytest.raises(ValueError, match="budgets"):
        visual_sources.build_visual_candidates(
            tmp_path / "invalid", "missing", max_gib=2, metadata_gib=3
        )
    assert not (tmp_path / "invalid").exists()


def test_pinned_catalog_transport_retry_is_bounded_and_does_not_retry_schema_errors(monkeypatch):
    httpx = pytest.importorskip("httpx")
    from minifrontier.data import public_sources

    monkeypatch.setattr(public_sources.time, "sleep", lambda _: None)
    calls, audit = [], {}

    def interrupted():
        calls.append(1)
        raise httpx.RemoteProtocolError("interrupted catalog response")

    with pytest.raises(httpx.RemoteProtocolError):
        public_sources.retry_transport(interrupted, audit=audit, operation="test_catalog")
    assert len(calls) == len(audit["transport_failures"]) == 3

    def invalid_schema():
        raise ValueError("schema mismatch")

    unchanged = {}
    with pytest.raises(ValueError, match="schema"):
        public_sources.retry_transport(invalid_schema, audit=unchanged, operation="test_schema")
    assert unchanged == {}
