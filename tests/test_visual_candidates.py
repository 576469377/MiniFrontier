"""Visual source ratings, complete answers and same-image split identity."""

import io
import json
import os
import random
import shutil
from types import SimpleNamespace

import pytest
from PIL import Image
from tokenizers import Tokenizer, models, pre_tokenizers

from minifrontier.data import sha256, visual_sources


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
                user="Please provide a detailed narrative of the image.",
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


def test_continuation_preserves_cursor_and_excludes_previous_pixels(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(free=900 * 1024**3))
    import httpx

    calls, interrupted = [], []
    items = [row(1), row(1), row(2), row(3)]

    def rows(name, *, seed, audit, specification, skip_rows=0):
        calls.append((seed, skip_rows))
        for i in range(skip_rows, len(items)):
            if skip_rows == 1 and i == 3 and not interrupted:
                interrupted.append(True)
                raise httpx.ReadTimeout("fixture transport interruption")
            yield items[i], f"fixture:row{i}"

    monkeypatch.setattr(visual_sources, "source_rows", rows)
    tokenizer = tmp_path / "tokenizer.json"
    Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]")).save(str(tokenizer))
    first, second = tmp_path / "first", tmp_path / "second"
    a = visual_sources.build_visual_candidates(first, tokenizer, targets={"allava_laion": 1})
    before = {p.name: sha256(p) for p in first.iterdir() if p.is_file()}
    continuation = tmp_path / "continuation.json"
    exported = visual_sources.export_visual_continuation(first, continuation)
    assert exported["sources"]["allava_laion"]["row_offset"] == 1
    assert len(exported["excluded_rgb_sha256"]) == a["unique_images"] == 1
    with pytest.raises(httpx.ReadTimeout):
        visual_sources.build_visual_candidates(
            second, tokenizer, targets={"allava_laion": 2}, continuation=continuation
        )
    partial = json.loads((second / "source-audit.json").read_text())
    assert partial["sources"]["allava_laion"]["read_rows"] == 2
    altered = tmp_path / "altered.json"
    modified = json.loads(continuation.read_text())
    modified["excluded_rgb_sha256"] = []
    altered.write_text(json.dumps(modified))
    with pytest.raises(ValueError, match="unchanged"):
        visual_sources.build_visual_candidates(
            second, tokenizer, targets={"allava_laion": 2}, continuation=altered, resume=True
        )
    result = visual_sources.build_visual_candidates(
        second, tokenizer, targets={"allava_laion": 2}, continuation=continuation, resume=True
    )
    assert calls == [(20260911, 0), (20260911, 1), (20260911, 3)]
    assert result["unique_images"] == 2 and result["formal_admission"] is False
    assert result["sources"]["allava_laion"]["rejected"]["previous_slice_image"] == 1
    assert len(list((second / "images").rglob("*.image"))) == 2
    third_cursor = visual_sources.export_visual_continuation(second, tmp_path / "third.json")
    assert third_cursor["sources"]["allava_laion"]["row_offset"] == 4
    assert len(third_cursor["excluded_rgb_sha256"]) == 3
    assert {p.name: sha256(p) for p in first.iterdir() if p.is_file()} == before


def test_quota_continuation_replays_uncommitted_boundary_without_changing_parent(quota_stopped):
    root, run = quota_stopped
    visual_sources.finalize_storage_limited_slice(root, run)
    before = sha256(root / "corpus.sqlite")
    exported = visual_sources.export_visual_continuation(root, root.parent / "continuation.json")
    assert exported["sources"]["allava_laion"]["row_offset"] == 1
    assert len(exported["excluded_rgb_sha256"]) == 1
    assert sha256(root / "corpus.sqlite") == before


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


@pytest.fixture
def quota_stopped(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(free=900 * 1024**3))
    items = [row(1), row(2)]

    def rows(*args, **kwargs):
        yield from ((item, f"fixture:row{i}") for i, item in enumerate(items))

    monkeypatch.setattr(visual_sources, "source_rows", rows)
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    root = tmp_path / "media"
    media_bytes = sum(len(i["images"][0]["bytes"]) for i in items)
    with pytest.raises(ValueError, match="media byte budget"):
        visual_sources.build_visual_candidates(
            root,
            tmp_path / "tokenizer.json",
            targets={"allava_laion": 2, "CoSyn_400k_document": 3},
            max_gib=3 + media_bytes * 0.75 / 1024**3,
            metadata_gib=3,
        )
    run = tmp_path / "run.json"
    run.write_text(
        json.dumps(
            dict(
                kind="data_construction",
                pid=2**31 - 1,
                command=["python", "-m", "minifrontier.data.visual_sources", "--output", str(root)],
            )
        )
    )
    return root, run


def test_finalize_quota_slice_keeps_pixels_and_explicit_unfilled_targets(
    quota_stopped, monkeypatch
):
    root, run = quota_stopped
    before = {p.name: sha256(p) for p in (root / "images").rglob("*.image")}
    old_audit = sha256(root / "source-audit.json")

    def no_download(*args, **kwargs):
        raise AssertionError("closing retained inventory must not contact a source")

    monkeypatch.setattr(visual_sources, "source_rows", no_download)
    report = visual_sources.finalize_storage_limited_slice(root, run)
    assert report["status"] == "candidate_inventory_below_target"
    assert report["remaining_independent_image_targets"] == {
        "allava_laion": 1,
        "CoSyn_400k_document": 3,
    }
    assert report["unique_images"] == sum(report["split_independent_images"].values()) == 1
    assert report["finalization"]["new_download_bytes"] == 0
    assert not report["formal_admission"] and not report["main_budget_eligible"]
    assert {p.name: sha256(p) for p in (root / "images").rglob("*.image")} == before
    assert sha256(root / "source-audit-before-slice-finalize.json") == old_audit
    assert report["review"]["sha256"] == sha256(root / "review-samples.jsonl")
    with pytest.raises(ValueError, match="unfinalized"):
        visual_sources.finalize_storage_limited_slice(root, run)


@pytest.mark.parametrize("fault", ["live", "pixels", "orphan", "counters", "transport"])
def test_slice_finalization_rejects_unverified_inventory_without_repartitioning(
    quota_stopped, fault
):
    root, run = quota_stopped
    before = sha256(root / "corpus.sqlite")
    if fault == "live":
        r = json.loads(run.read_text())
        r["pid"] = os.getpid()
        run.write_text(json.dumps(r))
    elif fault == "pixels":
        path = next((root / "images").rglob("*.image"))
        path.write_bytes(b"damaged")
    elif fault == "orphan":
        (root / "images/orphan.image").write_bytes(b"unreferenced")
    else:
        p = root / "source-audit.json"
        a = json.loads(p.read_text())
        if fault == "counters":
            a["sources"]["allava_laion"]["answer_reference_tokens"] += 1
        else:
            a["error"] = "ReadTimeoutError: source interrupted"
        p.write_text(json.dumps(a))
    with pytest.raises(ValueError):
        visual_sources.finalize_storage_limited_slice(root, run)
    assert sha256(root / "corpus.sqlite") == before
    assert not (root / "corpus-manifest.json").exists()


def test_compact_legacy_visual_metadata_preserves_all_rows_splits_and_pixels(quota_stopped):
    import sqlite3

    root, run = quota_stopped
    audit = visual_sources.finalize_storage_limited_slice(root, run)
    with sqlite3.connect(root / "corpus.sqlite") as db:
        db.execute("ALTER TABLE links RENAME TO unique_links")
        db.execute("CREATE TABLE links(id TEXT,key TEXT)")
        db.execute("CREATE INDEX link_key_idx ON links(key)")
        for _ in range(7):
            db.execute("INSERT INTO links SELECT * FROM unique_links")
        db.execute("DROP TABLE unique_links")
    manifest_path = root / "corpus-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["database_sha256"] = sha256(root / "corpus.sqlite")
    manifest_path.write_text(json.dumps(manifest))
    audit["corpus"] = manifest
    (root / "source-audit.json").write_text(json.dumps(audit))
    before = {p.name: sha256(p) for p in root.iterdir() if p.is_file()}
    output = root.with_name("compacted")
    result = visual_sources.compact_visual_inventory(root, output)
    assert result["status"] == "mechanical_checks_passed_pending_quality_admission"
    assert result["original_content_and_splits_unchanged"] and result["new_media_bytes"] == 0
    counts = result["tables"]["links"]
    assert counts["original_rows"] == 7 * counts["compact_rows"]
    assert result["database_bytes"] <= result["database_cap_bytes"]
    assert {p.name: sha256(p) for p in root.iterdir() if p.is_file()} == before
    old = next((root / "images").rglob("*.image"))
    assert os.path.samefile(old, output / old.relative_to(root))
    copied = json.loads((output / "source-audit.json").read_text())
    assert copied["split_independent_images"] == audit["split_independent_images"]
    assert copied["review"] == audit["review"] and not copied["formal_admission"]
    with pytest.raises(FileExistsError):
        visual_sources.compact_visual_inventory(root, output)


@pytest.mark.parametrize("fault", ["hash", "pixels", "budget", "admitted"])
def test_compaction_rejects_invalid_source_before_publishing_candidate(quota_stopped, fault):
    root, run = quota_stopped
    audit = visual_sources.finalize_storage_limited_slice(root, run)
    if fault == "hash":
        audit["corpus"]["database_sha256"] = "0" * 64
    elif fault == "pixels":
        next((root / "images").rglob("*.image")).write_bytes(b"broken")
    elif fault == "budget":
        audit["metadata_gib"] = 0.001
    else:
        audit["formal_admission"] = True
    (root / "source-audit.json").write_text(json.dumps(audit))
    before = sha256(root / "corpus.sqlite")
    output = root.with_name("invalid-compaction")
    with pytest.raises(ValueError):
        visual_sources.compact_visual_inventory(root, output)
    assert sha256(root / "corpus.sqlite") == before
    assert not output.exists()
