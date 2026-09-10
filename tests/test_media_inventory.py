"""Cross-host media grouping must propagate holdouts through origin and pHash links."""

import json
import shutil
from types import SimpleNamespace

import pytest

from minifrontier.data import sha256
from minifrontier.data.corpus import CorpusBuilder
from minifrontier.data.media_inventory import (
    FORMAT,
    audit_media_identities,
    export_media_identities,
)
from minifrontier.data.partitions import create_partition_view


@pytest.fixture(autouse=True)
def available_space(monkeypatch):
    monkeypatch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(free=900 * 1024**3))


def inventory(path, groups, images):
    path.write_text(
        json.dumps(
            dict(
                format=FORMAT,
                groups=groups,
                images=images,
                group_count=len(groups),
                image_count=len(images),
                corpus_manifest_sha256="fixture",
                database_sha256="fixture",
            )
        )
    )
    return path


def node(split, *origins):
    return dict(split=split, records=1, origin_keys=list(origins))


def test_cross_inventory_grouping_preserves_transitive_sealed_test_and_six_bit_boundary(tmp_path):
    # A(train) -- origin -- B(val) -- six changed bits -- C(test).
    a = inventory(
        tmp_path / "a.json",
        {"a": node("train", "shared-origin"), "safe": node("train")},
        {
            "a" * 64: dict(group="a", phash="ffffffffffffffff"),
            "d" * 64: dict(group="safe", phash="aaaaaaaaaaaaaaaa"),
        },
    )
    b = inventory(
        tmp_path / "b.json",
        {"b": node("val", "shared-origin")},
        {"b" * 64: dict(group="b", phash="0000000000000000")},
    )
    c = inventory(
        tmp_path / "c.json",
        {"c": node("test"), "seven": node("train")},
        {
            "c" * 64: dict(group="c", phash="0080200802008020"),
            "e" * 64: dict(group="seven", phash="5555555555555555"),
        },
    )
    # Six nonzero bits lie in six separate bands; the seventh band finds the pair.
    assert int("0080200802008020", 16).bit_count() == 6
    hashes = {path: sha256(path) for path in (a, b, c)}
    report = audit_media_identities(dict(a=a, b=b, c=c), tmp_path / "report.json")
    assert report["status"] == "split_conflicts_require_partition_update"
    assert len(report["split_conflicts"]) == 1
    conflict = report["split_conflicts"][0]
    assert conflict["required_split"] == "test"
    assert {(v["inventory"], v["group"]) for v in conflict["members"]} == {
        ("a", "a"),
        ("b", "b"),
        ("c", "c"),
    }
    assert not report["partition_changes_applied"] and not report["formal_admission"]
    assert report["connected_groups"] == 3
    assert {path: sha256(path) for path in hashes} == hashes
    # Moving just beyond the permitted distance removes the pHash edge.
    changed = json.loads(c.read_text())
    changed["images"]["c" * 64]["phash"] = "0080200802008021"
    c.write_text(json.dumps(changed))
    second = audit_media_identities(dict(a=a, b=b, c=c), tmp_path / "seven-report.json")
    assert second["connected_groups"] == 4
    assert second["split_conflicts"][0]["required_split"] == "val"


def test_exact_pixels_are_grouped_and_edge_budget_cannot_publish_a_pass(tmp_path):
    inventories = {
        name: inventory(
            tmp_path / (name + ".json"),
            {name: node("train")},
            {"a" * 64: dict(group=name, phash="0102030405060708")},
        )
        for name in ("a", "b", "c")
    }
    report = audit_media_identities(inventories, tmp_path / "report.json")
    assert report["unique_rgb_images"] == report["connected_groups"] == 1
    assert report["status"] == "mechanical_group_checks_passed"
    assert report["cross_inventory_link_counts"]["exact_rgb"] == 2
    with pytest.raises(ValueError, match="edge bound"):
        audit_media_identities(inventories, tmp_path / "too-small.json", max_edges=1)
    assert not (tmp_path / "too-small.json").exists()


def test_identity_export_retains_rejected_origin_aliases_and_actual_holdouts(tmp_path):
    root = tmp_path / "canonical"
    builder = CorpusBuilder(root, val_buckets=1, test_buckets=1)
    row = dict(
        source="fixture",
        revision="fixed",
        item_id="one",
        group_id="original",
        license="CC0",
        lang="en",
        task="caption",
        stage="pretrain",
        text="A coastline photographed on a clear afternoon.",
        media=[
            dict(
                kind="image",
                path="original.png",
                rgb_sha256="a" * 64,
                phash="0123456789abcdef",
                phash_version="dct32-low8-median-v1",
            )
        ],
    )
    assert builder.add(row)
    assert not builder.add(dict(row, item_id="alias", group_id="old-duplicate"))
    extra_media = dict(row["media"][0], rgb_sha256="b" * 64, phash="ffffffffffffffff")
    assert builder.add(dict(row, item_id="two", group_id="new-train", media=[extra_media]))
    builder.finalize(split_locks={"source-group:fixture:old-duplicate": "test"})
    builder.db.close()
    before = sha256(root / "corpus.sqlite")
    report = export_media_identities(root, tmp_path / "identities.json")
    assert report["image_count"] == report["group_count"] == 2
    group = report["groups"][report["images"]["a" * 64]["group"]]
    assert group["split"] == "test"
    assert "source-group:fixture:old-duplicate" in group["origin_keys"]
    assert not report["raw_media_copied"] and not report["raw_pixels_redecoded"]
    assert sha256(root / "corpus.sqlite") == before
    train = report["images"]["b" * 64]["group"]
    assert report["groups"][train]["split"] == "train"
    (root / "source-audit.json").write_text(
        json.dumps(
            dict(status="candidate_slice_complete_pending_admission", formal_admission=False)
        )
    )
    (root / "integrity-and-split-audit.json").write_text("{}")
    reservation = tmp_path / "reservation.json"
    reservation.write_text(
        json.dumps(
            dict(
                corpus_manifest_sha256=sha256(root / "corpus-manifest.json"),
                integrity_audit_sha256=sha256(root / "integrity-and-split-audit.json"),
                minimum_validation_group_fraction=0.3,
                groups={train: "val"},
            )
        )
    )
    view = tmp_path / "effective-view"
    create_partition_view(root, reservation, view)
    effective = export_media_identities(view, tmp_path / "effective-identities.json")
    assert effective["groups"][train]["split"] == "val"
    assert effective["split_groups"] == {"val": 1, "test": 1}
    assert sha256(root / "corpus.sqlite") == before
    with pytest.raises(ValueError, match="byte budget"):
        export_media_identities(root, tmp_path / "tiny.json", max_bytes=10)
    assert not (tmp_path / "tiny.json").exists()
