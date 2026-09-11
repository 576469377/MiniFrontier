"""Cross-host media grouping must propagate holdouts through origin and pHash links."""

import json
import shutil
from types import SimpleNamespace

import pytest

from minifrontier.data import sha256
from minifrontier.data.corpus import CorpusBuilder
from minifrontier.data.media_inventory import (
    FORMAT,
    TEXT_FORMAT,
    _text_neighbors,
    audit_media_identities,
    close_media_candidate_review,
    export_media_identities,
)
from minifrontier.data.partitions import (
    create_media_exclusion_view,
    create_partition_view,
    open_corpus,
)


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


def test_rendered_text_groups_across_fonts_without_joining_unrelated_page_layouts(tmp_path):
    def rendered(path, group, split, rgb, phash, text, origin=""):
        from minifrontier.data.media_inventory import _rendered_signature

        value = dict(group=group, phash=phash)
        if text is not None:
            value["rendered_text"] = _rendered_signature(
                dict(
                    source="MiniFrontier/source-grounded-ocr",
                    visual_answer=text,
                    rendering={"font": "fixture"},
                    text_origin={"sample_id": "source"},
                )
            )
        p = inventory(path, {group: node(split, origin) if origin else node(split)}, {rgb: value})
        data = json.loads(p.read_text())
        data["format"] = TEXT_FORMAT
        p.write_text(json.dumps(data))
        return p

    text = "The expedition carefully recorded every observation before returning to the coastal research station."
    inputs = {
        "train": rendered(tmp_path / "train.json", "a", "train", "a" * 64, "0" * 16, text),
        # A font/layout change alters pHash dramatically, but must preserve the text holdout.
        "held": rendered(tmp_path / "held.json", "b", "test", "b" * 64, "f" * 16, text),
        # An identical low-frequency layout must not join unrelated printed content.
        "unrelated": rendered(
            tmp_path / "other.json",
            "c",
            "val",
            "c" * 64,
            "0" * 16,
            "地下水的变化需要结合当地岩层结构进行分析，观测仪器每天记录水位和温度，随后整理全部结果。",  # noqa: RUF001
        ),
    }
    result = audit_media_identities(inputs, tmp_path / "text.json")
    assert result["connected_groups"] == 2
    assert result["link_counts"] == {"rendered_text_jaccard_ge_0.85": 1}
    assert {m["inventory"] for m in result["split_conflicts"][0]["members"]} == {"train", "held"}
    assert not result["unresolved_rendered_visual_candidates"]

    # A natural/document image lacks exact printed-text ground truth. Keep its
    # pHash candidate explicit, and never publish a pass merely by ignoring it.
    external = rendered(tmp_path / "external.json", "d", "val", "d" * 64, "0" * 16, None)
    result = audit_media_identities(
        {"ocr": inputs["train"], "external": external}, tmp_path / "external-audit.json"
    )
    assert result["status"] == "rendered_visual_candidates_require_verification"
    assert len(result["unresolved_rendered_visual_candidates"]) == 1
    assert result["connected_groups"] == 2

    # Old exports of known rendered OCR must be regenerated, not silently treated
    # as natural images because they lack the new text signatures.
    old = inventory(
        tmp_path / "old.json",
        {"e": node("train", "source-group:MiniFrontier/source-grounded-ocr:parent")},
        {"e" * 64: dict(group="e", phash="0" * 16)},
    )
    with pytest.raises(ValueError, match="text-aware"):
        audit_media_identities({"old": old, "external": external}, tmp_path / "old-audit.json")


def test_text_prefix_join_matches_brute_force_including_threshold_and_budget():
    import random

    rng = random.Random(712)
    signatures = {}
    for i in range(50):
        base = set(rng.sample(range(200), rng.randint(10, 60)))
        signatures[2 * i] = base
        signatures[2 * i + 1] = (base - set(sorted(base)[: rng.randint(0, 4)])) | {200 + i}
    expected = {
        (i, j)
        for i, a in signatures.items()
        for j, b in signatures.items()
        if i > j and 100 * len(a & b) >= 85 * len(a | b)
    }
    assert set(_text_neighbors(signatures, max_comparisons=10000)) == expected
    boundary = {0: set(range(20)), 1: set(range(17)), 2: set(range(16))}
    assert (1, 0) in set(_text_neighbors(boundary, max_comparisons=100))
    assert (2, 0) not in set(_text_neighbors(boundary, max_comparisons=100))
    with pytest.raises(ValueError, match="comparison budget"):
        list(_text_neighbors({i: set(range(20)) for i in range(5)}, max_comparisons=1))
    for queries in ({0, 19, 33, 98}, set(), set(signatures)):
        assert set(_text_neighbors(signatures, max_comparisons=10000, query_keys=queries)) == {
            (i, j) for i, j in expected if i in queries or j in queries
        }
    # Dense old duplicates must not consume the new-pair comparison budget.
    assert (
        list(
            _text_neighbors(
                {i: set(range(20)) for i in range(5)}, max_comparisons=1, query_keys=set()
            )
        )
        == []
    )


@pytest.mark.parametrize("added_first", [False, True])
def test_incremental_media_audit_matches_full_graph_and_only_emits_new_candidates(
    tmp_path, added_first
):
    import hashlib

    from minifrontier.data.media_inventory import _rendered_signature

    text_a = "The expedition recorded every observation before returning to the research station."
    text_b = "Local farmers plant seeds and measure soil temperature throughout the spring season."

    def make(name, rows):
        groups, images = {}, {}
        for group, split, rgb, code, text, origin in rows:
            groups[group] = node(split, origin) if origin else node(split)
            images[rgb] = dict(group=group, phash=f"{code:016x}")
            if text:
                images[rgb]["rendered_text"] = _rendered_signature(
                    dict(
                        source="MiniFrontier/source-grounded-ocr",
                        visual_answer=text,
                        rendering={"fixture": True},
                        text_origin={"fixture": True},
                    )
                )
        path = inventory(tmp_path / (name + ".json"), groups, images)
        value = json.loads(path.read_text())
        value["format"] = TEXT_FORMAT
        path.write_text(json.dumps(value))
        return path

    old = {
        "a": make("a", [("a", "train", "a" * 64, 0, None, "shared")]),
        "b": make("b", [("b", "train", "b" * 64, 63, None, "shared")]),
        "r": make(
            "r",
            [
                ("r", "train", "c" * 64, 2**64 - 1, text_a, ""),
                ("s", "train", "d" * 64, 0, text_b, ""),
            ],
        ),
    }
    parent = tmp_path / "parent.json"
    original = audit_media_identities(old, parent)
    assert len(original["unresolved_rendered_visual_candidates"]) == 2
    review = tmp_path / "fixture-review.json"
    decisions = [
        dict(
            candidate_index=i,
            candidate_sha256=hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest(),
            rendered_rgb_sha256=v["rendered"]["rgb_sha256"],
            external_rgb_sha256=v["external"]["rgb_sha256"],
            rendered_file_sha256="1" * 64,
            external_raw_sha256="2" * 64,
            disposition="not_visual_duplicate",
            reason="Synthetic fixture: unrelated prose and natural image metadata share their pHash.",
        )
        for i, v in enumerate(original["unresolved_rendered_visual_candidates"])
    ]
    review.write_text(
        json.dumps(
            dict(
                kind="model_assisted_visual_candidate_review",
                group_audit_sha256=sha256(parent),
                reviewer="synthetic fixture",
                decisions=decisions,
            )
        )
    )
    closed = tmp_path / "closed.json"
    close_media_candidate_review(parent, [review], closed)
    added = {
        "new-rendered": make("new-rendered", [("t", "test", "e" * 64, 0, text_a, "")]),
        "new-natural": make(
            "new-natural",
            [
                ("u", "train", "f" * 64, 2**64 - 1, None, ""),
                ("v", "test", "a" * 64, 0, None, "shared"),
            ],
        ),
    }
    inputs = dict(added, **old) if added_first else dict(old, **added)
    full = audit_media_identities(inputs, tmp_path / "full.json")
    incremental = audit_media_identities(
        inputs, tmp_path / "incremental.json", previous_audit=closed
    )
    for field in ("linked_components", "split_conflicts"):
        assert sorted(json.dumps(v, sort_keys=True) for v in full[field]) == sorted(
            json.dumps(v, sort_keys=True) for v in incremental[field]
        )
    for field in (
        "input_images",
        "unique_rgb_images",
        "input_groups",
        "connected_groups",
        "link_counts",
        "cross_inventory_link_counts",
        "rendered_images",
    ):
        assert full[field] == incremental[field]
    expected = [
        v
        for v in full["unresolved_rendered_visual_candidates"]
        if v["rendered"]["inventory"] in added or v["external"]["inventory"] in added
    ]
    assert sorted(json.dumps(v, sort_keys=True) for v in expected) == sorted(
        json.dumps(v, sort_keys=True) for v in incremental["unresolved_rendered_visual_candidates"]
    )
    assert incremental["incremental"]["previous_audit_sha256"] == sha256(closed)
    assert incremental["incremental"]["inherited_reviewed_visual_candidates"] == 2
    assert not incremental["formal_admission"] and incremental["split_conflicts"]
    with pytest.raises(ValueError, match="closed and bind"):
        audit_media_identities(inputs, tmp_path / "unreviewed.json", previous_audit=parent)
    with pytest.raises(ValueError, match="closed and bind"):
        audit_media_identities(old, tmp_path / "no-additions.json", previous_audit=closed)
    old["a"].write_text(old["a"].read_text() + "\n")
    with pytest.raises(ValueError, match="closed and bind"):
        audit_media_identities(inputs, tmp_path / "changed.json", previous_audit=closed)
    assert not (tmp_path / "changed.json").exists()


def test_visual_candidates_match_brute_force_with_stable_order_after_filtering(tmp_path):
    from minifrontier.data.media_inventory import _rendered_signature

    signature = _rendered_signature(
        dict(
            source="MiniFrontier/source-grounded-ocr",
            visual_answer="A complete printed sentence with distinct words.",
            rendering={"font": "fixture"},
            text_origin={"sample_id": "parent"},
        )
    )
    # Input order differs from hash order. Seven-bit neighbors share bands but
    # must not be emitted; exact RGB aliases must not become review candidates.
    hashes = [127, 63, 15, 31, 0, 0xAAAAAAAAAAAAAAAA]
    external = inventory(
        tmp_path / "external.json",
        {str(i): node("train") for i in range(len(hashes))},
        {f"{i:064x}": dict(group=str(i), phash=f"{code:016x}") for i, code in enumerate(hashes)},
    )
    rendered = inventory(
        tmp_path / "ocr.json",
        {"ocr": node("train")},
        {f"{4:064x}": dict(group="ocr", phash="0" * 16, rendered_text=signature)},
    )
    value = json.loads(rendered.read_text())
    value["format"] = TEXT_FORMAT
    rendered.write_text(json.dumps(value))
    events = []
    result = audit_media_identities(
        dict(external=external, ocr=rendered), tmp_path / "audit.json", progress=events.append
    )
    actual = [
        (x["external"]["rgb_sha256"], x["phash_distance"])
        for x in result["unresolved_rendered_visual_candidates"]
    ]
    expected = [
        (f"{i:064x}", code.bit_count())
        for code, i in sorted((code, i) for i, code in enumerate(hashes))
        if code.bit_count() <= 6 and i != 4
    ]
    assert actual == expected
    assert any(x["state"] == "matching_rendered_text" for x in events)
    assert any(x["state"] == "matching_rendered_external_images" for x in events)


def test_review_closure_requires_complete_bound_decisions_without_claiming_human_quality(tmp_path):
    import hashlib

    candidate = dict(
        rendered=dict(inventory="ocr", group="one", split="train", rgb_sha256="a" * 64),
        external=dict(inventory="external", group="two", split="val", rgb_sha256="b" * 64),
        phash_distance=6,
    )
    parent = tmp_path / "grouping.json"
    parent.write_text(
        json.dumps(
            dict(
                kind="cross_corpus_media_group_audit",
                split_conflicts=[],
                status="rendered_visual_candidates_require_verification",
                unresolved_rendered_visual_candidates=[candidate],
                formal_admission=False,
            )
        )
    )
    decision = dict(
        candidate_index=0,
        candidate_sha256=hashlib.sha256(json.dumps(candidate, sort_keys=True).encode()).hexdigest(),
        rendered_rgb_sha256="a" * 64,
        external_rgb_sha256="b" * 64,
        rendered_file_sha256="c" * 64,
        external_raw_sha256="d" * 64,
        disposition="not_visual_duplicate",
        reason="Printed prose versus a colored chart.",
    )
    review = tmp_path / "review.json"
    valid = dict(
        kind="model_assisted_visual_candidate_review",
        group_audit_sha256=sha256(parent),
        reviewer="model-assisted image inspection",
        decisions=[decision],
    )
    for change in [
        dict(decisions=[]),
        dict(decisions=[decision, decision]),
        dict(group_audit_sha256="e" * 64),
        dict(decisions=[dict(decision, candidate_index=True)]),
        dict(decisions=[dict(decision, candidate_sha256="e" * 64)]),
        dict(decisions=[dict(decision, disposition="duplicate")]),
        dict(decisions=[dict(decision, external_raw_sha256="missing")]),
    ]:
        review.write_text(json.dumps(dict(valid, **change)))
        with pytest.raises(ValueError):
            close_media_candidate_review(parent, [review], tmp_path / "rejected.json")
        assert not (tmp_path / "rejected.json").exists()
    review.write_text(json.dumps(valid))
    proof = close_media_candidate_review(parent, [review], tmp_path / "closed.json")
    assert proof["status"] == "mechanical_group_checks_passed_with_model_assisted_review"
    assert proof["reviewed_visual_candidates"] == 1
    assert not proof["unresolved_rendered_visual_candidates"]
    assert not proof["formal_admission"] and not proof["human_source_quality_review_completed"]
    assert proof["parent_group_audit_sha256"] == sha256(parent)
    assert json.loads(parent.read_text())["unresolved_rendered_visual_candidates"] == [candidate]
    changed = json.loads(parent.read_text())
    changed["split_conflicts"] = [{"unresolved": True}]
    parent.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="split conflicts"):
        close_media_candidate_review(parent, [review], tmp_path / "still-conflicting.json")
    # Visual decisions can be recorded while a separate partition repair is pending.
    # This must preserve the conflicts and must not publish a grouping pass.
    changed["status"] = "split_conflicts_require_partition_update"
    parent.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="bound"):
        close_media_candidate_review(parent, [review], tmp_path / "stale-review.json")
    valid["group_audit_sha256"] = sha256(parent)
    review.write_text(json.dumps(valid))
    pending = close_media_candidate_review(parent, [review], tmp_path / "visual-reviewed.json")
    assert pending["status"] == "split_conflicts_require_partition_update"
    assert pending["split_conflicts"] == changed["split_conflicts"]
    assert pending["unresolved_rendered_visual_candidates"] == []
    assert pending["reviewed_visual_candidates"] == 1
    assert not pending["formal_admission"] and not pending["human_source_quality_review_completed"]
    assert json.loads(parent.read_text()) == changed


def test_deferred_ocr_layout_hash_preserves_source_holds_and_reports_pending_near_audit(tmp_path):
    for enabled in (True, False):
        builder = CorpusBuilder(
            tmp_path / str(enabled), val_buckets=1, test_buckets=1, group_image_phash=enabled
        )
        for group, text, rgb in [
            ("held", "The first passage discusses a distant research station.", "a"),
            ("train", "A different passage records the method for planting seeds.", "b"),
        ]:
            assert builder.add(
                dict(
                    source="fixture",
                    revision="fixed",
                    item_id=group,
                    group_id=group,
                    license="CC0",
                    lang="en",
                    task="ocr_document",
                    stage="pretrain",
                    text=text,
                    media=[dict(kind="image", rgb_sha256=rgb * 64, phash="0" * 16)],
                )
            )
        manifest = builder.finalize(split_locks={"source-group:fixture:held": "test"})
        splits = {
            json.loads(p)["group_id"]: s
            for p, s in builder.db.execute("SELECT payload,split FROM samples")
        }
        assert splits["held"] == "test"
        assert splits["train"] == ("test" if enabled else "train")
        if not enabled:
            assert manifest["image_phash_grouping"] is False
            assert manifest["image_near_duplicate_audit"].startswith("deferred")
        builder.db.close()


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
        reference_tokens=10,
        answer_reference_tokens=8,
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
    last_media = dict(row["media"][0], rgb_sha256="c" * 64, phash="f0f0f0f0f0f0f0f0")
    assert builder.add(dict(row, item_id="three", group_id="quarantine", media=[last_media]))
    builder.finalize(split_locks={"source-group:fixture:old-duplicate": "test"})
    builder.db.close()
    before = sha256(root / "corpus.sqlite")
    report = export_media_identities(root, tmp_path / "identities.json")
    assert report["image_count"] == report["group_count"] == 3
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
    (root / "review-samples.jsonl").write_text("")
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
    assert effective["split_groups"] == {"val": 1, "test": 1, "train": 1}
    quarantine = effective["images"]["c" * 64]["group"]
    grouping = tmp_path / "grouping.json"
    evidence = dict(
        kind="cross_corpus_media_group_audit",
        inputs={
            "fixture": dict(
                corpus_manifest_sha256=sha256(view / "corpus-manifest.json"), database_sha256=before
            )
        },
        split_conflicts=[
            dict(
                required_split="test",
                members=[dict(inventory="fixture", group=quarantine, split="train", records=1)],
            )
        ],
    )
    grouping.write_text(json.dumps(evidence))
    excluded = tmp_path / "excluded"
    result = create_media_exclusion_view(view, grouping, "fixture", excluded)
    assert result["newly_excluded_records"] == 1
    isolated = export_media_identities(excluded, tmp_path / "isolated.json")
    assert isolated["split_groups"] == {"val": 1, "test": 1}
    assert quarantine not in isolated["groups"]
    assert sha256(root / "corpus.sqlite") == before
    with open_corpus(view) as original:
        assert original.execute("SELECT COUNT(*) FROM samples").fetchone() == (3,)
    # A grouping proposal cannot quarantine an existing validation group.
    evidence["split_conflicts"][0]["members"][0]["group"] = train
    grouping.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="membership differs"):
        create_media_exclusion_view(view, grouping, "fixture", tmp_path / "invalid-exclusion")
    assert sha256(root / "corpus.sqlite") == before
    with pytest.raises(ValueError, match="byte budget"):
        export_media_identities(root, tmp_path / "tiny.json", max_bytes=10)
    assert not (tmp_path / "tiny.json").exists()
