"""Read-only corpus partitions over one immutable database.

An explicit manifest applies additional validation groups to every consumer.
The view has no corpus.sqlite alias: readers which ignore the partition contract
fail instead of silently training on the newly reserved validation records.
"""

import argparse
import contextlib
import hashlib
import json
import math
import os
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any

from minifrontier.data import sha256
from minifrontier.storage import GIB, require_space

FORMAT = "corpus-partition-view-v1"
TASK_FORMAT = "corpus-partition-view-v2"
HOLDOUT_FORMAT = "corpus-partition-view-v3"
VIEW_FORMATS = {FORMAT, TASK_FORMAT, HOLDOUT_FORMAT}


def _reserve(db, groups, excluded_groups=(), task_classification=None, test_groups=()):
    if not isinstance(groups, dict) or any(
        not isinstance(g, str) or not g or v != "val" for g, v in groups.items()
    ):
        raise ValueError("partition refinements may only reserve complete validation groups")
    db.execute("CREATE TEMP TABLE extra_validation(group_root TEXT PRIMARY KEY)")
    db.executemany("INSERT INTO extra_validation VALUES (?)", ((g,) for g in groups))
    actual = list(
        db.execute(
            "SELECT s.group_root,s.split FROM main.samples s JOIN extra_validation v "
            "ON s.group_root=v.group_root GROUP BY s.group_root,s.split"
        )
    )
    if len(actual) != len(groups) or any(split != "train" for _, split in actual):
        raise ValueError("reservation refers to missing, mixed or already held-out groups")
    if (
        not isinstance(excluded_groups, (list, tuple))
        or any(not isinstance(g, str) or not g or g in groups for g in excluded_groups)
        or len(set(excluded_groups)) != len(excluded_groups)
    ):
        raise ValueError("excluded groups must be distinct and outside validation reservations")
    db.execute("CREATE TEMP TABLE excluded_training_groups(group_root TEXT PRIMARY KEY)")
    db.executemany(
        "INSERT INTO excluded_training_groups VALUES (?)", ((g,) for g in excluded_groups)
    )
    excluded = list(
        db.execute(
            "SELECT s.group_root,s.split FROM main.samples s JOIN excluded_training_groups x "
            "ON s.group_root=x.group_root GROUP BY s.group_root,s.split"
        )
    )
    if len(excluded) != len(excluded_groups) or any(split != "train" for _, split in excluded):
        raise ValueError("exclusion refers to missing, mixed or held-out groups")
    if (
        not isinstance(test_groups, (list, tuple))
        or any(not isinstance(g, str) or not g or g in excluded_groups for g in test_groups)
        or len(set(test_groups)) != len(test_groups)
    ):
        raise ValueError("test promotions must be distinct and outside training exclusions")
    db.execute("CREATE TEMP TABLE extra_test(group_root TEXT PRIMARY KEY)")
    db.executemany("INSERT INTO extra_test VALUES (?)", ((g,) for g in test_groups))
    promoted = list(
        db.execute("""
        SELECT s.group_root,CASE WHEN v.group_root IS NOT NULL THEN 'val' ELSE s.split END
        FROM main.samples s JOIN extra_test t ON s.group_root=t.group_root
        LEFT JOIN extra_validation v ON s.group_root=v.group_root
        GROUP BY s.group_root,2
    """)
    )
    if len(promoted) != len(test_groups) or any(split != "val" for _, split in promoted):
        raise ValueError("only existing whole validation groups can be promoted to test")
    view = "reserved_samples" if task_classification is not None else "samples"
    db.execute(f"""CREATE TEMP VIEW {view} AS
        SELECT s.id,s.stage,s.source,s.task,s.first_question,s.text,s.payload,
               s.simhash,s.group_root,
               CASE WHEN t.group_root IS NOT NULL THEN 'test'
                    WHEN v.group_root IS NOT NULL THEN 'val' ELSE s.split END AS split
        FROM main.samples s LEFT JOIN extra_validation v ON s.group_root=v.group_root
        LEFT JOIN extra_test t ON s.group_root=t.group_root
        LEFT JOIN excluded_training_groups x ON s.group_root=x.group_root
        WHERE x.group_root IS NULL
    """)
    if task_classification is not None:
        from minifrontier.data.media_tasks import effective_task, task_policy

        if task_classification != task_policy():
            raise ValueError("media task classifier differs from the bound source policy")
        db.create_function("effective_media_task", 4, effective_task, deterministic=True)
        db.execute("""CREATE TEMP VIEW samples AS
            SELECT id,stage,source,new_task AS task,first_question,text,
              CASE WHEN new_task=task THEN payload
                   ELSE json_set(payload,'$.task',new_task) END AS payload,
              simhash,group_root,split
            FROM (SELECT *,effective_media_task(source,json_extract(payload,'$.revision'),
                  task,json_extract(payload,'$.visual_question')) AS new_task
                  FROM reserved_samples)
        """)


def open_corpus(root):
    """Open the effective partition on the calling thread; caller closes it."""
    root = Path(root).resolve()
    path = root / "corpus-manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {}
    if manifest.get("format") not in VIEW_FORMATS:
        return sqlite3.connect((root / "corpus.sqlite").as_uri() + "?mode=ro", uri=True)
    base = (root / manifest["base_corpus"]["path"]).resolve()
    if sha256(base / "corpus-manifest.json") != manifest["base_corpus"]["manifest_sha256"]:
        raise ValueError("partition base manifest changed")
    original = json.loads((base / "corpus-manifest.json").read_text())
    if original.get("format") in VIEW_FORMATS:
        raise ValueError("partition views must reference the original database directly")
    if sha256(base / "corpus.sqlite") != manifest["database_sha256"]:
        raise ValueError("partition base database changed")
    partition = (root / manifest["partition_file"]).resolve()
    if not partition.is_relative_to(root) or partition.stat().st_size > 16 * 1024**2:
        raise ValueError("partition metadata path/size exceeds the local bound")
    if sha256(partition) != manifest["partition_sha256"]:
        raise ValueError("partition reservation hash differs")
    db = sqlite3.connect((base / "corpus.sqlite").as_uri() + "?mode=ro", uri=True)
    try:
        overrides = json.loads(partition.read_text())
        policy = overrides.get("task_policy")
        test_groups = overrides.get("test_groups", [])
        if (manifest["format"] == HOLDOUT_FORMAT) != bool(test_groups):
            raise ValueError("test promotions require their versioned partition format")
        if manifest["format"] != HOLDOUT_FORMAT and (
            (manifest["format"] == TASK_FORMAT) != (policy is not None)
        ):
            raise ValueError("task-aware partition format and policy disagree")
        if policy is not None and manifest.get("task_policy") != policy:
            raise ValueError("partition task policy differs from its manifest")
        _reserve(db, overrides["groups"], overrides.get("excluded_groups", []), policy, test_groups)
        return db
    except BaseException:
        db.close()
        raise


def corpus_storage_root(db):
    """Locate immutable payload/media storage after opening the effective partition."""
    return Path(
        next(path for _, name, path in db.execute("PRAGMA database_list") if name == "main")
    ).parent


def create_partition_view(corpus_root, reservation, output):
    """Apply a checked reservation without copying or rewriting the source database."""
    from minifrontier.data.minifrontier1 import write_json

    source, root, proposal_path = (
        Path(corpus_root).resolve(),
        Path(output).resolve(),
        Path(reservation).resolve(),
    )
    if root.exists():
        raise FileExistsError("partition views are immutable; choose a new output")
    proposal = json.loads(proposal_path.read_text())
    manifest = json.loads((source / "corpus-manifest.json").read_text())
    audit = json.loads((source / "source-audit.json").read_text())
    if (
        manifest.get("format") in VIEW_FORMATS
        or audit.get("formal_admission")
        or audit["status"] != "candidate_slice_complete_pending_admission"
    ):
        raise ValueError("partition refinement requires a completed unadmitted canonical corpus")
    if (
        proposal["corpus_manifest_sha256"] != sha256(source / "corpus-manifest.json")
        or sha256(source / "corpus.sqlite") != manifest["database_sha256"]
        or proposal["integrity_audit_sha256"] != sha256(source / "integrity-and-split-audit.json")
    ):
        raise ValueError("reservation and original corpus hashes disagree")
    fraction = proposal["minimum_validation_group_fraction"]
    if not isinstance(fraction, (int, float)) or not 0 < fraction < 0.5:
        raise ValueError("invalid minimum validation group fraction")
    metadata_source = source
    inherited: dict[str, Any] = {}
    prior_holdouts = {}
    if proposal.get("prior_partition"):
        prior = (proposal_path.parent / proposal["prior_partition"]).resolve()
        if (
            sha256(prior / "corpus-manifest.json") != proposal["prior_partition_manifest_sha256"]
            or sha256(prior / "source-audit.json") != proposal["prior_partition_audit_sha256"]
        ):
            raise ValueError("prior partition identity differs from the reservation")
        previous = json.loads((prior / "corpus-manifest.json").read_text())
        if previous.get("format") not in VIEW_FORMATS:
            raise ValueError("additional reservation requires an existing partition view")
        with contextlib.closing(open_corpus(prior)) as db:
            if corpus_storage_root(db) != source:
                raise ValueError("prior partition refers to another canonical corpus")
            prior_holdouts = dict(
                db.execute("SELECT id,split FROM samples WHERE split!='train' ORDER BY id")
            )
        inherited = json.loads((prior / previous["partition_file"]).read_text())
        if any(proposal["groups"].get(g) != split for g, split in inherited["groups"].items()):
            raise ValueError("additional reservation cannot release previous validation groups")
        audit = json.loads((prior / "source-audit.json").read_text())
        if (
            audit.get("formal_admission")
            or audit["status"] != "candidate_slice_complete_pending_admission"
        ):
            raise ValueError("prior partition must remain an unadmitted candidate")
        metadata_source = prior
    statistics: dict[str, dict[str, Any]] = {}
    totals: dict[str, int] = {}
    tokens: dict[str, dict[str, int]] = {}
    media_statistics = {}
    with contextlib.closing(open_corpus(source)) as db:
        _reserve(
            db,
            proposal["groups"],
            inherited.get("excluded_groups", []),
            inherited.get("task_policy"),
            inherited.get("test_groups", []),
        )
        current_holdouts = dict(
            db.execute("SELECT id,split FROM samples WHERE split!='train' ORDER BY id")
        )
        if any(
            current_holdouts.get(identity) != split for identity, split in prior_holdouts.items()
        ):
            raise ValueError("additional reservation changed a previous held-out member")
        for name, split, records, groups, count in db.execute(
            "SELECT source,split,COUNT(*),COUNT(DISTINCT group_root),"
            "SUM(json_extract(payload,'$.reference_tokens')) FROM samples GROUP BY source,split"
        ):
            statistics.setdefault(name, {})[split] = dict(
                records=records, groups=groups, reference_tokens=count
            )
            totals[split] = totals.get(split, 0) + records
            tokens.setdefault(split, {})[name] = count
        for splits in statistics.values():
            total = sum(s["groups"] for s in splits.values())
            heldout = splits.get("val", {}).get("groups", 0)
            if heldout < math.ceil(total * fraction):
                raise ValueError("some source still falls short of its validation group minimum")
            splits["validation_group_fraction"] = heldout / total
        if audit.get("kind") == "visual_candidate_inventory":
            images: dict[str, set[str]] = {}
            for split, payload in db.execute("SELECT split,payload FROM samples"):
                images.setdefault(split, set()).update(
                    m["rgb_sha256"] for m in json.loads(payload)["media"]
                )
            answers: dict[str, dict[str, int]] = {}
            for split, task, count in db.execute(
                "SELECT split,task,SUM(json_extract(payload,'$.answer_reference_tokens')) "
                "FROM samples GROUP BY split,task"
            ):
                answers.setdefault(split, {})[task] = count
            media_statistics = dict(
                split_independent_images={k: len(v) for k, v in images.items()},
                split_independent_groups=dict(
                    db.execute(
                        "SELECT split,COUNT(DISTINCT group_root) FROM samples GROUP BY split"
                    )
                ),
                split_answer_reference_tokens=answers,
            )
        if (metadata_source / "review-samples.jsonl").exists():
            with (metadata_source / "review-samples.jsonl").open() as review:
                for line in review:
                    row = json.loads(line)["record"]
                    if db.execute(
                        "SELECT split FROM samples WHERE id=?", (row["sample_id"],)
                    ).fetchone() != ("train",):
                        raise ValueError(
                            "reservation invalidates an existing training review sample"
                        )
    require_space(root, 16 * 1024**2, reserve_bytes=80 * GIB)
    root.mkdir(parents=True)
    partition_path = root / "split-overrides.json"
    write_json(
        partition_path,
        dict(
            inherited,
            schema_version=1,
            groups=proposal["groups"],
            reservation_sha256=sha256(proposal_path),
        ),
    )
    for name in ("source_allowlist.json", "reference-tokenizer.json", "review-samples.jsonl"):
        if (metadata_source / name).exists():
            shutil.copyfile(metadata_source / name, root / name)
    result = dict(
        manifest,
        format=HOLDOUT_FORMAT
        if inherited.get("test_groups")
        else (TASK_FORMAT if inherited.get("task_policy") else FORMAT),
        base_corpus=dict(
            path=os.path.relpath(source, root),
            manifest_sha256=sha256(source / "corpus-manifest.json"),
        ),
        partition_file=partition_path.name,
        partition_sha256=sha256(partition_path),
        splits=totals,
        split_rule=manifest["split_rule"]
        + "; minimal additional whole validation groups; prior test/val unchanged",
        minimum_validation_group_fraction=fraction,
        source_splits=statistics,
        **({"task_policy": inherited["task_policy"]} if inherited.get("task_policy") else {}),
        **(
            dict(previous_effective_manifest_sha256=proposal["prior_partition_manifest_sha256"])
            if inherited
            else {}
        ),
    )
    write_json(root / "corpus-manifest.json", result)
    audit = dict(
        audit,
        operation="apply_validation_group_reservation",
        base_source_audit_sha256=sha256(metadata_source / "source-audit.json"),
        reservation_sha256=sha256(proposal_path),
        reservation_applied=True,
        split_reference_tokens=tokens,
        corpus=result,
        partition_processor_sha256=sha256(__file__),
        source_data_changed=False,
        formal_admission=False,
        updated_unix=time.time(),
    )
    audit.update(media_statistics)
    if inherited:
        audit.update(
            prior_holdout_membership_sha256=hashlib.sha256(
                "".join(
                    f"{identity}:{split}\n" for identity, split in prior_holdouts.items()
                ).encode()
            ).hexdigest(),
            holdout_membership_sha256=hashlib.sha256(
                "".join(
                    f"{identity}:{split}\n" for identity, split in current_holdouts.items()
                ).encode()
            ).hexdigest(),
        )
    write_json(root / "source-audit.json", audit)
    return result


def create_media_exclusion_view(
    corpus_root, group_audit, inventory, output, *, resolve_validation_conflicts=False
):
    """Exclude train conflicts; optionally apply test precedence from closed media grouping."""
    return _create_group_exclusion_view(
        corpus_root, group_audit, inventory, output, "media", resolve_validation_conflicts
    )


def create_text_exclusion_view(
    corpus_root, group_audit, inventory, output, *, resolve_validation_conflicts=False
):
    """Exclude train conflicts; optionally apply audited test-over-val precedence."""
    return _create_group_exclusion_view(
        corpus_root, group_audit, inventory, output, "text", resolve_validation_conflicts
    )


def create_quality_exclusion_view(corpus_root, quality_review, inventory, output):
    """Remove reviewed defective training groups without changing any held-out record."""
    return _create_group_exclusion_view(corpus_root, quality_review, inventory, output, "quality")


def _create_group_exclusion_view(
    corpus_root, group_audit, inventory, output, kind, resolve_validation_conflicts=False
):
    from minifrontier.data.minifrontier1 import write_json

    source, report_path, root = (
        Path(corpus_root).resolve(),
        Path(group_audit).resolve(),
        Path(output).resolve(),
    )
    if root.exists():
        raise FileExistsError("group exclusion views require a new output")
    report = json.loads(report_path.read_text())
    manifest = json.loads((source / "corpus-manifest.json").read_text())
    audit = json.loads((source / "source-audit.json").read_text())
    binding = report["inputs"][inventory]
    quality = kind == "quality"
    evidence_key = "quality_review_sha256" if quality else "grouping_audit_sha256"
    if quality:
        kind = report.get("corpus_kind")
        if (
            kind not in {"text", "media"}
            or report.get("status") != "targeted_defects_confirmed"
            or report.get("review_method") not in {"human", "model_assisted"}
            or binding.get("source_audit_sha256") != sha256(source / "source-audit.json")
        ):
            raise ValueError("quality exclusions require a bound, explicit defect review")
    complete_grouping = (
        report.get("full_shared_text_cross_split_audit_complete") is True
        and report.get("heldout_self_join_complete") is True
        if kind == "text"
        else report.get("status") == "split_conflicts_require_partition_update"
        and report.get("unresolved_rendered_visual_candidates") == []
    )
    if type(resolve_validation_conflicts) is not bool or (
        resolve_validation_conflicts and not complete_grouping
    ):
        raise ValueError("validation conflicts require an explicit, complete grouping audit")
    if (
        report["kind"]
        != ("source_quality_exclusion_review" if quality else f"cross_corpus_{kind}_group_audit")
        or binding["corpus_manifest_sha256"] != sha256(source / "corpus-manifest.json")
        or binding["database_sha256"] != manifest["database_sha256"]
        or audit.get("formal_admission")
        or audit["status"]
        not in {"candidate_slice_complete_pending_admission", "candidate_inventory_below_target"}
    ):
        raise ValueError("exclusion evidence and unadmitted source differ")
    selected, promotions, test_anchors = {}, {}, {}
    if quality:
        for defect in report["excluded_training_groups"]:
            group = defect.get("group")
            count = defect.get("records")
            reason = defect.get("reason")
            if (
                not isinstance(group, str)
                or not group
                or group in selected
                or type(count) is not int
                or count <= 0
                or not isinstance(reason, str)
                or not reason.strip()
            ):
                raise ValueError("quality exclusions require distinct groups, counts and reasons")
            selected[group] = count
    for component in [] if quality else report["split_conflicts"]:
        if component["required_split"] not in {"val", "test"}:
            raise ValueError("exclusion has no held-out component")
        for member in component["members"]:
            if member["inventory"] == inventory and member["split"] == "train":
                selected[member["group"]] = member["records"]
            if (
                resolve_validation_conflicts
                and component["required_split"] == "test"
                and member["inventory"] == inventory
                and member["split"] == "val"
            ):
                anchors = {
                    m["group"]: m["records"]
                    for m in component["members"]
                    if m["inventory"] == inventory and m["split"] == "test"
                }
                external_anchors = [
                    m
                    for m in component["members"]
                    if m["inventory"] != inventory and m["split"] == "test"
                ]
                for anchor in external_anchors:
                    evidence = report["inputs"].get(anchor["inventory"], {})
                    if kind != "media" or any(
                        not evidence.get(key)
                        for key in ("sha256", "corpus_manifest_sha256", "database_sha256")
                    ):
                        raise ValueError("external test anchor lacks bound media identity evidence")
                if not anchors and not external_anchors:
                    raise ValueError("validation promotion has no test member in its component")
                test_anchors.update(anchors)
                promotions[member["group"]] = member["records"]
    if not selected and not promotions:
        raise ValueError("no groups require exclusion or held-out promotion for this inventory")
    overrides: dict[str, Any] = dict(groups={}, excluded_groups=[])
    base = source
    if manifest.get("format") in VIEW_FORMATS:
        base = (source / manifest["base_corpus"]["path"]).resolve()
        overrides = json.loads((source / manifest["partition_file"]).read_text())

    def holdout_hash(db, promote=()):
        digest = hashlib.sha256()
        for identity, split, group in db.execute(
            "SELECT id,split,group_root FROM samples WHERE split!='train' ORDER BY id"
        ):
            if group in promote:
                split = "test"
            digest.update((identity + ":" + split + "\n").encode())
        return digest.hexdigest()

    with contextlib.closing(open_corpus(source)) as db:
        if sha256(corpus_storage_root(db) / "corpus.sqlite") != binding["database_sha256"]:
            raise ValueError("media exclusion source database changed")
        prior_holdouts = holdout_hash(db)
        expected_holdouts = holdout_hash(db, promotions)
        checked_groups = set(selected) | set(promotions) | set(test_anchors)
        actual_membership: dict[str, list[tuple[str, int]]] = {}
        # A partition view need not have a group index. Count all groups once;
        # one filtered query per conflict would rescan the corpus repeatedly.
        for group, split, count in db.execute(
            "SELECT group_root,split,COUNT(*) FROM samples GROUP BY group_root,split"
        ):
            if group in checked_groups:
                actual_membership.setdefault(group, []).append((split, count))
        for expected, changed in (("train", selected), ("val", promotions), ("test", test_anchors)):
            for group, count in changed.items():
                if actual_membership.get(group) != [(expected, count)]:
                    raise ValueError("conflicting group membership differs from the audit")
    exclusions = sorted(set(overrides.get("excluded_groups", [])) | set(selected))
    test_groups = sorted(set(overrides.get("test_groups", [])) | set(promotions))
    statistics: dict[str, dict[str, Any]] = {}
    totals: dict[str, int] = {}
    tokens: dict[str, dict[str, int]] = {}
    images: dict[str, set[str]] = {}
    answers: dict[str, dict[str, int]] = {}
    review = []
    review_counts: dict[str, int] = {}
    with contextlib.closing(
        sqlite3.connect((base / "corpus.sqlite").as_uri() + "?mode=ro", uri=True)
    ) as db:
        _reserve(db, overrides["groups"], exclusions, overrides.get("task_policy"), test_groups)
        if holdout_hash(db) != expected_holdouts:
            raise ValueError("refinement changed holdouts outside the audited val-to-test groups")
        for name, split, records, groups, count in db.execute(
            "SELECT source,split,COUNT(*),COUNT(DISTINCT group_root),SUM(json_extract(payload,'$.reference_tokens')) FROM samples GROUP BY source,split"
        ):
            statistics.setdefault(name, {})[split] = dict(
                records=records, groups=groups, reference_tokens=count
            )
            totals[split] = totals.get(split, 0) + records
            tokens.setdefault(split, {})[name] = count
        for split, payload in db.execute("SELECT split,payload FROM samples ORDER BY id"):
            row = json.loads(payload)
            if kind == "media":
                images.setdefault(split, set()).update(m["rgb_sha256"] for m in row["media"])
                answers.setdefault(split, {})[row["task"]] = (
                    answers.setdefault(split, {}).get(row["task"], 0)
                    + row["answer_reference_tokens"]
                )
            key = row["source"] + ":" + row["task"]
            if split == "train" and review_counts.get(key, 0) < 100:
                review.append(dict(split=split, record=row))
                review_counts[key] = review_counts.get(key, 0) + 1
        split_groups = dict(
            db.execute("SELECT split,COUNT(DISTINCT group_root) FROM samples GROUP BY split")
        )
    require_space(root, 16 * 1024**2, reserve_bytes=80 * GIB)
    root.mkdir(parents=True)
    write_json(
        root / "split-overrides.json",
        dict(
            groups=overrides["groups"],
            excluded_groups=exclusions,
            **{evidence_key: sha256(report_path)},
            **(
                {"grouping_audit_sha256": overrides["grouping_audit_sha256"]}
                if quality and "grouping_audit_sha256" in overrides
                else {}
            ),
            **({"task_policy": overrides["task_policy"]} if "task_policy" in overrides else {}),
            **({"test_groups": test_groups} if test_groups else {}),
        ),
    )
    for name in ("source_allowlist.json", "reference-tokenizer.json"):
        if (source / name).exists():
            shutil.copyfile(source / name, root / name)
    review_path = root / "review-samples.jsonl"
    review_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in review))
    previous_review = {
        json.loads(line)["record"]["sample_id"]
        for line in (source / "review-samples.jsonl").read_text().splitlines()
    }
    current_review = {r["record"]["sample_id"] for r in review}
    result = dict(
        manifest,
        format=HOLDOUT_FORMAT
        if test_groups
        else (TASK_FORMAT if "task_policy" in overrides else FORMAT),
        base_corpus=dict(
            path=os.path.relpath(base, root), manifest_sha256=sha256(base / "corpus-manifest.json")
        ),
        partition_file="split-overrides.json",
        partition_sha256=sha256(root / "split-overrides.json"),
        splits=totals,
        source_splits=statistics,
        formal_admission=False,
        split_rule=manifest["split_rule"]
        + (
            "; exclude entire training groups with confirmed source-quality defects"
            if quality
            else f"; exclude entire train groups linked to held-out {kind} across corpora"
        )
        + (
            "; audited whole validation groups move to test; all original holdouts remain held out"
            if promotions
            else ""
        ),
        excluded_training_groups=len(exclusions),
        newly_excluded_records=sum(selected.values()),
        **{evidence_key: sha256(report_path)},
        previous_effective_manifest_sha256=sha256(source / "corpus-manifest.json"),
        **(
            dict(
                validation_groups_promoted_to_test=len(promotions),
                promoted_validation_records=sum(promotions.values()),
            )
            if promotions
            else {}
        ),
    )
    write_json(root / "corpus-manifest.json", result)
    write_json(
        root / "source-audit.json",
        dict(
            audit,
            operation="exclude_source_quality_train_groups"
            if quality
            else (
                f"resolve_{kind}_split_conflicts"
                if promotions
                else "exclude_cross_pool_train_groups"
            ),
            corpus=result,
            source_data_changed=False,
            formal_admission=False,
            main_budget_eligible=False,
            base_source_audit_sha256=sha256(source / "source-audit.json"),
            split_reference_tokens=tokens,
            split_independent_images={k: len(v) for k, v in images.items()},
            split_independent_groups=split_groups,
            split_answer_reference_tokens=answers,
            **{evidence_key: sha256(report_path)},
            holdout_membership_sha256=expected_holdouts,
            prior_holdout_membership_sha256=prior_holdouts,
            validation_groups_promoted_to_test=promotions,
            excluded_training_groups=selected,
            updated_unix=time.time(),
            review=dict(
                status="awaiting_manual_review",
                sha256=sha256(review_path),
                samples=review_counts,
                prior_sha256=sha256(source / "review-samples.jsonl"),
                removed_ids=sorted(previous_review - current_review),
                added_ids=sorted(current_review - previous_review),
            ),
        ),
    )
    return result


def create_task_classification_view(corpus_root, output):
    """Correct source task metadata while preserving every QA, pixel and split identity."""
    from collections import Counter

    from minifrontier.data.media_tasks import task_policy
    from minifrontier.data.minifrontier1 import write_json

    source, root = Path(corpus_root).resolve(), Path(output).resolve()
    if root.exists():
        raise FileExistsError("task classification requires a new immutable view")
    manifest = json.loads((source / "corpus-manifest.json").read_text())
    audit = json.loads((source / "source-audit.json").read_text())
    if audit.get("formal_admission") or audit["status"] not in {
        "candidate_slice_complete_pending_admission",
        "candidate_inventory_below_target",
    }:
        raise ValueError("task correction requires a completed, unadmitted corpus")
    base = source
    overrides: dict[str, Any] = dict(groups={}, excluded_groups=[])
    if manifest.get("format") in VIEW_FORMATS:
        base = (source / manifest["base_corpus"]["path"]).resolve()
        overrides = json.loads((source / manifest["partition_file"]).read_text())
    if "task_policy" in overrides:
        raise ValueError("this corpus already binds a task policy")
    policy = task_policy()
    counts: dict[str, Counter[str]] = {}
    changes: dict[str, Counter[str]] = {}
    answers: dict[str, Counter[str]] = {}
    reviews = []
    review_counts: Counter[str] = Counter()
    identity_hash, content_hash = hashlib.sha256(), hashlib.sha256()
    with (
        contextlib.closing(open_corpus(source)) as before,
        contextlib.closing(
            sqlite3.connect((base / "corpus.sqlite").as_uri() + "?mode=ro", uri=True)
        ) as after,
    ):
        if sha256(base / "corpus.sqlite") != manifest["database_sha256"]:
            raise ValueError("task correction source database changed")
        _reserve(
            after,
            overrides["groups"],
            overrides.get("excluded_groups", []),
            policy,
            overrides.get("test_groups", []),
        )
        query = "SELECT id,stage,split,group_root,task,payload FROM samples ORDER BY id"
        for old, new in zip(before.execute(query), after.execute(query), strict=True):
            if old[:4] != new[:4]:
                raise ValueError("task correction changed a sample, stage, split or group")
            old_row, row = json.loads(old[5]), json.loads(new[5])
            old_task = old_row.pop("task")
            new_task = row.pop("task")
            if old_row != row or old[4] != old_task or new[4] != new_task:
                raise ValueError("task correction changed content outside the domain label")
            identity_hash.update(json.dumps(old[:4]).encode())
            content_hash.update(json.dumps(row, sort_keys=True).encode())
            split = old[2]
            counts.setdefault(split, Counter())[new_task] += 1
            answers.setdefault(split, Counter())[new_task] += row.get("answer_reference_tokens", 0)
            if old_task != new_task:
                changes.setdefault(split, Counter())[old_task + "->" + new_task] += 1
            row["task"] = new_task
            key = row["source"] + ":" + new_task
            if split == "train" and review_counts[key] < 100:
                reviews.append(dict(split=split, record=row))
                review_counts[key] += 1
    if not changes:
        raise ValueError("the source corpus has no task labels requiring correction")
    require_space(root, 16 * 1024**2, reserve_bytes=80 * GIB)
    root.mkdir(parents=True)
    write_json(root / "split-overrides.json", dict(overrides, task_policy=policy))
    for name in ("source_allowlist.json", "reference-tokenizer.json"):
        if (source / name).exists():
            shutil.copyfile(source / name, root / name)
    review_path = root / "review-samples.jsonl"
    review_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in reviews))
    previous_review = {
        json.loads(line)["record"]["sample_id"]
        for line in (source / "review-samples.jsonl").read_text().splitlines()
    }
    current_review = {r["record"]["sample_id"] for r in reviews}
    result = dict(
        manifest,
        format=HOLDOUT_FORMAT if overrides.get("test_groups") else TASK_FORMAT,
        base_corpus=dict(
            path=os.path.relpath(base, root), manifest_sha256=sha256(base / "corpus-manifest.json")
        ),
        partition_file="split-overrides.json",
        partition_sha256=sha256(root / "split-overrides.json"),
        previous_effective_manifest_sha256=sha256(source / "corpus-manifest.json"),
        task_policy=policy,
        formal_admission=False,
    )
    write_json(root / "corpus-manifest.json", result)
    write_json(
        root / "source-audit.json",
        dict(
            audit,
            operation="correct_source_task_classification",
            corpus=result,
            task_policy=policy,
            task_changes=changes,
            effective_domain_records=counts,
            split_answer_reference_tokens=answers,
            sample_stage_split_group_sha256=identity_hash.hexdigest(),
            all_content_except_task_sha256=content_hash.hexdigest(),
            base_source_audit_sha256=sha256(source / "source-audit.json"),
            original_database_and_pixels_unchanged=True,
            formal_admission=False,
            main_budget_eligible=False,
            updated_unix=time.time(),
            review=dict(
                status="awaiting_manual_review",
                sha256=sha256(review_path),
                samples=review_counts,
                prior_sha256=sha256(source / "review-samples.jsonl"),
                removed_ids=sorted(previous_review - current_review),
                added_ids=sorted(current_review - previous_review),
            ),
        ),
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", required=True)
    parser.add_argument("--reservation", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(create_partition_view(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
