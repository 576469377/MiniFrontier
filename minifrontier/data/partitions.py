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


def _reserve(db, groups, excluded_groups=()):
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
    db.execute("""CREATE TEMP VIEW samples AS
        SELECT s.id,s.stage,s.source,s.task,s.first_question,s.text,s.payload,
               s.simhash,s.group_root,
               CASE WHEN v.group_root IS NOT NULL THEN 'val' ELSE s.split END AS split
        FROM main.samples s LEFT JOIN extra_validation v ON s.group_root=v.group_root
        LEFT JOIN excluded_training_groups x ON s.group_root=x.group_root
        WHERE x.group_root IS NULL
    """)


def open_corpus(root):
    """Open the effective partition on the calling thread; caller closes it."""
    root = Path(root).resolve()
    path = root / "corpus-manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {}
    if manifest.get("format") != FORMAT:
        return sqlite3.connect((root / "corpus.sqlite").as_uri() + "?mode=ro", uri=True)
    base = (root / manifest["base_corpus"]["path"]).resolve()
    if sha256(base / "corpus-manifest.json") != manifest["base_corpus"]["manifest_sha256"]:
        raise ValueError("partition base manifest changed")
    original = json.loads((base / "corpus-manifest.json").read_text())
    if original.get("format") == FORMAT:
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
        _reserve(db, overrides["groups"], overrides.get("excluded_groups", []))
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
        manifest.get("format") == FORMAT
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
    statistics: dict[str, dict[str, Any]] = {}
    totals: dict[str, int] = {}
    tokens: dict[str, dict[str, int]] = {}
    media_statistics = {}
    with contextlib.closing(open_corpus(source)) as db:
        _reserve(db, proposal["groups"])
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
        if (source / "review-samples.jsonl").exists():
            with (source / "review-samples.jsonl").open() as review:
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
        dict(schema_version=1, groups=proposal["groups"], reservation_sha256=sha256(proposal_path)),
    )
    for name in ("source_allowlist.json", "reference-tokenizer.json", "review-samples.jsonl"):
        if (source / name).exists():
            shutil.copyfile(source / name, root / name)
    result = dict(
        manifest,
        format=FORMAT,
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
    )
    write_json(root / "corpus-manifest.json", result)
    audit = dict(
        audit,
        operation="apply_validation_group_reservation",
        base_source_audit_sha256=sha256(source / "source-audit.json"),
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
    write_json(root / "source-audit.json", audit)
    return result


def create_media_exclusion_view(corpus_root, group_audit, inventory, output):
    """Quarantine train groups linked to holds; preserve original val/test and raw data."""
    from minifrontier.data.minifrontier1 import write_json

    source, report_path, root = (
        Path(corpus_root).resolve(),
        Path(group_audit).resolve(),
        Path(output).resolve(),
    )
    if root.exists():
        raise FileExistsError("media exclusion views require a new output")
    report = json.loads(report_path.read_text())
    manifest = json.loads((source / "corpus-manifest.json").read_text())
    audit = json.loads((source / "source-audit.json").read_text())
    binding = report["inputs"][inventory]
    if (
        report["kind"] != "cross_corpus_media_group_audit"
        or binding["corpus_manifest_sha256"] != sha256(source / "corpus-manifest.json")
        or binding["database_sha256"] != manifest["database_sha256"]
        or audit.get("formal_admission")
        or audit["status"]
        not in {"candidate_slice_complete_pending_admission", "candidate_inventory_below_target"}
    ):
        raise ValueError("media grouping evidence and unadmitted source differ")
    selected = {}
    for component in report["split_conflicts"]:
        if component["required_split"] not in {"val", "test"}:
            raise ValueError("exclusion has no held-out component")
        for member in component["members"]:
            if member["inventory"] == inventory and member["split"] == "train":
                selected[member["group"]] = member["records"]
    if not selected:
        raise ValueError("no train groups require exclusion for this inventory")
    overrides: dict[str, Any] = dict(groups={}, excluded_groups=[])
    base = source
    if manifest.get("format") == FORMAT:
        base = (source / manifest["base_corpus"]["path"]).resolve()
        overrides = json.loads((source / manifest["partition_file"]).read_text())

    def holdout_hash(db):
        digest = hashlib.sha256()
        for identity, split in db.execute(
            "SELECT id,split FROM samples WHERE split!='train' ORDER BY id"
        ):
            digest.update((identity + ":" + split + "\n").encode())
        return digest.hexdigest()

    with contextlib.closing(open_corpus(source)) as db:
        if sha256(corpus_storage_root(db) / "corpus.sqlite") != binding["database_sha256"]:
            raise ValueError("media exclusion source database changed")
        prior_holdouts = holdout_hash(db)
        for group, count in selected.items():
            actual = list(
                db.execute(
                    "SELECT split,COUNT(*) FROM samples WHERE group_root=? GROUP BY split", (group,)
                )
            )
            if actual != [("train", count)]:
                raise ValueError("conflicting group membership differs from the audit")
    exclusions = sorted(set(overrides.get("excluded_groups", [])) | set(selected))
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
        _reserve(db, overrides["groups"], exclusions)
        if holdout_hash(db) != prior_holdouts:
            raise ValueError("exclusion changed an existing validation/test record")
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
            images.setdefault(split, set()).update(m["rgb_sha256"] for m in row["media"])
            answers.setdefault(split, {})[row["task"]] = (
                answers.setdefault(split, {}).get(row["task"], 0) + row["answer_reference_tokens"]
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
            grouping_audit_sha256=sha256(report_path),
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
        format=FORMAT,
        base_corpus=dict(
            path=os.path.relpath(base, root), manifest_sha256=sha256(base / "corpus-manifest.json")
        ),
        partition_file="split-overrides.json",
        partition_sha256=sha256(root / "split-overrides.json"),
        splits=totals,
        source_splits=statistics,
        formal_admission=False,
        split_rule=manifest["split_rule"]
        + "; exclude entire train groups linked to held-out media across corpora",
        excluded_training_groups=len(exclusions),
        newly_excluded_records=sum(selected.values()),
        grouping_audit_sha256=sha256(report_path),
        previous_effective_manifest_sha256=sha256(source / "corpus-manifest.json"),
    )
    write_json(root / "corpus-manifest.json", result)
    write_json(
        root / "source-audit.json",
        dict(
            audit,
            operation="exclude_cross_pool_train_groups",
            corpus=result,
            source_data_changed=False,
            formal_admission=False,
            main_budget_eligible=False,
            base_source_audit_sha256=sha256(source / "source-audit.json"),
            split_reference_tokens=tokens,
            split_independent_images={k: len(v) for k, v in images.items()},
            split_independent_groups=split_groups,
            split_answer_reference_tokens=answers,
            grouping_audit_sha256=sha256(report_path),
            holdout_membership_sha256=prior_holdouts,
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", required=True)
    parser.add_argument("--reservation", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(create_partition_view(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
