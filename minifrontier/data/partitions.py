"""Read-only corpus partitions over one immutable database.

An explicit manifest applies additional validation groups to every consumer.
The view has no corpus.sqlite alias: readers which ignore the partition contract
fail instead of silently training on the newly reserved validation records.
"""

import argparse
import contextlib
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


def _reserve(db, groups):
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
    db.execute("""CREATE TEMP VIEW samples AS
        SELECT s.id,s.stage,s.source,s.task,s.first_question,s.text,s.payload,
               s.simhash,s.group_root,
               CASE WHEN v.group_root IS NOT NULL THEN 'val' ELSE s.split END AS split
        FROM main.samples s LEFT JOIN extra_validation v ON s.group_root=v.group_root
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
        _reserve(db, json.loads(partition.read_text())["groups"])
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", required=True)
    parser.add_argument("--reservation", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(create_partition_view(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
