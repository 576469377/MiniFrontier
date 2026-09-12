"""Construct a bounded candidate text slice; admission is a separate, evidence-bound step.

Reuse the corpus deduplicator and pinned public Parquet reader. This command
never trains models, downloads whole datasets, or promotes its output to formal
data. Token counts use an explicitly named reference tokenizer, not a byte guess.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import re
import shutil
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urldefrag

from tokenizers import Tokenizer

from minifrontier.data import normalized, sha256
from minifrontier.data.code_sources import CODE_SOURCE, code_rows, licensed_record
from minifrontier.data.corpus import CorpusBuilder
from minifrontier.data.minifrontier1 import write_json
from minifrontier.data.public_sources import SOURCES, normalized_source, source_rows
from minifrontier.storage import GIB, require_space, reserve_write

TEXT_SOURCES: dict[str, dict[str, Any]] = {
    k: dict(SOURCES[k]) for k in ("zh_edu", "en_edu", "ultrachat")
}
TEXT_SOURCES["code_licensed"] = CODE_SOURCE
TEXT_SOURCES["openwebmath"] = dict(
    repo="open-web-math/open-web-math",
    revision="fde8ef8de2300f5e778f56261843dab89f230815",
    prefix="data",
    license="ODC-BY-1.0; Common Crawl terms; underlying page rights retained",
    task="verified_math_science",
    lang="en",
)
DEFAULT_TARGETS = dict(
    zh_edu=225_000_000, en_edu=150_000_000, openwebmath=50_000_000, ultrachat=40_000_000
)


def clean_record(name, row, identity):
    """Return an auditable normalized record, or a concrete rejection reason."""
    if name == "code_licensed":
        return licensed_record(row, identity)
    spec = TEXT_SOURCES[name]
    if name == "openwebmath":
        if not row.get("url") or not row.get("text"):
            return None, "missing_origin_or_text"
        record = dict(
            source=spec["repo"],
            revision=spec["revision"],
            item_id=identity,
            group_id=row["url"],
            license=spec["license"],
            lang="en",
            task=spec["task"],
            stage="pretrain",
            text=row["text"],
            source_metadata={k: row[k] for k in ("url", "date", "metadata") if k in row},
        )
    else:
        if name == "zh_edu":
            score = row.get("score")
            if (
                not isinstance(score, (int, float))
                or not math.isfinite(score)
                or not 0 <= score <= 1
            ):
                return None, "score_outside_declared_0_1_range"
        record = normalized_source(name, row, identity)
        if record is None:
            return None, "missing_payload"
        if name == "ultrachat":
            if any(
                t.get("role") not in {"system", "user", "assistant"}
                or not isinstance(t.get("content"), str)
                for t in record["turns"]
            ):
                return None, "invalid_dialogue"
            record["text"] = "\n\n".join(
                t["role"].capitalize() + ": " + t["content"] for t in record.pop("turns")
            )
            record["stage"] = "pretrain"
    raw = record["text"]
    text = normalized(raw)
    if not 100 <= len(text) <= 100_000:
        return None, "length_outside_100_100000_characters"
    if "\ufffd" in text or any(ord(c) < 32 and c not in "\n\t\r" for c in text):
        return None, "invalid_encoding_or_controls"
    if len(re.findall(r"<\s*(?:script|style|div|span|html|body|nav)\b", text, re.I)) > 3:
        return None, "unremoved_html"
    letters = sum(c.isalpha() for c in text)
    if letters < 30:
        return None, "insufficient_language_content"
    chinese = sum("\u3400" <= c <= "\u9fff" for c in text)
    latin = sum(c.isascii() and c.isalpha() for c in text)
    if (spec["lang"] == "zh" and chinese / letters < 0.4) or (
        spec["lang"] == "en" and latin / letters < 0.7
    ):
        return None, "language_script_mismatch"
    if name == "zh_edu" and not row.get("source"):
        return None, "missing_chinese_subsource"
    url = record.get("source_metadata", {}).get("url")
    if url:
        record["document_id"] = urldefrag(url)[0]
    record.update(
        text=text,
        raw_text_sha256=hashlib.sha256(raw.encode()).hexdigest(),
        normalized_text_sha256=hashlib.sha256(text.encode()).hexdigest(),
        quality_flags=["candidate-not-formally-admitted"],
    )
    return record, None


def build_text_slice(
    output, reference_tokenizer, *, targets=None, seed=20260910, max_gib=12, resume=False
):
    root, reference = Path(output).resolve(), Path(reference_tokenizer).resolve()
    targets = DEFAULT_TARGETS if targets is None else targets
    if not targets or any(k not in TEXT_SOURCES or n <= 0 for k, n in targets.items()):
        raise ValueError(
            "choose positive budgets for reviewed candidate sources; unverified code stays quarantined"
        )
    previous = json.loads((root / "source-audit.json").read_text()) if resume else None
    if root.exists() and not resume:
        raise FileExistsError("choose a new immutable candidate slice")
    if previous and (
        set(targets) != {"code_licensed"}
        or previous["status"] != "interrupted_unadmitted"
        or previous.get("formal_admission")
        or previous["seed"] != seed
        or previous["target_candidate_tokens"] != targets
        or previous["reference_tokenizer_sha256"] != sha256(reference)
        or previous["max_gib"] != max_gib
        or previous["sources"]["code_licensed"]["specification"] != CODE_SOURCE
    ):
        raise ValueError("resume requires the same interrupted, unadmitted code inventory")
    require_space(root, int(max_gib * GIB), reserve_bytes=80 * GIB)
    builder = CorpusBuilder(root, seed=seed, max_gib=max_gib, val_buckets=50, test_buckets=100)
    tokenizer = Tokenizer.from_file(str(reference))
    shutil.copyfile(reference, root / "reference-tokenizer.json")
    audit = dict(
        schema_version=1,
        status="building",
        formal_admission=False,
        seed=seed,
        reference_tokenizer_sha256=sha256(reference),
        reference_is_final_training_tokenizer=False,
        target_candidate_tokens=targets,
        max_gib=max_gib,
        minimum_free_gib_for_new_allocation=80,
        processor_files={
            str(p.relative_to(Path(__file__).resolve().parents[2])): sha256(p)
            for p in (
                Path(__file__),
                Path(__file__).with_name("corpus.py"),
                Path(__file__).with_name("public_sources.py"),
                Path(__file__).with_name("code_sources.py"),
                Path(__file__).with_name("remote.py"),
            )
        },
        sources={},
        unresolved=[
            "code repository/blob/license evidence",
            "benchmark identity exclusion and sealed evaluation inventory",
            "stratified manual review of at least 100 records per major source",
            "final tokenizer quality and per-model stage mixture admission",
            "image/OCR/chart/video sources and independent held-out media",
        ],
    )
    if previous:
        if not previous.get("error", "").startswith(
            (
                "ValueError: code shard exceeds bounded",
                "ReadTimeout",
                "ConnectionError",
                "HTTPError",
            )
        ):
            builder.db.close()
            raise ValueError("only verified source-read interruptions can resume automatically")
        rows = builder.db.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
        if rows != previous["dedup_and_quality_counts"].get("accepted", 0) or rows != sum(
            source["accepted_records"] for source in previous["sources"].values()
        ):
            builder.db.close()
            raise ValueError("interrupted code audit and retained records disagree")
        write_json(
            root / f"source-audit-resume-{len(previous.get('resumes', [])) + 1}.json", previous
        )
        current_processors = audit["processor_files"]
        audit = previous
        audit.setdefault("resumes", []).append(
            dict(
                previous_error=audit.pop("error", None),
                retained_records=rows,
                previous_processor_files=audit["processor_files"],
                database_sha256=sha256(root / "corpus.sqlite"),
                resumed_unix=time.time(),
            )
        )
        audit.update(status="building", processor_files=current_processors)
        builder.counts = Counter(audit["dedup_and_quality_counts"])
        estimate = builder.db.execute(
            "SELECT COALESCE(SUM(4*length(CAST(payload AS BLOB))+1024),0) FROM samples"
        ).fetchone()[0]
        builder.approximate_bytes = max(builder.approximate_bytes, estimate + 16 * 1024**2)
    path = root / "source-audit.json"

    def progress():
        builder.db.commit()
        audit["updated_unix"] = time.time()
        audit["free_gib"] = shutil.disk_usage(root).free / GIB
        audit["database_bytes"] = (root / "corpus.sqlite").stat().st_size
        audit["dedup_and_quality_counts"] = dict(builder.counts)
        write_json(path, audit)

    write_json(
        root / "source_allowlist.json",
        dict(
            status="candidate_extraction_only",
            formal_admission=False,
            sources={k: TEXT_SOURCES[k] for k in targets},
            quarantined={
                "python_edu": "original repository/license mapping unresolved; old rows remain excluded"
            },
        ),
    )
    try:
        progress()
        for source_index, (name, target) in enumerate(targets.items()):
            entry: dict[str, Any] = dict(
                specification=TEXT_SOURCES[name],
                accepted_records=0,
                accepted_reference_tokens=0,
                rejected=Counter(),
                rejection_examples={},
                status="reading",
            )
            if previous:
                entry = audit["sources"][name]
                entry["rejected"] = Counter(entry["rejected"])
                entry["status"] = "reading"
            audit["sources"][name] = entry
            progress()
            reader = (
                code_rows(root=root, seed=seed, audit=entry, resume=resume)
                if name == "code_licensed"
                else source_rows(
                    name,
                    seed=seed + source_index * 104729,
                    audit=entry,
                    specification=TEXT_SOURCES[name],
                )
            )
            with contextlib.closing(reader):
                for row, identity in reader:
                    record, reason = clean_record(name, row, identity)
                    if record is None:
                        entry["rejected"][reason] += 1
                        examples = entry["rejection_examples"].setdefault(reason, [])
                        if len(examples) < 20:
                            examples.append(
                                dict(
                                    identity=identity,
                                    score=str(row.get("score")),
                                    subsource=row.get("source"),
                                )
                            )
                        continue
                    ids = tokenizer.encode(record["text"], add_special_tokens=False).ids
                    record["reference_tokens"] = len(ids)
                    if not builder.add(record):
                        continue
                    entry["accepted_records"] += 1
                    entry["accepted_reference_tokens"] += len(ids)
                    if entry["accepted_records"] % 1000 == 0:
                        # Stop ingestion before the reserved checkpoint headroom is spent.
                        require_space(root, 64 * 1024**2, reserve_bytes=80 * GIB)
                        progress()
                        print(
                            json.dumps(
                                dict(
                                    source=name,
                                    records=entry["accepted_records"],
                                    tokens=entry["accepted_reference_tokens"],
                                )
                            ),
                            flush=True,
                        )
                    if entry["accepted_reference_tokens"] >= target:
                        break
            entry["status"] = (
                "candidate_target_reached"
                if entry["accepted_reference_tokens"] >= target
                else "source_exhausted_below_target"
            )
            progress()
        audit["corpus"] = builder.finalize()
        counts: dict[str, Counter] = {}
        review = root / "review-samples.jsonl"
        selected: Counter[str] = Counter()
        with review.open("w") as handle:
            for split, source, payload in builder.db.execute(
                "SELECT split,source,payload FROM samples ORDER BY id"
            ):
                row = json.loads(payload)
                counts.setdefault(split, Counter())[source] += row["reference_tokens"]
                # Hash order provides a stable content-based sample; sealed test stays sealed.
                if split == "train" and selected[source] < 100:
                    handle.write(
                        json.dumps(dict(split=split, record=row), ensure_ascii=False) + "\n"
                    )
                    selected[source] += 1
        audit["split_reference_tokens"] = counts
        audit["review"] = dict(
            status="awaiting_manual_review", samples=dict(selected), sha256=sha256(review)
        )
        audit["status"] = (
            "candidate_slice_complete_pending_admission"
            if all(s["status"] == "candidate_target_reached" for s in audit["sources"].values())
            else "candidate_inventory_below_target"
        )
        progress()
    except BaseException as error:
        audit.update(
            status="interrupted_unadmitted", error=type(error).__name__ + ": " + str(error)
        )
        progress()
        raise
    finally:
        builder.db.close()
    return audit


def _prior_partition_controls(path, base, locks):
    """Read effective holdouts without dropping excluded rows before deduplication."""
    from minifrontier.data.partitions import VIEW_FORMATS, corpus_storage_root, open_corpus

    root = Path(path).resolve()
    manifest = json.loads((root / "corpus-manifest.json").read_text())
    audit = json.loads((root / "source-audit.json").read_text())
    if manifest.get("format") not in VIEW_FORMATS:
        raise ValueError("prior partition must be an explicit partition view")
    if audit.get("formal_admission") or audit["status"] != (
        "candidate_slice_complete_pending_admission"
    ):
        raise ValueError("prior partition must be a completed unadmitted candidate")
    with contextlib.closing(open_corpus(root)) as db:
        if corpus_storage_root(db) != base:
            raise ValueError("prior partition must refer to the first merge input")
        overlay_path = root / manifest["partition_file"]
        overlay = json.loads(overlay_path.read_text())
        if overlay.get("task_policy") is not None:
            raise ValueError("text merge cannot inherit a media task classification policy")
        splits = dict(db.execute("SELECT id,split FROM samples"))
        excluded = {
            row[0]
            for row in db.execute("SELECT id FROM main.samples EXCEPT SELECT id FROM temp.samples")
        }
        for key, split in db.execute(
            "SELECT key,split FROM links JOIN samples USING(id) WHERE split IN ('val','test')"
        ):
            if locks.get(key) != "test":
                locks[key] = split
    binding = dict(
        root=str(root),
        corpus_manifest_sha256=sha256(root / "corpus-manifest.json"),
        source_audit_sha256=sha256(root / "source-audit.json"),
        partition_sha256=sha256(overlay_path),
        database_sha256=manifest["database_sha256"],
        effective_splits=dict(Counter(splits.values())),
        inherited_excluded_records=len(excluded),
        inherited_excluded_groups=len(overlay.get("excluded_groups", [])),
    )
    return binding, splits, excluded


def _write_train_reuse_delta(root, db, prior_splits):
    """List only membership changes; existing encoded documents stay immutable."""
    current = dict(db.execute("SELECT id,split FROM samples"))
    changes: Counter[str] = Counter()
    path = root / "train-reuse-delta.jsonl"
    with path.open("w") as stream:
        for identity in sorted(prior_splits.keys() | current.keys()):
            old, new = prior_splits.get(identity), current.get(identity)
            if old in {"val", "test"} and new == "train":
                raise ValueError("merge released a prior held-out document into training")
            if old == "test" and new == "val":
                raise ValueError("merge demoted a sealed test document")
            if old == new:
                changes[f"unchanged_{old}"] += 1
                continue
            if old == "train" or new == "train":
                action = "add_train" if new == "train" else "remove_train"
                stream.write(
                    json.dumps(
                        dict(sample_id=identity, action=action, previous_split=old, split=new)
                    )
                    + "\n"
                )
                changes[action] += 1
            elif old in {"val", "test"}:
                changes[f"prior_{old}_to_{new or 'excluded'}"] += 1
    return dict(
        file=path.name,
        sha256=sha256(path),
        counts=dict(changes),
        policy="reuse prior train encodings only for IDs remaining in the final train split",
    )


def merge_text_slices(
    inputs, output, evaluation, *, seed=20260910, max_gib=12, prior_partition=None
):
    """Merge immutable candidates, preserving prior holdouts and duplicate aliases."""
    from minifrontier.data.evaluation import BenchmarkMatcher

    roots, root = [Path(p).resolve() for p in inputs], Path(output).resolve()
    if not roots or len(set(roots)) != len(roots) or root.exists():
        raise ValueError("choose distinct completed inputs and a new canonical output")
    audits, manifests, bindings = [], [], []
    locks: dict[str, str] = {}
    allowlist: dict[str, Any] = {}
    for source in roots:
        audit = json.loads((source / "source-audit.json").read_text())
        manifest = json.loads((source / "corpus-manifest.json").read_text())
        if audit["status"] not in {
            "candidate_slice_complete_pending_admission",
            "candidate_inventory_below_target",
        } or audit.get("formal_admission"):
            raise ValueError("merge requires completed, unadmitted source inventories")
        if sha256(source / "corpus.sqlite") != manifest["database_sha256"]:
            raise ValueError("candidate corpus database differs from its final manifest")
        if sha256(source / "reference-tokenizer.json") != audit["reference_tokenizer_sha256"]:
            raise ValueError("candidate reference tokenizer changed")
        with contextlib.closing(
            sqlite3.connect(f"file:{source / 'corpus.sqlite'}?mode=ro", uri=True)
        ) as db:
            if db.execute(
                "SELECT COUNT(*) FROM samples WHERE stage!='pretrain' OR json_array_length(json_extract(payload,'$.media'))>0"
            ).fetchone()[0]:
                raise ValueError("this merge entry consumes text pretraining candidates only")
            for key, split in db.execute(
                "SELECT key,split FROM links JOIN samples USING(id) WHERE split IN ('val','test')"
            ):
                if locks.get(key) != "test":
                    locks[key] = split
        for name, spec in json.loads((source / "source_allowlist.json").read_text())[
            "sources"
        ].items():
            if name in allowlist and allowlist[name] != spec:
                raise ValueError("conflicting source allowlist versions")
            allowlist[name] = spec
        bindings.append(
            dict(
                root=str(source),
                source_audit_sha256=sha256(source / "source-audit.json"),
                corpus_manifest_sha256=sha256(source / "corpus-manifest.json"),
                database_sha256=manifest["database_sha256"],
            )
        )
        audits.append(audit)
        manifests.append(manifest)
    if len({a["reference_tokenizer_sha256"] for a in audits}) != 1:
        raise ValueError("candidate inventories use different reference tokenizers")
    prior_binding, prior_splits, prior_excluded = None, {}, set()
    if prior_partition is not None:
        prior_binding, prior_splits, prior_excluded = _prior_partition_controls(
            prior_partition, roots[0], locks
        )
    matcher = BenchmarkMatcher(evaluation)
    require_space(root, int(max_gib * GIB), reserve_bytes=80 * GIB)
    root.mkdir(parents=True)
    # Keep the already-computed base dedup index; only ingest new source rows.
    with (
        reserve_write(
            root / "corpus.sqlite",
            (roots[0] / "corpus.sqlite").stat().st_size,
            reserve_bytes=80 * GIB,
        ),
        contextlib.closing(
            sqlite3.connect(f"file:{roots[0] / 'corpus.sqlite'}?mode=ro", uri=True)
        ) as source_db,
        contextlib.closing(sqlite3.connect(root / "corpus.sqlite")) as destination,
    ):
        source_db.backup(destination)
    shutil.copyfile(roots[0] / "reference-tokenizer.json", root / "reference-tokenizer.json")
    builder = CorpusBuilder(root, seed=seed, max_gib=max_gib, val_buckets=50, test_buckets=100)
    builder.counts = Counter(manifests[0]["counts"])
    write_json(root / "source_allowlist.json", dict(formal_admission=False, sources=allowlist))
    write_json(root / "prior-holdout-locks.json", locks)
    audit = dict(
        schema_version=1,
        kind="text_candidate_inventory",
        status="building",
        operation="merge_and_benchmark_exclusion",
        formal_admission=False,
        main_budget_eligible=False,
        inputs=bindings,
        prior_partition=prior_binding,
        seed=seed,
        max_gib=max_gib,
        reference_tokenizer_sha256=audits[0]["reference_tokenizer_sha256"],
        reference_is_final_training_tokenizer=False,
        evaluation_manifest_sha256=sha256(Path(evaluation) / "manifest.json"),
        prior_holdout_locks_sha256=sha256(root / "prior-holdout-locks.json"),
        sources={
            value.get("specification", {}).get("repo", key): {
                field: value[field] for field in ("accepted_records", "accepted_reference_tokens")
            }
            for key, value in audits[0].get("sources", {}).items()
        },
        processed_source_rows=0,
        scanned_for_benchmark=0,
        processor_files={
            p.name: sha256(p)
            for p in (
                Path(__file__),
                Path(__file__).with_name("corpus.py"),
                Path(__file__).with_name("evaluation.py"),
                Path(__file__).with_name("partitions.py"),
            )
        },
        unresolved=[
            "stratified quality review",
            "final train-only tokenizer and encoded data",
            "per-model token mixture and exposure admission",
            "visual inventory is separate and still requires admission",
        ],
    )

    def progress():
        builder.db.commit()
        require_space(root, 64 * 1024**2, reserve_bytes=80 * GIB)
        audit.update(
            updated_unix=time.time(),
            dedup_and_quality_counts=dict(builder.counts),
            database_bytes=(root / "corpus.sqlite").stat().st_size,
        )
        write_json(root / "source-audit.json", audit)
        print(
            json.dumps(
                dict(
                    operation="merge_and_benchmark_exclusion",
                    processed_source_rows=audit["processed_source_rows"],
                    scanned_for_benchmark=audit["scanned_for_benchmark"],
                    status=audit["status"],
                )
            ),
            flush=True,
        )

    try:
        progress()
        for source in roots[1:]:
            aliases = {}
            with contextlib.closing(
                sqlite3.connect(f"file:{source / 'corpus.sqlite'}?mode=ro", uri=True)
            ) as db:
                for identity, payload in db.execute("SELECT id,payload FROM samples ORDER BY id"):
                    row = json.loads(payload)
                    if builder.add(row):
                        entry = audit["sources"].setdefault(
                            row["source"], dict(accepted_records=0, accepted_reference_tokens=0)
                        )
                        entry["accepted_records"] += 1
                        entry["accepted_reference_tokens"] += row["reference_tokens"]
                    aliases[identity] = builder.last_retained_id
                    audit["processed_source_rows"] += 1
                    if audit["processed_source_rows"] % 1000 == 0:
                        progress()
                # Preserve aliases of duplicates removed during source construction,
                # including aliases now mapped to a cross-source retained document.
                builder.db.executemany(
                    "INSERT OR IGNORE INTO links VALUES (?,?)",
                    (
                        (aliases[i], key)
                        for i, key in db.execute("SELECT id,key FROM links")
                        if aliases.get(i)
                    ),
                )
        matches = set()
        matched_sources: Counter[str] = Counter()
        with (root / "benchmark-matches.jsonl").open("w") as stream:
            for identity, source, payload in builder.db.execute(
                "SELECT id,source,payload FROM samples ORDER BY id"
            ):
                hit = matcher.match(json.loads(payload))
                audit["scanned_for_benchmark"] += 1
                if hit:
                    matches.add(identity)
                    matched_sources[source] += 1
                    stream.write(
                        json.dumps(dict(sample_id=identity, source=source, match=hit)) + "\n"
                    )
                if audit["scanned_for_benchmark"] % 5000 == 0:
                    progress()
        audit["contamination"] = dict(
            direct_matches=len(matches),
            matched_sources=dict(matched_sources),
            matches_sha256=sha256(root / "benchmark-matches.jsonl"),
            evaluation_manifest_sha256=audit["evaluation_manifest_sha256"],
            rules=matcher.manifest["rules"],
            limitations=matcher.manifest["limitations"],
        )
        # Keep excluded base rows until now: their aliases must also quarantine
        # duplicates and newly connected repositories from the added slices.
        audit["corpus"] = builder.finalize(split_locks=locks, excluded_ids=matches | prior_excluded)
        audit["contamination"].update(audit["corpus"]["retained_split_controls"])
        if prior_partition is not None:
            audit["train_reuse_delta"] = _write_train_reuse_delta(root, builder.db, prior_splits)
        split_counts: dict[str, Counter] = {}
        strata: Counter[tuple[str, str, str]] = Counter()
        audit["sources"] = {}
        for split, source, payload in builder.db.execute(
            "SELECT split,source,payload FROM samples"
        ):
            row = json.loads(payload)
            count = row["reference_tokens"]
            split_counts.setdefault(split, Counter())[source] += count
            entry = audit["sources"].setdefault(
                source, dict(accepted_records=0, accepted_reference_tokens=0)
            )
            entry["accepted_records"] += 1
            entry["accepted_reference_tokens"] += count
            if split == "train":
                strata[
                    (source, row["task"], str(row.get("source_metadata", {}).get("subsource", "")))
                ] += 1
        # Source/domain strata receive 100 rows, apportioned across sub-sources.
        quotas = {}
        domains = {(s, t) for s, t, _ in strata}
        for domain in domains:
            keys = sorted(k for k in strata if k[:2] == domain)
            target = min(100, sum(strata[k] for k in keys))
            for k in keys:
                quotas[k] = 0
            for _ in range(target):
                candidates = [k for k in keys if quotas[k] < strata[k]]
                chosen = max(
                    candidates, key=lambda k: (quotas[k] == 0, strata[k] / (quotas[k] + 1), k)
                )
                quotas[chosen] += 1
        selected: Counter[tuple[str, str, str]] = Counter()
        with (root / "review-samples.jsonl").open("w") as review:
            for (payload,) in builder.db.execute(
                "SELECT payload FROM samples WHERE split='train' ORDER BY id"
            ):
                row = json.loads(payload)
                key = (
                    row["source"],
                    row["task"],
                    str(row.get("source_metadata", {}).get("subsource", "")),
                )
                if selected[key] < quotas[key]:
                    review.write(
                        json.dumps(dict(split="train", stratum=key, record=row), ensure_ascii=False)
                        + "\n"
                    )
                    selected[key] += 1
        audit.update(
            split_reference_tokens=split_counts,
            status="candidate_slice_complete_pending_admission",
            review=dict(
                status="awaiting_review_method_and_decisions",
                samples={json.dumps(k): v for k, v in selected.items()},
                sha256=sha256(root / "review-samples.jsonl"),
            ),
        )
        progress()
    except BaseException as error:
        audit.update(
            status="interrupted_unadmitted", error=type(error).__name__ + ": " + str(error)
        )
        progress()
        raise
    finally:
        builder.db.close()
    return audit


def main(argv=None):
    import argparse
    import sys

    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "merge":
        parser = argparse.ArgumentParser(description=merge_text_slices.__doc__)
        parser.add_argument("--inputs", nargs="+", required=True)
        parser.add_argument("--output", required=True)
        parser.add_argument("--evaluation", required=True)
        parser.add_argument("--seed", type=int, default=20260910)
        parser.add_argument("--max-gib", type=float, default=12)
        parser.add_argument(
            "--prior-partition", help="effective split/exclusion view over the first input"
        )
        print(json.dumps(merge_text_slices(**vars(parser.parse_args(argv[1:]))), indent=2))
        return

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference-tokenizer", required=True)
    parser.add_argument("--targets", help="JSON object of accepted reference-token targets")
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--max-gib", type=float, default=12)
    parser.add_argument(
        "--resume", action="store_true", help="resume an interrupted code candidate inventory"
    )
    args = parser.parse_args(argv)
    build_text_slice(
        args.output,
        args.reference_tokenizer,
        targets=json.loads(args.targets) if args.targets else None,
        seed=args.seed,
        max_gib=args.max_gib,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
