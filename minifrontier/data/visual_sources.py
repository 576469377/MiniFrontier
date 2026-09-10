"""Pinned visual candidates and the historical small ALLaVA pilot."""

import contextlib
import hashlib
import io
import json
import math
import os
import random
import shutil
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from minifrontier.data import sha256
from minifrontier.data.corpus import CorpusBuilder
from minifrontier.data.media_hash import decoded_hashes
from minifrontier.data.minifrontier1 import write_json
from minifrontier.data.public_sources import source_rows
from minifrontier.data.remote import RangeFile
from minifrontier.storage import GIB, require_space, reserve_write

REPO = "HuggingFaceM4/FineVision"
REVISION = "3c380a731a3429c1d04693d6ec16d7e683def84c"

VISUAL_CANDIDATES = {
    "allava_laion": dict(
        upstream="FreedomIntelligence/ALLaVA-4V",
        upstream_revision="0fd42fce5c047d387a4bb5318d588eae9a9797f0",
        license="CC-BY-NC-4.0; underlying LAION image rights retained",
        use_scope="research candidate; NC restrictions must accompany any weight release",
        task="caption_or_vqa",
    ),
    "CoSyn_400k_document": dict(
        upstream="allenai/CoSyn-400K",
        upstream_revision="86e46e1fd5e754d056169f0fb38f06c6997ff7de",
        license="ODC-BY-1.0; Ai2 responsible-use guidance; generated-content terms retained",
        use_scope="research/education; code-generated images and QAs have separate generator terms",
        task="ocr_document",
    ),
    "CoSyn_400k_chart": dict(
        upstream="allenai/CoSyn-400K",
        upstream_revision="86e46e1fd5e754d056169f0fb38f06c6997ff7de",
        license="ODC-BY-1.0; Ai2 responsible-use guidance; generated-content terms retained",
        use_scope="research/education; code-generated images and QAs have separate generator terms",
        task="chart_table",
    ),
}
VISUAL_TARGETS = dict(allava_laion=60_000, CoSyn_400k_document=20_000, CoSyn_400k_chart=20_000)


def _check_retained_media(db, audit, root):
    """Verify immutable pixels and the complete record inventory before further work."""
    rows = db.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
    if rows != audit["dedup_counts"].get("accepted", 0) or rows != sum(
        s["accepted_records"] for s in audit["sources"].values()
    ):
        raise ValueError("interrupted media audit and retained records disagree")
    seen: set[str] = set()
    media_files: dict[str, str] = {}
    source_counts: dict[str, Counter] = {}
    source_images: dict[str, set[str]] = {}
    source_domains: dict[str, Counter] = {}
    for (payload,) in db.execute("SELECT payload FROM samples"):
        record = json.loads(payload)
        source_counts.setdefault(record["source"], Counter()).update(
            accepted_records=1,
            accepted_reference_tokens=record["reference_tokens"],
            answer_reference_tokens=record["answer_reference_tokens"],
        )
        source_domains.setdefault(record["source"], Counter())[record["task"]] += 1
        for media in record["media"]:
            seen.add(media["rgb_sha256"])
            source_images.setdefault(record["source"], set()).add(media["rgb_sha256"])
            previous = media_files.setdefault(media["path"], media["sha256"])
            if previous != media["sha256"]:
                raise ValueError("retained records disagree on their media checksum")
    if set(source_counts) != {
        REPO + "/" + name for name, s in audit["sources"].items() if s["accepted_records"]
    }:
        raise ValueError("retained media source identities disagree")
    for name, source in audit["sources"].items():
        key = REPO + "/" + name
        if (
            any(
                source_counts.get(key, Counter())[field] != source[field]
                for field in (
                    "accepted_records",
                    "accepted_reference_tokens",
                    "answer_reference_tokens",
                )
            )
            or len(source_images.get(key, set())) != source["unique_images"]
            or source_domains.get(key, Counter()) != source["domain_records"]
        ):
            raise ValueError("retained media source counters disagree")
    total = 0
    images = root / "images"
    for relative, expected in media_files.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(images) or sha256(path) != expected:
            raise ValueError("retained candidate media path/hash differs")
        total += path.stat().st_size
    if len(seen) != audit["unique_images"] or total != audit["media_bytes"]:
        raise ValueError("interrupted media byte/image counters disagree")
    if {str(p.relative_to(root)) for p in images.rglob("*.image")} != set(media_files):
        raise ValueError("media inventory has unreferenced or missing files")
    if db.execute("PRAGMA quick_check").fetchone() != ("ok",):
        raise ValueError("retained corpus integrity check failed")
    return seen


def _finalize_visual_inventory(builder, audit):
    root = builder.root
    builder.db.commit()
    metadata_bytes = sum(p.stat().st_size for p in root.iterdir() if p.is_file())
    finalization_peak = metadata_bytes + (root / "corpus.sqlite").stat().st_size + 16 * 1024**2
    if finalization_peak > audit["metadata_gib"] * GIB:
        raise ValueError("visual metadata finalization exceeds the measured storage budget")
    require_space(root, finalization_peak - metadata_bytes, reserve_bytes=80 * GIB)
    audit["metadata_storage"] = dict(
        measurement="physical metadata plus worst-case database rollback journal and reports",
        before_finalize_bytes=metadata_bytes,
        finalization_peak_bound_bytes=finalization_peak,
        budget_bytes=int(audit["metadata_gib"] * GIB),
    )
    audit["corpus"] = builder.finalize()
    split_media: dict[str, set[str]] = {}
    split_tokens: dict[str, dict[str, int]] = {}
    review_counts: Counter[str] = Counter()
    with (root / "review-samples.jsonl").open("w") as review:
        for split, payload in builder.db.execute("SELECT split,payload FROM samples ORDER BY id"):
            record = json.loads(payload)
            split_media.setdefault(split, set()).update(m["rgb_sha256"] for m in record["media"])
            tokens = split_tokens.setdefault(split, {})
            tokens[record["task"]] = (
                tokens.get(record["task"], 0) + record["answer_reference_tokens"]
            )
            key = record["source"] + ":" + record["task"]
            if split == "train" and review_counts[key] < 100:
                review.write(
                    json.dumps(dict(split=split, record=record), ensure_ascii=False) + "\n"
                )
                review_counts[key] += 1
    targets_met = set(audit["sources"]) == set(audit["target_independent_images"]) and all(
        s["status"] == "candidate_target_reached" for s in audit["sources"].values()
    )
    audit.update(
        status="candidate_slice_complete_pending_admission"
        if targets_met
        else "candidate_inventory_below_target",
        split_independent_images={k: len(v) for k, v in split_media.items()},
        split_answer_reference_tokens=split_tokens,
        review=dict(
            status="awaiting_manual_review",
            samples=dict(review_counts),
            sha256=sha256(root / "review-samples.jsonl"),
        ),
    )


def finalize_storage_limited_slice(output, construction_run):
    """Close retained media after a verified quota stop; no download or admission."""
    import fcntl

    root = Path(output).resolve()
    with (root / ".finalize.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        audit_path = root / "source-audit.json"
        audit = json.loads(audit_path.read_text())
        run = json.loads(Path(construction_run).read_text())
        command = run.get("command", [])
        pid = run.get("pid")
        if (
            run.get("kind") != "data_construction"
            or "--output" not in command
            or Path(command[command.index("--output") + 1]).resolve() != root
            or not isinstance(pid, int)
            or pid < 1
        ):
            raise ValueError("construction identity does not bind this media output")
        if Path(f"/proc/{pid}").exists():
            raise ValueError("construction process is still present; do not finalize its database")
        if (
            audit.get("status") != "interrupted_unadmitted"
            or audit.get("formal_admission")
            or audit.get("source") != REPO
            or audit.get("revision") != REVISION
            or audit.get("error")
            != "ValueError: media byte budget reached; inventory remains unadmitted"
            or (root / "corpus-manifest.json").exists()
        ):
            raise ValueError("only an unfinalized media-byte-limited candidate can be closed")
        peak = 2 * (root / "corpus.sqlite").stat().st_size + 16 * 1024**2
        if peak > audit["metadata_gib"] * GIB:
            raise ValueError("finalization peak would exceed the existing metadata budget")
        require_space(root, peak, reserve_bytes=80 * GIB)
        db = sqlite3.connect((root / "corpus.sqlite").as_uri() + "?mode=ro", uri=True)
        try:
            seen = _check_retained_media(db, audit, root)
            if db.execute("SELECT COUNT(*) FROM samples WHERE split IS NOT NULL").fetchone()[0]:
                raise ValueError("the interrupted inventory already has a partition")
        finally:
            db.close()
        before_hash = sha256(audit_path)
        backup = root / "source-audit-before-slice-finalize.json"
        if backup.exists():
            raise ValueError("a previous finalization attempt needs inspection")
        shutil.copyfile(audit_path, backup)
        audit["finalization"] = dict(
            operation="close_media_byte_limited_slice",
            started_unix=time.time(),
            before_audit_sha256=before_hash,
            before_database_sha256=sha256(root / "corpus.sqlite"),
            construction_run_sha256=sha256(construction_run),
            construction_stop_reason=audit.pop("error"),
            processor_files={
                "visual_sources.py": sha256(__file__),
                "corpus.py": sha256(Path(__file__).with_name("corpus.py")),
            },
            media_files_unchanged=True,
            new_download_bytes=0,
        )
        builder = CorpusBuilder(
            root,
            seed=audit["seed"],
            max_gib=audit["metadata_gib"],
            val_buckets=50,
            test_buckets=100,
        )
        try:
            builder.counts = Counter(audit["dedup_counts"])
            for source in audit["sources"].values():
                if source["status"] == "reading":
                    source["status"] = "storage_slice_closed_below_target"
            _finalize_visual_inventory(builder, audit)
            audit.update(
                unique_images=len(seen),
                updated_unix=time.time(),
                database_bytes=(root / "corpus.sqlite").stat().st_size,
                free_gib=shutil.disk_usage(root).free / GIB,
                remaining_independent_image_targets={
                    name: max(0, target - audit["sources"].get(name, {}).get("unique_images", 0))
                    for name, target in audit["target_independent_images"].items()
                },
            )
            write_json(audit_path, audit)
        finally:
            builder.db.close()
        return audit


def compact_visual_inventory(corpus, output):
    """Copy immutable metadata with unique aliases; hard-link verified raw pixels."""
    source, root = Path(corpus).resolve(), Path(output).resolve()
    if root.exists():
        raise FileExistsError("compaction requires a new candidate version")
    audit_path, manifest_path = source / "source-audit.json", source / "corpus-manifest.json"
    audit, manifest = json.loads(audit_path.read_text()), json.loads(manifest_path.read_text())
    if (
        audit.get("formal_admission")
        or audit.get("status")
        not in {"candidate_slice_complete_pending_admission", "candidate_inventory_below_target"}
        or audit.get("kind") != "visual_candidate_inventory"
    ):
        raise ValueError("compaction requires a finalized unadmitted visual inventory")
    original_hash = sha256(source / "corpus.sqlite")
    if original_hash != manifest["database_sha256"] or audit["corpus"] != manifest:
        raise ValueError("original corpus/audit hashes disagree")
    metadata_limit = int(audit["metadata_gib"] * GIB)
    # One bounded destination database plus its possible rollback journal and reports.
    database_cap = min(
        (metadata_limit - 16 * 1024**2) // 2,
        (source / "corpus.sqlite").stat().st_size + 8 * 1024**2,
    )
    if database_cap < 64 * 1024:
        raise ValueError("metadata budget is smaller than the destination schema")
    incoming = 2 * database_cap + 48 * 1024**2
    if (
        audit["media_bytes"]
        + sum(p.stat().st_size for p in source.iterdir() if p.is_file())
        + incoming
        > audit["max_gib"] * GIB
    ):
        raise ValueError("compaction overlap exceeds the original total storage budget")
    require_space(root, incoming, reserve_bytes=80 * GIB)
    with contextlib.closing(
        sqlite3.connect((source / "corpus.sqlite").as_uri() + "?mode=ro", uri=True)
    ) as db:
        _check_retained_media(db, audit, source)
    builder = CorpusBuilder(root, seed=audit["seed"], max_gib=audit["metadata_gib"])
    report = dict(
        kind="visual_metadata_compaction",
        status="building",
        formal_admission=False,
        main_budget_eligible=False,
        started_unix=time.time(),
        source_manifest_sha256=sha256(manifest_path),
        source_audit_sha256=sha256(audit_path),
        source_database_sha256=original_hash,
        source_path=os.path.relpath(source, root),
        processor_files={
            "visual_sources.py": sha256(__file__),
            "corpus.py": sha256(Path(__file__).with_name("corpus.py")),
        },
        metadata_budget_bytes=metadata_limit,
        new_media_bytes=0,
        database_cap_bytes=database_cap,
        construction_peak_bound_bytes=incoming,
        tables={},
    )
    write_json(root / "compaction.json", report)
    try:
        db = builder.db
        db.execute("PRAGMA temp_store=MEMORY")
        db.execute("PRAGMA cache_size=-32768")
        pages = report["database_cap_bytes"] // db.execute("PRAGMA page_size").fetchone()[0]
        actual_cap = db.execute(f"PRAGMA max_page_count={pages}").fetchone()[0]
        if actual_cap > pages:
            raise ValueError("metadata budget is smaller than the destination schema")
        db.execute(
            "ATTACH DATABASE ? AS original", ((source / "corpus.sqlite").as_uri() + "?mode=ro",)
        )
        for table in ("samples", "bands", "image_bands", "links"):
            before = db.execute(f"SELECT COUNT(*) FROM original.{table}").fetchone()[0]
            columns = "id,key" if table == "links" else "*"
            db.execute(f"INSERT OR IGNORE INTO main.{table} SELECT {columns} FROM original.{table}")
            db.commit()
            after = db.execute(f"SELECT COUNT(*) FROM main.{table}").fetchone()[0]
            if table != "links" and before != after:
                raise ValueError("compaction changed a non-alias table count")
            for left, right in (("main", "original"), ("original", "main")):
                if db.execute(
                    f"SELECT {columns} FROM {left}.{table} EXCEPT SELECT {columns} FROM {right}.{table} LIMIT 1"
                ).fetchone():
                    raise ValueError("compaction changed corpus contents or grouping identities")
            report["tables"][table] = dict(
                original_rows=before, compact_rows=after, exact_set_equal=True
            )
            write_json(root / "compaction.json", report)
        if db.execute("PRAGMA quick_check").fetchone() != ("ok",):
            raise ValueError("compacted database integrity failed")
        db.execute("DETACH DATABASE original")
        db.close()
        images = root / "images"
        for old in (source / "images").rglob("*.image"):
            destination = images / old.relative_to(source / "images")
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(old, destination)
        for name in ("source_allowlist.json", "reference-tokenizer.json", "review-samples.jsonl"):
            if (source / name).exists():
                shutil.copyfile(source / name, root / name)
        if sha256(source / "corpus.sqlite") != original_hash:
            raise ValueError("source database changed during compaction")
        manifest = dict(manifest, database_sha256=sha256(root / "corpus.sqlite"))
        audit = dict(
            audit,
            corpus=manifest,
            database_bytes=(root / "corpus.sqlite").stat().st_size,
            updated_unix=time.time(),
            free_gib=shutil.disk_usage(root).free / GIB,
            metadata_compaction="compaction.json",
        )
        with contextlib.closing(
            sqlite3.connect((root / "corpus.sqlite").as_uri() + "?mode=ro", uri=True)
        ) as check:
            _check_retained_media(check, audit, root)
        report.update(
            status="mechanical_checks_passed_pending_quality_admission",
            database_sha256=manifest["database_sha256"],
            database_bytes=audit["database_bytes"],
            saved_database_bytes=(source / "corpus.sqlite").stat().st_size
            - audit["database_bytes"],
            completed_unix=time.time(),
            media_files=audit["unique_images"],
            original_content_and_splits_unchanged=True,
        )
        write_json(root / "compaction.json", report)
        write_json(root / "source-audit.json", audit)
        write_json(root / "corpus-manifest.json", manifest)
        actual_bytes = sum(p.stat().st_size for p in root.iterdir() if p.is_file())
        if actual_bytes > metadata_limit:
            raise ValueError("compacted metadata exceeds its original budget")
        return report
    except BaseException as error:
        builder.db.close()
        report.update(status="failed_unadmitted", error=f"{type(error).__name__}: {error}")
        write_json(root / "compaction.json", report)
        raise


def candidate_turns(subset, row):
    """Keep complete grounded QA turns, retaining ratings and original turn identity."""
    turns = row.get("texts", [])
    ratings = (
        "image_correspondence_ratings",
        "visual_dependency_ratings",
        "formatting_ratings",
        "relevance_ratings",
    )
    if not turns or any(len(row.get(key, [])) != len(turns) for key in ratings):
        return [], "missing_or_misaligned_ratings"
    selected = []
    for index, turn in enumerate(turns):
        scores = {key: row[key][index] for key in ratings}
        if any(
            not isinstance(v, (int, float)) or not math.isfinite(v) or not 3 <= v <= 5
            for v in scores.values()
        ):
            continue
        question, answer = turn.get("user"), turn.get("assistant")
        if (
            not isinstance(question, str)
            or not isinstance(answer, str)
            or not 8 <= len(question.strip()) <= 1200
            or not 40 <= len(answer.strip()) <= 2800
            or any(c in question + answer for c in ("\ufffd", "\x00"))
        ):
            continue
        task = VISUAL_CANDIDATES[subset]["task"]
        if task == "caption_or_vqa":
            task = (
                "caption"
                if any(
                    word in question.lower()
                    for word in ("describ", "description", "descriptive", "elaborate", "details")
                )
                else "vqa"
            )
        selected.append(
            dict(
                index=index,
                question=question.strip(),
                answer=answer.strip(),
                task=task,
                ratings=scores,
            )
        )
        if len(selected) == 8:
            break
    return selected, None if selected else "no_complete_grounded_turn"


def build_visual_candidates(
    output,
    reference_tokenizer,
    *,
    targets=None,
    seed=20260911,
    max_gib=20,
    metadata_gib=3,
    resume=False,
):
    """Build the first shared media inventory; never auto-admit or claim held-out quality."""
    from tokenizers import Tokenizer

    root = Path(output).resolve()
    targets = VISUAL_TARGETS if targets is None else targets
    if (root.exists() and not resume) or not 0 < metadata_gib < max_gib:
        raise ValueError("choose a new output and separate positive metadata/media budgets")
    if not targets or any(k not in VISUAL_CANDIDATES or n < 1 for k, n in targets.items()):
        raise ValueError("unknown subset or nonpositive independent-image target")
    previous = json.loads((root / "source-audit.json").read_text()) if resume else None
    if previous and (
        previous["status"] != "interrupted_unadmitted"
        or previous.get("formal_admission")
        or previous["seed"] != seed
        or previous["target_independent_images"] != targets
        or previous["source"] != REPO
        or previous["revision"] != REVISION
        or previous["max_gib"] != max_gib
        or previous["metadata_gib"] != metadata_gib
        or previous["reference_tokenizer_sha256"] != sha256(reference_tokenizer)
        or not previous.get("error", "").startswith(
            ("ReadTimeout", "RemoteProtocolError", "ConnectError", "ConnectionError")
        )
    ):
        raise ValueError(
            "resume requires an unchanged unadmitted inventory interrupted by source transport"
        )
    require_space(root, int(max_gib * GIB), reserve_bytes=80 * GIB)
    builder = CorpusBuilder(root, seed=seed, max_gib=metadata_gib, val_buckets=50, test_buckets=100)
    tokenizer = Tokenizer.from_file(str(reference_tokenizer))
    images = root / "images"
    images.mkdir(exist_ok=resume)
    audit: dict[str, Any] = dict(
        schema_version=1,
        kind="visual_candidate_inventory",
        status="building",
        formal_admission=False,
        main_budget_eligible=False,
        seed=seed,
        source=REPO,
        revision=REVISION,
        target_independent_images=targets,
        max_gib=max_gib,
        metadata_gib=metadata_gib,
        media_bytes=0,
        reference_tokenizer_sha256=sha256(reference_tokenizer),
        sources={},
        processor_files={
            str(Path(__file__).name): sha256(__file__),
            "corpus.py": sha256(Path(__file__).with_name("corpus.py")),
            "public_sources.py": sha256(Path(__file__).with_name("public_sources.py")),
            "remote.py": sha256(Path(__file__).with_name("remote.py")),
        },
        unresolved=[
            "upstream validation/benchmark image identity exclusion",
            "stratified source/domain manual review and fixed visual evaluation",
            "full OCR transcription, Chinese OCR, multiimage and video coverage",
            "per-model tokenizer, complete media/answer length and phase exposure audit",
        ],
    )
    write_json(
        root / "source_allowlist.json",
        dict(
            status="reviewed_for_candidate_extraction_only",
            formal_admission=False,
            sources={
                k: dict(VISUAL_CANDIDATES[k], dataset=REPO, revision=REVISION) for k in targets
            },
        ),
    )
    seen: set[str] = set()
    if previous:
        rows = builder.db.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
        try:
            seen = _check_retained_media(builder.db, previous, root)
        except BaseException:
            builder.db.close()
            raise
        write_json(
            root / f"source-audit-resume-{len(previous.get('resumes', [])) + 1}.json", previous
        )
        processors = audit["processor_files"]
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
        audit.update(status="building", processor_files=processors)
        builder.counts = Counter(audit["dedup_counts"])
        estimate = builder.db.execute(
            "SELECT COALESCE(SUM(4*length(CAST(payload AS BLOB))+1024),0) FROM samples"
        ).fetchone()[0]
        builder.approximate_bytes = max(builder.approximate_bytes, estimate + 16 * 1024**2)

    def progress():
        builder.db.commit()
        audit.update(
            updated_unix=time.time(),
            free_gib=shutil.disk_usage(root).free / GIB,
            database_bytes=(root / "corpus.sqlite").stat().st_size,
            unique_images=len(seen),
            dedup_counts=dict(builder.counts),
        )
        write_json(root / "source-audit.json", audit)

    try:
        progress()
        for number, (subset, target) in enumerate(targets.items()):
            source = dict(
                repo=REPO,
                revision=REVISION,
                prefix=subset,
                file_prefix="train-",
                bounded_http_ranges=True,
            )
            entry: dict[str, Any] = dict(
                specification=dict(VISUAL_CANDIDATES[subset], **source),
                status="reading",
                unique_images=0,
                accepted_records=0,
                accepted_reference_tokens=0,
                answer_reference_tokens=0,
                rejected=Counter(),
                domain_records=Counter(),
                read_rows=0,
                max_source_rows=target * 4,
            )
            if previous and subset in audit["sources"]:
                entry = audit["sources"][subset]
                entry["rejected"] = Counter(entry["rejected"])
                entry["domain_records"] = Counter(entry["domain_records"])
                if entry["status"] == "candidate_target_reached":
                    continue
            audit["sources"][subset] = entry
            progress()
            for row, identity in source_rows(
                subset,
                seed=seed + number * 104729,
                audit=entry,
                specification=source,
                **({"skip_rows": entry["read_rows"]} if resume else {}),
            ):
                if entry["read_rows"] >= entry["max_source_rows"]:
                    entry["scan_limit_reached"] = True
                    break
                entry["read_rows"] += 1
                turns, reason = candidate_turns(subset, row)
                if reason:
                    entry["rejected"][reason] += 1
                    continue
                if len(row.get("images", [])) != 1:
                    entry["rejected"]["not_single_image"] += 1
                    continue
                binary = row["images"][0].get("bytes")
                if not binary or len(binary) > 16 * 1024**2:
                    entry["rejected"]["invalid_media_byte_size"] += 1
                    continue
                try:
                    with Image.open(io.BytesIO(binary)) as image:
                        if image.width * image.height > 20_000_000 or min(image.size) < 32:
                            entry["rejected"]["invalid_media_dimensions"] += 1
                            continue
                        hashes = decoded_hashes(image)
                except (OSError, ValueError, Image.DecompressionBombError):
                    entry["rejected"]["media_decode_failed"] += 1
                    continue
                if hashes["rgb_sha256"] in seen:
                    entry["rejected"]["duplicate_image"] += 1
                    continue
                if audit["media_bytes"] + len(binary) > (max_gib - metadata_gib) * GIB:
                    raise ValueError("media byte budget reached; inventory remains unadmitted")
                path = images / hashes["rgb_sha256"][:2] / (hashes["rgb_sha256"] + ".image")
                with reserve_write(path, len(binary), reserve_bytes=80 * GIB):
                    path.write_bytes(binary)
                audit["media_bytes"] += len(binary)
                raw_sha256 = hashlib.sha256(binary).hexdigest()
                media = dict(
                    path=str(path.relative_to(root)), kind="image", sha256=raw_sha256, **hashes
                )
                accepted = 0
                for turn in turns:
                    text = "<|image|>\nUser: " + turn["question"] + "\nAssistant: " + turn["answer"]
                    record = dict(
                        source=REPO + "/" + subset,
                        revision=REVISION,
                        item_id=identity + ":qa" + str(turn["index"]),
                        group_id=hashes["rgb_sha256"],
                        license=VISUAL_CANDIDATES[subset]["license"],
                        lang="en",
                        task=turn["task"],
                        stage="pretrain",
                        text=text,
                        media=[media],
                        visual_question=turn["question"],
                        visual_answer=turn["answer"],
                        original_image_path=row["images"][0].get("path"),
                        source_metadata=dict(
                            source_name=row.get("source"),
                            origin=identity,
                            turn_index=turn["index"],
                            ratings=turn["ratings"],
                            upstream=VISUAL_CANDIDATES[subset],
                        ),
                        quality_flags=["candidate-not-formally-admitted"],
                        raw_pair_sha256=hashlib.sha256(
                            json.dumps(row["texts"][turn["index"]], sort_keys=True).encode()
                        ).hexdigest(),
                        reference_tokens=len(tokenizer.encode(text, add_special_tokens=False).ids),
                        answer_reference_tokens=len(
                            tokenizer.encode(turn["answer"], add_special_tokens=False).ids
                        ),
                    )
                    if builder.add(record):
                        accepted += 1
                        entry["accepted_records"] += 1
                        entry["domain_records"][turn["task"]] += 1
                        entry["accepted_reference_tokens"] += record["reference_tokens"]
                        entry["answer_reference_tokens"] += record["answer_reference_tokens"]
                if accepted:
                    seen.add(hashes["rgb_sha256"])
                    entry["unique_images"] += 1
                else:
                    path.unlink()  # This newly written candidate has no admitted record.
                    audit["media_bytes"] -= len(binary)
                if entry["unique_images"] % 250 == 0:
                    progress()
                    print(
                        json.dumps(
                            dict(
                                source=subset,
                                images=entry["unique_images"],
                                records=entry["accepted_records"],
                                bytes=audit["media_bytes"],
                            )
                        ),
                        flush=True,
                    )
                if entry["unique_images"] >= target:
                    break
            entry["status"] = (
                "candidate_target_reached"
                if entry["unique_images"] >= target
                else "scan_limit_reached_below_target"
                if entry.get("scan_limit_reached")
                else "source_exhausted_below_target"
            )
            progress()
        _finalize_visual_inventory(builder, audit)
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


def build_visual_pilot(output, *, images=96, seed=137, max_gib=4):
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile

    root = Path(output).resolve()
    if root.exists():
        raise FileExistsError("visual corpus is immutable; choose a new version")
    if not 1 <= images <= 4096:
        raise ValueError("this bounded pilot admits 1-4096 images")
    require_space(root, max_gib * GIB)
    builder = CorpusBuilder(root, seed=seed, max_gib=max_gib / 2)
    (root / "images").mkdir()
    files = [
        item
        for item in HfApi().list_repo_tree(
            REPO, path_in_repo="allava_laion", revision=REVISION, repo_type="dataset"
        )
        if isinstance(item, RepoFile) and item.path.endswith(".parquet")
    ]
    rng = random.Random(seed)
    rng.shuffle(files)
    audit = dict(
        source=REPO,
        revision=REVISION,
        subset="allava_laion",
        license="CC-BY-NC-4.0",
        upstream="FreedomIntelligence/ALLaVA-4V@0fd42fce5c047d387a4bb5318d588eae9a9797f0",
        main_budget_eligible=False,
        reason="small pilot; formal split/source review incomplete",
        seed=seed,
        unique_images=0,
        media_bytes=0,
        reads=[],
        status="building",
    )
    seen = set()
    try:
        for file in files:
            url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{file.path}"
            with RangeFile(url, file.size, chunk_size=1024**2, network_budget=2 * GIB) as source:
                parquet = pq.ParquetFile(source)
                groups = list(range(parquet.num_row_groups))
                rng.shuffle(groups)
                for group in groups:
                    if parquet.metadata.row_group(group).total_byte_size > 512 * 1024**2:
                        raise ValueError("visual row group exceeds 512 MiB memory cap")
                    rows = parquet.read_row_group(group).to_pylist()
                    order = list(range(len(rows)))
                    rng.shuffle(order)
                    audit["reads"].append(dict(file=file.path, group=group, rows=len(rows)))
                    for index in order:
                        row = rows[index]
                        if (
                            row["source"] != "allava_laion"
                            or len(row["images"]) != 1
                            or not row["texts"]
                        ):
                            continue
                        binary = row["images"][0]["bytes"]
                        if not binary or len(binary) > 16 * 1024**2:
                            continue
                        with Image.open(io.BytesIO(binary)) as image:
                            if image.width * image.height > 20_000_000:
                                continue
                            hashes = decoded_hashes(image)
                        if hashes["rgb_sha256"] in seen:
                            continue
                        caption = row["texts"][0]
                        # Later QA turns are not silently relabeled as captions.
                        if not any(
                            word in caption["user"].lower()
                            for word in (
                                "describ",
                                "description",
                                "descriptive",
                                "elaborate",
                                "details",
                            )
                        ):
                            continue
                        media_path = root / "images" / (hashes["rgb_sha256"] + ".image")
                        if audit["media_bytes"] + len(binary) > max_gib * GIB // 2:
                            raise ValueError("visual media byte budget reached")
                        with reserve_write(media_path, len(binary)):
                            media_path.write_bytes(binary)
                        audit["media_bytes"] += len(binary)
                        resource = dict(
                            path=str(media_path.relative_to(root)), kind="image", **hashes
                        )
                        common = dict(
                            source=REPO + "/allava_laion",
                            revision=REVISION,
                            item_id=f"{file.path}:rg{group}:row{index}",
                            group_id=hashes["rgb_sha256"],
                            license="CC-BY-NC-4.0; ALLaVA underlying LAION-image rights retained",
                            lang="en",
                            media=[resource],
                            original_image_path=row["images"][0].get("path"),
                            quality_flags=["pilot-only", "source-split-review-pending"],
                            upstream_source=audit["upstream"],
                        )
                        accepted = builder.add(
                            dict(
                                common,
                                stage="pretrain",
                                task="caption",
                                text="<|image|> " + caption["assistant"],
                            )
                        )
                        for turn in row["texts"][:3]:
                            builder.add(
                                dict(
                                    common,
                                    stage="sft",
                                    task="caption",
                                    turns=[
                                        dict(
                                            role="user",
                                            content="<|image|>\n" + turn["user"].strip(),
                                        ),
                                        dict(role="assistant", content=turn["assistant"].strip()),
                                    ],
                                )
                            )
                        if accepted:
                            seen.add(hashes["rgb_sha256"])
                            audit["unique_images"] = len(seen)
                        if len(seen) >= images:
                            break
                    (root / "source-audit.json").write_text(json.dumps(audit, indent=2))
                    print(
                        json.dumps(dict(images=len(seen), downloaded_bytes=source.transferred)),
                        flush=True,
                    )
                    if len(seen) >= images:
                        break
            if len(seen) >= images:
                break
        audit["corpus"] = builder.finalize()
        audit["status"] = "complete"
    except BaseException as error:
        builder.db.commit()
        audit.update(status="interrupted", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        (root / "source-audit.json").write_text(json.dumps(audit, indent=2))
    return audit


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Build a bounded shared visual candidate inventory"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference-tokenizer")
    parser.add_argument("--targets", help="JSON object of independent-image targets by subset")
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--max-gib", type=float, default=20)
    parser.add_argument("--metadata-gib", type=float, default=3)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--compact-from", help="completed visual inventory; create a new compact version"
    )
    mode.add_argument(
        "--resume", action="store_true", help="resume a transport-interrupted visual inventory"
    )
    mode.add_argument(
        "--finalize-slice",
        action="store_true",
        help="close a storage-limited slice using its original settings; no download",
    )
    parser.add_argument(
        "--construction-run", help="original producer run.json; required to close a slice"
    )
    args = parser.parse_args(argv)
    if args.compact_from:
        print(json.dumps(compact_visual_inventory(args.compact_from, args.output), indent=2))
        return
    if args.finalize_slice:
        if not args.construction_run:
            parser.error("--finalize-slice requires --construction-run")
        print(
            json.dumps(finalize_storage_limited_slice(args.output, args.construction_run), indent=2)
        )
        return
    if not args.reference_tokenizer or args.construction_run:
        parser.error(
            "building needs --reference-tokenizer; --construction-run is for slice finalization"
        )
    build_visual_candidates(
        args.output,
        args.reference_tokenizer,
        targets=json.loads(args.targets) if args.targets else None,
        seed=args.seed,
        max_gib=args.max_gib,
        metadata_gib=args.metadata_gib,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
