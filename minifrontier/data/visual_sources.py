"""Pinned visual candidates and the historical small ALLaVA pilot."""

import hashlib
import io
import json
import math
import random
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

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
    output, reference_tokenizer, *, targets=None, seed=20260911, max_gib=20, metadata_gib=3
):
    """Build the first shared media inventory; never auto-admit or claim held-out quality."""
    from tokenizers import Tokenizer

    root = Path(output).resolve()
    targets = VISUAL_TARGETS if targets is None else targets
    if root.exists() or not 0 < metadata_gib < max_gib:
        raise ValueError("choose a new output and separate positive metadata/media budgets")
    if not targets or any(k not in VISUAL_CANDIDATES or n < 1 for k, n in targets.items()):
        raise ValueError("unknown subset or nonpositive independent-image target")
    require_space(root, int(max_gib * GIB), reserve_bytes=80 * GIB)
    builder = CorpusBuilder(root, seed=seed, max_gib=metadata_gib, val_buckets=50, test_buckets=100)
    tokenizer = Tokenizer.from_file(str(reference_tokenizer))
    images = root / "images"
    images.mkdir()
    from minifrontier.data import sha256

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
            audit["sources"][subset] = entry
            progress()
            for row, identity in source_rows(
                subset, seed=seed + number * 104729, audit=entry, specification=source
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
        audit["corpus"] = builder.finalize()
        split_media: dict[str, set[str]] = {}
        review_counts: Counter[str] = Counter()
        with (root / "review-samples.jsonl").open("w") as review:
            for split, payload in builder.db.execute(
                "SELECT split,payload FROM samples ORDER BY id"
            ):
                record = json.loads(payload)
                split_media.setdefault(split, set()).update(
                    m["rgb_sha256"] for m in record["media"]
                )
                key = record["source"] + ":" + record["task"]
                if split == "train" and review_counts[key] < 100:
                    review.write(
                        json.dumps(dict(split=split, record=record), ensure_ascii=False) + "\n"
                    )
                    review_counts[key] += 1
        audit.update(
            status=(
                "candidate_slice_complete_pending_admission"
                if all(s["status"] == "candidate_target_reached" for s in audit["sources"].values())
                else "candidate_inventory_below_target"
            ),
            split_independent_images={k: len(v) for k, v in split_media.items()},
            review=dict(status="awaiting_manual_review", samples=dict(review_counts)),
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
    parser.add_argument("--reference-tokenizer", required=True)
    parser.add_argument("--targets", help="JSON object of independent-image targets by subset")
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--max-gib", type=float, default=20)
    parser.add_argument("--metadata-gib", type=float, default=3)
    args = parser.parse_args(argv)
    build_visual_candidates(
        args.output,
        args.reference_tokenizer,
        targets=json.loads(args.targets) if args.targets else None,
        seed=args.seed,
        max_gib=args.max_gib,
        metadata_gib=args.metadata_gib,
    )


if __name__ == "__main__":
    main()
