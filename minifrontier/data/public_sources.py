"""Pinned public-source sampling with remote Parquet ranges and bounded local output.

Directories and row groups are permuted before reading; this is not first-N
prefix sampling. The manifest records every chosen file/group and admission
count. No model weights or full source datasets are downloaded here.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from minifrontier.data.corpus import CorpusBuilder
from minifrontier.storage import GIB, require_space

SOURCES = {
    "zh_edu": dict(
        repo="opencsg/Fineweb-Edu-Chinese-V2.1",
        revision="a5b574efa48beb3a8f6887ef0b093becf004328b",
        prefix="4_5",
        license="apache-2.0; retain original CCI subsource",
        task="zh_edu",
        lang="zh",
    ),
    "en_edu": dict(
        repo="HuggingFaceFW/fineweb-edu",
        revision="87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
        prefix="sample/10BT",
        license="odc-by; original CommonCrawl page rights retained",
        task="en_edu",
        lang="en",
    ),
    "python_edu": dict(
        repo="HuggingFaceTB/smollm-corpus",
        revision="3ba9d605774198c5868892d7a8deda78031a781f",
        prefix="python-edu",
        license="The Stack v2 per-repository terms; index lacks original license field",
        task="code",
        lang="code",
    ),
    "ultrachat": dict(
        repo="HuggingFaceH4/ultrachat_200k",
        revision="8049631c405ae6576f93f445c6b8166f76f5505a",
        prefix="data",
        file_prefix="train_sft-",
        license="mit",
        task="dialogue",
        lang="en",
    ),
}


def retry_transport(call, *, audit, operation):
    """Retry transient transport failures only; permissions/schema failures remain errors."""
    import httpx

    for attempt in range(3):
        try:
            return call()
        except httpx.TransportError as error:
            audit.setdefault("transport_failures", []).append(
                dict(operation=operation, attempt=attempt + 1, error=type(error).__name__)
            )
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def source_rows(name, *, seed, audit, specification=None, skip_rows=0):
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, HfFileSystem

    source = specification or SOURCES[name]
    if skip_rows < 0:
        raise ValueError("source resume offset must be nonnegative")
    previous_reads = list(audit.get("reads", [])) if skip_rows else []
    audit["resume_skip_rows"] = skip_rows
    audit["files"] = {}
    files = retry_transport(
        lambda: [
            item.path
            for item in HfApi().list_repo_tree(
                source["repo"],
                path_in_repo=source["prefix"],
                revision=source["revision"],
                repo_type="dataset",
            )
            if item.path.endswith(".parquet")
            and Path(item.path).name.startswith(source.get("file_prefix", ""))
        ],
        audit=audit,
        operation="pinned_file_catalog",
    )
    if not files:
        raise ValueError(f"no source Parquet files: {name}")
    rng = random.Random(seed)
    rng.shuffle(files)
    audit["catalog_files"] = len(files)
    audit["catalog_sha256"] = hashlib.sha256("\n".join(sorted(files)).encode()).hexdigest()
    audit["sampling"] = (
        "uniform file permutation, uniform row-group permutation, uniform rows within each group"
    )
    audit["reads"] = []
    fs = HfFileSystem()
    for filename in files:
        remote = f"datasets/{source['repo']}@{source['revision']}/{filename}"
        if source.get("bounded_http_ranges"):
            from minifrontier.data.remote import RangeFile

            info = fs.info(remote)
            stream = RangeFile(
                f"https://huggingface.co/datasets/{source['repo']}/resolve/{source['revision']}/{filename}",
                info["size"],
                chunk_size=1024**2,
                network_budget=2 * info["size"] + 64 * 1024**2,
            )
        else:
            stream = fs.open(remote, block_size=4 * 1024**2)
            info = fs.info(stream.path)
        with stream:
            audit["files"][filename] = {
                key: info[key] for key in ("size", "blob_id", "lfs") if key in info
            }
            if source.get("bounded_http_ranges"):
                audit["files"][filename]["transport_failures"] = stream.transport_failures
            parquet = pq.ParquetFile(stream)
            groups = list(range(parquet.num_row_groups))
            rng.shuffle(groups)
            for group in groups:
                # Bound decompression before allocation. A pathological row group
                # is refused, rather than silently materializing an entire corpus.
                if parquet.metadata.row_group(group).total_byte_size > 512 * 1024**2:
                    raise ValueError("source row group exceeds 512 MiB memory bound")
                count = parquet.metadata.row_group(group).num_rows
                order = list(range(count))
                rng.shuffle(order)
                descriptor = dict(file=filename, row_group=group, rows=count)
                position = len(audit["reads"])
                if position < len(previous_reads) and any(
                    previous_reads[position].get(k) != v for k, v in descriptor.items()
                ):
                    raise ValueError("pinned source sampling order changed during resume")
                if skip_rows >= count:
                    skip_rows -= count
                    audit["reads"].append(descriptor)
                    continue
                table = parquet.read_row_group(group)
                audit["reads"].append(descriptor)
                rows = table.to_pylist()
                remaining, skip_rows = order[skip_rows:], 0
                for index in remaining:
                    yield rows[index], f"{filename}:rg{group}:row{index}"
    if skip_rows:
        raise ValueError("resume offset exceeds the pinned source")


def python_content(item):
    import requests

    row, identity = item
    blob = row["blob_id"]
    try:
        response = requests.get(
            f"https://softwareheritage.s3.amazonaws.com/content/{blob}", timeout=30
        )
        response.raise_for_status()
        if len(response.content) > 2 * 1024**2:
            raise ValueError("code blob exceeds compressed byte limit")
        content = gzip.decompress(response.content)
        if len(content) > 2 * 1024**2 or hashlib.sha1(content).hexdigest() != blob:
            raise ValueError("SWH content length/hash mismatch")
        row = dict(row, text=content.decode("utf-8"), content_url=response.url)
        return row, identity
    except (requests.RequestException, ValueError, UnicodeDecodeError, OSError) as error:
        return dict(download_error=type(error).__name__), identity


def normalized_source(name, row, item):
    spec = SOURCES[name]
    base = dict(
        source=spec["repo"],
        revision=spec["revision"],
        item_id=item,
        group_id=item,
        license=spec["license"],
        lang=spec["lang"],
        task=spec["task"],
        stage="pretrain",
        quality_flags=[],
        source_metadata={},
    )
    if name == "zh_edu":
        # This pinned 4_5 folder has scores near .8, rather than a 0..5 scale.
        score = float(row["score"])
        if not 0.0 <= score <= 1.0:
            raise ValueError("unexpected Chinese source score scale; inspect before filtering")
        base.update(
            text=row["text"],
            source_metadata=dict(score=score, subsource=row["source"], directory="4_5"),
        )
    elif name == "en_edu":
        base.update(
            text=row["text"],
            group_id=row.get("url", row.get("id", item)),
            source_metadata={
                k: row[k] for k in ("url", "id", "score", "int_score", "dump") if k in row
            },
        )
    elif name == "python_edu":
        if row.get("download_error"):
            return None
        base.update(
            text=row["text"],
            repo_id=row["repo_name"],
            group_id=row["repo_name"],
            item_id=row["blob_id"],
            license_status="original-license-unresolved",
            source_metadata={
                k: row[k]
                for k in ("blob_id", "repo_name", "path", "score", "int_score", "content_url")
            },
        )
    elif name == "ultrachat":
        turns = row.get("messages")
        if not turns:
            return None
        base.update(
            turns=turns,
            stage="sft",
            group_id=row.get("prompt_id", item),
            official_split="train_sft",
        )
    return base


def build_public(root, *, limits, seed=42, max_gib=24):
    """limits are accepted normalized UTF-8 payload bytes per source, not tokens."""
    root = Path(root)
    if root.exists():
        raise FileExistsError("choose a fresh immutable corpus version")
    require_space(root, int(max_gib * GIB))
    builder = CorpusBuilder(root, seed=seed, max_gib=max_gib)
    audit = dict(schema_version=1, seed=seed, limits_bytes=limits, sources={}, status="building")
    path = root / "source-audit.json"
    path.write_text(json.dumps(audit, indent=2))
    try:
        for index, (name, budget) in enumerate(limits.items()):
            entry: dict[str, Any] = dict(
                SOURCES[name], accepted_rows=0, accepted_bytes=0, rejected_rows=0
            )
            audit["sources"][name] = entry
            rows = iter(source_rows(name, seed=seed + index * 104729, audit=entry))
            with ThreadPoolExecutor(max_workers=12) as pool:
                while entry["accepted_bytes"] < budget:
                    # map is bounded by this batch even on Python versions without buffersize.
                    batch = []
                    for _ in range(96 if name == "python_edu" else 256):
                        try:
                            batch.append(next(rows))
                        except StopIteration:
                            break
                    if not batch:
                        entry["exhausted"] = True
                        break
                    if name == "python_edu":
                        batch = list(pool.map(python_content, batch))
                    for row, item in batch:
                        record = normalized_source(name, row, item)
                        if record is not None and builder.add(record):
                            entry["accepted_rows"] += 1
                            entry["accepted_bytes"] += len(
                                json.dumps(record, ensure_ascii=False).encode()
                            )
                        else:
                            entry["rejected_rows"] += 1
                        if entry["accepted_bytes"] >= budget:
                            break
                    if entry["accepted_rows"] % 2048 < len(batch):
                        path.write_text(json.dumps(audit, indent=2, default=str))
                        print(
                            json.dumps(
                                dict(
                                    source=name,
                                    rows=entry["accepted_rows"],
                                    mib=round(entry["accepted_bytes"] / 1024**2, 1),
                                )
                            ),
                            flush=True,
                        )
            path.write_text(json.dumps(audit, indent=2, default=str))
        manifest = builder.finalize()
        audit["status"] = "complete"
        audit["corpus"] = manifest
    except BaseException as error:
        builder.db.commit()
        audit["status"] = "interrupted"
        audit["error"] = type(error).__name__ + ": " + str(error)
        raise
    finally:
        path.write_text(json.dumps(audit, indent=2, default=str))
    return audit


def main(argv=None):
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--limits-mib", required=True, help='JSON object, e.g. {"zh_edu":128,"en_edu":64}'
    )
    p.add_argument("--max-gib", type=float, default=24)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    limits = {name: int(value * 1024**2) for name, value in json.loads(args.limits_mib).items()}
    if any(name not in SOURCES or n < 1 for name, n in limits.items()):
        p.error("unknown source or nonpositive byte budget")
    build_public(args.output, limits=limits, seed=args.seed, max_gib=args.max_gib)


if __name__ == "__main__":
    main()
