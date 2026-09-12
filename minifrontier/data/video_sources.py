"""Bounded PE-Video train candidates; no proxy, archive extraction, or admission claim.

The decoder is a separate command so a local PyAV installation can be used without
changing the environment of running trainers. Only eight JPEG frames are retained;
the full original clip is decoded before its complete human caption is accepted.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, cast

REPO = "facebook/PE-Video"
REVISION = "43a297dde47e2036721f259397df04b3c338d002"
LICENSE = "CC-BY-NC-4.0"
GIB = 1024**3
MIB = 1024**2
MAX_CLIP_BYTES = 32 * MIB
QUESTION = "Describe what happens in this video."
CARD_URL = f"https://huggingface.co/datasets/{REPO}/blob/{REVISION}/README.md"


def checksum(value):
    return hashlib.sha256(value).hexdigest()


def file_checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(MIB), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class NetworkBudget:
    """Charge requested bytes before every attempt, including retries and crashes."""

    def __init__(self, limit, *, journal=None):
        self.limit, self.journal = int(limit), Path(journal) if journal else None
        self.lock = threading.Lock()
        self.used = 0
        if self.journal and self.journal.exists():
            self.used = json.loads(self.journal.read_text())["charged_bytes"]

    def charge(self, size):
        with self.lock:
            if size < 0 or self.used + size > self.limit:
                raise ValueError("video network byte budget reached")
            self.used += size
            if self.journal:
                atomic_json(self.journal, dict(charged_bytes=self.used, limit_bytes=self.limit))


def direct_session():
    import requests

    session = requests.Session()
    session.trust_env = False
    session.headers["Accept-Encoding"] = "identity"
    return session


class BoundedRangeFile(io.RawIOBase):
    """Seek across a pinned TAR while fetching only bounded exact HTTP ranges."""

    def __init__(self, url, size, budget, *, session=None, prefetch=64 * 1024):
        super().__init__()
        self.url, self.size, self.budget = url, int(size), budget
        self.session = session if session is not None else direct_session()
        self.position, self.prefetch = 0, prefetch
        self.cache_start, self.cache = 0, b""
        self.resolved_url = url

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2):
            raise ValueError("invalid seek mode")
        position = offset + (self.position if whence == 1 else self.size if whence == 2 else 0)
        if not 0 <= position <= self.size:
            raise ValueError("TAR seek outside pinned file")
        self.position = position
        return position

    def read(self, size=-1):
        if size < 0 or size > MAX_CLIP_BYTES:
            raise ValueError("unbounded or oversized remote read")
        size = min(size, self.size - self.position)
        if not size:
            return b""
        start = self.position
        if not self.cache_start <= start < start + size <= self.cache_start + len(self.cache):
            length = min(max(size, self.prefetch), self.size - start)
            end = start + length - 1
            for attempt in range(3):
                self.budget.charge(length)
                try:
                    with self.session.get(
                        self.resolved_url,
                        headers={"Range": f"bytes={start}-{end}"},
                        timeout=(20, 90),
                        stream=True,
                    ) as response:
                        if response.status_code in (401, 403) and self.resolved_url != self.url:
                            self.resolved_url = self.url
                            continue
                        response.raise_for_status()
                        if response.status_code != 206 or response.headers.get("Content-Range") != (
                            f"bytes {start}-{end}/{self.size}"
                        ):
                            raise ValueError("server did not honor exact bounded HTTP Range")
                        value = response.raw.read(length + 1, decode_content=False)
                        if len(value) != length:
                            raise ValueError("HTTP Range body length differs from request")
                        self.resolved_url = response.url
                        self.cache_start, self.cache = start, value
                        break
                except OSError:
                    if attempt == 2:
                        raise
                    time.sleep(attempt + 1)
            else:
                raise ValueError("could not renew direct signed range URL")
        begin = start - self.cache_start
        self.position += size
        return self.cache[begin : begin + size]

    def close(self):
        self.session.close()
        super().close()


def safe_member(member):
    path = PurePosixPath(member.name)
    if (
        not member.isfile()
        or len(path.parts) != 1
        or not re.fullmatch(r"[0-9]+\.(json|mp4)", member.name)
        or member.size <= 0
    ):
        raise ValueError("unexpected PE-Video TAR member type or path")
    return path.stem, path.suffix


def caption_metadata(metadata, key):
    if not isinstance(metadata, dict) or str(metadata.get("video_id")) != key:
        raise ValueError("video metadata and archive key differ")
    caption = metadata.get("human_caption")
    if not isinstance(caption, str) or not 40 <= len(caption.strip()) <= 16000:
        raise ValueError("missing or out-of-bounds complete human caption")
    if any(c in caption for c in ("\x00", "\ufffd")):
        raise ValueError("invalid caption encoding")
    duration = metadata.get("video_duration_in_s")
    if (
        not isinstance(duration, (int, float))
        or not math.isfinite(duration)
        or not 2 <= duration <= 30
    ):
        raise ValueError("clip duration outside 2-30 seconds")
    count = metadata.get("num_frames")
    if (
        not isinstance(count, (int, float))
        or not math.isfinite(count)
        or count != int(count)
        or not 8 <= count <= 3600
    ):
        raise ValueError("invalid or unbounded source frame count")
    return caption.strip()


def frame_indices(count, frames=8):
    if count < frames or frames < 2:
        raise ValueError("not enough frames for uniform whole-clip sampling")
    return [round(i * (count - 1) / (frames - 1)) for i in range(frames)]


def rgb_hash(image):
    image = image.convert("RGB")
    return checksum(f"{image.width}x{image.height}:RGB\0".encode() + image.tobytes())


def decode_clip(clip, metadata, output):
    """Decode every video frame; retain actual PTS and the first/last source frames."""
    import av
    from PIL import Image

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    caption_metadata(metadata, str(metadata.get("video_id")))
    expected = int(metadata["num_frames"])
    wanted = set(frame_indices(expected))
    retained: list[dict[str, Any]] = []
    times: list[float] = []
    shape: tuple[int, int] | None = None
    count, first, last, period = 0, None, None, None
    retained_bytes = 0
    with av.open(str(clip)) as container:
        if len(container.streams.video) != 1:
            raise ValueError("expected one complete video stream")
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        if not 0 < stream.width * stream.height <= 4096 * 2160:
            raise ValueError("video dimensions exceed decoder bound")
        for frame in container.decode(stream):
            if count >= 3600 or frame.pts is None or frame.time_base is None:
                raise ValueError("unbounded frame stream or missing source timestamp")
            timestamp = float(frame.pts * frame.time_base)
            if (
                not math.isfinite(timestamp)
                or timestamp < 0
                or (last is not None and timestamp <= last)
            ):
                raise ValueError("source timestamps are not strictly increasing")
            first = timestamp if first is None else first
            period = timestamp - last if last is not None else None
            last = timestamp
            if count in wanted:
                image = frame.to_image().convert("RGB")
                shape = shape or image.size
                if image.size != shape:
                    raise ValueError("video dimensions change within a clip")
                original_hash = rgb_hash(image)
                path = output / f"{len(retained):02d}.jpg"
                encoded = io.BytesIO()
                image.save(encoded, format="JPEG", quality=92, subsampling=2, optimize=False)
                binary = encoded.getvalue()
                retained_bytes += len(binary)
                if len(binary) > MIB or retained_bytes > 8 * MIB:
                    raise ValueError("decoded JPEG frames exceed scratch storage bound")
                path.write_bytes(binary)
                with Image.open(path) as stored:
                    stored_hash = rgb_hash(stored)
                retained.append(
                    dict(
                        path=path.name,
                        sha256=checksum(path.read_bytes()),
                        rgb_sha256=stored_hash,
                        original_rgb_sha256=original_hash,
                        source_frame_index=count,
                    )
                )
                times.append(timestamp)
            count += 1
    if count != expected or len(retained) != 8 or first is None or last is None:
        raise ValueError("decoded frame count differs; incomplete clips are not relabeled")
    duration = last - first + (period or 0)
    if not 2 - 1e-6 <= duration <= 30 + 1e-6 or abs(
        duration - metadata["video_duration_in_s"]
    ) > max(0.2, 2 * (period or 0)):
        raise ValueError("decoded complete duration differs from metadata")
    if len({f["original_rgb_sha256"] for f in retained}) < 2:
        raise ValueError("eight sampled source frames have no decoded visual change")
    result = dict(
        frames=retained,
        timestamps=times,
        width=cast(tuple[int, int], shape)[0],
        height=cast(tuple[int, int], shape)[1],
        decoded_frames=count,
        decoded_duration_seconds=duration,
        transform="whole_clip_uniform_index_8_native_dimensions_jpeg92_420_v1",
        decoder=dict(
            pyav=av.__version__, libraries={k: list(v) for k, v in av.library_versions.items()}
        ),
    )
    atomic_json(output / "decoded.json", result)
    return result


def read_catalog(budget):
    url = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/train?limit=1000"
    with direct_session() as session:
        budget.charge(2 * MIB)
        with session.get(url, timeout=(20, 60), stream=True) as response:
            response.raise_for_status()
            if response.headers.get("Link"):
                raise ValueError("pinned train catalog unexpectedly requires pagination")
            raw = response.raw.read(2 * MIB + 1)
            if len(raw) > 2 * MIB:
                raise ValueError("source catalog exceeds metadata read cap")
    result = json.loads(raw)
    if not isinstance(result, list) or not result:
        raise ValueError("empty source catalog")
    for row in result:
        if (
            row.get("type") != "file"
            or not re.fullmatch(r"train/[0-9]{6}\.tar", row.get("path", ""))
            or row.get("size", 0) <= 0
        ):
            raise ValueError("unexpected path in pinned train catalog")
    return sorted(result, key=lambda row: row["path"])


class Shard:
    def __init__(self, row, budget, temporary, decoder_python, offset=0):
        self.row, self.temporary, self.decoder_python = row, temporary, decoder_python
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{row['path']}"
        self.remote = BoundedRangeFile(url, row["size"], budget)
        self.remote.seek(offset)
        self.archive = tarfile.open(fileobj=self.remote, mode="r:")  # noqa: SIM115

    def next_clip(self):
        rejected: Counter[str] = Counter()
        while (member := self.archive.next()) is not None:
            key, suffix = safe_member(member)
            if suffix != ".json" or member.size > 64 * 1024:
                raise ValueError("expected bounded JSON before its paired MP4")
            raw = cast(BinaryIO, self.archive.extractfile(member)).read()
            metadata = json.loads(raw)
            video = self.archive.next()
            if video is None or safe_member(video) != (key, ".mp4"):
                raise ValueError("JSON/MP4 archive pair is not aligned")
            cursor = self.archive.offset
            try:
                caption = caption_metadata(metadata, key)
                if video.size > MAX_CLIP_BYTES:
                    raise ValueError("original clip exceeds 32 MiB")
            except ValueError as error:
                rejected[str(error)] += 1
                continue
            directory = Path(tempfile.mkdtemp(prefix=f"{key}-", dir=self.temporary))
            try:
                blob = cast(BinaryIO, self.archive.extractfile(video)).read()
                clip = directory / "source.mp4"
                clip.write_bytes(blob)
                env = {k: v for k, v in os.environ.items() if not k.lower().endswith("_proxy")}
                env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
                atomic_json(directory / "source.json", metadata)
                command = [
                    self.decoder_python,
                    str(Path(__file__).resolve()),
                    "decode",
                    "--clip",
                    str(clip),
                    "--metadata",
                    str(directory / "source.json"),
                    "--output",
                    str(directory / "frames"),
                ]
                completed = subprocess.run(
                    command, env=env, capture_output=True, text=True, timeout=90
                )
                if completed.returncode:
                    rejected[
                        "decode_rejected:" + completed.stderr.strip().split("\n")[-1][:160]
                    ] += 1
                    shutil.rmtree(directory)
                    continue
                decoded = json.loads((directory / "frames" / "decoded.json").read_text())
                return dict(
                    key=key,
                    caption=caption,
                    metadata=metadata,
                    raw_metadata_sha256=checksum(raw),
                    source_sha256=checksum(blob),
                    source_bytes=len(blob),
                    directory=str(directory),
                    decoded=decoded,
                    cursor=cursor,
                    rejected=dict(rejected),
                )
            except (subprocess.TimeoutExpired, ValueError) as error:
                shutil.rmtree(directory)
                rejected[type(error).__name__] += 1
        return dict(cursor=self.row["size"], exhausted=True, rejected=dict(rejected))

    def close(self):
        self.archive.close()
        self.remote.close()


def corpus_record(item, shard):
    decoded, key = item["decoded"], item["key"]
    prefix = f"frames/{key}"
    hashes = [f["rgb_sha256"] for f in decoded["frames"]]
    media = dict(
        kind="video",
        video_id=f"{REPO}:{key}",
        frames=[f"{prefix}/{f['path']}" for f in decoded["frames"]],
        frame_sha256=[f["sha256"] for f in decoded["frames"]],
        frame_rgb_sha256=hashes,
        original_frame_rgb_sha256=[f["original_rgb_sha256"] for f in decoded["frames"]],
        source_frame_indices=[f["source_frame_index"] for f in decoded["frames"]],
        timestamps=decoded["timestamps"],
        width=decoded["width"],
        height=decoded["height"],
        sha256=item["source_sha256"],
        rgb_sha256=hashes[0],
        clip_rgb_sha256=checksum(json.dumps(hashes).encode()),
        frame_transform=decoded["transform"],
    )
    return dict(
        source=REPO,
        revision=REVISION,
        item_id=key,
        group_id=f"{REPO}:{key}",
        official_split="train",
        license=LICENSE,
        lang="en",
        task="video",
        stage="pretrain",
        text=f"<|video|>\nUser: {QUESTION}\nAssistant: {item['caption']}",
        media=[media],
        visual_question=QUESTION,
        visual_answer=item["caption"],
        turns=[
            dict(role="user", content=QUESTION),
            dict(role="assistant", content=item["caption"]),
        ],
        source_metadata=dict(
            archive=shard["path"],
            archive_lfs_sha256=shard.get("lfs", {}).get("oid"),
            archive_bytes=shard["size"],
            source_bytes=item["source_bytes"],
            metadata_sha256=item["raw_metadata_sha256"],
            original=item["metadata"],
            label_field="human_caption",
            decoded_frames=decoded["decoded_frames"],
            decoded_duration_seconds=decoded["decoded_duration_seconds"],
            decoder=decoded["decoder"],
        ),
        quality_flags=[
            "candidate-not-formally-admitted",
            "noncommercial-source",
            "cross-corpus-near-duplicate-audit-pending",
        ],
        license_record=dict(
            identifier=LICENSE,
            dataset=REPO,
            revision=REVISION,
            source_card=CARD_URL,
            attribution="Meta / PE-Video contributors",
            scope="source-declared dataset terms; underlying footage rights not independently established",
            distribution="not included in Apache-2.0 code or wheel; no weight release permission asserted",
        ),
    )


def count_train_groups(builder):
    """Preview the corpus connected-group split before closing the candidate set."""
    parents: dict[str, str] = {}
    first: dict[str, str] = {}

    def find(identity):
        parents.setdefault(identity, identity)
        while parents[identity] != identity:
            parents[identity] = parents[parents[identity]]
            identity = parents[identity]
        return identity

    for identity, key in builder.db.execute("SELECT id,key FROM links ORDER BY key,id"):
        a = find(identity)
        if key in first:
            b = find(first[key])
            parents[max(a, b)] = min(a, b)
        else:
            first[key] = identity
    roots = {find(identity) for identity in parents}
    return sum(
        int(hashlib.sha256(f"{builder.seed}:{root}".encode()).hexdigest()[:16], 16) % 10000
        >= builder.val_buckets + builder.test_buckets
        for root in roots
    )


def write_source_audit(root):
    """Expose a finalized candidate through the existing canonical encoder contract."""
    root = Path(root)
    manifest_path, candidate_path = root / "corpus-manifest.json", root / "candidate-audit.json"
    manifest = json.loads(manifest_path.read_text())
    candidate = json.loads(candidate_path.read_text())
    if candidate.get("status") not in {
        "candidate_complete",
        "insufficient_train_groups",
        "source_exhausted",
    }:
        raise ValueError("video construction has not finished")
    if file_checksum(root / "corpus.sqlite") != manifest["database_sha256"]:
        raise ValueError("final video corpus database checksum differs")
    result = dict(
        schema_version=1,
        source=REPO,
        revision=REVISION,
        status="candidate_slice_complete_pending_admission"
        if candidate.get("target_met")
        else "candidate_inventory_below_target",
        formal_admission=False,
        candidate_only=True,
        video_ce_ready=False,
        license=LICENSE,
        source_card=CARD_URL,
        corpus_manifest_sha256=checksum(manifest_path.read_bytes()),
        candidate_audit_sha256=checksum(candidate_path.read_bytes()),
        producer_config_sha256=checksum((root / "producer-config.json").read_bytes()),
        independent_train_groups=candidate["independent_train_groups"],
        corpus=manifest,
        media_bytes=candidate["frame_bytes"],
        provenance="Pinned official train files; source-declared noncommercial terms retained; underlying footage rights not independently established",
    )
    destination = root / "source-audit.json"
    if destination.exists() and json.loads(destination.read_text()) != result:
        raise FileExistsError("final video source audit is immutable")
    atomic_json(destination, result)
    return result


def produce(args):
    from minifrontier.data.corpus import CorpusBuilder
    from minifrontier.storage import require_space, reserve_write

    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    require_space(root, (args.workers + 1) * 40 * MIB, reserve_bytes=80 * GIB)
    config = dict(
        schema_version=1,
        repo=REPO,
        revision=REVISION,
        official_split="train",
        seed=args.seed,
        target_clips=args.target_clips,
        min_train_groups=args.min_train_groups,
        max_clips=args.max_clips,
        metadata_gib=args.metadata_gib,
        frame_gib=args.frame_gib,
        network_gib=args.network_gib,
        workers=args.workers,
        decoder_python=str(Path(args.decoder_python).resolve()),
        min_free_gib=80,
        max_clip_bytes=MAX_CLIP_BYTES,
        duration_seconds=[2, 30],
        frames=8,
        label="complete human_caption; original clip fully decoded; no temporal truncation",
        source_code_sha256=checksum(Path(__file__).read_bytes()),
    )
    config_path = root / "producer-config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("resume configuration/source changed; select a new candidate version")
    atomic_json(config_path, config)
    budget = NetworkBudget(args.network_gib * GIB, journal=root / "network-budget.json")
    catalog_path = root / "source-catalog.json"
    catalog = (
        json.loads(catalog_path.read_text()) if catalog_path.exists() else read_catalog(budget)
    )
    atomic_json(catalog_path, catalog)
    random.Random(args.seed).shuffle(catalog)
    audit_path = root / "candidate-audit.json"
    audit: dict[str, Any] = (
        json.loads(audit_path.read_text())
        if audit_path.exists()
        else dict(
            status="running",
            candidate_only=True,
            training_admitted=False,
            video_ce_ready=False,
            started_at=time.time(),
            accepted_clips=0,
            frame_bytes=0,
            cursors={},
            rejected={},
            license=LICENSE,
            source_card=CARD_URL,
            completed_shards=[],
        )
    )
    builder = CorpusBuilder(
        root, seed=args.seed, max_gib=args.metadata_gib / 2, group_image_phash=False
    )
    seen, media_hashes, frame_bytes = set(), set(), 0
    for (payload,) in builder.db.execute("SELECT payload FROM samples"):
        record = json.loads(payload)
        seen.add(record["item_id"])
        media_hashes.add(record["media"][0]["clip_rgb_sha256"])
        for relative, expected in zip(
            record["media"][0]["frames"], record["media"][0]["frame_sha256"], strict=True
        ):
            path = root / relative
            if checksum(path.read_bytes()) != expected:
                raise ValueError("retained candidate frame changed")
            frame_bytes += path.stat().st_size
    builder.counts = Counter(audit.get("dedup_counts", {}))
    builder.counts["accepted"] = len(seen)
    audit.update(accepted_clips=len(seen), frame_bytes=frame_bytes, status="running")
    audit.pop("error", None)
    frames_root = root / "frames"
    frames_root.mkdir(exist_ok=True)
    # A crash may leave a single uncommitted clip directory. Only numeric producer
    # paths without retained records are owned by this recovery operation.
    for child in frames_root.iterdir():
        if child.is_dir() and child.name.isdecimal() and child.name not in seen:
            shutil.rmtree(child)
    temporary = root / ".temporary"
    temporary.mkdir(exist_ok=True)
    pending = [row for row in catalog if row["path"] not in audit["completed_shards"]]
    rejected: Counter[str] = Counter(audit["rejected"])
    active = []

    def save():
        audit.update(
            accepted_clips=len(seen),
            frame_bytes=frame_bytes,
            rejected=dict(rejected),
            network_charged_bytes=budget.used,
            updated_at=time.time(),
            dedup_counts=dict(builder.counts),
        )
        atomic_json(audit_path, audit)

    def start_shard():
        if pending:
            row = pending.pop(0)
            return Shard(
                row, budget, temporary, args.decoder_python, audit["cursors"].get(row["path"], 0)
            )
        return None

    try:
        active = [start_shard() for _ in range(min(args.workers, len(pending)))]
        save()
        next_group_check = args.target_clips
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            while active and len(seen) < args.max_clips:
                if len(seen) >= next_group_check:
                    train_groups = count_train_groups(builder)
                    audit["independent_train_groups_preview"] = train_groups
                    save()
                    if train_groups >= args.min_train_groups:
                        break
                    next_group_check = min(
                        args.max_clips,
                        len(seen) + max(32, args.min_train_groups - train_groups + 32),
                    )
                require_space(root, (args.workers + 1) * 40 * MIB, reserve_bytes=80 * GIB)
                # Consume in stable shard order; completion timing never selects data.
                futures = [pool.submit(shard.next_clip) for shard in active]
                next_active = []
                for shard, future in zip(active, futures, strict=True):
                    item = future.result()
                    rejected.update(item["rejected"])
                    if item.get("exhausted"):
                        audit["completed_shards"].append(shard.row["path"])
                        shard.close()
                        replacement = start_shard()
                        if replacement:
                            next_active.append(replacement)
                        save()
                        continue
                    directory = Path(item["directory"])
                    try:
                        if len(seen) >= args.max_clips:
                            next_active.append(shard)
                            continue
                        record = corpus_record(item, shard.row)
                        size = sum(
                            (directory / "frames" / f["path"]).stat().st_size
                            for f in item["decoded"]["frames"]
                        )
                        if (
                            item["key"] in seen
                            or record["media"][0]["clip_rgb_sha256"] in media_hashes
                        ):
                            rejected["duplicate_source_or_decoded_clip"] += 1
                        elif frame_bytes + size > args.frame_gib * GIB:
                            raise ValueError("retained video frame storage budget reached")
                        else:
                            destination = frames_root / item["key"]
                            (directory / "frames" / "decoded.json").unlink()
                            with reserve_write(destination, size, reserve_bytes=80 * GIB):
                                (directory / "frames").replace(destination)
                            if builder.add(record):
                                builder.db.commit()
                                seen.add(item["key"])
                                media_hashes.add(record["media"][0]["clip_rgb_sha256"])
                                frame_bytes += size
                            else:
                                shutil.rmtree(destination)
                        audit["cursors"][shard.row["path"]] = item["cursor"]
                        next_active.append(shard)
                        save()
                        if len(seen) % 25 == 0:
                            print(
                                json.dumps(
                                    {
                                        k: audit[k]
                                        for k in (
                                            "accepted_clips",
                                            "frame_bytes",
                                            "network_charged_bytes",
                                            "rejected",
                                        )
                                    }
                                ),
                                flush=True,
                            )
                    finally:
                        shutil.rmtree(directory, ignore_errors=True)
                active = next_active
        metadata_bytes = sum(p.stat().st_size for p in root.iterdir() if p.is_file())
        peak = metadata_bytes + (root / "corpus.sqlite").stat().st_size + 16 * MIB
        if peak > args.metadata_gib * GIB:
            raise ValueError("metadata finalization would exceed its storage budget")
        require_space(root, peak - metadata_bytes, reserve_bytes=80 * GIB)
        audit["metadata_storage"] = dict(
            before_finalize_bytes=metadata_bytes,
            finalization_peak_bound_bytes=peak,
            budget_bytes=int(args.metadata_gib * GIB),
        )
        manifest = builder.finalize()
        train_clips = builder.db.execute(
            "SELECT COUNT(DISTINCT group_root) FROM samples WHERE split='train'"
        ).fetchone()[0]
        audit.update(
            status="candidate_complete"
            if train_clips >= args.min_train_groups
            else "insufficient_train_groups",
            finalized_at=time.time(),
            independent_train_groups=train_clips,
            splits=manifest["splits"],
            target_met=train_clips >= args.min_train_groups,
            minimum_train_groups=args.min_train_groups,
        )
        save()
        write_source_audit(root)
    except BaseException as error:
        builder.db.commit()
        audit.update(status="interrupted", error=f"{type(error).__name__}: {error}")
        save()
        raise
    finally:
        for shard in active:
            shard.close()
        builder.db.close()
        # No original MP4 remains in the candidate cache, including failed jobs.
        shutil.rmtree(temporary, ignore_errors=True)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    decoder = sub.add_parser("decode")
    decoder.add_argument("--clip", required=True)
    decoder.add_argument("--metadata", required=True)
    decoder.add_argument("--output", required=True)
    audit_parser = sub.add_parser("finalize-audit")
    audit_parser.add_argument("--output", required=True)
    producer = sub.add_parser("produce")
    producer.add_argument("--output", required=True)
    producer.add_argument("--decoder-python", default=sys.executable)
    producer.add_argument("--seed", type=int, default=20260912)
    producer.add_argument(
        "--target-clips",
        type=int,
        default=10200,
        help="initial candidate count before checking actual train groups",
    )
    producer.add_argument("--min-train-groups", type=int, default=10000)
    producer.add_argument("--max-clips", type=int, default=12000)
    producer.add_argument("--workers", type=int, default=4)
    producer.add_argument("--metadata-gib", type=float, default=3)
    producer.add_argument("--frame-gib", type=float, default=6)
    producer.add_argument("--network-gib", type=float, default=50)
    args = parser.parse_args()
    if args.command == "decode":
        decode_clip(args.clip, json.loads(Path(args.metadata).read_text()), args.output)
    elif args.command == "finalize-audit":
        print(json.dumps(write_source_audit(args.output), indent=2))
    else:
        if (
            min(
                args.target_clips, args.workers, args.metadata_gib, args.frame_gib, args.network_gib
            )
            <= 0
            or args.workers > 8
            or not 0 <= args.min_train_groups <= args.max_clips
            or args.target_clips > args.max_clips
        ):
            parser.error("positive budgets required; at most 8 decoder workers")
        Path(args.output).mkdir(parents=True, exist_ok=True)
        with (Path(args.output) / ".producer.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            print(json.dumps(produce(args), indent=2))


if __name__ == "__main__":
    main()
