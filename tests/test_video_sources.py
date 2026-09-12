"""Video candidates keep bounded transport and whole-clip caption provenance."""

import io
import tarfile
from types import SimpleNamespace

import pytest
from PIL import Image

from minifrontier.data.video_sources import (
    MAX_CLIP_BYTES,
    BoundedRangeFile,
    NetworkBudget,
    caption_metadata,
    corpus_record,
    direct_session,
    frame_indices,
    rgb_hash,
    safe_member,
)


class Response:
    def __init__(self, value, headers, status=206):
        self.raw = SimpleNamespace(read=lambda count, **kw: value[:count])
        self.headers, self.status_code = headers, status
        self.url = "https://cdn.example/immutable"

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        pass


class Session:
    def __init__(self, response):
        self.response, self.requests = response, []

    def get(self, url, **kwargs):
        self.requests.append((url, kwargs))
        return self.response

    def close(self):
        pass


def test_exact_range_and_precharged_budget(tmp_path):
    budget = NetworkBudget(8, journal=tmp_path / "budget.json")
    session = Session(Response(b"abcdefgh", {"Content-Range": "bytes 0-7/100"}))
    with BoundedRangeFile(
        "https://example/file", 100, budget, session=session, prefetch=8
    ) as source:
        assert source.read(3) == b"abc"
        assert source.read(5) == b"defgh"
        assert len(session.requests) == 1
        assert session.requests[0][1]["headers"] == {"Range": "bytes=0-7"}
        with pytest.raises(ValueError, match="network byte budget"):
            source.read(1)
        assert len(session.requests) == 1
        for offset in (-1, 101):
            with pytest.raises(ValueError, match="seek outside"):
                source.seek(offset)
        with pytest.raises(ValueError, match="unbounded"):
            source.read()
        with pytest.raises(ValueError, match="oversized"):
            source.read(MAX_CLIP_BYTES + 1)
    assert NetworkBudget(8, journal=tmp_path / "budget.json").used == 8


@pytest.mark.parametrize(
    "status,content_range,value",
    [
        (200, "bytes 0-7/100", b"abcdefgh"),
        (206, "bytes 0-7/99", b"abcdefgh"),
        (206, "bytes 1-8/100", b"abcdefgh"),
        (206, "bytes 0-7/100", b"short"),
        (206, "bytes 0-7/100", b"abcdefghi"),
    ],
)
def test_refuses_nonexact_or_unbounded_server_response(status, content_range, value):
    session = Session(Response(value, {"Content-Range": content_range}, status))
    budget = NetworkBudget(8)
    with (
        BoundedRangeFile(
            "https://example/file", 100, budget, session=session, prefetch=8
        ) as source,
        pytest.raises(ValueError, match="Range"),
    ):
        source.read(8)
    assert budget.used == 8


def test_http_ignores_environment_proxy(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://must-not-be-used.invalid:1")
    with direct_session() as session:
        assert session.trust_env is False


@pytest.mark.parametrize(
    "name,kind",
    [
        ("../1.mp4", tarfile.REGTYPE),
        ("/1.mp4", tarfile.REGTYPE),
        ("nested/1.json", tarfile.REGTYPE),
        ("1.mp4", tarfile.SYMTYPE),
        ("1.mp4", tarfile.LNKTYPE),
        ("1.txt", tarfile.REGTYPE),
    ],
)
def test_never_extracts_unsafe_archive_paths(name, kind):
    member = tarfile.TarInfo(name)
    member.size, member.type = 12, kind
    with pytest.raises(ValueError, match="member type or path"):
        safe_member(member)


def test_pax_names_are_checked_after_tar_resolution():
    value = io.BytesIO()
    with tarfile.open(fileobj=value, mode="w", format=tarfile.PAX_FORMAT) as archive:
        member = tarfile.TarInfo("42.json")
        member.size, member.pax_headers = 2, {"mtime": "0.1"}
        archive.addfile(member, io.BytesIO(b"{}"))
    value.seek(0)
    with tarfile.open(fileobj=value, mode="r:") as archive:
        assert safe_member(archive.next()) == ("42", ".json")


def test_caption_matches_complete_clip_and_human_field():
    metadata = dict(
        video_id=42,
        human_caption="A person walks across a room, picks up a cup and leaves.",
        model_caption="Must not silently replace the human annotation.",
        video_duration_in_s=10,
        num_frames=300,
    )
    assert caption_metadata(metadata, "42") == metadata["human_caption"]
    for update in (
        {"video_id": 99},
        {"human_caption": None},
        {"video_duration_in_s": 31},
        {"video_duration_in_s": float("nan")},
        {"num_frames": 300.2},
    ):
        with pytest.raises(ValueError):
            caption_metadata(metadata | update, "42")
    assert frame_indices(300) == [0, 43, 85, 128, 171, 214, 256, 299]
    with pytest.raises(ValueError):
        frame_indices(7)


def test_video_corpus_contract_and_decoded_hashes(tmp_path):
    from minifrontier.data.corpus import CorpusBuilder

    frames = []
    for i in range(8):
        path = tmp_path / f"{i:02d}.jpg"
        Image.new("RGB", (16, 16), (i * 20, 0, 0)).save(path)
        with Image.open(path) as image:
            digest = rgb_hash(image)
        frames.append(
            dict(
                path=path.name,
                sha256="a" * 64,
                rgb_sha256=digest,
                original_rgb_sha256="b" * 64,
                source_frame_index=i,
            )
        )
    item = dict(
        key="42",
        caption="A person walks across a room, picks up a cup and leaves.",
        metadata=dict(video_id=42),
        raw_metadata_sha256="c" * 64,
        source_sha256="d" * 64,
        source_bytes=123,
        decoded=dict(
            frames=frames,
            timestamps=list(range(8)),
            width=16,
            height=16,
            transform="test",
            decoded_frames=8,
            decoded_duration_seconds=8,
            decoder={},
        ),
    )
    record = corpus_record(item, dict(path="train/000000.tar", size=456))
    assert record["license"] == "CC-BY-NC-4.0"
    assert record["media"][0]["rgb_sha256"] == record["media"][0]["frame_rgb_sha256"][0]
    assert record["media"][0]["clip_rgb_sha256"] != record["media"][0]["rgb_sha256"]
    assert record["turns"][1]["content"] == item["caption"]
    assert record["group_id"] == record["media"][0]["video_id"]
    assert record["media"][0]["frames"] == [f"frames/42/{i:02d}.jpg" for i in range(8)]
    builder = CorpusBuilder(tmp_path / "corpus", max_gib=0.01, group_image_phash=False)
    try:
        assert builder.add(record)
        links = [key for (key,) in builder.db.execute("SELECT key FROM links")]
        assert all("rgb:" + f["rgb_sha256"] in links for f in frames)
        manifest = builder.finalize()
        assert sum(manifest["splits"].values()) == 1
        assert manifest["image_phash_grouping"] is False
    finally:
        builder.db.close()


@pytest.mark.parametrize("decoded_count", [80, 79])
def test_decoder_keeps_actual_pts_and_refuses_truncated_labels(
    tmp_path, monkeypatch, decoded_count
):
    import sys
    from fractions import Fraction

    from minifrontier.data.video_sources import decode_clip

    class Frame:
        def __init__(self, index):
            self.pts, self.time_base, self.index = index, Fraction(1, 25), index

        def to_image(self):
            return Image.new("RGB", (32, 24), (self.index * 3, self.index, 0))

    class Container:
        streams = SimpleNamespace(
            video=[SimpleNamespace(width=32, height=24, codec_context=SimpleNamespace())]
        )

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def decode(self, stream):
            yield from (Frame(i) for i in range(decoded_count))

    monkeypatch.setitem(
        sys.modules,
        "av",
        SimpleNamespace(open=lambda path: Container(), __version__="test", library_versions={}),
    )
    metadata = dict(
        video_id=42,
        human_caption="A person walks across a room, picks up a cup and leaves.",
        video_duration_in_s=3.2,
        num_frames=80,
    )
    if decoded_count != 80:
        with pytest.raises(ValueError, match="incomplete clips"):
            decode_clip(tmp_path / "source.mp4", metadata, tmp_path / "frames")
        return
    result = decode_clip(tmp_path / "source.mp4", metadata, tmp_path / "frames")
    assert result["timestamps"] == [index / 25 for index in frame_indices(80)]
    assert result["decoded_frames"] == 80
    assert result["decoded_duration_seconds"] == pytest.approx(3.2)
    assert len(result["frames"]) == 8
    for frame, index in zip(result["frames"], frame_indices(80), strict=True):
        assert frame["source_frame_index"] == index
        assert frame["original_rgb_sha256"] == rgb_hash(Frame(index).to_image())
        with Image.open(tmp_path / "frames" / frame["path"]) as image:
            assert frame["rgb_sha256"] == rgb_hash(image)


def test_train_minimum_counts_connected_groups_and_excludes_holdout(tmp_path):
    from minifrontier.data.corpus import CorpusBuilder
    from minifrontier.data.video_sources import count_train_groups

    builder = CorpusBuilder(
        tmp_path / "corpus",
        seed=13,
        max_gib=0.01,
        val_buckets=1500,
        test_buckets=1500,
        group_image_phash=False,
    )
    try:
        for i in range(40):
            record = dict(
                source="test",
                revision="fixed",
                item_id=str(i),
                group_id=str(i // 2),
                license="test",
                lang="en",
                task="text",
                stage="pretrain",
                text=f"This document has a distinct identifying number {i} and substantial content {i * i}.",
            )
            builder.add(record)
        count = count_train_groups(builder)
        builder.finalize()
        measured = builder.db.execute(
            "SELECT COUNT(DISTINCT group_root) FROM samples WHERE split='train'"
        ).fetchone()[0]
        assert count == measured
        assert 0 < measured < 20
        assert (
            measured
            < builder.db.execute("SELECT COUNT(*) FROM samples WHERE split='train'").fetchone()[0]
        )
    finally:
        builder.db.close()


def test_source_audit_only_publishes_closed_unchanged_candidates(tmp_path):
    import hashlib
    import json

    from minifrontier.data.video_sources import write_source_audit

    (tmp_path / "corpus.sqlite").write_bytes(b"immutable database")
    manifest = dict(
        database_sha256=hashlib.sha256(b"immutable database").hexdigest(), splits={"train": 10000}
    )
    (tmp_path / "corpus-manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "producer-config.json").write_text("{}")
    candidate = dict(
        status="running", independent_train_groups=10000, frame_bytes=123, target_met=True
    )
    candidate_path = tmp_path / "candidate-audit.json"
    candidate_path.write_text(json.dumps(candidate))
    with pytest.raises(ValueError, match="has not finished"):
        write_source_audit(tmp_path)
    candidate["status"] = "candidate_complete"
    candidate_path.write_text(json.dumps(candidate))
    result = write_source_audit(tmp_path)
    assert result["status"] == "candidate_slice_complete_pending_admission"
    assert result["formal_admission"] is False
    assert (
        result["candidate_audit_sha256"] == hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    )
    assert write_source_audit(tmp_path) == result
    candidate["frame_bytes"] = 124
    candidate_path.write_text(json.dumps(candidate))
    with pytest.raises(FileExistsError, match="immutable"):
        write_source_audit(tmp_path)
    (tmp_path / "corpus.sqlite").write_bytes(b"modified database")
    with pytest.raises(ValueError, match="checksum differs"):
        write_source_audit(tmp_path)
