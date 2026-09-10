"""Retained candidate rows survive source failures without counting replay twice."""

import gzip
import hashlib
import io
import json
import shutil
from types import SimpleNamespace

import pytest
import requests
from test_code_sources import APACHE
from test_code_sources import row as code_row
from test_visual_candidates import row as image_row
from tokenizers import Tokenizer, models, pre_tokenizers

from minifrontier.data import pretraining, public_sources, visual_sources
from minifrontier.data.code_sources import CODE_SOURCE, code_rows
from minifrontier.data.remote import RangeFile


@pytest.fixture
def reference(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(free=900 * 1024**3))
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))
    return path


def mock_code_files(monkeypatch, rows):
    import huggingface_hub

    contents = {f"file-{i}.json.gz": gzip.compress(raw) for i, raw in enumerate(rows)}
    files = [
        SimpleNamespace(
            rfilename=k, size=len(v), lfs=SimpleNamespace(sha256=hashlib.sha256(v).hexdigest())
        )
        for k, v in contents.items()
    ]
    monkeypatch.setattr(
        huggingface_hub,
        "HfApi",
        lambda token: SimpleNamespace(
            dataset_info=lambda *a, **k: SimpleNamespace(
                sha=CODE_SOURCE["revision"], gated=False, siblings=files
            )
        ),
    )

    class Response:
        def __init__(self, content):
            self.content = content

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        def iter_content(self, _):
            yield self.content

    def get(_self, url, **kwargs):
        return Response(contents[url.rsplit("/", 1)[1]])

    monkeypatch.setattr(requests.Session, "get", get)
    return get


def test_code_resume_retains_rows_and_accounts_for_replayed_network(
    reference, tmp_path, monkeypatch
):
    get = mock_code_files(
        monkeypatch,
        [
            json.dumps(code_row(APACHE + f"\ndef value():\n    return {i}\n")).encode() + b"\n"
            for i in range(2)
        ],
    )
    calls = []

    def interrupted(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise requests.ReadTimeout("source body interrupted")
        return get(*args, **kwargs)

    monkeypatch.setattr(requests.Session, "get", interrupted)
    output = tmp_path / "code"
    args = dict(targets={"code_licensed": 10000}, seed=17, max_gib=4)
    with pytest.raises(requests.ReadTimeout):
        pretraining.build_text_slice(output, reference, **args)
    before = json.loads((output / "source-audit.json").read_text())
    assert before["sources"]["code_licensed"]["accepted_records"] == 1
    monkeypatch.setattr(requests.Session, "get", get)
    after = pretraining.build_text_slice(output, reference, resume=True, **args)
    assert after["dedup_and_quality_counts"] == {"accepted": 2}
    assert after["sources"]["code_licensed"]["read_rows"] == 2
    assert (
        after["sources"]["code_licensed"]["downloaded_bytes"]
        > before["sources"]["code_licensed"]["downloaded_bytes"]
    )
    assert after["resumes"][0]["retained_records"] == 1
    assert after["status"] == "candidate_inventory_below_target" and not after["formal_admission"]
    assert not (output / ".code-source.json.gz").exists()


def test_large_code_json_is_drained_without_truncating_the_next_record(
    reference, tmp_path, monkeypatch
):
    large = json.dumps(dict(content="x" * (2 * 1024**2 + 64))).encode()
    mock_code_files(monkeypatch, [large + b"\n" + json.dumps(code_row()).encode() + b"\n"])
    audit = {}
    rows = list(code_rows(root=tmp_path, seed=42, audit=audit))
    assert len(rows) == 1 and rows[0][1].endswith(":line2")
    assert rows[0][0] == code_row()
    assert audit["reader_rejections"] == {"oversized_json_record": 1}
    assert audit["read_rows"] == audit["completed_read_rows"] == 2


def test_visual_resume_verifies_retained_pixels_and_preserves_counts(
    reference, tmp_path, monkeypatch
):
    calls = []

    def rows(name, *, seed, audit, specification, skip_rows=0):
        calls.append(skip_rows)
        if not skip_rows:
            yield image_row(1), "file:rg0:row0"
            raise requests.ConnectionError("source transport interrupted")
        assert skip_rows == 1
        yield image_row(2), "file:rg0:row1"

    monkeypatch.setattr(visual_sources, "source_rows", rows)
    output = tmp_path / "visual"
    args = dict(targets={"allava_laion": 2})
    with pytest.raises(requests.ConnectionError):
        visual_sources.build_visual_candidates(output, reference, **args)
    before = json.loads((output / "source-audit.json").read_text())
    after = visual_sources.build_visual_candidates(output, reference, resume=True, **args)
    assert before["unique_images"] == 1 and after["unique_images"] == 2
    assert after["dedup_counts"] == {"accepted": 2}
    assert after["sources"]["allava_laion"]["read_rows"] == 2
    assert after["media_bytes"] == sum(
        p.stat().st_size for p in (output / "images").rglob("*.image")
    )
    assert calls == [0, 1] and not after["formal_admission"]


def test_parquet_resume_skips_completed_groups_without_materializing_them(monkeypatch):
    import huggingface_hub
    import pyarrow as pa
    import pyarrow.parquet as pq

    stream = io.BytesIO()
    pq.write_table(pa.table({"value": list(range(9))}), stream, row_group_size=3)
    content = stream.getvalue()

    class Remote(io.BytesIO):
        path = "fixture.parquet"

    monkeypatch.setattr(
        huggingface_hub,
        "HfApi",
        lambda: SimpleNamespace(
            list_repo_tree=lambda *a, **k: [SimpleNamespace(path="fixture.parquet")]
        ),
    )
    monkeypatch.setattr(
        huggingface_hub,
        "HfFileSystem",
        lambda: SimpleNamespace(
            open=lambda *a, **k: Remote(content), info=lambda p: dict(size=len(content))
        ),
    )
    spec = dict(repo="fixture", revision="fixed", prefix="")
    original = pq.ParquetFile
    reads = []

    class Counted:
        def __init__(self, stream):
            self.file = original(stream)
            self.metadata = self.file.metadata
            self.num_row_groups = self.file.num_row_groups

        def read_row_group(self, group):
            reads.append(group)
            return self.file.read_row_group(group)

    monkeypatch.setattr(pq, "ParquetFile", Counted)
    audit = {}
    whole = list(public_sources.source_rows("fixture", seed=13, audit=audit, specification=spec))
    reads.clear()
    resumed = list(
        public_sources.source_rows("fixture", seed=13, audit=audit, specification=spec, skip_rows=4)
    )
    assert resumed == whole[4:] and len(reads) == 2
    audit["catalog_sha256"] = "0" * 64
    reads.clear()
    with pytest.raises(ValueError, match="catalog changed"):
        list(
            public_sources.source_rows(
                "fixture", seed=13, audit=audit, specification=spec, skip_rows=4
            )
        )
    assert not reads


def test_visual_resume_rejects_corrupt_retained_media_before_reading_source(
    reference, tmp_path, monkeypatch
):
    def rows(*args, **kwargs):
        yield image_row(1), "file:rg0:row0"
        raise requests.ConnectionError("source transport interrupted")

    monkeypatch.setattr(visual_sources, "source_rows", rows)
    output = tmp_path / "visual"
    args = dict(targets={"allava_laion": 2})
    with pytest.raises(requests.ConnectionError):
        visual_sources.build_visual_candidates(output, reference, **args)
    audit_before = (output / "source-audit.json").read_bytes()
    next((output / "images").rglob("*.image")).write_bytes(b"corrupted")

    def must_not_read(*args, **kwargs):
        pytest.fail("resume read source before validating retained media")

    monkeypatch.setattr(visual_sources, "source_rows", must_not_read)
    with pytest.raises(ValueError, match="retained candidate media path/hash differs"):
        visual_sources.build_visual_candidates(output, reference, resume=True, **args)
    assert (output / "source-audit.json").read_bytes() == audit_before


def test_stream_body_timeout_retries_the_whole_range_and_charges_each_attempt(monkeypatch):
    from urllib3.exceptions import ReadTimeoutError

    monkeypatch.setattr("minifrontier.data.remote.time.sleep", lambda _: None)
    calls = []

    class Response:
        status_code = 206

        def __init__(self):
            self.headers = {"Content-Range": "bytes 0-3/4"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        @property
        def raw(self):
            return self

        def read(self, *args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise ReadTimeoutError(None, "/file", "mid-body")
            return b"test"

    monkeypatch.setattr(requests.Session, "get", lambda *a, **k: Response())
    with RangeFile("https://example.invalid/file", 4, network_budget=8, chunk_size=4) as remote:
        assert remote.read(4) == b"test"
        assert remote.transferred == 8 and len(remote.transport_failures) == 1
