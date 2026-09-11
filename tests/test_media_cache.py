import functools
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from minifrontier.data.media_cache import METADATA_BYTES, MediaCache
from minifrontier.storage import StorageLimitError


@pytest.fixture
def origin(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    requests = []

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            requests.append(self.path)
            super().do_GET()

    server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Handler, directory=source))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    policy = dict(
        base_url=f"http://127.0.0.1:{server.server_port}/",
        uri_prefix="media/",
        cache_dir=str(tmp_path / "cache"),
        max_bytes=METADATA_BYTES + 8,
        max_file_bytes=4,
        reserve_bytes=0,
    )
    try:
        yield source, requests, policy
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def put(source, name, data):
    (source / name).write_bytes(data)
    return "media/" + name, hashlib.sha256(data).hexdigest()


@pytest.mark.skipif(not Path("/proc/self/fd").is_dir(), reason="Linux native processor adapter")
def test_processor_path_survives_cache_eviction_and_closes(origin):
    source, _requests, policy = origin
    cache = MediaCache(policy)
    a, b, c = [put(source, name, name.encode() * 4) for name in ("a", "b", "c")]
    with cache.local_path(*a) as path:
        cache.read(*b)
        cache.read(*c)
        assert not (cache.root / a[1]).exists()
        assert path.read_bytes() == b"aaaa"
    assert not path.exists()


def test_lru_keeps_validation_pins_across_consumers_and_restart(origin):
    source, requests, policy = origin
    a, b, c = [
        put(source, name, value) for name, value in [("a", b"aaaa"), ("b", b"bbbb"), ("c", b"cccc")]
    ]
    train, validation = MediaCache(policy), MediaCache(policy, pin=True)
    assert train.read(*a) == b"aaaa"
    assert validation.read(*b) == b"bbbb"
    restart = MediaCache(policy)
    assert restart.read(*c) == b"cccc"
    assert not (restart.root / a[1]).exists()
    (source / "b").unlink()  # A hit must work without the remote file.
    assert restart.read(*b) == b"bbbb"
    assert restart.read(*a) == b"aaaa"
    assert (restart.root / b[1]).exists() and not (restart.root / c[1]).exists()
    stats = restart.accounting()
    assert stats["payload_bytes"] == 8 and stats["pinned_files"] == 1
    assert stats["actual_bytes"] <= policy["max_bytes"]
    assert stats["downloads"] == 4 and stats["evictions"] == 2
    assert requests.count("/b") == 1


def test_full_pinned_cache_refuses_growth_without_removing_holdout(origin):
    source, _requests, policy = origin
    files = [put(source, name, name.encode() * 4) for name in ["a", "b", "c"]]
    cache = MediaCache(policy, pin=True)
    for item in files[:2]:
        cache.read(*item)
    with pytest.raises(ValueError, match="pinned media"):
        cache.read(*files[2])
    assert all((cache.root / item[1]).exists() for item in files[:2])
    assert cache.accounting()["payload_bytes"] == 8
    assert not list(cache.root.glob("*.part"))


def test_bad_source_cache_corruption_path_escape_and_size_are_rejected(origin):
    source, _requests, policy = origin
    good = put(source, "good", b"good")
    cache = MediaCache(policy)
    cache.read(*good)
    with pytest.raises(ValueError, match="remote media hash"):
        cache.read(good[0], "0" * 64)
    with pytest.raises(ValueError, match="file bound"):
        cache.read(*put(source, "large", b"large"))
    for uri in ["../good", "media/../good", "/media/good", "other/good"]:
        with pytest.raises(ValueError, match="escapes"):
            cache.read(uri, good[1])
    (cache.root / good[1]).write_bytes(b"evil")
    with pytest.raises(ValueError, match="cached media hash/size"):
        cache.read(*good)
    assert cache.accounting()["downloads"] == 1


def test_shared_cache_does_not_download_twice_or_exceed_quota(origin):
    source, requests, policy = origin
    item = put(source, "item", b"same")
    caches = [MediaCache(policy), MediaCache(policy, pin=True)]
    with ThreadPoolExecutor(2) as pool:
        assert list(pool.map(lambda c: c.read(*item), caches)) == [b"same", b"same"]
    assert requests.count("/item") == 1
    assert caches[0].accounting()["pinned_files"] == 1


def test_failed_reservation_leaves_no_partial_file_and_recovery_is_bounded(origin, monkeypatch):
    source, _requests, policy = origin
    item = put(source, "item", b"same")
    cache = MediaCache(policy)
    with monkeypatch.context() as patch:
        patch.setattr(
            "minifrontier.storage.shutil.disk_usage", lambda _: type("Usage", (), {"free": 0})()
        )
        with pytest.raises(StorageLimitError):
            cache.read(*item)
    assert cache.accounting()["files"] == 0 and not list(cache.root.glob("*.part"))
    (cache.root / ("0" * 64 + ".part")).write_bytes(b"leftover")
    (cache.root / item[1]).write_bytes(b"same")  # Rename survived, index commit did not.
    recovered = MediaCache(policy)
    assert recovered.read(*item) == b"same"
    assert not list(cache.root.glob("*.part"))
    assert recovered.accounting()["files"] == 1
    (cache.root / "unrelated.txt").write_text("keep")
    with pytest.raises(ValueError, match="does not own"):
        MediaCache(policy)
    assert (cache.root / "unrelated.txt").read_text() == "keep"


def test_cache_origin_and_bounds_cannot_silently_change(origin):
    _source, _requests, policy = origin
    MediaCache(policy)
    with pytest.raises(ValueError, match="different origin or storage bounds"):
        MediaCache(dict(policy, max_bytes=policy["max_bytes"] + 1))
