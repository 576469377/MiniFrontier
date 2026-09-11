"""A bounded, hash-checked raw-file cache for existing immutable media components."""

import contextlib
import ctypes
import fcntl
import hashlib
import json
import os
import re
import sqlite3
import tempfile
import time
from pathlib import Path, PurePosixPath
from urllib.parse import quote, urlsplit
from urllib.request import ProxyHandler, build_opener

from minifrontier.storage import require_space, reserve_write

METADATA_BYTES = 32 * 1024**2  # SQLite page cap, rollback journal and small control files.
MAX_ENTRIES = 32768
HASH = re.compile(r"[0-9a-f]{64}\Z")


def validate_policy(policy):
    required = {
        "base_url",
        "uri_prefix",
        "cache_dir",
        "max_bytes",
        "max_file_bytes",
        "reserve_bytes",
    }
    if not isinstance(policy, dict) or set(policy) != required:
        raise ValueError("media cache needs an explicit origin, path and storage bounds")
    url = urlsplit(policy["base_url"])
    prefix = policy["uri_prefix"]
    if (
        url.scheme not in {"http", "https"}
        or not url.hostname
        or url.username
        or url.password
        or url.query
        or url.fragment
        or not policy["base_url"].endswith("/")
        or not policy["cache_dir"]
        or (
            prefix
            and (
                prefix.startswith("/")
                or ".." in PurePosixPath(prefix).parts
                or not prefix.endswith("/")
            )
        )
    ):
        raise ValueError("invalid media cache origin or URI prefix")
    if any(type(policy[k]) is not int for k in ("max_bytes", "max_file_bytes", "reserve_bytes")):
        raise ValueError("media cache byte bounds must be integers")
    if not (
        0 < policy["max_file_bytes"] <= min(128 * 1024**2, policy["max_bytes"] - METADATA_BYTES)
        and policy["reserve_bytes"] >= 0
    ):
        raise ValueError("media cache cannot hold one bounded file plus metadata")
    return dict(policy)


class MediaCache:
    """Serialize eviction/write accounting across consumers; return verified bytes.

    Decoding uses returned bytes, so another process cannot evict an in-use file.
    Pinning persists across processes and restarts. Only files owned by this cache
    are reclaimable; original media never resides in this directory.
    """

    def __init__(self, policy, *, pin=False, base=None):
        self.policy = validate_policy(policy)
        self.root = ((Path(base) if base else Path.cwd()) / policy["cache_dir"]).resolve()
        self.pin = pin
        self.limit = policy["max_bytes"] - METADATA_BYTES
        self.opener = build_opener(ProxyHandler({}))
        require_space(self.root, METADATA_BYTES, reserve_bytes=policy["reserve_bytes"])
        self.root.mkdir(parents=True, exist_ok=True)
        # Use the same cache-lock -> shared-storage-lock order as cache misses.
        with (
            self._locked() as db,
            reserve_write(
                self.root / "cache.lock", METADATA_BYTES, reserve_bytes=policy["reserve_bytes"]
            ),
        ):
            marker = self.root / "policy.json"
            identity = {k: v for k, v in self.policy.items() if k != "cache_dir"}
            if marker.exists():
                if json.loads(marker.read_text()) != identity:
                    raise ValueError("existing media cache has different origin or storage bounds")
            else:
                if any(p.name not in {"cache.lock", "index.sqlite"} for p in self.root.iterdir()):
                    raise ValueError(
                        "media cache requires a new directory or its own policy marker"
                    )
                marker.write_text(json.dumps(identity, sort_keys=True) + "\n")
            db.executescript(
                "CREATE TABLE IF NOT EXISTS files (hash TEXT PRIMARY KEY, size INTEGER NOT NULL, "
                "used INTEGER NOT NULL, pinned INTEGER NOT NULL);"
                "CREATE INDEX IF NOT EXISTS lru ON files(pinned,used);"
                "CREATE TABLE IF NOT EXISTS stats (key TEXT PRIMARY KEY, value INTEGER NOT NULL);"
            )
            # Recover only this cache's interrupted writes and atomic renames.
            found = set()
            for path in self.root.iterdir():
                if path.name in {
                    "cache.lock",
                    "policy.json",
                    "index.sqlite",
                    "index.sqlite-journal",
                }:
                    continue
                if path.suffix == ".part" and HASH.fullmatch(path.stem) and not path.is_symlink():
                    path.unlink()
                    continue
                if not HASH.fullmatch(path.name) or not path.is_file() or path.is_symlink():
                    raise ValueError("media cache contains a file it does not own")
                found.add(path.name)
                size = path.stat().st_size
                if not 0 < size <= policy["max_file_bytes"]:
                    raise ValueError("cached media exceeds its file bound")
                db.execute(
                    "INSERT INTO files VALUES(?,?,?,0) ON CONFLICT(hash) DO UPDATE SET size=excluded.size",
                    (path.name, size, time.time_ns()),
                )
            for (key,) in db.execute("SELECT hash FROM files").fetchall():
                if key not in found:
                    db.execute("DELETE FROM files WHERE hash=?", (key,))
            db.commit()
            self._evict(db, 0, new_entry=False)

    @contextlib.contextmanager
    def _locked(self):
        with (self.root / "cache.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            with contextlib.closing(sqlite3.connect(self.root / "index.sqlite")) as db:
                if db.execute("PRAGMA page_size").fetchone()[0] != 4096:
                    raise ValueError("media cache index requires bounded 4096-byte pages")
                db.execute("PRAGMA max_page_count=2048")
                yield db

    @staticmethod
    def _count(db, name, value=1):
        db.execute(
            "INSERT INTO stats VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=value+excluded.value",
            (name, value),
        )

    def _evict(self, db, incoming, *, new_entry=True):
        size, count = db.execute("SELECT coalesce(sum(size),0),count(*) FROM files").fetchone()
        while size + incoming > self.limit or count + int(new_entry) > MAX_ENTRIES:
            row = db.execute(
                "SELECT hash,size FROM files WHERE pinned=0 ORDER BY used LIMIT 1"
            ).fetchone()
            if row is None:
                db.commit()
                raise ValueError("pinned media leaves insufficient cache capacity")
            key, removed = row
            (self.root / key).unlink(missing_ok=True)
            db.execute("DELETE FROM files WHERE hash=?", (key,))
            self._count(db, "evictions")
            size -= removed
            count -= 1
        db.commit()

    def read(self, uri, expected):
        if (
            not isinstance(uri, str)
            or not isinstance(expected, str)
            or not HASH.fullmatch(expected)
        ):
            raise ValueError("media cache requires a URI and a complete SHA256 identity")
        prefix = self.policy["uri_prefix"]
        parts = PurePosixPath(uri).parts
        if not uri.startswith(prefix) or uri.startswith("/") or ".." in parts or "\\" in uri:
            raise ValueError("media URI escapes the configured origin")
        relative = uri[len(prefix) :]
        if not relative or relative.startswith("/"):
            raise ValueError("media URI does not identify a file under its prefix")
        # A fixed set of process-shared stripes bounds lock metadata. Downloads
        # of different objects do not hold the cache-wide accounting lock.
        lock_root = Path(tempfile.gettempdir()) / "minifrontier-media-download-locks"
        lock_root.mkdir(exist_ok=True)
        namespace = hashlib.sha256(str(self.root).encode()).hexdigest()[:16]
        stripe = int(expected[:2], 16) % 64
        with (lock_root / f"{namespace}-{stripe}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            with self._locked() as db:
                cached = self._cached(db, expected)
                if cached is not None:
                    return cached
            url = self.policy["base_url"] + quote(relative, safe="/")
            with self.opener.open(url, timeout=30) as response:
                maximum = self.policy["max_file_bytes"]
                length = response.headers.get("Content-Length")
                if response.status != 200 or (
                    length is not None and not 0 < int(length) <= maximum
                ):
                    raise ValueError("remote media response exceeds its file bound")
                data = response.read(int(length) + 1 if length is not None else maximum + 1)
            if not data or len(data) > maximum or (length is not None and len(data) != int(length)):
                raise ValueError("remote media size differs from its bounded response")
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError("remote media hash differs")
            with self._locked() as db:
                # An older reader may have filled the cache during this download.
                cached = self._cached(db, expected)
                if cached is not None:
                    return cached
                self._evict(db, len(data))
                path = self.root / expected
                temporary = path.with_suffix(".part")
                try:
                    with reserve_write(
                        temporary,
                        len(data) + METADATA_BYTES,
                        reserve_bytes=self.policy["reserve_bytes"],
                    ):
                        with temporary.open("xb") as handle:
                            handle.write(data)
                        temporary.replace(path)
                        db.execute(
                            "INSERT INTO files VALUES(?,?,?,?)",
                            (expected, len(data), time.time_ns(), int(self.pin)),
                        )
                        self._count(db, "downloads")
                        self._count(db, "downloaded_bytes", len(data))
                        db.commit()
                finally:
                    temporary.unlink(missing_ok=True)
                return data

    def _cached(self, db, expected):
        path = self.root / expected
        row = db.execute("SELECT size FROM files WHERE hash=?", (expected,)).fetchone()
        if row is None:
            return None
        if path.is_symlink() or not 0 < row[0] <= self.policy["max_file_bytes"]:
            raise ValueError("cached media path or size is invalid")
        with path.open("rb") as handle:
            data = handle.read(row[0] + 1)
        if len(data) != row[0] or hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("cached media hash/size differs")
        db.execute(
            "UPDATE files SET used=?,pinned=max(pinned,?) WHERE hash=?",
            (time.time_ns(), int(self.pin), expected),
        )
        self._count(db, "hits")
        db.commit()
        return data

    @contextlib.contextmanager
    def local_path(self, uri, expected):
        """Expose verified bytes to a path-only processor through a scoped file.

        Linux anonymous files live only until the processor returns. Cache eviction
        cannot invalidate them, and original processor/tokenization identities stay
        unchanged. Each file is bounded by the cache's existing per-file limit.
        """
        if not Path("/proc/self/fd").is_dir():
            raise ValueError("native remote media paths require Linux anonymous files")
        data = self.read(uri, expected)
        fd = _memfd()
        anonymous_memory = fd is not None
        with (
            os.fdopen(fd, "w+b") if fd is not None else tempfile.TemporaryFile(dir=self.root.parent)
        ) as handle:
            # Some Python builds omit memfd_create. The fallback is unlinked,
            # bounded and on the data filesystem; release the storage lock before
            # yielding, since a multi-frame processor may open another file.
            reservation = (
                contextlib.nullcontext()
                if anonymous_memory
                else reserve_write(
                    self.root / "processor-staging",
                    len(data),
                    reserve_bytes=self.policy["reserve_bytes"],
                )
            )
            with reservation:
                handle.write(data)
                handle.flush()
            del data
            yield Path(f"/proc/self/fd/{handle.fileno()}")

    def accounting(self):
        with self._locked() as db:
            size, count, pinned = db.execute(
                "SELECT coalesce(sum(size),0),count(*),coalesce(sum(pinned),0) FROM files"
            ).fetchone()
            actual = sum(p.stat().st_size for p in self.root.iterdir() if p.is_file())
            if actual > self.policy["max_bytes"]:
                raise ValueError("actual media cache files exceed the storage bound")
            return dict(
                payload_bytes=size,
                files=count,
                pinned_files=pinned,
                actual_bytes=actual,
                max_bytes=self.policy["max_bytes"],
                **dict(db.execute("SELECT key,value FROM stats")),
            )


def _memfd():
    """Use the libc entry point when a Python build omits os.memfd_create."""
    if hasattr(os, "memfd_create"):
        try:
            return os.memfd_create("minifrontier-media", os.MFD_CLOEXEC)
        except OSError:
            return None
    create = getattr(ctypes.CDLL(None, use_errno=True), "memfd_create", None)
    if create is None:
        return None
    create.argtypes = [ctypes.c_char_p, ctypes.c_uint]
    create.restype = ctypes.c_int
    fd = create(b"minifrontier-media", 1)  # Linux MFD_CLOEXEC
    return fd if fd >= 0 else None
