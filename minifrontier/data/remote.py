"""Bounded parallel HTTP ranges for immutable Parquet; no whole-repo cache."""

import io
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import requests
from urllib3.exceptions import HTTPError as Urllib3HTTPError
from urllib3.util.retry import Retry


class RangeFile(io.RawIOBase):
    def __init__(self, url, size, *, network_budget=2 * 1024**3, chunk_size=4 * 1024**2):
        self.url, self.size = url, size
        self.position, self.transferred, self.budget = 0, 0, network_budget
        self.chunk_size = chunk_size
        self.pool = ThreadPoolExecutor(max_workers=4)
        self.local = threading.local()
        self.lock = threading.Lock()
        self.cache = OrderedDict()
        self.transport_failures = []
        super().__init__()

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset + (self.position if whence == 1 else self.size if whence == 2 else 0)
        if whence not in (0, 1, 2) or not 0 <= position <= self.size:
            raise ValueError("remote seek outside the immutable file")
        self.position = position
        return position

    def _chunk(self, index):
        start, end = index * self.chunk_size, min((index + 1) * self.chunk_size, self.size) - 1
        with self.lock:
            if index in self.cache:
                self.cache.move_to_end(index)
                return self.cache[index]
        if not hasattr(self.local, "session"):
            self.local.session = requests.Session()
            self.local.session.mount(
                "https://",
                requests.adapters.HTTPAdapter(
                    max_retries=Retry(
                        total=2,
                        connect=2,
                        backoff_factor=1,
                        status_forcelist=(429, 502, 503, 504),
                        allowed_methods=("GET",),
                    )
                ),
            )
        value = self._transfer(start, end)
        with self.lock:
            self.cache[index] = value
            while len(self.cache) > 4:
                self.cache.popitem(last=False)
        return value

    def _transfer(self, start, end):
        for attempt in range(3):
            with self.lock:
                if self.transferred + end - start + 1 > self.budget:
                    raise ValueError("remote read network-byte budget reached")
                self.transferred += end - start + 1
            try:
                with self.local.session.get(
                    self.url,
                    headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
                    timeout=(60, 120),
                    stream=True,
                ) as response:
                    response.raise_for_status()
                    if (
                        response.status_code != 206
                        or response.headers.get("Content-Range")
                        != f"bytes {start}-{end}/{self.size}"
                    ):
                        raise ValueError("server did not honor the exact byte range")
                    value = response.raw.read(end - start + 2, decode_content=False)
                    if len(value) != end - start + 1:
                        raise ValueError(
                            f"range {start}-{end}: expected {end - start + 1} bytes, got {len(value)}"
                        )
                    return value
            except (requests.ConnectionError, requests.Timeout, Urllib3HTTPError) as error:
                with self.lock:
                    self.transport_failures.append(
                        dict(start=start, end=end, attempt=attempt + 1, error=type(error).__name__)
                    )
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
        raise AssertionError("unreachable range retry exit")

    def read(self, size=-1):
        size = self.size - self.position if size < 0 else min(size, self.size - self.position)
        if size > 512 * 1024**2:
            raise ValueError("single remote read exceeds 512 MiB allocation cap")
        if not size:
            return b""
        begin, end = self.position, self.position + size
        self.position = end
        indices = range(begin // self.chunk_size, (end - 1) // self.chunk_size + 1)
        chunks = self.pool.map(self._chunk, indices)
        result = bytearray()
        for index, chunk in zip(indices, chunks, strict=True):
            offset = index * self.chunk_size
            result.extend(chunk[max(0, begin - offset) : min(len(chunk), end - offset)])
        return bytes(result)

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.cache.clear()
        super().close()
