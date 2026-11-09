import threading
from concurrent.futures import ThreadPoolExecutor

import torch

from minifrontier.storage import reserve_write
from minifrontier.training import runtime


def test_staging_allows_media_reservations_and_keeps_last_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIFRONTIER_MIN_FREE_GIB", "0")
    monkeypatch.setenv("MINIFRONTIER_STORAGE_LOCK_DIR", str(tmp_path))
    path = tmp_path / "checkpoint.pt"
    old = {"weight": torch.tensor([5]), "step": 1}
    new = {"weight": torch.tensor([6]), "step": 2}
    runtime.atomic_save(old, path)
    pinned, release = threading.Event(), threading.Event()
    original = runtime.cpu_tree

    def stage(value):
        if value is new:
            pinned.set()
            assert release.wait(10)
        return original(value)

    def read_media():
        with reserve_write(tmp_path / "media", 4096):
            return torch.load(path, weights_only=True)["step"]

    monkeypatch.setattr(runtime, "cpu_tree", stage)
    with ThreadPoolExecutor(max_workers=2) as pool:
        writer = pool.submit(runtime.atomic_save, new, path)
        try:
            assert pinned.wait(10)
            assert pool.submit(read_media).result(timeout=5) == 1
        finally:
            release.set()
        writer.result(timeout=10)
    saved = torch.load(path, weights_only=True)
    assert saved["step"] == 2
    torch.testing.assert_close(saved["weight"], new["weight"])
    assert not list(tmp_path.glob("*.tmp"))
