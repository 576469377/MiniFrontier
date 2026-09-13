import contextlib
import fcntl
import os
import signal
import sys
import threading

import pytest

from scripts import sync_mf1_tensorboard as sync

pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or not hasattr(fcntl, "F_NOTIFY"),
    reason="directory notifications require Linux dnotify",
)


def _expect_directory_write(monkeypatch, wait, action):
    """Write only after the watcher starts waiting; stale signals cannot pass."""
    ready = threading.Event()
    errors = []
    wakes = []
    original_select = sync.select.select

    def select_with_bounded_write(readers, writers, exceptional):
        # Prior directory creation can leave a queued notification. This wave
        # must wake for its own new filesystem write, not a previous signal.
        for fd in readers:
            with contextlib.suppress(BlockingIOError):
                while os.read(fd, 65536):
                    pass
        ready.set()
        result = original_select(readers, writers, exceptional, 3)
        wakes.append(bool(result[0]))
        return result

    def write_after_wait_begins():
        try:
            if not ready.wait(5):
                raise TimeoutError("watcher never started waiting")
            action()
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=write_after_wait_begins, daemon=True)
    with monkeypatch.context() as patch:
        patch.setattr(sync.select, "select", select_with_bounded_write)
        worker.start()
        try:
            wait()
            if not ready.is_set():
                # A newly created directory changes the watch set and requests
                # an immediate sync. The following wait must watch that directory.
                wait()
        finally:
            worker.join(6)
    assert not worker.is_alive(), "filesystem writer did not finish within its bound"
    assert not errors, errors
    assert wakes == [True], "the new filesystem write did not wake the watcher"


def test_future_run_and_tensorboard_directories_move_watches_inward(tmp_path, monkeypatch):
    source = tmp_path / "future_phase"
    tensorboard = source / "tensorboard"
    event = tensorboard / "events.out.tfevents.future"
    previous_handler = signal.getsignal(signal.SIGIO)

    with sync.directory_notifications([source, tensorboard]) as wait:
        assert not source.exists()  # Monitoring cannot create a trainer's output.
        _expect_directory_write(monkeypatch, wait, source.mkdir)
        assert source.is_dir() and not tensorboard.exists()
        _expect_directory_write(monkeypatch, wait, tensorboard.mkdir)
        _expect_directory_write(monkeypatch, wait, lambda: event.write_bytes(b"first"))

        def append_event():
            with event.open("ab") as handle:
                handle.write(b" next")

        _expect_directory_write(monkeypatch, wait, append_event)
        assert event.read_bytes() == b"first next"

    assert signal.getsignal(signal.SIGIO) == previous_handler
