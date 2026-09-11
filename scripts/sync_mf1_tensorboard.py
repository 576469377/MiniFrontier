"""Build grouped live views of all trainers, preserving original logs and wall times.

The registry is a JSON list of {name, source, publish} entries. Source is a run
folder; name is its relative TensorBoard run name. Optional publish points to an
existing display symlink (its parent must be a real directory). Use a fresh
output directory on every invocation; original run files are always read-only.
The historical script name is retained for existing MF1 service commands.
"""

import argparse
import contextlib
import json
import os
import select
import signal
import time
from pathlib import Path
from typing import Any

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

from minifrontier.data.minifrontier1 import write_json
from minifrontier.training.metrics import training_scalars


class RunView:
    def __init__(self, source, output):
        self.source = Path(source)
        self.output = Path(output)
        if self.output.exists():
            raise FileExistsError("use a fresh view directory; source runs must not be modified")
        self.events = EventAccumulator(
            str(self.source / "tensorboard"), size_guidance={"scalars": 0}
        )
        self.writer = SummaryWriter(str(self.output))
        self.offset = self.rows = 0
        self.inode = None
        self.last_step = 0

    def sync(self):
        path = self.source / "metrics.jsonl"
        stat = path.stat()
        if self.inode is not None and (stat.st_ino != self.inode or stat.st_size < self.offset):
            raise ValueError(
                "source log replaced/truncated; rebuild the view from a fresh directory"
            )
        self.inode = stat.st_ino
        self.events.Reload()
        tags = self.events.Tags()["scalars"]
        times = {}
        for kind, candidates in (
            ("train", ("train_lm_loss", "train/lm_loss")),
            ("eval", ("nll", "validation/nll", "validation/lm_loss", "eval/lm_loss")),
            ("phase_end", ("eval/phase_end_lm_loss",)),
        ):
            for tag in candidates:
                if tag in tags:
                    for event in self.events.Scalars(tag):
                        times[kind, event.step] = event.wall_time
        with path.open("rb") as handle:
            handle.seek(self.offset)
            while True:
                line = handle.readline()
                if not line.endswith(b"\n"):
                    break  # A training process may still be writing this JSON record.
                values = json.loads(line)
                scalars = training_scalars(values)
                if not scalars:
                    self.rows += 1
                    self.offset = handle.tell()
                    continue  # Start/cache/QK diagnostics need no chart timestamp.
                step = values["step"]
                kind = "eval" if values.get("event") in {"validation", "eval"} else "train"
                if kind == "eval" and values.get("evaluation_scope") == "phase_end":
                    kind = "phase_end"
                wall_time = times.get((kind, step))
                if wall_time is None and kind == "phase_end":
                    wall_time = times.get(("eval", step))  # Older native event names.
                if wall_time is None:
                    break  # JSON is written before TensorBoard flushes: retry next poll.
                for tag, value in scalars.items():
                    self.writer.add_scalar(tag, value, step, walltime=wall_time)
                self.last_step = step
                self.rows += 1
                self.offset = handle.tell()
        self.writer.flush()
        return dict(rows=self.rows, byte_offset=self.offset, last_step=self.last_step)

    def close(self):
        self.writer.close()


def publish_view(link, target):
    """Replace only a display symlink, never a training folder or its ancestor."""
    link, target = Path(link).absolute(), Path(target).resolve()
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.parent.resolve() != link.parent:
        raise ValueError("display parent must be a real directory, not a link into a source run")
    if link.exists() and not link.is_symlink():
        raise ValueError("refusing to replace a real directory")
    if link.is_symlink() and link.resolve() == target:
        return
    temporary = link.with_name(f".{link.name}.grouped-{os.getpid()}")
    temporary.symlink_to(target, target_is_directory=True)
    os.replace(temporary, link)


@contextlib.contextmanager
def directory_notifications(paths):
    """Wake on Linux directory writes without consuming inotify watches or idle polling."""
    import fcntl

    read_fd, write_fd = os.pipe2(os.O_NONBLOCK | os.O_CLOEXEC)
    previous_handler = signal.signal(signal.SIGIO, lambda *_: None)
    previous_wakeup = signal.set_wakeup_fd(write_fd, warn_on_full_buffer=False)
    watches = []

    def wait():
        select.select([read_fd], [], [])
        with contextlib.suppress(BlockingIOError):
            while os.read(read_fd, 65536):
                pass

    try:
        for path in sorted(set(paths)):
            fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
            watches.append(fd)
            fcntl.fcntl(
                fd,
                fcntl.F_NOTIFY,
                fcntl.DN_MODIFY | fcntl.DN_CREATE | fcntl.DN_RENAME | fcntl.DN_MULTISHOT,
            )
        yield wait
    finally:
        for fd in watches:
            os.close(fd)
        signal.set_wakeup_fd(previous_wakeup)
        signal.signal(signal.SIGIO, previous_handler)
        os.close(read_fd)
        os.close(write_fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument(
        "--event-driven",
        action="store_true",
        help="Linux: wait for file writes between syncs; all source directories must exist",
    )
    parser.add_argument("--interval", type=float, default=5)
    args = parser.parse_args()
    if args.interval < 1:
        raise ValueError("poll interval must be at least one second")
    if args.event_driven and not args.watch:
        raise ValueError("--event-driven requires --watch")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    entries = json.loads(Path(args.registry).read_text())
    if len({e["name"] for e in entries}) != len(entries):
        raise ValueError("run names must be unique")
    for entry in entries:
        name = Path(entry["name"])
        if name.is_absolute() or ".." in name.parts:
            raise ValueError("view names must stay inside the output directory")
        source = Path(entry["source"]).resolve()
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("source and view directories must be separate")
    stopped = False

    def stop(_signal, _frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    views: dict[str, RunView] = {}
    state: dict[str, Any] = dict(state="starting", runs={})
    notification_paths = [
        path
        for entry in entries
        for path in (
            Path(entry["source"]).resolve(),
            Path(entry["source"]).resolve() / "tensorboard",
        )
    ]
    notifications = (
        directory_notifications(notification_paths)
        if args.event_driven
        else contextlib.nullcontext(None)
    )
    try:
        with notifications as wait_for_write:
            while not stopped:
                runs = {}
                for entry in entries:
                    name, source = entry["name"], Path(entry["source"]).resolve()
                    if (
                        not (source / "metrics.jsonl").exists()
                        or not (source / "tensorboard").exists()
                    ):
                        runs[name] = dict(state="waiting_for_run")
                        continue
                    if name not in views:
                        views[name] = RunView(source, output / name)
                    status = views[name].sync()
                    if status["rows"] and entry.get("publish"):
                        publish_view(entry["publish"], output / name)
                    runs[name] = dict(state="syncing", **status)
                state.update(
                    state="watching" if args.watch else "complete",
                    runs=runs,
                    updated_at=time.time(),
                )
                write_json(output / "status.json", state)
                if not args.watch:
                    break
                if wait_for_write is not None and not stopped:
                    wait_for_write()
                if stopped:
                    break
                time.sleep(args.interval)
    except Exception as error:
        state.update(state="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        for view in views.values():
            with contextlib.suppress(Exception):
                view.close()
        if stopped:
            state["state"] = "stopped"
        write_json(output / "status.json", state)


if __name__ == "__main__":
    main()
