"""Build a grouped live view of frozen MF1 runs, preserving original logs and wall times.

The registry is a JSON list of {name, source, publish} entries. Source is a run
folder; name is its relative TensorBoard run name. Optional publish points to an
existing display symlink (its parent must be a real directory). Use a fresh
output directory on every invocation; original run files are always read-only.
"""

import argparse
import contextlib
import json
import os
import signal
import time
from pathlib import Path
from typing import Any

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

from minifrontier.data.minifrontier1 import write_json
from minifrontier.training.metrics import mf1_scalars


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
            ("eval", ("nll", "validation/nll", "eval/lm_loss")),
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
                step = values["step"]
                kind = "eval" if values.get("event") in {"validation", "eval"} else "train"
                wall_time = times.get((kind, step))
                if wall_time is None:
                    break  # JSON is written before TensorBoard flushes: retry next poll.
                for tag, value in mf1_scalars(values).items():
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=5)
    args = parser.parse_args()
    if args.interval < 1:
        raise ValueError("poll interval must be at least one second")
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
    try:
        while not stopped:
            runs = {}
            for entry in entries:
                name, source = entry["name"], Path(entry["source"]).resolve()
                if not (source / "metrics.jsonl").exists() or not (source / "tensorboard").exists():
                    runs[name] = dict(state="waiting_for_run")
                    continue
                if name not in views:
                    views[name] = RunView(source, output / name)
                status = views[name].sync()
                if status["rows"] and entry.get("publish"):
                    publish_view(entry["publish"], output / name)
                runs[name] = dict(state="syncing", **status)
            state.update(
                state="watching" if args.watch else "complete", runs=runs, updated_at=time.time()
            )
            write_json(output / "status.json", state)
            if not args.watch:
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
