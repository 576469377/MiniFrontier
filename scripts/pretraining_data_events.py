"""Observe data-task events and run declared, bounded completion hooks.

Linux filesystem notifications watch run records; pidfds report process exits. The
observer emits a transport heartbeat without repeatedly inspecting processes.
Task run.json files and the existing experiment registry remain authoritative.
This service does not allocate GPUs, retry producers, or grant data admission.
"""

import argparse
import base64
import ctypes
import fcntl
import hashlib
import json
import os
import platform
import selectors
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

LIMIT = 2 * 1024**2


def pidfd_open(pid):
    if hasattr(os, "pidfd_open"):
        return os.pidfd_open(pid)
    # Some Python builds omit the wrapper despite a supporting Linux kernel.
    # Linux x86_64 and asm-generic (aarch64) headers both assign syscall 434.
    if sys.platform != "linux" or platform.machine() not in {"x86_64", "aarch64"}:
        raise RuntimeError("event observation requires Linux pidfd support")
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.syscall(434, int(pid), 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "pidfd_open")
    os.set_inheritable(fd, False)
    return fd


def read(path):
    path = Path(path)
    if not path.exists():
        return {}
    if path.stat().st_size > LIMIT:
        raise ValueError(f"event metadata exceeds byte bound: {path.name}")
    return json.loads(path.read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def emit(value):
    print(json.dumps(value, separators=(",", ":")), flush=True)


class FileEvents:
    """Use dnotify when the shared user's inotify quota is exhausted.

    dnotify wakes a pipe on directory events. Only declared file signatures are
    compared on wakeup, so writes to derived observations do not query producers.
    Neither backend scans files on a timer or changes global kernel limits.
    """

    def __init__(self, tasks, *, backend="auto"):
        self.paths: dict[Path, set[str]] = {}
        for key, task in tasks.items():
            for filename in [task["run"], *task.get("files", [])]:
                self.paths.setdefault(Path(filename).resolve(), set()).add(key)
        self.directories = {p.parent for p in self.paths}
        self.watches: dict[int, Path] = {}
        self.backend = "inotify"
        self.libc = ctypes.CDLL(None, use_errno=True)
        self.fd = -1
        try:
            if backend == "dnotify":
                raise OSError("dnotify requested")
            if backend != "auto":
                raise ValueError("notification_backend must be auto or dnotify")
            self.fd = self.libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
            if self.fd < 0:
                raise OSError(ctypes.get_errno(), "inotify_init1")
            for parent in self.directories:
                wd = self.libc.inotify_add_watch(
                    self.fd, os.fsencode(parent), 0x8 | 0x80 | 0x100 | 0x200 | 0x40
                )
                if wd < 0:
                    raise OSError(ctypes.get_errno(), f"cannot watch {parent}")
                self.watches[wd] = parent
        except OSError:
            if self.fd >= 0:
                os.close(self.fd)
            self.backend = "dnotify"
            self.fd, self.pipe_write = os.pipe2(os.O_NONBLOCK | os.O_CLOEXEC)
            self.old_handler = signal.signal(signal.SIGIO, lambda *_: None)
            self.old_wakeup = signal.set_wakeup_fd(self.pipe_write)
            self.directory_fds = []
            try:
                for parent in self.directories:
                    fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
                    self.directory_fds.append(fd)
                    fcntl.fcntl(fd, fcntl.F_SETOWN, os.getpid())
                    fcntl.fcntl(
                        fd,
                        fcntl.F_NOTIFY,
                        fcntl.DN_MULTISHOT
                        | fcntl.DN_CREATE
                        | fcntl.DN_MODIFY
                        | fcntl.DN_DELETE
                        | fcntl.DN_RENAME,
                    )
            except BaseException:
                self.close()
                raise
        self.signatures = {p: self.signature(p) for p in self.paths}

    @staticmethod
    def signature(path):
        try:
            stat = path.stat()
            return stat.st_ino, stat.st_mtime_ns, stat.st_size
        except FileNotFoundError:
            return None

    def changed(self):
        data = os.read(self.fd, 65536)
        changed = set()
        if self.backend == "dnotify":
            for path, keys in self.paths.items():
                signature = self.signature(path)
                if signature != self.signatures[path]:
                    self.signatures[path] = signature
                    changed.update(keys)
        else:
            offset = 0
            while offset < len(data):
                wd, mask, _, length = struct.unpack_from("iIII", data, offset)
                offset += 16
                name = os.fsdecode(data[offset : offset + length].split(b"\0", 1)[0])
                offset += length
                if mask & 0x4000:
                    raise RuntimeError("filesystem event overflow; revalidate observer state")
                if wd in self.watches:
                    changed.update(self.paths.get(self.watches[wd] / name, ()))
        return changed

    def close(self):
        if self.backend == "dnotify":
            for fd in self.directory_fds:
                os.close(fd)
            signal.set_wakeup_fd(self.old_wakeup)
            signal.signal(signal.SIGIO, self.old_handler)
            os.close(self.pipe_write)
        os.close(self.fd)


def observe(plan_path, heartbeat=60):
    plan = read(plan_path)
    tasks = {t["id"]: t for t in plan["tasks"]}
    selector = selectors.DefaultSelector()
    notifications = FileEvents(tasks, backend=plan.get("notification_backend", "auto"))
    selector.register(notifications.fd, selectors.EVENT_READ, ("files", None))
    active: dict[str, tuple[int, int]] = {}
    last: dict[str, str] = {}

    def snapshot(key):
        task = tasks[key]
        run_path = Path(task["run"])
        raw_record = run_path.read_bytes() if run_path.exists() else b"{}"
        if len(raw_record) > LIMIT:
            raise ValueError("task record exceeds byte bound")
        record = json.loads(raw_record)
        alive = False
        previous = active.pop(key, None)
        if previous:
            selector.unregister(previous[0])
            os.close(previous[0])
        if record and not record.get("completed_unix"):
            try:
                pid = record["pid"]
                expected = [str(v).encode() for v in record["command"]]
                actual = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")[:-1]
                if not expected or actual != expected:
                    raise ProcessLookupError("task command differs")
                fd = pidfd_open(pid)
                selector.register(fd, selectors.EVENT_READ, ("exit", key))
                active[key] = (fd, pid)
                alive = True
            except (OSError, KeyError):
                # A final atomic record may have arrived between the first read
                # and pidfd_open. Missing observations never trigger a restart.
                raw_record = run_path.read_bytes() if run_path.exists() else b"{}"
                record = json.loads(raw_record)
        files = {}
        for name in [task["run"], *task.get("files", [])]:
            path = Path(name)
            if path.exists():
                raw = raw_record if name == task["run"] else path.read_bytes()
                if len(raw) > LIMIT:
                    raise ValueError("event file exceeds byte bound")
                files[name] = base64.b64encode(raw).decode()
        item = dict(id=key, record=record, alive=alive, files=files, observed_unix=time.time())
        # Some producers rewrite only their heartbeat timestamp while waiting.
        # Forward actual state/progress changes, not those periodic rewrites.
        signature = digest(
            dict(
                record={k: v for k, v in record.items() if k != "observed_unix"},
                alive=alive,
                files={k: v for k, v in files.items() if k != task["run"]},
            )
        )
        if signature != last.get(key):
            last[key] = signature
            emit(dict(kind="task", notification_backend=notifications.backend, **item))

    try:
        for key in tasks:
            snapshot(key)
        next_heartbeat = time.monotonic() + heartbeat
        while True:
            events = selector.select(max(0, next_heartbeat - time.monotonic()))
            changed = set()
            for selected, _ in events:
                kind, key = selected.data
                if kind == "exit":
                    changed.add(key)
                else:
                    changed.update(notifications.changed())
            for key in sorted(changed):
                snapshot(key)
            if time.monotonic() >= next_heartbeat:
                emit(
                    dict(
                        kind="heartbeat",
                        notification_backend=notifications.backend,
                        active={k: v[1] for k, v in active.items()},
                        observed_unix=time.time(),
                    )
                )
                next_heartbeat = time.monotonic() + heartbeat
    finally:
        for fd, _ in active.values():
            os.close(fd)
        selector.close()
        notifications.close()


def completion_key(task_id, record):
    if not record.get("completed_unix"):
        return None
    return digest(
        [
            task_id,
            record.get("state"),
            record["completed_unix"],
            record.get("manifest_sha256"),
            record.get("source_audit_sha256"),
        ]
    )


def run_hook(control, hook, event):
    """A persisted intent prevents replay after an ambiguous interrupted hook."""
    key = digest([hook["id"], completion_key(event["id"], event["record"])])
    marker = control / "actions" / (key + ".json")
    if marker.exists():
        return read(marker)
    for filename, expected in hook.get("input_sha256", {}).items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
            raise ValueError("completion hook input changed")
    event_path = control / "events" / (key + ".json")
    write(event_path, event)
    action = dict(
        hook=hook["id"],
        task=event["id"],
        state="started",
        started_unix=time.time(),
        event_sha256=hashlib.sha256(event_path.read_bytes()).hexdigest(),
    )
    write(marker, action)
    command = [*hook["command"], "--event", str(event_path)]
    with (control / "hook-output.log").open("ab") as log:
        process = subprocess.Popen(
            command,
            cwd=hook["cwd"],
            env=dict(os.environ, **hook.get("env", {})),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=hook.get("timeout_seconds", 120))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            code = -1
    action.update(
        state="complete" if code == 0 else "needs_attention" if code == 2 else "failed",
        returncode=code,
        completed_unix=time.time(),
    )
    write(marker, action)
    return action


def supervise(plan_path):
    plan = read(plan_path)
    control = Path(plan["control"])
    control.mkdir(parents=True, exist_ok=True)
    lock = (control / "service.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    selector = selectors.DefaultSelector()
    processes: dict[str, subprocess.Popen[bytes]] = {}
    observed: dict[str, dict[str, Any]] = {}
    actions: dict[str, dict[str, Any]] = {}
    attention: dict[str, str] = {}
    service = dict(
        kind="data_construction",
        phase="event_driven_data_followups",
        pid=os.getpid(),
        command=sys.argv[:],
        state="starting_event_observers",
        started_unix=time.time(),
        plan_sha256=hashlib.sha256(Path(plan_path).read_bytes()).hexdigest(),
        formal_admission=False,
        main_budget_eligible=False,
    )
    service["command"] = [
        sys.executable,
        str(Path(__file__).resolve()),
        "run",
        "--plan",
        str(Path(plan_path).resolve()),
    ]

    def save():
        write(
            control / "run.json",
            dict(
                service,
                observed_unix=time.time(),
                observers={k: p.pid for k, p in processes.items()},
                tasks=observed,
                actions=actions,
                needs_attention=attention,
            ),
        )

    try:
        for host in plan["observers"]:
            process = subprocess.Popen(
                host["command"],
                stdout=subprocess.PIPE,
                stderr=(control / (host["id"] + ".stderr.log")).open("ab"),
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                bufsize=0,
            )
            processes[host["id"]] = process
            assert process.stdout is not None
            selector.register(process.stdout, selectors.EVENT_READ, host)
        service["state"] = "watching_data_events"
        save()
        deadline = time.monotonic() + plan.get("wall_seconds", 43200)
        buffers = {h["id"]: b"" for h in plan["observers"]}
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("data event service reached its declared wall budget")
            for selected, _ in selector.select(remaining):
                host = selected.data
                chunk = os.read(selected.fd, 65536)
                if not chunk:
                    raise ConnectionError(
                        f"event observer disconnected: {host['id']}; no producer restarted"
                    )
                buffers[host["id"]] += chunk
                if len(buffers[host["id"]]) > 16 * LIMIT:
                    raise ValueError("observer message exceeds transport bound")
                while b"\n" in buffers[host["id"]]:
                    line, buffers[host["id"]] = buffers[host["id"]].split(b"\n", 1)
                    event = json.loads(line)
                    if event["kind"] == "heartbeat":
                        for key, pid in event["active"].items():
                            task = observed.get(key)
                            if task and task.get("pid") == pid and task.get("alive"):
                                task["observed_unix"] = event["observed_unix"]
                                write(
                                    Path(task["local_run"]).with_name("process-observation.json"),
                                    dict(
                                        pid=pid,
                                        argv_matches=True,
                                        observed_unix=event["observed_unix"],
                                        observation_method="open remote pidfd and event-stream heartbeat",
                                    ),
                                )
                        save()
                        continue
                    key, record = event["id"], event["record"]
                    task_spec = next(t for t in host["tasks"] if t["id"] == key)
                    for filename, encoded in event["files"].items():
                        if filename not in task_spec["files"]:
                            raise ValueError("observer sent undeclared metadata")
                        target = Path(task_spec["files"][filename])
                        if host.get("mirror"):
                            raw = base64.b64decode(encoded, validate=True)
                            if len(raw) > LIMIT:
                                raise ValueError("mirrored metadata exceeds byte bound")
                            target.parent.mkdir(parents=True, exist_ok=True)
                            temporary = target.with_name(target.name + ".event.tmp")
                            temporary.write_bytes(raw)
                            temporary.replace(target)
                    local_run = task_spec["local_run"]
                    observed[key] = dict(
                        state=record.get("state", "waiting_for_record"),
                        pid=record.get("pid"),
                        alive=event["alive"],
                        observed_unix=event["observed_unix"],
                        local_run=local_run,
                        completed_unix=record.get("completed_unix"),
                    )
                    write(
                        Path(local_run).with_name("process-observation.json"),
                        dict(
                            pid=record.get("pid"),
                            argv_matches=event["alive"],
                            observed_unix=event["observed_unix"],
                            observation_method="filesystem notifications and pidfd event stream",
                        ),
                    )
                    if record and not record.get("completed_unix") and not event["alive"]:
                        attention[key] = "process missing or identity differs; no automatic restart"
                    if completion_key(key, record):
                        for hook in plan.get("hooks", {}).get(key, []):
                            action = run_hook(control, hook, event)
                            actions[key + ":" + hook["id"]] = action
                            if action["state"] != "complete":
                                attention[key] = action["state"]
                    save()
                    if plan.get("on_event"):
                        subprocess.run(
                            plan["on_event"],
                            cwd=plan["workspace"],
                            check=True,
                            timeout=120,
                            stdout=subprocess.DEVNULL,
                        )
    except BaseException as error:
        service.update(
            state="stopped_needs_attention", error=repr(error), completed_unix=time.time()
        )
        save()
        raise
    finally:
        for process in processes.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
        selector.close()
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("observe", "run"))
    parser.add_argument("--plan", required=True, type=Path)
    args = parser.parse_args()
    if args.mode == "run":

        def stop(signum, _frame):
            raise SystemExit(128 + signum)

        signal.signal(signal.SIGTERM, stop)
    (observe if args.mode == "observe" else supervise)(args.plan)


if __name__ == "__main__":
    main()
