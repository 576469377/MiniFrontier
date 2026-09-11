import base64
import hashlib
import json
import selectors
import subprocess
import sys
import time
from pathlib import Path

import pytest

from scripts.pretraining_data_events import completion_key, run_hook, write

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux event APIs required")
SCRIPT = Path(__file__).resolve().parents[1] / "scripts/pretraining_data_events.py"


def message(process, timeout=5):
    with selectors.DefaultSelector() as selector:
        selector.register(process.stdout, selectors.EVENT_READ)
        assert selector.select(timeout), "event did not arrive"
        return json.loads(process.stdout.readline())


@pytest.mark.parametrize("backend", ["auto", "dnotify"])
def test_observer_reports_atomic_records_and_process_exit_without_a_final_record(tmp_path, backend):
    command = [sys.executable, "-c", "import time; time.sleep(60)"]
    child = subprocess.Popen(command)
    run = tmp_path / "run.json"
    record = dict(pid=child.pid, command=command, state="building")
    write(run, record)
    plan = tmp_path / "observer.json"
    write(plan, dict(tasks=[dict(id="data", run=str(run))], notification_backend=backend))
    observer = subprocess.Popen(
        [sys.executable, str(SCRIPT), "observe", "--plan", str(plan)],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        first = message(observer)
        assert first["record"] == record and first["alive"]
        assert json.loads(base64.b64decode(first["files"][str(run)])) == record
        changed = dict(record, state="encoding")
        write(run, changed)
        event = message(observer)
        assert event["record"] == changed and event["alive"]
        child.terminate()
        child.wait(timeout=5)
        exited = message(observer)
        assert not exited["alive"] and not exited["record"].get("completed_unix")
        assert exited["record"]["pid"] == child.pid
        assert not Path(f"/proc/{child.pid}").exists()
    finally:
        observer.terminate()
        observer.wait(timeout=5)
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=5)


def test_completed_record_is_observed_without_requiring_a_live_producer(tmp_path):
    run = tmp_path / "run.json"
    record = dict(
        pid=999999999, command=["fixture"], state="candidate_complete", completed_unix=123.0
    )
    write(run, record)
    plan = tmp_path / "observer.json"
    write(plan, dict(tasks=[dict(id="done", run=str(run))]))
    process = subprocess.Popen(
        [sys.executable, str(SCRIPT), "observe", "--plan", str(plan)],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        event = message(process)
        assert event["record"] == record and not event["alive"]
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_completion_hook_is_bound_and_never_replayed_after_success_or_interrupted_intent(tmp_path):
    control = tmp_path / "service"
    control.mkdir()
    script = tmp_path / "hook.py"
    counter = tmp_path / "counter"
    script.write_text(
        "from pathlib import Path\np=Path("
        + repr(str(counter))
        + ")\np.write_text(str(int(p.read_text())+1) if p.exists() else '1')\n"
    )
    hook = dict(
        id="encode",
        command=[sys.executable, str(script)],
        cwd=str(tmp_path),
        timeout_seconds=5,
        input_sha256={str(script): hashlib.sha256(script.read_bytes()).hexdigest()},
    )
    event = dict(id="data", record=dict(state="done", completed_unix=123))
    assert completion_key("data", {}) is None
    result = run_hook(control, hook, event)
    assert result["state"] == "complete" and counter.read_text() == "1"
    assert run_hook(control, hook, event) == result and counter.read_text() == "1"
    marker = next((control / "actions").glob("*.json"))
    write(marker, dict(state="started", hook="encode"))
    assert run_hook(control, hook, event)["state"] == "started"
    assert counter.read_text() == "1"
    script.write_text(script.read_text() + "# changed\n")
    with pytest.raises(ValueError, match="input changed"):
        run_hook(control, dict(hook, id="new-step"), event)
    assert counter.read_text() == "1"


def test_supervisor_runs_a_completion_once_and_cleans_up_observers(tmp_path):
    control = tmp_path / "service"
    record = dict(state="done", completed_unix=123)
    run = tmp_path / "task" / "run.json"
    write(run, record)
    event = dict(
        kind="task",
        id="data",
        record=record,
        alive=False,
        files={str(run): base64.b64encode(run.read_bytes()).decode()},
        observed_unix=time.time(),
    )
    observer = tmp_path / "observer.py"
    emit_line = "print(" + repr(json.dumps(event)) + ",flush=True)\n"
    observer.write_text("import time\n" + emit_line * 2 + "time.sleep(60)\n")
    counter = tmp_path / "counter"
    hook = tmp_path / "hook.py"
    hook.write_text(
        "from pathlib import Path\np=Path("
        + repr(str(counter))
        + ")\np.write_text(str(int(p.read_text())+1) if p.exists() else '1')\n"
    )
    plan = tmp_path / "plan.json"
    write(
        plan,
        dict(
            control=str(control),
            workspace=str(tmp_path),
            wall_seconds=60,
            observers=[
                dict(
                    id="local",
                    command=[sys.executable, str(observer)],
                    tasks=[dict(id="data", local_run=str(run), files={str(run): str(run)})],
                )
            ],
            hooks={
                "data": [dict(id="next", command=[sys.executable, str(hook)], cwd=str(tmp_path))]
            },
        ),
    )
    service = subprocess.Popen(
        [sys.executable, str(SCRIPT), "run", "--plan", str(plan)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if counter.exists() and list((control / "actions").glob("*.json")):
                action = json.loads(next((control / "actions").glob("*.json")).read_text())
                if action.get("state") == "complete":
                    break
            time.sleep(0.02)
        assert counter.read_text() == "1"
        state = json.loads((control / "run.json").read_text())
        observer_pid = state["observers"]["local"]
        service.terminate()
        service.wait(timeout=5)
        assert not Path(f"/proc/{observer_pid}").exists()
        assert json.loads((control / "run.json").read_text())["state"] == "stopped_needs_attention"
    finally:
        if service.poll() is None:
            service.terminate()
            service.wait(timeout=5)
