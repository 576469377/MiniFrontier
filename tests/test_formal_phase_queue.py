"""Formal successors wait for exact parents and never replay launch intents."""

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from scripts import run_exclusive_gpu_queue as queue
from tests.test_exclusive_gpu_queue import gpu


def write(path, value):
    path.write_text(json.dumps(value))


def fixture(tmp_path):
    parent = tmp_path / "parent.json"
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"complete-parent")
    publication = tmp_path / "launch.json"
    output = tmp_path / "next"
    spec = dict(
        id="D2",
        model="minideepseekv4",
        phase="D2",
        gpu_id=7,
        parent_status=str(parent),
        parent_ce=10,
        parent_checkpoint=str(checkpoint),
        launch_file=str(publication),
        output=str(output),
        phase_ce=20,
    )
    command = [
        sys.executable,
        "-c",
        "pass",
        "--init",
        str(checkpoint),
        "--output",
        str(output),
        "--pretraining-phase",
        "D2",
        "--run-kind",
        "strategy",
    ]
    job = dict(
        state="bound",
        model=spec["model"],
        phase="D2",
        output=str(output),
        parent_checkpoint_sha256=queue.sha256(checkpoint),
        command=command,
        source_root=str(tmp_path),
        source={"commit": "fixture"},
        inputs={str(checkpoint): queue.sha256(checkpoint)},
        memory_gib=6,
    )
    return spec, job


def test_phase_binding_requires_complete_budget_and_exact_parent(tmp_path):
    spec, job = fixture(tmp_path)
    write(Path(spec["launch_file"]), job)
    write(Path(spec["parent_status"]), dict(state="paused", token_ledger=dict(ce_tokens=10)))
    assert queue.phase_ready(spec) == ("waiting_parent", None)
    write(Path(spec["parent_status"]), dict(state="complete", token_ledger=dict(ce_tokens=9)))
    assert queue.phase_ready(spec) == ("waiting_parent", None)
    write(Path(spec["parent_status"]), dict(state="complete", token_ledger=dict(ce_tokens=10)))
    assert queue.phase_ready(spec)[0] == "ready"
    Path(spec["parent_checkpoint"]).write_bytes(b"another-parent")
    with pytest.raises(ValueError, match="another parent"):
        queue.phase_ready(spec)


def test_mf1_budget_completion_is_not_mistaken_for_missing_chat_qualification():
    assert queue.phase_finished(
        dict(state="budget_complete_unqualified", ledger=dict(phase_tokens=10)), 10
    )
    assert not queue.phase_finished(dict(state="paused", ledger=dict(phase_tokens=10)), 10)


def initial_fixture(tmp_path, model="minideepseekv41", phase="D1"):
    spec, job = fixture(tmp_path)
    for key in ("parent_status", "parent_checkpoint", "parent_ce"):
        spec.pop(key)
    spec.update(id=f"{model}-{phase}", model=model, phase=phase, random_initialization=True)
    job.pop("parent_checkpoint_sha256")
    job.update(model=model, phase=phase, random_initialization=True)
    command = job["command"]
    del command[3:5]
    command[command.index("--pretraining-phase") + 1] = phase
    if model == "minifrontier11":
        command[command.index("--pretraining-phase")] = "--phase"
    return spec, job


@pytest.mark.parametrize("model,phase", [("minideepseekv41", "D1"), ("minifrontier11", "p0")])
def test_initial_phase_requires_own_random_publication_without_parent(tmp_path, model, phase):
    spec, job = initial_fixture(tmp_path, model, phase)
    assert queue.phase_ready(spec) == ("waiting_binding", None)
    write(Path(spec["launch_file"]), job)
    assert queue.phase_ready(spec)[0] == "ready"
    job["random_initialization"] = False
    write(Path(spec["launch_file"]), job)
    with pytest.raises(ValueError, match="bind random initialization"):
        queue.phase_ready(spec)


@pytest.mark.parametrize("flag", ["--init", "--resume", "--init=old.pt", "--resume=old.pt"])
def test_initial_phase_rejects_weight_inheritance(tmp_path, flag):
    spec, job = initial_fixture(tmp_path)
    job["command"].append(flag)
    write(Path(spec["launch_file"]), job)
    with pytest.raises(ValueError, match="bind random initialization"):
        queue.phase_ready(spec)


@pytest.mark.parametrize(
    "change",
    [
        dict(phase="D2"),
        dict(model="unknown"),
        dict(parent_ce=0),
        dict(parent_status="old.json"),
        dict(random_initialization="true"),
    ],
)
def test_initial_phase_rejects_later_phases_implicit_flags_and_parents(tmp_path, change):
    spec, _ = initial_fixture(tmp_path)
    spec.update(change)
    with pytest.raises(ValueError, match=r"first phase|explicit boolean"):
        queue.phase_ready(spec)


def test_formal_successor_runs_once_and_respects_single_gpu_environment(tmp_path, monkeypatch):
    spec, job = fixture(tmp_path)
    write(Path(spec["parent_status"]), dict(state="complete", token_ledger=dict(ce_tokens=10)))
    result = dict(state="complete", token_ledger=dict(ce_tokens=20))
    script = (
        "import json,os; from pathlib import Path; "
        f"p=Path({spec['output']!r}); p.mkdir(); "
        f"(p/'status.json').write_text({json.dumps(result)!r}); "
        "(p/'environment.json').write_text(json.dumps(dict(os.environ)))"
    )
    job["command"][2] = script
    job["env"] = {"HTTPS_PROXY": "http://unused.invalid", "CUDA_VISIBLE_DEVICES": "wrong"}
    write(Path(spec["launch_file"]), job)
    plan = dict(
        kind="formal_pretraining_phase_queue",
        workspace=str(tmp_path),
        controller_sha256=queue.sha256(queue.__file__),
        main_budget_eligible=True,
        allowed_gpu_ids=[7],
        notification_backend="dnotify",
        jobs=[spec],
    )
    path = tmp_path / "plan.json"
    write(path, plan)
    monkeypatch.setattr(queue, "query_gpus", lambda: [gpu()])
    monkeypatch.setattr(queue, "compute_apps", lambda: {})
    monkeypatch.setattr(queue, "source_check", lambda *a: None)
    monkeypatch.setattr(queue, "require_space", lambda *a, **k: None)
    assert queue.execute_phases(path) == 0
    state = json.loads((tmp_path / "queue.json").read_text())
    assert state["jobs"]["D2"]["state"] == "complete"
    env = json.loads((Path(spec["output"]) / "environment.json").read_text())
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-7" and "HTTPS_PROXY" not in env
    assert env["NO_PROXY"] == "*" and env["MINIFRONTIER_MIN_FREE_GIB"] == "80"
    assert queue.execute_phases(path) == 0
    assert json.loads((tmp_path / "queue.json").read_text())["jobs"] == state["jobs"]


def test_initial_phase_cpu_dispatch_runs_once_without_a_parent(tmp_path, monkeypatch):
    spec, job = initial_fixture(tmp_path, "minifrontier11", "p0")
    result = dict(state="budget_complete_unqualified", ledger=dict(phase_tokens=20))
    job["command"][2] = (
        "from pathlib import Path; "
        f"p=Path({spec['output']!r}); p.mkdir(); "
        f"(p/'status.json').write_text({json.dumps(result)!r})"
    )
    write(Path(spec["launch_file"]), job)
    path = tmp_path / "plan.json"
    write(
        path,
        dict(
            kind="formal_pretraining_phase_queue",
            workspace=str(tmp_path),
            controller_sha256=queue.sha256(queue.__file__),
            main_budget_eligible=True,
            allowed_gpu_ids=[7],
            notification_backend="dnotify",
            jobs=[spec],
        ),
    )
    monkeypatch.setattr(queue, "query_gpus", lambda: [gpu()])
    monkeypatch.setattr(queue, "compute_apps", lambda: {})
    monkeypatch.setattr(queue, "source_check", lambda *a: None)
    monkeypatch.setattr(queue, "require_space", lambda *a, **k: None)
    assert queue.execute_phases(path) == 0
    state = json.loads((tmp_path / "queue.json").read_text())
    job_state = state["jobs"][spec["id"]]
    assert job_state["state"] == "complete"
    assert job_state["random_initialization"] is True
    assert job_state["parent_checkpoint_sha256"] is None
    assert queue.execute_phases(path) == 0
    assert json.loads((tmp_path / "queue.json").read_text())["jobs"] == state["jobs"]


def test_waiting_phase_does_not_wake_itself_or_scan_on_a_timer(tmp_path):
    spec, _ = fixture(tmp_path)
    write(Path(spec["parent_status"]), dict(state="running"))
    path = tmp_path / "plan.json"
    write(
        path,
        dict(
            kind="formal_pretraining_phase_queue",
            workspace=str(tmp_path),
            controller_sha256=queue.sha256(queue.__file__),
            main_budget_eligible=True,
            allowed_gpu_ids=[7],
            notification_backend="dnotify",
            jobs=[spec],
        ),
    )
    process = subprocess.Popen(
        [sys.executable, "-m", "scripts.run_exclusive_gpu_queue", "--plan", str(path)]
    )
    state = tmp_path / "queue.json"
    try:
        deadline = time.monotonic() + 5
        while not state.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert json.loads(state.read_text())["jobs"]["D2"]["state"] == "waiting_parent"
        before = state.stat().st_mtime_ns
        for _ in range(5):
            (tmp_path / "unrelated.log").write_text("no dependency changed")
            time.sleep(0.03)
        assert state.stat().st_mtime_ns == before
        write(Path(spec["parent_status"]), dict(state="complete", token_ledger=dict(ce_tokens=10)))
        deadline = time.monotonic() + 5
        while state.stat().st_mtime_ns == before and time.monotonic() < deadline:
            time.sleep(0.02)
        assert json.loads(state.read_text())["jobs"]["D2"]["state"] == "waiting_binding"
    finally:
        process.terminate()
        process.wait(timeout=5)
    assert json.loads(state.read_text())["state"] == "controller_stopped_workers_preserved"
