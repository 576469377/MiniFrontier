"""Exclusive dispatch, occupancy exceptions, learning gates, and restart safety."""

import json
import subprocess
from pathlib import Path

import pytest

from minifrontier.hardware import GPUInfo
from scripts import run_exclusive_gpu_queue as queue


def gpu(index=7, used=2, free=24124):
    return GPUInfo(index, f"GPU-{index}", "3090", 24576, used, free, 0)


def test_idle_utilization_does_not_override_context_or_lease():
    job = dict(memory_gib=6, reserve_gib=2)
    assert queue.eligible(gpu(), {}, set(), {}, job)
    assert not queue.eligible(gpu(), {"GPU-7": {100}}, set(), {}, job)
    assert not queue.eligible(gpu(), {}, {7}, {}, job)
    assert not queue.eligible(gpu(), None, set(), {}, job)
    assert not queue.eligible(gpu(used=1200), {}, set(), {}, job)


def test_documented_inert_context_exception_keeps_real_memory_budget():
    policy = {"ignored_contexts": {"6": {"uuid": "GPU-6", "host_pids": [123]}}}
    job = dict(memory_gib=5.5, reserve_gib=2)
    assert queue.eligible(gpu(6, 14900, 9220), {"GPU-6": {123}}, set(), policy, job)
    assert not queue.eligible(gpu(6, 14900, 8000), {"GPU-6": {123}}, set(), policy, job)
    assert not queue.eligible(gpu(6, 14900, 9220), {"GPU-6": {123, 456}}, set(), policy, job)
    policy["ignored_contexts"]["6"]["uuid"] = "other-device"
    assert not queue.eligible(gpu(6, 14900, 9220), {"GPU-6": {123}}, set(), policy, job)


def test_compute_query_failure_is_closed(monkeypatch):
    def fail(*args, **kwargs):
        raise subprocess.TimeoutExpired("nvidia-smi", 10)

    monkeypatch.setattr(queue.subprocess, "check_output", fail)
    assert queue.compute_apps() is None


def test_generation_gate_requires_training_recall_eos_and_visual_retention(tmp_path):
    job = dict(control_result=str(tmp_path))
    assert queue.gate(job)[0] == "waiting_control_result"
    (tmp_path / "supervisor.json").write_text(json.dumps(dict(state="complete")))
    (tmp_path / "checkpoint.pt").touch()
    train = dict(summaries={"language/matched": dict(correct=29, total=32)}, eos_count=30)
    val = dict(summaries={f"{d}/matched": dict(correct=14, total=16) for d in ("vision", "video")})
    (tmp_path / "generation-train.json").write_text(json.dumps(train))
    (tmp_path / "generation-val.json").write_text(json.dumps(val))
    assert queue.gate(job)[0] == "ready"
    train["eos_count"] = 29
    (tmp_path / "generation-train.json").write_text(json.dumps(train))
    assert queue.gate(job)[0] == "needs_diagnosis"
    (tmp_path / "supervisor.json").write_text(json.dumps(dict(state="stopped_disk")))
    assert queue.gate(job)[0] == "blocked_by_control_run"


def test_queue_reserves_gpu_before_cuda_context_appears(tmp_path, monkeypatch):
    (tmp_path / "outputs").mkdir()
    output = tmp_path / "outputs/queue"
    output.mkdir()
    jobs = [
        dict(
            id=str(i),
            output=str(output / f"job{i}"),
            source_root=str(tmp_path),
            source={},
            inputs={},
            command=["worker"],
            memory_gib=6,
        )
        for i in range(2)
    ]
    plan = dict(
        workspace=str(tmp_path),
        jobs=jobs,
        allowed_gpu_ids=[7],
        main_budget_eligible=False,
        controller_sha256=queue.sha256(queue.__file__),
    )
    path = output / "queue-plan.json"
    path.write_text(json.dumps(plan))
    clock = [0]
    starts = []
    monkeypatch.setattr(queue, "query_gpus", lambda: [gpu()])
    monkeypatch.setattr(queue, "compute_apps", lambda: {})
    monkeypatch.setattr(queue, "require_space", lambda *a: None)
    monkeypatch.setattr(queue, "source_check", lambda *a: None)
    monkeypatch.setattr(queue.signal, "signal", lambda *a: None)
    monkeypatch.setattr(queue.time, "sleep", lambda *a: clock.__setitem__(0, clock[0] + 1))

    class Worker:
        def __init__(self, *args, **kwargs):
            self.ordinal = len(starts)
            self.started = clock[0]
            self.pid = 10000 + self.ordinal
            self.returncode = None
            starts.append((self.started, kwargs))

        def poll(self):
            if clock[0] <= self.started:
                return None
            target = Path(jobs[self.ordinal]["output"])
            target.mkdir(exist_ok=True)
            (target / "status.json").write_text('{"state":"complete"}')
            self.returncode = 0
            return 0

    monkeypatch.setattr(queue.subprocess, "Popen", Worker)
    assert queue.execute(path) == 0
    assert [start for start, _ in starts] == [0, 1]
    assert all(options["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-7" for _, options in starts)
    assert all(len(options["pass_fds"]) == 1 for _, options in starts)
    # A completed queue is restartable without launching duplicate workers.
    assert queue.execute(path) == 0
    assert len(starts) == 2


def test_reject_distributed_commands_and_failed_benchmark(tmp_path):
    plan = dict(
        allowed_gpu_ids=[0],
        main_budget_eligible=False,
        jobs=[
            dict(
                id="bad",
                output="new",
                memory_gib=6,
                command=["python", "-m", "torch.distributed.run"],
            )
        ],
    )
    with pytest.raises(ValueError, match="distributed"):
        queue.validate(plan)
    (tmp_path / "report.json").write_text(
        json.dumps(dict(state="complete", cases=[dict(state="out_of_memory")]))
    )
    assert not queue.completed(dict(output=str(tmp_path), result_file="report.json"))
