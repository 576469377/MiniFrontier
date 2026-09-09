"""Two-slot admission, recipe isolation and handoff without duplicate workers."""

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from minifrontier.data import sha256
from minifrontier.hardware import GPUInfo
from scripts import run_shared_gpu_queue as queue


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    (tmp_path / "outputs").mkdir()
    old = tmp_path / "outputs/strategy-single-gpu-v2"
    old.mkdir()
    jobs = []
    for model, first in (("minikimik3", 0), ("miniqwen4", 2), ("minideepseekv4", 4)):
        config = tmp_path / f"{model}.json"
        config.write_text(
            json.dumps(
                dict(mtp_enabled=True, mtp_loss_coef=0.3 if first == 4 else 0.1, hidden_size=512)
            )
        )
        for offset, variant in enumerate(("reference", "lower-lr")):
            output = old / model / variant
            command = [
                "python",
                "-m",
                "minifrontier",
                "train",
                "--model",
                model,
                "--config",
                str(config),
                "--output",
                str(output),
                "--optimizer",
                "auto",
                "--ce-tokens",
                "20000000",
                "--batch-size",
                "2",
                "--input-batch-tokens",
                "16384",
                "--sequence-length",
                "512",
            ]
            jobs.append(
                dict(
                    id=f"{model}/{variant}",
                    model=model,
                    variant=variant,
                    gpu_id=first + offset,
                    output=str(output),
                    predecessor=str(tmp_path / "pilot.json"),
                    source={},
                    inputs={str(config): sha256(config)},
                    command=command,
                )
            )
    (old / "queue-plan.json").write_text(json.dumps(dict(jobs=jobs)))
    (old / "queue.json").write_text(
        json.dumps(
            dict(
                pid=99999999,
                jobs=[
                    dict(
                        id=j["id"],
                        gpu_id=j["gpu_id"],
                        output=j["output"],
                        state="waiting_predecessor",
                    )
                    for j in jobs
                ],
            )
        )
    )
    monkeypatch.setattr(queue, "source_check", lambda *_: None)
    monkeypatch.setattr(queue, "require_space", lambda *_: None)
    worker = tmp_path / "worker.py"
    worker.write_text("# test worker\n")
    output = tmp_path / "outputs/strategy-shared-gpu-v2"
    queue.prepare(tmp_path, tmp_path, output, worker)
    return output / "queue-plan.json"


def test_six_additions_change_only_mtp_weight_and_place_by_capacity(prepared):
    plan = queue.read_json(prepared)
    queue.validate_plan(plan)
    for job in plan["jobs"]:
        assert job["memory_gib"] + 1 == job["reserved_gib"]
        if job["role"] != "companion":
            continue
        reference = next(
            j
            for j in plan["jobs"]
            if j["role"] == "primary" and j["model"] == job["model"] and j["variant"] == "reference"
        )

        def config(j):
            args = j["command"]
            return queue.read_json(args[args.index("--config") + 1])

        actual, base = config(job), config(reference)
        assert [k for k in actual if actual[k] != base[k]] == ["mtp_loss_coef"]
        assert actual["mtp_enabled"] is True
        assert job["mtp_weight"] in ({0, 0.1} if job["model"] == "minideepseekv4" else {0, 0.2})
        assert job["command"][job["command"].index("--ce-tokens") + 1] == "20000000"
    assert [j["model"] for j in plan["jobs"][:6]] == ["miniqwen4"] * 2 + ["minikimik3"] * 2 + [
        "minideepseekv4"
    ] * 2


@pytest.mark.parametrize("mutation", ["third_slot", "unauthorized_gpu", "over_budget"])
def test_invalid_shared_plan_is_rejected(prepared, mutation):
    plan = copy.deepcopy(queue.read_json(prepared))
    if mutation == "third_slot":
        plan["jobs"][1]["gpu_id"] = 0
    elif mutation == "unauthorized_gpu":
        plan["jobs"][0]["gpu_id"] = 6
    else:
        plan["jobs"][0]["reserved_gib"] = 20
    with pytest.raises(ValueError):
        queue.validate_plan(plan)


def test_memory_admission_allows_one_companion_but_keeps_reserve():
    gpu = GPUInfo(0, "GPU-0", "3090", 24576, 8192, 16384, 50)
    assert queue.can_start(dict(reserved_gib=12), gpu, {"GPU-0": 1})
    assert not queue.can_start(dict(reserved_gib=12), gpu, {"GPU-0": 2})
    assert not queue.can_start(dict(reserved_gib=15), gpu, {"GPU-0": 1})
    assert not queue.can_start(dict(reserved_gib=12), gpu, None)


def test_process_identity_rejects_pid_reuse():
    identity = queue.process_identity(os.getpid())
    assert queue.alive(identity)
    assert not queue.alive(dict(identity, start_ticks=identity["start_ticks"] + 1))


def test_shared_dual_rank_history_tracks_each_gpu_separately(tmp_path):
    (tmp_path / "run.json").write_text("{}")
    queue.mark_sharing(tmp_path, 4, ["old-dual", "mtp-0"])
    queue.mark_sharing(tmp_path, 5, ["old-dual", "mtp-0.1"])
    queue.mark_sharing(tmp_path, 4, ["old-dual", "mtp-0"])
    info = queue.read_json(tmp_path / "co_residency.json")
    assert len(info["devices"]["4"]) == len(info["devices"]["5"]) == 1
    queue.mark_sharing(tmp_path, 4, [])
    info = queue.read_json(tmp_path / "co_residency.json")
    assert "ended_at" in info["devices"]["4"][-1]
    assert "ended_at" not in info["devices"]["5"][-1]


def test_queue_launches_twelve_once_and_mirrors_original_queue(prepared, monkeypatch):
    plan = queue.read_json(prepared)
    monkeypatch.setattr(queue, "predecessor_ready", lambda _: True)
    monkeypatch.setattr(
        queue,
        "query_gpus",
        lambda: [GPUInfo(i, f"GPU-{i}", "3090", 24576, 0, 24576, 0) for i in range(8)],
    )
    monkeypatch.setattr(queue, "app_counts", lambda: {})
    monkeypatch.setattr(queue.time, "sleep", lambda _: None)
    launches = []

    def spawn(command, *, env, **kwargs):
        destination = Path(command[command.index("--output") + 1])
        destination.mkdir()
        (destination / "status.json").write_text('{"state":"complete"}')
        launches.append(env["CUDA_VISIBLE_DEVICES"])
        assert "WORLD_SIZE" not in env
        assert float(command[command.index("--memory-gib") + 1]) <= 11
        return SimpleNamespace(pid=99999999 + len(launches), poll=lambda: 0)

    monkeypatch.setattr(queue.subprocess, "Popen", spawn)
    assert queue.execute(prepared) == 0
    assert sorted(launches) == sorted([f"GPU-{i}" for i in range(6)] * 2)
    state = queue.read_json(prepared.parent / "queue.json")
    assert len(state["jobs"]) == 12 and all(j["state"] == "complete" for j in state["jobs"])
    mirror = queue.read_json(Path(plan["old_plan"]).parent / "queue.json")
    assert len(mirror["jobs"]) == 6 and mirror["managed_by"] == str(prepared.parent / "queue.json")
    with pytest.raises(FileExistsError):
        queue.execute(prepared)
