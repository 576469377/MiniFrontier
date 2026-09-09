"""Single-device scheduling must preserve data accounting and avoid occupied GPUs."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from minifrontier.data import sha256
from minifrontier.hardware import GPUInfo
from minifrontier.storage import StorageLimitError
from minifrontier.training.media_mixture import MediaMixtureCursor
from minifrontier.training.mixture import TokenMixtureCursor
from scripts import run_single_gpu_queue as queue


@pytest.mark.parametrize("media", [False, True])
def test_single_microbatch_two_preserves_two_rank_global_stream(media):
    data = SimpleNamespace(
        domains=["zh"] * 100 + ["en"] * 100 + (["caption"] * 100 if media else []),
        ce_counts=[31] * 100 + [73] * 100 + ([17] * 100 if media else []),
        input_counts=[32] * 100 + [74] * 100 + ([89] * 100 if media else []),
        image_counts=[0] * 200 + ([1] * 100 if media else []),
        video_counts=[0] * (300 if media else 200),
    )
    mixture = {"zh": 0.6, "en": 0.4}
    if media:
        mixture = dict(
            schema_version=1,
            ce_token_budget=20000,
            image_occurrences=40,
            video_examples=0,
            text_mixture_tokens=mixture,
            image_mixture_samples={"caption": 1.0},
        )
    cls = MediaMixtureCursor if media else TokenMixtureCursor
    single = cls(data, mixture, batch_size=2, world_size=1, seed=42)
    ranks = [cls(data, mixture, batch_size=1, rank=r, world_size=2, seed=42) for r in range(2)]
    for _ in range(20):
        positions = 0
        while positions < 16384:
            expected = ranks[0].next() + ranks[1].next()
            assert single.next() == expected
            positions += sum(data.input_counts[i] for i in expected)
        assert single.offset == ranks[0].offset == ranks[1].offset
        if media:
            assert single.ce_seen == ranks[0].ce_seen == ranks[1].ce_seen
            assert single.seen == ranks[0].seen == ranks[1].seen


def test_commands_preserve_budget_and_change_only_requested_learning_rate():
    original = [
        "python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        "-m",
        "minifrontier",
        "train",
        "--output",
        "old",
        "--batch-size",
        "1",
        "--input-batch-tokens",
        "16384",
        "--ce-tokens",
        "20000000",
        "--sequence-length",
        "512",
        "--optimizer",
        "auto",
        "--lr",
        "0.0003",
        "--muon-lr",
        "0.01",
        "--seed",
        "42",
    ]
    for _, (_, _, flag, value) in queue.FAMILIES.items():
        reference = queue.single_command(original, "new", "reference", flag, value)
        lower = queue.single_command(original, "new", "lower-lr", flag, value)
        assert reference[:4] == ["python", "-m", "minifrontier", "train"]
        assert reference[reference.index("--batch-size") + 1] == "2"
        changed = [i for i, (a, b) in enumerate(zip(reference, lower, strict=True)) if a != b]
        assert changed == [lower.index(flag) + 1]
        assert lower[changed[0]] == value
    assert original[original.index("--batch-size") + 1] == "1"
    with pytest.raises(ValueError, match="random initialization"):
        queue.single_command([*original, "--resume", "old.pt"], "new", "reference", "--lr", "0.1")


def test_six_jobs_wait_for_predecessor_gpu_and_disk_and_isolate_devices(tmp_path, monkeypatch):
    workspace = tmp_path
    (workspace / "outputs").mkdir()
    output = workspace / "outputs/queue"
    output.mkdir()
    plan_path = output / "queue-plan.json"
    jobs = [
        dict(
            id=f"trial{i}",
            gpu_id=i,
            output=str(output / f"trial{i}"),
            predecessor="previous",
            source={},
            inputs={},
            command=["worker", "--output", str(output / f"trial{i}")],
        )
        for i in range(6)
    ]
    plan_path.write_text(
        json.dumps(
            dict(
                workspace=str(workspace),
                source_root=str(workspace),
                output=str(output),
                controller_sha256=sha256(queue.__file__),
                jobs=jobs,
            )
        )
    )
    clock = [0]
    histories, launches = [], []
    disk_failed = [False]
    monkeypatch.setattr(queue, "predecessor_ready", lambda _: clock[0] > 0)
    monkeypatch.setattr(queue, "source_check", lambda *_: None)
    monkeypatch.setattr(
        queue,
        "query_gpus",
        lambda: [GPUInfo(i, f"GPU-{i}", "3090", 24576, 0, 24576, 0) for i in range(8)],
    )
    monkeypatch.setattr(queue, "busy_gpu_uuids", lambda: {"GPU-1"} if clock[0] <= 1 else set())
    monkeypatch.setattr(queue.time, "sleep", lambda _: clock.__setitem__(0, clock[0] + 1))
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("RANK", "7")

    def reserve(*_):
        if not disk_failed[0]:
            disk_failed[0] = True
            raise StorageLimitError("test disk pressure")

    monkeypatch.setattr(queue, "require_space", reserve)
    original_write = queue.write_json

    def record(path, state):
        histories.append([j["state"] for j in state["jobs"]])
        original_write(path, state)

    monkeypatch.setattr(queue, "write_json", record)

    def spawn(command, *, env, **kwargs):
        assert clock[0] > 0
        assert env["CUDA_VISIBLE_DEVICES"] not in queue.busy_gpu_uuids()
        assert "WORLD_SIZE" not in env and "RANK" not in env
        assert env["MINIFRONTIER_MIN_FREE_GIB"] == "50"
        launches.append(env["CUDA_VISIBLE_DEVICES"])
        destination = Path(command[-1])
        destination.mkdir()
        (destination / "status.json").write_text(json.dumps(dict(state="complete")))
        return SimpleNamespace(pid=100 + len(launches), poll=lambda: 0)

    monkeypatch.setattr(queue.subprocess, "Popen", spawn)
    assert queue.execute(plan_path) == 0
    assert histories[0] == ["waiting_predecessor"] * 6
    assert "waiting_disk" in histories[1] and "waiting_gpu" in histories[1]
    assert sorted(launches) == [f"GPU-{i}" for i in range(6)]
    assert histories[-1] == ["complete"] * 6
    with pytest.raises(FileExistsError, match="already started"):
        queue.execute(plan_path)


def test_predecessor_needs_both_completed_trials_and_exports(tmp_path):
    path = tmp_path / "pilot.json"
    trials = []
    for optimizer in ("auto", "adamw"):
        output = tmp_path / optimizer
        output.mkdir()
        (output / "status.json").write_text(json.dumps(dict(state="complete")))
        trials.append(dict(optimizer=optimizer, output=str(output)))
    path.write_text(json.dumps(dict(stage=queue.COMPLETE, trials=trials)))
    assert not queue.predecessor_ready(path)
    for trial in trials:
        (Path(trial["output"]) / "model.pt").touch()
    assert queue.predecessor_ready(path)
    (tmp_path / "adamw/status.json").write_text(json.dumps(dict(state="running")))
    assert not queue.predecessor_ready(path)
