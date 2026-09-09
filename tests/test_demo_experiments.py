"""Keep experiment variants distinct and GPU indices physically unambiguous."""

import json
import os

import pytest

from minifrontier.hardware import GPUInfo
from minifrontier.inference import checkpoints, demo_device, serve


def make_run(root, variant, *, step=200, saved=True):
    folder = root / "single-gpu" / "minikimik3" / variant
    folder.mkdir(parents=True)
    (folder / "run.json").write_text(
        json.dumps(dict(kind="acceptance", model_name="minikimik3", stage="pretrain"))
    )
    if saved:
        (folder / "status.json").write_text(
            json.dumps(dict(stage="pretrain", step=step, state="running"))
        )
        (folder / "checkpoint.pt").write_bytes(b"checkpoint fixture")
    return folder


def test_variants_visible_only_in_explicit_experiment_view(tmp_path):
    make_run(tmp_path, "reference")
    make_run(tmp_path, "lower-lr", step=400)
    make_run(tmp_path, "not-yet-saved", saved=False)
    with pytest.raises(ValueError, match="include-experiments"):
        checkpoints(tmp_path, "experiments")
    assert not checkpoints(tmp_path, "accepted", include_experiments=True)
    assert not checkpoints(tmp_path, "pretrain", include_experiments=True)
    found = checkpoints(tmp_path, "experiments", include_experiments=True)
    assert len(found) == 2
    assert {entry["step"] for entry in found.values()} == {200, 400}
    assert all(entry["capability_status"] == "unassessed" for entry in found.values())
    assert {entry["id"].split("/")[-1] for entry in found.values()} == {"reference", "lower-lr"}


def test_refresh_discovers_atomic_replacement_and_ignores_partial_metadata(tmp_path):
    folder = make_run(tmp_path, "reference")
    first = checkpoints(tmp_path, "experiments", include_experiments=True)
    key = next(iter(first))
    (folder / "model.pt").write_bytes(b"final export")
    checkpoint = folder / "checkpoint.pt"
    os.utime(
        checkpoint,
        ns=(checkpoint.stat().st_atime_ns, (folder / "model.pt").stat().st_mtime_ns + 1000),
    )
    latest = checkpoints(tmp_path, "experiments", include_experiments=True)
    assert latest[key]["artifact"] == "checkpoint.pt"
    (folder / "status.json").write_text("{")
    assert not checkpoints(tmp_path, "experiments", include_experiments=True)
    (folder / "status.json").unlink()
    assert checkpoints(tmp_path, "experiments", include_experiments=True)[key]["step"] is None


def test_current_and_history_partition_by_directory_and_discover_new_saves(tmp_path):
    current = make_run(tmp_path, "reference")
    make_run(tmp_path / "recipe-pilots", "muon")
    make_run(tmp_path / "quickstart", "tiny")
    make_run(tmp_path / "single-gpu-old", "diagnostic")
    pending = make_run(tmp_path, "lower-lr", saved=False)
    scope = dict(include_experiments=True, experiment_roots=["single-gpu", "recipe-pilots"])
    active = checkpoints(tmp_path, "experiments", **scope)
    history = checkpoints(tmp_path, "history", **scope)
    all_runs = checkpoints(tmp_path, "experiments", include_experiments=True)
    assert len(active) == len(history) == 2
    assert active.keys().isdisjoint(history)
    assert active.keys() | history.keys() == all_runs.keys()
    assert active["single-gpu/minikimik3/reference"]["run_label"] == "reference"
    assert not checkpoints(tmp_path, "accepted", **scope)
    # Newly saved runs under either selected directory appear without a restart.
    (pending / "checkpoint.pt").write_bytes(b"new save")
    assert len(checkpoints(tmp_path, "experiments", **scope)) == 3
    (current / "status.json").write_text(
        json.dumps(dict(step=400, state="running", token_ledger=dict(ce_tokens=6_600_000)))
    )
    entry = checkpoints(tmp_path, "experiments", **scope)["single-gpu/minikimik3/reference"]
    assert entry["ce_tokens"] == 6_600_000 and entry["step"] == 400
    with pytest.raises(ValueError, match="include-experiments"):
        checkpoints(tmp_path, "history")
    with pytest.raises(ValueError, match="inside --root"):
        checkpoints(tmp_path, "experiments", include_experiments=True, experiment_roots=["../"])
    with pytest.raises(ValueError, match="requires --include-experiments"):
        serve(root=tmp_path, experiment_roots=["single-gpu"])


def test_physical_gpu_option_maps_uuid_without_initializing_cuda(monkeypatch):
    gpu = GPUInfo(6, "GPU-test-six", "RTX 3090", 24576, 14900, 9676, 0)
    monkeypatch.setattr("minifrontier.hardware.query_gpus", lambda: [gpu])
    monkeypatch.setattr("torch.cuda.is_initialized", lambda: False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    device, label = demo_device(gpu=6)
    assert device == "cuda:0" and "GPU 6" in label
    assert os.environ["CUDA_VISIBLE_DEVICES"] == gpu.uuid
    with pytest.raises(ValueError, match="not found"):
        demo_device(gpu=7)
    monkeypatch.setattr("torch.cuda.is_initialized", lambda: True)
    with pytest.raises(ValueError, match="before CUDA initialization"):
        demo_device(gpu=6)


def test_demo_cli_gpu_and_device_are_mutually_exclusive(monkeypatch):
    from minifrontier.cli import main

    calls = []
    monkeypatch.setattr("minifrontier.inference.serve", lambda **kwargs: calls.append(kwargs))
    main(
        [
            "demo",
            "--gpu",
            "6",
            "--include-experiments",
            "--experiment-root",
            "single-gpu",
            "--experiment-root",
            "recipe-pilots",
        ]
    )
    assert calls[0]["gpu"] == 6 and calls[0]["include_experiments"]
    assert calls[0]["experiment_roots"] == ["single-gpu", "recipe-pilots"]
    with pytest.raises(SystemExit):
        main(["demo", "--gpu", "6", "--device", "cuda:0"])
