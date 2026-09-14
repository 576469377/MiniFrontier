"""Keep experiment variants distinct and GPU indices physically unambiguous."""

import json
import os

import pytest

from minifrontier.hardware import GPUInfo
from minifrontier.inference.demo import checkpoints, demo_device, serve


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
    monkeypatch.setattr("minifrontier.inference.demo.serve", lambda **kwargs: calls.append(kwargs))
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


@pytest.mark.parametrize("family", ["minifrontier1", "minifrontier11"])
def test_formal_lists_every_run_including_mf1_without_loading_weights(
    tmp_path, monkeypatch, family
):
    monkeypatch.setattr("torch.load", lambda *a, **kw: pytest.fail("listing loaded weights"))
    for name in ("reference", "second-seed"):
        folder = make_run(tmp_path, name)
        run = json.loads((folder / "run.json").read_text())
        run["kind"] = "strategy"
        (folder / "run.json").write_text(json.dumps(run))
    make_run(tmp_path, "diagnostic")
    for phase, ce, main in (("p0", 200_012_219, 200_012_219), ("p1", 31_197_392, 231_209_611)):
        folder = tmp_path / "formal" / family / phase
        folder.mkdir(parents=True)
        (folder / "run.json").write_text(
            json.dumps(
                dict(
                    format="mf1-run-v1",
                    model_name=family,
                    kind="strategy",
                    mf1_phase=phase,
                    token_budget=200_000_000 if phase == "p0" else 800_000_000,
                    unit="ce_tokens",
                )
            )
        )
        (folder / "status.json").write_text(
            json.dumps(
                dict(
                    step=100,
                    stage="pretrain",
                    mf1_phase=phase,
                    state="budget_complete_unqualified" if phase == "p0" else "running",
                    ledger=dict(ce_tokens=ce, phase_tokens=ce, main_ce_tokens=main),
                )
            )
        )
        (folder / "checkpoint.pt").write_bytes(b"MF1 fixture")
    found = checkpoints(tmp_path, "formal")
    assert len(found) == 4
    assert set(found) == {m["id"] for m in found.values()}
    assert all(m["kind"] == "strategy" for m in found.values())
    mf1 = found[f"formal/{family}/p1"]
    assert mf1["model_name"] == family
    assert mf1["ce_tokens"] == mf1["phase_tokens"] == 31_197_392
    assert mf1["main_ce_tokens"] == 231_209_611
    assert mf1["token_budget"] == mf1["ce_token_budget"] == 800_000_000
    assert mf1["version"].startswith("stat-v1:")
    assert not checkpoints(tmp_path, "accepted")


def test_latest_keeps_educational_runs_and_stat_version_detects_same_size_replacement(tmp_path):
    folder = make_run(tmp_path, "lesson")
    (folder / "run.json").write_text(
        json.dumps(
            dict(
                kind="educational",
                model_name="minikimik3",
                stage="pretrain",
            )
        )
    )
    assert not checkpoints(tmp_path, "formal")
    first = checkpoints(tmp_path, "latest")
    key = next(iter(first))
    path = folder / "checkpoint.pt"
    old_stat = path.stat()
    replacement = folder / "replacement.pt"
    replacement.write_bytes(b"x" * old_stat.st_size)
    os.utime(replacement, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
    replacement.replace(path)
    assert checkpoints(tmp_path, "latest")[key]["version"] != first[key]["version"]


def test_v41_formal_checkpoint_keeps_its_model_identity(tmp_path, monkeypatch):
    monkeypatch.setattr("torch.load", lambda *a, **kw: pytest.fail("listing loaded weights"))
    folder = tmp_path / "formal" / "minideepseekv41" / "D1"
    folder.mkdir(parents=True)
    (folder / "run.json").write_text(
        json.dumps(dict(kind="strategy", model_name="minideepseekv41", stage="pretrain"))
    )
    (folder / "status.json").write_text(json.dumps(dict(step=100, state="running")))
    (folder / "checkpoint.pt").write_bytes(b"V4.1 fixture")
    entry = checkpoints(tmp_path, "formal")["formal/minideepseekv41/D1"]
    assert entry["model_name"] == "minideepseekv41"
    assert entry["name"] == "MiniDeepSeek-V4.1"
    assert entry["stage"] == "pretrain" and entry["step"] == 100


def test_mf1_posttraining_metadata_tracks_its_budget_and_excludes_frozen_draft(tmp_path):
    for phase in ("rl", "teacher", "opd", "dpo", "draft"):
        folder = tmp_path / "mf1-posttraining" / phase
        folder.mkdir(parents=True)
        (folder / "run.json").write_text(
            json.dumps(
                dict(
                    format="mf1-posttrain-v1",
                    kind="strategy",
                    phase=phase,
                    token_budget=1000,
                )
            )
        )
        (folder / "status.json").write_text(
            json.dumps(
                dict(
                    state="running",
                    phase=phase,
                    step=2,
                    ledger=dict(generated_tokens=100, response_positions=75),
                )
            )
        )
        (folder / "checkpoint.pt").write_bytes(b"posttraining metadata fixture")
    found = checkpoints(tmp_path, "formal")
    assert len(found) == 4 and all(row["model_name"] == "minifrontier1" for row in found.values())
    assert {row["stage"] for row in found.values()} == {"rl", "teacher", "opd", "dpo"}
    dpo = found["mf1-posttraining/dpo"]
    assert dpo["budget_unit"] == "response_positions" and dpo["phase_tokens"] == 75
    assert dpo["ce_tokens"] is None and dpo["ce_token_budget"] is None
    assert found["mf1-posttraining/rl"]["phase_tokens"] == 100
    assert len(checkpoints(tmp_path, "rl")) == 1
