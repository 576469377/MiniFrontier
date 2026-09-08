"""Regression checks for biased sampling, explicit budgets and checkpoint selection."""

import io
import json
import runpy
from pathlib import Path

import pytest
import torch

from minifrontier.data import download_prefix
from minifrontier.inference import checkpoints
from minifrontier.training.runtime import validation_indices

budget_steps = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/launch_training.py")
)["budget_steps"]


def test_validation_is_representative_disjoint_and_does_not_change_rng():
    before = torch.get_rng_state()
    ranks = [validation_indices(551, limit=128, seed=42, rank=r, world_size=2) for r in range(2)]
    assert len(ranks[0]) == len(ranks[1]) == 64
    assert not set(ranks[0]) & set(ranks[1])
    assert max(ranks[0]) > 400 and max(ranks[1]) > 400
    assert torch.equal(before, torch.get_rng_state())
    assert sorted(validation_indices(551, limit=0, seed=42)) == list(range(551))
    assert ranks[0] == validation_indices(551, limit=128, seed=42, world_size=2)


def test_epoch_budget_covers_the_actual_dataset_and_accounts_for_world_size():
    manifest = {
        "stages": {
            "pretrain": {"train": {"supervised_tokens": 7627201}},
            "sft": {"train": {"examples": 29449}},
        }
    }
    updates, coverage = budget_steps(manifest, "sft", 256, 8, epochs=1)
    assert updates == 3682 and coverage["examples_seen"] == 29456
    updates, coverage = budget_steps(manifest, "pretrain", 256, 8, steps=1000)
    assert updates == 1000 and coverage["predicted_tokens"] == 2040000
    assert coverage["epochs"] < 0.27
    with pytest.raises(ValueError, match="explicit"):
        budget_steps(manifest, "pretrain", 256, 8)


def test_reservoir_reads_the_full_source_and_is_reproducible(tmp_path, monkeypatch):
    raw = b"".join((json.dumps({"text": str(i)}) + "\n").encode() for i in range(1000))
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: io.BytesIO(raw))
    results = []
    for name in ["a", "b"]:
        root = tmp_path / name
        root.mkdir()
        path, meta = download_prefix(root, "sft", 30, "fixed", sampling="reservoir", seed=42)
        results.append(path.read_bytes())
        assert meta["source_rows"] == 1000 and meta["rows"] == 30
        assert max(int(json.loads(line)["text"]) for line in path.read_text().splitlines()) > 900
        with pytest.raises(ValueError, match="does not match"):
            download_prefix(root, "sft", 30, "fixed", sampling="prefix")
    assert results[0] == results[1]


def test_demo_can_compare_sft_and_dpo_and_does_not_reuse_stale_reviews(tmp_path):
    root = tmp_path / "minikimik3" / "run"
    quality = {"checkpoints": {}}
    for stage, step in [("sft", 500), ("dpo", 100)]:
        folder = root / stage
        folder.mkdir(parents=True)
        (folder / "run.json").write_text(
            json.dumps({"kind": "educational", "model_name": "minikimik3"})
        )
        (folder / "status.json").write_text(json.dumps({"stage": stage, "step": step}))
        artifact = folder / "model.pt"
        artifact.write_bytes(b"placeholder")
        stat = artifact.stat()
        quality["checkpoints"][stage] = dict(
            step=step,
            capability_status="failed",
            artifact=dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns),
        )
    (root / "quality.json").write_text(json.dumps(quality))
    assert checkpoints(tmp_path, "sft")["minikimik3"]["stage"] == "sft"
    assert checkpoints(tmp_path, "dpo")["minikimik3"]["capability_status"] == "failed"
    (root / "dpo/model.pt").write_bytes(b"replacement checkpoint")
    assert checkpoints(tmp_path, "dpo")["minikimik3"]["capability_status"] == "unassessed"
