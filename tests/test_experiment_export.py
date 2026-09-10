import csv
import json

import pytest

from scripts.export_experiments import export, export_registry


def test_registry_export_preserves_review_and_refuses_overwrite(tmp_path):
    workspace = tmp_path / "workspace"
    trial = workspace / "outputs/mf1-test/run"
    trial.mkdir(parents=True)
    (trial / "run.json").write_text(json.dumps(dict(model_name="minifrontier1")))
    (trial / "interruption.json").write_text(
        json.dumps(dict(state="stopped_for_review", reason=str(workspace)))
    )
    (workspace / "configs").mkdir()
    (workspace / "configs/experiments.json").write_text(
        json.dumps(
            dict(
                families=[
                    dict(
                        id="mf1", outputs=["mf1-*"], decision="diagnostic", limitations=["fixture"]
                    )
                ]
            )
        )
    )
    output = tmp_path / "snapshot"
    summary = export_registry(workspace, output)
    assert summary["records"] == 1 and summary["states"] == {"stopped_for_review": 1}
    raw = (output / "reviewed-registry.json").read_text()
    assert str(workspace) not in raw and "${WORKSPACE}" in raw
    with pytest.raises(FileExistsError):
        export_registry(workspace, output)


def test_export_reconciles_retired_queue_and_preserves_csv_token_axis(tmp_path):
    workspace = tmp_path / "workspace"
    queue = workspace / "outputs/strategy-single-gpu-v2"
    queue.mkdir(parents=True)
    jobs = []
    for name in ("complete", "running", "waiting"):
        root = queue / "minikimik3" / name
        jobs.append(
            dict(
                id=name, output=str(root), command=["train", "--output", str(root)], state="waiting"
            )
        )
        if name == "waiting":
            continue
        root.mkdir(parents=True)
        (root / "run.json").write_text(json.dumps(dict(model_name="minikimik3", kind="acceptance")))
        (root / "status.json").write_text(
            json.dumps(dict(state=name, token_ledger=dict(ce_tokens=100, input_tokens=120)))
        )
        (root / "metrics.jsonl").write_text(
            json.dumps(
                dict(
                    event="train",
                    step=5,
                    lm_loss=2.0,
                    token_ledger=dict(ce_tokens=100, input_tokens=120),
                )
            )
            + "\n"
        )
    (queue / "queue-plan.json").write_text(json.dumps(dict(jobs=jobs)))
    (queue / "queue.json").write_text(json.dumps(dict(jobs=jobs)))
    output = tmp_path / "snapshot"
    result = export(workspace, output)
    assert result["pending_jobs"] == [dict(id="waiting", state="waiting")]
    assert {x["id"]: x["state"] for x in result["queue_jobs"]} == dict(
        complete="complete", running="running", waiting="waiting"
    )
    exported = json.loads(
        (output / "strategy-single-gpu-v2--minikimik3--complete.json").read_text()
    )
    assert exported["run"]["kind"] == "acceptance"
    with (output / "strategy-single-gpu-v2--minikimik3--complete.csv").open() as f:
        row = next(csv.DictReader(f))
    assert row["ce_tokens"] == "100" and row["input_tokens"] == "120"
