import json
import time

from scripts.experiment_registry import collect, refresh


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_stopped_trial_is_not_complete_and_preserves_last_observed_tokens(tmp_path):
    trial = tmp_path / "outputs/strategy-example-gpu-v1/model"
    write(
        trial / "run.json", dict(model_name="miniqwen4", seed=42, batch_size=2, kind="acceptance")
    )
    write(trial / "status.json", dict(state="running", token_ledger=dict(ce_tokens=100)))
    write(trial / "interruption.json", dict(state="stopped_by_user", reason="batch replan"))
    (trial / "metrics.jsonl").write_text(
        json.dumps(dict(event="train", token_ledger=dict(ce_tokens=150))) + "\n"
    )
    first = refresh(tmp_path, tmp_path / "outputs/experiment-registry")
    second = refresh(tmp_path, tmp_path / "outputs/experiment-registry")
    entry = first["experiments"][0]
    assert entry["state"] == "stopped_by_user"
    assert entry["ce_tokens"] == 150
    assert entry["main_budget_eligible"] is False
    assert entry["id"] == second["experiments"][0]["id"]
    assert (
        len((tmp_path / "outputs/experiment-registry/events.jsonl").read_text().splitlines()) == 1
    )


def test_supervisor_resumed_training_overrides_old_paused_checkpoint(tmp_path):
    trial = tmp_path / "outputs/mf1-language-instructions-v1/trial"
    write(trial / "run.json", dict(model_name="minifrontier1", kind="acceptance"))
    write(trial / "status.json", dict(state="paused", ledger=dict(ce_tokens=2)))
    write(trial / "supervisor.json", dict(state="running", updated_at=time.time()))
    (trial / "metrics.jsonl").write_text(json.dumps(dict(train_lm_loss=1, ce_tokens=20, step=10)))
    entry = collect(tmp_path)["experiments"][0]
    assert entry["state"] == "running" and entry["ce_tokens"] == 20


def test_review_reports_coverage_and_global_batch_overshoot(tmp_path):
    trial = tmp_path / "outputs/strategy-example-gpu-v1/trial"
    write(
        trial / "run.json",
        dict(model_name="minikimik3", input_batch_tokens=16384, sequence_length=512),
    )
    write(trial / "performance.json", dict(updates=[dict(input_tokens=48000)]))
    write(
        tmp_path / "configs/experiments.json",
        dict(
            families=[
                dict(
                    id="screen",
                    outputs=["strategy-example-*"],
                    decision="execution_only",
                    limitations=["short probe"],
                )
            ]
        ),
    )
    result = collect(tmp_path)
    entry = result["experiments"][0]
    assert entry["review_family"] == "screen"
    assert "whole_microbatch_global_batch_overshoot" in entry["audit_findings"]
    assert result["issues"] == []


def test_remote_execution_supersedes_empty_local_reservation(tmp_path):
    outputs = tmp_path / "outputs"
    for _host, root, workspace, status in [
        ("local", outputs, str(tmp_path), "waiting_predecessor"),
        ("remote-test", outputs / "remote-test/outputs", "/remote", "running"),
    ]:
        queue = root / "strategy-example-gpu-v1"
        write(
            queue / "queue-plan.json",
            dict(
                workspace=workspace,
                jobs=[
                    dict(
                        id="trial",
                        output=workspace + "/outputs/strategy-example-gpu-v1/trial",
                        model="minikimik3",
                    )
                ],
            ),
        )
        write(
            queue / "queue.json",
            dict(updated_at=time.time(), jobs=[dict(id="trial", state=status, gpu_id=0)]),
        )
    result = collect(tmp_path)
    assert len(result["experiments"]) == 1
    assert result["experiments"][0]["host"] == "remote-test"
    assert len(result["experiments"][0]["superseded_reservations"]) == 1


def test_conflicting_gpu_assignments_are_visible(tmp_path):
    queue = tmp_path / "outputs/strategy-example-gpu-v1"
    jobs = [dict(id=name, output=str(queue / name), model="miniqwen4") for name in ["a", "b"]]
    write(queue / "queue-plan.json", dict(workspace=str(tmp_path), jobs=jobs))
    write(
        queue / "queue.json",
        dict(
            updated_at=time.time(), jobs=[dict(id=j["id"], state="running", gpu_id=0) for j in jobs]
        ),
    )
    result = collect(tmp_path)
    assert any(i["kind"] == "multiple_active_tasks_on_gpu" for i in result["issues"])


def test_sweep_and_child_keep_separate_ids_and_one_comparison_group(tmp_path):
    parent = tmp_path / "outputs/strategy-example-gpu-v1/trials/model"
    child = parent / "real/mb32"
    write(
        parent / "experiment.json",
        dict(
            model="minikimik3",
            cohort="batch-search",
            comparison_group="same-data",
            real_command=["train", "--ce-tokens", "1000000"],
        ),
    )
    write(
        parent / "status.json",
        dict(state="complete", real_candidates=[32], real=[dict(batch=32, state="complete")]),
    )
    write(child / "run.json", dict(model_name="minikimik3", kind="acceptance", batch_size=32))
    write(
        child / "status.json",
        dict(state="complete", token_ledger=dict(ce_tokens=1001234, optimizer_updates=40)),
    )
    write(
        parent.parent.parent / "queue-plan.json",
        dict(workspace=str(tmp_path), jobs=[dict(id="model", output=str(parent))]),
    )
    rows = collect(tmp_path)["experiments"]
    outer = next(e for e in rows if e["kind"] == "sweep")
    inner = next(e for e in rows if e["kind"] == "training")
    assert outer["id"] != inner["id"]
    assert inner["parent_experiment_id"] == outer["id"]
    assert outer["comparison_group"] == inner["comparison_group"] == "same-data"
    assert outer["ce_tokens"] == inner["ce_tokens"] == 1001234
    assert outer["ce_token_budget"] == 1000000
    write(
        parent / "status.json",
        dict(state="running", real_candidates=[32], real=[dict(batch=32, state="running")]),
    )
    (child / "status.json").unlink()
    (child / "metrics.jsonl").write_text(
        json.dumps(dict(event="train", token_ledger=dict(ce_tokens=1234))) + "\n"
    )
    active = next(e for e in collect(tmp_path)["experiments"] if e["id"] == inner["id"])
    assert active["state"] == "running"
    assert active["ce_tokens"] == 1234
