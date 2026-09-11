import hashlib
import json
import subprocess
import sys
import time

import pytest

from scripts.experiment_registry import collect, refresh


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_standalone_data_audit_uses_verified_process_and_preserves_scanned_counts(tmp_path):
    target = tmp_path / "outputs/strategy-pretraining-text-leakage-v1/audit"
    target.mkdir(parents=True)
    driver = target / "audit.py"
    driver.write_text("import time\ntime.sleep(60)\n")
    process = subprocess.Popen([sys.executable, str(driver)])
    record = dict(
        kind="data_construction",
        state="checking_training_holdouts",
        pid=process.pid,
        driver_sha256=hashlib.sha256(driver.read_bytes()).hexdigest(),
        scanned_records=dict(train=1000, val=20, test=30),
        matched_pairs=2,
    )
    try:
        write(target / "run.json", record)
        entry = collect(tmp_path)["experiments"][0]
        assert entry["state"] == "running"
        assert entry["data_progress"]["scanned_records"] == record["scanned_records"]
        assert entry["data_progress"]["operation_state"] == "checking_training_holdouts"
        assert entry["ce_tokens"] == 0 and not entry["main_budget_eligible"]
        record["driver_sha256"] = "0" * 64
        write(target / "run.json", record)
        assert collect(tmp_path)["experiments"][0]["state"] == "unverified_process_identity"
    finally:
        process.terminate()
        process.wait(timeout=5)
    assert collect(tmp_path)["experiments"][0]["state"] == "interrupted_without_final_status"
    record.update(state="split_conflicts_require_partition_update", completed_unix=time.time())
    write(target / "run.json", record)
    assert (
        collect(tmp_path)["experiments"][0]["state"] == "split_conflicts_require_partition_update"
    )


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


@pytest.mark.parametrize("bound_arguments", [False, True])
def test_candidate_data_progress_is_not_training_ce_or_formal_admission(tmp_path, bound_arguments):
    run = tmp_path / "outputs/strategy-pretraining-data-v1"
    write(
        run / "run.json",
        dict(
            kind="data_construction",
            pid=999999999,
            **(
                dict(arguments=dict(output=str(tmp_path / "data/candidate")))
                if bound_arguments
                else dict(data_output="data/candidate")
            ),
        ),
    )
    write(
        tmp_path / "data/candidate/source-audit.json",
        dict(
            status="interrupted_unadmitted",
            formal_admission=False,
            sources={"zh": dict(accepted_records=10, accepted_reference_tokens=12345)},
            error="schema mismatch",
            updated_unix=time.time(),
        ),
    )
    entry = collect(tmp_path)["experiments"][0]
    assert entry["kind"] == "data_construction" and entry["model"] == "shared-corpus"
    assert entry["ce_tokens"] == 0 and entry["main_budget_eligible"] is False
    assert entry["data_progress"]["candidate_reference_tokens"] == 12345
    assert not entry["data_progress"]["formal_admission"]
    assert entry["state"] == "interrupted_unadmitted"


@pytest.mark.parametrize("bound_arguments", [False, True])
def test_failed_data_attempt_cannot_inherit_a_later_success_at_the_same_output(
    tmp_path, bound_arguments
):
    parent = tmp_path / "outputs/strategy-pretraining-data-v1"
    target = (
        dict(arguments=dict(output="data/candidate"))
        if bound_arguments
        else dict(data_output="data/candidate")
    )
    write(
        parent / "first/run.json",
        dict(
            kind="data_construction",
            state="failed",
            error="bound audit rejected",
            **target,
        ),
    )
    write(parent / "retry/run.json", dict(kind="data_construction", state="complete", **target))
    write(
        tmp_path / "data/candidate/source-audit.json",
        dict(
            status="candidate_slice_complete_pending_admission",
            counts=dict(accepted=17),
            formal_admission=False,
        ),
    )
    entries = {row["output"].rsplit("/", 1)[-1]: row for row in collect(tmp_path)["experiments"]}
    failed, retry = entries["first"], entries["retry"]
    assert (
        failed["state"] == "failed" and failed["data_progress"]["error"] == "bound audit rejected"
    )
    assert failed["data_progress"]["accepted_records"] == 0
    assert retry["state"] == "candidate_slice_complete_pending_admission"
    assert retry["data_progress"]["accepted_records"] == 17
    assert failed["ce_tokens"] == retry["ce_tokens"] == 0


def test_performance_updates_stay_outside_the_formal_training_budget(tmp_path):
    run = tmp_path / "outputs/strategy-mf1-qualification/p0"
    write(run / "run.json", dict(kind="performance", model_name="minifrontier1"))
    write(
        run / "status.json",
        dict(
            state="measurement_complete_unqualified",
            ledger=dict(ce_tokens=4_000_000, optimizer_updates=250),
        ),
    )
    entry = collect(tmp_path)["experiments"][0]
    assert entry["kind"] == "performance" and entry["main_budget_eligible"] is False
    assert entry["ce_tokens"] == 4_000_000 and entry["optimizer_updates"] == 250
    assert entry["state"] == "measurement_complete_unqualified"


@pytest.mark.parametrize("remote", [False, True])
def test_partition_inventory_reports_retained_members_not_raw_construction_counts(tmp_path, remote):
    base = tmp_path / ("outputs/remote-test/outputs" if remote else "outputs")
    target = base / "strategy-pretraining-data-v1"
    output = tmp_path / "data/refined"
    write(
        target / "run.json",
        dict(kind="data_construction", state="complete", data_output=str(output)),
    )
    audit = dict(
        status="candidate_slice_complete_pending_admission",
        counts=dict(accepted=100),
        sources={"source": dict(accepted_reference_tokens=10000)},
        unique_images=50,
        corpus=dict(format="corpus-partition-view-v3", splits=dict(train=70, val=5, test=10)),
        split_reference_tokens=dict(
            train={"source": 700}, val={"source": 50}, test={"source": 100}
        ),
        split_independent_images=dict(train=35, val=2, test=5),
        split_answer_reference_tokens=dict(train={"vqa": 350}, val={"vqa": 20}, test={"vqa": 50}),
    )
    write(target / "data-audit.json" if remote else output / "source-audit.json", audit)
    entry = collect(tmp_path)["experiments"][0]
    progress = entry["data_progress"]
    assert progress["accepted_records"] == 85
    assert progress["raw_construction_records"] == 100
    assert progress["effective_split_records"] == dict(train=70, val=5, test=10)
    assert progress["candidate_reference_tokens"] == 850
    assert progress["unique_images"] == 42
    assert progress["candidate_answer_reference_tokens"] == 420
    assert entry["ce_tokens"] == 0 and not entry["main_budget_eligible"]
    write(
        target / "run.json",
        dict(kind="data_construction", state="failed", data_output=str(output)),
    )
    failed = collect(tmp_path)["experiments"][0]
    assert failed["data_progress"]["accepted_records"] == 0
    assert "effective_split_records" not in failed["data_progress"]


def test_supervisor_resumed_training_overrides_old_paused_checkpoint(tmp_path):
    trial = tmp_path / "outputs/mf1-language-instructions-v1/trial"
    write(trial / "run.json", dict(model_name="minifrontier1", kind="acceptance"))
    write(trial / "status.json", dict(state="paused", ledger=dict(ce_tokens=2)))
    write(trial / "supervisor.json", dict(state="running", updated_at=time.time()))
    (trial / "metrics.jsonl").write_text(json.dumps(dict(train_lm_loss=1, ce_tokens=20, step=10)))
    entry = collect(tmp_path)["experiments"][0]
    assert entry["state"] == "running" and entry["ce_tokens"] == 20


def test_remote_data_requires_fresh_remote_observation_and_keeps_candidate_counts(tmp_path):
    run = tmp_path / "outputs/remote-test/outputs/strategy-pretraining-media-v1"
    write(run / "run.json", dict(kind="data_construction", pid=12345, data_output="data/media"))
    write(
        run / "data-audit.json",
        dict(
            status="building",
            unique_images=500,
            media_bytes=123456,
            formal_admission=False,
            updated_unix=time.time(),
            sources={"images": dict(accepted_records=750)},
        ),
    )
    entry = collect(tmp_path)["experiments"][0]
    assert entry["state"] == "unverified_remote_process"
    assert entry["data_progress"]["unique_images"] == 500 and entry["ce_tokens"] == 0
    assert not entry["main_budget_eligible"]
    write(
        run / "process-observation.json",
        dict(pid=12345, argv_matches=True, observed_unix=time.time()),
    )
    assert collect(tmp_path)["experiments"][0]["state"] == "observed_running"
    write(
        run / "process-observation.json",
        dict(pid=12345, argv_matches=True, observed_unix=time.time() - 121),
    )
    assert collect(tmp_path)["experiments"][0]["state"] == "unverified_remote_process"


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


def test_readme_and_external_install_trials_are_included(tmp_path):
    for name in [
        "public-readme-check",
        "preview-quickstart-cpu-v1",
        "preview-installed-wheel-quickstart",
    ]:
        write(tmp_path / "outputs" / name / "train/run.json", dict(kind="acceptance"))
    assert len(collect(tmp_path)["experiments"]) == 3


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


@pytest.mark.parametrize("nested", [False, True])
def test_conflicting_gpu_assignments_are_visible(tmp_path, nested):
    queue = tmp_path / "outputs/strategy-example-gpu-v1"
    if nested:
        queue = queue / "second-stage"
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
