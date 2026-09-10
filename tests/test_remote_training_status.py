import json

from scripts.training_status import latest_train, strategy_status


def test_remote_queue_replaces_stale_local_waiting_entry(tmp_path, capsys):
    outputs = tmp_path / "outputs"
    original = outputs / "strategy-single-gpu-v2"
    original.mkdir(parents=True)
    target = original / "miniqwen4/reference"
    (original / "queue.json").write_text(
        json.dumps(
            dict(
                jobs=[
                    dict(
                        id="qwen/reference",
                        output=str(target),
                        state="waiting_predecessor",
                        gpu_id=2,
                    )
                ]
            )
        )
    )
    (original / "queue-plan.json").write_text('{"jobs": []}')
    mirror = outputs / "remote-example/outputs/strategy-remote-gpu-v1"
    mirror.mkdir(parents=True)
    remote_target = "/remote/outputs/strategy-single-gpu-v2/miniqwen4/reference"
    (mirror / "queue-plan.json").write_text(
        json.dumps(
            dict(
                workspace="/remote",
                host_label="remote-example",
                jobs=[dict(id="qwen/reference", token_budget=20000000)],
            )
        )
    )
    (mirror / "queue.json").write_text(
        json.dumps(
            dict(jobs=[dict(id="qwen/reference", output=remote_target, state="running", gpu_id=0)])
        )
    )
    strategy_status(outputs)
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    matches = [row for row in rows if row.get("trial") == "qwen/reference"]
    assert len(matches) == 1
    assert matches[0]["host"] == "remote-example"
    assert matches[0]["gpu_id"] == 0
    assert matches[0]["state"] == "running"


def test_mf1_flat_metrics_expose_their_actual_ledger(tmp_path):
    path = tmp_path / "metrics.jsonl"
    row = dict(step=15, train_lm_loss=2.5, ce_tokens=3456, optimizer_updates=15)
    path.write_text(json.dumps(row) + '\n{"incomplete":')
    latest = latest_train(path)
    assert latest["token_ledger"]["ce_tokens"] == 3456
    assert latest["token_ledger"]["optimizer_updates"] == 15
