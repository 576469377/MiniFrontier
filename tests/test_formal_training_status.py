import json

import pytest

from scripts.training_status import formal_row, formal_status, reverse_metrics, training_processes


def write_run(tmp_path, model="minideepseekv4", *, mf1=False):
    path = tmp_path / "outputs" / "strategy-base-pretraining-v1" / model / "P0"
    path.mkdir(parents=True)
    run = (
        dict(token_budget=10000, mf1_phase="p0")
        if mf1
        else dict(ce_token_budget=10000, pretraining_program=dict(phase="D1"))
    )
    (path / "run.json").write_text(json.dumps(run))
    (path / "status.json").write_text(
        json.dumps(
            dict(state="running", step=5, token_ledger=dict(ce_tokens=500, optimizer_updates=5))
        )
    )
    validation = dict(
        event="validation",
        step=4,
        main_ce_tokens=400,
        evaluation_scope="periodic",
        selection_sha256="fixed",
        **(dict(nll=3.5, ce_tokens=100) if mf1 else dict(lm_loss=3.5, supervised_tokens=100)),
    )
    train = dict(step=10, main_ce_tokens=1000, step_seconds=2)
    if mf1:
        train.update(
            train_lm_loss=3,
            ce_tokens=1000,
            phase_tokens=1000,
            optimizer_updates=10,
            ce_per_second=50,
        )
    else:
        train.update(
            event="train",
            lm_loss=3,
            tokens_per_second=50,
            token_ledger=dict(ce_tokens=1000, optimizer_updates=10),
        )
    (path / "metrics.jsonl").write_text(
        json.dumps(validation) + "\n" + json.dumps(train) + '\n{"partial":'
    )
    return path


@pytest.mark.parametrize("mf1", [False, True])
def test_live_formal_metrics_override_old_checkpoint_and_partial_line(tmp_path, mf1):
    path = write_run(tmp_path, "minifrontier1" if mf1 else "minideepseekv4", mf1=mf1)
    row = formal_row(path, {str(path): [123]})
    assert row["state"] == "running"
    assert row["step"] == 10 and row["ce_tokens"] == row["main_ce_tokens"] == 1000
    assert row["phase_progress"] == 0.1
    assert row["recent_tokens_per_second"] == 50
    assert row["estimate_remaining_update_hours"] == 0.05
    assert row["validation"] == dict(
        step=4,
        main_ce_tokens=400,
        nll=3.5,
        ce_tokens=100,
        scope="periodic",
        selection_sha256="fixed",
    )


@pytest.mark.parametrize("state", ["paused", "complete"])
def test_terminal_checkpoint_includes_updates_since_last_log(tmp_path, state):
    path = write_run(tmp_path)
    (path / "status.json").write_text(
        json.dumps(
            dict(state=state, step=11, token_ledger=dict(ce_tokens=1100, optimizer_updates=11))
        )
    )
    row = formal_row(path, {})
    assert row["state"] == ("completed" if state == "complete" else "paused")
    assert row["step"] == 11 and row["ce_tokens"] == row["main_ce_tokens"] == 1100
    assert formal_row(path, {str(path): [123]})["state"] == "running"


def test_stale_pause_does_not_hide_newer_train_or_resume_start(tmp_path):
    path = write_run(tmp_path)
    (path / "status.json").write_text('{"state":"paused","step":5}')
    assert formal_row(path, {})["state"] == "stopped"
    (path / "status.json").write_text('{"state":"paused","step":10}')
    (path / "metrics.jsonl").write_text('{"event":"start","step":10}\n')
    assert formal_row(path, {})["state"] == "stopped"
    assert formal_row(path, None)["state"] == "unknown"


def test_reverse_reader_crosses_buffers_to_find_validation(tmp_path):
    path = write_run(tmp_path)
    metrics = path / "metrics.jsonl"
    metrics.write_text(
        json.dumps(dict(event="validation", step=1, lm_loss=4))
        + "\n"
        + "\n".join(
            json.dumps(dict(event="train", step=i, padding="中" * 5000)) for i in range(2, 35)
        )
        + '\n{"broken":'
    )
    rows = list(reverse_metrics(metrics))
    assert [r["step"] for r in rows] == list(range(34, 0, -1))
    assert formal_row(path, {})["validation"]["nll"] == 4


@pytest.mark.parametrize(
    "prefix",
    [
        b"python\0-m\0minifrontier.training.train\0",
        b"python\0-m\0minifrontier\0train\0",
        b"python\0-m\0minifrontier\0mf1\0train\0",
        b"/env/bin/python\0/env/bin/minifrontier\0train\0",
    ],
)
def test_process_matching_uses_output_and_proc_absence_is_unknown(tmp_path, capsys, prefix):
    path = write_run(tmp_path)
    proc = tmp_path / "proc"
    (proc / "123").mkdir(parents=True)
    (proc / "123" / "cmdline").write_bytes(prefix + b"--output\0" + str(path).encode() + b"\0")
    assert training_processes(proc) == {str(path): [123]}
    assert training_processes(tmp_path / "missing") is None
    formal_status(tmp_path / "outputs", proc_root=proc)
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert rows[0]["reserve_gib"] == 80
    assert len(rows) == 5
    actual = next(r for r in rows if r.get("model") == "minideepseekv4")
    assert actual["state"] == "running" and actual["pids"] == [123]
    assert all(r["state"] == "not_started" for r in rows[1:] if r is not actual)


def test_partial_status_json_does_not_discard_valid_training_log(tmp_path):
    path = write_run(tmp_path)
    (path / "status.json").write_text('{"unfinished":')
    row = formal_row(path, {})
    assert row["step"] == 10 and row["main_ce_tokens"] == 1000
    assert row["state"] == "stopped"


def test_paused_event_preserves_unlogged_updates_with_partial_status(tmp_path):
    path = write_run(tmp_path)
    (path / "status.json").write_text('{"unfinished":')
    with (path / "metrics.jsonl").open("a") as stream:
        stream.write(
            "\n"
            + json.dumps(
                dict(
                    event="paused",
                    step=11,
                    main_ce_tokens=1100,
                    token_ledger=dict(ce_tokens=1100, optimizer_updates=11),
                )
            )
            + "\n"
        )
    row = formal_row(path, {})
    assert row["state"] == "paused"
    assert row["step"] == 11 and row["ce_tokens"] == row["main_ce_tokens"] == 1100
