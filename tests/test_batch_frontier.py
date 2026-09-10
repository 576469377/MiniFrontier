import json

import pytest

from scripts.run_batch_frontier import choose_real, execute, retire_weights, select_completed_batch


def test_throughput_selection_cannot_launch_long_recipe(tmp_path):
    with pytest.raises(ValueError, match="cannot promote"):
        execute(
            dict(
                mode="selected-real",
                real_command=["train", "--ce-tokens", "20000000"],
                output=str(tmp_path / "new"),
            )
        )
    assert not (tmp_path / "new").exists()


def test_screen_selection_excludes_oom_and_prefers_near_fastest_smaller_batch():
    cases = [
        dict(batch=16, state="measured", ce_per_second=100),
        dict(batch=32, state="measured", ce_per_second=180),
        dict(batch=64, state="measured", ce_per_second=300),
        dict(batch=128, state="measured", ce_per_second=310),
        dict(batch=256, state="memory_boundary"),
    ]
    assert choose_real(cases) == [16, 32, 64]
    assert choose_real([dict(batch=16, state="memory_boundary")]) == []


def test_ephemeral_retention_keeps_evidence_and_refuses_external_weights(tmp_path):
    trial = tmp_path / "trial"
    trial.mkdir()
    (trial / "checkpoint.pt").write_bytes(b"temporary weights")
    (trial / "metrics.jsonl").write_text('{"loss": 2}\n')
    retire_weights(trial)
    assert not (trial / "checkpoint.pt").exists()
    assert (trial / "metrics.jsonl").exists()
    retired = json.loads((trial / "retired-weights.json").read_text())
    assert retired["files"][0]["bytes"] == 17
    assert len(retired["files"][0]["sha256"]) == 64
    outside = tmp_path / "retained.pt"
    outside.write_bytes(b"keep")
    (trial / "best-model.pt").symlink_to(outside)
    with pytest.raises(ValueError, match="outside this probe"):
        retire_weights(trial)
    assert outside.read_bytes() == b"keep"


def test_real_selection_requires_matched_identity_and_completed_token_budget(tmp_path):
    cases = []
    for batch, speed in [(16, 100), (32, 200), (64, 206)]:
        p = tmp_path / "real" / f"mb{batch}"
        p.mkdir(parents=True)
        (p / "run.json").write_text(
            json.dumps(
                dict(model_name="minikimik3", seed=42, data_sha256="same", ce_token_budget=1000000)
            )
        )
        (p / "performance.json").write_text("{}")
        (p / "metrics.jsonl").write_text(json.dumps(dict(event="validation", lm_loss=5.0)) + "\n")
        cases.append(
            dict(
                batch=batch,
                state="complete",
                ce_per_second=speed,
                peak_reserved_gib=10,
                measured_updates=8,
                token_ledger=dict(ce_tokens=1000000),
            )
        )
    (tmp_path / "status.json").write_text(json.dumps(dict(state="complete", real=cases)))
    assert select_completed_batch([tmp_path])["selected_batch"] == 32
    (tmp_path / "real/mb64/run.json").write_text(
        json.dumps(
            dict(model_name="minikimik3", seed=43, data_sha256="same", ce_token_budget=1000000)
        )
    )
    with pytest.raises(ValueError, match="differs"):
        select_completed_batch([tmp_path])
