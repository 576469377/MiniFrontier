import json

import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

from minifrontier.training.metrics import mf1_scalars, training_scalars
from scripts.sync_mf1_tensorboard import RunView, publish_view


def test_mf1_groups_and_validation_denominators():
    train = mf1_scalars(
        dict(
            step=25,
            optimizer_updates=25,
            train_lm_loss=2.0,
            ce_tokens=4255,
            grad_norm=1.2,
            ce_per_second=4.1,
            peak_reserved_gib=4.3,
            data_preparation_seconds=0.2,
            ce_fraction=0.25,
            padding_fraction=0.1,
        )
    )
    validation = mf1_scalars(
        dict(
            event="validation",
            step=25,
            nll=3.0,
            ce_tokens=494,
            examples=138,
            capability_qualified=False,
            selection="full_validation",
            per_domain={"language": 3.5, "vision": 0.3},
            black_media_nll={"vision": 2.0},
        )
    )
    assert train["train/ce_tokens"] == 4255
    assert train["perf/data_preparation_seconds"] == 0.2
    assert train["perf/ce_fraction"] == 0.25
    assert train["perf/padding_fraction"] == 0.1
    assert validation["eval/ce_tokens"] == 494
    assert validation["eval/lm_loss_language"] == 3.5
    assert validation["eval/black_media_lm_loss_vision"] == 2.0
    assert {tag.split("/")[0] for tag in train | validation} == {"train", "eval", "perf"}
    assert not any(tag.split("/")[-1] == "step" or "qualified" in tag for tag in train | validation)


def test_view_rebuilds_old_mixed_counters_and_tails_without_duplicate_points(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    path = source / "metrics.jsonl"
    rows = [
        dict(step=1, train_lm_loss=4.0, ce_tokens=100),
        dict(step=1, event="validation", nll=3.0, ce_tokens=7, per_domain={"language": 3.0}),
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    original_bytes = path.read_bytes()
    with SummaryWriter(str(source / "tensorboard")) as original:
        original.add_scalar("train_lm_loss", 4.0, 1, walltime=100)
        original.add_scalar("ce_tokens", 100, 1, walltime=100)
        original.add_scalar("nll", 3.0, 1, walltime=110)
        original.add_scalar("ce_tokens", 7, 1, walltime=110)
        original.flush()
        view = RunView(source, tmp_path / "view")
        assert view.sync()["rows"] == 2
        assert view.sync()["rows"] == 2
        assert path.read_bytes() == original_bytes
        next_row = json.dumps(dict(step=2, train_lm_loss=2.0, ce_tokens=200))
        with path.open("a") as f:
            f.write(next_row)
        assert view.sync()["rows"] == 2
        with path.open("a") as f:
            f.write("\n")
        assert view.sync()["rows"] == 2  # Wait for matching original event timestamp.
        original.add_scalar("train_lm_loss", 2.0, 2, walltime=120)
        original.flush()
        assert view.sync()["rows"] == 3
        view.close()
    events = EventAccumulator(str(tmp_path / "view"), size_guidance={"scalars": 0}).Reload()
    assert [(r.step, r.value, r.wall_time) for r in events.Scalars("train/lm_loss")] == [
        (1, 4, 100),
        (2, 2, 120),
    ]
    assert [(r.step, r.value) for r in events.Scalars("train/ce_tokens")] == [(1, 100), (2, 200)]
    assert events.Scalars("eval/ce_tokens")[0].value == 7
    assert events.Scalars("eval/lm_loss_language")[0].wall_time == 110


def test_view_publisher_cannot_replace_real_training_data(tmp_path):
    source, view = tmp_path / "source", tmp_path / "view"
    source.mkdir()
    view.mkdir()
    with pytest.raises(ValueError, match="real directory"):
        publish_view(source, view)
    alias = tmp_path / "alias"
    alias.symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="parent"):
        publish_view(alias / "child", view)
    link = tmp_path / "display"
    link.symlink_to(source, target_is_directory=True)
    publish_view(link, view)
    assert source.is_dir() and link.resolve() == view


def test_native_metrics_preserve_values_and_skip_operational_counters():
    for event in ("start", "qk_clip", "cuda_cache_reclaim", "paused", "rank_verification"):
        assert training_scalars(dict(event=event, step=5, main_ce_tokens=100)) == {}
    train = training_scalars(
        dict(
            event="train",
            step=5,
            lm_loss=3.2,
            loss=3.5,
            data_offset=15,
            tokens_per_second=4000,
            optimizer_seconds=0.5,
            peak_allocated_mib=4096,
        )
    )
    assert train == {
        "train/lm_loss": 3.2,
        "train/loss": 3.5,
        "perf/tokens_per_second": 4000,
        "perf/optimizer_seconds": 0.5,
        "perf/peak_allocated_gib": 4,
    }
    metrics = dict(
        event="validation",
        step=5,
        evaluation_scope="phase_end",
        lm_loss=3.0,
        per_domain={"caption": dict(lm_loss=2.0, ce_tokens=7)},
        lm_loss_caption=2.0,
        ce_tokens_caption=7,
        has_learning_signal=True,
        loss_denominator="ce_tokens",
        reward=0.0,
    )
    assert training_scalars(metrics) == {
        "eval/phase_end_lm_loss": 3.0,
        "eval/phase_end_lm_loss_caption": 2.0,
        "eval/phase_end_ce_tokens_caption": 7,
    }
    metrics.update(loss_denominator="active_responses", reward=0.5)
    assert training_scalars(metrics)["eval/phase_end_reward"] == 0.5


def test_native_ledger_counters_match_flat_mf1_counters_without_duplicate_updates():
    ledger = dict(
        ce_tokens=100,
        input_tokens=130,
        response_tokens=0,
        optimizer_updates=5,
        skipped_windows=0,
        image_occurrences=2,
        video_examples=1,
        video_frames=8,
        image_features=196,
    )
    native = training_scalars(dict(event="train", step=5, token_ledger=ledger))
    assert native == training_scalars(dict(step=5, **ledger))
    assert native == {f"train/{k}": v for k, v in ledger.items() if k != "optimizer_updates"}
    overridden = training_scalars(dict(event="train", ce_tokens=101, token_ledger=ledger))
    assert overridden["train/ce_tokens"] == 101
    assert "train/lr" not in native and "perf/optimizer_seconds" not in native


@pytest.mark.parametrize(
    "stage", [None, "pretrain", "sft", "sparse_cpt", "dense_distill", "dpo", "grpo", "mopd", "opd"]
)
def test_native_throughput_requires_known_ce_stage(stage):
    # Even inherited nonzero CE counters cannot establish this update's unit.
    values = dict(event="train", tokens_per_second=40, token_ledger=dict(ce_tokens=100))
    scalars = training_scalars(values, stage=stage)
    tag = "ce_per_second" if stage in {"pretrain", "sft", "sparse_cpt"} else "tokens_per_second"
    assert scalars == {"train/ce_tokens": 100, f"perf/{tag}": 40}
    assert training_scalars(dict(values, ce_per_second=42), stage="pretrain") == {
        "train/ce_tokens": 100,
        "perf/ce_per_second": 42,
    }


def test_input_throughput_uses_current_batch_and_preserves_recorded_value():
    values = dict(
        event="train",
        input_batch_actual=160,
        step_seconds=2,
        token_ledger=dict(input_tokens=999999),
    )
    assert training_scalars(values)["perf/input_per_second"] == 80
    assert "input_per_second" not in values
    values["input_per_second"] = 75
    assert training_scalars(values)["perf/input_per_second"] == 75


@pytest.mark.parametrize(
    ("inputs", "seconds"),
    [
        (None, 2),
        (160, None),
        (160, 0),
        (160, -1),
        (True, 2),
        (160, float("nan")),
        (float("inf"), 2),
    ],
)
def test_input_throughput_does_not_invent_missing_or_invalid_measurements(inputs, seconds):
    scalars = training_scalars(dict(event="train", input_batch_actual=inputs, step_seconds=seconds))
    assert "perf/input_per_second" not in scalars


@pytest.mark.parametrize("scope", ["periodic", "phase_end"])
@pytest.mark.parametrize(
    "denominator", ["ce_tokens", "response_positions", "active_responses", "pairs", None]
)
def test_validation_aliases_preserve_non_ce_denominators(scope, denominator):
    values = dict(
        event="validation",
        evaluation_scope=scope,
        loss_denominator=denominator,
        supervised_tokens=17,
        seconds=2.5,
    )
    prefix = "eval/phase_end_" if scope == "phase_end" else "eval/"
    ce = denominator == "ce_tokens"
    assert training_scalars(values) == {
        prefix + ("ce_tokens" if ce else "supervised_tokens"): 17,
        prefix + ("duration_seconds" if ce else "seconds"): 2.5,
    }
    if ce:
        values.update(ce_tokens=19, duration_seconds=3.0)
        assert training_scalars(values) == {
            prefix + "ce_tokens": 19,
            prefix + "duration_seconds": 3.0,
        }


def test_native_view_skips_diagnostics_and_preserves_actual_event_times(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    rows = [
        dict(event="start", step=0, parameters=12),
        dict(event="validation", step=0, lm_loss=4.0),
        dict(event="qk_clip", step=1, main_ce_tokens=0),
        dict(event="cuda_cache_reclaim", step=1, main_ce_tokens=10),
        dict(event="train", step=1, lm_loss=3.0, peak_allocated_mib=4096),
        dict(event="validation", evaluation_scope="phase_end", step=1, lm_loss=2.0),
    ]
    path = source / "metrics.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    original_bytes = path.read_bytes()
    with SummaryWriter(str(source / "tensorboard")) as original:
        original.add_scalar("validation/lm_loss", 4.0, 0, walltime=100)
        original.add_scalar("train/lm_loss", 3.0, 1, walltime=110)
        original.add_scalar("validation/lm_loss", 2.0, 1, walltime=120)
    view = RunView(source, tmp_path / "view")
    assert view.sync()["rows"] == len(rows)
    assert view.sync()["rows"] == len(rows)
    view.close()
    assert path.read_bytes() == original_bytes
    events = EventAccumulator(str(tmp_path / "view"), size_guidance={"scalars": 0}).Reload()
    assert events.Scalars("eval/lm_loss")[0].wall_time == 100
    assert events.Scalars("train/lm_loss")[0].wall_time == 110
    assert events.Scalars("eval/phase_end_lm_loss")[0].wall_time == 120
    assert events.Scalars("perf/peak_allocated_gib")[0].value == 4
    assert {tag.split("/")[0] for tag in events.Tags()["scalars"]} == {"train", "eval", "perf"}
