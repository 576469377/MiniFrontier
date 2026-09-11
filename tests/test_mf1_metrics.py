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
