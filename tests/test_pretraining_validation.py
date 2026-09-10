"""CE validation preserves sampling, denominators, cadence and exact recovery."""

import json

import numpy as np
import pytest
import torch
from test_mf1_workflow import assert_tree_equal
from test_train_integration import arguments, config_for
from test_train_integration import corpus as corpus

from minifrontier.data import sha256
from minifrontier.training import train, validation
from minifrontier.training.validation import CEValidation, validation_due


class Dataset:
    def __init__(self):
        self.ce_counts = [0, 2, 3, 5, 4, 1, 7]
        self.domains = ["zero", "a", "b", "a", "b", "a", "b"]

    def __len__(self):
        return len(self.ce_counts)


def test_fixed_ce_selections_are_nested_and_disjoint_across_ranks():
    dataset = Dataset()
    policy = dict(validation.PRETRAINING_VALIDATION, periodic_ce_tokens=7, phase_end_ce_tokens=18)
    rng = torch.get_rng_state()
    selection = CEValidation(dataset, seed=7, policy=policy)
    assert torch.equal(rng, torch.get_rng_state())
    assert set(selection.selections["periodic"]) <= set(selection.selections["phase_end"])
    for final, scope in [(False, "periodic"), (True, "phase_end")]:
        partitions = []
        for rank in range(3):
            indices = []
            for domain, batch in selection.batches(
                final=final, batch_size=2, rank=rank, world_size=3
            ):
                assert 1 <= len(batch) <= 2
                assert {dataset.domains[i] for i in batch} == {domain}
                indices.extend(batch)
            partitions.extend(indices)
        assert len(partitions) == len(set(partitions))
        assert set(partitions) == set(selection.selections[scope]) and 0 not in partitions
        actual = sum(dataset.ce_counts[i] for i in partitions)
        requested = selection.binding[scope]["requested_ce_tokens"]
        assert requested <= actual < requested + max(dataset.ce_counts)
        assert actual == selection.binding[scope]["actual_ce_tokens"]
    with pytest.raises(ValueError, match="inventory"):
        CEValidation(dataset, seed=7)


def test_validation_cadence_uses_main_ce_across_phase_and_resume_boundaries():
    policy = validation.PRETRAINING_VALIDATION
    for before, after, expected in [
        (0, 1, False),
        (9_999_999, 10_000_100, True),
        (99_990_000, 100_010_000, True),
        (109_000_000, 110_000_000, False),
        (149_990_000, 150_020_000, True),
        (200_000_000, 200_010_000, False),
    ]:
        assert validation_due(before, after, policy) is expected


def _ragged(root):
    manifest = json.loads((root / "manifest.json").read_text())
    for split in ("train", "val"):
        prefix = f"pretrain.{split}"
        ids = np.fromfile(root / f"{prefix}.bin", dtype=np.int32).reshape(-1, 128)
        ids[:, 0], ids[:, -1] = 1, 2
        ids.tofile(root / f"{prefix}.bin")
        labels = ids.copy()
        labels[:, 0] = -100
        labels.tofile(root / f"{prefix}.labels.bin")
        np.save(root / f"{prefix}.index.npy", np.array([(i * 128, 128) for i in range(len(ids))]))
        (root / f"{prefix}.jsonl").write_text(
            "".join(json.dumps(dict(task="a" if i % 2 else "b")) + "\n" for i in range(len(ids)))
        )
        paths = {
            "file": f"{prefix}.bin",
            "labels_file": f"{prefix}.labels.bin",
            "index_file": f"{prefix}.index.npy",
            "metadata_file": f"{prefix}.jsonl",
        }
        manifest["stages"]["pretrain"][split] = dict(
            format="document-ragged-v2",
            stored_positions=ids.size,
            examples=len(ids),
            **paths,
            **{
                key.replace("file", "sha256") if key != "file" else "sha256": sha256(root / path)
                for key, path in paths.items()
            },
        )
    (root / "manifest.json").write_text(json.dumps(manifest))


def test_real_ce_evaluation_batches_and_recovery_do_not_change_training(
    corpus, tmp_path, monkeypatch
):
    _ragged(corpus)
    monkeypatch.setattr(
        validation,
        "PRETRAINING_VALIDATION",
        dict(
            periodic_ce_tokens=127,
            phase_end_ce_tokens=381,
            early_interval_ce=508,
            early_until_ce=1016,
            later_interval_ce=1016,
        ),
    )
    config = config_for("minikimik3", tmp_path)

    def args(output):
        return [
            *arguments("minikimik3", config, corpus, output, steps=4),
            "--batch-size",
            "2",
            "--ce-tokens",
            "1016",
            "--pretraining-eval",
            "--eval-batches",
            "0",
        ]

    reference, recovered = tmp_path / "reference", tmp_path / "recovered"
    train.main(args(reference))
    train.main([*args(recovered), "--stop-after-updates", "1"])
    train.main([*args(recovered), "--resume", str(recovered / "checkpoint.pt")])
    a, b = [torch.load(p / "checkpoint.pt", weights_only=True) for p in (reference, recovered)]
    for key in ("model", "optimizer", "rng", "token_ledger", "router_balance"):
        assert_tree_equal(a[key], b[key])
    events = [json.loads(line) for line in (reference / "metrics.jsonl").read_text().splitlines()]
    evaluations = [v for v in events if v["event"] == "validation"]
    assert [(v["step"], v["evaluation_scope"], v["supervised_tokens"]) for v in evaluations] == [
        (0, "periodic", 127),
        (1, "periodic", 127),
        (2, "phase_end", 381),
    ]
    for value in evaluations:
        assert (
            sum(d["ce_tokens"] for d in value["per_domain"].values()) == value["supervised_tokens"]
        )
        expected = (
            sum(d["ce_tokens"] * d["lm_loss"] for d in value["per_domain"].values())
            / value["supervised_tokens"]
        )
        assert value["lm_loss"] == pytest.approx(expected, rel=1e-6)
    assert (
        json.loads((reference / "best-validation.json").read_text())["evaluation_scope"]
        == "periodic"
    )
    with pytest.raises(ValueError, match="resume recipe differs"):
        train.main([*args(recovered), "--seed", "43", "--resume", str(recovered / "checkpoint.pt")])


def test_main_phase_gate_keeps_separate_periodic_and_phase_end_minima(tmp_path, monkeypatch):
    from minifrontier.training import strategy_gate

    monkeypatch.setattr(strategy_gate, "require_source_checkout", lambda: tmp_path)
    monkeypatch.setattr(
        strategy_gate, "source_identity", lambda: dict(commit="fixture", dirty=False)
    )
    (tmp_path / "strategy.md").write_text("fixture strategy")
    config = tmp_path / "config.json"
    config.write_text("{}")
    (tmp_path / "manifest.json").write_text(json.dumps(dict(tokenizer=dict(vocab_size=300))))
    tokenizer = tmp_path / "tokenizer-evidence.json"
    tokenizer.write_text(json.dumps(dict(frozen=True, selected=300)))
    profile = tmp_path / "performance.json"
    profile.write_text(
        json.dumps(
            dict(
                measured_updates=200,
                recipe=dict(
                    performance_profile=dict(warmup=50),
                    data_sha256=sha256(tmp_path / "manifest.json"),
                    phase="dense_pretrain",
                    source=dict(commit="fixture"),
                    config={},
                ),
            )
        )
    )
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            dict(
                model="miniqwen4",
                source_document="strategy.md",
                source_document_sha256=sha256(tmp_path / "strategy.md"),
                min_disk_free_gib=0,
                phases=[
                    dict(
                        id="Q1",
                        budget_scope="main",
                        attention_phase="dense_pretrain",
                        depends_on=[],
                        required_evidence=[],
                    )
                ],
            )
        )
    )
    evidence = tmp_path / "evidence.json"
    record = dict(
        source_commit="fixture",
        data_sha256=sha256(tmp_path / "manifest.json"),
        config_sha256=sha256(config),
        tokenizer_comparison=str(tokenizer),
        performance=str(profile),
        data_audit=dict(
            source_licenses_reviewed=True,
            split_groups_disjoint=True,
            sealed_test=True,
            readable_200_passed=True,
            minimum_source_holdout_fraction=0.005,
            periodic_validation_ce_tokens=1_000_000,
            phase_end_validation_ce_tokens=5_000_000,
        ),
    )
    for periodic, end, allowed in [
        (1_000_000, 5_000_000, True),
        (999_999, 5_000_000, False),
        (1_000_000, 4_999_999, False),
    ]:
        record["data_audit"].update(
            periodic_validation_ce_tokens=periodic, phase_end_validation_ce_tokens=end
        )
        evidence.write_text(json.dumps(record))
        result = strategy_gate.check(
            plan, "Q1", evidence, data=tmp_path, config=config, output=tmp_path
        )
        assert result["allowed"] is allowed
        if not allowed:
            assert len(result["errors"]) == 1 and "validation" in result["errors"][0]
