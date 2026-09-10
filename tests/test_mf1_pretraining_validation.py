"""Native CE validation must preserve complete targets, batching and exact recovery."""

import json
from dataclasses import asdict

import pytest
import torch
from test_mf1_canonical_text import encoded as encoded
from test_mf1_workflow import assert_tree_equal
from test_mf1_workflow import fixture_data as fixture_data

from minifrontier.data.minifrontier1 import RecordDataset
from minifrontier.data.minifrontier1_encoding import CompactDataset, encode_dataset
from minifrontier.models.minifrontier1 import MiniFrontier1Config, MiniFrontier1ForCausalLM
from minifrontier.training import minifrontier1 as runtime
from minifrontier.training import validation
from minifrontier.training.metrics import mf1_scalars
from minifrontier.training.validation import NativeCEValidation


def policy(total):
    return dict(
        validation.PRETRAINING_VALIDATION,
        periodic_ce_tokens=max(1, total // 4),
        phase_end_ce_tokens=total,
        early_interval_ce=1,
        early_until_ce=100_000,
    )


def test_native_selection_reads_masks_without_decoding_and_covers_targets(encoded, monkeypatch):
    output, config, manifest = encoded
    data = CompactDataset(output, "val", config)
    total = manifest["splits"]["val"]["counts"]["ce_tokens"]
    with monkeypatch.context() as patch:

        def no_decoding(*args, **kwargs):
            raise AssertionError("selection must not decode documents or pixels")

        patch.setattr(data, "_read_item", no_decoding)
        selection = NativeCEValidation(data, max_length=31, seed=11, policy=policy(total))
    assert sum(selection.counts) == total
    assert len(selection) > len(data)
    assert len(set(selection.windows)) == len(selection)
    assert set(selection.selections["periodic"]) < set(selection.selections["phase_end"])
    for i in range(len(selection)):
        item = selection[i]
        assert item["input_ids"].shape[1] <= 31
        assert int(item["labels"][:, 1:].ne(-100).sum()) == selection.counts[i]
    assert torch.equal(
        torch.cat([selection[i]["labels"][0, 1:] for i in range(len(selection))]),
        data[0]["labels"][0, 1:],
    )


def test_native_batch_evaluation_matches_individual_complete_media_answers(fixture_data, tmp_path):
    config = MiniFrontier1Config.tiny()
    encoded = tmp_path / "compact"
    encode_dataset(fixture_data, encoded, config, compact=True, shard_tokens=512)
    data = CompactDataset(encoded, "val", config)
    rows = [data[i] for i in range(len(data))]
    # Exercise a same-domain microbatch containing both native media and plain text.
    assert any(row["media"] for row in rows) and any(not row["media"] for row in rows)
    for row in rows:
        row["domain"] = "mixed"
    total = sum(int(row["labels"][:, 1:].ne(-100).sum()) for row in rows)
    compact_selection = NativeCEValidation(
        data, max_length=config.max_position_embeddings, seed=7, policy=policy(total)
    )
    assert compact_selection.counts == [int(row["labels"][:, 1:].ne(-100).sum()) for row in rows]
    selection = NativeCEValidation(
        rows, max_length=config.max_position_embeddings, seed=7, policy=policy(total)
    )
    model = MiniFrontier1ForCausalLM(config, "dense_pretrain").train()
    rng = torch.get_rng_state()
    a, b = [
        runtime.evaluate(
            model,
            rows,
            torch.device("cpu"),
            selection=selection,
            final=True,
            max_length=config.max_position_embeddings,
            batch_size=batch,
        )
        for batch in (1, 4)
    ]
    assert model.training and torch.equal(rng, torch.get_rng_state())
    assert a["ce_tokens"] == b["ce_tokens"] == total
    assert a["domain_ce"] == b["domain_ce"] == {"mixed": total}
    assert (
        a["black_media_ce"]
        == b["black_media_ce"]
        == {"mixed": sum(int(row["labels"][:, 1:].ne(-100).sum()) for row in rows if row["media"])}
    )
    assert a["nll"] == pytest.approx(b["nll"], rel=1e-6)
    assert a["black_media_nll"]["mixed"] == pytest.approx(b["black_media_nll"]["mixed"], rel=1e-6)
    assert b["micro_batches"] < a["micro_batches"]
    assert not b["capability_qualified"]
    with pytest.raises(ValueError, match="complete validation"):
        NativeCEValidation(rows, max_length=2, seed=7, policy=policy(total))
    selection.counts[selection.selections["phase_end"][0]] += 1
    with pytest.raises(ValueError, match="stored loss mask"):
        runtime.evaluate(
            model,
            rows,
            torch.device("cpu"),
            selection=selection,
            final=True,
            max_length=config.max_position_embeddings,
            batch_size=4,
        )
    assert model.training and torch.equal(rng, torch.get_rng_state())


def test_native_fixed_validation_cadence_and_resume(fixture_data, tmp_path, monkeypatch):
    config = MiniFrontier1Config.tiny()
    data = RecordDataset(fixture_data, "val", config)
    total = sum(int(data[i]["labels"][:, 1:].ne(-100).sum()) for i in range(len(data)))
    monkeypatch.setattr(validation, "PRETRAINING_VALIDATION", policy(total))
    args = dict(
        data=fixture_data,
        phase="p0",
        config=asdict(config),
        steps=2,
        input_batch_tokens=32,
        batch_size=4,
        save_every=2,
        eval_every=100,
        pretraining_eval=True,
    )
    reference, resumed = tmp_path / "reference", tmp_path / "resumed"
    runtime.train(**args, output=reference)
    runtime.train(**args, output=resumed, stop_after_updates=1)
    paused = json.loads((resumed / "evaluation.json").read_text())
    assert paused["metrics"]["evaluation_scope"] == "periodic"
    pause_events = [
        json.loads(line) for line in (resumed / "metrics.jsonl").read_text().splitlines()
    ]
    assert (
        sum(event.get("event") == "validation" and event["step"] == 1 for event in pause_events)
        == 1
    )
    runtime.train(**args, output=resumed, resume=resumed / "checkpoint.pt")
    a, b = [torch.load(root / "checkpoint.pt", weights_only=True) for root in (reference, resumed)]
    for key in ("model", "optimizer", "sampler", "router_balance", "ledger", "rng"):
        assert_tree_equal(a[key], b[key])
    assert a["run_spec"]["validation"] == b["run_spec"]["validation"]
    events = [json.loads(line) for line in (reference / "metrics.jsonl").read_text().splitlines()]
    evals = [event for event in events if event.get("event") == "validation"]
    assert [(event["step"], event["evaluation_scope"]) for event in evals] == [
        (0, "periodic"),
        (1, "periodic"),
        (2, "phase_end"),
    ]
    assert evals[-1]["ce_tokens"] == total
    assert evals[-1]["main_ce_tokens"] == a["ledger"]["main_ce_tokens"]
    assert "eval/lm_loss" in mf1_scalars(evals[0])
    assert "eval/phase_end_lm_loss" in mf1_scalars(evals[-1])
    assert "eval/phase_end_ce_tokens" in mf1_scalars(evals[-1])
    monkeypatch.setattr(
        validation, "PRETRAINING_VALIDATION", dict(policy(total), early_interval_ce=2)
    )
    with pytest.raises(ValueError, match="exact resume"):
        runtime.train(**args, output=resumed, resume=resumed / "checkpoint.pt")


def test_formal_main_automatically_requires_phase_end_inventory(
    fixture_data, tmp_path, monkeypatch
):
    monkeypatch.setattr(runtime, "validate_gate", lambda *args, **kwargs: {})
    monkeypatch.setitem(
        runtime.PHASES["p0"], "lengths", {MiniFrontier1Config.tiny().max_position_embeddings: 1.0}
    )
    evidence = tmp_path / "evidence.json"
    evidence.write_text("{}")
    # Bypass only the separately tested admission gate to reach the actual inventory check.
    with pytest.raises(ValueError, match="inventory"):
        runtime.train(
            data=fixture_data,
            output=tmp_path / "formal",
            phase="p0",
            config=asdict(MiniFrontier1Config.tiny()),
            run_kind="strategy",
            evidence=evidence,
        )
    assert not (tmp_path / "formal/checkpoint.pt").exists()
    with pytest.raises(ValueError, match="token budget"):
        runtime.train(
            data=fixture_data,
            output=tmp_path / "formal",
            phase="p0",
            steps=1,
            token_budget=runtime.PHASES["p0"]["budget"],
            run_kind="strategy",
        )


def test_sparse_validation_does_not_inherit_last_dense_replay_or_mutate_training(monkeypatch):
    config = MiniFrontier1Config.tiny()
    model = MiniFrontier1ForCausalLM(config, "sparse_cpt").train()
    attentions = [layer.attention for layer in model.layers if layer.kind != "kda"]
    for attention in attentions:
        attention.training_phase = "dense_distill"
    ids = torch.tensor([[1, *range(30, 48), 2]])
    labels = ids.clone()
    labels[:, 0] = -100
    rows = [dict(input_ids=ids, labels=labels, domain="text")]
    selection = NativeCEValidation(rows, max_length=32, seed=7, policy=policy(19))
    forward = runtime._forward
    gradient_flags = [p.requires_grad for p in model.parameters()]

    def sparse_forward(model, item):
        assert all(a.training_phase == "sparse_cpt" for a in attentions)
        return forward(model, item)

    monkeypatch.setattr(runtime, "_forward", sparse_forward)
    result = runtime.evaluate(
        model, rows, torch.device("cpu"), selection=selection, final=True, max_length=32
    )
    assert result["attention_phase"] == "sparse_cpt"
    assert all(a.training_phase == "dense_distill" for a in attentions)
    assert gradient_flags == [p.requires_grad for p in model.parameters()]
    assert model.training

    def invalid_forward(model, item):
        result = sparse_forward(model, item)
        result.lm_loss = torch.tensor(float("nan"))
        return result

    monkeypatch.setattr(runtime, "_forward", invalid_forward)
    with pytest.raises(ValueError, match="nonfinite"):
        runtime.evaluate(
            model, rows, torch.device("cpu"), selection=selection, final=True, max_length=32
        )
    assert all(a.training_phase == "dense_distill" for a in attentions) and model.training
