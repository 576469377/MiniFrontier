import base64
import copy
import json
import random
from dataclasses import asdict

import numpy as np
import pytest
import torch

from minifrontier.data import sha256
from minifrontier.data.minifrontier1 import (
    RecordDataset,
    encode_record,
    make_fixture,
    safe_text,
    validate_record,
)
from minifrontier.models.minifrontier1 import MiniFrontier1Config, MiniFrontier1ForCausalLM
from minifrontier.models.minifrontier1.draft import MF1Draft
from minifrontier.speculative import Proposal, SpeculativeSession, generate_speculative, probability
from minifrontier.training.minifrontier1 import train
from minifrontier.training.minifrontier1_posttrain import grpo_objective, opd_objective
from minifrontier.training.minifrontier1_strategy import bindings, validate_gate


@pytest.fixture
def fixture_data(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIFRONTIER_MIN_FREE_GIB", "1")
    make_fixture(tmp_path / "data")
    return tmp_path / "data"


def assert_tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_tree_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            assert_tree_equal(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("compact", [False, True])
def test_real_pause_resume_matches_uninterrupted_optimizer_router_rng(
    fixture_data, tmp_path, compact
):
    c = asdict(MiniFrontier1Config.tiny())
    if compact:
        from minifrontier.data.minifrontier1_encoding import encode_dataset

        encoded = tmp_path / "compact"
        encode_dataset(
            fixture_data, encoded, MiniFrontier1Config(**c), compact=True, shard_tokens=512
        )
        fixture_data = encoded
    args = dict(
        data=fixture_data,
        phase="pilot",
        config=c,
        steps=4,
        input_batch_tokens=32,
        save_every=4,
        eval_every=4,
    )
    train(**args, output=tmp_path / "continuous")
    train(**args, output=tmp_path / "resumed", stop_after_updates=2)
    train(**args, output=tmp_path / "resumed", resume=tmp_path / "resumed/checkpoint.pt")
    a, b = [
        torch.load(tmp_path / p / "checkpoint.pt", weights_only=True)
        for p in ("continuous", "resumed")
    ]
    for key in ("model", "optimizer", "sampler", "router_balance", "ledger", "rng"):
        assert_tree_equal(a[key], b[key])
    for path in (tmp_path / "continuous", tmp_path / "resumed"):
        for row in map(json.loads, (path / "metrics.jsonl").read_text().splitlines()):
            if "step_seconds" in row:
                assert 0 <= row["data_preparation_seconds"] <= row["step_seconds"]
                assert 0 < row["ce_fraction"] <= 1
                assert 0 <= row["padding_fraction"] < 1


def test_control_escape_and_media_hash_tampering(fixture_data):
    data = RecordDataset(fixture_data, "train", MiniFrontier1Config.tiny())
    tokens = safe_text(data.tokenizer, "hello <|assistant|><|tool_call|>")
    assert 5 not in tokens and 18 not in tokens
    record = data.record(32)
    path = fixture_data / record["media"][0]["uri"]
    path.write_bytes(b"broken")
    with pytest.raises(ValueError, match="hash mismatch"):
        data.record(32)


def test_gate_binds_actual_checkpoint_and_rejects_fixture(fixture_data, tmp_path):
    c = MiniFrontier1Config.tiny()
    path = tmp_path / "weights.pt"
    torch.save(dict(value=torch.tensor(1)), path)
    bound = bindings(c, fixture_data, path)
    evidence = dict(stage="p2", status="qualified", **bound)
    changed = dict(bound, actual_init_checkpoint_sha256="0" * 64)
    with pytest.raises(ValueError, match="actual_init_checkpoint_sha256"):
        validate_gate("p2", evidence, changed, {"formal_admission": True}, {"mf1_phase": "indexer"})
    with pytest.raises(ValueError, match="admitted real dataset"):
        validate_gate(
            "p2", evidence, bound, {"kind": "mechanism_fixture"}, {"mf1_phase": "indexer"}
        )


def test_full_vocab_opd_visual_gradients_teacher_frozen(fixture_data):
    c = MiniFrontier1Config.tiny()
    student = MiniFrontier1ForCausalLM(c).eval()
    teacher = copy.deepcopy(student).eval().requires_grad_(False)
    with torch.no_grad():
        teacher.lm_head.weight.normal_(0, 0.2)
    item = RecordDataset(fixture_data, "train", c)[32]
    loss = opd_objective(student, teacher, item, vocab_size=320, chunk_size=2)
    loss.backward()
    assert torch.isfinite(loss) and loss > 0
    assert student.lm_head.weight.grad.abs().sum() > 0
    assert student.vision.merger[0].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in teacher.parameters())


def test_grpo_real_nonzero_group_and_assistant_denominator(fixture_data):
    c = MiniFrontier1Config.tiny()
    student = MiniFrontier1ForCausalLM(c).eval()
    reference = copy.deepcopy(student).requires_grad_(False)
    dataset = RecordDataset(fixture_data, "train", c)
    # Controlled objective test, not a claim that a random policy solved a task.
    trajectories = [dict(dataset[i], reward=float(i == 0)) for i in range(4)]
    terms, report = grpo_objective(student, reference, trajectories, vocab_size=320)
    assert not report["zero_variance"]
    sum(terms).backward()
    assert student.lm_head.weight.grad.abs().sum() > 0
    assert report["response_positions"] == sum(
        int(t["labels"][:, 1:].ne(-100).sum()) for t in trajectories
    )


def test_speculative_reject_replays_all_fusion_states(monkeypatch):
    c = MiniFrontier1Config.tiny()
    model = MiniFrontier1ForCausalLM(c).eval()
    ids = torch.tensor([[30, 31, 32, 33, 34, 35, 36]])
    with torch.no_grad():
        session = SpeculativeSession(model, ids, vocab_size=320)
        worst = session.context.next_p.masked_fill(
            session.context.next_p.eq(0), float("inf")
        ).argmin(-1, keepdim=True)
        # A legal but low-probability proposal with u=1 must be rejected.
        monkeypatch.setattr(
            torch, "rand", lambda *args, **kwargs: torch.ones((), device=kwargs.get("device"))
        )
        q = torch.zeros_like(session.context.next_p).scatter(-1, worst, 1)
        emitted = session.advance(lambda context, maximum: Proposal(worst, q[:, None]), 1)
        assert session.stats["rejections"] == 1 and emitted.shape == (1, 1)
        expected = probability(model(session.context.ids).logits[:, -1], model, 320)
        torch.testing.assert_close(expected, session.context.next_p, atol=2e-6, rtol=2e-5)


def test_mtp_draft_target_frozen_self_fed_gradient_and_sampler(fixture_data):
    target = MiniFrontier1ForCausalLM(MiniFrontier1Config.tiny()).eval()
    draft = MF1Draft(target)
    ids = torch.tensor([[30, 31, 32, 33, 34]])
    loss, report = draft.unroll(ids, steps=2, vocab_size=320)
    loss.backward()
    assert report["positions"] > 0 and draft.mtp.hidden_proj.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in target.parameters())
    result, stats = generate_speculative(
        target, draft.eval(), ids, max_new_tokens=4, draft_steps=2, vocab_size=320
    )
    assert result.shape[1] > ids.shape[1] and stats["target_forwards"] > 1


def test_continuous_main_moments_survive_indexer_phase(fixture_data, tmp_path):
    c = asdict(MiniFrontier1Config.tiny())
    train(
        data=fixture_data,
        output=tmp_path / "p1",
        config=c,
        phase="p1",
        steps=1,
        input_batch_tokens=48,
        save_every=1,
        eval_every=1,
    )
    predecessor = tmp_path / "p1/checkpoint.pt"
    before = torch.load(predecessor, weights_only=True)
    train(
        data=fixture_data,
        output=tmp_path / "indexer",
        phase="indexer",
        steps=1,
        input_batch_tokens=48,
        init=predecessor,
        save_every=1,
        eval_every=1,
    )
    after = torch.load(tmp_path / "indexer/checkpoint.pt", weights_only=True)
    for name, state in before["continuous_optimizer_states"].items():
        if ".indexer." not in name:
            assert_tree_equal(state, after["continuous_optimizer_states"][name])
            assert torch.equal(before["model"][name], after["model"][name])
    assert after["ledger"]["ce_tokens"] == 0 and after["ledger"]["input_tokens"] > 0


@pytest.fixture
def local_checkpoint(fixture_data, tmp_path):
    train(
        data=fixture_data,
        output=tmp_path / "trained",
        config=asdict(MiniFrontier1Config.tiny()),
        phase="sft",
        steps=1,
        input_batch_tokens=32,
        eval_every=1,
        save_every=1,
    )
    return tmp_path / "trained/checkpoint.pt"


def test_export_roundtrip_and_int8_storage(local_checkpoint, tmp_path):
    from minifrontier.inference.minifrontier1_export import Int8Linear, export_checkpoint
    from minifrontier.inference.runtime import load_checkpoint

    model, _, _ = load_checkpoint(local_checkpoint)
    ids = torch.tensor([[30, 31, 32, 33, 34, 35]])
    with torch.no_grad():
        expected = model(ids).logits
        for quantized in (False, True):
            destination = tmp_path / ("int8" if quantized else "fp32")
            report = export_checkpoint(local_checkpoint, destination, int8=quantized)
            restored, _, _ = load_checkpoint(destination / "model.pt")
            actual = restored(ids).logits
            assert report["source_checkpoint_sha256"] == sha256(local_checkpoint)
            assert not report["capability_qualified"]
            for name, expected_hash in report["license_files"].items():
                assert sha256(destination / name) == expected_hash
            if quantized:
                assert any(
                    isinstance(m, Int8Linear) and m.qweight.dtype == torch.int8
                    for m in restored.modules()
                )
                assert (actual - expected).abs().max() < 0.02
            else:
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_encoded_binary_masks_positions_and_checksums(fixture_data, tmp_path):
    from minifrontier.data.minifrontier1_encoding import encode_dataset
    from minifrontier.models.minifrontier1.processing import token_metadata

    config = MiniFrontier1Config.tiny()
    destination = tmp_path / "encoded"
    manifest = encode_dataset(fixture_data, destination, config)
    dataset = RecordDataset(fixture_data, "train", config)
    entries = [
        json.loads(line) for line in (destination / "train.index.jsonl").read_text().splitlines()
    ]
    ids = np.fromfile(destination / "train.ids.bin", dtype=np.uint16)
    labels = np.fromfile(destination / "train.labels.bin", dtype=np.int32)
    positions = np.fromfile(destination / "train.positions.bin", dtype=np.int32).reshape(-1, 3)
    for i in (0, 32, 64):
        item, entry = dataset[i], entries[i]
        start, length = entry["offset"], entry["input_tokens"]
        metadata = token_metadata(item["input_ids"], config, item["media"])
        np.testing.assert_array_equal(ids[start : start + length], item["input_ids"][0])
        np.testing.assert_array_equal(labels[start : start + length], item["labels"][0])
        np.testing.assert_array_equal(
            positions[start : start + length], metadata["position_ids"][:, 0].T
        )
    assert not manifest["formal_admission"]
    for entry in manifest["splits"]["train"]["files"].values():
        assert sha256(destination / entry["name"]) == entry["sha256"]


def test_native_packing_equivalence_with_independent_media_budgets(fixture_data):
    from minifrontier.training.minifrontier1_curriculum import context_length, pack_records

    config = MiniFrontier1Config.tiny()
    config.protected_media_tokens = 8
    dataset = RecordDataset(fixture_data, "train", config)
    items = [dataset[32], dataset[64]]
    packed = pack_records(items, 256)
    model = MiniFrontier1ForCausalLM(config).eval()
    with torch.no_grad():
        result = model(
            packed["input_ids"], media=packed["media"], segment_ids=packed["segment_ids"]
        ).logits
        expected = torch.cat([model(i["input_ids"], media=i["media"]).logits for i in items], 1)
    torch.testing.assert_close(result, expected, atol=2e-6, rtol=2e-5)
    rng = random.Random(7)
    assert 8192 not in [context_length("p3", rng, 0, 8192) for _ in range(100)]
    assert 8192 in [context_length("p3", rng, 10_000_000, 8192) for _ in range(100)]


@pytest.mark.parametrize("phase", ["dense_pretrain", "sparse_cpt"])
def test_greedy_draft_matches_target_with_image_and_replay(fixture_data, phase):
    from minifrontier.inference.runtime import generate_ids

    torch.manual_seed(123)
    c = MiniFrontier1Config.tiny()
    model = MiniFrontier1ForCausalLM(c, training_phase=phase).eval()
    item = RecordDataset(fixture_data, "train", c)[32]
    draft = MF1Draft(model).eval()
    with torch.no_grad():
        expected = generate_ids(
            model,
            item["input_ids"],
            media=item["media"],
            max_new_tokens=8,
            temperature=0,
            vocab_size=320,
        )
        actual, stats = draft.generate_greedy(
            item["input_ids"], media=item["media"], max_new_tokens=8, steps=4, vocab_size=320
        )
    assert torch.equal(actual, expected)
    assert 0 <= stats["accepted"] <= stats["proposed"]


def test_rollout_keeps_actual_behavior_logp_and_tool_prompt(fixture_data):
    from minifrontier.training.minifrontier1_posttrain import response_logp, rollout

    model = MiniFrontier1ForCausalLM(MiniFrontier1Config.tiny()).eval()
    dataset = RecordDataset(fixture_data, "train", model.config)
    trajectory = rollout(model, dataset, dataset.record(32), max_tokens=4)
    with torch.no_grad():
        full_logp, mask = response_logp(model, trajectory, 320)
    torch.testing.assert_close(
        full_logp[mask], trajectory["behavior_logp"][mask], atol=2e-6, rtol=2e-5
    )
    record = dataset.record(0)
    record["tool_environment"] = {"tools": ["python"]}
    item = encode_record(
        record, dataset.tokenizer, model.config, dataset.media_root, generation_prompt=True
    )
    assert item["input_ids"][0, -1] == 5  # Policy selects final/thinking/tool channel itself.


def test_rl_window_accumulates_fresh_prompts_once(local_checkpoint, fixture_data, tmp_path):
    from minifrontier.training.minifrontier1_posttrain import train_post

    report = train_post(
        phase="rl",
        checkpoint=local_checkpoint,
        data=fixture_data,
        output=tmp_path / "rl",
        steps=1,
        max_tokens=2,
        prompts_per_update=2,
    )
    traces = [
        json.loads(line) for line in (tmp_path / "rl/rollouts.jsonl").read_text().splitlines()
    ]
    assert len(traces) == 8
    assert len({t["sample_id"] for t in traces}) == 2
    assert len({t["policy_version"] for t in traces}) == 1
    assert report["ledger"]["generated_tokens"] == sum(t["generated_tokens"] for t in traces)


def test_demo_preflight_preserves_real_video_timestamps(fixture_data):
    from minifrontier.inference.minifrontier1_demo import prepare_request

    c = MiniFrontier1Config.tiny()
    model = MiniFrontier1ForCausalLM(c)
    dataset = RecordDataset(fixture_data, "train", c)
    resource = dataset.record(64)["media"][0]
    payload = dict(
        prompt="Last color?",
        max_new_tokens=4,
        media=[
            dict(
                kind="video",
                timestamps=resource["timestamps"],
                frames=[
                    base64.b64encode((fixture_data / path).read_bytes()).decode()
                    for path in resource["frames"]
                ],
            )
        ],
    )
    ids, spans, report = prepare_request(payload, model, dataset.tokenizer)
    assert report["media"][0]["frames"] == 4
    assert spans[0]["temporal_positions"] == [0, 120]
    assert ids.eq(7).sum() == report["vision_tokens"]
    payload["max_new_tokens"] = 10000
    with pytest.raises(ValueError):
        prepare_request(payload, model, dataset.tokenizer)


def test_video_schema_cannot_silently_drop_temporal_information(fixture_data):
    dataset = RecordDataset(fixture_data, "train", MiniFrontier1Config.tiny())
    record = dataset.record(64)
    record["media"][0]["timestamps"][1] = float("nan")
    with pytest.raises(ValueError, match="source timestamps"):
        validate_record(record, fixture_data)
    record = dataset.record(32)
    record["messages"][0]["content"][0]["type"] = "video"
    with pytest.raises(ValueError, match="two hashed"):
        validate_record(record, fixture_data)


def test_document_views_preserve_source_boxes_and_one_exposure(fixture_data):
    from PIL import Image

    dataset = RecordDataset(fixture_data, "train", MiniFrontier1Config())
    record = dataset.record(32)
    resource = record["media"][0]
    path = fixture_data / "media/document.png"
    Image.new("RGB", (896, 896), "white").save(path)
    resource.update(
        uri="media/document.png",
        sha256=sha256(path),
        width=896,
        height=896,
        representation="document",
    )
    validate_record(record, fixture_data)
    item = encode_record(record, dataset.tokenizer, dataset.config, fixture_data)
    assert len(item["media"]) == 5 and item["media_exposures"] == 1
    assert sum(s["feature_count"] for s in item["media"]) == 833
    assert len({tuple(s["source_box"]) for s in item["media"]}) == 5
    assert item["labels"][item["input_ids"].eq(7)].eq(-100).all()


def test_fixture_content_groups_do_not_overlap_across_evaluation_splits(fixture_data):
    identities = {}
    for split in ("train", "val", "test", "demo"):
        dataset = RecordDataset(fixture_data, split, MiniFrontier1Config.tiny())
        for i in range(len(dataset)):
            record = dataset.record(i)
            identity = json.dumps(
                [record["messages"], [m["sha256"] for m in record["media"]]], sort_keys=True
            )
            assert identity not in identities, (split, identities.get(identity))
            identities[identity] = split


def test_qualified_registry_auto_opd_and_mismatched_slot_rejection(
    local_checkpoint, fixture_data, tmp_path
):
    from minifrontier.training.minifrontier1_posttrain import qualified_teacher, train_post

    # Synthetic qualification evidence exercises artifact binding, not model capability.
    base = torch.load(local_checkpoint, weights_only=True)
    evaluation = tmp_path / "unit-evaluation.json"
    evidence = dict(
        checkpoint_sha256=sha256(local_checkpoint),
        slot="math:direct",
        tokenizer_sha256=base["tokenizer_sha256"],
        metrics=dict(qualified=True),
    )
    evaluation.write_text(json.dumps(evidence))
    registry = tmp_path / "registry.json"
    entry = dict(
        checkpoint="trained/checkpoint.pt",
        checkpoint_sha256=sha256(local_checkpoint),
        status="qualified",
        evaluation=evaluation.name,
        evaluation_sha256=sha256(evaluation),
    )
    registry.write_text(json.dumps(dict(teachers={"math:direct": entry})))
    report = train_post(
        phase="opd",
        checkpoint=local_checkpoint,
        data=fixture_data,
        output=tmp_path / "opd",
        teacher_registry=registry,
        steps=1,
        max_tokens=2,
    )
    assert report["ledger"]["optimizer_updates"] == 1
    assert report["ledger"]["generated_tokens"] > 0
    evidence["slot"] = "vision:direct"
    evaluation.write_text(json.dumps(evidence))
    entry["evaluation_sha256"] = sha256(evaluation)
    registry.write_text(json.dumps(dict(teachers={"math:direct": entry})))
    with pytest.raises(ValueError, match="actual checkpoint"):
        qualified_teacher(registry, "math:direct", base)


def test_validation_covers_domains_after_old_twelve_record_prefix(monkeypatch):
    from types import SimpleNamespace

    import minifrontier.training.minifrontier1 as runtime

    data = [
        dict(labels=torch.tensor([[-100, 24, 2]]), domain="early" if i < 12 else "late")
        for i in range(18)
    ]
    monkeypatch.setattr(
        runtime,
        "_forward",
        lambda model, item: SimpleNamespace(
            lm_loss=torch.tensor(1.0 if item["domain"] == "early" else 5.0)
        ),
    )
    model = torch.nn.Linear(1, 1).train()
    result = runtime.evaluate(model, data, torch.device("cpu"))
    assert result["examples"] == 18 and result["ce_tokens"] == 36
    assert result["per_domain"] == {"early": 1.0, "late": 5.0}
    assert result["nll"] == pytest.approx(7 / 3)
    assert result["selection"] == "full_validation" and model.training
    assert (
        runtime.evaluate(model, data, torch.device("cpu"), limit=12)["selection"]
        == "explicit_prefix"
    )


def test_dense_sft_diagnosis_binds_attention_and_cannot_change_formal_plan(fixture_data, tmp_path):
    args = dict(
        data=fixture_data,
        config=asdict(MiniFrontier1Config.tiny()),
        phase="sft",
        steps=2,
        input_batch_tokens=32,
        output=tmp_path / "sft",
        diagnostic_attention="dense_pretrain",
    )
    train(**args, stop_after_updates=1)
    checkpoint = tmp_path / "sft/checkpoint.pt"
    saved = torch.load(checkpoint, weights_only=True)
    assert saved["phase"] == "dense_pretrain"
    assert saved["run_spec"]["diagnostic_attention"] == "dense_pretrain"
    with pytest.raises(ValueError, match="exact resume"):
        train(**dict(args, diagnostic_attention=None), resume=checkpoint)
    train(**args, resume=checkpoint)
    with pytest.raises(ValueError, match="restricted"):
        train(**dict(args, run_kind="strategy"))
