"""Compare phase inheritance against uninterrupted training and frozen teachers."""

import json

import pytest
import torch
from test_mf1_workflow import assert_tree_equal
from test_train_integration import arguments, config_for
from test_train_integration import corpus as corpus

from minifrontier.data import sha256
from minifrontier.training import train
from minifrontier.training.pretraining import (
    PretrainingProgram,
    schedule_factor,
    validate_parent_artifact,
)


def program_file(root, name, config, *, with_indexer):
    mtp = json.loads(config.read_text())["mtp_loss_coef"]
    phases = []
    stages = (
        ["pretrain", "dense_distill", "sparse_cpt"] if with_indexer else ["pretrain", "pretrain"]
    )
    if name == "minideepseekv4" and with_indexer:
        stages.append("sparse_cpt")
    main_budget = sum(stage != "dense_distill" for stage in stages) * 254
    for number, stage in enumerate(stages):
        indexer = stage == "dense_distill"
        phases.append(
            dict(
                id=f"phase-{number}",
                stage=stage,
                unit="input_tokens" if indexer else "ce_tokens",
                budget=256 if indexer else 254,
                mtp_loss_coef=mtp,
                sequence_lengths=[128],
                peak_lr=dict(
                    muon=3e-4 if name == "minideepseekv4" else 0.01, adam=3e-4, indexer=3e-4
                ),
                schedule=dict(
                    axis="phase_input" if indexer else "main_ce",
                    start=0,
                    warmup_tokens=50,
                    decay_start=50,
                    end=256 if indexer else main_budget,
                    final_factor=0.1,
                ),
            )
        )
    if name == "minideepseekv4" and with_indexer:
        phases[-1].update(mtp_loss_coef=0.1, init_transition="mtp-weight")
    recipe = dict(
        status="test_only_unadmitted",
        seed=42,
        global_input_tokens=128,
        optimizer={
            "minikimik3": "kimi_muon",
            "miniqwen4": "qwen_muon",
            "minideepseekv4": "deepseek_muon",
        }[name],
        weight_decay=0.1,
        adam_eps=1e-8,
        clip_grad=1.0,
        router_bias_rate=0.001,
        phases=phases,
    )
    path = root / "program.json"
    path.write_text(
        json.dumps(
            dict(
                execution_program=dict(
                    id="cpu-test", models={name: dict(main_ce=main_budget, training_recipe=recipe)}
                )
            )
        )
    )
    return path, phases


def phase_args(name, config, data, output, program, phase, parent=None):
    args = [
        *arguments(name, config, data, output, stage=phase["stage"], steps=8),
        "--schedule",
        "program",
        "--pretraining-program",
        str(program),
        "--pretraining-phase",
        phase["id"],
        "--input-batch-tokens",
        "128",
        "--input-tokens" if phase["unit"] == "input_tokens" else "--ce-tokens",
        str(phase["budget"]),
        "--muon-lr",
        str(phase["peak_lr"]["muon"]),
    ]
    return (
        args
        + (["--init", str(parent)] if parent else [])
        + (["--init-transition", phase["init_transition"]] if "init_transition" in phase else [])
    )


def load(output):
    return torch.load(output / "checkpoint.pt", weights_only=True)


def named_states(saved):
    result = {k: v["state"] for k, v in saved["pretraining_state"]["dormant_optimizer"].items()}
    for group, description in zip(
        saved["optimizer"]["param_groups"],
        saved["pretraining_state"]["optimizer_groups"],
        strict=True,
    ):
        for index, name in zip(group["params"], description["names"], strict=True):
            if index in saved["optimizer"]["state"]:
                result[name] = saved["optimizer"]["state"][index]
    return result


def test_dense_phase_change_matches_uninterrupted_real_optimizer_and_sample_stream(
    corpus, tmp_path
):
    name = "minikimik3"
    config = config_for(name, tmp_path)
    program, phases = program_file(tmp_path, name, config, with_indexer=False)
    first, second, reference = [tmp_path / p for p in ("first", "second", "reference")]
    train.main(phase_args(name, config, corpus, first, program, phases[0]))
    train.main(
        phase_args(name, config, corpus, second, program, phases[1], first / "checkpoint.pt")
    )
    train.main(
        [
            *arguments(name, config, corpus, reference, steps=8),
            "--input-batch-tokens",
            "128",
            "--ce-tokens",
            "508",
            "--warmup-tokens",
            "50",
        ]
    )
    actual, expected = load(second), load(reference)
    for key in ("model", "optimizer", "router_balance", "qk_clip", "rng"):
        assert_tree_equal(actual[key], expected[key])
    assert actual["data_offset"] == expected["data_offset"]
    assert (
        actual["pretraining_state"]["main_ce_tokens"]
        == expected["token_ledger"]["ce_tokens"]
        == 508
    )
    assert actual["pretraining_state"]["transition"]["sampler"] == "inherit"


@pytest.mark.parametrize("name", ["miniqwen4", "minideepseekv4"])
def test_main_moments_pause_across_indexer_and_resume_exactly(name, corpus, tmp_path):
    config = config_for(name, tmp_path)
    if name == "minideepseekv4":
        values = json.loads(config.read_text())
        values.update(mtp_enabled=True, mtp_loss_coef=0.3)
        config.write_text(json.dumps(values))
    program, phases = program_file(tmp_path, name, config, with_indexer=True)
    first, indexer, resumed, sparse = [
        tmp_path / p for p in ("first", "indexer", "resumed", "sparse")
    ]
    train.main(phase_args(name, config, corpus, first, program, phases[0]))
    before = load(first)
    for output in (indexer, resumed):
        args = phase_args(name, config, corpus, output, program, phases[1], first / "checkpoint.pt")
        train.main(args + (["--stop-after-updates", "1"] if output == resumed else []))
    args = phase_args(name, config, corpus, resumed, program, phases[1])
    train.main([*args, "--resume", str(resumed / "checkpoint.pt")])
    teacher, recovered = load(indexer), load(resumed)
    for key in ("model", "optimizer", "token_ledger", "pretraining_state", "router_balance", "rng"):
        assert_tree_equal(teacher[key], recovered[key])
    assert teacher["pretraining_state"]["main_ce_tokens"] == before["token_ledger"]["ce_tokens"]
    for key, value in before["model"].items():
        if ".indexer." not in key:
            assert_tree_equal(value, teacher["model"][key])
    previous_moments, paused_moments = named_states(before), named_states(teacher)
    for key, value in previous_moments.items():
        assert_tree_equal(value, paused_moments[key])
    assert any(".indexer." in key for key in paused_moments)
    train.main(
        phase_args(name, config, corpus, sparse, program, phases[2], indexer / "checkpoint.pt")
    )
    after = load(sparse)
    assert after["pretraining_state"]["main_ce_tokens"] == 508
    head = "lm_head.weight" if name == "miniqwen4" else "head.weight"
    assert named_states(after)[head]["step"] == previous_moments[head]["step"] + 2
    assert after["pretraining_state"]["dormant_optimizer"] == {}
    if name == "minideepseekv4":
        final_config = tmp_path / "mtp-final.json"
        final_config.write_text(json.dumps(dict(values, mtp_loss_coef=0.1)))
        final = tmp_path / "final"
        train.main(
            phase_args(
                name, final_config, corpus, final, program, phases[3], sparse / "checkpoint.pt"
            )
        )
        finished = load(final)
        assert finished["config"]["mtp_loss_coef"] == 0.1
        assert finished["pretraining_state"]["main_ce_tokens"] == 762
        assert named_states(finished)[head]["step"] == named_states(after)[head]["step"] + 2


def test_program_refuses_legacy_initialization_and_unfinished_predecessor(corpus, tmp_path):
    name = "minikimik3"
    config = config_for(name, tmp_path)
    program, phases = program_file(tmp_path, name, config, with_indexer=False)
    first = tmp_path / "first"
    train.main(
        [*phase_args(name, config, corpus, first, program, phases[0]), "--stop-after-updates", "1"]
    )
    with pytest.raises(ValueError, match="incomplete"):
        train.main(
            phase_args(
                name,
                config,
                corpus,
                tmp_path / "bad-parent",
                program,
                phases[1],
                first / "checkpoint.pt",
            )
        )
    with pytest.raises(ValueError, match="random initialization"):
        train.main(
            phase_args(
                name,
                config,
                corpus,
                tmp_path / "bad-init",
                program,
                phases[0],
                first / "checkpoint.pt",
            )
        )


def test_main_ce_schedule_does_not_count_indexer_inputs():
    schedule = dict(
        start=0,
        warmup_tokens=20_000_000,
        decay_start=20_000_000,
        end=2_000_000_000,
        final_factor=0.1,
    )
    assert schedule_factor(schedule, 10_000_000) == 0.5
    assert schedule_factor(schedule, 20_000_000) == 1
    assert schedule_factor(schedule, 2_000_000_000) == 0.1
    assert schedule_factor(schedule, 2_100_000_000) == 0.1
    plateau = dict(schedule, decay_start=2_000_000_000)
    assert schedule_factor(plateau, 2_000_000_000) == 1


def test_phase_quality_cannot_be_borrowed_from_another_checkpoint(tmp_path):
    measured, other, evidence = [tmp_path / p for p in ("measured.pt", "other.pt", "evidence.json")]
    measured.write_bytes(b"measured parent")
    other.write_bytes(b"different parent")
    evidence.write_text(
        json.dumps(
            dict(
                completed_phases={
                    "Q2": dict(quality_passed=True, checkpoint_sha256=sha256(measured))
                }
            )
        )
    )
    with pytest.raises(ValueError, match="actual initialization"):
        validate_parent_artifact(evidence, "Q2", other)
    validate_parent_artifact(evidence, "Q2", measured)
    with pytest.raises(ValueError, match="actual initialization"):
        validate_parent_artifact(evidence, "Q1", measured)


def test_formal_recipe_requires_actual_bindings_and_matching_profile(corpus, tmp_path, monkeypatch):
    from minifrontier.training import strategy_gate

    name = "minikimik3"
    config = config_for(name, tmp_path)
    path, phases = program_file(tmp_path, name, config, with_indexer=False)
    args = train.parser().parse_args(
        phase_args(name, config, corpus, tmp_path / "output", path, phases[0])
    )
    args.run_kind = "strategy"
    document = json.loads(path.read_text())
    recipe = document["execution_program"]["models"][name]["training_recipe"]
    recipe["status"] = "frozen"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="bindings differ"):
        PretrainingProgram(args)
    recipe["bindings"] = {
        phases[0]["id"]: dict(
            data_sha256=sha256(corpus / "manifest.json"),
            config_sha256=sha256(config),
            tokenizer_sha256=sha256(corpus / "tokenizer.json"),
            batch_size=1,
            sequence_length=128,
        )
    }
    path.write_text(json.dumps(document))
    binding = PretrainingProgram(args).binding
    plan, evidence, profile = [tmp_path / p for p in ("plan.json", "evidence.json", "profile.json")]
    plan.write_text(json.dumps(dict(gpu_ids=[0])))
    evidence.write_text(json.dumps(dict(performance=str(profile))))
    args.strategy_plan, args.strategy_phase, args.strategy_evidence = (
        str(plan),
        phases[0]["id"],
        str(evidence),
    )
    phase = dict(
        id=phases[0]["id"],
        budget_min=254,
        budget_max=254,
        objective="ce_tokens",
        budget_scope="main",
        sequence_lengths=[128],
        image_occurrences=0,
        video_examples=0,
    )
    monkeypatch.setattr(
        strategy_gate, "check", lambda *a, **k: dict(model=name, phase=phase, errors=[])
    )
    keys = (
        "sequence_length",
        "batch_size",
        "grad_accum",
        "input_batch_tokens",
        "input_batch_policy",
        "input_batch_schedule",
        "visual_warmup",
        "vision_lr",
        "projector_lr",
        "token_mixture",
        "media_mixture",
    )
    measured = {k: getattr(args, k) for k in keys}
    measured.update(world_size=1, pretraining_program=dict(binding, recipe_sha256="other recipe"))
    profile.write_text(json.dumps(dict(recipe=measured)))
    with pytest.raises(ValueError, match="frozen pretraining program"):
        strategy_gate.validate_arguments(args)
    measured["pretraining_program"] = binding
    profile.write_text(json.dumps(dict(recipe=measured)))
    strategy_gate.validate_arguments(args)
