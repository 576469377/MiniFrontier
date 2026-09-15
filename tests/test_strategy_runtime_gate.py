import json

import pytest
from test_pretraining_program import phase_args, program_file
from test_train_integration import config_for
from test_train_integration import corpus as corpus

from minifrontier.data import sha256
from minifrontier.training import strategy_gate, train
from minifrontier.training.strategy_gate import validate_runtime


def test_inherited_attention_and_actual_device_count_must_match_profile(tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(dict(gpu_ids=[2, 3], phases=[dict(id="Q-SFT", attention_phase="sparse_cpt")]))
    )
    validate_runtime(plan, "Q-SFT", "sparse_cpt", 2)
    with pytest.raises(ValueError, match="attention phase"):
        validate_runtime(plan, "Q-SFT", "dense_pretrain", 2)
    with pytest.raises(ValueError, match="device count"):
        validate_runtime(plan, "Q-SFT", "sparse_cpt", 1)


def test_sparse_continuation_observation_still_validates_exact_formal_program(
    corpus, tmp_path, monkeypatch
):
    name = "minideepseekv4"
    config = config_for(name, tmp_path)
    program, phases = program_file(tmp_path, name, config, with_indexer=True)
    phase = phases[2]
    document = json.loads(program.read_text())
    document["execution_program"]["models"][name]["training_recipe"].update(
        status="frozen",
        bindings={
            phase["id"]: dict(
                data_sha256=sha256(corpus / "manifest.json"),
                config_sha256=sha256(config),
                tokenizer_sha256=sha256(corpus / "tokenizer.json"),
                batch_size=1,
                sequence_length=128,
            )
        },
    )
    program.write_text(json.dumps(document))
    args = train.parser().parse_args(
        [
            *phase_args(
                name, config, corpus, tmp_path / "output", program, phase, tmp_path / "parent.pt"
            ),
            "--run-kind",
            "strategy",
            "--strategy-plan",
            str(tmp_path / "plan.json"),
            "--strategy-evidence",
            str(tmp_path / "evidence.json"),
            "--strategy-phase",
            phase["id"],
            "--pretraining-eval",
        ]
    )
    monkeypatch.setattr(
        strategy_gate,
        "check",
        lambda *a, **kw: dict(
            model=name,
            phase=dict(
                id=phase["id"],
                budget_scope="main",
                objective="ce_tokens",
                budget_min=254,
                budget_max=254,
                sequence_lengths=[128],
                image_occurrences=0,
                video_examples=0,
            ),
            profile_during_formal_updates=True,
            initial_start_authorized=False,
            errors=[],
        ),
    )
    assert not strategy_gate.validate_arguments(args)["errors"]
    args.input_batch_tokens = 256
    with pytest.raises(ValueError, match="batch/optimizer/context"):
        strategy_gate.validate_arguments(args)
    args.input_batch_tokens = 128
    args.batch_size = 2
    with pytest.raises(ValueError, match="microbatch bindings"):
        strategy_gate.validate_arguments(args)
    args.batch_size = 1
    args.init = None
    with pytest.raises(ValueError, match="program and full parent checkpoint"):
        strategy_gate.validate_arguments(args)
