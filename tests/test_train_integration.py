"""Executable text/RL stages and recovery after an interrupted optimizer update."""

import json
import os
import subprocess
import sys
from dataclasses import asdict

import numpy as np
import pytest
import torch
from test_miniqwen4 import tiny_config
from test_new_backbones import tiny_deepseek, tiny_kimi
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from minifrontier.data import SPECIAL_TOKENS, sha256
from minifrontier.inference.runtime import load_checkpoint, respond
from minifrontier.training import train
from minifrontier.training.rollouts import prepare_tasks


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        ["Calculate 12 + 34. Reply with the integer answer only. Reasoning effort: low. high."],
        trainers.BpeTrainer(
            vocab_size=300,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    tokenizer.save(str(root / "tokenizer.json"))
    manifest = dict(
        sequence_length=128,
        tokenizer=dict(
            vocab_size=tokenizer.get_vocab_size(), sha256=sha256(root / "tokenizer.json")
        ),
        stages={},
    )
    rng = np.random.default_rng(18)
    for stage in ("pretrain", "sft", "dpo"):
        manifest["stages"][stage] = {}
        for split in ("train", "val"):
            if stage == "pretrain":
                path = root / f"{stage}.{split}.bin"
                rng.integers(3, 60, (2048,), dtype=np.int32).tofile(path)
            else:
                path = root / f"{stage}.{split}.npy"
                values = rng.integers(
                    3, 60, (8, 1 if stage == "sft" else 2, 2, 128), dtype=np.int32
                )
                values[:, :, 1, :] = values[:, :, 0, :]
                values[:, :, 1, :32] = -100
                np.save(path, values)
            manifest["stages"][stage][split] = dict(file=path.name, sha256=sha256(path))
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def config_for(name, root):
    config = {
        "miniqwen4": lambda: tiny_config(vocab_size=300, max_position_embeddings=256),
        "minikimik3": lambda: tiny_kimi(vocab_size=300),
        "minideepseekv4": lambda: tiny_deepseek(vocab_size=300),
    }[name]()
    path = root / f"{name}.json"
    path.write_text(json.dumps(asdict(config)))
    return path


def arguments(name, config, corpus, output, stage="pretrain", steps=2):
    return [
        "--model",
        name,
        "--config",
        str(config),
        "--data",
        str(corpus),
        "--output",
        str(output),
        "--stage",
        stage,
        "--steps",
        str(steps),
        "--batch-size",
        "1",
        "--grad-accum",
        "2",
        "--sequence-length",
        "128",
        "--save-every",
        "1",
        "--eval-every",
        "2",
        "--eval-batches",
        "1",
        "--warmup-steps",
        "1",
        "--device",
        "cpu",
        "--no-tensorboard",
        "--run-kind",
        "acceptance",
    ]


@pytest.mark.parametrize("name", ["miniqwen4", "minikimik3", "minideepseekv4"])
def test_all_stages_export_and_generate(name, corpus, tmp_path):
    config = config_for(name, tmp_path)
    previous = None
    stages = (
        ["pretrain"]
        + ([] if name == "minikimik3" else ["dense_distill", "sparse_cpt"])
        + ["sft", "dpo"]
    )
    for stage in stages:
        output = tmp_path / name / stage
        args = arguments(name, config, corpus, output, stage, steps=1)
        if previous:
            args += ["--init", str(previous)]
        train.main(args)
        previous = output / "model.pt"
        assert json.loads((output / "status.json").read_text())["state"] == "complete"
    model, tokenizer, meta = load_checkpoint(previous)
    text = respond(model, tokenizer, "Hello", max_new_tokens=2, temperature=0.0)
    assert isinstance(text, str) and meta["stage"] == "dpo"


def test_actual_input_budget_controls_accumulation(corpus, tmp_path):
    name = "minideepseekv4"
    config = config_for(name, tmp_path)
    output = tmp_path / "actual-input"
    train.main([*arguments(name, config, corpus, output, steps=1), "--input-batch-tokens", "300"])
    state = json.loads((output / "status.json").read_text())
    assert state["token_ledger"]["input_tokens"] == 384
    assert state["token_ledger"]["optimizer_updates"] == 1


def test_large_microbatch_ceiling_preserves_global_batch_and_ramp_resume(corpus, tmp_path):
    name = "minideepseekv4"
    config = config_for(name, tmp_path)
    outputs = [tmp_path / part for part in ("small", "large", "resumed")]
    for cap, output in zip((1, 128, 128), outputs, strict=True):
        args = [
            *arguments(name, config, corpus, output, steps=3),
            "--batch-size",
            str(cap),
            "--ce-tokens",
            "1500",
            "--warmup-tokens",
            "100",
            "--input-batch-tokens",
            "256",
            "--input-batch-schedule",
            "0:256,250:512,700:768",
            "--log-every",
            "1",
        ]
        if output == outputs[-1]:
            train.main([*args, "--stop-after-updates", "1"])
            train.main([*args, "--resume", str(output / "checkpoint.pt")])
        else:
            train.main(args)
        events = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
        assert [e["input_batch_actual"] for e in events if e["event"] == "train"] == [256, 512, 768]
    saved = [torch.load(p / "checkpoint.pt", weights_only=True) for p in outputs]
    assert saved[0]["token_ledger"] == saved[1]["token_ledger"] == saved[2]["token_ledger"]
    assert saved[0]["data_offset"] == saved[1]["data_offset"] == saved[2]["data_offset"]
    for key, value in saved[1]["model"].items():
        torch.testing.assert_close(value, saved[2]["model"][key], atol=0, rtol=0)


@pytest.mark.parametrize("graceful_pause", [False, True])
def test_training_resume_matches_uninterrupted_run(corpus, tmp_path, monkeypatch, graceful_pause):
    name = "minideepseekv4"
    config = config_for(name, tmp_path)
    complete = tmp_path / "complete"
    interrupted = tmp_path / "interrupted"
    train.main(arguments(name, config, corpus, complete))
    save = train.atomic_save

    def interrupt(value, path):
        save(value, path)
        if value.get("step") == 1:
            raise RuntimeError("simulated interruption after committed checkpoint")

    args = arguments(name, config, corpus, interrupted)
    if graceful_pause:
        train.main([*args, "--stop-after-updates", "1"])
        assert json.loads((interrupted / "status.json").read_text())["state"] == "paused"
        assert not (interrupted / "model.pt").exists()
    else:
        monkeypatch.setattr(train, "atomic_save", interrupt)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            train.main(args)
    monkeypatch.setattr(train, "atomic_save", save)
    train.main([*args, "--resume", str(interrupted / "checkpoint.pt")])
    expected = torch.load(complete / "checkpoint.pt", weights_only=True)
    actual = torch.load(interrupted / "checkpoint.pt", weights_only=True)
    assert actual["data_offset"] == expected["data_offset"]
    assert actual["token_ledger"] == expected["token_ledger"]
    for key, value in expected["model"].items():
        torch.testing.assert_close(actual["model"][key], value, atol=0, rtol=0)


@pytest.mark.parametrize("name", ["minikimik3", "minideepseekv4"])
@pytest.mark.distributed
def test_two_rank_trainer_checks_parameters(name, corpus, tmp_path):
    config = config_for(name, tmp_path)
    args = arguments(name, config, corpus, tmp_path / "ddp")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            "-m",
            "minifrontier.training.train",
            *args,
        ],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2"),
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"exact_parameters": true' in result.stdout


def test_grpo_and_multi_teacher_on_policy_stage(corpus, tmp_path):
    name = "minikimik3"
    config = config_for(name, tmp_path)
    previous = None
    for stage in ("pretrain", "sft", "dpo"):
        args = arguments(name, config, corpus, tmp_path / stage, stage, 1)
        if previous:
            args += ["--init", str(previous)]
        train.main(args)
        previous = tmp_path / stage / "model.pt"
    tasks = tmp_path / "tasks"
    prepare_tasks(tasks, count=100)
    common = ["--rl-data", str(tasks), "--group-size", "2", "--rollout-tokens", "2"]
    # An untrained policy with a two-token answer limit gets all-zero rewards.
    # This must preserve weights and report no optimizer updates, not apply decay.
    with pytest.raises(ValueError, match="no RL learning signal"):
        train.main(
            arguments(name, config, corpus, tmp_path / "grpo", "grpo", 1)
            + common
            + ["--init", str(previous)]
        )
    skipped = torch.load(tmp_path / "grpo/checkpoint.pt", weights_only=True)
    before = torch.load(previous, weights_only=True)
    assert skipped["step"] == skipped["token_ledger"]["optimizer_updates"] == 0
    assert skipped["token_ledger"]["skipped_windows"] == 32
    for key, value in before["model"].items():
        torch.testing.assert_close(skipped["model"][key], value, rtol=0, atol=0)
    teachers = tmp_path / "teachers.json"
    teachers.write_text(
        json.dumps(
            {
                "arithmetic:low": str(tmp_path / "sft/model.pt"),
                "arithmetic:high": str(tmp_path / "dpo/model.pt"),
            }
        )
    )
    train.main(
        arguments(name, config, corpus, tmp_path / "mopd", "mopd", 1)
        + common
        + ["--init", str(previous), "--teacher-map", str(teachers)]
    )
    assert (tmp_path / "mopd/model.pt").exists()


def test_deepseek_full_vocabulary_opd_stage_and_response_ledger(corpus, tmp_path):
    name = "minideepseekv4"
    config = config_for(name, tmp_path)
    previous = None
    for stage, folder in (("pretrain", "pt"), ("sft", "teacher1"), ("sft", "teacher2")):
        args = arguments(name, config, corpus, tmp_path / folder, stage, 1)
        if previous:
            args += ["--init", str(previous)]
        train.main(args)
        previous = tmp_path / folder / "model.pt"
    tasks = tmp_path / "tasks"
    prepare_tasks(tasks, count=100)
    registry = tmp_path / "teachers.json"
    registry.write_text(
        json.dumps(
            {
                "arithmetic:low": str(tmp_path / "teacher1/model.pt"),
                "arithmetic:high": str(tmp_path / "teacher2/model.pt"),
            }
        )
    )
    args = arguments(name, config, corpus, tmp_path / "opd", "opd", 10)
    train.main(
        [
            *args,
            "--init",
            str(previous),
            "--teacher-map",
            str(registry),
            "--rl-data",
            str(tasks),
            "--group-size",
            "2",
            "--rollout-tokens",
            "2",
            "--response-tokens",
            "4",
        ]
    )
    saved = torch.load(tmp_path / "opd/checkpoint.pt", weights_only=True)
    assert saved["token_ledger"]["response_tokens"] >= 4
    assert saved["token_ledger"]["ce_tokens"] == 0
    assert saved["step"] == 1
