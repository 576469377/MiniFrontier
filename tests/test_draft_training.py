"""Real objective updates, exact interrupted recovery, and target-bound exports."""

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
from tokenizers import Tokenizer, models, trainers

from minifrontier.data import sha256
from minifrontier.data_v2 import STRATEGY_SPECIAL_TOKENS
from minifrontier.models.factory import build_model
from minifrontier.training.drafts import load_draft
from minifrontier.training.train_draft import main


def prepare(root, family):
    root.mkdir()
    tokenizer = Tokenizer(models.BPE())
    tokenizer.train_from_iterator(
        ["draft corpus contains independent prompts and complete answers"],
        trainers.BpeTrainer(vocab_size=64, special_tokens=STRATEGY_SPECIAL_TOKENS),
    )
    tokenizer.save(str(root / "tokenizer.json"))
    manifest = dict(
        sequence_length=64,
        tokenizer=dict(
            vocab_size=tokenizer.get_vocab_size(), sha256=sha256(root / "tokenizer.json")
        ),
        stages={"sft": {}},
    )
    rng = np.random.default_rng(731)
    for split in ("train", "val"):
        values = np.zeros((3, 1, 2, 64), dtype=np.int32)
        values[:, :, 0, :20] = rng.integers(21, 60, (3, 1, 20))
        values[:, :, 0, 0] = 1
        values[:, :, 0, 19] = 2
        values[:, :, 1] = -100
        values[:, :, 1, 7:20] = values[:, :, 0, 7:20]
        path = root / f"sft.{split}.npy"
        np.save(path, values)
        manifest["stages"]["sft"][split] = dict(file=path.name, sha256=sha256(path))
    (root / "manifest.json").write_text(json.dumps(manifest))
    config = {
        "minikimik3": lambda: tiny_kimi(num_hidden_layers=12, mtp_enabled=True),
        "miniqwen4": lambda: tiny_config(
            vocab_size=64, mtp_enabled=True, max_position_embeddings=128
        ),
        "minideepseekv4": lambda: tiny_deepseek(mtp_enabled=True),
    }[family]()
    torch.manual_seed(419)
    model = build_model(family, asdict(config))
    target = root / "target.pt"
    torch.save(
        dict(
            model_name=family,
            model=model.state_dict(),
            config=asdict(config),
            phase="dense_pretrain",
            step=0,
            stage="sft",
            tokenizer_sha256=manifest["tokenizer"]["sha256"],
        ),
        target,
    )
    return target


@pytest.mark.parametrize("family", ["minikimik3", "miniqwen4", "minideepseekv4"])
def test_draft_resume_is_exact_and_export_rejects_another_target(tmp_path, family, monkeypatch):
    monkeypatch.setenv("MINIFRONTIER_MIN_FREE_GIB", "1")
    target = prepare(tmp_path / "data", family)
    before = sha256(target)
    options = [
        "--target",
        str(target),
        "--data",
        str(target.parent),
        "--draft-positions",
        "1000",
        "--sequence-length",
        "64",
        "--rollout-tokens",
        "6",
        "--grad-accum",
        "2",
        "--device",
        "cpu",
        "--save-every",
        "1",
        "--eval-every",
        "2",
        "--eval-examples",
        "2",
        "--profile-warmup",
        "0",
        "--profile-updates",
        "2",
    ]
    main([*options, "--output", str(tmp_path / "full"), "--steps", "2"])
    main([*options, "--output", str(tmp_path / "resumed"), "--steps", "1"])
    main([*options, "--output", str(tmp_path / "resumed"), "--steps", "2", "--resume"])
    full = torch.load(tmp_path / "full/checkpoint.pt", weights_only=True)
    resumed = torch.load(tmp_path / "resumed/checkpoint.pt", weights_only=True)
    assert full["draft_positions"] == resumed["draft_positions"] > 0
    assert full["data_offset"] == resumed["data_offset"]
    for name, expected in full["draft"].items():
        torch.testing.assert_close(resumed["draft"][name], expected, rtol=0, atol=0, msg=name)
    assert len(resumed["profile"]) == 2
    assert sha256(target) == before
    loaded, draft, _tokenizer, _meta = load_draft(tmp_path / "full/draft.pt", target)
    assert draft.target() is loaded and not loaded.training
    assert all(not p.requires_grad for p in loaded.parameters())
    assert not any(name.startswith("target.") for name in full["draft"])
    other = target.with_name("other.pt")
    other.write_bytes(target.read_bytes() + b"altered")
    with pytest.raises(ValueError, match="different frozen target"):
        load_draft(tmp_path / "full/draft.pt", other)


@pytest.mark.cuda
@pytest.mark.distributed
@pytest.mark.parametrize("family", ["minikimik3", "miniqwen4", "minideepseekv4"])
def test_two_gpu_draft_backward_and_rank_equality(tmp_path, family):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    target = prepare(tmp_path / "data", family)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            "-m",
            "minifrontier",
            "train-draft",
            "--target",
            str(target),
            "--data",
            str(target.parent),
            "--output",
            str(tmp_path / "ddp"),
            "--draft-positions",
            "1",
            "--sequence-length",
            "64",
            "--rollout-tokens",
            "6",
            "--grad-accum",
            "2",
            "--steps",
            "2",
            "--eval-examples",
            "2",
            "--device",
            "cuda",
        ],
        env=dict(os.environ, MINIFRONTIER_MIN_FREE_GIB="1", OMP_NUM_THREADS="2"),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    state = json.loads((tmp_path / "ddp/status.json").read_text())
    assert state["state"] == "complete" and state["draft_positions"] >= 1
    saved = torch.load(tmp_path / "ddp/checkpoint.pt", weights_only=True)
    assert len(saved["rng"]) == 2
