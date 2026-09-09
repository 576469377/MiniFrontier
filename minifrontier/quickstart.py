"""Offline, tiny end-to-end example for the v0.1.0 research preview."""

import argparse
import json
import platform
import sqlite3
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

from minifrontier.data import sha256
from minifrontier.data_v2 import encode_corpus, train_tokenizer
from minifrontier.inference import load_checkpoint, respond
from minifrontier.storage import GIB, require_space
from minifrontier.training import train

MODELS = ("minideepseekv4", "minikimik3", "miniqwen4")


def tiny_config(name, vocab):
    if name == "minideepseekv4":
        from minifrontier.models.minideepseekv4 import MiniDeepSeekV4Config

        return MiniDeepSeekV4Config(
            vocab_size=vocab,
            dim=32,
            n_layers=3,
            n_heads=2,
            head_dim=16,
            rope_head_dim=8,
            q_lora_rank=16,
            o_groups=1,
            o_lora_rank=16,
            moe_inter_dim=16,
            n_routed_experts=4,
            n_activated_experts=2,
            n_hash_layers=1,
            compress_ratios=(0, 4, 128),
            index_n_heads=2,
            index_head_dim=16,
            window_size=128,
            max_seq_len=256,
        )
    if name == "minikimik3":
        from minifrontier.models.minikimik3 import MiniKimiK3Config

        return MiniKimiK3Config(
            vocab_size=vocab,
            hidden_size=32,
            num_hidden_layers=4,
            num_attention_heads=2,
            num_key_value_heads=2,
            intermediate_size=64,
            q_lora_rank=16,
            kv_lora_rank=16,
            qk_nope_head_dim=16,
            qk_rope_head_dim=8,
            v_head_dim=16,
            kda_head_dim=16,
            num_experts=4,
            num_experts_per_token=2,
            moe_intermediate_size=16,
            routed_expert_hidden_size=16,
            max_position_embeddings=256,
        )
    from minifrontier.models.miniqwen4 import MiniQwen4Config

    return MiniQwen4Config(
        vocab_size=vocab,
        hidden_size=16,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        hc_lowrank=4,
        ple_embed_dim=16,
        ngram_vocab_size_base=16,
        indexer_n_heads=2,
        indexer_head_dim=4,
        indexer_budget=4,
        indexer_compress_ratio=2,
        partial_rotary_factor=0.5,
        gradient_checkpointing=False,
        max_position_embeddings=256,
    )


def prepare(root):
    """Explicit operand-group split; no network, natural text or model downloads."""
    root.mkdir(parents=True)
    db = sqlite3.connect(root / "corpus.sqlite")
    db.execute(
        "CREATE TABLE samples (id TEXT PRIMARY KEY, stage TEXT, split TEXT, text TEXT, payload TEXT)"
    )
    counts = {split: 0 for split in ("train", "val", "test")}
    for a in range(14):
        split = "train" if a < 10 else "val" if a < 12 else "test"
        for b in range(8):
            question, answer = f"What is {a} + {b}?", str(a + b)
            for stage in ("pretrain", "sft"):
                text = f"Question: {question}\nAnswer: {answer}."
                row = dict(
                    source="minifrontier-offline-quickstart",
                    revision="1",
                    item_id=f"{a}:{b}:{stage}",
                    group_id=f"left-operand:{a}",
                    license="Apache-2.0",
                    lang="en",
                    task="addition",
                    stage=stage,
                    text=text,
                    turns=[
                        dict(role="user", content=question),
                        dict(role="assistant", content=answer),
                    ],
                    verifier=dict(kind="integer-addition", operands=[a, b], answer=a + b),
                )
                db.execute(
                    "INSERT INTO samples VALUES (?,?,?,?,?)",
                    (row["item_id"], stage, split, text, json.dumps(row)),
                )
                counts[split] += 1
    db.commit()
    db.close()
    (root / "corpus-manifest.json").write_text(
        json.dumps(
            dict(
                schema_version=2,
                counts=counts,
                main_budget_eligible=False,
                purpose="offline plumbing example, not a generalization benchmark",
                split_rule="left operand 0-9 train, 10-11 val, 12-13 sealed test; both stages share groups",
                database_sha256=sha256(root / "corpus.sqlite"),
            ),
            indent=2,
        )
    )
    tokenizer = root / "tokenizer.json"
    info = train_tokenizer(root, tokenizer, 512)
    encode_corpus(root, tokenizer, root / "encoded", max_length=64)
    return info


def run_example(root, name, device, updates=8):
    require_space(root.parent, 256 * 1024**2, reserve_bytes=GIB)
    if root.exists():
        raise FileExistsError("quickstart output exists; use a new directory")
    root.mkdir(parents=True)
    started = time.monotonic()
    tokenizer_info = prepare(root / "data")
    config = tiny_config(name, tokenizer_info["actual_vocab"])
    config_path = root / "config.json"
    config_path.write_text(json.dumps(asdict(config), indent=2))
    commands = []

    def execute(arguments):
        commands.append([sys.executable, "-m", "minifrontier", "train", *arguments])
        train.main(arguments)

    common = [
        "--model",
        name,
        "--config",
        str(config_path),
        "--data",
        str(root / "data/encoded"),
        "--steps",
        str(updates),
        "--sequence-length",
        "64",
        "--batch-size",
        "2",
        "--grad-accum",
        "1",
        "--optimizer",
        "adamw",
        "--lr",
        "0.001",
        "--warmup-steps",
        "1",
        "--seed",
        "42",
        "--save-every",
        "2",
        "--eval-every",
        "2",
        "--eval-batches",
        "0",
        "--log-every",
        "1",
        "--device",
        device,
        "--no-tensorboard",
        "--run-kind",
        "acceptance",
    ]
    pretrain = [*common, "--stage", "pretrain", "--output", str(root / "pretrain")]
    execute([*pretrain, "--stop-after-updates", str(updates // 2)])
    paused = json.loads((root / "pretrain/status.json").read_text())
    if paused["state"] != "paused":
        raise RuntimeError("example did not pause at a resumable boundary")
    execute([*pretrain, "--resume", str(root / "pretrain/checkpoint.pt")])
    execute(
        [
            *common,
            "--stage",
            "sft",
            "--output",
            str(root / "sft"),
            "--init",
            str(root / "pretrain/model.pt"),
        ]
    )
    model, tokenizer, _ = load_checkpoint(root / "sft/model.pt", device=device)
    prompt = "What is 10 + 2?"
    generation = respond(model, tokenizer, prompt, max_new_tokens=12, temperature=0.0)
    evaluations = []
    for line in (root / "sft/metrics.jsonl").read_text().splitlines():
        entry = json.loads(line)
        if entry["event"] == "validation":
            evaluations.append(entry)
    report = dict(
        schema_version=1,
        model=name,
        device=device,
        torch_version=str(torch.__version__),
        hardware=torch.cuda.get_device_name(device)
        if device.startswith("cuda")
        else platform.machine(),
        cpu_threads=torch.get_num_threads(),
        elapsed_seconds=time.monotonic() - started,
        parameters=sum(p.numel() for p in model.parameters()),
        tokenizer=tokenizer_info,
        resume_executed=True,
        pause_step=paused["step"],
        updates_per_stage=updates,
        final_pretrain=json.loads((root / "pretrain/status.json").read_text()),
        final_sft=json.loads((root / "sft/status.json").read_text()),
        evaluations=evaluations,
        prompt=prompt,
        generation=generation,
        commands=commands,
        data_sha256=sha256(root / "data/encoded/manifest.json"),
        config_sha256=sha256(config_path),
        main_budget_eligible=False,
        capability_status="unassessed",
        expected_result="data generated; PT paused/resumed; SFT, validation and CLI generation execute; answer correctness is not a pass criterion",
    )
    (root / "report.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            dict(
                model=name,
                report=str(root / "report.json"),
                seconds=report["elapsed_seconds"],
                generation=generation,
            )
        )
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=[*MODELS, "all"], default="minideepseekv4")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--updates", type=int, default=8)
    args = parser.parse_args(argv)
    if args.updates < 4:
        parser.error("updates must be at least 4 to demonstrate pause and resume")
    torch.set_num_threads(2)
    for name in MODELS if args.model == "all" else [args.model]:
        run_example(args.output.resolve() / name, name, args.device, args.updates)


if __name__ == "__main__":
    main()
