"""Runnable MF1 entry points. Formal stages never inherit acceptance qualification."""

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import torch

from minifrontier.data.minifrontier1 import RecordDataset, make_fixture, write_json
from minifrontier.models.minifrontier1 import MiniFrontier1Config, MiniFrontier1ForCausalLM
from minifrontier.training.minifrontier1_optim import parameter_report
from minifrontier.training.minifrontier1_strategy import PHASES, budget_report


def quickstart(output, device="cpu", updates=8):
    from minifrontier.training.minifrontier1 import train

    root = Path(output)
    manifest = make_fixture(root / "data")
    c = MiniFrontier1Config.tiny(manifest["vocab_size"])
    write_json(root / "model.json", asdict(c))
    results = []
    args = dict(
        data=root / "data",
        output=root / "pilot",
        config=root / "model.json",
        device=device,
        steps=updates,
        input_batch_tokens=64,
        eval_every=updates,
        save_every=updates,
        phase="pilot",
    )
    results.append(train(**args, stop_after_updates=max(1, updates // 2)))
    results.append(train(**args, resume=root / "pilot/checkpoint.pt"))
    previous = root / "pilot/checkpoint.pt"
    for phase in ("indexer", "p2", "sft"):
        results.append(
            train(
                data=root / "data",
                output=root / phase,
                config=root / "model.json",
                device=device,
                steps=2,
                input_batch_tokens=64,
                init=previous,
                phase=phase,
                save_every=2,
                eval_every=2,
            )
        )
        previous = root / phase / "checkpoint.pt"
    report = dict(
        kind="mechanism_acceptance",
        config=asdict(c),
        results=results,
        checkpoint=str(previous),
        formal_training_complete=False,
        usable_chat_model=False,
    )
    write_json(root / "report.json", report)
    return report


def main(argv=None):
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    params = commands.add_parser("params")
    params.add_argument("--config")
    params.add_argument("--output")
    commands.add_parser("recipe")
    fixture = commands.add_parser("prepare-fixture")
    fixture.add_argument("--output", required=True)
    fixture.add_argument("--seed", type=int, default=42)
    prepare = commands.add_parser("prepare-data")
    prepare.add_argument("--input", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--source-allowlist", required=True)
    prepare.add_argument("--max-gib", type=float, default=8)
    tokenizer = commands.add_parser("tokenizer")
    tokenizer.add_argument("--data", required=True)
    tokenizer.add_argument("--vocab-size", type=int, choices=[32768, 65536], default=32768)
    encode = commands.add_parser("encode")
    encode.add_argument("--data", required=True)
    encode.add_argument("--output", required=True)
    encode.add_argument("--config", required=True)
    encode.add_argument("--max-gib", type=float, default=16)
    q = commands.add_parser("quickstart")
    q.add_argument("--output", required=True)
    q.add_argument("--device", default="cpu")
    q.add_argument("--updates", type=int, default=8)
    t = commands.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--output", required=True)
    t.add_argument("--phase", choices=list(PHASES), default="pilot")
    t.add_argument("--config")
    t.add_argument("--device", default="cpu")
    t.add_argument("--steps", type=int)
    t.add_argument("--token-budget", type=int)
    t.add_argument("--input-batch-tokens", type=int, default=16384)
    t.add_argument(
        "--batch-size", type=int, default=8, help="maximum independent rows per microbatch"
    )
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--init")
    t.add_argument("--resume")
    t.add_argument("--run-kind", choices=["acceptance", "strategy"], default="acceptance")
    t.add_argument("--evidence")
    t.add_argument("--diagnostic-attention", choices=["dense_pretrain"])
    t.add_argument("--stop-after-updates", type=int)
    t.add_argument("--optimizer-kind", choices=["adamw", "muon"], default="adamw")
    t.add_argument("--lr", type=float)
    t.add_argument("--vision-lr", type=float)
    t.add_argument("--save-every", type=int, default=100)
    t.add_argument("--eval-every", type=int, default=100)
    t.add_argument("--mixture", help="JSON mapping every data domain to CE proportions")
    e = commands.add_parser("evaluate")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--data", required=True)
    e.add_argument("--device", default="cpu")
    e.add_argument("--output", required=True)
    e.add_argument("--split", choices=["val", "test"], default="val")
    e.add_argument(
        "--generation",
        action="store_true",
        help="fixed greedy generation and media dependency controls",
    )
    g = commands.add_parser("generate")
    g.add_argument("--checkpoint", required=True)
    g.add_argument("--prompt", required=True)
    g.add_argument("--image", action="append")
    g.add_argument("--video-frame", action="append")
    g.add_argument("--timestamps", help="JSON list of source seconds for video frames")
    g.add_argument("--mode", choices=["direct", "thinking"], default="direct")
    g.add_argument("--device", default="cpu")
    g.add_argument("--max-new-tokens", type=int, default=64)
    g.add_argument("--temperature", type=float, default=0)
    g.add_argument("--draft-checkpoint")
    g.add_argument("--draft-steps", type=int, choices=[2, 4, 6], default=4)
    post = commands.add_parser("posttrain")
    post.add_argument("--phase", choices=["rl", "teacher", "opd", "draft", "dpo"], required=True)
    post.add_argument("--checkpoint", required=True)
    post.add_argument("--data", required=True)
    post.add_argument("--output", required=True)
    post.add_argument("--device", default="cpu")
    post.add_argument("--steps", type=int)
    post.add_argument("--max-tokens", type=int)
    post.add_argument("--group-size", type=int, default=4)
    post.add_argument("--prompts-per-update", type=int)
    post.add_argument("--teacher-registry")
    post.add_argument("--teacher-slot")
    post.add_argument("--run-kind", choices=["acceptance", "strategy"], default="acceptance")
    post.add_argument("--evidence")
    post.add_argument("--resume")
    post.add_argument("--seed", type=int, default=42)
    post.add_argument("--token-budget", type=int)
    export = commands.add_parser("export")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    export.add_argument("--int8", action="store_true")
    export.add_argument("--group-size", type=int, choices=[64, 128], default=64)
    demo = commands.add_parser("demo")
    demo.add_argument("--checkpoint", required=True)
    demo.add_argument("--device", default="cpu")
    demo.add_argument("--host", default="127.0.0.1")
    demo.add_argument("--port", type=int, default=7861)
    demo.add_argument("--allow-unqualified", action="store_true")
    teachers = commands.add_parser("train-teachers")
    teachers.add_argument("--checkpoint", required=True)
    teachers.add_argument("--data", required=True)
    teachers.add_argument("--output", required=True)
    teachers.add_argument("--device", default="cpu")
    teachers.add_argument("--steps", type=int, default=2)
    teachers.add_argument("--max-tokens", type=int, default=32)
    qualify = commands.add_parser("qualify-teacher")
    qualify.add_argument("--checkpoint", required=True)
    qualify.add_argument("--baseline", required=True)
    qualify.add_argument("--data", required=True)
    qualify.add_argument("--slot", required=True)
    qualify.add_argument("--output", required=True)
    qualify.add_argument("--device", default="cpu")
    qualify.add_argument("--limit", type=int, default=64)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    if command == "params":
        c = (
            MiniFrontier1Config(**json.loads(Path(args["config"]).read_text()))
            if args["config"]
            else MiniFrontier1Config()
        )
        with torch.device("meta"):
            model = MiniFrontier1ForCausalLM(c)
        result = parameter_report(model)
        if args["output"]:
            write_json(args["output"], result)
    elif command == "recipe":
        result = dict(budget_report(), phases=PHASES)
    elif command == "prepare-fixture":
        result = make_fixture(**args)
    elif command == "prepare-data":
        from minifrontier.data.minifrontier1 import prepare_records

        result = prepare_records(
            args["input"],
            args["output"],
            json.loads(Path(args["source_allowlist"]).read_text()),
            max_gib=args["max_gib"],
        )
    elif command == "tokenizer":
        from minifrontier.data import sha256
        from minifrontier.data.minifrontier1 import train_tokenizer

        root = Path(args["data"])
        if (root / "tokenizer.json").exists():
            raise FileExistsError("tokenizer is frozen; compare in a new dataset version")
        with (root / "train.jsonl").open() as handle:
            tok = train_tokenizer(
                (json.loads(line) for line in handle), root / "tokenizer.json", args["vocab_size"]
            )
        result = json.loads((root / "manifest.json").read_text())
        result.update(
            tokenizer_sha256=sha256(root / "tokenizer.json"),
            vocab_size=tok.get_vocab_size(),
            control_template="mf1-control-v1",
        )
        write_json(root / "manifest.json", result)
    elif command == "quickstart":
        result = quickstart(**args)
    elif command == "encode":
        from minifrontier.data.minifrontier1_encoding import encode_dataset

        args["config"] = MiniFrontier1Config(**json.loads(Path(args["config"]).read_text()))
        result = encode_dataset(**args)
    elif command == "train":
        from minifrontier.training.minifrontier1 import train

        mixture = args.pop("mixture")
        result = train(**args, weights=json.loads(mixture) if mixture else None)
    elif command == "posttrain":
        from minifrontier.training.minifrontier1_posttrain import train_post

        result = train_post(**args)
    elif command == "export":
        from minifrontier.inference.minifrontier1_export import export_checkpoint

        result = export_checkpoint(**args)
    elif command == "demo":
        from minifrontier.inference.minifrontier1_demo import serve

        serve(**args)
        return
    elif command == "train-teachers":
        from minifrontier.training.minifrontier1_teachers import train_teachers

        result = train_teachers(**args)
    elif command == "qualify-teacher":
        from minifrontier.training.minifrontier1_teachers import qualify_teacher

        result = qualify_teacher(**args)
    elif command == "evaluate":
        from minifrontier.data import sha256
        from minifrontier.inference.runtime import load_checkpoint
        from minifrontier.training.minifrontier1 import evaluate

        if args["generation"]:
            from minifrontier.evaluation.minifrontier1 import generation_suite

            result = generation_suite(
                args["checkpoint"], args["data"], device=args["device"], split=args["split"]
            )
        else:
            model, _, _ = load_checkpoint(args["checkpoint"], args["device"])
            dataset = RecordDataset(args["data"], args["split"], model.config)
            result = dict(
                checkpoint_sha256=sha256(args["checkpoint"]),
                split=args["split"],
                metrics=evaluate(model, dataset, torch.device(args["device"]), limit=0),
            )
        write_json(args["output"], result)
    else:
        from minifrontier.inference.minifrontier1 import respond

        result = respond(**args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
