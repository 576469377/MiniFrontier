"""Public commands for the current MiniFrontier implementation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "prepare-public-data":
        from minifrontier.public_sources import main as public_data

        public_data(argv[1:])
        return
    if argv and argv[0] == "prepare-diagnostics":
        from minifrontier.diagnostic_data import build_diagnostics

        p = argparse.ArgumentParser(
            description="Generated K0/Q0/D0 diagnostics, excluded from formal data"
        )
        p.add_argument("--output", required=True)
        p.add_argument("--seed", type=int, default=142)
        p.add_argument("--images", type=int, default=256)
        p.add_argument("--texts", type=int, default=2048)
        print(json.dumps(build_diagnostics(**vars(p.parse_args(argv[1:]))), indent=2))
        return
    if argv and argv[0] == "prepare-tokenizers":
        from minifrontier.data_v2 import compare_tokenizers

        p = argparse.ArgumentParser(
            description="Compare 32K/64K on identical bytes and freeze strategy default"
        )
        p.add_argument("--corpus-root", required=True)
        p.add_argument("--output", required=True)
        p.add_argument("--byte-budget", type=int, default=64 * 1024**2)
        print(json.dumps(compare_tokenizers(**vars(p.parse_args(argv[1:]))), indent=2))
        return
    if argv and argv[0] == "encode-data":
        from minifrontier.data_v2 import encode_corpus
        from minifrontier.native_data import encode_native

        p = argparse.ArgumentParser(
            description="Immutable text or native multimodal token encoding"
        )
        p.add_argument("--corpus-root", required=True)
        p.add_argument("--tokenizer-path", required=True)
        p.add_argument("--output", required=True)
        p.add_argument("--max-length", type=int, default=4096)
        p.add_argument("--family", choices=["minikimik3", "miniqwen4", "minideepseekv4"])
        p.add_argument("--media-root")
        p.add_argument("--max-features", type=int, default=256)
        p.add_argument("--model-vocab-size", type=int, default=65536)
        values = vars(p.parse_args(argv[1:]))
        if values["family"]:
            result = encode_native(**values)
        else:
            for key in ("family", "media_root", "max_features", "model_vocab_size"):
                values.pop(key)
            result = encode_corpus(**values)
        print(json.dumps(result, indent=2))
        return
    if argv and argv[0] == "train":
        from minifrontier.training.train import main as train

        train(argv[1:])
        return
    if argv and argv[0] == "train-draft":
        from minifrontier.training.train_draft import main as train_draft

        train_draft(argv[1:])
        return
    if argv and argv[0] == "generate":
        from minifrontier.inference import main as generate

        generate(argv[1:])
        return
    if argv and argv[0] == "prepare-rl-data":
        from minifrontier.training.rollouts import prepare_tasks

        tasks_parser = argparse.ArgumentParser(description="Generate verifiable arithmetic tasks")
        tasks_parser.add_argument("--output", required=True)
        tasks_parser.add_argument("--count", type=int, default=10000)
        tasks_parser.add_argument("--seed", type=int, default=42)
        prepare_tasks(**vars(tasks_parser.parse_args(argv[1:])))
        return
    if argv and argv[0] == "prepare-data":
        from minifrontier.data import prepare_data

        data_parser = argparse.ArgumentParser(description="Build a reproducible educational corpus")
        data_parser.add_argument("--output", required=True)
        data_parser.add_argument("--pretrain-rows", type=int, default=60000)
        data_parser.add_argument("--sft-rows", type=int, default=30000)
        data_parser.add_argument("--dpo-rows", type=int, default=10000)
        data_parser.add_argument("--sequence-length", type=int, default=256)
        data_parser.add_argument("--vocab-size", type=int, default=65536)
        data_parser.add_argument("--seed", type=int, default=42)
        data_parser.add_argument("--revision")
        data_parser.add_argument(
            "--sampling",
            choices=["reservoir", "prefix"],
            default="reservoir",
            help="reservoir reads the full source for an unbiased sample; prefix is for smoke tests",
        )
        prepare_data(**vars(data_parser.parse_args(argv[1:])))
        return
    if argv and argv[0] == "demo":
        from minifrontier.inference import serve

        demo_parser = argparse.ArgumentParser(description="Run the local MiniFrontier browser demo")
        demo_parser.add_argument("--root", default="outputs")
        demo_parser.add_argument("--host", default="127.0.0.1")
        demo_parser.add_argument("--port", type=int, default=7860)
        demo_parser.add_argument("--device", default="cpu")
        serve(**vars(demo_parser.parse_args(argv[1:])))
        return
    from minifrontier import __version__

    parser = argparse.ArgumentParser(
        prog="minifrontier",
        description="Models, doctor, prepare-data, train, generate and demo for MiniFrontier.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    models = commands.add_parser("models", help="show current model status and configurations")
    models.add_argument("--manifest", help="model catalog JSON; defaults to the bundled catalog")
    models.add_argument(
        "--count-backbones",
        action="store_true",
        help="instantiate implemented models on CPU and count parameters",
    )
    commands.add_parser("doctor", help="inspect the local GPU environment")
    for name, help_text in {
        "prepare-public-data": "sample pinned public sources with a storage budget",
        "prepare-tokenizers": "train same-byte 32K/64K tokenizer candidates",
        "prepare-diagnostics": "construct the separate correctness corpus",
        "encode-data": "encode immutable text/native multimodal corpora",
        "prepare-data": "construct public training data and a tokenizer",
        "prepare-rl-data": "generate tasks with verifiable arithmetic rewards",
        "train": "train or resume any text model and stage",
        "train-draft": "adapt a target-bound Kimi, Qwen or DSpark draft",
        "generate": "generate text from a local checkpoint",
        "demo": "serve the local browser demo",
    }.items():
        commands.add_parser(name, help=help_text)

    args = parser.parse_args(argv)
    if args.command == "models":
        from minifrontier.catalog import default_manifest_path, inspect_current_models

        result = inspect_current_models(
            args.manifest or default_manifest_path(), count_backbones=args.count_backbones
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "doctor":
        from minifrontier.hardware import print_doctor

        print_doctor()
