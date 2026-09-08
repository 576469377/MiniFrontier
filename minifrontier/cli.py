"""Public commands for the current MiniFrontier implementation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "train":
        from minifrontier.training.train import main as train

        train(argv[1:])
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
        "prepare-data": "construct public training data and a tokenizer",
        "prepare-rl-data": "generate tasks with verifiable arithmetic rewards",
        "train": "train or resume any text model and stage",
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
