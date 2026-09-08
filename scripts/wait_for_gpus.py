#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import subprocess
import time
from collections.abc import Sequence

from minifrontier.hardware import query_gpus, select_free_gpus


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wait for free GPUs, then run a command on them.")
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--min-free-gib", type=float, default=20.0)
    parser.add_argument(
        "--name-contains",
        help="only select GPUs whose nvidia-smi name contains this text (case-insensitive)",
    )
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.count < 1:
        parser.error("count must be positive")
    if args.poll_seconds < 1:
        parser.error("poll-seconds must be positive")
    if not math.isfinite(args.min_free_gib) or args.min_free_gib <= 0:
        parser.error("min-free-gib must be finite and positive")
    name_contains = args.name_contains.strip() if args.name_contains is not None else None
    if args.name_contains is not None and not name_contains:
        parser.error("name-contains must be non-empty when provided")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("provide a command after --")
    while True:
        selected = select_free_gpus(
            args.count,
            args.min_free_gib,
            name_contains=name_contains,
        )
        if len(selected) == args.count:
            break
        status = ", ".join(f"gpu{gpu.index}:{gpu.free_gib:.1f}GiB" for gpu in query_gpus())
        print(
            f"waiting for {args.count} GPUs with {args.min_free_gib:.1f} GiB free ({status})",
            flush=True,
        )
        time.sleep(args.poll_seconds)
    environment = dict(os.environ)
    # UUIDs are stable across CUDA enumeration modes, whereas nvidia-smi's
    # numeric indices need not match CUDA's FASTEST_FIRST logical ordinals.
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(gpu.uuid for gpu in selected)
    selection = ", ".join(f"gpu{gpu.index}={gpu.uuid} ({gpu.name})" for gpu in selected)
    print(f"selected physical GPUs: {selection}", flush=True)
    raise SystemExit(subprocess.call(command, env=environment))


if __name__ == "__main__":
    main()
