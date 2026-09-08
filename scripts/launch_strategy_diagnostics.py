"""Launch the three independent two-GPU correctness runs from a frozen source checkout."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from minifrontier.provenance import source_identity
from minifrontier.storage import GIB, require_space


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ce-tokens", default=500_000, type=int)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["minikimik3", "miniqwen4", "minideepseekv4"],
        choices=["minikimik3", "miniqwen4", "minideepseekv4"],
    )
    args = parser.parse_args()
    if not 500_000 <= args.ce_tokens <= 2_000_000:
        raise ValueError("strategy diagnostic budget is 0.5M-2M CE tokens")
    identity = source_identity()
    if not identity["commit"] or identity["dirty"]:
        raise ValueError("diagnostics require a clean committed checkout; preserve it for resume")
    require_space(args.output.parent, 40 * GIB)
    source = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)
    pairs = dict(minikimik3="0,1", miniqwen4="2,3", minideepseekv4="4,5")
    launches = []
    for name in args.models:
        output = args.output / name
        if output.exists():
            raise FileExistsError(f"refusing to overwrite existing diagnostic run: {output}")
        data = (
            args.workspace
            / "data"
            / (
                "strategy-diagnostic-text-v2"
                if name == "minideepseekv4"
                else f"strategy-diagnostic-{name}-v2"
            )
        )
        mixture = args.output / f"{name}-mixture.json"
        proportions = (
            {"diagnostic_text": 1.0}
            if name == "minideepseekv4"
            else {"diagnostic_text": 0.8, "diagnostic_image": 0.2}
        )
        mixture.write_text(json.dumps(proportions))
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=pairs[name],
            OMP_NUM_THREADS="2",
            MKL_NUM_THREADS="2",
            TOKENIZERS_PARALLELISM="false",
            TRITON_CACHE_DIR=str(args.workspace / "outputs/kernel-cache"),
            TORCHINDUCTOR_CACHE_DIR=str(args.workspace / "outputs/inductor-cache"),
            HF_HOME=str(args.workspace / "data/hf-cache"),
            MINIFRONTIER_MIN_FREE_GIB="50",
        )
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            "-m",
            "minifrontier",
            "train",
            "--model",
            name,
            "--stage",
            "pretrain",
            "--config",
            str(source / "configs/strategies" / f"{name}-v2.json"),
            "--data",
            str(data),
            "--output",
            str(output),
            "--ce-tokens",
            str(args.ce_tokens),
            "--steps",
            "10000",
            "--sequence-length",
            "64",
            "--batch-size",
            "4",
            "--grad-accum",
            "4",
            "--lr",
            "0.0003",
            "--muon-lr",
            "0.01",
            "--warmup-tokens",
            "20000",
            "--eval-every",
            "100",
            "--eval-batches",
            "0",
            "--save-every",
            "100",
            "--log-every",
            "10",
            "--run-kind",
            "acceptance",
            "--token-mixture",
            str(mixture),
            "--profile-warmup",
            "50",
            "--profile-updates",
            "200",
            "--min-device-free-gib",
            "2",
        ]
        log_path = args.output / f"{name}.log"
        with log_path.open("xb") as log:
            process = subprocess.Popen(
                command,
                cwd=source,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        launches.append(
            dict(
                model=name,
                pid=process.pid,
                gpu_ids=pairs[name],
                command=command,
                log=str(log_path),
                source=identity,
                main_budget_eligible=False,
            )
        )
    (args.output / "launches.json").write_text(json.dumps(launches, indent=2))
    print(json.dumps(launches, indent=2))


if __name__ == "__main__":
    main()
