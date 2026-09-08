"""Two controlled 20M-CE trials per family, after learnability diagnostics.

Runs remain acceptance experiments. Neither trial completion nor a lower NLL
launches main PT/SFT/RL or promotes a demo checkpoint.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from minifrontier.data import sha256
from minifrontier.provenance import source_identity
from minifrontier.storage import GIB, require_space


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--model", choices=["minikimik3", "miniqwen4", "minideepseekv4"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--diagnostic", type=Path, help="explicit reviewed diagnostic continuation")
    p.add_argument(
        "--source-root", type=Path, help="frozen training checkout; also set PYTHONPATH to it"
    )
    args = p.parse_args()
    workspace, output = args.workspace.resolve(), args.output.resolve()
    source_root = (args.source_root or Path(__file__).resolve().parents[1]).resolve()
    import minifrontier

    if Path(minifrontier.__file__).resolve().parents[1] != source_root:
        raise ValueError("imported training implementation does not match frozen source-root")
    identity = source_identity()
    if identity["dirty"] or not identity["commit"]:
        raise ValueError("recipe experiments require an immutable source checkout")
    if output.exists():
        raise FileExistsError("pilot output already exists; preserve previous experiments")
    output.mkdir(parents=True)
    gpu_ids = {"minikimik3": "0,1", "miniqwen4": "2,3", "minideepseekv4": "4,5"}[args.model]
    diagnostic = (
        args.diagnostic or workspace / "outputs/strategy-diagnostics-v2" / args.model
    ).resolve()
    data = (
        workspace
        / "data"
        / (
            "strategy-recipe-encoded-64k-v2"
            if args.model == "minideepseekv4"
            else f"strategy-recipe-{args.model}-64k-v2"
        )
    )
    state = dict(
        source=identity,
        controller_sha256=sha256(__file__),
        diagnostic=str(diagnostic),
        model=args.model,
        gpu_ids=gpu_ids,
        stage="waiting_diagnostic",
        main_budget_eligible=False,
        comparisons="Muon vs AdamW, same 20M CE/data/seed/input batch",
        pending_comparisons=[
            "learning_rate",
            "mtp_coefficients",
            "32k_vs_64k_quality",
            "additional_seeds",
        ],
        trials=[],
    )

    def write():
        (output / "pilot.json").write_text(json.dumps(state, indent=2))
        print(json.dumps(dict(stage=state["stage"], trials=len(state["trials"]))), flush=True)

    env = dict(
        os.environ,
        OMP_NUM_THREADS="2",
        TOKENIZERS_PARALLELISM="false",
        CUDA_VISIBLE_DEVICES=gpu_ids,
        HF_HOME=str(workspace / "data/hf-cache"),
        TRITON_CACHE_DIR=str(workspace / "outputs/kernel-cache"),
        TORCHINDUCTOR_CACHE_DIR=str(workspace / "outputs/inductor-cache"),
        MINIFRONTIER_MIN_FREE_GIB="50",
    )
    try:
        write()
        deadline = time.monotonic() + 12 * 3600
        while True:
            status_path = diagnostic / "status.json"
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            if (
                status.get("state") == "complete"
                and (diagnostic / "model.pt").exists()
                and (data / "manifest.json").exists()
            ):
                break
            if time.monotonic() > deadline:
                raise TimeoutError("diagnostic/data preparation did not complete in 12 hours")
            time.sleep(30)
        state["stage"] = "checking_training_recall"
        write()
        recall = output / "diagnostic-arithmetic.json"
        with (output / "diagnostic-arithmetic.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.evaluate_arithmetic_diagnostics",
                    "--checkpoint",
                    str(diagnostic / "model.pt"),
                    "--corpus",
                    str(workspace / "data/strategy-diagnostic-v2"),
                    "--tokenizer",
                    str(data / "tokenizer.json"),
                    "--output",
                    str(recall),
                ],
                cwd=source_root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        measured = json.loads(recall.read_text())
        if measured["summary"]["train"] != {"total": 16, "correct": 16}:
            raise ValueError(
                "diagnostic training recall failed; extend or repair K0/Q0/D0 before a recipe trial"
            )
        state["diagnostic_recall"] = measured["summary"]
        if args.model != "minideepseekv4":
            visual = output / "diagnostic-visual.json"
            with (output / "diagnostic-visual.log").open("w") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "scripts.evaluate_visual_diagnostics",
                        "--checkpoint",
                        str(diagnostic / "checkpoint.pt"),
                        "--data",
                        str(workspace / f"data/strategy-diagnostic-{args.model}-v2"),
                        "--output",
                        str(visual),
                    ],
                    cwd=source_root,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            measured = json.loads(visual.read_text())
            if (
                measured["summary"]["correct_image"]["exact_color_sequences"]
                != measured["evaluated_images"]
                or measured["summary"]["wrong_image"]["exact_color_sequences"]
                > measured["evaluated_images"] // 2
            ):
                raise ValueError(
                    "diagnostic visual dependence failed; recipe pilot remains blocked"
                )
            state["diagnostic_visual"] = measured["summary"]
        plan = json.loads((source_root / f"configs/strategies/{args.model}-plan.json").read_text())
        mixture_path = output / "mixture.json"
        mixture = plan["text_mixture_tokens"]
        if args.model != "minideepseekv4":
            mixture = dict(
                schema_version=1,
                ce_token_budget=20_000_000,
                image_occurrences=1000,
                video_examples=0,
                text_mixture_tokens=mixture,
                image_mixture_samples={"caption": 1.0},
                scope="local recipe pilot with small real-image pool; formal visual pool composition not met",
            )
        mixture_path.write_text(json.dumps(mixture, indent=2))
        state["data_sha256"] = sha256(data / "manifest.json")
        for optimizer in ("auto", "adamw"):
            require_space(output, 12 * GIB)
            trial = output / ("muon-20m" if optimizer == "auto" else "adamw-20m")
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
                args.model,
                "--stage",
                "pretrain",
                "--config",
                str(source_root / f"configs/strategies/{args.model}-v2.json"),
                "--data",
                str(data),
                "--output",
                str(trial),
                "--ce-tokens",
                "20000000",
                "--steps",
                "100000",
                "--sequence-length",
                "512",
                "--batch-size",
                "1",
                "--input-batch-tokens",
                "16384",
                "--lr",
                "0.0003",
                "--muon-lr",
                "0.01",
                "--adam-eps",
                "1e-8",
                "--optimizer",
                optimizer,
                "--warmup-tokens",
                "400000",
                "--schedule",
                "wsd",
                "--save-every",
                "200",
                "--eval-every",
                "200",
                "--eval-batches",
                "0",
                "--log-every",
                "10",
                "--run-kind",
                "acceptance",
                "--profile-warmup",
                "50",
                "--profile-updates",
                "200",
                "--min-device-free-gib",
                "2",
                "--media-mixture" if args.model != "minideepseekv4" else "--token-mixture",
                str(mixture_path),
            ]
            if args.model != "minideepseekv4":
                command += ["--vision-lr", "0.0001", "--projector-lr", "0.0003"]
            state["stage"] = "running_" + optimizer
            entry = dict(
                optimizer=optimizer, output=str(trial), command=command, started_at=time.time()
            )
            state["trials"].append(entry)
            write()
            with trial.with_suffix(".log").open("w") as log:
                subprocess.run(
                    command,
                    cwd=source_root,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            entry["finished_at"] = time.time()
            entry["status"] = json.loads((trial / "status.json").read_text())
            write()
        state["stage"] = "two_trials_complete_further_comparisons_required"
        write()
    except BaseException as error:
        state.update(stage="stopped", error=type(error).__name__ + ": " + str(error))
        write()
        raise


if __name__ == "__main__":
    main()
