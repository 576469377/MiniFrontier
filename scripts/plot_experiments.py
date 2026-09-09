"""Redraw archived validation curves; requires the optional matplotlib package."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    args = parser.parse_args()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
    for ax, model in zip(axes, ("minikimik3", "miniqwen4", "minideepseekv4"), strict=True):
        for path in sorted(args.snapshot.glob(f"strategy-recipe-pilots-v2--{model}*.json")):
            record = json.loads(path.read_text())
            points = [row for row in record["metrics"] if row["event"] == "validation"]
            ax.plot(
                [p["step"] for p in points],
                [p["lm_loss"] for p in points],
                marker=".",
                label=f"{record['run']['optimizer']} ({record['state']})",
            )
        ax.set(title=model, xlabel="Optimizer updates", ylabel="Validation LM NLL")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Recipe pilots: partial runs retained; compare within each model only")
    fig.savefig(args.snapshot / "validation.svg", metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    main()
