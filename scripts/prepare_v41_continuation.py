"""Bind a V-C → temporary visual decoder → joint V4.1 run to local manifests."""

import argparse
from pathlib import Path

from minifrontier.data.text import sha256
from scripts.v41_vision_curriculum import read, status


def prepare(*, pairs, visual, mixed, run, output, config_dir):
    pairs, visual, mixed, run, output, config_dir = (
        Path(p).resolve() for p in (pairs, visual, mixed, run, output, config_dir)
    )
    if output.exists():
        raise FileExistsError(output)
    tok = pairs / "tokenizer.json"
    token_hash = sha256(tok)
    if read(pairs / "manifest.json")["tokenizer_sha256"] != token_hash:
        raise ValueError("caption manifest disagrees with tokenizer")
    visual_recipe, mixed_recipe = [p / "media-mixture.json" for p in (visual, mixed)]
    for data, recipe, budget in (
        (visual, visual_recipe, 8_000_000),
        (mixed, mixed_recipe, 2_500_000_000),
    ):
        if (
            read(data / "manifest.json")["tokenizer"]["sha256"] != token_hash
            or sha256(data / "tokenizer.json") != token_hash
        ):
            raise ValueError("native stage and V-C must share their tokenizer")
        if read(recipe)["ce_token_budget"] != budget:
            raise ValueError("native mixture differs from the curriculum CE budget")
    if read(mixed_recipe)["image_occurrences"] != 1_000_000:
        raise ValueError("joint recipe must include the planned one million image exposures")
    temporary = config_dir / "minideepseekv41-vision-temporary.json"
    main = config_dir / "minideepseekv41-vision-main.json"
    c = run / "V-C"
    stages = []

    def add(name, artifact, args, files):
        stages.append(
            dict(
                stage=name,
                artifact=str(artifact),
                receipt=str(output.parent / (name + "-complete.json")),
                args=list(map(str, args)),
                bindings={str(p): sha256(p) for p in files},
            )
        )

    common = [
        "--stage",
        "pretrain",
        "--model",
        "minideepseekv41",
        "--batch-size",
        "2",
        "--grad-accum",
        "4",
        "--sequence-length",
        "2048",
        "--seed",
        "42",
        "--save-every",
        "200",
        "--eval-every",
        "1000",
        "--eval-batches",
        "8",
        "--log-every",
        "5",
        "--lr",
        "0.0003",
        "--optimizer",
        "v41_muon_sinkhorn",
    ]
    add(
        "attach-temporary",
        run / "temporary-init.pt",
        [
            "attach",
            "--role",
            "temporary",
            "--tower",
            c / "tower.pt",
            "--config",
            temporary,
            "--tokenizer",
            tok,
            "--seed",
            "20261110",
        ],
        [temporary, tok],
    )
    add(
        "visual-ar",
        run / "V-AR/model.pt",
        [
            *common,
            "--data",
            visual,
            "--media-mixture",
            visual_recipe,
            "--ce-tokens",
            "8000000",
            "--max-data-epochs",
            "1",
            "--init",
            run / "temporary-init.pt",
            "--output",
            run / "V-AR",
        ],
        [visual / "manifest.json", visual_recipe, temporary],
    )
    add(
        "extract-vit",
        run / "generative-vit.pt",
        ["extract", "--checkpoint", run / "V-AR/model.pt"],
        [temporary],
    )
    add(
        "attach-main",
        run / "main-init.pt",
        [
            "attach",
            "--role",
            "main",
            "--tower",
            run / "generative-vit.pt",
            "--config",
            main,
            "--tokenizer",
            tok,
            "--seed",
            "20261111",
        ],
        [main, tok],
    )
    add(
        "joint",
        run / "joint/model.pt",
        [
            *common,
            "--data",
            mixed,
            "--media-mixture",
            mixed_recipe,
            "--ce-tokens",
            "2500000000",
            "--max-data-epochs",
            "6",
            "--schedule",
            "wsd",
            "--v41-thaw-ce",
            "1550000000",
            "--vision-lr",
            "0.00003",
            "--projector-lr",
            "0.0003",
            "--init",
            run / "main-init.pt",
            "--output",
            run / "joint",
        ],
        [mixed / "manifest.json", mixed_recipe, main],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    status(output, dict(schema="v41-continuation-v1", contrastive_output=str(c), stages=stages))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("pairs", "visual", "mixed", "run", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--config-dir", type=Path, default=Path("configs/strategies"))
    prepare(**vars(p.parse_args()))
