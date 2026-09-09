"""Small, generated learnability fixtures, explicitly excluded from main budgets."""
# ruff: noqa: RUF001

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from minifrontier.data.corpus import CorpusBuilder
from minifrontier.data.media_hash import decoded_hashes
from minifrontier.storage import require_space


def build_diagnostics(output, *, seed=142, images=256, texts=2048):
    root = Path(output).resolve()
    if root.exists():
        raise FileExistsError("diagnostic corpus is immutable; choose a new version")
    if not 64 <= images <= 512:
        raise ValueError("strategy diagnostic image budget is 64–512")
    require_space(root.parent, 64 * 1024**2)
    root.mkdir(parents=True)
    (root / "images").mkdir()
    builder = CorpusBuilder(root, seed=seed, max_gib=1)
    rng = random.Random(seed)
    colors = [
        ("red", "红色", (230, 35, 35)),
        ("green", "绿色", (30, 180, 55)),
        ("blue", "蓝色", (40, 70, 225)),
        ("yellow", "黄色", (230, 210, 30)),
    ]
    for index in range(images):
        pixels = np.random.default_rng(seed + index).integers(0, 256, (8, 8, 3), dtype=np.uint8)
        image = Image.fromarray(pixels).resize((112, 112), Image.Resampling.NEAREST)
        color, zh_color, rgb = colors[index % len(colors)]
        draw = ImageDraw.Draw(image)
        draw.rectangle((28, 28, 84, 84), fill=rgb)
        path = root / "images" / f"{index:04d}.png"
        image.save(path)
        text = (
            f"<|image|> The central square is {color}. The border is a colored mosaic."
            if index % 2
            else f"<|image|> 图片中央是一个{zh_color}正方形，周围是彩色马赛克背景。"
        )
        builder.add(
            dict(
                source="minifrontier-generated-visual-diagnostic",
                revision="1",
                item_id=str(index),
                group_id=str(index),
                license="Apache-2.0",
                lang="en" if index % 2 else "zh",
                task="diagnostic_image",
                stage="pretrain",
                text=text,
                quality_flags=["generated-diagnostic-only", "not-main-training-data"],
                media=[
                    dict(
                        path=str(path.relative_to(root)),
                        kind="image",
                        min_pixels=3136,
                        **decoded_hashes(image),
                    )
                ],
            )
        )
    for index in range(texts):
        a, b = rng.randrange(1, 5000), rng.randrange(1, 5000)
        text = (
            f"问题：{a} 加 {b} 等于多少？\n答案：{a} + {b} = {a + b}。这是两个正整数的加法运算。"
            if index % 2
            else f"Question: What is the sum of {a} and {b}?\nAnswer: {a} + {b} = {a + b}. Addition combines the two positive integers."
        )
        builder.add(
            dict(
                source="minifrontier-generated-math-diagnostic",
                revision="1",
                item_id=str(index),
                group_id=f"addition:{min(a, b)}:{max(a, b)}",
                license="Apache-2.0",
                lang="zh" if index % 2 else "en",
                task="diagnostic_text",
                stage="pretrain",
                text=text,
                verifier=dict(kind="integer-addition", operands=[a, b], answer=a + b),
                quality_flags=["generated-diagnostic-only", "not-main-training-data"],
            )
        )
    result = builder.finalize()
    (root / "diagnostic-recipe.json").write_text(
        json.dumps(
            dict(
                seed=seed,
                requested_images=images,
                requested_texts=texts,
                main_budget_eligible=False,
                purpose="K0/Q0/D0 learnability; this does not establish real-world language or vision quality",
            ),
            indent=2,
        )
    )
    return result
