"""Bounded public-text plus independently verified synthetic-math recipe corpus.

This is for equal-budget recipe comparisons, not admission to main pretraining:
the code-source license audit, real science coverage and validation scale remain
open. Generated arithmetic is explicitly identified instead of claiming a public
math/science dataset or silently repeating the 2,048 diagnostic questions.
"""

import hashlib
import json
import math
import random
import sqlite3
from fractions import Fraction
from pathlib import Path

from minifrontier.data import sha256
from minifrontier.data.corpus import CorpusBuilder
from minifrontier.storage import reserve_write


def math_document(index, seed=614):
    rng = random.Random(f"{seed}:{index}")
    problems = []
    answer: int | Fraction
    for case in range(12):
        a, b, c = [rng.randrange(2, 999) for _ in range(3)]
        kind = case % 6
        if kind == 0:
            answer = a * b + c
            assert (answer - c) // a == b
            question = f"({a} * {b}) + {c}"
            explanation = f"{a} * {b} = {a * b}; {a * b} + {c} = {answer}."
        elif kind == 1:
            answer = Fraction(a, b) + Fraction(c, a)
            assert answer * (a * b) == a * a + b * c
            question = f"{a}/{b} + {c}/{a}"
            explanation = f"({a * a} + {b * c})/{a * b} = {answer}."
        elif kind == 2:
            answer = math.gcd(a, b)
            assert a % answer == b % answer == 0
            question = f"gcd({a}, {b})"
            pairs, left, right = [], a, b
            while right:
                pairs.append(f"{left} = {left // right} * {right} + {left % right}")
                left, right = right, left % right
            assert left == answer
            explanation = "; ".join(pairs) + f"; gcd = {answer}."
        elif kind == 3:
            answer = c
            question = f"{a} * x + {b} = {a * c + b}"
            explanation = f"{a} * x = {a * c + b} - {b} = {a * c}; x = {a * c}/{a} = {answer}."
        elif kind == 4:
            values = [a, b, c, rng.randrange(2, 999), rng.randrange(2, 999)]
            answer = Fraction(sum(values), len(values))
            assert answer * len(values) == sum(values)
            question = "mean(" + ", ".join(map(str, values)) + ")"
            explanation = f"sum = {sum(values)}; count = {len(values)}; mean = {answer}."
        else:
            answer = a * c * c + b * c + a
            assert answer == (a * c + b) * c + a
            question = f"p(x) = {a}*x^2 + {b}*x + {a}; p({c})"
            explanation = f"p({c}) = {a * c * c} + {b * c} + {a} = {answer}."
        problems.append(
            dict(problem=question, answer=str(answer), checked=True, explanation=explanation)
        )
    rng.shuffle(problems)
    zh = index % 2 == 0
    question_label, answer_label = ("题目", "解答") if zh else ("Problem", "Solution")
    text = "\n\n".join(
        f"{question_label}: {p['problem']}\n{answer_label}: {p['explanation']}" for p in problems
    )
    return dict(
        source="minifrontier-generated-math-recipe",
        revision="1",
        item_id=str(index),
        group_id=hashlib.sha256(text.encode()).hexdigest(),
        license="Apache-2.0",
        lang="zh" if zh else "en",
        task="verified_math_science",
        stage="pretrain",
        text=text,
        verification=dict(kind="exact-integer-and-rational-invariants", problems=problems),
        quality_flags=["synthetic-math-recipe-only", "not-science-coverage", "not-main-budget"],
    )


def build_recipe_corpus(public_root, output, *, math_documents=8192):
    public_root, output = Path(public_root), Path(output)
    if output.exists():
        raise FileExistsError("recipe corpus is immutable")
    builder = CorpusBuilder(output, seed=42, max_gib=4)
    db = sqlite3.connect(f"file:{public_root / 'corpus.sqlite'}?mode=ro", uri=True)
    copied = 0
    for (payload,) in db.execute("SELECT payload FROM samples ORDER BY id"):
        row = json.loads(payload)
        if row["stage"] == "sft":
            row = dict(
                row,
                stage="pretrain",
                task="dialogue",
                text="\n".join(t["role"].capitalize() + ": " + t["content"] for t in row["turns"]),
            )
            row.pop("turns")
        copied += int(builder.add(row))
        if copied % 5000 == 0:
            print(json.dumps(dict(copied=copied)), flush=True)
    db.close()
    for index in range(math_documents):
        builder.add(math_document(index))
        if index % 1024 == 0:
            print(
                json.dumps(dict(generated=index, accepted=builder.counts["accepted"])), flush=True
            )
    manifest = builder.finalize()
    audit = dict(
        public_corpus_sha256=sha256(public_root / "corpus-manifest.json"),
        generated_math_documents=math_documents,
        main_budget_eligible=False,
        scope="20M-token controlled recipe pilot; formal source/validation gates incomplete",
        math_scope="synthetic arithmetic/algebra only, no science claim",
        corpus=manifest,
    )
    (output / "recipe-audit.json").write_text(json.dumps(audit, indent=2))
    return audit


def add_visual_recipe(text_root, visual_root, output):
    """New immutable pilot version; share original image files without pixel copies."""
    text_root, visual_root, output = Path(text_root), Path(visual_root), Path(output)
    if output.exists():
        raise FileExistsError("joint recipe corpus is immutable")
    output.mkdir(parents=True)
    path = output / "corpus.sqlite"
    with (
        reserve_write(path, (text_root / "corpus.sqlite").stat().st_size),
        sqlite3.connect(f"file:{text_root / 'corpus.sqlite'}?mode=ro", uri=True) as source,
        sqlite3.connect(path) as destination,
    ):
        source.backup(destination)
    builder = CorpusBuilder(output, seed=42, max_gib=5)
    builder.counts.update(json.loads((text_root / "corpus-manifest.json").read_text())["counts"])
    with sqlite3.connect(f"file:{visual_root / 'corpus.sqlite'}?mode=ro", uri=True) as source:
        for (payload,) in source.execute(
            "SELECT payload FROM samples WHERE stage='pretrain' ORDER BY id"
        ):
            row = json.loads(payload)
            for media in row["media"]:
                media["path"] = str((visual_root / media["path"]).resolve())
            builder.add(row)
    result = builder.finalize()
    (output / "recipe-audit.json").write_text(
        json.dumps(
            dict(
                text_corpus_sha256=sha256(text_root / "corpus-manifest.json"),
                visual_corpus_sha256=sha256(visual_root / "corpus-manifest.json"),
                main_budget_eligible=False,
                scope="small joint optimizer recipe pilot; no visual generalization gate",
                corpus=result,
            ),
            indent=2,
        )
    )
    return result
