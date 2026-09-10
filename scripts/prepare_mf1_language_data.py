"""Prepare bounded, verifiable SFT diagnostics with a checkpoint-compatible tokenizer."""
# ruff: noqa: RUF001 -- natural Chinese punctuation in training examples

import argparse
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

from minifrontier.data import sha256
from minifrontier.data.minifrontier1 import digest, encode_record, validate_record, write_json
from minifrontier.models.minifrontier1 import MiniFrontier1Config
from minifrontier.storage import require_space

VERSION = "mf1-language-diagnostic-v1"
QA = [
    ("你好。", "你好！有什么我可以帮助你的吗？"),
    ("请用一句话介绍你自己。", "我是 MiniFrontier，一个用于学习和研究的小型语言模型。"),
    ("谢谢你的帮助。", "不客气！"),
    ("再见。", "再见！祝你今天愉快。"),
    ("你叫什么名字？", "我叫 MiniFrontier。"),
    ("你是真人吗？", "我不是人，我是一个语言模型。"),
    ("把 hello 翻译成中文。", "你好。"),
    ("把谢谢翻译成英文。", "Thank you."),
    ("7+5=", "12"),
    ("3乘以4等于多少？", "12"),
    ("10减去6等于多少？", "4"),
    ("8除以2等于多少？", "4"),
    ("一年有几个月？", "一年有十二个月。"),
    ("一周有几天？", "一周有七天。"),
    ("太阳通常从哪个方向升起？", "太阳通常从东方升起。"),
    ("给出一个水果的名字。", "苹果。"),
    ("只输出字母 ABC。", "ABC"),
    ("请重复：春天来了。", "春天来了。"),
    ("句子“小明拿着红色的书”中，书是什么颜色？", "红色。"),
    ("句子“盒子里有两个苹果”中，盒子里有什么？", "两个苹果。"),
    ("Hello.", "Hello! How can I help you?"),
    ("What is your name?", "My name is MiniFrontier."),
    ("Are you a person?", "No. I am a language model."),
    ("Thank you.", "You are welcome!"),
    ("Goodbye.", "Goodbye! Have a nice day."),
    ("Translate apple into Chinese.", "苹果。"),
    ("Write the number after 4.", "5"),
    ("How many days are in a week?", "There are seven days in a week."),
    ("Name one color.", "Blue."),
    ("Repeat exactly: bright sky", "bright sky"),
    ("Output only the word yes.", "yes"),
    ("The box contains a green ball. What color is the ball?", "Green."),
]


def record(prompt, answer, group, task):
    ident = digest([VERSION, group, prompt])
    return dict(
        sample_id=ident,
        split_group=f"{VERSION}:{group}",
        language="zh" if any("\u4e00" <= x <= "\u9fff" for x in prompt) else "en",
        domain="language",
        task=task,
        expected=answer,
        source=dict(dataset=VERSION, revision="1", record_id=ident),
        provenance=dict(license_record="Apache-2.0; project-authored diagnostic examples"),
        supervision=dict(type="answer_ce"),
        media=[],
        messages=[
            dict(role="user", content=[dict(type="text", text=prompt)]),
            dict(role="assistant", channel="final", content=[dict(type="text", text=answer)]),
        ],
    )


def generated_records(seed=42):
    """Keep every paraphrase/language of a canonical problem in the same split."""
    result: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    for a in range(11, 35):
        for b in range(a, 35):
            group = f"add:{a}:{b}"
            split = "val" if int(digest([seed, group])[:8], 16) % 10 == 0 else "train"
            for prompt in (
                f"计算 {a}+{b}，只输出答案。",
                f"What is {b}+{a}? Output only the answer.",
            ):
                result[split].append(record(prompt, str(a + b), group, "addition"))
    for i in range(160):
        payload = f"word{i:03d}"
        group = f"copy:{payload}"
        split = "val" if int(digest([seed, group])[:8], 16) % 5 == 0 else "train"
        for prompt in (f"请原样输出：{payload}", f"Repeat exactly: {payload}"):
            result[split].append(record(prompt, payload, group, "copy"))
    return result


def prepare(source, output, config, seed=42):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("diagnostic datasets are immutable; choose a new directory")
    require_space(output, 64 * 1024**2)
    output.mkdir(parents=True)
    tokenizer = Tokenizer.from_file(str(source / "tokenizer.json"))
    parent = json.loads((source / "manifest.json").read_text())
    assert sha256(source / "tokenizer.json") == parent["tokenizer_sha256"]
    generated = generated_records(seed)
    media: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "val"):
        candidates = [
            json.loads(line) for line in (source / f"{split}.jsonl").read_text().splitlines()
        ]
        media[split] = []
        for domain, limit in (
            ("vision", 32 if split == "train" else 16),
            ("video", 8 if split == "train" else 16),
        ):
            media[split].extend([r for r in candidates if r["domain"] == domain][:limit])
    summaries = {}
    control = [record(p, a, f"authored:{i}", "authored_qa") for i, (p, a) in enumerate(QA)]
    # The same held-out generated problems are used in both recipes. They never enter training.
    validation = generated["val"] + media["val"]
    for name, text_train in (("control", control), ("instructions", control + generated["train"])):
        root = output / name
        root.mkdir()
        shutil.copyfile(source / "tokenizer.json", root / "tokenizer.json")
        groups, files = {}, {}
        counts: dict[str, dict[str, Any]] = {}
        for split, records in (("train", text_train + media["train"]), ("val", validation)):
            random.Random(seed).shuffle(records)
            groups[split] = {r["split_group"] for r in records}
            counts[split] = dict(
                records=len(records),
                domains=dict(Counter(r["domain"] for r in records)),
                ce_tokens=0,
                input_tokens=0,
                max_input_tokens=0,
            )
            for r in records:
                validate_record(r, source)
                item = encode_record(r, tokenizer, config, source)
                length = item["input_ids"].numel()
                if length > 512:
                    raise ValueError("diagnostic record exceeds 512 tokens; no implicit truncation")
                counts[split]["ce_tokens"] += int(item["labels"][:, 1:].ne(-100).sum())
                counts[split]["input_tokens"] += length
                counts[split]["max_input_tokens"] = max(counts[split]["max_input_tokens"], length)
            path = root / f"{split}.jsonl"
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
            files[split] = sha256(path)
        if groups["train"] & groups["val"]:
            raise ValueError("canonical problem or media group crosses training/validation")
        manifest = dict(
            format="mf1-native-records-v1",
            kind=VERSION,
            recipe=name,
            seed=seed,
            formal_admission=False,
            main_budget_eligible=False,
            media_root=os.path.relpath(source, root),
            tokenizer_sha256=sha256(root / "tokenizer.json"),
            vocab_size=tokenizer.get_vocab_size(),
            source_manifest_sha256=sha256(source / "manifest.json"),
            preparation_script_sha256=sha256(__file__),
            counts=counts,
            files=files,
            limitations=[
                "Controlled learning diagnosis, not general conversation training",
                "Training QA memorization and held-out task generalization must be reported separately",
                "Visual replay uses generated color tasks only",
                "Existing tokenizer copied byte-for-byte; sealed test not read",
            ],
        )
        write_json(root / "manifest.json", manifest)
        summaries[name] = manifest
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = MiniFrontier1Config(**json.loads(Path(args.config).read_text()))
    result = prepare(args.source, args.output, config, args.seed)
    print(json.dumps({k: v["counts"] for k, v in result.items()}, indent=2))


if __name__ == "__main__":
    main()
