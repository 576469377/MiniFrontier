"""Pinned evaluation inventory and bounded text-overlap exclusion for base data.

Only data files are read. No downloaded solution or generated code is executed.
The inventory is excluded from training, including upstream training splits;
public benchmark test answers are reserved for final evaluation, not LR selection.
"""

import argparse
import gzip
import hashlib
import io
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import ahocorasick
import requests

from minifrontier.data import sha256
from minifrontier.data.minifrontier1 import write_json
from minifrontier.storage import GIB, require_space, reserve_write

SOURCES: dict[str, dict[str, Any]] = {
    "humaneval": dict(
        repo="openai/human-eval",
        revision="6d43fb980f9fee3c892a914eda09951f772ad10d",
        file="data/HumanEval.jsonl.gz",
        blob_sha1="998d25196e17af24daf9b6cb3a975fe752528e46",
        license="MIT",
        expected_rows=164,
    ),
    "mbpp": dict(
        repo="google-research/google-research",
        revision="08a8d6736475776f42ffac23b2c13111a28e5795",
        file="mbpp/mbpp.jsonl",
        blob_sha1="96e2dc2f7dd95b7b87b38dc46e42cbed444251bb",
        license="CC-BY-4.0 (dataset); upstream repository license retained separately",
        license_evidence="https://huggingface.co/datasets/google-research-datasets/mbpp",
        expected_rows=974,
    ),
    "gsm8k_test": dict(
        repo="openai/grade-school-math",
        revision="3101c7d5072418e28b9008a6636bde82a006892c",
        file="grade_school_math/data/test.jsonl",
        blob_sha1="e4c2ff4942b9a78bd74f04141224c11e28d12dc9",
        license="MIT",
        expected_rows=1319,
    ),
    "gsm8k_train": dict(
        repo="openai/grade-school-math",
        revision="3101c7d5072418e28b9008a6636bde82a006892c",
        file="grade_school_math/data/train.jsonl",
        blob_sha1="7d97154b91aef28d01c3301741c81ce90039a4b1",
        license="MIT",
        expected_rows=7473,
    ),
}
RULES: dict[str, Any] = dict(
    normalization="NFKC + casefold + Unicode word boundaries; matching only",
    prompt_min_characters=32,
    overlapping_word_ngram=13,
    ngram_min_characters=64,
    matched_group_policy="remove the complete connected document/repository group",
)


def match_text(value):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", value).casefold()))


def _download(url, path, *, blob=None):
    # Keep network waits outside the shared filesystem writer lock.
    content = bytearray()
    with requests.get(url, stream=True, timeout=(30, 60)) as response:
        response.raise_for_status()
        for chunk in response.iter_content(65536):
            content.extend(chunk)
            if len(content) > 8 * 1024**2:
                raise ValueError("evaluation source exceeds its 8 MiB file bound")
    if (
        blob
        and hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
        != blob
    ):
        raise ValueError("evaluation source Git blob hash differs")
    with reserve_write(path, len(content), reserve_bytes=80 * GIB):
        path.write_bytes(content)


def prepare_evaluation(output):
    root = Path(output).resolve()
    if root.exists():
        raise FileExistsError("evaluation inventory is immutable; choose a new version")
    require_space(root, 128 * 1024**2, reserve_bytes=80 * GIB)
    root.mkdir(parents=True)
    files = {}
    counts: Counter[str] = Counter()
    with (root / "items.jsonl").open("w") as stream:
        for name, spec in SOURCES.items():
            raw = root / (name + (".jsonl.gz" if name == "humaneval" else ".jsonl"))
            prefix = f"https://raw.githubusercontent.com/{spec['repo']}/{spec['revision']}"
            _download(prefix + "/" + spec["file"], raw, blob=spec["blob_sha1"])
            files[raw.name] = sha256(raw)
            for notice in ("README.md", "LICENSE"):
                path = root / (name + "." + notice)
                source_path = (
                    "mbpp/README.md" if name == "mbpp" and notice == "README.md" else notice
                )
                _download(prefix + "/" + source_path, path)
                files[path.name] = sha256(path)
            content = raw.read_bytes()
            if name == "humaneval":
                with gzip.GzipFile(fileobj=io.BytesIO(content)) as handle:
                    content = handle.read(16 * 1024**2 + 1)
            if len(content) > 16 * 1024**2:
                raise ValueError("evaluation decompression limit exceeded")
            for number, line in enumerate(content.splitlines()):
                row = json.loads(line)
                identity = str(row.get("task_id", number))
                prompt = row.get("prompt", row.get("text", row.get("question")))
                answer = row.get("canonical_solution", row.get("code", row.get("answer")))
                if not isinstance(prompt, str) or not isinstance(answer, str):
                    raise ValueError("evaluation row is missing its prompt/answer")
                upstream_test = name in {"humaneval", "gsm8k_test"} or (
                    name == "mbpp" and 11 <= int(identity) <= 510
                )
                item = dict(
                    id=name + ":" + identity,
                    source=name,
                    prompt=prompt,
                    answer=answer,
                    tests=row.get("test", row.get("test_list")),
                    evaluation_role="sealed_test"
                    if upstream_test
                    else "excluded_upstream_non_test",
                    training_eligible=False,
                )
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
                counts[name] += 1
            if counts[name] != spec["expected_rows"]:
                raise ValueError("pinned evaluation row count differs")
    files["items.jsonl"] = sha256(root / "items.jsonl")
    manifest = dict(
        schema_version=1,
        kind="sealed_text_benchmark_inventory",
        sources=SOURCES,
        rules=RULES,
        files=files,
        counts=dict(counts),
        processor_sha256=sha256(__file__),
        limitations=[
            "exact normalized prompts and 13-word overlaps do not detect all paraphrases/translations",
            "this inventory does not audit visual benchmark identities",
            "no benchmark solution is executed by data preparation",
        ],
    )
    write_json(root / "manifest.json", manifest)
    return manifest


class BenchmarkMatcher:
    def __init__(self, root):
        root = Path(root)
        self.manifest = json.loads((root / "manifest.json").read_text())
        if self.manifest["rules"] != RULES:
            raise ValueError("benchmark matching rules differ from the frozen inventory")
        for name, expected in self.manifest["files"].items():
            path = (root / name).resolve()
            if not path.is_relative_to(root.resolve()) or sha256(path) != expected:
                raise ValueError("benchmark inventory file path/hash differs")
        self.automaton = ahocorasick.Automaton()
        for line in (root / "items.jsonl").read_text().splitlines():
            item = json.loads(line)
            for field in ("prompt", "answer"):
                words = match_text(item[field]).split()
                phrases = [" ".join(words)] if field == "prompt" else []
                ngram = RULES["overlapping_word_ngram"]
                phrases.extend(
                    " ".join(words[i : i + ngram])
                    for i in range(len(words) - ngram + 1)
                    if len(" ".join(words[i : i + ngram])) >= RULES["ngram_min_characters"]
                )
                for phrase in phrases:
                    if len(phrase) >= RULES["prompt_min_characters"]:
                        self.automaton.add_word(
                            " " + phrase + " ",
                            dict(
                                item_id=item["id"],
                                field=field,
                                phrase_sha256=hashlib.sha256(phrase.encode()).hexdigest(),
                            ),
                        )
        if not len(self.automaton):
            raise ValueError("empty benchmark exclusion inventory")
        self.automaton.make_automaton()

    def match(self, record):
        repo = str(record.get("repo_id", "")).casefold().removesuffix(".git")
        if repo in {
            "openai/human-eval",
            "openai/grade-school-math",
            "google-research/google-research",
        }:
            return dict(item_id="upstream_repository", field="repo_id")
        for _, match in self.automaton.iter(" " + match_text(record["text"]) + " "):
            return match
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print(json.dumps(prepare_evaluation(parser.parse_args().output), indent=2))


if __name__ == "__main__":
    main()
