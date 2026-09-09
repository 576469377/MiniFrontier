"""Immutable native-media records alongside shared memory-mapped text documents."""

import json
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from minifrontier.chat_controls import record_template, update_manifest
from minifrontier.data import sha256
from minifrontier.data.corpus import DocumentDataset, encode_corpus
from minifrontier.multimodal import prepare_record
from minifrontier.storage import reserve_write


def encode_native(
    corpus_root,
    tokenizer_path,
    output,
    family,
    *,
    max_length=4096,
    max_features=256,
    media_root=None,
    model_vocab_size=None,
):
    output, corpus_root = Path(output), Path(corpus_root)
    manifest = encode_corpus(corpus_root, tokenizer_path, output, max_length=max_length)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocab = model_vocab_size or tokenizer.get_vocab_size()
    db = sqlite3.connect(f"file:{corpus_root / 'corpus.sqlite'}?mode=ro", uri=True)
    for stage in ("pretrain", "sft"):
        for split in ("train", "val", "test"):
            path = output / f"{stage}.{split}.media.jsonl"
            offsets = []
            rejected: Counter[str] = Counter()
            template_counts: Counter[str] = Counter()
            examples = ce = image_count = frame_count = features = 0
            records = db.execute(
                "SELECT payload FROM samples WHERE stage=? AND split=? ORDER BY id", (stage, split)
            )
            with path.open("wb") as stream:
                for (payload,) in records:
                    row = json.loads(payload)
                    if not row.get("media"):
                        continue
                    # Decoder errors are surfaced during admission, never silently
                    # replaced by blank images during training.
                    prepared = prepare_record(
                        row,
                        tokenizer,
                        family,
                        root=media_root,
                        max_features=max_features,
                        model_vocab_size=vocab,
                    )
                    if prepared.input_ids.shape[1] > max_length:
                        rejected["complete_media_answer_exceeds_bucket"] += 1
                        continue
                    serialized = (
                        json.dumps(
                            dict(
                                record=row,
                                expected_ids=prepared.input_ids[0].tolist(),
                                expected_labels=prepared.labels[0].tolist(),
                            ),
                            ensure_ascii=False,
                        ).encode()
                        + b"\n"
                    )
                    with reserve_write(path, len(serialized) + 128):
                        offsets.append(
                            (stream.tell(), len(serialized), prepared.input_ids.shape[1])
                        )
                        stream.write(serialized)
                    examples += 1
                    if stage == "sft":
                        template_counts[record_template(row)] += 1
                    ce += int(prepared.labels[:, 1:].ne(-100).sum())
                    image_count += prepared.image_count
                    frame_count += prepared.frame_count
                    features += prepared.image_features
            index = path.with_suffix(".index.npy")
            np.save(index, np.asarray(offsets, dtype=np.int64).reshape(-1, 3))
            text = manifest["stages"][stage][split]
            media = dict(
                file=path.name,
                sha256=sha256(path),
                index_file=index.name,
                index_sha256=sha256(index),
                examples=examples,
                supervised_tokens=ce,
                image_occurrences=image_count,
                frames=frame_count,
                image_features=features,
                rejected=dict(rejected),
                chat_template_counts=dict(template_counts),
            )
            manifest["stages"][stage][split] = dict(
                format="hybrid-native-v2",
                text=text,
                media=media,
                examples=text["examples"] + examples,
                supervised_tokens=text["supervised_tokens"] + ce,
            )
    db.close()
    manifest.update(
        format="hybrid-native-v2",
        family=family,
        max_features=max_features,
        media_root=str(Path(media_root or ".").resolve()),
        model_vocab_size=vocab,
    )
    update_manifest(manifest)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


class NativeDataset:
    def __init__(self, root, record, stage, length):
        self.root, self.length = Path(root), length
        manifest = json.loads((self.root / "manifest.json").read_text())
        self.family, self.max_features = manifest["family"], manifest["max_features"]
        self.media_root, self.vocab = manifest["media_root"], manifest["model_vocab_size"]
        self.tokenizer = Tokenizer.from_file(str(self.root / "tokenizer.json"))
        self.text = DocumentDataset(root, record["text"], stage, length)
        media = record["media"]
        for path, digest in (
            (media["file"], media["sha256"]),
            (media["index_file"], media["index_sha256"]),
        ):
            if sha256(self.root / path) != digest:
                raise ValueError("native dataset hash mismatch")
        self.path = self.root / media["file"]
        index = np.load(self.root / media["index_file"], mmap_mode="r")
        self.index = index[index[:, 2] <= length]
        self.domains = list(self.text.domains)
        self.ce_counts = list(self.text.ce_counts)
        self.input_counts = list(self.text.input_counts)
        self.image_counts = [0] * len(self.text)
        self.video_counts = [0] * len(self.text)
        with self.path.open("rb") as source:
            for offset, size, positions in self.index:
                source.seek(int(offset))
                row = json.loads(source.read(int(size)))
                self.domains.append(row["record"]["task"])
                self.ce_counts.append(sum(value != -100 for value in row["expected_labels"][1:]))
                self.input_counts.append(int(positions))
                resources = row["record"].get("media", [])
                self.image_counts.append(sum(m.get("kind", "image") != "video" for m in resources))
                self.video_counts.append(sum(m.get("kind") == "video" for m in resources))

    def __len__(self):
        return len(self.text) + len(self.index)

    def __getitem__(self, index):
        if index < len(self.text):
            from minifrontier.multimodal import TrainingBatch

            x, y = self.text[index]
            return TrainingBatch(x[None], y[None])
        offset, size, _ = self.index[index - len(self.text)]
        with self.path.open("rb") as stream:
            stream.seek(int(offset))
            row = json.loads(stream.read(int(size)))
        prepared = prepare_record(
            row["record"],
            self.tokenizer,
            self.family,
            root=self.media_root,
            max_features=self.max_features,
            model_vocab_size=self.vocab,
        )
        if (
            prepared.input_ids[0].tolist() != row["expected_ids"]
            or prepared.labels[0].tolist() != row["expected_labels"]
        ):
            raise ValueError("native processor/template changed since immutable encoding")
        return prepared
