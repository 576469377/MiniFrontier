"""Immutable MF1 text/media compositions that reuse existing compact shards and pixels."""

import json
import os
import shutil
from bisect import bisect_right
from collections import Counter
from dataclasses import asdict
from itertools import accumulate
from pathlib import Path

from tokenizers import Tokenizer

from minifrontier.data import sha256
from minifrontier.data.media_cache import validate_policy
from minifrontier.data.minifrontier1 import digest, write_json
from minifrontier.data.minifrontier1_encoding import FORMAT, CompactDataset
from minifrontier.models.minifrontier1.processing import PROCESSOR_VERSION
from minifrontier.storage import require_space

COMPONENT_FORMAT = "mf1-compact-components-v1"


def assemble_components(components, output, config, *, media_access=None):
    """Bind disjoint canonical components; retain their domains, order and media roots.

    Only a tokenizer copy and the composition manifest are written. This checks
    metadata identity and split compatibility, not source quality or phase admission.
    """
    roots, output = [Path(p).resolve() for p in components], Path(output).resolve()
    if not roots or output.exists() or len(set(roots)) != len(roots):
        raise ValueError("choose distinct components and a new composition output")
    access = {Path(k).resolve(): validate_policy(v) for k, v in (media_access or {}).items()}
    if access.keys() - set(roots):
        raise ValueError("media access names a component outside this composition")
    references, manifests = [], []
    tokenizer_hash = None
    for root in roots:
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest.get("format") != FORMAT or manifest.get("kind") not in {
            "canonical_text_component",
            "canonical_image_component",
            "canonical_video_component",
        }:
            raise ValueError("composition requires completed canonical compact components")
        if (
            manifest["config_sha256"] != digest(asdict(config))
            or manifest["processor_version"] != PROCESSOR_VERSION
        ):
            raise ValueError("component model config or media processor differs")
        current = sha256(root / "tokenizer.json")
        if current != manifest["tokenizer_sha256"] or (
            tokenizer_hash is not None and current != tokenizer_hash
        ):
            raise ValueError("component tokenizer mapping differs")
        tokenizer_hash = current
        references.append(
            dict(path=os.path.relpath(root, output), manifest_sha256=sha256(root / "manifest.json"))
        )
        if root in access:
            if manifest["kind"] not in {"canonical_image_component", "canonical_video_component"}:
                raise ValueError("remote media access is only for canonical media components")
            references[-1]["media_access"] = access[root]
        manifests.append(manifest)
    samples: set[str] = set()
    sample_splits: dict[str, str] = {}
    groups: dict[str, str] = {}
    media: dict[str, str] = {}
    splits = {}
    for split in ("train", "val", "test"):
        counts: Counter[str] = Counter()
        domain_ce: Counter[str] = Counter()
        unique_media = set()
        for root, manifest in zip(roots, manifests, strict=True):
            dataset = CompactDataset(root, split, config)
            observed = 0
            for number, part in enumerate(dataset.parts):
                path = dataset._validated_file(number, "metadata.jsonl")
                part_records = 0
                with path.open() as handle:
                    for line in handle:
                        record = json.loads(line)
                        identity, group = record["sample_id"], record["split_group"]
                        if identity in samples:
                            raise ValueError("duplicate sample in compact composition")
                        samples.add(identity)
                        if sample_splits.setdefault(identity, split) != split:
                            raise ValueError("source text identity crosses derivative splits")
                        if groups.setdefault(group, split) != split:
                            raise ValueError("connected group crosses composition splits")
                        text_origin = record.get("origin", {}).get("text_origin")
                        if text_origin is not None:
                            if text_origin.get("split") != split or any(
                                not isinstance(text_origin.get(key), str) or not text_origin[key]
                                for key in ("sample_id", "group_root")
                            ):
                                raise ValueError(
                                    "OCR derivative lacks its inherited text partition"
                                )
                            if (
                                sample_splits.setdefault(text_origin["sample_id"], split) != split
                                or groups.setdefault(text_origin["group_root"], split) != split
                            ):
                                raise ValueError("source text identity crosses derivative splits")
                        for resource in record["resources"]:
                            key = resource.get("rgb_sha256", resource.get("sha256"))
                            if not key:
                                raise ValueError("canonical media resource has no identity")
                            identities = [key, *resource.get("frame_rgb_sha256", [])]
                            if any(
                                media.setdefault(identity, split) != split
                                for identity in identities
                            ):
                                raise ValueError("media identity crosses composition splits")
                            unique_media.add(key)
                        part_records += 1
                if part_records != part["counts"]["records"]:
                    raise ValueError("component metadata record count differs")
                observed += part_records
            if observed != manifest["splits"][split]["counts"].get("records", 0):
                raise ValueError("component split record count differs")
            counts.update(manifest["splits"][split]["counts"])
            domain_ce.update(manifest["splits"][split]["domain_ce"])
        splits[split] = dict(
            counts=dict(counts), domain_ce=dict(domain_ce), unique_media=len(unique_media)
        )
    result = dict(
        format=COMPONENT_FORMAT,
        kind="canonical_text_image_video_composition"
        if any(m["kind"] == "canonical_video_component" for m in manifests)
        else "canonical_text_image_composition",
        formal_admission=False,
        main_budget_eligible=False,
        config_sha256=digest(asdict(config)),
        processor_version=PROCESSOR_VERSION,
        tokenizer_sha256=tokenizer_hash,
        components=references,
        splits=splits,
        sample_order="component argument order, then original split order",
        shard_files_copied=0,
        raw_media_copied=False,
        identity_checks=dict(
            duplicate_samples=0, cross_split_connected_groups=0, cross_split_media_identities=0
        ),
        processor_sha256=sha256(__file__),
    )
    require_space(
        output,
        (roots[0] / "tokenizer.json").stat().st_size + len(json.dumps(result).encode()) + 1024**2,
    )
    output.mkdir(parents=True)
    shutil.copyfile(roots[0] / "tokenizer.json", output / "tokenizer.json")
    write_json(output / "manifest.json", result)
    return result


class ComponentDataset:
    """A stable index over compact components, using their existing window/media loaders."""

    def __init__(self, root, split, config):
        self.root, self.config = Path(root).resolve(), config
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if (
            self.manifest.get("format") != COMPONENT_FORMAT
            or self.manifest["config_sha256"] != digest(asdict(config))
            or self.manifest["processor_version"] != PROCESSOR_VERSION
            or sha256(self.root / "tokenizer.json") != self.manifest["tokenizer_sha256"]
        ):
            raise ValueError("composition config, processor or tokenizer differs")
        self.tokenizer = Tokenizer.from_file(str(self.root / "tokenizer.json"))
        self.datasets = []
        for reference in self.manifest["components"]:
            path = (self.root / reference["path"]).resolve()
            if sha256(path / "manifest.json") != reference["manifest_sha256"]:
                raise ValueError("component manifest changed after composition")
            policy = reference.get("media_access")
            if policy is not None:
                policy = dict(policy, cache_dir=str((self.root / policy["cache_dir"]).resolve()))
            dataset = CompactDataset(path, split, config, media_access=policy)
            if dataset.manifest["tokenizer_sha256"] != self.manifest["tokenizer_sha256"]:
                raise ValueError("component tokenizer differs from composition")
            self.datasets.append(dataset)
        self.ends = list(accumulate(len(d) for d in self.datasets))
        if not self.datasets or len(self) != self.manifest["splits"][split]["counts"].get(
            "records", 0
        ):
            raise ValueError("composition split length differs")

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def _locate(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        component = bisect_right(self.ends, index)
        return self.datasets[component], index - (self.ends[component - 1] if component else 0)

    def __getitem__(self, index):
        dataset, index = self._locate(index)
        return dataset[index]

    def length_at(self, index):
        dataset, index = self._locate(index)
        return dataset.length_at(index)

    def domain_at(self, index):
        dataset, index = self._locate(index)
        return dataset.domain_at(index)

    def windowable_at(self, index):
        dataset, index = self._locate(index)
        return dataset.windowable_at(index)

    def window_at(self, index, start, capacity):
        dataset, index = self._locate(index)
        return dataset.window_at(index, start, capacity)

    def ce_count_at(self, index, start=0, capacity=None):
        dataset, index = self._locate(index)
        return dataset.ce_count_at(index, start, capacity)
