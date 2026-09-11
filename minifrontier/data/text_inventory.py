"""Exact cross-split text checks over an immutable effective corpus partition.

Index only held-out documents and stream training documents. This closes gaps in
the construction-time Simhash filter without storing another copy of the corpus.
It does not certify semantic/paraphrase decontamination or within-train dedup.
"""

import argparse
import contextlib
import hashlib
import json
import time
from array import array
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from minifrontier.data import sha256
from minifrontier.data.corpus import text_shingles
from minifrontier.data.partitions import corpus_storage_root, open_corpus
from minifrontier.storage import GIB, reserve_write

METHOD_REFERENCE = "https://www.cse.unsw.edu.au/~lxue/WWW08.pdf"


def _prefix_size(size):
    return size - (85 * size + 99) // 100 + 1


class ExactTextIndex:
    """Jaccard >=85/100 with lossless prefix candidates and exact set overlap.

    See Xiao et al., WWW 2008, section 3. A common total order and prefixes of
    length n-ceil(.85*n)+1 guarantee overlap for qualifying pairs. Strings absent
    from the held-out vocabulary sort first; remaining strings sort by held-out
    frequency, then lexical order. Integer ranks are bijective, not hashes.
    """

    def __init__(self, signatures, *, max_candidates=20_000_000, max_comparisons=5_000_000):
        if not signatures or any(not values for values in signatures):
            raise ValueError("exact text index requires nonempty reference sets")
        if min(max_candidates, max_comparisons) < 1:
            raise ValueError("text comparison bounds must be positive")
        frequency = Counter(token for values in signatures for token in values)
        self.ranks = {
            token: rank
            for rank, token in enumerate(sorted(frequency, key=lambda t: (frequency[t], t)))
        }
        self.values = [{self.ranks[token] for token in values} for values in signatures]
        self.postings: dict[int, array] = {}
        for key, values in enumerate(self.values):
            for token in sorted(values)[: _prefix_size(len(values))]:
                self.postings.setdefault(token, array("I")).append(key)
        self.max_candidates, self.max_comparisons = max_candidates, max_comparisons
        self.candidates = self.comparisons = 0

    def neighbors(self, values, *, eligible=None):
        if not values:
            raise ValueError("text query must contain shingles")
        shared = {rank for token in values if (rank := self.ranks.get(token)) is not None}
        novel = len(values) - len(shared)
        remaining = _prefix_size(len(values)) - novel
        if remaining <= 0:
            return
        candidates = set()
        for token in sorted(shared)[:remaining]:
            for key in self.postings.get(token, ()):
                if key in candidates or (eligible is not None and not eligible(key)):
                    continue
                candidates.add(key)
                self.candidates += 1
                if self.candidates > self.max_candidates:
                    raise ValueError("text candidate budget exceeded; audit is incomplete")
        for key in sorted(candidates):
            other = self.values[key]
            if 100 * min(len(values), len(other)) < 85 * max(len(values), len(other)):
                continue
            self.comparisons += 1
            if self.comparisons > self.max_comparisons:
                raise ValueError("text comparison budget exceeded; audit is incomplete")
            overlap = len(shared & other)
            union = len(values) + len(other) - overlap
            if 100 * overlap >= 85 * union:
                yield key, overlap, union


def audit_text_holdouts(
    corpus,
    output,
    *,
    inventory="shared-text",
    max_reference_shingles=60_000_000,
    max_candidates=20_000_000,
    max_comparisons=5_000_000,
    max_matches=10_000,
    max_seconds=3600,
    max_bytes=16 * 1024**2,
    progress=None,
):
    """Check every train/val, train/test and val/test pair at the declared metric.

    Includes code regardless of AST equivalence: quarantining near-identical
    held-out content is conservative and does not assert program equivalence.
    A bound violation raises and never publishes a completed audit.
    """
    root, output = Path(corpus).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("text audit is immutable; choose a new output")
    if min(max_reference_shingles, max_matches, max_seconds, max_bytes) < 1 or not inventory:
        raise ValueError("text audit bounds and inventory must be positive/nonempty")
    started = time.monotonic()
    manifest_path = root / "corpus-manifest.json"
    manifest_hash = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    counts: Counter[str] = Counter()
    members, signatures = [], []
    groups = {}
    matches: list[dict[str, Any]] = []
    parent: dict[str, str] = {}
    identity_digest = hashlib.sha256()
    reference_shingles = 0
    index = None

    def report(state):
        elapsed = time.monotonic() - started
        if elapsed > max_seconds:
            raise ValueError("text audit time budget exceeded; audit is incomplete")
        if progress is not None:
            progress(
                dict(
                    state=state,
                    elapsed_seconds=elapsed,
                    scanned_records=dict(counts),
                    reference_shingles=reference_shingles,
                    candidates=index.candidates if index else 0,
                    comparisons=index.comparisons if index else 0,
                    matched_pairs=len(matches),
                )
            )

    def representative(group):
        parent.setdefault(group, group)
        while parent[group] != group:
            parent[group] = parent[parent[group]]
            group = parent[group]
        return group

    def compare(member, values, eligible):
        assert index is not None
        for key, overlap, union in index.neighbors(values, eligible=eligible):
            other = members[key]
            if len(matches) >= max_matches:
                raise ValueError("text match budget exceeded; audit is incomplete")
            matches.append(
                dict(
                    samples=[member["sample_id"], other["sample_id"]],
                    groups=[member["group"], other["group"]],
                    splits=[member["split"], other["split"]],
                    intersection=overlap,
                    union=union,
                    jaccard=overlap / union,
                )
            )
            left, right = representative(member["group"]), representative(other["group"])
            parent[max(left, right)] = min(left, right)

    report("validating_source")
    with contextlib.closing(open_corpus(root)) as db:
        database = corpus_storage_root(db) / "corpus.sqlite"
        database_stat = database.stat()
        database_identity = (
            database_stat.st_dev,
            database_stat.st_ino,
            database_stat.st_size,
            database_stat.st_mtime_ns,
        )
        if sha256(database) != manifest["database_sha256"]:
            raise ValueError("text audit source database differs from manifest")
        for group, split, count in db.execute(
            "SELECT group_root,split,COUNT(*) FROM samples GROUP BY group_root,split"
        ):
            if not group or group in groups or split not in {"train", "val", "test"}:
                raise ValueError("one effective group has invalid or mixed split membership")
            groups[group] = dict(inventory=inventory, group=group, split=split, records=count)
        if db.execute(
            "SELECT 1 FROM samples WHERE COALESCE(json_array_length(payload,'$.media'),0)>0 LIMIT 1"
        ).fetchone():
            raise ValueError("text holdout audit requires a text-only corpus")
        query = "SELECT id,group_root,split,text FROM samples WHERE split {} 'train' ORDER BY id"
        for identity, group, split, text in db.execute(query.format("!=")):
            values = text_shingles(text)
            reference_shingles += len(values)
            if reference_shingles > max_reference_shingles:
                raise ValueError("text reference shingle budget exceeded; audit is incomplete")
            members.append(dict(sample_id=identity, group=group, split=split))
            signatures.append(values)
            counts[split] += 1
            identity_digest.update(f"{identity}:{group}:{split}\n".encode())
            if len(members) % 1000 == 0:
                report("loading_holdouts")
        report("building_exact_prefix_index")
        index = ExactTextIndex(
            signatures, max_candidates=max_candidates, max_comparisons=max_comparisons
        )
        for key, (member, values) in enumerate(zip(members, signatures, strict=True)):
            compare(
                member,
                values,
                lambda k, key=key, split=member["split"]: k < key and members[k]["split"] != split,
            )
            if key % 1000 == 0:
                report("checking_validation_test")
        signatures.clear()
        for identity, group, split, text in db.execute(query.format("=")):
            compare(dict(sample_id=identity, group=group, split=split), text_shingles(text), None)
            counts[split] += 1
            identity_digest.update(f"{identity}:{group}:{split}\n".encode())
            if counts[split] % 1000 == 0:
                report("checking_training_holdouts")
        if dict(counts) != manifest["splits"]:
            raise ValueError("text audit effective record counts differ from manifest")
        final_stat = database.stat()
        if (
            sha256(manifest_path) != manifest_hash
            or (final_stat.st_dev, final_stat.st_ino, final_stat.st_size, final_stat.st_mtime_ns)
            != database_identity
        ):
            raise ValueError("text audit source changed during the read")
    components = defaultdict(list)
    for group in sorted(parent):
        components[representative(group)].append(groups[group])
    conflicts = [
        dict(
            required_split="test" if any(m["split"] == "test" for m in component) else "val",
            members=component,
        )
        for _, component in sorted(components.items())
    ]
    report("all_pairs_checked")
    result = dict(
        kind="cross_corpus_text_group_audit",
        status="split_conflicts_require_partition_update"
        if conflicts
        else "cross_split_check_passed",
        inputs={
            inventory: dict(
                corpus_manifest_sha256=manifest_hash,
                database_sha256=manifest["database_sha256"],
            )
        },
        scope="all train/val, train/test and val/test pairs; no within-split dedup or semantic/paraphrase guarantee",
        method="exact whitespace-free normalized casefold five-character shingle Jaccard >=85/100; collision-free prefix index; no Simhash prefilter",
        method_reference=METHOD_REFERENCE,
        code_policy="include code regardless of AST equivalence; quarantine is not semantic dedup",
        effective_records=dict(counts),
        member_sha256=identity_digest.hexdigest(),
        reference_shingles=reference_shingles,
        vocabulary_shingles=len(index.ranks),
        candidates=index.candidates,
        comparisons=index.comparisons,
        matches=matches,
        split_conflicts=conflicts,
        full_shared_text_cross_split_audit_complete=True,
        formal_admission=False,
        main_budget_eligible=False,
        processor_sha256=sha256(__file__),
        shingle_processor_sha256=sha256(Path(__file__).with_name("corpus.py")),
        elapsed_seconds=time.monotonic() - started,
    )
    content = (json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode()
    if len(content) > max_bytes:
        raise ValueError("text audit output byte budget exceeded; no complete report published")
    with reserve_write(output, len(content), reserve_bytes=80 * GIB), output.open("xb") as stream:
        stream.write(content)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", required=True)
    args = vars(parser.parse_args())
    result = audit_text_holdouts(**args, progress=lambda row: print(json.dumps(row), flush=True))
    print(json.dumps(dict(status=result["status"], matched_pairs=len(result["matches"]))))


if __name__ == "__main__":
    main()
