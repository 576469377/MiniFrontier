"""Explicit, checkpoint-bound transitions between equivalent execution versions."""

import json
import os
from pathlib import Path

from minifrontier.data import sha256


def resume_matches(previous, current, checkpoint):
    if previous == current:
        return True
    path = os.environ.get("MINIFRONTIER_EXECUTION_UPGRADE")
    if not path:
        return False
    report = json.loads(Path(path).read_text())
    before, after = dict(previous), dict(current)
    old_source, new_source = before.pop("source", None), after.pop("source", None)
    old_evidence, new_evidence = None, None
    if "strategy" in before and "strategy" in after:
        before["strategy"], after["strategy"] = dict(before["strategy"]), dict(after["strategy"])
        old_evidence = before["strategy"].pop("evidence_sha256", None)
        new_evidence = after["strategy"].pop("evidence_sha256", None)
    if before != after or not new_source or new_source.get("dirty"):
        return False
    if not (
        report.get("kind") == "execution_only_resume"
        and report.get("authorization")
        and report.get("model") == current.get("model_name")
        and report.get("previous_source") == old_source
        and report.get("current_source") == new_source
        and report.get("previous_evidence_sha256") == old_evidence
        and report.get("current_evidence_sha256") == new_evidence
        and report.get("checkpoint_sha256") == sha256(checkpoint)
    ):
        return False
    proof = report.get("verification", {})
    proof_path = Path(proof.get("path", ""))
    if not proof_path.is_file() or sha256(proof_path) != proof.get("sha256"):
        return False
    checks = json.loads(proof_path.read_text())
    return checks.get("passed") is True and checks.get("calculation_unchanged") is True
