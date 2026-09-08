"""Immutable completed rollout records; resumed runs never reuse old policy actions."""

import hashlib
import json
import os
from pathlib import Path

from minifrontier.storage import reserve_write
from minifrontier.training.tool_environment import VERSION


def persist(root, prepared, dataset, *, policy_hash, version, run_spec):
    rows = []
    for index, trace in enumerate(prepared["traces"]):
        task = dataset.rows[trace["task_index"]]
        length = int(prepared["inputs"][index].ne(0).sum())
        mask = prepared["labels"][index, :length].ne(-100).tolist()
        rows.append(
            dict(
                schema_version=1,
                task_id=task.get("id", trace["task_index"]),
                task_sha256=hashlib.sha256(
                    json.dumps(task, ensure_ascii=False, sort_keys=True).encode()
                ).hexdigest(),
                media=task.get("media", []),
                domain=task["domain"],
                mode=task.get("mode"),
                effort=task["effort"],
                chat_template=dataset.chat_template,
                policy_sha256=policy_hash,
                frozen_ratio_max_error=prepared["behavior_ratio_error"],
                frozen_ratio_tolerance=prepared["behavior_ratio_tolerance"],
                policy_version=version,
                run_sha256=hashlib.sha256(
                    json.dumps(run_spec, sort_keys=True).encode()
                ).hexdigest(),
                source=run_spec.get("source"),
                tokenizer_sha256=run_spec.get("tokenizer_sha256"),
                teacher_sha256=run_spec.get("rollout", {}).get("teacher_sha256", {}),
                input_ids=prepared["inputs"][index, :length].tolist(),
                action_mask=mask,
                behavior_logp=[0.0, *prepared["old_logp"][index, : length - 1].tolist()],
                termination=prepared["terminations"][index],
                reward_components=dict(
                    verified_task=prepared["rewards"][index],
                    tool_errors=trace.get("tool_errors", 0),
                    format_complete=prepared["terminations"][index] in {"final", "eos"},
                ),
                verifier=task.get("verifier", {"kind": "integer", "answer": task.get("answer")}),
                verifier_version="local-verifiers-v2",
                environment_version=VERSION if task.get("environment") else None,
                trace=trace,
                lifecycle="completed_synchronous_rollout_before_update; never reused on resume",
            )
        )
    content = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows
    ).encode()
    digest = hashlib.sha256(content).hexdigest()
    path = Path(root) / f"{version['split']}-s{version['step']}-r{version['rank']}-{digest}.jsonl"
    with reserve_write(path, len(content)):
        if path.exists():
            if path.read_bytes() != content:
                raise RuntimeError("immutable trajectory content mismatch")
        else:
            temporary = path.with_suffix(".tmp")
            try:
                with temporary.open("xb") as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
    return dict(path=str(path), sha256=digest, trajectories=len(rows), bytes=len(content))
