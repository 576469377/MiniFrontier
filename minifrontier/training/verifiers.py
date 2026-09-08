"""Explicit task rewards; expected code answers stay outside the student worker."""

import json
import os
import re
import selectors
import subprocess
import time
from pathlib import Path


def check_python(code, entry_point, tests):
    if len(code.encode()) > 64 * 1024 or not re.fullmatch(r"[A-Za-z_]\w*", entry_point):
        return dict(score=0.0, reason="invalid_code_or_entry_point")
    fenced = re.fullmatch(r"\s*```(?:python)?\s*\n(.*?)\n```\s*", code, re.S)
    code = fenced[1] if fenced else code
    payload = json.dumps(
        dict(
            code=code,
            entry_point=entry_point,
            cases=[
                {key: value for key, value in case.items() if key in {"args", "kwargs"}}
                for case in tests
            ],
        )
    ).encode()
    if not tests or len(payload) > 120 * 1024:
        raise ValueError("code verifier requires bounded nonempty test arguments")
    process = subprocess.Popen(
        ["/usr/bin/python3", "-I", "-S", str(Path(__file__).with_name("code_worker.py"))],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env={"PATH": "/usr/bin", "LANG": "C.UTF-8", "PYTHONHASHSEED": "0"},
        close_fds=True,
    )
    assert process.stdin is not None and process.stdout is not None
    selector = selectors.DefaultSelector()
    try:
        process.stdin.write(payload)
        process.stdin.close()
        selector.register(process.stdout, selectors.EVENT_READ)
        output, deadline = bytearray(), time.monotonic() + 5
        while time.monotonic() < deadline:
            if not selector.select(timeout=min(0.1, max(0, deadline - time.monotonic()))):
                continue
            chunk = os.read(process.stdout.fileno(), 8192)
            if not chunk:
                break
            output.extend(chunk)
            if len(output) > 64 * 1024:
                return dict(score=0.0, reason="output_limit")
        if process.poll() is None:
            try:
                process.wait(timeout=0.1)
            except subprocess.TimeoutExpired:
                return dict(score=0.0, reason="timeout")
        if process.returncode:
            return dict(
                score=0.0, reason="worker_rejected_or_resource_limit", returncode=process.returncode
            )
        try:
            result = json.loads(output)
        except (ValueError, UnicodeDecodeError):
            return dict(score=0.0, reason="invalid_worker_result")
        if result.get("isolation_failed"):
            raise RuntimeError("kernel seccomp isolation is unavailable; refusing generated code")
        values = result.get("outputs", [])
        passed = sum(value == case["expected"] for value, case in zip(values, tests, strict=False))
        return dict(
            score=passed / len(tests),
            passed=passed,
            total=len(tests),
            reason=result.get("error", "completed"),
        )
    finally:
        selector.close()
        if process.poll() is None:
            process.kill()
        process.wait()
        process.stdout.close()


def reward(text, specification):
    kind = specification["kind"]
    if kind == "integer":
        match = re.fullmatch(r"\s*([-+]?\d+)\s*[.!。]?\s*", text)
        return float(match is not None and int(match[1]) == int(specification["answer"]))
    if kind == "exact_text":
        return float(text.strip() == specification["answer"].strip())
    if kind == "json":
        try:
            return float(json.loads(text) == specification["expected"])
        except (ValueError, TypeError):
            return 0.0
    if kind == "python_function":
        return check_python(text, specification["entry_point"], specification["tests"])["score"]
    raise ValueError("unimplemented verifier; never silently substitute arithmetic reward")
