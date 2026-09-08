"""Explicit task rewards; expected code answers stay outside the student worker."""

import json
import math
import os
import re
import selectors
import subprocess
import time
from pathlib import Path


def run_python(code, entry_point, tests):
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
        if not isinstance(result, dict):
            return dict(reason="invalid_worker_result")
        if result.get("isolation_failed"):
            raise RuntimeError("kernel seccomp isolation is unavailable; refusing generated code")
        if not isinstance(result.get("outputs"), list) or len(result["outputs"]) != len(tests):
            return dict(reason=result.get("error", "invalid_worker_result"))
        return dict(outputs=result["outputs"], reason="completed")
    finally:
        selector.close()
        if process.poll() is None:
            process.kill()
        process.wait()
        process.stdout.close()


def check_python(code, entry_point, tests):
    result = run_python(code, entry_point, tests)
    passed = sum(
        value == case["expected"]
        for value, case in zip(result.get("outputs", []), tests, strict=False)
    )
    return dict(result, score=passed / len(tests), passed=passed, total=len(tests))


def reward(text, specification, *, environment=None):
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
    if kind == "quantity":
        # The task declares the dimension and trusted conversion factors. Never
        # extract the first number from prose, or compare unlike units implicitly.
        try:
            value = json.loads(text)
            if not isinstance(value, dict) or set(value) != {"value", "unit"}:
                return 0.0
            if type(value["value"]) not in {float, int} or not math.isfinite(value["value"]):
                return 0.0
            factors = specification["unit_factors"]
            actual = value["value"] * factors[value["unit"]]
            expected = specification["value"] * factors[specification["unit"]]
            return float(
                math.isclose(
                    actual,
                    expected,
                    rel_tol=specification.get("relative_tolerance", 0.0),
                    abs_tol=specification.get("absolute_tolerance", 0.0),
                )
            )
        except (ValueError, TypeError, KeyError):
            return 0.0
    if kind == "box":
        try:
            box = json.loads(text)
            if (
                not isinstance(box, list)
                or len(box) != 4
                or any(
                    type(v) not in {float, int} or not math.isfinite(v) or not 0 <= v <= 1
                    for v in box
                )
            ):
                return 0.0
            if box[0] > box[2] or box[1] > box[3]:
                return 0.0
            return float(
                max(abs(a - b) for a, b in zip(box, specification["expected"], strict=True))
                <= specification["max_coordinate_error"]
            )
        except (ValueError, TypeError, KeyError):
            return 0.0
    if kind == "python_function":
        return check_python(text, specification["entry_point"], specification["tests"])["score"]
    if kind == "tool_state":
        if environment is None:
            raise ValueError("tool-state rewards need an executed trajectory")
        checks = [
            environment.state.get(key) == expected
            for key, expected in specification.get("state", {}).items()
        ] + [
            environment.files.get(path) == expected
            for path, expected in specification.get("files", {}).items()
        ]
        if not checks:
            raise ValueError("tool-state verifier needs explicit terminal-state assertions")
        if specification.get("answer"):
            checks.append(bool(reward(text, specification["answer"])))
        return float(all(checks) and environment.errors == 0)
    raise ValueError("unimplemented verifier; never silently substitute arithmetic reward")
