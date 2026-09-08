"""Bind runs to the actual source tree, including local changes."""

import hashlib
import subprocess
from pathlib import Path


def source_identity():
    root = Path(__file__).resolve().parents[1]
    paths = sorted(
        path
        for folder in ("minifrontier", "scripts", "configs")
        for path in (root / folder).rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode() + b"\0")
        digest.update(path.read_bytes())
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True))
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit, dirty = None, True
    return dict(commit=commit, dirty=dirty, implementation_sha256=digest.hexdigest())
