"""Bind runs to the actual source tree, including local changes."""

import hashlib
import subprocess
from pathlib import Path


def checkout_root():
    root = Path(__file__).resolve().parents[1]
    if not (root / "pyproject.toml").is_file() or not (root / ".git").exists():
        return None
    return root


def require_source_checkout():
    root = checkout_root()
    if root is None:
        raise ValueError(
            "Formal strategy training in v0.1.0 requires a Git checkout installed with "
            "`uv sync` or `pip install -e .`. The wheel supports quickstart, models, "
            "explicit-config acceptance training and CLI inference; it does not bundle "
            "the strategy source documents or Git provenance."
        )
    return root


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
        if checkout_root() is None:
            raise FileNotFoundError("not a source checkout")
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True))
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit, dirty = None, True
    return dict(commit=commit, dirty=dirty, implementation_sha256=digest.hexdigest())
