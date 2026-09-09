"""Release-facing layout, links and current model entry points."""

import json
import re
import tomllib
from pathlib import Path
from urllib.parse import unquote

from minifrontier.catalog import default_manifest_path, inspect_current_models

ROOT = Path(__file__).resolve().parents[1]


def test_public_catalog_and_packaged_configuration_paths():
    assert default_manifest_path() == ROOT / "configs/models.json"
    result = inspect_current_models(default_manifest_path())
    assert set(result) == {"miniqwen4", "minikimik3", "minideepseekv4"}
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert config["project"]["readme"] == "README.md"
    for filenames in config["tool"]["setuptools"]["data-files"].values():
        assert all((ROOT / name).is_file() for name in filenames)


def test_local_documentation_links_resolve():
    documents = [*ROOT.glob("*.md"), *ROOT.joinpath("docs").rglob("*.md")]
    missing = []
    for document in documents:
        for target in re.findall(r"(?<!!)\[[^\]]*\]\(([^)]+)\)", document.read_text()):
            if target.startswith(("https://", "http://", "mailto:", "#")):
                continue
            relative = unquote(target.split("#", 1)[0])
            if not (document.parent / relative).exists():
                missing.append(f"{document.relative_to(ROOT)} -> {relative}")
    assert not missing, "\n".join(missing)


def test_repository_has_only_current_model_packages():
    folders = {
        p.name
        for p in (ROOT / "minifrontier/models").iterdir()
        if p.is_dir() and p.name != "__pycache__"
    }
    assert folders == {"miniqwen4", "minikimik3", "minideepseekv4"}
    # Historical documentation is retained as evidence; executable retired
    # model packages/configurations must still stay out of current entry points.
    assert (ROOT / "docs/training-failure-v1.md").is_file()
    assert all(p.suffix == ".md" for p in (ROOT / "docs/legacy").rglob("*") if p.is_file())
    assert not (ROOT / "configs/current").exists()
    assert set(p.name for p in (ROOT / "configs").iterdir()) == {
        "models.json",
        "strategies",
        "miniqwen4.json",
        "minikimik3.json",
        "minideepseekv4.json",
    }
    audit = json.loads((ROOT / "docs/audits/miniqwen4-acceptance.json").read_text())
    assert audit["formal_training_started"] is False
