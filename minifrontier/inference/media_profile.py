"""Read the image processor settings bound to an inference artifact."""

import json
from pathlib import Path


def checkpoint_media_profile(saved, path):
    if "inference_media_profile" in saved:
        return dict(saved["inference_media_profile"])
    sidecar = Path(path).parent / "inference_media_profile.json"
    if not sidecar.is_file():
        return {}
    profile = json.loads(sidecar.read_text())
    run = saved.get("run_spec", {})
    bound = (
        run.get("dataset_manifest_sha256")
        or run.get("data_sha256")
        or saved.get("selection", {}).get("data_sha256")
    )
    if bound is None:
        return {}
    if profile.pop("dataset_manifest_sha256") != bound:
        raise ValueError("media profile belongs to a different training corpus")
    return profile
