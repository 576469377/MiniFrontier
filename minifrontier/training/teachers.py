"""Audited teacher slots, loading at most one frozen teacher on the training device."""

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from minifrontier.data import sha256
from minifrontier.inference import load_checkpoint


def model_state_hash(path):
    """Ignore optimizer/RNG/file metadata when detecting cloned teachers."""
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    digest = hashlib.sha256()
    for name, tensor in sorted(checkpoint["model"].items()):
        digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
        raw = tensor.contiguous().reshape(-1).view(torch.uint8)
        for chunk in raw.split(8 * 1024**2):
            digest.update(chunk.numpy().tobytes())
    return digest.hexdigest()


class TeacherRegistry(Mapping):
    def __init__(self, path, tokenizer, device, *, require_complete=False):
        payload = json.loads(Path(path).read_text())
        self.entries = payload.get("teachers", payload)
        self.tokenizer, self.device = tokenizer, device
        self._loaded = None
        self._path = None
        paths = {self.path(key) for key in self.entries}
        if len(paths) < 2:
            raise ValueError("distillation requires distinct teacher checkpoints")
        for key in self.entries:
            if not Path(self.path(key)).is_file():
                raise FileNotFoundError(self.path(key))
        if require_complete:
            family = payload.get("model")
            slots = 9 if family == "minikimik3" else 12 if family == "minideepseekv4" else 0
            if not slots or len(self.entries) != slots or len(paths) != slots:
                raise ValueError("formal registry must contain every distinct domain/mode teacher")
            domains, modes = (
                (("general", "tools", "code"), ("low", "high", "max"))
                if family == "minikimik3"
                else (
                    ("math", "code", "general_tools", "vision"),
                    ("non-thinking", "thinking-high", "thinking-max"),
                )
            )
            if set(self.entries) != {f"{domain}:{mode}" for domain in domains for mode in modes}:
                raise ValueError("teacher registry does not cover the specified domain/mode grid")
            hashes = set()
            states = set()
            for key, entry in self.entries.items():
                if not isinstance(entry, dict):
                    raise ValueError("formal teacher needs checkpoint and heldout evidence")
                digest = sha256(self.path(key))
                if digest != entry.get("sha256") or digest in hashes:
                    raise ValueError("teacher hash mismatch or duplicated checkpoint")
                hashes.add(digest)
                state = model_state_hash(self.path(key))
                if state in states or entry.get("model_state_sha256") != state:
                    raise ValueError("teacher weights are duplicated or lack a matching state hash")
                states.add(state)
                report = json.loads(Path(entry["evaluation"]).read_text())
                if report.get("checkpoint_sha256") != digest or report.get("slot") != key:
                    raise ValueError(
                        "teacher heldout evidence does not identify this slot/checkpoint"
                    )
                if report.get("unique_prompt_groups", 0) < 2000 or not report.get(
                    "beats_common_sft", False
                ):
                    raise ValueError("teacher has not passed its domain heldout gate")
                if report.get("teacher_score", float("-inf")) <= report.get(
                    "common_sft_score", float("inf")
                ):
                    raise ValueError("teacher does not improve on the common SFT checkpoint")

    def path(self, key):
        entry = self.entries[key]
        return str(Path(entry["checkpoint"] if isinstance(entry, dict) else entry).resolve())

    def __len__(self):
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def __getitem__(self, key):
        path = self.path(key)
        if path != self._path:
            # Drop the previous GPU model before allocating its successor.
            self._loaded, self._path = None, None
            model, tokenizer, _ = load_checkpoint(path, self.device)
            if tokenizer.to_str() != self.tokenizer.to_str():
                raise ValueError("teacher and student tokenizers must match exactly")
            for p in model.parameters():
                if p.is_floating_point():
                    p.requires_grad_(False)
            self._loaded, self._path = model, path
        return self._loaded

    def unload(self):
        self._loaded, self._path = None, None
