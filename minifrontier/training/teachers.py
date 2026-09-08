"""Audited teacher slots, loading at most one frozen teacher on the training device."""

import json
from collections.abc import Mapping
from pathlib import Path

from minifrontier.data import sha256
from minifrontier.inference import load_checkpoint


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
            hashes = set()
            for key, entry in self.entries.items():
                if not isinstance(entry, dict):
                    raise ValueError("formal teacher needs checkpoint and heldout evidence")
                digest = sha256(self.path(key))
                if digest != entry.get("sha256") or digest in hashes:
                    raise ValueError("teacher hash mismatch or duplicated checkpoint")
                hashes.add(digest)
                report = json.loads(Path(entry["evaluation"]).read_text())
                if report.get("checkpoint_sha256") != digest or report.get("slot") != key:
                    raise ValueError(
                        "teacher heldout evidence does not identify this slot/checkpoint"
                    )
                if report.get("unique_prompt_groups", 0) < 2000 or not report.get(
                    "beats_common_sft", False
                ):
                    raise ValueError("teacher has not passed its domain heldout gate")

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
