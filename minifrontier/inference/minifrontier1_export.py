"""Portable inference exports and explicit weight-only INT8 reference deployment."""

import copy
import shutil
from importlib.metadata import distribution
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from minifrontier.data import sha256
from minifrontier.data.minifrontier1 import write_json
from minifrontier.models.factory import build_model
from minifrontier.training.runtime import atomic_save


def export_licenses(output):
    """Carry the exact component terms from a checkout or installed wheel."""
    root = Path(__file__).resolve().parents[2]
    required = [
        "LICENSE",
        "NOTICE",
        "THIRD_PARTY_NOTICES.md",
        "LICENSES/Apache-2.0.txt",
        "LICENSES/MIT-DeepSeek.txt",
        "LICENSES/LicenseRef-Kimi-K3.txt",
    ]
    dist = distribution("minifrontier")
    hashes = {}
    for name in required:
        source = root / name
        if not source.is_file():
            matches = [p for p in dist.files or [] if str(p).endswith("/licenses/" + name)]
            if len(matches) != 1:
                raise ValueError(f"export cannot locate the required component license: {name}")
            source = Path(str(dist.locate_file(matches[0])))
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        hashes[name] = sha256(destination)
    return hashes


class Int8Linear(nn.Module):
    """Real INT8 stored weights, FP32 scales; dequantized PyTorch matmul (no speed claim)."""

    qweight: torch.Tensor
    scales: torch.Tensor
    bias: torch.Tensor | None

    def __init__(self, linear, group_size=64):
        super().__init__()
        if group_size not in {64, 128}:
            raise ValueError("INT8 export group size must be 64 or 128")
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.group_size = group_size
        weight = linear.weight.detach().float()
        padding = (-weight.shape[1]) % group_size
        grouped = F.pad(weight, (0, padding)).unflatten(-1, (-1, group_size))
        scales = grouped.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 127
        self.register_buffer("qweight", (grouped / scales).round().clamp(-127, 127).to(torch.int8))
        self.register_buffer("scales", scales)
        self.register_buffer(
            "bias", linear.bias.detach().clone() if linear.bias is not None else None
        )

    def forward(self, x):
        weight = (self.qweight.float() * self.scales).flatten(-2)[:, : self.in_features]
        return F.linear(
            x, weight.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None
        )


def quantize_experts(model, group_size=64):
    replaced = []
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and (".experts." in name or ".shared." in name):
            parent_path, child = name.rsplit(".", 1)
            parent = model.get_submodule(parent_path)
            setattr(parent, child, Int8Linear(module, group_size))
            replaced.append(name)
    return replaced


def export_checkpoint(checkpoint, output, *, dtype="float32", int8=False, group_size=64):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("export is immutable; choose a fresh directory")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["model_name"] != "minifrontier1" or dtype not in {"float32", "bfloat16"}:
        raise ValueError("invalid MF1 export family or dtype")
    source = Path(checkpoint).parent / "tokenizer.json"
    if sha256(source) != saved["tokenizer_sha256"]:
        raise ValueError("checkpoint tokenizer was changed")
    output.mkdir(parents=True)
    licenses = export_licenses(output)
    shutil.copyfile(source, output / "tokenizer.json")
    model = build_model("minifrontier1", saved["config"], phase=saved["phase"])
    model.load_state_dict(saved["model"])
    model.eval()
    modules = quantize_experts(model, group_size) if int8 else []
    state = copy.copy(model.state_dict())
    if dtype == "bfloat16":
        state = {
            name: value.to(torch.bfloat16)
            if value.is_floating_point()
            and not any(
                key in name
                for key in ("norm", "A_log", "dt_bias", "correction_bias", "scales", "eta")
            )
            else value
            for name, value in state.items()
        }
    artifact = {
        key: saved[key]
        for key in (
            "model_name",
            "config",
            "phase",
            "stage",
            "step",
            "tokenizer_sha256",
            "chat_template",
            "run_spec",
        )
    }
    artifact.update(
        format="mf1-export-v1",
        model=state,
        source_checkpoint_sha256=sha256(checkpoint),
        quantization=dict(
            kind="int8-experts-reference" if int8 else "none",
            group_size=group_size,
            modules=modules,
        ),
        export_dtype=dtype,
        capability_qualified=False,
    )
    atomic_save(artifact, output / "model.pt")
    report = dict(
        format="mf1-export-manifest-v1",
        checkpoint_sha256=sha256(output / "model.pt"),
        source_checkpoint_sha256=sha256(checkpoint),
        tokenizer_sha256=sha256(source),
        bytes=(output / "model.pt").stat().st_size,
        dtype=dtype,
        quantization=artifact["quantization"],
        capability_qualified=False,
        deployment_qualified=False,
        license_expression="Apache-2.0 AND MIT AND LicenseRef-Kimi-K3",
        license_files=licenses,
        limitations=[
            "INT8 uses dequantized matmul, no measured speedup",
            "quantization/dtype changes require task and draft re-evaluation",
        ]
        if int8 or dtype != "float32"
        else [],
    )
    write_json(output / "manifest.json", report)
    write_json(output / "config.json", saved["config"])
    write_json(
        output / "processor.json",
        dict(
            version="mf1-native-v1",
            vision=saved["config"]["vision_config"],
            max_media_tokens=saved["config"]["protected_media_tokens"],
        ),
    )
    (output / "MODEL_CARD.md").write_text(
        "# MiniFrontier1.0 local export\n\nThis is an unqualified research artifact, not a released chat model.\n\nSee manifest.json for source/checkpoint/tokenizer hashes, storage format and license scope.\n"
    )
    return report


def load_export(path, device="cpu"):
    from tokenizers import Tokenizer

    path = Path(path)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if sha256(path.parent / "tokenizer.json") != saved["tokenizer_sha256"]:
        raise ValueError("export tokenizer differs from model")
    model = build_model("minifrontier1", saved["config"], phase=saved["phase"])
    quant = saved.get("quantization", {})
    if quant.get("kind") == "int8-experts-reference":
        modules = quantize_experts(model, quant["group_size"])
        if modules != quant["modules"]:
            raise ValueError("quantized export topology differs")
    model.load_state_dict(saved["model"], strict=True)
    return model.to(device).eval(), Tokenizer.from_file(str(path.parent / "tokenizer.json")), saved
