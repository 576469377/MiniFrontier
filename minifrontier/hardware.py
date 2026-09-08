from __future__ import annotations

import importlib.util
import json
import subprocess
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class GPUInfo:
    index: int
    uuid: str
    name: str
    total_mib: int
    used_mib: int
    free_mib: int
    utilization_percent: int

    @property
    def free_gib(self) -> float:
        return self.free_mib / 1024


def query_gpus() -> list[GPUInfo]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, timeout=10)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return []
    gpus = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 7:
            continue
        try:
            index, total_mib, used_mib, free_mib, utilization = (
                int(parts[0]),
                int(parts[3]),
                int(parts[4]),
                int(parts[5]),
                int(parts[6]),
            )
        except ValueError:
            # A transient N/A or an unexpected header must not crash a long-lived waiter.
            continue
        uuid = parts[1]
        if not uuid or not uuid.startswith(("GPU-", "MIG-")):
            continue
        gpus.append(
            GPUInfo(
                index=index,
                uuid=uuid,
                name=parts[2],
                total_mib=total_mib,
                used_mib=used_mib,
                free_mib=free_mib,
                utilization_percent=utilization,
            )
        )
    return gpus


def select_free_gpus(
    count: int,
    minimum_free_gib: float = 20.0,
    *,
    name_contains: str | None = None,
) -> list[GPUInfo]:
    """Select sufficiently free devices while retaining their stable UUIDs."""

    normalized_name = name_contains.casefold() if name_contains is not None else None
    candidates = [
        gpu
        for gpu in query_gpus()
        if gpu.free_gib >= minimum_free_gib
        and (normalized_name is None or normalized_name in gpu.name.casefold())
    ]
    candidates.sort(key=lambda gpu: (-gpu.free_mib, gpu.index))
    return sorted(candidates[:count], key=lambda gpu: gpu.index)


def doctor_report() -> dict[str, Any]:
    gpus = query_gpus()
    capabilities: list[dict[str, Any]] = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            major, minor = torch.cuda.get_device_capability(index)
            capabilities.append(
                {
                    "visible_index": index,
                    "name": torch.cuda.get_device_name(index),
                    "compute_capability": f"{major}.{minor}",
                }
            )
    warnings = []
    if capabilities and any(float(item["compute_capability"]) < 9.0 for item in capabilities):
        warnings.append(
            "SM<90: official FlashKDA/FlashQLA kernels are unavailable; this package uses "
            "its PyTorch correctness backends. No official fused-kernel equivalence is claimed."
        )
        warnings.append("SM86 has no native FP8/FP4 Tensor Core path; prefer BF16 on RTX 3090.")
    fla_detected = importlib.util.find_spec("fla") is not None
    if fla_detected:
        warnings.append(
            "fla is installed, but the current source-aligned models do not select it automatically."
        )
    return {
        "python_torch": {"torch": torch.__version__, "cuda_build": torch.version.cuda},
        "cuda_available": torch.cuda.is_available(),
        "visible_devices": capabilities,
        "nvidia_smi": [asdict(gpu) for gpu in gpus],
        "optional_backends": {
            "fla_detected": fla_detected,
            "fla_integrated": True,
            "fla_backends": [],
        },
        "warnings": warnings,
    }


def estimate_optimizer_state_bytes(
    model: nn.Module, optimizer: str, precision: str
) -> dict[str, int]:
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    parameter_bytes = 4 if precision == "fp32" else 2
    # These are persistent-state lower bounds. Activations, allocator fragmentation,
    # temporary attention buffers, and CUDA context are intentionally reported separately.
    model_bytes = total * parameter_bytes
    gradient_bytes = trainable * parameter_bytes
    if optimizer == "adamw":
        optimizer_bytes = trainable * 8
        master_bytes = trainable * (0 if precision == "fp32" else 4)
    else:
        # Muon keeps one momentum tensor; Adam-routed parameters make the true value
        # model-dependent, so use the conservative all-Adam upper bound.
        optimizer_bytes = trainable * 8
        master_bytes = trainable * (0 if precision == "fp32" else 4)
    return {
        "parameters": model_bytes,
        "gradients": gradient_bytes,
        "optimizer_upper_bound": optimizer_bytes,
        "fp32_master_upper_bound": master_bytes,
        "persistent_total_upper_bound": model_bytes
        + gradient_bytes
        + optimizer_bytes
        + master_bytes,
    }


def format_bytes(value: int) -> str:
    return f"{value / 2**30:.2f} GiB"


def print_doctor() -> None:
    print(json.dumps(doctor_report(), ensure_ascii=False, indent=2))
