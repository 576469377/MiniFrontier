from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from minifrontier import hardware
from minifrontier.hardware import GPUInfo

ROOT = Path(__file__).resolve().parents[1]


def _load_wait_module() -> ModuleType:
    path = ROOT / "scripts" / "wait_for_gpus.py"
    spec = importlib.util.spec_from_file_location("minifrontier_wait_for_gpus", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _gpu(
    index: int,
    free_mib: int,
    *,
    uuid: str | None = None,
    used_mib: int = 1024,
    name: str | None = None,
    utilization_percent: int = 0,
) -> GPUInfo:
    return GPUInfo(
        index=index,
        uuid=uuid or f"GPU-mock-{index}",
        name=name or f"mock-gpu-{index}",
        total_mib=24576,
        used_mib=used_mib,
        free_mib=free_mib,
        utilization_percent=utilization_percent,
    )


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--count", "0", "--", "worker"], "count must be positive"),
        (["--count", "-1", "--", "worker"], "count must be positive"),
        (["--count", "1.5", "--", "worker"], "invalid int value"),
        (["--poll-seconds", "0", "--", "worker"], "poll-seconds must be positive"),
        (["--poll-seconds", "-1", "--", "worker"], "poll-seconds must be positive"),
        (["--poll-seconds", "1.5", "--", "worker"], "invalid int value"),
        (["--min-free-gib", "0", "--", "worker"], "min-free-gib must be"),
        (["--min-free-gib", "-1", "--", "worker"], "min-free-gib must be"),
        (["--min-free-gib", "nan", "--", "worker"], "min-free-gib must be"),
        (["--min-free-gib", "inf", "--", "worker"], "min-free-gib must be"),
        (["--min-free-gib=-inf", "--", "worker"], "min-free-gib must be"),
        (["--name-contains", "  ", "--", "worker"], "name-contains must be non-empty"),
        ([], "provide a command after --"),
        (["--"], "provide a command after --"),
    ],
)
def test_wait_cli_rejects_invalid_arguments(arguments, message, capsys) -> None:  # type: ignore[no-untyped-def]
    module = _load_wait_module()

    with pytest.raises(SystemExit) as raised:
        module.main(arguments)

    assert raised.value.code == 2
    assert message in capsys.readouterr().err


def test_waits_until_enough_gpus_then_sets_child_environment_and_returns_exit_code(
    monkeypatch, capsys
) -> None:
    module = _load_wait_module()
    selections = [[], [_gpu(2, 23 * 1024), _gpu(5, 22 * 1024)]]
    select_calls: list[tuple[int, float, str | None]] = []
    sleeps: list[int] = []
    child_calls: list[tuple[tuple[str, ...], dict[str, str]]] = []

    def select(count: int, minimum: float, *, name_contains: str | None = None):  # type: ignore[no-untyped-def]
        select_calls.append((count, minimum, name_contains))
        return selections.pop(0)

    monkeypatch.setattr(module, "select_free_gpus", select)
    monkeypatch.setattr(module, "query_gpus", lambda: [_gpu(7, 3072)])
    monkeypatch.setattr(module.time, "sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "old-value")
    monkeypatch.setenv("WAIT_TEST_MARKER", "preserved")

    def child(command, *, env):  # type: ignore[no-untyped-def]
        child_calls.append((tuple(command), env))
        return 17

    monkeypatch.setattr(module.subprocess, "call", child)

    with pytest.raises(SystemExit) as raised:
        module.main(
            [
                "--count",
                "2",
                "--min-free-gib",
                "21",
                "--poll-seconds",
                "3",
                "--name-contains",
                " RTX 3090 ",
                "--",
                "worker",
                "--flag",
            ]
        )

    assert raised.value.code == 17
    assert select_calls == [(2, 21.0, "RTX 3090"), (2, 21.0, "RTX 3090")]
    assert sleeps == [3]
    assert len(child_calls) == 1
    command, environment = child_calls[0]
    assert command == ("worker", "--flag")
    assert environment["CUDA_VISIBLE_DEVICES"] == "GPU-mock-2,GPU-mock-5"
    assert environment["WAIT_TEST_MARKER"] == "preserved"
    assert module.os.environ["CUDA_VISIBLE_DEVICES"] == "old-value"
    output = capsys.readouterr().out
    assert "waiting for 2 GPUs with 21.0 GiB free (gpu7:3.0GiB)" in output
    assert "selected physical GPUs: gpu2=GPU-mock-2" in output
    assert "gpu5=GPU-mock-5" in output


@pytest.mark.parametrize("separator", [[], ["--"]])
def test_immediate_selection_accepts_existing_command_forms_and_skips_wait(
    separator, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    module = _load_wait_module()
    monkeypatch.setattr(
        module,
        "select_free_gpus",
        lambda count, minimum, *, name_contains=None: [_gpu(4, 23 * 1024)],
    )
    monkeypatch.setattr(
        module,
        "query_gpus",
        lambda: pytest.fail("status query is unnecessary after immediate selection"),
    )
    monkeypatch.setattr(
        module.time,
        "sleep",
        lambda seconds: pytest.fail("immediate selection must not sleep"),
    )
    child_calls: list[tuple[list[str], str]] = []

    def child(command, *, env):  # type: ignore[no-untyped-def]
        child_calls.append((command, env["CUDA_VISIBLE_DEVICES"]))
        return 0

    monkeypatch.setattr(module.subprocess, "call", child)

    with pytest.raises(SystemExit) as raised:
        module.main(["--count", "1", *separator, "worker", "argument"])

    assert raised.value.code == 0
    assert child_calls == [(["worker", "argument"], "GPU-mock-4")]


def test_query_gpus_parses_nvidia_smi_without_using_real_hardware(monkeypatch) -> None:
    output = "\n".join(
        [
            "2, GPU-two, NVIDIA GeForce RTX 3090, 24576, 2048, 22528, 3",
            "malformed row",
            "N/A, GPU-transient, transient-gpu, 24576, N/A, N/A, N/A",
            "1, missing-prefix, NVIDIA GeForce RTX 3090, 24576, 1024, 23552, 17",
            "0, GPU-zero, NVIDIA GeForce RTX 3090, 24576, 1024, 23552, 17",
        ]
    )

    def check_output(command, *, text, timeout):  # type: ignore[no-untyped-def]
        assert command == [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        assert text is True
        assert timeout == 10
        return output

    monkeypatch.setattr(hardware.subprocess, "check_output", check_output)

    assert hardware.query_gpus() == [
        _gpu(
            2,
            22528,
            uuid="GPU-two",
            used_mib=2048,
            name="NVIDIA GeForce RTX 3090",
            utilization_percent=3,
        ),
        _gpu(
            0,
            23552,
            uuid="GPU-zero",
            name="NVIDIA GeForce RTX 3090",
            utilization_percent=17,
        ),
    ]


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("nvidia-smi"),
        subprocess.CalledProcessError(1, ["nvidia-smi"]),
        subprocess.TimeoutExpired(["nvidia-smi"], 10),
    ],
)
def test_query_gpus_returns_empty_when_nvidia_smi_is_unavailable(monkeypatch, error) -> None:  # type: ignore[no-untyped-def]
    def fail(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise error

    monkeypatch.setattr(hardware.subprocess, "check_output", fail)

    assert hardware.query_gpus() == []


def test_select_free_gpus_prefers_most_free_then_returns_index_order(monkeypatch) -> None:
    monkeypatch.setattr(
        hardware,
        "query_gpus",
        lambda: [
            _gpu(0, 22 * 1024),
            _gpu(3, 24 * 1024),
            _gpu(1, 24 * 1024),
            _gpu(2, 20 * 1024),
        ],
    )

    assert [gpu.index for gpu in hardware.select_free_gpus(2, 21.0)] == [1, 3]
    assert [gpu.index for gpu in hardware.select_free_gpus(4, 21.0)] == [0, 1, 3]


def test_select_free_gpus_filters_names_case_insensitively(monkeypatch) -> None:
    monkeypatch.setattr(
        hardware,
        "query_gpus",
        lambda: [
            _gpu(0, 23 * 1024, name="NVIDIA A100-SXM4-80GB"),
            _gpu(1, 22 * 1024, name="NVIDIA GeForce RTX 3090"),
            _gpu(2, 21 * 1024, name="NVIDIA GeForce RTX 3090"),
        ],
    )

    selected = hardware.select_free_gpus(2, 21.0, name_contains="rtx 3090")

    assert [(gpu.index, gpu.uuid) for gpu in selected] == [
        (1, "GPU-mock-1"),
        (2, "GPU-mock-2"),
    ]
