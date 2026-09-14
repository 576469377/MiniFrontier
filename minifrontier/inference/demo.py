# ruff: noqa: RUF001
"""Discover saved runs and serve version-bound, single-device text diagnostics."""

import gc
import json
import math
import os
import pickle
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import torch

from minifrontier.inference.demo_web import page
from minifrontier.inference.runtime import generate_ids, load_checkpoint, respond

PAGE = page("checkpoint")
STAGES = {
    "formal",
    "pretrain",
    "dense_distill",
    "sparse_cpt",
    "sft",
    "dpo",
    "grpo",
    "mopd",
    "opd",
    "rl",
    "teacher",
    "accepted",
    "latest",
    "experiments",
    "history",
}
MODEL_NAMES = {
    "minifrontier11": "MiniFrontier1.1",
    "minifrontier1": "MiniFrontier1.0",
    "miniqwen4": "MiniQwen4",
    "minikimik3": "MiniKimi-K3",
    "minideepseekv4": "MiniDeepSeek-V4",
    "minideepseekv41": "MiniDeepSeek-V4.1",
}
MF1_MODELS = {"minifrontier1", "minifrontier11"}
CHAT_STAGES = {"sft", "dpo", "grpo", "mopd", "opd", "rl", "teacher", "accepted"}


class CheckpointChanged(ValueError):
    """The selected file or its saved metadata changed before loading finished."""


def experiment_directories(root, experiment_roots):
    """Keep demo selection local to its root without changing training manifests."""
    root = Path(root).resolve()
    selected = tuple((root / item).resolve() for item in (experiment_roots or ()))
    if any(not item.is_relative_to(root) for item in selected):
        raise ValueError("--experiment-root must be inside --root")
    return selected


def read_metadata(path):
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("run metadata must be an object")
    return value


def checkpoint_version(path, stat=None):
    """An opaque stat fingerprint; listing never reads multi-GB weight payloads."""
    stat = path.stat() if stat is None else stat
    return ":".join(
        str(value)
        for value in (
            "stat-v1",
            path.name,
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )
    )


def saved_progress(run, state):
    ledger = state.get("token_ledger", state.get("ledger", {}))
    if not isinstance(ledger, dict):
        ledger = {}
    posttrain_unit = (
        ("response_positions" if run.get("phase") == "dpo" else "generated_tokens")
        if run.get("format") == "mf1-posttrain-v1"
        else None
    )
    unit = (
        run.get("unit")
        or posttrain_unit
        or (
            "input_tokens"
            if run.get("input_token_budget")
            else "response_tokens"
            if run.get("response_token_budget")
            else "ce_tokens"
        )
    )
    if not isinstance(unit, str):
        unit = "ce_tokens"
    budget = run.get("token_budget") or run.get(
        {
            "ce_tokens": "ce_token_budget",
            "input_tokens": "input_token_budget",
            "response_tokens": "response_token_budget",
        }.get(unit, "token_budget")
    )
    consumed = ledger.get("phase_tokens", ledger.get(unit))
    return dict(
        ce_tokens=ledger.get("ce_tokens"),
        ce_token_budget=budget if unit == "ce_tokens" else None,
        main_ce_tokens=ledger.get("main_ce_tokens", ledger.get("ce_tokens")),
        phase_tokens=consumed,
        token_budget=budget,
        budget_unit=unit,
    )


def checkpoints(root, stage=None, *, include_experiments=False, experiment_roots=None):
    if stage is not None and (not isinstance(stage, str) or stage not in STAGES):
        raise ValueError("unknown checkpoint stage")
    experiment_view = stage in {"experiments", "history"}
    if experiment_view and not include_experiments:
        raise ValueError("experimental checkpoints require --include-experiments")
    found: dict[str, dict[str, Any]] = {}
    root = Path(root).resolve()
    selected = experiment_directories(root, experiment_roots)
    for run_path in root.rglob("run.json"):
        path = run_path.parent / "status.json"
        try:
            run = read_metadata(run_path)
            state = read_metadata(path) if path.exists() else {}
        except (OSError, ValueError):
            continue  # A live writer may be committing metadata for the next save.
        kind = run.get("kind")
        if not isinstance(kind, str):
            continue
        experimental = kind == "acceptance"
        matching = [item for item in selected if run_path.parent.is_relative_to(item)]
        current = not selected or bool(matching)
        if experiment_view:
            if not experimental or (stage == "experiments") != current:
                continue
        elif stage == "formal":
            if kind != "strategy":
                continue
        elif kind not in {"educational", "strategy"}:
            continue
        mf1_posttrain = run.get("format") == "mf1-posttrain-v1"
        family = run.get("model_name", "minifrontier1" if mf1_posttrain else None)
        if not isinstance(family, str) or family not in MODEL_NAMES:
            continue
        mf1_phase = (
            state.get("phase", run.get("phase"))
            if mf1_posttrain
            else state.get("mf1_phase", run.get("mf1_phase"))
        )
        if not isinstance(mf1_phase, (str, type(None))):
            continue
        if mf1_posttrain and mf1_phase not in {"rl", "teacher", "opd", "dpo"}:
            continue  # Draft checkpoints contain a frozen target, not a trained main model.
        saved_stage = state.get("stage", run.get("stage"))
        if saved_stage is None and mf1_posttrain:
            saved_stage = mf1_phase
        if saved_stage is None and family in MF1_MODELS:
            saved_stage = {
                "sft": "sft",
                "indexer": "dense_distill",
                "p2": "sparse_cpt",
                "p3": "sparse_cpt",
            }.get(mf1_phase or "", "pretrain")
        if not isinstance(saved_stage, (str, type(None))):
            continue
        if (
            stage not in {None, "latest", "formal", "accepted", "experiments", "history"}
            and saved_stage != stage
        ):
            continue
        candidates = [path.parent / name for name in ("model.pt", "checkpoint.pt")]
        try:
            ckpt = max((p for p in candidates if p.is_file()), key=lambda p: p.stat().st_mtime_ns)
            stat = ckpt.stat()
        except (ValueError, OSError):
            continue
        relative = str(path.parent.relative_to(root))
        group = max(matching, key=lambda item: len(item.parts)) if matching else root
        run_label = str(path.parent.relative_to(group))
        if run_label.startswith(family + "/"):
            run_label = run_label[len(family) + 1 :]
        quality_path = path.parent.parent / "quality.json"
        try:
            quality = read_metadata(quality_path) if quality_path.exists() else {}
            reviewed = quality.get("checkpoints", {}).get(saved_stage, {})
            if not isinstance(reviewed, dict):
                reviewed = {}
        except (OSError, ValueError, AttributeError):
            reviewed = {}
        artifact = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        capability = (
            reviewed.get("capability_status", "unassessed")
            if not experimental
            and family not in MF1_MODELS
            and reviewed.get("step") == state.get("step")
            and reviewed.get("artifact") == artifact
            else "unassessed"
        )
        if stage == "accepted" and capability != "passed":
            continue
        found[relative] = dict(
            id=relative,
            name=MODEL_NAMES[family],
            model_name=family,
            run=relative,
            run_label=run_label,
            kind=kind,
            stage=saved_stage,
            mf1_phase=mf1_phase,
            step=state.get("step"),
            state=state.get("state", "saved"),
            metadata_scope="saved_checkpoint",
            artifact=ckpt.name,
            version=checkpoint_version(ckpt, stat),
            path=ckpt,
            modified=stat.st_mtime,
            saved_at=stat.st_mtime,
            **saved_progress(run, state),
            capability_status=capability,
        )
    return dict(sorted(found.items(), key=lambda item: item[1]["modified"], reverse=True))


def demo_device(device="cpu", gpu=None):
    """Resolve a physical index before CUDA initialization, retaining UUID identity."""
    if gpu is None:
        return device, str(device)
    if torch.cuda.is_initialized():
        raise ValueError("--gpu must be selected before CUDA initialization")
    from minifrontier.hardware import query_gpus

    selected = next((item for item in query_gpus() if item.index == gpu), None)
    if selected is None:
        raise ValueError(f"physical GPU {gpu} was not found by nvidia-smi")
    os.environ["CUDA_VISIBLE_DEVICES"] = selected.uuid
    return "cuda:0", f"GPU {gpu} · {selected.name}"


def generation_settings(data):
    if not isinstance(data, dict):
        raise ValueError("请求必须是 JSON 对象")
    if not isinstance(data.get("model"), str) or not data["model"]:
        raise ValueError("请选择检查点")
    if not isinstance(data.get("version"), str) or not data["version"]:
        raise ValueError("缺少检查点版本，请刷新列表后重新选择")
    if not isinstance(data.get("prompt"), str) or not data["prompt"].strip():
        raise ValueError("请输入文本")
    if data.get("media") or data.get("images"):
        raise ValueError("此检查点页面仅支持文本；媒体输入请使用 MF1 多媒体页面")
    stage = data.get("stage", "formal")
    if not isinstance(stage, str) or stage not in STAGES:
        raise ValueError("未知检查点范围")
    mode = data.get("mode", "auto")
    if not isinstance(mode, str) or mode not in {"auto", "chat", "completion"}:
        raise ValueError("输入模式必须是 auto、chat 或 completion")
    count, seed = data.get("max_new_tokens", 64), data.get("seed", 0)
    if type(count) is not int or not 1 <= count <= 256:
        raise ValueError("生成长度必须是 1–256 的整数")
    if type(seed) is not int or not 0 <= seed <= 4294967295:
        raise ValueError("seed 必须是 0–4294967295 的整数")
    temperature, top_p = data.get("temperature", 0), data.get("top_p", 1)
    if (
        type(temperature) not in {int, float}
        or not math.isfinite(temperature)
        or not 0 <= temperature <= 2
    ):
        raise ValueError("temperature 必须是 0–2 的有限数值")
    if type(top_p) not in {int, float} or not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("top_p 必须是大于 0 且不超过 1 的有限数值")
    return dict(
        stage=stage,
        mode=mode,
        max_new_tokens=count,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    )


def verify_version(candidate, expected):
    try:
        current = checkpoint_version(candidate["path"])
    except OSError as error:
        raise CheckpointChanged("所选检查点已更新或不可用，请刷新列表后重新选择") from error
    if current != expected or candidate["version"] != expected:
        raise CheckpointChanged("所选检查点已更新，请刷新列表后重新选择")


def text_response(model, tokenizer, meta, prompt, *, chat, max_new_tokens, temperature, top_p):
    if meta["model_name"] not in MF1_MODELS:
        return respond(
            model,
            tokenizer,
            prompt,
            chat=chat,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
    # MF1 has its own control vocabulary; do not use the source models' chat template.
    from minifrontier.data.minifrontier1 import safe_text
    from minifrontier.inference.minifrontier1 import prepare_prompt

    ids = (
        prepare_prompt(model, tokenizer, prompt)[0]
        if chat
        else torch.tensor([[1, *safe_text(tokenizer, prompt)]])
    )
    result = generate_ids(
        model,
        ids.to(next(model.parameters()).device),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        vocab_size=tokenizer.get_vocab_size(),
    )
    return tokenizer.decode(result[0, ids.shape[1] :].tolist(), skip_special_tokens=True)


def create_server(
    root="outputs",
    host="127.0.0.1",
    port=7860,
    device="cpu",
    gpu=None,
    include_experiments=False,
    experiment_roots=None,
):
    """Build a server without starting its loop or loading a checkpoint."""
    if experiment_roots and not include_experiments:
        raise ValueError("--experiment-root requires --include-experiments")
    experiment_roots = experiment_directories(root, experiment_roots)
    device, device_label = demo_device(device, gpu)
    lock = threading.Lock()
    default_stage = "formal"

    def discover(stage):
        return checkpoints(
            root,
            stage,
            include_experiments=include_experiments,
            experiment_roots=experiment_roots,
        )

    class Handler(BaseHTTPRequestHandler):
        def reply(self, value, status=200):
            raw = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/":
                raw = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            elif url.path == "/api/models":
                stage = parse_qs(url.query).get("stage", [default_stage])[0]
                try:
                    self.reply(
                        [
                            {k: v for k, v in m.items() if k not in {"path", "modified"}}
                            for m in discover(stage).values()
                        ]
                    )
                except ValueError as error:
                    self.reply(dict(error=str(error), code="invalid_request"), 400)
            elif url.path == "/api/info":
                self.reply(
                    dict(
                        device=device_label,
                        include_experiments=include_experiments,
                        default_stage=default_stage,
                        experiments_scoped=bool(experiment_roots),
                        model_families=MODEL_NAMES,
                        modalities=["text"],
                        generation_defaults=dict(temperature=0, top_p=1, seed=0, max_new_tokens=64),
                    )
                )
            else:
                self.reply(dict(error="页面不存在", code="not_found"), 404)

        def do_POST(self):
            if urlparse(self.path).path != "/api/generate":
                self.reply(dict(error="接口不存在", code="not_found"), 404)
                return
            attempted_load = False
            completed = False
            owns_slot = False
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 32768:
                    raise ValueError("请求内容须在 32 KiB 以内")
                data = json.loads(self.rfile.read(length))
                settings = generation_settings(data)
                started = time.perf_counter()
                lock.acquire()
                owns_slot = True
                # Select and validate after waiting for earlier requests.
                candidate = discover(settings["stage"]).get(data["model"])
                if candidate is None:
                    raise CheckpointChanged("所选检查点已不可用，请刷新列表后重新选择")
                verify_version(candidate, data["version"])
                model = None
                try:
                    load_start = time.perf_counter()
                    attempted_load = True
                    try:
                        model, tokenizer, meta = load_checkpoint(candidate["path"], device)
                    except (
                        OSError,
                        ValueError,
                        KeyError,
                        TypeError,
                        RuntimeError,
                        EOFError,
                        pickle.UnpicklingError,
                    ) as error:
                        verify_version(candidate, data["version"])
                        if isinstance(error, torch.cuda.OutOfMemoryError):
                            raise
                        raise RuntimeError(str(error)) from error
                    verify_version(candidate, data["version"])
                    if meta.get("model_name") != candidate["model_name"] or any(
                        candidate[key] is not None and meta.get(key) != candidate[key]
                        for key in ("stage", "step")
                    ):
                        raise CheckpointChanged("检查点与保存记录暂不一致，请刷新列表后重试")
                    load_seconds = time.perf_counter() - load_start
                    chat = settings["mode"] == "chat" or (
                        settings["mode"] == "auto" and meta["stage"] in CHAT_STAGES
                    )
                    generation_start = time.perf_counter()
                    target = torch.device(device)
                    devices = (
                        [target.index if target.index is not None else torch.cuda.current_device()]
                        if target.type == "cuda"
                        else []
                    )
                    with torch.random.fork_rng(devices=devices):
                        torch.random.default_generator.manual_seed(settings["seed"])
                        for index in devices:
                            torch.cuda.default_generators[index].manual_seed(settings["seed"])
                        result = text_response(
                            model,
                            tokenizer,
                            meta,
                            data["prompt"],
                            chat=chat,
                            max_new_tokens=settings["max_new_tokens"],
                            temperature=settings["temperature"],
                            top_p=settings["top_p"],
                        )
                    generation_seconds = time.perf_counter() - generation_start
                    completed = True
                finally:
                    del model
                    if str(device).startswith("cuda"):
                        torch.cuda.empty_cache()
                self.reply(
                    dict(
                        text=result,
                        **meta,
                        device=device_label,
                        model=candidate["id"],
                        run=candidate["run"],
                        artifact=candidate["artifact"],
                        version=candidate["version"],
                        capability_status=candidate["capability_status"],
                        generation_mode="chat" if chat else "completion",
                        max_new_tokens=settings["max_new_tokens"],
                        temperature=settings["temperature"],
                        top_p=settings["top_p"],
                        seed=settings["seed"],
                        load_seconds=load_seconds,
                        generation_seconds=generation_seconds,
                        seconds=time.perf_counter() - started,
                    )
                )
            except CheckpointChanged as error:
                self.reply(dict(error=str(error), code="checkpoint_changed"), 409)
            except (ValueError, KeyError, TypeError, OverflowError, UnicodeError) as error:
                self.reply(dict(error=str(error), code="invalid_request"), 400)
            except torch.cuda.OutOfMemoryError:
                self.reply(
                    dict(
                        error="设备内存不足，请选择 CPU 或有足够空闲显存的设备",
                        code="insufficient_memory",
                    ),
                    503,
                )
            except (RuntimeError, OSError, EOFError, pickle.UnpicklingError) as error:
                self.reply(
                    dict(error=f"无法加载或运行所选检查点：{error}", code="generation_failed"), 422
                )
            finally:
                # Failed call tracebacks can retain tensors through the inner finally.
                # Release their cache after exception handlers have dropped those frames.
                try:
                    if attempted_load and not completed and str(device).startswith("cuda"):
                        gc.collect()
                        torch.cuda.empty_cache()
                finally:
                    if owns_slot:
                        lock.release()

    return ThreadingHTTPServer((host, port), Handler)


def serve(
    root="outputs",
    host="127.0.0.1",
    port=7860,
    device="cpu",
    gpu=None,
    include_experiments=False,
    experiment_roots=None,
):
    server = create_server(
        root=root,
        host=host,
        port=port,
        device=device,
        gpu=gpu,
        include_experiments=include_experiments,
        experiment_roots=experiment_roots,
    )
    print(f"MiniFrontier demo: http://{host}:{server.server_port} · formal", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
