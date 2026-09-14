# ruff: noqa: RUF001
"""Small local text/image/frame-video demo bound to one explicit exported checkpoint."""

import base64
import hashlib
import io
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import pairwise
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from minifrontier.data import sha256
from minifrontier.data.minifrontier1 import safe_text
from minifrontier.inference.demo_web import page
from minifrontier.inference.runtime import generate_ids, load_checkpoint
from minifrontier.models.minifrontier1.processing import process_frames
from minifrontier.multimodal import move

PAGE = page("mf1")


def load_demo_checkpoint(checkpoint, device):
    """Bind display metadata to a stable file version at load time."""
    path = Path(checkpoint).resolve()
    before = path.stat()
    checksum = sha256(path)
    model, tokenizer, metadata = load_checkpoint(path, device)
    after = path.stat()

    def identity(stat):
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    if identity(before) != identity(after):
        raise ValueError("加载期间检查点发生变化，请使用稳定的导出文件或重新启动")
    if metadata["model_name"] not in {"minifrontier1", "minifrontier11"}:
        raise ValueError("MF1 demo requires an MF1.0 or MF1.1 checkpoint")
    details = dict(
        name=path.name,
        run=path.parent.name,
        path=str(path),
        sha256=checksum,
        loaded_at=time.time(),
        saved_at=after.st_mtime,
        **metadata,
        device=str(device),
        parameters=sum(p.numel() for p in model.parameters() if p.is_floating_point()),
        context_length=model.config.max_position_embeddings,
    )
    return model, tokenizer, details


def prepare_request(payload, model, tokenizer):
    if not isinstance(payload, dict):
        raise ValueError("请求必须为 JSON 对象")
    if not isinstance(payload.get("prompt"), str) or not payload["prompt"].strip():
        raise ValueError("请输入文本问题")
    if payload.get("mode", "direct") not in {"direct", "thinking"}:
        raise ValueError("不支持此回答模式")
    input_mode = payload.get("input_mode", "chat")
    if input_mode not in {"chat", "completion"}:
        raise ValueError("不支持此输入模式")
    if input_mode == "completion" and payload.get("media"):
        raise ValueError("文本续写不接收媒体，请切换到对话模式")
    media = payload.get("media", [])
    if not isinstance(media, list):
        raise ValueError("媒体必须为列表")
    budget = payload.get("max_new_tokens", 64)
    if type(budget) is not int or not 1 <= budget <= 256:
        raise ValueError("生成预算必须为 1-256")
    ids = [1, 4] if input_mode == "chat" else [1]
    plan: list[dict[str, Any]] = []
    spans: list[dict[str, Any]] = []
    decoded_pixels = 0
    for resource in media:
        if not isinstance(resource, dict):
            raise ValueError("每份媒体必须为对象")
        video = resource.get("kind") == "video"
        frames = resource.get("frames", [])
        if (
            resource.get("kind") not in {"image", "video"}
            or not isinstance(frames, list)
            or not 1 <= len(frames) <= 16
            or any(not isinstance(frame, str) or not frame for frame in frames)
            or (not video and len(frames) != 1)
        ):
            raise ValueError("媒体应为图片或至多 16 帧短视频")
        if video:
            timestamps = resource.get("timestamps")
            if (
                len(frames) < 2
                or not isinstance(timestamps, list)
                or len(timestamps) != len(frames)
                or any(
                    type(t) not in {int, float} or not math.isfinite(t) or t < 0 for t in timestamps
                )
                or any(b <= a for a, b in pairwise(timestamps))
            ):
                raise ValueError("视频需要对应每帧的递增时间戳")
        images, hashes = [], []
        for encoded in frames:
            raw = base64.b64decode(encoded.split(",", 1)[-1], validate=True)
            with Image.open(io.BytesIO(raw)) as image:
                decoded_pixels += image.width * image.height
                if decoded_pixels > 32_000_000:
                    raise ValueError("媒体解码像素超过预算，请减少图片或帧尺寸")
                images.append(image.convert("RGB"))
            hashes.append(hashlib.sha256(raw).hexdigest())
        remaining = model.config.protected_media_tokens - sum(s["feature_count"] for s in spans)
        sample = process_frames(
            images,
            max_features=remaining,
            patch_size=model.config.vision_config.patch_size,
            timestamps=resource.get("timestamps") if video else None,
        )
        ids.append(20 if video else 9)
        sample.update(
            batch_index=0,
            start=len(ids),
            resource_kind="video" if video else "image",
            source_hashes=hashes,
        )
        ids.extend([7] * sample["feature_count"] + [21 if video else 10])
        spans.append(sample)
        plan.append(
            dict(
                kind=resource["kind"],
                name=str(resource.get("name", f"媒体 {len(plan) + 1}")),
                timestamps=resource.get("timestamps", []) if video else [],
                frames=len(images),
                width=sample["resized_size"][0],
                height=sample["resized_size"][1],
                tokens=sample["feature_count"],
            )
        )
    ids.extend(safe_text(tokenizer, payload["prompt"]))
    if input_mode == "chat":
        ids.extend([2, 5, 15 if payload.get("mode") == "thinking" else 17])
    if len(ids) + budget > model.config.max_position_embeddings:
        raise ValueError("输入与回答预算超过上下文，请减少媒体或文本")
    request = dict(
        prompt=payload["prompt"],
        input_mode=input_mode,
        mode=payload.get("mode", "direct") if input_mode == "chat" else None,
        max_new_tokens=budget,
        media=payload.get("media", []),
    )
    request_id = hashlib.sha256(
        json.dumps(request, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()
    if payload.get("request_id") not in {None, request_id}:
        raise ValueError("输入已变化，请重新检查输入预算")
    return (
        torch.tensor([ids]),
        spans,
        dict(
            input_tokens=len(ids),
            vision_tokens=sum(s["feature_count"] for s in spans),
            media=plan,
            request_id=request_id,
            max_new_tokens=budget,
            context_length=model.config.max_position_embeddings,
            remaining_tokens=model.config.max_position_embeddings - len(ids) - budget,
        ),
    )


def serve(checkpoint, *, device="cpu", host="127.0.0.1", port=7861, allow_unqualified=False):
    # A self-declared flag alone cannot qualify a public demo.
    qualified = False
    if not allow_unqualified:
        raise ValueError("当前 MF1 尚无能力验收发布包；诊断演示需显式 --allow-unqualified")
    model, tokenizer, checkpoint_info = load_demo_checkpoint(checkpoint, device)
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def reply(self, value, code=200):
            raw = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path == "/":
                raw = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(raw)
            elif self.path == "/api/info":
                self.reply(
                    dict(
                        qualified=qualified,
                        checkpoint_sha256=checkpoint_info["sha256"],
                        checkpoint=checkpoint_info,
                    )
                )
            else:
                self.reply(dict(error="not found"), 404)

        def do_POST(self):
            try:
                if self.path not in {"/api/prepare", "/api/generate"}:
                    self.reply(dict(error="not found"), 404)
                    return
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= 16 * 1024**2:
                    raise ValueError("上传内容超过 16 MiB 限制")
                payload = json.loads(self.rfile.read(size))
                if not isinstance(payload, dict):
                    raise ValueError("请求必须为 JSON 对象")
                with lock:
                    if self.path == "/api/generate" and (
                        payload.get("checkpoint_sha256") != checkpoint_info["sha256"]
                    ):
                        raise ValueError("已载入的检查点版本不同，请重新检查输入预算")
                    ids, spans, plan = prepare_request(payload, model, tokenizer)
                    plan["checkpoint_sha256"] = checkpoint_info["sha256"]
                    if self.path == "/api/prepare":
                        self.reply(plan)
                        return
                    if self.path != "/api/generate":
                        self.reply(dict(error="not found"), 404)
                        return
                    start = time.perf_counter()
                    result = generate_ids(
                        model,
                        ids.to(device),
                        media=move(spans, device),
                        max_new_tokens=plan["max_new_tokens"],
                        temperature=0,
                        top_p=1,
                        vocab_size=tokenizer.get_vocab_size(),
                    )
                    tokens = result[0, ids.shape[1] :].tolist()
                    self.reply(
                        dict(
                            text=tokenizer.decode(tokens, skip_special_tokens=True),
                            generated_tokens=len(tokens),
                            seconds=time.perf_counter() - start,
                            completed_at=time.time(),
                            finish_reason="eos" if tokens and tokens[-1] == 2 else "length",
                            request_id=plan["request_id"],
                            checkpoint=checkpoint_info,
                            plan=plan,
                            request=dict(
                                prompt=payload["prompt"],
                                input_mode=payload.get("input_mode", "chat"),
                                mode=(
                                    payload.get("mode", "direct")
                                    if payload.get("input_mode", "chat") == "chat"
                                    else None
                                ),
                                max_new_tokens=plan["max_new_tokens"],
                                temperature=0,
                                top_p=1,
                            ),
                        )
                    )
            except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
                self.reply(dict(error=str(error)), 400)

    print(f"MiniFrontier1 diagnostic demo: http://{host}:{port}", flush=True)
    ThreadingHTTPServer((host, port), Handler).serve_forever()
