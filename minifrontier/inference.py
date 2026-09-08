# ruff: noqa: RUF001
"""Portable checkpoint generation and a local browser demo for all three models."""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import torch
from tokenizers import Tokenizer

from minifrontier.data import chat_tokens, sha256
from minifrontier.models.factory import build_model, configure_posttraining


@torch.no_grad()
def generate_ids(
    model,
    input_ids,
    *,
    max_new_tokens=64,
    temperature=0.8,
    top_p=0.9,
    vocab_size=None,
    eos_token_id=2,
    return_behavior=False,
    media=None,
    use_cache=True,
):
    if max_new_tokens < 1 or temperature < 0 or not 0 < top_p <= 1:
        raise ValueError("invalid generation settings")
    if return_behavior and (temperature != 1 or top_p != 1):
        raise ValueError("behavior traces currently require temperature=1 and top_p=1")
    from minifrontier.training.distributions import action_logits, forbidden_actions

    behavior, action_masks = [], []
    model.eval()
    limit = getattr(
        model.config, "max_position_embeddings", getattr(model.config, "max_seq_len", 0)
    )
    if input_ids.shape[1] + max_new_tokens > limit:
        raise ValueError(f"prompt plus response exceeds context length {limit}")
    finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
    all_ids = input_ids
    cache: Any = None
    if use_cache and model.__class__.__name__ == "MiniQwen4ForCausalLM":
        from minifrontier.models.miniqwen4 import MiniQwen4Cache

        cache = MiniQwen4Cache()
    elif use_cache and model.__class__.__name__ == "MiniKimiK3ForCausalLM":
        from minifrontier.models.minikimik3 import MiniKimiK3Cache

        cache = MiniKimiK3Cache()
    elif use_cache and model.__class__.__name__ == "MiniDeepSeekV4ForCausalLM":
        from minifrontier.models.minideepseekv4 import MiniDeepSeekV4Cache

        cache = MiniDeepSeekV4Cache()
    current = all_ids
    for _ in range(max_new_tokens):
        with torch.autocast(input_ids.device.type, dtype=torch.bfloat16, enabled=input_ids.is_cuda):
            extras = {"media": media} if media and (cache is None or cache.length == 0) else {}
            out = (
                model(current, cache=cache, **extras)
                if cache is not None
                else model(all_ids, **extras)
            )
        logits = action_logits(out.logits[:, -1], vocab_size, forbidden_actions(model))
        if temperature == 0:
            token = logits.argmax(-1, keepdim=True)
        else:
            sorted_logits, order = (logits / temperature).sort(descending=True)
            cumulative = sorted_logits.softmax(-1).cumsum(-1)
            remove = cumulative > top_p
            remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0] = False
            probs = sorted_logits.masked_fill(remove, float("-inf")).softmax(-1)
            token = order.gather(-1, torch.multinomial(probs, 1))
        token = torch.where(finished[:, None], eos_token_id, token)
        if return_behavior:
            logp = logits.log_softmax(-1).gather(-1, token)[:, 0]
            behavior.append(logp.masked_fill(finished, 0))
            action_masks.append(~finished)
        finished |= token[:, 0] == eos_token_id
        all_ids = torch.cat((all_ids, token), dim=1)
        current = token
        if finished.all():
            break
    if return_behavior:
        return all_ids, torch.stack(behavior, 1), torch.stack(action_masks, 1)
    return all_ids


def load_checkpoint(path, device="cpu"):
    path = Path(path).resolve()
    saved = torch.load(path, map_location="cpu", weights_only=True)
    tokenizer_path = path.parent / "tokenizer.json"
    if sha256(tokenizer_path) != saved["tokenizer_sha256"]:
        raise ValueError("tokenizer does not match checkpoint")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    model = build_model(saved["model_name"], saved["config"], phase=saved["phase"])
    model.load_state_dict(saved["model"], strict=True)
    configure_posttraining(model)
    return (
        model.to(device).eval(),
        tokenizer,
        {key: saved[key] for key in ("model_name", "stage", "step", "phase")},
    )


def respond(model, tokenizer, prompt, *, chat=True, max_new_tokens=64, temperature=0.8, top_p=0.9):
    ids = (
        chat_tokens([{"role": "user", "content": prompt}], tokenizer, generation_prompt=True)[0]
        if chat
        else [1, *tokenizer.encode(prompt).ids]
    )
    inputs = torch.tensor([ids], device=next(model.parameters()).device)
    result = generate_ids(
        model,
        inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        vocab_size=tokenizer.get_vocab_size(),
    )
    return tokenizer.decode(result[0, len(ids) :].tolist(), skip_special_tokens=True)


PAGE = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MiniFrontier · 本地模型实验室</title><style>
body{font:16px system-ui,sans-serif;background:#101827;color:#edf2fa;margin:0}main{max-width:850px;margin:6vh auto;padding:24px}
h1{font-size:36px;margin-bottom:10px}p{color:#b1bfd2;line-height:1.7}select,textarea,button,input{font:inherit;border-radius:9px;padding:12px;border:1px solid #42526a;background:#1b2940;color:#edf2fa}
textarea{box-sizing:border-box;width:100%;height:150px;resize:vertical;margin:18px 0}button{background:#307bdf;cursor:pointer}button:disabled{opacity:.5}pre{white-space:pre-wrap;line-height:1.8;background:#17243a;border-radius:12px;padding:20px;min-height:120px}label{margin-right:12px}small{color:#93a7c0}a{color:#81b4fb}</style>
<main><small>MiniFrontier / 单卡可运行的旗舰架构学习实验</small><h1>和你训练的模型对话</h1>
<p>选择检查点观察生成效果。训练步骤完成不代表已经学会对话；请留意每个模型的对话验收状态。默认展示 SFT，可切换 DPO 对比。</p>
<select id="stage" onchange="refresh()"><option value="sft">SFT</option><option value="dpo">DPO</option><option value="pretrain">预训练</option><option value="latest">最新阶段</option></select>
<select id="model"></select> <button onclick="refresh()">刷新检查点</button><p id="meta"></p>
<textarea id="prompt" placeholder="输入问题或想续写的文本">请用简单的话解释什么是大语言模型。</textarea>
<label>生成长度 <input id="length" type="number" min="1" max="256" value="64" style="width:70px"></label>
<button id="send" onclick="send()">生成</button><pre id="answer">模型的回答会显示在这里。</pre><small id="status"></small>
<script>
async function refresh(){const r=await fetch('/api/models?stage='+document.querySelector('#stage').value);const data=await r.json();const select=document.querySelector('#model');select.replaceChildren();const labels={failed:'未通过对话验收',unassessed:'尚未完成对话验收',passed:'已通过对话验收'};for(const m of data){const o=document.createElement('option');o.value=m.id;o.textContent=m.name+' · '+m.stage+' · step '+m.step+' · '+(labels[m.capability_status]||labels.unassessed);select.appendChild(o)}document.querySelector('#meta').textContent=data.length?'已找到 '+data.length+' 个检查点。':'此阶段尚无检查点，可切换阶段。'}
async function send(){const button=document.querySelector('#send');button.disabled=true;document.querySelector('#status').textContent='正在加载检查点并生成…';try{const r=await fetch('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:document.querySelector('#model').value,stage:document.querySelector('#stage').value,prompt:document.querySelector('#prompt').value,max_new_tokens:Number(document.querySelector('#length').value)})});const data=await r.json();if(!r.ok)throw Error(data.error);document.querySelector('#answer').textContent=data.text;document.querySelector('#status').textContent=data.model_name+' · '+data.stage+' · step '+data.step}catch(e){document.querySelector('#status').textContent=e.message}finally{button.disabled=false}}refresh();</script></main></html>"""


def checkpoints(root, stage=None):
    if stage not in {None, "pretrain", "dense_distill", "sparse_cpt", "sft", "dpo", "grpo", "mopd"}:
        raise ValueError("unknown checkpoint stage")
    names = {
        "miniqwen4": "MiniQwen4",
        "minikimik3": "MiniKimi-K3",
        "minideepseekv4": "MiniDeepSeek-V4",
    }
    found: dict[str, dict[str, Any]] = {}
    for path in Path(root).glob("*/**/status.json"):
        if not (path.parent / "run.json").exists():
            continue
        run = json.loads((path.parent / "run.json").read_text())
        if run.get("kind") != "educational":
            continue
        try:
            state = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue  # A live writer may be atomically committing the next status.
        if stage is not None and state["stage"] != stage:
            continue

        ckpt = path.parent / "model.pt"
        if not ckpt.exists():
            ckpt = path.parent / "checkpoint.pt"
        if not ckpt.exists():
            continue
        key = run["model_name"]
        modified = ckpt.stat().st_mtime
        quality_path = path.parent.parent / "quality.json"
        quality = json.loads(quality_path.read_text()) if quality_path.exists() else {}
        reviewed = quality.get("checkpoints", {}).get(state["stage"], {})
        # A review of another saved step must not qualify a newer checkpoint.
        artifact = dict(size=ckpt.stat().st_size, mtime_ns=ckpt.stat().st_mtime_ns)
        capability = (
            reviewed.get("capability_status", "unassessed")
            if reviewed.get("step") == state["step"] and reviewed.get("artifact") == artifact
            else "unassessed"
        )
        if key not in found or modified > found[key]["modified"]:
            found[key] = dict(
                id=key,
                name=names[key],
                stage=state["stage"],
                step=state["step"],
                path=ckpt,
                modified=modified,
                capability_status=capability,
            )
    return found


def serve(root="outputs", host="127.0.0.1", port=7860, device="cpu"):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def reply(self, value, status=200):
            raw = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/":
                raw = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(raw)
            elif url.path == "/api/models":
                stage = parse_qs(url.query).get("stage", ["sft"])[0]
                if stage not in {"sft", "dpo", "pretrain", "latest"}:
                    self.reply({"error": "unknown checkpoint stage"}, 400)
                    return
                self.reply(
                    [
                        {k: v for k, v in m.items() if k not in {"path", "modified"}}
                        for m in checkpoints(root, None if stage == "latest" else stage).values()
                    ]
                )
            else:
                self.reply({"error": "not found"}, 404)

        def do_POST(self):
            if self.path != "/api/generate":
                self.reply({"error": "not found"}, 404)
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 32768:
                    raise ValueError("请求长度无效")
                data = json.loads(self.rfile.read(length))
                stage = data.get("stage", "sft")
                candidate = checkpoints(root, None if stage == "latest" else stage)[data["model"]]
                count = int(data.get("max_new_tokens", 64))
                if not 1 <= count <= 256:
                    raise ValueError("生成长度必须在 1–256 之间")
                if not isinstance(data["prompt"], str) or not data["prompt"].strip():
                    raise ValueError("请输入文本")
                with lock:
                    model, tokenizer, meta = load_checkpoint(candidate["path"], device)
                    result = respond(
                        model,
                        tokenizer,
                        data["prompt"],
                        chat=meta["stage"] in {"sft", "dpo", "grpo", "mopd"},
                        max_new_tokens=count,
                    )
                    del model
                    if str(device).startswith("cuda"):
                        torch.cuda.empty_cache()
                self.reply(dict(text=result, **meta))
            except (ValueError, KeyError, RuntimeError, OSError) as error:
                self.reply({"error": str(error)}, 400)

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"MiniFrontier demo: http://{host}:{port}", flush=True)
    server.serve_forever()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--device", default="cpu")
    p.add_argument("--completion", action="store_true")
    args = p.parse_args(argv)
    model, tokenizer, _meta = load_checkpoint(args.checkpoint, args.device)
    print(
        respond(
            model,
            tokenizer,
            args.prompt,
            chat=not args.completion,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    )


if __name__ == "__main__":
    main()
