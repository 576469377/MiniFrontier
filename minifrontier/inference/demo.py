# ruff: noqa: RUF001
"""Local browser experiment selection and serving for the source model baselines."""

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import torch

from minifrontier.inference.runtime import load_checkpoint, respond

PAGE = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MiniFrontier · 本地模型实验室</title><style>
body{font:16px system-ui,sans-serif;background:#101827;color:#edf2fa;margin:0}main{max-width:950px;margin:5vh auto;padding:24px}
h1{font-size:36px;margin-bottom:10px}p{color:#b1bfd2;line-height:1.7}select,textarea,button,input{font:inherit;border-radius:9px;padding:12px;border:1px solid #42526a;background:#1b2940;color:#edf2fa;max-width:100%}
textarea{box-sizing:border-box;width:100%;height:150px;resize:vertical;margin:18px 0}button{background:#307bdf;cursor:pointer}button:disabled{opacity:.5}pre{white-space:pre-wrap;line-height:1.8;background:#17243a;border-radius:12px;padding:20px;min-height:120px}label{display:inline-block;margin:8px 12px 8px 0}small{color:#93a7c0}#model{width:100%;margin-top:12px}#meta{overflow-wrap:anywhere}</style>
<main><small>MiniFrontier / 单卡模型实验室 · <span id="device">正在读取设备…</span></small><h1>试试训练中的模型</h1>
<p>每个试验只展示最新保存的一份权重，刷新即可更新。实验结束只代表该次预算跑完，不代表完整预训练完成或具备对话能力。</p>
<select id="stage" onchange="refresh()"><option value="accepted">通过能力验收</option><option value="sft">SFT 对照</option><option value="dpo">DPO</option><option value="pretrain">预训练</option><option value="latest">最新阶段</option></select>
<select id="family" aria-label="筛选模型" onchange="render()"><option value="all">全部模型</option><option value="minikimik3">MiniKimi-K3</option><option value="miniqwen4">MiniQwen4</option><option value="minideepseekv4">MiniDeepSeek-V4</option></select>
<button onclick="refresh()">刷新检查点</button><p id="scope"></p><select id="model" aria-label="选择检查点" onchange="describe()"></select><p id="meta"></p>
<textarea id="prompt" placeholder="输入问题或想续写的文本">人工智能是一种</textarea>
<label>输入模式 <select id="mode"><option value="auto">自动（预训练续写 / 后训练对话）</option><option value="completion">文本续写</option><option value="chat">对话模板</option></select></label>
<label>生成长度 <input id="length" type="number" min="1" max="256" value="64" style="width:70px"></label>
<button id="send" onclick="send()" disabled>生成</button><pre id="answer">模型的输出会显示在这里。</pre><small id="status"></small>
<script>
let models=[],scoped=false,busy=false,refreshId=0;
const labels={failed:'未通过能力验收',unassessed:'尚未完成能力验收',passed:'已通过能力验收'};
const stages={pretrain:'预训练',dense_distill:'稠密蒸馏',sparse_cpt:'稀疏续训',sft:'SFT',dpo:'DPO',grpo:'GRPO',mopd:'MOPD',opd:'OPD'};
const states={running:'记录状态：训练中',failed:'记录状态：失败',stopped:'记录状态：已停止',saved:'已保存'};
const el=id=>document.querySelector('#'+id);
const tokens=n=>n>=1e9?(n/1e9).toFixed(2)+'B':n>=1e6?(n/1e6).toFixed(2)+'M':Number(n).toLocaleString();
function describe(){
  const m=models.find(m=>m.id===el('model').value);
  el('send').disabled=busy||!m;
  if(!m){el('meta').textContent='该范围内暂无已保存的检查点；运行中的试验需等首次保存后才能选择。';return}
  const state=m.state==='complete'?(m.kind==='acceptance'?'本次实验已结束':'本次训练已结束'):(states[m.state]||m.state);
  let detail=m.run+' · '+m.artifact+' · '+(stages[m.stage]||m.stage)+' · '+state;
  if(m.ce_tokens!=null){detail+=' · 已保存进度 '+tokens(m.ce_tokens)+(m.ce_token_budget?' / '+tokens(m.ce_token_budget):'')+' CE tokens'}
  if(m.saved_at)detail+=' · 保存于 '+new Date(m.saved_at*1000).toLocaleString();
  el('meta').textContent=detail+' · '+(labels[m.capability_status]||labels.unassessed);
}
function render(){
  const selected=el('model').value,family=el('family').value;
  const visible=models.filter(m=>family==='all'||m.model_name===family);
  el('model').replaceChildren();
  for(const m of visible){const o=document.createElement('option');o.value=m.id;o.textContent=m.name+' · '+m.run_label+' · '+(stages[m.stage]||m.stage)+' · step '+(m.step??'待载入确认');el('model').appendChild(o)}
  if(visible.some(m=>m.id===selected))el('model').value=selected;
  const stage=el('stage').value;
  const note=stage==='history'?'历史实验与工程验证，用于复盘和对照。':stage==='experiments'?(scoped?'本轮实验；旧验收、诊断和 quickstart 请切换到历史入口。':'实验检查点，均未经能力验收。'):'按训练阶段查看检查点。';
  el('scope').textContent=note+' 当前显示 '+visible.length+' / '+models.length+' 个试验。';
  describe();
}
async function refresh(){
  const requestId=++refreshId;el('send').disabled=true;
  try{
    const r=await fetch('/api/models?stage='+el('stage').value);const data=await r.json();
    if(requestId!==refreshId)return;
    if(!r.ok)throw Error(data.error);
    models=data;render();
  }catch(e){if(requestId!==refreshId)return;models=[];render();el('status').textContent=e.message}
}
async function send(){
  busy=true;el('send').disabled=true;el('status').textContent='正在加载检查点并生成…';
  try{
    const r=await fetch('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:el('model').value,stage:el('stage').value,mode:el('mode').value,prompt:el('prompt').value,max_new_tokens:Number(el('length').value)})});
    const data=await r.json();if(!r.ok)throw Error(data.error);
    el('answer').textContent=data.text||'（模型立即结束，未生成可显示文本）';
    el('status').textContent=data.device+' · '+data.run+' · '+(stages[data.stage]||data.stage)+' · step '+data.step+' · '+(data.generation_mode==='chat'?'对话':'续写');
  }catch(e){el('status').textContent=e.message}finally{busy=false;describe()}
}
async function init(){
  try{
    const r=await fetch('/api/info');const info=await r.json();if(!r.ok)throw Error(info.error);
    el('device').textContent=info.device;scoped=info.experiments_scoped;
    if(info.include_experiments){
      const o=document.createElement('option');o.value='experiments';o.textContent=scoped?'本轮实验（未验收）':'实验检查点（未验收）';el('stage').prepend(o);
      if(scoped){const h=document.createElement('option');h.value='history';h.textContent='历史实验 / 工程验证（未验收）';el('stage').appendChild(h)}
    }
    el('stage').value=info.default_stage;await refresh();
  }catch(e){el('status').textContent=e.message}
}
init();</script></main></html>"""


def experiment_directories(root, experiment_roots):
    """Keep demo selection local to its root without changing training manifests."""
    root = Path(root).resolve()
    selected = tuple((root / item).resolve() for item in (experiment_roots or ()))
    if any(not item.is_relative_to(root) for item in selected):
        raise ValueError("--experiment-root must be inside --root")
    return selected


def checkpoints(root, stage=None, *, include_experiments=False, experiment_roots=None):
    if stage not in {
        None,
        "pretrain",
        "dense_distill",
        "sparse_cpt",
        "sft",
        "dpo",
        "grpo",
        "mopd",
        "opd",
        "accepted",
        "experiments",
        "history",
    }:
        raise ValueError("unknown checkpoint stage")
    experiment_view = stage in {"experiments", "history"}
    if experiment_view and not include_experiments:
        raise ValueError("experimental checkpoints require --include-experiments")
    names = {
        "miniqwen4": "MiniQwen4",
        "minikimik3": "MiniKimi-K3",
        "minideepseekv4": "MiniDeepSeek-V4",
    }
    found: dict[str, dict[str, Any]] = {}
    root = Path(root).resolve()
    selected = experiment_directories(root, experiment_roots)
    for run_path in root.rglob("run.json"):
        path = run_path.parent / "status.json"
        try:
            run = json.loads(run_path.read_text())
            state = json.loads(path.read_text()) if path.exists() else {}
        except (OSError, json.JSONDecodeError):
            continue  # A live writer may be atomically committing the next status.
        experimental = run.get("kind") == "acceptance"
        matching = [item for item in selected if run_path.parent.is_relative_to(item)]
        current = not selected or bool(matching)
        if experiment_view:
            if not experimental:
                continue
            if (stage == "experiments") != current:
                continue
        elif run.get("kind") not in {"educational", "strategy"}:
            continue
        saved_stage = state.get("stage", run.get("stage"))
        if stage not in {None, "accepted", "experiments", "history"} and saved_stage != stage:
            continue
        candidates = [path.parent / name for name in ("model.pt", "checkpoint.pt")]
        try:
            ckpt = max((p for p in candidates if p.is_file()), key=lambda p: p.stat().st_mtime_ns)
            stat = ckpt.stat()
        except (ValueError, OSError):
            continue
        family = run.get("model_name")
        if family not in names:
            continue
        relative = str(path.parent.relative_to(root))
        key = relative if experiment_view else family
        group = max(matching, key=lambda item: len(item.parts)) if matching else root
        run_label = str(path.parent.relative_to(group))
        if run_label.startswith(family + "/"):
            run_label = run_label[len(family) + 1 :]
        modified = stat.st_mtime
        quality_path = path.parent.parent / "quality.json"
        try:
            quality = json.loads(quality_path.read_text()) if quality_path.exists() else {}
        except (OSError, json.JSONDecodeError):
            quality = {}
        reviewed = quality.get("checkpoints", {}).get(saved_stage, {})
        # A review of another saved step must not qualify a newer checkpoint.
        artifact = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        capability = (
            reviewed.get("capability_status", "unassessed")
            if not experimental
            and reviewed.get("step") == state.get("step")
            and reviewed.get("artifact") == artifact
            else "unassessed"
        )
        if stage == "accepted" and capability != "passed":
            continue
        if key not in found or modified > found[key]["modified"]:
            found[key] = dict(
                id=key,
                name=names[family],
                model_name=family,
                run=relative,
                run_label=run_label,
                kind=run.get("kind"),
                stage=saved_stage,
                step=state.get("step"),
                state=state.get("state", "saved"),
                artifact=ckpt.name,
                path=ckpt,
                modified=modified,
                saved_at=modified,
                ce_tokens=state.get("token_ledger", {}).get("ce_tokens"),
                ce_token_budget=run.get("ce_token_budget"),
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


def serve(
    root="outputs",
    host="127.0.0.1",
    port=7860,
    device="cpu",
    gpu=None,
    include_experiments=False,
    experiment_roots=None,
):
    if experiment_roots and not include_experiments:
        raise ValueError("--experiment-root requires --include-experiments")
    experiment_roots = experiment_directories(root, experiment_roots)
    device, device_label = demo_device(device, gpu)
    lock = threading.Lock()
    default_stage = "experiments" if include_experiments else "accepted"

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
                stage = parse_qs(url.query).get("stage", [default_stage])[0]
                if stage not in {
                    "accepted",
                    "sft",
                    "dpo",
                    "pretrain",
                    "latest",
                    "experiments",
                    "history",
                } or (stage in {"experiments", "history"} and not include_experiments):
                    self.reply({"error": "unknown checkpoint stage"}, 400)
                    return
                self.reply(
                    [
                        {k: v for k, v in m.items() if k not in {"path", "modified"}}
                        for m in checkpoints(
                            root,
                            None if stage == "latest" else stage,
                            include_experiments=include_experiments,
                            experiment_roots=experiment_roots,
                        ).values()
                    ]
                )
            elif url.path == "/api/info":
                self.reply(
                    dict(
                        device=device_label,
                        include_experiments=include_experiments,
                        default_stage=default_stage,
                        experiments_scoped=bool(experiment_roots),
                    )
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
                stage = data.get("stage", default_stage)
                candidate = checkpoints(
                    root,
                    None if stage == "latest" else stage,
                    include_experiments=include_experiments,
                    experiment_roots=experiment_roots,
                )[data["model"]]
                count = int(data.get("max_new_tokens", 64))
                if not 1 <= count <= 256:
                    raise ValueError("生成长度必须在 1–256 之间")
                if not isinstance(data["prompt"], str) or not data["prompt"].strip():
                    raise ValueError("请输入文本")
                mode = data.get("mode", "auto")
                if mode not in {"auto", "chat", "completion"}:
                    raise ValueError("unknown generation mode")
                with lock:
                    model = None
                    try:
                        model, tokenizer, meta = load_checkpoint(candidate["path"], device)
                        chat = mode == "chat" or (
                            mode == "auto"
                            and meta["stage"] in {"sft", "dpo", "grpo", "mopd", "opd", "accepted"}
                        )
                        result = respond(
                            model, tokenizer, data["prompt"], chat=chat, max_new_tokens=count
                        )
                    finally:
                        del model
                        if str(device).startswith("cuda"):
                            torch.cuda.empty_cache()
                self.reply(
                    dict(
                        text=result,
                        **meta,
                        device=device_label,
                        run=candidate["run"],
                        capability_status=candidate["capability_status"],
                        generation_mode="chat" if chat else "completion",
                    )
                )
            except (ValueError, KeyError, RuntimeError, OSError) as error:
                self.reply({"error": str(error)}, 400)

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"MiniFrontier demo: http://{host}:{port} · {device_label} · {default_stage}", flush=True)
    server.serve_forever()
