# ruff: noqa: RUF001
"""Small local text/image/frame-video demo bound to one explicit exported checkpoint."""

import base64
import hashlib
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import torch
from PIL import Image

from minifrontier.data import sha256
from minifrontier.inference import generate_ids, load_checkpoint
from minifrontier.models.minifrontier1.processing import process_frames
from minifrontier.multimodal import move

from .data import safe_text

PAGE = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MiniFrontier1.0</title><style>body{font:16px system-ui;background:#121a29;color:#e8edf5;max-width:820px;margin:4vh auto;padding:24px}p{line-height:1.7;color:#b8c6d9}textarea,button,select,input{font:inherit;background:#203047;color:inherit;border:1px solid #56708d;border-radius:7px;padding:10px;margin:7px 0}textarea{box-sizing:border-box;width:100%;height:120px}pre{white-space:pre-wrap;background:#1b293d;padding:20px;border-radius:8px}button{cursor:pointer}label{display:block}</style>
<h1>MiniFrontier1.0</h1><p id="status">正在读取模型状态…</p>
<textarea id="prompt">请描述图片中的内容。</textarea>
<label>图片（支持多张） <input id="images" type="file" accept="image/*" multiple></label>
<label>短视频 <input id="video" type="file" accept="video/*"></label>
<label>视频抽帧数 <select id="frames"><option>4</option><option selected>8</option><option>16</option></select></label>
<label>回答模式 <select id="mode"><option value="direct">直接回答</option><option value="thinking">简短思考</option></select></label>
<label>最多生成 <input id="budget" type="number" min="1" max="256" value="64"> token</label>
<button id="prepare" onclick="prepare()">查看输入预算</button> <button id="generate" onclick="generate()" disabled>生成</button>
<p id="plan">先查看实际使用的图片大小、帧数与上下文占用。</p><pre id="answer"></pre>
<script>
let payload=null;const $=id=>document.getElementById(id);
function read(file){return new Promise((ok,no)=>{const r=new FileReader();r.onload=()=>ok(r.result);r.onerror=no;r.readAsDataURL(file)})}
async function videoFrames(file,count){const v=document.createElement('video');v.muted=true;const url=URL.createObjectURL(file);try{await new Promise((ok,no)=>{v.onloadedmetadata=ok;v.onerror=no;v.src=url});if(!Number.isFinite(v.duration)||v.duration<=0)throw Error('无法读取视频时长');const canvas=document.createElement('canvas');canvas.width=Math.min(v.videoWidth,1280);canvas.height=Math.round(v.videoHeight*canvas.width/v.videoWidth);const result=[],timestamps=[];for(let i=0;i<count;i++){const t=(i+.1)/count*v.duration;await new Promise(ok=>{v.onseeked=ok;v.currentTime=t});canvas.getContext('2d').drawImage(v,0,0,canvas.width,canvas.height);result.push(canvas.toDataURL('image/jpeg',.85));timestamps.push(t)}return{kind:'video',frames:result,timestamps}}finally{URL.revokeObjectURL(url)}}
async function call(path,value){const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(value)});const data=await r.json();if(!r.ok)throw Error(data.error);return data}
async function prepare(){try{$('generate').disabled=true;const media=[];for(const f of $('images').files)media.push({kind:'image',frames:[await read(f)]});if($('video').files.length)media.push(await videoFrames($('video').files[0],Number($('frames').value)));payload={prompt:$('prompt').value,mode:$('mode').value,max_new_tokens:Number($('budget').value),media};const plan=await call('/api/prepare',payload);$('plan').textContent='输入 '+plan.input_tokens+' token，其中视觉 '+plan.vision_tokens+' token；'+plan.media.map(m=>m.frames+' 帧，'+m.width+'×'+m.height+'，'+m.tokens+' 视觉 token').join('；');$('generate').disabled=false}catch(e){$('plan').textContent=e.message}}
async function generate(){if(!payload)return;$('generate').disabled=true;try{const r=await call('/api/generate',payload);$('answer').textContent=r.text||'（模型立即结束，没有可显示的文本）';$('plan').textContent+='；生成 '+r.generated_tokens+' token，用时 '+r.seconds.toFixed(2)+' 秒'}catch(e){$('answer').textContent=e.message}finally{$('generate').disabled=false}}
for(const id of ['prompt','images','video','frames','mode','budget'])$(id).addEventListener('input',()=>{$('generate').disabled=true;payload=null});
fetch('/api/info').then(r=>r.json()).then(d=>{$('status').textContent=d.qualified?'已通过所绑定权重的能力验收。':'研究诊断权重：尚未通过语言、视觉和视频能力验收，输出可能无效。'});
</script></html>"""


def prepare_request(payload, model, tokenizer):
    if not isinstance(payload.get("prompt"), str) or not payload["prompt"].strip():
        raise ValueError("请输入文本问题")
    if payload.get("mode", "direct") not in {"direct", "thinking"}:
        raise ValueError("不支持此回答模式")
    budget = payload.get("max_new_tokens", 64)
    if type(budget) is not int or not 1 <= budget <= 256:
        raise ValueError("生成预算必须为 1-256")
    ids, plan = [1, 4], []
    spans: list[dict[str, Any]] = []
    decoded_pixels = 0
    for resource in payload.get("media", []):
        video = resource.get("kind") == "video"
        if (
            resource.get("kind") not in {"image", "video"}
            or not 1 <= len(resource.get("frames", [])) <= 16
        ):
            raise ValueError("媒体应为图片或至多 16 帧短视频")
        images, hashes = [], []
        for encoded in resource["frames"]:
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
                frames=len(images),
                width=sample["resized_size"][0],
                height=sample["resized_size"][1],
                tokens=sample["feature_count"],
            )
        )
    ids.extend(
        [
            *safe_text(tokenizer, payload["prompt"]),
            2,
            5,
            15 if payload.get("mode") == "thinking" else 17,
        ]
    )
    if len(ids) + budget > model.config.max_position_embeddings:
        raise ValueError("输入与回答预算超过上下文，请减少媒体或文本")
    return (
        torch.tensor([ids]),
        spans,
        dict(
            input_tokens=len(ids), vision_tokens=sum(s["feature_count"] for s in spans), media=plan
        ),
    )


def serve(checkpoint, *, device="cpu", host="127.0.0.1", port=7861, allow_unqualified=False):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    # A self-declared flag alone cannot qualify a public demo; require an evaluated export.
    qualified = False
    if not allow_unqualified:
        raise ValueError("当前 MF1 尚无能力验收发布包；诊断演示需显式 --allow-unqualified")
    model, tokenizer, _ = load_checkpoint(checkpoint, device)
    if saved["model_name"] != "minifrontier1":
        raise ValueError("MF1 demo requires an MF1 checkpoint")
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
                self.reply(dict(qualified=qualified, checkpoint_sha256=sha256(checkpoint)))
            else:
                self.reply(dict(error="not found"), 404)

        def do_POST(self):
            try:
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= 16 * 1024**2:
                    raise ValueError("上传内容超过 16 MiB 限制")
                payload = json.loads(self.rfile.read(size))
                with lock:
                    ids, spans, plan = prepare_request(payload, model, tokenizer)
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
                        max_new_tokens=payload["max_new_tokens"],
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
                        )
                    )
            except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
                self.reply(dict(error=str(error)), 400)

    print(f"MiniFrontier1 diagnostic demo: http://{host}:{port}", flush=True)
    ThreadingHTTPServer((host, port), Handler).serve_forever()
