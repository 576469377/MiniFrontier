'use strict';
const $ = id => document.getElementById(id);
let checkpoint = null, prepared = null, revision = 0, epoch = 0;
let preparing = false, generating = false, previewController = null, results = [], serial = 0;
const stageNames = {pretrain:'预训练', dense_pretrain:'稠密预训练', dense_distill:'索引器蒸馏', sparse_cpt:'稀疏续训', sft:'SFT', rl:'RL', opd:'OPD', dpo:'DPO'};
const date = seconds => seconds ? new Date(seconds * 1000).toLocaleString('zh-CN', {hour12:false}) : '未记录';
const number = value => value == null ? '未记录' : Number(value).toLocaleString();
function node(tag, text, className) {
  const el = document.createElement(tag);
  if (text != null) el.textContent = text;
  if (className) el.className = className;
  return el;
}
function status(text, failed = false) {
  $('status').textContent = text;
  $('status').className = failed ? 'status error' : 'status';
}
function controls() {
  $('controls').disabled = !checkpoint || generating;
  $('media-fields').disabled = $('input-mode').value === 'completion';
  $('mode').disabled = $('input-mode').value === 'completion';
  $('prepare').disabled = !checkpoint || preparing || generating;
  $('generate').disabled = !checkpoint || !prepared || preparing || generating;
  $('clear-results').disabled = !results.length || generating;
  $('mode-hint').textContent = $('input-mode').value === 'completion'
    ? '文本续写不添加对话模板，仅接受文字。已选媒体请先移除，或切换到对话模式。'
    : '图像和视频随同一条问题输入；回答模式需要相应训练才能有效。';
}
function invalidate() {
  revision++; epoch++; prepared = null; preparing = false;
  if (previewController) previewController.abort();
  $('plan').textContent = '输入已变化，请重新检查预算。';
  $('usage').hidden = true; $('media-preview').replaceChildren(); $('preview-note').hidden = true;
  $('media-summary').textContent = `${$('images').files.length} 张图片 · ${$('video').files.length ? '1 个视频' : '无视频'}`;
  status('输入已更新。'); controls();
}
async function call(path, payload, signal) {
  const response = await fetch(path, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload), signal});
  const data = await response.json();
  if (!response.ok) throw Error(data.error || '请求失败，请重试。');
  return data;
}
function read(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(Error('无法读取文件：' + file.name));
    reader.readAsDataURL(file);
  });
}
function videoEvent(video, event, action) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => finish(Error('视频读取超时，请换用较短视频。')), 15000);
    const done = () => finish();
    const fail = () => finish(Error('浏览器无法解码此视频，请换用其他格式。'));
    function finish(error) {
      clearTimeout(timer); video.removeEventListener(event, done); video.removeEventListener('error', fail);
      error ? reject(error) : resolve();
    }
    video.addEventListener(event, done, {once:true}); video.addEventListener('error', fail, {once:true});
    action();
  });
}
async function videoFrames(file, count, current) {
  const video = document.createElement('video'); video.muted = true; video.preload = 'auto';
  const url = URL.createObjectURL(file);
  try {
    await videoEvent(video, 'loadedmetadata', () => { video.src = url; });
    if (!Number.isFinite(video.duration) || video.duration <= 0 || !video.videoWidth) throw Error('无法读取视频尺寸或时长。');
    const canvas = document.createElement('canvas');
    canvas.width = Math.min(video.videoWidth, 1280); canvas.height = Math.round(video.videoHeight * canvas.width / video.videoWidth);
    const frames = [], timestamps = [];
    for (let i = 0; i < count; i++) {
      if (!current()) throw new DOMException('输入已变化', 'AbortError');
      const time = (i + .1) / count * video.duration;
      await videoEvent(video, 'seeked', () => { video.currentTime = time; });
      canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
      frames.push(canvas.toDataURL('image/jpeg', .85)); timestamps.push(time);
    }
    return {kind:'video', name:file.name, frames, timestamps};
  } finally { video.removeAttribute('src'); video.load(); URL.revokeObjectURL(url); }
}
function showPlan(plan, media) {
  $('plan').textContent = `输入 ${number(plan.input_tokens)} token（视觉 ${number(plan.vision_tokens)}），回答预留 ${plan.max_new_tokens}；共 ${number(plan.input_tokens + plan.max_new_tokens)} / ${number(plan.context_length)}，余量 ${number(plan.remaining_tokens)}。`;
  $('usage').max = plan.context_length; $('usage').value = plan.input_tokens + plan.max_new_tokens; $('usage').hidden = false;
  $('media-preview').replaceChildren();
  media.forEach((resource, index) => {
    const item = plan.media[index];
    const label = node('p', `${index + 1}. ${resource.name} · ${item.frames} 帧 · ${item.width}×${item.height} · ${item.tokens} 视觉 token`, 'hint');
    label.style.gridColumn = '1 / -1'; $('media-preview').appendChild(label);
    resource.frames.forEach((src, frame) => {
      const figure = node('figure'); const image = node('img'); image.src = src;
      image.alt = `${resource.name}，${resource.kind === 'video' ? '采样帧 ' + (frame + 1) : '图片 ' + (index + 1)}`;
      const caption = resource.kind === 'video' ? `帧 ${frame + 1} · ${resource.timestamps[frame].toFixed(2)} 秒` : `图片 ${index + 1}`;
      figure.append(image, node('figcaption', caption));
      $('media-preview').appendChild(figure);
    });
  });
  $('preview-note').hidden = !media.length;
  $('budget-panel').open = true;
}
async function prepare() {
  if (!checkpoint || generating) return;
  const id = ++epoch, version = revision;
  if (previewController) previewController.abort();
  const controller = new AbortController(); previewController = controller;
  const current = () => id === epoch && version === revision;
  const snapshot = {prompt:$('prompt').value, input_mode:$('input-mode').value, mode:$('mode').value, max_new_tokens:Number($('budget').value)};
  const images = Array.from($('images').files), video = $('video').files[0], count = Number($('frames').value);
  prepared = null; preparing = true; controls(); status('正在读取媒体并检查输入预算…');
  try {
    if (!snapshot.prompt.trim()) throw Error('请输入提示词。');
    if (!Number.isInteger(snapshot.max_new_tokens) || snapshot.max_new_tokens < 1 || snapshot.max_new_tokens > 256) throw Error('生成预算应为 1–256 的整数。');
    if (snapshot.input_mode === 'completion' && (images.length || video)) throw Error('文本续写不接收媒体，请移除媒体或切换到对话模式。');
    if (images.reduce((sum, file) => sum + file.size, 0) > 12 * 1024 ** 2) throw Error('图片总量较大，请缩小图片后重试。');
    const media = [];
    for (const file of images) {
      media.push({kind:'image', name:file.name, frames:[await read(file)]});
      if (!current()) return;
    }
    if (video) media.push(await videoFrames(video, count, current));
    if (!current()) return;
    const payload = {...snapshot, media};
    if (new TextEncoder().encode(JSON.stringify(payload)).length > 16 * 1024 ** 2) throw Error('处理后的请求超过 16 MiB，请减少媒体或采样帧数。');
    const plan = await call('/api/prepare', payload, controller.signal);
    if (!current()) return;
    if (plan.checkpoint_sha256 !== checkpoint.sha256) {
      $('retry-info').hidden = false; $('checkpoint-panel').open = true;
      throw Error('服务已更换检查点，请重新读取状态并检查预算。');
    }
    prepared = {payload:{...payload, request_id:plan.request_id, checkpoint_sha256:plan.checkpoint_sha256}, plan, version};
    showPlan(plan, media); status('预算检查通过，可以生成。');
  } catch (error) {
    if (current()) { prepared = null; $('plan').textContent = '预算检查未完成。'; status(error.message, true); }
  } finally {
    if (current()) { preparing = false; previewController = null; controls(); }
  }
}
function renderResults() {
  $('answer').replaceChildren(); $('result-count').textContent = `${results.length} 次`;
  if (!results.length) $('answer').appendChild(node('p', '尚无生成记录。', 'empty-state'));
  for (const result of results) {
    const card = node('article', null, 'result-card'); const meta = result.checkpoint, request = result.request;
    const header = node('div', null, 'result-meta'), identity = node('div');
    identity.append(node('strong', `#${result.serial} · ${request.input_mode === 'completion' ? '文本续写' : '对话'} · step ${meta.step ?? '未记录'}`));
    identity.append(node('p', `${meta.run} / ${meta.name} · ${(meta.sha256 || '').slice(0, 12)}`, 'hint'));
    header.append(identity, node('span', date(result.completed_at), 'hint'));
    const content = node('div', null, 'result-content');
    content.append(node('p', request.prompt, 'result-prompt'));
    content.append(node('pre', result.text || '（模型立即结束，没有可显示文本）', 'result-output'));
    const details = node('details', null, 'result-details'); details.append(node('summary', '输入预算与本次参数'));
    details.append(node('pre', JSON.stringify({checkpoint:meta, request_id:result.request_id, request, plan:result.plan}, null, 2), 'request-json'));
    const actions = node('div', null, 'result-actions');
    actions.append(node('span', `${result.generated_tokens} token · 生成 ${result.seconds.toFixed(2)} 秒 · 总等待 ${result.wall_seconds.toFixed(2)} 秒 · ${meta.device} · ${result.finish_reason === 'eos' ? 'EOS 结束' : '达到生成预算'}`, 'hint'));
    const copy = node('button', '复制输出', 'secondary small-button'); copy.type = 'button';
    copy.addEventListener('click', async () => {
      try { await navigator.clipboard.writeText(result.text || ''); copy.textContent = '已复制'; }
      catch { status('无法访问剪贴板，请手动选择输出复制。', true); }
    });
    actions.append(copy); card.append(header, content, details, actions); $('answer').appendChild(card);
  }
  controls();
}
async function generate() {
  if (!prepared || generating || prepared.version !== revision) return;
  const snapshot = prepared, started = Date.now();
  generating = true; controls();
  const waiting = () => status(`正在生成 · 已等待 ${Math.floor((Date.now() - started) / 1000)} 秒；服务繁忙时会排队。`);
  waiting(); const timer = setInterval(waiting, 1000);
  try {
    const result = await call('/api/generate', snapshot.payload);
    if (result.request_id !== snapshot.plan.request_id || result.checkpoint.sha256 !== snapshot.payload.checkpoint_sha256) throw Error('返回结果与本次输入或检查点不匹配，请重新检查预算。');
    results.unshift({...result, wall_seconds:(Date.now() - started) / 1000, serial:++serial}); results = results.slice(0, 8); renderResults();
    status('生成完成。本次输入与检查点身份已记录。');
  } catch (error) { status(error.message, true); }
  finally { clearInterval(timer); generating = false; controls(); }
}
async function init() {
  $('retry-info').hidden = true;
  try {
    const response = await fetch('/api/info'); const data = await response.json();
    if (!response.ok || !data.checkpoint) throw Error(data.error || '无法读取检查点信息。');
    checkpoint = data.checkpoint;
    $('model-state').textContent = data.qualified ? '通过能力验收' : '研究诊断 · 未验收';
    $('device').textContent = checkpoint.device;
    $('checkpoint-info').replaceChildren();
    const modelName = {minifrontier1: 'MiniFrontier1.0', minifrontier11: 'MiniFrontier1.1'}[checkpoint.model_name] || checkpoint.model_name || 'MF1';
    document.title = `${modelName} · 媒体检查`;
    const fields = [['模型', modelName], ['文件', `${checkpoint.run} / ${checkpoint.name}`], ['训练阶段', stageNames[checkpoint.stage] || checkpoint.stage || checkpoint.phase], ['已保存 step', number(checkpoint.step)], ['浮点参数', number(checkpoint.parameters)], ['上下文上限', `${number(checkpoint.context_length)} token`], ['权重保存时间', date(checkpoint.saved_at)], ['本服务加载时间', date(checkpoint.loaded_at)]];
    for (const [label, value] of fields) $('checkpoint-info').append(node('dt', label), node('dd', value));
    $('checkpoint-path').textContent = checkpoint.path; $('checkpoint-sha').textContent = checkpoint.sha256;
    if (revision === 0) $('input-mode').value = ['sft','rl','opd','dpo','teacher'].includes(checkpoint.stage) ? 'chat' : 'completion';
    status('先检查输入预算，再生成。'); controls();
  } catch (error) { checkpoint = null; $('model-state').textContent = '状态读取失败'; $('retry-info').hidden = false; $('checkpoint-panel').open = true; status(error.message, true); controls(); }
}
for (const id of ['prompt','images','video','frames','input-mode','mode','budget']) $(id).addEventListener('input', invalidate);
$('prepare').addEventListener('click', prepare); $('generate').addEventListener('click', generate);
$('clear-media').addEventListener('click', () => { $('images').value = ''; $('video').value = ''; invalidate(); });
$('clear-results').addEventListener('click', () => { results = []; renderResults(); });
$('retry-info').addEventListener('click', init);
const compactLayout = window.matchMedia('(max-width: 760px)');
function layoutPanels() {
  for (const id of ['checkpoint-panel', 'budget-panel']) $(id).open = !compactLayout.matches;
}
layoutPanels();
compactLayout.addEventListener('change', layoutPanels);
init();
