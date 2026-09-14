"use strict";

const $ = id => document.getElementById(id);
const names = {minifrontier11: "MiniFrontier1.1", minifrontier1: "MiniFrontier1.0", minikimik3: "MiniKimi-K3", miniqwen4: "MiniQwen4", minideepseekv4: "MiniDeepSeek-V4", minideepseekv41: "MiniDeepSeek-V4.1"};
const stages = {pretrain: "预训练", dense_distill: "稠密蒸馏", sparse_cpt: "稀疏续训", sft: "SFT", dpo: "DPO", grpo: "GRPO", mopd: "MOPD", opd: "OPD", rl: "RL", teacher: "教师蒸馏"};
const units = {ce_tokens: "CE tokens", input_tokens: "输入 tokens", response_tokens: "回答 tokens"};
const states = {running: "记录：训练中", complete: "本次运行已结束", budget_complete_unqualified: "预算完成 · 未验收", failed: "记录：失败", stopped: "记录：已停止", saved: "已保存"};
const capabilities = {passed: "已验收", failed: "验收未通过", unassessed: "未验收"};
const presets = {
  zh: {prompt: "人工智能是一种", mode: "completion"},
  en: {prompt: "The purpose of scientific research is to", mode: "completion"},
  code: {prompt: "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n", mode: "completion"},
  chat: {prompt: "请用两句话解释为什么会有四季。", mode: "chat"}
};
let models = [], selectedId = null, busy = false, refreshing = false, ready = false, refreshId = 0;
let records = [], recordId = 0, comparisons = new Set();

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}
function tokens(value) {
  if (value == null) return "—";
  const n = Number(value);
  return n >= 1e9 ? (n / 1e9).toFixed(2) + "B" : n >= 1e6 ? (n / 1e6).toFixed(2) + "M" : n.toLocaleString();
}
function savedTime(value) {
  return value ? new Date(value * 1000).toLocaleString("zh-CN", {hour12: false}) : "未记录";
}
function selected() { return models.find(model => model.id === selectedId); }
function setStatus(message, error = false) { $("status").textContent = message; $("status").classList.toggle("error", error); }
function updateControls() {
  for (const id of ["stage", "family", "search", "mode", "length", "temperature", "top-p", "seed", "prompt", "reset-settings"]) $(id).disabled = busy || !ready;
  for (const button of document.querySelectorAll(".preset, .run-card")) button.disabled = busy || !ready;
  $("refresh").disabled = busy || refreshing;
  $("send").disabled = busy || refreshing || !ready || !selected() || !$("prompt").value.trim();
  $("send").textContent = busy ? "正在生成…" : "生成";
  for (const button of document.querySelectorAll("[data-reuse]")) button.disabled = busy;
  $("clear").disabled = busy || !records.length;
  $("export").disabled = !records.some(record => record.state !== "pending");
}
function isChat(model, mode) {
  return mode === "chat" || (mode === "auto" && ["sft", "dpo", "grpo", "mopd", "opd", "rl", "teacher", "accepted"].includes(model?.stage));
}
function describe() {
  const model = selected();
  const details = $("checkpoint-details");
  details.replaceChildren();
  $("capability").textContent = model ? (capabilities[model.capability_status] || "未验收") : "未选择";
  $("capability").className = "badge" + (model?.capability_status === "passed" ? " accent" : "");
  if (!model) {
    details.append(node("p", "hint", "选择左侧检查点查看详情。"));
    $("selected-summary").textContent = "请选择检查点";
    $("mode-summary").textContent = "预训练适合文本续写；后训练可使用对话模板。";
    updateControls();
    return;
  }
  details.append(node("h3", "", model.name));
  const labels = node("div", "row");
  labels.style.marginTop = "8px";
  labels.append(node("span", "badge accent", stages[model.stage] || model.stage || "阶段未记录"));
  if (model.mf1_phase) labels.append(node("span", "badge", model.mf1_phase.toUpperCase()));
  details.append(labels, node("div", "divider"));
  const metrics = node("div", "metric-grid");
  const metric = (label, value) => { const item = node("dl", "metric"); item.append(node("dt", "", label), node("dd", "", value)); return item; };
  metrics.append(metric("保存 step", model.step == null ? "待载入确认" : Number(model.step).toLocaleString()), metric("状态", states[model.state] || model.state || "已保存"));
  details.append(metrics);
  const progress = model.phase_tokens ?? model.ce_tokens;
  const budget = model.token_budget ?? model.ce_token_budget;
  if (progress != null) {
    const track = node("div", "progress-track");
    const fill = node("span", "progress-fill");
    fill.style.width = budget ? Math.max(0, Math.min(100, 100 * progress / budget)) + "%" : "0%";
    track.append(fill);
    details.append(track, node("p", "hint", "已保存 " + tokens(progress) + (budget ? " / " + tokens(budget) : "") + " " + (units[model.budget_unit] || "tokens")));
  }
  if (["minifrontier1", "minifrontier11"].includes(model.model_name) && model.main_ce_tokens != null) details.append(node("p", "hint", "累计主 CE " + tokens(model.main_ce_tokens) + " tokens"));
  details.append(node("div", "divider"));
  const list = node("dl", "detail-list");
  for (const [label, value] of [["运行", model.run_label || model.run], ["保存时间", savedTime(model.saved_at)], ["权重文件", model.artifact]]) {
    const item = node("div"); item.append(node("dt", "", label), node("dd", "", value)); list.append(item);
  }
  details.append(list, node("div", "divider"));
  const paths = node("details");
  paths.append(node("summary", "", "运行路径与版本"), node("code", "", model.run + "/" + model.artifact), node("p", "hint", "版本 " + model.version));
  details.append(paths);
  $("selected-summary").textContent = model.name + " · step " + (model.step ?? "待确认");
  $("mode-summary").textContent = (isChat(model, $("mode").value) ? "对话模板" : "文本续写") + " · Ctrl / ⌘ + Enter 生成";
  updateControls();
}
function renderModels() {
  const query = $("search").value.trim().toLowerCase();
  const visible = models.filter(model => ($("family").value === "all" || model.model_name === $("family").value) && [model.name, model.run, model.run_label].some(value => String(value).toLowerCase().includes(query)));
  if (!visible.some(model => model.id === selectedId)) selectedId = visible[0]?.id ?? null;
  $("models").replaceChildren();
  for (const model of visible) {
    const card = node("button", "run-card"); card.type = "button"; card.dataset.id = model.id;
    card.setAttribute("aria-pressed", String(model.id === selectedId));
    const top = node("div", "row"); top.append(node("span", "run-name", model.name), node("span", "badge", stages[model.stage] || model.stage || "—"));
    const bottom = node("div", "row"); bottom.append(node("span", "run-step", "step " + (model.step == null ? "待确认" : Number(model.step).toLocaleString())), node("span", "hint", tokens(model.phase_tokens ?? model.ce_tokens) + " tokens"));
    card.append(top, node("span", "run-label", model.run_label || model.run), bottom);
    card.addEventListener("click", () => { if (busy) return; selectedId = model.id; $("checkpoint-notice").hidden = true; renderModels(); });
    $("models").append(card);
  }
  if (!visible.length) {
    $("models").append(node("p", "empty-state", models.length ? "没有匹配的运行，请调整筛选。" : "当前范围没有已保存的检查点。可切换范围，或在训练首次保存后刷新。"));
  }
  $("scope").textContent = visible.length + " / " + models.length + " 个运行 · 按保存时间排序";
  describe();
}
async function request(path, payload) {
  const response = await fetch(path, payload ? {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)} : undefined);
  let data;
  try { data = await response.json(); } catch { throw Error("服务未返回有效数据，请检查 Demo 服务状态。"); }
  if (!response.ok) {
    const error = Error(data.error || "请求失败（HTTP " + response.status + "）");
    error.code = data.code; error.status = response.status; throw error;
  }
  return data;
}
async function refresh() {
  if (busy) return;
  const id = ++refreshId;
  const previous = selected();
  refreshing = true; updateControls(); $("models").setAttribute("aria-busy", "true");
  $("list-status").textContent = "正在读取保存记录…"; $("list-status").classList.remove("error");
  try {
    const data = await request("/api/models?stage=" + encodeURIComponent($("stage").value));
    if (id !== refreshId) return;
    if (!Array.isArray(data)) throw Error("检查点列表格式无效。");
    models = data; renderModels();
    $("list-status").textContent = "更新于 " + new Date().toLocaleTimeString("zh-CN", {hour12: false});
    if (previous && selected()?.id === previous.id && selected().version !== previous.version) {
      $("checkpoint-notice").textContent = "这个运行已有新权重，当前选择已更新。此前生成的记录仍保留原版本。";
      $("checkpoint-notice").hidden = false;
    }
  } catch (error) {
    if (id !== refreshId) return;
    models = []; selectedId = null; renderModels();
    $("list-status").textContent = error.message; $("list-status").classList.add("error");
  } finally {
    if (id === refreshId) { refreshing = false; $("models").setAttribute("aria-busy", "false"); updateControls(); }
  }
}
function settings() {
  for (const id of ["length", "temperature", "top-p", "seed"]) {
    if (!$(id).checkValidity()) $("settings-panel").open = true;
    if (!$(id).reportValidity()) throw Error("请检查生成参数的取值范围。");
  }
  return {mode: $("mode").value, max_new_tokens: Number($("length").value), temperature: Number($("temperature").value), top_p: Number($("top-p").value), seed: Number($("seed").value)};
}
function setSettings(values) {
  for (const [id, key] of [["mode", "mode"], ["length", "max_new_tokens"], ["temperature", "temperature"], ["top-p", "top_p"], ["seed", "seed"]]) $(id).value = values[key];
  describe();
}
function resultCard(record, compact = false) {
  const data = record.response || {}, source = record.checkpoint;
  const card = node("article", "result-card" + (record.state === "pending" ? " pending" : ""));
  const header = node("div", "result-meta"), identity = node("div");
  identity.append(node("strong", "", names[data.model_name] || source.name), node("p", "hint", (stages[data.stage || source.stage] || data.stage || source.stage || "—") + " · step " + (data.step ?? source.step ?? "待确认") + " · " + new Date(record.created_at).toLocaleTimeString("zh-CN", {hour12: false})));
  header.append(identity, node("span", "badge" + (record.state === "done" ? " accent" : ""), record.state === "pending" ? "处理中" : record.state === "error" ? "失败" : (data.generation_mode === "chat" ? "对话" : "续写")));
  const body = node("div", "result-content");
  body.append(node("p", "result-prompt", record.request.prompt));
  if (record.state === "pending") {
    const loading = node("p", "status"); loading.append(node("span", "spinner"), node("span", "", "请求处理中，包含权重加载与生成…")); body.append(loading);
  } else body.append(node("pre", "result-output" + (record.state === "error" ? " error" : ""), record.state === "error" ? record.error : data.text || "（模型立即结束，没有可显示文本）"));
  card.append(header, body);
  const identityText = (data.run || source.run) + "/" + (data.artifact || source.artifact);
  const details = node("details", "result-details");
  details.append(node("summary", "", "本次权重与参数"));
  details.append(node("code", "", identityText), node("code", "", "版本 " + (data.version || source.version)));
  const config = record.request;
  details.append(node("code", "", "Temperature " + (data.temperature ?? config.temperature) + " · Top-p " + (data.top_p ?? config.top_p) + " · seed " + (data.seed ?? config.seed) + " · 最多 " + config.max_new_tokens + " token"));
  if (data.device) details.append(node("code", "", data.device));
  if (data.load_seconds != null) details.append(node("code", "", "加载 " + data.load_seconds.toFixed(2) + " s · 生成 " + data.generation_seconds.toFixed(2) + " s"));
  card.append(details);
  if (compact) return card;
  const actions = node("div", "result-actions"), buttons = node("div", "row");
  if (record.state === "done") {
    const label = node("label", "compare-label"), input = node("input");
    input.type = "checkbox"; input.checked = comparisons.has(record.id); input.setAttribute("aria-label", "对比记录 " + record.id);
    input.addEventListener("change", () => {
      if (input.checked && comparisons.size >= 2) { input.checked = false; setStatus("每次可对比两条记录，请先取消一条。"); return; }
      if (input.checked) comparisons.add(record.id); else comparisons.delete(record.id);
      updateHistoryControls(); if (!$("comparison").hidden) showComparison();
    });
    label.append(input, document.createTextNode("加入对比")); actions.append(label);
    const copy = node("button", "secondary small-button", "复制输出"); copy.type = "button";
    copy.addEventListener("click", async () => {
      try { await navigator.clipboard.writeText(data.text || ""); copy.textContent = "已复制"; setTimeout(() => { copy.textContent = "复制输出"; }, 1800); }
      catch { setStatus("浏览器未允许复制，请选中输出文本手动复制。", true); }
    }); buttons.append(copy);
  } else actions.append(node("span", "hint", record.state === "pending" ? "正在使用此条记录中的检查点" : "未生成结果"));
  if (record.state !== "pending") {
    const reuse = node("button", "secondary small-button", "复用输入"); reuse.type = "button"; reuse.dataset.reuse = record.id;
    reuse.addEventListener("click", () => { if (busy) return; $("prompt").value = record.request.prompt; setSettings(record.request); $("prompt").focus(); setStatus("已复用输入和参数，可选择另一个检查点进行比较。"); });
    buttons.append(reuse);
    if (data.seconds != null) buttons.append(node("span", "hint", "共 " + data.seconds.toFixed(2) + " s"));
  }
  actions.append(buttons); card.append(actions);
  return card;
}
function updateHistoryControls() {
  $("result-count").textContent = records.filter(record => record.state !== "pending").length;
  $("compare").textContent = "对比（" + comparisons.size + "/2）";
  $("compare").disabled = comparisons.size !== 2;
  updateControls();
}
function renderHistory() {
  $("empty").hidden = records.length > 0;
  $("results").replaceChildren(...records.map(record => resultCard(record)));
  updateHistoryControls();
}
function showComparison() {
  const pair = records.filter(record => comparisons.has(record.id));
  $("comparison").hidden = pair.length !== 2;
  if (pair.length !== 2) return;
  const samePrompt = pair[0].request.prompt === pair[1].request.prompt;
  const sameParameters = ["mode", "max_new_tokens", "temperature", "top_p", "seed"].every(key => pair[0].request[key] === pair[1].request[key]) && pair[0].response.generation_mode === pair[1].response.generation_mode;
  $("comparison-note").textContent = samePrompt && sameParameters ? "两条记录使用相同输入和生成参数。" : "这两条记录的输入或生成参数不同，比较时请留意差异。";
  $("comparison-grid").replaceChildren(...pair.map(record => resultCard(record, true)));
}
async function send() {
  if (busy || refreshing || !selected() || !$("prompt").value.trim()) return;
  let config;
  try { config = settings(); } catch (error) { setStatus(error.message, true); return; }
  const checkpoint = {...selected()};
  const payload = {model: checkpoint.id, version: checkpoint.version, stage: $("stage").value, prompt: $("prompt").value, ...config};
  const record = {id: ++recordId, created_at: new Date().toISOString(), state: "pending", checkpoint, request: payload};
  records.unshift(record);
  for (const removed of records.splice(20)) comparisons.delete(removed.id);
  busy = true; renderHistory();
  const start = performance.now();
  setStatus("请求处理中 · 0 秒");
  const timer = setInterval(() => setStatus("请求处理中 · " + Math.floor((performance.now() - start) / 1000) + " 秒（包含加载与生成）"), 1000);
  try {
    record.response = await request("/api/generate", payload);
    record.state = "done";
    setStatus("生成完成，结果已保存到当前页面记录。");
  } catch (error) {
    record.state = "error"; record.error = error.message;
    if (error.status === 409) {
      record.error = "所选检查点已更新。请刷新列表，确认新版本后重新生成。";
      $("checkpoint-notice").textContent = record.error; $("checkpoint-notice").hidden = false;
    }
    setStatus(record.error, true);
  } finally {
    clearInterval(timer); busy = false; renderHistory();
    if (!$("comparison").hidden) showComparison();
  }
}
async function init() {
  updateControls();
  try {
    const info = await request("/api/info");
    $("device").textContent = info.device;
    if (info.include_experiments) {
      const option = node("option", "", info.experiments_scoped ? "本次实验（未验收）" : "实验与诊断（未验收）"); option.value = "experiments"; $("stage").append(option);
      if (info.experiments_scoped) { const history = node("option", "", "历史实验（未验收）"); history.value = "history"; $("stage").append(history); }
    }
    $("stage").value = info.default_stage || "formal";
    if (!$("stage").value) $("stage").value = "formal";
    ready = true; await refresh();
  } catch (error) { ready = true; $("device").textContent = "设备信息不可用"; setStatus(error.message, true); updateControls(); }
}

$("refresh").addEventListener("click", refresh);
$("stage").addEventListener("change", () => { $("checkpoint-notice").hidden = true; refresh(); });
$("family").addEventListener("change", renderModels);
$("search").addEventListener("input", renderModels);
$("mode").addEventListener("change", describe);
$("prompt").addEventListener("input", updateControls);
$("prompt").addEventListener("keydown", event => { if ((event.ctrlKey || event.metaKey) && event.key === "Enter") { event.preventDefault(); send(); } });
$("send").addEventListener("click", send);
$("reset-settings").addEventListener("click", () => setSettings({mode: "auto", max_new_tokens: 64, temperature: 0, top_p: 1, seed: 0}));
for (const button of document.querySelectorAll("[data-preset]")) button.addEventListener("click", () => { if (busy) return; const preset = presets[button.dataset.preset]; $("prompt").value = preset.prompt; $("mode").value = preset.mode; describe(); $("prompt").focus(); });
$("compare").addEventListener("click", showComparison);
$("close-comparison").addEventListener("click", () => { $("comparison").hidden = true; });
$("clear").addEventListener("click", () => { if (busy) return; records = []; comparisons.clear(); $("comparison").hidden = true; renderHistory(); });
$("export").addEventListener("click", () => {
  const exported = records.filter(record => record.state !== "pending");
  const url = URL.createObjectURL(new Blob([JSON.stringify({format: "minifrontier-demo-results-v1", exported_at: new Date().toISOString(), records: exported}, null, 2)], {type: "application/json"}));
  const link = node("a"); link.href = url; link.download = "minifrontier-results-" + new Date().toISOString().replaceAll(":", "-") + ".json";
  document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
});
const narrowScreen = window.matchMedia("(max-width: 760px)");
function arrangeDetails() { for (const id of ["weight-panel", "settings-panel"]) $(id).open = !narrowScreen.matches; }
narrowScreen.addEventListener("change", arrangeDetails);
arrangeDetails();
init();
