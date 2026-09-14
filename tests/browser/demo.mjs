// Optional browser regression checks. No server, model, GPU, or npm install required.
// DEMO_CHROMIUM=/path/to/chrome-headless-shell node tests/browser/demo.mjs
// Set DEMO_SCREENSHOTS=/tmp/demo-preview to retain desktop/mobile screenshots.
import assert from "node:assert/strict";
import {spawn} from "node:child_process";
import {existsSync} from "node:fs";
import {mkdtemp, mkdir, readFile, rm, writeFile} from "node:fs/promises";
import {tmpdir} from "node:os";
import {join} from "node:path";
import {setTimeout as delay} from "node:timers/promises";
import {fileURLToPath, pathToFileURL} from "node:url";

const assets = fileURLToPath(new URL("../../minifrontier/inference/web/", import.meta.url));
const browser = process.env.DEMO_CHROMIUM;
assert(browser && existsSync(browser), "Set DEMO_CHROMIUM to an existing Chromium/headless-shell executable (Node 22+).");
const temp = await mkdtemp(join(tmpdir(), "minifrontier-demo-browser-"));
let child, socket;
const failures = [], requests = new Map();
let sequence = 0;
const screenshotDir = process.env.DEMO_SCREENSHOTS;

const fixtures = [
  ["minifrontier1", "MiniFrontier1.0", "mf1-base/p1", "p1", 2400, 48000000, 500000000],
  ["minideepseekv4", "MiniDeepSeek-V4", "source-base/minideepseekv4/D2", null, 4900, 164108220, 500000000],
  ["miniqwen4", "MiniQwen4", "source-base/miniqwen4/Q2", null, 3600, 121000000, 500000000],
  ["minikimik3", "MiniKimi-K3", "source-base/minikimik3/K2", null, 3100, 103000000, 500000000],
  ["miniqwen4", "MiniQwen4", "source-base/miniqwen4/Q1", null, 2100, 72000000, 500000000],
  ["minifrontier11", "MiniFrontier1.1", "mf11-base/p0", "p0", 120, 2400000, 200000000],
  ["minideepseekv41", "MiniDeepSeek-V4.1", "source-base/minideepseekv41/D1", null, 100, 3200000, 2200000000],
].map(([model_name, name, id, mf1_phase, step, phase_tokens, token_budget], i) => ({
  model_name, name, id, run: id, run_label: id, mf1_phase, step, phase_tokens, token_budget,
  stage: "pretrain", kind: "strategy", state: "running", capability_status: "unassessed",
  artifact: "checkpoint.pt", version: "fixture-version-" + i, budget_unit: "ce_tokens", saved_at: 1789286400 - i * 600,
}));

// All API traffic stays inside the browser fixture, including delayed/failed requests.
function installMock(initialModels) {
  const originalFetch = window.fetch.bind(window);
  const loaded = {name: "checkpoint.pt", run: "mf11-base/p0", path: "fixtures/mf11-base/p0/checkpoint.pt", sha256: "a".repeat(64), loaded_at: 1789286400, saved_at: 1789285800, model_name: "minifrontier11", stage: "pretrain", phase: "dense", step: 120, device: "CPU · 浏览器测试数据", parameters: 210859393, context_length: 4096};
  const state = window.demoMock = {models: initialModels, loaded, posts: [], waiting: [], holdLists: false, holdPrepare: false, failGenerate: false, infoFailed: false, generated: 0};
  const answer = (value, status = 200) => new Response(JSON.stringify(value), {status, headers: {"Content-Type": "application/json"}});
  function plan(payload) {
    const media = (payload.media || []).map(resource => ({kind: resource.kind, name: resource.name, frames: resource.frames.length, width: 64, height: 48, tokens: 16 * resource.frames.length, timestamps: resource.timestamps || []}));
    const visual = media.reduce((total, item) => total + item.tokens, 0);
    return {request_id: "fixture-" + payload.prompt, checkpoint_sha256: loaded.sha256, input_tokens: visual + 24, vision_tokens: visual, media, max_new_tokens: payload.max_new_tokens, context_length: 4096, remaining_tokens: 4096 - visual - 24 - payload.max_new_tokens};
  }
  window.fetch = async (url, options) => {
    const path = String(url), mf1 = location.pathname.endsWith("mf1.html");
    if (!path.startsWith("/api/")) return originalFetch(url, options);
    if (path === "/api/info") return state.infoFailed ? answer({error: "模拟服务不可用"}, 503) : answer(mf1 ? {qualified: false, checkpoint: loaded, checkpoint_sha256: loaded.sha256} : {device: "CPU · 浏览器测试数据", include_experiments: true, experiments_scoped: true, default_stage: "formal"});
    if (path.startsWith("/api/models")) {
      const result = structuredClone(state.models);
      if (state.holdLists) return new Promise(resolve => state.waiting.push(value => resolve(answer(value ?? result))));
      return answer(result);
    }
    const payload = JSON.parse(options.body); state.posts.push({path, payload});
    if (path === "/api/prepare") {
      if (state.holdPrepare) return new Promise(resolve => state.waiting.push(() => resolve(answer(plan(payload)))));
      return answer(plan(payload));
    }
    if (path === "/api/generate") {
      await new Promise(resolve => setTimeout(resolve, 150));
      if (state.failGenerate) return answer({error: "checkpoint updated", code: "checkpoint_changed"}, 409);
      state.generated++;
      if (mf1) return answer({text: "媒体测试输出 <img src=x onerror=alert(1)>", generated_tokens: 18, seconds: .42, completed_at: 1789286500, finish_reason: "eos", request_id: plan(payload).request_id, checkpoint: loaded, plan: plan(payload), request: {prompt: payload.prompt, input_mode: payload.input_mode, mode: payload.mode, max_new_tokens: payload.max_new_tokens, temperature: 0, top_p: 1}});
      const model = state.models.find(item => item.id === payload.model);
      return answer({...model, text: "用于浏览器验证的示例输出。\n<img src=x onerror=alert(1)>\n生成记录应完整保留本次输入与检查点。", device: "CPU · 浏览器测试数据", generation_mode: payload.mode === "chat" ? "chat" : "completion", temperature: payload.temperature, top_p: payload.top_p, seed: payload.seed, load_seconds: .13, generation_seconds: .42, seconds: .55});
    }
    throw Error("Unexpected API request: " + path);
  };
}

async function cdp(method, params = {}) {
  const id = ++sequence;
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => { requests.delete(id); reject(Error("CDP timeout: " + method)); }, 15000);
    requests.set(id, {resolve: value => { clearTimeout(timeout); resolve(value); }, reject: error => { clearTimeout(timeout); reject(error); }});
    socket.send(JSON.stringify({id, method, params}));
  });
}
async function evaluate(expression) {
  const response = await cdp("Runtime.evaluate", {expression, returnByValue: true, awaitPromise: true});
  if (response.exceptionDetails) throw Error(JSON.stringify(response.exceptionDetails));
  return response.result.value;
}
async function until(expression) {
  for (let i = 0; i < 150; i++) { if (await evaluate(expression)) return; await delay(40); }
  throw Error("Condition timed out: " + expression);
}
async function screenshot(name, width = 1440, height = 1100) {
  await cdp("Emulation.setDeviceMetricsOverride", {width, height, deviceScaleFactor: 1, mobile: width < 760});
  await delay(60);
  assert(await evaluate("document.documentElement.scrollWidth <= window.innerWidth"), "Page must not overflow horizontally: " + name);
  if (!screenshotDir) return;
  await mkdir(screenshotDir, {recursive: true});
  const result = await cdp("Page.captureScreenshot", {format: "png", captureBeyondViewport: false});
  await writeFile(join(screenshotDir, name + ".png"), Buffer.from(result.data, "base64"));
}
async function navigate(name) {
  const css = await readFile(join(assets, "demo.css"), "utf8"), js = await readFile(join(assets, name + ".js"), "utf8");
  const html = (await readFile(join(assets, name + ".html"), "utf8")).replace("<!--STYLE-->", () => css).replace("<!--SCRIPT-->", () => js);
  const path = join(temp, name + ".html"); await writeFile(path, html);
  await cdp("Page.navigate", {url: pathToFileURL(path).href});
  await until("location.href === " + JSON.stringify(pathToFileURL(path).href) + " && document.readyState === 'complete'");
}
async function upload(selector, file) {
  const root = await cdp("DOM.getDocument");
  const result = await cdp("DOM.querySelector", {nodeId: root.root.nodeId, selector});
  await cdp("DOM.setFileInputFiles", {nodeId: result.nodeId, files: [file]});
}

try {
  child = spawn(browser, ["--headless", "--no-sandbox", "--disable-gpu", "--use-gl=disabled", "--disable-features=Vulkan", "--disable-dev-shm-usage", "--disable-background-networking", "--no-first-run", "--remote-debugging-port=0", "--user-data-dir=" + temp, "about:blank"], {stdio: "ignore", env: {...process.env, CUDA_VISIBLE_DEVICES: "", LIBGL_ALWAYS_SOFTWARE: "1"}});
  child.on("error", error => failures.push(error.message));
  let port;
  for (let i = 0; i < 100; i++) { try { port = (await readFile(join(temp, "DevToolsActivePort"), "utf8")).split("\n")[0]; break; } catch { await delay(50); } }
  assert(port, "Browser did not start: " + failures.join("; "));
  const tabs = await (await fetch("http://127.0.0.1:" + port + "/json/list")).json();
  socket = new WebSocket(tabs[0].webSocketDebuggerUrl);
  await new Promise(resolve => socket.addEventListener("open", resolve, {once: true}));
  socket.addEventListener("message", event => {
    const data = JSON.parse(event.data);
    if (data.id && requests.has(data.id)) { const pending = requests.get(data.id); requests.delete(data.id); data.error ? pending.reject(Error(JSON.stringify(data.error))) : pending.resolve(data.result); }
    if (data.method === "Runtime.exceptionThrown") failures.push(data.params.exceptionDetails.text);
  });
  await cdp("Page.enable"); await cdp("Runtime.enable");
  await cdp("Page.addScriptToEvaluateOnNewDocument", {source: "(" + installMock.toString() + ")(" + JSON.stringify(fixtures) + ")"});
  await navigate("checkpoint");
  await until("document.querySelectorAll('.run-card').length === 7 && !document.getElementById('send').disabled");
  assert.equal(await evaluate("document.getElementById('stage').value"), "formal");
  for (const family of ["minifrontier11", "minideepseekv41"]) {
    await evaluate("document.getElementById('family').value = " + JSON.stringify(family) + "; document.getElementById('family').dispatchEvent(new Event('change'))");
    await until("document.querySelectorAll('.run-card').length === 1");
    assert(await evaluate("document.getElementById('selected-summary').textContent.includes(" + JSON.stringify(family === "minifrontier11" ? "MiniFrontier1.1" : "MiniDeepSeek-V4.1") + ")"));
  }
  await evaluate("document.getElementById('family').value = 'all'; document.getElementById('family').dispatchEvent(new Event('change'))");
  await until("document.querySelectorAll('.run-card').length === 7");
  await evaluate("document.querySelector('.run-card').click()");
  await screenshot("checkpoint-desktop");
  await evaluate("document.getElementById('search').value = 'does-not-exist'; document.getElementById('search').dispatchEvent(new Event('input'))");
  assert(await evaluate("document.getElementById('send').disabled && !document.querySelector('.run-card')"));
  await evaluate("document.getElementById('search').value = ''; document.getElementById('search').dispatchEvent(new Event('input')); document.getElementById('send').click()");
  assert(await evaluate("['stage','family','prompt','temperature'].every(id => document.getElementById(id).disabled)"));
  await until("document.querySelector('.result-output') && !document.getElementById('send').disabled");
  assert.equal(await evaluate("document.querySelectorAll('.result-output img').length"), 0);
  assert.equal(await evaluate("demoMock.posts[0].payload.version"), "fixture-version-0");
  await evaluate("document.querySelectorAll('.run-card')[1].click()");
  assert(await evaluate("document.querySelector('.result-meta').textContent.includes('MiniFrontier1.0') && document.getElementById('selected-summary').textContent.includes('MiniDeepSeek')"));
  await evaluate("document.getElementById('send').click()");
  await until("document.querySelectorAll('.result-output').length === 2 && !document.getElementById('send').disabled");
  await evaluate("document.querySelectorAll('.compare-label input').forEach(input => input.click()); document.getElementById('compare').click()");
  assert(await evaluate("!document.getElementById('comparison').hidden && document.getElementById('comparison-note').textContent.includes('相同')"));
  await screenshot("checkpoint-comparison");
  await evaluate("document.getElementById('close-comparison').click(); demoMock.failGenerate = true; document.getElementById('send').click()");
  await until("document.querySelector('.result-output.error') && !document.getElementById('send').disabled");
  assert(await evaluate("document.getElementById('checkpoint-notice').textContent.includes('刷新')"));
  await evaluate("demoMock.failGenerate = false; demoMock.models[1].version = 'updated-version'; document.getElementById('refresh').click()");
  await until("!document.getElementById('refresh').disabled");
  assert(await evaluate("document.getElementById('checkpoint-notice').textContent.includes('已有新权重')"));
  // Resolve the newer list before the older one; the older response must be ignored.
  await evaluate("demoMock.holdLists = true; document.getElementById('stage').value = 'latest'; document.getElementById('stage').dispatchEvent(new Event('change')); document.getElementById('stage').value = 'formal'; document.getElementById('stage').dispatchEvent(new Event('change'))");
  await until("demoMock.waiting.length === 2");
  await evaluate("demoMock.waiting[1](demoMock.models.slice(0, 1)); demoMock.waiting[0](demoMock.models); demoMock.holdLists = false");
  await until("!document.getElementById('refresh').disabled");
  assert.equal(await evaluate("document.querySelectorAll('.run-card').length"), 1);
  await screenshot("checkpoint-mobile", 390, 844);

  await navigate("mf1");
  await until("!document.getElementById('prepare').disabled");
  assert(await evaluate("document.title.includes('MiniFrontier1.1')"));
  assert.equal(await evaluate("document.getElementById('input-mode').value"), "completion");
  // An input edit invalidates an in-flight preparation even if fetch ignores abort.
  await evaluate("demoMock.holdPrepare = true; document.getElementById('prepare').click()");
  await until("demoMock.waiting.length === 1");
  await evaluate("document.getElementById('prompt').value = '新输入'; document.getElementById('prompt').dispatchEvent(new Event('input')); demoMock.waiting[0](); demoMock.holdPrepare = false");
  await delay(80);
  assert(await evaluate("document.getElementById('generate').disabled"));
  await evaluate("document.getElementById('prepare').click()");
  await until("!document.getElementById('generate').disabled");
  await evaluate("document.getElementById('generate').click()");
  assert(await evaluate("document.getElementById('controls').disabled"));
  await until("document.querySelector('#answer .result-output') && !document.getElementById('controls').disabled");
  assert.equal(await evaluate("document.querySelectorAll('#answer .result-output img').length"), 0);
  assert.equal(await evaluate("demoMock.posts.at(-1).payload.checkpoint_sha256"), "a".repeat(64));
  // A tiny image exercises the native browser file input and visible media preview.
  const png = join(temp, "fixture.png");
  await writeFile(png, Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl6AAAAAElFTkSuQmCC", "base64"));
  await evaluate("document.getElementById('input-mode').value = 'chat'; document.getElementById('input-mode').dispatchEvent(new Event('input'))");
  await upload("#images", png);
  await evaluate("document.getElementById('prepare').click()");
  await until("document.querySelector('#media-preview img') && !document.getElementById('generate').disabled");
  assert(await evaluate("document.getElementById('media-preview').textContent.includes('fixture.png')"));
  await screenshot("mf1-desktop");
  await screenshot("mf1-mobile", 390, 844);
  assert.deepEqual(failures, []);
  console.log("PASS: checkpoint selection, request identity, history/compare, version conflicts, stale refresh, MF1 prepare race, SHA binding, image preview, safe output text, desktop/mobile layout.");
  if (screenshotDir) console.log("Screenshots: " + screenshotDir);
} finally {
  if (socket) socket.close();
  if (child && child.exitCode === null) {
    const stopped = new Promise(resolve => child.once("exit", resolve)); child.kill("SIGTERM");
    await Promise.race([stopped, delay(2000)]);
    if (child.exitCode === null) { child.kill("SIGKILL"); await stopped; }
  }
  await rm(temp, {recursive: true, force: true});
}
