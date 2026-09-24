#!/usr/bin/env node
// Dependency-free mock of the Engine Console API (ARCHITECTURE.md §4 + §8 addendum v1.1, response shapes mirror
// backend/src/engine_console/domain/models.py) plus a static host for the frontend.
// Usage: node dev/mock-server.mjs [--port 8791] [--host 127.0.0.1]   (MOCK_KEY=secret to require a bearer key)
// All data is canned/simulated. Nothing here is a verified engine fact.
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { vllmParams, sglangParams, presets, models } from './mock-catalogs.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
const argv = (n, d) => { const i = args.indexOf(`--${n}`); return i >= 0 ? args[i + 1] : d; };
const PORT = Number(argv('port', process.env.PORT ?? 8791));
const HOST = argv('host', '127.0.0.1');
const KEY = process.env.MOCK_KEY || '';
// --proxy http://127.0.0.1:8791 forwards /api/v1 and /metrics to a real backend (dev/screenshot use only), keeping the static host + /__shot helpers.
const PROXY = argv('proxy', process.env.MOCK_PROXY || '');
const G = 2 ** 30;
const now = () => Date.now() / 1000; // epoch seconds, like the backend
const uid = (p) => `${p}${crypto.randomBytes(6).toString('hex')}`;
let seed = 42; const rnd = () => { seed = (seed * 1664525 + 1013904223) % 4294967296; return seed / 4294967296; };

// ---------------- state ----------------
const GPUS = [
  { index: 0, uuid: 'GPU-mock-0000-a', name: 'NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition', total_gib: 95.6, base: 62, util: 40, temp: 58, power: 210 },
  { index: 1, uuid: 'GPU-mock-0000-b', name: 'NVIDIA RTX PRO 6000 Blackwell Workstation Edition', total_gib: 95.6, base: 8, util: 3, temp: 41, power: 62 },
];
const state = {
  instances: [], downloads: [], profiles: [{ id: 'prof_1', name: 'qwen-agent', engine: 'vllm', repo_id: 'Qwen/Qwen3.6-35B-A3B-FP8', description: '', created_at: now() - 5e5, updated_at: now() - 5e5, params: { max_model_len: 65536, enable_auto_tool_choice: true, tool_call_parser: 'hermes', hf_token: '[set]' } }],
  conversations: [], messages: new Map(), prompts: [{ id: 'prm_1', title: 'Terse reviewer', content: 'You are a terse senior code reviewer. Reply with findings only.', tags: [], created_at: now(), updated_at: now() }],
  benches: [], matches: new Map(), ratings: new Map(), audit: [], auditSeq: 0,
  settings: { hf_cache_dir: '/fast/models/hf', default_gpu_ids: [0], idle_ttl_s: 0, image_pins: { vllm: 'vllm/vllm-openai:v0.27.1', sglang: 'lmsysorg/sglang:v0.6.0' }, hf_token: '[set, 37 chars]', docker_context: 'rootless', docker_network: 'ai-lab', port_range: [18000, 18099], hf_allowed_hosts: ['huggingface.co', '*.hf.co'] },
  keys: [{ id: 'k_1', name: 'bootstrap', role: 'admin', prefix: 'ec_3fa9', created_at: now() - 864e3 * 6, last_used_at: now() - 300, secret: null }],
  local: [
    { repo_id: 'Qwen/Qwen3.6-35B-A3B-FP8', size_bytes: 36 * G, revisions: ['main'], last_used: now() - 3600, path: '/fast/models/hf/models--Qwen--Qwen3.6-35B-A3B-FP8' },
    { repo_id: 'openai/gpt-oss-120b', size_bytes: 65 * G, revisions: ['main'], last_used: now() - 864e2 * 2, path: '/fast/models/hf/models--openai--gpt-oss-120b' },
  ],
};
const logs = new Map();
const streams = { metrics: new Set(), downloads: new Set(), logs: new Map(), bench: new Map() };
const history = new Map();
const usageLog = [];

function addLog(id, line) {
  const a = logs.get(id) || []; a.push(`${new Date().toISOString().slice(11, 23)} ${line}`); if (a.length > 2000) a.shift(); logs.set(id, a);
  for (const r of streams.logs.get(id) || []) sse(r, a.at(-1), 'log');
}
function makeInstance({ engine, repo_id, params = {}, gpu_ids = [0], name, profile_id, ready = false, ttl_idle_s }) {
  const id = uid('inst_');
  const idx = state.instances.length;
  const inst = { id, name: name || repo_id.split('/').pop().toLowerCase().slice(0, 24), engine, repo_id, params, gpu_ids, gpu_uuids: gpu_ids.map((g) => GPUS[g]?.uuid), port: 18000 + idx,
    container_name: `ec-${engine}-${id.slice(5, 11)}`, image: state.settings.image_pins[engine], state: 'starting', phase: 'loading_weights', progress_pct: 0, pinned: false, ttl_idle_s: ttl_idle_s ?? null, profile_id: profile_id || null,
    error: null, last_logs: null, fit: null, created_at: now(), started_at: null, last_request_at: null, uptime_s: null };
  state.instances.push(inst);
  addLog(id, `[${engine}] launching ${repo_id} on GPU ${gpu_ids.join(',')} port ${inst.port}`);
  const setReady = (t0) => { inst.state = 'ready'; inst.started_at = t0; inst.phase = 'ready'; inst.progress_pct = 100; };
  if (ready) { setReady(now() - 3600); return inst; }
  const fail = repo_id.includes('FAIL');
  const phases = [['loading_weights', 'Loading weights'], ['compiling', 'torch.compile'], ['capturing_graphs', 'Capturing CUDA graphs']];
  let step = 0, pct = 0;
  const tick = setInterval(() => {
    if (!state.instances.includes(inst) || inst.state === 'stopped' || inst.state === 'stopping') return clearInterval(tick);
    inst.state = 'loading'; pct += 8 + rnd() * 10;
    if (pct >= 100) { pct = 0; step++; }
    if (fail && step === 1) { inst.state = 'failed'; inst.error = 'CUDA out of memory while compiling (mock failure: repo id contains FAIL).'; inst.last_logs = (logs.get(id) || []).slice(-20).join('\n'); addLog(id, 'ERROR torch.OutOfMemoryError: CUDA out of memory'); return clearInterval(tick); }
    if (step >= phases.length) { setReady(now()); addLog(id, 'INFO Application startup complete.'); return clearInterval(tick); }
    inst.phase = phases[step][0]; inst.progress_pct = pct;
    addLog(id, `INFO ${phases[step][1]}… ${Math.round(pct)}%`);
  }, 900);
  return inst;
}
const view = (i) => ({ ...i, uptime_s: i.state === 'ready' && i.started_at ? now() - i.started_at : null });
{
  const i = makeInstance({ engine: 'vllm', repo_id: 'Qwen/Qwen3.6-35B-A3B-FP8', gpu_ids: [0], name: 'qwen-agent', params: { max_model_len: 65536 }, ready: true });
  for (const l of ['INFO Started server process', 'INFO Loading weights took 41.2 s', 'INFO GPU KV cache size: 812,304 tokens', 'INFO Application startup complete.']) addLog(i.id, l);
}

// ---------------- helpers ----------------
const MIME = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.mjs': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8', '.json': 'application/json', '.svg': 'image/svg+xml', '.ico': 'image/x-icon' };
const CSP = "default-src 'self'; connect-src 'self'; img-src 'self' data: blob:; style-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'";
const json = (res, code, obj) => { const b = JSON.stringify(obj); res.writeHead(code, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' }); res.end(b); };
const noContent = (res) => { res.writeHead(204); res.end(); };
const problem = (res, code, title, detail, c) => { const cc = c || title.toLowerCase().replace(/\W+/g, '_'); res.writeHead(code, { 'Content-Type': 'application/problem+json' }); res.end(JSON.stringify({ type: `urn:engine-console:${cc}`, title, status: code, detail, code: cc })); };
function sse(res, data, event) { if (res.writableEnded) return; const p = typeof data === 'string' ? data : JSON.stringify(data); res.write(`${event ? `event: ${event}\n` : ''}${(p.split('\n')).map((l) => `data: ${l}`).join('\n')}\n\n`); }
function openSSE(req, res, set) { res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-store', Connection: 'keep-alive' }); res.write(': ok\n\n'); set?.add(res); req.on('close', () => set?.delete(res)); }
const readBody = (req) => new Promise((ok, no) => { const c = []; let n = 0; req.on('data', (d) => { n += d.length; if (n > 40e6) { no(new Error('too large')); req.destroy(); } else c.push(d); }); req.on('end', () => { const raw = Buffer.concat(c).toString(); if (!raw) return ok({}); try { ok(JSON.parse(raw)); } catch { ok({ __raw: raw }); } }); req.on('error', no); });
const page = (items) => ({ items, next_cursor: null });

function gpuStats() {
  const busy = state.instances.filter((i) => i.state === 'ready' || i.state === 'loading');
  return GPUS.map((g) => {
    const on = busy.some((i) => i.gpu_ids.includes(g.index));
    const used = Math.max(1, g.base + (on ? 0 : -g.base * 0.85) + Math.sin(now() / 4 + g.index) * 0.4);
    return { index: g.index, uuid: g.uuid, name: g.name, total_gib: g.total_gib, free_gib: g.total_gib - used, used_gib: used,
      util_pct: on ? Math.max(2, Math.min(99, g.util + Math.sin(now() / 2.5 + g.index) * 30 + rnd() * 20)) : rnd() * 3,
      temp_c: (on ? g.temp : 34) + Math.sin(now() / 9) * 3 + rnd(), power_w: (on ? g.power : 38) + Math.sin(now() / 3) * 25 + rnd() * 8, fan_pct: on ? 55 : 30, compute_capability: '12.0' };
  });
}

// ---------------- fit estimate (mock; real shape = domain/models.py FitReport) ----------------
const findModel = (repo) => models.find((m) => m.repo_id === repo);
const verdictOf = (r) => (r <= 0.9 ? 'fits' : r <= 1 ? 'tight' : 'wont_fit');
function fitEstimate({ engine, repo_id, params = {}, gpu_ids = [0], concurrency = 1 }) {
  const m = findModel(repo_id);
  if (!m) return { verdict: 'unknown', confidence: 'low', tp: 1, tp_required: null, concurrency, max_len: null, per_gpu: [], max_context_at_current_concurrency: null, max_concurrency_at_current_context: null, fits_if_stop: [], notes: [`No metadata for ${repo_id} in the mock catalog (the real backend reads config.json from the Hub).`], compat: [] };
  const v = engine === 'vllm';
  const tp = Math.max(1, Number(params[v ? 'tensor_parallel_size' : 'tp_size'] || 1));
  const frac = Number(params[v ? 'gpu_memory_utilization' : 'mem_fraction_static'] || (v ? 0.9 : 0.88));
  const len = Number(params[v ? 'max_model_len' : 'context_length'] || Math.min(m.ctx, 32768));
  const seqs = Math.max(1, Number(concurrency) || 1);
  const kvBytes = /fp8/.test(String(params.kv_cache_dtype || '')) ? 1 : 2;
  const eager = !!(params.enforce_eager || params.disable_cuda_graph);
  const compat = [];
  if (tp > gpu_ids.length) compat.push({ level: 'block', code: 'tp_gt_gpus', message: `Tensor parallel ${tp} needs ${tp} GPUs but ${gpu_ids.length} selected. Select more GPUs or lower TP.` });
  if (m.quant === 'nvfp4' && m.is_moe && v) compat.push({ level: 'warn', code: 'nvfp4_moe_sm120', message: 'NVFP4 MoE falls back to a slow Marlin kernel on sm_120 in vLLM. Use FP8 or SGLang.' });
  if (m.quant === 'mxfp4' && m.repo_id !== 'openai/gpt-oss-120b') compat.push({ level: 'block', code: 'mxfp4_moe_sm120', message: 'MXFP4 MoE asserts on sm_120 except gpt-oss-120b.' });
  if (m.gated) compat.push({ level: 'warn', code: 'gated', message: 'Gated model: HF_TOKEN and licence acceptance required to download.' });
  const used = gpu_ids.slice(0, Math.max(tp, 1));
  const perTokB = 2 * m.layers * m.kv_heads * m.head_dim * kvBytes;
  const stats = gpuStats();
  const per_gpu = used.map((gid) => {
    const g = GPUS[gid] || GPUS[0];
    const w = (m.size_bytes / G) / tp, kv = (len * seqs * perTokB) / G / tp, act = 1.6, gr = eager ? 0 : 1.4, ov = 1.2, total = w + kv + act + gr + ov, budget = g.total_gib * frac;
    return { gpu_id: gid, uuid: g.uuid, name: g.name, weights_gib: w, kv_cache_gib: kv, activations_gib: act, cuda_graphs_gib: gr, overhead_gib: ov, total_gib: total, budget_gib: budget, vram_total_gib: g.total_gib, free_gib: stats[gid]?.free_gib ?? g.total_gib, utilization_pct: (total / budget) * 100, verdict: verdictOf(total / budget) };
  });
  const worst = per_gpu.reduce((a, g) => Math.max(a, g.total_gib / g.budget_gib), 0);
  let tp_required = null;
  for (const n of [1, 2, 4, 8]) if ((m.size_bytes / G) / n <= 95.6 * frac * 0.9) { tp_required = n; break; }
  const g0 = per_gpu[0];
  const room = g0 ? Math.max(0, g0.budget_gib - g0.weights_gib - g0.activations_gib - g0.cuda_graphs_gib - g0.overhead_gib) : 0;
  const resident = state.instances.filter((i) => (i.state === 'ready' || i.state === 'loading') && i.gpu_ids.some((x) => gpu_ids.includes(x)));
  const verdict = compat.some((c) => c.level === 'block') ? 'wont_fit' : verdictOf(worst);
  return { verdict, confidence: m.is_moe ? 'medium' : 'high', tp, tp_required, concurrency: seqs, max_len: len, kv_bytes_per_token: perTokB / tp, per_gpu,
    max_context_at_current_concurrency: Math.round(((room * G * tp) / perTokB) / seqs), max_concurrency_at_current_context: Math.round((room * G * tp) / perTokB / len), compat,
    fits_if_stop: verdict === 'wont_fit' && resident.length ? resident.map((i) => i.name) : [],
    notes: [`Mock estimate: KV = 2 x layers x kv_heads x head_dim x ${kvBytes} B x ${len.toLocaleString('en-US')} tokens x ${seqs} seqs.`] };
}

// ---------------- metrics ----------------
function sample(inst, t = now()) {
  const busy = inst.state === 'ready';
  const w = Math.sin(t / 15) * 0.5 + 0.5;
  const running = busy ? Math.round(2 + w * 10 + rnd() * 3) : 0;
  return { t, values: busy ? { requests_running: running, requests_waiting: Math.max(0, Math.round((w - 0.6) * 12 + rnd())), kv_cache_usage_pct: 8 + w * 60 + rnd() * 5,
    prefix_cache_hit_pct: 40 + w * 30 + rnd() * 5, generation_tps: running * (55 + rnd() * 12), prompt_tps: 300 + w * 3200 + rnd() * 200,
    ttft_p50_s: 0.08 + w * 0.15 + rnd() * 0.02, ttft_p95_s: 0.3 + w * 0.5 + rnd() * 0.05, itl_p50_s: 0.014 + w * 0.01, e2e_p50_s: 2 + w * 3, preemptions_total: 0 } : {} };
}
for (let k = 0; k < 450; k++) for (const i of state.instances) { const a = history.get(i.id) || []; a.push(sample(i, now() - (450 - k) * 2)); history.set(i.id, a); }
setInterval(() => {
  for (const i of state.instances) {
    const a = history.get(i.id) || []; const s = sample(i); a.push(s); if (a.length > 45000) a.shift(); history.set(i.id, a);
    if (i.state === 'ready') for (const r of streams.metrics) sse(r, { instance_id: i.id, ...s }, 'metrics');
  }
}, 2000);
setInterval(() => { for (const i of state.instances) if (i.state === 'ready' && rnd() > 0.4) addLog(i.id, `INFO Avg generation throughput: ${(300 + rnd() * 500).toFixed(1)} tokens/s, Running: ${Math.round(rnd() * 8)} reqs, KV cache usage: ${(rnd() * 40).toFixed(1)}%`); }, 1800);
setInterval(() => {
  for (const d of state.downloads) {
    if (d.state === 'queued' && state.downloads.filter((x) => x.state === 'running').length < 2) d.state = 'running';
    if (d.state !== 'running') continue;
    d.speed_bps = (180 + rnd() * 120) * 2 ** 20; d.done_bytes = Math.min(d.total_bytes, d.done_bytes + d.speed_bps * 0.5);
    d.files_done = Math.floor((d.done_bytes / d.total_bytes) * d.files_total); d.eta_s = (d.total_bytes - d.done_bytes) / d.speed_bps; d.updated_at = now();
    if (d.done_bytes >= d.total_bytes) { d.state = 'completed'; d.speed_bps = 0; d.eta_s = 0; if (!state.local.some((l) => l.repo_id === d.repo_id)) state.local.push({ repo_id: d.repo_id, size_bytes: d.total_bytes, revisions: ['main'], last_used: null, path: `/fast/models/hf/${d.repo_id}` }); }
    for (const r of streams.downloads) sse(r, d, 'progress');
  }
}, 500);

// ---------------- benchmark (real shape: results = {concurrency:[level], single, prefix_cache}) ----------------
const SUITES = { quick: { concurrency: [1, 4], prompt_tokens: 512, max_tokens: 128, rounds: 1, prefix_cache: false }, standard: { concurrency: [1, 4, 16, 64], prompt_tokens: 1024, max_tokens: 256, rounds: 2, prefix_cache: true } };
function runBench(inst, suite) {
  const spec = typeof suite === 'string' ? SUITES[suite] : { ...SUITES.quick, ...suite };
  const b = { id: uid('bench_'), instance_id: inst.id, engine: inst.engine, repo_id: inst.repo_id, profile_id: inst.profile_id, params: inst.params, suite: spec, state: 'running', results: {}, error: null, created_at: now(), finished_at: null };
  state.benches.unshift(b);
  const emit = (ev) => { for (const r of streams.bench.get(b.id) || []) sse(r, { id: b.id, ...ev }, 'progress'); };
  const k = (inst.repo_id.length % 7) / 20 + 0.9 * (inst.engine === 'sglang' ? 1.06 : 1);
  const levels = []; let i = 0;
  const step = () => {
    if (i >= spec.concurrency.length) {
      b.results = { concurrency: levels, single: (({ concurrency, throughput_tps, ...r }) => r)(levels[0]), ...(spec.prefix_cache ? { prefix_cache: { cold_ttft_s: 0.9 / k, warm_ttft_s: 0.12 / k, speedup: 7.5 } } : {}) };
      b.state = 'completed'; b.finished_at = now(); emit({ event: 'done', state: 'completed' });
      for (const r of streams.bench.get(b.id) || []) r.end();
      return;
    }
    const c = spec.concurrency[i++]; emit({ event: 'level_start', concurrency: c });
    setTimeout(() => {
      const level = { concurrency: c, requests: c * spec.rounds, errors: 0, ttft_p50_s: 0.11 / k * (1 + c / 40), ttft_p95_s: 0.29 / k * (1 + c / 30), itl_p50_s: 0.0108 / k, itl_p95_s: 0.02 / k, e2e_p50_s: 2.4 / k, e2e_p95_s: 3.1 / k,
        decode_tps: 92 * k * (1 - Math.log2(c + 1) * 0.04), prefill_tps: 5200 * k + rnd() * 300, throughput_tps: 92 * k * c * (1 - Math.log2(c + 1) * 0.06) + rnd() * 20 };
      levels.push(level); emit({ event: 'level_done', level }); step();
    }, 1200);
  };
  step();
  return b;
}

// ---------------- chat ----------------
function reply(model, text, hasImage) {
  const reasoning = `The user asks: "${text.slice(0, 80)}". Let me think about the best structure: a short answer, one code sample, then caveats.`;
  const body = `${hasImage ? '_I can see the attached image._\n\n' : ''}Here is a response from **${model}** (mock stream).\n\n### Key points\n\n- Markdown is rendered *safely*: \`<script>alert(1)</script>\` stays inert text.\n- Lists, **bold**, \`inline code\` and [links](https://example.com) work.\n\n\`\`\`python\ndef fib(n: int) -> int:\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n\`\`\`\n\n| Engine | Prefill | Decode |\n|---|---:|---:|\n| vLLM | 5.2k | 92 |\n| SGLang | 5.5k | 97 |\n\n> Numbers above are fabricated by the mock server.\n`;
  return { reasoning, body };
}
const userText = (msgs) => { const last = [...(msgs || [])].reverse().find((m) => m.role === 'user'); return { text: typeof last?.content === 'string' ? last.content : (last?.content || []).filter((p) => p.type === 'text').map((p) => p.text).join(' '), hasImage: Array.isArray(last?.content) && last.content.some((p) => p.type === 'image_url'), raw: last?.content }; };
const addMsg = (cid, m) => { const a = state.messages.get(cid) || []; a.push({ ts: now(), reasoning: null, usage: null, ...m }); state.messages.set(cid, a); const c = state.conversations.find((x) => x.id === cid); if (c) c.updated_at = now(); };
async function chat(req, res, b) {
  const inst = state.instances.find((i) => i.id === b.instance_id);
  if (!inst) return problem(res, 404, 'Not found', `no such instance: ${b.instance_id}`, 'instance_not_found');
  if (inst.state !== 'ready') return problem(res, 409, 'Instance not ready', `instance ${inst.name} is ${inst.state}`, 'instance_not_ready');
  const u = userText(b.messages);
  const { reasoning, body } = reply(inst.name, u.text, u.hasImage);
  const toks = (s) => s.match(/\s*\S+/g) || [];
  const usage = { prompt_tokens: Math.round(u.text.length / 4) + 12, completion_tokens: toks(body).length };
  if (b.stream === false || b.stream === undefined) { addUsage(inst, usage); return json(res, 200, { choices: [{ index: 0, message: { role: 'assistant', content: body, reasoning_content: reasoning }, finish_reason: 'stop' }], usage }); }
  openSSE(req, res);
  let closed = false; req.on('close', () => { closed = true; });
  const chunk = (delta) => sse(res, { choices: [{ index: 0, delta }] });
  for (const t of toks(reasoning)) { if (closed) break; chunk({ reasoning_content: t }); await new Promise((r) => setTimeout(r, 12)); }
  for (const t of toks(body)) { if (closed) break; chunk({ content: t }); await new Promise((r) => setTimeout(r, 18)); }
  if (!closed) { sse(res, { choices: [{ index: 0, delta: {}, finish_reason: 'stop' }], usage }); res.write('data: [DONE]\n\n'); res.end(); }
  addUsage(inst, usage);
  if (b.conversation_id && state.messages.has(b.conversation_id) !== undefined) { addMsg(b.conversation_id, { role: 'user', content: typeof u.raw === 'string' ? u.raw : JSON.stringify(u.raw) }); addMsg(b.conversation_id, { role: 'assistant', content: body, reasoning, usage }); }
}
function addUsage(inst, u) { usageLog.push({ ts: now(), model: inst.repo_id, p: u.prompt_tokens, c: u.completion_tokens, ms: 2400 }); }

// ---------------- usage ----------------
function usageRows(group) {
  const ms = ['Qwen/Qwen3.6-35B-A3B-FP8', 'openai/gpt-oss-120b', 'google/gemma-4-31b-it'];
  if (group === 'hour') { const q = (() => { let x = 7; return () => (x = (x * 9301 + 49297) % 233280) / 233280; })(); const rows = []; for (let d = 0; d < 7; d++) for (let h = 0; h < 24; h++) { const t = new Date(Date.now() - ((6 - d) * 24 + (23 - h)) * 36e5); rows.push({ key: `${t.toISOString().slice(0, 13)}:00Z`, requests: Math.round(h >= 8 && h <= 19 && ![0, 6].includes(t.getUTCDay()) ? 40 + q() * 90 : q() * 25), prompt_tokens: 1000, completion_tokens: 500, avg_latency_ms: 2100 }); } return rows; }
  if (group === 'day') return Array.from({ length: 14 }, (_, i) => ({ key: new Date(Date.now() - (13 - i) * 864e5).toISOString().slice(0, 10), requests: 80 + ((i * 37) % 90), prompt_tokens: 120000 + i * 9000, completion_tokens: 60000 + ((i * 5300) % 40000), avg_latency_ms: 2100 }));
  const extra = usageLog.reduce((a, u) => { a[u.model] = a[u.model] || { r: 0, p: 0, c: 0 }; a[u.model].r++; a[u.model].p += u.p; a[u.model].c += u.c; return a; }, {});
  return ms.map((m, i) => ({ key: m, requests: 900 - i * 260 + (extra[m]?.r || 0), prompt_tokens: 2.4e6 - i * 7e5 + (extra[m]?.p || 0), completion_tokens: 1.1e6 - i * 3e5 + (extra[m]?.c || 0), avg_latency_ms: 2000 + i * 900 }));
}

// ---------------- router ----------------
const SAFE = new Set(['GET', 'HEAD']);
const SECRET_KEY = /token|secret|password|api[_-]?key/i;
const redactParams = (p) => Object.fromEntries(Object.entries(p || {}).map(([k, v]) => [k, SECRET_KEY.test(k) ? '[set]' : v]));
let openStreams = 0;
async function api(req, res, url) {
  // DNS-rebinding guard (v1.1): only loopback Host headers are served.
  const host = (req.headers.host || '').replace(/:\d+$/, '');
  if (!['127.0.0.1', 'localhost', '[::1]'].includes(host)) return problem(res, 421, 'Misdirected Request', 'unexpected Host header', 'bad_host');
  // SSE cap (v1.1): the server holds at most 8 live streams and answers 429 too_many_streams beyond that.
  if (/\/(stream)$/.test(url.pathname) && url.searchParams.get('once') !== 'true') {
    if (openStreams >= 8) return problem(res, 429, 'Too Many Requests', 'too many open streams (max 8)', 'too_many_streams');
    openStreams++; res.on('close', () => { openStreams--; });
  }
  const p = url.pathname.replace(/^\/api\/v1/, '') || '/';
  const q = url.searchParams, M = req.method, once = q.get('once') === 'true';
  if (p === '/health') return json(res, 200, { status: 'ok', mock: true, version: 'mock', engines: ['vllm', 'sglang'], docker_context: 'rootless' });
  if (KEY && req.headers.authorization !== `Bearer ${KEY}`) return problem(res, 401, 'Unauthorized', 'a valid bearer API key is required', 'unauthorized');
  // CSRF / DNS-rebinding guard, same as the real backend: mutating calls must carry X-Engine-Console.
  if (!SAFE.has(M) && !req.headers['x-engine-console']) return problem(res, 403, 'Forbidden', 'missing X-Engine-Console header', 'csrf_header_required');
  const body = !SAFE.has(M) ? await readBody(req).catch(() => ({})) : {};
  if (!SAFE.has(M) && body.__raw !== undefined && p !== '/profiles/import') return problem(res, 415, 'Unsupported Media Type', 'body must be JSON', 'unsupported_media_type');
  if (!SAFE.has(M)) state.audit.unshift({ id: ++state.auditSeq, ts: now(), actor: KEY ? 'key:mock' : 'loopback', role: 'admin', method: M, path: url.pathname, status: 200, params: { body: JSON.parse(JSON.stringify(body, (k, v) => (/token|secret|password|api[_-]?key/i.test(k) ? '[redacted]' : v))) } });
  let m;
  if (p === '/hardware' && M === 'GET') return json(res, 200, { gpus: gpuStats(), host_ram_gib: 246, host_ram_free_gib: 171, source: 'mock' });
  if (p === '/engines') return json(res, 200, [
    { id: 'vllm', display_name: 'vLLM', default_image: state.settings.image_pins.vllm, presets: presets.vllm },
    { id: 'sglang', display_name: 'SGLang', default_image: state.settings.image_pins.sglang, presets: presets.sglang }]);
  if ((m = p.match(/^\/engines\/(\w+)\/params$/))) return m[1] === 'vllm' ? json(res, 200, vllmParams) : m[1] === 'sglang' ? json(res, 200, sglangParams) : problem(res, 404, 'Not found', `unknown engine ${m[1]}`);
  if (p === '/hf/search') {
    let r = models.filter((x) => (!q.get('q') || q.get('q').toLowerCase().split(/\s+/).every((w) => x.repo_id.toLowerCase().includes(w) || x.quant.includes(w))) && (!q.get('task') || x.pipeline_tag === q.get('task')) && (!q.get('quant') || x.quant === q.get('quant'))
      && (!q.get('max_gib') || x.size_bytes / G <= Number(q.get('max_gib'))) && (!q.get('gated') || String(x.gated) === q.get('gated')));
    const s = q.get('sort') || 'downloads'; r = r.sort((a, b) => (s === 'likes' ? b.likes - a.likes : s === 'updated' ? b.repo_id.localeCompare(a.repo_id) : b.downloads - a.downloads));
    await new Promise((ok) => setTimeout(ok, 250));
    return json(res, 200, page(r.slice(0, Number(q.get('limit') || 30)).map((x) => ({ repo_id: x.repo_id, downloads: x.downloads, likes: x.likes, pipeline_tag: x.pipeline_tag, library_name: 'transformers', gated: x.gated, tags: [x.quant, x.pipeline_tag], num_params: x.num_params, approx_size_gib: Math.round((x.size_bytes / G) * 100) / 100, quantization: x.quant === 'bf16' ? null : x.quant }))));
  }
  if ((m = p.match(/^\/hf\/models\/(.+)$/)) && M === 'GET') {
    const id = decodeURIComponent(m[1]), x = findModel(id);
    if (!x) return problem(res, 404, 'Not found', `no such model: ${id}`, 'hf_not_found');
    await new Promise((ok) => setTimeout(ok, 200));
    return json(res, 200, { info: { repo_id: id, revision: null, architectures: [x.arch], model_type: x.arch.replace(/For.*/, '').toLowerCase(), num_params: x.num_params, num_active_params: x.active || null, is_moe: !!x.is_moe, weight_bytes: x.size_bytes, quantization: x.quant === 'bf16' ? null : x.quant, dtype: 'bfloat16',
      num_layers: x.layers, num_kv_heads: x.kv_heads, head_dim: x.head_dim, max_position_embeddings: x.ctx, gated: x.gated, license: x.license, pipeline_tag: x.pipeline_tag, raw_config: {} }, sha: 'deadbeef',
      card: `# ${id}\n\nMock model card. **${x.arch}**, license \`${x.license}\`.\n\n## Usage\n\n\`\`\`bash\nvllm serve ${id}\n\`\`\`\n\n## Notes\n\n- <script>alert('xss')</script> would be escaped here.\n- See [the docs](https://huggingface.co/${id}).\n`,
      files: [{ path: 'config.json', size: 2100 }, ...Array.from({ length: 4 }, (_, i) => ({ path: `model-0000${i + 1}-of-00004.safetensors`, size: Math.round(x.size_bytes / 4) })), { path: 'tokenizer.json', size: 11e6 }] });
  }
  if (p === '/fit' && M === 'POST') { await new Promise((ok) => setTimeout(ok, 120)); return json(res, 200, fitEstimate(body)); }
  if (p === '/downloads/stream') { openSSE(req, res, once ? null : streams.downloads); sse(res, state.downloads, 'snapshot'); if (once) res.end(); return; }
  if (p === '/downloads' && M === 'GET') return json(res, 200, page(state.downloads));
  if (p === '/downloads' && M === 'POST') {
    const x = findModel(body.repo_id); if (!body.repo_id) return problem(res, 422, 'Validation error', 'repo_id required', 'validation_error');
    const d = { id: uid('dl_'), repo_id: body.repo_id, revision: null, commit_sha: null, allow_patterns: null, state: 'queued', total_bytes: x?.size_bytes || 8 * G, done_bytes: 0, speed_bps: 0, eta_s: null, files_total: 5, files_done: 0, current_file: null, error: null, error_code: null, created_at: now(), updated_at: now() };
    if (x?.gated && !state.settings.hf_token) { d.state = 'failed'; d.error = 'Gated model: set HF_TOKEN and accept the licence.'; }
    state.downloads.unshift(d); return json(res, 201, d);
  }
  if ((m = p.match(/^\/downloads\/([^/]+)(?:\/(pause|resume|cancel))?$/))) {
    const d = state.downloads.find((x) => x.id === m[1]); if (!d) return problem(res, 404, 'Not found', `no such download: ${m[1]}`);
    if (M === 'DELETE') { state.downloads = state.downloads.filter((x) => x !== d); return noContent(res); }
    if (m[2]) { d.state = m[2] === 'pause' ? 'paused' : m[2] === 'resume' ? 'running' : 'cancelled'; d.updated_at = now(); for (const r of streams.downloads) sse(r, d, 'progress'); }
    return json(res, 200, d);
  }
  if (p === '/models' && M === 'GET') return json(res, 200, page(state.local.map((l) => ({ ...l, engines_that_fit: ['vllm', 'sglang'].filter((e) => ['fits', 'tight'].includes(fitEstimate({ engine: e, repo_id: l.repo_id, params: {}, gpu_ids: [0] }).verdict)) }))));
  if ((m = p.match(/^\/models\/(.+)$/)) && M === 'DELETE') { state.local = state.local.filter((l) => l.repo_id !== decodeURIComponent(m[1])); return noContent(res); }
  if (p === '/instances' && M === 'GET') return json(res, 200, page(state.instances.map(view)));
  if (p === '/instances/preflight' && M === 'POST') {
    const f = fitEstimate(body); const checks = [...f.compat.map((c) => ({ code: c.code, level: c.level, message: c.message })), ...(f.verdict === 'wont_fit' && !f.compat.some((c) => c.level === 'block') ? [{ code: 'fit', level: 'warn', message: 'The estimate says this will not fit in memory.' }] : []), { code: 'port', level: 'ok', message: 'A host port is free' }];
    return json(res, 200, { ok: !checks.some((c) => c.level === 'block'), gpu_ids: body.gpu_ids || [], checks, fit: f });
  }
  if (p === '/instances' && M === 'POST') {
    if (body.engine === 'sglang' && body.params?.api_key) return problem(res, 422, 'Unprocessable', 'SGLang can only take api_key on its command line, which would expose the secret in the process list. Remove api_key, or use an engine that reads it from the environment.', 'secret_in_argv');
    if (!body.engine || !body.repo_id) return problem(res, 422, 'Validation error', 'engine and repo_id required', 'validation_error');
    const f = fitEstimate(body); if (f.compat.some((c) => c.level === 'block')) return problem(res, 409, 'Preflight failed', f.compat.find((c) => c.level === 'block').message, 'preflight_blocked');
    return json(res, 201, view(makeInstance({ ...body, params: redactParams(body.params) })));
  }
  if ((m = p.match(/^\/instances\/([^/]+)(?:\/(stop|start|restart|logs|command))?(?:\/(stream))?$/))) {
    const inst = state.instances.find((x) => x.id === m[1]); if (!inst) return problem(res, 404, 'Not found', `no such instance: ${m[1]}`, 'instance_not_found');
    if (m[2] === 'logs' && m[3] === 'stream') {
      const set = streams.logs.get(inst.id) || new Set(); streams.logs.set(inst.id, set); openSSE(req, res, once ? null : set);
      for (const l of (logs.get(inst.id) || []).slice(-Number(q.get('tail') || 100))) sse(res, l, 'log');
      if (once) res.end(); return;
    }
    if (m[2] === 'logs') { res.writeHead(200, { 'Content-Type': 'text/plain; charset=utf-8' }); return res.end((logs.get(inst.id) || []).slice(-Number(q.get('tail') || 200)).join('\n')); }
    if (m[2] === 'command') { const flags = Object.entries(inst.params).map(([k, v]) => `--${k.replace(/_/g, '-')}${typeof v === 'boolean' ? '' : ' ' + JSON.stringify(v)}`).join(' \\\n  ');
      return json(res, 200, { docker_run: `docker run -d --gpus '"device=${inst.gpu_ids.join(',')}"' -p 127.0.0.1:${inst.port}:8000 \\\n  ${inst.image} \\\n  --model ${inst.repo_id} ${flags}`, compose_yaml: `services:\n  ${inst.name}:\n    image: ${inst.image}\n    command: ["--model", "${inst.repo_id}"]\n`, engine_cli: `${inst.engine === 'vllm' ? 'vllm serve' : 'python -m sglang.launch_server --model-path'} ${inst.repo_id} ${flags}` }); }
    if (M === 'GET') return json(res, 200, view(inst));
    if (M === 'PATCH') { for (const k of ['pinned', 'ttl_idle_s', 'name']) if (body[k] !== undefined && body[k] !== null) inst[k] = body[k]; return json(res, 200, view(inst)); }
    if (M === 'DELETE') { state.instances = state.instances.filter((x) => x !== inst); return noContent(res); }
    if (m[2] === 'stop') { inst.state = 'stopped'; inst.started_at = null; addLog(inst.id, 'INFO stopped by user'); return json(res, 200, view(inst)); }
    if (m[2] === 'start' || m[2] === 'restart') { Object.assign(inst, { state: 'loading', error: null, phase: 'loading_weights', progress_pct: 40 }); addLog(inst.id, `INFO ${m[2]} requested`); setTimeout(() => { if (inst.state === 'loading') { inst.state = 'ready'; inst.started_at = now(); inst.phase = 'ready'; inst.progress_pct = 100; } }, 3500); return json(res, 200, view(inst)); }
  }
  if (p === '/profiles' && M === 'GET') return json(res, 200, page(state.profiles));
  if (p === '/profiles' && M === 'POST') { const pr = { id: uid('prof_'), name: body.name, engine: body.engine, repo_id: body.repo_id ?? null, params: redactParams(body.params), description: body.description || '', created_at: now(), updated_at: now() }; state.profiles.push(pr); return json(res, 201, pr); }
  if (p === '/profiles/import' && M === 'POST') {
    const yaml = typeof body.yaml === 'string' ? body.yaml : body.__raw || '';
    const nm = /^name:\s*(.+)$/m.exec(yaml); if (!nm) return problem(res, 400, 'Bad request', 'profile YAML needs a name: line', 'invalid_profile');
    const pr = { id: uid('prof_'), name: nm[1].trim(), engine: /^engine:\s*(\w+)/m.exec(yaml)?.[1] || 'vllm', repo_id: null, params: {}, description: '', created_at: now(), updated_at: now() }; state.profiles.push(pr); return json(res, 201, pr);
  }
  if ((m = p.match(/^\/profiles\/([^/]+)\/export$/))) { const pr = state.profiles.find((x) => x.id === m[1]); if (!pr) return problem(res, 404, 'Not found', m[1]); res.writeHead(200, { 'Content-Type': 'application/yaml' }); return res.end(`name: ${pr.name}\nengine: ${pr.engine}\nparams: ${JSON.stringify(pr.params)}\n`); }
  if ((m = p.match(/^\/profiles\/([^/]+)$/))) { const pr = state.profiles.find((x) => x.id === m[1]); if (!pr) return problem(res, 404, 'Not found', `no such profile: ${m[1]}`); if (M === 'PUT') Object.assign(pr, body, { updated_at: now() }); if (M === 'DELETE') { state.profiles = state.profiles.filter((x) => x !== pr); return noContent(res); } return json(res, 200, pr); }
  if ((m = p.match(/^\/metrics\/instances\/([^/]+)$/))) { const secs = { '5m': 300, '15m': 900, '1h': 3600, '24h': 86400 }[q.get('window') || '15m'] || 900; const cut = now() - secs; const h = (history.get(m[1]) || []).filter((s) => s.t >= cut && Object.keys(s.values).length); const step = Math.max(1, Math.ceil(h.length / 300)); return json(res, 200, { instance_id: m[1], window: q.get('window') || '15m', resolution: 'raw', points: h.filter((_, i) => i % step === 0) }); }
  if (p === '/metrics/stream') { openSSE(req, res, once ? null : streams.metrics); for (const i of state.instances.filter((x) => x.state === 'ready')) { const s = history.get(i.id)?.at(-1); if (s) sse(res, { instance_id: i.id, ...s }, 'metrics'); } if (once) res.end(); return; }
  if (p === '/bench' && M === 'POST') { const inst = state.instances.find((i) => i.id === body.instance_id); if (!inst || inst.state !== 'ready') return problem(res, 409, 'Instance not ready', 'benchmarks need a ready instance', 'instance_not_ready'); if (typeof body.suite === 'string' && !SUITES[body.suite]) return problem(res, 400, 'Bad request', `unknown suite '${body.suite}'`, 'unknown_suite'); res.writeHead(202, { 'Content-Type': 'application/json' }); return res.end(JSON.stringify(runBench(inst, body.suite || 'quick'))); }
  if (p === '/bench' && M === 'GET') return json(res, 200, page(state.benches));
  if ((m = p.match(/^\/bench\/([^/]+)(\/stream)?$/))) {
    const b = state.benches.find((x) => x.id === m[1]); if (!b) return problem(res, 404, 'Not found', `no such bench run: ${m[1]}`);
    if (m[2]) { const live = b.state === 'running' && !once; const set = streams.bench.get(b.id) || new Set(); streams.bench.set(b.id, set); openSSE(req, res, live ? set : null); sse(res, b, 'snapshot'); if (!live) res.end(); return; }
    return json(res, 200, b);
  }
  if (p === '/chat/completions' && M === 'POST') return chat(req, res, body);
  if (p === '/conversations' && M === 'GET') return json(res, 200, page(state.conversations.slice().sort((a, b) => b.updated_at - a.updated_at)));
  if (p === '/conversations' && M === 'POST') { const c = { id: uid('conv_'), title: body.title || 'New chat', system_prompt: body.system_prompt ?? null, instance_id: body.instance_id ?? null, created_at: now(), updated_at: now(), messages: null }; state.conversations.unshift(c); state.messages.set(c.id, []); return json(res, 201, c); }
  if ((m = p.match(/^\/conversations\/([^/]+)$/))) { const c = state.conversations.find((x) => x.id === m[1]); if (!c) return problem(res, 404, 'Not found', `no such conversation: ${m[1]}`); if (M === 'DELETE') { state.conversations = state.conversations.filter((x) => x !== c); return noContent(res); } return json(res, 200, { ...c, messages: state.messages.get(c.id) || [] }); }
  if (p === '/prompts' && M === 'GET') return json(res, 200, page(state.prompts));
  if (p === '/prompts' && M === 'POST') { const pr = { id: uid('prm_'), title: body.title, content: body.content, tags: body.tags || [], created_at: now(), updated_at: now() }; state.prompts.push(pr); return json(res, 201, pr); }
  if ((m = p.match(/^\/prompts\/([^/]+)$/))) { const pr = state.prompts.find((x) => x.id === m[1]); if (!pr) return problem(res, 404, 'Not found', `no such prompt: ${m[1]}`); if (M === 'DELETE') { state.prompts = state.prompts.filter((x) => x !== pr); return noContent(res); } Object.assign(pr, body, { updated_at: now() }); return json(res, 200, pr); }
  if (p === '/arena/matches' && M === 'POST') {
    const ia = state.instances.find((i) => i.id === body.instance_a), ib = state.instances.find((i) => i.id === body.instance_b);
    if (body.instance_a === body.instance_b) return problem(res, 400, 'Bad request', 'pick two different instances', 'same_instance');
    if (!ia || !ib || ia.state !== 'ready' || ib.state !== 'ready') return problem(res, 409, 'Instance not ready', 'both instances must be ready', 'instance_not_ready');
    await new Promise((ok) => setTimeout(ok, 1500));
    const mt = { id: uid('match_'), prompt: body.prompt, blind: body.blind !== false, response_a: reply(ia.name, body.prompt, false).body, response_b: reply(ib.name, body.prompt, false).body.replace('Here is a response', 'Sure. Here is a shorter response'), winner: null, model_a: ia.repo_id, model_b: ib.repo_id };
    state.matches.set(mt.id, mt); return json(res, 201, { ...mt, model_a: mt.blind ? null : mt.model_a, model_b: mt.blind ? null : mt.model_b });
  }
  if ((m = p.match(/^\/arena\/matches\/([^/]+)\/vote$/)) && M === 'POST') {
    const mt = state.matches.get(m[1]); if (!mt) return problem(res, 404, 'Not found', `no such match: ${m[1]}`);
    if (!['a', 'b', 'tie'].includes(body.winner)) return problem(res, 422, 'Validation error', 'winner must be a, b or tie', 'validation_error');
    if (mt.winner) return problem(res, 409, 'Conflict', 'this match already has a vote', 'already_voted');
    mt.winner = body.winner;
    const R = (id) => state.ratings.get(id) || { model: id, rating: 1000, games: 0, wins: 0, losses: 0, ties: 0 };
    const A = R(mt.model_a), B = R(mt.model_b), sa = { a: 1, b: 0, tie: 0.5 }[body.winner], ea = 1 / (1 + 10 ** ((B.rating - A.rating) / 400));
    A.rating += 32 * (sa - ea); B.rating -= 32 * (sa - ea); A.games++; B.games++; if (sa === 1) { A.wins++; B.losses++; } else if (sa === 0) { B.wins++; A.losses++; } else { A.ties++; B.ties++; }
    state.ratings.set(A.model, A); state.ratings.set(B.model, B);
    return json(res, 200, { ...mt, ratings: { [A.model]: Math.round(A.rating * 10) / 10, [B.model]: Math.round(B.rating * 10) / 10 } });
  }
  if (p === '/arena/leaderboard') return json(res, 200, [...state.ratings.values()].sort((x, y) => y.rating - x.rating).map((r) => ({ ...r, rating: Math.round(r.rating * 10) / 10 })));
  if (p === '/usage/export.csv') { res.writeHead(200, { 'Content-Type': 'text/csv' }); return res.end(['key,requests,prompt_tokens,completion_tokens,avg_latency_ms', ...usageRows(q.get('group_by') || 'model').map((r) => `${r.key},${r.requests},${r.prompt_tokens},${r.completion_tokens},${r.avg_latency_ms}`)].join('\n')); }
  if (p === '/usage') return json(res, 200, usageRows(q.get('group_by') || 'model'));
  if (p === '/audit') return json(res, 200, page(state.audit.slice(0, Number(q.get('limit') || 100))));
  if (p === '/settings') { if (M === 'PUT') Object.assign(state.settings, Object.fromEntries(Object.entries(body).filter(([k]) => ['default_gpu_ids', 'idle_ttl_s'].includes(k)))); return json(res, 200, state.settings); }
  if (p === '/keys' && M === 'GET') return json(res, 200, state.keys);
  if (p === '/keys' && M === 'POST') { const secret = `ec_${crypto.randomBytes(24).toString('base64url')}`; const k = { id: uid('k_'), name: body.name, role: body.role || 'viewer', prefix: secret.slice(0, 7), created_at: now(), last_used_at: null, secret: null }; state.keys.push(k); return json(res, 201, { ...k, secret }); }
  if ((m = p.match(/^\/keys\/([^/]+)$/)) && M === 'DELETE') { state.keys = state.keys.filter((k) => k.id !== m[1]); return noContent(res); }
  return problem(res, 404, 'Not found', `${M} ${p}`, 'not_found');
}

// Dev-only helpers for headless screenshots: /__shot serves index.html plus a slow 1px image so the browser's
// `load` event (when screenshot tools fire) is held until the SPA has finished its async rendering.
function serveShot(res, url) {
  const ms = Math.min(15000, Number(url.searchParams.get('ms') || 4000));
  if (url.pathname === '/__demo.js') {
    const name = url.searchParams.get('name');
    const W = `const until=async(f)=>{for(let i=0;i<80;i++){const v=f();if(v)return v;await new Promise(r=>setTimeout(r,100));}};`;
    const demos = {
      chat: W + `const ta=await until(()=>document.querySelector('ec-chat .composer textarea')); await until(()=>document.querySelector('ec-chat select option[value^=inst]')); await new Promise(r=>setTimeout(r,300)); ta.value='Show me a fibonacci function and a comparison table.'; document.querySelector('ec-chat .composer .btn.primary').click();`,
      wontfit: W + `const b=await until(()=>[...document.querySelectorAll('ec-launch .btn')].find(x=>x.textContent.trim()==='long-context')); b.click();`,
      palette: W + `await until(()=>document.querySelector('ec-palette')); await new Promise(r=>setTimeout(r,800)); window.dispatchEvent(new Event('ec:palette')); await new Promise(r=>setTimeout(r,600)); const i=document.querySelector('ec-palette input'); i.value='chat'; i.dispatchEvent(new Event('input'));`,
      arena: W + `const ta=await until(()=>document.querySelector('ec-arena textarea')); ta.value='Explain KV cache in two sentences.'; [...document.querySelectorAll('ec-arena .btn')].find(x=>x.textContent.includes('Run match')).click();`,
    };
    res.writeHead(200, { 'Content-Type': 'text/javascript' }); return res.end(demos[name] || '');
  }
  if (url.pathname === '/__slow.gif') return setTimeout(() => { res.writeHead(200, { 'Content-Type': 'image/gif' }); res.end(Buffer.from('R0lGODlhAQABAAAAACw=', 'base64')); }, ms);
  // ?demo=<name> adds a same-origin script (CSP-safe) that drives the UI, e.g. sends a chat message.
  const demo = /^\w+$/.test(url.searchParams.get('demo') || '') ? `<script type="module" src="/__demo.js?name=${url.searchParams.get('demo')}"></script>` : '';
  const html = fs.readFileSync(path.join(ROOT, 'index.html'), 'utf8').replace('</body>', `<img src="/__slow.gif?ms=${ms}" width="1" height="1" alt="">${demo}</body>`);
  res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8', 'Content-Security-Policy': CSP }); res.end(html);
}

function serveStatic(req, res, url) {
  let rel = decodeURIComponent(url.pathname);
  if (rel === '/') rel = '/index.html';
  const abs = path.resolve(ROOT, '.' + rel);
  // Containment: never serve outside the frontend root, and keep dev/ + tests/ private.
  if (!abs.startsWith(ROOT + path.sep) || /^\/(dev|tests)\//.test(rel)) { res.writeHead(404); return res.end('not found'); }
  fs.readFile(abs, (err, buf) => {
    if (err) { res.writeHead(404); return res.end('not found'); }
    res.writeHead(200, { 'Content-Type': MIME[path.extname(abs)] || 'application/octet-stream', 'Content-Security-Policy': CSP, 'X-Content-Type-Options': 'nosniff', 'Cache-Control': 'no-store' });
    res.end(buf);
  });
}

function proxy(req, res) {
  const t = new URL(PROXY);
  // The proxy stands in for same-origin serving: rewrite Origin/Referer so the backend's cross-origin guard sees the page as its own.
  const up = http.request({ host: t.hostname, port: t.port, path: req.url, method: req.method, headers: { ...req.headers, host: t.host, ...(req.headers.origin ? { origin: `http://${t.host}` } : {}), ...(req.headers.referer ? { referer: `http://${t.host}/` } : {}) } }, (r) => { res.writeHead(r.statusCode, r.headers); r.pipe(res); });
  up.on('error', (e) => problem(res, 502, 'Bad gateway', String(e.message), 'proxy_error'));
  res.on('close', () => up.destroy()); req.pipe(up);
}

export const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, 'http://x');
  try {
    if (PROXY && (url.pathname.startsWith('/api/') || url.pathname === '/metrics')) return proxy(req, res);
    if (url.pathname.startsWith('/api/v1')) return await api(req, res, url);
    if (url.pathname === '/metrics') { res.writeHead(200, { 'Content-Type': 'text/plain' }); return res.end('# mock\nec_up 1\n'); }
    if (['/__shot', '/__slow.gif', '/__demo.js'].includes(url.pathname)) return serveShot(res, url);
    return serveStatic(req, res, url);
  } catch (e) { if (!res.headersSent) problem(res, 500, 'Internal error', String(e.message)); else res.end(); }
});
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  server.listen(PORT, HOST, () => console.log(`mock engine-console: http://${HOST}:${server.address().port}/  (${KEY ? 'bearer key required' : 'no auth'})`));
}
