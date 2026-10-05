#!/usr/bin/env node
// Dependency-free mock of the Engine Console API (ARCHITECTURE.md §4 + §8 addendum v1.1 + §10 contract v2, the fleet; response
// shapes mirror backend/src/engine_console/domain/{models,fleet,refs,learn}.py) plus a static host for the frontend.
// Usage: node dev/mock-server.mjs [--port 8791] [--host 127.0.0.1]   (MOCK_KEY=secret to require a bearer key)
//   --proxy <url>            serve the frontend here but forward /api/v1 and /metrics to a real backend (MOCK_PROXY)
//   --learn-dir <dir>        serve GET /learn from the real content files (backend learn/*.json) instead of the sample (MOCK_LEARN_DIR)
//   --catalogs-file <json>   use a dump of the real adapters' engines, parameter catalogs and presets (MOCK_CATALOGS_FILE)
// All data is canned/simulated. Nothing here is a verified engine fact. Fixtures live in mock-fleet.mjs (machines, instances,
// library, scripts), mock-learn.mjs (learn bundle) and mock-catalogs.mjs (parameters, presets, Hugging Face models).
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { catalogs, presets, models, engineMeta } from './mock-catalogs.mjs';
import { learnBundle as builtinLearn, params as builtinLearnParams, bundleFromDir } from './mock-learn.mjs';
import * as F from './mock-fleet.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
const argv = (n, d) => { const i = args.indexOf(`--${n}`); return i >= 0 ? args[i + 1] : d; };
const PORT = Number(argv('port', process.env.PORT ?? 8791));
const HOST = argv('host', '127.0.0.1');
const KEY = process.env.MOCK_KEY || '';
// --learn-dir <path> serves GET /learn (and the params' explain) from a directory of the real content files (backend learn/*.json) instead of the built-in sample.
const LEARN_DIR = argv('learn-dir', process.env.MOCK_LEARN_DIR || '');
const realLearn = LEARN_DIR ? bundleFromDir(LEARN_DIR) : null;
const learnBundle = () => realLearn || builtinLearn();
const learnParams = realLearn ? realLearn.params : builtinLearnParams;
// --catalogs-file <json> replaces the sample engines, parameter catalogs and presets with a dump of the real adapters
// ({engines: [...as GET /engines], params: {engine: [ParamSpec...]}}), so the Launch form can be checked on the production parameters.
const CATALOGS_FILE = argv('catalogs-file', process.env.MOCK_CATALOGS_FILE || '');
if (CATALOGS_FILE) {
  const real = JSON.parse(fs.readFileSync(CATALOGS_FILE, 'utf8'));
  for (const k of Object.keys(catalogs)) delete catalogs[k];
  for (const k of Object.keys(presets)) delete presets[k];
  Object.assign(catalogs, real.params);
  for (const e of real.engines) presets[e.id] = e.presets;
  engineMeta.splice(0, engineMeta.length, ...real.engines.map(({ id, display_name, default_image, model_formats, images_by_arch, multi_model }) => ({ id, display_name, default_image, model_formats, images_by_arch, multi_model })));
}
// --proxy http://127.0.0.1:8791 forwards /api/v1 and /metrics to a real backend (dev/screenshot use only), keeping the static host + /__shot helpers.
const PROXY = argv('proxy', process.env.MOCK_PROXY || '');
const G = 2 ** 30;
const now = () => Date.now() / 1000; // epoch seconds, like the backend
const uid = (p) => `${p}${crypto.randomBytes(6).toString('hex')}`;
let seed = 42; const rnd = () => { seed = (seed * 1664525 + 1013904223) % 4294967296; return seed / 4294967296; };

// ---------------- state ----------------
const state = {
  instances: F.instances, downloads: [], profiles: [{ id: 'prof_1', name: 'qwen-agent', engine: 'vllm', repo_id: 'Qwen/Qwen3.6-35B-A3B-FP8', description: '', created_at: now() - 5e5, updated_at: now() - 5e5, params: { max_model_len: 65536, enable_auto_tool_choice: true, tool_call_parser: 'hermes', hf_token: '[set]' } }],
  conversations: [], messages: new Map(), prompts: [{ id: 'prm_1', title: 'Terse reviewer', content: 'You are a terse senior code reviewer. Reply with findings only.', tags: [], created_at: now(), updated_at: now() }],
  benches: [], matches: new Map(), ratings: new Map(), audit: [], auditSeq: 0,
  settings: { hf_cache_dir: '~/.cache/huggingface', default_gpu_ids: [0], idle_ttl_s: 0, image_pins: { vllm: 'vllm/vllm-openai:v0.27.1', sglang: 'lmsysorg/sglang:v0.6.0' }, host_image_pins: {}, hf_token: '[set, 37 chars]', docker_context: 'rootless', docker_network: 'engine-console-engines', port_range: [18000, 18099], hf_allowed_hosts: ['huggingface.co', '*.hf.co'] },
  keys: [{ id: 'k_1', name: 'bootstrap', role: 'admin', prefix: 'ec_3fa9', created_at: now() - 864e3 * 6, last_used_at: now() - 300, secret: null }],
};
const logs = new Map();
const streams = { metrics: new Set(), downloads: new Set(), logs: new Map(), bench: new Map(), fleet: new Set() };
const history = new Map();
const usageLog = [];

function addLog(id, line) {
  const a = logs.get(id) || []; a.push(`${new Date().toISOString().slice(11, 23)} ${line}`); if (a.length > 2000) a.shift(); logs.set(id, a);
  for (const r of streams.logs.get(id) || []) sse(r, a.at(-1), 'log');
}
const imageFor = (hostId, engine, requested) => {
  const em = engineMeta.find((e) => e.id === engine);
  const arch = F.hostDef(hostId)?.node?.arch || 'x86_64';
  return requested || state.settings.host_image_pins[hostId]?.[engine] || state.settings.image_pins[engine] || em?.images_by_arch?.[arch] || em?.default_image || null;
};
// Policy patterns are exact references or "repo:*" (a trailing * matches any suffix), as in docs/ARCHITECTURE.md section 10.
const imageAllowed = (patterns, ref) => (patterns || []).some((p) => p === ref || (p.endsWith('*') && ref.startsWith(p.slice(0, -1))));
const nextPort = (hostId) => { const [lo, hi] = F.hostDef(hostId)?.policy?.port_range || [18000, 18099]; const used = new Set(F.instancesOn(hostId).map((i) => i.port)); for (let p = lo; p <= hi; p++) if (!used.has(p)) return p; return null; };
const instMem = (engine) => ({ vllm: { total_gib: 34, anon_gib: 20.1, cache_gib: 12, shmem_gib: 1.4, kernel_gib: 0.5, source: 'cgroup' }, sglang: { total_gib: 28, anon_gib: 14.2, cache_gib: 12.8, shmem_gib: 0.8, kernel_gib: 0.2, source: 'cgroup' }, llamacpp: { total_gib: 6, anon_gib: 2.2, cache_gib: 3.7, shmem_gib: 0.0, kernel_gib: 0.1, source: 'cgroup' } }[engine] || { total_gib: 8, anon_gib: 4, cache_gib: 3.8, shmem_gib: 0.1, kernel_gib: 0.1, source: 'cgroup' });

function makeInstance({ engine, ref, host_id = 'local', params = {}, gpu_ids = [0], name, profile_id, ready = false, ttl_idle_s, image, fit }) {
  const id = uid('inst_');
  const display = ref.format === 'hf' ? ref.repo_id : ref.format === 'ollama' ? ref.name : String(ref.rel || '').split('/').pop().replace(/\.gguf$/, '');
  const port = nextPort(host_id);
  const found = F.resolveModel(host_id, ref);
  const loc = found ? (() => { const d = F.hostDef(host_id); const r = d.roots.find((x) => x.id === (found.copy.host_id === host_id ? found.copy.root_id : ref.root_id)) || d.roots[0]; const m = d.disks.find((x) => x.mount === r.mountAt); return { host_id, root_id: r.id, root_label: r.label, path: `${r.path}/${found.copy.rel}`.replace(/\/+/g, '/'), drive: m?.label ?? null, fstype: m?.fstype ?? null, remote: !!m?.remote, shared_from: r.shared_from }; })() : null;
  const perGpu = (fit?.per_gpu || []).map((g) => ({ gpu_index: g.gpu_id, gib: g.total_gib }));
  const inst = { id, name: name || display.split('/').pop().toLowerCase().replace(/[^a-z0-9._-]+/g, '-').slice(0, 24), engine, repo_id: display, params, gpu_ids, gpu_uuids: [], port, container_name: `ec-${engine}-${id.slice(5, 11)}`, container_port: 8000, internal_endpoint: `http://ec-${engine}-${id.slice(5, 11)}:8000`,
    image: image || imageFor(host_id, engine), state: 'starting', phase: 'loading_weights', progress_pct: 0, pinned: false, ttl_idle_s: ttl_idle_s ?? null, profile_id: profile_id || null,
    error: null, last_logs: null, fit: null, created_at: now(), started_at: null, last_request_at: null, uptime_s: null,
    managed: true, source: 'console', endpoint: `http://127.0.0.1:${port}`, served_models: [display], state_reason: null, history_since: null, host_memory: instMem(engine),
    host_id, management: 'managed', script: null, model: F.fullRef(ref), model_location: loc, vramTemplate: perGpu.length ? perGpu : gpu_ids.map((g) => ({ gpu_index: g, gib: 20 })), noMetrics: false };
  state.instances.push(inst);
  addLog(id, `[${engine}] launching ${display} on ${F.hostDef(host_id).label} GPU ${gpu_ids.join(',')} port ${port}`);
  const setReady = (t0) => { inst.state = 'ready'; inst.started_at = t0; inst.phase = 'ready'; inst.progress_pct = 100; };
  if (ready) { setReady(now() - 3600); return inst; }
  const fail = display.includes('FAIL');
  const phases = [['loading_weights', 'Loading weights'], ['compiling', 'torch.compile'], ['capturing_graphs', 'Capturing CUDA graphs'], ['warming_up', 'Warming up']];
  let step = 0, pct = 0;
  const tick = setInterval(() => {
    if (!state.instances.includes(inst) || inst.state === 'stopped' || inst.state === 'stopping') return clearInterval(tick);
    inst.state = 'loading'; pct += 8 + rnd() * 10;
    if (pct >= 100) { pct = 0; step++; }
    if (fail && step === 1) { inst.state = 'failed'; inst.error = 'CUDA out of memory while compiling (mock failure: the model name contains FAIL).'; inst.last_logs = (logs.get(id) || []).slice(-20).join('\n'); addLog(id, 'ERROR torch.OutOfMemoryError: CUDA out of memory'); return clearInterval(tick); }
    if (step >= phases.length) { setReady(now()); addLog(id, 'INFO Application startup complete.'); return clearInterval(tick); }
    inst.phase = phases[step][0]; inst.progress_pct = pct;
    addLog(id, `INFO ${phases[step][1]}… ${Math.round(pct)}%`);
  }, 900);
  return inst;
}
for (const i of state.instances) {
  if (i.management === 'managed' && i.state === 'ready') for (const l of ['INFO Started server process', 'INFO Loading weights took 41.2 s', 'INFO GPU KV cache size: 812,304 tokens', 'INFO Application startup complete.']) addLog(i.id, l);
  if (i.state === 'failed') for (const l of i.last_logs.split('\n')) addLog(i.id, l);
}
const view = F.instanceView;
const notManaged = (res, inst, why) => problem(res, 409, 'Conflict', why || `instance ${inst.name} was not started by the console and is monitor-only`, 'instance_not_managed');
const removeInstance = (inst) => { const k = state.instances.indexOf(inst); if (k >= 0) state.instances.splice(k, 1); };

// ---------------- helpers ----------------
const MIME = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.mjs': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8', '.json': 'application/json', '.svg': 'image/svg+xml', '.ico': 'image/x-icon' };
// The strictest policy the app is meant to run under: no inline script or style, no remote anything.
const CSP = "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data: blob:; base-uri 'none'; form-action 'self'; frame-ancestors 'none'";
const json = (res, code, obj) => { const b = JSON.stringify(obj); res.writeHead(code, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' }); res.end(b); };
const noContent = (res) => { res.writeHead(204); res.end(); };
const problem = (res, code, title, detail, c, extra) => { const cc = c || title.toLowerCase().replace(/\W+/g, '_'); res.writeHead(code, { 'Content-Type': 'application/problem+json' }); res.end(JSON.stringify({ type: `urn:engine-console:${cc}`, title, status: code, detail, code: cc, ...(extra || {}) })); };
const TITLES = { 400: 'Bad request', 403: 'Forbidden', 404: 'Not found', 409: 'Conflict', 422: 'Unprocessable', 501: 'Not implemented', 502: 'Bad gateway', 503: 'Service unavailable', 504: 'Gateway timeout', 507: 'Insufficient storage' };
const fail = (res, e) => problem(res, e.status, TITLES[e.status] || 'Error', e.detail, e.code, e.extra);
function sse(res, data, event) { if (res.writableEnded) return; const p = typeof data === 'string' ? data : JSON.stringify(data); res.write(`${event ? `event: ${event}\n` : ''}${(p.split('\n')).map((l) => `data: ${l}`).join('\n')}\n\n`); }
function openSSE(req, res, set) { res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-store', Connection: 'keep-alive' }); res.write(': ok\n\n'); set?.add(res); req.on('close', () => set?.delete(res)); }
const readBody = (req) => new Promise((ok, no) => { const c = []; let n = 0; req.on('data', (d) => { n += d.length; if (n > 40e6) { no(new Error('too large')); req.destroy(); } else c.push(d); }); req.on('end', () => { const raw = Buffer.concat(c).toString(); if (!raw) return ok({}); try { ok(JSON.parse(raw)); } catch { ok({ __raw: raw }); } }); req.on('error', no); });
const page = (items) => ({ items, next_cursor: null });
const hostOr404 = (res, id) => { const d = F.hostDef(id); if (!d) { problem(res, 404, 'Not found', `no such host: ${id}`, 'unknown_host'); return null; } return d; };
const needUp = (res, id) => { const d = F.hostDef(id); if (!F.hostUp(id)) { fail(res, F.NODE_ERR(d.label, 'node_unavailable', `the link to ${d.label} is ${F.hostSummary(id).link.state}: ${F.hostSummary(id).link.error || 'not connected'}`)); return false; } return true; };

const hostGpuStats = (hostId) => F.hostSummary(hostId).gpus.map((g) => {
  const mem = F.hostMem(hostId);
  const used = g.unified_memory ? (mem ? mem.used_gib : 0) : g.used_gib ?? 0, total = g.total_gib;
  return { index: g.index, uuid: g.uuid, name: g.name, total_gib: total, free_gib: g.unified_memory ? (mem ? mem.available_gib : 0) : g.free_gib ?? total - used, used_gib: used, util_pct: g.util_pct, temp_c: g.temp_c, power_w: g.power_w, fan_pct: g.fan_pct, compute_capability: g.compute_capability };
});
const sysMem = (hostId = 'local') => F.hostMem(hostId);

// ---------------- fit estimate (mock; real shape = domain/models.py FitReport) ----------------
const findModel = (repo) => models.find((m) => m.repo_id === repo);
const verdictOf = (r) => (r <= 0.9 ? 'fits' : r <= 1 ? 'tight' : 'wont_fit');
// Which parameter keys carry tensor parallelism, the memory share, the context and the KV precision, per engine.
const EKEYS = {
  vllm: { tp: 'tensor_parallel_size', frac: 'gpu_memory_utilization', len: 'max_model_len', eager: 'enforce_eager', kvq: 'kv_cache_dtype', defFrac: 0.9 },
  sglang: { tp: 'tp_size', frac: 'mem_fraction_static', len: 'context_length', eager: 'disable_cuda_graph', kvq: 'kv_cache_dtype', defFrac: 0.88 },
  llamacpp: { tp: null, frac: null, len: 'ctx_size', eager: null, kvq: 'cache_type_k', defFrac: 0.95, seqs: 'parallel' },
  ollama: { tp: null, frac: null, len: 'context_length', eager: null, kvq: 'kv_cache_type', defFrac: 0.95, seqs: 'num_parallel' },
  trtllm: { tp: 'tp_size', frac: 'kv_cache_free_gpu_memory_fraction', len: 'max_seq_len', eager: null, kvq: null, defFrac: 0.9 },
};
const refOf = (b) => b.model || (b.repo_id ? { format: 'hf', repo_id: b.repo_id } : null);
function modelMeta(hostId, ref) {
  if (!ref) return null;
  if (ref.format === 'hf' || (!ref.format && ref.repo_id)) { const m = findModel(ref.repo_id); if (m) return { ...m, display: ref.repo_id }; }
  const f = F.resolveModel(hostId, ref);
  if (!f) return null;
  const big = (f.model.params || 0) > 20e9;
  return { repo_id: f.model.name, display: f.model.name, size_bytes: f.model.size_bytes, layers: big ? 64 : 36, kv_heads: 8, head_dim: 128, ctx: f.model.context || 8192, is_moe: false, quant: f.model.quant ? String(f.model.quant).toLowerCase() : 'bf16', gated: false, num_params: f.model.params };
}
function fitEstimate(body) {
  const hostId = body.host_id || 'local', d = F.hostDef(hostId);
  const engine = body.engine, K = EKEYS[engine] || EKEYS.vllm, params = body.params || {}, ref = refOf(body);
  const m = modelMeta(hostId, ref);
  const label = d?.label || hostId;
  if (!m) return { verdict: 'unknown', confidence: 'low', tp: 1, tp_required: null, concurrency: 1, max_len: null, per_gpu: [], max_context_at_current_concurrency: null, max_concurrency_at_current_context: null, fits_if_stop: [], notes: [`No metadata for ${ref?.repo_id || ref?.name || ref?.rel || 'this model'} on ${label} (the real backend reads config.json or the GGUF header).`], compat: [] };
  const all = d.gpus.map((g) => g.index);
  const gpu_ids = (body.gpu_ids?.length ? body.gpu_ids : all).filter((i) => all.includes(i));
  const tp = K.tp ? Math.max(1, Number(params[K.tp] || 1)) : 1;
  const frac = Number((K.frac && params[K.frac]) || K.defFrac);
  const len = Number(params[K.len] || Math.min(m.ctx, 32768));
  const seqs = Math.max(1, (Number(body.concurrency) || 1) * (K.seqs ? Number(params[K.seqs] || 1) : 1));
  const kvBytes = /fp8|q8/.test(String(params[K.kvq] || '')) ? 1 : /q4/.test(String(params[K.kvq] || '')) ? 0.5 : 2;
  const eager = !!(K.eager && params[K.eager]);
  const n = K.tp ? tp : engine === 'llamacpp' ? Math.max(1, gpu_ids.length) : 1;
  const compat = [];
  if (K.tp && tp > gpu_ids.length) compat.push({ level: 'block', code: 'tp_gt_gpus', message: `Tensor parallel ${tp} needs ${tp} GPUs but ${gpu_ids.length} selected. Select more GPUs or lower TP.` });
  if (m.quant === 'nvfp4' && m.is_moe && engine === 'vllm') compat.push({ level: 'warn', code: 'nvfp4_moe_sm120', message: 'NVFP4 MoE falls back to a slow Marlin kernel on sm_120 in vLLM. Use FP8 or SGLang.' });
  if (m.quant === 'mxfp4' && m.repo_id !== 'openai/gpt-oss-120b' && ref?.format === 'hf') compat.push({ level: 'block', code: 'mxfp4_moe_sm120', message: 'MXFP4 MoE asserts on sm_120 except gpt-oss-120b.' });
  if (m.gated) compat.push({ level: 'warn', code: 'gated', message: 'Gated model: HF_TOKEN and licence acceptance required to download.' });
  const used = gpu_ids.slice(0, Math.max(n, 1));
  const perTokB = 2 * m.layers * m.kv_heads * m.head_dim * kvBytes;
  const live = F.hostSummary(hostId).gpus;
  const unified = d.gpus.some((g) => g.unified_memory);
  const per_gpu = used.map((gid) => {
    const g = d.gpus.find((x) => x.index === gid), lv = live.find((x) => x.index === gid);
    const w = (m.size_bytes / G) / n, kv = (len * seqs * perTokB) / G / n, act = 1.6, gr = eager ? 0 : 1.4, ov = 1.2, total = w + kv + act + gr + ov, budget = g.total_gib * frac;
    return { gpu_id: gid, uuid: g.uuid, name: g.name, weights_gib: w, kv_cache_gib: kv, activations_gib: act, cuda_graphs_gib: gr, overhead_gib: ov, total_gib: total, budget_gib: budget, vram_total_gib: g.total_gib, free_gib: unified ? (F.hostMem(hostId)?.available_gib ?? 0) : lv?.free_gib ?? g.total_gib, utilization_pct: (total / budget) * 100, verdict: verdictOf(total / budget) };
  });
  const worst = per_gpu.reduce((a, g) => Math.max(a, g.total_gib / g.budget_gib), 0);
  const biggest = Math.max(...d.gpus.map((g) => g.total_gib));
  let tp_required = null;
  for (const k of [1, 2, 4, 8]) if ((m.size_bytes / G) / k <= biggest * frac * 0.9) { tp_required = k; break; }
  const g0 = per_gpu[0];
  const room = g0 ? Math.max(0, g0.budget_gib - g0.weights_gib - g0.activations_gib - g0.cuda_graphs_gib - g0.overhead_gib) : 0;
  const resident = F.instancesOn(hostId).filter((i) => (i.state === 'ready' || i.state === 'loading') && i.gpu_ids.some((x) => gpu_ids.includes(x)));
  const verdict = compat.some((c) => c.level === 'block') ? 'wont_fit' : per_gpu.length ? verdictOf(worst) : 'unknown';
  const mem = F.hostMem(hostId);
  const off = Number(params.cpu_offload_gb || 0) * used.length, gpuOnRam = unified ? per_gpu.reduce((a, g) => a + g.total_gib, 0) : 0;
  const need = 4 + off + gpuOnRam + (unified ? 0 : (m.size_bytes / G) * 0.05), avail = mem?.available_gib ?? 0;
  return { verdict, confidence: m.is_moe ? 'medium' : ref?.format === 'hf' ? 'high' : 'medium', tp: n, tp_required, concurrency: seqs, max_len: len, kv_bytes_per_token: perTokB / n, per_gpu,
    max_context_at_current_concurrency: Math.round(((room * G * n) / perTokB) / seqs), max_concurrency_at_current_context: Math.round((room * G * n) / perTokB / len), compat,
    fits_if_stop: verdict === 'wont_fit' && resident.length && !compat.some((c) => c.level === 'block') ? resident.map((i) => i.name) : [],
    host_ram: { needed_gib: need, available_gib: avail, total_gib: mem?.total_gib ?? null, verdict: need > avail ? 'wont_fit' : need > avail * 0.9 ? 'tight' : 'ok', breakdown: { 'engine process': 4, ...(off ? { 'CPU offload': off } : {}), ...(unified ? { 'GPU memory (shared pool)': gpuOnRam } : { 'page cache (weights)': (m.size_bytes / G) * 0.05 }) },
      notes: [...(off ? ['CPU offload keeps weights in pinned host RAM.'] : []), ...(unified ? [`${label} has unified memory: the GPU and the system share one pool.`] : [])] },
    notes: [`Mock estimate for ${label}: KV = 2 x layers x kv_heads x head_dim x ${kvBytes} B x ${len.toLocaleString('en-US')} tokens x ${seqs} seqs.`] };
}

// ---------------- preflight (check codes are the hub's own: services/lifecycle.py) ----------------
function validateParams(engine, params) {
  const specs = catalogs[engine] || [];
  const errs = [];
  for (const [k, v] of Object.entries(params || {})) {
    const s = specs.find((x) => x.key === k);
    if (!s) { errs.push(`unknown parameter ${k}`); continue; }
    if ((s.type === 'int' || s.type === 'float') && typeof v === 'number') { if (s.min != null && v < s.min) errs.push(`${k}: minimum is ${s.min}`); if (s.max != null && v > s.max) errs.push(`${k}: maximum is ${s.max}`); }
  }
  return errs;
}
function preflight(body) {
  const hostId = body.host_id || 'local', d = F.hostDef(hostId);
  const label = d.label, checks = [], gpu_ids = body.gpu_ids || [];
  const sum = F.hostSummary(hostId);
  if (!F.hostUp(hostId)) { checks.push({ code: 'host_down', level: 'block', message: `${label} is not connected (${sum.link.error || sum.link.state}): nothing can be started there until its link is up` }); return { ok: false, gpu_ids, checks, fit: null }; }
  if (d.policy.mode !== 'manage') checks.push({ code: 'monitor_only', level: 'block', message: `${label}'s node is in monitor mode (${d.policy.path}): it can be watched but not changed. Its operator must switch "mode" to "manage" in that policy file.` });
  const em = engineMeta.find((e) => e.id === body.engine), ref = refOf(body);
  const errs = em ? validateParams(body.engine, body.params) : [`unknown engine ${body.engine}`];
  if (errs.length) checks.push({ code: 'invalid_params', level: 'block', message: errs.join('; ') });
  const formatOk = !!(em && ref && em.model_formats.includes(ref.format || 'hf'));
  if (em && ref && !formatOk) checks.push({ code: 'format_unsupported', level: 'block', message: `${em.display_name} cannot load ${ref.format} models (it loads: ${em.model_formats.join(', ')})` });
  const found = ref ? F.resolveModel(hostId, ref) : null;
  const display = ref?.repo_id || ref?.name || (ref?.rel || '').split('/').pop() || 'the model';
  // An Ollama server pulls by name, so only the other formats must already be in that machine's library.
  if (ref && ref.format !== 'ollama' && !found) checks.push({ code: 'model_missing', level: 'block', message: `${display} is not in ${label}'s model library: download or copy it to that machine first` });
  let fit = null;
  if (found && formatOk && !errs.length) {
    fit = fitEstimate({ ...body, host_id: hostId });
    checks.push({ code: 'fit', level: { fits: 'ok', tight: 'warn', unknown: 'warn', wont_fit: 'block' }[fit.verdict], message: `fit verdict: ${fit.verdict} (${fit.confidence} confidence)${fit.fits_if_stop.length ? `; fits if you stop ${fit.fits_if_stop.join(', ')}` : ''}` });
    for (const c of fit.compat) if (c.level !== 'ok') checks.push({ code: c.code, level: c.level, message: c.message });
  } else if (found && formatOk) checks.push({ code: 'fit_unavailable', level: 'warn', message: 'no fit estimate: the parameters are invalid' });
  for (const [k, v] of Object.entries(body.params || {})) if (/trust_remote/.test(k) && v) checks.push({ code: 'trust_remote_code', level: 'warn', message: `${k} runs Python code from the model repo inside the container; it is off unless you set it explicitly` });
  const port = nextPort(hostId);
  checks.push(port ? { code: 'port', level: 'ok', message: `host port ${port}` } : { code: 'no_free_port', level: 'block', message: `no free port in ${d.policy.port_range.join('-')} on ${label}` });
  const have = d.gpus.map((g) => g.index);
  const missing = gpu_ids.filter((i) => !have.includes(i));
  if (!have.length) checks.push({ code: 'no_gpu', level: 'block', message: `${label} reports no GPU` });
  else if (missing.length) checks.push({ code: 'gpu_unknown', level: 'block', message: `${label} has no GPU ${missing.join(', ')} (it has ${have.join(', ')})` });
  else {
    const chosen = gpu_ids.length ? gpu_ids : have;
    const clash = F.instancesOn(hostId).filter((i) => ['ready', 'loading'].includes(i.state) && chosen.some((g) => i.vramTemplate.some((v) => v.gpu_index === g))).map((i) => i.name);
    if (clash.length) checks.push({ code: 'gpu_shared', level: 'warn', message: `GPU already used by ${clash.join(', ')}; relying on free VRAM` });
  }
  const image = imageFor(hostId, body.engine, body.image);
  if (!image || !imageAllowed(d.policy.engine_images, image)) checks.push({ code: 'image_not_allowed', level: 'block', message: `image ${image} is not allowed by ${label}'s policy (engine_images)` });
  else if (!d.images.some((i) => i.ref === image)) checks.push({ code: 'image_missing', level: 'block', message: `image ${image} is not present on ${label}: pull it on that machine (the console never pulls)` });
  if (!d.policy.gateway_image) checks.push({ code: 'gateway_missing', level: 'block', message: `${label}'s policy names no gateway_image, so engines there cannot get their loopback gateway` });
  if (body.engine === 'sglang' && body.params?.api_key) checks.push({ code: 'secret_in_argv', level: 'block', message: 'SGLang can only take api_key on its command line, which would expose the secret in the process list. Remove api_key, or use an engine that reads it from the environment.' });
  return { ok: !checks.some((c) => c.level === 'block'), gpu_ids: gpu_ids.length ? gpu_ids : have.slice(0, 1), checks, fit };
}

// ---------------- metrics ----------------
function sample(inst, t = now()) {
  const busy = inst.state === 'ready' && !inst.noMetrics && F.hostUp(inst.host_id);
  const w = Math.sin(t / 15) * 0.5 + 0.5;
  const running = busy ? Math.round(2 + w * 10 + rnd() * 3) : 0;
  return { t, values: busy ? { requests_running: running, requests_waiting: Math.max(0, Math.round((w - 0.6) * 12 + rnd())), kv_cache_usage_pct: 8 + w * 60 + rnd() * 5,
    prefix_cache_hit_pct: 40 + w * 30 + rnd() * 5, generation_tps: running * (55 + rnd() * 12), prompt_tps: 300 + w * 3200 + rnd() * 200,
    ttft_p50_s: 0.08 + w * 0.15 + rnd() * 0.02, ttft_p95_s: 0.3 + w * 0.5 + rnd() * 0.05, itl_p50_s: 0.014 + w * 0.01, e2e_p50_s: 2 + w * 3, preemptions_total: 0,
    ...(inst.host_memory ? { host_ram_gib: inst.host_memory.total_gib + w, host_ram_anon_gib: inst.host_memory.anon_gib + w * 0.6, host_ram_cache_gib: inst.host_memory.cache_gib + w * 0.4, host_ram_shmem_gib: inst.host_memory.shmem_gib } : {}) } : {} };
}
for (let k = 0; k < 450; k++) for (const i of state.instances) { const a = history.get(i.id) || []; a.push(sample(i, now() - (450 - k) * 2)); history.set(i.id, a); }
setInterval(() => {
  for (const i of state.instances) {
    const a = history.get(i.id) || []; const s = sample(i); a.push(s); if (a.length > 45000) a.shift(); history.set(i.id, a);
    if (i.state === 'ready' && Object.keys(s.values).length) for (const r of streams.metrics) sse(r, { instance_id: i.id, ...s }, 'metrics');
  }
}, 2000);
setInterval(() => { for (const i of state.instances) if (i.management === 'managed' && i.state === 'ready' && rnd() > 0.4) addLog(i.id, `INFO Avg generation throughput: ${(300 + rnd() * 500).toFixed(1)} tokens/s, Running: ${Math.round(rnd() * 8)} reqs, KV cache usage: ${(rnd() * 40).toFixed(1)}%`); }, 1800);

// ---------------- fleet stream: every host on connect, then all of them every 3 s, and at once on a link change ----------------
setInterval(() => { for (const r of streams.fleet) for (const id of F.hostIds()) sse(r, F.hostSummary(id), 'host'); }, 3000);
F.onHostChange((id) => { for (const r of streams.fleet) sse(r, F.hostSummary(id), 'host'); });

// ---------------- downloads and copies ----------------
const dl = (o) => ({ id: uid('dl_'), repo_id: '', revision: null, commit_sha: null, allow_patterns: null, state: 'queued', total_bytes: 0, done_bytes: 0, speed_bps: 0, eta_s: null, files_total: 6, files_done: 0, current_file: null, error: null, error_code: null, created_at: now() - 600, updated_at: now(), host_id: 'local', root_id: null, kind: 'hf', source_copy_id: null, ...o });
const srcCopy = F.libraryView().find((m) => m.key === 'hf:Qwen/Qwen3.6-35B-A3B-FP8').copies.find((c) => c.host_id === 'local');
state.downloads.push(
  dl({ repo_id: 'openai/gpt-oss-120b', state: 'running', total_bytes: 65 * G, done_bytes: 26 * G, speed_bps: 310 * 2 ** 20, eta_s: 130, files_total: 20, files_done: 8, current_file: 'model-00009-of-00014.safetensors', host_id: 'edge-1', root_id: 'hf-main', created_at: now() - 300 }),
  dl({ repo_id: 'Qwen/Qwen3.6-35B-A3B-FP8', kind: 'copy', source_copy_id: srcCopy.id, state: 'running', total_bytes: 36 * G, done_bytes: 7 * G, speed_bps: 240 * 2 ** 20, eta_s: 125, files_total: 14, files_done: 3, current_file: 'model-00004-of-00008.safetensors', host_id: 'edge-1', root_id: 'hf-main', created_at: now() - 120 }),
  dl({ repo_id: 'Qwen/Qwen3-Embedding-8B', state: 'queued', total_bytes: 16 * G, files_total: 9, host_id: 'local', root_id: null, created_at: now() - 20 }),
  dl({ repo_id: 'bartowski/gemma-4-31b-it-GGUF', allow_patterns: ['*Q4_K_M*.gguf'], state: 'completed', total_bytes: 18.9 * G, done_bytes: 18.9 * G, files_total: 1, files_done: 1, host_id: 'gpu-box-2', root_id: 'gguf-shared', created_at: now() - 7200 }),
  dl({ repo_id: 'meta-llama/Llama-3.3-70B-Instruct', state: 'failed', total_bytes: 141 * G, files_total: 30, host_id: 'gpu-box-2', root_id: 'hf-main', error: 'Gated model: accept the licence on Hugging Face and set HF_TOKEN on the console.', error_code: 'gated_model', created_at: now() - 5400 }),
);
setInterval(() => {
  for (const d of state.downloads) {
    if (d.state === 'queued' && state.downloads.filter((x) => x.state === 'running').length < 2) d.state = 'running';
    if (d.state !== 'running') continue;
    d.speed_bps = (180 + rnd() * 120) * 2 ** 20; d.done_bytes = Math.min(d.total_bytes, d.done_bytes + d.speed_bps * 0.5);
    d.files_done = Math.floor((d.done_bytes / d.total_bytes) * d.files_total); d.eta_s = (d.total_bytes - d.done_bytes) / d.speed_bps; d.updated_at = now();
    if (d.done_bytes >= d.total_bytes) { d.state = 'completed'; d.speed_bps = 0; d.eta_s = 0; F.onTransferDone(d, d.kind === 'copy' ? F.findCopy(d.source_copy_id) : null); }
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
  if (!F.hostUp(inst.host_id)) return fail(res, F.NODE_ERR(F.hostDef(inst.host_id).label, 'node_unavailable', 'the link is not up'));
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

// ---------------- engines + params with the learn merge ----------------
const engineList = () => engineMeta.map((e) => ({ id: e.id, display_name: e.display_name, default_image: state.settings.image_pins[e.id] || e.default_image, presets: presets[e.id] || [], presets_info: learnBundle().presets[e.id] || {}, model_formats: e.model_formats, images_by_arch: e.images_by_arch, multi_model: e.multi_model }));
const paramsWithLearn = (engine) => (catalogs[engine] || []).map((s) => ({ ...s, explain: learnParams[engine]?.[s.key] ?? null }));
const ggufGroups = (m) => (m.gguf || []).map(([quant, gib]) => {
  const total = Math.round(gib * G), stem = `${m.repo_id.split('/')[1].replace(/-GGUF$/i, '')}-${quant}`, parts = gib > 24 ? Math.ceil(gib / 24) : 1;
  return { quant, files: parts === 1 ? [`${stem}.gguf`] : Array.from({ length: parts }, (_, k) => `${quant}/${stem}-${String(k + 1).padStart(5, '0')}-of-${String(parts).padStart(5, '0')}.gguf`), total_bytes: total };
});

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
  if (p === '/health') return json(res, 200, { status: 'ok', mock: true, version: 'mock', engines: engineMeta.map((e) => e.id), docker_context: 'rootless', hosts: F.hostIds().length });
  if (KEY && req.headers.authorization !== `Bearer ${KEY}`) return problem(res, 401, 'Unauthorized', 'a valid bearer API key is required', 'unauthorized');
  // CSRF / DNS-rebinding guard, same as the real backend: mutating calls must carry X-Engine-Console.
  if (!SAFE.has(M) && !req.headers['x-engine-console']) return problem(res, 403, 'Forbidden', 'missing X-Engine-Console header', 'csrf_header_required');
  const body = !SAFE.has(M) ? await readBody(req).catch(() => ({})) : {};
  if (!SAFE.has(M) && body.__raw !== undefined && p !== '/profiles/import') return problem(res, 415, 'Unsupported Media Type', 'body must be JSON', 'unsupported_media_type');
  if (!SAFE.has(M)) state.audit.unshift({ id: ++state.auditSeq, ts: now(), actor: KEY ? 'key:mock' : 'loopback', role: 'admin', method: M, path: url.pathname, status: 200, params: { body: JSON.parse(JSON.stringify(body, (k, v) => (/token|secret|password|api[_-]?key/i.test(k) ? '[redacted]' : v))) } });
  let m;

  // ---- fleet (contract v2, section 10.8) ----
  if (p === '/hosts' && M === 'GET') return json(res, 200, page(F.hostIds().map(F.hostSummary)));
  if ((m = p.match(/^\/hosts\/([^/]+)$/)) && M === 'GET') { if (!hostOr404(res, m[1])) return; return json(res, 200, F.hostDetail(m[1])); }
  if ((m = p.match(/^\/hosts\/([^/]+)\/reconnect$/)) && M === 'POST') { if (!hostOr404(res, m[1])) return; const r = F.reconnect(m[1]); if (r?.error) return fail(res, r.error); return json(res, 200, r); }
  if (p === '/fleet/stream') { openSSE(req, res, once ? null : streams.fleet); for (const id of F.hostIds()) sse(res, F.hostSummary(id), 'host'); if (once) res.end(); return; }
  if (p === '/learn' && M === 'GET') return json(res, 200, learnBundle());
  if (p === '/library' && M === 'GET') {
    if (q.get('host') && !F.hostDef(q.get('host'))) return problem(res, 404, 'Not found', `no such host: ${q.get('host')}`, 'unknown_host');
    return json(res, 200, page(F.libraryView({ host: q.get('host') || '', format: q.get('format') || '', q: q.get('q') || '' })));
  }
  if (p === '/library/rescan' && M === 'POST') {
    if (body.host_id && !hostOr404(res, body.host_id)) return;
    const ids = (body.host_id ? [body.host_id] : F.hostIds()).filter((id) => F.hostUp(id));
    res.writeHead(202, { 'Content-Type': 'application/json' }); return res.end(JSON.stringify({ hosts: ids }));
  }
  if ((m = p.match(/^\/library\/copies\/([^/]+)$/)) && M === 'DELETE') { const r = F.deleteCopy(m[1]); if (r.error) return fail(res, r.error); return noContent(res); }
  if ((m = p.match(/^\/library\/copies\/([^/]+)\/copy$/)) && M === 'POST') {
    const f = F.findCopy(m[1]); if (!f) return problem(res, 404, 'Not found', 'no such model copy', 'copy_not_found');
    if (!body.host_id || !/^[a-z0-9][a-z0-9-]{0,31}$/.test(body.host_id)) return problem(res, 422, 'Validation error', 'host_id is required', 'validation_error');
    const d = hostOr404(res, body.host_id); if (!d) return;
    if (f.model.copies.some((c) => c.host_id === body.host_id || c.visible_on.includes(body.host_id))) return problem(res, 409, 'Conflict', `${d.label} already holds or can read this model`, 'conflict');
    if (!needUp(res, body.host_id)) return;
    if (d.policy.mode !== 'manage') return fail(res, F.NODE_ERR(d.label, 'monitor_only', 'this machine is in monitor mode: files cannot be written there'));
    const ch = F.chooseRoot(body.host_id, body.root_id, f.model.format); if (ch.error) return fail(res, ch.error);
    if (F.rootFree(body.host_id, ch.root) < f.view.size_bytes) return fail(res, F.NODE_ERR(d.label, 'no_space', `${Math.round(f.view.size_bytes / G)} GiB needed, ${Math.round(F.rootFree(body.host_id, ch.root) / G)} GiB free on ${ch.root.path}`));
    const row = dl({ repo_id: f.model.name, kind: 'copy', source_copy_id: f.view.id, state: 'queued', total_bytes: f.view.size_bytes, files_total: f.view.files || 1, host_id: body.host_id, root_id: ch.root.id, created_at: now() });
    state.downloads.unshift(row); return json(res, 201, row);
  }
  if (p === '/scripts' && M === 'GET') return json(res, 200, page(F.scriptsAll()));
  if ((m = p.match(/^\/scripts\/jobs\/([^/]+)(\/stream)?$/)) && M === 'GET') {
    const job = F.jobs.get(m[1]); if (!job) return problem(res, 404, 'Not found', `no such script job: ${m[1]}`, 'unknown_job');
    if (m[2]) return F.attachJobStream(job, res, once);
    return json(res, 200, job);
  }
  if ((m = p.match(/^\/scripts\/([^/]+)\/([^/]+)\/([^/]+)$/)) && M === 'POST') {
    const r = F.runScript(decodeURIComponent(m[1]), decodeURIComponent(m[2]), decodeURIComponent(m[3]), body.arg ?? null);
    if (r.error) return fail(res, r.error);
    res.writeHead(202, { 'Content-Type': 'application/json' }); return res.end(JSON.stringify(r.job));
  }

  // ---- v1 + v2 hardware / engines ----
  if (p === '/hardware' && M === 'GET') {
    const id = q.get('host') || 'local'; if (!hostOr404(res, id)) return; if (!needUp(res, id)) return;
    const mem = F.hostMem(id);
    return json(res, 200, { gpus: hostGpuStats(id), host_ram_gib: mem.total_gib, host_ram_free_gib: mem.available_gib, source: 'mock', memory: mem, alerts: F.alertsFor(mem), host_id: id });
  }
  if (p === '/engines') return json(res, 200, engineList());
  if ((m = p.match(/^\/engines\/(\w+)\/params$/))) return catalogs[m[1]] ? json(res, 200, paramsWithLearn(m[1])) : problem(res, 404, 'Not found', `unknown engine ${m[1]}`);
  if (p === '/hf/search') {
    let r = models.filter((x) => (!q.get('q') || q.get('q').toLowerCase().split(/\s+/).every((w) => x.repo_id.toLowerCase().includes(w) || x.quant.includes(w))) && (!q.get('task') || x.pipeline_tag === q.get('task')) && (!q.get('quant') || x.quant === q.get('quant'))
      && (!q.get('max_gib') || x.size_bytes / G <= Number(q.get('max_gib'))) && (!q.get('gated') || String(x.gated) === q.get('gated')));
    const s = q.get('sort') || 'downloads'; r = r.sort((a, b) => (s === 'likes' ? b.likes - a.likes : s === 'updated' ? b.repo_id.localeCompare(a.repo_id) : b.downloads - a.downloads));
    await new Promise((ok) => setTimeout(ok, 250));
    return json(res, 200, page(r.slice(0, Number(q.get('limit') || 30)).map((x) => ({ repo_id: x.repo_id, downloads: x.downloads, likes: x.likes, pipeline_tag: x.pipeline_tag, library_name: x.gguf ? 'gguf' : 'transformers', gated: x.gated, tags: [x.quant, x.pipeline_tag], num_params: x.num_params, approx_size_gib: Math.round((x.size_bytes / G) * 100) / 100, quantization: x.quant === 'bf16' ? null : x.quant }))));
  }
  if ((m = p.match(/^\/hf\/models\/(.+)$/)) && M === 'GET') {
    const id = decodeURIComponent(m[1]), x = findModel(id);
    if (!x) return problem(res, 404, 'Not found', `no such model: ${id}`, 'hf_not_found');
    await new Promise((ok) => setTimeout(ok, 200));
    const groups = x.gguf ? ggufGroups(x) : [];
    return json(res, 200, { info: { repo_id: id, revision: null, architectures: [x.arch], model_type: x.arch.replace(/For.*/, '').toLowerCase(), num_params: x.num_params, num_active_params: x.active || null, is_moe: !!x.is_moe, weight_bytes: x.size_bytes, quantization: x.quant === 'bf16' ? null : x.quant, dtype: 'bfloat16',
      num_layers: x.layers, num_kv_heads: x.kv_heads, head_dim: x.head_dim, max_position_embeddings: x.ctx, gated: x.gated, license: x.license, pipeline_tag: x.pipeline_tag, raw_config: {} }, sha: 'deadbeef',
      card: `# ${id}\n\nMock model card. **${x.arch}**, license \`${x.license}\`.\n\n## Usage\n\n\`\`\`bash\nvllm serve ${id}\n\`\`\`\n\n## Notes\n\n- <script>alert('xss')</script> would be escaped here.\n- See [the docs](https://huggingface.co/${id}).\n`,
      files: x.gguf ? [{ path: 'README.md', size: 4200 }, ...groups.flatMap((g) => g.files.map((f) => ({ path: f, size: Math.round(g.total_bytes / g.files.length) })))]
        : [{ path: 'config.json', size: 2100 }, ...Array.from({ length: 4 }, (_, i) => ({ path: `model-0000${i + 1}-of-00004.safetensors`, size: Math.round(x.size_bytes / 4) })), { path: 'tokenizer.json', size: 11e6 }],
      gguf_groups: groups });
  }
  if (p === '/fit' && M === 'POST') {
    if (body.host_id && !hostOr404(res, body.host_id)) return;
    if (!refOf(body)) return problem(res, 422, 'Validation error', 'give repo_id or model', 'validation_error');
    await new Promise((ok) => setTimeout(ok, 120)); return json(res, 200, fitEstimate(body));
  }
  if (p === '/downloads/stream') { openSSE(req, res, once ? null : streams.downloads); sse(res, state.downloads, 'snapshot'); if (once) res.end(); return; }
  if (p === '/downloads' && M === 'GET') return json(res, 200, page(state.downloads));
  if (p === '/downloads' && M === 'POST') {
    const x = findModel(body.repo_id); if (!body.repo_id) return problem(res, 422, 'Validation error', 'repo_id required', 'validation_error');
    const hostId = body.host_id || 'local', d = hostOr404(res, hostId); if (!d) return;
    if (!needUp(res, hostId)) return;
    if (d.policy.mode !== 'manage') return fail(res, F.NODE_ERR(d.label, 'monitor_only', 'this machine is in monitor mode: files cannot be written there'));
    const isGguf = !!(body.allow_patterns || []).some((pt) => /gguf/i.test(pt)) || !!x?.gguf;
    const ch = F.chooseRoot(hostId, body.root_id, isGguf ? 'gguf' : 'hf'); if (ch.error) return fail(res, ch.error);
    let total = x?.size_bytes || 8 * G;
    if (isGguf && x?.gguf) { const pats = (body.allow_patterns || []).join(' '); const g = x.gguf.find(([qn]) => pats.toLowerCase().includes(qn.toLowerCase())); if (g) total = g[1] * G; }
    if (F.rootFree(hostId, ch.root) < total) return fail(res, F.NODE_ERR(d.label, 'no_space', `${Math.round(total / G)} GiB needed, ${Math.round(F.rootFree(hostId, ch.root) / G)} GiB free on ${ch.root.path}`));
    const row = dl({ repo_id: body.repo_id, revision: body.revision ?? null, allow_patterns: body.allow_patterns ?? null, state: 'queued', total_bytes: total, files_total: isGguf ? Math.max(1, (body.allow_patterns || []).length) : 5, host_id: hostId, root_id: ch.root.id, created_at: now() });
    if (x?.gated && !state.settings.hf_token) { row.state = 'failed'; row.error = 'Gated model: set HF_TOKEN and accept the licence.'; }
    state.downloads.unshift(row); return json(res, 201, row);
  }
  if ((m = p.match(/^\/downloads\/([^/]+)(?:\/(pause|resume|cancel))?$/))) {
    const d = state.downloads.find((x) => x.id === m[1]); if (!d) return problem(res, 404, 'Not found', `no such download: ${m[1]}`);
    if (M === 'DELETE') { state.downloads = state.downloads.filter((x) => x !== d); return noContent(res); }
    if (m[2]) { d.state = m[2] === 'pause' ? 'paused' : m[2] === 'resume' ? 'running' : 'cancelled'; d.updated_at = now(); for (const r of streams.downloads) sse(r, d, 'progress'); }
    return json(res, 200, d);
  }
  // v1 /models: Hugging Face models of every host, each with its host_id
  if (p === '/models' && M === 'GET') {
    const rows = F.libraryView({ format: 'hf' }).flatMap((mm) => mm.copies.map((c) => ({ repo_id: mm.name, size_bytes: mm.size_bytes, revisions: ['main'], last_used: c.atime, path: c.path, host_id: c.host_id,
      engines_that_fit: ['vllm', 'sglang'].filter((e) => { try { return ['fits', 'tight'].includes(fitEstimate({ engine: e, repo_id: mm.name, params: {}, gpu_ids: [], host_id: c.host_id }).verdict); } catch { return false; } }) })));
    return json(res, 200, page(rows));
  }
  if ((m = p.match(/^\/models\/(.+)$/)) && M === 'DELETE') {
    const repo = decodeURIComponent(m[1]), host = q.get('host') || 'local';
    const mm = F.libraryView({ format: 'hf' }).find((x) => x.name === repo); const c = mm?.copies.find((x) => x.host_id === host);
    if (c) { const r = F.deleteCopy(c.id); if (r.error) return fail(res, r.error); }
    return noContent(res);
  }

  // ---- instances ----
  if (p === '/instances' && M === 'GET') { if (q.get('host') && !hostOr404(res, q.get('host'))) return; return json(res, 200, page(state.instances.filter((i) => !q.get('host') || i.host_id === q.get('host')).map(view))); }
  if (p === '/instances/discover' && M === 'POST') { for (const i of state.instances) if (i.managed === false && i.state === 'ready') i.last_request_at = now(); return json(res, 200, page(state.instances.map(view))); }
  if (p === '/instances/preflight' && M === 'POST') {
    if (body.host_id && !hostOr404(res, body.host_id)) return;
    if (!body.engine || !refOf(body)) return problem(res, 422, 'Validation error', 'engine and a model are required', 'validation_error');
    return json(res, 200, preflight(body));
  }
  if (p === '/instances' && M === 'POST') {
    if (!body.engine || !refOf(body)) return problem(res, 422, 'Validation error', 'engine and a model are required', 'validation_error');
    if (body.host_id && !hostOr404(res, body.host_id)) return;
    const pf = preflight(body);
    if (!pf.ok) return problem(res, 409, 'Preflight failed', pf.checks.find((c) => c.level === 'block')?.message || 'preflight failed', 'preflight_failed', { preflight: pf });
    const ref = refOf(body), hostId = body.host_id || 'local';
    return json(res, 201, view(makeInstance({ engine: body.engine, ref, host_id: hostId, params: redactParams(body.params), gpu_ids: pf.gpu_ids, name: body.name, profile_id: body.profile_id, ttl_idle_s: body.ttl_idle_s, image: imageFor(hostId, body.engine, body.image), fit: pf.fit })));
  }
  if ((m = p.match(/^\/instances\/([^/]+)(?:\/(stop|start|restart|logs|command))?(?:\/(stream))?$/))) {
    const inst = state.instances.find((x) => x.id === m[1]); if (!inst) return problem(res, 404, 'Not found', `no such instance: ${m[1]}`, 'instance_not_found');
    const c = F.controlsFor(inst);
    if (m[2] === 'logs' && m[3] === 'stream') {
      if (!c.logs) return notManaged(res, inst, `logs are only available for engines the console started (${inst.name} is ${inst.management})`);
      const set = streams.logs.get(inst.id) || new Set(); streams.logs.set(inst.id, set); openSSE(req, res, once ? null : set);
      for (const l of (logs.get(inst.id) || []).slice(-Number(q.get('tail') || 100))) sse(res, l, 'log');
      if (once) res.end(); return;
    }
    if (m[2] === 'logs') { if (!c.logs) return notManaged(res, inst, `logs are only available for engines the console started (${inst.name} is ${inst.management})`); res.writeHead(200, { 'Content-Type': 'text/plain; charset=utf-8' }); return res.end((logs.get(inst.id) || []).slice(-Number(q.get('tail') || 200)).join('\n')); }
    if (m[2] === 'command') {
      if (inst.management !== 'managed') return notManaged(res, inst);
      const flags = Object.entries(inst.params).map(([k, v]) => `--${k.replace(/_/g, '-')}${typeof v === 'boolean' ? '' : ' ' + JSON.stringify(v)}`).join(' \\\n  ');
      return json(res, 200, { docker_run: `docker run -d --gpus '"device=${inst.gpu_ids.join(',')}"' -p 127.0.0.1:${inst.port}:8000 \\\n  ${inst.image} \\\n  --model ${inst.repo_id} ${flags}`, compose_yaml: `services:\n  ${inst.name}:\n    image: ${inst.image}\n    command: ["--model", "${inst.repo_id}"]\n`, engine_cli: `${inst.engine === 'vllm' ? 'vllm serve' : 'python -m sglang.launch_server --model-path'} ${inst.repo_id} ${flags}` });
    }
    if (M === 'GET') return json(res, 200, view(inst));
    if (M === 'PATCH') { if (inst.management !== 'managed') return notManaged(res, inst); for (const k of ['pinned', 'ttl_idle_s', 'name']) if (body[k] !== undefined && body[k] !== null) inst[k] = body[k]; return json(res, 200, view(inst)); }
    if (M === 'DELETE') {
      if (inst.management !== 'managed') return notManaged(res, inst);
      if (!F.hostUp(inst.host_id)) return fail(res, F.NODE_ERR(F.hostDef(inst.host_id).label, 'node_unavailable', `the link to ${F.hostDef(inst.host_id).label} is not up`));
      removeInstance(inst); return noContent(res);
    }
    if (m[2] === 'stop' || m[2] === 'start' || m[2] === 'restart') {
      const r = F.instanceAction(inst, m[2]); if (r.error) return fail(res, r.error);
      if (m[2] === 'stop') { inst.state = 'stopped'; inst.started_at = null; addLog(inst.id, 'INFO stopped by user'); return json(res, 200, view(inst)); }
      Object.assign(inst, { state: 'loading', error: null, phase: 'loading_weights', progress_pct: 40 }); addLog(inst.id, `INFO ${m[2]} requested`);
      setTimeout(() => { if (inst.state === 'loading') { inst.state = 'ready'; inst.started_at = now(); inst.phase = 'ready'; inst.progress_pct = 100; } }, 3500);
      return json(res, 200, view(inst));
    }
  }

  // ---- profiles ----
  if (p === '/profiles' && M === 'GET') return json(res, 200, page(state.profiles));
  if (p === '/profiles' && M === 'POST') { const pr = { id: uid('prof_'), name: body.name, engine: body.engine, repo_id: body.repo_id ?? null, params: redactParams(body.params), description: body.description || '', created_at: now(), updated_at: now() }; state.profiles.push(pr); return json(res, 201, pr); }
  if (p === '/profiles/import' && M === 'POST') {
    const yaml = typeof body.yaml === 'string' ? body.yaml : body.__raw || '';
    const nm = /^name:\s*(.+)$/m.exec(yaml); if (!nm) return problem(res, 400, 'Bad request', 'profile YAML needs a name: line', 'invalid_profile');
    const pr = { id: uid('prof_'), name: nm[1].trim(), engine: /^engine:\s*(\w+)/m.exec(yaml)?.[1] || 'vllm', repo_id: null, params: {}, description: '', created_at: now(), updated_at: now() }; state.profiles.push(pr); return json(res, 201, pr);
  }
  if ((m = p.match(/^\/profiles\/([^/]+)\/export$/))) { const pr = state.profiles.find((x) => x.id === m[1]); if (!pr) return problem(res, 404, 'Not found', m[1]); res.writeHead(200, { 'Content-Type': 'application/yaml' }); return res.end(`name: ${pr.name}\nengine: ${pr.engine}\nparams: ${JSON.stringify(pr.params)}\n`); }
  if ((m = p.match(/^\/profiles\/([^/]+)$/))) { const pr = state.profiles.find((x) => x.id === m[1]); if (!pr) return problem(res, 404, 'Not found', `no such profile: ${m[1]}`); if (M === 'PUT') Object.assign(pr, body, { updated_at: now() }); if (M === 'DELETE') { state.profiles = state.profiles.filter((x) => x !== pr); return noContent(res); } return json(res, 200, pr); }

  // ---- metrics ----
  if (p === '/metrics/system') {
    const id = q.get('host') || 'local'; if (!hostOr404(res, id)) return; if (!needUp(res, id)) return;
    const secs = { '5m': 300, '15m': 900, '1h': 3600, '24h': 86400 }[q.get('window') || '15m'] || 900; const n = 120, base = F.hostMem(id);
    const pts = Array.from({ length: n }, (_, i) => { const t = now() - secs + (i * secs) / n; const w = Math.sin(t / 40) * 0.5 + 0.5; return { t, values: { ram_used_gib: base.used_gib - 3 + w * 6, ram_available_gib: base.available_gib + 3 - w * 6, shmem_gib: base.shmem_gib, swap_used_gib: base.swap_used_gib } }; });
    return json(res, 200, { window: q.get('window') || '15m', resolution: 'raw', host_id: id, points: pts });
  }
  if ((m = p.match(/^\/metrics\/instances\/([^/]+)$/))) { const secs = { '5m': 300, '15m': 900, '1h': 3600, '24h': 86400 }[q.get('window') || '15m'] || 900; const cut = now() - secs; const h = (history.get(m[1]) || []).filter((s) => s.t >= cut && Object.keys(s.values).length); const step = Math.max(1, Math.ceil(h.length / 300)); return json(res, 200, { instance_id: m[1], window: q.get('window') || '15m', resolution: 'raw', history_since: state.instances.find((i) => i.id === m[1])?.history_since ?? null, points: h.filter((_, i) => i % step === 0) }); }
  if (p === '/metrics/stream') { openSSE(req, res, once ? null : streams.metrics); for (const i of state.instances.filter((x) => x.state === 'ready')) { const s = history.get(i.id)?.at(-1); if (s && Object.keys(s.values).length) sse(res, { instance_id: i.id, ...s }, 'metrics'); } if (once) res.end(); return; }

  // ---- benchmark / chat / arena / usage ----
  if (p === '/bench' && M === 'POST') { const inst = state.instances.find((i) => i.id === body.instance_id); if (!inst || inst.state !== 'ready') return problem(res, 409, 'Instance not ready', 'benchmarks need a ready instance', 'instance_not_ready'); if (inst.management !== 'managed' && body.confirm_external !== true) return problem(res, 409, 'Confirmation required', 'this engine is not owned by the console; resend with confirm_external: true', 'confirm_external_required'); if (typeof body.suite === 'string' && !SUITES[body.suite]) return problem(res, 400, 'Bad request', `unknown suite '${body.suite}'`, 'unknown_suite'); res.writeHead(202, { 'Content-Type': 'application/json' }); return res.end(JSON.stringify(runBench(inst, body.suite || 'quick'))); }
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

  // ---- settings (hf_cache_dir is read-only; host_image_pins is host id -> engine -> image, checked against that host's policy) ----
  if (p === '/settings') {
    if (M === 'PUT') {
      if (body.host_image_pins !== undefined) {
        const pins = body.host_image_pins;
        if (pins === null || typeof pins !== 'object' || Array.isArray(pins)) return problem(res, 422, 'Validation error', 'host_image_pins must be an object of host id to engine to image', 'validation_error');
        for (const [hid, per] of Object.entries(pins)) {
          const d = F.hostDef(hid); if (!d) return problem(res, 422, 'Validation error', `host_image_pins: no such host ${hid}`, 'unknown_host');
          for (const [eng, ref] of Object.entries(per || {})) {
            if (!engineMeta.some((e) => e.id === eng)) return problem(res, 422, 'Validation error', `host_image_pins: no such engine ${eng}`, 'validation_error');
            if (d.policy && !imageAllowed(d.policy.engine_images, ref)) return problem(res, 422, 'Validation error', `host_image_pins: ${ref} is not allowed by ${d.label}'s policy (engine_images)`, 'image_not_allowed');
          }
        }
        state.settings.host_image_pins = JSON.parse(JSON.stringify(pins));
      }
      Object.assign(state.settings, Object.fromEntries(Object.entries(body).filter(([k]) => ['default_gpu_ids', 'idle_ttl_s'].includes(k))));
    }
    return json(res, 200, state.settings);
  }
  if (p === '/keys' && M === 'GET') return json(res, 200, state.keys);
  if (p === '/keys' && M === 'POST') { const secret = `ec_${crypto.randomBytes(24).toString('base64url')}`; const k = { id: uid('k_'), name: body.name, role: body.role || 'viewer', prefix: secret.slice(0, 7), created_at: now(), last_used_at: null, secret: null }; state.keys.push(k); return json(res, 201, { ...k, secret }); }
  if ((m = p.match(/^\/keys\/([^/]+)$/)) && M === 'DELETE') { state.keys = state.keys.filter((k) => k.id !== m[1]); return noContent(res); }
  return problem(res, 404, 'Not found', `${M} ${p}`, 'not_found');
}

// ---------------- dev-only helpers for headless screenshots ----------------
// /__shot serves index.html plus a slow 1px image so the browser's `load` event (when screenshot tools fire) is held until the
// SPA has finished its async rendering. ?diag=1 adds a same-origin script that records script errors, CSP violations and layout
// facts and POSTs them to /__report (read the last one with GET /__report). ?errs=1 also paints the errors on the page.
const DIAG = `
const errs = [];
const note = (s) => { errs.push(String(s).slice(0, 400)); const box = document.getElementById('__errs'); if (box) { box.textContent = errs.join('\\n'); box.hidden = false; } };
window.addEventListener('error', (e) => note('error: ' + e.message + ' @' + (e.filename || '').split('/').pop() + ':' + e.lineno));
window.addEventListener('unhandledrejection', (e) => note('rejection: ' + (e.reason && e.reason.message || e.reason)));
document.addEventListener('securitypolicyviolation', (e) => note('csp: ' + e.violatedDirective + ' blocked ' + e.blockedURI));
// popover/dialog open-close timing is noisy: only traced when asked for (?toggles=1)
if (new URLSearchParams(location.search).get('toggles')) {
  document.addEventListener('toggle', (e) => note('toggle ' + e.newState + ' on ' + (e.target.id || e.target.tagName) + ' @' + Math.round(performance.now())), true);
  document.addEventListener('beforetoggle', (e) => note('beforetoggle ' + e.newState + ' @' + Math.round(performance.now())), true);
}
for (const k of ['error', 'warn']) { const o = console[k]; console[k] = (...a) => { note('console.' + k + ': ' + a.map(String).join(' ')); o.apply(console, a); }; }
if (new URLSearchParams(location.search).get('errs')) { const b = document.createElement('pre'); b.id = '__errs'; b.hidden = true; Object.assign(b.style, { position: 'fixed', left: '0', right: '0', bottom: '0', zIndex: '999', margin: '0', padding: '6px 10px', background: '#300', color: '#fcc', fontSize: '11px', maxHeight: '30vh', overflow: 'auto', whiteSpace: 'pre-wrap' }); document.body.append(b); }
const wait = Number(new URLSearchParams(location.search).get('diagms') || 3500);
setTimeout(() => {
  const de = document.documentElement, vw = innerWidth;
  const label = (el) => el.tagName.toLowerCase() + (el.id ? '#' + el.id : '') + (el.className && el.className.baseVal === undefined ? '.' + String(el.className).trim().split(/\\s+/).slice(0, 2).join('.') : '') + ' ' + (el.textContent || '').trim().slice(0, 30);
  const overflow = [...document.querySelectorAll('main *, .topbar *')].filter((el) => { const r = el.getBoundingClientRect(); const cs = getComputedStyle(el); return r.width > 0 && r.right > vw + 1 && cs.position !== 'fixed' && !el.closest('.tablewrap') && !el.closest('pre') && !el.closest('.sr-only'); })
    .map((el) => [el, el.getBoundingClientRect().right]).sort((a, b) => b[1] - a[1]).slice(0, 8).map(([el, right]) => label(el) + ' right=' + Math.round(right) + ' w=' + Math.round(el.getBoundingClientRect().width));
  // text that should never reach a person: a stringified null/undefined/NaN/object leaking out of a template
  const txt = (document.querySelector('main')?.innerText || '') + '\\n' + [...document.querySelectorAll('dialog[open]')].map((d) => d.innerText).join('\\n');
  const badText = [...txt.matchAll(/(^|[^\\w.-])(null|undefined|NaN|\\[object \\w+\\])($|[^\\w-])|\\b(GiB|GB|TiB|MiB)\\s+(GiB|GB|TiB|MiB)\\b/gm)].slice(0, 5).map((m) => txt.slice(Math.max(0, m.index - 30), m.index + 40).replace(/\\n/g, ' | '));
  const wide = [...document.querySelectorAll('.tablewrap, pre, .stack, .grid')].filter((el) => el.scrollWidth > el.clientWidth + 1 && getComputedStyle(el).overflowX !== 'visible').slice(0, 6).map((el) => label(el) + ' scroll=' + el.scrollWidth + ' client=' + el.clientWidth);
  const tiny = [...document.querySelectorAll('button, a.btn, .info-btn')].filter((el) => { const r = el.getBoundingClientRect(); return r.width > 0 && (r.width < 24 || r.height < 24) && getComputedStyle(el).visibility !== 'hidden'; }).length;
  const sh = document.querySelector('.shell'); const sidebar = document.querySelector('.sidebar');
  fetch('/__report', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ at: location.href, scrollY: Math.round(window.scrollY), vw, vh: innerHeight, scrollW: de.scrollWidth, scrollH: de.scrollHeight, errs, overflow, sideways: wide, badText, popoversOpen: (() => { try { return document.querySelectorAll(':popover-open').length; } catch (e) { return -1; } })(), dialogsOpen: document.querySelectorAll('dialog[open]').length, popRect: (() => { try { const e = document.querySelector(':popover-open'); if (!e) return null; const r = e.getBoundingClientRect(); const cs = getComputedStyle(e); return [Math.round(r.left), Math.round(r.top), Math.round(r.width), Math.round(r.height), cs.position, cs.display, cs.visibility, cs.opacity, cs.zIndex]; } catch (e) { return String(e); } })(), focus: document.activeElement ? (document.activeElement.tagName.toLowerCase() + '.' + String(document.activeElement.className).slice(0, 30)) : null, tipButtons: document.querySelectorAll('.info-btn').length, smallTargets: tiny,
    sidebarH: sidebar ? Math.round(sidebar.getBoundingClientRect().height) : null, rows: sh ? getComputedStyle(sh).gridTemplateRows : null, view: document.querySelector('main > *') ? document.querySelector('main > *').tagName.toLowerCase() : null, text: document.querySelector('main') ? document.querySelector('main').innerText.length : 0 }) });
}, wait);
`;
let lastReport = null;
function serveShot(req, res, url) {
  const ms = Math.min(15000, Number(url.searchParams.get('ms') || 4000));
  if (url.pathname === '/__report') {
    if (req.method === 'POST') return readBody(req).then((b) => { lastReport = b; res.writeHead(204); res.end(); });
    return json(res, 200, lastReport || {});
  }
  if (url.pathname === '/__diag.js') { res.writeHead(200, { 'Content-Type': 'text/javascript' }); return res.end(DIAG); }
  if (url.pathname === '/__demo.js') {
    const name = url.searchParams.get('name');
    const W = `const until=async(f)=>{for(let i=0;i<80;i++){const v=f();if(v)return v;await new Promise(r=>setTimeout(r,100));}};const sleep=(ms)=>new Promise(r=>setTimeout(r,ms));const byText=(sel,t)=>[...document.querySelectorAll(sel)].find(x=>x.textContent.trim().includes(t));`;
    const demos = {
      chat: W + `const ta=await until(()=>document.querySelector('ec-chat .composer textarea')); await until(()=>document.querySelector('ec-chat select option[value^=inst]')); await sleep(300); ta.value='Show me a fibonacci function and a comparison table.'; document.querySelector('ec-chat .composer .btn.primary').click();`,
      wontfit: W + `const b=await until(()=>[...document.querySelectorAll('ec-launch .btn')].find(x=>x.textContent.trim()==='long-context')); b.click();`,
      benchext: W + `const s=await until(()=>document.querySelector('ec-bench select')); await sleep(400); s.value='ext-local-vllm'; s.dispatchEvent(new Event('change')); [...document.querySelectorAll('ec-bench .btn')].find(x=>x.textContent.trim()==='Run').click();`,
      palette: W + `await until(()=>document.querySelector('ec-palette')); await sleep(800); window.dispatchEvent(new Event('ec:palette')); await sleep(600); const i=document.querySelector('ec-palette input'); i.value='chat'; i.dispatchEvent(new Event('input'));`,
      arena: W + `const ta=await until(()=>document.querySelector('ec-arena textarea')); ta.value='Explain KV cache in two sentences.'; [...document.querySelectorAll('ec-arena .btn')].find(x=>x.textContent.includes('Run match')).click();`,
      // scrolls the page (?y=2800) once the view has rendered, so lower parts of a long page can be screenshotted
      scroll: W + `const sp=new URLSearchParams(location.search); await sleep(Number(sp.get('wait')||3800)); const sel=sp.get('sel'); if(sel){ const el=document.querySelector(sel); if(el) el.scrollIntoView({block:'start'}); } else window.scrollTo(0, Number(sp.get('y')||0));`,
      // opens the n-th info popover (?n=2) on the current view
      tip: W + `const sp=new URLSearchParams(location.search); const n=Number(sp.get('n')||0); await until(()=>document.querySelectorAll('main .info-btn').length>n); await sleep(Number(sp.get('settle')||2500)); document.querySelectorAll('main .info-btn')[n].click(); await sleep(300); const t=document.querySelector('.info-pop button.term-chip'); if(sp.get('term')&&t) t.click();`,
      // logs (through the diagnostics' console.warn capture) the class and width of every stacked-table host
      probe: W + `await sleep(3500); console.warn('PROBE '+[...document.querySelectorAll('.stack-host')].map(h=>{ const t=h.querySelector('table'); return h.className+' host='+h.clientWidth+' tableScroll='+t.scrollWidth+' tableW='+Math.round(t.getBoundingClientRect().width); }).join(' | '));`,
      // clicks the first button whose text contains ?btn= (after the view settled), then optionally a second ?btn2=
      click: W + `await sleep(Number(new URLSearchParams(location.search).get('wait')||1500)); const sp=new URLSearchParams(location.search); const b=await until(()=>byText('main button, main a.btn, dialog button',sp.get('btn'))); b.click(); if(sp.get('btn2')){ await sleep(900); const c=await until(()=>byText('dialog[open] button',sp.get('btn2'))||byText('main button',sp.get('btn2'))); c.click(); }`,
    };
    res.writeHead(200, { 'Content-Type': 'text/javascript' }); return res.end(demos[name] || '');
  }
  if (url.pathname === '/__slow.gif') return setTimeout(() => { res.writeHead(200, { 'Content-Type': 'image/gif' }); res.end(Buffer.from('R0lGODlhAQABAAAAACw=', 'base64')); }, ms);
  // ?demo=<name> adds a same-origin script (CSP-safe) that drives the UI, e.g. sends a chat message.
  const demo = /^\w+$/.test(url.searchParams.get('demo') || '') ? `<script type="module" src="/__demo.js?name=${url.searchParams.get('demo')}&${url.searchParams.toString()}"></script>` : '';
  const diag = url.searchParams.get('diag') || url.searchParams.get('errs') ? '<script type="module" src="/__diag.js"></script>' : '';
  const html = fs.readFileSync(path.join(ROOT, 'index.html'), 'utf8').replace('</body>', `<img src="/__slow.gif?ms=${ms}" width="1" height="1" alt="">${diag}${demo}</body>`);
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
    if (['/__shot', '/__slow.gif', '/__demo.js', '/__diag.js', '/__report'].includes(url.pathname)) return serveShot(req, res, url);
    return serveStatic(req, res, url);
  } catch (e) { if (!res.headersSent) problem(res, 500, 'Internal error', String(e.message)); else res.end(); }
});
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  server.listen(PORT, HOST, () => console.log(`mock engine-console: http://${HOST}:${server.address().port}/  (${KEY ? 'bearer key required' : 'no auth'})`));
}
