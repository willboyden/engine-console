import test, { before, after } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { server } from '../dev/mock-server.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
let base;
before(async () => { await new Promise((r) => server.listen(0, '127.0.0.1', r)); base = `http://127.0.0.1:${server.address().port}`; });
after(() => { server.closeAllConnections?.(); server.close(); setTimeout(() => process.exit(0), 50).unref(); });
const H = { 'content-type': 'application/json', 'x-engine-console': '1' };
const j = async (p, o) => { const r = await fetch(base + p, o); return { r, b: r.headers.get('content-type')?.includes('json') ? await r.json() : await r.text() }; };

test('every module reachable from index.html (static + dynamic view imports) is served 200', async () => {
  const seen = new Set(); const queue = ['/index.html'];
  const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
  for (const m of html.matchAll(/(?:src|href)="([^"#]+)"/g)) if (!m[1].startsWith('data:')) queue.push('/' + m[1]);
  const app = fs.readFileSync(path.join(root, 'js/app.js'), 'utf8');
  for (const m of app.matchAll(/mod: '(\w+)'/g)) queue.push(`/js/views/${m[1]}.js`);
  queue.push('/i18n/en.json');
  while (queue.length) {
    const u = queue.pop(); if (seen.has(u)) continue; seen.add(u);
    const { r, b } = await j(u); assert.equal(r.status, 200, u);
    if (u.endsWith('.js')) for (const m of b.matchAll(/(?:from|import)\s+['"](\.[^'"]+)['"]/g)) queue.push(path.posix.normalize(path.posix.join(path.posix.dirname(u), m[1])));
  }
  const walk = (d) => fs.readdirSync(d, { withFileTypes: true }).flatMap((e) => (e.isDirectory() ? walk(path.join(d, e.name)) : [path.join(d, e.name)]));
  const orphans = walk(path.join(root, 'js')).map((f) => '/' + path.relative(root, f).split(path.sep).join('/')).filter((u) => !seen.has(u));
  assert.deepEqual(orphans, [], 'js files never referenced from index.html/app.js');
});

test('static server refuses traversal and hides dev/ and tests/', async () => {
  for (const p of ['/dev/mock-server.mjs', '/tests/api.test.mjs', '/..%2f..%2fetc/passwd']) assert.equal((await j(p)).r.status, 404, p);
  assert.match((await j('/index.html')).r.headers.get('content-security-policy'), /default-src 'self'/);
});

test('API: engines, params, hardware, fit, instances shapes', async () => {
  const eng = (await j('/api/v1/engines')).b; assert.deepEqual(eng.map((e) => e.id), ['vllm', 'sglang']);
  for (const e of eng) { const specs = (await j(`/api/v1/engines/${e.id}/params`)).b; assert.ok(specs.length > 20); assert.ok(specs.every((s) => s.key && s.flag && s.type && s.group)); }
  const hw = (await j('/api/v1/hardware')).b; assert.equal(hw.gpus.length, 2);
  const fit = (await j('/api/v1/fit', { method: 'POST', headers: H, body: JSON.stringify({ engine: 'vllm', repo_id: 'Qwen/Qwen3.6-35B-A3B-FP8', params: { max_model_len: 32768 }, gpu_ids: [0] }) })).b;
  assert.equal(fit.verdict, 'fits'); assert.equal(fit.per_gpu.length, 1);
  const bigger = (await j('/api/v1/fit', { method: 'POST', headers: H, body: JSON.stringify({ engine: 'vllm', repo_id: 'Qwen/Qwen3.6-35B-A3B-FP8', params: { max_model_len: 262144, max_num_seqs: 32 }, gpu_ids: [0] }) })).b;
  assert.ok(bigger.per_gpu[0].kv_cache_gib > fit.per_gpu[0].kv_cache_gib * 4, 'KV grows with context');
  const tp = (await j('/api/v1/fit', { method: 'POST', headers: H, body: JSON.stringify({ engine: 'vllm', repo_id: 'meta-llama/Llama-3.3-70B-Instruct', params: { tensor_parallel_size: 2 }, gpu_ids: [0] }) })).b;
  assert.ok(tp.compat.some((c) => c.level === 'block'));
  assert.ok((await j('/api/v1/instances')).b.items.length >= 1);
  assert.equal((await j('/api/v1/nope')).r.status, 404);
});

test('mutating calls without X-Engine-Console are rejected (CSRF guard), GETs are not', async () => {
  const bad = await fetch(`${base}/api/v1/fit`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: '{}' });
  assert.equal(bad.status, 403); assert.equal((await bad.json()).code, 'csrf_header_required');
  assert.equal((await fetch(`${base}/api/v1/hardware`)).status, 200);
});

test('list endpoints are {items, next_cursor}; bare lists only where the backend has them', async () => {
  for (const p of ['/instances', '/downloads', '/models', '/profiles', '/bench', '/conversations', '/prompts', '/audit']) { const b = (await j(`/api/v1${p}`)).b; assert.ok(Array.isArray(b.items) && 'next_cursor' in b, p); }
  for (const p of ['/engines', '/usage?group_by=model', '/keys', '/arena/leaderboard']) assert.ok(Array.isArray((await j(`/api/v1${p}`)).b), p);
});

test('SSE ?once=true sends the first snapshot and closes', async () => {
  const dl = await fetch(`${base}/api/v1/downloads/stream?once=true`); const t = await dl.text(); assert.match(t, /event: snapshot/);
  const mt = await fetch(`${base}/api/v1/metrics/stream?once=true`); assert.match(await mt.text(), /event: metrics/);
});

test('instance preflight, patch and delete follow the v1.1 shapes', async () => {
  const pf = (await j('/api/v1/instances/preflight', { method: 'POST', headers: H, body: JSON.stringify({ engine: 'vllm', repo_id: 'Qwen/Qwen3.6-35B-A3B-FP8', params: {}, gpu_ids: [1] }) })).b;
  assert.equal(pf.ok, true); assert.ok(pf.checks.length && pf.fit.per_gpu.length);
  const inst = (await j('/api/v1/instances')).b.items[0];
  const patched = (await j(`/api/v1/instances/${inst.id}`, { method: 'PATCH', headers: H, body: JSON.stringify({ ttl_idle_s: 600, pinned: true }) })).b;
  assert.equal(patched.ttl_idle_s, 600); assert.equal(patched.pinned, true);
  assert.equal((await fetch(`${base}/api/v1/instances/nope`, { method: 'DELETE', headers: H })).status, 404);
});

test('errors are problem+json', async () => {
  const { r, b } = await j('/api/v1/instances/zzz'); assert.equal(r.status, 404); assert.match(r.headers.get('content-type'), /problem\+json/); assert.ok(b.code);
});

test('chat completions streams reasoning then content then [DONE]', async () => {
  const inst = (await j('/api/v1/instances')).b.items.find((i) => i.state === 'ready');
  const r = await fetch(`${base}/api/v1/chat/completions`, { method: 'POST', headers: H, body: JSON.stringify({ instance_id: inst.id, messages: [{ role: 'user', content: 'hi' }], stream: true }) });
  const text = await r.text();
  assert.match(text, /reasoning_content/); assert.match(text, /"content"/); assert.match(text, /data: \[DONE\]\n\n$/);
});

test('v1.1 hardening: bad Host -> 421, secret_in_argv -> 422, secrets come back as "[set]"', async () => {
  const http = await import('node:http');
  const status = await new Promise((r) => { const q = http.request({ host: '127.0.0.1', port: server.address().port, path: '/api/v1/hardware', headers: { host: 'evil.example' } }, (res) => { res.resume(); r(res.statusCode); }); q.end(); });
  assert.equal(status, 421);
  const sg = await j('/api/v1/instances', { method: 'POST', headers: H, body: JSON.stringify({ engine: 'sglang', repo_id: 'Qwen/Qwen3.6-35B-A3B-FP8', params: { api_key: 'sekret-value' }, gpu_ids: [1] }) });
  assert.equal(sg.r.status, 422); assert.equal(sg.b.code, 'secret_in_argv'); assert.ok(!JSON.stringify(sg.b).includes('sekret-value'));
  const ok = await j('/api/v1/instances', { method: 'POST', headers: H, body: JSON.stringify({ engine: 'vllm', repo_id: 'Qwen/Qwen3.6-27B-FP8', params: { hf_token: 'hf_realvalue', max_model_len: 4096 }, gpu_ids: [1] }) });
  assert.equal(ok.r.status, 201); assert.equal(ok.b.params.hf_token, '[set]'); assert.equal(ok.b.params.max_model_len, 4096);
  assert.ok(!JSON.stringify((await j('/api/v1/audit')).b).includes('hf_realvalue'), 'audit must not hold the secret');
  const prof = (await j('/api/v1/profiles')).b.items.find((p) => p.name === 'qwen-agent'); assert.equal(prof.params.hf_token, '[set]');
});

test('settings: hf_cache_dir cannot be changed through PUT', async () => {
  const before = (await j('/api/v1/settings')).b.hf_cache_dir;
  const after = (await j('/api/v1/settings', { method: 'PUT', headers: H, body: JSON.stringify({ hf_cache_dir: '/tmp/x', idle_ttl_s: 60 }) })).b;
  assert.equal(after.hf_cache_dir, before); assert.equal(after.idle_ttl_s, 60);
});

test('SSE cap: the 9th concurrent live stream gets 429 too_many_streams', async () => {
  const ctl = new AbortController(); const open = [];
  for (let i = 0; i < 8; i++) { const r = await fetch(`${base}/api/v1/metrics/stream`, { signal: ctl.signal }); assert.equal(r.status, 200); open.push(r); }
  const ninth = await fetch(`${base}/api/v1/metrics/stream`, { signal: ctl.signal });
  assert.equal(ninth.status, 429); assert.equal((await ninth.json()).code, 'too_many_streams');
  ctl.abort();
});

test('external fixtures: one per state, monitor-only rules, discover, bench confirmation, metrics history_since', async () => {
  const list = (await j('/api/v1/instances')).b.items; const ext = list.filter((i) => i.managed === false);
  assert.deepEqual([...new Set(ext.map((i) => i.state))].sort(), ['auth_required', 'ready', 'stopped', 'unreachable']);
  for (const i of ext) { assert.equal(i.source, 'external'); assert.ok(i.image && i.endpoint && Array.isArray(i.served_models)); }
  assert.ok(ext.find((i) => i.state === 'auth_required').state_reason && ext.find((i) => i.state === 'unreachable').state_reason);
  const ready = ext.find((i) => i.id === 'ext_vllm');
  for (const [m, p, b] of [['POST', 'stop'], ['POST', 'restart'], ['DELETE', ''], ['PATCH', '', { pinned: true }]]) {
    const r = await j(`/api/v1/instances/${ready.id}${p ? '/' + p : ''}`, { method: m, headers: H, body: b ? JSON.stringify(b) : undefined });
    assert.equal(r.r.status, 409, `${m} ${p}`); assert.equal(r.b.code, 'instance_not_managed');
  }
  const d = await j('/api/v1/instances/discover', { method: 'POST', headers: H, body: '{}' }); assert.equal(d.r.status, 200); assert.ok(d.b.items.length >= list.length);
  const nb = await j('/api/v1/bench', { method: 'POST', headers: H, body: JSON.stringify({ instance_id: ready.id, suite: 'quick' }) }); assert.equal(nb.r.status, 409);
  const ok = await j('/api/v1/bench', { method: 'POST', headers: H, body: JSON.stringify({ instance_id: ready.id, suite: 'quick', confirm_external: true }) }); assert.equal(ok.r.status, 202);
  const mm = (await j(`/api/v1/metrics/instances/${ready.id}`)).b; assert.ok(mm.history_since && mm.points.length);
  assert.equal((await j('/api/v1/metrics/instances/ext_ollama')).b.points.length, 0, 'engine without /metrics has an empty series');
});
