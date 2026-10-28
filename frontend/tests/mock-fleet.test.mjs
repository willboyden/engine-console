// Contract tests for the v2 (fleet) endpoints of the dev mock server: shapes follow backend domain/fleet.rs, refs.rs, learn.rs and
// docs/ARCHITECTURE.md section 10.8. The mock must keep matching the contract exactly, because the UI is built against it.
import test, { before, after } from 'node:test';
import assert from 'node:assert/strict';
import { server } from '../dev/mock-server.mjs';
import { CONCEPT_FAMILIES, METRIC_KEYS } from '../js/learn-keys.js';

let base;
before(async () => { await new Promise((r) => server.listen(0, '127.0.0.1', r)); base = `http://127.0.0.1:${server.address().port}`; });
after(() => { server.closeAllConnections?.(); server.close(); setTimeout(() => process.exit(0), 50).unref(); });
const H = { 'content-type': 'application/json', 'x-engine-console': '1' };
const j = async (p, o) => { const r = await fetch(base + p, o); const ct = r.headers.get('content-type') || ''; return { r, s: r.status, b: ct.includes('json') ? await r.json() : await r.text() }; };
const get = (p) => j(`/api/v1${p}`);
const post = (p, body) => j(`/api/v1${p}`, { method: 'POST', headers: H, body: JSON.stringify(body ?? {}) });
const keys = (o, expected, where) => assert.deepEqual(Object.keys(o).sort(), [...expected].sort(), where);
const isProblem = (res, status, code) => { assert.equal(res.s, status); assert.match(res.r.headers.get('content-type'), /problem\+json/); assert.equal(res.b.code, code, JSON.stringify(res.b)); assert.ok(res.b.detail && res.b.title && res.b.status === status); };

const LINK = ['state', 'error', 'retry_in_s', 'notes', 'last_seen', 'connected_since', 'rtt_ms', 'skew_s'];
const NODE = ['version', 'hostname', 'arch', 'os', 'kernel', 'cpu_model', 'cpu_cores', 'cpu_threads', 'python', 'driver', 'cuda', 'gpu_source', 'forced_cmd', 'via_ssh'];
const GPU = ['index', 'uuid', 'name', 'total_gib', 'used_gib', 'free_gib', 'util_pct', 'temp_c', 'power_w', 'power_limit_w', 'fan_pct', 'compute_capability', 'unified_memory', 'procs'];
const PROC = ['pid', 'name', 'container', 'instance_id', 'used_gib'];
const POLICY = ['path', 'mode', 'error', 'engine_images', 'gateway_image', 'engine_network', 'port_range', 'allow_pull', 'allow_delete', 'allow_ipc_host', 'external_control', 'max_shm_gib'];
const DOCKER = ['available', 'context', 'rootless', 'version', 'nvidia_runtime', 'error'];
const MOUNT = ['mount', 'label', 'fstype', 'source', 'device', 'model', 'size_bytes', 'used_bytes', 'free_bytes', 'readonly', 'remote', 'remote_host', 'remote_host_id', 'remote_path', 'roots'];
const ROOT = ['id', 'path', 'kind', 'writable', 'label', 'mount', 'shared_from', 'visible_on'];
const SUMMARY = ['id', 'label', 'remote', 'link', 'node', 'gpus', 'memory', 'cpu', 'docker', 'policy', 'counts', 'alerts'];
const DETAIL = [...SUMMARY, 'disks', 'roots', 'images', 'scripts', 'addresses'];
const COUNTS = ['engines_running', 'engines_total', 'models', 'models_bytes'];
const MEM = ['total_gib', 'used_gib', 'available_gib', 'free_gib', 'cached_gib', 'shmem_gib', 'swap_total_gib', 'swap_used_gib', 'updated_at'];
const INSTANCE = ['id', 'name', 'engine', 'repo_id', 'params', 'gpu_ids', 'gpu_uuids', 'port', 'container_name', 'container_port', 'internal_endpoint', 'image', 'state', 'phase', 'progress_pct', 'pinned', 'ttl_idle_s', 'profile_id', 'error', 'last_logs', 'fit',
  'created_at', 'started_at', 'last_request_at', 'uptime_s', 'managed', 'source', 'endpoint', 'served_models', 'state_reason', 'history_since', 'host_memory', 'host_id', 'host_label', 'management', 'script', 'vram', 'model', 'model_location', 'controls'];
const COPY = ['id', 'host_id', 'host_label', 'root_id', 'root_label', 'rel', 'path', 'size_bytes', 'files', 'mtime', 'atime', 'drive', 'visible_on', 'writable', 'deletable', 'in_use_by'];
const LIBMODEL = ['key', 'name', 'format', 'revision', 'quant', 'params', 'arch', 'context', 'shards', 'size_bytes', 'engines', 'copies', 'warnings'];
const SCRIPT = ['host_id', 'name', 'label', 'engine', 'description', 'status', 'detail', 'checked_at', 'actions', 'containers'];
const JOB = ['id', 'host_id', 'name', 'action', 'arg', 'state', 'rc', 'started_at', 'finished_at', 'output_tail'];
const DOWNLOAD = ['id', 'repo_id', 'revision', 'commit_sha', 'allow_patterns', 'state', 'total_bytes', 'done_bytes', 'speed_bps', 'eta_s', 'files_total', 'files_done', 'current_file', 'error', 'error_code', 'created_at', 'updated_at', 'host_id', 'root_id', 'kind', 'source_copy_id'];

test('GET /hosts: {items, next_cursor} of HostSummary covering every link state, with generic names', async () => {
  const { s, b } = await get('/hosts');
  assert.equal(s, 200); assert.ok(Array.isArray(b.items) && 'next_cursor' in b && b.next_cursor === null);
  for (const h of b.items) {
    keys(h, SUMMARY, `host ${h.id}`); keys(h.link, LINK, `link ${h.id}`); keys(h.counts, COUNTS); assert.match(h.id, /^[a-z0-9][a-z0-9-]{0,31}$/); assert.equal(typeof h.remote, 'boolean'); assert.ok(Array.isArray(h.alerts));
    for (const g of h.gpus) { keys(g, GPU, `gpu ${h.id}`); for (const p of g.procs) keys(p, PROC); }
    if (h.node) keys(h.node, ['version', 'hostname', ...NODE.filter((k) => h.node[k] !== undefined && !['version', 'hostname'].includes(k))].filter((k, i, a) => a.indexOf(k) === i), `node ${h.id}`);
    if (h.policy) keys(h.policy, POLICY); if (h.docker) keys(h.docker, DOCKER); if (h.memory) keys(h.memory, MEM);
  }
  const by = Object.fromEntries(b.items.map((h) => [h.id, h]));
  assert.equal(by.local.remote, false); assert.equal(by.local.link.state, 'up'); assert.equal(by.local.gpus.length, 2); assert.equal(by.local.node.arch, 'x86_64'); assert.equal(by.local.policy.mode, 'manage');
  assert.equal(by['gpu-box-2'].gpus.length, 1); assert.ok(by['gpu-box-2'].gpus[0].total_gib < 40 && by['gpu-box-2'].gpus[0].total_gib > 30, 'a 32 GB card');
  for (const id of ['edge-1', 'edge-2', 'edge-3', 'edge-4']) { assert.equal(by[id].remote, true); assert.ok(by[id], id); }
  for (const id of ['edge-1', 'edge-2', 'edge-3']) { assert.equal(by[id].node.arch, 'aarch64'); assert.equal(by[id].gpus.length, 1); assert.equal(by[id].gpus[0].unified_memory, true); assert.equal(by[id].gpus[0].used_gib, null, 'the driver cannot say'); assert.ok(by[id].gpus[0].total_gib > 100 && by[id].gpus[0].total_gib < 130, '128 GB shared'); }
  assert.equal(by['edge-2'].policy.mode, 'monitor');
  assert.deepEqual(new Set(b.items.map((h) => h.link.state)), new Set(['up', 'stale', 'down', 'incompatible', 'disabled']), 'every non-transient link state appears once');
  assert.equal(by['edge-3'].link.state, 'stale'); assert.ok(by['edge-3'].link.last_seen < Date.now() / 1000 - 5, 'a stale host has an old last_seen');
  assert.equal(by['edge-4'].link.state, 'down'); assert.match(by['edge-4'].link.error, /key is not authorized|Login refused/); assert.ok(by['edge-4'].link.retry_in_s > 0, 'a down host counts down to the next try'); assert.deepEqual(by['edge-4'].gpus, []);
  assert.equal(by['edge-5'].link.state, 'incompatible'); assert.match(by['edge-5'].link.error, /protocol/); assert.equal(by.spare1 ?? by['spare-1'].link.state, 'disabled');
  const unrestricted = b.items.filter((h) => h.node?.forced_cmd === false); assert.ok(unrestricted.length >= 1, 'one machine was reached with an unrestricted key');
  assert.ok(unrestricted.every((h) => h.link.notes.some((n) => /UNRESTRICTED/.test(n))), 'LinkStatus.notes says so');
  const raw = JSON.stringify(b); for (const bad of [/\/home\//, /@gmail|@example\.com/, /192\.168\./, /\.ssh/]) assert.doesNotMatch(raw, bad, 'publish-safe mock data');
});

test('GpuProc rows point at instances that exist, and an engine\'s VRAM per GPU matches its processes', async () => {
  const hosts = (await get('/hosts')).b.items, inst = (await get('/instances')).b.items;
  for (const h of hosts) for (const g of h.gpus) for (const p of g.procs) {
    if (p.instance_id) { const i = inst.find((x) => x.id === p.instance_id); assert.ok(i, `${p.instance_id} on ${h.id}`); assert.equal(i.host_id, h.id); if (!g.unified_memory) assert.ok(i.vram.some((v) => v.gpu_index === g.index && v.gpu_uuid === g.uuid && v.used_gib === p.used_gib), 'vram[] and the GPU\'s process agree'); }
  }
  const local = hosts.find((h) => h.id === 'local');
  assert.ok(local.gpus.some((g) => g.procs.filter((p) => p.instance_id).length >= 2) && local.gpus.some((g) => g.procs.some((p) => p.instance_id === null)), 'a card shared by several engines plus a foreign process (the "other" segment)');
  for (const g of local.gpus) assert.ok(Math.abs(g.used_gib - g.procs.reduce((a, p) => a + (p.used_gib || 0), 0)) < 1, 'used is the sum of the processes');
});

test('GET /hosts/{id}: HostDetail with disks, roots (writable, shared_from, visible_on), images, scripts and addresses', async () => {
  const { s, b } = await get('/hosts/gpu-box-2'); assert.equal(s, 200); keys(b, DETAIL);
  for (const d of b.disks) keys(d, MOUNT); for (const r of b.roots) { keys(r, ROOT); if (r.mount) keys(r.mount, MOUNT); assert.ok(r.visible_on.length >= 1 && r.visible_on[0] === b.id, 'visible_on lists this host first'); }
  assert.deepEqual(b.addresses, ['10.0.0.5/24 (eth0)']); assert.ok(b.images.every((i) => ['ref', 'id', 'size_bytes', 'created', 'arch'].every((k) => k in i)));
  const share = b.roots.find((r) => r.id === 'gguf-shared'); assert.deepEqual(share.visible_on, ['gpu-box-2', 'edge-1', 'edge-2']); assert.equal(share.writable, true);
  const e1 = (await get('/hosts/edge-1')).b; const nfs = e1.roots.find((r) => r.id === 'nfs-shared'); assert.equal(nfs.shared_from, 'gpu-box-2'); assert.equal(nfs.writable, false); assert.equal(nfs.mount.remote, true); assert.equal(nfs.mount.remote_host_id, 'gpu-box-2'); assert.equal(nfs.mount.label, 'NFS from gpu-box-2');
  const local = (await get('/hosts/local')).b; assert.ok(local.disks.some((d) => d.label === 'tank (zfs)' && d.fstype === 'zfs')); assert.ok(local.disks.every((d) => d.used_bytes + d.free_bytes === d.size_bytes));
  assert.ok(local.disks.some((d) => d.free_bytes / d.size_bytes < 0.05), 'a nearly full drive for the low-space warning');
  assert.ok(local.scripts.length >= 2 && local.scripts.every((x) => (keys(x, SCRIPT), true)));
  const down = (await get('/hosts/edge-4')).b; keys(down, DETAIL); assert.deepEqual([down.disks, down.roots, down.images, down.scripts], [[], [], [], []]);
  isProblem(await get('/hosts/nope'), 404, 'unknown_host');
});

test('drive labels follow the hub\'s naming: "<pool> (zfs)", "NFS from <host>", "<mount> (<fstype>, <drive model>)"', async () => {
  const all = (await Promise.all(['local', 'gpu-box-2', 'edge-1'].map(async (id) => (await get(`/hosts/${id}`)).b.disks))).flat();
  for (const d of all) {
    if (d.fstype === 'zfs') assert.match(d.label, /^[a-z0-9-]+ \(zfs\)$/); else if (d.remote) assert.match(d.label, /^(NFS|CIFS) from \S+$/); else if (d.model) assert.equal(d.label, `${d.mount} (${d.fstype}, ${d.model})`); else assert.equal(d.label, `${d.mount} (${d.fstype})`);
  }
});

test('POST /hosts/{id}/reconnect: needs the CSRF header, returns a HostSummary, moves a host to connecting, refuses a disabled one', async () => {
  const noCsrf = await j('/api/v1/hosts/edge-4/reconnect', { method: 'POST', headers: { 'content-type': 'application/json' }, body: '{}' }); isProblem(noCsrf, 403, 'csrf_header_required');
  const r = await post('/hosts/edge-4/reconnect'); assert.equal(r.s, 200); keys(r.b, SUMMARY); assert.equal(r.b.link.state, 'connecting');
  isProblem(await post('/hosts/nope/reconnect'), 404, 'unknown_host'); isProblem(await post('/hosts/spare-1/reconnect'), 409, 'conflict');
  await new Promise((ok) => setTimeout(ok, 2100)); assert.equal((await get('/hosts/edge-4')).b.link.state, 'down', 'the host with no authorised key goes back to down with its explanation');
});

test('GET /fleet/stream?once=true: SSE "host" events carrying a HostSummary, one per machine; the live stream is capped like the others', async () => {
  const r = await fetch(`${base}/api/v1/fleet/stream?once=true`); assert.equal(r.status, 200); assert.match(r.headers.get('content-type'), /text\/event-stream/);
  const text = await r.text(); const events = [...text.matchAll(/event: host\ndata: (.+)\n\n/g)].map((m) => JSON.parse(m[1]));
  const hosts = (await get('/hosts')).b.items; assert.deepEqual(events.map((e) => e.id).sort(), hosts.map((h) => h.id).sort()); for (const e of events) keys(e, SUMMARY);
});

test('GET /hardware?host=: the v1 report plus host_id, per machine; 404 for an unknown host, 503 for one that is down', async () => {
  const l = await get('/hardware'); assert.equal(l.s, 200); assert.equal(l.b.host_id, 'local'); assert.equal(l.b.gpus.length, 2); keys(l.b.memory, MEM);
  for (const id of ['gpu-box-2', 'edge-1']) { const x = (await get(`/hardware?host=${id}`)).b; assert.equal(x.host_id, id); assert.ok(x.gpus.every((g) => ['index', 'uuid', 'name', 'total_gib', 'free_gib', 'used_gib', 'util_pct', 'temp_c', 'power_w', 'fan_pct', 'compute_capability'].every((k) => k in g))); assert.ok(x.gpus.every((g) => typeof g.used_gib === 'number' && typeof g.free_gib === 'number'), 'GpuStat fields are plain numbers even on unified memory'); }
  isProblem(await get('/hardware?host=nope'), 404, 'unknown_host'); isProblem(await get('/hardware?host=edge-5'), 503, 'node_unavailable');
});

test('GET /instances: the v2 Instance shape on every machine, all three management classes, per-GPU vram, host_memory, model_location, controls with reasons', async () => {
  const { b } = await get('/instances'); assert.ok(b.items.length >= 12);
  for (const i of b.items) {
    keys(i, INSTANCE, i.id); keys(i.controls, ['start', 'stop', 'restart', 'remove', 'logs', 'configure', 'reason']); assert.ok(['managed', 'script', 'external'].includes(i.management)); assert.equal(typeof i.host_label, 'string');
    for (const v of i.vram) keys(v, ['gpu_index', 'gpu_uuid', 'used_gib']);
    if (i.model) keys(i.model, ['format', 'repo_id', 'root_id', 'rel', 'name']); if (i.model_location) keys(i.model_location, ['host_id', 'root_id', 'root_label', 'path', 'drive', 'fstype', 'remote', 'shared_from']);
    if (i.management === 'managed') assert.match(i.id, /^inst_/); if (i.management === 'external') assert.match(i.id, /^ext-[a-z0-9-]+$/); if (i.management === 'script') { assert.match(i.id, /^scr-[a-z0-9-]+$/); assert.ok(i.script); }
    if (!Object.values(i.controls).slice(0, 6).some(Boolean) || i.management !== 'managed') assert.ok(i.controls.reason || i.management === 'external', `${i.id}: a control that is off or special says why`);
  }
  const hosts = new Set(b.items.map((i) => i.host_id)); for (const h of ['local', 'gpu-box-2', 'edge-1', 'edge-2', 'edge-3']) assert.ok(hosts.has(h), h);
  const cls = (m) => b.items.filter((i) => i.management === m);
  assert.ok(cls('managed').length >= 4 && cls('script').length >= 2 && cls('external').length >= 5);
  const withStop = cls('external').filter((i) => i.controls.stop), without = cls('external').filter((i) => !i.controls.stop);
  assert.ok(withStop.length >= 1 && without.length >= 1, 'external engines with and without stop/start'); assert.ok(withStop.every((i) => i.controls.start && !i.controls.restart && i.controls.configure)); assert.ok(without.every((i) => /outside the console|monitor mode|not report|stale|down/.test(i.controls.reason)), 'the reason explains');
  const script = cls('script').find((i) => i.state === 'ready'); assert.deepEqual([script.controls.start, script.controls.stop, script.controls.restart, script.controls.remove], [true, true, true, false]); assert.match(script.controls.reason, new RegExp(script.script));
  assert.ok(script.vram.length === 2 && script.vram.every((v) => v.used_gib > 0), 'a tensor-parallel engine holds memory on both GPUs');
  const stoppedScripts = cls('script').filter((i) => i.state === 'stopped'); assert.ok(stoppedScripts.length >= 6, 'many stopped script engines on the workstation');
  const empty = cls('script').find((i) => !i.controls.start && !i.controls.stop && !i.controls.restart); assert.ok(empty && /no up or down action/.test(empty.controls.reason), 'a script engine whose controls are empty');
  const sc = (await get('/scripts')).b.items; assert.ok(sc.some((x) => x.name === empty.script && x.actions.length && !x.actions.some((a) => ['up', 'down'].includes(a.name))), 'and the script still has actions to offer');
  assert.ok(cls('script').concat(cls('external')).every((i) => i.name === i.name.toLowerCase()), 'container-style names');
  assert.ok(cls('script').find((i) => i.state === 'ready').served_models.length === 1, 'a running script engine serves a model (the UI shows it as the title)');
  const unified = b.items.find((i) => i.id === 'inst_edgechat03'); assert.ok(unified.vram.every((v) => v.used_gib === null), 'unified memory cannot be attributed per GPU'); assert.equal(unified.host_memory.source, 'cgroup'); assert.ok(unified.host_memory.total_gib > 10);
  const shared = b.items.find((i) => i.id === 'inst_edgechat03').model_location; assert.equal(shared.remote, true); assert.equal(shared.shared_from, 'gpu-box-2'); assert.equal(shared.drive, 'NFS from gpu-box-2');
  const stale = b.items.find((i) => i.host_id === 'edge-3'); assert.equal(Object.values({ ...stale.controls, reason: null, logs: false, configure: false }).some(Boolean), false); assert.match(stale.controls.reason, /not reported|answers again/);
  const mon = b.items.find((i) => i.id === 'ext-edge--2-sglang'); assert.equal(mon.controls.stop, false); assert.match(mon.controls.reason, /monitor mode/);
  assert.deepEqual((await get('/instances?host=gpu-box-2')).b.items.map((i) => i.host_id), ['gpu-box-2', 'gpu-box-2', 'gpu-box-2']); isProblem(await get('/instances?host=nope'), 404, 'unknown_host');
});

test('instance actions follow the controls: external stop only where the policy allows, script via its action, monitor and stale hosts refuse', async () => {
  isProblem(await post('/instances/ext-local-vllm/stop'), 409, 'instance_not_managed');
  isProblem(await post('/instances/ext-gpu--box--2-ollama/restart'), 409, 'instance_not_managed');
  assert.equal((await post('/instances/ext-gpu--box--2-ollama/stop')).s, 200); assert.equal((await get('/instances/ext-gpu--box--2-ollama')).b.state, 'stopped'); assert.equal((await post('/instances/ext-gpu--box--2-ollama/start')).s, 200);
  isProblem(await post('/instances/ext-edge--2-sglang/stop'), 409, 'monitor_only'); isProblem(await post('/instances/inst_edgebatch05/stop'), 503, 'node_unavailable');
  assert.equal((await post('/instances/scr-local-small-model/start')).s, 200); isProblem(await j('/api/v1/instances/scr-local-big-model', { method: 'DELETE', headers: H }), 409, 'instance_not_managed');
  isProblem(await get('/instances/scr-local-big-model/logs'), 409, 'instance_not_managed');
  assert.equal((await get('/instances/inst_qwenagent01/command')).s, 200);
});

test('POST /instances and /instances/preflight take host_id, model and image; the checks use the hub\'s own codes and block with the reason', async () => {
  const ok = await post('/instances/preflight', { engine: 'vllm', repo_id: 'Qwen/Qwen3.6-27B-FP8', host_id: 'local', gpu_ids: [1], params: {} });
  assert.equal(ok.s, 200); keys(ok.b, ['ok', 'gpu_ids', 'checks', 'fit']); assert.equal(ok.b.ok, true); for (const c of ok.b.checks) keys(c, ['code', 'level', 'message']); assert.ok(ok.b.checks.some((c) => c.code === 'fit') && ok.b.checks.some((c) => c.code === 'port'));
  const mon = (await post('/instances/preflight', { engine: 'vllm', repo_id: 'Qwen/Qwen3.6-27B-FP8', host_id: 'edge-2', gpu_ids: [0] })).b; assert.equal(mon.ok, false); assert.ok(mon.checks.some((c) => c.code === 'monitor_only' && c.level === 'block'));
  const down = (await post('/instances/preflight', { engine: 'vllm', repo_id: 'x/y', host_id: 'edge-4', gpu_ids: [0] })).b; assert.deepEqual(down.checks.map((c) => c.code), ['host_down']);
  const noimg = (await post('/instances/preflight', { engine: 'sglang', repo_id: 'Qwen/Qwen3.6-27B-FP8', host_id: 'gpu-box-2', gpu_ids: [0] })).b; assert.ok(noimg.checks.some((c) => c.code === 'image_missing' && c.level === 'block'));
  const disallowed = (await post('/instances/preflight', { engine: 'vllm', repo_id: 'Qwen/Qwen3.6-27B-FP8', host_id: 'local', gpu_ids: [0], image: 'evil/miner:1' })).b; assert.ok(disallowed.checks.some((c) => c.code === 'image_not_allowed'));
  const nomodel = (await post('/instances/preflight', { engine: 'llamacpp', model: { format: 'gguf', root_id: 'gguf-models', rel: 'nope.gguf' }, host_id: 'local', gpu_ids: [0] })).b; assert.ok(nomodel.checks.some((c) => c.code === 'model_missing'));
  const fmt = (await post('/instances/preflight', { engine: 'llamacpp', repo_id: 'Qwen/Qwen3.6-27B-FP8', host_id: 'local', gpu_ids: [0] })).b; assert.ok(fmt.checks.some((c) => c.code === 'format_unsupported'));
  const badgpu = (await post('/instances/preflight', { engine: 'vllm', repo_id: 'Qwen/Qwen3.6-27B-FP8', host_id: 'local', gpu_ids: [7] })).b; assert.ok(badgpu.checks.some((c) => c.code === 'gpu_unknown'));
  const bad = await post('/instances', { engine: 'vllm', repo_id: 'Qwen/Qwen3.6-35B-A3B-FP8', host_id: 'gpu-box-2', gpu_ids: [0] }); isProblem(bad, 409, 'preflight_failed'); assert.ok(bad.b.preflight.checks.length && bad.b.preflight.ok === false, 'the failure carries the checks');
  isProblem(await post('/instances', { engine: 'vllm', host_id: 'nope', repo_id: 'a/b' }), 404, 'unknown_host'); isProblem(await post('/instances', { repo_id: 'a/b' }), 422, 'validation_error');
  const made = await post('/instances', { engine: 'llamacpp', model: { format: 'gguf', root_id: 'nfs-shared', rel: 'Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf' }, host_id: 'edge-1', gpu_ids: [0], params: { ctx_size: 4096 }, name: 'contract-test' });
  assert.equal(made.s, 201); keys(made.b, INSTANCE); assert.equal(made.b.host_id, 'edge-1'); assert.equal(made.b.management, 'managed'); assert.equal(made.b.model.format, 'gguf'); assert.equal(made.b.model_location.shared_from, 'gpu-box-2'); assert.match(made.b.image, /llama\.cpp/); assert.ok(made.b.port >= 18000);
  assert.equal((await j(`/api/v1/instances/${made.b.id}`, { method: 'DELETE', headers: H })).s, 204);
});

test('POST /fit takes host_id and model: estimates against that machine\'s GPUs, handles unified memory, and 404s an unknown host', async () => {
  const local = (await post('/fit', { engine: 'vllm', repo_id: 'Qwen/Qwen3.6-35B-A3B-FP8', host_id: 'local', gpu_ids: [0], params: {} })).b; assert.equal(local.verdict, 'fits'); assert.equal(local.per_gpu[0].vram_total_gib, 95.6);
  const small = (await post('/fit', { engine: 'vllm', repo_id: 'Qwen/Qwen3.6-35B-A3B-FP8', host_id: 'gpu-box-2', gpu_ids: [0], params: {} })).b; assert.equal(small.verdict, 'wont_fit', 'a 36 GB model on a 32 GB card'); assert.ok(small.per_gpu[0].vram_total_gib < 40);
  const uni = (await post('/fit', { engine: 'vllm', repo_id: 'Qwen/Qwen3.6-35B-A3B-FP8', host_id: 'edge-1', gpu_ids: [0], params: {} })).b; assert.ok(['fits', 'tight'].includes(uni.verdict)); assert.match(uni.host_ram.notes.join(' '), /unified memory/i); assert.ok(uni.host_ram.needed_gib > 36, 'on unified memory the model counts against RAM too');
  const gguf = (await post('/fit', { engine: 'llamacpp', host_id: 'local', model: { format: 'gguf', root_id: 'gguf-models', rel: 'Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf' }, gpu_ids: [0], params: {} })).b; assert.equal(gguf.verdict, 'fits'); assert.ok(gguf.per_gpu[0].weights_gib > 15 && gguf.per_gpu[0].weights_gib < 18);
  assert.equal((await post('/fit', { engine: 'vllm', host_id: 'nope', repo_id: 'a/b' })).s, 404); isProblem(await post('/fit', { engine: 'vllm' }), 422, 'validation_error');
});

test('GET /library: LibraryModel[] with copies on several machines, shared storage, one in use, every format', async () => {
  const { s, b } = await get('/library'); assert.equal(s, 200); assert.ok('items' in b && b.next_cursor === null);
  for (const m of b.items) { keys(m, LIBMODEL, m.key); assert.match(m.key, new RegExp(`^${m.format}:`)); for (const c of m.copies) { keys(c, COPY); assert.match(c.id, /^[0-9a-f]{16}$/); assert.ok(c.visible_on.includes(c.host_id)); } }
  assert.deepEqual(new Set(b.items.map((m) => m.format)), new Set(['hf', 'gguf', 'ollama', 'plain', 'other']));
  const multi = b.items.find((m) => new Set(m.copies.map((c) => c.host_id)).size >= 3); assert.ok(multi, 'a model with copies on several machines');
  const shared = b.items.find((m) => m.copies.some((c) => c.visible_on.length >= 3)); assert.ok(shared, 'a copy on shared storage visible on several machines'); assert.equal(shared.format, 'gguf');
  assert.ok(b.items.some((m) => m.copies.some((c) => c.in_use_by.length)), 'a copy that is in use'); assert.ok(b.items.some((m) => m.warnings.length), 'a model with a warning');
  assert.deepEqual(b.items.find((m) => m.format === 'hf').engines, ['vllm', 'sglang', 'trtllm']); assert.deepEqual(b.items.find((m) => m.format === 'gguf').engines, ['llamacpp']); assert.deepEqual(b.items.find((m) => m.format === 'ollama').engines, ['ollama']); assert.deepEqual(b.items.find((m) => m.format === 'other').engines, []);
  const inUse = b.items.flatMap((m) => m.copies).find((c) => c.in_use_by.length); assert.equal(inUse.deletable, false);
  assert.ok(b.items.flatMap((m) => m.copies).filter((c) => c.writable && !c.in_use_by.length).every((c) => c.deletable));
  assert.ok((await get('/library?host=edge-1')).b.items.every((m) => m.copies.some((c) => c.host_id === 'edge-1' || c.visible_on.includes('edge-1'))));
  assert.ok((await get('/library?format=gguf')).b.items.every((m) => m.format === 'gguf')); const q = (await get('/library?q=gemma')).b.items; assert.ok(q.length >= 1 && q.every((m) => /gemma/i.test(m.name)));
  isProblem(await get('/library?host=nope'), 404, 'unknown_host');
});

test('v1 GET /models stays: Hugging Face models of every machine, each with host_id', async () => {
  const { b } = await get('/models'); assert.ok(b.items.length >= 5 && b.items.every((m) => ['repo_id', 'size_bytes', 'revisions', 'last_used', 'path', 'engines_that_fit', 'host_id'].every((k) => k in m)));
  assert.ok(new Set(b.items.map((m) => m.host_id)).size >= 3);
});

test('POST /library/rescan -> 202 {hosts}; copy and delete enforce the node rules with the contract\'s statuses', async () => {
  const all = await post('/library/rescan'); assert.equal(all.s, 202); assert.ok(all.b.hosts.includes('local') && !all.b.hosts.includes('edge-4'), 'only machines that are up are rescanned');
  assert.deepEqual((await post('/library/rescan', { host_id: 'local' })).b, { hosts: ['local'] }); isProblem(await post('/library/rescan', { host_id: 'nope' }), 404, 'unknown_host');
  const lib = (await get('/library')).b.items; const q4 = lib.find((m) => m.key === 'gguf:unsloth/Qwen3.6-27B-GGUF:Q4_K_M'); const inUse = q4.copies.find((c) => c.in_use_by.length);
  isProblem(await j(`/api/v1/library/copies/${inUse.id}`, { method: 'DELETE', headers: H }), 409, 'model_in_use'); isProblem(await j('/api/v1/library/copies/0000000000000000', { method: 'DELETE', headers: H }), 404, 'copy_not_found');
  const noCsrf = await j(`/api/v1/library/copies/${q4.copies[0].id}`, { method: 'DELETE' }); isProblem(noCsrf, 403, 'csrf_header_required');
  const ro = lib.find((m) => m.format === 'ollama').copies.find((c) => !c.writable); isProblem(await j(`/api/v1/library/copies/${ro.id}`, { method: 'DELETE', headers: H }), 403, 'policy_denied');
  const hf = lib.find((m) => m.key === 'hf:Qwen/Qwen3.6-35B-A3B-FP8').copies.find((c) => c.host_id === 'local');
  isProblem(await post(`/library/copies/${hf.id}/copy`, { host_id: 'edge-2' }), 409, 'monitor_only'); isProblem(await post(`/library/copies/${hf.id}/copy`, { host_id: 'edge-4' }), 503, 'node_unavailable'); isProblem(await post(`/library/copies/${hf.id}/copy`, { host_id: 'edge-3' }), 503, 'node_unavailable');
  isProblem(await post(`/library/copies/${hf.id}/copy`, { host_id: 'local' }), 409, 'conflict'); isProblem(await post(`/library/copies/${hf.id}/copy`, { host_id: 'nope' }), 404, 'unknown_host'); isProblem(await post(`/library/copies/${hf.id}/copy`, {}), 422, 'validation_error');
  isProblem(await post(`/library/copies/${hf.id}/copy`, { host_id: 'edge-1', root_id: 'nfs-shared' }), 403, 'policy_denied'); isProblem(await post('/library/copies/0000000000000000/copy', { host_id: 'edge-1' }), 404, 'copy_not_found');
  isProblem(await post(`/library/copies/${hf.id}/copy`, { host_id: 'gpu-box-2', root_id: 'hf-main' }), 409, 'conflict'); // gpu-box-2 already holds a copy of this model
  const copyOk = await post(`/library/copies/${lib.find((m) => m.key === 'hf:Qwen/Qwen3.6-27B-FP8').copies.find((c) => c.host_id === 'local').id}/copy`, { host_id: 'gpu-box-2' });
  assert.equal(copyOk.s, 201); keys(copyOk.b, DOWNLOAD); assert.equal(copyOk.b.kind, 'copy'); assert.equal(copyOk.b.host_id, 'gpu-box-2'); assert.ok(copyOk.b.source_copy_id && copyOk.b.root_id === 'hf-main');
  const del = lib.find((m) => m.key === 'other:scratch').copies[0]; assert.equal((await j(`/api/v1/library/copies/${del.id}`, { method: 'DELETE', headers: H })).s, 204); assert.ok(!(await get('/library')).b.items.some((m) => m.key === 'other:scratch'), 'the last copy removes the model');
});

test('downloads gain host_id and root_id and kind copy; targets are checked (machine, folder, space) and GGUF quants download by file pattern', async () => {
  const { b } = await get('/downloads'); for (const d of b.items) keys(d, DOWNLOAD);
  assert.ok(b.items.some((d) => d.host_id !== 'local'), 'downloads targeting remote machines'); const copy = b.items.find((d) => d.kind === 'copy' && d.host_id === 'edge-1'); assert.ok(copy && copy.source_copy_id && copy.root_id === 'hf-main', 'a copy job');
  assert.ok(b.items.some((d) => d.state === 'running' && d.host_id === 'edge-1')); assert.ok(b.items.some((d) => d.error_code === 'gated_model'));
  const ok = await post('/downloads', { repo_id: 'Qwen/Qwen3-Embedding-8B', host_id: 'gpu-box-2', root_id: 'hf-main' }); assert.equal(ok.s, 201); assert.equal(ok.b.host_id, 'gpu-box-2'); assert.equal(ok.b.root_id, 'hf-main'); assert.equal(ok.b.kind, 'hf');
  const dflt = await post('/downloads', { repo_id: 'Qwen/Qwen3-Embedding-8B' }); assert.equal(dflt.b.host_id, 'local'); assert.equal(dflt.b.root_id, 'hf-main', 'the default folder is the first writable HF root');
  isProblem(await post('/downloads', { repo_id: 'a/b', host_id: 'edge-2' }), 409, 'monitor_only'); isProblem(await post('/downloads', { repo_id: 'a/b', host_id: 'edge-4' }), 503, 'node_unavailable'); isProblem(await post('/downloads', { repo_id: 'a/b', host_id: 'nope' }), 404, 'unknown_host');
  isProblem(await post('/downloads', { repo_id: 'a/b', host_id: 'local', root_id: 'ollama-store' }), 403, 'policy_denied'); isProblem(await post('/downloads', { repo_id: 'a/b', host_id: 'local', root_id: 'zzz' }), 404, 'not_found');
  isProblem(await post('/downloads', { repo_id: 'Qwen/Qwen3.6-27B-FP8', host_id: 'local', root_id: 'usb-hf' }), 507, 'no_space'); // the nearly full USB drive: 28 GiB wanted, about 10 free
  const gg = await post('/downloads', { repo_id: 'unsloth/Qwen3.6-27B-GGUF', host_id: 'gpu-box-2', allow_patterns: ['Qwen3.6-27B-Q5_K_M.gguf'] }); assert.equal(gg.s, 201); assert.equal(gg.b.root_id, 'gguf-shared', 'a GGUF repo goes to a GGUF folder'); assert.deepEqual(gg.b.allow_patterns, ['Qwen3.6-27B-Q5_K_M.gguf']); assert.ok(Math.abs(gg.b.total_bytes - 19.2 * 2 ** 30) < 1e6, 'the size of that quantization');
});

test('GET /hf/models/{repo} gains gguf_groups: [{quant, files, total_bytes}] for GGUF repos and an empty list otherwise', async () => {
  const g = (await get('/hf/models/unsloth/Qwen3.6-27B-GGUF')).b; assert.ok(g.gguf_groups.length >= 4);
  for (const x of g.gguf_groups) { keys(x, ['quant', 'files', 'total_bytes']); assert.ok(x.files.length >= 1 && x.files.every((f) => f.endsWith('.gguf')) && x.total_bytes > 2 ** 30); }
  assert.ok(g.gguf_groups.some((x) => x.quant === 'Q4_K_M') && g.gguf_groups.some((x) => x.files.length > 1), 'big quantizations are split into shards');
  assert.deepEqual((await get('/hf/models/Qwen/Qwen3.6-27B-FP8')).b.gguf_groups, []);
});

test('GET /scripts: ScriptInfo[] with actions (args, confirm, danger); POST runs one as a 202 ScriptJob and streams lines then end', async () => {
  const { b } = await get('/scripts'); assert.ok(b.items.length >= 3); for (const x of b.items) { keys(x, SCRIPT); for (const a of x.actions) keys(a, ['name', 'label', 'args', 'confirm', 'danger']); assert.ok(['up', 'down', 'unknown', 'error'].includes(x.status)); }
  const big = b.items.find((x) => x.name === 'big-model'); assert.ok(big.actions.some((a) => a.confirm) && big.actions.some((a) => a.danger) && big.actions.some((a) => a.args.length === 2)); assert.equal(big.status, 'up'); assert.ok(big.containers.length);
  isProblem(await post('/scripts/local/big-model/switch', { arg: 'nope' }), 400, 'invalid_arg'); isProblem(await post('/scripts/local/big-model/switch', {}), 400, 'arg_required'); isProblem(await post('/scripts/local/big-model/status', { arg: 'x' }), 400, 'invalid_arg');
  isProblem(await post('/scripts/local/nope/up'), 404, 'unknown_script'); isProblem(await post('/scripts/local/big-model/nope'), 404, 'unknown_action'); isProblem(await post('/scripts/nope/big-model/up'), 404, 'unknown_host');
  isProblem(await post('/scripts/edge-2/anything/up'), 404, 'unknown_script'); isProblem(await post('/scripts/edge-4/x/up'), 404, 'unknown_script');
  const run = await post('/scripts/local/small-model/status'); assert.equal(run.s, 202); keys(run.b, JOB); assert.equal(run.b.state, 'running'); assert.equal(run.b.rc, null); assert.equal(run.b.action, 'status');
  isProblem(await post('/scripts/local/small-model/up'), 409, 'script_busy');
  const stream = await fetch(`${base}/api/v1/scripts/jobs/${run.b.id}/stream`); const text = await stream.text();
  const lines = [...text.matchAll(/event: line\ndata: (.*)\n\n/g)].map((m) => m[1]); assert.ok(lines.length >= 4 && /small-model status/.test(lines[0]));
  const end = text.match(/event: end\ndata: (.+)\n\n$/); assert.ok(end, 'ends with an "end" event'); const job = JSON.parse(end[1]); keys(job, JOB); assert.equal(job.state, 'completed'); assert.equal(job.rc, 0); assert.ok(job.finished_at >= job.started_at);
  const again = (await get(`/scripts/jobs/${run.b.id}`)).b; keys(again, JOB); assert.equal(again.state, 'completed'); assert.ok(again.output_tail.includes('done'));
  isProblem(await get('/scripts/jobs/nope'), 404, 'unknown_job'); const noJob = await fetch(`${base}/api/v1/scripts/jobs/nope/stream`); assert.equal(noJob.status, 404);
  const up = await post('/scripts/local/small-model/up'); assert.equal(up.s, 202); await new Promise((ok) => setTimeout(ok, 4500)); assert.equal((await get('/instances/scr-local-small-model')).b.state, 'ready', 'the script\'s start action brings its engine up');
  assert.equal((await get('/scripts')).b.items.find((x) => x.name === 'small-model').status, 'up');
});

test('GET /engines gains model_formats, images_by_arch, multi_model and covers vLLM, SGLang, llama.cpp, Ollama and TensorRT-LLM', async () => {
  const { b } = await get('/engines'); assert.ok(Array.isArray(b));
  for (const e of b) { keys(e, ['id', 'display_name', 'default_image', 'presets', 'presets_info', 'model_formats', 'images_by_arch', 'multi_model']); assert.equal(typeof e.presets_info, 'object'); assert.ok(e.model_formats.length && e.model_formats.every((f) => ['hf', 'gguf', 'ollama', 'plain'].includes(f))); assert.doesNotMatch(e.default_image, /:latest$/, 'pinned, never latest'); assert.equal(typeof e.multi_model, 'boolean'); }
  const by = Object.fromEntries(b.map((e) => [e.id, e])); assert.deepEqual(by.llamacpp.model_formats, ['gguf']); assert.deepEqual(by.ollama.model_formats, ['ollama']); assert.equal(by.ollama.multi_model, true); assert.ok(by.vllm.model_formats.includes('hf')); assert.ok(by.vllm.images_by_arch.aarch64);
});

test('GET /engines/{id}/params carries level, unit and explain for every parameter of the vLLM and SGLang catalogs', async () => {
  for (const e of ['vllm', 'sglang']) { const { b } = await get(`/engines/${e}/params`); for (const s of b) { assert.ok(['basic', 'advanced', 'expert'].includes(s.level), `${e}.${s.key} level`); assert.ok('unit' in s); assert.ok(s.explain?.summary, `${e}.${s.key} explain`); keys(s.explain, ['summary', 'details', 'when_to_change', 'risks', 'example', 'terms', 'level']); } assert.ok(b.some((s) => s.level === 'basic') && b.some((s) => s.level === 'expert')); }
  for (const e of ['llamacpp', 'ollama', 'trtllm']) { const { b } = await get(`/engines/${e}/params`); assert.ok(b.length >= 5 && b.every((s) => s.key && s.flag && s.level)); assert.ok(b.some((s) => s.explain), e); }
  isProblem(await get('/engines/nope/params'), 404, 'not_found');
});

test('GET /learn: a LearnBundle with 30+ glossary terms, explains for metrics, concepts, params and presets, and every key the UI uses', async () => {
  const { s, b } = await get('/learn'); assert.equal(s, 200); keys(b, ['version', 'glossary', 'metrics', 'concepts', 'params', 'presets']); assert.equal(b.version, 1);
  assert.ok(Object.keys(b.glossary).length >= 30); for (const t of Object.values(b.glossary)) keys(t, ['title', 'short', 'long', 'see_also']);
  for (const [f, ids] of Object.entries(CONCEPT_FAMILIES)) for (const id of ids) assert.ok(b.concepts[`${f}.${id}`]?.summary, `${f}.${id}`);
  for (const k of METRIC_KEYS) assert.ok(b.metrics[k]?.summary, `metric ${k}`);
  const checks = Object.keys(b.concepts).filter((k) => k.startsWith('check.')); const pf = (await post('/instances/preflight', { engine: 'vllm', repo_id: 'Qwen/Qwen3.6-35B-A3B-FP8', host_id: 'gpu-box-2', gpu_ids: [0] })).b;
  for (const c of pf.checks) assert.ok(checks.includes(`check.${c.code}`), `the check ${c.code} the mock hub returns has an explanation`);
  for (const e of ['vllm', 'sglang', 'llamacpp', 'ollama', 'trtllm']) assert.ok(b.params[e] && b.presets[e], e);
});

test('GET/PUT /settings gain host_image_pins (host -> engine -> image), checked against that machine\'s policy; hf_cache_dir stays read-only', async () => {
  assert.deepEqual((await get('/settings')).b.host_image_pins, {});
  const ok = await j('/api/v1/settings', { method: 'PUT', headers: H, body: JSON.stringify({ host_image_pins: { 'edge-1': { vllm: 'vllm/vllm-openai:v0.27.1-aarch64' }, local: { sglang: 'lmsysorg/sglang:v0.6.1' } } }) });
  assert.equal(ok.s, 200); assert.equal(ok.b.host_image_pins['edge-1'].vllm, 'vllm/vllm-openai:v0.27.1-aarch64'); assert.deepEqual((await get('/settings')).b.host_image_pins, ok.b.host_image_pins, 'persisted');
  const bad = await j('/api/v1/settings', { method: 'PUT', headers: H, body: JSON.stringify({ host_image_pins: { 'edge-1': { vllm: 'evil/image:1' } } }) }); isProblem(bad, 422, 'image_not_allowed');
  isProblem(await j('/api/v1/settings', { method: 'PUT', headers: H, body: JSON.stringify({ host_image_pins: { nope: { vllm: 'x' } } }) }), 422, 'unknown_host'); isProblem(await j('/api/v1/settings', { method: 'PUT', headers: H, body: JSON.stringify({ host_image_pins: 'x' }) }), 422, 'validation_error');
  const pre = await post('/instances/preflight', { engine: 'vllm', repo_id: 'Qwen/Qwen3.6-27B-FP8', host_id: 'edge-1', gpu_ids: [0] }); assert.ok(pre.b.checks.every((c) => c.code !== 'image_not_allowed'), 'the pin is used for the launch image');
  const before = (await get('/settings')).b.hf_cache_dir; assert.equal((await j('/api/v1/settings', { method: 'PUT', headers: H, body: JSON.stringify({ hf_cache_dir: '/tmp/x' }) })).b.hf_cache_dir, before);
});

test('GET /metrics/system?host= follows the machine; a down one is 503', async () => {
  const a = (await get('/metrics/system?window=5m&host=gpu-box-2')).b; assert.equal(a.host_id, 'gpu-box-2'); assert.ok(a.points[0].values.ram_used_gib < 70, 'the 62 GiB machine, not the workstation');
  const l = (await get('/metrics/system?window=5m')).b; assert.ok(l.points[0].values.ram_used_gib > 150); isProblem(await get('/metrics/system?host=edge-4'), 503, 'node_unavailable'); isProblem(await get('/metrics/system?host=nope'), 404, 'unknown_host');
});
