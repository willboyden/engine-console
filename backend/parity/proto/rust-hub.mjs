#!/usr/bin/env node
// The reverse full stack (Node 22, zero deps): the RUST hub's NodeLink driving the deployed-compatible PYTHON node,
// compared with the Python hub driving the same Python node. Both hubs run in the same hermetic environment
// (env.mjs: temp HOME, monitor-mode policy, fake docker / nvidia-smi / ip, seeded model roots), with the node started
// through NODE_COMMAND (the reference's node venv: python -I -m ecnode --stdio --policy <policy>: stdlib only, so it reads the fake
// nvidia-smi). Compared, normalised (matrix.mjs normalise + hub-node.mjs rules): GET /api/v1/hosts, /hosts/local,
// /hardware?host=local and the /fleet/stream?once=true snapshot; then POST /hosts/local/reconnect must bring the link
// back up on both. Diffs must be listed in expected-diffs.json "rust_hub". Only diff PATHS are printed.
//
//   node backend/parity/proto/rust-hub.mjs [--no-build] [--keep] [--show-expected]
import { spawn, spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, chmodSync } from 'node:fs';
import { homedir } from 'node:os';
import { createServer } from 'node:net';
import { join } from 'node:path';
import { makeEnv, FRONTEND, WORKSPACE } from './env.mjs';
import { pythonRelease } from '../lib/python-release.mjs';
import { diff, globMatch, loadExpected, normalise, parseTyped, privacyLeaks } from './matrix.mjs';

// The Python hub and node: the last Python release (lib/python-release.mjs; EC_PYTHON_REF overrides it).
const PYREL = pythonRelease();
const PY_HUB = PYREL.hubPy;
const RUST_HUB = join(WORKSPACE, 'target', 'debug', 'engine-console');
const PY_NODE = PYREL.nodePy;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const freePort = () => new Promise((resolve, reject) => {
  const s = createServer(); s.once('error', reject);
  s.listen(0, '127.0.0.1', () => { const { port } = s.address(); s.close(() => resolve(port)); });
});

/** Volatile link/time fields of the hub's API (as hub-node.mjs). */
const LINK_KEYS = new Set(['last_seen', 'connected_since', 'rtt_ms', 'skew_s', 'retry_in_s', 'updated_at', 'checked_at']);
function normaliseHub(v) {
  if (Array.isArray(v)) return v.map(normaliseHub);
  if (v === null || typeof v !== 'object' || '$f' in v) return v;
  const out = {};
  for (const [k, x] of Object.entries(v)) {
    if (LINK_KEYS.has(k) && x !== null) out[k] = '<volatile>';
    else if (k === 'memory' && x && typeof x === 'object') out[k] = Object.fromEntries(Object.keys(x).map((m) => [m, x[m] === null ? null : '<num>']));
    else if (k === 'cpu' && x && typeof x === 'object' && 'load1' in x) out[k] = Object.fromEntries(Object.keys(x).map((m) => [m, x[m] === null ? null : '<num>']));
    else if (['host_ram_gib', 'host_ram_free_gib'].includes(k) && x !== null) out[k] = '<num>';
    else if (k === 'alerts') out[k] = '<alerts: from this machine\'s RAM pressure>';
    else if (k === 'node' && x && typeof x === 'object') {
      out[k] = Object.fromEntries(Object.entries(x).map(([nk, nv]) => [nk,
        ['hostname', 'os', 'kernel', 'cpu_model', 'cpu_cores', 'cpu_threads', 'python'].includes(nk) && nv !== null ? `<${nk}>` : nv]));
    } else if (k === 'label' && 'link' in v) out[k] = '<hostname>';
    else if (k === 'disks' && Array.isArray(x)) out[k] = x.map((m) => m.roots).filter((r) => r && r.length);
    else if (k === 'mount' && x && typeof x === 'object' && 'mount' in x) out[k] = { roots: x.roots };
    else if (k === 'addresses' && Array.isArray(x)) out[k] = `<${x.length} address(es)>`;
    else out[k] = normaliseHub(x);
  }
  return out;
}

function buildHub() {
  const env = { ...process.env, PATH: `${homedir()}/.cargo/bin:${process.env.PATH}`, CARGO_BUILD_JOBS: '6' };
  const r = spawnSync('cargo', ['build', '-p', 'ec-hub', '--bins'], { cwd: WORKSPACE, env, stdio: 'inherit' });
  if (r.status !== 0) throw new Error('cargo build failed');
}

/** SSE text -> [{event, data}] */
function sse(text) {
  return text.trim().split('\n\n').filter(Boolean).map((block) => {
    const lines = block.split('\n');
    const ev = lines.find((l) => l.startsWith('event: '))?.slice(7);
    const data = lines.filter((l) => l.startsWith('data: ')).map((l) => l.slice(6)).join('\n');
    return { event: ev, data: parseTyped(data) };
  });
}

async function startHub(e, kind, nodeCommand) {
  const port = await freePort();
  const data = join(e.dir, `data-${kind}`);
  mkdirSync(data, { recursive: true }); chmodSync(data, 0o700);
  const env = {
    ...e.env, ENGINE_CONSOLE_PORT: String(port), ENGINE_CONSOLE_DATA_DIR: data, ENGINE_CONSOLE_TRUST_LOOPBACK: 'false',
    ENGINE_CONSOLE_ENV_FILE: join(e.dir, 'no-env-file'), OTLP_ENABLED: 'false', HF_ENDPOINT: 'https://127.0.0.1:9',
    HOSTS_FILE: join(e.dir, 'no-hosts.json'), NODE_POLICY: e.policy, NODE_COMMAND: JSON.stringify(nodeCommand),
    HF_CACHE_DIR: join(e.dir, 'hf-cache'), HUB_KEY_FILE: join(e.dir, 'no-key'), FRONTEND_DIR: FRONTEND,
    // the Rust hub's local node gets the hub's fixed tool PATH unless this says otherwise (ec-hub tools.rs): the fakes
    ENGINE_CONSOLE_TOOL_PATH: e.env.PATH,
  };
  const argv = kind === 'rust' ? [RUST_HUB] : [PY_HUB, '-I', '-m', 'engine_console.main'];
  const proc = spawn(argv[0], argv.slice(1), { cwd: kind === 'rust' ? WORKSPACE : PYREL.backend, env, stdio: ['ignore', 'pipe', 'pipe'] });
  let log = ''; proc.stdout.on('data', (d) => { log += d; }); proc.stderr.on('data', (d) => { log += d; });
  const base = `http://127.0.0.1:${port}`;
  for (let i = 0; i < 150; i++) {
    if (proc.exitCode !== null) throw new Error(`${kind} hub exited early (${proc.exitCode})`);
    try { if ((await fetch(`${base}/api/v1/health`)).ok) break; } catch { /* not up yet */ }
    await sleep(200);
  }
  const keyFile = join(data, 'bootstrap-admin.key');
  for (let i = 0; i < 50 && !existsSync(keyFile); i++) await sleep(100);
  const key = readFileSync(keyFile, 'utf8').trim();
  const req = async (method, p) => {
    const r = await fetch(`${base}${p}`, { method, headers: { Authorization: `Bearer ${key}` } });
    const text = await r.text();
    return { status: r.status, body: (r.headers.get('content-type') || '').startsWith('text/event-stream') ? sse(text) : parseTyped(text) };
  };
  return { proc, req, log: () => log };
}

async function waitUp(hub, what) {
  for (let i = 0; i < 150; i++) {
    const h = await hub.req('GET', '/api/v1/hosts/local');
    if (h.status === 200 && h.body.link?.state === 'up') return;
    await sleep(200);
  }
  throw new Error(`the local node never came up (${what})`);
}

async function snapshot(e, kind) {
  const hub = await startHub(e, kind, [PY_NODE, '-I', '-m', 'ecnode', '--stdio', '--policy', e.policy]);
  try {
    await waitUp(hub, kind);
    await sleep(4500);                      // two sample periods and the first inventory
    const out = {
      hosts: await hub.req('GET', '/api/v1/hosts'),
      local: await hub.req('GET', '/api/v1/hosts/local'),
      hardware: await hub.req('GET', '/api/v1/hardware?host=local'),
      stream: await hub.req('GET', '/api/v1/fleet/stream?once=true'),
    };
    const rc = await hub.req('POST', '/api/v1/hosts/local/reconnect');
    out.reconnect = { status: rc.status, id: rc.body.id };
    await sleep(300);
    await waitUp(hub, `${kind} after reconnect`);
    out.after = { state: (await hub.req('GET', '/api/v1/hosts/local')).body.link?.state };
    return out;
  } catch (err) {
    process.stderr.write(`${kind} hub log (tail):\n${hub.log().slice(-2000).split(e.dir).join('<tmp>')}\n`);
    throw err;
  } finally {
    hub.proc.kill('SIGTERM');
    for (let i = 0; i < 50 && hub.proc.exitCode === null; i++) await sleep(100);
    if (hub.proc.exitCode === null) hub.proc.kill('SIGKILL');
  }
}

async function main(argv) {
  if (!argv.includes('--no-build')) buildHub();
  const e = makeEnv('monitor');
  let failed = false;
  try {
    const py = await snapshot(e, 'python');
    const rs = await snapshot(e, 'rust');
    const expected = loadExpected().rust_hub || [];
    const used = new Set();
    for (const what of ['hosts', 'local', 'hardware', 'stream', 'reconnect', 'after']) {
      const a = normaliseHub(normalise(py[what])); const b = normaliseHub(normalise(rs[what]));
      const paths = diff(a, b, `/${what}`);
      const unexpected = [];
      for (const p of paths) {
        const hit = expected.findIndex((x) => globMatch(x.path, p));
        if (hit >= 0) used.add(hit); else unexpected.push(p);
      }
      const leaks = privacyLeaks(JSON.stringify(a)).length + privacyLeaks(JSON.stringify(b)).length;
      console.log(`${what}: ${paths.length} diff(s), ${paths.length - unexpected.length} expected, ${unexpected.length} unexpected`
        + (leaks ? ` (normalised output still holds ${leaks} identity strings: not printed)` : ''));
      for (const p of unexpected) console.log(`  unexpected ${p}`);
      if (argv.includes('--show-expected')) for (const p of paths.filter((x) => !unexpected.includes(x))) console.log(`  expected ${p}`);
      if (unexpected.length) failed = true;
    }
    console.log(`python hub link: ${py.local.body.link?.state}; rust hub link: ${rs.local.body.link?.state}; `
      + `after reconnect: python ${py.after.state}, rust ${rs.after.state}`);
    for (const [i, x] of expected.entries()) if (!used.has(i)) console.log(`note: expected diff never seen: ${x.path}`);
  } finally {
    if (!argv.includes('--keep')) e.cleanup();
  }
  console.log(failed ? 'rust-hub: FAILED' : 'rust-hub: OK (the Rust hub runs the Python node; diffs all expected)');
  return failed ? 1 : 0;
}

process.exit(await main(process.argv.slice(2)));
