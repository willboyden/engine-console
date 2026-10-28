#!/usr/bin/env node
// Full-stack check (Node 22, zero deps): the PYTHON hub driving its local node, once with the Python ecnode and once
// with the Rust ecnode (both through NODE_COMMAND), in the same hermetic environment (env.mjs, monitor policy, fake docker /
// nvidia-smi / ip). Compares GET /api/v1/hosts and /api/v1/hosts/local, normalised. Diffs must be listed in
// expected-diffs.json "hub_node". Only diff PATHS are printed (values could hold this machine's facts).
//
//   node backend/parity/proto/hub-node.mjs [--no-build] [--keep]
import { spawn } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, chmodSync } from 'node:fs';
import { createServer } from 'node:net';
import { join } from 'node:path';
import { makeEnv, FRONTEND, WORKSPACE } from './env.mjs';
import { pythonRelease } from '../lib/python-release.mjs';
import { buildRust, diff, globMatch, loadExpected, normalise, parseTyped, privacyLeaks } from './matrix.mjs';

// The Python hub and node: the last Python release (lib/python-release.mjs; EC_PYTHON_REF overrides it).
const PYREL = pythonRelease();
const RUST_NODE = join(WORKSPACE, 'target', 'debug', 'ecnode');
// the reference node's venv has no nvidia-ml-py (the hub's does): the Python node then uses the fake nvidia-smi, like the matrix
const PY_NODE = PYREL.nodePy;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const freePort = () => new Promise((resolve, reject) => {
  const s = createServer(); s.once('error', reject);
  s.listen(0, '127.0.0.1', () => { const { port } = s.address(); s.close(() => resolve(port)); });
});

/** Volatile link/time fields of the hub's API (in addition to matrix.mjs normalise). */
const LINK_KEYS = new Set(['last_seen', 'connected_since', 'rtt_ms', 'skew_s', 'retry_in_s', 'updated_at', 'checked_at']);
function normaliseHub(v) {
  if (Array.isArray(v)) return v.map(normaliseHub);
  if (v === null || typeof v !== 'object' || '$f' in v) return v;
  const out = {};
  for (const [k, x] of Object.entries(v)) {
    if (LINK_KEYS.has(k) && x !== null) out[k] = '<volatile>';
    else if (k === 'memory' && x && typeof x === 'object') out[k] = Object.fromEntries(Object.keys(x).map((m) => [m, x[m] === null ? null : '<num>']));
    else if (k === 'cpu' && x && typeof x === 'object' && 'load1' in x) out[k] = Object.fromEntries(Object.keys(x).map((m) => [m, x[m] === null ? null : '<num>']));
    else if (k === 'alerts') out[k] = '<alerts: from this machine\'s RAM pressure>';
    else if (k === 'node' && x && typeof x === 'object') {
      out[k] = Object.fromEntries(Object.entries(x).map(([nk, nv]) => [nk,
        ['hostname', 'os', 'kernel', 'cpu_model', 'cpu_cores', 'cpu_threads', 'python'].includes(nk) && nv !== null ? `<${nk}>` : nv]));
    } else if (k === 'label' && 'link' in v) out[k] = '<hostname>';
    else if (k === 'disks' && Array.isArray(x)) out[k] = x.map((m) => m.roots).filter((r) => r && r.length);
    else if (k === 'mount' && x && typeof x === 'object' && 'mount' in x) out[k] = { roots: x.roots };
    else out[k] = normaliseHub(x);
  }
  return out;
}

async function startHub(e, nodeCommand) {
  const port = await freePort();
  const data = join(e.dir, nodeCommand.includes(RUST_NODE) ? 'data-rust' : 'data-py');
  mkdirSync(data, { recursive: true }); chmodSync(data, 0o700);
  const env = {
    ...e.env, ENGINE_CONSOLE_PORT: String(port), ENGINE_CONSOLE_DATA_DIR: data, ENGINE_CONSOLE_TRUST_LOOPBACK: 'false',
    ENGINE_CONSOLE_ENV_FILE: join(e.dir, 'no-env-file'), OTLP_ENABLED: 'false', HF_ENDPOINT: 'https://127.0.0.1:9',
    HOSTS_FILE: join(e.dir, 'no-hosts.json'), NODE_POLICY: e.policy,
    NODE_COMMAND: JSON.stringify(nodeCommand), FRONTEND_DIR: FRONTEND,
  };
  const proc = spawn(PYREL.hubPy, ['-I', '-m', 'engine_console.main'], { cwd: PYREL.backend, env, stdio: ['ignore', 'pipe', 'pipe'] });
  let log = ''; proc.stdout.on('data', (d) => { log += d; }); proc.stderr.on('data', (d) => { log += d; });
  const base = `http://127.0.0.1:${port}`;
  for (let i = 0; i < 150; i++) {
    if (proc.exitCode !== null) throw new Error(`hub exited early (${proc.exitCode})`);
    try { if ((await fetch(`${base}/api/v1/health`)).ok) break; } catch { /* not up yet */ }
    await sleep(200);
  }
  const keyFile = join(data, 'bootstrap-admin.key');
  for (let i = 0; i < 50 && !existsSync(keyFile); i++) await sleep(100);
  const key = readFileSync(keyFile, 'utf8').trim();
  const get = async (p) => {
    const r = await fetch(`${base}${p}`, { headers: { Authorization: `Bearer ${key}` } });
    return { status: r.status, body: parseTyped(await r.text()) };
  };
  return { proc, get, log: () => log };
}

async function snapshot(e, nodeCommand) {
  const hub = await startHub(e, nodeCommand);
  try {
    let up = false;
    for (let i = 0; i < 100 && !up; i++) {
      const h = await hub.get('/api/v1/hosts/local');
      up = h.status === 200 && h.body.link?.state === 'up';
      if (!up) await sleep(200);
    }
    if (!up) throw new Error('the local node never came up');
    await sleep(4500);                      // two sample periods and the first inventory
    return { hosts: await hub.get('/api/v1/hosts'), local: await hub.get('/api/v1/hosts/local') };
  } finally {
    hub.proc.kill('SIGTERM');
    for (let i = 0; i < 50 && hub.proc.exitCode === null; i++) await sleep(100);
    if (hub.proc.exitCode === null) hub.proc.kill('SIGKILL');
  }
}

async function main(argv) {
  if (!argv.includes('--no-build')) buildRust();
  const e = makeEnv('monitor');
  let failed = false;
  try {
    const py = await snapshot(e, [PY_NODE, '-I', '-m', 'ecnode', '--stdio', '--policy', e.policy]);
    // NVML off for the Rust node (§A2) and its tools from the fakes (ECNODE_TOOL_PATH: it never searches PATH); the hub
    // passes its node no other variables, so they ride in the command
    const rs = await snapshot(e, ['/usr/bin/env', 'ECNODE_NO_NVML=1', `ECNODE_TOOL_PATH=${e.env.ECNODE_TOOL_PATH}`,
      RUST_NODE, '--stdio', '--policy', e.policy]);
    const expected = loadExpected().hub_node;
    const used = new Set();
    for (const what of ['hosts', 'local']) {
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
    console.log(`python node link: ${py.local.body.link?.state}; rust node link: ${rs.local.body.link?.state}; `
      + `rust node policy mode: ${rs.local.body.policy?.mode}`);
    for (const [i, x] of expected.entries()) if (!used.has(i)) console.log(`note: expected diff never seen: ${x.path}`);
  } finally {
    if (!argv.includes('--keep')) e.cleanup();
  }
  console.log(failed ? 'hub-node: FAILED' : 'hub-node: OK (the Python hub runs the Rust node; diffs all expected)');
  return failed ? 1 : 0;
}

process.exit(await main(process.argv.slice(2)));
