#!/usr/bin/env node
// ecnode protocol cross-matrix (Node 22, zero deps).
//
//   node backend/parity/proto/matrix.mjs            run {py, rust} driver x {py, rust} node over every scenario
//   node backend/parity/proto/matrix.mjs --record   re-record golden/*.json with the Python driver and Python node
//   options: --only <scenario>  --no-build  --keep (keep temp dirs)  --show-expected
//
// Every run gets a fresh hermetic environment (env.mjs). Transcripts are normalised (NORMALISE.md) and compared
// with golden/<scenario>.json as JSON values; numbers keep their int/float kind (JSON.parse source access), so a
// node that sends 5.0 where Python sends 5 is a diff. Rules:
//   py-driver x py-node and rust-driver x py-node   must equal the golden transcript exactly;
//   * x rust-node                                    may differ only where expected-diffs.json says (with a reason).
// Exit 0 when every rule holds; 1 otherwise. Every transcript is also scanned for this machine's identity.
import { spawn, spawnSync } from 'node:child_process';
import { readFileSync, writeFileSync, readdirSync, existsSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { hostname, userInfo, networkInterfaces, homedir } from 'node:os';
import { join } from 'node:path';
import { makeEnv, HERE, WORKSPACE } from './env.mjs';
import { pythonRelease } from '../lib/python-release.mjs';

// The Python side is the last Python release (lib/python-release.mjs; EC_PYTHON_REF overrides it). Its node venv runs
// both the reference node and drive.py (which imports the reference `ecnode.proto`).
const PY = () => pythonRelease().nodePy;
const RUST_DRIVE = join(WORKSPACE, 'target', 'debug', 'ecproto-drive');
const RUST_NODE = join(WORKSPACE, 'target', 'debug', 'ecnode');
const DRIVE_PY = join(HERE, 'drive.py');

export const DRIVERS = {
  py: (node, scen) => [PY(), DRIVE_PY, '--node', JSON.stringify(node), '--scenario', scen],
  rust: (node, scen) => [RUST_DRIVE, '--node', JSON.stringify(node), '--scenario', scen],
};
export const NODES = {
  py: (policy) => [PY(), '-I', '-m', 'ecnode', '--stdio', '--policy', policy],
  rust: (policy) => [RUST_NODE, '--stdio', '--policy', policy],
};

// ── typed JSON: floats are tagged so int/float kinds survive the comparison ─────────────────────────────
export function parseTyped(text) {
  return JSON.parse(text, (k, v, ctx) => (typeof v === 'number' && /[.eE]/.test(ctx?.source ?? '') ? { $f: v } : v));
}
const isFloat = (v) => v !== null && typeof v === 'object' && !Array.isArray(v) && Object.keys(v).length === 1 && '$f' in v;
const isNum = (v) => typeof v === 'number' || isFloat(v);

// ── normalisation (every rule is listed in NORMALISE.md) ────────────────────────────────────────────────
const TIME_KEYS = new Set(['t', 'mtime', 'atime', 'scanned_at', 'duration_s', 'boot_time']);
const COUNTER_KEYS = new Set(['generation', 'seq']);

export function normalise(v) {
  if (Array.isArray(v)) return v.map(normalise);
  if (v === null || typeof v !== 'object' || isFloat(v)) return v;
  const out = {};
  const helloLike = 'proto' in v && 'node_id' in v && 'host' in v;
  const sampleLike = 'mem' in v && 'cpu' in v && 'gpus' in v && 'containers' in v;
  for (const [k, x] of Object.entries(v)) {
    if (TIME_KEYS.has(k) && isNum(x)) out[k] = '<num>';
    else if (COUNTER_KEYS.has(k) && isNum(x)) out[k] = '<int>';
    else if (k === 'credit_from_node' && typeof x === 'number') out[k] = x > 0 ? '<granted>' : 0;
    else if (helloLike && k === 'node_id' && typeof x === 'string') out[k] = '<node_id>';
    else if (helloLike && k === 'host' && x && typeof x === 'object') {
      out[k] = Object.fromEntries(Object.entries(x).map(([hk, hv]) => [hk, hv === null ? null : `<host.${hk}>`]));
    } else if (sampleLike && k === 'mem' && x && typeof x === 'object') {
      out[k] = Object.fromEntries(Object.entries(x).map(([mk, mv]) => [mk, isNum(mv) ? '<int>' : mv]));
    } else if (sampleLike && k === 'cpu' && x && typeof x === 'object') {
      out[k] = Object.fromEntries(Object.entries(x).map(([ck, cv]) => [ck, isNum(cv) ? '<num>' : cv]));
    } else if (k === 'mounts' && Array.isArray(x) && x.every((m) => m && typeof m === 'object' && 'mount' in m)) {
      // this machine's filesystems: only which model roots share a filesystem is kept
      out[k] = x.map((m) => m.roots).filter((r) => Array.isArray(r) && r.length);
    } else out[k] = normalise(x);
  }
  return out;
}

// ── diff ─────────────────────────────────────────────────────────────────────────────────────────────────
export function diff(a, b, path = '', out = []) {
  if (isFloat(a) && isFloat(b)) { if (a.$f !== b.$f) out.push(path); return out; }
  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) out.push(`${path}/length`);
    for (let i = 0; i < Math.min(a.length, b.length); i++) diff(a[i], b[i], `${path}/${i}`, out);
    return out;
  }
  const ao = a && typeof a === 'object' && !Array.isArray(a) && !isFloat(a);
  const bo = b && typeof b === 'object' && !Array.isArray(b) && !isFloat(b);
  if (ao && bo) {
    for (const k of new Set([...Object.keys(a), ...Object.keys(b)])) {
      if (!(k in a) || !(k in b)) out.push(`${path}/${k}${k in a ? ' (missing)' : ' (extra)'}`);
      else diff(a[k], b[k], `${path}/${k}`, out);
    }
    return out;
  }
  if (JSON.stringify(a) !== JSON.stringify(b)) out.push(path);
  return out;
}

/** Name step paths after the step (`steps/<i>:<op or do>`), so expected diffs survive renumbering. */
export function labelPaths(paths, scenario) {
  return paths.map((p) => p.replace(/^\/steps\/(\d+)/, (_, i) => {
    const s = scenario.steps[Number(i)] || {};
    const what = typeof s.op === 'string' ? s.op : s.do;
    return `/steps/${i}:${what}${s.label ? `:${s.label}` : ''}`;
  }));
}

/** glob over '/'-separated segments: `*` within a segment, `**` for any rest. A pattern also matches everything
 * below the path it names (a trailing `/**` changes nothing). */
export function globMatch(pattern, path) {
  const esc = (s) => s.replace(/[.+^${}()|[\]\\]/g, '\\$&');
  const re = pattern.replace(/\/\*\*$/, '').split('/').map((seg) => (seg === '**' ? '.*' : esc(seg).replace(/\*/g, '[^/]*'))).join('/');
  return new RegExp(`^/?${re}(/.*)?$`).test(path);
}

// ── privacy: no identity of this machine in a transcript or golden file ──────────────────────────────────
export function identityStrings() {
  const out = new Set([hostname(), userInfo().username, homedir()]);
  try {
    const mid = readFileSync('/etc/machine-id', 'utf8').trim();
    if (mid) { out.add(mid); out.add(createHash('sha256').update(mid).digest('hex').slice(0, 16)); }
  } catch { /* none */ }
  for (const addrs of Object.values(networkInterfaces())) {
    for (const a of addrs || []) if (!a.internal && !a.address.startsWith('fe80')) out.add(a.address);
  }
  return [...out].filter((s) => s && s.length >= 4);
}

export function privacyLeaks(text, ids = identityStrings()) {
  const leaks = ids.filter((s) => text.includes(s));
  const gpus = [...text.matchAll(/GPU-[0-9a-f-]{20,}/g)].map((m) => m[0])
    .filter((g) => !/^GPU-(a{8}|b{8})-1111-2222-3333-444444444444$/.test(g));
  return [...leaks, ...gpus];
}

// ── running ──────────────────────────────────────────────────────────────────────────────────────────────
function run(driver, node, scenarioPath, kind, keep) {
  return new Promise((resolve) => {
    const e = makeEnv(kind);
    const argv = DRIVERS[driver](NODES[node](e.policy), scenarioPath);
    const p = spawn(argv[0], argv.slice(1), { env: e.env, stdio: ['ignore', 'pipe', 'pipe'] });
    let out = ''; let err = '';
    p.stdout.on('data', (d) => { out += d; });
    p.stderr.on('data', (d) => { err += d; });
    const timer = setTimeout(() => p.kill('SIGKILL'), 240000);
    p.on('close', (code) => {
      clearTimeout(timer);
      if (!keep) e.cleanup();
      resolve({ code, text: out.split(e.dir).join('<tmp>'), err: err.split(e.dir).join('<tmp>') });
    });
  });
}

async function pool(tasks, n) {
  const results = []; let i = 0;
  await Promise.all(Array.from({ length: n }, async () => {
    while (i < tasks.length) { const j = i++; results[j] = await tasks[j](); }
  }));
  return results;
}

export function buildRust() {
  const env = { ...process.env, PATH: `${homedir()}/.cargo/bin:${process.env.PATH}`, CARGO_BUILD_JOBS: '8' };
  const r = spawnSync('cargo', ['build', '-p', 'ec-proto', '-p', 'ecnode', '--bins'], { cwd: WORKSPACE, env, stdio: 'inherit' });
  if (r.status !== 0) throw new Error('cargo build failed');
}

export function loadExpected() {
  return JSON.parse(readFileSync(join(HERE, 'expected-diffs.json'), 'utf8'));
}

async function main(argv) {
  const record = argv.includes('--record');
  const keep = argv.includes('--keep');
  const only = argv.includes('--only') ? argv[argv.indexOf('--only') + 1] : null;
  if (!argv.includes('--no-build')) buildRust();
  const scenarios = readdirSync(join(HERE, 'scenarios')).filter((f) => f.endsWith('.json')).sort()
    .map((f) => ({ path: join(HERE, 'scenarios', f), def: JSON.parse(readFileSync(join(HERE, 'scenarios', f), 'utf8')) }))
    .filter((s) => !only || s.def.name === only);
  const ids = identityStrings();
  let failed = false;
  if (record) {
    const res = await pool(scenarios.map((s) => () => run('py', 'py', s.path, s.def.policy, keep)), 3);
    for (const [i, s] of scenarios.entries()) {
      const r = res[i];
      if (r.code !== 0) { console.error(`record ${s.def.name}: driver exited ${r.code}\n${r.err.slice(-2000)}`); failed = true; continue; }
      const norm = normalise(parseTyped(r.text));
      const text = JSON.stringify(norm, null, 1) + '\n';
      const leaks = privacyLeaks(text, ids);
      if (leaks.length) { console.error(`record ${s.def.name}: REFUSED, it contains this machine's identity (${leaks.length} strings)`); failed = true; continue; }
      writeFileSync(join(HERE, 'golden', `${s.def.name}.json`), text);
      console.log(`recorded golden/${s.def.name}.json`);
    }
    return failed ? 1 : 0;
  }
  const expected = loadExpected().proto;
  const combos = [['py', 'py'], ['rust', 'py'], ['py', 'rust'], ['rust', 'rust']];
  const tasks = [];
  for (const s of scenarios) for (const [d, n] of combos) tasks.push({ s, d, n });
  const results = await pool(tasks.map((t) => () => run(t.d, t.n, t.s.path, t.s.def.policy, keep)), 4);
  const rows = [];
  const used = new Set();
  for (const [i, t] of tasks.entries()) {
    const r = results[i];
    const name = t.s.def.name;
    const goldenPath = join(HERE, 'golden', `${name}.json`);
    let status; const problems = [];
    if (r.code !== 0) { status = `DRIVER EXIT ${r.code}`; problems.push(r.err.slice(-1500)); }
    else if (!existsSync(goldenPath)) { status = 'NO GOLDEN (run --record)'; }
    else {
      const got = normalise(parseTyped(r.text));
      const want = JSON.parse(readFileSync(goldenPath, 'utf8'));   // golden already holds the float tags
      const paths = labelPaths(diff(want, got), t.s.def);
      const leaks = privacyLeaks(JSON.stringify(got), ids);
      if (leaks.length) problems.push(`transcript holds this machine's identity (${leaks.length} strings)`);
      const unexpected = [];
      let excused = 0;
      for (const p of paths) {
        const hit = t.n === 'rust' && expected.findIndex((x) => x.node === 'rust' && (x.scenario === name || x.scenario === '*') && globMatch(x.path, p));
        if (hit !== false && hit >= 0) { excused++; used.add(hit); } else unexpected.push(p);
      }
      problems.push(...unexpected.map((p) => `unexpected diff ${p}`));
      status = unexpected.length || leaks.length ? `${unexpected.length} UNEXPECTED` : (excused ? `${excused} expected` : 'equal');
      if (argv.includes('--show-expected') && excused) problems.push(...paths.filter((p) => !unexpected.includes(p)).map((p) => `expected ${p}`));
    }
    if (status.includes('UNEXPECTED') || status.startsWith('DRIVER') || status.startsWith('NO GOLDEN')) failed = true;
    rows.push({ scenario: name, driver: t.d, node: t.n, status, problems });
  }
  const w = Math.max(...rows.map((r) => r.scenario.length));
  console.log(`${'scenario'.padEnd(w)}  driver  node  result`);
  for (const r of rows) {
    console.log(`${r.scenario.padEnd(w)}  ${r.driver.padEnd(6)}  ${r.node.padEnd(4)}  ${r.status}`);
    for (const p of r.problems.slice(0, argv.includes('--all') ? Infinity : 40)) console.log(`${' '.repeat(w)}    ${p}`);
  }
  const stale = expected.map((x, i) => [x, i]).filter(([x, i]) => !used.has(i) && (!only || x.scenario === only || x.scenario === '*'));
  for (const [x] of stale) console.log(`note: expected diff never seen (remove it once ported?): ${x.scenario} ${x.path}`);
  console.log(failed ? 'matrix: FAILED' : 'matrix: OK (py x py and rust-driver x py-node equal golden; rust-node diffs all expected)');
  return failed ? 1 : 0;
}

if (import.meta.url === `file://${process.argv[1]}`) process.exit(await main(process.argv.slice(2)));
