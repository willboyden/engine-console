// The differential harness: start backend A and backend B with identical hermetic setups, replay every corpus step
// on both, normalise (lib/normalise.mjs) and compare. Used by run.mjs (report) and routes-coverage.mjs (coverage).
import { mkdtempSync, readFileSync, readdirSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { makeShared, PARITY, startBackend, stopBackend } from './backend.mjs';
import { startFakeHf } from './fakehf.mjs';
import { send } from './http.mjs';
import { diffValues, JsonError, Num, parseJson, show } from './json.mjs';
import { Normaliser, parseSse } from './normalise.mjs';
import { RouteTable } from './routes.mjs';

export const CORPUS_DIR = join(PARITY, 'corpus');
// Headers compared between the backends. etag and last-modified: presence only (their values derive from file
// mtimes and the hashing scheme; the 304 round trip is compared through the status of the follow-up request).
export const COMPARED_HEADERS = [
  'content-type', 'cache-control', 'content-security-policy', 'x-content-type-options', 'referrer-policy',
  'x-frame-options', 'permissions-policy', 'content-disposition', 'x-accel-buffering', 'allow', 'location',
];
export const PRESENCE_HEADERS = ['etag', 'last-modified'];
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export function loadCorpus(dir = CORPUS_DIR) {
  return readdirSync(dir).filter((f) => f.endsWith('.json')).sort().map((f) => {
    const doc = JSON.parse(readFileSync(join(dir, f), 'utf8'));
    return { file: f, modes: doc.modes || ['nofleet'], scenarios: doc.scenarios || [] };
  });
}

/** Replace {{var}} in strings (deeply) with this backend's captured values. */
function subst(v, vars) {
  if (typeof v === 'string') {
    return v.replace(/\{\{([A-Za-z0-9_]+)\}\}/g, (m, k) => (k in vars ? String(vars[k]) : m));
  }
  if (Array.isArray(v)) return v.map((x) => subst(x, vars));
  if (v && typeof v === 'object') return Object.fromEntries(Object.entries(v).map(([k, x]) => [k, subst(x, vars)]));
  return v;
}

function buildQuery(q) {
  if (!q) return '';
  const p = new URLSearchParams();
  for (const [k, v] of Array.isArray(q) ? q : Object.entries(q)) p.append(k, String(v));
  const s = p.toString();
  return s ? `?${s}` : '';
}

/** Read a captured value from the raw (not normalised) answer: "header:<name>" or a dotted path in the JSON body. */
function pick(resp, spec) {
  if (spec.startsWith('header:')) return resp.headers[spec.slice(7).toLowerCase()];
  let v = resp.json;
  for (const part of spec.split('.').filter(Boolean)) {
    if (v === undefined || v === null) return undefined;
    v = Array.isArray(v) ? v[Number(part)] : v[part];
  }
  return v instanceof Num ? v.raw : v;
}

function bodyKind(resp, method, path) {
  if (method === 'HEAD') return 'none';
  const ct = (resp.headers['content-type'] || '').toLowerCase();
  if (ct.includes('text/event-stream')) return 'sse';
  if (ct.includes('json')) return 'json';
  if (path.split('?')[0] === '/metrics' && ct.startsWith('text/plain')) return 'prom';
  return 'text';
}

/** Execute one step on one backend; returns the raw answer plus its normalised comparable form. */
async function execStep(b, step, vars, norm, route) {
  const s = subst(step, vars);
  const headers = { ...(s.headers || {}) };
  const auth = s.auth || 'admin';
  if (auth === 'admin') headers.Authorization = `Bearer ${b.key}`;
  else if (auth === 'viewer') headers.Authorization = `Bearer ${vars.viewer_key}`;
  else if (auth === 'bad') headers.Authorization = 'Bearer ec_not_a_real_key_0000000000000000000000000000000';
  else if (auth === 'basic') headers.Authorization = 'Basic dXNlcjpwYXNz';
  if (s.csrf) headers['X-Engine-Console'] = '1';
  let body = null; let chunks = null;
  if ('json' in s) {
    body = Buffer.from(JSON.stringify(s.json));
    if (!Object.keys(headers).some((k) => k.toLowerCase() === 'content-type')) headers['Content-Type'] = 'application/json';
  } else if ('raw' in s) {
    body = Buffer.from(s.raw);
  } else if (s.raw_repeat) {
    body = Buffer.alloc(s.raw_repeat.count, s.raw_repeat.char || 'x');
    if (s.raw_repeat.prefix) body = Buffer.concat([Buffer.from(s.raw_repeat.prefix), body, Buffer.from(s.raw_repeat.suffix || '')]);
  } else if (s.chunked) {
    chunks = [];
    for (let i = 0; i < s.chunked.parts; i++) chunks.push(Buffer.alloc(s.chunked.size, s.chunked.char || 'x'));
  }
  const path = s.path + buildQuery(s.query);
  const resp = await send({ port: b.port, method: s.method, path, headers, body, chunks, timeoutMs: s.timeout_ms || 15000 });
  const kind = s.compare || bodyKind(resp, s.method, s.path);
  const text = resp.body.toString('utf8');
  resp.text = text;
  let comparable;
  if (kind === 'json') {
    try {
      resp.json = parseJson(text);
      comparable = norm.json(resp.json, route);
    } catch (e) {
      if (!(e instanceof JsonError)) throw e;
      comparable = { invalid_json: norm.text(text.slice(0, 500)) };
    }
  } else if (kind === 'sse') {
    comparable = parseSse(text).map((ev) => {
      let data;
      try { data = norm.json(parseJson(ev.data), route); } catch { data = norm.text(ev.data); }
      return { event: ev.event, data };
    });
  } else if (kind === 'prom') {
    comparable = norm.prometheus(text);
  } else if (kind === 'none') {
    comparable = resp.body.length ? `<${resp.body.length} bytes>` : '';
  } else if (kind === 'length') {
    comparable = `<${resp.body.length} bytes>`;
  } else {
    comparable = norm.text(text);
  }
  const hdrs = {};
  for (const h of COMPARED_HEADERS) if (h in resp.headers) hdrs[h] = resp.headers[h];
  for (const h of PRESENCE_HEADERS) if (h in resp.headers) hdrs[h] = '<present>';
  if (s.capture) for (const [k, spec] of Object.entries(s.capture)) {
    const v = pick(resp, spec);
    if (v !== undefined) vars[k] = v;
  }
  return { status: resp.status, error: resp.error, headers: hdrs, body: comparable, raw: resp };
}

function compare(a, b) {
  const diffs = [];
  if (a.status !== b.status) diffs.push({ path: 'status', a: a.status, b: b.status });
  if (a.error || b.error) diffs.push({ path: 'transport', a: a.error || '', b: b.error || '' });
  diffValues(a.headers, b.headers, 'headers', diffs, 40);
  diffValues(a.body, b.body, 'body', diffs, 40);
  return diffs;
}

/** The named, listed differences the harness accepts (allowed-diffs.json): name -> {status, headers, steps}. */
export function loadAllowances(file = join(PARITY, 'allowed-diffs.json')) {
  const doc = JSON.parse(readFileSync(file, 'utf8'));
  return Object.entries(doc).filter(([k]) => !k.startsWith('_')).map(([name, a]) => ({ name, ...a }));
}

/** The allowance that covers every one of `diffs` for this step, or null. An allowance accepts only its headers,
 * only absent on A and present on B, only when both answered its status: status and body still have to be equal. */
export function allowedBy(allowances, file, scenario, index, ra, rb, diffs) {
  for (const al of allowances) {
    if (!al.steps.some((s) => s.file === file && s.scenario === scenario && s.index === index)) continue;
    if (ra.status !== al.status || rb.status !== al.status) continue;
    const covered = (d) => d.path.startsWith('headers.') && al.headers.includes(d.path.slice(8)) && d.a === '<absent>';
    if (diffs.length && diffs.every(covered)) return al.name;
  }
  return null;
}

/** The Rust skeleton's answer for a handler that is not ported yet: 501 with detail "not ported: <module>". */
export function unportedModule(r) {
  if (r.status !== 501 || !r.raw.json || typeof r.raw.json !== 'object') return null;
  const d = r.raw.json.detail;
  return typeof d === 'string' && d.startsWith('not ported') ? (d.split(':')[1] || '').trim() || '?' : null;
}

/** B answers as if the route did not exist although A serves it (404 "no such file" from the SPA fallback, or 405). */
export function notRegistered(a, b) {
  const aServed = a.status !== 404 && a.status !== 405;
  const bDetail = b.raw.json && typeof b.raw.json === 'object' ? b.raw.json.detail : null;
  return aServed && ((b.status === 404 && bDetail === 'no such file') || b.status === 405);
}

/** Wait (in lock step, so both backends serve the same number of requests) until both report a connected local
 * node with its first sample and a scanned library, or `ms` passes. Returns per-backend readiness. */
async function waitFleet(backends, ms) {
  const t0 = Date.now();
  const ready = backends.map(() => false);
  while (Date.now() - t0 < ms && ready.some((r) => !r)) {
    for (let i = 0; i < backends.length; i++) {
      const b = backends[i];
      const h = await send({ port: b.port, method: 'GET', path: '/api/v1/hosts/local', headers: { Authorization: `Bearer ${b.key}` } });
      const l = await send({ port: b.port, method: 'GET', path: '/api/v1/library', headers: { Authorization: `Bearer ${b.key}` } });
      try {
        const host = JSON.parse(h.body.toString());
        const lib = JSON.parse(l.body.toString());
        ready[i] = host.link?.state === 'up' && (host.gpus || []).length > 0 && host.memory != null
          && (host.disks || []).length > 0 && (lib.items || []).length >= 2;
      } catch { ready[i] = false; }
    }
    if (ready.some((r) => !r)) await sleep(500);
  }
  return ready;
}

/**
 * Run the corpus on two backends. opts: {a, b, modes, node, filter, keep, log}
 * Returns {steps: [...], modes: {...}, errors: [...]}.
 */
export async function runHarness(opts) {
  const routes = RouteTable.load();
  const corpus = loadCorpus();
  const log = opts.log || (() => {});
  const results = { a: opts.a, b: opts.b, node: opts.node, steps: [], errors: [], ready: {} };
  for (const mode of opts.modes) {
    const files = corpus.filter((f) => f.modes.includes(mode) && (!opts.filter || f.file.includes(opts.filter)));
    if (!files.length) continue;
    const tmp = mkdtempSync(join(tmpdir(), 'ec-parity-'));
    const shared = makeShared(tmp);
    // `hf`: the fleet setup plus the fake Hugging Face hub behind an egress proxy (lib/fakehf.mjs)
    const fleet = mode === 'fleet' || mode === 'hf';
    let A; let B; let hf = null;
    try {
      if (mode === 'hf') hf = await startFakeHf(tmp);
      [A, B] = await Promise.all([
        startBackend(opts.a, tmp, { shared, fleet, node: opts.node, hf }),
        startBackend(opts.b, tmp, { shared, fleet, node: opts.node, hf }),
      ]);
      const backends = [A, B];
      const norms = backends.map((x) => new Normaliser([x.root, tmp], x.version));
      const vars = backends.map((x) => ({ port: x.port, admin_key: x.key }));
      const allowances = opts.a === opts.b ? [] : loadAllowances();
      if (fleet) {
        const ready = await waitFleet(backends, opts.fleetWaitMs || 45000);
        results.ready[mode] = { a: ready[0], b: ready[1] };
        log(`[${mode}] fleet ready: ${opts.a}=${ready[0]} ${opts.b}=${ready[1]}`);
      }
      for (const f of files) {
        for (const sc of f.scenarios) {
          for (const [i, step] of (sc.steps || []).entries()) {
            if (step.sleep_ms) { await sleep(step.sleep_ms); continue; }
            const route = step.route || routes.routeFor(step.method, step.path).key;
            const ra = await execStep(A, step, vars[0], norms[0], route);
            const rb = await execStep(B, step, vars[1], norms[1], route);
            const diffs = compare(ra, rb);
            const mod = diffs.length ? unportedModule(rb) : null;
            let verdict = 'same';
            if (diffs.length) verdict = mod ? 'unported' : (notRegistered(ra, rb) ? 'not-registered' : 'diff');
            const allowed = verdict === 'diff' ? allowedBy(allowances, f.file, sc.name, i, ra, rb, diffs) : null;
            if (allowed) verdict = 'allowed';
            const expectFail = step.expect && step.expect.status !== undefined && ra.status !== step.expect.status
              ? `expected ${step.expect.status} from ${opts.a}, got ${ra.status}` : null;
            results.steps.push({
              mode, file: f.file, scenario: sc.name, index: i, name: step.name || '', method: step.method,
              path: step.path, route, verdict, module: mod, allowed, diffs, expectFail,
              a: { status: ra.status, body: brief(ra.body) }, b: { status: rb.status, body: brief(rb.body) },
            });
          }
        }
      }
    } catch (e) {
      results.errors.push(`[${mode}] ${e.message}`);
    } finally {
      await Promise.all([stopBackend(A), stopBackend(B)]);
      if (hf) {
        results.hf = { connects: [...new Set(hf.connects)].sort(), requests: hf.requests.length };
        await hf.stop();
      }
      if (!opts.keep) rmSync(tmp, { recursive: true, force: true });
      else log(`[${mode}] kept ${tmp}`);
    }
  }
  return results;
}

function brief(v) {
  const s = typeof v === 'string' ? v : show(v);
  return s.length > 200 ? `${s.slice(0, 200)}...` : s;
}
