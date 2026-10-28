#!/usr/bin/env node
// Route coverage of the parity corpus, and route registration of the Rust port.
//
//   node backend/parity/routes-coverage.mjs                       static: every Python route has at least one corpus step
//   node backend/parity/routes-coverage.mjs --results <file>      also judge a run saved with `run.mjs --json-out <file>`
//   node backend/parity/routes-coverage.mjs --rust                run python vs rust now, then judge it
//
// Fails when a route of backend/parity/routes.python.json (plus FastAPI's /api/v1/openapi.json) has no corpus step, or
// when the Rust server answers as if a route Python serves did not exist (404 "no such file" from the SPA fallback,
// or 405). A Rust answer of 501 "not ported: <module>" is not a failure here: those routes are listed per module.
import { readFileSync } from 'node:fs';
import { loadCorpus, runHarness } from './lib/harness.mjs';
import { RouteTable } from './lib/routes.mjs';

function parseArgs(argv) {
  const o = { results: null, rust: false };
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--results') o.results = argv[++i];
    else if (argv[i] === '--rust') o.rust = true;
    else throw new Error(`unknown option ${argv[i]}`);
  }
  return o;
}

export function staticCoverage(table = RouteTable.load(), corpus = loadCorpus()) {
  const hits = new Map(table.keys().map((k) => [k, 0]));
  for (const f of corpus) {
    for (const sc of f.scenarios) {
      for (const st of sc.steps || []) {
        if (!st.method) continue;
        const key = st.route || table.routeFor(st.method, st.path.replace(/\{\{[A-Za-z0-9_]+\}\}/g, 'x')).key;
        if (hits.has(key)) hits.set(key, hits.get(key) + 1);
      }
    }
  }
  return hits;
}

export async function main(argv = process.argv.slice(2)) {
  let o;
  try { o = parseArgs(argv); } catch (e) { console.error(`routes-coverage: ${e.message}`); return 2; }
  const table = RouteTable.load();
  const hits = staticCoverage(table);
  const missing = [...hits].filter(([, n]) => n === 0).map(([k]) => k);
  console.log(`routes-coverage: ${hits.size} routes (routes.python.json + openapi), ${hits.size - missing.length} covered by the corpus`);
  for (const k of missing) console.log(`  NO CORPUS STEP: ${k}`);
  let res = null;
  if (o.results) res = JSON.parse(readFileSync(o.results, 'utf8'));
  else if (o.rust) res = await runHarness({ a: 'python', b: 'rust', modes: ['nofleet', 'fleet'], node: 'python', log: (m) => console.log(m) });
  let unregistered = [];
  if (res) {
    const per = new Map();
    for (const s of res.steps) {
      if (!per.has(s.route)) per.set(s.route, { verdicts: new Set(), modules: new Set() });
      per.get(s.route).verdicts.add(s.verdict);
      if (s.module) per.get(s.route).modules.add(s.module);
    }
    unregistered = [...per].filter(([k, v]) => hits.has(k) && v.verdicts.has('not-registered')).map(([k]) => k);
    const byModule = new Map();
    for (const [k, v] of per) for (const m of v.modules) {
      if (!byModule.has(m)) byModule.set(m, []);
      byModule.get(m).push(k);
    }
    const diff = [...per].filter(([, v]) => v.verdicts.has('diff')).map(([k]) => k);
    const same = [...per].filter(([k, v]) => hits.has(k) && [...v.verdicts].every((x) => x === 'same')).map(([k]) => k);
    console.log(`routes-coverage: ${res.a} vs ${res.b}: ${same.length} routes identical on every step, ${byModule.size ? [...byModule.values()].flat().length : 0} route(s) with unported steps, ${diff.length} with diffs, ${unregistered.length} not registered`);
    for (const k of unregistered) console.log(`  NOT REGISTERED in ${res.b}: ${k}`);
    for (const [m, ks] of [...byModule].sort()) console.log(`  unported (${m}): ${[...new Set(ks)].sort().join(', ')}`);
    for (const k of diff) console.log(`  diffs: ${k}`);
  }
  const ok = missing.length === 0 && unregistered.length === 0;
  console.log(ok ? 'routes-coverage: OK' : 'routes-coverage: FAILED');
  return ok ? 0 : 1;
}

if (import.meta.url === `file://${process.argv[1]}`) process.exit(await main());
