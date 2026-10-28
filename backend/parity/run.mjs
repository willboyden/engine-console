#!/usr/bin/env node
// Differential harness: the same corpus against two Engine Console backends, compared after normalisation.
//
//   node backend/parity/run.mjs                         the last Python release vs the Rust hub (backend/target/debug)
//   node backend/parity/run.mjs --b python              Python vs Python (self-test: must be 0 diffs)
//   options: --a python[@<ref>]|rust  --b python[@<ref>]|rust  --modes nofleet,fleet,hf
//            --node python[@<ref>]|rust|default  --filter <corpus file substring>  --json-out <file>  --keep  --verbose
//
// `python` is the Python hub and node of the tag `python-final` (else commit 15ac7a6), exported and set up once by
// lib/python-release.mjs; `python@<ref>` names another commit or tag, and EC_PYTHON_REF changes the default.
// Exit 0 when every step matches, except (python vs rust only) steps an allowance (allowed-diffs.json) covers and
// steps a backend answers with 501 "not ported: <module>", which are listed per module. See backend/parity/README.md.
import { writeFileSync } from 'node:fs';
import { loadAllowances, runHarness } from './lib/harness.mjs';

function args(argv) {
  const o = { a: 'python', b: 'rust', modes: ['nofleet', 'fleet', 'hf'], node: 'python', filter: null, jsonOut: null,
    keep: false, verbose: false };
  for (let i = 0; i < argv.length; i++) {
    const k = argv[i];
    const next = () => { if (i + 1 >= argv.length) throw new Error(`${k} needs a value`); return argv[++i]; };
    if (k === '--a') o.a = next();
    else if (k === '--b') o.b = next();
    else if (k === '--modes') o.modes = next().split(',').filter(Boolean);
    else if (k === '--node') o.node = next();
    else if (k === '--filter') o.filter = next();
    else if (k === '--json-out') o.jsonOut = next();
    else if (k === '--keep') o.keep = true;
    else if (k === '--verbose') o.verbose = true;
    else throw new Error(`unknown option ${k}`);
  }
  if (o.node === 'default') o.node = null;
  return o;
}

export function summarise(res) {
  const by = { same: 0, allowed: 0, diff: 0, unported: 0, 'not-registered': 0 };
  const unported = new Map();
  for (const s of res.steps) {
    by[s.verdict] = (by[s.verdict] || 0) + 1;
    if (s.verdict === 'unported') {
      if (!unported.has(s.module)) unported.set(s.module, new Set());
      unported.get(s.module).add(s.route);
    }
  }
  const expectFails = res.steps.filter((s) => s.expectFail);
  return { by, unported, expectFails };
}

export async function main(argv = process.argv.slice(2)) {
  let o;
  try { o = args(argv); } catch (e) { console.error(`run.mjs: ${e.message}`); return 2; }
  const t0 = Date.now();
  console.log(`parity: ${o.a} vs ${o.b}; modes ${o.modes.join('+')}; local node ${o.node ?? 'each hub\'s default'}`);
  const res = await runHarness({ ...o, log: (m) => console.log(m) });
  if (o.jsonOut) writeFileSync(o.jsonOut, JSON.stringify(res, null, 1));
  const { by, unported, expectFails } = summarise(res);
  // allowances (allowed-diffs.json): which ones applied, and listed steps that no longer need one (remove those)
  if (o.a !== o.b) {
    const used = new Set(res.steps.filter((s) => s.allowed).map((s) => `${s.allowed}|${s.file}|${s.scenario}|${s.index}`));
    const ran = new Set(res.steps.map((s) => `${s.file}|${s.scenario}|${s.index}`));
    for (const al of loadAllowances()) {
      const n = res.steps.filter((s) => s.allowed === al.name).length;
      if (n) console.log(`allowed: ${al.name} on ${n} step(s) (headers only; status and body equal)`);
      for (const st of al.steps) {
        const key = `${st.file}|${st.scenario}|${st.index}`;
        if (ran.has(key) && !used.has(`${al.name}|${key}`)) console.log(`STALE ALLOWANCE ${al.name}: ${st.file} › ${st.scenario} #${st.index} no longer differs that way: remove it`);
      }
    }
  }
  const bad = res.steps.filter((s) => s.verdict === 'diff' || s.verdict === 'not-registered');
  const shown = o.verbose ? bad : bad.slice(0, 40);
  for (const s of shown) {
    console.log(`DIFF [${s.mode}] ${s.file} › ${s.scenario} #${s.index} ${s.method} ${s.path}  (${s.route})${s.verdict === 'not-registered' ? '  NOT REGISTERED' : ''}`);
    for (const d of s.diffs.slice(0, o.verbose ? 40 : 6)) console.log(`    ${d.path}: ${o.a}=${d.a}  ${o.b}=${d.b}`);
  }
  if (bad.length > shown.length) console.log(`... ${bad.length - shown.length} more diff steps (--verbose)`);
  for (const s of expectFails) console.log(`EXPECTATION [${s.mode}] ${s.file} › ${s.scenario} #${s.index} ${s.method} ${s.path}: ${s.expectFail}`);
  if (unported.size) {
    console.log(`unported on ${o.b} (501 "not ported"), by module:`);
    for (const [mod, routes] of [...unported].sort()) console.log(`  ${mod}: ${[...routes].sort().join(', ')}`);
  }
  if (res.hf) console.log(`hf mode: ${res.hf.requests} requests to the fake hub; CONNECT targets: ${res.hf.connects.join(', ')}`);
  for (const e of res.errors) console.log(`ERROR ${e}`);
  const steps = res.steps.length;
  console.log(`parity: ${steps} steps: ${by.same} same, ${by.allowed} allowed, ${by.diff} diff, ${by['not-registered']} not registered, ${by.unported} unported; ${expectFails.length} expectation failures; ${res.errors.length} errors; ${((Date.now() - t0) / 1000).toFixed(1)} s`);
  const selfTest = o.a === o.b;
  const ok = steps > 0 && !res.errors.length && !bad.length && !expectFails.length && (!selfTest || by.unported === 0);
  console.log(ok ? 'parity: OK' : 'parity: FAILED');
  return ok ? 0 : 1;
}

if (import.meta.url === `file://${process.argv[1]}`) process.exit(await main());
