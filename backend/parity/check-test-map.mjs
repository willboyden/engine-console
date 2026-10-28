#!/usr/bin/env node
// Check backend/parity/test-map.json (written by gen/test_map.py, the source of backend/TEST-MAP.md):
//   * every Rust test it names (`<crate>::<fn name>`) exists as a TEST: a `fn <name>` under backend/crates/<crate>/
//     carrying a test attribute (`#[test]`, `#[tokio::test(..)]`, `#[test_case]`, `#[rstest]`, or `#[test]` inside a
//     proptest! block), not merely a helper with that name;
//   * with --cargo, every one is also compiled into a test binary and listed by `cargo test -- --list` (a test file
//     that no binary includes, e.g. a tests/<dir>/*.rs without its `mod` line, never runs);
//   * every row is mapped to Rust test(s) or marked `N/A: <reason>`.
// Prints mapped / N/A / unmapped counts per area and exits non-zero on a missing Rust test or an unmapped row.
//
//   node backend/parity/check-test-map.mjs                    every area
//   node backend/parity/check-test-map.mjs --owner fleet      one area (the `owner` field of test-map.json)
//   node backend/parity/check-test-map.mjs --cargo            also require each test in `cargo test --locked -- --list`
//   node backend/parity/check-test-map.mjs --allow-unmapped   report unmapped rows without failing
import { spawnSync } from 'node:child_process';
import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const PARITY = dirname(fileURLToPath(import.meta.url));
const CRATES = join(dirname(PARITY), 'crates');

function rsFiles(dir, out = []) {
  if (!existsSync(dir)) return out;
  for (const name of readdirSync(dir)) {
    if (name === 'target') continue;
    const p = join(dir, name);
    const st = statSync(p);
    if (st.isDirectory()) rsFiles(p, out);
    else if (name.endsWith('.rs')) out.push(p);
  }
  return out;
}

const TEST_ATTR = /^\s*#\[(test|tokio::test|test_case|rstest)\b/;

/** Is the `fn` on line `i` a test: a test attribute among the attributes / doc comments right above it? */
function isTest(lines, i) {
  for (let j = i - 1; j >= 0 && j >= i - 16; j--) {
    if (TEST_ATTR.test(lines[j])) return true;
    const p = lines[j].trim();
    // other attributes (also multi-line ones such as #[ignore = "..."]), doc comments, comments, blank lines
    if (p.startsWith('#[') || p.startsWith('//') || p === '' || p.endsWith(')]') || p.endsWith(']') || p.startsWith('"')) continue;
    return false;
  }
  return false;
}

/** crate -> { all: Set of fn names defined anywhere, tests: Set of those carrying a test attribute }. */
export function rustFns() {
  const out = new Map();
  if (!existsSync(CRATES)) return out;
  for (const crate of readdirSync(CRATES)) {
    const all = new Set(); const tests = new Set();
    for (const f of rsFiles(join(CRATES, crate))) {
      const lines = readFileSync(f, 'utf8').split('\n');
      lines.forEach((l, i) => {
        for (const m of l.matchAll(/\bfn\s+([A-Za-z_][A-Za-z0-9_]*)/g)) {
          all.add(m[1]);
          if (isTest(lines, i)) tests.add(m[1]);
        }
      });
    }
    out.set(crate, { all, tests });
  }
  return out;
}

/** crate -> Set of test fn names (last path segment) that `cargo test --locked -p <crate> -- --list` reports. */
function cargoListed(crates) {
  const out = new Map();
  for (const crate of crates) {
    const r = spawnSync('cargo', ['test', '--offline', '--locked', '-q', '-p', crate, '--all-features', '--', '--list',
      '--format', 'terse'], { cwd: dirname(PARITY), encoding: 'utf8', maxBuffer: 64 << 20 });
    if (r.error) throw new Error(`cannot run cargo (${r.error.code || r.error.message}): put ~/.cargo/bin on PATH`);
    if (r.status !== 0) throw new Error(`cargo test --list failed for ${crate}:\n${(r.stderr || '').slice(-2000)}`);
    const names = new Set();
    for (const line of r.stdout.split('\n')) {
      if (line.endsWith(': test')) names.add(line.slice(0, -': test'.length).split('::').pop());
    }
    out.set(crate, names);
  }
  return out;
}

export function main(argv = process.argv.slice(2)) {
  let only = null; let allowUnmapped = false; let cargo = false;
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--owner') only = argv[++i];
    else if (argv[i] === '--allow-unmapped') allowUnmapped = true;
    else if (argv[i] === '--cargo') cargo = true;
    else { console.error(`check-test-map: unknown option ${argv[i]}`); return 2; }
  }
  const file = join(PARITY, 'test-map.json');
  if (!existsSync(file)) { console.error('check-test-map: backend/parity/test-map.json missing: run python3 backend/parity/gen/test_map.py'); return 2; }
  const map = JSON.parse(readFileSync(file, 'utf8'));
  const fns = rustFns();
  let listed = null;
  if (cargo) {
    try { listed = cargoListed([...fns.keys()]); } catch (e) { console.error(`check-test-map: ${e.message}`); return 2; }
  }
  const counts = new Map();
  const missing = [];
  const bad = [];
  let stale = 0;
  for (const [id, row] of Object.entries(map)) {
    if (only && row.owner !== only) continue;
    if (row.stale) { stale++; continue; }
    const c = counts.get(row.owner) || { mapped: 0, na: 0, unmapped: 0 };
    counts.set(row.owner, c);
    const rust = row.rust;
    if (typeof rust === 'string') {
      if (/^N\/A: \S/.test(rust)) c.na++;
      else { bad.push(`${id}: "${rust}" is neither a list of Rust tests nor "N/A: <reason>"`); c.unmapped++; }
    } else if (Array.isArray(rust) && rust.length) {
      c.mapped++;
      for (const ref of rust) {
        const m = /^([A-Za-z0-9_-]+)::([A-Za-z_][A-Za-z0-9_]*)$/.exec(ref);
        if (!m) { bad.push(`${id}: "${ref}" is not <crate>::<fn name>`); continue; }
        const f = fns.get(m[1]);
        if (!f?.all.has(m[2])) missing.push(`${id}: ${ref} not found (no \`fn ${m[2]}\` in backend/crates/${m[1]})`);
        else if (!f.tests.has(m[2])) missing.push(`${id}: ${ref} is not a test (no #[test]-style attribute on \`fn ${m[2]}\`)`);
        else if (listed && !listed.get(m[1])?.has(m[2])) missing.push(`${id}: ${ref} is not compiled into any test binary (not in \`cargo test -p ${m[1]} -- --list\`)`);
      }
    } else {
      c.unmapped++;
    }
  }
  let total = { mapped: 0, na: 0, unmapped: 0 };
  for (const [o, c] of [...counts].sort()) {
    console.log(`  ${o.padEnd(5)} ${String(c.mapped).padStart(5)} mapped  ${String(c.na).padStart(4)} N/A  ${String(c.unmapped).padStart(5)} unmapped`);
    total = { mapped: total.mapped + c.mapped, na: total.na + c.na, unmapped: total.unmapped + c.unmapped };
  }
  for (const m of missing) console.log(`MISSING RUST TEST ${m}`);
  for (const b of bad) console.log(`BAD ROW ${b}`);
  console.log(`check-test-map: ${total.mapped} mapped, ${total.na} N/A, ${total.unmapped} unmapped${stale ? `, ${stale} stale` : ''}${only ? ` (area ${only})` : ''}${cargo ? '; every mapped test compiled and listed by cargo' : ''}`);
  const ok = !missing.length && !bad.length && (allowUnmapped || total.unmapped === 0);
  console.log(ok ? 'check-test-map: OK' : 'check-test-map: FAILED');
  return ok ? 0 : 1;
}

if (import.meta.url === `file://${process.argv[1]}`) process.exit(main());
