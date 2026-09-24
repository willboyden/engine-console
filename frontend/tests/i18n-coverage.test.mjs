import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const en = JSON.parse(fs.readFileSync(path.join(root, 'i18n/en.json'), 'utf8'));
const walk = (d) => fs.readdirSync(d, { withFileTypes: true }).flatMap((e) => (e.isDirectory() ? walk(path.join(d, e.name)) : e.name.endsWith('.js') ? [path.join(d, e.name)] : []));
const files = walk(path.join(root, 'js'));
const ns = new Set(Object.keys(en).map((k) => k.split('.')[0]));

test('every literal t("key") and namespaced key literal used in js/ exists in en.json', () => {
  const missing = [];
  for (const f of files) {
    const src = fs.readFileSync(f, 'utf8');
    const keys = new Set();
    for (const m of src.matchAll(/\bt\(\s*['"`]([a-z_]+\.[a-z_0-9.]+)['"`]/g)) keys.add(m[1]);
    for (const m of src.matchAll(/['"]([a-z_]+\.[a-z_0-9]+)['"]/g)) if (ns.has(m[1].split('.')[0]) && /^(nav|dl|models|bench|inst|arena|usage|settings|audit|launch|chat|dash|metrics|fit|params|common|palette|auth|topbar)\./.test(m[1])) keys.add(m[1]);
    for (const k of keys) {
      if (/\.(md|json|csv)$/.test(k)) continue; // file names, not keys
      // plural forms live as key_one / key_other
      if (!(k in en) && !(`${k}_one` in en)) missing.push(`${path.relative(root, f)}: ${k}`);
    }
  }
  assert.deepEqual(missing, []);
});

test('dynamic key families are complete', () => {
  const fam = { 'nav.': ['dashboard', 'models', 'downloads', 'launch', 'instances', 'metrics', 'bench', 'chat', 'arena', 'usage', 'settings'],
    'group.': ['memory', 'parallelism', 'context', 'quantization', 'scheduling', 'tools_reasoning', 'performance', 'speculative', 'network', 'advanced'],
    'state.': ['stopped', 'starting', 'loading', 'ready', 'stopping', 'failed'], 'phase.': ['loading_weights', 'compiling', 'capturing_graphs', 'ready'],
    'bench.suite_': ['quick', 'standard', 'full', 'prefix'], 'chat.role_': ['user', 'assistant', 'system'], 'inst.did_': ['start', 'stop', 'restart'], 'models.sort_': ['downloads', 'likes', 'updated'] };
  for (const [p, ks] of Object.entries(fam)) for (const k of ks) assert.ok(`${p}${k}` in en, `${p}${k}`);
});

test('no en.json value is empty', () => { for (const [k, v] of Object.entries(en)) assert.ok(typeof v === 'string' && v.length, k); });

test('no source file references an external origin (zero CDN / zero third-party)', () => {
  for (const f of [...files, path.join(root, 'index.html'), ...fs.readdirSync(path.join(root, 'css')).map((x) => path.join(root, 'css', x))]) {
    const src = fs.readFileSync(f, 'utf8');
    // allow the SVG namespace URI and doc links rendered as data (docs_url comes from the backend, not this source)
    const bad = [...src.matchAll(/https?:\/\/[^\s'"`)]+/g)].map((m) => m[0]).filter((u) => !u.startsWith('http://www.w3.org/2000/svg'));
    assert.deepEqual(bad, [], path.relative(root, f));
  }
});
