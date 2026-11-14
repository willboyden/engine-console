// The config editor's pure modules: dotenv line editing (comments and order kept, values read like the hub reads
// them), JSON parse/serialise/paths, the line diff, the field catalogs (every (i) has an explanation in the real learn
// bundle) and the install commands (text only).
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseLines, readValue, formatValue, values, setValue, removeKey, MAX_LINE } from '../js/cfg-dotenv.js';
import { parseJson, toText, getPath, setPath } from '../js/cfg-json.js';
import { lineDiff, diffText } from '../js/cfg-diff.js';
import {
  allFieldConcepts,
  fieldValue,
  fieldText,
  installCommands,
  shq,
  looksSecret,
  ENV_KIND,
  CFG_FILES,
} from '../js/cfg-fields.js';
import { envSettings, configFile, validate } from '../dev/mock-config.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const concepts = JSON.parse(
  fs.readFileSync(path.join(root, '../backend/crates/ec-hub/learn/concepts.json'), 'utf8')
).concepts;

test('dotenv values are read as the hub reads them (quotes, escapes, comments, export)', () => {
  const text = '# head\nexport A=1\nB=\'x # y\'\nC="q\\n\\"z\\"" \nD=plain # trailing\n\nE=\nnot a pair\n';
  const v = values(text);
  assert.deepEqual(Object.fromEntries(Object.entries(v).map(([k, x]) => [k, x.value])), {
    A: '1',
    B: 'x # y',
    C: 'q\n"z"',
    D: 'plain',
    E: '',
  });
  assert.equal(v.D.n, 5);
  assert.deepEqual(
    parseLines(text).map((l) => l.kind),
    ['comment', 'pair', 'pair', 'pair', 'pair', 'blank', 'pair', 'other', 'blank']
  );
  assert.equal(readValue("  'a'b"), 'a');
});

test('long lines parse in linear time; a line over MAX_LINE is not read', () => {
  const t0 = Date.now();
  for (const line of [' '.repeat(40000), `x${' '.repeat(40000)}`, `A${' '.repeat(4000)}`, `${'k'.repeat(4000)}y`]) {
    for (let i = 0; i < 20; i++) parseLines(line);
  }
  // the same shapes inside the line cap, many times over
  const near = `x${' '.repeat(MAX_LINE - 2)}\n`.repeat(256);
  assert.equal(parseLines(near).filter((l) => l.kind === 'other').length, 256);
  assert.ok(Date.now() - t0 < 1500, `took ${Date.now() - t0} ms`);
  assert.deepEqual(parseLines(`A=${'v'.repeat(MAX_LINE)}`)[0], {
    n: 1,
    raw: `A=${'v'.repeat(MAX_LINE)}`,
    kind: 'other',
    long: true,
  });
  // the key still ends at the first `=` and is trimmed, as before
  assert.deepEqual(parseLines('  export  K E Y  = v=w')[0].key, 'K E Y');
  assert.equal(parseLines('export =1')[0].key, 'export');
  assert.equal(parseLines('#=x')[0].kind, 'comment');
  assert.equal(parseLines('=x')[0].kind, 'other');
});

test('formatValue round-trips through readValue for awkward values', () => {
  for (const v of [
    'plain',
    'with space',
    'a # b',
    "it's",
    '["a", "b"]',
    'x${HOME}y',
    ' lead',
    'tab\there',
    'line\nbreak',
    '"quoted"',
    "'single'",
    'back\\slash',
    '',
  ]) {
    assert.equal(readValue(formatValue(v)), v, JSON.stringify(v));
  }
  assert.equal(formatValue('http://127.0.0.1:8082'), 'http://127.0.0.1:8082');
});

test('setValue rewrites only the last assignment line and keeps comments, order and export', () => {
  const text = '# proxy\nexport EGRESS_PROXY=http://a\nMAX_SSE_PER_KEY=4\n# end\nMAX_SSE_PER_KEY=5\n';
  const out = setValue(text, 'MAX_SSE_PER_KEY', '3');
  assert.equal(out, '# proxy\nexport EGRESS_PROXY=http://a\nMAX_SSE_PER_KEY=4\n# end\nMAX_SSE_PER_KEY=3\n');
  assert.equal(
    setValue(text, 'EGRESS_PROXY', 'http://b #c'),
    "# proxy\nexport EGRESS_PROXY='http://b #c'\nMAX_SSE_PER_KEY=4\n# end\nMAX_SSE_PER_KEY=5\n"
  );
  assert.equal(setValue('# only a comment', 'NEW', 'x'), '# only a comment\nNEW=x\n');
  assert.equal(setValue('', 'NEW', 'x'), 'NEW=x\n');
  assert.equal(removeKey(text, 'MAX_SSE_PER_KEY'), '# proxy\nexport EGRESS_PROXY=http://a\n# end\n');
  assert.equal(removeKey(text, 'NOPE'), text);
});

test('parseJson reports a line, toText keeps key order, setPath copies and deletes', () => {
  assert.deepEqual(parseJson('{"a": 1}'), { ok: true, value: { a: 1 } });
  const bad = parseJson('{\n  "a": 1,\n  "b" 2\n}');
  assert.equal(bad.ok, false);
  assert.equal(bad.line, 3);
  const doc = { _comment: 'keep', version: 1, model_roots: [{ id: 'a', path: '/data/a' }] };
  const next = setPath(doc, ['model_roots', 0, 'writable'], true);
  assert.equal(doc.model_roots[0].writable, undefined, 'the input is not changed');
  assert.deepEqual(Object.keys(next), ['_comment', 'version', 'model_roots']);
  assert.equal(getPath(next, ['model_roots', 0, 'writable']), true);
  assert.deepEqual(setPath(next, ['model_roots', 0], undefined).model_roots, []);
  assert.deepEqual(setPath({}, ['docker', 'context'], null), { docker: { context: null } });
  assert.equal(toText({ a: [1] }), '{\n  "a": [\n    1\n  ]\n}\n');
});

test('lineDiff and diffText show only the changed lines with context', () => {
  const a = ['1', '2', '3', '4', '5', '6', '7', '8', '9'].join('\n');
  const b = ['1', '2', '3', '4', 'five', '6', '7', '8', '9', '10'].join('\n');
  const ops = lineDiff(a, b);
  assert.deepEqual(
    ops.filter((o) => o.op !== ' '),
    [
      { op: '-', text: '5' },
      { op: '+', text: 'five' },
      { op: '+', text: '10' },
    ]
  );
  const d = diffText(ops, 1);
  assert.match(d, /^…\n {2}4\n- 5\n\+ five\n {2}6\n…\n {2}9\n\+ 10$/);
  assert.equal(diffText(lineDiff('same', 'same')), '');
  assert.equal(lineDiff('x\n'.repeat(3000), 'y\n'.repeat(3000)), null, 'too large is refused, not frozen');
});

test('form values parse by kind; bad input is reported, never silently changed', () => {
  assert.deepEqual(fieldValue('int', ' 2222 '), { ok: true, value: 2222 });
  assert.equal(fieldValue('int', '22.5').ok, false);
  assert.deepEqual(fieldValue('text', ''), { ok: true, value: undefined });
  assert.deepEqual(fieldValue('text', '', { nullable: true }), { ok: true, value: null });
  assert.deepEqual(fieldValue('pair', '18000, 18099'), { ok: true, value: [18000, 18099] });
  assert.equal(fieldValue('pair', '18000').ok, false);
  assert.deepEqual(fieldValue('ints', '8000 8080,11434'), { ok: true, value: [8000, 8080, 11434] });
  assert.deepEqual(fieldValue('lines', 'a:*\n\n b@sha256:* '), { ok: true, value: ['a:*', 'b@sha256:*'] });
  assert.deepEqual(fieldValue('bool', false), { ok: true, value: false });
  assert.equal(fieldText('pair', [1, 2]), '1, 2');
  assert.equal(fieldText('text', null), '');
});

test('every (i) of the Form view has an explanation in the real learn bundle', () => {
  const names = envSettings().map((f) => f.name);
  assert.ok(names.length >= 50, `${names.length} env settings read from envfile.rs`);
  const missing = allFieldConcepts(names).filter((k) => !concepts[k]?.summary);
  assert.deepEqual(missing, []);
  for (const k of ['page.config', 'field.default_ttl', 'field.host_image_pins']) assert.ok(concepts[k]?.summary, k);
  for (const f of envSettings()) assert.ok(ENV_KIND[f.kind], `no control for ${f.kind}`);
});

test('install commands are text with the right modes and never echo anything secret', () => {
  const h = installCommands('hosts', { path: '~/.config/engine-console/hosts.json', download: 'hosts.json' });
  assert.match(h, /install -D -m 600 \.\/hosts\.json ~\/\.config\/engine-console\/hosts\.json/);
  assert.match(h, /engine-console hosts check/);
  const p = installCommands('policy', { path: '/etc/ecnode/policy.json', download: 'policy.json' });
  assert.match(p, /sudo install -D -m 644 -o root -g root \.\/policy\.json \/etc\/ecnode\/policy\.json/);
  assert.match(p, /ecnode check --policy \/etc\/ecnode\/policy\.json/);
  assert.match(
    installCommands('env', { path: '/srv/my env', download: 'env' }),
    /install -D -m 600 \.\/env '\/srv\/my env'/
  );
  assert.equal(shq("it's"), "'it'\\''s'");
  // the plaintext download of an env draft is removed after it is installed
  const e = installCommands('env', { path: '/srv/ec/env', download: 'env' }).split('\n');
  assert.ok(e.indexOf('rm -- ./env') > e.indexOf('install -D -m 600 ./env /srv/ec/env'), e.join('\n'));
});

test('a draft with something secret-looking is flagged for the Export card', () => {
  const pairs = (text) => parseLines(text).filter((l) => l.kind === 'pair');
  for (const text of [
    'HF_TOKEN=abc',
    'EGRESS_PROXY=http://user:pw@127.0.0.1:8082',
    'X=hf_abcdefghijk',
    'MY_API_KEY="k"',
  ]) {
    assert.equal(looksSecret(text, pairs(text)), true, text);
  }
  for (const text of [
    'HF_TOKEN=',
    'HF_TOKEN_FILE=/path/to/hf_token',
    'EGRESS_PROXY=http://127.0.0.1:8082',
    '# HF_TOKEN=abc',
  ]) {
    assert.equal(looksSecret(text, pairs(text)), false, text);
  }
  assert.equal(looksSecret('{"hosts": [{"id": "local", "label": "hf_abcdefghijk"}]}'), true);
  assert.equal(looksSecret('{"hosts": [{"id": "local"}]}'), false);
  for (const text of ['http://user:pw@host', 'postgres://u:p@h']) assert.equal(looksSecret(text), true, text);
  for (const text of ['http://127.0.0.1:8082', 'see http://host/a:b@c']) assert.equal(looksSecret(text), false, text);
});

test('the secret scan is linear: a 1 MiB hostile draft is scanned at once', () => {
  // second audit F2: a scheme-anchored URL pattern rescanned a long run of letters from every position (seconds at
  // 64 KiB, much longer at the 1 MiB cap)
  const mib = 1024 * 1024;
  // 64 KiB first: against the old pattern it fails in seconds instead of hanging on the 1 MiB inputs
  for (const text of [
    'a'.repeat(mib / 16),
    'a'.repeat(mib),
    `://${'a'.repeat(mib / 2)}:${'b'.repeat(mib / 2)}`,
    'a://b:'.repeat(mib / 6),
    'Bearer '.repeat(mib / 8),
  ]) {
    const t0 = performance.now();
    looksSecret(text);
    const ms = performance.now() - t0;
    assert.ok(ms < 250, `${text.slice(0, 12)}...: ${ms.toFixed(0)} ms`);
  }
});

test("the mock serves the config routes in the hub's shapes", () => {
  assert.deepEqual(
    CFG_FILES.map((f) => f.id),
    ['hosts', 'env', 'policy']
  );
  assert.equal(configFile('env').text, null);
  assert.equal(configFile('policy').served, false);
  assert.equal(configFile('nope'), null);
  assert.equal(validate('hosts', '{"hosts": [{"id": "local"}]}').ok, true);
  assert.equal(validate('policy', '{"allow_delet": true}').ok, false);
  assert.deepEqual(validate('env', 'NOT_A_SETTING=1\nEGRESS_PROXY=x').outline, {
    set: ['EGRESS_PROXY'],
    unknown: ['NOT_A_SETTING'],
  });
});

// ---- the line editor against the hub's own reader (the engine module, ADR 0023; skipped without `make wasm`) ----
const PKG = path.join(root, 'pkg');
const pkgBuilt = fs.existsSync(path.join(PKG, 'ec_engines_wasm_bg.wasm'));

test(
  "dotenv values: the line editor reads every value the way the hub's reader does (without ${VAR} expansion)",
  { skip: pkgBuilt ? false : 'frontend/pkg/ not built (run `make wasm`)' },
  async () => {
    const { createLoader } = await import('../js/engines-wasm.js');
    const { pathToFileURL } = await import('node:url');
    const mod = await createLoader({
      glueUrl: () => `${pathToFileURL(path.join(PKG, 'ec_engines_wasm.js')).href}?cfg-test=1`,
      wasmSource: () => fs.readFileSync(path.join(PKG, 'ec_engines_wasm_bg.wasm')),
    }).load();
    assert.ok(mod);
    const lines = [
      'A=1',
      'export B=two words',
      "C='single # kept'",
      'D="dq \\n \\t \\" \\\' \\\\ \\x"',
      'E=plain # trailing comment',
      'F=a#b',
      'G=',
      'H= padded  ',
      "I='unclosed",
      'J="unclosed \\"',
      '  K = spaced',
      'export   L=x',
      'M="q" junk after',
      "N='' # empty",
      'O=1=2',
      'A=last wins',
      '# Z=comment',
      'not a pair',
    ];
    const text = `${lines.join('\n')}\n`;
    const hub = Object.fromEntries(mod.dotenv(text).ok);
    const ours = Object.fromEntries(Object.entries(values(text)).map(([k, v]) => [k, v.value]));
    assert.deepEqual(ours, hub);
    // and per line, so a difference names the line
    for (const l of lines.filter((x) => x.includes('=') && !x.startsWith('#'))) {
      const h = mod.dotenv(`${l}\n`).ok;
      const o = Object.entries(values(`${l}\n`)).map(([k, v]) => [k, v.value]);
      assert.deepEqual(o, h, l);
    }
  }
);
