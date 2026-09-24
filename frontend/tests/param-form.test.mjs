import test from 'node:test';
import assert from 'node:assert/strict';
import { defaultsOf, filterSpecs, groupSpecs, coerce, validateValue, validateAll, memoryFingerprint, normalizePresets, toCliArgs, cliString, shellQuote, pruneValues, changedCount } from '../js/param-form.js';
import { vllmParams, sglangParams } from '../dev/mock-catalogs.mjs';

const S = (o) => ({ key: 'k', flag: '--k', label: 'K', help: 'h', type: 'int', group: 'memory', default: null, choices: null, min: null, max: null, advanced: false, affects_memory: false, requires_restart: true, bool_style: 'flag', docs_url: null, ...o });

test('coerce by type; empty means unset', () => {
  assert.deepEqual(coerce(S({}), ''), { ok: true, value: undefined });
  assert.deepEqual(coerce(S({}), '42'), { ok: true, value: 42 });
  assert.equal(coerce(S({}), '4.2').ok, false);
  assert.equal(coerce(S({ type: 'float' }), '0.9').value, 0.9);
  assert.equal(coerce(S({ type: 'float' }), 'abc').ok, false);
  assert.equal(coerce(S({ type: 'bool' }), 'false').value, false);
  assert.deepEqual(coerce(S({ type: 'string_list' }), 'a, b\nc,, ').value, ['a', 'b', 'c']);
  assert.deepEqual(coerce(S({ type: 'json' }), '{"a":1}').value, { a: 1 });
  assert.equal(coerce(S({ type: 'json' }), '{oops').ok, false);
});

test('validateValue: min/max and enum choices', () => {
  const s = S({ type: 'float', min: 0.1, max: 0.99 });
  assert.equal(validateValue(s, 0.5), null); assert.match(validateValue(s, 1.2), /Maximum/); assert.match(validateValue(s, 0), /Minimum/);
  assert.match(validateValue(S({ type: 'enum', choices: ['a'] }), 'b'), /allowed/);
  assert.equal(validateValue(s, undefined), null);
  assert.deepEqual(validateAll([s, S({ key: 'z', type: 'int', min: 1 })], { k: 0.5, z: 0 }), { z: 'Minimum is 1' });
});

test('filterSpecs hides advanced unless shown or searched; search hits key/flag/help', () => {
  const specs = [S({ key: 'a', label: 'Alpha' }), S({ key: 'b', label: 'Beta', advanced: true, help: 'offload to cpu' })];
  assert.deepEqual(filterSpecs(specs, {}).map((s) => s.key), ['a']);
  assert.deepEqual(filterSpecs(specs, { showAdvanced: true }).map((s) => s.key), ['a', 'b']);
  assert.deepEqual(filterSpecs(specs, { query: 'CPU' }).map((s) => s.key), ['b']);
  assert.deepEqual(filterSpecs(specs, { query: '--k' }).map((s) => s.key), ['a', 'b']);
});

test('groupSpecs uses canonical group order and keeps unknown groups last', () => {
  const g = groupSpecs([S({ key: 'x', group: 'advanced' }), S({ key: 'y', group: 'memory' }), S({ key: 'z', group: 'weird' })]);
  assert.deepEqual(g.map((x) => x.group), ['memory', 'advanced', 'weird']);
});

test('memoryFingerprint changes only when an affects_memory value changes', () => {
  const specs = [S({ key: 'm', affects_memory: true }), S({ key: 'n' })];
  const a = memoryFingerprint(specs, { m: 1, n: 1 });
  assert.equal(memoryFingerprint(specs, { m: 1, n: 2 }), a);
  assert.notEqual(memoryFingerprint(specs, { m: 2, n: 1 }), a);
});

test('normalizePresets accepts map and array forms', () => {
  assert.deepEqual(normalizePresets({ balanced: { a: 1 } }), [{ name: 'balanced', label: 'balanced', params: { a: 1 } }]);
  assert.deepEqual(normalizePresets([{ id: 'x', params: { b: 2 } }]), [{ name: 'x', label: 'x', params: { b: 2 } }]);
  assert.deepEqual(normalizePresets(null), []);
});

test('toCliArgs honours bool_style and types, only emits set keys', () => {
  const specs = [S({ key: 'i', flag: '--i' }), S({ key: 'f', flag: '--f', type: 'bool' }), S({ key: 'v', flag: '--v', type: 'bool', bool_style: 'value' }),
    S({ key: 'l', flag: '--l', type: 'string_list' }), S({ key: 'j', flag: '--j', type: 'json' }), S({ key: 'off', flag: '--off', type: 'bool' }), S({ key: 'unset', flag: '--unset' })];
  const args = toCliArgs(specs, { i: 3, f: true, v: false, l: ['a', 'b'], j: { x: 1 }, off: false });
  assert.deepEqual(args, ['--i', '3', '--f', '--v', 'false', '--l', 'a', 'b', '--j', '{"x":1}']);
});

test('shell quoting neutralises metacharacters', () => {
  assert.equal(shellQuote('safe-1.2/x'), 'safe-1.2/x');
  assert.equal(shellQuote('a b; rm -rf /'), "'a b; rm -rf /'");
  assert.equal(shellQuote("it's"), `'it'\\''s'`);
  assert.equal(cliString(['--j', '{"x": 1}']), `--j '{"x": 1}'`);
});

test('pruneValues drops unknown keys and undefined; changedCount', () => {
  assert.deepEqual(pruneValues([S({ key: 'a' })], { a: 1, gone: 2, b: undefined }), { a: 1 });
  assert.equal(changedCount({ a: 1, b: undefined }), 1);
  assert.deepEqual(defaultsOf([S({ key: 'a', default: 3 }), S({ key: 'b' })]), { a: 3 });
});

test('mock catalogs conform to the ParamSpec shape (keys unique, enums have choices, flags are --x, env:NAME or @image)', () => {
  for (const cat of [vllmParams, sglangParams]) {
    const keys = new Set();
    for (const s of cat) {
      assert.ok(!keys.has(s.key), `dup ${s.key}`); keys.add(s.key);
      assert.match(s.flag, /^(--|env:[A-Z_]+$|@image$)/); assert.ok(s.help && s.label);
      assert.ok(['int', 'float', 'bool', 'enum', 'string', 'string_list', 'json'].includes(s.type));
      if (s.type === 'enum') assert.ok(s.choices?.length, s.key);
    }
    assert.ok(cat.some((s) => s.affects_memory) && cat.some((s) => s.advanced));
  }
});

import { isNonCli, isSecretSpec, toEnvLines } from '../js/param-form.js';
test('env:NAME and @image params are not CLI flags; secrets are masked in previews (AGENTS.md rule 6)', () => {
  const specs = [S({ key: 'hf_token', flag: 'env:HF_TOKEN', type: 'string' }), S({ key: 'nccl', flag: 'env:NCCL_DEBUG', type: 'string' }), S({ key: 'image', flag: '@image', type: 'string' }), S({ key: 'k', flag: '--k' })];
  const values = { hf_token: 'hf_supersecretvalue', nccl: 'INFO', image: 'x:1', k: 1 };
  assert.ok(isNonCli(specs[0]) && isNonCli(specs[2]) && !isNonCli(specs[3]));
  assert.ok(isSecretSpec(specs[0]) && !isSecretSpec(specs[1]));
  assert.deepEqual(toCliArgs(specs, values), ['--k', '1']);
  const lines = toEnvLines(specs, values);
  assert.deepEqual(lines, ['HF_TOKEN=[set, 19 chars]', 'NCCL_DEBUG=INFO']);
  assert.ok(!lines.join('').includes('supersecret') && !cliString(toCliArgs(specs, values)).includes('supersecret'));
});

import { splitRedacted, REDACTED } from '../js/param-form.js';
test('"[set]" placeholders are never round-tripped: pruneValues drops them, splitRedacted reports the keys', () => {
  const specs = [S({ key: 'hf_token', flag: 'env:HF_TOKEN' }), S({ key: 'a' })];
  assert.equal(REDACTED, '[set]');
  assert.deepEqual(pruneValues(specs, { hf_token: '[set]', a: 1 }), { a: 1 });
  assert.deepEqual(splitRedacted({ hf_token: '[set]', a: 1, b: '[set]' }), { values: { a: 1 }, redacted: ['hf_token', 'b'] });
  assert.deepEqual(pruneValues(specs, { hf_token: 'hf_real' }), { hf_token: 'hf_real' });
});
