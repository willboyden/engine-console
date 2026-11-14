// The engine module's loader (js/engines-wasm.js) and the module itself (frontend/pkg/, built by `make wasm` from
// backend/crates/ec-engines-wasm, ADR 0023). The loader tests use fake modules and always run; the module tests load the
// real .wasm and skip cleanly when frontend/pkg/ has not been built.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { wrap, createLoader, envelopeError, API_VERSION } from '../js/engines-wasm.js';

const PKG = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../pkg');
const GLUE = path.join(PKG, 'ec_engines_wasm.js');
const WASM = path.join(PKG, 'ec_engines_wasm_bg.wasm');
const built = fs.existsSync(GLUE) && fs.existsSync(WASM);
const opts = { skip: built ? false : 'frontend/pkg/ not built (run `make wasm`)' };

/** A loader over the real glue and .wasm: a fresh module record per instance (the URL's query differs). */
const realLoader = () =>
  createLoader({
    glueUrl: (n) => `${pathToFileURL(GLUE).href}?instance=${n}`,
    wasmSource: () => fs.readFileSync(WASM),
  });

const HW = {
  gpu_ids: [0],
  gpu_uuids: ['GPU-0'],
  gpu_names: ['NVIDIA RTX PRO 6000'],
  gpu_total_gib: [96],
  gpu_free_gib: [90],
  compute_capability: '12.0',
  host_ram_gib: 256,
};
const ctx = (params, model = 'org/m') => ({
  model,
  served_name: model,
  hf_cache_container_path: '/root/.cache/huggingface',
  params,
  hw: HW,
});

// ---- the loader, with fake modules ----

test('wrap parses JSON answers and marks the instance dead after a throw, calling onTrap once', () => {
  let traps = 0;
  const mod = {
    build_version: () => '9.9.9',
    engines: () => '{"ok": []}',
    catalog: () => {
      throw new WebAssembly.RuntimeError('unreachable');
    },
  };
  const w = wrap(mod, { onTrap: () => traps++ });
  assert.equal(w.version, '9.9.9');
  assert.deepEqual(w.engines(), { ok: [] });
  assert.equal(w.dead, false);
  const r = w.catalog('vllm');
  assert.equal(r.unavailable, 'trap');
  assert.equal(w.dead, true);
  assert.deepEqual(w.engines(), { unavailable: 'dead' });
  assert.equal(traps, 1);
  // a JS error (not a trap) is 'error'
  const w2 = wrap({ engines: () => JSON.parse('{') });
  assert.equal(w2.engines().unavailable, 'error');
  // the image rule answers null, never a guess, when the module cannot answer
  assert.equal(wrap({}).imageAllowed(['a:*'], 'a:1'), null);
});

test('a loader whose glue cannot be imported reports unavailable and never rejects', async () => {
  const l = createLoader({ glueUrl: () => 'file:///nonexistent/ec_engines_wasm.js', wasmSource: () => null });
  const seen = [];
  l.onChange((s) => seen.push(s.state));
  assert.equal(await l.load(), null);
  assert.equal(l.status().state, 'unavailable');
  assert.equal(l.get(), null);
  assert.deepEqual(seen, ['unavailable']);
});

test('a module of another API version is refused (a stale frontend/pkg/)', async () => {
  const dir = fs.mkdtempSync(path.join(fs.realpathSync(path.dirname(fileURLToPath(import.meta.url))), '.tmp-glue-'));
  try {
    const f = path.join(dir, 'glue.mjs');
    fs.writeFileSync(f, `export default async () => {}; export const api_version = () => ${API_VERSION + 1};`);
    const l = createLoader({ glueUrl: () => pathToFileURL(f).href, wasmSource: () => null });
    assert.equal(await l.load(), null);
    assert.match(l.status().reason, /run make wasm/);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('envelopeError gives the text of a problem, an exception or an unavailable answer', () => {
  assert.equal(envelopeError({ problem: { detail: 'nope', code: 'x' } }), 'nope');
  assert.equal(envelopeError({ raised: { kind: 'OverflowError', msg: 'big' } }), 'OverflowError: big');
  assert.equal(envelopeError({ unavailable: 'trap', message: 'unreachable' }), 'unreachable');
  assert.equal(envelopeError({ ok: 1 }), '');
});

// ---- the real module ----

test('the real module loads, speaks this API version and lists the five engines', opts, async () => {
  const l = realLoader();
  const m = await l.load();
  assert.ok(m, l.status().reason);
  assert.equal(l.status().state, 'ready');
  assert.deepEqual(
    m.engines().ok.map((e) => e.id),
    ['vllm', 'sglang', 'trtllm', 'llamacpp', 'ollama']
  );
  assert.match(m.version, /^\d+\.\d+\.\d+/);
});

test(
  'vLLM: the module renders a false bool as the hub does (--no-<flag>), where the old JS preview dropped it',
  opts,
  async () => {
    const m = await realLoader().load();
    const r = m.render('vllm', ctx({ enable_prefix_caching: false }));
    assert.ok(r.ok, JSON.stringify(r));
    assert.ok(r.ok.argv.includes('--no-enable-prefix-caching'), r.ok.argv.join(' '));
    assert.ok(r.ok.argv.join(' ').includes('--served-model-name org/m'));
    assert.ok('VLLM_NO_USAGE_STATS' in r.ok.env);
  }
);

test('validate gives the cross-field rules the old JS validation never had', opts, async () => {
  const m = await realLoader().load();
  const errs = m.validate('vllm', { enable_auto_tool_choice: true }).ok;
  assert.ok(
    errs.some((e) => e.includes('requires tool_call_parser')),
    errs.join('; ')
  );
  assert.deepEqual(m.validate('vllm', { enable_auto_tool_choice: true, tool_call_parser: 'hermes' }).ok, []);
});

test("the image rule is the node's: a tag plus a digest is refused even for repo:*", opts, async () => {
  const m = await realLoader().load();
  const d = '0123456789abcdef'.repeat(4);
  assert.equal(m.imageAllowed(['vllm/vllm-openai:*'], 'vllm/vllm-openai:v1'), true);
  assert.equal(m.imageAllowed(['repo/x:*'], `repo/x:v1@sha256:${d}`), false);
  assert.equal(m.imageAllowed(['repo/x@sha256:*'], `repo/x@sha256:${d}`), true);
  assert.equal(m.imageAllowed(['repo/x:v1'], 'repo/x:v2'), false);
});

test('config checks: an env draft is checked in the page and its token never appears in the answer', opts, async () => {
  const m = await realLoader().load();
  const r = m.configCheck('env', 'HF_TOKEN=hf_abcdefghijklmnop\nENGINE_CONSOLE_PORT=x\n').ok;
  assert.equal(r.file, 'env');
  assert.equal(r.ok, false);
  assert.ok(!JSON.stringify(r).includes('hf_abcdefghijklmnop'));
  assert.equal(r.errors[0].line, 2);
  const p = m.configCheck('policy', '{"version": 1, "mode": "monitor"}').ok;
  assert.equal(p.ok, true);
  assert.equal(p.outline.mode, 'monitor');
  assert.equal(m.configCheck('hosts', '{}').problem.code, 'not_in_browser');
});

test('an adapter panic traps the module; the loader drops it and the next instance answers', opts, async () => {
  const l = realLoader();
  const m = await l.load();
  // An exception returned as a value comes back as `raised` here too ...
  assert.equal(m.render('trtllm', ctx({ kv_cache_max_gib: 1e308 })).raised?.kind, 'OverflowError');
  // ... but one raised by a panic (SGLang validating a null ep_size) traps on wasm32 (panic = abort)
  const r = m.validate('sglang', { ep_size: null });
  assert.equal(r.unavailable, 'trap', JSON.stringify(r));
  assert.equal(m.dead, true);
  assert.equal(l.status().state, 'trapped');
  const fresh = await l.load();
  assert.ok(fresh && fresh !== m, 'a fresh instance');
  assert.ok(fresh.render('vllm', ctx({})).ok, 'the fresh instance answers');
  assert.deepEqual(m.engines(), { unavailable: 'dead' }, 'the trapped instance is never called again');
});
