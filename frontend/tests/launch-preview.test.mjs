// The Launch page's command preview (js/launch-preview.js): the LaunchContext rebuilt from the page's choices, and the
// preview from the hub's own adapters (the real .wasm, skipped when frontend/pkg/ is not built) or, without the module,
// the old approximation.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { createLoader } from '../js/engines-wasm.js';
import {
  displayOf,
  modelPathOf,
  hardwareOf,
  launchContext,
  specText,
  approxText,
  preview,
  HF_CACHE_CONTAINER_PATH,
} from '../js/launch-preview.js';

const PKG = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../pkg');
const built = fs.existsSync(path.join(PKG, 'ec_engines_wasm_bg.wasm'));
const opts = { skip: built ? false : 'frontend/pkg/ not built (run `make wasm`)' };
let n = 0;
const realModule = () =>
  createLoader({
    glueUrl: () => `${pathToFileURL(path.join(PKG, 'ec_engines_wasm.js')).href}?preview=${++n}`,
    wasmSource: () => fs.readFileSync(path.join(PKG, 'ec_engines_wasm_bg.wasm')),
  }).load();

const host = {
  id: 'local',
  node: { arch: 'x86_64' },
  memory: { total_gib: 251.5 },
  gpus: [
    {
      index: 0,
      uuid: 'GPU-aaa',
      name: 'NVIDIA RTX PRO 6000',
      total_gib: 95.6,
      free_gib: 90,
      compute_capability: '12.0',
    },
    {
      index: 1,
      uuid: 'GPU-bbb',
      name: 'NVIDIA RTX PRO 6000',
      total_gib: 95.6,
      free_gib: 80,
      compute_capability: '12.0',
    },
  ],
};
const hf = { format: 'hf', repo_id: 'org/model' };

test('displayOf and modelPathOf follow the hub (runspec.rs display_of / model_mounts)', () => {
  assert.equal(displayOf(hf), 'org/model');
  assert.equal(displayOf({ format: 'ollama', name: 'llama3:8b' }), 'llama3:8b');
  assert.equal(displayOf({ format: 'gguf', root_id: 'r', rel: 'q/Model-Q4_K_M-00001-of-00003.gguf' }), 'Model-Q4_K_M');
  assert.equal(displayOf({ format: 'plain', root_id: 'r', rel: 'ckpts/tiny/' }), 'tiny');
  assert.equal(displayOf({ format: 'plain', root_id: 'r', rel: '' }), 'r');
  assert.equal(modelPathOf(hf), null);
  assert.equal(modelPathOf({ format: 'plain', rel: 'x' }), '/models');
  assert.equal(modelPathOf({ format: 'gguf', rel: 'q/m.gguf' }), '/models/m.gguf');
  assert.equal(
    modelPathOf({ format: 'gguf', rel: 'models--o--m/snapshots/abc/m.gguf' }),
    '/models/snapshots/abc/m.gguf'
  );
});

test('hardwareOf takes the chosen GPUs of the host detail, in index order', () => {
  const hw = hardwareOf(host, new Set([1, 0]));
  assert.deepEqual(hw.gpu_ids, [0, 1]);
  assert.deepEqual(hw.gpu_uuids, ['GPU-aaa', 'GPU-bbb']);
  assert.equal(hw.compute_capability, '12.0');
  assert.equal(hw.host_ram_gib, 251.5);
  assert.equal(hw.arch, 'x86_64');
  assert.deepEqual(hardwareOf(host, [7]).gpu_ids, [], 'a GPU the host does not have is dropped');
});

test('launchContext is what the hub builds; nothing without a model', () => {
  const specs = [{ key: 'max_model_len', type: 'int', flag: '--max-model-len' }];
  const c = launchContext({ ref: hf, specs, values: { max_model_len: 4096, stray: 1 }, host, gpuIds: [0] });
  assert.equal(c.model, 'org/model');
  assert.equal(c.served_name, 'org/model');
  assert.equal(c.model_format, 'hf');
  assert.equal(c.hf_cache_container_path, HF_CACHE_CONTAINER_PATH);
  assert.deepEqual(c.params, { max_model_len: 4096 }, 'only catalog keys, as the launch request sends them');
  assert.equal(launchContext({ ref: null, specs, values: {}, host, gpuIds: [0] }), null);
});

test('specText masks secret env names and secret parameter values wherever the adapter put them', () => {
  const spec = {
    env: { HF_TOKEN: 'hf_secretsecret', VLLM_NO_USAGE_STATS: '1', NOTE: 'a b' },
    files: { '/config/x.yaml': 'k: v\n' },
    argv: ['serve', '--api-key', 'sk-very-secret-value'],
  };
  const { text, secretInArgv } = specText(spec, { secrets: ['sk-very-secret-value'] });
  assert.ok(!text.includes('hf_secretsecret'));
  assert.ok(!text.includes('sk-very-secret-value'));
  assert.match(text, /HF_TOKEN=\[set, 15 chars\]/);
  assert.match(text, /NOTE='a b'/);
  assert.match(text, /^# \/config\/x\.yaml\nk: v$/m);
  assert.match(text, /serve --api-key \[set, 20 chars\]$/);
  assert.equal(secretInArgv, true);
});

test('without the module the preview is the old approximation, and says so', () => {
  const specs = [{ key: 'enable_prefix_caching', type: 'bool', flag: '--enable-prefix-caching' }];
  const p = preview(null, {
    engine: 'vllm',
    ref: hf,
    specs,
    values: { enable_prefix_caching: false },
    host,
    gpuIds: [0],
  });
  assert.equal(p.source, 'approx');
  assert.equal(p.text, approxText({ specs, values: { enable_prefix_caching: false }, ref: hf }));
  assert.equal(p.text, '--model org/model', 'the approximation drops the false bool (the bug the module fixes)');
  assert.equal(preview(null, { engine: '', ref: hf }).source, 'none');
  // a module that traps mid-way: the approximation, with the reason
  const trapped = { validate: () => ({ unavailable: 'trap', message: 'unreachable' }) };
  const t = preview(trapped, { engine: 'vllm', ref: hf, specs, values: {}, host, gpuIds: [0] });
  assert.equal(t.source, 'approx');
  assert.equal(t.reason, 'unreachable');
});

test(
  'with the module, the vLLM preview is the hub argv: --no-<flag> for a false bool, the served name, its env',
  opts,
  async () => {
    const mod = await realModule();
    const specs = mod.catalog('vllm').ok;
    const p = preview(mod, {
      engine: 'vllm',
      ref: hf,
      specs,
      values: { enable_prefix_caching: false, max_model_len: 8192 },
      host,
      gpuIds: [0],
    });
    assert.equal(p.source, 'hub');
    assert.equal(p.refused, null);
    assert.deepEqual(p.errors, []);
    assert.match(p.text, /--no-enable-prefix-caching/);
    assert.match(p.text, /--served-model-name org\/model/);
    assert.match(p.text, /--max-model-len 8192/);
    assert.match(p.text, /^VLLM_NO_USAGE_STATS=/m);
  }
);

test("with the module, the adapter's cross-field rules and refusals reach the page", opts, async () => {
  const mod = await realModule();
  const specs = mod.catalog('vllm').ok;
  const p = preview(mod, {
    engine: 'vllm',
    ref: hf,
    specs,
    values: { enable_auto_tool_choice: true },
    host,
    gpuIds: [0],
  });
  assert.ok(
    p.errors.some((e) => e.includes('requires tool_call_parser')),
    p.errors.join('; ')
  );
  assert.ok(p.refused, 'the hub refuses to build this launch');
  assert.equal(p.text, '');
});

test('TensorRT-LLM: yaml parameters are shown as the options file, not as fake flags', opts, async () => {
  const mod = await realModule();
  const specs = mod.catalog('trtllm').ok;
  const p = preview(mod, { engine: 'trtllm', ref: hf, specs, values: { kv_cache_dtype: 'fp8' }, host, gpuIds: [0] });
  assert.equal(p.source, 'hub');
  assert.match(p.text, /^# \/config\/extra-llm-api-options\.yaml$/m);
  assert.match(p.text, /dtype: fp8/);
  assert.doesNotMatch(p.text, /yaml:/);
});

test('a validate answer that is not a list, or an engine the module does not know, gives the approximation', () => {
  const specs = [{ key: 'x', type: 'int', flag: '--x' }];
  const args = { engine: 'vllm', ref: hf, specs, values: {}, host, gpuIds: [0] };
  const raised = { validate: () => ({ raised: { kind: 'TypeError', msg: 'bad' } }), render: () => ({ ok: {} }) };
  const p = preview(raised, args);
  assert.equal(p.source, 'approx');
  assert.equal(p.reason, 'TypeError: bad');
  const unknown = {
    validate: () => ({ ok: [] }),
    render: () => ({ problem: { status: 404, code: 'unknown_engine', detail: 'unknown engine: nope' } }),
  };
  const u = preview(unknown, { ...args, engine: 'nope' });
  assert.equal(u.source, 'approx');
  assert.equal(u.refused, null);
  assert.equal(preview({ validate: () => ({ problem: { code: 'unknown_engine' } }) }, args).source, 'approx');
});
