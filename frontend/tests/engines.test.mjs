import test from 'node:test';
import assert from 'node:assert/strict';
import { enginesForFormat, imageAllowed, defaultImage, engineUsable, engineMatchesImage, imageChoices, pickEngine, sortEngines, formatsOf } from '../js/engines.js';
import { engineMeta } from '../dev/mock-catalogs.mjs';

const engines = engineMeta.map((e) => ({ ...e }));
const host = (o = {}) => ({ id: 'h', label: 'Box', node: { arch: 'x86_64' }, policy: { engine_images: ['vllm/vllm-openai:*', 'lmsysorg/sglang:*', 'ghcr.io/ggml-org/llama.cpp:*'] }, images: [], ...o });

test('engines are offered per model format, in the usual order', () => {
  assert.deepEqual(enginesForFormat(engines, 'hf').map((e) => e.id), ['vllm', 'sglang', 'trtllm']);
  assert.deepEqual(enginesForFormat(engines, 'gguf').map((e) => e.id), ['llamacpp']); assert.deepEqual(enginesForFormat(engines, 'ollama').map((e) => e.id), ['ollama']);
  assert.deepEqual(enginesForFormat(engines, 'plain').map((e) => e.id), ['vllm', 'sglang', 'trtllm']); assert.deepEqual(enginesForFormat(engines, 'other'), []);
  assert.deepEqual(enginesForFormat([{ id: 'old' }], 'hf').map((e) => e.id), ['old'], 'a v1 engine without model_formats loads hf'); assert.deepEqual(formatsOf({}), ['hf']);
  assert.deepEqual(sortEngines([{ id: 'zzz' }, { id: 'ollama' }, { id: 'vllm' }]).map((e) => e.id), ['vllm', 'ollama', 'zzz']);
});

test('policy image patterns: exact, or a trailing * for any suffix; nothing allowed by default', () => {
  assert.equal(imageAllowed(['a/b:1'], 'a/b:1'), true); assert.equal(imageAllowed(['a/b:1'], 'a/b:2'), false); assert.equal(imageAllowed(['a/b:*'], 'a/b:v9-x'), true); assert.equal(imageAllowed(['a/b:*'], 'a/c:1'), false);
  assert.equal(imageAllowed([], 'a/b:1'), false); assert.equal(imageAllowed(undefined, 'a'), false); assert.equal(imageAllowed(['a/b:*'], null), false);
  assert.equal(imageAllowed(['a/b:*-arm'], 'a/b:1-arm'), true, 'a * may sit anywhere'); assert.equal(imageAllowed(['a/b:*-arm'], 'a/b:1-x86'), false); assert.equal(imageAllowed(['a.b/c:*'], 'aXb/c:1'), false, 'dots are literal');
});

test('the default image depends on the machine architecture', () => {
  const v = engines.find((e) => e.id === 'vllm');
  assert.equal(defaultImage(v, 'x86_64'), 'vllm/vllm-openai:v0.27.1'); assert.equal(defaultImage(v, 'aarch64'), 'vllm/vllm-openai:v0.27.1-aarch64'); assert.equal(defaultImage(v, undefined), 'vllm/vllm-openai:v0.27.1');
});

test('an engine is usable only where the policy allows its image; the reason says which machine and engine', () => {
  const v = engines.find((e) => e.id === 'vllm'), t = engines.find((e) => e.id === 'trtllm');
  assert.deepEqual(engineUsable(v, host()), { ok: true, reason: null });
  const no = engineUsable(t, host()); assert.equal(no.ok, false); assert.equal(no.reason.key, 'engines.image_not_allowed'); assert.deepEqual(no.reason.vars, { engine: 'TensorRT-LLM', host: 'Box' });
  assert.equal(engineUsable(v, host({ policy: { engine_images: [] } })).reason.key, 'engines.no_images_allowed');
  assert.equal(engineUsable(v, host({ node: { arch: 'aarch64' }, policy: { engine_images: ['vllm/vllm-openai:*-aarch64'] } })).ok, true);
  assert.equal(engineUsable(v, host({ node: { arch: 'aarch64' }, policy: { engine_images: ['vllm/vllm-openai:*-aarch64'] }, images: [] })).ok, true);
  const present = host({ policy: { engine_images: ['vllm/vllm-openai:v9*'] }, images: [{ ref: 'vllm/vllm-openai:v9.1' }] });
  assert.equal(engineUsable(v, present).ok, true, 'an allowed image that is present makes the engine usable even when the pinned default is not allowed');
});

test('images offered for an engine on a machine: those it has, that match the engine and that the policy allows, newest first', () => {
  const v = engines.find((e) => e.id === 'vllm');
  const h = host({ images: [{ ref: 'vllm/vllm-openai:v0.27.1', created: 1 }, { ref: 'vllm/vllm-openai:v0.28.0', created: 5 }, { ref: 'lmsysorg/sglang:v0.6.0', created: 9 }, { ref: 'vllm/vllm-openai:v0.22', created: 0 }], policy: { engine_images: ['vllm/vllm-openai:v0.27*', 'vllm/vllm-openai:v0.22'] } });
  assert.deepEqual(imageChoices(v, h).map((i) => i.ref), ['vllm/vllm-openai:v0.27.1', 'vllm/vllm-openai:v0.22']);
  assert.equal(engineMatchesImage(v, 'vllm/vllm-openai:anything'), true); assert.equal(engineMatchesImage(v, 'lmsysorg/sglang:v0.6.0'), false);
});

test('the first engine that loads the format AND that the machine may run is picked', () => {
  assert.equal(pickEngine(engines, 'hf', host()).id, 'vllm');
  assert.equal(pickEngine(engines, 'hf', host({ policy: { engine_images: ['lmsysorg/sglang:*'] } })).id, 'sglang');
  assert.equal(pickEngine(engines, 'gguf', host()).id, 'llamacpp'); assert.equal(pickEngine(engines, 'ollama', host()), null, 'not allowed on this machine');
});
