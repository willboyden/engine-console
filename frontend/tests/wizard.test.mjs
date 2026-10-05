import test from 'node:test';
import assert from 'node:assert/strict';
import { STEPS, GOALS, initialState, reduce, canAdvance, furthestStep, stepIndex, tpParamKey, pickGpus, fitWord, fitReason, gpusNeeded } from '../js/wizard.js';
import { nonDefaultSettings, formatValue } from '../js/param-form.js';

const m1 = { key: 'hf:a/b', name: 'a/b', format: 'hf' }, m2 = { key: 'hf:c/d', name: 'c/d', format: 'hf' };
const ref = { format: 'hf', repo_id: 'a/b' };
const host = (s, id = 'local', extra = {}) => reduce(s, { type: 'host', hostId: id, ref, gpuIds: [0], baseParams: {}, engine: 'vllm', ...extra });

test('steps are model, machine, goal, review; the launch is the action after the last', () => {
  assert.deepEqual(STEPS, ['model', 'machine', 'goal', 'review']);
  assert.ok(GOALS.includes('low-latency') && GOALS.includes('max-throughput') && GOALS.includes('long-context') && GOALS.includes('tool-agent') && GOALS.includes('balanced'));
});

test('a fresh wizard starts at the model step and cannot advance without a model', () => {
  const s = initialState();
  assert.equal(s.step, 'model'); assert.deepEqual(canAdvance(s), { ok: false, reason: 'wizard.need_model' }); assert.equal(furthestStep(s), 0);
  assert.equal(reduce(s, { type: 'next' }), s, 'next is a no-op while the step is incomplete');
  assert.equal(reduce(s, { type: 'goto', step: 'review' }).step, 'model', 'cannot jump ahead');
});

test('choosing a model, a machine and a goal walks the steps and carries the machine-specific choices', () => {
  let s = reduce(initialState(), { type: 'model', model: m1 });
  assert.equal(s.step, 'machine'); assert.equal(s.model, m1); assert.equal(furthestStep(s), 1);
  s = host(s, 'edge-1', { gpuIds: [0, 1], baseParams: { tensor_parallel_size: 2 }, engine: 'sglang' });
  assert.equal(s.step, 'goal'); assert.equal(s.hostId, 'edge-1'); assert.deepEqual(s.ref, ref); assert.deepEqual(s.gpuIds, [0, 1]); assert.equal(s.engine, 'sglang');
  s = reduce(s, { type: 'goal', goal: 'long-context', params: { max_model_len: 131072 } });
  assert.equal(s.step, 'review'); assert.deepEqual(s.params, { tensor_parallel_size: 2, max_model_len: 131072 }, 'the preset is applied on top of what the fit needs');
  assert.equal(furthestStep(s), 3); assert.equal(stepIndex(s), 3);
});

test('going back keeps every choice; the stepper can jump back or to the furthest reached step, never past it', () => {
  let s = reduce(initialState(), { type: 'model', model: m1 }); s = host(s); s = reduce(s, { type: 'goal', goal: 'balanced', params: { a: 1 } });
  s = reduce(s, { type: 'back' }); assert.equal(s.step, 'goal'); assert.equal(s.goal, 'balanced');
  s = reduce(s, { type: 'goto', step: 'model' }); assert.equal(s.step, 'model'); assert.equal(s.model, m1);
  s = reduce(s, { type: 'goto', step: 'review' }); assert.equal(s.step, 'review', 'review is reachable because a goal was chosen');
  assert.equal(reduce(initialState(), { type: 'back' }).step, 'model'); assert.equal(reduce(s, { type: 'goto', step: 'nonsense' }), s);
});

test('a different model clears what was chosen after it; choosing the same model again keeps it', () => {
  let s = reduce(initialState(), { type: 'model', model: m1 }); s = host(s); s = reduce(s, { type: 'goal', goal: 'balanced', params: { a: 1 } });
  const same = reduce(s, { type: 'model', model: m1 });
  assert.equal(same.hostId, 'local'); assert.equal(same.goal, 'balanced'); assert.equal(same.step, 'machine');
  const other = reduce(s, { type: 'model', model: m2 });
  assert.deepEqual([other.hostId, other.goal, other.engine, other.ref, other.gpuIds, other.params, other.baseParams], [null, null, null, null, [], {}, {}]); assert.equal(other.model, m2);
});

test('a different machine clears the goal and settings (the fit and engine depend on it); the same machine keeps them', () => {
  let s = reduce(initialState(), { type: 'model', model: m1 }); s = host(s); s = reduce(s, { type: 'goal', goal: 'balanced', params: { a: 1 } });
  const moved = host(s, 'gpu-box-2'); assert.equal(moved.goal, null); assert.deepEqual(moved.params, {}); assert.equal(moved.step, 'goal');
  const same = host(s, 'local'); assert.equal(same.hostId, 'local'); assert.equal(same.step, 'goal');
});

test('changing the engine resets the goal; name and image are plain fields; reset starts over', () => {
  let s = reduce(initialState(), { type: 'model', model: m1 }); s = host(s); s = reduce(s, { type: 'goal', goal: 'balanced', params: { a: 1 } });
  const e = reduce(s, { type: 'engine', engine: 'sglang' }); assert.equal(e.engine, 'sglang'); assert.equal(e.goal, null); assert.deepEqual(e.params, {});
  assert.equal(reduce(s, { type: 'engine', engine: 'vllm' }), s, 'same engine is a no-op');
  assert.equal(reduce(s, { type: 'name', name: 'mine' }).name, 'mine'); assert.equal(reduce(s, { type: 'image', image: 'x:1' }).image, 'x:1');
  assert.equal(reduce(s, { type: 'reset' }).step, 'model'); assert.equal(reduce(s, { type: 'bogus' }), s);
});

test('prefill from a deep link lands on the right step', () => {
  assert.equal(initialState({ model: m1 }).step, 'machine'); assert.equal(initialState({ model: m1, hostId: 'local', ref }).step, 'goal'); assert.equal(initialState({ hostId: 'local' }).step, 'model');
});

test('the tensor-parallel parameter is found by name across engines; the freest GPUs are picked', () => {
  assert.equal(tpParamKey([{ key: 'max_model_len' }, { key: 'tensor_parallel_size' }]), 'tensor_parallel_size'); assert.equal(tpParamKey([{ key: 'tp_size' }]), 'tp_size'); assert.equal(tpParamKey([{ key: 'ctx_size' }]), null); assert.equal(tpParamKey(undefined), null);
  const gpus = [{ index: 0, free_gib: 30 }, { index: 1, free_gib: 80 }, { index: 2, free_gib: 80 }, { index: 3, free_gib: 10 }];
  assert.deepEqual(pickGpus(gpus, 1), [1]); assert.deepEqual(pickGpus(gpus, 2), [1, 2]); assert.deepEqual(pickGpus(gpus, 9), [0, 1, 2, 3]); assert.deepEqual(pickGpus([{ index: 0, total_gib: 119 }], 1), [0], 'unified-memory GPUs have no free figure'); assert.deepEqual(pickGpus([], 1), []);
});

test('verdicts become plain words; the reason names the numbers', () => {
  assert.deepEqual(['fits', 'tight', 'wont_fit', 'unknown', 'x'].map(fitWord), ['wizard.fit_easy', 'wizard.fit_tight', 'wizard.fit_wont', 'wizard.fit_unknown', 'wizard.fit_unknown']);
  const g = (total, budget) => ({ gpu_id: 0, weights_gib: total - 4, kv_cache_gib: 1, activations_gib: 1, cuda_graphs_gib: 1, overhead_gib: 1, total_gib: total, budget_gib: budget });
  assert.deepEqual(fitReason({ verdict: 'fits', per_gpu: [g(40, 86)] }), { key: 'wizard.reason_fits', vars: { need: '40.0', have: '86.0', pct: 47 } });
  assert.equal(fitReason({ verdict: 'tight', per_gpu: [g(80, 86)] }).key, 'wizard.reason_tight');
  assert.equal(fitReason({ verdict: 'wont_fit', per_gpu: [g(120, 86)], tp_required: 1 }).key, 'wizard.reason_wont');
  const tp = fitReason({ verdict: 'wont_fit', per_gpu: [g(120, 86)], tp_required: 2 }); assert.equal(tp.key, 'wizard.reason_wont_tp'); assert.equal(tp.vars.n, 2);
  assert.deepEqual(fitReason({ verdict: 'wont_fit', per_gpu: [g(1, 2)], compat: [{ level: 'block', code: 'x', message: 'Not on this GPU.' }] }), { key: 'wizard.reason_blocked', vars: { message: 'Not on this GPU.' } });
  assert.equal(fitReason({ verdict: 'unknown', per_gpu: [] }).key, 'wizard.reason_unknown'); assert.equal(fitReason(null).key, 'wizard.reason_unknown');
  assert.equal(gpusNeeded({ per_gpu: [], tp_required: 4 }, 2), 2, 'capped by what the machine has'); assert.equal(gpusNeeded({ per_gpu: [], tp_required: null }, 2), 1);
});

test('the review lists every non-default setting with a readable value, skipping defaults and redacted secrets', () => {
  const specs = [{ key: 'a', label: 'A', default: 4, unit: 'tokens' }, { key: 'b', label: 'B', default: true }, { key: 'c', label: 'C', default: null }, { key: 'tok', label: 'Token', default: null }, { key: 'j', label: 'J', default: null }];
  const r = nonDefaultSettings(specs, { a: 8, b: true, c: 'x', tok: '[set]', j: { n: 1 }, gone: 1 });
  assert.deepEqual(r.map((x) => [x.key, x.text]), [['a', '8 tokens'], ['c', 'x'], ['j', '{"n":1}']]);
  assert.equal(formatValue({}, true), 'on'); assert.equal(formatValue({}, false), 'off'); assert.equal(formatValue({ unit: 'GiB' }, 16), '16 GiB');
});
