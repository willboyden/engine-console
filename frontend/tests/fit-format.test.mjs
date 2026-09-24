import test from 'node:test';
import assert from 'node:assert/strict';
import { normalizeReport, verdictFor, quickFit, compatLevel, verdictMeta } from '../js/fit-format.js';

test('normalizeReport: per_gpu / gpus aliases, derived total, defaults', () => {
  const r = normalizeReport({ verdict: 'tight', confidence: 'high', per_gpu: [{ gpu_id: 1, weights: 40, kv_cache: 10, activations: 2, cuda_graphs: 1, overhead: 1, budget: 86 }], tp_required: 1,
    max_context_at_current_concurrency: 1000, compat: [{ level: 'warn', code: 'x', message: 'm' }] });
  assert.equal(r.gpus[0].id, 1); assert.equal(r.gpus[0].total, 54); assert.equal(r.tpRequired, 1); assert.equal(r.maxContext, 1000);
  assert.equal(normalizeReport({ gpus: [{ weights: 1 }] }).gpus[0].total, 1);
  assert.equal(normalizeReport(null), null);
  assert.equal(normalizeReport({}).verdict, 'unknown');
});

test('verdictFor thresholds: <=90% fits, 90-100 tight, >100 wont_fit', () => {
  assert.equal(verdictFor(90, 100), 'fits'); assert.equal(verdictFor(90.1, 100), 'tight'); assert.equal(verdictFor(100, 100), 'tight'); assert.equal(verdictFor(100.1, 100), 'wont_fit'); assert.equal(verdictFor(1, 0), 'unknown');
});

test('quickFit picks the smallest GPU count that fits and never fabricates for missing data', () => {
  assert.deepEqual(quickFit(30 * 2 ** 30, [95, 95]), { verdict: 'fits', gpus: 1 });
  assert.deepEqual(quickFit(140 * 2 ** 30, [95, 95]), { verdict: 'fits', gpus: 2 }.verdict === 'fits' ? quickFit(140 * 2 ** 30, [95, 95]) : null);
  assert.equal(quickFit(500 * 2 ** 30, [95, 95]).verdict, 'wont_fit');
  assert.equal(quickFit(undefined, [95]).verdict, 'unknown'); assert.equal(quickFit(10, []).verdict, 'unknown');
});

test('compatLevel worst-wins; verdictMeta falls back', () => {
  assert.equal(compatLevel([{ level: 'ok' }, { level: 'warn' }]), 'warn');
  assert.equal(compatLevel([{ level: 'warn' }, { level: 'block' }, { level: 'warn' }]), 'block');
  assert.equal(compatLevel(undefined), 'ok'); assert.equal(verdictMeta('nope').cls, 'muted');
});

test('normalizeReport accepts the backend *_gib field names', () => {
  const r = normalizeReport({ verdict: 'fits', confidence: 'high', tp: 1, tp_required: 1, concurrency: 4, per_gpu: [{ gpu_id: 0, weights_gib: 30, kv_cache_gib: 10, activations_gib: 1, cuda_graphs_gib: 1, overhead_gib: 1, total_gib: 43, budget_gib: 86, vram_total_gib: 95.6, free_gib: 60, verdict: 'fits' }] });
  const g = r.gpus[0]; assert.equal(g.weights, 30); assert.equal(g.kv_cache, 10); assert.equal(g.total, 43); assert.equal(g.budget, 86); assert.equal(g.capacity, 95.6); assert.equal(r.concurrency, 4);
});

test('normalizeReport carries the host_ram block', () => {
  const r = normalizeReport({ verdict: 'fits', per_gpu: [], host_ram: { needed_gib: 70, available_gib: 54, total_gib: 246, verdict: 'wont_fit', breakdown: { 'CPU offload': 64 }, notes: ['n'] } });
  assert.equal(r.hostRam.verdict, 'wont_fit'); assert.deepEqual(r.hostRam.breakdown, [['CPU offload', 64]]); assert.equal(normalizeReport({ per_gpu: [] }).hostRam, null);
});
