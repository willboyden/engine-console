import test from 'node:test';
import assert from 'node:assert/strict';
import { widths, systemSegments, hostSegments, engineMemory, normSystemMemory, normHostRam, MIN_PCT } from '../js/memory.js';

const sum = (a) => a.reduce((x, y) => x + y, 0);

test('system segments: engines + other sum to used, then available; sorted by size', () => {
  const { segments, usedSum } = systemSegments({ total: 246, used: 192, available: 54, engines: [{ id: 'v', name: 'vllm', gib: 34 }, { id: 's', name: 'sglang', gib: 68.6 }] });
  assert.deepEqual(segments.map((s) => s.key), ['s', 'v', 'other', 'available']);
  const usedGib = sum(segments.filter((s) => s.kind !== 'available').map((s) => s.gib));
  assert.ok(Math.abs(usedGib - 192) < 1e-9); assert.ok(Math.abs(usedSum - 192) < 1e-9);
  assert.ok(Math.abs(sum(segments.map((s) => s.gib)) - 246) < 1e-9);
  assert.ok(Math.abs(sum(segments.map((s) => s.pct)) - 100) < 1e-6);
  assert.notEqual(segments[0].cls, segments[1].cls);
});
test('engines exceeding "used" never yield a negative other segment', () => {
  const { segments } = systemSegments({ total: 100, used: 50, available: 50, engines: [{ id: 'a', name: 'a', gib: 60 }] });
  assert.ok(!segments.some((s) => s.key === 'other')); assert.ok(segments.every((s) => s.gib > 0));
});
test('tiny nonzero segments stay visible and widths never exceed 100%', () => {
  const w = widths([200, 0.4, 0, 45], 246);
  assert.ok(w[1] >= MIN_PCT - 1e-9); assert.equal(w[2], 0); assert.ok(sum(w) <= 100 + 1e-6);
  const s = systemSegments({ total: 246, used: 100.2, available: 145.8, engines: [{ id: 'x', name: 'x', gib: 0.2 }] });
  assert.ok(s.segments.find((x) => x.key === 'x').pct >= MIN_PCT - 1e-9);
});
test('nulls, zero totals and empty input are safe', () => {
  assert.deepEqual(widths([1, 2], 0), [0, 0]);
  assert.deepEqual(systemSegments({ engines: [], total: null, used: null, available: null }).segments, []);
  assert.deepEqual(engineMemory([{ id: 'a', host_memory: null }, { id: 'b' }, { id: 'c', name: 'C', host_memory: { total_gib: 3 } }]), [{ id: 'c', name: 'C', gib: 3 }]);
  assert.equal(hostSegments(null), null); assert.equal(normSystemMemory(null), null); assert.equal(normHostRam(undefined), null);
  assert.deepEqual(hostSegments({ total_gib: 0, anon_gib: 0 }).segments, []);
});
test('hostSegments: anon/cache/shm(+kernel) sum to the parts, rss is flagged approx', () => {
  const h = hostSegments({ total_gib: 68.6, anon_gib: 3.3, cache_gib: 0.5, shmem_gib: 64, kernel_gib: 0.8, source: 'cgroup' });
  assert.deepEqual(h.segments.map((s) => s.key), ['anon', 'cache', 'shm', 'kernel']); assert.equal(h.approx, false);
  assert.ok(Math.abs(sum(h.segments.map((s) => s.gib)) - 68.6) < 1e-9); assert.ok(Math.abs(sum(h.segments.map((s) => s.pct)) - 100) < 1e-6);
  assert.ok(h.segments.find((s) => s.key === 'cache').pct >= MIN_PCT - 1e-9);
  assert.equal(hostSegments({ total_gib: 5, anon_gib: 5, source: 'rss' }).approx, true);
});
test('normalizers map contract field names', () => {
  const m = normSystemMemory({ total_gib: 246, used_gib: 192, available_gib: 54, free_gib: 8, cached_gib: 100, shmem_gib: 98, swap_total_gib: 32, swap_used_gib: 24, updated_at: 5 });
  assert.deepEqual([m.total, m.used, m.available, m.shmem, m.swapUsed, m.swapTotal], [246, 192, 54, 98, 24, 32]);
  const h = normHostRam({ needed_gib: 70, available_gib: 54, total_gib: 246, verdict: 'wont_fit', breakdown: { 'CPU offload': 64, x: null }, notes: ['n'] });
  assert.equal(h.verdict, 'wont_fit'); assert.deepEqual(h.breakdown, [['CPU offload', 64]]); assert.deepEqual(h.notes, ['n']);
});
