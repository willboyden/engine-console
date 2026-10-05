import test from 'node:test';
import assert from 'node:assert/strict';
import { normDownload, normInstance, uptimeOf, normHit, normLocalModel, normPoint, normBench, normKey, normPrompt, parseMessage, toGrid } from '../js/adapt.js';

test('normDownload maps backend fields (state/total_bytes/done_bytes)', () => {
  const d = normDownload({ id: 'd', state: 'completed', total_bytes: 10, done_bytes: 10 });
  assert.equal(d.status, 'completed'); assert.equal(d.bytes_total, 10); assert.equal(d.bytes_done, 10);
  assert.equal(normDownload({ status: 'running', bytes_total: 5, bytes_done: 1 }).bytes_total, 5);
});

test('normInstance maps phase/progress_pct/ttl_idle_s; uptimeOf handles uptime_s, epoch seconds and ISO', () => {
  const i = normInstance({ phase: 'compiling', progress_pct: 40, ttl_idle_s: 60 });
  assert.deepEqual(i.progress, { phase: 'compiling', pct: 40 }); assert.equal(i.ttl_s, 60);
  assert.equal(uptimeOf({ uptime_s: 12 }), 12);
  assert.equal(uptimeOf({ started_at: 1000 }, 1000 * 1000 + 5000), 5);
  assert.equal(uptimeOf({ started_at: '2026-01-01T00:00:00Z' }, Date.parse('2026-01-01T00:00:30Z')), 30);
  assert.equal(uptimeOf({}), null);
});

test('normHit / normLocalModel', () => {
  assert.equal(normHit({ approx_size_gib: 2, quantization: 'fp8' }).size_bytes, 2 * 2 ** 30); assert.equal(normHit({ quantization: 'fp8' }).quant, 'fp8'); assert.equal(normHit({}).size_bytes, undefined);
  assert.deepEqual(normLocalModel({ engines_that_fit: ['vllm'] }).engines_fit, { vllm: 'fits' });
});

test('normPoint flattens {t, values}', () => assert.deepEqual(normPoint({ t: 1, values: { a: 2 } }), { t: 1, a: 2 }));

test('normBench flattens the backend result (single + concurrency levels)', () => {
  const r = normBench({ state: 'completed', results: { single: { decode_tps: 90, ttft_p50_s: 0.1 }, concurrency: [{ concurrency: 1, throughput_tps: 90 }, { concurrency: 4, throughput_tps: 300 }], prefix_cache: { speedup: 5 } } });
  assert.equal(r.status, 'completed'); assert.equal(r.results.decode_tps, 90); assert.deepEqual(r.results.throughput, [{ concurrency: 1, tps: 90 }, { concurrency: 4, tps: 300 }]);
  assert.equal(normBench({ state: 'running', results: {} }).results, null);
  assert.equal(normBench({ state: 'completed', results: { concurrency: [{ concurrency: 4, decode_tps: 50, throughput_tps: 1 }] } }).results.decode_tps, 50);
});

test('normKey / normPrompt', () => {
  assert.equal(normKey({ secret: 's', last_used_at: 5 }).key, 's'); assert.equal(normKey({ last_used_at: 5 }).last_used, 5);
  assert.deepEqual([normPrompt({ title: 'a', content: 'b' }).name, normPrompt({ title: 'a', content: 'b' }).text], ['a', 'b']);
});

test('parseMessage decodes JSON content-part arrays (multimodal turns) and leaves normal text alone', () => {
  const m = parseMessage({ role: 'user', content: JSON.stringify([{ type: 'text', text: 'look' }, { type: 'image_url', image_url: { url: 'data:image/png;base64,AA' } }]) });
  assert.equal(m.text, 'look'); assert.deepEqual(m.images, ['data:image/png;base64,AA']);
  assert.equal(parseMessage({ role: 'assistant', content: '[{not json', reasoning: 'r' }).text, '[{not json');
  assert.equal(parseMessage({ role: 'assistant', content: 'hi', reasoning: null }).reasoning, '');
});

test('toGrid folds ISO hour keys (Mon=0) and {dow,hour} rows', () => {
  const g = toGrid([{ key: '2026-09-21T13:00Z', requests: 4 }, { key: '2026-09-21T13:00Z', requests: 1 }, { key: 'garbage', requests: 9 }], { utc: true }); // 2026-09-21 is a Monday
  assert.equal(g[0][13], 5); assert.equal(g.flat().reduce((a, b) => a + b), 5);
  assert.equal(toGrid([{ dow: 6, hour: 23, requests: 2 }, { dow: 9, hour: 1, requests: 1 }])[6][23], 2);
});

import { isExternal, stateReason, listSignature } from '../js/adapt.js';
test('external instances: managed=false / source=external are detected; managed defaults to true', () => {
  assert.equal(isExternal({ managed: false }), true); assert.equal(isExternal({ source: 'external' }), true); assert.equal(isExternal({ managed: true, source: 'console' }), false); assert.equal(isExternal({}), false);
  const e = normInstance({ managed: false, served_models: undefined }); assert.equal(e.external, true); assert.deepEqual(e.served_models, []);
  assert.equal(normInstance({ id: 'a' }).managed, true); assert.equal(normInstance({ id: 'a' }).external, false);
});
test('stateReason prefers reason, falls back to error', () => {
  assert.equal(stateReason({ reason: 'r', error: 'e' }), 'r'); assert.equal(stateReason({ error: 'e' }), 'e'); assert.equal(stateReason({}), null);
});
test('listSignature is stable for identical data and changes on visible fields only', () => {
  const a = [{ id: '1', state: 'ready', uptime_s: 5 }], b = [{ id: '1', state: 'ready', uptime_s: 99 }];
  assert.equal(listSignature(a), listSignature(b));
  assert.notEqual(listSignature(a), listSignature([{ id: '1', state: 'unreachable', reason: 'x' }]));
  assert.notEqual(listSignature(a), listSignature([{ id: '1', state: 'ready', served_models: ['m'] }]));
});

import { normHost, normInstance as normInst2, listSignature as sig2, isExternal as isExt2 } from '../js/adapt.js';
test('v2 instance fields default for a v1 backend: one machine, the class implied by `managed`, no VRAM detail', () => {
  const v1 = normInst2({ id: 'a', managed: true }); assert.deepEqual([v1.host_id, v1.management, v1.vram, v1.script, v1.model_location], ['local', 'managed', [], null, null]);
  assert.equal(normInst2({ id: 'b', managed: false }).management, 'external'); assert.equal(normInst2({ id: 'c', source: 'external' }).management, 'external');
  const v2 = normInst2({ id: 'd', management: 'script', script: 's', host_id: 'edge-1', host_label: 'Edge', vram: [{ gpu_index: 0, used_gib: 1 }], managed: false });
  assert.deepEqual([v2.management, v2.script, v2.host_id, v2.host_label, v2.vram.length], ['script', 's', 'edge-1', 'Edge', 1]); assert.equal(isExt2(v2), true);
  assert.equal(normInst2({ id: 'e', management: 'bogus' }).management, 'managed');
});

test('a host is normalised so views never branch on undefined lists', () => {
  const h = normHost({ id: 'x', label: 'X', remote: 1 });
  assert.deepEqual([h.remote, h.gpus, h.alerts, h.disks, h.roots, h.images, h.scripts, h.addresses], [true, [], [], [], [], [], [], []]);
  assert.equal(h.link.state, 'down'); assert.deepEqual(h.counts, { engines_running: 0, engines_total: 0, models: 0, models_bytes: 0 }); assert.equal(normHost(null), null);
});

test('downloads gain where they land and whether they are copies', () => {
  const d = normDownload({ id: 'd', state: 'running', total_bytes: 10, done_bytes: 5 });
  assert.deepEqual([d.host_id, d.root_id, d.kind, d.source_copy_id, d.status, d.bytes_total], ['local', null, 'hf', null, 'running', 10]);
  const c = normDownload({ id: 'c', kind: 'copy', source_copy_id: 'abc', host_id: 'edge-1', root_id: 'hf-main', state: 'queued' }); assert.deepEqual([c.kind, c.host_id, c.root_id, c.source_copy_id], ['copy', 'edge-1', 'hf-main', 'abc']);
  assert.equal(normDownload({ kind: 'weird' }).kind, 'hf');
});

test('the list signature notices machine, management, controls and VRAM changes (so polling repaints when they change)', () => {
  const base = { id: 'a', state: 'ready', host_id: 'local', management: 'managed', controls: { start: false, stop: true }, vram: [{ used_gib: 10.2 }] };
  assert.notEqual(sig2([base]), sig2([{ ...base, host_id: 'edge-1' }])); assert.notEqual(sig2([base]), sig2([{ ...base, management: 'script' }]));
  assert.notEqual(sig2([base]), sig2([{ ...base, controls: { start: true, stop: true } }])); assert.notEqual(sig2([base]), sig2([{ ...base, vram: [{ used_gib: 40 }] }]));
  assert.equal(sig2([base]), sig2([{ ...base, vram: [{ used_gib: 10.4 }] }]), 'sub-GiB jitter does not repaint');
});
