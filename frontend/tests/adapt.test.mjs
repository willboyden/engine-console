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
