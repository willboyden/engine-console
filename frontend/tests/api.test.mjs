import test from 'node:test';
import assert from 'node:assert/strict';
import { createClient, ApiError, buildQuery } from '../js/api.js';

const resp = (status, body, ct = 'application/json') => ({ status, ok: status < 400, headers: { get: () => ct }, json: async () => body, text: async () => String(body) });

test('buildQuery skips empty values', () => assert.equal(buildQuery({ a: 1, b: '', c: null, d: 'x y' }), '?a=1&d=x+y'));

test('sends bearer key and JSON body', async () => {
  let call;
  const c = createClient({ getKey: () => 'k123', fetchImpl: async (u, o) => { call = { u, o }; return resp(200, { ok: 1 }); } });
  await c.post('/fit', { a: 1 });
  assert.equal(call.u, '/api/v1/fit'); assert.equal(call.o.headers.Authorization, 'Bearer k123'); assert.equal(call.o.body, '{"a":1}'); assert.equal(call.o.method, 'POST');
});

test('no Authorization header without a key', async () => {
  let h; const c = createClient({ fetchImpl: async (u, o) => { h = o.headers; return resp(200, {}); } });
  await c.get('/health'); assert.equal(h.Authorization, undefined);
});

test('401 triggers onUnauthorized and throws ApiError with problem fields', async () => {
  let fired = 0;
  const c = createClient({ onUnauthorized: () => fired++, fetchImpl: async () => resp(401, { title: 'Unauthorized', detail: 'bad key', code: 'unauthorized' }) });
  await assert.rejects(c.get('/x'), (e) => e instanceof ApiError && e.status === 401 && e.code === 'unauthorized' && e.message === 'bad key');
  assert.equal(fired, 1);
});

test('non-JSON error bodies still produce ApiError', async () => {
  const c = createClient({ fetchImpl: async () => ({ status: 502, ok: false, headers: { get: () => 'text/html' }, json: async () => { throw new Error('no'); } }) });
  await assert.rejects(c.get('/x'), (e) => e.status === 502 && /502/.test(e.message));
});

test('network failure becomes ApiError(0)', async () => {
  const c = createClient({ fetchImpl: async () => { throw new TypeError('fetch failed'); } });
  await assert.rejects(c.get('/x'), (e) => e.status === 0 && e.code === 'network');
});

test('204 resolves null; repo ids keep their slash but are segment-encoded', async () => {
  let u; const c = createClient({ fetchImpl: async (url) => { u = url; return { status: 204, ok: true, headers: { get: () => '' } }; } });
  assert.equal(await c.deleteModel('org/we ird'), null); assert.equal(u, '/api/v1/models/org/we%20ird');
});

test('search passes filters', async () => {
  let u; const c = createClient({ fetchImpl: async (url) => { u = url; return resp(200, { items: [] }); } });
  await c.hfSearch('qwen', { quant: 'fp8', gated: '' }); assert.equal(u, '/api/v1/hf/search?q=qwen&quant=fp8');
});

test('CSRF header X-Engine-Console is sent on every non-GET/HEAD request and never on GET', async () => {
  const seen = [];
  const c = createClient({ fetchImpl: async (u, o) => { seen.push([o.method, o.headers['X-Engine-Console']]); return resp(200, {}); } });
  await c.get('/x'); await c.post('/x', {}); await c.put('/x', {}); await c.patch('/x', {}); await c.del('/x'); await c.importProfile('name: a');
  assert.deepEqual(seen, [['GET', undefined], ['POST', '1'], ['PUT', '1'], ['PATCH', '1'], ['DELETE', '1'], ['POST', '1']]);
});

test('importProfile sends JSON {yaml}', async () => {
  let o; const c = createClient({ fetchImpl: async (u, opts) => { o = { u, opts }; return resp(201, {}); } });
  await c.importProfile('name: x\n'); assert.equal(o.u, '/api/v1/profiles/import'); assert.equal(o.opts.headers['Content-Type'], 'application/json'); assert.equal(o.opts.body, '{"yaml":"name: x\\n"}');
});

test('listAll follows next_cursor and passes bare arrays through', async () => {
  const urls = [];
  const c = createClient({ fetchImpl: async (u) => { urls.push(u); return u.includes('cursor=2') ? resp(200, { items: [3], next_cursor: null }) : resp(200, { items: [1, 2], next_cursor: '2' }); } });
  assert.deepEqual(await c.listAll('/things'), [1, 2, 3]); assert.equal(urls.length, 2);
  const bare = createClient({ fetchImpl: async () => resp(200, [9]) });
  assert.deepEqual(await bare.listAll('/x'), [9]);
});

test('new v1.1 endpoints hit the right paths', async () => {
  const urls = []; const c = createClient({ fetchImpl: async (u, o) => { urls.push(`${o.method} ${u}`); return resp(200, {}); } });
  await c.preflight({}); await c.download('d1'); await c.conversation('c1'); await c.patchInstance('i1', { pinned: true });
  assert.deepEqual(urls, ['POST /api/v1/instances/preflight', 'GET /api/v1/downloads/d1', 'GET /api/v1/conversations/c1', 'PATCH /api/v1/instances/i1']);
});
