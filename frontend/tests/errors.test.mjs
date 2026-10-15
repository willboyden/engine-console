import test from 'node:test';
import assert from 'node:assert/strict';
import { setMessages } from '../js/i18n.js';
import { describeError } from '../js/errors.js';
import { ApiError } from '../js/api.js';
import { openStream } from '../js/sse.js';
import fs from 'node:fs';

setMessages(JSON.parse(fs.readFileSync(new URL('../i18n/en.json', import.meta.url), 'utf8')));

test('describeError maps the hardening statuses to actionable text', () => {
  assert.match(describeError(new ApiError(413, {})), /too large/i);
  assert.match(describeError(new ApiError(415, {})), /JSON or YAML/);
  assert.match(describeError(new ApiError(421, {})), /127\.0\.0\.1|localhost/);
  assert.match(describeError(new ApiError(429, { code: 'too_many_streams' })), /for this key or on the whole console/);
  assert.match(describeError(new ApiError(429, { code: 'rate' })), /Too many requests/);
});
test('secret_in_argv surfaces the server message verbatim; 422 lists field errors', () => {
  assert.equal(describeError(new ApiError(422, { code: 'secret_in_argv', detail: 'remove api_key' })), 'remove api_key');
  const e = new ApiError(422, { code: 'validation_error', detail: 'request validation failed', errors: [{ loc: ['body', 'repo_id'], msg: 'string does not match pattern' }] });
  assert.match(describeError(e), /repo_id: string does not match pattern/);
  assert.equal(describeError(new ApiError(500, { detail: 'boom' })), 'boom');
});
test('too_many_streams (429) is not retried; a plain 429 waits at least 5 s before retrying', async () => {
  let calls = 0;
  await new Promise((resolve) => openStream('/x', { fetchImpl: async () => { calls++; return { ok: false, status: 429, text: async () => '{"code":"too_many_streams","detail":"max 8"}' }; }, sleep: async () => { throw new Error('must not sleep'); }, onClose: resolve }));
  assert.equal(calls, 1);
  const waits = []; let n = 0; let h;
  await new Promise((resolve) => { h = openStream('/x', { sleep: async (ms) => { waits.push(ms); }, onClose: resolve, fetchImpl: async () => { if (++n === 2) { h.close(); } return { ok: false, status: 429, text: async () => '{"code":"rate"}' }; } }); });
  assert.ok(waits[0] >= 5000, `waited ${waits[0]}`);
});

test('409 instance_not_managed maps to a plain explanation', () => {
  assert.match(describeError(new ApiError(409, { code: 'instance_not_managed', detail: 'x' })), /monitor/i);
});

test('contract v2 node errors map to advice a person can act on (ARCHITECTURE section 10.11)', () => {
  const d = (status, code, detail = '') => describeError(new ApiError(status, { code, detail }));
  assert.match(d(503, 'node_unavailable', 'edge-4: link is down'), /not connected[\s\S]*edge-4: link is down/, 'advice first, then the server detail that names the machine');
  assert.match(d(409, 'monitor_only', 'edge-2: monitor mode'), /monitor mode[\s\S]*manage mode/);
  assert.match(d(403, 'policy_denied'), /policy does not allow[\s\S]*administrator/);
  assert.match(d(409, 'in_use'), /in use by a running engine/); assert.match(d(409, 'model_in_use'), /in use by a running engine/);
  assert.match(d(409, 'port_in_use'), /free network port/); assert.match(d(507, 'no_space'), /not enough free space/);
  assert.match(d(504, 'timeout'), /may or may not have happened/); assert.match(d(501, 'unsupported'), /does not support/);
  assert.match(d(502, 'hash_mismatch'), /integrity/); assert.match(d(502, 'docker_error', 'x'), /machine reported a failure/);
  assert.match(d(404, 'unknown_host'), /not in the console's hosts file/); assert.match(d(503, 'busy'), /busy/); assert.match(d(409, 'script_busy'), /busy/);
  assert.match(d(400, 'invalid_arg', "'x' is not allowed"), /request was not valid/); assert.match(d(409, 'preflight_failed', 'fit verdict: wont_fit'), /launch checks failed[\s\S]*wont_fit/);
});

test('the auth, CSRF and permission statuses say what to do', () => {
  const d = (status, code) => describeError(new ApiError(status, { code, detail: '' }));
  assert.match(d(401, 'unauthorized'), /API key/); assert.match(d(403, 'csrf_header_required'), /Reload the page/); assert.match(d(403, 'cross_origin'), /own address/); assert.match(d(403, 'forbidden'), /viewer key is read-only/);
  assert.match(d(409, 'whatever'), /conflicts with the current state/); assert.match(d(503, 'whatever'), /unavailable/); assert.match(d(404, 'whatever'), /no longer exists/);
  assert.match(describeError(new ApiError(0, { code: 'network', detail: 'fetch failed' })), /Cannot reach the console backend[\s\S]*fetch failed/);
});

test('a node timeout is flagged as uncertain so the UI warns the action may have happened', async () => {
  const { isUncertain } = await import('../js/errors.js');
  assert.equal(isUncertain(new ApiError(504, {})), true); assert.equal(isUncertain(new ApiError(500, { code: 'timeout' })), true); assert.equal(isUncertain(new ApiError(503, {})), false); assert.equal(isUncertain(null), false);
});

test('the server detail after the advice starts a new sentence, but ids and identifiers keep their spelling', () => {
  const d = (status, detail) => describeError(new ApiError(status, { detail }));
  assert.match(d(401, 'a valid bearer API key is required'), /\. A valid bearer API key is required$/);
  assert.match(d(401, 'gpu-box-2 refused the key'), /\. gpu-box-2 refused the key$/);
  assert.match(d(401, '401 from the proxy'), /\. 401 from the proxy$/);
  assert.equal(d(500, 'boom'), 'boom');
});

test('only a role refusal (403 forbidden) counts as "this key may not": gated models, policy and CSRF 403s do not', async () => {
  const { isRoleRefusal } = await import('../js/errors.js');
  assert.equal(isRoleRefusal(new ApiError(403, { code: 'forbidden', detail: 'admin role required' })), true);
  for (const code of ['hf_gated', 'egress_denied', 'policy_denied', 'csrf_header_required', 'cross_origin']) assert.equal(isRoleRefusal(new ApiError(403, { code })), false, code);
  assert.equal(isRoleRefusal(new ApiError(401, { code: 'forbidden' })), false); assert.equal(isRoleRefusal(null), false);
});
