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
  assert.match(describeError(new ApiError(429, { code: 'too_many_streams' })), /8/);
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
