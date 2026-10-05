import test from 'node:test';
import assert from 'node:assert/strict';
import { refFor, refToQuery, refFromQuery, refLabel, refEquals, refBody, MODEL_FORMATS } from '../js/model-ref.js';

const copy = { root_id: 'gguf-shared', rel: 'r/m-Q4.gguf' };

test('each format uses its own fields: repo id, root + path, or the Ollama name', () => {
  assert.deepEqual(refFor({ format: 'hf', name: 'org/m' }), { format: 'hf', repo_id: 'org/m' });
  assert.deepEqual(refFor({ format: 'gguf' }, copy), { format: 'gguf', root_id: 'gguf-shared', rel: 'r/m-Q4.gguf' });
  assert.deepEqual(refFor({ format: 'plain' }, copy), { format: 'plain', root_id: 'gguf-shared', rel: 'r/m-Q4.gguf' });
  assert.deepEqual(refFor({ format: 'ollama', name: 'qwen3:8b' }), { format: 'ollama', name: 'qwen3:8b' });
  assert.equal(refFor({ format: 'gguf' }), null, 'a file needs a copy'); assert.equal(refFor({ format: 'other' }, copy), null); assert.equal(refFor(null), null);
  assert.deepEqual(MODEL_FORMATS, ['hf', 'gguf', 'ollama', 'plain']);
});

test('a ModelRef survives a trip through a hash query; a bare repo is the v1 deep link', () => {
  for (const ref of [{ format: 'hf', repo_id: 'org/m' }, { format: 'gguf', root_id: 'r', rel: 'a/b.gguf' }, { format: 'ollama', name: 'qwen3:8b' }, { format: 'plain', root_id: 'x', rel: 'dir' }]) assert.deepEqual(refFromQuery(refToQuery(ref)), ref);
  assert.deepEqual(refFromQuery({ repo: 'org/m' }), { format: 'hf', repo_id: 'org/m' });
  assert.equal(refFromQuery({}), null); assert.equal(refFromQuery({ mfmt: 'gguf', root: 'r' }), null, 'a file model needs root and path'); assert.equal(refFromQuery({ mfmt: 'zzz', repo: 'a/b' })?.format, 'hf'); assert.deepEqual(refToQuery(null), {});
});

test('labels, equality and request bodies', () => {
  assert.equal(refLabel({ format: 'hf', repo_id: 'org/m' }), 'org/m'); assert.equal(refLabel({ format: 'gguf', root_id: 'r', rel: 'dir/file.gguf' }), 'file.gguf'); assert.equal(refLabel({ format: 'ollama', name: 'a:b' }), 'a:b'); assert.equal(refLabel(null), '');
  assert.equal(refEquals({ format: 'hf', repo_id: 'a' }, { format: 'hf', repo_id: 'a' }), true); assert.equal(refEquals({ format: 'hf', repo_id: 'a' }, { format: 'hf', repo_id: 'b' }), false); assert.equal(refEquals(null, null), true); assert.equal(refEquals(null, {}), false);
  assert.deepEqual(refBody({ format: 'hf', repo_id: 'a/b' }), { repo_id: 'a/b', model: { format: 'hf', repo_id: 'a/b' } });
  assert.deepEqual(refBody({ format: 'ollama', name: 'x:1' }), { model: { format: 'ollama', name: 'x:1' } }); assert.deepEqual(refBody(null), {});
});
