import test, { mock } from 'node:test';
import assert from 'node:assert/strict';
import { normalizeBundle, normExplain, explainFor, paramExplain, popoverModel, termModel, termChips, glossaryEntries, searchGlossary, glossaryLetters, loadLearn, setBundle, resetLearn, getBundle, isLoaded,
  missingKeys, lookup, setDevHooks, expectedMissing, paragraphs, LIMITS, emptyBundle } from '../js/learn.js';
import { learnBundle } from '../dev/mock-learn.mjs';
import { catalogs } from '../dev/mock-catalogs.mjs';
import { CONCEPT_FAMILIES, METRIC_KEYS, allConceptKeys } from '../js/learn-keys.js';

const E = (summary, o = {}) => ({ summary, details: '', when_to_change: '', risks: '', example: '', terms: [], level: 'advanced', ...o });
const bundle = () => ({ version: 1,
  glossary: { vram: { title: 'VRAM', short: 'GPU memory.', long: 'First paragraph.\n\nSecond paragraph.', see_also: ['gpu', 'ghost'] }, gpu: { title: 'GPU', short: 'The graphics card.', long: '', see_also: [] }, kv_cache: { title: 'KV cache', short: 'Working memory.', long: 'It holds vram contents.', see_also: [] } },
  metrics: { generation_tps: E('Tokens per second.', { terms: ['vram'] }) },
  concepts: { 'state.ready': E('Ready to answer.', { level: 'basic', details: 'More.', when_to_change: 'Never.', risks: 'None.', example: 'e.g.', terms: ['gpu', 'nope'] }), 'field.gpu_pick': E('Pick GPUs.') },
  params: { vllm: { max_model_len: E('Context length.') } }, presets: { vllm: { balanced: E('Everyday.') } } });

test('normalizeBundle tolerates missing sections, wrong types and non-objects', () => {
  assert.deepEqual(normalizeBundle(null), emptyBundle()); assert.deepEqual(normalizeBundle([]), emptyBundle()); assert.deepEqual(normalizeBundle('x'), emptyBundle());
  const b = normalizeBundle({ version: 'two', glossary: [], metrics: 5, concepts: { a: E('x') }, params: { vllm: 'bad', sglang: { k: E('y') } }, presets: null });
  assert.equal(b.version, 1); assert.deepEqual(b.glossary, {}); assert.deepEqual(b.metrics, {}); assert.deepEqual(b.params.vllm, {}); assert.ok(b.params.sglang.k); assert.deepEqual(b.presets, {});
});

test('normExplain needs a summary, coerces level, drops junk terms and trims text', () => {
  assert.equal(normExplain(null), null); assert.equal(normExplain({ summary: '   ' }), null); assert.equal(normExplain({ details: 'no summary' }), null);
  const e = normExplain({ summary: ' Hi ', level: 'wrong', terms: ['a', 3, '', 'b'], details: null });
  assert.equal(e.summary, 'Hi'); assert.equal(e.level, 'advanced'); assert.deepEqual(e.terms, ['a', 'b']); assert.equal(e.details, '');
  assert.equal(normExplain({ summary: 's', level: 'basic' }).level, 'basic');
});

test('explainFor finds each kind, falls back (metric -> concept) and returns null for the unknown', () => {
  const b = normalizeBundle(bundle());
  assert.equal(explainFor({ kind: 'concept', key: 'state.ready' }, b).summary, 'Ready to answer.');
  assert.equal(explainFor({ kind: 'metric', key: 'generation_tps' }, b).summary, 'Tokens per second.');
  assert.equal(explainFor({ kind: 'metric', key: 'state.ready' }, b).level, 'basic', 'a metric key may live in concepts');
  assert.equal(explainFor({ kind: 'param', key: 'max_model_len', engine: 'vllm' }, b).summary, 'Context length.');
  assert.equal(explainFor({ kind: 'param', key: 'max_model_len', engine: 'sglang' }, b), null);
  assert.equal(explainFor({ kind: 'preset', key: 'balanced', engine: 'vllm' }, b).summary, 'Everyday.');
  const term = explainFor({ kind: 'term', key: 'vram' }, b);
  assert.equal(term.summary, 'GPU memory.'); assert.deepEqual(term.terms, ['gpu', 'ghost']);
  assert.equal(explainFor({ kind: 'concept', key: 'nope' }, b), null); assert.equal(explainFor({}, b), null); assert.equal(explainFor(undefined, b), null);
});

test('concept keys accept the likely spellings (field.gpus finds field.gpu_pick)', () => {
  const b = normalizeBundle(bundle());
  assert.equal(explainFor({ kind: 'concept', key: 'field.gpus' }, b).summary, 'Pick GPUs.');
});

test('paramExplain prefers the catalog spec over the bundle', () => {
  const b = normalizeBundle(bundle());
  assert.equal(paramExplain({ key: 'max_model_len', explain: E('From the catalog.') }, 'vllm', b).summary, 'From the catalog.');
  assert.equal(paramExplain({ key: 'max_model_len', explain: null }, 'vllm', b).summary, 'Context length.');
  assert.equal(paramExplain({ key: 'zzz' }, 'vllm', b), null);
});

test('popoverModel builds title, summary, non-empty sections in order, and resolves only existing glossary terms', () => {
  const b = normalizeBundle(bundle());
  const m = popoverModel(explainFor({ kind: 'concept', key: 'state.ready' }, b), { title: 'ready', bundle: b });
  assert.equal(m.title, 'ready'); assert.equal(m.summary, 'Ready to answer.');
  assert.deepEqual(m.sections.map((s) => s.id), ['details', 'when', 'risks', 'example']);
  assert.equal(m.sections[0].label, null, 'the details paragraph has no heading'); assert.equal(m.sections[1].label, 'learn.when_to_change'); assert.equal(m.sections[2].label, 'learn.what_can_go_wrong');
  assert.deepEqual(m.terms, [{ key: 'gpu', title: 'GPU' }], 'unknown term "nope" is dropped');
  const bare = popoverModel(E('Only a summary.'), { title: 't', bundle: b });
  assert.deepEqual(bare.sections, []); assert.deepEqual(bare.terms, []); assert.equal(popoverModel(null), null);
});

test('popover text is data, never markup: hostile text passes through untouched for textContent rendering', () => {
  const hostile = '<img src=x onerror=alert(1)><script>alert(2)</script>';
  const m = popoverModel(E(hostile, { details: hostile }), { title: hostile, bundle: emptyBundle() });
  assert.equal(m.summary, hostile); assert.equal(m.sections[0].paragraphs[0], hostile); assert.equal(m.title, hostile);
});

test('termModel: paragraphs from the long text, see_also limited to real terms, null when unknown', () => {
  const b = normalizeBundle(bundle());
  const tm = termModel('vram', b);
  assert.equal(tm.title, 'VRAM'); assert.equal(tm.summary, 'GPU memory.'); assert.deepEqual(tm.paragraphs, ['First paragraph.', 'Second paragraph.']);
  assert.deepEqual(tm.seeAlso, [{ key: 'gpu', title: 'GPU' }]);
  assert.equal(termModel('ghost', b), null);
  assert.deepEqual(termChips(['gpu', 'gpu', 'ghost'], b), [{ key: 'gpu', title: 'GPU' }]);
  assert.deepEqual(paragraphs('a\n\n\n\n b \n\nc'), ['a', 'b', 'c']);
});

test('glossary: sorted case-insensitively, search needs every word, title hits rank first, letters in order', () => {
  const e = glossaryEntries(normalizeBundle(bundle()));
  assert.deepEqual(e.map((x) => x.key), ['gpu', 'kv_cache', 'vram']);
  assert.deepEqual(searchGlossary(e, '').map((x) => x.key), ['gpu', 'kv_cache', 'vram']);
  assert.deepEqual(searchGlossary(e, 'vram').map((x) => x.key), ['vram', 'kv_cache'], 'title match first, then mentions in the text');
  assert.deepEqual(searchGlossary(e, 'working memory').map((x) => x.key), ['kv_cache']);
  assert.deepEqual(searchGlossary(e, 'zzz'), []);
  assert.deepEqual(glossaryLetters(e), ['G', 'K', 'V']);
  assert.deepEqual(glossaryEntries({ glossary: { a: { title: 'A' }, b: { title: 'B', short: 'ok' } } }).map((x) => x.key), ['b'], 'entries without a short sentence are not entries');
});

test('loadLearn fetches once, shares an in-flight request, survives a failure and retries only after the cooldown', async () => {
  resetLearn();
  let calls = 0, fail = true;
  const api = { learn: async () => { calls++; if (fail) throw new Error('offline'); return bundle(); } };
  let clock = 1000;
  const [a, b] = await Promise.all([loadLearn(api, { now: () => clock }), loadLearn(api, { now: () => clock })]);
  assert.equal(calls, 1, 'concurrent callers share one request'); assert.deepEqual(a, emptyBundle()); assert.equal(isLoaded(), false);
  assert.equal(b, a);
  await loadLearn(api, { now: () => clock }); assert.equal(calls, 1, 'no retry inside the cooldown');
  clock += 20000; fail = false;
  const ok = await loadLearn(api, { now: () => clock });
  assert.equal(calls, 2); assert.equal(isLoaded(), true); assert.ok(ok.glossary.vram);
  await loadLearn(api, { now: () => clock }); assert.equal(calls, 2, 'cached in memory once loaded');
  resetLearn();
});

test('missing keys are remembered for a dev-only warning and cleared when a bundle loads', () => {
  mock.timers.enable({ apis: ['setTimeout'] });
  try {
    resetLearn(); const warned = [];
    let dev = false;
    setDevHooks({ dev: () => dev, warnFn: (m) => warned.push(m) });
    setBundle(bundle());
    assert.equal(lookup({ kind: 'concept', key: 'state.stopped' }), null); assert.deepEqual(missingKeys(), ['concept:state.stopped']);
    mock.timers.tick(2000); assert.equal(warned.length, 0, 'silent outside dev mode');
    dev = true; lookup({ kind: 'metric', key: 'prompt_tps' }); lookup({ kind: 'param', key: 'zzz', engine: 'vllm', spec: { key: 'zzz' } });
    mock.timers.tick(2000);
    assert.equal(warned.length >= 1, true); assert.match(warned.at(-1), /metric:prompt_tps/); assert.match(warned.at(-1), /param:vllm\/zzz/);
    assert.ok(lookup({ kind: 'concept', key: 'state.ready' }), 'found keys are not recorded');
    setBundle(bundle()); assert.deepEqual(missingKeys(), [], 'a fresh bundle clears the misses');
  } finally { mock.timers.reset(); resetLearn(); setDevHooks({ dev: () => false, warnFn: () => {} }); }
});

test('the mock bundle is a solid sample: 30+ terms, limits hold, every referenced term exists, every UI key is explained', () => {
  const b = learnBundle();
  assert.ok(Object.keys(b.glossary).length >= 30, `${Object.keys(b.glossary).length} terms`);
  const bad = [];
  const chk = (where, e) => {
    for (const [f, n] of [['summary', LIMITS.summary], ['details', LIMITS.details], ['when_to_change', LIMITS.when_to_change], ['risks', LIMITS.risks], ['example', LIMITS.example]]) if ((e[f] || '').length > n) bad.push(`${where}.${f} is ${e[f].length} > ${n}`);
    if (!e.summary) bad.push(`${where}: empty summary`);
    for (const k of e.terms || []) if (!b.glossary[k]) bad.push(`${where}: unknown term ${k}`);
  };
  for (const [k, e] of Object.entries(b.metrics)) chk(`metrics.${k}`, e);
  for (const [k, e] of Object.entries(b.concepts)) chk(`concepts.${k}`, e);
  for (const [en, m] of Object.entries(b.params)) for (const [k, e] of Object.entries(m)) chk(`params.${en}.${k}`, e);
  for (const [en, m] of Object.entries(b.presets)) for (const [k, e] of Object.entries(m)) chk(`presets.${en}.${k}`, e);
  for (const [k, t] of Object.entries(b.glossary)) { if (t.short.length > LIMITS.short) bad.push(`glossary.${k}.short`); if (t.long.length > LIMITS.long) bad.push(`glossary.${k}.long`); for (const s of t.see_also) if (!b.glossary[s]) bad.push(`glossary.${k}: unknown see_also ${s}`); }
  assert.deepEqual(bad, []);
  assert.deepEqual(expectedMissing(normalizeBundle(b)), [], 'every concept and metric key the UI asks for has an explanation');
});

test('every parameter of the mock vLLM and SGLang catalogs has an explanation, and the other engines have some', () => {
  const b = learnBundle();
  for (const engine of ['vllm', 'sglang']) for (const s of catalogs[engine]) assert.ok(b.params[engine][s.key]?.summary, `${engine}.${s.key}`);
  for (const engine of ['llamacpp', 'ollama', 'trtllm']) assert.ok(Object.keys(b.params[engine]).length >= 5, engine);
  for (const engine of Object.keys(catalogs)) assert.ok(Object.keys(b.presets[engine]).length >= 2, `presets for ${engine}`);
});

test('the key families the UI uses are all listed (learn-keys.js)', () => {
  for (const f of ['state', 'link', 'management', 'verdict', 'format', 'root', 'alert', 'page', 'field']) assert.ok(CONCEPT_FAMILIES[f]?.length, f);
  assert.equal(new Set(allConceptKeys()).size, allConceptKeys().length); assert.ok(METRIC_KEYS.includes('generation_tps') && METRIC_KEYS.includes('ram_used_gib'));
  assert.ok(getBundle() !== undefined);
});
