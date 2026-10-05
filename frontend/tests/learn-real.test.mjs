// Cross-check of the UI against the learn content the hub really ships (backend/src/engine_console/learn/*.json), assembled the way the
// hub assembles it. Skipped when the backend tree is not next to the frontend (a standalone copy of this directory).
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { bundleFromDir } from '../dev/mock-learn.mjs';
import { normalizeBundle, explainFor, popoverModel, termModel, glossaryEntries, searchGlossary, LIMITS } from '../js/learn.js';
import { allConceptKeys, METRIC_KEYS } from '../js/learn-keys.js';

const DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../backend/src/engine_console/learn');
const have = fs.existsSync(path.join(DIR, 'concepts.json'));
const real = have ? normalizeBundle(bundleFromDir(DIR)) : null;
const opts = { skip: have ? false : 'backend learn content not found next to the frontend' };

test('every concept and metric key the UI asks for has an explanation in the shipped content (aliases allowed)', opts, () => {
  const missing = [...allConceptKeys().filter((k) => !explainFor({ kind: 'concept', key: k }, real)).map((k) => `concept:${k}`),
    ...METRIC_KEYS.filter((k) => !explainFor({ kind: 'metric', key: k }, real)).map((k) => `metric:${k}`)];
  assert.deepEqual(missing, [], 'ask the content writer for these, or drop the ⓘ');
});

test('every shipped explanation renders into a popover model and stays inside the documented limits', opts, () => {
  const bad = [];
  const each = (where, e) => {
    for (const f of ['summary', 'details', 'when_to_change', 'risks', 'example']) if ((e[f] || '').length > LIMITS[f]) bad.push(`${where}.${f}`);
    const m = popoverModel(e, { title: where, bundle: real });
    if (!m || !Array.isArray(m.sections)) bad.push(`${where}: no model`);
  };
  for (const [k, e] of Object.entries(real.metrics)) each(`metrics.${k}`, e);
  for (const [k, e] of Object.entries(real.concepts)) each(`concepts.${k}`, e);
  for (const [en, m] of Object.entries(real.params)) for (const [k, e] of Object.entries(m)) each(`params.${en}.${k}`, e);
  for (const [en, m] of Object.entries(real.presets)) for (const [k, e] of Object.entries(m)) each(`presets.${en}.${k}`, e);
  assert.deepEqual(bad, []);
});

test('every glossary term shown by the Glossary page resolves, and every term an explanation links to exists', opts, () => {
  const refs = new Set();
  for (const m of [real.metrics, real.concepts, ...Object.values(real.params), ...Object.values(real.presets)]) for (const e of Object.values(m)) for (const t of e.terms || []) refs.add(t);
  assert.deepEqual([...refs].filter((t) => !real.glossary[t]), []);
  for (const k of Object.keys(real.glossary)) assert.ok(termModel(k, real)?.title, `term ${k} renders`);
  const entries = glossaryEntries(real);
  assert.equal(entries.length, Object.keys(real.glossary).length);
  assert.ok(searchGlossary(entries, 'vram').some((e) => e.key === 'vram'));
});
