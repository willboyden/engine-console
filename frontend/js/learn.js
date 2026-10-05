// Learn mode data layer (no DOM): the GET /learn bundle, tolerant lookups, popover content and glossary search.
// Everything here is pure or module-state only so it can be unit-tested in Node. Text from the bundle is untrusted:
// callers render it with textContent (h()), never as markup.
//
// Missing keys are normal (the content is authored separately from the UI): a lookup that finds nothing returns
// null, the caller shows no icon, and the key is remembered so a dev-only console warning can list them.
import { allConceptKeys, METRIC_KEYS } from './learn-keys.js';

export const LEVELS = ['basic', 'advanced', 'expert'];

export const emptyBundle = () => ({ version: 1, glossary: {}, metrics: {}, concepts: {}, params: {}, presets: {} });
const isObj = (x) => x !== null && typeof x === 'object' && !Array.isArray(x);
const objOr = (x) => (isObj(x) ? x : {});
const text = (v) => (typeof v === 'string' ? v.trim() : '');

/** Coerce one Explain-shaped value; null when it has no usable summary (an entry without a summary is no entry). */
export function normExplain(e) {
  if (!isObj(e)) return null;
  const summary = text(e.summary);
  if (!summary) return null;
  return {
    summary, details: text(e.details), when_to_change: text(e.when_to_change), risks: text(e.risks), example: text(e.example),
    terms: Array.isArray(e.terms) ? e.terms.filter((k) => typeof k === 'string' && k) : [],
    level: LEVELS.includes(e.level) ? e.level : 'advanced',
  };
}

/** A glossary Term {title, short, long, see_also} viewed as an Explain, so one popover renderer serves both. */
export function termAsExplain(t) {
  if (!isObj(t)) return null;
  const short = text(t.short);
  if (!short) return null;
  return { summary: short, details: text(t.long), when_to_change: '', risks: '', example: '', terms: Array.isArray(t.see_also) ? t.see_also.filter((k) => typeof k === 'string') : [], level: 'basic' };
}

export function normalizeBundle(raw) {
  const b = emptyBundle();
  if (!isObj(raw)) return b;
  if (Number.isFinite(raw.version)) b.version = raw.version;
  b.glossary = objOr(raw.glossary);
  b.metrics = objOr(raw.metrics);
  b.concepts = objOr(raw.concepts);
  b.params = Object.fromEntries(Object.entries(objOr(raw.params)).map(([engine, m]) => [engine, objOr(m)]));
  b.presets = Object.fromEntries(Object.entries(objOr(raw.presets)).map(([engine, m]) => [engine, objOr(m)]));
  return b;
}

// ---- store ----
let bundle = emptyBundle();
let loaded = false;
let inflight = null;
let lastTry = 0;
const listeners = new Set();
export const getBundle = () => bundle;
export const isLoaded = () => loaded;
export function setBundle(raw) {
  bundle = normalizeBundle(raw); loaded = true; missing.clear();
  if (devCheck()) { const gone = expectedMissing(bundle); if (gone.length) warn(`[learn] ${gone.length} key(s) the UI uses have no explanation in the bundle: ${gone.join(', ')}`); }
  for (const fn of [...listeners]) fn(bundle);
  return bundle;
}
export function resetLearn() { bundle = emptyBundle(); loaded = false; inflight = null; lastTry = 0; missing.clear(); }
export const onLearn = (fn) => { listeners.add(fn); return () => listeners.delete(fn); };

/**
 * Fetch /learn once and cache it in memory. A failure (offline, older backend, 401) leaves an empty bundle and the UI
 * works without help icons; the next call retries after `retryAfterMs`.
 */
export function loadLearn(api, { retryAfterMs = 10000, now = Date.now } = {}) {
  if (loaded) return Promise.resolve(bundle);
  if (inflight) return inflight;
  if (lastTry && now() - lastTry < retryAfterMs) return Promise.resolve(bundle);
  lastTry = now();
  inflight = Promise.resolve()
    .then(() => api.learn())
    .then((raw) => setBundle(raw), () => bundle)
    .finally(() => { inflight = null; });
  return inflight;
}

// ---- lookups ----
// Names in the contract are prose ("host, GPU pick, root pick"): accept the likely spellings of a concept key.
export const ALIASES = {
  'field.gpus': ['field.gpu', 'field.gpu_pick', 'field.gpu_ids'],
  'field.root': ['field.root_pick', 'field.model_root'],
  'field.ttl': ['field.idle_ttl', 'field.ttl_idle'],
  'field.host': ['field.machine'],
  'field.pin': ['field.pinned'],
  'field.concurrency': ['field.max_concurrency'],
  'field.image': ['field.engine_image'],
  'field.quant': ['field.gguf_quant', 'field.quantization'],
  'field.policy_mode': ['field.mode'],
  'field.target_host': ['field.host'],
  'page.instance': ['page.instances'],
  'page.hosts': ['page.dashboard'],
};

/** ref = {kind: 'concept'|'metric'|'param'|'preset'|'term', key, engine?}. Returns a normalised Explain or null. */
export function explainFor(ref, b = bundle) {
  const { kind, key, engine } = ref || {};
  if (!key) return null;
  for (const k of [key, ...(ALIASES[key] || [])]) {
    let e = null;
    if (kind === 'concept') e = b.concepts[k];
    else if (kind === 'metric') e = b.metrics[k] ?? b.concepts[k];
    else if (kind === 'param') e = b.params[engine]?.[k];
    else if (kind === 'preset') e = b.presets[engine]?.[k];
    else if (kind === 'term') e = termAsExplain(b.glossary[k]);
    const n = normExplain(e);
    if (n) return n;
  }
  return null;
}

/** The Explain for one engine parameter: the catalog's own `explain` first, then the bundle. */
export function paramExplain(spec, engine, b = bundle) {
  return normExplain(spec?.explain) || explainFor({ kind: 'param', key: spec?.key, engine }, b);
}

export const refId = (ref) => `${ref.kind}:${ref.engine ? `${ref.engine}/` : ''}${ref.key}`;

// ---- missing keys (dev-only warning) ----
const missing = new Set();
let devCheck = () => {
  try { return /[?&]dev(=|&|$)/.test(globalThis.location?.search || '') || globalThis.localStorage?.getItem('ec.dev') === '1'; } catch { return false; }
};
let warn = (...a) => console.warn(...a);
let timer = null;
export function setDevHooks({ dev, warnFn } = {}) { if (dev) devCheck = dev; if (warnFn) warn = warnFn; }
export const missingKeys = () => [...missing].sort();
export function recordMissing(ref) {
  if (!ref?.key) return;
  missing.add(refId(ref));
  if (timer || !devCheck()) return;
  timer = setTimeout(() => { timer = null; if (missing.size) warn(`[learn] ${missing.size} UI key(s) have no explanation: ${missingKeys().join(', ')}`); }, 1500);
  timer.unref?.();
}
/** Keys this UI asks for (learn-keys.js) that the loaded bundle has no explanation for: the content writer's checklist. */
export function expectedMissing(b = bundle) {
  return [...allConceptKeys().filter((k) => !normExplain(b.concepts[k])).map((k) => `concept:${k}`), ...METRIC_KEYS.filter((k) => !normExplain(b.metrics[k])).map((k) => `metric:${k}`)];
}
/** Look up and remember a miss; the one call sites use. */
export function lookup(ref, b = bundle) {
  const e = ref.kind === 'param' && ref.spec ? paramExplain(ref.spec, ref.engine, b) : explainFor(ref, b);
  if (!e) recordMissing(ref.kind === 'param' && ref.spec ? { ...ref, key: ref.spec.key } : ref);
  return e;
}

// ---- popover content ----
export const SECTIONS = [
  { id: 'details', field: 'details', label: null },
  { id: 'when', field: 'when_to_change', label: 'learn.when_to_change' },
  { id: 'risks', field: 'risks', label: 'learn.what_can_go_wrong' },
  { id: 'example', field: 'example', label: 'learn.example' },
];
export const paragraphs = (s) => String(s || '').split(/\n{2,}/).map((p) => p.trim()).filter(Boolean);

export const termTitle = (key, b = bundle) => text(b.glossary[key]?.title) || key;
/** Glossary entries an Explain points at; unknown keys are dropped (and remembered as missing). */
export function termChips(keys, b = bundle) {
  const out = [];
  for (const key of [...new Set(keys || [])]) {
    if (isObj(b.glossary[key]) && text(b.glossary[key].short)) out.push({ key, title: termTitle(key, b) });
    else recordMissing({ kind: 'term', key });
  }
  return out;
}

/** What the info popover shows for an Explain: no DOM, so the shape is testable. */
export function popoverModel(explain, { title = '', bundle: b = bundle } = {}) {
  if (!explain) return null;
  return {
    title,
    summary: explain.summary,
    sections: SECTIONS.map((s) => ({ id: s.id, label: s.label, paragraphs: paragraphs(explain[s.field]) })).filter((s) => s.paragraphs.length),
    terms: termChips(explain.terms, b),
    level: explain.level,
  };
}

/** The view shown after clicking a glossary chip inside the popover (null when the term does not exist). */
export function termModel(key, b = bundle) {
  const t = b.glossary[key];
  if (!isObj(t) || !text(t.short)) return null;
  return { key, title: termTitle(key, b), summary: text(t.short), paragraphs: paragraphs(t.long), seeAlso: termChips(t.see_also, b) };
}

// ---- glossary page ----
export function glossaryEntries(b = bundle) {
  return Object.entries(b.glossary)
    .filter(([, t]) => isObj(t) && text(t.short))
    .map(([key, t]) => ({ key, title: text(t.title) || key, short: text(t.short), long: text(t.long), see_also: Array.isArray(t.see_also) ? t.see_also : [] }))
    .sort((a, b2) => a.title.localeCompare(b2.title, 'en', { sensitivity: 'base' }));
}

/** All query words must appear in title/short/long/key; title hits rank first. An empty query keeps everything. */
export function searchGlossary(entries, query) {
  const words = String(query || '').toLowerCase().split(/\s+/).filter(Boolean);
  if (!words.length) return entries;
  const scored = [];
  for (const e of entries) {
    const title = e.title.toLowerCase(), hay = `${title} ${e.key} ${e.short} ${e.long}`.toLowerCase();
    if (!words.every((w) => hay.includes(w))) continue;
    const rank = words.every((w) => title.startsWith(w)) ? 0 : words.every((w) => title.includes(w)) ? 1 : 2;
    scored.push([rank, e]);
  }
  return scored.sort((a, b2) => a[0] - b2[0] || a[1].title.localeCompare(b2[1].title, 'en', { sensitivity: 'base' })).map((s) => s[1]);
}

/** First letters present in a (sorted) entry list, for the A-Z jump bar. */
export const glossaryLetters = (entries) => [...new Set(entries.map((e) => e.title.charAt(0).toUpperCase()))];

// ---- coverage helpers (tests, and the mock-bundle sanity check) ----
export const LIMITS = { summary: 160, details: 900, when_to_change: 400, risks: 400, example: 200, short: 160, long: 1500 };
