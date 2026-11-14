// Getting started + FAQ content of every page (i18n/help.en.json). Pure logic, no DOM, so tests import it.
// The file holds one entry per page, keyed by the nav id of js/app.js ROUTES ({lead, steps: [{title, body, cmd?, tab?}],
// faq: [{q, a, cmd?, tab?, route?}]}) plus `_console` ({faq}), the questions that apply on every page. Text is plain
// text: `backticks` mark inline code and a blank line splits paragraphs; nothing in it is ever parsed as HTML.

export const HELP_URL = 'i18n/help.en.json';
export const CONSOLE_KEY = '_console';
/** The tabs every page frame has (components/page-tabs.js): the page itself, then the two help tabs. */
export const HELP_TABS = ['page', 'start', 'faq'];
const STEP_KEYS = new Set(['title', 'body', 'cmd', 'tab']);
const FAQ_KEYS = new Set(['q', 'a', 'cmd', 'tab', 'route']);

let pending = null;
/** Fetch the help file once per page load; a failed fetch is not cached, so Retry fetches again. */
export function loadHelp(url = HELP_URL, fetchFn = globalThis.fetch) {
  pending ||= fetchFn(url)
    .then((r) => {
      if (!r.ok) throw new Error(`help load failed: ${r.status}`);
      return r.json();
    })
    .catch((e) => {
      pending = null;
      throw e;
    });
  return pending;
}
export function resetHelp() {
  pending = null;
}

/** Split text into paragraphs of [{code: bool, text}] runs: `x` is inline code, a blank line starts a paragraph. */
export function richRuns(text) {
  return String(text ?? '')
    .split(/\n\s*\n/)
    .map((para) =>
      para
        .split('`')
        .map((s, i) => ({ code: i % 2 === 1, text: s }))
        .filter((r) => r.text !== '')
    )
    .filter((p) => p.length);
}

/** FAQ items whose question, answer or command contains every word of the query (case-insensitive). Empty keeps all. */
export function filterFaq(items, query) {
  const words = String(query ?? '')
    .toLowerCase()
    .split(/\s+/)
    .filter(Boolean);
  if (!words.length) return items.slice();
  return items.filter((it) => {
    const hay = `${it.q} ${it.a} ${it.cmd || ''}`.toLowerCase();
    return words.every((w) => hay.includes(w));
  });
}

/**
 * Problems with a help document, for the tests: every page has a lead, at least three steps and four questions,
 * every field is a non-empty string, `tab` names one of HELP_TABS, `route` a nav route, and no URL leaves loopback.
 * @param {object} doc     the parsed help file
 * @param {string[]} pages the page keys that must have an entry (the nav ids of the visible routes)
 * @param {string[]} routes nav paths (`/settings`, ...)
 */
export function helpProblems(doc, pages, routes) {
  const out = [];
  const str = (v) => typeof v === 'string' && v.trim().length > 0;
  const checkItem = (where, it, keys, required) => {
    if (!it || typeof it !== 'object') {
      out.push(`${where}: not an object`);
      return;
    }
    for (const k of Object.keys(it)) if (!keys.has(k)) out.push(`${where}: unknown key ${k}`);
    for (const k of required) if (!str(it[k])) out.push(`${where}: ${k} missing or empty`);
    for (const k of ['cmd', 'tab', 'route']) if (k in it && !str(it[k])) out.push(`${where}: ${k} empty`);
    if (it.tab && !HELP_TABS.includes(it.tab)) out.push(`${where}: unknown tab ${it.tab}`);
    if (it.route && !routes.includes(it.route)) out.push(`${where}: unknown route ${it.route}`);
    for (const [k, v] of Object.entries(it)) {
      if (typeof v !== 'string') continue;
      if (/https?:\/\/(?!(127\.0\.0\.1|localhost)[:/])/.test(v)) out.push(`${where}.${k}: external URL`); // loopback examples are fine
      if ((v.match(/`/g) || []).length % 2) out.push(`${where}.${k}: unbalanced backtick`);
    }
  };
  for (const id of Object.keys(doc || {})) if (id !== CONSOLE_KEY && !pages.includes(id)) out.push(`${id}: not a page`);
  const common = doc?.[CONSOLE_KEY];
  if (!Array.isArray(common?.faq) || !common.faq.length) out.push(`${CONSOLE_KEY}: faq missing`);
  else common.faq.forEach((it, i) => checkItem(`${CONSOLE_KEY}.faq[${i}]`, it, FAQ_KEYS, ['q', 'a']));
  for (const id of pages) {
    const e = doc?.[id];
    if (!e) {
      out.push(`${id}: no entry`);
      continue;
    }
    if (!str(e.lead)) out.push(`${id}: lead missing`);
    for (const k of Object.keys(e)) if (!['lead', 'steps', 'faq'].includes(k)) out.push(`${id}: unknown key ${k}`);
    if (typeof e.lead === 'string' && (e.lead.match(/`/g) || []).length % 2)
      out.push(`${id}.lead: unbalanced backtick`);
    if (!Array.isArray(e.steps) || e.steps.length < 3) out.push(`${id}: fewer than 3 steps`);
    else e.steps.forEach((it, i) => checkItem(`${id}.steps[${i}]`, it, STEP_KEYS, ['title', 'body']));
    if (!Array.isArray(e.faq) || e.faq.length < 4) out.push(`${id}: fewer than 4 questions`);
    else e.faq.forEach((it, i) => checkItem(`${id}.faq[${i}]`, it, FAQ_KEYS, ['q', 'a']));
    const qs = (e.faq || []).map((f) => f?.q);
    if (new Set(qs).size !== qs.length) out.push(`${id}: duplicate question`);
  }
  const cq = (common?.faq || []).map((f) => f?.q);
  if (new Set(cq).size !== cq.length) out.push(`${CONSOLE_KEY}: duplicate question`);
  return out;
}
