import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { helpProblems, richRuns, filterFaq, loadHelp, resetHelp, CONSOLE_KEY, HELP_TABS } from '../js/help.js';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const doc = JSON.parse(fs.readFileSync(path.join(root, 'i18n/help.en.json'), 'utf8'));
// ROUTES of js/app.js, read as text (app.js boots the UI on import): every route's nav id, and the visible paths.
const app = fs.readFileSync(path.join(root, 'js/app.js'), 'utf8');
const routes = [...app.matchAll(/\{ path: '([^']+)', tag: '[^']+', mod: '[^']+', nav: '([a-z]+)'.*\},?$/gm)].map((m) => ({ path: m[1], nav: m[2], hidden: /hidden: true/.test(m[0]) }));
const pages = [...new Set(routes.map((r) => r.nav))];
const navPaths = routes.filter((r) => !r.hidden).map((r) => r.path);

test('the route table was read', () => {
  assert.ok(routes.length >= 18, `${routes.length} routes`);
  assert.ok(pages.includes('config') && pages.includes('settings') && pages.includes('dashboard'));
  assert.ok(!navPaths.includes('/hosts/:id') && navPaths.includes('/config'), navPaths.join(' '));
});

test('help.en.json has a valid guide and FAQ for every page (hidden routes use their nav page) and the console questions', () => {
  assert.deepEqual(helpProblems(doc, pages, navPaths), []);
  assert.deepEqual(Object.keys(doc).filter((k) => k !== CONSOLE_KEY).sort(), [...pages].sort());
});

test('every page gets the same three tabs from the one frame component', () => {
  assert.deepEqual(HELP_TABS, ['page', 'start', 'faq']);
  assert.match(app, /pageFrame\(\{ view, key: route\.nav/);
  const frame = fs.readFileSync(path.join(root, 'js/components/page-tabs.js'), 'utf8');
  for (const want of ["role: 'tab'", "role: 'tabpanel'", "role: 'tablist'", "'aria-controls'", "'aria-labelledby'", 'aria-selected', 'ArrowRight', 'Home', 'End', 'replaceState']) assert.ok(frame.includes(want), want);
});

test('helpProblems catches the mistakes it is meant to catch', () => {
  const bad = {
    [CONSOLE_KEY]: { faq: [{ q: 'Q', a: 'A', tab: 'nope' }] },
    library: { lead: 'x', steps: [{ title: 't', body: 'b' }], faq: [{ q: 'q', a: 'see https://example.org, not http://127.0.0.1:8791/' }, { q: 'q', a: 'odd `tick' }, { q: 'r', a: 'a', route: '/nowhere' }, { q: 's', a: 'a', extra: 1 }] },
    nosuch: {},
  };
  const out = helpProblems(bad, ['library', 'chat'], ['/library']);
  for (const want of ['nosuch: not a page', 'unknown tab nope', 'fewer than 3 steps', 'external URL', 'unbalanced backtick', 'unknown route /nowhere', 'unknown key extra', 'duplicate question', 'chat: no entry']) {
    assert.ok(out.some((p) => p.includes(want)), `${want} in ${JSON.stringify(out)}`);
  }
  assert.ok(!out.some((p) => p.includes('127.0.0.1')), 'a loopback URL is fine');
});

test('richRuns splits paragraphs and inline code, and never yields markup', () => {
  assert.deepEqual(richRuns('Run `make all` now.\n\nThen <b>stop</b>.'), [
    [{ code: false, text: 'Run ' }, { code: true, text: 'make all' }, { code: false, text: ' now.' }],
    [{ code: false, text: 'Then <b>stop</b>.' }],
  ]);
  assert.deepEqual(richRuns(''), []);
  assert.deepEqual(richRuns(undefined), []);
});

test('filterFaq matches every word in question, answer or command, case-insensitively', () => {
  const items = [{ q: 'Up timed out', a: 'Raise the timeout.' }, { q: 'Why 409?', a: 'stale base', cmd: 'engine-console hosts check' }];
  assert.equal(filterFaq(items, '').length, 2);
  assert.deepEqual(filterFaq(items, 'TIMEOUT').map((i) => i.q), ['Up timed out']);
  assert.deepEqual(filterFaq(items, '409 stale').map((i) => i.q), ['Why 409?']);
  assert.deepEqual(filterFaq(items, 'hosts').map((i) => i.q), ['Why 409?']);
  assert.deepEqual(filterFaq(items, 'timeout 409'), []);
});

test('loadHelp fetches once, and a failure is not cached', async () => {
  resetHelp();
  let calls = 0;
  const failing = async () => { calls++; return { ok: false, status: 500 }; };
  await assert.rejects(loadHelp('x', failing));
  const ok = async () => { calls++; return { ok: true, json: async () => ({ a: 1 }) }; };
  assert.deepEqual(await loadHelp('x', ok), { a: 1 });
  assert.deepEqual(await loadHelp('x', ok), { a: 1 });
  assert.equal(calls, 2);
  resetHelp();
});

test('help text names no lab machine, private address or home path', () => {
  const all = JSON.stringify(doc);
  assert.doesNotMatch(all, /\/home\/|\b10\.\d+\.\d+\.\d+|\b192\.168\.|\b172\.(1[6-9]|2\d|3[01])\./);
});
