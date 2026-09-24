import test from 'node:test';
import assert from 'node:assert/strict';
import { fmtBytes, fmtCompact, fmtDuration, fmtEta, fmtMs, timeAgo, items, debounce, csvCell, fmtTps, fmtPct } from '../js/format.js';
import { fuzzyScore, fuzzyFilter } from '../js/fuzzy.js';
import { createStore, safeStorage } from '../js/store.js';
import { setMessages, t } from '../js/i18n.js';

test('fmtBytes', () => { assert.equal(fmtBytes(0), '0 B'); assert.equal(fmtBytes(1536), '1.5 KiB'); assert.equal(fmtBytes(36 * 2 ** 30), '36.0 GiB'); assert.equal(fmtBytes(null), '–'); });
test('fmtCompact', () => { assert.equal(fmtCompact(999), '999'); assert.equal(fmtCompact(1520000), '1.5M'); assert.equal(fmtCompact(35e9), '35B'); assert.equal(fmtCompact(NaN), '–'); });
test('fmtDuration / fmtEta', () => { assert.equal(fmtDuration(59), '59s'); assert.equal(fmtDuration(125), '2m 05s'); assert.equal(fmtDuration(3700), '1h 01m'); assert.equal(fmtDuration(90000), '1d 1h'); assert.equal(fmtEta(-1), '–'); });
test('fmtMs / fmtTps / fmtPct', () => { assert.equal(fmtMs(0.0123), '12 ms'); assert.equal(fmtMs(2.5), '2.50 s'); assert.equal(fmtTps(92.34), '92.3 tok/s'); assert.equal(fmtPct(12.34, 1), '12.3%'); });
test('timeAgo', () => { const n = 1e12 + 1e7; assert.equal(timeAgo(n - 30000, n), '30s ago'); assert.equal(timeAgo(n - 7200e3, n), '2h ago'); assert.equal(timeAgo('garbage', n), '–'); });
test('items unwraps arrays and envelopes', () => { assert.deepEqual(items([1]), [1]); assert.deepEqual(items({ items: [2] }), [2]); assert.deepEqual(items(null), []); });
test('csvCell quotes', () => { assert.equal(csvCell('a,b'), '"a,b"'); assert.equal(csvCell('q"x'), '"q""x"'); assert.equal(csvCell(null), ''); });
test('debounce coalesces and cancels', async () => {
  let n = 0; const d = debounce(() => n++, 10); d(); d(); d(); await new Promise((r) => setTimeout(r, 30)); assert.equal(n, 1);
  d(); d.cancel(); await new Promise((r) => setTimeout(r, 30)); assert.equal(n, 1);
});

test('fuzzy: subsequence match, ranking, no-match', () => {
  assert.ok(fuzzyScore('lnch', 'Launch a model') >= 0); assert.equal(fuzzyScore('zzz', 'Launch'), -1); assert.equal(fuzzyScore('', 'x'), 0);
  assert.equal(fuzzyFilter('chat', [{ label: 'Open settings' }, { label: 'Chat with qwen' }, { label: 'Go to chat' }])[0].label, 'Chat with qwen');
});

test('store: set/get, selector subscription only fires on change, unsubscribe', () => {
  const s = createStore({ a: 1, b: 1 }); const seen = [];
  const un = s.subscribe((v) => seen.push(v), (st) => st.a);
  s.set({ b: 2 }); s.set({ a: 2 }); s.set((st) => ({ a: st.a + 1 })); un(); s.set({ a: 9 });
  assert.deepEqual(seen, [2, 3]); assert.equal(s.get().a, 9);
});

test('safeStorage survives a throwing/absent backend', () => {
  const bad = safeStorage(() => { throw new Error('denied'); });
  assert.equal(bad.get('k', 'dflt'), 'dflt'); assert.equal(bad.set('k', 'v'), false); bad.remove('k');
  const mem = new Map(); const ok = safeStorage(() => ({ getItem: (k) => mem.get(k) ?? null, setItem: (k, v) => mem.set(k, v), removeItem: (k) => mem.delete(k) }));
  ok.set('a', '1'); assert.equal(ok.get('a'), '1'); ok.remove('a'); assert.equal(ok.get('a'), null);
});

test('i18n interpolation, plurals, missing keys', () => {
  setMessages({ hi: 'Hi {name}', 'n_one': '{count} item', 'n_other': '{count} items' });
  assert.equal(t('hi', { name: 'A' }), 'Hi A'); assert.equal(t('n', { count: 1 }), '1 item'); assert.equal(t('n', { count: 3 }), '3 items'); assert.equal(t('missing.key'), 'missing.key'); assert.equal(t('hi', {}), 'Hi {name}');
});
