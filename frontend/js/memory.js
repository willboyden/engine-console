// Pure segment maths for the host-RAM bars (no DOM). Two invariants the tests pin down:
//  * the segments of a "used" bar sum (in GiB) to the used figure, and
//  * a tiny-but-nonzero segment still gets a visible width (MIN_PCT) without the widths exceeding 100%.
export const MIN_PCT = 1.2;
// Okabe-Ito colour-blind-safe hues; index -> CSS class m1..m8 (defined in components.css for both themes).
export const PALETTE_SIZE = 8;
const num = (v) => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const pos = (v) => (num(v) !== null && v > 0 ? v : 0);

/** Widths in % for `values` (GiB) over `total`; nonzero values get at least MIN_PCT, the rest is shrunk to fit. */
export function widths(values, total, min = MIN_PCT) {
  if (!(total > 0)) return values.map(() => 0);
  const raw = values.map((v) => (pos(v) / total) * 100);
  const small = raw.map((p, i) => pos(values[i]) > 0 && p < min);
  const fixed = small.reduce((a, s) => a + (s ? min : 0), 0);
  const big = raw.reduce((a, p, i) => a + (small[i] ? 0 : p), 0);
  const room = Math.max(0, Math.min(100, raw.reduce((a, p) => a + p, 0)) - fixed);
  const scale = big > 0 ? room / big : 0;
  return raw.map((p, i) => (small[i] ? min : p * scale));
}

/**
 * System bar: one segment per engine, then "other used", then available.
 * engines: [{id, name, gib}]. Returns {segments:[{key,label,gib,pct,cls,kind}], usedSum}.
 */
export function systemSegments({ engines = [], total, used, available }) {
  const eng = engines.filter((e) => pos(e.gib) > 0).sort((a, b) => b.gib - a.gib);
  const engSum = eng.reduce((a, e) => a + e.gib, 0);
  const other = Math.max(0, (num(used) ?? engSum) - engSum);
  const avail = pos(available);
  const items = [...eng.map((e, i) => ({ key: e.id, label: e.name, gib: e.gib, cls: `m${(i % (PALETTE_SIZE - 1)) + 1}`, kind: 'engine' })),
    { key: 'other', label: null, gib: other, cls: 'm-other', kind: 'other' }, { key: 'available', label: null, gib: avail, cls: 'm-avail', kind: 'available' }].filter((s) => s.gib > 0);
  const denom = total > 0 ? total : items.reduce((a, s) => a + s.gib, 0);
  const w = widths(items.map((s) => s.gib), denom);
  return { segments: items.map((s, i) => ({ ...s, pct: w[i] })), usedSum: engSum + other };
}

/** Per-instance host_memory -> anon/cache/shm(/kernel) segments; null when unavailable. */
export function hostSegments(hm) {
  if (!hm || num(hm.total_gib) === null) return null;
  const parts = [['anon', hm.anon_gib, 'hm-anon'], ['cache', hm.cache_gib, 'hm-cache'], ['shm', hm.shmem_gib, 'hm-shm'], ['kernel', hm.kernel_gib, 'hm-kernel']];
  const items = parts.map(([key, gib, cls]) => ({ key, gib: pos(gib), cls })).filter((s) => s.gib > 0);
  const w = widths(items.map((s) => s.gib), items.reduce((a, s) => a + s.gib, 0));
  return { total: hm.total_gib, approx: hm.source === 'rss', segments: items.map((s, i) => ({ ...s, pct: w[i] })) };
}

export const engineMemory = (instances) => (instances || []).filter((i) => num(i.host_memory?.total_gib) !== null && i.host_memory.total_gib > 0).map((i) => ({ id: i.id, name: i.name || i.id, gib: i.host_memory.total_gib }));

export const normSystemMemory = (m) => (m ? { total: num(m.total_gib), used: num(m.used_gib), available: num(m.available_gib), free: num(m.free_gib), cached: num(m.cached_gib), shmem: num(m.shmem_gib), swapTotal: num(m.swap_total_gib), swapUsed: num(m.swap_used_gib), updatedAt: m.updated_at ?? null } : null);
// Host-RAM verdict block of the fit report.
export const normHostRam = (h) => (h ? { needed: num(h.needed_gib), available: num(h.available_gib), total: num(h.total_gib), verdict: h.verdict || 'unknown', breakdown: Object.entries(h.breakdown || {}).filter(([, v]) => num(v) !== null), notes: h.notes || [] } : null);
