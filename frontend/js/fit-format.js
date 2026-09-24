// Pure helpers for FitReport rendering. The backend contract (ARCHITECTURE §3 `fit`) names the fields but
// not the per-GPU array key, so normalizeReport accepts per_gpu | gpus | breakdown.
export const SEGMENTS = [
  { key: 'weights', label: 'fit.weights', cls: 'seg-weights' },
  { key: 'kv_cache', label: 'fit.kv', cls: 'seg-kv' },
  { key: 'activations', label: 'fit.activations', cls: 'seg-act' },
  { key: 'cuda_graphs', label: 'fit.graphs', cls: 'seg-graphs' },
  { key: 'overhead', label: 'fit.overhead', cls: 'seg-over' },
];
const num = (v) => (typeof v === 'number' && Number.isFinite(v) ? v : 0);

export function normalizeReport(r) {
  if (!r || typeof r !== 'object') return null;
  const arr = r.per_gpu || r.gpus || r.breakdown || [];
  const gpus = (Array.isArray(arr) ? arr : Object.entries(arr).map(([k, v]) => ({ gpu_id: +k, ...v }))).map((g, i) => {
    // Backend names are *_gib (weights_gib, kv_cache_gib, ...); the older short names are also accepted.
    const seg = Object.fromEntries(SEGMENTS.map((s) => [s.key, num(g[`${s.key}_gib`] ?? g[s.key])]));
    const tot = g.total_gib ?? g.total;
    const total = typeof tot === 'number' ? tot : Object.values(seg).reduce((a, b) => a + b, 0);
    return { id: g.gpu_id ?? g.id ?? i, ...seg, total, budget: num(g.budget_gib ?? g.budget), capacity: num(g.vram_total_gib ?? g.capacity ?? g.gpu_total_gib), free: g.free_gib ?? g.free ?? null, verdict: g.verdict };
  });
  return {
    verdict: r.verdict || 'unknown',
    confidence: r.confidence || 'low',
    gpus,
    notes: r.notes || [],
    compat: r.compat || [],
    tpRequired: r.tp_required ?? null,
    tp: r.tp ?? null, concurrency: r.concurrency ?? null,
    maxContext: r.max_context_at_current_concurrency ?? null,
    maxConcurrency: r.max_concurrency_at_current_context ?? null,
    fitsIfStop: Array.isArray(r.fits_if_stop) ? (r.fits_if_stop.length ? r.fits_if_stop : null) : r.fits_if_stop || null,
  };
}

export const VERDICT = {
  fits: { cls: 'ok', label: 'fit.fits' },
  tight: { cls: 'warn', label: 'fit.tight' },
  wont_fit: { cls: 'bad', label: 'fit.wont' },
  unknown: { cls: 'muted', label: 'fit.unknown' },
};
export const verdictMeta = (v) => VERDICT[v] || VERDICT.unknown;

// Verdict thresholds from the contract: fits <= 90% of budget, tight 90-100%, else wont_fit.
export function verdictFor(total, budget) {
  if (!(budget > 0)) return 'unknown';
  const r = total / budget;
  return r <= 0.9 ? 'fits' : r <= 1 ? 'tight' : 'wont_fit';
}

export function compatLevel(compat) {
  return (compat || []).reduce((w, c) => (c.level === 'block' ? 'block' : c.level === 'warn' && w !== 'block' ? 'warn' : w), 'ok');
}

// Cheap client-side estimate for search rows (weights only, no KV) — always labelled "est." in the UI.
export function quickFit(weightBytes, gpuTotalsGiB, util = 0.9) {
  if (!(weightBytes > 0) || !gpuTotalsGiB?.length) return { verdict: 'unknown', gpus: null };
  const gib = weightBytes / 2 ** 30;
  const per = Math.max(...gpuTotalsGiB) * util;
  for (let n = 1; n <= gpuTotalsGiB.length; n++) {
    if (gib / n <= per * 0.8) return { verdict: 'fits', gpus: n };
    if (gib / n <= per) return { verdict: 'tight', gpus: n };
  }
  return { verdict: 'wont_fit', gpus: null };
}
