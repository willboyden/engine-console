// Normalizers from the backend's response shapes (ARCHITECTURE §8 + domain/models.py) to the flat shapes the views use.
// Each accepts both the backend field names and the older/simpler ones so the mock and the real server both work.
const num = (v) => (typeof v === 'number' && Number.isFinite(v) ? v : undefined);
const GIB = 2 ** 30;

export function normDownload(d) {
  if (!d) return d;
  return { ...d, status: d.state ?? d.status, bytes_total: d.total_bytes ?? d.bytes_total ?? 0, bytes_done: d.done_bytes ?? d.bytes_done ?? 0 };
}

// External instances (engines that were running before the console started) are monitor-only: managed=false / source="external".
export const isExternal = (i) => i?.managed === false || i?.source === 'external';
// A short "why" for states that are informational rather than failures.
export const stateReason = (i) => i?.state_reason ?? i?.reason ?? i?.error ?? null;
// Stable fingerprint so pollers can skip repainting when nothing changed (no flicker).
export const listSignature = (list) => JSON.stringify((list || []).map((i) => [i.id, i.state, i.pinned, i.name, i.port, i.phase, Math.round(i.progress_pct ?? -1), i.state_reason ?? i.reason ?? i.error ?? '', (i.served_models || []).join(','), i.ttl_idle_s, i.image, i.host_memory ? Math.round(i.host_memory.total_gib) : null]));

export function normInstance(i) {
  if (!i) return i;
  return { ...i, managed: i.managed !== false, external: isExternal(i), served_models: i.served_models || [], ttl_s: i.ttl_idle_s ?? i.ttl_s ?? 0, progress: { phase: i.phase ?? i.progress?.phase ?? null, pct: i.progress_pct ?? i.progress?.pct } };
}

// Uptime in seconds from either `uptime_s` or `started_at` (epoch seconds or ISO string).
export function uptimeOf(i, now = Date.now()) {
  if (num(i.uptime_s) !== undefined) return i.uptime_s;
  if (i.started_at == null) return null;
  const t = typeof i.started_at === 'number' ? (i.started_at < 1e12 ? i.started_at * 1000 : i.started_at) : Date.parse(i.started_at);
  return Number.isFinite(t) ? Math.max(0, (now - t) / 1000) : null;
}

export function normHit(m) {
  return { ...m, size_bytes: m.size_bytes ?? (num(m.approx_size_gib) !== undefined ? m.approx_size_gib * GIB : undefined), quant: m.quant ?? m.quantization ?? null };
}

export function normLocalModel(m) {
  const fit = m.engines_fit || Object.fromEntries((m.engines_that_fit || []).map((e) => [e, 'fits']));
  return { ...m, engines_fit: fit };
}

// A metrics point: {t, values:{...}} (backend) or flat {t, key:val} (mock).
export function normPoint(p) { return p && p.values ? { t: p.t, ...p.values } : p; }

// Bench run -> flat results the compare table/chart use. Backend: results = {concurrency:[level...], single, prefix_cache}.
export function normBench(r) {
  const res = r.results || {};
  const levels = Array.isArray(res.concurrency) ? res.concurrency : [];
  const base = res.single || levels.find((l) => l.concurrency === 1) || levels[0] || (res.decode_tps !== undefined ? res : null);
  const results = base || levels.length ? {
    prefill_tps: base?.prefill_tps, decode_tps: base?.decode_tps, ttft_p50_s: base?.ttft_p50_s, ttft_p95_s: base?.ttft_p95_s, itl_p50_s: base?.itl_p50_s, e2e_p50_s: base?.e2e_p50_s,
    throughput: res.throughput || levels.map((l) => ({ concurrency: l.concurrency, tps: l.throughput_tps })), prefix_cache: res.prefix_cache } : null;
  return { ...r, status: r.state ?? r.status, results };
}

export const normKey = (k) => ({ ...k, last_used: k.last_used_at ?? k.last_used, key: k.secret ?? k.key });
export const normPrompt = (p) => ({ ...p, name: p.title ?? p.name, text: p.content ?? p.text });

// Conversation message content may be a JSON-encoded content-part array (the server stores multimodal turns as JSON text).
export function parseMessage(m) {
  let text = m.content, images = [];
  if (typeof text === 'string' && text.startsWith('[{')) {
    try {
      const parts = JSON.parse(text);
      if (Array.isArray(parts)) { images = parts.filter((p) => p.type === 'image_url').map((p) => p.image_url?.url).filter(Boolean); text = parts.filter((p) => p.type === 'text').map((p) => p.text).join('\n'); }
    } catch { /* plain text that happens to start with [{ */ }
  }
  return { role: m.role, text: typeof text === 'string' ? text : '', reasoning: m.reasoning || '', images, usage: m.usage || undefined };
}

// Usage rows for group_by=hour have ISO keys ("2026-09-23T14:00Z") -> 7x24 grid (Mon=0). Also accepts {dow,hour}.
export function toGrid(rows, { utc = false } = {}) {
  const g = Array.from({ length: 7 }, () => Array(24).fill(0));
  for (const r of rows) {
    let dow = r.dow, hour = r.hour;
    if (dow === undefined && typeof r.key === 'string') {
      const d = new Date(r.key);
      if (Number.isNaN(d.getTime())) continue;
      dow = ((utc ? d.getUTCDay() : d.getDay()) + 6) % 7; hour = utc ? d.getUTCHours() : d.getHours();
    }
    if (dow >= 0 && dow < 7 && hour >= 0 && hour < 24) g[dow][hour] += r.requests || 0;
  }
  return g;
}
