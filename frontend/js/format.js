// Pure formatting helpers (no DOM).
export const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
export const items = (x) => (Array.isArray(x) ? x : Array.isArray(x?.items) ? x.items : []);
const isNum = (n) => typeof n === 'number' && Number.isFinite(n);

export function fmtBytes(n, digits = 1) {
  if (!isNum(n)) return '–';
  const u = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
  let i = 0, v = Math.abs(n);
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
  return `${n < 0 ? '-' : ''}${i === 0 ? v.toFixed(0) : v.toFixed(digits)} ${u[i]}`;
}
export const fmtGiB = (g, d = 1) => (isNum(g) ? `${g.toFixed(d)} GiB` : '–');
export const fmtInt = (n) => (isNum(n) ? Math.round(n).toLocaleString('en-US') : '–');
export function fmtCompact(n) {
  if (!isNum(n)) return '–';
  const a = Math.abs(n);
  if (a >= 1e9) return `${(n / 1e9).toFixed(a >= 1e10 ? 0 : 1)}B`;
  if (a >= 1e6) return `${(n / 1e6).toFixed(a >= 1e7 ? 0 : 1)}M`;
  if (a >= 1e3) return `${(n / 1e3).toFixed(a >= 1e4 ? 0 : 1)}k`;
  return String(Math.round(n));
}
export const fmtPct = (p, d = 0) => (isNum(p) ? `${p.toFixed(d)}%` : '–');
export const fmtTps = (v) => (isNum(v) ? `${v >= 100 ? v.toFixed(0) : v.toFixed(1)} tok/s` : '–');
export const fmtMs = (s) => (isNum(s) ? (s < 1 ? `${(s * 1000).toFixed(0)} ms` : `${s.toFixed(2)} s`) : '–');
export function fmtDuration(sec) {
  if (!isNum(sec) || sec < 0) return '–';
  sec = Math.round(sec);
  const d = Math.floor(sec / 86400), h = Math.floor((sec % 86400) / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  if (d) return `${d}d ${h}h`;
  if (h) return `${h}h ${String(m).padStart(2, '0')}m`;
  if (m) return `${m}m ${String(s).padStart(2, '0')}s`;
  return `${s}s`;
}
export const fmtEta = (s) => (isNum(s) && s >= 0 ? fmtDuration(s) : '–');
export const fmtRate = (bps) => (isNum(bps) ? `${fmtBytes(bps)}/s` : '–');
export function timeAgo(ts, now = Date.now()) {
  const t = typeof ts === 'number' ? (ts < 1e12 ? ts * 1000 : ts) : Date.parse(ts);
  if (!isNum(t)) return '–';
  const s = Math.max(0, Math.round((now - t) / 1000));
  if (s < 5) return 'just now';
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}
export function fmtDate(ts) {
  const t = typeof ts === 'number' ? (ts < 1e12 ? ts * 1000 : ts) : Date.parse(ts);
  return isNum(t) ? new Date(t).toISOString().replace('T', ' ').slice(0, 19) : '–';
}
export function fmtClock(ts) {
  const t = ts < 1e12 ? ts * 1000 : ts;
  const d = new Date(t);
  return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
}
export function debounce(fn, ms) {
  let id;
  const d = (...a) => { clearTimeout(id); id = setTimeout(() => fn(...a), ms); };
  d.cancel = () => clearTimeout(id);
  return d;
}
export function csvCell(v) {
  const s = v == null ? '' : String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}
