// Every normalisation rule of the differential harness, in one place. Each rule has an id (N1...) and a reason in
// backend/parity/corpus/NORMALISE.md. Only genuinely volatile values are touched: wall-clock times, generated ids and
// secrets, durations, live readings of the machine (RAM, CPU load, disk free space), and the per-backend temp dirs.
// One Normaliser per backend per run: generated ids are renamed in order of first appearance, so the same request
// sequence gives the same placeholders on both sides and a wrong id still shows up as a diff.
import { Num, mapValues } from './json.mjs';

// N1: wall-clock timestamps (epoch seconds). null stays null, so a missing timestamp is still a diff.
export const TIMESTAMP_KEYS = new Set([
  'created_at', 'updated_at', 'ts', 'last_used_at', 'started_at', 'finished_at', 'last_request_at', 'voted_at',
  'checked_at', 'last_seen', 'connected_since', 'scanned_at', 'history_since', 'mtime', 'atime', 'created',
  'boot_time', 'revoked_at', 'last_used',
]);
// N2: durations and rates measured by the clock.
export const DURATION_KEYS = new Set([
  'uptime_s', 'rtt_ms', 'skew_s', 'duration_s', 'retry_in_s', 'eta_s', 'speed_bps', 'avg_latency_ms', 'latency_ms',
]);
// N3: generated ids (services/common.py new_id: prefix + 12 hex).
const ID_RE = /\b(key|prof|conv|prm|match|dl|inst|bench|job)_[0-9a-f]{12}\b/g;
// N4: API key secrets (`ec_` + token_urlsafe(32), 43 chars) and their 8-character display prefix.
const SECRET_RE = /\bec_[A-Za-z0-9_-]{43}(?![A-Za-z0-9_-])/g;
const PREFIX_RE = /^ec_[A-Za-z0-9_-]{5}$/;
// N5: live readings of the machine the node runs on (host RAM, CPU load, disk usage, GPU-sharing RAM pool). Matched
// on the JSON path, so a model's own `size_bytes` or a GPU's `used_gib` from the fake nvidia-smi stay compared.
const LIVE_PATHS = [
  /\.memory\.(used|available|free|cached|shmem|swap_used)_gib$/, /\.host_ram_free_gib$/,
  /\.cpu\.(load1|load5|load15|util_pct)$/,
  /\.(drive|mount)\.(used_bytes|free_bytes)$/, /\.disks\[\d+\]\.(used_bytes|free_bytes)$/,
  /\.host_ram\.available_gib$/,
];
// N11: live host-RAM readings inside sentences (hostmem alerts, the fit's host_ram notes): "X GiB of host RAM",
// "X GiB of N GiB host RAM", "(X of N GiB)" of swap. Each backend's node reads /proc/meminfo at its own moment; the
// numbers in fields are N5, these are the same readings formatted into text. Other GiB figures stay compared.
const LIVE_TEXT = [
  [/\b\d+(?:\.\d+)? GiB of host RAM\b/g, '<live> GiB of host RAM'],
  [/\b\d+(?:\.\d+)? GiB of (\d+) GiB\b/g, '<live> GiB of $1 GiB'],
  [/\(\d+(?:\.\d+)? of (\d+) GiB\)/g, '(<live> of $1 GiB)'],
];
// N12: the hub's own release number, where the API states it: GET /api/v1/health `version` and the OpenAPI document's
// `info.version` (the Python reference is 0.4.x, the Rust hub 0.5.0 and later). A node's version is not touched.
const VERSION_PATHS = new Set(['$.version', '$.info.version']);
const VERSION_ROUTES = new Set(['GET /api/v1/health', 'GET /api/v1/openapi.json', 'HEAD /api/v1/openapi.json']);
// N6: the per-host system series (/metrics/system): how many 2-second samples exist depends on wall-clock time.
const POINT_SERIES_ROUTES = new Set(['GET /api/v1/metrics/system']);
// N9: /metrics series whose values are durations, creation times or live machine readings.
const PROM_VOLATILE = [/_created$/, /_bucket$/, /_sum$/, /^engine_console_host_(ram_used|ram_available|shmem|swap_used)_gib$/];

export class Normaliser {
  /** @param {string[]} roots per-backend directories (HOME, data dir) that must not leak into a comparison (N8)
   * @param {string|null} version this backend's own release number (N12) */
  constructor(roots = [], version = null) {
    this.version = version;
    this.ids = new Map();
    this.perPrefix = new Map();
    this.roots = roots.filter(Boolean).sort((a, b) => b.length - a.length);
  }

  /** N3: numbered per prefix (keys, profiles, instances... each count on their own), so an id one backend never
   * generated (a route the other answers 501 for) does not shift the numbers of every later id of another kind. */
  id(raw) {
    if (!this.ids.has(raw)) {
      const prefix = raw.split('_')[0];
      const n = (this.perPrefix.get(prefix) || 0) + 1;
      this.perPrefix.set(prefix, n);
      this.ids.set(raw, `<${prefix}:${n}>`);
    }
    return this.ids.get(raw);
  }

  /** N3/N4/N8/N11 inside any string. */
  text(s) {
    let out = s.replace(ID_RE, (m) => this.id(m)).replace(SECRET_RE, '<secret>');
    for (const [re, to] of LIVE_TEXT) out = out.replace(re, to);
    for (const r of this.roots) out = out.split(r).join('<backend-root>');
    return out;
  }

  /** A parsed JSON body (lib/json.mjs values) of the response to `route` ("METHOD /template"). */
  json(v, route = '') {
    return mapValues(v, (x, key, path) => {
      if (typeof x === 'string') {
        if (key === 'prefix' && PREFIX_RE.test(x)) return '<prefix>';
        if (this.version && x === this.version && VERSION_PATHS.has(path) && VERSION_ROUTES.has(route)) return '<hub-version>';
        return this.text(x);
      }
      if (x instanceof Num) {
        if (TIMESTAMP_KEYS.has(key)) return '<ts>';
        if (DURATION_KEYS.has(key)) return '<duration>';
        if (key === 't' && /\.points\[\d+\]\.t$|^\$\.t$/.test(path)) return '<ts>';
        if (LIVE_PATHS.some((re) => re.test(path))) return '<live>';
        return undefined;
      }
      if (key === 'points' && Array.isArray(x) && POINT_SERIES_ROUTES.has(route) && path === '$.points') return '<points>';
      if (key === 'values' && x && typeof x === 'object' && POINT_SERIES_ROUTES.has(route)) return '<values>';
      return undefined;
    });
  }

  /** Prometheus text exposition: one entry per sample line, sorted; values canonicalised as numbers (N9). */
  prometheus(text) {
    const out = [];
    for (const raw of text.split('\n')) {
      const line = raw.trimEnd();
      if (!line) continue;
      if (line.startsWith('#')) { out.push(this.text(line)); continue; }
      const m = /^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{.*\})?\s+(\S+)(\s+\S+)?$/.exec(line);
      if (!m) { out.push(`?${this.text(line)}`); continue; }
      const [, name, labels = '', value] = m;
      const v = PROM_VOLATILE.some((re) => re.test(name)) ? '<volatile>' : canonNumber(value);
      out.push(`${name}${this.text(labels)} ${v}`);
    }
    return out.sort();
  }
}

function canonNumber(v) {
  if (v === '+Inf' || v === '-Inf' || v === 'NaN') return v;
  const n = Number(v);
  return Number.isNaN(n) ? v : String(n);
}

/** Server-Sent Events: [{event, data}] with data parsed as JSON when it is JSON. */
export function parseSse(text) {
  const events = [];
  for (const block of text.split(/\r?\n\r?\n/)) {
    if (!block.trim()) continue;
    let event = null; const data = [];
    for (const line of block.split(/\r?\n/)) {
      if (line.startsWith(':')) continue;
      const i = line.indexOf(':');
      const field = i < 0 ? line : line.slice(0, i);
      let val = i < 0 ? '' : line.slice(i + 1);
      if (val.startsWith(' ')) val = val.slice(1);
      if (field === 'event') event = val;
      else if (field === 'data') data.push(val);
    }
    events.push({ event, data: data.join('\n') });
  }
  return events;
}
