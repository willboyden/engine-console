// Pure helpers for the fleet views (dashboard cards, host detail). No DOM and no i18n: reasons come back as
// {key, vars} descriptors that the view turns into text with t(). Shapes follow backend domain/fleet.py.
import { widths, systemSegments, engineMemory } from './memory.js';

const num = (v) => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const pos = (v) => (num(v) !== null && v > 0 ? v : 0);

// ---- link and policy state ----
export const LINK_STATES = ['connecting', 'up', 'stale', 'down', 'incompatible', 'disabled'];
const LINK_KIND = { up: 'ok', connecting: 'info', stale: 'warn', down: 'bad', incompatible: 'bad', disabled: 'muted' };
export const linkKind = (state) => LINK_KIND[state] || 'muted';
export const isUp = (h) => h?.link?.state === 'up';
/** A card still has samples to show (a stale host shows its last ones, dimmed). */
export const hasData = (h) => h?.link?.state === 'up' || h?.link?.state === 'stale';
export const policyMode = (h) => (h?.policy ? (h.policy.mode === 'manage' ? 'manage' : 'monitor') : null);
export const isRemote = (h) => !!h?.remote;

/** Why the console cannot change anything on this host right now, or null when it can. */
export function manageBlock(h) {
  if (!h) return { key: 'fleet.block_unknown' };
  const s = h.link?.state;
  if (s && s !== 'up') return { key: `fleet.block_${LINK_STATES.includes(s) ? s : 'down'}`, vars: { host: h.label || h.id } };
  if (policyMode(h) !== 'manage') return { key: 'fleet.block_monitor', vars: { host: h.label || h.id } };
  if (h.docker && h.docker.available === false) return { key: 'fleet.block_docker', vars: { host: h.label || h.id, error: h.docker.error || '' } };
  return null;
}
export const canManage = (h) => manageBlock(h) === null;

/** Seconds until the hub retries a dropped link, counted down client-side from when the summary arrived. */
export function retryCountdown(link, receivedAtMs, nowMs = Date.now()) {
  if (!link || num(link.retry_in_s) === null) return null;
  return Math.max(0, Math.ceil(link.retry_in_s - (nowMs - receivedAtMs) / 1000));
}
/** Seconds since the last message from a node (stale hosts). */
export function silentFor(link, nowMs = Date.now()) {
  if (!link || num(link.last_seen) === null) return null;
  return Math.max(0, Math.round(nowMs / 1000 - link.last_seen));
}
/** True for the "reached with an unrestricted key" case: a REMOTE node says its ssh key is not limited to the node command.
 * The local node runs as a subprocess, not over ssh, so it always reports forced_cmd false and is never warned about (the
 * hub applies the same rule: services/fleet.py only notes it for hosts with `ssh`). */
export const unrestrictedKey = (h) => h?.remote === true && h?.node?.forced_cmd === false;
/** The link's own notes minus the one the UI already explains in its own words (so the unrestricted key is not warned about twice). */
export const visibleNotes = (h) => (h?.link?.notes || []).filter((n) => !(unrestrictedKey(h) && /unrestricted/i.test(String(n))));
/** Rootful docker means the docker group is root-equivalent on that machine: an explained warning in the UI. */
export const rootfulDocker = (h) => h?.docker?.available === true && h.docker.rootless === false;

// ---- instances by machine ----
export function groupByHost(instances, hosts) {
  const order = hosts.map((h) => h.id);
  const by = new Map(order.map((id) => [id, []]));
  for (const i of instances || []) {
    const id = i.host_id || 'local';
    if (!by.has(id)) by.set(id, []);
    by.get(id).push(i);
  }
  return [...by.entries()].filter(([, list]) => list.length).map(([id, list]) => {
    const host = hosts.find((h) => h.id === id) || null;
    return { id, host, label: host?.label || list[0]?.host_label || id, instances: list };
  });
}

// ---- colours: one per instance on a machine, shared by its VRAM and RAM bars ----
export const PALETTE_COLORS = 6;
export function engineColors(instances) {
  const ids = [...new Set((instances || []).map((i) => i.id))].sort();
  return new Map(ids.map((id, k) => [id, `m${(k % PALETTE_COLORS) + 1}`]));
}
export const instanceNames = (instances) => new Map((instances || []).map((i) => [i.id, i.name || i.id]));

/**
 * One GPU's VRAM split by engine. Processes map to console instances through GpuProc.instance_id; everything else is
 * "other". Unified-memory GPUs have no VRAM of their own (they share host RAM), so there is no bar to draw.
 * Returns {unified, known, total, used, free, segments:[{key,label,gib,cls,kind,pct}], otherNames}.
 */
export function gpuSegments(gpu, { names = new Map(), colors = new Map() } = {}) {
  const total = pos(gpu?.total_gib);
  if (gpu?.unified_memory) return { unified: true, known: false, total, used: null, free: null, segments: [], otherNames: [] };
  const by = new Map();
  let otherKnown = 0;
  const otherNames = [];
  for (const p of gpu?.procs || []) {
    const g = num(p.used_gib);
    if (p.instance_id) by.set(p.instance_id, (by.get(p.instance_id) || 0) + (g ?? 0));
    else { otherKnown += g ?? 0; otherNames.push(p.container || p.name || `pid ${p.pid}`); }
  }
  const engines = [...by.entries()].filter(([, g]) => g > 0).sort((a, b) => b[1] - a[1])
    .map(([id, gib]) => ({ key: id, label: names.get(id) || id, gib, cls: colors.get(id) || 'm7', kind: 'engine' }));
  const engineSum = engines.reduce((a, e) => a + e.gib, 0);
  const driverUsed = num(gpu?.used_gib) ?? (num(gpu?.free_gib) !== null ? total - gpu.free_gib : null);
  const known = driverUsed !== null;
  const used = Math.min(total || Infinity, Math.max(0, driverUsed ?? engineSum + otherKnown));
  const other = Math.max(0, used - engineSum);
  const free = Math.max(0, total - used);
  const items = [...engines, { key: 'other', label: null, gib: other, cls: 'm-other', kind: 'other' }, { key: 'free', label: null, gib: free, cls: 'm-avail', kind: 'free' }].filter((s) => s.gib > 0);
  const w = widths(items.map((s) => s.gib), total || items.reduce((a, s) => a + s.gib, 0));
  return { unified: false, known, total, used, free, segments: items.map((s, i) => ({ ...s, pct: w[i] })), otherNames };
}

/** Host RAM split by engine (Instance.host_memory), then other used, then available. Null without memory data. */
export function ramSegments(mem, instances, colors = new Map(), names = new Map()) {
  if (!mem || num(mem.total_gib) === null) return null;
  const running = (instances || []).filter((i) => i.state !== 'stopped');
  const engines = engineMemory(running).map((e) => ({ ...e, name: names.get(e.id) || e.name, cls: colors.get(e.id) }));
  const out = systemSegments({ engines, total: mem.total_gib, used: mem.used_gib, available: mem.available_gib });
  return { total: mem.total_gib, used: num(mem.used_gib), available: num(mem.available_gib), segments: out.segments };
}

// ---- disks ----
export function diskUsage(m) {
  const size = num(m?.size_bytes), free = num(m?.free_bytes);
  const used = num(m?.used_bytes) ?? (size !== null && free !== null ? size - free : null);
  return { size, used, free, pct: size > 0 && used !== null ? Math.min(100, (used / size) * 100) : null };
}
export const lowDisk = (m, minFreeFraction = 0.1) => { const d = diskUsage(m); return d.size > 0 && d.free !== null && d.free / d.size < minFreeFraction; };
/** Writable model roots with the free space of the drive they live on, biggest first. */
export function rootChoices(host, { kinds } = {}) {
  return (host?.roots || []).filter((r) => r.writable && (!kinds || kinds.includes(r.kind)))
    .map((r) => ({ root: r, free: num(r.mount?.free_bytes), label: r.label || r.id }))
    .sort((a, b) => (b.free ?? -1) - (a.free ?? -1));
}

// ---- fleet totals ----
export function fleetSummary(hosts) {
  const out = { total: 0, up: 0, stale: 0, down: 0, other: 0, enginesRunning: 0, enginesTotal: 0, models: 0 };
  for (const h of hosts || []) {
    out.total++;
    const s = h.link?.state;
    if (s === 'up') out.up++; else if (s === 'stale') out.stale++; else if (s === 'down') out.down++; else out.other++;
    out.enginesRunning += h.counts?.engines_running || 0; out.enginesTotal += h.counts?.engines_total || 0; out.models += h.counts?.models || 0;
  }
  return out;
}

/**
 * A short fingerprint of what a host card shows, rounded so sensor jitter does not trigger a repaint on every sample.
 */
export function hostSignature(h, extra = '') {
  const g = (h.gpus || []).map((x) => [x.index, Math.round((x.used_gib ?? -1) * 2) / 2, Math.round((x.util_pct ?? -1) / 5), Math.round((x.temp_c ?? 0) / 3), Math.round((x.power_w ?? 0) / 20),
    (x.procs || []).map((p) => `${p.instance_id || p.pid}:${Math.round(p.used_gib ?? -1)}`).join(',')]);
  const m = h.memory ? [Math.round(h.memory.used_gib), Math.round(h.memory.swap_used_gib)] : null;
  // The retry countdown is repainted by its own timer, so only whether a retry is pending belongs in the signature.
  return JSON.stringify([h.link?.state, h.link?.error, h.link?.retry_in_s == null ? null : 'retry', h.link?.notes, h.policy?.mode, h.counts, (h.alerts || []).map((a) => a.code + a.level), g, m, extra]);
}
