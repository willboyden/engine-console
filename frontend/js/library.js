// Pure helpers for the Library page and the pickers built on it: filtering, display facts, copy chips and the state of
// each per-copy action. No DOM and no i18n: reasons are {key, vars} descriptors. Shapes follow backend domain/fleet.py
// (LibraryModel, LibraryCopy, HostSummary/HostDetail).
import { fmtCompact } from './format.js';
import { manageBlock, rootChoices } from './fleet.js';
import { refFor } from './model-ref.js';
import { quickFit } from './fit-format.js';

const GIB = 2 ** 30;
export const FORMAT_ORDER = ['hf', 'gguf', 'ollama', 'plain', 'other'];
export const ROOT_KINDS_BY_FORMAT = { hf: ['hf-home', 'hf-hub', 'auto'], gguf: ['gguf', 'auto'], plain: ['plain', 'auto'], ollama: ['ollama'], other: ['other'] };

/** Host ids that hold a copy or can read one over shared storage. */
export function hostsThatSee(model) {
  const s = new Set();
  for (const c of model?.copies || []) { s.add(c.host_id); for (const h of c.visible_on || []) s.add(h); }
  return s;
}

const haystack = (m) => [m.name, m.key, m.quant, m.arch, m.format, ...(m.copies || []).flatMap((c) => [c.host_label, c.host_id, c.root_label, c.drive?.label])].filter(Boolean).join(' ').toLowerCase();

export function filterLibrary(models, { host = '', format = '', q = '' } = {}) {
  const words = String(q || '').toLowerCase().split(/\s+/).filter(Boolean);
  return (models || []).filter((m) => {
    if (format && m.format !== format) return false;
    if (host && !hostsThatSee(m).has(host)) return false;
    if (words.length) { const hay = haystack(m); if (!words.every((w) => hay.includes(w))) return false; }
    return true;
  });
}

export function sortLibrary(models, by = 'name') {
  const list = [...(models || [])];
  if (by === 'size') return list.sort((a, b) => b.size_bytes - a.size_bytes || a.name.localeCompare(b.name));
  if (by === 'format') return list.sort((a, b) => FORMAT_ORDER.indexOf(a.format) - FORMAT_ORDER.indexOf(b.format) || a.name.localeCompare(b.name));
  return list.sort((a, b) => a.name.localeCompare(b.name, 'en', { sensitivity: 'base' }));
}

/** Plain-language size band for the wizard ("a medium model"). */
export function sizeBand(bytes) {
  const g = bytes / GIB;
  return g < 8 ? 'small' : g < 40 ? 'medium' : g < 100 ? 'large' : 'huge';
}
/** "35B" style parameter count, '' when unknown. */
export const paramsLabel = (n) => (typeof n === 'number' && n > 0 ? fmtCompact(n) : '');
export const inUse = (model) => (model?.copies || []).some((c) => (c.in_use_by || []).length > 0);
export const usedBy = (model) => [...new Set((model?.copies || []).flatMap((c) => c.in_use_by || []))];
export const totalCopies = (models) => (models || []).reduce((a, m) => a + (m.copies || []).length, 0);

/** What a copy chip shows: the machine and the drive it sits on, with a shared marker for network storage. */
export function copyChip(copy) {
  const others = (copy.visible_on || []).filter((h) => h !== copy.host_id);
  return {
    host: copy.host_label || copy.host_id,
    drive: copy.drive?.label || copy.root_label || copy.root_id,
    shared: others.length > 0,
    remote: !!copy.drive?.remote,
    visibleOn: copy.visible_on || [],
    alsoOn: others,
  };
}

// ---- per-copy actions ----
const hostById = (hosts) => (hosts instanceof Map ? hosts : new Map((hosts || []).map((h) => [h.id, h])));

/** Hosts a copy can be launched on (its own host plus every host that mounts the same storage), each with a block reason. */
export function launchHosts(copy, hosts) {
  const by = hostById(hosts);
  const ids = [...new Set([copy.host_id, ...(copy.visible_on || [])])];
  return ids.map((id) => { const host = by.get(id) || null; const block = manageBlock(host); return { id, host, label: host?.label || id, enabled: block === null, reason: block }; });
}

/** Whether the per-copy Delete button works, and if not why. Order matters: the most specific, fixable reason first. */
export function deleteState(copy, host, nameOf = (x) => x) {
  if ((copy.in_use_by || []).length) return { enabled: false, reason: { key: 'library.reason_in_use', vars: { names: copy.in_use_by.map(nameOf).join(', ') } } };
  if (!copy.writable) return { enabled: false, reason: { key: 'library.reason_read_only' } };
  if (host && host.link?.state && host.link.state !== 'up') return { enabled: false, reason: { key: 'library.reason_host_down', vars: { host: host.label || host.id } } };
  if (host?.policy && host.policy.allow_delete === false) return { enabled: false, reason: { key: 'library.reason_policy_delete', vars: { host: host.label || host.id } } };
  if (!copy.deletable) return { enabled: false, reason: { key: 'library.reason_not_deletable' } };
  return { enabled: true, reason: null };
}

/**
 * Machines a copy can be copied to: those that cannot already read it, with a reason when one is not an option. When the model
 * is given, a machine that already holds ANY copy of it (or reads one over shared storage) is not offered either.
 */
export function copyTargets(copy, hosts, model) {
  const seen = new Set([copy.host_id, ...(copy.visible_on || []), ...(model ? hostsThatSee(model) : [])]);
  return (hosts || []).filter((h) => !seen.has(h.id)).map((h) => {
    const block = manageBlock(h);
    return { id: h.id, host: h, label: h.label || h.id, enabled: block === null, reason: block };
  });
}
/** Writable roots on a (detailed) host that suit a model format, with the free bytes of the drive under each. */
export const targetRoots = (hostDetail, format) => rootChoices(hostDetail, { kinds: ROOT_KINDS_BY_FORMAT[format] || undefined });

export function copyActions(copy, hosts, model, nameOf) {
  const by = hostById(hosts);
  const lh = launchHosts(copy, by);
  const targets = copyTargets(copy, [...by.values()], model);
  const launchable = lh.some((x) => x.enabled);
  const copyable = targets.some((x) => x.enabled);
  return {
    launch: { enabled: launchable, reason: launchable ? null : (lh[0]?.reason || { key: 'fleet.block_unknown' }) },
    copy: { enabled: copyable, reason: copyable ? null : { key: targets.length ? 'library.reason_no_target_manage' : 'library.reason_everywhere' } },
    delete: deleteState(copy, by.get(copy.host_id), nameOf),
  };
}

// ---- which copy to use on which machine ----
/**
 * The copy a machine should load, and the ModelRef to send. A copy on shared storage belongs to the machine that owns the
 * disk; on another machine the same files sit under that machine's own root (the one whose `shared_from` names the
 * owner), so the root id is translated when the target's roots are known.
 */
export function resolveOnHost(model, hostId, hostDetails) {
  const copies = model?.copies || [];
  const own = copies.find((c) => c.host_id === hostId);
  if (own) return { copy: own, ref: refFor(model, own), via: 'own' };
  const shared = copies.find((c) => (c.visible_on || []).includes(hostId));
  if (!shared) return null;
  const ref = refFor(model, shared);
  if (!ref) return null;
  if (ref.root_id) {
    const detail = hostDetails instanceof Map ? hostDetails.get(hostId) : hostDetails?.[hostId];
    const mine = (detail?.roots || []).find((r) => r.shared_from === shared.host_id);
    if (mine) ref.root_id = mine.id;
  }
  return { copy: shared, ref, via: 'shared' };
}

/** Every machine as a launch candidate for one model: loadable ones first, then blocked, then those needing a copy. */
export function hostChoicesForModel(model, hosts, hostDetails) {
  const out = (hosts || []).map((host) => {
    const found = resolveOnHost(model, host.id, hostDetails);
    const block = manageBlock(host);
    // A machine that cannot take files anyway (down, monitor-only...) is told apart from one that only needs the model copied.
    if (!found) return { host, state: 'no_copy', enabled: false, reason: block || { key: 'library.reason_no_copy', vars: { host: host.label || host.id } }, copy: null, ref: null, via: null };
    if (block) return { host, state: 'blocked', enabled: false, reason: block, copy: found.copy, ref: found.ref, via: found.via };
    return { host, state: 'ok', enabled: true, reason: null, copy: found.copy, ref: found.ref, via: found.via };
  });
  const order = { ok: 0, blocked: 1, no_copy: 2 };
  return out.sort((a, b) => order[a.state] - order[b.state]);
}

/** A quick weights-only "might fit" test for a card, without calling the hub: any GPU pool on a host holding the model. */
export function quickFitHosts(model, hosts) {
  const names = [];
  for (const id of hostsThatSee(model)) {
    const h = (hosts || []).find((x) => x.id === id);
    if (!h || !h.gpus?.length) continue;
    const totals = h.gpus.map((g) => g.total_gib);
    const v = quickFit(model.size_bytes, totals).verdict;
    if (v === 'fits' || v === 'tight') names.push({ id, label: h.label || id, verdict: v });
  }
  return names;
}
