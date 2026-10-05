// Segmented bars for the fleet views. Every bar is role=img with a full text alternative (aria-label) and a visible legend that
// repeats the numbers, so nothing is conveyed by colour alone. The maths lives in fleet.js / memory.js (pure, tested).
import { h } from '../dom.js';
import { t } from '../i18n.js';
import { fmtGiB, fmtBytes } from '../format.js';
import { diskUsage, lowDisk } from '../fleet.js';
import { memBar, memLegend } from './mem-bar.js';
import { progress, badge } from './ui.js';
import { withTip } from './info-tip.js';

const segLabel = (s) => s.label ?? t(`mem.seg_${s.key}`);
export const shortGpuName = (n) => String(n || '').replace(/^NVIDIA\s+/i, '');

/** One GPU's VRAM split by engine (model = fleet.gpuSegments). Unified-memory GPUs have no bar of their own. */
export function vramBlock(model, gpuLabel) {
  if (model.unified) return h('p', { class: 'unified-note small' }, t('fleet.unified_note', { total: fmtGiB(model.total, 0) }), ' ', withTip(h('span', { class: 'muted' }, t('fleet.unified_what')), { kind: 'term', key: 'unified_memory' }, t('fleet.unified_what')));
  if (!model.segments.length) return h('p', { class: 'muted small' }, t('fleet.vram_unknown'));
  const parts = model.segments.map((s) => `${segLabel(s)} ${fmtGiB(s.gib)}`).join(', ');
  const aria = t('fleet.vram_aria', { gpu: gpuLabel, used: fmtGiB(model.used), total: fmtGiB(model.total), parts });
  return h('div', { class: 'stack-v tight' }, memBar(model.segments, { ariaLabel: aria }), memLegend(model.segments));
}

/** Host RAM split by engine (model = fleet.ramSegments). */
export function ramBlock(ram, label) {
  if (!ram) return h('p', { class: 'muted small' }, t('mem.unavailable'));
  const parts = ram.segments.map((s) => `${segLabel(s)} ${fmtGiB(s.gib)}`).join(', ');
  const aria = t('fleet.ram_aria', { label, used: fmtGiB(ram.used), total: fmtGiB(ram.total), parts });
  return h('div', { class: 'stack-v tight' }, memBar(ram.segments, { ariaLabel: aria }), memLegend(ram.segments));
}

/** One drive: label, free space of the total, a used/free bar, and a marker for network storage. */
export function diskRow(m, { compact = false, hideLabel = false, hideExtras = false } = {}) {
  const d = diskUsage(m);
  const low = lowDisk(m);
  const text = d.size ? (compact ? t('fleet.disk_free_short', { free: fmtBytes(d.free, 0) }) : t('fleet.disk_free', { free: fmtBytes(d.free, 0), size: fmtBytes(d.size, 0) })) : '–';
  const aria = t('fleet.disk_aria', { label: m.label, free: fmtBytes(d.free, 0), size: fmtBytes(d.size, 0) });
  const bar = progress(d.pct ?? 0, { kind: low ? 'bad' : d.pct > 80 ? 'warn' : '', label: aria });
  const extras = hideExtras ? [] : [m.remote ? badge(t('fleet.disk_shared'), 'info', m.remote_host ? t('fleet.disk_shared_tip', { host: m.remote_host_id || m.remote_host }) : undefined) : null, m.readonly ? badge(t('fleet.disk_readonly'), 'muted') : null].filter(Boolean);
  return h('div', { class: `disk-row${compact ? ' compact' : ''}` },
    h('div', { class: 'disk-line' }, hideLabel ? null : h('span', { class: 'disk-label ellipsis', title: `${m.label}${d.size ? ` · ${fmtBytes(d.size, 0)}` : ''}` }, m.label), ...(compact ? extras : []), h('span', { class: `num small disk-free ${low ? 'bad-text' : 'muted'}` }, text)),
    bar,
    !compact && extras.length ? h('div', { class: 'row gap wrap small' }, ...extras) : null);
}

/** A metric label with its info button, e.g. statLabel('Temperature', 'gpu_temp_c'). */
export const statLabel = (text, metricKey) => withTip(h('span', { class: 'muted' }, text), { kind: 'metric', key: metricKey }, text);
