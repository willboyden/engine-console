// Host-RAM bars. Every segment has a text alternative: the bar is role=img with a full aria-label, and a visible legend repeats the numbers.
import { h } from '../dom.js';
import { t } from '../i18n.js';
import { fmtGiB } from '../format.js';
import { hostSegments, systemSegments } from '../memory.js';
import { badge } from './ui.js';

const segLabel = (s) => s.label ?? t(`mem.seg_${s.key}`);

export function memBar(segments, { ariaLabel, small = false } = {}) {
  const bar = h('div', { class: `stack mem${small ? ' small' : ''}`, role: 'img', 'aria-label': ariaLabel });
  for (const s of segments) {
    const seg = h('div', { class: `seg ${s.cls}`, title: `${segLabel(s)}: ${fmtGiB(s.gib)}` });
    seg.style.width = `${s.pct}%`; // CSSOM, allowed under the strict CSP (no style attributes in markup)
    bar.append(seg);
  }
  return bar;
}

export function memLegend(segments) {
  return h('ul', { class: 'legend mem-legend' }, segments.map((s) => h('li', h('i', { class: `sw ${s.cls}` }), `${segLabel(s)} `, h('span', { class: 'num' }, fmtGiB(s.gib)))));
}

const hostAria = (segs, total) => `${t('mem.host_ram')} ${fmtGiB(total)}: ${segs.map((s) => `${segLabel(s)} ${fmtGiB(s.gib)}`).join(', ')}`;

/** "Host RAM 68.3 GiB" row with an anon/cache/shm bar; dash + tooltip when host_memory is null; "approx." for rss. */
export function hostRamRow(i, { legend = true } = {}) {
  const hs = hostSegments(i.host_memory);
  if (!hs) return h('div', { class: 'row between small', title: t('mem.unavailable_tip') }, h('span', { class: 'muted' }, t('mem.host_ram')), h('span', { class: 'num muted' }, '–'));
  const segs = hs.segments.map((s) => ({ ...s, label: t(`mem.hm_${s.key}`) }));
  return h('div', { class: 'stack-v tight host-ram' },
    h('div', { class: 'row between small' }, h('span', { class: 'muted' }, t('mem.host_ram')), h('span', { class: 'row gap' }, hs.approx ? badge(t('mem.approx'), 'muted', t('mem.approx_tip')) : null, h('strong', { class: 'num' }, fmtGiB(hs.total)))),
    memBar(segs, { small: true, ariaLabel: hostAria(segs, hs.total) }), legend ? memLegend(segs) : null);
}

export function systemBar(mem, engines) {
  const { segments } = systemSegments({ engines, total: mem.total, used: mem.used, available: mem.available });
  const aria = `${t('mem.system')}: ${segments.map((s) => `${segLabel(s)} ${fmtGiB(s.gib)}`).join(', ')}`;
  return { bar: memBar(segments, { ariaLabel: aria }), legend: memLegend(segments) };
}
