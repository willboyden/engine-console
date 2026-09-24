// Stacked per-GPU bar: weights / KV / activations / graphs / overhead against the budget marker.
import { h } from '../dom.js';
import { t } from '../i18n.js';
import { fmtGiB } from '../format.js';
import { SEGMENTS, verdictMeta, normalizeReport, compatLevel } from '../fit-format.js';
import { badge } from './ui.js';

export function fitMeter(rawReport, { gpuTotals = [], loading = false, error = null } = {}) {
  const r = normalizeReport(rawReport);
  const root = h('div', { class: `fit-meter${loading ? ' is-loading' : ''}`, 'aria-live': 'polite' });
  if (error) { root.append(h('div', { class: 'state error', role: 'alert' }, error.message || String(error))); return root; }
  if (!r) { root.append(h('p', { class: 'hint' }, t('fit.empty'))); return root; }
  const vm = verdictMeta(r.verdict);
  root.append(h('div', { class: 'row gap between' },
    h('div', { class: 'row gap' }, badge(t(vm.label), vm.cls), badge(t('fit.confidence', { level: r.confidence }), 'muted')),
    r.tpRequired ? badge(t('fit.tp', { n: r.tpRequired }), 'info') : null));
  for (const g of r.gpus) {
    const cap = g.capacity || gpuTotals[g.id] || Math.max(g.budget, g.total) * 1.1;
    const scale = Math.max(cap, g.total);
    const bar = h('div', { class: 'stack', role: 'img', 'aria-label': t('fit.gpu_aria', { id: g.id, used: g.total.toFixed(1), budget: g.budget.toFixed(1) }) });
    for (const s of SEGMENTS) {
      if (!(g[s.key] > 0)) continue;
      const seg = h('div', { class: `seg ${s.cls}`, title: `${t(s.label)}: ${fmtGiB(g[s.key])}` });
      seg.style.width = `${(g[s.key] / scale) * 100}%`;
      bar.append(seg);
    }
    const mark = h('div', { class: 'budget-mark', title: t('fit.budget_tip', { gib: g.budget.toFixed(1) }) });
    mark.style.left = `${(g.budget / scale) * 100}%`;
    bar.append(mark);
    root.append(h('div', { class: 'gpu-row' },
      h('div', { class: 'row between' }, h('strong', t('fit.gpu', { id: g.id })),
        h('span', { class: `num ${g.total > g.budget ? 'bad-text' : ''}` }, `${g.total.toFixed(1)} / ${g.budget.toFixed(1)} GiB`)),
      bar));
  }
  root.append(h('ul', { class: 'legend' }, SEGMENTS.map((s) => h('li', h('i', { class: `sw ${s.cls}` }), t(s.label)))));
  const facts = [];
  if (r.maxContext != null) facts.push(h('span', t('fit.max_ctx', { n: Number(r.maxContext).toLocaleString('en-US') })));
  if (r.maxConcurrency != null) facts.push(h('span', t('fit.max_conc', { n: r.maxConcurrency })));
  if (facts.length) root.append(h('div', { class: 'row gap wrap muted small' }, facts));
  if (r.fitsIfStop) root.append(h('p', { class: 'note info' }, t('fit.if_stop', { what: [].concat(r.fitsIfStop).join(', ') })));
  const lvl = compatLevel(r.compat);
  for (const c of r.compat) if (c.level !== 'ok') root.append(h('p', { class: `note ${c.level === 'block' ? 'bad' : 'warn'}`, role: 'note' }, h('strong', c.level === 'block' ? t('fit.blocked') : t('fit.warning')), ' ', c.message));
  for (const n of r.notes) root.append(h('p', { class: 'note muted' }, n));
  root.dataset.compat = lvl;
  return root;
}
