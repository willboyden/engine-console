import { EcView } from '../components/base.js';
import { h, clear, setTrustedHtml } from '../dom.js';
import { t } from '../i18n.js';
import { items, fmtTps, fmtMs, timeAgo } from '../format.js';
import { normBench, normInstance } from '../adapt.js';
import { openModal } from '../components/dialog.js';
import { lineChart, legend } from '../charts.js';
import { btn, card, select, field, emptyBox, errorBox, skeleton, stateBadge, progress, setProgress } from '../components/ui.js';
import { toast, toastError } from '../components/toast.js';

// metric key, label key, formatter, "better" direction
const METRICS = [
  ['decode_tps', 'bench.decode', fmtTps, 1], ['prefill_tps', 'bench.prefill', fmtTps, 1],
  ['ttft_p50_s', 'bench.ttft50', fmtMs, -1], ['ttft_p95_s', 'bench.ttft95', fmtMs, -1],
  ['itl_p50_s', 'bench.itl', fmtMs, -1], ['e2e_p50_s', 'bench.e2e', fmtMs, -1],
];
export function bestIndex(values, dir) {
  const v = values.map((x) => (typeof x === 'number' ? x : NaN));
  const ok = v.filter((x) => !Number.isNaN(x));
  if (ok.length < 2) return -1;
  return v.indexOf(dir > 0 ? Math.max(...ok) : Math.min(...ok));
}

class EcBench extends EcView {
  setup() {
    this.picked = new Set(); this.suite = 'quick';
    this.controls = h('div', { class: 'row gap wrap bottom' });
    this.live = h('div');
    this.runsHost = h('div', {}, skeleton(3));
    this.compare = h('div');
    this.append(h('h1', t('nav.bench')), card(t('bench.run'), h('div', { class: 'stack-v' }, this.controls, this.live)), card(t('bench.runs'), this.runsHost), this.compare);
    this.init();
  }
  async init() {
    try {
      this.insts = items(await this.api.instances()).map(normInstance).filter((i) => i.state === 'ready');
    } catch (e) { clear(this.controls).append(errorBox(e, () => this.init())); return; }
    if (!this._alive) return;
    this.sel = this.query?.instance && this.insts.some((i) => i.id === this.query.instance) ? this.query.instance : this.insts[0]?.id || '';
    this.runBtn = btn(t('bench.start'), { kind: 'primary', icon: 'play', disabled: !this.sel, onClick: () => this.start() });
    clear(this.controls).append(
      field(t('bench.instance'), select(this.insts.length ? this.insts.map((i) => ({ value: i.id, label: `${i.name || i.id}${i.external ? ` (${t('bench.external_tag')})` : ''}` })) : [{ value: '', label: t('bench.none_ready') }], this.sel, (v) => { this.sel = v; })),
      field(t('bench.suite'), select(['quick', 'standard'].map((v) => ({ value: v, label: t(`bench.suite_${v}`) })), this.suite, (v) => { this.suite = v; })), this.runBtn);
    this.loadRuns();
  }
  // Load on an engine we don't own needs explicit consent; the backend then requires confirm_external: true.
  confirmExternal() {
    return new Promise((resolve) => {
      let ok = false;
      const m = openModal(t('bench.confirm_external_title'), h('div', { class: 'stack-v' }, h('p', t('bench.confirm_external')),
        h('div', { class: 'row gap end' }, btn(t('common.cancel'), { onClick: () => m.close() }), btn(t('bench.confirm_external_btn'), { kind: 'danger', onClick: () => { ok = true; m.close(); } }))), { onClose: () => resolve(ok) });
    });
  }
  async start() {
    const inst = this.insts.find((i) => i.id === this.sel);
    const external = !!inst?.external;
    if (external && !(await this.confirmExternal())) return;
    this.runBtn.disabled = true;
    try {
      const run = await this.api.runBench({ instance_id: this.sel, suite: this.suite, ...(external ? { confirm_external: true } : {}) });
      toast(t('bench.started'), { kind: 'ok', timeout: 2000 });
      const bar = progress(0, { label: t('bench.progress') }), phase = h('span');
      clear(this.live).append(h('div', { class: 'row between small' }, phase, h('span', { class: 'muted' }, run.id)), bar);
      // Backend SSE: `snapshot` (the run), then `progress` events {event: level_start|level_done|done, ...}.
      const total = (run.suite?.concurrency || []).length || 1;
      let done = 0, finished = false;
      const finish = (state, error) => {
        if (finished) return; finished = true;
        clear(this.live).append(h('p', { class: `note ${state === 'failed' ? 'bad' : 'ok'}` }, state === 'failed' ? (error || t('bench.failed')) : t('bench.finished')));
        this.runBtn.disabled = !this.sel; this.loadRuns();
      };
      this.stream(`/bench/${encodeURIComponent(run.id)}/stream`, { reconnect: false,
        onEvent: (ev) => {
          const d = ev.data || {};
          if (ev.event === 'snapshot' && d.state && d.state !== 'running') return finish(d.state, d.error);
          if (d.event === 'level_start') phase.textContent = t('bench.level', { n: d.concurrency });
          if (d.event === 'level_done') { done++; setProgress(bar, (done / total) * 100); }
          if (d.event === 'done') finish(d.state);
        }, onClose: () => finish('completed') });
    } catch (e) { toastError(e); this.runBtn.disabled = !this.sel; }
  }
  async loadRuns() {
    try { this.runs = items(await this.api.benches()).map(normBench); } catch (e) { if (this._alive) clear(this.runsHost).append(errorBox(e, () => this.loadRuns())); return; }
    if (!this._alive) return;
    if (!this.runs.length) { clear(this.runsHost).append(emptyBox(t('bench.empty'), t('bench.empty_hint'))); return; }
    clear(this.runsHost).append(h('div', { class: 'tablewrap' }, h('table',
      h('thead', h('tr', h('th', { scope: 'col' }, t('bench.compare')), ['bench.when', 'bench.model', 'bench.suite', 'inst.state'].map((k) => h('th', { scope: 'col' }, t(k))), METRICS.slice(0, 3).map(([, l]) => h('th', { scope: 'col' }, t(l))))),
      h('tbody', this.runs.map((r) => h('tr', {},
        h('td', h('input', { type: 'checkbox', checked: this.picked.has(r.id), 'aria-label': t('bench.compare_named', { id: r.id }), disabled: r.status !== 'completed', onChange: (e) => { e.target.checked ? this.picked.add(r.id) : this.picked.delete(r.id); this.paintCompare(); } })),
        h('td', r.created_at ? timeAgo(r.created_at) : '–'), h('td', r.repo_id || r.instance_id), h('td', typeof r.suite === 'string' ? r.suite : (r.suite?.concurrency || []).map((c) => `c${c}`).join(' ')), h('td', stateBadge(r.status)),
        METRICS.slice(0, 3).map(([k, , f]) => h('td', { class: 'num' }, r.results ? f(r.results[k]) : '–'))))))));
    this.paintCompare();
  }
  paintCompare() {
    const sel = (this.runs || []).filter((r) => this.picked.has(r.id) && r.results);
    if (!sel.length) { clear(this.compare); return; }
    const name = (r) => `${r.repo_id || r.instance_id} · ${r.engine || ''} · ${r.id.slice(0, 6)}`;
    const rows = METRICS.map(([k, l, f, dir]) => {
      const best = bestIndex(sel.map((r) => r.results[k]), dir);
      return h('tr', h('th', { scope: 'row' }, t(l)), sel.map((r, i) => h('td', { class: `num${i === best ? ' best' : ''}` }, f(r.results[k]), i === best ? h('span', { class: 'sr-only' }, ` (${t('bench.best')})`) : null)));
    });
    const series = sel.map((r) => ({ name: name(r), points: (r.results.throughput || []).map((p) => [p.concurrency, p.tps]) }));
    const chart = h('div'); setTrustedHtml(chart, lineChart(series, { xFormat: (v) => `c=${v}`, yFormat: (v) => String(Math.round(v)), ariaLabel: t('bench.tput_chart') }) + legend(series));
    clear(this.compare).append(card(t('bench.comparison'), h('div', { class: 'stack-v' },
      h('div', { class: 'tablewrap' }, h('table', h('thead', h('tr', h('th', ''), sel.map((r) => h('th', { scope: 'col' }, name(r))))), h('tbody', rows))),
      h('h3', t('bench.tput_vs_conc')), chart)));
  }
}
customElements.define('ec-bench', EcBench);
