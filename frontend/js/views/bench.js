import { EcView } from '../components/base.js';
import { h, clear, setTrustedHtml } from '../dom.js';
import { t } from '../i18n.js';
import { items, fmtTps, fmtMs, timeAgo } from '../format.js';
import { normBench, normInstance } from '../adapt.js';
import { openModal } from '../components/dialog.js';
import { lineChart, legend, chartLabel } from '../charts.js';
import { btn, card, select, field, emptyBox, errorBox, skeleton, stateBadge, progress, setProgress, pageHead } from '../components/ui.js';
import { toast, toastError } from '../components/toast.js';
import { instanceLabel, scriptLabelMap } from '../controls.js';
import { loadScripts } from '../components/script-actions.js';

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
    // "Custom…" suite fields; clamped server-side, hinted in the form.
    this.custom = { levels: '1, 4', prompt: '512', max: '128', rounds: '1', prefix: false };
    this.controls = h('div', { class: 'row gap wrap bottom' });
    this.customHost = h('div', { class: 'stack-v', hidden: true });
    this.live = h('div');
    this.runsHost = h('div', {}, skeleton(3));
    this.compare = h('div');
    this.append(pageHead({ page: 'bench', title: t('nav.bench') }), card(t('bench.run'), h('div', { class: 'stack-v' }, this.controls, this.customHost, this.live)), card(t('bench.runs'), this.runsHost), this.compare);
    this.init();
  }
  async init() {
    try {
      const [all, scripts] = await Promise.all([this.api.instances().then(items), loadScripts()]); this.labels = scriptLabelMap(scripts); this.insts = all.map(normInstance).filter((i) => i.state === 'ready');
    } catch (e) { clear(this.controls).append(errorBox(e, () => this.init())); return; }
    if (!this._alive) return;
    this.sel = this.query?.instance && this.insts.some((i) => i.id === this.query.instance) ? this.query.instance : this.insts[0]?.id || '';
    this.runBtn = btn(t('bench.start'), { kind: 'primary', icon: 'play', disabled: !this.sel, onClick: () => this.start() });
    const suiteOptions = ['quick', 'standard', 'latency', 'throughput', 'burst', 'full', 'quality', 'custom'];
    clear(this.controls).append(
      field(t('bench.instance'), select(this.insts.length ? this.insts.map((i) => ({ value: i.id, label: `${instanceLabel(i, this.labels)} · ${i.host_label || i.host_id}${i.external ? ` (${t('bench.external_tag')})` : ''}` })) : [{ value: '', label: t('bench.none_ready') }], this.sel, (v) => { this.sel = v; })),
      field(t('bench.suite'), select(suiteOptions.map((v) => ({ value: v, label: t(`bench.suite_${v}`) })), this.suite, (v) => { this.suite = v; this.customHost.hidden = v !== 'custom'; })), this.runBtn);
    this.paintCustom();
    this.loadRuns();
  }
  // The "Custom…" form: a preset name is one of the seven above; custom sends a dict the backend
  // merges over "quick" and clamps (1-8 levels, 1-64 each, <=32768 in, <=4096 out, 1-5 rounds).
  paintCustom() {
    const num = (key, attrs = {}) => h('input', { type: 'number', min: 1, step: 1, value: this.custom[key],
      onChange: (e) => { this.custom[key] = e.target.value; }, ...attrs });
    const levels = h('input', { type: 'text', value: this.custom.levels, inputmode: 'numeric', size: 12,
      onChange: (e) => { this.custom.levels = e.target.value; } });
    const prefix = h('label', { class: 'check' }, h('input', { type: 'checkbox', checked: this.custom.prefix,
      onChange: (e) => { this.custom.prefix = e.target.checked; } }), h('span', t('bench.custom_prefix')));
    clear(this.customHost).append(h('div', { class: 'row gap wrap' },
      field(t('bench.custom_levels'), levels, t('bench.custom_levels_hint')),
      field(t('bench.custom_prompt'), num('prompt'), t('bench.custom_prompt_hint')),
      field(t('bench.custom_max'), num('max'), t('bench.custom_max_hint')),
      field(t('bench.custom_rounds'), num('rounds', { max: 5 }), t('bench.custom_rounds_hint')),
      prefix));
  }
  // 'custom' -> the dict from the form; a preset name stays a string. The backend validates both.
  suiteSpec() {
    if (this.suite !== 'custom') return this.suite;
    const levels = (this.custom.levels || '').split(/[^0-9]+/).filter(Boolean).map(Number).filter((n) => n > 0);
    return { concurrency: levels, prompt_tokens: Number(this.custom.prompt), max_tokens: Number(this.custom.max),
      rounds: Number(this.custom.rounds), prefix_cache: !!this.custom.prefix };
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
      const run = await this.api.runBench({ instance_id: this.sel, suite: this.suiteSpec(), ...(external ? { confirm_external: true } : {}) });
      toast(t('bench.started'), { kind: 'ok', timeout: 2000 });
      const bar = progress(0, { label: t('bench.progress') }), phase = h('span');
      clear(this.live).append(h('div', { class: 'row between small' }, phase, h('span', { class: 'muted' }, run.id)), bar);
      // Backend SSE: `snapshot` (the run), then `progress` events {event: level_start|level_done|
      // scenario_done|done, ...}. Load runs step per concurrency level; quality runs per scenario.
      const quality = run.suite?.mode === 'quality';
      const total = quality
        ? (run.suite?.only?.length || run.suite?.scenarios || 0) * (run.suite?.repeats || 1)
        : (run.suite?.concurrency || []).length || 1;
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
          if (d.event === 'scenario_done') {
            done++; setProgress(bar, (done / total) * 100);
            phase.textContent = t('bench.quality_progress', { n: done, total, scenario: d.scenario, rep: d.rep });
          }
          if (d.event === 'done') finish(d.state);
        }, onClose: () => finish('completed') });
    } catch (e) { toastError(e); this.runBtn.disabled = !this.sel; }
  }
  suiteCell(r) {
    if (typeof r.suite === 'string') return r.suite;
    if (r.suite?.mode === 'quality') {
      const reps = r.suite?.repeats || 1;
      return reps > 1 ? `${t('bench.suite_quality')} ×${reps}` : t('bench.suite_quality');
    }
    return (r.suite?.concurrency || []).map((c) => `c${c}`).join(' ');
  }
  scoreCell(r) {
    if (r.results?.mode !== 'quality') return '–';
    const score = r.results.score;
    const runs = `${r.results.completed_runs ?? 0}/${r.results.runs ?? 0}`;
    return score == null ? runs : `${score}% · ${runs}`;
  }
  async loadRuns() {
    try { this.runs = items(await this.api.benches()).map(normBench); } catch (e) { if (this._alive) clear(this.runsHost).append(errorBox(e, () => this.loadRuns())); return; }
    if (!this._alive) return;
    if (!this.runs.length) { clear(this.runsHost).append(emptyBox(t('bench.empty'), t('bench.empty_hint'))); return; }
    clear(this.runsHost).append(h('div', { class: 'tablewrap' }, h('table',
      h('thead', h('tr', h('th', { scope: 'col' }, t('bench.compare')), ['bench.when', 'bench.model', 'bench.suite', 'inst.state', 'bench.score'].map((k) => h('th', { scope: 'col' }, t(k))), METRICS.slice(0, 3).map(([, l]) => h('th', { scope: 'col' }, t(l))))),
      h('tbody', this.runs.map((r) => h('tr', {},
        h('td', h('input', { type: 'checkbox', checked: this.picked.has(r.id), 'aria-label': t('bench.compare_named', { id: r.id }), disabled: r.status !== 'completed', onChange: (e) => { e.target.checked ? this.picked.add(r.id) : this.picked.delete(r.id); this.paintCompare(); } })),
        h('td', r.created_at ? timeAgo(r.created_at) : '–'), h('td', r.repo_id || r.instance_id), h('td', this.suiteCell(r)), h('td', stateBadge(r.status)),
        h('td', { class: 'num' }, this.scoreCell(r)),
        METRICS.slice(0, 3).map(([k, , f]) => h('td', { class: 'num' }, r.results ? f(r.results[k]) : '–'))))))));
    this.paintCompare();
  }
  // A picked quality run's per-scenario breakdown: one table per run, failed criteria spelled out.
  qualityCard(r) {
    const name = (x) => `${x.repo_id || x.instance_id} · ${x.engine || ''} · ${x.id.slice(0, 6)}`;
    const verdict = (ok) => h('span', { class: ok ? 'ok' : 'bad' }, ok ? '✓' : '✗');
    const rows = (r.results.scenarios || []).map((s) => {
      const failed = (s.criteria || []).filter((c) => !c.passed).map((c) => `${c.kind}: ${c.detail}`);
      return h('tr', {},
        h('td', s.id), h('td', { class: 'num' }, String(s.rep || 1)), h('td', verdict(s.passed)),
        h('td', { class: 'num' }, `${s.rounds_used ?? 0}/${s.max_rounds ?? '?'}`),
        h('td', { class: 'num' }, `${s.criteria_passed ?? 0}/${s.criteria_total ?? 0}`),
        h('td', failed.length ? h('ul', { class: 'note' }, failed.map((f) => h('li', f))) : ''));
    });
    return card(`${t('bench.quality_details')} — ${name(r)}`, h('div', { class: 'stack-v' },
      h('p', { class: 'note' }, t('bench.quality_summary', { score: r.results.score ?? '–', completed: r.results.completed_runs ?? 0, runs: r.results.runs ?? 0 })),
      h('div', { class: 'tablewrap' }, h('table',
        h('thead', h('tr', ['bench.quality_scenario', 'bench.quality_rep', 'inst.state', 'bench.quality_rounds', 'bench.quality_criteria', 'bench.quality_failed'].map((k) => h('th', { scope: 'col' }, t(k))))),
        h('tbody', rows)))));
  }
  paintCompare() {
    const sel = (this.runs || []).filter((r) => this.picked.has(r.id) && r.results);
    if (!sel.length) { clear(this.compare); return; }
    const name = (r) => `${r.repo_id || r.instance_id} · ${r.engine || ''} · ${r.id.slice(0, 6)}`;
    const load = sel.filter((r) => r.results.mode !== 'quality');
    const qual = sel.filter((r) => r.results.mode === 'quality');
    clear(this.compare);
    if (load.length) {
      const rows = METRICS.map(([k, l, f, dir]) => {
        const best = bestIndex(load.map((r) => r.results[k]), dir);
        return h('tr', h('th', { scope: 'row' }, t(l)), load.map((r, i) => h('td', { class: `num${i === best ? ' best' : ''}` }, f(r.results[k]), i === best ? h('span', { class: 'sr-only' }, ` (${t('bench.best')})`) : null)));
      });
      const series = load.map((r) => ({ name: name(r), points: (r.results.throughput || []).map((p) => [p.concurrency, p.tps]) }));
      const chart = h('div'); setTrustedHtml(chart, lineChart(series, { xFormat: (v) => `c=${v}`, yFormat: (v) => String(Math.round(v)), ariaLabel: chartLabel(t('bench.tput_chart'), series, { yFormat: (v) => String(Math.round(v)), template: t('chart.series_summary') }), noData: t('chart.no_data') }) + legend(series));
      this.compare.append(card(t('bench.comparison'), h('div', { class: 'stack-v' },
        h('div', { class: 'tablewrap' }, h('table', h('thead', h('tr', h('th', ''), load.map((r) => h('th', { scope: 'col' }, name(r))))), h('tbody', rows))),
        h('h3', t('bench.tput_vs_conc')), chart)));
    }
    qual.forEach((r) => this.compare.append(this.qualityCard(r)));
  }
}
customElements.define('ec-bench', EcBench);
