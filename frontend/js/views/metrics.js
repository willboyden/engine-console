import { EcView } from '../components/base.js';
import { h, clear, setTrustedHtml } from '../dom.js';
import { t } from '../i18n.js';
import { items, fmtClock, fmtMs, fmtCompact } from '../format.js';
import { normPoint } from '../adapt.js';
import { lineChart, legend } from '../charts.js';
import { card, select, field, emptyBox, errorBox, skeleton } from '../components/ui.js';

const WINDOWS = ['5m', '15m', '1h', '24h'];
// Each chart: series of [canonical metric key, label]; unit formatter for the y axis.
const CHARTS = [
  { id: 'tput', title: 'metrics.throughput', series: [['generation_tps', 'metrics.gen_tps'], ['prompt_tps', 'metrics.prompt_tps']], y: (v) => fmtCompact(v) },
  { id: 'lat', title: 'metrics.latency', series: [['ttft_p50_s', 'TTFT p50'], ['ttft_p95_s', 'TTFT p95'], ['itl_p50_s', 'ITL p50']], y: (v) => fmtMs(v) },
  { id: 'queue', title: 'metrics.queue', series: [['requests_running', 'metrics.running'], ['requests_waiting', 'metrics.waiting']], y: (v) => String(v) },
  { id: 'kv', title: 'metrics.kv', series: [['kv_cache_usage_pct', 'metrics.kv_usage'], ['prefix_cache_hit_pct', 'metrics.prefix_hit']], y: (v) => `${v}%`, yMax: 100 },
];

class EcMetrics extends EcView {
  setup() {
    this.sel = this.query?.instance || ''; this.win = '15m';
    this.controls = h('div', { class: 'row gap wrap bottom' });
    this.grid = h('div', { class: 'grid two' }, skeleton(4));
    this.append(h('h1', t('nav.metrics')), this.controls, this.grid);
    this.init();
  }
  async init() {
    try {
      this.insts = items(await this.api.instances());
      if (!this.sel || !this.insts.some((i) => i.id === this.sel)) this.sel = (this.insts.find((i) => i.state === 'ready') || this.insts[0])?.id || '';
    } catch (e) { clear(this.grid).append(errorBox(e, () => this.init())); return; }
    if (!this._alive) return;
    clear(this.controls).append(
      field(t('metrics.instance'), select(this.insts.map((i) => ({ value: i.id, label: i.name || i.id })), this.sel, (v) => { this.sel = v; this.load(); })),
      field(t('metrics.window'), select(WINDOWS, this.win, (v) => { this.win = v; this.load(); })));
    if (!this.insts.length) { clear(this.grid).append(emptyBox(t('metrics.no_instances'), t('metrics.no_instances_hint'))); return; }
    this.load(); this.every(5000, () => this.load(true));
  }
  async load(quiet) {
    try { this.data = await this.api.metrics(this.sel, this.win); this.err = null; } catch (e) { this.err = e; }
    if (!this._alive) return;
    if (this.err) { if (!quiet || !this.data) clear(this.grid).append(errorBox(this.err, () => this.load())); return; }
    const pts = (this.data.points || this.data.items || []).map(normPoint);
    if (!pts.length) { clear(this.grid).append(emptyBox(t('metrics.no_data'), t('metrics.no_data_hint'))); return; }
    const label = (k) => (k.includes('.') ? t(k) : k);
    clear(this.grid).append(...CHARTS.map((c) => {
      const series = c.series.map(([k, l]) => ({ name: label(l), points: pts.filter((p) => p[k] != null).map((p) => [p.t, p[k]]) }));
      const body = h('div');
      setTrustedHtml(body, lineChart(series, { xFormat: fmtClock, yFormat: c.y, yMax: c.yMax, ariaLabel: t(c.title) }) + legend(series));
      return card(t(c.title), body);
    }));
  }
}
customElements.define('ec-metrics', EcMetrics);
