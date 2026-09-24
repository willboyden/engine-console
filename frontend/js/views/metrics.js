import { EcView } from '../components/base.js';
import { h, clear, setTrustedHtml } from '../dom.js';
import { t } from '../i18n.js';
import { items, fmtClock, fmtMs, fmtCompact } from '../format.js';
import { normPoint, normInstance } from '../adapt.js';
import { fmtDate } from '../format.js';
import { lineChart, legend, stackedChart } from '../charts.js';
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
  hostRamCard(pts) {
    const has = pts.some((p) => p.host_ram_gib != null);
    const body = h('div');
    if (!has) return card(t('metrics.host_ram'), h('p', { class: 'hint' }, t('metrics.no_host_ram')));
    const series = [['host_ram_anon_gib', 'anon', 'hm-anon'], ['host_ram_cache_gib', 'cache', 'hm-cache'], ['host_ram_shmem_gib', 'shm', 'hm-shm']].map(([k, n, cls]) => ({ name: t(`metrics.ram_${n}`), cls, points: pts.filter((p) => p[k] != null).map((p) => [p.t, p[k]]) }));
    setTrustedHtml(body, stackedChart(series, { xFormat: fmtClock, yFormat: (v) => `${v}`, ariaLabel: t('metrics.host_ram') }) + `<ul class="legend">${series.map((s) => `<li><i class="sw ${s.cls}"></i>${s.name}</li>`).join('')}</ul>`);
    return card(t('metrics.host_ram'), body);
  }
  sysMemCard() {
    const body = h('div');
    const pts = this.sys || [];
    const series = [['ram_used_gib', 'metrics.ram_used'], ['ram_available_gib', 'metrics.ram_avail'], ['shmem_gib', 'metrics.ram_shmem'], ['swap_used_gib', 'metrics.swap_used']].map(([k, l]) => ({ name: t(l), points: pts.filter((p) => p[k] != null).map((p) => [p.t, p[k]]) }));
    setTrustedHtml(body, lineChart(series, { xFormat: fmtClock, yFormat: (v) => `${v}`, ariaLabel: t('metrics.system_mem') }) + legend(series));
    return card(t('metrics.system_mem'), body);
  }
  setup() {
    this.sel = this.query?.instance || ''; this.win = '15m';
    this.controls = h('div', { class: 'row gap wrap bottom' });
    this.grid = h('div', { class: 'grid two' }, skeleton(4));
    this.note = h('div', { role: 'status' });
    this.append(h('h1', t('nav.metrics')), this.controls, this.note, this.grid);
    this.init();
  }
  async init() {
    try {
      this.insts = items(await this.api.instances()).map(normInstance);
      if (!this.sel || !this.insts.some((i) => i.id === this.sel)) this.sel = (this.insts.find((i) => i.state === 'ready') || this.insts[0])?.id || '';
    } catch (e) { clear(this.grid).append(errorBox(e, () => this.init())); return; }
    if (!this._alive) return;
    clear(this.controls).append(
      field(t('metrics.instance'), select(this.insts.map((i) => ({ value: i.id, label: `${i.name || i.id}${i.external ? ` (${t('metrics.external_tag')})` : ''}` })), this.sel, (v) => { this.sel = v; this.load(); })),
      field(t('metrics.window'), select(WINDOWS, this.win, (v) => { this.win = v; this.load(); })));
    if (!this.insts.length) { clear(this.grid).append(emptyBox(t('metrics.no_instances'), t('metrics.no_instances_hint'))); return; }
    this.load(); this.every(5000, () => this.load(true));
  }
  async load(quiet) {
    try { this.data = await this.api.metrics(this.sel, this.win); this.err = null; } catch (e) { this.err = e; }
    try { const s = await this.api.systemMetrics(this.win); this.sys = (s.points || []).map(normPoint); } catch { this.sys = null; }
    if (!this._alive) return;
    if (this.err) { if (!quiet || !this.data) clear(this.grid).append(errorBox(this.err, () => this.load())); return; }
    const pts = (this.data.points || this.data.items || []).map(normPoint);
    const inst = this.insts.find((i) => i.id === this.sel);
    const since = this.data.history_since ?? inst?.history_since;
    clear(this.note).append(...(since ? [h('p', { class: 'hint' }, t('metrics.history_since', { time: fmtDate(since) }))] : []));
    if (!pts.length && inst?.external) {
      clear(this.grid).append(emptyBox(inst.state === 'auth_required' ? t('metrics.auth_ext') : t('metrics.no_metrics_ext'), inst.state === 'auth_required' ? null : t('metrics.no_metrics_ext_hint')));
      return;
    }
    if (!pts.length) { clear(this.grid).append(emptyBox(t('metrics.no_data'), t('metrics.no_data_hint'))); return; }
    const label = (k) => (k.includes('.') ? t(k) : k);
    clear(this.grid).append(...CHARTS.map((c) => {
      const series = c.series.map(([k, l]) => ({ name: label(l), points: pts.filter((p) => p[k] != null).map((p) => [p.t, p[k]]) }));
      const body = h('div');
      setTrustedHtml(body, lineChart(series, { xFormat: fmtClock, yFormat: c.y, yMax: c.yMax, ariaLabel: t(c.title) }) + legend(series));
      return card(t(c.title), body);
    }), this.hostRamCard(pts), this.sysMemCard());
  }
}
customElements.define('ec-metrics', EcMetrics);
