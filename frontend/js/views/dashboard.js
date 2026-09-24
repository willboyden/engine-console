import { EcView } from '../components/base.js';
import { h, clear, setTrustedHtml } from '../dom.js';
import { t } from '../i18n.js';
import { items, fmtGiB, fmtPct, fmtTps } from '../format.js';
import { normInstance } from '../adapt.js';
import { sparkline } from '../charts.js';
import { badge, card, progress, stateBadge, emptyBox, errorBox, skeleton, btn } from '../components/ui.js';

const HIST = 90; // ~3 minutes at 2 s polling; kept module-level so sparklines survive navigation.
const history = new Map();
const push = (key, v) => { const a = history.get(key) || []; a.push(v); if (a.length > HIST) a.shift(); history.set(key, a); return a; };

class EcDashboard extends EcView {
  setup() {
    this.live = new Map(); // instance_id -> latest canonical metrics
    this.alerts = h('div', { class: 'alerts', role: 'region', 'aria-label': t('dash.alerts') });
    this.gpus = h('div', { class: 'grid gpus' }, skeleton(2));
    this.tiles = h('div', { class: 'grid tiles' }, skeleton(2));
    this.append(h('h1', { class: 'sr-only' }, t('nav.dashboard')), this.alerts,
      h('h2', { class: 'section' }, t('dash.gpus')), this.gpus,
      h('h2', { class: 'section' }, t('dash.instances')), this.tiles);
    this.hw = null; this.inst = [];
    this.tick();
    this.every(2000, () => this.tick());
    this.stream('/metrics/stream', { onEvent: (ev) => {
      const d = ev.data;
      if (d && d.instance_id) { this.live.set(d.instance_id, d.values || d.metrics || {}); this.paintTiles(); }
    } });
  }
  async tick() {
    const [hw, inst] = await Promise.allSettled([this.api.hardware(), this.api.instances()]);
    if (!this._alive) return;
    if (hw.status === 'fulfilled') { this.hw = hw.value; this.hwErr = null; } else this.hwErr = hw.reason;
    if (inst.status === 'fulfilled') { this.inst = items(inst.value).map(normInstance); this.instErr = null; } else this.instErr = inst.reason;
    this.paint();
  }
  paint() { this.paintAlerts(); this.paintGpus(); this.paintTiles(); }
  paintAlerts() {
    const out = [];
    for (const g of this.hw?.gpus || []) {
      const used = g.total_gib - g.free_gib;
      if (g.temp_c >= 85) out.push(h('div', { class: 'alert bad' }, t('dash.alert_temp', { gpu: g.index, temp: g.temp_c })));
      if (used / g.total_gib > 0.97) out.push(h('div', { class: 'alert warn' }, t('dash.alert_vram', { gpu: g.index, pct: Math.round((used / g.total_gib) * 100) })));
    }
    for (const i of this.inst.filter((x) => x.state === 'failed')) out.push(h('div', { class: 'alert bad' }, t('dash.alert_failed', { name: i.name || i.id }), ' ', h('a', { href: `#/instances/${encodeURIComponent(i.id)}` }, t('dash.view_logs'))));
    clear(this.alerts).append(...out);
  }
  paintGpus() {
    if (this.hwErr && !this.hw) { clear(this.gpus).append(errorBox(this.hwErr, () => this.tick())); return; }
    if (!this.hw) return;
    clear(this.gpus).append(...this.hw.gpus.map((g) => {
      const used = g.total_gib - g.free_gib, pct = (used / g.total_gib) * 100;
      const sp = (key, val, cls) => { const el = h('div', { class: 'spark-wrap' }); setTrustedHtml(el, sparkline(push(`${g.uuid || g.index}:${key}`, val), { cls })); return el; };
      return h('article', { class: 'card gpu' },
        h('header', { class: 'card-head' }, h('h3', `GPU ${g.index}`), h('span', { class: 'muted small' }, g.name)),
        h('div', { class: 'card-body' },
          h('div', { class: 'row between' }, h('span', t('dash.vram')), h('span', { class: 'num' }, `${fmtGiB(used)} / ${fmtGiB(g.total_gib, 0)}`)),
          progress(pct, { kind: pct > 92 ? 'bad' : pct > 80 ? 'warn' : '', label: t('dash.vram') }),
          sp('vram', pct, 's1'),
          h('div', { class: 'stats' },
            h('div', h('span', { class: 'muted' }, t('dash.util')), h('strong', { class: 'num' }, fmtPct(g.util_pct)), sp('util', g.util_pct, 's2')),
            h('div', h('span', { class: 'muted' }, t('dash.temp')), h('strong', { class: `num ${g.temp_c >= 85 ? 'bad-text' : ''}` }, `${Math.round(g.temp_c)} °C`), sp('temp', g.temp_c, 's3')),
            h('div', h('span', { class: 'muted' }, t('dash.power')), h('strong', { class: 'num' }, `${Math.round(g.power_w)} W`), sp('power', g.power_w, 's4')))));
    }));
  }
  paintTiles() {
    if (this.instErr && !this.inst.length) { clear(this.tiles).append(errorBox(this.instErr, () => this.tick())); return; }
    if (!this.hw && !this.inst.length) return;
    if (!this.inst.length) { clear(this.tiles).append(emptyBox(t('dash.no_instances'), t('dash.no_instances_hint'), btn(t('nav.launch'), { kind: 'primary', icon: 'launch', onClick: () => { location.hash = '#/launch'; } }))); return; }
    clear(this.tiles).append(...this.inst.map((i) => {
      const m = this.live.get(i.id) || {};
      return h('article', { class: 'card tile' },
        h('header', { class: 'card-head' }, h('h3', h('a', { href: `#/instances/${encodeURIComponent(i.id)}` }, i.name || i.id)), stateBadge(i.state)),
        h('div', { class: 'card-body' },
          h('div', { class: 'muted small ellipsis', title: i.repo_id }, `${i.engine} · ${i.repo_id}`),
          h('div', { class: 'row between' }, h('span', t('dash.gen_tps')), h('strong', { class: 'num big' }, i.state === 'ready' ? fmtTps(m.generation_tps) : '–')),
          h('div', { class: 'row between small' }, h('span', { class: 'muted' }, t('dash.kv_cache')), h('span', { class: 'num' }, i.state === 'ready' ? fmtPct(m.kv_cache_usage_pct) : '–')),
          progress(m.kv_cache_usage_pct ?? 0, { kind: (m.kv_cache_usage_pct ?? 0) > 90 ? 'warn' : '', label: t('dash.kv_cache') }),
          h('div', { class: 'row gap small muted' }, badge(`${t('dash.running')} ${m.requests_running ?? 0}`, 'muted'), badge(`${t('dash.waiting')} ${m.requests_waiting ?? 0}`, (m.requests_waiting ?? 0) > 0 ? 'warn' : 'muted'))));
    }));
  }
}
customElements.define('ec-dashboard', EcDashboard);
