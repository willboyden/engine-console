// Fleet overview: one card per machine (link, policy mode, GPUs and RAM split by engine, drives, alerts), then the engines
// grouped by machine with live throughput and KV-cache use. Live through /fleet/stream, with polling as the fallback.
import { EcView } from '../components/base.js';
import { h, clear } from '../dom.js';
import { t } from '../i18n.js';
import { items, fmtGiB, fmtPct, fmtTps } from '../format.js';
import { normInstance, normHost, listSignature } from '../adapt.js';
import { engineColors, instanceNames, groupByHost, fleetSummary, hostSignature, retryCountdown } from '../fleet.js';
import { instanceTitle, scriptLabelMap, splitInstances } from '../controls.js';
import { loadScripts } from '../components/script-actions.js';
import { availableSection, titleBlock } from '../components/instance-rows.js';
import { doInstanceAction } from '../components/instance-actions.js';
import { hostRamRow } from '../components/mem-bar.js';
import { hostCard } from '../components/host-card.js';
import { statLabel } from '../components/fleet-bars.js';
import { withTip, busyWithin } from '../components/info-tip.js';
import { badge, progress, stateBadge, managementBadge, linkBadge, emptyBox, errorBox, skeleton, btn, instanceFacts, reasonNote, pageHead } from '../components/ui.js';
import { toast, toastError } from '../components/toast.js';

const enc = encodeURIComponent;

class EcDashboard extends EcView {
  setup() {
    this.hosts = new Map(); this.order = []; this.rx = new Map(); this.details = new Map(); this.detailAt = new Map();
    this.cards = new Map(); this.sigs = new Map();
    this.inst = []; this.live = new Map(); this.streamLive = false; this.scripts = []; this.labels = new Map();
    this.summary = h('div', { class: 'fleet-summary row gap wrap', role: 'status' });
    this.alerts = h('div', { class: 'alerts', role: 'region', 'aria-label': t('dash.alerts') });
    this.grid = h('div', { class: 'grid hosts' }, skeleton(2));
    this.tiles = h('div', { class: 'stack-v engines-by-host' }, skeleton(2));
    this.append(pageHead({ page: 'dashboard', title: t('nav.dashboard') }), this.summary, this.alerts,
      h('h2', { class: 'section' }, withTip(h('span', t('fleet.machines')), { kind: 'concept', key: 'page.hosts' }, t('fleet.machines'))), this.grid,
      h('h2', { class: 'section' }, withTip(h('span', t('dash.instances')), { kind: 'concept', key: 'page.instances' }, t('dash.instances'))), this.tiles);
    this.loadHosts(); this.tick();
    this.every(3000, () => this.tick());
    this.every(5000, () => { if (!this.streamLive) this.loadHosts(); });
    this.every(1000, () => this.tickCountdowns());
    this.stream('/fleet/stream', {
      onOpen: () => { this.streamLive = true; },
      onError: () => { this.streamLive = false; },
      onClose: () => { this.streamLive = false; },
      onEvent: (ev) => { if (ev.event === 'host' && ev.data?.id) this.setHost(ev.data); },
    });
    this.stream('/metrics/stream', { onEvent: (ev) => {
      const d = ev.data;
      if (d && d.instance_id) { this.live.set(d.instance_id, d.values || d.metrics || {}); this.paintTilesSoon(); }
    } });
    this.own(() => clearTimeout(this._tt));
  }
  /** Metrics arrive for every engine every couple of seconds: coalesce them into one repaint. */
  paintTilesSoon() { if (this._tt) return; this._tt = setTimeout(() => { this._tt = 0; if (this._alive) this.paintTiles(); }, 800); }

  // ---------- data ----------
  async loadHosts() {
    try {
      const list = items(await this.api.hosts()).map(normHost);
      if (!this._alive) return;
      this.hostsErr = null;
      this.order = list.map((x) => x.id);
      for (const x of list) { this.hosts.set(x.id, x); this.rx.set(x.id, Date.now()); }
    } catch (e) { this.hostsErr = e; }
    this.paintAll();
    this.loadDetails();
  }
  setHost(raw) {
    const x = normHost(raw);
    this.hosts.set(x.id, x); this.rx.set(x.id, Date.now());
    if (!this.order.includes(x.id)) this.order.push(x.id);
    if (!this._paintQueued) { this._paintQueued = true; requestAnimationFrame(() => { this._paintQueued = false; if (this._alive) this.paintAll(); }); }
    this.loadDetails();
  }
  /** Drives and roots come from HostDetail; fetched per machine with data, then refreshed every 30 s. */
  loadDetails() {
    for (const [id, x] of this.hosts) {
      if (!['up', 'stale'].includes(x.link.state)) continue;
      if (Date.now() - (this.detailAt.get(id) || 0) < 30000) continue;
      this.detailAt.set(id, Date.now());
      this.api.host(id).then((d) => { if (!this._alive) return; this.details.set(id, normHost(d)); this.sigs.delete(id); this.paintHosts(); }, () => { this.detailAt.set(id, Date.now() - 20000); });
    }
  }
  async tick() {
    try {
      const [list, scripts] = await Promise.all([this.api.instances().then((r) => items(r).map(normInstance)), loadScripts()]);
      if (!this._alive) return;
      this.instErr = null; this.scripts = scripts; this.labels = scriptLabelMap(scripts);
      const sig = listSignature(list) + scripts.map((x) => x.host_id + x.name + x.label).join();
      if (sig !== this._instSig) { this._instSig = sig; this.inst = list; this.paintAll(); } else this.inst = list;
    } catch (e) { this.instErr = e; if (this._alive) this.paintTiles(); }
  }
  tickCountdowns() {
    for (const el of this.grid.querySelectorAll('.retry-count')) {
      const host = this.hosts.get(el.dataset.host);
      if (host) el.textContent = t('fleet.retry_in', { count: retryCountdown(host.link, this.rx.get(host.id)) ?? 0 });
    }
  }

  // ---------- painting ----------
  paintAll() { this.paintSummary(); this.paintAlerts(); this.paintHosts(); this.paintTiles(); }
  hostList() { return this.order.map((id) => this.hosts.get(id)).filter(Boolean); }
  paintSummary() {
    const s = fleetSummary(this.hostList());
    clear(this.summary).append(...(s.total ? [
      badge(t('fleet.sum_machines', { count: s.total }), 'muted'),
      s.up ? badge(t('fleet.sum_up', { count: s.up }), 'ok') : null,
      s.stale ? badge(t('fleet.sum_stale', { count: s.stale }), 'warn') : null,
      s.down ? badge(t('fleet.sum_down', { count: s.down }), 'bad') : null,
      s.other ? badge(t('fleet.sum_other', { count: s.other }), 'muted') : null,
      badge(t('fleet.sum_engines', { running: s.enginesRunning, total: s.enginesTotal }), 'info')] : []).filter(Boolean));
  }
  paintAlerts() {
    const out = [];
    for (const x of this.hostList()) {
      if (x.link.state !== 'up') continue;
      for (const g of x.gpus) {
        const used = g.used_gib, pct = used != null && g.total_gib ? (used / g.total_gib) * 100 : null;
        if (g.temp_c >= 85) out.push(h('div', { class: 'alert bad' }, t('fleet.alert_temp', { host: x.label, gpu: g.index, temp: Math.round(g.temp_c) })));
        if (pct > 97) out.push(h('div', { class: 'alert warn' }, t('fleet.alert_vram', { host: x.label, gpu: g.index, pct: Math.round(pct) })));
      }
    }
    for (const i of this.inst.filter((x) => x.state === 'failed' && x.management === 'managed')) out.push(h('div', { class: 'alert bad' }, t('fleet.alert_failed', { name: i.name || i.id, host: i.host_label || i.host_id }), ' ', h('a', { href: `#/instances/${enc(i.id)}` }, t('dash.view_logs'))));
    clear(this.alerts).append(...out);
  }
  paintHosts() {
    if (this.hostsErr && !this.hosts.size) { clear(this.grid).append(errorBox(this.hostsErr, () => this.loadHosts())); return; }
    if (!this.hosts.size) return;
    const hosts = this.hostList();
    if (this.grid.querySelector('.skeleton-group, .state')) clear(this.grid);
    let order = false;
    hosts.forEach((x, k) => {
      const mine = this.inst.filter((i) => i.host_id === x.id);
      const sig = hostSignature(x, `${listSignature(mine)}|${this.details.get(x.id)?.disks?.map((d) => Math.round((d.free_bytes || 0) / 2 ** 30)).join(',') || ''}`);
      const old = this.cards.get(x.id);
      if (old && this.sigs.get(x.id) === sig) return;
      if (old && busyWithin(old)) return; // retry on the next sample, never under the user's hands
      const card = hostCard(x, { instances: mine, detail: this.details.get(x.id), names: new Map(mine.map((i) => [i.id, instanceTitle(i, this.labels).title])), colors: engineColors(mine), receivedAt: this.rx.get(x.id), now: Date.now(), onReconnect: (id) => this.reconnect(id) });
      this.sigs.set(x.id, sig);
      if (old && old.isConnected) old.replaceWith(card); else order = true;
      this.cards.set(x.id, card);
    });
    for (const id of [...this.cards.keys()]) if (!this.hosts.has(id)) { this.cards.get(id).remove(); this.cards.delete(id); }
    if (order || this.grid.children.length !== hosts.length) this.grid.replaceChildren(...hosts.map((x) => this.cards.get(x.id)));
  }
  async reconnect(id) {
    try { const r = await this.api.reconnect(id); this.setHost(r); toast(t('fleet.reconnecting', { host: this.hosts.get(id)?.label || id }), { kind: 'ok', timeout: 2500 }); }
    catch (e) { toastError(e, t('fleet.reconnect')); }
  }

  vramLine(i) {
    if (!i.vram?.length) return null;
    return h('div', { class: 'vram-line small' }, withTip(h('span', { class: 'muted' }, t('dash.vram')), { kind: 'metric', key: 'vram_used_gib' }, t('dash.vram')), ' ',
      i.vram.map((v) => h('span', { class: 'chip-inline num' }, `GPU ${v.gpu_index} ${v.used_gib != null ? fmtGiB(v.used_gib) : t('fleet.shared_pool')}`)));
  }
  tile(i) {
    const m = this.live.get(i.id) || {}, ready = i.state === 'ready';
    const { title } = instanceTitle(i, this.labels);
    return h('article', { class: 'card tile' },
      h('header', { class: 'card-head' }, h('h3', { class: 'tile-title' }, titleBlock(i, this.labels)),
        h('div', { class: 'row gap tile-badges' }, managementBadge(i.management), stateBadge(i.state, { tip: true }))),
      h('div', { class: 'card-body' },
        instanceFacts(i, { title }), reasonNote(i), i.state === 'failed' && i.error ? h('p', { class: 'note bad small', role: 'status' }, i.error) : null, this.vramLine(i), hostRamRow(i),
        ready ? [
          h('div', { class: 'row between' }, withTip(h('span', t('dash.gen_tps')), { kind: 'metric', key: 'generation_tps' }, t('dash.gen_tps')), h('strong', { class: 'num big' }, fmtTps(m.generation_tps))),
          h('div', { class: 'row between small' }, statLabel(t('dash.kv_cache'), 'kv_cache_usage_pct'), h('span', { class: 'num' }, fmtPct(m.kv_cache_usage_pct))),
          progress(m.kv_cache_usage_pct ?? 0, { kind: (m.kv_cache_usage_pct ?? 0) > 90 ? 'warn' : '', label: t('dash.kv_cache') }),
          h('div', { class: 'row gap small muted wrap' },
            withTip(badge(`${t('dash.running')} ${m.requests_running ?? 0}`, 'muted'), { kind: 'metric', key: 'requests_running' }, t('dash.running')),
            withTip(badge(`${t('dash.waiting')} ${m.requests_waiting ?? 0}`, (m.requests_waiting ?? 0) > 0 ? 'warn' : 'muted'), { kind: 'metric', key: 'requests_waiting' }, t('dash.waiting')))] : null));
  }
  paintTiles() {
    if (this.instErr && !this.inst.length) { clear(this.tiles).append(errorBox(this.instErr, () => this.tick())); return; }
    if (!this.hosts.size && !this.inst.length) return;
    if (!this.inst.length) { clear(this.tiles).append(emptyBox(t('dash.no_instances'), t('dash.no_instances_hint'), btn(t('nav.launch'), { kind: 'primary', icon: 'launch', onClick: () => { location.hash = '#/launch'; } }))); return; }
    if (busyWithin(this.tiles)) return;
    const open = new Map([...this.tiles.querySelectorAll('details.available')].map((d) => [d.dataset.host, d.open]));
    clear(this.tiles).append(...groupByHost(this.inst, this.hostList()).map((g) => {
      const { active, available } = splitInstances(g.instances);
      const avail = availableSection(available, { scripts: this.scripts, labels: this.labels, hostLabel: g.label, open: open.get(g.id) ?? !active.length, onChange: () => this.tick(), onAction: (id, i) => doInstanceAction(i, id, { after: () => this.tick() }) });
      if (avail) avail.dataset.host = g.id;
      return h('section', { class: 'machine-group', 'aria-label': g.label },
        h('div', { class: 'row gap machine-head' }, h('h3', {}, h('a', { href: `#/hosts/${enc(g.id)}` }, g.label)), g.host ? linkBadge(g.host.link.state) : null, h('span', { class: 'muted small' }, t('fleet.engines_count', { running: g.instances.filter((i) => i.state === 'ready').length, total: g.instances.length }))),
        active.length ? h('div', { class: 'grid tiles' }, active.map((i) => this.tile(i))) : null, avail);
    }));
  }
}
customElements.define('ec-dashboard', EcDashboard);
