import { EcView } from '../components/base.js';
import { h, clear, append, copyText, download, icon } from '../dom.js';
import { t } from '../i18n.js';
import { items, fmtDuration, fmtGiB, timeAgo } from '../format.js';
import { normInstance, normHost, uptimeOf, listSignature } from '../adapt.js';
import { controlButtons, scriptFor, scriptLabelMap, instanceTitle, splitInstances } from '../controls.js';
import { hostRamRow } from '../components/mem-bar.js';
import { btn, card, badge, stateBadge, managementBadge, linkBadge, progress, emptyBox, errorBox, skeleton, kv, kvTip, instanceFacts, reasonNote, field, select, pageHead, reasonText, tr, stackTable } from '../components/ui.js';
import { withTip, busyWithin, infoTip } from '../components/info-tip.js';
import { controlBar } from '../components/instance-controls.js';
import { doInstanceAction } from '../components/instance-actions.js';
import { scriptPanel, loadScripts } from '../components/script-actions.js';
import { titleBlock, scriptExtras, availableSection } from '../components/instance-rows.js';
import { toast, toastError } from '../components/toast.js';
import { statLabel } from '../components/fleet-bars.js';

const STEPS = ['starting', 'loading', 'ready'];
const MAX_LINES = 5000;
const enc = encodeURIComponent;

/** "GPU 0 38.2 GiB" per GPU the engine holds memory on (the driver cannot attribute it on unified-memory machines). */
const vramText = (i) => (i.vram || []).map((v) => `GPU ${v.gpu_index} ${v.used_gib != null ? fmtGiB(v.used_gib) : t('fleet.shared_pool')}`);

class EcInstances extends EcView {
  setup() { this.params?.id ? this.detail(this.params.id) : this.list(); }

  // ---------- list ----------
  list() {
    this.hostFilter = this.query?.host || '';
    this.hosts = new Map(); this.scripts = []; this.labels = new Map();
    this.host = h('div', {}, skeleton(3)); this.avail = h('div');
    this.filterHost = h('div');
    this.refreshBtn = btn(t('inst.refresh'), { icon: 'restart', onClick: () => this.discover() });
    this.append(pageHead({ page: 'instances', title: t('nav.instances'), actions: [this.refreshBtn, btn(t('nav.launch'), { kind: 'primary', icon: 'launch', onClick: () => { location.hash = '#/launch'; } })] }), this.filterHost, this.host, this.avail);
    this.api.hosts().then((r) => { for (const x of items(r)) this.hosts.set(x.id, normHost(x)); this.paintFilter(); this.load?.(true); }).catch(() => {});
    const load = async (force) => {
      try {
        const [list, scripts] = await Promise.all([this.api.instances(this.hostFilter || undefined).then((r) => items(r).map(normInstance)), loadScripts()]);
        if (!this._alive) return;
        this.scripts = scripts; this.labels = scriptLabelMap(scripts);
        // Poll quietly: repaint only when something visible changed, so the table never flickers.
        const sig = listSignature(list) + this.hostFilter + scripts.map((x) => `${x.host_id}${x.name}${x.status}`).join();
        if (sig === this.sig && !force) return;
        if ((busyWithin(this.host) || busyWithin(this.avail)) && this.sig && !force) return;
        this.sig = sig;
        if (!list.length) { clear(this.host).append(emptyBox(t('inst.empty'), t('inst.empty_hint'))); clear(this.avail); return; }
        const { active, available } = splitInstances(list);
        clear(this.host).append(active.length ? stackTable(h('div', { class: 'tablewrap' }, h('table', { class: 'inst-table' },
          h('thead', h('tr', ['inst.name', 'inst.state', 'inst.model', 'inst.vram', 'inst.actions'].map((k) => h('th', { scope: 'col', class: k === 'inst.actions' ? 'sr-th' : null }, h('span', { class: k === 'inst.actions' ? 'sr-only' : null }, t(k)))))),
          h('tbody', active.map((i) => this.row(i))))), { key: 'instances' }) : emptyBox(t('inst.none_running'), t('inst.none_running_hint')));
        const wasOpen = this.avail.querySelector('details')?.open;
        clear(this.avail).append(...[availableSection(available, { scripts, labels: this.labels, showHost: true, title: t('inst.available_title_all'), open: wasOpen ?? !active.length, onChange: () => load(true),
          onAction: (id, i) => doInstanceAction(i, id, { after: () => load(true) }) })].filter(Boolean));
      } catch (e) { if (this._alive) clear(this.host).append(errorBox(e, load)); }
    };
    this.load = load; load(true); this.every(3000, () => load());
  }
  paintFilter() {
    clear(this.filterHost).append(h('div', { class: 'row gap wrap bottom inst-filter' }, field(t('library.machine'), select([{ value: '', label: t('library.all_machines') }, ...[...this.hosts.values()].map((x) => ({ value: x.id, label: x.label }))], this.hostFilter, (v) => { this.hostFilter = v; this.load(true); }), null, { kind: 'concept', key: 'field.machine_filter', inline: false })));
  }
  row(i) {
    const hostLabel = i.host_label || this.hosts.get(i.host_id)?.label || i.host_id;
    const bar = controlBar(i, { size: 'sm', iconOnly: true, only: ['start', 'stop', 'restart'], onAction: (id) => doInstanceAction(i, id, { after: () => this.load(true) }) });
    const extras = scriptExtras(i, this.scripts, { hostLabel, onChange: () => this.load(true) });
    return h('tr', {},
      h('td', titleBlock(i, this.labels), h('div', { class: 'row gap wrap small' }, managementBadge(i.management), h('a', { class: 'chip-link', href: `#/hosts/${enc(i.host_id)}`, title: t('inst.machine') }, hostLabel), i.pinned ? badge(t('inst.pinned'), 'info') : null), reasonNote(i)),
      h('td', h('div', { class: 'stack-v tight' }, stateBadge(i.state, { tip: true }), i.state === 'ready' ? h('span', { class: 'muted small num nowrap' }, t('inst.up_for', { time: fmtDuration(uptimeOf(i)) })) : null)), h('td', instanceFacts(i, { title: instanceTitle(i, this.labels).title }), i.port != null ? h('div', { class: 'muted small num' }, t('inst.port_line', { port: i.port })) : null, hostRamRow(i, { legend: false })),
      h('td', { class: 'num small nowrap' }, vramText(i).length ? h('div', { class: 'stack-v tight' }, vramText(i).map((x) => h('span', x))) : '–'),
      h('td', { class: 'inst-actions' }, h('div', { class: 'inst-actions-box' }, h('span', { class: 'inst-icons' }, ...bar.buttons), extras.length ? h('a', { class: 'btn sm', href: `#/instances/${enc(i.id)}` }, t('inst.script_actions_link')) : null)));
  }
  async discover() {
    this.refreshBtn.disabled = true;
    try { await this.api.discover(); toast(t('inst.refreshed'), { kind: 'ok', timeout: 1800 }); await this.load(true); } catch (e) { toastError(e); }
    this.refreshBtn.disabled = false;
  }

  // ---------- detail ----------
  detail(id) {
    this.id = id; this.lines = []; this.follow = true; this.filter = ''; this.scripts = []; this.labels = new Map();
    this.head = h('div'); this.timeline = h('div'); this.info = h('div'); this.cmdHost = h('div', { class: 'stack-v' }); this.scriptHost = h('div');
    this.logEl = h('pre', { class: 'logs', role: 'log', tabindex: '0', 'aria-label': t('inst.logs'), 'aria-live': 'off' });
    this.append(h('nav', { class: 'crumbs', 'aria-label': t('common.breadcrumb') }, h('a', { href: '#/instances' }, t('nav.instances')), ' / ', id),
      this.head, this.timeline, this.scriptHost, h('div', { class: 'grid two' }, card(t('inst.details'), this.info), card(t('inst.command'), this.cmdHost)),
      card(t('inst.logs'), this.logsHost = h('div', {}, this.logsUI())));
    this.refresh().then(() => this.afterFirst()); this.every(2000, () => this.refresh());
  }
  // Logs and launch commands exist only for console-managed engines (controls.logs); the others say why.
  afterFirst() {
    if (!this._alive || this._inited || !this.inst) return;
    this._inited = true;
    const lg = controlButtons(this.inst).logs;
    if (!lg.enabled) {
      clear(this.cmdHost).append(h('p', { class: 'hint' }, t('inst.no_ext_cmd')));
      clear(this.logsHost).append(h('p', { class: 'hint' }, this.inst.management === 'script' ? t('inst.no_script_logs') : t('inst.no_ext_logs')));
    } else { this.loadLogs(); this.cmd(); }
    if (this.inst.management === 'script') this.loadScript();
  }
  async loadScript() {
    this.scripts = await loadScripts({ force: true });
    if (!this._alive) return;
    this.labels = scriptLabelMap(this.scripts);
    this._sig = null; this.paintHead();
    const s = scriptFor(this.inst, this.scripts);
    clear(this.scriptHost).append(...(s ? [card(t('inst.script_actions', { script: s.label }), scriptPanel(s, { hostLabel: this.inst.host_label, onChange: () => { this.refresh(); this.loadScript(); } }))] : []));
  }
  async refresh() {
    try { this.inst = normInstance(await this.api.instance(this.id)); this.err = null; } catch (e) { this.err = e; }
    if (!this._alive) return;
    if (this.err && !this.inst) { clear(this.head).append(errorBox(this.err, () => this.refresh())); return; }
    const sig = listSignature([this.inst]) + Math.floor((uptimeOf(this.inst) || 0) / 30);
    if (sig === this._sig) return;
    this._sig = sig;
    this.paintHead(); this.paintTimeline(); this.paintInfo();
  }
  paintHead() {
    const i = this.inst, s = i.state, { title, secondary } = instanceTitle(i, this.labels);
    const bar = controlBar(i, { onAction: (id) => doInstanceAction(i, id, { after: (kind) => { if (kind === 'removed') location.hash = '#/instances'; else this.refresh(); } }) });
    append(clear(this.head), [
      h('div', { class: 'row between wrap' },
        h('div', { class: 'stack-v tight' }, h('div', { class: 'row gap wrap' }, h('h1', { class: 'wrap-anywhere', title }, title), infoTip({ kind: 'concept', key: 'page.instance', title }), stateBadge(s, { tip: true }), managementBadge(i.management), i.pinned ? badge(t('inst.pinned'), 'info') : null,
          h('a', { class: 'chip-link', href: `#/hosts/${enc(i.host_id)}` }, i.host_label || i.host_id)), secondary ? h('span', { class: 'muted small' }, secondary) : null),
        h('div', { class: 'row gap wrap' },
          s === 'ready' ? btn(t('nav.chat'), { icon: 'chat', onClick: () => { location.hash = `#/chat?instance=${enc(i.id)}`; } }) : null,
          ...bar.buttons)),
      bar.note]);
  }
  paintTimeline() {
    const i = this.inst, s = i.state, idx = STEPS.indexOf(s);
    const pct = i.progress?.pct;
    if (i.management !== 'managed') { clear(this.timeline).append(...[reasonNote(i)].filter(Boolean)); return; }
    clear(this.timeline).append(h('section', { class: 'card' }, h('div', { class: 'card-body' },
      h('ol', { class: 'timeline', 'aria-label': t('inst.timeline') }, STEPS.map((st, k) => h('li', { class: k < idx || s === 'ready' ? 'done' : k === idx ? 'now' : '', 'aria-current': k === idx ? 'step' : null }, h('span', { class: 'dot' }, k < idx || s === 'ready' ? icon('check', 12) : k + 1), t(`state.${st}`)))),
      s === 'failed' ? h('div', { class: 'stack-v' }, h('p', { class: 'note bad', role: 'alert' }, i.error || t('inst.failed_generic')), i.last_logs ? h('pre', { class: 'cmd', tabindex: '0', 'aria-label': t('inst.last_logs') }, i.last_logs) : null) : null,
      ['starting', 'loading'].includes(s) ? h('div', {}, h('div', { class: 'row between small' }, h('span', i.progress?.phase ? tr('phase', i.progress.phase) : t('inst.starting')), h('span', { class: 'num' }, pct != null ? `${Math.round(pct)}%` : '')), progress(pct ?? 0, { label: t('inst.startup') })) : null)));
  }
  paintInfo() {
    const i = this.inst, managed = i.management === 'managed', loc = i.model_location;
    clear(this.info).append(h('dl', { class: 'facts' },
      kv(t('inst.machine'), h('a', { href: `#/hosts/${enc(i.host_id)}` }, i.host_label || i.host_id)),
      kvTip(t('inst.management'), { kind: 'concept', key: `management.${i.management}` }, t(`management.${i.management}`)),
      i.script ? kv(t('inst.script'), h('code', i.script)) : null,
      kv(t('inst.model'), i.repo_id || '–'), kv(t('inst.engine'), i.engine), i.image ? kv(t('inst.image'), h('code', i.image)) : null,
      loc ? kv(t('inst.model_location'), h('div', { class: 'stack-v tight' }, h('code', { class: 'wrap-anywhere' }, loc.path), h('span', { class: 'muted small' }, [loc.drive, loc.remote ? t('inst.on_network') : null, loc.shared_from ? t('inst.shared_from', { host: loc.shared_from }) : null].filter(Boolean).join(' · ')))) : null,
      (i.served_models || []).length ? kv(t('inst.served'), h('span', { class: 'row gap wrap' }, i.served_models.map((m) => badge(m, 'muted')))) : null,
      i.endpoint ? kv(t('inst.endpoint'), h('code', i.endpoint)) : null, kv(t('inst.source'), managed ? t('inst.source_console') : t('inst.source_external')),
      i.gpu_ids?.length ? kv(t('inst.gpus'), i.gpu_ids.join(', ')) : null,
      (i.vram || []).length ? kv(t('inst.vram'), h('div', { class: 'stack-v tight' }, withTip(h('span', { class: 'muted small' }, t('dash.vram')), { kind: 'metric', key: 'vram_used_gib' }, t('dash.vram')), vramText(i).map((x) => h('span', { class: 'num' }, x)))) : null,
      kv(t('mem.host_ram'), hostRamRow(i)),
      managed && i.port != null ? kv(t('inst.port'), `127.0.0.1:${i.port}`) : null, kv(t('inst.uptime'), i.state === 'ready' ? fmtDuration(uptimeOf(i)) : '–'),
      managed ? kvTip(t('inst.ttl'), { kind: 'concept', key: 'field.ttl' }, h('span', { class: 'row gap' }, i.ttl_s ? fmtDuration(i.ttl_s) : t('inst.no_ttl'), btn(t('inst.set_ttl'), { size: 'sm', onClick: () => this.setTtl() }))) : null,
      kv(t('inst.created'), i.created_at ? timeAgo(i.created_at) : '–')),
      h('details', {}, h('summary', managed ? t('inst.params') : t('inst.external_params')), Object.keys(i.params || {}).length ? h('dl', { class: 'facts' }, Object.entries(i.params).map(([k, v]) => kv(k, v === '[set]' ? badge(t('inst.secret_set'), 'ok', t('inst.secret_tip')) : typeof v === 'object' ? JSON.stringify(v) : String(v)))) : h('p', { class: 'hint' }, t('inst.no_params'))));
  }
  async setTtl() {
    const v = prompt(t('inst.ttl_prompt'), String(Math.round((this.inst.ttl_s || 0) / 60)));
    if (v == null) return;
    const min = Number(v);
    if (!Number.isFinite(min) || min < 0) { toast(t('inst.ttl_invalid'), { kind: 'bad' }); return; }
    try { await this.api.patchInstance(this.id, { ttl_idle_s: Math.round(min * 60) }); this.refresh(); } catch (e) { toastError(e); }
  }
  async cmd() {
    this.cmdHost.append(skeleton(2));
    try {
      const c = await this.api.command(this.id);
      const tabs = [['docker_run', 'docker run'], ['compose_yaml', 'compose'], ['engine_cli', 'engine CLI']].filter(([k]) => c[k]);
      const pre = h('pre', { class: 'cmd', tabindex: '0' });
      const show = (k) => { pre.textContent = c[k]; this.cmdHost.querySelectorAll('.tab').forEach((b) => b.classList.toggle('on', b.dataset.k === k)); this.curCmd = c[k]; };
      clear(this.cmdHost).append(h('div', { class: 'tabs' }, tabs.map(([k, l]) => h('button', { type: 'button', class: 'tab', dataset: { k }, onClick: () => show(k) }, l))), pre,
        btn(t('launch.copy_cmd'), { icon: 'copy', size: 'sm', onClick: async () => { await copyText(this.curCmd); toast(t('common.copied'), { kind: 'ok', timeout: 1800 }); } }));
      if (tabs.length) show(tabs[0][0]);
    } catch (e) { clear(this.cmdHost).append(errorBox(e, () => this.cmd())); }
  }

  // ---------- logs ----------
  logsUI() {
    const filter = h('input', { type: 'search', placeholder: t('inst.filter'), 'aria-label': t('inst.filter'), onInput: (e) => { this.filter = e.target.value.toLowerCase(); this.paintLogs(true); } });
    const follow = h('label', { class: 'check' }, h('input', { type: 'checkbox', checked: true, onChange: (e) => { this.follow = e.target.checked; if (this.follow) this.paintLogs(true); } }), h('span', t('inst.follow')));
    this.logStatus = h('span', { class: 'muted small' });
    return h('div', { class: 'stack-v' }, h('div', { class: 'row gap wrap between' }, h('div', { class: 'row gap wrap' }, filter, follow, this.logStatus),
      btn(t('inst.download_logs'), { icon: 'download', size: 'sm', onClick: () => download(`${this.id}.log`, this.lines.join('\n')) })), this.logEl);
  }
  async loadLogs() {
    try {
      const r = await this.api.logs(this.id, 500); // text/plain from the backend; {lines} tolerated
      this.lines = (typeof r === 'string' ? r.split('\n') : r.lines || []).filter((l) => l !== '');
    } catch (e) { this.logStatus.textContent = e.message; }
    this.paintLogs(true);
    this.stream(`/instances/${enc(this.id)}/logs/stream`, {
      json: false,
      onOpen: () => { this.logStatus.textContent = t('inst.live'); },
      onError: () => { this.logStatus.textContent = t('inst.reconnecting'); },
      onEvent: (ev) => {
        let line = ev.data;
        try { const j = JSON.parse(line); if (j && typeof j.line === 'string') line = j.line; } catch { /* plain text line */ }
        this.lines.push(String(line));
        if (this.lines.length > MAX_LINES) this.lines.splice(0, this.lines.length - MAX_LINES);
        this.paintLogs();
      } });
  }
  paintLogs(force) {
    if (this._raf && !force) return;
    this._raf = requestAnimationFrame(() => {
      this._raf = 0;
      if (!this._alive) return;
      const shown = this.filter ? this.lines.filter((l) => l.toLowerCase().includes(this.filter)) : this.lines;
      this.logEl.textContent = shown.join('\n') || t('inst.no_logs');
      if (this.follow) this.logEl.scrollTop = this.logEl.scrollHeight;
    });
  }
}
customElements.define('ec-instances', EcInstances);
