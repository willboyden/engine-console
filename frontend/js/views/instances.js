import { EcView } from '../components/base.js';
import { h, clear, copyText, download, icon } from '../dom.js';
import { t } from '../i18n.js';
import { items, fmtDuration, timeAgo } from '../format.js';
import { normInstance, uptimeOf, listSignature } from '../adapt.js';
import { btn, card, badge, stateBadge, progress, setProgress, emptyBox, errorBox, skeleton, kv, externalBadge, instanceFacts, reasonNote } from '../components/ui.js';
import { toast, toastError } from '../components/toast.js';
import { confirmDialog } from '../components/dialog.js';

const STEPS = ['starting', 'loading', 'ready'];
const MAX_LINES = 5000;

class EcInstances extends EcView {
  setup() { this.params?.id ? this.detail(this.params.id) : this.list(); }

  // ---------- list ----------
  list() {
    this.host = h('div', {}, skeleton(3));
    this.refreshBtn = btn(t('inst.refresh'), { icon: 'restart', onClick: () => this.discover() });
    this.append(h('div', { class: 'row between' }, h('h1', t('nav.instances')), h('div', { class: 'row gap' }, this.refreshBtn, btn(t('nav.launch'), { kind: 'primary', icon: 'launch', onClick: () => { location.hash = '#/launch'; } }))), this.host);
    const load = async (force) => {
      try {
        const list = items(await this.api.instances()).map(normInstance);
        if (!this._alive) return;
        // Poll quietly: repaint only when something visible changed, so the table never flickers.
        const sig = listSignature(list);
        if (sig === this.sig && !force) return;
        this.sig = sig;
        if (!list.length) { clear(this.host).append(emptyBox(t('inst.empty'), t('inst.empty_hint'))); return; }
        clear(this.host).append(h('div', { class: 'tablewrap' }, h('table',
          h('thead', h('tr', ['inst.name', 'inst.state', 'inst.model', 'inst.engine', 'inst.gpus', 'inst.port', 'inst.uptime'].map((k) => h('th', { scope: 'col' }, t(k))))),
          h('tbody', list.map((i) => h('tr', {},
            h('td', h('div', { class: 'row gap wrap' }, h('a', { class: 'name-link', href: `#/instances/${encodeURIComponent(i.id)}`, title: i.name || i.id }, i.name || i.id), i.external ? externalBadge() : null, i.pinned ? badge(t('inst.pinned'), 'info') : null), reasonNote(i)),
            h('td', stateBadge(i.state)), h('td', instanceFacts(i)), h('td', i.engine), h('td', (i.gpu_ids || []).join(', ')), h('td', { class: 'num' }, i.port ?? '–'),
            h('td', { class: 'num' }, i.state === 'ready' ? fmtDuration(uptimeOf(i)) : '–')))))));
      } catch (e) { if (this._alive) clear(this.host).append(errorBox(e, load)); }
    };
    this.load = load; load(true); this.every(3000, () => load());
  }
  async discover() {
    this.refreshBtn.disabled = true;
    try { await this.api.discover(); toast(t('inst.refreshed'), { kind: 'ok', timeout: 1800 }); await this.load(true); } catch (e) { toastError(e); }
    this.refreshBtn.disabled = false;
  }

  // ---------- detail ----------
  detail(id) {
    this.id = id; this.lines = []; this.follow = true; this.filter = '';
    this.head = h('div'); this.timeline = h('div'); this.info = h('div'); this.cmdHost = h('div', { class: 'stack-v' });
    this.logEl = h('pre', { class: 'logs', role: 'log', tabindex: '0', 'aria-label': t('inst.logs'), 'aria-live': 'off' });
    this.append(h('nav', { class: 'crumbs', 'aria-label': 'breadcrumb' }, h('a', { href: '#/instances' }, t('nav.instances')), ' / ', id),
      this.head, this.timeline, h('div', { class: 'grid two' }, card(t('inst.details'), this.info), card(t('inst.command'), this.cmdHost)),
      card(t('inst.logs'), this.logsHost = h('div', {}, this.logsUI())));
    this.refresh().then(() => this.afterFirst()); this.every(2000, () => this.refresh());
  }
  // Logs and launch commands exist only for console-managed engines.
  afterFirst() {
    if (!this._alive || this._inited || !this.inst) return;
    this._inited = true;
    if (this.inst.external) {
      clear(this.cmdHost).append(h('p', { class: 'hint' }, t('inst.no_ext_cmd')));
      clear(this.logsHost).append(h('p', { class: 'hint' }, t('inst.no_ext_logs')));
    } else { this.loadLogs(); this.cmd(); }
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
  async act(a) {
    try {
      if (a === 'delete') { if (!(await confirmDialog(t('inst.confirm_remove'), { danger: true }))) return; await this.api.deleteInstance(this.id); location.hash = '#/instances'; return; }
      await this.api.instanceAction(this.id, a); toast(t(`inst.did_${a}`), { kind: 'ok', timeout: 2200 }); this.refresh();
    } catch (e) { toastError(e); }
  }
  paintHead() {
    const i = this.inst, s = i.state, ext = i.external, tip = ext ? t('inst.not_managed_tip') : null;
    // External engines: control buttons stay visible but disabled with an explanation (never silently missing).
    const ctl = (label, o) => btn(label, { ...o, disabled: ext || o.disabled, title: tip || o.title, onClick: ext ? undefined : o.onClick });
    clear(this.head).append(h('div', { class: 'row between wrap' },
      h('div', { class: 'row gap wrap' }, h('h1', { class: 'wrap-anywhere', title: i.name || i.id }, i.name || i.id), stateBadge(s), ext ? externalBadge() : null, i.pinned ? badge(t('inst.pinned'), 'info') : null),
      h('div', { class: 'row gap wrap' },
        s === 'ready' ? btn(t('nav.chat'), { icon: 'chat', onClick: () => { location.hash = `#/chat?instance=${encodeURIComponent(i.id)}`; } }) : null,
        ['stopped', 'failed'].includes(s) ? ctl(t('inst.start'), { icon: 'play', kind: 'primary', onClick: () => this.act('start') }) : null,
        ['ready', 'loading', 'starting'].includes(s) || ext ? ctl(t('inst.stop'), { icon: 'stop', onClick: () => this.act('stop') }) : null,
        s !== 'stopped' || ext ? ctl(t('inst.restart'), { icon: 'restart', onClick: () => this.act('restart') }) : null,
        ext ? null : btn(t('inst.edit'), { onClick: () => { location.hash = `#/launch?from=${encodeURIComponent(i.id)}`; } }),
        ctl(i.pinned ? t('inst.unpin') : t('inst.pin'), { icon: 'pin', onClick: async () => { try { await this.api.patchInstance(this.id, { pinned: !i.pinned }); this.refresh(); } catch (e) { toastError(e); this.refresh(); } } }),
        ctl('', { icon: 'trash', title: t('inst.remove'), onClick: () => this.act('delete') }))));
  }
  paintTimeline() {
    const i = this.inst, s = i.state, idx = STEPS.indexOf(s);
    const pct = i.progress?.pct;
    if (i.external) { clear(this.timeline).append(...[reasonNote(i)].filter(Boolean)); return; }
    clear(this.timeline).append(h('section', { class: 'card' }, h('div', { class: 'card-body' },
      h('ol', { class: 'timeline', 'aria-label': t('inst.timeline') }, STEPS.map((st, k) => h('li', { class: k < idx || s === 'ready' ? 'done' : k === idx ? 'now' : '', 'aria-current': k === idx ? 'step' : null }, h('span', { class: 'dot' }, k < idx || s === 'ready' ? icon('check', 12) : k + 1), t(`state.${st}`)))),
      s === 'failed' ? h('div', { class: 'stack-v' }, h('p', { class: 'note bad', role: 'alert' }, i.error || t('inst.failed_generic')), i.last_logs ? h('pre', { class: 'cmd', tabindex: '0', 'aria-label': t('inst.last_logs') }, i.last_logs) : null) : null,
      ['starting', 'loading'].includes(s) ? h('div', {}, h('div', { class: 'row between small' }, h('span', i.progress?.phase ? t(`phase.${i.progress.phase}`) : t('inst.starting')), h('span', { class: 'num' }, pct != null ? `${Math.round(pct)}%` : '')), progress(pct ?? 0, { label: t('inst.startup') })) : null)));
  }
  paintInfo() {
    const i = this.inst;
    clear(this.info).append(h('dl', { class: 'facts' },
      kv(t('inst.model'), i.repo_id || '–'), kv(t('inst.engine'), i.engine), i.image ? kv(t('inst.image'), h('code', i.image)) : null,
      (i.served_models || []).length ? kv(t('inst.served'), h('span', { class: 'row gap wrap' }, i.served_models.map((m) => badge(m, 'muted')))) : null,
      i.endpoint ? kv(t('inst.endpoint'), h('code', i.endpoint)) : null, kv(t('inst.source'), i.external ? t('inst.source_external') : t('inst.source_console')),
      i.external ? null : kv(t('inst.gpus'), (i.gpu_ids || []).join(', ') || '–'),
      i.external || i.port == null ? null : kv(t('inst.port'), `127.0.0.1:${i.port}`), kv(t('inst.uptime'), i.state === 'ready' ? fmtDuration(uptimeOf(i)) : '–'),
      i.external ? null : kv(t('inst.ttl'), h('span', { class: 'row gap' }, i.ttl_s ? fmtDuration(i.ttl_s) : t('inst.no_ttl'),
        btn(t('inst.set_ttl'), { size: 'sm', onClick: () => this.setTtl() }))),
      kv(t('inst.created'), i.created_at ? timeAgo(i.created_at) : '–')),
      h('details', {}, h('summary', i.external ? t('inst.external_params') : t('inst.params')), Object.keys(i.params || {}).length ? h('dl', { class: 'facts' }, Object.entries(i.params).map(([k, v]) => kv(k, v === '[set]' ? badge(t('inst.secret_set'), 'ok', t('inst.secret_tip')) : typeof v === 'object' ? JSON.stringify(v) : String(v)))) : h('p', { class: 'hint' }, t('inst.no_params'))));
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
    this.stream(`/instances/${encodeURIComponent(this.id)}/logs/stream`, {
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
