// Library: every model stored on any machine, grouped under one entry per model with one row per copy. From a copy you can
// launch the model on a machine that can read it, copy it to another machine, or delete it. State of each action (and the
// reason when it is off) is decided in library.js; a machine that is down keeps its last-known copies, shown greyed out.
import { EcView } from '../components/base.js';
import { h, clear } from '../dom.js';
import { t } from '../i18n.js';
import { items, fmtBytes, fmtInt, timeAgo, debounce } from '../format.js';
import { normHost } from '../adapt.js';
import { buildHash } from '../router.js';
import { isBeginner } from '../beginner.js';
import { filterLibrary, sortLibrary, copyChip, copyActions, launchHosts, resolveOnHost, paramsLabel, inUse, totalCopies, hostsThatSee, ROOT_KINDS_BY_FORMAT, FORMAT_ORDER } from '../library.js';
import { refToQuery } from '../model-ref.js';
import { card, badge, btn, field, select, formatBadge, emptyBox, errorBox, skeleton, pageHead, reasonText } from '../components/ui.js';
import { openModal, confirmDialog } from '../components/dialog.js';
import { hostRootPicker } from '../components/host-root-picker.js';
import { toast, toastError } from '../components/toast.js';
import { busyWithin } from '../components/info-tip.js';
import { instanceTitle, scriptLabelMap } from '../controls.js';
import { loadScripts } from '../components/script-actions.js';

const enc = encodeURIComponent;

class EcLibrary extends EcView {
  setup() {
    const q = this.query || {};
    this.f = { host: q.host || '', format: q.format || '', q: q.q || '', sort: 'name' };
    this.models = []; this.hosts = []; this.engines = new Map(); this.details = new Map(); this.names = new Map();
    this.body = h('div', { class: 'stack-v' }, skeleton(4));
    this.summary = h('p', { class: 'muted small', role: 'status' });
    this.rescanBtn = btn(t('library.rescan'), { icon: 'restart', onClick: () => this.rescan() });
    this.toolbar = h('div', { class: 'filters lib-filters' });
    this.append(pageHead({ page: 'library', title: t('nav.library'), actions: [this.rescanBtn, btn(t('library.find_models'), { icon: 'discover', onClick: () => { location.hash = '#/discover'; } })] }), this.toolbar, this.summary, this.body);
    this.load();
    this.every(6000, () => this.load(true));
  }
  get hostsById() { return new Map(this.hosts.map((x) => [x.id, x])); }
  async load(quiet) {
    try {
      const [models, hosts, engines, insts, scripts] = await Promise.all([this.api.library(), this.api.hosts(), this.engines.size ? null : this.api.engines().catch(() => []), this.api.instances().catch(() => []), loadScripts().catch(() => [])]);
      if (!this._alive) return;
      if (engines) for (const e of items(engines)) this.engines.set(e.id, e);
      // The engines that use a copy are named by what they serve, not by their container.
      const labels = scriptLabelMap(scripts);
      this.names = new Map(items(insts).map((i) => [i.id, instanceTitle(i, labels).title]));
      const hostList = items(hosts).map(normHost);
      const sig = JSON.stringify([models.map((m) => [m.key, m.copies.map((c) => [c.id, c.in_use_by, c.deletable])]), hostList.map((x) => [x.id, x.link.state, x.policy?.mode])]);
      this.models = models; this.hosts = hostList; this.err = null;
      if (!this.toolbar.childElementCount) this.paintToolbar();
      if (sig === this._sig && quiet) return;
      this._sig = sig;
    } catch (e) { this.err = e; if (quiet && this.models.length) return; }
    this.paint();
  }
  paintToolbar() {
    const input = h('input', { type: 'search', value: this.f.q, placeholder: t('library.search_ph'), 'aria-label': t('library.search'), autocomplete: 'off', onInput: debounce((e) => { this.f.q = e.target.value; this.paint(); }, 200) });
    clear(this.toolbar).append(
      h('div', { class: 'field grow' }, h('label', { for: 'lib-q' }, t('library.search')), Object.assign(input, { id: 'lib-q' })),
      field(t('library.machine'), select([{ value: '', label: t('library.all_machines') }, ...this.hosts.map((x) => ({ value: x.id, label: x.label }))], this.f.host, (v) => { this.f.host = v; this.paint(); }), null, { kind: 'concept', key: 'field.machine_filter', inline: false }),
      field(t('library.format'), select([{ value: '', label: t('common.any') }, ...FORMAT_ORDER.map((f) => ({ value: f, label: t(`format.${f}`) }))], this.f.format, (v) => { this.f.format = v; this.paint(); }), null, { kind: 'concept', key: 'field.format_filter', inline: false }),
      field(t('library.sort'), select([{ value: 'name', label: t('library.sort_name') }, { value: 'size', label: t('library.sort_size') }, { value: 'format', label: t('library.sort_format') }], this.f.sort, (v) => { this.f.sort = v; this.paint(); })));
  }

  paint() {
    if (this.err && !this.models.length) { clear(this.body).append(errorBox(this.err, () => this.load())); return; }
    if (!this.models.length) { clear(this.body).append(emptyBox(t('library.empty'), t('library.empty_hint'), btn(t('library.find_models'), { kind: 'primary', icon: 'discover', onClick: () => { location.hash = '#/discover'; } }))); clear(this.summary); return; }
    const shown = sortLibrary(filterLibrary(this.models, this.f), this.f.sort);
    this.summary.textContent = t('library.summary', { models: shown.length, copies: totalCopies(shown), size: fmtBytes(shown.reduce((a, m) => a + m.size_bytes * m.copies.length, 0), 0) });
    if (busyWithin(this.body)) return;
    clear(this.body).append(...(shown.length ? shown.map((m) => this.modelCard(m)) : [emptyBox(t('library.no_match'), t('library.no_match_hint'))]));
  }

  modelCard(m) {
    const eng = (id) => this.engines.get(id)?.display_name || id;
    const facts = [
      m.quant ? badge(m.quant, 'muted', t('library.quant_tip')) : null,
      paramsLabel(m.params) ? badge(t('library.params', { n: paramsLabel(m.params) }), 'muted') : null,
      m.context ? badge(t('library.context', { n: fmtInt(m.context) }), 'muted') : null,
      m.arch ? badge(m.arch, 'muted') : null,
    ].filter(Boolean);
    return h('article', { class: 'card lib-card' },
      h('header', { class: 'lib-head' },
        h('div', { class: 'stack-v tight lib-title' }, h('h3', { class: 'wrap-anywhere' }, m.name), h('div', { class: 'row gap wrap small' }, formatBadge(m.format), ...facts, h('span', { class: 'num strong' }, fmtBytes(m.size_bytes, 1)))),
        h('div', { class: 'row gap wrap lib-badges' }, m.engines.length ? m.engines.map((id) => badge(eng(id), 'info', t('library.engine_tip', { engine: eng(id) }))) : badge(t('library.no_engine'), 'muted', t('library.no_engine_tip')), inUse(m) ? badge(t('library.in_use'), 'warn') : null)),
      (m.warnings || []).length ? h('div', { class: 'card-body lib-warn' }, m.warnings.map((w) => h('p', { class: 'note warn small' }, w))) : null,
      h('ul', { class: 'copy-list' }, m.copies.map((c) => this.copyRow(m, c))));
  }

  copyRow(m, c) {
    const nameOf = (id) => this.names.get(id) || id;
    const chip = copyChip(c), act = copyActions(c, this.hostsById, m, nameOf), host = this.hostsById.get(c.host_id);
    const offline = host && host.link.state !== 'up';
    const label = (id) => this.hostsById.get(id)?.label || id;
    return h('li', { class: `copy-row${offline ? ' is-offline' : ''}` },
      h('div', { class: 'copy-main' },
        h('div', { class: 'row gap wrap' },
          h('a', { class: 'host-chip', href: `#/hosts/${enc(c.host_id)}`, title: c.path }, h('strong', chip.host), h('span', { class: 'muted', 'aria-hidden': 'true' }, '·'), h('span', { class: 'muted' }, chip.drive)),
          chip.shared ? badge(t('library.shared'), 'info', t('library.shared_tip', { hosts: chip.alsoOn.map(label).join(', ') })) : null,
          offline ? badge(t('library.offline', { state: t(`link.${host.link.state}`) }), 'warn', t('library.offline_tip')) : null,
          (c.in_use_by || []).length ? badge(t('library.in_use_by', { names: c.in_use_by.map(nameOf).join(', ') }), 'warn') : null),
        h('div', { class: 'muted small copy-meta' }, [fmtBytes(c.size_bytes, 1), c.files != null ? t('library.files', { count: c.files }) : null, chip.alsoOn.length ? t('library.visible_on', { hosts: chip.alsoOn.map(label).join(', ') }) : null, c.atime ? t('library.last_used', { when: timeAgo(c.atime) }) : null].filter(Boolean).join(' · '))),
      h('div', { class: 'row gap wrap copy-actions' },
        btn(t('library.launch_on'), { size: 'sm', kind: 'primary', icon: 'launch', disabled: !act.launch.enabled, reason: act.launch.reason, onClick: () => this.launchOn(m, c) }),
        btn(t('library.copy_to'), { size: 'sm', icon: 'copy', disabled: !act.copy.enabled, reason: act.copy.reason, onClick: () => this.copyDialog(m, c) }),
        btn(t('common.delete'), { size: 'sm', icon: 'trash', kind: act.delete.enabled ? '' : 'ghost', disabled: !act.delete.enabled, reason: act.delete.reason, onClick: () => this.deleteCopy(m, c) })));
  }

  // ---- launch ----
  async launchOn(m, c) {
    const choices = launchHosts(c, this.hostsById);
    const ok = choices.filter((x) => x.enabled);
    if (ok.length === 1 && choices.length === 1) return this.goLaunch(m, ok[0].id);
    const pick = { id: (ok.find((x) => x.id === c.host_id) || ok[0])?.id };
    const body = h('form', { class: 'stack-v', onSubmit: (e) => { e.preventDefault(); dlg.close(); this.goLaunch(m, pick.id); } },
      h('p', {}, t('library.launch_pick_intro', { name: m.name })),
      h('div', { class: 'stack-v tight', role: 'radiogroup', 'aria-label': t('library.launch_pick') }, choices.map((x) => h('label', { class: `pick-row${x.enabled ? '' : ' is-disabled'}` },
        h('input', { type: 'radio', name: 'lhost', value: x.id, checked: x.id === pick.id, disabled: !x.enabled, onChange: () => { pick.id = x.id; } }),
        h('span', {}, h('strong', x.label), x.id === c.host_id ? h('span', { class: 'muted' }, ` ${t('library.holds_copy')}`) : h('span', { class: 'muted' }, ` ${t('library.reads_shared')}`), !x.enabled ? h('span', { class: 'muted small block' }, reasonText(x.reason)) : null)))),
      h('div', { class: 'row gap end' }, btn(t('common.cancel'), { onClick: () => dlg.close() }), btn(t('library.launch'), { kind: 'primary', type: 'submit', icon: 'launch' })));
    const dlg = openModal(t('library.launch_on'), body);
  }
  async goLaunch(m, hostId) {
    try {
      if (!this.details.has(hostId)) this.details.set(hostId, normHost(await this.api.host(hostId)));
      const r = resolveOnHost(m, hostId, this.details);
      if (!r?.ref) { toast(t('library.cannot_launch'), { kind: 'bad' }); return; }
      const q = { host: hostId, mkey: m.key, ...refToQuery(r.ref) };
      location.hash = isBeginner() ? buildHash('/launch/wizard', q) : buildHash('/launch', { ...q, advanced: 1 });
    } catch (e) { toastError(e, t('library.launch_on')); }
  }

  // ---- copy to another machine ----
  copyDialog(m, c) {
    const exclude = new Set([c.host_id, ...(c.visible_on || []), ...hostsThatSee(m)]);
    let picked = null;
    const need = h('p', { class: 'small', role: 'status' });
    const submit = btn(t('library.copy_start'), { kind: 'primary', icon: 'copy', type: 'submit', disabled: true });
    const paint = () => {
      const free = picked?.free;
      clear(need).append(`${t('library.copy_needs', { size: fmtBytes(c.size_bytes, 1) })}${free != null ? ` · ${t('dl.dialog_free', { free: fmtBytes(free, 0) })}` : ''}`);
      if (free != null && c.size_bytes > free) need.append(' ', h('strong', { class: 'bad-text' }, t('dl.dialog_no_space')));
      submit.disabled = !picked?.ok;
    };
    const picker = hostRootPicker({ api: this.api, hosts: this.hosts, kinds: ROOT_KINDS_BY_FORMAT[m.format], exclude, onChange: (s) => { picked = s; paint(); } });
    const form = h('form', { class: 'stack-v', onSubmit: async (e) => {
      e.preventDefault();
      if (!picked?.ok) return;
      submit.disabled = true;
      try { await this.api.copyCopy(c.id, { host_id: picked.hostId, root_id: picked.rootId }); toast(t('library.copy_queued', { name: m.name, host: picked.detail?.label || picked.hostId }), { kind: 'ok' }); dlg.close(); }
      catch (err) { toastError(err, t('library.copy_to')); submit.disabled = false; }
    } }, h('p', {}, t('library.copy_intro', { name: m.name, host: chipHost(c) })), picker.el, need,
    h('div', { class: 'row gap end' }, btn(t('common.cancel'), { onClick: () => dlg.close() }), submit));
    const dlg = openModal(t('library.copy_to'), form);
    picker.whenReady.then(paint);
  }

  async deleteCopy(m, c) {
    if (!(await confirmDialog(t('library.confirm_delete', { name: m.name, host: chipHost(c), path: c.path }), { danger: true, confirmLabel: t('common.delete') }))) return;
    try { await this.api.deleteCopy(c.id); toast(t('library.deleted', { name: m.name }), { kind: 'ok' }); this.load(); } catch (e) { toastError(e, t('common.delete')); }
  }
  async rescan() {
    this.rescanBtn.disabled = true;
    try { const r = await this.api.rescanLibrary(); toast(t('library.rescanning', { count: (r.hosts || []).length }), { kind: 'ok', timeout: 2500 }); setTimeout(() => { if (this._alive) this.load(); }, 1200); }
    catch (e) { toastError(e, t('library.rescan')); }
    this.rescanBtn.disabled = false;
  }
}
const chipHost = (c) => c.host_label || c.host_id;
customElements.define('ec-library', EcLibrary);
