// Downloads and copies: where each one is going (machine and folder), progress, and for a copy, the machine it comes from.
import { EcView } from '../components/base.js';
import { h, clear } from '../dom.js';
import { t } from '../i18n.js';
import { items, fmtBytes, fmtRate, fmtEta } from '../format.js';
import { normDownload, normHost } from '../adapt.js';
import { btn, badge, progress, stateBadge, emptyBox, errorBox, skeleton, field, pageHead, stackTable } from '../components/ui.js';
import { toast, toastError } from '../components/toast.js';
import { confirmDialog } from '../components/dialog.js';
import { openDownloadDialog } from '../components/download-dialog.js';
import { withTip } from '../components/info-tip.js';

class EcDownloads extends EcView {
  setup() {
    this.rows = new Map(); this.hosts = new Map(); this.copySrc = new Map();
    this.list = h('div', {}, skeleton(3));
    const repo = h('input', { type: 'text', placeholder: 'org/model-name', autocomplete: 'off', spellcheck: 'false', 'aria-label': t('dl.repo') });
    const form = h('form', { class: 'row gap bottom wrap', onSubmit: (e) => {
      e.preventDefault();
      const r = repo.value.trim();
      if (!r) return;
      openDownloadDialog({ api: this.api, repo: r, onQueued: () => { repo.value = ''; this.refresh(); } });
    } }, field(t('dl.repo'), repo, t('dl.repo_hint')), btn(t('dl.choose_machine'), { kind: 'primary', icon: 'download', type: 'submit' }));
    this.append(pageHead({ page: 'downloads', title: t('nav.downloads') }), h('section', { class: 'card' }, h('div', { class: 'card-body' }, form)), this.list);
    this.api.hosts().then((r) => { for (const x of items(r)) this.hosts.set(x.id, normHost(x)); this.paint(); }).catch(() => {});
    this.refresh();
    // Backend SSE: `snapshot` (array of every download) then `progress` (one download) events.
    this.stream('/downloads/stream', { onEvent: (ev) => {
      const d = ev.data;
      if (Array.isArray(d)) this.rows = new Map(d.map(normDownload).map((x) => [x.id, x]));
      else if (d && d.id) this.rows.set(d.id, { ...this.rows.get(d.id), ...normDownload(d) });
      else return;
      this.paint();
    } });
  }
  async refresh() {
    try { this.rows = new Map(items(await this.api.downloads()).map(normDownload).map((d) => [d.id, d])); this.err = null; } catch (e) { this.err = e; }
    this.paint();
    this.loadCopySources();
  }
  /** A copy row names its source machine: look the source copy up in the library once. */
  async loadCopySources() {
    if (![...this.rows.values()].some((d) => d.kind === 'copy' && d.source_copy_id && !this.copySrc.has(d.source_copy_id))) return;
    try { for (const m of await this.api.library()) for (const c of m.copies) this.copySrc.set(c.id, { host: c.host_label || c.host_id, name: m.name }); this.paint(); } catch { /* the row just omits the source */ }
  }
  async act(d, action) {
    try {
      if (action === 'delete') { if (!(await confirmDialog(t('dl.confirm_remove', { repo: d.repo_id }), { danger: true }))) return; await this.api.deleteDownload(d.id); this.rows.delete(d.id); this.paint(); }
      else { await this.api.downloadAction(d.id, action); this.refresh(); }
    } catch (e) { toastError(e); }
  }
  target(d) {
    const host = this.hosts.get(d.host_id);
    return h('div', { class: 'stack-v tight' },
      h('a', { href: `#/hosts/${encodeURIComponent(d.host_id)}` }, host?.label || d.host_id),
      h('span', { class: 'muted small' }, d.root_id ? t('dl.in_root', { root: d.root_id }) : t('dl.default_root')));
  }
  paint() {
    if (this.err) { clear(this.list).append(errorBox(this.err, () => this.refresh())); return; }
    const rows = [...this.rows.values()];
    if (!rows.length) { clear(this.list).append(emptyBox(t('dl.empty'), t('dl.empty_hint'))); return; }
    const tbody = h('tbody', rows.map((d) => {
      const pct = d.bytes_total ? (d.bytes_done / d.bytes_total) * 100 : 0;
      const active = d.status === 'running' || d.status === 'queued';
      const src = d.kind === 'copy' ? this.copySrc.get(d.source_copy_id) : null;
      return h('tr', {},
        h('td', h('strong', d.repo_id), d.kind === 'copy' ? h('div', { class: 'row gap wrap small' }, badge(t('dl.kind_copy'), 'info'), src ? h('span', { class: 'muted' }, t('dl.copy_from', { host: src.host })) : null) : null,
          d.allow_patterns?.length ? h('div', { class: 'muted small ellipsis', title: d.allow_patterns.join(', ') }, d.allow_patterns.join(', ')) : null,
          d.error ? h('div', { class: 'bad-text small' }, d.error) : null),
        h('td', this.target(d)),
        h('td', stateBadge(d.status)),
        h('td', { class: 'wide' }, progress(pct, { label: d.repo_id, kind: d.status === 'failed' ? 'bad' : d.status === 'completed' ? 'ok' : '' }),
          h('div', { class: 'small muted num' }, `${fmtBytes(d.bytes_done)} / ${fmtBytes(d.bytes_total)}${d.files_total ? ` · ${d.files_done ?? 0}/${d.files_total} ${t('dl.files')}` : ''}`)),
        h('td', { class: 'num nowrap' }, d.status === 'running' ? fmtRate(d.speed_bps) : '–'),
        h('td', { class: 'num nowrap' }, d.status === 'running' ? fmtEta(d.eta_s) : '–'),
        h('td', { class: 'row gap end' },
          d.status === 'running' ? btn('', { icon: 'pause', size: 'sm', title: t('dl.pause'), onClick: () => this.act(d, 'pause') }) : null,
          d.status === 'paused' || d.status === 'failed' ? btn('', { icon: 'play', size: 'sm', title: t('dl.resume'), onClick: () => this.act(d, 'resume') }) : null,
          active || d.status === 'paused' ? btn('', { icon: 'stop', size: 'sm', title: t('dl.cancel'), onClick: () => this.act(d, 'cancel') }) : null,
          !active ? btn('', { icon: 'trash', size: 'sm', title: t('dl.remove'), onClick: () => this.act(d, 'delete') }) : null));
    }));
    clear(this.list).append(stackTable(h('div', { class: 'tablewrap' }, h('table', h('thead', h('tr', ['dl.model', 'dl.target', 'dl.status', 'dl.progress', 'dl.speed', 'dl.eta', ''].map((k) => h('th', { scope: 'col' }, k === 'dl.target' ? withTip(t(k), { kind: 'concept', key: 'field.root' }, t(k)) : k ? t(k) : '')))), tbody)), { key: 'downloads' }));
  }
}
customElements.define('ec-downloads', EcDownloads);
