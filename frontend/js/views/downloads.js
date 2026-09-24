import { EcView } from '../components/base.js';
import { h, clear } from '../dom.js';
import { t } from '../i18n.js';
import { items, fmtBytes, fmtRate, fmtEta } from '../format.js';
import { normDownload } from '../adapt.js';
import { btn, progress, stateBadge, emptyBox, errorBox, skeleton, field } from '../components/ui.js';
import { toast, toastError } from '../components/toast.js';
import { confirmDialog } from '../components/dialog.js';

class EcDownloads extends EcView {
  setup() {
    this.rows = new Map();
    this.list = h('div', {}, skeleton(3));
    const repo = h('input', { type: 'text', placeholder: 'org/model-name', autocomplete: 'off', spellcheck: 'false' });
    const form = h('form', { class: 'row gap bottom wrap', onSubmit: async (e) => {
      e.preventDefault();
      if (!repo.value.trim()) return;
      try { await this.api.startDownload({ repo_id: repo.value.trim() }); repo.value = ''; toast(t('dl.queued'), { kind: 'ok' }); this.refresh(); } catch (err) { toastError(err); }
    } }, field(t('dl.repo'), repo), btn(t('dl.start'), { kind: 'primary', icon: 'download', type: 'submit' }));
    this.append(h('h1', t('nav.downloads')), h('section', { class: 'card' }, h('div', { class: 'card-body' }, form)), this.list);
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
  }
  async act(d, action) {
    try {
      if (action === 'delete') { if (!(await confirmDialog(t('dl.confirm_remove', { repo: d.repo_id }), { danger: true }))) return; await this.api.deleteDownload(d.id); this.rows.delete(d.id); this.paint(); }
      else { await this.api.downloadAction(d.id, action); this.refresh(); }
    } catch (e) { toastError(e); }
  }
  paint() {
    if (this.err) { clear(this.list).append(errorBox(this.err, () => this.refresh())); return; }
    const rows = [...this.rows.values()];
    if (!rows.length) { clear(this.list).append(emptyBox(t('dl.empty'), t('dl.empty_hint'))); return; }
    const tbody = h('tbody', rows.map((d) => {
      const pct = d.bytes_total ? (d.bytes_done / d.bytes_total) * 100 : 0;
      const active = d.status === 'running' || d.status === 'queued';
      return h('tr', {},
        h('td', h('strong', d.repo_id), d.error ? h('div', { class: 'bad-text small' }, d.error) : null),
        h('td', stateBadge(d.status)),
        h('td', { class: 'wide' }, progress(pct, { label: d.repo_id, kind: d.status === 'failed' ? 'bad' : d.status === 'completed' ? 'ok' : '' }),
          h('div', { class: 'small muted num' }, `${fmtBytes(d.bytes_done)} / ${fmtBytes(d.bytes_total)}${d.files_total ? ` · ${d.files_done ?? 0}/${d.files_total} ${t('dl.files')}` : ''}`)),
        h('td', { class: 'num' }, d.status === 'running' ? fmtRate(d.speed_bps) : '–'),
        h('td', { class: 'num' }, d.status === 'running' ? fmtEta(d.eta_s) : '–'),
        h('td', { class: 'row gap end' },
          d.status === 'running' ? btn('', { icon: 'pause', size: 'sm', title: t('dl.pause'), onClick: () => this.act(d, 'pause') }) : null,
          d.status === 'paused' || d.status === 'failed' ? btn('', { icon: 'play', size: 'sm', title: t('dl.resume'), onClick: () => this.act(d, 'resume') }) : null,
          active || d.status === 'paused' ? btn('', { icon: 'stop', size: 'sm', title: t('dl.cancel'), onClick: () => this.act(d, 'cancel') }) : null,
          !active ? btn('', { icon: 'trash', size: 'sm', title: t('dl.remove'), onClick: () => this.act(d, 'delete') }) : null));
    }));
    clear(this.list).append(h('div', { class: 'tablewrap' }, h('table', h('thead', h('tr', ['dl.model', 'dl.status', 'dl.progress', 'dl.speed', 'dl.eta', ''].map((k) => h('th', { scope: 'col' }, k ? t(k) : '')))), tbody)));
  }
}
customElements.define('ec-downloads', EcDownloads);
