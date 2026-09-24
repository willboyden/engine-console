import { EcView } from '../components/base.js';
import { h, clear, setTrustedHtml } from '../dom.js';
import { t } from '../i18n.js';
import { items, fmtBytes, fmtCompact, timeAgo, debounce } from '../format.js';
import { normHit, normLocalModel } from '../adapt.js';
import { renderMarkdown } from '../markdown.js';
import { quickFit, verdictMeta } from '../fit-format.js';
import { badge, btn, emptyBox, errorBox, skeleton, select, field, kv } from '../components/ui.js';
import { openDrawer, confirmDialog } from '../components/dialog.js';
import { toast, toastError } from '../components/toast.js';
import { copyText } from '../dom.js';

const TASKS = ['', 'text-generation', 'image-text-to-text', 'feature-extraction', 'automatic-speech-recognition'];
const QUANTS = ['', 'fp8', 'nvfp4', 'awq', 'gptq', 'mxfp4', 'gguf', 'bf16'];

export function pickInfo(d) {
  const info = d?.info || d?.model_info || d || {};
  return { info, card: d?.card ?? d?.card_text ?? d?.readme ?? '', files: d?.files || [] };
}

class EcModels extends EcView {
  setup() {
    this.tab = this.query?.tab === 'local' ? 'local' : 'hf';
    this.f = { q: '', task: '', quant: '', max_gib: '', gated: '', sort: 'downloads' };
    this.hwTotals = [];
    this.api.hardware().then((hw) => { this.hwTotals = (hw.gpus || []).map((g) => g.total_gib); if (this.results) this.renderResults(); }).catch(() => {});
    this.tabs = h('div', { class: 'tabs', role: 'tablist', 'aria-label': t('models.tabs') });
    this.body = h('div', { class: 'tabpanel', role: 'tabpanel' });
    this.append(h('h1', t('nav.models')), this.tabs, this.body);
    this.paintTabs(); this.show();
  }
  paintTabs() {
    clear(this.tabs).append(...[['hf', 'models.tab_hf'], ['local', 'models.tab_local']].map(([id, k]) =>
      h('button', { type: 'button', role: 'tab', 'aria-selected': String(this.tab === id), class: `tab${this.tab === id ? ' on' : ''}`, onClick: () => { this.tab = id; this.paintTabs(); this.show(); } }, t(k))));
  }
  show() { this.tab === 'hf' ? this.showHF() : this.showLocal(); }

  // ---------- Hugging Face search ----------
  showHF() {
    const f = this.f;
    const run = debounce(() => this.search(), 300);
    const bind = (k) => (e) => { f[k] = e.target.value; run(); };
    const q = h('input', { type: 'search', value: f.q, placeholder: t('models.search_ph'), 'aria-label': t('models.search'), onInput: bind('q'), autocomplete: 'off' });
    const filters = h('form', { class: 'filters', onSubmit: (e) => { e.preventDefault(); run.cancel(); this.search(); } },
      h('div', { class: 'field grow' }, h('label', { for: 'hfq' }, t('models.search')), Object.assign(q, { id: 'hfq' })),
      field(t('models.task'), select(TASKS.map((v) => ({ value: v, label: v || t('common.any') })), f.task, (v) => { f.task = v; run(); })),
      field(t('models.quant'), select(QUANTS.map((v) => ({ value: v, label: v || t('common.any') })), f.quant, (v) => { f.quant = v; run(); })),
      field(t('models.max_gib'), h('input', { type: 'number', min: 0, step: 1, value: f.max_gib, onInput: bind('max_gib'), placeholder: '96' })),
      field(t('models.gated'), select([{ value: '', label: t('common.any') }, { value: 'false', label: t('common.no') }, { value: 'true', label: t('common.yes') }], f.gated, (v) => { f.gated = v; run(); })),
      field(t('models.sort'), select(['downloads', 'likes', 'updated'].map((v) => ({ value: v, label: t(`models.sort_${v}`) })), f.sort, (v) => { f.sort = v; run(); })));
    this.resultsHost = h('div', { 'aria-live': 'polite' });
    clear(this.body).append(filters, this.resultsHost);
    this.search();
  }
  async search() {
    const my = (this._s = (this._s || 0) + 1);
    clear(this.resultsHost).append(skeleton(5));
    try {
      const r = await this.api.hfSearch(this.f.q, { task: this.f.task, quant: this.f.quant, max_gib: this.f.max_gib, gated: this.f.gated, sort: this.f.sort, limit: 30 });
      if (my !== this._s || !this._alive) return;
      this.results = items(r).map(normHit); this.renderResults();
    } catch (e) { if (my === this._s && this._alive) clear(this.resultsHost).append(errorBox(e, () => this.search())); }
  }
  fitBadge(m) {
    const q = quickFit(m.size_bytes ?? m.weight_bytes, this.hwTotals);
    if (q.verdict === 'unknown') return badge(t('fit.unknown'), 'muted');
    const vm = verdictMeta(q.verdict);
    return badge(`${t(vm.label)}${q.gpus ? ` · ${q.gpus} GPU` : ''} (${t('fit.est')})`, vm.cls, t('fit.est_tip'));
  }
  renderResults() {
    if (!this.results.length) { clear(this.resultsHost).append(emptyBox(t('models.no_results'), t('models.no_results_hint'))); return; }
    clear(this.resultsHost).append(h('div', { class: 'tablewrap' }, h('table',
      h('thead', h('tr', ['models.col_model', 'models.col_size', 'models.col_downloads', 'models.col_fit', ''].map((k) => h('th', { scope: 'col' }, k ? t(k) : '')))),
      h('tbody', this.results.map((m) => h('tr', {},
        h('td', h('button', { type: 'button', class: 'link', onClick: () => this.openCard(m.repo_id) }, m.repo_id),
          h('div', { class: 'row gap wrap small' }, m.pipeline_tag ? badge(m.pipeline_tag, 'muted') : null, m.quant ? badge(m.quant, 'info') : null, m.gated ? badge(t('models.gated_badge'), 'warn', t('models.gated_tip')) : null)),
        h('td', { class: 'num' }, m.size_bytes ? fmtBytes(m.size_bytes) : '–'),
        h('td', { class: 'num' }, `${fmtCompact(m.downloads)} ↓ · ${fmtCompact(m.likes)} ♥`),
        h('td', this.fitBadge(m)),
        h('td', { class: 'row gap end' },
          btn(t('common.details'), { size: 'sm', onClick: () => this.openCard(m.repo_id) }),
          btn(t('models.download'), { size: 'sm', kind: 'primary', icon: 'download', onClick: () => this.download(m.repo_id) }))))))));
  }
  async download(repo) {
    try { await this.api.startDownload({ repo_id: repo }); toast(t('dl.queued_named', { repo }), { kind: 'ok' }); }
    catch (e) { toastError(e, t('models.download')); }
  }

  // ---------- model card drawer ----------
  openCard(repo) {
    const body = h('div', { class: 'stack-v' }, skeleton(4));
    const dr = openDrawer(repo, body);
    (async () => {
      try {
        const [detail, engines, hw] = await Promise.all([this.api.hfModel(repo), this.api.engines().catch(() => []), this.api.hardware().catch(() => null)]);
        const { info, card, files } = pickInfo(detail);
        const fitHost = h('div', {}, skeleton(2));
        const cardEl = h('div', { class: 'md' });
        setTrustedHtml(cardEl, renderMarkdown(card || '_No model card._', { copyLabel: t('common.copy') }));
        cardEl.addEventListener('click', (e) => { const b = e.target.closest('[data-action="copy-code"]'); if (b) copyText(b.closest('.codeblock').querySelector('code').textContent).then(() => toast(t('common.copied'), { kind: 'ok' })); });
        clear(body).append(
          h('div', { class: 'row gap wrap' },
            btn(t('models.download'), { kind: 'primary', icon: 'download', onClick: () => this.download(repo) }),
            btn(t('nav.launch'), { icon: 'launch', onClick: () => { dr.close(); location.hash = `#/launch?repo=${encodeURIComponent(repo)}`; } })),
          info.gated ? h('p', { class: 'note warn' }, t('models.gated_note')) : null,
          h('dl', { class: 'facts' },
            kv(t('models.f_arch'), (info.architectures || []).join(', ') || '–'),
            kv(t('models.f_params'), info.num_params ? `${fmtCompact(info.num_params)}${info.is_moe && info.num_active_params ? ` (${fmtCompact(info.num_active_params)} ${t('models.active')})` : ''}` : '–'),
            kv(t('models.f_quant'), info.quantization || info.dtype || '–'),
            kv(t('models.f_ctx'), info.max_position_embeddings ? Number(info.max_position_embeddings).toLocaleString('en-US') : '–'),
            kv(t('models.f_weights'), fmtBytes(info.weight_bytes)),
            kv(t('models.f_license'), info.license || '–')),
          h('h3', t('models.fit_matrix')), fitHost,
          files.length ? h('details', {}, h('summary', t('models.files', { count: files.length })), h('ul', { class: 'files' }, files.map((f) => h('li', h('code', f.path || f.name), h('span', { class: 'num muted' }, fmtBytes(f.size))))))
            : null,
          h('h3', t('models.card')), cardEl);
        this.fillFit(fitHost, repo, engines, hw);
      } catch (e) { clear(body).append(errorBox(e)); }
    })();
  }
  async fillFit(host, repo, engines, hw) {
    const gpus = hw?.gpus || [];
    if (!engines.length || !gpus.length) { clear(host).append(h('p', { class: 'hint' }, t('models.fit_unavailable'))); return; }
    const cells = await Promise.all(engines.map((e) => Promise.all(gpus.map((g) =>
      this.api.fit({ engine: e.id, repo_id: repo, params: {}, gpu_ids: [g.index] }).then((r) => r.verdict, () => 'unknown')))));
    if (!this._alive) return;
    clear(host).append(h('div', { class: 'tablewrap' }, h('table', { class: 'compact' },
      h('thead', h('tr', h('th', { scope: 'col' }, t('models.engine')), gpus.map((g) => h('th', { scope: 'col' }, `GPU ${g.index}`)))),
      h('tbody', engines.map((e, i) => h('tr', h('th', { scope: 'row' }, e.display_name || e.id), cells[i].map((v) => h('td', badge(t(verdictMeta(v).label), verdictMeta(v).cls)))))))),
      h('p', { class: 'hint' }, t('models.fit_hint')));
  }

  // ---------- local library ----------
  async showLocal() {
    const host = h('div', {}, skeleton(3));
    clear(this.body).append(host);
    const draw = async () => {
      try {
        const list = items(await this.api.models()).map(normLocalModel);
        if (!this._alive) return;
        if (!list.length) { clear(host).append(emptyBox(t('models.local_empty'), t('models.local_empty_hint'))); return; }
        clear(host).append(h('div', { class: 'tablewrap' }, h('table',
          h('thead', h('tr', ['models.col_model', 'models.col_size', 'models.last_used', 'models.col_fit', ''].map((k) => h('th', { scope: 'col' }, k ? t(k) : '')))),
          h('tbody', list.map((m) => h('tr', {},
            h('td', h('strong', m.repo_id)), h('td', { class: 'num' }, fmtBytes(m.size_bytes)), h('td', m.last_used ? timeAgo(m.last_used) : t('models.never')),
            h('td', { class: 'row gap wrap' }, Object.entries(m.engines_fit || {}).map(([e, v]) => badge(`${e}: ${t(verdictMeta(v).label)}`, verdictMeta(v).cls))),
            h('td', { class: 'row gap end' },
              btn(t('nav.launch'), { size: 'sm', kind: 'primary', icon: 'launch', onClick: () => { location.hash = `#/launch?repo=${encodeURIComponent(m.repo_id)}`; } }),
              btn('', { size: 'sm', icon: 'trash', title: t('models.delete'), onClick: async () => { if (await confirmDialog(t('models.confirm_delete', { repo: m.repo_id }), { danger: true })) { try { await this.api.deleteModel(m.repo_id); draw(); } catch (e) { toastError(e); } } } }))))))));
      } catch (e) { clear(host).append(errorBox(e, draw)); }
    };
    draw();
  }
}
customElements.define('ec-models', EcModels);
