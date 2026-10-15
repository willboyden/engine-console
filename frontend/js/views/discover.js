// Discover: search Hugging Face and download a model onto a machine of your choice (machine, folder and, for GGUF
// repositories, one quantization). Models you already have live in the Library.
import { EcView } from '../components/base.js';
import { h, clear, append, setTrustedHtml, copyText } from '../dom.js';
import { t } from '../i18n.js';
import { items, fmtBytes, fmtCompact, debounce } from '../format.js';
import { normHit, normHost } from '../adapt.js';
import { renderMarkdown } from '../markdown.js';
import { quickFit, verdictMeta } from '../fit-format.js';
import { enginesForFormat } from '../engines.js';
import { canManage } from '../fleet.js';
import { badge, btn, emptyBox, errorBox, skeleton, select, field, kv, pageHead, verdictBadge, stackTable } from '../components/ui.js';
import { openDrawer } from '../components/dialog.js';
import { openDownloadDialog } from '../components/download-dialog.js';
import { toast, toastError } from '../components/toast.js';
import { isRoleRefusal } from '../errors.js';

const TASKS = ['', 'text-generation', 'image-text-to-text', 'feature-extraction', 'automatic-speech-recognition'];
const QUANTS = ['', 'fp8', 'nvfp4', 'awq', 'gptq', 'mxfp4', 'gguf', 'bf16'];

export function pickInfo(d) {
  const info = d?.info || d?.model_info || d || {};
  return { info, card: d?.card ?? d?.card_text ?? d?.readme ?? '', files: d?.files || [], groups: d?.gguf_groups || [] };
}

/** Run async tasks with at most `n` in flight (a model card asks for one estimate per machine and engine). */
export async function pool(tasks, n = 4) {
  const out = new Array(tasks.length); let next = 0;
  await Promise.all(Array.from({ length: Math.min(n, tasks.length) }, async () => { while (next < tasks.length) { const i = next++; out[i] = await tasks[i](); } }));
  return out;
}

// Searching and reading model cards spend the operator's Hugging Face token, so the server allows them to admin keys only.
// A viewer gets a plain note and the way to the Library, not an error with a retry that would fail again.
const adminOnly = () => emptyBox(t('discover.admin_only'), t('discover.admin_only_hint'),
  h('a', { class: 'btn', href: '#/library' }, t('discover.to_library')));

class EcDiscover extends EcView {
  setup() {
    this.f = { q: '', task: '', quant: '', max_gib: '', gated: '', sort: 'downloads' };
    this.hosts = [];
    this.api.hosts().then((r) => { this.hosts = items(r).map(normHost); if (this.results) this.renderResults(); }).catch(() => {});
    this.body = h('div', { class: 'tabpanel' });
    this.append(pageHead({ page: 'discover', title: t('nav.discover'), actions: [h('a', { class: 'btn', href: '#/library' }, t('discover.to_library'))] }), this.body);
    this.showHF();
  }
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
    } catch (e) { if (my === this._s && this._alive) clear(this.resultsHost).append(isRoleRefusal(e) ? adminOnly() : errorBox(e, () => this.search())); }
  }
  /** Weights-only estimate against every machine that has GPUs: the best verdict and how many machines it holds on. */
  fitBadge(m) {
    const size = m.size_bytes ?? m.weight_bytes;
    const gpuHosts = this.hosts.filter((x) => x.link.state === 'up' && x.gpus.length);
    if (!size || !gpuHosts.length) return badge(t('fit.unknown'), 'muted');
    const per = gpuHosts.map((x) => quickFit(size, x.gpus.map((g) => g.total_gib)).verdict);
    const ok = per.filter((v) => v === 'fits' || v === 'tight').length;
    if (!ok) return badge(`${t('fit.wont')} (${t('fit.est')})`, 'bad', t('fit.est_tip'));
    const best = per.includes('fits') ? 'fits' : 'tight';
    const vm = verdictMeta(best);
    return badge(`${t(vm.label)} · ${t('discover.on_machines', { count: ok })} (${t('fit.est')})`, vm.cls, t('fit.est_tip'));
  }
  renderResults() {
    if (!this.results.length) { clear(this.resultsHost).append(emptyBox(t('models.no_results'), t('models.no_results_hint'))); return; }
    clear(this.resultsHost).append(stackTable(h('div', { class: 'tablewrap', 'data-stack-key': 'discover-results' }, h('table',
      h('thead', h('tr', ['models.col_model', 'models.col_size', 'models.col_downloads', 'models.col_fit', ''].map((k) => h('th', { scope: 'col' }, k ? t(k) : '')))),
      h('tbody', this.results.map((m) => h('tr', {},
        h('td', h('button', { type: 'button', class: 'link', onClick: () => this.openCard(m.repo_id) }, m.repo_id),
          h('div', { class: 'row gap wrap small' }, m.pipeline_tag ? badge(m.pipeline_tag, 'muted') : null, m.quant ? badge(m.quant, 'info') : null, m.gated ? badge(t('models.gated_badge'), 'warn', t('models.gated_tip')) : null)),
        h('td', { class: 'num' }, m.size_bytes ? fmtBytes(m.size_bytes) : '–'),
        h('td', { class: 'num' }, `${fmtCompact(m.downloads)} ↓ · ${fmtCompact(m.likes)} ♥`),
        h('td', this.fitBadge(m)),
        h('td', { class: 'row gap end' },
          btn(t('common.details'), { size: 'sm', onClick: () => this.openCard(m.repo_id) }),
          btn(t('models.download'), { size: 'sm', kind: 'primary', icon: 'download', onClick: () => this.download(m.repo_id) })))))))));
  }
  download(repo) { openDownloadDialog({ api: this.api, repo, hosts: this.hosts.length ? this.hosts : undefined, onQueued: () => { /* the Downloads page shows progress */ } }); }

  // ---------- model card drawer ----------
  openCard(repo) {
    const body = h('div', { class: 'stack-v' }, skeleton(4));
    const dr = openDrawer(repo, body);
    (async () => {
      try {
        const [detail, engines] = await Promise.all([this.api.hfModel(repo), this.api.engines().then(items).catch(() => [])]);
        const { info, card, files, groups } = pickInfo(detail);
        const fitHost = h('div', {}, skeleton(2));
        const cardEl = h('div', { class: 'md' });
        setTrustedHtml(cardEl, renderMarkdown(card || '_No model card._', { copyLabel: t('common.copy') }));
        cardEl.addEventListener('click', (e) => { const b = e.target.closest('[data-action="copy-code"]'); if (b) copyText(b.closest('.codeblock').querySelector('code').textContent).then(() => toast(t('common.copied'), { kind: 'ok' })); });
        append(clear(body), [
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
          groups.length ? [h('h3', t('discover.gguf_files')), h('div', { class: 'tablewrap' }, h('table', { class: 'compact' },
            h('thead', h('tr', ['dl.quant', 'models.col_size', 'dl.files_title'].map((k) => h('th', { scope: 'col' }, t(k))))),
            h('tbody', groups.map((g) => h('tr', h('th', { scope: 'row' }, g.quant), h('td', { class: 'num' }, fmtBytes(g.total_bytes, 1)), h('td', { class: 'num' }, g.files.length))))))] : [h('h3', t('models.fit_matrix')), fitHost],
          files.length ? h('details', {}, h('summary', t('models.files', { count: files.length })), h('ul', { class: 'files' }, files.map((f) => h('li', h('code', f.path || f.name), h('span', { class: 'num muted' }, fmtBytes(f.size))))))
            : null,
          h('h3', t('models.card')), cardEl]);
        if (!groups.length) this.fillFit(fitHost, repo, engines);
      } catch (e) { clear(body).append(isRoleRefusal(e) ? adminOnly() : errorBox(e)); }
    })();
  }
  /** One real estimate per machine (all its GPUs, default settings) and engine that loads Hugging Face models. */
  async fillFit(host, repo, engines) {
    const hosts = this.hosts.filter((x) => x.link.state === 'up' && x.gpus.length);
    const eng = enginesForFormat(engines, 'hf');
    if (!eng.length || !hosts.length) { clear(host).append(h('p', { class: 'hint' }, t('models.fit_unavailable'))); return; }
    const cells = new Map();
    await pool(hosts.flatMap((x) => eng.map((e) => async () => { cells.set(`${x.id}|${e.id}`, await this.api.fit({ engine: e.id, repo_id: repo, host_id: x.id, params: {}, gpu_ids: [] }).then((r) => r.verdict, () => 'unknown')); })), 4);
    if (!this._alive) return;
    clear(host).append(h('div', { class: 'tablewrap' }, h('table', { class: 'compact fit-matrix' },
      h('thead', h('tr', h('th', { scope: 'col' }, t('models.engine')), hosts.map((x) => h('th', { scope: 'col' }, x.label, canManage(x) ? null : h('div', { class: 'muted small' }, t('mode.monitor')))))),
      h('tbody', eng.map((e) => h('tr', h('th', { scope: 'row' }, e.display_name || e.id), hosts.map((x) => { const v = cells.get(`${x.id}|${e.id}`); return h('td', verdictBadge(v, { tip: false })); })))))),
    h('div', { class: 'row gap wrap verdict-legend' }, ['fits', 'tight', 'wont_fit'].map((v) => verdictBadge(v))),
    h('p', { class: 'hint' }, t('discover.fit_hint')));
  }
}
customElements.define('ec-discover', EcDiscover);
