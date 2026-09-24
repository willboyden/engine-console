import { EcView } from '../components/base.js';
import { h, clear, setTrustedHtml, download } from '../dom.js';
import { t } from '../i18n.js';
import { items, fmtCompact, csvCell } from '../format.js';
import { toGrid } from '../adapt.js';
import { heatmap, hbars, lineChart } from '../charts.js';
import { btn, card, emptyBox, errorBox, skeleton } from '../components/ui.js';

class EcUsage extends EcView {
  setup() {
    this.hostHeat = h('div', {}, skeleton(3)); this.hostModel = h('div', {}, skeleton(3)); this.hostDay = h('div', {}, skeleton(3));
    this.append(h('div', { class: 'row between' }, h('h1', t('nav.usage')), btn(t('usage.export'), { icon: 'download', onClick: () => this.exportCsv() })),
      card(t('usage.heatmap'), this.hostHeat), h('div', { class: 'grid two' }, card(t('usage.by_model'), this.hostModel), card(t('usage.by_day'), this.hostDay)));
    this.load();
  }
  async load() {
    // usage endpoint returns {rows:[...]} or a bare list; unwrap once per request.
    const fetchRows = async (g) => { const r = await this.api.usage(g); return Array.isArray(r) ? r : r.rows || r.items || []; };
    const go = async (host, g, draw) => { try { const rows = await fetchRows(g); if (!this._alive) return; if (!rows.length) clear(host).append(emptyBox(t('usage.empty'), t('usage.empty_hint'))); else draw(rows); } catch (e) { if (this._alive) clear(host).append(errorBox(e, () => this.load())); } };
    go(this.hostHeat, 'hour', (rows) => { const el = h('div', { class: 'heat-wrap' }); setTrustedHtml(el, heatmap(toGrid(rows), { label: t('usage.requests') })); clear(this.hostHeat).append(el); });
    go(this.hostModel, 'model', (rows) => {
      this.modelRows = rows;
      const el = h('div'); setTrustedHtml(el, hbars(rows.map((r) => ({ label: r.key, value: (r.prompt_tokens || 0) + (r.completion_tokens || 0) })), { fmt: fmtCompact }));
      clear(this.hostModel).append(el, h('div', { class: 'tablewrap' }, h('table', { class: 'compact' },
        h('thead', h('tr', ['usage.model', 'usage.col_requests', 'usage.prompt', 'usage.completion'].map((k) => h('th', { scope: 'col' }, t(k))))),
        h('tbody', rows.map((r) => h('tr', h('td', r.key), h('td', { class: 'num' }, fmtCompact(r.requests)), h('td', { class: 'num' }, fmtCompact(r.prompt_tokens)), h('td', { class: 'num' }, fmtCompact(r.completion_tokens))))))));
    });
    go(this.hostDay, 'day', (rows) => {
      const s = [{ name: t('usage.requests'), points: rows.map((r, i) => [i, r.requests]) }, { name: t('usage.tokens_k'), points: rows.map((r, i) => [i, ((r.prompt_tokens || 0) + (r.completion_tokens || 0)) / 1000]) }];
      const el = h('div'); setTrustedHtml(el, lineChart(s, { xFormat: (i) => (rows[Math.round(i)]?.key || '').slice(5), yFormat: (v) => fmtCompact(v), ariaLabel: t('usage.by_day') }));
      clear(this.hostDay).append(el);
    });
  }
  async exportCsv() {
    try {
      const res = await this.api.raw('GET', '/usage/export.csv');
      download('usage.csv', await res.text(), 'text/csv');
    } catch (e) { const rows = this.modelRows || []; download('usage.csv', ['model,requests,prompt_tokens,completion_tokens', ...rows.map((r) => [r.key, r.requests, r.prompt_tokens, r.completion_tokens].map(csvCell).join(','))].join('\n'), 'text/csv'); }
  }
}
customElements.define('ec-usage', EcUsage);
