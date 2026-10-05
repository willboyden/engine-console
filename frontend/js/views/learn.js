// Glossary: every term the learn bundle defines, with a search box, an anchor per term (#/learn?term=kv_cache deep-links and
// highlights it) and links between related terms. Text comes from the bundle (untrusted): text nodes only.
import { EcView } from '../components/base.js';
import { h, clear, icon } from '../dom.js';
import { t } from '../i18n.js';
import { getBundle, glossaryEntries, searchGlossary, glossaryLetters, paragraphs, termTitle } from '../learn.js';
import { debounce } from '../format.js';
import { emptyBox, pageHead } from '../components/ui.js';

const enc = encodeURIComponent;
const anchorId = (key) => `term-${key}`;

class EcLearn extends EcView {
  setup() {
    this.entries = glossaryEntries(getBundle());
    this.target = this.query?.term || '';
    this.q = this.query?.q || '';
    const input = h('input', { type: 'search', id: 'gl-q', value: this.q, placeholder: t('learn.search_ph'), 'aria-label': t('learn.search'), autocomplete: 'off', onInput: debounce((e) => { this.q = e.target.value; this.paint(); }, 150) });
    this.status = h('p', { class: 'muted small', role: 'status' });
    this.letters = h('nav', { class: 'letter-bar', 'aria-label': t('learn.letters') });
    this.list = h('div', { class: 'glossary' });
    this.append(pageHead({ page: 'learn', title: t('nav.learn') }),
      h('div', { class: 'field' }, h('label', { for: 'gl-q' }, t('learn.search')), input), this.status, this.letters, this.list);
    this.paint();
    if (this.target) this.focusTarget();
  }
  paint() {
    if (!this.entries.length) { clear(this.list).append(emptyBox(t('learn.empty'), t('learn.empty_hint'))); clear(this.letters); clear(this.status); return; }
    const list = searchGlossary(this.entries, this.q), shown = list;
    this.status.textContent = this.q ? t('learn.found', { count: shown.length, total: this.entries.length }) : t('learn.count', { count: this.entries.length });
    clear(this.letters).append(...(this.q ? [] : glossaryLetters(list).map((l) => h('a', { href: `#/learn${this.target ? `?term=${enc(this.target)}` : ''}`, class: 'letter', onClick: (e) => { e.preventDefault(); this.querySelector(`[data-letter="${l}"]`)?.scrollIntoView({ block: 'start' }); } }, l))));
    clear(this.list).append(...(list.length ? list.map((e) => this.term(e)) : [emptyBox(t('learn.no_match'), t('learn.no_match_hint'))]));
  }
  term(e) {
    const on = e.key === this.target;
    return h('article', { class: `term${on ? ' target' : ''}`, id: anchorId(e.key), dataset: { letter: e.title.charAt(0).toUpperCase() }, 'aria-labelledby': `${anchorId(e.key)}-h` },
      h('h2', { id: `${anchorId(e.key)}-h`, tabindex: '-1' }, e.title, h('a', { class: 'permalink', href: `#/learn?term=${enc(e.key)}`, 'aria-label': t('learn.permalink', { term: e.title }), title: t('learn.permalink', { term: e.title }) }, icon('link', 14))),
      h('p', { class: 'term-short' }, e.short),
      ...paragraphs(e.long).map((p) => h('p', p)),
      e.see_also.length ? h('p', { class: 'small see-also' }, h('strong', `${t('learn.see_also')}: `), ...e.see_also.flatMap((k, i) => [i ? ', ' : null, h('a', { href: `#/learn?term=${enc(k)}` }, termTitle(k))])) : null);
  }
  focusTarget() {
    requestAnimationFrame(() => {
      const el = this.querySelector(`#${CSS.escape(anchorId(this.target))}`);
      if (!el) return;
      el.scrollIntoView({ block: 'start' });
      el.querySelector('h2')?.focus({ preventScroll: true });
    });
  }
}
customElements.define('ec-learn', EcLearn);
