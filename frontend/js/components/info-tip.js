// Learn mode: the info button (ⓘ) and its popover, plus the inline summary shown in beginner mode.
//
// The button is an accessible <button aria-label="What is X?"> that opens a native popover (Popover API, light dismiss:
// Esc and a click outside close it). The popover is positioned beside the button with CSSOM style properties (allowed by
// the strict CSP; no style attributes), takes focus when it opens and gives it back to the button when it closes. It shows
// the title, summary, details, "When to change it", "What can go wrong", an example and the glossary terms the text relies
// on; clicking a term shows that term in the same popover with a back link and a link to the Glossary page.
// Text comes from the learn bundle (untrusted): it is only ever added as text nodes.
import { h, icon, clear, append } from '../dom.js';
import { t } from '../i18n.js';
import { lookup, popoverModel, termModel, termTitle } from '../learn.js';
import { isBeginner } from '../beginner.js';

let seq = 0;
// Native Popover API when the browser has it. `?nopopover` (or localStorage ec.nopopover=1) forces the fallback, which is how the
// fallback path is exercised and screenshotted on a browser that has the API.
const forced = () => { try { return /[?&]nopopover\b/.test(globalThis.location?.search || '') || globalThis.localStorage?.getItem('ec.nopopover') === '1'; } catch { return false; } };
const canPopover = () => typeof HTMLElement !== 'undefined' && typeof HTMLElement.prototype.showPopover === 'function' && !forced();
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const GAP = 8;

/** Where the popover goes: beside the button when there is room (right, then left), else below or above, kept on screen. */
export function placePopover({ anchor, width, height, vw, vh }) {
  let left, top;
  if (anchor.right + GAP + width <= vw - GAP) { left = anchor.right + GAP; top = anchor.top - GAP; }
  else if (anchor.left - GAP - width >= GAP) { left = anchor.left - GAP - width; top = anchor.top - GAP; }
  else {
    left = anchor.left + anchor.width / 2 - width / 2;
    top = anchor.bottom + GAP;
    if (top + height > vh - GAP) top = anchor.top - GAP - height;
  }
  return { left: Math.round(clamp(left, GAP, Math.max(GAP, vw - width - GAP))), top: Math.round(clamp(top, GAP, Math.max(GAP, vh - height - GAP))) };
}

/**
 * infoTip({kind, key, engine, spec, title, explain}) -> the ⓘ button (inside a span) or null when there is no explanation.
 * kind: concept | metric | param (give `spec` and `engine`) | preset | term.
 */
export function infoTip({ kind = 'concept', key, engine, spec, title, explain } = {}) {
  const ex = explain || lookup({ kind, key, engine, spec });
  if (!ex) return null;
  const name = title || (kind === 'term' ? termTitle(key) : spec?.label || key);
  const id = `tip${++seq}`;
  let stack = [];
  const native = canPopover();
  const btn = h('button', { type: 'button', class: 'info-btn', 'aria-label': t('learn.what_is', { title: name }), 'aria-expanded': 'false', 'aria-controls': id, 'aria-haspopup': 'dialog' }, icon('info', 16));
  const pop = h('div', { id, class: 'info-pop', role: 'dialog', tabindex: '-1', 'aria-labelledby': `${id}-t` });
  const wrap = h('span', { class: 'info-tip' }, btn, pop);
  const isOpen = () => (native ? pop.matches(':popover-open') : !pop.hidden);

  const chip = (tm) => h('button', { type: 'button', class: 'term-chip', onClick: () => { stack.push(tm.key); render(); place(); pop.focus({ preventScroll: true }); } }, tm.title);
  const closeBtn = () => h('button', { type: 'button', class: 'btn ghost sm pop-close', 'aria-label': t('common.close'), onClick: () => close() }, icon('close', 14));

  function render() {
    clear(pop);
    const view = stack.at(-1);
    if (view === undefined) {
      const m = popoverModel(ex, { title: name });
      append(pop, [
        h('header', { class: 'pop-head' }, h('h3', { id: `${id}-t`, class: 'pop-title' }, name), closeBtn()),
        h('p', { class: 'pop-summary' }, m.summary),
        ...m.sections.map((s) => h('section', { class: `pop-sec pop-${s.id}` }, s.label ? h('h4', t(s.label)) : null, ...s.paragraphs.map((p) => h('p', p)))),
        m.terms.length ? h('section', { class: 'pop-sec pop-terms' }, h('h4', t('learn.related_terms')), h('div', { class: 'term-chips' }, m.terms.map(chip))) : null,
        h('footer', { class: 'pop-foot' }, kind === 'param' ? h('span', { class: 'muted small' }, t(`learn.level_${ex.level}`)) : null,
          h('a', { href: '#/learn', onClick: () => close() }, t('learn.open_glossary')))]);
      return;
    }
    const tm = termModel(view);
    append(pop, [
      h('header', { class: 'pop-head' }, h('button', { type: 'button', class: 'btn ghost sm', onClick: () => { stack.pop(); render(); place(); pop.focus({ preventScroll: true }); } }, icon('arrowleft', 14), h('span', t('learn.back'))), closeBtn()),
      h('h3', { id: `${id}-t`, class: 'pop-title' }, tm ? tm.title : termTitle(view)),
      tm ? [h('p', { class: 'pop-summary' }, tm.summary), ...tm.paragraphs.map((p) => h('p', p)),
        tm.seeAlso.length ? h('section', { class: 'pop-sec pop-terms' }, h('h4', t('learn.see_also')), h('div', { class: 'term-chips' }, tm.seeAlso.map(chip))) : null] : h('p', { class: 'muted' }, t('learn.no_term')),
      h('footer', { class: 'pop-foot' }, h('a', { href: `#/learn?term=${encodeURIComponent(view)}`, onClick: () => close() }, t('learn.open_term_in_glossary')))]);
  }

  function place() {
    if (!isOpen()) return;
    const vw = document.documentElement.clientWidth, vh = document.documentElement.clientHeight;
    pop.style.maxHeight = `${Math.min(vh - 2 * GAP, 560)}px`;
    const { left, top } = placePopover({ anchor: btn.getBoundingClientRect(), width: pop.offsetWidth, height: pop.offsetHeight, vw, vh });
    pop.style.left = `${left}px`; pop.style.top = `${top}px`;
  }
  function afterClose() {
    stack = [];
    btn.setAttribute('aria-expanded', 'false');
    // Give focus back to the button, unless the user already moved on to something else (a click outside).
    const a = document.activeElement;
    if (!a || a === document.body || pop.contains(a)) btn.focus({ preventScroll: true });
  }
  function close() { if (!isOpen()) return; if (native) pop.hidePopover(); else { pop.hidden = true; afterClose(); } }
  function open() {
    stack = [];
    render();
    btn.setAttribute('aria-expanded', 'true');
    if (!native) pop.hidden = false;
    place(); pop.focus({ preventScroll: true });
  }

  if (native) {
    pop.setAttribute('popover', 'auto');
    btn.setAttribute('popovertarget', id);
    btn.setAttribute('popovertargetaction', 'toggle');
    // Light dismiss and the browser's own toggling do the opening and closing; these hooks fill, place and focus.
    pop.addEventListener('beforetoggle', (e) => { if (e.newState === 'open') { stack = []; render(); } });
    pop.addEventListener('toggle', (e) => { if (e.newState === 'open') { btn.setAttribute('aria-expanded', 'true'); place(); pop.focus({ preventScroll: true }); } else afterClose(); });
  } else {
    pop.hidden = true;
    btn.addEventListener('click', () => (isOpen() ? close() : open()));
    const onDoc = (e) => { if (isOpen() && !wrap.contains(e.target)) close(); };
    document.addEventListener('click', onDoc);
    pop.addEventListener('keydown', (e) => { if (e.key === 'Escape') close(); });
  }
  btn.addEventListener('click', (e) => e.stopPropagation());
  return wrap;
}

/**
 * True while an info popover (native or fallback) is open inside `root`, or focus is inside it: a live view must not repaint
 * what the reader is looking at.
 */
export function busyWithin(root) {
  if (!root) return false;
  let open = false;
  try { open = !!root.querySelector(':popover-open, .info-pop:not([popover]):not([hidden])'); } catch { open = !!root.querySelector('.info-pop:not([popover]):not([hidden])'); }
  return open || root.matches(':focus-within');
}

/** The summary as a quiet line under a label, only in beginner mode and only when there is one. */
export function inlineHelp(ref, { className = '' } = {}) {
  if (!isBeginner()) return null;
  const ex = ref.explain || lookup(ref);
  return ex ? h('p', { class: `learn-inline ${className}`.trim() }, ex.summary) : null;
}

/** A label (text or node) followed by its info button; just the label when nothing is known. */
export function withTip(content, ref, title) {
  const tip = infoTip({ ...ref, title: title ?? (typeof content === 'string' ? content : undefined) });
  return tip ? h('span', { class: 'with-tip' }, content, tip) : content;
}
