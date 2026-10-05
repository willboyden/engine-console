// Small presentational helpers built on h().
import { h, icon } from '../dom.js';
import { t } from '../i18n.js';
import { describeError } from '../errors.js';
import { infoTip, inlineHelp, withTip } from './info-tip.js';
import { linkKind } from '../fleet.js';
import { verdictMeta } from '../fit-format.js';

export const badge = (text, kind = 'muted', title) => h('span', { class: `badge ${kind}`, title }, text);
/** "warming_up" / "host-ram-low" -> "Warming up" / "Host ram low": a readable form of an id nobody wrote copy for. */
export const humanize = (id) => {
  const s = String(id ?? '').replace(/[_-]+/g, ' ').replace(/\s+/g, ' ').trim();
  return s ? s.charAt(0).toUpperCase() + s.slice(1) : '';
};
/**
 * i18n text for `prefix.id`. Codes and ids that come from the server (a startup phase an adapter added, a new alert code) may have
 * no entry yet: they fall back to a readable form of the raw value, never to a missing-key string.
 */
export const tr = (prefix, id) => { const k = `${prefix}.${id}`; const v = t(k); return v === k ? humanize(id) : v; };

// A reason ({text} from the server, or {key, vars} for the UI's own wording) as display text.
export const reasonText = (r) => (!r ? '' : typeof r === 'string' ? r : r.text != null ? String(r.text) : r.key ? t(r.key, r.vars) : '');

let whySeq = 0;
/**
 * Button. A disabled button that has a `reason` stays focusable (aria-disabled) and carries the reason as its accessible
 * description (and tooltip), so a keyboard or screen-reader user can find out why; without a reason it is a plain disabled one.
 */
export function btn(label, { icon: ic, kind = '', size = '', onClick, title, type = 'button', disabled, ariaLabel, reason } = {}) {
  const why = disabled ? reasonText(reason) : '';
  const soft = !!why;
  const id = soft ? `why${++whySeq}` : null;
  const el = h('button', { type, class: `btn ${kind} ${size}`.trim(), onClick: soft ? undefined : onClick, title: why || title, disabled: disabled && !soft, 'aria-disabled': soft ? 'true' : null, 'aria-describedby': id,
    'aria-label': ariaLabel || (label ? null : title) }, ic ? icon(ic, 16) : null, label ? h('span', label) : null, soft ? h('span', { id, hidden: true }, why) : null);
  if (soft) el.addEventListener('click', (e) => e.preventDefault());
  return el;
}

export const card = (title, body, actions, { tip } = {}) => h('section', { class: 'card' },
  title || actions ? h('header', { class: 'card-head' }, h('h2', title, tip || null), actions ? h('div', { class: 'row gap' }, actions) : null) : null,
  h('div', { class: 'card-body' }, body));

const STATE_KIND = { ready: 'ok', loading: 'info', starting: 'info', stopping: 'warn', stopped: 'muted', failed: 'bad',
  running: 'info', auth_required: 'warn', unreachable: 'warn', queued: 'muted', paused: 'warn', done: 'ok', cancelled: 'muted', completed: 'ok', error: 'bad' };
export const stateBadge = (s, { tip = false } = {}) => {
  const label = tr('state', s);
  const b = badge(label, STATE_KIND[s] || 'muted');
  return tip ? withTip(b, { kind: 'concept', key: `state.${s}` }, label) : b;
};

// ---- fleet badges, each with its info button ----
export const linkBadge = (state, { tip = true } = {}) => {
  const label = tr('link', state);
  const b = badge(label, linkKind(state));
  return tip ? withTip(b, { kind: 'concept', key: `link.${state}` }, label) : b;
};
export const modeBadge = (mode, { tip = true } = {}) => {
  if (!mode) return null;
  const label = t(`mode.${mode}`);
  const b = badge(label, mode === 'manage' ? 'ok' : 'muted', t(`mode.${mode}_tip`));
  return tip ? withTip(b, { kind: 'concept', key: 'field.policy_mode' }, t('mode.title')) : b;
};
const MGMT_KIND = { managed: 'info', script: 'purple', external: 'muted' };
export const managementBadge = (m, { tip = true } = {}) => {
  const label = tr('management', m);
  const b = badge(label, MGMT_KIND[m] || 'muted');
  return tip ? withTip(b, { kind: 'concept', key: `management.${m}` }, label) : b;
};
export const formatBadge = (f, { tip = true } = {}) => {
  const label = tr('format', f);
  const b = badge(label, 'info');
  return tip ? withTip(b, { kind: 'concept', key: `format.${f}` }, label) : b;
};
export const rootKindBadge = (kind, { tip = true } = {}) => {
  const label = tr('root', kind);
  const b = badge(label, 'muted');
  return tip ? withTip(b, { kind: 'concept', key: `root.${kind}` }, label) : b;
};
export const verdictBadge = (v, { tip = true, text } = {}) => {
  const vm = verdictMeta(v);
  const label = text || t(vm.label);
  const b = badge(label, vm.cls);
  return tip ? withTip(b, { kind: 'concept', key: `verdict.${v}` }, label) : b;
};

export function progress(pct, { kind = '', label } = {}) {
  const p = Math.max(0, Math.min(100, Number.isFinite(pct) ? pct : 0));
  const fill = h('div', { class: `fill ${kind}` });
  fill.style.width = `${p}%`;
  return h('div', { class: 'bar', role: 'progressbar', 'aria-valuemin': 0, 'aria-valuemax': 100, 'aria-valuenow': Math.round(p), 'aria-label': label || null }, fill);
}
export function setProgress(bar, pct) {
  const p = Math.max(0, Math.min(100, Number.isFinite(pct) ? pct : 0));
  bar.firstChild.style.width = `${p}%`; bar.setAttribute('aria-valuenow', Math.round(p));
}

let uid = 0;
/**
 * Labelled form field. `learn` (optional) is a learn ref {kind, key, ...}: the label gets an info button and, in beginner
 * mode, the summary appears under it.
 */
export function field(label, control, hint, learn) {
  const id = control.id || `f${++uid}`;
  control.id = id;
  const lab = h('label', { for: id }, label);
  const tip = learn ? infoTip({ ...learn, title: label }) : null;
  return h('div', { class: 'field' }, tip ? h('div', { class: 'label-row' }, lab, tip) : lab, learn && learn.inline !== false ? inlineHelp(learn) : null, control, hint ? h('div', { class: 'hint' }, hint) : null);
}
const stackedBefore = new Map(); // key -> whether that table was stacked last time, so a repainted table does not flash unstacked
/**
 * Narrow-container layout for a table inside a `.tablewrap`: when the space it is given is smaller than the table's own minimum
 * width (a phone, a tablet, a half-width window), every row becomes a card with its column title above each value instead of
 * the table scrolling sideways. The stacking is CSS keyed on `.stack-on`, which a ResizeObserver toggles by comparing the host's
 * width with the table's measured minimum width; this also supplies the titles from the header cells and re-asserts the table
 * roles that `display: block` can strip in some screen readers. `key` names the table so a repainted one starts in the state its
 * predecessor settled in. Returns the element to insert.
 */
export function stackTable(wrap, opts = {}) {
  const table = wrap?.querySelector?.('table');
  if (!table) return wrap;
  const key = opts.key || wrap.dataset?.stackKey || '';
  const heads = [...table.querySelectorAll('thead th')].map((th) => (th.classList.contains('sr-th') ? '' : th.textContent.trim()));
  wrap.classList.add('stacked');
  table.setAttribute('role', 'table');
  table.querySelectorAll('thead, tbody').forEach((g) => g.setAttribute('role', 'rowgroup'));
  table.querySelectorAll('tr').forEach((tr) => [...tr.children].forEach((c, i) => {
    tr.setAttribute('role', 'row');
    if (c.tagName === 'TH') { c.setAttribute('role', 'columnheader'); return; }
    c.setAttribute('role', 'cell');
    if (heads[i]) c.setAttribute('data-label', heads[i]);
  }));
  const host = h('div', { class: `stack-host${key && stackedBefore.get(key) ? ' stack-on' : ''}` }, wrap);
  if (typeof ResizeObserver === 'function') {
    let raf = 0;
    // The class change alters the host's height, so it is applied on the next frame rather than inside the observer callback
    // (changing observed sizes there raises "ResizeObserver loop completed with undelivered notifications").
    const update = () => {
      raf = 0;
      if (!host.isConnected) { ro.disconnect(); return; }
      if (host.clientWidth <= 0) return;
      // Measure as a real table with every column as narrow as it can be (small screens style tables as scrolling blocks), then
      // settle the class; all in one task, so nothing is painted in between.
      host.classList.remove('stack-on');
      Object.assign(table.style, { display: 'table', width: 'min-content', maxWidth: 'none' });
      const need = Math.ceil(table.getBoundingClientRect().width);
      Object.assign(table.style, { display: '', width: '', maxWidth: '' });
      const on = host.clientWidth + 2 < need;
      host.classList.toggle('stack-on', on);
      if (key) stackedBefore.set(key, on);
    };
    const ro = new ResizeObserver(() => { if (!raf) raf = requestAnimationFrame(update); });
    ro.observe(host);
  }
  return host;
}
export function select(options, value, onChange, attrs = {}) {
  return h('select', { ...attrs, onChange: (e) => onChange?.(e.target.value) },
    options.map((o) => { const v = typeof o === 'string' ? { value: o, label: o } : o; return h('option', { value: v.value, selected: v.value === value, disabled: v.disabled }, v.label); }));
}

export const skeleton = (n = 3) => h('div', { class: 'skeleton-group', 'aria-busy': 'true', 'aria-label': t('common.loading') },
  Array.from({ length: n }, () => h('div', { class: 'skeleton' })));
export function errorBox(err, retry) {
  return h('div', { class: 'state error', role: 'alert' }, icon('warn', 22),
    h('div', h('strong', err?.title || t('common.error')), h('p', describeError(err))),
    err?.status === 401 ? h('p', { class: 'hint' }, t('auth.required')) : null,
    retry ? btn(t('common.retry'), { icon: 'restart', onClick: retry }) : null);
}
export const emptyBox = (title, hint, action) => h('div', { class: 'state empty' }, h('strong', title), hint ? h('p', hint) : null, action || null);
export const kv = (k, v) => h('div', { class: 'kv' }, h('dt', k), h('dd', v));
/** A wrapping grid of small label-over-value facts: [[label, value], ...] (null entries are skipped). */
export const factGrid = (pairs) => h('dl', { class: 'fact-grid' }, pairs.filter(Boolean).map(([k, v]) => h('div', { class: 'fact' }, h('dt', k), h('dd', v))));
/** kv row whose key has an info button. */
export const kvTip = (k, ref, v) => h('div', { class: 'kv' }, h('dt', withTip(k, ref, k)), h('dd', v));

/** Page heading with its info button and (beginner mode) the one-line purpose of the page. `page` is the page.<name> key. */
export function pageHead({ page, title, actions, back }) {
  const ref = page ? { kind: 'concept', key: `page.${page}` } : null;
  return h('div', { class: 'page-head' },
    back ? h('nav', { class: 'crumbs', 'aria-label': t('common.breadcrumb') }, h('a', { href: back.href }, back.label), ' / ', title) : null,
    h('div', { class: 'page-head-row' }, h('h1', {}, title), ref ? infoTip({ ...ref, title }) : null, actions ? h('div', { class: 'page-actions row gap wrap' }, actions) : null),
    ref ? inlineHelp(ref, { className: 'page-intro' }) : null);
}

// ---- external (monitor-only) instances ----
export const externalBadge = () => badge(t('inst.external'), 'info', t('inst.external_tip'));
// Informational note for auth_required / unreachable: these are states to explain, not errors to alarm about.
export function reasonNote(i) {
  if (i.state !== 'auth_required' && i.state !== 'unreachable') return null;
  const why = i.reason ?? i.state_reason ?? i.error;
  return h('p', { class: 'note warn', role: 'status' }, i.state === 'auth_required' ? t('inst.reason_auth') : t('inst.reason_unreach'), why ? [' ', h('strong', t('inst.reason_prefix')), ' ', why] : null);
}
// Engine type · image · served model names, shown on tiles and rows.
export function instanceFacts(i, { title } = {}) {
  // The served models the title already names are not repeated as badges.
  const served = (i.served_models || []).filter((m) => m !== title && m !== i.repo_id);
  return h('div', { class: 'stack-v tight' },
    h('div', { class: 'muted small ellipsis', title: i.repo_id }, `${i.engine}${i.repo_id ? ` · ${i.repo_id}` : ''}`),
    i.image ? h('div', { class: 'muted small ellipsis', title: i.image }, h('code', i.image)) : null,
    served.length ? h('div', { class: 'row gap wrap small' }, served.slice(0, 4).map((m) => badge(m, 'muted')), served.length > 4 ? h('span', { class: 'muted' }, `+${served.length - 4}`) : null) : null);
}
