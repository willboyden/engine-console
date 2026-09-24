// Small presentational helpers built on h().
import { h, icon } from '../dom.js';
import { t } from '../i18n.js';
import { describeError } from '../errors.js';

export const badge = (text, kind = 'muted', title) => h('span', { class: `badge ${kind}`, title }, text);
export function btn(label, { icon: ic, kind = '', size = '', onClick, title, type = 'button', disabled, ariaLabel } = {}) {
  return h('button', { type, class: `btn ${kind} ${size}`.trim(), onClick, title, disabled, 'aria-label': ariaLabel || (label ? null : title) },
    ic ? icon(ic, 16) : null, label ? h('span', label) : null);
}
export const card = (title, body, actions) => h('section', { class: 'card' },
  title || actions ? h('header', { class: 'card-head' }, h('h2', title), actions ? h('div', { class: 'row gap' }, actions) : null) : null,
  h('div', { class: 'card-body' }, body));

const STATE_KIND = { ready: 'ok', loading: 'info', starting: 'info', stopping: 'warn', stopped: 'muted', failed: 'bad',
  running: 'info', auth_required: 'warn', unreachable: 'warn', queued: 'muted', paused: 'warn', done: 'ok', cancelled: 'muted', completed: 'ok', error: 'bad' };
export const stateBadge = (s) => badge(t(`state.${s}`) === `state.${s}` ? s : t(`state.${s}`), STATE_KIND[s] || 'muted');

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
export function field(label, control, hint) {
  const id = control.id || `f${++uid}`;
  control.id = id;
  return h('div', { class: 'field' }, h('label', { for: id }, label), control, hint ? h('div', { class: 'hint' }, hint) : null);
}
export function select(options, value, onChange, attrs = {}) {
  return h('select', { ...attrs, onChange: (e) => onChange?.(e.target.value) },
    options.map((o) => { const v = typeof o === 'string' ? { value: o, label: o } : o; return h('option', { value: v.value, selected: v.value === value }, v.label); }));
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

// ---- external (monitor-only) instances ----
export const externalBadge = () => badge(t('inst.external'), 'info', t('inst.external_tip'));
// Informational note for auth_required / unreachable: these are states to explain, not errors to alarm about.
export function reasonNote(i) {
  if (i.state !== 'auth_required' && i.state !== 'unreachable') return null;
  const why = i.reason ?? i.state_reason ?? i.error;
  return h('p', { class: 'note warn', role: 'status' }, i.state === 'auth_required' ? t('inst.reason_auth') : t('inst.reason_unreach'), why ? [' ', h('strong', t('inst.reason_prefix')), ' ', why] : null);
}
// Engine type · image · served model names, shown on tiles and rows.
export function instanceFacts(i) {
  return h('div', { class: 'stack-v tight' },
    h('div', { class: 'muted small ellipsis', title: i.repo_id }, `${i.engine}${i.repo_id ? ` · ${i.repo_id}` : ''}`),
    i.image ? h('div', { class: 'muted small ellipsis', title: i.image }, h('code', i.image)) : null,
    (i.served_models || []).length ? h('div', { class: 'row gap wrap small' }, i.served_models.slice(0, 4).map((m) => badge(m, 'muted')), i.served_models.length > 4 ? h('span', { class: 'muted' }, `+${i.served_models.length - 4}`) : null) : null);
}
