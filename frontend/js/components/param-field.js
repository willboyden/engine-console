// DOM for the schema-driven parameter form. Logic lives in param-form.js (pure, tested).
import { h, icon } from '../dom.js';
import { t } from '../i18n.js';
import { filterSpecs, groupSpecs, isSet, coerce, validateValue, validateAll, isSecretSpec } from '../param-form.js';
import { badge } from './ui.js';

/**
 * renderParamForm(host, {specs, values, onChange, search, showAdvanced}) — `values` holds only user overrides;
 * unset params show the engine default as placeholder. Re-renders only the changed input's error state,
 * so typing never loses focus.
 */
export function renderParamForm(host, { specs, values, onChange, search = '', showAdvanced = false, redacted = new Set() }) {
  host.replaceChildren();
  const visible = filterSpecs(specs, { query: search, showAdvanced });
  if (!visible.length) { host.append(h('p', { class: 'hint' }, t('params.none'))); return; }
  const errors = validateAll(specs, values);
  for (const { group, specs: gs } of groupSpecs(visible)) {
    const setCount = gs.filter((s) => isSet(values, s.key)).length;
    const det = h('details', { class: 'pgroup', open: true },
      h('summary', h('span', t(`group.${group}`)), setCount ? badge(t('params.modified', { n: setCount }), 'info') : null));
    const grid = h('div', { class: 'pgrid' });
    for (const s of gs) grid.append(paramRow(s, values, errors[s.key], (v) => onChange(s.key, v), redacted.has(s.key)));
    det.append(grid);
    host.append(det);
  }
}

function paramRow(spec, values, error, set, needsSecret = false) {
  const id = `p-${spec.key}`, cur = values[spec.key], isOver = isSet(values, spec.key);
  const errEl = h('div', { class: 'perr', id: `${id}-err`, role: 'alert' }, error || '');
  let input, row, resetBtn;
  // Commit in place (no re-render) so tabbing between fields never loses focus.
  const commit = (raw) => {
    const c = coerce(spec, raw);
    const msg = !c.ok ? c.error : validateValue(spec, c.value);
    errEl.textContent = msg || '';
    input.setAttribute('aria-invalid', msg ? 'true' : 'false');
    if (c.ok && !msg) {
      set(c.value);
      row.classList.toggle('modified', c.value !== undefined);
      resetBtn.hidden = c.value === undefined;
    }
  };
  const ph = spec.default != null ? String(spec.default) : t('params.engine_default');
  if (spec.type === 'bool') {
    const state = isOver ? String(!!cur) : '';
    input = h('select', { id, onChange: (e) => commit(e.target.value) },
      h('option', { value: '', selected: state === '' }, spec.default != null ? t('params.default_value', { v: String(spec.default) }) : t('params.engine_default')),
      h('option', { value: 'true', selected: state === 'true' }, t('common.on')),
      h('option', { value: 'false', selected: state === 'false' }, t('common.off')));
  } else if (spec.type === 'enum') {
    input = h('select', { id, onChange: (e) => commit(e.target.value) },
      h('option', { value: '', selected: !isOver }, spec.default != null ? t('params.default_value', { v: String(spec.default) }) : t('params.engine_default')),
      (spec.choices || []).map((c) => h('option', { value: c, selected: isOver && String(cur) === c }, c)));
  } else if (spec.type === 'json') {
    input = h('textarea', { id, rows: 3, placeholder: ph, spellcheck: 'false', value: isOver ? JSON.stringify(cur, null, 1) : '', onChange: (e) => commit(e.target.value) });
  } else if (spec.type === 'string_list') {
    input = h('input', { id, type: 'text', placeholder: ph, value: isOver ? cur.join(', ') : '', onChange: (e) => commit(e.target.value) });
  } else {
    const num = spec.type === 'int' || spec.type === 'float';
    const secret = isSecretSpec(spec);
    input = h('input', { id, type: num ? 'number' : secret ? 'password' : 'text', inputmode: num ? 'decimal' : null, step: spec.type === 'float' ? 'any' : '1',
      min: spec.min ?? null, max: spec.max ?? null, placeholder: needsSecret ? t('params.redacted_ph') : secret ? t('params.secret_ph') : ph, value: isOver ? String(cur) : '', autocomplete: secret ? 'new-password' : 'off', spellcheck: 'false',
      onChange: (e) => commit(e.target.value) });
  }
  input.setAttribute('aria-describedby', `${id}-help ${id}-err`);
  if (error) input.setAttribute('aria-invalid', 'true');
  resetBtn = h('button', { type: 'button', class: 'btn ghost sm', hidden: !isOver, title: t('params.reset'), 'aria-label': t('params.reset_named', { name: spec.label }), onClick: () => { input.value = ''; commit(''); input.focus(); } }, icon('restart', 14));
  const reset = resetBtn;
  row = h('div', { class: `prow${isOver ? ' modified' : ''}${needsSecret ? ' needs-secret' : ''}` },
    h('div', { class: 'prow-head' },
      h('label', { for: id }, spec.label),
      spec.affects_memory ? badge(t('params.memory'), 'info', t('params.memory_tip')) : null,
      spec.requires_restart ? null : badge(t('params.live'), 'ok'),
      spec.docs_url ? h('a', { href: spec.docs_url, target: '_blank', rel: 'noopener noreferrer', class: 'docs', 'aria-label': t('params.docs_named', { name: spec.label }) }, icon('external', 13)) : null),
    h('div', { class: 'prow-input' }, input, reset),
    h('code', { class: 'flag' }, spec.flag.startsWith('env:') ? `env ${spec.flag.slice(4)}` : spec.flag === '@image' ? t('params.image_flag') : spec.flag),
    h('div', { class: 'hint', id: `${id}-help` }, spec.help),
    errEl);
  return row;
}
