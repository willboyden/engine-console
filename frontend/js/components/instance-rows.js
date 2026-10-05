// Shared pieces for listing engines. The title is what the engine serves (or its script's label) with the container name as
// secondary text; a script engine gets its script actions inline (every one of them when its own controls are empty); and
// stopped engines go under a collapsible "Available engines" section instead of burying the running ones.
import { h } from '../dom.js';
import { t } from '../i18n.js';
import { fmtGiB } from '../format.js';
import { instanceTitle, scriptFor, scriptActionsFor, scriptLabelMap, managementOf } from '../controls.js';
import { badge, stateBadge, managementBadge } from './ui.js';
import { controlBar } from './instance-controls.js';
import { scriptActionButtons } from './script-actions.js';

const enc = encodeURIComponent;

/** Title (a link to the engine's page) over a quiet secondary line. */
export function titleBlock(i, labels, { link = true, className = '' } = {}) {
  const { title, secondary } = instanceTitle(i, labels);
  return h('div', { class: `stack-v tight inst-title ${className}`.trim() },
    link ? h('a', { class: 'name-link', href: `#/instances/${enc(i.id)}`, title }, title) : h('span', { class: 'name-link', title }, title),
    secondary ? h('span', { class: 'muted small ellipsis', title: secondary }, secondary) : null);
}

/** Script action buttons for a script engine (empty array for the others, or when the script is not known). */
export function scriptExtras(i, scripts, { hostLabel, onChange } = {}) {
  if (managementOf(i) !== 'script') return [];
  const s = scriptFor(i, scripts);
  if (!s) return [];
  const actions = scriptActionsFor(i, s);
  return actions.length ? scriptActionButtons(s, { actions, hostLabel: hostLabel || i.host_label, onChange }) : [];
}

const vramText = (i) => (i.vram || []).map((v) => `GPU ${v.gpu_index} ${v.used_gib != null ? fmtGiB(v.used_gib) : t('fleet.shared_pool')}`).join(' · ');

/**
 * One engine as a row: who it is and what state it is in on the left, its buttons on the right (they wrap, so many controls
 * plus script actions never push anything off screen). opts: {labels, scripts, hostLabel, showHost, only, onChange, onAction(id, instance)}.
 */
export function engineRow(i, { labels, scripts = [], hostLabel, showHost = false, only, onAction, onChange } = {}) {
  const bar = controlBar(i, { size: 'sm', only, onAction: (id) => onAction?.(id, i) });
  const extras = scriptExtras(i, scripts, { hostLabel: i.host_label || hostLabel, onChange });
  const facts = [i.engine, i.repo_id].filter(Boolean).join(' · ');
  const gpu = vramText(i);
  return h('li', { class: 'engine-row' },
    h('div', { class: 'engine-main' }, titleBlock(i, labels),
      h('div', { class: 'row gap wrap small' }, managementBadge(i.management), stateBadge(i.state, { tip: true }),
        showHost ? h('a', { class: 'chip-link', href: `#/hosts/${enc(i.host_id)}` }, i.host_label || i.host_id) : null),
      facts || gpu ? h('div', { class: 'muted small ellipsis', title: facts }, [facts, gpu].filter(Boolean).join(' · ')) : null,
      bar.noteText && managementOf(i) === 'script' ? h('p', { class: 'muted small engine-note' }, bar.noteText) : null),
    h('div', { class: 'engine-actions' }, ...bar.buttons, ...extras));
}

/** A list of engine rows. */
export const engineList = (instances, opts) => h('ul', { class: 'engine-list' }, instances.map((i) => engineRow(i, opts)));

/**
 * The collapsible list of stopped engines. opts: {scripts, labels, hostLabel, showHost, onAction, onChange, open, title}.
 * Returns null when there is nothing to show.
 */
export function availableSection(instances, { scripts = [], labels, hostLabel, showHost = false, onAction, onChange, open = false, title } = {}) {
  if (!instances.length) return null;
  const lab = labels || scriptLabelMap(scripts);
  return h('details', { class: 'available', open: open || null },
    h('summary', {}, h('span', { class: 'avail-title' }, title || t('inst.available_title', { host: hostLabel || '' })), badge(String(instances.length), 'muted')),
    h('p', { class: 'hint avail-hint' }, t('inst.available_hint')),
    engineList(instances, { labels: lab, scripts, hostLabel, showHost, only: ['start'], onAction, onChange }));
}
