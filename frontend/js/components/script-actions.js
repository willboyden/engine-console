// A lifecycle script registered in a machine's policy: its status and one button per action. Actions flagged `confirm` ask
// first, `danger` ones are styled as destructive, and an action with allowed arguments gets a picker. Running one opens its
// output, streamed line by line from /scripts/jobs/{id}/stream (events "line" and "end").
import { h, clear, append } from '../dom.js';
import { t } from '../i18n.js';
import { api, stream } from '../app-context.js';
import { scriptActions, SCRIPT_STATUS_KIND } from '../controls.js';
import { items } from '../format.js';
import { badge, btn, select } from './ui.js';
import { openModal, confirmDialog } from './dialog.js';
import { toast, toastError } from './toast.js';

const MAX_LINES = 2000;

// GET /scripts, shared by the views that need script labels and actions (a few seconds old is fine).
let cache = null, cacheAt = 0, inflight = null;
export function loadScripts({ maxAgeMs = 15000, force = false } = {}) {
  if (!force && cache && Date.now() - cacheAt < maxAgeMs) return Promise.resolve(cache);
  if (inflight) return inflight;
  inflight = api.scripts().then((r) => { cache = items(r); cacheAt = Date.now(); return cache; }, () => cache || []).finally(() => { inflight = null; });
  return inflight;
}
export const forgetScripts = () => { cacheAt = 0; };

/** The output window of one job. Closing it stops listening (the job keeps running on the machine). */
export function openJobOutput({ job, title, onEnd }) {
  const out = h('pre', { class: 'logs job-output', role: 'log', tabindex: '0', 'aria-label': t('script.output'), 'aria-live': 'off' });
  const status = h('div', { class: 'row gap' });
  const paintStatus = (j) => append(clear(status), [badge(t(`script.job_${j.state}`), j.state === 'completed' ? 'ok' : j.state === 'failed' ? 'bad' : 'info'), j.rc != null ? h('span', { class: 'muted small num' }, t('script.exit_code', { rc: j.rc })) : null]);
  paintStatus(job);
  const lines = [];
  const finish = (j) => { paintStatus(j); onEnd?.(j); };
  let s = null;
  const m = openModal(title, h('div', { class: 'stack-v' }, status, out), { onClose: () => s?.close() });
  s = stream(`/scripts/jobs/${encodeURIComponent(job.id)}/stream`, {
    json: false, reconnect: false,
    onEvent: (ev) => {
      if (ev.event === 'line') { lines.push(String(ev.data)); if (lines.length > MAX_LINES) lines.shift(); out.textContent = lines.join('\n'); out.scrollTop = out.scrollHeight; }
      else if (ev.event === 'end') { let j = job; try { j = JSON.parse(ev.data); } catch { /* keep the job we have */ } finish(j); s.close(); }
    },
    onError: (e) => { lines.push(`[${e.message || 'stream error'}]`); out.textContent = lines.join('\n'); },
  });
  return m;
}

/** Run one action (after its confirmation, if the policy asks for one) and show the job's output. `arg` is the chosen option. */
export async function runScriptAction(script, a, { arg, hostLabel, onChange } = {}) {
  const chosen = a.args.length ? arg || a.args[0] : undefined;
  const ask = t(chosen ? 'script.confirm_arg' : 'script.confirm', { action: a.label, arg: chosen, script: script.label, host: hostLabel || script.host_id });
  if (a.confirm && !(await confirmDialog(ask, { confirmLabel: a.label, danger: a.danger }))) return;
  try {
    const job = await api.runScript(script.host_id, script.name, a.name, chosen);
    toast(t('script.started', { action: a.label }), { kind: 'ok', timeout: 2000 });
    forgetScripts();
    openJobOutput({ job, title: `${script.label}: ${a.label}`, onEnd: () => onChange?.() });
  } catch (e) { toastError(e, a.label); }
}

/** One button per action (with an option picker for actions that take one). `actions` limits and orders them. */
export function scriptActionButtons(script, { actions, hostLabel, onChange, size = 'sm' } = {}) {
  const list = actions || scriptActions(script);
  const args = new Map();
  return list.map((a) => h('span', { class: 'row gap script-action' },
    a.args.length ? select(a.args.map((v) => ({ value: v, label: v })), a.args[0], (v) => args.set(a.name, v), { 'aria-label': t('script.arg_for', { action: a.label }), class: 'arg-select' }) : null,
    btn(a.label, { kind: a.danger ? 'danger' : '', size, onClick: () => runScriptAction(script, a, { arg: args.get(a.name), hostLabel, onChange }), title: a.confirm ? t('script.asks_first') : undefined })));
}

/** scriptPanel(script, {hostLabel, onChange}) -> element. `script` is a ScriptInfo. */
export function scriptPanel(script, { hostLabel, onChange } = {}) {
  return h('div', { class: 'script-panel stack-v tight' },
    h('div', { class: 'row gap wrap' }, h('strong', script.label), badge(t(`script.status_${script.status}`) === `script.status_${script.status}` ? script.status : t(`script.status_${script.status}`), SCRIPT_STATUS_KIND[script.status] || 'muted'),
      script.engine ? h('span', { class: 'muted small' }, script.engine) : null),
    script.description ? h('p', { class: 'muted small' }, script.description) : null,
    script.detail ? h('p', { class: 'muted small' }, script.detail) : null,
    h('div', { class: 'row gap wrap script-actions' }, scriptActionButtons(script, { hostLabel, onChange })));
}

/** The machine's scripts as compact rows (label, status, description on the left; every action on the right). */
export function scriptList(scripts, { hostLabel, onChange } = {}) {
  return h('ul', { class: 'engine-list script-list' }, scripts.map((sc) => h('li', { class: 'engine-row' },
    h('div', { class: 'engine-main' },
      h('div', { class: 'row gap wrap' }, h('strong', sc.label), badge(t(`script.status_${sc.status}`) === `script.status_${sc.status}` ? sc.status : t(`script.status_${sc.status}`), SCRIPT_STATUS_KIND[sc.status] || 'muted'), sc.engine ? h('span', { class: 'muted small' }, sc.engine) : null),
      sc.description ? h('p', { class: 'muted small engine-note' }, sc.description) : null,
      sc.detail ? h('p', { class: 'muted small engine-note' }, sc.detail) : null),
    h('div', { class: 'engine-actions' }, ...scriptActionButtons(sc, { hostLabel, onChange })))));
}
