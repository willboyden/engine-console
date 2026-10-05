// Pure mapping from an Instance (domain/models.py: management, controls, state) to the buttons the UI shows.
// `controls` says what the hub will do; `state` says what is relevant right now. A control that is relevant but not
// allowed stays visible and disabled with a reason, never silently missing. No DOM and no i18n: reasons are
// {text} (server text, untrusted, shown as text) or {key, vars} descriptors.
export const MANAGEMENT = ['managed', 'script', 'external'];

export const managementOf = (i) => (MANAGEMENT.includes(i?.management) ? i.management : i?.managed === false || i?.source === 'external' ? 'external' : 'managed');

/** InstanceControls with a fallback for a v1 backend that sends none: managed = everything, external = clone only. */
export function normControls(i) {
  const c = i?.controls;
  if (c && typeof c === 'object') return { start: !!c.start, stop: !!c.stop, restart: !!c.restart, remove: !!c.remove, logs: !!c.logs, configure: !!c.configure, reason: typeof c.reason === 'string' && c.reason ? c.reason : null };
  const managed = managementOf(i) === 'managed';
  return { start: managed, stop: managed, restart: managed, remove: managed, logs: managed, configure: true, reason: null };
}

const STOPPED = ['stopped', 'failed'];
const RUNNING = ['ready', 'loading', 'starting'];
const ACTIVE = ['ready', 'loading', 'starting', 'unreachable', 'auth_required'];
const RELEVANT = {
  start: (s) => STOPPED.includes(s),
  stop: (s) => RUNNING.includes(s) || ACTIVE.includes(s),
  restart: (s) => s !== 'stopped' && s !== 'stopping',
};

export function whyNot(i, id) {
  const c = normControls(i);
  if (c.reason) return { text: c.reason };
  const mgmt = managementOf(i);
  return { key: `controls.reason_${mgmt}`, vars: { script: i?.script || '', host: i?.host_label || i?.host_id || '' } };
}

/**
 * -> { management, buttons:[{id, enabled, reason, kind}], logs:{enabled, reason}, note }
 * ids: start, stop, restart, configure, pin, remove. `kind` is a style hint: primary | danger | ''.
 */
export function controlButtons(i) {
  const c = normControls(i);
  const mgmt = managementOf(i);
  const state = i?.state || 'stopped';
  const buttons = [];
  const add = (id, visible, enabled, kind = '') => { if (visible) buttons.push({ id, enabled, reason: enabled ? null : whyNot(i, id), kind }); };
  add('start', RELEVANT.start(state), c.start, 'primary');
  add('stop', RELEVANT.stop(state), c.stop);
  // An external engine the console may stop and start has no restart: it is not offered rather than shown disabled.
  const hideRestart = mgmt === 'external' && c.stop && !c.restart;
  add('restart', RELEVANT.restart(state) && !hideRestart, c.restart);
  add('configure', mgmt !== 'script', c.configure);
  add('pin', mgmt === 'managed', mgmt === 'managed' && c.configure);
  add('remove', mgmt === 'managed', c.remove, 'danger');
  const anyOff = buttons.some((b) => !b.enabled);
  // Server text on controls.reason is informative even when everything works (script engines name their script).
  const note = c.reason && (anyOff || mgmt !== 'managed') ? { text: c.reason } : null;
  return { management: mgmt, buttons, logs: { enabled: c.logs, reason: c.logs ? null : whyNot(i, 'logs') }, note };
}

/** The button the page should make primary for a state, if any: start when stopped, otherwise none. */
export const primaryAction = (i) => controlButtons(i).buttons.find((b) => b.kind === 'primary' && b.enabled)?.id || null;

/** Script actions a script engine offers (ScriptInfo.actions), shaped for rendering; `arg` choices come from the policy. */
export function scriptActions(script) {
  return (script?.actions || []).map((a) => ({ name: a.name, label: a.label || a.name, args: a.args || [], confirm: !!a.confirm, danger: !!a.danger }));
}
export const scriptFor = (instance, scripts) => (scripts || []).find((s) => s.host_id === (instance.host_id || 'local') && s.name === instance.script) || null;

/** Which script statuses are worth a badge. */
export const SCRIPT_STATUS_KIND = { up: 'ok', down: 'muted', unknown: 'muted', error: 'bad' };

// ---- naming and grouping ----
// Engines the console did not start (scripts, external containers) are often named after their container ("big-model-sglang-1"):
// that is a poor title. Show the model the engine serves (or the script's label) and keep the container name as secondary text.
export const scriptKey = (hostId, name) => `${hostId || 'local'}|${name}`;
/** Map "host|script name" -> the script's label, from GET /scripts (or any list of ScriptInfo). */
export const scriptLabelMap = (scripts) => new Map((scripts || []).filter((s) => s?.name).map((s) => [scriptKey(s.host_id, s.name), s.label || s.name]));

/**
 * -> {title, secondary}. managed: its name. script: the served model, else the script's label, else the script name.
 * external: the served model, else its name. The secondary line lists what the title left out (script label, container).
 */
export function instanceTitle(i, labels) {
  const mgmt = managementOf(i);
  const served = (i?.served_models || []).find((m) => typeof m === 'string' && m.trim()) || '';
  const label = i?.script ? labels?.get?.(scriptKey(i.host_id, i.script)) || '' : '';
  const container = i?.container_name || '';
  let title;
  if (mgmt === 'managed') title = i?.name || i?.id || '';
  else if (mgmt === 'script') title = served || label || i?.script || i?.name || i?.id || '';
  else title = served || i?.name || i?.id || '';
  const parts = [];
  if (mgmt === 'script' && label && label !== title) parts.push(label);
  if (mgmt !== 'managed') for (const x of [container, i?.name]) if (x && x !== title && !parts.includes(x)) parts.push(x);
  return { title, secondary: parts.join(' · ') };
}
/** One line for selects and menus: "Title (secondary)". */
export function instanceLabel(i, labels) { const { title, secondary } = instanceTitle(i, labels); return secondary ? `${title} (${secondary})` : title; }

/** Stopped engines are "available to start": listed apart from the running ones, so many stopped scripts do not bury what runs. */
export function splitInstances(list) {
  const active = [], available = [];
  for (const i of list || []) (i?.state === 'stopped' ? available : active).push(i);
  return { active, available };
}

/**
 * Which of a script's actions to offer next to its engine. When the engine's own controls already cover start/stop/restart the
 * rest are offered (status, switch...); when they cover none (an empty `controls`) every action is, so a script is never unusable.
 */
export function scriptActionsFor(instance, script) {
  const all = scriptActions(script);
  const c = normControls(instance);
  const covered = c.start || c.stop || c.restart;
  return covered ? all.filter((a) => a.name !== 'up' && a.name !== 'down') : all;
}
