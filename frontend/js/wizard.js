// Guided launch wizard: a small pure state machine (model -> machine -> goal -> review, then the launch action) plus the
// helpers its steps use. No DOM and no i18n; the view renders state and dispatches actions.
import { normalizeReport } from './fit-format.js';

export const STEPS = ['model', 'machine', 'goal', 'review'];
// Preset names every adapter offers (ARCHITECTURE section 3, `profiles`), in the order the goal step lists them.
export const GOALS = ['low-latency', 'balanced', 'max-throughput', 'long-context', 'tool-agent'];
export const stepIndex = (s) => STEPS.indexOf(s.step);

export function initialState(prefill = {}) {
  const s = { step: 'model', model: null, ref: null, hostId: null, engine: null, goal: null, params: {}, baseParams: {}, gpuIds: [], image: '', name: '', ...prefill };
  s.step = s.model && s.hostId ? 'goal' : s.model ? 'machine' : 'model';
  return s;
}

/** Furthest step index the current choices allow (what the stepper lets the user jump back or forward to). */
export function furthestStep(s) {
  if (!s.model) return 0;
  if (!s.hostId) return 1;
  if (!s.goal) return 2;
  return 3;
}

export function canAdvance(s) {
  switch (s.step) {
    case 'model': return s.model ? { ok: true } : { ok: false, reason: 'wizard.need_model' };
    case 'machine': return s.hostId ? { ok: true } : { ok: false, reason: 'wizard.need_machine' };
    case 'goal': return s.goal ? { ok: true } : { ok: false, reason: 'wizard.need_goal' };
    default: return { ok: true };
  }
}

export function reduce(s, a) {
  switch (a.type) {
    case 'model': {
      // A different model invalidates everything chosen after it; choosing the same one again keeps the rest.
      const changed = !s.model || s.model.key !== a.model.key;
      const base = changed ? { ...s, ref: null, hostId: null, engine: null, goal: null, params: {}, baseParams: {}, gpuIds: [], image: '' } : s;
      return { ...base, model: a.model, step: 'machine' };
    }
    case 'host': {
      // The machine decides which copy (and so which ModelRef) is loaded, which GPUs are used and any settings the fit needs.
      const changed = s.hostId !== a.hostId;
      const base = changed ? { ...s, engine: null, goal: null, params: {}, image: '' } : s;
      return { ...base, hostId: a.hostId, ref: a.ref ?? null, gpuIds: [...(a.gpuIds || [])], baseParams: { ...(a.baseParams || {}) }, engine: a.engine ?? base.engine, step: 'goal' };
    }
    case 'engine': return s.engine === a.engine ? s : { ...s, engine: a.engine, goal: null, params: {}, image: '' };
    case 'goal': return { ...s, goal: a.goal, params: { ...s.baseParams, ...(a.params || {}) }, step: 'review' };
    case 'params': return { ...s, params: { ...a.params } };
    case 'gpus': return { ...s, gpuIds: [...a.gpuIds] };
    case 'image': return { ...s, image: a.image || '' };
    case 'name': return { ...s, name: a.name };
    case 'next': { if (!canAdvance(s).ok) return s; const i = stepIndex(s); return i < STEPS.length - 1 ? { ...s, step: STEPS[i + 1] } : s; }
    case 'back': { const i = stepIndex(s); return i > 0 ? { ...s, step: STEPS[i - 1] } : s; }
    case 'goto': { const i = STEPS.indexOf(a.step); return i >= 0 && i <= furthestStep(s) ? { ...s, step: a.step } : s; }
    case 'reset': return initialState();
    default: return s;
  }
}

// ---- helpers the steps use ----
/** The keys of the engine's parameter that shards a model over several GPUs, if the catalog has one. */
export function tpParamKey(specs) {
  return (specs || []).find((s) => /^(tensor_parallel_size|tp_size|tensor_parallel)$/.test(s.key))?.key || null;
}

/** The `count` GPUs with the most free memory, as sorted indices (fewer when the host has fewer). */
export function pickGpus(gpus, count = 1) {
  const free = (g) => (typeof g.free_gib === 'number' ? g.free_gib : typeof g.total_gib === 'number' ? g.total_gib : 0);
  return [...(gpus || [])].sort((a, b) => free(b) - free(a) || a.index - b.index).slice(0, Math.max(1, count)).map((g) => g.index).sort((a, b) => a - b);
}

const FIT_WORDS = { fits: 'wizard.fit_easy', tight: 'wizard.fit_tight', wont_fit: 'wizard.fit_wont', unknown: 'wizard.fit_unknown' };
/** i18n key of the plain-language verdict for a machine card ("fits easily / tight / won't fit"). */
export const fitWord = (verdict) => FIT_WORDS[verdict] || FIT_WORDS.unknown;

/** {key, vars} for the one-line reason under a machine's verdict, from a FitReport. */
export function fitReason(rawReport) {
  const r = normalizeReport(rawReport);
  if (!r) return { key: 'wizard.reason_unknown' };
  const block = r.compat.find((c) => c.level === 'block');
  if (block) return { key: 'wizard.reason_blocked', vars: { message: block.message } };
  const worst = r.gpus.reduce((w, g) => (g.budget > 0 && g.total / g.budget > (w ? w.total / w.budget : 0) ? g : w), null);
  const f1 = (n) => (Math.round(n * 10) / 10).toFixed(1);
  if (!worst) return { key: 'wizard.reason_unknown' };
  const vars = { need: f1(worst.total), have: f1(worst.budget), pct: Math.round((worst.total / worst.budget) * 100) };
  if (r.verdict === 'wont_fit') return r.tpRequired > 1 ? { key: 'wizard.reason_wont_tp', vars: { ...vars, n: r.tpRequired } } : { key: 'wizard.reason_wont', vars };
  if (r.verdict === 'tight') return { key: 'wizard.reason_tight', vars };
  if (r.verdict === 'fits') return { key: 'wizard.reason_fits', vars };
  return { key: 'wizard.reason_unknown' };
}

/** How many GPUs to use for a report: what the estimator says it needs, capped by what the machine has. */
export const gpusNeeded = (rawReport, gpuCount) => Math.max(1, Math.min(gpuCount || 1, normalizeReport(rawReport)?.tpRequired || 1));
