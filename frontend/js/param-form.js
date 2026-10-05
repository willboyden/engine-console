// Pure logic behind the schema-driven parameter form (ParamSpec from adapters/base.py). No DOM here.
export const GROUP_ORDER = ['memory', 'parallelism', 'context', 'quantization', 'scheduling', 'performance',
  'tools_reasoning', 'speculative', 'network', 'advanced'];

export const isSet = (values, key) => Object.prototype.hasOwnProperty.call(values, key) && values[key] !== undefined;

export function defaultsOf(specs) {
  return Object.fromEntries(specs.filter((s) => s.default != null).map((s) => [s.key, s.default]));
}

export function filterSpecs(specs, { query = '', showAdvanced = false } = {}) {
  const q = query.trim().toLowerCase();
  return specs.filter((s) => {
    if (q) return [s.key, s.flag, s.label, s.help, s.group].some((x) => String(x || '').toLowerCase().includes(q));
    return showAdvanced || !s.advanced;
  });
}

export function groupSpecs(specs) {
  const by = new Map();
  for (const s of specs) { if (!by.has(s.group)) by.set(s.group, []); by.get(s.group).push(s); }
  const order = [...GROUP_ORDER, ...[...by.keys()].filter((g) => !GROUP_ORDER.includes(g))];
  return order.filter((g) => by.has(g)).map((group) => ({ group, specs: by.get(group) }));
}

// Turn raw input text into a typed value. Empty input means "unset" (value undefined, ok true).
export function coerce(spec, raw) {
  if (raw === '' || raw == null) return { ok: true, value: undefined };
  switch (spec.type) {
    case 'int': { if (!/^-?\d+$/.test(String(raw).trim())) return { ok: false, error: 'Enter a whole number' }; return { ok: true, value: parseInt(raw, 10) }; }
    case 'float': { const n = Number(raw); return Number.isFinite(n) ? { ok: true, value: n } : { ok: false, error: 'Enter a number' }; }
    case 'bool': return { ok: true, value: raw === true || raw === 'true' };
    case 'string_list': return { ok: true, value: String(raw).split(/[\n,]/).map((x) => x.trim()).filter(Boolean) };
    case 'json': { try { return { ok: true, value: JSON.parse(raw) }; } catch (e) { return { ok: false, error: 'Invalid JSON' }; } }
    default: return { ok: true, value: String(raw) };
  }
}

export function validateValue(spec, v) {
  if (v === undefined || v === null) return null;
  if ((spec.type === 'int' || spec.type === 'float') && typeof v === 'number') {
    if (spec.min != null && v < spec.min) return `Minimum is ${spec.min}`;
    if (spec.max != null && v > spec.max) return `Maximum is ${spec.max}`;
  }
  if (spec.type === 'enum' && spec.choices && !spec.choices.includes(String(v))) return 'Not one of the allowed choices';
  return null;
}

export function validateAll(specs, values) {
  const errs = {};
  for (const s of specs) { if (isSet(values, s.key)) { const e = validateValue(s, values[s.key]); if (e) errs[s.key] = e; } }
  return errs;
}

export function memoryFingerprint(specs, values) {
  return JSON.stringify(specs.filter((s) => s.affects_memory).map((s) => [s.key, values[s.key] ?? null]));
}

// Accepts {name: params} (adapter.presets()) or [{name|id, label?, params}] and returns the array form.
export function normalizePresets(p) {
  if (!p) return [];
  if (Array.isArray(p)) return p.map((x) => ({ name: x.name || x.id, label: x.label || x.name || x.id, params: x.params || {} }));
  return Object.entries(p).map(([name, params]) => ({ name, label: name, params }));
}

export const changedCount = (values) => Object.values(values).filter((v) => v !== undefined).length;

// The backend returns secret params as "[set]" (never the value). That placeholder must never be sent back:
// it would overwrite the real secret with the literal string. pruneValues drops it; splitRedacted reports which keys it hid.
export const REDACTED = '[set]';
export const isRedacted = (v) => v === REDACTED;
export function splitRedacted(values) {
  const clean = {}, redacted = [];
  for (const [k, v] of Object.entries(values || {})) { if (isRedacted(v)) redacted.push(k); else clean[k] = v; }
  return { values: clean, redacted };
}

export function pruneValues(specs, values) {
  const known = new Set(specs.map((s) => s.key));
  return Object.fromEntries(Object.entries(values).filter(([k, v]) => known.has(k) && v !== undefined && !isRedacted(v)));
}

export function shellQuote(a) { return /^[\w@%+=:,./-]+$/.test(a) ? a : `'${String(a).replace(/'/g, `'\\''`)}'`; }

// Contract v1.1: flags `env:NAME` (container env var) and `@image` (image override) are not CLI flags.
export const isEnvSpec = (s) => /^env:/.test(s.flag);
export const isImageSpec = (s) => s.flag === '@image';
export const isNonCli = (s) => isEnvSpec(s) || isImageSpec(s);
export const envName = (s) => s.flag.slice(4);
// Security principle: secret-looking values are never rendered or copied; the input is type=password and previews mask them.
export const isSecretSpec = (s) => /(token|secret|password|api[_-]?key)/i.test(`${s.key} ${s.flag}`);

/** `NAME=value` lines for env params; secrets are masked as [set, N chars]. */
export function toEnvLines(specs, values) {
  return specs.filter((s) => isEnvSpec(s) && isSet(values, s.key)).map((s) => `${envName(s)}=${isSecretSpec(s) ? `[set, ${String(values[s.key]).length} chars]` : shellQuote(String(values[s.key]))}`);
}

// Approximate engine argv for "copy as command" preview (the backend's /instances/{id}/command is authoritative).
export function toCliArgs(specs, values) {
  const out = [];
  for (const s of specs) {
    if (!isSet(values, s.key) || isNonCli(s)) continue;
    const v = values[s.key];
    if (s.type === 'bool') {
      if (s.bool_style === 'value') out.push(s.flag, v ? 'true' : 'false'); else if (v) out.push(s.flag);
    } else if (s.type === 'string_list') { if (v.length) out.push(s.flag, ...v); }
    else if (s.type === 'json') out.push(s.flag, JSON.stringify(v));
    else out.push(s.flag, String(v));
  }
  return out;
}
export const cliString = (args) => args.map(shellQuote).join(' ');

// ---- learn levels (contract v2: ParamSpec.level = basic | advanced | expert) ----
export const LEVEL_ORDER = ['basic', 'advanced', 'expert'];
export const levelRank = (l) => Math.max(0, LEVEL_ORDER.indexOf(l));
/** The effective level of a spec: its own `level`, else "advanced" when the v1 `advanced` flag is set, else "basic". */
export const levelOf = (s) => (LEVEL_ORDER.includes(s?.level) ? s.level : s?.advanced ? 'advanced' : 'basic');
export const nextLevel = (l) => LEVEL_ORDER[Math.min(LEVEL_ORDER.length - 1, levelRank(l) + 1)];

/**
 * Specs shown up to `level`. A search ignores the level (find anything), and a parameter the user has already set is always
 * shown so a preset's effect never hides behind "show more".
 */
export function filterSpecsByLevel(specs, { level = 'basic', query = '', values = {} } = {}) {
  const q = query.trim().toLowerCase();
  return specs.filter((s) => {
    if (q) return [s.key, s.flag, s.label, s.help, s.group].some((x) => String(x || '').toLowerCase().includes(q));
    return levelRank(levelOf(s)) <= levelRank(level) || isSet(values, s.key);
  });
}
export function levelCounts(specs) {
  const c = { basic: 0, advanced: 0, expert: 0 };
  for (const s of specs) c[levelOf(s)]++;
  return c;
}

/** "4096 tokens", "0.9", "on": how a value reads in a summary. */
export function formatValue(spec, v) {
  if (v === true) return 'on';
  if (v === false) return 'off';
  if (v !== null && typeof v === 'object') return JSON.stringify(v);
  return spec?.unit ? `${v} ${spec.unit}` : String(v);
}

/** Every setting whose value differs from the engine default, in catalog order (the wizard's review list). */
export function nonDefaultSettings(specs, values) {
  const out = [];
  for (const s of specs) {
    if (!isSet(values, s.key) || isRedacted(values[s.key])) continue;
    const v = values[s.key];
    if (s.default != null && JSON.stringify(s.default) === JSON.stringify(v)) continue;
    out.push({ key: s.key, label: s.label, value: v, text: formatValue(s, v), spec: s });
  }
  return out;
}
