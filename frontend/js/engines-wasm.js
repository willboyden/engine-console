// The hub's own engine adapters in the browser (ADR 0023): the WebAssembly module `make wasm` builds from
// backend/crates/ec-engines-wasm into frontend/pkg/. The Launch page's command preview and validation, the image-policy
// badge and the Config files page's checks call it instead of hand-written copies of the hub's rules.
//
// Everything it answers is advisory: the hub validates again on preflight and launch. When the module cannot load (not
// built, an old browser, a CSP mistake) or an adapter traps (a Rust panic aborts the module: panic = abort on wasm),
// `engines()` is null and the pages say "preview unavailable"; Launch still works. After a trap the dead instance is
// dropped and a fresh one is loaded (a new import of the glue, so nothing of the trapped memory is reused).
//
// The module's exports take and return JSON text; `wrap` parses it. Answers are envelopes: {ok: value}, {problem:
// {status, code, detail, ...}} (what the hub would answer), {raised: {kind, msg}} (an exception the adapter raised), or,
// from this wrapper, {unavailable: 'trap' | 'error' | 'dead'}.

/** The JSON API version this page speaks (ec_engines_wasm::API_VERSION); another version is a stale frontend/pkg/. */
export const API_VERSION = 1;

/**
 * Wrap an initialised wasm-bindgen module. A call that throws (a trap is a WebAssembly.RuntimeError) marks the instance
 * dead: every later call answers {unavailable: 'dead'} and `onTrap(error)` runs once.
 */
export function wrap(mod, { onTrap } = {}) {
  let dead = false;
  const call = (name, ...args) => {
    if (dead) return { unavailable: 'dead' };
    try {
      const r = mod[name](...args);
      return typeof r === 'string' ? JSON.parse(r) : r;
    } catch (e) {
      dead = true;
      const trap = typeof WebAssembly !== 'undefined' && e instanceof WebAssembly.RuntimeError;
      try {
        onTrap?.(e);
      } catch {
        /* a listener's failure must not hide the trap */
      }
      return { unavailable: trap ? 'trap' : 'error', message: String(e?.message || e) };
    }
  };
  return {
    get dead() {
      return dead;
    },
    version: (() => {
      try {
        return mod.build_version();
      } catch {
        return '';
      }
    })(),
    engines: () => call('engines'),
    catalog: (engine) => call('catalog', engine),
    presets: (engine) => call('presets', engine),
    validate: (engine, params) => call('validate', engine, JSON.stringify(params ?? {})),
    render: (engine, ctx) => call('render', engine, JSON.stringify(ctx)),
    compatibility: (engine, input) => call('compatibility', engine, JSON.stringify(input)),
    configCheck: (file, text) => call('config_check', file, String(text ?? '')),
    /** The env file as the hub reads it: [[name, value], ...] (the line editor's tests compare with it). */
    dotenv: (text) => call('dotenv', String(text ?? '')),
    /** true / false from the node's rule; null when the module cannot answer. */
    imageAllowed: (patterns, ref) => {
      const r = call('image_allowed', JSON.stringify(Array.isArray(patterns) ? patterns : []), String(ref ?? ''));
      return typeof r === 'boolean' ? r : null;
    },
  };
}

/**
 * A loader: `glueUrl()` is the URL of the wasm-bindgen glue for the n-th instance (a new URL per instance gives a fresh
 * module record), `wasmSource()` what its init takes (a URL to fetch, or the bytes). Loads at most one at a time.
 */
export function createLoader({ glueUrl, wasmSource }) {
  let current = null;
  let loading = null;
  let generation = 0;
  let status = { state: 'idle', reason: '' };
  const listeners = new Set();
  const notify = () => {
    for (const f of listeners) {
      try {
        f(status);
      } catch {
        /* one listener's failure does not stop the others */
      }
    }
  };
  const load = () => {
    if (current && !current.dead) return Promise.resolve(current);
    if (loading) return loading;
    const n = ++generation;
    status = { state: 'loading', reason: status.reason };
    loading = (async () => {
      try {
        const mod = await import(glueUrl(n));
        await mod.default({ module_or_path: await wasmSource() });
        const v = mod.api_version();
        if (v !== API_VERSION)
          throw new Error(`the engine module speaks API ${v}, this page ${API_VERSION}: run make wasm`);
        current = wrap(mod, {
          onTrap: (e) => {
            status = { state: 'trapped', reason: String(e?.message || e) };
            notify();
            // a fresh instance for the next answer; the trapped one is never called again
            Promise.resolve().then(load);
          },
        });
        status = { state: 'ready', reason: '' };
      } catch (e) {
        current = null;
        status = { state: 'unavailable', reason: String(e?.message || e) };
      } finally {
        loading = null;
      }
      notify();
      return current;
    })();
    return loading;
  };
  return {
    load,
    /** The live instance, or null (not loaded yet, unavailable, or trapped and reloading). */
    get: () => (current && !current.dead ? current : null),
    status: () => status,
    /** Call `f(status)` on every change; returns the unsubscribe function. */
    onChange: (f) => {
      listeners.add(f);
      return () => listeners.delete(f);
    },
    /** Tests: use this instance (or null) instead of loading. */
    set: (inst) => {
      current = inst;
      status = inst ? { state: 'ready', reason: '' } : { state: 'unavailable', reason: 'not loaded' };
      notify();
    },
  };
}

// The page's loader: frontend/pkg/ next to js/, served by the hub under /pkg/ (application/wasm).
const PKG = new URL('../pkg/', import.meta.url);
export const loader = createLoader({
  glueUrl: (n) => {
    const u = new URL('ec_engines_wasm.js', PKG);
    if (n > 1) u.searchParams.set('instance', String(n));
    return u.href;
  },
  wasmSource: () => new URL('ec_engines_wasm_bg.wasm', PKG),
});

/** The loaded module, or null. */
export const engines = () => loader.get();
/** Load the module once (later calls share it); resolves to the instance or null. Never rejects. */
export const loadEngines = () => loader.load();
/** {state: idle | loading | ready | unavailable | trapped, reason}. */
export const enginesStatus = () => loader.status();
export const onEnginesChange = (f) => loader.onChange(f);

/** The node policy's image rule (ec_proto::image): true / false, or null when the module is not available. */
export const imageAllowed = (patterns, ref) => engines()?.imageAllowed(patterns, ref) ?? null;

/** The Problem or exception text of an envelope that is not {ok}, for the page. */
export function envelopeError(r) {
  if (!r) return 'no answer';
  if (r.problem) return r.problem.detail || r.problem.code || 'refused';
  if (r.raised) return `${r.raised.kind}: ${r.raised.msg}`;
  if (r.unavailable) return r.message || r.unavailable;
  return '';
}
