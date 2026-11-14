// The Launch page's command preview from the hub's own engine adapters (ADR 0023): the LaunchContext the hub would
// build for the form's choices, rendered by the WebAssembly module (js/engines-wasm.js) into the LaunchSpec the hub
// would hand the node (argv, env, files), plus the adapter's validation (per-parameter and cross-field rules). Pure:
// no DOM, no i18n, the module is passed in, so node --test drives it with the real .wasm.
//
// What is exact and what is not: the engine command, its env and its files are the adapter's own output for these
// inputs. The LaunchContext is rebuilt here from what the page knows, the way services/lifecycle.rs `build` builds it:
// the model argument and served name (runspec.rs `display_of`), the model path in the container for gguf / plain
// models (runspec.rs `model_mounts`), the chosen GPUs from the host's detail, and the container path of the HF cache
// at its default (the hub's HF_CACHE_CONTAINER_PATH setting is not served to the page). The `docker run` around the
// command (mounts, ports, labels, HF_HUB_OFFLINE) is the node's and is not shown. Everything here is advisory: the
// hub validates again on preflight and launch, and GET /instances/{id}/command is authoritative after creation.
import { pruneValues, cliString, toCliArgs, toEnvLines, isSecretSpec } from './param-form.js';

/** The hub's default HF_CACHE_CONTAINER_PATH (config.rs Settings::defaults). */
export const HF_CACHE_CONTAINER_PATH = '/root/.cache/huggingface';
/** Where the node mounts a gguf or plain model (adapters/base.rs MODEL_MOUNT). */
export const MODEL_MOUNT = '/models';
// runspec.rs SECRET_ENV and SHARD_SUFFIX
const SECRET_ENV = /(key|token|secret|password)/i;
const SHARD_SUFFIX = /-0*1-of-0*[0-9]+$/;

const base = (p) => {
  const parts = String(p || '').split('/');
  return parts[parts.length - 1] || '';
};

/** runspec.rs `display_of`: the served model name the hub gives a ModelRef. */
export function displayOf(ref) {
  if (!ref) return 'model';
  if (ref.format === 'hf') return ref.repo_id || 'model';
  if (ref.format === 'ollama') return ref.name || 'ollama';
  let b = base(String(ref.rel || '').replace(/\/+$/, ''));
  if (!b) b = ref.root_id || 'model';
  if (ref.format === 'gguf') {
    if (b.toLowerCase().endsWith('.gguf')) b = [...b].slice(0, -5).join('');
    b = b.replace(SHARD_SUFFIX, '');
  }
  return b || 'model';
}

/** runspec.rs `model_mounts`' model path in the container (null for hf and ollama, whose engines find the model). */
export function modelPathOf(ref) {
  if (!ref) return null;
  const rel = String(ref.rel || '');
  if (ref.format === 'gguf') {
    const i = rel.indexOf('/snapshots/');
    if (i > 0 && rel.length > i + '/snapshots/'.length) return `${MODEL_MOUNT}/snapshots/${rel.slice(i + 11)}`;
    return `${MODEL_MOUNT}/${base(rel)}`;
  }
  if (ref.format === 'plain') return MODEL_MOUNT;
  return null;
}

/** The adapters' Hardware for the chosen GPUs of a host detail (HostDetail: gpus, memory, node.arch). */
export function hardwareOf(host, gpuIds) {
  const ids = [...(gpuIds || [])].sort((a, b) => a - b);
  const gpus = ids.map((i) => (host?.gpus || []).find((g) => g.index === i)).filter(Boolean);
  return {
    gpu_ids: gpus.map((g) => g.index),
    gpu_uuids: gpus.map((g) => g.uuid || `GPU-${g.index}`),
    gpu_names: gpus.map((g) => g.name || ''),
    gpu_total_gib: gpus.map((g) => Number(g.total_gib) || 0),
    gpu_free_gib: gpus.map((g) => Number(g.free_gib) || 0),
    compute_capability: gpus.find((g) => g.compute_capability)?.compute_capability || '',
    host_ram_gib: Number(host?.memory?.total_gib) || 0,
    host_id: host?.id || 'local',
    arch: host?.node?.arch === 'aarch64' ? 'aarch64' : 'x86_64',
    unified_memory: gpus.some((g) => g.unified_memory === true),
  };
}

/** The LaunchContext the hub builds for these choices (lifecycle.rs `build`), or null when one is missing. */
export function launchContext({ ref, specs, values, host, gpuIds }) {
  if (!ref) return null;
  const model =
    ref.format === 'hf' ? ref.repo_id || displayOf(ref) : ref.format === 'ollama' ? ref.name || '' : displayOf(ref);
  return {
    model,
    model_format: ref.format,
    model_path: modelPathOf(ref),
    served_name: displayOf(ref),
    hf_cache_container_path: HF_CACHE_CONTAINER_PATH,
    params: pruneValues(specs || [], values || {}),
    hw: hardwareOf(host, gpuIds),
  };
}

/** The secret parameter values of `values` (never shown): spec keys that look secret, as the hub's secret store. */
function secretValues(specs, values) {
  return (specs || [])
    .filter((s) => isSecretSpec(s) && typeof values?.[s.key] === 'string' && values[s.key])
    .map((s) => values[s.key]);
}
const masked = (v) => `[set, ${String(v).length} chars]`;

/**
 * The text of a LaunchSpec: `NAME=value` env lines (secret-looking names masked), each file as `# <path>` and its
 * content, then the command line. A secret parameter value is never printed, wherever the adapter put it; `secretInArgv`
 * says whether one landed on the command line (the hub refuses that launch: `secret_in_argv`).
 */
export function specText(spec, { secrets = [] } = {}) {
  const hide = (s) =>
    secrets.reduce((acc, v) => (v && acc.includes(v) ? acc.split(v).join(masked(v)) : acc), String(s));
  const env = Object.entries(spec?.env || {}).map(([k, v]) =>
    SECRET_ENV.test(k)
      ? `${k}=${masked(typeof v === 'string' ? v : JSON.stringify(v))}`
      : `${k}=${hide(cliString([String(v)]))}`
  );
  const files = Object.entries(spec?.files || {}).flatMap(([path, content]) => [
    `# ${path}`,
    ...hide(String(content ?? ''))
      .replace(/\n$/, '')
      .split('\n'),
  ]);
  const argv = (spec?.argv || []).map(String);
  const secretInArgv = argv.some((a) => secrets.some((v) => v && a.includes(v)));
  return { text: [...env, ...files, hide(cliString(argv))].join('\n'), secretInArgv };
}

/** The old hand-written approximation (engine module unavailable): env params, then `--model` and the CLI flags. */
export function approxText({ specs, values, ref, label }) {
  return [
    ...toEnvLines(specs || [], values || {}),
    cliString(['--model', label || ref?.repo_id || ref?.name || '<model>', ...toCliArgs(specs || [], values || {})]),
  ].join('\n');
}

/**
 * The preview for the Launch page. `mod` is the engine module (js/engines-wasm.js `engines()`), or null.
 * Returns {source, text, errors, refused, reason}:
 *   source 'hub'    : the adapter's own command (text) and validation messages (errors); `refused` is the hub's refusal
 *                     of this launch (a problem or a raised exception) when it would refuse it, else null.
 *   source 'approx' : the module is not available (reason says why): the old approximation, no errors.
 *   source 'none'   : no model or engine chosen yet.
 */
export function preview(mod, { engine, ref, specs, values, host, gpuIds, label }) {
  if (!engine || !ref) return { source: 'none', text: '', errors: [], refused: null, reason: '' };
  const approx = (reason) => ({
    source: 'approx',
    text: approxText({ specs, values, ref, label }),
    errors: [],
    refused: null,
    reason,
  });
  if (!mod) return approx('');
  const ctx = launchContext({ ref, specs, values, host, gpuIds });
  const v = mod.validate(engine, ctx.params);
  if (v?.unavailable) return approx(v.message || v.unavailable);
  const errors = Array.isArray(v?.ok) ? v.ok.map(String) : [];
  const r = mod.render(engine, ctx);
  if (r?.unavailable) return approx(r.message || r.unavailable);
  if (!r?.ok) {
    const refused = r?.problem
      ? r.problem.detail || r.problem.code || 'refused'
      : r?.raised
        ? `${r.raised.kind}: ${r.raised.msg}`
        : 'no answer';
    return { source: 'hub', text: '', errors, refused, reason: '' };
  }
  const { text, secretInArgv } = specText(r.ok, { secrets: secretValues(specs, ctx.params) });
  return { source: 'hub', text, errors, refused: secretInArgv ? 'secret_in_argv' : null, reason: '' };
}
