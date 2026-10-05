// Pure helpers about engines (GET /engines) and which of them a machine may run. No DOM, no i18n.
export const ENGINE_ORDER = ['vllm', 'sglang', 'llamacpp', 'ollama', 'trtllm'];

const rank = (id) => { const i = ENGINE_ORDER.indexOf(id); return i < 0 ? ENGINE_ORDER.length : i; };
export const sortEngines = (engines) => [...(engines || [])].sort((a, b) => rank(a.id) - rank(b.id) || String(a.id).localeCompare(String(b.id)));
export const formatsOf = (e) => (Array.isArray(e?.model_formats) && e.model_formats.length ? e.model_formats : ['hf']);
/** Engines that can load a model of this format (an engine without model_formats is a v1 engine: hf only). */
export const enginesForFormat = (engines, format) => sortEngines(engines).filter((e) => formatsOf(e).includes(format));

/**
 * Node policy patterns: an exact reference, or a glob where `*` matches any run of characters ("repo:*" is the usual form).
 * The hub makes the real decision; this only decides what the UI offers, so it errs on the side of offering.
 */
export function imageAllowed(patterns, ref) {
  if (!ref) return false;
  return (patterns || []).some((p) => {
    if (p === ref) return true;
    if (!p.includes('*')) return false;
    const re = new RegExp(`^${p.split('*').map((x) => x.replace(/[.+?^${}()|[\]\\]/g, '\\$&')).join('.*')}$`);
    return re.test(ref);
  });
}

/** The pinned default image of an engine on a host of this architecture. */
export const defaultImage = (engine, arch) => (arch && engine?.images_by_arch?.[arch]) || engine?.default_image || null;

/**
 * Whether a host can run an engine: the machine's policy must allow the engine's image. When the host lists the images it
 * has, the allowed ones that are present come first so the picker can offer exactly those.
 */
export function engineUsable(engine, host) {
  const patterns = host?.policy?.engine_images || [];
  if (!patterns.length) return { ok: false, reason: { key: 'engines.no_images_allowed', vars: { host: host?.label || host?.id || '' } } };
  const img = defaultImage(engine, host?.node?.arch);
  if (imageAllowed(patterns, img)) return { ok: true, reason: null };
  const present = (host?.images || []).some((i) => imageAllowed(patterns, i.ref) && engineMatchesImage(engine, i.ref));
  return present ? { ok: true, reason: null } : { ok: false, reason: { key: 'engines.image_not_allowed', vars: { engine: engine?.display_name || engine?.id || '', host: host?.label || host?.id || '' } } };
}

// An image belongs to an engine when it shares the engine's default image repository (the part before the tag).
const repoOf = (ref) => String(ref || '').replace(/[:@].*$/, '');
export function engineMatchesImage(engine, ref) {
  const repos = new Set([engine?.default_image, ...Object.values(engine?.images_by_arch || {})].map(repoOf).filter(Boolean));
  return repos.has(repoOf(ref));
}

/** Images the host has that this engine may use (allowed by policy and present), newest first. */
export function imageChoices(engine, host) {
  const patterns = host?.policy?.engine_images || [];
  return (host?.images || []).filter((i) => engineMatchesImage(engine, i.ref) && imageAllowed(patterns, i.ref))
    .sort((a, b) => (b.created || 0) - (a.created || 0));
}

/** The first engine (in the usual order) that can load the format and that the host may run; null when none. */
export function pickEngine(engines, format, host) {
  return enginesForFormat(engines, format).find((e) => engineUsable(e, host).ok) || null;
}
