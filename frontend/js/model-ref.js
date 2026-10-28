// ModelRef helpers (backend domain/refs.rs: format, repo_id, root_id, rel, name). Pure: no DOM, no i18n.
// A ModelRef says what to load on the chosen host; LibraryModel + LibraryCopy say where it lives.
export const MODEL_FORMATS = ['hf', 'gguf', 'ollama', 'plain'];

const str = (v) => (typeof v === 'string' && v ? v : null);

/** ModelRef for one library model, using one of its copies for the formats that need a root + path. */
export function refFor(model, copy) {
  if (!model) return null;
  switch (model.format) {
    case 'hf': return { format: 'hf', repo_id: model.name };
    case 'ollama': return { format: 'ollama', name: model.name };
    case 'gguf':
    case 'plain': return copy ? { format: model.format, root_id: copy.root_id, rel: copy.rel } : null;
    default: return null;
  }
}

// Query-string encoding (hash routes carry a model between pages: Library -> Launch / Wizard).
export function refToQuery(ref) {
  if (!ref) return {};
  const q = { mfmt: ref.format };
  if (ref.repo_id) q.repo = ref.repo_id;
  if (ref.root_id) q.root = ref.root_id;
  if (ref.rel) q.rel = ref.rel;
  if (ref.name) q.oname = ref.name;
  return q;
}

/** Inverse of refToQuery; a bare `repo` (the v1 deep link) means an hf model. Returns null when nothing usable is there. */
export function refFromQuery(q = {}) {
  const format = MODEL_FORMATS.includes(q.mfmt) ? q.mfmt : (str(q.repo) && !q.root && !q.oname ? 'hf' : null);
  if (!format) return null;
  if (format === 'hf') return str(q.repo) ? { format, repo_id: q.repo } : null;
  if (format === 'ollama') return str(q.oname) ? { format, name: q.oname } : null;
  return str(q.root) && str(q.rel) ? { format, root_id: q.root, rel: q.rel } : null;
}

const base = (p) => String(p || '').split('/').filter(Boolean).pop() || '';
export function refLabel(ref) {
  if (!ref) return '';
  if (ref.format === 'hf') return ref.repo_id || '';
  if (ref.format === 'ollama') return ref.name || '';
  return base(ref.rel) || ref.root_id || '';
}

export function refEquals(a, b) {
  if (!a || !b) return !a && !b;
  return a.format === b.format && (a.repo_id || null) === (b.repo_id || null) && (a.root_id || null) === (b.root_id || null)
    && (a.rel || null) === (b.rel || null) && (a.name || null) === (b.name || null);
}

/** Request body fields for POST /fit and POST /instances: the hub accepts repo_id (hf shortcut) or a full model ref. */
export function refBody(ref) {
  if (!ref) return {};
  return ref.format === 'hf' ? { repo_id: ref.repo_id, model: ref } : { model: ref };
}
