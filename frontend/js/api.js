// REST client for /api/v1. RFC 7807 problem+json errors become ApiError; 401 triggers onUnauthorized.
export class ApiError extends Error {
  constructor(status, problem = {}) {
    super(problem.detail || problem.title || `HTTP ${status}`);
    this.name = 'ApiError';
    this.status = status;
    this.code = problem.code || null;
    this.title = problem.title || null;
    this.problem = problem;
  }
}

export const CSRF_HEADER = 'X-Engine-Console';
export const SAFE_METHODS = new Set(['GET', 'HEAD']);

export function buildQuery(query) {
  const p = new URLSearchParams();
  for (const [k, v] of Object.entries(query || {})) {
    if (v == null || v === '') continue;
    p.set(k, String(v));
  }
  const s = p.toString();
  return s ? `?${s}` : '';
}

const enc = encodeURIComponent;
// repo ids contain '/', which the backend accepts literally in `{repo_id:path}` routes.
const repoPath = (id) => id.split('/').map(enc).join('/');

export function createClient({ base = '/api/v1', getKey = () => null, onUnauthorized = () => {}, fetchImpl } = {}) {
  const f = (...a) => (fetchImpl || globalThis.fetch)(...a);
  const authHeaders = () => { const k = getKey(); return k ? { Authorization: `Bearer ${k}` } : {}; };
  const url = (path, query) => `${base}${path}${buildQuery(query)}`;

  async function raw(method, path, { query, body, signal, headers } = {}) {
    const h = { Accept: 'application/json', ...authHeaders(), ...headers };
    // CSRF / DNS-rebinding guard (contract v1.1 hardening): a custom header a cross-site form or <img> cannot set.
    if (!SAFE_METHODS.has(method.toUpperCase())) h[CSRF_HEADER] = '1';
    let payload;
    if (body !== undefined) { h['Content-Type'] = 'application/json'; payload = JSON.stringify(body); }
    let res;
    try { res = await f(url(path, query), { method, headers: h, body: payload, signal }); }
    catch (e) { if (e?.name === 'AbortError') throw e; throw new ApiError(0, { title: 'Network error', detail: e?.message || 'Cannot reach the console backend', code: 'network' }); }
    if (res.status === 401) onUnauthorized();
    if (!res.ok) {
      let problem = {};
      try { problem = await res.json(); } catch { /* not json */ }
      throw new ApiError(res.status, problem);
    }
    return res;
  }
  async function request(method, path, opts) {
    const res = await raw(method, path, opts);
    if (res.status === 204) return null;
    const ct = res.headers.get?.('content-type') || '';
    return ct.includes('json') ? res.json() : res.text();
  }
  const get = (p, q, o) => request('GET', p, { ...o, query: q });
  const post = (p, body, o) => request('POST', p, { ...o, body: body ?? {} });
  const put = (p, body, o) => request('PUT', p, { ...o, body });
  const patch = (p, body, o) => request('PATCH', p, { ...o, body });
  const del = (p, o) => request('DELETE', p, o);

  // Follow `next_cursor` (ARCHITECTURE §8: every list is {items, next_cursor}); bare arrays pass through unchanged.
  async function listAll(path, query, { maxPages = 10 } = {}) {
    const out = [];
    let cursor;
    for (let i = 0; i < maxPages; i++) {
      const page = await get(path, { limit: 100, ...query, cursor });
      if (Array.isArray(page)) return page;
      out.push(...(page.items || []));
      if (!page.next_cursor) break;
      cursor = page.next_cursor;
    }
    return out;
  }

  return {
    url, authHeaders, raw, request, get, post, put, patch, del, listAll,
    health: () => get('/health'),
    hardware: () => get('/hardware'),
    engines: () => get('/engines'),
    params: (engine) => get(`/engines/${enc(engine)}/params`),
    hfSearch: (q, o) => get('/hf/search', { q, ...o }),
    hfModel: (id) => get(`/hf/models/${repoPath(id)}`),
    fit: (body, o) => post('/fit', body, o),
    downloads: () => listAll('/downloads'),
    download: (id) => get(`/downloads/${enc(id)}`),
    startDownload: (body) => post('/downloads', body),
    downloadAction: (id, a) => post(`/downloads/${enc(id)}/${a}`),
    deleteDownload: (id) => del(`/downloads/${enc(id)}`),
    models: () => listAll('/models'),
    deleteModel: (id) => del(`/models/${repoPath(id)}`),
    instances: () => listAll('/instances'),
    preflight: (body) => post('/instances/preflight', body),
    instance: (id) => get(`/instances/${enc(id)}`),
    createInstance: (body) => post('/instances', body),
    instanceAction: (id, a) => post(`/instances/${enc(id)}/${a}`),
    patchInstance: (id, body) => patch(`/instances/${enc(id)}`, body),
    deleteInstance: (id) => del(`/instances/${enc(id)}`),
    logs: (id, tail = 500) => get(`/instances/${enc(id)}/logs`, { tail }),
    command: (id) => get(`/instances/${enc(id)}/command`),
    profiles: () => listAll('/profiles'),
    saveProfile: (body) => post('/profiles', body),
    updateProfile: (id, body) => put(`/profiles/${enc(id)}`, body),
    deleteProfile: (id) => del(`/profiles/${enc(id)}`),
    importProfile: (yaml) => post('/profiles/import', { yaml }),
    exportProfile: async (id) => (await raw('GET', `/profiles/${enc(id)}/export`)).text(),
    metrics: (id, window) => get(`/metrics/instances/${enc(id)}`, { window }),
    runBench: (body) => post('/bench', body),
    benches: () => listAll('/bench'),
    bench: (id) => get(`/bench/${enc(id)}`),
    conversations: () => listAll('/conversations', { limit: 50 }, { maxPages: 2 }),
    conversation: (id) => get(`/conversations/${enc(id)}`),
    createConversation: (body) => post('/conversations', body),
    deleteConversation: (id) => del(`/conversations/${enc(id)}`),
    prompts: () => listAll('/prompts'),
    savePrompt: (body) => (body.id ? put(`/prompts/${enc(body.id)}`, { title: body.title, content: body.content, tags: body.tags || [] }) : post('/prompts', body)),
    deletePrompt: (id) => del(`/prompts/${enc(id)}`),
    createMatch: (body) => post('/arena/matches', body),
    vote: (id, body) => post(`/arena/matches/${enc(id)}/vote`, body),
    leaderboard: () => get('/arena/leaderboard'),
    usage: (group_by) => get('/usage', { group_by }),
    audit: (q) => get('/audit', q),
    settings: () => get('/settings'),
    saveSettings: (body) => put('/settings', body),
    keys: () => get('/keys'),
    createKey: (body) => post('/keys', body),
    deleteKey: (id) => del(`/keys/${enc(id)}`),
  };
}
