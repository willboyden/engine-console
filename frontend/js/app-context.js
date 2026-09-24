// Singletons shared by views: store, API client (bearer key from sessionStorage), stream opener.
import { createStore, session, local } from './store.js';
import { createClient, CSRF_HEADER, SAFE_METHODS } from './api.js';
import { openStream } from './sse.js';
import { toastError } from './components/toast.js';

export const store = createStore({
  apiKey: session.get('ec.apiKey', ''),
  authRequired: false,
  theme: local.get('ec.theme', 'dark'),
  lang: 'en',
  hardware: null,
});

export const api = createClient({
  getKey: () => store.get().apiKey,
  onUnauthorized: () => store.set({ authRequired: true }),
});

export function setApiKey(key) {
  if (key) session.set('ec.apiKey', key); else session.remove('ec.apiKey');
  store.set({ apiKey: key || '', authRequired: false });
}

export function stream(path, opts = {}) {
  const { query, ...rest } = opts;
  return openStream(api.url(path, query), {
    ...rest,
    // Non-GET streams (chat completions) need the same CSRF header as any other mutating request.
    headers: { ...api.authHeaders(), ...(SAFE_METHODS.has((rest.method || 'GET').toUpperCase()) ? {} : { [CSRF_HEADER]: '1' }), ...(rest.headers || {}) },
    onError: (e) => {
      if (e?.status === 401) store.set({ authRequired: true });
      if (e?.status === 429 && e?.code === 'too_many_streams') toastError(e);
      rest.onError?.(e);
    },
  });
}

export function applyTheme(theme) {
  const eff = theme === 'system' ? (matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark') : theme;
  document.documentElement.dataset.theme = eff;
  local.set('ec.theme', theme);
  store.set({ theme });
}
