// Bootstrap: i18n, theme, shell (nav/topbar), router -> lazy-loaded view custom elements, auth dialog.
import { loadMessages, t } from './i18n.js';
import { store, api, setApiKey, applyTheme } from './app-context.js';
import { createRouter } from './router.js';
import { h, icon, clear, $ } from './dom.js';
import { toast } from './components/toast.js';
import { registerCommands } from './components/palette.js';
import { openModal } from './components/dialog.js';
import './components/toast.js';
import './components/palette.js';

export const ROUTES = [
  { path: '/', tag: 'ec-dashboard', mod: 'dashboard', nav: 'dashboard', icon: 'dashboard' },
  { path: '/models', tag: 'ec-models', mod: 'models', nav: 'models', icon: 'models' },
  { path: '/downloads', tag: 'ec-downloads', mod: 'downloads', nav: 'downloads', icon: 'downloads' },
  { path: '/launch', tag: 'ec-launch', mod: 'launch', nav: 'launch', icon: 'launch' },
  { path: '/instances', tag: 'ec-instances', mod: 'instances', nav: 'instances', icon: 'instances' },
  { path: '/instances/:id', tag: 'ec-instances', mod: 'instances', nav: 'instances', icon: 'instances', hidden: true },
  { path: '/metrics', tag: 'ec-metrics', mod: 'metrics', nav: 'metrics', icon: 'metrics' },
  { path: '/bench', tag: 'ec-bench', mod: 'bench', nav: 'bench', icon: 'bench' },
  { path: '/chat', tag: 'ec-chat', mod: 'chat', nav: 'chat', icon: 'chat' },
  { path: '/arena', tag: 'ec-arena', mod: 'arena', nav: 'arena', icon: 'arena' },
  { path: '/usage', tag: 'ec-usage', mod: 'usage', nav: 'usage', icon: 'usage' },
  { path: '/settings', tag: 'ec-settings', mod: 'settings', nav: 'settings', icon: 'settings' },
];

function buildShell(root) {
  const nav = h('nav', { class: 'nav', 'aria-label': t('nav.primary') },
    ROUTES.filter((r) => !r.hidden).map((r) => h('a', { href: `#${r.path}`, 'data-route': r.path }, icon(r.icon), h('span', t(`nav.${r.nav}`)))));
  const themeBtn = h('button', { type: 'button', class: 'btn ghost', id: 'theme-btn', 'aria-label': t('topbar.theme'), onClick: toggleTheme });
  const keyBtn = h('button', { type: 'button', class: 'btn ghost', 'aria-label': t('topbar.api_key'), title: t('topbar.api_key'), onClick: () => promptKey() }, icon('key'));
  const palBtn = h('button', { type: 'button', class: 'btn ghost palette-btn', onClick: () => window.dispatchEvent(new Event('ec:palette')) },
    icon('search', 16), h('span', t('palette.open')), h('kbd', 'Ctrl K'));
  root.append(
    h('a', { class: 'skip', href: '#main' }, t('common.skip')),
    h('aside', { class: 'sidebar' }, h('div', { class: 'brand' }, h('span', { class: 'logo', 'aria-hidden': 'true' }, '◆'), h('span', t('app.name'))), nav),
    h('div', { class: 'content' },
      h('header', { class: 'topbar' }, h('div', { id: 'crumb', class: 'crumb', role: 'heading', 'aria-level': '1' }), h('div', { class: 'row gap' }, palBtn, themeBtn, keyBtn)),
      h('main', { id: 'main', tabindex: '-1' })),
    document.createElement('ec-toasts'), document.createElement('ec-palette'));
  syncThemeBtn();
}
function syncThemeBtn() {
  const b = $('#theme-btn'); if (!b) return;
  const light = document.documentElement.dataset.theme === 'light';
  clear(b).append(icon(light ? 'moon' : 'sun'));
  b.title = light ? t('topbar.to_dark') : t('topbar.to_light');
}
function toggleTheme() { applyTheme(document.documentElement.dataset.theme === 'light' ? 'dark' : 'light'); syncThemeBtn(); }

let authDialog = null;
function promptKey() {
  if (authDialog) return;
  const input = h('input', { type: 'password', autocomplete: 'off', spellcheck: 'false', 'aria-label': t('auth.key_label'), placeholder: 'ec_…', value: store.get().apiKey || '' });
  const form = h('form', { onSubmit: (e) => { e.preventDefault(); setApiKey(input.value.trim()); authDialog?.close(); toast(t('auth.saved'), { kind: 'ok' }); router.refresh(); } },
    h('p', t('auth.explain')), h('div', { class: 'field' }, h('label', { for: 'key-in' }, t('auth.key_label')), input),
    h('div', { class: 'row gap end' }, h('button', { type: 'button', class: 'btn', onClick: () => { setApiKey(''); authDialog?.close(); } }, t('auth.clear')), h('button', { type: 'submit', class: 'btn primary' }, t('common.save'))));
  input.id = 'key-in';
  authDialog = openModal(t('auth.title'), form, { onClose: () => { authDialog = null; } });
  input.focus();
}

let current = null, seq = 0;
async function navigate({ path, route, params, query }) {
  const main = $('#main');
  const my = ++seq;
  if (!route) { clear(main).append(h('div', { class: 'state empty' }, h('strong', t('common.not_found')), h('a', { href: '#/' }, t('nav.dashboard')))); return; }
  document.querySelectorAll('.nav a').forEach((a) => {
    const on = a.dataset.route === path || (a.dataset.route !== '/' && path.startsWith(a.dataset.route + '/'));
    if (on) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current');
  });
  $('#crumb').textContent = t(`nav.${route.nav}`);
  document.title = `${t(`nav.${route.nav}`)} · ${t('app.name')}`;
  try { await import(`./views/${route.mod}.js`); }
  catch (e) { clear(main).append(h('div', { class: 'state error', role: 'alert' }, `${t('common.error')}: ${e.message}`)); return; }
  if (my !== seq) return;
  current?.remove();
  current = document.createElement(route.tag);
  current.params = params || {}; current.query = query || {}; current.path = path;
  clear(main).append(current);
  main.focus({ preventScroll: true });
}

let router;
async function boot() {
  try { await loadMessages(); } catch (e) { console.error(e); }
  // A ?theme=light|dark query (used for screenshots) wins over the stored preference for this load only.
  if (/[?&]theme=(light|dark)/.test(location.search)) store.set({ theme: document.documentElement.dataset.theme }); else applyTheme(store.get().theme);
  buildShell($('#app'));
  store.subscribe((a) => { if (a) promptKey(); }, (s) => s.authRequired);
  const goto = (p, q) => () => router.go(p, q);
  registerCommands(() => [
    ...ROUTES.filter((r) => !r.hidden).map((r) => ({ label: t('palette.goto', { name: t(`nav.${r.nav}`) }), run: goto(r.path) })),
    { label: t('palette.launch'), run: goto('/launch') },
    { label: t('palette.toggle_theme'), run: toggleTheme },
    { label: t('palette.set_key'), run: () => promptKey() },
  ]);
  registerCommands(async () => {
    const list = await api.instances().catch(() => []);
    return (list.items || list).flatMap((i) => [
      { label: t('palette.chat_with', { name: i.name || i.id }), hint: i.state, run: goto('/chat', { instance: i.id }) },
      { label: t('palette.logs_for', { name: i.name || i.id }), hint: i.state, run: goto(`/instances/${encodeURIComponent(i.id)}`) },
    ]);
  });
  router = createRouter({ routes: ROUTES, onNavigate: navigate });
  router.start();
}
boot();
