// Bootstrap: i18n, theme, learn bundle, shell (nav/topbar), router -> lazy-loaded view custom elements, sign-in dialog.
import { loadMessages, t } from './i18n.js';
import { store, api, setApiKey, applyTheme } from './app-context.js';
import { createRouter, buildHash } from './router.js';
import { h, icon, clear, $ } from './dom.js';
import { items } from './format.js';
import { loadLearn, resetLearn } from './learn.js';
import { isBeginner } from './beginner.js';
import { toast } from './components/toast.js';
import { registerCommands } from './components/palette.js';
import { openModal } from './components/dialog.js';
import { instanceTitle, scriptLabelMap } from './controls.js';
import { loadScripts } from './components/script-actions.js';
import './components/toast.js';
import './components/palette.js';

// `navPath` highlights another nav entry for a hidden route; `redirect(query)` returns a path to go to instead (or null).
export const ROUTES = [
  { path: '/', tag: 'ec-dashboard', mod: 'dashboard', nav: 'dashboard', icon: 'dashboard' },
  { path: '/hosts/:id', tag: 'ec-host', mod: 'host', nav: 'dashboard', icon: 'server', hidden: true, navPath: '/' },
  { path: '/library', tag: 'ec-library', mod: 'library', nav: 'library', icon: 'library' },
  { path: '/discover', tag: 'ec-discover', mod: 'discover', nav: 'discover', icon: 'discover' },
  { path: '/models', tag: 'ec-discover', mod: 'discover', nav: 'discover', icon: 'discover', hidden: true, navPath: '/discover', redirect: (q) => (q.tab === 'local' ? '/library' : '/discover') },
  { path: '/downloads', tag: 'ec-downloads', mod: 'downloads', nav: 'downloads', icon: 'downloads' },
  { path: '/launch', tag: 'ec-launch', mod: 'launch', nav: 'launch', icon: 'launch', redirect: (q) => (isBeginner() && !q.advanced && !q.from ? '/launch/wizard' : null) },
  { path: '/launch/wizard', tag: 'ec-wizard', mod: 'wizard', nav: 'launch', icon: 'wand', hidden: true, navPath: '/launch' },
  { path: '/instances', tag: 'ec-instances', mod: 'instances', nav: 'instances', icon: 'instances' },
  { path: '/instances/:id', tag: 'ec-instances', mod: 'instances', nav: 'instances', icon: 'instances', hidden: true, navPath: '/instances' },
  { path: '/metrics', tag: 'ec-metrics', mod: 'metrics', nav: 'metrics', icon: 'metrics' },
  { path: '/bench', tag: 'ec-bench', mod: 'bench', nav: 'bench', icon: 'bench' },
  { path: '/chat', tag: 'ec-chat', mod: 'chat', nav: 'chat', icon: 'chat' },
  { path: '/arena', tag: 'ec-arena', mod: 'arena', nav: 'arena', icon: 'arena' },
  { path: '/usage', tag: 'ec-usage', mod: 'usage', nav: 'usage', icon: 'usage' },
  { path: '/learn', tag: 'ec-learn', mod: 'learn', nav: 'learn', icon: 'book' },
  { path: '/settings', tag: 'ec-settings', mod: 'settings', nav: 'settings', icon: 'settings' },
];

function buildShell(root) {
  const nav = h('nav', { class: 'nav', 'aria-label': t('nav.primary') },
    ROUTES.filter((r) => !r.hidden).map((r) => h('a', { href: `#${r.path}`, 'data-route': r.path }, icon(r.icon), h('span', t(`nav.${r.nav}`)))));
  const themeBtn = h('button', { type: 'button', class: 'btn ghost', id: 'theme-btn', 'aria-label': t('topbar.theme'), onClick: toggleTheme });
  const keyBtn = h('button', { type: 'button', class: 'btn ghost', 'aria-label': t('topbar.api_key'), title: t('topbar.api_key'), onClick: () => promptKey() }, icon('key'));
  const palBtn = h('button', { type: 'button', class: 'btn ghost palette-btn', 'aria-label': t('palette.open'), onClick: () => window.dispatchEvent(new Event('ec:palette')) },
    icon('search', 16), h('span', { class: 'palette-label' }, t('palette.open')), h('kbd', 'Ctrl K'));
  root.append(
    h('a', { class: 'skip', href: '#main' }, t('common.skip')),
    h('aside', { class: 'sidebar' }, h('div', { class: 'brand' }, h('span', { class: 'logo', 'aria-hidden': 'true' }, '◆'), h('span', t('app.name'))), nav),
    h('div', { class: 'content' },
      h('header', { class: 'topbar' }, h('div', { id: 'crumb', class: 'crumb' }), h('div', { class: 'row gap' }, palBtn, themeBtn, keyBtn)),
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

// ---- sign in ----
// Shaped like a login form so the browser's password manager can save and fill the key: a (read-only) user name field with
// autocomplete=username and the key as the current-password. The key itself stays in this tab's sessionStorage (setApiKey).
export const AUTH_USER = 'engine-console';
let authDialog = null;
function promptKey({ rejected = false } = {}) {
  if (authDialog) return;
  const user = h('input', { type: 'text', id: 'auth-user', name: 'username', value: AUTH_USER, autocomplete: 'username', readonly: true, class: 'auth-user' });
  const input = h('input', { type: 'password', id: 'key-in', name: 'password', autocomplete: 'current-password', spellcheck: 'false', required: true, placeholder: 'ec_…', 'aria-describedby': 'auth-help', value: store.get().apiKey || '' });
  const form = h('form', { class: 'stack-v auth-form', method: 'post', onSubmit: (e) => {
    e.preventDefault();
    setApiKey(input.value.trim()); resetLearn();
    authDialog?.close(); toast(t('auth.saved'), { kind: 'ok' }); router.refresh();
  } },
  h('p', { id: 'auth-help' }, t('auth.explain')),
  rejected ? h('p', { class: 'note bad', role: 'alert' }, t('auth.rejected')) : null,
  h('div', { class: 'field' }, h('label', { for: 'auth-user' }, t('auth.user_label')), user),
  h('div', { class: 'field' }, h('label', { for: 'key-in' }, t('auth.key_label')), input, h('div', { class: 'hint' }, t('auth.manager_hint'))),
  h('div', { class: 'row gap end' }, h('button', { type: 'button', class: 'btn', onClick: () => { setApiKey(''); authDialog?.close(); } }, t('auth.clear')), h('button', { type: 'submit', class: 'btn primary' }, t('auth.sign_in'))));
  authDialog = openModal(t('auth.title'), form, { onClose: () => { authDialog = null; } });
  input.focus();
}

// ---- routing ----
let current = null, seq = 0, router;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function navigate({ path, route, params, query }) {
  const main = $('#main');
  const my = ++seq;
  if (!route) { clear(main).append(h('div', { class: 'state empty' }, h('strong', t('common.not_found')), h('a', { href: '#/' }, t('nav.dashboard')))); return; }
  const to = route.redirect?.(query || {});
  if (to && to !== path) { location.replace(buildHash(to, query)); return; }
  const active = route.navPath || path;
  document.querySelectorAll('.nav a').forEach((a) => {
    const on = a.dataset.route === active || (a.dataset.route !== '/' && active.startsWith(`${a.dataset.route}/`));
    if (on) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current');
  });
  $('#crumb').textContent = t(`nav.${route.nav}`);
  document.title = `${t(`nav.${route.nav}`)} · ${t('app.name')}`;
  // The learn bundle is fetched once (cached in memory); wait briefly so the first paint already has its help icons.
  await Promise.race([loadLearn(api), sleep(2500)]);
  try { await import(`./views/${route.mod}.js`); }
  catch (e) { clear(main).append(h('div', { class: 'state error', role: 'alert' }, h('strong', `${t('common.error')}: ${e.message}`), h('p', t('err.module')))); return; }
  if (my !== seq) return;
  current?.remove();
  current = document.createElement(route.tag);
  current.params = params || {}; current.query = query || {}; current.path = path;
  clear(main).append(current);
  main.focus({ preventScroll: true });
}

async function boot() {
  try { await loadMessages(); } catch (e) { console.error(e); }
  // A ?theme=light|dark query (used for screenshots) wins over the stored preference for this load only.
  if (/[?&]theme=(light|dark)/.test(location.search)) store.set({ theme: document.documentElement.dataset.theme }); else applyTheme(store.get().theme);
  buildShell($('#app'));
  store.subscribe((a) => { if (a) promptKey({ rejected: !!store.get().apiKey }); }, (s) => s.authRequired);
  const goto = (p, q) => () => router.go(p, q);
  registerCommands(() => [
    ...ROUTES.filter((r) => !r.hidden).map((r) => ({ label: t('palette.goto', { name: t(`nav.${r.nav}`) }), run: goto(r.path) })),
    { label: t('palette.launch'), run: goto('/launch') },
    { label: t('palette.wizard'), run: goto('/launch/wizard') },
    { label: t('palette.toggle_theme'), run: toggleTheme },
    { label: t('palette.set_key'), run: () => promptKey() },
  ]);
  registerCommands(async () => {
    const [list, scripts] = await Promise.all([api.instances().catch(() => []), loadScripts().catch(() => [])]);
    const labels = scriptLabelMap(scripts);
    return (list.items || list).flatMap((i) => { const name = instanceTitle(i, labels).title; return [
      { label: t('palette.chat_with', { name }), hint: i.state, run: goto('/chat', { instance: i.id }) },
      { label: t('palette.logs_for', { name }), hint: i.state, run: goto(`/instances/${encodeURIComponent(i.id)}`) },
    ]; });
  });
  registerCommands(async () => items(await api.hosts().catch(() => [])).map((x) => ({ label: t('palette.open_host', { name: x.label || x.id }), hint: x.link?.state, run: goto(`/hosts/${encodeURIComponent(x.id)}`) })));
  router = createRouter({ routes: ROUTES, onNavigate: navigate });
  router.start();
}
boot();
