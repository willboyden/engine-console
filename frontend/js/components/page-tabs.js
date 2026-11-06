// The frame around every page: ARIA tabs "<Page> | Getting started | FAQ" (help.js HELP_TABS) above the page. The page's
// own view stays mounted in the first panel while a help tab is open (so it keeps its state and its live updates); the
// help panels are built on first open. The open tab is mirrored into the hash query as `help=start|faq`
// (history.replaceState: no remount, deep links work). Arrow keys, Home and End move between the tabs (roving
// tabindex, automatic activation).
import { h } from '../dom.js';
import { t } from '../i18n.js';
import { buildHash } from '../router.js';
import { HELP_TABS } from '../help.js';
import { gettingStartedPanel, faqPanel } from './help-panel.js';

/**
 * @param {object} o
 * @param {HTMLElement} o.view  the page's view element (already created, not yet connected)
 * @param {string} o.key        the help entry (the route's nav id)
 * @param {string} o.name       the page's display name (first tab label, headings)
 * @param {string} o.path      the current route path and query, for the hash mirror
 * @param {object} o.query
 * @returns {HTMLElement} the frame (append it to #main)
 */
export function pageFrame({ view, key, name, path, query = {} }) {
  const frame = h('div', { class: 'page-frame' });
  const alive = () => frame.isConnected;
  const labels = { page: name, start: t('help.tab_start'), faq: t('help.tab_faq') };
  const nav = { tabLabel: (id) => labels[id] ?? id, goTab: (id) => select(id, true) };
  const tabs = HELP_TABS.map((id) =>
    h(
      'button',
      {
        type: 'button',
        role: 'tab',
        class: 'tab',
        id: `ptab-${id}`,
        'aria-controls': `ppanel-${id}`,
        dataset: { tab: id },
        onClick: () => select(id),
      },
      labels[id]
    )
  );
  const panels = Object.fromEntries(
    HELP_TABS.map((id) => [
      id,
      h('div', {
        role: 'tabpanel',
        id: `ppanel-${id}`,
        class: `page-panel page-panel-${id}`,
        'aria-labelledby': `ptab-${id}`,
        tabindex: '-1',
      }),
    ])
  );
  panels.page.append(view);
  const built = new Set(['page']);
  const list = h('div', { class: 'tabs page-tabs', role: 'tablist', 'aria-label': t('help.tabs', { name }) }, tabs);
  list.addEventListener('keydown', (e) => {
    const i = tabs.indexOf(document.activeElement);
    if (i < 0) return;
    const to = { ArrowRight: i + 1, ArrowLeft: i - 1, Home: 0, End: tabs.length - 1 }[e.key];
    if (to === undefined) return;
    e.preventDefault();
    select(HELP_TABS[(to + tabs.length) % tabs.length], true);
  });

  function select(id, focus = false) {
    if (!HELP_TABS.includes(id)) id = 'page';
    if (!built.has(id)) {
      built.add(id);
      const ctx = { key, name, nav, alive, path };
      panels[id].append(id === 'start' ? gettingStartedPanel(ctx) : faqPanel(ctx));
    }
    for (const [i, b] of tabs.entries()) {
      const on = HELP_TABS[i] === id;
      b.setAttribute('aria-selected', String(on));
      b.tabIndex = on ? 0 : -1;
      b.classList.toggle('on', on);
      panels[HELP_TABS[i]].hidden = !on;
      if (on && focus) b.focus();
    }
    frame.dataset.tab = id;
    try {
      history.replaceState(history.state, '', buildHash(path, { ...query, help: id === 'page' ? null : id }));
    } catch {
      /* ignore */
    }
  }

  frame.append(list, ...HELP_TABS.map((id) => panels[id]));
  select(HELP_TABS.includes(query.help) ? query.help : 'page');
  return frame;
}
