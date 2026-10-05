// Launch checks (PreflightCheck: code, level, message) as a list: a status icon that is also spelled out for screen readers, a
// plain-language title for the check code with its info button (concept check.<code>), and the hub's own message beneath.
import { h, icon } from '../dom.js';
import { t } from '../i18n.js';
import { withTip } from './info-tip.js';

const ORDER = { block: 0, warn: 1, ok: 2 };
const LEVEL_ICON = { ok: 'check', warn: 'warn', block: 'close' };
const levelWord = (level) => { const k = `check.level.${level}`; const v = t(k); return v === k ? level : v; };

/** Humanise an unknown code ("no_free_port" -> "No free port") so nothing shows as a bare key. */
export const codeTitle = (code) => {
  const k = `check.${code}`;
  const v = t(k);
  if (v !== k) return v;
  const s = String(code).replace(/_/g, ' ');
  return s.charAt(0).toUpperCase() + s.slice(1);
};

export function checkList(checks, { sort = true } = {}) {
  const list = sort ? [...checks].sort((a, b) => (ORDER[a.level] ?? 3) - (ORDER[b.level] ?? 3)) : checks;
  return h('ul', { class: 'check-list' }, list.map((c) => h('li', { class: `check-item ${c.level}` },
    h('span', { class: 'check-icon', 'aria-hidden': 'true' }, icon(LEVEL_ICON[c.level] || 'warn', 14)),
    h('div', { class: 'check-body' },
      h('span', { class: 'sr-only' }, `${levelWord(c.level)}: `),
      withTip(h('strong', codeTitle(c.code)), { kind: 'concept', key: `check.${c.code}` }, codeTitle(c.code)),
      h('div', { class: 'muted small check-msg' }, c.message)))));
}
