// Machine + folder pickers used wherever files are written to a machine (download, copy). The machine list shows why a
// machine is not an option (down, monitor-only...); the folder list shows each writable root with the free space of its
// drive. Host details (roots) are fetched on demand and cached.
import { h, clear } from '../dom.js';
import { t } from '../i18n.js';
import { fmtBytes } from '../format.js';
import { manageBlock, rootChoices } from '../fleet.js';
import { normHost } from '../adapt.js';
import { field, reasonText } from './ui.js';

/**
 * hostRootPicker({api, hosts, kinds, exclude, initialHost, onChange})
 * -> {el, get(), whenReady}; get() = {hostId, rootId, root, free, detail, ok}. `exclude` is a Set of host ids that are not
 * offered at all (e.g. machines that already hold the model).
 */
export function hostRootPicker({ api, hosts, kinds, exclude = new Set(), initialHost, onChange }) {
  const list = hosts.filter((x) => !exclude.has(x.id));
  const details = new Map();
  const state = { hostId: null, rootId: null, root: null, free: null, detail: null, ok: false };
  const hostSel = h('select', { id: 'pick-host' });
  const rootSel = h('select', { id: 'pick-root' });
  const rootNote = h('div', { class: 'hint', role: 'status' });
  const blocked = list.filter((x) => manageBlock(x));
  const eligible = list.filter((x) => !manageBlock(x));
  // A machine that cannot take files is listed (so it is clear it exists) but not selectable, with the short reason in its label.
  const word = (x) => (x.link?.state && x.link.state !== 'up' ? t(`link.${x.link.state}`) : t('mode.monitor'));
  for (const x of list) hostSel.append(h('option', { value: x.id, disabled: !!manageBlock(x) }, manageBlock(x) ? `${x.label} (${word(x)})` : x.label));
  // Why a machine is greyed out, one click away rather than a wall of text.
  const blockedNote = blocked.length ? h('details', { class: 'blocked-details' }, h('summary', { class: 'small' }, t('picker.blocked_summary', { count: blocked.length })),
    h('ul', { class: 'hint blocked-list' }, blocked.map((x) => h('li', {}, h('strong', x.label), ': ', reasonText(manageBlock(x)))))) : null;

  const publish = () => onChange?.({ ...state });
  async function loadHost(id) {
    state.hostId = id; state.rootId = null; state.root = null; state.free = null; state.ok = false;
    clear(rootSel).append(h('option', { value: '' }, t('common.loading')));
    clear(rootNote);
    try {
      if (!details.has(id)) details.set(id, normHost(await api.host(id)));
    } catch (e) { clear(rootSel).append(h('option', { value: '' }, t('picker.root_failed'))); rootNote.textContent = e.message || ''; publish(); return; }
    state.detail = details.get(id);
    const choices = rootChoices(state.detail, { kinds });
    clear(rootSel);
    if (!choices.length) { rootSel.append(h('option', { value: '' }, t('picker.no_root'))); rootNote.textContent = t('picker.no_root_hint', { host: state.detail.label }); publish(); return; }
    for (const c of choices) rootSel.append(h('option', { value: c.root.id }, `${c.label} · ${c.free != null ? t('picker.free', { free: fmtBytes(c.free, 0) }) : t('picker.free_unknown')} · ${c.root.path}`));
    pickRoot(choices[0].root.id, choices);
  }
  function pickRoot(id, choices) {
    const c = (choices || rootChoices(state.detail, { kinds })).find((x) => x.root.id === id);
    state.rootId = c?.root.id ?? null; state.root = c?.root ?? null; state.free = c?.free ?? null; state.ok = !!c;
    rootSel.value = state.rootId || '';
    publish();
  }
  hostSel.addEventListener('change', () => loadHost(hostSel.value));
  rootSel.addEventListener('change', () => pickRoot(rootSel.value));

  const first = (initialHost && eligible.find((x) => x.id === initialHost)) || eligible[0];
  const el = h('div', { class: 'stack-v picker' },
    field(t('picker.machine'), hostSel, null, { kind: 'concept', key: 'field.target_host', inline: false }), blockedNote,
    field(t('picker.root'), rootSel, null, { kind: 'concept', key: 'field.root' }), rootNote);
  let whenReady = Promise.resolve();
  if (first) { hostSel.value = first.id; whenReady = loadHost(first.id); }
  else { hostSel.disabled = true; rootSel.disabled = true; rootNote.textContent = t('picker.no_machine'); }
  return { el, get: () => ({ ...state }), whenReady };
}
