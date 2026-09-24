import { EcView } from '../components/base.js';
import { h, clear, icon } from '../dom.js';
import { t } from '../i18n.js';
import { items, fmtDate, timeAgo } from '../format.js';
import { normKey } from '../adapt.js';
import { store, applyTheme, setApiKey } from '../app-context.js';
import { btn, card, field, select, badge, emptyBox, errorBox, skeleton } from '../components/ui.js';
import { toast, toastError } from '../components/toast.js';
import { openModal, confirmDialog } from '../components/dialog.js';
import { copyText } from '../dom.js';

class EcSettings extends EcView {
  setup() {
    this.gen = h('div', {}, skeleton(3)); this.keysHost = h('div', {}, skeleton(2)); this.auditHost = h('div', {}, skeleton(3));
    const theme = select([{ value: 'dark', label: t('settings.dark') }, { value: 'light', label: t('settings.light') }, { value: 'system', label: t('settings.system') }], store.get().theme, (v) => applyTheme(v));
    const lang = select([{ value: 'en', label: 'English' }], 'en', () => {}, { disabled: true });
    this.append(h('h1', t('nav.settings')),
      card(t('settings.appearance'), h('div', { class: 'row gap wrap' }, field(t('settings.theme'), theme), field(t('settings.language'), lang, t('settings.language_hint')))),
      card(t('settings.general'), this.gen), card(t('settings.keys'), this.keysHost, btn(t('settings.new_key'), { icon: 'plus', size: 'sm', onClick: () => this.newKey() })),
      card(t('settings.session'), h('div', { class: 'row gap' }, h('span', store.get().apiKey ? t('settings.session_key_set') : t('settings.session_key_none')), btn(t('auth.clear'), { size: 'sm', onClick: () => { setApiKey(''); toast(t('settings.key_cleared'), { kind: 'ok' }); } }))),
      card(t('settings.audit'), this.auditHost));
    this.loadGeneral(); this.loadKeys(); this.loadAudit();
  }
  async loadGeneral() {
    try {
      const s = await this.api.settings(); if (!this._alive) return;
      const ttl = h('input', { type: 'number', min: 0, value: String(Math.round((s.idle_ttl_s || 0) / 60)) });
      const form = h('form', { class: 'stack-v', onSubmit: async (e) => { e.preventDefault(); try { await this.api.saveSettings({ idle_ttl_s: Math.round(Number(ttl.value) * 60) }); toast(t('settings.saved'), { kind: 'ok' }); } catch (err) { toastError(err); } } },
        h('div', { class: 'stack-v' }, h('div', { class: 'field' }, h('span', { class: 'lbl' }, t('settings.cache_dir_ro')), h('code', { class: 'ro' }, s.hf_cache_dir || '–'), h('div', { class: 'hint' }, t('settings.cache_dir_hint'))), field(t('settings.idle_ttl'), ttl, t('settings.idle_ttl_hint'))),
        // Token value is never returned by the API; we only ever show the status string it gives us.
        h('div', { class: 'row gap' }, h('strong', t('settings.hf_token')), badge(s.hf_token || t('settings.unset'), /set/.test(s.hf_token || '') && !/unset/.test(s.hf_token || '') ? 'ok' : 'warn'), h('span', { class: 'hint' }, t('settings.hf_token_hint'))),
        s.image_pins && Object.keys(s.image_pins).length ? h('div', { class: 'small muted' }, `${t('settings.images')} (${t('settings.read_only')}): ${Object.entries(s.image_pins).map(([k, v]) => `${k} ${v}`).join(' · ')}`) : null,
        h('div', { class: 'row' }, btn(t('common.save'), { kind: 'primary', type: 'submit' })));
      clear(this.gen).append(form);
    } catch (e) { if (this._alive) clear(this.gen).append(errorBox(e, () => this.loadGeneral())); }
  }
  async loadKeys() {
    try {
      const keys = items(await this.api.keys()).map(normKey); if (!this._alive) return;
      clear(this.keysHost).append(keys.length ? h('div', { class: 'tablewrap' }, h('table', h('thead', h('tr', ['settings.key_name', 'settings.key_role', 'settings.key_created', 'settings.key_used', ''].map((k) => h('th', { scope: 'col' }, k ? t(k) : '')))),
        h('tbody', keys.map((k) => h('tr', h('td', h('strong', k.name), ' ', h('code', k.prefix || '')), h('td', badge(k.role, k.role === 'admin' ? 'info' : 'muted')), h('td', k.created_at ? timeAgo(k.created_at) : '–'), h('td', k.last_used ? timeAgo(k.last_used) : t('settings.never')),
          h('td', btn('', { icon: 'trash', size: 'sm', title: t('settings.revoke'), onClick: async () => { if (await confirmDialog(t('settings.confirm_revoke', { name: k.name }), { danger: true })) { try { await this.api.deleteKey(k.id); this.loadKeys(); } catch (e) { toastError(e); } } } }))))))) : emptyBox(t('settings.no_keys')));
    } catch (e) { if (this._alive) clear(this.keysHost).append(errorBox(e, () => this.loadKeys())); }
  }
  newKey() {
    const name = h('input', { type: 'text', placeholder: t('settings.key_name_ph') });
    const role = select(['admin', 'viewer'], 'viewer');
    const m = openModal(t('settings.new_key'), h('form', { class: 'stack-v', onSubmit: async (e) => {
      e.preventDefault();
      try {
        const k = normKey(await this.api.createKey({ name: name.value.trim() || 'key', role: role.value }));
        clear(m.body).append(h('p', { class: 'note warn' }, t('settings.key_once')), h('pre', { class: 'cmd', tabindex: '0' }, k.key),
          h('div', { class: 'row gap end' }, btn(t('common.copy'), { icon: 'copy', onClick: async () => { await copyText(k.key); toast(t('common.copied'), { kind: 'ok', timeout: 1500 }); } }), btn(t('common.close'), { onClick: () => m.close() })));
        this.loadKeys();
      } catch (err) { toastError(err); }
    } }, field(t('settings.key_name'), name), field(t('settings.key_role'), role), h('div', { class: 'row end' }, btn(t('settings.create'), { kind: 'primary', type: 'submit' })))); name.focus();
  }
  async loadAudit() {
    try {
      const rows = items(await this.api.audit({ limit: 100 })); if (!this._alive) return;
      clear(this.auditHost).append(rows.length ? h('div', { class: 'tablewrap tall' }, h('table', { class: 'compact' }, h('thead', h('tr', ['audit.when', 'audit.actor', 'audit.action', 'audit.status'].map((k) => h('th', { scope: 'col' }, t(k))))),
        h('tbody', rows.map((r) => h('tr', h('td', { class: 'num' }, fmtDate(r.ts)), h('td', r.actor || '–'), h('td', h('code', `${r.method || ''} ${r.path || r.action || ''}`), r.params ? h('div', { class: 'muted small ellipsis' }, typeof r.params === 'string' ? r.params : JSON.stringify(r.params)) : null), h('td', badge(String(r.status ?? '–'), (r.status ?? 200) >= 400 ? 'bad' : 'ok'))))))) : emptyBox(t('audit.empty')));
    } catch (e) { if (this._alive) clear(this.auditHost).append(errorBox(e, () => this.loadAudit())); }
  }
}
customElements.define('ec-settings', EcSettings);
