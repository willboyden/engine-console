import { EcView } from '../components/base.js';
import { h, clear, download, icon } from '../dom.js';
import { t } from '../i18n.js';
import { items, timeAgo } from '../format.js';
import { normPrompt, parseMessage } from '../adapt.js';
import { btn, select, field, emptyBox, errorBox, skeleton } from '../components/ui.js';
import { toast, toastError } from '../components/toast.js';
import { openDrawer, openModal, confirmDialog } from '../components/dialog.js';
import { toWire, streamCompletion, messageEl } from '../components/chat-core.js';

const MAX_IMG = 8 * 1024 * 1024;

class EcChat extends EcView {
  setup() {
    this.msgs = []; this.convId = null; this.busy = false; this.attach = [];
    this.gen = { temperature: 0.7, top_p: 1, max_tokens: 2048, system: '' };
    this.side = h('nav', { class: 'chat-side', 'aria-label': t('chat.history') });
    this.log = h('div', { class: 'chat-log', role: 'log', 'aria-live': 'polite', 'aria-label': t('chat.messages') });
    this.top = h('div', { class: 'row gap wrap between chat-top' });
    this.composer = this.buildComposer();
    this.append(h('div', { class: 'chat' }, this.side, h('div', { class: 'chat-main' }, this.top, this.log, this.composer)));
    this.init();
  }
  async init() {
    try {
      const all = items(await this.api.instances()); this.insts = all.filter((i) => i.state === 'ready');
      this.instId = this.query?.instance && this.insts.some((i) => i.id === this.query.instance) ? this.query.instance : this.insts[0]?.id || '';
    } catch (e) { clear(this.log).append(errorBox(e, () => this.init())); return; }
    if (!this._alive) return;
    this.paintTop(); this.paintLog(); this.loadHistory();
  }
  paintTop() {
    const opts = this.insts.length ? this.insts.map((i) => ({ value: i.id, label: `${i.name || i.id} (${i.repo_id})` })) : [{ value: '', label: t('chat.no_ready') }];
    clear(this.top).append(field(t('chat.model'), select(opts, this.instId, (v) => { this.instId = v; })),
      h('div', { class: 'row gap' }, btn(t('chat.prompts'), { onClick: () => this.prompts() }), btn(t('chat.params'), { onClick: () => this.paramsDrawer() }),
        btn(t('chat.export'), { icon: 'download', onClick: () => this.export() })));
  }
  paintLog() {
    clear(this.log);
    if (!this.msgs.length) { this.log.append(emptyBox(t('chat.empty'), this.insts.length ? t('chat.empty_hint') : t('chat.no_ready_hint'))); return; }
    for (const m of this.msgs) { const me = messageEl({ role: m.role }); this.log.append(me.el); me.flush({ text: m.text, reasoning: m.reasoning, images: m.images, usage: m.usage }); }
    this.log.scrollTop = this.log.scrollHeight;
  }

  // ----- history -----
  async loadHistory() {
    this.side.append(skeleton(3));
    try { this.convs = items(await this.api.conversations()); } catch (e) { clear(this.side).append(errorBox(e, () => this.loadHistory())); return; }
    if (!this._alive) return;
    this.paintSide();
  }
  paintSide() {
    clear(this.side).append(btn(t('chat.new'), { icon: 'plus', kind: 'primary', onClick: () => { this.msgs = []; this.convId = null; this.paintLog(); this.paintSide(); } }),
      this.convs.length ? h('ul', { class: 'hist' }, this.convs.map((c) => h('li', { class: c.id === this.convId ? 'on' : '' },
        h('button', { type: 'button', class: 'link', onClick: () => this.open(c) }, c.title || t('chat.untitled')), h('small', { class: 'muted' }, c.updated_at ? timeAgo(c.updated_at) : ''),
        h('button', { type: 'button', class: 'btn ghost sm', 'aria-label': t('chat.delete_named', { name: c.title || '' }), onClick: async () => { if (await confirmDialog(t('chat.confirm_delete'), { danger: true })) { try { await this.api.deleteConversation(c.id); this.convs = this.convs.filter((x) => x.id !== c.id); if (this.convId === c.id) { this.msgs = []; this.convId = null; this.paintLog(); } this.paintSide(); } catch (e) { toastError(e); } } } }, icon('trash', 14)))))
        : h('p', { class: 'hint' }, t('chat.no_history')));
  }
  async open(c) {
    let full = c;
    try { full = await this.api.conversation(c.id); } catch (e) { toastError(e); return; }
    if (!this._alive) return;
    this.convId = full.id; this.msgs = (full.messages || []).map(parseMessage);
    if (full.system_prompt != null) this.gen.system = full.system_prompt;
    if (full.instance_id && this.insts.some((i) => i.id === full.instance_id)) { this.instId = full.instance_id; this.paintTop(); }
    this.paintLog(); this.paintSide();
  }
  // ----- composer -----
  buildComposer() {
    this.ta = h('textarea', { rows: 2, placeholder: t('chat.placeholder'), 'aria-label': t('chat.placeholder'),
      onKeydown: (e) => { if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); this.send(); } } });
    this.thumbs = h('div', { class: 'row gap wrap' });
    const file = h('input', { type: 'file', accept: 'image/*', multiple: true, class: 'sr-only', id: 'chat-file', tabindex: '-1', onChange: (e) => this.addFiles([...e.target.files]).then(() => { e.target.value = ''; }) });
    this.sendBtn = btn(t('chat.send'), { icon: 'send', kind: 'primary', onClick: () => (this.busy ? this.stop() : this.send()) });
    return h('div', { class: 'composer' }, this.thumbs, h('div', { class: 'row gap end' }, this.ta,
      file, h('button', { type: 'button', class: 'btn', title: t('chat.attach'), 'aria-label': t('chat.attach'), onClick: () => file.click() }, icon('image')), this.sendBtn));
  }
  addFiles(files) {
    return Promise.all(files.map((f) => new Promise((res) => {
      if (!f.type.startsWith('image/')) { toast(t('chat.not_image', { name: f.name }), { kind: 'warn' }); return res(); }
      if (f.size > MAX_IMG) { toast(t('chat.too_big', { name: f.name }), { kind: 'warn' }); return res(); }
      const r = new FileReader(); r.onload = () => { this.attach.push(r.result); res(); }; r.onerror = () => res(); r.readAsDataURL(f);
    }))).then(() => this.paintThumbs());
  }
  paintThumbs() {
    clear(this.thumbs).append(...this.attach.map((u, i) => h('span', { class: 'thumb-wrap' }, h('img', { class: 'thumb', src: u, alt: t('chat.attached') }),
      h('button', { type: 'button', class: 'btn ghost sm', 'aria-label': t('chat.remove_image'), onClick: () => { this.attach.splice(i, 1); this.paintThumbs(); } }, icon('close', 12)))));
  }
  setBusy(b) {
    this.busy = b; this.ta.disabled = false;
    clear(this.sendBtn).append(icon(b ? 'stop' : 'send', 16), h('span', b ? t('chat.stop') : t('chat.send')));
  }
  async send() {
    const text = this.ta.value.trim();
    if ((!text && !this.attach.length) || !this.instId || this.busy) { if (!this.instId) toast(t('chat.no_ready'), { kind: 'warn' }); return; }
    this.setBusy(true);
    // The backend persists both turns itself when the request carries conversation_id, so create the conversation first.
    if (!this.convId) {
      try {
        const c = await this.api.createConversation({ title: (text || t('chat.image_chat')).slice(0, 60), system_prompt: this.gen.system || null, instance_id: this.instId });
        this.convId = c.id; this.convs.unshift(c); this.paintSide();
      } catch (e) { this.setBusy(false); toastError(e, t('chat.failed')); return; }
    }
    if (!this.msgs.length) clear(this.log);
    const user = { role: 'user', text, images: this.attach.slice() };
    this.msgs.push(user);
    const um = messageEl({ role: 'user' }); this.log.append(um.el); um.flush({ text, images: user.images });
    this.ta.value = ''; this.attach = []; this.paintThumbs();
    const ai = { role: 'assistant', text: '', reasoning: '' };
    this.msgs.push(ai);
    const am = messageEl({ role: 'assistant' }); this.log.append(am.el);
    const wire = [...(this.gen.system ? [{ role: 'system', content: this.gen.system }] : []), ...this.msgs.slice(0, -1).map(toWire)];
    const finish = (err) => {
      this.setBusy(false); this.handle = null;
      if (err) toastError(err, t('chat.failed'));
      am.flush({ text: ai.text, reasoning: ai.reasoning, usage: ai.usage });
      this.refreshHistory();
      this.ta.focus();
    };
    this.handle = streamCompletion({ instanceId: this.instId, conversationId: this.convId, messages: wire, params: { temperature: this.gen.temperature, top_p: this.gen.top_p, max_tokens: this.gen.max_tokens },
      onUpdate: (a) => { ai.text = a.text; ai.reasoning = a.reasoning; am.update({ text: a.text, reasoning: a.reasoning, streaming: true }); this.log.scrollTop = this.log.scrollHeight; },
      onDone: (a) => { ai.text = a.text; ai.reasoning = a.reasoning; ai.usage = a.usage; finish(); }, onError: finish });
    this.own(() => this.handle?.close());
  }
  async refreshHistory() { try { this.convs = items(await this.api.conversations()); if (this._alive) this.paintSide(); } catch { /* list refresh is cosmetic */ } }
  stop() { this.handle?.close(); this.setBusy(false); }

  // ----- drawers -----
  paramsDrawer() {
    const g = this.gen;
    const num = (k, o) => h('input', { type: 'number', value: String(g[k]), ...o, onChange: (e) => { g[k] = Number(e.target.value); } });
    const sys = h('textarea', { rows: 6, value: g.system, placeholder: t('chat.system_ph'), onChange: (e) => { g.system = e.target.value; } });
    openDrawer(t('chat.params'), h('div', { class: 'stack-v' }, field(t('chat.system'), sys),
      field(t('chat.temperature'), num('temperature', { min: 0, max: 2, step: 0.05 })), field('top_p', num('top_p', { min: 0, max: 1, step: 0.05 })), field(t('chat.max_tokens'), num('max_tokens', { min: 1, step: 1 }))));
  }
  async prompts() {
    const body = h('div', { class: 'stack-v' }, skeleton(3));
    const m = openModal(t('chat.prompts'), body);
    const draw = async () => {
      try {
        const list = items(await this.api.prompts()).map(normPrompt);
        const name = h('input', { type: 'text', placeholder: t('chat.prompt_name') });
        clear(body).append(list.length ? h('ul', { class: 'plist' }, list.map((p) => h('li', h('div', h('strong', p.name), h('div', { class: 'muted small ellipsis' }, p.text)),
          h('div', { class: 'row gap' }, btn(t('chat.use_system'), { size: 'sm', onClick: () => { this.gen.system = p.text; toast(t('chat.system_set'), { kind: 'ok', timeout: 1800 }); m.close(); } }),
            btn(t('chat.insert'), { size: 'sm', onClick: () => { this.ta.value = p.text; m.close(); this.ta.focus(); } }),
            btn('', { size: 'sm', icon: 'trash', title: t('common.delete'), onClick: async () => { try { await this.api.deletePrompt(p.id); draw(); } catch (e) { toastError(e); } } }))))) : emptyBox(t('chat.no_prompts')),
          h('form', { class: 'row gap end', onSubmit: async (e) => { e.preventDefault(); if (!name.value.trim() || !this.gen.system) { toast(t('chat.prompt_need'), { kind: 'warn' }); return; } try { await this.api.savePrompt({ title: name.value.trim(), content: this.gen.system }); draw(); } catch (err) { toastError(err); } } },
            field(t('chat.prompt_name'), name), btn(t('chat.save_system'), { type: 'submit' })));
      } catch (e) { clear(body).append(errorBox(e, draw)); }
    };
    draw();
  }
  export() {
    if (!this.msgs.length) { toast(t('chat.nothing_export'), { kind: 'warn' }); return; }
    const md = this.msgs.map((m) => `### ${m.role}\n\n${m.text}\n`).join('\n');
    const m = openModal(t('chat.export'), h('div', { class: 'row gap' },
      btn('Markdown', { onClick: () => { download('chat.md', md, 'text/markdown'); m.close(); } }),
      btn('JSON', { onClick: () => { download('chat.json', JSON.stringify({ instance_id: this.instId, messages: this.msgs.map(({ role, text, reasoning }) => ({ role, content: text, reasoning })) }, null, 2), 'application/json'); m.close(); } })));
  }
}
customElements.define('ec-chat', EcChat);
