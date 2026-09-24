// Shared streaming + rendering used by Chat and Arena.
import { h, setTrustedHtml, copyText } from '../dom.js';
import { t } from '../i18n.js';
import { api, stream } from '../app-context.js';
import { renderMarkdown, splitThink } from '../markdown.js';
import { toast } from './toast.js';

export function toWire(msg) {
  if (msg.images?.length) return { role: msg.role, content: [{ type: 'text', text: msg.text || '' }, ...msg.images.map((url) => ({ type: 'image_url', image_url: { url } }))] };
  return { role: msg.role, content: msg.text || '' };
}

// Stream one completion. Handlers receive accumulated {text, reasoning}. Returns {close()}.
export function streamCompletion({ instanceId, conversationId, messages, params = {}, onUpdate, onDone, onError }) {
  const acc = { text: '', reasoning: '', usage: null };
  let finished = false;
  const done = (err) => { if (finished) return; finished = true; err ? onError?.(err) : onDone?.(acc); };
  const body = { instance_id: instanceId, ...(conversationId ? { conversation_id: conversationId } : {}), messages, stream: true, stream_options: { include_usage: true },
    ...Object.fromEntries(Object.entries(params).filter(([, v]) => v != null && v !== '')) };
  return stream('/chat/completions', {
    method: 'POST', body: JSON.stringify(body), reconnect: false, headers: { 'Content-Type': 'application/json' },
    onEvent: (ev) => {
      if (ev.event === 'done') return done();
      if (ev.event === 'error') return done(new Error(ev.data?.detail || ev.data?.message || 'stream error'));
      const c = ev.data?.choices?.[0]?.delta;
      if (c?.content) acc.text += c.content;
      const r = c?.reasoning_content ?? c?.reasoning;
      if (r) acc.reasoning += r;
      if (ev.data?.usage) acc.usage = ev.data.usage;
      onUpdate?.(acc);
    },
    onError: (e) => done(e),
    onClose: () => done(),
  });
}

// Message bubble whose body can be re-rendered cheaply while streaming (rAF-throttled).
export function messageEl({ role, label }) {
  const reasoning = h('details', { class: 'reasoning', hidden: true }, h('summary', t('chat.reasoning')), h('pre', { class: 'reasoning-text' }));
  const body = h('div', { class: 'md' });
  const meta = h('div', { class: 'msg-meta muted small' });
  const imgs = h('div', { class: 'row gap wrap' });
  const el = h('article', { class: `msg ${role}`, 'aria-label': label || role }, h('div', { class: 'msg-role' }, label || t(`chat.role_${role}`)), imgs, reasoning, body, meta);
  body.addEventListener('click', (e) => {
    const b = e.target.closest('[data-action="copy-code"]');
    if (b) copyText(b.closest('.codeblock').querySelector('code').textContent).then(() => toast(t('common.copied'), { kind: 'ok', timeout: 1500 }));
  });
  let raf = 0, last = null, autoOpened = false;
  const paint = () => {
    raf = 0;
    const { text, reasoning: rs, images, usage, streaming } = last;
    const sp = splitThink(text);
    const allReason = [rs, sp.reasoning].filter(Boolean).join('\n');
    reasoning.hidden = !allReason;
    if (allReason) {
      reasoning.querySelector('pre').textContent = allReason;
      // Open while the model is still thinking so the user sees progress; collapsed once the answer starts.
      if (streaming && !sp.content) { reasoning.open = true; autoOpened = true; }
      else if (autoOpened) { reasoning.open = false; autoOpened = false; } // fold once the answer starts
      reasoning.querySelector('summary').textContent = streaming && !sp.content ? t('chat.thinking') : t('chat.reasoning');
    }
    setTrustedHtml(body, renderMarkdown(sp.content, { copyLabel: t('common.copy') }));
    if (streaming) body.classList.add('streaming'); else body.classList.remove('streaming');
    if (images && imgs.childElementCount !== images.length) { imgs.replaceChildren(...images.map((u) => h('img', { class: 'thumb', src: u, alt: t('chat.attached') }))); }
    meta.textContent = usage ? t('chat.usage', { p: usage.prompt_tokens ?? '–', c: usage.completion_tokens ?? '–' }) : '';
  };
  return {
    el,
    update(state) { last = state; if (!raf) raf = requestAnimationFrame(paint); },
    flush(state) { last = state; if (raf) cancelAnimationFrame(raf); paint(); },
  };
}
