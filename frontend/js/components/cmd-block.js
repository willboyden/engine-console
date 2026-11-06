// A command for the human to run in a terminal (the console never runs it): a labelled, focusable block with a Copy
// button. Separate from ui.js because it needs the toast element, which ui.js (imported by Node tests) must not load.
import { h, copyText } from '../dom.js';
import { t } from '../i18n.js';
import { btn } from './ui.js';
import { toast } from './toast.js';

export function cmdBlock(command, label) {
  const copy = btn(t('common.copy'), { icon: 'copy', size: 'sm', onClick: async () => { const ok = await copyText(command); toast(ok ? t('common.copied') : t('common.copy_failed'), { kind: ok ? 'ok' : 'bad', timeout: 1800 }); } });
  return h('div', { class: 'cmd-block' }, label ? h('span', { class: 'cmd-label' }, label) : null, h('pre', { class: 'cmd', tabindex: '0' }, command), copy);
}
