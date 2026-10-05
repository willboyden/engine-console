// The action buttons of one instance, straight from `controls` (controls.js): an enabled control acts, a control that is
// relevant but not allowed stays visible, disabled, with its reason as the accessible description. Nothing is silently missing.
import { h } from '../dom.js';
import { t } from '../i18n.js';
import { controlButtons } from '../controls.js';
import { btn, reasonText } from './ui.js';

const ICON = { start: 'play', stop: 'stop', restart: 'restart', pin: 'pin', remove: 'trash', configure: 'copy' };

/**
 * controlBar(instance, {onAction, size, iconOnly, only}) -> {buttons: HTMLElement[], note: HTMLElement|null, noteText: string|null}
 * onAction(id) receives start | stop | restart | pin | remove | configure. `iconOnly` draws icon buttons whose name is the
 * action (for table rows); `only` limits which controls are drawn.
 */
export function controlBar(inst, { onAction, size = '', iconOnly = false, only } = {}) {
  const { buttons, note, management } = controlButtons(inst);
  const label = (id) => ({ start: t('inst.start'), stop: t('inst.stop'), restart: t('inst.restart'), configure: management === 'external' ? t('inst.clone') : t('inst.edit'),
    pin: inst.pinned ? t('inst.unpin') : t('inst.pin'), remove: t('inst.remove') }[id]);
  const els = buttons.filter((b) => !only || only.includes(b.id)).map((b) => {
    const icon = iconOnly || b.id === 'remove';
    return btn(icon ? '' : label(b.id), { icon: ICON[b.id], kind: iconOnly ? '' : b.kind, size, disabled: !b.enabled, reason: b.reason, ariaLabel: icon ? label(b.id) : undefined, title: icon ? label(b.id) : undefined, onClick: () => onAction?.(b.id) });
  });
  const noteText = note ? reasonText(note) : null;
  return { buttons: els, note: note ? h('p', { class: 'note info small controls-note' }, noteText) : null, noteText };
}
