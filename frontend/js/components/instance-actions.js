// What the instance buttons do (shared by the instance list, the instance page and the machine page). The hub decides what
// is allowed (`controls`); this only performs the call and explains a refusal.
import { t } from '../i18n.js';
import { api } from '../app-context.js';
import { toast, toastError } from './toast.js';
import { confirmDialog } from './dialog.js';
import { managementOf } from '../controls.js';

/** id: start | stop | restart | pin | remove | configure. `after(kind)` runs when the data should be reloaded. */
export async function doInstanceAction(inst, id, { after } = {}) {
  try {
    if (id === 'configure') { location.hash = `#/launch?from=${encodeURIComponent(inst.id)}`; return; }
    if (id === 'remove') {
      if (!(await confirmDialog(t('inst.confirm_remove'), { danger: true }))) return;
      await api.deleteInstance(inst.id); after?.('removed'); return;
    }
    if (id === 'pin') { await api.patchInstance(inst.id, { pinned: !inst.pinned }); after?.(); return; }
    const mgmt = managementOf(inst);
    // Stopping something the console did not start itself, or that a script owns, is worth a second look.
    if (id !== 'start' && mgmt !== 'managed') {
      const msg = mgmt === 'script' ? t('inst.confirm_script', { action: t(`inst.${id}`).toLowerCase(), script: inst.script || inst.name }) : t('inst.confirm_external', { action: t(`inst.${id}`).toLowerCase(), name: inst.name || inst.id });
      if (!(await confirmDialog(msg, { confirmLabel: t(`inst.${id}`), danger: true }))) return;
    }
    await api.instanceAction(inst.id, id);
    toast(t(`inst.did_${id}`), { kind: 'ok', timeout: 2200});
    after?.();
  } catch (e) { toastError(e); after?.(); }
}
