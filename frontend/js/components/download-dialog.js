// "Download to a machine": pick the machine and folder (with free space) and, for a GGUF repository, one quantization.
// Opens from Discover; the request is POST /downloads {repo_id, host_id, root_id, allow_patterns}.
import { h, clear, append } from '../dom.js';
import { t } from '../i18n.js';
import { items, fmtBytes } from '../format.js';
import { normHost } from '../adapt.js';
import { quickFit } from '../fit-format.js';
import { openModal } from './dialog.js';
import { btn, skeleton, errorBox, badge, field } from './ui.js';
import { toast, toastError } from './toast.js';
import { hostRootPicker } from './host-root-picker.js';
import { infoTip } from './info-tip.js';

const PREFERRED_QUANT = 'Q4_K_M';

/** The default quantization for a GGUF repo: the popular middle ground when offered, else the first group. */
export const defaultQuant = (groups) => (groups.find((g) => g.quant === PREFERRED_QUANT) || groups[0])?.quant ?? null;

export function openDownloadDialog({ api, repo, hosts, initialHost, onQueued }) {
  const body = h('div', { class: 'stack-v' }, skeleton(3));
  const dlg = openModal(t('dl.dialog_title', { repo }), body);
  (async () => {
    let groups = [], weightBytes = null;
    const [detail, hostList] = await Promise.all([
      api.hfModel(repo).catch(() => null),
      hosts ? Promise.resolve(hosts) : api.hosts().then((r) => items(r).map(normHost)).catch((e) => e),
    ]);
    if (hostList instanceof Error) { clear(body).append(errorBox(hostList)); return; }
    groups = detail?.gguf_groups || [];
    weightBytes = detail?.info?.weight_bytes ?? null;
    let quant = defaultQuant(groups), picked = null;
    const need = () => (groups.length ? groups.find((g) => g.quant === quant)?.total_bytes ?? null : weightBytes);
    const note = h('div', { class: 'stack-v tight', role: 'status' });
    const submit = btn(t('models.download'), { kind: 'primary', icon: 'download', type: 'submit' });
    const paint = () => {
      const p = picked || {};
      const n = need();
      const gpuTotals = (p.detail?.gpus || []).map((g) => g.total_gib);
      const fitHint = n && gpuTotals.length ? quickFit(n, gpuTotals) : null;
      append(clear(note), [
        n ? h('p', { class: 'small' }, t('dl.dialog_needs', { size: fmtBytes(n, 1) }), p.free != null ? ` · ${t('dl.dialog_free', { free: fmtBytes(p.free, 0) })}` : '') : null,
        n && p.free != null && n > p.free ? h('p', { class: 'note bad', role: 'alert' }, t('dl.dialog_no_space')) : null,
        fitHint && fitHint.verdict !== 'unknown' ? h('p', { class: 'hint' }, t(`dl.dialog_fit_${fitHint.verdict}`, { host: p.detail?.label || '' })) : null]);
      submit.disabled = !p.ok || (groups.length > 0 && !quant);
    };
    const picker = hostRootPicker({ api, hosts: hostList, kinds: groups.length ? ['gguf', 'auto'] : ['hf-home', 'hf-hub', 'auto'], initialHost, onChange: (s) => { picked = s; paint(); } });
    const quantBox = groups.length ? h('fieldset', { class: 'quant-pick' },
      h('legend', {}, t('dl.quant'), infoTip({ kind: 'concept', key: 'field.quant', title: t('dl.quant') })),
      h('div', { class: 'stack-v tight' }, groups.map((g) => h('label', { class: 'quant-row' },
        h('input', { type: 'radio', name: 'quant', value: g.quant, checked: g.quant === quant, onChange: () => { quant = g.quant; paint(); } }),
        h('strong', g.quant), h('span', { class: 'muted num' }, fmtBytes(g.total_bytes, 1)), h('span', { class: 'muted small' }, t('dl.files_count', { count: g.files.length })),
        g.quant === PREFERRED_QUANT ? badge(t('dl.quant_popular'), 'info') : null)))) : null;
    const form = h('form', { class: 'stack-v', onSubmit: async (e) => {
      e.preventDefault();
      const p = picked;
      if (!p?.ok) return;
      submit.disabled = true;
      try {
        const g = groups.find((x) => x.quant === quant);
        const d = await api.startDownload({ repo_id: repo, host_id: p.hostId, root_id: p.rootId, ...(g ? { allow_patterns: g.files } : {}) });
        toast(t('dl.queued_on', { repo, host: p.detail?.label || p.hostId }), { kind: 'ok' });
        onQueued?.(d); dlg.close();
      } catch (err) { toastError(err, t('models.download')); submit.disabled = false; }
    } }, h('p', { class: 'hint' }, groups.length ? t('dl.dialog_intro_gguf') : t('dl.dialog_intro')), picker.el, quantBox, note,
    h('div', { class: 'row gap end' }, btn(t('common.cancel'), { onClick: () => dlg.close() }), submit));
    clear(body).append(form);
    await picker.whenReady; paint();
  })();
  return dlg;
}
