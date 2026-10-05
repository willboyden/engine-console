// One machine on the fleet dashboard: link state, policy mode, each GPU's VRAM split by engine, host RAM split by engine,
// drives and alerts. Pure data work is in fleet.js; this only builds DOM (text nodes only: node text is untrusted).
import { h } from '../dom.js';
import { t } from '../i18n.js';
import { fmtGiB, fmtPct } from '../format.js';
import { gpuSegments, ramSegments, hasData, policyMode, retryCountdown, silentFor, unrestrictedKey, rootfulDocker } from '../fleet.js';
import { badge, btn, linkBadge, modeBadge, tr } from './ui.js';
import { withTip } from './info-tip.js';
import { vramBlock, ramBlock, diskRow, statLabel, shortGpuName } from './fleet-bars.js';

function linkProblem(host, ctx) {
  const l = host.link, s = l.state;
  const kind = s === 'down' || s === 'incompatible' ? 'bad' : s === 'stale' ? 'warn' : 'info';
  const out = [];
  if (l.error) out.push(h('p', { class: `note ${kind}`, role: s === 'down' ? 'status' : null }, l.error));
  else if (s === 'connecting') out.push(h('p', { class: 'note info' }, t('fleet.connecting_text', { host: host.label })));
  else if (s === 'disabled') out.push(h('p', { class: 'note' }, t('fleet.disabled_text')));
  if (s === 'stale') out.push(h('p', { class: 'muted small' }, l.error ? t('fleet.stale_last') : t('fleet.stale_for', { n: silentFor(l) ?? 0 })));
  if (l.retry_in_s != null) out.push(h('p', { class: 'small' }, `${t('fleet.retry_label')} `, h('span', { class: 'retry-count num', dataset: { host: host.id } }, t('fleet.retry_in', { count: retryCountdown(l, ctx.receivedAt, ctx.now) ?? l.retry_in_s }))));
  if (s !== 'disabled' && s !== 'connecting') out.push(btn(t('fleet.reconnect'), { icon: 'restart', size: 'sm', onClick: () => ctx.onReconnect?.(host.id) }));
  return h('div', { class: 'link-problem stack-v tight' }, out);
}

function gpuBlock(g, ctx) {
  const seg = gpuSegments(g, { names: ctx.names, colors: ctx.colors });
  const label = `GPU ${g.index}`;
  return h('div', { class: 'gpu-block' },
    h('div', { class: 'row between gap' }, h('strong', label), h('span', { class: 'muted small ellipsis', title: g.name }, shortGpuName(g.name))),
    seg.unified ? null : h('div', { class: 'row between small' }, statLabel(t('dash.vram'), 'vram_used_gib'), h('span', { class: 'num' }, seg.known || seg.used != null ? `${fmtGiB(seg.used)} / ${fmtGiB(seg.total, 0)}` : '–')),
    vramBlock(seg, label),
    h('div', { class: 'gpu-stats small' },
      h('div', statLabel(t('dash.util'), 'gpu_util_pct'), h('strong', { class: 'num' }, fmtPct(g.util_pct))),
      h('div', statLabel(t('dash.temp'), 'gpu_temp_c'), h('strong', { class: `num ${g.temp_c >= 85 ? 'bad-text' : ''}` }, g.temp_c != null ? `${Math.round(g.temp_c)} °C` : '–')),
      h('div', statLabel(t('dash.power'), 'gpu_power_w'), h('strong', { class: 'num' }, g.power_w != null ? `${Math.round(g.power_w)} W` : '–'))));
}

function alertRow(a) {
  const tip = withTip(h('strong', `${a.level === 'crit' ? t('mem.crit') : t('mem.warn')}: `), { kind: 'concept', key: `alert.${a.code}` }, tr('alert', a.code));
  return h('div', { class: `alert ${a.level === 'crit' ? 'crit' : 'warn'}`, role: a.level === 'crit' ? 'alert' : 'status' }, tip, a.message);
}

/**
 * ctx: {instances, detail, names, colors, receivedAt, now, onReconnect}. `instances` are this machine's instances,
 * `detail` its HostDetail when loaded (drives).
 */
export function hostCard(host, ctx) {
  const state = host.link.state;
  const mode = policyMode(host);
  const node = host.node || {};
  const data = hasData(host);
  const unified = host.gpus.some((g) => g.unified_memory);
  const sub = [node.arch, node.os].filter(Boolean).join(' · ');
  const counts = host.counts || {};
  const ram = data ? ramSegments(host.memory, ctx.instances, ctx.colors, ctx.names) : null;
  const disks = (ctx.detail?.disks || []).filter((d) => d.size_bytes);
  const card = h('article', { class: `card host-card is-${state}`, dataset: { host: host.id }, 'aria-label': t('fleet.card_aria', { host: host.label, state: t(`link.${state}`) }) },
    h('header', { class: 'card-head host-head' },
      h('div', { class: 'host-title' },
        h('h3', {}, h('a', { href: `#/hosts/${encodeURIComponent(host.id)}`, title: t('fleet.open_host', { host: host.label }) }, host.label)),
        sub ? h('span', { class: 'muted small ellipsis host-sub', title: sub }, sub) : null),
      h('div', { class: 'row gap wrap host-badges' }, linkBadge(state), data ? modeBadge(mode) : null, host.remote ? null : badge(t('fleet.this_machine'), 'muted'))),
    h('div', { class: 'card-body' },
      state !== 'up' ? linkProblem(host, ctx) : null,
      unrestrictedKey(host) ? h('p', { class: 'note warn small' }, t('fleet.unrestricted_short'), ' ', h('a', { href: `#/hosts/${encodeURIComponent(host.id)}` }, t('fleet.see_details'))) : null,
      rootfulDocker(host) ? h('p', { class: 'note warn small' }, t('fleet.rootful_short'), ' ', h('a', { href: `#/hosts/${encodeURIComponent(host.id)}` }, t('fleet.see_details'))) : null,
      data ? host.gpus.map((g) => gpuBlock(g, ctx)) : null,
      data && host.gpus.length === 0 ? h('p', { class: 'muted small' }, t('fleet.no_gpus')) : null,
      data && host.memory ? h('div', { class: 'ram-block stack-v tight' },
        h('div', { class: 'row between small' }, statLabel(unified ? t('fleet.ram_shared') : t('mem.host_ram'), 'ram_used_gib'), h('span', { class: 'num' }, `${fmtGiB(host.memory.used_gib, 0)} / ${fmtGiB(host.memory.total_gib, 0)}`)),
        ramBlock(ram, unified ? t('fleet.ram_shared') : t('mem.host_ram'))) : null,
      data && disks.length ? h('div', { class: 'disk-list stack-v tight' }, h('div', { class: 'small' }, statLabel(t('fleet.drives'), 'disk_free_gib')), disks.slice(0, 4).map((d) => diskRow(d, { compact: true }))) : null,
      h('div', { class: 'row between small host-foot' },
        h('span', { class: 'muted' }, t('fleet.engines_count', { running: counts.engines_running ?? 0, total: counts.engines_total ?? 0 })),
        h('span', { class: 'muted' }, t('fleet.models_count', { count: counts.models ?? 0 }))),
      data && (host.alerts || []).length ? h('div', { class: 'stack-v tight host-alerts', role: 'region', 'aria-label': t('dash.alerts') }, host.alerts.map(alertRow)) : null));
  return card;
}
