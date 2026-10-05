// One machine in full: connection, node facts, policy (and how to change it), GPUs with their processes, memory, drives,
// model roots, images, lifecycle scripts and the engines running there. All node-supplied text is untrusted: text nodes only.
import { EcView } from '../components/base.js';
import { h, clear } from '../dom.js';
import { t } from '../i18n.js';
import { items, fmtBytes, fmtGiB, fmtPct, timeAgo, fmtDuration } from '../format.js';
import { normHost, normInstance, listSignature } from '../adapt.js';
import { engineColors, gpuSegments, ramSegments, retryCountdown, silentFor, unrestrictedKey, visibleNotes, rootfulDocker, policyMode, hostSignature } from '../fleet.js';
import { imageAllowed } from '../engines.js';
import { card, badge, btn, errorBox, skeleton, kv, kvTip, factGrid, pageHead, linkBadge, modeBadge, managementBadge, stateBadge, rootKindBadge, tr, stackTable } from '../components/ui.js';
import { withTip, busyWithin } from '../components/info-tip.js';
import { vramBlock, ramBlock, diskRow, statLabel, shortGpuName } from '../components/fleet-bars.js';
import { scriptList, loadScripts } from '../components/script-actions.js';
import { controlBar } from '../components/instance-controls.js';
import { engineList, availableSection } from '../components/instance-rows.js';
import { instanceTitle, scriptLabelMap, splitInstances } from '../controls.js';
import { doInstanceAction } from '../components/instance-actions.js';
import { toast, toastError } from '../components/toast.js';

const enc = encodeURIComponent;
const yesNo = (v) => badge(v ? t('common.yes') : t('common.no'), v ? 'ok' : 'muted');

class EcHost extends EcView {
  setup() {
    this.id = this.params.id;
    this.host = null; this.inst = []; this.scripts = []; this.scriptLabels = new Map(); this.labels = new Map(); this.rx = Date.now();
    this.head = h('div'); this.body = h('div', { class: 'stack-v host-detail' }, skeleton(4));
    this.append(this.head, this.body);
    this.load();
    this.every(4000, () => this.load(true));
    this.every(1000, () => { const el = this.querySelector('.retry-count'); if (el && this.host) el.textContent = t('fleet.retry_in', { count: retryCountdown(this.host.link, this.rx) ?? 0 }); });
  }
  async load(quiet) {
    try {
      const [host, inst, hosts, scripts] = await Promise.all([this.api.host(this.id), this.api.instances(this.id).catch(() => []), this.labels.size ? null : this.api.hosts().catch(() => []), loadScripts().catch(() => [])]);
      if (!this._alive) return;
      if (hosts) for (const x of items(hosts)) this.labels.set(x.id, x.label);
      this.host = normHost(host); this.inst = items(inst).map(normInstance); this.rx = Date.now(); this.err = null;
      // The machine's own script list, plus any the shared list knows about for it (an engine's script must never be unreachable).
      const mine = this.host.scripts || [];
      this.scripts = [...mine, ...scripts.filter((sc) => sc.host_id === this.id && !mine.some((m) => m.name === sc.name))];
      this.scriptLabels = scriptLabelMap(this.scripts);
    } catch (e) { this.err = e; if (!quiet || !this.host) { if (this._alive) { clear(this.head); clear(this.body).append(errorBox(e, () => this.load())); } return; } }
    const sig = hostSignature(this.host, `${listSignature(this.inst)}|${(this.scripts || []).map((s) => s.name + s.status).join()}|${(this.host.disks || []).map((d) => Math.round((d.free_bytes || 0) / 2 ** 30)).join()}|${(this.host.images || []).length}`);
    if (sig === this._sig) return;
    if (busyWithin(this.body) && this._sig) return; // do not repaint under the reader's hands; the next sample retries
    this._sig = sig;
    const d = this.body.querySelector('details.available'); if (d) this._availOpen = d.open;
    this.paint();
  }
  async reconnect() {
    try {
      this.host = { ...this.host, ...normHost(await this.api.reconnect(this.id)) }; this.rx = Date.now();
      toast(t('fleet.reconnecting', { host: this.host.label }), { kind: 'ok', timeout: 2500 });
      this._sig = null; this.paint();
      const id = setTimeout(() => { if (this._alive) this.load(); }, 2200); this.own(() => clearTimeout(id));
    } catch (e) { toastError(e, t('fleet.reconnect')); }
  }

  paint() {
    const x = this.host, state = x.link.state, data = state === 'up' || state === 'stale';
    const mine = this.inst;
    const colors = engineColors(mine), names = new Map(mine.map((i) => [i.id, instanceTitle(i, this.scriptLabels).title]));
    clear(this.head).append(pageHead({ page: 'host', title: x.label, back: { href: '#/', label: t('nav.dashboard') },
      actions: [linkBadge(state), data ? modeBadge(policyMode(x)) : null, state !== 'disabled' && state !== 'connecting' ? btn(t('fleet.reconnect'), { icon: 'restart', size: 'sm', onClick: () => this.reconnect() }) : null] }));
    // What a person came to do first: the engines and their controls, then the hardware. A machine in monitor mode shows its
    // policy right after the connection, because that is why nothing can be changed there.
    const monitor = policyMode(x) !== 'manage';
    clear(this.body).append(this.linkCard(x), ...(data ? [monitor ? this.policyCard(x) : null, this.enginesCard(x, mine), this.scriptsCard(x), ...x.gpus.map((g) => this.gpuCard(g, names, colors)), this.memoryCard(x, mine, colors),
      this.drivesCard(x), this.rootsCard(x), this.imagesCard(x), this.factsCard(x), monitor ? null : this.policyCard(x)].filter(Boolean) : [h('p', { class: 'hint' }, t('host.no_data'))]));
  }

  // ---- connection ----
  linkCard(x) {
    const l = x.link, kind = l.state === 'down' || l.state === 'incompatible' ? 'bad' : l.state === 'stale' ? 'warn' : 'info';
    return card(t('host.link'), h('div', { class: 'stack-v' },
      l.error ? h('p', { class: `note ${kind}` }, l.error) : l.state === 'up' ? h('p', { class: 'muted' }, t('host.link_up_text')) : null,
      l.retry_in_s != null ? h('p', {}, `${t('fleet.retry_label')} `, h('span', { class: 'retry-count num', 'aria-live': 'off' }, t('fleet.retry_in', { count: retryCountdown(l, this.rx) ?? l.retry_in_s }))) : null,
      unrestrictedKey(x) ? h('div', { class: 'note warn' }, h('strong', t('host.unrestricted_title')), ' ', t('host.unrestricted_text')) : null,
      ...visibleNotes(x).map((n) => h('p', { class: 'note info small' }, n)),
      factGrid([[t('host.f_state'), linkBadge(l.state, { tip: false })],
        l.rtt_ms != null ? [t('host.f_rtt'), `${l.rtt_ms} ms`] : null,
        l.skew_s != null ? [t('host.f_skew'), `${l.skew_s.toFixed(1)} s`] : null,
        l.last_seen != null ? [t('host.f_last_seen'), l.state === 'stale' ? t('host.silent_for', { n: silentFor(l) }) : timeAgo(l.last_seen)] : null,
        l.connected_since != null ? [t('host.f_connected'), `${fmtDuration(Date.now() / 1000 - l.connected_since)}`] : null,
        x.remote ? [t('host.f_address'), (x.addresses || []).join(', ') || '–'] : null])));
  }

  // ---- node facts ----
  factsCard(x) {
    const n = x.node || {}, d = x.docker;
    return card(t('host.node'), h('div', { class: 'stack-v' },
      rootfulDocker(x) ? h('div', { class: 'note warn' }, h('strong', t('host.rootful_title')), ' ', t('host.rootful_text')) : null,
      h('dl', { class: 'facts two-col' },
        kv(t('host.f_hostname'), n.hostname || '–'), kv(t('host.f_os'), n.os || '–'), kv(t('host.f_arch'), n.arch || '–'), kv(t('host.f_kernel'), n.kernel || '–'),
        kv(t('host.f_cpu'), n.cpu_model ? `${n.cpu_model}${n.cpu_cores ? ` (${n.cpu_cores} ${t('host.cores')}${n.cpu_threads ? `, ${n.cpu_threads} ${t('host.threads')}` : ''})` : ''}` : '–'),
        kv(t('host.f_driver'), n.driver || '–'), kv(t('host.f_cuda'), n.cuda || '–'), kv(t('host.f_python'), n.python || '–'), kv(t('host.f_node'), n.version || '–'),
        kv(t('host.f_gpu_source'), n.gpu_source || '–'),
        x.cpu ? kv(t('host.f_load'), `${x.cpu.load1 ?? '–'} / ${x.cpu.load5 ?? '–'} / ${x.cpu.load15 ?? '–'}`) : null,
        d ? kv(t('host.f_docker'), d.available ? [d.version || '', d.context ? ` · ${t('host.docker_context', { ctx: d.context })}` : '', ' ', badge(d.rootless ? t('host.rootless') : t('host.rootful'), d.rootless ? 'ok' : 'warn'), d.nvidia_runtime === false ? [' ', badge(t('host.no_nvidia_runtime'), 'warn')] : null] : [badge(t('host.docker_missing'), 'bad'), d.error ? ` ${d.error}` : '']) : null)));
  }

  // ---- policy ----
  policyCard(x) {
    const p = x.policy;
    if (!p) return card(t('host.policy'), h('p', { class: 'muted' }, t('host.no_policy')));
    const manage = p.mode === 'manage';
    return card(t('host.policy'), h('div', { class: 'stack-v' },
      h('div', { class: 'row gap wrap' }, modeBadge(p.mode), h('strong', manage ? t('host.mode_manage_title') : t('host.mode_monitor_title'))),
      h('p', {}, manage ? t('host.mode_manage_text') : t('host.mode_monitor_text')),
      p.error ? h('p', { class: 'note bad' }, t('host.policy_error', { error: p.error })) : null,
      !manage ? h('div', { class: 'note info' }, h('strong', t('host.howto_title')), ' ', t('host.howto_text', { path: p.path || t('host.policy_file') })) : h('p', { class: 'hint' }, t('host.policy_readonly', { path: p.path || t('host.policy_file') })),
      h('dl', { class: 'facts' },
        kv(t('host.p_images'), (p.engine_images || []).length ? h('span', { class: 'row gap wrap' }, p.engine_images.map((i) => h('code', { class: 'chip-code' }, i))) : t('host.p_none')),
        kv(t('host.p_gateway'), p.gateway_image ? h('code', p.gateway_image) : '–'), kv(t('host.p_network'), p.engine_network || '–'),
        kv(t('host.p_ports'), (p.port_range || []).length === 2 ? `${p.port_range[0]}–${p.port_range[1]}` : '–'),
        kv(t('host.p_pull'), h('span', { class: 'row gap' }, yesNo(p.allow_pull), h('span', { class: 'muted small' }, t('host.p_pull_help')))),
        kv(t('host.p_delete'), h('span', { class: 'row gap' }, yesNo(p.allow_delete), h('span', { class: 'muted small' }, t('host.p_delete_help')))),
        kv(t('host.p_ipc'), h('span', { class: 'row gap' }, yesNo(p.allow_ipc_host), h('span', { class: 'muted small' }, t('host.p_ipc_help')))),
        kv(t('host.p_external'), h('span', { class: 'stack-v tight' }, badge(p.external_control === 'stop_start' ? t('host.p_external_stop_start') : t('host.p_external_none'), p.external_control === 'stop_start' ? 'info' : 'muted'), h('span', { class: 'muted small' }, p.external_control === 'stop_start' ? t('host.p_external_stop_start_help') : t('host.p_external_none_help')))),
        p.max_shm_gib != null ? kv(t('host.p_shm'), fmtGiB(p.max_shm_gib, 0)) : null)));
  }

  // ---- GPUs ----
  gpuCard(g, names, colors) {
    const seg = gpuSegments(g, { names, colors });
    const label = `GPU ${g.index}`;
    const procs = g.procs || [];
    return card(`${label} · ${shortGpuName(g.name)}`, h('div', { class: 'stack-v' },
      seg.unified ? null : h('div', { class: 'row between' }, statLabel(t('dash.vram'), 'vram_used_gib'), h('span', { class: 'num' }, seg.used != null ? `${fmtGiB(seg.used)} / ${fmtGiB(seg.total, 0)}` : '–')),
      vramBlock(seg, label),
      h('div', { class: 'gpu-stats' },
        h('div', statLabel(t('dash.util'), 'gpu_util_pct'), h('strong', { class: 'num' }, fmtPct(g.util_pct))),
        h('div', statLabel(t('dash.temp'), 'gpu_temp_c'), h('strong', { class: `num ${g.temp_c >= 85 ? 'bad-text' : ''}` }, g.temp_c != null ? `${Math.round(g.temp_c)} °C` : '–')),
        h('div', statLabel(t('dash.power'), 'gpu_power_w'), h('strong', { class: 'num' }, g.power_w != null ? `${Math.round(g.power_w)}${g.power_limit_w ? ` / ${Math.round(g.power_limit_w)}` : ''} W` : '–'))),
      h('p', { class: 'muted small' }, [g.compute_capability ? t('host.cc', { cc: g.compute_capability }) : null, g.uuid ? ` · ${g.uuid}` : null].filter(Boolean).join('')),
      procs.length ? h('div', { class: 'tablewrap' }, h('table', { class: 'compact' },
        h('thead', h('tr', ['host.proc_process', 'host.proc_engine', 'host.proc_used'].map((k) => h('th', { scope: 'col' }, t(k))))),
        h('tbody', procs.map((p) => { const inst = this.inst.find((i) => i.id === p.instance_id); return h('tr', {},
          h('td', {}, h('code', p.name || `pid ${p.pid}`), p.container ? h('div', { class: 'muted small' }, p.container) : null),
          h('td', {}, inst ? h('a', { href: `#/instances/${enc(inst.id)}` }, instanceTitle(inst, this.scriptLabels).title) : h('span', { class: 'muted' }, t('host.proc_other'))),
          h('td', { class: 'num' }, p.used_gib != null ? fmtGiB(p.used_gib) : t('fleet.shared_pool'))); })))) : h('p', { class: 'muted small' }, t('host.no_procs'))));
  }

  // ---- memory ----
  memoryCard(x, mine, colors) {
    const m = x.memory;
    if (!m) return card(t('mem.system'), h('p', { class: 'hint' }, t('mem.unavailable')));
    const unified = x.gpus.some((g) => g.unified_memory);
    const ram = ramSegments(m, mine, colors);
    return card(unified ? t('fleet.ram_shared') : t('mem.system'), h('div', { class: 'stack-v' },
      unified ? h('p', { class: 'note info small' }, t('host.unified_text')) : null,
      h('div', { class: 'row between' }, h('span', { class: 'big-num num' }, fmtGiB(m.used_gib, 0)), h('span', { class: 'muted small' }, t('mem.used_of', { total: fmtGiB(m.total_gib, 0) }))),
      ramBlock(ram, t('mem.host_ram')),
      h('dl', { class: 'facts three-col' },
        kvTip(t('mem.available'), { kind: 'metric', key: 'ram_available_gib' }, fmtGiB(m.available_gib)),
        kvTip(t('mem.shmem'), { kind: 'metric', key: 'shmem_gib' }, fmtGiB(m.shmem_gib)),
        kvTip(t('mem.swap'), { kind: 'metric', key: 'swap_used_gib' }, t('mem.swap_used', { used: fmtGiB(m.swap_used_gib, 0), total: fmtGiB(m.swap_total_gib, 0) }))),
      (x.alerts || []).length ? h('div', { class: 'stack-v tight' }, x.alerts.map((a) => h('div', { class: `alert ${a.level === 'crit' ? 'crit' : 'warn'}` }, withTip(h('strong', `${a.level === 'crit' ? t('mem.crit') : t('mem.warn')}: `), { kind: 'concept', key: `alert.${a.code}` }, tr('alert', a.code)), a.message))) : null));
  }

  // ---- drives, roots, images ----
  drivesCard(x) {
    const ds = x.disks || [];
    if (!ds.length) return card(t('host.drives'), h('p', { class: 'muted' }, t('host.no_drives')));
    return card(t('host.drives'), stackTable(h('div', { class: 'tablewrap', 'data-stack-key': 'host-drives' }, h('table',
      h('thead', h('tr', ['host.d_drive', 'host.d_type', 'host.d_use', 'host.d_size', 'host.d_roots'].map((k) => h('th', { scope: 'col' }, t(k))))),
      h('tbody', ds.map((m) => h('tr', {},
        h('td', {}, h('strong', m.label), h('div', { class: 'muted small' }, m.mount), m.model ? h('div', { class: 'muted small' }, m.model) : null),
        h('td', {}, m.fstype || '–', ' ', m.remote ? badge(t('fleet.disk_shared'), 'info', t('fleet.disk_shared_tip', { host: m.remote_host_id || m.remote_host || '?' })) : null, m.readonly ? badge(t('fleet.disk_readonly'), 'muted') : null),
        h('td', { class: 'wide' }, diskRow(m, { compact: true, hideLabel: true, hideExtras: true })),
        h('td', { class: 'num' }, fmtBytes(m.size_bytes, 1)),
        h('td', {}, (m.roots || []).length ? h('span', { class: 'row gap wrap' }, m.roots.map((r) => h('code', { class: 'chip-code' }, r))) : '–'))))))));
  }
  rootsCard(x) {
    const rs = x.roots || [];
    if (!rs.length) return card(t('host.roots'), h('p', { class: 'muted' }, t('host.no_roots')));
    return card(t('host.roots'), stackTable(h('div', { class: 'tablewrap', 'data-stack-key': 'host-roots' }, h('table',
      h('thead', h('tr', ['host.r_folder', 'host.r_kind', 'host.r_writable', 'host.r_shared', 'host.r_visible'].map((k) => h('th', { scope: 'col' }, t(k))))),
      h('tbody', rs.map((r) => h('tr', {},
        h('td', {}, h('strong', r.label || r.id), h('div', { class: 'muted small' }, h('code', r.path)), r.mount ? h('div', { class: 'muted small' }, r.mount.label) : null),
        h('td', {}, rootKindBadge(r.kind)), h('td', {}, yesNo(r.writable)),
        h('td', {}, r.shared_from ? badge(t('host.r_from', { host: this.labels.get(r.shared_from) || r.shared_from }), 'info') : '–'),
        h('td', {}, h('span', { class: 'row gap wrap' }, (r.visible_on || []).map((id) => h('a', { class: 'chip-link', href: `#/hosts/${enc(id)}` }, this.labels.get(id) || id)))))))))));
  }
  imagesCard(x) {
    const im = x.images || [], pats = x.policy?.engine_images || [];
    if (!im.length) return card(t('host.images'), h('p', { class: 'muted' }, t('host.no_images')));
    return card(t('host.images'), stackTable(h('div', { class: 'tablewrap', 'data-stack-key': 'host-images' }, h('table',
      h('thead', h('tr', ['host.i_image', 'host.i_arch', 'host.i_size', 'host.i_created', 'host.i_policy'].map((k) => h('th', { scope: 'col' }, t(k))))),
      h('tbody', im.map((i) => h('tr', {}, h('td', {}, h('code', i.ref)), h('td', i.arch || '–'), h('td', { class: 'num' }, fmtBytes(i.size_bytes, 1)), h('td', i.created ? timeAgo(i.created) : '–'),
        h('td', {}, imageAllowed(pats, i.ref) ? badge(t('host.i_allowed'), 'ok') : badge(t('host.i_not_listed'), 'muted')))))))));
  }

  // ---- scripts and engines ----
  scriptsCard(x) {
    const ss = this.scripts || [];
    return card(t('host.scripts'), ss.length ? h('div', { class: 'stack-v' }, h('p', { class: 'hint' }, t('host.scripts_intro')), scriptList(ss, { hostLabel: x.label, onChange: () => { this._sig = null; this.load(); } })) : h('p', { class: 'muted' }, t('host.no_scripts')));
  }
  /** Running and attention-needing engines as rows; stopped ones, which can be many, in a collapsible list below. */
  enginesCard(x, mine) {
    const after = () => { this._sig = null; this.load(); };
    if (!mine.length) return card(t('host.engines'), h('p', { class: 'muted' }, t('inst.empty_hint')), btn(t('nav.launch'), { kind: 'primary', size: 'sm', icon: 'launch', onClick: () => { location.hash = `#/launch?host=${enc(x.id)}`; } }));
    const { active, available } = splitInstances(mine);
    const opts = { labels: this.scriptLabels, scripts: this.scripts, hostLabel: x.label, onChange: after, onAction: (id, i) => doInstanceAction(i, id, { after }) };
    // Engines the console did not start sit in their own group under the machine's one explanation of why they are watch-only,
    // so the note never reads as if it were about the console's own engines above it.
    const outside = active.filter((i) => i.management === 'external'), own = active.filter((i) => i.management !== 'external');
    const shared = [...new Set(outside.filter((i) => i.controls?.reason).map((i) => i.controls.reason))];
    return card(t('host.engines'), h('div', { class: 'stack-v' },
      own.length ? engineList(own, opts) : null,
      outside.length ? h('section', { class: 'engine-group stack-v tight', 'aria-label': t('host.external_group') },
        h('h3', { class: 'group-title' }, withTip(h('span', t('host.external_group')), { kind: 'concept', key: 'management.external' }, t('host.external_group')), badge(String(outside.length), 'muted')),
        ...shared.map((r) => h('p', { class: 'note info small' }, r)),
        engineList(outside, opts)) : null,
      active.length ? null : h('p', { class: 'muted' }, t('inst.none_running_here')),
      availableSection(available, { ...opts, open: this._availOpen ?? !active.length })));
  }
}
customElements.define('ec-host', EcHost);
