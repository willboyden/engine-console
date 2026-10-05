// Advanced launch form: pick a machine, a model that machine can read, an engine that loads that model's format, an image the
// machine allows and its GPUs, then tune the settings (shown by level, each with an explanation). The memory estimate and the
// launch checks run against the chosen machine. Beginners are sent to the guided wizard instead (see app.js).
import { EcView } from '../components/base.js';
import { h, clear, append, copyText, download, icon } from '../dom.js';
import { t } from '../i18n.js';
import { items, debounce, fmtGiB } from '../format.js';
import { memoryFingerprint, normalizePresets, pruneValues, validateAll, toCliArgs, cliString, changedCount, isNonCli, toEnvLines, splitRedacted, levelOf, levelRank, nextLevel } from '../param-form.js';
import { describeError } from '../errors.js';
import { compatLevel, normalizeReport } from '../fit-format.js';
import { normHost } from '../adapt.js';
import { enginesForFormat, engineUsable, imageChoices, defaultImage, pickEngine } from '../engines.js';
import { manageBlock, canManage } from '../fleet.js';
import { resolveOnHost } from '../library.js';
import { pickGpus } from '../wizard.js';
import { refFromQuery, refBody, refLabel, refEquals } from '../model-ref.js';
import { defaultLevel, isBeginner } from '../beginner.js';
import { renderParamForm } from '../components/param-field.js';
import { fitMeter } from '../components/fit-meter.js';
import { btn, card, field, select, errorBox, skeleton, badge, pageHead, formatBadge, reasonText, tr } from '../components/ui.js';
import { infoTip, withTip } from '../components/info-tip.js';
import { openModal, confirmDialog } from '../components/dialog.js';
import { toast, toastError } from '../components/toast.js';
import { checkList } from '../components/check-list.js';

const CUSTOM = '__custom__';
const shortName = (n) => String(n || '').replace(/^NVIDIA\s+/i, '');

class EcLaunch extends EcView {
  setup() {
    const q = this.query || {};
    this.st = { hostId: q.host || '', mkey: q.mkey || '', custom: false, customRepo: '', ref: refFromQuery(q), engine: q.engine || '', image: '', name: '', gpus: new Set(), values: {}, level: defaultLevel(), search: '', profileId: '', concurrency: 1 };
    this.needSecrets = new Set(); // keys whose saved value is "[set]": the server never returns secrets, so the user must retype them
    this.catalogs = new Map(); this.hostDetails = new Map(); this.models = [];
    this.report = null; this.fitErr = null; this.fitBusy = false; this.lastFp = null; this.fitSeq = 0; this.pf = null;
    this.scheduleFit = debounce(() => this.runFit(), 350);
    this.own(() => this.scheduleFit.cancel());
    this.formHost = h('div', { class: 'stack-v' }, skeleton(6));
    this.fitHost = h('div'); this.checksHost = h('div');
    this.append(pageHead({ page: 'launch', title: t('nav.launch'), actions: [h('a', { class: 'btn', href: '#/launch/wizard' }, icon('wand', 16), h('span', t('launch.use_wizard')))] }),
      h('div', { class: 'launch-grid' }, this.formHost, h('aside', { class: 'launch-side' }, card(t('fit.title'), this.fitHost, null), card(t('launch.checks'), this.checksHost, null), this.actionsHost = h('div', { class: 'card' }))));
    this.init(q);
  }

  // ---------- loading ----------
  async init(q) {
    try {
      const [engines, hosts, profiles, settings] = await Promise.all([this.api.engines(), this.api.hosts(), this.api.profiles().catch(() => []), this.api.settings().catch(() => null)]);
      if (!this._alive) return;
      Object.assign(this, { engines: items(engines), hosts: items(hosts).map(normHost), profiles: items(profiles), settings });
      let first = null;
      if (q.from) {
        const i = await this.api.instance(q.from);
        const sp = splitRedacted(i.params || {}); sp.redacted.forEach((k) => this.needSecrets.add(k));
        Object.assign(this.st, { hostId: i.host_id || 'local', engine: i.engine, values: sp.values, gpus: new Set(i.gpu_ids || []), name: i.name ? `${i.name}-2` : '', ref: i.model || (i.repo_id ? { format: 'hf', repo_id: i.repo_id } : null), mkey: '' });
        first = i.gpu_ids?.length ? new Set(i.gpu_ids) : null;
      }
      const start = this.hosts.find((x) => x.id === this.st.hostId) || this.hosts.find((x) => canManage(x) && x.id === 'local') || this.hosts.find((x) => canManage(x)) || this.hosts[0];
      if (!start) { clear(this.formHost).append(errorBox(new Error(t('launch.no_hosts')))); return; }
      await this.pickHost(start.id, { keepGpus: first });
    } catch (e) { clear(this.formHost).append(errorBox(e, () => this.init(q))); }
  }
  async pickHost(id, { keepGpus } = {}) {
    const st = this.st;
    st.hostId = id;
    clear(this.formHost).append(skeleton(4));
    try {
      const [detail, lib] = await Promise.all([this.hostDetails.has(id) ? this.hostDetails.get(id) : this.api.host(id).then(normHost), this.api.library({ host: id }).catch(() => [])]);
      if (!this._alive || st.hostId !== id) return;
      this.hostDetails.set(id, detail);
      this.models = items(lib).filter((m) => resolveOnHost(m, id, this.hostDetails));
      const valid = (keepGpus ? [...keepGpus] : [...st.gpus]).filter((i) => detail.gpus.some((g) => g.index === i));
      st.gpus = new Set(valid.length ? valid : pickGpus(detail.gpus, 1));
      this.syncModel(); this.syncEngine();
      await this.loadCatalog();
      this.renderForm(); this.renderActions(); this.renderChecks(); this.scheduleFit();
    } catch (e) { if (this._alive) clear(this.formHost).append(errorBox(e, () => this.pickHost(id))); }
  }
  host() { return this.hostDetails.get(this.st.hostId) || null; }
  currentModel() { return this.models.find((m) => m.key === this.st.mkey) || null; }
  currentFormat() { return this.st.custom ? 'hf' : this.currentModel()?.format || null; }
  currentRef() {
    const st = this.st;
    if (st.custom) return st.customRepo.trim() ? { format: 'hf', repo_id: st.customRepo.trim() } : null;
    const m = this.currentModel();
    return m ? resolveOnHost(m, st.hostId, this.hostDetails)?.ref ?? null : null;
  }
  /** Keep the model choice valid for the machine: a library model it can read, or (hf) a typed repo id. */
  syncModel() {
    const st = this.st;
    let m = st.mkey ? this.models.find((x) => x.key === st.mkey) : null;
    if (!m && st.ref) m = this.models.find((x) => { const r = resolveOnHost(x, st.hostId, this.hostDetails); return r && refEquals(r.ref, st.ref); });
    if (m) { st.mkey = m.key; st.custom = false; st.ref = null; }
    else if (st.ref?.format === 'hf' && st.ref.repo_id) { st.custom = true; st.customRepo = st.ref.repo_id; st.mkey = ''; }
    else if (!st.custom) st.mkey = '';
  }
  syncEngine() {
    const st = this.st, fmt = this.currentFormat(), host = this.host();
    const list = fmt ? enginesForFormat(this.engines, fmt) : this.engines;
    if (!list.some((e) => e.id === st.engine)) st.engine = (pickEngine(this.engines, fmt || 'hf', host) || list[0])?.id || '';
  }
  async loadCatalog() {
    const e = this.st.engine;
    if (!e) { this.specs = []; this.presets = []; return; }
    if (!this.catalogs.has(e)) this.catalogs.set(e, await this.api.params(e));
    // Engine images are pinned per machine (contract v1.1/v2); an `@image` param, if a catalog still lists one, is not offered.
    this.specs = this.catalogs.get(e).filter((s) => s.flag !== '@image');
    this.presets = normalizePresets(this.engines.find((x) => x.id === e)?.presets);
    this.st.values = pruneValues(this.specs, this.st.values);
  }
  async setEngine(e) {
    this.st.engine = e; this.st.profileId = ''; this.st.image = '';
    try { await this.loadCatalog(); } catch (err) { toastError(err); return; }
    this.renderForm(); this.renderActions(); this.scheduleFit();
  }
  fitParams() { return pruneValues(this.specs.filter((s) => !isNonCli(s)), this.st.values); }
  onParam(key, value) {
    if (value === undefined) delete this.st.values[key]; else this.st.values[key] = value;
    if (this.needSecrets.delete(key)) this.paintSecretsNote();
    if (memoryFingerprint(this.specs, this.st.values) !== this.lastFp) this.scheduleFit();
    this.paintCommand();
  }
  applyValues(values) { const sp = splitRedacted(values); this.needSecrets = new Set(sp.redacted.filter((k) => this.specs.some((s) => s.key === k))); this.st.values = pruneValues(this.specs, sp.values); this.renderParams(); this.paintSecretsNote(); this.scheduleFit(); this.paintCommand(); }

  // ---------- form ----------
  hostLabel(x) {
    const b = manageBlock(x);
    if (b) return `${x.label} (${x.link.state !== 'up' ? t(`link.${x.link.state}`) : t('mode.monitor')})`;
    const gpus = x.gpus || [];
    if (!gpus.length) return x.label;
    const free = gpus.reduce((a, g) => a + (g.free_gib ?? 0), 0);
    return `${x.label} · ${gpus.length} × ${shortName(gpus[0].name)} · ${t('launch.free_total', { free: fmtGiB(gpus[0].unified_memory ? x.memory?.available_gib ?? free : free, 0) })}`;
  }
  defaultImageText(eng, host) {
    const s = this.settings;
    return s?.host_image_pins?.[host.id]?.[eng.id] || s?.image_pins?.[eng.id] || defaultImage(eng, host.node?.arch) || t('launch.image_pinned');
  }
  renderForm() {
    const st = this.st, host = this.host();
    const blocked = manageBlock(host);
    // Machine
    const hostSel = h('select', { id: 'launch-host', onChange: (e) => { this.pickHost(e.target.value).then(() => document.getElementById('launch-host')?.focus()); } },
      this.hosts.map((x) => h('option', { value: x.id, selected: x.id === st.hostId, disabled: !!manageBlock(x) && x.id !== st.hostId }, this.hostLabel(x))));
    // GPUs of that machine, with free memory
    const gpus = h('div', { class: 'row gap wrap', role: 'group', 'aria-label': t('launch.gpus') }, (host.gpus || []).map((g) =>
      h('label', { class: 'check' }, h('input', { type: 'checkbox', checked: st.gpus.has(g.index), onChange: (e) => { e.target.checked ? st.gpus.add(g.index) : st.gpus.delete(g.index); this.scheduleFit(); this.renderActions(); } }),
        h('span', `GPU ${g.index}`), h('small', { class: 'muted' }, g.unified_memory ? `${shortName(g.name)} · ${t('launch.unified_free', { free: fmtGiB(host.memory?.available_gib ?? 0, 0) })}` : `${shortName(g.name)} · ${fmtGiB(g.free_gib, 0)} ${t('launch.free')}`))));
    // Model: what this machine can read, grouped by format; or a typed Hugging Face repo id
    const modelSel = h('select', { id: 'launch-model', onChange: (e) => this.changeModel(e.target.value) }, h('option', { value: '', selected: !st.mkey && !st.custom }, this.models.length ? t('launch.choose_model') : t('launch.no_models_here')));
    for (const f of ['hf', 'gguf', 'ollama', 'plain']) {
      const group = this.models.filter((m) => m.format === f);
      if (group.length) modelSel.append(h('optgroup', { label: t(`format.${f}`) }, group.map((m) => h('option', { value: m.key, selected: m.key === st.mkey }, `${m.name}${m.quant && m.format !== 'hf' ? ` · ${m.quant}` : ''} · ${fmtGiB(m.size_bytes / 2 ** 30, 1)}`))));
    }
    modelSel.append(h('option', { value: CUSTOM, selected: st.custom }, t('launch.model_custom')));
    const customIn = st.custom ? h('input', { type: 'text', value: st.customRepo, placeholder: 'org/model-name', autocomplete: 'off', spellcheck: 'false', 'aria-label': t('launch.model_custom'), onInput: (e) => { st.customRepo = e.target.value; this.syncEngine(); this.scheduleFit(); this.renderActions(); } }) : null;
    // Engine: only those that load this model's format; one the machine may not run is shown with the reason
    const fmt = this.currentFormat();
    const engList = fmt ? enginesForFormat(this.engines, fmt) : this.engines;
    const eng = h('div', { class: 'seg-control', role: 'radiogroup', 'aria-label': t('launch.engine') }, engList.map((e) => {
      const u = engineUsable(e, host); const on = e.id === st.engine;
      return h('button', { type: 'button', role: 'radio', 'aria-checked': String(on), class: on ? 'on' : '', 'aria-disabled': u.ok ? null : 'true', title: u.ok ? null : reasonText(u.reason), onClick: u.ok ? () => this.setEngine(e.id) : (ev) => ev.preventDefault() }, e.display_name || e.id);
    }));
    const engObj = this.engines.find((e) => e.id === st.engine);
    const imgs = engObj ? imageChoices(engObj, host) : [];
    const imageSel = select([{ value: '', label: t('launch.image_default', { image: engObj ? this.defaultImageText(engObj, host) : '–' }) }, ...imgs.map((i) => ({ value: i.ref, label: i.ref }))], st.image, (v) => { st.image = v; this.renderChecks(); });
    const profSel = select([{ value: '', label: t('launch.no_profile') }, ...this.profiles.filter((p) => !p.engine || p.engine === st.engine).map((p) => ({ value: p.id, label: p.name }))], st.profileId, (id) => {
      st.profileId = id; const p = this.profiles.find((x) => x.id === id); if (p) { clear(this.serverNote); if (p.repo_id && !this.currentRef()) { st.custom = true; st.customRepo = p.repo_id; this.renderForm(); } this.applyValues({ ...p.params }); }
    });
    const presetBar = h('div', { class: 'row gap wrap', role: 'group', 'aria-label': t('launch.presets') }, this.presets.map((p) => withTip(btn(p.label, { size: 'sm', onClick: () => { st.profileId = ''; this.applyValues({ ...p.params }); toast(t('launch.preset_applied', { name: p.label }), { kind: 'ok', timeout: 2000 }); } }), { kind: 'preset', key: p.name, engine: st.engine }, p.label)),
      btn(t('launch.clear_all'), { size: 'sm', kind: 'ghost', onClick: () => this.applyValues({}) }));
    const search = h('input', { type: 'search', placeholder: t('params.search'), 'aria-label': t('params.search'), value: st.search, onInput: debounce((e) => { st.search = e.target.value; this.renderParams(); }, 150) });
    this.paramsHost = h('div', { class: 'params' }); this.levelHost = h('div');
    this.secretsNote = h('div', { role: 'status' }); this.serverNote = h('div', { role: 'alert' });
    clear(this.formHost).append(
      card(t('launch.target'), h('div', { class: 'stack-v' },
        blocked ? h('p', { class: 'note warn', role: 'status' }, reasonText(blocked)) : null,
        field(t('launch.machine'), hostSel, null, { kind: 'concept', key: 'field.host' }),
        h('div', { class: 'field' }, h('div', { class: 'label-row' }, h('label', t('launch.gpus')), infoTip({ kind: 'concept', key: 'field.gpus', title: t('launch.gpus') })), gpus),
        field(t('launch.model'), modelSel, this.models.length ? t('launch.model_hint_v2') : t('launch.model_hint_none'), { kind: 'concept', key: 'field.model' }), customIn,
        this.currentModel() ? h('div', { class: 'row gap wrap small' }, formatBadge(this.currentModel().format), this.currentModel().warnings?.length ? badge(t('launch.model_warn'), 'warn', this.currentModel().warnings.join(' ')) : null) : null,
        h('div', { class: 'field' }, h('div', { class: 'label-row' }, h('label', t('launch.engine')), infoTip({ kind: 'concept', key: 'field.engine', title: t('launch.engine') })), eng),
        field(t('launch.image'), imageSel, t('launch.image_hint_v2'), { kind: 'concept', key: 'field.image' }),
        h('div', { class: 'row gap wrap bottom' },
          field(t('launch.concurrency'), h('input', { type: 'number', min: 1, max: 4096, step: 1, value: String(st.concurrency), onChange: (e) => { st.concurrency = Math.max(1, Math.min(4096, parseInt(e.target.value, 10) || 1)); e.target.value = String(st.concurrency); this.scheduleFit(); } }), t('launch.concurrency_hint'), { kind: 'concept', key: 'field.concurrency' }),
          field(t('launch.name'), h('input', { type: 'text', value: st.name, placeholder: t('launch.name_ph'), onInput: (e) => { st.name = e.target.value.trim(); } }), null, { kind: 'concept', key: 'field.name' }),
          field(t('launch.profile'), profSel, null, { kind: 'concept', key: 'field.profile', inline: false }), btn(t('launch.profile_import'), { size: 'sm', onClick: () => this.importProfile() }), st.profileId ? btn(t('launch.profile_export'), { size: 'sm', onClick: () => this.exportProfile() }) : null))),
      card(t('launch.params'), h('div', { class: 'stack-v' }, this.serverNote, this.secretsNote, presetBar, h('div', { class: 'row gap wrap between' }, h('div', { class: 'grow' }, search)), this.levelHost, this.paramsHost)));
    this.renderParams();
  }
  changeModel(v) {
    const st = this.st;
    if (v === CUSTOM) { st.custom = true; st.mkey = ''; } else { st.custom = false; st.mkey = v; st.ref = null; }
    this.syncEngine();
    this.loadCatalog().then(() => { this.renderForm(); this.renderActions(); this.scheduleFit(); document.getElementById(v === CUSTOM ? 'launch-model' : 'launch-model')?.focus(); }).catch((e) => toastError(e));
  }
  renderParams() {
    const st = this.st;
    renderParamForm(this.paramsHost, { specs: this.specs, values: st.values, onChange: (k, v) => this.onParam(k, v), search: st.search, level: st.level, engine: st.engine, redacted: this.needSecrets });
    const hidden = this.specs.filter((s) => levelRank(levelOf(s)) > levelRank(st.level) && !(s.key in st.values)).length;
    clear(this.levelHost).append(h('div', { class: 'row gap wrap level-bar' },
      h('span', { class: 'muted small' }, t('launch.level_showing', { level: t(`params.level_name_${st.level}`) })),
      hidden && st.level !== 'expert' ? btn(t('launch.show_more', { n: hidden }), { size: 'sm', icon: 'plus', onClick: () => { st.level = nextLevel(st.level); this.renderParams(); } }) : null,
      st.level !== 'basic' ? btn(t('launch.show_fewer'), { size: 'sm', kind: 'ghost', onClick: () => { st.level = 'basic'; this.renderParams(); } }) : null));
    this.paintCommand();
  }
  paintSecretsNote() {
    if (!this.secretsNote) return;
    const labels = [...this.needSecrets].map((k) => this.specs.find((s) => s.key === k)?.label || k).join(', ');
    clear(this.secretsNote).append(...(this.needSecrets.size ? [h('p', { class: 'note warn' }, t('launch.secrets_note', { keys: labels }))] : []));
    this.paintGate();
  }

  // ---------- side column ----------
  renderActions() {
    this.launchBtn = btn(t('launch.launch'), { kind: 'primary', icon: 'launch', onClick: () => this.launch() });
    this.cmdOut = h('pre', { class: 'cmd', tabindex: '0', 'aria-label': t('launch.flags') });
    clear(this.actionsHost).append(h('div', { class: 'card-body stack-v' },
      h('div', { class: 'row gap wrap' }, this.launchBtn,
        btn(t('launch.save_profile'), { onClick: () => this.saveProfile() }),
        btn(t('launch.copy_cmd'), { icon: 'copy', onClick: async () => { await copyText(this.cmdText()); toast(t('common.copied'), { kind: 'ok', timeout: 1800 }); } })),
      h('div', { class: 'muted small' }, t('launch.modified', { n: changedCount(this.st.values) })),
      this.cmdOut));
    this.paintCommand(); this.paintGate();
  }
  cmdText() { const env = toEnvLines(this.specs, this.st.values); return [...env, cliString(['--model', refLabel(this.currentRef()) || '<model>', ...toCliArgs(this.specs, this.st.values)])].join('\n'); }
  paintCommand() { if (this.cmdOut) this.cmdOut.textContent = this.cmdText(); const n = this.actionsHost?.querySelector('.muted.small'); if (n) n.textContent = t('launch.modified', { n: changedCount(this.st.values) }); this.paintGate(); }
  gate() {
    const st = this.st, host = this.host();
    const b = manageBlock(host);
    if (b) return reasonText(b);
    if (!this.currentRef()) return t('launch.need_model');
    if (!st.engine) return t('launch.need_engine');
    if (!st.gpus.size) return t('launch.need_gpu');
    if (Object.keys(validateAll(this.specs, st.values)).length) return t('launch.fix_errors');
    if (this.needSecrets.size) return t('launch.need_secrets', { keys: [...this.needSecrets].join(', ') });
    if (compatLevel(this.report?.compat) === 'block') return t('launch.blocked');
    return null;
  }
  paintGate() { if (!this.launchBtn) return; const g = this.gate(); this.launchBtn.disabled = !!g; this.launchBtn.title = g || ''; }

  // ---------- estimate and checks ----------
  requestBody(forCreate) {
    const st = this.st, ref = this.currentRef();
    return { engine: st.engine, host_id: st.hostId, ...refBody(ref), params: forCreate ? pruneValues(this.specs, st.values) : this.fitParams(), gpu_ids: [...st.gpus].sort(), ...(forCreate ? { image: st.image || undefined, profile_id: st.profileId || undefined, name: st.name || undefined } : { concurrency: st.concurrency }) };
  }
  async runFit() {
    const st = this.st, my = ++this.fitSeq;
    if (!this.currentRef() || !st.gpus.size || !st.engine) { this.report = null; this.paintFit(); return; }
    this.lastFp = memoryFingerprint(this.specs, st.values);
    this.fitBusy = true; this.paintFit();
    try {
      const r = await this.api.fit(this.requestBody(false));
      if (my !== this.fitSeq) return; // a newer estimate is in flight; drop stale answers
      this.report = r; this.fitErr = null;
    } catch (e) { if (my !== this.fitSeq) return; this.fitErr = e; }
    this.fitBusy = false; if (this._alive) this.paintFit();
  }
  paintFit() {
    const totals = Object.fromEntries((this.host()?.gpus || []).map((g) => [g.index, g.total_gib]));
    append(clear(this.fitHost), [this.host() ? h('p', { class: 'muted small' }, t('launch.fit_on', { host: this.host().label })) : null, fitMeter(this.report, { gpuTotals: totals, loading: this.fitBusy, error: this.fitErr })]);
    this.paintGate();
  }
  renderChecks() {
    clear(this.checksHost).append(this.pf ? checkList(this.pf.checks || []) : h('p', { class: 'hint' }, t('launch.checks_hint')),
      btn(t('launch.run_checks'), { size: 'sm', icon: 'check', onClick: () => this.runChecks() }));
  }
  async runChecks() {
    if (this.gate()) { toast(this.gate(), { kind: 'warn' }); return; }
    try { this.pf = await this.api.preflight(this.requestBody(true)); this.renderChecks(); } catch (e) { toastError(e, t('launch.run_checks')); }
  }

  async launch() {
    const r = normalizeReport(this.report);
    if (r?.verdict === 'wont_fit' && !(await confirmDialog(t('launch.confirm_wont_fit'), { confirmLabel: t('launch.launch_anyway'), danger: true }))) return;
    this.launchBtn.disabled = true;
    try {
      clear(this.serverNote);
      const body = this.requestBody(true);
      // Dry-run the hub's own launch checks first so blockers are explained instead of surfacing as a failed start.
      const pf = await this.api.preflight(body);
      this.pf = pf; this.renderChecks();
      const blocks = (pf.checks || []).filter((c) => c.level === 'block');
      if (pf.ok === false || blocks.length) { this.showPreflight(pf); this.paintGate(); return; }
      const inst = await this.api.createInstance(body);
      toast(t('launch.started'), { kind: 'ok' });
      location.hash = `#/instances/${encodeURIComponent(inst.id)}`;
    } catch (e) {
      // The hub answers a failed launch with the checks that failed: show them, not only a toast.
      if (e.problem?.preflight) { this.pf = e.problem.preflight; this.renderChecks(); this.showPreflight(e.problem.preflight); }
      else clear(this.serverNote).append(h('p', { class: 'note bad' }, h('strong', t('launch.server_rejected')), ' ', describeError(e)));
      toastError(e, t('launch.launch')); this.paintGate();
    }
  }
  showPreflight(pf) {
    const rows = (pf.checks || []).filter((c) => c.level !== 'ok');
    openModal(t('launch.preflight_failed'), h('div', { class: 'stack-v' }, rows.length ? checkList(rows) : h('p', t('launch.preflight_unknown'))));
  }
  importProfile() {
    const ta = h('textarea', { rows: 10, spellcheck: 'false', placeholder: t('launch.import_ph'), 'aria-label': t('launch.profile_import') });
    const m = openModal(t('launch.profile_import'), h('form', { class: 'stack-v', onSubmit: async (e) => {
      e.preventDefault();
      if (!ta.value.trim()) return;
      try { const p = await this.api.importProfile(ta.value); this.profiles.push(p); toast(t('launch.profile_saved'), { kind: 'ok' }); m.close(); this.renderForm(); } catch (err) { toastError(err); }
    } }, ta, h('div', { class: 'row end' }, btn(t('launch.profile_import'), { kind: 'primary', type: 'submit' })))); ta.focus();
  }
  async exportProfile() {
    try { const y = await this.api.exportProfile(this.st.profileId); download(`${this.profiles.find((p) => p.id === this.st.profileId)?.name || 'profile'}.yaml`, y, 'application/yaml'); } catch (e) { toastError(e); }
  }
  saveProfile() {
    const name = h('input', { type: 'text', placeholder: t('launch.profile_name'), autocomplete: 'off' });
    const form = h('form', { onSubmit: async (e) => {
      e.preventDefault();
      if (!name.value.trim()) return;
      try { const ref = this.currentRef(); const p = await this.api.saveProfile({ name: name.value.trim(), engine: this.st.engine, repo_id: ref?.format === 'hf' ? ref.repo_id : undefined, params: pruneValues(this.specs, this.st.values) }); this.profiles.push(p); toast(t('launch.profile_saved'), { kind: 'ok' }); m.close(); this.renderForm(); }
      catch (err) { toastError(err); }
    } }, field(t('launch.profile_name'), name), h('div', { class: 'row end' }, btn(t('common.save'), { kind: 'primary', type: 'submit' })));
    const m = openModal(t('launch.save_profile'), form); name.focus();
  }
}
customElements.define('ec-launch', EcLaunch);
