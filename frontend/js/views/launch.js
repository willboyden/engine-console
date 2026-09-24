import { EcView } from '../components/base.js';
import { h, clear, copyText, download } from '../dom.js';
import { t } from '../i18n.js';
import { items, debounce } from '../format.js';
import { memoryFingerprint, normalizePresets, pruneValues, validateAll, toCliArgs, cliString, changedCount, isNonCli, toEnvLines, splitRedacted } from '../param-form.js';
import { describeError } from '../errors.js';
import { compatLevel, normalizeReport } from '../fit-format.js';
import { renderParamForm } from '../components/param-field.js';
import { fitMeter } from '../components/fit-meter.js';
import { btn, card, field, select, errorBox, skeleton, badge } from '../components/ui.js';
import { openModal, confirmDialog } from '../components/dialog.js';
import { toast, toastError } from '../components/toast.js';

class EcLaunch extends EcView {
  setup() {
    const q = this.query || {};
    this.st = { engine: q.engine || 'vllm', repo: q.repo || '', name: '', gpus: new Set(), values: {}, showAdvanced: false, search: '', profileId: '', concurrency: 1 };
    this.needSecrets = new Set(); // keys whose saved value is "[set]": the server never returns secrets, so the user must retype them
    this.catalogs = new Map(); this.report = null; this.fitErr = null; this.fitBusy = false; this.lastFp = null; this.fitSeq = 0;
    this.scheduleFit = debounce(() => this.runFit(), 350);
    this.own(() => this.scheduleFit.cancel());
    this.formHost = h('div', {}, skeleton(6));
    this.fitHost = h('div');
    this.append(h('h1', t('nav.launch')), h('div', { class: 'launch-grid' }, this.formHost, h('aside', { class: 'launch-side' }, card(t('fit.title'), this.fitHost, null), this.actionsHost = h('div', { class: 'card' }))));
    this.init(q);
  }
  async init(q) {
    try {
      const [engines, hw, profiles, models] = await Promise.all([this.api.engines(), this.api.hardware(), this.api.profiles().catch(() => []), this.api.models().catch(() => [])]);
      if (!this._alive) return;
      Object.assign(this, { engines: items(engines), hw, profiles: items(profiles), local: items(models) });
      this.st.gpus = new Set(hw.gpus.slice(0, 1).map((g) => g.index));
      if (q.from) {
        const i = await this.api.instance(q.from);
        const sp = splitRedacted(i.params || {}); sp.redacted.forEach((k) => this.needSecrets.add(k));
        Object.assign(this.st, { engine: i.engine, repo: i.repo_id, values: sp.values, gpus: new Set(i.gpu_ids || [...this.st.gpus]), name: i.name ? `${i.name}-2` : '' });
      }
      if (!this.engines.some((e) => e.id === this.st.engine)) this.st.engine = this.engines[0]?.id;
      await this.loadCatalog();
      this.renderForm(); this.renderActions(); this.scheduleFit();
    } catch (e) { clear(this.formHost).append(errorBox(e, () => this.init(q))); }
  }
  async loadCatalog() {
    const e = this.st.engine;
    if (!this.catalogs.has(e)) this.catalogs.set(e, await this.api.params(e));
    // Engine images are pinned server-side (contract v1.1); an `@image` param, if a catalog still lists one, is not offered.
    this.specs = this.catalogs.get(e).filter((s) => s.flag !== '@image');
    this.presets = normalizePresets(this.engines.find((x) => x.id === e)?.presets);
    this.st.values = pruneValues(this.specs, this.st.values);
  }
  async setEngine(e) {
    this.st.engine = e; this.st.profileId = '';
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

  renderForm() {
    const st = this.st;
    const repo = h('input', { type: 'text', value: st.repo, list: 'local-models', placeholder: 'org/model-name', autocomplete: 'off', spellcheck: 'false', onInput: (e) => { st.repo = e.target.value.trim(); this.scheduleFit(); } });
    const dl = h('datalist', { id: 'local-models' }, this.local.map((m) => h('option', { value: m.repo_id })));
    const eng = h('div', { class: 'seg-control', role: 'radiogroup', 'aria-label': t('launch.engine') },
      this.engines.map((e) => h('button', { type: 'button', role: 'radio', 'aria-checked': String(e.id === st.engine), class: e.id === st.engine ? 'on' : '', onClick: () => this.setEngine(e.id) }, e.display_name || e.id)));
    const gpus = h('div', { class: 'row gap wrap', role: 'group', 'aria-label': t('launch.gpus') }, this.hw.gpus.map((g) =>
      h('label', { class: 'check' }, h('input', { type: 'checkbox', checked: st.gpus.has(g.index), onChange: (e) => { e.target.checked ? st.gpus.add(g.index) : st.gpus.delete(g.index); this.scheduleFit(); this.renderActions(); } }),
        h('span', `GPU ${g.index}`), h('small', { class: 'muted' }, `${g.name.replace('NVIDIA ', '')} · ${g.free_gib.toFixed(0)} GiB ${t('launch.free')}`))));
    const profSel = select([{ value: '', label: t('launch.no_profile') }, ...this.profiles.filter((p) => !p.engine || p.engine === st.engine).map((p) => ({ value: p.id, label: p.name }))], st.profileId, (id) => {
      st.profileId = id; const p = this.profiles.find((x) => x.id === id); if (p) { clear(this.serverNote); if (p.repo_id && !st.repo) { st.repo = p.repo_id; repo.value = p.repo_id; } this.applyValues({ ...p.params }); }
    });
    const presetBar = h('div', { class: 'row gap wrap', role: 'group', 'aria-label': t('launch.presets') }, this.presets.map((p) => btn(p.label, { size: 'sm', onClick: () => { st.profileId = ''; this.applyValues({ ...p.params }); toast(t('launch.preset_applied', { name: p.label }), { kind: 'ok', timeout: 2000 }); } })),
      btn(t('launch.clear_all'), { size: 'sm', kind: 'ghost', onClick: () => this.applyValues({}) }));
    const search = h('input', { type: 'search', placeholder: t('params.search'), 'aria-label': t('params.search'), value: st.search, onInput: debounce((e) => { st.search = e.target.value; this.renderParams(); }, 150) });
    const adv = h('label', { class: 'check' }, h('input', { type: 'checkbox', checked: st.showAdvanced, onChange: (e) => { st.showAdvanced = e.target.checked; this.renderParams(); } }), h('span', t('params.show_advanced')));
    this.paramsHost = h('div', { class: 'params' });
    this.secretsNote = h('div', { role: 'status' }); this.serverNote = h('div', { role: 'alert' });
    clear(this.formHost).append(
      card(t('launch.target'), h('div', { class: 'stack-v' },
        field(t('launch.engine'), eng), h('div', { class: 'hint' }, `${t('launch.image')}: `, h('code', this.engines.find((x) => x.id === st.engine)?.default_image || '–'), ` · ${t('launch.image_hint')}`),
        field(t('launch.model'), repo, t('launch.model_hint')), dl,
        h('div', { class: 'field' }, h('label', t('launch.gpus')), gpus),
        h('div', { class: 'row gap wrap bottom' }, field(t('launch.concurrency'), h('input', { type: 'number', min: 1, max: 4096, step: 1, value: String(st.concurrency), onChange: (e) => { st.concurrency = Math.max(1, Math.min(4096, parseInt(e.target.value, 10) || 1)); e.target.value = String(st.concurrency); this.scheduleFit(); } }), t('launch.concurrency_hint')), field(t('launch.name'), h('input', { type: 'text', value: st.name, placeholder: t('launch.name_ph'), onInput: (e) => { st.name = e.target.value.trim(); } })), field(t('launch.profile'), profSel), btn(t('launch.profile_import'), { size: 'sm', onClick: () => this.importProfile() }), st.profileId ? btn(t('launch.profile_export'), { size: 'sm', onClick: () => this.exportProfile() }) : null))),
      card(t('launch.params'), h('div', { class: 'stack-v' }, this.serverNote, this.secretsNote, presetBar, h('div', { class: 'row gap wrap between' }, h('div', { class: 'grow' }, search), adv), this.paramsHost)));
    this.renderParams();
  }
  renderParams() { renderParamForm(this.paramsHost, { specs: this.specs, values: this.st.values, onChange: (k, v) => this.onParam(k, v), search: this.st.search, showAdvanced: this.st.showAdvanced, redacted: this.needSecrets }); this.paintCommand(); }
  paintSecretsNote() {
    if (!this.secretsNote) return;
    const labels = [...this.needSecrets].map((k) => this.specs.find((s) => s.key === k)?.label || k).join(', ');
    clear(this.secretsNote).append(...(this.needSecrets.size ? [h('p', { class: 'note warn' }, t('launch.secrets_note', { keys: labels }))] : []));
    this.paintGate();
  }

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
  cmdText() { const env = toEnvLines(this.specs, this.st.values); return [...env, cliString(['--model', this.st.repo || '<model>', ...toCliArgs(this.specs, this.st.values)])].join('\n'); }
  paintCommand() { if (this.cmdOut) this.cmdOut.textContent = this.cmdText(); const n = this.actionsHost?.querySelector('.muted.small'); if (n) n.textContent = t('launch.modified', { n: changedCount(this.st.values) }); this.paintGate(); }
  gate() {
    const st = this.st;
    if (!st.repo) return t('launch.need_model');
    if (!st.gpus.size) return t('launch.need_gpu');
    if (Object.keys(validateAll(this.specs, st.values)).length) return t('launch.fix_errors');
    if (this.needSecrets.size) return t('launch.need_secrets', { keys: [...this.needSecrets].join(', ') });
    if (compatLevel(this.report?.compat) === 'block') return t('launch.blocked');
    return null;
  }
  paintGate() { if (!this.launchBtn) return; const g = this.gate(); this.launchBtn.disabled = !!g; this.launchBtn.title = g || ''; }

  async runFit() {
    const st = this.st, my = ++this.fitSeq;
    if (!st.repo || !st.gpus.size) { this.report = null; this.paintFit(); return; }
    this.lastFp = memoryFingerprint(this.specs, st.values);
    this.fitBusy = true; this.paintFit();
    try {
      const r = await this.api.fit({ engine: st.engine, repo_id: st.repo, params: this.fitParams(), gpu_ids: [...st.gpus].sort(), concurrency: st.concurrency });
      if (my !== this.fitSeq) return; // a newer estimate is in flight; drop stale answers
      this.report = r; this.fitErr = null;
    } catch (e) { if (my !== this.fitSeq) return; this.fitErr = e; }
    this.fitBusy = false; if (this._alive) this.paintFit();
  }
  paintFit() {
    const totals = Object.fromEntries((this.hw?.gpus || []).map((g) => [g.index, g.total_gib]));
    clear(this.fitHost).append(fitMeter(this.report, { gpuTotals: totals, loading: this.fitBusy, error: this.fitErr }));
    this.paintGate();
  }

  async launch() {
    const st = this.st, r = normalizeReport(this.report);
    if (r?.verdict === 'wont_fit' && !(await confirmDialog(t('launch.confirm_wont_fit'), { confirmLabel: t('launch.launch_anyway'), danger: true }))) return;
    this.launchBtn.disabled = true;
    try {
      clear(this.serverNote);
      const body = { engine: st.engine, repo_id: st.repo, params: pruneValues(this.specs, st.values), gpu_ids: [...st.gpus].sort(), profile_id: st.profileId || undefined, name: st.name || undefined };
      // Dry-run the server's own launch checks first so blockers are explained instead of surfacing as a failed start.
      const pf = await this.api.preflight(body);
      const blocks = (pf.checks || []).filter((c) => c.level === 'block');
      if (pf.ok === false || blocks.length) { this.showPreflight(pf); this.paintGate(); return; }
      const inst = await this.api.createInstance(body);
      toast(t('launch.started'), { kind: 'ok' });
      location.hash = `#/instances/${encodeURIComponent(inst.id)}`;
    } catch (e) {
      // Server-side rejections (secret_in_argv, 422 validation, preflight_blocked) are shown next to the form, not only as a toast.
      clear(this.serverNote).append(h('p', { class: 'note bad' }, h('strong', t('launch.server_rejected')), ' ', describeError(e)));
      toastError(e, t('launch.launch')); this.paintGate();
    }
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
  showPreflight(pf) {
    const rows = (pf.checks || []).filter((c) => c.level !== 'ok');
    openModal(t('launch.preflight_failed'), h('div', { class: 'stack-v' }, rows.length ? rows.map((c) => h('p', { class: `note ${c.level === 'block' ? 'bad' : 'warn'}`, role: 'note' }, h('strong', c.code), ' ', c.message)) : h('p', t('launch.preflight_unknown'))));
  }
  saveProfile() {
    const name = h('input', { type: 'text', placeholder: t('launch.profile_name'), autocomplete: 'off' });
    const form = h('form', { onSubmit: async (e) => {
      e.preventDefault();
      if (!name.value.trim()) return;
      try { const p = await this.api.saveProfile({ name: name.value.trim(), engine: this.st.engine, repo_id: this.st.repo || undefined, params: pruneValues(this.specs, this.st.values) }); this.profiles.push(p); toast(t('launch.profile_saved'), { kind: 'ok' }); m.close(); this.renderForm(); }
      catch (err) { toastError(err); }
    } }, field(t('launch.profile_name'), name), h('div', { class: 'row end' }, btn(t('common.save'), { kind: 'primary', type: 'submit' })));
    const m = openModal(t('launch.save_profile'), form); name.focus();
  }
}
customElements.define('ec-launch', EcLaunch);
