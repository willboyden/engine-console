// Guided launch: choose a model -> choose a machine -> choose a goal -> review -> launch. Written for someone who has never run
// an inference engine: sizes in plain words, machines judged "fits easily / tight / won't fit" with the reason, goals instead of
// settings, and every non-default setting explained before anything starts. The state machine is wizard.js (pure, tested).
import { EcView } from '../components/base.js';
import { h, clear, append, icon } from '../dom.js';
import { t } from '../i18n.js';
import { items, fmtGiB, fmtInt, debounce } from '../format.js';
import { normHost } from '../adapt.js';
import { initialState, reduce, canAdvance, furthestStep, STEPS, GOALS, stepIndex, pickGpus, tpParamKey, fitWord, fitReason, gpusNeeded } from '../wizard.js';
import { hostChoicesForModel, sizeBand, paramsLabel, quickFitHosts, filterLibrary, sortLibrary, inUse, resolveOnHost } from '../library.js';
import { pickEngine, enginesForFormat, engineUsable } from '../engines.js';
import { refBody, refLabel, refFromQuery, refEquals } from '../model-ref.js';
import { normalizePresets, nonDefaultSettings, pruneValues, formatValue } from '../param-form.js';
import { explainFor, paramExplain } from '../learn.js';
import { describeError } from '../errors.js';
import { manageBlock } from '../fleet.js';
import { btn, badge, card, field, select, emptyBox, errorBox, skeleton, pageHead, formatBadge, linkBadge, modeBadge, verdictBadge, reasonText, humanize } from '../components/ui.js';
import { infoTip, withTip, inlineHelp } from '../components/info-tip.js';
import { fitMeter } from '../components/fit-meter.js';
import { checkList } from '../components/check-list.js';
import { openModal, confirmDialog } from '../components/dialog.js';
import { toast, toastError } from '../components/toast.js';

const enc = encodeURIComponent;
const shortName = (n) => String(n || '').replace(/^NVIDIA\s+/i, '');
const GOAL_TITLE = (name) => { const k = `goal.${name}.title`; const v = t(k); return v === k ? humanize(name) : v; };

class EcWizard extends EcView {
  setup() {
    this.s = initialState();
    this.models = []; this.hosts = []; this.engines = []; this.details = new Map(); this.specs = new Map(); this.fits = new Map();
    this.mq = { q: '', format: '' };
    this.stepper = h('ol', { class: 'wiz-steps', 'aria-label': t('wizard.steps') });
    this.stage = h('section', { class: 'wiz-stage', 'aria-live': 'polite' }, skeleton(4));
    this.nav = h('div', { class: 'row between wiz-nav' });
    this.append(pageHead({ page: 'wizard', title: t('wizard.title'), actions: [h('a', { class: 'btn', href: '#/launch?advanced=1' }, t('wizard.advanced_form'))] }), this.stepper, this.stage, this.nav);
    this.init(this.query || {});
  }
  async init(q) {
    try {
      const [engines, hosts, models] = await Promise.all([this.api.engines(), this.api.hosts(), this.api.library()]);
      if (!this._alive) return;
      this.engines = items(engines); this.hosts = items(hosts).map(normHost); this.models = models;
      // Deep link from the Library: ?mkey=<model key>&host=<id> (and the ModelRef fields of the copy).
      const ref = refFromQuery(q);
      const m = this.models.find((x) => x.key === q.mkey) || (ref ? this.models.find((x) => { const r = resolveOnHost(x, q.host, this.details); return r && refEquals(r.ref, ref); }) : null);
      if (m) { this.dispatch({ type: 'model', model: m }, false); if (q.host) await this.chooseHost(q.host, false); }
    } catch (e) { clear(this.stage).append(errorBox(e, () => this.init(q))); return; }
    this.paint();
  }
  dispatch(a, paint = true) {
    const before = this.s.step;
    this.s = reduce(this.s, a);
    if (!paint) return;
    this.paint();
    // A new step: move focus to its heading so a screen-reader user hears where they are.
    if (this.s.step !== before) this.stage.querySelector('h2')?.focus({ preventScroll: true });
  }
  engineObj(id = this.s.engine) { return this.engines.find((e) => e.id === id) || null; }
  async specsFor(engine) { if (!this.specs.has(engine)) this.specs.set(engine, (await this.api.params(engine)).filter((p) => p.flag !== '@image')); return this.specs.get(engine); }
  async detailFor(id) { if (!this.details.has(id)) this.details.set(id, normHost(await this.api.host(id))); return this.details.get(id); }

  // ---------- frame ----------
  paint() {
    this.paintStepper();
    clear(this.stage);
    ({ model: () => this.stepModel(), machine: () => this.stepMachine(), goal: () => this.stepGoal(), review: () => this.stepReview() })[this.s.step]();
    this.paintNav();
    this.stage.querySelector('h2')?.setAttribute('tabindex', '-1');
  }
  paintStepper() {
    const far = furthestStep(this.s), cur = stepIndex(this.s);
    const labels = [...STEPS.map((k) => t(`wizard.step_${k}_short`)), t('wizard.step_launch_short')];
    clear(this.stepper).append(...labels.map((label, i) => {
      const reach = i <= far && i < STEPS.length;
      const inner = [h('span', { class: 'dot' }, i < cur ? icon('check', 12) : String(i + 1)), h('span', label)];
      return h('li', { class: i < cur ? 'done' : i === cur ? 'now' : '', 'aria-current': i === cur ? 'step' : null },
        reach && i !== cur ? h('button', { type: 'button', class: 'step-btn', onClick: () => this.dispatch({ type: 'goto', step: STEPS[i] }) }, ...inner) : h('span', { class: 'step-btn static' }, ...inner));
    }));
  }
  paintNav() {
    const s = this.s, i = stepIndex(s), adv = canAdvance(s);
    clear(this.nav);
    this.nav.append(
      i > 0 ? btn(t('common.back'), { icon: 'arrowleft', onClick: () => this.dispatch({ type: 'back' }) }) : h('span'),
      s.step === 'review' ? this.launchBtn = btn(t('wizard.launch'), { kind: 'primary', icon: 'launch', disabled: !this.canLaunch(), reason: this.launchReason(), onClick: () => this.launch() })
        : btn(t('common.next'), { kind: 'primary', disabled: !adv.ok, reason: adv.reason ? { key: adv.reason } : null, onClick: () => { if (s.step === 'machine' && !s.hostId) return; this.dispatch({ type: 'next' }); } }));
  }
  heading(key) { return h('div', { class: 'wiz-head' }, h('h2', {}, t(key)), h('p', { class: 'muted' }, t(`${key}_intro`))); }

  // ---------- 1. model ----------
  stepModel() {
    const input = h('input', { type: 'search', value: this.mq.q, placeholder: t('library.search_ph'), 'aria-label': t('library.search'), autocomplete: 'off', onInput: debounce((e) => { this.mq.q = e.target.value; this.paintModelList(); }, 200) });
    const fmt = select([{ value: '', label: t('common.any') }, ...['hf', 'gguf', 'ollama', 'plain'].filter((f) => this.models.some((m) => m.format === f)).map((f) => ({ value: f, label: t(`format.${f}`) }))], this.mq.format, (v) => { this.mq.format = v; this.paintModelList(); }, { 'aria-label': t('library.format') });
    this.modelList = h('div', { class: 'pick-grid' });
    append(this.stage, [this.heading('wizard.step_model'),
      this.models.length ? h('div', { class: 'row gap wrap bottom wiz-filter' }, h('div', { class: 'field grow' }, input), field(t('library.format'), fmt, null, { kind: 'concept', key: 'field.format_filter', inline: false })) : null,
      this.models.length ? this.modelList : emptyBox(t('wizard.no_models'), t('wizard.no_models_hint'), h('a', { class: 'btn primary', href: '#/discover' }, t('library.find_models')))]);
    if (this.models.length) this.paintModelList();
  }
  paintModelList() {
    const list = sortLibrary(filterLibrary(this.models, this.mq), 'name');
    clear(this.modelList).append(...(list.length ? list.map((m) => this.modelCard(m)) : [emptyBox(t('library.no_match'), t('library.no_match_hint'))]));
  }
  modelCard(m) {
    const sel = this.s.model?.key === m.key;
    const band = sizeBand(m.size_bytes);
    const fits = quickFitHosts(m, this.hosts);
    const canRun = enginesForFormat(this.engines, m.format).length > 0;
    const hostNames = [...new Set(m.copies.flatMap((c) => [c.host_label])) ].join(', ');
    return h('article', { class: `card pick-card${sel ? ' selected' : ''}` },
      h('div', { class: 'card-body' },
        h('h3', { class: 'wrap-anywhere' }, m.name),
        h('div', { class: 'row gap wrap small' }, formatBadge(m.format), m.quant ? badge(m.quant, 'muted') : null, paramsLabel(m.params) ? badge(t('library.params', { n: paramsLabel(m.params) }), 'muted') : null, inUse(m) ? badge(t('library.in_use'), 'warn') : null),
        h('p', {}, t('wizard.size_words', { size: fmtGiB(m.size_bytes / 2 ** 30, m.size_bytes < 2 ** 33 ? 1 : 0), band: t(`wizard.band_${band}`) })),
        h('p', { class: 'small muted' }, t('wizard.stored_on', { hosts: hostNames })),
        !canRun ? h('p', { class: 'note warn small' }, t('wizard.no_engine_for_format')) : fits.length ? h('p', { class: 'small' }, h('strong', t('wizard.fits_on')), ' ', fits.map((f) => f.label).join(', ')) : h('p', { class: 'note warn small' }, t('wizard.fits_nowhere')),
        (m.warnings || []).map((w) => h('p', { class: 'note warn small' }, w)),
        btn(sel ? t('wizard.selected') : t('wizard.choose'), { kind: sel ? '' : 'primary', icon: sel ? 'check' : undefined, disabled: !canRun, reason: !canRun ? { key: 'wizard.no_engine_for_format' } : null, onClick: () => this.dispatch({ type: 'model', model: m }) })));
  }

  // ---------- 2. machine ----------
  async stepMachine() {
    const m = this.s.model;
    this.machineList = h('div', { class: 'pick-grid' }, skeleton(2));
    append(this.stage, [this.heading('wizard.step_machine'), h('p', { class: 'chosen' }, h('strong', t('wizard.chosen_model')), ' ', m.name, ' ', formatBadge(m.format)), this.machineList]);
    // Machines that read the model through shared storage need their own root id: fetch the details of those.
    const sharing = this.hosts.filter((x) => m.copies.some((c) => c.host_id !== x.id && (c.visible_on || []).includes(x.id)));
    await Promise.all(sharing.map((x) => this.detailFor(x.id).catch(() => null)));
    if (!this._alive || this.s.step !== 'machine' || this.s.model !== m) return;
    this.paintMachines();
  }
  paintMachines() {
    const m = this.s.model;
    const choices = hostChoicesForModel(m, this.hosts, this.details);
    clear(this.machineList).append(...choices.map((c) => this.machineCard(m, c)));
    for (const c of choices) if (c.state === 'ok') this.loadFit(m, c);
  }
  /** One estimate per (model, machine): on the freest GPU, or on as many as the estimator says it needs. Cached per pair. */
  fitFor(m, c) {
    const key = `${m.key}|${c.host.id}`;
    if (!this.fits.has(key)) {
      const engine = pickEngine(this.engines, m.format, c.host);
      this.fits.set(key, !engine ? Promise.resolve({ none: true }) : (async () => {
        const base = { engine: engine.id, host_id: c.host.id, ...refBody(c.ref), params: {}, concurrency: 1 };
        const one = pickGpus(c.host.gpus, 1);
        let report = await this.api.fit({ ...base, gpu_ids: one });
        let gpuIds = one, baseParams = {};
        const need = gpusNeeded(report, c.host.gpus.length);
        if (report.verdict === 'wont_fit' && need > 1) {
          const tp = tpParamKey(await this.specsFor(engine.id));
          if (tp) { gpuIds = pickGpus(c.host.gpus, need); baseParams = { [tp]: need }; report = await this.api.fit({ ...base, gpu_ids: gpuIds, params: baseParams }); }
        }
        return { engine: engine.id, report, gpuIds, baseParams };
      })().catch((e) => ({ error: e })));
    }
    return this.fits.get(key);
  }
  loadFit(m, c) {
    this.fitFor(m, c).then((r) => { if (!this._alive || this.s.step !== 'machine' || this.s.model !== m) return; const slot = this.machineList?.querySelector(`[data-host="${c.host.id}"] .verdict-slot`); if (slot) this.fillVerdict(slot, m, c, r); });
  }
  fillVerdict(slot, m, c, r) {
    clear(slot);
    if (r.none) { append(slot, [h('p', { class: 'note warn small' }, t('wizard.no_engine_here'))]); this.setUse(c, false, { key: 'wizard.no_engine_here' }); return; }
    if (r.error) { append(slot, [h('p', { class: 'note warn small' }, describeError(r.error))]); this.setUse(c, true); return; }
    const v = r.report.verdict, reason = fitReason(r.report);
    append(slot, [h('div', { class: 'row gap wrap' }, verdictBadge(v, { text: t(fitWord(v)) })), h('p', { class: 'small' }, t(reason.key, reason.vars)),
      r.gpuIds.length > 1 ? h('p', { class: 'small muted' }, t('wizard.uses_gpus', { n: r.gpuIds.length })) : null]);
    this.setUse(c, v !== 'wont_fit', v === 'wont_fit' ? { key: 'wizard.wont_fit_reason' } : null, r);
  }
  /** The estimate arrived: a machine where it will not fit stays focusable but says why (title + accessible description). */
  setUse(c, ok, reason) {
    const b = this.machineList?.querySelector(`[data-host="${c.host.id}"] .use-btn`);
    if (!b) return;
    let d = b.querySelector('.why');
    if (ok) { b.removeAttribute('aria-disabled'); b.removeAttribute('aria-describedby'); b.title = ''; d?.remove(); return; }
    if (!d) { d = h('span', { class: 'why', hidden: true, id: `why-${c.host.id}` }); b.append(d); }
    d.textContent = reasonText(reason); b.setAttribute('aria-describedby', d.id); b.setAttribute('aria-disabled', 'true'); b.title = d.textContent;
  }
  /** Take a machine: its copy decides the ModelRef, its estimate decides GPUs, settings the fit needs and the engine. */
  async chooseHost(hostId, paint = true) {
    const m = this.s.model, x = this.hosts.find((y) => y.id === hostId);
    if (!m || !x) return;
    if (m.copies.some((c) => c.host_id !== hostId && (c.visible_on || []).includes(hostId))) await this.detailFor(hostId).catch(() => null);
    const c = hostChoicesForModel(m, this.hosts, this.details).find((y) => y.host.id === hostId);
    if (!c || c.state !== 'ok') return;
    const r = await this.fitFor(m, c);
    if (r?.none) return;
    this.dispatch({ type: 'host', hostId, ref: c.ref, gpuIds: r?.gpuIds || pickGpus(x.gpus, 1), baseParams: r?.baseParams || {}, engine: r?.engine || pickEngine(this.engines, m.format, x)?.id }, paint);
  }
  machineCard(m, c) {
    const x = c.host, ok = c.state === 'ok', sel = this.s.hostId === x.id;
    const gpus = x.gpus || [], g0 = gpus[0];
    const gpuText = !gpus.length ? t('fleet.no_gpus') : g0.unified_memory ? t('wizard.gpu_unified', { total: fmtGiB(g0.total_gib, 0) }) : t('wizard.gpu_line', { n: gpus.length, name: shortName(g0.name), mem: fmtGiB(g0.total_gib, 0), free: fmtGiB(gpus.reduce((a, g) => a + (g.free_gib ?? 0), 0), 0) });
    const use = btn(sel ? t('wizard.selected') : t('wizard.use_machine'), { kind: sel ? '' : 'primary', icon: sel ? 'check' : undefined, disabled: !ok, reason: !ok ? c.reason : null, onClick: (e) => {
      if (e.currentTarget.getAttribute('aria-disabled') === 'true') return;
      this.chooseHost(x.id);
    } });
    use.classList.add('use-btn');
    const copyable = c.state === 'no_copy' && manageBlock(x) === null;
    return h('article', { class: `card pick-card machine-card${sel ? ' selected' : ''}${ok ? '' : ' is-disabled'}`, dataset: { host: x.id } },
      h('div', { class: 'card-body' },
        h('div', { class: 'row gap wrap between' }, h('h3', {}, x.label), h('div', { class: 'row gap wrap' }, linkBadge(x.link.state), x.policy ? modeBadge(x.policy.mode === 'manage' ? 'manage' : 'monitor') : null)),
        h('p', { class: 'small muted' }, gpuText),
        ok ? h('div', { class: 'verdict-slot' }, h('p', { class: 'muted small' }, t('wizard.checking_fit'))) : h('div', {}, h('p', { class: 'note warn small' }, reasonText(c.reason)),
          copyable ? h('a', { class: 'small', href: `#/library?q=${enc(m.name)}` }, t('wizard.copy_there')) : null),
        use));
  }

  // ---------- 3. goal ----------
  async stepGoal() {
    const s = this.s, m = s.model, x = this.hosts.find((y) => y.id === s.hostId);
    this.goalList = h('div', { class: 'pick-grid' }, skeleton(2));
    const engines = enginesForFormat(this.engines, m.format).filter((e) => engineUsable(e, x).ok);
    const engSel = engines.length > 1 ? h('details', { class: 'wiz-advanced' }, h('summary', t('wizard.change_engine')),
      field(t('launch.engine'), select(engines.map((e) => ({ value: e.id, label: e.display_name || e.id })), s.engine, (v) => this.dispatch({ type: 'engine', engine: v })), t('wizard.change_engine_hint'), { kind: 'concept', key: 'field.engine' })) : null;
    append(this.stage, [this.heading('wizard.step_goal'),
      h('p', { class: 'chosen' }, h('strong', t('wizard.chosen_model')), ' ', m.name, ' · ', h('strong', t('wizard.chosen_machine')), ' ', x?.label, ' · ', h('strong', t('launch.engine')), ' ', this.engineObj()?.display_name || s.engine, infoTip({ kind: 'concept', key: 'field.engine', title: t('launch.engine') })),
      engSel, this.goalList]);
    let specs = [];
    try { specs = await this.specsFor(s.engine); } catch { /* the goal cards still work without the setting names */ }
    if (!this._alive || this.s.step !== 'goal') return;
    const presets = normalizePresets(this.engineObj()?.presets);
    const ordered = [...GOALS.filter((g) => presets.some((p) => p.name === g)), ...presets.map((p) => p.name).filter((n) => !GOALS.includes(n))];
    clear(this.goalList).append(...ordered.map((name) => this.goalCard(presets.find((p) => p.name === name), specs)));
  }
  goalCard(p, specs) {
    const s = this.s, sel = s.goal === p.name;
    const ex = explainFor({ kind: 'preset', key: p.name, engine: s.engine });
    const hint = t(`goal.${p.name}.hint`) === `goal.${p.name}.hint` ? '' : t(`goal.${p.name}.hint`);
    const sets = Object.entries(p.params).map(([k, v]) => { const sp = specs.find((q) => q.key === k); return sp ? `${sp.label}: ${formatValue(sp, v)}` : null; }).filter(Boolean);
    return h('article', { class: `card pick-card goal-card${sel ? ' selected' : ''}` }, h('div', { class: 'card-body' },
      h('div', { class: 'row gap' }, h('h3', {}, GOAL_TITLE(p.name)), ex ? infoTip({ kind: 'preset', key: p.name, engine: s.engine, title: GOAL_TITLE(p.name) }) : null),
      // The bundle's own explanation of the preset when there is one; otherwise this UI's one-line description of the goal.
      h('p', {}, ex ? ex.summary : hint),
      sets.length ? h('details', { class: 'small' }, h('summary', t('wizard.goal_sets', { count: sets.length })), h('ul', { class: 'plain-list' }, sets.map((x) => h('li', x)))) : null,
      btn(sel ? t('wizard.selected') : t('wizard.choose_goal'), { kind: sel ? '' : 'primary', icon: sel ? 'check' : undefined, onClick: () => this.dispatch({ type: 'goal', goal: p.name, params: p.params }) })));
  }

  // ---------- 4. review ----------
  bodyFor() {
    const s = this.s, specs = this.specs.get(s.engine) || [];
    return { engine: s.engine, host_id: s.hostId, ...refBody(s.ref), params: pruneValues(specs, s.params), gpu_ids: s.gpuIds, image: s.image || undefined, name: s.name || undefined };
  }
  async stepReview() {
    const s = this.s, m = s.model, x = this.hosts.find((y) => y.id === s.hostId);
    this.reviewHost = h('div', { class: 'stack-v' }, skeleton(3));
    this.stage.append(this.heading('wizard.step_review'), this.reviewHost);
    this.pf = null; this.reviewFit = null;
    let specs = [];
    try { specs = await this.specsFor(s.engine); } catch (e) { clear(this.reviewHost).append(errorBox(e)); return; }
    if (!this._alive || this.s.step !== 'review') return;
    const changed = nonDefaultSettings(specs, s.params);
    const name = h('input', { type: 'text', value: s.name, placeholder: t('launch.name_ph'), onInput: (e) => { this.s = reduce(this.s, { type: 'name', name: e.target.value.trim() }); } });
    this.checksHost = h('div', {}, skeleton(2)); this.fitHost = h('div');
    clear(this.reviewHost).append(
      card(t('wizard.review_summary'), h('dl', { class: 'facts' },
        h('div', { class: 'kv' }, h('dt', t('launch.model')), h('dd', {}, m.name, ' ', formatBadge(m.format))),
        h('div', { class: 'kv' }, h('dt', t('launch.machine')), h('dd', {}, x?.label || s.hostId)),
        h('div', { class: 'kv' }, h('dt', withTip(t('launch.engine'), { kind: 'concept', key: 'field.engine' }, t('launch.engine'))), h('dd', {}, this.engineObj()?.display_name || s.engine)),
        h('div', { class: 'kv' }, h('dt', t('wizard.review_goal')), h('dd', {}, GOAL_TITLE(s.goal))),
        h('div', { class: 'kv' }, h('dt', t('launch.gpus')), h('dd', {}, s.gpuIds.map((i) => `GPU ${i}`).join(', ') || '–')))),
      card(t('wizard.review_settings'), changed.length ? h('div', { class: 'stack-v' }, h('p', { class: 'muted small' }, t('wizard.review_settings_intro')), h('ul', { class: 'setting-list' }, changed.map((c) => this.settingRow(c, s.engine)))) : h('p', { class: 'muted' }, t('wizard.review_no_changes'))),
      card(t('wizard.review_name'), field(t('launch.name'), name, null, { kind: 'concept', key: 'field.name' })),
      card(t('fit.title'), this.fitHost), card(t('launch.checks'), this.checksHost));
    this.runReviewChecks();
  }
  settingRow(c, engine) {
    const ex = paramExplain(c.spec, engine);
    return h('li', { class: 'setting-row' }, h('div', { class: 'row gap wrap' }, h('strong', c.label), h('code', { class: 'chip-code' }, c.text), infoTip({ kind: 'param', spec: c.spec, engine, title: c.label })),
      ex ? h('p', { class: 'muted small' }, ex.summary) : h('p', { class: 'muted small' }, c.spec.help));
  }
  async runReviewChecks() {
    const s = this.s, body = this.bodyFor(), my = (this._rv = (this._rv || 0) + 1);
    try {
      const fitBody = { engine: body.engine, host_id: body.host_id, ...refBody(s.ref), params: body.params, gpu_ids: body.gpu_ids, concurrency: 1 };
      const [pf, fit] = await Promise.all([this.api.preflight(body), this.api.fit(fitBody).catch(() => null)]);
      if (!this._alive || my !== this._rv || this.s.step !== 'review') return;
      this.pf = pf; this.reviewFit = fit;
      const totals = Object.fromEntries((this.hosts.find((y) => y.id === s.hostId)?.gpus || []).map((g) => [g.index, g.total_gib]));
      clear(this.fitHost).append(fitMeter(fit, { gpuTotals: totals }));
      clear(this.checksHost).append(checkList(pf.checks || []));
    } catch (e) { if (this._alive && my === this._rv) clear(this.checksHost).append(errorBox(e, () => this.runReviewChecks())); }
    if (this.s.step === 'review') this.paintNav();
  }
  blocks() { return (this.pf?.checks || []).filter((c) => c.level === 'block'); }
  canLaunch() { return !!this.pf && this.pf.ok !== false && !this.blocks().length && !this.launching; }
  launchReason() { if (!this.pf) return { key: 'wizard.checking' }; const b = this.blocks()[0]; return b ? { text: b.message } : null; }
  async launch() {
    if (!this.canLaunch()) return;
    this.launching = true; this.paintNav();
    try {
      const inst = await this.api.createInstance(this.bodyFor());
      toast(t('launch.started'), { kind: 'ok' });
      location.hash = `#/instances/${enc(inst.id)}`;
    } catch (e) {
      if (e.problem?.preflight) { this.pf = e.problem.preflight; clear(this.checksHost).append(checkList(this.pf.checks || [])); }
      toastError(e, t('wizard.launch'));
    }
    this.launching = false; if (this._alive) this.paintNav();
  }
}
customElements.define('ec-wizard', EcWizard);
