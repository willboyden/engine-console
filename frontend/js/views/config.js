// Config files: draft, validate, export. The hub keeps its config files out of the browser's reach on purpose, so this
// page never writes one: it loads what the hub may show (hosts.json read-only, when its loader accepts it; the env file
// and a node policy never), a template from the repository or a local file (read in the browser, sent nowhere but the
// check), edits a draft in a Form view (typed controls, each with an (i) explanation from the learn bundle) or a Text
// view (the same draft), checks it with the hub's real loaders (POST /config/validate), and hands the result back as a
// diff, Copy, Download and the commands that install it, which the human runs.
import { EcView } from '../components/base.js';
import { h, clear, append, download, copyText } from '../dom.js';
import { t } from '../i18n.js';
import { btn, card, field, select, badge, emptyBox, errorBox, skeleton, pageHead } from '../components/ui.js';
import { infoTip } from '../components/info-tip.js';
import { cmdBlock } from '../components/cmd-block.js';
import { toast, toastError } from '../components/toast.js';
import {
  CFG_FILES,
  HOST_FIELDS,
  POLICY_FIELDS,
  ROOT_FIELDS,
  ENV_KIND,
  ENV_COMMON,
  fileConcept,
  fieldConcept,
  fieldValue,
  fieldText,
  installCommands,
  looksSecret,
} from '../cfg-fields.js';
import { parseJson, toText, getPath, setPath } from '../cfg-json.js';
import { parseLines, values as envValues, setValue as envSet, removeKey as envRemove } from '../cfg-dotenv.js';
import { lineDiff, diffText } from '../cfg-diff.js';

const FILE_IDS = CFG_FILES.map((f) => f.id);
const utf8Bytes = (s) => new TextEncoder().encode(s).length;
const meta = (id) => CFG_FILES.find((f) => f.id === id);
let uid = 0;

/** A small ARIA tablist (roving tabindex, arrows/Home/End). defs: [{id, label}]; ids get `prefix-`. */
function miniTabs(defs, active, onSelect, label, prefix) {
  const buttons = defs.map((d) =>
    h(
      'button',
      {
        type: 'button',
        role: 'tab',
        class: 'tab',
        id: `${prefix}-${d.id}`,
        'aria-controls': `${prefix}-panel`,
        dataset: { id: d.id },
        onClick: () => pick(d.id),
      },
      d.label
    )
  );
  const list = h('div', { class: 'tabs mini-tabs', role: 'tablist', 'aria-label': label }, buttons);
  const mark = (id, focus) =>
    buttons.forEach((b) => {
      const on = b.dataset.id === id;
      b.setAttribute('aria-selected', String(on));
      b.tabIndex = on ? 0 : -1;
      b.classList.toggle('on', on);
      if (on && focus) b.focus();
    });
  const pick = (id, focus = false) => {
    mark(id, focus);
    onSelect(id);
  };
  list.addEventListener('keydown', (e) => {
    const i = buttons.indexOf(document.activeElement);
    if (i < 0) return;
    const to = { ArrowRight: i + 1, ArrowLeft: i - 1, Home: 0, End: buttons.length - 1 }[e.key];
    if (to === undefined) return;
    e.preventDefault();
    pick(defs[(to + defs.length) % defs.length].id, true);
  });
  mark(active, false);
  return list;
}

class EcConfig extends EcView {
  setup() {
    this.drafts = {}; // file -> {text, base, baseLabel}
    this.infos = {}; // file -> GET /config/{file}
    this.view = 'form';
    this.file = FILE_IDS.includes(this.query?.file) ? this.query.file : 'hosts';
    this.body = h('div', { id: 'cfgfile-panel', role: 'tabpanel', class: 'stack-v' });
    const tabs = miniTabs(
      CFG_FILES.map((f) => ({ id: f.id, label: t(`cfg.file_${f.id}`) })),
      this.file,
      (id) => {
        this.file = id;
        this.body.setAttribute('aria-labelledby', `cfgfile-${id}`);
        this.open();
      },
      t('cfg.files'),
      'cfgfile'
    );
    this.body.setAttribute('aria-labelledby', `cfgfile-${this.file}`);
    this.append(
      pageHead({ page: 'config', title: t('nav.config') }),
      h('p', { class: 'note info' }, h('strong', t('cfg.never_writes')), ' ', t('cfg.never_writes_hint')),
      tabs,
      this.body
    );
    this.open();
  }

  async open() {
    const file = this.file;
    clear(this.body).append(skeleton(4));
    try {
      const info = this.infos[file] || (this.infos[file] = await this.api.configFile(file));
      if (!this._alive || this.file !== file) return;
      if (!this.drafts[file]) {
        const tpl = Object.entries(info.templates || {})[0];
        this.drafts[file] =
          info.served && typeof info.text === 'string'
            ? { text: info.text, base: info.text, baseLabel: t('cfg.base_current') }
            : {
                text: tpl ? tpl[1] : '',
                base: tpl ? tpl[1] : '',
                baseLabel: tpl ? t('cfg.base_template', { name: tpl[0] }) : t('cfg.base_empty'),
              };
      }
      this.render();
    } catch (e) {
      if (this._alive && this.file === file) clear(this.body).append(errorBox(e, () => this.open()));
    }
  }

  get draft() {
    return this.drafts[this.file];
  }
  get info() {
    return this.infos[this.file];
  }

  /** Replace the draft (and the diff base) from a source the user picked. */
  load(text, label) {
    this.drafts[this.file] = { text, base: text, baseLabel: label };
    this.result = null;
    this.render();
    toast(t('cfg.loaded', { what: label }), { kind: 'ok', timeout: 1800 });
  }

  /** The draft text changed (Form or Text): keep the other view, the live check and the export in step. */
  setText(text, { rerenderForm = false } = {}) {
    this.draft.text = text;
    this.result = null;
    if (this.textArea && this.textArea.value !== text) this.textArea.value = text;
    this.paintLive();
    this.paintResult();
    this.paintExport();
    if (rerenderForm && this.view === 'form') this.paintForm();
  }

  render() {
    const file = this.file,
      info = this.info,
      m = meta(file);
    this.liveEl = h('p', { class: 'small', role: 'status', 'aria-live': 'polite' });
    this.resultEl = h('div', { class: 'stack-v cfg-result', 'aria-live': 'polite' });
    this.exportEl = h('div', { class: 'stack-v' });
    this.formEl = h('div', { class: 'stack-v' });
    this.textArea = h('textarea', {
      class: 'cfg-text',
      rows: '22',
      spellcheck: 'false',
      autocomplete: 'off',
      'aria-label': t('cfg.text_label', { name: m.name }),
      value: this.draft.text,
      onInput: (e) => {
        this.draft.text = e.target.value;
        this.result = null;
        clearTimeout(this._typing);
        this._typing = setTimeout(() => {
          if (this._alive) {
            this.paintLive();
            this.paintResult();
            this.paintExport();
          }
        }, 250);
      },
    });
    this.textArea.addEventListener('keydown', (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 's') {
        e.preventDefault();
        this.validate();
      }
    });
    const panel = h('div', { id: 'cfgview-panel', role: 'tabpanel', class: 'stack-v' });
    const showView = (v) => {
      this.view = v;
      panel.setAttribute('aria-labelledby', `cfgview-${v}`);
      if (v === 'form') {
        this.paintForm();
        clear(panel).append(this.formEl);
      } else clear(panel).append(this.textArea);
    };
    const views = miniTabs(
      [
        { id: 'form', label: t('cfg.view_form') },
        { id: 'text', label: t('cfg.view_text') },
      ],
      this.view,
      showView,
      t('cfg.views'),
      'cfgview'
    );
    append(clear(this.body), [
      this.sourceCard(file, info),
      card(
        t('cfg.draft_title', { name: t(`cfg.file_${file}`) }),
        h('div', { class: 'stack-v' }, views, panel, this.liveEl),
        null,
        { tip: infoTip({ kind: 'concept', key: fileConcept(file), title: m.name }) }
      ),
      card(
        t('cfg.check_title'),
        h(
          'div',
          { class: 'stack-v' },
          h(
            'div',
            { class: 'row gap wrap' },
            btn(t('cfg.validate'), { icon: 'check', kind: 'primary', onClick: () => this.validate() }),
            h('span', { class: 'hint' }, t('cfg.validate_hint', { what: t(`cfg.checker_${file}`) }))
          ),
          this.resultEl
        )
      ),
      card(t('cfg.export_title'), this.exportEl),
    ]);
    showView(this.view);
    this.paintLive();
    this.paintResult();
    this.paintExport();
  }

  sourceCard(file, info) {
    const m = meta(file);
    const fileIn = h('input', {
      type: 'file',
      class: 'sr-only',
      accept: file === 'env' ? '' : '.json,application/json',
      'aria-label': t('cfg.open_local'),
      onChange: (e) => this.readLocal(e.target),
    });
    const loaders = [
      info.served
        ? btn(t('cfg.load_current'), { icon: 'download', onClick: () => this.load(info.text, t('cfg.base_current')) })
        : null,
      ...Object.entries(info.templates || {}).map(([name, text]) =>
        btn(t('cfg.load_template', { name: t(`cfg.tpl_${name}`) }), {
          onClick: () => this.load(text, t('cfg.base_template', { name })),
        })
      ),
      btn(t('cfg.open_local'), { icon: 'folder', onClick: () => fileIn.click() }),
      fileIn,
      btn(t('cfg.start_empty'), {
        onClick: () =>
          this.load(
            file === 'env'
              ? ''
              : file === 'hosts'
                ? toText({ hosts: [{ id: 'local' }] })
                : toText({ version: 1, mode: 'monitor' }),
            t('cfg.base_empty')
          ),
      }),
    ];
    const where =
      file === 'policy'
        ? h(
            'div',
            { class: 'stack-v tight' },
            h('span', t('cfg.policy_where')),
            h(
              'ul',
              { class: 'plain-list' },
              (info.locations || []).map((p) => h('li', h('code', p)))
            ),
            info.node_policy
              ? h('span', { class: 'hint' }, t('cfg.policy_local'), ' ', h('code', info.node_policy))
              : null
          )
        : h(
            'div',
            { class: 'row gap wrap' },
            h('code', { class: 'cfg-path' }, info.path || '–'),
            badge(info.exists ? t('cfg.exists') : t('cfg.not_there'), info.exists ? 'ok' : 'muted')
          );
    const reasonKind =
      { missing: 'info', refused: 'bad', invalid: 'bad', secret: 'warn', remote: 'info' }[info.reason] || 'info';
    return card(
      t('cfg.source_title', { name: t(`cfg.file_${file}`) }),
      h(
        'div',
        { class: 'stack-v' },
        where,
        info.served
          ? h('p', { class: 'note ok' }, t('cfg.served'))
          : h('p', { class: `note ${reasonKind}` }, t(`cfg.reason_${file}_${info.reason}`)),
        info.problem ? h('pre', { class: 'cmd cfg-problem' }, info.problem) : null,
        h('div', { class: 'row gap wrap' }, loaders),
        h('p', { class: 'hint' }, t('cfg.local_only'))
      ),
      null,
      { tip: infoTip({ kind: 'concept', key: fileConcept(file), title: m.name }) }
    );
  }

  readLocal(input) {
    const f = input.files?.[0];
    input.value = '';
    if (!f) return;
    const max = this.info.max_bytes || 1024 * 1024;
    if (f.size > max) {
      toast(t('cfg.too_big', { name: f.name, max: Math.round(max / 1024) }), { kind: 'bad' });
      return;
    }
    const r = new FileReader();
    r.onload = () => {
      if (this._alive) this.load(String(r.result ?? ''), t('cfg.base_local', { name: f.name }));
    };
    r.onerror = () => toast(t('cfg.read_failed', { name: f.name }), { kind: 'bad' });
    r.readAsText(f);
  }

  /** Why the hub would refuse to check this draft at all (over its size or line cap), as a message, or null. */
  overLimit() {
    const text = this.draft.text,
      info = this.info || {};
    const max = info.max_bytes || 1024 * 1024,
      bytes = utf8Bytes(text);
    if (bytes > max) return t('cfg.too_big_draft', { size: Math.ceil(bytes / 1024), max: Math.round(max / 1024) });
    if (info.max_lines) {
      const lines = text.split('\n').length - (text.endsWith('\n') ? 1 : 0);
      if (lines > info.max_lines) return t('cfg.too_many_lines', { lines, max: info.max_lines });
    }
    return null;
  }

  // ---------- live check (in the browser; the hub's check decides) ----------
  paintLive() {
    if (!this.liveEl) return;
    const m = meta(this.file);
    const over = this.overLimit();
    if (over) {
      this.liveEl.className = 'small bad-text';
      this.liveEl.textContent = over;
      return;
    }
    if (m.format === 'json') {
      const p = parseJson(this.draft.text);
      this.liveEl.className = `small ${p.ok ? 'muted' : 'bad-text'}`;
      this.liveEl.textContent = p.ok
        ? t('cfg.live_json_ok')
        : t('cfg.live_json_bad', { msg: p.message, line: p.line ?? '?' });
    } else {
      const lines = parseLines(this.draft.text);
      const other = lines.filter((l) => l.kind === 'other');
      const long = other.filter((l) => l.long);
      this.liveEl.className = `small ${other.length ? 'warn-text' : 'muted'}`;
      this.liveEl.textContent = long.length
        ? t('cfg.live_env_long', {
            lines: long
              .map((l) => l.n)
              .slice(0, 8)
              .join(', '),
          })
        : other.length
          ? t('cfg.live_env_other', {
              lines: other
                .map((l) => l.n)
                .slice(0, 8)
                .join(', '),
            })
          : t('cfg.live_env_ok', { count: lines.filter((l) => l.kind === 'pair').length });
    }
  }

  // ---------- validate (POST /config/validate) ----------
  async validate() {
    const file = this.file,
      text = this.draft.text;
    const over = this.overLimit();
    if (over) {
      toast(over, { kind: 'bad' });
      return;
    }
    clear(this.resultEl).append(skeleton(2));
    try {
      const r = await this.api.validateConfig(file, text);
      if (!this._alive || this.file !== file || this.draft.text !== text) return;
      this.result = r;
    } catch (e) {
      if (!this._alive || this.file !== file) return;
      this.result = null;
      clear(this.resultEl).append(errorBox(e, () => this.validate()));
      return;
    }
    this.paintResult();
    this.paintExport();
  }

  jumpTo(line) {
    const showText = () => document.getElementById('cfgview-text')?.click();
    if (this.view !== 'text') showText();
    const ta = this.textArea,
      lines = ta.value.split('\n');
    const start = lines.slice(0, Math.max(0, line - 1)).reduce((n, l) => n + l.length + 1, 0);
    ta.focus();
    ta.setSelectionRange(start, start + (lines[line - 1]?.length ?? 0));
  }

  paintResult() {
    if (!this.resultEl) return;
    const r = this.result;
    if (!r) {
      clear(this.resultEl).append(h('p', { class: 'muted small' }, t('cfg.not_checked')));
      return;
    }
    const item = (d, kind) =>
      h(
        'li',
        { class: `cfg-diag ${kind}` },
        badge(t(`cfg.${kind}`), kind === 'error' ? 'bad' : 'warn'),
        ' ',
        h('span', d.message),
        d.line
          ? [
              ' ',
              h(
                'button',
                { type: 'button', class: 'btn ghost sm', onClick: () => this.jumpTo(d.line) },
                t('cfg.line', { line: d.line })
              ),
            ]
          : null,
        d.field ? [' ', infoTip({ kind: 'concept', key: fieldConcept(this.file, d.field), title: d.field })] : null
      );
    const policy = this.file === 'policy';
    append(clear(this.resultEl), [
      h(
        'p',
        { class: `note ${r.ok ? 'ok' : 'bad'}`, role: 'status' },
        h('strong', r.ok ? t(policy ? 'cfg.ok_policy' : 'cfg.ok') : t('cfg.not_ok', { count: r.errors.length })),
        ' ',
        r.ok ? t(policy ? 'cfg.ok_policy_hint' : 'cfg.ok_hint') : t('cfg.not_ok_hint')
      ),
      r.errors.length || r.warnings.length
        ? h(
            'ul',
            { class: 'plain-list cfg-diags' },
            r.errors.map((d) => item(d, 'error')),
            r.warnings.map((d) => item(d, 'warning'))
          )
        : null,
      r.ok && r.outline ? this.outline(r.outline) : null,
    ]);
  }

  outline(o) {
    if (this.file === 'hosts') {
      return h(
        'ul',
        { class: 'plain-list small' },
        (o.hosts || []).map((x) =>
          h(
            'li',
            h('code', x.id),
            ` ${x.label} · `,
            x.remote ? t('cfg.via_ssh') : t('cfg.this_machine'),
            x.enabled ? '' : ` · ${t('cfg.disabled')}`
          )
        )
      );
    }
    if (this.file === 'env') {
      return h(
        'p',
        { class: 'small muted' },
        (o.set || []).length ? t('cfg.env_sets', { names: o.set.join(', ') }) : t('cfg.env_sets_none')
      );
    }
    return h(
      'p',
      { class: 'small muted' },
      t('cfg.policy_outline', {
        mode: o.mode,
        roots: o.model_roots,
        writable: o.writable_roots,
        images: o.engine_images,
        scripts: o.scripts,
      })
    );
  }

  // ---------- export ----------
  paintExport() {
    if (!this.exportEl) return;
    const m = meta(this.file),
      d = this.draft,
      info = this.info;
    const ops = lineDiff(d.base, d.text);
    const diff = ops === null ? null : diffText(ops);
    const diffEl =
      diff === null
        ? h('p', { class: 'hint' }, t('cfg.diff_too_big'))
        : diff === ''
          ? h('p', { class: 'muted small' }, t('cfg.no_changes', { base: d.baseLabel }))
          : h(
              'pre',
              { class: 'cmd cfg-diff', tabindex: '0', 'aria-label': t('cfg.diff_label', { base: d.baseLabel }) },
              diff
                .split('\n')
                .map((l) => h('span', { class: l.startsWith('+') ? 'add' : l.startsWith('-') ? 'del' : '' }, `${l}\n`))
            );
    const r = this.result;
    const target = this.file === 'policy' ? (info.locations || [])[0] : info.path;
    // A draft over the hub's caps is not scanned for secrets (nor shown in the Form view): it cannot be checked anyway.
    const over = this.overLimit();
    const pairs = !over && m.format === 'dotenv' ? parseLines(d.text).filter((l) => l.kind === 'pair') : [];
    const secret = !over && looksSecret(d.text, pairs);
    append(clear(this.exportEl), [
      h(
        'div',
        { class: 'stack-v tight' },
        h('strong', { class: 'small' }, t('cfg.diff_title', { base: d.baseLabel })),
        diffEl
      ),
      r && !r.ok
        ? h('p', { class: 'note warn' }, t('cfg.export_invalid'))
        : !r
          ? h('p', { class: 'note info' }, t('cfg.export_unchecked'))
          : null,
      secret ? h('p', { class: 'note warn', role: 'note' }, t('cfg.export_secret')) : null,
      over ? h('p', { class: 'note warn', role: 'note' }, over, ' ', t('cfg.export_over_limit')) : null,
      h(
        'div',
        { class: 'row gap wrap' },
        btn(t('common.copy'), {
          icon: 'copy',
          onClick: async () => {
            const ok = await copyText(d.text);
            toast(ok ? t('common.copied') : t('common.copy_failed'), { kind: ok ? 'ok' : 'bad', timeout: 1800 });
          },
        }),
        btn(t('cfg.download', { name: m.name }), {
          icon: 'download',
          onClick: () => download(m.name, d.text, m.format === 'json' ? 'application/json' : 'text/plain'),
        })
      ),
      h('p', { class: 'hint' }, t(`cfg.install_hint_${this.file}`)),
      cmdBlock(installCommands(this.file, { path: target, download: m.name }), t('cfg.install_cmds')),
    ]);
  }

  // ---------- form views ----------
  paintForm() {
    const m = meta(this.file);
    const over = this.overLimit();
    if (over) {
      append(clear(this.formEl), [h('p', { class: 'note bad' }, over, ' ', t('cfg.form_over_limit'))]);
      return;
    }
    if (m.format === 'dotenv') {
      append(clear(this.formEl), [this.envForm()]);
      return;
    }
    const p = parseJson(this.draft.text);
    if (!p.ok) {
      append(clear(this.formEl), [h('p', { class: 'note bad' }, t('cfg.form_needs_json', { msg: p.message }))]);
      return;
    }
    append(clear(this.formEl), [this.file === 'hosts' ? this.hostsForm(p.value) : this.policyForm(p.value)]);
  }

  /** Apply one value to the JSON draft (re-serialised); `rerender` after structural changes. */
  setJson(path, value, rerender = false) {
    const p = parseJson(this.draft.text);
    if (!p.ok) return;
    this.setText(toText(setPath(p.value, path, value)), { rerenderForm: rerender });
  }

  /** A typed control for a JSON field at `path` with its (i) from `concept`. */
  jsonControl(def, doc, path, concept) {
    const cur = getPath(doc, path);
    const id = `cf${++uid}`;
    const err = h('div', { class: 'hint bad-text', id: `${id}-e`, hidden: true });
    const commit = (raw) => {
      const v = fieldValue(def.kind, raw, { nullable: def.nullable });
      if (!v.ok) {
        err.textContent = t(v.why);
        err.hidden = false;
        ctl.setAttribute('aria-invalid', 'true');
        return;
      }
      err.hidden = true;
      ctl.removeAttribute('aria-invalid');
      this.setJson(path, v.value);
    };
    let ctl;
    if (def.kind === 'bool')
      ctl = h('input', {
        type: 'checkbox',
        id,
        checked: cur === undefined ? !!def.dflt : !!cur,
        onChange: (e) => commit(e.target.checked),
      });
    else if (def.kind === 'choice') {
      const opts = [
        ...(cur === undefined ? [{ value: '', label: t('cfg.default') }] : []),
        ...def.options.map((o) => ({ value: o, label: o })),
        ...(cur !== undefined && !def.options.includes(cur) ? [{ value: String(cur), label: String(cur) }] : []),
      ];
      ctl = select(opts, cur === undefined ? '' : String(cur), (v) => commit(v), { id });
    } else if (def.kind === 'lines')
      ctl = h('textarea', {
        id,
        rows: '4',
        spellcheck: 'false',
        value: fieldText(def.kind, cur),
        onChange: (e) => commit(e.target.value),
      });
    else
      ctl = h('input', {
        type: 'text',
        id,
        inputmode: def.kind === 'int' || def.kind === 'float' ? 'decimal' : null,
        spellcheck: 'false',
        autocomplete: 'off',
        value: fieldText(def.kind, cur),
        placeholder: cur === undefined ? t('cfg.unset') : null,
        onChange: (e) => commit(e.target.value),
      });
    ctl.setAttribute('aria-describedby', `${id}-e`);
    const label = def.key;
    const f = field(
      label,
      ctl,
      def.kind === 'lines'
        ? t('cfg.one_per_line')
        : def.kind === 'ints' || def.kind === 'pair'
          ? t('cfg.comma_list')
          : null,
      { kind: 'concept', key: concept, inline: false }
    );
    f.classList.add('cfg-field', `kind-${def.kind}`);
    f.append(err);
    return f;
  }

  extraKeys(obj, known) {
    const extra = Object.keys(obj || {}).filter((k) => !known.includes(k) && !k.startsWith('_'));
    return extra.length ? h('p', { class: 'hint' }, t('cfg.extra_keys', { keys: extra.join(', ') })) : null;
  }

  hostsForm(doc) {
    const list = Array.isArray(doc?.hosts) ? doc.hosts : null;
    if (!list) return h('p', { class: 'note bad' }, t('cfg.hosts_shape'));
    const cards = list.map((host, i) =>
      h(
        'fieldset',
        { class: 'cfg-group' },
        h('legend', h('span', `hosts[${i}]`), host?.id ? [' ', h('code', String(host.id))] : null),
        h(
          'div',
          { class: 'cfg-grid' },
          HOST_FIELDS.map((d) => this.jsonControl(d, doc, ['hosts', i, d.key], fieldConcept('hosts', d.key)))
        ),
        this.extraKeys(
          host,
          HOST_FIELDS.map((d) => d.key)
        ),
        h(
          'div',
          { class: 'row' },
          btn(t('cfg.remove_host'), {
            icon: 'trash',
            size: 'sm',
            onClick: () => this.setJson(['hosts', i], undefined, true),
          })
        )
      )
    );
    return h(
      'div',
      { class: 'stack-v' },
      h(
        'div',
        { class: 'row gap' },
        h('strong', 'hosts'),
        infoTip({ kind: 'concept', key: fieldConcept('hosts', 'hosts'), title: 'hosts' }),
        h('span', { class: 'hint' }, t('cfg.hosts_count', { count: list.length }))
      ),
      cards.length ? cards : emptyBox(t('cfg.hosts_empty')),
      h(
        'div',
        { class: 'row' },
        btn(t('cfg.add_host'), {
          icon: 'plus',
          onClick: () =>
            this.setJson(['hosts', list.length], { id: `host-${list.length + 1}`, ssh: 'user@192.0.2.10' }, true),
        })
      )
    );
  }

  policyForm(doc) {
    if (!doc || typeof doc !== 'object' || Array.isArray(doc))
      return h('p', { class: 'note bad' }, t('cfg.policy_shape'));
    const top = POLICY_FIELDS.map((d) => this.jsonControl(d, doc, d.key.split('.'), fieldConcept('policy', d.key)));
    const roots = Array.isArray(doc.model_roots) ? doc.model_roots : [];
    const scripts = Array.isArray(doc.scripts) ? doc.scripts : [];
    return h(
      'div',
      { class: 'stack-v' },
      h(
        'fieldset',
        { class: 'cfg-group' },
        h('legend', t('cfg.policy_main')),
        h('div', { class: 'cfg-grid' }, top),
        this.extraKeys(
          doc,
          [...new Set(POLICY_FIELDS.map((d) => d.key.split('.')[0]))].concat(['model_roots', 'scripts'])
        )
      ),
      h(
        'div',
        { class: 'row gap' },
        h('strong', 'model_roots'),
        infoTip({ kind: 'concept', key: fieldConcept('policy', 'model_roots'), title: 'model_roots' })
      ),
      roots.length
        ? roots.map((root, i) =>
            h(
              'fieldset',
              { class: 'cfg-group' },
              h('legend', h('span', `model_roots[${i}]`), root?.id ? [' ', h('code', String(root.id))] : null),
              h(
                'div',
                { class: 'cfg-grid' },
                ROOT_FIELDS.map((d) =>
                  this.jsonControl(d, doc, ['model_roots', i, d.key], fieldConcept('policy', `model_roots.${d.key}`))
                )
              ),
              this.extraKeys(
                root,
                ROOT_FIELDS.map((d) => d.key)
              ),
              h(
                'div',
                { class: 'row' },
                btn(t('cfg.remove_root'), {
                  icon: 'trash',
                  size: 'sm',
                  onClick: () => this.setJson(['model_roots', i], undefined, true),
                })
              )
            )
          )
        : emptyBox(t('cfg.roots_empty')),
      h(
        'div',
        { class: 'row' },
        btn(t('cfg.add_root'), {
          icon: 'plus',
          onClick: () =>
            this.setJson(
              ['model_roots', roots.length],
              { id: `models${roots.length + 1}`, path: '/data/models', kind: 'auto', writable: false },
              true
            ),
        })
      ),
      h(
        'div',
        { class: 'row gap' },
        h('strong', 'scripts'),
        infoTip({ kind: 'concept', key: fieldConcept('policy', 'scripts'), title: 'scripts' }),
        h('span', { class: 'hint' }, t('cfg.scripts_text', { count: scripts.length }))
      )
    );
  }

  envForm() {
    const text = this.draft.text,
      fields = this.info.fields || [];
    const byVar = new Map(fields.map((f) => [f.var, f]));
    const vals = envValues(text);
    const rows = Object.entries(vals).map(([key, { value, n }]) => {
      const f = byVar.get(key);
      const remove = btn('', {
        icon: 'trash',
        size: 'sm',
        title: t('cfg.remove_setting', { name: key }),
        onClick: () => this.setText(envRemove(this.draft.text, key), { rerenderForm: true }),
      });
      if (!f) {
        return h(
          'div',
          { class: 'cfg-env-row unknown' },
          h(
            'div',
            { class: 'field' },
            h('span', { class: 'lbl' }, h('code', key)),
            h('span', { class: 'hint' }, t('cfg.env_unknown', { line: n }))
          ),
          remove
        );
      }
      const kind = ENV_KIND[f.kind] || 'text';
      const id = `ce${++uid}`;
      const secret = f.name === 'hf_token';
      const set = (v) => this.setText(envSet(this.draft.text, key, v));
      const ctl =
        kind === 'bool'
          ? select(
              [
                ...(['true', 'false'].includes(value.toLowerCase()) ? [] : [{ value, label: value || t('cfg.empty') }]),
                { value: 'true', label: 'true' },
                { value: 'false', label: 'false' },
              ],
              ['true', 'false'].includes(value.toLowerCase()) ? value.toLowerCase() : value,
              set,
              { id }
            )
          : h('input', {
              type: secret ? 'password' : 'text',
              id,
              value,
              spellcheck: 'false',
              autocomplete: 'off',
              inputmode: kind === 'int' || kind === 'float' ? 'decimal' : null,
              onChange: (e) => set(e.target.value),
            });
      const hint = [
        f.no_effect ? t('cfg.env_no_effect') : null,
        kind === 'json' ? t('cfg.env_json') : null,
        secret ? t('cfg.env_secret') : null,
      ]
        .filter(Boolean)
        .join(' ');
      const fl = field(key, ctl, hint || null, { kind: 'concept', key: fieldConcept('env', f.name), inline: false });
      fl.classList.add('cfg-field');
      return h('div', { class: `cfg-env-row${f.no_effect ? ' no-effect' : ''}` }, fl, remove);
    });
    const present = new Set(Object.keys(vals));
    const free = fields.filter((f) => !present.has(f.var));
    const order = [
      ...free.filter((f) => ENV_COMMON.includes(f.name)),
      ...free.filter((f) => !ENV_COMMON.includes(f.name)),
    ];
    const add = select(
      [
        { value: '', label: t('cfg.env_pick') },
        ...order.map((f) => ({ value: f.var, label: f.no_effect ? `${f.var} (${t('cfg.no_effect_short')})` : f.var })),
      ],
      '',
      () => {},
      { 'aria-label': t('cfg.env_add') }
    );
    const comments = parseLines(text).filter((l) => l.kind === 'comment').length;
    return h(
      'div',
      { class: 'stack-v' },
      rows.length ? h('div', { class: 'stack-v' }, rows) : emptyBox(t('cfg.env_none'), t('cfg.env_none_hint')),
      comments ? h('p', { class: 'hint' }, t('cfg.env_comments', { count: comments })) : null,
      h(
        'div',
        { class: 'row gap wrap' },
        add,
        btn(t('cfg.env_add'), {
          icon: 'plus',
          onClick: () => {
            if (add.value) this.setText(envSet(this.draft.text, add.value, ''), { rerenderForm: true });
          },
        })
      )
    );
  }
}
customElements.define('ec-config', EcConfig);
