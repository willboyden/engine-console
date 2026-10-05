import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const en = JSON.parse(fs.readFileSync(path.join(root, 'i18n/en.json'), 'utf8'));
const walk = (d) => fs.readdirSync(d, { withFileTypes: true }).flatMap((e) => (e.isDirectory() ? walk(path.join(d, e.name)) : e.name.endsWith('.js') ? [path.join(d, e.name)] : []));
const files = walk(path.join(root, 'js'));
const ns = new Set(Object.keys(en).map((k) => k.split('.')[0]));

test('every literal t("key") and namespaced key literal used in js/ exists in en.json', () => {
  const missing = [];
  for (const f of files) {
    const src = fs.readFileSync(f, 'utf8');
    const keys = new Set();
    for (const m of src.matchAll(/\bt\(\s*['"`]([a-z_]+\.[a-z_0-9.]+)['"`]/g)) keys.add(m[1]);
    for (const m of src.matchAll(/['"]([a-z_]+\.[a-z_0-9]+)['"]/g)) if (ns.has(m[1].split('.')[0]) && /^(nav|dl|models|bench|inst|arena|usage|settings|audit|launch|chat|dash|metrics|fit|params|common|palette|auth|topbar|fleet|host|library|discover|wizard|script|picker|learn|mode|controls|engines|err|mem)\./.test(m[1])) keys.add(m[1]);
    for (const k of keys) {
      if (/\.(md|json|csv)$/.test(k)) continue; // file names, not keys
      // plural forms live as key_one / key_other
      if (!(k in en) && !(`${k}_one` in en)) missing.push(`${path.relative(root, f)}: ${k}`);
    }
  }
  assert.deepEqual(missing, []);
});

test('dynamic key families are complete', () => {
  const fam = { 'nav.': ['dashboard', 'library', 'discover', 'models', 'downloads', 'launch', 'instances', 'metrics', 'bench', 'chat', 'arena', 'usage', 'learn', 'settings'],
    'group.': ['memory', 'parallelism', 'context', 'quantization', 'scheduling', 'tools_reasoning', 'performance', 'speculative', 'network', 'advanced'],
    'state.': ['stopped', 'starting', 'loading', 'ready', 'stopping', 'failed', 'auth_required', 'unreachable'], 'phase.': ['loading_weights', 'compiling', 'capturing_graphs', 'warming_up', 'ready'],
    'link.': ['connecting', 'up', 'stale', 'down', 'incompatible', 'disabled'], 'mode.': ['manage', 'monitor', 'manage_tip', 'monitor_tip'], 'management.': ['managed', 'script', 'external'],
    'format.': ['hf', 'gguf', 'ollama', 'plain', 'other'], 'root.': ['auto', 'hf-home', 'hf-hub', 'plain', 'gguf', 'ollama', 'other'], 'alert.': ['host_ram_low', 'swap_heavy', 'shmem_high'],
    'fleet.block_': ['unknown', 'connecting', 'up', 'stale', 'down', 'incompatible', 'disabled', 'monitor', 'docker'], 'controls.reason_': ['managed', 'script', 'external'],
    'bench.suite_': ['quick', 'standard', 'full', 'prefix'], 'chat.role_': ['user', 'assistant', 'system'], 'inst.did_': ['start', 'stop', 'restart'], 'models.sort_': ['downloads', 'likes', 'updated'],
    'library.sort_': ['name', 'size', 'format'], 'library.reason_': ['in_use', 'read_only', 'host_down', 'policy_delete', 'not_deletable', 'no_copy', 'everywhere', 'no_target_manage'],
    'script.status_': ['up', 'down', 'unknown', 'error'], 'script.job_': ['running', 'completed', 'failed'], 'params.level_': ['advanced', 'expert'], 'params.level_name_': ['basic', 'advanced', 'expert'],
    'learn.level_': ['basic', 'advanced', 'expert'], 'dl.dialog_fit_': ['fits', 'tight', 'wont_fit'], 'wizard.band_': ['small', 'medium', 'large', 'huge'], 'check.level.': ['ok', 'warn', 'block'],
    'wizard.fit_': ['easy', 'tight', 'wont', 'unknown'], 'wizard.reason_': ['fits', 'tight', 'wont', 'wont_tp', 'blocked', 'unknown'], 'wizard.need_': ['model', 'machine', 'goal'],
    'engines.': ['no_images_allowed', 'image_not_allowed'] };
  for (const [p, ks] of Object.entries(fam)) for (const k of ks) assert.ok(`${p}${k}` in en, `${p}${k}`);
  for (const k of ['model', 'machine', 'goal', 'review']) for (const sfx of ['', '_short', '_intro']) assert.ok(`wizard.step_${k}${sfx}` in en, `wizard.step_${k}${sfx}`);
  assert.ok('wizard.step_launch_short' in en);
  for (const g of ['low-latency', 'balanced', 'max-throughput', 'long-context', 'tool-agent']) for (const sfx of ['title', 'hint']) assert.ok(`goal.${g}.${sfx}` in en, `goal.${g}.${sfx}`);
});

test('every http status the UI is meant to explain has advice (401/403/409/413/415/421/429/503, plus the node statuses)', () => {
  for (const k of ['err.unauthorized', 'err.forbidden', 'err.conflict_state', 'err.too_large', 'err.unsupported_media', 'err.bad_host', 'err.rate_limited', 'err.too_many_streams', 'err.unavailable',
    'err.node_unavailable', 'err.monitor_only', 'err.policy_denied', 'err.no_space', 'err.timeout', 'err.unsupported', 'err.node_failed', 'err.csrf', 'err.network']) assert.ok(k in en, k);
});

test('each preflight check code the mock hub can return has a plain title and a learn concept, and every title key is a concept', async () => {
  const { learnBundle } = await import('../dev/mock-learn.mjs');
  const concepts = Object.keys(learnBundle().concepts).filter((k) => k.startsWith('check.'));
  assert.ok(concepts.length >= 15);
  for (const k of concepts) assert.ok(k in en, `no title for ${k} in en.json (the check list would show a humanised code)`);
});

test('every {placeholder} in a string is a simple word, and plural pairs come in twos', () => {
  for (const [k, v] of Object.entries(en)) { for (const m of v.matchAll(/\{([^}]*)\}/g)) assert.match(m[1], /^\w+$/, `${k}: {${m[1]}}`); }
  // (a key that merely ends in "_other", like mem.seg_other, is not a plural form)
  for (const k of Object.keys(en)) if (k.endsWith('_one')) assert.ok(`${k.slice(0, -4)}_other` in en, `${k} has no _other`);
});

test('no en.json value is empty', () => { for (const [k, v] of Object.entries(en)) assert.ok(typeof v === 'string' && v.length, k); });

test('no source file references an external origin (zero CDN / zero third-party)', () => {
  for (const f of [...files, path.join(root, 'index.html'), ...fs.readdirSync(path.join(root, 'css')).map((x) => path.join(root, 'css', x))]) {
    const src = fs.readFileSync(f, 'utf8');
    // allow the SVG namespace URI and doc links rendered as data (docs_url comes from the backend, not this source)
    const bad = [...src.matchAll(/https?:\/\/[^\s'"`)]+/g)].map((m) => m[0]).filter((u) => !u.startsWith('http://www.w3.org/2000/svg'));
    assert.deepEqual(bad, [], path.relative(root, f));
  }
});

test('ids the server may add without UI copy (a new startup phase, state, alert code) show a readable form, never a missing-key string', async () => {
  const { setMessages } = await import('../js/i18n.js'); setMessages(en);
  const { humanize } = await import('../js/components/ui.js').catch(() => ({}));
  // ui.js builds DOM at call time only, so it can be imported in Node; humanize/tr are pure
  assert.equal(humanize('warming_up'), 'Warming up'); assert.equal(humanize('host-ram-low'), 'Host ram low'); assert.equal(humanize('  x  '), 'X'); assert.equal(humanize(null), ''); assert.equal(humanize(undefined), '');
  const { tr } = await import('../js/components/ui.js');
  assert.equal(tr('phase', 'warming_up'), 'Warming up'); assert.equal(tr('phase', 'brand_new_phase'), 'Brand new phase', 'an unknown phase is humanised');
  assert.equal(tr('state', 'ready'), 'ready'); assert.equal(tr('alert', 'host_ram_low'), 'Low host RAM'); assert.equal(tr('link', 'something_new'), 'Something new');
});
