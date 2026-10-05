// Static guards for the strict CSP (default-src 'none'; script-src 'self'; style-src 'self') and for untrusted text: nothing
// inline, nothing evaluated, and HTML only ever comes from the two escaping generators (markdown.js, charts.js).
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const walk = (d) => fs.readdirSync(d, { withFileTypes: true }).flatMap((e) => (e.isDirectory() ? walk(path.join(d, e.name)) : e.name.endsWith('.js') ? [path.join(d, e.name)] : []));
const files = walk(path.join(root, 'js')).map((f) => [path.relative(root, f), fs.readFileSync(f, 'utf8')]);
// strip comments so prose that mentions a forbidden word does not trip the scan
const code = (s) => s.replace(/\/\*[\s\S]*?\*\//g, '').replace(/(^|[^:'"`])\/\/.*$/gm, '$1');

test('no eval, Function constructor, string timers, javascript: URLs or inline handlers in any module', () => {
  const bad = [];
  for (const [f, src] of files) {
    const c = code(src);
    if (/\beval\s*\(/.test(c)) bad.push(`${f}: eval`);
    if (/new\s+Function\s*\(/.test(c)) bad.push(`${f}: new Function`);
    if (/set(Timeout|Interval)\s*\(\s*['"`]/.test(c)) bad.push(`${f}: string timer`);
    if (/javascript:/i.test(c)) bad.push(`${f}: javascript: url`);
    if (/\bon[a-z]+\s*=\s*["']/.test(c.replace(/\bon[A-Z]\w*\s*:/g, ''))) bad.push(`${f}: inline handler attribute`);
  }
  assert.deepEqual(bad, []);
});

test('no inline styles: no style attribute or <style> in markup strings, no cssText, no setAttribute("style")', () => {
  const bad = [];
  for (const [f, src] of files) {
    const c = code(src);
    if (/\bstyle\s*=\s*["'`\\]/.test(c)) bad.push(`${f}: style= in markup`);
    if (/<style[\s>]/i.test(c)) bad.push(`${f}: <style>`);
    if (/\.cssText\b/.test(c)) bad.push(`${f}: cssText`);
    if (/setAttribute\(\s*['"]style['"]/.test(c)) bad.push(`${f}: setAttribute style`);
    if (/['"]style['"]\s*:/.test(c)) bad.push(`${f}: style attribute via h()`);
  }
  assert.deepEqual(bad, []);
});

test('index.html: every script is an external module or classic file, nothing inline, no style attributes', () => {
  const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
  for (const m of html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/g)) { assert.match(m[1], /\bsrc="/, 'script without src'); assert.equal(m[2].trim(), '', 'inline script body'); }
  assert.doesNotMatch(html, /<style\b/); assert.doesNotMatch(html, /\sstyle\s*=/); assert.doesNotMatch(html, /\son[a-z]+\s*=/);
  assert.match(html, /css\/responsive\.css/); assert.ok(html.indexOf('css/responsive.css') > html.indexOf('css/components.css'), 'the responsive tiers load after the component rules they override');
});

test('HTML strings are only ever produced by the escaping generators: setTrustedHtml callers import markdown.js or charts.js', () => {
  const bad = [];
  for (const [f, src] of files) {
    if (f.endsWith('js/dom.js')) continue;
    const c = code(src);
    if (/\.(innerHTML|outerHTML)\s*=/.test(c) || /insertAdjacentHTML|document\.write/.test(c)) bad.push(`${f}: raw HTML sink`);
    if (/setTrustedHtml\(/.test(c) && !/from '\.\.?\/(\.\.\/)?(markdown|charts)\.js'/.test(c)) bad.push(`${f}: setTrustedHtml without markdown.js/charts.js`);
  }
  assert.deepEqual(bad, []);
});

test('the fleet and learn modules, which show node and bundle text, never touch an HTML sink at all', () => {
  const guarded = ['js/learn.js', 'js/fleet.js', 'js/library.js', 'js/controls.js', 'js/wizard.js', 'js/engines.js', 'js/model-ref.js', 'js/components/info-tip.js', 'js/components/host-card.js', 'js/components/fleet-bars.js', 'js/components/host-root-picker.js',
    'js/components/download-dialog.js', 'js/components/instance-controls.js', 'js/components/script-actions.js', 'js/components/check-list.js', 'js/views/host.js', 'js/views/library.js', 'js/views/wizard.js', 'js/views/learn.js', 'js/views/launch.js', 'js/views/instances.js', 'js/views/downloads.js'];
  for (const f of guarded) {
    const src = files.find(([n]) => n === f)?.[1];
    assert.ok(src, `${f} exists`);
    assert.doesNotMatch(code(src), /setTrustedHtml|innerHTML|outerHTML|insertAdjacentHTML/, f);
  }
});

test('the sign-in form is shaped for a password manager: user name (autocomplete=username) and key (current-password)', () => {
  const app = fs.readFileSync(path.join(root, 'js/app.js'), 'utf8');
  assert.match(app, /autocomplete: 'username'/); assert.match(app, /autocomplete: 'current-password'/); assert.match(app, /type: 'password'/); assert.match(app, /AUTH_USER = 'engine-console'/);
  assert.match(app, /session|setApiKey/); assert.doesNotMatch(app, /localStorage/, 'the key is never written to localStorage here');
  const ctx = fs.readFileSync(path.join(root, 'js/app-context.js'), 'utf8');
  assert.match(ctx, /session\.set\('ec\.apiKey'/, 'the key stays in sessionStorage');
});
