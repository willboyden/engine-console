// A DOM node's own .append() turns null into the text "null" and an array into "[object ...]". The h() helper and dom.js append()
// filter and flatten, so every conditional child must go through them. This scans for a native .append( call that has a
// conditional-null child, which is how a literal "null" ends up on screen.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const walk = (d) => fs.readdirSync(d, { withFileTypes: true }).flatMap((e) => (e.isDirectory() ? walk(path.join(d, e.name)) : e.name.endsWith('.js') ? [path.join(d, e.name)] : []));

/** Top-level arguments of the call whose "(" is at src[open]; nested brackets, strings and template literals are skipped. */
export function topLevelArgs(src, open) {
  const args = []; let depth = 0, cur = '', i = open + 1;
  const skipString = (q) => { let out = q; i++; while (i < src.length) { const c = src[i]; out += c; if (c === '\\') { out += src[++i] ?? ''; i++; continue; } if (c === q) { i++; return out; } if (q === '`' && c === '$' && src[i + 1] === '{') { let d = 0; do { const x = src[i]; out += x; if (x === '{') d++; if (x === '}') d--; i++; } while (d > 0 && i < src.length); continue; } i++; } return out; };
  for (; i < src.length;) {
    const c = src[i];
    if (c === '"' || c === "'" || c === '`') { cur += depth === 0 ? skipString(c) : (skipString(c), ''); continue; }
    if ('([{'.includes(c)) { depth++; if (depth === 1) cur += c; i++; continue; }
    if (')]}'.includes(c)) { if (depth === 0) { args.push(cur.trim()); return args; } depth--; if (depth === 0) cur += c; i++; continue; }
    if (c === ',' && depth === 0) { args.push(cur.trim()); cur = ''; i++; continue; }
    if (depth === 0) cur += c;
    i++;
  }
  return args;
}
// native append() stringifies null/false/undefined AND arrays; a ternary with an array branch is just as wrong
const risky = (arg) => { const a = arg.replace(/\s+/g, ' '); return !a.startsWith('...') && (/\?[\s\S]*:\s*(null|undefined|false)\s*$/.test(a) || /^\[/.test(a) || /[?:]\s*\[/.test(a)) || /^[\w.]+\s*&&\s*/.test(a); };

test('the scanner understands the shapes it must catch and the ones it must leave alone', () => {
  const find = (code) => { const out = []; for (const m of code.matchAll(/\.append\(/g)) out.push(...topLevelArgs(code, m.index + m[0].length - 1).filter(risky)); return out; };
  assert.equal(find("el.append(h('a'), cond ? h('b', {x: 1}) : null)").length, 1);
  assert.equal(find("el.append(h('a'), ok && h('b'))").length, 1);
  assert.equal(find("el.append(h('a'), ...(cond ? [h('b')] : []))").length, 0, 'a spread of a possibly-empty array is safe');
  assert.equal(find("el.append(h('a', cond ? 'x' : null), `t ${a ? 1 : null}`)").length, 0, 'nested in h() or a template string is fine');
  assert.equal(find("append(el, [cond ? h('b') : null])").length, 0, 'the safe helper is not .append(');
  assert.equal(find("el.append(a ? b : c)").length, 0);
  assert.equal(find("el.append(h('a'), cond ? [h('b'), h('c')] : h('d'))").length, 1, 'an array child is stringified');
  assert.equal(find("el.append([h('a')])").length, 1);
});

test('no native .append( in js/ passes a conditional null/false child (use append() from dom.js or h())', () => {
  const bad = [];
  for (const f of walk(path.join(root, 'js'))) {
    const src = fs.readFileSync(f, 'utf8');
    for (const m of src.matchAll(/\.append\(/g)) {
      for (const a of topLevelArgs(src, m.index + m[0].length - 1).filter(risky)) bad.push(`${path.relative(root, f)}:${src.slice(0, m.index).split('\n').length}: ${a.slice(0, 80)}`);
    }
  }
  assert.deepEqual(bad, []);
});
