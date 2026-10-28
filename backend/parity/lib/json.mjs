// JSON parsing that keeps what JSON.parse throws away: whether a number was written as an integer (`1`) or a float
// (`1.0`, `1e5`). Python's `1.0` must not compare equal to Rust's `1` (COMMON-RULES: "integers stay integers and
// floats stay floats"). Numbers become `Num` objects; everything else is plain JS (objects keep their key order, but
// comparison ignores it).

export class Num {
  constructor(raw) {
    this.raw = raw;
    this.isFloat = /[.eE]/.test(raw);
  }

  /** Value equality within the same kind: floats as IEEE doubles (Python repr and Rust ryu both round-trip, so the
   * same double may be spelled `1e-05` or `1e-5`), integers exactly (BigInt, no 2**53 rounding). */
  equals(o) {
    if (!(o instanceof Num) || o.isFloat !== this.isFloat) return false;
    if (this.isFloat) return Number(this.raw) === Number(o.raw);
    return BigInt(this.raw) === BigInt(o.raw);
  }

  toString() { return this.isFloat ? `${this.raw}(float)` : `${this.raw}(int)`; }
  toJSON() { return Number(this.raw); }
}

export class JsonError extends Error {}

/** Parse JSON text into plain values with `Num` numbers. Throws JsonError on invalid input. */
export function parseJson(text) {
  let i = 0;
  const n = text.length;
  const ws = () => { while (i < n && (text[i] === ' ' || text[i] === '\t' || text[i] === '\n' || text[i] === '\r')) i++; };
  const fail = (m) => { throw new JsonError(`${m} at ${i}`); };
  function value() {
    ws();
    if (i >= n) fail('unexpected end');
    const c = text[i];
    if (c === '{') {
      i++; const out = Object.create(null); ws();
      if (text[i] === '}') { i++; return out; }
      for (;;) {
        ws(); if (text[i] !== '"') fail('expected key');
        const k = string(); ws();
        if (text[i] !== ':') fail('expected :'); i++;
        out[k] = value(); ws();
        if (text[i] === ',') { i++; continue; }
        if (text[i] === '}') { i++; return out; }
        fail('expected , or }');
      }
    }
    if (c === '[') {
      i++; const out = []; ws();
      if (text[i] === ']') { i++; return out; }
      for (;;) {
        out.push(value()); ws();
        if (text[i] === ',') { i++; continue; }
        if (text[i] === ']') { i++; return out; }
        fail('expected , or ]');
      }
    }
    if (c === '"') return string();
    if (text.startsWith('true', i)) { i += 4; return true; }
    if (text.startsWith('false', i)) { i += 5; return false; }
    if (text.startsWith('null', i)) { i += 4; return null; }
    const m = /^-?(0|[1-9]\d*)(\.\d+)?([eE][+-]?\d+)?/.exec(text.slice(i, i + 400));
    if (!m) fail(`unexpected ${JSON.stringify(c)}`);
    i += m[0].length;
    return new Num(m[0]);
  }
  function string() {
    const start = i; i++;
    for (;;) {
      if (i >= n) fail('unterminated string');
      const c = text[i];
      if (c === '\\') { i += 2; continue; }
      if (c === '"') { i++; break; }
      i++;
    }
    return JSON.parse(text.slice(start, i));
  }
  const v = value(); ws();
  if (i !== n) fail('trailing data');
  return v;
}

/** Recursive equality: objects by key set (order ignored), arrays in order, numbers by kind and value. Returns the
 * list of differences as {path, a, b} (at most `limit`). Absent and null are different. */
export function diffValues(a, b, path = '$', out = [], limit = 20) {
  if (out.length >= limit) return out;
  const t = (v) => (v === null ? 'null' : v instanceof Num ? 'num' : Array.isArray(v) ? 'array' : typeof v);
  const ta = t(a); const tb = t(b);
  if (ta !== tb) { out.push({ path, a: show(a), b: show(b) }); return out; }
  if (ta === 'num') { if (!a.equals(b)) out.push({ path, a: show(a), b: show(b) }); return out; }
  if (ta === 'array') {
    if (a.length !== b.length) out.push({ path: `${path}.length`, a: a.length, b: b.length });
    for (let k = 0; k < Math.min(a.length, b.length); k++) diffValues(a[k], b[k], `${path}[${k}]`, out, limit);
    return out;
  }
  if (ta === 'object') {
    const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
    for (const k of [...keys].sort()) {
      if (!(k in a)) out.push({ path: `${path}.${k}`, a: '<absent>', b: show(b[k]) });
      else if (!(k in b)) out.push({ path: `${path}.${k}`, a: show(a[k]), b: '<absent>' });
      else diffValues(a[k], b[k], `${path}.${k}`, out, limit);
      if (out.length >= limit) break;
    }
    return out;
  }
  if (a !== b) out.push({ path, a: show(a), b: show(b) });
  return out;
}

export function show(v) {
  if (v instanceof Num) return v.toString();
  const s = JSON.stringify(v, (k, x) => (x instanceof Num ? x.toString() : x));
  return s === undefined ? String(v) : (s.length > 300 ? `${s.slice(0, 300)}...` : s);
}

/** Deep map over a parsed value: `fn(value, key, parentPath)` may return a replacement (or undefined to recurse). */
export function mapValues(v, fn, key = null, path = '$') {
  const r = fn(v, key, path);
  if (r !== undefined) return r;
  if (Array.isArray(v)) return v.map((x, k) => mapValues(x, fn, k, `${path}[${k}]`));
  if (v && typeof v === 'object' && !(v instanceof Num)) {
    const out = Object.create(null);
    for (const [k, x] of Object.entries(v)) out[k] = mapValues(x, fn, k, `${path}.${k}`);
    return out;
  }
  return v;
}
