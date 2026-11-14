// The hub env file (dotenv) as the config editor edits it: line by line, so comments, blank lines, order and any
// line the editor does not understand stay exactly as written. Reading a value follows the hub's own reader
// (backend/crates/ec-hub/src/config.rs parse_dotenv: `export ` prefix, `#` comments, single quotes literal, double
// quotes with \n \t \r \" \' \\ escapes, an unquoted value ends at " #"), without ${VAR} expansion: the editor shows
// what is written. Pure functions, no DOM.

// Linear: the key runs greedily to the first `=` (it cannot pass one) and is trimmed afterwards, so a long line without
// `=` costs one scan, not one per character (a lazy key followed by `\s*=` was quadratic).
const ASSIGN = /^(\s*)(export\s+)?([^=#\s][^=]*)=(.*)$/;
/** A line longer than this is not parsed (kind 'other', `long`): real settings are far shorter. */
export const MAX_LINE = 4096;

/** One parsed line: {n (1-based), raw, kind: 'blank'|'comment'|'pair'|'other', key?, value?, exported?, long?}. */
export function parseLines(text) {
  return String(text ?? '')
    .split('\n')
    .map((raw, i) => {
      const n = i + 1;
      if (raw.length > MAX_LINE) return { n, raw, kind: 'other', long: true };
      const line = raw.replace(/\r$/, ''),
        lead = line.trimStart();
      if (!lead) return { n, raw, kind: 'blank' };
      if (lead.startsWith('#')) return { n, raw, kind: 'comment' };
      const m = ASSIGN.exec(line);
      if (!m) return { n, raw, kind: 'other' };
      return { n, raw, kind: 'pair', key: m[3].trim(), exported: !!m[2], value: readValue(m[4]) };
    });
}

/** The value of the text after `=`, read as the hub reads it (no expansion). */
export function readValue(rest) {
  const r = String(rest).trimStart();
  if (r.startsWith("'")) {
    const end = r.indexOf("'", 1);
    return end < 0 ? r.slice(1) : r.slice(1, end);
  }
  if (r.startsWith('"')) {
    let out = '';
    for (let i = 1; i < r.length; i++) {
      const c = r[i];
      if (c === '"') break;
      if (c === '\\') {
        const nx = r[++i];
        const esc = { n: '\n', t: '\t', r: '\r', '"': '"', "'": "'", '\\': '\\' }[nx];
        if (nx === undefined) out += '\\';
        else if (esc !== undefined) out += esc;
        else out += `\\${nx}`;
        continue;
      }
      out += c;
    }
    return out;
  }
  const hash = r.indexOf(' #');
  return (hash < 0 ? r : r.slice(0, hash)).trimEnd();
}

/** How `value` is written so the hub reads it back unchanged: bare when that is safe, else quoted. */
export function formatValue(value) {
  const v = String(value ?? '');
  if (v === '') return '';
  const bareOk = !/^['"]/.test(v) && !v.includes(' #') && !/[\n\r\t]/.test(v) && v === v.trim() && !v.includes('${');
  if (bareOk) return v;
  if (!v.includes("'") && !/[\n\r]/.test(v)) return `'${v}'`;
  return `"${v.replace(/\\/g, '\\\\').replace(/"/g, '\\"').replace(/\n/g, '\\n').replace(/\r/g, '\\r').replace(/\t/g, '\\t')}"`;
}

/** {key: {value, n}} of the assignments, the last one of a key winning (as in the hub). */
export function values(text) {
  const out = {};
  for (const l of parseLines(text)) if (l.kind === 'pair') out[l.key] = { value: l.value, n: l.n };
  return out;
}

/** The text with `key` set to `value`: the last assignment of the key is rewritten in place (keeping `export `),
 * or a new line is appended. Every other line is untouched. */
export function setValue(text, key, value) {
  const src = String(text ?? '');
  const lines = src.split('\n');
  const parsed = parseLines(src);
  const last = [...parsed].reverse().find((l) => l.kind === 'pair' && l.key === key);
  const line = (exported) => `${exported ? 'export ' : ''}${key}=${formatValue(value)}`;
  if (last) {
    lines[last.n - 1] = line(last.exported);
    return lines.join('\n');
  }
  if (src === '') return `${line(false)}\n`;
  const body = src.endsWith('\n') ? src : `${src}\n`;
  return `${body}${line(false)}\n`;
}

/** The text without any assignment of `key`. */
export function removeKey(text, key) {
  const parsed = parseLines(text);
  return String(text ?? '')
    .split('\n')
    .filter((_, i) => !(parsed[i].kind === 'pair' && parsed[i].key === key))
    .join('\n');
}
