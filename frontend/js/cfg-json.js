// JSON drafts of the config editor (hosts.json, a node policy): parse with a line number on error, re-serialise, and
// read or change a value by path. A Form edit re-serialises the whole document (2-space indent, key order kept, so
// `_comment` keys stay where they were); the Text view keeps whatever the user typed. Pure functions, no DOM.

/** {ok: true, value} or {ok: false, message, line?}: the browser's own JSON.parse, with the line worked out. */
export function parseJson(text) {
  try {
    return { ok: true, value: JSON.parse(String(text ?? '')) };
  } catch (e) {
    const msg = String(e.message || e);
    let line = null;
    const lc = /line (\d+) column \d+/.exec(msg);
    const pos = /position (\d+)/.exec(msg);
    if (lc) line = Number(lc[1]);
    else if (pos) line = String(text).slice(0, Number(pos[1])).split('\n').length;
    return { ok: false, message: msg, line };
  }
}

export const toText = (value) => `${JSON.stringify(value, null, 2)}\n`;

const isObj = (x) => x !== null && typeof x === 'object' && !Array.isArray(x);

/** The value at `path` (an array of keys and indices), or undefined. */
export function getPath(doc, path) {
  let cur = doc;
  for (const k of path) {
    if (cur == null || typeof cur !== 'object') return undefined;
    cur = cur[k];
  }
  return cur;
}

/** A copy of `doc` with `path` set to `value`; `undefined` deletes the key (an index is spliced out). Missing objects
 * on the way are created. */
export function setPath(doc, path, value) {
  if (!path.length) return value;
  const [k, ...rest] = path;
  const base = Array.isArray(doc) ? doc.slice() : isObj(doc) ? { ...doc } : typeof k === 'number' ? [] : {};
  if (!rest.length) {
    if (value === undefined) {
      if (Array.isArray(base)) base.splice(k, 1);
      else delete base[k];
    } else base[k] = value;
    return base;
  }
  base[k] = setPath(base[k], rest, value);
  return base;
}
