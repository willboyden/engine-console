// A line diff between the text the editor loaded and the draft, for the Export card. Longest common subsequence on
// lines (the files are small: a hosts file is at most 64 KiB, a policy 256 KiB); past MAX_CELLS the diff is refused
// rather than freezing the tab. Pure, no DOM.
export const MAX_CELLS = 4_000_000;

/** [{op: ' '|'-'|'+', text}] turning `a` into `b`, or null when the texts are too large to compare here. */
export function lineDiff(a, b) {
  const x = String(a ?? '').split('\n'), y = String(b ?? '').split('\n');
  // common head and tail first: most edits touch a few lines
  let s = 0; while (s < x.length && s < y.length && x[s] === y[s]) s++;
  let e = 0; while (e < x.length - s && e < y.length - s && x[x.length - 1 - e] === y[y.length - 1 - e]) e++;
  const xm = x.slice(s, x.length - e), ym = y.slice(s, y.length - e);
  if ((xm.length + 1) * (ym.length + 1) > MAX_CELLS) return null;
  const w = ym.length + 1;
  const L = new Uint32Array((xm.length + 1) * w);
  for (let i = xm.length - 1; i >= 0; i--) for (let j = ym.length - 1; j >= 0; j--) {
    L[i * w + j] = xm[i] === ym[j] ? L[(i + 1) * w + j + 1] + 1 : Math.max(L[(i + 1) * w + j], L[i * w + j + 1]);
  }
  const out = x.slice(0, s).map((text) => ({ op: ' ', text }));
  let i = 0, j = 0;
  while (i < xm.length && j < ym.length) {
    if (xm[i] === ym[j]) { out.push({ op: ' ', text: xm[i] }); i++; j++; }
    else if (L[(i + 1) * w + j] >= L[i * w + j + 1]) out.push({ op: '-', text: xm[i++] });
    else out.push({ op: '+', text: ym[j++] });
  }
  while (i < xm.length) out.push({ op: '-', text: xm[i++] });
  while (j < ym.length) out.push({ op: '+', text: ym[j++] });
  for (const text of x.slice(x.length - e)) out.push({ op: ' ', text });
  return out;
}

/** The diff as unified-style text with `context` unchanged lines around each change ('' when nothing changed). */
export function diffText(ops, context = 3) {
  if (!ops || !ops.some((o) => o.op !== ' ')) return '';
  const keep = new Set();
  ops.forEach((o, i) => { if (o.op !== ' ') for (let k = Math.max(0, i - context); k <= Math.min(ops.length - 1, i + context); k++) keep.add(k); });
  const out = [];
  let gap = false;
  ops.forEach((o, i) => { if (keep.has(i)) { if (gap) out.push('…'); gap = false; out.push(`${o.op} ${o.text}`); } else gap = true; });
  if (gap) out.push('…');
  return out.join('\n');
}
