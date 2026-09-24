// Hand-rolled SVG charts returning strings (labels are escaped; styling is via CSS classes so a strict
// style-src CSP works). Everything numeric is formatted here; nothing untrusted is interpolated raw.
import { escapeHtml as esc } from './markdown.js';

export const scaleLinear = (d0, d1, r0, r1) => (v) => (d1 === d0 ? (r0 + r1) / 2 : r0 + ((v - d0) / (d1 - d0)) * (r1 - r0));

export function niceTicks(min, max, count = 5) {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [0, 1];
  if (max === min) { max = min + 1; }
  const raw = (max - min) / Math.max(1, count);
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) || 10 * mag;
  const out = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

const n1 = (v) => (Math.round(v * 10) / 10).toString();
export function linePath(pts) {
  return pts.map(([x, y], i) => `${i ? 'L' : 'M'}${n1(x)} ${n1(y)}`).join('');
}

export function sparkline(values, { w = 120, h = 28, cls = 's1' } = {}) {
  const v = (values || []).filter((x) => Number.isFinite(x));
  if (v.length < 2) return `<svg class="spark" viewBox="0 0 ${w} ${h}" role="img" aria-label="no data"></svg>`;
  const lo = Math.min(...v), hi = Math.max(...v);
  const x = scaleLinear(0, v.length - 1, 1, w - 1), y = scaleLinear(lo, hi === lo ? lo + 1 : hi, h - 2, 2);
  const pts = v.map((val, i) => [x(i), y(val)]);
  return `<svg class="spark" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" aria-hidden="true"><path class="area ${cls}" d="${linePath(pts)}L${w - 1} ${h}L1 ${h}Z"/><path class="line ${cls}" d="${linePath(pts)}"/></svg>`;
}

/**
 * series: [{name, points:[[x,y],...]}]; series i uses class s{i%6+1}.
 * opts: width,height,xFormat,yFormat,yMin,yMax,ariaLabel
 */
export function lineChart(series, opts = {}) {
  const { width = 640, height = 220, xFormat = String, yFormat = (v) => String(v), yMin, yMax, ariaLabel = 'chart' } = opts;
  const m = { l: 46, r: 12, t: 10, b: 24 };
  const all = series.flatMap((s) => s.points).filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y));
  if (!all.length) return `<svg class="chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(ariaLabel)}"><text class="empty" x="${width / 2}" y="${height / 2}" text-anchor="middle">No data</text></svg>`;
  const xs = all.map((p) => p[0]), ys = all.map((p) => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const lo = yMin ?? Math.min(0, ...ys);
  const ticks = niceTicks(lo, yMax ?? Math.max(...ys), 4);
  const y0 = ticks[0], y1 = ticks[ticks.length - 1];
  const X = scaleLinear(x0, x1, m.l, width - m.r), Y = scaleLinear(y0, y1, height - m.b, m.t);
  let g = '';
  for (const tk of ticks) g += `<line class="gl" x1="${m.l}" x2="${width - m.r}" y1="${n1(Y(tk))}" y2="${n1(Y(tk))}"/><text class="tick" x="${m.l - 6}" y="${n1(Y(tk) + 4)}" text-anchor="end">${esc(yFormat(tk))}</text>`;
  const xt = niceTicks(x0, x1, 5);
  for (const tk of xt) g += `<text class="tick" x="${n1(X(tk))}" y="${height - 6}" text-anchor="middle">${esc(xFormat(tk))}</text>`;
  const step = Math.max(1, Math.ceil(Math.max(...series.map((s) => s.points.length)) / 60));
  let body = '';
  series.forEach((s, i) => {
    const cls = `s${(i % 6) + 1}`;
    const pts = s.points.filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y)).map(([x, y]) => [X(x), Y(y), x, y]);
    if (!pts.length) return;
    body += `<path class="line ${cls}" d="${linePath(pts)}"/>`;
    // Hover targets carry native <title> tooltips (keyboard-inert, so the chart also ships a legend and data summary).
    pts.forEach((p, k) => { if (k % step === 0 || k === pts.length - 1) body += `<circle class="hit ${cls}" cx="${n1(p[0])}" cy="${n1(p[1])}" r="5"><title>${esc(s.name)}: ${esc(yFormat(p[3]))} @ ${esc(xFormat(p[2]))}</title></circle>`; });
  });
  return `<svg class="chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(ariaLabel)}">${g}${body}</svg>`;
}

export function legend(series) {
  return `<ul class="legend">${series.map((s, i) => `<li><i class="sw s${(i % 6) + 1}"></i>${esc(s.name)}</li>`).join('')}</ul>`;
}

/** grid[dow][hour] -> number; cell intensity via 5 discrete classes (also in <title>, so not colour-only). */
export function heatmap(grid, { cellW = 22, cellH = 20, dayLabels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], label = 'requests' } = {}) {
  const max = Math.max(1, ...grid.flat());
  const ox = 34, oy = 18, w = ox + 24 * cellW + 4, h = oy + grid.length * cellH + 4;
  let s = `<svg class="heatmap" viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(label)} by weekday and hour">`;
  for (let hr = 0; hr < 24; hr += 3) s += `<text class="tick" x="${ox + hr * cellW + cellW / 2}" y="12" text-anchor="middle">${hr}</text>`;
  grid.forEach((row, d) => {
    s += `<text class="tick" x="${ox - 6}" y="${oy + d * cellH + cellH / 2 + 4}" text-anchor="end">${esc(dayLabels[d] ?? d)}</text>`;
    row.forEach((v, hr) => {
      const lvl = v <= 0 ? 0 : Math.min(4, 1 + Math.floor((v / max) * 3.999));
      s += `<rect class="cell l${lvl}" x="${ox + hr * cellW}" y="${oy + d * cellH}" width="${cellW - 2}" height="${cellH - 2}" rx="3"><title>${esc(dayLabels[d] ?? d)} ${String(hr).padStart(2, '0')}:00 — ${v} ${esc(label)}</title></rect>`;
    });
  });
  return s + '</svg>';
}

/** Horizontal bars: rows [{label, value, cls?}] */
export function hbars(rows, { width = 520, rowH = 26, fmt = String } = {}) {
  const max = Math.max(1e-9, ...rows.map((r) => r.value || 0));
  const lw = Math.min(300, Math.max(120, Math.max(0, ...rows.map((r) => String(r.label).length)) * 6.4)), h = rows.length * rowH + 4;
  return `<svg class="chart" viewBox="0 0 ${width} ${h}" role="img" aria-label="bar chart">${rows.map((r, i) => {
    const bw = Math.max(1, ((r.value || 0) / max) * (width - lw - 70));
    return `<text class="tick" x="${lw - 8}" y="${i * rowH + 17}" text-anchor="end">${esc(r.label)}</text><rect class="hbar ${r.cls || `s${(i % 6) + 1}`}" x="${lw}" y="${i * rowH + 5}" width="${n1(bw)}" height="${rowH - 10}" rx="3"/><text class="tick val" x="${n1(lw + bw + 6)}" y="${i * rowH + 17}">${esc(fmt(r.value))}</text>`;
  }).join('')}</svg>`;
}


/**
 * Stacked area chart. series: [{name, cls, points:[[x,y],...]}] sharing x values (missing y = 0).
 * Bands use CSS classes (band + cls) so colours come from tokens; total is the top edge.
 */
export function stackedChart(series, opts = {}) {
  const { width = 640, height = 220, xFormat = String, yFormat = (v) => String(v), ariaLabel = 'chart' } = opts;
  const m = { l: 46, r: 12, t: 10, b: 24 };
  const xs = [...new Set(series.flatMap((s) => s.points.map((p) => p[0])))].filter(Number.isFinite).sort((a, b) => a - b);
  if (xs.length < 2) return `<svg class="chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(ariaLabel)}"><text class="empty" x="${width / 2}" y="${height / 2}" text-anchor="middle">No data</text></svg>`;
  const lookup = series.map((s) => new Map(s.points.map(([x, y]) => [x, Number.isFinite(y) ? y : 0])));
  const cum = xs.map((x) => { let acc = 0; return lookup.map((mp) => (acc += mp.get(x) ?? 0)); });
  const ticks = niceTicks(0, Math.max(...cum.map((c) => c[c.length - 1]), 1e-9), 4);
  const X = scaleLinear(xs[0], xs[xs.length - 1], m.l, width - m.r), Y = scaleLinear(ticks[0], ticks[ticks.length - 1], height - m.b, m.t);
  let g = '';
  for (const tk of ticks) g += `<line class="gl" x1="${m.l}" x2="${width - m.r}" y1="${n1(Y(tk))}" y2="${n1(Y(tk))}"/><text class="tick" x="${m.l - 6}" y="${n1(Y(tk) + 4)}" text-anchor="end">${esc(yFormat(tk))}</text>`;
  for (const tk of niceTicks(xs[0], xs[xs.length - 1], 5)) g += `<text class="tick" x="${n1(X(tk))}" y="${height - 6}" text-anchor="middle">${esc(xFormat(tk))}</text>`;
  let body = '';
  series.forEach((s, k) => {
    const top = xs.map((x, i) => [X(x), Y(cum[i][k])]), bottom = xs.map((x, i) => [X(x), Y(k ? cum[i][k - 1] : 0)]).reverse();
    body += `<path class="band ${esc(s.cls)}" d="${linePath(top)}L${linePath(bottom).slice(1)}Z"><title>${esc(s.name)}</title></path>`;
  });
  return `<svg class="chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(ariaLabel)}">${g}${body}</svg>`;
}
