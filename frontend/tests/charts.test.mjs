import test from 'node:test';
import assert from 'node:assert/strict';
import { niceTicks, scaleLinear, linePath, sparkline, lineChart, heatmap, hbars, legend } from '../js/charts.js';
import { toGrid } from '../js/adapt.js';

test('niceTicks produce round steps covering the range', () => {
  assert.deepEqual(niceTicks(0, 100, 5), [0, 20, 40, 60, 80, 100]);
  const t = niceTicks(0.013, 0.87, 4); assert.ok(t.length >= 3 && t.every((v, i) => i === 0 || v > t[i - 1]) && t[0] >= 0.013 && t.at(-1) <= 0.87 + 0.25);
  assert.deepEqual(niceTicks(5, 5, 3).length > 0, true);
});
test('scaleLinear maps endpoints; degenerate domain does not NaN', () => {
  const s = scaleLinear(0, 10, 0, 100); assert.equal(s(5), 50); assert.equal(scaleLinear(3, 3, 0, 10)(3), 5);
});
test('linePath', () => assert.equal(linePath([[0, 1], [2.34, 3]]), 'M0 1L2.3 3'));
test('sparkline with <2 points is an empty labelled svg; otherwise has a path', () => {
  assert.match(sparkline([1]), /aria-label="no data"/); assert.match(sparkline([1, 2, 3]), /<path class="line/);
});
test('lineChart: empty state, and series names are escaped (no markup injection)', () => {
  assert.match(lineChart([]), /No data/);
  const svg = lineChart([{ name: '<img src=x onerror=1>', points: [[0, 1], [1, 2]] }], { yFormat: (v) => `${v}<`, xFormat: (v) => `${v}"` });
  assert.ok(!/<img/.test(svg)); assert.ok(!/<\/text>[^<]*"><script/.test(svg)); assert.match(svg, /&lt;img/);
  assert.ok(!/NaN/.test(svg));
});
test('legend escapes names', () => assert.ok(!/<b>/.test(legend([{ name: '<b>x</b>' }]))));
test('heatmap: 7x24 cells, zero-safe, titles carry the value', () => {
  const g = Array.from({ length: 7 }, () => Array(24).fill(0)); g[2][5] = 10;
  const svg = heatmap(g); assert.equal((svg.match(/<rect/g) || []).length, 168); assert.match(svg, /Wed 05:00 — 10/); assert.match(svg, /class="cell l4"/);
});
test('hbars escapes labels and never yields NaN widths', () => {
  const svg = hbars([{ label: '<x>', value: 0 }, { label: 'b', value: 5 }]); assert.ok(!/<x>/.test(svg)); assert.ok(!/NaN/.test(svg));
});

