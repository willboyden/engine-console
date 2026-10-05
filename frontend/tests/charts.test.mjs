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


import { stackedChart } from '../js/charts.js';
test('stackedChart: bands stack cumulatively, names escaped, empty/single-point safe', () => {
  const svg = stackedChart([{ name: '<b>a</b>', cls: 'hm-anon', points: [[0, 1], [1, 2]] }, { name: 'b', cls: 'hm-shm', points: [[0, 1], [1, 1]] }]);
  assert.equal((svg.match(/<path class="band/g) || []).length, 2); assert.ok(!svg.includes('<b>a')); assert.ok(!/NaN/.test(svg));
  assert.match(stackedChart([]), /No data/); assert.match(stackedChart([{ name: 'a', cls: 'x', points: [[0, 1]] }]), /No data/);
});

import { describeSeries, chartLabel, heatmapPeak } from '../js/charts.js';
test('describeSeries gives every series its latest, lowest and highest value from a caller-supplied template', () => {
  const s = [{ name: 'Gen', points: [[0, 5], [1, 2], [2, 9], [3, 4]] }, { name: 'Empty', points: [] }];
  assert.equal(describeSeries(s), 'Gen: now 4, lowest 2, highest 9');
  assert.equal(describeSeries(s, { yFormat: (v) => `${v}!`, template: '{name} / {last} / {min} / {max} / {nope}' }), 'Gen / 4! / 2! / 9! / {nope}');
  assert.equal(describeSeries(null), '');
  assert.equal(describeSeries([{ name: 'x', points: [[0, NaN]] }]), '');
});
test('chartLabel is the title plus the summary, or just the title when there is nothing to say', () => {
  assert.equal(chartLabel('Throughput', [{ name: 'a', points: [[0, 1]] }]), 'Throughput. a: now 1, lowest 1, highest 1');
  assert.equal(chartLabel('Throughput', []), 'Throughput');
});
test('heatmapPeak finds the busiest cell and is null for an empty grid', () => {
  const g = Array.from({ length: 7 }, () => Array(24).fill(0)); assert.equal(heatmapPeak(g), null);
  g[3][14] = 7; g[1][2] = 9; assert.deepEqual(heatmapPeak(g), { day: 1, hour: 2, value: 9 });
});
test('heatmap and charts take their wording from the caller (translated copy), not from hard-coded English', () => {
  const g = Array.from({ length: 7 }, () => Array(24).fill(1));
  const svg = heatmap(g, { dayLabels: ['a', 'b', 'c', 'd', 'e', 'f', 'g'], label: 'demandes', ariaLabel: 'Demandes par jour' });
  assert.match(svg, /aria-label="Demandes par jour"/); assert.ok(!/by weekday and hour/.test(svg)); assert.match(svg, /<title>c 05:00 — 1 demandes/);
  assert.match(lineChart([], { noData: 'Aucune donnee' }), /Aucune donnee/); assert.ok(!/No data/.test(lineChart([], { noData: 'Aucune donnee' })));
  assert.match(stackedChart([], { noData: 'Aucune donnee' }), /Aucune donnee/);
});
test('hbars: a long label is cut with an ellipsis inside the gutter, keeps its full text in a title, and the svg name lists the bars', () => {
  const long = 'Qwen/Qwen3.6-35B-A3B-FP8-with-a-very-long-suffix-that-would-never-fit-in-the-gutter';
  const svg = hbars([{ label: long, value: 5 }, { label: 'short', value: 2 }], { ariaLabel: 'Tokens by model', fmt: (v) => `${v}t` });
  assert.match(svg, /aria-label="Tokens by model: Qwen\/Qwen3.6-35B-A3B-FP8-with-a-very-long-suffix[^"]*5t, short 2t"/);
  assert.match(svg, /…<\/text>/); assert.ok(svg.includes(`<title>${long}</title>`)); assert.match(svg, /<title>short: 2t<\/title>/);
  const visible = [...svg.matchAll(/<\/title>([^<]*)<\/text>/g)].map((m) => m[1]);
  assert.ok(visible.every((v) => v.length <= 35), `labels fit the gutter: ${visible.join(' | ')}`);
  assert.equal(hbars([], { ariaLabel: 'Empty' }).match(/aria-label="([^"]*)"/)[1], 'Empty');
});
