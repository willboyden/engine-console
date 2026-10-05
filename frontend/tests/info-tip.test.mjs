import test from 'node:test';
import assert from 'node:assert/strict';
import { placePopover } from '../js/components/info-tip.js';

const anchor = (left, top, size = 24) => ({ left, top, right: left + size, bottom: top + size, width: size, height: size });

test('the popover goes beside the button: to the right when there is room, else to the left, else below, else above', () => {
  const vw = 1000, vh = 800, width = 380, height = 300;
  const right = placePopover({ anchor: anchor(100, 200), width, height, vw, vh });
  assert.equal(right.left, 124 + 8); assert.equal(right.top, 200 - 8);
  const left = placePopover({ anchor: anchor(900, 200), width, height, vw, vh });
  assert.equal(left.left, 900 - 8 - width);
  // CSS limits the popover to the viewport minus 16px, so on a 390px phone it is 374px wide
  const below = placePopover({ anchor: anchor(200, 100), width: 374, height, vw: 390, vh });
  assert.equal(below.top, 124 + 8, 'on a phone it opens under the button'); assert.ok(below.left >= 8 && below.left + 374 <= 390 - 8, 'and stays on screen');
  const above = placePopover({ anchor: anchor(200, 700), width: 374, height, vw: 390, vh });
  assert.equal(above.top, 700 - 8 - height, 'no room below: above');
});

test('the popover is always clamped into the viewport, even when taller than it', () => {
  const p = placePopover({ anchor: anchor(10, 780), width: 380, height: 300, vw: 1000, vh: 800 });
  assert.ok(p.top >= 8 && p.top + 300 <= 800 - 8, JSON.stringify(p));
  const tall = placePopover({ anchor: anchor(10, 10), width: 380, height: 2000, vw: 1000, vh: 800 });
  assert.equal(tall.top, 8, 'never above the top edge');
  const narrow = placePopover({ anchor: anchor(5, 5), width: 380, height: 100, vw: 300, vh: 800 });
  assert.equal(narrow.left, 8, 'never past the left edge');
});
