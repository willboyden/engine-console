import test from 'node:test';
import assert from 'node:assert/strict';
import { gpuSegments, ramSegments, engineColors, instanceNames, groupByHost, manageBlock, canManage, retryCountdown, silentFor, unrestrictedKey, rootfulDocker, policyMode, linkKind, diskUsage, lowDisk, rootChoices, fleetSummary, hostSignature, isUp, hasData } from '../js/fleet.js';
import { MIN_PCT } from '../js/memory.js';

const sum = (a) => a.reduce((x, y) => x + y, 0);
const inst = (id, o = {}) => ({ id, name: id, host_id: 'local', state: 'ready', host_memory: null, ...o });
const gpu = (o = {}) => ({ index: 0, uuid: 'GPU-0', name: 'NVIDIA Test', total_gib: 96, used_gib: 63, free_gib: 33, unified_memory: false, procs: [], ...o });
const host = (o = {}) => ({ id: 'h', label: 'Box', remote: true, link: { state: 'up', notes: [] }, policy: { mode: 'manage', engine_images: ['x:*'] }, docker: { available: true, rootless: true }, node: {}, gpus: [], counts: {}, alerts: [], ...o });

test('VRAM is split by engine through GpuProc.instance_id, the rest is "other", the remainder "free"', () => {
  const g = gpu({ procs: [{ pid: 1, instance_id: 'a', used_gib: 38 }, { pid: 2, instance_id: 'b', used_gib: 20 }, { pid: 3, instance_id: null, name: 'Xorg', used_gib: 1.4 }] });
  const names = instanceNames([inst('a', { name: 'qwen' }), inst('b', { name: 'gemma' })]);
  const m = gpuSegments(g, { names, colors: engineColors([inst('a'), inst('b')]) });
  assert.deepEqual(m.segments.map((s) => s.key), ['a', 'b', 'other', 'free'], 'engines by size, then other, then free');
  assert.deepEqual(m.segments.map((s) => s.label), ['qwen', 'gemma', null, null]);
  assert.ok(Math.abs(m.segments.find((s) => s.key === 'other').gib - (63 - 58)) < 1e-9, 'other = driver used minus engines');
  assert.ok(Math.abs(sum(m.segments.map((s) => s.gib)) - 96) < 1e-9 && Math.abs(sum(m.segments.map((s) => s.pct)) - 100) < 1e-6, 'segments cover the whole card');
  assert.equal(m.known, true); assert.equal(m.unified, false);
  assert.notEqual(m.segments[0].cls, m.segments[1].cls); assert.deepEqual(m.otherNames, ['Xorg']);
});

test('processes of one instance on one GPU add up; tiny segments stay visible; an empty card is all free', () => {
  const m = gpuSegments(gpu({ used_gib: 40.2, free_gib: 55.8, procs: [{ pid: 1, instance_id: 'a', used_gib: 20 }, { pid: 2, instance_id: 'a', used_gib: 20 }, { pid: 3, instance_id: 'b', used_gib: 0.1 }] }));
  assert.equal(m.segments.find((s) => s.key === 'a').gib, 40); assert.ok(m.segments.find((s) => s.key === 'b').pct >= MIN_PCT - 1e-9);
  const idle = gpuSegments(gpu({ used_gib: 0, free_gib: 96, procs: [] }));
  assert.deepEqual(idle.segments.map((s) => s.key), ['free']);
});

test('when the driver cannot attribute memory the engines still show and "other" absorbs the difference; no used figure falls back to the sum', () => {
  const m = gpuSegments(gpu({ used_gib: 50, procs: [{ pid: 1, instance_id: 'a', used_gib: null }, { pid: 2, instance_id: null, used_gib: null }] }));
  assert.deepEqual(m.segments.map((s) => s.key), ['other', 'free']); assert.equal(m.segments[0].gib, 50);
  const noUsed = gpuSegments(gpu({ used_gib: null, free_gib: null, procs: [{ pid: 1, instance_id: 'a', used_gib: 30 }] }));
  assert.equal(noUsed.known, false); assert.equal(noUsed.used, 30); assert.equal(noUsed.segments[0].key, 'a');
  assert.equal(gpuSegments(null).total, 0, 'no data is not a crash');
});

test('unified-memory GPUs have no VRAM bar of their own', () => {
  const m = gpuSegments(gpu({ unified_memory: true, used_gib: null, free_gib: null, total_gib: 119, procs: [{ pid: 1, instance_id: 'a', used_gib: null }] }));
  assert.equal(m.unified, true); assert.deepEqual(m.segments, []); assert.equal(m.total, 119);
});

test('host RAM is split by engine (running ones only), then other used, then available; colours match the VRAM bar', () => {
  const insts = [inst('a', { host_memory: { total_gib: 34 } }), inst('b', { host_memory: { total_gib: 12 } }), inst('c', { state: 'stopped', host_memory: { total_gib: 99 } }), inst('d', { host_memory: null })];
  const colors = engineColors(insts);
  const r = ramSegments({ total_gib: 246, used_gib: 190, available_gib: 56 }, insts, colors);
  assert.deepEqual(r.segments.map((s) => s.key), ['a', 'b', 'other', 'available']);
  assert.equal(r.segments[0].cls, colors.get('a'), 'the same instance keeps its colour in both bars');
  assert.ok(Math.abs(sum(r.segments.filter((s) => s.kind !== 'available').map((s) => s.gib)) - 190) < 1e-9);
  assert.equal(ramSegments(null, insts), null);
  const named = ramSegments({ total_gib: 246, used_gib: 190, available_gib: 56 }, insts, colors, new Map([['a', 'Qwen/Qwen3.6']]));
  assert.equal(named.segments.find((s) => s.key === 'a').label, 'Qwen/Qwen3.6', 'the legend can use the engine\'s title instead of its container name');
});

test('colours are stable per instance id regardless of list order, and distinct within a machine', () => {
  const a = engineColors([inst('b'), inst('a'), inst('c')]), b = engineColors([inst('c'), inst('b'), inst('a')]);
  for (const k of ['a', 'b', 'c']) assert.equal(a.get(k), b.get(k));
  assert.equal(new Set(a.values()).size, 3);
});

test('instances group by machine in host order; unknown hosts are kept; empty machines are omitted', () => {
  const hosts = [host({ id: 'local', label: 'Workstation' }), host({ id: 'edge-1', label: 'Edge 1' }), host({ id: 'edge-2', label: 'Edge 2' })];
  const g = groupByHost([inst('x', { host_id: 'edge-1' }), inst('y', { host_id: 'local' }), inst('z', { host_id: 'gone', host_label: 'Gone box' }), inst('w', { host_id: 'local' })], hosts);
  assert.deepEqual(g.map((x) => x.id), ['local', 'edge-1', 'gone']); assert.deepEqual(g[0].instances.map((i) => i.id), ['y', 'w']);
  assert.equal(g[2].label, 'Gone box'); assert.equal(g[2].host, null);
  assert.equal(groupByHost([inst('q', { host_id: undefined })], hosts)[0].id, 'local', 'a v1 instance without host_id belongs to local');
});

test('manageBlock explains why a machine cannot be changed, in order: link, then monitor mode, then docker', () => {
  assert.equal(manageBlock(host()), null); assert.equal(canManage(host()), true);
  assert.equal(manageBlock(host({ link: { state: 'down' } })).key, 'fleet.block_down');
  assert.equal(manageBlock(host({ link: { state: 'stale' } })).key, 'fleet.block_stale');
  assert.equal(manageBlock(host({ link: { state: 'incompatible' } })).key, 'fleet.block_incompatible');
  assert.equal(manageBlock(host({ link: { state: 'disabled' } })).key, 'fleet.block_disabled');
  assert.equal(manageBlock(host({ policy: { mode: 'monitor' } })).key, 'fleet.block_monitor');
  assert.equal(manageBlock(host({ policy: null })).key, 'fleet.block_monitor', 'no policy yet means no changes');
  assert.equal(manageBlock(host({ docker: { available: false, error: 'no socket' } })).key, 'fleet.block_docker');
  assert.equal(manageBlock(null).key, 'fleet.block_unknown'); assert.equal(manageBlock(host({ label: 'Box' })), null);
  assert.equal(manageBlock(host({ link: { state: 'down' }, label: 'Box' })).vars.host, 'Box');
});

test('link helpers: badge kinds, countdown from when the summary arrived, silence, and the two explained warnings', () => {
  assert.deepEqual(['up', 'connecting', 'stale', 'down', 'incompatible', 'disabled', 'weird'].map(linkKind), ['ok', 'info', 'warn', 'bad', 'bad', 'muted', 'muted']);
  assert.equal(retryCountdown({ retry_in_s: 47 }, 1000, 1000 + 10_000), 37); assert.equal(retryCountdown({ retry_in_s: 5 }, 0, 60_000), 0, 'never negative');
  assert.equal(retryCountdown({}, 0, 1), null); assert.equal(retryCountdown(null, 0, 1), null);
  assert.equal(silentFor({ last_seen: 100 }, 114_400), 14); assert.equal(silentFor({}, 1), null);
  assert.equal(unrestrictedKey(host({ node: { forced_cmd: false } })), true); assert.equal(unrestrictedKey(host({ node: { forced_cmd: null } })), false, 'unknown is not a warning'); assert.equal(unrestrictedKey(host({ node: { forced_cmd: true } })), false);
  assert.equal(rootfulDocker(host({ docker: { available: true, rootless: false } })), true); assert.equal(rootfulDocker(host({ docker: { available: true, rootless: true } })), false); assert.equal(rootfulDocker(host({ docker: { available: true, rootless: null } })), false);
  assert.equal(policyMode(host()), 'manage'); assert.equal(policyMode(host({ policy: { mode: 'weird' } })), 'monitor'); assert.equal(policyMode(host({ policy: null })), null);
  assert.equal(isUp(host()), true); assert.equal(hasData(host({ link: { state: 'stale' } })), true); assert.equal(hasData(host({ link: { state: 'down' } })), false);
});

test('drives: used/free from bytes, low-space flag, and the writable roots with free space, biggest first', () => {
  const G = 2 ** 30;
  assert.deepEqual(diskUsage({ size_bytes: 100 * G, free_bytes: 25 * G }), { size: 100 * G, used: 75 * G, free: 25 * G, pct: 75 });
  assert.equal(diskUsage({}).pct, null); assert.equal(lowDisk({ size_bytes: 100, free_bytes: 5 }), true); assert.equal(lowDisk({ size_bytes: 100, free_bytes: 50 }), false); assert.equal(lowDisk({}), false);
  const h = host({ roots: [{ id: 'a', kind: 'hf-home', writable: true, mount: { free_bytes: 10 } }, { id: 'ro', kind: 'hf-home', writable: false, mount: { free_bytes: 999 } }, { id: 'b', kind: 'gguf', writable: true, mount: { free_bytes: 50 } }, { id: 'c', kind: 'hf-home', writable: true, label: 'Big', mount: { free_bytes: 70 } }] });
  assert.deepEqual(rootChoices(h).map((r) => r.root.id), ['c', 'b', 'a']);
  assert.deepEqual(rootChoices(h, { kinds: ['hf-home'] }).map((r) => r.root.id), ['c', 'a']); assert.equal(rootChoices(h)[0].label, 'Big');
});

test('fleet summary counts links and engines', () => {
  const s = fleetSummary([host({ link: { state: 'up' }, counts: { engines_running: 2, engines_total: 3, models: 4 } }), host({ id: 'b', link: { state: 'stale' } }), host({ id: 'c', link: { state: 'down' } }), host({ id: 'd', link: { state: 'disabled' } })]);
  assert.deepEqual(s, { total: 4, up: 1, stale: 1, down: 1, other: 1, enginesRunning: 2, enginesTotal: 3, models: 4 });
});

test('the card signature ignores sensor jitter and countdown ticks but notices real changes', () => {
  const base = host({ gpus: [gpu({ used_gib: 40.1, util_pct: 71, temp_c: 56.2, power_w: 211 })], memory: { used_gib: 100.2, swap_used_gib: 2 } });
  const jitter = host({ gpus: [gpu({ used_gib: 40.2, util_pct: 72, temp_c: 56.9, power_w: 212 })], memory: { used_gib: 100.4, swap_used_gib: 2 } });
  assert.equal(hostSignature(base), hostSignature(jitter));
  assert.notEqual(hostSignature(base), hostSignature(host({ gpus: [gpu({ used_gib: 55 })], memory: base.memory })));
  assert.notEqual(hostSignature(base), hostSignature({ ...base, link: { state: 'stale' } }));
  assert.equal(hostSignature(host({ link: { state: 'down', retry_in_s: 40 } })), hostSignature(host({ link: { state: 'down', retry_in_s: 39 } })), 'the countdown has its own timer');
  assert.notEqual(hostSignature(base, 'a'), hostSignature(base, 'b'));
});

import { visibleNotes } from '../js/fleet.js';
test('visibleNotes hides the hub note about an unrestricted key only when the UI explains it itself', () => {
  const notes = ['Reached with an UNRESTRICTED ssh key: restrict it.', 'Clock differs by 3 s.'];
  assert.deepEqual(visibleNotes({ node: { forced_cmd: false }, link: { notes } }), ['Clock differs by 3 s.']);
  assert.deepEqual(visibleNotes({ node: { forced_cmd: true }, link: { notes } }), notes);
  assert.deepEqual(visibleNotes({ link: {} }), []);
  assert.deepEqual(visibleNotes(null), []);
});
