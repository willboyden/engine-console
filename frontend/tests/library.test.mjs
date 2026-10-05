import test from 'node:test';
import assert from 'node:assert/strict';
import { filterLibrary, sortLibrary, sizeBand, paramsLabel, inUse, usedBy, totalCopies, copyChip, launchHosts, deleteState, copyTargets, copyActions, resolveOnHost, hostChoicesForModel, quickFitHosts, hostsThatSee, targetRoots } from '../js/library.js';

const G = 2 ** 30;
const copy = (o = {}) => ({ id: 'c1', host_id: 'local', host_label: 'Workstation', root_id: 'hf-main', root_label: 'HF cache', rel: 'models--a--b', path: '/data/hf/models--a--b', size_bytes: 10 * G, visible_on: ['local'], writable: true, deletable: true, in_use_by: [], drive: { label: 'tank (zfs)' }, ...o });
const model = (o = {}, copies = [copy()]) => ({ key: 'hf:a/b', name: 'a/b', format: 'hf', quant: null, params: 7e9, arch: 'X', context: 8192, size_bytes: 10 * G, engines: ['vllm'], copies, warnings: [], ...o });
const host = (id, o = {}) => ({ id, label: id.toUpperCase(), link: { state: 'up' }, policy: { mode: 'manage', allow_delete: true, engine_images: ['x:*'] }, docker: { available: true }, gpus: [], roots: [], ...o });
const hosts = [host('local'), host('gpu-box-2'), host('edge-1'), host('edge-2', { policy: { mode: 'monitor', allow_delete: false } }), host('edge-4', { link: { state: 'down' } })];

test('filters by machine (a copy on it OR readable from it), format and every word of the text', () => {
  const shared = model({ key: 'gguf:r:Q4', name: 'r/model-GGUF', format: 'gguf', quant: 'Q4_K_M' }, [copy({ host_id: 'gpu-box-2', host_label: 'GPU Box 2', visible_on: ['gpu-box-2', 'edge-1'] })]);
  const mine = model({}, [copy()]);
  const list = [mine, shared];
  assert.deepEqual(filterLibrary(list, { host: 'edge-1' }).map((m) => m.key), ['gguf:r:Q4'], 'shared storage counts as available on edge-1');
  assert.deepEqual(filterLibrary(list, { host: 'local' }).map((m) => m.key), ['hf:a/b']);
  assert.deepEqual(filterLibrary(list, { format: 'gguf' }).map((m) => m.key), ['gguf:r:Q4']);
  assert.deepEqual(filterLibrary(list, { q: 'q4_k box' }).map((m) => m.key), ['gguf:r:Q4'], 'matches quant and machine label');
  assert.deepEqual(filterLibrary(list, { q: 'q4 nothing' }), []);
  assert.equal(filterLibrary(list, {}).length, 2); assert.deepEqual([...hostsThatSee(shared)].sort(), ['edge-1', 'gpu-box-2']);
});

test('sorting by name (case-insensitive), size and format', () => {
  const a = model({ key: 'k1', name: 'beta', size_bytes: 5 * G }), b = model({ key: 'k2', name: 'Alpha', size_bytes: 9 * G }), c = model({ key: 'k3', name: 'gamma', format: 'gguf', size_bytes: 1 * G });
  assert.deepEqual(sortLibrary([a, b, c]).map((m) => m.name), ['Alpha', 'beta', 'gamma']);
  assert.deepEqual(sortLibrary([a, b, c], 'size').map((m) => m.name), ['Alpha', 'beta', 'gamma']);
  assert.deepEqual(sortLibrary([c, a], 'format').map((m) => m.name), ['beta', 'gamma']);
});

test('plain-language size bands, parameter labels, in-use and copy totals', () => {
  assert.deepEqual([1, 7.9, 8, 39, 40, 99, 100, 300].map((g) => sizeBand(g * G)), ['small', 'small', 'medium', 'medium', 'large', 'large', 'huge', 'huge']);
  assert.equal(paramsLabel(35e9), '35B'); assert.equal(paramsLabel(null), ''); assert.equal(paramsLabel(0), '');
  const used = model({}, [copy({ in_use_by: ['i1'] }), copy({ id: 'c2', host_id: 'x', in_use_by: ['i1', 'i2'] })]);
  assert.equal(inUse(used), true); assert.deepEqual(usedBy(used), ['i1', 'i2']); assert.equal(inUse(model()), false); assert.equal(totalCopies([used, model()]), 3);
});

test('a copy chip names the machine and drive and flags shared storage', () => {
  const c = copyChip(copy({ visible_on: ['local', 'edge-1', 'edge-2'] }));
  assert.equal(c.host, 'Workstation'); assert.equal(c.drive, 'tank (zfs)'); assert.equal(c.shared, true); assert.deepEqual(c.alsoOn, ['edge-1', 'edge-2']);
  const own = copyChip(copy({ drive: null })); assert.equal(own.shared, false); assert.equal(own.drive, 'HF cache', 'falls back to the folder label');
});

test('Delete is off, with the most specific reason, when in use, read-only, machine down or policy forbids; else on', () => {
  assert.deepEqual(deleteState(copy(), host('local')), { enabled: true, reason: null });
  assert.equal(deleteState(copy({ in_use_by: ['inst_a', 'inst_b'] }), host('local')).reason.key, 'library.reason_in_use');
  assert.equal(deleteState(copy({ in_use_by: ['inst_a'] }), host('local')).reason.vars.names, 'inst_a');
  assert.equal(deleteState(copy({ in_use_by: ['inst_a', 'inst_b'] }), host('local'), (id) => ({ inst_a: 'Qwen/Big' }[id] || id)).reason.vars.names, 'Qwen/Big, inst_b', 'ids can be shown as engine titles');
  assert.equal(deleteState(copy({ writable: false, deletable: false }), host('local')).reason.key, 'library.reason_read_only');
  assert.equal(deleteState(copy(), hosts.find((x) => x.id === 'edge-4')).reason.key, 'library.reason_host_down');
  assert.equal(deleteState(copy(), hosts.find((x) => x.id === 'edge-2')).reason.key, 'library.reason_policy_delete');
  assert.equal(deleteState(copy({ deletable: false }), host('local')).reason.key, 'library.reason_not_deletable');
  assert.equal(deleteState(copy({ in_use_by: ['x'], writable: false }), host('local')).reason.key, 'library.reason_in_use', 'in use is reported first');
});

test('copy targets exclude machines that already read the copy; monitor-only and down ones are listed with a reason', () => {
  const c = copy({ visible_on: ['local', 'gpu-box-2'] });
  const t = copyTargets(c, hosts);
  assert.deepEqual(t.map((x) => x.id), ['edge-1', 'edge-2', 'edge-4']);
  assert.deepEqual(t.map((x) => x.enabled), [true, false, false]);
  assert.equal(t[1].reason.key, 'fleet.block_monitor'); assert.equal(t[2].reason.key, 'fleet.block_down');
});

test('copy targets also exclude a machine that holds another copy of the same model', () => {
  const m = model({}, [copy({ id: 'a', host_id: 'local', visible_on: ['local'] }), copy({ id: 'b', host_id: 'gpu-box-2', visible_on: ['gpu-box-2', 'edge-1'] })]);
  assert.deepEqual(copyTargets(m.copies[0], hosts, m).map((x) => x.id), ['edge-2', 'edge-4'], 'gpu-box-2 holds a copy and edge-1 reads it');
  assert.deepEqual(copyTargets(m.copies[0], hosts).map((x) => x.id), ['gpu-box-2', 'edge-1', 'edge-2', 'edge-4'], 'without the model only the copy itself is considered');
  assert.equal(copyActions(m.copies[0], hosts, m).copy.reason?.key, 'library.reason_no_target_manage');
});

test('per-copy actions: Launch needs a machine that can run it, Copy needs a target that accepts files, Delete has its own rule', () => {
  const a = copyActions(copy({ visible_on: ['local', 'gpu-box-2', 'edge-1', 'edge-2', 'edge-4'] }), hosts);
  assert.equal(a.launch.enabled, true); assert.equal(a.copy.enabled, false); assert.equal(a.copy.reason.key, 'library.reason_everywhere'); assert.equal(a.delete.enabled, true);
  const onlyMonitor = copyActions(copy({ host_id: 'edge-2', visible_on: ['edge-2'] }), hosts);
  assert.equal(onlyMonitor.launch.enabled, false); assert.equal(onlyMonitor.launch.reason.key, 'fleet.block_monitor');
  const noTarget = copyActions(copy({ visible_on: ['local', 'gpu-box-2', 'edge-1'] }), hosts);
  assert.equal(noTarget.copy.enabled, false); assert.equal(noTarget.copy.reason.key, 'library.reason_no_target_manage', 'edge-2 and edge-4 exist but cannot take files');
  assert.deepEqual(launchHosts(copy({ visible_on: ['local', 'edge-1'] }), hosts).map((x) => x.id), ['local', 'edge-1']);
});

test('resolving a copy for a machine: its own copy, or shared storage through that machine\'s own root', () => {
  const m = model({ key: 'gguf:r:Q4', name: 'r', format: 'gguf' }, [copy({ host_id: 'gpu-box-2', root_id: 'gguf-shared', rel: 'r/m.gguf', visible_on: ['gpu-box-2', 'edge-1'] })]);
  const own = resolveOnHost(m, 'gpu-box-2', {});
  assert.equal(own.via, 'own'); assert.deepEqual(own.ref, { format: 'gguf', root_id: 'gguf-shared', rel: 'r/m.gguf' });
  const details = new Map([['edge-1', { roots: [{ id: 'nfs-shared', shared_from: 'gpu-box-2' }, { id: 'hf-main', shared_from: null }] }]]);
  const shared = resolveOnHost(m, 'edge-1', details);
  assert.equal(shared.via, 'shared'); assert.deepEqual(shared.ref, { format: 'gguf', root_id: 'nfs-shared', rel: 'r/m.gguf' }, 'the root id is translated to the reading machine\'s own root');
  assert.equal(m.copies[0].root_id, 'gguf-shared', 'the library data is not mutated');
  assert.equal(resolveOnHost(m, 'edge-9', details), null);
  assert.deepEqual(resolveOnHost(m, 'edge-1', {}).ref.root_id, 'gguf-shared', 'without that machine\'s roots it falls back to the owner\'s root id');
  assert.deepEqual(resolveOnHost(model(), 'local', {}).ref, { format: 'hf', repo_id: 'a/b' });
  assert.deepEqual(resolveOnHost(model({ format: 'ollama', name: 'qwen3:8b' }), 'local', {}).ref, { format: 'ollama', name: 'qwen3:8b' });
});

test('launch candidates: loadable machines first, then blocked ones with the reason, then those that need a copy', () => {
  const m = model({}, [copy({ visible_on: ['local', 'edge-2', 'edge-4'] })]);
  const c = hostChoicesForModel(m, hosts, new Map());
  assert.deepEqual(c.map((x) => [x.host.id, x.state]), [['local', 'ok'], ['edge-2', 'blocked'], ['edge-4', 'blocked'], ['gpu-box-2', 'no_copy'], ['edge-1', 'no_copy']]);
  assert.equal(c[0].enabled, true); assert.equal(c[1].reason.key, 'fleet.block_monitor'); assert.equal(c[3].reason.key, 'library.reason_no_copy'); assert.equal(c[3].ref, null);
  const none = hostChoicesForModel(model({}, [copy()]), hosts, new Map()); assert.equal(none.find((x) => x.host.id === 'edge-4').reason.key, 'fleet.block_down', 'a down machine without the model: the useful reason is that it is down');
  assert.equal(none.find((x) => x.host.id === 'edge-2').reason.key, 'fleet.block_monitor');
});

test('the quick "might fit" hint uses only machines that can see the model and have GPUs', () => {
  const m = model({ size_bytes: 30 * G }, [copy({ visible_on: ['local', 'edge-1'] })]);
  const hs = [host('local', { gpus: [{ total_gib: 95 }, { total_gib: 95 }] }), host('edge-1', { gpus: [{ total_gib: 24 }] }), host('gpu-box-2', { gpus: [{ total_gib: 95 }] })];
  assert.deepEqual(quickFitHosts(m, hs).map((x) => x.id), ['local'], 'edge-1 is too small, gpu-box-2 cannot see the model');
});

test('target roots suit the format: hf into hf folders, gguf into gguf folders, and only writable ones', () => {
  const h = host('x', { roots: [{ id: 'hf', kind: 'hf-home', writable: true, mount: { free_bytes: 5 } }, { id: 'gg', kind: 'gguf', writable: true, mount: { free_bytes: 9 } }, { id: 'ro', kind: 'gguf', writable: false, mount: { free_bytes: 99 } }] });
  assert.deepEqual(targetRoots(h, 'hf').map((r) => r.root.id), ['hf']); assert.deepEqual(targetRoots(h, 'gguf').map((r) => r.root.id), ['gg']); assert.deepEqual(targetRoots(h, 'ollama'), []);
});
