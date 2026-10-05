import test from 'node:test';
import assert from 'node:assert/strict';
import { controlButtons, normControls, managementOf, whyNot, scriptActions, scriptFor, primaryAction } from '../js/controls.js';

const C = (o = {}) => ({ start: false, stop: false, restart: false, remove: false, logs: false, configure: false, reason: null, ...o });
const ids = (r) => r.buttons.map((b) => b.id);
const byId = (r, id) => r.buttons.find((b) => b.id === id);

test('a managed instance gets every control, shown by state: start when stopped, stop and restart when running', () => {
  const all = C({ start: true, stop: true, restart: true, remove: true, logs: true, configure: true });
  const ready = controlButtons({ management: 'managed', state: 'ready', controls: all });
  assert.deepEqual(ids(ready), ['stop', 'restart', 'configure', 'pin', 'remove']); assert.ok(ready.buttons.every((b) => b.enabled && b.reason === null)); assert.equal(ready.logs.enabled, true); assert.equal(ready.note, null);
  const stopped = controlButtons({ management: 'managed', state: 'stopped', controls: all });
  assert.deepEqual(ids(stopped), ['start', 'configure', 'pin', 'remove']); assert.equal(byId(stopped, 'start').kind, 'primary'); assert.equal(primaryAction({ management: 'managed', state: 'stopped', controls: all }), 'start');
  assert.deepEqual(ids(controlButtons({ management: 'managed', state: 'failed', controls: all })), ['start', 'restart', 'configure', 'pin', 'remove']);
  assert.deepEqual(ids(controlButtons({ management: 'managed', state: 'loading', controls: all })), ['stop', 'restart', 'configure', 'pin', 'remove']);
  assert.equal(byId(ready, 'remove').kind, 'danger');
});

test('a managed instance on a machine that is not up keeps its buttons, disabled, with the server reason', () => {
  const r = controlButtons({ management: 'managed', state: 'ready', controls: C({ logs: true, configure: true, reason: 'Box is stale: nothing can be changed.' }) });
  assert.deepEqual(r.buttons.filter((b) => !b.enabled).map((b) => b.id), ['stop', 'restart', 'remove']);
  assert.deepEqual(byId(r, 'stop').reason, { text: 'Box is stale: nothing can be changed.' }); assert.deepEqual(r.note, { text: 'Box is stale: nothing can be changed.' });
  assert.equal(byId(r, 'pin').enabled, true, 'configure stays available so the form can be opened');
});

test('a script instance maps start/stop/restart to the script, has no remove, no pin, no configure, and says which script', () => {
  const c = C({ start: true, stop: true, restart: true, reason: 'Started by the script "big-model".' });
  const r = controlButtons({ management: 'script', script: 'big-model', state: 'ready', controls: c });
  assert.deepEqual(ids(r), ['stop', 'restart']); assert.ok(r.buttons.every((b) => b.enabled)); assert.deepEqual(r.note, { text: 'Started by the script "big-model".' }, 'the note is shown even when everything works');
  assert.equal(r.logs.enabled, false); assert.equal(r.logs.reason.text, 'Started by the script "big-model".');
  assert.deepEqual(ids(controlButtons({ management: 'script', state: 'stopped', controls: c })), ['start']);
});

test('an external engine: without permission every relevant control is disabled with the reason; with stop_start only stop and start are offered', () => {
  const locked = controlButtons({ management: 'external', state: 'ready', controls: C({ configure: true, reason: 'Started outside the console; the policy forbids stopping it.' }) });
  assert.deepEqual(ids(locked), ['stop', 'restart', 'configure']); assert.deepEqual(locked.buttons.filter((b) => !b.enabled).map((b) => b.id), ['stop', 'restart']);
  assert.match(byId(locked, 'stop').reason.text, /policy forbids/);
  const allowed = controlButtons({ management: 'external', state: 'ready', controls: C({ start: true, stop: true, configure: true }) });
  assert.deepEqual(ids(allowed), ['stop', 'configure'], 'restart is not offered for an external engine that can only be stopped and started');
  assert.ok(byId(allowed, 'stop').enabled); assert.deepEqual(ids(controlButtons({ management: 'external', state: 'stopped', controls: C({ start: true, stop: true, configure: true }) })), ['start', 'configure']);
  assert.deepEqual(ids(controlButtons({ management: 'external', state: 'auth_required', controls: C({ configure: true }) })), ['stop', 'restart', 'configure']);
  assert.equal(byId(locked, 'configure').enabled, true, 'clone as managed is the way forward');
});

test('without `controls` (a v1 backend) managed means everything and external means clone only', () => {
  assert.deepEqual(normControls({ managed: true }), C({ start: true, stop: true, restart: true, remove: true, logs: true, configure: true }));
  assert.deepEqual(normControls({ managed: false }), C({ configure: true })); assert.equal(managementOf({ managed: false }), 'external'); assert.equal(managementOf({ source: 'external' }), 'external'); assert.equal(managementOf({}), 'managed'); assert.equal(managementOf({ management: 'script' }), 'script');
  const r = controlButtons({ managed: false, state: 'ready' });
  assert.equal(byId(r, 'stop').enabled, false); assert.equal(whyNot({ managed: false }, 'stop').key, 'controls.reason_external');
});

test('reasons: the server text wins; otherwise a UI key per management class that names the script and machine', () => {
  assert.deepEqual(whyNot({ management: 'managed', controls: C({ reason: 'x' }) }, 'stop'), { text: 'x' });
  assert.deepEqual(whyNot({ management: 'script', script: 's', host_label: 'Box', controls: C() }, 'stop'), { key: 'controls.reason_script', vars: { script: 's', host: 'Box' } });
  assert.equal(whyNot({ management: 'managed', controls: C() }, 'stop').key, 'controls.reason_managed');
  assert.deepEqual(normControls({ controls: { start: 1, reason: 5 } }), C({ start: true }), 'values are coerced; a non-text reason is dropped');
});

test('script actions keep their policy flags and arguments; the instance finds its script by host and name', () => {
  const s = { host_id: 'local', name: 'big-model', actions: [{ name: 'up', label: 'Start', confirm: true }, { name: 'switch', label: 'Switch', args: ['a', 'b'], danger: true }, { name: 'status' }] };
  assert.deepEqual(scriptActions(s), [{ name: 'up', label: 'Start', args: [], confirm: true, danger: false }, { name: 'switch', label: 'Switch', args: ['a', 'b'], confirm: false, danger: true }, { name: 'status', label: 'status', args: [], confirm: false, danger: false }]);
  assert.equal(scriptFor({ host_id: 'local', script: 'big-model' }, [s]), s); assert.equal(scriptFor({ host_id: 'edge-1', script: 'big-model' }, [s]), null); assert.equal(scriptFor({ script: 'big-model' }, [s]), s, 'a v1 instance is on local');
});

import { instanceTitle, instanceLabel, scriptLabelMap, scriptKey, splitInstances, scriptActionsFor } from '../js/controls.js';

test('engines named after their container show the served model (or script label) as the title and the container as secondary text', () => {
  const labels = scriptLabelMap([{ host_id: 'local', name: 'big-model', label: 'Big model (both GPUs)' }, { host_id: 'local', name: 'coder-model', label: 'Coding model' }]);
  assert.equal(labels.get(scriptKey('local', 'big-model')), 'Big model (both GPUs)'); assert.equal(scriptKey(undefined, 'x'), 'local|x');
  const running = { management: 'script', host_id: 'local', script: 'big-model', name: 'ai-vllm-big-1', container_name: 'ai-vllm-big-1', served_models: ['org/Big-27B'] };
  assert.deepEqual(instanceTitle(running, labels), { title: 'org/Big-27B', secondary: 'Big model (both GPUs) · ai-vllm-big-1' });
  const stopped = { management: 'script', host_id: 'local', script: 'coder-model', name: 'coder-model-vllm-1', container_name: null, served_models: [] };
  assert.deepEqual(instanceTitle(stopped, labels), { title: 'Coding model', secondary: 'coder-model-vllm-1' }, 'a stopped script engine serves nothing: the script label is the title');
  assert.deepEqual(instanceTitle({ ...stopped, script: 'unlisted' }, labels), { title: 'unlisted', secondary: 'coder-model-vllm-1' }, 'without a label the script name stands in');
  assert.deepEqual(instanceTitle({ management: 'external', name: 'engine-vllm', container_name: 'engine-vllm', served_models: ['google/gemma-4-31b-it'] }), { title: 'google/gemma-4-31b-it', secondary: 'engine-vllm' });
  assert.deepEqual(instanceTitle({ management: 'external', name: 'api-gateway', container_name: 'api-gateway', served_models: [] }), { title: 'api-gateway', secondary: '' }, 'nothing is repeated');
  assert.deepEqual(instanceTitle({ management: 'managed', name: 'qwen-agent', container_name: 'ec-vllm-ab12', served_models: ['org/m'] }), { title: 'qwen-agent', secondary: '' }, 'a name the console chose is already a good title');
  assert.deepEqual(instanceTitle({ management: 'external', name: 'x', served_models: ['  ', 7, 'real/model'] }).title, 'real/model', 'blank and odd entries are skipped');
  assert.deepEqual(instanceTitle({ id: 'inst_1' }), { title: 'inst_1', secondary: '' });
  assert.equal(instanceLabel(running, labels), 'org/Big-27B (Big model (both GPUs) · ai-vllm-big-1)'); assert.equal(instanceLabel({ management: 'managed', name: 'a' }), 'a');
});

test('stopped engines are grouped apart from the active ones, order kept', () => {
  const l = [{ id: 'a', state: 'ready' }, { id: 'b', state: 'stopped' }, { id: 'c', state: 'failed' }, { id: 'd', state: 'stopped' }, { id: 'e', state: 'auth_required' }, { id: 'f', state: 'loading' }];
  const { active, available } = splitInstances(l);
  assert.deepEqual(active.map((i) => i.id), ['a', 'c', 'e', 'f'], 'failed and unreachable engines need attention: they stay with the active ones'); assert.deepEqual(available.map((i) => i.id), ['b', 'd']);
  assert.deepEqual(splitInstances(undefined), { active: [], available: [] });
});

test('a script engine whose controls are empty still gets every script action; one with working controls gets only the extras', () => {
  const script = { host_id: 'local', name: 's', actions: [{ name: 'up', label: 'Start' }, { name: 'down', label: 'Stop' }, { name: 'status', label: 'Status' }, { name: 'switch', label: 'Switch', args: ['a'] }] };
  const none = { management: 'script', state: 'stopped', controls: { start: false, stop: false, restart: false, remove: false, logs: false, configure: false, reason: null } };
  assert.deepEqual(scriptActionsFor(none, script).map((a) => a.name), ['up', 'down', 'status', 'switch']);
  const mapped = { ...none, controls: { ...none.controls, start: true, stop: true, restart: true } };
  assert.deepEqual(scriptActionsFor(mapped, script).map((a) => a.name), ['status', 'switch']);
  assert.deepEqual(scriptActionsFor({ management: 'script', state: 'ready' }, script).map((a) => a.name), ['up', 'down', 'status', 'switch'], 'a v1 backend sends no controls for a script engine: nothing is mapped, so every action is offered');
  assert.deepEqual(scriptActionsFor(none, null), []);
});
