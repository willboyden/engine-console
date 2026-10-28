// A fake `docker` CLI for the Rust node's tests (port of node/tests/fakes/fake_docker.py, Node 22, no deps):
// state in $FAKE_DOCKER_STATE (JSON), every call appended to $FAKE_DOCKER_LOG as {"argv", "env", "env_names"}:
// `env` holds NAME -> value for each bare `-e NAME`, `env_names` is the whole environment the node gave this
// process (taken by the wrapper before it adds the fake's own settings). Never touches a real daemon.
'use strict';
const fs = require('node:fs');
const crypto = require('node:crypto');

const PS_FORMAT = '{{.ID}}\t{{.Names}}\t{{.State}}\t{{.Status}}';
const out = (s) => process.stdout.write(s + '\n');
const err = (s) => process.stderr.write(s + '\n');

function load() { return JSON.parse(fs.readFileSync(process.env.FAKE_DOCKER_STATE, 'utf8')); }
function save(s) { fs.writeFileSync(process.env.FAKE_DOCKER_STATE, JSON.stringify(s)); }

function log(argv, child) {
  const env = {};
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '-e' && i + 1 < argv.length && !argv[i + 1].includes('=')) {
      env[argv[i + 1]] = child[argv[i + 1]] ?? null;
    }
  }
  fs.appendFileSync(process.env.FAKE_DOCKER_LOG,
    JSON.stringify({ argv, env, env_names: Object.keys(child).sort() }) + '\n');
}

function find(state, ref) {
  return state.containers.find((c) => c.Name.replace(/^\//, '') === ref || c.Id === ref
    || (ref.length >= 6 && c.Id.startsWith(ref))) || null;
}
const opt = (args, flag) => (args.includes(flag) ? args[args.indexOf(flag) + 1] : null);

function containerDoc(name, image, labels, running, args) {
  return {
    Id: crypto.randomBytes(32).toString('hex'), Name: '/' + name, Image: 'sha256:' + 'ab'.repeat(32),
    Created: '2026-09-01T10:00:00.123456789Z', Path: args.length ? args[0] : '', Args: args.slice(1),
    State: { Status: running ? 'running' : 'created', Running: running, Pid: running ? 4242 : 0, ExitCode: 0,
      StartedAt: '2026-09-01T10:00:01Z' },
    Config: { Image: image, Labels: labels, Env: ['SHOULD_NEVER_BE_READ=value'] },
    HostConfig: { NetworkMode: 'bridge' }, NetworkSettings: { Ports: {} }, Mounts: [],
  };
}

function parseRun(args) {
  const takes = new Set(['--name', '--network', '--network-alias', '--restart', '--cap-drop', '--security-opt',
    '--pids-limit', '--gpus', '--shm-size', '--ipc', '--label', '-v', '-e', '-p', '--tmpfs', '--user',
    '--entrypoint', '--pull', '--ulimit']);
  let i = 0; const labels = {}; let name = null;
  while (i < args.length && args[i].startsWith('-')) {
    const a = args[i];
    if (a === '-d' || a === '--read-only') { i++; continue; }
    if (takes.has(a)) {
      const v = args[i + 1];
      if (a === '--label') { const k = v.split('=')[0]; labels[k] = v.slice(k.length + 1); }
      if (a === '--name') name = v;
      i += 2; continue;
    }
    throw new Error(`fake docker: unknown flag ${a}`);
  }
  return [name, labels, args[i], args.slice(i + 1)];
}

function main(child) {
  let argv = process.argv.slice(2);
  log(argv, child);
  const state = load();
  if (argv[0] === '--context') argv = argv.slice(2);
  if (!argv.length) return 1;
  const [verb, ...args] = argv;
  if (state.daemon_down && verb !== 'context') {
    err('Cannot connect to the Docker daemon at unix:///run/user/1000/docker.sock. Is the docker daemon running?');
    if (verb === 'info') out(JSON.stringify({ ServerErrors: ['Cannot connect to the Docker daemon. Is the docker daemon running?'] }));
    return 1;
  }
  if (verb === 'context' && args[0] === 'show') { out(state.context ?? 'default'); return 0; }
  if (verb === 'context' && args[0] === 'ls') { for (const n of state.contexts ?? ['default']) out(n); return 0; }
  if (verb === 'info') { out(JSON.stringify(state.info)); return 0; }
  if (verb === 'ps') {
    if (args.includes('-q')) { for (const c of state.containers) if (c.State.Running) out(c.Id); return 0; }
    if (opt(args, '--format') !== PS_FORMAT) { err("the node's ps format changed"); return 2; }
    const label = opt(args, '--filter');
    for (const c of state.containers) {
      if (!args.includes('-a') && !c.State.Running) continue;
      if (label) {
        const kv = label.replace(/^label=/, ''); const k = kv.split('=')[0]; const v = kv.slice(k.length + 1);
        if ((c.Config.Labels || {})[k] !== v) continue;
      }
      const status = c.State.Running ? 'Up 3 minutes' : 'Exited (0) 1 minute ago';
      out(`${c.Id}\t${c.Name.replace(/^\//, '')}\t${c.State.Status}\t${status}`);
    }
    return 0;
  }
  if ((verb === 'container' && args[0] === 'inspect') || verb === 'inspect') {
    const refs = verb === 'container' ? args.slice(1) : args;
    const found = []; const missing = [];
    for (const r of refs) { const c = find(state, r); if (c) found.push(c); else missing.push(r); }
    out(JSON.stringify(found));
    for (const r of missing) err(`Error: No such container: ${r}`);
    return missing.length ? 1 : 0;
  }
  if (verb === 'image' && args[0] === 'inspect') {
    const fmt = opt(args, '--format');
    const refs = args.slice(1).filter((a) => a !== '--format' && a !== fmt);
    const found = []; const missing = [];
    for (const r of refs) {
      const img = state.images.find((i) => [...(i.RepoTags || []), ...(i.RepoDigests || [])].includes(r) || i.Id === r);
      if (img) found.push(img); else missing.push(r);
    }
    if (fmt === '{{.Id}}') { for (const i of found) out(i.Id); } else out(JSON.stringify(found));
    for (const r of missing) err(`Error response from daemon: No such image: ${r}`);
    return missing.length ? 1 : 0;
  }
  if (verb === 'image' && args[0] === 'ls') { for (const i of state.images) out(i.Id); return 0; }
  if (verb === 'images') {
    for (const i of state.images) for (const d of i.RepoDigests || []) out(d);
    return 0;
  }
  if (verb === 'network') {
    const [sub, ...rest] = args; const nets = (state.networks ||= {});
    if (sub === 'inspect') {
      const name = rest[rest.length - 1];
      if (!(name in nets)) { err(`Error response from daemon: network ${name} not found`); return 1; }
      out(nets[name].internal ? 'true' : 'false'); return 0;
    }
    if (sub === 'create') {
      nets[rest[rest.length - 1]] = { internal: rest.includes('--internal') }; save(state);
      out(crypto.randomBytes(32).toString('hex')); return 0;
    }
    if (sub === 'connect') return find(state, rest[1]) ? 0 : 1;
  }
  if (verb === 'run' || verb === 'create') {
    if ((state.fail || {})[verb]) { err(state.fail[verb]); return 125; }
    const [name, labels, image, cargs] = parseRun(args);
    if (find(state, name)) {
      err(`docker: Error response from daemon: Conflict. The container name "/${name}" is already in use by container "abc".`);
      return 125;
    }
    if (!state.images.some((i) => [...(i.RepoTags || []), ...(i.RepoDigests || [])].includes(image))) {
      err(`docker: Error response from daemon: No such image: ${image}`); return 125;
    }
    const doc = containerDoc(name, image, labels, verb === 'run', cargs);
    state.containers.push(doc); save(state); out(doc.Id); return 0;
  }
  if (verb === 'start' || verb === 'stop' || verb === 'rm') {
    const ref = args[args.length - 1]; const c = find(state, ref);
    if (!c) { err(`Error response from daemon: No such container: ${ref}`); return 1; }
    if (verb === 'rm') state.containers.splice(state.containers.indexOf(c), 1);
    else { c.State.Running = verb === 'start'; c.State.Status = verb === 'start' ? 'running' : 'exited'; }
    save(state); out(ref); return 0;
  }
  if (verb === 'logs') {
    const c = find(state, args[args.length - 1]);
    if (!c) { err(`Error response from daemon: No such container: ${args[args.length - 1]}`); return 1; }
    for (const line of (state.logs || {})[c.Name.replace(/^\//, '')] || []) out(line);
    if (args.includes('-f')) { setInterval(() => {}, 200); return null; }
    return 0;
  }
  if (verb === 'pull') {
    for (const line of ['Pulling from library/x', 'Digest: sha256:' + 'cd'.repeat(32), 'Status: Downloaded newer image']) out(line);
    state.images.push({ Id: 'sha256:' + 'ef'.repeat(32), RepoTags: [args[args.length - 1]], Size: 1, Created: '', Architecture: 'amd64' });
    save(state); return 0;
  }
  err(`fake docker: unsupported ${JSON.stringify(argv)}`);
  return 2;
}

module.exports = { main };
