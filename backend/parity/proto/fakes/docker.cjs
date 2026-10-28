// A fake `docker` CLI for the protocol cross-matrix (subset of node/tests/fakes/fake_docker.py, in Node so the
// matrix survives the Python reference's removal). State: $FAKE_STATE_DIR/docker.json (the wrapper sets it, because
// the node gives every child a scrubbed environment). Read-only except `network create`. Never touches a real daemon.
'use strict';
const fs = require('node:fs');
const path = require('node:path');

const STATE = path.join(process.env.FAKE_STATE_DIR, 'docker.json');
const PS_FORMAT = '{{.ID}}\t{{.Names}}\t{{.State}}\t{{.Status}}';
const out = (s) => process.stdout.write(s + '\n');
const err = (s) => process.stderr.write(s + '\n');
const load = () => JSON.parse(fs.readFileSync(STATE, 'utf8'));
const save = (s) => fs.writeFileSync(STATE, JSON.stringify(s));
const opt = (args, flag) => (args.includes(flag) ? args[args.indexOf(flag) + 1] : null);
const find = (state, ref) => state.containers.find((c) => c.Name.replace(/^\//, '') === ref || c.Id === ref
  || (ref.length >= 6 && c.Id.startsWith(ref))) || null;

function main(argv) {
  const state = load();
  if (argv[0] === '--context') argv = argv.slice(2);
  if (!argv.length) return 1;
  const [verb, ...args] = argv;
  if (verb === 'context' && args[0] === 'show') { out(state.context || 'default'); return 0; }
  if (verb === 'context' && args[0] === 'ls') { for (const n of state.contexts || ['default']) out(n); return 0; }
  if (verb === 'info') { out(JSON.stringify(state.info)); return 0; }
  if (verb === 'ps') {
    if (args.includes('-q')) { for (const c of state.containers) if (c.State.Running) out(c.Id); return 0; }
    if (opt(args, '--format') !== PS_FORMAT) { err("fake docker: the node's ps format changed"); return 2; }
    const label = opt(args, '--filter');
    for (const c of state.containers) {
      if (!args.includes('-a') && !c.State.Running) continue;
      if (label) {
        const [k, v] = label.replace(/^label=/, '').split('=');
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
    if (fmt === '{{.Id}}') for (const i of found) out(i.Id); else out(JSON.stringify(found));
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
    const name = rest[rest.length - 1];
    if (sub === 'inspect') {
      if (!(name in nets)) { err(`Error response from daemon: network ${name} not found`); return 1; }
      out(nets[name].internal ? 'true' : 'false'); return 0;
    }
    if (sub === 'create') { nets[name] = { internal: rest.includes('--internal') }; save(state); out('n'.repeat(64)); return 0; }
  }
  if (verb === 'logs') {
    const c = find(state, args[args.length - 1]);
    if (!c) { err(`Error response from daemon: No such container: ${args[args.length - 1]}`); return 1; }
    for (const line of (state.logs || {})[c.Name.replace(/^\//, '')] || []) out(line);
    if (args.includes('-f')) setInterval(() => {}, 200);     // follow: never ends (the node kills its group)
    return args.includes('-f') ? null : 0;
  }
  err(`fake docker: unsupported ${JSON.stringify(argv)}`);
  return 2;
}

const code = main(process.argv.slice(2));
if (code !== null) process.exitCode = code;
