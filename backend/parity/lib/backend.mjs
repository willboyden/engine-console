// Start one Engine Console backend (Python reference or the Rust port) with a hermetic setup: its own throwaway HOME
// and data dir, an ephemeral loopback port, OTLP off, Hugging Face pointed at a closed port, no env file, no hosts
// file. Everything the two backends share (seeded model roots, the node policy, fake CLIs) is created once per run by
// `makeShared` and is read-only for them, so the same paths appear in both answers.
import { spawn } from 'node:child_process';
import { chmodSync, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { createConnection, createServer } from 'node:net';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { pythonRelease, refOf } from './python-release.mjs';

export const PARITY = dirname(dirname(fileURLToPath(import.meta.url)));
/** The Cargo workspace (backend/). */
export const WORKSPACE = dirname(PARITY);
export const REPO = dirname(WORKSPACE);
export const FRONTEND = join(REPO, 'frontend');
export const RUST_BIN = join(WORKSPACE, 'target', 'debug', 'engine-console');
export const RUST_NODE_BIN = join(WORKSPACE, 'target', 'debug', 'ecnode');
export const FAKES = join(PARITY, 'fakes', 'bin');
// The fakes come first; the system dirs after them hold only what a node needs that is not faked (cat, sh...). The
// fake `docker` and `nvidia-smi` shadow any real ones, so a real daemon or GPU is never asked anything.
export const HERMETIC_PATH = `${FAKES}:/usr/local/bin:/usr/bin:/bin`;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export const freePort = () => new Promise((resolve, reject) => {
  const s = createServer();
  s.once('error', reject);
  s.listen(0, '127.0.0.1', () => { const { port } = s.address(); s.close(() => resolve(port)); });
});

const RESERVED = new Set([8791, 8792, 8793, 8794, 8795, 18796]);

async function ephemeralPort() {
  for (;;) {
    const p = await freePort();
    if (!RESERVED.has(p)) return p;
  }
}

/** A tiny but well-formed model (config.json + a safetensors header), as frontend/e2e/run.mjs seeds. */
function writeTinyModel(dir) {
  mkdirSync(dir, { recursive: true });
  writeFileSync(join(dir, 'config.json'), JSON.stringify({
    architectures: ['LlamaForCausalLM'], model_type: 'llama', hidden_size: 64, intermediate_size: 128,
    num_hidden_layers: 2, num_attention_heads: 4, num_key_value_heads: 2, vocab_size: 1000,
    max_position_embeddings: 4096, torch_dtype: 'bfloat16',
  }));
  const header = Buffer.from(JSON.stringify({
    __metadata__: { format: 'pt' },
    'model.embed_tokens.weight': { dtype: 'BF16', shape: [2, 2], data_offsets: [0, 8] },
    'lm_head.weight': { dtype: 'BF16', shape: [2, 2], data_offsets: [8, 16] },
  }));
  const len = Buffer.alloc(8); len.writeBigUInt64LE(BigInt(header.length));
  writeFileSync(join(dir, 'model.safetensors'), Buffer.concat([len, header, Buffer.alloc(16)]));
}

/** Everything both backends read: seeded model roots, the node policy, an empty HF cache dir. */
export function makeShared(tmp) {
  const shared = join(tmp, 'shared');
  const hub = join(shared, 'models', 'hf', 'hub');
  const rev = 'e2e0000000000000000000000000000000000001';
  const repo = join(hub, 'models--e2e--tiny-llama');
  mkdirSync(join(repo, 'refs'), { recursive: true });
  writeFileSync(join(repo, 'refs', 'main'), rev);
  writeTinyModel(join(repo, 'snapshots', rev));
  writeTinyModel(join(shared, 'models', 'plain', 'tiny-plain'));
  const etc = join(shared, 'etc');
  mkdirSync(etc, { recursive: true });
  chmodSync(etc, 0o755);
  const policy = join(etc, 'policy.json');
  writeFileSync(policy, JSON.stringify({
    _comment: 'parity harness: monitor mode, fake docker, no discovery probes',
    version: 1, mode: 'monitor', docker: { binary: join(FAKES, 'docker'), context: 'rootless' }, discovery_ports: [],
    model_roots: [
      { id: 'e2e-hf', path: hub, kind: 'hf-hub', label: 'e2e Hugging Face cache' },
      { id: 'e2e-plain', path: join(shared, 'models', 'plain'), kind: 'plain', label: 'e2e checkpoints' },
    ],
    scripts: [{
      name: 'demo', label: 'Demo engine script', engine: 'vllm', description: 'parity harness: never run (monitor mode)',
      containers: ['demo-vllm-*'], status_action: 'status',
      actions: [
        { name: 'status', label: 'Status', argv: ['/bin/true'] },
        { name: 'up', label: 'Start', argv: ['/bin/false'], args: ['small', 'large'], confirm: true },
        { name: 'down', label: 'Stop', argv: ['/bin/false'], danger: true },
      ],
    }],
  }, null, 1), { mode: 0o644 });
  chmodSync(policy, 0o644);            // a group- or world-writable policy is refused
  const hf = join(shared, 'hf-cache');
  mkdirSync(hf, { recursive: true });
  return { dir: shared, policy, hfCache: hf, hostsFile: join(shared, 'no-hosts.json') };
}

/** The local node command a hub runs, so both hubs talk to the same kind of node. `python` (or `python@<ref>`) is the
 * reference node of lib/python-release.mjs (stdlib only, no NVML: it reads the fake nvidia-smi on PATH); `rust` is
 * backend/target/debug/ecnode with
 * NVML switched off (PORTING.md §A2: `ECNODE_NO_NVML=1`) and its tools in ECNODE_TOOL_PATH = the hermetic PATH (the
 * Rust node never searches PATH: node hardening), so it too reads the fake nvidia-smi, docker and ip. The variables
 * ride in the command (`/usr/bin/env`) because a hub passes its local node only HOME, PATH, USER, LOGNAME, LANG and
 * DOCKER_*. */
export const RUST_NODE_PREFIX = ['/usr/bin/env', 'ECNODE_NO_NVML=1', `ECNODE_TOOL_PATH=${HERMETIC_PATH}`];
export function nodeCommand(kind, policy) {
  const ref = refOf(kind);
  if (ref) return [pythonRelease(ref).nodePy, '-I', '-m', 'ecnode', '--stdio', '--policy', policy];
  if (kind === 'rust') return [...RUST_NODE_PREFIX, RUST_NODE_BIN, '--stdio', '--policy', policy];
  return null;
}

/** Environment of one backend. `fleet`: false = FLEET_ENABLED=false (every host disabled), true = a real local node. */
export function backendEnv({ home, data, port, shared, fleet, node, hf }) {
  const env = {
    PATH: HERMETIC_PATH, HOME: home, LANG: 'C.UTF-8',
    ENGINE_CONSOLE_PORT: String(port), ENGINE_CONSOLE_DATA_DIR: data, ENGINE_CONSOLE_TRUST_LOOPBACK: 'false',
    ENGINE_CONSOLE_ENV_FILE: join(home, 'no-env-file'), OTLP_ENABLED: 'false',
    HF_ENDPOINT: 'https://127.0.0.1:9', HF_CACHE_DIR: shared.hfCache, HOSTS_FILE: shared.hostsFile,
    FLEET_ENABLED: fleet ? 'true' : 'false',
    // Both hubs serve this checkout's frontend: the exported Python tree has no frontend/ of its own.
    FRONTEND_DIR: FRONTEND,
    // The Rust hub finds its tools and gives its local node a fixed PATH, never the caller's (ec-hub tools.rs); this
    // is its explicit override, so its node reads the same fakes as the Python hub's. Python ignores the variable.
    ENGINE_CONSOLE_TOOL_PATH: HERMETIC_PATH,
  };
  if (hf) {
    // `hf` mode: the fake hub of lib/fakehf.mjs, reached only through its egress proxy (fail closed)
    Object.assign(env, {
      HF_ENDPOINT: hf.endpoint, HF_ALLOWED_HOSTS: JSON.stringify(hf.allowed), EGRESS_PROXY: hf.proxy,
      EGRESS_CA_BUNDLE: hf.ca, REQUIRE_EGRESS_PROXY: 'true', HF_TOKEN: hf.token,
    });
  }
  if (fleet) {
    env.NODE_POLICY = shared.policy;
    const cmd = nodeCommand(node, shared.policy);
    if (cmd) env.NODE_COMMAND = JSON.stringify(cmd);
  }
  return env;
}

function waitPort(port, ms) {
  return new Promise((resolve) => {
    const s = createConnection({ host: '127.0.0.1', port });
    const t = setTimeout(() => { s.destroy(); resolve(false); }, ms);
    s.once('connect', () => { clearTimeout(t); s.destroy(); resolve(true); });
    s.once('error', () => { clearTimeout(t); resolve(false); });
  });
}

/** The first `version = "..."` of a pyproject.toml or Cargo.toml (the package's, or the workspace's). */
function packageVersion(file) {
  const m = /^version\s*=\s*"([^"]+)"/m.exec(readFileSync(file, 'utf8'));
  if (!m) throw new Error(`${file}: no version`);
  return m[1];
}

/** Start a backend; resolves once its port accepts connections and its bootstrap admin key exists. Readiness is a
 * TCP connect, never an HTTP request, so both backends have served exactly the same requests when the corpus starts
 * (their request counters in /metrics are compared). */
export async function startBackend(kind, tmp, { shared, fleet, node, hf }) {
  const root = join(tmp, kind.replace(/[^A-Za-z0-9._-]/g, '_') + '-' + Math.random().toString(16).slice(2, 8));
  const home = join(root, 'home'); const data = join(root, 'data');
  for (const d of [home, data]) mkdirSync(d, { recursive: true });
  chmodSync(data, 0o700);
  const port = await ephemeralPort();
  const env = backendEnv({ home, data, port, shared, fleet, node, hf });
  let cmd; let args; let cwd;
  const ref = refOf(kind);
  let version;
  if (ref) {
    const py = pythonRelease(ref);
    cmd = py.hubPy; args = ['-I', '-m', 'engine_console.main']; cwd = py.backend;
    version = packageVersion(join(py.backend, 'pyproject.toml'));
  } else if (kind === 'rust') {
    if (!existsSync(RUST_BIN)) throw new Error(`${RUST_BIN} missing: run \`cargo build --locked\` in backend/ first`);
    cmd = RUST_BIN; args = []; cwd = WORKSPACE;
    version = packageVersion(join(WORKSPACE, 'Cargo.toml'));
  } else {
    throw new Error(`unknown backend ${kind}`);
  }
  const nodeBin = node && nodeCommand(node, shared.policy).find((a) => a.startsWith('/') && !a.startsWith('/usr/bin/env'));
  if (fleet && node && !existsSync(nodeBin)) {
    throw new Error(`the ${node} node is missing: ${nodeBin}`);
  }
  const proc = spawn(cmd, args, { cwd, env, stdio: ['ignore', 'pipe', 'pipe'] });
  track(proc);
  let log = '';
  const keep = (d) => { log += d; if (log.length > 200_000) log = log.slice(-100_000); };
  proc.stdout.on('data', keep); proc.stderr.on('data', keep);
  const exited = new Promise((r) => proc.once('exit', (code, sig) => r({ code, sig })));
  const b = { kind, proc, port, base: `http://127.0.0.1:${port}`, home, data, root, env, log: () => log, exited, version };
  for (let i = 0; i < 300; i++) {
    if (proc.exitCode !== null || proc.signalCode !== null) {
      throw new Error(`${kind} backend exited early (${proc.exitCode ?? proc.signalCode}):\n${log.slice(-3000)}`);
    }
    if (await waitPort(port, 200)) break;
    await sleep(100);
    if (i === 299) throw new Error(`${kind} backend never listened on ${port}:\n${log.slice(-3000)}`);
  }
  const keyFile = join(data, 'bootstrap-admin.key');
  for (let i = 0; i < 100 && !existsSync(keyFile); i++) await sleep(50);
  if (!existsSync(keyFile)) throw new Error(`${kind} backend wrote no ${keyFile}:\n${log.slice(-3000)}`);
  b.key = readFileSync(keyFile, 'utf8').trim();
  return b;
}

// Safety net: a harness that dies (an uncaught error) must not leave a server behind. Only children this process
// spawned are tracked, and only those are killed.
const LIVE = new Set();
function track(proc) {
  LIVE.add(proc);
  proc.once('exit', () => LIVE.delete(proc));
}
function killAll() {
  for (const p of LIVE) { try { p.kill('SIGKILL'); } catch { /* already gone */ } }
}
process.once('exit', killAll);
for (const ev of ['uncaughtException', 'unhandledRejection']) {
  process.once(ev, (e) => { killAll(); console.error(e); process.exit(1); });
}
for (const sig of ['SIGINT', 'SIGTERM']) process.once(sig, () => { killAll(); process.exit(130); });

/** Stop a backend this harness started (SIGTERM, then SIGKILL after 5 s). Never anything else. */
export async function stopBackend(b) {
  if (!b || b.proc.exitCode !== null || b.proc.signalCode !== null) return;
  b.proc.kill('SIGTERM');
  const done = await Promise.race([b.exited.then(() => true), sleep(5000).then(() => false)]);
  if (!done) { b.proc.kill('SIGKILL'); await Promise.race([b.exited, sleep(2000)]); }
}
