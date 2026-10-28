#!/usr/bin/env node
// Browser end-to-end check of Engine Console: the REAL backend (with its real local node), a real headless Chrome, the
// same auth mode as a real install (loopback NOT trusted; an admin key in sessionStorage, exactly where the UI keeps
// it). Zero dependencies: Node 22's built-in WebSocket speaks the DevTools protocol (ADR 0004: no npm).
//
//   cd frontend && node e2e/run.mjs          (or `make e2e` from the repo root)
//
// Needs: the debug binaries backend/target/debug/{engine-console,ecnode} (`make build`; `make e2e` builds them) and a Chrome: $CHROME_BIN (`make e2e` also reads `git config engine-console.chrome`),
// else google-chrome / chromium / chrome-headless-shell on PATH. Every route of js/app.js is opened twice, in Beginner
// mode (the default) and with it off, and every tab on the way; the run FAILS on:
//   * the error state ("Something went wrong", class `state error`) or the router's not-found state,
//   * the sign-in dialog (the key was refused),
//   * an uncaught exception, a console error, an API answer of 5xx, or a frontend file that does not load,
//   * a frontend file served without `Cache-Control: no-cache` (a browser that reuses a stale module next to a new one
//     breaks the new pages; Nemo Console 0.3.0 shipped broken for a user that way).
// The run is hermetic, the same on a laptop, a GPU server and CI: a throwaway HOME and data dir (nothing of the
// developer's env file, hosts.json, keys or Hugging Face cache is read), an environment built from scratch (no key
// variable of the caller's shell reaches the backend), OTLP off and pointed at a closed port, Hugging Face pointed at
// a closed loopback port, PATH = the fake CLIs of backend/parity/fakes/bin (docker, nvidia-smi: two fixed GPUs, ip) then
// /usr/bin:/bin, and the local node started with a throwaway MONITOR-mode policy whose docker binary does not exist and
// which probes no ports, with NVML off (ECNODE_NO_NVML=1) and its tools in ECNODE_TOOL_PATH = the fakes, so it reads the
// fake nvidia-smi; plus two tiny seeded models (a Hugging Face cache and a plain folder) so the Library, Launch, the fit
// estimate and the wizard render real rows. Monitor mode means the console cannot start, stop, pull or delete anything, and with no docker
// binary the node does not even list containers. A down or empty fleet must still render every page.
// E2E_LIVE=1 is the explicit opt-in to a live run: the caller's PATH and the hub's own default local node (the ecnode
// next to engine-console), which then reads this machine's real GPUs (read-only, NVML when the driver has it, else
// nvidia-smi).
import { spawn } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, existsSync, mkdirSync, chmodSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:net';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const FRONTEND = dirname(dirname(fileURLToPath(import.meta.url)));
// The Cargo workspace: the hub backend/target/debug/engine-console and its local node backend/target/debug/ecnode.
const BACKEND = join(FRONTEND, '..', 'backend');
const E2E_LIVE = process.env.E2E_LIVE === '1';
const FAKES = join(BACKEND, 'parity', 'fakes', 'bin');
const CLOSED_HOSTPORT = '127.0.0.1:9';   // nothing listens on port 9 (discard)
const SETTLE_MS = 400;          // the page must stay free of skeletons for this long to count as rendered
const PAGE_TIMEOUT_MS = 20000;
// Error states that are the correct render in this hermetic run, each with the reason. Nothing else is excused.
// Matched against the route the page ended on (after a redirect such as /models -> /discover).
const EXPECTED_ERRORS = {
  // Hugging Face is deliberately unreachable (HF_ENDPOINT is a closed loopback port, which the egress allowlist refuses):
  // Discover must say exactly that, with a retry, which is what an offline install shows. Until the e2e suite found it,
  // this error told an admin to "use an admin key"; such advice is never excused.
  '/discover': /egress allowlist[\s\S]*not on the HF allowlist/,
};
const NEVER_EXCUSED = /viewer key|admin key/i;
const EN = JSON.parse(readFileSync(join(FRONTEND, 'i18n', 'en.json'), 'utf8'));
const UNRESTRICTED = [EN['host.unrestricted_title'], EN['fleet.unrestricted_short']].filter(Boolean);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function findChrome() {
  if (process.env.CHROME_BIN) return process.env.CHROME_BIN;
  for (const dir of (process.env.PATH || '').split(':')) {
    for (const name of ['google-chrome', 'google-chrome-stable', 'chromium', 'chromium-browser', 'chrome-headless-shell']) {
      if (dir && existsSync(join(dir, name))) return join(dir, name);
    }
  }
  throw new Error('no Chrome found: set CHROME_BIN, or `git config engine-console.chrome <path>`, or put google-chrome / chromium / chrome-headless-shell on PATH');
}

const freePort = () => new Promise((resolve, reject) => {
  const s = createServer();
  s.once('error', reject);
  s.listen(0, '127.0.0.1', () => { const { port } = s.address(); s.close(() => resolve(port)); });
});

/** Minimal DevTools client: one browser WebSocket, flat sessions. */
export class Cdp {
  constructor(ws) { this.ws = ws; this.id = 0; this.waiting = new Map(); this.handlers = []; }
  static async connect(url) {
    const ws = new WebSocket(url);
    await new Promise((res, rej) => { ws.onopen = res; ws.onerror = () => rej(new Error(`cannot connect to ${url}`)); });
    const c = new Cdp(ws);
    ws.onmessage = (m) => {
      const msg = JSON.parse(m.data);
      if (msg.id && c.waiting.has(msg.id)) {
        const { resolve, reject } = c.waiting.get(msg.id); c.waiting.delete(msg.id);
        msg.error ? reject(new Error(`${msg.error.message} (${msg.error.code})`)) : resolve(msg.result);
      } else if (msg.method) { for (const h of c.handlers) h(msg); }
    };
    return c;
  }
  send(method, params = {}, sessionId) {
    const id = ++this.id;
    this.ws.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }));
    return new Promise((resolve, reject) => this.waiting.set(id, { resolve, reject }));
  }
  on(h) { this.handlers.push(h); }
  close() { try { this.ws.close(); } catch { /* ignore */ } }
}

/** A tiny but well-formed model in `dir`: config.json plus a safetensors file with a valid header (two BF16 tensors). It is
 * never loaded (monitor mode), but the library, the launch form, the fit estimate and the wizard all read it. */
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

/** Model roots for the run: a Hugging Face cache holding e2e/tiny-llama, and a plain checkpoint folder. */
export function seedModels(dir) {
  const hub = join(dir, 'hf', 'hub'); const rev = 'e2e0000000000000000000000000000000000001';
  const repo = join(hub, 'models--e2e--tiny-llama');
  mkdirSync(join(repo, 'refs'), { recursive: true });
  writeFileSync(join(repo, 'refs', 'main'), rev);
  writeTinyModel(join(repo, 'snapshots', rev));
  writeTinyModel(join(dir, 'plain', 'tiny-plain'));
  return [
    { id: 'e2e-hf', path: hub, kind: 'hf-hub', label: 'e2e Hugging Face cache' },
    { id: 'e2e-plain', path: join(dir, 'plain'), kind: 'plain', label: 'e2e checkpoints' },
  ];
}

/** The local node's policy for the run: monitor mode, a docker binary that does not exist, no port probes, and the
 * seeded model roots (read-only). */
export function monitorPolicy(dir, roots = []) {
  const path = join(dir, 'policy.json');
  writeFileSync(path, JSON.stringify({
    _comment: 'e2e: monitor mode, no docker, no discovery probes',
    version: 1, mode: 'monitor', docker: { binary: '/nonexistent/e2e/docker' }, discovery_ports: [], model_roots: roots,
  }), { mode: 0o644 });
  chmodSync(path, 0o644);           // a group- or world-writable policy is refused; the umask may have widened it
  return path;
}

export async function startBackend(tmp, { port } = {}) {
  for (const k of ['E2E_BACKEND', 'E2E_NODE']) {
    if (process.env[k] && process.env[k] !== 'rust' && process.env[k] !== 'default') {
      throw new Error(`${k}=${process.env[k]}: the Python hub and node were removed in 0.5.0; the suite runs the Rust ones`);
    }
  }
  const hub = join(BACKEND, 'target', 'debug', 'engine-console');
  const rustNode = join(BACKEND, 'target', 'debug', 'ecnode');
  for (const bin of [hub, rustNode]) {
    if (!existsSync(bin)) throw new Error(`${bin} missing: run \`make build\` (or \`make e2e\`, which builds) first`);
  }
  port ||= await freePort();
  const home = join(tmp, 'home'); const data = join(tmp, 'data'); const etc = join(tmp, 'etc');
  for (const d of [home, data, etc]) mkdirSync(d, { recursive: true });
  chmodSync(data, 0o700); chmodSync(etc, 0o755);
  if (!E2E_LIVE && !existsSync(join(FAKES, 'nvidia-smi'))) throw new Error(`${FAKES} missing: the hermetic run needs the fake CLIs (or set E2E_LIVE=1)`);
  const env = {
    PATH: E2E_LIVE ? (process.env.PATH || '/usr/bin:/bin') : `${FAKES}:/usr/bin:/bin`, HOME: home, LANG: 'C.UTF-8',
    ENGINE_CONSOLE_PORT: String(port), ENGINE_CONSOLE_DATA_DIR: data, ENGINE_CONSOLE_TRUST_LOOPBACK: 'false',
    ENGINE_CONSOLE_ENV_FILE: join(home, 'no-env-file'), OTLP_ENABLED: 'false', OTLP_ENDPOINT: CLOSED_HOSTPORT,
    NODE_POLICY: monitorPolicy(etc, seedModels(join(tmp, 'models'))), HF_ENDPOINT: `https://${CLOSED_HOSTPORT}`,
  };
  // The local node. Hermetic: backend/target/debug/ecnode with NVML off and its tools (it never searches PATH) in
  // ECNODE_TOOL_PATH = the fakes; both ride in the command, because a hub passes its node only HOME, PATH, USER,
  // LOGNAME, LANG and DOCKER_*. E2E_LIVE=1: the hub's own default node (the ecnode next to it, with NODE_POLICY).
  if (!E2E_LIVE) {
    const node = ['/usr/bin/env', 'ECNODE_NO_NVML=1', `ECNODE_TOOL_PATH=${FAKES}:/usr/bin:/bin`, rustNode];
    env.NODE_COMMAND = JSON.stringify([...node, '--stdio', '--policy', env.NODE_POLICY]);
  }
  // The hub never searches PATH for its tools and gives its local node a fixed PATH (ec-hub tools.rs); this explicit
  // override keeps both on the run's PATH (the fakes, or the caller's with E2E_LIVE=1).
  env.ENGINE_CONSOLE_TOOL_PATH = env.PATH;
  const proc = spawn(hub, [], { cwd: BACKEND, env, stdio: ['ignore', 'pipe', 'pipe'] });
  let log = ''; proc.stdout.on('data', (d) => { log += d; }); proc.stderr.on('data', (d) => { log += d; });
  const base = `http://127.0.0.1:${port}`;
  for (let i = 0; i < 150; i++) {
    if (proc.exitCode !== null) throw new Error(`backend exited early (${proc.exitCode}):\n${log.slice(-2000)}`);
    try { if ((await fetch(`${base}/api/v1/health`)).ok) break; } catch { /* not up yet */ }
    await sleep(200);
  }
  const keyFile = join(data, 'bootstrap-admin.key');
  for (let i = 0; i < 50 && !existsSync(keyFile); i++) await sleep(100);
  const key = readFileSync(keyFile, 'utf8').trim();
  return { proc, base, key, log: () => log };
}

export async function startChrome(profile) {
  const port = await freePort();
  const proc = spawn(findChrome(), [
    '--headless', `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`, '--no-sandbox', '--disable-gpu',
    '--disable-dev-shm-usage', '--disable-background-networking', '--disable-sync', '--no-first-run',
    '--no-default-browser-check', '--window-size=1440,1000', 'about:blank'], { stdio: ['ignore', 'ignore', 'pipe'] });
  let err = ''; proc.stderr.on('data', (d) => { err += d; });
  for (let i = 0; i < 100; i++) {
    try {
      const v = await (await fetch(`http://127.0.0.1:${port}/json/version`)).json();
      if (v.webSocketDebuggerUrl) return { proc, wsUrl: v.webSocketDebuggerUrl, version: v.Browser };
    } catch { /* not listening yet */ }
    await sleep(150);
  }
  proc.kill('SIGKILL');
  throw new Error(`Chrome never exposed DevTools:\n${err.slice(-1500)}`);
}

/** The routes to open, from js/app.js's ROUTES. Parameterised ones get a value that exists in a hermetic run
 * ('/hosts/:id' -> the local machine) or are skipped (an instance needs an engine); the advanced launch form is reached
 * with `?advanced=1` because Beginner mode turns /launch into the wizard. */
export function routesFrom(appSource) {
  const out = [];
  for (const [, path] of appSource.matchAll(/\{ path: '(\/[^']*)'/g)) {
    if (path === '/hosts/:id') out.push('/hosts/local');
    else if (!path.includes(':')) out.push(path);
  }
  out.push('/launch?advanced=1');
  return out;
}

export async function main(argv = process.argv.slice(2)) {
  const keepProfile = argv.includes('--keep-profile');
  const tmp = mkdtempSync(join(tmpdir(), 'ec-e2e-'));
  const failures = [];
  let backend; let chrome; let cdp;
  try {
    backend = await startBackend(tmp);
    chrome = await startChrome(join(tmp, 'profile'));
    cdp = await Cdp.connect(chrome.wsUrl);
    const { targetId } = await cdp.send('Target.createTarget', { url: 'about:blank' });
    const { sessionId } = await cdp.send('Target.attachToTarget', { targetId, flatten: true });
    const s = (m, p) => cdp.send(m, p, sessionId);
    await Promise.all([s('Page.enable'), s('Runtime.enable'), s('Log.enable'), s('Network.enable')]);

    let route = '(sign-in)';
    // Before the key is in sessionStorage the app's first calls answer 401 by design: not a failure.
    const problems = (msg) => { if (route !== '(sign-in)') failures.push(`${route}: ${msg}`); };
    const seenHeaders = new Map();
    cdp.on((m) => {
      if (m.sessionId !== sessionId) return;
      if (m.method === 'Runtime.exceptionThrown') {
        const d = m.params.exceptionDetails; problems(`uncaught exception: ${d.exception?.description || d.text}`);
      } else if (m.method === 'Runtime.consoleAPICalled' && m.params.type === 'error') {
        problems(`console.error: ${m.params.args.map((a) => a.value ?? a.description ?? '').join(' ').slice(0, 300)}`);
      } else if (m.method === 'Log.entryAdded' && m.params.entry.level === 'error' && m.params.entry.source !== 'network') {
        problems(`browser error: ${m.params.entry.text.slice(0, 300)} ${m.params.entry.url || ''}`);    // network: below
      } else if (m.method === 'Network.responseReceived') {
        const r = m.params.response; const u = new URL(r.url);
        if (u.origin !== backend.base) return;
        if (u.pathname.startsWith('/api/')) { if (r.status >= 500) problems(`HTTP ${r.status} from ${u.pathname}`); return; }
        if (r.status >= 400) problems(`frontend file ${u.pathname} answered ${r.status}`);
        seenHeaders.set(u.pathname, Object.entries(r.headers).find(([k]) => k.toLowerCase() === 'cache-control')?.[1] || '');
      }
    });

    const evaluate = async (expr) => (await s('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: true })).result.value;
    // Sign in the way a person does: the key goes into this tab's sessionStorage, then the app boots with it.
    await s('Page.navigate', { url: `${backend.base}/` });
    await sleep(500);
    await evaluate(`sessionStorage.setItem('ec.apiKey', ${JSON.stringify(backend.key)}); true`);
    await s('Page.reload', { ignoreCache: false });
    await sleep(800);

    /** Wait until the page has rendered (no skeleton for SETTLE_MS) and report the visible error states. */
    const settle = async () => {
      const t0 = Date.now(); let quietSince = 0;
      while (Date.now() - t0 < PAGE_TIMEOUT_MS) {
        const st = await evaluate(`(() => {
          const main = document.querySelector('#main');
          const visible = (e) => e.offsetParent !== null;
          const busy = [...document.querySelectorAll('#main .skeleton, #main [aria-busy="true"]')].some(visible);
          const errs = [...document.querySelectorAll('#main .state.error')].filter(visible)
            .map((e) => e.innerText.replace(/\\s+/g, ' ').slice(0, 600));
          const notFound = !!document.querySelector('#main > .state.empty');   // the router's own not-found state
          const signIn = [...document.querySelectorAll('#key-in')].some(visible);
          const text = main ? main.innerText : '';
          const unrestricted = [${JSON.stringify(UNRESTRICTED)}].flat().some((w) => text.includes(w));
          return { busy, errs, notFound, signIn, unrestricted, hasContent: !!main && main.children.length > 0 };
        })()`);
        if (st.errs.length || st.notFound || st.signIn) return st;
        if (!st.busy && st.hasContent) { quietSince ||= Date.now(); if (Date.now() - quietSince >= SETTLE_MS) return st; }
        else quietSince = 0;
        await sleep(100);
      }
      return { timeout: true, errs: [], notFound: false, signIn: false };
    };
    const judge = async (st) => {
      if (st.timeout) problems('did not finish rendering');
      if (st.notFound) problems('shows the not-found state');
      if (st.signIn) problems('shows the sign-in dialog (the key was refused)');
      // The run's only machine is local: its node is a subprocess, never an ssh login, so no key warning applies.
      if (st.unrestricted) problems('warns about an unrestricted ssh key on the local machine, which has no ssh link');
      const shown = (await evaluate('location.hash')).replace(/^#/, '').split('?')[0] || '/';
      const excused = EXPECTED_ERRORS[shown];
      for (const e of st.errs) if (!excused?.test(e) || NEVER_EXCUSED.test(e)) problems(`error state: ${e}`);
    };

    const paths = routesFrom(readFileSync(join(FRONTEND, 'js', 'app.js'), 'utf8'));
    if (paths.length < 12) throw new Error(`only ${paths.length} routes parsed from js/app.js`);
    let pages = 0;
    for (const beginner of [true, false]) {
      if (!beginner) {               // the same routes with Beginner mode off (a browser-local preference)
        route = '(beginner off)';
        await evaluate(`localStorage.setItem('ec.beginner', '0'); true`);
        await s('Page.reload', { ignoreCache: false });
        await sleep(800);
      }
      const mode = beginner ? 'beginner' : 'advanced';
      for (const path of paths) {
        route = `${path} [${mode}]`;
        await evaluate(`location.hash = '#${path}'; true`);
        await sleep(150);
        let st = await settle(); pages++;
        await judge(st);
        const tabs = await evaluate(`[...document.querySelectorAll('#main [role="tab"], #main .tabs > .tab')]
          .map((b, i) => b.id || (b.dataset.e2e = 'tab' + i, '[data-e2e="tab' + i + '"]'))`);
        for (const id of tabs || []) {
          route = `${path} [${mode}] › ${id}`;
          const sel = id.startsWith('[') ? id : `#${id}`;
          await evaluate(`document.querySelector(${JSON.stringify(sel)})?.click(); true`);
          await sleep(150);
          st = await settle(); pages++;
          await judge(st);
        }
      }
    }
    route = '(static files)';
    for (const [p, cc] of seenHeaders) if (!/\bno-cache\b/.test(cc)) problems(`${p} served without Cache-Control: no-cache (got '${cc}')`);
    if (!seenHeaders.size) problems('no frontend file was observed (the page never loaded?)');

    console.log(`e2e: Rust hub and local node, ${E2E_LIVE ? 'LIVE (caller PATH, real GPUs read-only)' : 'hermetic (fake CLIs, NVML off)'}`);
    console.log(`e2e: ${chrome.version}; ${paths.length} routes x 2 modes, ${pages} pages/tabs rendered, ${seenHeaders.size} static files checked`);
  } catch (e) {
    failures.push(`harness: ${e.message}`);
  } finally {
    cdp?.close();
    chrome?.proc.kill('SIGKILL');
    if (backend) {
      backend.proc.kill('SIGTERM');
      for (let i = 0; i < 50 && backend.proc.exitCode === null; i++) await sleep(100);
      if (backend.proc.exitCode === null) backend.proc.kill('SIGKILL');
    }
    if (!keepProfile) rmSync(tmp, { recursive: true, force: true });
  }
  if (failures.length) {
    console.error(`e2e: FAILED (${failures.length})`);
    for (const f of failures) console.error(`  - ${f}`);
    if (backend) console.error(`backend log tail:\n${backend.log().slice(-1500)}`);
    return 1;
  }
  console.log('e2e: all pages rendered with no error state, no console error, and revalidating static files');
  return 0;
}

if (import.meta.url === `file://${process.argv[1]}`) process.exit(await main());
