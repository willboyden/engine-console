#!/usr/bin/env node
// Dump what GET /engines and GET /engines/{id}/params serve, from the real hub, for
// `node dev/mock-server.mjs --catalogs-file <out.json>`. That lets the Launch form and the wizard be checked on the
// production parameters and wording instead of the mock's samples.
//
//   node frontend/dev/dump-catalogs.mjs /tmp/catalogs.json      (from the repository root, after `make build`)
//
// It starts backend/target/debug/engine-console on an ephemeral loopback port with a throwaway HOME and data dir, the
// fleet off, OTLP off and Hugging Face pointed at a closed port, asks the two endpoints with the hub's bootstrap admin
// key, writes {engines: [...], params: {engine: [ParamSpec...]}} and stops the hub again. Nothing else is read or written.
import { spawn } from 'node:child_process';
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:net';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const REPO = dirname(dirname(dirname(fileURLToPath(import.meta.url))));
const HUB = join(REPO, 'backend', 'target', 'debug', 'engine-console');
const RESERVED = new Set([8791, 8792, 8793, 8794, 8795, 18796]);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const freePort = () => new Promise((resolve, reject) => {
  const s = createServer();
  s.once('error', reject);
  s.listen(0, '127.0.0.1', () => { const { port } = s.address(); s.close(() => resolve(port)); });
});

async function main(out) {
  if (!existsSync(HUB)) throw new Error(`${HUB} missing: run \`make build\` first`);
  let port;
  do port = await freePort(); while (RESERVED.has(port));
  const tmp = mkdtempSync(join(tmpdir(), 'ec-dump-'));
  const data = join(tmp, 'data');
  const env = {
    PATH: '/usr/bin:/bin', HOME: tmp, LANG: 'C.UTF-8', ENGINE_CONSOLE_PORT: String(port), ENGINE_CONSOLE_DATA_DIR: data,
    ENGINE_CONSOLE_TRUST_LOOPBACK: 'false', ENGINE_CONSOLE_ENV_FILE: join(tmp, 'no-env-file'), OTLP_ENABLED: 'false',
    HF_ENDPOINT: 'https://127.0.0.1:9', HOSTS_FILE: join(tmp, 'no-hosts.json'), FLEET_ENABLED: 'false',
  };
  const proc = spawn(HUB, [], { cwd: tmp, env, stdio: ['ignore', 'ignore', 'pipe'] });
  let log = '';
  proc.stderr.on('data', (d) => { log += d; });
  try {
    const base = `http://127.0.0.1:${port}`;
    const keyFile = join(data, 'bootstrap-admin.key');
    for (let i = 0; i < 150 && !existsSync(keyFile); i++) {
      if (proc.exitCode !== null) throw new Error(`the hub exited early:\n${log.slice(-2000)}`);
      await sleep(100);
    }
    const key = readFileSync(keyFile, 'utf8').trim();
    const get = async (p) => {
      for (let i = 0; ; i++) {
        try {
          const r = await fetch(base + p, { headers: { Authorization: `Bearer ${key}` } });
          if (!r.ok) throw new Error(`GET ${p}: ${r.status}`);
          return await r.json();
        } catch (e) {
          if (i > 50) throw e;
          await sleep(100);
        }
      }
    };
    const engines = await get('/api/v1/engines');
    const params = {};
    for (const e of engines) params[e.id] = await get(`/api/v1/engines/${encodeURIComponent(e.id)}/params`);
    writeFileSync(out, JSON.stringify({ engines, params }));
    console.log(Object.fromEntries(Object.entries(params).map(([k, v]) => [k, v.length])));
  } finally {
    proc.kill('SIGTERM');
    await new Promise((r) => (proc.exitCode !== null ? r() : proc.once('exit', r)));
    rmSync(tmp, { recursive: true, force: true });
  }
}

main(process.argv[2] || 'catalogs.json').catch((e) => { console.error(`dump-catalogs: ${e.message}`); process.exit(1); });
