// A fake Hugging Face hub for the harness's `hf` mode: both backends reach it exactly as they reach the real
// one behind an egress proxy. HF_ENDPOINT is https://hf.test (the .test TLD never resolves, so nothing can leak to a
// real host); EGRESS_PROXY is a loopback CONNECT proxy that tunnels hf.test and *.hf.test to a local HTTPS server
// and refuses every other target; EGRESS_CA_BUNDLE is a CA made for this run (openssl CLI, in the run's temp dir)
// that signed the server's certificate; REQUIRE_EGRESS_PROXY is on, so a backend that tried to go direct would
// fail instead. Node stdlib only.
//
// The hub serves the REST calls the HF client makes (huggingface_hub's shapes): GET /api/models (search),
// GET /api/models/<repo>/revision/<rev> (model info with siblings), GET /<repo>/resolve/<rev>/<path> (files, with
// Range), GET /<repo>/raw/<rev>/README.md (the card), plus the failures the client maps: a gated repo (401/403), a
// missing one (404), a hub error (500), an answer that is not JSON, a redirect inside the allowlist and one off it.
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { createServer as createHttpServer } from 'node:http';
import { createServer as createHttpsServer } from 'node:https';
import { connect } from 'node:net';
import { join } from 'node:path';

export const HF_HOST = 'hf.test';
export const HF_ENDPOINT = `https://${HF_HOST}`;
export const HF_ALLOWED = [HF_HOST, `*.${HF_HOST}`];
/** A fake token: it proves the backends send it to the hub (and only there), it is not a secret. */
export const HF_FAKE_TOKEN = 'hf_parity_fake_token_0000';
export const COMMIT = 'c0ffee00000000000000000000000000000000ff';

function openssl(dir, args) {
  const r = spawnSync('openssl', args, { cwd: dir, stdio: ['ignore', 'pipe', 'pipe'], env: { PATH: '/usr/bin:/bin' } });
  if (r.status !== 0) throw new Error(`openssl ${args.join(' ')}: ${r.stderr}`);
}

/** A CA and a server certificate for hf.test and *.hf.test, made for this run only. */
export function makeCerts(dir) {
  mkdirSync(dir, { recursive: true });
  openssl(dir, ['req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-keyout', 'ca.key', '-out', 'ca.pem', '-days', '2',
    '-subj', '/CN=parity-hf-test-ca', '-addext', 'basicConstraints=critical,CA:TRUE',
    '-addext', 'keyUsage=critical,keyCertSign,cRLSign']);
  openssl(dir, ['req', '-newkey', 'rsa:2048', '-nodes', '-keyout', 'srv.key', '-out', 'srv.csr', '-subj', `/CN=${HF_HOST}`]);
  writeFileSync(join(dir, 'ext.cnf'), `subjectAltName=DNS:${HF_HOST},DNS:*.${HF_HOST}\nextendedKeyUsage=serverAuth\n`
    + 'basicConstraints=CA:FALSE\nkeyUsage=digitalSignature,keyEncipherment\n');
  openssl(dir, ['x509', '-req', '-in', 'srv.csr', '-CA', 'ca.pem', '-CAkey', 'ca.key', '-CAcreateserial', '-out', 'srv.pem',
    '-days', '2', '-extfile', 'ext.cnf']);
  return { ca: join(dir, 'ca.pem'), cert: join(dir, 'srv.pem'), key: join(dir, 'srv.key') };
}

const sha256 = (b) => createHash('sha256').update(b).digest('hex');
const LFS = ['.safetensors', '.gguf', '.bin'];

/** Deterministic bytes. */
function bytes(n, seed) {
  const out = Buffer.alloc(n);
  let x = seed >>> 0 || 1;
  for (let i = 0; i < n; i++) { x ^= x << 13; x >>>= 0; x ^= x >>> 17; x ^= x << 5; x >>>= 0; out[i] = x & 0xff; }
  return out;
}

const QWEN_CFG = {
  architectures: ['Qwen3ForCausalLM'], model_type: 'qwen3', hidden_size: 5120, intermediate_size: 25600,
  num_hidden_layers: 64, num_attention_heads: 64, num_key_value_heads: 8, head_dim: 128, vocab_size: 151936,
  max_position_embeddings: 40960, torch_dtype: 'bfloat16', rope_theta: 1000000,
};
const TINY_CFG = {
  architectures: ['LlamaForCausalLM'], model_type: 'llama', hidden_size: 64, intermediate_size: 128,
  num_hidden_layers: 2, num_attention_heads: 4, num_key_value_heads: 2, vocab_size: 1000,
  max_position_embeddings: 4096, torch_dtype: 'bfloat16',
};
const MOE_CFG = {
  architectures: ['Qwen3MoeForCausalLM'], model_type: 'qwen3_moe', hidden_size: 2048, num_hidden_layers: 48,
  num_attention_heads: 32, num_key_value_heads: 4, head_dim: 128, num_experts: 128, num_experts_per_tok: 8,
  moe_intermediate_size: 768, max_position_embeddings: 262144, torch_dtype: 'bfloat16',
  quantization_config: { quant_method: 'fp8', weight_block_size: [128, 128] },
};

/** The repos: `files` are [path, Buffer], `meta` the model-info fields, `fail` a forced answer. */
function repos() {
  const json = (o) => Buffer.from(JSON.stringify(o));
  const r = {};
  const add = (id, files, meta = {}, extra = {}) => { r[id] = { id, files, meta, ...extra }; };
  add('Qwen/Qwen3-32B', [['config.json', json(QWEN_CFG)], ['README.md', Buffer.from('# Qwen3-32B\nA model card.\n')],
    ['model-00001-of-00002.safetensors', bytes(4096, 1)], ['model-00002-of-00002.safetensors', bytes(2048, 2)],
    ['tokenizer.json', json({ version: '1.0' })]],
  { downloads: 500000, likes: 900, pipeline_tag: 'text-generation', library_name: 'transformers',
    tags: ['transformers', 'safetensors', 'qwen3', 'text-generation', 'license:apache-2.0'],
    safetensors: { total: 32762000000, parameters: { BF16: 32762000000 } } });
  add('Qwen/Qwen3-30B-A3B-FP8', [['config.json', json(MOE_CFG)], ['model.safetensors', bytes(3000, 3)]],
    { downloads: 120000, likes: 300, pipeline_tag: 'text-generation', library_name: 'transformers',
      tags: ['transformers', 'fp8', 'qwen3_moe'], cardData: { license: 'apache-2.0' },
      safetensors: { total: 30532000000, parameters: { F8_E4M3: 30000000000, BF16: 532000000 } } });
  add('e2e/tiny-llama', [['config.json', json(TINY_CFG)], ['model.safetensors', bytes(1024, 4)],
    ['README.md', Buffer.from('tiny\n')]],
  { downloads: 3, likes: 0, pipeline_tag: 'text-generation', library_name: 'transformers', tags: ['llama'],
    safetensors: { total: 16, parameters: { BF16: 16 } } });
  add('org/m-GGUF', [['m-Q4_K_M.gguf', bytes(400, 5)], ['m-Q8_0-00001-of-00002.gguf', bytes(500, 6)],
    ['m-Q8_0-00002-of-00002.gguf', bytes(300, 7)], ['mmproj-F16.gguf', bytes(50, 8)], ['README.md', Buffer.from('gguf\n')]],
  { downloads: 77, likes: 7, pipeline_tag: 'text-generation', library_name: 'gguf', tags: ['gguf', 'license:mit'] });
  add('org/noconfig', [['weights.bin', bytes(100, 9)]], { downloads: 1, likes: 1, tags: [] });
  add('meta/gated', [['config.json', json(TINY_CFG)]], { gated: 'manual', downloads: 10, likes: 2, tags: ['llama'] },
    { fail: 403 });
  add('meta/private', [['config.json', json(TINY_CFG)]], { downloads: 0, likes: 0, tags: [] }, { fail: 401 });
  add('org/broken', [], { tags: [] }, { fail: 500 });
  add('org/notjson', [], { tags: [] }, { fail: 'notjson' });
  add('org/moved', [], { tags: [] }, { redirect: `${HF_ENDPOINT}/api/models/e2e/tiny-llama/revision/main` });
  add('org/offlist', [], { tags: [] }, { redirect: 'https://evil.test/api/models/x/y' });
  add('org/cfg-gated', [['config.json', json(TINY_CFG)], ['model.safetensors', bytes(64, 10)]],
    { downloads: 4, likes: 0, tags: [] }, { configFail: 401 });
  return r;
}

function info(repo, full) {
  const siblings = repo.files.map(([p, b]) => {
    const s = { rfilename: p };
    if (full) {
      s.size = b.length;
      if (LFS.some((x) => p.endsWith(x))) s.lfs = { sha256: sha256(b), size: b.length, pointerSize: 134 };
    }
    return s;
  });
  return { id: repo.id, modelId: repo.id, sha: COMMIT, private: false, gated: repo.meta.gated ?? false,
    downloads: repo.meta.downloads ?? 0, likes: repo.meta.likes ?? 0, tags: repo.meta.tags ?? [],
    pipeline_tag: repo.meta.pipeline_tag, library_name: repo.meta.library_name,
    ...(repo.meta.cardData ? { cardData: repo.meta.cardData } : {}),
    ...(repo.meta.safetensors ? { safetensors: repo.meta.safetensors } : {}), siblings };
}

function send(res, status, body, headers = {}) {
  const b = Buffer.isBuffer(body) ? body : Buffer.from(typeof body === 'string' ? body : JSON.stringify(body));
  const ct = Buffer.isBuffer(body) ? 'application/octet-stream' : typeof body === 'string' ? 'text/plain; charset=utf-8'
    : 'application/json';
  res.writeHead(status, { 'content-type': ct, 'content-length': String(b.length), ...headers });
  res.end(b);
}

/** One request to the hub. `log` collects [method, path, has-authorization] for the run's report. */
function handle(hub, req, res, log) {
  const url = new URL(req.url, HF_ENDPOINT);
  const path = decodeURIComponent(url.pathname);
  log.push([req.method, url.pathname + url.search, Boolean(req.headers.authorization)]);
  if (req.method !== 'GET') return send(res, 405, { error: 'method not allowed' });
  if (path === '/api/models') {
    const q = (url.searchParams.get('search') || '').toLowerCase();
    const limit = Number(url.searchParams.get('limit') || '20');
    const sort = url.searchParams.get('sort') || 'downloads';
    const filter = url.searchParams.get('filter');
    const task = url.searchParams.get('pipeline_tag');
    const library = url.searchParams.get('library');
    let list = Object.values(hub).filter((r) => !r.fail && !r.redirect && r.id.toLowerCase().includes(q))
      .filter((r) => !filter || (r.meta.tags || []).includes(filter))
      .filter((r) => !task || r.meta.pipeline_tag === task)
      .filter((r) => !library || r.meta.library_name === library);
    list.sort((a, b) => (b.meta[sort] ?? 0) - (a.meta[sort] ?? 0) || a.id.localeCompare(b.id));
    list = list.slice(0, Math.max(0, limit));
    return send(res, 200, list.map((r) => info(r, false)));
  }
  let m = /^\/api\/models\/(.+)\/revision\/([^/]+)$/.exec(path);
  if (m) {
    const repo = hub[m[1]];
    if (!repo) return send(res, 404, { error: 'Repository not found' });
    if (repo.redirect) return send(res, 307, '', { location: repo.redirect });
    if (repo.fail === 'notjson') return send(res, 200, '<html>not json</html>');
    if (repo.fail === 401 && req.headers.authorization) return send(res, 200, info(repo, true));
    if (repo.fail) return send(res, repo.fail, { error: `forced ${repo.fail}` });
    return send(res, 200, info(repo, true));
  }
  m = /^\/(.+?)\/(resolve|raw)\/([^/]+)\/(.+)$/.exec(path);
  if (m) {
    const repo = hub[m[1]];
    if (!repo) return send(res, 404, { error: 'Repository not found' });
    if (repo.fail === 401 && req.headers.authorization) { /* a token opens it */ } else if (repo.fail) {
      return send(res, typeof repo.fail === 'number' ? repo.fail : 500, { error: 'forced' });
    }
    if (m[4] === 'config.json' && repo.configFail) return send(res, repo.configFail, { error: 'gated config' });
    const f = repo.files.find(([p]) => p === m[4]);
    if (!f) return send(res, 404, { error: 'Entry not found' });
    const body = f[1];
    const range = /^bytes=(\d+)-$/.exec(req.headers.range || '');
    if (range) {
      const start = Math.min(Number(range[1]), body.length);
      return send(res, 206, body.subarray(start), { 'content-range': `bytes ${start}-${body.length - 1}/${body.length}` });
    }
    return send(res, 200, m[2] === 'raw' ? body.toString('utf8') : body);
  }
  return send(res, 404, { error: 'not found' });
}

/**
 * Start the hub and its egress proxy. Returns {endpoint, proxy, ca, allowed, token, connects, requests, stop()}.
 * `connects` lists every CONNECT target the proxy saw (refused ones included), `requests` every hub request.
 */
export async function startFakeHf(dir) {
  const certs = makeCerts(join(dir, 'hf-pki'));
  const hub = repos();
  const requests = [];
  const connects = [];
  const https = createHttpsServer({ key: readFileSync(certs.key), cert: readFileSync(certs.cert) },
    (req, res) => handle(hub, req, res, requests));
  await new Promise((r) => https.listen(0, '127.0.0.1', r));
  const hubPort = https.address().port;
  const proxy = createHttpServer((req, res) => { res.writeHead(405); res.end(); });
  const sockets = new Set();
  const keep = (x) => { sockets.add(x); x.once('close', () => sockets.delete(x)); };
  https.on('connection', keep);
  proxy.on('connection', keep);
  proxy.on('connect', (req, sock, head) => {
    connects.push(req.url);
    const [host, port] = req.url.split(':');
    const ok = (host === HF_HOST || host.endsWith(`.${HF_HOST}`)) && port === '443';
    if (!ok) {
      // the lab proxy's block page, as the HF client recognises it
      sock.end('HTTP/1.1 403 Forbidden\r\nContent-Type: text/plain\r\nConnection: close\r\n\r\n'
        + 'egress blocked by addon_guard: host not in allowlist.txt\n');
      return;
    }
    const up = connect(hubPort, '127.0.0.1', () => {
      sock.write('HTTP/1.1 200 Connection established\r\n\r\n');
      if (head && head.length) up.write(head);
      up.pipe(sock); sock.pipe(up);
    });
    keep(up);
    up.on('error', () => sock.destroy());
    sock.on('error', () => up.destroy());
  });
  await new Promise((r) => proxy.listen(0, '127.0.0.1', r));
  return {
    endpoint: HF_ENDPOINT, proxy: `http://127.0.0.1:${proxy.address().port}`, ca: certs.ca, allowed: HF_ALLOWED,
    token: HF_FAKE_TOKEN, connects, requests,
    async stop() {
      // keep-alive and tunnelled connections would hold close() open
      for (const x of sockets) x.destroy();
      await Promise.all([new Promise((r) => https.close(r)), new Promise((r) => proxy.close(r))]);
    },
  };
}
