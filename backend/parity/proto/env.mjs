// Hermetic environment for one node run (Node 22, zero deps): a temp dir with HOME, a fake bin dir (docker,
// nvidia-smi, ip: backend/parity/proto/fakes/*.cjs behind wrappers that pin their state dir), seeded model roots and a
// policy file (monitor or manage). Nothing here touches the real docker, GPUs, network or the user's files; the node
// still reads this machine's /proc and /etc facts, which the normaliser replaces (NORMALISE.md).
import { chmodSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

export const HERE = dirname(fileURLToPath(import.meta.url));
/** The Cargo workspace (backend/). */
export const WORKSPACE = join(HERE, '..', '..');
export const REPO = join(WORKSPACE, '..');
export const FRONTEND = join(REPO, 'frontend');
export const GPU_A = 'GPU-aaaaaaaa-1111-2222-3333-444444444444';
export const GPU_B = 'GPU-bbbbbbbb-1111-2222-3333-444444444444';
export const GATEWAY_IMAGE = 'nginx@sha256:' + '0'.repeat(64);
export const ARGV_SECRET = 'sk-test-not-a-real-key-0000';
export const BIG = 9 * 1024 * 1024 + 4321;          // > WINDOW (8 MiB): fs.read needs a credit grant

/** byte i = (i * 31 + 7) % 251, as drive.py `pattern` */
export function pattern(n) {
  const base = Buffer.alloc(251);
  for (let i = 0; i < 251; i++) base[i] = (i * 31 + 7) % 251;
  const out = Buffer.alloc(n);
  for (let i = 0; i < n; i += 251) base.copy(out, i, 0, Math.min(251, n - i));
  return out;
}

function container(name, { id, image, args, labels = {} }) {
  return {
    Id: id, Name: '/' + name, Image: 'sha256:' + 'ab'.repeat(32), Created: '2026-09-01T10:00:00.123456789Z',
    Path: args[0], Args: args.slice(1),
    // never running: the node then reads no /proc/<pid> of this machine for them
    State: { Status: 'exited', Running: false, Pid: 0, ExitCode: 0, StartedAt: '2026-09-01T10:00:01.5Z' },
    Config: { Image: image, Labels: labels, Env: ['API_TOKEN=NEVER-LEAK-THIS-ENV-VALUE', 'PATH=/usr/bin'] },
    HostConfig: { NetworkMode: 'bridge', DeviceRequests: [] }, NetworkSettings: { Ports: {} }, Mounts: [],
  };
}

function dockerState() {
  return {
    context: 'testctx', contexts: ['default', 'testctx'],
    info: { ServerVersion: '29.0.0', SecurityOptions: ['name=seccomp,profile=builtin', 'name=rootless'],
      Runtimes: { runc: {}, nvidia: {} } },
    containers: [
      container('ec-owned-1', { id: 'e'.repeat(64), image: 'vllm/vllm-openai:v0.11.0',
        args: ['vllm', 'serve', 'org/model-a', '--api-key', ARGV_SECRET, '--port', '8000'],
        labels: { 'engine-console': '1', 'engine-console.role': 'engine', 'engine-console.instance': 'inst_1',
          'engine-console.engine': 'vllm' } }),
      container('other-engine', { id: 'f'.repeat(64), image: 'lmsysorg/sglang:v0.5.0',
        args: ['python3', '-m', 'sglang.launch_server', '--model-path', 'org/model-b', '--api-key=topsecretvalue',
          '--port', '30000'], labels: { 'com.docker.compose.project': 'lab', 'com.docker.compose.service': 'sglang' } }),
    ],
    images: [
      { Id: 'sha256:' + '11'.repeat(32), RepoTags: ['vllm/vllm-openai:v0.11.0'], Size: 123456789,
        Created: '2026-08-01T00:00:00Z', Architecture: 'amd64' },
      { Id: 'sha256:' + '22'.repeat(32), RepoTags: [], RepoDigests: [GATEWAY_IMAGE], Size: 1234,
        Created: '2026-08-02T00:00:00Z', Architecture: 'amd64' },
    ],
    networks: {},
    logs: { 'ec-owned-1': ['INFO starting', `non-default args: {'api_key': ['${ARGV_SECRET}']}`, 'INFO ready'] },
  };
}

function smiState() {
  const gpu = { 'memory.total': '97887', compute_cap: '12.0', driver_version: '595.0', 'utilization.gpu': '7',
    'memory.used': '1024', 'temperature.gpu': '41', 'power.draw': '71.50', 'power.limit': '300.00', 'fan.speed': '30' };
  return { driver: '595.0', cuda: '13.0', apps: [],
    gpus: [{ ...gpu, index: 0, uuid: GPU_A, name: 'NVIDIA Test GPU', 'pci.bus_id': '00000000:01:00.0' },
      { ...gpu, index: 1, uuid: GPU_B, name: 'NVIDIA Test GPU', 'pci.bus_id': '00000000:02:00.0' }] };
}

/** A tiny well-formed model (frontend/e2e/run.mjs writeTinyModel). */
function tinyModel(dir) {
  mkdirSync(dir, { recursive: true });
  writeFileSync(join(dir, 'config.json'), JSON.stringify({
    architectures: ['LlamaForCausalLM'], model_type: 'llama', hidden_size: 64, intermediate_size: 128,
    num_hidden_layers: 2, num_attention_heads: 4, num_key_value_heads: 2, vocab_size: 1000,
    max_position_embeddings: 4096, torch_dtype: 'bfloat16' }));
  const header = Buffer.from(JSON.stringify({ __metadata__: { format: 'pt' },
    'model.embed_tokens.weight': { dtype: 'BF16', shape: [2, 2], data_offsets: [0, 8] },
    'lm_head.weight': { dtype: 'BF16', shape: [2, 2], data_offsets: [8, 16] } }));
  const len = Buffer.alloc(8); len.writeBigUInt64LE(BigInt(header.length));
  writeFileSync(join(dir, 'model.safetensors'), Buffer.concat([len, header, Buffer.alloc(16)]));
}

/** A fresh environment. kind: "monitor" | "manage". Returns {dir, env, policy, cleanup}. */
export function makeEnv(kind) {
  const dir = mkdtempSync(join(tmpdir(), 'ec-proto-'));
  const home = join(dir, 'home'); const bin = join(dir, 'bin'); const state = join(dir, 'state');
  const etc = join(dir, 'etc'); const models = join(dir, 'models');
  for (const d of [home, bin, state, etc, models]) mkdirSync(d, { recursive: true });
  chmodSync(etc, 0o755);
  writeFileSync(join(state, 'docker.json'), JSON.stringify(dockerState()));
  writeFileSync(join(state, 'smi.json'), JSON.stringify(smiState()));
  for (const name of ['docker', 'nvidia-smi', 'ip']) {
    const p = join(bin, name);
    writeFileSync(p, `#!${process.execPath}\nprocess.env.FAKE_STATE_DIR = ${JSON.stringify(state)};\n`
      + `require(${JSON.stringify(join(HERE, 'fakes', name + '.cjs'))});\n`);
    chmodSync(p, 0o755);
  }
  // model roots: an HF cache with e2e/tiny-llama, a plain checkpoint folder, a big blob, a writable root
  const hub = join(models, 'hf', 'hub'); const rev = 'e2e0000000000000000000000000000000000001';
  const repo = join(hub, 'models--e2e--tiny-llama');
  mkdirSync(join(repo, 'refs'), { recursive: true });
  writeFileSync(join(repo, 'refs', 'main'), rev);
  tinyModel(join(repo, 'snapshots', rev));
  tinyModel(join(models, 'plain', 'tiny-plain'));
  mkdirSync(join(models, 'big'), { recursive: true });
  writeFileSync(join(models, 'big', 'blob.bin'), pattern(BIG));
  mkdirSync(join(models, 'w'), { recursive: true });
  const roots = [
    { id: 'e2e-hf', path: hub, kind: 'hf-hub', label: 'e2e Hugging Face cache' },
    { id: 'e2e-plain', path: join(models, 'plain'), kind: 'plain', label: 'e2e checkpoints' },
    { id: 'big', path: join(models, 'big'), kind: 'other' },
  ];
  if (kind === 'manage') roots.push({ id: 'w', path: join(models, 'w'), kind: 'plain', writable: true });
  const policy = join(etc, 'policy.json');
  writeFileSync(policy, JSON.stringify({
    _comment: 'backend/parity/proto: hermetic node policy', version: 1, mode: kind,
    docker: { binary: 'docker', context: 'testctx' }, engine_images: ['vllm/vllm-openai:*'],
    gateway_image: GATEWAY_IMAGE, discovery_ports: [], model_roots: roots,
    allow_delete: kind === 'manage', max_shm_gib: 32,
  }), { mode: 0o644 });
  chmodSync(policy, 0o644);
  // ECNODE_NO_NVML: the Rust node reads the fake nvidia-smi, never the real NVML (PORTING.md §A2); Python ignores it
  // (the reference's node venv has no pynvml, so it reads the fake nvidia-smi anyway). ECNODE_TOOL_PATH: the Rust node finds its tools
  // there, never on PATH (node hardening); Python ignores it and searches PATH, which holds the same fakes first.
  const tools = `${bin}:/usr/bin:/bin`;
  const env = { HOME: home, PATH: tools, LANG: 'C.UTF-8', ECNODE_NO_NVML: '1', ECNODE_TOOL_PATH: tools };
  return { dir, env, policy, cleanup: () => rmSync(dir, { recursive: true, force: true }) };
}
