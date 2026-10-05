// Fleet fixtures and simulation for the dev mock server: several machines, their GPUs and drives, instances of all three
// management classes, a model library with shared storage, lifecycle scripts and script jobs. Shapes follow backend
// domain/fleet.py and domain/refs.py. Generic names and documentation addresses only; everything here is canned, none of it
// is a fact about real hardware.
import crypto from 'node:crypto';
import { engineMeta, models as hfCatalog } from './mock-catalogs.mjs';

export const G = 2 ** 30;
const nowS = () => Date.now() / 1000;
export const sha16 = (s) => crypto.createHash('sha1').update(s).digest('hex').slice(0, 16);
let seed = 7;
const rnd = () => { seed = (seed * 1664525 + 1013904223) % 4294967296; return seed / 4294967296; };
const r1 = (n) => Math.round(n * 10) / 10;
// NodeError -> HTTP as in ARCHITECTURE section 10.11; the detail is prefixed with the machine's label like the hub does.
const NODE_ERR = (host, code, message) => ({ status: { node_unavailable: 503, busy: 503, monitor_only: 409, conflict: 409, in_use: 409, port_in_use: 409, cancelled: 409, policy_denied: 403, not_owned: 403, not_found: 404, no_space: 507, timeout: 504, unsupported: 501 }[code] || 502, code, detail: `${host}: ${message}` });
const PROBLEM = (status, code, detail) => ({ status, code, detail });

// ───────────────────────────────── machines ─────────────────────────────────
const RTX = 'NVIDIA RTX PRO 6000 Blackwell Workstation Edition';
const gpu = (index, name, total, uuid, o = {}) => ({ index, uuid, name, total_gib: total, compute_capability: '12.0', unified_memory: false, power_limit_w: 600, ...o });
const mount = (mnt, label, o = {}) => ({ mount: mnt, label, fstype: 'ext4', source: null, device: null, model: null, size_bytes: 0, used_bytes: 0, free_bytes: 0, readonly: false, remote: false, remote_host: null, remote_host_id: null, remote_path: null, roots: [], ...o });
const disk = (mnt, label, sizeTiB, usedTiB, o = {}) => {
  const size = Math.round(sizeTiB * 1024 * G), used = Math.round(usedTiB * 1024 * G);
  return mount(mnt, label, { size_bytes: size, used_bytes: used, free_bytes: size - used, ...o });
};
const IMAGES_ALL = ['vllm/vllm-openai:*', 'lmsysorg/sglang:*', 'ghcr.io/ggml-org/llama.cpp:*', 'ghcr.io/ggml-org/llama.cpp@sha256:*', 'ollama/ollama:*', 'nvcr.io/nvidia/tensorrt-llm/release:*'];
const policy = (o = {}) => ({ path: '/etc/ecnode/policy.json', mode: 'manage', error: null, engine_images: IMAGES_ALL, gateway_image: 'engine-console-gateway@sha256:0f3d9c41b2a7', engine_network: 'engine-console-engines', port_range: [18000, 18099], allow_pull: true, allow_delete: true, allow_ipc_host: true, external_control: 'none', max_shm_gib: 64, ...o });
const docker = (o = {}) => ({ available: true, context: 'rootless', rootless: true, version: '28.3.0', nvidia_runtime: true, error: null, ...o });
const img = (ref, sizeGiB, ageDays, arch = 'x86_64') => ({ ref, id: `sha256:${sha16(ref)}${sha16(ref + 'x')}`, size_bytes: Math.round(sizeGiB * G), created: nowS() - ageDays * 86400, arch });
const root = (id, p, kind, o = {}) => ({ id, path: p, kind, writable: false, label: null, shared_from: null, visible_on: [], ...o });

const defs = {
  local: {
    id: 'local', label: 'workstation', remote: false, enabled: true,
    node: { version: '2.0.0', hostname: 'workstation', arch: 'x86_64', os: 'Ubuntu 24.04.3 LTS', kernel: '6.14.0-27-generic', cpu_model: 'AMD Ryzen Threadripper PRO 7975WX', cpu_cores: 32, cpu_threads: 64, python: '3.12.3', driver: '575.57.08', cuda: '12.9', gpu_source: 'nvml', forced_cmd: null, via_ssh: false },
    gpus: [gpu(0, RTX, 95.6, 'GPU-5b3e0c1a-0001-4a7e-9d21-0000000000a1'), gpu(1, RTX, 95.6, 'GPU-5b3e0c1a-0002-4a7e-9d21-0000000000a2')],
    otherProcs: [{ gpu: 1, pid: 3120, name: 'Xorg', gib: 1.4 }],
    mem: { total: 246, otherBase: 78.8, shmem: 98, cached: 100, swapTotal: 32, swapUsed: 24 },
    docker: docker(), policy: policy(),
    disks: [disk('/', '/ (ext4, NVMe SSD 2 TB)', 1.8, 0.9, { device: '/dev/nvme0n1p2', model: 'NVMe SSD 2 TB' }),
      disk('/data', 'tank (zfs)', 7.0, 5.2, { fstype: 'zfs', source: 'tank/data', roots: ['hf-main', 'gguf-models', 'ollama-store', 'finetunes'] }),
      disk('/mnt/usb', '/mnt/usb (exfat, USB SSD 500 GB)', 0.488, 0.478, { fstype: 'exfat', device: '/dev/sdb1', model: 'USB SSD 500 GB', roots: ['usb-hf'] })],
    roots: [root('hf-main', '/data/hf', 'hf-home', { writable: true, label: 'Hugging Face cache', mountAt: '/data' }),
      root('gguf-models', '/data/gguf', 'gguf', { writable: true, label: 'GGUF files', mountAt: '/data' }),
      root('ollama-store', '/data/ollama', 'ollama', { writable: false, label: 'Ollama store', mountAt: '/data' }),
      root('finetunes', '/data/finetunes', 'plain', { writable: true, label: 'Fine-tunes', mountAt: '/data' }),
      root('usb-hf', '/mnt/usb/hf', 'hf-home', { writable: true, label: 'USB cache', mountAt: '/mnt/usb' })],
    images: [img('vllm/vllm-openai:v0.27.1', 21.4, 20), img('lmsysorg/sglang:v0.6.0', 24.8, 11), img('ghcr.io/ggml-org/llama.cpp:server-cuda', 3.1, 6), img('ollama/ollama:0.12.0', 4.7, 30), img('engine-console-gateway@sha256:0f3d9c41b2a7', 0.1, 40)],
    addresses: ['10.0.0.2/24 (enp5s0)'],
    link: { state: 'up' },
  },
  'gpu-box-2': {
    id: 'gpu-box-2', label: 'gpu-box-2', remote: true, enabled: true,
    node: { version: '2.0.0', hostname: 'gpu-box-2', arch: 'x86_64', os: 'Debian GNU/Linux 12 (bookworm)', kernel: '6.1.0-37-amd64', cpu_model: 'Intel Core i9-14900K', cpu_cores: 24, cpu_threads: 32, python: '3.11.2', driver: '575.57.08', cuda: '12.9', gpu_source: 'nvml', forced_cmd: false, via_ssh: true },
    gpus: [gpu(0, 'NVIDIA GeForce RTX 5090', 31.4, 'GPU-9d4417e2-0001-45c0-b1aa-0000000000b1', { power_limit_w: 575 })],
    otherProcs: [],
    mem: { total: 62.5, otherBase: 9.5, shmem: 3.1, cached: 18, swapTotal: 8, swapUsed: 0.3 },
    docker: docker({ rootless: false, context: 'default', version: '27.5.1' }), policy: policy({ external_control: 'stop_start' }),
    disks: [disk('/', '/ (ext4, NVMe SSD 1 TB)', 0.9, 0.35, { device: '/dev/nvme0n1p2', model: 'NVMe SSD 1 TB' }),
      disk('/data', '/data (ext4, NVMe SSD 4 TB)', 3.6, 2.9, { device: '/dev/nvme1n1', model: 'NVMe SSD 4 TB', roots: ['hf-main', 'ollama-store'] }),
      disk('/srv/models', '/srv/models (ext4, SATA SSD 8 TB)', 7.3, 4.1, { device: '/dev/sda1', model: 'SATA SSD 8 TB', roots: ['gguf-shared'] })],
    roots: [root('hf-main', '/data/hf', 'hf-home', { writable: true, label: 'Hugging Face cache', mountAt: '/data' }),
      root('gguf-shared', '/srv/models', 'gguf', { writable: true, label: 'Shared GGUF library', mountAt: '/srv/models', visible_on: ['gpu-box-2', 'edge-1', 'edge-2'] }),
      root('ollama-store', '/data/ollama', 'ollama', { writable: false, label: 'Ollama store', mountAt: '/data' })],
    images: [img('vllm/vllm-openai:v0.27.1', 21.4, 25), img('ghcr.io/ggml-org/llama.cpp:server-cuda', 3.1, 6), img('ollama/ollama:0.12.0', 4.7, 14)],
    addresses: ['10.0.0.5/24 (eth0)'],
    link: { state: 'up' },
  },
  'edge-1': edge('edge-1', '10.0.0.11', { state: 'up' }),
  'edge-2': edge('edge-2', '10.0.0.12', { state: 'up' }, { mode: 'monitor' }),
  'edge-3': edge('edge-3', '10.0.0.13', { state: 'stale' }),
  'edge-4': {
    id: 'edge-4', label: 'edge-4', remote: true, enabled: true, node: null, gpus: [], otherProcs: [], mem: null, docker: null, policy: null, disks: [], roots: [], images: [], addresses: [],
    link: { state: 'down', error: 'Login refused: the console\'s key is not authorized on edge-4. Add the key line printed by "engine-console hosts init-key" to that machine, then press Reconnect.', retryIn: 241 },
  },
  'edge-5': {
    id: 'edge-5', label: 'edge-5', remote: true, enabled: true, node: { version: '1.4.0', hostname: 'edge-5' }, gpus: [], otherProcs: [], mem: null, docker: null, policy: null, disks: [], roots: [], images: [], addresses: [],
    link: { state: 'incompatible', error: 'The node on edge-5 speaks protocol 1 but this console speaks protocol 2. Update the node on edge-5 (or the console) so both match.', retryIn: 288 },
  },
  'spare-1': {
    id: 'spare-1', label: 'spare-1', remote: true, enabled: false, node: null, gpus: [], otherProcs: [], mem: null, docker: null, policy: null, disks: [], roots: [], images: [], addresses: [],
    link: { state: 'disabled', notes: ['Turned off in the hosts file (enabled: false).'] },
  },
};

// A compact aarch64 box with one unified-memory GPU (its VRAM is the machine's RAM).
function edge(id, ip, link, pol = {}) {
  const n = Number(id.split('-')[1]);
  return {
    id, label: id, remote: true, enabled: true,
    node: { version: '2.0.0', hostname: id, arch: 'aarch64', os: 'Ubuntu 24.04.2 LTS', kernel: '6.11.0-1016-nvidia', cpu_model: 'NVIDIA Grace (20-core Arm)', cpu_cores: 20, cpu_threads: 20, python: '3.12.3', driver: '580.95.05', cuda: '13.0', gpu_source: 'nvml', forced_cmd: true, via_ssh: true },
    gpus: [gpu(0, 'NVIDIA GB10', 119.1, `GPU-c7${n}1e4d0-0001-4f0e-8c33-0000000000c${n}`, { compute_capability: '12.1', unified_memory: true, power_limit_w: 140 })],
    otherProcs: [],
    mem: { total: 119.1, otherBase: 9.5 + n, shmem: 2.4, cached: 22, swapTotal: 16, swapUsed: 0.1 },
    docker: docker(),
    policy: policy({ engine_images: ['vllm/vllm-openai:v0.27.1-aarch64', 'lmsysorg/sglang:v0.6.0-aarch64', 'ghcr.io/ggml-org/llama.cpp:*'], port_range: [18000, 18049], ...pol, ...(pol.mode === 'monitor' ? { engine_images: [], allow_pull: false, allow_delete: false } : {}) }),
    disks: [disk('/', '/ (ext4, NVMe SSD 4 TB)', 3.6, 0.6, { device: '/dev/nvme0n1p2', model: 'NVMe SSD 4 TB' }),
      mount('/mnt/shared', 'NFS from gpu-box-2', { fstype: 'nfs4', source: '10.0.0.5:/srv/models', remote: true, remote_host: '10.0.0.5', remote_host_id: 'gpu-box-2', remote_path: '/srv/models', size_bytes: Math.round(7.3 * 1024 * G), used_bytes: Math.round(4.1 * 1024 * G), free_bytes: Math.round(3.2 * 1024 * G), roots: ['nfs-shared'] })],
    roots: [root('hf-main', '/data/hf', 'hf-home', { writable: pol.mode !== 'monitor', label: 'Hugging Face cache', mountAt: '/' }),
      root('nfs-shared', '/mnt/shared', 'gguf', { writable: false, label: 'Shared GGUF library', mountAt: '/mnt/shared', shared_from: 'gpu-box-2', visible_on: [id, 'gpu-box-2', ...['edge-1', 'edge-2'].filter((x) => x !== id)] })],
    images: pol.mode === 'monitor' ? [] : [img('vllm/vllm-openai:v0.27.1-aarch64', 19.8, 18, 'aarch64'), img('ghcr.io/ggml-org/llama.cpp:server-cuda-arm64', 2.9, 6, 'aarch64')],
    addresses: [`${ip}/24 (enP7s7)`],
    link,
  };
}

// runtime link state per machine (the mock changes it on Reconnect)
const links = new Map(Object.values(defs).map((d) => [d.id, { ...d.link, since: d.link.state === 'up' ? nowS() - 3 * 86400 : null, rtt: d.remote ? 3 + Math.round(rnd() * 12) : 1 }]));
const hostEvents = new Set(); // listeners notified when a link changes (the SSE stream)
export const onHostChange = (fn) => { hostEvents.add(fn); return () => hostEvents.delete(fn); };
const emit = (id) => { for (const fn of [...hostEvents]) fn(id); };

export const hostIds = () => Object.keys(defs);
export const hostDef = (id) => defs[id] || null;
const link = (id) => links.get(id);
export const hostUp = (id) => link(id)?.state === 'up';
export const hostMode = (id) => defs[id]?.policy?.mode || null;

// ───────────────────────────────── instances ─────────────────────────────────
const MEM = (total, anon, cache, shmem, kernel, source = 'cgroup') => ({ total_gib: total, anon_gib: anon, cache_gib: cache, shmem_gib: shmem, kernel_gib: kernel, source });
const loc = (host_id, root_id, rel) => { const d = defs[host_id], r = d.roots.find((x) => x.id === root_id); const m = d.disks.find((x) => x.mount === r.mountAt); return { host_id, root_id, root_label: r.label, path: `${r.path}/${rel}`.replace(/\/+/g, '/'), drive: m?.label ?? null, fstype: m?.fstype ?? null, remote: !!m?.remote, shared_from: r.shared_from }; };
const procName = (e) => ({ vllm: 'VLLM::EngineCore', sglang: 'sglang::scheduler', llamacpp: 'llama-server', ollama: 'ollama', trtllm: 'trtllm-serve' }[e] || 'python3');

/** domain/refs.py ModelRef: every field is present, the ones its format does not use are null. */
export const fullRef = (r) => ({ format: 'hf', repo_id: null, root_id: null, rel: null, name: null, ...r });
// `vram` is the engine's steady-state GPU use per card; the live view scales it by state.
export const instances = [];
const mk = (o) => {
  const i = { id: o.id, name: o.name, engine: o.engine, repo_id: o.repo_id ?? '', params: o.params || {}, gpu_ids: o.gpu_ids || [], port: o.port ?? null, container_name: o.container_name ?? null, container_port: o.port ? 8000 : null, internal_endpoint: o.port ? `http://${o.container_name}:8000` : null,
    image: o.image ?? null, state: o.state || 'ready', phase: o.state === 'ready' || !o.state ? 'ready' : null, progress_pct: o.state === 'ready' || !o.state ? 100 : null, pinned: false, ttl_idle_s: null, profile_id: null, error: o.error ?? null, last_logs: o.last_logs ?? null, fit: null,
    created_at: nowS() - (o.ageH ?? 6) * 3600, started_at: (o.state || 'ready') === 'ready' ? nowS() - (o.ageH ?? 6) * 3600 : null, last_request_at: null, uptime_s: null,
    managed: o.management === 'managed', source: o.management === 'external' || o.management === 'script' ? 'external' : 'console', endpoint: o.port ? `http://127.0.0.1:${o.port}` : (o.endpoint ?? null), served_models: o.served_models || (o.repo_id ? [o.repo_id] : []), state_reason: o.state_reason ?? null,
    history_since: o.management === 'managed' ? null : nowS() - 1800, host_memory: o.host_memory ?? null,
    host_id: o.host_id, management: o.management, script: o.script ?? null, model: o.model ? fullRef(o.model) : null, model_location: o.model_location ?? null, vramTemplate: o.vram || [], noMetrics: !!o.noMetrics };
  instances.push(i);
  return i;
};
const D = (host, gi) => defs[host].gpus.find((g) => g.index === gi);

mk({ id: 'inst_qwenagent01', name: 'qwen-agent', engine: 'vllm', repo_id: 'Qwen/Qwen3.6-35B-A3B-FP8', host_id: 'local', management: 'managed', gpu_ids: [0], port: 18000, container_name: 'ec-vllm-qwenagent01', image: 'vllm/vllm-openai:v0.27.1',
  params: { max_model_len: 65536, gpu_memory_utilization: 0.45, enable_prefix_caching: true }, ageH: 7, vram: [{ gpu_index: 0, gib: 38.2 }], host_memory: MEM(34, 20.1, 12, 1.4, 0.5),
  model: { format: 'hf', repo_id: 'Qwen/Qwen3.6-35B-A3B-FP8' }, model_location: loc('local', 'hf-main', 'hub/models--Qwen--Qwen3.6-35B-A3B-FP8') });
mk({ id: 'scr-local-big-model', name: 'big-model-sglang-1', engine: 'sglang', repo_id: 'Qwen/Qwen3.6-27B-FP8', host_id: 'local', management: 'script', script: 'big-model', gpu_ids: [0, 1], port: null, container_name: 'big-model-sglang-1', endpoint: 'http://big-model-sglang-1:30000', image: 'lmsysorg/sglang:v0.6.0',
  served_models: ['Qwen/Qwen3.6-27B-FP8'], params: { tp_size: 2, context_length: 262144, mem_fraction_static: 0.4 }, ageH: 30, vram: [{ gpu_index: 0, gib: 24.6 }, { gpu_index: 1, gib: 24.6 }], host_memory: MEM(22.5, 6.2, 15.9, 0.4, 0.2),
  model: { format: 'hf', repo_id: 'Qwen/Qwen3.6-27B-FP8' }, model_location: loc('local', 'hf-main', 'hub/models--Qwen--Qwen3.6-27B-FP8') });
mk({ id: 'scr-local-small-model', name: 'small-model-vllm-1', engine: 'vllm', repo_id: 'Qwen/Qwen3.6-35B-A3B-FP8', host_id: 'local', management: 'script', script: 'small-model', gpu_ids: [1], state: 'stopped', port: null, container_name: null, endpoint: null, image: 'vllm/vllm-openai:v0.27.1',
  params: {}, vram: [{ gpu_index: 1, gib: 20 }], state_reason: null, model: null });
for (const [name, engine, repo, gpus, gib] of [['coder-model', 'vllm', 'Qwen/Qwen3.6-35B-A3B-FP8', [0], 38], ['vision-model', 'sglang', 'google/gemma-4-31b-it', [1], 62], ['reasoning-model', 'sglang', 'openai/gpt-oss-120b', [0, 1], 34],
  ['embedding-model', 'vllm', 'Qwen/Qwen3-Embedding-8B', [1], 17], ['chat-model', 'vllm', 'Qwen/Qwen3.6-27B-FP8', [0], 30]]) {
  mk({ id: `scr-local-${name}`, name: `${name}-${engine}-1`, engine, repo_id: repo, host_id: 'local', management: 'script', script: name, gpu_ids: gpus, state: 'stopped', container_name: null, endpoint: null, image: engine === 'vllm' ? 'vllm/vllm-openai:v0.27.1' : 'lmsysorg/sglang:v0.6.0',
    vram: gpus.map((g) => ({ gpu_index: g, gib })), state_reason: null, model: null });
}
// a script that only offers custom actions: the console cannot map start/stop to it, so its controls are empty
mk({ id: 'scr-local-switcher', name: 'switcher', engine: 'sglang', repo_id: '', host_id: 'local', management: 'script', script: 'switcher', gpu_ids: [], state: 'stopped', container_name: null, endpoint: null, image: null, vram: [], state_reason: null, model: null });
mk({ id: 'ext-local-vllm', name: 'engine-vllm', engine: 'vllm', repo_id: 'google/gemma-4-31b-it', host_id: 'local', management: 'external', gpu_ids: [1], container_name: 'engine-vllm', image: 'vllm/vllm-openai:v0.23.0', endpoint: 'http://engine-vllm:8000',
  params: { max_model_len: 131072, api_key: '[set]' }, ageH: 2, vram: [{ gpu_index: 1, gib: 31 }], host_memory: MEM(68.6, 3.3, 0.5, 64, 0.8), model: { format: 'hf', repo_id: 'google/gemma-4-31b-it' }, model_location: loc('local', 'hf-main', 'hub/models--google--gemma-4-31b-it') });
mk({ id: 'ext-local-ollama', name: 'ollama', engine: 'ollama', repo_id: '', host_id: 'local', management: 'external', gpu_ids: [1], container_name: 'ollama', image: 'ollama/ollama:0.12.0', endpoint: 'http://ollama:11434', served_models: ['gpt-oss:120b', 'qwen3:8b'],
  ageH: 25, vram: [{ gpu_index: 1, gib: 6.2 }], host_memory: MEM(12, 12, 0, 0, 0, 'rss'), noMetrics: true, model: null });
mk({ id: 'ext-local-gateway', name: 'api-gateway', engine: 'openai_compatible', repo_id: '', host_id: 'local', management: 'external', container_name: 'api-gateway', image: 'example/openai-gateway:1.0', endpoint: 'http://127.0.0.1:4000', served_models: [], state: 'auth_required', state_reason: 'HTTP 401 from /v1/models: an API key is required' });
mk({ id: 'ext-local-dynamo', name: 'dynamo-frontend', engine: 'dynamo', repo_id: '', host_id: 'local', management: 'external', container_name: 'dynamo-frontend', image: 'nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.5.0', endpoint: 'http://dynamo-frontend:8000', served_models: ['Qwen/Qwen3.8-27B'], state: 'unreachable', state_reason: 'connection refused after 3 attempts' });
mk({ id: 'ext-local-llamacpp', name: 'llama-server', engine: 'llamacpp', repo_id: '', host_id: 'local', management: 'external', container_name: 'llama-server', image: 'ghcr.io/ggml-org/llama.cpp:server-cuda', endpoint: 'http://llama-server:8080', served_models: ['gemma-3-4b-q4'], state: 'stopped', state_reason: 'container exited' });

mk({ id: 'inst_llamachat02', name: 'llama-chat', engine: 'llamacpp', repo_id: 'Qwen3.6-27B-Q4_K_M', host_id: 'gpu-box-2', management: 'managed', gpu_ids: [0], port: 18000, container_name: 'ec-llamacpp-llamachat02', image: 'ghcr.io/ggml-org/llama.cpp:server-cuda',
  params: { ctx_size: 32768, parallel: 4, flash_attn: true }, ageH: 3, vram: [{ gpu_index: 0, gib: 19.8 }], host_memory: MEM(4.1, 2.1, 1.9, 0.1, 0.1), model: { format: 'gguf', root_id: 'gguf-shared', rel: 'Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf' },
  model_location: loc('gpu-box-2', 'gguf-shared', 'Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf') });
mk({ id: 'ext-gpu-box-2-ollama', name: 'ollama', engine: 'ollama', repo_id: '', host_id: 'gpu-box-2', management: 'external', container_name: 'ollama', image: 'ollama/ollama:0.12.0', endpoint: 'http://ollama:11434', served_models: ['qwen3:8b'], ageH: 40,
  vram: [{ gpu_index: 0, gib: 6 }], host_memory: MEM(6.4, 6.4, 0, 0, 0, 'rss'), noMetrics: true, model: { format: 'ollama', name: 'qwen3:8b' } });
mk({ id: 'inst_failed04', name: 'qwen-27b-try', engine: 'vllm', repo_id: 'Qwen/Qwen3.6-27B-FP8', host_id: 'gpu-box-2', management: 'managed', gpu_ids: [0], state: 'failed', port: 18001, container_name: 'ec-vllm-failed04', image: 'vllm/vllm-openai:v0.27.1', ageH: 1,
  error: 'CUDA out of memory while loading weights: 28.5 GiB needed, 5.1 GiB free on GPU 0 (another engine holds the rest).', last_logs: 'INFO Loading weights...\nERROR torch.OutOfMemoryError: CUDA out of memory.', params: {}, vram: [{ gpu_index: 0, gib: 29 }],
  model: { format: 'hf', repo_id: 'Qwen/Qwen3.6-27B-FP8' } });

mk({ id: 'inst_edgechat03', name: 'edge-chat', engine: 'llamacpp', repo_id: 'Qwen3.6-27B-Q4_K_M', host_id: 'edge-1', management: 'managed', gpu_ids: [0], port: 18000, container_name: 'ec-llamacpp-edgechat03', image: 'ghcr.io/ggml-org/llama.cpp:server-cuda-arm64',
  params: { ctx_size: 16384, parallel: 2 }, ageH: 5, vram: [{ gpu_index: 0, gib: 21.7 }], host_memory: MEM(22.4, 3.2, 18.8, 0.2, 0.2), model: { format: 'gguf', root_id: 'nfs-shared', rel: 'Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf' },
  model_location: loc('edge-1', 'nfs-shared', 'Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf') });
mk({ id: 'scr-edge-1-edge-serve', name: 'edge-serve-vllm-1', engine: 'vllm', repo_id: 'Qwen/Qwen3.6-27B-FP8', host_id: 'edge-1', management: 'script', script: 'edge-serve', gpu_ids: [0], state: 'stopped', container_name: null, endpoint: null, image: 'vllm/vllm-openai:v0.27.1-aarch64',
  vram: [{ gpu_index: 0, gib: 36 }], state_reason: 'container exited' });
mk({ id: 'ext-edge-2-sglang', name: 'sglang-server', engine: 'sglang', repo_id: 'Qwen/Qwen3.6-27B-FP8', host_id: 'edge-2', management: 'external', gpu_ids: [0], container_name: 'sglang-server', image: 'lmsysorg/sglang:v0.6.0-aarch64', endpoint: 'http://sglang-server:30000', ageH: 12,
  vram: [{ gpu_index: 0, gib: 44 }], host_memory: MEM(41.2, 5.8, 35, 0.3, 0.1), model: { format: 'hf', repo_id: 'Qwen/Qwen3.6-27B-FP8' } });
mk({ id: 'inst_edgebatch05', name: 'edge-batch', engine: 'vllm', repo_id: 'Qwen/Qwen3.6-27B-FP8', host_id: 'edge-3', management: 'managed', gpu_ids: [0], port: 18000, container_name: 'ec-vllm-edgebatch05', image: 'vllm/vllm-openai:v0.27.1-aarch64', ageH: 9,
  params: { max_model_len: 32768 }, vram: [{ gpu_index: 0, gib: 52 }], host_memory: MEM(52.6, 9.1, 43, 0.4, 0.1), model: { format: 'hf', repo_id: 'Qwen/Qwen3.6-27B-FP8' } });

export const instanceById = (id) => instances.find((i) => i.id === id) || null;
export const instancesOn = (hostId) => instances.filter((i) => i.host_id === hostId);

/** What an instance currently holds per GPU: the template scaled by state, so loading engines grow and stopped ones release. */
export function vramOf(i) {
  const f = i.state === 'ready' ? 1 : i.state === 'loading' ? Math.min(0.9, (i.progress_pct || 0) / 100) : i.state === 'unreachable' || i.state === 'auth_required' ? 1 : 0;
  return f > 0 ? i.vramTemplate.map((v) => ({ gpu_index: v.gpu_index, gpu_uuid: D(i.host_id, v.gpu_index).uuid, used_gib: defs[i.host_id].gpus.find((g) => g.index === v.gpu_index)?.unified_memory ? null : r1(v.gib * f + (f === 1 ? Math.sin(nowS() / 30 + v.gpu_index) * 0.05 : 0)) })) : [];
}

// ───────────────────────────────── controls ─────────────────────────────────
const blockReason = (id) => {
  const d = defs[id], st = link(id).state;
  if (st !== 'up') return st === 'stale' ? `${d.label} has not reported for a while, so nothing can be changed until it answers again.` : `${d.label} is ${st === 'disabled' ? 'turned off' : st}: nothing can be changed until the connection is up.`;
  if (d.policy.mode !== 'manage') return `${d.label} is in monitor mode: the console can look but not change anything. An administrator switches it to manage mode in that machine's policy file.`;
  return null;
};
export function controlsFor(i) {
  const off = { start: false, stop: false, restart: false, remove: false, logs: false, configure: false, reason: null };
  const block = blockReason(i.host_id);
  if (i.management === 'managed') return block ? { ...off, logs: link(i.host_id).state === 'up', configure: true, reason: block } : { start: true, stop: true, restart: true, remove: true, logs: true, configure: true, reason: null };
  if (i.management === 'script') {
    if (block) return { ...off, reason: block };
    const acts = (scriptDefs.find((x) => x.host_id === i.host_id && x.name === i.script)?.actions || []).map((a) => a.name);
    const up = acts.includes('up'), down = acts.includes('down');
    return up || down ? { ...off, start: up, stop: down, restart: up && down, reason: `Started and stopped by the script "${i.script}" on ${defs[i.host_id].label}: the buttons run its up and down actions.` }
      : { ...off, reason: `The script "${i.script}" has no up or down action, so the console cannot start or stop it for you. Use the script's own actions.` };
  }
  if (block) return { ...off, configure: true, reason: block };
  if (defs[i.host_id].policy.external_control === 'stop_start') return { ...off, start: true, stop: true, configure: true, reason: null };
  return { ...off, configure: true, reason: `This engine was started outside the console, and ${defs[i.host_id].label}'s policy does not let the console stop or start engines it did not create. Use "Clone as managed" to run your own copy.` };
}

export function instanceView(i) {
  const { vramTemplate, noMetrics, ...rest } = i;
  return { ...rest, host_label: defs[i.host_id].label, gpu_uuids: i.gpu_ids.map((g) => D(i.host_id, g)?.uuid).filter(Boolean), uptime_s: i.state === 'ready' && i.started_at ? nowS() - i.started_at : null, vram: vramOf(i), controls: controlsFor(i) };
}

// ───────────────────────────────── live host views ─────────────────────────────────
const running = (i) => ['ready', 'loading', 'unreachable', 'auth_required'].includes(i.state);
function gpuLive(id) {
  const d = defs[id], t = nowS();
  return d.gpus.map((g) => {
    const procs = [];
    let used = 0;
    for (const i of instancesOn(id)) {
      for (const v of vramOf(i)) if (v.gpu_index === g.index) { procs.push({ pid: 20000 + (parseInt(sha16(i.id + g.index).slice(0, 4), 16) % 9000), name: procName(i.engine), container: i.container_name, instance_id: i.id, used_gib: v.used_gib }); used += v.used_gib ?? 0; }
    }
    for (const o of d.otherProcs.filter((x) => x.gpu === g.index)) { procs.push({ pid: o.pid, name: o.name, container: null, instance_id: null, used_gib: g.unified_memory ? null : o.gib }); used += g.unified_memory ? 0 : o.gib; }
    const busy = instancesOn(id).some((i) => i.state === 'ready' && i.vramTemplate.some((v) => v.gpu_index === g.index));
    const util = busy ? Math.max(3, Math.min(99, 42 + Math.sin(t / 2.5 + g.index) * 30 + rnd() * 18)) : rnd() * 3;
    const stale = link(id).state === 'stale';
    return { index: g.index, uuid: g.uuid, name: g.name, total_gib: g.total_gib, used_gib: g.unified_memory ? null : r1(used + (stale ? 0 : Math.sin(t / 4 + g.index) * 0.3)), free_gib: g.unified_memory ? null : r1(g.total_gib - used),
      util_pct: r1(util), temp_c: r1((busy ? 58 : 34) + Math.sin(t / 9) * 3 + rnd()), power_w: r1(Math.min(g.power_limit_w, g.power_limit_w * (busy ? 0.55 : 0.07) + Math.sin(t / 3) * g.power_limit_w * 0.04 + rnd() * 8)), power_limit_w: g.power_limit_w, fan_pct: busy ? 55 : 30, compute_capability: g.compute_capability, unified_memory: g.unified_memory, procs };
  });
}
export function hostMem(id) {
  const d = defs[id];
  if (!d.mem) return null;
  const w = Math.sin(nowS() / 40) * 0.5 + 0.5;
  const held = instancesOn(id).filter(running).reduce((a, i) => a + (i.host_memory?.total_gib || 0), 0);
  const used = Math.min(d.mem.total * 0.97, held + d.mem.otherBase + w * (d.mem.total > 100 ? 6 : 0.6));
  return { total_gib: d.mem.total, used_gib: r1(used), available_gib: r1(d.mem.total - used), free_gib: r1(Math.max(0.5, d.mem.total * 0.03)), cached_gib: d.mem.cached, shmem_gib: d.mem.shmem, swap_total_gib: d.mem.swapTotal, swap_used_gib: d.mem.swapUsed, updated_at: nowS() };
}
export function alertsFor(m) {
  if (!m) return [];
  const out = [], pct = (m.available_gib / m.total_gib) * 100;
  if (pct < 10) out.push({ level: pct < 5 ? 'crit' : 'warn', code: 'host_ram_low', message: `Only ${Math.round(m.available_gib)} GiB of host RAM is available (${Math.round(pct)}% of ${Math.round(m.total_gib)} GiB).` });
  if (m.swap_total_gib && m.swap_used_gib / m.swap_total_gib > 0.5) out.push({ level: m.swap_used_gib / m.swap_total_gib > 0.7 ? 'crit' : 'warn', code: 'swap_heavy', message: `Swap is ${Math.round((m.swap_used_gib / m.swap_total_gib) * 100)}% used (${Math.round(m.swap_used_gib)} of ${Math.round(m.swap_total_gib)} GiB): engines may stall.` });
  if (m.shmem_gib / m.total_gib > 0.25) out.push({ level: 'warn', code: 'shmem_high', message: `Shared memory is ${Math.round(m.shmem_gib)} GiB and cannot be reclaimed.` });
  return out;
}
function linkStatus(id) {
  const l = link(id), d = defs[id], t = nowS();
  const notes = [...(l.notes || d.link.notes || [])];
  if (d.node?.forced_cmd === false && l.state !== 'disabled') notes.push('Reached with an UNRESTRICTED ssh key: the key on this machine is not limited to the node command, so a stolen hub key could do more than start the node. Restrict it with restrict,from=...,command=... in that machine\'s authorized keys.');
  const base = { state: l.state, error: l.error ?? d.link.error ?? null, retry_in_s: null, notes, last_seen: null, connected_since: null, rtt_ms: null, skew_s: null };
  if (l.state === 'up') return { ...base, last_seen: t - rnd() * 1.5, connected_since: l.since, rtt_ms: l.rtt, skew_s: d.remote ? 0.2 : 0 };
  if (l.state === 'stale') return { ...base, error: 'No reading from edge-3 for 14 seconds. The network may be slow or the machine busy.', last_seen: t - 14, connected_since: l.since || t - 7200, rtt_ms: null };
  if (l.state === 'connecting') return { ...base, error: null };
  if (l.state === 'down' || l.state === 'incompatible') return { ...base, retry_in_s: Math.max(1, Math.round((l.retryAt ?? t + (d.link.retryIn || 60)) - t)) };
  return base;
}
const modelsOn = (id) => lib.filter((m) => m.copies.some((c) => c.host_id === id));
function counts(id) {
  const mine = instancesOn(id), ms = modelsOn(id);
  return { engines_running: mine.filter((i) => i.state === 'ready').length, engines_total: mine.length, models: ms.length, models_bytes: ms.reduce((a, m) => a + m.size_bytes * m.copies.filter((c) => c.host_id === id).length, 0) };
}
export function hostSummary(id) {
  const d = defs[id];
  if (!d) return null;
  const st = link(id).state, live = st === 'up' || st === 'stale';
  const m = live ? hostMem(id) : null;
  return { id, label: d.label, remote: d.remote, link: linkStatus(id), node: d.node, gpus: live ? gpuLive(id) : [], memory: m,
    cpu: live ? { load1: r1(2 + Math.sin(nowS() / 20) * 1.5 + rnd()), load5: 2.4, load15: 2.1, util_pct: r1(6 + Math.abs(Math.sin(nowS() / 11)) * 12) } : null,
    docker: live ? d.docker : null, policy: live ? d.policy : null, counts: counts(id), alerts: alertsFor(m) };
}
function rootInfo(id, r) {
  const d = defs[id], m = d.disks.find((x) => x.mount === r.mountAt);
  const { mountAt, ...rest } = r;
  return { ...rest, mount: m ? { ...m } : null, visible_on: r.visible_on?.length ? r.visible_on : [id] };
}
export function hostDetail(id) {
  const d = defs[id], s = hostSummary(id);
  if (!s) return null;
  const live = s.link.state === 'up' || s.link.state === 'stale';
  return { ...s, disks: live ? d.disks : [], roots: live ? d.roots.map((r) => rootInfo(id, r)) : [], images: live ? d.images : [], scripts: live ? scriptsOn(id) : [], addresses: live ? d.addresses : [] };
}

/** Reconnect: connecting for a moment, then back up (or down again for the host that has no authorized key). */
export function reconnect(id) {
  const l = link(id), d = defs[id];
  if (!d) return null;
  if (l.state === 'disabled') return { error: NODE_ERR(d.label, 'conflict', 'this machine is disabled in the hosts file') };
  l.state = 'connecting'; l.error = null; emit(id);
  setTimeout(() => {
    if (id === 'edge-4') { Object.assign(l, { state: 'down', error: d.link.error, retryAt: nowS() + d.link.retryIn }); }
    else if (id === 'edge-5') { Object.assign(l, { state: 'incompatible', error: d.link.error, retryAt: nowS() + d.link.retryIn }); }
    else { l.state = 'up'; l.since = nowS(); l.error = null; }
    emit(id);
  }, 1800);
  return hostSummary(id);
}
link('edge-4').retryAt = nowS() + 241; link('edge-5').retryAt = nowS() + 288;

// ───────────────────────────────── library ─────────────────────────────────
const ago = (days) => nowS() - days * 86400;
const cp = (host_id, root_id, rel, sizeGiB, o = {}) => ({ host_id, root_id, rel, size_bytes: Math.round(sizeGiB * G), files: o.files ?? null, mtime: ago(o.mtimeDays ?? 12), atime: ago(o.atimeDays ?? 1), visible_on: o.visible_on || [host_id] });
const lm = (key, name, format, size, o, copies) => ({ key, name, format, revision: null, quant: o.quant ?? null, params: o.params ?? null, arch: o.arch ?? null, context: o.context ?? null, shards: o.shards ?? null, size_bytes: Math.round(size * G), warnings: o.warnings || [], copies });
const FORMAT_ENGINES = { hf: ['vllm', 'sglang', 'trtllm'], plain: ['vllm', 'sglang', 'trtllm'], gguf: ['llamacpp'], ollama: ['ollama'], other: [] };

export const lib = [
  lm('hf:Qwen/Qwen3.6-35B-A3B-FP8', 'Qwen/Qwen3.6-35B-A3B-FP8', 'hf', 36, { quant: 'fp8', params: 35e9, arch: 'Qwen3MoeForCausalLM', context: 262144, shards: 8 },
    [cp('local', 'hf-main', 'hub/models--Qwen--Qwen3.6-35B-A3B-FP8', 36, { files: 14, atimeDays: 0 }), cp('gpu-box-2', 'hf-main', 'hub/models--Qwen--Qwen3.6-35B-A3B-FP8', 36, { files: 14, mtimeDays: 3 })]),
  lm('hf:Qwen/Qwen3.6-27B-FP8', 'Qwen/Qwen3.6-27B-FP8', 'hf', 28.5, { quant: 'fp8', params: 27e9, arch: 'Qwen3ForCausalLM', context: 262144, shards: 6 },
    [cp('local', 'hf-main', 'hub/models--Qwen--Qwen3.6-27B-FP8', 28.5, { files: 12 }), cp('edge-1', 'hf-main', 'hub/models--Qwen--Qwen3.6-27B-FP8', 28.5, { files: 12, mtimeDays: 6 }), cp('edge-2', 'hf-main', 'hub/models--Qwen--Qwen3.6-27B-FP8', 28.5, { files: 12, mtimeDays: 6 }),
      cp('edge-3', 'hf-main', 'hub/models--Qwen--Qwen3.6-27B-FP8', 28.5, { files: 12, mtimeDays: 9 })]),
  lm('hf:openai/gpt-oss-120b', 'openai/gpt-oss-120b', 'hf', 65, { quant: 'mxfp4', params: 117e9, arch: 'GptOssForCausalLM', context: 131072, shards: 14 }, [cp('local', 'hf-main', 'hub/models--openai--gpt-oss-120b', 65, { files: 20, atimeDays: 2 })]),
  lm('hf:google/gemma-4-31b-it', 'google/gemma-4-31b-it', 'hf', 62, { quant: null, params: 31e9, arch: 'Gemma4ForConditionalGeneration', context: 262144, shards: 13 }, [cp('local', 'hf-main', 'hub/models--google--gemma-4-31b-it', 62, { files: 22 })]),
  lm('hf:Qwen/Qwen3-Embedding-8B', 'Qwen/Qwen3-Embedding-8B', 'hf', 16, { params: 8e9, arch: 'Qwen3Model', context: 32768, shards: 4 }, [cp('gpu-box-2', 'hf-main', 'hub/models--Qwen--Qwen3-Embedding-8B', 16, { files: 9 })]),
  lm('gguf:unsloth/Qwen3.6-27B-GGUF:Q4_K_M', 'unsloth/Qwen3.6-27B-GGUF', 'gguf', 16.5, { quant: 'Q4_K_M', params: 27e9, arch: 'qwen3', context: 262144 },
    [cp('gpu-box-2', 'gguf-shared', 'Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf', 16.5, { files: 1, visible_on: ['gpu-box-2', 'edge-1', 'edge-2'], atimeDays: 0 }), cp('local', 'gguf-models', 'Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf', 16.5, { files: 1, mtimeDays: 30 })]),
  lm('gguf:unsloth/Qwen3.6-27B-GGUF:Q8_0', 'unsloth/Qwen3.6-27B-GGUF', 'gguf', 28.7, { quant: 'Q8_0', params: 27e9, arch: 'qwen3', context: 262144 }, [cp('local', 'gguf-models', 'Qwen3.6-27B-GGUF/Qwen3.6-27B-Q8_0.gguf', 28.7, { files: 1 })]),
  lm('gguf:bartowski/gemma-4-31b-it-GGUF:Q4_K_M', 'bartowski/gemma-4-31b-it-GGUF', 'gguf', 18.9, { quant: 'Q4_K_M', params: 31e9, arch: 'gemma4', context: 262144 },
    [cp('gpu-box-2', 'gguf-shared', 'gemma-4-31b-it-GGUF/gemma-4-31b-it-Q4_K_M.gguf', 18.9, { files: 1, visible_on: ['gpu-box-2', 'edge-1', 'edge-2'], mtimeDays: 20 })]),
  lm('ollama:qwen3:8b', 'qwen3:8b', 'ollama', 5.2, { quant: 'Q4_K_M', params: 8e9, arch: 'qwen3', context: 40960 },
    [cp('local', 'ollama-store', 'manifests/registry.ollama.ai/library/qwen3/8b', 5.2, { files: 2 }), cp('gpu-box-2', 'ollama-store', 'manifests/registry.ollama.ai/library/qwen3/8b', 5.2, { files: 2 })]),
  lm('ollama:gpt-oss:120b', 'gpt-oss:120b', 'ollama', 65, { quant: 'MXFP4', params: 117e9, arch: 'gptoss', context: 131072 }, [cp('local', 'ollama-store', 'manifests/registry.ollama.ai/library/gpt-oss/120b', 65, { files: 2 })]),
  lm('plain:my-finetune-v3', 'my-finetune-v3', 'plain', 14.3, { params: 7e9, arch: 'LlamaForCausalLM', context: 8192, shards: 3 }, [cp('local', 'finetunes', 'my-finetune-v3', 14.3, { files: 8, mtimeDays: 2 })]),
  lm('plain:legacy-checkpoint', 'legacy-checkpoint', 'plain', 9.7, { warnings: ['No config.json found: parameters and context length are unknown, so the memory fit cannot be estimated.'] }, [cp('local', 'finetunes', 'legacy-checkpoint', 9.7, { files: 5, mtimeDays: 200, atimeDays: 120 })]),
  lm('other:scratch', 'scratch', 'other', 3.1, {}, [cp('local', 'finetunes', 'scratch', 3.1, { files: 40, mtimeDays: 70, atimeDays: 70 })]),
];

const instUsesCopy = (inst, m, c) => {
  const ref = inst.model;
  if (!ref || ref.format !== m.format) return false;
  if (m.format === 'hf') return ref.repo_id === m.name && (inst.host_id === c.host_id || c.visible_on.includes(inst.host_id));
  if (m.format === 'ollama') return ref.name === m.name && inst.host_id === c.host_id;
  if (ref.rel !== c.rel) return false;
  if (inst.host_id === c.host_id) return ref.root_id === c.root_id;
  const mine = defs[inst.host_id].roots.find((r) => r.id === ref.root_id);
  return !!mine && mine.shared_from === c.host_id && c.visible_on.includes(inst.host_id);
};
function copyView(m, c) {
  const d = defs[c.host_id], r = d.roots.find((x) => x.id === c.root_id);
  const mnt = d.disks.find((x) => x.mount === r.mountAt) || null;
  const inUse = instances.filter((i) => instUsesCopy(i, m, c)).map((i) => i.id);
  const writable = r.writable;
  return { id: sha16(`${c.host_id}|${c.root_id}|${c.rel}`), host_id: c.host_id, host_label: d.label, root_id: c.root_id, root_label: r.label, rel: c.rel, path: `${r.path}/${c.rel}`.replace(/\/+/g, '/'), size_bytes: c.size_bytes, files: c.files, mtime: c.mtime, atime: c.atime,
    drive: mnt ? { ...mnt } : null, visible_on: c.visible_on, writable, deletable: writable && d.policy?.allow_delete === true && hostUp(c.host_id) && !inUse.length, in_use_by: inUse };
}
export function libraryView({ host, format, q } = {}) {
  const words = String(q || '').toLowerCase().split(/\s+/).filter(Boolean);
  const out = lib.map((m) => ({ key: m.key, name: m.name, format: m.format, revision: m.revision, quant: m.quant, params: m.params, arch: m.arch, context: m.context, shards: m.shards, size_bytes: m.size_bytes, engines: FORMAT_ENGINES[m.format] || [], warnings: m.warnings, copies: m.copies.map((c) => copyView(m, c)) }))
    .filter((m) => (!format || m.format === format) && (!host || m.copies.some((c) => c.host_id === host || c.visible_on.includes(host))))
    .filter((m) => !words.length || words.every((w) => `${m.name} ${m.key} ${m.quant || ''} ${m.arch || ''} ${m.format}`.toLowerCase().includes(w)));
  return out;
}
export function findCopy(id) {
  for (const m of lib) for (const c of m.copies) if (sha16(`${c.host_id}|${c.root_id}|${c.rel}`) === id) return { model: m, copy: c, view: copyView(m, c) };
  return null;
}
export function deleteCopy(id) {
  const f = findCopy(id);
  if (!f) return { error: PROBLEM(404, 'copy_not_found', 'no such model copy') };
  const d = defs[f.copy.host_id];
  if (!hostUp(d.id)) return { error: NODE_ERR(d.label, 'node_unavailable', `${d.label} is ${link(d.id).state}: the link is not up`) };
  if (!d.policy.allow_delete) return { error: NODE_ERR(d.label, 'policy_denied', 'deleting models is not allowed by this machine\'s policy') };
  if (!f.view.writable) return { error: NODE_ERR(d.label, 'policy_denied', `root ${f.copy.root_id} is read-only`) };
  if (f.view.in_use_by.length) return { error: PROBLEM(409, 'model_in_use', `the model is used by ${f.view.in_use_by.join(', ')}; stop it first`) };
  f.model.copies = f.model.copies.filter((c) => c !== f.copy);
  if (!f.model.copies.length) lib.splice(lib.indexOf(f.model), 1);
  return { ok: true };
}
/** Pick the target root for a download or copy: the one asked for (must be writable), else the first writable of a matching kind. */
export function chooseRoot(hostId, rootId, format) {
  const d = defs[hostId];
  const kinds = { hf: ['hf-home', 'hf-hub', 'auto'], gguf: ['gguf', 'auto'], plain: ['plain', 'auto'], ollama: ['ollama'], other: ['other'] }[format] || ['hf-home'];
  if (rootId) {
    const r = d.roots.find((x) => x.id === rootId);
    if (!r) return { error: NODE_ERR(d.label, 'not_found', `no model root "${rootId}"`) };
    if (!r.writable) return { error: NODE_ERR(d.label, 'policy_denied', `root ${rootId} is read-only`) };
    return { root: r };
  }
  const r = d.roots.find((x) => x.writable && kinds.includes(x.kind));
  return r ? { root: r } : { error: NODE_ERR(d.label, 'conflict', `no writable ${format} model root`) };
}
export const rootFree = (hostId, r) => defs[hostId].disks.find((m) => m.mount === r.mountAt)?.free_bytes ?? Infinity;
export function addCopy(model, hostId, root, rel) {
  const c = cp(hostId, root.id, rel, model.size_bytes / G, { files: 8, mtimeDays: 0, atimeDays: 0, visible_on: root.visible_on?.length ? root.visible_on : [hostId] });
  if (!model.copies.some((x) => x.host_id === hostId && x.root_id === root.id && x.rel === rel)) model.copies.push(c);
  const mnt = defs[hostId].disks.find((x) => x.mount === root.mountAt);
  if (mnt) { mnt.free_bytes = Math.max(0, mnt.free_bytes - model.size_bytes); mnt.used_bytes += model.size_bytes; }
}
/** A finished HF download or copy becomes a library copy (creating the library model when it is new). */
export function onTransferDone(d, src) {
  const host = d.host_id, root = defs[host]?.roots.find((r) => r.id === d.root_id) || defs[host]?.roots.find((r) => r.writable);
  if (!root) return;
  if (src) { addCopy(src.model, host, root, src.copy.rel); return; }
  const meta = hfCatalog.find((x) => x.repo_id === d.repo_id);
  const quant = (d.allow_patterns || []).map((p) => /([QqIiBbFf][A-Za-z0-9_]+)\.gguf$/.exec(p)?.[1] || /\*([A-Za-z0-9_]+)\*/.exec(p)?.[1]).find(Boolean) || null;
  if (quant) {
    const key = `gguf:${d.repo_id}:${quant.toUpperCase()}`;
    let m = lib.find((x) => x.key === key);
    if (!m) { m = lm(key, d.repo_id, 'gguf', d.total_bytes / G, { quant: quant.toUpperCase(), params: meta?.num_params ?? null, arch: meta?.arch ?? null, context: meta?.ctx ?? null }, []); lib.push(m); }
    addCopy(m, host, root, `${d.repo_id.split('/')[1]}/${d.repo_id.split('/')[1]}-${quant.toUpperCase()}.gguf`);
    return;
  }
  const key = `hf:${d.repo_id}`;
  let m = lib.find((x) => x.key === key);
  if (!m) { m = lm(key, d.repo_id, 'hf', d.total_bytes / G, { quant: meta && meta.quant !== 'bf16' ? meta.quant : null, params: meta?.num_params ?? null, arch: meta?.arch ?? null, context: meta?.ctx ?? null, shards: 6 }, []); lib.push(m); }
  addCopy(m, host, root, `hub/models--${d.repo_id.replace('/', '--')}`);
}
/** The library model a ModelRef points to on a host (own copy, or shared storage that host mounts). */
export function resolveModel(hostId, ref) {
  if (!ref) return null;
  for (const m of lib) {
    if (ref.format && m.format !== ref.format) continue;
    if (m.format === 'hf' && ref.repo_id !== m.name) continue;
    if (m.format === 'ollama' && ref.name !== m.name) continue;
    const c = m.copies.find((x) => (x.host_id === hostId || x.visible_on.includes(hostId)) && (m.format === 'hf' || m.format === 'ollama' || x.rel === ref.rel));
    if (c) return { model: m, copy: c };
  }
  return null;
}

// ───────────────────────────────── scripts ─────────────────────────────────
const act = (name, label, o = {}) => ({ name, label, args: [], confirm: false, danger: false, ...o });
const named = (name, label, engine, description) => ({ host_id: 'local', name, label, engine, description, containers: [`${name}-*`], instance: `scr-local-${name}`, actions: [act('up', 'Start', { confirm: true }), act('down', 'Stop', { confirm: true, danger: true }), act('status', 'Check status')] });
const scriptDefs = [
  { host_id: 'local', name: 'big-model', label: 'Big model (both GPUs)', engine: 'sglang', description: 'Serves one large model across both GPUs. The script starts it before the small model, so use these buttons rather than the console\'s own start.', containers: ['big-model-sglang-*'], instance: 'scr-local-big-model',
    actions: [act('up', 'Start', { confirm: true }), act('down', 'Stop', { confirm: true, danger: true }), act('status', 'Check status'), act('switch', 'Switch model', { args: ['model-a', 'model-b'], confirm: true })] },
  { host_id: 'local', name: 'small-model', label: 'Small model (one GPU)', engine: 'vllm', description: 'A smaller model on the second GPU. Starts after the big model.', containers: ['small-model-*'], instance: 'scr-local-small-model',
    actions: [act('up', 'Start', { confirm: true }), act('down', 'Stop', { confirm: true, danger: true }), act('status', 'Check status')] },
  named('coder-model', 'Coding model (one GPU)', 'vllm', 'A coding-tuned model on the first GPU.'),
  named('vision-model', 'Vision model (one GPU)', 'sglang', 'Understands images as well as text.'),
  named('reasoning-model', 'Reasoning model (both GPUs)', 'sglang', 'A large reasoning model that needs both GPUs: stop the big model first.'),
  named('embedding-model', 'Embedding model', 'vllm', 'Turns text into numbers for search.'),
  named('chat-model', 'General chat model (one GPU)', 'vllm', 'Everyday chat on the first GPU.'),
  { host_id: 'local', name: 'switcher', label: 'Model switcher', engine: 'sglang', description: 'Switches which model the big-model script serves. It has no start or stop of its own.', containers: [], instance: 'scr-local-switcher',
    actions: [act('switch', 'Switch model', { args: ['model-a', 'model-b'], confirm: true }), act('status', 'Check status')] },
  { host_id: 'edge-1', name: 'edge-serve', label: 'Edge model server', engine: 'vllm', description: 'Starts the edge model with the settings this site has tested.', containers: ['edge-serve-*'], instance: 'scr-edge-1-edge-serve',
    actions: [act('up', 'Start'), act('down', 'Stop', { confirm: true, danger: true })] },
];
export function scriptsOn(hostId) {
  return scriptDefs.filter((s) => s.host_id === hostId).map((s) => {
    const i = instanceById(s.instance);
    const up = i && (i.state === 'ready' || i.state === 'loading');
    const status = up ? 'up' : s.containers.length ? 'down' : 'unknown';
    return { host_id: s.host_id, name: s.name, label: s.label, engine: s.engine, description: s.description, status, detail: up ? `container ${i.container_name} is running` : s.containers.length ? 'no matching container is running' : 'this script owns no container: ask it for its status', checked_at: nowS() - 20, actions: s.actions, containers: s.containers };
  });
}
export const scriptsAll = () => hostIds().filter((id) => hostUp(id) || link(id).state === 'stale').flatMap(scriptsOn);

export const jobs = new Map();
const jobStreams = new Map();
let jobSeq = 0;
export function runScript(hostId, name, actionName, arg) {
  const d = defs[hostId];
  if (!d) return { error: PROBLEM(404, 'unknown_host', `no such host: ${hostId}`) };
  const s = scriptDefs.find((x) => x.host_id === hostId && x.name === name);
  if (!s) return { error: PROBLEM(404, 'unknown_script', `no script '${name}' is registered in the policy of host ${hostId}`) };
  const a = s.actions.find((x) => x.name === actionName);
  if (!a) return { error: PROBLEM(404, 'unknown_action', `script '${name}' has no action '${actionName}' (it offers: ${s.actions.map((x) => x.name).join(', ')})`) };
  if (a.args.length && arg == null) return { error: PROBLEM(400, 'arg_required', `action '${actionName}' needs one of: ${a.args.join(', ')}`) };
  if (arg != null && !a.args.includes(arg)) return { error: PROBLEM(400, 'invalid_arg', `'${arg}' is not an allowed argument for '${actionName}'${a.args.length ? ` (allowed: ${a.args.join(', ')})` : ' (it takes none)'}`) };
  if (!hostUp(hostId)) return { error: NODE_ERR(d.label, 'node_unavailable', `the link to ${d.label} is not up`) };
  if (d.policy.mode !== 'manage') return { error: NODE_ERR(d.label, 'monitor_only', 'this machine\'s node policy is in monitor mode: scripts cannot run until its operator switches the mode to manage') };
  if ([...jobs.values()].some((j) => j.host_id === hostId && j.name === name && j.state === 'running')) return { error: PROBLEM(409, 'script_busy', `script '${name}' on host ${hostId} is already running; wait for it to finish`) };
  const job = { id: `job_${(++jobSeq).toString(36)}${sha16(String(nowS())).slice(0, 6)}`, host_id: hostId, name, action: actionName, arg: arg ?? null, state: 'running', rc: null, started_at: nowS(), finished_at: null, output_tail: '' };
  jobs.set(job.id, job);
  const lines = [`$ ${s.name} ${actionName}${arg ? ' ' + arg : ''}`, 'checking the other engines on this machine...',
    actionName === 'up' ? 'no engine holds the GPUs this script needs' : actionName === 'down' ? 'stopping container' : 'reading container state',
    actionName === 'status' ? `status: ${scriptsOn(hostId).find((x) => x.name === name)?.status}` : actionName === 'switch' ? `switching to ${arg}` : 'waiting for the health check', 'done'];
  const inst = instanceById(s.instance);
  let k = 0;
  const push = (line) => { job.output_tail = `${job.output_tail}${line}\n`; for (const r of jobStreams.get(job.id) || []) sseLine(r, 'line', line); };
  const tick = setInterval(() => {
    if (k < lines.length) { push(lines[k++]); return; }
    clearInterval(tick);
    job.state = 'completed'; job.rc = 0; job.finished_at = nowS();
    if (inst) {
      if (actionName === 'up') { Object.assign(inst, { state: 'ready', phase: 'ready', progress_pct: 100, started_at: nowS(), state_reason: null, container_name: `${s.name}-1`, endpoint: `http://${s.name}-1:30000` }); }
      else if (actionName === 'down') Object.assign(inst, { state: 'stopped', started_at: null, state_reason: 'stopped by script' });
    }
    for (const r of jobStreams.get(job.id) || []) { sseLine(r, 'end', JSON.stringify(job)); r.end(); }
    jobStreams.delete(job.id);
  }, 650);
  return { job };
}
function sseLine(res, event, data) { if (!res.writableEnded) res.write(`event: ${event}\n${String(data).split('\n').map((l) => `data: ${l}`).join('\n')}\n\n`); }
export function attachJobStream(job, res, once) {
  res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-store', Connection: 'keep-alive' });
  res.write(': ok\n\n');
  for (const l of job.output_tail.split('\n').filter(Boolean)) sseLine(res, 'line', l);
  if (job.state !== 'running' || once) { sseLine(res, 'end', JSON.stringify(job)); res.end(); return; }
  const set = jobStreams.get(job.id) || new Set();
  set.add(res); jobStreams.set(job.id, set);
  res.on('close', () => set.delete(res));
}

// ───────────────────────────────── instance actions ─────────────────────────────────
/** start / stop / restart for any management class, following the controls the hub would offer. */
export function instanceAction(i, action) {
  const d = defs[i.host_id], c = controlsFor(i);
  if (link(i.host_id).state !== 'up') return { error: NODE_ERR(d.label, 'node_unavailable', `the link to ${d.label} is ${link(i.host_id).state}`) };
  if (d.policy.mode !== 'manage') return { error: NODE_ERR(d.label, 'monitor_only', 'this machine is in monitor mode') };
  if (!c[action]) return { error: i.management === 'external' ? { status: 409, code: 'instance_not_managed', detail: `${i.name} was not started by the console and ${d.label}'s policy does not allow ${action}` } : NODE_ERR(d.label, 'policy_denied', `${action} is not available for ${i.name}`) };
  return { ok: true };
}

export function summaryOf() { return hostIds().map(hostSummary); }
export const FORMATS_BY_ENGINE = Object.fromEntries(engineMeta.map((e) => [e.id, e.model_formats]));
export { defs as hostDefs, NODE_ERR, PROBLEM, FORMAT_ENGINES, nowS };
