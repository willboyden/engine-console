// The concept keys this UI asks the learn bundle for (docs/ARCHITECTURE.md section 10.7). Listing them here, once, lets a
// test prove the mock bundle covers them and gives the content writer an exact checklist. Dynamic lookups such as
// `state.${s}` are built from these families.
export const CONCEPT_FAMILIES = {
  state: ['stopped', 'starting', 'loading', 'ready', 'stopping', 'failed', 'auth_required', 'unreachable'],
  link: ['connecting', 'up', 'stale', 'down', 'incompatible', 'disabled'],
  management: ['managed', 'script', 'external'],
  verdict: ['fits', 'tight', 'wont_fit', 'unknown'],
  format: ['hf', 'gguf', 'ollama', 'plain', 'other'],
  root: ['auto', 'hf-home', 'hf-hub', 'plain', 'gguf', 'ollama', 'other'],
  alert: ['host_ram_low', 'swap_heavy', 'shmem_high'],
  page: ['dashboard', 'hosts', 'host', 'library', 'discover', 'downloads', 'launch', 'wizard', 'instances', 'instance', 'metrics', 'bench', 'chat', 'arena', 'usage', 'learn', 'settings'],
  field: ['host', 'gpus', 'root', 'image', 'engine', 'model', 'preset', 'concurrency', 'name', 'ttl', 'pin', 'quant', 'policy_mode', 'machine_filter', 'format_filter', 'target_host', 'beginner_mode', 'profile'],
};

// Preflight check codes are chosen by the hub; the UI looks up `check.<code>` for whatever comes back.
export const CONCEPT_PREFIXES_DYNAMIC = ['check'];

// Metric keys shown with an info icon (section 5 canonical keys plus the host metrics).
export const METRIC_KEYS = [
  'requests_running', 'requests_waiting', 'kv_cache_usage_pct', 'prefix_cache_hit_pct', 'prompt_tokens_total', 'generation_tokens_total',
  'ttft_p50_s', 'ttft_p95_s', 'itl_p50_s', 'e2e_p50_s', 'e2e_p95_s', 'prompt_tps', 'generation_tps', 'preemptions_total', 'spec_decode_accept_pct',
  'vram_used_gib', 'gpu_util_pct', 'gpu_temp_c', 'gpu_power_w', 'host_ram_gib', 'host_ram_anon_gib', 'host_ram_cache_gib', 'host_ram_shmem_gib',
  'ram_used_gib', 'ram_available_gib', 'shmem_gib', 'swap_used_gib', 'disk_free_gib',
];

export const conceptKey = (family, id) => `${family}.${id}`;
export const allConceptKeys = () => Object.entries(CONCEPT_FAMILIES).flatMap(([f, ids]) => ids.map((id) => conceptKey(f, id)));
