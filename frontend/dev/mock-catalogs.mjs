// Mock data for the dev server. NOT authoritative: the real ParamSpec catalogs come from the backend adapters,
// which are sourced from the pinned engine versions. These exist so the schema-driven form can be exercised.
// Contract v2 additions per ParamSpec: `level` (basic | advanced | expert), `unit`, and `explain` (merged in by the server).
const P = (key, flag, label, help, type, group, o = {}) => ({ key, flag, label, help, type, group, default: null, choices: null, min: null, max: null,
  advanced: false, affects_memory: false, requires_restart: true, bool_style: 'flag', docs_url: null, level: null, unit: null, explain: null, ...o });

// level + unit per parameter, chosen deliberately: basic = most people should know it, expert = debugging / unusual hardware.
const META = {
  vllm: {
    basic: ['max_model_len', 'gpu_memory_utilization', 'tensor_parallel_size', 'max_num_seqs', 'enable_prefix_caching', 'enable_auto_tool_choice', 'tool_call_parser', 'reasoning_parser'],
    expert: ['pipeline_parallel_size', 'enable_expert_parallel', 'max_cudagraph_capture_size', 'swap_space', 'block_size', 'allowed_origins', 'seed', 'nccl_debug', 'generation_config'],
    units: { max_model_len: 'tokens', max_num_batched_tokens: 'tokens', max_cudagraph_capture_size: 'requests', max_num_seqs: 'requests', cpu_offload_gb: 'GiB', swap_space: 'GiB', block_size: 'tokens' },
  },
  sglang: {
    basic: ['context_length', 'mem_fraction_static', 'tp_size', 'max_running_requests', 'tool_call_parser', 'reasoning_parser'],
    expert: ['dp_size', 'ep_size', 'enable_mixed_chunk', 'attention_backend', 'cuda_graph_max_bs', 'enable_torch_compile', 'speculative_num_steps', 'random_seed'],
    units: { context_length: 'tokens', max_total_tokens: 'tokens', chunked_prefill_size: 'tokens', max_running_requests: 'requests', cuda_graph_max_bs: 'requests', cpu_offload_gb: 'GiB' },
  },
  llamacpp: {
    basic: ['ctx_size', 'n_gpu_layers', 'parallel'], expert: ['batch_size', 'ubatch_size', 'mlock', 'no_mmap'],
    units: { ctx_size: 'tokens', batch_size: 'tokens', ubatch_size: 'tokens' },
  },
  ollama: { basic: ['num_parallel', 'max_loaded_models', 'keep_alive', 'context_length'], expert: ['max_queue'], units: { context_length: 'tokens', num_parallel: 'requests' } },
  trtllm: { basic: ['tp_size', 'max_batch_size', 'max_seq_len', 'kv_cache_free_gpu_memory_fraction'], expert: ['ep_size', 'pp_size'], units: { max_seq_len: 'tokens', max_num_tokens: 'tokens', max_batch_size: 'requests' } },
};
const withMeta = (engine, specs) => specs.map((s) => {
  const m = META[engine];
  return { ...s, level: m.basic.includes(s.key) ? 'basic' : m.expert.includes(s.key) ? 'expert' : 'advanced', unit: m.units[s.key] || s.unit };
});

const VD = 'https://docs.vllm.ai/en/stable/configuration/engine_args.html';
const vllmRaw = [
  P('max_model_len', '--max-model-len', 'Max model length', 'Maximum context (prompt + output) in tokens. Larger values reserve more KV cache.', 'int', 'context', { min: 128, affects_memory: true, docs_url: VD }),
  P('gpu_memory_utilization', '--gpu-memory-utilization', 'GPU memory utilization', 'Fraction of each GPU the engine may claim (weights + KV cache + graphs).', 'float', 'memory', { default: 0.9, min: 0.1, max: 0.99, affects_memory: true, docs_url: VD }),
  P('tensor_parallel_size', '--tensor-parallel-size', 'Tensor parallel size', 'Shard weights across this many GPUs.', 'int', 'parallelism', { default: 1, min: 1, max: 8, affects_memory: true, docs_url: VD }),
  P('pipeline_parallel_size', '--pipeline-parallel-size', 'Pipeline parallel size', 'Split layers across GPUs in stages.', 'int', 'parallelism', { default: 1, min: 1, max: 8, advanced: true, affects_memory: true }),
  P('enable_expert_parallel', '--enable-expert-parallel', 'Expert parallel', 'Distribute MoE experts across GPUs instead of sharding each expert.', 'bool', 'parallelism', { advanced: true, affects_memory: true }),
  P('dtype', '--dtype', 'Data type', 'Compute/weights dtype for unquantized layers.', 'enum', 'quantization', { default: 'auto', choices: ['auto', 'bfloat16', 'float16', 'float32'], affects_memory: true }),
  P('quantization', '--quantization', 'Quantization method', 'Force a quantization backend. Usually auto-detected from the checkpoint. Prefer FP8 on sm_120.', 'enum', 'quantization', { choices: ['fp8', 'awq', 'gptq', 'modelopt', 'compressed-tensors'], affects_memory: true }),
  P('kv_cache_dtype', '--kv-cache-dtype', 'KV cache dtype', 'FP8 KV halves cache memory at a small quality cost.', 'enum', 'quantization', { default: 'auto', choices: ['auto', 'fp8', 'fp8_e4m3', 'fp8_e5m2'], affects_memory: true }),
  P('max_num_seqs', '--max-num-seqs', 'Max concurrent sequences', 'Upper bound on sequences decoded per step.', 'int', 'scheduling', { default: 256, min: 1, affects_memory: true }),
  P('max_num_batched_tokens', '--max-num-batched-tokens', 'Max batched tokens', 'Token budget per scheduler step; larger favours throughput, smaller favours latency.', 'int', 'scheduling', { min: 1, advanced: true, affects_memory: true }),
  P('enable_prefix_caching', '--enable-prefix-caching', 'Prefix caching', 'Reuse KV blocks for shared prompt prefixes.', 'bool', 'performance', { default: true, bool_style: 'flag' }),
  P('enable_chunked_prefill', '--enable-chunked-prefill', 'Chunked prefill', 'Interleave long prefills with decode to keep ITL stable.', 'bool', 'performance', { default: true }),
  P('enforce_eager', '--enforce-eager', 'Enforce eager', 'Disable CUDA graphs. Saves memory and startup time, costs decode speed.', 'bool', 'performance', { affects_memory: true }),
  P('max_cudagraph_capture_size', '--max-cudagraph-capture-size', 'Max CUDA-graph batch', 'Largest batch size to capture graphs for.', 'int', 'performance', { min: 1, advanced: true, affects_memory: true }),
  P('cpu_offload_gb', '--cpu-offload-gb', 'CPU offload (GiB)', 'Offload this much weight memory per GPU to host RAM (slow).', 'float', 'memory', { default: 0, min: 0, advanced: true, affects_memory: true }),
  P('swap_space', '--swap-space', 'Swap space (GiB)', 'Host memory per GPU for swapped KV blocks.', 'float', 'memory', { default: 4, min: 0, advanced: true }),
  P('block_size', '--block-size', 'KV block size', 'Tokens per KV block.', 'enum', 'memory', { choices: ['8', '16', '32', '64', '128'], advanced: true, affects_memory: true }),
  P('enable_auto_tool_choice', '--enable-auto-tool-choice', 'Auto tool choice', 'Let the model decide when to call tools (needs a tool-call parser).', 'bool', 'tools_reasoning'),
  P('tool_call_parser', '--tool-call-parser', 'Tool-call parser', 'Parser matching the model family (e.g. hermes, qwen3_coder, openai).', 'string', 'tools_reasoning'),
  P('reasoning_parser', '--reasoning-parser', 'Reasoning parser', 'Splits <think> output into reasoning_content (e.g. qwen3, deepseek_r1, gpt_oss).', 'string', 'tools_reasoning'),
  P('speculative_config', '--speculative-config', 'Speculative config', 'JSON: method, model, num_speculative_tokens.', 'json', 'speculative', { advanced: true, affects_memory: true }),
  P('limit_mm_per_prompt', '--limit-mm-per-prompt', 'Multimodal limits', 'JSON: max images/videos per prompt, e.g. {"image": 4}.', 'json', 'context', { advanced: true }),
  P('trust_remote_code', '--trust-remote-code', 'Trust remote code', 'Allow executing model repo code. Only for repos you have audited.', 'bool', 'advanced', { advanced: true }),
  P('served_model_name', '--served-model-name', 'Served model name', 'Name exposed on /v1/models.', 'string', 'network', { advanced: true }),
  P('allowed_origins', '--allowed-origins', 'CORS origins', 'Allowed CORS origins.', 'string_list', 'network', { advanced: true }),
  P('chat_template', '--chat-template', 'Chat template path', 'Override the tokenizer chat template.', 'string', 'advanced', { advanced: true }),
  P('seed', '--seed', 'Random seed', 'Seed for sampling reproducibility.', 'int', 'advanced', { advanced: true }),
  P('hf_token', 'env:HF_TOKEN', 'Hugging Face token', 'Passed to the container by env-var name; the server never returns the value.', 'string', 'network', { advanced: true }),
  P('api_key', 'env:VLLM_API_KEY', 'Engine API key', 'Bearer key the engine itself requires (env VLLM_API_KEY, never on the command line).', 'string', 'network', { advanced: true }),
  P('nccl_debug', 'env:NCCL_DEBUG', 'NCCL debug level', 'Sets NCCL_DEBUG in the container.', 'enum', 'advanced', { advanced: true, choices: ['WARN', 'INFO', 'TRACE'] }),
  P('generation_config', '--generation-config', 'Generation config', 'Where default sampling params come from.', 'enum', 'advanced', { default: 'auto', choices: ['auto', 'vllm'], advanced: true, requires_restart: true }),
];

const SD = 'https://docs.sglang.ai/advanced_features/server_arguments.html';
const sglangRaw = [
  P('context_length', '--context-length', 'Context length', 'Maximum context in tokens; reserves KV accordingly.', 'int', 'context', { min: 128, affects_memory: true, docs_url: SD }),
  P('mem_fraction_static', '--mem-fraction-static', 'Static memory fraction', 'Fraction of GPU memory for weights + KV pool.', 'float', 'memory', { default: 0.88, min: 0.1, max: 0.99, affects_memory: true, docs_url: SD }),
  P('tp_size', '--tp-size', 'Tensor parallel size', 'Shard weights across GPUs.', 'int', 'parallelism', { default: 1, min: 1, max: 8, affects_memory: true, docs_url: SD }),
  P('dp_size', '--dp-size', 'Data parallel size', 'Replicate the model across GPU groups.', 'int', 'parallelism', { default: 1, min: 1, advanced: true, affects_memory: true }),
  P('ep_size', '--ep-size', 'Expert parallel size', 'Shard MoE experts.', 'int', 'parallelism', { default: 1, min: 1, advanced: true, affects_memory: true }),
  P('quantization', '--quantization', 'Quantization', 'Force a quantization method. Prefer fp8 on sm_120.', 'enum', 'quantization', { choices: ['fp8', 'awq', 'gptq', 'modelopt_fp4', 'w8a8_fp8'], affects_memory: true }),
  P('kv_cache_dtype', '--kv-cache-dtype', 'KV cache dtype', 'FP8 KV halves cache memory.', 'enum', 'quantization', { default: 'auto', choices: ['auto', 'fp8_e5m2', 'fp8_e4m3'], affects_memory: true }),
  P('max_running_requests', '--max-running-requests', 'Max running requests', 'Concurrent requests in the decode batch.', 'int', 'scheduling', { min: 1, affects_memory: true }),
  P('max_total_tokens', '--max-total-tokens', 'Max total tokens', 'Cap on KV pool tokens.', 'int', 'scheduling', { min: 1, advanced: true, affects_memory: true }),
  P('chunked_prefill_size', '--chunked-prefill-size', 'Chunked prefill size', 'Tokens per prefill chunk.', 'int', 'scheduling', { default: 8192, min: 1, advanced: true }),
  P('schedule_policy', '--schedule-policy', 'Schedule policy', 'Request ordering policy.', 'enum', 'scheduling', { default: 'fcfs', choices: ['fcfs', 'lpm', 'random', 'dfs-weight'], advanced: true }),
  P('enable_mixed_chunk', '--enable-mixed-chunk', 'Mixed chunk', 'Mix prefill and decode in one batch.', 'bool', 'scheduling', { advanced: true }),
  P('attention_backend', '--attention-backend', 'Attention backend', 'Kernel backend; pick per architecture.', 'enum', 'performance', { choices: ['flashinfer', 'triton', 'fa3', 'torch_native'], advanced: true }),
  P('disable_radix_cache', '--disable-radix-cache', 'Disable radix cache', 'Turn off prefix caching.', 'bool', 'performance'),
  P('disable_cuda_graph', '--disable-cuda-graph', 'Disable CUDA graph', 'Saves memory, slows decode.', 'bool', 'performance', { affects_memory: true }),
  P('cuda_graph_max_bs', '--cuda-graph-max-bs', 'CUDA-graph max batch', 'Largest captured batch.', 'int', 'performance', { min: 1, advanced: true, affects_memory: true }),
  P('enable_torch_compile', '--enable-torch-compile', 'torch.compile', 'Compile the model (longer startup).', 'bool', 'performance', { advanced: true }),
  P('cpu_offload_gb', '--cpu-offload-gb', 'CPU offload (GiB)', 'Offload weights to host RAM.', 'float', 'memory', { default: 0, min: 0, advanced: true, affects_memory: true }),
  P('tool_call_parser', '--tool-call-parser', 'Tool-call parser', 'Parser for the model family.', 'string', 'tools_reasoning'),
  P('reasoning_parser', '--reasoning-parser', 'Reasoning parser', 'Split reasoning content.', 'string', 'tools_reasoning'),
  P('speculative_algorithm', '--speculative-algorithm', 'Speculative algorithm', 'EAGLE / EAGLE3 / NGRAM.', 'enum', 'speculative', { choices: ['EAGLE', 'EAGLE3', 'NEXTN', 'NGRAM'], advanced: true, affects_memory: true }),
  P('speculative_num_steps', '--speculative-num-steps', 'Speculative steps', 'Draft depth.', 'int', 'speculative', { min: 1, advanced: true }),
  P('trust_remote_code', '--trust-remote-code', 'Trust remote code', 'Allow repo code execution. Audit first.', 'bool', 'advanced', { advanced: true }),
  P('chat_template', '--chat-template', 'Chat template', 'Override chat template.', 'string', 'advanced', { advanced: true }),
  P('random_seed', '--random-seed', 'Random seed', 'Sampling seed.', 'int', 'advanced', { advanced: true }),
  P('api_key', '--api-key', 'API key', 'SGLang only accepts this as a CLI argument, which the API rejects (it would leak into the process list).', 'string', 'network', { advanced: true }),
];

export const vllmParams = withMeta('vllm', vllmRaw);
export const sglangParams = withMeta('sglang', sglangRaw);

// llama.cpp's llama-server (GGUF models). Canned subset for the dev server; flag names follow the project's CLI.
const LD = 'https://github.com/ggml-org/llama.cpp/tree/master/tools/server';
export const llamacppParams = withMeta('llamacpp', [
  P('ctx_size', '--ctx-size', 'Context size', 'Total context in tokens, shared by all parallel slots.', 'int', 'context', { default: 4096, min: 128, affects_memory: true, docs_url: LD }),
  P('n_gpu_layers', '--n-gpu-layers', 'GPU layers', 'How many layers to place on the GPU; a large number puts the whole model there.', 'int', 'memory', { default: 999, min: 0, affects_memory: true, docs_url: LD }),
  P('parallel', '--parallel', 'Parallel slots', 'Conversations served at once; the context is divided between them.', 'int', 'scheduling', { default: 1, min: 1, affects_memory: true }),
  P('flash_attn', '--flash-attn', 'Flash attention', 'Use the memory-saving attention kernel.', 'bool', 'performance', { default: true }),
  P('cache_type_k', '--cache-type-k', 'KV cache type (keys)', 'Precision of the cached attention keys.', 'enum', 'quantization', { default: 'f16', choices: ['f16', 'q8_0', 'q4_0'], affects_memory: true }),
  P('cache_type_v', '--cache-type-v', 'KV cache type (values)', 'Precision of the cached attention values.', 'enum', 'quantization', { default: 'f16', choices: ['f16', 'q8_0', 'q4_0'], affects_memory: true }),
  P('split_mode', '--split-mode', 'Split mode', 'How a model is divided between several GPUs.', 'enum', 'parallelism', { default: 'layer', choices: ['none', 'layer', 'row'] }),
  P('threads', '--threads', 'CPU threads', 'CPU threads used for the parts that run on the processor.', 'int', 'performance', { min: 1 }),
  P('batch_size', '--batch-size', 'Batch size', 'Largest batch of prompt tokens processed together.', 'int', 'performance', { default: 2048, min: 1 }),
  P('ubatch_size', '--ubatch-size', 'Micro-batch size', 'Physical batch size inside a batch.', 'int', 'performance', { default: 512, min: 1 }),
  P('mlock', '--mlock', 'Lock in RAM', 'Keep the model in RAM so the system cannot swap it out.', 'bool', 'memory'),
  P('no_mmap', '--no-mmap', 'Disable mmap', 'Read the whole file into memory instead of mapping it.', 'bool', 'memory'),
  P('jinja', '--jinja', 'Use the model chat template', 'Apply the Jinja chat template stored in the model (needed for tool calling).', 'bool', 'tools_reasoning', { default: true }),
  P('alias', '--alias', 'Served model name', 'Name exposed on /v1/models.', 'string', 'network'),
]);

// Ollama is configured through environment variables of the server process (one server, many models loaded on demand).
export const ollamaParams = withMeta('ollama', [
  P('num_parallel', 'env:OLLAMA_NUM_PARALLEL', 'Parallel requests per model', 'Requests handled at once for each loaded model.', 'int', 'scheduling', { default: 1, min: 1, affects_memory: true }),
  P('max_loaded_models', 'env:OLLAMA_MAX_LOADED_MODELS', 'Models kept loaded', 'How many different models may stay in memory at once.', 'int', 'memory', { default: 1, min: 1, affects_memory: true }),
  P('keep_alive', 'env:OLLAMA_KEEP_ALIVE', 'Keep alive', 'How long an idle model stays in memory, e.g. 5m, 1h or -1 for always.', 'string', 'memory'),
  P('context_length', 'env:OLLAMA_CONTEXT_LENGTH', 'Default context length', 'Context in tokens for models that do not set their own.', 'int', 'context', { default: 4096, min: 128, affects_memory: true }),
  P('flash_attention', 'env:OLLAMA_FLASH_ATTENTION', 'Flash attention', 'Use the memory-saving attention kernel.', 'bool', 'performance', { bool_style: 'value' }),
  P('kv_cache_type', 'env:OLLAMA_KV_CACHE_TYPE', 'KV cache type', 'Precision of the cached attention state.', 'enum', 'quantization', { default: 'f16', choices: ['f16', 'q8_0', 'q4_0'], affects_memory: true }),
  P('max_queue', 'env:OLLAMA_MAX_QUEUE', 'Maximum queue', 'Requests that may wait before the server answers busy.', 'int', 'scheduling', { default: 512, min: 1 }),
]);

// TensorRT-LLM's trtllm-serve.
export const trtllmParams = withMeta('trtllm', [
  P('tp_size', '--tp_size', 'Tensor parallel size', 'Shard weights across this many GPUs.', 'int', 'parallelism', { default: 1, min: 1, affects_memory: true }),
  P('pp_size', '--pp_size', 'Pipeline parallel size', 'Split layers across GPUs in stages.', 'int', 'parallelism', { default: 1, min: 1, affects_memory: true }),
  P('ep_size', '--ep_size', 'Expert parallel size', 'Shard MoE experts.', 'int', 'parallelism', { min: 1, affects_memory: true }),
  P('max_batch_size', '--max_batch_size', 'Max batch size', 'Largest number of requests in one step.', 'int', 'scheduling', { default: 64, min: 1, affects_memory: true }),
  P('max_num_tokens', '--max_num_tokens', 'Max tokens per step', 'Token budget per scheduler step.', 'int', 'scheduling', { default: 8192, min: 1, affects_memory: true }),
  P('max_seq_len', '--max_seq_len', 'Max sequence length', 'Longest sequence (prompt plus answer) in tokens.', 'int', 'context', { min: 128, affects_memory: true }),
  P('kv_cache_free_gpu_memory_fraction', '--kv_cache_free_gpu_memory_fraction', 'KV cache memory share', 'Share of the GPU memory left after loading that the KV cache may use.', 'float', 'memory', { default: 0.9, min: 0.05, max: 0.99, affects_memory: true }),
  P('backend', '--backend', 'Backend', 'Which runtime executes the model.', 'enum', 'performance', { default: 'pytorch', choices: ['pytorch', 'tensorrt'] }),
  P('reasoning_parser', '--reasoning_parser', 'Reasoning parser', 'Splits thinking output from the answer.', 'string', 'tools_reasoning'),
  P('trust_remote_code', '--trust_remote_code', 'Trust remote code', 'Allow executing model repo code. Audit first.', 'bool', 'advanced', { advanced: true }),
]);

export const catalogs = { vllm: vllmParams, sglang: sglangParams, llamacpp: llamacppParams, ollama: ollamaParams, trtllm: trtllmParams };

export const presets = {
  vllm: [
    { name: 'balanced', params: { gpu_memory_utilization: 0.9, max_model_len: 32768, enable_prefix_caching: true } },
    { name: 'max-throughput', params: { gpu_memory_utilization: 0.94, max_model_len: 16384, max_num_seqs: 512, kv_cache_dtype: 'fp8', enable_chunked_prefill: true } },
    { name: 'low-latency', params: { gpu_memory_utilization: 0.88, max_model_len: 8192, max_num_seqs: 16 } },
    { name: 'long-context', params: { gpu_memory_utilization: 0.92, max_model_len: 131072, kv_cache_dtype: 'fp8', max_num_seqs: 8 } },
    { name: 'tool-agent', params: { max_model_len: 65536, enable_auto_tool_choice: true, tool_call_parser: 'hermes', enable_prefix_caching: true } },
  ],
  sglang: [
    { name: 'balanced', params: { mem_fraction_static: 0.88, context_length: 32768 } },
    { name: 'max-throughput', params: { mem_fraction_static: 0.92, context_length: 16384, max_running_requests: 512, kv_cache_dtype: 'fp8_e4m3' } },
    { name: 'low-latency', params: { mem_fraction_static: 0.85, context_length: 8192, max_running_requests: 16 } },
    { name: 'long-context', params: { mem_fraction_static: 0.9, context_length: 131072, kv_cache_dtype: 'fp8_e4m3' } },
    { name: 'tool-agent', params: { context_length: 65536, tool_call_parser: 'qwen25' } },
  ],
  llamacpp: [
    { name: 'balanced', params: { ctx_size: 16384, parallel: 2, flash_attn: true } },
    { name: 'low-latency', params: { ctx_size: 8192, parallel: 1, flash_attn: true } },
    { name: 'max-throughput', params: { ctx_size: 32768, parallel: 8, flash_attn: true, cache_type_k: 'q8_0', cache_type_v: 'q8_0' } },
    { name: 'long-context', params: { ctx_size: 131072, parallel: 1, flash_attn: true, cache_type_k: 'q8_0', cache_type_v: 'q8_0' } },
    { name: 'tool-agent', params: { ctx_size: 65536, parallel: 2, jinja: true } },
  ],
  ollama: [
    { name: 'balanced', params: { num_parallel: 2, context_length: 8192 } },
    { name: 'low-latency', params: { num_parallel: 1, context_length: 4096, keep_alive: '-1' } },
    { name: 'long-context', params: { num_parallel: 1, context_length: 65536 } },
  ],
  trtllm: [
    { name: 'balanced', params: { max_batch_size: 64, max_seq_len: 32768 } },
    { name: 'low-latency', params: { max_batch_size: 8, max_seq_len: 8192 } },
    { name: 'max-throughput', params: { max_batch_size: 256, max_seq_len: 16384, kv_cache_free_gpu_memory_fraction: 0.92 } },
    { name: 'long-context', params: { max_batch_size: 8, max_seq_len: 131072 } },
  ],
};

// Engine metadata (contract v2 GET /engines): which model formats it loads, per-architecture images, one-server-many-models.
export const engineMeta = [
  { id: 'vllm', display_name: 'vLLM', default_image: 'vllm/vllm-openai:v0.27.1', images_by_arch: { aarch64: 'vllm/vllm-openai:v0.27.1-aarch64' }, model_formats: ['hf', 'plain'], multi_model: false },
  { id: 'sglang', display_name: 'SGLang', default_image: 'lmsysorg/sglang:v0.6.0', images_by_arch: { aarch64: 'lmsysorg/sglang:v0.6.0-aarch64' }, model_formats: ['hf', 'plain'], multi_model: false },
  { id: 'llamacpp', display_name: 'llama.cpp', default_image: 'ghcr.io/ggml-org/llama.cpp:server-cuda', images_by_arch: { aarch64: 'ghcr.io/ggml-org/llama.cpp:server-cuda-arm64' }, model_formats: ['gguf'], multi_model: false },
  { id: 'ollama', display_name: 'Ollama', default_image: 'ollama/ollama:0.12.0', images_by_arch: {}, model_formats: ['ollama'], multi_model: true },
  { id: 'trtllm', display_name: 'TensorRT-LLM', default_image: 'nvcr.io/nvidia/tensorrt-llm/release:1.0.0', images_by_arch: { aarch64: 'nvcr.io/nvidia/tensorrt-llm/release:1.0.0-arm64' }, model_formats: ['hf', 'plain'], multi_model: false },
];

const G = 2 ** 30;
// Canned HF models. layers/kv_heads/head_dim drive the mock fit estimate.
export const models = [
  { repo_id: 'Qwen/Qwen3.6-35B-A3B-FP8', pipeline_tag: 'text-generation', quant: 'fp8', downloads: 412000, likes: 1830, gated: false, size_bytes: 36 * G, is_moe: true, num_params: 35e9, active: 3e9, layers: 40, kv_heads: 4, head_dim: 128, ctx: 262144, license: 'apache-2.0', arch: 'Qwen3MoeForCausalLM' },
  { repo_id: 'Qwen/Qwen3.6-27B-FP8', pipeline_tag: 'text-generation', quant: 'fp8', downloads: 288000, likes: 1204, gated: false, size_bytes: 28.5 * G, num_params: 27e9, layers: 64, kv_heads: 8, head_dim: 128, ctx: 262144, license: 'apache-2.0', arch: 'Qwen3ForCausalLM' },
  { repo_id: 'openai/gpt-oss-120b', pipeline_tag: 'text-generation', quant: 'mxfp4', downloads: 1520000, likes: 4100, gated: false, size_bytes: 65 * G, is_moe: true, num_params: 117e9, active: 5.1e9, layers: 36, kv_heads: 8, head_dim: 64, ctx: 131072, license: 'apache-2.0', arch: 'GptOssForCausalLM' },
  { repo_id: 'google/gemma-4-31b-it', pipeline_tag: 'image-text-to-text', quant: 'bf16', downloads: 610000, likes: 2210, gated: true, size_bytes: 62 * G, num_params: 31e9, layers: 60, kv_heads: 16, head_dim: 128, ctx: 262144, license: 'gemma', arch: 'Gemma4ForConditionalGeneration' },
  { repo_id: 'nvidia/Gemma-4-31B-IT-NVFP4', pipeline_tag: 'image-text-to-text', quant: 'nvfp4', downloads: 92000, likes: 310, gated: false, size_bytes: 19 * G, num_params: 31e9, layers: 60, kv_heads: 16, head_dim: 128, ctx: 262144, license: 'gemma', arch: 'Gemma4ForConditionalGeneration' },
  { repo_id: 'NousResearch/Hermes-4.3-36B', pipeline_tag: 'text-generation', quant: 'bf16', downloads: 77000, likes: 640, gated: false, size_bytes: 72 * G, num_params: 36e9, layers: 64, kv_heads: 8, head_dim: 128, ctx: 131072, license: 'apache-2.0', arch: 'Qwen3ForCausalLM' },
  { repo_id: 'meta-llama/Llama-3.3-70B-Instruct', pipeline_tag: 'text-generation', quant: 'bf16', downloads: 2400000, likes: 2600, gated: true, size_bytes: 141 * G, num_params: 70e9, layers: 80, kv_heads: 8, head_dim: 128, ctx: 131072, license: 'llama3.3', arch: 'LlamaForCausalLM' },
  { repo_id: 'RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic', pipeline_tag: 'text-generation', quant: 'fp8', downloads: 184000, likes: 210, gated: false, size_bytes: 71 * G, num_params: 70e9, layers: 80, kv_heads: 8, head_dim: 128, ctx: 131072, license: 'llama3.3', arch: 'LlamaForCausalLM' },
  { repo_id: 'zai-org/GLM-4.7-Flash', pipeline_tag: 'text-generation', quant: 'bf16', downloads: 133000, likes: 900, gated: false, size_bytes: 60 * G, is_moe: true, num_params: 30e9, active: 3e9, layers: 47, kv_heads: 4, head_dim: 128, ctx: 202752, license: 'mit', arch: 'Glm4MoeForCausalLM' },
  { repo_id: 'mistralai/Mistral-Small-4-119B-Instruct-4bit', pipeline_tag: 'text-generation', quant: 'awq', downloads: 58000, likes: 420, gated: false, size_bytes: 66 * G, is_moe: true, num_params: 119e9, active: 6e9, layers: 56, kv_heads: 8, head_dim: 128, ctx: 131072, license: 'apache-2.0', arch: 'MistralForCausalLM' },
  { repo_id: 'ibm-granite/granite-4.1-30b-instruct', pipeline_tag: 'text-generation', quant: 'bf16', downloads: 41000, likes: 260, gated: false, size_bytes: 60 * G, num_params: 30e9, layers: 40, kv_heads: 8, head_dim: 128, ctx: 131072, license: 'apache-2.0', arch: 'GraniteForCausalLM' },
  { repo_id: 'Qwen/Qwen3-Embedding-8B', pipeline_tag: 'feature-extraction', quant: 'bf16', downloads: 902000, likes: 1450, gated: false, size_bytes: 16 * G, num_params: 8e9, layers: 36, kv_heads: 8, head_dim: 128, ctx: 32768, license: 'apache-2.0', arch: 'Qwen3Model' },
  // GGUF repos: `gguf` lists [quant, GiB] per file group (served as `gguf_groups` by GET /hf/models/{repo}).
  { repo_id: 'unsloth/Qwen3.6-27B-GGUF', pipeline_tag: 'text-generation', quant: 'gguf', downloads: 355000, likes: 980, gated: false, size_bytes: 16.5 * G, num_params: 27e9, layers: 64, kv_heads: 8, head_dim: 128, ctx: 262144, license: 'apache-2.0', arch: 'Qwen3ForCausalLM',
    gguf: [['Q2_K', 10.1], ['Q4_K_M', 16.5], ['Q5_K_M', 19.2], ['Q6_K', 22.2], ['Q8_0', 28.7], ['BF16', 54.1]] },
  { repo_id: 'bartowski/gemma-4-31b-it-GGUF', pipeline_tag: 'text-generation', quant: 'gguf', downloads: 140000, likes: 410, gated: false, size_bytes: 18.9 * G, num_params: 31e9, layers: 60, kv_heads: 16, head_dim: 128, ctx: 262144, license: 'gemma', arch: 'Gemma4ForConditionalGeneration',
    gguf: [['Q3_K_M', 15.1], ['Q4_K_M', 18.9], ['Q6_K', 25.4], ['Q8_0', 32.9]] },
  { repo_id: 'openai/whisper-large-v3', pipeline_tag: 'automatic-speech-recognition', quant: 'bf16', downloads: 5100000, likes: 5200, gated: false, size_bytes: 3 * G, num_params: 1.5e9, layers: 32, kv_heads: 20, head_dim: 64, ctx: 448, license: 'apache-2.0', arch: 'WhisperForConditionalGeneration' },
];
