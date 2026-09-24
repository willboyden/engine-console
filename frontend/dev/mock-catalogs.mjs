// Mock data for the dev server. NOT authoritative: the real ParamSpec catalogs come from the backend adapters,
// which are sourced from the pinned engine versions. These exist so the schema-driven form can be exercised.
const P = (key, flag, label, help, type, group, o = {}) => ({ key, flag, label, help, type, group, default: null, choices: null, min: null, max: null,
  advanced: false, affects_memory: false, requires_restart: true, bool_style: 'flag', docs_url: null, ...o });

const VD = 'https://docs.vllm.ai/en/stable/configuration/engine_args.html';
export const vllmParams = [
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
export const sglangParams = [
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
};

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
  { repo_id: 'openai/whisper-large-v3', pipeline_tag: 'automatic-speech-recognition', quant: 'bf16', downloads: 5100000, likes: 5200, gated: false, size_bytes: 3 * G, num_params: 1.5e9, layers: 32, kv_heads: 20, head_dim: 64, ctx: 448, license: 'apache-2.0', arch: 'WhisperForConditionalGeneration' },
];
