"""vLLM parameter catalog: data only (no logic), split out of `vllm.py` to keep both readable.

Provenance: names, defaults and choices were read from `vllm serve --help=all` of the pinned image
`vllm/vllm-openai:v0.23.0` (run with a GPU attached — the image exits 1 without one) and cross-checked
against v0.27.1, a newer image. Flags that exist in only one of the two are excluded
so every catalog entry works on both (e.g. --max-num-partial-prefills is 0.23-only and left out).
Enum choices for --quantization, --reasoning-parser and --attention-backend are NOT in --help; they
were read from the image's python registries (QuantizationMethods, reasoning/__init__.py,
AttentionBackendEnum).

Two pseudo-flag conventions (both handled in `VllmAdapter.build_launch`, never emitted as argv):
  * ``env:NAME``  -> the value is set as environment variable NAME in the container.
  * ``@image``    -> overrides `LaunchSpec.image` (pinned tags only; validated).
"""
from __future__ import annotations

from typing import Any

from .base import ParamGroup, ParamSpec, ParamType

VLLM_VERSION = "0.23.0"
DEFAULT_IMAGE = f"vllm/vllm-openai:v{VLLM_VERSION}"
DOCS_BASE = f"https://docs.vllm.ai/en/v{VLLM_VERSION}"
ENV_DOCS = f"{DOCS_BASE}/configuration/env_vars/"

# --- choices read from the pinned image (see module docstring) ---------------------------------
QUANTIZATION = ["awq", "fp8", "fbgemm_fp8", "fp_quant", "modelopt", "modelopt_fp4", "modelopt_mxfp8",
                "modelopt_mixed", "gguf", "auto_gptq", "gptq", "gptq_marlin", "awq_marlin", "humming",
                "compressed-tensors", "bitsandbytes", "experts_int8", "quark", "moe_wna16", "torchao",
                "inc", "mxfp4", "gpt_oss_mxfp4", "deepseek_v4_fp8", "online", "fp8_per_tensor",
                "fp8_per_block", "int8_per_channel_weight_only", "mxfp8"]
KV_CACHE_DTYPE = ["auto", "bfloat16", "float16", "fp8", "fp8_ds_mla", "fp8_e4m3", "fp8_e5m2", "fp8_inc",
                  "fp8_per_token_head", "int8_per_token_head", "nvfp4", "turboquant_3bit_nc",
                  "turboquant_4bit_nc", "turboquant_k3v4_nc", "turboquant_k8v4"]
TOOL_CALL_PARSERS = ["apertus", "cohere_command3", "cohere_command4", "deepseek_v3", "deepseek_v31",
                     "deepseek_v32", "deepseek_v4", "ernie45", "functiongemma", "gemma4", "gigachat3",
                     "glm45", "glm47", "granite", "granite-20b-fc", "granite4", "hermes", "hunyuan_a13b",
                     "hy_v3", "internlm", "jamba", "kimi_k2", "lfm2", "llama3_json", "llama4_json",
                     "llama4_pythonic", "longcat", "mimo", "minicpm5", "minimax", "minimax_m2", "mistral",
                     "olmo3", "openai", "phi4_mini_json", "poolside_v1", "pythonic", "qwen3_coder",
                     "qwen3_xml", "seed_oss", "step3", "step3p5", "xlam"]
REASONING_PARSERS = ["deepseek_r1", "deepseek_v3", "deepseek_v4", "poolside_v1", "cohere_command3",
                     "cohere_command4", "ernie45", "gemma4", "glm45", "openai_gptoss", "granite", "holo2",
                     "hunyuan_a13b", "hy_v3", "kimi_k2", "mimo", "minimax_m2", "minimax_m2_append_think",
                     "mistral", "nemotron_v3", "olmo3", "qwen3", "seed_oss", "step3", "step3p5"]
# CUDA-relevant members of AttentionBackendEnum (ROCm/XPU members omitted on purpose).
ATTENTION_BACKENDS = ["auto", "FLASH_ATTN", "FLASH_ATTN_DIFFKV", "TRITON_ATTN", "FLASHINFER", "TORCH_SDPA",
                      "FLASHINFER_MLA", "FLASHINFER_MLA_SPARSE", "TRITON_MLA", "CUTLASS_MLA", "FLASHMLA",
                      "FLASHMLA_SPARSE", "TOKENSPEED_MLA"]
MOE_BACKENDS = ["auto", "cutlass", "deep_gemm", "deep_gemm_mega_moe", "emulation", "flashinfer_b12x",
                "flashinfer_cutedsl", "flashinfer_cutlass", "flashinfer_trtllm", "humming", "marlin",
                "triton", "triton_unfused"]
LINEAR_BACKENDS = ["auto", "conch", "cutlass", "deep_gemm", "emulation", "exllama", "fbgemm",
                   "flashinfer_cudnn", "flashinfer_cutlass", "flashinfer_trtllm", "machete", "marlin",
                   "torch", "triton"]
SPEC_METHODS = ["custom_class", "deepseek_mtp", "dflash", "draft_model", "eagle", "eagle3", "ernie_mtp",
                "exaone4_5_mtp", "exaone_moe_mtp", "extract_hidden_states", "gemma4_mtp",
                "glm4_moe_lite_mtp", "glm4_moe_mtp", "glm_ocr_mtp", "hy_v3_mtp", "longcat_flash_mtp",
                "medusa", "mimo_mtp", "mimo_v2_mtp", "mlp_speculator", "mtp", "nemotron_h_mtp", "ngram",
                "ngram_gpu", "pangu_ultra_moe_mtp", "qwen3_5_mtp", "qwen3_next_mtp", "step3p5_mtp",
                "suffix"]

# Keys whose CLI flag takes several space-separated values (argparse nargs) — emitted as
# `--flag v1 v2`, not repeated.
NARGS_KEYS = {"served_model_name", "lora_modules", "cudagraph_capture_sizes", "allowed_media_domains",
              "kv_cache_dtype_skip_layers"}

# Boolean flags that are BooleanOptionalAction in --help (`--x, --no-x`): false is emitted as `--no-x`
# so tri-state engine defaults can be forced off. Any other bool emits nothing when false.
NEGATABLE_KEYS = {
    "enable_prefix_caching", "enable_chunked_prefill", "async_scheduling", "enable_expert_parallel",
    "enable_eplb", "enforce_eager", "trust_remote_code", "enable_lora", "enable_sleep_mode",
    "language_model_only", "disable_hybrid_kv_cache_manager", "enable_flashinfer_autotune",
    "disable_sliding_window", "enable_prompt_embeds", "skip_mm_profiling", "enable_log_requests",
    "enable_dbo", "enable_mfu_metrics", "kv_cache_metrics", "cudagraph_metrics", "fully_sharded_loras",
    "enable_auto_tool_choice", "enable_prompt_tokens_details", "enable_server_load_tracking",
    "enable_request_id_headers", "disable_fastapi_docs", "exclude_tools_when_tool_choice_none",
    "numa_bind", "disable_custom_all_reduce",
}


def _p(key: str, flag: str, label: str, help: str, type: ParamType, group: ParamGroup, *,
       default: Any = None, choices: list[str] | None = None, min: float | None = None,
       max: float | None = None, adv: bool = False, mem: bool = False,
       docs: str | None = None) -> ParamSpec:
    if docs is None:
        docs = ENV_DOCS if flag.startswith("env:") else f"{DOCS_BASE}/cli/serve/#{flag.lstrip('-')}"
    return ParamSpec(key=key, flag=flag, label=label, help=help, type=type, group=group, default=default,
                     choices=choices, min=min, max=max, advanced=adv, affects_memory=mem,
                     requires_restart=True, docs_url=docs)


def build_catalog() -> list[ParamSpec]:
    c: list[ParamSpec] = []
    a = c.append
    # ---------------- memory ----------------
    a(_p("gpu_memory_utilization", "--gpu-memory-utilization", "GPU memory utilization",
         "Fraction of each GPU's memory vLLM may claim (weights + activations + KV). Startup fails if less "
         "than this is free, so co-resident engines need a smaller value.",
         "float", "memory", default=0.92, min=0.05, max=1.0, mem=True))
    a(_p("kv_cache_memory_bytes", "--kv-cache-memory-bytes", "Fixed KV cache size (bytes)",
         "Pin the KV cache to an exact size per GPU; when set, gpu-memory-utilization is ignored. Useful "
         "for co-tenant engines because the footprint stops depending on free VRAM.",
         "int", "memory", min=1, adv=True, mem=True))
    a(_p("kv_cache_dtype", "--kv-cache-dtype", "KV cache dtype",
         "Storage type for the KV cache. fp8 roughly doubles token capacity at some accuracy risk; on "
         "sm_120 pair it with the FLASHINFER attention backend.",
         "enum", "memory", default="auto", choices=KV_CACHE_DTYPE, mem=True))
    a(_p("kv_cache_dtype_skip_layers", "--kv-cache-dtype-skip-layers", "KV quantization skip layers",
         "Layer indices or attention types that keep the full-precision KV cache when kv-cache-dtype is "
         "quantized.", "string_list", "memory", adv=True))
    a(_p("cpu_offload_gb", "--cpu-offload-gb", "CPU weight offload (GiB/GPU)",
         "Offload this many GiB of weights per GPU to host RAM, streamed over PCIe every forward pass. "
         "Large slowdown; a last resort to fit a model.", "float", "memory", default=0, min=0, mem=True))
    a(_p("kv_offloading_size", "--kv-offloading-size", "KV offload buffer (GiB)",
         "Size of the CPU KV-cache offloading buffer; summed across TP ranks. Enables offloading when set.",
         "float", "memory", min=0, adv=True, mem=True))
    a(_p("kv_offloading_backend", "--kv-offloading-backend", "KV offload backend",
         "Backend for KV offloading: vLLM-native CPU offload or LMCache.", "enum", "memory",
         default="native", choices=["native", "lmcache"], adv=True))
    a(_p("block_size", "--block-size", "KV block size (tokens)",
         "Tokens per KV cache block. Leave unset: hybrid/Mamba models pick a large value automatically "
         "(some Qwen3.x builds log 1600).", "int", "memory", min=1, adv=True))
    a(_p("num_gpu_blocks_override", "--num-gpu-blocks-override", "GPU blocks override",
         "Force the number of KV blocks instead of the profiled value. For tests only.", "int", "memory",
         min=1, adv=True, mem=True))
    a(_p("mamba_cache_dtype", "--mamba-cache-dtype", "Mamba cache dtype",
         "dtype of the conv+SSM state for Mamba/hybrid layers.", "enum", "memory", default="auto",
         choices=["auto", "bfloat16", "float16", "float32"], adv=True, mem=True))
    a(_p("mamba_ssm_cache_dtype", "--mamba-ssm-cache-dtype", "Mamba SSM state dtype",
         "dtype of the SSM state only (conv state follows mamba-cache-dtype).", "enum", "memory",
         default="auto", choices=["auto", "bfloat16", "float16", "float32"], adv=True, mem=True))
    a(_p("enable_sleep_mode", "--enable-sleep-mode", "Sleep mode",
         "Allow releasing GPU memory while idle. NOTE: the /sleep endpoints additionally need "
         "VLLM_SERVER_DEV_MODE, which this console deliberately does not expose (it also opens an "
         "unauthenticated /collective_rpc).", "bool", "memory", default=False, adv=True))
    # ---------------- parallelism ----------------
    a(_p("tensor_parallel_size", "--tensor-parallel-size", "Tensor parallel size",
         "Shard each layer across this many GPUs. Without NVLink, all-reduce runs "
         "over PCIe; prefer one model per card unless the model cannot fit.",
         "int", "parallelism", default=1, min=1, max=64, mem=True))
    a(_p("pipeline_parallel_size", "--pipeline-parallel-size", "Pipeline parallel size",
         "Split the layer stack into this many sequential stages across GPUs.", "int", "parallelism",
         default=1, min=1, max=64, mem=True))
    a(_p("data_parallel_size", "--data-parallel-size", "Data parallel size",
         "Run this many full replicas (MoE layers shard over tp*dp). Each replica needs its own GPUs.",
         "int", "parallelism", default=1, min=1, max=64, mem=True))
    a(_p("enable_expert_parallel", "--enable-expert-parallel", "Expert parallel",
         "Shard MoE experts across GPUs instead of tensor-parallel splitting them. MoE models only.",
         "bool", "parallelism", default=False))
    a(_p("decode_context_parallel_size", "--decode-context-parallel-size", "Decode context parallel",
         "Split long-context decode KV across GPUs; reuses the TP group rather than adding GPUs.",
         "int", "parallelism", default=1, min=1, adv=True))
    a(_p("enable_eplb", "--enable-eplb", "Expert load balancing",
         "Rebalance MoE experts across ranks at runtime. Only meaningful with expert parallelism.",
         "bool", "parallelism", default=False, adv=True))
    a(_p("expert_placement_strategy", "--expert-placement-strategy", "Expert placement",
         "How experts are laid out across EP ranks.", "enum", "parallelism", default="linear",
         choices=["linear", "round_robin"], adv=True))
    a(_p("distributed_executor_backend", "--distributed-executor-backend", "Distributed executor",
         "Worker launch backend for multi-GPU. Default picks mp on one node.", "enum", "parallelism",
         choices=["mp", "ray", "uni", "external_launcher"], adv=True))
    a(_p("disable_custom_all_reduce", "--disable-custom-all-reduce", "Disable custom all-reduce",
         "Fall back to NCCL all-reduce. Try this if TP>1 hangs or corrupts output over PCIe.", "bool",
         "parallelism", default=False, adv=True))
    a(_p("all2all_backend", "--all2all-backend", "All2All backend",
         "MoE expert-parallel communication backend (e.g. allgather_reducescatter, deepep_*).",
         "string", "parallelism", default="allgather_reducescatter", adv=True))
    a(_p("enable_dbo", "--enable-dbo", "Dual batch overlap",
         "Overlap compute and communication across two micro-batches (multi-GPU MoE).", "bool",
         "parallelism", default=False, adv=True))
    a(_p("numa_bind", "--numa-bind", "NUMA bind workers",
         "Pin GPU worker processes to their GPU's local NUMA node CPUs.", "bool", "parallelism",
         default=False, adv=True))
    # ---------------- context ----------------
    a(_p("max_model_len", "--max-model-len", "Max context length",
         "Prompt + output tokens per request. Unset derives it from the model config. Larger values need "
         "more KV memory for even one request; vLLM refuses to start if a single full-length request "
         "cannot fit.", "int", "context", min=128, mem=True))
    a(_p("hf_overrides", "--hf-overrides", "HF config overrides (JSON)",
         "JSON merged into the model's HF config, e.g. a YaRN rope_parameters block to extend context. "
         "It REPLACES the nested dict wholesale, so restate every original key; for VL models target "
         "text_config.", "json", "context", adv=True))
    a(_p("allow_long_max_model_len", "env:VLLM_ALLOW_LONG_MAX_MODEL_LEN", "Allow max-len above model max",
         "Sets VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 so a max-model-len beyond the config's "
         "max_position_embeddings is accepted. Quality beyond the trained length is not guaranteed.",
         "bool", "context", default=False, adv=True))
    a(_p("limit_mm_per_prompt", "--limit-mm-per-prompt", "Multimodal items per prompt (JSON)",
         'Max items per modality per request, e.g. {"image": 8, "video": 1}. Lower values reduce peak '
         "memory during profiling.", "json", "context", adv=True, mem=True))
    a(_p("language_model_only", "--language-model-only", "Language model only",
         "Disable all multimodal inputs (limits set to 0), skipping the vision/audio encoders.", "bool",
         "context", default=False, mem=True))
    a(_p("media_io_kwargs", "--media-io-kwargs", "Media I/O kwargs (JSON)",
         'Per-modality media decode options, e.g. {"video": {"fps": 2, "num_frames": 256}}.', "json",
         "context", adv=True))
    a(_p("mm_processor_kwargs", "--mm-processor-kwargs", "MM processor kwargs (JSON)",
         "Arguments forwarded to the model's multimodal processor (e.g. image size limits).", "json",
         "context", adv=True))
    a(_p("mm_processor_cache_gb", "--mm-processor-cache-gb", "MM processor cache (GiB)",
         "Host-memory cache for processed multimodal inputs; 0 disables.", "float", "context",
         default=4, min=0, adv=True))
    a(_p("video_pruning_rate", "--video-pruning-rate", "Video pruning rate",
         "Efficient Video Sampling pruning fraction in [0,1) for models that support it.", "float",
         "context", min=0, max=0.99, adv=True))
    a(_p("skip_mm_profiling", "--skip-mm-profiling", "Skip MM profiling",
         "Skip multimodal memory profiling at startup (risks OOM on the first large image/video).",
         "bool", "context", default=False, adv=True))
    a(_p("allowed_local_media_path", "--allowed-local-media-path", "Allowed local media path",
         "Directory API requests may read media from via file:// URLs. Never '/': it lets any prompt "
         "read any container file.", "string", "context", adv=True))
    a(_p("allowed_media_domains", "--allowed-media-domains", "Allowed media domains",
         "If set, only media URLs on these domains are fetched (egress control).", "string_list",
         "context", adv=True))
    a(_p("disable_sliding_window", "--disable-sliding-window", "Disable sliding window",
         "Cap context at the sliding-window size instead of using windowed attention.", "bool",
         "context", default=False, adv=True))
    a(_p("generation_config", "--generation-config", "Generation config source",
         "'auto' loads the model's generation_config.json defaults (temperature etc.); 'vllm' uses vLLM "
         "defaults; or a folder path. max_new_tokens there becomes a server-wide cap.", "string",
         "context", default="auto", adv=True))
    a(_p("override_generation_config", "--override-generation-config", "Override generation config (JSON)",
         'Server-side default sampling params, e.g. {"temperature": 0.6}.', "json", "context", adv=True))
    a(_p("max_logprobs", "--max-logprobs", "Max logprobs", "Maximum logprobs a request may ask for.",
         "int", "context", default=20, min=0, adv=True))
    a(_p("enable_prompt_embeds", "--enable-prompt-embeds", "Accept prompt embeds",
         "Allow passing text embeddings instead of token ids. Can crash the engine on bad input.",
         "bool", "context", default=False, adv=True))
    # ---------------- quantization / model loading ----------------
    a(_p("quantization", "--quantization", "Quantization",
         "Weight quantization method. Leave unset for pre-quantized checkpoints (NVFP4, compressed-tensors, "
         "AWQ): passing e.g. fp8 for an NVFP4 checkpoint misloads it. Use fp8 to quantize a BF16 model "
         "online; FP8 is the usual choice on recent GPUs.", "enum", "quantization", choices=QUANTIZATION, mem=True))
    a(_p("dtype", "--dtype", "Activation dtype",
         "Compute dtype. auto uses BF16 for FP32/BF16 models and FP16 for FP16 models.", "enum",
         "quantization", default="auto",
         choices=["auto", "bfloat16", "float", "float16", "float32", "half"], mem=True))
    a(_p("moe_backend", "--moe-backend", "MoE kernel backend",
         "Kernel for MoE expert GEMMs. On sm_120 NVFP4 MoE auto-selects a slow Marlin path; cutlass, "
         "marlin and flashinfer_b12x are the ones worth A/B-timing.", "enum", "quantization",
         default="auto", choices=MOE_BACKENDS))
    a(_p("linear_backend", "--linear-backend", "Quantized linear backend",
         "Kernel for quantized dense GEMMs.", "enum", "quantization", default="auto",
         choices=LINEAR_BACKENDS, adv=True))
    a(_p("nvfp4_gemm_backend", "env:VLLM_NVFP4_GEMM_BACKEND", "NVFP4 GEMM backend (env)",
         "Sets VLLM_NVFP4_GEMM_BACKEND for NVFP4 dense GEMM selection. Prefer linear-backend.",
         "string", "quantization", adv=True))
    a(_p("revision", "--revision", "Model revision",
         "Branch, tag or commit of the model. Pin this: an unpinned main can change size or tokenizer "
         "between restarts.", "string", "quantization"))
    a(_p("tokenizer", "--tokenizer", "Tokenizer override",
         "Tokenizer repo or path if it differs from the model.", "string", "quantization", adv=True))
    a(_p("tokenizer_mode", "--tokenizer-mode", "Tokenizer mode", "Tokenizer implementation to use.",
         "enum", "quantization", default="auto",
         choices=["auto", "deepseek_v32", "deepseek_v4", "hf", "mistral", "slow"], adv=True))
    a(_p("trust_remote_code", "--trust-remote-code", "Trust remote code",
         "Execute model-repo Python code. Required by some architectures; it is arbitrary code from the "
         "Hub, so only for repos you have reviewed.", "bool", "quantization", default=False))
    a(_p("model_impl", "--model-impl", "Model implementation",
         "vllm native, or the transformers backend fallback.", "enum", "quantization", default="auto",
         choices=["auto", "terratorch", "transformers", "vllm"], adv=True))
    a(_p("load_format", "--load-format", "Load format",
         "Weight loading format (auto, safetensors, pt, runai_streamer, dummy, ...).", "string",
         "quantization", default="auto", adv=True))
    a(_p("safetensors_load_strategy", "--safetensors-load-strategy", "Safetensors load strategy",
         "None = mmap lazily; 'prefetch' reads files into the page cache first (helps network FS).",
         "string", "quantization", adv=True))
    # ---------------- scheduling ----------------
    a(_p("max_num_seqs", "--max-num-seqs", "Max concurrent sequences",
         "Upper bound on sequences per scheduler step. Raise for throughput; lower for latency or when a "
         "speculator (DFlash) requires it.", "int", "scheduling", min=1, mem=True))
    a(_p("max_num_batched_tokens", "--max-num-batched-tokens", "Max batched tokens",
         "Token budget per scheduler step (prefill chunk size). Larger favours throughput/TTFT for long "
         "prompts; smaller favours inter-token latency.", "int", "scheduling", min=1, mem=True))
    a(_p("enable_prefix_caching", "--enable-prefix-caching", "Prefix caching",
         "Reuse KV blocks for shared prompt prefixes (system prompts, agent loops).", "bool",
         "scheduling"))
    a(_p("enable_chunked_prefill", "--enable-chunked-prefill", "Chunked prefill",
         "Split long prefills into chunks so decodes are not starved.", "bool", "scheduling"))
    a(_p("async_scheduling", "--async-scheduling", "Async scheduling",
         "Overlap CPU scheduling with GPU execution to remove idle gaps.", "bool", "scheduling"))
    a(_p("scheduling_policy", "--scheduling-policy", "Scheduling policy",
         "fcfs, or priority (requests may carry a priority).", "enum", "scheduling", default="fcfs",
         choices=["fcfs", "priority"]))
    a(_p("stream_interval", "--stream-interval", "Stream interval (tokens)",
         "Tokens buffered per streamed chunk; 1 is smoothest, larger reduces CPU at high concurrency.",
         "int", "scheduling", default=1, min=1, adv=True))
    a(_p("long_prefill_token_threshold", "--long-prefill-token-threshold", "Long prefill threshold",
         "For chunked prefill, prompts longer than this count as 'long'.", "int", "scheduling",
         default=0, min=0, adv=True))
    a(_p("prefix_caching_hash_algo", "--prefix-caching-hash-algo", "Prefix hash algorithm",
         "Hash for prefix-cache blocks; xxhash is faster, sha256 collision-safe across tenants.",
         "enum", "scheduling", default="sha256",
         choices=["sha256", "sha256_cbor", "xxhash", "xxhash_cbor"], adv=True))
    a(_p("disable_hybrid_kv_cache_manager", "--disable-hybrid-kv-cache-manager",
         "Disable hybrid KV manager",
         "Allocate equal KV to every attention layer even for sliding-window/hybrid models (wastes "
         "memory).", "bool", "scheduling", adv=True))
    a(_p("seed", "--seed", "Random seed", "Global seed for reproducibility.", "int", "scheduling",
         default=0, adv=True))
    a(_p("api_server_count", "--api-server-count", "API server processes",
         "Frontend processes; defaults to data-parallel-size.", "int", "scheduling", min=1, adv=True))
    # ---------------- tools & reasoning ----------------
    a(_p("enable_auto_tool_choice", "--enable-auto-tool-choice", "Auto tool choice",
         "Let the model decide when to call tools. Requires tool-call-parser.", "bool",
         "tools_reasoning"))
    a(_p("tool_call_parser", "--tool-call-parser", "Tool-call parser",
         "Parser matching the model's tool-call dialect (hermes, qwen3_xml, qwen3_coder, llama3_json, "
         "openai for gpt-oss, poolside_v1, ...). A mismatch leaks raw tool markup into content.", "enum",
         "tools_reasoning", choices=TOOL_CALL_PARSERS))
    a(_p("reasoning_parser", "--reasoning-parser", "Reasoning parser",
         "Splits chain-of-thought into reasoning_content (qwen3, deepseek_r1, nemotron_v3, openai_gptoss, "
         "poolside_v1, ...).", "enum", "tools_reasoning", choices=REASONING_PARSERS))
    a(_p("chat_template", "--chat-template", "Chat template",
         "Path or inline Jinja chat template overriding the model's.", "string", "tools_reasoning"))
    a(_p("chat_template_content_format", "--chat-template-content-format", "Chat content format",
         "How message content is passed to the template: auto, openai (parts list) or string.", "enum",
         "tools_reasoning", default="auto", choices=["auto", "openai", "string"], adv=True))
    a(_p("default_chat_template_kwargs", "--default-chat-template-kwargs", "Default template kwargs (JSON)",
         'Server default kwargs for the chat template, e.g. {"enable_thinking": false}.', "json",
         "tools_reasoning"))
    a(_p("exclude_tools_when_tool_choice_none", "--exclude-tools-when-tool-choice-none",
         "Hide tools when tool_choice=none", "Drop tool definitions from the prompt when tool_choice is "
         "'none'.", "bool", "tools_reasoning", adv=True))
    a(_p("tool_parser_plugin", "--tool-parser-plugin", "Tool parser plugin",
         "Path to a Python file registering a custom tool parser.", "string", "tools_reasoning", adv=True))
    a(_p("reasoning_parser_plugin", "--reasoning-parser-plugin", "Reasoning parser plugin",
         "Path to a Python file registering a custom reasoning parser.", "string", "tools_reasoning",
         adv=True))
    a(_p("structured_outputs_config", "--structured-outputs-config", "Structured outputs config (JSON)",
         'Guided-decoding config, e.g. {"backend": "xgrammar"}.', "json", "tools_reasoning", adv=True))
    a(_p("response_role", "--response-role", "Response role", "Role name of generated messages.",
         "string", "tools_reasoning", default="assistant", adv=True))
    # ---------------- performance ----------------
    a(_p("enforce_eager", "--enforce-eager", "Enforce eager mode",
         "Disable CUDA graphs. Saves ~1 GiB and startup time, costs decode speed. Debugging only.",
         "bool", "performance", default=False, mem=True))
    a(_p("performance_mode", "--performance-mode", "Performance mode",
         "balanced, interactivity (fine-grained CUDA graphs, latency kernels) or throughput (larger "
         "graphs, aggressive batching).", "enum", "performance", default="balanced",
         choices=["balanced", "interactivity", "throughput"]))
    a(_p("optimization_level", "--optimization-level", "Optimization level",
         "0 = fastest startup, 3 = best steady-state performance (compile effort).", "int",
         "performance", default=2, min=0, max=3))
    a(_p("compilation_config", "--compilation-config", "Compilation config (JSON)",
         'torch.compile / CUDA-graph config, e.g. {"mode": 3, "cudagraph_capture_sizes": [1,2,4,8]}.',
         "json", "performance", adv=True))
    a(_p("cudagraph_capture_sizes", "--cudagraph-capture-sizes", "CUDA graph capture sizes",
         "Batch sizes to capture as CUDA graphs. Fewer sizes = faster startup and less graph memory.",
         "string_list", "performance", adv=True, mem=True))
    a(_p("max_cudagraph_capture_size", "--max-cudagraph-capture-size", "Max CUDA graph size",
         "Largest batch size captured as a CUDA graph.", "int", "performance", min=1, adv=True,
         mem=True))
    a(_p("attention_backend", "--attention-backend", "Attention backend",
         "Attention kernel. FLASHINFER is required for the FP4/FP8-KV paths on sm_120; FLASH_ATTN is FA2 "
         "there. NOTE: the VLLM_ATTENTION_BACKEND env var no longer exists in v0.23.0 (this flag "
         "replaced it).", "enum", "performance", choices=ATTENTION_BACKENDS))
    a(_p("enable_flashinfer_autotune", "--enable-flashinfer-autotune", "FlashInfer autotune",
         "Run FlashInfer autotuning during warmup (results are cached across restarts).", "bool",
         "performance", adv=True))
    a(_p("flashinfer_moe_backend", "env:VLLM_FLASHINFER_MOE_BACKEND", "FlashInfer MoE mode (env)",
         "Sets VLLM_FLASHINFER_MOE_BACKEND (throughput, latency, masked_gemm) for FlashInfer MoE.",
         "enum", "performance", choices=["throughput", "latency", "masked_gemm"], adv=True))
    a(_p("mamba_backend", "--mamba-backend", "Mamba SSU backend", "Backend for Mamba state updates.",
         "string", "performance", adv=True))
    a(_p("gdn_prefill_backend", "--gdn-prefill-backend", "GDN prefill backend",
         "Prefill backend for gated-delta-net (Qwen3.5/3.8 linear attention) layers.", "enum",
         "performance", choices=["flashinfer", "triton", "cutedsl"], adv=True))
    # ---------------- speculative ----------------
    a(_p("speculative_config", "--speculative-config", "Speculative decoding config (JSON)",
         'Full spec-decode config, e.g. {"method":"mtp","num_speculative_tokens":2} or a DFlash/EAGLE '
         "draft model with revision pinned. Mutually exclusive with the spec_* shortcuts.", "json",
         "speculative", mem=True))
    a(_p("spec_method", "--spec-method", "Speculative method",
         "Shortcut: speculation method (mtp, eagle3, ngram, dflash, draft_model, ...).", "enum",
         "speculative", choices=SPEC_METHODS, adv=True))
    a(_p("spec_model", "--spec-model", "Draft model",
         "Shortcut: draft model / EAGLE head repo.", "string", "speculative", adv=True, mem=True))
    a(_p("spec_tokens", "--spec-tokens", "Speculative tokens",
         "Shortcut: draft tokens per step.", "int", "speculative", min=1, max=32, adv=True))
    # ---------------- network ----------------
    a(_p("served_model_name", "--served-model-name", "Served model name(s)",
         "Extra names the API answers to. The console always serves its own name first.",
         "string_list", "network", adv=True))
    a(_p("api_key", "env:VLLM_API_KEY", "API key",
         "Require this bearer key. Passed through the VLLM_API_KEY environment variable, not argv, so "
         "it is not visible in the process list.", "string", "network"))
    a(_p("uvicorn_log_level", "--uvicorn-log-level", "Uvicorn log level", "HTTP server log level.",
         "enum", "network", default="info",
         choices=["critical", "debug", "error", "info", "trace", "warning"], adv=True))
    a(_p("disable_access_log_for_endpoints", "--disable-access-log-for-endpoints",
         "Silence access log for paths", "Comma-separated paths to omit from access logs, e.g. "
         "/health,/metrics (the console polls both).", "string", "network", adv=True))
    a(_p("enable_request_id_headers", "--enable-request-id-headers", "Request-ID headers",
         "Add X-Request-Id to responses.", "bool", "network", default=False, adv=True))
    a(_p("enable_prompt_tokens_details", "--enable-prompt-tokens-details", "Prompt token details",
         "Return cached-token counts in usage.", "bool", "network", adv=True))
    a(_p("enable_server_load_tracking", "--enable-server-load-tracking", "Server load tracking",
         "Expose in-flight load via /load.", "bool", "network", adv=True))
    a(_p("enable_log_requests", "--enable-log-requests", "Log requests",
         "Log each request (params at INFO, prompts at DEBUG). Prompts may contain secrets.", "bool",
         "network", default=False, adv=True))
    a(_p("max_log_len", "--max-log-len", "Max logged prompt length",
         "Truncate logged prompts/outputs to this many characters/tokens.", "int", "network", min=0,
         adv=True))
    a(_p("disable_fastapi_docs", "--disable-fastapi-docs", "Disable FastAPI docs",
         "Turn off /docs, /redoc and the OpenAPI schema.", "bool", "network", default=False, adv=True))
    a(_p("root_path", "--root-path", "Root path", "FastAPI root_path behind a path-routing proxy.",
         "string", "network", adv=True))
    a(_p("otlp_traces_endpoint", "--otlp-traces-endpoint", "OTLP traces endpoint",
         "Send OpenTelemetry traces here (for example http://otel-collector:4317).",
         "string", "network", adv=True))
    a(_p("logging_level", "env:VLLM_LOGGING_LEVEL", "Log level",
         "Sets VLLM_LOGGING_LEVEL for the engine.", "enum", "network", default="INFO",
         choices=["DEBUG", "INFO", "WARNING", "ERROR"], adv=True))
    # ---------------- advanced ----------------
    a(_p("enable_lora", "--enable-lora", "Enable LoRA", "Serve LoRA adapters on top of the base model.",
         "bool", "advanced"))
    a(_p("lora_modules", "--lora-modules", "LoRA modules",
         "Adapters as name=path (or JSON) entries; requires enable_lora.", "string_list", "advanced"))
    a(_p("max_loras", "--max-loras", "Max LoRAs per batch", "Adapters active in one batch.", "int",
         "advanced", default=1, min=1, adv=True, mem=True))
    a(_p("max_lora_rank", "--max-lora-rank", "Max LoRA rank", "Largest adapter rank accepted.", "enum",
         "advanced", default="16",
         choices=["1", "8", "16", "32", "64", "128", "256", "320", "512"], adv=True, mem=True))
    a(_p("max_cpu_loras", "--max-cpu-loras", "Max CPU LoRAs", "Adapters cached in host RAM.", "int",
         "advanced", min=1, adv=True))
    a(_p("lora_dtype", "--lora-dtype", "LoRA dtype", "dtype for adapter weights.", "enum", "advanced",
         default="auto", choices=["auto", "bfloat16", "float16"], adv=True))
    a(_p("fully_sharded_loras", "--fully-sharded-loras", "Fully sharded LoRAs",
         "Shard all LoRA math over TP (helps at high rank / large TP).", "bool", "advanced",
         default=False, adv=True))
    a(_p("enable_mfu_metrics", "--enable-mfu-metrics", "MFU metrics",
         "Export estimated FLOPs/bytes metrics.", "bool", "advanced", default=False, adv=True))
    a(_p("kv_cache_metrics", "--kv-cache-metrics", "KV residency metrics",
         "Sampled KV block lifetime/reuse histograms.", "bool", "advanced", default=False, adv=True))
    a(_p("cudagraph_metrics", "--cudagraph-metrics", "CUDA graph metrics",
         "Padding / dispatch-mode metrics for CUDA graphs.", "bool", "advanced", default=False,
         adv=True))
    a(_p("show_hidden_metrics_for_version", "--show-hidden-metrics-for-version", "Show hidden metrics",
         "Re-enable Prometheus metrics hidden since this version (e.g. 0.10) to keep old dashboards "
         "working.", "string", "advanced", adv=True))
    a(_p("kv_transfer_config", "--kv-transfer-config", "KV transfer config (JSON)",
         "Disaggregated prefill / KV connector configuration.", "json", "advanced", adv=True))
    a(_p("additional_config", "--additional-config", "Additional platform config (JSON)",
         "Platform-specific extra config.", "json", "advanced", adv=True))
    a(_p("worker_multiproc_method", "env:VLLM_WORKER_MULTIPROC_METHOD", "Worker start method (env)",
         "Sets VLLM_WORKER_MULTIPROC_METHOD (fork or spawn).", "enum", "advanced", default="fork",
         choices=["fork", "spawn"], adv=True))
    a(_p("image", "@image", "Container image",
         "Pinned vLLM image tag override (never :latest). Newer families need newer images: Laguna >= "
         "0.25.0, Qwen3.8 / Nemotron Lightning use 0.27.1, Nemotron Omni needs exactly 0.20.0.",
         "string", "advanced", default=DEFAULT_IMAGE, adv=True))
    return c
