"""SGLang parameter catalog (data only).

Source of truth: `python3 -m sglang.launch_server --help` and the `ServerArgs` dataclass defaults
of the pinned image (lmsysorg/sglang:v0.5.14-cu130), both
read from that image locally. Flags/choices/defaults below match it. `docs_url` fragments point at
the official server-arguments page section; the section anchors are recalled, not fetched.

Deliberately NOT in the catalog (owned by build_launch): --model-path, --host, --port.
"""
from __future__ import annotations

from typing import Any

from engine_console.adapters.base import ParamGroup, ParamSpec, ParamType

DOCS = "https://docs.sglang.ai/advanced_features/server_arguments.html"

QUANT_CHOICES = [
    "awq", "fp8", "mxfp8", "gptq", "marlin", "gptq_marlin", "awq_marlin", "bitsandbytes", "gguf",
    "modelopt", "modelopt_fp8", "modelopt_fp4", "nvfp4_online", "modelopt_mixed", "petit_nvfp4",
    "w8a8_int8", "w8a8_fp8", "moe_wna16", "qoq", "w4afp8", "mxfp4", "auto-round",
    "compressed-tensors", "modelslim", "quark", "quark_int4fp8_moe", "quark_mxfp4", "unquant",
]
ATTENTION_BACKENDS = [
    "triton", "torch_native", "flex_attention", "cutlass_mla", "fa3", "fa4", "flashinfer",
    "flashmla", "trtllm_mla", "cutedsl_mla", "trtllm_mha", "dual_chunk_flash_attn",
]
TOOL_PARSERS = [
    "auto", "apertus2509", "cohere_command4", "deepseekv3", "deepseekv31", "deepseekv32",
    "deepseekv4", "glm", "glm45", "glm47", "gpt-oss", "kimi_k2", "lfm2", "llama3", "mimo",
    "minicpm5", "mistral", "poolside_v1", "pythonic", "qwen", "qwen25", "qwen3_coder", "step3",
    "step3p5", "minimax-m2", "trinity", "interns1", "hermes", "hunyuan", "gigachat3", "gemma4",
]
REASONING_PARSERS = [
    "auto", "apertus2509", "deepseek-r1", "deepseek-v3", "deepseek-v4", "glm45", "hunyuan",
    "gpt-oss", "kimi", "kimi_k2", "mimo", "poolside_v1", "qwen3", "qwen3-thinking", "minimax",
    "minimax-append-think", "step3", "step3p5", "mistral", "nemotron_3", "interns1", "gemma4",
    "cohere_command4",
]
LOAD_FORMATS = [
    "auto", "pt", "safetensors", "npcache", "dummy", "sharded_state", "gguf", "bitsandbytes",
    "mistral", "layered", "flash_rl", "remote", "remote_instance", "fastsafetensors", "private",
    "runai_streamer",
]
MOE_RUNNERS = [
    "auto", "deep_gemm", "triton", "triton_kernel", "flashinfer_trtllm", "flashinfer_trtllm_routed",
    "flashinfer_cutlass", "flashinfer_mxfp4", "flashinfer_cutedsl", "cutlass", "marlin",
]
KV_DTYPES = ["auto", "fp8_e5m2", "fp8_e4m3", "bf16", "bfloat16", "fp4_e2m1"]
SPEC_ALGOS = ["EAGLE", "EAGLE3", "NEXTN", "STANDALONE", "NGRAM", "DFLASH"]

_ANCHOR: dict[str, str] = {
    "memory": "#memory-and-scheduling-parameters", "parallelism": "#parallelism",
    "context": "#model-and-tokenizer", "quantization": "#quantization-and-data-type",
    "scheduling": "#memory-and-scheduling-parameters", "tools_reasoning": "#tool-call-parser",
    "performance": "#kernel-backends-attention-sampling-grammar", "speculative": "#speculative-decoding",
    "network": "#api-related-configuration", "advanced": "#logging",
}


def _p(key: str, typ: ParamType, group: ParamGroup, label: str, help_: str, default: Any = None, *,
       choices: list[str] | None = None, lo: float | None = None, hi: float | None = None,
       adv: bool = False, mem: bool = False, restart: bool = True, flag: str | None = None) -> ParamSpec:
    return ParamSpec(
        key=key, flag=flag or "--" + key.replace("_", "-"), label=label, help=help_, type=typ,
        group=group, default=default, choices=choices, min=lo, max=hi, advanced=adv,
        affects_memory=mem, requires_restart=restart, docs_url=DOCS + _ANCHOR[group],
    )


def build_catalog() -> list[ParamSpec]:
    p = _p
    return [
        # ---- memory ----
        p("mem_fraction_static", "float", "memory", "Static memory fraction",
          "Fraction of GPU memory for weights + KV pool (NOT weights only). The remainder must cover "
          "activations, CUDA graphs and kernel scratch: measured OOM risk above ~0.92 on sm_120.",
          None, lo=0.1, hi=0.99, mem=True),
        p("max_total_tokens", "int", "memory", "Max total tokens (KV pool)",
          "Hard cap on tokens in the KV pool; normally derived from mem-fraction-static. Debug/dev use.",
          None, lo=1, adv=True, mem=True),
        p("kv_cache_dtype", "enum", "memory", "KV cache dtype",
          "KV storage type. fp8_e4m3 halves KV memory vs bf16; fp4_e2m1 is experimental.",
          "auto", choices=KV_DTYPES, mem=True),
        p("cpu_offload_gb", "int", "memory", "CPU offload (GB)",
          "GB of host RAM used to offload weights. Slow over PCIe; last resort.",
          0, lo=0, adv=True, mem=True),
        p("swa_full_tokens_ratio", "float", "memory", "SWA/full KV token ratio",
          "Sliding-window-layer KV tokens as a ratio of full-attention KV tokens (hybrid SWA models).",
          0.8, lo=0.0, hi=1.0, adv=True, mem=True),
        p("disable_hybrid_swa_memory", "bool", "memory", "Disable hybrid SWA pool",
          "Disable the hybrid sliding-window KV pool.", False, adv=True, mem=True),
        p("page_size", "int", "memory", "KV page size",
          "Tokens per KV page. Some backends (FlashMLA, trtllm) need 64; recipe for hybrid/mamba uses 64.",
          None, lo=1, adv=True, mem=True),
        p("enable_memory_saver", "bool", "memory", "Memory saver",
          "Allow release/resume of memory occupation (RL/colocation setups).", False, adv=True),
        # ---- context ----
        p("context_length", "int", "context", "Context length",
          "Max context (prompt+output). Defaults to the model config. Pool-capped rather than "
          "VRAM-capped: raising it costs ~no extra memory until the KV pool is exhausted. "
          "Longer than the model's native length needs SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1.",
          None, lo=128, mem=True),
        p("chunked_prefill_size", "int", "context", "Chunked prefill size",
          "Tokens per prefill chunk; -1 disables chunked prefill. Smaller = lower activation memory "
          "and smoother decode latency. Auto-chosen from GPU memory when unset.", None, lo=-1, mem=True),
        p("max_prefill_tokens", "int", "context", "Max prefill tokens",
          "Max tokens in a prefill batch (real bound is max of this and context length).",
          16384, lo=1, adv=True, mem=True),
        p("allow_auto_truncate", "bool", "context", "Auto-truncate long prompts",
          "Truncate over-long inputs instead of returning an error.", False, adv=True),
        p("json_model_override_args", "json", "context", "Model config overrides (JSON)",
          "JSON dict overriding config.json values, e.g. rope scaling for YaRN.", None, adv=True),
        p("revision", "string", "context", "Model revision",
          "Branch, tag or commit of the model repo.", None, adv=True),
        p("trust_remote_code", "bool", "context", "Trust remote code",
          "Allow custom modelling code from the Hub. Executes downloaded code: only for reviewed repos.",
          False),
        p("tokenizer_path", "string", "context", "Tokenizer path",
          "Tokenizer repo/path if different from the model.", None, adv=True),
        p("model_impl", "string", "context", "Model implementation",
          "auto | sglang | transformers. auto uses SGLang's native implementation, else Transformers.",
          "auto", adv=True),
        p("enable_multimodal", "bool", "context", "Enable multimodal",
          "Enable the multimodal path for the served model (no-op for text-only models).", False),
        p("is_embedding", "bool", "context", "Embedding mode",
          "Serve a CausalLM as an embedding model.", False, adv=True),
        p("dtype", "enum", "quantization", "Weights/activation dtype",
          "auto uses the checkpoint dtype (BF16 for BF16 models).",
          "auto", choices=["auto", "half", "float16", "bfloat16", "float", "float32"], mem=True),
        # ---- quantization ----
        p("quantization", "enum", "quantization", "Quantization",
          "Weight quantization method. Prefer fp8 on sm_120; pre-quantized checkpoints "
          "(NVFP4/AWQ/GPTQ) carry their own config, so leave unset or use the matching value "
          "(modelopt_fp4 for ModelOpt NVFP4). NVFP4-MoE is best served here rather than vLLM.",
          None, choices=QUANT_CHOICES, mem=True),
        p("load_format", "enum", "quantization", "Weight load format",
          "Format of the weights to load (auto picks safetensors when present).",
          "auto", choices=LOAD_FORMATS, adv=True),
        p("fp8_gemm_backend", "enum", "quantization", "FP8 GEMM backend",
          "Runner for blockwise FP8 GEMM.", None,
          choices=["auto", "deep_gemm", "flashinfer_trtllm", "flashinfer_cutlass",
                   "flashinfer_deepgemm", "cutlass", "triton", "aiter"], adv=True),
        p("moe_runner_backend", "enum", "quantization", "MoE runner backend",
          "Kernel backend for MoE experts. On sm_120 NVFP4-MoE prefer the default (auto) or "
          "flashinfer_cutlass; test before forcing.", "auto", choices=MOE_RUNNERS, adv=True),
        # ---- parallelism ----
        p("tp_size", "int", "parallelism", "Tensor parallel size",
          "GPUs per replica (--tp-size). Must not exceed the GPUs selected.", 1, lo=1, mem=True),
        p("dp_size", "int", "parallelism", "Data parallel size",
          "Number of replicas (or attention DP groups when enable-dp-attention).", 1, lo=1, mem=True),
        p("ep_size", "int", "parallelism", "Expert parallel size",
          "Expert parallelism for MoE models; must divide tp-size.", 1, lo=1, adv=True, mem=True),
        p("pp_size", "int", "parallelism", "Pipeline parallel size",
          "Pipeline stages.", 1, lo=1, adv=True, mem=True),
        p("enable_dp_attention", "bool", "parallelism", "DP attention",
          "Data-parallel attention + tensor-parallel FFN. dp-size must equal tp-size. Supported for "
          "DeepSeek-style and Qwen MoE models.", False, adv=True, mem=True),
        p("moe_a2a_backend", "enum", "parallelism", "MoE all-to-all backend",
          "All-to-all backend for expert parallelism.", None,
          choices=["none", "deepep", "mooncake", "nixl", "mori", "ascend_fuseep", "flashinfer", "megamoe"],
          adv=True),
        p("enable_eplb", "bool", "parallelism", "Expert-parallel load balancer",
          "Enable EPLB (needs expert parallelism).", False, adv=True),
        p("ep_num_redundant_experts", "int", "parallelism", "Redundant experts",
          "Extra redundant experts for EPLB.", None, lo=0, adv=True, mem=True),
        p("disable_custom_all_reduce", "bool", "parallelism", "Disable custom all-reduce",
          "Fall back to NCCL all-reduce (try if TP hangs on PCIe-only boxes).", False, adv=True),
        p("nnodes", "int", "parallelism", "Node count",
          "Multi-node only; this console runs single-node.", 1, lo=1, adv=True),
        # ---- scheduling ----
        p("max_running_requests", "int", "scheduling", "Max running requests",
          "Cap on concurrently running requests (also sizes CUDA-graph batch and mamba caches).",
          None, lo=1, mem=True),
        p("max_queued_requests", "int", "scheduling", "Max queued requests",
          "Reject new requests once this many are queued.", None, lo=1, adv=True),
        p("schedule_policy", "enum", "scheduling", "Schedule policy",
          "Request ordering. lpm (longest prefix match) favours cache hits; fcfs is fair.",
          "fcfs", choices=["lpm", "random", "fcfs", "dfs-weight", "lof", "priority", "routing-key"]),
        p("schedule_conservativeness", "float", "scheduling", "Schedule conservativeness",
          "Larger = more conservative admission. Raise if requests are retracted often.",
          1.0, lo=0.0),
        p("enable_mixed_chunk", "bool", "scheduling", "Mixed chunk",
          "Mix prefill and decode tokens in a batch (needs chunked prefill).", False),
        p("enable_priority_scheduling", "bool", "scheduling", "Priority scheduling",
          "Higher-priority integers scheduled first.", False, adv=True),
        p("disable_overlap_schedule", "bool", "scheduling", "Disable overlap scheduler",
          "Turn off CPU/GPU overlap scheduling (debugging).", False, adv=True),
        p("stream_interval", "int", "scheduling", "Stream interval",
          "Tokens per streamed chunk: lower = smoother, higher = more throughput.", 1, lo=1, adv=True),
        p("sleep_on_idle", "bool", "scheduling", "Sleep on idle",
          "Reduce CPU usage while idle.", False, adv=True),
        p("watchdog_timeout", "int", "scheduling", "Watchdog timeout (s)",
          "Crash if one forward batch exceeds this; raise for very long prefills.", 300, lo=1, adv=True),
        # ---- performance ----
        p("attention_backend", "enum", "performance", "Attention backend",
          "Attention kernels. flashinfer is the safe default on sm_120; fa3/flashmla/*_mla target "
          "Hopper/datacenter Blackwell.", None, choices=ATTENTION_BACKENDS),
        p("prefill_attention_backend", "enum", "performance", "Prefill attention backend",
          "Override the attention backend for prefill only.", None, choices=ATTENTION_BACKENDS, adv=True),
        p("decode_attention_backend", "enum", "performance", "Decode attention backend",
          "Override the attention backend for decode only.", None, choices=ATTENTION_BACKENDS, adv=True),
        p("sampling_backend", "enum", "performance", "Sampling backend",
          "Kernels for sampling layers.", None, choices=["pytorch", "flashinfer", "ascend"], adv=True),
        p("grammar_backend", "enum", "performance", "Grammar backend",
          "Backend for constrained (JSON/regex) decoding.", None,
          choices=["xgrammar", "outlines", "llguidance", "none"], adv=True),
        p("disable_radix_cache", "bool", "performance", "Disable radix cache",
          "Turn off RadixAttention prefix caching (hurts agent/multi-turn workloads).", False),
        p("radix_eviction_policy", "enum", "performance", "Radix eviction policy",
          "Prefix-cache eviction policy.", "lru", choices=["lru", "lfu", "slru", "priority"], adv=True),
        p("enable_torch_compile", "bool", "performance", "torch.compile",
          "Experimental; long warmup, helps small-batch decode.", False, adv=True),
        p("torch_compile_max_bs", "int", "performance", "torch.compile max batch",
          "Max batch size with torch.compile.", 32, lo=1, adv=True),
        p("cuda_graph_max_bs_decode", "int", "performance", "CUDA graph max batch (decode)",
          "Largest decode batch captured as a CUDA graph; bigger = more graph memory + capture time. "
          "(`--cuda-graph-max-bs` is a deprecated alias in this version.)",
          None, lo=1, mem=True, flag="--cuda-graph-max-bs-decode"),
        p("cuda_graph_backend_decode", "enum", "performance", "CUDA graph backend (decode)",
          "full | breakable | tc_piecewise | disabled. Replaces the deprecated --disable-cuda-graph.",
          None, choices=["full", "breakable", "tc_piecewise", "disabled"], adv=True, mem=True),
        p("cuda_graph_backend_prefill", "enum", "performance", "CUDA graph backend (prefill)",
          "Prefill-phase graph backend; disabled saves memory and capture time.",
          None, choices=["full", "breakable", "tc_piecewise", "disabled"], adv=True, mem=True),
        p("cuda_graph_max_bs_prefill", "int", "performance", "CUDA graph max tokens (prefill)",
          "Largest prefill size captured.", None, lo=1, adv=True, mem=True),
        p("enable_tokenizer_batch_encode", "bool", "performance", "Batch tokenization",
          "Batch-tokenize concurrent text inputs (not for image inputs).", False, adv=True),
        p("enable_deterministic_inference", "bool", "performance", "Deterministic inference",
          "Batch-invariant kernels for reproducible outputs; slower.", False, adv=True),
        p("skip_server_warmup", "bool", "performance", "Skip warmup",
          "Skip the warmup request at startup.", False, adv=True),
        # ---- tools & reasoning ----
        p("tool_call_parser", "enum", "tools_reasoning", "Tool-call parser",
          "REQUIRED for agentic clients: without it tool calls stay raw text and tool_calls is empty. "
          "Must match the model's dialect (Qwen3.x Coder-style XML needs qwen3_coder, not qwen).",
          None, choices=TOOL_PARSERS),
        p("reasoning_parser", "enum", "tools_reasoning", "Reasoning parser",
          "Splits <think> output into reasoning_content.", None, choices=REASONING_PARSERS),
        p("chat_template", "string", "tools_reasoning", "Chat template",
          "Builtin template name or a template file path inside the container.", None, adv=True),
        p("served_model_name", "string", "tools_reasoning", "Served model name",
          "Name returned by /v1/models. Defaults to the console-provided name.", None),
        p("preferred_sampling_params", "json", "tools_reasoning", "Preferred sampling params",
          "JSON sampling defaults advertised by /get_model_info.", None, adv=True),
        p("enable_cache_report", "bool", "tools_reasoning", "Report cached tokens",
          "Return cached-token counts in usage.prompt_tokens_details.", False, adv=True),
        # ---- speculative ----
        p("speculative_algorithm", "enum", "speculative", "Speculative algorithm",
          "EAGLE/EAGLE3/STANDALONE need a draft model; NEXTN uses the model's built-in MTP head; "
          "NGRAM needs no draft model.", None, choices=SPEC_ALGOS, mem=True),
        p("speculative_draft_model_path", "string", "speculative", "Draft model",
          "HF repo or path of the draft model.", None, mem=True),
        p("speculative_num_steps", "int", "speculative", "Draft steps",
          "Steps sampled from the draft model.", None, lo=1, mem=True),
        p("speculative_eagle_topk", "int", "speculative", "EAGLE top-k",
          "Tokens sampled per draft step (1 = chain drafting).", None, lo=1),
        p("speculative_num_draft_tokens", "int", "speculative", "Draft tokens",
          "Tokens verified per step.", None, lo=1),
        p("speculative_accept_threshold_single", "float", "speculative", "Accept threshold",
          "Accept a draft token if its target probability exceeds this.", 1.0, lo=0.0, hi=1.0, adv=True),
        p("speculative_draft_model_quantization", "enum", "speculative", "Draft quantization",
          "Quantization of the draft model (unquant keeps it bf16).", None, choices=QUANT_CHOICES, adv=True),
        # ---- LoRA ----
        p("enable_lora", "bool", "advanced", "Enable LoRA",
          "Enable LoRA serving (implied by lora-paths).", False),
        p("lora_paths", "string_list", "advanced", "LoRA adapters",
          "Adapters as <path> or <name>=<path>.", None),
        p("max_loras_per_batch", "int", "advanced", "Max LoRAs per batch",
          "Adapters (incl. base-only) in one running batch.", 8, lo=1, adv=True, mem=True),
        p("max_lora_rank", "int", "advanced", "Max LoRA rank",
          "Inferred from adapters when unset.", None, lo=1, adv=True, mem=True),
        p("lora_backend", "enum", "advanced", "LoRA backend",
          "Kernel backend for multi-LoRA.", "csgmv",
          choices=["triton", "csgmv", "ascend", "torch_native"], adv=True),
        # ---- hierarchical cache ----
        p("enable_hierarchical_cache", "bool", "memory", "Hierarchical KV cache (HiCache)",
          "Spill KV to host RAM/storage. Needs the radix cache enabled.", False, adv=True),
        p("hicache_ratio", "float", "memory", "HiCache host/device ratio",
          "Host KV pool size relative to the device pool.", 2.0, lo=1.0, adv=True),
        p("hicache_size", "int", "memory", "HiCache host size (GB)",
          "Host pool size in GB; overrides hicache-ratio.", None, lo=1, adv=True),
        p("hicache_write_policy", "enum", "memory", "HiCache write policy",
          "How device KV is written to host.", "write_through",
          choices=["write_back", "write_through", "write_through_selective"], adv=True),
        p("hicache_io_backend", "enum", "memory", "HiCache IO backend",
          "CPU-GPU transfer kernel.", "kernel", choices=["direct", "kernel", "kernel_ascend"], adv=True),
        p("hicache_mem_layout", "enum", "memory", "HiCache host layout",
          "Host memory pool layout.", "page_first",
          choices=["layer_first", "page_first", "page_first_direct", "page_first_kv_split", "page_head"],
          adv=True),
        p("hicache_storage_backend", "enum", "memory", "HiCache storage backend",
          "Optional third tier.", None,
          choices=["file", "mooncake", "hf3fs", "nixl", "aibrix", "dynamic", "eic", "simm"], adv=True),
        # media-url-max-file-size-mb / default-chat-template-kwargs exist only in the custom flashnext
        # custom builds, not in the pinned stock image, so they are not offered.
        # ---- network / API ----
        p("api_key", "string", "network", "API key",
          "Bearer key required by the server. Secret: appears in the container argv (docker inspect); "
          "prefer an authenticating proxy in front of the engine.", None),
        p("enable_metrics", "bool", "network", "Prometheus metrics",
          "Exposes /metrics. The console forces this on; setting false is ignored.", True, adv=True),
        p("enable_request_time_stats_logging", "bool", "network", "Request time stats logging",
          "Log per-request timing stats.", False, adv=True),
        # ---- advanced / logging ----
        p("random_seed", "int", "advanced", "Random seed", "Seed for sampling reproducibility.", None, adv=True),
        p("log_level", "enum", "advanced", "Log level", "Logging level of all loggers.", "info",
          choices=["debug", "info", "warning", "error", "critical"], adv=True),
        p("decode_log_interval", "int", "advanced", "Decode log interval",
          "Decode iterations between log/metric lines.", 40, lo=1, adv=True),
        p("log_requests", "bool", "advanced", "Log requests",
          "Log metadata/inputs/outputs of every request: leaks prompts into logs.", False, adv=True),
        p("enable_trace", "bool", "advanced", "OpenTelemetry tracing",
          "Emit OTel traces (needs otlp-traces-endpoint).", False, adv=True),
        p("otlp_traces_endpoint", "string", "advanced", "OTLP endpoint",
          "host:port of the collector when tracing is enabled.", "localhost:4317", adv=True),
    ]


CATALOG_VERSION_NOTE = "lmsysorg/sglang:v0.5.14-cu130 (--help + ServerArgs defaults read locally)"
