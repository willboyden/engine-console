"""SGLang parameter catalog (data only).

Pinned engine: ``lmsysorg/sglang:v0.5.20-cu130`` (a local image; nothing was pulled).

Provenance (what was actually run, and what was not):
  * Flags, defaults and choices were read inside the pinned image, without a GPU, by building the server's own argument
    parser (``ServerArgs.add_cli_args``) and dumping every action: ``docker run --rm --network none --cap-drop ALL
    --entrypoint python3 lmsysorg/sglang:v0.5.20-cu130``. The printed ``python3 -m sglang.launch_server --help`` of the
    same image was read as well. A condensed copy (flag, aliases, choices, default, store-true or not) is committed as
    ``tests/sglang/fixtures/help_flags_v0.5.20.json``; ``tests/sglang`` checks this catalog against it offline.
  * SGLang 0.5.20 prints no defaults in ``--help``. The defaults here are the parser's declared defaults, which are the
    raw inputs: several (``mem_fraction_static``, ``chunked_prefill_size``, ``max_running_requests``, the CUDA-graph batch
    size) are resolved from the card at start-up, which is why they stay ``None`` here.
  * Choice lists that contain entries for hardware this console does not run (AMD ROCm, Ascend NPU, Intel XPU/AMX, Apple
    MLX) are shortened: ``OMITTED_CHOICES`` records exactly what was left out, per flag, so the snapshot test can prove
    that catalog + omitted == the image's list.
  * ``docs_url`` points at the section of https://docs.sglang.io/docs/advanced_features/server_arguments that tabulates the
    flag (heading ids read from the page's HTML on 2026-10-04). The page documents the main branch, not v0.5.20, and SGLang
    publishes no per-release docs, so the image's own ``--help`` stays the reference for this version.
  * Also present in the other local image (``lmsysorg/sglang:v0.5.14-cu130``, same parser dump): see ``NOT_IN_V0_5_14`` for
    the catalog flags that image does not have; every other flag below exists in both.

Renamed or deprecated spellings (checked in both local images): ``--cuda-graph-max-bs`` exists in 0.5.14 but not in 0.5.20, so
the catalog uses ``--cuda-graph-max-bs-decode``, which both have; ``--disable-cuda-graph`` is deprecated in 0.5.20 in favour of
``--cuda-graph-backend-decode`` / ``--cuda-graph-backend-prefill`` set to ``disabled`` (the catalog offers those).

Removed from the previous catalog: ``api_key``. SGLang 0.5.20 reads its API key only from ``--api-key`` (a search of the
package finds no environment variable that supplies it), and the console never puts a secret on a command line, so the
parameter would always be refused. Put an authenticating proxy in front of the engine instead.
Added: the hybrid-model controls (``max_mamba_cache_size``, ``mamba_ssm_dtype``, ``mamba_radix_cache_strategy``,
``mamba_track_interval``, ``mamba_backend``, ``linear_attn_*_backend``), ``fp4_gemm_backend``, ``language_model_only``,
``default_chat_template_kwargs``, ``sampling_defaults``, ``media_url_max_file_size_mb``, ``disable_flashinfer_autotune`` and
``allow_longer_context_len``, which sets the environment variable ``SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN`` (the image's
``configs/model_config.py`` refuses a ``--context-length`` above the derived length unless it is 1, and the variable is
defined in ``environ.py`` and listed on https://docs.sglang.io/docs/references/environment_variables).
Pseudo-flag convention, handled in ``SglangAdapter.build_launch_ctx`` and never emitted as argv: ``env:NAME`` sets the
container environment variable NAME.

Deliberately NOT in the catalog (owned by build_launch): --model-path, --host, --port, --enable-metrics' forced value.
"""
from __future__ import annotations

from typing import Any

from engine_console.adapters.base import LearnLevel, ParamGroup, ParamSpec, ParamType

SGLANG_VERSION = "0.5.20"
DEFAULT_IMAGE = f"lmsysorg/sglang:v{SGLANG_VERSION}-cu130"
DOCS = "https://docs.sglang.io/docs/advanced_features/server_arguments"
ENV_DOCS = "https://docs.sglang.io/docs/references/environment_variables"

# Catalog flags that the v0.5.14 image's parser does not have (filled in by the check described above).
NOT_IN_V0_5_14: tuple[str, ...] = ("--language-model-only", "--media-url-max-file-size-mb", "--default-chat-template-kwargs")

# --- choices read from the pinned image (see module docstring) ---------------------------------------------------------
QUANT_CHOICES = [
    "awq", "fp8", "mxfp8", "gptq", "gptq_marlin", "awq_marlin", "bitsandbytes", "modelopt", "modelopt_fp8",
    "modelopt_fp4", "nvfp4_online", "modelopt_mixed", "w8a8_int8", "w8a8_fp8", "moe_wna16", "w4afp8", "mxfp4",
    "auto-round", "auto-round-int8", "compressed-tensors", "mxfp_w4a8", "unquant", "humming",
]
ATTENTION_BACKENDS = [
    "triton", "torch_native", "flex_attention", "dsa", "nsa", "qsa", "dsv4", "compressed", "fa3", "fa4",
    "flashinfer", "flashmla", "trtllm_mla", "cutedsl_mla", "tokenspeed_mla", "trtllm_mha", "hpc_ops",
    "minicpm_flashattn", "minicpm_flashinfer",
]
LOAD_FORMATS = [
    "auto", "pt", "safetensors", "npcache", "dummy", "sharded_state", "presharded", "gguf", "expert_pack",
    "bitsandbytes", "mistral", "layered", "flash_rl", "remote", "remote_instance", "fastsafetensors", "private",
    "runai_streamer",
]
MOE_RUNNERS = [
    "auto", "deep_gemm", "triton", "triton_kernel", "flashinfer_trtllm", "experimental_sgl_trtllm",
    "flashinfer_trtllm_routed", "flashinfer_cutlass", "flashinfer_mxfp4", "flashinfer_cutedsl",
    "flashinfer_megamoe", "cutlass", "marlin", "humming", "experimental_sgl_marlin", "hpc_ops", "megamoe",
]
FP8_GEMM = ["auto", "deep_gemm", "flashinfer_trtllm", "flashinfer_cutlass", "flashinfer_deepgemm",
            "flashinfer_cutedsl", "cutlass", "triton"]
FP4_GEMM = ["auto", "flashinfer_cudnn", "flashinfer_cutedsl", "flashinfer_cutlass", "flashinfer_trtllm", "marlin"]
KV_DTYPES = ["auto", "fp8_e5m2", "fp8_e4m3", "mxfp8", "bf16", "bfloat16", "nvfp4", "fp4_mx_block16", "fp4_e2m1"]
MOE_A2A = ["none", "deepep", "mooncake", "nixl", "flashinfer", "megamoe", "deepep_v2", "pplx", "flashinfer_megamoe"]
TOOL_PARSERS = [
    "auto", "apertus2509", "cohere_command4", "deepseekv3", "deepseekv31", "deepseekv32", "deepseekv4", "dots",
    "glm", "glm45", "glm47", "gpt-oss", "k2_horizon", "kimi_k2", "kimi_k3", "lfm2", "ling3", "llama3", "mimo",
    "minicpm5", "mistral", "muse", "poolside_v1", "pythonic", "qwen", "qwen25", "qwen3_coder", "spark25", "step3",
    "step3p5", "minimax-m2", "minimax-m3", "nanbeige", "trinity", "interns1", "hermes", "hunyuan", "gigachat3",
    "gemma4", "inkling",
]
REASONING_PARSERS = [
    "auto", "apertus2509", "deepseek-r1", "deepseek-v3", "deepseek-v4", "dots", "glm45", "ling3", "hunyuan",
    "gpt-oss", "k2_horizon", "kimi", "kimi_k2", "kimi_k3", "mimo", "muse", "poolside_v1", "qwen3", "qwen3-thinking",
    "minimax", "minimax-append-think", "minimax-m3", "nanbeige", "step3", "step3p5", "mistral", "nemotron_3",
    "granite_thinking_parser", "interns1", "gemma4", "inkling", "cohere_command4",
]
SCHEDULE_POLICIES = ["lpm", "random", "fcfs", "dfs-weight", "lof", "priority", "routing-key", "hrrn"]
HICACHE_STORAGE = ["file", "mooncake", "hf3fs", "nixl", "aibrix", "dynamic", "eic", "simm", "shm"]
LINEAR_ATTN = ["triton", "cutedsl", "flashinfer", "flashkda", "nvidia_kda", "ptx_kda", "helion"]
LORA_BACKENDS = ["triton", "csgmv", "torch_native"]
SAMPLING_BACKENDS = ["flashinfer", "pytorch"]
GRAMMAR_BACKENDS = ["xgrammar", "outlines", "llguidance", "none"]
CUDA_GRAPH_BACKENDS = ["full", "breakable", "tc_piecewise", "disabled"]
# `--speculative-algorithm` prints no choices; the help text names these builtins. UNO needs a draft LoRA flag that this
# catalog does not offer.
SPEC_ALGOS = ["EAGLE", "EAGLE3", "NEXTN", "STANDALONE", "NGRAM", "DFLASH", "DSPARK"]

# flag -> choices the image lists that the catalog leaves out (non-NVIDIA hardware, Apple MLX, GGUF).
OMITTED_CHOICES: dict[str, frozenset[str]] = {
    "--quantization": frozenset({"petit_nvfp4", "modelslim", "quark", "quark_int4fp8_moe", "quark_mxfp4", "mlx_q4",
                                 "mlx_q8", "gguf"}),
    "--speculative-draft-model-quantization": frozenset({"petit_nvfp4", "modelslim", "quark", "quark_int4fp8_moe",
                                                         "quark_mxfp4", "mlx_q4", "mlx_q8", "gguf"}),
    "--attention-backend": frozenset({"aiter", "wave", "intel_amx", "ascend", "intel_xpu"}),
    "--prefill-attention-backend": frozenset({"aiter", "wave", "intel_amx", "ascend", "intel_xpu"}),
    "--decode-attention-backend": frozenset({"aiter", "wave", "intel_amx", "ascend", "intel_xpu"}),
    "--moe-runner-backend": frozenset({"aiter", "intel_xpu"}),
    "--fp8-gemm-backend": frozenset({"aiter"}),
    "--moe-a2a-backend": frozenset({"mori", "ascend_fuseep", "ascend_tp"}),
    "--hicache-storage-backend": frozenset({"sim", "npu_memcache", "mori"}),
    "--hicache-io-backend": frozenset({"kernel_ascend"}),
    "--linear-attn-backend": frozenset({"intel_xpu"}),
    "--linear-attn-decode-backend": frozenset({"intel_xpu"}),
    "--linear-attn-prefill-backend": frozenset({"intel_xpu"}),
    "--lora-backend": frozenset({"ascend"}),
    "--sampling-backend": frozenset({"ascend"}),
}

# docs section id -> catalog keys (section ids are the heading ids of the server-arguments page).
_SECTIONS: dict[str, list[str]] = {
    "api-related": [
        "default_chat_template_kwargs", "sampling_defaults", "tool_call_parser", "reasoning_parser",
        "chat_template", "served_model_name", "enable_cache_report",
    ],
    "data-parallelism": ["dp_size"],
    "encode-prefill-disaggregation": ["language_model_only"],
    "for-multi-modal": ["media_url_max_file_size_mb"],
    "hierarchical-cache": [
        "enable_hierarchical_cache", "hicache_ratio", "hicache_size", "hicache_write_policy",
        "hicache_io_backend", "hicache_mem_layout", "hicache_storage_backend",
    ],
    "http-server": ["skip_server_warmup"],
    "kernel-backends-attention-sampling-grammar-gemm": [
        "fp8_gemm_backend", "fp4_gemm_backend", "attention_backend", "prefill_attention_backend",
        "decode_attention_backend", "sampling_backend", "grammar_backend", "disable_flashinfer_autotune",
    ],
    "logging": [
        "enable_metrics", "enable_request_time_stats_logging", "log_level", "decode_log_interval",
        "log_requests", "enable_trace", "otlp_traces_endpoint",
    ],
    "lora": ["enable_lora", "lora_paths", "max_loras_per_batch", "max_lora_rank", "lora_backend"],
    "mamba-cache": [
        "max_mamba_cache_size", "mamba_ssm_dtype", "mamba_radix_cache_strategy", "mamba_track_interval",
        "mamba_backend", "linear_attn_backend", "linear_attn_decode_backend", "linear_attn_prefill_backend",
    ],
    "memory-and-scheduling": [
        "mem_fraction_static", "max_total_tokens", "swa_full_tokens_ratio", "disable_hybrid_swa_memory",
        "page_size", "chunked_prefill_size", "max_prefill_tokens", "max_running_requests", "max_queued_requests",
        "schedule_policy", "schedule_conservativeness", "enable_priority_scheduling", "radix_eviction_policy",
    ],
    "model-and-tokenizer": [
        "context_length", "revision", "trust_remote_code", "tokenizer_path", "model_impl", "enable_multimodal",
        "is_embedding", "load_format",
    ],
    "model-override-args": ["json_model_override_args", "preferred_sampling_params"],
    "moe": ["moe_runner_backend", "ep_size", "moe_a2a_backend", "enable_eplb", "ep_num_redundant_experts"],
    "multi-node-distributed-serving": ["nnodes"],
    "offloading": ["cpu_offload_gb"],
    "optimization/debug-options": [
        "enable_memory_saver", "allow_auto_truncate", "enable_dp_attention", "disable_custom_all_reduce",
        "enable_mixed_chunk", "disable_overlap_schedule", "disable_radix_cache", "enable_torch_compile",
        "torch_compile_max_bs", "cuda_graph_max_bs_decode", "cuda_graph_backend_decode",
        "cuda_graph_backend_prefill", "cuda_graph_max_bs_prefill", "enable_tokenizer_batch_encode",
        "enable_deterministic_inference",
    ],
    "quantization-and-data-type": ["kv_cache_dtype", "dtype", "quantization"],
    "runtime-options": ["tp_size", "pp_size", "stream_interval", "sleep_on_idle", "watchdog_timeout", "random_seed"],
    "speculative-decoding": [
        "speculative_algorithm", "speculative_draft_model_path", "speculative_num_steps", "speculative_eagle_topk",
        "speculative_num_draft_tokens", "speculative_accept_threshold_single", "speculative_draft_model_quantization",
    ],
}
SECTION_OF: dict[str, str] = {key: section for section, keys in _SECTIONS.items() for key in keys}


def _p(key: str, typ: ParamType, group: ParamGroup, label: str, help_: str, default: Any = None, *,
       choices: list[str] | None = None, lo: float | None = None, hi: float | None = None,
       adv: bool = False, mem: bool = False, lvl: LearnLevel = "advanced", unit: str | None = None,
       restart: bool = True, env: str | None = None) -> ParamSpec:
    """One catalog entry. `env` makes it a container environment variable (flag ``env:NAME``) instead of a CLI flag."""
    flag = f"env:{env}" if env else "--" + key.replace("_", "-")
    docs = ENV_DOCS if env else f"{DOCS}#{SECTION_OF[key]}"
    return ParamSpec(
        key=key, flag=flag, label=label, help=help_, type=typ,
        group=group, default=default, choices=choices, min=lo, max=hi, advanced=adv,
        affects_memory=mem, requires_restart=restart, docs_url=docs,
        level=lvl, unit=unit,
    )


def build_catalog() -> list[ParamSpec]:
    p = _p
    return [
        # ---- memory ----
        p("mem_fraction_static", "float", "memory", "Static memory fraction",
          "Share of the GPU memory free at start-up that weights and the KV cache pool may use; the rest is headroom "
          "for activations, CUDA graphs and kernel scratch. Unset: SGLang derives it from the card.",
          None, lo=0.1, hi=0.99, mem=True, lvl="basic"),
        p("max_total_tokens", "int", "memory", "Max total tokens (KV pool)",
          "Hard cap on the tokens in the KV pool; normally derived from mem-fraction-static. For development and "
          "debugging.", None, lo=1, adv=True, mem=True, lvl="expert", unit="tokens"),
        p("kv_cache_dtype", "enum", "memory", "KV cache dtype",
          "KV storage type. fp8_e4m3 halves KV memory against bf16; the nvfp4 and fp4 types need CUDA 12.8 or newer "
          "and are less proven.", "auto", choices=KV_DTYPES, mem=True, lvl="basic"),
        p("cpu_offload_gb", "int", "memory", "CPU offload",
          "GB of host RAM used to offload weights. Slow over PCIe; a last resort.", 0, lo=0, adv=True, mem=True,
          unit="GB"),
        p("swa_full_tokens_ratio", "float", "memory", "SWA/full KV token ratio",
          "Sliding-window-layer KV tokens as a ratio of full-attention KV tokens (hybrid sliding-window models).",
          None, lo=0.0, hi=1.0, adv=True, mem=True, lvl="expert"),
        p("disable_hybrid_swa_memory", "bool", "memory", "Disable hybrid SWA pool",
          "Disable the hybrid sliding-window KV pool.", False, adv=True, mem=True, lvl="expert"),
        p("page_size", "int", "memory", "KV page size",
          "Tokens per KV page. Some backends need a fixed size (FlashMLA and the TRT-LLM ones use 64).",
          None, lo=1, adv=True, mem=True, unit="tokens"),
        p("enable_memory_saver", "bool", "memory", "Memory saver",
          "Allow the engine to release and resume its memory (reinforcement-learning and co-location setups).",
          False, adv=True, lvl="expert"),
        p("max_mamba_cache_size", "int", "memory", "Mamba state slots",
          "Maximum size of the Mamba (linear-attention) state cache of hybrid models; it also limits how many "
          "requests can run at once.", None, lo=1, adv=True, mem=True),
        p("mamba_ssm_dtype", "enum", "memory", "Mamba SSM state dtype",
          "Data type of the SSM state in the Mamba cache. Unset reads it from the model config.", None,
          choices=["float32", "bfloat16", "float16"], adv=True, mem=True),
        p("mamba_radix_cache_strategy", "enum", "memory", "Mamba radix cache strategy",
          "How Mamba states are cached for prefix reuse: auto, no_buffer, extra_buffer or extra_buffer_lazy.",
          "auto", choices=["auto", "no_buffer", "extra_buffer", "extra_buffer_lazy"], adv=True, mem=True),
        p("mamba_track_interval", "int", "memory", "Mamba track interval",
          "Interval, in tokens, at which the Mamba state is tracked during decode for the prefix cache.", 256, lo=1,
          adv=True, lvl="expert", unit="tokens"),
        p("enable_hierarchical_cache", "bool", "memory", "Hierarchical KV cache (HiCache)",
          "Spill KV blocks to host RAM or storage. Needs the radix cache.", False, adv=True, mem=True),
        p("hicache_ratio", "float", "memory", "HiCache host/device ratio",
          "Host KV pool size relative to the device pool. Unset: 2.0 in cache mode.", None, lo=1.0, adv=True,
          mem=True),
        p("hicache_size", "int", "memory", "HiCache host size",
          "Host pool size in GB; overrides hicache-ratio.", None, lo=1, adv=True, mem=True, unit="GB"),
        p("hicache_write_policy", "enum", "memory", "HiCache write policy",
          "How device KV is written to the host tier.", "write_through",
          choices=["write_back", "write_through", "write_through_selective"], adv=True, lvl="expert"),
        p("hicache_io_backend", "enum", "memory", "HiCache IO backend",
          "Kernel used to copy KV between CPU and GPU.", "kernel", choices=["direct", "kernel"], adv=True,
          lvl="expert"),
        p("hicache_mem_layout", "enum", "memory", "HiCache host layout",
          "Layout of the host memory pool.", "page_first",
          choices=["layer_first", "page_first", "page_first_direct", "page_first_kv_split", "page_head"],
          adv=True, lvl="expert"),
        p("hicache_storage_backend", "enum", "memory", "HiCache storage backend",
          "Optional third tier below host RAM.", None, choices=HICACHE_STORAGE, adv=True, lvl="expert"),
        # ---- context ----
        p("context_length", "int", "context", "Context length",
          "Maximum context (prompt plus output). Defaults to the model config. Longer than the model's native length "
          "needs SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 and rope scaling in json_model_override_args.",
          None, lo=128, mem=True, lvl="basic", unit="tokens"),
        p("allow_longer_context_len", "bool", "context", "Allow context beyond the model's length",
          "Sets SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 so a context_length above the derived length is accepted "
          "with a warning instead of an error. Pair it with a rope-scaling block in json_model_override_args; quality "
          "beyond the trained length is not guaranteed.", False, adv=True, env="SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"),
        p("chunked_prefill_size", "int", "context", "Chunked prefill size",
          "Tokens per prefill chunk; -1 disables chunked prefill. Smaller means less activation memory and smoother "
          "decode latency. Unset: chosen from the card's memory.", None, lo=-1, mem=True, unit="tokens"),
        p("max_prefill_tokens", "int", "context", "Max prefill tokens",
          "Maximum tokens in one prefill batch (the real bound is the larger of this and the context length).",
          16384, lo=1, adv=True, mem=True, lvl="expert", unit="tokens"),
        p("allow_auto_truncate", "bool", "context", "Auto-truncate long prompts",
          "Truncate over-long inputs instead of returning an error.", False, adv=True, lvl="expert"),
        p("json_model_override_args", "json", "context", "Model config overrides (JSON)",
          "JSON object that overrides config.json values, for example a rope-scaling block for YaRN.", None,
          adv=True),
        p("revision", "string", "context", "Model revision",
          "Branch, tag or commit of the model repository.", None, adv=True),
        p("trust_remote_code", "bool", "context", "Trust remote code",
          "Allow custom modelling code from the Hub. It executes downloaded code: only for reviewed repositories.",
          False, lvl="basic"),
        p("tokenizer_path", "string", "context", "Tokenizer path",
          "Tokenizer repository or path, if it differs from the model's.", None, adv=True),
        p("model_impl", "string", "context", "Model implementation",
          "auto, sglang or transformers. auto uses SGLang's own implementation and falls back to Transformers.",
          "auto", adv=True, lvl="expert"),
        p("enable_multimodal", "bool", "context", "Enable multimodal",
          "Enable the multimodal path for the served model (no effect on a text-only model).", None),
        p("language_model_only", "bool", "context", "Language model only",
          "Skip the multimodal encoder entirely: its weights are never loaded, which frees that memory for KV cache. "
          "Image and video requests are rejected.", False, mem=True),
        p("is_embedding", "bool", "context", "Embedding mode",
          "Serve a causal language model as an embedding model.", False, adv=True, lvl="expert"),
        p("media_url_max_file_size_mb", "int", "context", "Max remote media size",
          "Largest remote image, video or audio download one request may trigger; 0 disables the limit.", 64, lo=0,
          adv=True, unit="MiB"),
        p("default_chat_template_kwargs", "json", "context", "Default template kwargs (JSON)",
          'Default kwargs for the chat template on every request, for example {"enable_thinking": true}. A request\'s '
          "own chat_template_kwargs wins.", None),
        p("sampling_defaults", "enum", "context", "Sampling defaults source",
          "Where default sampling parameters come from: the model's generation_config.json (model) or SGLang's own "
          "(openai).", "model", choices=["openai", "model"], adv=True),
        # ---- quantization ----
        p("dtype", "enum", "quantization", "Weights/activation dtype",
          "Compute dtype. auto uses BF16 for BF16 models and FP16 for FP16 and FP32 models.",
          "auto", choices=["auto", "half", "float16", "bfloat16", "float", "float32"], mem=True),
        p("quantization", "enum", "quantization", "Quantization",
          "Weight quantization method. Pre-quantized checkpoints (NVFP4, AWQ, GPTQ) carry their own config, so leave "
          "it unset or give the matching value (modelopt_fp4 for a ModelOpt NVFP4 checkpoint). fp8 quantizes a BF16 "
          "model while loading.", None, choices=QUANT_CHOICES, mem=True, lvl="basic"),
        p("load_format", "enum", "quantization", "Weight load format",
          "Format of the weights to load; auto picks safetensors when present.",
          "auto", choices=LOAD_FORMATS, adv=True, lvl="expert"),
        p("fp8_gemm_backend", "enum", "quantization", "FP8 GEMM backend",
          "Runner for blockwise FP8 GEMM. cutlass is the SM120-tuned choice per the help; auto selects by hardware.",
          "auto", choices=FP8_GEMM, adv=True),
        p("fp4_gemm_backend", "enum", "quantization", "FP4 GEMM backend",
          "Runner for NVFP4 GEMM. auto picks flashinfer_cutlass on SM120 and marlin (weight-only) on SM80 to SM90.",
          "auto", choices=FP4_GEMM, adv=True),
        p("moe_runner_backend", "enum", "quantization", "MoE runner backend",
          "Kernel backend for MoE experts. Test before forcing one: auto and flashinfer_cutlass are the usual "
          "choices for NVFP4 MoE on sm_120.", "auto", choices=MOE_RUNNERS, adv=True),
        # ---- parallelism ----
        p("tp_size", "int", "parallelism", "Tensor parallel size",
          "GPUs per replica (--tp-size). Must not exceed the GPUs selected.", 1, lo=1, mem=True, lvl="basic",
          unit="GPUs"),
        p("dp_size", "int", "parallelism", "Data parallel size",
          "Number of replicas (or attention data-parallel groups when enable-dp-attention is on).", 1, lo=1, mem=True,
          unit="GPUs"),
        p("ep_size", "int", "parallelism", "Expert parallel size",
          "Expert parallelism for MoE models; must divide tp-size.", 1, lo=1, adv=True, mem=True, unit="GPUs"),
        p("pp_size", "int", "parallelism", "Pipeline parallel size",
          "Number of pipeline stages.", 1, lo=1, adv=True, mem=True, unit="GPUs"),
        p("enable_dp_attention", "bool", "parallelism", "DP attention",
          "Data-parallel attention with tensor-parallel FFN layers. dp-size must equal tp-size. Supported for "
          "DeepSeek-style and Qwen MoE models.", False, adv=True, mem=True),
        p("moe_a2a_backend", "enum", "parallelism", "MoE all-to-all backend",
          "All-to-all backend for expert parallelism.", "none", choices=MOE_A2A, adv=True, lvl="expert"),
        p("enable_eplb", "bool", "parallelism", "Expert-parallel load balancer",
          "Enable the expert-parallel load balancer (needs expert parallelism).", False, adv=True, lvl="expert"),
        p("ep_num_redundant_experts", "int", "parallelism", "Redundant experts",
          "Extra redundant experts for the load balancer.", 0, lo=0, adv=True, mem=True, lvl="expert"),
        p("disable_custom_all_reduce", "bool", "parallelism", "Disable custom all-reduce",
          "Fall back to NCCL all-reduce (try this if tensor parallelism hangs on a PCIe-only machine).", False,
          adv=True),
        p("nnodes", "int", "parallelism", "Node count",
          "Number of machines. This console runs single-node engines.", 1, lo=1, adv=True, lvl="expert"),
        # ---- scheduling ----
        p("max_running_requests", "int", "scheduling", "Max running requests",
          "Cap on requests running at once; it also sizes the CUDA-graph batch and the Mamba caches. Unset: derived "
          "from the card.", None, lo=1, mem=True, lvl="basic", unit="requests"),
        p("max_queued_requests", "int", "scheduling", "Max queued requests",
          "Reject new requests once this many are queued.", None, lo=1, adv=True, unit="requests"),
        p("schedule_policy", "enum", "scheduling", "Schedule policy",
          "Request ordering. lpm (longest prefix match) favours cache hits; fcfs is first come, first served.",
          "fcfs", choices=SCHEDULE_POLICIES),
        p("schedule_conservativeness", "float", "scheduling", "Schedule conservativeness",
          "How conservative admission is: larger means more careful. Raise it if requests are retracted often.",
          1.0, lo=0.0),
        p("enable_mixed_chunk", "bool", "scheduling", "Mixed chunk",
          "Mix prefill and decode tokens in one batch (needs chunked prefill).", False),
        p("enable_priority_scheduling", "bool", "scheduling", "Priority scheduling",
          "Schedule requests with higher priority numbers first.", False, adv=True),
        p("disable_overlap_schedule", "bool", "scheduling", "Disable overlap scheduler",
          "Turn off the overlap of CPU scheduling with GPU work (debugging).", False, adv=True, lvl="expert"),
        p("stream_interval", "int", "scheduling", "Stream interval",
          "Tokens per streamed chunk: lower is smoother, higher gives more throughput.", 1, lo=1, adv=True,
          unit="tokens"),
        p("sleep_on_idle", "bool", "scheduling", "Sleep on idle",
          "Reduce CPU usage while the engine is idle.", False, adv=True),
        p("watchdog_timeout", "int", "scheduling", "Watchdog timeout",
          "Crash if one forward batch takes longer than this; raise it for very long prefills.", 300, lo=1, adv=True,
          unit="s"),
        # ---- performance ----
        p("attention_backend", "enum", "performance", "Attention backend",
          "Attention kernels. flashinfer, triton and trtllm_mha run on sm_120; fa3 is built for Hopper and the "
          "*_mla backends for DeepSeek-style models.", None, choices=ATTENTION_BACKENDS),
        p("prefill_attention_backend", "enum", "performance", "Prefill attention backend",
          "Override the attention backend for prefill only.", None, choices=ATTENTION_BACKENDS, adv=True,
          lvl="expert"),
        p("decode_attention_backend", "enum", "performance", "Decode attention backend",
          "Override the attention backend for decode only.", None, choices=ATTENTION_BACKENDS, adv=True,
          lvl="expert"),
        p("sampling_backend", "enum", "performance", "Sampling backend",
          "Kernels for the sampling layers.", None, choices=SAMPLING_BACKENDS, adv=True, lvl="expert"),
        p("grammar_backend", "enum", "performance", "Grammar backend",
          "Backend for constrained (JSON or regex) decoding.", None, choices=GRAMMAR_BACKENDS, adv=True),
        p("disable_radix_cache", "bool", "performance", "Disable radix cache",
          "Turn off RadixAttention prefix caching (it hurts agent and multi-turn workloads).", False),
        p("radix_eviction_policy", "enum", "performance", "Radix eviction policy",
          "Which cached prefixes are dropped first when the cache is full.", "lru",
          choices=["lru", "lfu", "slru", "priority"], adv=True, lvl="expert"),
        p("enable_torch_compile", "bool", "performance", "torch.compile",
          "Experimental: long warm-up, helps small-batch decode.", False, adv=True),
        p("torch_compile_max_bs", "int", "performance", "torch.compile max batch",
          "Largest batch size compiled with torch.compile.", 32, lo=1, adv=True, lvl="expert", unit="requests"),
        p("cuda_graph_max_bs_decode", "int", "performance", "CUDA graph max batch (decode)",
          "Largest decode batch captured as a CUDA graph; bigger means more graph memory and capture time. Unset: "
          "derived from the card.", None, lo=1, mem=True, unit="requests"),
        p("cuda_graph_backend_decode", "enum", "performance", "CUDA graph backend (decode)",
          "full, breakable, tc_piecewise or disabled. Replaces the deprecated --disable-cuda-graph.",
          None, choices=CUDA_GRAPH_BACKENDS, adv=True, mem=True, lvl="expert"),
        p("cuda_graph_backend_prefill", "enum", "performance", "CUDA graph backend (prefill)",
          "Prefill-phase graph backend; disabled saves memory and capture time.",
          None, choices=CUDA_GRAPH_BACKENDS, adv=True, mem=True, lvl="expert"),
        p("cuda_graph_max_bs_prefill", "int", "performance", "CUDA graph max tokens (prefill)",
          "Largest prefill size captured as a CUDA graph.", None, lo=1, adv=True, mem=True, lvl="expert",
          unit="tokens"),
        p("enable_tokenizer_batch_encode", "bool", "performance", "Batch tokenization",
          "Tokenize concurrent text inputs in one batch (not for image inputs).", False, adv=True, lvl="expert"),
        p("enable_deterministic_inference", "bool", "performance", "Deterministic inference",
          "Batch-invariant kernels for reproducible outputs; slower.", False, adv=True),
        p("skip_server_warmup", "bool", "performance", "Skip warmup",
          "Skip the warm-up request at start-up.", False, adv=True, lvl="expert"),
        p("disable_flashinfer_autotune", "bool", "performance", "Disable FlashInfer autotune",
          "Skip FlashInfer's autotuning pass at start-up (faster start, possibly slower kernels).", False, adv=True),
        p("mamba_backend", "enum", "performance", "Mamba SSM backend",
          "Kernel backend for Mamba state operations: triton (the default) or flashinfer.", "triton",
          choices=["triton", "flashinfer"], adv=True),
        p("linear_attn_backend", "enum", "performance", "Linear-attention backend",
          "Default kernel backend for linear attention (gated delta net and KDA layers).", "triton",
          choices=LINEAR_ATTN, adv=True),
        p("linear_attn_decode_backend", "enum", "performance", "Linear-attention decode backend",
          "Override the linear-attention backend for decode. Unset follows linear-attn-backend.", None,
          choices=LINEAR_ATTN, adv=True),
        p("linear_attn_prefill_backend", "enum", "performance", "Linear-attention prefill backend",
          "Override the linear-attention backend for prefill. Unset follows linear-attn-backend.", None,
          choices=LINEAR_ATTN, adv=True),
        # ---- tools & reasoning ----
        p("tool_call_parser", "enum", "tools_reasoning", "Tool-call parser",
          "Needed by agentic clients: without it tool calls stay raw text and tool_calls is empty. It must match the "
          "model's dialect (Qwen3.x and Coder-style XML needs qwen3_coder, not qwen).", None, choices=TOOL_PARSERS,
          lvl="basic"),
        p("reasoning_parser", "enum", "tools_reasoning", "Reasoning parser",
          "Splits the model's <think> output into reasoning_content.", None, choices=REASONING_PARSERS, lvl="basic"),
        p("chat_template", "string", "tools_reasoning", "Chat template",
          "Built-in template name or a template file path inside the container.", None, adv=True),
        p("served_model_name", "string", "tools_reasoning", "Served model name",
          "Name returned by /v1/models. Overrides the name the console provides.", None, lvl="basic"),
        p("preferred_sampling_params", "json", "tools_reasoning", "Preferred sampling params",
          "JSON sampling defaults advertised by /get_model_info.", None, adv=True, lvl="expert"),
        p("enable_cache_report", "bool", "tools_reasoning", "Report cached tokens",
          "Return the number of cached prompt tokens in usage.prompt_tokens_details.", False, adv=True),
        # ---- speculative ----
        p("speculative_algorithm", "enum", "speculative", "Speculative algorithm",
          "EAGLE, EAGLE3 and STANDALONE need a draft model; NEXTN uses the model's built-in MTP head; NGRAM needs no "
          "draft model; DFLASH and DSPARK are newer drafters.", None, choices=SPEC_ALGOS, mem=True),
        p("speculative_draft_model_path", "string", "speculative", "Draft model",
          "Hugging Face repository or path of the draft model.", None, mem=True),
        p("speculative_num_steps", "int", "speculative", "Draft steps",
          "Steps sampled from the draft model.", None, lo=1, mem=True),
        p("speculative_eagle_topk", "int", "speculative", "EAGLE top-k",
          "Tokens sampled per draft step (1 means chain drafting).", None, lo=1),
        p("speculative_num_draft_tokens", "int", "speculative", "Draft tokens",
          "Tokens verified per step. With top-k 1, SGLang forces this to steps plus one.", None, lo=1,
          unit="tokens"),
        p("speculative_accept_threshold_single", "float", "speculative", "Accept threshold",
          "Accept a draft token if its probability in the target model exceeds this.", 1.0, lo=0.0, hi=1.0, adv=True,
          lvl="expert"),
        p("speculative_draft_model_quantization", "enum", "speculative", "Draft quantization",
          "Quantization of the draft model (unquant keeps it unquantized).", None, choices=QUANT_CHOICES, adv=True),
        # ---- LoRA ----
        p("enable_lora", "bool", "advanced", "Enable LoRA",
          "Enable LoRA serving (implied by lora-paths).", None),
        p("lora_paths", "string_list", "advanced", "LoRA adapters",
          "Adapters as <path> or <name>=<path>.", None),
        p("max_loras_per_batch", "int", "advanced", "Max LoRAs per batch",
          "Adapters (including base-only requests) in one running batch.", 8, lo=1, adv=True, mem=True),
        p("max_lora_rank", "int", "advanced", "Max LoRA rank",
          "Largest adapter rank. Inferred from the adapters when unset.", None, lo=1, adv=True, mem=True),
        p("lora_backend", "enum", "advanced", "LoRA backend",
          "Kernel backend for multi-LoRA serving.", "csgmv", choices=LORA_BACKENDS, adv=True, lvl="expert"),
        # ---- network / API ----
        p("enable_metrics", "bool", "network", "Prometheus metrics",
          "Exposes /metrics. The console always turns this on; setting it to false has no effect.", True, adv=True,
          lvl="expert"),
        p("enable_request_time_stats_logging", "bool", "network", "Request time stats logging",
          "Log per-request timing statistics.", False, adv=True, lvl="expert"),
        # ---- advanced / logging ----
        p("random_seed", "int", "advanced", "Random seed", "Seed for sampling reproducibility.", None, adv=True,
          lvl="expert"),
        p("log_level", "enum", "advanced", "Log level", "Logging level of all loggers.", "info",
          choices=["debug", "info", "warning", "error", "critical"], adv=True, lvl="expert"),
        p("decode_log_interval", "int", "advanced", "Decode log interval",
          "Decode iterations between log and metric lines.", 40, lo=1, adv=True, lvl="expert"),
        p("log_requests", "bool", "advanced", "Log requests",
          "Log metadata, inputs and outputs of every request: this puts prompts into the logs.", False, adv=True,
          lvl="expert"),
        p("enable_trace", "bool", "advanced", "OpenTelemetry tracing",
          "Emit OpenTelemetry traces (needs otlp-traces-endpoint).", False, adv=True, lvl="expert"),
        p("otlp_traces_endpoint", "string", "advanced", "OTLP endpoint",
          "host:port of the collector when tracing is enabled.", "localhost:4317", adv=True, lvl="expert"),
    ]


CATALOG_VERSION_NOTE = f"{DEFAULT_IMAGE} (parser dump and --help read from the local image)"
