//! Port of backend/src/engine_console/adapters/sglang_params.py: the SGLang parameter
//! catalog (data only).
//!
//! Pinned engine: `lmsysorg/sglang:v0.5.20-cu130`. Flags, defaults and choices were read inside the pinned image by
//! building the server's own argument parser; a condensed copy is committed as a fixture (`tests/data/sglang/`) and the
//! tests check this catalog against it offline. Choice lists for hardware this console does not run are shortened and
//! [`OMITTED_CHOICES`] records exactly what was left out. Provenance in detail: the Python module's docstring.
//!
//! Pseudo-flag convention, handled in `SglangAdapter::build_launch_ctx`, never emitted as argv: `env:NAME` sets the
//! container environment variable NAME. Deliberately NOT in the catalog (owned by build_launch): --model-path, --host,
//! --port, --enable-metrics' forced value.

use serde_json::{Value, json};

use super::base::ParamSpec;

pub const SGLANG_VERSION: &str = "0.5.20";
pub const DEFAULT_IMAGE: &str = "lmsysorg/sglang:v0.5.20-cu130";
pub const DOCS: &str = "https://docs.sglang.io/docs/advanced_features/server_arguments";
pub const ENV_DOCS: &str = "https://docs.sglang.io/docs/references/environment_variables";

/// Catalog flags that the v0.5.14 image's parser does not have.
pub const NOT_IN_V0_5_14: &[&str] = &[
    "--language-model-only",
    "--media-url-max-file-size-mb",
    "--default-chat-template-kwargs",
];

// --- choices read from the pinned image -------------------------------------------------------------------------------
pub const QUANT_CHOICES: &[&str] = &[
    "awq",
    "fp8",
    "mxfp8",
    "gptq",
    "gptq_marlin",
    "awq_marlin",
    "bitsandbytes",
    "modelopt",
    "modelopt_fp8",
    "modelopt_fp4",
    "nvfp4_online",
    "modelopt_mixed",
    "w8a8_int8",
    "w8a8_fp8",
    "moe_wna16",
    "w4afp8",
    "mxfp4",
    "auto-round",
    "auto-round-int8",
    "compressed-tensors",
    "mxfp_w4a8",
    "unquant",
    "humming",
];
pub const ATTENTION_BACKENDS: &[&str] = &[
    "triton",
    "torch_native",
    "flex_attention",
    "dsa",
    "nsa",
    "qsa",
    "dsv4",
    "compressed",
    "fa3",
    "fa4",
    "flashinfer",
    "flashmla",
    "trtllm_mla",
    "cutedsl_mla",
    "tokenspeed_mla",
    "trtllm_mha",
    "hpc_ops",
    "minicpm_flashattn",
    "minicpm_flashinfer",
];
pub const LOAD_FORMATS: &[&str] = &[
    "auto",
    "pt",
    "safetensors",
    "npcache",
    "dummy",
    "sharded_state",
    "presharded",
    "gguf",
    "expert_pack",
    "bitsandbytes",
    "mistral",
    "layered",
    "flash_rl",
    "remote",
    "remote_instance",
    "fastsafetensors",
    "private",
    "runai_streamer",
];
pub const MOE_RUNNERS: &[&str] = &[
    "auto",
    "deep_gemm",
    "triton",
    "triton_kernel",
    "flashinfer_trtllm",
    "experimental_sgl_trtllm",
    "flashinfer_trtllm_routed",
    "flashinfer_cutlass",
    "flashinfer_mxfp4",
    "flashinfer_cutedsl",
    "flashinfer_megamoe",
    "cutlass",
    "marlin",
    "humming",
    "experimental_sgl_marlin",
    "hpc_ops",
    "megamoe",
];
pub const FP8_GEMM: &[&str] = &[
    "auto",
    "deep_gemm",
    "flashinfer_trtllm",
    "flashinfer_cutlass",
    "flashinfer_deepgemm",
    "flashinfer_cutedsl",
    "cutlass",
    "triton",
];
pub const FP4_GEMM: &[&str] = &[
    "auto",
    "flashinfer_cudnn",
    "flashinfer_cutedsl",
    "flashinfer_cutlass",
    "flashinfer_trtllm",
    "marlin",
];
pub const KV_DTYPES: &[&str] = &[
    "auto",
    "fp8_e5m2",
    "fp8_e4m3",
    "mxfp8",
    "bf16",
    "bfloat16",
    "nvfp4",
    "fp4_mx_block16",
    "fp4_e2m1",
];
pub const MOE_A2A: &[&str] = &[
    "none",
    "deepep",
    "mooncake",
    "nixl",
    "flashinfer",
    "megamoe",
    "deepep_v2",
    "pplx",
    "flashinfer_megamoe",
];
pub const TOOL_PARSERS: &[&str] = &[
    "auto",
    "apertus2509",
    "cohere_command4",
    "deepseekv3",
    "deepseekv31",
    "deepseekv32",
    "deepseekv4",
    "dots",
    "glm",
    "glm45",
    "glm47",
    "gpt-oss",
    "k2_horizon",
    "kimi_k2",
    "kimi_k3",
    "lfm2",
    "ling3",
    "llama3",
    "mimo",
    "minicpm5",
    "mistral",
    "muse",
    "poolside_v1",
    "pythonic",
    "qwen",
    "qwen25",
    "qwen3_coder",
    "spark25",
    "step3",
    "step3p5",
    "minimax-m2",
    "minimax-m3",
    "nanbeige",
    "trinity",
    "interns1",
    "hermes",
    "hunyuan",
    "gigachat3",
    "gemma4",
    "inkling",
];
pub const REASONING_PARSERS: &[&str] = &[
    "auto",
    "apertus2509",
    "deepseek-r1",
    "deepseek-v3",
    "deepseek-v4",
    "dots",
    "glm45",
    "ling3",
    "hunyuan",
    "gpt-oss",
    "k2_horizon",
    "kimi",
    "kimi_k2",
    "kimi_k3",
    "mimo",
    "muse",
    "poolside_v1",
    "qwen3",
    "qwen3-thinking",
    "minimax",
    "minimax-append-think",
    "minimax-m3",
    "nanbeige",
    "step3",
    "step3p5",
    "mistral",
    "nemotron_3",
    "granite_thinking_parser",
    "interns1",
    "gemma4",
    "inkling",
    "cohere_command4",
];
pub const SCHEDULE_POLICIES: &[&str] = &[
    "lpm",
    "random",
    "fcfs",
    "dfs-weight",
    "lof",
    "priority",
    "routing-key",
    "hrrn",
];
pub const HICACHE_STORAGE: &[&str] = &[
    "file", "mooncake", "hf3fs", "nixl", "aibrix", "dynamic", "eic", "simm", "shm",
];
pub const LINEAR_ATTN: &[&str] = &[
    "triton",
    "cutedsl",
    "flashinfer",
    "flashkda",
    "nvidia_kda",
    "ptx_kda",
    "helion",
];
pub const LORA_BACKENDS: &[&str] = &["triton", "csgmv", "torch_native"];
pub const SAMPLING_BACKENDS: &[&str] = &["flashinfer", "pytorch"];
pub const GRAMMAR_BACKENDS: &[&str] = &["xgrammar", "outlines", "llguidance", "none"];
pub const CUDA_GRAPH_BACKENDS: &[&str] = &["full", "breakable", "tc_piecewise", "disabled"];
/// `--speculative-algorithm` prints no choices; the help text names these builtins (UNO needs a draft LoRA flag that
/// this catalog does not offer).
pub const SPEC_ALGOS: &[&str] = &[
    "EAGLE",
    "EAGLE3",
    "NEXTN",
    "STANDALONE",
    "NGRAM",
    "DFLASH",
    "DSPARK",
];

const QUANT_OMITTED: &[&str] = &[
    "petit_nvfp4",
    "modelslim",
    "quark",
    "quark_int4fp8_moe",
    "quark_mxfp4",
    "mlx_q4",
    "mlx_q8",
    "gguf",
];
const ATTN_OMITTED: &[&str] = &["aiter", "wave", "intel_amx", "ascend", "intel_xpu"];

/// flag -> choices the image lists that the catalog leaves out (non-NVIDIA hardware, Apple MLX, GGUF). Frozensets in
/// Python: only membership is used.
pub const OMITTED_CHOICES: &[(&str, &[&str])] = &[
    ("--quantization", QUANT_OMITTED),
    ("--speculative-draft-model-quantization", QUANT_OMITTED),
    ("--attention-backend", ATTN_OMITTED),
    ("--prefill-attention-backend", ATTN_OMITTED),
    ("--decode-attention-backend", ATTN_OMITTED),
    ("--moe-runner-backend", &["aiter", "intel_xpu"]),
    ("--fp8-gemm-backend", &["aiter"]),
    ("--moe-a2a-backend", &["mori", "ascend_fuseep", "ascend_tp"]),
    (
        "--hicache-storage-backend",
        &["sim", "npu_memcache", "mori"],
    ),
    ("--hicache-io-backend", &["kernel_ascend"]),
    ("--linear-attn-backend", &["intel_xpu"]),
    ("--linear-attn-decode-backend", &["intel_xpu"]),
    ("--linear-attn-prefill-backend", &["intel_xpu"]),
    ("--lora-backend", &["ascend"]),
    ("--sampling-backend", &["ascend"]),
];

/// docs section id -> catalog keys (heading ids of the server-arguments page), in Python's order.
pub const SECTIONS: &[(&str, &[&str])] = &[
    (
        "api-related",
        &[
            "default_chat_template_kwargs",
            "sampling_defaults",
            "tool_call_parser",
            "reasoning_parser",
            "chat_template",
            "served_model_name",
            "enable_cache_report",
        ],
    ),
    ("data-parallelism", &["dp_size"]),
    ("encode-prefill-disaggregation", &["language_model_only"]),
    ("for-multi-modal", &["media_url_max_file_size_mb"]),
    (
        "hierarchical-cache",
        &[
            "enable_hierarchical_cache",
            "hicache_ratio",
            "hicache_size",
            "hicache_write_policy",
            "hicache_io_backend",
            "hicache_mem_layout",
            "hicache_storage_backend",
        ],
    ),
    ("http-server", &["skip_server_warmup"]),
    (
        "kernel-backends-attention-sampling-grammar-gemm",
        &[
            "fp8_gemm_backend",
            "fp4_gemm_backend",
            "attention_backend",
            "prefill_attention_backend",
            "decode_attention_backend",
            "sampling_backend",
            "grammar_backend",
            "disable_flashinfer_autotune",
        ],
    ),
    (
        "logging",
        &[
            "enable_metrics",
            "enable_request_time_stats_logging",
            "log_level",
            "decode_log_interval",
            "log_requests",
            "enable_trace",
            "otlp_traces_endpoint",
        ],
    ),
    (
        "lora",
        &[
            "enable_lora",
            "lora_paths",
            "max_loras_per_batch",
            "max_lora_rank",
            "lora_backend",
        ],
    ),
    (
        "mamba-cache",
        &[
            "max_mamba_cache_size",
            "mamba_ssm_dtype",
            "mamba_radix_cache_strategy",
            "mamba_track_interval",
            "mamba_backend",
            "linear_attn_backend",
            "linear_attn_decode_backend",
            "linear_attn_prefill_backend",
        ],
    ),
    (
        "memory-and-scheduling",
        &[
            "mem_fraction_static",
            "max_total_tokens",
            "swa_full_tokens_ratio",
            "disable_hybrid_swa_memory",
            "page_size",
            "chunked_prefill_size",
            "max_prefill_tokens",
            "max_running_requests",
            "max_queued_requests",
            "schedule_policy",
            "schedule_conservativeness",
            "enable_priority_scheduling",
            "radix_eviction_policy",
        ],
    ),
    (
        "model-and-tokenizer",
        &[
            "context_length",
            "revision",
            "trust_remote_code",
            "tokenizer_path",
            "model_impl",
            "enable_multimodal",
            "is_embedding",
            "load_format",
        ],
    ),
    (
        "model-override-args",
        &["json_model_override_args", "preferred_sampling_params"],
    ),
    (
        "moe",
        &[
            "moe_runner_backend",
            "ep_size",
            "moe_a2a_backend",
            "enable_eplb",
            "ep_num_redundant_experts",
        ],
    ),
    ("multi-node-distributed-serving", &["nnodes"]),
    ("offloading", &["cpu_offload_gb"]),
    (
        "optimization/debug-options",
        &[
            "enable_memory_saver",
            "allow_auto_truncate",
            "enable_dp_attention",
            "disable_custom_all_reduce",
            "enable_mixed_chunk",
            "disable_overlap_schedule",
            "disable_radix_cache",
            "enable_torch_compile",
            "torch_compile_max_bs",
            "cuda_graph_max_bs_decode",
            "cuda_graph_backend_decode",
            "cuda_graph_backend_prefill",
            "cuda_graph_max_bs_prefill",
            "enable_tokenizer_batch_encode",
            "enable_deterministic_inference",
        ],
    ),
    (
        "quantization-and-data-type",
        &["kv_cache_dtype", "dtype", "quantization"],
    ),
    (
        "runtime-options",
        &[
            "tp_size",
            "pp_size",
            "stream_interval",
            "sleep_on_idle",
            "watchdog_timeout",
            "random_seed",
        ],
    ),
    (
        "speculative-decoding",
        &[
            "speculative_algorithm",
            "speculative_draft_model_path",
            "speculative_num_steps",
            "speculative_eagle_topk",
            "speculative_num_draft_tokens",
            "speculative_accept_threshold_single",
            "speculative_draft_model_quantization",
        ],
    ),
];

/// `SECTION_OF[key]`: the docs section of a catalog key.
pub fn section_of(key: &str) -> Option<&'static str> {
    SECTIONS
        .iter()
        .find(|(_, keys)| keys.contains(&key))
        .map(|(s, _)| *s)
}

/// The keyword arguments of Python's `_p` (defaults as there).
struct O {
    default: Value,
    choices: Option<&'static [&'static str]>,
    lo: Option<f64>,
    hi: Option<f64>,
    adv: bool,
    mem: bool,
    lvl: &'static str,
    unit: Option<&'static str>,
    restart: bool,
    env: Option<&'static str>,
}

impl O {
    const DEF: O = O {
        default: Value::Null,
        choices: None,
        lo: None,
        hi: None,
        adv: false,
        mem: false,
        lvl: "advanced",
        unit: None,
        restart: true,
        env: None,
    };
}

/// One catalog entry. `env` makes it a container environment variable (flag `env:NAME`) instead of a CLI flag.
#[allow(clippy::needless_pass_by_value)]
fn p(key: &str, typ: &str, group: &str, label: &str, help: &str, o: O) -> ParamSpec {
    let flag = o.env.map_or_else(
        || format!("--{}", key.replace('_', "-")),
        |e| format!("env:{e}"),
    );
    let docs = if o.env.is_some() {
        ENV_DOCS.to_string()
    } else {
        format!("{DOCS}#{}", section_of(key).unwrap_or(""))
    };
    ParamSpec {
        key: key.into(),
        flag,
        label: label.into(),
        help: help.into(),
        type_: typ.into(),
        group: group.into(),
        default: o.default,
        choices: o
            .choices
            .map(|c| c.iter().map(|s| (*s).to_string()).collect()),
        min: o.lo,
        max: o.hi,
        advanced: o.adv,
        affects_memory: o.mem,
        requires_restart: o.restart,
        bool_style: "flag".into(),
        docs_url: Some(docs),
        explain: None,
        level: Some(o.lvl.into()),
        unit: o.unit.map(str::to_string),
    }
}

/// The catalog, in Python's order.
#[allow(clippy::too_many_lines)]
pub fn build_catalog() -> Vec<ParamSpec> {
    vec![
        p(
            "mem_fraction_static",
            "float",
            "memory",
            "Static memory fraction",
            "Share of the GPU memory free at start-up that weights and the KV cache pool may use; the rest is headroom for activations, CUDA graphs and kernel scratch. Unset: SGLang derives it from the card.",
            O {
                lo: Some(0.1),
                hi: Some(0.99),
                mem: true,
                lvl: "basic",
                ..O::DEF
            },
        ),
        p(
            "max_total_tokens",
            "int",
            "memory",
            "Max total tokens (KV pool)",
            "Hard cap on the tokens in the KV pool; normally derived from mem-fraction-static. For development and debugging.",
            O {
                lo: Some(1.0),
                adv: true,
                mem: true,
                lvl: "expert",
                unit: Some("tokens"),
                ..O::DEF
            },
        ),
        p(
            "kv_cache_dtype",
            "enum",
            "memory",
            "KV cache dtype",
            "KV storage type. fp8_e4m3 halves KV memory against bf16; the nvfp4 and fp4 types need CUDA 12.8 or newer and are less proven.",
            O {
                default: json!("auto"),
                choices: Some(KV_DTYPES),
                mem: true,
                lvl: "basic",
                ..O::DEF
            },
        ),
        p(
            "cpu_offload_gb",
            "int",
            "memory",
            "CPU offload",
            "GB of host RAM used to offload weights. Slow over PCIe; a last resort.",
            O {
                default: json!(0),
                lo: Some(0.0),
                adv: true,
                mem: true,
                unit: Some("GB"),
                ..O::DEF
            },
        ),
        p(
            "swa_full_tokens_ratio",
            "float",
            "memory",
            "SWA/full KV token ratio",
            "Sliding-window-layer KV tokens as a ratio of full-attention KV tokens (hybrid sliding-window models).",
            O {
                lo: Some(0.0),
                hi: Some(1.0),
                adv: true,
                mem: true,
                lvl: "expert",
                ..O::DEF
            },
        ),
        p(
            "disable_hybrid_swa_memory",
            "bool",
            "memory",
            "Disable hybrid SWA pool",
            "Disable the hybrid sliding-window KV pool.",
            O {
                default: json!(false),
                adv: true,
                mem: true,
                lvl: "expert",
                ..O::DEF
            },
        ),
        p(
            "page_size",
            "int",
            "memory",
            "KV page size",
            "Tokens per KV page. Some backends need a fixed size (FlashMLA and the TRT-LLM ones use 64).",
            O {
                lo: Some(1.0),
                adv: true,
                mem: true,
                unit: Some("tokens"),
                ..O::DEF
            },
        ),
        p(
            "enable_memory_saver",
            "bool",
            "memory",
            "Memory saver",
            "Allow the engine to release and resume its memory (reinforcement-learning and co-location setups).",
            O {
                default: json!(false),
                adv: true,
                lvl: "expert",
                ..O::DEF
            },
        ),
        p(
            "max_mamba_cache_size",
            "int",
            "memory",
            "Mamba state slots",
            "Maximum size of the Mamba (linear-attention) state cache of hybrid models; it also limits how many requests can run at once.",
            O {
                lo: Some(1.0),
                adv: true,
                mem: true,
                ..O::DEF
            },
        ),
        p(
            "mamba_ssm_dtype",
            "enum",
            "memory",
            "Mamba SSM state dtype",
            "Data type of the SSM state in the Mamba cache. Unset reads it from the model config.",
            O {
                choices: Some(&["float32", "bfloat16", "float16"]),
                adv: true,
                mem: true,
                ..O::DEF
            },
        ),
        p(
            "mamba_radix_cache_strategy",
            "enum",
            "memory",
            "Mamba radix cache strategy",
            "How Mamba states are cached for prefix reuse: auto, no_buffer, extra_buffer or extra_buffer_lazy.",
            O {
                default: json!("auto"),
                choices: Some(&["auto", "no_buffer", "extra_buffer", "extra_buffer_lazy"]),
                adv: true,
                mem: true,
                ..O::DEF
            },
        ),
        p(
            "mamba_track_interval",
            "int",
            "memory",
            "Mamba track interval",
            "Interval, in tokens, at which the Mamba state is tracked during decode for the prefix cache.",
            O {
                default: json!(256),
                lo: Some(1.0),
                adv: true,
                lvl: "expert",
                unit: Some("tokens"),
                ..O::DEF
            },
        ),
        p(
            "enable_hierarchical_cache",
            "bool",
            "memory",
            "Hierarchical KV cache (HiCache)",
            "Spill KV blocks to host RAM or storage. Needs the radix cache.",
            O {
                default: json!(false),
                adv: true,
                mem: true,
                ..O::DEF
            },
        ),
        p(
            "hicache_ratio",
            "float",
            "memory",
            "HiCache host/device ratio",
            "Host KV pool size relative to the device pool. Unset: 2.0 in cache mode.",
            O {
                lo: Some(1.0),
                adv: true,
                mem: true,
                ..O::DEF
            },
        ),
        p(
            "hicache_size",
            "int",
            "memory",
            "HiCache host size",
            "Host pool size in GB; overrides hicache-ratio.",
            O {
                lo: Some(1.0),
                adv: true,
                mem: true,
                unit: Some("GB"),
                ..O::DEF
            },
        ),
        p(
            "hicache_write_policy",
            "enum",
            "memory",
            "HiCache write policy",
            "How device KV is written to the host tier.",
            O {
                default: json!("write_through"),
                choices: Some(&["write_back", "write_through", "write_through_selective"]),
                adv: true,
                lvl: "expert",
                ..O::DEF
            },
        ),
        p(
            "hicache_io_backend",
            "enum",
            "memory",
            "HiCache IO backend",
            "Kernel used to copy KV between CPU and GPU.",
            O {
                default: json!("kernel"),
                choices: Some(&["direct", "kernel"]),
                adv: true,
                lvl: "expert",
                ..O::DEF
            },
        ),
        p(
            "hicache_mem_layout",
            "enum",
            "memory",
            "HiCache host layout",
            "Layout of the host memory pool.",
            O {
                default: json!("page_first"),
                choices: Some(&[
                    "layer_first",
                    "page_first",
                    "page_first_direct",
                    "page_first_kv_split",
                    "page_head",
                ]),
                adv: true,
                lvl: "expert",
                ..O::DEF
            },
        ),
        p(
            "hicache_storage_backend",
            "enum",
            "memory",
            "HiCache storage backend",
            "Optional third tier below host RAM.",
            O {
                choices: Some(HICACHE_STORAGE),
                adv: true,
                lvl: "expert",
                ..O::DEF
            },
        ),
        p(
            "context_length",
            "int",
            "context",
            "Context length",
            "Maximum context (prompt plus output). Defaults to the model config. Longer than the model's native length needs SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 and rope scaling in json_model_override_args.",
            O {
                lo: Some(128.0),
                mem: true,
                lvl: "basic",
                unit: Some("tokens"),
                ..O::DEF
            },
        ),
        p(
            "allow_longer_context_len",
            "bool",
            "context",
            "Allow context beyond the model's length",
            "Sets SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 so a context_length above the derived length is accepted with a warning instead of an error. Pair it with a rope-scaling block in json_model_override_args; quality beyond the trained length is not guaranteed.",
            O {
                env: Some("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"),
                default: json!(false),
                adv: true,
                ..O::DEF
            },
        ),
        p(
            "chunked_prefill_size",
            "int",
            "context",
            "Chunked prefill size",
            "Tokens per prefill chunk; -1 disables chunked prefill. Smaller means less activation memory and smoother decode latency. Unset: chosen from the card's memory.",
            O {
                lo: Some(-1.0),
                mem: true,
                unit: Some("tokens"),
                ..O::DEF
            },
        ),
        p(
            "max_prefill_tokens",
            "int",
            "context",
            "Max prefill tokens",
            "Maximum tokens in one prefill batch (the real bound is the larger of this and the context length).",
            O {
                default: json!(16384),
                lo: Some(1.0),
                adv: true,
                mem: true,
                lvl: "expert",
                unit: Some("tokens"),
                ..O::DEF
            },
        ),
        p(
            "allow_auto_truncate",
            "bool",
            "context",
            "Auto-truncate long prompts",
            "Truncate over-long inputs instead of returning an error.",
            O {
                default: json!(false),
                adv: true,
                lvl: "expert",
                ..O::DEF
            },
        ),
        p(
            "json_model_override_args",
            "json",
            "context",
            "Model config overrides (JSON)",
            "JSON object that overrides config.json values, for example a rope-scaling block for YaRN.",
            O {
                adv: true,
                ..O::DEF
            },
        ),
        p(
            "revision",
            "string",
            "context",
            "Model revision",
            "Branch, tag or commit of the model repository.",
            O {
                adv: true,
                ..O::DEF
            },
        ),
        p(
            "trust_remote_code",
            "bool",
            "context",
            "Trust remote code",
            "Allow custom modelling code from the Hub. It executes downloaded code: only for reviewed repositories.",
            O {
                default: json!(false),
                lvl: "basic",
                ..O::DEF
            },
        ),
        p(
            "tokenizer_path",
            "string",
            "context",
            "Tokenizer path",
            "Tokenizer repository or path, if it differs from the model's.",
            O {
                adv: true,
                ..O::DEF
            },
        ),
        p(
            "model_impl",
            "string",
            "context",
            "Model implementation",
            "auto, sglang or transformers. auto uses SGLang's own implementation and falls back to Transformers.",
            O {
                default: json!("auto"),
                adv: true,
                lvl: "expert",
                ..O::DEF
            },
        ),
        p(
            "enable_multimodal",
            "bool",
            "context",
            "Enable multimodal",
            "Enable the multimodal path for the served model (no effect on a text-only model).",
            O { ..O::DEF },
        ),
        p(
            "language_model_only",
            "bool",
            "context",
            "Language model only",
            "Skip the multimodal encoder entirely: its weights are never loaded, which frees that memory for KV cache. Image and video requests are rejected.",
            O {
                default: json!(false),
                mem: true,
                ..O::DEF
            },
        ),
        p(
            "is_embedding",
            "bool",
            "context",
            "Embedding mode",
            "Serve a causal language model as an embedding model.",
            O {
                default: json!(false),
                adv: true,
                lvl: "expert",
                ..O::DEF
            },
        ),
        p(
            "media_url_max_file_size_mb",
            "int",
            "context",
            "Max remote media size",
            "Largest remote image, video or audio download one request may trigger; 0 disables the limit.",
            O {
                default: json!(64),
                lo: Some(0.0),
                adv: true,
                unit: Some("MiB"),
                ..O::DEF
            },
        ),
        p(
            "default_chat_template_kwargs",
            "json",
            "context",
            "Default template kwargs (JSON)",
            "Default kwargs for the chat template on every request, for example {\"enable_thinking\": true}. A request's own chat_template_kwargs wins.",
            O { ..O::DEF },
        ),
        p(
            "sampling_defaults",
            "enum",
            "context",
            "Sampling defaults source",
            "Where default sampling parameters come from: the model's generation_config.json (model) or SGLang's own (openai).",
            O {
                default: json!("model"),
                choices: Some(&["openai", "model"]),
                adv: true,
                ..O::DEF
            },
        ),
        p(
            "dtype",
            "enum",
            "quantization",
            "Weights/activation dtype",
            "Compute dtype. auto uses BF16 for BF16 models and FP16 for FP16 and FP32 models.",
            O {
                default: json!("auto"),
                choices: Some(&["auto", "half", "float16", "bfloat16", "float", "float32"]),
                mem: true,
                ..O::DEF
            },
        ),
        p(
            "quantization",
            "enum",
            "quantization",
            "Quantization",
            "Weight quantization method. Pre-quantized checkpoints (NVFP4, AWQ, GPTQ) carry their own config, so leave it unset or give the matching value (modelopt_fp4 for a ModelOpt NVFP4 checkpoint). fp8 quantizes a BF16 model while loading.",
            O {
                choices: Some(QUANT_CHOICES),
                mem: true,
                lvl: "basic",
                ..O::DEF
            },
        ),
        p(
            "load_format",
            "enum",
            "quantization",
            "Weight load format",
            "Format of the weights to load; auto picks safetensors when present.",
            O {
                default: json!("auto"),
                choices: Some(LOAD_FORMATS),
                adv: true,
                lvl: "expert",
                ..O::DEF
            },
        ),
        p(
            "fp8_gemm_backend",
            "enum",
            "quantization",
            "FP8 GEMM backend",
            "Runner for blockwise FP8 GEMM. cutlass is the SM120-tuned choice per the help; auto selects by hardware.",
            O {
                default: json!("auto"),
                choices: Some(FP8_GEMM),
                adv: true,
                ..O::DEF
            },
        ),
        p(
            "fp4_gemm_backend",
            "enum",
            "quantization",
            "FP4 GEMM backend",
            "Runner for NVFP4 GEMM. auto picks flashinfer_cutlass on SM120 and marlin (weight-only) on SM80 to SM90.",
            O {
                default: json!("auto"),
                choices: Some(FP4_GEMM),
                adv: true,
                ..O::DEF
            },
        ),
        p(
            "moe_runner_backend",
            "enum",
            "quantization",
            "MoE runner backend",
            "Kernel backend for MoE experts. Test before forcing one: auto and flashinfer_cutlass are the usual choices for NVFP4 MoE on sm_120.",
            O {
                default: json!("auto"),
                choices: Some(MOE_RUNNERS),
                adv: true,
                ..O::DEF
            },
        ),
        p(
            "tp_size",
            "int",
            "parallelism",
            "Tensor parallel size",
            "GPUs per replica (--tp-size). Must not exceed the GPUs selected.",
            O {
                default: json!(1),
                lo: Some(1.0),
                mem: true,
                lvl: "basic",
                unit: Some("GPUs"),
                ..O::DEF
            },
        ),
        p(
            "dp_size",
            "int",
            "parallelism",
            "Data parallel size",
            "Number of replicas (or attention data-parallel groups when enable-dp-attention is on).",
            O {
                default: json!(1),
                lo: Some(1.0),
                mem: true,
                unit: Some("GPUs"),
                ..O::DEF
            },
        ),
        p(
            "ep_size",
            "int",
            "parallelism",
            "Expert parallel size",
            "Expert parallelism for MoE models; must divide tp-size.",
            O {
                default: json!(1),
                lo: Some(1.0),
                adv: true,
                mem: true,
                unit: Some("GPUs"),
                ..O::DEF
            },
        ),
        p(
            "pp_size",
            "int",
            "parallelism",
            "Pipeline parallel size",
            "Number of pipeline stages.",
            O {
                default: json!(1),
                lo: Some(1.0),
                adv: true,
                mem: true,
                unit: Some("GPUs"),
                ..O::DEF
            },
        ),
        p(
            "enable_dp_attention",
            "bool",
            "parallelism",
            "DP attention",
            "Data-parallel attention with tensor-parallel FFN layers. dp-size must equal tp-size. Supported for DeepSeek-style and Qwen MoE models.",
            O {
                default: json!(false),
                adv: true,
                mem: true,
                ..O::DEF
            },
        ),
        p(
            "moe_a2a_backend",
            "enum",
            "parallelism",
            "MoE all-to-all backend",
            "All-to-all backend for expert parallelism.",
            O {
                default: json!("none"),
                choices: Some(MOE_A2A),
                adv: true,
                lvl: "expert",
                ..O::DEF
            },
        ),
        p(
            "enable_eplb",
            "bool",
            "parallelism",
            "Expert-parallel load balancer",
            "Enable the expert-parallel load balancer (needs expert parallelism).",
            O {
                default: json!(false),
                adv: true,
                lvl: "expert",
                ..O::DEF
            },
        ),
        p(
            "ep_num_redundant_experts",
            "int",
            "parallelism",
            "Redundant experts",
            "Extra redundant experts for the load balancer.",
            O {
                default: json!(0),
                lo: Some(0.0),
                adv: true,
                mem: true,
                lvl: "expert",
                ..O::DEF
            },
        ),
        p(
            "disable_custom_all_reduce",
            "bool",
            "parallelism",
            "Disable custom all-reduce",
            "Fall back to NCCL all-reduce (try this if tensor parallelism hangs on a PCIe-only machine).",
            O {
                default: json!(false),
                adv: true,
                ..O::DEF
            },
        ),
        p(
            "nnodes",
            "int",
            "parallelism",
            "Node count",
            "Number of machines. This console runs single-node engines.",
            O {
                default: json!(1),
                lo: Some(1.0),
                adv: true,
                lvl: "expert",
                ..O::DEF
            },
        ),
        p(
            "max_running_requests",
            "int",
            "scheduling",
            "Max running requests",
            "Cap on requests running at once; it also sizes the CUDA-graph batch and the Mamba caches. Unset: derived from the card.",
            O {
                lo: Some(1.0),
                mem: true,
                lvl: "basic",
                unit: Some("requests"),
                ..O::DEF
            },
        ),
        p(
            "max_queued_requests",
            "int",
            "scheduling",
            "Max queued requests",
            "Reject new requests once this many are queued.",
            O {
                lo: Some(1.0),
                adv: true,
                unit: Some("requests"),
                ..O::DEF
            },
        ),
        p(
            "schedule_policy",
            "enum",
            "scheduling",
            "Schedule policy",
            "Request ordering. lpm (longest prefix match) favours cache hits; fcfs is first come, first served.",
            O {
                default: json!("fcfs"),
                choices: Some(SCHEDULE_POLICIES),
                ..O::DEF
            },
        ),
        p(
            "schedule_conservativeness",
            "float",
            "scheduling",
            "Schedule conservativeness",
            "How conservative admission is: larger means more careful. Raise it if requests are retracted often.",
            O {
                default: json!(1.0),
                lo: Some(0.0),
                ..O::DEF
            },
        ),
        p(
            "enable_mixed_chunk",
            "bool",
            "scheduling",
            "Mixed chunk",
            "Mix prefill and decode tokens in one batch (needs chunked prefill).",
            O {
                default: json!(false),
                ..O::DEF
            },
        ),
        p(
            "enable_priority_scheduling",
            "bool",
            "scheduling",
            "Priority scheduling",
            "Schedule requests with higher priority numbers first.",
            O {
                default: json!(false),
                adv: true,
                ..O::DEF
            },
        ),
        p(
            "disable_overlap_schedule",
            "bool",
            "scheduling",
            "Disable overlap scheduler",
            "Turn off the overlap of CPU scheduling with GPU work (debugging).",
            O {
                default: json!(false),
                adv: true,
                lvl: "expert",
                ..O::DEF
            },
        ),
        p(
            "stream_interval",
            "int",
            "scheduling",
            "Stream interval",
            "Tokens per streamed chunk: lower is smoother, higher gives more throughput.",
            O {
                default: json!(1),
                lo: Some(1.0),
                adv: true,
                unit: Some("tokens"),
                ..O::DEF
            },
        ),
        p(
            "sleep_on_idle",
            "bool",
            "scheduling",
            "Sleep on idle",
            "Reduce CPU usage while the engine is idle.",
            O {
                default: json!(false),
                adv: true,
                ..O::DEF
            },
        ),
        p(
            "watchdog_timeout",
            "int",
            "scheduling",
            "Watchdog timeout",
            "Crash if one forward batch takes longer than this; raise it for very long prefills.",
            O {
                default: json!(300),
                lo: Some(1.0),
                adv: true,
                unit: Some("s"),
                ..O::DEF
            },
        ),
        p(
            "attention_backend",
            "enum",
            "performance",
            "Attention backend",
            "Attention kernels. flashinfer, triton and trtllm_mha run on sm_120; fa3 is built for Hopper and the *_mla backends for DeepSeek-style models.",
            O {
                choices: Some(ATTENTION_BACKENDS),
                ..O::DEF
            },
        ),
        p(
            "prefill_attention_backend",
            "enum",
            "performance",
            "Prefill attention backend",
            "Override the attention backend for prefill only.",
            O {
                choices: Some(ATTENTION_BACKENDS),
                adv: true,
                lvl: "expert",
                ..O::DEF
            },
        ),
        p(
            "decode_attention_backend",
            "enum",
            "performance",
            "Decode attention backend",
            "Override the attention backend for decode only.",
            O {
                choices: Some(ATTENTION_BACKENDS),
                adv: true,
                lvl: "expert",
                ..O::DEF
            },
        ),
        p(
            "sampling_backend",
            "enum",
            "performance",
            "Sampling backend",
            "Kernels for the sampling layers.",
            O {
                choices: Some(SAMPLING_BACKENDS),
                adv: true,
                lvl: "expert",
                ..O::DEF
            },
        ),
        p(
            "grammar_backend",
            "enum",
            "performance",
            "Grammar backend",
            "Backend for constrained (JSON or regex) decoding.",
            O {
                choices: Some(GRAMMAR_BACKENDS),
                adv: true,
                ..O::DEF
            },
        ),
        p(
            "disable_radix_cache",
            "bool",
            "performance",
            "Disable radix cache",
            "Turn off RadixAttention prefix caching (it hurts agent and multi-turn workloads).",
            O {
                default: json!(false),
                ..O::DEF
            },
        ),
        p(
            "radix_eviction_policy",
            "enum",
            "performance",
            "Radix eviction policy",
            "Which cached prefixes are dropped first when the cache is full.",
            O {
                default: json!("lru"),
                choices: Some(&["lru", "lfu", "slru", "priority"]),
                adv: true,
                lvl: "expert",
                ..O::DEF
            },
        ),
        p(
            "enable_torch_compile",
            "bool",
            "performance",
            "torch.compile",
            "Experimental: long warm-up, helps small-batch decode.",
            O {
                default: json!(false),
                adv: true,
                ..O::DEF
            },
        ),
        p(
            "torch_compile_max_bs",
            "int",
            "performance",
            "torch.compile max batch",
            "Largest batch size compiled with torch.compile.",
            O {
                default: json!(32),
                lo: Some(1.0),
                adv: true,
                lvl: "expert",
                unit: Some("requests"),
                ..O::DEF
            },
        ),
        p(
            "cuda_graph_max_bs_decode",
            "int",
            "performance",
            "CUDA graph max batch (decode)",
            "Largest decode batch captured as a CUDA graph; bigger means more graph memory and capture time. Unset: derived from the card.",
            O {
                lo: Some(1.0),
                mem: true,
                unit: Some("requests"),
                ..O::DEF
            },
        ),
        p(
            "cuda_graph_backend_decode",
            "enum",
            "performance",
            "CUDA graph backend (decode)",
            "full, breakable, tc_piecewise or disabled. Replaces the deprecated --disable-cuda-graph.",
            O {
                choices: Some(CUDA_GRAPH_BACKENDS),
                adv: true,
                mem: true,
                lvl: "expert",
                ..O::DEF
            },
        ),
        p(
            "cuda_graph_backend_prefill",
            "enum",
            "performance",
            "CUDA graph backend (prefill)",
            "Prefill-phase graph backend; disabled saves memory and capture time.",
            O {
                choices: Some(CUDA_GRAPH_BACKENDS),
                adv: true,
                mem: true,
                lvl: "expert",
                ..O::DEF
            },
        ),
        p(
            "cuda_graph_max_bs_prefill",
            "int",
            "performance",
            "CUDA graph max tokens (prefill)",
            "Largest prefill size captured as a CUDA graph.",
            O {
                lo: Some(1.0),
                adv: true,
                mem: true,
                lvl: "expert",
                unit: Some("tokens"),
                ..O::DEF
            },
        ),
        p(
            "enable_tokenizer_batch_encode",
            "bool",
            "performance",
            "Batch tokenization",
            "Tokenize concurrent text inputs in one batch (not for image inputs).",
            O {
                default: json!(false),
                adv: true,
                lvl: "expert",
                ..O::DEF
            },
        ),
        p(
            "enable_deterministic_inference",
            "bool",
            "performance",
            "Deterministic inference",
            "Batch-invariant kernels for reproducible outputs; slower.",
            O {
                default: json!(false),
                adv: true,
                ..O::DEF
            },
        ),
        p(
            "skip_server_warmup",
            "bool",
            "performance",
            "Skip warmup",
            "Skip the warm-up request at start-up.",
            O {
                default: json!(false),
                adv: true,
                lvl: "expert",
                ..O::DEF
            },
        ),
        p(
            "disable_flashinfer_autotune",
            "bool",
            "performance",
            "Disable FlashInfer autotune",
            "Skip FlashInfer's autotuning pass at start-up (faster start, possibly slower kernels).",
            O {
                default: json!(false),
                adv: true,
                ..O::DEF
            },
        ),
        p(
            "mamba_backend",
            "enum",
            "performance",
            "Mamba SSM backend",
            "Kernel backend for Mamba state operations: triton (the default) or flashinfer.",
            O {
                default: json!("triton"),
                choices: Some(&["triton", "flashinfer"]),
                adv: true,
                ..O::DEF
            },
        ),
        p(
            "linear_attn_backend",
            "enum",
            "performance",
            "Linear-attention backend",
            "Default kernel backend for linear attention (gated delta net and KDA layers).",
            O {
                default: json!("triton"),
                choices: Some(LINEAR_ATTN),
                adv: true,
                ..O::DEF
            },
        ),
        p(
            "linear_attn_decode_backend",
            "enum",
            "performance",
            "Linear-attention decode backend",
            "Override the linear-attention backend for decode. Unset follows linear-attn-backend.",
            O {
                choices: Some(LINEAR_ATTN),
                adv: true,
                ..O::DEF
            },
        ),
        p(
            "linear_attn_prefill_backend",
            "enum",
            "performance",
            "Linear-attention prefill backend",
            "Override the linear-attention backend for prefill. Unset follows linear-attn-backend.",
            O {
                choices: Some(LINEAR_ATTN),
                adv: true,
                ..O::DEF
            },
        ),
        p(
            "tool_call_parser",
            "enum",
            "tools_reasoning",
            "Tool-call parser",
            "Needed by agentic clients: without it tool calls stay raw text and tool_calls is empty. It must match the model's dialect (Qwen3.x and Coder-style XML needs qwen3_coder, not qwen).",
            O {
                choices: Some(TOOL_PARSERS),
                lvl: "basic",
                ..O::DEF
            },
        ),
        p(
            "reasoning_parser",
            "enum",
            "tools_reasoning",
            "Reasoning parser",
            "Splits the model's <think> output into reasoning_content.",
            O {
                choices: Some(REASONING_PARSERS),
                lvl: "basic",
                ..O::DEF
            },
        ),
        p(
            "chat_template",
            "string",
            "tools_reasoning",
            "Chat template",
            "Built-in template name or a template file path inside the container.",
            O {
                adv: true,
                ..O::DEF
            },
        ),
        p(
            "served_model_name",
            "string",
            "tools_reasoning",
            "Served model name",
            "Name returned by /v1/models. Overrides the name the console provides.",
            O {
                lvl: "basic",
                ..O::DEF
            },
        ),
        p(
            "preferred_sampling_params",
            "json",
            "tools_reasoning",
            "Preferred sampling params",
            "JSON sampling defaults advertised by /get_model_info.",
            O {
                adv: true,
                lvl: "expert",
                ..O::DEF
            },
        ),
        p(
            "enable_cache_report",
            "bool",
            "tools_reasoning",
            "Report cached tokens",
            "Return the number of cached prompt tokens in usage.prompt_tokens_details.",
            O {
                default: json!(false),
                adv: true,
                ..O::DEF
            },
        ),
        p(
            "speculative_algorithm",
            "enum",
            "speculative",
            "Speculative algorithm",
            "EAGLE, EAGLE3 and STANDALONE need a draft model; NEXTN uses the model's built-in MTP head; NGRAM needs no draft model; DFLASH and DSPARK are newer drafters.",
            O {
                choices: Some(SPEC_ALGOS),
                mem: true,
                ..O::DEF
            },
        ),
        p(
            "speculative_draft_model_path",
            "string",
            "speculative",
            "Draft model",
            "Hugging Face repository or path of the draft model.",
            O {
                mem: true,
                ..O::DEF
            },
        ),
        p(
            "speculative_num_steps",
            "int",
            "speculative",
            "Draft steps",
            "Steps sampled from the draft model.",
            O {
                lo: Some(1.0),
                mem: true,
                ..O::DEF
            },
        ),
        p(
            "speculative_eagle_topk",
            "int",
            "speculative",
            "EAGLE top-k",
            "Tokens sampled per draft step (1 means chain drafting).",
            O {
                lo: Some(1.0),
                ..O::DEF
            },
        ),
        p(
            "speculative_num_draft_tokens",
            "int",
            "speculative",
            "Draft tokens",
            "Tokens verified per step. With top-k 1, SGLang forces this to steps plus one.",
            O {
                lo: Some(1.0),
                unit: Some("tokens"),
                ..O::DEF
            },
        ),
        p(
            "speculative_accept_threshold_single",
            "float",
            "speculative",
            "Accept threshold",
            "Accept a draft token if its probability in the target model exceeds this.",
            O {
                default: json!(1.0),
                lo: Some(0.0),
                hi: Some(1.0),
                adv: true,
                lvl: "expert",
                ..O::DEF
            },
        ),
        p(
            "speculative_draft_model_quantization",
            "enum",
            "speculative",
            "Draft quantization",
            "Quantization of the draft model (unquant keeps it unquantized).",
            O {
                choices: Some(QUANT_CHOICES),
                adv: true,
                ..O::DEF
            },
        ),
        p(
            "enable_lora",
            "bool",
            "advanced",
            "Enable LoRA",
            "Enable LoRA serving (implied by lora-paths).",
            O { ..O::DEF },
        ),
        p(
            "lora_paths",
            "string_list",
            "advanced",
            "LoRA adapters",
            "Adapters as <path> or <name>=<path>.",
            O { ..O::DEF },
        ),
        p(
            "max_loras_per_batch",
            "int",
            "advanced",
            "Max LoRAs per batch",
            "Adapters (including base-only requests) in one running batch.",
            O {
                default: json!(8),
                lo: Some(1.0),
                adv: true,
                mem: true,
                ..O::DEF
            },
        ),
        p(
            "max_lora_rank",
            "int",
            "advanced",
            "Max LoRA rank",
            "Largest adapter rank. Inferred from the adapters when unset.",
            O {
                lo: Some(1.0),
                adv: true,
                mem: true,
                ..O::DEF
            },
        ),
        p(
            "lora_backend",
            "enum",
            "advanced",
            "LoRA backend",
            "Kernel backend for multi-LoRA serving.",
            O {
                default: json!("csgmv"),
                choices: Some(LORA_BACKENDS),
                adv: true,
                lvl: "expert",
                ..O::DEF
            },
        ),
        p(
            "enable_metrics",
            "bool",
            "network",
            "Prometheus metrics",
            "Exposes /metrics. The console always turns this on; setting it to false has no effect.",
            O {
                default: json!(true),
                adv: true,
                lvl: "expert",
                ..O::DEF
            },
        ),
        p(
            "enable_request_time_stats_logging",
            "bool",
            "network",
            "Request time stats logging",
            "Log per-request timing statistics.",
            O {
                default: json!(false),
                adv: true,
                lvl: "expert",
                ..O::DEF
            },
        ),
        p(
            "random_seed",
            "int",
            "advanced",
            "Random seed",
            "Seed for sampling reproducibility.",
            O {
                adv: true,
                lvl: "expert",
                ..O::DEF
            },
        ),
        p(
            "log_level",
            "enum",
            "advanced",
            "Log level",
            "Logging level of all loggers.",
            O {
                default: json!("info"),
                choices: Some(&["debug", "info", "warning", "error", "critical"]),
                adv: true,
                lvl: "expert",
                ..O::DEF
            },
        ),
        p(
            "decode_log_interval",
            "int",
            "advanced",
            "Decode log interval",
            "Decode iterations between log and metric lines.",
            O {
                default: json!(40),
                lo: Some(1.0),
                adv: true,
                lvl: "expert",
                ..O::DEF
            },
        ),
        p(
            "log_requests",
            "bool",
            "advanced",
            "Log requests",
            "Log metadata, inputs and outputs of every request: this puts prompts into the logs.",
            O {
                default: json!(false),
                adv: true,
                lvl: "expert",
                ..O::DEF
            },
        ),
        p(
            "enable_trace",
            "bool",
            "advanced",
            "OpenTelemetry tracing",
            "Emit OpenTelemetry traces (needs otlp-traces-endpoint).",
            O {
                default: json!(false),
                adv: true,
                lvl: "expert",
                ..O::DEF
            },
        ),
        p(
            "otlp_traces_endpoint",
            "string",
            "advanced",
            "OTLP endpoint",
            "host:port of the collector when tracing is enabled.",
            O {
                default: json!("localhost:4317"),
                adv: true,
                lvl: "expert",
                ..O::DEF
            },
        ),
    ]
}

/// `CATALOG_VERSION_NOTE`.
pub const CATALOG_VERSION_NOTE: &str =
    "lmsysorg/sglang:v0.5.20-cu130 (parser dump and --help read from the local image)";
