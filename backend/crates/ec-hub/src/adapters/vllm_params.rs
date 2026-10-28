//! Port of backend/src/engine_console/adapters/vllm_params.py: the vLLM parameter
//! catalog (data only).
//!
//! Pinned engine: `vllm/vllm-openai:v0.27.1`. Flags, defaults and choices come from `vllm serve --help=all` of the
//! pinned image; the lists `--help` does not print (quantization, reasoning parsers, attention backends) were read from
//! the image's python registries. Both are committed as fixtures (`tests/data/vllm/`) and the tests check this catalog
//! against them, so a typo or a drifted choice list fails offline. Provenance in detail: the Python module's docstring.
//!
//! Two pseudo-flag conventions (handled in `VllmAdapter::build_launch_ctx`, never emitted as argv): `env:NAME` sets the
//! container environment variable NAME; `@image` is no longer used by this catalog (the core refuses it).

use serde_json::{Value, json};

use super::base::ParamSpec;

pub const VLLM_VERSION: &str = "0.27.1";
pub const DEFAULT_IMAGE: &str = "vllm/vllm-openai:v0.27.1";
pub const DOCS_BASE: &str = "https://docs.vllm.ai/en/v0.27.1";
pub const CLI_DOCS: &str = "https://docs.vllm.ai/en/v0.27.1/cli/serve/";
pub const ENV_DOCS: &str = "https://docs.vllm.ai/en/v0.27.1/configuration/env_vars/";
pub const TOOL_DOCS: &str = "https://docs.vllm.ai/en/v0.27.1/features/tool_calling/";
pub const REASONING_DOCS: &str = "https://docs.vllm.ai/en/v0.27.1/features/reasoning_outputs/";
pub const SPEC_DOCS: &str = "https://docs.vllm.ai/en/v0.27.1/features/spec_decode/";

/// Catalog flags that the v0.30.0 image's `--help=all` does not list.
pub const NOT_IN_V0_30_0: &[&str] = &[];

// --- choices read from the pinned image -------------------------------------------------------------------------------
pub const QUANTIZATION: &[&str] = &[
    "awq",
    "auto_awq",
    "fp8",
    "fbgemm_fp8",
    "fp_quant",
    "modelopt",
    "modelopt_fp4",
    "modelopt_mxfp8",
    "modelopt_mixed",
    "auto_gptq",
    "gptq",
    "gptq_marlin",
    "awq_marlin",
    "humming",
    "compressed-tensors",
    "bitsandbytes",
    "experts_int8",
    "moe_wna16",
    "torchao",
    "mxfp4",
    "gpt_oss_mxfp4",
    "deepseek_v4_fp8",
    "online",
    "fp8_per_tensor",
    "fp8_per_block",
    "fp8_per_channel",
    "int8_per_channel_weight_only",
    "nvfp4_per_token",
    "mxfp8",
];
pub const KV_CACHE_DTYPE: &[&str] = &[
    "auto",
    "bfloat16",
    "float16",
    "fp8",
    "fp8_ds_mla",
    "fp8_e4m3",
    "fp8_e5m2",
    "fp8_inc",
    "fp8_per_token_head",
    "int4_per_token_head",
    "int8_per_token_head",
    "nvfp4",
    "turboquant_3bit_nc",
    "turboquant_4bit_nc",
    "turboquant_k3v4_nc",
    "turboquant_k8v4",
];
pub const TOOL_CALL_PARSERS: &[&str] = &[
    "apertus",
    "cohere_command3",
    "cohere_command4",
    "deepseek_v3",
    "deepseek_v31",
    "deepseek_v32",
    "deepseek_v4",
    "ernie45",
    "functiongemma",
    "gemma4",
    "gigachat3",
    "glm45",
    "glm47",
    "granite",
    "granite-20b-fc",
    "granite4",
    "hermes",
    "hunyuan_a13b",
    "hy_v3",
    "inkling",
    "internlm",
    "jamba",
    "kimi_k2",
    "kimi_k3",
    "lfm2",
    "llama3_json",
    "llama4_json",
    "llama4_pythonic",
    "longcat",
    "mimo",
    "minicpm5",
    "minimax_m2",
    "minimax_m3",
    "mistral",
    "olmo3",
    "openai",
    "phi4_mini_json",
    "poolside_v1",
    "pythonic",
    "qwen3_coder",
    "qwen3_xml",
    "seed_oss",
    "step3",
    "step3p5",
    "xlam",
];
pub const REASONING_PARSERS: &[&str] = &[
    "deepseek_r1",
    "deepseek_v3",
    "deepseek_v4",
    "poolside_v1",
    "cohere_command3",
    "cohere_command4",
    "ernie45",
    "gemma4",
    "glm45",
    "glm47",
    "openai_gptoss",
    "granite",
    "holo2",
    "hunyuan_a13b",
    "hy_v3",
    "kimi_k2",
    "kimi_k3",
    "mimo",
    "minimax_m2",
    "minimax_m2_append_think",
    "minimax_m3",
    "mistral",
    "nemotron_v3",
    "olmo3",
    "qwen3",
    "seed_oss",
    "step3",
    "step3p5",
    "inkling",
];
/// CUDA-relevant members of AttentionBackendEnum (ROCm/XPU/CPU members, ViT-only TORCH_SDPA, NO_ATTENTION, CUSTOM and
/// the model-driven DeepSeek-V4 / MiniMax-M3 members are left out).
pub const ATTENTION_BACKENDS: &[&str] = &[
    "auto",
    "FLASH_ATTN",
    "FLASH_ATTN_DIFFKV",
    "TRITON_ATTN",
    "TRITON_ATTN_DIFFKV",
    "FLASHINFER",
    "FLEX_ATTENTION",
    "TURBOQUANT",
    "HPC_ATTN",
    "TRITON_MLA",
    "FLASHINFER_MLA",
    "FLASHINFER_MLA_SPARSE",
    "FLASHINFER_MLA_SPARSE_SM120",
    "CUTLASS_MLA",
    "FLASHMLA",
    "FLASHMLA_SPARSE",
    "FLASH_ATTN_MLA",
    "FLASH_ATTN_MLA_SPARSE",
    "TOKENSPEED_MLA",
];
/// Backends whose supports_compute_capability(12.0) returned False in the pinned image (a frozenset in Python).
pub const ATTENTION_NOT_ON_SM120: &[&str] = &[
    "FLASHINFER_MLA",
    "FLASHINFER_MLA_SPARSE",
    "CUTLASS_MLA",
    "FLASHMLA",
    "FLASHMLA_SPARSE",
    "FLASH_ATTN_MLA",
    "FLASH_ATTN_MLA_SPARSE",
    "TOKENSPEED_MLA",
];
/// `supported_kv_cache_dtypes` of the backends that DO run on sm_120 (frozensets in Python: only membership and
/// `sorted()` are used). FLASH_ATTN is listed with fp8, but on sm_120 vLLM resolves FlashAttention 2, which takes no
/// quantized KV cache, so the adapter treats fp8 KV as unsupported for it there.
pub const ATTENTION_KV_DTYPES: &[(&str, &[&str])] = &[
    (
        "FLASH_ATTN",
        &["auto", "float16", "bfloat16", "fp8", "fp8_e4m3"],
    ),
    (
        "FLASH_ATTN_DIFFKV",
        &["auto", "float16", "bfloat16", "fp8", "fp8_e4m3"],
    ),
    (
        "TRITON_ATTN",
        &[
            "auto",
            "float16",
            "bfloat16",
            "fp8",
            "fp8_e4m3",
            "fp8_e5m2",
            "int4_per_token_head",
            "int8_per_token_head",
            "fp8_per_token_head",
        ],
    ),
    ("TRITON_ATTN_DIFFKV", &["auto", "bfloat16"]),
    (
        "FLASHINFER",
        &[
            "auto", "float16", "bfloat16", "fp8", "fp8_e4m3", "fp8_e5m2", "nvfp4",
        ],
    ),
    ("FLEX_ATTENTION", &["auto", "float16", "bfloat16"]),
    (
        "TURBOQUANT",
        &[
            "turboquant_k8v4",
            "turboquant_4bit_nc",
            "turboquant_k3v4_nc",
            "turboquant_3bit_nc",
        ],
    ),
    (
        "TRITON_MLA",
        &["auto", "float16", "bfloat16", "fp8", "fp8_e4m3"],
    ),
    (
        "FLASHINFER_MLA_SPARSE_SM120",
        &["auto", "fp8", "fp8_e4m3", "fp8_ds_mla"],
    ),
];
pub const MOE_BACKENDS: &[&str] = &[
    "aiter",
    "auto",
    "batched_triton",
    "cutlass",
    "deep_gemm",
    "deep_gemm_mega_moe",
    "emulation",
    "flashinfer_b12x",
    "flashinfer_cutedsl",
    "flashinfer_cutlass",
    "flashinfer_trtllm",
    "flydsl",
    "hpc",
    "humming",
    "marlin",
    "triton",
    "triton_unfused",
];
pub const LINEAR_BACKENDS: &[&str] = &[
    "aiter",
    "auto",
    "conch",
    "cutlass",
    "deep_gemm",
    "emulation",
    "exllama",
    "fbgemm",
    "flashinfer_b12x",
    "flashinfer_cudnn",
    "flashinfer_cutedsl",
    "flashinfer_cutlass",
    "flashinfer_trtllm",
    "humming",
    "machete",
    "marlin",
    "torch",
    "triton",
    "xpu",
    "xpu_woq",
];
pub const SPEC_METHODS: &[&str] = &[
    "bailing_hybrid_mtp",
    "custom_class",
    "deepseek_mtp",
    "dflash",
    "draft_model",
    "dspark",
    "eagle",
    "eagle3",
    "ernie_mtp",
    "exaone4_5_mtp",
    "exaone_moe_mtp",
    "extract_hidden_states",
    "gemma4_mtp",
    "glm4_moe_lite_mtp",
    "glm4_moe_mtp",
    "glm_ocr_mtp",
    "hy_v3_mtp",
    "inkling_mtp",
    "kimi_k3_mtp",
    "longcat_flash_mtp",
    "medusa",
    "mimo_mtp",
    "mimo_v2_mtp",
    "minimax_m3_mtp",
    "mlp_speculator",
    "mtp",
    "nemotron_h_mtp",
    "ngram",
    "ngram_gpu",
    "pangu_ultra_moe_mtp",
    "qwen3_5_mtp",
    "qwen3_next_mtp",
    "step3p5_mtp",
    "suffix",
];
pub const TOKENIZER_MODES: &[&str] = &[
    "auto",
    "cohere",
    "deepseek_v32",
    "deepseek_v4",
    "hf",
    "inkling",
    "kimi_k3",
    "mistral",
    "slow",
];
pub const ALL2ALL_BACKENDS: &[&str] = &[
    "allgather_reducescatter",
    "deepep_high_throughput",
    "deepep_low_latency",
    "deepep_v2",
    "flashinfer_all2allv",
    "flashinfer_nvlink_one_sided",
    "flashinfer_nvlink_two_sided",
    "mori_high_throughput",
    "mori_low_latency",
    "naive",
    "nixl_ep",
    "pplx",
];

/// Keys whose CLI flag takes several space-separated values (argparse nargs): emitted as `--flag v1 v2`.
pub const NARGS_KEYS: &[&str] = &[
    "served_model_name",
    "lora_modules",
    "cudagraph_capture_sizes",
    "allowed_media_domains",
    "kv_cache_dtype_skip_layers",
];

/// Boolean flags that are BooleanOptionalAction in --help (`--x, --no-x`): false is emitted as `--no-x`. Any other bool
/// emits nothing when false.
pub const NEGATABLE_KEYS: &[&str] = &[
    "enable_prefix_caching",
    "enable_chunked_prefill",
    "async_scheduling",
    "enable_expert_parallel",
    "enable_eplb",
    "enforce_eager",
    "trust_remote_code",
    "enable_lora",
    "enable_sleep_mode",
    "language_model_only",
    "disable_hybrid_kv_cache_manager",
    "enable_flashinfer_autotune",
    "disable_sliding_window",
    "enable_prompt_embeds",
    "skip_mm_profiling",
    "enable_log_requests",
    "enable_dbo",
    "enable_mfu_metrics",
    "kv_cache_metrics",
    "cudagraph_metrics",
    "fully_sharded_loras",
    "enable_auto_tool_choice",
    "enable_prompt_tokens_details",
    "enable_server_load_tracking",
    "enable_request_id_headers",
    "disable_fastapi_docs",
    "exclude_tools_when_tool_choice_none",
    "numa_bind",
    "disable_custom_all_reduce",
];

/// Short aliases that appear in the docs heading id of a flag (`--tensor-parallel-size, -tp`).
const SHORT: &[(&str, &str)] = &[
    ("--tensor-parallel-size", "tp"),
    ("--pipeline-parallel-size", "pp"),
    ("--data-parallel-size", "dp"),
    ("--enable-expert-parallel", "ep"),
    ("--quantization", "q"),
    ("--decode-context-parallel-size", "dcp"),
    ("--compilation-config", "cc"),
    ("--speculative-config", "sc"),
    ("--api-server-count", "asc"),
];

/// The heading id of `flag` on the v0.27.1 CLI reference page.
pub fn doc_anchor(flag: &str, negatable: bool) -> String {
    let name = flag.get(2..).unwrap_or("");
    let mut anchor = format!("-{name}");
    if negatable {
        anchor.push_str(&format!("-no-{name}"));
    }
    if let Some((_, s)) = SHORT.iter().find(|(f, _)| *f == flag) {
        anchor.push('-');
        anchor.push_str(s);
    }
    anchor
}

/// The keyword arguments of Python's `_p` (defaults as there).
struct O {
    default: Value,
    choices: Option<&'static [&'static str]>,
    min: Option<f64>,
    max: Option<f64>,
    adv: bool,
    mem: bool,
    lvl: &'static str,
    unit: Option<&'static str>,
    docs: Option<&'static str>,
}

impl O {
    const DEF: O = O {
        default: Value::Null,
        choices: None,
        min: None,
        max: None,
        adv: false,
        mem: false,
        lvl: "advanced",
        unit: None,
        docs: None,
    };
}

#[allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]
fn p(key: &str, flag: &str, label: &str, help: &str, type_: &str, group: &str, o: O) -> ParamSpec {
    let docs = o.docs.map_or_else(
        || {
            if flag.starts_with("env:") {
                ENV_DOCS.to_string()
            } else {
                format!(
                    "{CLI_DOCS}#{}",
                    doc_anchor(flag, NEGATABLE_KEYS.contains(&key))
                )
            }
        },
        str::to_string,
    );
    ParamSpec {
        key: key.into(),
        flag: flag.into(),
        label: label.into(),
        help: help.into(),
        type_: type_.into(),
        group: group.into(),
        default: o.default,
        choices: o
            .choices
            .map(|c| c.iter().map(|s| (*s).to_string()).collect()),
        min: o.min,
        max: o.max,
        advanced: o.adv,
        affects_memory: o.mem,
        requires_restart: true,
        bool_style: "flag".into(),
        docs_url: Some(docs),
        explain: None,
        level: Some(o.lvl.into()),
        unit: o.unit.map(str::to_string),
    }
}

/// The catalog, in Python's order (a push per entry, like the Python list).
#[allow(clippy::too_many_lines, clippy::vec_init_then_push)]
pub fn build_catalog() -> Vec<ParamSpec> {
    let mut c: Vec<ParamSpec> = Vec::new();
    c.push(p("gpu_memory_utilization", "--gpu-memory-utilization", "GPU memory utilization",
        "Fraction of each GPU's memory this vLLM instance may claim (weights + activations + KV cache). Startup fails if less than this is free, so engines sharing a card need a smaller value.",
        "float", "memory", O { default: json!(0.92), min: Some(0.05), max: Some(1.0), mem: true, lvl: "basic", ..O::DEF }));
    c.push(p("kv_cache_memory_bytes", "--kv-cache-memory-bytes", "Fixed KV cache size",
        "Pin the KV cache to an exact size per GPU; when set, gpu-memory-utilization is ignored. Useful for engines that share a card, because the footprint stops depending on free VRAM.",
        "int", "memory", O { min: Some(1.0), adv: true, mem: true, unit: Some("bytes"), ..O::DEF }));
    c.push(p("kv_cache_dtype", "--kv-cache-dtype", "KV cache dtype",
        "Storage type for the KV cache. fp8 roughly doubles the tokens that fit, at some accuracy risk; on sm_120 use it with FLASHINFER or TRITON_ATTN, not FLASH_ATTN.",
        "enum", "memory", O { default: json!("auto"), choices: Some(KV_CACHE_DTYPE), mem: true, lvl: "basic", ..O::DEF }));
    c.push(p("kv_cache_dtype_skip_layers", "--kv-cache-dtype-skip-layers", "KV quantization skip layers",
        "Layer indices or attention types (for example sliding_window) that keep the full-precision KV cache when kv-cache-dtype is quantized.",
        "string_list", "memory", O { adv: true, lvl: "expert", ..O::DEF }));
    c.push(p("cpu_offload_gb", "--cpu-offload-gb", "CPU weight offload",
        "Offload this many GiB of weights per GPU to host RAM, streamed over PCIe on every forward pass. Large slowdown; a last resort to fit a model.",
        "float", "memory", O { default: json!(0), min: Some(0.0), mem: true, unit: Some("GiB"), ..O::DEF }));
    c.push(p("kv_offloading_size", "--kv-offloading-size", "KV offload buffer",
        "Size of the host-RAM KV cache offloading buffer in GiB, summed across tensor-parallel ranks. Setting it turns offloading on.",
        "float", "memory", O { min: Some(0.0), adv: true, mem: true, unit: Some("GiB"), ..O::DEF }));
    c.push(p("kv_offloading_backend", "--kv-offloading-backend", "KV offload backend",
        "Backend for KV offloading: vLLM's native CPU offload or LMCache. Only used when kv-offloading-size is set.",
        "enum", "memory", O { default: json!("native"), choices: Some(&["native", "lmcache"]), adv: true, lvl: "expert", ..O::DEF }));
    c.push(p("block_size", "--block-size", "KV block size",
        "Tokens per KV cache block. Leave unset: hybrid and Mamba-style models pick a large value automatically.",
        "int", "memory", O { min: Some(1.0), adv: true, lvl: "expert", unit: Some("tokens"), ..O::DEF }));
    c.push(p("num_gpu_blocks_override", "--num-gpu-blocks-override", "GPU blocks override",
        "Force the number of KV blocks instead of the profiled value. Meant for testing preemption.",
        "int", "memory", O { min: Some(1.0), adv: true, mem: true, lvl: "expert", ..O::DEF }));
    c.push(p("mamba_cache_dtype", "--mamba-cache-dtype", "Mamba cache dtype",
        "dtype of the convolution and state cache of Mamba and hybrid layers. auto follows the model config.",
        "enum", "memory", O { default: json!("auto"), choices: Some(&["auto", "bfloat16", "float16", "float32"]), adv: true, mem: true, ..O::DEF }));
    c.push(p(
        "mamba_ssm_cache_dtype",
        "--mamba-ssm-cache-dtype",
        "Mamba SSM state dtype",
        "dtype of the SSM state only (the convolution state follows mamba-cache-dtype).",
        "enum",
        "memory",
        O {
            default: json!("auto"),
            choices: Some(&["auto", "bfloat16", "float16", "float32"]),
            adv: true,
            mem: true,
            ..O::DEF
        },
    ));
    c.push(p("mamba_cache_mode", "--mamba-cache-mode", "Mamba cache mode",
        "How Mamba layers cache state for prefix reuse: none (prefix caching off), all (every block boundary) or align (only the last token of each scheduler step on a block boundary).",
        "enum", "memory", O { default: json!("none"), choices: Some(&["align", "all", "none"]), adv: true, ..O::DEF }));
    c.push(p("enable_sleep_mode", "--enable-sleep-mode", "Sleep mode",
        "Allow releasing GPU memory while idle. The /sleep endpoints also need VLLM_SERVER_DEV_MODE, which this console deliberately never sets (it also opens an unauthenticated /collective_rpc).",
        "bool", "memory", O { default: json!(false), adv: true, lvl: "expert", ..O::DEF }));
    c.push(p("tensor_parallel_size", "--tensor-parallel-size", "Tensor parallel size",
        "Shard each layer across this many GPUs. Without NVLink the all-reduce runs over PCIe; prefer one model per card unless the model cannot fit.",
        "int", "parallelism", O { default: json!(1), min: Some(1.0), max: Some(64.0), mem: true, lvl: "basic", unit: Some("GPUs"), ..O::DEF }));
    c.push(p(
        "pipeline_parallel_size",
        "--pipeline-parallel-size",
        "Pipeline parallel size",
        "Split the layer stack into this many sequential stages across GPUs.",
        "int",
        "parallelism",
        O {
            default: json!(1),
            min: Some(1.0),
            max: Some(64.0),
            mem: true,
            unit: Some("GPUs"),
            ..O::DEF
        },
    ));
    c.push(p("data_parallel_size", "--data-parallel-size", "Data parallel size",
        "Run this many full replicas (MoE layers shard over tensor x data parallel). Each replica needs its own GPUs.",
        "int", "parallelism", O { default: json!(1), min: Some(1.0), max: Some(64.0), mem: true, unit: Some("GPUs"), ..O::DEF }));
    c.push(p(
        "enable_expert_parallel",
        "--enable-expert-parallel",
        "Expert parallel",
        "Shard MoE experts across GPUs instead of tensor-parallel splitting them. MoE models only.",
        "bool",
        "parallelism",
        O {
            default: json!(false),
            ..O::DEF
        },
    ));
    c.push(p("decode_context_parallel_size", "--decode-context-parallel-size", "Decode context parallel",
        "Shard the decode KV cache of very long contexts across GPUs. Reuses the tensor-parallel ranks; adds no GPUs.",
        "int", "parallelism", O { default: json!(1), min: Some(1.0), adv: true, lvl: "expert", unit: Some("GPUs"), ..O::DEF }));
    c.push(p(
        "enable_eplb",
        "--enable-eplb",
        "Expert load balancing",
        "Rebalance MoE experts across ranks at runtime. Only meaningful with expert parallelism.",
        "bool",
        "parallelism",
        O {
            default: json!(false),
            adv: true,
            lvl: "expert",
            ..O::DEF
        },
    ));
    c.push(p("expert_placement_strategy", "--expert-placement-strategy", "Expert placement",
        "How experts are laid out across expert-parallel ranks: contiguous (linear) or interleaved (round_robin).",
        "enum", "parallelism", O { default: json!("linear"), choices: Some(&["linear", "round_robin"]), adv: true, lvl: "expert", ..O::DEF }));
    c.push(p("distributed_executor_backend", "--distributed-executor-backend", "Distributed executor",
        "How multi-GPU workers are launched. Left unset, vLLM uses multiprocessing (mp) on a single machine.",
        "enum", "parallelism", O { choices: Some(&["external_launcher", "mp", "ray", "uni"]), adv: true, lvl: "expert", ..O::DEF }));
    c.push(p("disable_custom_all_reduce", "--disable-custom-all-reduce", "Disable custom all-reduce",
        "Fall back to NCCL all-reduce. Try this if tensor parallelism hangs or corrupts output over PCIe.",
        "bool", "parallelism", O { default: json!(false), adv: true, ..O::DEF }));
    c.push(p(
        "all2all_backend",
        "--all2all-backend",
        "All2All backend",
        "Communication backend for expert-parallel MoE layers.",
        "enum",
        "parallelism",
        O {
            default: json!("allgather_reducescatter"),
            choices: Some(ALL2ALL_BACKENDS),
            adv: true,
            lvl: "expert",
            ..O::DEF
        },
    ));
    c.push(p(
        "enable_dbo",
        "--enable-dbo",
        "Dual batch overlap",
        "Overlap compute and communication across two micro-batches (multi-GPU MoE).",
        "bool",
        "parallelism",
        O {
            default: json!(false),
            adv: true,
            lvl: "expert",
            ..O::DEF
        },
    ));
    c.push(p(
        "numa_bind",
        "--numa-bind",
        "NUMA bind workers",
        "Pin each GPU worker process to the CPUs and memory of its GPU's NUMA node.",
        "bool",
        "parallelism",
        O {
            default: json!(false),
            adv: true,
            lvl: "expert",
            ..O::DEF
        },
    ));
    c.push(p("max_model_len", "--max-model-len", "Max context length",
        "Prompt plus output tokens per request. Unset derives it from the model config. vLLM refuses to start if one full-length request cannot fit in the KV cache.",
        "int", "context", O { min: Some(128.0), mem: true, lvl: "basic", unit: Some("tokens"), ..O::DEF }));
    c.push(p("hf_overrides", "--hf-overrides", "HF config overrides (JSON)",
        "JSON merged into the model's Hugging Face config, for example a YaRN rope_parameters block to extend context. A nested dict replaces the original wholesale, so restate every original key; for vision-language models target text_config.",
        "json", "context", O { adv: true, ..O::DEF }));
    c.push(p("allow_long_max_model_len", "env:VLLM_ALLOW_LONG_MAX_MODEL_LEN", "Allow max-len above model max",
        "Sets VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 so a max-model-len beyond the config's max_position_embeddings is accepted. Quality beyond the trained length is not guaranteed.",
        "bool", "context", O { default: json!(false), adv: true, ..O::DEF }));
    c.push(p("limit_mm_per_prompt", "--limit-mm-per-prompt", "Multimodal items per prompt (JSON)",
        "Maximum items per modality in one request, for example {\"image\": 8, \"video\": 1}. Lower values reduce peak memory during profiling.",
        "json", "context", O { adv: true, mem: true, ..O::DEF }));
    c.push(p("language_model_only", "--language-model-only", "Language model only",
        "Disable all multimodal inputs (every limit set to 0), skipping the vision and audio encoders.",
        "bool", "context", O { default: json!(false), mem: true, ..O::DEF }));
    c.push(p(
        "media_io_kwargs",
        "--media-io-kwargs",
        "Media I/O kwargs (JSON)",
        "Per-modality media decoding options, for example {\"video\": {\"num_frames\": 40}}.",
        "json",
        "context",
        O {
            adv: true,
            lvl: "expert",
            ..O::DEF
        },
    ));
    c.push(p(
        "mm_processor_kwargs",
        "--mm-processor-kwargs",
        "MM processor kwargs (JSON)",
        "Arguments forwarded to the model's multimodal processor (for example image size limits).",
        "json",
        "context",
        O {
            adv: true,
            lvl: "expert",
            ..O::DEF
        },
    ));
    c.push(p("mm_processor_cache_gb", "--mm-processor-cache-gb", "MM processor cache",
        "Host-memory cache of processed multimodal inputs, duplicated per API and engine process; 0 disables it.",
        "float", "context", O { default: json!(4), min: Some(0.0), adv: true, unit: Some("GiB"), ..O::DEF }));
    c.push(p(
        "video_pruning_rate",
        "--video-pruning-rate",
        "Video pruning rate",
        "Fraction of video tokens to prune from each video, in [0, 1), for models that support it.",
        "float",
        "context",
        O {
            min: Some(0.0),
            max: Some(0.99),
            adv: true,
            lvl: "expert",
            ..O::DEF
        },
    ));
    c.push(p("skip_mm_profiling", "--skip-mm-profiling", "Skip MM profiling",
        "Skip multimodal memory profiling at startup. Faster start, but the first large image or video can run out of memory.",
        "bool", "context", O { default: json!(false), adv: true, lvl: "expert", ..O::DEF }));
    c.push(p("allowed_local_media_path", "--allowed-local-media-path", "Allowed local media path",
        "Directory that API requests may read images and videos from. Never '/': it lets any prompt read any file in the container.",
        "string", "context", O { adv: true, ..O::DEF }));
    c.push(p(
        "allowed_media_domains",
        "--allowed-media-domains",
        "Allowed media domains",
        "If set, only media URLs on these domains are fetched (egress control).",
        "string_list",
        "context",
        O {
            adv: true,
            ..O::DEF
        },
    ));
    c.push(p(
        "disable_sliding_window",
        "--disable-sliding-window",
        "Disable sliding window",
        "Cap the context at the sliding-window size instead of using windowed attention.",
        "bool",
        "context",
        O {
            default: json!(false),
            adv: true,
            lvl: "expert",
            ..O::DEF
        },
    ));
    c.push(p("generation_config", "--generation-config", "Generation config source",
        "'auto' loads the model's generation_config.json defaults (temperature and so on); 'vllm' uses vLLM's own; or a folder path. A max_new_tokens there becomes a server-wide cap.",
        "string", "context", O { default: json!("auto"), adv: true, ..O::DEF }));
    c.push(p(
        "override_generation_config",
        "--override-generation-config",
        "Override generation config (JSON)",
        "Server-side default sampling parameters, for example {\"temperature\": 0.6}.",
        "json",
        "context",
        O {
            adv: true,
            ..O::DEF
        },
    ));
    c.push(p(
        "max_logprobs",
        "--max-logprobs",
        "Max logprobs",
        "Maximum log-probabilities a request may ask for per token.",
        "int",
        "context",
        O {
            default: json!(20),
            min: Some(0.0),
            adv: true,
            lvl: "expert",
            ..O::DEF
        },
    ));
    c.push(p("enable_prompt_embeds", "--enable-prompt-embeds", "Accept prompt embeds",
        "Allow passing text embeddings instead of token ids. A wrongly shaped input can crash the engine.",
        "bool", "context", O { default: json!(false), adv: true, lvl: "expert", ..O::DEF }));
    c.push(p("quantization", "--quantization", "Quantization",
        "Weight quantization method. Leave unset for pre-quantized checkpoints (NVFP4, compressed-tensors, AWQ): passing fp8 for an NVFP4 checkpoint misloads it. Use fp8 to quantize a BF16 model while loading.",
        "enum", "quantization", O { choices: Some(QUANTIZATION), mem: true, lvl: "basic", ..O::DEF }));
    c.push(p(
        "dtype",
        "--dtype",
        "Activation dtype",
        "Compute dtype. auto uses BF16 for FP32 and BF16 models and FP16 for FP16 models.",
        "enum",
        "quantization",
        O {
            default: json!("auto"),
            choices: Some(&["auto", "bfloat16", "float", "float16", "float32", "half"]),
            mem: true,
            ..O::DEF
        },
    ));
    c.push(p("moe_backend", "--moe-backend", "MoE kernel backend",
        "Kernel for the MoE expert GEMMs. flashinfer_b12x targets sm_12x cards (RTX PRO 6000); an NVFP4 MoE left on auto can fall back to the slower Marlin path there.",
        "enum", "quantization", O { default: json!("auto"), choices: Some(MOE_BACKENDS), ..O::DEF }));
    c.push(p(
        "linear_backend",
        "--linear-backend",
        "Quantized linear backend",
        "Kernel for quantized dense GEMMs. flashinfer_b12x is the NVFP4 path for sm_120 and newer.",
        "enum",
        "quantization",
        O {
            default: json!("auto"),
            choices: Some(LINEAR_BACKENDS),
            adv: true,
            ..O::DEF
        },
    ));
    c.push(p("revision", "--revision", "Model revision",
        "Branch, tag or commit of the model. Pin it: an unpinned main can change size or tokenizer between restarts.",
        "string", "quantization", O { ..O::DEF }));
    c.push(p(
        "tokenizer",
        "--tokenizer",
        "Tokenizer override",
        "Tokenizer repo or path, if it differs from the model's.",
        "string",
        "quantization",
        O {
            adv: true,
            ..O::DEF
        },
    ));
    c.push(p(
        "tokenizer_mode",
        "--tokenizer-mode",
        "Tokenizer mode",
        "Tokenizer implementation to use.",
        "enum",
        "quantization",
        O {
            default: json!("auto"),
            choices: Some(TOKENIZER_MODES),
            adv: true,
            lvl: "expert",
            ..O::DEF
        },
    ));
    c.push(p("trust_remote_code", "--trust-remote-code", "Trust remote code",
        "Run Python code from the model repository. Some architectures need it; it is arbitrary code from the Hub, so only for repositories you have reviewed.",
        "bool", "quantization", O { default: json!(false), lvl: "basic", ..O::DEF }));
    c.push(p("model_impl", "--model-impl", "Model implementation",
        "vLLM's own implementation, or the Transformers fallback. auto prefers vLLM's and falls back.",
        "enum", "quantization", O { default: json!("auto"), choices: Some(&["auto", "terratorch", "transformers", "vllm"]), adv: true, lvl: "expert", ..O::DEF }));
    c.push(p("load_format", "--load-format", "Load format",
        "Weight loading format: auto, safetensors, pt, npcache, dummy, runai_streamer, sharded_state, bitsandbytes, mistral, tensorizer, instanttensor and others.",
        "string", "quantization", O { default: json!("auto"), adv: true, lvl: "expert", ..O::DEF }));
    c.push(p("safetensors_load_strategy", "--safetensors-load-strategy", "Safetensors load strategy",
        "Unset memory-maps weights lazily (and prefetches on NFS). eager reads whole files into RAM first, prefetch warms the page cache; both help on network storage.",
        "enum", "quantization", O { choices: Some(&["eager", "lazy", "prefetch", "torchao"]), adv: true, ..O::DEF }));
    c.push(p("max_num_seqs", "--max-num-seqs", "Max concurrent sequences",
        "Upper bound on sequences processed in one step. Unset, vLLM picks 1024 on GPUs of 70 GiB or more (A100 excepted) and 256 on the others. Lower it for hybrid models whose state cache is small.",
        "int", "scheduling", O { min: Some(1.0), mem: true, lvl: "basic", unit: Some("requests"), ..O::DEF }));
    c.push(p("max_num_batched_tokens", "--max-num-batched-tokens", "Max batched tokens",
        "Token budget per scheduler step (the prefill chunk size). Larger favours throughput and long-prompt latency; smaller favours inter-token latency. Speculative decoding uses part of it for draft tokens.",
        "int", "scheduling", O { min: Some(1.0), mem: true, unit: Some("tokens"), ..O::DEF }));
    c.push(p("enable_prefix_caching", "--enable-prefix-caching", "Prefix caching",
        "Reuse KV blocks for shared prompt prefixes (system prompts, agent loops). Unset: on for ordinary models, off for hybrid (Mamba-style) models.",
        "bool", "scheduling", O { ..O::DEF }));
    c.push(p(
        "enable_chunked_prefill",
        "--enable-chunked-prefill",
        "Chunked prefill",
        "Split long prefills into chunks so running decodes are not starved.",
        "bool",
        "scheduling",
        O { ..O::DEF },
    ));
    c.push(p(
        "async_scheduling",
        "--async-scheduling",
        "Async scheduling",
        "Overlap CPU scheduling with GPU execution to remove idle gaps.",
        "bool",
        "scheduling",
        O { ..O::DEF },
    ));
    c.push(p("scheduling_policy", "--scheduling-policy", "Scheduling policy",
        "fcfs (first come, first served) or priority (requests may carry a priority; lower runs first).",
        "enum", "scheduling", O { default: json!("fcfs"), choices: Some(&["fcfs", "priority"]), ..O::DEF }));
    c.push(p("stream_interval", "--stream-interval", "Stream interval",
        "Tokens buffered per streamed chunk; 1 is smoothest, larger values cut CPU load at high concurrency.",
        "int", "scheduling", O { default: json!(1), min: Some(1.0), adv: true, unit: Some("tokens"), ..O::DEF }));
    c.push(p(
        "long_prefill_token_threshold",
        "--long-prefill-token-threshold",
        "Long prefill threshold",
        "With chunked prefill, prompts longer than this count as long; 0 means no cap.",
        "int",
        "scheduling",
        O {
            default: json!(0),
            min: Some(0.0),
            adv: true,
            lvl: "expert",
            unit: Some("tokens"),
            ..O::DEF
        },
    ));
    c.push(p("prefix_caching_hash_algo", "--prefix-caching-hash-algo", "Prefix hash algorithm",
        "Hash used for prefix-cache blocks. sha256 is collision-safe across tenants; xxhash is faster.",
        "enum", "scheduling", O { default: json!("sha256"), choices: Some(&["sha256", "sha256_cbor", "xxhash", "xxhash_cbor"]), adv: true, lvl: "expert", ..O::DEF }));
    c.push(p("disable_hybrid_kv_cache_manager", "--disable-hybrid-kv-cache-manager", "Disable hybrid KV manager",
        "Allocate the same KV size to every attention layer even for sliding-window or hybrid models (wastes memory).",
        "bool", "scheduling", O { adv: true, lvl: "expert", ..O::DEF }));
    c.push(p(
        "seed",
        "--seed",
        "Random seed",
        "Global seed for reproducibility.",
        "int",
        "scheduling",
        O {
            default: json!(0),
            adv: true,
            lvl: "expert",
            ..O::DEF
        },
    ));
    c.push(p(
        "api_server_count",
        "--api-server-count",
        "API server processes",
        "How many HTTP API server processes to run; defaults to data-parallel-size.",
        "int",
        "scheduling",
        O {
            min: Some(1.0),
            adv: true,
            lvl: "expert",
            ..O::DEF
        },
    ));
    c.push(p(
        "enable_auto_tool_choice",
        "--enable-auto-tool-choice",
        "Auto tool choice",
        "Let the model decide when to call tools. Requires tool-call-parser.",
        "bool",
        "tools_reasoning",
        O {
            lvl: "basic",
            docs: Some(TOOL_DOCS),
            ..O::DEF
        },
    ));
    c.push(p("tool_call_parser", "--tool-call-parser", "Tool-call parser",
        "Parser matching the model's tool-call dialect (hermes, qwen3_xml, qwen3_coder, llama3_json, openai for gpt-oss, glm47, poolside_v1, ...). A mismatch leaves raw tool markup in the message content.",
        "enum", "tools_reasoning", O { choices: Some(TOOL_CALL_PARSERS), lvl: "basic", docs: Some(TOOL_DOCS), ..O::DEF }));
    c.push(p("reasoning_parser", "--reasoning-parser", "Reasoning parser",
        "Splits the chain of thought into reasoning_content (qwen3, deepseek_r1, nemotron_v3, openai_gptoss, glm45, poolside_v1, ...).",
        "enum", "tools_reasoning", O { choices: Some(REASONING_PARSERS), lvl: "basic", docs: Some(REASONING_DOCS), ..O::DEF }));
    c.push(p(
        "chat_template",
        "--chat-template",
        "Chat template",
        "Path or inline Jinja chat template that overrides the model's own.",
        "string",
        "tools_reasoning",
        O { ..O::DEF },
    ));
    c.push(p(
        "chat_template_content_format",
        "--chat-template-content-format",
        "Chat content format",
        "How message content is handed to the template: auto, openai (list of parts) or string.",
        "enum",
        "tools_reasoning",
        O {
            default: json!("auto"),
            choices: Some(&["auto", "openai", "string"]),
            adv: true,
            lvl: "expert",
            ..O::DEF
        },
    ));
    c.push(p(
        "default_chat_template_kwargs",
        "--default-chat-template-kwargs",
        "Default template kwargs (JSON)",
        "Server default kwargs for the chat template, for example {\"enable_thinking\": false}.",
        "json",
        "tools_reasoning",
        O { ..O::DEF },
    ));
    c.push(p(
        "exclude_tools_when_tool_choice_none",
        "--exclude-tools-when-tool-choice-none",
        "Hide tools when tool_choice=none",
        "Drop tool definitions from the prompt when tool_choice is 'none'.",
        "bool",
        "tools_reasoning",
        O {
            adv: true,
            lvl: "expert",
            ..O::DEF
        },
    ));
    c.push(p(
        "tool_parser_plugin",
        "--tool-parser-plugin",
        "Tool parser plugin",
        "Path to a Python file that registers a custom tool parser.",
        "string",
        "tools_reasoning",
        O {
            adv: true,
            lvl: "expert",
            ..O::DEF
        },
    ));
    c.push(p(
        "reasoning_parser_plugin",
        "--reasoning-parser-plugin",
        "Reasoning parser plugin",
        "Path to a Python file that registers a custom reasoning parser.",
        "string",
        "tools_reasoning",
        O {
            adv: true,
            lvl: "expert",
            ..O::DEF
        },
    ));
    c.push(p(
        "structured_outputs_config",
        "--structured-outputs-config",
        "Structured outputs config (JSON)",
        "Guided-decoding configuration, for example {\"backend\": \"xgrammar\"}.",
        "json",
        "tools_reasoning",
        O {
            adv: true,
            lvl: "expert",
            ..O::DEF
        },
    ));
    c.push(p(
        "response_role",
        "--response-role",
        "Response role",
        "Role name of generated messages.",
        "string",
        "tools_reasoning",
        O {
            default: json!("assistant"),
            adv: true,
            lvl: "expert",
            ..O::DEF
        },
    ));
    c.push(p("enforce_eager", "--enforce-eager", "Enforce eager mode",
        "Disable CUDA graphs. Saves about 1 GiB and start-up time, costs decode speed. Debugging only.",
        "bool", "performance", O { default: json!(false), mem: true, ..O::DEF }));
    c.push(p("performance_mode", "--performance-mode", "Performance mode",
        "balanced; interactivity (fine-grained CUDA graphs, latency kernels); or throughput (larger graphs, aggressive batching).",
        "enum", "performance", O { default: json!("balanced"), choices: Some(&["balanced", "interactivity", "throughput"]), ..O::DEF }));
    c.push(p("optimization_level", "--optimization-level", "Optimization level",
        "0 = fastest start-up, 3 = best steady-state speed (more compile effort). The default is 2.",
        "int", "performance", O { default: json!(2), min: Some(0.0), max: Some(3.0), ..O::DEF }));
    c.push(p("compilation_config", "--compilation-config", "Compilation config (JSON)",
        "torch.compile and CUDA-graph settings, for example {\"mode\": 3, \"cudagraph_capture_sizes\": [1, 2, 4, 8]}.",
        "json", "performance", O { adv: true, lvl: "expert", ..O::DEF }));
    c.push(p("cudagraph_capture_sizes", "--cudagraph-capture-sizes", "CUDA graph capture sizes",
        "Batch sizes to capture as CUDA graphs. Fewer sizes mean a faster start and less graph memory.",
        "string_list", "performance", O { adv: true, mem: true, ..O::DEF }));
    c.push(p(
        "max_cudagraph_capture_size",
        "--max-cudagraph-capture-size",
        "Max CUDA graph size",
        "Largest batch size captured as a CUDA graph. Unset: min(max-num-seqs x 2, 512).",
        "int",
        "performance",
        O {
            min: Some(1.0),
            adv: true,
            mem: true,
            unit: Some("requests"),
            ..O::DEF
        },
    ));
    c.push(p("attention_backend", "--attention-backend", "Attention backend",
        "Attention kernel. On sm_120 FLASH_ATTN runs FlashAttention 2, which cannot use an fp8 KV cache; FLASHINFER and TRITON_ATTN can. Replaces the old VLLM_ATTENTION_BACKEND variable.",
        "enum", "performance", O { choices: Some(ATTENTION_BACKENDS), ..O::DEF }));
    c.push(p("enable_flashinfer_autotune", "--enable-flashinfer-autotune", "FlashInfer autotune",
        "Run FlashInfer autotuning during warm-up (results are cached across restarts). Unset lets vLLM decide.",
        "bool", "performance", O { adv: true, ..O::DEF }));
    c.push(p(
        "mamba_backend",
        "--mamba-backend",
        "Mamba SSU backend",
        "Kernel backend for Mamba selective-state updates: triton (the default) or flashinfer.",
        "enum",
        "performance",
        O {
            default: json!("triton"),
            choices: Some(&["triton", "flashinfer"]),
            adv: true,
            ..O::DEF
        },
    ));
    c.push(p(
        "gdn_prefill_backend",
        "--gdn-prefill-backend",
        "GDN prefill backend",
        "Prefill kernel for gated-delta-net (linear attention) layers, as in Qwen3.5 and 3.8.",
        "enum",
        "performance",
        O {
            choices: Some(&["flashinfer", "triton", "cutedsl"]),
            adv: true,
            ..O::DEF
        },
    ));
    c.push(p("speculative_config", "--speculative-config", "Speculative decoding config (JSON)",
        "Full speculative-decoding config, for example {\"method\":\"mtp\",\"num_speculative_tokens\":2}, or a draft model with its revision pinned. Cannot be combined with the spec_* shortcuts.",
        "json", "speculative", O { mem: true, docs: Some(SPEC_DOCS), ..O::DEF }));
    c.push(p(
        "spec_method",
        "--spec-method",
        "Speculative method",
        "Shortcut: the speculation method (mtp, eagle3, ngram, dflash, dspark, draft_model, ...).",
        "enum",
        "speculative",
        O {
            choices: Some(SPEC_METHODS),
            adv: true,
            docs: Some(SPEC_DOCS),
            ..O::DEF
        },
    ));
    c.push(p(
        "spec_model",
        "--spec-model",
        "Draft model",
        "Shortcut: the draft model or EAGLE head repository.",
        "string",
        "speculative",
        O {
            adv: true,
            mem: true,
            docs: Some(SPEC_DOCS),
            ..O::DEF
        },
    ));
    c.push(p(
        "spec_tokens",
        "--spec-tokens",
        "Speculative tokens",
        "Shortcut: draft tokens per step.",
        "int",
        "speculative",
        O {
            min: Some(1.0),
            max: Some(32.0),
            adv: true,
            unit: Some("tokens"),
            docs: Some(SPEC_DOCS),
            ..O::DEF
        },
    ));
    c.push(p(
        "served_model_name",
        "--served-model-name",
        "Extra model names",
        "Extra names the API answers to. The console always serves its own name first.",
        "string_list",
        "network",
        O {
            adv: true,
            ..O::DEF
        },
    ));
    c.push(p("api_key", "env:VLLM_API_KEY", "API key",
        "Require this bearer key. Passed through the VLLM_API_KEY environment variable, never on the command line, so it does not show in the process list.",
        "string", "network", O { ..O::DEF }));
    c.push(p(
        "uvicorn_log_level",
        "--uvicorn-log-level",
        "Uvicorn log level",
        "Log level of the HTTP server.",
        "enum",
        "network",
        O {
            default: json!("info"),
            choices: Some(&["critical", "debug", "error", "info", "trace", "warning"]),
            adv: true,
            lvl: "expert",
            ..O::DEF
        },
    ));
    c.push(p("disable_access_log_for_endpoints", "--disable-access-log-for-endpoints", "Silence access log for paths",
        "Comma-separated paths to leave out of the access log, for example /health,/metrics (the console polls both).",
        "string", "network", O { adv: true, lvl: "expert", ..O::DEF }));
    c.push(p(
        "enable_request_id_headers",
        "--enable-request-id-headers",
        "Request-ID headers",
        "Add an X-Request-Id header to responses.",
        "bool",
        "network",
        O {
            default: json!(false),
            adv: true,
            lvl: "expert",
            ..O::DEF
        },
    ));
    c.push(p(
        "enable_prompt_tokens_details",
        "--enable-prompt-tokens-details",
        "Prompt token details",
        "Return the number of cached prompt tokens in the usage block.",
        "bool",
        "network",
        O {
            adv: true,
            ..O::DEF
        },
    ));
    c.push(p(
        "enable_server_load_tracking",
        "--enable-server-load-tracking",
        "Server load tracking",
        "Expose the number of in-flight requests at /load.",
        "bool",
        "network",
        O {
            adv: true,
            lvl: "expert",
            ..O::DEF
        },
    ));
    c.push(p(
        "enable_log_requests",
        "--enable-log-requests",
        "Log requests",
        "Log every request (parameters at INFO, prompts at DEBUG). Prompts may contain secrets.",
        "bool",
        "network",
        O {
            default: json!(false),
            adv: true,
            lvl: "expert",
            ..O::DEF
        },
    ));
    c.push(p(
        "max_log_len",
        "--max-log-len",
        "Max logged prompt length",
        "Truncate logged prompts and outputs to this many characters or tokens.",
        "int",
        "network",
        O {
            min: Some(0.0),
            adv: true,
            lvl: "expert",
            ..O::DEF
        },
    ));
    c.push(p(
        "disable_fastapi_docs",
        "--disable-fastapi-docs",
        "Disable FastAPI docs",
        "Turn off /docs, /redoc and the OpenAPI schema.",
        "bool",
        "network",
        O {
            default: json!(false),
            adv: true,
            lvl: "expert",
            ..O::DEF
        },
    ));
    c.push(p(
        "root_path",
        "--root-path",
        "Root path",
        "FastAPI root_path, for use behind a path-routing proxy.",
        "string",
        "network",
        O {
            adv: true,
            lvl: "expert",
            ..O::DEF
        },
    ));
    c.push(p(
        "otlp_traces_endpoint",
        "--otlp-traces-endpoint",
        "OTLP traces endpoint",
        "Send OpenTelemetry traces to this URL (for example http://otel-collector:4317).",
        "string",
        "network",
        O {
            adv: true,
            ..O::DEF
        },
    ));
    c.push(p(
        "logging_level",
        "env:VLLM_LOGGING_LEVEL",
        "Log level",
        "Sets VLLM_LOGGING_LEVEL, the verbosity of vLLM's own logs.",
        "enum",
        "network",
        O {
            default: json!("INFO"),
            choices: Some(&["DEBUG", "INFO", "WARNING", "ERROR"]),
            adv: true,
            lvl: "expert",
            ..O::DEF
        },
    ));
    c.push(p(
        "enable_lora",
        "--enable-lora",
        "Enable LoRA",
        "Serve LoRA adapters on top of the base model.",
        "bool",
        "advanced",
        O { ..O::DEF },
    ));
    c.push(p(
        "lora_modules",
        "--lora-modules",
        "LoRA modules",
        "Adapters as name=path (or JSON) entries; needs enable_lora.",
        "string_list",
        "advanced",
        O { ..O::DEF },
    ));
    c.push(p(
        "max_loras",
        "--max-loras",
        "Max LoRAs per batch",
        "Adapters that can be active in one batch.",
        "int",
        "advanced",
        O {
            default: json!(1),
            min: Some(1.0),
            adv: true,
            mem: true,
            ..O::DEF
        },
    ));
    c.push(p(
        "max_lora_rank",
        "--max-lora-rank",
        "Max LoRA rank",
        "Largest adapter rank accepted.",
        "enum",
        "advanced",
        O {
            default: json!("16"),
            choices: Some(&["1", "8", "16", "32", "64", "128", "256", "320", "512"]),
            adv: true,
            mem: true,
            ..O::DEF
        },
    ));
    c.push(p(
        "max_cpu_loras",
        "--max-cpu-loras",
        "Max CPU LoRAs",
        "Adapters cached in host RAM (at least max_loras).",
        "int",
        "advanced",
        O {
            min: Some(1.0),
            adv: true,
            lvl: "expert",
            ..O::DEF
        },
    ));
    c.push(p(
        "lora_dtype",
        "--lora-dtype",
        "LoRA dtype",
        "dtype of the adapter weights; auto follows the base model.",
        "enum",
        "advanced",
        O {
            default: json!("auto"),
            choices: Some(&["auto", "bfloat16", "float16"]),
            adv: true,
            lvl: "expert",
            ..O::DEF
        },
    ));
    c.push(p("fully_sharded_loras", "--fully-sharded-loras", "Fully sharded LoRAs",
        "Shard all LoRA math across tensor-parallel ranks (faster at high rank or large tensor-parallel size).",
        "bool", "advanced", O { default: json!(false), adv: true, lvl: "expert", ..O::DEF }));
    c.push(p(
        "enable_mfu_metrics",
        "--enable-mfu-metrics",
        "MFU metrics",
        "Export estimated model FLOPs utilization metrics.",
        "bool",
        "advanced",
        O {
            default: json!(false),
            adv: true,
            lvl: "expert",
            ..O::DEF
        },
    ));
    c.push(p(
        "kv_cache_metrics",
        "--kv-cache-metrics",
        "KV residency metrics",
        "Sampled KV block lifetime and reuse histograms.",
        "bool",
        "advanced",
        O {
            default: json!(false),
            adv: true,
            lvl: "expert",
            ..O::DEF
        },
    ));
    c.push(p(
        "cudagraph_metrics",
        "--cudagraph-metrics",
        "CUDA graph metrics",
        "Padding and dispatch-mode metrics for CUDA graphs.",
        "bool",
        "advanced",
        O {
            default: json!(false),
            adv: true,
            lvl: "expert",
            ..O::DEF
        },
    ));
    c.push(p("show_hidden_metrics_for_version", "--show-hidden-metrics-for-version", "Show hidden metrics",
        "Re-enable Prometheus metrics hidden since this version (for example 0.10) so old dashboards keep working.",
        "string", "advanced", O { adv: true, lvl: "expert", ..O::DEF }));
    c.push(p(
        "kv_transfer_config",
        "--kv-transfer-config",
        "KV transfer config (JSON)",
        "Disaggregated prefill and KV connector configuration.",
        "json",
        "advanced",
        O {
            adv: true,
            lvl: "expert",
            ..O::DEF
        },
    ));
    c.push(p(
        "additional_config",
        "--additional-config",
        "Additional platform config (JSON)",
        "Platform-specific extra configuration.",
        "json",
        "advanced",
        O {
            adv: true,
            lvl: "expert",
            ..O::DEF
        },
    ));
    c.push(p(
        "worker_multiproc_method",
        "env:VLLM_WORKER_MULTIPROC_METHOD",
        "Worker start method",
        "Sets VLLM_WORKER_MULTIPROC_METHOD: how worker processes start (fork or spawn).",
        "enum",
        "advanced",
        O {
            default: json!("fork"),
            choices: Some(&["fork", "spawn"]),
            adv: true,
            lvl: "expert",
            ..O::DEF
        },
    ));
    c
}
