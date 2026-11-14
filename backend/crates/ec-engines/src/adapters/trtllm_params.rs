//! Python: backend/src/engine_console/adapters/trtllm_params.py.
//!
//! TensorRT-LLM (`trtllm-serve`, PyTorch backend) parameter catalog: data only, no logic. Everything was read from
//! the pinned image `nvcr.io/nvidia/tensorrt-llm/release:1.2.1` (see the Python module for the provenance).
//!
//! Pseudo-flag conventions (`ParamSpec.flag`; never emitted as argv): `--name` a real option of `trtllm-serve
//! serve`; `yaml:a.b.c` a key path inside the generated `--extra_llm_api_options` file; `yaml:*` free-form top-level
//! keys merged into that file; `env:NAME` a container environment variable; `@image` overrides `LaunchSpec.image`.
//!
//! The body of [`build_catalog`] is generated from the Python calls by backend/parity/gen/catalog_ws_a2.py (one `p(...)`
//! per `_p(...)`, keyword for keyword); `tests/test_ws_a2_goldens.rs` proves it equals Python's catalog.

use serde_json::Value;

use super::base::ParamSpec;

macro_rules! repo {
    () => {
        "https://github.com/NVIDIA/TensorRT-LLM/blob/376f7e1bd8ed543f75014309e3fd4b237e9b0e73"
    };
}
macro_rules! docs {
    () => {
        concat!(repo!(), "/docs/source")
    };
}
macro_rules! doc_serve {
    () => {
        concat!(docs!(), "/commands/trtllm-serve/trtllm-serve.rst")
    };
}

pub const TRTLLM_VERSION: &str = "1.2.1";
pub const IMAGE_REPO: &str = "nvcr.io/nvidia/tensorrt-llm/release";
pub const DEFAULT_IMAGE: &str = "nvcr.io/nvidia/tensorrt-llm/release:1.2.1";
/// Manifest digest of the amd64 image the catalog was read from (the tag, not the digest, is what node policies match).
pub const IMAGE_DIGEST: &str =
    "sha256:33cd085b772947bd22b7273886539331420404e5d2a4a039945241945ff927b9";
/// `TRT_LLM_GIT_COMMIT` in the image environment; docs links point at this exact commit.
pub const SOURCE_COMMIT: &str = "376f7e1bd8ed543f75014309e3fd4b237e9b0e73";

pub const DOCS_BASE: &str = docs!();
pub const DOC_SERVE: &str = doc_serve!();
pub const DOC_YAML: &str = concat!(doc_serve!(), "#configuring-with-yaml-files");
pub const DOC_METRICS: &str = concat!(doc_serve!(), "#metrics-endpoint");
pub const DOC_KV: &str = concat!(docs!(), "/features/kvcache.md");
pub const DOC_PARALLEL: &str = concat!(docs!(), "/features/parallel-strategy.md");
pub const DOC_QUANT: &str = concat!(docs!(), "/features/quantization.md");
pub const DOC_SPEC: &str = concat!(docs!(), "/features/speculative-decoding.md");
pub const DOC_ATTENTION: &str = concat!(docs!(), "/features/attention.md");
pub const DOC_OVERLAP: &str = concat!(docs!(), "/features/overlap-scheduler.md");
pub const DOC_COMPILE: &str = concat!(
    docs!(),
    "/features/torch_compile_and_piecewise_cuda_graph.md"
);
pub const DOC_GUIDED: &str = concat!(docs!(), "/features/guided-decoding.md");
pub const DOC_SAMPLING: &str = concat!(docs!(), "/features/sampling.md");
pub const DOC_MULTIMODAL: &str = concat!(docs!(), "/features/multi-modality.md");
pub const DOC_COMBINATIONS: &str = concat!(docs!(), "/features/feature-combination-matrix.md");
pub const DOC_HARDWARE: &str = concat!(docs!(), "/supported-hardware.md");
pub const SRC_LLM_ARGS: &str = concat!(repo!(), "/tensorrt_llm/llmapi/llm_args.py");

pub const YAML_PREFIX: &str = "yaml:";
pub const YAML_EXTRA_FLAG: &str = "yaml:*";
pub const EXTRA_OPTIONS_PATH: &str = "/config/extra-llm-api-options.yaml";

// ---- choices read from the image ----------------------------------------------------------------------------------
pub const LOG_LEVELS: &[&str] = &[
    "internal_error",
    "error",
    "warning",
    "info",
    "verbose",
    "debug",
    "trace",
];
pub const REASONING_PARSERS: &[&str] = &["deepseek-r1", "qwen3", "nano-v3"];
pub const TOOL_PARSERS: &[&str] = &[
    "qwen3",
    "qwen3_coder",
    "kimi_k2",
    "deepseek_v3",
    "deepseek_v31",
    "deepseek_v32",
];
/// model_loader._VALID_KV_CACHE_DTYPES
pub const KV_CACHE_DTYPES: &[&str] = &["auto", "fp8", "nvfp4"];
pub const MAMBA_CACHE_DTYPES: &[&str] = &["auto", "float16", "bfloat16", "float32"];
pub const MOE_BACKENDS: &[&str] = &[
    "CUTLASS", "CUTEDSL", "WIDEEP", "TRTLLM", "DEEPGEMM", "VANILLA", "TRITON",
];
/// get_attention_backend() accepts these four; any other string silently falls back to TRTLLM.
pub const ATTENTION_BACKENDS: &[&str] = &[
    "TRTLLM",
    "FLASHINFER",
    "VANILLA",
    "FLASHINFER_STAR_ATTENTION",
];
pub const ALLREDUCE_STRATEGIES: &[&str] = &[
    "AUTO",
    "NCCL",
    "UB",
    "MINLATENCY",
    "ONESHOT",
    "TWOSHOT",
    "LOWPRECISION",
    "MNNVL",
    "NCCL_SYMMETRIC",
];
pub const SAMPLER_TYPES: &[&str] = &["auto", "TorchSampler", "TRTLLMSampler"];
pub const CAPACITY_POLICIES: &[&str] = &["GUARANTEED_NO_EVICT", "MAX_UTILIZATION", "STATIC_BATCH"];
pub const CHUNKING_POLICIES: &[&str] = &["FIRST_COME_FIRST_SERVED", "EQUAL_PROGRESS"];
pub const GUIDED_BACKENDS: &[&str] = &["xgrammar", "llguidance"];
pub const NVFP4_GEMM_BACKENDS: &[&str] = &["cutlass", "cublaslt", "cutedsl", "cuda_core"];
/// The schema also lists "ray", but the image ships no ray package, so only rpc can work.
pub const ORCHESTRATORS: &[&str] = &["rpc"];
pub const TOKENIZER_MODES: &[&str] = &["auto", "slow"];
/// the one built-in alias; import paths would load arbitrary code
pub const CUSTOM_TOKENIZERS: &[&str] = &["deepseek_v32"];
/// `decoding_type` values that work on the PyTorch backend and need no Python object.
pub const SPEC_TYPES: &[&str] = &["MTP", "Eagle", "NGram", "DraftTarget", "AUTO"];

/// Options of `trtllm-serve serve` the adapter deliberately does not expose, with the reason (in Python's order).
pub const EXCLUDED_SERVE_FLAGS: &[(&str, &str)] = &[
    (
        "--host",
        "fixed to 0.0.0.0 inside the container; the console publishes the port to loopback only",
    ),
    (
        "--port",
        "fixed to the adapter's default_port; the console maps it",
    ),
    (
        "--backend",
        "always the PyTorch backend: the TensorRT-engine backend needs prebuilt engines the console cannot manage",
    ),
    (
        "--custom_module_dirs",
        "imports arbitrary Python from a directory",
    ),
    (
        "--config",
        "generated by the adapter from the yaml: parameters",
    ),
    (
        "--extra_llm_api_options",
        "generated by the adapter from the yaml: parameters",
    ),
    (
        "--gpus_per_node",
        "auto-detected from the GPUs the container sees; only matters across machines",
    ),
    (
        "--moe_cluster_parallel_size",
        "parsed by click but dropped by serve() in 1.2.1 (it is not a named argument of get_llm_args, whose extra keywords serve() discards); the YAML key of the same name is honoured and is what the moe_cluster_parallel_size parameter uses",
    ),
    (
        "--fail_fast_on_attention_window_too_large",
        "read only by the TensorRT-engine backend (llmapi/llm.py builds its ExecutorConfig with it); the PyTorch backend never reads it and lowers an oversized attention window with a warning instead",
    ),
    (
        "--metadata_server_config_file",
        "disaggregated serving only",
    ),
    ("--server_role", "disaggregated serving only"),
    ("--disagg_cluster_uri", "disaggregated serving only"),
];

/// Top-level fields of TorchLlmArgs (83), as introspected in the image.
pub const TORCH_LLM_ARGS_FIELDS: &[&str] = &[
    "allreduce_strategy",
    "attention_dp_config",
    "attn_backend",
    "backend",
    "batch_wait_max_tokens_ratio",
    "batch_wait_timeout_iters",
    "batch_wait_timeout_ms",
    "batched_logits_processor",
    "cache_transceiver_config",
    "checkpoint_format",
    "checkpoint_loader",
    "context_parallel_size",
    "cp_config",
    "cuda_graph_config",
    "custom_tokenizer",
    "decoding_config",
    "disable_flashinfer_sampling",
    "disable_overlap_scheduler",
    "dtype",
    "enable_attention_dp",
    "enable_autotuner",
    "enable_chunked_prefill",
    "enable_iter_perf_stats",
    "enable_iter_req_stats",
    "enable_layerwise_nvtx_marker",
    "enable_lm_head_tp_in_adp",
    "enable_lora",
    "enable_min_latency",
    "enable_sleep",
    "env_overrides",
    "fail_fast_on_attention_window_too_large",
    "force_dynamic_quantization",
    "garbage_collection_gen0_threshold",
    "gather_generation_logits",
    "gpus_per_node",
    "guided_decoding_backend",
    "iter_stats_max_iterations",
    "kv_cache_config",
    "kv_connector_config",
    "load_format",
    "lora_config",
    "max_batch_size",
    "max_beam_width",
    "max_input_len",
    "max_num_tokens",
    "max_seq_len",
    "max_stats_len",
    "mm_encoder_only",
    "model",
    "moe_cluster_parallel_size",
    "moe_config",
    "moe_expert_parallel_size",
    "moe_tensor_parallel_size",
    "mpi_session",
    "num_postprocess_workers",
    "nvfp4_gemm_config",
    "orchestrator_type",
    "otlp_traces_endpoint",
    "peft_cache_config",
    "perf_metrics_max_requests",
    "pipeline_parallel_size",
    "postprocess_tokenizer_dir",
    "pp_partition",
    "print_iter_log",
    "ray_placement_config",
    "ray_worker_extension_cls",
    "reasoning_parser",
    "request_stats_max_iterations",
    "return_perf_metrics",
    "revision",
    "sampler_force_async_worker",
    "sampler_type",
    "scheduler_config",
    "skip_tokenizer_init",
    "sparse_attention_config",
    "speculative_config",
    "stream_interval",
    "tensor_parallel_size",
    "tokenizer",
    "tokenizer_mode",
    "tokenizer_revision",
    "torch_compile_config",
    "trust_remote_code",
];

/// Nested config classes a yaml: path may descend into: top-level field -> class name in the fixture.
pub const YAML_NESTED: &[(&str, &str)] = &[
    ("kv_cache_config", "KvCacheConfig"),
    ("cuda_graph_config", "CudaGraphConfig"),
    ("moe_config", "MoeConfig"),
    ("scheduler_config", "SchedulerConfig"),
    ("torch_compile_config", "TorchCompileConfig"),
    ("nvfp4_gemm_config", "Nvfp4GemmConfig"),
    ("attention_dp_config", "AttentionDpConfig"),
];

/// Top-level YAML keys that `extra_llm_options` may NOT set, with the reason (in Python's order).
pub const EXTRA_OPTIONS_DENIED: &[(&str, &str)] = &[
    ("model", "set by the launch (the model you picked)"),
    ("backend", "always pytorch"),
    ("mpi_session", "an in-process object, not a setting"),
    ("checkpoint_loader", "needs a Python object"),
    ("batched_logits_processor", "needs a Python object"),
    (
        "env_overrides",
        "sets arbitrary environment variables inside the engine process",
    ),
    ("kv_connector_config", "imports a Python module by name"),
    ("ray_worker_extension_cls", "imports a Python class by name"),
    ("ray_placement_config", "needs Ray objects"),
    ("cache_transceiver_config", "disaggregated serving only"),
    (
        "lora_config",
        "needs adapter directories the console does not mount",
    ),
    (
        "peft_cache_config",
        "only meaningful with LoRA adapters the console does not mount",
    ),
    (
        "enable_lora",
        "needs adapter directories the console does not mount",
    ),
    ("postprocess_tokenizer_dir", "managed by the engine"),
    ("gpus_per_node", "auto-detected"),
];

/// PyTorch-backend model classes registered in the image (`MODEL_CLASS_MAPPING`), sorted.
pub const SUPPORTED_ARCHITECTURES: &[&str] = &[
    "BertForSequenceClassification",
    "CLIPVisionModel",
    "DeciLMForCausalLM",
    "DeepseekV32ForCausalLM",
    "DeepseekV3ForCausalLM",
    "EAGLE3LlamaForCausalLM",
    "Eagle3DeepSeekV3ForCausalLM",
    "Exaone4ForCausalLM",
    "ExaoneMoEForCausalLM",
    "Gemma3ForCausalLM",
    "Gemma3ForConditionalGeneration",
    "Glm4MoeForCausalLM",
    "GptOssForCausalLM",
    "HCXVisionForCausalLM",
    "HCXVisionModel",
    "HunYuanDenseV1ForCausalLM",
    "HunYuanMoEV1ForCausalLM",
    "Llama4ForConditionalGeneration",
    "LlamaForCausalLM",
    "LlavaLlamaModel",
    "LlavaNextForConditionalGeneration",
    "MTPDraftModelForCausalLM",
    "MiniMaxM2ForCausalLM",
    "Mistral3ForConditionalGeneration",
    "MistralForCausalLM",
    "MistralLarge3EagleForCausalLM",
    "MistralLarge3ForCausalLM",
    "MixtralForCausalLM",
    "MllamaForConditionalGeneration",
    "NemotronForCausalLM",
    "NemotronHForCausalLM",
    "NemotronH_Nano_VL_V2",
    "Phi3ForCausalLM",
    "Phi4MMForCausalLM",
    "PixtralForConditionalGeneration",
    "PixtralVisionModel",
    "Qwen2ForCausalLM",
    "Qwen2ForProcessRewardModel",
    "Qwen2ForRewardModel",
    "Qwen2MoeForCausalLM",
    "Qwen2VLForConditionalGeneration",
    "Qwen2_5_VLForConditionalGeneration",
    "Qwen3ForCausalLM",
    "Qwen3MoeForCausalLM",
    "Qwen3NextForCausalLM",
    "Qwen3VLForConditionalGeneration",
    "Qwen3VLMoeForConditionalGeneration",
    "SeedOssForCausalLM",
    "SiglipVisionModel",
    "Starcoder2ForCausalLM",
];

/// The `_p()` helper: one parameter, with the conventions of this catalog applied by [`P::build`].
pub struct P {
    spec: ParamSpec,
    docs: Option<String>,
}

fn p(key: &str, flag: &str, label: &str, help: &str, type_: &str, group: &str) -> P {
    P {
        spec: ParamSpec {
            key: key.into(),
            flag: flag.into(),
            label: label.into(),
            help: help.into(),
            type_: type_.into(),
            group: group.into(),
            default: Value::Null,
            choices: None,
            min: None,
            max: None,
            advanced: false,
            affects_memory: false,
            requires_restart: true,
            bool_style: "flag".into(),
            docs_url: None,
            explain: None,
            level: None,
            unit: None,
        },
        docs: None,
    }
}

impl P {
    fn default(mut self, v: Value) -> Self {
        self.spec.default = v;
        self
    }
    fn choices(mut self, c: &[&str]) -> Self {
        self.spec.choices = Some(c.iter().map(|s| (*s).to_string()).collect());
        self
    }
    fn min(mut self, v: f64) -> Self {
        self.spec.min = Some(v);
        self
    }
    fn max(mut self, v: f64) -> Self {
        self.spec.max = Some(v);
        self
    }
    fn adv(mut self) -> Self {
        self.spec.advanced = true;
        self
    }
    fn mem(mut self) -> Self {
        self.spec.affects_memory = true;
        self
    }
    fn docs(mut self, d: &str) -> Self {
        self.docs = Some(d.into());
        self
    }
    fn unit(mut self, u: &str) -> Self {
        self.spec.unit = Some(u.into());
        self
    }
    fn lvl(mut self, l: &str) -> Self {
        self.spec.level = Some(l.into());
        self
    }
    /// `_p`: the docs link defaults to the serve page (`--` flags) or the YAML section; YAML and env booleans are
    /// emitted explicitly (true/false), so a boolean can force an engine default off.
    fn build(mut self) -> ParamSpec {
        let flag = self.spec.flag.clone();
        self.spec.docs_url = Some(self.docs.unwrap_or_else(|| {
            if flag.starts_with("--") {
                format!("{DOC_SERVE}#syntax")
            } else {
                DOC_YAML.to_string()
            }
        }));
        if self.spec.type_ == "bool" && (flag.starts_with("yaml:") || flag.starts_with("env:")) {
            self.spec.bool_style = "value".into();
        }
        self.spec
    }
}

// ---- generated by backend/parity/gen/catalog_ws_a2.py: begin ----
pub fn build_catalog() -> Vec<ParamSpec> {
    vec![
        p("free_gpu_memory_fraction", "--free_gpu_memory_fraction", "KV cache share of free GPU memory", "Share of the GPU memory that is still free after the weights are loaded and a profiling pass has run, which the KV cache may claim. Default 0.9; must stay below 1.", "float", "memory").default(Value::from(0.9)).min(0.01).max(0.99).mem().docs(DOC_KV).lvl("basic").build(),
        p("kv_cache_dtype", "yaml:kv_cache_config.dtype", "KV cache dtype", "Storage type of the KV cache: auto follows the checkpoint, fp8 about halves the cache for a small accuracy risk, nvfp4 needs a checkpoint prepared for it (not listed as supported on sm_120).", "enum", "memory").default(Value::from("auto")).choices(KV_CACHE_DTYPES).mem().docs(DOC_QUANT).lvl("basic").build(),
        p("kv_cache_max_tokens", "yaml:kv_cache_config.max_tokens", "KV cache token cap", "Upper bound on the tokens held in the KV cache. If the free-memory fraction allows fewer, the smaller value wins.", "int", "memory").min(1.0).adv().mem().unit("tokens").docs(DOC_KV).build(),
        p("kv_cache_max_gib", "yaml:kv_cache_config.max_gpu_total_bytes", "KV cache size cap (GiB per GPU)", "Upper bound on the GPU memory each GPU gives the KV cache, in GiB; the smaller of this and the fraction-based size is used. 0 or unset means no cap.", "float", "memory").min(0.0).adv().mem().unit("GiB").docs(DOC_KV).build(),
        p("kv_host_cache_gib", "yaml:kv_cache_config.host_cache_size", "KV host offload (GiB per GPU)", "Host RAM, per GPU worker, that evicted KV blocks are copied to so they can be reused without recomputing. Unset or 0 turns offloading off.", "float", "memory").min(0.0).adv().mem().unit("GiB").docs(DOC_KV).build(),
        p("tokens_per_block", "yaml:kv_cache_config.tokens_per_block", "KV block size (tokens)", "Tokens stored in one KV cache block. Must be a power of two greater than 1; the default is 32.", "int", "memory").default(Value::from(32_i64)).min(2.0).adv().mem().unit("tokens").docs(DOC_KV).lvl("expert").build(),
        p("kv_max_attention_window", "yaml:kv_cache_config.max_attention_window", "Attention window per layer", "Sliding-window sizes in tokens, repeated across the layers, for models with windowed attention. Leave unset unless you know the model's layer pattern.", "string_list", "memory").adv().mem().unit("tokens").docs(DOC_KV).lvl("expert").build(),
        p("mamba_ssm_cache_dtype", "yaml:kv_cache_config.mamba_ssm_cache_dtype", "Recurrent-state dtype", "Data type of the recurrent state of Mamba and linear-attention layers in hybrid models; auto follows the model config.", "enum", "memory").default(Value::from("auto")).choices(MAMBA_CACHE_DTYPES).adv().mem().docs(SRC_LLM_ARGS).lvl("expert").build(),
        p("tensor_parallel_size", "--tensor_parallel_size", "Tensor parallel size", "Split every layer across this many GPUs. Without NVLink the GPUs exchange data over PCIe on every layer; NVIDIA's own RTX PRO 6000 benchmarks used pipeline parallelism instead.", "int", "parallelism").default(Value::from(1_i64)).min(1.0).mem().docs(DOC_PARALLEL).lvl("basic").build(),
        p("pipeline_parallel_size", "--pipeline_parallel_size", "Pipeline parallel size", "Split the model into this many consecutive stages, one per GPU. Needs less GPU-to-GPU traffic than tensor parallelism, but cannot be combined with MTP or EAGLE-3 speculation.", "int", "parallelism").default(Value::from(1_i64)).min(1.0).mem().docs(DOC_PARALLEL).lvl("advanced").build(),
        p("context_parallel_size", "--context_parallel_size", "Context parallel size", "Spread very long prompts over several GPUs. Multiplies the number of GPUs needed.", "int", "parallelism").default(Value::from(1_i64)).min(1.0).adv().mem().docs(DOC_PARALLEL).lvl("expert").build(),
        p("moe_expert_parallel_size", "--moe_expert_parallel_size", "MoE expert parallel size", "For mixture-of-experts models: how many GPUs the experts are spread over. Must divide the tensor-parallel size; leave unset for dense models.", "int", "parallelism").min(1.0).adv().docs(DOC_PARALLEL).build(),
        p("moe_tensor_parallel_size", "yaml:moe_tensor_parallel_size", "MoE tensor parallel size", "For MoE models: how many GPUs each expert's weights are sliced across. The product with the expert and cluster sizes must equal the tensor-parallel size.", "int", "parallelism").min(1.0).adv().docs(DOC_PARALLEL).lvl("expert").build(),
        p("moe_cluster_parallel_size", "yaml:moe_cluster_parallel_size", "MoE cluster parallel size", "Expert cluster parallelism, an extra grouping inside the expert split. Only with expert parallel size 1.", "int", "parallelism").min(1.0).adv().docs(DOC_PARALLEL).lvl("expert").build(),
        p("enable_attention_dp", "yaml:enable_attention_dp", "Attention data parallel", "Run the attention layers data-parallel (every GPU serves different requests) while the rest stays tensor-parallel. Mostly useful for large MoE models under heavy load.", "bool", "parallelism").default(Value::Bool(false)).adv().docs(DOC_PARALLEL).build(),
        p("allreduce_strategy", "yaml:allreduce_strategy", "All-reduce strategy", "How GPUs sum partial results after each layer. AUTO lets the engine choose; NCCL is the plain choice that always works.", "enum", "parallelism").default(Value::from("AUTO")).choices(ALLREDUCE_STRATEGIES).adv().docs(SRC_LLM_ARGS).lvl("expert").build(),
        p("orchestrator_type", "yaml:orchestrator_type", "Worker orchestrator", "How the server launches and talks to its GPU worker processes. Unset uses MPI; rpc avoids MPI. Prototype in 1.2.1 (the image has no Ray, so the ray choice is left out).", "enum", "parallelism").choices(ORCHESTRATORS).adv().docs(SRC_LLM_ARGS).lvl("expert").build(),
        p("nccl_p2p_disable", "env:NCCL_P2P_DISABLE", "Disable GPU peer-to-peer (NCCL)", "Sets NCCL_P2P_DISABLE=1 so GPUs stop copying directly to each other and go through host memory. Only for multi-GPU launches that hang because direct copies do not work.", "bool", "parallelism").default(Value::Bool(false)).adv().docs(DOC_PARALLEL).lvl("expert").build(),
        p("max_seq_len", "--max_seq_len", "Max sequence length", "Longest request (prompt plus answer) in tokens. Unset reads it from the model config; a larger value than the model supports is cut down by the engine.", "int", "context").min(1.0).mem().unit("tokens").lvl("basic").build(),
        p("max_beam_width", "--max_beam_width", "Max beam width", "Largest beam-search width a request may use. Keep 1 unless you serve beam search; larger values cost memory.", "int", "context").default(Value::from(1_i64)).min(1.0).adv().mem().build(),
        p("media_io_kwargs", "--media_io_kwargs", "Media I/O kwargs (JSON)", "Options for decoding images, video and audio, as a JSON object (for example {\"image\": {...}}). Multimodal models only.", "json", "context").adv().docs(DOC_MULTIMODAL).build(),
        p("revision", "--revision", "Model revision", "Branch, tag or commit of the model in the Hugging Face cache. Pin it: an unpinned main can change between restarts.", "string", "quantization").lvl("advanced").build(),
        p("tokenizer", "--tokenizer", "Tokenizer override", "Tokenizer repository or path if it differs from the model's. It must already be in the local Hugging Face cache: the engine runs offline.", "string", "quantization").adv().build(),
        p("tokenizer_mode", "yaml:tokenizer_mode", "Tokenizer mode", "auto uses the fast tokenizer when there is one; slow forces the Python implementation.", "enum", "quantization").default(Value::from("auto")).choices(TOKENIZER_MODES).adv().docs(SRC_LLM_ARGS).build(),
        p("custom_tokenizer", "--custom_tokenizer", "Custom tokenizer", "A built-in tokenizer implementation by alias (deepseek_v32). Python import paths are not accepted here because they would run arbitrary code.", "enum", "quantization").choices(CUSTOM_TOKENIZERS).adv().lvl("expert").build(),
        p("trust_remote_code", "--trust_remote_code", "Trust remote code", "Let Hugging Face transformers run Python code shipped with the model repository. Only for repositories you have reviewed.", "bool", "quantization").default(Value::Bool(false)).lvl("advanced").build(),
        p("moe_backend", "yaml:moe_config.backend", "MoE kernel backend", "Kernel family for mixture-of-experts layers. CUTLASS is the default; the image's code raises an error for TRTLLM on sm_120, and the DEEPGEMM, CUTEDSL and WIDEEP paths check for sm_100/103.", "enum", "quantization").default(Value::from("CUTLASS")).choices(MOE_BACKENDS).adv().docs(SRC_LLM_ARGS).build(),
        p("moe_max_num_tokens", "yaml:moe_config.max_num_tokens", "MoE chunk size (tokens)", "If set, MoE layers process at most this many tokens at a time and loop over larger batches, trading speed for lower peak memory.", "int", "quantization").min(1.0).adv().mem().unit("tokens").docs(SRC_LLM_ARGS).lvl("expert").build(),
        p("nvfp4_gemm_backends", "yaml:nvfp4_gemm_config.allowed_backends", "NVFP4 GEMM backends", "Kernel backends the engine may pick from for NVFP4 matrix multiplies (cutlass, cublaslt, cutedsl, cuda_core). cutedsl is left out by default to keep start-up short.", "string_list", "quantization").adv().docs(SRC_LLM_ARGS).lvl("expert").build(),
        p("max_batch_size", "--max_batch_size", "Max batch size", "Most requests handled in one step. Default 2048; it also bounds the CUDA graph batch sizes.", "int", "scheduling").default(Value::from(2048_i64)).min(1.0).mem().unit("requests").lvl("basic").build(),
        p("max_num_tokens", "--max_num_tokens", "Max tokens per step", "Most tokens processed in one step, summed over all requests. It also sizes the working memory reserved at start-up. Default 8192.", "int", "scheduling").default(Value::from(8192_i64)).min(1.0).mem().unit("tokens").lvl("basic").build(),
        p("enable_chunked_prefill", "--enable_chunked_prefill", "Chunked prefill", "Split long prompts into chunks so the decode steps of other requests are not starved.", "bool", "scheduling").default(Value::Bool(false)).build(),
        p("enable_block_reuse", "yaml:kv_cache_config.enable_block_reuse", "KV block reuse (prefix caching)", "Reuse KV blocks between requests that share a prompt prefix (system prompts, agent loops). On by default; the engine switches it off itself for the FlashInfer attention backend.", "bool", "scheduling").default(Value::Bool(true)).docs(DOC_KV).lvl("basic").build(),
        p("enable_partial_reuse", "yaml:kv_cache_config.enable_partial_reuse", "Partial block reuse", "Also reuse a block when only some of its tokens match a cached prefix. On by default; only matters when block reuse is on.", "bool", "scheduling").default(Value::Bool(true)).adv().docs(DOC_KV).build(),
        p("capacity_scheduler_policy", "yaml:scheduler_config.capacity_scheduler_policy", "Capacity policy", "How requests are admitted when the KV cache runs short: GUARANTEED_NO_EVICT (the default) admits only what can finish, MAX_UTILIZATION packs more and may pause requests, STATIC_BATCH is a legacy mode.", "enum", "scheduling").default(Value::from("GUARANTEED_NO_EVICT")).choices(CAPACITY_POLICIES).adv().docs(SRC_LLM_ARGS).build(),
        p("context_chunking_policy", "yaml:scheduler_config.context_chunking_policy", "Chunking policy", "How chunked prefill shares each step between prompts: FIRST_COME_FIRST_SERVED (the default when chunking is on) or EQUAL_PROGRESS.", "enum", "scheduling").choices(CHUNKING_POLICIES).adv().docs(SRC_LLM_ARGS).lvl("expert").build(),
        p("disable_overlap_scheduler", "yaml:disable_overlap_scheduler", "Disable overlap scheduler", "Turn off overlapping CPU scheduling with GPU work. It is on by default for speed; switch it off only to debug.", "bool", "scheduling").default(Value::Bool(false)).adv().docs(DOC_OVERLAP).build(),
        p("stream_interval", "yaml:stream_interval", "Stream interval (steps)", "Send a streamed chunk every N steps instead of every step. Larger values cut CPU overhead at high concurrency but make streaming less smooth.", "int", "scheduling").default(Value::from(1_i64)).min(1.0).adv().docs(SRC_LLM_ARGS).unit("steps").build(),
        p("batch_wait_timeout_ms", "yaml:batch_wait_timeout_ms", "Batch wait (ms)", "Wait up to this long for a full batch before running a step. 0 never waits. Prototype.", "float", "scheduling").default(Value::from(0_i64)).min(0.0).adv().docs(SRC_LLM_ARGS).unit("ms").lvl("expert").build(),
        p("batch_wait_timeout_iters", "yaml:batch_wait_timeout_iters", "Batch wait (steps)", "Wait up to this many steps to gather new requests before running a batch. 0 never waits. Prototype.", "int", "scheduling").default(Value::from(0_i64)).min(0.0).adv().docs(SRC_LLM_ARGS).unit("steps").lvl("expert").build(),
        p("batch_wait_max_tokens_ratio", "yaml:batch_wait_max_tokens_ratio", "Batch wait token threshold", "Hold requests back until the waiting tokens reach this share of max_num_tokens. 0 turns it off. Prototype.", "float", "scheduling").default(Value::from(0_i64)).min(0.0).max(1.0).adv().docs(SRC_LLM_ARGS).lvl("expert").build(),
        p("num_postprocess_workers", "--num_postprocess_workers", "Post-processing workers", "Extra processes that turn raw results into OpenAI-format responses (detokenizing). 0 does it inside the server process; raise it for high streaming concurrency.", "int", "scheduling").default(Value::from(0_i64)).min(0.0).adv().build(),
        p("sampler_type", "yaml:sampler_type", "Sampler", "auto, TorchSampler or TRTLLMSampler. auto uses TorchSampler unless beam search is requested; TRTLLMSampler cannot be combined with MTP or EAGLE-3.", "enum", "scheduling").default(Value::from("auto")).choices(SAMPLER_TYPES).adv().docs(DOC_SAMPLING).lvl("expert").build(),
        p("tool_parser", "--tool_parser", "Tool-call parser", "Parser for the model's tool-call format. Only the six dialects the image ships are available; other model families cannot return structured tool calls.", "enum", "tools_reasoning").choices(TOOL_PARSERS).lvl("basic").build(),
        p("reasoning_parser", "--reasoning_parser", "Reasoning parser", "Splits the model's thinking from its answer into reasoning_content. deepseek-r1, qwen3 and nano-v3 all expect <think> ... </think> markers.", "enum", "tools_reasoning").choices(REASONING_PARSERS).lvl("basic").build(),
        p("chat_template", "--chat_template", "Chat template", "A Jinja chat template: an absolute file path inside the container, or the template text itself (it must contain braces or a newline, otherwise it is read as a file name).", "string", "tools_reasoning").adv().build(),
        p("guided_decoding_backend", "yaml:guided_decoding_backend", "Structured output backend", "Engine for JSON-schema and grammar-constrained output: xgrammar or llguidance. Unset turns structured output off.", "enum", "tools_reasoning").choices(GUIDED_BACKENDS).adv().docs(DOC_GUIDED).build(),
        p("enable_cuda_graphs", "yaml:cuda_graph_config", "CUDA graphs", "Replay recorded GPU work for decode steps. On by default (batch sizes up to 128); each graph can take up to about 200 MB. Off removes them, which costs decode speed.", "bool", "performance").default(Value::Bool(true)).mem().docs(DOC_COMPILE).lvl("basic").build(),
        p("cuda_graph_max_batch_size", "yaml:cuda_graph_config.max_batch_size", "CUDA graph max batch", "Largest batch size to record a CUDA graph for; the sizes below it are generated. 0 or unset means 128. Do not combine with explicit batch sizes.", "int", "performance").default(Value::from(0_i64)).min(0.0).adv().mem().unit("requests").docs(DOC_COMPILE).build(),
        p("cuda_graph_batch_sizes", "yaml:cuda_graph_config.batch_sizes", "CUDA graph batch sizes", "Exact batch sizes to record graphs for. Fewer sizes start faster and use less memory.", "string_list", "performance").adv().mem().unit("requests").docs(DOC_COMPILE).build(),
        p("cuda_graph_padding", "yaml:cuda_graph_config.enable_padding", "Pad batches to graph sizes", "Round a batch up to the nearest recorded graph size so the graph can be used. Usually a net win for speed.", "bool", "performance").default(Value::Bool(false)).adv().docs(DOC_COMPILE).build(),
        p("attention_backend", "yaml:attn_backend", "Attention backend", "TRTLLM is the default and recommended kernel family. FLASHINFER disables KV block reuse; VANILLA is a slow reference implementation; FLASHINFER_STAR_ATTENTION is for star-attention long context.", "enum", "performance").default(Value::from("TRTLLM")).choices(ATTENTION_BACKENDS).adv().docs(DOC_ATTENTION).build(),
        p("enable_torch_compile", "yaml:torch_compile_config", "torch.compile", "Compile the model with torch.compile. Prototype: longer start-up, sometimes faster steps.", "bool", "performance").default(Value::Bool(false)).adv().docs(DOC_COMPILE).lvl("expert").build(),
        p("torch_compile_piecewise_cuda_graph", "yaml:torch_compile_config.enable_piecewise_cuda_graph", "Piecewise CUDA graphs (prefill)", "With torch.compile, also record CUDA graphs for prompt processing. Needs torch.compile on.", "bool", "performance").default(Value::Bool(false)).adv().mem().docs(DOC_COMPILE).lvl("expert").build(),
        p("torch_compile_inductor", "yaml:torch_compile_config.enable_inductor", "Inductor backend", "With torch.compile, use the Inductor code generator. Needs torch.compile on.", "bool", "performance").default(Value::Bool(false)).adv().docs(DOC_COMPILE).lvl("expert").build(),
        p("enable_autotuner", "yaml:enable_autotuner", "Kernel autotuner", "Time candidate kernels during start-up warm-up and keep the fastest. The image describes it as a debugging switch: turning it off can slow the engine noticeably.", "bool", "performance").default(Value::Bool(true)).adv().docs(SRC_LLM_ARGS).lvl("expert").build(),
        p("spec_decoding_type", "yaml:speculative_config.decoding_type", "Speculative method", "MTP uses the model's own multi-token-prediction heads, Eagle uses an EAGLE-3 draft model, NGram guesses from the prompt, DraftTarget uses a small draft model, AUTO lets the engine pick.", "enum", "speculative").choices(SPEC_TYPES).mem().docs(DOC_SPEC).lvl("advanced").build(),
        p("spec_max_draft_len", "yaml:speculative_config.max_draft_len", "Draft tokens per step", "How many tokens the drafter proposes per step. Required for Eagle, NGram and DraftTarget; MTP takes it from the predict-layer count.", "int", "speculative").min(1.0).adv().docs(DOC_SPEC).build(),
        p("spec_draft_model", "yaml:speculative_config.speculative_model", "Draft model", "Hugging Face repository (already in the local cache) or path of the draft model for Eagle and DraftTarget.", "string", "speculative").adv().mem().docs(DOC_SPEC).build(),
        p("spec_num_nextn_predict_layers", "yaml:speculative_config.num_nextn_predict_layers", "MTP predict layers", "How many MTP heads to use for multi-token prediction (the model must have them). Default 1.", "int", "speculative").min(1.0).adv().docs(DOC_SPEC).build(),
        p("spec_ngram_max_matching_size", "yaml:speculative_config.max_matching_ngram_size", "NGram match length", "Longest run of tokens the NGram drafter searches for in earlier text. Required for NGram.", "int", "speculative").min(1.0).adv().docs(DOC_SPEC).build(),
        p("spec_eagle3_one_model", "yaml:speculative_config.eagle3_one_model", "EAGLE-3 single engine", "Run the EAGLE-3 drafter inside the same engine as the target model (default) instead of as a second engine.", "bool", "speculative").default(Value::Bool(true)).adv().docs(DOC_SPEC).lvl("expert").build(),
        p("spec_max_concurrency", "yaml:speculative_config.max_concurrency", "Speculation batch limit", "Stop speculating when more than this many requests run at once, because drafting stops paying off at large batches.", "int", "speculative").min(1.0).adv().unit("requests").docs(DOC_SPEC).build(),
        p("speculative_config", "yaml:speculative_config", "Speculative config (JSON)", "Full speculative-decoding config as a JSON object that includes \"decoding_type\", for options the shortcut parameters do not cover. Cannot be combined with them.", "json", "speculative").adv().mem().docs(DOC_SPEC).lvl("expert").build(),
        p("log_level", "--log_level", "Log level", "Engine log verbosity. The console's start-up progress and fit checks read the info level, so lower levels hide them.", "enum", "network").default(Value::from("info")).choices(LOG_LEVELS).adv().build(),
        p("otlp_traces_endpoint", "--otlp_traces_endpoint", "OTLP traces endpoint", "Send OpenTelemetry traces to this URL (for example http://otel-collector:4317).", "string", "network").adv().lvl("expert").build(),
        p("enable_iter_perf_stats", "yaml:enable_iter_perf_stats", "Iteration statistics (live metrics)", "Collect per-step statistics (running and queued requests, KV cache use) that the /metrics endpoint serves. The engine default is off; the console turns it on so its metrics page works. Slight overhead.", "bool", "network").default(Value::Bool(false)).adv().docs(DOC_METRICS).build(),
        p("iter_stats_max_entries", "yaml:max_stats_len", "Iteration statistics kept", "How many of the newest per-step statistics the engine keeps between two reads of /metrics. The engine keeps 1000; the console keeps 100 when statistics are on, so one read stays far below the 512 KiB the console accepts per scrape.", "int", "network").default(Value::from(1000_i64)).min(1.0).adv().docs(DOC_METRICS).lvl("expert").build(),
        p("print_iter_log", "yaml:print_iter_log", "Log every step", "Write one log line per engine step with timings and batch sizes. Very chatty under load.", "bool", "network").default(Value::Bool(false)).adv().docs(SRC_LLM_ARGS).lvl("expert").build(),
        p("extra_llm_options", YAML_EXTRA_FLAG, "Extra engine options (JSON)", "Additional top-level settings for the generated engine options file, as a JSON object. Unknown names make the engine refuse to start; settings that have their own parameter are rejected here.", "json", "advanced").adv().docs(DOC_YAML).lvl("expert").build(),
        p("image", "@image", "Container image", "Pinned TensorRT-LLM image tag override (never :latest). The catalog was read from 1.2.1; other releases can differ.", "string", "advanced").default(Value::from(DEFAULT_IMAGE)).adv().build(),
    ]
}
// ---- generated by backend/parity/gen/catalog_ws_a2.py: end ----

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_compose_like_python() {
        assert_eq!(DEFAULT_IMAGE, format!("{IMAGE_REPO}:{TRTLLM_VERSION}"));
        assert_eq!(
            DOCS_BASE,
            format!("https://github.com/NVIDIA/TensorRT-LLM/blob/{SOURCE_COMMIT}/docs/source")
        );
        assert_eq!(TORCH_LLM_ARGS_FIELDS.len(), 83);
        assert_eq!(SUPPORTED_ARCHITECTURES.len(), 50);
        assert!(SUPPORTED_ARCHITECTURES.windows(2).all(|w| w[0] < w[1]));
        let _ = (DOC_COMBINATIONS, DOC_HARDWARE, IMAGE_DIGEST);
    }
}
