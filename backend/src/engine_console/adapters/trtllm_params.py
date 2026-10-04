"""TensorRT-LLM (`trtllm-serve`, PyTorch backend) parameter catalog: data only, no logic.

Provenance. Everything below was read from the pinned image `nvcr.io/nvidia/tensorrt-llm/release:1.2.1`
(amd64, manifest digest in `IMAGE_DIGEST`; source commit in `SOURCE_COMMIT`, taken from the image's own
`TRT_LLM_GIT_COMMIT`):

  * CLI flags, defaults and choices: `trtllm-serve serve --help` and the click objects in
    `tensorrt_llm/commands/serve.py` (the image's help text is committed as a test fixture).
  * YAML-only settings: the pydantic classes in `tensorrt_llm/llmapi/llm_args.py` (`TorchLlmArgs` and its nested
    `KvCacheConfig`, `CudaGraphConfig`, `MoeConfig`, ...), introspected inside the image. They are strict models
    (`extra="forbid"`): one unknown key makes the server refuse to start, so every YAML key emitted by the adapter
    is checked against the committed introspection fixture by the tests.
  * Hardware notes: the docs shipped inside the image (`/app/tensorrt_llm/docs/source`: supported-hardware,
    quantization hardware matrix, feature combination matrix, performance overview) and the `get_sm_version()`
    branches in the image's Python sources.

Pseudo-flag conventions (`ParamSpec.flag`; never emitted as argv):
  * ``--name``        a real option of `trtllm-serve serve`.
  * ``yaml:a.b.c``    a key path inside the generated `--extra_llm_api_options` file. The CLI has no flag for
                      these settings. The file is written by the node (`LaunchSpec.files`) and mounted read-only.
  * ``yaml:*``        free-form top-level keys merged into that file (the `extra_llm_options` escape hatch).
  * ``env:NAME``      a container environment variable.
  * ``@image``        overrides `LaunchSpec.image` (pinned tags only).

Every `yaml:` path must resolve in `TorchLlmArgs`; `YAML_NESTED` lists the nested config classes so the tests
can walk a path without importing the engine.
"""
from __future__ import annotations

from typing import Any, Literal

from .base import LearnLevel, ParamGroup, ParamSpec, ParamType

TRTLLM_VERSION = "1.2.1"
IMAGE_REPO = "nvcr.io/nvidia/tensorrt-llm/release"
DEFAULT_IMAGE = f"{IMAGE_REPO}:{TRTLLM_VERSION}"
# Manifest digest of the amd64 image the catalog was read from (the tag, not the digest, is what node policies match).
IMAGE_DIGEST = "sha256:33cd085b772947bd22b7273886539331420404e5d2a4a039945241945ff927b9"
# `TRT_LLM_GIT_COMMIT` in the image environment; docs links point at this exact commit so they match the pin.
SOURCE_COMMIT = "376f7e1bd8ed543f75014309e3fd4b237e9b0e73"

_REPO = f"https://github.com/NVIDIA/TensorRT-LLM/blob/{SOURCE_COMMIT}"
DOCS_BASE = f"{_REPO}/docs/source"
DOC_SERVE = f"{DOCS_BASE}/commands/trtllm-serve/trtllm-serve.rst"
DOC_YAML = f"{DOC_SERVE}#configuring-with-yaml-files"
DOC_METRICS = f"{DOC_SERVE}#metrics-endpoint"
DOC_KV = f"{DOCS_BASE}/features/kvcache.md"
DOC_PARALLEL = f"{DOCS_BASE}/features/parallel-strategy.md"
DOC_QUANT = f"{DOCS_BASE}/features/quantization.md"
DOC_SPEC = f"{DOCS_BASE}/features/speculative-decoding.md"
DOC_ATTENTION = f"{DOCS_BASE}/features/attention.md"
DOC_OVERLAP = f"{DOCS_BASE}/features/overlap-scheduler.md"
DOC_COMPILE = f"{DOCS_BASE}/features/torch_compile_and_piecewise_cuda_graph.md"
DOC_GUIDED = f"{DOCS_BASE}/features/guided-decoding.md"
DOC_SAMPLING = f"{DOCS_BASE}/features/sampling.md"
DOC_MULTIMODAL = f"{DOCS_BASE}/features/multi-modality.md"
DOC_COMBINATIONS = f"{DOCS_BASE}/features/feature-combination-matrix.md"
DOC_HARDWARE = f"{DOCS_BASE}/supported-hardware.md"
SRC_LLM_ARGS = f"{_REPO}/tensorrt_llm/llmapi/llm_args.py"

YAML_PREFIX = "yaml:"
YAML_EXTRA_FLAG = "yaml:*"
EXTRA_OPTIONS_PATH = "/config/extra-llm-api-options.yaml"

# ---- choices read from the image (see the module docstring) --------------------------------------------------
LOG_LEVELS = ["internal_error", "error", "warning", "info", "verbose", "debug", "trace"]
REASONING_PARSERS = ["deepseek-r1", "qwen3", "nano-v3"]
TOOL_PARSERS = ["qwen3", "qwen3_coder", "kimi_k2", "deepseek_v3", "deepseek_v31", "deepseek_v32"]
KV_CACHE_DTYPES = ["auto", "fp8", "nvfp4"]                       # model_loader._VALID_KV_CACHE_DTYPES
MAMBA_CACHE_DTYPES = ["auto", "float16", "bfloat16", "float32"]
MOE_BACKENDS = ["CUTLASS", "CUTEDSL", "WIDEEP", "TRTLLM", "DEEPGEMM", "VANILLA", "TRITON"]
# get_attention_backend() accepts these four; any other string silently falls back to TRTLLM, so the catalog
# is a closed enum instead of a free string.
ATTENTION_BACKENDS = ["TRTLLM", "FLASHINFER", "VANILLA", "FLASHINFER_STAR_ATTENTION"]
ALLREDUCE_STRATEGIES = ["AUTO", "NCCL", "UB", "MINLATENCY", "ONESHOT", "TWOSHOT", "LOWPRECISION", "MNNVL",
                        "NCCL_SYMMETRIC"]
SAMPLER_TYPES = ["auto", "TorchSampler", "TRTLLMSampler"]
CAPACITY_POLICIES = ["GUARANTEED_NO_EVICT", "MAX_UTILIZATION", "STATIC_BATCH"]
CHUNKING_POLICIES = ["FIRST_COME_FIRST_SERVED", "EQUAL_PROGRESS"]
GUIDED_BACKENDS = ["xgrammar", "llguidance"]
NVFP4_GEMM_BACKENDS = ["cutlass", "cublaslt", "cutedsl", "cuda_core"]
# The schema also lists "ray", but the image ships no ray package, so only rpc can work.
ORCHESTRATORS = ["rpc"]
TOKENIZER_MODES = ["auto", "slow"]
CUSTOM_TOKENIZERS = ["deepseek_v32"]          # the one built-in alias; import paths would load arbitrary code
# `decoding_type` values of DecodingBaseConfig.from_dict that work on the PyTorch backend and need no Python
# object: MTP, Eagle (EAGLE-3), NGram, DraftTarget, AUTO. (Medusa is TensorRT-backend only, SaveState writes
# files, UserProvided needs a drafter object, Lookahead belongs to the TensorRT backend.)
SPEC_TYPES = ["MTP", "Eagle", "NGram", "DraftTarget", "AUTO"]

# ---- options of `trtllm-serve serve` the adapter deliberately does not expose --------------------------------
# Every long option of the pinned `serve --help` is either a catalog flag or listed here with the reason
# (tests/trtllm/test_catalog.py enforces it, so a new image cannot add a flag silently).
EXCLUDED_SERVE_FLAGS: dict[str, str] = {
    "--host": "fixed to 0.0.0.0 inside the container; the console publishes the port to loopback only",
    "--port": "fixed to the adapter's default_port; the console maps it",
    "--backend": "always the PyTorch backend: the TensorRT-engine backend needs prebuilt engines the console "
                 "cannot manage",
    "--custom_module_dirs": "imports arbitrary Python from a directory",
    "--config": "generated by the adapter from the yaml: parameters",
    "--extra_llm_api_options": "generated by the adapter from the yaml: parameters",
    "--gpus_per_node": "auto-detected from the GPUs the container sees; only matters across machines",
    "--moe_cluster_parallel_size": "parsed by click but dropped by serve() in 1.2.1 (it is not a named argument of "
                                   "get_llm_args, whose extra keywords serve() discards); the YAML key of the same "
                                   "name is honoured and is what the moe_cluster_parallel_size parameter uses",
    "--fail_fast_on_attention_window_too_large": "read only by the TensorRT-engine backend (llmapi/llm.py builds its "
                                                 "ExecutorConfig with it); the PyTorch backend never reads it and "
                                                 "lowers an oversized attention window with a warning instead",
    "--metadata_server_config_file": "disaggregated serving only",
    "--server_role": "disaggregated serving only",
    "--disagg_cluster_uri": "disaggregated serving only",
}

# ---- YAML structure of the pinned image ----------------------------------------------------------------------
# Top-level fields of TorchLlmArgs (83), as introspected in the image.
TORCH_LLM_ARGS_FIELDS: tuple[str, ...] = (
    "allreduce_strategy", "attention_dp_config", "attn_backend", "backend",
    "batch_wait_max_tokens_ratio", "batch_wait_timeout_iters", "batch_wait_timeout_ms",
    "batched_logits_processor", "cache_transceiver_config", "checkpoint_format", "checkpoint_loader",
    "context_parallel_size", "cp_config", "cuda_graph_config", "custom_tokenizer", "decoding_config",
    "disable_flashinfer_sampling", "disable_overlap_scheduler", "dtype", "enable_attention_dp",
    "enable_autotuner", "enable_chunked_prefill", "enable_iter_perf_stats", "enable_iter_req_stats",
    "enable_layerwise_nvtx_marker", "enable_lm_head_tp_in_adp", "enable_lora", "enable_min_latency",
    "enable_sleep", "env_overrides", "fail_fast_on_attention_window_too_large",
    "force_dynamic_quantization", "garbage_collection_gen0_threshold", "gather_generation_logits",
    "gpus_per_node", "guided_decoding_backend", "iter_stats_max_iterations", "kv_cache_config",
    "kv_connector_config", "load_format", "lora_config", "max_batch_size", "max_beam_width",
    "max_input_len", "max_num_tokens", "max_seq_len", "max_stats_len", "mm_encoder_only", "model",
    "moe_cluster_parallel_size", "moe_config", "moe_expert_parallel_size", "moe_tensor_parallel_size",
    "mpi_session", "num_postprocess_workers", "nvfp4_gemm_config", "orchestrator_type",
    "otlp_traces_endpoint", "peft_cache_config", "perf_metrics_max_requests", "pipeline_parallel_size",
    "postprocess_tokenizer_dir", "pp_partition", "print_iter_log", "ray_placement_config",
    "ray_worker_extension_cls", "reasoning_parser", "request_stats_max_iterations",
    "return_perf_metrics", "revision", "sampler_force_async_worker", "sampler_type",
    "scheduler_config", "skip_tokenizer_init", "sparse_attention_config", "speculative_config",
    "stream_interval", "tensor_parallel_size", "tokenizer", "tokenizer_mode", "tokenizer_revision",
    "torch_compile_config", "trust_remote_code",
)
# Nested config classes a yaml: path may descend into: top-level field -> class name in the fixture.
YAML_NESTED: dict[str, str] = {
    "kv_cache_config": "KvCacheConfig", "cuda_graph_config": "CudaGraphConfig", "moe_config": "MoeConfig",
    "scheduler_config": "SchedulerConfig", "torch_compile_config": "TorchCompileConfig",
    "nvfp4_gemm_config": "Nvfp4GemmConfig", "attention_dp_config": "AttentionDpConfig",
}
# Top-level YAML keys that `extra_llm_options` may NOT set, with the reason. The rest of TORCH_LLM_ARGS_FIELDS
# not claimed by a catalog parameter is allowed (the engine validates the values).
EXTRA_OPTIONS_DENIED: dict[str, str] = {
    "model": "set by the launch (the model you picked)",
    "backend": "always pytorch",
    "mpi_session": "an in-process object, not a setting",
    "checkpoint_loader": "needs a Python object",
    "batched_logits_processor": "needs a Python object",
    "env_overrides": "sets arbitrary environment variables inside the engine process",
    "kv_connector_config": "imports a Python module by name",
    "ray_worker_extension_cls": "imports a Python class by name",
    "ray_placement_config": "needs Ray objects",
    "cache_transceiver_config": "disaggregated serving only",
    "lora_config": "needs adapter directories the console does not mount",
    "peft_cache_config": "only meaningful with LoRA adapters the console does not mount",
    "enable_lora": "needs adapter directories the console does not mount",
    "postprocess_tokenizer_dir": "managed by the engine",
    "gpus_per_node": "auto-detected",
}

# PyTorch-backend model classes registered in the image (`MODEL_CLASS_MAPPING`). `AutoModelForCausalLM.from_config`
# raises "Unknown architecture" for anything else, using `config.json` `architectures[0]`.
SUPPORTED_ARCHITECTURES: frozenset[str] = frozenset({
    "BertForSequenceClassification", "CLIPVisionModel", "DeciLMForCausalLM", "DeepseekV32ForCausalLM",
    "DeepseekV3ForCausalLM", "EAGLE3LlamaForCausalLM", "Eagle3DeepSeekV3ForCausalLM",
    "Exaone4ForCausalLM", "ExaoneMoEForCausalLM", "Gemma3ForCausalLM",
    "Gemma3ForConditionalGeneration", "Glm4MoeForCausalLM", "GptOssForCausalLM",
    "HCXVisionForCausalLM", "HCXVisionModel", "HunYuanDenseV1ForCausalLM", "HunYuanMoEV1ForCausalLM",
    "Llama4ForConditionalGeneration", "LlamaForCausalLM", "LlavaLlamaModel",
    "LlavaNextForConditionalGeneration", "MTPDraftModelForCausalLM", "MiniMaxM2ForCausalLM",
    "Mistral3ForConditionalGeneration", "MistralForCausalLM", "MistralLarge3EagleForCausalLM",
    "MistralLarge3ForCausalLM", "MixtralForCausalLM", "MllamaForConditionalGeneration",
    "NemotronForCausalLM", "NemotronHForCausalLM", "NemotronH_Nano_VL_V2", "Phi3ForCausalLM",
    "Phi4MMForCausalLM", "PixtralForConditionalGeneration", "PixtralVisionModel", "Qwen2ForCausalLM",
    "Qwen2ForProcessRewardModel", "Qwen2ForRewardModel", "Qwen2MoeForCausalLM",
    "Qwen2VLForConditionalGeneration", "Qwen2_5_VLForConditionalGeneration", "Qwen3ForCausalLM",
    "Qwen3MoeForCausalLM", "Qwen3NextForCausalLM", "Qwen3VLForConditionalGeneration",
    "Qwen3VLMoeForConditionalGeneration", "SeedOssForCausalLM", "SiglipVisionModel",
    "Starcoder2ForCausalLM",
})


def _p(key: str, flag: str, label: str, help: str, type: ParamType, group: ParamGroup, *,
       default: Any = None, choices: list[str] | None = None, min: float | None = None,
       max: float | None = None, adv: bool = False, mem: bool = False, docs: str | None = None,
       unit: str | None = None, lvl: LearnLevel | None = None) -> ParamSpec:
    if docs is None:
        docs = DOC_SERVE + "#syntax" if flag.startswith("--") else DOC_YAML
    # YAML values are always emitted explicitly (true/false), so a boolean can force an engine default off.
    style: Literal["flag", "value"] = "value" if type == "bool" and flag.startswith(("yaml:", "env:")) else "flag"
    return ParamSpec(key=key, flag=flag, label=label, help=help, type=type, group=group, default=default,
                     choices=choices, min=min, max=max, advanced=adv, affects_memory=mem,
                     requires_restart=True, bool_style=style, docs_url=docs, unit=unit, level=lvl)


def build_catalog() -> list[ParamSpec]:
    c: list[ParamSpec] = []
    a = c.append
    # ---------------- memory ----------------
    a(_p("free_gpu_memory_fraction", "--free_gpu_memory_fraction", "KV cache share of free GPU memory",
         "Share of the GPU memory that is still free after the weights are loaded and a profiling pass has run, "
         "which the KV cache may claim. Default 0.9; must stay below 1.",
         "float", "memory", default=0.9, min=0.01, max=0.99, mem=True, docs=DOC_KV, lvl="basic"))
    a(_p("kv_cache_dtype", "yaml:kv_cache_config.dtype", "KV cache dtype",
         "Storage type of the KV cache: auto follows the checkpoint, fp8 about halves the cache for a small "
         "accuracy risk, nvfp4 needs a checkpoint prepared for it (not listed as supported on sm_120).",
         "enum", "memory", default="auto", choices=KV_CACHE_DTYPES, mem=True, docs=DOC_QUANT, lvl="basic"))
    a(_p("kv_cache_max_tokens", "yaml:kv_cache_config.max_tokens", "KV cache token cap",
         "Upper bound on the tokens held in the KV cache. If the free-memory fraction allows fewer, the smaller "
         "value wins.", "int", "memory", min=1, adv=True, mem=True, unit="tokens", docs=DOC_KV))
    a(_p("kv_cache_max_gib", "yaml:kv_cache_config.max_gpu_total_bytes", "KV cache size cap (GiB per GPU)",
         "Upper bound on the GPU memory each GPU gives the KV cache, in GiB; the smaller of this and the "
         "fraction-based size is used. 0 or unset means no cap.",
         "float", "memory", min=0, adv=True, mem=True, unit="GiB", docs=DOC_KV))
    a(_p("kv_host_cache_gib", "yaml:kv_cache_config.host_cache_size", "KV host offload (GiB per GPU)",
         "Host RAM, per GPU worker, that evicted KV blocks are copied to so they can be reused without "
         "recomputing. Unset or 0 turns offloading off.",
         "float", "memory", min=0, adv=True, mem=True, unit="GiB", docs=DOC_KV))
    a(_p("tokens_per_block", "yaml:kv_cache_config.tokens_per_block", "KV block size (tokens)",
         "Tokens stored in one KV cache block. Must be a power of two greater than 1; the default is 32.",
         "int", "memory", default=32, min=2, adv=True, mem=True, unit="tokens", docs=DOC_KV, lvl="expert"))
    a(_p("kv_max_attention_window", "yaml:kv_cache_config.max_attention_window", "Attention window per layer",
         "Sliding-window sizes in tokens, repeated across the layers, for models with windowed attention. "
         "Leave unset unless you know the model's layer pattern.",
         "string_list", "memory", adv=True, mem=True, unit="tokens", docs=DOC_KV, lvl="expert"))
    a(_p("mamba_ssm_cache_dtype", "yaml:kv_cache_config.mamba_ssm_cache_dtype", "Recurrent-state dtype",
         "Data type of the recurrent state of Mamba and linear-attention layers in hybrid models; auto follows "
         "the model config.", "enum", "memory", default="auto", choices=MAMBA_CACHE_DTYPES, adv=True, mem=True,
         docs=SRC_LLM_ARGS, lvl="expert"))
    # ---------------- parallelism ----------------
    a(_p("tensor_parallel_size", "--tensor_parallel_size", "Tensor parallel size",
         "Split every layer across this many GPUs. Without NVLink the GPUs exchange data over PCIe on every "
         "layer; NVIDIA's own RTX PRO 6000 benchmarks used pipeline parallelism instead.",
         "int", "parallelism", default=1, min=1, mem=True, docs=DOC_PARALLEL, lvl="basic"))
    a(_p("pipeline_parallel_size", "--pipeline_parallel_size", "Pipeline parallel size",
         "Split the model into this many consecutive stages, one per GPU. Needs less GPU-to-GPU traffic than "
         "tensor parallelism, but cannot be combined with MTP or EAGLE-3 speculation.",
         "int", "parallelism", default=1, min=1, mem=True, docs=DOC_PARALLEL, lvl="advanced"))
    a(_p("context_parallel_size", "--context_parallel_size", "Context parallel size",
         "Spread very long prompts over several GPUs. Multiplies the number of GPUs needed.",
         "int", "parallelism", default=1, min=1, adv=True, mem=True, docs=DOC_PARALLEL, lvl="expert"))
    a(_p("moe_expert_parallel_size", "--moe_expert_parallel_size", "MoE expert parallel size",
         "For mixture-of-experts models: how many GPUs the experts are spread over. Must divide the "
         "tensor-parallel size; leave unset for dense models.",
         "int", "parallelism", min=1, adv=True, docs=DOC_PARALLEL))
    a(_p("moe_tensor_parallel_size", "yaml:moe_tensor_parallel_size", "MoE tensor parallel size",
         "For MoE models: how many GPUs each expert's weights are sliced across. The product with the expert "
         "and cluster sizes must equal the tensor-parallel size.",
         "int", "parallelism", min=1, adv=True, docs=DOC_PARALLEL, lvl="expert"))
    # The CLI flag of the same name is parsed but dropped by serve() in 1.2.1 (see EXCLUDED_SERVE_FLAGS), so the
    # setting goes through the YAML key, which the engine does honour.
    a(_p("moe_cluster_parallel_size", "yaml:moe_cluster_parallel_size", "MoE cluster parallel size",
         "Expert cluster parallelism, an extra grouping inside the expert split. Only with expert parallel "
         "size 1.", "int", "parallelism", min=1, adv=True, docs=DOC_PARALLEL, lvl="expert"))
    a(_p("enable_attention_dp", "yaml:enable_attention_dp", "Attention data parallel",
         "Run the attention layers data-parallel (every GPU serves different requests) while the rest stays "
         "tensor-parallel. Mostly useful for large MoE models under heavy load.",
         "bool", "parallelism", default=False, adv=True, docs=DOC_PARALLEL))
    a(_p("allreduce_strategy", "yaml:allreduce_strategy", "All-reduce strategy",
         "How GPUs sum partial results after each layer. AUTO lets the engine choose; NCCL is the plain choice "
         "that always works.", "enum", "parallelism", default="AUTO", choices=ALLREDUCE_STRATEGIES, adv=True,
         docs=SRC_LLM_ARGS, lvl="expert"))
    a(_p("orchestrator_type", "yaml:orchestrator_type", "Worker orchestrator",
         "How the server launches and talks to its GPU worker processes. Unset uses MPI; rpc avoids MPI. "
         "Prototype in 1.2.1 (the image has no Ray, so the ray choice is left out).", "enum", "parallelism",
         choices=ORCHESTRATORS, adv=True, docs=SRC_LLM_ARGS, lvl="expert"))
    a(_p("nccl_p2p_disable", "env:NCCL_P2P_DISABLE", "Disable GPU peer-to-peer (NCCL)",
         "Sets NCCL_P2P_DISABLE=1 so GPUs stop copying directly to each other and go through host memory. Only "
         "for multi-GPU launches that hang because direct copies do not work.",
         "bool", "parallelism", default=False, adv=True, docs=DOC_PARALLEL, lvl="expert"))
    # ---------------- context ----------------
    a(_p("max_seq_len", "--max_seq_len", "Max sequence length",
         "Longest request (prompt plus answer) in tokens. Unset reads it from the model config; a larger value "
         "than the model supports is cut down by the engine.",
         "int", "context", min=1, mem=True, unit="tokens", lvl="basic"))
    a(_p("max_beam_width", "--max_beam_width", "Max beam width",
         "Largest beam-search width a request may use. Keep 1 unless you serve beam search; larger values cost "
         "memory.", "int", "context", default=1, min=1, adv=True, mem=True))
    a(_p("media_io_kwargs", "--media_io_kwargs", "Media I/O kwargs (JSON)",
         'Options for decoding images, video and audio, as a JSON object (for example {"image": {...}}). '
         "Multimodal models only.", "json", "context", adv=True, docs=DOC_MULTIMODAL))
    # ---------------- quantization / model loading ----------------
    a(_p("revision", "--revision", "Model revision",
         "Branch, tag or commit of the model in the Hugging Face cache. Pin it: an unpinned main can change "
         "between restarts.", "string", "quantization", lvl="advanced"))
    a(_p("tokenizer", "--tokenizer", "Tokenizer override",
         "Tokenizer repository or path if it differs from the model's. It must already be in the local "
         "Hugging Face cache: the engine runs offline.", "string", "quantization", adv=True))
    a(_p("tokenizer_mode", "yaml:tokenizer_mode", "Tokenizer mode",
         "auto uses the fast tokenizer when there is one; slow forces the Python implementation.",
         "enum", "quantization", default="auto", choices=TOKENIZER_MODES, adv=True, docs=SRC_LLM_ARGS))
    a(_p("custom_tokenizer", "--custom_tokenizer", "Custom tokenizer",
         "A built-in tokenizer implementation by alias (deepseek_v32). Python import paths are not accepted "
         "here because they would run arbitrary code.", "enum", "quantization", choices=CUSTOM_TOKENIZERS,
         adv=True, lvl="expert"))
    a(_p("trust_remote_code", "--trust_remote_code", "Trust remote code",
         "Let Hugging Face transformers run Python code shipped with the model repository. Only for "
         "repositories you have reviewed.", "bool", "quantization", default=False, lvl="advanced"))
    a(_p("moe_backend", "yaml:moe_config.backend", "MoE kernel backend",
         "Kernel family for mixture-of-experts layers. CUTLASS is the default; the image's code raises an error "
         "for TRTLLM on sm_120, and the DEEPGEMM, CUTEDSL and WIDEEP paths check for sm_100/103.",
         "enum", "quantization", default="CUTLASS", choices=MOE_BACKENDS, adv=True, docs=SRC_LLM_ARGS))
    a(_p("moe_max_num_tokens", "yaml:moe_config.max_num_tokens", "MoE chunk size (tokens)",
         "If set, MoE layers process at most this many tokens at a time and loop over larger batches, trading "
         "speed for lower peak memory.", "int", "quantization", min=1, adv=True, mem=True, unit="tokens",
         docs=SRC_LLM_ARGS, lvl="expert"))
    a(_p("nvfp4_gemm_backends", "yaml:nvfp4_gemm_config.allowed_backends", "NVFP4 GEMM backends",
         "Kernel backends the engine may pick from for NVFP4 matrix multiplies (cutlass, cublaslt, cutedsl, "
         "cuda_core). cutedsl is left out by default to keep start-up short.",
         "string_list", "quantization", adv=True, docs=SRC_LLM_ARGS, lvl="expert"))
    # ---------------- scheduling ----------------
    a(_p("max_batch_size", "--max_batch_size", "Max batch size",
         "Most requests handled in one step. Default 2048; it also bounds the CUDA graph batch sizes.",
         "int", "scheduling", default=2048, min=1, mem=True, unit="requests", lvl="basic"))
    a(_p("max_num_tokens", "--max_num_tokens", "Max tokens per step",
         "Most tokens processed in one step, summed over all requests. It also sizes the working memory "
         "reserved at start-up. Default 8192.", "int", "scheduling", default=8192, min=1, mem=True,
         unit="tokens", lvl="basic"))
    a(_p("enable_chunked_prefill", "--enable_chunked_prefill", "Chunked prefill",
         "Split long prompts into chunks so the decode steps of other requests are not starved.",
         "bool", "scheduling", default=False))
    a(_p("enable_block_reuse", "yaml:kv_cache_config.enable_block_reuse", "KV block reuse (prefix caching)",
         "Reuse KV blocks between requests that share a prompt prefix (system prompts, agent loops). On by "
         "default; the engine switches it off itself for the FlashInfer attention backend.",
         "bool", "scheduling", default=True, docs=DOC_KV, lvl="basic"))
    a(_p("enable_partial_reuse", "yaml:kv_cache_config.enable_partial_reuse", "Partial block reuse",
         "Also reuse a block when only some of its tokens match a cached prefix. On by default; only matters "
         "when block reuse is on.", "bool", "scheduling", default=True, adv=True, docs=DOC_KV))
    a(_p("capacity_scheduler_policy", "yaml:scheduler_config.capacity_scheduler_policy", "Capacity policy",
         "How requests are admitted when the KV cache runs short: GUARANTEED_NO_EVICT (the default) admits only "
         "what can finish, MAX_UTILIZATION packs more and may pause requests, STATIC_BATCH is a legacy mode.",
         "enum", "scheduling", default="GUARANTEED_NO_EVICT", choices=CAPACITY_POLICIES, adv=True,
         docs=SRC_LLM_ARGS))
    a(_p("context_chunking_policy", "yaml:scheduler_config.context_chunking_policy", "Chunking policy",
         "How chunked prefill shares each step between prompts: FIRST_COME_FIRST_SERVED (the default when "
         "chunking is on) or EQUAL_PROGRESS.", "enum", "scheduling", choices=CHUNKING_POLICIES, adv=True,
         docs=SRC_LLM_ARGS, lvl="expert"))
    a(_p("disable_overlap_scheduler", "yaml:disable_overlap_scheduler", "Disable overlap scheduler",
         "Turn off overlapping CPU scheduling with GPU work. It is on by default for speed; switch it off only "
         "to debug.", "bool", "scheduling", default=False, adv=True, docs=DOC_OVERLAP))
    a(_p("stream_interval", "yaml:stream_interval", "Stream interval (steps)",
         "Send a streamed chunk every N steps instead of every step. Larger values cut CPU overhead at high "
         "concurrency but make streaming less smooth.", "int", "scheduling", default=1, min=1, adv=True,
         docs=SRC_LLM_ARGS, unit="steps"))
    a(_p("batch_wait_timeout_ms", "yaml:batch_wait_timeout_ms", "Batch wait (ms)",
         "Wait up to this long for a full batch before running a step. 0 never waits. Prototype.",
         "float", "scheduling", default=0, min=0, adv=True, docs=SRC_LLM_ARGS, unit="ms", lvl="expert"))
    a(_p("batch_wait_timeout_iters", "yaml:batch_wait_timeout_iters", "Batch wait (steps)",
         "Wait up to this many steps to gather new requests before running a batch. 0 never waits. Prototype.",
         "int", "scheduling", default=0, min=0, adv=True, docs=SRC_LLM_ARGS, unit="steps", lvl="expert"))
    a(_p("batch_wait_max_tokens_ratio", "yaml:batch_wait_max_tokens_ratio", "Batch wait token threshold",
         "Hold requests back until the waiting tokens reach this share of max_num_tokens. 0 turns it off. "
         "Prototype.", "float", "scheduling", default=0, min=0, max=1, adv=True, docs=SRC_LLM_ARGS,
         lvl="expert"))
    a(_p("num_postprocess_workers", "--num_postprocess_workers", "Post-processing workers",
         "Extra processes that turn raw results into OpenAI-format responses (detokenizing). 0 does it inside "
         "the server process; raise it for high streaming concurrency.", "int", "scheduling", default=0,
         min=0, adv=True))
    a(_p("sampler_type", "yaml:sampler_type", "Sampler",
         "auto, TorchSampler or TRTLLMSampler. auto uses TorchSampler unless beam search is requested; "
         "TRTLLMSampler cannot be combined with MTP or EAGLE-3.", "enum", "scheduling", default="auto",
         choices=SAMPLER_TYPES, adv=True, docs=DOC_SAMPLING, lvl="expert"))
    # ---------------- tools & reasoning ----------------
    a(_p("tool_parser", "--tool_parser", "Tool-call parser",
         "Parser for the model's tool-call format. Only the six dialects the image ships are available; other "
         "model families cannot return structured tool calls.", "enum", "tools_reasoning",
         choices=TOOL_PARSERS, lvl="basic"))
    a(_p("reasoning_parser", "--reasoning_parser", "Reasoning parser",
         "Splits the model's thinking from its answer into reasoning_content. deepseek-r1, qwen3 and nano-v3 "
         "all expect <think> ... </think> markers.", "enum", "tools_reasoning", choices=REASONING_PARSERS,
         lvl="basic"))
    a(_p("chat_template", "--chat_template", "Chat template",
         "A Jinja chat template: an absolute file path inside the container, or the template text itself (it must "
         "contain braces or a newline, otherwise it is read as a file name).", "string", "tools_reasoning",
         adv=True))
    a(_p("guided_decoding_backend", "yaml:guided_decoding_backend", "Structured output backend",
         "Engine for JSON-schema and grammar-constrained output: xgrammar or llguidance. Unset turns "
         "structured output off.", "enum", "tools_reasoning", choices=GUIDED_BACKENDS, adv=True,
         docs=DOC_GUIDED))
    # ---------------- performance ----------------
    a(_p("enable_cuda_graphs", "yaml:cuda_graph_config", "CUDA graphs",
         "Replay recorded GPU work for decode steps. On by default (batch sizes up to 128); each graph can take "
         "up to about 200 MB. Off removes them, which costs decode speed.", "bool", "performance",
         default=True, mem=True, docs=DOC_COMPILE, lvl="basic"))
    a(_p("cuda_graph_max_batch_size", "yaml:cuda_graph_config.max_batch_size", "CUDA graph max batch",
         "Largest batch size to record a CUDA graph for; the sizes below it are generated. 0 or unset means "
         "128. Do not combine with explicit batch sizes.", "int", "performance", default=0, min=0, adv=True,
         mem=True, unit="requests", docs=DOC_COMPILE))
    a(_p("cuda_graph_batch_sizes", "yaml:cuda_graph_config.batch_sizes", "CUDA graph batch sizes",
         "Exact batch sizes to record graphs for. Fewer sizes start faster and use less memory.",
         "string_list", "performance", adv=True, mem=True, unit="requests", docs=DOC_COMPILE))
    a(_p("cuda_graph_padding", "yaml:cuda_graph_config.enable_padding", "Pad batches to graph sizes",
         "Round a batch up to the nearest recorded graph size so the graph can be used. Usually a net win "
         "for speed.", "bool", "performance", default=False, adv=True, docs=DOC_COMPILE))
    a(_p("attention_backend", "yaml:attn_backend", "Attention backend",
         "TRTLLM is the default and recommended kernel family. FLASHINFER disables KV block reuse; VANILLA is "
         "a slow reference implementation; FLASHINFER_STAR_ATTENTION is for star-attention long context.",
         "enum", "performance", default="TRTLLM", choices=ATTENTION_BACKENDS, adv=True, docs=DOC_ATTENTION))
    a(_p("enable_torch_compile", "yaml:torch_compile_config", "torch.compile",
         "Compile the model with torch.compile. Prototype: longer start-up, sometimes faster steps.",
         "bool", "performance", default=False, adv=True, docs=DOC_COMPILE, lvl="expert"))
    a(_p("torch_compile_piecewise_cuda_graph", "yaml:torch_compile_config.enable_piecewise_cuda_graph",
         "Piecewise CUDA graphs (prefill)",
         "With torch.compile, also record CUDA graphs for prompt processing. Needs torch.compile on.",
         "bool", "performance", default=False, adv=True, mem=True, docs=DOC_COMPILE, lvl="expert"))
    a(_p("torch_compile_inductor", "yaml:torch_compile_config.enable_inductor", "Inductor backend",
         "With torch.compile, use the Inductor code generator. Needs torch.compile on.",
         "bool", "performance", default=False, adv=True, docs=DOC_COMPILE, lvl="expert"))
    a(_p("enable_autotuner", "yaml:enable_autotuner", "Kernel autotuner",
         "Time candidate kernels during start-up warm-up and keep the fastest. The image describes it as a "
         "debugging switch: turning it off can slow the engine noticeably.", "bool", "performance",
         default=True, adv=True, docs=SRC_LLM_ARGS, lvl="expert"))
    # ---------------- speculative ----------------
    a(_p("spec_decoding_type", "yaml:speculative_config.decoding_type", "Speculative method",
         "MTP uses the model's own multi-token-prediction heads, Eagle uses an EAGLE-3 draft model, NGram "
         "guesses from the prompt, DraftTarget uses a small draft model, AUTO lets the engine pick.",
         "enum", "speculative", choices=SPEC_TYPES, mem=True, docs=DOC_SPEC, lvl="advanced"))
    a(_p("spec_max_draft_len", "yaml:speculative_config.max_draft_len", "Draft tokens per step",
         "How many tokens the drafter proposes per step. Required for Eagle, NGram and DraftTarget; MTP takes "
         "it from the predict-layer count.", "int", "speculative", min=1, adv=True, docs=DOC_SPEC))
    a(_p("spec_draft_model", "yaml:speculative_config.speculative_model", "Draft model",
         "Hugging Face repository (already in the local cache) or path of the draft model for Eagle and "
         "DraftTarget.", "string", "speculative", adv=True, mem=True, docs=DOC_SPEC))
    a(_p("spec_num_nextn_predict_layers", "yaml:speculative_config.num_nextn_predict_layers",
         "MTP predict layers",
         "How many MTP heads to use for multi-token prediction (the model must have them). Default 1.",
         "int", "speculative", min=1, adv=True, docs=DOC_SPEC))
    a(_p("spec_ngram_max_matching_size", "yaml:speculative_config.max_matching_ngram_size",
         "NGram match length",
         "Longest run of tokens the NGram drafter searches for in earlier text. Required for NGram.",
         "int", "speculative", min=1, adv=True, docs=DOC_SPEC))
    a(_p("spec_eagle3_one_model", "yaml:speculative_config.eagle3_one_model", "EAGLE-3 single engine",
         "Run the EAGLE-3 drafter inside the same engine as the target model (default) instead of as a "
         "second engine.", "bool", "speculative", default=True, adv=True, docs=DOC_SPEC, lvl="expert"))
    a(_p("spec_max_concurrency", "yaml:speculative_config.max_concurrency", "Speculation batch limit",
         "Stop speculating when more than this many requests run at once, because drafting stops paying off "
         "at large batches.", "int", "speculative", min=1, adv=True, unit="requests", docs=DOC_SPEC))
    a(_p("speculative_config", "yaml:speculative_config", "Speculative config (JSON)",
         'Full speculative-decoding config as a JSON object that includes "decoding_type", for options the '
         "shortcut parameters do not cover. Cannot be combined with them.", "json", "speculative", adv=True,
         mem=True, docs=DOC_SPEC, lvl="expert"))
    # ---------------- network / observability ----------------
    a(_p("log_level", "--log_level", "Log level",
         "Engine log verbosity. The console's start-up progress and fit checks read the info level, so lower "
         "levels hide them.", "enum", "network", default="info", choices=LOG_LEVELS, adv=True))
    a(_p("otlp_traces_endpoint", "--otlp_traces_endpoint", "OTLP traces endpoint",
         "Send OpenTelemetry traces to this URL (for example http://otel-collector:4317).", "string",
         "network", adv=True, lvl="expert"))
    a(_p("enable_iter_perf_stats", "yaml:enable_iter_perf_stats", "Iteration statistics (live metrics)",
         "Collect per-step statistics (running and queued requests, KV cache use) that the /metrics endpoint "
         "serves. The engine default is off; the console turns it on so its metrics page works. Slight "
         "overhead.", "bool", "network", default=False, adv=True, docs=DOC_METRICS))
    a(_p("iter_stats_max_entries", "yaml:max_stats_len", "Iteration statistics kept",
         "How many of the newest per-step statistics the engine keeps between two reads of /metrics. The engine "
         "keeps 1000; the console keeps 100 when statistics are on, so one read stays far below the 512 KiB the "
         "console accepts per scrape.", "int", "network", default=1000, min=1, adv=True, docs=DOC_METRICS,
         lvl="expert"))
    a(_p("print_iter_log", "yaml:print_iter_log", "Log every step",
         "Write one log line per engine step with timings and batch sizes. Very chatty under load.",
         "bool", "network", default=False, adv=True, docs=SRC_LLM_ARGS, lvl="expert"))
    # ---------------- advanced ----------------
    a(_p("extra_llm_options", YAML_EXTRA_FLAG, "Extra engine options (JSON)",
         "Additional top-level settings for the generated engine options file, as a JSON object. Unknown "
         "names make the engine refuse to start; settings that have their own parameter are rejected here.",
         "json", "advanced", adv=True, docs=DOC_YAML, lvl="expert"))
    a(_p("image", "@image", "Container image",
         f"Pinned TensorRT-LLM image tag override (never :latest). The catalog was read from {TRTLLM_VERSION}; "
         "other releases can differ.", "string", "advanced", default=DEFAULT_IMAGE, adv=True))
    return c
