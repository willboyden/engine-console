"""vLLM adapter: implements the `EngineAdapter` port for the vLLM OpenAI server (pinned image v0.27.1).

Where the facts below come from (details in `vllm_params.py`): the image's own ``vllm serve --help=all`` and python
source, read in a throw-away container started from the local image. Nothing in this module was checked against a
running model; the compatibility rules marked "image source" were read from code, not provoked.

Launch conventions (the LaunchSpec has no field for them, so the lifecycle service adds them):
  * The image entrypoint is already ``vllm serve``; `argv` starts with ``--model``.
  * docker flags NOT expressible in argv/env: ``--ipc=host`` (or a large ``--shm-size``; tensor-parallel workers use
    shared memory), GPU pinning by UUID, the model mounts, and the engine network.
  * The torch.compile / FlashInfer caches live in ``/root/.cache/vllm`` and ``/root/.cache/flashinfer``; the core mounts
    named volumes there so a restart does not recompile kernels.
  * HF_TOKEN is deliberately NOT placed in `LaunchSpec.env`: the core injects it by name only, so it never enters the
    spec, logs or the audit trail. `api_key` is carried in env (VLLM_API_KEY), never in argv.
  * Param keys whose catalog `flag` is ``env:NAME`` become container env vars; everything else becomes a CLI flag
    (see `_render`).
  * VLLM_SERVER_DEV_MODE is never set (it exposes an unauthenticated /collective_rpc).

Model formats: ``hf`` (a Hugging Face repo id, the HF cache mounted read-only) and ``plain`` (a model directory the core
mounts read-only at ``/models``; vLLM loads a local directory when given its path). The pinned image has no GGUF weight
loader, so ``gguf`` is not offered. Every local image of this engine is amd64, so ``images_by_arch`` stays empty.
"""
from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Callable
from typing import Any, Literal

from .base import (
    MODEL_MOUNT,
    Compat,
    EngineAdapter,
    Hardware,
    LaunchContext,
    LaunchSpec,
    ModelInfo,
    ParamSpec,
)
from .vllm_params import (
    ATTENTION_KV_DTYPES,
    ATTENTION_NOT_ON_SM120,
    DEFAULT_IMAGE,
    NARGS_KEYS,
    NEGATABLE_KEYS,
    build_catalog,
)
from .vllm_parse import Scrape, parse_log_line, scrape_to_canonical

_GIB = 1024 ** 3
_OUTSIDE_POOL_GIB = 2.5            # measured once, see `memory_model`
_NCCL_PER_EXTRA_RANK_GIB = 0.5     # not measured
# Approximate bytes per KV element (scales and metadata not included): a planning figure for the fit estimator.
_KV_BYTES = {"fp8": 1.0, "fp8_e4m3": 1.0, "fp8_e5m2": 1.0, "fp8_inc": 1.0, "fp8_ds_mla": 1.0,
             "fp8_per_token_head": 1.0, "int8_per_token_head": 1.0, "int4_per_token_head": 0.5, "nvfp4": 0.5625,
             "turboquant_3bit_nc": 0.4, "turboquant_4bit_nc": 0.5, "turboquant_k3v4_nc": 0.45,
             "turboquant_k8v4": 0.75}
# Parameters removed from the catalog, with the reason a saved profile that still carries them should read.
_RETIRED = {
    "image": "'image' is no longer a parameter: engine images are pinned in Settings, not per instance",
    "nvfp4_gemm_backend": "'nvfp4_gemm_backend' is gone: VLLM_NVFP4_GEMM_BACKEND does not exist in vLLM 0.27.1; "
                          "use linear_backend (flashinfer_b12x, flashinfer_cutlass, marlin, ...)",
    "flashinfer_moe_backend": "'flashinfer_moe_backend' is gone: VLLM_FLASHINFER_MOE_BACKEND does not exist in vLLM "
                              "0.27.1; use moe_backend (flashinfer_b12x, flashinfer_cutlass, marlin, ...)",
}
# Architecture / model-type fragments of hybrid models that keep a per-sequence Mamba or linear-attention state.
_HYBRID_MARKERS = ("nemotronh", "nemotron_h", "mamba", "jamba", "zamba", "bamba", "falconh1", "falcon_h1",
                   "qwen3next", "qwen3_next", "qwen3_5", "qwen3.5", "granitemoehybrid", "plamo2")
# repo-name regex -> tool-call parsers that match the model's dialect (first = recommended). Pairs come from the
# v0.27.1 "tool calling" docs page and, for the Qwen3 / Nemotron / GLM / Poolside rows, from parsers that load
# and serve those families. `qwen3_coder` and `qwen3_xml` are one class in the image; `glm45` and `glm47` as well.
_TOOL_DIALECTS: list[tuple[re.Pattern[str], tuple[str, ...]]] = [
    (re.compile(r"gpt-oss", re.I), ("openai",)),
    (re.compile(r"qwen3[._-]?(coder|next)|qwen3[._][5-9]", re.I), ("qwen3_xml", "qwen3_coder")),
    (re.compile(r"nemotron", re.I), ("qwen3_coder", "qwen3_xml")),
    (re.compile(r"qwen3(?![._\d])|qwen2\.?5|qwq", re.I), ("hermes",)),
    (re.compile(r"hermes", re.I), ("hermes",)),
    (re.compile(r"llama[-_]?4", re.I), ("llama4_pythonic", "llama4_json")),
    (re.compile(r"llama[-_]?3", re.I), ("llama3_json", "pythonic")),
    (re.compile(r"mistral|ministral|devstral", re.I), ("mistral",)),
    (re.compile(r"deepseek[-_]?v?3\.?2", re.I), ("deepseek_v32",)),
    (re.compile(r"deepseek[-_]?v?3\.?1", re.I), ("deepseek_v31",)),
    (re.compile(r"deepseek[-_]?v?4", re.I), ("deepseek_v4",)),
    (re.compile(r"deepseek[-_]?(v3|r1)", re.I), ("deepseek_v3",)),
    (re.compile(r"glm[-_]?(4\.[567]|5)", re.I), ("glm47", "glm45")),
    (re.compile(r"kimi[-_]?k2", re.I), ("kimi_k2",)),
    (re.compile(r"kimi[-_]?k3", re.I), ("kimi_k3",)),
    (re.compile(r"minimax[-_]?m2", re.I), ("minimax_m2",)),
    (re.compile(r"minimax[-_]?m3", re.I), ("minimax_m3",)),
    (re.compile(r"laguna", re.I), ("poolside_v1",)),
    (re.compile(r"granite[-_]?4", re.I), ("granite4",)),
    (re.compile(r"gemma[-_]?4", re.I), ("gemma4",)),
]
# repo-name regex -> reasoning parser (docs page "reasoning outputs" plus the families verified in use).
_REASONING_HINTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"gpt-oss", re.I), "openai_gptoss"), (re.compile(r"deepseek-r1|qwq", re.I), "deepseek_r1"),
    (re.compile(r"qwen3", re.I), "qwen3"), (re.compile(r"nemotron", re.I), "nemotron_v3"),
    (re.compile(r"glm[-_]?(4\.[567]|5)", re.I), "glm45"), (re.compile(r"kimi-k2-thinking", re.I), "kimi_k2"),
    (re.compile(r"laguna", re.I), "poolside_v1"), (re.compile(r"minimax[-_]?m2", re.I), "minimax_m2_append_think"),
    (re.compile(r"olmo-3", re.I), "olmo3"), (re.compile(r"gemma[-_]?4", re.I), "gemma4"),
]
_NON_REASONING_MARKERS = ("coder", "instruct-2507")


def _image_version(image: str) -> tuple[int, int, int] | None:
    m = re.search(r":v?(\d+)\.(\d+)\.(\d+)", image)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def _cc_major(hw: Hardware) -> int | None:
    m = re.match(r"^(\d+)\.\d+$", hw.compute_capability.strip())
    return int(m.group(1)) if m else None


def _tool_dialect(repo: str) -> tuple[str, ...] | None:
    for pattern, allowed in _TOOL_DIALECTS:
        if pattern.search(repo):
            return allowed
    return None


def _reasoning_hint(repo: str) -> str | None:
    for pattern, parser in _REASONING_HINTS:
        if pattern.search(repo):
            return parser
    return None


class VllmAdapter(EngineAdapter):
    id = "vllm"
    display_name = "vLLM"
    default_image = DEFAULT_IMAGE
    default_port = 8000
    model_formats = frozenset({"hf", "plain"})
    metric_prefixes: tuple[str, ...] = ("vllm:",)

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._catalog = build_catalog()
        self._by_key: dict[str, ParamSpec] = {p.key: p for p in self._catalog}
        self._clock = clock
        # rate state per served-model-name set: (t, prompt_total, generation_total)
        self._prev: dict[tuple[str, ...], tuple[float, float | None, float | None]] = {}

    # ------------------------------------------------------------------ catalog / presets
    def param_catalog(self) -> list[ParamSpec]:
        return [p.model_copy() for p in self._catalog]

    def presets(self) -> dict[str, dict[str, Any]]:
        return {
            # vLLM's own defaults plus a 64K window; safe on any model whose native length allows it.
            "balanced": {"gpu_memory_utilization": 0.90, "max_model_len": 65536,
                         "enable_prefix_caching": True, "performance_mode": "balanced"},
            "max-throughput": {"gpu_memory_utilization": 0.92, "max_model_len": 32768,
                               "max_num_seqs": 512, "max_num_batched_tokens": 16384,
                               "enable_prefix_caching": True, "enable_chunked_prefill": True,
                               "async_scheduling": True, "performance_mode": "throughput"},
            "low-latency": {"gpu_memory_utilization": 0.90, "max_model_len": 16384,
                            "max_num_seqs": 8, "max_num_batched_tokens": 2048,
                            "enable_prefix_caching": True, "async_scheduling": True,
                            "performance_mode": "interactivity", "stream_interval": 1},
            # fp8 KV doubles token capacity; few sequences because each one is big. Attention backend left on auto:
            # on sm_120 forcing FLASH_ATTN with fp8 KV is refused (see compatibility()).
            "long-context": {"gpu_memory_utilization": 0.92, "max_model_len": 262144,
                             "kv_cache_dtype": "fp8", "max_num_seqs": 8,
                             "max_num_batched_tokens": 8192, "enable_prefix_caching": True,
                             "enable_chunked_prefill": True},
            # hermes suits Hermes-family and Qwen2.5-style models; change the parser for other families
            # (compatibility() suggests one).
            "tool-agent": {"gpu_memory_utilization": 0.90, "max_model_len": 65536, "max_num_seqs": 32,
                           "enable_prefix_caching": True, "enable_auto_tool_choice": True,
                           "tool_call_parser": "hermes"},
            # qwen3 + qwen3_xml is the usual pairing for Qwen3.x reasoning models.
            "reasoning": {"gpu_memory_utilization": 0.90, "max_model_len": 131072, "max_num_seqs": 16,
                          "enable_prefix_caching": True, "enable_chunked_prefill": True,
                          "reasoning_parser": "qwen3", "enable_auto_tool_choice": True,
                          "tool_call_parser": "qwen3_xml"},
        }

    # ------------------------------------------------------------------ validation
    def validate(self, params: dict[str, Any]) -> list[str]:
        errs: list[str] = []
        for key, val in params.items():
            spec = self._by_key.get(key)
            if spec is None:
                errs.append(_RETIRED.get(key, f"unknown parameter '{key}'"))
                continue
            if val is None:
                continue
            errs.extend(self._check_value(spec, val))
        if errs:
            return errs
        g = params.get
        if g("enable_auto_tool_choice") and not g("tool_call_parser"):
            errs.append("enable_auto_tool_choice requires tool_call_parser (vLLM refuses to start otherwise)")
        if g("lora_modules") and not g("enable_lora"):
            errs.append("lora_modules requires enable_lora=true")
        # vLLM merges the spec_* shortcuts into speculative_config and only refuses a key given in both places
        # (engine/arg_utils.py: "--spec-method and --speculative-config['method'] are mutually exclusive").
        sc = g("speculative_config")
        if isinstance(sc, str):
            try:
                sc = json.loads(sc)
            except json.JSONDecodeError:
                sc = None
        if isinstance(sc, dict):
            for key, short in (("method", "spec_method"), ("model", "spec_model"),
                               ("num_speculative_tokens", "spec_tokens")):
                if g(short) is not None and key in sc:
                    errs.append(f"{short} and speculative_config['{key}'] are mutually exclusive (vLLM refuses to start)")
        cpu_loras, loras = g("max_cpu_loras"), g("max_loras")
        if cpu_loras and loras and cpu_loras < loras:
            errs.append("max_cpu_loras must be >= max_loras")
        if g("allowed_local_media_path") == "/":
            errs.append("allowed_local_media_path must not be '/': any prompt could read any container file")
        return errs

    def _check_value(self, spec: ParamSpec, val: Any) -> list[str]:
        k, t = spec.key, spec.type
        if t == "bool":
            return [] if isinstance(val, bool) else [f"{k}: expected true/false"]
        if t in ("int", "float"):
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
                return [f"{k}: expected a number"]
            if t == "int" and int(val) != val:
                return [f"{k}: expected an integer"]
            if spec.min is not None and val < spec.min:
                return [f"{k}: must be >= {spec.min:g}"]
            if spec.max is not None and val > spec.max:
                return [f"{k}: must be <= {spec.max:g}"]
            return []
        if t == "enum":
            if str(val) not in (spec.choices or []):
                return [f"{k}: '{val}' is not one of {', '.join(spec.choices or [])}"]
            return []
        if t == "string":
            if not isinstance(val, str) or not val.strip():
                return [f"{k}: expected a non-empty string"]
            if val.startswith("-") and not spec.flag.startswith("env:"):
                return [f"{k}: a value must not start with '-' (it would be read as another flag)"]
            return []
        if t == "string_list":
            if not isinstance(val, list) or not val or not all(isinstance(x, (str, int)) and str(x).strip()
                                                               and not isinstance(x, bool) for x in val):
                return [f"{k}: expected a non-empty list of strings"]
            if k == "cudagraph_capture_sizes" and not all(str(x).isdigit() and int(x) > 0 for x in val):
                return [f"{k}: entries must be positive integers"]
            if any(str(x).startswith("-") for x in val):
                return [f"{k}: entries must not start with '-' (they would be read as another flag)"]
            return []
        if t == "json":
            try:
                obj = json.loads(val) if isinstance(val, str) else val
            except json.JSONDecodeError as e:
                return [f"{k}: invalid JSON ({e.msg})"]
            if not isinstance(obj, (dict, list)):
                return [f"{k}: expected a JSON object"]
            return []
        return []

    # ------------------------------------------------------------------ launch
    def _render(self, spec: ParamSpec, val: Any) -> list[str]:
        """CLI tokens for one param. bool -> --x / --no-x (only when the flag is negatable)."""
        if spec.type == "bool":
            if val:
                return [spec.flag]
            return ["--no-" + spec.flag[2:]] if spec.key in NEGATABLE_KEYS else []
        if spec.type == "string_list":
            items = [str(x) for x in val]
            if spec.key in NARGS_KEYS:
                return [spec.flag, *items]
            return [tok for x in items for tok in (spec.flag, x)]
        if spec.type == "json":
            obj = json.loads(val) if isinstance(val, str) else val
            return [spec.flag, json.dumps(obj, separators=(",", ":"))]
        if spec.type == "float":
            return [spec.flag, format(float(val), "g")]
        return [spec.flag, str(int(val) if spec.type == "int" else val)]

    def build_launch(self, model: str, params: dict[str, Any], hw: Hardware, *,
                     served_name: str, hf_cache_container_path: str) -> LaunchSpec:
        """The v1 entry point: a Hugging Face repo id. Builds a context and defers to `build_launch_ctx`."""
        return self.build_launch_ctx(LaunchContext(
            model=model, model_format="hf", served_name=served_name,
            hf_cache_container_path=hf_cache_container_path, params=params, hw=hw))

    def build_launch_ctx(self, ctx: LaunchContext) -> LaunchSpec:
        if ctx.model_format not in self.model_formats:
            raise ValueError(f"vLLM cannot load {ctx.model_format} models "
                             f"(it loads: {', '.join(sorted(self.model_formats))})")
        errs = self.validate(ctx.params)
        if errs:
            raise ValueError("; ".join(errs))
        plain = ctx.model_format == "plain"
        # A plain model is a directory the core mounts read-only at MODEL_MOUNT; vLLM takes its path as the model.
        model_arg = (ctx.model_path or MODEL_MOUNT) if plain else ctx.model
        if not model_arg or model_arg.startswith("-"):
            raise ValueError("invalid model identifier")
        argv = ["--model", model_arg, "--host", "0.0.0.0",  # noqa: S104 (container-internal; core publishes to loopback)
                "--port", str(self.default_port)]
        # Telemetry off (deny-by-default egress): vLLM honours VLLM_NO_USAGE_STATS and DO_NOT_TRACK.
        env = {"VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1", "HF_HUB_DISABLE_TELEMETRY": "1"}
        # JIT kernels compile for the architecture list given here; the host's real capability avoids building for others.
        if re.fullmatch(r"\d+\.\d+", ctx.hw.compute_capability.strip()):
            env["TORCH_CUDA_ARCH_LIST"] = ctx.hw.compute_capability.strip()
        if plain:
            env["HF_HUB_OFFLINE"] = "1"          # a local directory: never wait on the hub
        else:
            env["HF_HOME"] = ctx.hf_cache_container_path
        params = ctx.params
        names = [ctx.served_name] + [n for n in params.get("served_model_name") or [] if n != ctx.served_name]
        argv += ["--served-model-name", *names]
        for spec in self._catalog:
            val = params.get(spec.key)
            if val is None or spec.key == "served_model_name":
                continue
            if spec.key == "revision" and plain:
                continue                          # a directory has no hub revision
            if spec.flag.startswith("env:"):
                env[spec.flag[4:]] = ("1" if val else "0") if spec.type == "bool" else str(val)
            else:
                argv += self._render(spec, val)
        return LaunchSpec(image=self.default_image, argv=argv, env=env, container_port=self.default_port,
                          health_path="/health", metrics_path="/metrics", needs_hf_cache=not plain)

    # ------------------------------------------------------------------ compatibility
    def compatibility(self, model: ModelInfo, params: dict[str, Any], hw: Hardware) -> list[Compat]:
        out: list[Compat] = []

        def add(lvl: Literal["ok", "warn", "block"], code: str, msg: str) -> None:
            out.append(Compat(level=lvl, code=code, message=msg))

        g = params.get
        repo = model.repo_id.lower()
        arch_blob = " ".join([model.model_type or "", *model.architectures]).lower()
        quant = (model.quantization or "").lower()
        cc = _cc_major(hw)
        is_nvfp4 = "nvfp4" in quant or ("fp4" in quant and "mx" not in quant) or "nvfp4" in repo
        is_mxfp4 = "mxfp4" in quant or "mxfp4" in repo
        is_gpt_oss = "gpt-oss" in repo or any("GptOss" in a for a in model.architectures)
        moe_backend = g("moe_backend")
        tp, pp, dp = g("tensor_parallel_size") or 1, g("pipeline_parallel_size") or 1, g("data_parallel_size") or 1
        ver = _image_version(self.default_image)

        # --- quantization versus GPU generation --------------------------------------------------------------------
        if cc == 12 and is_nvfp4 and model.is_moe and moe_backend in (None, "auto"):
            add("warn", "nvfp4_moe_sm120",
                "NVFP4 MoE on sm_120 can fall back to a slow Marlin kernel in vLLM. Use the FP8 build, run it on "
                "SGLang, or time moe_backend flashinfer_b12x (the sm_12x kernel in the vLLM 0.27.1 help), "
                "cutlass and marlin.")
        if cc == 12 and is_mxfp4 and model.is_moe and not is_gpt_oss:
            add("block", "mxfp4_moe_sm120",
                "MXFP4 MoE asserts on sm_120 except gpt-oss. Use an FP8 build of this model.")
        if is_gpt_oss:
            add("warn", "gpt_oss_prefer_ollama",
                "gpt-oss ships native MXFP4 weights: leave quantization unset. If vLLM does not start on your GPU "
                "generation, run the model with llama.cpp or Ollama instead.")
        if model.gated:
            add("warn", "gated_needs_hf_token",
                "Gated model: set HF_TOKEN (Settings) and accept the license on the model page before "
                "downloading or starting.")
        # --- parallel sizes ----------------------------------------------------------------------------------------
        world = tp * pp * dp
        if world > len(hw.gpu_ids):
            add("block", "tp_exceeds_gpus",
                f"tensor*pipeline*data parallel = {world} but only {len(hw.gpu_ids)} GPU(s) selected. "
                "Lower the parallel sizes or select more GPUs.")
        elif tp > 1:
            add("warn", "tp_over_pcie",
                "Tensor parallelism all-reduces on every layer; without NVLink that traffic crosses PCIe (workstation "
                "cards usually have no NVLink). Prefer one model per card unless it does not fit.")
        heads, kv = model.num_attention_heads, model.num_kv_heads
        if tp > 1 and heads and heads % tp:
            add("block", "tp_heads_indivisible",
                f"{heads} attention heads are not divisible by tensor_parallel_size={tp}. Pick a divisor.")
        elif tp > 1 and kv and kv % tp and tp % kv:
            add("block", "tp_kv_heads_indivisible",
                f"{kv} KV heads and tensor_parallel_size={tp}: one must divide the other.")
        # --- KV cache dtype and attention backend (image source: supported_kv_cache_dtypes, supports_compute_capability)
        kvd, backend = g("kv_cache_dtype"), g("attention_backend")
        if kvd and backend in ATTENTION_KV_DTYPES:
            if cc == 12 and str(kvd).startswith("fp8") and backend in ("FLASH_ATTN", "FLASH_ATTN_DIFFKV"):
                add("block", "kv_fp8_flash_attn",
                    "On sm_120 FLASH_ATTN runs FlashAttention 2, which cannot use an fp8 KV cache; vLLM stops with "
                    "'Selected backend FLASH_ATTN is not valid for this configuration'. Set attention_backend to "
                    "FLASHINFER (or leave it unset), or set kv_cache_dtype to auto.")
            elif str(kvd) not in ATTENTION_KV_DTYPES[str(backend)]:
                add("block", "kv_dtype_backend_mismatch",
                    f"attention_backend {backend} does not support kv_cache_dtype {kvd} (it supports: "
                    f"{', '.join(sorted(ATTENTION_KV_DTYPES[str(backend)]))}). Pick another backend or dtype.")
        if kvd and str(kvd).startswith("fp8") and quant in ("", "none"):
            add("warn", "kv_fp8_accuracy",
                "fp8 KV cache trades accuracy for capacity; vLLM logs 'Using uncalibrated q_scale 1.0 and/or "
                "prob_scale 1.0 with fp8 attention' when the checkpoint ships no scales. Compare outputs on your "
                "task before adopting it.")
        if cc == 12 and backend in ATTENTION_NOT_ON_SM120:
            add("block", "mla_backend_arch",
                f"{backend} does not support compute capability 12.0 in vLLM 0.27.1. Use TRITON_MLA (or "
                "FLASHINFER_MLA_SPARSE_SM120 for sparse-MLA models), or leave attention_backend unset.")
        # --- kernel backends ----------------------------------------------------------------------------------------
        if moe_backend in ("aiter", "flydsl") or g("linear_backend") in ("aiter", "xpu", "xpu_woq") \
                or (moe_backend == "hpc" and cc is not None and cc != 9):
            add("block", "kernel_backend_not_for_gpu",
                "The chosen moe_backend or linear_backend is not built for this GPU: aiter and flydsl are AMD-only, "
                "xpu is Intel-only and hpc is FP8 on Hopper only (vLLM 0.27.1 help). Use auto, cutlass, marlin or "
                "flashinfer_b12x.")
        if moe_backend == "deep_gemm" and quant not in ("fp8", "fp8_block"):
            add("warn", "deep_gemm_fp8_only",
                "The deep_gemm MoE backend only runs FP8 block-quantized checkpoints (vLLM 0.27.1 help). Use an FP8 "
                "build, or leave moe_backend on auto.")
        if g("enable_expert_parallel") and not model.is_moe:
            add("warn", "ep_dense_model", "enable_expert_parallel has no effect on a dense model.")
        if g("prefix_caching_hash_algo") in ("xxhash", "xxhash_cbor"):
            add("warn", "prefix_hash_xxhash_missing",
                "The xxhash prefix-cache hash needs the optional xxhash package, which the vLLM 0.27.1 image does not "
                "include; start-up fails unless your image has it. Use sha256 (the default) or sha256_cbor.")
        q = g("quantization")
        if q and model.quantization and str(q) not in quant and quant not in str(q) and \
                model.quantization.lower() not in ("none",):
            add("block", "quant_override_prequantized",
                f"The checkpoint is already {model.quantization}; forcing quantization={q} misloads it. "
                "Unset quantization and let vLLM auto-detect.")
        if (not q and not model.quantization and model.weight_bytes and hw.gpu_total_gib
                and model.weight_bytes / _GIB > 0.9 * min(hw.gpu_total_gib)):
            add("warn", "fp8_recommended",
                "Unquantized weights nearly fill a GPU. Set quantization=fp8 (the usual "
                "choice) or pick an FP8/NVFP4-dense build.")
        # --- context length ----------------------------------------------------------------------------------------
        pos = model.max_position_embeddings
        mml = g("max_model_len")
        if mml and pos and mml > pos:
            if g("hf_overrides"):
                add("warn", "yarn_short_context_quality",
                    "Context extended past the trained length via hf_overrides (static YaRN scales all "
                    "positions): short-prompt quality can drop. Measure before adopting.")
            elif not g("allow_long_max_model_len"):
                add("block", "max_len_exceeds_model",
                    f"max_model_len {mml} exceeds the model's {pos}. Extend with a YaRN hf_overrides "
                    "block or lower max_model_len.")
        # --- memory share (image source: request_memory in v1/worker/utils.py) --------------------------------------
        util = g("gpu_memory_utilization") or 0.92
        if not g("kv_cache_memory_bytes"):
            for gid, total, free in zip(hw.gpu_ids, hw.gpu_total_gib, hw.gpu_free_gib, strict=False):
                if util * total > free + 0.5:
                    add("block", "insufficient_free_vram",
                        f"GPU {gid}: gpu_memory_utilization {util:g} needs {util * total:.1f} GiB free but "
                        f"only {free:.1f} GiB is. Stop whatever else uses the card or lower it to "
                        f"{max(0.05, (free - 1) / total):.2f}.")
                    break
        if util > 0.95:
            add("warn", "util_very_high", "gpu_memory_utilization > 0.95 leaves little headroom for CUDA "
                                          "graphs and activation spikes; OOM at startup is likely.")
        if any(m in arch_blob for m in _HYBRID_MARKERS) and not g("max_num_seqs") and not g("enforce_eager"):
            add("warn", "hybrid_max_num_seqs",
                "Hybrid (Mamba-style) models need one cache block per running sequence. With the default max_num_seqs "
                "(1024 on cards of 70 GiB or more) vLLM can stop at CUDA graph capture with 'max_num_seqs (N) exceeds "
                "available Mamba cache blocks (M)', whatever the memory share. Set max_num_seqs (for example 256) or "
                "raise gpu_memory_utilization.")
        if g("enforce_eager"):
            add("warn", "eager_slow", "enforce_eager disables CUDA graphs and noticeably slows decode.")
        if g("cpu_offload_gb"):
            add("warn", "cpu_offload_slow", "CPU weight offload streams weights over PCIe every step; "
                                            "expect a large slowdown.")
        if g("trust_remote_code"):
            add("warn", "trust_remote_code", "trust_remote_code runs Python from the model repo. Only "
                                             "enable for repos you have reviewed.")
        if g("enable_sleep_mode"):
            add("warn", "sleep_mode_dev_router", "The /sleep endpoints are registered only when VLLM_SERVER_DEV_MODE "
                                                 "is set, which also exposes unauthenticated /collective_rpc; the "
                                                 "console never sets it.")
        # --- tool and reasoning parsers ----------------------------------------------------------------------------
        agentic = model.pipeline_tag in (None, "text-generation", "image-text-to-text", "any-to-any")
        tool = g("tool_call_parser")
        allowed = _tool_dialect(repo)
        if agentic and not tool and allowed:
            add("warn", "tool_parser_missing",
                f"No tool-call parser set; agent/tool use will not work. For this model family set "
                f"enable_auto_tool_choice=true and tool_call_parser={allowed[0]}.")
        if tool and allowed and tool not in allowed:
            add("warn", "tool_parser_dialect_mismatch",
                f"tool_call_parser={tool} does not match this model family's tool-call dialect; use "
                f"{' or '.join(allowed)}. A mismatch leaves raw tool markup in the message content.")
        if tool and not g("enable_auto_tool_choice"):
            add("warn", "tool_parser_inactive", "tool_call_parser has no effect without "
                                                "enable_auto_tool_choice=true.")
        if agentic and not g("reasoning_parser") and not any(m in repo for m in _NON_REASONING_MARKERS):
            hint = _reasoning_hint(repo)
            if hint:
                add("warn", "reasoning_parser_missing",
                    f"This family emits reasoning; without reasoning_parser={hint} the chain-of-thought "
                    "leaks into message.content.")
        # --- image pin versus model family (recalled from the model card, not checked in the image) -----------------
        if "omni" in repo and "nemotron" in repo and ver is not None and ver != (0, 20, 0):
            add("warn", "image_pin_nemotron_omni",
                "NVIDIA's model card names vLLM 0.20.0 for Nemotron 3 Nano Omni; this console pins "
                f"vLLM {'.'.join(map(str, ver))}, so the model may fail to load. If it does, pin "
                "vllm/vllm-openai:v0.20.0 in Settings.")
        return out

    # ------------------------------------------------------------------ memory model
    def memory_model(self, model: ModelInfo, params: dict[str, Any]) -> dict[str, float]:
        g = params.get
        tp = int(g("tensor_parallel_size") or 1)
        out: dict[str, float] = {
            "mem_fraction": float(g("gpu_memory_utilization") or 0.92),
            "kv_bytes_per_elem": _KV_BYTES.get(str(g("kv_cache_dtype") or "auto"), 2.0),
            # Memory the process holds on top of its gpu_memory_utilization share: the CUDA context and NCCL buffers,
            # created before the worker's free-memory snapshot (0.27.1 gpu_worker.init_device), so never inside the
            # share. One measurement (vLLM 0.27.1, one GPU, enforce_eager, a 0.5B model, share 0.15 of a 95.6 GiB
            # RTX PRO 6000) showed 16.86 GiB held against a 14.34 GiB share, so about 2.5 GiB; the extra 0.5 GiB per
            # additional tensor-parallel rank for NCCL buffers is an unmeasured guess.
            "overhead_gib": 0.0,
            "outside_share_gib": _OUTSIDE_POOL_GIB + _NCCL_PER_EXTRA_RANK_GIB * (tp - 1),
            # 0.27.1 worker/utils.py request_memory: refuses to start unless share x card total is free, then fills the
            # share with KV. It checks even when kv_cache_memory_bytes fixes the pool.
            "claims_share": 1.0,
            # vLLM picks 1024 on GPUs of 70 GiB or more and 256 below that (image source); hardware is not known here.
            "max_seqs": float(g("max_num_seqs") or 256),
            "tp": float(tp),
            # measured 0.59-0.89 GiB CUDA-graph pool on a Qwen3-class model (with MTP) on vLLM 0.27.1. Inside the share:
            # 0.27.1 estimates it and takes it off the KV pool (VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS defaults on).
            "cuda_graph_gib": 0.0 if g("enforce_eager") else 0.75,
        }
        max_len = g("max_model_len") or model.max_position_embeddings
        if max_len:
            out["max_len"] = float(max_len)
        kvb = g("kv_cache_memory_bytes")
        if kvb:
            # vLLM ignores gpu_memory_utilization when this is set; extra key lets the estimator use it.
            out["kv_cache_fixed_gib"] = float(kvb) / _GIB
        return out

    def host_memory_gib(self, model: ModelInfo, params: dict[str, Any], hw: Hardware) -> dict[str, float]:
        g = params.get
        gpus = int(g("tensor_parallel_size") or 1) * int(g("pipeline_parallel_size") or 1)
        out: dict[str, float] = {}
        if g("cpu_offload_gb"):
            out["cpu_weight_offload"] = float(g("cpu_offload_gb") or 0) * gpus       # GiB per GPU
        if g("kv_offloading_size"):
            out["kv_offload_buffer"] = float(g("kv_offloading_size") or 0)           # already summed across TP ranks
        # `swap_space` (CPU swap for preempted sequences) is not in the pinned catalog, so it is not modelled.
        return out

    # ------------------------------------------------------------------ observability
    def parse_metrics(self, prometheus_text: str) -> dict[str, float]:
        scrape = Scrape(prometheus_text)
        out = scrape_to_canonical(scrape)
        # Counters carry no rate in one scrape, so keep the previous sample per served-model-name set.
        # Two instances serving the same name would share state — the rate is then skipped (delta < 0)
        # or noisy, never fabricated.
        key = scrape.model_names()
        now = self._clock()
        prompt, gen = out.get("prompt_tokens_total"), out.get("generation_tokens_total")
        prev = self._prev.get(key)
        if prev is not None:
            dt = now - prev[0]
            if dt >= 0.2:
                if prompt is not None and prev[1] is not None and prompt >= prev[1]:
                    out["prompt_tps"] = (prompt - prev[1]) / dt
                if gen is not None and prev[2] is not None and gen >= prev[2]:
                    out["generation_tps"] = (gen - prev[2]) / dt
        if prompt is not None or gen is not None:
            self._prev[key] = (now, prompt, gen)
        return out

    def parse_startup_log(self, line: str) -> dict[str, Any] | None:
        return parse_log_line(line)
