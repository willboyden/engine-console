"""SGLang engine adapter (image pinned to ``lmsysorg/sglang:v0.5.20-cu130``, see `sglang_params.py` for provenance).

Container needs the LaunchSpec cannot express (the core adds them):
  * shm_size 16g (32g for multimodal or video builds) and, for TP>1, ``ipc: host`` (NCCL / torch shared memory)
  * GPU pinning by UUID, the engine network, the model mounts

Model formats: ``hf`` (a Hugging Face repo id, the HF cache mounted read-only) and ``plain`` (a model directory the core
mounts read-only at ``/models``; SGLang loads a local directory when given its path). ``gguf`` is not offered. Every local
image of this engine is amd64, so ``images_by_arch`` stays empty.

The launch command is ``python3 -m sglang.launch_server``. The 0.5.20 image prints that this spelling "is still supported,
but 'sglang serve' is the recommended entrypoint"; the catalog was verified against the parser of the former, and every
catalog flag is also accepted by ``sglang serve`` (its help is a superset of the launch_server help).
"""
from __future__ import annotations

import json
import re
from typing import Any

from engine_console.adapters import sglang_parsers
from engine_console.adapters.base import (
    MODEL_MOUNT,
    Compat,
    EngineAdapter,
    Hardware,
    LaunchContext,
    LaunchSpec,
    ModelInfo,
    ParamSpec,
)
from engine_console.adapters.sglang_params import DEFAULT_IMAGE, build_catalog

_PORT = 30000
_BIND_ALL = "0.0.0.0"  # noqa: S104 - inside the container; core publishes to 127.0.0.1
# Backends refused or unbuilt on sm_120 (see compatibility()): fa3 asserts SM 8.x/9.x in attention_registry.py.
_HOPPER_ONLY = {"fa3", "flashmla"}
# Backends written for datacenter Blackwell (sm_100) or untested on sm_120; allowed, but unconfirmed.
_DC_BLACKWELL = {"trtllm_mla", "cutedsl_mla", "tokenspeed_mla", "fa4"}
# Parameters removed from the catalog, with the reason a saved profile that still carries them should read.
_RETIRED = {
    "api_key": "'api_key' is not offered: SGLang 0.5.20 accepts its API key only as a command-line argument, and the "
               "console never puts a secret on a command line. Put an authenticating proxy in front of the engine",
}

# model family regex -> tool parsers that match its dialect (first = recommended). `qwen` and `qwen25` are one detector in
# the pinned image, as are `qwen3_coder`, `step3p5` and `nanbeige`.
_TOOL_DIALECTS: list[tuple[re.Pattern[str], tuple[str, ...]]] = [
    (re.compile(r"qwen3[._-]?(coder|next)|qwen3[._][5-9]", re.I), ("qwen3_coder",)),
    (re.compile(r"qwen3(?![._\d])|qwen2\.?5|qwq", re.I), ("qwen25", "qwen", "hermes")),
    (re.compile(r"gpt-oss", re.I), ("gpt-oss",)),
    (re.compile(r"deepseek[-_]?v?3\.?2", re.I), ("deepseekv32",)),
    (re.compile(r"deepseek[-_]?v?3\.?1", re.I), ("deepseekv31",)),
    (re.compile(r"deepseek[-_]?v?4", re.I), ("deepseekv4",)),
    (re.compile(r"deepseek[-_]?(v3|r1)", re.I), ("deepseekv3",)),
    (re.compile(r"glm[-_]?(4\.7|5)", re.I), ("glm47",)),
    (re.compile(r"glm[-_]?4\.?[56]", re.I), ("glm45", "glm")),
    (re.compile(r"kimi[-_]?k2", re.I), ("kimi_k2",)),
    (re.compile(r"llama[-_]?3", re.I), ("llama3", "pythonic")),
    (re.compile(r"hermes", re.I), ("hermes",)),
    (re.compile(r"mistral|ministral|devstral", re.I), ("mistral",)),
]


class SglangAdapter(EngineAdapter):
    id = "sglang"
    display_name = "SGLang"
    default_image = DEFAULT_IMAGE
    default_port = _PORT
    model_formats = frozenset({"hf", "plain"})
    metric_prefixes: tuple[str, ...] = ("sglang:",)

    def __init__(self) -> None:
        self._catalog = build_catalog()
        self._by_key = {p.key: p for p in self._catalog}

    # ------------------------------------------------------------------ catalog
    def param_catalog(self) -> list[ParamSpec]:
        return [p.model_copy(deep=True) for p in self._catalog]

    def presets(self) -> dict[str, dict[str, Any]]:
        # Values are generic; parser presets follow a typical Qwen3 recipe (qwen3_coder + qwen3).
        return {
            "balanced": {"mem_fraction_static": 0.85, "chunked_prefill_size": 8192,
                         "schedule_policy": "lpm", "max_running_requests": 32},
            "max-throughput": {"mem_fraction_static": 0.88, "chunked_prefill_size": 16384,
                               "max_prefill_tokens": 32768, "schedule_policy": "lpm",
                               "enable_mixed_chunk": True, "cuda_graph_max_bs_decode": 256,
                               "stream_interval": 8},
            "low-latency": {"mem_fraction_static": 0.80, "chunked_prefill_size": 2048,
                            "schedule_policy": "fcfs", "max_running_requests": 8,
                            "stream_interval": 1, "schedule_conservativeness": 1.0},
            "long-context": {"mem_fraction_static": 0.85, "chunked_prefill_size": 4096,
                             "kv_cache_dtype": "fp8_e4m3", "max_running_requests": 4,
                             "schedule_policy": "lpm"},
            "tool-agent": {"mem_fraction_static": 0.85, "chunked_prefill_size": 8192,
                           "schedule_policy": "lpm", "tool_call_parser": "qwen3_coder",
                           "reasoning_parser": "qwen3", "enable_cache_report": True,
                           "max_running_requests": 16},
            "reasoning": {"mem_fraction_static": 0.85, "chunked_prefill_size": 8192,
                          "reasoning_parser": "qwen3", "schedule_policy": "lpm",
                          "watchdog_timeout": 1800},
        }

    # ------------------------------------------------------------------ validate
    def validate(self, params: dict[str, Any]) -> list[str]:
        errs: list[str] = []
        for k, v in params.items():
            spec = self._by_key.get(k)
            if spec is None:
                errs.append(_RETIRED.get(k, f"unknown parameter '{k}'"))
                continue
            if v is None:
                continue
            errs.extend(self._check_value(spec, v))
        if errs:
            return errs
        g = params.get
        algo = g("speculative_algorithm")
        if algo in ("EAGLE", "EAGLE3", "STANDALONE") and not g("speculative_draft_model_path"):
            errs.append(f"speculative_algorithm={algo} requires speculative_draft_model_path")
        if algo is None and any(g(k) for k in ("speculative_draft_model_path", "speculative_num_steps",
                                               "speculative_eagle_topk", "speculative_num_draft_tokens")):
            errs.append("speculative_* options are set but speculative_algorithm is not")
        if g("enable_dp_attention") and g("dp_size", 1) != g("tp_size", 1):
            errs.append("enable_dp_attention requires dp_size == tp_size")
        tp, ep = g("tp_size", 1), g("ep_size", 1)
        if ep > 1 and tp % ep != 0:
            errs.append(f"ep_size ({ep}) must divide tp_size ({tp})")
        if g("enable_hierarchical_cache") and g("disable_radix_cache"):
            errs.append("enable_hierarchical_cache requires the radix cache (disable_radix_cache must be off)")
        if g("enable_trace") and not g("otlp_traces_endpoint"):
            errs.append("enable_trace requires otlp_traces_endpoint")
        if g("hicache_size") and not g("enable_hierarchical_cache"):
            errs.append("hicache_* options need enable_hierarchical_cache")
        return errs

    @staticmethod
    def _check_value(spec: ParamSpec, v: Any) -> list[str]:
        k, t = spec.key, spec.type
        if t == "bool":
            return [] if isinstance(v, bool) else [f"{k}: expected boolean"]
        if t in ("int", "float"):
            if isinstance(v, bool) or not isinstance(v, (int, float)) or (t == "int" and not float(v).is_integer()):
                return [f"{k}: expected {t}"]
            if spec.min is not None and v < spec.min:
                return [f"{k}: {v} is below minimum {spec.min:g}"]
            if spec.max is not None and v > spec.max:
                return [f"{k}: {v} is above maximum {spec.max:g}"]
            return []
        if t == "enum":
            if not isinstance(v, str) or (spec.choices and v not in spec.choices):
                return [f"{k}: '{v}' not one of {spec.choices}"]
            return []
        if t == "string":
            if not isinstance(v, str) or not v:
                return [f"{k}: expected non-empty string"]
            if v.startswith("-") or "\x00" in v or "\n" in v:
                return [f"{k}: value must not start with '-' or contain control characters"]
            return []
        if t == "string_list":
            ok = isinstance(v, list) and v and all(isinstance(x, str) and x and not x.startswith("-") for x in v)
            return [] if ok else [f"{k}: expected non-empty list of strings"]
        if t == "json":
            if isinstance(v, (dict, list)):
                return []
            if isinstance(v, str):
                try:
                    json.loads(v)
                    return []
                except ValueError:
                    return [f"{k}: invalid JSON"]
            return [f"{k}: expected JSON object or string"]
        return []

    # ------------------------------------------------------------------ launch
    def build_launch(self, model: str, params: dict[str, Any], hw: Hardware, *,
                     served_name: str, hf_cache_container_path: str) -> LaunchSpec:
        """The v1 entry point: a Hugging Face repo id. Builds a context and defers to `build_launch_ctx`."""
        return self.build_launch_ctx(LaunchContext(
            model=model, model_format="hf", served_name=served_name,
            hf_cache_container_path=hf_cache_container_path, params=params, hw=hw))

    def build_launch_ctx(self, ctx: LaunchContext) -> LaunchSpec:
        if ctx.model_format not in self.model_formats:
            raise ValueError(f"SGLang cannot load {ctx.model_format} models "
                             f"(it loads: {', '.join(sorted(self.model_formats))})")
        errs = self.validate(ctx.params)
        if errs:
            raise ValueError("; ".join(errs))
        plain = ctx.model_format == "plain"
        # A plain model is a directory the core mounts read-only at MODEL_MOUNT; SGLang takes its path as the model.
        model_arg = (ctx.model_path or MODEL_MOUNT) if plain else ctx.model
        if not model_arg or model_arg.startswith("-"):
            raise ValueError("invalid model identifier")
        params = ctx.params
        argv = ["python3", "-m", "sglang.launch_server", "--model-path", model_arg,
                "--host", _BIND_ALL, "--port", str(_PORT),
                "--served-model-name", str(params.get("served_model_name") or ctx.served_name),
                "--enable-metrics"]  # forced: the console's metrics/dashboards depend on /metrics
        extra_env: dict[str, str] = {}
        for spec in self._catalog:
            if spec.key in ("enable_metrics", "served_model_name"):
                continue
            if spec.key == "revision" and plain:
                continue                          # a directory has no hub revision
            v = params.get(spec.key)
            if v is None:
                continue
            if spec.flag.startswith("env:"):
                extra_env[spec.flag[4:]] = ("1" if v else "0") if spec.type == "bool" else str(v)
            elif spec.type == "bool":
                if v:
                    argv.append(spec.flag)
            elif spec.type == "string_list":
                argv += [spec.flag, *[str(x) for x in v]]
            elif spec.type == "json":
                argv += [spec.flag, v if isinstance(v, str) else json.dumps(v, separators=(",", ":"))]
            elif spec.type == "float":
                argv += [spec.flag, repr(float(v))]
            else:
                argv += [spec.flag, str(v)]
        # Telemetry off (deny-by-default egress). A plain model is local: never wait on the hub.
        env = {"HF_HUB_DISABLE_TELEMETRY": "1"}
        if plain:
            env["HF_HUB_OFFLINE"] = "1"
        else:
            env["HF_HOME"] = ctx.hf_cache_container_path
        env.update(extra_env)
        return LaunchSpec(image=self.default_image, argv=argv, env=env,
                          container_port=_PORT, health_path="/health",
                          metrics_path="/metrics", models_path="/v1/models", needs_hf_cache=not plain)

    # ------------------------------------------------------------------ compat
    def compatibility(self, model: ModelInfo, params: dict[str, Any], hw: Hardware) -> list[Compat]:
        out: list[Compat] = []
        g = params.get
        tp, dp, pp = g("tp_size", 1), g("dp_size", 1), g("pp_size", 1)
        world = tp * pp * (1 if g("enable_dp_attention") else dp)
        if world > len(hw.gpu_ids):
            out.append(Compat(level="block", code="gpus_insufficient",
                              message=f"tp*pp*dp needs {world} GPUs but only {len(hw.gpu_ids)} selected; "
                                      "lower tp_size/dp_size or select more GPUs."))
        # SGLang stops at start-up when the selected GPUs' free memory differs by more than 10 % (image source:
        # distributed/bootstrap.py, SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK is on by default).
        free = list(hw.gpu_free_gib)
        if tp > 1 and len(free) > 1 and max(free) > 0 and min(free) < 0.9 * max(free):
            out.append(Compat(level="block", code="tp_memory_unbalanced",
                              message=f"The selected GPUs have {min(free):.1f} and {max(free):.1f} GiB free; SGLang stops "
                                      "with 'The memory capacity is unbalanced' when they differ by more than 10 %. "
                                      "Free the busier card, stop what shares it, or pick cards with similar free memory."))
        frac = g("mem_fraction_static")
        if frac is not None and frac > 0.92:
            out.append(Compat(level="warn", code="mem_fraction_high",
                              message=f"mem_fraction_static={frac} leaves too little headroom for activations, CUDA graphs "
                                      "and kernel scratch: SGLang's own automatic choice normally keeps at least 10 GiB "
                                      "free on cards over 60 GB. Out-of-memory during a long prefill is likely. Keep it at "
                                      "0.92 or below unless the card is dedicated and you tested it."))
        quant = (model.quantization or "").lower() or None
        q_arg = g("quantization")
        name = f"{model.repo_id} {model.model_type or ''}"
        is_gptoss = bool(re.search(r"gpt[-_]?oss", name, re.I))
        cc = hw.compute_capability
        if quant == "nvfp4" or q_arg in ("modelopt_fp4", "nvfp4_online"):
            out.append(Compat(
                level="ok", code="nvfp4_ok_sglang",
                message="NVFP4 " + ("MoE: SGLang is the recommended engine on sm_120 (vLLM can fall back to a slow "
                                    "Marlin kernel)." if model.is_moe else
                                    "dense: supported on sm_120.")))
        if quant == "mxfp4" or q_arg == "mxfp4":
            if model.is_moe and not is_gptoss and cc.startswith("12"):
                out.append(Compat(level="block", code="mxfp4_moe_sm120",
                                  message="MXFP4 MoE asserts on sm_120 except gpt-oss. Use an FP8 "
                                          "checkpoint (or quantization=fp8)."))
            elif is_gptoss:
                out.append(Compat(level="warn", code="mxfp4_gptoss_engine",
                                  message="gpt-oss ships native MXFP4 weights; the SGLang path for them on sm_120 is "
                                          "untested here. If it fails, run the model with llama.cpp or Ollama."))
        if q_arg == "fp8" and quant not in (None, "fp8"):
            out.append(Compat(level="warn", code="quant_override_prequantized",
                              message=f"quantization=fp8 on a pre-quantized ({quant}) checkpoint misloads; "
                                      "unset quantization so SGLang auto-detects its config."))
        if q_arg == "modelopt_fp4" and quant not in ("nvfp4", None):
            out.append(Compat(level="warn", code="modelopt_fp4_mismatch",
                              message="modelopt_fp4 needs an NVFP4 ModelOpt checkpoint."))
        if q_arg and q_arg.startswith("modelopt") and quant is None:
            out.append(Compat(level="warn", code="modelopt_unquantized",
                              message="ModelOpt quantization expects a pre-quantized checkpoint; "
                                      "for a bf16 model use quantization=fp8."))
        # --- attention backends versus GPU generation (image source: layers/attention/attention_registry.py,
        #     arg_groups/attention_hook.py) -------------------------------------------------------------------------
        backends = {b for b in (g("attention_backend"), g("prefill_attention_backend"), g("decode_attention_backend")) if b}
        if cc.startswith("12"):
            hopper = sorted(backends & _HOPPER_ONLY)
            if hopper:
                extra = (" SGLang stops with 'FlashAttention v3 Backend requires SM>=80 and SM<=90'." if "fa3" in hopper
                         else "")
                out.append(Compat(level="block", code="attention_backend_hopper_only",
                                  message=f"attention_backend {', '.join(hopper)} is built for Hopper (sm_90), not "
                                          f"sm_120.{extra} Use flashinfer or triton."))
            dc = sorted(backends & _DC_BLACKWELL)
            if dc:
                out.append(Compat(level="warn", code="attention_backend_dc_blackwell",
                                  message=f"attention_backend {', '.join(dc)} targets datacenter Blackwell (sm_100); "
                                          "unconfirmed on sm_120. Prefer flashinfer or triton."))
            if (g("attention_backend") == "trtllm_mha" or g("prefill_attention_backend") == "trtllm_mha") \
                    and g("kv_cache_dtype") == "fp8_e4m3":
                out.append(Compat(level="block", code="trtllm_mha_fp8_kv_sm120",
                                  message="On sm_120 the trtllm_mha prefill kernel does not support an fp8_e4m3 KV cache "
                                          "(SGLang stops at start-up). Use kv_cache_dtype auto, or another prefill "
                                          "attention backend such as flashinfer."))
        # --- hybrid-model kernels (image source: arg_groups/attention_hook.py, handle_linear_attn_backend) ---------------
        cc_major = int(cc.split(".")[0]) if re.fullmatch(r"\d+\.\d+", cc.strip()) else None
        if cc_major is not None and cc_major >= 10 and (g("linear_attn_decode_backend") or g("linear_attn_backend")) == "flashinfer":
            ssm = g("mamba_ssm_dtype")
            if ssm is not None and ssm != "bfloat16":
                out.append(Compat(level="block", code="linear_attn_flashinfer_needs_bf16_state",
                                  message=f"The FlashInfer linear-attention decode kernel needs mamba_ssm_dtype bfloat16 on "
                                          f"this GPU generation (SGLang stops with '--linear-attn-decode-backend flashinfer "
                                          f"on SM100+ requires --mamba-ssm-dtype bfloat16'; sm_120 counts); you set {ssm}. "
                                          "Set it to bfloat16 or use the triton decode backend."))
            elif ssm is None:
                out.append(Compat(level="warn", code="linear_attn_flashinfer_needs_bf16_state",
                                  message="The FlashInfer linear-attention decode kernel needs mamba_ssm_dtype bfloat16 on "
                                          "this GPU generation; SGLang stops otherwise. Set mamba_ssm_dtype to bfloat16 "
                                          "unless the model's own configuration already says so."))
        # --- tool-call dialect -------------------------------------------------------------------------------------
        tool = g("tool_call_parser")
        if tool and tool != "auto":
            for pat, allowed in _TOOL_DIALECTS:
                if pat.search(name):
                    if tool not in allowed:
                        out.append(Compat(
                            level="warn", code="tool_parser_dialect_mismatch",
                            message=f"tool_call_parser={tool} does not match this model's dialect; use "
                                    f"{' / '.join(allowed)}. Qwen3.x/Coder emit XML "
                                    "(<function=..>) which needs qwen3_coder; qwen expects JSON in <tool_call>."))
                    break
        elif not tool and re.search(r"qwen3|llama|glm|kimi|mistral|hermes|deepseek", name, re.I):
            out.append(Compat(level="warn", code="no_tool_parser",
                              message="No tool_call_parser: agentic clients get raw text in content and an "
                                      "empty tool_calls, then stall. Set one if you serve tool-using clients."))
        if g("enable_dp_attention"):
            if g("dp_size", 1) != tp:
                out.append(Compat(level="block", code="dp_attention_dp_ne_tp",
                                  message="enable_dp_attention requires dp_size == tp_size."))
            if not model.is_moe and not model.kv_lora_rank:
                out.append(Compat(level="warn", code="dp_attention_unsupported_arch",
                                  message="DP attention is supported for DeepSeek-style (MLA) and Qwen MoE "
                                          "models; this model looks dense."))
        if g("ep_size", 1) > 1 and not model.is_moe:
            out.append(Compat(level="warn", code="ep_dense_model",
                              message="ep_size > 1 has no effect on a dense model."))
        ctx, native = g("context_length"), model.max_position_embeddings
        if ctx and native and ctx > native:
            if g("allow_longer_context_len"):
                out.append(Compat(level="warn", code="context_beyond_native",
                                  message=f"context_length {ctx} > native {native}: SGLang accepts it with a warning that "
                                          "'This may lead to incorrect model outputs or CUDA errors'. Configure rope scaling "
                                          "(for example YaRN in json_model_override_args) and measure quality before "
                                          "relying on it."))
            else:
                out.append(Compat(level="warn", code="context_beyond_native",
                                  message=f"context_length {ctx} > native {native}: SGLang stops at start-up with 'User-"
                                          "specified context_length ... is greater than the derived context_length' unless "
                                          "you turn on allow_longer_context_len and configure rope scaling (for example "
                                          "YaRN in json_model_override_args), or lower context_length."))
        if g("chunked_prefill_size") == -1 and g("enable_mixed_chunk"):
            out.append(Compat(level="warn", code="mixed_chunk_needs_chunked_prefill",
                              message="enable_mixed_chunk needs chunked prefill; chunked_prefill_size=-1 disables it."))
        if g("kv_cache_dtype") in ("fp4_e2m1", "nvfp4", "fp4_mx_block16"):
            out.append(Compat(level="warn", code="kv_fp4_experimental",
                              message="FP4 KV cache types are new in SGLang and need CUDA 12.8 or newer; "
                                      "output quality is unvalidated here."))
        if g("enable_torch_compile"):
            out.append(Compat(level="warn", code="torch_compile_slow_start",
                              message="torch.compile is experimental and adds minutes to startup."))
        if g("cpu_offload_gb"):
            out.append(Compat(level="warn", code="cpu_offload_slow",
                              message="CPU weight offload streams over PCIe every step; expect a large slowdown."))
        if g("enable_metrics") is False:
            out.append(Compat(level="warn", code="metrics_forced",
                              message="enable_metrics=false is ignored: the console needs /metrics."))
        if g("log_requests"):
            out.append(Compat(level="warn", code="log_requests_leaks",
                              message="log_requests writes prompts and outputs into container logs."))
        if g("trust_remote_code"):
            out.append(Compat(level="warn", code="trust_remote_code",
                              message="trust_remote_code executes model-repo Python code in the container."))
        if model.gated:
            out.append(Compat(level="warn", code="gated_model",
                              message="Gated model: accept the license on Hugging Face and provide HF_TOKEN."))
        return out

    # ------------------------------------------------------------------ memory
    def memory_model(self, model: ModelInfo, params: dict[str, Any]) -> dict[str, float]:
        """SGLang semantics (0.5.20, kv_cache_configurator.py): the static pool is what is free when the engine starts times
        mem_fraction_static, and it holds weights plus the KV pool only. Activations, CUDA graphs and scratch come out of
        the remaining share, so they are NOT charged against the budget: overhead/cuda_graph are 0 here and
        `nonstatic_reserve_frac` says how much is left for them. KV pool = fraction * free - weights (per GPU, after TP
        split). Unset, SGLang derives the fraction from the card (normally at least 10 GiB kept free above 60 GB); 0.85 is
        a conservative planning value, not the engine's number.

        Not charged: the CUDA context and library workspaces that the process holds on top of its share. A vLLM launch
        measured about 2.5 GiB of that kind per GPU; SGLang's figure has not been measured, so `overhead_gib` stays 0
        here instead of guessing. The share itself is applied to the memory that is free when the engine starts
        (kv_cache_configurator.py), so another engine's memory is already excluded."""
        kv = params.get("kv_cache_dtype")
        kv_bytes = 1.0 if kv in ("fp8_e5m2", "fp8_e4m3") else 0.5 if kv in ("fp4_e2m1", "nvfp4", "fp4_mx_block16") else 2.0
        frac = params.get("mem_fraction_static")
        out: dict[str, float] = {
            "mem_fraction": float(frac) if frac is not None else 0.85,
            "mem_fraction_is_default": 0.0 if frac is not None else 1.0,
            "kv_bytes_per_elem": kv_bytes,
            "overhead_gib": 0.0,
            "cuda_graph_gib": 0.0,
            "tp": float(params.get("tp_size", 1)),
        }
        out["nonstatic_reserve_frac"] = 1.0 - out["mem_fraction"]
        ctx = params.get("context_length") or model.max_position_embeddings
        if ctx:
            out["max_len"] = float(ctx)
        if params.get("max_running_requests"):
            out["max_seqs"] = float(params["max_running_requests"])
        return out

    def host_memory_gib(self, model: ModelInfo, params: dict[str, Any], hw: Hardware) -> dict[str, float]:
        g = params.get
        tp = int(g("tp_size") or 1)
        out: dict[str, float] = {}
        if g("cpu_offload_gb"):
            out["cpu_weight_offload"] = float(g("cpu_offload_gb") or 0) * tp        # assumed per TP rank (like vLLM)
        if g("enable_hierarchical_cache"):
            key = "hicache_host_pool (assumed per TP rank)"
            if g("hicache_size"):
                out[key] = float(g("hicache_size") or 0) * tp                       # GB; overrides hicache_ratio
            else:
                # ratio is relative to the DEVICE KV pool: static budget minus this rank's weights
                frac = float(g("mem_fraction_static") or 0.85)
                if hw.gpu_total_gib and model.weight_bytes:
                    pool = max(frac * min(hw.gpu_total_gib) - model.weight_bytes / 1024**3 / tp, 0.0)
                    out[key] = float(g("hicache_ratio") or 2.0) * pool * tp
                else:
                    out[key] = float("nan")                                    # device pool unknown: never guess
        return out

    # ------------------------------------------------------------------ parsers
    def parse_metrics(self, prometheus_text: str) -> dict[str, float]:
        return sglang_parsers.parse_metrics(prometheus_text)

    def parse_startup_log(self, line: str) -> dict[str, Any] | None:
        return sglang_parsers.parse_startup_log(line)
