"""SGLang engine adapter (image pinned by tag in `_IMAGE`).

Container needs the LaunchSpec cannot express (the core adds them):
  * shm_size 16g  (32g for multimodal/video builds) and, for TP>1, `ipc: host` (NCCL/torch shm)
  * GPU pinning by UUID, the engine network, HF cache bind-mounted at `hf_cache_container_path`
"""
from __future__ import annotations

import json
import re
from typing import Any

from engine_console.adapters import sglang_parsers
from engine_console.adapters.base import (
    Compat,
    EngineAdapter,
    Hardware,
    LaunchSpec,
    ModelInfo,
    ParamSpec,
)
from engine_console.adapters.sglang_params import build_catalog

_IMAGE = "lmsysorg/sglang:v0.5.14-cu130"
_PORT = 30000
_BIND_ALL = "0.0.0.0"  # noqa: S104 - inside the container; core publishes to 127.0.0.1
_DENSE_QUANTS = {"nvfp4", "modelopt_fp4", "fp8", "awq", "gptq", "mxfp4"}
# Backends built for Hopper (sm_90) / datacenter Blackwell (sm_100); not confirmed on sm_120.
_HOPPER_ONLY = {"fa3", "flashmla"}
_DC_BLACKWELL = {"cutlass_mla", "trtllm_mla", "trtllm_mha", "cutedsl_mla", "fa4"}

# model family regex -> tool parsers that match its dialect (first = recommended)
_TOOL_DIALECTS: list[tuple[re.Pattern[str], tuple[str, ...]]] = [
    (re.compile(r"qwen3[._-]?(coder|[5-9]|next)|qwen3\.\d", re.I), ("qwen3_coder",)),
    (re.compile(r"qwen3(?![._\d])|qwen2\.?5", re.I), ("qwen25", "qwen", "hermes")),
    (re.compile(r"gpt-oss", re.I), ("gpt-oss",)),
    (re.compile(r"deepseek[-_]?v3\.?1", re.I), ("deepseekv31",)),
    (re.compile(r"deepseek[-_]?v3\.?2", re.I), ("deepseekv32",)),
    (re.compile(r"deepseek[-_]?(v3|r1)", re.I), ("deepseekv3",)),
    (re.compile(r"glm[-_]?4\.?7", re.I), ("glm47",)),
    (re.compile(r"glm[-_]?4\.?[56]", re.I), ("glm45",)),
    (re.compile(r"kimi[-_]?k2", re.I), ("kimi_k2",)),
    (re.compile(r"llama[-_]?3", re.I), ("llama3", "pythonic")),
    (re.compile(r"hermes", re.I), ("hermes",)),
    (re.compile(r"mistral|ministral|devstral", re.I), ("mistral",)),
]


class SglangAdapter(EngineAdapter):
    id = "sglang"
    display_name = "SGLang"
    default_image = _IMAGE
    default_port = _PORT

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
                errs.append(f"unknown parameter '{k}'")
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
        errs = self.validate(params)
        if errs:
            raise ValueError("; ".join(errs))
        if not model or model.startswith("-"):
            raise ValueError("invalid model identifier")
        argv = ["python3", "-m", "sglang.launch_server", "--model-path", model,
                "--host", _BIND_ALL, "--port", str(_PORT),
                "--served-model-name", str(params.get("served_model_name") or served_name),
                "--enable-metrics"]  # forced: the console's metrics/dashboards depend on /metrics
        for spec in self._catalog:
            if spec.key in ("enable_metrics", "served_model_name"):
                continue
            v = params.get(spec.key)
            if v is None:
                continue
            if spec.type == "bool":
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
        return LaunchSpec(image=self.default_image, argv=argv,
                          env={"HF_HOME": hf_cache_container_path},
                          container_port=_PORT, health_path="/health",
                          metrics_path="/metrics", models_path="/v1/models")

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
        frac = g("mem_fraction_static")
        if frac is not None and frac > 0.92:
            out.append(Compat(level="warn", code="mem_fraction_high",
                              message=f"mem_fraction_static={frac} leaves <8% for CUDA graphs, prefill and "
                                      "FlashInfer scratch (hungrier on sm_120): OOM mid-request is likely. "
                                      "Keep <= 0.92 unless the card is dedicated and you tested it."))
        quant = (model.quantization or "").lower() or None
        q_arg = g("quantization")
        name = f"{model.repo_id} {model.model_type or ''}"
        is_gptoss = bool(re.search(r"gpt[-_]?oss", name, re.I))
        if quant == "nvfp4" or q_arg in ("modelopt_fp4", "nvfp4_online"):
            out.append(Compat(
                level="ok", code="nvfp4_ok_sglang",
                message="NVFP4 " + ("MoE: SGLang is the recommended engine on sm_120 (vLLM falls back to a "
                                    "slow Marlin kernel)." if model.is_moe else
                                    "dense: supported on sm_120.")))
        if quant == "mxfp4" or q_arg == "mxfp4":
            if model.is_moe and not is_gptoss:
                out.append(Compat(level="block", code="mxfp4_moe_sm120",
                                  message="MXFP4 MoE asserts on sm_120 except gpt-oss-120b. Use an FP8 "
                                          "checkpoint (or quantization=fp8)."))
            elif is_gptoss:
                out.append(Compat(level="warn", code="mxfp4_gptoss_engine",
                                  message="gpt-oss (native MXFP4) is validated here via Ollama/llama.cpp; "
                                          "the SGLang path on sm_120 is unverified."))
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
        ab = g("attention_backend")
        cc = hw.compute_capability
        if cc.startswith("12"):
            if ab in _HOPPER_ONLY:
                out.append(Compat(level="block", code="attention_backend_hopper_only",
                                  message=f"attention_backend={ab} targets Hopper (sm_90), not sm_120; use flashinfer or triton."))
            elif ab in _DC_BLACKWELL:
                out.append(Compat(level="warn", code="attention_backend_dc_blackwell",
                                  message=f"attention_backend={ab} targets datacenter Blackwell (sm_100); "
                                          "unconfirmed on sm_120. Prefer flashinfer or triton."))
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
            out.append(Compat(level="warn", code="context_beyond_native",
                              message=f"context_length {ctx} > native {native}: SGLang refuses unless "
                                      "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 is set and rope scaling "
                                      "(e.g. YaRN via json_model_override_args) is configured."))
        if g("chunked_prefill_size") == -1 and g("enable_mixed_chunk"):
            out.append(Compat(level="warn", code="mixed_chunk_needs_chunked_prefill",
                              message="enable_mixed_chunk needs chunked prefill; chunked_prefill_size=-1 disables it."))
        if g("kv_cache_dtype") == "fp4_e2m1":
            out.append(Compat(level="warn", code="kv_fp4_experimental",
                              message="fp4_e2m1 KV cache is experimental; quality is unvalidated here."))
        if g("enable_torch_compile"):
            out.append(Compat(level="warn", code="torch_compile_slow_start",
                              message="torch.compile is experimental and adds minutes to startup."))
        if g("cpu_offload_gb"):
            out.append(Compat(level="warn", code="cpu_offload_slow",
                              message="CPU weight offload streams over PCIe every step; expect a large slowdown."))
        if g("enable_metrics") is False:
            out.append(Compat(level="warn", code="metrics_forced",
                              message="enable_metrics=false is ignored: the console needs /metrics."))
        if g("api_key"):
            out.append(Compat(level="warn", code="api_key_in_argv",
                              message="api_key is visible in the container argv (docker inspect); "
                                      "prefer router-level auth."))
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
        """SGLang semantics: mem_fraction_static * total covers weights + KV pool ONLY. Activations,
        CUDA graphs and scratch come out of the remaining (1 - fraction) share, so they are NOT charged
        against the budget: overhead/cuda_graph are 0 here and `nonstatic_reserve_frac` says how much
        is left for them. KV pool = fraction*total - weights (per GPU, after TP split)."""
        kv = params.get("kv_cache_dtype")
        kv_bytes = 1.0 if kv in ("fp8_e5m2", "fp8_e4m3") else 0.5 if kv == "fp4_e2m1" else 2.0
        frac = params.get("mem_fraction_static")
        out: dict[str, float] = {
            "mem_fraction": float(frac) if frac is not None else 0.85,  # engine auto-picks; 0.85 = repo default
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

    # ------------------------------------------------------------------ parsers
    def parse_metrics(self, prometheus_text: str) -> dict[str, float]:
        return sglang_parsers.parse_metrics(prometheus_text)

    def parse_startup_log(self, line: str) -> dict[str, Any] | None:
        return sglang_parsers.parse_startup_log(line)
