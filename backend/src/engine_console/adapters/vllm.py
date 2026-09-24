"""vLLM adapter — implements the `EngineAdapter` port for the vLLM OpenAI server.

Launch conventions (the LaunchSpec has no field for them, so lifecycle/core must add them):
  * The image entrypoint is already ``vllm serve``; `argv` starts with ``--model``.
  * docker flags NOT expressible in argv/env and REQUIRED by vLLM:
      ``--ipc=host`` (or a large ``--shm-size``; tensor-parallel workers use shared memory — the
      usual compose recipe sets ``ipc: host``), GPU pinning by UUID (``--gpus device=<uuid>``),
      the HF cache bind-mount at `hf_cache_container_path`, and the engine network.
  * Persistent caches are worth mounting (first start JIT-compiles FP4 kernels for ~15 min):
      ``/root/.cache/vllm`` (torch.compile + flashinfer autotune) and ``/root/.cache/flashinfer``.
  * HF_TOKEN is deliberately NOT placed in `LaunchSpec.env`: core should inject it with a value-less
    ``-e HF_TOKEN`` so it never enters the spec, logs or the audit trail. `api_key`, in contrast, is
    carried in env (VLLM_API_KEY) rather than argv, and core must redact it when displaying specs.
  * Param keys whose catalog `flag` is ``env:NAME`` become container env vars; ``@image`` overrides
    the image. Everything else becomes a CLI flag (see `_render`).
  * VLLM_ATTENTION_BACKEND does not exist in v0.23.0; the equivalent is ``--attention-backend``.
  * VLLM_SERVER_DEV_MODE is never set (it exposes an unauthenticated /collective_rpc).
"""
from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Callable
from typing import Any, Literal

from .base import Compat, EngineAdapter, Hardware, LaunchSpec, ModelInfo, ParamSpec
from .vllm_params import DEFAULT_IMAGE, NARGS_KEYS, NEGATABLE_KEYS, build_catalog
from .vllm_parse import Scrape, parse_log_line, scrape_to_canonical

_GIB = 1024 ** 3
_IMAGE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-/]*(:\d+)?(/[A-Za-z0-9._\-/]+)*:[A-Za-z0-9._\-]+$")
_KV_BYTES = {"fp8": 1.0, "fp8_e4m3": 1.0, "fp8_e5m2": 1.0, "fp8_inc": 1.0, "fp8_ds_mla": 1.0,
             "fp8_per_token_head": 1.0, "int8_per_token_head": 1.0, "nvfp4": 0.5625,
             "turboquant_3bit_nc": 0.4, "turboquant_4bit_nc": 0.5, "turboquant_k3v4_nc": 0.45,
             "turboquant_k8v4": 0.75}
# repo-name substring -> parser vLLM needs. Order matters (first match wins).
_TOOL_PARSER_HINTS = [
    ("gpt-oss", "openai"), ("qwen3-coder", "qwen3_coder"), ("qwen3", "qwen3_xml"),
    ("nemotron", "qwen3_coder"), ("hermes", "hermes"), ("llama-3", "llama3_json"),
    ("llama-4", "llama4_pythonic"), ("mistral", "mistral"), ("deepseek-v3.2", "deepseek_v32"),
    ("deepseek-v3.1", "deepseek_v31"), ("deepseek-v3", "deepseek_v3"), ("glm-4.7", "glm47"),
    ("glm-4.5", "glm45"), ("glm-4.6", "glm45"), ("kimi-k2", "kimi_k2"), ("minimax-m2", "minimax_m2"),
    ("laguna", "poolside_v1"), ("granite-4", "granite4"), ("gemma-4", "gemma4"),
]
_REASONING_HINTS = [
    ("gpt-oss", "openai_gptoss"), ("deepseek-r1", "deepseek_r1"), ("qwen3", "qwen3"),
    ("nemotron", "nemotron_v3"), ("glm-4.5", "glm45"), ("glm-4.6", "glm45"), ("glm-4.7", "glm45"),
    ("kimi-k2-thinking", "kimi_k2"), ("laguna", "poolside_v1"), ("minimax-m2", "minimax_m2"),
    ("olmo-3", "olmo3"), ("gemma-4", "gemma4"),
]
_NON_REASONING_MARKERS = ("coder", "instruct-2507")


def _image_version(image: str) -> tuple[int, int, int] | None:
    m = re.search(r":v?(\d+)\.(\d+)\.(\d+)", image)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def _hint(table: list[tuple[str, str]], repo: str) -> str | None:
    low = repo.lower()
    for needle, parser in table:
        if needle in low:
            return parser
    return None


class VllmAdapter(EngineAdapter):
    id = "vllm"
    display_name = "vLLM"
    default_image = DEFAULT_IMAGE
    default_port = 8000

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
            # vLLM's own defaults plus prefix caching; safe on any model.
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
            # fp8 KV doubles token capacity; few sequences because each one is big. Attention backend
            # left on auto: pick FLASHINFER explicitly if the model needs FP4/FP8-KV kernels.
            "long-context": {"gpu_memory_utilization": 0.92, "max_model_len": 262144,
                             "kv_cache_dtype": "fp8", "max_num_seqs": 8,
                             "max_num_batched_tokens": 8192, "enable_prefix_caching": True,
                             "enable_chunked_prefill": True},
            # hermes suits Hermes-family models; change the parser for other families
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
                errs.append(f"unknown parameter '{key}'")
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
        if g("speculative_config") and any(g(k) for k in ("spec_method", "spec_model", "spec_tokens")):
            errs.append("speculative_config cannot be combined with spec_method/spec_model/spec_tokens")
        cpu_loras, loras = g("max_cpu_loras"), g("max_loras")
        if cpu_loras and loras and cpu_loras < loras:
            errs.append("max_cpu_loras must be >= max_loras")
        if g("allowed_local_media_path") == "/":
            errs.append("allowed_local_media_path must not be '/': any prompt could read any container file")
        img = g("image")
        if img is not None and (not _IMAGE_RE.match(img) or img.endswith(":latest")):
            errs.append("image must be a pinned 'repo:tag' reference (never :latest)")
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
            return [] if isinstance(val, str) and val.strip() else [f"{k}: expected a non-empty string"]
        if t == "string_list":
            if not isinstance(val, list) or not val or not all(isinstance(x, (str, int)) and str(x).strip()
                                                               and not isinstance(x, bool) for x in val):
                return [f"{k}: expected a non-empty list of strings"]
            if k == "cudagraph_capture_sizes" and not all(str(x).isdigit() and int(x) > 0 for x in val):
                return [f"{k}: entries must be positive integers"]
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
        errs = self.validate(params)
        if errs:
            raise ValueError("; ".join(errs))
        argv = ["--model", model, "--host", "0.0.0.0",  # noqa: S104 (container-internal; core publishes to loopback)
                "--port", str(self.default_port)]
        # TORCH_CUDA_ARCH_LIST: image default already covers 12.0+PTX; pinned to exactly 12.0 to
        # avoid JIT for other archs. Telemetry off: deny-by-default egress.
        env = {"TORCH_CUDA_ARCH_LIST": "12.0", "HF_HOME": hf_cache_container_path,
               "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1", "HF_HUB_DISABLE_TELEMETRY": "1"}
        image = self.default_image
        names = [served_name] + [n for n in params.get("served_model_name") or [] if n != served_name]
        argv += ["--served-model-name", *names]
        for spec in self._catalog:
            val = params.get(spec.key)
            if val is None or spec.key == "served_model_name":
                continue
            if spec.flag == "@image":
                image = str(val)
            elif spec.flag.startswith("env:"):
                env[spec.flag[4:]] = ("1" if val else "0") if spec.type == "bool" else str(val)
            else:
                argv += self._render(spec, val)
        return LaunchSpec(image=image, argv=argv, env=env, container_port=self.default_port,
                          health_path="/health", metrics_path="/metrics")

    # ------------------------------------------------------------------ compatibility
    def compatibility(self, model: ModelInfo, params: dict[str, Any], hw: Hardware) -> list[Compat]:
        out: list[Compat] = []

        def add(lvl: Literal["ok", "warn", "block"], code: str, msg: str) -> None:
            out.append(Compat(level=lvl, code=code, message=msg))

        g = params.get
        repo = model.repo_id.lower()
        quant = (model.quantization or "").lower()
        is_nvfp4 = "nvfp4" in quant or "fp4" in quant and "mx" not in quant or "nvfp4" in repo
        is_mxfp4 = "mxfp4" in quant or "mxfp4" in repo
        is_gpt_oss = "gpt-oss" in repo or any("GptOss" in a for a in model.architectures)
        moe_backend = g("moe_backend")
        tp, pp, dp = g("tensor_parallel_size") or 1, g("pipeline_parallel_size") or 1, g("data_parallel_size") or 1
        image = g("image") or self.default_image
        ver = _image_version(image)

        if is_nvfp4 and model.is_moe and moe_backend in (None, "auto"):
            add("warn", "nvfp4_moe_sm120",
                "NVFP4 MoE on sm_120 falls back to a slow Marlin kernel in vLLM. Use the FP8 build, run it "
                "on SGLang, or A/B --moe-backend (cutlass / flashinfer_b12x / marlin).")
        if is_mxfp4 and model.is_moe and not is_gpt_oss:
            add("block", "mxfp4_moe_sm120",
                "MXFP4 MoE asserts on sm_120 except gpt-oss. Use an FP8 build of this model.")
        if is_gpt_oss:
            add("warn", "gpt_oss_prefer_ollama",
                "gpt-oss-120b is native MXFP4; this repo serves it via Ollama/llama.cpp. On vLLM check "
                "the model recipe and keep quantization unset.")
        if model.gated:
            add("warn", "gated_needs_hf_token",
                "Gated model: set HF_TOKEN (Settings) and accept the license on the model page before "
                "downloading or starting.")
        world = tp * pp * dp
        if world > len(hw.gpu_ids):
            add("block", "tp_exceeds_gpus",
                f"tensor*pipeline*data parallel = {world} but only {len(hw.gpu_ids)} GPU(s) selected. "
                "Lower the parallel sizes or select more GPUs.")
        elif tp > 1:
            add("warn", "tp_over_pcie",
                "The two cards have no NVLink; tensor parallelism all-reduces over PCIe. Prefer one model "
                "per card unless it does not fit.")
        heads, kv = model.num_attention_heads, model.num_kv_heads
        if tp > 1 and heads and heads % tp:
            add("block", "tp_heads_indivisible",
                f"{heads} attention heads are not divisible by tensor_parallel_size={tp}. Pick a divisor.")
        elif tp > 1 and kv and kv % tp and tp % kv:
            add("block", "tp_kv_heads_indivisible",
                f"{kv} KV heads and tensor_parallel_size={tp}: one must divide the other.")
        kvd, backend = g("kv_cache_dtype"), g("attention_backend")
        if kvd and str(kvd).startswith("fp8") and backend == "FLASH_ATTN":
            add("warn", "kv_fp8_flash_attn",
                "fp8 KV cache with FLASH_ATTN (FA2 on sm_120) is likely unsupported. Use "
                "attention_backend=FLASHINFER, which supports fp8 KV.")
        if kvd and str(kvd).startswith("fp8") and quant in ("", "none"):
            add("warn", "kv_fp8_accuracy",
                "fp8 KV cache trades accuracy for capacity (some models log a warning on it). "
                "Compare outputs on your task before adopting it.")
        if backend in ("CUTLASS_MLA", "FLASHMLA", "FLASHMLA_SPARSE"):
            add("block", "mla_backend_arch",
                f"{backend} targets data-centre Hopper/sm_100 GPUs, not sm_120. Use FLASHINFER_MLA, "
                "TRITON_MLA or auto.")
        if g("moe_backend") == "deep_gemm":
            add("warn", "deep_gemm_sm120",
                "DeepGEMM MoE kernels are FP8-block-quant only and target Hopper/sm_100; on sm_120 prefer "
                "auto, cutlass or flashinfer_b12x.")
        if g("enable_expert_parallel") and not model.is_moe:
            add("warn", "ep_dense_model", "enable_expert_parallel has no effect on a dense model.")
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
        util = g("gpu_memory_utilization") or 0.92
        if not g("kv_cache_memory_bytes"):
            for gid, total, free in zip(hw.gpu_ids, hw.gpu_total_gib, hw.gpu_free_gib, strict=False):
                if util * total > free + 0.5:
                    add("block", "insufficient_free_vram",
                        f"GPU {gid}: gpu_memory_utilization {util:g} needs {util * total:.1f} GiB free but "
                        f"only {free:.1f} GiB is. Stop the co-resident engine or lower it to "
                        f"{max(0.05, (free - 1) / total):.2f}.")
                    break
        if util > 0.95:
            add("warn", "util_very_high", "gpu_memory_utilization > 0.95 leaves little headroom for CUDA "
                                          "graphs and activation spikes; OOM at startup is likely.")
        if g("enforce_eager"):
            add("warn", "eager_slow", "enforce_eager disables CUDA graphs and noticeably slows decode.")
        if g("cpu_offload_gb"):
            add("warn", "cpu_offload_slow", "CPU weight offload streams weights over PCIe every step; "
                                            "expect a large slowdown.")
        if g("trust_remote_code"):
            add("warn", "trust_remote_code", "trust_remote_code runs Python from the model repo. Only "
                                             "enable for repos you have reviewed.")
        if g("enable_sleep_mode"):
            add("warn", "sleep_mode_dev_router", "The /sleep endpoints need VLLM_SERVER_DEV_MODE, which "
                                                 "also exposes unauthenticated /collective_rpc; the "
                                                 "console does not enable it.")
        # tool / reasoning parsers
        agentic = model.pipeline_tag in (None, "text-generation", "image-text-to-text", "any-to-any")
        if agentic and not g("tool_call_parser"):
            hint = _hint(_TOOL_PARSER_HINTS, repo)
            if hint:
                add("warn", "tool_parser_missing",
                    f"No tool-call parser set; agent/tool use will not work. For this model family set "
                    f"enable_auto_tool_choice=true and tool_call_parser={hint}.")
        if g("tool_call_parser") and not g("enable_auto_tool_choice"):
            add("warn", "tool_parser_inactive", "tool_call_parser has no effect without "
                                                "enable_auto_tool_choice=true.")
        if agentic and not g("reasoning_parser") and not any(m in repo for m in _NON_REASONING_MARKERS):
            hint = _hint(_REASONING_HINTS, repo)
            if hint:
                add("warn", "reasoning_parser_missing",
                    f"This family emits reasoning; without reasoning_parser={hint} the chain-of-thought "
                    "leaks into message.content.")
        # speculative decoding
        sc = g("speculative_config")
        if isinstance(sc, str):
            try:
                sc = json.loads(sc)
            except json.JSONDecodeError:
                sc = None
        method = (sc or {}).get("method") if isinstance(sc, dict) else g("spec_method")
        if method == "dflash" and (g("max_num_seqs") or 256) > 8:
            add("warn", "dflash_max_num_seqs",
                "DFlash speculation crashed the draft path at the default max_num_seqs (256) in lab tests; "
                "set max_num_seqs to 4-8.")
        # image version vs model family
        if "laguna" in repo and ver is not None and ver < (0, 25, 0):
            add("block", "image_too_old_laguna",
                f"Laguna needs vLLM >= 0.25.0 (older builds emit nondeterministic garbage); image is {image}. "
                "Set image to vllm/vllm-openai:v0.25.1 or newer.")
        if "omni" in repo and "nemotron" in repo and ver is not None and ver != (0, 20, 0):
            add("warn", "image_pin_nemotron_omni",
                "NVIDIA's card requires exactly vLLM 0.20.0 for Nemotron 3 Nano Omni; other versions may "
                "fail to load it.")
        if "qwen3.8" in repo and ver is not None and ver < (0, 27, 0):
            add("warn", "image_old_qwen38",
                f"Qwen3.8 needs vLLM 0.27+; {image} is older and untested for it.")
        return out

    # ------------------------------------------------------------------ memory model
    def memory_model(self, model: ModelInfo, params: dict[str, Any]) -> dict[str, float]:
        g = params.get
        tp = int(g("tensor_parallel_size") or 1)
        out: dict[str, float] = {
            "mem_fraction": float(g("gpu_memory_utilization") or 0.92),
            "kv_bytes_per_elem": _KV_BYTES.get(str(g("kv_cache_dtype") or "auto"), 2.0),
            # ~0.6 GiB non-torch (lab: qwen38 "weights + non-torch 22.71" vs 22.13 weights) + NCCL
            # buffers when sharded. An approximation, not a guarantee.
            "overhead_gib": 1.0 + (0.5 * (tp - 1) if tp > 1 else 0.0),
            "max_seqs": float(g("max_num_seqs") or 256),
            "tp": float(tp),
            # lab measurements: 0.59-0.89 GiB CUDA-graph pool on qwen38 (with MTP) v0.27.1.
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
