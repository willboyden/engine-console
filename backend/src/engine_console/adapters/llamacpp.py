"""llama.cpp adapter: `llama-server` serving ONE GGUF model (id `llamacpp`).

Launch conventions (the image entrypoint is already `/app/llama-server`, so `argv` starts with `-m`):
  * The core mounts the model folder read-only at MODEL_MOUNT and hands over `ctx.model_path` (the first shard of a split
    set; `/models/snapshots/<rev>/<file>` for a repo kept in an HF cache). Nothing is downloaded: `--offline` is always
    passed and no `-hf`/`-mu` flag exists in the catalog.
  * Always on, not tunable: `--host 0.0.0.0` (container-internal; the core publishes to loopback behind a gateway),
    `--port 8080`, `--alias <served name>[,extras]`, `--metrics` (Prometheus needs it), `--offline`.
  * `--cors-origins localhost` unless the user sets `cors_origins`. Measured on the pinned build: with its default (`*`)
    the server reflects ANY origin in `Access-Control-Allow-Origin`, so a web page open in the operator's browser can
    read answers from the loopback port; with `localhost` an unrelated origin gets no such header while local pages
    still work. The console never needs CORS (it calls the engine from the server side).
  * The API key travels as `LLAMA_API_KEY` in the environment, never on argv. With a key set the server still answers
    /health and /v1/models without it (measured), but /metrics returns 401, so live metrics are lost (a compat warning).
  * A custom chat template is written through `LaunchSpec.files` and passed with `--chat-template-file`, because a
    template easily exceeds the 4096-character argument limit of the node protocol.
  * No NCCL and one process: `ipc_host` is False and the shared-memory size is small.

Fit model (what `memory_model` / `host_memory_gib` tell the shared estimator):
  * llama-server splits `-c` over its slots, and the KV buffer is sized for `-c` tokens in total either way, so
    `max_len` is the total context and `max_seqs` only informational.
  * Partial offload (`-ngl N` below the layer count, `--no-kv-offload`, `-dev none`) is reported through two extra keys,
    `weights_gpu_fraction` and `kv_gpu_fraction` (0..1); the estimator (`services/fit.py`) charges the GPU only for those
    shares and notes the rest as host RAM. Layers are
    assumed equal in size, and the KV cache is treated as full attention on every layer (sliding-window and recurrent
    layers would need less).
  * Unset `-c` / `-ngl` with `--fit on` (the engine default) are chosen by llama-server itself at start-up; the
    estimate then uses the model's native context and says so (confidence drops), and `compatibility` explains it.
"""
from __future__ import annotations

import json
import math
import re
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
from .llamacpp_params import (
    CACHE_TYPE_BYTES,
    CHAT_TEMPLATE_PATH,
    CHAT_TEMPLATES,
    CONTAINER_PORT,
    DEFAULT_IMAGE,
    LAYER_KEYS,
    MAX_TEMPLATE_BYTES,
    NEGATABLE_KEYS,
    PATH_KEYS,
    build_catalog,
)
from .llamacpp_parse import parse_log_line, scrape_to_canonical

_GIB = 1024 ** 3
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+\-]*$")
_TENSOR_SPLIT_RE = re.compile(r"^\d+(\.\d+)?(,\d+(\.\d+)?)*$")
_DEVICE_RE = re.compile(r"^(none|[A-Za-z0-9_.\-]+(,[A-Za-z0-9_.\-]+)*)$")
_ORIGINS_RE = re.compile(r"^[A-Za-z0-9*:/.,_\-\[\]]+$")
_CTRL_RE = re.compile(r"[\x00-\x1f\x7f]")
_NO_QUANT_V = frozenset({"f16", "bf16", "f32"})       # the KV types that do not need flash attention
_DRAFT_MODEL_TYPES = frozenset({"draft-simple", "draft-eagle3", "draft-dflash"})
_EXPERT_COUNT_KEYS = ("expert_count", "num_experts", "n_routed_experts", "num_local_experts")
_EXPERT_USED_KEYS = ("expert_used_count", "num_experts_per_tok", "n_experts_per_tok")
_MIN_SLOT_TOKENS = 2048
_FIT_MARGIN_MIB = 1024          # the pinned build's default --fit-target
_CACHE_RAM_MIB = 8192           # the pinned build's default --cache-ram


def _model_path_error(path: str | None, what: str = "model_path") -> str | None:
    """A .gguf file under the model mount: no '..', no control characters, no leading '-' (it would read as a flag)."""
    if not path:
        return f"{what} is required"
    if _CTRL_RE.search(path) or not path.startswith(MODEL_MOUNT + "/") or path.startswith("-"):
        return f"{what} must be a file inside {MODEL_MOUNT}/"
    if ".." in path.split("/"):
        return f"{what} must not contain '..'"
    if not path.lower().endswith(".gguf"):
        return f"{what} must be a .gguf file"
    return None


def _num(params: dict[str, Any], key: str, default: float = 0.0) -> float:
    """A numeric param, or `default` when it is unset (bools are not numbers here)."""
    v = params.get(key)
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else default


def _layer_count(v: Any) -> int | None:
    """An explicit layer count, or None for unset / 'auto' / 'all'."""
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, str) and v.isdigit():
        return int(v)
    return None


def _kv_bytes_per_elem(params: dict[str, Any]) -> float:
    """Mean bytes per KV element over the K and V halves (the shared estimator takes a single number)."""
    k = CACHE_TYPE_BYTES.get(str(params.get("cache_type_k") or "f16"), 2.0)
    v = CACHE_TYPE_BYTES.get(str(params.get("cache_type_v") or "f16"), 2.0)
    return (k + v) / 2.0


def _kv_bytes_per_token(model: ModelInfo, params: dict[str, Any]) -> float | None:
    """K and V of every layer for one token (full attention on every layer); None when the model facts are missing."""
    if not (model.num_layers and model.num_kv_heads and model.head_dim):
        return None
    return 2.0 * model.num_layers * model.num_kv_heads * model.head_dim * _kv_bytes_per_elem(params)


def _offload(params: dict[str, Any], model: ModelInfo) -> tuple[float, float]:
    """(share of the weights on the GPU, share of the KV cache on the GPU). 1.0 when unknown, 'auto' or 'all'.
    The output layer counts as one extra offloadable layer, as in llama.cpp's own "offloaded N/M layers" line."""
    if params.get("device") == "none":
        return 0.0, 0.0
    weights = kv = 1.0
    n = _layer_count(params.get("n_gpu_layers"))
    if n is not None:
        layers = model.num_layers
        if n == 0:
            weights = kv = 0.0
        elif layers:
            weights = min(n, layers + 1) / (layers + 1)
            kv = min(n, layers) / layers
    if params.get("kv_offload") is False:
        kv = 0.0
    return weights, kv


def _raw_int(model: ModelInfo, names: tuple[str, ...]) -> int | None:
    for key, val in model.raw_config.items():
        base = key.rsplit(".", 1)[-1]
        if base in names and isinstance(val, int) and not isinstance(val, bool) and val > 0:
            return val
    return None


def _expert_share(model: ModelInfo) -> float | None:
    """Share of the weight bytes that sit in expert tensors, from parameter counts and expert counts; None when any
    of them is unknown (never guessed)."""
    total, active = model.num_params, model.num_active_params
    experts, used = _raw_int(model, _EXPERT_COUNT_KEYS), _raw_int(model, _EXPERT_USED_KEYS)
    if not (total and active and experts and used) or used >= experts or active >= total:
        return None
    share = (total - active) / (total * (1.0 - used / experts))
    return min(max(share, 0.0), 1.0)


def _moe_cpu_layers(params: dict[str, Any], model: ModelInfo) -> float:
    """Fraction of the layers whose expert weights are forced to the CPU: 0.0 when none, NaN when unknowable."""
    if params.get("cpu_moe"):
        return 1.0
    n = params.get("n_cpu_moe")
    if n:
        return min(int(n) / model.num_layers, 1.0) if model.num_layers else math.nan
    ot = str(params.get("override_tensor") or "")
    if "=CPU" in ot.upper():
        return 1.0 if "exps" in ot else math.nan
    return 0.0


class LlamaCppAdapter(EngineAdapter):
    id = "llamacpp"
    display_name = "llama.cpp"
    default_image = DEFAULT_IMAGE
    default_port = CONTAINER_PORT
    model_formats = frozenset({"gguf"})
    multi_model = False
    # Not part of the locked port: the hub reads it (when present) to tell each node which metric names to keep.
    metric_prefixes: tuple[str, ...] = ("llamacpp:",)

    def __init__(self) -> None:
        self._catalog = build_catalog()
        self._by_key: dict[str, ParamSpec] = {p.key: p for p in self._catalog}

    # ------------------------------------------------------------------ catalog / presets
    def param_catalog(self) -> list[ParamSpec]:
        return [p.model_copy(deep=True) for p in self._catalog]

    def presets(self) -> dict[str, dict[str, Any]]:
        # n_gpu_layers is left on 'auto' in every preset except the MoE one: with the engine's own fit on, a model that
        # does not fit is offloaded partly instead of failing to start.
        return {
            "balanced": {"ctx_size": 16384, "parallel": 1, "flash_attn": "on"},
            "max-throughput": {"parallel": 8, "ctx_size": 65536, "kv_unified": True, "cont_batching": True,
                               "ubatch_size": 1024, "flash_attn": "on"},
            "low-latency": {"parallel": 1, "ctx_size": 8192, "flash_attn": "on", "cache_reuse": 256},
            "long-context": {"ctx_size": 131072, "parallel": 1, "cache_type_k": "q8_0", "cache_type_v": "q8_0",
                             "flash_attn": "on"},
            "tool-agent": {"jinja": True, "ctx_size": 65536, "parallel": 2, "flash_attn": "on", "cache_reuse": 256},
            "reasoning": {"jinja": True, "reasoning": "on", "reasoning_format": "deepseek", "ctx_size": 32768,
                          "parallel": 1, "flash_attn": "on"},
            "big-model-offload": {"fit": "on", "ctx_size": 8192, "parallel": 1, "cache_type_k": "q8_0",
                                  "cache_type_v": "q8_0", "flash_attn": "on"},
            "moe-experts-on-cpu": {"cpu_moe": True, "n_gpu_layers": "all", "ctx_size": 32768, "parallel": 1,
                                   "flash_attn": "on"},
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
        if params.get("chat_template") and params.get("chat_template_jinja"):
            errs.append("chat_template and chat_template_jinja are alternatives: set only one")
        return errs

    def _check_value(self, spec: ParamSpec, val: Any) -> list[str]:
        k, t = spec.key, spec.type
        if k in LAYER_KEYS:
            ok = (isinstance(val, int) and not isinstance(val, bool) and val >= 0) or \
                (isinstance(val, str) and (val in ("auto", "all") or val.isdigit()))
            return [] if ok else [f"{k}: expected a layer count, 'auto' or 'all'"]
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
            return [] if str(val) in (spec.choices or []) else [f"{k}: '{val}' is not one of {', '.join(spec.choices or [])}"]
        if t == "string":
            return self._check_string(k, val)
        if t == "string_list":
            if not isinstance(val, list) or not val or not all(isinstance(x, str) and _NAME_RE.fullmatch(x) for x in val):
                return [f"{k}: expected a non-empty list of names (letters, digits and . _ : / + -)"]
            return []
        if t == "json":
            try:
                obj = json.loads(val) if isinstance(val, str) else val
            except json.JSONDecodeError as e:
                return [f"{k}: invalid JSON ({e.msg})"]
            return [] if isinstance(obj, dict) else [f"{k}: expected a JSON object"]
        return []

    @staticmethod
    def _check_string(k: str, val: Any) -> list[str]:
        if not isinstance(val, str) or not val.strip():
            return [f"{k}: expected a non-empty string"]
        if k == "chat_template_jinja":
            if "\x00" in val:
                return [f"{k}: must not contain NUL characters"]
            return [] if len(val.encode()) <= MAX_TEMPLATE_BYTES else [f"{k}: template is over {MAX_TEMPLATE_BYTES} bytes"]
        if k == "chat_template":
            if "\x00" in val or len(val) > 4000:
                return [f"{k}: too long for the command line (over 4000 characters): use chat_template_jinja"]
            return []
        if _CTRL_RE.search(val):
            return [f"{k}: must not contain control characters"]
        if k == "api_key":
            return [] if not re.search(r"\s", val) and len(val) <= 512 else [f"{k}: no whitespace, at most 512 characters"]
        if val.startswith("-"):
            return [f"{k}: must not start with '-' (it would be read as a flag)"]
        if k in PATH_KEYS:
            items = val.split(",") if k in ("lora", "lora_scaled") else [val]
            for item in items:
                path = item.rsplit(":", 1)[0] if k == "lora_scaled" else item
                if (err := _model_path_error(path, k)) is not None:
                    return [err]
                if k == "lora_scaled" and not re.fullmatch(r".+:-?\d+(\.\d+)?", item):
                    return [f"{k}: entries look like /models/<file>:<scale>"]
            return []
        if k == "tensor_split":
            return [] if _TENSOR_SPLIT_RE.fullmatch(val) else [f"{k}: expected proportions such as 3,1"]
        if k == "device":
            return [] if _DEVICE_RE.fullmatch(val) else [f"{k}: expected device names such as CUDA0,CUDA1, or none"]
        if k == "cors_origins":
            return [] if _ORIGINS_RE.fullmatch(val) else [f"{k}: expected origins such as localhost or https://example.org"]
        if k == "override_tensor":
            return [] if all("=" in part for part in val.split(",")) else [f"{k}: expected pattern=buffer entries"]
        if k == "override_kv":
            return [] if all("=" in p and ":" in p for p in val.split(",")) else [f"{k}: expected KEY=TYPE:VALUE entries"]
        return []

    # ------------------------------------------------------------------ launch
    @staticmethod
    def _render(spec: ParamSpec, val: Any) -> list[str]:
        """CLI tokens for one param. bool -> --x, or --no-x when the flag has a negation (false is otherwise omitted)."""
        if spec.type == "bool":
            if val:
                return [spec.flag]
            return ["--no-" + spec.flag[2:]] if spec.key in NEGATABLE_KEYS else []
        if spec.type == "json":
            obj = json.loads(val) if isinstance(val, str) else val
            return [spec.flag, json.dumps(obj, separators=(",", ":"))]
        if spec.type == "float":
            return [spec.flag, format(float(val), "g")]
        if spec.type == "int":
            return [spec.flag, str(int(val))]
        return [spec.flag, str(val)]

    def build_launch(self, model: str, params: dict[str, Any], hw: Hardware, *,
                     served_name: str, hf_cache_container_path: str) -> LaunchSpec:
        """v1 bridge. llama.cpp serves a mounted GGUF, so `model` must already be a path under the model mount."""
        path = model if model.startswith(MODEL_MOUNT + "/") else None
        return self.build_launch_ctx(LaunchContext(model=model, model_format="gguf", model_path=path,
                                                   served_name=served_name,
                                                   hf_cache_container_path=hf_cache_container_path,
                                                   params=params, hw=hw))

    def build_launch_ctx(self, ctx: LaunchContext) -> LaunchSpec:
        if ctx.model_format != "gguf":
            raise ValueError(f"llama.cpp loads GGUF files only, not {ctx.model_format} models")
        if (path_err := _model_path_error(ctx.model_path)) is not None:
            raise ValueError(path_err)
        errs = self.validate(ctx.params)
        if errs:
            raise ValueError("; ".join(errs))
        served = ctx.served_name.strip()
        if not served or "," in served or served.startswith("-") or _CTRL_RE.search(served):
            raise ValueError("served name must be non-empty, without commas or control characters")
        params = ctx.params
        names = [served] + [n for n in params.get("served_model_name") or [] if n != served]
        argv = ["-m", str(ctx.model_path), "--host", "0.0.0.0",  # noqa: S104 (container-internal; core publishes to loopback)
                "--port", str(CONTAINER_PORT), "--alias", ",".join(names), "--metrics", "--offline"]
        env: dict[str, str] = {}
        files: dict[str, str] = {}
        if params.get("jinja") is None and (params.get("chat_template_jinja") or (
                params.get("chat_template") and params["chat_template"] not in CHAT_TEMPLATES)):
            argv.append("--jinja")        # the help: a custom template is accepted only with --jinja set before it
        for spec in self._catalog:
            val = params.get(spec.key)
            if val is None or spec.key == "served_model_name":
                continue
            if spec.flag.startswith("env:"):
                env[spec.flag[4:]] = str(val)
            elif spec.key == "chat_template_jinja":
                files[CHAT_TEMPLATE_PATH] = str(val)
                argv += [spec.flag, CHAT_TEMPLATE_PATH]
            else:
                argv += self._render(spec, val)
        if params.get("cors_origins") is None:
            argv += ["--cors-origins", "localhost"]
        return LaunchSpec(image=self.default_image, argv=argv, env=env, container_port=CONTAINER_PORT,
                          health_path="/health", metrics_path="/metrics", models_path="/v1/models",
                          shm_size="1g", ipc_host=False, files=files, needs_hf_cache=False)

    # ------------------------------------------------------------------ compatibility
    def compatibility(self, model: ModelInfo, params: dict[str, Any], hw: Hardware) -> list[Compat]:
        out: list[Compat] = []

        def add(level: Literal["ok", "warn", "block"], code: str, message: str) -> None:
            out.append(Compat(level=level, code=code, message=message))

        g = params.get
        n_gpu = len(hw.gpu_ids)
        ctx = int(g("ctx_size") or 0)
        split = g("split_mode")
        native = model.max_position_embeddings
        kv_tok = _kv_bytes_per_token(model, params)
        w_frac, kv_frac = _offload(params, model)

        # Blackwell: the pinned image carries sm_120a kernels (CUDA 12.8).
        if hw.compute_capability == "12.0":
            add("ok", "blackwell_cuda_build",
                "This GPU is Blackwell (compute capability 12.0): the pinned llama.cpp image is built with CUDA 12.8 and "
                "carries sm_120a kernels, so no extra settings are needed.")
        # The engine's own fit.
        fit_on = g("fit") != "off"
        blockers = [name for name, on in (("tensor_split", bool(g("tensor_split"))),
                                          ("override_tensor", bool(g("override_tensor"))),
                                          (f"split_mode={split}", split in ("row", "tensor"))) if on]
        unset = [name for name, is_unset in (("ctx_size", not ctx), ("n_gpu_layers", g("n_gpu_layers") in (None, "auto")))
                 if is_unset]
        if fit_on and not blockers and unset:
            margin = int(_num(params, "fit_target", _FIT_MARGIN_MIB))
            add("ok", "fit_auto_adjust",
                f"llama-server's own fit is on: it will choose {' and '.join(unset)} so the model fits the free GPU "
                f"memory, keeping {margin} MiB free per GPU. The start-up log says what it picked.")
        elif (not fit_on or blockers) and not ctx:
            why = "fit is off" if not fit_on else f"{', '.join(blockers)} turns the automatic fit off"
            size = ""
            if native and kv_tok:
                size = f" ({native} tokens needs about {native * kv_tok * kv_frac / _GIB:.1f} GiB of KV cache)"
            add("warn", "fit_auto_adjust",
                f"Nothing will shrink the context for you ({why}), so the engine allocates the model's full trained "
                f"context{size}. Set ctx_size to a value that fits.")
        # KV cache type needs flash attention.
        if g("cache_type_v") not in (None, *_NO_QUANT_V) and g("flash_attn") == "off":
            add("block", "kv_quant_needs_flash_attn",
                f"cache_type_v={g('cache_type_v')} needs flash attention: llama-server refuses to start with 'V cache "
                "quantization requires flash_attn'. Set flash_attn to on (or auto), or use f16 for the values.")
        # Multi-GPU placement.
        if g("tensor_split"):
            used = sum(1 for x in str(g("tensor_split")).split(",") if float(x) > 0)
            if n_gpu and used > n_gpu:
                add("warn", "tensor_split_mismatch",
                    f"tensor_split names {used} GPUs but only {n_gpu} {'is' if n_gpu == 1 else 'are'} selected. "
                    "Give one proportion per selected GPU.")
        if g("main_gpu") is not None and n_gpu and int(_num(params, "main_gpu")) >= n_gpu:
            add("block", "main_gpu_out_of_range",
                f"main_gpu={g('main_gpu')} but the container sees {n_gpu} GPU(s), numbered from 0. Pick a lower index.")
        if split == "none" and n_gpu > 1:
            add("warn", "split_mode_none_multi_gpu",
                f"split_mode=none uses one GPU, yet {n_gpu} are selected: the others stay idle. Select one GPU, or use "
                "the layer split to share the model.")
        if split in ("row", "tensor") and n_gpu > 1:
            add("warn", "tp_over_pcie",
                "Row and tensor splits exchange data between the GPUs for every layer. Without NVLink that goes over "
                "PCIe and can be slower than the default layer split; prefer one model per card unless it does not fit.")
        if split == "tensor":
            level: Literal["warn", "block"] = "block" if g("flash_attn") == "off" else "warn"
            add(level, "split_mode_tensor_experimental",
                "The tensor split is marked EXPERIMENTAL in the help, needs flash attention, and turns the automatic fit "
                "off. " + ("Set flash_attn to on or auto." if level == "block" else "Prefer the layer split."))
        # CPU offload.
        n = _layer_count(g("n_gpu_layers"))
        if g("device") == "none" or n == 0:
            add("warn", "cpu_offload_slow",
                "No layers are offloaded: the whole model runs on the CPU, which is far slower than a GPU.")
        elif w_frac < 1.0 and n is not None and model.num_layers:
            add("warn", "cpu_offload_slow",
                f"n_gpu_layers={n} of {model.num_layers + 1} layers: the rest run on the CPU and every token waits for them. "
                "Expect a large slowdown.")
        # Context beyond what the model was trained for, and context per slot.
        par = g("parallel")
        unified = g("kv_unified") is True or par in (None, -1)
        per_slot = ctx if (unified or not par or par < 2) else ctx // par
        if ctx and native and per_slot > native:
            if g("rope_scaling") in ("yarn", "linear") or g("rope_scale"):
                add("warn", "context_beyond_native",
                    f"A {per_slot}-token context is past the {native} the model was trained on; rope scaling stretches it, "
                    "but short-prompt quality can drop. Measure on your own prompts.")
            else:
                factor = math.ceil(per_slot / native)
                add("warn", "context_beyond_native",
                    f"A {per_slot}-token context is past the {native} the model was trained on and quality degrades "
                    f"without scaling. Set rope_scaling=yarn with rope_scale={factor} and yarn_orig_ctx={native}, or lower "
                    "ctx_size.")
        if ctx and par and par > 1 and g("kv_unified") is not True and ctx // par < _MIN_SLOT_TOKENS:
            add("warn", "ctx_per_slot_small",
                f"ctx_size {ctx} over {par} slots leaves {ctx // par} tokens for each conversation, so ordinary prompts "
                "are cut off. Raise ctx_size, lower parallel, or turn on kv_unified so slots share the budget.")
        # KV cache against free VRAM.
        if ctx and kv_tok and hw.gpu_free_gib:
            free = hw.gpu_free_gib[0] if split == "none" else sum(hw.gpu_free_gib)
            kv_gib = ctx * kv_tok * kv_frac / _GIB
            if kv_gib > 0.9 * free:
                add("warn", "kv_exceeds_vram",
                    f"At ctx_size {ctx} the KV cache alone needs about {kv_gib:.1f} GiB and only {free:.1f} GiB of GPU memory "
                    "is free. Lower ctx_size, set cache_type_k and cache_type_v to q8_0 (about half the size), or set "
                    "kv_offload=false (slow).")
        # Authentication, templates, speculation, MoE placement.
        if g("api_key"):
            add("warn", "api_key_blocks_metrics",
                "With an API key set, llama-server still answers /health and /v1/models but returns 401 for /metrics, so "
                "live metrics for this instance are lost. The engine is on an isolated network behind a loopback gateway; "
                "leave the key unset unless something else needs it.")
        if g("jinja") is False:
            add("warn", "jinja_disabled",
                "Without the Jinja engine tool calls and reasoning output are not parsed, and most modern chat templates "
                "do not render. Turn jinja back on unless you use one of the built-in simple templates.")
        stype = g("spec_type")
        if stype in _DRAFT_MODEL_TYPES and not g("spec_draft_model"):
            add("warn", "spec_type_needs_model",
                f"spec_type={stype} needs a draft model file: set spec_draft_model (a GGUF inside the model folder).")
        elif stype == "draft-mtp":
            add("warn", "spec_type_needs_model",
                "spec_type=draft-mtp needs a GGUF that contains the model's own MTP layers; llama-server stops with "
                "'context type MTP requested but model doesn't contain MTP layers' otherwise.")
        if _moe_cpu_layers(params, model) != 0.0:       # NaN (unknowable) also counts as "some"
            if not model.is_moe:
                add("warn", "moe_experts_on_cpu",
                    "cpu_moe / n_cpu_moe / override_tensor only move mixture-of-experts weights, and this model is not one.")
            else:
                hint = (" The engine logs a hint to turn memory-mapping off (load_mode none) when tensors are overridden "
                        "to the CPU." if g("load_mode") in (None, "mmap") else "")
                add("ok", "moe_experts_on_cpu",
                    "Expert weights stay in host RAM, usually most of an MoE model's size, while attention and shared "
                    "layers run on the GPU. Speed then depends on RAM bandwidth." + hint)
        return out

    # ------------------------------------------------------------------ memory model
    def memory_model(self, model: ModelInfo, params: dict[str, Any]) -> dict[str, float]:
        g = params.get
        fit_margin_gib = _num(params, "fit_target", _FIT_MARGIN_MIB) / 1024
        out: dict[str, float] = {
            "mem_fraction": 1.0,       # no VRAM share to configure: llama.cpp takes what it needs, margin below
            "kv_bytes_per_elem": _kv_bytes_per_elem(params),
            "overhead_gib": max(1.0, fit_margin_gib) if g("fit") != "off" else 1.0,
            "cuda_graph_gib": 0.0,
            "max_batched_tokens": _num(params, "ubatch_size", 512.0),
        }
        ctx = int(_num(params, "ctx_size"))
        if ctx > 0:
            out["max_len"] = float(ctx)   # the KV buffer is sized for this many tokens in total, whatever the slot count
        if _num(params, "parallel") > 0:
            out["max_seqs"] = _num(params, "parallel")
        tensor_split = [float(x) for x in str(g("tensor_split")).split(",")] if g("tensor_split") else []
        if g("split_mode") == "none":
            out["tp"] = 1.0
        elif tensor_split:
            out["tp"] = float(max(1, sum(1 for x in tensor_split if x > 0)))
        w_frac, kv_frac = _offload(params, model)
        if w_frac < 1.0:
            out["weights_gpu_fraction"] = w_frac
        if kv_frac < 1.0:
            out["kv_gpu_fraction"] = kv_frac
        return out

    def host_memory_gib(self, model: ModelInfo, params: dict[str, Any], hw: Hardware) -> dict[str, float]:
        """Host RAM on top of the engine baseline. Weights that stay on the CPU count even though a memory-mapped file is
        reclaimable page cache: evicting them would turn every token into disk reads. NaN means needed but unknowable."""
        g = params.get
        out: dict[str, float] = {}
        w_frac, kv_frac = _offload(params, model)
        weights_gib = model.weight_bytes / _GIB if model.weight_bytes else None
        if w_frac < 1.0:
            out["cpu_resident_weights"] = weights_gib * (1.0 - w_frac) if weights_gib is not None else math.nan
        moe = _moe_cpu_layers(params, model)
        if moe != 0.0:
            share = _expert_share(model)
            if weights_gib is None or share is None or math.isnan(moe):
                out["moe_expert_weights_on_cpu"] = math.nan
            else:
                # experts of layers that are on the CPU anyway are already in cpu_resident_weights
                out["moe_expert_weights_on_cpu"] = weights_gib * share * moe * w_frac
        if kv_frac < 1.0:
            kv_tok, ctx = _kv_bytes_per_token(model, params), int(g("ctx_size") or 0)
            out["kv_cache_on_cpu"] = ctx * kv_tok * (1.0 - kv_frac) / _GIB if kv_tok and ctx else math.nan
        cache = g("cache_ram")
        cache_mib = _CACHE_RAM_MIB if cache is None else int(cache)
        if cache_mib < 0:
            out["prompt_cache"] = math.nan          # no limit: it grows with use
        elif cache_mib > 0:
            out["prompt_cache"] = cache_mib / 1024  # a ceiling, reached only when many distinct prompts were cached
        return out

    # ------------------------------------------------------------------ observability
    def parse_metrics(self, prometheus_text: str) -> dict[str, float]:
        return scrape_to_canonical(prometheus_text)

    def parse_startup_log(self, line: str) -> dict[str, Any] | None:
        return parse_log_line(line)
