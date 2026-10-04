"""Ollama adapter: one `ollama serve` that loads models from a store on demand (id `ollama`, `multi_model`).

Models. The core mounts the host's Ollama models directory read-only at `ollama_params.MODELS_DIR` and passes
`ctx.model` = `name:tag` (or "" for a bare server). Only the "ollama" format is loadable: Ollama reads its own
manifest and blob layout, so a
GGUF file or a Hugging Face repo would first have to be imported into a store (`ollama create`), which needs a writable
store and is a later phase. Measured on the pinned image with a read-only mount: the server starts when the store has
`blobs/` and `manifests/` directories (an empty store without `blobs/` fails with "mkdir .../blobs: read-only file system",
so the core should only offer stores that have both), serves `/api/version`, `/api/tags`, `/v1/models` and `/api/ps`
normally, and writes its own identity key under `/root/.ollama` of the container's writable layer (so the container
must not be started with a read-only root file system).

Preloading. Nothing is loaded at start: Ollama loads a model on its first request, so the first chat is slow while the
weights are read. A caller that wants an eager load can POST the body from `preload_request()` after `/api/version`
answers (not part of the locked port; the call is documented by Ollama as loading a model when asked to generate with
no prompt, recalled here and not exercised).

Always on, not tunable (`ollama_params.RESERVED_ENV`): OLLAMA_HOST (0.0.0.0:11434 inside the container), OLLAMA_MODELS,
OLLAMA_NO_CLOUD=1 (no cloud inference or web search: deny-by-default egress) and OLLAMA_NOPRUNE=1 (a start-up prune would
try to delete blobs from the user's real library if it were ever mounted writable).

Metrics. Ollama has no Prometheus endpoint (`GET /metrics` is a 404, measured). `metric_prefixes` is empty and
`parse_metrics` returns {}. `metrics_path` is `/api/version`, a tiny always-200 read-only call that the node's HTTP
allowlist accepts, so the hub's pull sees a healthy empty scrape instead of an endless 404.

Authentication. Ollama has no API key for its local server; isolation comes from the engine network and the loopback
gateway, and from the read-only store (pull, create and delete cannot write).

Fit model. Memory is per loaded model: the KV cache is sized for `context_length x num_parallel` tokens (the Ollama FAQ:
parallel requests multiply the context), `max_len` carries that product, and a model is assumed to sit on one GPU unless
`sched_spread` is on. Ollama may keep up to `max_loaded_models` models resident and offloads layers to the CPU when a model
does not fit; neither is modelled, so the estimate is for one fully offloaded model.
"""
from __future__ import annotations

import math
import re
from typing import Any, Literal

from .base import (
    Compat,
    EngineAdapter,
    Hardware,
    LaunchContext,
    LaunchSpec,
    ModelInfo,
    ParamSpec,
)
from .ollama_params import (
    CONTAINER_PORT,
    DEFAULT_IMAGE,
    KV_BYTES,
    MODELS_DIR,
    OMIT_WHEN_FALSE,
    build_catalog,
)

_GIB = 1024 ** 3
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+\-]{0,254}$")
_DURATION_RE = re.compile(r"^-?(\d+(\.\d+)?(ns|us|µs|ms|s|m|h))+$")
_SECONDS_RE = re.compile(r"^-?\d+$")
_LIBRARY_RE = re.compile(r"^[a-z0-9_]+$")
_CTRL_RE = re.compile(r"[\x00-\x1f\x7f]")
_DEFAULT_BATCH = 512.0          # Ollama's default batch size (recalled); only feeds the activation heuristic


def default_context(total_gib: float) -> int:
    """Context tokens Ollama picks when none is set, by total GPU memory (docs.ollama.com/context-length: below 24 GiB
    4k, from 24 GiB 32k, from 48 GiB 256k). The help text says "4k/32k/256k based on VRAM"; the thresholds are the
    docs', and the start-up line `vram-based default context` reports the figure actually chosen."""
    if total_gib >= 48:
        return 262144
    if total_gib >= 24:
        return 32768
    return 4096


def _num(params: dict[str, Any], key: str, default: float = 0.0) -> float:
    v = params.get(key)
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else default


class OllamaAdapter(EngineAdapter):
    id = "ollama"
    display_name = "Ollama"
    default_image = DEFAULT_IMAGE
    default_port = CONTAINER_PORT
    model_formats = frozenset({"ollama"})
    multi_model = True
    # Not part of the locked port: the hub reads it (when present) to tell each node which metric names to keep.
    metric_prefixes: tuple[str, ...] = ()

    def __init__(self) -> None:
        self._catalog = build_catalog()
        self._by_key: dict[str, ParamSpec] = {p.key: p for p in self._catalog}

    # ------------------------------------------------------------------ catalog / presets
    def param_catalog(self) -> list[ParamSpec]:
        return [p.model_copy(deep=True) for p in self._catalog]

    def presets(self) -> dict[str, dict[str, Any]]:
        return {
            "balanced": {"context_length": 8192, "keep_alive": "30m"},
            "max-throughput": {"num_parallel": 4, "context_length": 8192, "flash_attention": True},
            "low-latency": {"num_parallel": 1, "context_length": 8192, "keep_alive": "-1", "flash_attention": True},
            "long-context": {"context_length": 131072, "num_parallel": 1, "flash_attention": True,
                             "kv_cache_type": "q8_0"},
            "low-memory": {"context_length": 4096, "kv_cache_type": "q8_0", "flash_attention": True,
                           "keep_alive": "1m", "max_loaded_models": 1},
            "many-models": {"max_loaded_models": 3, "keep_alive": "10m", "context_length": 8192},
            "big-model-spread": {"sched_spread": True, "context_length": 16384},
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
        return errs

    @staticmethod
    def _check_value(spec: ParamSpec, val: Any) -> list[str]:
        k, t = spec.key, spec.type
        if t == "bool":
            return [] if isinstance(val, bool) else [f"{k}: expected true/false"]
        if t == "int":
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or int(val) != val:
                return [f"{k}: expected an integer"]
            if spec.min is not None and val < spec.min:
                return [f"{k}: must be >= {spec.min:g}"]
            return []
        if t == "enum":
            return [] if str(val) in (spec.choices or []) else [f"{k}: '{val}' is not one of {', '.join(spec.choices or [])}"]
        if t == "string_list":
            ok = isinstance(val, list) and bool(val) and all(
                isinstance(x, str) and x and not re.search(r"[\s,]", x) and not _CTRL_RE.search(x) for x in val)
            return [] if ok else [f"{k}: expected a non-empty list of origins without spaces or commas"]
        # strings (and numbers for the duration params, which accept a number of seconds)
        if k in ("keep_alive", "load_timeout"):
            text = str(val).strip() if isinstance(val, (str, int)) and not isinstance(val, bool) else ""
            ok = bool(_SECONDS_RE.fullmatch(text) or _DURATION_RE.fullmatch(text))
            if k == "load_timeout" and text.startswith("-"):
                ok = False
            return [] if ok else [f"{k}: expected a duration such as 5m or 24h, or a number of seconds"]
        if not isinstance(val, str) or not val.strip() or _CTRL_RE.search(val):
            return [f"{k}: expected a non-empty string"]
        if k == "llm_library" and not _LIBRARY_RE.fullmatch(val):
            return [f"{k}: expected a library name such as cuda_v13"]
        return []

    # ------------------------------------------------------------------ launch
    @staticmethod
    def _render(spec: ParamSpec, val: Any) -> str:
        if spec.type == "bool":
            return "1" if val else "0"
        if spec.type == "string_list":
            return ",".join(str(x) for x in val)
        return str(val).strip()

    def build_launch(self, model: str, params: dict[str, Any], hw: Hardware, *,
                     served_name: str, hf_cache_container_path: str) -> LaunchSpec:
        """v1 bridge: `model` is an Ollama name:tag (or empty for a bare server)."""
        return self.build_launch_ctx(LaunchContext(model=model, model_format="ollama", model_path=None,
                                                   served_name=served_name,
                                                   hf_cache_container_path=hf_cache_container_path,
                                                   params=params, hw=hw))

    def build_launch_ctx(self, ctx: LaunchContext) -> LaunchSpec:
        if ctx.model_format != "ollama":
            raise ValueError("Ollama loads models from its own store (format 'ollama'); importing a GGUF file or a "
                             f"Hugging Face repo ({ctx.model_format}) into a managed Ollama is not supported yet")
        if ctx.model and not _NAME_RE.fullmatch(ctx.model):
            raise ValueError("model must be an Ollama name such as llama3.1:8b")
        errs = self.validate(ctx.params)
        if errs:
            raise ValueError("; ".join(errs))
        env = {"OLLAMA_HOST": f"0.0.0.0:{CONTAINER_PORT}", "OLLAMA_MODELS": MODELS_DIR,  # noqa: S104 (container-internal)
               "OLLAMA_NO_CLOUD": "1", "OLLAMA_NOPRUNE": "1"}
        for spec in self._catalog:
            val = ctx.params.get(spec.key)
            if val is None or (val is False and spec.key in OMIT_WHEN_FALSE):
                continue
            env[spec.flag[4:]] = self._render(spec, val)
        return LaunchSpec(image=self.default_image, argv=["serve"], env=env, container_port=CONTAINER_PORT,
                          health_path="/api/version", metrics_path="/api/version", models_path="/v1/models",
                          shm_size="1g", ipc_host=False, files={}, needs_hf_cache=False)

    def preload_request(self, model: str, params: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """(path, JSON body) of a POST that loads `model` without generating text. Optional helper for the core."""
        if not _NAME_RE.fullmatch(model):
            raise ValueError("model must be an Ollama name such as llama3.1:8b")
        return "/api/generate", {"model": model, "stream": False}

    # ------------------------------------------------------------------ compatibility
    def compatibility(self, model: ModelInfo, params: dict[str, Any], hw: Hardware) -> list[Compat]:
        out: list[Compat] = []

        def add(level: Literal["ok", "warn", "block"], code: str, message: str) -> None:
            out.append(Compat(level=level, code=code, message=message))

        g = params.get
        ctx = int(_num(params, "context_length"))
        par = max(1, int(_num(params, "num_parallel", 1)))
        total = sum(hw.gpu_total_gib)
        free = sum(hw.gpu_free_gib)
        kv_tok = None
        if model.num_layers and model.num_kv_heads and model.head_dim:
            kv_tok = (2.0 * model.num_layers * model.num_kv_heads * model.head_dim
                      * KV_BYTES.get(str(g("kv_cache_type") or "f16"), 2.0))

        if hw.compute_capability == "12.0":
            add("ok", "blackwell_cuda_build",
                "This GPU is Blackwell (compute capability 12.0): the pinned Ollama image ships CUDA 12 and 13 runners and "
                "the CUDA 12 runner carries sm_120a kernels, so no extra settings are needed.")
        if not ctx and total:
            chosen = default_context(total)
            if chosen >= 32768:
                cost = f", about {chosen * par * kv_tok / _GIB:.1f} GiB of KV cache per loaded model" if kv_tok else ""
                add("warn", "default_context_by_vram",
                    f"No context_length is set, so Ollama picks {chosen} tokens for {total:.0f} GiB of GPU memory{cost}. "
                    "Set context_length to what you need (for example 32768) to keep memory for the model itself.")
        elif ctx and kv_tok and free and ctx * par * kv_tok / _GIB > 0.9 * free:
            add("warn", "kv_exceeds_vram",
                f"A context of {ctx} tokens x {par} request(s) needs about {ctx * par * kv_tok / _GIB:.1f} GiB of KV cache and only "
                f"{free:.1f} GiB of GPU memory is free. Lower context_length or num_parallel, or set kv_cache_type to q8_0.")
        if par > 1:
            add("ok", "parallel_multiplies_context",
                f"num_parallel={par}: Ollama multiplies the context by the number of parallel requests, so every loaded "
                "model holds that many tokens of KV cache.")
        if g("kv_cache_type") in ("q8_0", "q4_0") and g("flash_attention") is False:
            add("warn", "kv_quant_needs_flash_attn",
                f"kv_cache_type={g('kv_cache_type')} only takes effect with flash attention, and flash_attention is off. "
                "Turn flash_attention on or leave it unset.")
        ka = str(g("keep_alive") if g("keep_alive") is not None else "").strip()
        if ka.startswith("-"):
            add("warn", "keep_alive_forever",
                "A negative keep_alive keeps the model in memory until the server stops, so that GPU memory is not "
                "available to other engines. Use a duration such as 30m unless the model must always be hot.")
        if str(g("llm_library") or "").startswith("cpu"):
            add("warn", "cpu_offload_slow", "llm_library selects the CPU: models run without the GPU, which is far slower.")
        return out

    # ------------------------------------------------------------------ memory model
    def memory_model(self, model: ModelInfo, params: dict[str, Any]) -> dict[str, float]:
        g = params.get
        par = max(1, int(_num(params, "num_parallel", 1)))
        ctx = int(_num(params, "context_length"))
        out: dict[str, float] = {
            "mem_fraction": 1.0,        # Ollama places layers by free memory; there is no share to configure
            "kv_bytes_per_elem": KV_BYTES.get(str(g("kv_cache_type") or "f16"), 2.0),
            "overhead_gib": 1.0 + _num(params, "gpu_overhead") / _GIB,
            "cuda_graph_gib": 0.0,
            "max_batched_tokens": _DEFAULT_BATCH,
            "max_seqs": float(par),
        }
        if ctx > 0:
            out["max_len"] = float(ctx * par)   # the KV cache covers every parallel slot (Ollama FAQ)
        if not g("sched_spread"):
            out["tp"] = 1.0                      # one GPU unless the model needs more or spreading is forced
        return out

    # host_memory_gib: the default {} is right. The CPU share of a partly offloaded model is chosen by Ollama's
    # scheduler at load time and cannot be derived from the parameters.

    # ------------------------------------------------------------------ observability
    def parse_metrics(self, prometheus_text: str) -> dict[str, float]:
        """Ollama exports no Prometheus metrics."""
        return {}

    def parse_startup_log(self, line: str) -> dict[str, Any] | None:
        return parse_log_line(line)


# ------------------------------------------------------------------ log lines
_MSG = re.compile(r'\bmsg=(?:"((?:[^"\\]|\\.)*)"|(\S+))')
_LEVEL = re.compile(r"\blevel=(\w+)")
_LISTENING = re.compile(r"Listening on (\S+) \(version ([^)]+)\)")
_FIELD = {name: re.compile(rf'\b{name}=(?:"([^"]*)"|(\S+))')
          for name in ("id", "library", "total", "available", "total_vram")}
_DEFAULT_CTX = re.compile(r"default_num_ctx=(\d+)")
_OFFLOADED = re.compile(r"offloaded (\d+)/(\d+) layers to GPU")
_STARTED = re.compile(r"runner started in ([\d.]+) seconds")
_SIZE = re.compile(r"^([\d.]+)\s*(B|KiB|MiB|GiB|TiB)$")
_UNIT_GIB = {"B": 1 / _GIB, "KiB": 1 / 1024**2, "MiB": 1 / 1024, "GiB": 1.0, "TiB": 1024.0}
_ERRORS: tuple[tuple[str, str], ...] = (
    ("read-only file system", "models_dir_read_only"),
    ("address already in use", "port_in_use"),
    ("cudaMalloc failed", "cuda_oom"),
    ("out of memory", "cuda_oom"),
)


def _field(line: str, name: str) -> str | None:
    m = _FIELD[name].search(line)
    return (m.group(1) if m.group(1) is not None else m.group(2)) if m else None


def _gib(text: str | None) -> float | None:
    m = _SIZE.match(text.strip()) if text else None
    return round(float(m.group(1)) * _UNIT_GIB[m.group(2)], 2) if m else None


def parse_log_line(line: str) -> dict[str, Any] | None:
    """Structured progress for one Ollama log line, or None. Ollama logs `time=... level=... source=... msg="..."`.

    Captured on a bare server (no model): the listening line, the GPU discovery line and the default-context line.
    The runner lines ("offloaded N/M layers", "runner started in X seconds") are the llama.cpp wording and were not
    seen in a live Ollama run here."""
    stripped = line.strip()
    if not stripped:
        return None
    m = _MSG.search(stripped)
    text = (m.group(1) if m and m.group(1) is not None else m.group(2) if m else stripped)
    for needle, code in _ERRORS:
        if needle in stripped:
            return {"error": code, "message": stripped[-300:]}
    if (lm := _LISTENING.search(text)):
        return {"phase": "ready", "pct": 100.0, "listening": lm.group(1), "version": lm.group(2)}
    if text == "inference compute":
        gpu: dict[str, Any] = {"id": _field(stripped, "id"), "library": _field(stripped, "library"),
                               "total_gib": _gib(_field(stripped, "total")),
                               "available_gib": _gib(_field(stripped, "available"))}
        out: dict[str, Any] = {"gpu": gpu}
        if gpu["library"] == "cpu":
            out["warning"] = "cpu_only"         # no GPU was found: models will run on the CPU
        return out
    if text == "vram-based default context" and (dm := _DEFAULT_CTX.search(stripped)):
        return {"default_num_ctx": int(dm.group(1)), "total_vram_gib": _gib(_field(stripped, "total_vram"))}
    if (om := _OFFLOADED.search(text)):
        return {"phase": "loading_weights", "gpu_layers": int(om.group(1)), "total_layers": int(om.group(2))}
    if (sm := _STARTED.search(text)):
        return {"model_load_seconds": float(sm.group(1))}
    if "Ollama cloud disabled" in text:
        return {"cloud_disabled": text.rstrip().endswith("true")}
    lvl = _LEVEL.search(stripped)
    if lvl and lvl.group(1) == "ERROR":
        return {"error": "engine_error", "message": stripped[-300:]}
    if stripped.startswith("Error:"):
        return {"error": "engine_error", "message": stripped[-300:]}
    return None

