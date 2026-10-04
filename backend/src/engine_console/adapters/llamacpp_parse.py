"""Pure parsers for llama-server Prometheus text and log lines (no I/O, no engine imports, no state).

Metrics. llama-server's `/metrics` (switched on by `--metrics`) carries no labels. Its names were read from the string
table of `libllama-server-impl.so` in the pinned image and compared with the metrics list in the server README at the
same build tag; no live scrape was possible without loading a model, so the test fixture is synthetic:

    counters  llamacpp:prompt_tokens_total  prompt_seconds_total  tokens_predicted_total
              tokens_predicted_seconds_total  n_decode_total  n_tokens_max
    gauges    llamacpp:prompt_tokens_seconds  predicted_tokens_seconds  requests_processing  requests_deferred
              n_busy_slots_per_decode

This build exports NO KV-cache metric (older builds had `kv_cache_usage_ratio` and `kv_cache_tokens`; both are read when
present). The two `*_seconds` gauges are llama-server's own averages: per-slot speed while generating, not the aggregate
over all slots, and whether the window is "since the last scrape" or longer is not verified here. The parser keeps no
state, because the metrics have no label that tells two engine instances apart.

Log lines. Real lines of this build look like `0.00.007.493 I srv    load_model: loading model '/models/x.gguf'`: a
timestamp, a level letter, a source tag and a function name. The patterns below match message text only and ignore that
prefix, so they survive a different prefix or its absence. The captured samples are the no-model failure run and the
router-mode start-up; the model-loading lines are built from the format strings found in the image's libraries (see
`tests/llamacpp/fixtures/startup_synthetic.log`, clearly marked synthetic).
"""
from __future__ import annotations

import math
import re
from typing import Any

PREFIX = "llamacpp:"
_LINE = re.compile(r'^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{(.*)\})?\s+(\S+)(?:\s+\S+)?\s*$')


def parse_samples(text: str) -> list[tuple[str, float]]:
    """(name, value) per sample line; comments, labels and malformed lines are dropped. NaN values are kept out."""
    out: list[tuple[str, float]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m or not m.group(1).startswith(PREFIX):
            continue
        try:
            value = float(m.group(3))
        except ValueError:
            continue
        if not math.isnan(value):
            out.append((m.group(1)[len(PREFIX):], value))
    return out


def scrape_to_canonical(text: str) -> dict[str, float]:
    """The canonical keys of docs/ARCHITECTURE.md section 5 that llama-server can supply; absent keys are omitted."""
    sums: dict[str, float] = {}
    for name, value in parse_samples(text):
        sums[name] = sums.get(name, 0.0) + value       # more than one series (router mode, forks): add them up

    def get(*names: str) -> float | None:
        for n in names:
            # counters carry `_total` on the wire; a build or fork that drops (or adds) it is read the same way
            for cand in (n, n[:-6] if n.endswith("_total") else n + "_total"):
                if cand in sums:
                    return sums[cand]
        return None

    out: dict[str, float] = {}

    def put(key: str, val: float | None, scale: float = 1.0) -> None:
        if val is not None:
            out[key] = val * scale

    put("requests_running", get("requests_processing"))
    put("requests_waiting", get("requests_deferred"))
    put("prompt_tokens_total", get("prompt_tokens_total"))
    put("generation_tokens_total", get("tokens_predicted_total"))
    put("prompt_tps", get("prompt_tokens_seconds"))
    put("generation_tps", get("predicted_tokens_seconds"))
    put("kv_cache_usage_pct", get("kv_cache_usage_ratio"), 100.0)     # not exported by the pinned build
    return out


# ------------------------------------------------------------------ log lines
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_PREFIX = re.compile(r"^\s*\d+\.\d{2}\.\d{3}\.\d{3}\s+[A-Z]\s+")
_DOTS = re.compile(r"^\.{10,}$")
_LOADING = re.compile(r"\bloading model '")
_META = re.compile(r"loaded meta data with (\d+) key-value pairs and (\d+) tensors from (\S+)")
_OFFLOADED = re.compile(r"offloaded (\d+)/(\d+) layers to GPU")
_MODEL_BUF = re.compile(r"(\S+)\s+model buffer size\s*=\s*([\d.]+) MiB")
_KV_TOTAL = re.compile(r"size\s*=\s*([\d.]+) MiB \(\s*(\d+) cells,\s*(\d+) layers")
_KV_BUF = re.compile(r"(\S+)\s+KV buffer size\s*=\s*([\d.]+) MiB")
_N_CTX = re.compile(r"\bn_ctx\s+=\s+(\d+)")
_N_CTX_TRAIN = re.compile(r"\bn_ctx_train\s+=\s+(\d+)")
_SLOTS = re.compile(r"initializing(?: slots)?, n_slots = (\d+)(?:, n_ctx_slot = (\d+))?")
_AUTO_PARALLEL = re.compile(r"n_parallel is set to auto, using n_parallel = (\d+) and kv_unified = (true|false)")
_FIT_CTX = re.compile(r"context size reduced from (\d+) to (\d+)")
_FIT_PROJECTED = re.compile(r"projected to use (\d+) MiB of device memory vs\. (\d+) MiB of free device memory")
_FIT_FAILED = re.compile(r"(?:failed to fit params to free device memory|encountered an error while trying to fit "
                         r"params to free device memory): (.*)")
_CUDA_DEVICES = re.compile(r"found (\d+) CUDA devices")
_CUDA_DEVICE = re.compile(r"Device (\d+): (.+?), compute capability (\d+\.\d+), VMM: (\w+), VRAM: (\d+) MiB")
_BUILD = re.compile(r"build (\d+) \(([0-9a-f]+)\) with ")
_LISTENING = re.compile(r"\blistening on (\S+)")
_READY = re.compile(r"(?:^|\s)model loaded\s*$|all slots are idle")

# (needle, error code), most specific first: the first needle found in a line decides.
_ERRORS: tuple[tuple[str, str], ...] = (
    ("failed to open GGUF file", "model_file_missing"),
    ("cudaMalloc failed", "cuda_oom"),
    ("out of memory", "cuda_oom"),
    ("V cache quantization requires flash_attn", "kv_quant_needs_flash_attn"),
    ("SPLIT_MODE_TENSOR requires flash_attn", "split_tensor_needs_flash_attn"),
    ("couldn't bind HTTP server socket", "port_in_use"),
    ("unsupported model architecture", "unsupported_architecture"),
    ("failed to create llama_context", "context_create_failed"),
    ("exiting due to HTTP server error", "http_server_failed"),
    ("failed to initialize HTTP server", "http_server_failed"),
    ("failed to load model", "model_load_failed"),
    ("exiting due to model loading error", "model_load_failed"),
)
_ALLOC_CUDA = re.compile(r"failed to allocate CUDA\d* buffer")


def parse_log_line(line: str) -> dict[str, Any] | None:
    """Structured progress for one llama-server log line, or None when it carries none.

    Keys: `phase` in {loading_weights, warming_up, ready} with `pct`; `error` (a code) with `message`; `warning`;
    plus facts such as `gpu_layers`, `total_layers`, `kv_cache_gib`, `kv_cache_tokens`, `n_slots`, `n_ctx_slot`, `gpu`,
    `fit_ctx_reduced`. A line yields the first thing it matches, errors first."""
    text = _PREFIX.sub("", _ANSI.sub("", line)).strip()
    if not text:
        return None
    for needle, code in _ERRORS:
        if needle in text:
            return {"error": code, "message": text[-300:]}
    if _ALLOC_CUDA.search(text):
        return {"error": "cuda_oom", "message": text[-300:]}
    if _DOTS.match(text):
        return {"phase": "loading_weights", "pct": float(min(len(text), 100))}
    if _LOADING.search(text):
        return {"phase": "loading_weights", "pct": 0.0}
    if (m := _META.search(text)):
        return {"phase": "loading_weights", "pct": 0.0, "tensors": int(m.group(2))}
    if "loading model tensors" in text:
        return {"phase": "loading_weights", "pct": 0.0}
    if (m := _OFFLOADED.search(text)):
        out: dict[str, Any] = {"phase": "loading_weights", "gpu_layers": int(m.group(1)), "total_layers": int(m.group(2))}
        if int(m.group(1)) == 0:
            out["warning"] = "cpu_only"        # the GPU was not used at all: no device visible, or --n-gpu-layers 0
        return out
    if (m := _MODEL_BUF.search(text)):
        return {"buffer": m.group(1), "model_buffer_gib": round(float(m.group(2)) / 1024, 3)}
    if (m := _KV_TOTAL.search(text)):
        return {"kv_cache_gib": round(float(m.group(1)) / 1024, 3), "kv_cache_tokens": int(m.group(2))}
    if (m := _KV_BUF.search(text)):
        return {"kv_buffer": {"device": m.group(1), "gib": round(float(m.group(2)) / 1024, 3)}}
    if (m := _AUTO_PARALLEL.search(text)):
        return {"n_slots": int(m.group(1)), "kv_unified": m.group(2) == "true"}
    if (m := _SLOTS.search(text)):
        slots: dict[str, Any] = {"n_slots": int(m.group(1))}
        if m.group(2):
            slots["n_ctx_slot"] = int(m.group(2))
        return slots
    if (m := _N_CTX.search(text)):
        return {"n_ctx": int(m.group(1))}
    if (m := _N_CTX_TRAIN.search(text)):
        return {"n_ctx_train": int(m.group(1))}
    if "warming up the model" in text:
        return {"phase": "warming_up"}
    if (m := _FIT_CTX.search(text)):
        return {"fit_ctx_reduced": {"from": int(m.group(1)), "to": int(m.group(2))}}
    if (m := _FIT_PROJECTED.search(text)):
        return {"fit_projected_mib": int(m.group(1)), "fit_free_mib": int(m.group(2))}
    if (m := _FIT_FAILED.search(text)):
        return {"warning": "fit_failed", "message": m.group(1)[-300:]}
    if (m := _CUDA_DEVICE.search(text)):
        return {"gpu": {"index": int(m.group(1)), "name": m.group(2), "compute_capability": m.group(3),
                        "vram_gib": round(int(m.group(5)) / 1024, 2)}}
    if (m := _CUDA_DEVICES.search(text)):
        return {"gpus_found": int(m.group(1))}
    if "failed to mlock" in text:
        return {"warning": "mlock_failed", "message": text[-300:]}
    if "entering sleeping state" in text:
        return {"state": "sleeping"}
    if "exiting sleeping state" in text:
        return {"state": "awake"}
    if (m := _BUILD.search(text)):
        return {"build": m.group(1), "commit": m.group(2)}
    if _READY.search(text):
        return {"phase": "ready", "pct": 100.0}
    if (m := _LISTENING.search(text)):
        return {"listening": m.group(1)}
    return None
