"""Pure parsers for TensorRT-LLM metrics text and container log lines (no I/O, no engine imports).

What `trtllm-serve` 1.2.1 exposes, read from `serve/openai_server.py` and `executor/` in the pinned image, and
checked with the engine's own classes where noted (tests/trtllm/fixtures/README.md lists what was run):

  * ``/metrics`` is NOT Prometheus. It answers with one compact JSON line (a Starlette JSONResponse emits no
    newlines): an array with one object per engine iteration since the previous read, and reading drains them. The
    array stays empty until `enable_iter_perf_stats: true` is in the engine options and a first request has been
    submitted. Each read takes at least two seconds: the engine's own `IterationResult` waits two seconds for further
    items (measured), so a scraper whose read timeout is two seconds or less never gets an answer. The object keys
    are those of the C++ `IterationStats` serializer (fixture `iter_stats_1.2.1.json`, values arbitrary).
  * ``/prometheus/metrics`` exists only with `return_perf_metrics: true`. It carries `trtllm_*` histograms
    (time to first token, time per output token, end-to-end latency, queue time) and one counter
    (`trtllm_request_success_total`); there are no token counters, queue gauges or KV-cache gauges in it.
    The console's node does not forward that path (its allowlist covers `/metrics` only), so the JSON
    parser is the one that matters; the Prometheus parser is kept for the day the path is reachable.

`parse_metrics_text` accepts either body. A scraper that clips the body (the node caps one scrape at 512 KiB)
leaves a truncated JSON array: complete objects are still used. No state is kept between calls: iteration stats
carry no instance identity, so cross-scrape rates cannot be keyed safely. Rates are computed inside one scrape
window instead.

Log lines. The Python logger prints ``[MM/DD/YYYY-HH:MM:SS] [TRT-LLM] [RANK n] [I] message``; the C++ runtime and
uvicorn use other prefixes, so every pattern below matches on the message text only. Message texts come from the
image's sources (`model_loader.py`, `_util.py`, `model_engine.py`, `resource_manager.py`, `py_executor_creator.py`,
and the compiled library for the `[MemUsageChange]` line).
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from datetime import datetime
from statistics import median
from typing import Any

_LINE = re.compile(r'^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{(.*)\})?\s+(\S+)(?:\s+\S+)?\s*$')
_LABEL = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="((?:[^"\\]|\\.)*)"')
_TS_FORMAT = "%m-%d-%Y %H:%M:%S.%f"          # IterationStats.timestamp, as set in py_executor._get_init_iter_stats


def _num(v: Any) -> float | None:
    """A finite float, or None (bools and strings are not numbers here)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return None
    return float(v)


# ------------------------------------------------------------------ Prometheus text
def parse_samples(text: str) -> list[tuple[str, dict[str, str], float]]:
    """Parse Prometheus exposition text into (name, labels, value); skips comments and bad lines."""
    out: list[tuple[str, dict[str, str], float]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        try:
            value = float(m.group(3))
        except ValueError:
            continue
        labels = {k: v.replace('\\"', '"').replace("\\\\", "\\") for k, v in _LABEL.findall(m.group(2) or "")}
        out.append((m.group(1), labels, value))
    return out


def histogram_quantile(q: float, buckets: list[tuple[float, float]]) -> float | None:
    """Prometheus-style quantile with linear interpolation inside the bucket (lifetime counts: the caller diffs).

    A quantile landing in the +Inf bucket returns the highest finite bound, a lower bound on the truth."""
    if not buckets or buckets[-1][1] <= 0:
        return None
    total = buckets[-1][1]
    rank = q * total
    prev_le, prev_count = 0.0, 0.0
    for le, count in buckets:
        if count >= rank:
            if math.isinf(le):
                finite = [b for b, _ in buckets if not math.isinf(b)]
                return finite[-1] if finite else None
            span = count - prev_count
            if span <= 0:
                return le
            return prev_le + (le - prev_le) * ((rank - prev_count) / span)
        prev_le, prev_count = le, count
    return None


def _histogram(samples: list[tuple[str, dict[str, str], float]], name: str) -> list[tuple[float, float]] | None:
    """Merged cumulative buckets for `<name>_bucket`, summed over every other label set."""
    merged: dict[float, float] = defaultdict(float)
    for sample, labels, value in samples:
        if sample != name + "_bucket" or "le" not in labels:
            continue
        le = labels["le"]
        try:
            bound = math.inf if le in ("+Inf", "Inf") else float(le)
        except ValueError:
            continue
        merged[bound] += value
    return sorted(merged.items()) if merged else None


_PROM_HISTOGRAMS = (
    ("ttft_p50_s", 0.5, "trtllm_time_to_first_token_seconds"),
    ("ttft_p95_s", 0.95, "trtllm_time_to_first_token_seconds"),
    ("itl_p50_s", 0.5, "trtllm_time_per_output_token_seconds"),
    ("e2e_p50_s", 0.5, "trtllm_e2e_request_latency_seconds"),
    ("e2e_p95_s", 0.95, "trtllm_e2e_request_latency_seconds"),
)


def _prometheus_to_canonical(text: str) -> dict[str, float]:
    samples = [s for s in parse_samples(text) if s[0].startswith("trtllm_")]
    out: dict[str, float] = {}
    for key, q, name in _PROM_HISTOGRAMS:
        buckets = _histogram(samples, name)
        if buckets is not None:
            val = histogram_quantile(q, buckets)
            if val is not None and math.isfinite(val):
                out[key] = val
    return out


# ------------------------------------------------------------------ JSON iteration statistics
def iteration_stats(text: str) -> list[dict[str, Any]]:
    """The iteration-statistics objects in a `/metrics` body, oldest first; tolerates a clipped tail."""
    body = text.strip()
    if not body:
        return []
    try:
        data = json.loads(body)
    except ValueError:
        data = None
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        return [data]
    if data is not None:
        return []
    # Truncated array (or several documents): decode object by object and keep the complete ones.
    decoder, out, i = json.JSONDecoder(), [], body.find("{")
    while 0 <= i < len(body):
        try:
            obj, end = decoder.raw_decode(body, i)
        except ValueError:
            break
        if isinstance(obj, dict):
            out.append(obj)
        i = body.find("{", end)
    return out


def _sub(stat: dict[str, Any], key: str) -> dict[str, Any]:
    v = stat.get(key)
    return v if isinstance(v, dict) else {}


def _timestamp(stat: dict[str, Any]) -> datetime | None:
    raw = stat.get("timestamp")
    if not isinstance(raw, str):
        return None
    try:
        return datetime.strptime(raw, _TS_FORMAT)
    except ValueError:
        return None


def _stats_to_canonical(stats: list[dict[str, Any]]) -> dict[str, float]:
    """Canonical keys from the iterations of one scrape window (oldest first). Missing keys are omitted."""
    out: dict[str, float] = {}
    if not stats:
        return out
    last = stats[-1]
    running, queued = _num(last.get("numActiveRequests")), _num(last.get("numQueuedRequests"))
    if running is not None:
        out["requests_running"] = running
    if queued is not None:
        out["requests_waiting"] = queued
    kv = _sub(last, "kvCacheStats")
    used, cap = _num(kv.get("usedNumBlocks")), _num(kv.get("maxNumBlocks"))
    if used is not None and cap is not None and cap > 0:
        out["kv_cache_usage_pct"] = min(100.0, max(0.0, 100.0 * used / cap))
    rate = _num(kv.get("cacheHitRate"))
    if rate is None:
        reused, missed = _num(kv.get("reusedBlocks")), _num(kv.get("missedBlocks"))
        if reused is not None and missed is not None and reused + missed > 0:
            rate = reused / (reused + missed)
    if rate is not None:
        out["prefix_cache_hit_pct"] = min(100.0, max(0.0, 100.0 * rate))   # cumulative since engine start

    drafted = accepted = 0.0
    gen_tokens, busy_s = 0.0, 0.0
    step_times: list[float] = []
    for st in stats:
        ib, spec = _sub(st, "inflightBatchingStats"), _sub(st, "specDecodingStats")
        gen = _num(ib.get("numGenRequests")) or 0.0
        ctx = _num(ib.get("numContextRequests")) or 0.0
        latency_ms = _num(st.get("iterLatencyMS"))
        d, acc = _num(spec.get("numDraftTokens")), _num(spec.get("numAcceptedTokens"))
        if d is not None and d > 0:
            drafted += d
            accepted += acc or 0.0
        gen_tokens += gen + ((acc or 0.0) if spec else 0.0)
        if latency_ms is not None and latency_ms > 0:
            busy_s += latency_ms / 1000.0
            # Without speculation every decode-only step emits one token per request, so its latency is the
            # inter-token latency a streaming client sees (up to host overhead).
            if gen > 0 and ctx == 0 and not spec:
                step_times.append(latency_ms / 1000.0)
    if drafted > 0:
        out["spec_decode_accept_pct"] = min(100.0, 100.0 * accepted / drafted)
    if step_times:
        out["itl_p50_s"] = float(median(step_times))
    # Decode tokens per second over the window: tokens counted in the window divided by the wall time it spans
    # (first iteration start to last iteration end), or by busy time when the timestamps cannot be read.
    first_t, last_t = _timestamp(stats[0]), _timestamp(stats[-1])
    last_latency = _num(stats[-1].get("iterLatencyMS")) or 0.0
    span = busy_s
    if first_t is not None and last_t is not None and len(stats) > 1:
        wall = (last_t - first_t).total_seconds() + last_latency / 1000.0
        if wall > 0:
            span = wall
    if gen_tokens > 0 and span > 0:
        out["generation_tps"] = gen_tokens / span
    return out


def parse_metrics_text(text: str) -> dict[str, float]:
    """Canonical metric keys from a `/metrics` JSON body or a `/prometheus/metrics` text body."""
    head = text.lstrip()[:1]
    if head in ("[", "{"):
        return _stats_to_canonical(iteration_stats(text))
    return _prometheus_to_canonical(text)


# ------------------------------------------------------------------ log lines
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_WEIGHTS = re.compile(r"Use ([\d.]+) GB for model weights")           # label says GB, the value is GiB
_MODEL_INIT = re.compile(r"Model init total -- ([\d.]+)s")
_LOAD_BAR = re.compile(r"Loading (?:safetensors|bin) weights in parallel:\s+(\d+)%")
_PREFETCH = re.compile(r"Prefetching ([\d.]+)GB checkpoint files")
_KV_AVAILABLE = re.compile(r"available KV cache memory when calculating max tokens: ([\d.]+) GiB")
_KV_ESTIMATE = re.compile(r"Estimated max memory in KV cache : ([\d.]+) GiB")
_KV_ALLOC = re.compile(r"Allocated ([\d.]+) GiB for max tokens in paged KV cache \((\d+)\)")
_GRAPH_COUNT = re.compile(r"Creating CUDA graph instances for (\d+) batch sizes")
_GRAPH_RUN = re.compile(r"Run generation-only CUDA graph warmup for batch size=(\d+)")
_PIECEWISE = re.compile(r"Run piecewise CUDA graph warmup for num tokens=(\d+)")
_WARMUP = re.compile(r"Run warmup with (\d+) tokens, include (\d+) generation tokens")
_SEQ_INFERRED = re.compile(r"max_seq_len is not specified, using inferred value (\d+)")
_SEQ_CAPPED = re.compile(r"Specified max_seq_len=(\d+) is larger than what the model can support \((\d+)\)")
_SEQ_ADJUSTED = re.compile(r"Adjusted max_seq_len to (\d+)")
_OOM_FREE = re.compile(r"Free GPU memory before component creation attempt \(GiB\): ([\d.]+)")
_OOM_TOTAL = re.compile(r"Total GPU memory \(GiB\): ([\d.]+)")
_ARCH = re.compile(r"Unknown architecture for AutoModelForCausalLM: (\S+)")
_ITER_LOG = re.compile(
    r"iter = (\d+), global_rank = \d+, rank = (\d+), currank_total_requests = (\d+)/(\d+), "
    r"host_step_time = ([\d.]+)ms, prev_device_step_time = ([\d.]+ms|N/A), timestamp = [\d\- :]+, "
    r"num_scheduled_requests: (\d+), states = (\{.*\})")


def _tail(line: str) -> str:
    return line.strip()[-300:]


def parse_log_line(line: str) -> dict[str, Any] | None:
    """Map one engine log line to structured progress, or None when it carries none.

    Keys: `phase` in {loading_weights, warming_up, capturing_graphs, compiling, ready}, `pct`, `weights_gib`,
    `load_seconds`, `kv_cache_gib`, `kv_cache_tokens`, `max_model_len`, `graphs_total`, `batch_size`, plus
    `warning` (the engine changed a setting on its own), `error` (a failure the UI can explain) and `stats`
    (the per-step line printed with `print_iter_log`). A tqdm line may hold several redraws; the last wins."""
    line = _ANSI.sub("", line)
    if "Application startup complete" in line:
        return {"phase": "ready", "pct": 100.0}
    m = _KV_ALLOC.search(line)
    if m:
        return {"kv_cache_gib": float(m.group(1)), "kv_cache_tokens": int(m.group(2))}
    m = _KV_AVAILABLE.search(line) or _KV_ESTIMATE.search(line)
    if m:
        return {"kv_cache_gib": float(m.group(1))}
    m = _WEIGHTS.search(line)
    if m:
        return {"phase": "loading_weights", "weights_gib": float(m.group(1))}
    m = _MODEL_INIT.search(line)
    if m:
        return {"phase": "loading_weights", "pct": 100.0, "load_seconds": float(m.group(1))}
    bars = _LOAD_BAR.findall(line)
    if bars:
        return {"phase": "loading_weights", "pct": float(bars[-1])}
    if _PREFETCH.search(line) or ("Downloading model" in line and "from HuggingFace" in line):
        return {"phase": "loading_weights", "pct": 0.0}
    m = _GRAPH_COUNT.search(line)
    if m:
        return {"phase": "capturing_graphs", "graphs_total": int(m.group(1))}
    m = _GRAPH_RUN.search(line)
    if m:
        return {"phase": "capturing_graphs", "batch_size": int(m.group(1))}
    m = _PIECEWISE.search(line)
    if m or "Running piecewise CUDA graph warmup" in line:
        return {"phase": "capturing_graphs"}
    if "Running torch.compile warmup" in line:
        return {"phase": "compiling"}
    if "Running autotuner warmup" in line or "[Autotuner] Autotuning process starts" in line:
        return {"phase": "warming_up", "step": "autotuner"}
    m = _WARMUP.search(line)
    if m:
        return {"phase": "warming_up", "step": "warmup", "tokens": int(m.group(1))}
    m = _SEQ_INFERRED.search(line)
    if m:
        return {"max_model_len": int(m.group(1))}
    m = _SEQ_CAPPED.search(line)
    if m:
        return {"max_model_len": int(m.group(2)), "warning": "max_seq_len_capped", "requested": int(m.group(1))}
    m = _SEQ_ADJUSTED.search(line)
    if m:
        return {"max_model_len": int(m.group(1)), "warning": "max_seq_len_reduced_by_kv_cache"}
    m = _ITER_LOG.search(line)
    if m:
        st: dict[str, Any] = {"iter": int(m.group(1)), "rank": int(m.group(2)), "host_step_ms": float(m.group(5)),
                              "scheduled": int(m.group(7))}
        states = re.findall(r"'(num_ctx_requests|num_ctx_tokens|num_generation_tokens)': (\d+)", m.group(8))
        for key, val in states:
            st[key] = int(val)
        return {"stats": st}
    return _error(line)


def _error(line: str) -> dict[str, Any] | None:
    # The engine explains an out-of-memory failure over several lines; each carries the same error code, the
    # detail lines add the numbers.
    for pat, key in ((_OOM_FREE, "free_gib"), (_OOM_TOTAL, "total_gib")):
        m = pat.search(line)
        if m:
            return {"error": "cuda_oom", key: float(m.group(1)), "message": _tail(line)}
    if "insufficient GPU memory" in line or "OutOfMemoryError" in line or "CUDA out of memory" in line \
            or "out of memory" in line:
        return {"error": "cuda_oom", "message": _tail(line)}
    m = _ARCH.search(line)
    if m:
        return {"error": "unsupported_architecture", "architecture": m.group(1), "message": _tail(line)}
    if "does not support SM120" in line:
        return {"error": "kernel_unsupported_on_gpu", "message": _tail(line)}
    if "No GPU devices found on the node" in line or "libcuda.so.1: cannot open shared object file" in line:
        return {"error": "no_gpu", "message": _tail(line)}
    if "Failed to bind socket to" in line:
        return {"error": "port_in_use", "message": _tail(line)}
    if "Failed, NCCL error" in line or ("unhandled cuda error" in line and "NCCL" in line):
        return {"error": "nccl_error", "message": _tail(line)}
    if "Impossible to fit in any sequence in kvCache" in line or "Impossibe to fit in any sequence" in line:
        return {"error": "kv_cache_too_small", "message": _tail(line)}
    if "Cannot find an appropriate cached snapshot folder" in line:
        return {"error": "model_not_in_cache", "message": _tail(line)}
    if "No weight files found in" in line:
        return {"error": "no_weights", "message": _tail(line)}
    return None
