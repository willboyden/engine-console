"""Pure parsers for vLLM Prometheus text and container log lines (no I/O, no engine imports).

Metric names were confirmed against a live v0.27.1 `/metrics` scrape and the metric registry in the
v0.23.0 image. prometheus_client appends `_total` to counters, so `vllm:prompt_tokens` in the source
is `vllm:prompt_tokens_total` on the wire; lookups strip a trailing `_total` so either spelling works.
Older releases (pre-0.10, still seen in forks) are tolerated through the alias tables below.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

Labels = tuple[tuple[str, str], ...]
_LINE = re.compile(r'^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{(.*)\})?\s+(\S+)(?:\s+\S+)?\s*$')
_LABEL = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="((?:[^"\\]|\\.)*)"')


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


class Scrape:
    """Indexed view of one scrape, tolerant to `_total` and legacy names."""

    def __init__(self, text: str) -> None:
        self.by_name: dict[str, list[tuple[dict[str, str], float]]] = defaultdict(list)
        for name, labels, value in parse_samples(text):
            if name.startswith("vllm:"):
                self.by_name[name].append((labels, value))

    def _rows(self, names: Iterable[str], suffix: str = "") -> list[tuple[dict[str, str], float]]:
        for base in names:
            for cand in (base + suffix, base + "_total" + suffix):
                if cand in self.by_name:
                    return self.by_name[cand]
        return []

    def total(self, *names: str) -> float | None:
        """Sum over all label sets (engines / DP ranks) of the first metric name that exists."""
        rows = self._rows(names)
        return sum(v for _, v in rows) if rows else None

    def mean(self, *names: str) -> float | None:
        rows = self._rows(names)
        return sum(v for _, v in rows) / len(rows) if rows else None

    def histogram(self, *names: str) -> tuple[list[tuple[float, float]], float] | None:
        """Merged cumulative buckets [(le, count)] sorted by le, and total count."""
        rows = self._rows(names, "_bucket")
        if not rows:
            return None
        merged: dict[float, float] = defaultdict(float)
        for labels, v in rows:
            le = labels.get("le")
            if le is None:
                continue
            merged[math.inf if le in ("+Inf", "Inf") else float(le)] += v
        buckets = sorted(merged.items())
        return (buckets, buckets[-1][1]) if buckets else None

    def model_names(self) -> tuple[str, ...]:
        names = {lab["model_name"] for rows in self.by_name.values() for lab, _ in rows if "model_name" in lab}
        return tuple(sorted(names))


def histogram_quantile(q: float, buckets: list[tuple[float, float]]) -> float | None:
    """Prometheus-style quantile with linear interpolation inside the bucket.

    A quantile landing in the +Inf bucket returns the highest finite bound (a lower bound on the truth).
    """
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


def scrape_to_canonical(s: Scrape) -> dict[str, float]:
    """Instantaneous canonical keys. Rates (`*_tps`) are added by the adapter, which keeps state."""
    out: dict[str, float] = {}

    def put(key: str, val: float | None) -> None:
        if val is not None and not math.isnan(val):
            out[key] = val

    put("requests_running", s.total("vllm:num_requests_running"))
    put("requests_waiting", s.total("vllm:num_requests_waiting"))
    # 0-1 fraction on every version we know (gpu_cache_usage_perc is the pre-rename spelling).
    kv = s.mean("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")
    put("kv_cache_usage_pct", None if kv is None else kv * 100.0)

    hits = s.total("vllm:prefix_cache_hits", "vllm:gpu_prefix_cache_hits")
    queries = s.total("vllm:prefix_cache_queries", "vllm:gpu_prefix_cache_queries")
    if hits is not None and queries:
        put("prefix_cache_hit_pct", 100.0 * hits / queries)
    elif queries is None:
        legacy = s.mean("vllm:gpu_prefix_cache_hit_rate", "vllm:cpu_prefix_cache_hit_rate")
        put("prefix_cache_hit_pct", None if legacy is None else legacy * 100.0)

    put("prompt_tokens_total", s.total("vllm:prompt_tokens"))
    put("generation_tokens_total", s.total("vllm:generation_tokens"))
    put("preemptions_total", s.total("vllm:num_preemptions"))

    for key, q, names in (
        ("ttft_p50_s", 0.5, ("vllm:time_to_first_token_seconds",)),
        ("ttft_p95_s", 0.95, ("vllm:time_to_first_token_seconds",)),
        # request_time_per_output_token_seconds is a different (per-request mean) series: not a fallback.
        ("itl_p50_s", 0.5, ("vllm:inter_token_latency_seconds", "vllm:time_per_output_token_seconds")),
        ("e2e_p50_s", 0.5, ("vllm:e2e_request_latency_seconds",)),
        ("e2e_p95_s", 0.95, ("vllm:e2e_request_latency_seconds",)),
    ):
        h = s.histogram(*names)
        if h is not None:
            put(key, histogram_quantile(q, h[0]))

    accepted = s.total("vllm:spec_decode_num_accepted_tokens")
    drafted = s.total("vllm:spec_decode_num_draft_tokens")
    if accepted is not None and drafted:
        put("spec_decode_accept_pct", 100.0 * accepted / drafted)
    elif drafted is None:
        legacy_rate = s.mean("vllm:spec_decode_draft_acceptance_rate")
        put("spec_decode_accept_pct", None if legacy_rate is None else legacy_rate * 100.0)
    return out


# ------------------------------------------------------------------ log lines
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_SHARDS = re.compile(r"Loading (?:safetensors checkpoint|pt checkpoint|checkpoint)[^:]*shards:\s+(\d+)%")
_CUDA_GRAPH = re.compile(r"Capturing CUDA graphs[^|]*?:\s*(\d+)%")
_CUDA_GRAPH_N = re.compile(r"(\d+)/(\d+)\s*\[")
_STATS = re.compile(
    r"Avg prompt throughput: ([\d.]+) tokens/s, Avg generation throughput: ([\d.]+) tokens/s, "
    r"Running: (\d+) reqs?, Waiting: (\d+) reqs?, GPU KV cache usage: ([\d.]+)%"
    r"(?:, Prefix cache hit rate: ([\d.]+)%)?")


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def parse_log_line(line: str) -> dict[str, Any] | None:
    """Map one engine log line to structured progress, or None when it carries none.

    Patterns come from real v0.27.1 lab logs (qwen38, nemotron-lightning). Keys: `phase` in
    {loading_weights, compiling, capturing_graphs, warming_up, ready}, `pct`, `kv_cache_tokens`,
    `kv_cache_gib`, `max_concurrency`, `weights_gib`, `graph_gib`, `error`, plus `stats` for the
    periodic throughput line. A single tqdm line may hold many progress redraws; the last wins.
    """
    line = _ANSI.sub("", line)
    if "Application startup complete" in line:
        return {"phase": "ready", "pct": 100.0}
    m = re.search(r"GPU KV cache size: ([\d,]+) tokens", line)
    if m:
        return {"kv_cache_tokens": int(_num(m.group(1)))}
    m = re.search(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x", line)
    if m:
        return {"max_concurrency": float(m.group(2)), "max_model_len": int(_num(m.group(1)))}
    m = re.search(r"Available KV cache memory: ([\d.]+) GiB", line)
    if m:
        return {"kv_cache_gib": float(m.group(1))}
    m = re.search(r"reserved ([\d.]+) GiB memory for KV Cache", line)
    if m:
        return {"kv_cache_gib": float(m.group(1))}
    m = re.search(r"Model loading took ([\d.]+) GiB(?: memory)? and ([\d.]+) seconds", line)
    if m:
        return {"phase": "loading_weights", "pct": 100.0, "weights_gib": float(m.group(1)),
                "load_seconds": float(m.group(2))}
    if "Starting to load model" in line:
        return {"phase": "loading_weights", "pct": 0.0}
    shard = list(_SHARDS.finditer(line))
    if shard:
        return {"phase": "loading_weights", "pct": float(shard[-1].group(1))}
    if "Capturing CUDA graphs" in line:
        pcts = _CUDA_GRAPH.findall(line)
        out: dict[str, Any] = {"phase": "capturing_graphs"}
        if pcts:
            out["pct"] = float(pcts[-1])
        return out
    m = re.search(r"Graph capturing finished in ([\d.]+) secs?, took ([\d.]+) GiB", line)
    if m:
        return {"phase": "capturing_graphs", "pct": 100.0, "graph_gib": float(m.group(2))}
    m = re.search(r"init engine \(profile, create kv cache, warmup model\) took ([\d.]+) s", line)
    if m:
        return {"phase": "warming_up", "init_seconds": float(m.group(1))}
    if ("torch.compile took" in line or "Dynamo bytecode transform" in line
            or "Compiling a graph for" in line or "Directly load AOT compilation" in line):
        return {"phase": "compiling"}
    m = _STATS.search(line)
    if m:
        st: dict[str, Any] = {
            "prompt_tps": float(m.group(1)), "generation_tps": float(m.group(2)),
            "running": int(m.group(3)), "waiting": int(m.group(4)), "kv_cache_pct": float(m.group(5))}
        if m.group(6) is not None:
            st["prefix_hit_pct"] = float(m.group(6))
        return {"stats": st}
    low = line
    if "OutOfMemoryError" in low or "CUDA out of memory" in low:
        return {"error": "cuda_oom", "message": line.strip()[-300:]}
    # Two wordings across versions; the first is the newer one (recalled from vLLM source/lab notes,
    # not captured from a live failure).
    if "is larger than the available KV cache memory" in low \
            or "larger than the maximum number of tokens that can be stored in KV cache" in low:
        return {"error": "kv_cache_too_small", "message": line.strip()[-300:]}
    if "No CUDA runtime is found" in low or "Failed to infer device type" in low:
        return {"error": "no_gpu", "message": line.strip()[-300:]}
    if "Free memory on device" in low and "is less than desired GPU memory utilization" in low:
        return {"error": "insufficient_free_vram", "message": line.strip()[-300:]}
    return None
