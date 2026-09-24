"""Pure parsers for SGLang output: Prometheus text -> canonical metrics, and startup log lines.

Metric names come from `sglang/srt/observability/metrics_collector.py` in the pinned image
(lmsysorg/sglang:v0.5.14-cu130). Names are normalised (prefix `sglang:`/`sglang_`, `_total` suffix
stripped) so a rename between those spellings is tolerated; unknown metrics are ignored.
"""
from __future__ import annotations

import math
import re
from typing import Any

_SAMPLE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+(\S+)(?:\s+-?\d+)?\s*$")
_LE = re.compile(r'(?:^|,)\s*le="([^"]*)"')


def _stem(name: str) -> str:
    n = re.sub(r"^sglang[:_]", "", name)
    return n[: -len("_total")] if n.endswith("_total") else n


def _num(s: str) -> float | None:
    try:
        v = float(s)
    except ValueError:
        return None
    return None if math.isnan(v) else v


def histogram_quantile(q: float, buckets: list[tuple[float, float]]) -> float | None:
    """Prometheus-style quantile from cumulative (le, count) pairs, linear interpolation in-bucket."""
    b = sorted(buckets)
    if not b or b[-1][1] <= 0:
        return None
    target = q * b[-1][1]
    prev_le, prev_c = 0.0, 0.0
    for le, c in b:
        if c >= target:
            if math.isinf(le):
                return prev_le if prev_le > 0 else None
            if c == prev_c:
                return le
            return prev_le + (le - prev_le) * (target - prev_c) / (c - prev_c)
        prev_le, prev_c = le, c
    return None


def parse_prometheus(text: str) -> tuple[dict[str, list[float]], dict[str, dict[float, float]]]:
    """-> (scalar samples by stem, histogram cumulative buckets by stem summed over label sets)."""
    scalars: dict[str, list[float]] = {}
    hists: dict[str, dict[float, float]] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _SAMPLE.match(line)
        if not m:
            continue
        name, labels, val = m.group(1), m.group(2) or "", _num(m.group(3))
        if val is None:
            continue
        if name.endswith("_bucket"):
            le = _LE.search(labels)
            if le:
                le_v = math.inf if le.group(1) in ("+Inf", "Inf") else _num(le.group(1))
                if le_v is not None:
                    d = hists.setdefault(_stem(name[: -len("_bucket")]), {})
                    d[le_v] = d.get(le_v, 0.0) + val
            continue
        if name.endswith(("_sum", "_count", "_created")):
            continue
        scalars.setdefault(_stem(name), []).append(val)
    return scalars, hists


def _first(scalars: dict[str, list[float]], names: tuple[str, ...]) -> list[float] | None:
    for n in names:
        if n in scalars:
            return scalars[n]
    return None


def parse_metrics(text: str) -> dict[str, float]:
    sc, hs = parse_prometheus(text)
    out: dict[str, float] = {}

    def total(key: str, names: tuple[str, ...]) -> None:
        v = _first(sc, names)
        if v:
            out[key] = sum(v)

    def mean_pct(key: str, names: tuple[str, ...]) -> None:
        v = _first(sc, names)
        if v:
            m = sum(v) / len(v)
            out[key] = round(m * 100.0 if m <= 1.0 else m, 4)  # sglang gauges are 0-1 fractions

    total("requests_running", ("num_running_reqs",))
    total("requests_waiting", ("num_queue_reqs",))
    mean_pct("kv_cache_usage_pct", ("token_usage", "full_token_usage"))
    if "kv_cache_usage_pct" not in out:
        used, cap = _first(sc, ("kv_used_tokens",)), _first(sc, ("max_total_num_tokens",))
        if used and cap and sum(cap) > 0:
            out["kv_cache_usage_pct"] = round(sum(used) / sum(cap) * 100.0, 4)
    mean_pct("prefix_cache_hit_pct", ("cache_hit_rate",))
    total("prompt_tokens_total", ("prompt_tokens",))
    total("generation_tokens_total", ("generation_tokens",))
    total("generation_tps", ("gen_throughput",))
    total("preemptions_total", ("num_retracted_requests", "num_retracted_reqs_total"))
    mean_pct("spec_decode_accept_pct", ("spec_accept_rate",))
    v = _first(sc, ("spec_accept_length",))
    if v:  # extra (non-canonical) key: mean accepted tokens per verify step
        out["spec_accept_length"] = sum(v) / len(v)

    for key, hist, q in (
        ("ttft_p50_s", "time_to_first_token_seconds", 0.5), ("ttft_p95_s", "time_to_first_token_seconds", 0.95),
        ("itl_p50_s", "inter_token_latency_seconds", 0.5), ("e2e_p50_s", "e2e_request_latency_seconds", 0.5),
        ("e2e_p95_s", "e2e_request_latency_seconds", 0.95),
    ):
        if hist in hs:
            r = histogram_quantile(q, list(hs[hist].items()))
            if r is not None:
                out[key] = r
    return out


# ---------------- startup log ----------------
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_TQDM = re.compile(r"(\d+)%\|[^|]*\|\s*(\d+)/(\d+)")
_TQDM_TXT = re.compile(r"(\d+)%\s+Completed\s*\|\s*(\d+)/(\d+)")


def parse_startup_log(line: str) -> dict[str, Any] | None:
    s = _ANSI.sub("", line)
    if "\r" in s:  # tqdm redraws: keep the last frame
        s = s.split("\r")[-1] or s
    s = s.strip()
    if not s:
        return None
    if "The server is fired up and ready to roll" in s:
        return {"phase": "ready", "pct": 100.0}
    if re.search(r"CUDA out of memory|OutOfMemoryError", s):
        return {"phase": "failed", "error": "oom"}
    m = re.search(r"Capturing batches \(bs=(\d+).*?\)", s)
    if m:
        t = _TQDM.search(s)
        d: dict[str, Any] = {"phase": "capturing_graphs", "batch_size": int(m.group(1))}
        if t:
            d["pct"] = float(t.group(1))
        return d
    if "Capture cuda graph begin" in s:
        return {"phase": "capturing_graphs", "pct": 0.0}
    m = re.search(r"Capture cuda graph end.*?mem usage=([\d.]+) GB", s)
    if m:
        return {"phase": "capturing_graphs", "pct": 100.0, "cuda_graph_gib": float(m.group(1))}
    if "Capture cuda graph end" in s:
        return {"phase": "capturing_graphs", "pct": 100.0}
    if re.search(r"shards|checkpoint", s, re.I) and re.search(r"Loading|loading", s):
        t = _TQDM_TXT.search(s) or _TQDM.search(s)
        if t:
            return {"phase": "loading_weights", "pct": float(t.group(1))}
    m = re.search(r"Load weight end\..*?mem usage=([\d.]+) GB", s)
    if m:
        return {"phase": "loading_weights", "pct": 100.0, "weights_gib": float(m.group(1))}
    if "Load weight begin" in s:
        return {"phase": "loading_weights", "pct": 0.0}
    m = re.search(r"KV Cache is allocated\..*?#tokens: (\d+)", s)
    if m:
        return {"phase": "allocating_kv", "kv_cache_tokens": int(m.group(1))}
    m = re.search(r"max_total_num_tokens=(\d+)", s)
    if m:
        d = {"phase": "allocating_kv", "kv_cache_tokens": int(m.group(1))}
        for key, pat in (("max_running_requests", r"max_running_requests=(\d+)"),
                         ("context_len", r"context_len=(\d+)")):
            k = re.search(pat, s)
            if k:
                d[key] = int(k.group(1))
        return d
    if re.search(r"torch\.compile|Compiling|Compile ", s):
        return {"phase": "compiling"}
    return None
