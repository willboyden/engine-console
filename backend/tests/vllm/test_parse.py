from __future__ import annotations

import math

import pytest

from engine_console.adapters.vllm import VllmAdapter
from engine_console.adapters.vllm_parse import histogram_quantile, parse_samples

from .conftest import read  # type: ignore[import-not-found]


def test_prometheus_text_parsing_edge_cases() -> None:
    s = parse_samples('# HELP x\nvllm:a{l="q\\"x"} 1.5\nbad line\nvllm:b 2 1712345\nvllm:c{} NaN\n')
    assert s[0] == ("vllm:a", {"l": 'q"x'}, 1.5)
    assert s[1] == ("vllm:b", {}, 2.0)
    assert math.isnan(s[2][2])


def test_histogram_quantile_exact() -> None:
    b = [(0.1, 0.0), (0.2, 50.0), (0.5, 100.0), (math.inf, 100.0)]
    assert histogram_quantile(0.5, b) == pytest.approx(0.2)
    assert histogram_quantile(0.75, b) == pytest.approx(0.35)
    assert histogram_quantile(0.5, [(1.0, 0.0), (math.inf, 0.0)]) is None
    assert histogram_quantile(0.99, [(1.0, 1.0), (math.inf, 10.0)]) == 1.0   # +Inf bucket -> last finite


def test_real_v0271_scrape(adapter: VllmAdapter) -> None:
    m = adapter.parse_metrics(read("metrics_v0.27.1.prom"))
    assert m["requests_running"] == 0 and m["requests_waiting"] == 0
    assert m["kv_cache_usage_pct"] == 0
    assert m["prefix_cache_hit_pct"] == 0
    assert m["prompt_tokens_total"] == 1106 and m["generation_tokens_total"] == 7723
    assert m["preemptions_total"] == 0
    assert m["spec_decode_accept_pct"] == pytest.approx(100 * 4512 / 6402)
    assert 0.02 < m["ttft_p50_s"] <= m["ttft_p95_s"] < 1.0
    assert 0 < m["itl_p50_s"] < 0.1
    assert 0 < m["e2e_p50_s"] <= m["e2e_p95_s"]
    assert "prompt_tps" not in m                    # first sample: no rate yet
    assert set(m) <= {"requests_running", "requests_waiting", "kv_cache_usage_pct", "prefix_cache_hit_pct",
                      "prompt_tokens_total", "generation_tokens_total", "ttft_p50_s", "ttft_p95_s",
                      "itl_p50_s", "e2e_p50_s", "e2e_p95_s", "prompt_tps", "generation_tps",
                      "preemptions_total", "spec_decode_accept_pct"}


def test_rates_from_two_scrapes() -> None:
    t = [100.0]
    a = VllmAdapter(clock=lambda: t[0])
    tpl = ('vllm:prompt_tokens_total{{model_name="m"}} {p}\nvllm:generation_tokens_total{{model_name="m"}} {g}\n')
    a.parse_metrics(tpl.format(p=1000, g=5000))
    t[0] = 102.0
    m = a.parse_metrics(tpl.format(p=1400, g=5600))
    assert m["prompt_tps"] == pytest.approx(200) and m["generation_tps"] == pytest.approx(300)
    t[0] = 104.0
    m = a.parse_metrics(tpl.format(p=10, g=20))       # counter reset (restart): no negative rate
    assert "prompt_tps" not in m and "generation_tps" not in m


def test_legacy_names_and_multi_engine(adapter: VllmAdapter) -> None:
    text = "\n".join([
        'vllm:gpu_cache_usage_perc{engine="0"} 0.25', 'vllm:gpu_cache_usage_perc{engine="1"} 0.75',
        'vllm:num_requests_running{engine="0"} 2', 'vllm:num_requests_running{engine="1"} 3',
        'vllm:gpu_prefix_cache_hits_total 30', 'vllm:gpu_prefix_cache_queries_total 120',
        'vllm:time_per_output_token_seconds_bucket{le="0.05"} 10', 'vllm:time_per_output_token_seconds_bucket{le="+Inf"} 10',
        'vllm:spec_decode_draft_acceptance_rate 0.6',
        'vllm:num_preemptions_total 4', 'vllm:prompt_tokens_total 5'])
    m = adapter.parse_metrics(text)
    assert m["kv_cache_usage_pct"] == 50.0 and m["requests_running"] == 5
    assert m["prefix_cache_hit_pct"] == 25.0
    assert m["itl_p50_s"] == pytest.approx(0.025)
    assert m["spec_decode_accept_pct"] == pytest.approx(60.0)
    assert m["preemptions_total"] == 4


def test_missing_metrics_omitted(adapter: VllmAdapter) -> None:
    assert adapter.parse_metrics("") == {}
    assert adapter.parse_metrics("process_cpu_seconds_total 3\n") == {}
    m = adapter.parse_metrics('vllm:num_requests_waiting{model_name="z"} 7\n')
    assert m == {"requests_waiting": 7.0}


def test_startup_log_real_lines(adapter: VllmAdapter) -> None:
    p = adapter.parse_startup_log
    lines = read("startup_v0.27.1.log").splitlines()
    parsed = [p(x) for x in lines]
    assert {"phase": "loading_weights", "pct": 0.0} in parsed
    assert {"phase": "loading_weights", "pct": 50.0} in parsed
    assert {"phase": "loading_weights", "pct": 100.0, "weights_gib": 22.13, "load_seconds": 3.962537} in parsed
    assert {"kv_cache_tokens": 534714} in parsed
    assert {"max_concurrency": 2.04, "max_model_len": 262144} in parsed
    assert {"phase": "ready", "pct": 100.0} in parsed
    assert any(x and x.get("weights_gib") == 22.13 for x in parsed)
    assert any(x and x.get("phase") == "compiling" for x in parsed)
    assert any(x and x.get("graph_gib") == 0.59 for x in parsed)
    assert any(x and x.get("phase") == "warming_up" and x["init_seconds"] == 15.73 for x in parsed)


def test_tqdm_takes_last_redraw(adapter: VllmAdapter) -> None:
    line = ("(EngineCore pid=421) Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):   0%|   | 0/43 "
            "[00:00<?, ?it/s]Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  47%|▏ | 20/43 [00:00<00:04]")
    assert adapter.parse_startup_log(line) == {"phase": "capturing_graphs", "pct": 47.0}
    real = adapter.parse_startup_log(read("tqdm_line.log"))   # truncated real v0.27.1 line
    assert real is not None and real["phase"] == "capturing_graphs" and 0 < real["pct"] < 100
    assert adapter.parse_startup_log("Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:01<00:00]") \
        == {"phase": "loading_weights", "pct": 100.0}
    assert adapter.parse_startup_log("INFO Graph capturing finished in 3 secs, took 0.59 GiB")["graph_gib"] == 0.59
    assert adapter.parse_startup_log("[gpu_worker.py] Available KV cache memory: 60.11 GiB") == {"kv_cache_gib": 60.11}


def test_stats_line_and_errors(adapter: VllmAdapter) -> None:
    s = adapter.parse_startup_log(
        "(APIServer pid=1) INFO 09-24 02:41:33 [loggers.py:310] Engine 000: Avg prompt throughput: 5.4 tokens/s, "
        "Avg generation throughput: 110.6 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 2.8%, "
        "Prefix cache hit rate: 0.0%")
    assert s == {"stats": {"prompt_tps": 5.4, "generation_tps": 110.6, "running": 1, "waiting": 0,
                           "kv_cache_pct": 2.8, "prefix_hit_pct": 0.0}}
    assert adapter.parse_startup_log("torch.OutOfMemoryError: CUDA out of memory. Tried")["error"] == "cuda_oom"
    assert adapter.parse_startup_log("RuntimeError: No CUDA runtime is found")["error"] == "no_gpu"
    assert adapter.parse_startup_log(
        "ValueError: ... (9.6 GiB KV cache is needed, which is larger than the available KV cache memory (2.45 GiB)."
    )["error"] == "kv_cache_too_small"
    assert adapter.parse_startup_log("INFO random unrelated line") is None
    assert adapter.parse_startup_log("") is None
