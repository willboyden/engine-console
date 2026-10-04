"""Metrics and start-up log parsing, against fixtures from the pinned image (see fixtures/README.md for which parts
are captured from the engine's own code and which are synthetic)."""
from __future__ import annotations

import json
import math

import pytest

from engine_console.adapters.trtllm import TrtllmAdapter
from engine_console.adapters.trtllm_parse import (
    histogram_quantile,
    iteration_stats,
    parse_log_line,
    parse_metrics_text,
    parse_samples,
)

from .conftest import read, read_json

CANONICAL = {"requests_running", "requests_waiting", "kv_cache_usage_pct", "prefix_cache_hit_pct",
             "prompt_tokens_total", "generation_tokens_total", "ttft_p50_s", "ttft_p95_s", "itl_p50_s",
             "e2e_p50_s", "e2e_p95_s", "prompt_tps", "generation_tps", "preemptions_total",
             "spec_decode_accept_pct"}


def stats_body(**kw: object) -> str:
    """One object of the captured iteration-stats shape with a few fields replaced."""
    base = read_json("iter_stats_1.2.1.json")["plain"]
    return json.dumps(base | kw)


# --------------------------------------------------------------- JSON /metrics
def test_metrics_endpoint_json_from_the_engine_serializer(adapter: TrtllmAdapter) -> None:
    doc = read_json("iter_stats_1.2.1.json")
    m = adapter.parse_metrics(json.dumps([doc["plain"]]))
    assert m["requests_running"] == 3 and m["requests_waiting"] == 2
    assert m["kv_cache_usage_pct"] == pytest.approx(25.0)               # 250 of 1000 blocks
    assert m["prefix_cache_hit_pct"] == pytest.approx(25.0)             # cacheHitRate 0.25 is a fraction
    assert m["generation_tps"] == pytest.approx(2 / 0.0125)             # 2 generation requests in a 12.5 ms step
    assert "itl_p50_s" not in m                                         # this step also has a context request
    assert set(m) <= CANONICAL
    assert "prompt_tps" not in m and "prompt_tokens_total" not in m     # the PyTorch backend does not report them


def test_speculation_and_decode_only_iterations(adapter: TrtllmAdapter) -> None:
    doc = read_json("iter_stats_1.2.1.json")
    spec = doc["with_spec_and_requests"]
    m = adapter.parse_metrics(json.dumps([spec]))
    assert m["spec_decode_accept_pct"] == pytest.approx(100 * 5 / 8)
    decode = json.loads(stats_body())
    decode["inflightBatchingStats"]["numContextRequests"] = 0
    decode["iterLatencyMS"] = 20.0
    m = adapter.parse_metrics(json.dumps([decode, decode | {"iterLatencyMS": 30.0}, decode | {"iterLatencyMS": 40.0}]))
    assert m["itl_p50_s"] == pytest.approx(0.030)                       # median step time of decode-only steps
    spec_decode = json.loads(json.dumps(decode)) | {"specDecodingStats": spec["specDecodingStats"]}
    assert "itl_p50_s" not in adapter.parse_metrics(json.dumps([spec_decode]))   # several tokens per step: no ITL


def test_window_rate_uses_the_timestamps(adapter: TrtllmAdapter) -> None:
    first = json.loads(stats_body(timestamp="10-04-2026 09:00:00.000000"))
    last = json.loads(stats_body(timestamp="10-04-2026 09:00:01.000000", iterLatencyMS=500.0))
    m = adapter.parse_metrics(json.dumps([first, last]))
    # 4 generation tokens over (1.0 s + 0.5 s) of wall time
    assert m["generation_tps"] == pytest.approx(4 / 1.5)
    broken = json.loads(stats_body(timestamp="not a time"))
    assert adapter.parse_metrics(json.dumps([broken, broken]))["generation_tps"] > 0   # falls back to busy time


def test_cache_hit_rate_falls_back_to_block_counters(adapter: TrtllmAdapter) -> None:
    body = json.loads(stats_body())
    del body["kvCacheStats"]["cacheHitRate"]
    body["kvCacheStats"].update(reusedBlocks=4, missedBlocks=3121)
    assert adapter.parse_metrics(json.dumps([body]))["prefix_cache_hit_pct"] == pytest.approx(100 * 4 / 3125)


def test_empty_and_odd_bodies_never_raise(adapter: TrtllmAdapter) -> None:
    for text in ("", "   ", "[]", "{}", "null", "42", '"x"', "not json", "[1, 2]", "# HELP x\n", "process_cpu_seconds_total 3\n"):
        assert adapter.parse_metrics(text) == {}, text
    assert adapter.parse_metrics('[{"numActiveRequests": "many", "kvCacheStats": {"maxNumBlocks": 0}}]') == {}
    nan = '[{"numActiveRequests": NaN, "numQueuedRequests": 1}]'
    assert adapter.parse_metrics(nan) == {"requests_waiting": 1.0}


def test_clipped_body_keeps_the_complete_objects(adapter: TrtllmAdapter) -> None:
    one = stats_body()
    clipped = "[" + one + ", " + one + ", " + one[: len(one) // 2]      # the node caps a scrape at 512 KiB
    assert len(iteration_stats(clipped)) == 2
    assert adapter.parse_metrics(clipped)["requests_running"] == 3
    assert iteration_stats(f"{one}{one}") == [json.loads(one)] * 2     # concatenated documents
    assert iteration_stats(one) == [json.loads(one)]                    # a bare object


def test_parser_keeps_no_state_between_scrapes(adapter: TrtllmAdapter) -> None:
    body = json.dumps([json.loads(stats_body())])
    assert adapter.parse_metrics(body) == adapter.parse_metrics(body) == TrtllmAdapter().parse_metrics(body)


# --------------------------------------------------------------- Prometheus text (return_perf_metrics)
def test_prometheus_text_from_the_engine_collector(adapter: TrtllmAdapter) -> None:
    text = read("collector_1.2.1.prom")
    m = adapter.parse_metrics(text)
    # four synthetic requests were observed: e2e 0.6/1.2/3.4/9.8 s, ttft 0.04/0.08/0.21/0.9 s
    assert 0.5 < m["e2e_p50_s"] <= 1.5 and 5.0 <= m["e2e_p95_s"] <= 10.0
    assert 0.04 < m["ttft_p50_s"] < 0.1 and 0.5 < m["ttft_p95_s"] <= 1.0
    assert 0.01 < m["itl_p50_s"] < 0.05
    assert m["e2e_p50_s"] <= m["e2e_p95_s"] and m["ttft_p50_s"] <= m["ttft_p95_s"]
    assert set(m) <= CANONICAL and "requests_running" not in m          # that endpoint has no queue or cache gauges


def test_histogram_helpers() -> None:
    s = parse_samples('# HELP x\ntrtllm_a{l="q\\"x"} 1.5\nbad line\ntrtllm_b 2 1712345\ntrtllm_c{} NaN\n')
    assert s[0] == ("trtllm_a", {"l": 'q"x'}, 1.5) and s[1] == ("trtllm_b", {}, 2.0) and math.isnan(s[2][2])
    b = [(0.1, 0.0), (0.2, 50.0), (0.5, 100.0), (math.inf, 100.0)]
    assert histogram_quantile(0.5, b) == pytest.approx(0.2) and histogram_quantile(0.75, b) == pytest.approx(0.35)
    assert histogram_quantile(0.5, [(1.0, 0.0), (math.inf, 0.0)]) is None
    assert histogram_quantile(0.99, [(1.0, 1.0), (math.inf, 10.0)]) == 1.0


# --------------------------------------------------------------- start-up log
def parsed_fixture(adapter: TrtllmAdapter) -> list[dict[str, object] | None]:
    return [adapter.parse_startup_log(line) for line in read("startup_1.2.1.log").replace("\r", "\n").splitlines()]


def test_startup_progress_from_fixture_lines(adapter: TrtllmAdapter) -> None:
    parsed = parsed_fixture(adapter)
    assert {"phase": "loading_weights", "pct": 0.0} in parsed                                   # download / prefetch
    assert {"phase": "loading_weights", "weights_gib": 14.96} in parsed
    assert {"phase": "loading_weights", "pct": 100.0, "load_seconds": 21.34} in parsed
    assert {"phase": "loading_weights", "pct": 25.0} in parsed and {"phase": "loading_weights", "pct": 100.0} in parsed   # tqdm
    assert {"max_model_len": 131072} in parsed
    assert {"max_model_len": 131072, "warning": "max_seq_len_capped", "requested": 262144} in parsed
    assert {"max_model_len": 98304, "warning": "max_seq_len_reduced_by_kv_cache"} in parsed
    assert {"kv_cache_gib": 64.8} in parsed
    assert {"phase": "warming_up", "step": "autotuner"} in parsed
    assert {"phase": "capturing_graphs", "graphs_total": 34} in parsed
    assert {"phase": "capturing_graphs", "batch_size": 128} in parsed and {"phase": "capturing_graphs", "batch_size": 1} in parsed
    assert {"phase": "compiling"} in parsed
    assert {"phase": "warming_up", "step": "warmup", "tokens": 8192} in parsed
    assert {"phase": "ready", "pct": 100.0} in parsed                                           # uvicorn, real lines


def test_ready_is_only_the_server_startup_line(adapter: TrtllmAdapter) -> None:
    ready = [line for line in read("startup_1.2.1.log").splitlines() if (adapter.parse_startup_log(line) or {}).get("phase") == "ready"]
    assert ready == ["INFO:     Application startup complete."]
    assert adapter.parse_startup_log("INFO:     Waiting for application startup.") is None
    assert adapter.parse_startup_log("INFO:     Uvicorn running on http://127.0.0.1:18999 (Press CTRL+C to quit)") is None


def test_per_step_log_line(adapter: TrtllmAdapter) -> None:
    line = [x for x in read("startup_1.2.1.log").splitlines() if "iter = 154" in x][0]
    assert adapter.parse_startup_log(line) == {"stats": {
        "iter": 154, "rank": 0, "host_step_ms": 9.8, "scheduled": 3, "num_ctx_requests": 0, "num_ctx_tokens": 0,
        "num_generation_tokens": 3}}


def test_rank_prefix_does_not_matter(adapter: TrtllmAdapter) -> None:
    assert adapter.parse_startup_log("[10/04/2026-18:20:34] [TRT-LLM] [RANK 1] [I] Use 7.48 GB for model weights.") == {
        "phase": "loading_weights", "weights_gib": 7.48}
    assert adapter.parse_startup_log("[TensorRT-LLM][INFO] [MemUsageChange] Allocated 64.80 GiB for max tokens in "
                                     "paged KV cache (528384).") == {"kv_cache_gib": 64.8, "kv_cache_tokens": 528384}


def test_tqdm_redraws_take_the_last_percentage(adapter: TrtllmAdapter) -> None:
    line = ("\rLoading safetensors weights in parallel:   0%|          | 0/4 [00:00<?, ?it/s]"
            "\rLoading safetensors weights in parallel:  75%|███████▌  | 3/4 [00:00<00:00,  6.4it/s]")
    assert adapter.parse_startup_log(line) == {"phase": "loading_weights", "pct": 75.0}
    assert adapter.parse_startup_log("Loading bin weights in parallel: 50%|") == {"phase": "loading_weights", "pct": 50.0}


def test_failures_are_named(adapter: TrtllmAdapter) -> None:
    """Failure texts copied from the image's sources (synthetic lines: no failing start was run)."""
    p = adapter.parse_startup_log
    oom = p("RuntimeError: Executor creation failed due to insufficient GPU memory.")
    assert oom is not None and oom["error"] == "cuda_oom"
    assert p("Total GPU memory (GiB): 94.99") == {"error": "cuda_oom", "total_gib": 94.99, "message": "Total GPU memory (GiB): 94.99"}
    free = p("Free GPU memory before component creation attempt (GiB): 12.31")
    assert free is not None and free["free_gib"] == 12.31
    assert (p("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB") or {})["error"] == "cuda_oom"
    arch = p("ValueError: Unknown architecture for AutoModelForCausalLM: FooForCausalLM")
    assert arch is not None and arch["error"] == "unsupported_architecture" and arch["architecture"] == "FooForCausalLM"
    assert (p("NotImplementedError: TRTLLMGenFusedMoE does not support SM120 and above.") or {})["error"] == "kernel_unsupported_on_gpu"
    assert (p("ValueError: No GPU devices found on the node") or {})["error"] == "no_gpu"
    assert (p("ImportError: libcuda.so.1: cannot open shared object file: No such file or directory") or {})["error"] == "no_gpu"
    assert (p("RuntimeError: Failed to bind socket to 0.0.0.0:8000: [Errno 98] Address already in use") or {})["error"] == "port_in_use"
    assert (p("RuntimeError: Failed, NCCL error /app/x.cpp:12 'unhandled cuda error (run with NCCL_DEBUG=INFO for details)'") or {})["error"] == "nccl_error"
    assert (p("AssertionError: Impossible to fit in any sequence in kvCache") or {})["error"] == "kv_cache_too_small"
    assert (p("Cannot find an appropriate cached snapshot folder for the specified revision on the local disk and "
              "outgoing traffic has been disabled.") or {})["error"] == "model_not_in_cache"
    assert (p("RuntimeError: No weight files found in /x.") or {})["error"] == "no_weights"


def test_ordinary_lines_carry_nothing(adapter: TrtllmAdapter) -> None:
    for line in ("", "[10/04/2026-18:20:34] [TRT-LLM] [I] Validating KV Cache config against kv_cache_dtype=\"auto\"",
                 "[TensorRT-LLM] TensorRT LLM version: 1.2.1", "INFO:     Started server process [1]",
                 "INFO:     Shutting down"):
        assert adapter.parse_startup_log(line) is None, line


def test_metrics_text_entry_point_matches_the_adapter(adapter: TrtllmAdapter) -> None:
    body = json.dumps([json.loads(stats_body())])
    assert parse_metrics_text(body) == adapter.parse_metrics(body)
    assert parse_log_line("INFO:     Application startup complete.") == adapter.parse_startup_log("INFO:     Application startup complete.")
