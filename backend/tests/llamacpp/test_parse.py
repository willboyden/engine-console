"""parse_metrics() and parse_startup_log().

Captured (real, from the pinned image, no model started): startup_no_model_b10156.log, router_mode_probe_b10156.log,
api_key_probe_b10156.log, cors_probe_b10156.log, version_b10156.txt.
Synthetic (labelled so in the file): metrics_synthetic.prom, startup_synthetic.log, and the inline error lines below.
"""
from __future__ import annotations

import math
import re

import pytest

from engine_console.adapters.llamacpp import LlamaCppAdapter
from engine_console.adapters.llamacpp_parse import parse_log_line, parse_samples, scrape_to_canonical

from .helpers import read

CANONICAL = {"requests_running", "requests_waiting", "kv_cache_usage_pct", "prefix_cache_hit_pct", "prompt_tokens_total",
             "generation_tokens_total", "ttft_p50_s", "ttft_p95_s", "itl_p50_s", "e2e_p50_s", "e2e_p95_s", "prompt_tps",
             "generation_tps", "preemptions_total", "spec_decode_accept_pct"}


# ------------------------------------------------------------------ metrics
def test_synthetic_scrape_maps_to_canonical_keys(adapter: LlamaCppAdapter) -> None:
    text = read("metrics_synthetic.prom")
    assert text.startswith("# SYNTHETIC FIXTURE")                      # the file says what it is
    m = adapter.parse_metrics(text)
    assert m == {"requests_running": 2.0, "requests_waiting": 1.0, "prompt_tokens_total": 1106.0,
                 "generation_tokens_total": 7723.0, "prompt_tps": 442.4, "generation_tps": 126.2}
    assert set(m) <= CANONICAL


def test_metric_names_of_the_fixture_are_the_ones_the_docs_and_image_list() -> None:
    """The names below were read from the image's string table and the README at the same build tag."""
    names = {n for n, _ in parse_samples(read("metrics_synthetic.prom"))}
    assert names == {"prompt_tokens_total", "prompt_seconds_total", "tokens_predicted_total", "tokens_predicted_seconds_total",
                     "n_decode_total", "n_tokens_max", "prompt_tokens_seconds", "predicted_tokens_seconds",
                     "requests_processing", "requests_deferred", "n_busy_slots_per_decode"}


def test_the_pinned_build_exports_no_kv_cache_metric() -> None:
    """Older builds had llamacpp:kv_cache_usage_ratio / kv_cache_tokens; neither is in the pinned fixture, so the
    canonical kv_cache_usage_pct is simply absent (never zero)."""
    text = read("metrics_synthetic.prom")
    assert "kv_cache" not in text
    assert "kv_cache_usage_pct" not in scrape_to_canonical(text)


def test_older_or_newer_builds_that_export_kv_usage_are_read() -> None:
    m = scrape_to_canonical("llamacpp:kv_cache_usage_ratio 0.25\nllamacpp:kv_cache_tokens 1024\n")
    assert m == {"kv_cache_usage_pct": 25.0}


def test_missing_and_foreign_metrics_are_omitted(adapter: LlamaCppAdapter) -> None:
    assert adapter.parse_metrics("") == {}
    assert adapter.parse_metrics("process_cpu_seconds_total 3\nvllm:num_requests_running 2\n") == {}
    assert adapter.parse_metrics("llamacpp:requests_deferred 7\n") == {"requests_waiting": 7.0}
    assert adapter.parse_metrics("llamacpp:requests_processing 0\n") == {"requests_running": 0.0}      # zero is kept, absent is not
    assert adapter.parse_metrics("# HELP llamacpp:requests_processing x\n# TYPE llamacpp:requests_processing gauge\n") == {}


def test_tolerates_total_suffix_variants_and_labels() -> None:
    text = "\n".join(['llamacpp:tokens_predicted 10', 'llamacpp:prompt_tokens{model="a"} 5', 'llamacpp:prompt_tokens{model="b"} 7',
                      'llamacpp:requests_processing{slot="0"} 1', 'llamacpp:requests_processing{slot="1"} 2'])
    m = scrape_to_canonical(text)
    assert m == {"generation_tokens_total": 10.0, "prompt_tokens_total": 12.0, "requests_running": 3.0}


def test_parse_samples_edge_cases() -> None:
    s = parse_samples('# HELP x\nllamacpp:a 1.5\nbad line\nllamacpp:b 2 1712345\nllamacpp:c NaN\nllamacpp:d{l="q\\"x"} 4\nllamacpp:e abc\n')
    assert s == [("a", 1.5), ("b", 2.0), ("d", 4.0)]
    assert not any(math.isnan(v) for _, v in s)


def test_parser_keeps_no_state(adapter: LlamaCppAdapter) -> None:
    """No label tells two instances apart, so nothing may be remembered between scrapes."""
    first = adapter.parse_metrics("llamacpp:tokens_predicted_total 100\n")
    second = adapter.parse_metrics("llamacpp:tokens_predicted_total 50\n")
    assert first == {"generation_tokens_total": 100.0} and second == {"generation_tokens_total": 50.0}
    assert not any(k.endswith("_tps") for k in first | second)
    assert vars(adapter).keys() == {"_catalog", "_by_key"}


# ------------------------------------------------------------------ startup log: captured lines
def test_captured_no_model_failure(adapter: LlamaCppAdapter) -> None:
    lines = read("startup_no_model_b10156.log").splitlines()
    parsed = [adapter.parse_startup_log(line) for line in lines]
    assert len(parsed) == len(lines) == 19
    assert parsed[0] is None                                                    # the LLAMA_ARG_HOST warning
    assert parsed[1] is None                                                    # "verbosity = 3"
    assert all(p is None for p in parsed[2:7])                                  # the CORS warning block
    assert parsed[7] == {"phase": "loading_weights", "pct": 0.0}                # "loading model '...'"
    errors = [p["error"] for p in parsed if p and "error" in p]
    assert errors[0] == "model_file_missing" and errors.count("model_file_missing") == 2
    assert set(errors) == {"model_file_missing", "model_load_failed"} and errors[-1] == "model_load_failed"
    missing = next(p for p in parsed if p and p.get("error") == "model_file_missing")
    assert "does-not-exist.gguf" in missing["message"] and "No such file or directory" in missing["message"]
    last = next(p for p in reversed(parsed) if p)
    assert "exiting due to model loading error" in last["message"]
    assert parsed[-2] is None                                                   # "cleaning up before exit"


def test_captured_router_mode_start(adapter: LlamaCppAdapter) -> None:
    log = read("router_mode_probe_b10156.log").split("== LOG\n", 1)[1]
    parsed = [adapter.parse_startup_log(line) for line in log.splitlines()]
    assert {"listening": "http://127.0.0.1:8080"} in parsed
    assert sum(1 for p in parsed if p) == 1                                     # nothing else in it is progress


def test_real_lines_one_by_one() -> None:
    p = parse_log_line
    assert p("0.00.007.493 I srv    load_model: loading model '/models/x.gguf'") == {"phase": "loading_weights", "pct": 0.0}
    assert p("0.00.008.904 I srv  llama_server: listening on http://127.0.0.1:8080") == {"listening": "http://127.0.0.1:8080"}
    assert p("0.00.006.233 W srv  llama_server: -----------------") is None
    assert p("warn: LLAMA_ARG_HOST environment variable is set, but will be overwritten by command line argument --host") is None
    assert p("") is None and p("   ") is None


# ------------------------------------------------------------------ startup log: synthetic success path
def test_synthetic_success_path(adapter: LlamaCppAdapter) -> None:
    text = read("startup_synthetic.log")
    assert text.startswith("# SYNTHETIC FIXTURE")
    parsed = [adapter.parse_startup_log(line) for line in text.splitlines()]
    got = [p for p in parsed if p]
    assert {"phase": "loading_weights", "pct": 0.0} in got
    assert {"n_slots": 4, "kv_unified": True} in got                                           # auto parallel
    assert {"gpus_found": 1} in got
    assert {"gpu": {"index": 0, "name": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition", "compute_capability": "12.0",
                    "vram_gib": 95.59}} in got
    assert {"fit_projected_mib": 21000, "fit_free_mib": 90000} in got
    assert {"fit_ctx_reduced": {"from": 262144, "to": 65536}} in got
    assert {"phase": "loading_weights", "pct": 0.0, "tensors": 291} in got
    assert {"n_ctx_train": 262144} in got
    assert {"phase": "loading_weights", "gpu_layers": 49, "total_layers": 49} in got
    assert {"buffer": "CUDA0", "model_buffer_gib": 17.0} in got
    assert {"buffer": "CPU_Mapped", "model_buffer_gib": 0.308} in got
    assert {"phase": "loading_weights", "pct": 98.0} in got                                    # the row of progress dots
    assert {"n_ctx": 65536} in got
    assert {"kv_buffer": {"device": "CUDA0", "gib": 3.0}} in got
    assert {"kv_cache_gib": 3.0, "kv_cache_tokens": 65536} in got
    assert {"phase": "warming_up"} in got
    assert {"n_slots": 4, "n_ctx_slot": 65536} in got
    assert {"listening": "http://0.0.0.0:8080"} in got
    assert got[-1] == {"phase": "ready", "pct": 100.0}                                         # all slots are idle
    phases = [p["phase"] for p in got if "phase" in p]
    assert phases.index("warming_up") < phases.index("ready") and "loading_weights" in phases[:3]


@pytest.mark.parametrize("line,expected", [
    ("0.01.0 I load_tensors: offloaded 0/33 layers to GPU", {"phase": "loading_weights", "gpu_layers": 0, "total_layers": 33, "warning": "cpu_only"}),
    ("load_tensors: offloaded 20/33 layers to GPU", {"phase": "loading_weights", "gpu_layers": 20, "total_layers": 33}),
    ("main: model loaded", {"phase": "ready", "pct": 100.0}),
    ("srv  update_slots: all slots are idle", {"phase": "ready", "pct": 100.0}),
    ("srv  load_model: server is entering sleeping state", {"state": "sleeping"}),
    ("srv  load_model: server is exiting sleeping state", {"state": "awake"}),
    ("llama_context: n_seq_max     = 4", None),
    ("common_init_from_params: warming up the model with an empty run - please wait ... (--no-warmup to disable)", {"phase": "warming_up"}),
    ("0.00.1 I cmn  common_param: build 10156 (91f8c9c5f) with GNU 14.2.0 for Linux x86_64", {"build": "10156", "commit": "91f8c9c5f"}),
    ("........................................", {"phase": "loading_weights", "pct": 40.0}),
    ("." * 120, {"phase": "loading_weights", "pct": 100.0}),
    ("..", None),
    ("warning: failed to mlock 8192-byte buffer (after previously locking 0 bytes): Cannot allocate memory", None),
    ("common_fit_params: failed to fit params to free device memory: model does not fit", {"warning": "fit_failed", "message": "model does not fit"}),
    ("common_fit_params: context size reduced from 8192 to 4096 -> need 100 MiB less memory in total", {"fit_ctx_reduced": {"from": 8192, "to": 4096}}),
    ("\x1b[0;32mmain: model loaded\x1b[0m", {"phase": "ready", "pct": 100.0}),
])
def test_synthetic_single_lines(line: str, expected: dict | None) -> None:  # type: ignore[type-arg]
    got = parse_log_line(line)
    if "failed to mlock" in line:
        assert got is not None and got["warning"] == "mlock_failed" and "Cannot allocate memory" in got["message"]
    else:
        assert got == expected


@pytest.mark.parametrize("line,code", [
    ("ggml_backend_cuda_buffer_type_alloc_buffer: allocating 98304.00 MiB on device 0: cudaMalloc failed: out of memory", "cuda_oom"),
    ("alloc_tensor_range: failed to allocate CUDA0 buffer of size 105226698752", "cuda_oom"),
    ("CUDA error: out of memory", "cuda_oom"),
    ("llama_context: V cache quantization requires flash_attn", "kv_quant_needs_flash_attn"),
    ("llama_context: SPLIT_MODE_TENSOR requires flash_attn to be enabled", "split_tensor_needs_flash_attn"),
    ("srv  operator(): couldn't bind HTTP server socket, hostname: 0.0.0.0, port: 8080", "port_in_use"),
    ("llama_model_load: error loading model: unsupported model architecture: 'newarch'", "unsupported_architecture"),
    ("llama_init_from_model: failed to create llama_context from model", "context_create_failed"),
    ("srv  llama_server: exiting due to HTTP server error", "http_server_failed"),
    ("srv  load_model: failed to load model, '/models/x.gguf'", "model_load_failed"),
])
def test_synthetic_error_lines(line: str, code: str) -> None:
    got = parse_log_line("0.00.9 E " + line)
    assert got is not None and got["error"] == code and line[-40:] in got["message"]


def test_errors_win_over_progress_in_the_same_line() -> None:
    line = "load_tensors: offloaded 33/33 layers to GPU cudaMalloc failed: out of memory"
    assert parse_log_line(line) == {"error": "cuda_oom", "message": line}


def test_long_messages_are_cut_and_nothing_else_leaks() -> None:
    got = parse_log_line("E failed to load model " + "x" * 1000)
    assert got is not None and len(got["message"]) == 300


def test_no_pattern_matches_ordinary_chatter() -> None:
    for line in ("srv  log_server_r: request: GET /health 127.0.0.1 200", "slot launch_slot_: id  0 | task 0 | processing task",
                 "slot update_slots: id  0 | task 0 | prompt processing progress, n_past = 512, n_tokens = 512, progress = 0.5",
                 "print_timings: prompt eval time = 12.0 ms / 10 tokens", re.sub("x", "y", "xxx")):
        assert parse_log_line(line) is None, line
