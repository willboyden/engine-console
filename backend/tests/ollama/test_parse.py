"""parse_metrics() (Ollama has none) and parse_startup_log().

Captured (real, from the pinned image, a bare server with no model): serve_bare_0.30.11.log,
serve_readonly_no_blobs_0.30.11.log, endpoints_bare_0.30.11.txt.
Synthetic (inline, labelled): the runner lines, whose wording is the llama.cpp one and was not seen in a live Ollama run.
"""
from __future__ import annotations

import pytest

from engine_console.adapters.base import Hardware
from engine_console.adapters.ollama import OllamaAdapter, parse_log_line

from .helpers import read


def test_ollama_exports_no_prometheus_metrics(adapter: OllamaAdapter) -> None:
    assert adapter.parse_metrics("") == {}
    assert adapter.parse_metrics("# HELP x y\nvllm:num_requests_running 3\nllamacpp:requests_processing 1\nollama_x 1\n") == {}
    assert adapter.parse_metrics('{"version":"0.30.11"}') == {}                      # what the metrics_path really returns
    assert adapter.metric_prefixes == ()


def test_the_metrics_path_is_a_cheap_read_only_call_the_node_allows(adapter: OllamaAdapter, hw: Hardware) -> None:
    probe = read("endpoints_bare_0.30.11.txt")
    assert "== /metrics\nHTTP/1.0 404 Not Found" in probe                              # why there is nothing to scrape
    assert "HTTP/1.0 200 OK" in probe.split("== /api/version", 1)[1].split("==", 1)[0]
    assert '{"version":"0.30.11"}' in probe
    spec = adapter.build_launch("", {}, hw, served_name="x", hf_cache_container_path="/hf")
    assert spec.metrics_path == spec.health_path == "/api/version" and spec.container_port == 11434
    assert spec.metrics_path.startswith("/api/")                                      # FLEET-PROTOCOL: `/api/` (Ollama) is allowed


# ------------------------------------------------------------------ startup log: captured
def test_captured_bare_server_start(adapter: OllamaAdapter) -> None:
    lines = read("serve_bare_0.30.11.log").splitlines()
    parsed = [(line, adapter.parse_startup_log(line)) for line in lines]
    got = [p for _, p in parsed if p]
    assert {"cloud_disabled": True} in got
    assert {"phase": "ready", "pct": 100.0, "listening": "127.0.0.1:11434", "version": "0.30.11"} in got
    assert {"gpu": {"id": "cpu", "library": "cpu", "total_gib": 246.8, "available_gib": 246.7}, "warning": "cpu_only"} in got
    assert {"default_num_ctx": 4096, "total_vram_gib": 0.0} in got
    assert len(got) == 4
    # the request log, the key-generation notice and the server config carry no progress
    for line, p in parsed:
        if line.startswith("[GIN]") or "server config" in line or "Generating new private key" in line:
            assert p is None, line


def test_captured_read_only_store_failure(adapter: OllamaAdapter) -> None:
    lines = [ln for ln in read("serve_readonly_no_blobs_0.30.11.log").splitlines() if ln.strip()]
    got = [p for p in (adapter.parse_startup_log(ln) for ln in lines) if p]
    assert got[-1]["error"] == "models_dir_read_only"
    assert "mkdir /root/.ollama/models/blobs" in got[-1]["message"] and "read-only file system" in got[-1]["message"]
    assert not any(p.get("phase") == "ready" for p in got)                           # it never started listening


def test_captured_lines_one_by_one() -> None:
    p = parse_log_line
    assert p('time=2026-10-04T13:53:10.815Z level=INFO source=routes.go:2023 msg="Listening on 127.0.0.1:11434 (version 0.30.11)"') == {
        "phase": "ready", "pct": 100.0, "listening": "127.0.0.1:11434", "version": "0.30.11"}
    assert p('time=2026-10-04T13:53:11.082Z level=INFO source=routes.go:2073 msg="vram-based default context" total_vram="0 B" default_num_ctx=4096') == {
        "default_num_ctx": 4096, "total_vram_gib": 0.0}
    assert p('time=2026-10-04T13:53:10.816Z level=INFO source=runner.go:60 msg="discovering available GPUs..."') is None
    assert p("") is None and p("   ") is None


# ------------------------------------------------------------------ startup log: synthetic
@pytest.mark.parametrize("line,expected", [
    # a GPU discovery line in the captured shape, with GPU values made up
    ('time=2026-10-04T13:53:11.082Z level=INFO source=types.go:50 msg="inference compute" id=GPU-aaaa library=CUDA compute=12.0 name=CUDA0 total="95.0 GiB" available="94.0 GiB"',
     {"gpu": {"id": "GPU-aaaa", "library": "CUDA", "total_gib": 95.0, "available_gib": 94.0}}),
    ('time=2026-10-04T13:53:11.082Z level=INFO source=routes.go:2073 msg="vram-based default context" total_vram="191.2 GiB" default_num_ctx=262144',
     {"default_num_ctx": 262144, "total_vram_gib": 191.2}),
    ('time=x level=INFO source=routes.go:1 msg="vram-based default context" total_vram="512 MiB" default_num_ctx=4096',
     {"default_num_ctx": 4096, "total_vram_gib": 0.5}),
    ("load_tensors: offloaded 33/33 layers to GPU", {"phase": "loading_weights", "gpu_layers": 33, "total_layers": 33}),
    ('time=x level=INFO msg="llama runner started in 4.52 seconds"', {"model_load_seconds": 4.52}),
    ('time=x level=INFO msg="Ollama cloud disabled: false"', {"cloud_disabled": False}),
])
def test_synthetic_lines(line: str, expected: dict) -> None:  # type: ignore[type-arg]
    assert parse_log_line(line) == expected


@pytest.mark.parametrize("line,code", [
    ('time=x level=ERROR source=sched.go:1 msg="cudaMalloc failed: out of memory"', "cuda_oom"),
    ('time=x level=ERROR msg="llama runner terminated: CUDA error: out of memory"', "cuda_oom"),
    ("Error: listen tcp 0.0.0.0:11434: bind: address already in use", "port_in_use"),
    ("Error: mkdir /root/.ollama/models/blobs: read-only file system: ensure path elements are traversable", "models_dir_read_only"),
    ('time=x level=ERROR source=routes.go:9 msg="something broke"', "engine_error"),
    ("Error: unexpected failure", "engine_error"),
])
def test_synthetic_errors(line: str, code: str) -> None:
    got = parse_log_line(line)
    assert got is not None and got["error"] == code and got["message"]


def test_long_messages_are_cut() -> None:
    got = parse_log_line("Error: " + "x" * 1000)
    assert got is not None and len(got["message"]) == 300


def test_ordinary_info_lines_carry_no_progress() -> None:
    for line in ('time=x level=INFO source=model_list_cache.go:111 msg="model list cache hydration complete" models=0 failures=0',
                 'time=x level=INFO msg="server config" env="map[OLLAMA_DEBUG:INFO]"',
                 '[GIN] 2026/10/04 - 13:53:18 | 200 | 139.641µs | 127.0.0.1 | GET "/api/version"',
                 "plain text", 'time=x level=WARN msg="slow"'):
        assert parse_log_line(line) is None, line
