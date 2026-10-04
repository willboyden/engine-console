"""The claims the adapter makes about llama-server's behaviour that were MEASURED on the pinned image (router mode, no
model, no GPU, no network): API-key handling and CORS. The captured outputs are in fixtures/*_probe_b10156.log."""
from __future__ import annotations

import re

from engine_console.adapters.base import Hardware, LaunchContext, ModelInfo
from engine_console.adapters.llamacpp import LlamaCppAdapter

from .helpers import read


def _statuses(block: str) -> dict[str, int]:
    return {m.group(1): int(m.group(2)) for m in re.finditer(r"== (\S+) -> (\d+)", block)}


def test_api_key_protects_metrics_but_not_health_or_models(adapter: LlamaCppAdapter, hw: Hardware, dense: ModelInfo) -> None:
    text = read("api_key_probe_b10156.log")
    without, rest = text.split("== with key:", 1)
    with_key, log = rest.split("== LOG", 1)
    open_paths = {"/health": 200, "/v1/health": 200, "/v1/models": 200, "/models": 200}
    closed = {"/metrics": 401, "/props": 401, "/slots": 401, "/api/tags": 401}
    assert _statuses(without) == open_paths | closed
    got = _statuses(with_key)
    assert got["/health"] == 200 and got["/v1/models"] == 200 and got["/metrics"] != 401     # the key was accepted
    assert log.count("unauthorized: Invalid API Key") == 4                                    # one per refused path
    # what the adapter does with that knowledge
    assert adapter.metric_prefixes == ("llamacpp:",)
    codes = {c.code for c in adapter.compatibility(dense, {"api_key": "k"}, hw)}
    assert "api_key_blocks_metrics" in codes


def test_default_cors_reflects_any_origin_and_localhost_does_not() -> None:
    text = read("cors_probe_b10156.log")
    local, default = text.split("== default CORS", 1)
    assert "Access-Control-Allow-Origin: http://evil.example" not in local            # --cors-origins localhost
    assert "Access-Control-Allow-Origin: http://localhost:3000" in local
    assert "Access-Control-Allow-Origin: http://127.0.0.1:9000" in local
    assert "Access-Control-Allow-Origin: http://evil.example" in default              # the engine's default
    # how often the engine's own "CORS is set to allow all origins" warning was printed: only without the option
    counts = re.search(r"== LOG1 \(cors localhost\)\n(\d+)\n== LOG2 \(default\)\n(\d+)", text)
    assert counts and (counts.group(1), counts.group(2)) == ("0", "1")


def test_the_launch_applies_the_measured_hardening(adapter: LlamaCppAdapter, hw: Hardware) -> None:
    ctx = LaunchContext(model="m", model_format="gguf", model_path="/models/a.gguf", served_name="a",
                        hf_cache_container_path="/hf", params={}, hw=hw)
    argv = adapter.build_launch_ctx(ctx).argv
    assert argv[argv.index("--cors-origins") + 1] == "localhost"


def test_router_mode_has_no_metrics_without_a_model_so_no_live_scrape_exists() -> None:
    """Why the metrics fixture is synthetic: in router mode /metrics asks for a model name (400), and single-model mode
    needs a loaded model."""
    probe = read("router_mode_probe_b10156.log")
    assert "HTTP/1.1 400 Bad Request" in probe and "model name is missing from the request" in probe
    assert '{"status":"ok"}' in probe                                             # /health in router mode
