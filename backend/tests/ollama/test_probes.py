"""What was MEASURED on the pinned image (a bare `ollama serve`, no model, no GPU, no network) and what the adapter does
about it. The captured outputs are in fixtures/."""
from __future__ import annotations

import re

from engine_console.adapters.base import Hardware, LaunchContext
from engine_console.adapters.ollama import OllamaAdapter

from .helpers import read


def _section(probe: str, path: str) -> str:
    body = probe.split(f"== {path}\n", 1)[1]
    return re.split(r"\n== ", body, maxsplit=1)[0]


def test_endpoints_of_a_bare_server() -> None:
    probe = read("endpoints_bare_0.30.11.txt")
    assert '{"version":"0.30.11"}' in _section(probe, "/api/version")
    assert "Ollama is running" in _section(probe, "/")
    assert '{"models":[]}' in _section(probe, "/api/tags") and '{"models":[]}' in _section(probe, "/api/ps")
    assert '{"object":"list","data":null}' in _section(probe, "/v1/models")        # data is null, not [], when the store is empty
    assert "404 Not Found" in _section(probe, "/metrics")


def test_a_read_only_store_needs_blobs_and_manifests_directories() -> None:
    """Why the module docstring tells the core to offer only stores that have both."""
    failed = read("serve_readonly_no_blobs_0.30.11.log")
    assert "mkdir /root/.ollama/models/blobs: read-only file system" in failed
    assert "Listening on" not in failed
    started = read("serve_bare_0.30.11.log")
    assert "Listening on" in started and "OLLAMA_NOPRUNE:true" in started       # with both directories present it starts


def test_the_server_generates_its_own_identity_under_home_not_the_store() -> None:
    """The container needs a writable /root/.ollama (its own layer): the store itself stays read-only."""
    for name in ("serve_bare_0.30.11.log", "serve_readonly_no_blobs_0.30.11.log"):
        text = read(name)
        notice = next(ln for ln in text.splitlines() if "Generating new private key" in ln)
        assert "/root/.ollama/" in notice and "/root/.ollama/models" not in notice
        assert not re.search(r"AAAA[0-9A-Za-z+/]{20,}", text)                   # the throwaway public key was not committed


def test_the_launch_never_asks_for_a_writable_store_or_the_cloud(adapter: OllamaAdapter, hw: Hardware) -> None:
    spec = adapter.build_launch_ctx(LaunchContext(model="", model_format="ollama", served_name="x", hf_cache_container_path="/hf",
                                                  params={}, hw=hw))
    assert spec.env["OLLAMA_NOPRUNE"] == "1" and spec.env["OLLAMA_NO_CLOUD"] == "1"
    assert spec.files == {} and spec.needs_hf_cache is False
