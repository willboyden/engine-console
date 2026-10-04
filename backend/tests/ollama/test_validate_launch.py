"""validate() and the launch (golden env/argv) of the Ollama adapter."""
from __future__ import annotations

from typing import Any

import pytest

from engine_console.adapters.base import Hardware, LaunchContext, LaunchSpec
from engine_console.adapters.ollama import OllamaAdapter
from engine_console.adapters.ollama_params import DEFAULT_IMAGE

RESERVED = {"OLLAMA_HOST": "0.0.0.0:11434", "OLLAMA_MODELS": "/root/.ollama/models", "OLLAMA_NO_CLOUD": "1", "OLLAMA_NOPRUNE": "1"}


def launch(adapter: OllamaAdapter, hw: Hardware, params: dict[str, Any] | None = None, *, model: str = "llama3.1:8b",
           fmt: str = "ollama") -> LaunchSpec:
    ctx = LaunchContext(model=model, model_format=fmt, model_path=None,  # type: ignore[arg-type]
                        served_name="ignored", hf_cache_container_path="/hf", params=params or {}, hw=hw)
    return adapter.build_launch_ctx(ctx)


# ------------------------------------------------------------------ validate
def test_presets_validate(adapter: OllamaAdapter) -> None:
    pre = adapter.presets()
    assert set(pre) == {"balanced", "max-throughput", "low-latency", "long-context", "low-memory", "many-models",
                        "big-model-spread"}
    for name, p in pre.items():
        assert adapter.validate(p) == [], name


def test_validate_unknown_and_none(adapter: OllamaAdapter) -> None:
    assert adapter.validate({"nope": 1}) == ["unknown parameter 'nope'"]
    assert adapter.validate({"context_length": None, "flash_attention": None}) == []
    assert adapter.validate({}) == []


@pytest.mark.parametrize("params", [
    {"context_length": -1}, {"context_length": 1.5}, {"context_length": True}, {"context_length": "8192"},
    {"context_length": float("inf")}, {"num_parallel": 0}, {"num_parallel": -2}, {"max_loaded_models": -1},
    {"max_queue": 0}, {"gpu_overhead": -1}, {"fit_target": -5}, {"max_transfer_streams": 0},
    {"flash_attention": 1}, {"flash_attention": "true"}, {"sched_spread": "yes"}, {"debug": 1}, {"igpu_enable": "on"},
    {"kv_cache_type": "q5_0"}, {"kv_cache_type": "F16"}, {"fit": "auto"},
    {"keep_alive": "soon"}, {"keep_alive": "5 minutes"}, {"keep_alive": ""}, {"keep_alive": True}, {"keep_alive": "5m;rm"},
    {"keep_alive": "m5"}, {"load_timeout": "-1"}, {"load_timeout": "-5m"}, {"load_timeout": "later"},
    {"origins": []}, {"origins": "https://a.example"}, {"origins": ["a b"]}, {"origins": ["a,b"]}, {"origins": [""]},
    {"llm_library": "cuda v13"}, {"llm_library": "../x"}, {"llm_library": "CUDA"}, {"llm_library": ""},
])
def test_validate_rejects(adapter: OllamaAdapter, params: dict[str, Any]) -> None:
    assert adapter.validate(params), params


@pytest.mark.parametrize("params", [
    {"keep_alive": "5m"}, {"keep_alive": "24h"}, {"keep_alive": "1h30m"}, {"keep_alive": "90s"}, {"keep_alive": "0"},
    {"keep_alive": "-1"}, {"keep_alive": "-1m"}, {"keep_alive": 3600}, {"keep_alive": "0.5h"}, {"keep_alive": "300ms"},
    {"load_timeout": "30m"}, {"load_timeout": 600}, {"origins": ["https://chat.example.org", "app://*"]},
    {"llm_library": "cuda_v13"}, {"llm_library": "vulkan"}, {"fit": "off"}, {"fit_target": 0},
    {"kv_cache_type": "q4_0"}, {"gpu_overhead": 2147483648}, {"num_parallel": 16}, {"max_loaded_models": 0},
    {"context_length": 0}, {"context_length": 262144},
])
def test_validate_accepts(adapter: OllamaAdapter, params: dict[str, Any]) -> None:
    assert adapter.validate(params) == [], params


# ------------------------------------------------------------------ launch: golden env
def test_golden_bare_server(adapter: OllamaAdapter, hw: Hardware) -> None:
    spec = launch(adapter, hw, model="")
    assert spec.image == DEFAULT_IMAGE == "ollama/ollama:0.30.11"
    assert spec.argv == ["serve"]
    assert spec.env == RESERVED
    assert (spec.container_port, spec.health_path, spec.metrics_path, spec.models_path) == (
        11434, "/api/version", "/api/version", "/v1/models")
    assert spec.needs_hf_cache is False and spec.ipc_host is False and spec.shm_size == "1g" and spec.files == {}


def test_golden_balanced_with_a_named_model(adapter: OllamaAdapter, hw: Hardware) -> None:
    spec = launch(adapter, hw, adapter.presets()["balanced"], model="llama3.1:8b")
    assert spec.argv == ["serve"]                                            # nothing is preloaded: the first request loads it
    assert spec.env == {**RESERVED, "OLLAMA_CONTEXT_LENGTH": "8192", "OLLAMA_KEEP_ALIVE": "30m"}


def test_golden_complex(adapter: OllamaAdapter, hw: Hardware) -> None:
    params = {
        "context_length": 32768, "num_parallel": 4, "max_loaded_models": 2, "keep_alive": "-1", "kv_cache_type": "q8_0",
        "flash_attention": True, "gpu_overhead": 2147483648, "sched_spread": True, "max_queue": 128, "load_timeout": "30m",
        "origins": ["https://chat.example.org", "http://localhost:3000"], "llm_library": "cuda_v13", "igpu_enable": False,
        "fit": "off", "fit_target": 2048, "max_transfer_streams": 8, "debug": True,
    }
    assert launch(adapter, hw, params).env == {
        **RESERVED,
        "OLLAMA_CONTEXT_LENGTH": "32768", "OLLAMA_NUM_PARALLEL": "4", "OLLAMA_MAX_LOADED_MODELS": "2",
        "OLLAMA_KEEP_ALIVE": "-1", "OLLAMA_KV_CACHE_TYPE": "q8_0", "OLLAMA_FLASH_ATTENTION": "1",
        "OLLAMA_GPU_OVERHEAD": "2147483648", "OLLAMA_SCHED_SPREAD": "1", "OLLAMA_MAX_QUEUE": "128",
        "OLLAMA_LOAD_TIMEOUT": "30m", "OLLAMA_ORIGINS": "https://chat.example.org,http://localhost:3000",
        "OLLAMA_LLM_LIBRARY": "cuda_v13", "LLAMA_ARG_FIT": "off",           # igpu_enable=False sends nothing (see test_live_env)
        "LLAMA_ARG_FIT_TARGET": "2048", "OLLAMA_MAX_TRANSFER_STREAMS": "8", "OLLAMA_DEBUG": "1"}


def test_every_preset_builds(adapter: OllamaAdapter, hw: Hardware) -> None:
    for name, params in adapter.presets().items():
        spec = launch(adapter, hw, params)
        assert spec.argv == ["serve"] and RESERVED.items() <= spec.env.items(), name
        assert set(spec.env) - set(RESERVED) == {p.flag[4:] for p in adapter.param_catalog() if p.key in params}, name


def test_the_reserved_variables_cannot_be_overridden(adapter: OllamaAdapter, hw: Hardware) -> None:
    """Cloud features stay off, pruning stays off, and the listen address and store path are the adapter's."""
    spec = launch(adapter, hw, {"debug": True, "origins": ["https://a.example"]})
    for name, value in RESERVED.items():
        assert spec.env[name] == value
    for key in ("ollama_no_cloud", "no_cloud", "noprune", "host", "models"):
        with pytest.raises(ValueError, match="unknown parameter"):
            launch(adapter, hw, {key: "0"})


def test_false_booleans_are_explicit_zeros(adapter: OllamaAdapter, hw: Hardware) -> None:
    env = launch(adapter, hw, {"flash_attention": False, "sched_spread": False, "debug": False}).env
    assert env["OLLAMA_FLASH_ATTENTION"] == "0" and env["OLLAMA_SCHED_SPREAD"] == "0" and env["OLLAMA_DEBUG"] == "0"
    unset = launch(adapter, hw, {}).env
    assert "OLLAMA_FLASH_ATTENTION" not in unset                             # tri-state: unset lets Ollama decide


def test_duration_values_are_trimmed_and_numbers_become_seconds(adapter: OllamaAdapter, hw: Hardware) -> None:
    assert launch(adapter, hw, {"keep_alive": " 10m "}).env["OLLAMA_KEEP_ALIVE"] == "10m"
    assert launch(adapter, hw, {"keep_alive": 3600}).env["OLLAMA_KEEP_ALIVE"] == "3600"


@pytest.mark.parametrize("name", ["llama3.1:8b", "library/qwen3:30b-a3b-q4_K_M", "hf.co/org/repo-GGUF:Q4_K_M",
                                  "registry.example.org/team/model:1.0", "gemma3", ""])
def test_model_names_pass(adapter: OllamaAdapter, hw: Hardware, name: str) -> None:
    assert launch(adapter, hw, model=name).argv == ["serve"]


@pytest.mark.parametrize("name", ["-x", "a b", "a;b", "a\nb", "../etc", " llama", "x" * 300, "model|cat"])
def test_bad_model_names_are_refused(adapter: OllamaAdapter, hw: Hardware, name: str) -> None:
    with pytest.raises(ValueError, match="Ollama name"):
        launch(adapter, hw, model=name)


@pytest.mark.parametrize("fmt", ["hf", "gguf", "plain"])
def test_other_formats_are_not_loadable_yet(adapter: OllamaAdapter, hw: Hardware, fmt: str) -> None:
    with pytest.raises(ValueError) as e:
        launch(adapter, hw, fmt=fmt)
    assert "own store" in str(e.value) and "not supported yet" in str(e.value) and fmt in str(e.value)


def test_invalid_params_raise(adapter: OllamaAdapter, hw: Hardware) -> None:
    with pytest.raises(ValueError, match="unknown parameter"):
        launch(adapter, hw, {"bogus": 1})
    with pytest.raises(ValueError, match="num_parallel"):
        launch(adapter, hw, {"num_parallel": 0})


def test_v1_build_launch_bridges_to_the_context_path(adapter: OllamaAdapter, hw: Hardware) -> None:
    spec = adapter.build_launch("llama3.1:8b", {"num_parallel": 2}, hw, served_name="x", hf_cache_container_path="/hf")
    assert spec.argv == ["serve"] and spec.env["OLLAMA_NUM_PARALLEL"] == "2" and spec.needs_hf_cache is False
    with pytest.raises(ValueError, match="Ollama name"):
        adapter.build_launch("org/Model with spaces", {}, hw, served_name="x", hf_cache_container_path="/hf")


def test_no_secret_can_reach_the_environment(adapter: OllamaAdapter, hw: Hardware) -> None:
    """Ollama has no API key for its local server, so nothing secret-named is ever set."""
    env = launch(adapter, hw, adapter.presets()["many-models"]).env
    assert not any(any(w in k.lower() for w in ("key", "token", "secret", "password")) for k in env)


def test_preload_request_is_a_plain_generate_without_a_prompt(adapter: OllamaAdapter) -> None:
    path, body = adapter.preload_request("llama3.1:8b", {})
    assert path == "/api/generate" and body == {"model": "llama3.1:8b", "stream": False} and "prompt" not in body
    with pytest.raises(ValueError, match="Ollama name"):
        adapter.preload_request("", {})
    with pytest.raises(ValueError, match="Ollama name"):
        adapter.preload_request("bad name", {})
