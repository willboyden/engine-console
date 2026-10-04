"""The adapter's launch environment against the LIVE start-up of the pinned image (fixtures/live_env_0.30.11.json, produced by
live_check.py: a bare server, no model, no GPU, no network, an empty read-only store): the effective value the server logged
for every variable, and how it treats the value forms the validator accepts and rejects."""
from __future__ import annotations

import json
from typing import Any

import pytest

from engine_console.adapters.base import Hardware, LaunchContext
from engine_console.adapters.ollama import OllamaAdapter
from engine_console.adapters.ollama_params import DEFAULT_IMAGE

from .helpers import FIX
from .live_cases import ADAPTER_PARAMS, FORMS

RERUN = "the environment the adapter builds changed since the live check: run `uv run python tests/ollama/live_check.py`"
FOREVER = "2562047h47m16.854775807s"       # what the server shows for a negative keep-alive: the largest duration


def record() -> dict[str, Any]:
    out: dict[str, Any] = json.loads((FIX / "live_env_0.30.11.json").read_text(encoding="utf-8"))
    return out


def runs() -> dict[str, dict[str, Any]]:
    return {r["name"]: r for r in record()["runs"]}


def test_the_recorded_run_used_the_pinned_image() -> None:
    assert record()["image"] == DEFAULT_IMAGE == "ollama/ollama:0.30.11"


def test_the_adapter_still_builds_the_recorded_environment(adapter: OllamaAdapter, hw: Hardware) -> None:
    spec = adapter.build_launch_ctx(LaunchContext(model="", model_format="ollama", served_name="x", hf_cache_container_path="/hf",
                                                  params=ADAPTER_PARAMS, hw=hw))
    assert dict(spec.env) == runs()["adapter"]["env"], RERUN
    assert [(n, e) for n, e in FORMS] == [(r["name"], r["env"]) for r in record()["runs"][1:]], RERUN
    assert set(spec.env) - {"OLLAMA_HOST", "OLLAMA_MODELS", "OLLAMA_NO_CLOUD", "OLLAMA_NOPRUNE"} == {
        p.flag[4:] for p in adapter.param_catalog()}                       # one of every parameter was exercised


def test_every_variable_took_the_effective_value_the_adapter_meant() -> None:
    eff = runs()["adapter"]["effective"]
    assert {k: eff[k] for k in ("OLLAMA_CONTEXT_LENGTH", "OLLAMA_NUM_PARALLEL", "OLLAMA_MAX_LOADED_MODELS", "OLLAMA_MAX_QUEUE",
                                "OLLAMA_MAX_TRANSFER_STREAMS", "OLLAMA_GPU_OVERHEAD", "LLAMA_ARG_FIT_TARGET")} == {
        "OLLAMA_CONTEXT_LENGTH": "32768", "OLLAMA_NUM_PARALLEL": "4", "OLLAMA_MAX_LOADED_MODELS": "2", "OLLAMA_MAX_QUEUE": "128",
        "OLLAMA_MAX_TRANSFER_STREAMS": "8", "OLLAMA_GPU_OVERHEAD": "2147483648", "LLAMA_ARG_FIT_TARGET": "2048"}
    assert (eff["OLLAMA_FLASH_ATTENTION"], eff["OLLAMA_SCHED_SPREAD"], eff["OLLAMA_DEBUG"]) == ("true", "true", "DEBUG")
    assert eff["OLLAMA_KEEP_ALIVE"] == FOREVER and eff["OLLAMA_LOAD_TIMEOUT"] == "30m0s"
    assert eff["OLLAMA_KV_CACHE_TYPE"] == "q8_0" and eff["OLLAMA_LLM_LIBRARY"] == "cuda_v13" and eff["LLAMA_ARG_FIT"] == "off"
    assert eff["OLLAMA_IGPU_ENABLE"] == "1"
    assert eff["OLLAMA_HOST"] == "http://0.0.0.0:11434" and eff["OLLAMA_MODELS"] == "/root/.ollama/models"


def test_the_reserved_variables_took_effect() -> None:
    eff = runs()["adapter"]["effective"]
    assert eff["OLLAMA_NO_CLOUD"] == "true" and eff["OLLAMA_NOPRUNE"] == "true"


def test_origins_are_added_to_the_built_in_list_not_replacing_it() -> None:
    value = runs()["adapter"]["effective"]["OLLAMA_ORIGINS"].strip("[]").split()
    assert value[:2] == ["https://chat.example.org", "http://localhost:3000"]
    assert {"http://localhost", "app://*", "vscode-file://*"} <= set(value) and "*" not in value


@pytest.mark.parametrize("name,expected", [
    ("bools_zero", {"OLLAMA_FLASH_ATTENTION": "false", "OLLAMA_SCHED_SPREAD": "false", "OLLAMA_DEBUG": "INFO"}),
    ("bools_one", {"OLLAMA_FLASH_ATTENTION": "true", "OLLAMA_SCHED_SPREAD": "true", "OLLAMA_DEBUG": "DEBUG"}),
    ("bools_false", {"OLLAMA_FLASH_ATTENTION": "false", "OLLAMA_SCHED_SPREAD": "false", "OLLAMA_DEBUG": "INFO"}),
])
def test_boolean_variables_read_one_and_zero_as_the_adapter_assumes(name: str, expected: dict[str, str]) -> None:
    assert runs()[name]["effective"] == expected


def test_the_igpu_variable_is_a_plain_string_so_false_is_never_sent(adapter: OllamaAdapter, hw: Hardware) -> None:
    """The server echoes OLLAMA_IGPU_ENABLE verbatim ("0" and "false" stay as typed, unlike the real booleans), so no one can
    tell from its configuration that they would read as off: the adapter leaves the variable unset instead."""
    assert [runs()[n]["effective"]["OLLAMA_IGPU_ENABLE"] for n in ("igpu_zero", "igpu_false", "igpu_one")] == ["0", "false", "1"]
    off = adapter.build_launch_ctx(LaunchContext(model="", model_format="ollama", served_name="x", hf_cache_container_path="/hf",
                                                 params={"igpu_enable": False}, hw=hw))
    on = adapter.build_launch_ctx(LaunchContext(model="", model_format="ollama", served_name="x", hf_cache_container_path="/hf",
                                                params={"igpu_enable": True}, hw=hw))
    assert "OLLAMA_IGPU_ENABLE" not in off.env and on.env["OLLAMA_IGPU_ENABLE"] == "1"


@pytest.mark.parametrize("name,effective,param", [
    ("keep_alive_seconds", "1h0m0s", "3600"), ("keep_alive_zero", "0s", "0"), ("keep_alive_minus_minutes", FOREVER, "-1m"),
    ("keep_alive_compound", "1h30m0s", "1h30m"),
])
def test_keep_alive_forms_the_validator_accepts_mean_what_the_help_says(adapter: OllamaAdapter, name: str, effective: str, param: str) -> None:
    assert runs()[name]["effective"]["OLLAMA_KEEP_ALIVE"] == effective
    assert adapter.validate({"keep_alive": param}) == []


def test_a_junk_keep_alive_is_silently_ignored_by_the_server_so_the_validator_rejects_it(adapter: OllamaAdapter) -> None:
    assert runs()["keep_alive_junk"]["effective"]["OLLAMA_KEEP_ALIVE"] == "5m0s"          # fell back to the default
    assert adapter.validate({"keep_alive": "soon"})


def test_a_load_timeout_in_seconds_is_read_as_seconds(adapter: OllamaAdapter) -> None:
    assert runs()["load_timeout_seconds"]["effective"]["OLLAMA_LOAD_TIMEOUT"] == "10m0s" and adapter.validate({"load_timeout": 600}) == []


def test_the_server_does_not_validate_the_kv_type_or_parallelism_so_the_adapter_must(adapter: OllamaAdapter) -> None:
    assert runs()["kv_type_junk"]["effective"]["OLLAMA_KV_CACHE_TYPE"] == "bogus"          # echoed, never rejected
    assert runs()["parallel_zero"]["effective"]["OLLAMA_NUM_PARALLEL"] == "0"
    assert adapter.validate({"kv_cache_type": "bogus"}) and adapter.validate({"num_parallel": 0})
