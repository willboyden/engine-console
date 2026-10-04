"""The Ollama catalog against the captured help and the captured "server config" start-up line of the pinned image."""
from __future__ import annotations

import re

from engine_console.adapters.base import ParamSpec
from engine_console.adapters.ollama import OllamaAdapter
from engine_console.adapters.ollama_params import (
    CONTAINER_PORT,
    DEFAULT_IMAGE,
    IMAGE_DIGEST,
    KV_BYTES,
    KV_CACHE_TYPES,
    MODELS_DIR,
    RESERVED_ENV,
    VERSION,
)

from .helpers import help_env, read, server_config

SECRET_KEY = re.compile(r"(key|token|secret|password)", re.I)      # services/secrets_store.py


def by_key(adapter: OllamaAdapter) -> dict[str, ParamSpec]:
    return {p.key: p for p in adapter.param_catalog()}


def env_of(adapter: OllamaAdapter) -> dict[str, ParamSpec]:
    return {p.flag[4:]: p for p in adapter.param_catalog()}


def test_help_fixture_is_the_pinned_version() -> None:
    assert VERSION == "0.30.11" and DEFAULT_IMAGE == "ollama/ollama:0.30.11"
    assert '{"version":"0.30.11"}' in read("endpoints_bare_0.30.11.txt")
    assert "version 0.30.11" in read("serve_bare_0.30.11.log")
    assert len(help_env()) == 21                                      # the parser reads the whole list


def test_catalog_is_the_help_minus_what_the_adapter_owns(adapter: OllamaAdapter) -> None:
    listed = set(help_env())
    exposed = set(env_of(adapter))
    assert exposed | set(RESERVED_ENV) == listed and not exposed & set(RESERVED_ENV)
    assert len(exposed) == 17 and len(RESERVED_ENV) == 4
    assert set(RESERVED_ENV) == {"OLLAMA_HOST", "OLLAMA_MODELS", "OLLAMA_NO_CLOUD", "OLLAMA_NOPRUNE"}


def test_catalog_size_and_uniqueness(adapter: OllamaAdapter) -> None:
    cat = adapter.param_catalog()
    assert len({p.key for p in cat}) == len({p.flag for p in cat}) == len(cat)


def test_every_param_is_well_formed(adapter: OllamaAdapter) -> None:
    for p in adapter.param_catalog():
        assert re.fullmatch(r"[a-z][a-z0-9]*(_[a-z0-9]+)*", p.key), p.key
        assert re.fullmatch(r"env:[A-Z0-9_]+", p.flag), p.flag          # Ollama is configured through the environment only
        assert p.help.strip() and p.label.strip() and p.help.endswith(".")
        assert p.docs_url and p.docs_url.startswith("https://docs.ollama.com/")
        assert p.requires_restart
        if p.type == "enum":
            assert p.choices, p.key
            if p.default is not None:
                assert str(p.default) in p.choices, p.key
        else:
            assert p.choices is None, p.key
        if p.type in ("int", "float") and p.default is not None and p.min is not None:
            assert p.default >= p.min, p.key
        assert p.unit is None or p.unit in {"tokens", "models", "bytes", "MiB", "requests"}, p.key
        assert p.level in (None, "expert") and (p.level != "expert" or p.advanced), p.key


def test_no_param_key_looks_like_a_secret(adapter: OllamaAdapter) -> None:
    """The core replaces any string value under a key matching key|token|secret|password and keeps it out of argv."""
    assert not [p.key for p in adapter.param_catalog() if SECRET_KEY.search(p.key)]


def test_reserved_variables_are_never_parameters(adapter: OllamaAdapter) -> None:
    for name in RESERVED_ENV:
        assert name not in env_of(adapter) and adapter.validate({name.lower(): "x"}) != []
    for key in ("host", "models", "no_cloud", "noprune", "ollama_host", "ollama_models"):
        assert adapter.validate({key: "x"}) == [f"unknown parameter '{key}'"]


def test_help_descriptions_back_the_catalog_text(adapter: OllamaAdapter) -> None:
    helps = help_env()
    assert "4k/32k/256k based on VRAM" in helps["OLLAMA_CONTEXT_LENGTH"]
    assert 'default "5m"' in helps["OLLAMA_KEEP_ALIVE"] and 'default "5m"' in helps["OLLAMA_LOAD_TIMEOUT"]
    assert "per GPU" in helps["OLLAMA_MAX_LOADED_MODELS"]
    assert "(default: f16)" in helps["OLLAMA_KV_CACHE_TYPE"]
    assert "bytes" in helps["OLLAMA_GPU_OVERHEAD"]
    assert "(default 4)" in helps["OLLAMA_MAX_TRANSFER_STREAMS"]
    assert 'default "on"' in helps["LLAMA_ARG_FIT"] and "MiB" in helps["LLAMA_ARG_FIT_TARGET"]
    assert "comma separated" in helps["OLLAMA_ORIGINS"]
    assert "cloud" in helps["OLLAMA_NO_CLOUD"] and "prune" in helps["OLLAMA_NOPRUNE"]


def _same_duration(a: str, b: str) -> bool:
    def seconds(text: str) -> float:
        unit = {"s": 1.0, "m": 60.0, "h": 3600.0}
        return sum(float(n) * unit[u] for n, u in re.findall(r"(\d+(?:\.\d+)?)([smh])", text))
    return seconds(a) == seconds(b)


def test_defaults_match_the_server_config_line_of_the_pinned_image(adapter: OllamaAdapter) -> None:
    """`msg="server config"` is the engine's own registry of effective values; every default the catalog states as a
    literal is compared with it (an unset variable shows as an empty value there)."""
    cfg = server_config()
    cat = env_of(adapter)
    for name, key in (("OLLAMA_CONTEXT_LENGTH", "context_length"), ("OLLAMA_NUM_PARALLEL", "num_parallel"),
                      ("OLLAMA_MAX_LOADED_MODELS", "max_loaded_models"), ("OLLAMA_MAX_QUEUE", "max_queue"),
                      ("OLLAMA_MAX_TRANSFER_STREAMS", "max_transfer_streams"), ("OLLAMA_GPU_OVERHEAD", "gpu_overhead")):
        assert int(cfg[name]) == cat[name].default, name
    assert _same_duration(cfg["OLLAMA_KEEP_ALIVE"], str(cat["OLLAMA_KEEP_ALIVE"].default))
    assert _same_duration(cfg["OLLAMA_LOAD_TIMEOUT"], str(cat["OLLAMA_LOAD_TIMEOUT"].default))
    assert cfg["OLLAMA_SCHED_SPREAD"] == "false" and cat["OLLAMA_SCHED_SPREAD"].default is False
    assert cfg["OLLAMA_DEBUG"] == "INFO" and cat["OLLAMA_DEBUG"].default is False      # INFO is the non-debug log level
    assert cfg["OLLAMA_KV_CACHE_TYPE"] == "" and cat["OLLAMA_KV_CACHE_TYPE"].default == "f16"   # unset means f16 (the help)
    assert cfg["LLAMA_ARG_FIT"] == "" and cat["LLAMA_ARG_FIT"].default == "on"                  # unset means on (the help)


def test_defaults_the_config_line_cannot_state(adapter: OllamaAdapter) -> None:
    """Tri-state or computed: left as None so the form shows "engine decides" and no variable is set."""
    cfg = server_config()
    cat = env_of(adapter)
    assert cfg["OLLAMA_FLASH_ATTENTION"] == "false" and cat["OLLAMA_FLASH_ATTENTION"].default is None   # the FAQ: unset lets Ollama decide
    for name in ("OLLAMA_IGPU_ENABLE", "OLLAMA_LLM_LIBRARY", "LLAMA_ARG_FIT_TARGET", "OLLAMA_ORIGINS"):
        assert cat[name].default is None, name
    origins = cfg["OLLAMA_ORIGINS"].strip("[]").split()
    assert "http://localhost" in origins and "app://*" in origins and "*" not in origins     # not open to every website


def test_the_config_line_confirms_the_reserved_settings_took_effect() -> None:
    """The bare server was started with the same reserved variables the adapter sets."""
    cfg = server_config()
    assert cfg["OLLAMA_NO_CLOUD"] == "true" and cfg["OLLAMA_NOPRUNE"] == "true"
    assert cfg["OLLAMA_MODELS"] == MODELS_DIR == "/root/.ollama/models"
    assert "Ollama cloud disabled: true" in read("serve_bare_0.30.11.log")


def test_kv_cache_types_match_the_help_and_the_docs(adapter: OllamaAdapter) -> None:
    assert by_key(adapter)["kv_cache_type"].choices == KV_CACHE_TYPES == ["f16", "q8_0", "q4_0"] == list(KV_BYTES)
    assert KV_BYTES == {"f16": 2.0, "q8_0": 1.0625, "q4_0": 0.5625}
    assert 0.5 < KV_BYTES["q8_0"] / KV_BYTES["f16"] < 0.55 and 0.25 < KV_BYTES["q4_0"] / KV_BYTES["f16"] < 0.3   # "about 1/2" and "about 1/4"


def test_the_env_flags_cover_the_documented_tri_states(adapter: OllamaAdapter) -> None:
    cat = by_key(adapter)
    assert cat["flash_attention"].type == "bool" and cat["flash_attention"].default is None
    assert cat["keep_alive"].type == "string" and cat["load_timeout"].type == "string"
    assert cat["origins"].type == "string_list"


def test_required_params_present_and_groups(adapter: OllamaAdapter) -> None:
    keys = set(by_key(adapter))
    assert {"context_length", "num_parallel", "max_loaded_models", "keep_alive", "kv_cache_type", "flash_attention",
            "gpu_overhead", "sched_spread", "max_queue", "load_timeout", "origins", "llm_library", "igpu_enable", "fit",
            "fit_target", "max_transfer_streams", "debug"} == keys
    assert {p.group for p in adapter.param_catalog()} == {"context", "memory", "scheduling", "parallelism", "performance",
                                                          "network", "advanced"}


def test_memory_params_flagged(adapter: OllamaAdapter) -> None:
    cat = by_key(adapter)
    for k in ("context_length", "num_parallel", "kv_cache_type", "max_loaded_models", "gpu_overhead", "sched_spread",
              "flash_attention"):
        assert cat[k].affects_memory, k
    assert not cat["keep_alive"].affects_memory and not cat["debug"].affects_memory


def test_adapter_identity_and_pin(adapter: OllamaAdapter) -> None:
    assert (adapter.id, adapter.display_name, adapter.default_port) == ("ollama", "Ollama", 11434) and CONTAINER_PORT == 11434
    assert adapter.default_image == "ollama/ollama:0.30.11" and ":latest" not in adapter.default_image
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", IMAGE_DIGEST)
    assert adapter.model_formats == frozenset({"ollama"}) and adapter.multi_model is True
    assert adapter.metric_prefixes == ()
    assert adapter.image_for("aarch64") == adapter.image_for("x86_64") == DEFAULT_IMAGE


def test_catalog_returns_copies(adapter: OllamaAdapter) -> None:
    adapter.param_catalog()[0].label = "mutated"
    assert adapter.param_catalog()[0].label != "mutated"
    adapter.presets()["balanced"]["context_length"] = 1
    assert adapter.presets()["balanced"]["context_length"] == 8192
