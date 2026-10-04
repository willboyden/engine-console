"""The llama.cpp catalog against the captured help of the pinned image (fixtures/llama-server-help.txt)."""
from __future__ import annotations

import math
import re

import pytest

from engine_console.adapters.base import ParamSpec
from engine_console.adapters.llamacpp import LlamaCppAdapter
from engine_console.adapters.llamacpp_params import (
    BUILD,
    CACHE_TYPE_BYTES,
    CACHE_TYPES,
    CHAT_TEMPLATES,
    COMMIT,
    DEFAULT_IMAGE,
    DOCS_URL,
    IMAGE_DIGEST,
    LAYER_KEYS,
    LOAD_MODES,
    NEGATABLE_KEYS,
    PATH_KEYS,
    SPEC_TYPES,
)

from .helpers import entry_for, help_flags, parse_help, read

SECRET_KEY = re.compile(r"(key|token|secret|password)", re.I)      # services/secrets_store.py


def by_key(adapter: LlamaCppAdapter) -> dict[str, ParamSpec]:
    return {p.key: p for p in adapter.param_catalog()}


def test_help_fixture_is_the_pinned_build() -> None:
    version = read("version_b10156.txt")
    assert f"version: {BUILD[1:]} ({COMMIT})" in version
    entries = parse_help(read("llama-server-help.txt"))
    assert len(entries) > 250                                  # the parser really reads the whole help
    assert "--n-gpu-layers" in help_flags() and "--fit" in help_flags()


def test_catalog_size_and_uniqueness(adapter: LlamaCppAdapter) -> None:
    cat = adapter.param_catalog()
    assert len(cat) >= 75
    assert len({p.key for p in cat}) == len(cat)
    assert len({p.flag for p in cat}) == len(cat)


def test_every_param_is_well_formed(adapter: LlamaCppAdapter) -> None:
    for p in adapter.param_catalog():
        assert re.fullmatch(r"[a-z][a-z0-9]*(_[a-z0-9]+)*", p.key), p.key
        assert re.fullmatch(r"--[a-z0-9][a-z0-9-]*|env:[A-Z0-9_]+", p.flag), p.flag
        assert p.help.strip() and p.label.strip() and p.help.endswith(".")
        assert p.docs_url == DOCS_URL and f"/{BUILD}/" in DOCS_URL
        assert p.requires_restart
        if p.type == "enum":
            assert p.choices, p.key
            if p.default is not None:
                assert str(p.default) in p.choices, p.key
        else:
            assert p.choices is None, p.key
        if p.min is not None and p.max is not None:
            assert p.min <= p.max
        if p.type in ("int", "float") and p.default is not None and p.min is not None:
            assert p.default >= p.min, p.key
        if p.type == "bool":
            assert p.default in (None, True, False)
        assert p.unit is None or p.unit in {"MiB", "tokens", "layers", "slots", "s", "threads"}, p.key


def test_levels_are_consistent(adapter: LlamaCppAdapter) -> None:
    for p in adapter.param_catalog():
        assert p.level in (None, "expert"), p.key              # basic/advanced follow `advanced`
        if p.level == "expert":
            assert p.advanced, p.key


def test_every_flag_exists_in_the_pinned_help(adapter: LlamaCppAdapter) -> None:
    known = help_flags()
    missing = [f"{p.key}: {p.flag}" for p in adapter.param_catalog() if not p.flag.startswith("env:") and p.flag not in known]
    assert not missing, missing


def test_env_flags_are_documented_in_the_help(adapter: LlamaCppAdapter) -> None:
    envs = {e.env for e in parse_help(read("llama-server-help.txt")) if e.env}
    for p in adapter.param_catalog():
        if p.flag.startswith("env:"):
            assert p.flag[4:] in envs, p.key
    assert by_key(adapter)["api_key"].flag == "env:LLAMA_API_KEY"


def test_negations_exist_in_the_pinned_help(adapter: LlamaCppAdapter) -> None:
    known = help_flags()
    cat = by_key(adapter)
    for key in NEGATABLE_KEYS:
        assert cat[key].type == "bool", key
        assert "--no-" + cat[key].flag[2:] in known, key
    # a bool that is NOT negatable must not have a negation we forgot about
    for p in cat.values():
        if p.type == "bool" and p.key not in NEGATABLE_KEYS:
            assert "--no-" + p.flag[2:] not in known, p.key


def _help_default(spec: ParamSpec) -> str | None:
    return entry_for(spec.flag).default if not spec.flag.startswith("env:") else None


def _same(spec: ParamSpec, literal: str) -> bool:
    low = literal.lower()
    if spec.type == "bool":
        return {"enabled": True, "prefill enabled": True, "true": True, "disabled": False, "false": False}.get(low) is spec.default
    if spec.type in ("int", "float"):
        try:
            return math.isclose(float(low), float(spec.default), rel_tol=1e-9)
        except ValueError:
            return False
    return low == str(spec.default).lower()


# The console deliberately differs from the engine here (documented in the param's help and the adapter docstring).
DEFAULT_DIFFERS = {"cors_origins"}


def test_literal_defaults_match_the_pinned_help(adapter: LlamaCppAdapter) -> None:
    """Every default the help states as a literal equals the catalog default; the rest are computed or unstated."""
    checked = 0
    for p in adapter.param_catalog():
        literal = _help_default(p)
        if literal is None or p.default is None or p.key in DEFAULT_DIFFERS:
            continue
        if not re.fullmatch(r"-?[\d.]+|enabled|prefill enabled|disabled|true|false|[a-z0-9]+", literal.lower()):
            continue                                            # "same", "read", "loaded", ... are not literals
        assert _same(p, literal), f"{p.key}: help says {literal}, catalog says {p.default}"
        checked += 1
    assert checked >= 52, checked                               # the check really compared most of the catalog


def test_defaults_that_the_help_does_not_state_as_literals(adapter: LlamaCppAdapter) -> None:
    cat = by_key(adapter)
    assert cat["split_mode"].default == "layer"                 # "- layer (default)" in the help text
    assert "layer (default)" in entry_for("--split-mode").text
    assert cat["cors_origins"].default == "localhost" and entry_for("--cors-origins").default == "*"
    assert cat["kv_unified"].default is None and "enabled if number of slots is auto" in entry_for("--kv-unified").text


def test_choices_match_the_pinned_help(adapter: LlamaCppAdapter) -> None:
    cat = by_key(adapter)
    assert cat["cache_type_k"].choices == cat["cache_type_v"].choices == CACHE_TYPES
    allowed = re.search(r"allowed values: ([^()]+?) \(default", entry_for("--cache-type-k").text)
    assert allowed and [x.strip() for x in allowed.group(1).split(",")] == CACHE_TYPES
    assert set(CACHE_TYPES) == set(CACHE_TYPE_BYTES)
    assert cat["split_mode"].choices == ["none", "layer", "row", "tensor"]
    assert "{none,layer,row,tensor}" in read("llama-server-help.txt")
    assert cat["rope_scaling"].choices == ["none", "linear", "yarn"] and "{none,linear,yarn}" in read("llama-server-help.txt")
    assert cat["pooling"].choices == ["none", "mean", "cls", "last", "rank"] and "{none,mean,cls,last,rank}" in read("llama-server-help.txt")
    for key, flag in (("flash_attn", "--flash-attn"), ("reasoning", "--reasoning"), ("fit", "--fit")):
        assert f"[{'|'.join(cat[key].choices or [])}]" in read("llama-server-help.txt"), key
        assert entry_for(flag)
    spec_line = next(ln for ln in read("llama-server-help.txt").splitlines() if ln.startswith("--spec-type"))
    assert spec_line.split()[1].split(",") == SPEC_TYPES
    for mode in LOAD_MODES:
        assert f"- {mode}:" in entry_for("--load-mode").text
    for fmt in cat["reasoning_format"].choices or []:
        assert f"- {fmt}:" in entry_for("--reasoning-format").text
    for name in CHAT_TEMPLATES:
        assert name in entry_for("--chat-template").text, name


def test_deprecated_flags_are_replaced_by_load_mode(adapter: LlamaCppAdapter) -> None:
    flags = {p.flag for p in adapter.param_catalog()}
    for old in ("--mlock", "--mmap", "--no-mmap", "--direct-io"):
        assert old not in flags
        assert entry_for(old).deprecated and "--load-mode" in entry_for(old).text
    assert "--load-mode" in flags and not entry_for("--load-mode").deprecated


def test_catalog_contains_no_deprecated_or_removed_flag(adapter: LlamaCppAdapter) -> None:
    for p in adapter.param_catalog():
        if not p.flag.startswith("env:"):
            e = entry_for(p.flag)
            assert not e.deprecated, p.flag
            assert "has been removed" not in e.text, p.flag


def test_flags_the_console_owns_or_must_never_expose_are_absent(adapter: LlamaCppAdapter) -> None:
    flags = {p.flag for p in adapter.param_catalog()}
    never = {"--host", "--port", "--metrics", "--offline", "--model", "--model-url", "--hf-repo", "--hf-file",
             "--hf-token", "--docker-repo",                                  # launch-owned or downloads
             "--tools", "--agent", "--mcp-servers-config", "--mcp-servers-json", "--ui-mcp-proxy",   # lets a prompt act
             "--props", "--models-dir", "--models-preset", "--models-max", "--models-autoload",       # server surface
             "--api-key", "--api-key-file", "--ssl-key-file", "--ssl-cert-file", "--path", "--api-prefix",
             "--log-file", "--log-prompts-dir", "--slot-save-path", "--media-path"}
    assert not flags & never
    for flag in ("--tools", "--agent", "--mcp-servers-json", "--ui-mcp-proxy"):
        assert "do not enable in untrusted environments" in entry_for(flag).text    # the reason, from the help itself


def test_secret_looking_keys_are_only_the_api_key_or_non_strings(adapter: LlamaCppAdapter) -> None:
    """The core replaces any string value under a key matching key|token|secret|password and keeps it out of argv."""
    for p in adapter.param_catalog():
        if SECRET_KEY.search(p.key):
            assert p.key == "api_key" or p.type in ("int", "float", "bool"), p.key
    assert by_key(adapter)["api_key"].flag.startswith("env:")


def test_required_params_present(adapter: LlamaCppAdapter) -> None:
    keys = set(by_key(adapter))
    needed = {"n_gpu_layers", "ctx_size", "parallel", "flash_attn", "cache_type_k", "cache_type_v", "threads",
              "batch_size", "ubatch_size", "load_mode", "override_tensor", "split_mode", "tensor_split", "main_gpu",
              "jinja", "reasoning", "reasoning_format", "rope_scaling", "rope_scale", "yarn_orig_ctx", "cont_batching",
              "kv_offload", "cpu_moe", "n_cpu_moe", "fit", "spec_type", "mmproj", "lora", "api_key", "cors_origins",
              "served_model_name", "chat_template", "chat_template_jinja", "cache_ram", "sleep_idle_seconds"}
    assert needed <= keys


def test_all_groups_used(adapter: LlamaCppAdapter) -> None:
    assert {p.group for p in adapter.param_catalog()} == {
        "memory", "parallelism", "context", "quantization", "scheduling", "tools_reasoning", "performance",
        "speculative", "network", "advanced"}


def test_memory_params_flagged(adapter: LlamaCppAdapter) -> None:
    cat = by_key(adapter)
    for k in ("n_gpu_layers", "ctx_size", "cache_type_k", "cache_type_v", "kv_offload", "parallel", "cache_ram",
              "split_mode", "tensor_split", "cpu_moe", "n_cpu_moe", "fit", "fit_target", "ubatch_size", "load_mode"):
        assert cat[k].affects_memory, k
    assert not cat["temperature"].affects_memory and not cat["jinja"].affects_memory


def test_helper_sets_reference_real_keys(adapter: LlamaCppAdapter) -> None:
    cat = by_key(adapter)
    assert {k: cat[k].type for k in LAYER_KEYS} == {"n_gpu_layers": "string", "spec_draft_ngl": "string"}
    for k in PATH_KEYS:
        assert cat[k].type == "string", k


def test_default_image_is_pinned_by_digest(adapter: LlamaCppAdapter) -> None:
    assert adapter.default_image == DEFAULT_IMAGE == f"ghcr.io/ggml-org/llama.cpp@{IMAGE_DIGEST}"
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", IMAGE_DIGEST)
    assert ":latest" not in adapter.default_image and ":server-cuda" not in adapter.default_image


def test_adapter_identity(adapter: LlamaCppAdapter) -> None:
    assert (adapter.id, adapter.display_name, adapter.default_port) == ("llamacpp", "llama.cpp", 8080)
    assert adapter.model_formats == frozenset({"gguf"}) and adapter.multi_model is False
    assert adapter.metric_prefixes == ("llamacpp:",)
    assert adapter.image_for("x86_64") == adapter.image_for("aarch64") == DEFAULT_IMAGE


def test_catalog_returns_copies(adapter: LlamaCppAdapter) -> None:
    adapter.param_catalog()[0].label = "mutated"
    assert adapter.param_catalog()[0].label != "mutated"
    adapter.presets()["balanced"]["ctx_size"] = 1
    assert adapter.presets()["balanced"]["ctx_size"] == 16384


@pytest.mark.parametrize("key,flag", [("n_gpu_layers", "--n-gpu-layers"), ("ctx_size", "--ctx-size"),
                                      ("cache_type_k", "--cache-type-k"), ("kv_offload", "--kv-offload"),
                                      ("flash_attn", "--flash-attn"), ("parallel", "--parallel"),
                                      ("temperature", "--temp"), ("served_model_name", "--alias")])
def test_aliases_resolve_to_the_primary_spelling(adapter: LlamaCppAdapter, key: str, flag: str) -> None:
    """Short and long aliases share one entry in the help; the catalog uses the long primary one."""
    assert by_key(adapter)[key].flag == flag
    assert flag in entry_for(flag).flags and len(entry_for(flag).flags) >= 2
