"""The vLLM catalog against what the pinned image itself reports (offline: committed fixtures, no docker)."""
from __future__ import annotations

import json
from typing import Any

from engine_console.adapters.vllm import VllmAdapter
from engine_console.adapters.vllm_params import (
    ATTENTION_BACKENDS,
    ATTENTION_KV_DTYPES,
    ATTENTION_NOT_ON_SM120,
    CLI_DOCS,
    NOT_IN_V0_30_0,
    QUANTIZATION,
    REASONING_PARSERS,
    TOOL_CALL_PARSERS,
    doc_anchor,
)

from .conftest import FIX  # type: ignore[import-not-found]

# Quantization entries of the 0.27.1 registry that are left out on purpose (AMD / Intel only).
OMITTED_QUANT = {"quark", "inc"}


def _cli(adapter: VllmAdapter) -> list[Any]:
    return [p for p in adapter.param_catalog() if not p.flag.startswith("env:")]


def test_every_flag_exists_in_the_pinned_image(adapter: VllmAdapter, help_flags: dict[str, Any]) -> None:
    missing = [f"{p.key}: {p.flag}" for p in _cli(adapter) if p.flag not in help_flags]
    assert not missing, f"flags that `vllm serve --help=all` of v0.27.1 does not list: {missing}"


def test_flags_in_the_other_local_image(adapter: VllmAdapter) -> None:
    other = set(json.loads((FIX / "help_flags_v0.30.0.json").read_text())["flags"])
    absent = tuple(sorted(p.flag for p in _cli(adapter) if p.flag not in other))
    assert absent == tuple(sorted(NOT_IN_V0_30_0)), (
        f"module docstring/NOT_IN_V0_30_0 disagrees with the v0.30.0 help: {absent}")


def test_enum_choices_equal_the_help(adapter: VllmAdapter, help_flags: dict[str, Any]) -> None:
    for p in _cli(adapter):
        listed = help_flags[p.flag]["choices"]
        if p.type != "enum" or not listed:
            continue
        assert set(p.choices or []) == {c for c in listed if c != "None"}, p.key


def test_bool_and_list_shapes_match_the_help(adapter: VllmAdapter, help_flags: dict[str, Any]) -> None:
    from engine_console.adapters.vllm_params import NARGS_KEYS, NEGATABLE_KEYS
    for p in _cli(adapter):
        h = help_flags[p.flag]
        if p.type == "bool":
            assert (p.key in NEGATABLE_KEYS) == h["negatable"], f"{p.key}: --no-<flag> form"
        assert (p.key in NARGS_KEYS) == (h["nargs"] in ("+", "*")), f"{p.key}: several values per flag"


def test_defaults_match_the_help(adapter: VllmAdapter, help_flags: dict[str, Any]) -> None:
    for p in _cli(adapter):
        shown = help_flags[p.flag]["default"]
        if p.default is None or shown is None or p.key == "mamba_backend":    # that one prints MambaBackendEnum.TRITON
            continue
        if isinstance(p.default, bool):
            assert str(p.default) == shown, p.key
        elif isinstance(p.default, (int, float)):
            assert float(shown) == float(p.default), p.key
        else:
            assert str(p.default) == shown, p.key


def test_docs_anchors_follow_the_flag_table(adapter: VllmAdapter, help_flags: dict[str, Any]) -> None:
    for p in _cli(adapter):
        if not p.docs_url or not p.docs_url.startswith(CLI_DOCS):
            continue                                       # a feature page (tool calling, spec decode)
        h = help_flags[p.flag]
        want = "-" + p.flag[2:] + (f"-no-{p.flag[2:]}" if h["negatable"] else "")
        shorts = [a for a in h["aliases"] if not a.startswith("--")]
        if shorts:
            want += "-" + shorts[0].lstrip("-")
        assert p.docs_url == f"{CLI_DOCS}#{want}", p.key
        assert doc_anchor(p.flag, h["negatable"]) == want


def test_registry_lists_match_the_image(registries: dict[str, Any]) -> None:
    assert set(QUANTIZATION) == set(registries["quantization_methods"]) - OMITTED_QUANT
    assert "gguf" not in QUANTIZATION and "gguf" not in registries["quantization_methods"]
    assert registries["reasoning_parsers"] == REASONING_PARSERS
    assert "qwen3" in REASONING_PARSERS and "nemotron_v3" in REASONING_PARSERS and "openai_gptoss" in REASONING_PARSERS


def test_attention_facts_match_the_image(registries: dict[str, Any]) -> None:
    att = registries["attention_backends"]
    named = [b for b in ATTENTION_BACKENDS if b != "auto"]
    assert all(b in att for b in named), [b for b in named if b not in att]
    for b in named:
        assert att[b]["supports_cc_12_0"] == (b not in ATTENTION_NOT_ON_SM120), b
    for b, dtypes in ATTENTION_KV_DTYPES.items():
        assert set(att[b]["kv_cache_dtypes"]) == set(dtypes), b
        assert att[b]["supports_cc_12_0"], f"{b} is in the sm_120 table but the image says it is not supported"


def test_env_params_use_variables_the_image_reads(adapter: VllmAdapter, registries: dict[str, Any]) -> None:
    names = {p.flag[4:] for p in adapter.param_catalog() if p.flag.startswith("env:")}
    assert names == {"VLLM_API_KEY", "VLLM_ALLOW_LONG_MAX_MODEL_LEN", "VLLM_LOGGING_LEVEL",
                     "VLLM_WORKER_MULTIPROC_METHOD"}
    assert names <= set(registries["env_present"])
    assert not names & set(registries["env_absent"])


def test_parser_choices_cover_the_dialect_tables(adapter: VllmAdapter) -> None:
    from engine_console.adapters import vllm as mod
    for _, allowed in mod._TOOL_DIALECTS:
        assert set(allowed) <= set(TOOL_CALL_PARSERS), allowed
    for _, parser in mod._REASONING_HINTS:
        assert parser in REASONING_PARSERS, parser
    for name in adapter.presets().values():
        for key, val in name.items():
            if key == "tool_call_parser":
                assert val in TOOL_CALL_PARSERS
            if key == "reasoning_parser":
                assert val in REASONING_PARSERS


def test_aliased_parsers_are_one_class_in_the_image(registries: dict[str, Any]) -> None:
    classes = registries["tool_parser_classes"]
    assert classes["qwen3_coder"] == classes["qwen3_xml"]          # either name works for Qwen3.x XML tool calls
    assert classes["glm45"] == classes["glm47"]
