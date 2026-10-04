"""The SGLang catalog against what the pinned image itself reports (offline: committed fixtures, no docker)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from engine_console.adapters.sglang import SglangAdapter
from engine_console.adapters.sglang_params import (
    _SECTIONS,
    DOCS,
    ENV_DOCS,
    NOT_IN_V0_5_14,
    OMITTED_CHOICES,
    SECTION_OF,
    SPEC_ALGOS,
)

FIX = Path(__file__).parent / "fixtures"
A = SglangAdapter()


@pytest.fixture(scope="module")
def snap() -> dict[str, Any]:
    flags: dict[str, Any] = json.loads((FIX / "help_flags_v0.5.20.json").read_text())["flags"]
    return flags


def _find(snap: dict[str, Any], flag: str) -> dict[str, Any] | None:
    if flag in snap:
        return snap[flag]
    return next((d for d in snap.values() if flag in d["aliases"]), None)


def _cli() -> list[Any]:
    return [p for p in A.param_catalog() if not p.flag.startswith("env:")]


def test_every_flag_exists_in_the_pinned_image(snap: dict[str, Any]) -> None:
    missing = [f"{p.key}: {p.flag}" for p in _cli() if _find(snap, p.flag) is None]
    assert not missing, f"flags that the 0.5.20 parser does not have: {missing}"


def test_flags_in_the_other_local_image() -> None:
    other = set(json.loads((FIX / "help_flags_v0.5.14.json").read_text())["flags"])
    absent = tuple(sorted(p.flag for p in _cli() if p.flag not in other))
    assert absent == tuple(sorted(NOT_IN_V0_5_14)), f"docstring/NOT_IN_V0_5_14 disagrees with the 0.5.14 parser: {absent}"


def test_choices_are_the_help_minus_the_recorded_omissions(snap: dict[str, Any]) -> None:
    for p in _cli():
        listed = (_find(snap, p.flag) or {})["choices"]
        if p.type != "enum" or not listed:
            continue
        omitted = OMITTED_CHOICES.get(p.flag, frozenset())
        have, help_ = set(p.choices or []), set(listed)
        assert not have & omitted, p.key
        assert have | omitted == help_, f"{p.key}: catalog-only {sorted(have - help_)}, unexplained {sorted(help_ - have - omitted)}"


def test_omitted_choices_all_exist_in_the_image(snap: dict[str, Any]) -> None:
    for flag, omitted in OMITTED_CHOICES.items():
        listed = set((_find(snap, flag) or {})["choices"] or [])
        assert omitted <= listed, f"{flag}: {sorted(omitted - listed)} is not in the image's list"


def test_bool_and_list_shapes_match_the_parser(snap: dict[str, Any]) -> None:
    for p in _cli():
        h = _find(snap, p.flag) or {}
        if p.type == "bool":
            assert h["store_true"], f"{p.key}: build_launch emits only the bare flag, so the action must be store_true"
        if p.type == "string_list":
            assert h["nargs"] in ("*", "+"), p.key


def test_defaults_match_the_parser(snap: dict[str, Any]) -> None:
    intended = {"enable_metrics"}            # the console forces this one on (the parser's own default is False)
    for p in _cli():
        shown = (_find(snap, p.flag) or {})["default"]
        if p.default is None or shown is None or p.key in intended:
            continue
        if isinstance(p.default, bool):
            assert str(p.default) == shown, p.key
        elif isinstance(p.default, (int, float)):
            assert float(shown) == float(p.default), p.key
        else:
            assert f"'{p.default}'" == shown, (p.key, shown)


def test_speculative_algorithms_are_the_builtins_in_the_help() -> None:
    assert SPEC_ALGOS == ["EAGLE", "EAGLE3", "NEXTN", "STANDALONE", "NGRAM", "DFLASH", "DSPARK"]
    # `--speculative-algorithm` prints no choices; the help names EAGLE, EAGLE3, NEXTN, STANDALONE, NGRAM, DFLASH, DSPARK, UNO.


def test_docs_sections_are_complete_and_known() -> None:
    keys = {p.key for p in _cli()}
    assert set(SECTION_OF) == keys, (sorted(keys - set(SECTION_OF)), sorted(set(SECTION_OF) - keys))
    assert sum(len(v) for v in _SECTIONS.values()) == len(SECTION_OF)       # no key listed twice
    for p in _cli():
        assert p.docs_url == f"{DOCS}#{SECTION_OF[p.key]}"
    for p in A.param_catalog():
        if p.flag.startswith("env:"):
            assert p.docs_url == ENV_DOCS, p.key
    # the heading ids of the page's second-level sections (read from its HTML on 2026-10-04)
    page_sections = {"common-launch-commands", "model-and-tokenizer", "http-server", "quantization-and-data-type",
                     "memory-and-scheduling", "runtime-options", "logging", "api-related", "data-parallelism",
                     "multi-node-distributed-serving", "model-override-args", "lora",
                     "kernel-backends-attention-sampling-grammar-gemm", "speculative-decoding", "moe", "mamba-cache",
                     "hierarchical-cache", "offloading", "optimization/debug-options", "for-multi-modal",
                     "encode-prefill-disaggregation"}
    assert set(_SECTIONS) <= page_sections


def test_removed_flags_are_really_gone_from_the_image(snap: dict[str, Any]) -> None:
    """quantization marlin/qoq and attention cutlass_mla/dual_chunk_flash_attn were choices in 0.5.14, not in 0.5.20."""
    q = set(snap["--quantization"]["choices"])
    assert not {"marlin", "qoq"} & q
    assert not {"cutlass_mla", "dual_chunk_flash_attn"} & set(snap["--attention-backend"]["choices"])
    assert "--api-key" in snap and snap["--api-key"]["nargs"] is None       # a plain argument: no env-var form


def test_env_param_is_a_variable_the_image_defines() -> None:
    env = [p for p in A.param_catalog() if p.flag.startswith("env:")]
    assert [(p.key, p.flag) for p in env] == [("allow_longer_context_len",
                                              "env:SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN")]
    # read in the image: srt/environ.py defines SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN = EnvBool(False) and
    # srt/configs/model_config.py raises "... set the env var SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1" without it


def test_renamed_and_deprecated_spellings(snap: dict[str, Any]) -> None:
    """--cuda-graph-max-bs is a 0.5.14 spelling that 0.5.20 dropped; the catalog uses the -decode form both images have."""
    old = set(json.loads((FIX / "help_flags_v0.5.14.json").read_text())["flags"])
    assert "--cuda-graph-max-bs" in old and "--cuda-graph-max-bs" not in snap
    flags = {p.flag for p in A.param_catalog()}
    assert "--cuda-graph-max-bs-decode" in flags and "--cuda-graph-max-bs-decode" in old
    assert "--cuda-graph-max-bs" not in flags and "--disable-cuda-graph" not in flags
    assert "--disable-cuda-graph" in snap          # still accepted by 0.5.20, but deprecated there
