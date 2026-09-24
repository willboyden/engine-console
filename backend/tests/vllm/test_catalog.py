from __future__ import annotations

import re

from engine_console.adapters.vllm import VllmAdapter
from engine_console.adapters.vllm_params import DEFAULT_IMAGE, NARGS_KEYS, NEGATABLE_KEYS


def test_catalog_size_and_uniqueness(adapter: VllmAdapter) -> None:
    cat = adapter.param_catalog()
    assert len(cat) >= 60
    assert len({p.key for p in cat}) == len(cat)
    assert len({p.flag for p in cat}) == len(cat)


def test_every_param_is_well_formed(adapter: VllmAdapter) -> None:
    for p in adapter.param_catalog():
        assert re.fullmatch(r"[a-z][a-z0-9_]*", p.key), p.key
        assert re.fullmatch(r"--[a-z0-9][a-z0-9-]*|env:[A-Z0-9_]+|@image", p.flag), p.flag
        assert p.help.strip() and p.label.strip()
        assert p.docs_url and p.docs_url.startswith("https://docs.vllm.ai/en/v0.23.0/")
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


def test_required_params_present(adapter: VllmAdapter) -> None:
    keys = {p.key for p in adapter.param_catalog()}
    needed = {"gpu_memory_utilization", "max_model_len", "max_num_seqs", "max_num_batched_tokens",
              "tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size",
              "enable_expert_parallel", "quantization", "kv_cache_dtype", "dtype", "enable_prefix_caching",
              "enable_chunked_prefill", "block_size", "cpu_offload_gb", "enforce_eager", "cudagraph_capture_sizes",
              "tool_call_parser", "enable_auto_tool_choice", "reasoning_parser", "chat_template",
              "served_model_name", "speculative_config", "enable_lora", "lora_modules",
              "limit_mm_per_prompt", "hf_overrides", "seed", "api_key", "attention_backend"}
    assert needed <= keys


def test_all_groups_used(adapter: VllmAdapter) -> None:
    groups = {p.group for p in adapter.param_catalog()}
    assert groups == {"memory", "parallelism", "context", "quantization", "scheduling", "tools_reasoning",
                      "performance", "speculative", "network", "advanced"}


def test_memory_params_flagged(adapter: VllmAdapter) -> None:
    by = {p.key: p for p in adapter.param_catalog()}
    for k in ("gpu_memory_utilization", "max_model_len", "kv_cache_dtype", "tensor_parallel_size",
              "quantization", "max_num_seqs"):
        assert by[k].affects_memory, k


def test_helper_sets_reference_real_keys(adapter: VllmAdapter) -> None:
    by = {p.key: p for p in adapter.param_catalog()}
    for k in NEGATABLE_KEYS:
        assert by[k].type == "bool", k
    for k in NARGS_KEYS:
        assert by[k].type == "string_list", k


def test_default_image_pinned(adapter: VllmAdapter) -> None:
    assert adapter.default_image == DEFAULT_IMAGE == "vllm/vllm-openai:v0.23.0"
    assert not adapter.default_image.endswith(":latest")


def test_catalog_returns_copies(adapter: VllmAdapter) -> None:
    adapter.param_catalog()[0].label = "mutated"
    assert adapter.param_catalog()[0].label != "mutated"
