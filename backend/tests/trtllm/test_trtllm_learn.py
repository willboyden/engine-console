"""Learn-mode content for the TensorRT-LLM adapter: every parameter and preset is explained, in the house style.

The strict gate lives in tests/learn; this module keeps the engine's own files honest on their own (no dependency
on the glossary being complete) and pins the glossary words the explanations may rely on."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from engine_console.adapters.base import Explain
from engine_console.adapters.trtllm import TrtllmAdapter
from engine_console.learn import LEARN_DIR

# The starter glossary the explanations were written against (every term is also a key of learn/glossary.json).
STARTER_TERMS = frozenset({
    "model", "weights", "parameters", "inference_engine", "serving", "gpu", "vram", "unified_memory",
    "host_ram", "shared_memory", "page_cache", "swap", "cgroup", "container", "docker_image", "image_tag",
    "gateway", "internal_network", "port", "endpoint", "openai_compatible_api", "token", "tokenizer",
    "context_window", "prompt", "prefill", "decode", "ttft", "itl", "throughput", "latency", "batch",
    "continuous_batching", "concurrency", "kv_cache", "paged_attention", "prefix_caching", "chunked_prefill",
    "quantization", "bf16", "fp16", "fp8", "nvfp4", "mxfp4", "int8", "int4", "awq", "gptq", "gguf",
    "gguf_quant_types", "safetensors", "hugging_face_hub", "hf_cache", "repo_id", "revision", "gated_model",
    "hf_token", "moe", "active_parameters", "dense_model", "tensor_parallelism", "pipeline_parallelism",
    "expert_parallelism", "data_parallelism", "nccl", "nvlink", "pcie", "peer_to_peer", "cuda", "cuda_graphs",
    "torch_compile", "compute_capability", "blackwell", "attention_backend", "flash_attention",
    "speculative_decoding", "draft_model", "mtp", "rope_scaling", "yarn", "sliding_window_attention", "gqa",
    "mla", "lora", "tool_calling", "reasoning_model", "chat_template", "trust_remote_code",
    "served_model_name", "cpu_offload", "kv_cache_offload", "gpu_layers_offload", "tensor_split", "mmap",
    "mlock", "cpu_threads", "numa", "fit_estimate", "preflight", "prometheus_metrics", "sampling",
    "temperature", "max_tokens", "embedding_model", "vision_language_model", "multimodal", "benchmark",
    "idle_ttl", "health_check", "warmup", "oom",
})
NEW_TERMS: dict[str, str] = {}      # none: the explanations use starter terms only
LIMITS = {"summary": 160, "details": 900, "when_to_change": 400, "risks": 400, "example": 200}
SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9`'\"(])")


def load(section: str, name: str) -> dict[str, dict[str, Any]]:
    path = Path(LEARN_DIR) / section / f"{name}.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert set(doc) == {"version", section} and doc["version"] == 1
    body = doc[section]
    assert list(body) == sorted(body), f"{path.name}: keys must be sorted"
    assert isinstance(body, dict)
    return body


@pytest.fixture(scope="module")
def params() -> dict[str, dict[str, Any]]:
    return load("params", "trtllm")


@pytest.fixture(scope="module")
def presets() -> dict[str, dict[str, Any]]:
    return load("presets", "trtllm")


def test_every_parameter_has_an_explanation(adapter: TrtllmAdapter, params: dict[str, dict[str, Any]]) -> None:
    keys = [p.key for p in adapter.param_catalog()]
    assert sorted(params) == sorted(keys), (
        f"missing {sorted(set(keys) - set(params))}, stale {sorted(set(params) - set(keys))}")


def test_every_preset_has_an_explanation(adapter: TrtllmAdapter, presets: dict[str, dict[str, Any]]) -> None:
    assert sorted(presets) == sorted(adapter.presets())


@pytest.mark.parametrize("section", ["params", "presets"])
def test_entries_follow_the_style_guide(section: str, params: dict[str, dict[str, Any]],
                                        presets: dict[str, dict[str, Any]]) -> None:
    body = params if section == "params" else presets
    for key, entry in body.items():
        explain = Explain(**entry)                                    # no unknown fields, valid level
        for field, limit in LIMITS.items():
            text = getattr(explain, field)
            assert text.strip(), f"{key}.{field} is empty"
            assert len(text) <= limit, f"{key}.{field} is {len(text)} chars (limit {limit})"
            assert text == text.strip() and "  " not in text and "\n" not in text, f"{key}.{field}: whitespace"
        assert explain.summary.endswith("."), key
        assert not SENTENCE_BREAK.search(explain.summary), f"{key}: the summary must be one sentence"
        assert explain.level in ("basic", "advanced", "expert")
        assert explain.terms == sorted(set(explain.terms), key=explain.terms.index), f"{key}: duplicate terms"


def test_terms_are_starter_terms_or_declared_new(params: dict[str, dict[str, Any]],
                                                 presets: dict[str, dict[str, Any]]) -> None:
    known = STARTER_TERMS | set(NEW_TERMS)
    for body in (params, presets):
        for key, entry in body.items():
            unknown = set(entry["terms"]) - known
            assert not unknown, f"{key}: terms outside the starter list and NEW_TERMS: {sorted(unknown)}"


def test_terms_exist_in_the_glossary(params: dict[str, dict[str, Any]], presets: dict[str, dict[str, Any]]) -> None:
    path = Path(LEARN_DIR) / "glossary.json"
    if not path.is_file():
        pytest.skip("glossary.json not written yet")
    glossary = set(json.loads(path.read_text(encoding="utf-8"))["terms"])
    for body in (params, presets):
        for key, entry in body.items():
            assert set(entry["terms"]) <= glossary, f"{key}: {sorted(set(entry['terms']) - glossary)}"


def test_levels_agree_with_the_catalog(adapter: TrtllmAdapter, params: dict[str, dict[str, Any]]) -> None:
    """The learn level is the catalog level (explicit, or advanced for parameters hidden behind 'Show advanced',
    basic for the rest), so the form and the popover never disagree."""
    for p in adapter.param_catalog():
        expected = p.level or ("advanced" if p.advanced else "basic")
        assert params[p.key]["level"] == expected, p.key


def test_text_is_publish_safe(params: dict[str, dict[str, Any]], presets: dict[str, dict[str, Any]]) -> None:
    blob = json.dumps([params, presets])
    assert not re.search(r"/(?:home|Users)/|@[A-Za-z0-9-]+\.[a-z]{2,}", blob)
    assert not re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", blob)                # no IP addresses
    assert not re.search(r"hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}", blob)


def test_explanations_state_facts_that_the_catalog_enforces(params: dict[str, dict[str, Any]]) -> None:
    """A few claims that would rot silently if the code changed; the tests above pin the catalog side."""
    assert "0.9" in params["free_gpu_memory_fraction"]["details"]
    assert "2048" in params["max_batch_size"]["details"] and "8192" in params["max_num_tokens"]["details"]
    assert "32" in params["tokens_per_block"]["details"]
    assert "ignored" in params["moe_cluster_parallel_size"]["details"]          # the CLI flag serve() drops
    for name in ("qwen3_coder", "kimi_k2", "deepseek_v32"):
        assert name in params["tool_parser"]["details"]
    for name in ("deepseek-r1", "qwen3", "nano-v3"):
        assert name in params["reasoning_parser"]["details"]
