"""Learn mode for the SGLang adapter: every parameter and preset has a plain-language explanation (offline).

`tests/learn` enforces the general rules (limits, sorted keys, one-sentence summaries, glossary links, publish safety).
These tests cover what only this adapter knows: its own catalog and preset names, and the glossary terms it may use."""
from __future__ import annotations

import json
from pathlib import Path

from engine_console.adapters.base import Explain
from engine_console.adapters.sglang import SglangAdapter
from engine_console.learn import LEARN_DIR

# The glossary starter list the learn-content writer was given. Every term this file set uses comes from it.
STARTER_TERMS = frozenset({
    "model", "weights", "parameters", "inference_engine", "serving", "gpu", "vram", "unified_memory",
    "host_ram", "shared_memory", "page_cache", "swap", "cgroup", "container", "docker_image", "image_tag",
    "gateway", "internal_network", "port", "endpoint", "openai_compatible_api", "token", "tokenizer",
    "context_window", "prompt", "prefill", "decode", "ttft", "itl", "throughput", "latency", "batch",
    "continuous_batching", "concurrency", "kv_cache", "paged_attention", "prefix_caching",
    "chunked_prefill", "quantization", "bf16", "fp16", "fp8", "nvfp4", "mxfp4", "int8", "int4", "awq",
    "gptq", "gguf", "gguf_quant_types", "safetensors", "hugging_face_hub", "hf_cache", "repo_id",
    "revision", "gated_model", "hf_token", "moe", "active_parameters", "dense_model", "tensor_parallelism",
    "pipeline_parallelism", "expert_parallelism", "data_parallelism", "nccl", "nvlink", "pcie",
    "peer_to_peer", "cuda", "cuda_graphs", "torch_compile", "compute_capability", "blackwell",
    "attention_backend", "flash_attention", "speculative_decoding", "draft_model", "mtp", "rope_scaling",
    "yarn", "sliding_window_attention", "gqa", "mla", "lora", "tool_calling", "reasoning_model",
    "chat_template", "trust_remote_code", "served_model_name", "cpu_offload", "kv_cache_offload",
    "gpu_layers_offload", "tensor_split", "mmap", "mlock", "cpu_threads", "numa", "fit_estimate",
    "preflight", "prometheus_metrics", "sampling", "temperature", "max_tokens", "embedding_model",
    "vision_language_model", "multimodal", "benchmark", "idle_ttl", "health_check", "warmup", "oom",
})
NEW_TERMS: frozenset[str] = frozenset()      # none: no glossary term outside the starter list is needed

A = SglangAdapter()


def _section(folder: str, section: str) -> dict[str, dict[str, object]]:
    doc = json.loads(Path(LEARN_DIR, folder, "sglang.json").read_text(encoding="utf-8"))
    assert set(doc) == {"version", section}
    body: dict[str, dict[str, object]] = doc[section]
    return body


def test_every_catalog_key_has_an_explanation_and_nothing_else() -> None:
    wanted = {p.key for p in A.param_catalog()}
    have = set(_section("params", "params"))
    assert wanted - have == set(), f"unexplained parameters: {sorted(wanted - have)}"
    assert have - wanted == set(), f"explanations for parameters that no longer exist: {sorted(have - wanted)}"


def test_every_preset_has_an_explanation_and_nothing_else() -> None:
    assert set(_section("presets", "presets")) == set(A.presets())


def test_terms_come_from_the_starter_list_or_the_declared_new_terms() -> None:
    allowed = STARTER_TERMS | NEW_TERMS
    offenders = {f"{folder}:{key}": sorted(set(e["terms"]) - allowed)    # type: ignore[call-overload]
                 for folder, section in (("params", "params"), ("presets", "presets"))
                 for key, e in _section(folder, section).items() if set(e["terms"]) - allowed}  # type: ignore[call-overload]
    assert not offenders, offenders


def test_explanation_level_matches_the_catalog_level() -> None:
    have = _section("params", "params")
    for p in A.param_catalog():
        assert have[p.key]["level"] == p.level, f"{p.key}: catalog {p.level}, explanation {have[p.key]['level']}"
    assert {e["level"] for e in _section("presets", "presets").values()} == {"basic"}


def test_every_explanation_is_complete_and_parses() -> None:
    for folder, section in (("params", "params"), ("presets", "presets")):
        for key, raw in _section(folder, section).items():
            e = Explain.model_validate(raw)
            assert e.summary.endswith(".") and e.details and e.when_to_change and e.risks, f"{folder}/{key}"
    for key, raw in _section("params", "params").items():
        if raw["level"] == "basic":
            assert raw.get("example"), f"basic setting {key} needs a concrete example"


def test_beginner_settings_say_what_breaks() -> None:
    have = _section("params", "params")
    basics = [p.key for p in A.param_catalog() if p.level == "basic"]
    assert len(basics) >= 8
    for key in basics:
        assert len(str(have[key]["risks"])) > 40, key


def test_the_hardware_facts_and_the_memory_semantics_are_in_the_text() -> None:
    have = _section("params", "params")
    assert "free when it starts" in str(have["mem_fraction_static"]["details"])   # the share applies to free memory
    assert "unbalanced" in str(have["tp_size"]["details"])                        # the 10 % rule of the image
    assert "SM>=80 and SM<=90" in str(have["attention_backend"]["details"])        # fa3 is refused on sm_120
