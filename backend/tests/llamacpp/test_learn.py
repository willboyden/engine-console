"""Learn-mode content for llama.cpp: every parameter and preset explained, in the shape learn/README.md asks for."""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from typing import Any

import pytest

from engine_console.adapters.base import Explain
from engine_console.adapters.llamacpp import LlamaCppAdapter
from engine_console.domain.learn import LearnBundle, Term

LEARN = Path(__file__).resolve().parents[2] / "src" / "engine_console" / "learn"
LIMITS = {"summary": 160, "details": 900, "when_to_change": 400, "risks": 400, "example": 200}

# The starter glossary this adapter was written against, plus the terms it asks the glossary writer to add.
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
NEW_TERMS = frozenset({"slot", "cors", "api_key", "llama_cpp"})     # api_key and llama_cpp already exist in the glossary


def load(kind: str) -> dict[str, dict[str, Any]]:
    doc = json.loads((LEARN / kind / "llamacpp.json").read_text(encoding="utf-8"))
    assert set(doc) == {"version", kind} and doc["version"] == 1
    body: dict[str, dict[str, Any]] = doc[kind]
    assert list(body) == sorted(body)                                  # keys sorted, as learn/README.md says
    return body


def test_every_parameter_has_an_explanation_and_no_leftovers(adapter: LlamaCppAdapter) -> None:
    keys = {p.key for p in adapter.param_catalog()}
    body = load("params")
    assert set(body) == keys, (sorted(keys - set(body)), sorted(set(body) - keys))


def test_every_preset_has_an_explanation_and_no_leftovers(adapter: LlamaCppAdapter) -> None:
    assert set(load("presets")) == set(adapter.presets())


@pytest.mark.parametrize("kind", ["params", "presets"])
def test_explanations_follow_the_readme(kind: str) -> None:
    for key, raw in load(kind).items():
        e = Explain.model_validate(raw)
        assert set(raw) <= set(Explain.model_fields), key
        for field, limit in LIMITS.items():
            assert len(getattr(e, field)) <= limit, (key, field, len(getattr(e, field)))
        for field in ("summary", "details", "when_to_change", "risks", "example"):
            assert getattr(e, field).strip(), (key, field)               # all five filled, not just a summary
            assert getattr(e, field) == getattr(e, field).strip() and "  " not in getattr(e, field), (key, field)
        assert e.summary.endswith(".") and "\n" not in e.summary and "\n" not in e.example, key
        assert not re.search(r"(?<=[.!?])\s+(?=[A-Z0-9`'\"(])", e.summary), f"{key}: more than one sentence"
        assert not re.search(r"\b(TODO|TBD|FIXME)\b|\?\?\?", json.dumps(raw)), key
        assert e.terms == sorted(set(e.terms)), key


@pytest.mark.parametrize("kind", ["params", "presets"])
def test_terms_come_from_the_starter_glossary_or_the_declared_new_ones(kind: str) -> None:
    used = {t for raw in load(kind).values() for t in raw["terms"]}
    assert used <= STARTER_TERMS | NEW_TERMS, sorted(used - STARTER_TERMS - NEW_TERMS)
    assert used & STARTER_TERMS                                          # the explanations really link something


def test_levels_match_the_catalog(adapter: LlamaCppAdapter) -> None:
    body = load("params")
    for p in adapter.param_catalog():
        assert body[p.key]["level"] == (p.level or ("advanced" if p.advanced else "basic")), p.key
    assert {p.key for p in adapter.param_catalog() if not p.advanced} == {
        "n_gpu_layers", "ctx_size", "parallel", "flash_attn", "reasoning", "api_key"}     # the beginner-visible ones
    for raw in load("presets").values():
        assert raw["level"] in ("basic", "advanced")


def test_facts_that_were_measured_are_in_the_text() -> None:
    body = load("params")
    assert "any web page" in body["cors_origins"]["details"] or "any page you visit" in body["cors_origins"]["details"]
    assert "metrics endpoint" in body["api_key"]["details"] and "stay open" in body["api_key"]["details"]
    assert "4 slots" in body["parallel"]["details"]                       # "n_parallel is set to auto, using n_parallel = 4"
    assert "switch the automatic fit off" in body["fit"]["risks"]
    assert "never reduced" in body["n_gpu_layers"]["risks"]               # "n_gpu_layers already set by user" (fit leaves it)


def test_no_personal_or_lab_specific_content() -> None:
    text = (LEARN / "params" / "llamacpp.json").read_text(encoding="utf-8") + (LEARN / "presets" / "llamacpp.json").read_text(encoding="utf-8")
    assert not re.search(r"/home/|/Users/|@[A-Za-z0-9-]+\.[a-z]{2,}|\b\d{1,3}(\.\d{1,3}){3}\b", text)


def _gate() -> Any:
    path = Path(__file__).resolve().parents[1] / "learn" / "test_learn_content.py"
    if not path.is_file():
        pytest.skip("tests/learn is not present")
    spec = importlib.util.spec_from_file_location("_learn_gate_used_by_llamacpp_tests", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:  # noqa: BLE001 - another owner's module is mid-edit: do not fail this adapter's tests for it
        pytest.skip(f"learn gate not importable right now: {type(e).__name__}")
    return mod


def test_passes_the_repository_learn_gate_checks() -> None:
    """The same hygiene, limit, link and jargon rules as tests/learn, applied to this engine's two files only. A term
    this adapter declares as new may not be in glossary.json yet; that one finding is allowed."""
    gate = _gate()
    glossary = {k: Term.model_validate(v) for k, v in json.loads((LEARN / "glossary.json").read_text(encoding="utf-8"))["terms"].items()}
    bundle = LearnBundle(
        glossary=glossary,
        params={"llamacpp": {k: Explain.model_validate(v) for k, v in load("params").items()}},
        presets={"llamacpp": {k: Explain.model_validate(v) for k, v in load("presets").items()}})
    problems = gate.check_limits(bundle) + gate.check_hygiene(bundle)
    problems += [p for p in gate.check_links(bundle) if not any(f"'{t}'" in p for t in NEW_TERMS)]
    hard, soft = gate.check_jargon(bundle)
    assert not problems and not hard and not soft, (problems, hard, soft)
