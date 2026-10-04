"""compatibility() of the Ollama adapter: every rule, its level, and the fix it names."""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import engine_console.adapters.ollama as module
from engine_console.adapters.base import Compat, Hardware, ModelInfo
from engine_console.adapters.ollama import OllamaAdapter, default_context

# every code the adapter can emit (the learn gate needs a `check.<code>` concept for each; the report lists them)
EXPECTED_CODES = {"blackwell_cuda_build", "default_context_by_vram", "kv_exceeds_vram", "parallel_multiplies_context",
                  "kv_quant_needs_flash_attn", "keep_alive_forever", "cpu_offload_slow"}


def codes(adapter: OllamaAdapter, m: ModelInfo, p: dict[str, Any], hw: Hardware) -> dict[str, Compat]:
    found = adapter.compatibility(m, p, hw)
    assert len({c.code for c in found}) == len(found), "a code was emitted twice"
    return {c.code: c for c in found}


def test_the_emitted_codes_are_exactly_the_documented_ones() -> None:
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    emitted = {n.args[1].value for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "add"
               and len(n.args) >= 2 and isinstance(n.args[1], ast.Constant) and isinstance(n.args[1].value, str)}
    assert emitted == EXPECTED_CODES


def test_default_context_tiers_follow_the_docs() -> None:
    """docs.ollama.com/context-length: below 24 GiB 4k, 24-48 GiB 32k, from 48 GiB 256k."""
    assert [default_context(g) for g in (0, 8, 16, 23.9)] == [4096] * 4
    assert [default_context(g) for g in (24, 32, 47.9)] == [32768] * 3
    assert [default_context(g) for g in (48, 95.6, 191.2)] == [262144] * 3


def test_a_large_gpu_gets_a_large_default_context_and_a_warning(adapter: OllamaAdapter, hw: Hardware, llama8b: ModelInfo) -> None:
    c = codes(adapter, llama8b, {}, hw)["default_context_by_vram"]
    assert c.level == "warn" and "262144 tokens" in c.message and "96 GiB" in c.message
    assert "32.0 GiB of KV cache per loaded model" in c.message              # 262144 x 128 KiB
    assert "context_length" in c.message
    parallel = codes(adapter, llama8b, {"num_parallel": 4}, hw)["default_context_by_vram"]
    assert "128.0 GiB" in parallel.message                                    # the product with the parallel requests
    quant = codes(adapter, llama8b, {"kv_cache_type": "q8_0"}, hw)["default_context_by_vram"]
    assert "17.0 GiB" in quant.message


def test_the_default_context_note_depends_on_the_tier(adapter: OllamaAdapter, hw: Hardware, small_gpu: Hardware, llama8b: ModelInfo) -> None:
    assert "default_context_by_vram" not in codes(adapter, llama8b, {}, small_gpu)                      # 16 GiB: 4k, harmless
    mid = small_gpu.model_copy(update={"gpu_total_gib": [32.0], "gpu_free_gib": [30.0]})
    assert "32768 tokens" in codes(adapter, llama8b, {}, mid)["default_context_by_vram"].message
    assert "default_context_by_vram" not in codes(adapter, llama8b, {"context_length": 8192}, hw)       # chosen explicitly
    no_gpu = hw.model_copy(update={"gpu_ids": [], "gpu_uuids": [], "gpu_names": [], "gpu_total_gib": [], "gpu_free_gib": []})
    assert "default_context_by_vram" not in codes(adapter, llama8b, {}, no_gpu)


def test_the_cost_is_left_out_when_the_model_facts_are_missing(adapter: OllamaAdapter, hw: Hardware, bare: ModelInfo) -> None:
    c = codes(adapter, bare, {}, hw)["default_context_by_vram"]
    assert "262144 tokens" in c.message and "GiB of KV cache" not in c.message


def test_two_gpus_add_up_for_the_tier(adapter: OllamaAdapter, hw2: Hardware, llama8b: ModelInfo) -> None:
    assert "191 GiB" in codes(adapter, llama8b, {}, hw2)["default_context_by_vram"].message


def test_an_explicit_context_that_will_not_fit_is_flagged(adapter: OllamaAdapter, small_gpu: Hardware, llama8b: ModelInfo) -> None:
    c = codes(adapter, llama8b, {"context_length": 131072}, small_gpu)["kv_exceeds_vram"]
    assert c.level == "warn" and "16.0 GiB of KV cache" in c.message and "14.0 GiB" in c.message and "q8_0" in c.message
    assert "kv_exceeds_vram" not in codes(adapter, llama8b, {"context_length": 131072, "kv_cache_type": "q8_0"}, small_gpu)
    assert "kv_exceeds_vram" not in codes(adapter, llama8b, {"context_length": 8192}, small_gpu)
    parallel = codes(adapter, llama8b, {"context_length": 32768, "num_parallel": 4}, small_gpu)["kv_exceeds_vram"]
    assert "x 4 request(s)" in parallel.message                                # 32768 x 4 = 128k tokens = 16 GiB


def test_parallel_requests_multiply_the_context(adapter: OllamaAdapter, hw: Hardware, llama8b: ModelInfo) -> None:
    c = codes(adapter, llama8b, {"context_length": 8192, "num_parallel": 4}, hw)["parallel_multiplies_context"]
    assert c.level == "ok" and "num_parallel=4" in c.message
    assert "parallel_multiplies_context" not in codes(adapter, llama8b, {"context_length": 8192}, hw)
    assert "parallel_multiplies_context" not in codes(adapter, llama8b, {"num_parallel": 1}, hw)


def test_quantized_cache_with_flash_attention_off(adapter: OllamaAdapter, hw: Hardware, llama8b: ModelInfo) -> None:
    for t in ("q8_0", "q4_0"):
        c = codes(adapter, llama8b, {"kv_cache_type": t, "flash_attention": False, "context_length": 8192}, hw)["kv_quant_needs_flash_attn"]
        assert c.level == "warn" and t in c.message and "flash_attention" in c.message
    for params in ({"kv_cache_type": "q8_0"}, {"kv_cache_type": "q8_0", "flash_attention": True},
                   {"kv_cache_type": "f16", "flash_attention": False}, {"flash_attention": False}):
        assert "kv_quant_needs_flash_attn" not in codes(adapter, llama8b, params | {"context_length": 8192}, hw), params


def test_keep_alive_forever(adapter: OllamaAdapter, hw: Hardware, llama8b: ModelInfo) -> None:
    for v in ("-1", "-1m", -1, " -5 "):
        c = codes(adapter, llama8b, {"keep_alive": v, "context_length": 8192}, hw)["keep_alive_forever"]
        assert c.level == "warn" and "other engines" in c.message
    for v in ("5m", "0", "24h", 3600):
        assert "keep_alive_forever" not in codes(adapter, llama8b, {"keep_alive": v, "context_length": 8192}, hw), v
    assert "keep_alive_forever" not in codes(adapter, llama8b, {"context_length": 8192}, hw)


def test_a_cpu_library_is_slow(adapter: OllamaAdapter, hw: Hardware, llama8b: ModelInfo) -> None:
    assert codes(adapter, llama8b, {"llm_library": "cpu", "context_length": 8192}, hw)["cpu_offload_slow"].level == "warn"
    assert "cpu_offload_slow" not in codes(adapter, llama8b, {"llm_library": "cuda_v13", "context_length": 8192}, hw)


def test_blackwell_note_only_on_compute_capability_12(adapter: OllamaAdapter, hw: Hardware, small_gpu: Hardware, llama8b: ModelInfo) -> None:
    c = codes(adapter, llama8b, {"context_length": 8192}, hw)["blackwell_cuda_build"]
    assert c.level == "ok" and "sm_120a" in c.message and "CUDA 12 and 13" in c.message
    assert "blackwell_cuda_build" not in codes(adapter, llama8b, {"context_length": 8192}, small_gpu)


def test_nothing_is_blocked_and_a_clean_launch_has_only_a_note(adapter: OllamaAdapter, hw: Hardware, llama8b: ModelInfo) -> None:
    found = adapter.compatibility(llama8b, {"context_length": 8192, "keep_alive": "30m"}, hw)
    assert [c.code for c in found] == ["blackwell_cuda_build"]
    worst = adapter.compatibility(llama8b, {"kv_cache_type": "q4_0", "flash_attention": False, "keep_alive": "-1",
                                            "num_parallel": 8, "llm_library": "cpu"}, hw)
    assert all(c.level in ("ok", "warn") for c in worst)                        # advice, never a hard stop
    assert all(c.message and c.code for c in worst)


def test_invalid_kv_type_does_not_crash_the_rules(adapter: OllamaAdapter, hw: Hardware, llama8b: ModelInfo) -> None:
    """compatibility() may be asked about params that were not validated first."""
    assert adapter.compatibility(llama8b, {"kv_cache_type": "bogus", "context_length": 8192}, hw)
