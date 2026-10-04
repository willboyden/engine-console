"""memory_model() of the Ollama adapter and how the shared estimator consumes it."""
from __future__ import annotations

import pytest

from engine_console.adapters.base import EngineAdapter, Hardware, ModelInfo
from engine_console.adapters.ollama import OllamaAdapter
from engine_console.services.fit import estimate_fit

GIB = 1024 ** 3
KV_PER_TOKEN = 2 * 32 * 8 * 128 * 2            # K+V, 32 layers, 8 KV heads of 128, f16 -> 128 KiB per token


def test_defaults(adapter: OllamaAdapter, llama8b: ModelInfo) -> None:
    assert adapter.memory_model(llama8b, {}) == {
        "mem_fraction": 1.0, "kv_bytes_per_elem": 2.0, "overhead_gib": 1.0, "cuda_graph_gib": 0.0,
        "max_batched_tokens": 512.0, "max_seqs": 1.0, "tp": 1.0}


def test_the_kv_cache_covers_every_parallel_slot(adapter: OllamaAdapter, llama8b: ModelInfo) -> None:
    """The Ollama FAQ: "a 2K context with 4 parallel requests will result in an 8K context and additional memory"."""
    mem = adapter.memory_model(llama8b, {"context_length": 2048, "num_parallel": 4})
    assert mem["max_len"] == 8192.0 and mem["max_seqs"] == 4.0
    assert adapter.memory_model(llama8b, {"context_length": 8192})["max_len"] == 8192.0
    assert "max_len" not in adapter.memory_model(llama8b, {"num_parallel": 4})        # context unset: Ollama picks it
    assert "max_len" not in adapter.memory_model(llama8b, {"context_length": 0})


@pytest.mark.parametrize("kv,expected", [("f16", 2.0), ("q8_0", 1.0625), ("q4_0", 0.5625), (None, 2.0), ("bogus", 2.0)])
def test_kv_bytes_per_element(adapter: OllamaAdapter, llama8b: ModelInfo, kv: str | None, expected: float) -> None:
    params = {} if kv is None else {"kv_cache_type": kv}
    assert adapter.memory_model(llama8b, params)["kv_bytes_per_elem"] == expected


def test_overhead_includes_the_reserved_gpu_memory(adapter: OllamaAdapter, llama8b: ModelInfo) -> None:
    assert adapter.memory_model(llama8b, {"gpu_overhead": 2 * GIB})["overhead_gib"] == 3.0
    assert adapter.memory_model(llama8b, {"gpu_overhead": 0})["overhead_gib"] == 1.0


def test_one_gpu_unless_spreading_is_forced(adapter: OllamaAdapter, llama8b: ModelInfo) -> None:
    assert adapter.memory_model(llama8b, {})["tp"] == 1.0
    assert adapter.memory_model(llama8b, {"sched_spread": False})["tp"] == 1.0
    assert "tp" not in adapter.memory_model(llama8b, {"sched_spread": True})          # the estimator then uses every selected GPU


def test_values_are_plain_floats_the_core_can_convert(adapter: OllamaAdapter, llama8b: ModelInfo) -> None:
    for params in ({}, {"context_length": 4096, "num_parallel": 2, "kv_cache_type": "q4_0", "gpu_overhead": 1}):
        assert all(isinstance(v, float) for v in adapter.memory_model(llama8b, params).values())


def test_no_host_memory_is_modelled(adapter: OllamaAdapter, llama8b: ModelInfo, hw: Hardware) -> None:
    """The CPU share of a partly offloaded model is chosen by Ollama's scheduler at load time."""
    assert "host_memory_gib" not in EngineAdapter.__abstractmethods__
    assert adapter.host_memory_gib(llama8b, {"context_length": 262144, "num_parallel": 8}, hw) == {}


# ------------------------------------------------------------------ the shared estimator
def test_estimator_sizes_one_model_with_parallel_slots(adapter: OllamaAdapter, llama8b: ModelInfo, hw: Hardware) -> None:
    mem = adapter.memory_model(llama8b, {"context_length": 8192, "num_parallel": 4})
    gpu = estimate_fit(llama8b, mem, hw).per_gpu[0]
    assert gpu.kv_cache_gib == pytest.approx(8192 * 4 * KV_PER_TOKEN / GIB, abs=0.01) == pytest.approx(4.0, abs=0.01)
    assert gpu.weights_gib == pytest.approx(4.6, abs=0.01) and gpu.cuda_graphs_gib == 0.0 and gpu.budget_gib == pytest.approx(95.6)


def test_a_quantized_cache_shrinks_the_estimate(adapter: OllamaAdapter, llama8b: ModelInfo, hw: Hardware) -> None:
    f16 = estimate_fit(llama8b, adapter.memory_model(llama8b, {"context_length": 65536}), hw).per_gpu[0].kv_cache_gib
    q8 = estimate_fit(llama8b, adapter.memory_model(llama8b, {"context_length": 65536, "kv_cache_type": "q8_0"}), hw).per_gpu[0].kv_cache_gib
    q4 = estimate_fit(llama8b, adapter.memory_model(llama8b, {"context_length": 65536, "kv_cache_type": "q4_0"}), hw).per_gpu[0].kv_cache_gib
    assert f16 == pytest.approx(8.0, abs=0.01)
    assert q8 / f16 == pytest.approx(1.0625 / 2, abs=0.01) and q4 / f16 == pytest.approx(0.5625 / 2, abs=0.01)


def test_the_default_256k_context_does_not_fit_a_small_card(adapter: OllamaAdapter, llama8b: ModelInfo, small_gpu: Hardware) -> None:
    """Why the compat note exists: with the context set to what Ollama picks on a big card, a 16 GiB card is out."""
    report = estimate_fit(llama8b, adapter.memory_model(llama8b, {"context_length": 262144}), small_gpu)
    assert report.verdict == "wont_fit" and report.per_gpu[0].kv_cache_gib == pytest.approx(32.0, abs=0.01)
    assert estimate_fit(llama8b, adapter.memory_model(llama8b, {"context_length": 8192}), small_gpu).verdict == "fits"


def test_estimator_with_unset_context_uses_the_native_length_and_says_so(adapter: OllamaAdapter, llama8b: ModelInfo, hw: Hardware) -> None:
    report = estimate_fit(llama8b, adapter.memory_model(llama8b, {}), hw)
    assert report.max_len == 131072 and any("max_len not set" in n for n in report.notes) and report.confidence != "high"


def test_a_model_without_layer_facts_degrades_to_unknown_not_a_guess(adapter: OllamaAdapter, bare: ModelInfo, hw: Hardware) -> None:
    report = estimate_fit(bare, adapter.memory_model(bare, {"context_length": 8192}), hw)
    assert report.confidence == "low" and any("cannot be computed" in n for n in report.notes)
    assert report.verdict in ("unknown", "wont_fit")
