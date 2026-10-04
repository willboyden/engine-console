"""memory_model() and host_memory_gib() of the llama.cpp adapter, and how the shared estimator consumes them."""
from __future__ import annotations

import math

import pytest

from engine_console.adapters.base import EngineAdapter, Hardware, ModelInfo
from engine_console.adapters.llamacpp import LlamaCppAdapter
from engine_console.services.fit import estimate_fit

GIB = 1024 ** 3
KV_PER_TOKEN = 2 * 64 * 8 * 128 * 2            # K+V, 64 layers, 8 KV heads of 128, f16 -> 256 KiB per token


def test_defaults(adapter: LlamaCppAdapter, dense: ModelInfo) -> None:
    mem = adapter.memory_model(dense, {})
    assert mem == {"mem_fraction": 1.0, "kv_bytes_per_elem": 2.0, "overhead_gib": 1.0, "cuda_graph_gib": 0.0,
                   "max_batched_tokens": 512.0}


def test_context_is_the_total_and_slots_are_informational(adapter: LlamaCppAdapter, dense: ModelInfo) -> None:
    mem = adapter.memory_model(dense, {"ctx_size": 32768, "parallel": 4, "ubatch_size": 1024})
    assert mem["max_len"] == 32768.0 and mem["max_seqs"] == 4.0 and mem["max_batched_tokens"] == 1024.0
    assert "max_len" not in adapter.memory_model(dense, {"ctx_size": 0})
    assert "max_seqs" not in adapter.memory_model(dense, {"parallel": -1})


@pytest.mark.parametrize("k,v,expected", [
    ("f16", "f16", 2.0), ("q8_0", "q8_0", 1.0625), ("q4_0", "q4_0", 0.5625), ("q8_0", "f16", 1.53125),
    ("f32", "f16", 3.0), ("bf16", "bf16", 2.0), ("q5_1", "q5_0", 0.71875), ("iq4_nl", "q4_1", 0.59375)])
def test_kv_bytes_per_element_is_the_mean_of_k_and_v(adapter: LlamaCppAdapter, dense: ModelInfo, k: str, v: str, expected: float) -> None:
    assert adapter.memory_model(dense, {"cache_type_k": k, "cache_type_v": v})["kv_bytes_per_elem"] == pytest.approx(expected)


def test_overhead_follows_the_fit_margin(adapter: LlamaCppAdapter, dense: ModelInfo) -> None:
    assert adapter.memory_model(dense, {"fit_target": 2048})["overhead_gib"] == 2.0
    assert adapter.memory_model(dense, {"fit_target": 0})["overhead_gib"] == 1.0
    assert adapter.memory_model(dense, {"fit_target": 4096, "fit": "off"})["overhead_gib"] == 1.0
    assert adapter.memory_model(dense, {"fit": "on"})["overhead_gib"] == 1.0


@pytest.mark.parametrize("params,tp", [
    ({"split_mode": "none"}, 1.0), ({"tensor_split": "3,1"}, 2.0), ({"tensor_split": "1,0,0"}, 1.0),
    ({"tensor_split": "0.5,0.5,0"}, 2.0), ({"split_mode": "none", "tensor_split": "3,1"}, 1.0)])
def test_tp_counts_the_gpus_the_model_is_spread_over(adapter: LlamaCppAdapter, dense: ModelInfo, params: dict, tp: float) -> None:  # type: ignore[type-arg]
    assert adapter.memory_model(dense, params)["tp"] == tp
    assert "tp" not in adapter.memory_model(dense, {"split_mode": "layer"})        # the estimator then uses the selected GPUs


def test_partial_offload_is_reported_as_gpu_fractions(adapter: LlamaCppAdapter, dense: ModelInfo) -> None:
    mem = adapter.memory_model(dense, {"n_gpu_layers": 20})
    assert mem["weights_gpu_fraction"] == pytest.approx(20 / 65) and mem["kv_gpu_fraction"] == pytest.approx(20 / 64)
    assert mem["weights_gpu_fraction"] * 18.6 == pytest.approx(5.72, abs=0.01)         # GiB of weights that stay on the GPU
    full = adapter.memory_model(dense, {"n_gpu_layers": 65})                      # 64 layers + the output layer: all of it
    assert "weights_gpu_fraction" not in full and "kv_gpu_fraction" not in full
    for params in ({}, {"n_gpu_layers": "all"}, {"n_gpu_layers": "auto"}, {"n_gpu_layers": 999}):
        out = adapter.memory_model(dense, params)
        assert "weights_gpu_fraction" not in out and "kv_gpu_fraction" not in out, params
    nothing = adapter.memory_model(dense, {"n_gpu_layers": 0})
    assert nothing["weights_gpu_fraction"] == 0.0 and nothing["kv_gpu_fraction"] == 0.0
    cpu = adapter.memory_model(dense, {"device": "none"})
    assert cpu["weights_gpu_fraction"] == 0.0 and cpu["kv_gpu_fraction"] == 0.0
    kv_host = adapter.memory_model(dense, {"kv_offload": False})
    assert "weights_gpu_fraction" not in kv_host and kv_host["kv_gpu_fraction"] == 0.0
    assert "weights_gpu_fraction" not in adapter.memory_model(ModelInfo(repo_id="x"), {"n_gpu_layers": 20})   # layers unknown


# ------------------------------------------------------------------ host RAM
def test_host_memory_defaults_to_the_prompt_cache_ceiling(adapter: LlamaCppAdapter, dense: ModelInfo, hw: Hardware) -> None:
    assert adapter.host_memory_gib(dense, {}, hw) == {"prompt_cache": 8.0}
    assert adapter.host_memory_gib(dense, {"cache_ram": 2048}, hw) == {"prompt_cache": 2.0}
    assert adapter.host_memory_gib(dense, {"cache_ram": 0}, hw) == {}
    assert math.isnan(adapter.host_memory_gib(dense, {"cache_ram": -1}, hw)["prompt_cache"])


def test_layers_left_on_the_cpu_count_as_host_ram(adapter: LlamaCppAdapter, dense: ModelInfo, hw: Hardware) -> None:
    out = adapter.host_memory_gib(dense, {"n_gpu_layers": 20, "cache_ram": 0}, hw)
    assert out["cpu_resident_weights"] == pytest.approx(18.6 * (1 - 20 / 65))
    assert out["kv_cache_on_cpu"] != out["kv_cache_on_cpu"]                      # context unset: needed but unknowable (NaN)
    with_ctx = adapter.host_memory_gib(dense, {"n_gpu_layers": 20, "ctx_size": 32768, "cache_ram": 0}, hw)
    assert with_ctx["kv_cache_on_cpu"] == pytest.approx(32768 * KV_PER_TOKEN * (1 - 20 / 64) / GIB)
    cpu_only = adapter.host_memory_gib(dense, {"device": "none", "ctx_size": 4096, "cache_ram": 0}, hw)
    assert cpu_only["cpu_resident_weights"] == pytest.approx(18.6) and cpu_only["kv_cache_on_cpu"] == pytest.approx(1.0)
    unknown = adapter.host_memory_gib(ModelInfo(repo_id="x"), {"n_gpu_layers": 0, "cache_ram": 0}, hw)
    assert math.isnan(unknown["cpu_resident_weights"])                           # file size unknown


def test_kv_on_the_host_when_offload_is_off(adapter: LlamaCppAdapter, dense: ModelInfo, hw: Hardware) -> None:
    out = adapter.host_memory_gib(dense, {"kv_offload": False, "ctx_size": 16384, "cache_ram": 0}, hw)
    assert out == {"kv_cache_on_cpu": pytest.approx(16384 * KV_PER_TOKEN / GIB)}
    quantized = adapter.host_memory_gib(dense, {"kv_offload": False, "ctx_size": 16384, "cache_ram": 0,
                                                "cache_type_k": "q8_0", "cache_type_v": "q8_0"}, hw)
    assert quantized["kv_cache_on_cpu"] == pytest.approx(16384 * KV_PER_TOKEN / GIB * 1.0625 / 2)


def test_moe_experts_on_the_cpu(adapter: LlamaCppAdapter, moe: ModelInfo, hw: Hardware) -> None:
    share = (30.5e9 - 3.3e9) / (30.5e9 * (1 - 8 / 128))                        # expert share of the weight bytes
    out = adapter.host_memory_gib(moe, {"cpu_moe": True, "cache_ram": 0}, hw)
    assert out == {"moe_expert_weights_on_cpu": pytest.approx(17.5 * share)}
    assert 0.9 < share < 1.0
    half = adapter.host_memory_gib(moe, {"n_cpu_moe": 24, "cache_ram": 0}, hw)
    assert half["moe_expert_weights_on_cpu"] == pytest.approx(17.5 * share * 0.5)
    exps = adapter.host_memory_gib(moe, {"override_tensor": "exps=CPU", "cache_ram": 0}, hw)
    assert exps["moe_expert_weights_on_cpu"] == pytest.approx(17.5 * share)
    other = adapter.host_memory_gib(moe, {"override_tensor": r"blk\.0\.=CPU", "cache_ram": 0}, hw)
    assert math.isnan(other["moe_expert_weights_on_cpu"])                       # which tensors? unknowable
    # layers that are on the CPU anyway are counted once, in cpu_resident_weights
    both = adapter.host_memory_gib(moe, {"cpu_moe": True, "n_gpu_layers": 24, "cache_ram": 0}, hw)
    w_frac = 24 / 49
    assert both["cpu_resident_weights"] == pytest.approx(17.5 * (1 - w_frac))
    assert both["moe_expert_weights_on_cpu"] == pytest.approx(17.5 * share * w_frac)


def test_moe_without_enough_facts_is_unknown_not_guessed(adapter: LlamaCppAdapter, moe: ModelInfo, hw: Hardware) -> None:
    for update in ({"num_active_params": None}, {"num_params": None}, {"raw_config": {}}, {"weight_bytes": None}):
        thin = moe.model_copy(update=update)
        assert math.isnan(adapter.host_memory_gib(thin, {"cpu_moe": True, "cache_ram": 0}, hw)["moe_expert_weights_on_cpu"]), update
    assert not math.isnan(adapter.host_memory_gib(moe, {"n_cpu_moe": 5, "cache_ram": 0}, hw)["moe_expert_weights_on_cpu"])
    no_layers = moe.model_copy(update={"num_layers": None})
    assert math.isnan(adapter.host_memory_gib(no_layers, {"n_cpu_moe": 5, "cache_ram": 0}, hw)["moe_expert_weights_on_cpu"])


def test_base_class_default_is_untouched_by_the_other_adapters(adapter: LlamaCppAdapter) -> None:
    assert "host_memory_gib" not in EngineAdapter.__abstractmethods__


# ------------------------------------------------------------------ the shared estimator
def test_estimator_sizes_the_kv_cache_from_the_context(adapter: LlamaCppAdapter, dense: ModelInfo, hw: Hardware) -> None:
    mem = adapter.memory_model(dense, {"ctx_size": 32768})
    report = estimate_fit(dense, mem, hw)
    gpu = report.per_gpu[0]
    assert gpu.kv_cache_gib == pytest.approx(32768 * KV_PER_TOKEN / GIB, abs=0.01) == pytest.approx(8.0, abs=0.01)
    assert gpu.weights_gib == pytest.approx(18.6, abs=0.01) and gpu.cuda_graphs_gib == 0.0
    assert gpu.budget_gib == pytest.approx(95.6) and report.verdict == "fits" and report.tp == 1
    assert report.max_len == 32768


def test_estimator_with_quantized_cache_and_two_gpus(adapter: LlamaCppAdapter, dense: ModelInfo, hw2: Hardware) -> None:
    mem = adapter.memory_model(dense, {"ctx_size": 32768, "cache_type_k": "q8_0", "cache_type_v": "q8_0"})
    report = estimate_fit(dense, mem, hw2)
    assert report.tp == 2 and len(report.per_gpu) == 2
    assert report.per_gpu[0].kv_cache_gib == pytest.approx(32768 * KV_PER_TOKEN * 1.0625 / 2 / GIB / 2, abs=0.01)
    assert report.per_gpu[0].weights_gib == pytest.approx(18.6 / 2, abs=0.01)


def test_estimator_without_a_context_uses_the_native_length_and_says_so(adapter: LlamaCppAdapter, dense: ModelInfo, hw: Hardware) -> None:
    report = estimate_fit(dense, adapter.memory_model(dense, {}), hw)
    assert report.max_len == 40960 and any("max_len not set" in n for n in report.notes)
    assert report.confidence != "high"


def test_estimator_accepts_the_partial_offload_keys(adapter: LlamaCppAdapter, dense: ModelInfo, hw: Hardware) -> None:
    mem = adapter.memory_model(dense, {"n_gpu_layers": 20, "ctx_size": 8192})
    assert {"weights_gpu_fraction", "kv_gpu_fraction"} <= set(mem)
    assert all(isinstance(v, float) and not math.isnan(v) for v in mem.values())      # the core converts them with float()
    assert estimate_fit(dense, mem, hw).verdict in ("fits", "tight", "wont_fit")
