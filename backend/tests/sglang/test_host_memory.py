"""SGLang adapter: host-RAM components (HiCache host pool, CPU offload)."""
from __future__ import annotations

import math

from engine_console.adapters.base import Hardware, ModelInfo
from engine_console.adapters.sglang import SglangAdapter

A = SglangAdapter()
GIB = 1024**3


def hw(n: int = 1) -> Hardware:
    return Hardware(gpu_ids=list(range(n)), gpu_uuids=[f"GPU-{i}" for i in range(n)], gpu_names=["GPU"] * n,
                    gpu_total_gib=[95.0] * n, gpu_free_gib=[94.0] * n, compute_capability="12.0", host_ram_gib=246.0)


def mi(**kw: object) -> ModelInfo:
    return ModelInfo(repo_id="Qwen/Qwen3.6-35B-A3B", **kw)  # type: ignore[arg-type]


def test_defaults_need_no_extra_host_ram() -> None:
    assert A.host_memory_gib(mi(), {}, hw()) == {}
    assert A.host_memory_gib(mi(), {"hicache_size": 16}, hw()) == {}          # hicache options without the feature: ignored


def test_hicache_size_overrides_ratio_and_scales_with_tp() -> None:
    out = A.host_memory_gib(mi(), {"enable_hierarchical_cache": True, "hicache_size": 16, "hicache_ratio": 9.0, "tp_size": 2}, hw(2))
    assert out == {"hicache_host_pool (assumed per TP rank)": 32.0}


def test_hicache_ratio_is_relative_to_the_device_kv_pool() -> None:
    model = mi(weight_bytes=40 * GIB)
    out = A.host_memory_gib(model, {"enable_hierarchical_cache": True, "hicache_ratio": 2.0, "mem_fraction_static": 0.8}, hw(1))
    pool = 0.8 * 95.0 - 40.0                                                   # device KV pool on one GPU
    assert math.isclose(out["hicache_host_pool (assumed per TP rank)"], 2.0 * pool)
    tp2 = A.host_memory_gib(model, {"enable_hierarchical_cache": True, "tp_size": 2, "mem_fraction_static": 0.8}, hw(2))
    assert math.isclose(tp2["hicache_host_pool (assumed per TP rank)"], 2.0 * (0.8 * 95.0 - 20.0) * 2)   # default ratio 2.0


def test_hicache_ratio_without_weight_size_is_unknown_not_guessed() -> None:
    out = A.host_memory_gib(mi(), {"enable_hierarchical_cache": True}, hw())
    assert math.isnan(out["hicache_host_pool (assumed per TP rank)"])


def test_cpu_offload_scales_with_tp() -> None:
    assert A.host_memory_gib(mi(), {"cpu_offload_gb": 8, "tp_size": 2}, hw(2)) == {"cpu_weight_offload": 16.0}


def test_params_that_change_host_ram_re_trigger_the_fit() -> None:
    by = {p.key: p for p in A.param_catalog()}
    for k in ("cpu_offload_gb", "enable_hierarchical_cache", "hicache_ratio", "hicache_size"):
        assert by[k].affects_memory, k
