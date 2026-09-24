"""vLLM adapter: host-RAM components for the fit estimator."""
from __future__ import annotations

import math

from engine_console.adapters.base import EngineAdapter, Hardware, ModelInfo
from engine_console.adapters.vllm import VllmAdapter


def test_base_default_is_empty_and_non_abstract(adapter: VllmAdapter, dense: ModelInfo, hw: Hardware) -> None:
    assert "host_memory_gib" not in EngineAdapter.__abstractmethods__
    assert adapter.host_memory_gib(dense, {}, hw) == {}


def test_cpu_offload_is_per_gpu_times_gpu_count(adapter: VllmAdapter, dense: ModelInfo, hw: Hardware) -> None:
    assert adapter.host_memory_gib(dense, {"cpu_offload_gb": 10, "tensor_parallel_size": 2}, hw) == {"cpu_weight_offload": 20.0}
    assert adapter.host_memory_gib(dense, {"cpu_offload_gb": 5}, hw) == {"cpu_weight_offload": 5.0}
    assert adapter.host_memory_gib(dense, {"cpu_offload_gb": 4, "tensor_parallel_size": 2, "pipeline_parallel_size": 2}, hw) == {
        "cpu_weight_offload": 16.0}


def test_kv_offload_buffer_is_already_a_total(adapter: VllmAdapter, dense: ModelInfo, hw: Hardware) -> None:
    out = adapter.host_memory_gib(dense, {"kv_offloading_size": 8, "tensor_parallel_size": 2, "cpu_offload_gb": 1}, hw)
    assert out == {"cpu_weight_offload": 2.0, "kv_offload_buffer": 8.0}
    assert not any(math.isnan(v) for v in out.values())


def test_params_that_change_host_ram_re_trigger_the_fit(adapter: VllmAdapter) -> None:
    by = {p.key: p for p in adapter.param_catalog()}
    assert by["cpu_offload_gb"].affects_memory and by["kv_offloading_size"].affects_memory
    assert "swap_space" not in by            # not in the pinned catalog, so not modelled
