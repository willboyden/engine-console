"""Fit-estimator inputs: memory_model and host_memory_gib."""
from __future__ import annotations

import math

from engine_console.adapters.base import Hardware, ModelInfo
from engine_console.adapters.trtllm import TrtllmAdapter
from engine_console.services.fit import estimate_fit


def test_memory_model_defaults(adapter: TrtllmAdapter, dense: ModelInfo) -> None:
    mm = adapter.memory_model(dense, {})
    assert mm == {"mem_fraction": 0.9, "kv_bytes_per_elem": 2.0, "overhead_gib": 1.0, "cuda_graph_gib": 1.0,
                  "max_batched_tokens": 8192.0, "max_seqs": 2048.0, "tp": 1.0, "max_len": 131072.0}


def test_memory_model_follows_parameters(adapter: TrtllmAdapter, dense: ModelInfo) -> None:
    mm = adapter.memory_model(dense, {"free_gpu_memory_fraction": 0.8, "kv_cache_dtype": "fp8",
                                      "tensor_parallel_size": 2, "pipeline_parallel_size": 2, "max_seq_len": 65536,
                                      "max_num_tokens": 4096, "max_batch_size": 32})
    assert mm["mem_fraction"] == 0.8 and mm["kv_bytes_per_elem"] == 1.0
    assert mm["tp"] == 4.0                                  # world size: weights and KV are split over every GPU
    assert mm["overhead_gib"] == 2.5                        # 1.0 + 0.5 per extra GPU (the estimator's own heuristic)
    assert mm["max_len"] == 65536.0 and mm["max_batched_tokens"] == 4096.0 and mm["max_seqs"] == 32.0
    assert adapter.memory_model(dense, {"kv_cache_dtype": "nvfp4"})["kv_bytes_per_elem"] == 0.5625
    assert adapter.memory_model(dense, {"enable_cuda_graphs": False})["cuda_graph_gib"] == 0.0


def test_fixed_kv_pool_from_a_size_cap(adapter: TrtllmAdapter, dense: ModelInfo) -> None:
    assert adapter.memory_model(dense, {"kv_cache_max_gib": 12.5})["kv_cache_fixed_gib"] == 12.5
    assert "kv_cache_fixed_gib" not in adapter.memory_model(dense, {"kv_cache_max_tokens": 1000})


def test_no_max_len_when_the_model_does_not_say(adapter: TrtllmAdapter) -> None:
    assert "max_len" not in adapter.memory_model(ModelInfo(repo_id="a/b"), {})
    assert adapter.memory_model(ModelInfo(repo_id="a/b"), {"max_seq_len": 4096})["max_len"] == 4096.0


def test_max_len_is_what_the_engine_will_really_serve(adapter: TrtllmAdapter, dense: ModelInfo) -> None:
    """A request above the model's own length is cut down by the engine, so the fit uses the smaller number."""
    assert adapter.memory_model(dense, {"max_seq_len": 262144})["max_len"] == 131072.0
    assert adapter.memory_model(dense, {"max_seq_len": 4096})["max_len"] == 4096.0
    linear = dense.model_copy(update={"raw_config": {"max_position_embeddings": 131072,
                                                     "rope_scaling": {"type": "linear", "factor": 2.0}}})
    assert adapter.memory_model(linear, {})["max_len"] == 262144.0


def test_memory_model_feeds_the_estimator(adapter: TrtllmAdapter, dense: ModelInfo, hw1: Hardware) -> None:
    """Smoke test of the contract: the keys are the ones the estimator reads, and a fit is produced."""
    full = dense.model_copy(update={"num_layers": 80, "head_dim": 128, "hidden_size": 8192})
    report = estimate_fit(full, adapter.memory_model(full, {"max_seq_len": 32768}), hw1)
    assert report.verdict in ("fits", "tight") and report.per_gpu[0].kv_cache_gib > 0
    capped = estimate_fit(full, adapter.memory_model(full, {"max_seq_len": 32768, "kv_cache_max_gib": 2.0}), hw1)
    assert capped.per_gpu[0].kv_cache_gib == 2.0


def test_host_memory_is_the_offload_cache_per_worker(adapter: TrtllmAdapter, dense: ModelInfo, hw: Hardware) -> None:
    assert adapter.host_memory_gib(dense, {}, hw) == {}
    assert adapter.host_memory_gib(dense, {"kv_host_cache_gib": 8}, hw) == {"kv_host_cache (per GPU worker)": 8.0}
    two = adapter.host_memory_gib(dense, {"kv_host_cache_gib": 8, "tensor_parallel_size": 2}, hw)
    assert two == {"kv_host_cache (per GPU worker)": 16.0}
    assert not any(math.isnan(v) for v in two.values())


def test_params_that_change_memory_retrigger_the_fit(adapter: TrtllmAdapter) -> None:
    by = {p.key: p for p in adapter.param_catalog()}
    assert by["kv_host_cache_gib"].affects_memory and by["kv_cache_max_gib"].affects_memory


def test_unvalidated_garbage_never_raises(adapter: TrtllmAdapter, dense: ModelInfo, hw1: Hardware) -> None:
    """The fit service may hand over parameters that were never validated; fall back to the defaults instead."""
    junk = {"free_gpu_memory_fraction": "high", "tensor_parallel_size": None, "pipeline_parallel_size": "2",
            "max_num_tokens": float("nan"), "max_batch_size": True, "max_seq_len": "long", "kv_cache_max_gib": "x",
            "kv_host_cache_gib": [], "moe_expert_parallel_size": "two"}
    mm = adapter.memory_model(dense, junk)
    assert mm["mem_fraction"] == 0.9 and mm["tp"] == 1.0 and mm["max_batched_tokens"] == 8192.0
    assert mm["max_seqs"] == 2048.0 and mm["max_len"] == 131072.0 and "kv_cache_fixed_gib" not in mm
    assert adapter.host_memory_gib(dense, junk, hw1) == {}
    assert isinstance(adapter.compatibility(dense, junk, hw1), list)
