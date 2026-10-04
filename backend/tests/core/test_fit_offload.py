"""Partial GPU offload (llama.cpp -ngl, --no-kv-offload): the estimator charges the GPU only for the share an engine keeps
there, as reported by memory_model()'s weights_gpu_fraction / kv_gpu_fraction."""
from __future__ import annotations

import pytest

from engine_console.services.fit import _gpu_fraction, estimate_fit

from .model_fixtures import hw, qwen3_32b_dense

MEM = {"mem_fraction": 0.9, "max_len": 32768, "tp": 1}


def test_half_the_layers_on_the_gpu_halves_the_gpu_weights_and_says_so() -> None:
    full = estimate_fit(qwen3_32b_dense(), MEM, hw())
    half = estimate_fit(qwen3_32b_dense(), {**MEM, "weights_gpu_fraction": 0.5}, hw())
    assert half.per_gpu[0].weights_gib == pytest.approx(full.per_gpu[0].weights_gib / 2, rel=1e-6)
    assert half.per_gpu[0].kv_cache_gib == pytest.approx(full.per_gpu[0].kv_cache_gib, rel=1e-6)
    assert any("50% of the weights stay in host RAM" in n for n in half.notes)
    assert not any("host RAM" in n for n in full.notes)


def test_kv_kept_in_host_ram_leaves_the_gpu_budget_and_stretches_the_context() -> None:
    on_gpu = estimate_fit(qwen3_32b_dense(), MEM, hw())
    in_ram = estimate_fit(qwen3_32b_dense(), {**MEM, "kv_gpu_fraction": 0.0}, hw())
    assert in_ram.per_gpu[0].kv_cache_gib == 0.0 and on_gpu.per_gpu[0].kv_cache_gib > 0
    assert in_ram.per_gpu[0].total_gib < on_gpu.per_gpu[0].total_gib
    assert in_ram.max_context_at_current_concurrency is None          # the GPU no longer bounds the context
    half = estimate_fit(qwen3_32b_dense(), {**MEM, "kv_gpu_fraction": 0.5}, hw())
    assert on_gpu.max_concurrency_at_current_context is not None
    assert half.max_concurrency_at_current_context is not None
    assert half.max_concurrency_at_current_context >= 2 * on_gpu.max_concurrency_at_current_context - 1


def test_fraction_is_clamped_and_defaults_to_all_on_the_gpu() -> None:
    assert _gpu_fraction({}, "weights_gpu_fraction") == 1.0
    assert _gpu_fraction({"weights_gpu_fraction": 1.7}, "weights_gpu_fraction") == 1.0
    assert _gpu_fraction({"weights_gpu_fraction": -3}, "weights_gpu_fraction") == 0.0
    assert _gpu_fraction({"weights_gpu_fraction": float("nan")}, "weights_gpu_fraction") == 1.0
    assert _gpu_fraction({"weights_gpu_fraction": "x"}, "weights_gpu_fraction") == 1.0      # type: ignore[dict-item]
