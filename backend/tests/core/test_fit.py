"""Exhaustive tests for the pure fit estimator. Expected numbers are hand-derived from the formula in
services/fit.py, not copied from its output."""
from __future__ import annotations

import pytest

from engine_console.adapters.base import Compat, ModelInfo
from engine_console.domain.models import ResidentUse
from engine_console.services.fit import _Kv, _verdict, estimate_fit

from .model_fixtures import (
    GIB,
    deepseek_v2_lite_mla,
    gpt_oss_120b,
    hw,
    llama33_70b,
    qwen3_30b_a3b_moe,
    qwen3_32b_dense,
)

MEM = {"mem_fraction": 0.9, "max_len": 32768}


def gib(x: float) -> float:
    return x / GIB


# ---- KV formulas -------------------------------------------------------------------------------
def test_gqa_kv_bytes_per_token_and_gib() -> None:
    r = estimate_fit(qwen3_32b_dense(), MEM, hw())
    assert r.kv_bytes_per_token == 2 * 64 * 8 * 128 * 2 == 262_144
    assert r.per_gpu[0].kv_cache_gib == pytest.approx(8.0, abs=1e-3)   # 32768 tokens * 256 KiB


def test_mqa_single_kv_head() -> None:
    r = estimate_fit(qwen3_32b_dense(num_kv_heads=1), MEM, hw())
    assert r.kv_bytes_per_token == 2 * 64 * 1 * 128 * 2
    assert r.per_gpu[0].kv_cache_gib == pytest.approx(1.0, abs=1e-3)


def test_mha_when_kv_heads_missing_is_flagged_low_confidence() -> None:
    r = estimate_fit(qwen3_32b_dense(num_kv_heads=None), MEM, hw())
    assert r.kv_bytes_per_token == 2 * 64 * 64 * 128 * 2   # upper bound: every head has its own KV
    assert r.confidence == "low"
    assert any("MHA" in n for n in r.notes)


def test_mla_uses_latent_and_is_not_split_by_tp() -> None:
    m = deepseek_v2_lite_mla()
    r1 = estimate_fit(m, {"mem_fraction": 0.9, "max_len": 8192, "tp": 1}, hw(2))
    assert r1.kv_bytes_per_token == 27 * (512 + 64) * 2 == 31_104
    r2 = estimate_fit(m, {"mem_fraction": 0.9, "max_len": 8192, "tp": 2}, hw(2))
    assert r2.per_gpu[0].kv_cache_gib == pytest.approx(r1.per_gpu[0].kv_cache_gib)   # replicated, not halved
    assert any("MLA" in n for n in r1.notes)


def test_mla_much_smaller_than_equivalent_mha() -> None:
    mla = estimate_fit(deepseek_v2_lite_mla(), MEM, hw())
    mha = estimate_fit(deepseek_v2_lite_mla(kv_lora_rank=None), MEM, hw())
    assert mla.per_gpu[0].kv_cache_gib * 5 < mha.per_gpu[0].kv_cache_gib


def test_sliding_window_layers_cap_at_window() -> None:
    m = gpt_oss_120b()
    r = estimate_fit(m, {"mem_fraction": 0.9, "max_len": 131072}, hw())
    per_layer = 2 * 8 * 64 * 2   # K+V, kv_heads, head_dim, bf16
    expect = 18 * (131072 + 128) * per_layer   # 18 full layers + 18 windowed layers
    assert r.per_gpu[0].kv_cache_gib == pytest.approx(gib(expect), abs=1e-3)
    # windowed layers must not inflate: far below a hypothetical all-full-attention model
    full = estimate_fit(gpt_oss_120b(sliding_window=None), {"mem_fraction": 0.9, "max_len": 131072}, hw())
    assert full.per_gpu[0].kv_cache_gib > 1.9 * r.per_gpu[0].kv_cache_gib


def test_short_context_below_window_is_unaffected() -> None:
    r = estimate_fit(gpt_oss_120b(), {"mem_fraction": 0.9, "max_len": 100}, hw())
    assert r.per_gpu[0].kv_cache_gib == pytest.approx(gib(36 * 100 * 2048), abs=1e-3)


def test_sliding_window_disabled_by_use_sliding_window_false() -> None:
    m = qwen3_32b_dense(sliding_window=4096)   # fixture already carries use_sliding_window: False
    r = estimate_fit(m, MEM, hw())
    assert r.per_gpu[0].kv_cache_gib == pytest.approx(8.0, abs=1e-3)


def test_sliding_window_pattern_every_sixth_layer_global() -> None:
    m = qwen3_32b_dense(num_layers=12, sliding_window=512,
                        raw_config={"sliding_window_pattern": 6})
    r = estimate_fit(m, {"mem_fraction": 0.9, "max_len": 8192}, hw())
    per_layer = 2 * 8 * 128 * 2
    assert r.per_gpu[0].kv_cache_gib == pytest.approx(gib(per_layer * (2 * 8192 + 10 * 512)), abs=1e-4)


def test_sliding_window_without_pattern_assumes_all_layers_windowed_medium_confidence() -> None:
    m = qwen3_32b_dense(sliding_window=1024, raw_config={})
    r = estimate_fit(m, MEM, hw())
    assert r.per_gpu[0].kv_cache_gib == pytest.approx(gib(2 * 8 * 128 * 2 * 64 * 1024), abs=1e-4)
    assert r.confidence == "medium"


def test_moe_kv_depends_on_attention_not_experts_and_weights_are_full_size() -> None:
    m = qwen3_30b_a3b_moe()
    r = estimate_fit(m, MEM, hw())
    assert r.kv_bytes_per_token == 2 * 48 * 4 * 128 * 2
    assert r.per_gpu[0].weights_gib == pytest.approx(gib(61_000_000_000), abs=1e-3)   # all experts, not 3B active


def test_fp8_kv_halves_cache() -> None:
    bf = estimate_fit(qwen3_32b_dense(), MEM, hw())
    f8 = estimate_fit(qwen3_32b_dense(), {**MEM, "kv_bytes_per_elem": 1.0}, hw())
    assert f8.per_gpu[0].kv_cache_gib == pytest.approx(bf.per_gpu[0].kv_cache_gib / 2, rel=1e-6)


def test_concurrency_scales_kv() -> None:
    r = estimate_fit(qwen3_32b_dense(), MEM, hw(), concurrency=3)
    assert r.per_gpu[0].kv_cache_gib == pytest.approx(24.0, abs=1e-3)


# ---- tensor parallel -----------------------------------------------------------------------------
def test_tp2_splits_weights_and_kv_heads() -> None:
    m = llama33_70b()
    r = estimate_fit(m, {"mem_fraction": 0.9, "max_len": 8192, "tp": 2}, hw(2))
    assert len(r.per_gpu) == 2
    g = r.per_gpu[0]
    assert g.weights_gib == pytest.approx(gib(141_107_000_000) / 2, abs=1e-3)
    assert g.kv_cache_gib == pytest.approx(gib(2 * 80 * 4 * 128 * 2 * 8192), abs=1e-3)   # 4 KV heads per rank
    assert r.verdict == "fits"
    assert r.tp_required == 2


def test_tp1_70b_bf16_wont_fit_one_96gb_card() -> None:
    r = estimate_fit(llama33_70b(), {"mem_fraction": 0.9, "max_len": 8192, "tp": 1}, hw(2))
    assert r.verdict == "wont_fit"
    assert r.tp_required == 2
    assert r.confidence in ("high", "medium")


def test_tp_larger_than_selected_gpus_wont_fit() -> None:
    r = estimate_fit(qwen3_32b_dense(), {**MEM, "tp": 2}, hw(1))
    assert r.verdict == "wont_fit"
    assert any("needs 2 GPUs" in n for n in r.notes)


def test_tp_exceeding_kv_heads_replicates() -> None:
    m = qwen3_32b_dense(num_kv_heads=2)
    r = estimate_fit(m, {"mem_fraction": 0.9, "max_len": 1024, "tp": 4}, hw(4))
    assert r.kv_bytes_per_token == 2 * 64 * 1 * 128 * 2   # ceil(2/4) = 1 head per rank
    assert any("replicated" in n for n in r.notes)


def test_tp_required_none_when_nothing_fits() -> None:
    huge = llama33_70b(weight_bytes=2_000_000_000_000)
    r = estimate_fit(huge, MEM, hw(2))
    assert r.tp_required is None
    assert r.verdict == "wont_fit"


# ---- verdict thresholds --------------------------------------------------------------------------
@pytest.mark.parametrize(("ratio", "expected"), [(0.5, "fits"), (0.90, "fits"), (0.9001, "tight"),
                                                  (1.0, "tight"), (1.0001, "wont_fit"), (2.0, "wont_fit")])
def test_verdict_thresholds(ratio: float, expected: str) -> None:
    assert _verdict(ratio * 100, 100.0) == expected


def test_verdict_zero_budget() -> None:
    assert _verdict(1.0, 0.0) == "wont_fit"


@pytest.mark.parametrize(("kv_gib", "expected"), [(80.0, "fits"), (80.5, "tight"), (90.0, "tight"), (90.5, "wont_fit")])
def test_end_to_end_thresholds(kv_gib: float, expected: str) -> None:
    m = ModelInfo(repo_id="x/y", weight_bytes=10 * GIB, num_layers=1, hidden_size=1)
    mem = {"mem_fraction": 1.0, "max_len": 1, "activations_gib": 0, "cuda_graph_gib": 0, "overhead_gib": 0,
           "kv_cache_fixed_gib": kv_gib}
    r = estimate_fit(m, mem, hw(1, total=100.0))
    assert r.per_gpu[0].total_gib == pytest.approx(10 + kv_gib)
    assert r.verdict == expected


# ---- max context / concurrency ---------------------------------------------------------------------
def test_max_context_and_concurrency_from_pool() -> None:
    m = qwen3_32b_dense(max_position_embeddings=1_000_000)
    r = estimate_fit(m, MEM, hw())
    g = r.per_gpu[0]
    pool = g.budget_gib - (g.weights_gib + g.activations_gib + g.cuda_graphs_gib + g.overhead_gib)
    assert r.max_context_at_current_concurrency == pytest.approx(pool * GIB / 262_144, abs=2)
    assert r.max_concurrency_at_current_context == int(pool // 8.0)


def test_max_context_capped_at_native_length() -> None:
    r = estimate_fit(qwen3_32b_dense(), MEM, hw())   # pool would allow ~90k tokens, model native is 40960
    assert r.max_context_at_current_concurrency == 40960


def test_max_context_divides_by_concurrency() -> None:
    m = qwen3_32b_dense(max_position_embeddings=1_000_000)
    one = estimate_fit(m, MEM, hw(), concurrency=1).max_context_at_current_concurrency
    four = estimate_fit(m, MEM, hw(), concurrency=4).max_context_at_current_concurrency
    assert one is not None and four is not None
    assert four == pytest.approx(one / 4, abs=2)


def test_max_ctx_closed_form_matches_brute_force_with_sliding_window() -> None:
    kv = _Kv(2048.0, n_full=18, n_sl=18, window=128)
    for pool in (1e6, 5e7, 3e9, 4e10):
        for seqs in (1, 3):
            x = kv.max_ctx(pool, seqs)
            assert x is not None
            assert kv.bytes_for(x, seqs) <= pool
            assert kv.bytes_for(x + 3, seqs) > pool


def test_pool_exhausted_gives_zero() -> None:
    r = estimate_fit(llama33_70b(), {"mem_fraction": 0.9, "max_len": 8192, "tp": 1}, hw(1))
    assert r.verdict == "wont_fit"
    assert r.max_context_at_current_concurrency == 0
    assert r.max_concurrency_at_current_context == 0


# ---- degradation ---------------------------------------------------------------------------------
def test_missing_layers_degrades_to_unknown_low() -> None:
    m = qwen3_32b_dense(num_layers=None)
    r = estimate_fit(m, MEM, hw())
    assert r.verdict == "unknown"
    assert r.confidence == "low"
    assert r.kv_bytes_per_token is None
    assert r.max_context_at_current_concurrency is None


def test_missing_kv_but_weights_alone_exceed_budget_is_wont_fit() -> None:
    m = llama33_70b(num_layers=None)
    r = estimate_fit(m, {"mem_fraction": 0.9, "max_len": 8192, "tp": 1}, hw(1))
    assert r.verdict == "wont_fit"
    assert r.confidence == "medium"   # lower bound already over budget


def test_missing_weights_and_params_is_unknown() -> None:
    r = estimate_fit(qwen3_32b_dense(weight_bytes=None, num_params=None), MEM, hw())
    assert r.verdict == "unknown"
    assert r.confidence == "low"
    assert any("weights size unknown" in n for n in r.notes)


def test_weights_estimated_from_params_lowers_confidence() -> None:
    r = estimate_fit(qwen3_32b_dense(weight_bytes=None), MEM, hw())   # 32.76e9 params * 2 B
    assert r.per_gpu[0].weights_gib == pytest.approx(gib(32_762_000_000 * 2), abs=1e-3)
    assert r.confidence == "medium"


def test_missing_hidden_size_uses_default_activations_medium() -> None:
    r = estimate_fit(qwen3_32b_dense(hidden_size=None), MEM, hw())
    assert r.per_gpu[0].activations_gib == 1.5
    assert r.confidence == "medium"


def test_max_len_falls_back_to_native_with_note() -> None:
    r = estimate_fit(qwen3_32b_dense(), {"mem_fraction": 0.9}, hw())
    assert r.max_len == 40960
    assert any("native" in n for n in r.notes)


def test_no_context_info_at_all_is_unknown() -> None:
    r = estimate_fit(qwen3_32b_dense(max_position_embeddings=None), {"mem_fraction": 0.9}, hw())
    assert r.verdict == "unknown"
    assert r.confidence == "low"


def test_max_len_over_native_notes_rope_scaling() -> None:
    r = estimate_fit(qwen3_32b_dense(), {**MEM, "max_len": 131072}, hw())
    assert any("rope scaling" in n for n in r.notes)


def test_no_gpus_is_unknown() -> None:
    from engine_console.adapters.base import Hardware
    empty = Hardware(gpu_ids=[], gpu_uuids=[], gpu_names=[], gpu_total_gib=[], gpu_free_gib=[],
                     compute_capability="unknown", host_ram_gib=1.0)
    r = estimate_fit(qwen3_32b_dense(), MEM, empty)
    assert r.verdict == "unknown"
    assert r.per_gpu == []


def test_default_mem_fraction_noted() -> None:
    r = estimate_fit(qwen3_32b_dense(), {"max_len": 4096}, hw())
    assert r.per_gpu[0].budget_gib == pytest.approx(96 * 0.9)
    assert any("mem_fraction" in n for n in r.notes)


# ---- currently used VRAM ---------------------------------------------------------------------------------
def test_free_vram_smaller_than_need_is_wont_fit_and_suggests_stopping() -> None:
    res = [ResidentUse(name="qwen-a", gpu_ids=[0], gib_per_gpu=60.0)]
    r = estimate_fit(qwen3_32b_dense(), MEM, hw(1, free=30.0), resident=res)
    assert r.verdict == "wont_fit"
    assert r.fits_if_stop == ["qwen-a"]
    assert any("only 30.0 GiB is free" in n for n in r.notes)


def test_stopping_resident_would_not_help_when_still_too_big() -> None:
    res = [ResidentUse(name="tiny", gpu_ids=[0], gib_per_gpu=1.0)]
    r = estimate_fit(qwen3_32b_dense(), MEM, hw(1, free=30.0), resident=res)
    assert r.verdict == "wont_fit"
    assert r.fits_if_stop == []


def test_resident_on_other_gpu_is_not_blamed() -> None:
    res = [ResidentUse(name="other", gpu_ids=[1], gib_per_gpu=60.0)]
    r = estimate_fit(qwen3_32b_dense(), MEM, hw(2, free=[30.0, 96.0]), resident=res)
    assert r.fits_if_stop == []


def test_plenty_free_keeps_fits() -> None:
    r = estimate_fit(qwen3_32b_dense(), MEM, hw(1, free=90.0))
    assert r.verdict == "fits"


def test_worst_gpu_decides_overall_verdict() -> None:
    r = estimate_fit(qwen3_32b_dense(), {**MEM, "tp": 2}, hw(2, free=[90.0, 10.0]))
    assert [g.verdict for g in r.per_gpu] == ["fits", "wont_fit"]
    assert r.verdict == "wont_fit"


# ---- engine-specific memory models ---------------------------------------------------------------------
def test_sglang_style_reserve_does_not_charge_activations_to_budget() -> None:
    mem = {**MEM, "mem_fraction": 0.85, "overhead_gib": 0.0, "cuda_graph_gib": 0.0, "nonstatic_reserve_frac": 0.15}
    r = estimate_fit(qwen3_32b_dense(), mem, hw())
    g = r.per_gpu[0]
    assert g.activations_gib == 0.0
    assert g.total_gib == pytest.approx(g.weights_gib + g.kv_cache_gib)


def test_sglang_reserve_too_small_for_activations_warns() -> None:
    mem = {**MEM, "mem_fraction": 0.99, "overhead_gib": 0.0, "cuda_graph_gib": 0.0, "nonstatic_reserve_frac": 0.01}
    r = estimate_fit(qwen3_32b_dense(), mem, hw())
    assert any("non-static" in n for n in r.notes)


def test_unknown_extra_memory_keys_are_ignored() -> None:
    r = estimate_fit(qwen3_32b_dense(), {**MEM, "mem_fraction_is_default": 1.0, "spec_accept_length": 3.0}, hw())
    assert r.verdict == "fits"


def test_compat_passthrough_and_block_note() -> None:
    c = [Compat(level="block", code="nvfp4_moe_sm120", message="use FP8 or SGLang")]
    r = estimate_fit(qwen3_32b_dense(), MEM, hw(), compat=c)
    assert r.compat == c
    assert any("compatibility" in n for n in r.notes)


def test_fits_case_full_breakdown_sums() -> None:
    r = estimate_fit(qwen3_32b_dense(), MEM, hw())
    g = r.per_gpu[0]
    assert g.total_gib == pytest.approx(g.weights_gib + g.kv_cache_gib + g.activations_gib + g.cuda_graphs_gib + g.overhead_gib, abs=0.01)
    assert g.utilization_pct == pytest.approx(g.total_gib / g.budget_gib * 100, abs=0.1)
    assert r.verdict == "fits" and r.confidence == "high"
