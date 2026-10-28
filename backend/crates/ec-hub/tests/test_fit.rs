//! Port of backend/tests/core/test_fit.py: exhaustive tests of the pure fit estimator (expected numbers derived by
//! hand from the formula), plus the cross-language goldens (tests/data/fit_goldens.json, made by
//! backend/parity/gen/fit_goldens.py) and property tests.

#![allow(clippy::unwrap_used, clippy::float_cmp, clippy::cast_precision_loss)]

#[path = "common/wsc_models.rs"]
mod wsc_models;

use ec_hub::adapters::base::{Compat, FloatMap, Hardware, ModelInfo};
use ec_hub::domain::models::{FitReport, HostMemInfo, ResidentUse};
use ec_hub::services::fit::{Kv, Mem, estimate_fit, estimate_host_ram, verdict_of};
use proptest::prelude::*;
use serde_json::{Value, json};
use wsc_models::*;

fn mem0() -> FloatMap {
    mem(&[("mem_fraction", 0.9), ("max_len", 32768.0)])
}

fn est(m: &ModelInfo, mm: &FloatMap, h: &Hardware) -> FitReport {
    estimate_fit(m, &Mem::from_map(mm), h, 1, &[], &[], None).unwrap()
}

fn est_c(m: &ModelInfo, mm: &FloatMap, h: &Hardware, conc: i64) -> FitReport {
    estimate_fit(m, &Mem::from_map(mm), h, conc, &[], &[], None).unwrap()
}

fn est_r(m: &ModelInfo, mm: &FloatMap, h: &Hardware, res: &[ResidentUse]) -> FitReport {
    estimate_fit(m, &Mem::from_map(mm), h, 1, res, &[], None).unwrap()
}

fn gib(x: f64) -> f64 {
    x / GIB
}

fn approx(a: f64, b: f64, abs: f64) -> bool {
    (a - b).abs() <= abs
}

/// pytest.approx default: rel 1e-6
fn rel(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-6 * b.abs().max(1e-12)
}

fn res(name: &str, gpu: i64, gib: f64) -> ResidentUse {
    ResidentUse {
        name: name.into(),
        gpu_ids: vec![gpu],
        gib_per_gpu: gib,
    }
}

// ---- KV formulas -------------------------------------------------------------------------------
#[test]
fn test_gqa_kv_bytes_per_token_and_gib() {
    let r = est(&qwen3_32b_dense(), &mem0(), &hw(1));
    assert_eq!(r.kv_bytes_per_token, Some(262_144.0));
    assert!(approx(r.per_gpu[0].kv_cache_gib, 8.0, 1e-3));
}

#[test]
fn test_mqa_single_kv_head() {
    let mut m = qwen3_32b_dense();
    m.num_kv_heads = Some(1);
    let r = est(&m, &mem0(), &hw(1));
    assert_eq!(r.kv_bytes_per_token, Some((2 * 64 * 128 * 2) as f64));
    assert!(approx(r.per_gpu[0].kv_cache_gib, 1.0, 1e-3));
}

#[test]
fn test_mha_when_kv_heads_missing_is_flagged_low_confidence() {
    let mut m = qwen3_32b_dense();
    m.num_kv_heads = None;
    let r = est(&m, &mem0(), &hw(1));
    assert_eq!(r.kv_bytes_per_token, Some((2 * 64 * 64 * 128 * 2) as f64));
    assert_eq!(r.confidence, "low");
    assert!(r.notes.iter().any(|n| n.contains("MHA")));
}

#[test]
fn test_mla_uses_latent_and_is_not_split_by_tp() {
    let m = deepseek_v2_lite_mla();
    let r1 = est(
        &m,
        &mem(&[("mem_fraction", 0.9), ("max_len", 8192.0), ("tp", 1.0)]),
        &hw(2),
    );
    assert_eq!(r1.kv_bytes_per_token, Some(31_104.0));
    let r2 = est(
        &m,
        &mem(&[("mem_fraction", 0.9), ("max_len", 8192.0), ("tp", 2.0)]),
        &hw(2),
    );
    assert!(rel(r2.per_gpu[0].kv_cache_gib, r1.per_gpu[0].kv_cache_gib));
    assert!(r1.notes.iter().any(|n| n.contains("MLA")));
}

#[test]
fn test_mla_much_smaller_than_equivalent_mha() {
    let mla = est(&deepseek_v2_lite_mla(), &mem0(), &hw(1));
    let mut m = deepseek_v2_lite_mla();
    m.kv_lora_rank = None;
    let mha = est(&m, &mem0(), &hw(1));
    assert!(mla.per_gpu[0].kv_cache_gib * 5.0 < mha.per_gpu[0].kv_cache_gib);
}

#[test]
fn test_sliding_window_layers_cap_at_window() {
    let m = gpt_oss_120b();
    let mm = mem(&[("mem_fraction", 0.9), ("max_len", 131_072.0)]);
    let r = est(&m, &mm, &hw(1));
    let per_layer = (2 * 8 * 64 * 2) as f64;
    let expect = 18.0 * (131_072.0 + 128.0) * per_layer;
    assert!(approx(r.per_gpu[0].kv_cache_gib, gib(expect), 1e-3));
    let mut full_m = gpt_oss_120b();
    full_m.sliding_window = None;
    let full = est(&full_m, &mm, &hw(1));
    assert!(full.per_gpu[0].kv_cache_gib > 1.9 * r.per_gpu[0].kv_cache_gib);
}

#[test]
fn test_short_context_below_window_is_unaffected() {
    let r = est(
        &gpt_oss_120b(),
        &mem(&[("mem_fraction", 0.9), ("max_len", 100.0)]),
        &hw(1),
    );
    assert!(approx(
        r.per_gpu[0].kv_cache_gib,
        gib(36.0 * 100.0 * 2048.0),
        1e-3
    ));
}

#[test]
fn test_sliding_window_disabled_by_use_sliding_window_false() {
    let mut m = qwen3_32b_dense();
    m.sliding_window = Some(4096);
    let r = est(&m, &mem0(), &hw(1));
    assert!(approx(r.per_gpu[0].kv_cache_gib, 8.0, 1e-3));
}

#[test]
fn test_sliding_window_pattern_every_sixth_layer_global() {
    let mut m = qwen3_32b_dense();
    m.num_layers = Some(12);
    m.sliding_window = Some(512);
    m.raw_config = json!({"sliding_window_pattern": 6})
        .as_object()
        .cloned()
        .unwrap();
    let r = est(
        &m,
        &mem(&[("mem_fraction", 0.9), ("max_len", 8192.0)]),
        &hw(1),
    );
    let per_layer = (2 * 8 * 128 * 2) as f64;
    assert!(approx(
        r.per_gpu[0].kv_cache_gib,
        gib(per_layer * (2.0 * 8192.0 + 10.0 * 512.0)),
        1e-4
    ));
}

#[test]
fn test_sliding_window_without_pattern_assumes_all_layers_windowed_medium_confidence() {
    let mut m = qwen3_32b_dense();
    m.sliding_window = Some(1024);
    m.raw_config = serde_json::Map::new();
    let r = est(&m, &mem0(), &hw(1));
    assert!(approx(
        r.per_gpu[0].kv_cache_gib,
        gib((2 * 8 * 128 * 2 * 64 * 1024) as f64),
        1e-4
    ));
    assert_eq!(r.confidence, "medium");
}

#[test]
fn test_moe_kv_depends_on_attention_not_experts_and_weights_are_full_size() {
    let r = est(&qwen3_30b_a3b_moe(), &mem0(), &hw(1));
    assert_eq!(r.kv_bytes_per_token, Some((2 * 48 * 4 * 128 * 2) as f64));
    assert!(approx(
        r.per_gpu[0].weights_gib,
        gib(61_000_000_000.0),
        1e-3
    ));
}

#[test]
fn test_fp8_kv_halves_cache() {
    let bf = est(&qwen3_32b_dense(), &mem0(), &hw(1));
    let f8 = est(
        &qwen3_32b_dense(),
        &with(&mem0(), &[("kv_bytes_per_elem", 1.0)]),
        &hw(1),
    );
    assert!(rel(
        f8.per_gpu[0].kv_cache_gib,
        bf.per_gpu[0].kv_cache_gib / 2.0
    ));
}

#[test]
fn test_concurrency_scales_kv() {
    let r = est_c(&qwen3_32b_dense(), &mem0(), &hw(1), 3);
    assert!(approx(r.per_gpu[0].kv_cache_gib, 24.0, 1e-3));
}

// ---- tensor parallel -----------------------------------------------------------------------------
#[test]
fn test_tp2_splits_weights_and_kv_heads() {
    let r = est(
        &llama33_70b(),
        &mem(&[("mem_fraction", 0.9), ("max_len", 8192.0), ("tp", 2.0)]),
        &hw(2),
    );
    assert_eq!(r.per_gpu.len(), 2);
    let g = &r.per_gpu[0];
    assert!(approx(g.weights_gib, gib(141_107_000_000.0) / 2.0, 1e-3));
    assert!(approx(
        g.kv_cache_gib,
        gib((2 * 80 * 4 * 128 * 2 * 8192) as f64),
        1e-3
    ));
    assert_eq!(r.verdict, "fits");
    assert_eq!(r.tp_required, Some(2));
}

#[test]
fn test_tp1_70b_bf16_wont_fit_one_96gb_card() {
    let r = est(
        &llama33_70b(),
        &mem(&[("mem_fraction", 0.9), ("max_len", 8192.0), ("tp", 1.0)]),
        &hw(2),
    );
    assert_eq!(r.verdict, "wont_fit");
    assert_eq!(r.tp_required, Some(2));
    assert!(r.confidence == "high" || r.confidence == "medium");
}

#[test]
fn test_tp_larger_than_selected_gpus_wont_fit() {
    let r = est(&qwen3_32b_dense(), &with(&mem0(), &[("tp", 2.0)]), &hw(1));
    assert_eq!(r.verdict, "wont_fit");
    assert!(r.notes.iter().any(|n| n.contains("needs 2 GPUs")));
}

#[test]
fn test_tp_exceeding_kv_heads_replicates() {
    let mut m = qwen3_32b_dense();
    m.num_kv_heads = Some(2);
    let r = est(
        &m,
        &mem(&[("mem_fraction", 0.9), ("max_len", 1024.0), ("tp", 4.0)]),
        &hw(4),
    );
    assert_eq!(r.kv_bytes_per_token, Some((2 * 64 * 128 * 2) as f64));
    assert!(r.notes.iter().any(|n| n.contains("replicated")));
}

#[test]
fn test_tp_required_none_when_nothing_fits() {
    let mut huge = llama33_70b();
    huge.weight_bytes = Some(2_000_000_000_000);
    let r = est(&huge, &mem0(), &hw(2));
    assert_eq!(r.tp_required, None);
    assert_eq!(r.verdict, "wont_fit");
}

// ---- verdict thresholds --------------------------------------------------------------------------
#[test]
fn test_verdict_thresholds() {
    for (ratio, expected) in [
        (0.5, "fits"),
        (0.90, "fits"),
        (0.9001, "tight"),
        (1.0, "tight"),
        (1.0001, "wont_fit"),
        (2.0, "wont_fit"),
    ] {
        assert_eq!(verdict_of(ratio * 100.0, 100.0), expected, "{ratio}");
    }
}

#[test]
fn test_verdict_zero_budget() {
    assert_eq!(verdict_of(1.0, 0.0), "wont_fit");
}

#[test]
fn test_end_to_end_thresholds() {
    for (kv_gib, expected) in [
        (80.0, "fits"),
        (80.5, "tight"),
        (90.0, "tight"),
        (90.5, "wont_fit"),
    ] {
        let m = ModelInfo {
            repo_id: "x/y".into(),
            weight_bytes: Some(10 * 1_073_741_824),
            num_layers: Some(1),
            hidden_size: Some(1),
            ..ModelInfo::default()
        };
        let mm = mem(&[
            ("mem_fraction", 1.0),
            ("max_len", 1.0),
            ("activations_gib", 0.0),
            ("cuda_graph_gib", 0.0),
            ("overhead_gib", 0.0),
            ("kv_cache_fixed_gib", kv_gib),
        ]);
        let r = est(&m, &mm, &hw_with(1, 100.0, None));
        assert!(rel(r.per_gpu[0].total_gib, 10.0 + kv_gib));
        assert_eq!(r.verdict, expected, "{kv_gib}");
    }
}

// ---- max context / concurrency ---------------------------------------------------------------------
#[test]
fn test_max_context_and_concurrency_from_pool() {
    let mut m = qwen3_32b_dense();
    m.max_position_embeddings = Some(1_000_000);
    let r = est(&m, &mem0(), &hw(1));
    let g = &r.per_gpu[0];
    let pool =
        g.budget_gib - (g.weights_gib + g.activations_gib + g.cuda_graphs_gib + g.overhead_gib);
    let mc = r.max_context_at_current_concurrency.unwrap() as f64;
    assert!(approx(mc, pool * GIB / 262_144.0, 2.0));
    #[allow(clippy::cast_possible_truncation)]
    let expect = (pool / 8.0).floor() as i64;
    assert_eq!(r.max_concurrency_at_current_context, Some(expect));
}

#[test]
fn test_max_context_capped_at_native_length() {
    let r = est(&qwen3_32b_dense(), &mem0(), &hw(1));
    assert_eq!(r.max_context_at_current_concurrency, Some(40960));
}

#[test]
fn test_max_context_divides_by_concurrency() {
    let mut m = qwen3_32b_dense();
    m.max_position_embeddings = Some(1_000_000);
    let one = est_c(&m, &mem0(), &hw(1), 1)
        .max_context_at_current_concurrency
        .unwrap() as f64;
    let four = est_c(&m, &mem0(), &hw(1), 4)
        .max_context_at_current_concurrency
        .unwrap() as f64;
    assert!(approx(four, one / 4.0, 2.0));
}

#[test]
fn test_max_ctx_closed_form_matches_brute_force_with_sliding_window() {
    let kv = Kv::new(2048.0, 18, 18, Some(128));
    for pool in [1e6, 5e7, 3e9, 4e10] {
        for seqs in [1, 3] {
            let x = kv.max_ctx(pool, seqs).unwrap().unwrap();
            assert!(kv.bytes_for(x, seqs) <= pool);
            assert!(kv.bytes_for(x + 3, seqs) > pool);
        }
    }
}

#[test]
fn test_pool_exhausted_gives_zero() {
    let r = est(
        &llama33_70b(),
        &mem(&[("mem_fraction", 0.9), ("max_len", 8192.0), ("tp", 1.0)]),
        &hw(1),
    );
    assert_eq!(r.verdict, "wont_fit");
    assert_eq!(r.max_context_at_current_concurrency, Some(0));
    assert_eq!(r.max_concurrency_at_current_context, Some(0));
}

// ---- degradation ---------------------------------------------------------------------------------
#[test]
fn test_missing_layers_degrades_to_unknown_low() {
    let mut m = qwen3_32b_dense();
    m.num_layers = None;
    let r = est(&m, &mem0(), &hw(1));
    assert_eq!(r.verdict, "unknown");
    assert_eq!(r.confidence, "low");
    assert_eq!(r.kv_bytes_per_token, None);
    assert_eq!(r.max_context_at_current_concurrency, None);
}

#[test]
fn test_missing_kv_but_weights_alone_exceed_budget_is_wont_fit() {
    let mut m = llama33_70b();
    m.num_layers = None;
    let r = est(
        &m,
        &mem(&[("mem_fraction", 0.9), ("max_len", 8192.0), ("tp", 1.0)]),
        &hw(1),
    );
    assert_eq!(r.verdict, "wont_fit");
    assert_eq!(r.confidence, "medium");
}

#[test]
fn test_missing_weights_and_params_is_unknown() {
    let mut m = qwen3_32b_dense();
    m.weight_bytes = None;
    m.num_params = None;
    let r = est(&m, &mem0(), &hw(1));
    assert_eq!(r.verdict, "unknown");
    assert_eq!(r.confidence, "low");
    assert!(r.notes.iter().any(|n| n.contains("weights size unknown")));
}

#[test]
fn test_weights_estimated_from_params_lowers_confidence() {
    let mut m = qwen3_32b_dense();
    m.weight_bytes = None;
    let r = est(&m, &mem0(), &hw(1));
    assert!(approx(
        r.per_gpu[0].weights_gib,
        gib(32_762_000_000.0 * 2.0),
        1e-3
    ));
    assert_eq!(r.confidence, "medium");
    assert!(
        r.notes
            .contains(&"weights estimated from 32,762,000,000 params at 2 B/param".to_string())
    );
}

#[test]
fn test_missing_hidden_size_uses_default_activations_medium() {
    let mut m = qwen3_32b_dense();
    m.hidden_size = None;
    let r = est(&m, &mem0(), &hw(1));
    assert_eq!(r.per_gpu[0].activations_gib, 1.5);
    assert_eq!(r.confidence, "medium");
}

#[test]
fn test_max_len_falls_back_to_native_with_note() {
    let r = est(&qwen3_32b_dense(), &mem(&[("mem_fraction", 0.9)]), &hw(1));
    assert_eq!(r.max_len, Some(40960));
    assert!(r.notes.iter().any(|n| n.contains("native")));
}

#[test]
fn test_no_context_info_at_all_is_unknown() {
    let mut m = qwen3_32b_dense();
    m.max_position_embeddings = None;
    let r = est(&m, &mem(&[("mem_fraction", 0.9)]), &hw(1));
    assert_eq!(r.verdict, "unknown");
    assert_eq!(r.confidence, "low");
}

#[test]
fn test_max_len_over_native_notes_rope_scaling() {
    let r = est(
        &qwen3_32b_dense(),
        &with(&mem0(), &[("max_len", 131_072.0)]),
        &hw(1),
    );
    assert!(r.notes.iter().any(|n| n.contains("rope scaling")));
}

#[test]
fn test_no_gpus_is_unknown() {
    let empty = Hardware {
        gpu_ids: vec![],
        gpu_uuids: vec![],
        gpu_names: vec![],
        gpu_total_gib: vec![],
        gpu_free_gib: vec![],
        compute_capability: "unknown".into(),
        host_ram_gib: 1.0,
        host_id: "local".into(),
        arch: "x86_64".into(),
        unified_memory: false,
    };
    let r = est(&qwen3_32b_dense(), &mem0(), &empty);
    assert_eq!(r.verdict, "unknown");
    assert!(r.per_gpu.is_empty());
}

#[test]
fn test_default_mem_fraction_noted() {
    let r = est(&qwen3_32b_dense(), &mem(&[("max_len", 4096.0)]), &hw(1));
    assert!(rel(r.per_gpu[0].budget_gib, 96.0 * 0.9));
    assert!(
        r.notes
            .contains(&"engine gave no mem_fraction: assumed 0.9".to_string())
    );
}

// ---- currently used VRAM ---------------------------------------------------------------------------------
#[test]
fn test_free_vram_smaller_than_need_is_wont_fit_and_suggests_stopping() {
    let r = est_r(
        &qwen3_32b_dense(),
        &mem0(),
        &hw_with(1, 96.0, Some(vec![30.0])),
        &[res("qwen-a", 0, 60.0)],
    );
    assert_eq!(r.verdict, "wont_fit");
    assert_eq!(r.fits_if_stop, vec!["qwen-a".to_string()]);
    assert!(r.notes.iter().any(|n| n.contains("only 30.0 GiB is free")));
}

#[test]
fn test_stopping_resident_would_not_help_when_still_too_big() {
    let r = est_r(
        &qwen3_32b_dense(),
        &mem0(),
        &hw_with(1, 96.0, Some(vec![30.0])),
        &[res("tiny", 0, 1.0)],
    );
    assert_eq!(r.verdict, "wont_fit");
    assert!(r.fits_if_stop.is_empty());
}

#[test]
fn test_resident_on_other_gpu_is_not_blamed() {
    let r = est_r(
        &qwen3_32b_dense(),
        &mem0(),
        &hw_with(2, 96.0, Some(vec![30.0, 96.0])),
        &[res("other", 1, 60.0)],
    );
    assert!(r.fits_if_stop.is_empty());
}

#[test]
fn test_plenty_free_keeps_fits() {
    let r = est(
        &qwen3_32b_dense(),
        &mem0(),
        &hw_with(1, 96.0, Some(vec![90.0])),
    );
    assert_eq!(r.verdict, "fits");
}

#[test]
fn test_worst_gpu_decides_overall_verdict() {
    let r = est(
        &qwen3_32b_dense(),
        &with(&mem0(), &[("tp", 2.0)]),
        &hw_with(2, 96.0, Some(vec![90.0, 10.0])),
    );
    let v: Vec<&str> = r.per_gpu.iter().map(|g| g.verdict.as_str()).collect();
    assert_eq!(v, ["fits", "wont_fit"]);
    assert_eq!(r.verdict, "wont_fit");
}

// ---- engine-specific memory models ---------------------------------------------------------------------
#[test]
fn test_sglang_style_reserve_does_not_charge_activations_to_budget() {
    let mm = with(
        &mem0(),
        &[
            ("mem_fraction", 0.85),
            ("overhead_gib", 0.0),
            ("cuda_graph_gib", 0.0),
            ("nonstatic_reserve_frac", 0.15),
        ],
    );
    let r = est(&qwen3_32b_dense(), &mm, &hw(1));
    let g = &r.per_gpu[0];
    assert_eq!(g.activations_gib, 0.0);
    assert!(rel(g.total_gib, g.weights_gib + g.kv_cache_gib));
}

#[test]
fn test_sglang_reserve_too_small_for_activations_warns() {
    let mm = with(
        &mem0(),
        &[
            ("mem_fraction", 0.99),
            ("overhead_gib", 0.0),
            ("cuda_graph_gib", 0.0),
            ("nonstatic_reserve_frac", 0.01),
        ],
    );
    let r = est(&qwen3_32b_dense(), &mm, &hw(1));
    assert!(r.notes.iter().any(|n| n.contains("non-static")));
}

#[test]
fn test_unknown_extra_memory_keys_are_ignored() {
    let r = est(
        &qwen3_32b_dense(),
        &with(
            &mem0(),
            &[
                ("mem_fraction_is_default", 1.0),
                ("spec_accept_length", 3.0),
            ],
        ),
        &hw(1),
    );
    assert_eq!(r.verdict, "fits");
}

#[test]
fn test_compat_passthrough_and_block_note() {
    let c = vec![Compat {
        level: "block".into(),
        code: "nvfp4_moe_sm120".into(),
        message: "use FP8 or SGLang".into(),
    }];
    let r = estimate_fit(
        &qwen3_32b_dense(),
        &Mem::from_map(&mem0()),
        &hw(1),
        1,
        &[],
        &c,
        None,
    )
    .unwrap();
    assert_eq!(r.compat, c);
    assert!(r.notes.iter().any(|n| n.contains("compatibility")));
}

#[test]
fn test_fits_case_full_breakdown_sums() {
    let r = est(&qwen3_32b_dense(), &mem0(), &hw(1));
    let g = &r.per_gpu[0];
    assert!(approx(
        g.total_gib,
        g.weights_gib + g.kv_cache_gib + g.activations_gib + g.cuda_graphs_gib + g.overhead_gib,
        0.01
    ));
    assert!(approx(
        g.utilization_pct,
        g.total_gib / g.budget_gib * 100.0,
        0.1
    ));
    assert!(r.verdict == "fits" && r.confidence == "high");
}

// ---- how the engine sizes its share (claims_share, share_of_free, outside_share_gib) ------------------------------
fn vllm_like() -> FloatMap {
    mem(&[
        ("mem_fraction", 0.9),
        ("max_len", 4096.0),
        ("overhead_gib", 0.0),
        ("cuda_graph_gib", 0.75),
        ("claims_share", 1.0),
        ("outside_share_gib", 2.5),
    ])
}

#[test]
fn test_an_engine_that_claims_its_share_needs_all_of_it_free_plus_what_it_holds_outside() {
    let r = est_r(
        &qwen3_32b_dense(),
        &vllm_like(),
        &hw_with(1, 96.0, Some(vec![70.0])),
        &[res("qwen-a", 0, 30.0)],
    );
    let g = &r.per_gpu[0];
    assert!(
        g.overhead_gib == 0.0
            && approx(g.total_gib, gib(65_530_000_000.0) + 1.25 + 0.75 + 1.0, 0.01)
    );
    assert!(r.verdict == "wont_fit" && r.fits_if_stop == vec!["qwen-a".to_string()]);
    assert!(r.notes.iter().any(|n| n.contains(
        "needs 88.9 GiB (the engine needs its whole 0.9 share free at start) but only 70.0 GiB is free"
    )));
    let uses: FloatMap = vllm_like()
        .into_iter()
        .filter(|(k, _)| k != "claims_share")
        .collect();
    assert_eq!(
        est(
            &qwen3_32b_dense(),
            &uses,
            &hw_with(1, 96.0, Some(vec![70.0]))
        )
        .verdict,
        "fits"
    );
}

#[test]
fn test_memory_held_outside_the_share_is_not_charged_against_it() {
    let inside = est(
        &qwen3_32b_dense(),
        &with(&vllm_like(), &[("outside_share_gib", 0.0)]),
        &hw(1),
    );
    let outside = est(&qwen3_32b_dense(), &vllm_like(), &hw(1));
    assert_eq!(outside.per_gpu[0].total_gib, inside.per_gpu[0].total_gib);
    assert_eq!(
        outside.max_context_at_current_concurrency,
        inside.max_context_at_current_concurrency
    );
    assert!(outside.verdict == inside.verdict && inside.verdict == "fits");
    assert_eq!(
        est(
            &qwen3_32b_dense(),
            &vllm_like(),
            &hw_with(1, 96.0, Some(vec![88.0]))
        )
        .verdict,
        "wont_fit"
    );
}

fn sglang_like() -> FloatMap {
    mem(&[
        ("mem_fraction", 0.85),
        ("max_len", 32768.0),
        ("overhead_gib", 0.0),
        ("cuda_graph_gib", 0.0),
        ("nonstatic_reserve_frac", 0.15),
        ("share_of_free", 1.0),
    ])
}

#[test]
fn test_a_share_of_free_memory_shrinks_with_a_resident_engine() {
    let r = est_r(
        &qwen3_32b_dense(),
        &sglang_like(),
        &hw_with(1, 96.0, Some(vec![75.0])),
        &[res("small-engine", 0, 21.0)],
    );
    let g = &r.per_gpu[0];
    assert!(rel(g.budget_gib, 0.85 * 75.0));
    assert!(r.verdict == "wont_fit" && r.fits_if_stop == vec!["small-engine".to_string()]);
    assert!(r.notes.iter().any(|n| {
        n.contains("the share is 0.85 of the 75.0 GiB free at launch, not of the 96.0 GiB card")
    }));
    assert!(
        r.notes
            .iter()
            .any(|n| n.contains("0.85 of the 75.0 GiB free is only 63.8 GiB"))
    );
    let pool = (0.85 * 75.0 - gib(65_530_000_000.0)) * GIB;
    #[allow(clippy::cast_possible_truncation)]
    let expect = (pool / 262_144.0) as i64;
    assert_eq!(r.max_context_at_current_concurrency, Some(expect));
    assert_eq!(
        est(&qwen3_32b_dense(), &sglang_like(), &hw(1)).verdict,
        "fits"
    );
}

#[test]
fn test_a_share_of_free_memory_does_not_blame_an_engine_whose_memory_would_not_help() {
    let r = est_r(
        &qwen3_32b_dense(),
        &sglang_like(),
        &hw_with(1, 96.0, Some(vec![75.0])),
        &[res("tiny", 0, 1.0)],
    );
    assert!(r.verdict == "wont_fit" && r.fits_if_stop.is_empty());
}

// ---- cross-language goldens (Python's estimate_fit on 600 seeded random inputs) ---------------------------------
fn goldens() -> Value {
    serde_json::from_str(include_str!("data/fit_goldens.json")).unwrap()
}

#[test]
fn test_fit_matches_python_goldens() {
    let g = goldens();
    let cases = g["fit"].as_array().unwrap();
    assert!(cases.len() >= 500);
    for (i, c) in cases.iter().enumerate() {
        let model: ModelInfo = serde_json::from_value(c["model"].clone()).unwrap();
        let mm: FloatMap = c["mem"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), v.as_f64().unwrap()))
            .collect();
        let hw: Hardware = serde_json::from_value(c["hw"].clone()).unwrap();
        let hw_all: Option<Hardware> = serde_json::from_value(c["hw_all"].clone()).unwrap();
        let resident: Vec<ResidentUse> = serde_json::from_value(c["resident"].clone()).unwrap();
        let compat: Vec<Compat> = serde_json::from_value(c["compat"].clone()).unwrap();
        let conc = c["concurrency"].as_i64().unwrap();
        let got = estimate_fit(
            &model,
            &Mem::from_map(&mm),
            &hw,
            conc,
            &resident,
            &compat,
            hw_all.as_ref(),
        );
        match (c.get("error"), got) {
            (Some(e), Ok(r)) => panic!("case {i}: Python raised {e}, Rust answered {}", r.verdict),
            (Some(_), Err(p)) => assert_eq!(p.status, 500, "case {i}"),
            (None, Err(p)) => panic!("case {i}: Rust failed ({p}) where Python answered"),
            (None, Ok(r)) => {
                let want = &c["report"];
                let have = serde_json::to_value(&r).unwrap();
                assert_eq!(&have, want, "case {i}\ninputs: {c}");
            }
        }
    }
}

#[test]
fn test_host_ram_matches_python_goldens() {
    let g = goldens();
    for (i, c) in g["host_ram"].as_array().unwrap().iter().enumerate() {
        let bd: FloatMap = c["breakdown"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| {
                (
                    p[0].as_str().unwrap().to_string(),
                    p[1].as_f64().unwrap_or(f64::NAN),
                )
            })
            .collect();
        let host: Option<HostMemInfo> = serde_json::from_value(c["host"].clone()).unwrap();
        let r = estimate_host_ram(&bd, host.as_ref());
        assert_eq!(serde_json::to_value(&r).unwrap(), c["result"], "case {i}");
    }
}

// ---- properties ---------------------------------------------------------------------------------------------
proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// "unknown" means exactly: no GPUs, or neither the weights nor (when the weights alone do not already exceed
    /// a budget) the KV cache could be sized; the confidence is then low.
    #[test]
    fn prop_unknown_only_when_a_fact_is_missing(
        layers in prop::option::of(0i64..100), heads in prop::option::of(0i64..128),
        kvh in prop::option::of(0i64..32), hidden in prop::option::of(0i64..10_000),
        weights in prop::option::of(0i64..200_000_000_000), ctx in prop::option::of(0i64..200_000),
        n in 0usize..4, max_len in prop::option::of(0.0f64..100_000.0),
    ) {
        let m = ModelInfo {
            repo_id: "p/q".into(), num_layers: layers, num_attention_heads: heads, num_kv_heads: kvh,
            hidden_size: hidden, weight_bytes: weights, max_position_embeddings: ctx, ..ModelInfo::default()
        };
        let mut mm = mem(&[("mem_fraction", 0.9)]);
        if let Some(l) = max_len { mm.push(("max_len".into(), l)); }
        let r = estimate_fit(&m, &Mem::from_map(&mm), &hw(n), 1, &[], &[], None).unwrap();
        if r.verdict == "unknown" {
            prop_assert_eq!(r.confidence.as_str(), "low");
            prop_assert!(n == 0 || r.per_gpu.iter().all(|g| g.verdict == "unknown")
                || weights.is_none_or(|w| w == 0));
        }
        if n > 0 && weights.is_some_and(|w| w > 0) && r.kv_bytes_per_token.is_some() {
            prop_assert_ne!(r.verdict.as_str(), "unknown");
        }
        for g in &r.per_gpu {
            prop_assert!(g.total_gib >= 0.0 && g.budget_gib >= 0.0);
        }
    }
}
