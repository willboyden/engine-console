//! Port of backend/tests/core/test_fit_offload.py: partial GPU offload (llama.cpp -ngl, --no-kv-offload) charges
//! the GPU only for the share an engine keeps there (memory_model's weights_gpu_fraction / kv_gpu_fraction).

#![allow(clippy::unwrap_used, clippy::float_cmp)]

#[path = "common/wsc_models.rs"]
mod wsc_models;

use ec_hub::adapters::base::FloatMap;
use ec_hub::domain::models::FitReport;
use ec_hub::services::fit::{Mem, estimate_fit, gpu_fraction};
use wsc_models::*;

fn mem0() -> FloatMap {
    mem(&[("mem_fraction", 0.9), ("max_len", 32768.0), ("tp", 1.0)])
}

fn est(mm: &FloatMap) -> FitReport {
    estimate_fit(
        &qwen3_32b_dense(),
        &Mem::from_map(mm),
        &hw(1),
        1,
        &[],
        &[],
        None,
    )
    .unwrap()
}

fn rel(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-6 * b.abs()
}

#[test]
fn test_half_the_layers_on_the_gpu_halves_the_gpu_weights_and_says_so() {
    let full = est(&mem0());
    let half = est(&with(&mem0(), &[("weights_gpu_fraction", 0.5)]));
    assert!(rel(
        half.per_gpu[0].weights_gib,
        full.per_gpu[0].weights_gib / 2.0
    ));
    assert!(rel(
        half.per_gpu[0].kv_cache_gib,
        full.per_gpu[0].kv_cache_gib
    ));
    assert!(
        half.notes
            .iter()
            .any(|n| n.contains("50% of the weights stay in host RAM"))
    );
    assert!(!full.notes.iter().any(|n| n.contains("host RAM")));
}

#[test]
fn test_kv_kept_in_host_ram_leaves_the_gpu_budget_and_stretches_the_context() {
    let on_gpu = est(&mem0());
    let in_ram = est(&with(&mem0(), &[("kv_gpu_fraction", 0.0)]));
    assert!(in_ram.per_gpu[0].kv_cache_gib == 0.0 && on_gpu.per_gpu[0].kv_cache_gib > 0.0);
    assert!(in_ram.per_gpu[0].total_gib < on_gpu.per_gpu[0].total_gib);
    assert_eq!(in_ram.max_context_at_current_concurrency, None);
    let half = est(&with(&mem0(), &[("kv_gpu_fraction", 0.5)]));
    let one = on_gpu.max_concurrency_at_current_context.unwrap();
    let two = half.max_concurrency_at_current_context.unwrap();
    assert!(two >= 2 * one - 1);
}

#[test]
fn test_fraction_is_clamped_and_defaults_to_all_on_the_gpu() {
    let f =
        |pairs: &[(&str, f64)]| gpu_fraction(&Mem::from_map(&mem(pairs)), "weights_gpu_fraction");
    assert_eq!(f(&[]), 1.0);
    assert_eq!(f(&[("weights_gpu_fraction", 1.7)]), 1.0);
    assert_eq!(f(&[("weights_gpu_fraction", -3.0)]), 0.0);
    assert_eq!(f(&[("weights_gpu_fraction", f64::NAN)]), 1.0);
    // Python's last case passes a string ("x" -> 1.0); an adapter's memory model is a FloatMap here, so a value
    // that is not a number cannot reach the estimator at all.
}
