//! Port of backend/tests/ollama/test_memory.py: memory_model() of the Ollama adapter and how the shared estimator
//! consumes it (the estimator rows run the real `services::fit::estimate_fit`).

use crate::a2::{approx, estimate_fit, fget, oll, p};
use ec_hub::adapters::base::{EngineAdapter, FloatMap, ModelInfo};
use ec_hub::adapters::ollama::OllamaAdapter;
use serde_json::{Value, json};

const GIB: f64 = 1_073_741_824.0;
/// K+V, 32 layers, 8 KV heads of 128, f16 -> 128 KiB per token
const KV_PER_TOKEN: f64 = 2.0 * 32.0 * 8.0 * 128.0 * 2.0;

fn mm(m: &ModelInfo, params: Value) -> FloatMap {
    OllamaAdapter::new().memory_model(m, &p(params))
}

fn get(m: &FloatMap, k: &str) -> f64 {
    fget(m, k).unwrap_or_else(|| panic!("{k} not in {m:?}"))
}

/// `pytest.approx(b, abs=tol)`.
fn near(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

#[test]
fn test_defaults_ollama() {
    let want: FloatMap = vec![
        ("mem_fraction".into(), 1.0),
        ("kv_bytes_per_elem".into(), 2.0),
        ("overhead_gib".into(), 1.0),
        ("cuda_graph_gib".into(), 0.0),
        ("max_batched_tokens".into(), 512.0),
        ("max_seqs".into(), 1.0),
        ("tp".into(), 1.0),
    ];
    assert_eq!(mm(&oll::llama8b(), json!({})), want);
}

/// The Ollama FAQ: "a 2K context with 4 parallel requests will result in an 8K context and additional memory".
#[test]
fn test_the_kv_cache_covers_every_parallel_slot_ollama() {
    let m = oll::llama8b();
    let mem = mm(&m, json!({"context_length": 2048, "num_parallel": 4}));
    assert!(get(&mem, "max_len") == 8192.0 && get(&mem, "max_seqs") == 4.0);
    assert_eq!(
        get(&mm(&m, json!({"context_length": 8192})), "max_len"),
        8192.0
    );
    assert!(fget(&mm(&m, json!({"num_parallel": 4})), "max_len").is_none()); // context unset: Ollama picks it
    assert!(fget(&mm(&m, json!({"context_length": 0})), "max_len").is_none());
}

/// 5 parametrized rows: (kv, expected).
#[test]
fn test_kv_bytes_per_element_ollama() {
    let m = oll::llama8b();
    for (kv, expected) in [
        (Some("f16"), 2.0),
        (Some("q8_0"), 1.0625),
        (Some("q4_0"), 0.5625),
        (None, 2.0),
        (Some("bogus"), 2.0),
    ] {
        let params = kv.map_or_else(|| json!({}), |k| json!({"kv_cache_type": k}));
        assert_eq!(
            get(&mm(&m, params), "kv_bytes_per_elem"),
            expected,
            "{kv:?}"
        );
    }
}

#[test]
fn test_overhead_includes_the_reserved_gpu_memory_ollama() {
    let m = oll::llama8b();
    assert_eq!(
        get(
            &mm(&m, json!({"gpu_overhead": 2_i64 * 1024 * 1024 * 1024})),
            "overhead_gib"
        ),
        3.0
    );
    assert_eq!(
        get(&mm(&m, json!({"gpu_overhead": 0})), "overhead_gib"),
        1.0
    );
}

#[test]
fn test_one_gpu_unless_spreading_is_forced_ollama() {
    let m = oll::llama8b();
    assert_eq!(get(&mm(&m, json!({})), "tp"), 1.0);
    assert_eq!(get(&mm(&m, json!({"sched_spread": false})), "tp"), 1.0);
    assert!(fget(&mm(&m, json!({"sched_spread": true})), "tp").is_none()); // the estimator then uses every selected GPU
}

/// Python checks `isinstance(v, float)`; a FloatMap holds only f64, so the check is that each is a real number.
#[test]
fn test_values_are_plain_floats_the_core_can_convert_ollama() {
    let m = oll::llama8b();
    for params in [
        json!({}),
        json!({"context_length": 4096, "num_parallel": 2, "kv_cache_type": "q4_0", "gpu_overhead": 1}),
    ] {
        assert!(mm(&m, params).iter().all(|(_, v)| v.is_finite()));
    }
}

/// The CPU share of a partly offloaded model is chosen by Ollama's scheduler at load time. (Python also checks that
/// `host_memory_gib` is not abstract: in Rust it is a provided trait method.)
#[test]
fn test_no_host_memory_is_modelled_ollama() {
    let out = OllamaAdapter::new().host_memory_gib(
        &oll::llama8b(),
        &p(json!({"context_length": 262_144, "num_parallel": 8})),
        &oll::hw(),
    );
    assert!(out.is_empty());
}

// ------------------------------------------------------------------ the shared estimator (services/fit.rs)
#[test]
fn test_estimator_sizes_one_model_with_parallel_slots_ollama() {
    let (m, hw) = (oll::llama8b(), oll::hw());
    let gpu = estimate_fit(
        &m,
        &mm(&m, json!({"context_length": 8192, "num_parallel": 4})),
        &hw,
    )
    .per_gpu[0]
        .clone();
    assert!(
        near(gpu.kv_cache_gib, 8192.0 * 4.0 * KV_PER_TOKEN / GIB, 0.01)
            && near(gpu.kv_cache_gib, 4.0, 0.01)
    );
    assert!(
        near(gpu.weights_gib, 4.6, 0.01)
            && gpu.cuda_graphs_gib == 0.0
            && approx(gpu.budget_gib, 95.6)
    );
}

#[test]
fn test_a_quantized_cache_shrinks_the_estimate_ollama() {
    let (m, hw) = (oll::llama8b(), oll::hw());
    let kv = |params: Value| estimate_fit(&m, &mm(&m, params), &hw).per_gpu[0].kv_cache_gib;
    let f16 = kv(json!({"context_length": 65536}));
    let q8 = kv(json!({"context_length": 65536, "kv_cache_type": "q8_0"}));
    let q4 = kv(json!({"context_length": 65536, "kv_cache_type": "q4_0"}));
    assert!(near(f16, 8.0, 0.01));
    assert!(near(q8 / f16, 1.0625 / 2.0, 0.01) && near(q4 / f16, 0.5625 / 2.0, 0.01));
}

/// Why the compat note exists: with the context set to what Ollama picks on a big card, a 16 GiB card is out.
#[test]
fn test_the_default_256k_context_does_not_fit_a_small_card_ollama() {
    let (m, small) = (oll::llama8b(), oll::small_gpu());
    let report = estimate_fit(&m, &mm(&m, json!({"context_length": 262_144})), &small);
    assert!(report.verdict == "wont_fit" && near(report.per_gpu[0].kv_cache_gib, 32.0, 0.01));
    assert_eq!(
        estimate_fit(&m, &mm(&m, json!({"context_length": 8192})), &small).verdict,
        "fits"
    );
}

#[test]
fn test_estimator_with_unset_context_uses_the_native_length_and_says_so_ollama() {
    let (m, hw) = (oll::llama8b(), oll::hw());
    let report = estimate_fit(&m, &mm(&m, json!({})), &hw);
    assert!(
        report.max_len == Some(131_072)
            && report.notes.iter().any(|n| n.contains("max_len not set"))
    );
    assert_ne!(report.confidence, "high");
}

#[test]
fn test_a_model_without_layer_facts_degrades_to_unknown_not_a_guess_ollama() {
    let (b, hw) = (oll::bare(), oll::hw());
    let report = estimate_fit(&b, &mm(&b, json!({"context_length": 8192})), &hw);
    assert!(
        report.confidence == "low"
            && report
                .notes
                .iter()
                .any(|n| n.contains("cannot be computed"))
    );
    assert!(report.verdict == "unknown" || report.verdict == "wont_fit");
}
