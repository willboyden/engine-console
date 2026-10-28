//! Port of backend/tests/llamacpp/test_memory.py: memory_model() and host_memory_gib() of the llama.cpp adapter, and how
//! the shared estimator consumes them (those four rows run the real `services::fit::estimate_fit`).

use crate::a2::{approx, estimate_fit, fget, keys, llama, model, p, with};
use ec_hub::adapters::base::{EngineAdapter, FloatMap, Hardware, ModelInfo};
use ec_hub::adapters::llamacpp::LlamaCppAdapter;
use ec_hub::adapters::stub::StubAdapter;
use serde_json::{Value, json};

const GIB: f64 = 1_073_741_824.0;
/// K+V, 64 layers, 8 KV heads of 128, f16 -> 256 KiB per token
const KV_PER_TOKEN: f64 = 2.0 * 64.0 * 8.0 * 128.0 * 2.0;

fn mm(m: &ModelInfo, params: Value) -> FloatMap {
    LlamaCppAdapter::new().memory_model(m, &p(params))
}

fn host(m: &ModelInfo, params: Value, hw: &Hardware) -> FloatMap {
    LlamaCppAdapter::new().host_memory_gib(m, &p(params), hw)
}

fn get(m: &FloatMap, k: &str) -> f64 {
    fget(m, k).unwrap_or_else(|| panic!("{k} not in {m:?}"))
}

/// `pytest.approx(b, abs=tol)`: only the absolute tolerance counts.
fn near(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

fn bare() -> ModelInfo {
    model(json!({"repo_id": "x"}))
}

#[test]
fn test_defaults_llamacpp() {
    let want: FloatMap = vec![
        ("mem_fraction".into(), 1.0),
        ("kv_bytes_per_elem".into(), 2.0),
        ("overhead_gib".into(), 1.0),
        ("cuda_graph_gib".into(), 0.0),
        ("max_batched_tokens".into(), 512.0),
    ];
    assert_eq!(mm(&llama::dense(), json!({})), want);
}

#[test]
fn test_context_is_the_total_and_slots_are_informational_llamacpp() {
    let dense = llama::dense();
    let mem = mm(
        &dense,
        json!({"ctx_size": 32768, "parallel": 4, "ubatch_size": 1024}),
    );
    assert!(
        get(&mem, "max_len") == 32768.0
            && get(&mem, "max_seqs") == 4.0
            && get(&mem, "max_batched_tokens") == 1024.0
    );
    assert!(fget(&mm(&dense, json!({"ctx_size": 0})), "max_len").is_none());
    assert!(fget(&mm(&dense, json!({"parallel": -1})), "max_seqs").is_none());
}

/// 8 parametrized rows: (k, v, expected).
#[test]
fn test_kv_bytes_per_element_is_the_mean_of_k_and_v_llamacpp() {
    let dense = llama::dense();
    for (k, v, expected) in [
        ("f16", "f16", 2.0),
        ("q8_0", "q8_0", 1.0625),
        ("q4_0", "q4_0", 0.5625),
        ("q8_0", "f16", 1.53125),
        ("f32", "f16", 3.0),
        ("bf16", "bf16", 2.0),
        ("q5_1", "q5_0", 0.71875),
        ("iq4_nl", "q4_1", 0.59375),
    ] {
        let got = get(
            &mm(&dense, json!({"cache_type_k": k, "cache_type_v": v})),
            "kv_bytes_per_elem",
        );
        assert!(approx(got, expected), "{k}/{v}: {got}");
    }
}

#[test]
fn test_overhead_follows_the_fit_margin_llamacpp() {
    let dense = llama::dense();
    assert_eq!(
        get(&mm(&dense, json!({"fit_target": 2048})), "overhead_gib"),
        2.0
    );
    assert_eq!(
        get(&mm(&dense, json!({"fit_target": 0})), "overhead_gib"),
        1.0
    );
    assert_eq!(
        get(
            &mm(&dense, json!({"fit_target": 4096, "fit": "off"})),
            "overhead_gib"
        ),
        1.0
    );
    assert_eq!(get(&mm(&dense, json!({"fit": "on"})), "overhead_gib"), 1.0);
}

/// 5 parametrized rows: (params, tp).
#[test]
fn test_tp_counts_the_gpus_the_model_is_spread_over_llamacpp() {
    let dense = llama::dense();
    for (params, tp) in [
        (json!({"split_mode": "none"}), 1.0),
        (json!({"tensor_split": "3,1"}), 2.0),
        (json!({"tensor_split": "1,0,0"}), 1.0),
        (json!({"tensor_split": "0.5,0.5,0"}), 2.0),
        (json!({"split_mode": "none", "tensor_split": "3,1"}), 1.0),
    ] {
        assert_eq!(get(&mm(&dense, params.clone()), "tp"), tp, "{params}");
        // the estimator then uses the selected GPUs
        assert!(fget(&mm(&dense, json!({"split_mode": "layer"})), "tp").is_none());
    }
}

#[test]
fn test_partial_offload_is_reported_as_gpu_fractions_llamacpp() {
    let dense = llama::dense();
    let mem = mm(&dense, json!({"n_gpu_layers": 20}));
    assert!(
        approx(get(&mem, "weights_gpu_fraction"), 20.0 / 65.0)
            && approx(get(&mem, "kv_gpu_fraction"), 20.0 / 64.0)
    );
    assert!(near(get(&mem, "weights_gpu_fraction") * 18.6, 5.72, 0.01)); // GiB of weights that stay on the GPU
    let full = mm(&dense, json!({"n_gpu_layers": 65})); // 64 layers + the output layer: all of it
    assert!(
        fget(&full, "weights_gpu_fraction").is_none() && fget(&full, "kv_gpu_fraction").is_none()
    );
    for params in [
        json!({}),
        json!({"n_gpu_layers": "all"}),
        json!({"n_gpu_layers": "auto"}),
        json!({"n_gpu_layers": 999}),
    ] {
        let out = mm(&dense, params.clone());
        assert!(
            fget(&out, "weights_gpu_fraction").is_none() && fget(&out, "kv_gpu_fraction").is_none(),
            "{params}"
        );
    }
    let nothing = mm(&dense, json!({"n_gpu_layers": 0}));
    assert!(
        get(&nothing, "weights_gpu_fraction") == 0.0 && get(&nothing, "kv_gpu_fraction") == 0.0
    );
    let cpu = mm(&dense, json!({"device": "none"}));
    assert!(get(&cpu, "weights_gpu_fraction") == 0.0 && get(&cpu, "kv_gpu_fraction") == 0.0);
    let kv_host = mm(&dense, json!({"kv_offload": false}));
    assert!(
        fget(&kv_host, "weights_gpu_fraction").is_none() && get(&kv_host, "kv_gpu_fraction") == 0.0
    );
    assert!(
        fget(
            &mm(&bare(), json!({"n_gpu_layers": 20})),
            "weights_gpu_fraction"
        )
        .is_none()
    ); // layers unknown
}

// ------------------------------------------------------------------ host RAM
#[test]
fn test_host_memory_defaults_to_the_prompt_cache_ceiling_llamacpp() {
    let (dense, hw) = (llama::dense(), llama::hw());
    assert_eq!(
        host(&dense, json!({}), &hw),
        vec![("prompt_cache".to_string(), 8.0)]
    );
    assert_eq!(
        host(&dense, json!({"cache_ram": 2048}), &hw),
        vec![("prompt_cache".to_string(), 2.0)]
    );
    assert!(host(&dense, json!({"cache_ram": 0}), &hw).is_empty());
    assert!(get(&host(&dense, json!({"cache_ram": -1}), &hw), "prompt_cache").is_nan());
}

#[test]
fn test_layers_left_on_the_cpu_count_as_host_ram_llamacpp() {
    let (dense, hw) = (llama::dense(), llama::hw());
    let out = host(&dense, json!({"n_gpu_layers": 20, "cache_ram": 0}), &hw);
    assert!(approx(
        get(&out, "cpu_resident_weights"),
        18.6 * (1.0 - 20.0 / 65.0)
    ));
    assert!(get(&out, "kv_cache_on_cpu").is_nan()); // context unset: needed but unknowable (NaN)
    let with_ctx = host(
        &dense,
        json!({"n_gpu_layers": 20, "ctx_size": 32768, "cache_ram": 0}),
        &hw,
    );
    assert!(approx(
        get(&with_ctx, "kv_cache_on_cpu"),
        32768.0 * KV_PER_TOKEN * (1.0 - 20.0 / 64.0) / GIB
    ));
    let cpu_only = host(
        &dense,
        json!({"device": "none", "ctx_size": 4096, "cache_ram": 0}),
        &hw,
    );
    assert!(
        approx(get(&cpu_only, "cpu_resident_weights"), 18.6)
            && approx(get(&cpu_only, "kv_cache_on_cpu"), 1.0)
    );
    let unknown = host(&bare(), json!({"n_gpu_layers": 0, "cache_ram": 0}), &hw);
    assert!(get(&unknown, "cpu_resident_weights").is_nan()); // file size unknown
}

#[test]
fn test_kv_on_the_host_when_offload_is_off_llamacpp() {
    let (dense, hw) = (llama::dense(), llama::hw());
    let out = host(
        &dense,
        json!({"kv_offload": false, "ctx_size": 16384, "cache_ram": 0}),
        &hw,
    );
    assert_eq!(keys(&out), ["kv_cache_on_cpu"]);
    assert!(approx(
        get(&out, "kv_cache_on_cpu"),
        16384.0 * KV_PER_TOKEN / GIB
    ));
    let quantized = host(
        &dense,
        json!({"kv_offload": false, "ctx_size": 16384, "cache_ram": 0, "cache_type_k": "q8_0", "cache_type_v": "q8_0"}),
        &hw,
    );
    assert!(approx(
        get(&quantized, "kv_cache_on_cpu"),
        16384.0 * KV_PER_TOKEN / GIB * 1.0625 / 2.0
    ));
}

#[test]
fn test_moe_experts_on_the_cpu_llamacpp() {
    let (moe, hw) = (llama::moe(), llama::hw());
    let share = (30.5e9 - 3.3e9) / (30.5e9 * (1.0 - 8.0 / 128.0)); // expert share of the weight bytes
    let out = host(&moe, json!({"cpu_moe": true, "cache_ram": 0}), &hw);
    assert_eq!(keys(&out), ["moe_expert_weights_on_cpu"]);
    assert!(approx(get(&out, "moe_expert_weights_on_cpu"), 17.5 * share));
    assert!(0.9 < share && share < 1.0);
    let half = host(&moe, json!({"n_cpu_moe": 24, "cache_ram": 0}), &hw);
    assert!(approx(
        get(&half, "moe_expert_weights_on_cpu"),
        17.5 * share * 0.5
    ));
    let exps = host(
        &moe,
        json!({"override_tensor": "exps=CPU", "cache_ram": 0}),
        &hw,
    );
    assert!(approx(
        get(&exps, "moe_expert_weights_on_cpu"),
        17.5 * share
    ));
    let other = host(
        &moe,
        json!({"override_tensor": r"blk\.0\.=CPU", "cache_ram": 0}),
        &hw,
    );
    assert!(get(&other, "moe_expert_weights_on_cpu").is_nan()); // which tensors? unknowable
    // layers that are on the CPU anyway are counted once, in cpu_resident_weights
    let both = host(
        &moe,
        json!({"cpu_moe": true, "n_gpu_layers": 24, "cache_ram": 0}),
        &hw,
    );
    let w_frac = 24.0 / 49.0;
    assert!(approx(
        get(&both, "cpu_resident_weights"),
        17.5 * (1.0 - w_frac)
    ));
    assert!(approx(
        get(&both, "moe_expert_weights_on_cpu"),
        17.5 * share * w_frac
    ));
}

#[test]
fn test_moe_without_enough_facts_is_unknown_not_guessed_llamacpp() {
    let (moe, hw) = (llama::moe(), llama::hw());
    for update in [
        json!({"num_active_params": null}),
        json!({"num_params": null}),
        json!({"raw_config": {}}),
        json!({"weight_bytes": null}),
    ] {
        let thin = with(&moe, update.clone());
        let out = host(&thin, json!({"cpu_moe": true, "cache_ram": 0}), &hw);
        assert!(get(&out, "moe_expert_weights_on_cpu").is_nan(), "{update}");
    }
    assert!(
        !get(
            &host(&moe, json!({"n_cpu_moe": 5, "cache_ram": 0}), &hw),
            "moe_expert_weights_on_cpu"
        )
        .is_nan()
    );
    let no_layers = with(&moe, json!({"num_layers": null}));
    assert!(
        get(
            &host(&no_layers, json!({"n_cpu_moe": 5, "cache_ram": 0}), &hw),
            "moe_expert_weights_on_cpu"
        )
        .is_nan()
    );
}

/// Python: `host_memory_gib` is not abstract on the base class. Rust: it is a provided trait method; an adapter that does
/// not override it (the stub adapter) gets the base default, an empty map.
#[test]
fn test_base_class_default_is_untouched_by_the_other_adapters_llamacpp() {
    let stub = StubAdapter {
        id: "x",
        display_name: "x",
        default_image: "x:1",
        default_port: 8000,
        model_formats: &["hf"],
        multi_model: false,
        metric_prefixes: None,
        module: "x",
    };
    assert!(
        stub.host_memory_gib(
            &llama::dense(),
            &p(json!({"cache_ram": 2048})),
            &llama::hw()
        )
        .is_empty()
    );
}

// ------------------------------------------------------------------ the shared estimator (services/fit.rs)
#[test]
fn test_estimator_sizes_the_kv_cache_from_the_context_llamacpp() {
    let (dense, hw) = (llama::dense(), llama::hw());
    let report = estimate_fit(&dense, &mm(&dense, json!({"ctx_size": 32768})), &hw);
    let gpu = &report.per_gpu[0];
    assert!(
        near(gpu.kv_cache_gib, 32768.0 * KV_PER_TOKEN / GIB, 0.01)
            && near(gpu.kv_cache_gib, 8.0, 0.01)
    );
    assert!(near(gpu.weights_gib, 18.6, 0.01) && gpu.cuda_graphs_gib == 0.0);
    assert!(approx(gpu.budget_gib, 95.6) && report.verdict == "fits" && report.tp == 1);
    assert_eq!(report.max_len, Some(32768));
}

#[test]
fn test_estimator_with_quantized_cache_and_two_gpus_llamacpp() {
    let (dense, hw2) = (llama::dense(), llama::hw2());
    let mem = mm(
        &dense,
        json!({"ctx_size": 32768, "cache_type_k": "q8_0", "cache_type_v": "q8_0"}),
    );
    let report = estimate_fit(&dense, &mem, &hw2);
    assert!(report.tp == 2 && report.per_gpu.len() == 2);
    assert!(near(
        report.per_gpu[0].kv_cache_gib,
        32768.0 * KV_PER_TOKEN * 1.0625 / 2.0 / GIB / 2.0,
        0.01
    ));
    assert!(near(report.per_gpu[0].weights_gib, 18.6 / 2.0, 0.01));
}

#[test]
fn test_estimator_without_a_context_uses_the_native_length_and_says_so_llamacpp() {
    let (dense, hw) = (llama::dense(), llama::hw());
    let report = estimate_fit(&dense, &mm(&dense, json!({})), &hw);
    assert!(
        report.max_len == Some(40960) && report.notes.iter().any(|n| n.contains("max_len not set"))
    );
    assert_ne!(report.confidence, "high");
}

#[test]
fn test_estimator_accepts_the_partial_offload_keys_llamacpp() {
    let (dense, hw) = (llama::dense(), llama::hw());
    let mem = mm(&dense, json!({"n_gpu_layers": 20, "ctx_size": 8192}));
    assert!(
        fget(&mem, "weights_gpu_fraction").is_some() && fget(&mem, "kv_gpu_fraction").is_some()
    );
    assert!(mem.iter().all(|(_, v)| !v.is_nan())); // the core converts them with float()
    assert!(
        ["fits", "tight", "wont_fit"].contains(&estimate_fit(&dense, &mem, &hw).verdict.as_str())
    );
}
