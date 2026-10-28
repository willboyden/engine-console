//! Port of backend/tests/trtllm/test_trtllm_memory.py: fit-estimator inputs, memory_model and host_memory_gib.

use crate::a2::{estimate_fit, fget, model, p, trt, with};
use ec_hub::adapters::base::{EngineAdapter, FloatMap};
use ec_hub::adapters::trtllm::TrtllmAdapter;
use serde_json::{Value, json};

fn mm(m: &ec_hub::adapters::base::ModelInfo, params: Value) -> FloatMap {
    TrtllmAdapter::new().memory_model(m, &p(params))
}

#[test]
fn test_memory_model_defaults_trtllm() {
    let got = mm(&trt::dense(), json!({}));
    let want: FloatMap = vec![
        ("mem_fraction".into(), 0.9),
        ("kv_bytes_per_elem".into(), 2.0),
        ("overhead_gib".into(), 1.0),
        ("cuda_graph_gib".into(), 1.0),
        ("max_batched_tokens".into(), 8192.0),
        ("max_seqs".into(), 2048.0),
        ("tp".into(), 1.0),
        ("max_len".into(), 131_072.0),
    ];
    assert_eq!(got, want);
}

#[test]
fn test_memory_model_follows_parameters_trtllm() {
    let dense = trt::dense();
    let m = mm(
        &dense,
        json!({"free_gpu_memory_fraction": 0.8, "kv_cache_dtype": "fp8",
                              "tensor_parallel_size": 2, "pipeline_parallel_size": 2, "max_seq_len": 65536,
                              "max_num_tokens": 4096, "max_batch_size": 32}),
    );
    assert!(fget(&m, "mem_fraction") == Some(0.8) && fget(&m, "kv_bytes_per_elem") == Some(1.0));
    assert_eq!(fget(&m, "tp"), Some(4.0)); // world size: weights and KV are split over every GPU
    assert_eq!(fget(&m, "overhead_gib"), Some(2.5)); // 1.0 + 0.5 per extra GPU (the estimator's own heuristic)
    assert_eq!(fget(&m, "max_len"), Some(65536.0));
    assert!(fget(&m, "max_batched_tokens") == Some(4096.0) && fget(&m, "max_seqs") == Some(32.0));
    assert_eq!(
        fget(
            &mm(&dense, json!({"kv_cache_dtype": "nvfp4"})),
            "kv_bytes_per_elem"
        ),
        Some(0.5625)
    );
    assert_eq!(
        fget(
            &mm(&dense, json!({"enable_cuda_graphs": false})),
            "cuda_graph_gib"
        ),
        Some(0.0)
    );
}

#[test]
fn test_fixed_kv_pool_from_a_size_cap_trtllm() {
    let dense = trt::dense();
    assert_eq!(
        fget(
            &mm(&dense, json!({"kv_cache_max_gib": 12.5})),
            "kv_cache_fixed_gib"
        ),
        Some(12.5)
    );
    assert_eq!(
        fget(
            &mm(&dense, json!({"kv_cache_max_tokens": 1000})),
            "kv_cache_fixed_gib"
        ),
        None
    );
}

#[test]
fn test_no_max_len_when_the_model_does_not_say_trtllm() {
    let bare = model(json!({"repo_id": "a/b"}));
    assert_eq!(fget(&mm(&bare, json!({})), "max_len"), None);
    assert_eq!(
        fget(&mm(&bare, json!({"max_seq_len": 4096})), "max_len"),
        Some(4096.0)
    );
}

#[test]
fn test_max_len_is_what_the_engine_will_really_serve_trtllm() {
    // A request above the model's own length is cut down by the engine, so the fit uses the smaller number.
    let dense = trt::dense();
    assert_eq!(
        fget(&mm(&dense, json!({"max_seq_len": 262_144})), "max_len"),
        Some(131_072.0)
    );
    assert_eq!(
        fget(&mm(&dense, json!({"max_seq_len": 4096})), "max_len"),
        Some(4096.0)
    );
    let linear = with(
        &dense,
        json!({"raw_config": {"max_position_embeddings": 131_072,
                                                    "rope_scaling": {"type": "linear", "factor": 2.0}}}),
    );
    assert_eq!(fget(&mm(&linear, json!({})), "max_len"), Some(262_144.0));
}

#[test]
fn test_memory_model_feeds_the_estimator_trtllm() {
    // Smoke test of the contract: the keys are the ones the estimator reads, and a fit is produced.
    let full = with(
        &trt::dense(),
        json!({"num_layers": 80, "head_dim": 128, "hidden_size": 8192}),
    );
    let hw1 = trt::hw1();
    let report = estimate_fit(&full, &mm(&full, json!({"max_seq_len": 32768})), &hw1);
    assert!(
        ["fits", "tight"].contains(&report.verdict.as_str())
            && report.per_gpu[0].kv_cache_gib > 0.0
    );
    let capped = estimate_fit(
        &full,
        &mm(
            &full,
            json!({"max_seq_len": 32768, "kv_cache_max_gib": 2.0}),
        ),
        &hw1,
    );
    assert_eq!(capped.per_gpu[0].kv_cache_gib, 2.0);
}

#[test]
fn test_host_memory_is_the_offload_cache_per_worker_trtllm() {
    let (a, dense, hw) = (TrtllmAdapter::new(), trt::dense(), trt::hw());
    assert!(a.host_memory_gib(&dense, &p(json!({})), &hw).is_empty());
    assert_eq!(
        a.host_memory_gib(&dense, &p(json!({"kv_host_cache_gib": 8})), &hw),
        vec![("kv_host_cache (per GPU worker)".to_string(), 8.0)]
    );
    let two = a.host_memory_gib(
        &dense,
        &p(json!({"kv_host_cache_gib": 8, "tensor_parallel_size": 2})),
        &hw,
    );
    assert_eq!(
        two,
        vec![("kv_host_cache (per GPU worker)".to_string(), 16.0)]
    );
    assert!(!two.iter().any(|(_, v)| v.is_nan()));
}

#[test]
fn test_params_that_change_memory_retrigger_the_fit_trtllm() {
    let cat = TrtllmAdapter::new().param_catalog();
    let by = |k: &str| {
        cat.iter()
            .find(|p| p.key == k)
            .expect("param")
            .affects_memory
    };
    assert!(by("kv_host_cache_gib") && by("kv_cache_max_gib"));
}

#[test]
fn test_unvalidated_garbage_never_raises_trtllm() {
    // The fit service may hand over parameters that were never validated; fall back to the defaults instead.
    // (Python's float("nan") for max_num_tokens cannot be carried by JSON, so "nan" stands in as an unparsed string.)
    let (a, dense, hw1) = (TrtllmAdapter::new(), trt::dense(), trt::hw1());
    let junk = p(
        json!({"free_gpu_memory_fraction": "high", "tensor_parallel_size": null, "pipeline_parallel_size": "2",
                        "max_num_tokens": "nan", "max_batch_size": true, "max_seq_len": "long", "kv_cache_max_gib": "x",
                        "kv_host_cache_gib": [], "moe_expert_parallel_size": "two"}),
    );
    let m = a.try_memory_model(&dense, &junk).expect("never raises");
    assert!(
        fget(&m, "mem_fraction") == Some(0.9)
            && fget(&m, "tp") == Some(1.0)
            && fget(&m, "max_batched_tokens") == Some(8192.0)
    );
    assert!(fget(&m, "max_seqs") == Some(2048.0) && fget(&m, "max_len") == Some(131_072.0));
    assert_eq!(fget(&m, "kv_cache_fixed_gib"), None);
    assert!(
        a.try_host_memory_gib(&dense, &junk, &hw1)
            .expect("never raises")
            .is_empty()
    );
    assert!(a.try_compatibility(&dense, &junk, &hw1).is_ok());
}
