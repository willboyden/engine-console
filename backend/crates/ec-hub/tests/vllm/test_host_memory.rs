//! Port of backend/tests/vllm/test_host_memory.py: host-RAM components for the fit estimator.

use ec_hub::adapters::base::EngineAdapter;
use ec_hub::adapters::stub::StubAdapter;
use ec_hub::adapters::vllm::VllmAdapter;
use serde_json::json;

use crate::ws_a1_kit::{dense, hw_vllm, obj};

fn hm(p: serde_json::Value) -> Vec<(String, f64)> {
    VllmAdapter::new().host_memory_gib(&dense(), &obj(p), &hw_vllm())
}

#[test]
fn test_base_default_is_empty_and_non_abstract_vllm() {
    // `host_memory_gib` is not abstract: the trait's default (here through the core's placeholder adapter) is empty
    let stub = StubAdapter {
        id: "x",
        display_name: "X",
        default_image: "x:1",
        default_port: 1,
        model_formats: &["hf"],
        multi_model: false,
        metric_prefixes: None,
        module: "x",
    };
    assert!(
        stub.host_memory_gib(&dense(), &obj(json!({})), &hw_vllm())
            .is_empty()
    );
    assert!(hm(json!({})).is_empty());
}

#[test]
fn test_cpu_offload_is_per_gpu_times_gpu_count_vllm() {
    assert_eq!(
        hm(json!({"cpu_offload_gb": 10, "tensor_parallel_size": 2})),
        vec![("cpu_weight_offload".into(), 20.0)]
    );
    assert_eq!(
        hm(json!({"cpu_offload_gb": 5})),
        vec![("cpu_weight_offload".into(), 5.0)]
    );
    assert_eq!(
        hm(json!({"cpu_offload_gb": 4, "tensor_parallel_size": 2, "pipeline_parallel_size": 2})),
        vec![("cpu_weight_offload".into(), 16.0)]
    );
}

#[test]
fn test_kv_offload_buffer_is_already_a_total_vllm() {
    let out = hm(json!({"kv_offloading_size": 8, "tensor_parallel_size": 2, "cpu_offload_gb": 1}));
    assert_eq!(
        out,
        vec![
            ("cpu_weight_offload".into(), 2.0),
            ("kv_offload_buffer".into(), 8.0)
        ]
    );
    assert!(!out.iter().any(|(_, v)| v.is_nan()));
}

#[test]
fn test_params_that_change_host_ram_re_trigger_the_fit_vllm() {
    let cat = VllmAdapter::new().param_catalog();
    let by = |k: &str| cat.iter().find(|p| p.key == k);
    assert!(
        by("cpu_offload_gb").unwrap().affects_memory
            && by("kv_offloading_size").unwrap().affects_memory
    );
    assert!(by("swap_space").is_none()); // not in the pinned catalog, so not modelled
}
