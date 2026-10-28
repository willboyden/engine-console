//! Port of backend/tests/sglang/test_host_memory.py: host-RAM components (HiCache host pool, CPU offload).

use ec_hub::adapters::base::{EngineAdapter, FloatMap, Hardware, ModelInfo};
use ec_hub::adapters::sglang::SglangAdapter;
use serde_json::{Value, json};

use crate::ws_a1_kit::{get, model, obj};

const GIB: i64 = 1024 * 1024 * 1024;
const KEY: &str = "hicache_host_pool (assumed per TP rank)";

fn hw(n: usize) -> Hardware {
    Hardware {
        gpu_names: vec!["GPU".into(); n],
        ..crate::ws_a1_kit::hw_n(n, "12.0", None)
    }
}

fn mi() -> ModelInfo {
    model("Qwen/Qwen3.6-35B-A3B")
}

fn hm(m: &ModelInfo, p: Value, h: &Hardware) -> FloatMap {
    SglangAdapter::new().host_memory_gib(m, &obj(p), h)
}

#[test]
fn test_defaults_need_no_extra_host_ram_sglang() {
    assert!(hm(&mi(), json!({}), &hw(1)).is_empty());
    // hicache options without the feature: ignored
    assert!(hm(&mi(), json!({"hicache_size": 16}), &hw(1)).is_empty());
}

#[test]
fn test_hicache_size_overrides_ratio_and_scales_with_tp_sglang() {
    let out = hm(
        &mi(),
        json!({"enable_hierarchical_cache": true, "hicache_size": 16, "hicache_ratio": 9.0, "tp_size": 2}),
        &hw(2),
    );
    assert_eq!(out, vec![(KEY.to_string(), 32.0)]);
}

#[test]
fn test_hicache_ratio_is_relative_to_the_device_kv_pool_sglang() {
    let m = ModelInfo {
        weight_bytes: Some(40 * GIB),
        ..mi()
    };
    let out = hm(
        &m,
        json!({"enable_hierarchical_cache": true, "hicache_ratio": 2.0, "mem_fraction_static": 0.8}),
        &hw(1),
    );
    let pool = 0.8 * 95.0 - 40.0; // device KV pool on one GPU
    assert!((get(&out, KEY).unwrap() - 2.0 * pool).abs() <= 1e-9 * pool);
    let tp2 = hm(
        &m,
        json!({"enable_hierarchical_cache": true, "tp_size": 2, "mem_fraction_static": 0.8}),
        &hw(2),
    );
    let want = 2.0 * (0.8 * 95.0 - 20.0) * 2.0; // default ratio 2.0
    assert!((get(&tp2, KEY).unwrap() - want).abs() <= 1e-9 * want);
}

#[test]
fn test_hicache_ratio_without_weight_size_is_unknown_not_guessed_sglang() {
    let out = hm(&mi(), json!({"enable_hierarchical_cache": true}), &hw(1));
    assert!(get(&out, KEY).unwrap().is_nan());
}

#[test]
fn test_cpu_offload_scales_with_tp_sglang() {
    assert_eq!(
        hm(&mi(), json!({"cpu_offload_gb": 8, "tp_size": 2}), &hw(2)),
        vec![("cpu_weight_offload".to_string(), 16.0)]
    );
}

#[test]
fn test_params_that_change_host_ram_re_trigger_the_fit_sglang() {
    let cat = SglangAdapter::new().param_catalog();
    for k in [
        "cpu_offload_gb",
        "enable_hierarchical_cache",
        "hicache_ratio",
        "hicache_size",
    ] {
        assert!(
            cat.iter().find(|p| p.key == k).unwrap().affects_memory,
            "{k}"
        );
    }
}
