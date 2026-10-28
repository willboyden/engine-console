//! Port of backend/tests/sglang/test_learn_content.py, plus the SGLang rows of
//! backend/tests/learn/test_learn_content.py (`test_param_coverage[sglang]`, `test_preset_coverage[sglang]`).

use std::collections::BTreeSet;

use ec_hub::adapters::base::{EngineAdapter, Explain};
use ec_hub::adapters::sglang::SglangAdapter;
use serde_json::{Map, Value};

use crate::ws_a1_kit::learn_file;

/// The glossary starter list the learn-content writer was given (the same list as vLLM's).
const STARTER_TERMS: &[&str] = &[
    "model",
    "weights",
    "parameters",
    "inference_engine",
    "serving",
    "gpu",
    "vram",
    "unified_memory",
    "host_ram",
    "shared_memory",
    "page_cache",
    "swap",
    "cgroup",
    "container",
    "docker_image",
    "image_tag",
    "gateway",
    "internal_network",
    "port",
    "endpoint",
    "openai_compatible_api",
    "token",
    "tokenizer",
    "context_window",
    "prompt",
    "prefill",
    "decode",
    "ttft",
    "itl",
    "throughput",
    "latency",
    "batch",
    "continuous_batching",
    "concurrency",
    "kv_cache",
    "paged_attention",
    "prefix_caching",
    "chunked_prefill",
    "quantization",
    "bf16",
    "fp16",
    "fp8",
    "nvfp4",
    "mxfp4",
    "int8",
    "int4",
    "awq",
    "gptq",
    "gguf",
    "gguf_quant_types",
    "safetensors",
    "hugging_face_hub",
    "hf_cache",
    "repo_id",
    "revision",
    "gated_model",
    "hf_token",
    "moe",
    "active_parameters",
    "dense_model",
    "tensor_parallelism",
    "pipeline_parallelism",
    "expert_parallelism",
    "data_parallelism",
    "nccl",
    "nvlink",
    "pcie",
    "peer_to_peer",
    "cuda",
    "cuda_graphs",
    "torch_compile",
    "compute_capability",
    "blackwell",
    "attention_backend",
    "flash_attention",
    "speculative_decoding",
    "draft_model",
    "mtp",
    "rope_scaling",
    "yarn",
    "sliding_window_attention",
    "gqa",
    "mla",
    "lora",
    "tool_calling",
    "reasoning_model",
    "chat_template",
    "trust_remote_code",
    "served_model_name",
    "cpu_offload",
    "kv_cache_offload",
    "gpu_layers_offload",
    "tensor_split",
    "mmap",
    "mlock",
    "cpu_threads",
    "numa",
    "fit_estimate",
    "preflight",
    "prometheus_metrics",
    "sampling",
    "temperature",
    "max_tokens",
    "embedding_model",
    "vision_language_model",
    "multimodal",
    "benchmark",
    "idle_ttl",
    "health_check",
    "warmup",
    "oom",
];
const NEW_TERMS: &[&str] = &[];

fn section(folder: &str, name: &str) -> Map<String, Value> {
    let doc = learn_file(folder, "sglang");
    let keys: BTreeSet<&str> = doc
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(keys, BTreeSet::from(["version", name]));
    doc[name].as_object().unwrap().clone()
}

fn catalog_keys() -> Vec<String> {
    SglangAdapter::new()
        .param_catalog()
        .into_iter()
        .map(|p| p.key)
        .collect()
}

#[test]
fn test_every_catalog_key_has_an_explanation_and_nothing_else_sglang() {
    let wanted: BTreeSet<String> = catalog_keys().into_iter().collect();
    let have: BTreeSet<String> = section("params", "params").keys().cloned().collect();
    let unexplained: Vec<&String> = wanted.difference(&have).collect();
    assert!(
        unexplained.is_empty(),
        "unexplained parameters: {unexplained:?}"
    );
    let stale: Vec<&String> = have.difference(&wanted).collect();
    assert!(
        stale.is_empty(),
        "explanations for parameters that no longer exist: {stale:?}"
    );
}

#[test]
fn test_every_preset_has_an_explanation_and_nothing_else_sglang() {
    let have: BTreeSet<String> = section("presets", "presets").keys().cloned().collect();
    let want: BTreeSet<String> = SglangAdapter::new()
        .presets()
        .into_iter()
        .map(|(n, _)| n)
        .collect();
    assert_eq!(have, want);
}

#[test]
fn test_terms_come_from_the_starter_list_or_the_declared_new_terms_sglang() {
    let mut offenders = Vec::new();
    for (folder, name) in [("params", "params"), ("presets", "presets")] {
        for (key, e) in section(folder, name) {
            let bad: Vec<String> = e["terms"]
                .as_array()
                .unwrap()
                .iter()
                .filter_map(Value::as_str)
                .filter(|t| !STARTER_TERMS.contains(t) && !NEW_TERMS.contains(t))
                .map(str::to_string)
                .collect();
            if !bad.is_empty() {
                offenders.push(format!("{folder}:{key}: {bad:?}"));
            }
        }
    }
    assert!(offenders.is_empty(), "{offenders:?}");
}

#[test]
fn test_explanation_level_matches_the_catalog_level_sglang() {
    let have = section("params", "params");
    for p in SglangAdapter::new().param_catalog() {
        assert_eq!(
            have[&p.key]["level"].as_str(),
            p.level.as_deref(),
            "{}",
            p.key
        );
    }
    let levels: BTreeSet<String> = section("presets", "presets")
        .values()
        .map(|e| e["level"].as_str().unwrap_or("").to_string())
        .collect();
    assert_eq!(levels, BTreeSet::from(["basic".to_string()]));
}

#[test]
fn test_every_explanation_is_complete_and_parses_sglang() {
    for (folder, name) in [("params", "params"), ("presets", "presets")] {
        for (key, raw) in section(folder, name) {
            let e: Explain = serde_json::from_value(raw).expect("Explain");
            assert!(
                e.summary.ends_with('.')
                    && !e.details.is_empty()
                    && !e.when_to_change.is_empty()
                    && !e.risks.is_empty(),
                "{folder}/{key}"
            );
        }
    }
    for (key, raw) in section("params", "params") {
        if raw["level"] == "basic" {
            assert!(
                raw.get("example")
                    .and_then(Value::as_str)
                    .is_some_and(|s| !s.is_empty()),
                "basic setting {key} needs a concrete example"
            );
        }
    }
}

#[test]
fn test_beginner_settings_say_what_breaks_sglang() {
    let have = section("params", "params");
    let basics: Vec<String> = SglangAdapter::new()
        .param_catalog()
        .into_iter()
        .filter(|p| p.level.as_deref() == Some("basic"))
        .map(|p| p.key)
        .collect();
    assert!(basics.len() >= 8);
    for key in basics {
        assert!(
            have[&key]["risks"].as_str().unwrap().chars().count() > 40,
            "{key}"
        );
    }
}

#[test]
fn test_the_hardware_facts_and_the_memory_semantics_are_in_the_text_sglang() {
    let have = section("params", "params");
    // the share applies to free memory
    assert!(
        have["mem_fraction_static"]["details"]
            .as_str()
            .unwrap()
            .contains("free when it starts")
    );
    // the 10 % rule of the image
    assert!(
        have["tp_size"]["details"]
            .as_str()
            .unwrap()
            .contains("unbalanced")
    );
    // fa3 is refused on sm_120
    assert!(
        have["attention_backend"]["details"]
            .as_str()
            .unwrap()
            .contains("SM>=80 and SM<=90")
    );
}

// ---- backend/tests/learn/test_learn_content.py, the SGLang parameter of its per-engine rows ----
/// `test_param_coverage[sglang]`.
#[test]
fn test_param_coverage_sglang() {
    let b = ec_hub::learn::bundle().expect("learn bundle");
    let have = b
        .params
        .get("sglang")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let wanted = catalog_keys();
    let missing: Vec<&String> = wanted.iter().filter(|k| !have.contains_key(*k)).collect();
    let stale: Vec<&String> = have.keys().filter(|k| !wanted.contains(k)).collect();
    assert!(
        missing.is_empty() && stale.is_empty(),
        "missing {missing:?}, stale {stale:?}"
    );
}

/// `test_preset_coverage[sglang]`.
#[test]
fn test_preset_coverage_sglang() {
    let b = ec_hub::learn::bundle().expect("learn bundle");
    let have = b
        .presets
        .get("sglang")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let wanted: Vec<String> = SglangAdapter::new()
        .presets()
        .into_iter()
        .map(|(n, _)| n)
        .collect();
    let missing: Vec<&String> = wanted.iter().filter(|k| !have.contains_key(*k)).collect();
    let stale: Vec<&String> = have.keys().filter(|k| !wanted.contains(k)).collect();
    assert!(
        missing.is_empty() && stale.is_empty(),
        "missing {missing:?}, stale {stale:?}"
    );
}
