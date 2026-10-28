//! Port of backend/tests/ollama/test_compat.py: compatibility() of the Ollama adapter, every rule, its level and the
//! fix it names.

use std::collections::{BTreeMap, BTreeSet};

use crate::a2::{hw_with, oll, p, rust_add_codes};
use ec_hub::adapters::base::{Compat, EngineAdapter, Hardware, ModelInfo};
use ec_hub::adapters::ollama::{OllamaAdapter, default_context};
use serde_json::{Value, json};

/// every code the adapter can emit (the learn gate needs a `check.<code>` concept for each; the report lists them)
const EXPECTED_CODES: &[&str] = &[
    "blackwell_cuda_build",
    "default_context_by_vram",
    "kv_exceeds_vram",
    "parallel_multiplies_context",
    "kv_quant_needs_flash_attn",
    "keep_alive_forever",
    "cpu_offload_slow",
];

fn codes(m: &ModelInfo, params: Value, hw: &Hardware) -> BTreeMap<String, Compat> {
    let found = OllamaAdapter::new().compatibility(m, &p(params), hw);
    let unique: BTreeSet<&String> = found.iter().map(|c| &c.code).collect();
    assert_eq!(unique.len(), found.len(), "a code was emitted twice");
    found.into_iter().map(|c| (c.code.clone(), c)).collect()
}

/// `params | {"context_length": 8192}`
fn with_ctx(mut params: Value) -> Value {
    params["context_length"] = json!(8192);
    params
}

#[test]
fn test_the_emitted_codes_are_exactly_the_documented_ones_ollama() {
    // Python walks the module's AST for add(<level>, "<code>", ...); the Rust twin scans ollama.rs the same way.
    let emitted: BTreeSet<String> = rust_add_codes("ollama.rs").into_iter().collect();
    let want: BTreeSet<String> = EXPECTED_CODES.iter().map(|c| (*c).to_string()).collect();
    assert_eq!(emitted, want);
}

/// docs.ollama.com/context-length: below 24 GiB 4k, 24-48 GiB 32k, from 48 GiB 256k.
#[test]
fn test_default_context_tiers_follow_the_docs_ollama() {
    assert!(
        [0.0, 8.0, 16.0, 23.9]
            .iter()
            .all(|g| default_context(*g) == 4096)
    );
    assert!(
        [24.0, 32.0, 47.9]
            .iter()
            .all(|g| default_context(*g) == 32768)
    );
    assert!(
        [48.0, 95.6, 191.2]
            .iter()
            .all(|g| default_context(*g) == 262_144)
    );
}

#[test]
fn test_a_large_gpu_gets_a_large_default_context_and_a_warning_ollama() {
    let (hw, m) = (oll::hw(), oll::llama8b());
    let c = &codes(&m, json!({}), &hw)["default_context_by_vram"];
    assert!(
        c.level == "warn" && c.message.contains("262144 tokens") && c.message.contains("96 GiB")
    );
    assert!(c.message.contains("32.0 GiB of KV cache per loaded model")); // 262144 x 128 KiB
    assert!(c.message.contains("context_length"));
    let parallel = &codes(&m, json!({"num_parallel": 4}), &hw)["default_context_by_vram"];
    assert!(parallel.message.contains("128.0 GiB")); // the product with the parallel requests
    let quant = &codes(&m, json!({"kv_cache_type": "q8_0"}), &hw)["default_context_by_vram"];
    assert!(quant.message.contains("17.0 GiB"));
}

#[test]
fn test_the_default_context_note_depends_on_the_tier_ollama() {
    let (hw, small, m) = (oll::hw(), oll::small_gpu(), oll::llama8b());
    assert!(!codes(&m, json!({}), &small).contains_key("default_context_by_vram")); // 16 GiB: 4k, harmless
    let mid = hw_with(
        &small,
        json!({"gpu_total_gib": [32.0], "gpu_free_gib": [30.0]}),
    );
    assert!(
        codes(&m, json!({}), &mid)["default_context_by_vram"]
            .message
            .contains("32768 tokens")
    );
    // chosen explicitly
    assert!(
        !codes(&m, json!({"context_length": 8192}), &hw).contains_key("default_context_by_vram")
    );
    let no_gpu = hw_with(
        &hw,
        json!({"gpu_ids": [], "gpu_uuids": [], "gpu_names": [], "gpu_total_gib": [], "gpu_free_gib": []}),
    );
    assert!(!codes(&m, json!({}), &no_gpu).contains_key("default_context_by_vram"));
}

#[test]
fn test_the_cost_is_left_out_when_the_model_facts_are_missing_ollama() {
    let c = &codes(&oll::bare(), json!({}), &oll::hw())["default_context_by_vram"];
    assert!(c.message.contains("262144 tokens") && !c.message.contains("GiB of KV cache"));
}

#[test]
fn test_two_gpus_add_up_for_the_tier_ollama() {
    assert!(
        codes(&oll::llama8b(), json!({}), &oll::hw2())["default_context_by_vram"]
            .message
            .contains("191 GiB")
    );
}

#[test]
fn test_an_explicit_context_that_will_not_fit_is_flagged_ollama() {
    let (small, m) = (oll::small_gpu(), oll::llama8b());
    let c = &codes(&m, json!({"context_length": 131_072}), &small)["kv_exceeds_vram"];
    assert!(
        c.level == "warn"
            && c.message.contains("16.0 GiB of KV cache")
            && c.message.contains("14.0 GiB")
    );
    assert!(c.message.contains("q8_0"));
    let q8 = json!({"context_length": 131_072, "kv_cache_type": "q8_0"});
    assert!(!codes(&m, q8, &small).contains_key("kv_exceeds_vram"));
    assert!(!codes(&m, json!({"context_length": 8192}), &small).contains_key("kv_exceeds_vram"));
    let parallel = &codes(
        &m,
        json!({"context_length": 32768, "num_parallel": 4}),
        &small,
    )["kv_exceeds_vram"];
    assert!(parallel.message.contains("x 4 request(s)")); // 32768 x 4 = 128k tokens = 16 GiB
}

#[test]
fn test_parallel_requests_multiply_the_context_ollama() {
    let (hw, m) = (oll::hw(), oll::llama8b());
    let c = &codes(&m, json!({"context_length": 8192, "num_parallel": 4}), &hw)["parallel_multiplies_context"];
    assert!(c.level == "ok" && c.message.contains("num_parallel=4"));
    assert!(
        !codes(&m, json!({"context_length": 8192}), &hw)
            .contains_key("parallel_multiplies_context")
    );
    assert!(
        !codes(&m, json!({"num_parallel": 1}), &hw).contains_key("parallel_multiplies_context")
    );
}

#[test]
fn test_quantized_cache_with_flash_attention_off_ollama() {
    let (hw, m) = (oll::hw(), oll::llama8b());
    for t in ["q8_0", "q4_0"] {
        let params = json!({"kv_cache_type": t, "flash_attention": false, "context_length": 8192});
        let c = &codes(&m, params, &hw)["kv_quant_needs_flash_attn"];
        assert!(
            c.level == "warn" && c.message.contains(t) && c.message.contains("flash_attention")
        );
    }
    for params in [
        json!({"kv_cache_type": "q8_0"}),
        json!({"kv_cache_type": "q8_0", "flash_attention": true}),
        json!({"kv_cache_type": "f16", "flash_attention": false}),
        json!({"flash_attention": false}),
    ] {
        assert!(
            !codes(&m, with_ctx(params.clone()), &hw).contains_key("kv_quant_needs_flash_attn"),
            "{params}"
        );
    }
}

#[test]
fn test_keep_alive_forever_ollama() {
    let (hw, m) = (oll::hw(), oll::llama8b());
    for v in [json!("-1"), json!("-1m"), json!(-1), json!(" -5 ")] {
        let c =
            &codes(&m, json!({"keep_alive": v, "context_length": 8192}), &hw)["keep_alive_forever"];
        assert!(c.level == "warn" && c.message.contains("other engines"));
    }
    for v in [json!("5m"), json!("0"), json!("24h"), json!(3600)] {
        assert!(
            !codes(&m, json!({"keep_alive": v, "context_length": 8192}), &hw)
                .contains_key("keep_alive_forever"),
            "{v}"
        );
    }
    assert!(!codes(&m, json!({"context_length": 8192}), &hw).contains_key("keep_alive_forever"));
}

#[test]
fn test_a_cpu_library_is_slow_ollama() {
    let (hw, m) = (oll::hw(), oll::llama8b());
    assert_eq!(
        codes(
            &m,
            json!({"llm_library": "cpu", "context_length": 8192}),
            &hw
        )["cpu_offload_slow"]
            .level,
        "warn"
    );
    assert!(
        !codes(
            &m,
            json!({"llm_library": "cuda_v13", "context_length": 8192}),
            &hw
        )
        .contains_key("cpu_offload_slow")
    );
}

#[test]
fn test_blackwell_note_only_on_compute_capability_12_ollama() {
    let (hw, small, m) = (oll::hw(), oll::small_gpu(), oll::llama8b());
    let c = &codes(&m, json!({"context_length": 8192}), &hw)["blackwell_cuda_build"];
    assert!(
        c.level == "ok" && c.message.contains("sm_120a") && c.message.contains("CUDA 12 and 13")
    );
    assert!(
        !codes(&m, json!({"context_length": 8192}), &small).contains_key("blackwell_cuda_build")
    );
}

#[test]
fn test_nothing_is_blocked_and_a_clean_launch_has_only_a_note_ollama() {
    let (a, hw, m) = (OllamaAdapter::new(), oll::hw(), oll::llama8b());
    let found = a.compatibility(
        &m,
        &p(json!({"context_length": 8192, "keep_alive": "30m"})),
        &hw,
    );
    assert!(
        found
            .iter()
            .map(|c| c.code.as_str())
            .eq(["blackwell_cuda_build"])
    );
    let worst = a.compatibility(
        &m,
        &p(json!({"kv_cache_type": "q4_0", "flash_attention": false, "keep_alive": "-1", "num_parallel": 8,
                  "llm_library": "cpu"})),
        &hw,
    );
    assert!(worst.iter().all(|c| c.level == "ok" || c.level == "warn")); // advice, never a hard stop
    assert!(
        worst
            .iter()
            .all(|c| !c.message.is_empty() && !c.code.is_empty())
    );
}

/// compatibility() may be asked about params that were not validated first.
#[test]
fn test_invalid_kv_type_does_not_crash_the_rules_ollama() {
    let found = OllamaAdapter::new()
        .try_compatibility(
            &oll::llama8b(),
            &p(json!({"kv_cache_type": "bogus", "context_length": 8192})),
            &oll::hw(),
        )
        .expect("never raises");
    assert!(!found.is_empty());
}
