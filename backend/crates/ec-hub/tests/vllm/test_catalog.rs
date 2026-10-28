//! Port of backend/tests/vllm/test_catalog.py.

use std::collections::BTreeSet;

use ec_hub::adapters::base::EngineAdapter;
use ec_hub::adapters::vllm::VllmAdapter;
use ec_hub::adapters::vllm_params::{DEFAULT_IMAGE, DOCS_BASE, NARGS_KEYS, NEGATABLE_KEYS};
use regex::Regex;

const UNITS: &[&str] = &[
    "GiB", "GB", "MiB", "bytes", "tokens", "requests", "GPUs", "s",
];

fn a() -> VllmAdapter {
    VllmAdapter::new()
}

#[test]
fn test_catalog_size_and_uniqueness_vllm() {
    let cat = a().param_catalog();
    assert!(cat.len() >= 100);
    assert_eq!(
        cat.iter().map(|p| &p.key).collect::<BTreeSet<_>>().len(),
        cat.len()
    );
    assert_eq!(
        cat.iter().map(|p| &p.flag).collect::<BTreeSet<_>>().len(),
        cat.len()
    );
}

#[test]
fn test_every_param_is_well_formed_vllm() {
    let key = Regex::new(r"^[a-z][a-z0-9_]*$").unwrap();
    let flag = Regex::new(r"^(?:--[a-z0-9][a-z0-9-]*|env:[A-Z0-9_]+)$").unwrap();
    for p in a().param_catalog() {
        assert!(key.is_match(&p.key), "{}", p.key);
        assert!(flag.is_match(&p.flag), "{}", p.flag);
        assert!(!p.help.trim().is_empty() && !p.label.trim().is_empty());
        assert!(
            p.docs_url
                .as_deref()
                .is_some_and(|d| d.starts_with(&format!("{DOCS_BASE}/"))),
            "{}",
            p.key
        );
        if p.type_ == "enum" {
            let choices = p.choices.clone().expect("enum choices");
            assert!(!choices.is_empty(), "{}", p.key);
            if !p.default.is_null() {
                assert!(
                    choices.contains(&ec_proto::pyfmt::str_value(&p.default)),
                    "{}",
                    p.key
                );
            }
        } else {
            assert!(p.choices.is_none(), "{}", p.key);
        }
        if let (Some(lo), Some(hi)) = (p.min, p.max) {
            assert!(lo <= hi);
        }
        if (p.type_ == "int" || p.type_ == "float")
            && let (Some(d), Some(lo)) = (p.default.as_f64(), p.min)
        {
            assert!(d >= lo, "{}", p.key);
        }
    }
}

#[test]
fn test_levels_and_units_vllm() {
    let cat = a().param_catalog();
    for p in &cat {
        let level = p.level.as_deref().unwrap_or("");
        assert!(
            ["basic", "advanced", "expert"].contains(&level),
            "{}",
            p.key
        );
        if level == "basic" {
            assert!(
                !p.advanced,
                "{}: a basic setting must not be hidden behind 'Show advanced'",
                p.key
            );
        }
        if level == "expert" {
            assert!(
                p.advanced,
                "{}: an expert setting belongs behind 'Show advanced'",
                p.key
            );
        }
        if let Some(u) = &p.unit {
            assert!(UNITS.contains(&u.as_str()), "{} {u}", p.key);
        }
    }
    let basic: BTreeSet<&str> = cat
        .iter()
        .filter(|p| p.level.as_deref() == Some("basic"))
        .map(|p| p.key.as_str())
        .collect();
    assert!((8..=14).contains(&basic.len()));
    for k in [
        "max_model_len",
        "gpu_memory_utilization",
        "tensor_parallel_size",
        "max_num_seqs",
        "quantization",
        "enable_auto_tool_choice",
        "tool_call_parser",
        "reasoning_parser",
    ] {
        assert!(basic.contains(k), "{k}");
    }
    let by = |k: &str| cat.iter().find(|p| p.key == k).unwrap().unit.clone();
    assert_eq!(by("max_model_len").as_deref(), Some("tokens"));
    assert_eq!(by("cpu_offload_gb").as_deref(), Some("GiB"));
    assert_eq!(by("kv_cache_memory_bytes").as_deref(), Some("bytes"));
    assert_eq!(by("max_num_seqs").as_deref(), Some("requests"));
}

#[test]
fn test_required_params_present_vllm() {
    let keys: BTreeSet<String> = a().param_catalog().into_iter().map(|p| p.key).collect();
    for k in [
        "gpu_memory_utilization",
        "max_model_len",
        "max_num_seqs",
        "max_num_batched_tokens",
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "data_parallel_size",
        "enable_expert_parallel",
        "quantization",
        "kv_cache_dtype",
        "dtype",
        "enable_prefix_caching",
        "enable_chunked_prefill",
        "block_size",
        "cpu_offload_gb",
        "enforce_eager",
        "cudagraph_capture_sizes",
        "tool_call_parser",
        "enable_auto_tool_choice",
        "reasoning_parser",
        "chat_template",
        "served_model_name",
        "speculative_config",
        "enable_lora",
        "lora_modules",
        "limit_mm_per_prompt",
        "hf_overrides",
        "seed",
        "api_key",
        "attention_backend",
        "mamba_cache_mode",
        "moe_backend",
        "linear_backend",
        "kv_offloading_size",
    ] {
        assert!(keys.contains(k), "{k}");
    }
}

#[test]
fn test_retired_keys_are_gone_vllm() {
    let cat = a().param_catalog();
    for k in ["image", "nvfp4_gemm_backend", "flashinfer_moe_backend"] {
        assert!(!cat.iter().any(|p| p.key == k));
    }
    assert!(!cat.iter().any(|p| p.flag == "@image"));
}

#[test]
fn test_all_groups_used_vllm() {
    let groups: BTreeSet<String> = a().param_catalog().into_iter().map(|p| p.group).collect();
    let want: BTreeSet<String> = [
        "memory",
        "parallelism",
        "context",
        "quantization",
        "scheduling",
        "tools_reasoning",
        "performance",
        "speculative",
        "network",
        "advanced",
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect();
    assert_eq!(groups, want);
}

#[test]
fn test_memory_params_flagged_vllm() {
    let cat = a().param_catalog();
    for k in [
        "gpu_memory_utilization",
        "max_model_len",
        "kv_cache_dtype",
        "tensor_parallel_size",
        "quantization",
        "max_num_seqs",
    ] {
        assert!(
            cat.iter().find(|p| p.key == k).unwrap().affects_memory,
            "{k}"
        );
    }
}

#[test]
fn test_helper_sets_reference_real_keys_vllm() {
    let cat = a().param_catalog();
    let by = |k: &str| {
        cat.iter()
            .find(|p| p.key == k)
            .unwrap_or_else(|| panic!("{k}"))
            .type_
            .clone()
    };
    for k in NEGATABLE_KEYS {
        assert_eq!(by(k), "bool", "{k}");
    }
    for k in NARGS_KEYS {
        assert_eq!(by(k), "string_list", "{k}");
    }
}

#[test]
fn test_default_image_pinned_vllm() {
    let a = a();
    assert!(a.default_image() == DEFAULT_IMAGE && DEFAULT_IMAGE == "vllm/vllm-openai:v0.27.1");
    assert!(!a.default_image().ends_with(":latest"));
}

#[test]
fn test_v2_adapter_fields_vllm() {
    let a = a();
    let mut f = a.model_formats();
    f.sort();
    assert_eq!(f, ["hf", "plain"]); // the pinned image has no GGUF loader
    assert!(a.images_by_arch().is_empty()); // every local vLLM image is amd64
    assert!(
        a.image_for("aarch64") == a.image_for("x86_64") && a.image_for("x86_64") == DEFAULT_IMAGE
    );
    assert_eq!(a.metric_prefixes(), Some(vec!["vllm:".to_string()]));
    assert!(!a.multi_model());
    assert!(a.preset_info().is_empty()); // preset text lives in learn/presets/vllm.json
}

#[test]
fn test_catalog_returns_copies_vllm() {
    let a = a();
    let mut first = a.param_catalog();
    first[0].label = "mutated".into();
    assert_ne!(a.param_catalog()[0].label, "mutated");
}
