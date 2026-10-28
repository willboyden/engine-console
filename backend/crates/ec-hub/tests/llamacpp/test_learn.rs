//! Port of backend/tests/llamacpp/test_learn.py (learn-mode content for llama.cpp: every parameter and preset
//! explained, in the shape learn/README.md asks for), plus the llama.cpp rows of backend/tests/learn/
//! test_learn_content.py (`test_param_coverage[llamacpp]`, `test_preset_coverage[llamacpp]`).
//!
//! `test_passes_the_repository_learn_gate_checks` imports the learn gate's check functions; in Rust those live in the
//! CORE test binary tests/test_learn_content.rs, whose test_limits, test_hygiene, test_links_resolve and
//! test_jargon_is_linked run them over the whole bundle, this engine's files included (backend/TEST-MAP.md maps the row
//! there).

use std::collections::BTreeSet;

use crate::a2::engine_learn::{
    explanations_follow_the_readme, levels_match, no_personal_or_lab_specific_content,
    terms_from_the_starter_glossary_or_new, text,
};
use crate::a2::learn_section;
use ec_hub::adapters::base::EngineAdapter;
use ec_hub::adapters::llamacpp::LlamaCppAdapter;
use serde_json::Value;

/// api_key and llama_cpp already exist in the glossary
const NEW_TERMS: &[&str] = &["slot", "cors", "api_key", "llama_cpp"];
const ENGINE: &str = "llamacpp";

fn catalog_keys() -> BTreeSet<String> {
    LlamaCppAdapter::new()
        .param_catalog()
        .into_iter()
        .map(|p| p.key)
        .collect()
}

fn preset_names() -> BTreeSet<String> {
    LlamaCppAdapter::new()
        .presets()
        .into_iter()
        .map(|(n, _)| n)
        .collect()
}

#[test]
fn test_every_parameter_has_an_explanation_and_no_leftovers_llamacpp() {
    let body: BTreeSet<String> = learn_section("params", ENGINE).keys().cloned().collect();
    let keys = catalog_keys();
    assert_eq!(
        body,
        keys,
        "{:?} {:?}",
        keys.difference(&body).collect::<Vec<_>>(),
        body.difference(&keys).collect::<Vec<_>>()
    );
}

#[test]
fn test_every_preset_has_an_explanation_and_no_leftovers_llamacpp() {
    let body: BTreeSet<String> = learn_section("presets", ENGINE).keys().cloned().collect();
    assert_eq!(body, preset_names());
}

/// `[params]` and `[presets]`.
#[test]
fn test_explanations_follow_the_readme_llamacpp() {
    explanations_follow_the_readme("params", ENGINE);
    explanations_follow_the_readme("presets", ENGINE);
}

/// `[params]` and `[presets]`.
#[test]
fn test_terms_come_from_the_starter_glossary_or_the_declared_new_ones_llamacpp() {
    terms_from_the_starter_glossary_or_new("params", ENGINE, NEW_TERMS);
    terms_from_the_starter_glossary_or_new("presets", ENGINE, NEW_TERMS);
}

#[test]
fn test_levels_match_the_catalog_llamacpp() {
    let catalog = LlamaCppAdapter::new().param_catalog();
    levels_match(&catalog, ENGINE);
    let basic: BTreeSet<&str> = catalog
        .iter()
        .filter(|p| !p.advanced)
        .map(|p| p.key.as_str())
        .collect();
    // the beginner-visible ones
    assert_eq!(
        basic,
        BTreeSet::from([
            "n_gpu_layers",
            "ctx_size",
            "parallel",
            "flash_attn",
            "reasoning",
            "api_key"
        ])
    );
}

#[test]
fn test_facts_that_were_measured_are_in_the_text_llamacpp() {
    let body = learn_section("params", ENGINE);
    let cors = text(&body, "cors_origins", "details");
    assert!(cors.contains("any web page") || cors.contains("any page you visit"));
    let key = text(&body, "api_key", "details");
    assert!(key.contains("metrics endpoint") && key.contains("stay open"));
    assert!(text(&body, "parallel", "details").contains("4 slots")); // "n_parallel is set to auto, using n_parallel = 4"
    assert!(text(&body, "fit", "risks").contains("switch the automatic fit off"));
    assert!(text(&body, "n_gpu_layers", "risks").contains("never reduced")); // fit leaves a user-set value alone
}

#[test]
fn test_no_personal_or_lab_specific_content_llamacpp() {
    no_personal_or_lab_specific_content(ENGINE);
}

// ---- backend/tests/learn/test_learn_content.py, the llama.cpp parameter of its per-engine rows ----
/// `test_param_coverage[llamacpp]`.
#[test]
fn test_param_coverage_llamacpp() {
    let bundle = ec_hub::learn::bundle().expect("learn bundle");
    let have = bundle
        .params
        .get(ENGINE)
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let wanted: Vec<String> = LlamaCppAdapter::new()
        .param_catalog()
        .into_iter()
        .map(|p| p.key)
        .collect();
    let missing: Vec<&String> = wanted.iter().filter(|k| !have.contains_key(*k)).collect();
    let stale: Vec<&String> = have.keys().filter(|k| !wanted.contains(k)).collect();
    assert!(
        missing.is_empty() && stale.is_empty(),
        "missing {missing:?}; stale {stale:?}"
    );
}

/// `test_preset_coverage[llamacpp]`.
#[test]
fn test_preset_coverage_llamacpp() {
    let bundle = ec_hub::learn::bundle().expect("learn bundle");
    let have: BTreeSet<String> = bundle
        .presets
        .get(ENGINE)
        .and_then(Value::as_object)
        .map(|o| o.keys().cloned().collect())
        .unwrap_or_default();
    assert_eq!(have, preset_names());
}
