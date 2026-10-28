//! Port of backend/tests/ollama/test_learn.py (learn-mode content for Ollama: every parameter and preset explained, in
//! the shape learn/README.md asks for), plus the Ollama rows of backend/tests/learn/test_learn_content.py
//! (`test_param_coverage[ollama]`, `test_preset_coverage[ollama]`).
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
use ec_hub::adapters::ollama::OllamaAdapter;
use serde_json::Value;

/// api_key and llama_cpp already exist in the glossary
const NEW_TERMS: &[&str] = &["slot", "cors", "api_key", "llama_cpp"];
const ENGINE: &str = "ollama";

fn preset_names() -> BTreeSet<String> {
    OllamaAdapter::new()
        .presets()
        .into_iter()
        .map(|(n, _)| n)
        .collect()
}

#[test]
fn test_every_parameter_has_an_explanation_and_no_leftovers_ollama() {
    let body: BTreeSet<String> = learn_section("params", ENGINE).keys().cloned().collect();
    let keys: BTreeSet<String> = OllamaAdapter::new()
        .param_catalog()
        .into_iter()
        .map(|p| p.key)
        .collect();
    assert_eq!(body, keys);
}

#[test]
fn test_every_preset_has_an_explanation_and_no_leftovers_ollama() {
    let body: BTreeSet<String> = learn_section("presets", ENGINE).keys().cloned().collect();
    assert_eq!(body, preset_names());
}

/// `[params]` and `[presets]`.
#[test]
fn test_explanations_follow_the_readme_ollama() {
    explanations_follow_the_readme("params", ENGINE);
    explanations_follow_the_readme("presets", ENGINE);
}

/// `[params]` and `[presets]`.
#[test]
fn test_terms_come_from_the_starter_glossary_or_the_declared_new_ones_ollama() {
    terms_from_the_starter_glossary_or_new("params", ENGINE, NEW_TERMS);
    terms_from_the_starter_glossary_or_new("presets", ENGINE, NEW_TERMS);
}

#[test]
fn test_levels_match_the_catalog_ollama() {
    let catalog = OllamaAdapter::new().param_catalog();
    levels_match(&catalog, ENGINE);
    let basic: BTreeSet<&str> = catalog
        .iter()
        .filter(|p| !p.advanced)
        .map(|p| p.key.as_str())
        .collect();
    assert_eq!(
        basic,
        BTreeSet::from(["context_length", "keep_alive", "num_parallel"])
    );
}

#[test]
fn test_facts_from_the_docs_and_the_image_are_in_the_text_ollama() {
    let body = learn_section("params", ENGINE);
    let ctx = text(&body, "context_length", "details");
    // docs.ollama.com/context-length
    assert!(
        ctx.contains("4k tokens below 24 GiB")
            && ctx.contains("32k from 24 GiB")
            && ctx.contains("256k from 48 GiB")
    );
    // the FAQ example
    assert!(
        text(&body, "num_parallel", "details")
            .contains("2k context with 4 parallel requests behaves like an 8k context")
    );
    assert!(text(&body, "max_loaded_models", "summary").contains("3 per GPU"));
    assert!(text(&body, "kv_cache_type", "details").contains("q8_0 needs about half the memory"));
    assert!(text(&body, "max_transfer_streams", "details").contains("read-only"));
}

#[test]
fn test_no_personal_or_lab_specific_content_ollama() {
    no_personal_or_lab_specific_content(ENGINE);
}

// ---- backend/tests/learn/test_learn_content.py, the Ollama parameter of its per-engine rows ----
/// `test_param_coverage[ollama]`.
#[test]
fn test_param_coverage_ollama() {
    let bundle = ec_hub::learn::bundle().expect("learn bundle");
    let have = bundle
        .params
        .get(ENGINE)
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let wanted: Vec<String> = OllamaAdapter::new()
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

/// `test_preset_coverage[ollama]`.
#[test]
fn test_preset_coverage_ollama() {
    let bundle = ec_hub::learn::bundle().expect("learn bundle");
    let have: BTreeSet<String> = bundle
        .presets
        .get(ENGINE)
        .and_then(Value::as_object)
        .map(|o| o.keys().cloned().collect())
        .unwrap_or_default();
    assert_eq!(have, preset_names());
}
