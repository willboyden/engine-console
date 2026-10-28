//! Port of backend/tests/trtllm/test_trtllm_learn.py (learn-mode content for the TensorRT-LLM adapter: every parameter
//! and preset is explained, in the house style), plus the TensorRT-LLM rows of backend/tests/learn/test_learn_content.py
//! (`test_param_coverage[trtllm]`, `test_preset_coverage[trtllm]`).

use std::collections::BTreeSet;

use crate::a2::{STARTER_TERMS, learn_dir, learn_section};
use ec_hub::adapters::base::{EngineAdapter, Explain};
use ec_hub::adapters::trtllm::TrtllmAdapter;
use ec_hub::adapters::trtllm_parse::py;
use regex::Regex;
use serde_json::{Map, Value};

/// none: the explanations use starter terms only
const NEW_TERMS: &[&str] = &[];
const LIMITS: [(&str, usize); 5] = [
    ("summary", 160),
    ("details", 900),
    ("when_to_change", 400),
    ("risks", 400),
    ("example", 200),
];

fn params() -> Map<String, Value> {
    learn_section("params", "trtllm")
}

fn presets() -> Map<String, Value> {
    learn_section("presets", "trtllm")
}

fn catalog_keys() -> Vec<String> {
    TrtllmAdapter::new()
        .param_catalog()
        .into_iter()
        .map(|p| p.key)
        .collect()
}

fn preset_names() -> Vec<String> {
    let mut names: Vec<String> = TrtllmAdapter::new()
        .presets()
        .into_iter()
        .map(|(n, _)| n)
        .collect();
    names.sort();
    names
}

fn terms(entry: &Value) -> Vec<String> {
    entry["terms"]
        .as_array()
        .expect("terms")
        .iter()
        .map(|t| t.as_str().expect("term").to_string())
        .collect()
}

#[test]
fn test_every_parameter_has_an_explanation_trtllm() {
    let have: BTreeSet<String> = params().keys().cloned().collect();
    let keys: BTreeSet<String> = catalog_keys().into_iter().collect();
    let missing: Vec<&String> = keys.difference(&have).collect();
    let stale: Vec<&String> = have.difference(&keys).collect();
    assert!(
        missing.is_empty() && stale.is_empty(),
        "missing {missing:?}, stale {stale:?}"
    );
    assert_eq!(params().len(), catalog_keys().len()); // sorted(params) == sorted(keys): no duplicates either
}

#[test]
fn test_every_preset_has_an_explanation_trtllm() {
    let have: Vec<String> = presets().keys().cloned().collect(); // already sorted (learn_section checks it)
    assert_eq!(have, preset_names());
}

/// `(?<=[.!?])\s+(?=[A-Z0-9`'"(])`: a sentence break inside the summary (look-arounds written as one match).
fn sentence_break(text: &str) -> bool {
    Regex::new(r#"[.!?]\s+[A-Z0-9`'"(]"#)
        .expect("re")
        .is_match(text)
}

fn style_guide(section: &str, body: &Map<String, Value>) {
    for (key, entry) in body {
        let explain: Explain = serde_json::from_value(entry.clone())
            .unwrap_or_else(|e| panic!("{section}/{key}: {e}"));
        for (field, limit) in LIMITS {
            let text = match field {
                "summary" => &explain.summary,
                "details" => &explain.details,
                "when_to_change" => &explain.when_to_change,
                "risks" => &explain.risks,
                _ => &explain.example,
            };
            assert!(!py::strip(text).is_empty(), "{key}.{field} is empty");
            let n = text.chars().count();
            assert!(n <= limit, "{key}.{field} is {n} chars (limit {limit})");
            assert!(
                text == py::strip(text) && !text.contains("  ") && !text.contains('\n'),
                "{key}.{field}: whitespace"
            );
        }
        assert!(explain.summary.ends_with('.'), "{key}");
        assert!(
            !sentence_break(&explain.summary),
            "{key}: the summary must be one sentence"
        );
        assert!(["basic", "advanced", "expert"].contains(&explain.level.as_str()));
        let unique: BTreeSet<&String> = explain.terms.iter().collect();
        assert_eq!(unique.len(), explain.terms.len(), "{key}: duplicate terms");
    }
}

/// `test_entries_follow_the_style_guide[params]` and `[presets]`.
#[test]
fn test_entries_follow_the_style_guide_trtllm() {
    style_guide("params", &params());
    style_guide("presets", &presets());
}

#[test]
fn test_terms_are_starter_terms_or_declared_new_trtllm() {
    for body in [params(), presets()] {
        for (key, entry) in &body {
            let unknown: BTreeSet<String> = terms(entry)
                .into_iter()
                .filter(|t| {
                    !STARTER_TERMS.contains(&t.as_str()) && !NEW_TERMS.contains(&t.as_str())
                })
                .collect();
            assert!(
                unknown.is_empty(),
                "{key}: terms outside the starter list and NEW_TERMS: {unknown:?}"
            );
        }
    }
}

#[test]
fn test_terms_exist_in_the_glossary_trtllm() {
    let path = learn_dir().join("glossary.json");
    // Python skips while glossary.json is not written; it is committed, so a missing file fails here.
    let doc: Value = serde_json::from_str(&std::fs::read_to_string(&path).expect("glossary.json"))
        .expect("json");
    let glossary: BTreeSet<&String> = doc["terms"].as_object().expect("terms").keys().collect();
    for body in [params(), presets()] {
        for (key, entry) in &body {
            let gone: Vec<String> = terms(entry)
                .into_iter()
                .filter(|t| !glossary.contains(t))
                .collect();
            assert!(gone.is_empty(), "{key}: {gone:?}");
        }
    }
}

/// The learn level is the catalog level (explicit, or advanced for parameters hidden behind 'Show advanced', basic for
/// the rest), so the form and the popover never disagree.
#[test]
fn test_levels_agree_with_the_catalog_trtllm() {
    let have = params();
    for p in TrtllmAdapter::new().param_catalog() {
        let expected = match p.level.as_deref() {
            Some(l) if !l.is_empty() => l.to_string(),
            _ if p.advanced => "advanced".into(),
            _ => "basic".into(),
        };
        assert_eq!(
            have[&p.key]["level"].as_str(),
            Some(expected.as_str()),
            "{}",
            p.key
        );
    }
}

#[test]
fn test_text_is_publish_safe_trtllm() {
    // json.dumps([params, presets]) escapes non-ASCII, so the patterns only ever see ASCII: (?-u) keeps \d and \b ASCII.
    let blob = serde_json::to_string(&Value::Array(vec![
        Value::Object(params()),
        Value::Object(presets()),
    ]))
    .expect("json");
    assert!(
        !Regex::new(r"/(?:home|Users)/|@[A-Za-z0-9-]+\.[a-z]{2,}")
            .expect("re")
            .is_match(&blob)
    );
    assert!(
        !Regex::new(r"(?-u:\b)(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?-u:\b)")
            .expect("re")
            .is_match(&blob)
    ); // no IPs
    assert!(
        !Regex::new(r"hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}")
            .expect("re")
            .is_match(&blob)
    );
}

/// A few claims that would rot silently if the code changed; the tests above pin the catalog side.
#[test]
fn test_explanations_state_facts_that_the_catalog_enforces_trtllm() {
    let p = params();
    let details = |k: &str| p[k]["details"].as_str().expect("details").to_string();
    assert!(details("free_gpu_memory_fraction").contains("0.9"));
    assert!(
        details("max_batch_size").contains("2048") && details("max_num_tokens").contains("8192")
    );
    assert!(details("tokens_per_block").contains("32"));
    assert!(details("moe_cluster_parallel_size").contains("ignored")); // the CLI flag serve() drops
    for name in ["qwen3_coder", "kimi_k2", "deepseek_v32"] {
        assert!(details("tool_parser").contains(name), "{name}");
    }
    for name in ["deepseek-r1", "qwen3", "nano-v3"] {
        assert!(details("reasoning_parser").contains(name), "{name}");
    }
}

// ---- backend/tests/learn/test_learn_content.py, the TensorRT-LLM parameter of its per-engine rows ----
/// `test_param_coverage[trtllm]`: every ParamSpec.key has an Explain in learn/params/trtllm.json, and no leftovers.
#[test]
fn test_param_coverage_trtllm() {
    let bundle = ec_hub::learn::bundle().expect("learn bundle");
    let have = bundle
        .params
        .get("trtllm")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let wanted = catalog_keys();
    let missing: Vec<&String> = wanted.iter().filter(|k| !have.contains_key(*k)).collect();
    let mut stale: Vec<&String> = have.keys().filter(|k| !wanted.contains(k)).collect();
    stale.sort();
    assert!(
        missing.is_empty() && stale.is_empty(),
        "learn/params/trtllm.json is missing {} of {} parameter explanation(s): {missing:?}; stale: {stale:?}",
        missing.len(),
        wanted.len()
    );
}

/// `test_preset_coverage[trtllm]`.
#[test]
fn test_preset_coverage_trtllm() {
    let bundle = ec_hub::learn::bundle().expect("learn bundle");
    let have = bundle
        .presets
        .get("trtllm")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let wanted = preset_names();
    let missing: Vec<&String> = wanted.iter().filter(|k| !have.contains_key(*k)).collect();
    let stale: Vec<&String> = have.keys().filter(|k| !wanted.contains(k)).collect();
    assert!(
        missing.is_empty() && stale.is_empty(),
        "missing {missing:?}, stale {stale:?}"
    );
}
