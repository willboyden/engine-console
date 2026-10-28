//! Port of backend/tests/sglang/test_sglang_snapshot.py: the SGLang catalog against what the pinned image itself
//! reports (offline: committed fixtures, no docker).

use std::collections::BTreeSet;

use ec_hub::adapters::base::{EngineAdapter, ParamSpec};
use ec_hub::adapters::sglang::SglangAdapter;
use ec_hub::adapters::sglang_params::{
    DOCS, ENV_DOCS, NOT_IN_V0_5_14, OMITTED_CHOICES, SECTIONS, SPEC_ALGOS, section_of,
};
use serde_json::Value;

use crate::ws_a1_kit::{fixture_json, flag_names};

fn snap() -> Value {
    fixture_json("sglang", "help_flags_v0.5.20.json")["flags"].clone()
}

fn find(snap: &Value, flag: &str) -> Option<Value> {
    if let Some(d) = snap.get(flag) {
        return Some(d.clone());
    }
    snap.as_object()?
        .values()
        .find(|d| {
            d["aliases"]
                .as_array()
                .is_some_and(|a| a.iter().any(|x| x == flag))
        })
        .cloned()
}

fn cli() -> Vec<ParamSpec> {
    SglangAdapter::new()
        .param_catalog()
        .into_iter()
        .filter(|p| !p.flag.starts_with("env:"))
        .collect()
}

fn strs(v: &Value) -> BTreeSet<String> {
    v.as_array()
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

fn omitted(flag: &str) -> BTreeSet<String> {
    OMITTED_CHOICES
        .iter()
        .find(|(f, _)| *f == flag)
        .map(|(_, o)| o.iter().map(|s| (*s).to_string()).collect())
        .unwrap_or_default()
}

#[test]
fn test_every_flag_exists_in_the_pinned_image_sglang() {
    let s = snap();
    let missing: Vec<String> = cli()
        .iter()
        .filter(|p| find(&s, &p.flag).is_none())
        .map(|p| format!("{}: {}", p.key, p.flag))
        .collect();
    assert!(
        missing.is_empty(),
        "flags that the 0.5.20 parser does not have: {missing:?}"
    );
}

#[test]
fn test_flags_in_the_other_local_image_sglang() {
    let other = flag_names(&fixture_json("sglang", "help_flags_v0.5.14.json")["flags"]);
    let mut absent: Vec<String> = cli()
        .into_iter()
        .filter(|p| !other.contains(&p.flag))
        .map(|p| p.flag)
        .collect();
    absent.sort();
    let mut want: Vec<String> = NOT_IN_V0_5_14.iter().map(|s| (*s).to_string()).collect();
    want.sort();
    assert_eq!(
        absent, want,
        "NOT_IN_V0_5_14 disagrees with the 0.5.14 parser"
    );
}

#[test]
fn test_choices_are_the_help_minus_the_recorded_omissions_sglang() {
    let s = snap();
    for p in cli() {
        let listed = find(&s, &p.flag)
            .map(|d| d["choices"].clone())
            .unwrap_or(Value::Null);
        if p.type_ != "enum" || listed.as_array().is_none_or(Vec::is_empty) {
            continue;
        }
        let om = omitted(&p.flag);
        let have: BTreeSet<String> = p.choices.clone().unwrap_or_default().into_iter().collect();
        let help = strs(&listed);
        assert!(have.is_disjoint(&om), "{}", p.key);
        let union: BTreeSet<String> = have.union(&om).cloned().collect();
        assert_eq!(union, help, "{}", p.key);
    }
}

#[test]
fn test_omitted_choices_all_exist_in_the_image_sglang() {
    let s = snap();
    for (flag, om) in OMITTED_CHOICES {
        let listed = strs(
            &find(&s, flag)
                .map(|d| d["choices"].clone())
                .unwrap_or(Value::Null),
        );
        for o in *om {
            assert!(
                listed.contains(*o),
                "{flag}: {o} is not in the image's list"
            );
        }
    }
}

#[test]
fn test_bool_and_list_shapes_match_the_parser_sglang() {
    let s = snap();
    for p in cli() {
        let h = find(&s, &p.flag).unwrap_or(Value::Null);
        if p.type_ == "bool" {
            assert!(
                h["store_true"].as_bool().unwrap_or(false),
                "{}: build_launch emits only the bare flag, so the action must be store_true",
                p.key
            );
        }
        if p.type_ == "string_list" {
            assert!(matches!(h["nargs"].as_str(), Some("*" | "+")), "{}", p.key);
        }
    }
}

#[test]
fn test_defaults_match_the_parser_sglang() {
    let s = snap();
    // the console forces enable_metrics on (the parser's own default is False)
    for p in cli() {
        let shown = find(&s, &p.flag)
            .map(|d| d["default"].clone())
            .unwrap_or(Value::Null);
        if p.default.is_null() || shown.is_null() || p.key == "enable_metrics" {
            continue;
        }
        let shown = shown.as_str().unwrap();
        match &p.default {
            Value::Bool(b) => assert_eq!(if *b { "True" } else { "False" }, shown, "{}", p.key),
            Value::Number(n) => assert_eq!(
                shown.parse::<f64>().unwrap(),
                n.as_f64().unwrap(),
                "{}",
                p.key
            ),
            other => assert_eq!(
                format!("'{}'", ec_proto::pyfmt::str_value(other)),
                shown,
                "{}",
                p.key
            ),
        }
    }
}

#[test]
fn test_speculative_algorithms_are_the_builtins_in_the_help_sglang() {
    assert_eq!(
        SPEC_ALGOS,
        [
            "EAGLE",
            "EAGLE3",
            "NEXTN",
            "STANDALONE",
            "NGRAM",
            "DFLASH",
            "DSPARK"
        ]
    );
}

#[test]
fn test_docs_sections_are_complete_and_known_sglang() {
    let keys: BTreeSet<String> = cli().into_iter().map(|p| p.key).collect();
    let section_keys: BTreeSet<String> = SECTIONS
        .iter()
        .flat_map(|(_, k)| k.iter().map(|s| (*s).to_string()))
        .collect();
    assert_eq!(section_keys, keys);
    let listed: usize = SECTIONS.iter().map(|(_, k)| k.len()).sum();
    assert_eq!(listed, section_keys.len()); // no key listed twice
    for p in cli() {
        assert_eq!(
            p.docs_url.as_deref(),
            Some(format!("{DOCS}#{}", section_of(&p.key).unwrap()).as_str())
        );
    }
    for p in SglangAdapter::new().param_catalog() {
        if p.flag.starts_with("env:") {
            assert_eq!(p.docs_url.as_deref(), Some(ENV_DOCS), "{}", p.key);
        }
    }
    // the heading ids of the page's second-level sections (read from its HTML on 2026-10-04)
    let page_sections = [
        "common-launch-commands",
        "model-and-tokenizer",
        "http-server",
        "quantization-and-data-type",
        "memory-and-scheduling",
        "runtime-options",
        "logging",
        "api-related",
        "data-parallelism",
        "multi-node-distributed-serving",
        "model-override-args",
        "lora",
        "kernel-backends-attention-sampling-grammar-gemm",
        "speculative-decoding",
        "moe",
        "mamba-cache",
        "hierarchical-cache",
        "offloading",
        "optimization/debug-options",
        "for-multi-modal",
        "encode-prefill-disaggregation",
    ];
    for (sec, _) in SECTIONS {
        assert!(page_sections.contains(sec), "{sec}");
    }
}

/// quantization marlin/qoq and attention cutlass_mla/dual_chunk_flash_attn were choices in 0.5.14, not in 0.5.20.
#[test]
fn test_removed_flags_are_really_gone_from_the_image_sglang() {
    let s = snap();
    let q = strs(&s["--quantization"]["choices"]);
    assert!(!q.contains("marlin") && !q.contains("qoq"));
    let att = strs(&s["--attention-backend"]["choices"]);
    assert!(!att.contains("cutlass_mla") && !att.contains("dual_chunk_flash_attn"));
    // a plain argument: no env-var form
    assert!(s.get("--api-key").is_some() && s["--api-key"]["nargs"].is_null());
}

#[test]
fn test_env_param_is_a_variable_the_image_defines_sglang() {
    let env: Vec<(String, String)> = SglangAdapter::new()
        .param_catalog()
        .into_iter()
        .filter(|p| p.flag.starts_with("env:"))
        .map(|p| (p.key, p.flag))
        .collect();
    assert_eq!(
        env,
        [(
            "allow_longer_context_len".to_string(),
            "env:SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN".to_string()
        )]
    );
}

/// --cuda-graph-max-bs is a 0.5.14 spelling that 0.5.20 dropped; the catalog uses the -decode form both images have.
#[test]
fn test_renamed_and_deprecated_spellings_sglang() {
    let s = snap();
    let old = flag_names(&fixture_json("sglang", "help_flags_v0.5.14.json")["flags"]);
    assert!(old.contains("--cuda-graph-max-bs") && s.get("--cuda-graph-max-bs").is_none());
    let flags: BTreeSet<String> = SglangAdapter::new()
        .param_catalog()
        .into_iter()
        .map(|p| p.flag)
        .collect();
    assert!(
        flags.contains("--cuda-graph-max-bs-decode") && old.contains("--cuda-graph-max-bs-decode")
    );
    assert!(!flags.contains("--cuda-graph-max-bs") && !flags.contains("--disable-cuda-graph"));
    assert!(s.get("--disable-cuda-graph").is_some()); // still accepted by 0.5.20, but deprecated there
}
