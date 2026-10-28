//! Port of backend/tests/vllm/test_help_snapshot.py: the vLLM catalog against what the pinned image itself reports
//! (offline: committed fixtures, no docker).

use std::collections::BTreeSet;

use ec_hub::adapters::base::{EngineAdapter, ParamSpec};
use ec_hub::adapters::vllm::{REASONING_HINTS, TOOL_DIALECTS, VllmAdapter};
use ec_hub::adapters::vllm_params::{
    ATTENTION_BACKENDS, ATTENTION_KV_DTYPES, ATTENTION_NOT_ON_SM120, CLI_DOCS, NARGS_KEYS,
    NEGATABLE_KEYS, NOT_IN_V0_30_0, QUANTIZATION, REASONING_PARSERS, TOOL_CALL_PARSERS, doc_anchor,
};
use serde_json::Value;

use crate::ws_a1_kit::fixture_json;

/// Quantization entries of the 0.27.1 registry that are left out on purpose (AMD / Intel only).
const OMITTED_QUANT: &[&str] = &["quark", "inc"];

fn help_flags() -> Value {
    fixture_json("vllm", "help_flags_v0.27.1.json")["flags"].clone()
}

fn registries() -> Value {
    fixture_json("vllm", "registries_v0.27.1.json")
}

fn cli() -> Vec<ParamSpec> {
    VllmAdapter::new()
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

#[test]
fn test_every_flag_exists_in_the_pinned_image_vllm() {
    let h = help_flags();
    let missing: Vec<String> = cli()
        .iter()
        .filter(|p| h.get(&p.flag).is_none())
        .map(|p| format!("{}: {}", p.key, p.flag))
        .collect();
    assert!(
        missing.is_empty(),
        "flags that `vllm serve --help=all` of v0.27.1 does not list: {missing:?}"
    );
}

#[test]
fn test_flags_in_the_other_local_image_vllm() {
    // `set(doc["flags"])`: the v0.30.0 file lists the flags as a plain list
    let other =
        crate::ws_a1_kit::flag_names(&fixture_json("vllm", "help_flags_v0.30.0.json")["flags"]);
    let mut absent: Vec<String> = cli()
        .into_iter()
        .filter(|p| !other.contains(&p.flag))
        .map(|p| p.flag)
        .collect();
    absent.sort();
    let mut want: Vec<String> = NOT_IN_V0_30_0.iter().map(|s| (*s).to_string()).collect();
    want.sort();
    assert_eq!(
        absent, want,
        "NOT_IN_V0_30_0 disagrees with the v0.30.0 help"
    );
}

#[test]
fn test_enum_choices_equal_the_help_vllm() {
    let h = help_flags();
    for p in cli() {
        let listed = &h[&p.flag]["choices"];
        if p.type_ != "enum" || listed.as_array().is_none_or(Vec::is_empty) {
            continue;
        }
        let mut want = strs(listed);
        want.remove("None");
        let have: BTreeSet<String> = p.choices.clone().unwrap_or_default().into_iter().collect();
        assert_eq!(have, want, "{}", p.key);
    }
}

#[test]
fn test_bool_and_list_shapes_match_the_help_vllm() {
    let h = help_flags();
    for p in cli() {
        let f = &h[&p.flag];
        if p.type_ == "bool" {
            assert_eq!(
                NEGATABLE_KEYS.contains(&p.key.as_str()),
                f["negatable"].as_bool().unwrap(),
                "{}: --no-<flag> form",
                p.key
            );
        }
        let nargs = f["nargs"].as_str();
        assert_eq!(
            NARGS_KEYS.contains(&p.key.as_str()),
            matches!(nargs, Some("+" | "*")),
            "{}: several values per flag",
            p.key
        );
    }
}

#[test]
fn test_defaults_match_the_help_vllm() {
    let h = help_flags();
    for p in cli() {
        let shown = &h[&p.flag]["default"];
        // mamba_backend prints MambaBackendEnum.TRITON
        if p.default.is_null() || shown.is_null() || p.key == "mamba_backend" {
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
            other => assert_eq!(ec_proto::pyfmt::str_value(other), shown, "{}", p.key),
        }
    }
}

#[test]
fn test_docs_anchors_follow_the_flag_table_vllm() {
    let h = help_flags();
    for p in cli() {
        let Some(url) = p.docs_url.as_deref().filter(|u| u.starts_with(CLI_DOCS)) else {
            continue; // a feature page (tool calling, spec decode)
        };
        let f = &h[&p.flag];
        let negatable = f["negatable"].as_bool().unwrap();
        let name = &p.flag[2..];
        let mut want = format!("-{name}");
        if negatable {
            want.push_str(&format!("-no-{name}"));
        }
        let shorts: Vec<&str> = f["aliases"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(Value::as_str)
            .filter(|a| !a.starts_with("--"))
            .collect();
        if let Some(s) = shorts.first() {
            want.push('-');
            want.push_str(s.trim_start_matches('-'));
        }
        assert_eq!(url, format!("{CLI_DOCS}#{want}"), "{}", p.key);
        assert_eq!(doc_anchor(&p.flag, negatable), want);
    }
}

#[test]
fn test_registry_lists_match_the_image_vllm() {
    let r = registries();
    let mut methods = strs(&r["quantization_methods"]);
    let q: BTreeSet<String> = QUANTIZATION.iter().map(|s| (*s).to_string()).collect();
    for o in OMITTED_QUANT {
        methods.remove(*o);
    }
    assert_eq!(q, methods);
    assert!(!QUANTIZATION.contains(&"gguf") && !strs(&r["quantization_methods"]).contains("gguf"));
    let parsers: Vec<&str> = r["reasoning_parsers"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(Value::as_str)
        .collect();
    assert_eq!(parsers, REASONING_PARSERS);
    for p in ["qwen3", "nemotron_v3", "openai_gptoss"] {
        assert!(REASONING_PARSERS.contains(&p));
    }
}

#[test]
fn test_attention_facts_match_the_image_vllm() {
    let r = registries();
    let att = &r["attention_backends"];
    for b in ATTENTION_BACKENDS.iter().filter(|b| **b != "auto") {
        assert!(att.get(*b).is_some(), "{b}");
        assert_eq!(
            att[*b]["supports_cc_12_0"].as_bool().unwrap(),
            !ATTENTION_NOT_ON_SM120.contains(b),
            "{b}"
        );
    }
    for (b, dtypes) in ATTENTION_KV_DTYPES {
        let want: BTreeSet<String> = dtypes.iter().map(|s| (*s).to_string()).collect();
        assert_eq!(strs(&att[*b]["kv_cache_dtypes"]), want, "{b}");
        assert!(
            att[*b]["supports_cc_12_0"].as_bool().unwrap(),
            "{b} is in the sm_120 table but the image says it is not supported"
        );
    }
}

#[test]
fn test_env_params_use_variables_the_image_reads_vllm() {
    let r = registries();
    let names: BTreeSet<String> = VllmAdapter::new()
        .param_catalog()
        .into_iter()
        .filter_map(|p| p.flag.strip_prefix("env:").map(str::to_string))
        .collect();
    let want: BTreeSet<String> = [
        "VLLM_API_KEY",
        "VLLM_ALLOW_LONG_MAX_MODEL_LEN",
        "VLLM_LOGGING_LEVEL",
        "VLLM_WORKER_MULTIPROC_METHOD",
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect();
    assert_eq!(names, want);
    assert!(names.is_subset(&strs(&r["env_present"])));
    assert!(names.is_disjoint(&strs(&r["env_absent"])));
}

#[test]
fn test_parser_choices_cover_the_dialect_tables_vllm() {
    for (_, allowed) in TOOL_DIALECTS.iter() {
        for t in *allowed {
            assert!(TOOL_CALL_PARSERS.contains(t), "{t}");
        }
    }
    for (_, parser) in REASONING_HINTS.iter() {
        assert!(REASONING_PARSERS.contains(parser), "{parser}");
    }
    for (_, preset) in VllmAdapter::new().presets() {
        for (key, val) in &preset {
            if key == "tool_call_parser" {
                assert!(TOOL_CALL_PARSERS.contains(&val.as_str().unwrap()));
            }
            if key == "reasoning_parser" {
                assert!(REASONING_PARSERS.contains(&val.as_str().unwrap()));
            }
        }
    }
}

#[test]
fn test_aliased_parsers_are_one_class_in_the_image_vllm() {
    let r = registries();
    let classes = &r["tool_parser_classes"];
    // either name works for Qwen3.x XML tool calls
    assert_eq!(classes["qwen3_coder"], classes["qwen3_xml"]);
    assert_eq!(classes["glm45"], classes["glm47"]);
}
