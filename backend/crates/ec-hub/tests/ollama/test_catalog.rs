//! Port of backend/tests/ollama/test_catalog.py: the Ollama catalog against the captured help and the captured "server
//! config" start-up line of the pinned image (tests/data/ollama, read by `a2::ollama_help`, the port of helpers.py).

use std::collections::{BTreeMap, BTreeSet};

use crate::a2::ollama_help::{help_env, server_config};
use crate::a2::{p, read};
use ec_hub::adapters::base::{EngineAdapter, ParamSpec};
use ec_hub::adapters::ollama::OllamaAdapter;
use ec_hub::adapters::ollama_params::{
    CONTAINER_PORT, DEFAULT_IMAGE, IMAGE_DIGEST, KV_BYTES, KV_CACHE_TYPES, MODELS_DIR,
    RESERVED_ENV, VERSION,
};
use ec_hub::adapters::trtllm_parse::py::{self, PyVal};
use regex::Regex;
use serde_json::{Value, json};

fn adapter() -> OllamaAdapter {
    OllamaAdapter::new()
}

fn by_key() -> BTreeMap<String, ParamSpec> {
    adapter()
        .param_catalog()
        .into_iter()
        .map(|p| (p.key.clone(), p))
        .collect()
}

fn env_of() -> BTreeMap<String, ParamSpec> {
    adapter()
        .param_catalog()
        .into_iter()
        .map(|p| (p.flag[4..].to_string(), p))
        .collect()
}

fn reserved() -> BTreeSet<String> {
    RESERVED_ENV.iter().map(|(n, _)| (*n).to_string()).collect()
}

fn full(pattern: &str) -> Regex {
    Regex::new(&format!("^(?:{pattern})$")).expect("re")
}

#[test]
fn test_help_fixture_is_the_pinned_version_ollama() {
    assert!(VERSION == "0.30.11" && DEFAULT_IMAGE == "ollama/ollama:0.30.11");
    assert!(read("ollama", "endpoints_bare_0.30.11.txt").contains(r#"{"version":"0.30.11"}"#));
    assert!(read("ollama", "serve_bare_0.30.11.log").contains("version 0.30.11"));
    assert_eq!(help_env().len(), 21); // the parser reads the whole list
}

#[test]
fn test_catalog_is_the_help_minus_what_the_adapter_owns_ollama() {
    let listed: BTreeSet<String> = help_env().into_keys().collect();
    let exposed: BTreeSet<String> = env_of().into_keys().collect();
    let res = reserved();
    assert!(
        exposed.union(&res).cloned().collect::<BTreeSet<_>>() == listed
            && exposed.is_disjoint(&res)
    );
    assert!(exposed.len() == 17 && RESERVED_ENV.len() == 4);
    let want: BTreeSet<String> = [
        "OLLAMA_HOST",
        "OLLAMA_MODELS",
        "OLLAMA_NO_CLOUD",
        "OLLAMA_NOPRUNE",
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect();
    assert_eq!(res, want);
}

#[test]
fn test_catalog_size_and_uniqueness_ollama() {
    let cat = adapter().param_catalog();
    let keys: BTreeSet<&String> = cat.iter().map(|p| &p.key).collect();
    let flags: BTreeSet<&String> = cat.iter().map(|p| &p.flag).collect();
    assert!(keys.len() == flags.len() && flags.len() == cat.len());
}

#[test]
fn test_every_param_is_well_formed_ollama() {
    let key_re = full(r"[a-z][a-z0-9]*(_[a-z0-9]+)*");
    let flag_re = full(r"env:[A-Z0-9_]+"); // Ollama is configured through the environment only
    for p in adapter().param_catalog() {
        assert!(key_re.is_match(&p.key), "{}", p.key);
        assert!(flag_re.is_match(&p.flag), "{}", p.flag);
        assert!(
            !py::strip(&p.help).is_empty()
                && !py::strip(&p.label).is_empty()
                && p.help.ends_with('.')
        );
        assert!(
            p.docs_url
                .as_deref()
                .is_some_and(|d| d.starts_with("https://docs.ollama.com/"))
        );
        assert!(p.requires_restart);
        if p.type_ == "enum" {
            let choices = p.choices.as_ref().unwrap_or_else(|| panic!("{}", p.key));
            assert!(!choices.is_empty(), "{}", p.key);
            if !p.default.is_null() {
                assert!(
                    choices.contains(&PyVal::from_json(&p.default).py_str()),
                    "{}",
                    p.key
                );
            }
        } else {
            assert!(p.choices.is_none(), "{}", p.key);
        }
        if (p.type_ == "int" || p.type_ == "float")
            && let (Some(d), Some(lo)) = (p.default.as_f64(), p.min)
        {
            assert!(d >= lo, "{}", p.key);
        }
        let units = ["tokens", "models", "bytes", "MiB", "requests"];
        assert!(
            p.unit.as_deref().is_none_or(|u| units.contains(&u)),
            "{}",
            p.key
        );
        let level = p.level.as_deref();
        assert!(
            matches!(level, None | Some("expert")) && (level != Some("expert") || p.advanced),
            "{}",
            p.key
        );
    }
}

/// The core replaces any string value under a key matching key|token|secret|password and keeps it out of argv.
#[test]
fn test_no_param_key_looks_like_a_secret_ollama() {
    let secret = Regex::new(r"(?i)(key|token|secret|password)").expect("re");
    let found: Vec<String> = adapter()
        .param_catalog()
        .into_iter()
        .map(|p| p.key)
        .filter(|k| secret.is_match(k))
        .collect();
    assert!(found.is_empty(), "{found:?}");
}

#[test]
fn test_reserved_variables_are_never_parameters_ollama() {
    let env = env_of();
    for (name, _) in RESERVED_ENV {
        let mut params = serde_json::Map::new();
        params.insert(name.to_lowercase(), json!("x"));
        assert!(
            !env.contains_key(*name) && !adapter().validate(&params).is_empty(),
            "{name}"
        );
    }
    for key in [
        "host",
        "models",
        "no_cloud",
        "noprune",
        "ollama_host",
        "ollama_models",
    ] {
        assert_eq!(
            adapter().validate(&p(json!({key: "x"}))),
            vec![format!("unknown parameter '{key}'")]
        );
    }
}

#[test]
fn test_help_descriptions_back_the_catalog_text_ollama() {
    let helps = help_env();
    assert!(helps["OLLAMA_CONTEXT_LENGTH"].contains("4k/32k/256k based on VRAM"));
    assert!(
        helps["OLLAMA_KEEP_ALIVE"].contains("default \"5m\"")
            && helps["OLLAMA_LOAD_TIMEOUT"].contains("default \"5m\"")
    );
    assert!(helps["OLLAMA_MAX_LOADED_MODELS"].contains("per GPU"));
    assert!(helps["OLLAMA_KV_CACHE_TYPE"].contains("(default: f16)"));
    assert!(helps["OLLAMA_GPU_OVERHEAD"].contains("bytes"));
    assert!(helps["OLLAMA_MAX_TRANSFER_STREAMS"].contains("(default 4)"));
    assert!(
        helps["LLAMA_ARG_FIT"].contains("default \"on\"")
            && helps["LLAMA_ARG_FIT_TARGET"].contains("MiB")
    );
    assert!(helps["OLLAMA_ORIGINS"].contains("comma separated"));
    assert!(
        helps["OLLAMA_NO_CLOUD"].contains("cloud") && helps["OLLAMA_NOPRUNE"].contains("prune")
    );
}

fn same_duration(a: &str, b: &str) -> bool {
    let seconds = |text: &str| -> f64 {
        Regex::new(r"(\d+(?:\.\d+)?)([smh])")
            .expect("re")
            .captures_iter(text)
            .map(|c| {
                let unit = match &c[2] {
                    "s" => 1.0,
                    "m" => 60.0,
                    _ => 3600.0,
                };
                py::float(&c[1]).expect("number") * unit
            })
            .sum()
    };
    seconds(a) == seconds(b)
}

fn py_int(s: &str) -> i64 {
    py::int_digits(s).expect("int").parse().expect("i64")
}

/// `msg="server config"` is the engine's own registry of effective values; every default the catalog states as a
/// literal is compared with it (an unset variable shows as an empty value there).
#[test]
fn test_defaults_match_the_server_config_line_of_the_pinned_image_ollama() {
    let cfg = server_config();
    let cat = env_of();
    for name in [
        "OLLAMA_CONTEXT_LENGTH",
        "OLLAMA_NUM_PARALLEL",
        "OLLAMA_MAX_LOADED_MODELS",
        "OLLAMA_MAX_QUEUE",
        "OLLAMA_MAX_TRANSFER_STREAMS",
        "OLLAMA_GPU_OVERHEAD",
    ] {
        assert_eq!(
            Some(py_int(&cfg[name])),
            cat[name].default.as_i64(),
            "{name}"
        );
    }
    let s = |v: &Value| PyVal::from_json(v).py_str();
    assert!(same_duration(
        &cfg["OLLAMA_KEEP_ALIVE"],
        &s(&cat["OLLAMA_KEEP_ALIVE"].default)
    ));
    assert!(same_duration(
        &cfg["OLLAMA_LOAD_TIMEOUT"],
        &s(&cat["OLLAMA_LOAD_TIMEOUT"].default)
    ));
    assert!(cfg["OLLAMA_SCHED_SPREAD"] == "false" && cat["OLLAMA_SCHED_SPREAD"].default == false);
    // INFO is the non-debug log level
    assert!(cfg["OLLAMA_DEBUG"] == "INFO" && cat["OLLAMA_DEBUG"].default == false);
    // unset means f16 / on (the help)
    assert!(cfg["OLLAMA_KV_CACHE_TYPE"].is_empty() && cat["OLLAMA_KV_CACHE_TYPE"].default == "f16");
    assert!(cfg["LLAMA_ARG_FIT"].is_empty() && cat["LLAMA_ARG_FIT"].default == "on");
}

/// Tri-state or computed: left as None so the form shows "engine decides" and no variable is set.
#[test]
fn test_defaults_the_config_line_cannot_state_ollama() {
    let cfg = server_config();
    let cat = env_of();
    // the FAQ: unset lets Ollama decide
    assert!(
        cfg["OLLAMA_FLASH_ATTENTION"] == "false" && cat["OLLAMA_FLASH_ATTENTION"].default.is_null()
    );
    for name in [
        "OLLAMA_IGPU_ENABLE",
        "OLLAMA_LLM_LIBRARY",
        "LLAMA_ARG_FIT_TARGET",
        "OLLAMA_ORIGINS",
    ] {
        assert!(cat[name].default.is_null(), "{name}");
    }
    let origins: Vec<&str> = cfg["OLLAMA_ORIGINS"]
        .trim_matches(['[', ']'])
        .split_whitespace()
        .collect();
    // not open to every website
    assert!(
        origins.contains(&"http://localhost")
            && origins.contains(&"app://*")
            && !origins.contains(&"*")
    );
}

/// The bare server was started with the same reserved variables the adapter sets.
#[test]
fn test_the_config_line_confirms_the_reserved_settings_took_effect_ollama() {
    let cfg = server_config();
    assert!(cfg["OLLAMA_NO_CLOUD"] == "true" && cfg["OLLAMA_NOPRUNE"] == "true");
    assert!(cfg["OLLAMA_MODELS"] == MODELS_DIR && MODELS_DIR == "/root/.ollama/models");
    assert!(read("ollama", "serve_bare_0.30.11.log").contains("Ollama cloud disabled: true"));
}

#[test]
fn test_kv_cache_types_match_the_help_and_the_docs_ollama() {
    let choices = by_key()["kv_cache_type"].choices.clone().expect("choices");
    let names: Vec<&str> = KV_BYTES.iter().map(|(k, _)| *k).collect();
    assert!(
        choices == KV_CACHE_TYPES
            && KV_CACHE_TYPES == ["f16", "q8_0", "q4_0"]
            && names == KV_CACHE_TYPES
    );
    assert_eq!(KV_BYTES, [("f16", 2.0), ("q8_0", 1.0625), ("q4_0", 0.5625)]);
    let kv = |n: &str| KV_BYTES.iter().find(|(k, _)| *k == n).expect("kv").1;
    // "about 1/2" and "about 1/4"
    assert!(0.5 < kv("q8_0") / kv("f16") && kv("q8_0") / kv("f16") < 0.55);
    assert!(0.25 < kv("q4_0") / kv("f16") && kv("q4_0") / kv("f16") < 0.3);
}

#[test]
fn test_the_env_flags_cover_the_documented_tri_states_ollama() {
    let cat = by_key();
    assert!(cat["flash_attention"].type_ == "bool" && cat["flash_attention"].default.is_null());
    assert!(cat["keep_alive"].type_ == "string" && cat["load_timeout"].type_ == "string");
    assert_eq!(cat["origins"].type_, "string_list");
}

#[test]
fn test_required_params_present_and_groups_ollama() {
    let keys: BTreeSet<String> = by_key().into_keys().collect();
    let want: BTreeSet<String> = [
        "context_length",
        "num_parallel",
        "max_loaded_models",
        "keep_alive",
        "kv_cache_type",
        "flash_attention",
        "gpu_overhead",
        "sched_spread",
        "max_queue",
        "load_timeout",
        "origins",
        "llm_library",
        "igpu_enable",
        "fit",
        "fit_target",
        "max_transfer_streams",
        "debug",
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect();
    assert_eq!(keys, want);
    let groups: BTreeSet<String> = adapter()
        .param_catalog()
        .into_iter()
        .map(|p| p.group)
        .collect();
    let want_groups: BTreeSet<String> = [
        "context",
        "memory",
        "scheduling",
        "parallelism",
        "performance",
        "network",
        "advanced",
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect();
    assert_eq!(groups, want_groups);
}

#[test]
fn test_memory_params_flagged_ollama() {
    let cat = by_key();
    for k in [
        "context_length",
        "num_parallel",
        "kv_cache_type",
        "max_loaded_models",
        "gpu_overhead",
        "sched_spread",
        "flash_attention",
    ] {
        assert!(cat[k].affects_memory, "{k}");
    }
    assert!(!cat["keep_alive"].affects_memory && !cat["debug"].affects_memory);
}

#[test]
fn test_adapter_identity_and_pin_ollama() {
    let a = adapter();
    assert_eq!(
        (a.id(), a.display_name(), a.default_port()),
        ("ollama", "Ollama", 11434)
    );
    assert_eq!(CONTAINER_PORT, 11434);
    assert!(a.default_image() == "ollama/ollama:0.30.11" && !a.default_image().contains(":latest"));
    assert!(full(r"sha256:[0-9a-f]{64}").is_match(IMAGE_DIGEST));
    assert!(a.model_formats() == ["ollama"] && a.multi_model());
    assert_eq!(a.metric_prefixes(), Some(Vec::new())); // `()`: declared, and empty
    assert!(
        a.image_for("aarch64") == a.image_for("x86_64") && a.image_for("x86_64") == DEFAULT_IMAGE
    );
}

#[test]
fn test_catalog_returns_copies_ollama() {
    let a = adapter();
    let mut cat = a.param_catalog();
    cat[0].label = "mutated".into();
    assert_ne!(a.param_catalog()[0].label, "mutated");
    let mut pre = a.presets();
    pre.iter_mut()
        .find(|(n, _)| n == "balanced")
        .expect("balanced")
        .1
        .insert("context_length".into(), json!(1));
    assert_eq!(
        a.presets()
            .iter()
            .find(|(n, _)| n == "balanced")
            .expect("balanced")
            .1["context_length"],
        8192
    );
}
