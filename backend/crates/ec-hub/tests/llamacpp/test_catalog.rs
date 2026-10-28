//! Port of backend/tests/llamacpp/test_catalog.py: the llama.cpp catalog against the captured help of the pinned image
//! (tests/data/llamacpp/llama-server-help.txt, read by `a2::llama_help`, the port of helpers.py).

use std::collections::{BTreeMap, BTreeSet};

use crate::a2::llama_help::{entry_for, help, help_flags, parse_help};
use crate::a2::read;
use ec_hub::adapters::base::{EngineAdapter, ParamSpec};
use ec_hub::adapters::llamacpp::LlamaCppAdapter;
use ec_hub::adapters::llamacpp_params::{
    BUILD, CACHE_TYPE_BYTES, CACHE_TYPES, CHAT_TEMPLATES, COMMIT, DEFAULT_IMAGE, DOCS_URL,
    IMAGE_DIGEST, LAYER_KEYS, LOAD_MODES, NEGATABLE_KEYS, PATH_KEYS, SPEC_TYPES,
};
use ec_hub::adapters::trtllm_parse::py::{self, PyVal};
use regex::Regex;
use serde_json::Value;

fn adapter() -> LlamaCppAdapter {
    LlamaCppAdapter::new()
}

fn by_key() -> BTreeMap<String, ParamSpec> {
    adapter()
        .param_catalog()
        .into_iter()
        .map(|p| (p.key.clone(), p))
        .collect()
}

fn full(pattern: &str) -> Regex {
    Regex::new(&format!("^(?:{pattern})$")).expect("re")
}

fn strs(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| (*s).to_string()).collect()
}

#[test]
fn test_help_fixture_is_the_pinned_build_llamacpp() {
    let version = read("llamacpp", "version_b10156.txt");
    assert!(version.contains(&format!("version: {} ({COMMIT})", &BUILD[1..])));
    let entries = parse_help(&help());
    assert!(entries.len() > 250); // the parser really reads the whole help
    assert!(help_flags().contains("--n-gpu-layers") && help_flags().contains("--fit"));
}

#[test]
fn test_catalog_size_and_uniqueness_llamacpp() {
    let cat = adapter().param_catalog();
    assert!(cat.len() >= 75);
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
fn test_every_param_is_well_formed_llamacpp() {
    let key_re = full(r"[a-z][a-z0-9]*(_[a-z0-9]+)*");
    let flag_re = full(r"--[a-z0-9][a-z0-9-]*|env:[A-Z0-9_]+");
    for p in adapter().param_catalog() {
        assert!(key_re.is_match(&p.key), "{}", p.key);
        assert!(flag_re.is_match(&p.flag), "{}", p.flag);
        assert!(
            !py::strip(&p.help).is_empty()
                && !py::strip(&p.label).is_empty()
                && p.help.ends_with('.')
        );
        assert!(
            p.docs_url.as_deref() == Some(DOCS_URL) && DOCS_URL.contains(&format!("/{BUILD}/"))
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
        if let (Some(lo), Some(hi)) = (p.min, p.max) {
            assert!(lo <= hi);
        }
        if (p.type_ == "int" || p.type_ == "float")
            && let (Some(d), Some(lo)) = (p.default.as_f64(), p.min)
        {
            assert!(d >= lo, "{}", p.key);
        }
        if p.type_ == "bool" {
            assert!(p.default.is_null() || p.default.is_boolean());
        }
        let units = ["MiB", "tokens", "layers", "slots", "s", "threads"];
        assert!(
            p.unit.as_deref().is_none_or(|u| units.contains(&u)),
            "{}",
            p.key
        );
    }
}

#[test]
fn test_levels_are_consistent_llamacpp() {
    for p in adapter().param_catalog() {
        assert!(
            matches!(p.level.as_deref(), None | Some("expert")),
            "{}",
            p.key
        ); // basic/advanced follow `advanced`
        if p.level.as_deref() == Some("expert") {
            assert!(p.advanced, "{}", p.key);
        }
    }
}

#[test]
fn test_every_flag_exists_in_the_pinned_help_llamacpp() {
    let known = help_flags();
    let missing: Vec<String> = adapter()
        .param_catalog()
        .iter()
        .filter(|p| !p.flag.starts_with("env:") && !known.contains(&p.flag))
        .map(|p| format!("{}: {}", p.key, p.flag))
        .collect();
    assert!(missing.is_empty(), "{missing:?}");
}

#[test]
fn test_env_flags_are_documented_in_the_help_llamacpp() {
    let envs: BTreeSet<String> = parse_help(&help())
        .into_iter()
        .filter_map(|e| e.env)
        .collect();
    for p in adapter().param_catalog() {
        if let Some(name) = p.flag.strip_prefix("env:") {
            assert!(envs.contains(name), "{}", p.key);
        }
    }
    assert_eq!(by_key()["api_key"].flag, "env:LLAMA_API_KEY");
}

#[test]
fn test_negations_exist_in_the_pinned_help_llamacpp() {
    let known = help_flags();
    let cat = by_key();
    for key in NEGATABLE_KEYS {
        assert_eq!(cat[*key].type_, "bool", "{key}");
        assert!(
            known.contains(&format!("--no-{}", &cat[*key].flag[2..])),
            "{key}"
        );
    }
    // a bool that is NOT negatable must not have a negation we forgot about
    for p in cat.values() {
        if p.type_ == "bool" && !NEGATABLE_KEYS.contains(&p.key.as_str()) {
            assert!(
                !known.contains(&format!("--no-{}", &p.flag[2..])),
                "{}",
                p.key
            );
        }
    }
}

fn help_default(spec: &ParamSpec) -> Option<String> {
    if spec.flag.starts_with("env:") {
        None
    } else {
        entry_for(&spec.flag).default
    }
}

/// `_same`: the help's literal against the catalog default.
fn same(spec: &ParamSpec, literal: &str) -> bool {
    let low = literal.to_lowercase();
    if spec.type_ == "bool" {
        let b = match low.as_str() {
            "enabled" | "prefill enabled" | "true" => Some(true),
            "disabled" | "false" => Some(false),
            _ => None,
        };
        // `{...}.get(low) is spec.default`: None is never a bool default here (the caller skips None defaults)
        return b.is_some() && spec.default.as_bool() == b;
    }
    if spec.type_ == "int" || spec.type_ == "float" {
        let (Some(a), Some(b)) = (py::float(&low), spec.default.as_f64()) else {
            return false;
        };
        // math.isclose(a, b, rel_tol=1e-9)
        return a == b || (a - b).abs() <= 1e-9 * a.abs().max(b.abs());
    }
    low == PyVal::from_json(&spec.default).py_str().to_lowercase()
}

/// The console deliberately differs from the engine here (documented in the param's help and the adapter docstring).
const DEFAULT_DIFFERS: &[&str] = &["cors_origins"];

/// Every default the help states as a literal equals the catalog default; the rest are computed or unstated.
#[test]
fn test_literal_defaults_match_the_pinned_help_llamacpp() {
    let literal_re = full(r"-?[\d.]+|enabled|prefill enabled|disabled|true|false|[a-z0-9]+");
    let mut checked = 0;
    for p in adapter().param_catalog() {
        let Some(literal) = help_default(&p) else {
            continue;
        };
        if p.default.is_null() || DEFAULT_DIFFERS.contains(&p.key.as_str()) {
            continue;
        }
        if !literal_re.is_match(&literal.to_lowercase()) {
            continue; // "same", "read", "loaded", ... are not literals
        }
        assert!(
            same(&p, &literal),
            "{}: help says {literal}, catalog says {}",
            p.key,
            p.default
        );
        checked += 1;
    }
    assert!(checked >= 52, "{checked}"); // the check really compared most of the catalog
}

#[test]
fn test_defaults_that_the_help_does_not_state_as_literals_llamacpp() {
    let cat = by_key();
    assert_eq!(cat["split_mode"].default, "layer"); // "- layer (default)" in the help text
    assert!(entry_for("--split-mode").text.contains("layer (default)"));
    assert!(
        cat["cors_origins"].default == "localhost"
            && entry_for("--cors-origins").default.as_deref() == Some("*")
    );
    assert!(cat["kv_unified"].default.is_null());
    assert!(
        entry_for("--kv-unified")
            .text
            .contains("enabled if number of slots is auto")
    );
}

#[test]
fn test_choices_match_the_pinned_help_llamacpp() {
    let cat = by_key();
    let text = help();
    assert!(cat["cache_type_k"].choices == cat["cache_type_v"].choices);
    assert_eq!(cat["cache_type_k"].choices, Some(strs(CACHE_TYPES)));
    let k_text = entry_for("--cache-type-k").text;
    let allowed = Regex::new(r"allowed values: ([^()]+?) \(default")
        .expect("re")
        .captures(&k_text)
        .expect("allowed");
    let listed: Vec<String> = allowed[1]
        .split(',')
        .map(|x| py::strip(x).to_string())
        .collect();
    assert_eq!(listed, strs(CACHE_TYPES));
    let bytes: BTreeSet<&str> = CACHE_TYPE_BYTES.iter().map(|(k, _)| *k).collect();
    assert_eq!(CACHE_TYPES.iter().copied().collect::<BTreeSet<_>>(), bytes);
    assert_eq!(
        cat["split_mode"].choices,
        Some(strs(&["none", "layer", "row", "tensor"]))
    );
    assert!(text.contains("{none,layer,row,tensor}"));
    assert!(
        cat["rope_scaling"].choices == Some(strs(&["none", "linear", "yarn"]))
            && text.contains("{none,linear,yarn}")
    );
    assert!(
        cat["pooling"].choices == Some(strs(&["none", "mean", "cls", "last", "rank"]))
            && text.contains("{none,mean,cls,last,rank}")
    );
    for (key, flag) in [
        ("flash_attn", "--flash-attn"),
        ("reasoning", "--reasoning"),
        ("fit", "--fit"),
    ] {
        let choices = cat[key].choices.clone().unwrap_or_default();
        assert!(text.contains(&format!("[{}]", choices.join("|"))), "{key}");
        let _ = entry_for(flag);
    }
    let spec_line = py_lines(&text)
        .into_iter()
        .find(|l| l.starts_with("--spec-type"))
        .expect("--spec-type line");
    let second = spec_line.split_whitespace().nth(1).expect("types");
    assert_eq!(second.split(',').collect::<Vec<_>>(), SPEC_TYPES);
    let load_mode = entry_for("--load-mode").text;
    for mode in LOAD_MODES {
        assert!(load_mode.contains(&format!("- {mode}:")), "{mode}");
    }
    let reasoning_format = entry_for("--reasoning-format").text;
    for fmt in cat["reasoning_format"].choices.clone().unwrap_or_default() {
        assert!(reasoning_format.contains(&format!("- {fmt}:")), "{fmt}");
    }
    let chat_template = entry_for("--chat-template").text;
    for name in CHAT_TEMPLATES {
        assert!(chat_template.contains(name), "{name}");
    }
}

fn py_lines(text: &str) -> Vec<&str> {
    ec_hub::services::common::py_splitlines(text)
}

#[test]
fn test_deprecated_flags_are_replaced_by_load_mode_llamacpp() {
    let flags: BTreeSet<String> = adapter()
        .param_catalog()
        .into_iter()
        .map(|p| p.flag)
        .collect();
    for old in ["--mlock", "--mmap", "--no-mmap", "--direct-io"] {
        assert!(!flags.contains(old));
        let e = entry_for(old);
        assert!(e.deprecated() && e.text.contains("--load-mode"), "{old}");
    }
    assert!(flags.contains("--load-mode") && !entry_for("--load-mode").deprecated());
}

#[test]
fn test_catalog_contains_no_deprecated_or_removed_flag_llamacpp() {
    for p in adapter().param_catalog() {
        if !p.flag.starts_with("env:") {
            let e = entry_for(&p.flag);
            assert!(!e.deprecated(), "{}", p.flag);
            assert!(!e.text.contains("has been removed"), "{}", p.flag);
        }
    }
}

#[test]
fn test_flags_the_console_owns_or_must_never_expose_are_absent_llamacpp() {
    let flags: BTreeSet<String> = adapter()
        .param_catalog()
        .into_iter()
        .map(|p| p.flag)
        .collect();
    let never = [
        "--host",
        "--port",
        "--metrics",
        "--offline",
        "--model",
        "--model-url",
        "--hf-repo",
        "--hf-file",
        "--hf-token",
        "--docker-repo", // launch-owned or downloads
        "--tools",
        "--agent",
        "--mcp-servers-config",
        "--mcp-servers-json",
        "--ui-mcp-proxy", // lets a prompt act
        "--props",
        "--models-dir",
        "--models-preset",
        "--models-max",
        "--models-autoload", // server surface
        "--api-key",
        "--api-key-file",
        "--ssl-key-file",
        "--ssl-cert-file",
        "--path",
        "--api-prefix",
        "--log-file",
        "--log-prompts-dir",
        "--slot-save-path",
        "--media-path",
    ];
    assert!(never.iter().all(|f| !flags.contains(*f)));
    for flag in ["--tools", "--agent", "--mcp-servers-json", "--ui-mcp-proxy"] {
        // the reason, from the help itself
        assert!(
            entry_for(flag)
                .text
                .contains("do not enable in untrusted environments"),
            "{flag}"
        );
    }
}

/// The core replaces any string value under a key matching key|token|secret|password and keeps it out of argv.
#[test]
fn test_secret_looking_keys_are_only_the_api_key_or_non_strings_llamacpp() {
    let secret = Regex::new(r"(?i)(key|token|secret|password)").expect("re");
    for p in adapter().param_catalog() {
        if secret.is_match(&p.key) {
            assert!(
                p.key == "api_key" || ["int", "float", "bool"].contains(&p.type_.as_str()),
                "{}",
                p.key
            );
        }
    }
    assert!(by_key()["api_key"].flag.starts_with("env:"));
}

#[test]
fn test_required_params_present_llamacpp() {
    let keys = by_key();
    for k in [
        "n_gpu_layers",
        "ctx_size",
        "parallel",
        "flash_attn",
        "cache_type_k",
        "cache_type_v",
        "threads",
        "batch_size",
        "ubatch_size",
        "load_mode",
        "override_tensor",
        "split_mode",
        "tensor_split",
        "main_gpu",
        "jinja",
        "reasoning",
        "reasoning_format",
        "rope_scaling",
        "rope_scale",
        "yarn_orig_ctx",
        "cont_batching",
        "kv_offload",
        "cpu_moe",
        "n_cpu_moe",
        "fit",
        "spec_type",
        "mmproj",
        "lora",
        "api_key",
        "cors_origins",
        "served_model_name",
        "chat_template",
        "chat_template_jinja",
        "cache_ram",
        "sleep_idle_seconds",
    ] {
        assert!(keys.contains_key(k), "{k}");
    }
}

#[test]
fn test_all_groups_used_llamacpp() {
    let groups: BTreeSet<String> = adapter()
        .param_catalog()
        .into_iter()
        .map(|p| p.group)
        .collect();
    let want: BTreeSet<String> = strs(&[
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
    ])
    .into_iter()
    .collect();
    assert_eq!(groups, want);
}

#[test]
fn test_memory_params_flagged_llamacpp() {
    let cat = by_key();
    for k in [
        "n_gpu_layers",
        "ctx_size",
        "cache_type_k",
        "cache_type_v",
        "kv_offload",
        "parallel",
        "cache_ram",
        "split_mode",
        "tensor_split",
        "cpu_moe",
        "n_cpu_moe",
        "fit",
        "fit_target",
        "ubatch_size",
        "load_mode",
    ] {
        assert!(cat[k].affects_memory, "{k}");
    }
    assert!(!cat["temperature"].affects_memory && !cat["jinja"].affects_memory);
}

#[test]
fn test_helper_sets_reference_real_keys_llamacpp() {
    let cat = by_key();
    let layer: BTreeMap<&str, &str> = LAYER_KEYS
        .iter()
        .map(|k| (*k, cat[*k].type_.as_str()))
        .collect();
    assert_eq!(
        layer,
        BTreeMap::from([("n_gpu_layers", "string"), ("spec_draft_ngl", "string")])
    );
    for k in PATH_KEYS {
        assert_eq!(cat[*k].type_, "string", "{k}");
    }
}

#[test]
fn test_default_image_is_pinned_by_digest_llamacpp() {
    let a = adapter();
    assert!(
        a.default_image() == DEFAULT_IMAGE
            && DEFAULT_IMAGE == format!("ghcr.io/ggml-org/llama.cpp@{IMAGE_DIGEST}")
    );
    assert!(full(r"sha256:[0-9a-f]{64}").is_match(IMAGE_DIGEST));
    assert!(!a.default_image().contains(":latest") && !a.default_image().contains(":server-cuda"));
}

#[test]
fn test_adapter_identity_llamacpp() {
    let a = adapter();
    assert_eq!(
        (a.id(), a.display_name(), a.default_port()),
        ("llamacpp", "llama.cpp", 8080)
    );
    assert!(a.model_formats() == ["gguf"] && !a.multi_model());
    assert_eq!(a.metric_prefixes(), Some(vec!["llamacpp:".to_string()]));
    assert!(
        a.image_for("x86_64") == a.image_for("aarch64") && a.image_for("x86_64") == DEFAULT_IMAGE
    );
}

#[test]
fn test_catalog_returns_copies_llamacpp() {
    let a = adapter();
    let mut cat = a.param_catalog();
    cat[0].label = "mutated".into();
    assert_ne!(a.param_catalog()[0].label, "mutated");
    let mut pre = a.presets();
    let balanced = &mut pre
        .iter_mut()
        .find(|(n, _)| n == "balanced")
        .expect("balanced")
        .1;
    balanced.insert("ctx_size".into(), Value::from(1));
    let fresh = a.presets();
    assert_eq!(
        fresh
            .iter()
            .find(|(n, _)| n == "balanced")
            .expect("balanced")
            .1["ctx_size"],
        16384
    );
}

/// Short and long aliases share one entry in the help; the catalog uses the long primary one (8 parametrized rows).
#[test]
fn test_aliases_resolve_to_the_primary_spelling_llamacpp() {
    let cat = by_key();
    for (key, flag) in [
        ("n_gpu_layers", "--n-gpu-layers"),
        ("ctx_size", "--ctx-size"),
        ("cache_type_k", "--cache-type-k"),
        ("kv_offload", "--kv-offload"),
        ("flash_attn", "--flash-attn"),
        ("parallel", "--parallel"),
        ("temperature", "--temp"),
        ("served_model_name", "--alias"),
    ] {
        assert_eq!(cat[key].flag, flag);
        let e = entry_for(flag);
        assert!(
            e.flags.iter().any(|f| f == flag) && e.flags.len() >= 2,
            "{flag}"
        );
    }
}
