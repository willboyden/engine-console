//! Port of backend/tests/ollama/test_live_env.py (with live_cases.py): the adapter's launch environment against the LIVE
//! start-up of the pinned image (tests/data/ollama/live_env_0.30.11.json, made by live_check.py: a bare server, no
//! model, no GPU, no network, an empty read-only store): the effective value the server logged for every variable, and
//! how it treats the value forms the validator accepts and rejects.

use std::collections::BTreeSet;

use crate::a2::{oll, p, read_json};
use ec_hub::adapters::base::{EngineAdapter, LaunchContext, LaunchSpec, Obj};
use ec_hub::adapters::ollama::OllamaAdapter;
use ec_hub::adapters::ollama_params::DEFAULT_IMAGE;
use serde_json::{Map, Value, json};

const RERUN: &str = "the environment the adapter builds changed since the live check: run tests/ollama/live_check.py of the \
                     Python reference (backend/parity/lib/python-release.mjs) with its backend venv";
/// what the server shows for a negative keep-alive: the largest duration
const FOREVER: &str = "2562047h47m16.854775807s";

/// One of every parameter, so every variable the adapter can set was exercised (live_cases.ADAPTER_PARAMS).
fn adapter_params() -> Value {
    json!({
        "context_length": 32768, "num_parallel": 4, "max_loaded_models": 2, "keep_alive": "-1", "kv_cache_type": "q8_0",
        "flash_attention": true, "gpu_overhead": 2_147_483_648_i64, "sched_spread": true, "max_queue": 128,
        "load_timeout": "30m", "origins": ["https://chat.example.org", "http://localhost:3000"], "llm_library": "cuda_v13",
        "igpu_enable": true, "fit": "off", "fit_target": 2048, "max_transfer_streams": 8, "debug": true,
    })
}

/// Value forms the adapter's validation accepts or rejects (live_cases.FORMS).
fn forms() -> Vec<(&'static str, Value)> {
    vec![
        (
            "bools_zero",
            json!({"OLLAMA_FLASH_ATTENTION": "0", "OLLAMA_SCHED_SPREAD": "0", "OLLAMA_DEBUG": "0"}),
        ),
        (
            "bools_one",
            json!({"OLLAMA_FLASH_ATTENTION": "1", "OLLAMA_SCHED_SPREAD": "1", "OLLAMA_DEBUG": "1"}),
        ),
        (
            "bools_false",
            json!({"OLLAMA_FLASH_ATTENTION": "false", "OLLAMA_SCHED_SPREAD": "false", "OLLAMA_DEBUG": "false"}),
        ),
        ("igpu_zero", json!({"OLLAMA_IGPU_ENABLE": "0"})),
        ("igpu_false", json!({"OLLAMA_IGPU_ENABLE": "false"})),
        ("igpu_one", json!({"OLLAMA_IGPU_ENABLE": "1"})),
        ("keep_alive_seconds", json!({"OLLAMA_KEEP_ALIVE": "3600"})),
        ("keep_alive_zero", json!({"OLLAMA_KEEP_ALIVE": "0"})),
        (
            "keep_alive_minus_minutes",
            json!({"OLLAMA_KEEP_ALIVE": "-1m"}),
        ),
        ("keep_alive_compound", json!({"OLLAMA_KEEP_ALIVE": "1h30m"})),
        ("keep_alive_junk", json!({"OLLAMA_KEEP_ALIVE": "soon"})),
        (
            "load_timeout_seconds",
            json!({"OLLAMA_LOAD_TIMEOUT": "600"}),
        ),
        ("kv_type_junk", json!({"OLLAMA_KV_CACHE_TYPE": "bogus"})),
        ("parallel_zero", json!({"OLLAMA_NUM_PARALLEL": "0"})),
    ]
}

fn record() -> Value {
    read_json("ollama", "live_env_0.30.11.json")
}

fn run(name: &str) -> Value {
    record()["runs"]
        .as_array()
        .expect("runs")
        .iter()
        .find(|r| r["name"] == name)
        .expect("run")
        .clone()
}

fn effective(name: &str) -> Map<String, Value> {
    run(name)["effective"]
        .as_object()
        .expect("effective")
        .clone()
}

fn launch(params: Obj) -> LaunchSpec {
    let ctx = LaunchContext {
        model: String::new(),
        model_format: "ollama".into(),
        model_path: None,
        served_name: "x".into(),
        hf_cache_container_path: "/hf".into(),
        params,
        hw: oll::hw(),
    };
    OllamaAdapter::new().build_launch_ctx(&ctx).expect("spec")
}

#[test]
fn test_the_recorded_run_used_the_pinned_image_ollama() {
    assert!(record()["image"] == DEFAULT_IMAGE && DEFAULT_IMAGE == "ollama/ollama:0.30.11");
}

#[test]
fn test_the_adapter_still_builds_the_recorded_environment_ollama() {
    let spec = launch(p(adapter_params()));
    assert_eq!(
        Value::Object(spec.env.clone()),
        run("adapter")["env"],
        "{RERUN}"
    );
    let rec = record();
    let recorded: Vec<(String, Value)> = rec["runs"].as_array().expect("runs")[1..]
        .iter()
        .map(|r| {
            (
                r["name"].as_str().expect("name").to_string(),
                r["env"].clone(),
            )
        })
        .collect();
    let mine: Vec<(String, Value)> = forms()
        .into_iter()
        .map(|(n, e)| (n.to_string(), e))
        .collect();
    assert_eq!(mine, recorded, "{RERUN}");
    let reserved = [
        "OLLAMA_HOST",
        "OLLAMA_MODELS",
        "OLLAMA_NO_CLOUD",
        "OLLAMA_NOPRUNE",
    ];
    let set: BTreeSet<String> = spec
        .env
        .keys()
        .filter(|k| !reserved.contains(&k.as_str()))
        .cloned()
        .collect();
    let flags: BTreeSet<String> = OllamaAdapter::new()
        .param_catalog()
        .into_iter()
        .map(|p| p.flag[4..].to_string())
        .collect();
    assert_eq!(set, flags); // one of every parameter was exercised
}

#[test]
fn test_every_variable_took_the_effective_value_the_adapter_meant_ollama() {
    let eff = effective("adapter");
    let picked: Map<String, Value> = [
        "OLLAMA_CONTEXT_LENGTH",
        "OLLAMA_NUM_PARALLEL",
        "OLLAMA_MAX_LOADED_MODELS",
        "OLLAMA_MAX_QUEUE",
        "OLLAMA_MAX_TRANSFER_STREAMS",
        "OLLAMA_GPU_OVERHEAD",
        "LLAMA_ARG_FIT_TARGET",
    ]
    .iter()
    .map(|k| ((*k).to_string(), eff[*k].clone()))
    .collect();
    assert_eq!(
        Value::Object(picked),
        json!({"OLLAMA_CONTEXT_LENGTH": "32768", "OLLAMA_NUM_PARALLEL": "4", "OLLAMA_MAX_LOADED_MODELS": "2",
               "OLLAMA_MAX_QUEUE": "128", "OLLAMA_MAX_TRANSFER_STREAMS": "8", "OLLAMA_GPU_OVERHEAD": "2147483648",
               "LLAMA_ARG_FIT_TARGET": "2048"})
    );
    assert!(
        eff["OLLAMA_FLASH_ATTENTION"] == "true"
            && eff["OLLAMA_SCHED_SPREAD"] == "true"
            && eff["OLLAMA_DEBUG"] == "DEBUG"
    );
    assert!(eff["OLLAMA_KEEP_ALIVE"] == FOREVER && eff["OLLAMA_LOAD_TIMEOUT"] == "30m0s");
    assert!(
        eff["OLLAMA_KV_CACHE_TYPE"] == "q8_0"
            && eff["OLLAMA_LLM_LIBRARY"] == "cuda_v13"
            && eff["LLAMA_ARG_FIT"] == "off"
    );
    assert_eq!(eff["OLLAMA_IGPU_ENABLE"], "1");
    assert!(
        eff["OLLAMA_HOST"] == "http://0.0.0.0:11434"
            && eff["OLLAMA_MODELS"] == "/root/.ollama/models"
    );
}

#[test]
fn test_the_reserved_variables_took_effect_ollama() {
    let eff = effective("adapter");
    assert!(eff["OLLAMA_NO_CLOUD"] == "true" && eff["OLLAMA_NOPRUNE"] == "true");
}

#[test]
fn test_origins_are_added_to_the_built_in_list_not_replacing_it_ollama() {
    let eff = effective("adapter");
    let value: Vec<&str> = eff["OLLAMA_ORIGINS"]
        .as_str()
        .expect("origins")
        .trim_matches(['[', ']'])
        .split_whitespace()
        .collect();
    assert_eq!(
        value[..2],
        ["https://chat.example.org", "http://localhost:3000"]
    );
    assert!(
        ["http://localhost", "app://*", "vscode-file://*"]
            .iter()
            .all(|o| value.contains(o))
            && !value.contains(&"*")
    );
}

/// 3 parametrized rows: (name, expected).
#[test]
fn test_boolean_variables_read_one_and_zero_as_the_adapter_assumes_ollama() {
    for (name, expected) in [
        (
            "bools_zero",
            json!({"OLLAMA_FLASH_ATTENTION": "false", "OLLAMA_SCHED_SPREAD": "false", "OLLAMA_DEBUG": "INFO"}),
        ),
        (
            "bools_one",
            json!({"OLLAMA_FLASH_ATTENTION": "true", "OLLAMA_SCHED_SPREAD": "true", "OLLAMA_DEBUG": "DEBUG"}),
        ),
        (
            "bools_false",
            json!({"OLLAMA_FLASH_ATTENTION": "false", "OLLAMA_SCHED_SPREAD": "false", "OLLAMA_DEBUG": "INFO"}),
        ),
    ] {
        assert_eq!(Value::Object(effective(name)), expected, "{name}");
    }
}

/// The server echoes OLLAMA_IGPU_ENABLE verbatim ("0" and "false" stay as typed, unlike the real booleans), so no one can
/// tell from its configuration that they would read as off: the adapter leaves the variable unset instead.
#[test]
fn test_the_igpu_variable_is_a_plain_string_so_false_is_never_sent_ollama() {
    let seen: Vec<Value> = ["igpu_zero", "igpu_false", "igpu_one"]
        .iter()
        .map(|n| effective(n)["OLLAMA_IGPU_ENABLE"].clone())
        .collect();
    assert_eq!(seen, [json!("0"), json!("false"), json!("1")]);
    let off = launch(p(json!({"igpu_enable": false})));
    let on = launch(p(json!({"igpu_enable": true})));
    assert!(!off.env.contains_key("OLLAMA_IGPU_ENABLE") && on.env["OLLAMA_IGPU_ENABLE"] == "1");
}

/// 4 parametrized rows: (name, effective, param).
#[test]
fn test_keep_alive_forms_the_validator_accepts_mean_what_the_help_says_ollama() {
    for (name, eff, param) in [
        ("keep_alive_seconds", "1h0m0s", "3600"),
        ("keep_alive_zero", "0s", "0"),
        ("keep_alive_minus_minutes", FOREVER, "-1m"),
        ("keep_alive_compound", "1h30m0s", "1h30m"),
    ] {
        assert_eq!(effective(name)["OLLAMA_KEEP_ALIVE"], eff, "{name}");
        assert!(
            OllamaAdapter::new()
                .validate(&p(json!({"keep_alive": param})))
                .is_empty(),
            "{param}"
        );
    }
}

#[test]
fn test_a_junk_keep_alive_is_silently_ignored_by_the_server_so_the_validator_rejects_it_ollama() {
    assert_eq!(effective("keep_alive_junk")["OLLAMA_KEEP_ALIVE"], "5m0s"); // fell back to the default
    assert!(
        !OllamaAdapter::new()
            .validate(&p(json!({"keep_alive": "soon"})))
            .is_empty()
    );
}

#[test]
fn test_a_load_timeout_in_seconds_is_read_as_seconds_ollama() {
    assert_eq!(
        effective("load_timeout_seconds")["OLLAMA_LOAD_TIMEOUT"],
        "10m0s"
    );
    assert!(
        OllamaAdapter::new()
            .validate(&p(json!({"load_timeout": 600})))
            .is_empty()
    );
}

#[test]
fn test_the_server_does_not_validate_the_kv_type_or_parallelism_so_the_adapter_must_ollama() {
    assert_eq!(effective("kv_type_junk")["OLLAMA_KV_CACHE_TYPE"], "bogus"); // echoed, never rejected
    assert_eq!(effective("parallel_zero")["OLLAMA_NUM_PARALLEL"], "0");
    let a = OllamaAdapter::new();
    assert!(!a.validate(&p(json!({"kv_cache_type": "bogus"}))).is_empty());
    assert!(!a.validate(&p(json!({"num_parallel": 0}))).is_empty());
}
