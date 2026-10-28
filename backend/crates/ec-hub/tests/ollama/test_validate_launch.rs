//! Port of backend/tests/ollama/test_validate_launch.py: validate() and the launch (golden env/argv) of the Ollama
//! adapter.

use std::collections::BTreeSet;

use crate::a2::{oll, p};
use ec_hub::adapters::base::{EngineAdapter, LaunchContext, LaunchSpec, Obj};
use ec_hub::adapters::ollama::OllamaAdapter;
use ec_hub::adapters::ollama_params::DEFAULT_IMAGE;
use ec_hub::adapters::trtllm_parse::py::PyExc;
use regex::Regex;
use serde_json::{Map, Value, json};

fn reserved() -> Map<String, Value> {
    json!({"OLLAMA_HOST": "0.0.0.0:11434", "OLLAMA_MODELS": "/root/.ollama/models", "OLLAMA_NO_CLOUD": "1",
           "OLLAMA_NOPRUNE": "1"})
    .as_object()
    .expect("reserved")
    .clone()
}

fn adapter() -> OllamaAdapter {
    OllamaAdapter::new()
}

fn ctx(params: Obj, model: &str, fmt: &str) -> LaunchContext {
    LaunchContext {
        model: model.into(),
        model_format: fmt.into(),
        model_path: None,
        served_name: "ignored".into(),
        hf_cache_container_path: "/hf".into(),
        params,
        hw: oll::hw(),
    }
}

fn try_launch(params: Obj, model: &str, fmt: &str) -> Result<LaunchSpec, PyExc> {
    adapter().py_build_launch_ctx(&ctx(params, model, fmt))
}

fn launch(params: Value) -> LaunchSpec {
    try_launch(p(params), "llama3.1:8b", "ollama").unwrap_or_else(|e| panic!("{e}"))
}

/// `pytest.raises(ValueError, match=pattern)`; returns the message.
fn raises(result: Result<LaunchSpec, PyExc>, pattern: &str) -> String {
    let e = match result {
        Ok(spec) => panic!("expected a ValueError, got {:?}", spec.env),
        Err(e) => e,
    };
    assert_eq!(e.kind, "ValueError", "{e}");
    assert!(
        Regex::new(pattern).expect("re").is_match(&e.msg),
        "{} !~ {pattern}",
        e.msg
    );
    e.msg
}

fn env_with(extra: Value) -> Value {
    let mut env = reserved();
    let Value::Object(extra) = extra else {
        panic!("extra must be an object")
    };
    for (k, v) in extra {
        env.insert(k, v);
    }
    Value::Object(env)
}

fn preset(name: &str) -> Obj {
    adapter()
        .presets()
        .into_iter()
        .find(|(n, _)| n == name)
        .expect("preset")
        .1
}

// ------------------------------------------------------------------ validate
#[test]
fn test_presets_validate_ollama() {
    let pre = adapter().presets();
    let names: BTreeSet<&str> = pre.iter().map(|(n, _)| n.as_str()).collect();
    let want = BTreeSet::from([
        "balanced",
        "max-throughput",
        "low-latency",
        "long-context",
        "low-memory",
        "many-models",
        "big-model-spread",
    ]);
    assert_eq!(names, want);
    for (name, params) in &pre {
        assert!(adapter().validate(params).is_empty(), "{name}");
    }
}

#[test]
fn test_validate_unknown_and_none_ollama() {
    let a = adapter();
    assert_eq!(
        a.validate(&p(json!({"nope": 1}))),
        vec!["unknown parameter 'nope'"]
    );
    assert!(
        a.validate(&p(json!({"context_length": null, "flash_attention": null})))
            .is_empty()
    );
    assert!(a.validate(&Obj::new()).is_empty());
}

/// 38 parametrized rows. `{"context_length": float("inf")}` cannot be written in JSON, the only way parameters reach the
/// Rust adapter (serde_json has no infinity), so that row is covered by the type check on the other non-integers.
#[test]
fn test_validate_rejects_ollama() {
    for params in [
        json!({"context_length": -1}),
        json!({"context_length": 1.5}),
        json!({"context_length": true}),
        json!({"context_length": "8192"}),
        json!({"num_parallel": 0}),
        json!({"num_parallel": -2}),
        json!({"max_loaded_models": -1}),
        json!({"max_queue": 0}),
        json!({"gpu_overhead": -1}),
        json!({"fit_target": -5}),
        json!({"max_transfer_streams": 0}),
        json!({"flash_attention": 1}),
        json!({"flash_attention": "true"}),
        json!({"sched_spread": "yes"}),
        json!({"debug": 1}),
        json!({"igpu_enable": "on"}),
        json!({"kv_cache_type": "q5_0"}),
        json!({"kv_cache_type": "F16"}),
        json!({"fit": "auto"}),
        json!({"keep_alive": "soon"}),
        json!({"keep_alive": "5 minutes"}),
        json!({"keep_alive": ""}),
        json!({"keep_alive": true}),
        json!({"keep_alive": "5m;rm"}),
        json!({"keep_alive": "m5"}),
        json!({"load_timeout": "-1"}),
        json!({"load_timeout": "-5m"}),
        json!({"load_timeout": "later"}),
        json!({"origins": []}),
        json!({"origins": "https://a.example"}),
        json!({"origins": ["a b"]}),
        json!({"origins": ["a,b"]}),
        json!({"origins": [""]}),
        json!({"llm_library": "cuda v13"}),
        json!({"llm_library": "../x"}),
        json!({"llm_library": "CUDA"}),
        json!({"llm_library": ""}),
    ] {
        assert!(
            !adapter().validate(&p(params.clone())).is_empty(),
            "{params}"
        );
    }
}

/// 23 parametrized rows.
#[test]
fn test_validate_accepts_ollama() {
    for params in [
        json!({"keep_alive": "5m"}),
        json!({"keep_alive": "24h"}),
        json!({"keep_alive": "1h30m"}),
        json!({"keep_alive": "90s"}),
        json!({"keep_alive": "0"}),
        json!({"keep_alive": "-1"}),
        json!({"keep_alive": "-1m"}),
        json!({"keep_alive": 3600}),
        json!({"keep_alive": "0.5h"}),
        json!({"keep_alive": "300ms"}),
        json!({"load_timeout": "30m"}),
        json!({"load_timeout": 600}),
        json!({"origins": ["https://chat.example.org", "app://*"]}),
        json!({"llm_library": "cuda_v13"}),
        json!({"llm_library": "vulkan"}),
        json!({"fit": "off"}),
        json!({"fit_target": 0}),
        json!({"kv_cache_type": "q4_0"}),
        json!({"gpu_overhead": 2_147_483_648_i64}),
        json!({"num_parallel": 16}),
        json!({"max_loaded_models": 0}),
        json!({"context_length": 0}),
        json!({"context_length": 262_144}),
    ] {
        assert_eq!(
            adapter().validate(&p(params.clone())),
            Vec::<String>::new(),
            "{params}"
        );
    }
}

// ------------------------------------------------------------------ launch: golden env
#[test]
fn test_golden_bare_server_ollama() {
    let spec = try_launch(Obj::new(), "", "ollama").expect("spec");
    assert!(spec.image == DEFAULT_IMAGE && DEFAULT_IMAGE == "ollama/ollama:0.30.11");
    assert_eq!(spec.argv, ["serve"]);
    assert_eq!(spec.env, reserved());
    assert_eq!(
        (
            spec.container_port,
            spec.health_path.as_str(),
            spec.metrics_path.as_str(),
            spec.models_path.as_str()
        ),
        (11434, "/api/version", "/api/version", "/v1/models")
    );
    assert!(
        !spec.needs_hf_cache && !spec.ipc_host && spec.shm_size == "1g" && spec.files.is_empty()
    );
}

#[test]
fn test_golden_balanced_with_a_named_model_ollama() {
    let spec = try_launch(preset("balanced"), "llama3.1:8b", "ollama").expect("spec");
    assert_eq!(spec.argv, ["serve"]); // nothing is preloaded: the first request loads it
    assert_eq!(
        Value::Object(spec.env),
        env_with(json!({"OLLAMA_CONTEXT_LENGTH": "8192", "OLLAMA_KEEP_ALIVE": "30m"}))
    );
}

#[test]
fn test_golden_complex_ollama() {
    let params = json!({
        "context_length": 32768, "num_parallel": 4, "max_loaded_models": 2, "keep_alive": "-1", "kv_cache_type": "q8_0",
        "flash_attention": true, "gpu_overhead": 2_147_483_648_i64, "sched_spread": true, "max_queue": 128,
        "load_timeout": "30m", "origins": ["https://chat.example.org", "http://localhost:3000"], "llm_library": "cuda_v13",
        "igpu_enable": false, "fit": "off", "fit_target": 2048, "max_transfer_streams": 8, "debug": true,
    });
    assert_eq!(
        Value::Object(launch(params).env),
        env_with(json!({
            "OLLAMA_CONTEXT_LENGTH": "32768", "OLLAMA_NUM_PARALLEL": "4", "OLLAMA_MAX_LOADED_MODELS": "2",
            "OLLAMA_KEEP_ALIVE": "-1", "OLLAMA_KV_CACHE_TYPE": "q8_0", "OLLAMA_FLASH_ATTENTION": "1",
            "OLLAMA_GPU_OVERHEAD": "2147483648", "OLLAMA_SCHED_SPREAD": "1", "OLLAMA_MAX_QUEUE": "128",
            "OLLAMA_LOAD_TIMEOUT": "30m", "OLLAMA_ORIGINS": "https://chat.example.org,http://localhost:3000",
            "OLLAMA_LLM_LIBRARY": "cuda_v13", "LLAMA_ARG_FIT": "off", // igpu_enable=False sends nothing (test_live_env)
            "LLAMA_ARG_FIT_TARGET": "2048", "OLLAMA_MAX_TRANSFER_STREAMS": "8", "OLLAMA_DEBUG": "1"}))
    );
}

#[test]
fn test_every_preset_builds_ollama() {
    let res = reserved();
    for (name, params) in adapter().presets() {
        let spec = try_launch(params.clone(), "llama3.1:8b", "ollama").expect("spec");
        assert_eq!(spec.argv, ["serve"], "{name}");
        assert!(
            res.iter().all(|(k, v)| spec.env.get(k) == Some(v)),
            "{name}"
        );
        let set: BTreeSet<String> = spec
            .env
            .keys()
            .filter(|k| !res.contains_key(*k))
            .cloned()
            .collect();
        let want: BTreeSet<String> = adapter()
            .param_catalog()
            .into_iter()
            .filter(|p| params.contains_key(&p.key))
            .map(|p| p.flag[4..].to_string())
            .collect();
        assert_eq!(set, want, "{name}");
    }
}

/// Cloud features stay off, pruning stays off, and the listen address and store path are the adapter's.
#[test]
fn test_the_reserved_variables_cannot_be_overridden_ollama() {
    let spec = launch(json!({"debug": true, "origins": ["https://a.example"]}));
    for (name, value) in reserved() {
        assert_eq!(spec.env[&name], value);
    }
    for key in ["ollama_no_cloud", "no_cloud", "noprune", "host", "models"] {
        raises(
            try_launch(p(json!({key: "0"})), "llama3.1:8b", "ollama"),
            "unknown parameter",
        );
    }
}

#[test]
fn test_false_booleans_are_explicit_zeros_ollama() {
    let env = launch(json!({"flash_attention": false, "sched_spread": false, "debug": false})).env;
    assert!(
        env["OLLAMA_FLASH_ATTENTION"] == "0"
            && env["OLLAMA_SCHED_SPREAD"] == "0"
            && env["OLLAMA_DEBUG"] == "0"
    );
    let unset = launch(json!({})).env;
    assert!(!unset.contains_key("OLLAMA_FLASH_ATTENTION")); // tri-state: unset lets Ollama decide
}

#[test]
fn test_duration_values_are_trimmed_and_numbers_become_seconds_ollama() {
    assert_eq!(
        launch(json!({"keep_alive": " 10m "})).env["OLLAMA_KEEP_ALIVE"],
        "10m"
    );
    assert_eq!(
        launch(json!({"keep_alive": 3600})).env["OLLAMA_KEEP_ALIVE"],
        "3600"
    );
}

/// 6 parametrized rows.
#[test]
fn test_model_names_pass_ollama() {
    for name in [
        "llama3.1:8b",
        "library/qwen3:30b-a3b-q4_K_M",
        "hf.co/org/repo-GGUF:Q4_K_M",
        "registry.example.org/team/model:1.0",
        "gemma3",
        "",
    ] {
        assert_eq!(
            try_launch(Obj::new(), name, "ollama").expect("spec").argv,
            ["serve"],
            "{name}"
        );
    }
}

/// 8 parametrized rows.
#[test]
fn test_bad_model_names_are_refused_ollama() {
    let long = "x".repeat(300);
    for name in [
        "-x",
        "a b",
        "a;b",
        "a\nb",
        "../etc",
        " llama",
        long.as_str(),
        "model|cat",
    ] {
        raises(try_launch(Obj::new(), name, "ollama"), "Ollama name");
    }
}

/// 3 parametrized rows.
#[test]
fn test_other_formats_are_not_loadable_yet_ollama() {
    for fmt in ["hf", "gguf", "plain"] {
        let msg = raises(try_launch(Obj::new(), "llama3.1:8b", fmt), "");
        assert!(
            msg.contains("own store") && msg.contains("not supported yet") && msg.contains(fmt),
            "{msg}"
        );
    }
}

#[test]
fn test_invalid_params_raise_ollama() {
    raises(
        try_launch(p(json!({"bogus": 1})), "llama3.1:8b", "ollama"),
        "unknown parameter",
    );
    raises(
        try_launch(p(json!({"num_parallel": 0})), "llama3.1:8b", "ollama"),
        "num_parallel",
    );
}

#[test]
fn test_v1_build_launch_bridges_to_the_context_path_ollama() {
    let a = adapter();
    let spec = a
        .build_launch(
            "llama3.1:8b",
            &p(json!({"num_parallel": 2})),
            &oll::hw(),
            "x",
            "/hf",
        )
        .expect("spec");
    assert!(
        spec.argv == ["serve"] && spec.env["OLLAMA_NUM_PARALLEL"] == "2" && !spec.needs_hf_cache
    );
    let err = a
        .build_launch("org/Model with spaces", &Obj::new(), &oll::hw(), "x", "/hf")
        .expect_err("refused");
    assert!(
        err.status == 422 && err.detail.contains("Ollama name"),
        "{err:?}"
    );
}

/// Ollama has no API key for its local server, so nothing secret-named is ever set.
#[test]
fn test_no_secret_can_reach_the_environment_ollama() {
    let spec = try_launch(preset("many-models"), "llama3.1:8b", "ollama").expect("spec");
    assert!(!spec.env.keys().any(|k| {
        ["key", "token", "secret", "password"]
            .iter()
            .any(|w| k.to_lowercase().contains(w))
    }));
}

#[test]
fn test_preload_request_is_a_plain_generate_without_a_prompt_ollama() {
    let a = adapter();
    let (path, body) = a
        .preload_request("llama3.1:8b", &Obj::new())
        .expect("preload");
    assert!(
        path == "/api/generate"
            && body == json!({"model": "llama3.1:8b", "stream": false})
            && body.get("prompt").is_none()
    );
    // Python raises ValueError; the Rust helper returns the message
    assert!(
        a.preload_request("", &Obj::new())
            .expect_err("refused")
            .contains("Ollama name")
    );
    assert!(
        a.preload_request("bad name", &Obj::new())
            .expect_err("refused")
            .contains("Ollama name")
    );
}
