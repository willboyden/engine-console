//! Port of backend/tests/llamacpp/test_validate_launch.py: validate() and the launch (golden argv/env) of the llama.cpp
//! adapter.

use std::collections::BTreeSet;

use crate::a2::{llama, p};
use ec_hub::adapters::base::{
    EngineAdapter, Hardware, LaunchContext, LaunchSpec, MODEL_MOUNT, Obj,
};
use ec_hub::adapters::llamacpp::LlamaCppAdapter;
use ec_hub::adapters::llamacpp_params::{CHAT_TEMPLATE_PATH, DEFAULT_IMAGE};
use ec_hub::adapters::trtllm_parse::py::PyExc;
use regex::Regex;
use serde_json::{Value, json};

fn model() -> String {
    format!("{MODEL_MOUNT}/demo-Q4_K_M.gguf")
}

fn adapter() -> LlamaCppAdapter {
    LlamaCppAdapter::new()
}

fn ctx(params: Obj, hw: &Hardware, path: Option<&str>, served: &str, fmt: &str) -> LaunchContext {
    LaunchContext {
        model: "example-org/chat-model-GGUF:Q4_K_M".into(),
        model_format: fmt.into(),
        model_path: path.map(str::to_string),
        served_name: served.into(),
        hf_cache_container_path: "/hf".into(),
        params,
        hw: hw.clone(),
    }
}

/// `launch(adapter, hw, params)` with the default path, served name and format.
fn launch(params: Value) -> LaunchSpec {
    launch_obj(p(params))
}

fn launch_obj(params: Obj) -> LaunchSpec {
    let m = model();
    adapter()
        .py_build_launch_ctx(&ctx(params, &llama::hw(), Some(&m), "demo", "gguf"))
        .unwrap_or_else(|e| panic!("{e}"))
}

/// `pytest.raises(ValueError, match=pattern)`.
fn raises(c: &LaunchContext, pattern: &str) {
    let e: PyExc = match adapter().py_build_launch_ctx(c) {
        Ok(spec) => panic!("expected a ValueError, got {:?}", spec.argv),
        Err(e) => e,
    };
    assert_eq!(e.kind, "ValueError", "{e}");
    assert!(
        Regex::new(pattern).expect("re").is_match(&e.msg),
        "{} !~ {pattern}",
        e.msg
    );
}

fn after(argv: &[String], flag: &str) -> String {
    argv[argv
        .iter()
        .position(|a| a == flag)
        .unwrap_or_else(|| panic!("{flag} not in argv"))
        + 1]
    .clone()
}

fn pos(argv: &[String], flag: &str) -> usize {
    argv.iter()
        .position(|a| a == flag)
        .unwrap_or_else(|| panic!("{flag} not in argv"))
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
fn test_presets_validate_llamacpp() {
    let pre = adapter().presets();
    let names: BTreeSet<&str> = pre.iter().map(|(n, _)| n.as_str()).collect();
    let want = BTreeSet::from([
        "balanced",
        "max-throughput",
        "low-latency",
        "long-context",
        "tool-agent",
        "reasoning",
        "big-model-offload",
        "moe-experts-on-cpu",
    ]);
    assert_eq!(names, want);
    for (name, params) in &pre {
        assert!(adapter().validate(params).is_empty(), "{name}");
    }
}

#[test]
fn test_validate_unknown_and_none_llamacpp() {
    let a = adapter();
    assert_eq!(
        a.validate(&p(json!({"nope": 1}))),
        vec!["unknown parameter 'nope'"]
    );
    assert!(
        a.validate(&p(json!({"ctx_size": null, "flash_attn": null})))
            .is_empty()
    );
    assert!(a.validate(&Obj::new()).is_empty());
}

/// 55 parametrized rows. `{"ctx_size": float("nan")}` cannot be written in JSON, the only way parameters reach the Rust
/// adapter (serde_json has no NaN), so that row is covered by the type check on the other non-integers.
#[test]
fn test_validate_rejects_llamacpp() {
    let x = |n: usize| "x".repeat(n);
    for params in [
        json!({"ctx_size": -1}),
        json!({"ctx_size": 1.5}),
        json!({"ctx_size": true}),
        json!({"ctx_size": "8192"}),
        json!({"parallel": -2}),
        json!({"fit_target": -1}),
        json!({"top_p": 1.5}),
        json!({"min_p": -0.1}),
        json!({"log_verbosity": 6}),
        json!({"temperature": -1}),
        json!({"flash_attn": "maybe"}),
        json!({"cache_type_v": "fp7"}),
        json!({"split_mode": "diagonal"}),
        json!({"load_mode": "mlock+dio"}),
        json!({"jinja": 1}),
        json!({"kv_offload": "yes"}),
        json!({"cpu_moe": "true"}),
        json!({"n_gpu_layers": -1}),
        json!({"n_gpu_layers": "most"}),
        json!({"n_gpu_layers": 1.5}),
        json!({"n_gpu_layers": true}),
        json!({"spec_draft_ngl": "some"}),
        json!({"spec_type": "magic"}),
        json!({"tensor_split": "3;1"}),
        json!({"tensor_split": "a,b"}),
        json!({"device": "CUDA0;rm"}),
        json!({"cors_origins": "a b"}),
        json!({"api_key": "has space"}),
        json!({"api_key": ""}),
        json!({"api_key": x(600)}),
        json!({"chat_template_kwargs": "{not json"}),
        json!({"chat_template_kwargs": "[1]"}),
        json!({"served_model_name": []}),
        json!({"served_model_name": ["a,b"]}),
        json!({"served_model_name": ["-x"]}),
        json!({"served_model_name": "a"}),
        json!({"override_tensor": "exps"}),
        json!({"override_kv": "tokenizer.x=bool"}),
        json!({"n_cpu_moe": -1}),
        json!({"mmproj": "/etc/passwd"}),
        json!({"mmproj": "/models/../etc/passwd"}),
        json!({"mmproj": "mmproj.gguf"}),
        json!({"mmproj": "/models/readme.txt"}),
        json!({"lora": "/models/a.gguf,/tmp/b.gguf"}),
        json!({"lora_scaled": "/models/a.gguf"}),
        json!({"lora_scaled": "/models/a.gguf:x"}),
        json!({"spec_draft_model": "-m"}),
        json!({"spec_draft_model": "/models/a\u{0}.gguf"}),
        json!({"chat_template": x(4001)}),
        json!({"chat_template_jinja": x(130_000)}),
        json!({"chat_template_jinja": "a\u{0}b"}),
        json!({"override_tensor": "-x=CPU"}),
        json!({"cache_reuse": 1.5}),
    ] {
        assert!(
            !adapter().validate(&p(params.clone())).is_empty(),
            "{params}"
        );
    }
}

#[test]
fn test_validate_rejects_both_template_forms_llamacpp() {
    let errs = adapter().validate(&p(
        json!({"chat_template": "chatml", "chat_template_jinja": "{{ x }}"}),
    ));
    assert_eq!(
        errs,
        vec!["chat_template and chat_template_jinja are alternatives: set only one"]
    );
}

#[test]
fn test_validate_accepts_valid_values_llamacpp() {
    let m = MODEL_MOUNT;
    let ok = json!({
        "n_gpu_layers": "all", "spec_draft_ngl": 12, "ctx_size": 0, "parallel": -1, "tensor_split": "0.7,0.3",
        "device": "CUDA0,CUDA1", "cors_origins": "https://chat.example.org,localhost", "api_key": "k-1.2_3",
        "chat_template_kwargs": "{\"enable_thinking\": false}", "served_model_name": ["gpt-4o", "org/model:q4"],
        "override_tensor": r"\.ffn_.*_exps\.=CPU,blk\.1\.=CUDA0", "override_kv": "tokenizer.ggml.add_bos_token=bool:false",
        "mmproj": format!("{m}/mmproj.gguf"), "lora": format!("{m}/a.gguf,{m}/b.gguf"),
        "lora_scaled": format!("{m}/a.gguf:0.5,{m}/b.gguf:-1"), "spec_draft_model": format!("{m}/snapshots/abc/d.gguf"),
        "chat_template_jinja": "{% for m in messages %}{{ m.content }}{% endfor %}\n", "temperature": 0, "top_p": 1.0,
        "fit_target": 0, "log_verbosity": 5, "reasoning_budget": -1, "cache_ram": -1, "sleep_idle_seconds": 600,
        "cache_type_k": "iq4_nl", "numa": "isolate", "pooling": "rank",
    });
    assert_eq!(adapter().validate(&p(ok)), Vec::<String>::new());
    assert!(
        adapter()
            .validate(&p(json!({"chat_template": "chatml"})))
            .is_empty()
    );
}

// ------------------------------------------------------------------ launch: golden argv
#[test]
fn test_golden_balanced_llamacpp() {
    let spec = launch_obj(preset("balanced"));
    assert_eq!(spec.image, DEFAULT_IMAGE);
    let m = model();
    assert_eq!(
        spec.argv,
        [
            "-m",
            &m,
            "--host",
            "0.0.0.0",
            "--port",
            "8080",
            "--alias",
            "demo",
            "--metrics",
            "--offline",
            "--ctx-size",
            "16384",
            "--parallel",
            "1",
            "--flash-attn",
            "on",
            "--cors-origins",
            "localhost"
        ]
    );
    assert!(spec.env.is_empty() && spec.files.is_empty());
    assert_eq!(
        (
            spec.container_port,
            spec.health_path.as_str(),
            spec.metrics_path.as_str(),
            spec.models_path.as_str()
        ),
        (8080, "/health", "/metrics", "/v1/models")
    );
    assert!(!spec.needs_hf_cache && !spec.ipc_host && spec.shm_size == "1g");
}

#[test]
fn test_golden_complex_llamacpp() {
    let params = json!({
        "n_gpu_layers": "all", "kv_offload": false, "cache_type_k": "q8_0", "cache_type_v": "q8_0", "cpu_moe": true,
        "split_mode": "layer", "tensor_split": "3,1", "main_gpu": 0, "ctx_size": 32768, "rope_scaling": "yarn",
        "rope_scale": 4, "context_shift": true, "parallel": 4, "kv_unified": false, "cont_batching": true,
        "flash_attn": "on", "threads": 16, "batch_size": 4096, "jinja": true, "chat_template": "chatml",
        "chat_template_kwargs": {"enable_thinking": false}, "reasoning": "off", "reasoning_format": "deepseek",
        "spec_type": "ngram-mod", "served_model_name": ["alias-a", "demo"], "api_key": "secret-value",
        "cors_origins": "https://chat.example.org", "ui": false, "temperature": 0.6, "top_k": 20, "top_p": 0.95,
        "min_p": 0.0, "seed": 42,
    });
    let spec = launch(params);
    let m = model();
    assert_eq!(
        spec.argv,
        [
            "-m",
            &m,
            "--host",
            "0.0.0.0",
            "--port",
            "8080",
            "--alias",
            "demo,alias-a",
            "--metrics",
            "--offline",
            "--n-gpu-layers",
            "all",
            "--no-kv-offload",
            "--cache-type-k",
            "q8_0",
            "--cache-type-v",
            "q8_0",
            "--cpu-moe",
            "--split-mode",
            "layer",
            "--tensor-split",
            "3,1",
            "--main-gpu",
            "0",
            "--ctx-size",
            "32768",
            "--rope-scaling",
            "yarn",
            "--rope-scale",
            "4",
            "--context-shift",
            "--parallel",
            "4",
            "--no-kv-unified",
            "--cont-batching",
            "--flash-attn",
            "on",
            "--threads",
            "16",
            "--batch-size",
            "4096",
            "--jinja",
            "--chat-template",
            "chatml",
            "--chat-template-kwargs",
            "{\"enable_thinking\":false}",
            "--reasoning",
            "off",
            "--reasoning-format",
            "deepseek",
            "--spec-type",
            "ngram-mod",
            "--cors-origins",
            "https://chat.example.org",
            "--no-ui",
            "--temp",
            "0.6",
            "--top-k",
            "20",
            "--top-p",
            "0.95",
            "--min-p",
            "0",
            "--seed",
            "42"
        ]
    );
    assert_eq!(
        Value::Object(spec.env),
        json!({"LLAMA_API_KEY": "secret-value"})
    );
}

#[test]
fn test_every_preset_builds_llamacpp() {
    let m = model();
    for (name, params) in adapter().presets() {
        let argv = launch_obj(params).argv;
        assert!(argv[..2] == ["-m", m.as_str()], "{name}");
        assert!(
            argv.contains(&"--metrics".to_string()) && argv.contains(&"--offline".to_string()),
            "{name}"
        );
    }
}

#[test]
fn test_metrics_offline_and_loopback_hardening_are_always_present_llamacpp() {
    let argv = launch(json!({})).argv;
    let count = |a: &[String], f: &str| a.iter().filter(|x| *x == f).count();
    assert!(count(&argv, "--metrics") == 1 && count(&argv, "--offline") == 1);
    assert!(after(&argv, "--host") == "0.0.0.0" && after(&argv, "--port") == "8080");
    assert_eq!(after(&argv, "--cors-origins"), "localhost"); // measured: the engine's default reflects any origin
    let explicit = launch(json!({"cors_origins": "*"})).argv;
    assert!(count(&explicit, "--cors-origins") == 1 && after(&explicit, "--cors-origins") == "*");
}

/// Loose files, the first shard of a split set, and files inside an HF-cache repo (3 parametrized rows).
#[test]
fn test_model_paths_pass_through_llamacpp() {
    for path in [
        format!("{MODEL_MOUNT}/a.gguf"),
        format!("{MODEL_MOUNT}/qwen/Qwen3-30B-A3B-Q4_K_M-00001-of-00003.gguf"),
        format!("{MODEL_MOUNT}/snapshots/0123456789abcdef/sub dir/model.Q8_0.gguf"),
    ] {
        let spec = adapter()
            .py_build_launch_ctx(&ctx(Obj::new(), &llama::hw(), Some(&path), "demo", "gguf"))
            .unwrap_or_else(|e| panic!("{e}"));
        assert!(spec.argv[..2] == ["-m", path.as_str()], "{path}");
    }
}

/// 10 parametrized rows.
#[test]
fn test_bad_model_paths_are_refused_llamacpp() {
    let hw = llama::hw();
    let paths: Vec<Option<String>> = vec![
        None,
        Some(String::new()),
        Some("/models".into()),
        Some("/models/".into()),
        Some("/etc/passwd".into()),
        Some("models/a.gguf".into()),
        Some(format!("{MODEL_MOUNT}/../etc/x.gguf")),
        Some("-m".into()),
        Some(format!("{MODEL_MOUNT}/a\n.gguf")),
        Some("/models2/a.gguf".into()),
    ];
    for path in paths {
        let mut c = ctx(Obj::new(), &hw, path.as_deref(), "s", "gguf");
        c.model = "m".into();
        raises(&c, "model_path");
    }
}

/// 3 parametrized rows.
#[test]
fn test_only_gguf_is_loadable_llamacpp() {
    let m = model();
    for fmt in ["hf", "plain", "ollama"] {
        raises(
            &ctx(Obj::new(), &llama::hw(), Some(&m), "demo", fmt),
            "GGUF files only",
        );
    }
}

#[test]
fn test_invalid_params_raise_llamacpp() {
    let m = model();
    raises(
        &ctx(
            p(json!({"bogus": 1})),
            &llama::hw(),
            Some(&m),
            "demo",
            "gguf",
        ),
        "unknown parameter",
    );
    raises(
        &ctx(
            p(json!({"ctx_size": -5})),
            &llama::hw(),
            Some(&m),
            "demo",
            "gguf",
        ),
        "ctx_size",
    );
}

/// 5 parametrized rows.
#[test]
fn test_bad_served_names_are_refused_llamacpp() {
    let m = model();
    for served in ["", "  ", "a,b", "-x", "a\nb"] {
        raises(
            &ctx(Obj::new(), &llama::hw(), Some(&m), served, "gguf"),
            "served name",
        );
    }
}

#[test]
fn test_aliases_put_the_console_name_first_and_drop_duplicates_llamacpp() {
    let argv = launch(json!({"served_model_name": ["extra", "demo", "extra2"]})).argv;
    assert_eq!(after(&argv, "--alias"), "demo,extra,extra2");
}

#[test]
fn test_api_key_travels_in_the_environment_never_in_argv_llamacpp() {
    let spec = launch(json!({"api_key": "s3cr3t-token-value"}));
    assert_eq!(
        Value::Object(spec.env.clone()),
        json!({"LLAMA_API_KEY": "s3cr3t-token-value"})
    );
    assert!(!spec.argv.join(" ").contains("s3cr3t-token-value"));
    assert!(!spec.argv.contains(&"--api-key".to_string()));
}

#[test]
fn test_bools_follow_the_negation_table_llamacpp() {
    let on: BTreeSet<String> =
        launch(json!({"jinja": true, "warmup": true, "swa_full": true, "embedding": true}))
            .argv
            .into_iter()
            .collect();
    assert!(
        ["--jinja", "--warmup", "--swa-full", "--embedding"]
            .iter()
            .all(|f| on.contains(*f))
    );
    let off: BTreeSet<String> = launch(
        json!({"jinja": false, "warmup": false, "swa_full": false, "embedding": false,
                                              "cpu_moe": false, "kv_unified": false}),
    )
    .argv
    .into_iter()
    .collect();
    assert!(
        ["--no-jinja", "--no-warmup", "--no-kv-unified"]
            .iter()
            .all(|f| off.contains(*f))
    );
    let never = [
        "--swa-full",
        "--embedding",
        "--cpu-moe",
        "--no-swa-full",
        "--no-embedding",
        "--no-cpu-moe",
    ];
    assert!(never.iter().all(|f| !off.contains(*f)));
}

#[test]
fn test_floats_render_compactly_llamacpp() {
    let argv =
        launch(json!({"temperature": 0.6, "rope_scale": 2.5, "min_p": 0.05, "slot_prompt_similarity": 0.1})).argv;
    for (flag, value) in [
        ("--temp", "0.6"),
        ("--rope-scale", "2.5"),
        ("--min-p", "0.05"),
        ("--slot-prompt-similarity", "0.1"),
    ] {
        assert_eq!(after(&argv, flag), value);
    }
}

// ------------------------------------------------------------------ chat templates
#[test]
fn test_custom_template_goes_through_a_generated_file_llamacpp() {
    let text = "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}\n{% endfor %}";
    let spec = launch(json!({"chat_template_jinja": text}));
    assert_eq!(
        Value::Object(spec.files.clone()),
        json!({CHAT_TEMPLATE_PATH: text})
    );
    let argv = spec.argv;
    assert_eq!(after(&argv, "--chat-template-file"), CHAT_TEMPLATE_PATH);
    assert!(!argv.contains(&text.to_string()));
    assert!(pos(&argv, "--jinja") < pos(&argv, "--chat-template-file")); // the help: --jinja must come first
}

#[test]
fn test_inline_custom_template_gets_jinja_first_but_a_builtin_name_does_not_llamacpp() {
    let custom = launch(json!({"chat_template": "{{ messages }}"})).argv;
    assert!(pos(&custom, "--jinja") < pos(&custom, "--chat-template"));
    let builtin = launch(json!({"chat_template": "chatml"})).argv;
    assert!(!builtin.contains(&"--jinja".to_string()));
    let explicit_off = launch(json!({"chat_template_jinja": "{{ x }}", "jinja": false})).argv;
    assert!(
        explicit_off.contains(&"--no-jinja".to_string())
            && !explicit_off.contains(&"--jinja".to_string())
    );
}

// ------------------------------------------------------------------ v1 bridge
#[test]
fn test_v1_build_launch_bridges_to_the_context_path_llamacpp() {
    let a = adapter();
    let m = model();
    let spec = a
        .build_launch(
            &m,
            &p(json!({"ctx_size": 4096})),
            &llama::hw(),
            "demo",
            "/hf",
        )
        .expect("spec");
    assert!(spec.argv[..2] == ["-m", m.as_str()] && spec.argv.contains(&"--ctx-size".to_string()));
    let err = a
        .build_launch(
            "example-org/model",
            &Obj::new(),
            &llama::hw(),
            "demo",
            "/hf",
        )
        .expect_err("refused");
    assert!(
        err.status == 422 && err.detail.contains("model_path"),
        "{err:?}"
    );
}
