//! Port of backend/tests/sglang/test_sglang_adapter.py (offline: only the adapter; no docker, no network), plus native
//! property tests for the parsers.

use std::collections::{BTreeMap, BTreeSet};

use ec_hub::adapters::base::{
    Compat, EngineAdapter, Hardware, LaunchContext, MODEL_MOUNT, ModelInfo,
};
use ec_hub::adapters::sglang::SglangAdapter;
use ec_hub::adapters::sglang_params::DEFAULT_IMAGE;
use ec_hub::adapters::sglang_parsers::histogram_quantile;
use proptest::prelude::*;
use regex::{Regex, RegexBuilder};
use serde_json::{Map, Value, json};

use crate::ws_a1_kit::{approx, fixture, get, hw_n, keys, model, obj};

const UNITS: &[&str] = &[
    "GiB", "GB", "MiB", "bytes", "tokens", "requests", "GPUs", "s",
];

fn a() -> SglangAdapter {
    SglangAdapter::new()
}

fn hw() -> Hardware {
    hw_n(1, "12.0", None)
}

fn mi() -> ModelInfo {
    model("Qwen/Qwen3.6-35B-A3B")
}

fn codes_on(m: &ModelInfo, p: Value, h: &Hardware) -> BTreeMap<String, Compat> {
    a().compatibility(m, &obj(p), h)
        .into_iter()
        .map(|c| (c.code.clone(), c))
        .collect()
}

fn codes(m: &ModelInfo, p: Value) -> BTreeMap<String, String> {
    codes_on(m, p, &hw())
        .into_iter()
        .map(|(k, c)| (k, c.level))
        .collect()
}

// ---- catalog ----
#[test]
fn test_catalog_size_and_uniqueness_sglang() {
    let cat = a().param_catalog();
    assert!(cat.len() >= 100);
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
fn test_catalog_shape_sglang() {
    let flag = Regex::new(r"^(?:--[a-z0-9][a-z0-9-]*|env:[A-Z0-9_]+)$").unwrap();
    for p in a().param_catalog() {
        assert!(
            flag.is_match(&p.flag) && p.docs_url.is_some() && !p.help.is_empty(),
            "{}",
            p.key
        );
        let docs = p.docs_url.clone().unwrap();
        if p.flag.starts_with("env:") {
            assert_eq!(
                docs, "https://docs.sglang.io/docs/references/environment_variables",
                "{}",
                p.key
            );
        } else {
            assert!(
                docs.starts_with("https://docs.sglang.io/docs/advanced_features/server_arguments#"),
                "{}",
                p.key
            );
        }
        if p.type_ == "enum" {
            let choices = p.choices.clone().unwrap();
            assert!(!choices.is_empty());
            assert!(
                p.default.is_null()
                    || p.default
                        .as_str()
                        .is_some_and(|d| choices.iter().any(|c| c == d))
            );
        }
        if let (Some(lo), Some(hi)) = (p.min, p.max) {
            assert!(lo < hi);
        }
    }
    let keys: BTreeSet<String> = a().param_catalog().into_iter().map(|p| p.key).collect();
    for k in [
        "mem_fraction_static",
        "tool_call_parser",
        "enable_metrics",
        "tp_size",
        "quantization",
        "speculative_algorithm",
        "enable_hierarchical_cache",
        "enable_lora",
        "fp4_gemm_backend",
        "max_mamba_cache_size",
        "default_chat_template_kwargs",
        "media_url_max_file_size_mb",
    ] {
        assert!(keys.contains(k), "{k}");
    }
}

#[test]
fn test_levels_and_units_sglang() {
    let cat = a().param_catalog();
    for p in &cat {
        let level = p.level.as_deref().unwrap_or("");
        assert!(
            ["basic", "advanced", "expert"].contains(&level),
            "{}",
            p.key
        );
        if level == "basic" {
            assert!(!p.advanced, "{}", p.key);
        }
        if level == "expert" {
            assert!(p.advanced, "{}", p.key);
        }
        if let Some(u) = &p.unit {
            assert!(UNITS.contains(&u.as_str()), "{} {u}", p.key);
        }
    }
    let basic: BTreeSet<&str> = cat
        .iter()
        .filter(|p| p.level.as_deref() == Some("basic"))
        .map(|p| p.key.as_str())
        .collect();
    assert!((8..=14).contains(&basic.len()));
    for k in [
        "context_length",
        "mem_fraction_static",
        "tp_size",
        "max_running_requests",
        "quantization",
        "tool_call_parser",
        "reasoning_parser",
    ] {
        assert!(basic.contains(k), "{k}");
    }
    let by = |k: &str| cat.iter().find(|p| p.key == k).unwrap().unit.clone();
    assert!(
        by("context_length").as_deref() == Some("tokens")
            && by("watchdog_timeout").as_deref() == Some("s")
    );
    assert!(
        by("max_running_requests").as_deref() == Some("requests")
            && by("cpu_offload_gb").as_deref() == Some("GB")
    );
}

#[test]
fn test_retired_and_added_keys_sglang() {
    assert!(!a().param_catalog().iter().any(|p| p.key == "api_key"));
    let msg = a().validate(&obj(json!({"api_key": "s3cret"})));
    assert!(
        msg.len() == 1 && msg[0].contains("authenticating proxy") && !msg[0].contains("s3cret")
    );
}

#[test]
fn test_v2_adapter_fields_sglang() {
    let a = a();
    let mut f = a.model_formats();
    f.sort();
    assert_eq!(f, ["hf", "plain"]);
    assert!(a.images_by_arch().is_empty() && a.image_for("aarch64") == DEFAULT_IMAGE);
    assert_eq!(DEFAULT_IMAGE, "lmsysorg/sglang:v0.5.20-cu130");
    assert_eq!(a.metric_prefixes(), Some(vec!["sglang:".to_string()]));
    assert!(a.preset_info().is_empty());
}

#[test]
fn test_catalog_returns_copies_sglang() {
    let a = a();
    let mut first = a.param_catalog();
    first[0].label = "mutated".into();
    assert_ne!(a.param_catalog()[0].label, "mutated");
}

// ---- presets / validate ----
/// `test_presets_validate[balanced|max-throughput|low-latency|long-context|tool-agent|reasoning]`.
#[test]
fn test_presets_validate_sglang() {
    let a = a();
    let presets = a.presets();
    for name in [
        "balanced",
        "max-throughput",
        "low-latency",
        "long-context",
        "tool-agent",
        "reasoning",
    ] {
        let p = &presets.iter().find(|(n, _)| n == name).unwrap().1;
        assert!(a.validate(p).is_empty(), "{name}");
    }
}

#[test]
fn test_validate_errors_sglang() {
    let v = |p: Value| a().validate(&obj(p));
    assert!(v(json!({"nope": 1})).contains(&"unknown parameter 'nope'".to_string()));
    for p in [
        json!({"mem_fraction_static": 1.5}),
        json!({"schedule_policy": "bogus"}),
        json!({"tp_size": true}),
        json!({"tp_size": 1.5}),
        json!({"revision": "--evil"}),
        json!({"speculative_algorithm": "EAGLE3"}),
        json!({"speculative_num_steps": 3}),
        json!({"enable_dp_attention": true, "tp_size": 2, "dp_size": 1}),
        json!({"tp_size": 4, "ep_size": 3}),
        json!({"enable_hierarchical_cache": true, "disable_radix_cache": true}),
        json!({"json_model_override_args": "{bad"}),
        json!({"quantization": "marlin"}), // no longer a choice in 0.5.20
        json!({"attention_backend": "cutlass_mla"}), // gone in 0.5.20
    ] {
        assert!(!v(p.clone()).is_empty(), "{p}");
    }
    assert!(v(json!({"speculative_algorithm": "NEXTN", "speculative_num_steps": 3})).is_empty());
}

// ---- build_launch ----
#[test]
fn test_build_launch_golden_sglang() {
    let spec = a()
        .build_launch(
            "Qwen/Qwen3.6-35B-A3B",
            &obj(json!({"quantization": "fp8", "context_length": 262_144, "mem_fraction_static": 0.85,
                        "tool_call_parser": "qwen3_coder", "reasoning_parser": "qwen3", "disable_radix_cache": false,
                        "enable_mixed_chunk": true, "schedule_conservativeness": 1})),
            &hw(),
            "qwen",
            "/root/.cache/huggingface",
        )
        .unwrap();
    assert!(spec.image == "lmsysorg/sglang:v0.5.20-cu130" && !spec.image.contains(":latest"));
    assert_eq!(
        spec.argv,
        [
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            "Qwen/Qwen3.6-35B-A3B",
            "--host",
            "0.0.0.0",
            "--port",
            "30000",
            "--served-model-name",
            "qwen",
            "--enable-metrics",
            "--mem-fraction-static",
            "0.85",
            "--context-length",
            "262144",
            "--quantization",
            "fp8",
            "--schedule-conservativeness",
            "1.0",
            "--enable-mixed-chunk",
            "--tool-call-parser",
            "qwen3_coder",
            "--reasoning-parser",
            "qwen3",
        ]
    );
    assert_eq!(
        Value::Object(spec.env.clone()),
        json!({"HF_HUB_DISABLE_TELEMETRY": "1", "HF_HOME": "/root/.cache/huggingface"})
    );
    assert!(spec.needs_hf_cache);
    assert_eq!(
        (
            spec.container_port,
            spec.health_path.as_str(),
            spec.metrics_path.as_str()
        ),
        (30000, "/health", "/metrics")
    );
}

#[test]
fn test_env_param_becomes_a_container_variable_not_an_argument_sglang() {
    let spec = a()
        .build_launch(
            "m",
            &obj(json!({"allow_longer_context_len": true, "context_length": 1_048_576})),
            &hw(),
            "s",
            "/c",
        )
        .unwrap();
    assert_eq!(spec.env["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"], "1");
    assert!(
        !spec
            .argv
            .iter()
            .any(|x| x.contains("ALLOW_OVERWRITE") || x.starts_with("env:"))
    );
    assert!(
        spec.argv.contains(&"--context-length".to_string())
            && spec.argv.contains(&"1048576".to_string())
    );
    let off = a()
        .build_launch(
            "m",
            &obj(json!({"allow_longer_context_len": false})),
            &hw(),
            "s",
            "/c",
        )
        .unwrap();
    assert_eq!(off.env["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"], "0");
}

#[test]
fn test_metrics_forced_and_served_override_sglang() {
    let spec = a()
        .build_launch(
            "m",
            &obj(json!({"enable_metrics": false, "served_model_name": "mine"})),
            &hw(),
            "x",
            "/c",
        )
        .unwrap();
    assert_eq!(
        spec.argv
            .iter()
            .filter(|x| *x == "--enable-metrics")
            .count(),
        1
    );
    let i = spec
        .argv
        .iter()
        .position(|x| x == "--served-model-name")
        .unwrap();
    assert_eq!(spec.argv[i + 1], "mine");
}

#[test]
fn test_build_launch_list_json_and_rejects_sglang() {
    let spec = a()
        .build_launch(
            "m",
            &obj(json!({"lora_paths": ["a=/x", "b=/y"], "json_model_override_args": {"k": 1},
                        "cuda_graph_max_bs_decode": 64, "default_chat_template_kwargs": {"enable_thinking": true}})),
            &hw(),
            "s",
            "/c",
        )
        .unwrap();
    let i = spec.argv.iter().position(|x| x == "--lora-paths").unwrap();
    assert_eq!(spec.argv[i + 1..i + 3], ["a=/x", "b=/y"]);
    assert!(
        spec.argv.contains(&"{\"k\":1}".to_string())
            && spec
                .argv
                .contains(&"--cuda-graph-max-bs-decode".to_string())
    );
    assert!(
        spec.argv
            .contains(&"{\"enable_thinking\":true}".to_string())
    );
    assert!(
        a().build_launch("--x", &Map::new(), &hw(), "s", "/c")
            .is_err()
    );
    assert!(
        a().build_launch("m", &obj(json!({"bad": 1})), &hw(), "s", "/c")
            .is_err()
    );
}

#[test]
fn test_no_secret_can_reach_the_command_line_sglang() {
    let e = a()
        .build_launch("m", &obj(json!({"api_key": "s3cret"})), &hw(), "s", "/c")
        .unwrap_err();
    assert!(e.detail.contains("authenticating proxy"));
    let spec = a()
        .build_launch("m", &Map::new(), &hw(), "s", "/c")
        .unwrap();
    assert!(!spec.argv.iter().any(|x| x.to_lowercase().contains("key")));
}

fn ctx(model: &str, fmt: &str, path: Option<&str>, params: Map<String, Value>) -> LaunchContext {
    LaunchContext {
        model: model.into(),
        model_format: fmt.into(),
        model_path: path.map(str::to_string),
        served_name: "chat".into(),
        hf_cache_container_path: "/hf".into(),
        params,
        hw: hw(),
    }
}

#[test]
fn test_context_launch_for_hf_matches_build_launch_sglang() {
    let params = a()
        .presets()
        .into_iter()
        .find(|(n, _)| n == "tool-agent")
        .unwrap()
        .1;
    assert_eq!(
        a().build_launch_ctx(&ctx("org/m", "hf", None, params.clone()))
            .unwrap(),
        a().build_launch("org/m", &params, &hw(), "chat", "/hf")
            .unwrap()
    );
}

#[test]
fn test_plain_model_directory_is_launched_by_path_sglang() {
    let spec = a()
        .build_launch_ctx(&ctx(
            "my-model-dir",
            "plain",
            Some(MODEL_MOUNT),
            obj(json!({"context_length": 8192, "revision": "main"})),
        ))
        .unwrap();
    let av = &spec.argv;
    assert_eq!(
        av[..5],
        [
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            "/models"
        ]
    );
    let i = av.iter().position(|x| x == "--served-model-name").unwrap();
    assert_eq!(av[i + 1], "chat");
    // a directory has no hub revision
    assert!(
        !av.contains(&"--revision".to_string()) && av.contains(&"--context-length".to_string())
    );
    assert!(!spec.needs_hf_cache);
    assert!(!spec.env.contains_key("HF_HOME") && spec.env["HF_HUB_OFFLINE"] == "1");
    assert!(!av.join(" ").contains("my-model-dir"));
    assert_eq!(
        a().build_launch_ctx(&ctx("org/m", "plain", None, Map::new()))
            .unwrap()
            .argv[4],
        MODEL_MOUNT
    );
}

/// `test_unsupported_formats_are_refused[gguf]` and `[ollama]`.
#[test]
fn test_unsupported_formats_are_refused_sglang() {
    for fmt in ["gguf", "ollama"] {
        let e = a()
            .build_launch_ctx(&ctx("org/m", fmt, Some("/models/x.gguf"), Map::new()))
            .unwrap_err();
        assert!(e.detail.contains("cannot load"), "{fmt}");
    }
}

// ---- compatibility ----
#[test]
fn test_compat_gpus_and_mem_fraction_sglang() {
    assert_eq!(
        codes_on(&mi(), json!({"tp_size": 2}), &hw_n(1, "12.0", None))["gpus_insufficient"].level,
        "block"
    );
    assert!(
        !codes_on(&mi(), json!({"tp_size": 2}), &hw_n(2, "12.0", None))
            .contains_key("gpus_insufficient")
    );
    assert_eq!(
        codes(&mi(), json!({"mem_fraction_static": 0.95}))["mem_fraction_high"],
        "warn"
    );
    assert!(!codes(&mi(), json!({"mem_fraction_static": 0.85})).contains_key("mem_fraction_high"));
}

#[test]
fn test_compat_unbalanced_tensor_parallel_memory_sglang() {
    let busy = hw_n(2, "12.0", Some(vec![35.0, 0.6]));
    let c = codes_on(&mi(), json!({"tp_size": 2}), &busy);
    assert!(
        c["tp_memory_unbalanced"].level == "block"
            && c["tp_memory_unbalanced"].message.contains("unbalanced")
    );
    // within 10 %
    assert!(
        !codes_on(
            &mi(),
            json!({"tp_size": 2}),
            &hw_n(2, "12.0", Some(vec![90.0, 85.0]))
        )
        .contains_key("tp_memory_unbalanced")
    );
    assert!(!codes_on(&mi(), json!({}), &busy).contains_key("tp_memory_unbalanced")); // tp 1 is not affected
}

#[test]
fn test_compat_quant_rules_sglang() {
    let q = |repo: &str, quant: &str, moe: bool| ModelInfo {
        quantization: Some(quant.into()),
        is_moe: moe,
        ..model(repo)
    };
    assert_eq!(
        codes(&q("Qwen/Qwen3.6-35B-A3B", "nvfp4", true), json!({}))["nvfp4_ok_sglang"],
        "ok"
    );
    assert_eq!(
        codes(&q("openai/gpt-oss-120b", "mxfp4", true), json!({}))["mxfp4_gptoss_engine"],
        "warn"
    );
    assert_eq!(
        codes(&q("x/moe", "mxfp4", true), json!({}))["mxfp4_moe_sm120"],
        "block"
    );
    assert!(
        !codes_on(
            &q("x/moe", "mxfp4", true),
            json!({}),
            &hw_n(1, "10.0", None)
        )
        .contains_key("mxfp4_moe_sm120")
    );
    assert_eq!(
        codes(
            &q("Qwen/Qwen3.6-35B-A3B", "nvfp4", false),
            json!({"quantization": "fp8"})
        )["quant_override_prequantized"],
        "warn"
    );
    assert_eq!(
        codes(&mi(), json!({"quantization": "modelopt_fp4"}))["modelopt_unquantized"],
        "warn"
    );
}

#[test]
fn test_compat_attention_backend_sglang() {
    let c = codes_on(&mi(), json!({"attention_backend": "fa3"}), &hw());
    assert_eq!(c["attention_backend_hopper_only"].level, "block");
    // the engine's own assertion
    assert!(
        c["attention_backend_hopper_only"]
            .message
            .contains("SM>=80 and SM<=90")
    );
    assert_eq!(
        codes(&mi(), json!({"decode_attention_backend": "flashmla"}))["attention_backend_hopper_only"],
        "block"
    );
    assert_eq!(
        codes(&mi(), json!({"attention_backend": "trtllm_mla"}))["attention_backend_dc_blackwell"],
        "warn"
    );
    assert_eq!(
        codes(&mi(), json!({"attention_backend": "fa4"}))["attention_backend_dc_blackwell"],
        "warn"
    );
    // trtllm_mha is explicitly allowed on SM120 by the image (SM90, SM100 and SM120 checks)
    assert!(
        !codes(&mi(), json!({"attention_backend": "trtllm_mha"}))
            .keys()
            .any(|c| c.starts_with("attention_backend"))
    );
    assert!(
        !codes(&mi(), json!({"attention_backend": "flashinfer"}))
            .keys()
            .any(|c| c.starts_with("attention_backend"))
    );
    assert!(
        !codes_on(
            &mi(),
            json!({"attention_backend": "fa3"}),
            &hw_n(1, "9.0", None)
        )
        .contains_key("attention_backend_hopper_only")
    );
}

#[test]
fn test_compat_trtllm_mha_with_fp8_kv_on_sm120_sglang() {
    let p = json!({"attention_backend": "trtllm_mha", "kv_cache_dtype": "fp8_e4m3"});
    assert_eq!(codes(&mi(), p.clone())["trtllm_mha_fp8_kv_sm120"], "block");
    assert!(!codes_on(&mi(), p, &hw_n(1, "10.0", None)).contains_key("trtllm_mha_fp8_kv_sm120"));
    assert!(
        !codes(&mi(), json!({"attention_backend": "trtllm_mha"}))
            .contains_key("trtllm_mha_fp8_kv_sm120")
    );
}

/// Image source: handle_linear_attn_backend raises for SM100+ (capability major >= 10, so sm_120 too).
#[test]
fn test_compat_flashinfer_linear_attention_needs_a_bf16_state_sglang() {
    let k = "linear_attn_flashinfer_needs_bf16_state";
    let bad = codes_on(
        &mi(),
        json!({"linear_attn_decode_backend": "flashinfer", "mamba_ssm_dtype": "float32"}),
        &hw(),
    );
    assert_eq!(bad[k].level, "block");
    assert!(
        bad[k]
            .message
            .contains("requires --mamba-ssm-dtype bfloat16")
    );
    assert_eq!(
        codes(&mi(), json!({"linear_attn_backend": "flashinfer"}))[k],
        "warn"
    );
    assert!(
        !codes(
            &mi(),
            json!({"linear_attn_decode_backend": "flashinfer", "mamba_ssm_dtype": "bfloat16"})
        )
        .contains_key(k)
    );
    assert!(!codes(&mi(), json!({"linear_attn_decode_backend": "triton"})).contains_key(k));
    assert!(
        !codes_on(
            &mi(),
            json!({"linear_attn_decode_backend": "flashinfer", "mamba_ssm_dtype": "float32"}),
            &hw_n(1, "9.0", None)
        )
        .contains_key(k)
    );
}

#[test]
fn test_compat_tool_parser_dialect_sglang() {
    let k = "tool_parser_dialect_mismatch";
    assert_eq!(codes(&mi(), json!({"tool_call_parser": "qwen"}))[k], "warn");
    assert!(!codes(&mi(), json!({"tool_call_parser": "qwen3_coder"})).contains_key(k));
    assert_eq!(
        codes(
            &model("Qwen/Qwen3-32B"),
            json!({"tool_call_parser": "qwen3_coder"})
        )[k],
        "warn"
    );
    assert!(
        !codes(
            &model("Qwen/Qwen3-32B"),
            json!({"tool_call_parser": "qwen25"})
        )
        .contains_key(k)
    );
    // a size suffix such as "-8B" must not turn Qwen3-8B into a "Qwen3.x" XML model
    assert!(
        !codes(
            &model("Qwen/Qwen3-8B"),
            json!({"tool_call_parser": "hermes"})
        )
        .contains_key(k)
    );
    assert_eq!(
        codes(
            &model("openai/gpt-oss-20b"),
            json!({"tool_call_parser": "hermes"})
        )[k],
        "warn"
    );
    assert_eq!(codes(&mi(), json!({}))["no_tool_parser"], "warn");
}

#[test]
fn test_compat_misc_sglang() {
    let moe = ModelInfo {
        is_moe: true,
        ..mi()
    };
    let two = hw_n(2, "12.0", None);
    assert_eq!(
        codes_on(
            &moe,
            json!({"enable_dp_attention": true, "tp_size": 2, "dp_size": 1}),
            &two
        )["dp_attention_dp_ne_tp"]
            .level,
        "block"
    );
    assert_eq!(
        codes_on(
            &mi(),
            json!({"enable_dp_attention": true, "tp_size": 2, "dp_size": 2}),
            &two
        )["dp_attention_unsupported_arch"]
            .level,
        "warn"
    );
    let short = ModelInfo {
        max_position_embeddings: Some(32768),
        ..mi()
    };
    assert_eq!(
        codes(&short, json!({"context_length": 65536}))["context_beyond_native"],
        "warn"
    );
    let refuse = codes_on(&short, json!({"context_length": 65536}), &hw());
    assert!(
        refuse["context_beyond_native"]
            .message
            .contains("allow_longer_context_len")
    );
    let allowed = codes_on(
        &short,
        json!({"context_length": 65536, "allow_longer_context_len": true}),
        &hw(),
    );
    assert!(
        allowed["context_beyond_native"]
            .message
            .contains("incorrect model outputs")
    );
    let gated = ModelInfo {
        gated: true,
        ..mi()
    };
    assert_eq!(codes(&gated, json!({}))["gated_model"], "warn");
    assert_eq!(
        codes(&mi(), json!({"enable_metrics": false}))["metrics_forced"],
        "warn"
    );
    assert_eq!(
        codes(&mi(), json!({"kv_cache_dtype": "nvfp4"}))["kv_fp4_experimental"],
        "warn"
    );
    assert!(!codes(&mi(), json!({})).contains_key("api_key_in_argv"));
}

/// The repository is public: no site-specific wording in anything a user is shown.
#[test]
fn test_every_message_is_generic_text_sglang() {
    let banned =
        RegexBuilder::new(r"\blab\b|this repo|validated here|/home/|@[a-z0-9-]+\.[a-z]{2,}")
            .case_insensitive(true)
            .build()
            .unwrap();
    let models = [
        mi(),
        ModelInfo {
            quantization: Some("mxfp4".into()),
            is_moe: true,
            ..model("openai/gpt-oss-120b")
        },
        ModelInfo {
            quantization: Some("nvfp4".into()),
            gated: true,
            ..model("Qwen/Qwen3-8B")
        },
    ];
    let params = [
        json!({}),
        json!({"tp_size": 2, "attention_backend": "fa3", "mem_fraction_static": 0.95, "tool_call_parser": "hermes",
               "kv_cache_dtype": "fp8_e4m3", "enable_dp_attention": true, "log_requests": true,
               "trust_remote_code": true, "cpu_offload_gb": 2}),
    ];
    let mut seen = 0;
    for m in &models {
        for p in &params {
            for c in a().compatibility(m, &obj(p.clone()), &hw_n(2, "12.0", Some(vec![90.0, 20.0])))
            {
                seen += 1;
                assert!(!banned.is_match(&c.message), "{c:?}");
            }
        }
    }
    assert!(seen > 10);
}

// ---- memory model ----
#[test]
fn test_memory_model_sglang() {
    let long = ModelInfo {
        max_position_embeddings: Some(262_144),
        ..mi()
    };
    let m = a().memory_model(
        &long,
        &obj(json!({"mem_fraction_static": 0.85, "kv_cache_dtype": "fp8_e4m3", "tp_size": 2, "max_running_requests": 8})),
    );
    assert!(
        get(&m, "mem_fraction") == Some(0.85)
            && get(&m, "kv_bytes_per_elem") == Some(1.0)
            && get(&m, "tp") == Some(2.0)
    );
    assert!(
        get(&m, "max_len") == Some(262_144.0)
            && get(&m, "max_seqs") == Some(8.0)
            && get(&m, "overhead_gib") == Some(0.0)
    );
    assert!((get(&m, "nonstatic_reserve_frac").unwrap() - 0.15).abs() < 1e-9);
    // 0.5.20: fraction x memory free at launch
    assert!(get(&m, "share_of_free") == Some(1.0) && get(&m, "claims_share").is_none());
    let d = a().memory_model(&mi(), &Map::new());
    assert!(
        get(&d, "mem_fraction") == Some(0.85)
            && get(&d, "kv_bytes_per_elem") == Some(2.0)
            && get(&d, "mem_fraction_is_default") == Some(1.0)
    );
    assert_eq!(
        get(
            &a().memory_model(&mi(), &obj(json!({"kv_cache_dtype": "nvfp4"}))),
            "kv_bytes_per_elem"
        ),
        Some(0.5)
    );
}

// ---- metrics ----
const PROM: &str = r#"# HELP sglang:num_running_reqs The number of running requests
# TYPE sglang:num_running_reqs gauge
sglang:num_running_reqs{model_name="qwen",engine_type="unified",tp_rank="0",pp_rank="0"} 3.0
sglang:num_queue_reqs{model_name="qwen",tp_rank="0"} 5.0
sglang:token_usage{model_name="qwen",tp_rank="0"} 0.42
sglang:cache_hit_rate{model_name="qwen",tp_rank="0"} 0.75
sglang:gen_throughput{model_name="qwen",tp_rank="0"} 812.5
sglang:spec_accept_length{model_name="qwen",tp_rank="0"} 2.6
sglang:spec_accept_rate{model_name="qwen",tp_rank="0"} 0.65
sglang:prompt_tokens_total{model_name="qwen"} 120000.0
sglang:generation_tokens_total{model_name="qwen"} 45000.0
sglang:num_retracted_reqs{model_name="qwen",tp_rank="0"} 7.0
sglang:num_retracted_requests_total{model_name="qwen"} 2.0
sglang:time_to_first_token_seconds_bucket{le="0.1",model_name="qwen"} 10.0
sglang:time_to_first_token_seconds_bucket{le="0.5",model_name="qwen"} 90.0
sglang:time_to_first_token_seconds_bucket{le="1.0",model_name="qwen"} 100.0
sglang:time_to_first_token_seconds_bucket{le="+Inf",model_name="qwen"} 100.0
sglang:time_to_first_token_seconds_sum{model_name="qwen"} 40.0
sglang:time_to_first_token_seconds_count{model_name="qwen"} 100.0
sglang:inter_token_latency_seconds_bucket{le="0.01",model_name="qwen"} 50.0
sglang:inter_token_latency_seconds_bucket{le="0.02",model_name="qwen"} 100.0
sglang:inter_token_latency_seconds_bucket{le="+Inf",model_name="qwen"} 100.0
sglang:e2e_request_latency_seconds_bucket{le="1.0",model_name="qwen"} 20.0
sglang:e2e_request_latency_seconds_bucket{le="5.0",model_name="qwen"} 100.0
sglang:e2e_request_latency_seconds_bucket{le="+Inf",model_name="qwen"} 100.0
"#;

#[test]
fn test_parse_metrics_sglang() {
    let m = a().parse_metrics(PROM);
    let g = |k: &str| get(&m, k).unwrap_or_else(|| panic!("{k}"));
    assert!(g("requests_running") == 3.0 && g("requests_waiting") == 5.0);
    assert!(approx(g("kv_cache_usage_pct"), 42.0) && approx(g("prefix_cache_hit_pct"), 75.0));
    assert!(g("generation_tps") == 812.5 && g("prompt_tokens_total") == 120_000.0);
    assert_eq!(g("generation_tokens_total"), 45000.0);
    assert_eq!(g("preemptions_total"), 2.0); // the cumulative counter, never the per-interval gauge (7)
    assert!(approx(g("spec_decode_accept_pct"), 65.0) && g("spec_accept_length") == 2.6);
    assert!(approx(g("ttft_p50_s"), 0.1 + 0.4 * (50.0 - 10.0) / 80.0));
    assert!(0.5 < g("ttft_p95_s") && g("ttft_p95_s") < 1.0);
    assert!(approx(g("itl_p50_s"), 0.01));
    assert!(approx(g("e2e_p50_s"), 1.0 + 4.0 * (50.0 - 20.0) / 80.0));
    assert!(get(&m, "prompt_tps").is_none());
}

#[test]
fn test_gauge_without_the_counter_is_not_reported_as_a_total_sglang() {
    assert!(
        get(
            &a().parse_metrics("sglang:num_retracted_reqs{tp_rank=\"0\"} 7\n"),
            "preemptions_total"
        )
        .is_none()
    );
}

/// A real SGLang 0.5.20 scrape (idle, tensor-parallel 2).
#[test]
fn test_real_v0520_scrape_sglang() {
    let m = a().parse_metrics(&fixture("sglang", "metrics_v0.5.20.prom"));
    let g = |k: &str| get(&m, k).unwrap_or_else(|| panic!("{k}"));
    assert!(g("requests_running") == 0.0 && g("requests_waiting") == 0.0);
    assert!(
        g("kv_cache_usage_pct") == 0.0
            && g("prefix_cache_hit_pct") == 0.0
            && g("generation_tps") == 0.0
    );
    // the token counters are split by is_streaming and summed
    assert_eq!(
        g("prompt_tokens_total"),
        (21_261_962_i64 + 111_648_670) as f64
    );
    assert_eq!(g("generation_tokens_total"), (720_460_i64 + 819_339) as f64);
    assert!(
        (g("spec_decode_accept_pct") - 14.1667).abs() <= 1e-3 && g("spec_accept_length") == 1.425
    );
    assert!(0.1 < g("ttft_p50_s") && g("ttft_p50_s") <= g("ttft_p95_s") && g("ttft_p95_s") < 5.0);
    assert!(0.0 < g("itl_p50_s") && g("itl_p50_s") < 0.05);
    assert!(0.0 < g("e2e_p50_s") && g("e2e_p50_s") <= g("e2e_p95_s"));
    // the retraction counter is created lazily (first retraction); the per-interval gauge must not stand in for it
    assert!(get(&m, "preemptions_total").is_none());
    let allowed = [
        "requests_running",
        "requests_waiting",
        "kv_cache_usage_pct",
        "prefix_cache_hit_pct",
        "prompt_tokens_total",
        "generation_tokens_total",
        "ttft_p50_s",
        "ttft_p95_s",
        "itl_p50_s",
        "e2e_p50_s",
        "e2e_p95_s",
        "prompt_tps",
        "generation_tps",
        "preemptions_total",
        "spec_decode_accept_pct",
        "spec_accept_length",
    ];
    assert!(keys(&m).iter().all(|k| allowed.contains(k)));
}

/// Every rank reports the pool size but only rank 0 reports the used tokens (as in the live data).
#[test]
fn test_kv_usage_fallback_under_tensor_parallelism_sglang() {
    let text = "sglang:kv_used_tokens{tp_rank=\"0\"} 500\nsglang:max_total_num_tokens{tp_rank=\"0\"} 1000\n\
                sglang:max_total_num_tokens{tp_rank=\"1\"} 1000\n";
    assert!(approx(
        get(&a().parse_metrics(text), "kv_cache_usage_pct").unwrap(),
        50.0
    ));
}

#[test]
fn test_parse_metrics_tolerance_sglang() {
    let a = a();
    assert!(a.parse_metrics("").is_empty());
    assert!(
        a.parse_metrics("garbage line\nsglang:token_usage NaN\n")
            .is_empty()
    );
    let alt = a.parse_metrics("sglang_num_running_reqs{tp_rank=\"0\"} 2\nsglang_prompt_tokens 9\n");
    assert_eq!(
        alt,
        vec![
            ("requests_running".to_string(), 2.0),
            ("prompt_tokens_total".to_string(), 9.0)
        ]
    );
    let two = a.parse_metrics(
        "sglang:num_running_reqs{dp_rank=\"0\"} 2\nsglang:num_running_reqs{dp_rank=\"1\"} 3\n\
         sglang:token_usage{dp_rank=\"0\"} 0.2\nsglang:token_usage{dp_rank=\"1\"} 0.4\n",
    );
    assert!(
        get(&two, "requests_running") == Some(5.0)
            && approx(get(&two, "kv_cache_usage_pct").unwrap(), 30.0)
    );
    let fb = a.parse_metrics("sglang:kv_used_tokens 500\nsglang:max_total_num_tokens 1000\n");
    assert!(approx(get(&fb, "kv_cache_usage_pct").unwrap(), 50.0));
    assert_eq!(
        keys(&a.parse_metrics("sglang:num_running_reqs 0\n")),
        ["requests_running"]
    );
}

#[test]
fn test_histogram_quantile_sglang() {
    assert_eq!(histogram_quantile(0.5, &[]), None);
    assert_eq!(
        histogram_quantile(0.5, &[(1.0, 0.0), (f64::INFINITY, 0.0)]),
        None
    );
    assert_eq!(
        histogram_quantile(0.99, &[(1.0, 5.0), (f64::INFINITY, 10.0)]),
        Some(1.0)
    );
}

// ---- startup log ----
// Lines built from the f-strings of the pinned 0.5.20 source (and, where marked, the 0.5.14 wording); none was
// captured from a live engine.
/// Every `test_startup_log[<line>-expected<N>]` case.
#[test]
fn test_startup_log_sglang() {
    let cases: Vec<(&str, Value)> = vec![
        (
            "[2026-09-23 10:00:00] Load weight begin. avail mem=93.10 GB",
            json!({"phase": "loading_weights", "pct": 0.0}),
        ),
        (
            "Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:10<00:10,  5.0s/it]",
            json!({"phase": "loading_weights", "pct": 50.0}),
        ),
        (
            "Multi-thread loading shards:  25%|██▌       | 1/4 [00:02<00:06]",
            json!({"phase": "loading_weights", "pct": 25.0}),
        ),
        (
            "Load weight end. elapsed=41.30 s, type=Qwen3MoeForCausalLM, dtype=torch.bfloat16, avail mem=58.70 GB, \
             mem usage=34.50 GB.",
            json!({"phase": "loading_weights", "pct": 100.0, "weights_gib": 34.5, "load_seconds": 41.3}),
        ),
        (
            "Load weight end. type=Qwen3MoeForCausalLM, dtype=torch.bfloat16, avail mem=58.70 GB, mem usage=34.50 GB.",
            json!({"phase": "loading_weights", "pct": 100.0, "weights_gib": 34.5}), // 0.5.14 wording
        ),
        (
            "KV Cache is allocated. dtype: torch.bfloat16, #tokens: 1234567, K size: 20.00 GB, V size: 20.00 GB",
            json!({"phase": "allocating_kv", "kv_cache_tokens": 1_234_567}),
        ),
        (
            "Full KV Cache VA upper bound. dtype: torch.bfloat16, #tokens: 99, K size: 1.00 GB, V size: 1.00 GB",
            json!({"phase": "allocating_kv", "kv_cache_tokens": 99}),
        ),
        (
            "KV Cache is allocated. #tokens: 1234567, K size: 20.00 GB, V size: 20.00 GB",
            json!({"phase": "allocating_kv", "kv_cache_tokens": 1_234_567}), // 0.5.14 wording
        ),
        (
            "max_total_num_tokens=1234567, chunked_prefill_size=8192, max_prefill_tokens=16384, \
             max_running_requests=2048, context_len=262144, available_gpu_mem=9.5 GB",
            json!({"phase": "allocating_kv", "kv_cache_tokens": 1_234_567, "max_running_requests": 2048,
                   "context_len": 262_144}),
        ),
        (
            "Capture decode CUDA graph begin. backend=full, num_tokens_per_req=1, bs=[1, 2, 4], avail mem=9.50 GB",
            json!({"phase": "capturing_graphs", "pct": 0.0}),
        ),
        (
            "Capture prefill CUDA graph begin. backend=tc_piecewise, num_tokens=4096, avail mem=9.50 GB",
            json!({"phase": "capturing_graphs", "pct": 0.0}),
        ),
        (
            "Capture cuda graph begin. This can take up to several minutes. avail mem=9.50 GB",
            json!({"phase": "capturing_graphs", "pct": 0.0}), // 0.5.14 wording
        ),
        (
            "Capturing batches (bs=16 avail_mem=9.10 GB):  40%|████      | 4/10 [00:03<00:04]",
            json!({"phase": "capturing_graphs", "batch_size": 16, "pct": 40.0}),
        ),
        (
            "Capturing num tokens (num_tokens=4096 avail_mem=9.00 GB):  50%|█████     | 5/10 [00:03<00:04]",
            json!({"phase": "capturing_graphs", "num_tokens": 4096, "pct": 50.0}),
        ),
        (
            "Capture decode CUDA graph end. elapsed=12.31 s, mem usage=1.20 GB, avail mem=8.30 GB.",
            json!({"phase": "capturing_graphs", "pct": 100.0, "cuda_graph_gib": 1.2}),
        ),
        (
            "Capture cuda graph end. Time elapsed: 12.31 s. mem usage=1.20 GB. avail mem=8.30 GB.",
            json!({"phase": "capturing_graphs", "pct": 100.0, "cuda_graph_gib": 1.2}), // 0.5.14 wording
        ),
        (
            "INFO:     The server is fired up and ready to roll!",
            json!({"phase": "ready", "pct": 100.0}),
        ),
        (
            "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2 GiB",
            json!({"phase": "failed", "error": "oom"}),
        ),
        (
            "RuntimeError: The memory capacity is unbalanced. Some GPUs may be occupied by other processes. \
             pre_model_load_memory=10.0, local_gpu_memory=35.0",
            json!({"phase": "failed", "error": "memory_unbalanced"}),
        ),
        (
            "ValueError: Loaded weights leave no GPU memory for the KV cache under --mem-fraction-static=0.5. Raise \
             --mem-fraction-static above 0.612 (minimum viable = 1 - available/pre = 0.6111).",
            json!({"phase": "failed", "error": "mem_fraction_too_low", "suggested_mem_fraction_static": 0.612}),
        ),
        (
            "AssertionError: FlashAttention v3 Backend requires SM>=80 and SM<=90. Please use `--attention-backend \
             flashinfer`.",
            json!({"phase": "failed", "error": "attention_backend_unsupported"}),
        ),
        (
            "Capture cuda graph failed: boom\nPossible solutions: ...",
            json!({"phase": "failed", "error": "cuda_graph_capture_failed"}),
        ),
    ];
    assert_eq!(cases.len(), 22);
    for (line, expected) in cases {
        let got = a()
            .parse_startup_log(line)
            .map_or(Value::Null, Value::Object);
        assert_eq!(got, expected, "{line:?}");
    }
}

#[test]
fn test_startup_log_noise_and_tqdm_cr_sglang() {
    let a = a();
    assert!(a.parse_startup_log("").is_none());
    assert!(
        a.parse_startup_log("Prefill batch, #new-seq: 1, #new-token: 12")
            .is_none()
    );
    let r = a.parse_startup_log(
        "Loading safetensors checkpoint shards:   0% Completed | 0/4\rLoading safetensors checkpoint shards:  75% \
         Completed | 3/4 [00:1<00:1]",
    );
    assert_eq!(
        r.map(Value::Object),
        Some(json!({"phase": "loading_weights", "pct": 75.0}))
    );
    assert_eq!(
        a.parse_startup_log("\x1b[32mThe server is fired up and ready to roll!\x1b[0m")
            .map(Value::Object),
        Some(json!({"phase": "ready", "pct": 100.0}))
    );
}

// ---- Rust-native property tests ----
proptest! {
    #[test]
    fn prop_sglang_parsers_never_panic(
        text in "(\\PC|[ \\n\\r\\t{}=\",#_:.0-9a-z+%|/-]){0,400}",
        names in proptest::collection::vec(prop::sample::select(vec![
            "sglang:num_running_reqs", "sglang_prompt_tokens", "sglang:time_to_first_token_seconds_bucket",
            "sglang:token_usage", "sglang:kv_used_tokens", "sglang:max_total_num_tokens", "x"]), 0..6),
        vals in proptest::collection::vec("[-+0-9.eNaInf_]{1,8}", 0..6),
    ) {
        let a = SglangAdapter::new();
        let mut t = text;
        for (n, v) in names.iter().zip(&vals) {
            t.push_str(&format!("\n{n}{{le=\"{v}\"}} {v}"));
        }
        let _ = a.parse_metrics(&t);
        for line in t.lines() {
            if let Some(o) = a.parse_startup_log(line) {
                prop_assert!(o.keys().all(|k| [
                    "phase", "pct", "kv_cache_tokens", "weights_gib", "cuda_graph_gib", "error", "load_seconds",
                    "batch_size", "num_tokens", "suggested_mem_fraction_static", "max_running_requests", "context_len",
                ].contains(&k.as_str())));
            }
        }
    }
}
