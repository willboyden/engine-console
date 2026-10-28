//! Port of backend/tests/vllm/test_validate_launch.py.

use ec_hub::adapters::base::{EngineAdapter, Hardware, LaunchContext, MODEL_MOUNT};
use ec_hub::adapters::vllm::VllmAdapter;
use serde_json::{Map, Value, json};

use crate::ws_a1_kit::{approx, dense, get, hw_vllm, obj};

fn v(p: Value) -> Vec<String> {
    VllmAdapter::new().validate(&obj(p))
}

fn preset(name: &str) -> Map<String, Value> {
    VllmAdapter::new()
        .presets()
        .into_iter()
        .find(|(n, _)| n == name)
        .map(|(_, p)| p)
        .unwrap()
}

fn env_of(spec: &ec_hub::adapters::base::LaunchSpec) -> Value {
    Value::Object(spec.env.clone())
}

#[test]
fn test_presets_validate_vllm() {
    let a = VllmAdapter::new();
    let pre = a.presets();
    for name in [
        "balanced",
        "max-throughput",
        "low-latency",
        "long-context",
        "tool-agent",
        "reasoning",
    ] {
        assert!(pre.iter().any(|(n, _)| n == name), "{name}");
    }
    for (name, p) in pre {
        assert!(a.validate(&p).is_empty(), "{name}");
    }
}

#[test]
fn test_validate_errors_vllm() {
    assert_eq!(v(json!({"nope": 1})), ["unknown parameter 'nope'"]);
    for p in [
        json!({"gpu_memory_utilization": 1.5}),
        json!({"gpu_memory_utilization": "0.9"}),
        json!({"max_model_len": true}),
        json!({"max_model_len": 1000.5}),
        json!({"kv_cache_dtype": "fp7"}),
        json!({"enforce_eager": 1}),
        json!({"hf_overrides": "{not json"}),
        json!({"cudagraph_capture_sizes": [1, 0]}),
        json!({"allowed_local_media_path": "/"}),
        json!({"enable_auto_tool_choice": true}),
        json!({"lora_modules": ["a=/x"]}),
        json!({"speculative_config": {"method": "mtp"}, "spec_method": "mtp"}),
        json!({"speculative_config": "{\"model\": \"org/draft\"}", "spec_model": "org/draft"}),
        json!({"revision": "--evil"}),
        json!({"served_model_name": ["--evil"]}),
        json!({"quantization": "gguf"}), // gone from the 0.27.1 registry
        json!({"quantization": "quark"}), // AMD only, left out of the catalog on purpose
    ] {
        assert!(!v(p.clone()).is_empty(), "{p}");
    }
}

/// vLLM 0.27.1 merges --spec-method/--spec-model/--spec-tokens into --speculative-config and refuses only a key that
/// appears in both (engine/arg_utils.py).
#[test]
fn test_spec_shortcuts_merge_unless_a_key_is_given_twice_vllm() {
    assert!(
        v(json!({"speculative_config": {"num_speculative_tokens": 2}, "spec_method": "mtp"}))
            .is_empty()
    );
    assert!(v(json!({"speculative_config": "{\"method\": \"mtp\"}", "spec_tokens": 3})).is_empty());
    let e = v(json!({"speculative_config": {"method": "mtp"}, "spec_method": "mtp"}));
    assert!(e.len() == 1 && e[0].contains("mutually exclusive") && e[0].contains("spec_method"));
    let e2 = v(json!({"speculative_config": {"num_speculative_tokens": 2}, "spec_tokens": 3}));
    assert!(e2.len() == 1 && e2[0].contains("num_speculative_tokens"));
}

#[test]
fn test_retired_keys_say_what_to_do_instead_vllm() {
    let img = v(json!({"image": "vllm/vllm-openai:v0.25.1"}));
    assert!(img.len() == 1 && img[0].contains("pinned in Settings"));
    let gemm = v(json!({"nvfp4_gemm_backend": "marlin"}));
    assert!(
        gemm.len() == 1
            && gemm[0].contains("linear_backend")
            && gemm[0].contains("does not exist in vLLM 0.27.1")
    );
    let moe = v(json!({"flashinfer_moe_backend": "latency"}));
    assert!(moe.len() == 1 && moe[0].contains("moe_backend"));
}

#[test]
fn test_validate_accepts_valid_and_none_vllm() {
    assert!(v(json!({"max_model_len": null})).is_empty());
    assert!(
        v(json!({"hf_overrides": "{\"a\": 1}", "max_lora_rank": 16, "enable_lora": true, "lora_modules": ["x=/y"],
                 "kv_cache_dtype": "int4_per_token_head", "mamba_cache_mode": "align",
                 "moe_backend": "flashinfer_b12x"}))
        .is_empty()
    );
}

#[test]
fn test_build_launch_golden_balanced_vllm() {
    let spec = VllmAdapter::new()
        .build_launch(
            "example-org/chat-model-36B",
            &preset("balanced"),
            &hw_vllm(),
            "chat",
            "/root/.cache/huggingface",
        )
        .unwrap();
    assert_eq!(spec.image, "vllm/vllm-openai:v0.27.1");
    assert_eq!(
        spec.argv,
        [
            "--model",
            "example-org/chat-model-36B",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--served-model-name",
            "chat",
            "--gpu-memory-utilization",
            "0.9",
            "--max-model-len",
            "65536",
            "--enable-prefix-caching",
            "--performance-mode",
            "balanced",
        ]
    );
    assert_eq!(
        env_of(&spec),
        json!({"VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
               "TORCH_CUDA_ARCH_LIST": "12.0", "HF_HOME": "/root/.cache/huggingface"})
    );
    assert!(spec.needs_hf_cache);
    assert_eq!(
        (
            spec.container_port,
            spec.health_path.as_str(),
            spec.metrics_path.as_str()
        ),
        (8000, "/health", "/metrics")
    );
}

#[test]
fn test_build_launch_golden_complex_vllm() {
    let params = obj(json!({
        "quantization": "fp8", "enable_auto_tool_choice": true, "tool_call_parser": "hermes",
        "enable_chunked_prefill": false, "enable_prefix_caching": true, "enforce_eager": false,
        "speculative_config": {"method": "mtp", "num_speculative_tokens": 2},
        "hf_overrides": "{\"text_config\": {\"rope_parameters\": {\"factor\": 4.0}}}",
        "cudagraph_capture_sizes": [1, 2, 4], "served_model_name": ["alias", "chat"],
        "attention_backend": "FLASHINFER", "api_key": "sekret", "allow_long_max_model_len": true,
        "kv_cache_dtype": "fp8", "moe_backend": "marlin", "mamba_cache_mode": "align",
    }));
    let spec = VllmAdapter::new()
        .build_launch("org/m", &params, &hw_vllm(), "chat", "/hf")
        .unwrap();
    assert_eq!(spec.image, "vllm/vllm-openai:v0.27.1");
    assert_eq!(
        spec.argv,
        [
            "--model",
            "org/m",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--served-model-name",
            "chat",
            "alias",
            "--kv-cache-dtype",
            "fp8",
            "--mamba-cache-mode",
            "align",
            "--hf-overrides",
            "{\"text_config\":{\"rope_parameters\":{\"factor\":4.0}}}",
            "--quantization",
            "fp8",
            "--moe-backend",
            "marlin",
            "--enable-prefix-caching",
            "--no-enable-chunked-prefill",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "hermes",
            "--no-enforce-eager",
            "--cudagraph-capture-sizes",
            "1",
            "2",
            "4",
            "--attention-backend",
            "FLASHINFER",
            "--speculative-config",
            "{\"method\":\"mtp\",\"num_speculative_tokens\":2}",
        ]
    );
    assert_eq!(spec.env["VLLM_API_KEY"], "sekret");
    assert!(!spec.argv.join(" ").contains("sekret")); // a secret never rides on the command line
    assert_eq!(spec.env["VLLM_ALLOW_LONG_MAX_MODEL_LEN"], "1");
    assert!(!spec.env.keys().any(|k| k.contains("ATTENTION_BACKEND"))); // VLLM_ATTENTION_BACKEND no longer exists
    assert!(!spec.env.contains_key("HF_TOKEN"));
}

#[test]
fn test_non_negatable_false_bool_is_omitted_vllm() {
    let spec = VllmAdapter::new()
        .build_launch(
            "m",
            &obj(json!({"enable_request_id_headers": false, "disable_custom_all_reduce": false})),
            &hw_vllm(),
            "s",
            "/hf",
        )
        .unwrap();
    assert!(
        !spec
            .argv
            .contains(&"--enable-request-id-headers".to_string())
    );
    assert!(
        spec.argv
            .contains(&"--no-disable-custom-all-reduce".to_string())
    );
}

#[test]
fn test_nargs_flags_are_emitted_once_with_all_values_vllm() {
    let spec = VllmAdapter::new()
        .build_launch(
            "m",
            &obj(json!({"allowed_media_domains": ["a.example", "b.example"],
                        "kv_cache_dtype_skip_layers": ["0", "sliding_window"]})),
            &hw_vllm(),
            "s",
            "/hf",
        )
        .unwrap();
    let a = &spec.argv;
    let i = a
        .iter()
        .position(|x| x == "--allowed-media-domains")
        .unwrap();
    assert_eq!(a[i + 1..i + 3], ["a.example", "b.example"]);
    assert_eq!(
        a.iter().filter(|x| *x == "--allowed-media-domains").count(),
        1
    );
    assert_eq!(
        a.iter()
            .filter(|x| *x == "--kv-cache-dtype-skip-layers")
            .count(),
        1
    );
}

#[test]
fn test_arch_list_follows_the_hardware_vllm() {
    let a = VllmAdapter::new();
    let hopper = Hardware {
        compute_capability: "9.0".into(),
        ..hw_vllm()
    };
    assert_eq!(
        a.build_launch("m", &Map::new(), &hopper, "s", "/h")
            .unwrap()
            .env["TORCH_CUDA_ARCH_LIST"],
        "9.0"
    );
    let odd = Hardware {
        compute_capability: "unknown".into(),
        ..hw_vllm()
    };
    assert!(
        !a.build_launch("m", &Map::new(), &odd, "s", "/h")
            .unwrap()
            .env
            .contains_key("TORCH_CUDA_ARCH_LIST")
    );
}

#[test]
fn test_build_launch_rejects_invalid_vllm() {
    let a = VllmAdapter::new();
    let e = a
        .build_launch("m", &obj(json!({"bogus": 1})), &hw_vllm(), "s", "/hf")
        .unwrap_err();
    assert!(
        e.detail.contains("unknown parameter") && e.status == 422 && e.code == "invalid_params"
    );
    let e = a
        .build_launch("--evil", &Map::new(), &hw_vllm(), "s", "/hf")
        .unwrap_err();
    assert!(e.detail.contains("invalid model identifier"));
}

// ---- contract v2: launch from a context ----
fn ctx(model: &str, fmt: &str, path: Option<&str>, params: Map<String, Value>) -> LaunchContext {
    LaunchContext {
        model: model.into(),
        model_format: fmt.into(),
        model_path: path.map(str::to_string),
        served_name: "chat".into(),
        hf_cache_container_path: "/hf".into(),
        params,
        hw: hw_vllm(),
    }
}

#[test]
fn test_context_launch_for_hf_matches_build_launch_vllm() {
    let a = VllmAdapter::new();
    let params = preset("tool-agent");
    let via_ctx = a
        .build_launch_ctx(&ctx("org/m", "hf", None, params.clone()))
        .unwrap();
    let direct = a
        .build_launch("org/m", &params, &hw_vllm(), "chat", "/hf")
        .unwrap();
    assert!(via_ctx == direct && via_ctx.needs_hf_cache);
}

#[test]
fn test_plain_model_directory_is_launched_by_path_vllm() {
    let spec = VllmAdapter::new()
        .build_launch_ctx(&ctx(
            "my-model-dir",
            "plain",
            Some(MODEL_MOUNT),
            obj(json!({"max_model_len": 8192, "revision": "main", "served_model_name": ["alias"]})),
        ))
        .unwrap();
    assert_eq!(spec.argv[..2], ["--model", "/models"]);
    let i = spec
        .argv
        .iter()
        .position(|x| x == "--served-model-name")
        .unwrap();
    assert_eq!(spec.argv[i + 1..i + 3], ["chat", "alias"]);
    assert!(!spec.argv.contains(&"--revision".to_string())); // a directory has no hub revision
    assert!(spec.argv.contains(&"--max-model-len".to_string()));
    assert!(!spec.needs_hf_cache);
    assert!(!spec.env.contains_key("HF_HOME") && spec.env["HF_HUB_OFFLINE"] == "1");
    assert!(!spec.argv.join(" ").contains("my-model-dir")); // the display name is never used as a path
}

#[test]
fn test_plain_model_defaults_to_the_mount_when_no_path_is_given_vllm() {
    let spec = VllmAdapter::new()
        .build_launch_ctx(&ctx("org/m", "plain", None, Map::new()))
        .unwrap();
    assert_eq!(spec.argv[..2], ["--model", MODEL_MOUNT]);
}

/// `test_unsupported_formats_are_refused[gguf]` and `[ollama]`.
#[test]
fn test_unsupported_formats_are_refused_vllm() {
    for fmt in ["gguf", "ollama"] {
        let e = VllmAdapter::new()
            .build_launch_ctx(&ctx("org/m", fmt, Some("/models/x.gguf"), Map::new()))
            .unwrap_err();
        assert!(e.detail.contains("cannot load"), "{fmt}: {e}");
    }
}

#[test]
fn test_memory_model_vllm() {
    let a = VllmAdapter::new();
    let d = dense();
    let mm = a.memory_model(
        &d,
        &obj(json!({"gpu_memory_utilization": 0.9, "kv_cache_dtype": "fp8", "tensor_parallel_size": 2,
                    "max_num_seqs": 16})),
    );
    assert!(
        get(&mm, "mem_fraction") == Some(0.9)
            && get(&mm, "kv_bytes_per_elem") == Some(1.0)
            && get(&mm, "tp") == Some(2.0)
    );
    assert!(get(&mm, "max_len") == Some(131_072.0) && get(&mm, "max_seqs") == Some(16.0));
    // 2.5 GiB held outside the share (measured) + 0.5 GiB per extra rank (guess); nothing of it is charged to the share
    assert!(
        get(&mm, "outside_share_gib") == Some(3.0)
            && get(&mm, "overhead_gib") == Some(0.0)
            && get(&mm, "claims_share") == Some(1.0)
    );
    assert_eq!(
        get(
            &a.memory_model(&d, &obj(json!({"enforce_eager": true}))),
            "cuda_graph_gib"
        ),
        Some(0.0)
    );
    let single = a.memory_model(
        &d,
        &obj(json!({"gpu_memory_utilization": 0.15, "enforce_eager": true})),
    );
    // live measurement: a 0.15 share of a 95.6 GiB card plus what is held outside it matches the 16.86 GiB the
    // driver showed
    let total = 0.15 * 95.6 + get(&single, "outside_share_gib").unwrap();
    assert!(
        approx((total * 10.0).round() / 10.0, 16.8) && get(&single, "cuda_graph_gib") == Some(0.0)
    );
    // graphs are inside the share (0.27.1)
    assert_eq!(
        get(&a.memory_model(&d, &Map::new()), "cuda_graph_gib"),
        Some(0.75)
    );
    assert_eq!(
        get(&a.memory_model(&d, &Map::new()), "kv_bytes_per_elem"),
        Some(2.0)
    );
    assert_eq!(
        get(
            &a.memory_model(&d, &obj(json!({"kv_cache_dtype": "int4_per_token_head"}))),
            "kv_bytes_per_elem"
        ),
        Some(0.5)
    );
    let fixed = a.memory_model(
        &d,
        &obj(json!({"kv_cache_memory_bytes": 7_000_000_000_i64})),
    );
    assert!(approx(
        (get(&fixed, "kv_cache_fixed_gib").unwrap() * 100.0).round() / 100.0,
        6.52
    ));
}
