//! Port of backend/tests/trtllm/test_trtllm_launch.py: validation and launch building: argv, the generated
//! engine-options YAML, env, and LaunchSpec fields. Python parses the generated file back with yaml.safe_load; here the
//! tree is the adapter's own `yaml_tree`, and every launch asserts the file text IS the header plus PyYAML's dump of it
//! (the emitter is pinned to PyYAML by tests/test_ws_a2_goldens.rs).

use std::collections::BTreeSet;

use crate::a2::{p, read_json, trt};
use ec_hub::adapters::base::{EngineAdapter, Hardware, LaunchContext, LaunchSpec, Obj};
use ec_hub::adapters::trtllm::{STATS_HISTORY, TrtllmAdapter};
use ec_hub::adapters::trtllm_params::{EXTRA_OPTIONS_PATH, TORCH_LLM_ARGS_FIELDS, YAML_NESTED};
use ec_hub::adapters::trtllm_parse::py::PyExc;
use regex::Regex;
use serde_json::{Value, json};

const HF: &str = "/root/.cache/huggingface";

fn adapter() -> TrtllmAdapter {
    TrtllmAdapter::new()
}

fn ctx_with(
    params: Obj,
    hw: &Hardware,
    model: &str,
    format: &str,
    path: Option<&str>,
) -> LaunchContext {
    LaunchContext {
        model: model.into(),
        model_format: format.into(),
        model_path: path.map(str::to_string),
        served_name: "chat".into(),
        hf_cache_container_path: HF.into(),
        params,
        hw: hw.clone(),
    }
}

fn ctx(params: Value, hw: &Hardware) -> LaunchContext {
    ctx_with(p(params), hw, "example-org/chat-model-36B", "hf", None)
}

fn build(c: &LaunchContext) -> LaunchSpec {
    adapter()
        .py_build_launch_ctx(c)
        .unwrap_or_else(|e| panic!("{e}"))
}

fn build_err(c: &LaunchContext) -> PyExc {
    match adapter().py_build_launch_ctx(c) {
        Ok(spec) => panic!("expected a ValueError, got {:?}", spec.argv),
        Err(e) => e,
    }
}

/// `pytest.raises(ValueError, match=...)`.
fn raises(c: &LaunchContext, pattern: &str) {
    let e = build_err(c);
    assert_eq!(e.kind, "ValueError", "{e}");
    assert!(
        Regex::new(pattern).expect("re").is_match(&e.msg),
        "{} !~ {pattern}",
        e.msg
    );
}

/// `yaml.safe_load(spec.files[EXTRA_OPTIONS_PATH])`: the tree, after checking the file is exactly its rendering.
fn yaml_of(spec: &LaunchSpec, params: &Obj) -> Value {
    let tree = adapter().yaml_tree(params).expect("tree");
    let text = spec
        .files
        .get(EXTRA_OPTIONS_PATH)
        .and_then(Value::as_str)
        .expect("options file");
    assert_eq!(text, TrtllmAdapter::render_yaml(&tree));
    let mut o = serde_json::Map::new();
    for (k, v) in tree {
        o.insert(k, v.to_json());
    }
    Value::Object(o)
}

fn schema() -> Value {
    read_json("trtllm", "llm_args_1.2.1.json")
}

fn literal(type_str: &str) -> Option<Vec<String>> {
    let m = Regex::new(r"Literal\[(.*?)\]")
        .expect("re")
        .captures(type_str)?;
    Some(
        Regex::new(r"'([^']+)'")
            .expect("re")
            .captures_iter(&m[1])
            .map(|c| c[1].to_string())
            .collect(),
    )
}

fn is_int(v: &Value) -> bool {
    matches!(v, Value::Number(n) if !n.is_f64())
}

/// `_type_problem`: a light check of one YAML value against the image's annotation string.
fn type_problem(value: &Value, type_str: &str) -> Option<String> {
    if value.is_null() {
        return (!(type_str.contains("Optional") || type_str.contains("NoneType")))
            .then(|| "null for a non-optional field".into());
    }
    if let Some(allowed) = literal(type_str) {
        return (!value
            .as_str()
            .is_some_and(|s| allowed.iter().any(|a| a == s)))
        .then(|| format!("{value} not in {allowed:?}"));
    }
    if type_str.contains("List[int]") {
        let ok = value.as_array().is_some_and(|a| a.iter().all(is_int));
        return (!ok).then(|| format!("{value} is not a list of ints"));
    }
    if type_str.contains("List[str]") {
        let ok = value
            .as_array()
            .is_some_and(|a| a.iter().all(Value::is_string));
        return (!ok).then(|| "not a list of strings".into());
    }
    if type_str.contains("bool") && !type_str.contains("int") {
        return (!value.is_boolean()).then(|| format!("{value} is not a bool"));
    }
    if type_str.contains("float") {
        return (!value.is_number()).then(|| format!("{value} is not a number"));
    }
    if type_str.contains("int") {
        return (!is_int(value)).then(|| format!("{value} is not an int"));
    }
    if type_str.contains("str") && !type_str.contains("Union") {
        return (!value.is_string()).then(|| format!("{value} is not a string"));
    }
    None
}

/// `schema_problems`: every key of a generated options file must exist in the image's TorchLlmArgs (nested blocks
/// too), with a type that fits.
fn schema_problems(tree: &Value) -> Vec<String> {
    let s = schema();
    let mut out = Vec::new();
    let top = s["classes"]["TorchLlmArgs"].as_object().expect("top");
    for (name, value) in tree.as_object().expect("tree") {
        let Some(entry) = top.get(name) else {
            out.push(format!("unknown top-level key {name}"));
            continue;
        };
        let cls = YAML_NESTED.iter().find(|(f, _)| f == name).map(|(_, c)| *c);
        if let (Some(cls), Some(sub)) = (cls, value.as_object()) {
            let fields = s["classes"][cls].as_object().expect("class");
            for (k, v) in sub {
                match fields.get(k) {
                    None => out.push(format!("unknown key {name}.{k}")),
                    Some(f) => {
                        if let Some(prob) = type_problem(v, f["type"].as_str().unwrap_or_default())
                        {
                            out.push(format!("{name}.{k}: {prob}"));
                        }
                    }
                }
            }
        } else if name == "speculative_config" {
            let class = match value["decoding_type"].as_str().unwrap_or_default() {
                "MTP" => "MTPDecodingConfig",
                "Eagle" => "EagleDecodingConfig",
                "NGram" => "NGramDecodingConfig",
                "DraftTarget" => "DraftTargetDecodingConfig",
                _ => "DecodingBaseConfig",
            };
            let fields = s["classes"][class].as_object().expect("class");
            for k in value.as_object().expect("spec").keys() {
                if k != "decoding_type" && !fields.contains_key(k) {
                    out.push(format!("unknown key speculative_config.{k}"));
                }
            }
        } else if cls.is_none()
            && let Some(prob) = type_problem(value, entry["type"].as_str().unwrap_or_default())
        {
            out.push(format!("{name}: {prob}"));
        }
    }
    out
}

fn argv_after(spec: &LaunchSpec, flag: &str) -> String {
    let i = spec
        .argv
        .iter()
        .position(|a| a == flag)
        .unwrap_or_else(|| panic!("{flag} not in argv"));
    spec.argv[i + 1].clone()
}

// --------------------------------------------------------------- validation
#[test]
fn test_presets_validate_trtllm() {
    let pre = adapter().presets();
    let names: BTreeSet<&str> = pre.iter().map(|(n, _)| n.as_str()).collect();
    for n in [
        "balanced",
        "max-throughput",
        "low-latency",
        "long-context",
        "tool-agent",
        "reasoning",
    ] {
        assert!(names.contains(n), "{n}");
    }
    for (name, params) in &pre {
        assert!(adapter().validate(params).is_empty(), "{name}");
    }
}

#[test]
fn test_every_sample_and_scenario_validates_trtllm() {
    for key in trt::samples().keys() {
        assert!(
            adapter().validate(&trt::sample_params(key)).is_empty(),
            "{key}"
        );
    }
    for (name, params) in trt::scenarios() {
        assert!(adapter().validate(&params).is_empty(), "{name}");
    }
}

#[test]
fn test_validate_value_errors_trtllm() {
    assert_eq!(
        adapter().validate(&p(json!({"nope": 1}))),
        vec!["unknown parameter 'nope'"]
    );
    let long = "x".repeat(4001);
    for bad in [
        json!({"free_gpu_memory_fraction": 1.0}),
        json!({"free_gpu_memory_fraction": "0.9"}),
        json!({"max_seq_len": true}),
        json!({"max_seq_len": 1000.5}),
        json!({"kv_cache_dtype": "fp4"}),
        json!({"enable_block_reuse": 1}),
        json!({"tokens_per_block": 48}),
        json!({"tokens_per_block": 1}),
        json!({"cuda_graph_batch_sizes": ["1", "0"]}),
        json!({"cuda_graph_batch_sizes": []}),
        json!({"kv_max_attention_window": ["a"]}),
        json!({"nvfp4_gemm_backends": ["magic"]}),
        json!({"image": "nvcr.io/nvidia/tensorrt-llm/release:latest"}),
        json!({"image": "nvcr.io/nvidia/tensorrt-llm/release"}),
        json!({"revision": "-bad"}),
        json!({"tokenizer": "a b"}),
        json!({"chat_template": "-x"}),
        json!({"chat_template": long}),
        json!({"chat_template": "plainword"}),
        json!({"chat_template": "relative/path.jinja"}),
        json!({"media_io_kwargs": "{not json"}),
        json!({"media_io_kwargs": "[1]"}),
        json!({"speculative_config": {"decoding_type": "Medusa"}}),
        json!({"speculative_config": {}}),
        json!({"orchestrator_type": "ray"}),
        json!({"custom_tokenizer": "a.b.C"}),
        json!({"tool_parser": "hermes"}),
        json!({"reasoning_parser": "gptoss"}),
        json!({"attention_backend": "NOPE"}),
        json!({"tensor_parallel_size": 0}),
        json!({"batch_wait_max_tokens_ratio": 1.5}),
    ] {
        assert!(!adapter().validate(&p(bad.clone())).is_empty(), "{bad}");
    }
}

#[test]
fn test_json_parameters_respect_the_nodes_size_limits_trtllm() {
    // ecnode.proto: one argv entry is at most 4096 characters, one generated file at most 128 KiB.
    let v = |x: Value| adapter().validate(&p(x));
    assert!(
        v(json!({"media_io_kwargs": {"video": {"note": "x".repeat(3600)}}}))[0]
            .contains("at most 3500")
    );
    assert!(v(json!({"media_io_kwargs": {"video": {"note": "x".repeat(3000)}}})).is_empty());
    assert!(
        v(json!({"extra_llm_options": {"enable_min_latency": "x".repeat(60000)}}))[0]
            .contains("at most 50000")
    );
    assert!(
        v(json!({"speculative_config": {"decoding_type": "NGram", "note": "x".repeat(60000)}}))[0]
            .contains("at most 50000")
    );
}

#[test]
fn test_chat_template_is_a_path_or_a_template_trtllm() {
    // load_chat_template opens the value as a file; only { } or a newline mark it as the template text.
    let v = |x: &str| adapter().validate(&p(json!({"chat_template": x})));
    assert!(v("/models/chat_template.jinja").is_empty());
    assert!(v("{{ messages[0].content }}").is_empty());
    assert!(v("line one\nline two").is_empty());
    assert!(v("template.jinja")[0].contains("file name"));
}

#[test]
fn test_validate_accepts_none_and_json_strings_trtllm() {
    let v = |x: Value| adapter().validate(&p(x));
    assert!(v(json!({"max_seq_len": null, "tool_parser": null})).is_empty());
    assert!(v(json!({"media_io_kwargs": "{\"video\": {\"num_frames\": 4}}"})).is_empty());
    assert!(v(json!({"speculative_config": "{\"decoding_type\": \"AUTO\"}"})).is_empty());
}

#[test]
fn test_validate_relations_trtllm() {
    let v = |x: Value| adapter().validate(&p(x));
    assert!(
        v(json!({"enable_cuda_graphs": false, "cuda_graph_padding": true}))
            .iter()
            .any(|e| e.contains("enable_cuda_graphs"))
    );
    assert!(
        !v(json!({"cuda_graph_batch_sizes": ["1", "2"], "cuda_graph_max_batch_size": 8}))
            .is_empty()
    );
    assert!(
        v(json!({"cuda_graph_batch_sizes": ["1", "2"], "cuda_graph_max_batch_size": 0})).is_empty()
    );
    assert!(!v(json!({"torch_compile_inductor": true})).is_empty());
    assert!(v(json!({"torch_compile_inductor": true, "enable_torch_compile": true})).is_empty());
    assert!(!v(json!({"spec_max_draft_len": 3})).is_empty()); // needs spec_decoding_type
    assert!(!v(json!({"spec_decoding_type": "Eagle", "spec_max_draft_len": 3})).is_empty()); // needs a draft model
    assert!(!v(json!({"spec_decoding_type": "NGram", "spec_max_draft_len": 3})).is_empty()); // needs the match length
    assert!(!v(json!({"spec_decoding_type": "MTP", "spec_max_draft_len": 3})).is_empty()); // engine overwrites it
    assert!(!v(json!({"spec_decoding_type": "MTP", "spec_ngram_max_matching_size": 3})).is_empty());
    assert!(!v(json!({"spec_decoding_type": "AUTO", "spec_max_concurrency": 4})).is_empty());
    assert!(v(json!({"spec_decoding_type": "AUTO"})).is_empty());
    assert!(
        !v(json!({"speculative_config": {"decoding_type": "AUTO"}, "spec_decoding_type": "AUTO"}))
            .is_empty()
    );
}

#[test]
fn test_validate_moe_parallel_sizes_trtllm() {
    let v = |x: Value| adapter().validate(&p(x));
    let ok = json!({"tensor_parallel_size": 4, "moe_expert_parallel_size": 2, "moe_tensor_parallel_size": 2});
    assert!(v(ok.clone()).is_empty());
    let mut bad = ok;
    bad["moe_tensor_parallel_size"] = json!(4); // 2 x 4 != 4
    assert!(!v(bad).is_empty());
    assert!(!v(json!({"tensor_parallel_size": 4, "moe_expert_parallel_size": 3})).is_empty()); // 3 does not divide 4
    assert!(v(json!({"tensor_parallel_size": 4, "moe_expert_parallel_size": 4})).is_empty());
    assert!(!v(json!({"tensor_parallel_size": 4, "moe_cluster_parallel_size": 2, "moe_expert_parallel_size": 2})).is_empty());
    assert!(v(json!({"tensor_parallel_size": 4, "moe_cluster_parallel_size": 2})).is_empty());
    assert!(!v(json!({"moe_expert_parallel_size": 2})).is_empty()); // tp defaults to 1
}

#[test]
fn test_validate_extra_options_trtllm() {
    let v = |x: Value| adapter().validate(&p(json!({"extra_llm_options": x})));
    assert!(v(json!({"garbage_collection_gen0_threshold": 5})).is_empty());
    for (bad, why) in [
        (json!({"model": "x"}), "not allowed"),
        (json!({"env_overrides": {"A": "1"}}), "not allowed"),
        (
            json!({"kv_cache_free_gpu_memory_fraction": 0.8}),
            "refuses unknown",
        ),
        (
            json!({"kv_cache_config": {"dtype": "fp8"}}),
            "own parameter",
        ),
        (json!({"max_batch_size": 4}), "own parameter"),
        (json!({"enable_iter_perf_stats": true}), "own parameter"),
        (json!({"max_stats_len": 50}), "own parameter"),
    ] {
        let errs = v(bad.clone());
        assert!(!errs.is_empty() && errs[0].contains(why), "{bad} {errs:?}");
    }
}

// --------------------------------------------------------------- golden launches
#[test]
fn test_golden_empty_launch_trtllm() {
    let spec = build(&ctx(json!({}), &trt::hw()));
    assert_eq!(spec.image, "nvcr.io/nvidia/tensorrt-llm/release:1.2.1");
    assert_eq!(
        spec.argv,
        [
            "trtllm-serve",
            "serve",
            "example-org/chat-model-36B",
            "--backend",
            "pytorch",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--extra_llm_api_options",
            "/config/extra-llm-api-options.yaml"
        ]
    );
    assert_eq!(
        Value::Object(spec.files.clone()),
        json!({EXTRA_OPTIONS_PATH: "# Generated by Engine Console for `trtllm-serve --extra_llm_api_options`.\n# Edit the launch parameters, not this file.\nenable_iter_perf_stats: true\nmax_stats_len: 100\n"})
    );
    assert_eq!(
        serde_json::to_string(&spec.env).expect("env"),
        serde_json::to_string(
            &json!({"HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1",
                                      "PYTHONUNBUFFERED": "1", "HF_HOME": HF})
        )
        .expect("env")
    );
    assert_eq!(
        (
            spec.container_port,
            spec.health_path.as_str(),
            spec.metrics_path.as_str(),
            spec.models_path.as_str()
        ),
        (8000, "/health", "/metrics", "/v1/models")
    );
    assert!(spec.ipc_host && spec.shm_size == "16g" && spec.needs_hf_cache);
}

#[test]
fn test_no_file_and_no_flag_when_nothing_needs_it_trtllm() {
    let spec = build(&ctx(
        json!({"enable_iter_perf_stats": false, "max_batch_size": 8}),
        &trt::hw(),
    ));
    assert!(spec.files.is_empty() && !spec.argv.contains(&"--extra_llm_api_options".to_string()));
    assert_eq!(spec.argv[spec.argv.len() - 2..], ["--max_batch_size", "8"]);
}

#[test]
fn test_golden_tensor_parallel_fp8_launch_trtllm() {
    let params = json!({"tensor_parallel_size": 2, "free_gpu_memory_fraction": 0.85, "max_batch_size": 64,
                        "max_num_tokens": 8192, "enable_chunked_prefill": true, "kv_cache_dtype": "fp8",
                        "enable_block_reuse": true, "tool_parser": "qwen3", "reasoning_parser": "qwen3",
                        "cuda_graph_padding": true, "nccl_p2p_disable": true, "trust_remote_code": false});
    let spec = build(&ctx(params.clone(), &trt::hw()));
    assert_eq!(
        spec.argv,
        [
            "trtllm-serve",
            "serve",
            "example-org/chat-model-36B",
            "--backend",
            "pytorch",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--free_gpu_memory_fraction",
            "0.85",
            "--tensor_parallel_size",
            "2",
            "--max_batch_size",
            "64",
            "--max_num_tokens",
            "8192",
            "--enable_chunked_prefill",
            "--tool_parser",
            "qwen3",
            "--reasoning_parser",
            "qwen3",
            "--extra_llm_api_options",
            "/config/extra-llm-api-options.yaml"
        ]
    );
    assert!(!spec.argv.contains(&"--trust_remote_code".to_string())); // a false flag is simply absent
    assert_eq!(
        yaml_of(&spec, &p(params)),
        json!({"enable_iter_perf_stats": true, "max_stats_len": 100, "cuda_graph_config": {"enable_padding": true},
               "kv_cache_config": {"dtype": "fp8", "enable_block_reuse": true}})
    );
    assert_eq!(spec.env["NCCL_P2P_DISABLE"], "1");
    let text = spec.files[EXTRA_OPTIONS_PATH].as_str().expect("file");
    assert!(text.ends_with(
        "cuda_graph_config:\n  enable_padding: true\nenable_iter_perf_stats: true\nkv_cache_config:\n  dtype: fp8\n  enable_block_reuse: true\nmax_stats_len: 100\n"
    )); // keys sorted: stable diffs
}

#[test]
fn test_plain_model_folder_launch_trtllm() {
    let hw1 = trt::hw1();
    let spec = build(&ctx_with(Obj::new(), &hw1, "", "plain", Some("/models")));
    assert_eq!(spec.argv[..3], ["trtllm-serve", "serve", "/models"]);
    assert!(
        !spec.needs_hf_cache
            && !spec.env.contains_key("HF_HOME")
            && spec.env["HF_HUB_OFFLINE"] == "1"
    );
    raises(
        &ctx_with(Obj::new(), &hw1, "x", "plain", None),
        "model reference",
    );
}

#[test]
fn test_unsupported_formats_and_hostile_model_refs_trtllm() {
    let hw = trt::hw();
    for fmt in ["gguf", "ollama"] {
        raises(
            &ctx_with(Obj::new(), &hw, "example-org/chat-model-36B", fmt, None),
            "not",
        );
    }
    for bad in ["--evil", "-x", "a b", "a\nb", ""] {
        raises(
            &ctx_with(Obj::new(), &hw, bad, "hf", None),
            "model reference",
        );
    }
}

#[test]
fn test_build_rejects_invalid_params_trtllm() {
    raises(&ctx(json!({"bogus": 1}), &trt::hw()), "unknown parameter");
}

#[test]
fn test_build_launch_bridge_uses_the_hf_path_trtllm() {
    let spec = adapter()
        .build_launch(
            "example-org/m",
            &p(json!({"max_seq_len": 4096})),
            &trt::hw(),
            "s",
            "/hf",
        )
        .expect("spec");
    assert!(
        spec.argv[..3] == ["trtllm-serve", "serve", "example-org/m"]
            && spec.argv.contains(&"--max_seq_len".to_string())
    );
    assert_eq!(spec.env["HF_HOME"], "/hf");
}

#[test]
fn test_image_override_and_types_trtllm() {
    let spec = build(&ctx(
        json!({"image": "nvcr.io/nvidia/tensorrt-llm/release:1.3.0"}),
        &trt::hw(),
    ));
    assert!(
        spec.image == "nvcr.io/nvidia/tensorrt-llm/release:1.3.0"
            && !spec.argv.join(" ").contains("--image")
    );
}

#[test]
fn test_values_are_typed_in_cli_and_yaml_trtllm() {
    let params = json!({"free_gpu_memory_fraction": 0.5, "kv_cache_max_gib": 12.5, "kv_host_cache_gib": 4,
                        "tokens_per_block": 64, "stream_interval": 4, "batch_wait_timeout_ms": 5,
                        "batch_wait_max_tokens_ratio": 0.25, "enable_partial_reuse": false, "enable_autotuner": false,
                        "cuda_graph_batch_sizes": ["8", "4", "1"], "kv_max_attention_window": [4096, "512"],
                        "nvfp4_gemm_backends": ["cutlass"], "media_io_kwargs": {"video": {"num_frames": 8}},
                        "otlp_traces_endpoint": "http://otel-collector:4317"});
    let spec = build(&ctx(params.clone(), &trt::hw()));
    assert_eq!(
        argv_after(&spec, "--media_io_kwargs"),
        r#"{"video":{"num_frames":8}}"#
    ); // compact JSON, one entry
    assert_eq!(argv_after(&spec, "--free_gpu_memory_fraction"), "0.5");
    let tree = yaml_of(&spec, &p(params));
    let kv = &tree["kv_cache_config"];
    assert_eq!(
        *kv,
        json!({"max_gpu_total_bytes": 13_421_772_800_i64, "host_cache_size": 4_294_967_296_i64,
                           "tokens_per_block": 64, "enable_partial_reuse": false, "max_attention_window": [4096, 512]})
    );
    assert_eq!(tree["cuda_graph_config"], json!({"batch_sizes": [8, 4, 1]})); // ints, order kept
    assert!(tree["stream_interval"] == json!(4) && !tree["stream_interval"].is_f64());
    assert!(tree["batch_wait_timeout_ms"].is_f64() && tree["batch_wait_timeout_ms"] == json!(5.0));
    assert!(
        tree["enable_autotuner"] == json!(false)
            && tree["nvfp4_gemm_config"] == json!({"allowed_backends": ["cutlass"]})
    );
    assert!(
        schema_problems(&tree).is_empty(),
        "{:?}",
        schema_problems(&tree)
    );
}

#[test]
fn test_cuda_graphs_off_is_an_explicit_null_trtllm() {
    let hw = trt::hw();
    let spec = build(&ctx(json!({"enable_cuda_graphs": false}), &hw));
    let text = spec.files[EXTRA_OPTIONS_PATH].as_str().expect("file");
    assert!(text.contains("cuda_graph_config: null\n"));
    assert!(
        yaml_of(&spec, &p(json!({"enable_cuda_graphs": false})))["cuda_graph_config"].is_null()
    );
    let on = build(&ctx(json!({"enable_cuda_graphs": true}), &hw));
    assert!(
        !on.files[EXTRA_OPTIONS_PATH]
            .as_str()
            .expect("file")
            .contains("cuda_graph_config")
    ); // engine default is on
}

#[test]
fn test_torch_compile_block_trtllm() {
    let hw = trt::hw();
    let tree = |params: Value| yaml_of(&build(&ctx(params.clone(), &hw)), &p(params));
    assert_eq!(
        tree(json!({"enable_torch_compile": true}))["torch_compile_config"],
        json!({})
    );
    assert_eq!(
        tree(json!({"enable_torch_compile": true, "torch_compile_inductor": true}))["torch_compile_config"],
        json!({"enable_inductor": true})
    );
    assert!(
        tree(json!({"enable_torch_compile": false}))
            .get("torch_compile_config")
            .is_none()
    );
}

#[test]
fn test_scheduler_block_restates_the_stock_defaults_trtllm() {
    let hw = trt::hw();
    let tree = |params: Value| yaml_of(&build(&ctx(params.clone(), &hw)), &p(params));
    assert_eq!(
        tree(json!({"context_chunking_policy": "EQUAL_PROGRESS"}))["scheduler_config"],
        json!({"capacity_scheduler_policy": "GUARANTEED_NO_EVICT", "context_chunking_policy": "EQUAL_PROGRESS",
               "dynamic_batch_config": {"enable_batch_size_tuning": true, "enable_max_num_tokens_tuning": false,
                                        "dynamic_batch_moving_average_window": 128}})
    );
    let overridden = tree(json!({"capacity_scheduler_policy": "MAX_UTILIZATION"}));
    assert_eq!(
        overridden["scheduler_config"]["capacity_scheduler_policy"],
        "MAX_UTILIZATION"
    );
    assert!(tree(json!({})).get("scheduler_config").is_none());
}

#[test]
fn test_scenarios_build_and_match_the_schema_trtllm() {
    let hw = trt::hw();
    for (name, params) in trt::scenarios() {
        let spec = build(&ctx_with(
            params.clone(),
            &hw,
            "example-org/chat-model-36B",
            "hf",
            None,
        ));
        assert_eq!(spec.argv[..2], ["trtllm-serve", "serve"], "{name}");
        if !spec.files.is_empty() {
            let problems = schema_problems(&yaml_of(&spec, &params));
            assert!(problems.is_empty(), "{name}: {problems:?}");
        }
    }
}

#[test]
fn test_each_sample_builds_and_matches_the_schema_trtllm() {
    let hw = trt::hw();
    let samples = trt::samples();
    let catalog = adapter().param_catalog();
    for key in samples.keys() {
        let params = trt::sample_params(key);
        let spec = build(&ctx_with(
            params.clone(),
            &hw,
            "example-org/chat-model-36B",
            "hf",
            None,
        ));
        let tree = if spec.files.is_empty() {
            json!({})
        } else {
            yaml_of(&spec, &params)
        };
        assert!(
            schema_problems(&tree).is_empty(),
            "{key}: {:?}",
            schema_problems(&tree)
        );
        assert!(
            tree.as_object()
                .expect("tree")
                .keys()
                .all(|k| TORCH_LLM_ARGS_FIELDS.contains(&k.as_str()))
        );
        let flag = catalog
            .iter()
            .find(|s| &s.key == key)
            .expect("param")
            .flag
            .clone();
        if flag.starts_with("--") {
            assert!(spec.argv.contains(&flag), "{key}");
        } else if let Some(name) = flag.strip_prefix("env:") {
            assert!(spec.env.contains_key(name), "{key}");
        } else if flag == "yaml:*" {
            let extra = samples[key].as_object().expect("extra");
            assert!(extra.keys().all(|k| tree.get(k).is_some()), "{key}");
        } else if flag == "@image" {
            assert_eq!(spec.image, samples[key].as_str().unwrap_or_default());
        } else if key == "enable_iter_perf_stats" {
            assert!(spec.files.is_empty()); // false is the engine default: nothing to write
        } else {
            let top = flag[5..].split('.').next().unwrap_or_default();
            assert!(tree.get(top).is_some(), "{key}");
        }
    }
}

#[test]
fn test_speculative_blocks_trtllm() {
    let hw = trt::hw();
    let scen = |n: &str| {
        trt::scenarios()
            .into_iter()
            .find(|(s, _)| *s == n)
            .expect("scenario")
            .1
    };
    let tree = |params: Obj| {
        yaml_of(
            &build(&ctx_with(
                params.clone(),
                &hw,
                "example-org/chat-model-36B",
                "hf",
                None,
            )),
            &params,
        )
    };
    assert_eq!(
        tree(scen("eagle3"))["speculative_config"],
        json!({"decoding_type": "Eagle", "speculative_model": "example-org/eagle3-head", "max_draft_len": 3,
               "eagle3_one_model": true})
    );
    assert_eq!(
        tree(scen("mtp"))["speculative_config"],
        json!({"decoding_type": "MTP", "num_nextn_predict_layers": 2, "max_concurrency": 16})
    );
    assert_eq!(
        tree(scen("spec_json"))["speculative_config"],
        json!({"decoding_type": "NGram", "max_draft_len": 2, "max_matching_ngram_size": 2})
    );
}

#[test]
fn test_extra_options_are_merged_not_overriding_trtllm() {
    let params = trt::scenarios()
        .into_iter()
        .find(|(s, _)| *s == "extra")
        .expect("scenario")
        .1;
    let tree = yaml_of(
        &build(&ctx_with(
            params.clone(),
            &trt::hw(),
            "example-org/chat-model-36B",
            "hf",
            None,
        )),
        &params,
    );
    assert!(
        tree["garbage_collection_gen0_threshold"] == json!(10000)
            && tree["enable_min_latency"] == json!(true)
    );
    assert!(tree["enable_iter_perf_stats"] == json!(true) && tree["max_stats_len"] == json!(100));
}

#[test]
fn test_iteration_statistics_default_and_history_trtllm() {
    // The console turns the statistics on and keeps a short history; both are the user's to change.
    let hw = trt::hw();
    let tree = |params: Value| yaml_of(&build(&ctx(params.clone(), &hw)), &p(params));
    assert_eq!(
        tree(json!({})),
        json!({"enable_iter_perf_stats": true, "max_stats_len": 100})
    );
    assert_eq!(
        tree(json!({"iter_stats_max_entries": 250})),
        json!({"enable_iter_perf_stats": true, "max_stats_len": 250})
    );
    assert_eq!(
        tree(json!({"enable_iter_perf_stats": true})),
        json!({"enable_iter_perf_stats": true, "max_stats_len": 100})
    );
    assert!(
        build(&ctx(json!({"enable_iter_perf_stats": false}), &hw))
            .files
            .is_empty()
    );
    assert_eq!(
        tree(json!({"enable_iter_perf_stats": false, "iter_stats_max_entries": 50})),
        json!({"max_stats_len": 50})
    );
}

#[test]
fn test_history_keeps_one_scrape_below_the_node_cap_trtllm() {
    // Why 100: the node (MAX_METRICS_TEXT) drops a scrape body above 512 KiB; the stock history of 1000 is larger.
    let doc = read_json("trtllm", "iter_stats_1.2.1.json");
    let one = i64::try_from(
        serde_json::to_string(&doc["with_spec_and_requests"])
            .expect("json")
            .len(),
    )
    .expect("len");
    assert!(800 < one && one < 3000);
    assert!(STATS_HISTORY * one < 512 * 1024 / 2);
    assert!(1000 * one > 512 * 1024 / 2 || one < 1100); // the engine default would sit near or past the cap
}

#[test]
fn test_no_secret_reaches_argv_env_or_files_trtllm() {
    // The engine has no key or token option; nothing secret-shaped may appear anywhere in a launch.
    let hw = trt::hw();
    let secret = Regex::new(r"(?i)(token|secret|password|api[_-]?key)\s*[:=]").expect("re");
    for key in trt::samples().keys() {
        let spec = build(&ctx_with(
            trt::sample_params(key),
            &hw,
            "example-org/chat-model-36B",
            "hf",
            None,
        ));
        let env: Vec<String> = spec
            .env
            .iter()
            .map(|(k, v)| format!("{k}={}", v.as_str().unwrap_or_default()))
            .collect();
        let files: Vec<&str> = spec.files.values().filter_map(Value::as_str).collect();
        let blob = format!(
            "{} {}{}",
            spec.argv.join(" "),
            env.join(" "),
            files.join("")
        );
        for name in [
            "HF_TOKEN",
            "HUGGING_FACE_HUB_TOKEN",
            "API_KEY",
            "TRTLLM_API_KEY",
        ] {
            assert!(!spec.env.contains_key(name));
        }
        assert!(!secret.is_match(&blob), "{key}");
    }
    // `token(?!izer|s)` without look-ahead: a "token" not followed by "izer" or "s"
    let tok = Regex::new(r"token(izer|s)?").expect("re");
    let other = Regex::new(r"secret|password|api_?key").expect("re");
    let allowed = [
        "max_num_tokens",
        "kv_cache_max_tokens",
        "batch_wait_max_tokens_ratio",
        "moe_max_num_tokens",
    ];
    for p in adapter().param_catalog() {
        if allowed.contains(&p.key.as_str()) {
            continue;
        }
        let bare_token = tok.captures_iter(&p.key).any(|c| c.get(1).is_none());
        assert!(
            !bare_token && !other.is_match(&p.key),
            "a secret-looking parameter was added: {}",
            p.key
        );
    }
}
