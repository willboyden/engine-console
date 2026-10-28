//! Test kit of the TensorRT-LLM, llama.cpp and Ollama adapters: fixtures of the Python conftest files
//! (backend/tests/{trtllm,llamacpp,ollama}/conftest.py), the captured engine fixtures (kept in tests/data/<engine>/
//! so they outlive backend/), the Python-made goldens (tests/data/ws_a2/, backend/parity/gen/goldens_ws_a2.py) and an
//! exact, order-aware JSON comparison. Include with `#[path = "common/a2_kit.rs"] mod a2;`.

#![allow(dead_code, clippy::panic, clippy::expect_used, unreachable_pub)]

use std::path::{Path, PathBuf};

use ec_hub::adapters::base::{Compat, FloatMap, Hardware, ModelInfo, Obj};
use ec_hub::domain::models::FitReport;
use serde_json::{Value, json};

pub const GIB: f64 = 1_073_741_824.0;

pub fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data")
}

/// A captured fixture of one engine (tests/data/<engine>/<name>).
pub fn read(engine: &str, name: &str) -> String {
    let p = data_dir().join(engine).join(name);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

pub fn read_json(engine: &str, name: &str) -> Value {
    serde_json::from_str(&read(engine, name)).unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// A golden file of tests/data/ws_a2/.
pub fn golden(name: &str) -> Value {
    read_json("ws_a2", name)
}

/// The learn content of the Python package (embedded at build time by the hub as well, PORTING.md D6).
pub fn learn_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("learn")
}

/// The starter glossary the learn explanations of every engine were written against (backend/tests/*/test_*learn*.py
/// STARTER_TERMS; the same 111 terms in each file, and every one is a key of learn/glossary.json).
pub const STARTER_TERMS: &[&str] = &[
    "model",
    "weights",
    "parameters",
    "inference_engine",
    "serving",
    "gpu",
    "vram",
    "unified_memory",
    "host_ram",
    "shared_memory",
    "page_cache",
    "swap",
    "cgroup",
    "container",
    "docker_image",
    "image_tag",
    "gateway",
    "internal_network",
    "port",
    "endpoint",
    "openai_compatible_api",
    "token",
    "tokenizer",
    "context_window",
    "prompt",
    "prefill",
    "decode",
    "ttft",
    "itl",
    "throughput",
    "latency",
    "batch",
    "continuous_batching",
    "concurrency",
    "kv_cache",
    "paged_attention",
    "prefix_caching",
    "chunked_prefill",
    "quantization",
    "bf16",
    "fp16",
    "fp8",
    "nvfp4",
    "mxfp4",
    "int8",
    "int4",
    "awq",
    "gptq",
    "gguf",
    "gguf_quant_types",
    "safetensors",
    "hugging_face_hub",
    "hf_cache",
    "repo_id",
    "revision",
    "gated_model",
    "hf_token",
    "moe",
    "active_parameters",
    "dense_model",
    "tensor_parallelism",
    "pipeline_parallelism",
    "expert_parallelism",
    "data_parallelism",
    "nccl",
    "nvlink",
    "pcie",
    "peer_to_peer",
    "cuda",
    "cuda_graphs",
    "torch_compile",
    "compute_capability",
    "blackwell",
    "attention_backend",
    "flash_attention",
    "speculative_decoding",
    "draft_model",
    "mtp",
    "rope_scaling",
    "yarn",
    "sliding_window_attention",
    "gqa",
    "mla",
    "lora",
    "tool_calling",
    "reasoning_model",
    "chat_template",
    "trust_remote_code",
    "served_model_name",
    "cpu_offload",
    "kv_cache_offload",
    "gpu_layers_offload",
    "tensor_split",
    "mmap",
    "mlock",
    "cpu_threads",
    "numa",
    "fit_estimate",
    "preflight",
    "prometheus_metrics",
    "sampling",
    "temperature",
    "max_tokens",
    "embedding_model",
    "vision_language_model",
    "multimodal",
    "benchmark",
    "idle_ttl",
    "health_check",
    "warmup",
    "oom",
];

/// `json.loads(learn/<folder>/<engine>.json)`, checked the way the engine learn tests load it: exactly the keys
/// `version` (== 1) and `<folder>`, and the body's keys sorted. Returns the body.
pub fn learn_section(folder: &str, engine: &str) -> serde_json::Map<String, Value> {
    let path = learn_dir().join(folder).join(format!("{engine}.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let doc: Value = serde_json::from_str(&text).expect("learn json");
    let keys: std::collections::BTreeSet<&str> = doc
        .as_object()
        .expect("object")
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(
        keys,
        std::collections::BTreeSet::from(["version", folder]),
        "{}",
        path.display()
    );
    assert_eq!(doc["version"], 1);
    let body = doc[folder].as_object().expect("body").clone();
    let names: Vec<&String> = body.keys().collect();
    let mut want = names.clone();
    want.sort();
    assert_eq!(names, want, "{}: keys must be sorted", path.display());
    body
}

/// The node agent's `filter_metrics` (node/src/ecnode/agent.py), the real one: ecnode is a dev-dependency of ec-hub.
/// Keeps the samples and # HELP / # TYPE lines of the families whose name starts with a prefix, up to the frame cap.
pub fn node_filter_metrics(text: &str, prefixes: &[String]) -> String {
    ecnode::agent::filter_metrics(text, prefixes, ec_proto::MAX_METRICS_TEXT)
}

/// The shared checks of backend/tests/{llamacpp,ollama}/test_learn.py (the two files differ only in the engine name
/// and the facts they pin).
pub mod engine_learn {
    use super::{STARTER_TERMS, learn_dir, learn_section};
    use ec_hub::adapters::base::Explain;
    use ec_hub::adapters::trtllm_parse::py;
    use regex::Regex;
    use serde_json::Value;

    pub const LIMITS: [(&str, usize); 5] = [
        ("summary", 160),
        ("details", 900),
        ("when_to_change", 400),
        ("risks", 400),
        ("example", 200),
    ];
    /// `Explain.model_fields`
    pub const EXPLAIN_FIELDS: [&str; 7] = [
        "summary",
        "details",
        "when_to_change",
        "risks",
        "example",
        "terms",
        "level",
    ];

    fn field<'a>(e: &'a Explain, name: &str) -> &'a str {
        match name {
            "summary" => &e.summary,
            "details" => &e.details,
            "when_to_change" => &e.when_to_change,
            "risks" => &e.risks,
            _ => &e.example,
        }
    }

    /// `test_explanations_follow_the_readme[kind]`.
    pub fn explanations_follow_the_readme(kind: &str, engine: &str) {
        let todo = Regex::new(r"\b(TODO|TBD|FIXME)\b|\?\?\?").expect("re");
        let sentence_break = Regex::new(r#"[.!?]\s+[A-Z0-9`'"(]"#).expect("re");
        for (key, raw) in learn_section(kind, engine) {
            let e: Explain =
                serde_json::from_value(raw.clone()).unwrap_or_else(|err| panic!("{key}: {err}"));
            assert!(
                ["basic", "advanced", "expert"].contains(&e.level.as_str()),
                "{key}"
            );
            assert!(
                raw.as_object()
                    .expect("entry")
                    .keys()
                    .all(|k| EXPLAIN_FIELDS.contains(&k.as_str())),
                "{key}"
            );
            for (name, limit) in LIMITS {
                let n = field(&e, name).chars().count();
                assert!(n <= limit, "{key} {name} {n}");
            }
            for name in ["summary", "details", "when_to_change", "risks", "example"] {
                let text = field(&e, name);
                assert!(!py::strip(text).is_empty(), "{key} {name}"); // all five filled, not just a summary
                assert!(
                    text == py::strip(text) && !text.contains("  "),
                    "{key} {name}"
                );
            }
            assert!(
                e.summary.ends_with('.') && !e.summary.contains('\n') && !e.example.contains('\n'),
                "{key}"
            );
            assert!(
                !sentence_break.is_match(&e.summary),
                "{key}: more than one sentence"
            );
            // json.dumps(raw): non-ASCII escaped, as the Python check sees it
            assert!(!todo.is_match(&ec_hub::pyjson::dumps(&raw)), "{key}");
            let mut sorted = e.terms.clone();
            sorted.sort();
            sorted.dedup();
            assert_eq!(e.terms, sorted, "{key}");
        }
    }

    /// `test_terms_come_from_the_starter_glossary_or_the_declared_new_ones[kind]`.
    pub fn terms_from_the_starter_glossary_or_new(kind: &str, engine: &str, new_terms: &[&str]) {
        let used: std::collections::BTreeSet<String> = learn_section(kind, engine)
            .values()
            .flat_map(|raw| raw["terms"].as_array().cloned().unwrap_or_default())
            .filter_map(|t| t.as_str().map(str::to_string))
            .collect();
        let outside: Vec<&String> = used
            .iter()
            .filter(|t| !STARTER_TERMS.contains(&t.as_str()) && !new_terms.contains(&t.as_str()))
            .collect();
        assert!(outside.is_empty(), "{outside:?}");
        assert!(used.iter().any(|t| STARTER_TERMS.contains(&t.as_str()))); // the explanations really link something
    }

    /// `test_no_personal_or_lab_specific_content`.
    pub fn no_personal_or_lab_specific_content(engine: &str) {
        let read = |kind: &str| {
            std::fs::read_to_string(learn_dir().join(kind).join(format!("{engine}.json")))
                .expect("learn file")
        };
        let text = read("params") + &read("presets");
        let re = Regex::new(r"/home/|/Users/|@[A-Za-z0-9-]+\.[a-z]{2,}|\b\d{1,3}(\.\d{1,3}){3}\b")
            .expect("re");
        assert!(!re.is_match(&text));
    }

    /// `test_levels_match_the_catalog`, the catalog part.
    pub fn levels_match(catalog: &[ec_hub::adapters::base::ParamSpec], engine: &str) {
        let body = learn_section("params", engine);
        for p in catalog {
            let expected = match p.level.as_deref() {
                Some(l) if !l.is_empty() => l.to_string(),
                _ if p.advanced => "advanced".to_string(),
                _ => "basic".to_string(),
            };
            assert_eq!(
                body[&p.key]["level"].as_str(),
                Some(expected.as_str()),
                "{}",
                p.key
            );
        }
        for raw in learn_section("presets", engine).values() {
            assert!(matches!(raw["level"].as_str(), Some("basic" | "advanced")));
        }
    }

    pub fn text(body: &serde_json::Map<String, Value>, key: &str, field: &str) -> String {
        body[key][field]
            .as_str()
            .unwrap_or_else(|| panic!("{key}.{field}"))
            .to_string()
    }
}

/// `services.fit.estimate_fit(model, mem, hw)` with Python's defaults (concurrency 1, nothing resident, no compat
/// checks, no wider hardware): the real estimator.
pub fn estimate_fit(model: &ModelInfo, mem: &FloatMap, hw: &Hardware) -> FitReport {
    ec_hub::services::fit::estimate_fit(
        model,
        &ec_hub::services::fit::Mem::from_map(mem),
        hw,
        1,
        &[],
        &[],
        None,
    )
    .expect("estimate_fit")
}

/// `model.model_copy(update={...})`.
pub fn with(m: &ModelInfo, update: Value) -> ModelInfo {
    let mut v = serde_json::to_value(m).expect("model");
    let Value::Object(update) = update else {
        panic!("update must be an object")
    };
    for (k, x) in update {
        v[&k] = x;
    }
    model(v)
}

/// `hw.model_copy(update={...})`.
pub fn hw_with(h: &Hardware, update: Value) -> Hardware {
    let mut v = serde_json::to_value(h).expect("hw");
    let Value::Object(update) = update else {
        panic!("update must be an object")
    };
    for (k, x) in update {
        v[&k] = x;
    }
    hardware(v)
}

/// `{...}` as a params object.
pub fn p(v: Value) -> Obj {
    match v {
        Value::Object(m) => m,
        other => panic!("params must be an object, got {other}"),
    }
}

pub fn hardware(v: Value) -> Hardware {
    serde_json::from_value(v).expect("hardware")
}

pub fn model(v: Value) -> ModelInfo {
    serde_json::from_value(v).expect("model")
}

/// The codes of a compatibility answer, in order.
pub fn codes(found: &[Compat]) -> Vec<String> {
    found.iter().map(|c| c.code.clone()).collect()
}

/// The finding with this code (panics with the whole answer when absent).
pub fn find<'a>(found: &'a [Compat], code: &str) -> &'a Compat {
    found
        .iter()
        .find(|c| c.code == code)
        .unwrap_or_else(|| panic!("no {code} in {:?}", codes(found)))
}

pub fn has(found: &[Compat], code: &str) -> bool {
    found.iter().any(|c| c.code == code)
}

/// `dict(memory_model(...))[key]`.
pub fn fget(m: &FloatMap, key: &str) -> Option<f64> {
    m.iter().find(|(k, _)| k == key).map(|(_, v)| *v)
}

pub fn keys(m: &FloatMap) -> Vec<&str> {
    m.iter().map(|(k, _)| k.as_str()).collect()
}

pub fn approx(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-6 * b.abs().max(1e-12)
}

/// A FloatMap as the golden encoding: a JSON object, non-finite floats as {"$f": "nan"|"inf"|"-inf"}.
pub fn floatmap_json(m: &FloatMap) -> Value {
    let mut o = serde_json::Map::new();
    for (k, v) in m {
        o.insert(k.clone(), float_json(*v));
    }
    Value::Object(o)
}

pub fn float_json(v: f64) -> Value {
    if v.is_nan() {
        json!({"$f": "nan"})
    } else if v.is_infinite() {
        json!({"$f": if v > 0.0 { "inf" } else { "-inf" }})
    } else {
        Value::from(v)
    }
}

/// Exact JSON equality: ints equal ints, floats equal floats bit for bit (so `1` is not `1.0` and `-0.0` is not
/// `0.0`), list order matters, and object key order matters too when `ordered` (env and files order become argv).
pub fn same(a: &Value, b: &Value, ordered: bool) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => {
            if x.is_f64() || y.is_f64() {
                x.is_f64()
                    && y.is_f64()
                    && x.as_f64().map(f64::to_bits) == y.as_f64().map(f64::to_bits)
            } else {
                x == y
            }
        }
        (Value::Array(x), Value::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| same(p, q, ordered))
        }
        (Value::Object(x), Value::Object(y)) => {
            if x.len() != y.len() {
                return false;
            }
            if ordered && !x.keys().eq(y.keys()) {
                return false;
            }
            x.iter()
                .all(|(k, v)| y.get(k).is_some_and(|w| same(v, w, ordered)))
        }
        _ => a == b,
    }
}

#[track_caller]
pub fn assert_same(got: &Value, want: &Value, ordered: bool, what: &str) {
    assert!(
        same(got, want, ordered),
        "{what}\n  got:  {}\n  want: {}",
        serde_json::to_string(got).unwrap_or_default(),
        serde_json::to_string(want).unwrap_or_default()
    );
}

// ---- fixtures of backend/tests/trtllm/conftest.py ----------------------------------------------------------------
pub mod trt {
    use super::*;

    pub fn hw() -> Hardware {
        hardware(
            json!({"gpu_ids": [0, 1], "gpu_uuids": ["GPU-aaaa", "GPU-bbbb"],
            "gpu_names": ["RTX PRO 6000 Blackwell", "RTX PRO 6000 Blackwell"], "gpu_total_gib": [95.6, 95.6],
            "gpu_free_gib": [94.0, 94.0], "compute_capability": "12.0", "host_ram_gib": 246.0}),
        )
    }

    /// One selected GPU.
    pub fn hw1() -> Hardware {
        hardware(
            json!({"gpu_ids": [0], "gpu_uuids": ["GPU-aaaa"], "gpu_names": ["RTX PRO 6000 Blackwell"],
            "gpu_total_gib": [95.6], "gpu_free_gib": [94.0], "compute_capability": "12.0", "host_ram_gib": 246.0}),
        )
    }

    /// A Llama-family dense model with FP8-sized weights: 36 GiB.
    pub fn dense() -> ModelInfo {
        model(
            json!({"repo_id": "example-org/chat-model-36B", "architectures": ["LlamaForCausalLM"],
            "model_type": "llama", "num_params": 36_000_000_000_i64, "weight_bytes": 36_i64 * 1024 * 1024 * 1024,
            "num_attention_heads": 64, "num_kv_heads": 8, "max_position_embeddings": 131_072,
            "pipeline_tag": "text-generation"}),
        )
    }

    /// backend/tests/trtllm/samples.py SAMPLES: one valid sample value per catalog key.
    pub fn samples() -> Obj {
        // parsed from text: the literal is too deep for the json! macro
        p(serde_json::from_str(
            r#"{
            "free_gpu_memory_fraction": 0.8, "kv_cache_dtype": "fp8", "kv_cache_max_tokens": 65536,
            "kv_cache_max_gib": 20.0, "kv_host_cache_gib": 8.0, "tokens_per_block": 64,
            "kv_max_attention_window": ["4096", "256"], "mamba_ssm_cache_dtype": "float32",
            "tensor_parallel_size": 2, "pipeline_parallel_size": 2, "context_parallel_size": 2,
            "moe_expert_parallel_size": 2, "moe_tensor_parallel_size": 2, "moe_cluster_parallel_size": 2,
            "enable_attention_dp": true, "allreduce_strategy": "NCCL", "orchestrator_type": "rpc",
            "nccl_p2p_disable": true,
            "max_seq_len": 32768, "max_beam_width": 2, "media_io_kwargs": {"video": {"num_frames": 8}},
            "revision": "main", "tokenizer": "example-org/tokenizer-repo", "tokenizer_mode": "slow",
            "custom_tokenizer": "deepseek_v32", "trust_remote_code": true, "moe_backend": "CUTLASS",
            "moe_max_num_tokens": 4096, "nvfp4_gemm_backends": ["cutlass", "cuda_core"],
            "max_batch_size": 256, "max_num_tokens": 4096, "enable_chunked_prefill": true,
            "enable_block_reuse": false, "enable_partial_reuse": false,
            "capacity_scheduler_policy": "MAX_UTILIZATION", "context_chunking_policy": "EQUAL_PROGRESS",
            "disable_overlap_scheduler": true, "stream_interval": 4, "batch_wait_timeout_ms": 5.0,
            "batch_wait_timeout_iters": 2, "batch_wait_max_tokens_ratio": 0.5, "num_postprocess_workers": 2,
            "sampler_type": "TorchSampler",
            "tool_parser": "qwen3_coder", "reasoning_parser": "qwen3",
            "chat_template": "{{ messages[0]['content'] }}", "guided_decoding_backend": "xgrammar",
            "enable_cuda_graphs": false, "cuda_graph_max_batch_size": 64, "cuda_graph_batch_sizes": ["1", "2", "4", "8"],
            "cuda_graph_padding": true, "attention_backend": "FLASHINFER", "enable_torch_compile": true,
            "torch_compile_piecewise_cuda_graph": true, "torch_compile_inductor": true, "enable_autotuner": false,
            "spec_decoding_type": "NGram", "spec_max_draft_len": 4, "spec_draft_model": "example-org/draft-model-1B",
            "spec_num_nextn_predict_layers": 2, "spec_ngram_max_matching_size": 3, "spec_eagle3_one_model": false,
            "spec_max_concurrency": 8,
            "speculative_config": {"decoding_type": "MTP", "num_nextn_predict_layers": 2},
            "log_level": "debug", "otlp_traces_endpoint": "http://otel-collector:4317", "enable_iter_perf_stats": false,
            "iter_stats_max_entries": 250, "print_iter_log": true,
            "extra_llm_options": {"enable_min_latency": true, "garbage_collection_gen0_threshold": 10000},
            "image": "nvcr.io/nvidia/tensorrt-llm/release:1.2.1"
        }"#,
        )
        .expect("samples"))
    }

    /// samples.py COMPANIONS: what a key needs next to it so the combination validates.
    pub fn companions(key: &str) -> Obj {
        p(match key {
            "moe_expert_parallel_size"
            | "moe_tensor_parallel_size"
            | "moe_cluster_parallel_size" => {
                json!({"tensor_parallel_size": 2})
            }
            "torch_compile_piecewise_cuda_graph" | "torch_compile_inductor" => {
                json!({"enable_torch_compile": true})
            }
            "spec_decoding_type" => {
                json!({"spec_max_draft_len": 4, "spec_ngram_max_matching_size": 3})
            }
            "spec_max_draft_len" => {
                json!({"spec_decoding_type": "NGram", "spec_ngram_max_matching_size": 3})
            }
            "spec_draft_model" => {
                json!({"spec_decoding_type": "DraftTarget", "spec_max_draft_len": 3})
            }
            "spec_num_nextn_predict_layers" | "spec_max_concurrency" => {
                json!({"spec_decoding_type": "MTP"})
            }
            "spec_ngram_max_matching_size" => {
                json!({"spec_decoding_type": "NGram", "spec_max_draft_len": 4})
            }
            "spec_eagle3_one_model" => {
                json!({"spec_decoding_type": "Eagle", "spec_draft_model": "example-org/eagle3-head",
                                                "spec_max_draft_len": 3})
            }
            _ => json!({}),
        })
    }

    /// `sample_params(key)`: the key with its sample value and the companions it needs.
    pub fn sample_params(key: &str) -> Obj {
        let mut out = Obj::new();
        out.insert(
            key.into(),
            samples().get(key).cloned().unwrap_or(Value::Null),
        );
        for (k, v) in companions(key) {
            out.insert(k, v);
        }
        out
    }

    /// samples.py SCENARIOS (in sorted order, as the parametrized test runs them).
    pub fn scenarios() -> Vec<(&'static str, Obj)> {
        let mut s = vec![
            ("empty", json!({})),
            (
                "mtp",
                json!({"spec_decoding_type": "MTP", "spec_num_nextn_predict_layers": 2, "spec_max_concurrency": 16}),
            ),
            (
                "eagle3",
                json!({"spec_decoding_type": "Eagle", "spec_draft_model": "example-org/eagle3-head",
                              "spec_max_draft_len": 3, "spec_eagle3_one_model": true}),
            ),
            (
                "ngram",
                json!({"spec_decoding_type": "NGram", "spec_max_draft_len": 4, "spec_ngram_max_matching_size": 3}),
            ),
            (
                "draft_target",
                json!({"spec_decoding_type": "DraftTarget", "spec_draft_model": "example-org/draft-model-1B",
                                    "spec_max_draft_len": 3}),
            ),
            ("auto_spec", json!({"spec_decoding_type": "AUTO"})),
            (
                "spec_json",
                json!({"speculative_config": {"decoding_type": "NGram", "max_draft_len": 2,
                                                        "max_matching_ngram_size": 2}}),
            ),
            ("graphs_off", json!({"enable_cuda_graphs": false})),
            (
                "graphs_sized",
                json!({"cuda_graph_batch_sizes": ["1", "2", "4", "8", "16"], "cuda_graph_padding": true}),
            ),
            (
                "compile",
                json!({"enable_torch_compile": true, "torch_compile_piecewise_cuda_graph": true,
                               "torch_compile_inductor": true}),
            ),
            (
                "scheduler",
                json!({"capacity_scheduler_policy": "MAX_UTILIZATION",
                                 "context_chunking_policy": "EQUAL_PROGRESS", "enable_chunked_prefill": true}),
            ),
            (
                "kv",
                json!({"kv_cache_dtype": "fp8", "enable_block_reuse": false, "kv_cache_max_gib": 12.5,
                          "kv_host_cache_gib": 4.0, "tokens_per_block": 64, "kv_max_attention_window": ["4096", "512"]}),
            ),
            (
                "moe_tp4",
                json!({"tensor_parallel_size": 4, "moe_expert_parallel_size": 2, "moe_tensor_parallel_size": 2,
                               "moe_backend": "CUTLASS", "moe_max_num_tokens": 8192}),
            ),
            (
                "parsers",
                json!({"tool_parser": "qwen3", "reasoning_parser": "qwen3", "guided_decoding_backend": "llguidance",
                               "chat_template": "{{ messages }}"}),
            ),
            (
                "extra",
                json!({"extra_llm_options": {"garbage_collection_gen0_threshold": 10000, "enable_min_latency": true}}),
            ),
            (
                "stats_history",
                json!({"enable_iter_perf_stats": true, "iter_stats_max_entries": 250, "print_iter_log": true}),
            ),
            (
                "everything_on_cli",
                json!({"tensor_parallel_size": 2, "pipeline_parallel_size": 1, "max_seq_len": 65536,
                                         "max_batch_size": 64, "max_num_tokens": 8192, "free_gpu_memory_fraction": 0.85,
                                         "num_postprocess_workers": 2, "trust_remote_code": true, "revision": "main",
                                         "tokenizer": "example-org/tokenizer-repo", "log_level": "info",
                                         "otlp_traces_endpoint": "http://otel-collector:4317", "enable_chunked_prefill": true,
                                         "max_beam_width": 1,
                                         "media_io_kwargs": {"video": {"num_frames": 8}}, "reasoning_parser": "deepseek-r1",
                                         "tool_parser": "deepseek_v3", "chat_template": "{{ messages }}"}),
            ),
        ];
        s.sort_by(|a, b| a.0.cmp(b.0));
        s.into_iter().map(|(n, v)| (n, p(v))).collect()
    }
}

/// Port of backend/tests/llamacpp/helpers.py: reads the captured `llama-server --help` into entries.
pub mod llama_help {
    use super::read;
    use ec_hub::adapters::trtllm_parse::py;
    use regex::Regex;

    #[derive(Debug, Clone)]
    pub struct HelpEntry {
        pub flags: Vec<String>,
        /// description lines joined with single spaces
        pub text: String,
        pub env: Option<String>,
        /// the text after the first "default:" of the entry (a literal, or a phrase such as "same as ...")
        pub default: Option<String>,
    }

    impl HelpEntry {
        pub fn deprecated(&self) -> bool {
            self.text.contains("DEPRECATED")
        }
    }

    pub fn help() -> String {
        read("llamacpp", "llama-server-help.txt")
    }

    /// `_FLAG.findall`: `(?<![\w-])(--?[A-Za-z][\w-]*)`. A start inside a refused match is also preceded by a word
    /// character or `-`, so skipping to the end of a refused match finds the same flags as Python's scan.
    fn flags_of(part: &str) -> Vec<String> {
        let re = Regex::new(r"--?[A-Za-z][\w-]*").expect("re");
        re.find_iter(part)
            .filter(|m| {
                part[..m.start()]
                    .chars()
                    .next_back()
                    .is_none_or(|c| !(c == '-' || c == '_' || c.is_alphanumeric()))
            })
            .map(|m| m.as_str().to_string())
            .collect()
    }

    /// An entry starts at a line that begins with '-'; its description is the rest of that line and the indented lines.
    pub fn parse_help(text: &str) -> Vec<HelpEntry> {
        let env_re = Regex::new(r"\(env: ([A-Z0-9_]+)\)").expect("re");
        let default_re = Regex::new(r"default:\s*'?([^,()';=]+)'?").expect("re");
        let mut entries = Vec::new();
        let mut current: Vec<&str> = Vec::new();
        let mut flush = |current: &[&str]| {
            let Some(head) = current.first() else { return };
            let chars: Vec<char> = head.chars().collect();
            // the help prints flags in the first 40 columns and the description from column 40; a longer flag list
            // leaves the description to the following lines
            let same_line =
                chars.len() > 40 && chars[38] == ' ' && chars[39] == ' ' && chars[40] != ' ';
            let (flags_part, first_desc): (String, String) = if same_line {
                (chars[..40].iter().collect(), chars[40..].iter().collect())
            } else {
                ((*head).to_string(), String::new())
            };
            let flags = flags_of(&flags_part);
            let mut parts = vec![py::strip(&first_desc).to_string()];
            parts.extend(current[1..].iter().map(|l| py::strip(l).to_string()));
            let desc = py::strip(&parts.join(" ")).to_string();
            let env = env_re.captures(&desc).map(|c| c[1].to_string());
            let default = default_re
                .captures(&desc)
                .map(|c| py::strip(&c[1]).to_string());
            entries.push(HelpEntry {
                flags,
                text: desc,
                env,
                default,
            });
        };
        for line in ec_hub::services::common::py_splitlines(text) {
            if line.starts_with('-') {
                flush(&current);
                current = vec![line];
            } else if !current.is_empty() && line.starts_with(' ') {
                current.push(line);
            } else {
                flush(&current);
                current = Vec::new();
            }
        }
        flush(&current);
        entries
    }

    /// The pinned help, parsed once per test binary.
    fn entries() -> &'static [HelpEntry] {
        static ENTRIES: std::sync::OnceLock<Vec<HelpEntry>> = std::sync::OnceLock::new();
        ENTRIES.get_or_init(|| parse_help(&help()))
    }

    pub fn help_flags() -> std::collections::BTreeSet<String> {
        entries().iter().flat_map(|e| e.flags.clone()).collect()
    }

    pub fn entry_for(flag: &str) -> HelpEntry {
        entries()
            .iter()
            .find(|e| e.flags.iter().any(|f| f == flag))
            .cloned()
            .unwrap_or_else(|| panic!("KeyError: {flag}"))
    }
}

/// Port of backend/tests/ollama/helpers.py: readers for the captured Ollama output (the `ollama serve --help` variable
/// list and the "server config" start-up line, the effective value of every variable).
pub mod ollama_help {
    use super::read;
    use ec_hub::adapters::trtllm_parse::py;
    use ec_hub::services::common::py_splitlines;
    use regex::Regex;
    use std::collections::BTreeMap;

    /// name -> description of every variable under "Environment Variables:" in the help.
    pub fn help_env() -> BTreeMap<String, String> {
        let text = read("ollama", "ollama-serve-help.txt");
        let body = text
            .split_once("Environment Variables:")
            .expect("Environment Variables:")
            .1;
        // re.match(r"\s+([A-Z][A-Z0-9_]+)\s{2,}(.+)$", line): anchored at the start; `$` also before a final newline,
        // which splitlines() never leaves
        let re = Regex::new(r"^\s+([A-Z][A-Z0-9_]+)\s{2,}(.+)$").expect("re");
        py_splitlines(body)
            .into_iter()
            .filter_map(|line| {
                re.captures(line)
                    .map(|m| (m[1].to_string(), py::strip(&m[2]).to_string()))
            })
            .collect()
    }

    /// name -> effective value, from `msg="server config" env="map[NAME:value ...]"` (lists keep their brackets).
    pub fn server_config() -> BTreeMap<String, String> {
        let text = read("ollama", "serve_bare_0.30.11.log");
        let line = py_splitlines(&text)
            .into_iter()
            .find(|l| l.contains("msg=\"server config\""))
            .expect("server config");
        let inner = line.split_once("env=\"map[").expect("env=map").1;
        let inner = inner.rsplit_once("]\"").expect("]\"").0;
        Regex::new(r"([A-Za-z_]+):(\[[^\]]*\]|[^\s\[\]]*)")
            .expect("re")
            .captures_iter(inner)
            .map(|m| (m[1].to_string(), m[2].to_string()))
            .collect()
    }
}

// ---- fixtures of backend/tests/llamacpp/conftest.py ---------------------------------------------------------------
pub mod llama {
    use super::*;

    /// One Blackwell card with 90 GiB free.
    pub fn hw() -> Hardware {
        hardware(
            json!({"gpu_ids": [0], "gpu_uuids": ["GPU-aaaa"], "gpu_names": ["RTX PRO 6000 Blackwell"],
            "gpu_total_gib": [95.6], "gpu_free_gib": [90.0], "compute_capability": "12.0", "host_ram_gib": 246.0}),
        )
    }

    pub fn hw2() -> Hardware {
        hardware(
            json!({"gpu_ids": [0, 1], "gpu_uuids": ["GPU-aaaa", "GPU-bbbb"],
            "gpu_names": ["RTX PRO 6000 Blackwell", "RTX PRO 6000 Blackwell"], "gpu_total_gib": [95.6, 95.6],
            "gpu_free_gib": [90.0, 40.0], "compute_capability": "12.0", "host_ram_gib": 246.0}),
        )
    }

    /// A 32B-class dense GGUF (Q4_K_M, 18.6 GiB): 64 layers, 8 KV heads of 128.
    pub fn dense() -> ModelInfo {
        #[allow(clippy::cast_possible_truncation)]
        let wb = (18.6 * GIB) as i64;
        model(
            json!({"repo_id": "example-org/chat-model-32B-GGUF", "architectures": ["qwen3"], "model_type": "qwen3",
            "num_params": 32_000_000_000_i64, "weight_bytes": wb, "quantization": "q4_k_m",
            "hidden_size": 5120, "num_layers": 64, "num_attention_heads": 64, "num_kv_heads": 8, "head_dim": 128,
            "max_position_embeddings": 40960}),
        )
    }

    /// A 30B-A3B-class mixture-of-experts GGUF (17.5 GiB): 48 layers, 4 KV heads of 128, 8 of 128 experts.
    pub fn moe() -> ModelInfo {
        #[allow(clippy::cast_possible_truncation)]
        let wb = (17.5 * GIB) as i64;
        model(
            json!({"repo_id": "example-org/moe-30B-A3B-GGUF", "architectures": ["qwen3moe"], "model_type": "qwen3moe",
            "num_params": 30_500_000_000_i64, "num_active_params": 3_300_000_000_i64, "is_moe": true,
            "weight_bytes": wb, "quantization": "q4_k_m", "hidden_size": 2048, "num_layers": 48,
            "num_attention_heads": 32, "num_kv_heads": 4, "head_dim": 128, "max_position_embeddings": 262_144,
            "raw_config": {"qwen3moe.expert_count": 128, "qwen3moe.expert_used_count": 8}}),
        )
    }
}

// ---- fixtures of backend/tests/ollama/conftest.py ----------------------------------------------------------------
pub mod oll {
    use super::*;

    pub fn hw() -> Hardware {
        super::llama::hw()
    }

    pub fn hw2() -> Hardware {
        super::llama::hw2()
    }

    /// A 16 GiB card with 14 GiB free (Ada, compute capability 8.9).
    pub fn small_gpu() -> Hardware {
        hardware(
            json!({"gpu_ids": [0], "gpu_uuids": ["GPU-cccc"], "gpu_names": ["RTX 4080"], "gpu_total_gib": [16.0],
            "gpu_free_gib": [14.0], "compute_capability": "8.9", "host_ram_gib": 64.0}),
        )
    }

    /// An 8B-class dense model from an Ollama store: 32 layers, 8 KV heads of 128.
    pub fn llama8b() -> ModelInfo {
        #[allow(clippy::cast_possible_truncation)]
        let wb = (4.6 * GIB) as i64;
        model(
            json!({"repo_id": "llama3.1:8b", "architectures": ["llama"], "model_type": "llama",
            "num_params": 8_000_000_000_i64, "weight_bytes": wb, "quantization": "q4_k_m", "hidden_size": 4096,
            "num_layers": 32, "num_attention_heads": 32, "num_kv_heads": 8, "head_dim": 128,
            "max_position_embeddings": 131_072}),
        )
    }

    /// What the library knows about an Ollama model whose config blob carries no layer or head counts.
    pub fn bare() -> ModelInfo {
        model(json!({"repo_id": "mystery:latest", "weight_bytes": 8_i64 * 1024 * 1024 * 1024}))
    }
}

/// The `add("<level>", "<code>", ...)` literals of the last Python adapter module (the learn gate's view of the codes
/// a module can emit): every code with its level, in source order. Frozen at the cutover from the last Python backend
/// (tests/data/ws_a2/python_add_codes.json).
pub fn python_add_codes(module: &str) -> Vec<(String, String)> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/ws_a2/python_add_codes.json");
    let doc: Value = serde_json::from_str(
        &std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display())),
    )
    .expect("python_add_codes.json");
    doc[module]
        .as_array()
        .unwrap_or_else(|| panic!("no frozen codes for {module}"))
        .iter()
        .map(|p| {
            (
                p[0].as_str().unwrap_or_default().to_string(),
                p[1].as_str().unwrap_or_default().to_string(),
            )
        })
        .collect()
}

/// The codes a Rust adapter module can emit: every `add("<level>", "<code>", ...)` in its source, plus the one
/// computed-level call of llamacpp.rs.
pub fn rust_add_codes(module: &str) -> Vec<String> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/adapters")
        .join(module);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let re =
        regex::Regex::new(r#"(?s)\badd\(\s*(?:"(?:ok|warn|block)"|if [^,]+),\s*"([a-z0-9_]+)""#)
            .expect("re");
    re.captures_iter(&text).map(|c| c[1].to_string()).collect()
}
