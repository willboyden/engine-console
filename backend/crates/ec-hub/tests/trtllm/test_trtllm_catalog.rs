//! Port of backend/tests/trtllm/test_trtllm_catalog.py: the TensorRT-LLM catalog against the fixtures captured from
//! the pinned image (tests/data/trtllm: the click dump and pydantic field tables in llm_args_1.2.1.json, the help).

use std::collections::{BTreeMap, BTreeSet};

use crate::a2::{read, read_json, trt};
use ec_hub::adapters::base::{EngineAdapter, ParamSpec};
use ec_hub::adapters::trtllm::{TrtllmAdapter, scheduler_defaults};
use ec_hub::adapters::trtllm_params::{
    DEFAULT_IMAGE, DOCS_BASE, EXCLUDED_SERVE_FLAGS, IMAGE_DIGEST, SOURCE_COMMIT, SPEC_TYPES,
    SUPPORTED_ARCHITECTURES, TORCH_LLM_ARGS_FIELDS, YAML_EXTRA_FLAG, YAML_NESTED, YAML_PREFIX,
};
use regex::Regex;
use serde_json::Value;

fn schema() -> Value {
    read_json("trtllm", "llm_args_1.2.1.json")
}

fn adapter() -> TrtllmAdapter {
    TrtllmAdapter::new()
}

fn by_key(a: &TrtllmAdapter) -> BTreeMap<String, ParamSpec> {
    a.param_catalog()
        .into_iter()
        .map(|p| (p.key.clone(), p))
        .collect()
}

fn re(p: &str) -> Regex {
    Regex::new(p).expect("regex")
}

fn literal_values(type_str: &str) -> Option<Vec<String>> {
    let m = re(r"Literal\[(.*?)\]").captures(type_str)?;
    Some(
        re(r"'([^']+)'")
            .captures_iter(&m[1])
            .map(|c| c[1].to_string())
            .collect(),
    )
}

/// The fixture field table that holds the last path element, and that element's name.
fn class_for_path<'a>(
    schema: &'a Value,
    path: &[&'a str],
) -> (&'a serde_json::Map<String, Value>, &'a str) {
    let mut fields = schema["classes"]["TorchLlmArgs"]
        .as_object()
        .expect("TorchLlmArgs");
    for part in &path[..path.len() - 1] {
        let cls = YAML_NESTED
            .iter()
            .find(|(f, _)| f == part)
            .map(|(_, c)| *c)
            .unwrap_or_else(|| panic!("{part} is not a nested config class in YAML_NESTED"));
        fields = schema["classes"][cls].as_object().expect("class");
    }
    (fields, path[path.len() - 1])
}

fn serve_option_names(schema: &Value, only_options: bool) -> BTreeSet<String> {
    schema["serve_params"]
        .as_array()
        .expect("serve_params")
        .iter()
        .filter(|p| !only_options || p["kind"] == "Option")
        .flat_map(|p| p["opts"].as_array().cloned().unwrap_or_default())
        .filter_map(|o| o.as_str().map(str::to_string))
        .collect()
}

/// Python's `==` on JSON values (1 == 1.0 == True).
fn py_eq(a: &Value, b: &Value) -> bool {
    let num = |v: &Value| match v {
        Value::Bool(b) => Some(f64::from(u8::from(*b))),
        Value::Number(n) => n.as_f64(),
        _ => None,
    };
    match (num(a), num(b)) {
        (Some(x), Some(y)) if !(a.is_string() || b.is_string()) => x == y,
        _ => a == b,
    }
}

#[test]
fn test_catalog_size_and_uniqueness_trtllm() {
    let cat = adapter().param_catalog();
    assert!(cat.len() >= 60);
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
fn test_every_param_is_well_formed_trtllm() {
    let flag_re =
        re(r"^(?:--[a-z0-9_]+|yaml:[a-z0-9_]+(?:\.[a-z0-9_]+)*|yaml:\*|env:[A-Z0-9_]+|@image)$");
    let key_re = re(r"^[a-z][a-z0-9_]*$");
    for p in adapter().param_catalog() {
        assert!(key_re.is_match(&p.key), "{}", p.key);
        assert!(flag_re.is_match(&p.flag), "{}", p.flag);
        assert!(!p.help.trim().is_empty() && !p.label.trim().is_empty());
        let docs = p.docs_url.clone().unwrap_or_default();
        assert!(
            docs.starts_with(&format!(
                "https://github.com/NVIDIA/TensorRT-LLM/blob/{SOURCE_COMMIT}/"
            )),
            "{}",
            p.key
        );
        if p.type_ == "enum" {
            let choices = p.choices.clone().unwrap_or_default();
            assert!(!choices.is_empty(), "{}", p.key);
            if !p.default.is_null() {
                assert!(
                    choices.contains(&ec_proto::pyfmt::str_value(&p.default)),
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
        if p.type_ == "bool" && (p.flag.starts_with("yaml:") || p.flag.starts_with("env:")) {
            assert_eq!(p.bool_style, "value", "{}", p.key); // explicit true/false is emitted
        }
        assert!(p.help.chars().count() <= 400, "{}", p.key);
    }
}

#[test]
fn test_required_params_present_trtllm() {
    let keys: BTreeSet<String> = by_key(&adapter()).into_keys().collect();
    for k in [
        "free_gpu_memory_fraction",
        "kv_cache_dtype",
        "enable_block_reuse",
        "max_seq_len",
        "max_batch_size",
        "max_num_tokens",
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "moe_expert_parallel_size",
        "enable_chunked_prefill",
        "enable_cuda_graphs",
        "attention_backend",
        "tool_parser",
        "reasoning_parser",
        "chat_template",
        "spec_decoding_type",
        "speculative_config",
        "guided_decoding_backend",
        "enable_iter_perf_stats",
        "trust_remote_code",
        "revision",
        "image",
        "extra_llm_options",
        "moe_backend",
        "log_level",
    ] {
        assert!(keys.contains(k), "{k}");
    }
}

#[test]
fn test_all_groups_used_trtllm() {
    let groups: BTreeSet<String> = adapter()
        .param_catalog()
        .into_iter()
        .map(|p| p.group)
        .collect();
    let want: BTreeSet<String> = [
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
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect();
    assert_eq!(groups, want);
}

#[test]
fn test_memory_params_flagged_trtllm() {
    let by = by_key(&adapter());
    for k in [
        "free_gpu_memory_fraction",
        "kv_cache_dtype",
        "max_seq_len",
        "max_num_tokens",
        "max_batch_size",
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "kv_cache_max_gib",
        "kv_host_cache_gib",
        "enable_cuda_graphs",
        "spec_decoding_type",
    ] {
        assert!(by[k].affects_memory, "{k}");
    }
}

#[test]
fn test_samples_cover_catalog_trtllm() {
    let keys: BTreeSet<String> = by_key(&adapter()).into_keys().collect();
    let samples: BTreeSet<String> = trt::samples().keys().cloned().collect();
    assert_eq!(samples, keys, "samples must cover the catalog exactly");
    for k in [
        "moe_expert_parallel_size",
        "moe_tensor_parallel_size",
        "moe_cluster_parallel_size",
        "torch_compile_piecewise_cuda_graph",
        "torch_compile_inductor",
        "spec_decoding_type",
        "spec_max_draft_len",
        "spec_draft_model",
        "spec_num_nextn_predict_layers",
        "spec_ngram_max_matching_size",
        "spec_eagle3_one_model",
        "spec_max_concurrency",
    ] {
        assert!(keys.contains(k) && !trt::companions(k).is_empty(), "{k}");
    }
    for (name, params) in trt::scenarios() {
        assert!(params.keys().all(|k| keys.contains(k)), "{name}");
    }
}

#[test]
fn test_default_image_pinned_trtllm() {
    let a = adapter();
    assert_eq!(a.default_image(), DEFAULT_IMAGE);
    assert_eq!(DEFAULT_IMAGE, "nvcr.io/nvidia/tensorrt-llm/release:1.2.1");
    assert!(!a.default_image().ends_with(":latest"));
    assert!(re(r"^sha256:[0-9a-f]{64}$").is_match(IMAGE_DIGEST));
    assert!(re(r"^[0-9a-f]{40}$").is_match(SOURCE_COMMIT));
    assert_eq!(schema()["tensorrt_llm_version"], "1.2.1");
    assert!(DOCS_BASE.starts_with("https://github.com/NVIDIA/TensorRT-LLM/blob/"));
}

#[test]
fn test_adapter_identity_trtllm() {
    let a = adapter();
    assert_eq!((a.id(), a.default_port()), ("trtllm", 8000));
    let formats: BTreeSet<String> = a.model_formats().into_iter().collect();
    assert_eq!(
        formats,
        ["hf".to_string(), "plain".to_string()]
            .into_iter()
            .collect()
    );
    assert!(!a.multi_model() && a.images_by_arch().is_empty());
    assert_eq!(
        a.metric_prefixes(),
        Some(vec!["trtllm_".to_string(), "[".to_string()])
    );
}

#[test]
fn test_catalog_returns_copies_trtllm() {
    let a = adapter();
    let mut first = a.param_catalog();
    first[0].label = "mutated".into();
    assert_ne!(a.param_catalog()[0].label, "mutated");
}

// --------------------------------------------------------------- against the image's click options
#[test]
fn test_cli_flags_exist_in_the_serve_command_trtllm() {
    let options = serve_option_names(&schema(), true);
    for p in adapter().param_catalog() {
        if p.flag.starts_with("--") {
            assert!(
                options.contains(&p.flag),
                "{}: {} is not an option of `trtllm-serve serve`",
                p.key,
                p.flag
            );
        }
    }
}

#[test]
fn test_help_text_fixture_agrees_with_the_click_dump_trtllm() {
    let help_text = read("trtllm", "serve_help_1.2.1.txt");
    let dumped: BTreeSet<String> = serve_option_names(&schema(), true)
        .into_iter()
        .filter(|o| o.starts_with("--"))
        .collect();
    let mut in_help: BTreeSet<String> = re(r"--[a-z0-9_]+")
        .find_iter(&help_text)
        .map(|m| m.as_str().to_string())
        .collect();
    in_help.insert("--help".into());
    let missing: Vec<&String> = dumped.difference(&in_help).collect();
    assert!(missing.is_empty(), "{missing:?}");
    assert!(help_text.contains("Usage: trtllm-serve serve [OPTIONS] MODEL"));
    let top = read("trtllm", "top_help_1.2.1.txt");
    let subs: BTreeSet<String> = re(r"(?m)^  (\w+)\s{2,}")
        .captures_iter(&top)
        .map(|c| c[1].to_string())
        .collect();
    for s in ["disaggregated", "mm_embedding_serve", "serve"] {
        assert!(subs.contains(s), "{s}");
    }
}

#[test]
fn test_every_serve_option_is_exposed_or_excluded_trtllm() {
    let a = adapter();
    let s = schema();
    let exposed: BTreeSet<String> = a
        .param_catalog()
        .into_iter()
        .filter(|p| p.flag.starts_with("--"))
        .map(|p| p.flag)
        .collect();
    let excluded: BTreeSet<String> = EXCLUDED_SERVE_FLAGS
        .iter()
        .map(|(f, _)| (*f).to_string())
        .collect();
    for p in s["serve_params"].as_array().expect("serve_params") {
        let opts: Vec<String> = p["opts"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|o| o.as_str().map(str::to_string))
            .collect();
        if p["kind"] != "Option" || opts.iter().any(|o| o == "--help") {
            continue;
        }
        let names: BTreeSet<&String> = opts.iter().filter(|o| o.starts_with("--")).collect();
        assert!(
            names
                .iter()
                .any(|n| exposed.contains(*n) || excluded.contains(*n)),
            "{names:?} is neither exposed nor excluded"
        );
    }
    let options = serve_option_names(&s, false);
    assert!(
        excluded.is_subset(&options),
        "an excluded flag no longer exists in the image"
    );
    // the flag serve() drops must not be what the catalog uses; a flag only the TensorRT backend reads is not offered
    assert!(excluded.contains("--moe_cluster_parallel_size"));
    assert!(excluded.contains("--fail_fast_on_attention_window_too_large"));
    let by = by_key(&a);
    assert!(!by.contains_key("fail_fast_on_attention_window_too_large"));
    assert_eq!(
        by["moe_cluster_parallel_size"].flag,
        "yaml:moe_cluster_parallel_size"
    );
}

#[test]
fn test_cli_defaults_and_choices_match_the_image_trtllm() {
    let s = schema();
    let click: BTreeMap<String, Value> = s["serve_params"]
        .as_array()
        .expect("serve_params")
        .iter()
        .map(|p| {
            (
                p["name"].as_str().unwrap_or_default().to_string(),
                p.clone(),
            )
        })
        .collect();
    let by = by_key(&adapter());
    for key in [
        "free_gpu_memory_fraction",
        "max_batch_size",
        "max_num_tokens",
        "max_beam_width",
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "context_parallel_size",
        "log_level",
        "num_postprocess_workers",
    ] {
        assert!(py_eq(&by[key].default, &click[key]["default"]), "{key}");
    }
    assert!(py_eq(
        &s["build_config_defaults"]["max_batch_size"],
        &by["max_batch_size"].default
    ));
    assert!(py_eq(
        &s["build_config_defaults"]["max_num_tokens"],
        &by["max_num_tokens"].default
    ));
    for key in ["log_level", "reasoning_parser", "tool_parser"] {
        let want: Vec<String> =
            serde_json::from_value(click[key]["choices"].clone()).expect("choices");
        assert_eq!(by[key].choices.clone().unwrap_or_default(), want, "{key}");
    }
    let sorted = |k: &str| {
        let mut c = by[k].choices.clone().unwrap_or_default();
        c.sort();
        serde_json::to_value(c).expect("json")
    };
    assert_eq!(s["reasoning_parsers"], sorted("reasoning_parser"));
    assert_eq!(s["tool_parsers"], sorted("tool_parser"));
    for key in ["trust_remote_code", "enable_chunked_prefill"] {
        assert!(
            click[key]["is_flag"] == Value::Bool(true) && by[key].type_ == "bool",
            "{key}"
        );
    }
}

// --------------------------------------------------------------- against the image's YAML schema
#[test]
fn test_top_level_fields_match_the_image_trtllm() {
    let s = schema();
    let mut fields: Vec<&String> = s["classes"]["TorchLlmArgs"]
        .as_object()
        .expect("class")
        .keys()
        .collect();
    fields.sort();
    assert_eq!(
        TORCH_LLM_ARGS_FIELDS.to_vec(),
        fields.iter().map(|f| f.as_str()).collect::<Vec<_>>()
    );
    assert_eq!(TORCH_LLM_ARGS_FIELDS.len(), 83);
}

#[test]
fn test_architecture_registry_matches_the_image_trtllm() {
    let want: Vec<String> =
        serde_json::from_value(schema()["architectures"].clone()).expect("architectures");
    let mut have: Vec<String> = SUPPORTED_ARCHITECTURES
        .iter()
        .map(|a| (*a).to_string())
        .collect();
    have.sort();
    assert_eq!(have, want);
    for a in ["LlamaForCausalLM", "Qwen3ForCausalLM", "GptOssForCausalLM"] {
        assert!(SUPPORTED_ARCHITECTURES.contains(&a));
    }
}

#[test]
fn test_yaml_paths_resolve_in_the_image_schema_trtllm() {
    let s = schema();
    let mut decoding: BTreeSet<String> = [
        "MTPDecodingConfig",
        "EagleDecodingConfig",
        "NGramDecodingConfig",
        "DraftTargetDecodingConfig",
        "DecodingBaseConfig",
    ]
    .iter()
    .flat_map(|c| s["classes"][c].as_object().expect("class").keys().cloned())
    .collect();
    decoding.insert("decoding_type".into()); // the dispatch key of DecodingBaseConfig.from_dict
    for p in adapter().param_catalog() {
        let Some(rest) = p.flag.strip_prefix(YAML_PREFIX) else {
            continue;
        };
        if p.flag == YAML_EXTRA_FLAG {
            continue;
        }
        let path: Vec<&str> = rest.split('.').collect();
        assert!(
            TORCH_LLM_ARGS_FIELDS.contains(&path[0]),
            "{}: top-level '{}' is not a TorchLlmArgs field",
            p.key,
            path[0]
        );
        if path[0] == "speculative_config" && path.len() == 2 {
            assert!(
                decoding.contains(path[1]),
                "{}: '{}' is not a speculative config field",
                p.key,
                path[1]
            );
            continue;
        }
        assert!(path.len() <= 2, "{}", p.key);
        let (fields, leaf) = class_for_path(&s, &path);
        assert!(
            fields.contains_key(leaf),
            "{}: '{leaf}' is not a field of the config class",
            p.key
        );
    }
}

#[test]
fn test_yaml_defaults_and_choices_match_the_image_trtllm() {
    let s = schema();
    let a = adapter();
    let special = [
        "enable_cuda_graphs",
        "enable_torch_compile",
        "speculative_config",
        "extra_llm_options",
        "kv_cache_max_gib",
        "kv_host_cache_gib",
    ];
    for p in a.param_catalog() {
        let Some(rest) = p.flag.strip_prefix(YAML_PREFIX) else {
            continue;
        };
        if p.flag == YAML_EXTRA_FLAG || special.contains(&p.key.as_str()) {
            continue;
        }
        let path: Vec<&str> = rest.split('.').collect();
        if path[0] == "speculative_config" {
            continue;
        }
        let (fields, leaf) = class_for_path(&s, &path);
        let entry = &fields[leaf];
        if p.key == "enable_iter_perf_stats" {
            // the engine's default; the console's is applied at build
            assert!(entry["default"] == Value::Bool(false) && p.default == Value::Bool(false));
            continue;
        }
        if !p.default.is_null() {
            assert!(
                py_eq(&entry["default"], &p.default),
                "{}: image default {}, catalog {}",
                p.key,
                entry["default"],
                p.default
            );
        }
        if let Some(literal) = literal_values(entry["type"].as_str().unwrap_or_default())
            && p.type_ == "enum"
        {
            for c in p.choices.clone().unwrap_or_default() {
                assert!(
                    literal.contains(&c),
                    "{}: choice {c} outside the schema Literal {literal:?}",
                    p.key
                );
            }
        }
    }
    let by = by_key(&a);
    let set = |k: &str| {
        by[k]
            .choices
            .clone()
            .unwrap_or_default()
            .into_iter()
            .collect::<BTreeSet<_>>()
    };
    let json_set =
        |v: &Value| -> BTreeSet<String> { serde_json::from_value(v.clone()).expect("list") };
    let lit_set = |v: &Value| -> BTreeSet<String> {
        literal_values(v["type"].as_str().unwrap_or_default())
            .unwrap_or_default()
            .into_iter()
            .collect()
    };
    assert!(!set("capacity_scheduler_policy").is_empty());
    assert_eq!(
        set("capacity_scheduler_policy"),
        json_set(&s["enums"]["CapacitySchedulerPolicy"])
    );
    assert_eq!(
        set("context_chunking_policy"),
        json_set(&s["enums"]["ContextChunkingPolicy"])
    );
    assert_eq!(set("sampler_type"), json_set(&s["enums"]["SamplerType"]));
    assert_eq!(
        set("moe_backend"),
        lit_set(&s["classes"]["MoeConfig"]["backend"])
    );
    assert_eq!(
        set("allreduce_strategy"),
        lit_set(&s["classes"]["TorchLlmArgs"]["allreduce_strategy"])
    );
    assert_eq!(
        set("guided_decoding_backend"),
        lit_set(&s["classes"]["TorchLlmArgs"]["guided_decoding_backend"])
    );
    assert_eq!(
        set("spec_decoding_type"),
        SPEC_TYPES.iter().map(|t| (*t).to_string()).collect()
    );
}

#[test]
fn test_cuda_graph_and_scheduler_defaults_match_the_image_trtllm() {
    let s = schema();
    let graphs = &s["classes"]["CudaGraphConfig"];
    assert!(py_eq(&graphs["max_batch_size"]["default"], &Value::from(0)));
    assert_eq!(graphs["enable_padding"]["default"], Value::Bool(false));
    let sched = &s["classes"]["SchedulerConfig"];
    assert_eq!(
        sched["capacity_scheduler_policy"]["default"],
        "GUARANTEED_NO_EVICT"
    );
    // the stock CLI restates this dynamic batch config (serve.get_llm_args); the adapter copies it
    let defaults = scheduler_defaults();
    assert_eq!(defaults[0].0, "capacity_scheduler_policy");
    assert_eq!(defaults[0].1.py_str(), "GUARANTEED_NO_EVICT");
    let dynamic: BTreeSet<String> = match &defaults[1].1 {
        ec_hub::adapters::trtllm_parse::py::PyVal::Dict(d) => {
            d.iter().map(|(k, _)| k.clone()).collect()
        }
        other => panic!("{other:?}"),
    };
    let want: BTreeSet<String> = s["classes"]["DynamicBatchConfig"]
        .as_object()
        .expect("class")
        .keys()
        .cloned()
        .collect();
    assert_eq!(dynamic, want);
}
