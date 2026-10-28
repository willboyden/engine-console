//! Port of backend/tests/llamacpp/test_parse.py: parse_metrics() and parse_startup_log().
//!
//! Captured (real, from the pinned image, no model started): startup_no_model_b10156.log, router_mode_probe_b10156.log,
//! api_key_probe_b10156.log, cors_probe_b10156.log, version_b10156.txt. Synthetic (labelled so in the file):
//! metrics_synthetic.prom, startup_synthetic.log, and the inline error lines below.

use std::collections::{BTreeMap, BTreeSet};

use crate::a2::read;
use ec_hub::adapters::base::{EngineAdapter, FloatMap};
use ec_hub::adapters::llamacpp::LlamaCppAdapter;
use ec_hub::adapters::llamacpp_parse::{parse_log_line, parse_samples, scrape_to_canonical};
use ec_hub::services::common::py_splitlines;
use serde_json::{Value, json};

const CANONICAL: &[&str] = &[
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
];

fn adapter() -> LlamaCppAdapter {
    LlamaCppAdapter::new()
}

/// A FloatMap as a dict (Python compares dicts without order).
fn dict(m: &FloatMap) -> BTreeMap<String, f64> {
    m.iter().cloned().collect()
}

fn want(pairs: &[(&str, f64)]) -> BTreeMap<String, f64> {
    pairs.iter().map(|(k, v)| ((*k).to_string(), *v)).collect()
}

fn line(l: &str) -> Option<Value> {
    parse_log_line(l)
        .unwrap_or_else(|e| panic!("{l:?} raised {e}"))
        .map(Value::Object)
}

fn startup(l: &str) -> Option<Value> {
    adapter().parse_startup_log(l).map(Value::Object)
}

// ------------------------------------------------------------------ metrics
#[test]
fn test_synthetic_scrape_maps_to_canonical_keys_llamacpp() {
    let text = read("llamacpp", "metrics_synthetic.prom");
    assert!(text.starts_with("# SYNTHETIC FIXTURE")); // the file says what it is
    let m = dict(&adapter().parse_metrics(&text));
    assert_eq!(
        m,
        want(&[
            ("requests_running", 2.0),
            ("requests_waiting", 1.0),
            ("prompt_tokens_total", 1106.0),
            ("generation_tokens_total", 7723.0),
            ("prompt_tps", 442.4),
            ("generation_tps", 126.2),
        ])
    );
    assert!(m.keys().all(|k| CANONICAL.contains(&k.as_str())));
}

/// The names below were read from the image's string table and the README at the same build tag.
#[test]
fn test_metric_names_of_the_fixture_are_the_ones_the_docs_and_image_list_llamacpp() {
    let names: BTreeSet<String> = parse_samples(&read("llamacpp", "metrics_synthetic.prom"))
        .into_iter()
        .map(|(n, _)| n)
        .collect();
    let expected: BTreeSet<String> = [
        "prompt_tokens_total",
        "prompt_seconds_total",
        "tokens_predicted_total",
        "tokens_predicted_seconds_total",
        "n_decode_total",
        "n_tokens_max",
        "prompt_tokens_seconds",
        "predicted_tokens_seconds",
        "requests_processing",
        "requests_deferred",
        "n_busy_slots_per_decode",
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect();
    assert_eq!(names, expected);
}

/// Older builds had llamacpp:kv_cache_usage_ratio / kv_cache_tokens; neither is in the pinned fixture, so the canonical
/// kv_cache_usage_pct is simply absent (never zero).
#[test]
fn test_the_pinned_build_exports_no_kv_cache_metric_llamacpp() {
    let text = read("llamacpp", "metrics_synthetic.prom");
    assert!(!text.contains("kv_cache"));
    assert!(!dict(&scrape_to_canonical(&text)).contains_key("kv_cache_usage_pct"));
}

#[test]
fn test_older_or_newer_builds_that_export_kv_usage_are_read_llamacpp() {
    let m =
        scrape_to_canonical("llamacpp:kv_cache_usage_ratio 0.25\nllamacpp:kv_cache_tokens 1024\n");
    assert_eq!(dict(&m), want(&[("kv_cache_usage_pct", 25.0)]));
}

#[test]
fn test_missing_and_foreign_metrics_are_omitted_llamacpp() {
    let a = adapter();
    assert!(a.parse_metrics("").is_empty());
    assert!(
        a.parse_metrics("process_cpu_seconds_total 3\nvllm:num_requests_running 2\n")
            .is_empty()
    );
    assert_eq!(
        dict(&a.parse_metrics("llamacpp:requests_deferred 7\n")),
        want(&[("requests_waiting", 7.0)])
    );
    // zero is kept, absent is not
    assert_eq!(
        dict(&a.parse_metrics("llamacpp:requests_processing 0\n")),
        want(&[("requests_running", 0.0)])
    );
    let help_only =
        "# HELP llamacpp:requests_processing x\n# TYPE llamacpp:requests_processing gauge\n";
    assert!(a.parse_metrics(help_only).is_empty());
}

#[test]
fn test_tolerates_total_suffix_variants_and_labels_llamacpp() {
    let text = [
        "llamacpp:tokens_predicted 10",
        "llamacpp:prompt_tokens{model=\"a\"} 5",
        "llamacpp:prompt_tokens{model=\"b\"} 7",
        "llamacpp:requests_processing{slot=\"0\"} 1",
        "llamacpp:requests_processing{slot=\"1\"} 2",
    ]
    .join("\n");
    assert_eq!(
        dict(&scrape_to_canonical(&text)),
        want(&[
            ("generation_tokens_total", 10.0),
            ("prompt_tokens_total", 12.0),
            ("requests_running", 3.0)
        ])
    );
}

#[test]
fn test_parse_samples_edge_cases_llamacpp() {
    let s = parse_samples(
        "# HELP x\nllamacpp:a 1.5\nbad line\nllamacpp:b 2 1712345\nllamacpp:c NaN\nllamacpp:d{l=\"q\\\"x\"} 4\nllamacpp:e abc\n",
    );
    assert_eq!(
        s,
        vec![
            ("a".to_string(), 1.5),
            ("b".to_string(), 2.0),
            ("d".to_string(), 4.0)
        ]
    );
    assert!(!s.iter().any(|(_, v)| v.is_nan()));
}

/// No label tells two instances apart, so nothing may be remembered between scrapes. (Python also checks that the
/// instance holds only `_catalog` and `_by_key`; in Rust `parse_metrics(&self)` cannot change the adapter.)
#[test]
fn test_parser_keeps_no_state_llamacpp() {
    let a = adapter();
    let first = dict(&a.parse_metrics("llamacpp:tokens_predicted_total 100\n"));
    let second = dict(&a.parse_metrics("llamacpp:tokens_predicted_total 50\n"));
    assert!(
        first == want(&[("generation_tokens_total", 100.0)])
            && second == want(&[("generation_tokens_total", 50.0)])
    );
    assert!(
        !first
            .keys()
            .chain(second.keys())
            .any(|k| k.ends_with("_tps"))
    );
}

// ------------------------------------------------------------------ startup log: captured lines
#[test]
fn test_captured_no_model_failure_llamacpp() {
    let text = read("llamacpp", "startup_no_model_b10156.log");
    let lines = py_splitlines(&text);
    let parsed: Vec<Option<Value>> = lines.iter().map(|l| startup(l)).collect();
    assert!(parsed.len() == lines.len() && lines.len() == 19);
    assert!(parsed[0].is_none()); // the LLAMA_ARG_HOST warning
    assert!(parsed[1].is_none()); // "verbosity = 3"
    assert!(parsed[2..7].iter().all(Option::is_none)); // the CORS warning block
    assert_eq!(
        parsed[7],
        Some(json!({"phase": "loading_weights", "pct": 0.0}))
    ); // "loading model '...'"
    let errors: Vec<&str> = parsed
        .iter()
        .flatten()
        .filter_map(|p| p.get("error").and_then(Value::as_str))
        .collect();
    assert!(
        errors[0] == "model_file_missing"
            && errors
                .iter()
                .filter(|e| **e == "model_file_missing")
                .count()
                == 2
    );
    let set: BTreeSet<&str> = errors.iter().copied().collect();
    assert!(
        set == BTreeSet::from(["model_file_missing", "model_load_failed"])
            && errors.last() == Some(&"model_load_failed")
    );
    let missing = parsed
        .iter()
        .flatten()
        .find(|p| p.get("error") == Some(&json!("model_file_missing")))
        .expect("missing");
    let msg = missing["message"].as_str().expect("message");
    assert!(msg.contains("does-not-exist.gguf") && msg.contains("No such file or directory"));
    let last = parsed.iter().rev().flatten().next().expect("last");
    assert!(
        last["message"]
            .as_str()
            .expect("message")
            .contains("exiting due to model loading error")
    );
    assert!(parsed[parsed.len() - 2].is_none()); // "cleaning up before exit"
}

#[test]
fn test_captured_router_mode_start_llamacpp() {
    let text = read("llamacpp", "router_mode_probe_b10156.log");
    let log = text.split_once("== LOG\n").expect("== LOG").1;
    let parsed: Vec<Option<Value>> = py_splitlines(log).iter().map(|l| startup(l)).collect();
    assert!(parsed.contains(&Some(json!({"listening": "http://127.0.0.1:8080"}))));
    assert_eq!(parsed.iter().flatten().count(), 1); // nothing else in it is progress
}

#[test]
fn test_real_lines_one_by_one_llamacpp() {
    assert_eq!(
        line("0.00.007.493 I srv    load_model: loading model '/models/x.gguf'"),
        Some(json!({"phase": "loading_weights", "pct": 0.0}))
    );
    assert_eq!(
        line("0.00.008.904 I srv  llama_server: listening on http://127.0.0.1:8080"),
        Some(json!({"listening": "http://127.0.0.1:8080"}))
    );
    assert_eq!(
        line("0.00.006.233 W srv  llama_server: -----------------"),
        None
    );
    assert_eq!(
        line(
            "warn: LLAMA_ARG_HOST environment variable is set, but will be overwritten by command line argument --host"
        ),
        None
    );
    assert!(line("").is_none() && line("   ").is_none());
}

// ------------------------------------------------------------------ startup log: synthetic success path
#[test]
fn test_synthetic_success_path_llamacpp() {
    let text = read("llamacpp", "startup_synthetic.log");
    assert!(text.starts_with("# SYNTHETIC FIXTURE"));
    let got: Vec<Value> = py_splitlines(&text)
        .iter()
        .filter_map(|l| startup(l))
        .collect();
    for expected in [
        json!({"phase": "loading_weights", "pct": 0.0}),
        json!({"n_slots": 4, "kv_unified": true}), // auto parallel
        json!({"gpus_found": 1}),
        json!({"gpu": {"index": 0, "name": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition", "compute_capability": "12.0",
                       "vram_gib": 95.59}}),
        json!({"fit_projected_mib": 21000, "fit_free_mib": 90000}),
        json!({"fit_ctx_reduced": {"from": 262_144, "to": 65536}}),
        json!({"phase": "loading_weights", "pct": 0.0, "tensors": 291}),
        json!({"n_ctx_train": 262_144}),
        json!({"phase": "loading_weights", "gpu_layers": 49, "total_layers": 49}),
        json!({"buffer": "CUDA0", "model_buffer_gib": 17.0}),
        json!({"buffer": "CPU_Mapped", "model_buffer_gib": 0.308}),
        json!({"phase": "loading_weights", "pct": 98.0}), // the row of progress dots
        json!({"n_ctx": 65536}),
        json!({"kv_buffer": {"device": "CUDA0", "gib": 3.0}}),
        json!({"kv_cache_gib": 3.0, "kv_cache_tokens": 65536}),
        json!({"phase": "warming_up"}),
        json!({"n_slots": 4, "n_ctx_slot": 65536}),
        json!({"listening": "http://0.0.0.0:8080"}),
    ] {
        assert!(got.contains(&expected), "{expected} not in {got:?}");
    }
    assert_eq!(got.last(), Some(&json!({"phase": "ready", "pct": 100.0}))); // all slots are idle
    let phases: Vec<&str> = got
        .iter()
        .filter_map(|p| p.get("phase").and_then(Value::as_str))
        .collect();
    let pos = |name: &str| phases.iter().position(|p| *p == name).expect("phase");
    assert!(pos("warming_up") < pos("ready") && phases[..3].contains(&"loading_weights"));
}

/// 16 parametrized rows: (line, expected).
#[test]
fn test_synthetic_single_lines_llamacpp() {
    let dots120 = ".".repeat(120);
    let cases: Vec<(&str, Option<Value>)> = vec![
        (
            "0.01.0 I load_tensors: offloaded 0/33 layers to GPU",
            Some(
                json!({"phase": "loading_weights", "gpu_layers": 0, "total_layers": 33, "warning": "cpu_only"}),
            ),
        ),
        (
            "load_tensors: offloaded 20/33 layers to GPU",
            Some(json!({"phase": "loading_weights", "gpu_layers": 20, "total_layers": 33})),
        ),
        (
            "main: model loaded",
            Some(json!({"phase": "ready", "pct": 100.0})),
        ),
        (
            "srv  update_slots: all slots are idle",
            Some(json!({"phase": "ready", "pct": 100.0})),
        ),
        (
            "srv  load_model: server is entering sleeping state",
            Some(json!({"state": "sleeping"})),
        ),
        (
            "srv  load_model: server is exiting sleeping state",
            Some(json!({"state": "awake"})),
        ),
        ("llama_context: n_seq_max     = 4", None),
        (
            "common_init_from_params: warming up the model with an empty run - please wait ... (--no-warmup to disable)",
            Some(json!({"phase": "warming_up"})),
        ),
        (
            "0.00.1 I cmn  common_param: build 10156 (91f8c9c5f) with GNU 14.2.0 for Linux x86_64",
            Some(json!({"build": "10156", "commit": "91f8c9c5f"})),
        ),
        (
            "........................................",
            Some(json!({"phase": "loading_weights", "pct": 40.0})),
        ),
        (
            &dots120,
            Some(json!({"phase": "loading_weights", "pct": 100.0})),
        ),
        ("..", None),
        (
            "warning: failed to mlock 8192-byte buffer (after previously locking 0 bytes): Cannot allocate memory",
            None,
        ),
        (
            "common_fit_params: failed to fit params to free device memory: model does not fit",
            Some(json!({"warning": "fit_failed", "message": "model does not fit"})),
        ),
        (
            "common_fit_params: context size reduced from 8192 to 4096 -> need 100 MiB less memory in total",
            Some(json!({"fit_ctx_reduced": {"from": 8192, "to": 4096}})),
        ),
        (
            "\x1b[0;32mmain: model loaded\x1b[0m",
            Some(json!({"phase": "ready", "pct": 100.0})),
        ),
    ];
    for (l, expected) in cases {
        let got = line(l);
        if l.contains("failed to mlock") {
            let g = got.expect("mlock");
            assert!(
                g["warning"] == "mlock_failed"
                    && g["message"]
                        .as_str()
                        .expect("m")
                        .contains("Cannot allocate memory")
            );
        } else {
            assert_eq!(got, expected, "{l:?}");
        }
    }
}

/// The last 40 characters of a line (`line[-40:]`).
fn tail40(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    chars[chars.len().saturating_sub(40)..].iter().collect()
}

/// 10 parametrized rows: (line, code).
#[test]
fn test_synthetic_error_lines_llamacpp() {
    for (l, code) in [
        (
            "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 98304.00 MiB on device 0: cudaMalloc failed: out of memory",
            "cuda_oom",
        ),
        (
            "alloc_tensor_range: failed to allocate CUDA0 buffer of size 105226698752",
            "cuda_oom",
        ),
        ("CUDA error: out of memory", "cuda_oom"),
        (
            "llama_context: V cache quantization requires flash_attn",
            "kv_quant_needs_flash_attn",
        ),
        (
            "llama_context: SPLIT_MODE_TENSOR requires flash_attn to be enabled",
            "split_tensor_needs_flash_attn",
        ),
        (
            "srv  operator(): couldn't bind HTTP server socket, hostname: 0.0.0.0, port: 8080",
            "port_in_use",
        ),
        (
            "llama_model_load: error loading model: unsupported model architecture: 'newarch'",
            "unsupported_architecture",
        ),
        (
            "llama_init_from_model: failed to create llama_context from model",
            "context_create_failed",
        ),
        (
            "srv  llama_server: exiting due to HTTP server error",
            "http_server_failed",
        ),
        (
            "srv  load_model: failed to load model, '/models/x.gguf'",
            "model_load_failed",
        ),
    ] {
        let got = line(&format!("0.00.9 E {l}")).unwrap_or_else(|| panic!("{l}"));
        assert!(
            got["error"] == code && got["message"].as_str().expect("m").contains(&tail40(l)),
            "{l}"
        );
    }
}

#[test]
fn test_errors_win_over_progress_in_the_same_line_llamacpp() {
    let l = "load_tensors: offloaded 33/33 layers to GPU cudaMalloc failed: out of memory";
    assert_eq!(line(l), Some(json!({"error": "cuda_oom", "message": l})));
}

#[test]
fn test_long_messages_are_cut_and_nothing_else_leaks_llamacpp() {
    let got = line(&format!("E failed to load model {}", "x".repeat(1000))).expect("error");
    assert_eq!(got["message"].as_str().expect("m").chars().count(), 300);
}

#[test]
fn test_no_pattern_matches_ordinary_chatter_llamacpp() {
    for l in [
        "srv  log_server_r: request: GET /health 127.0.0.1 200",
        "slot launch_slot_: id  0 | task 0 | processing task",
        "slot update_slots: id  0 | task 0 | prompt processing progress, n_past = 512, n_tokens = 512, progress = 0.5",
        "print_timings: prompt eval time = 12.0 ms / 10 tokens",
        "yyy", // re.sub("x", "y", "xxx")
    ] {
        assert_eq!(line(l), None, "{l}");
    }
}
