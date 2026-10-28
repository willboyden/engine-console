//! Port of backend/tests/ollama/test_parse.py: parse_metrics() (Ollama has none) and parse_startup_log().
//!
//! Captured (real, from the pinned image, a bare server with no model): serve_bare_0.30.11.log,
//! serve_readonly_no_blobs_0.30.11.log, endpoints_bare_0.30.11.txt. Synthetic (inline, labelled): the runner lines,
//! whose wording is the llama.cpp one and was not seen in a live Ollama run.

use crate::a2::{oll, read};
use ec_hub::adapters::base::{EngineAdapter, Obj};
use ec_hub::adapters::ollama::{OllamaAdapter, parse_log_line};
use ec_hub::adapters::trtllm_parse::py;
use ec_hub::services::common::py_splitlines;
use serde_json::{Value, json};

fn line(l: &str) -> Option<Value> {
    parse_log_line(l)
        .unwrap_or_else(|e| panic!("{l:?} raised {e}"))
        .map(Value::Object)
}

fn startup(l: &str) -> Option<Value> {
    OllamaAdapter::new().parse_startup_log(l).map(Value::Object)
}

#[test]
fn test_ollama_exports_no_prometheus_metrics_ollama() {
    let a = OllamaAdapter::new();
    assert!(a.parse_metrics("").is_empty());
    let foreign =
        "# HELP x y\nvllm:num_requests_running 3\nllamacpp:requests_processing 1\nollama_x 1\n";
    assert!(a.parse_metrics(foreign).is_empty());
    assert!(a.parse_metrics(r#"{"version":"0.30.11"}"#).is_empty()); // what the metrics_path really returns
    assert_eq!(a.metric_prefixes(), Some(Vec::new()));
}

#[test]
fn test_the_metrics_path_is_a_cheap_read_only_call_the_node_allows_ollama() {
    let probe = read("ollama", "endpoints_bare_0.30.11.txt");
    assert!(probe.contains("== /metrics\nHTTP/1.0 404 Not Found")); // why there is nothing to scrape
    let version = probe.split_once("== /api/version").expect("/api/version").1;
    assert!(
        version
            .split_once("==")
            .map_or(version, |(a, _)| a)
            .contains("HTTP/1.0 200 OK")
    );
    assert!(probe.contains(r#"{"version":"0.30.11"}"#));
    let spec = OllamaAdapter::new()
        .build_launch("", &Obj::new(), &oll::hw(), "x", "/hf")
        .expect("spec");
    assert!(
        spec.metrics_path == spec.health_path
            && spec.health_path == "/api/version"
            && spec.container_port == 11434
    );
    assert!(spec.metrics_path.starts_with("/api/")); // FLEET-PROTOCOL: `/api/` (Ollama) is allowed
}

// ------------------------------------------------------------------ startup log: captured
#[test]
fn test_captured_bare_server_start_ollama() {
    let text = read("ollama", "serve_bare_0.30.11.log");
    let parsed: Vec<(&str, Option<Value>)> = py_splitlines(&text)
        .into_iter()
        .map(|l| (l, startup(l)))
        .collect();
    let got: Vec<&Value> = parsed.iter().filter_map(|(_, p)| p.as_ref()).collect();
    for expected in [
        json!({"cloud_disabled": true}),
        json!({"phase": "ready", "pct": 100.0, "listening": "127.0.0.1:11434", "version": "0.30.11"}),
        json!({"gpu": {"id": "cpu", "library": "cpu", "total_gib": 246.8, "available_gib": 246.7}, "warning": "cpu_only"}),
        json!({"default_num_ctx": 4096, "total_vram_gib": 0.0}),
    ] {
        assert!(got.contains(&&expected), "{expected}");
    }
    assert_eq!(got.len(), 4);
    // the request log, the key-generation notice and the server config carry no progress
    for (l, p) in &parsed {
        if l.starts_with("[GIN]")
            || l.contains("server config")
            || l.contains("Generating new private key")
        {
            assert!(p.is_none(), "{l}");
        }
    }
}

#[test]
fn test_captured_read_only_store_failure_ollama() {
    let text = read("ollama", "serve_readonly_no_blobs_0.30.11.log");
    let got: Vec<Value> = py_splitlines(&text)
        .into_iter()
        .filter(|l| !py::strip(l).is_empty())
        .filter_map(startup)
        .collect();
    let last = got.last().expect("a parsed line");
    assert_eq!(last["error"], "models_dir_read_only");
    let msg = last["message"].as_str().expect("message");
    assert!(
        msg.contains("mkdir /root/.ollama/models/blobs") && msg.contains("read-only file system")
    );
    assert!(!got.iter().any(|p| p.get("phase") == Some(&json!("ready")))); // it never started listening
}

#[test]
fn test_captured_lines_one_by_one_ollama() {
    assert_eq!(
        line(
            r#"time=2026-10-04T13:53:10.815Z level=INFO source=routes.go:2023 msg="Listening on 127.0.0.1:11434 (version 0.30.11)""#
        ),
        Some(
            json!({"phase": "ready", "pct": 100.0, "listening": "127.0.0.1:11434", "version": "0.30.11"})
        )
    );
    assert_eq!(
        line(
            r#"time=2026-10-04T13:53:11.082Z level=INFO source=routes.go:2073 msg="vram-based default context" total_vram="0 B" default_num_ctx=4096"#
        ),
        Some(json!({"default_num_ctx": 4096, "total_vram_gib": 0.0}))
    );
    assert_eq!(
        line(
            r#"time=2026-10-04T13:53:10.816Z level=INFO source=runner.go:60 msg="discovering available GPUs...""#
        ),
        None
    );
    assert!(line("").is_none() && line("   ").is_none());
}

// ------------------------------------------------------------------ startup log: synthetic
/// 6 parametrized rows: (line, expected).
#[test]
fn test_synthetic_lines_ollama() {
    for (l, expected) in [
        // a GPU discovery line in the captured shape, with GPU values made up
        (
            r#"time=2026-10-04T13:53:11.082Z level=INFO source=types.go:50 msg="inference compute" id=GPU-aaaa library=CUDA compute=12.0 name=CUDA0 total="95.0 GiB" available="94.0 GiB""#,
            json!({"gpu": {"id": "GPU-aaaa", "library": "CUDA", "total_gib": 95.0, "available_gib": 94.0}}),
        ),
        (
            r#"time=2026-10-04T13:53:11.082Z level=INFO source=routes.go:2073 msg="vram-based default context" total_vram="191.2 GiB" default_num_ctx=262144"#,
            json!({"default_num_ctx": 262_144, "total_vram_gib": 191.2}),
        ),
        (
            r#"time=x level=INFO source=routes.go:1 msg="vram-based default context" total_vram="512 MiB" default_num_ctx=4096"#,
            json!({"default_num_ctx": 4096, "total_vram_gib": 0.5}),
        ),
        (
            "load_tensors: offloaded 33/33 layers to GPU",
            json!({"phase": "loading_weights", "gpu_layers": 33, "total_layers": 33}),
        ),
        (
            r#"time=x level=INFO msg="llama runner started in 4.52 seconds""#,
            json!({"model_load_seconds": 4.52}),
        ),
        (
            r#"time=x level=INFO msg="Ollama cloud disabled: false""#,
            json!({"cloud_disabled": false}),
        ),
    ] {
        assert_eq!(line(l), Some(expected), "{l}");
    }
}

/// 6 parametrized rows: (line, code).
#[test]
fn test_synthetic_errors_ollama() {
    for (l, code) in [
        (
            r#"time=x level=ERROR source=sched.go:1 msg="cudaMalloc failed: out of memory""#,
            "cuda_oom",
        ),
        (
            r#"time=x level=ERROR msg="llama runner terminated: CUDA error: out of memory""#,
            "cuda_oom",
        ),
        (
            "Error: listen tcp 0.0.0.0:11434: bind: address already in use",
            "port_in_use",
        ),
        (
            "Error: mkdir /root/.ollama/models/blobs: read-only file system: ensure path elements are traversable",
            "models_dir_read_only",
        ),
        (
            r#"time=x level=ERROR source=routes.go:9 msg="something broke""#,
            "engine_error",
        ),
        ("Error: unexpected failure", "engine_error"),
    ] {
        let got = line(l).unwrap_or_else(|| panic!("{l}"));
        assert!(
            got["error"] == code && got["message"].as_str().is_some_and(|m| !m.is_empty()),
            "{l}"
        );
    }
}

#[test]
fn test_long_messages_are_cut_ollama() {
    let got = line(&format!("Error: {}", "x".repeat(1000))).expect("error");
    assert_eq!(
        got["message"].as_str().expect("message").chars().count(),
        300
    );
}

#[test]
fn test_ordinary_info_lines_carry_no_progress_ollama() {
    for l in [
        r#"time=x level=INFO source=model_list_cache.go:111 msg="model list cache hydration complete" models=0 failures=0"#,
        r#"time=x level=INFO msg="server config" env="map[OLLAMA_DEBUG:INFO]""#,
        "[GIN] 2026/10/04 - 13:53:18 | 200 | 139.641µs | 127.0.0.1 | GET \"/api/version\"",
        "plain text",
        r#"time=x level=WARN msg="slow""#,
    ] {
        assert_eq!(line(l), None, "{l}");
    }
}
