//! Port of backend/tests/llamacpp/test_probes.py: the claims the adapter makes about llama-server's behaviour that were
//! MEASURED on the pinned image (router mode, no model, no GPU, no network): API-key handling and CORS. The captured
//! outputs are tests/data/llamacpp/*_probe_b10156.log.

use std::collections::BTreeMap;

use crate::a2::{llama, p, read};
use ec_hub::adapters::base::{EngineAdapter, LaunchContext, Obj};
use ec_hub::adapters::llamacpp::LlamaCppAdapter;
use regex::Regex;
use serde_json::json;

fn statuses(block: &str) -> BTreeMap<String, i64> {
    Regex::new(r"== (\S+) -> (\d+)")
        .expect("re")
        .captures_iter(block)
        .map(|c| (c[1].to_string(), c[2].parse().expect("status")))
        .collect()
}

#[test]
fn test_api_key_protects_metrics_but_not_health_or_models_llamacpp() {
    let text = read("llamacpp", "api_key_probe_b10156.log");
    let (without, rest) = text.split_once("== with key:").expect("with key");
    let (with_key, log) = rest.split_once("== LOG").expect("LOG");
    let want: BTreeMap<String, i64> = [
        ("/health", 200),
        ("/v1/health", 200),
        ("/v1/models", 200),
        ("/models", 200),
        ("/metrics", 401),
        ("/props", 401),
        ("/slots", 401),
        ("/api/tags", 401),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();
    assert_eq!(statuses(without), want);
    let got = statuses(with_key);
    assert!(got["/health"] == 200 && got["/v1/models"] == 200 && got["/metrics"] != 401); // the key was accepted
    assert_eq!(log.matches("unauthorized: Invalid API Key").count(), 4); // one per refused path
    // what the adapter does with that knowledge
    let a = LlamaCppAdapter::new();
    assert_eq!(a.metric_prefixes(), Some(vec!["llamacpp:".to_string()]));
    let codes: Vec<String> = a
        .compatibility(&llama::dense(), &p(json!({"api_key": "k"})), &llama::hw())
        .into_iter()
        .map(|c| c.code)
        .collect();
    assert!(codes.iter().any(|c| c == "api_key_blocks_metrics"));
}

#[test]
fn test_default_cors_reflects_any_origin_and_localhost_does_not_llamacpp() {
    let text = read("llamacpp", "cors_probe_b10156.log");
    let (local, default) = text.split_once("== default CORS").expect("default CORS");
    assert!(!local.contains("Access-Control-Allow-Origin: http://evil.example")); // --cors-origins localhost
    assert!(local.contains("Access-Control-Allow-Origin: http://localhost:3000"));
    assert!(local.contains("Access-Control-Allow-Origin: http://127.0.0.1:9000"));
    assert!(default.contains("Access-Control-Allow-Origin: http://evil.example")); // the engine's default
    // how often the engine's own "CORS is set to allow all origins" warning was printed: only without the option
    let counts = Regex::new(r"== LOG1 \(cors localhost\)\n(\d+)\n== LOG2 \(default\)\n(\d+)")
        .expect("re")
        .captures(&text)
        .expect("counts");
    assert_eq!((&counts[1], &counts[2]), ("0", "1"));
}

#[test]
fn test_the_launch_applies_the_measured_hardening_llamacpp() {
    let ctx = LaunchContext {
        model: "m".into(),
        model_format: "gguf".into(),
        model_path: Some("/models/a.gguf".into()),
        served_name: "a".into(),
        hf_cache_container_path: "/hf".into(),
        params: Obj::new(),
        hw: llama::hw(),
    };
    let argv = LlamaCppAdapter::new()
        .build_launch_ctx(&ctx)
        .expect("spec")
        .argv;
    let i = argv
        .iter()
        .position(|a| a == "--cors-origins")
        .expect("--cors-origins");
    assert_eq!(argv[i + 1], "localhost");
}

/// Why the metrics fixture is synthetic: in router mode /metrics asks for a model name (400), and single-model mode
/// needs a loaded model.
#[test]
fn test_router_mode_has_no_metrics_without_a_model_so_no_live_scrape_exists_llamacpp() {
    let probe = read("llamacpp", "router_mode_probe_b10156.log");
    assert!(
        probe.contains("HTTP/1.1 400 Bad Request")
            && probe.contains("model name is missing from the request")
    );
    assert!(probe.contains(r#"{"status":"ok"}"#)); // /health in router mode
}
