//! Port of backend/tests/ollama/test_probes.py: what was MEASURED on the pinned image (a bare `ollama serve`, no model,
//! no GPU, no network) and what the adapter does about it. The captured outputs are in tests/data/ollama.

use crate::a2::{oll, read};
use ec_hub::adapters::base::{EngineAdapter, LaunchContext, Obj};
use ec_hub::adapters::ollama::OllamaAdapter;
use ec_hub::services::common::py_splitlines;
use regex::Regex;

fn section(probe: &str, path: &str) -> String {
    let body = probe
        .split_once(&format!("== {path}\n"))
        .expect("section")
        .1;
    body.split_once("\n== ")
        .map_or(body, |(a, _)| a)
        .to_string()
}

#[test]
fn test_endpoints_of_a_bare_server_ollama() {
    let probe = read("ollama", "endpoints_bare_0.30.11.txt");
    assert!(section(&probe, "/api/version").contains(r#"{"version":"0.30.11"}"#));
    assert!(section(&probe, "/").contains("Ollama is running"));
    assert!(section(&probe, "/api/tags").contains(r#"{"models":[]}"#));
    assert!(section(&probe, "/api/ps").contains(r#"{"models":[]}"#));
    // data is null, not [], when the store is empty
    assert!(section(&probe, "/v1/models").contains(r#"{"object":"list","data":null}"#));
    assert!(section(&probe, "/metrics").contains("404 Not Found"));
}

/// Why the module docstring tells the core to offer only stores that have both.
#[test]
fn test_a_read_only_store_needs_blobs_and_manifests_directories_ollama() {
    let failed = read("ollama", "serve_readonly_no_blobs_0.30.11.log");
    assert!(failed.contains("mkdir /root/.ollama/models/blobs: read-only file system"));
    assert!(!failed.contains("Listening on"));
    let started = read("ollama", "serve_bare_0.30.11.log");
    // with both directories present it starts
    assert!(started.contains("Listening on") && started.contains("OLLAMA_NOPRUNE:true"));
}

/// The container needs a writable /root/.ollama (its own layer): the store itself stays read-only.
#[test]
fn test_the_server_generates_its_own_identity_under_home_not_the_store_ollama() {
    let key = Regex::new(r"AAAA[0-9A-Za-z+/]{20,}").expect("re");
    for name in [
        "serve_bare_0.30.11.log",
        "serve_readonly_no_blobs_0.30.11.log",
    ] {
        let text = read("ollama", name);
        let notice = py_splitlines(&text)
            .into_iter()
            .find(|l| l.contains("Generating new private key"))
            .expect("notice");
        assert!(
            notice.contains("/root/.ollama/") && !notice.contains("/root/.ollama/models"),
            "{name}"
        );
        assert!(!key.is_match(&text), "{name}"); // the throwaway public key was not committed
    }
}

#[test]
fn test_the_launch_never_asks_for_a_writable_store_or_the_cloud_ollama() {
    let ctx = LaunchContext {
        model: String::new(),
        model_format: "ollama".into(),
        model_path: None,
        served_name: "x".into(),
        hf_cache_container_path: "/hf".into(),
        params: Obj::new(),
        hw: oll::hw(),
    };
    let spec = OllamaAdapter::new().build_launch_ctx(&ctx).expect("spec");
    assert!(spec.env["OLLAMA_NOPRUNE"] == "1" && spec.env["OLLAMA_NO_CLOUD"] == "1");
    assert!(spec.files.is_empty() && !spec.needs_hf_cache);
}
