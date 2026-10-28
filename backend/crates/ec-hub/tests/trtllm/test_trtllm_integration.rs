//! Port of backend/tests/trtllm/test_trtllm_integration.py: how the adapter meets the rest of the console, the registry
//! and the node's metric filter.

use std::sync::Arc;

use crate::a2::{node_filter_metrics, read, read_json};
use ec_hub::adapters::base::EngineAdapter;
use ec_hub::adapters::trtllm::{self, TrtllmAdapter};
use ec_hub::adapters::trtllm_parse::py::PyVal;
use ec_hub::adapters::{AdapterRegistry, load_default_adapters};

/// `load_default_adapters(("engine_console.adapters.trtllm",))`: Rust has no per-module loading, so the registry is
/// built from that one module's `adapter()`; the default registry is checked to carry it too.
#[test]
fn test_registry_finds_and_instantiates_the_adapter_trtllm() {
    let reg = AdapterRegistry::new([trtllm::adapter()]);
    let adapter = reg.get("trtllm").expect("trtllm");
    assert_eq!(adapter.display_name(), "TensorRT-LLM");
    assert!(reg.contains("trtllm"));
    assert_eq!(
        reg.list()
            .iter()
            .map(|a| a.id().to_string())
            .collect::<Vec<_>>(),
        ["trtllm"]
    );
    let default = load_default_adapters();
    assert_eq!(
        default
            .get("trtllm")
            .expect("default registry")
            .display_name(),
        "TensorRT-LLM"
    );
}

#[test]
fn test_adapter_implements_the_whole_port_trtllm() {
    // No abstract methods is the compiler's job in Rust: the trait object exists.
    let adapter: Arc<dyn EngineAdapter> = Arc::new(TrtllmAdapter::new());
    assert_eq!(adapter.image_for("x86_64"), adapter.default_image());
    assert_eq!(adapter.image_for("aarch64"), adapter.default_image());
}

/// The node keeps a line only if its metric name starts with one of `metric_prefixes`. `/metrics` is a single JSON line
/// without metric names; the filter reads the text before the first `{` as the name, which is `[`.
#[test]
fn test_the_nodes_metric_filter_keeps_what_the_adapter_parses_trtllm() {
    let adapter = TrtllmAdapter::new();
    let prefixes = adapter.metric_prefixes().expect("metric_prefixes");
    let plain = read_json("trtllm", "iter_stats_1.2.1.json")["plain"].clone();
    let body = PyVal::from_json(&serde_json::Value::Array(vec![plain])).dumps_compact(); // what the server sends
    assert!(!body.contains('\n'));
    let kept = node_filter_metrics(&body, &prefixes);
    assert_eq!(kept.trim(), body);
    let parsed = adapter.parse_metrics(&kept);
    assert_eq!(
        parsed
            .iter()
            .find(|(k, _)| k == "requests_running")
            .map(|(_, v)| *v),
        Some(3.0)
    );
    // a filter without "[" would silently drop the only data this engine's /metrics carries
    assert_eq!(node_filter_metrics(&body, &["trtllm_".to_string()]), "");
    let prom = read("trtllm", "collector_1.2.1.prom");
    let kept_prom = node_filter_metrics(
        &format!("# HELP python_info x\npython_info 1\n{prom}"),
        &prefixes,
    );
    assert!(
        !kept_prom.contains("python_info")
            && kept_prom.contains("trtllm_e2e_request_latency_seconds_bucket")
    );
    let keys = |m: &ec_hub::adapters::base::FloatMap| {
        let mut k: Vec<String> = m.iter().map(|(k, _)| k.clone()).collect();
        k.sort();
        k
    };
    assert_eq!(
        keys(&adapter.parse_metrics(&kept_prom)),
        keys(&adapter.parse_metrics(&prom))
    );
}

#[test]
fn test_prefixes_are_short_strings_trtllm() {
    let prefixes = TrtllmAdapter::new()
        .metric_prefixes()
        .expect("metric_prefixes");
    assert!(
        !prefixes.is_empty()
            && prefixes
                .iter()
                .all(|p| (1..=64).contains(&p.chars().count()))
    );
}
