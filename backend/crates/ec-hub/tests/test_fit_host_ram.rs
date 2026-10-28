//! Port of backend/tests/core/test_fit_host_ram.py: host-RAM fit verdicts (pure) and the fit endpoint's host-aware
//! wiring (through the fake fleet).

#![allow(clippy::unwrap_used, clippy::float_cmp)]

#[path = "common/kit.rs"]
mod kit;

use std::sync::Arc;

use ec_hub::adapters::base::EngineAdapter;
use ec_hub::domain::models::{HostMemInfo, SettingsPatch};
use ec_hub::services::fit::estimate_host_ram;
use ec_hub::services::hf::build_model_info;
use kit::FakeMultiAdapter;
use kit::fakefleet::{FleetEnv, GpuFree, HelloOpts, qwen_cfg};
use serde_json::{Value, json};

const P: &str = "/api/v1";

fn mem(total: f64, avail: f64, shmem: f64) -> HostMemInfo {
    HostMemInfo {
        total_gib: total,
        used_gib: total - avail,
        available_gib: avail,
        free_gib: 1.0,
        cached_gib: 1.0,
        shmem_gib: shmem,
        swap_total_gib: 0.0,
        swap_used_gib: 0.0,
        updated_at: 0.0,
    }
}

#[test]
fn test_host_ram_thresholds() {
    let host = mem(200.0, 100.0, 0.0);
    for (extra, verdict) in [
        (76.0, "ok"),
        (76.5, "tight"),
        (96.0, "tight"),
        (96.5, "wont_fit"),
    ] {
        let r = estimate_host_ram(&vec![("pool".into(), extra)], Some(&host));
        assert!(
            r.verdict == verdict && r.available_gib == Some(100.0) && r.total_gib == Some(200.0)
        );
        assert!((r.needed_gib - (extra + 4.0)).abs() < 1e-9);
        assert_eq!(
            Value::Object(r.breakdown.clone()),
            json!({"pool": extra, "engine_process": 4.0})
        );
    }
}

#[test]
fn test_host_ram_unknown_never_fabricated() {
    let r = estimate_host_ram(
        &vec![
            ("hicache".into(), f64::NAN),
            ("cpu_weight_offload".into(), 8.0),
        ],
        Some(&mem(200.0, 100.0, 0.0)),
    );
    assert!(r.verdict == "unknown" && r.needed_gib == 12.0 && !r.breakdown.contains_key("hicache"));
    assert!(
        r.notes
            .iter()
            .any(|n| n.contains("could not be computed") && n.contains("hicache"))
    );
    let none = estimate_host_ram(&Vec::new(), None);
    assert!(none.verdict == "unknown" && none.available_gib.is_none());
    assert!(none.notes.iter().any(|n| n.contains("could not be read")));
}

#[test]
fn test_host_ram_notes_explain_page_cache_and_shared_memory() {
    let r = estimate_host_ram(&Vec::new(), Some(&mem(100.0, 50.0, 40.0)));
    assert!(
        r.verdict == "ok"
            && r.notes
                .iter()
                .any(|n| n.contains("page cache") && n.contains("unreclaimable"))
    );
    assert!(
        r.notes
            .iter()
            .any(|n| n.contains("shared memory") && n.contains("cannot be reclaimed"))
    );
}

fn seed_hub(env: &FleetEnv) {
    env.hub.add(
        "Qwen/Qwen3-32B",
        &[("config.json", b"{}")],
        Some(json!({"num_hidden_layers": 2, "hidden_size": 64, "num_attention_heads": 2, "num_key_value_heads": 1})),
        &[("BF16", 1_000_000)],
    );
}

fn with_info(env: &FleetEnv) {
    env.library
        .set_info_builder(Arc::new(|h| build_model_info(h, None).unwrap()));
}

#[tokio::test]
async fn test_fit_endpoint_includes_host_ram_of_the_chosen_host() {
    let mut env = FleetEnv::new();
    seed_hub(&env);
    let body =
        json!({"engine": "fake", "repo_id": "Qwen/Qwen3-32B", "params": {"max_model_len": 512}});
    let h = env
        .post(&format!("{P}/fit"))
        .json(&body)
        .send()
        .await
        .json()["host_ram"]
        .clone();
    assert!(h["verdict"] == "ok" && h["available_gib"] == 54.0, "{h}");
    assert_eq!(h["breakdown"], json!({"engine_process": 4.0}));
    let small = env.add_host(
        "small",
        "Small box",
        HelloOpts {
            ram_gib: 16.0,
            ..HelloOpts::default()
        },
    );
    small.set_mem(&[("total", 16.0), ("available", 5.0)]);
    let mut b2 = body.clone();
    b2["host_id"] = json!("small");
    let r = env.post(&format!("{P}/fit")).json(&b2).send().await.json();
    assert!(
        r["host_ram"]["available_gib"] == 5.0 && r["host_ram"]["verdict"] == "ok",
        "{r}"
    );
    small.set_mem(&[("available", 3.0)]);
    let r = env.post(&format!("{P}/fit")).json(&b2).send().await.json();
    assert_eq!(r["host_ram"]["verdict"], "wont_fit");
    env.node.set_state("down", None);
    let r2 = env.post(&format!("{P}/fit")).json(&body).send().await;
    assert!(
        r2.status == 503 && r2.json()["code"] == "node_unavailable",
        "{}",
        r2.text()
    );
}

#[tokio::test]
async fn test_fit_uses_the_hosts_gpus_and_a_library_copy_when_a_model_is_given() {
    let mut env = FleetEnv::new();
    with_info(&env);
    env.seed_local_model("Qwen/Qwen3-32B", None, 65_530_000_000, "local");
    env.node.set_gpu_free(GpuFree::Each(vec![90.0, 12.0]));
    let body = json!({"engine": "fake", "model": {"format": "hf", "repo_id": "Qwen/Qwen3-32B"},
                      "params": {"max_model_len": 4096}, "gpu_ids": [1]});
    let r = env
        .post(&format!("{P}/fit"))
        .json(&body)
        .send()
        .await
        .json();
    assert!(
        r["verdict"] == "wont_fit" && r["per_gpu"][0]["uuid"] == "GPU-uuid-1",
        "{r}"
    );
    assert_eq!(r["per_gpu"][0]["free_gib"], 12.0);
    assert!(env.hub.calls.lock().unwrap().is_empty());
    let mut b = body.clone();
    b["gpu_ids"] = json!([0]);
    let r = env.post(&format!("{P}/fit")).json(&b).send().await.json();
    assert!(r["verdict"] == "fits" || r["verdict"] == "tight", "{r}");
    assert_eq!(r["per_gpu"][0]["free_gib"], 90.0);
    let _ = qwen_cfg();
}

#[tokio::test]
async fn test_fit_for_a_format_the_engine_cannot_load_is_422() {
    let mut env = FleetEnv::new();
    let r = env
        .post(&format!("{P}/fit"))
        .json(&json!({"engine": "fake", "model": {"format": "gguf", "root_id": "gguf", "rel": "a/m.gguf"}}))
        .send()
        .await;
    assert!(
        r.status == 422 && r.json()["code"] == "format_unsupported",
        "{}",
        r.text()
    );
}

#[tokio::test]
async fn test_fit_for_a_model_with_no_readable_facts_says_how_to_fix_it() {
    let mut env = FleetEnv::new();
    env.adapters
        .push(Arc::new(FakeMultiAdapter) as Arc<dyn EngineAdapter>);
    let r = env
        .post(&format!("{P}/fit"))
        .json(&json!({"engine": "fakecpp", "model": {"format": "gguf", "root_id": "gguf", "rel": "a/m.gguf"}}))
        .send()
        .await;
    let j = r.json();
    assert!(
        r.status == 422 && j["code"] == "model_info_unavailable",
        "{}",
        r.text()
    );
    assert!(j["detail"].as_str().unwrap().contains("rescan"));
}

#[tokio::test]
async fn test_fit_defaults_the_gpu_choice_per_host_ignoring_default_indices_the_host_does_not_have()
{
    let mut env = FleetEnv::new();
    seed_hub(&env);
    env.container()
        .settings
        .update(&SettingsPatch {
            default_gpu_ids: Some(vec![5]),
            ..SettingsPatch::default()
        })
        .unwrap();
    let r = env
        .post(&format!("{P}/fit"))
        .json(&json!({"engine": "fake", "repo_id": "Qwen/Qwen3-32B", "params": {"max_model_len": 512}}))
        .send()
        .await
        .json();
    let ids: Vec<i64> = r["per_gpu"]
        .as_array()
        .unwrap()
        .iter()
        .map(|g| g["gpu_id"].as_i64().unwrap())
        .collect();
    assert_eq!(ids, [0], "{r}");
}

#[test]
fn test_qwen_config_fixture_is_still_the_dense_gqa_model() {
    assert_eq!(qwen_cfg()["num_key_value_heads"], 8);
}
