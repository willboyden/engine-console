//! The library and data rows of backend/tests/core/test_security.py: the HF client's URL quoting, secret params in
//! profiles and benchmarks, and the benchmark suite clamps (also checked directly on resolve_suite).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

#[path = "common/kit.rs"]
mod kit;
#[path = "common/wsb_egress.rs"]
mod wsb_egress;
#[path = "common/wsc_tls.rs"]
mod wsc_tls;

use std::sync::Arc;

use ec_hub::domain::errors::AppError;
use ec_hub::domain::ports::HubClient;
use ec_hub::services::bench::BenchService;
use ec_hub::services::hf_http::{EgressConfig, HfHttpClient};
use kit::fakefleet::{HttpHandler, HttpResp, http_handler};
use kit::{Resp, TestEnv};
use serde_json::{Value, json};
use wsb_egress::{ConnectProxy, has_openssl, make_certs};
use wsc_tls::TlsRecorder;

const P: &str = "/api/v1";
const SECRET: &str = "sk-live-abcdef123456";

fn problem_code(r: &Resp) -> String {
    assert!(
        r.header("content-type")
            .unwrap_or_default()
            .starts_with("application/problem+json"),
        "{}",
        r.text()
    );
    assert_eq!(r.json()["status"], json!(r.status));
    r.code()
}

/// The seven suites of `test_bench_suite_is_clamped` (suite0..suite6).
fn clamped_suites() -> [Value; 7] {
    [
        json!({"concurrency": [1000]}),
        json!({"concurrency": (1..20).collect::<Vec<i64>>()}),
        json!({"prompt_tokens": 10_000_000}),
        json!({"max_tokens": 1_000_000}),
        json!({"rounds": 500}),
        json!({"concurrency": [true]}),
        json!({"surprise": 1}),
    ]
}

fn sse_engine() -> HttpHandler {
    http_handler(|req| {
        let sent: Value = serde_json::from_slice(&req.body).unwrap_or(Value::Null);
        let chunks = [
            json!({"choices": [{"delta": {"content": "Hi"}}]}),
            json!({"choices": [], "usage": {"prompt_tokens": 20, "completion_tokens": 3}}),
        ];
        if sent["stream"] == json!(true) {
            let mut text: String = chunks.iter().map(|c| format!("data: {c}\n\n")).collect();
            text.push_str("data: [DONE]\n\n");
            HttpResp::bytes(200, text.into_bytes()).header("content-type", "text/event-stream")
        } else {
            HttpResp::json(200, &json!({"choices": [{"message": {"content": "Hi"}}]}))
        }
    })
}

async fn make_ready_instance(env: &mut TestEnv) -> Value {
    env.seed_local_model("Qwen/Qwen3-32B", None, 65_000_000_000, "local");
    env.engine_up();
    let r = env
        .post(&format!("{P}/instances"))
        .json(
            &json!({"engine": "fake", "repo_id": "Qwen/Qwen3-32B", "name": "qwen",
                      "params": {"max_model_len": 8192}, "gpu_ids": [0]}),
        )
        .send()
        .await;
    assert_eq!(r.status, 201, "{}", r.text());
    let inst = r.json();
    let name = inst["container_name"].as_str().unwrap().to_string();
    env.node
        .st()
        .logs
        .insert(name, "loading weights 40%\n".into());
    env.container().lifecycle.tick().await.unwrap();
    env.get(&format!("{P}/instances/{}", inst["id"].as_str().unwrap()))
        .send()
        .await
        .json()
}

// ---- the HF client builds URLs from quoted parts --------------------------------------------------------------------
#[tokio::test]
async fn test_hub_urls_quote_repo_id() {
    if !has_openssl() {
        eprintln!("SKIP: no openssl CLI");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let certs = make_certs(&dir.path().join("pki"));
    let hub = TlsRecorder::start(&certs);
    let proxy = ConnectProxy::start().await;
    let c = HfHttpClient::new(
        &hub.endpoint(),
        vec!["127.0.0.1".into()],
        Arc::new(|| None),
        EgressConfig {
            proxy: Some(proxy.url()),
            ca_bundle: Some(certs.ca.clone()),
            require_proxy: true,
        },
    );
    // every path answers 404 -> NotFound
    let e = c.model("org/na me?x#y", None).await.err().unwrap();
    assert!(
        matches!(&e, AppError::Problem(p) if p.status == 404),
        "{e:?}"
    );
    let raw = hub.requests().first().unwrap().target.clone();
    assert!(
        raw.contains("%20") && raw.contains("%3Fx") && raw.contains("%23y"),
        "{raw}"
    );
    assert_eq!(
        raw.matches('?').count(),
        1,
        "only the real query separator: {raw}"
    );
}

// ---- secret params --------------------------------------------------------------------------------------------------
/// The profile half of `test_profile_and_export_drop_secrets_and_placeholder_must_be_resolved` (runs now).
#[tokio::test]
async fn test_profile_and_export_drop_secrets() {
    let mut env = TestEnv::new();
    let p = env
        .post(&format!("{P}/profiles"))
        .json(&json!({"name": "k", "engine": "fake", "params": {"api_key": SECRET, "tp": 1}}))
        .send()
        .await;
    assert!(
        p.status == 201 && p.json()["params"]["api_key"] == "[set]",
        "{}",
        p.text()
    );
    assert!(!p.text().contains(SECRET));
    let pid = p.json()["id"].as_str().unwrap().to_string();
    let y = env.get(&format!("{P}/profiles/{pid}/export")).send().await;
    assert!(
        y.status == 200 && !y.text().contains(SECRET),
        "{}",
        y.text()
    );
    let rows = env
        .container()
        .store
        .all("SELECT * FROM profiles", &[])
        .unwrap();
    assert!(!format!("{rows:?}").contains(SECRET));
}

#[tokio::test]
async fn test_profile_and_export_drop_secrets_and_placeholder_must_be_resolved() {
    let mut env = TestEnv::new();
    let p = env
        .post(&format!("{P}/profiles"))
        .json(&json!({"name": "k", "engine": "fake", "params": {"api_key": SECRET, "tp": 1}}))
        .send()
        .await;
    assert!(
        p.status == 201 && p.json()["params"]["api_key"] == "[set]" && !p.text().contains(SECRET)
    );
    let pid = p.json()["id"].as_str().unwrap().to_string();
    assert!(
        !env.get(&format!("{P}/profiles/{pid}/export"))
            .send()
            .await
            .text()
            .contains(SECRET)
    );
    env.seed_local_model("Qwen/Qwen3-32B", None, 65_000_000_000, "local");
    let body = json!({"engine": "fake", "repo_id": "Qwen/Qwen3-32B", "profile_id": pid});
    let r = env.post(&format!("{P}/instances")).json(&body).send().await;
    assert_eq!(r.status, 409, "{}", r.text());
    assert!(r.json()["preflight"].to_string().contains("placeholder"));
    let mut with = body.clone();
    with["params"] = json!({"api_key": SECRET});
    let ok = env.post(&format!("{P}/instances")).json(&with).send().await;
    assert!(
        ok.status == 201 && !ok.text().contains(SECRET),
        "{}",
        ok.text()
    );
}

#[tokio::test]
async fn test_bench_params_do_not_carry_secrets() {
    let mut env = TestEnv::new();
    env.seed_local_model("Qwen/Qwen3-32B", None, 65_000_000_000, "local");
    env.engine_up();
    env.engine_route("/v1/chat/completions", sse_engine());
    let inst = env
        .post(&format!("{P}/instances"))
        .json(&json!({"engine": "fake", "repo_id": "Qwen/Qwen3-32B",
                      "params": {"api_key": SECRET, "max_model_len": 4096}}))
        .send()
        .await
        .json();
    env.container().lifecycle.tick().await.unwrap();
    let r = env
        .post(&format!("{P}/bench"))
        .json(&json!({"instance_id": inst["id"]}))
        .send()
        .await;
    assert_eq!(r.status, 202, "{}", r.text());
    let bid = r.json()["id"].as_str().unwrap().to_string();
    env.container().bench.wait(&bid).await;
    assert!(
        !env.get(&format!("{P}/bench"))
            .send()
            .await
            .text()
            .contains(SECRET)
    );
}

// ---- benchmark suites are clamped -----------------------------------------------------------------------------------
#[tokio::test]
async fn test_bench_suite_is_clamped() {
    for suite in clamped_suites() {
        let mut env = TestEnv::new();
        let inst = make_ready_instance(&mut env).await;
        let r = env
            .post(&format!("{P}/bench"))
            .json(&json!({"instance_id": inst["id"], "suite": suite}))
            .send()
            .await;
        assert!(
            r.status == 400 && problem_code(&r) == "invalid_suite",
            "{suite}: {}",
            r.text()
        );
    }
}

/// The same seven suites through `resolve_suite`, which the route reaches once the instance is ready.
#[test]
fn test_bench_suite_is_clamped_by_resolve_suite() {
    for suite in clamped_suites() {
        let e = BenchService::resolve_suite(&suite).unwrap_err();
        assert!(
            matches!(&e, AppError::Problem(p) if p.status == 400 && p.code == "invalid_suite"),
            "{suite}: {e:?}"
        );
    }
    // a bool is an int to Python for the other bounds: `rounds: true` is 1 round
    assert_eq!(
        BenchService::resolve_suite(&json!({"rounds": true})).unwrap()["rounds"],
        json!(true)
    );
    let e = BenchService::resolve_suite(&json!("nope")).unwrap_err();
    assert!(
        matches!(&e, AppError::Problem(p) if p.code == "unknown_suite"),
        "{e:?}"
    );
}
