//! Port of backend/tests/core/test_security.py: the core rows (security middleware, validation, settings, audit, stream
//! caps). Rows that need another service (instances, profiles, bench, the HF client) are with that service's tests
//! (backend/TEST-MAP.md). Where the Python test reaches the middleware through a stub route (/profiles), the Rust test
//! uses a ported route (/keys, /settings) as the vehicle: the middleware is the same for every route.
#![allow(clippy::panic, clippy::expect_used, clippy::unwrap_used)] // test helpers outside #[test] fns

#[path = "common/kit.rs"]
mod kit;

use std::time::Duration;

use ec_hub::adapters::load_default_adapters;
use ec_hub::config::Settings;
use ec_hub::domain::repo_id::check_repo_id;
use ec_hub::services::settings::SettingsService;
use ec_hub::services::store::Store;
use kit::{Resp, TestEnv};
use serde_json::json;

const P: &str = "/api/v1";

/// `problem_code`: a problem+json body whose `status` is the response's.
fn problem_code(r: &Resp) -> String {
    assert!(
        r.header("content-type")
            .unwrap_or("")
            .starts_with("application/problem+json"),
        "{}",
        r.text()
    );
    assert_eq!(r.json()["status"], json!(r.status));
    r.code()
}

// ---- 1. Host allowlist, Origin, CSRF header, content-type ---------------------------------------------------
#[tokio::test]
async fn test_host_allowlist_returns_421() {
    let mut env = TestEnv::new();
    assert_eq!(
        env.get(&format!("{P}/health"))
            .header("host", "evil.example:8791")
            .send()
            .await
            .status,
        421
    );
    assert_eq!(
        env.get("/")
            .header("host", "attacker.test")
            .send()
            .await
            .status,
        421
    ); // static too (rebinding)
    assert_eq!(
        env.get(&format!("{P}/health"))
            .header("host", "localhost:8791")
            .send()
            .await
            .status,
        200
    );
    assert_eq!(
        env.get(&format!("{P}/health"))
            .header("host", "127.0.0.1:8791")
            .send()
            .await
            .status,
        200
    );
    let r = env
        .get(&format!("{P}/health"))
        .header("host", "evil.example")
        .send()
        .await;
    assert_eq!(problem_code(&r), "misdirected_request");
}

#[tokio::test]
async fn test_unsafe_methods_need_custom_header_and_same_origin() {
    let mut env = TestEnv::new();
    let keys = format!("{P}/keys");
    let r = env
        .post(&keys)
        .without("x-engine-console")
        .json(&json!({"name": "n"}))
        .send()
        .await;
    assert_eq!(
        (r.status, problem_code(&r).as_str()),
        (403, "csrf_header_required")
    );
    let ok = env.post(&keys).json(&json!({"name": "n"})).send().await;
    assert_eq!(ok.status, 201);
    let cross = env
        .post(&keys)
        .header("origin", "https://evil.example")
        .json(&json!({"name": "m"}))
        .send()
        .await;
    assert_eq!(
        (cross.status, problem_code(&cross).as_str()),
        (403, "cross_origin")
    );
    let id = ok.json()["id"].as_str().unwrap().to_string();
    let null = env
        .req("DELETE", &format!("{keys}/{id}"))
        .header("origin", "null")
        .send()
        .await;
    assert_eq!(null.status, 403);
    let same = env
        .post(&keys)
        .header("origin", "http://testserver")
        .json(&json!({"name": "s"}))
        .send()
        .await;
    assert_eq!(same.status, 201);
    assert_eq!(
        env.get(&keys)
            .without("x-engine-console")
            .send()
            .await
            .status,
        200
    ); // safe methods are exempt
    let evil_get = env
        .get(&keys)
        .without("x-engine-console")
        .header("origin", "https://evil.example")
        .send()
        .await;
    assert_eq!(evil_get.status, 200);
    let pre = env
        .req("OPTIONS", &keys)
        .without("x-engine-console")
        .header("origin", "https://evil.example")
        .header("access-control-request-method", "POST")
        .send()
        .await;
    assert!(pre.header("access-control-allow-origin").is_none()); // no CORS is ever granted
}

#[tokio::test]
async fn test_bearer_clients_are_exempt_from_custom_header() {
    let mut env = TestEnv::new();
    let key = env.admin_key();
    let r = env
        .post(&format!("{P}/keys"))
        .client("203.0.113.9")
        .without("x-engine-console")
        .bearer(&key)
        .json(&json!({"name": "ci"}))
        .send()
        .await;
    assert_eq!(r.status, 201);
}

#[tokio::test]
async fn test_profile_import_requires_json_or_yaml_content_type() {
    let mut env = TestEnv::new();
    let path = format!("{P}/profiles/import");
    let body = "name: x\nengine: fake\n";
    let r = env
        .post(&path)
        .header("content-type", "text/plain")
        .raw(body)
        .send()
        .await;
    assert_eq!(
        (r.status, problem_code(&r).as_str()),
        (415, "unsupported_media_type")
    );
    let r = env
        .post(&path)
        .header("content-type", "application/x-www-form-urlencoded")
        .raw(body)
        .send()
        .await;
    assert_eq!(r.status, 415);
    // past the content-type gate (Python: 201 from the profiles service)
    let r = env
        .post(&path)
        .header("content-type", "application/yaml")
        .raw(body)
        .send()
        .await;
    assert_ne!(r.status, 415);
}

// ---- 2. hf_cache_dir -------------------------------------------------------------------------------------------
fn env_of(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
        .collect()
}

#[test]
fn test_cache_dir_validated_at_startup() {
    for bad in ["relative/path", "/tmp/a:/etc", "/tmp/a,b", "/tmp/a\nb"] {
        let e = Settings::load(&env_of(&[("HF_CACHE_DIR", bad)]), None).unwrap_err();
        assert_eq!(e.field, "hf_cache_dir", "{bad:?}");
    }
}

#[test]
fn test_cache_dir_symlink_rejected() {
    let d = tempfile::tempdir().unwrap();
    std::fs::create_dir(d.path().join("real")).unwrap();
    std::os::unix::fs::symlink(d.path().join("real"), d.path().join("link")).unwrap();
    let link = d.path().join("link").to_string_lossy().into_owned();
    assert!(Settings::load(&env_of(&[("HF_CACHE_DIR", &link)]), None).is_err());
}

/// The settings half of test_cache_dir_not_settable_via_api_and_mount_is_readonly (the read-only mount half needs
/// the lifecycle and is with its tests).
#[tokio::test]
async fn test_cache_dir_not_settable_via_api() {
    let mut env = TestEnv::new();
    let before = env.get(&format!("{P}/settings")).send().await.json()["hf_cache_dir"].clone();
    let r = env
        .req("PUT", &format!("{P}/settings"))
        .json(&json!({"hf_cache_dir": "/etc"}))
        .send()
        .await;
    assert_eq!(r.status, 200);
    assert_eq!(r.json()["hf_cache_dir"], before);
}

// ---- 3. request validation -------------------------------------------------------------------------------------
#[tokio::test]
async fn test_revision_and_repo_validation() {
    let mut env = TestEnv::new();
    for rev in ["../x", "a b", &"x".repeat(101), "a;b", "a/b"] {
        let r = env
            .post(&format!("{P}/downloads"))
            .json(&json!({"repo_id": "a/b", "revision": rev}))
            .send()
            .await;
        assert_eq!(r.status, 422, "{rev}");
    }
    assert_eq!(
        env.get(&format!("{P}/hf/models/a/b?revision=../x"))
            .send()
            .await
            .status,
        422
    );
}

// ---- 4. RepoId ---------------------------------------------------------------------------------------------------
#[test]
fn test_repo_id_rejects() {
    let long = format!("a/{}", "x".repeat(97));
    for bad in [
        "a--b/c",
        "a/b--c",
        "../a/b",
        "a/../b",
        "a/b/c",
        "a",
        "/a/b",
        "a/b\n",
        "-a/b",
        "a/.hidden",
        "a/b c",
        "a/b?x=1",
        long.as_str(),
    ] {
        assert!(check_repo_id(bad).is_err(), "{bad:?}");
    }
}

#[test]
fn test_repo_id_accepts() {
    for good in [
        "Qwen/Qwen3-32B",
        "org/m",
        "a/b.c_d-e",
        "openai/gpt-oss-120b",
    ] {
        assert!(check_repo_id(good).is_ok(), "{good}");
    }
}

#[tokio::test]
async fn test_every_route_validates_repo_id() {
    let mut env = TestEnv::new();
    let bad = "a--b/c";
    for (path, body) in [
        ("fit", json!({"engine": "fake", "repo_id": bad})),
        ("downloads", json!({"repo_id": bad})),
        ("instances", json!({"engine": "fake", "repo_id": bad})),
        (
            "instances/preflight",
            json!({"engine": "fake", "repo_id": bad}),
        ),
        (
            "profiles",
            json!({"name": "n", "engine": "fake", "repo_id": bad}),
        ),
    ] {
        assert_eq!(
            env.post(&format!("{P}/{path}"))
                .json(&body)
                .send()
                .await
                .status,
            422,
            "{path}"
        );
    }
    assert_eq!(
        env.get(&format!("{P}/hf/models/{bad}")).send().await.status,
        422
    );
    assert_eq!(
        env.req("DELETE", &format!("{P}/models/{bad}"))
            .send()
            .await
            .status,
        422
    );
}

// ---- 6. image control --------------------------------------------------------------------------------------------
/// The settings half of test_the_image_must_be_allowed_by_the_nodes_policy_for_pins_defaults_and_explicit_choices
/// (the preflight half needs the lifecycle and is with its tests).
#[tokio::test]
async fn test_settings_image_pins_must_be_on_the_allowlist() {
    let mut env = TestEnv::new();
    let r = env
        .req("PUT", &format!("{P}/settings"))
        .json(&json!({"image_pins": {"fake": "evil/miner:1"}}))
        .send()
        .await;
    assert_eq!(
        (r.status, problem_code(&r).as_str()),
        (400, "image_not_allowed")
    );
    let ok = env
        .req("PUT", &format!("{P}/settings"))
        .json(&json!({"image_pins": {"fake": "fake/engine:2"}}))
        .send()
        .await;
    assert_eq!(ok.status, 200);
}

#[test]
fn test_engine_image_allowlist_is_exact_match_with_adapter_pins() {
    let reg = load_default_adapters();
    let images: Vec<String> = reg
        .list()
        .iter()
        .map(|a| a.default_image().to_string())
        .collect();
    let cfg = std::sync::Arc::new(Settings::defaults());
    assert!(cfg.engine_image_allowlist.is_empty());
    let pins = images.clone();
    let svc = SettingsService::new(
        std::sync::Arc::new(Store::memory().unwrap()),
        cfg,
        Box::new(move || pins.clone()),
    );
    for image in &images {
        assert!(svc.engine_image_allowed(image), "{image}"); // pinned adapter defaults
    }
    assert!(!svc.engine_image_allowed("vllm/vllm-openai:evil")); // no prefix matching
    assert!(!svc.engine_image_allowed("vllm/vllm-openai:v0.23.0-evil"));
    assert!(!svc.engine_image_allowed(&Settings::defaults().gateway_image)); // the gateway is not an engine image
}

// ---- 9. audit hygiene ------------------------------------------------------------------------------------------------
#[tokio::test]
async fn test_audit_chat_is_metadata_only_and_query_redacted() {
    let mut env = TestEnv::new();
    // the chat service has its own tests; the audit records the request whatever the route answers
    env.post(&format!("{P}/chat/completions"))
        .json(&json!({"instance_id": "i1", "messages": [{"role": "user", "content": "my private diary"}]}))
        .send()
        .await;
    env.post(&format!("{P}/prompts?access_token=hf_abcdefgh123456&x=1"))
        .json(&json!({"title": "t", "content": "secret prompt text"}))
        .send()
        .await;
    let audit = env.get(&format!("{P}/audit")).send().await.json();
    let dump = audit.to_string();
    assert!(
        !dump.contains("private diary")
            && !dump.contains("secret prompt text")
            && !dump.contains("hf_abcdefgh")
    );
    let items = audit["items"].as_array().unwrap();
    let chat = items
        .iter()
        .find(|a| a["path"].as_str().unwrap().ends_with("/chat/completions"))
        .unwrap();
    assert_eq!(chat["params"]["body"]["messages"], json!(1));
    assert!(
        chat["params"]["body"]["fields"]
            .as_array()
            .unwrap()
            .contains(&json!("messages"))
    );
    let q = items
        .iter()
        .find(|a| a["path"].as_str().unwrap().ends_with("/prompts"))
        .unwrap();
    assert_eq!(q["params"]["query"]["access_token"], json!("[redacted]"));
    assert_eq!(q["params"]["query"]["x"], json!("1"));
}

#[tokio::test]
async fn test_audit_records_key_id_as_principal() {
    let mut env = TestEnv::new();
    let key = env.admin_key();
    env.post(&format!("{P}/keys"))
        .client("203.0.113.9")
        .bearer(&key)
        .json(&json!({"name": "n"}))
        .send()
        .await;
    let c = env.container();
    let kid = c.settings.list_keys().unwrap()[0].id.clone();
    let top = c.audit.list_page(5, None).unwrap().items[0].clone();
    assert_eq!(top["actor"], json!(kid));
    assert!(!top.to_string().contains(&key));
}

// ---- 10. size caps, stream limits ------------------------------------------------------------------------------------
#[tokio::test]
async fn test_body_caps() {
    let mut env = TestEnv::new();
    let big = json!({"engine": "fake", "repo_id": "a/b", "params": {"pad": "x".repeat(1024 * 1024 + 10)}});
    let r = env.post(&format!("{P}/fit")).json(&big).send().await;
    assert_eq!(
        (r.status, problem_code(&r).as_str()),
        (413, "payload_too_large")
    );
    let chunked = env
        .post(&format!("{P}/fit"))
        .header("content-type", "application/json")
        .raw([vec![b'x'; 600_000], vec![b'y'; 600_000]].concat())
        .chunked()
        .send()
        .await;
    assert_eq!(chunked.status, 413); // no Content-Length: counted while it is read
    let mut imp = b"a: ".to_vec();
    imp.extend(vec![b'x'; 256 * 1024 + 1]);
    let r = env
        .post(&format!("{P}/profiles/import"))
        .header("content-type", "text/yaml")
        .raw(imp)
        .send()
        .await;
    assert_eq!(r.status, 413);
    // past the size gate: chat has a larger cap (Python answers 404 for the missing instance: the chat service)
    let chat = json!({"instance_id": "nope", "messages": [{"role": "user", "content": "x".repeat(2 * 1024 * 1024)}]});
    assert_ne!(
        env.post(&format!("{P}/chat/completions"))
            .json(&chat)
            .send()
            .await
            .status,
        413
    );
    let huge = json!({"instance_id": "nope", "messages": [{"role": "user", "content": "x".repeat(33 * 1024 * 1024)}]});
    assert_eq!(
        env.post(&format!("{P}/chat/completions"))
            .json(&huge)
            .send()
            .await
            .status,
        413
    );
}

#[tokio::test]
async fn test_sse_connections_are_capped_and_released() {
    let mut env = TestEnv::with(|c| c.max_sse_connections = 1);
    let c = env.container();
    assert!(c.sse.acquire("")); // simulate one open stream
    let r = env
        .get(&format!("{P}/metrics/stream?once=true"))
        .send()
        .await;
    assert_eq!(
        (r.status, problem_code(&r).as_str()),
        (429, "too_many_streams")
    );
    c.sse.release("");
    assert_eq!(
        env.get(&format!("{P}/metrics/stream?once=true"))
            .send()
            .await
            .status,
        200
    );
    assert_eq!(c.sse.active(), 0);
}

/// Python uses /downloads/stream (the downloads service); any silent stream shows the idle timeout: /metrics/stream
/// without `once` only ends that way.
#[tokio::test]
async fn test_sse_idle_timeout_ends_silent_stream() {
    let mut env = TestEnv::with(|c| c.sse_idle_timeout_s = 0.05);
    let c = env.container();
    let mut r = env.get(&format!("{P}/metrics/stream")).open().await;
    assert_eq!(r.status, 200);
    let started = std::time::Instant::now();
    let body = r.rest(Duration::from_secs(5)).await;
    assert!(
        started.elapsed() < Duration::from_secs(4),
        "the idle timeout ended the stream"
    );
    assert!(body.is_empty());
    r.leave();
    assert_eq!(c.sse.active(), 0);
}

#[tokio::test]
async fn test_security_headers_on_every_answer() {
    let mut env = TestEnv::new();
    for path in ["/api/v1/health", "/", "/js/app.js", "/api/v1/nope", "/nope"] {
        let r = env.get(path).send().await;
        for (h, v) in ec_hub::api::security::SECURITY_HEADERS {
            assert_eq!(r.header(h), Some(*v), "{path}: {h}");
        }
        if path.starts_with("/api/") {
            assert_eq!(r.header("cache-control"), Some("no-store"), "{path}");
        }
    }
    let r = env.get("/js/app.js").send().await;
    assert_eq!(r.header("cache-control"), Some("no-cache"));
}
