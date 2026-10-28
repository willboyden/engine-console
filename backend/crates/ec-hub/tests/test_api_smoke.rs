//! Port of backend/tests/core/test_api_smoke.py: the core rows (settings, keys, audit, auth, static files, error
//! shapes). The per-router smoke tests of instances, profiles, metrics, bench, chat, arena, HF and hardware belong to
//! the tests of each service (backend/TEST-MAP.md). Python's vehicle for "any guarded read" is GET
//! /instances (the lifecycle); here a 200 is shown on GET /settings and /instances is only checked to pass auth.
#![allow(clippy::panic, clippy::expect_used, clippy::unwrap_used)] // test helpers outside #[test] fns

#[path = "common/kit.rs"]
mod kit;

use std::os::unix::fs::PermissionsExt;

use kit::{Resp, TestEnv};
use serde_json::{Value, json};

const P: &str = "/api/v1";
const REMOTE: &str = "203.0.113.9";

/// test_api_smoke.problem_code: the problem+json content type, `status` equal to the HTTP status, and the code.
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

// ---- system: the core half of test_health_hardware_engines (/hardware: fleet tests) -----------------------------
#[tokio::test]
async fn test_health_and_engines() {
    let mut env = TestEnv::new();
    assert_eq!(
        env.get(&format!("{P}/health")).send().await.json()["status"],
        "ok"
    );
    let eng = env.get(&format!("{P}/engines")).send().await.json();
    assert_eq!(eng[0]["id"], "fake");
    assert_eq!(eng[0]["presets"][0]["name"], "balanced");
    assert_eq!(eng[0]["model_formats"], json!(["hf"]));
    let params = env
        .get(&format!("{P}/engines/fake/params"))
        .send()
        .await
        .json();
    let mut keys: Vec<&str> = params
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p["key"].as_str().unwrap())
        .collect();
    keys.sort_unstable();
    let mut want = [
        "max_model_len",
        "tp",
        "mem_fraction",
        "kv_dtype",
        "api_key",
        "leaky_key",
        "image",
        "trust_remote_code",
    ];
    want.sort_unstable();
    assert_eq!(keys, want);
    assert_eq!(
        problem_code(&env.get(&format!("{P}/engines/nope/params")).send().await),
        "unknown_engine"
    );
}

// ---- admin: settings, keys, audit, auth ---------------------------------------------------------------------------------
#[tokio::test]
async fn test_settings_keys_audit() {
    let mut env = TestEnv::new();
    env.admin_key(); // the start-up bootstrap key (Python's TestClient runs the lifespan)
    let s = env.get(&format!("{P}/settings")).send().await.json();
    assert_eq!(
        (s["hf_token"].clone(), s["docker_context"].clone()),
        (json!("[unset]"), json!("rootless"))
    );
    let upd = env
        .req("PUT", &format!("{P}/settings"))
        .json(&json!({"idle_ttl_s": 600, "image_pins": {"fake": "fake/engine:2.0"}}))
        .send()
        .await
        .json();
    assert_eq!(
        (upd["idle_ttl_s"].clone(), upd["image_pins"].clone()),
        (json!(600), json!({"fake": "fake/engine:2.0"}))
    );
    let bad = env
        .req("PUT", &format!("{P}/settings"))
        .json(&json!({"image_pins": {"fake": "fake/engine:latest"}}))
        .send()
        .await;
    assert_eq!(problem_code(&bad), "invalid_setting");
    assert_eq!(
        env.get(&format!("{P}/engines")).send().await.json()[0]["default_image"],
        "fake/engine:2.0"
    );
    let k = env
        .post(&format!("{P}/keys"))
        .json(&json!({"name": "ci", "role": "viewer"}))
        .send()
        .await
        .json();
    let secret = k["secret"].as_str().unwrap().to_string();
    assert!(secret.starts_with("ec_") && secret.len() > 30);
    let listed = env.get(&format!("{P}/keys")).send().await.json();
    let listed = listed.as_array().unwrap();
    assert!(listed.iter().all(|x| x["secret"].is_null()));
    let mut names: Vec<&str> = listed.iter().map(|x| x["name"].as_str().unwrap()).collect();
    names.sort_unstable();
    assert_eq!(names, ["bootstrap-admin", "ci"]);
    let kid = k["id"].as_str().unwrap();
    assert_eq!(
        env.req("DELETE", &format!("{P}/keys/{kid}"))
            .send()
            .await
            .status,
        204
    );
    assert_eq!(
        problem_code(&env.req("DELETE", &format!("{P}/keys/{kid}")).send().await),
        "not_found"
    );
    let audit = env.get(&format!("{P}/audit")).send().await.json()["items"].clone();
    let audit = audit.as_array().unwrap();
    assert_eq!(
        (audit[0]["method"].clone(), audit[0]["actor"].clone()),
        (json!("DELETE"), json!("loopback"))
    );
    assert!(
        audit
            .iter()
            .any(|a| a["path"].as_str().unwrap().ends_with("/keys") && a["method"] == "POST")
    );
    assert!(!serde_json::to_string(audit).unwrap().contains(&secret)); // secrets never reach the audit log
    let put = audit
        .iter()
        .find(|a| a["method"] == "PUT" && a["status"] == 200)
        .unwrap();
    assert_eq!(put["params"]["body"]["idle_ttl_s"], 600);
}

#[tokio::test]
async fn test_hf_token_status_never_leaks() {
    let mut env = TestEnv::with(|c| c.hf_token = Some("hf_supersecrettokenvalue123".into()));
    let text = env.get(&format!("{P}/settings")).send().await.text();
    assert!(text.contains("[set, 27 chars]") && !text.contains("supersecret"));
}

#[tokio::test]
async fn test_auth_roles_for_remote_clients() {
    let mut env = TestEnv::new();
    let remote = |env: &mut TestEnv, m: &str, path: &str| env.req(m, path).client(REMOTE);
    assert_eq!(
        remote(&mut env, "GET", &format!("{P}/health"))
            .send()
            .await
            .status,
        200
    ); // liveness is open
    let r = remote(&mut env, "GET", &format!("{P}/instances"))
        .send()
        .await;
    assert_eq!((r.status, problem_code(&r).as_str()), (401, "unauthorized"));
    assert_eq!(remote(&mut env, "GET", "/metrics").send().await.status, 401);
    let boot = env.admin_key();
    let key_mode = std::fs::metadata(env.cfg.data_dir.join("bootstrap-admin.key"))
        .unwrap()
        .permissions()
        .mode()
        & 0o777;
    assert_eq!(key_mode, 0o600);
    assert_eq!(
        remote(&mut env, "GET", &format!("{P}/settings"))
            .bearer(&boot)
            .send()
            .await
            .status,
        200
    );
    let s = remote(&mut env, "GET", &format!("{P}/instances"))
        .bearer(&boot)
        .send()
        .await
        .status;
    assert!(s != 401 && s != 403, "{s}");
    let viewer = remote(&mut env, "POST", &format!("{P}/keys"))
        .bearer(&boot)
        .json(&json!({"name": "v", "role": "viewer"}))
        .send()
        .await
        .json()["secret"]
        .as_str()
        .unwrap()
        .to_string();
    assert_eq!(
        remote(&mut env, "GET", &format!("{P}/settings"))
            .bearer(&viewer)
            .send()
            .await
            .status,
        200
    );
    let s = remote(&mut env, "GET", &format!("{P}/instances"))
        .bearer(&viewer)
        .send()
        .await
        .status;
    assert!(s != 401 && s != 403, "{s}");
    let forbid = remote(&mut env, "POST", &format!("{P}/profiles"))
        .bearer(&viewer)
        .json(&json!({"name": "n", "engine": "fake"}))
        .send()
        .await;
    assert_eq!(
        (forbid.status, problem_code(&forbid).as_str()),
        (403, "forbidden")
    );
    assert_eq!(
        remote(&mut env, "GET", &format!("{P}/keys"))
            .bearer(&viewer)
            .send()
            .await
            .status,
        403
    ); // admin-only read
    assert_eq!(
        remote(&mut env, "GET", &format!("{P}/instances"))
            .bearer("ec_wrong")
            .send()
            .await
            .status,
        401
    );
    let fit = remote(&mut env, "POST", &format!("{P}/fit"))
        .bearer(&viewer)
        .json(&json!({"engine": "fake", "repo_id": "a/b"}))
        .send()
        .await;
    assert_ne!(fit.status, 403);
    let page = env.container().audit.list_page(50, None).unwrap();
    let denied: Vec<&Value> = page
        .items
        .iter()
        .filter(|a| a["status"] == 401 || a["status"] == 403)
        .collect();
    assert!(
        !denied.is_empty(),
        "denied mutating attempts are audited too"
    );
}

#[tokio::test]
async fn test_x_forwarded_for_disables_loopback_trust() {
    let mut env = TestEnv::new();
    assert_eq!(
        env.get(&format!("{P}/instances"))
            .header("x-forwarded-for", "8.8.8.8")
            .send()
            .await
            .status,
        401
    );
}

// ---- static + errors -----------------------------------------------------------------------------------------------------
#[tokio::test]
async fn test_static_spa_and_error_shapes() {
    let mut env = TestEnv::new();
    assert!(env.get("/").send().await.text().contains("console"));
    assert_eq!(env.get("/js/app.js").send().await.text(), "console.log(1)");
    assert_eq!(env.get("/models/deep/link").send().await.status, 404); // only index + asset dirs are served
    assert_eq!(env.get("/..%2f..%2fetc/passwd").send().await.status, 404);
    let nf = env.get(&format!("{P}/nope")).send().await;
    assert_eq!((nf.status, problem_code(&nf).as_str()), (404, "not_found"));
    let v = env
        .post(&format!("{P}/fit"))
        .json(&json!({"engine": 5}))
        .send()
        .await;
    assert_eq!(
        (v.status, problem_code(&v).as_str()),
        (422, "validation_error")
    );
    assert!(!v.text().contains("input"));
    assert_eq!(
        env.get(&format!("{P}/openapi.json")).send().await.json()["info"]["title"],
        "Engine Console"
    );
    assert_eq!(env.get("/api/docs").send().await.status, 404);
}

/// Without Cache-Control a browser caches the ES modules heuristically, so after an upgrade a page can pair a new view
/// with an old shared module and fail its import. Every frontend file says `no-cache` and answers 304 while unchanged.
#[tokio::test]
async fn test_static_files_are_revalidated_on_every_load() {
    let mut env = TestEnv::new();
    for path in ["/", "/index.html", "/js/app.js"] {
        let r = env.get(path).send().await;
        assert_eq!(
            (r.status, r.header("cache-control")),
            (200, Some("no-cache")),
            "{path}"
        );
        let etag = r.header("etag").unwrap().to_string();
        let same = env.get(path).header("if-none-match", &etag).send().await;
        assert_eq!(same.status, 304, "{path}");
        assert!(same.body.is_empty(), "{path}");
        assert_eq!(
            (same.header("etag"), same.header("cache-control")),
            (Some(etag.as_str()), Some("no-cache"))
        );
        let weak = format!("\"other\", W/{etag}");
        assert_eq!(
            env.get(path)
                .header("if-none-match", &weak)
                .send()
                .await
                .status,
            304,
            "{path}"
        );
        assert_eq!(
            env.get(path)
                .header("if-none-match", "\"other\"")
                .send()
                .await
                .status,
            200,
            "{path}"
        );
    }
    let old = env
        .get("/js/app.js")
        .send()
        .await
        .header("etag")
        .unwrap()
        .to_string();
    std::fs::write(
        env.tmp.path().join("fe").join("js").join("app.js"),
        "console.log(2)  // a new release",
    )
    .unwrap();
    let fresh = env
        .get("/js/app.js")
        .header("if-none-match", &old)
        .send()
        .await;
    assert_eq!(fresh.status, 200);
    assert!(fresh.text().starts_with("console.log(2)"));
    assert_eq!(
        env.get("/js/missing.js")
            .header("if-none-match", &old)
            .send()
            .await
            .status,
        404
    );
}
