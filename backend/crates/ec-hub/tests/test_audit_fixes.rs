//! Port of backend/tests/core/test_audit_fixes.py (pre-release audit findings): the core rows. Rows that need the
//! lifecycle or the HF client are with those services' tests (backend/TEST-MAP.md).
#![allow(clippy::panic, clippy::expect_used, clippy::unwrap_used)] // test helpers outside #[test] fns

#[path = "common/kit.rs"]
mod kit;

use std::collections::HashMap;
use std::os::unix::fs::PermissionsExt;
use std::sync::{Arc, Mutex};

use ec_hub::api::deps::enforce_roles;
use ec_hub::api::http::{BodyCapture, Ctx, Fail};
use ec_hub::api::security::{CSP, MUTATING, viewer_may};
use ec_hub::config::{GATEWAY_IMAGE, Settings};
use ec_hub::services::secrets_store::SecretStore;
use ec_hub::services::settings::Principal;
use kit::{Resp, TestEnv};
use serde_json::{Value, json};

const P: &str = "/api/v1";

fn mode(p: &std::path::Path) -> u32 {
    std::fs::metadata(p).unwrap().permissions().mode() & 0o777
}

// ---- 1. no Swagger page; openapi behind auth ---------------------------------------------------------------------
#[tokio::test]
async fn test_no_docs_pages_and_openapi_needs_auth() {
    let mut env = TestEnv::new();
    for path in ["/api/docs", "/api/redoc", "/api/v1/openapi.json"] {
        let s = env.get(path).client("203.0.113.9").send().await.status;
        assert!(s == 401 || s == 404, "{path}: {s}");
    }
    assert_eq!(
        env.get(&format!("{P}/openapi.json"))
            .client("203.0.113.9")
            .send()
            .await
            .status,
        401
    );
    assert_eq!(env.get("/api/docs").send().await.status, 404);
    assert_eq!(
        env.get(&format!("{P}/openapi.json")).send().await.status,
        200
    );
}

// ---- 2. security headers on every response ---------------------------------------------------------------------------
fn check_headers(r: &Resp) {
    for (k, v) in [
        ("x-content-type-options", "nosniff"),
        ("referrer-policy", "no-referrer"),
        ("x-frame-options", "DENY"),
        (
            "permissions-policy",
            "camera=(), microphone=(), geolocation=()",
        ),
        ("content-security-policy", CSP),
    ] {
        assert_eq!(r.header(k), Some(v), "{k}");
    }
}

#[tokio::test]
async fn test_security_headers_everywhere() {
    let mut env = TestEnv::new();
    let rs = vec![
        env.get(&format!("{P}/health")).send().await,
        env.get(&format!("{P}/nope")).send().await,
        env.get("/").send().await,
        env.get("/js/x.js").send().await,
        env.get(&format!("{P}/instances"))
            .header("host", "evil.example")
            .send()
            .await,
        env.post(&format!("{P}/fit"))
            .json(&json!({"engine": 3}))
            .send()
            .await,
        env.get("/metrics").send().await,
    ];
    for r in &rs {
        check_headers(r);
    }
    assert_eq!(
        env.get(&format!("{P}/health"))
            .send()
            .await
            .header("cache-control"),
        Some("no-store")
    );
    assert_eq!(
        env.get(&format!("{P}/nope"))
            .send()
            .await
            .header("cache-control"),
        Some("no-store")
    );
    assert_ne!(
        env.get("/").send().await.header("cache-control"),
        Some("no-store")
    ); // only /api/* is no-store
    let r = env
        .get(&format!("{P}/instances"))
        .client("203.0.113.9")
        .send()
        .await;
    assert_eq!(r.status, 401);
    check_headers(&r);
    assert!(
        CSP.contains("script-src 'self'")
            && CSP.contains("frame-ancestors 'none'")
            && !CSP.contains("unsafe-inline")
    );
}

// ---- 3. static allowlist -------------------------------------------------------------------------------------------------
#[tokio::test]
async fn test_static_serves_only_shell_and_asset_dirs() {
    let mut env = TestEnv::new();
    let fe = env.tmp.path().join("fe");
    for (d, f) in [
        ("js", "a.js"),
        ("css", "a.css"),
        ("i18n", "en.json"),
        ("dev", "mock.mjs"),
        ("tests", "t.mjs"),
    ] {
        std::fs::create_dir_all(fe.join(d)).unwrap();
        std::fs::write(fe.join(d).join(f), "x").unwrap();
    }
    std::fs::write(fe.join("package.json"), "{}").unwrap();
    std::fs::write(fe.join("secret.txt"), "nope").unwrap();
    for ok in [
        "/",
        "/index.html",
        "/js/a.js",
        "/css/a.css",
        "/i18n/en.json",
    ] {
        assert_eq!(env.get(ok).send().await.status, 200, "{ok}");
    }
    for bad in [
        "/dev/mock.mjs",
        "/tests/t.mjs",
        "/package.json",
        "/secret.txt",
        "/js",
        "/js/",
        "/js/missing.js",
        "/js/../secret.txt",
        "/js/..%2fsecret.txt",
        "/%2e%2e/etc/passwd",
        "/app.js",
    ] {
        assert_eq!(env.get(bad).send().await.status, 404, "{bad}");
    }
    std::os::unix::fs::symlink("/etc/hostname", fe.join("js").join("link")).unwrap();
    assert_eq!(env.get("/js/link").send().await.status, 404); // symlink escape
}

// ---- 6. roles: every mutating route --------------------------------------------------------------------------------------
fn concrete(path: &str) -> String {
    let re = regex::Regex::new(r"\{[^}]+\}").unwrap();
    re.replace_all(path, "x").into_owned()
}

#[tokio::test]
async fn test_viewer_gets_403_on_every_mutating_route_except_the_allowed_set() {
    let mut env = TestEnv::new();
    // enumerate from the OpenAPI document so newly added routes are picked up automatically
    let doc: Value = env.get(&format!("{P}/openapi.json")).send().await.json();
    let mut routes = Vec::new();
    for (path, ops) in doc["paths"].as_object().unwrap() {
        for m in ops.as_object().unwrap().keys() {
            let m = m.to_uppercase();
            if MUTATING.contains(&m.as_str()) {
                routes.push((m, path.clone()));
            }
        }
    }
    assert!(routes.len() >= 30, "route enumeration looks wrong");
    let boot = env.admin_key();
    let secret = env
        .post(&format!("{P}/keys"))
        .client("203.0.113.9")
        .bearer(&boot)
        .json(&json!({"name": "v", "role": "viewer"}))
        .send()
        .await
        .json()["secret"]
        .as_str()
        .unwrap()
        .to_string();
    let mut allowed = std::collections::BTreeSet::new();
    for (method, path) in &routes {
        let r = env
            .req(method, &concrete(path))
            .client("203.0.113.9")
            .bearer(&secret)
            .json(&json!({}))
            .send()
            .await;
        if viewer_may(method, &concrete(path)) {
            allowed.insert(path.clone());
            assert_ne!(r.status, 403, "{method} {path}");
        } else {
            assert_eq!(r.status, 403, "{method} {path}");
        }
    }
    let want: std::collections::BTreeSet<String> = [
        "/api/v1/chat/completions",
        "/api/v1/fit",
        "/api/v1/conversations",
        "/api/v1/arena/matches",
        "/api/v1/arena/matches/{mid}/vote",
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect();
    assert_eq!(allowed, want);
}

#[test]
fn test_viewer_cannot_delete_conversations_or_touch_prompts() {
    assert!(
        !viewer_may("DELETE", &format!("{P}/conversations/x"))
            && !viewer_may("POST", &format!("{P}/prompts"))
    );
    assert!(
        !viewer_may("POST", &format!("{P}/conversations/x/../keys"))
            && !viewer_may("POST", &format!("{P}/fitx"))
    );
    assert!(viewer_may("POST", &format!("{P}/fit")) && !viewer_may("GET", &format!("{P}/fit")));
}

#[tokio::test]
async fn test_enforce_roles_dependency_is_independent_of_the_middleware() {
    let mut env = TestEnv::new();
    let c = env.container();
    let req = |method: &str, path: &str, role: &str| {
        Ctx::new(
            Arc::clone(&c),
            method.into(),
            path.into(),
            String::new(),
            HashMap::new(),
            Principal::new("k", role, None),
            1024,
            Arc::new(Mutex::new(BodyCapture::default())),
            axum::body::Body::empty(),
        )
    };
    assert!(enforce_roles(&req("GET", &format!("{P}/keys"), "viewer")).is_ok());
    assert!(enforce_roles(&req("DELETE", &format!("{P}/keys/x"), "admin")).is_ok());
    assert!(enforce_roles(&req("POST", &format!("{P}/chat/completions"), "viewer")).is_ok());
    let forbidden = |r: Result<(), Fail>| matches!(r, Err(Fail::App(e)) if e.status() == 403 && e.code() == "forbidden");
    for (method, path) in [
        ("DELETE", "/conversations/x"),
        ("POST", "/instances"),
        ("PUT", "/settings"),
    ] {
        assert!(
            forbidden(enforce_roles(&req(method, &format!("{P}{path}"), "viewer"))),
            "{method} {path}"
        );
    }
    assert!(forbidden(enforce_roles(&req(
        "POST",
        &format!("{P}/instances"),
        "public"
    ))));
}

// ---- 7. images -------------------------------------------------------------------------------------------------------------
#[tokio::test]
async fn test_gateway_image_cannot_be_pinned_as_an_engine_image() {
    let mut env = TestEnv::new();
    let put = |pin: &str| json!({"image_pins": {"fake": pin}});
    let r = env
        .req("PUT", &format!("{P}/settings"))
        .json(&put(GATEWAY_IMAGE))
        .send()
        .await;
    assert_eq!((r.status, r.code().as_str()), (400, "image_not_allowed"));
    assert_eq!(
        env.req("PUT", &format!("{P}/settings"))
            .json(&put("fake/engine:2.0-evil"))
            .send()
            .await
            .status,
        400
    );
    assert_eq!(
        env.req("PUT", &format!("{P}/settings"))
            .json(&put("fake/engine:2.0"))
            .send()
            .await
            .status,
        200
    );
}

// ---- 8. names are constrained ------------------------------------------------------------------------------------------------
#[tokio::test]
async fn test_instance_names_are_constrained() {
    let mut env = TestEnv::new();
    let long = "a".repeat(65);
    for bad in ["a\nb", "x;y", "../a", "", long.as_str()] {
        let r = env
            .post(&format!("{P}/instances"))
            .json(&json!({"engine": "fake", "repo_id": "Qwen/Qwen3-32B", "name": bad}))
            .send()
            .await;
        assert_eq!(r.status, 422, "{bad:?}");
    }
}

// ---- 9. bounded metric labels -------------------------------------------------------------------------------------------------
#[tokio::test]
async fn test_method_metric_label_is_bounded() {
    let mut env = TestEnv::new();
    let long = "X".repeat(30);
    for m in ["BREW", "PROPFIND", long.as_str()] {
        env.req(m, &format!("{P}/health")).send().await;
    }
    let text = env.get("/metrics").send().await.text();
    assert!(text.contains("method=\"OTHER\""));
    assert!(!text.contains("BREW") && !text.contains("PROPFIND") && !text.contains("XXXXXXXX"));
}

// ---- 11. secret file modes, health warning, startup log -------------------------------------------------------------------------
#[test]
fn test_secret_files_are_forced_to_0600_even_if_preexisting() {
    let t = tempfile::tempdir().unwrap();
    let d = t.path().join("s");
    std::fs::create_dir(&d).unwrap();
    std::fs::set_permissions(&d, std::fs::Permissions::from_mode(0o755)).unwrap();
    let f = d.join("inst_1.json");
    std::fs::write(&f, "{}").unwrap();
    std::fs::set_permissions(&f, std::fs::Permissions::from_mode(0o644)).unwrap();
    SecretStore::new(d.clone())
        .put("inst_1", &[("api_key".into(), "k".into())])
        .unwrap();
    assert_eq!((mode(&f), mode(&d)), (0o600, 0o700));
}

#[tokio::test]
async fn test_bootstrap_key_rewritten_with_0600() {
    let mut env = TestEnv::new();
    std::fs::create_dir_all(&env.cfg.data_dir).unwrap();
    let key = env.cfg.data_dir.join("bootstrap-admin.key");
    std::fs::write(&key, "old").unwrap();
    std::fs::set_permissions(&key, std::fs::Permissions::from_mode(0o666)).unwrap();
    let dir = env.cfg.data_dir.clone();
    env.container().settings.ensure_bootstrap_key(&dir).unwrap();
    assert_eq!(mode(&key), 0o600);
}

#[tokio::test]
async fn test_health_warns_about_loopback_trust() {
    let mut on = TestEnv::new();
    let w = on.get(&format!("{P}/health")).send().await.json()["warnings"].clone();
    assert!(
        w.as_array()
            .unwrap()
            .iter()
            .any(|x| x.as_str().unwrap().contains("trust_loopback"))
    );
    let mut off = TestEnv::with(|c| c.trust_loopback = false);
    let w = off.get(&format!("{P}/health")).send().await.json()["warnings"].clone();
    assert!(
        !w.as_array()
            .unwrap()
            .iter()
            .any(|x| x.as_str().unwrap().contains("trust_loopback"))
    );
}

#[test]
fn test_loopback_is_not_trusted_by_default_any_more() {
    // Decision 21: the console can start containers on several machines, so a local process is not an admin
    assert!(!Settings::defaults().trust_loopback);
}

#[derive(Clone, Default)]
struct Buf(Arc<Mutex<Vec<u8>>>);

impl std::io::Write for Buf {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().write(b)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
fn test_startup_tells_operator_to_delete_bootstrap_key() {
    let buf = Buf::default();
    let w = buf.clone();
    let sub = tracing_subscriber::fmt()
        .with_ansi(false)
        .with_writer(move || w.clone())
        .finish();
    let _guard = tracing::subscriber::set_default(sub);
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let mut env = TestEnv::new();
    let c = env.container();
    rt.block_on(async {
        let tasks = ec_hub::app::lifespan_start(&c).await;
        ec_hub::app::lifespan_stop(&c, tasks).await;
    });
    let text = String::from_utf8(buf.0.lock().unwrap().clone()).unwrap();
    assert!(
        text.contains("DELETE this file") && text.contains("first start"),
        "{text}"
    );
    let key = std::fs::read_to_string(env.cfg.data_dir.join("bootstrap-admin.key")).unwrap();
    assert!(!text.contains(key.trim()));
}

// ---- 12. systemd unit --------------------------------------------------------------------------------------------------------------
#[test]
fn test_service_unit_hardening() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../deploy/engine-console.service"
    );
    let text = std::fs::read_to_string(path).unwrap();
    let live: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect();
    for want in [
        "RestrictNamespaces=yes",
        "RestrictRealtime=yes",
        "ProtectClock=yes",
        "ProtectHostname=yes",
        "PrivateUsers=yes",
        "UMask=0077",
        "CapabilityBoundingSet=",
        "SystemCallFilter=@system-service",
        "SystemCallArchitectures=native",
        "NoNewPrivileges=yes",
        "ProtectSystem=strict",
    ] {
        assert!(live.contains(&want), "{want}");
    }
    assert!(!live.iter().any(|l| l.starts_with("MemoryDenyWriteExecute")));
    // systemd does not support these in a per-user unit: listing them would only look like protection
    assert!(
        !live
            .iter()
            .any(|l| l.starts_with("ProtectProc=") || l.starts_with("ProtectControlGroups="))
    );
    assert!(
        live.contains(&"Environment=ENGINE_CONSOLE_HOST=127.0.0.1")
            && live.contains(&"Environment=ENGINE_CONSOLE_PORT=8791")
    );
    assert!(
        !live
            .iter()
            .any(|l| l.starts_with("Environment=HOST=") || l.starts_with("Environment=PORT="))
    );
    let rw: Vec<&str> = live
        .iter()
        .copied()
        .filter(|l| l.starts_with("ReadWritePaths="))
        .collect();
    let rw = rw.join(" ");
    assert!(
        rw.contains("%h/.local/state/ecnode")
            && rw.contains("/CHANGEME/")
            && !rw.contains("huggingface")
    );
    assert!(
        !live
            .iter()
            .any(|l| l.contains("EGRESS_PROXY") || l.contains("REQUIRE_EGRESS")),
        "unit must not change egress"
    );
    let lines: Vec<&str> = text.lines().collect();
    for (i, ln) in lines.iter().enumerate() {
        if ln.starts_with("WorkingDirectory=")
            || ln.starts_with("ExecStart=")
            || ln.starts_with("ReadWritePaths=")
        {
            let prev = |k: usize| if i >= k { lines[i - k] } else { "" };
            assert!(
                prev(1).contains("CHANGEME") || prev(2).contains("CHANGEME"),
                "{ln}"
            );
        }
    }
}

#[tokio::test]
async fn test_with_loopback_untrusted_a_local_client_needs_a_key_like_anyone_else() {
    let mut env = TestEnv::with(|c| c.trust_loopback = false);
    let r = env.get(&format!("{P}/instances")).send().await;
    assert_eq!((r.status, r.code().as_str()), (401, "unauthorized"));
    assert_eq!(env.get(&format!("{P}/health")).send().await.status, 200); // liveness stays open
    assert_eq!(env.get("/").send().await.status, 200); // and so does the app shell with its sign-in form
    assert_eq!(
        env.post(&format!("{P}/scripts/local/x/up"))
            .send()
            .await
            .status,
        401
    );
    assert_eq!(
        env.post(&format!("{P}/hosts/local/reconnect"))
            .send()
            .await
            .status,
        401
    );
    let key = env.admin_key();
    // Python checks GET /instances (the lifecycle); any guarded read shows the key is accepted
    assert_eq!(
        env.get(&format!("{P}/settings"))
            .bearer(&key)
            .send()
            .await
            .status,
        200
    );
    let s = env
        .get(&format!("{P}/instances"))
        .bearer(&key)
        .send()
        .await
        .status;
    assert!(s != 401 && s != 403);
}

// ---- security review 2026-10-04: L6 and L8 ----------------------------------------------------------------------------------
#[test]
fn test_l6_the_secret_store_takes_every_instance_id_shape_and_never_raises_on_delete() {
    let t = tempfile::tempdir().unwrap();
    let store = SecretStore::new(t.path().join("s"));
    let k = vec![("api_key".to_string(), "k".to_string())];
    for iid in ["inst_1", "ext-local-vllm", "scr-gpu--2-big-model"] {
        store.put(iid, &k).unwrap();
        assert_eq!(store.get(iid), k);
        store.delete(iid);
        assert!(store.get(iid).is_empty());
    }
    for bad in ["../x", "a/b", ".hidden", ""] {
        store.delete(bad); // used to raise ValueError -> HTTP 500
        assert!(store.get(bad).is_empty());
        assert!(store.put(bad, &k).is_err());
    }
}

#[tokio::test]
async fn test_l8_a_param_the_secret_store_treats_as_secret_is_masked_in_the_audit_log() {
    let mut env = TestEnv::new();
    let body = json!({"engine": "fake", "repo_id": "org/m",
                      "params": {"zz_access_key": "VERYSECRETVALUE-123", "api_key": "OTHER-456"}});
    env.post(&format!("{P}/instances")).json(&body).send().await;
    let c = env.container();
    let path = format!("{P}/instances");
    let rows = c
        .store
        .all(
            "SELECT params FROM audit WHERE method='POST' AND path=?",
            &[&path],
        )
        .unwrap();
    assert!(!rows.is_empty());
    for r in rows {
        let p = r.text("params").unwrap();
        assert!(
            !p.contains("VERYSECRETVALUE") && !p.contains("OTHER-456"),
            "{p}"
        );
    }
}
