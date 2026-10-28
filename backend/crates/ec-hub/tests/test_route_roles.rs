//! Structural role enforcement (hub hardening, finding 1). Rust-native: no Python test file; the role
//! table is pinned against the live Python app by `tests/data/route_roles.json`, written by
//! `backend/parity/gen/route_roles_python.py` (run with the Python reference) from each FastAPI route's dependencies.
//!
//! Every route declares the role its Python dependencies require (`api::router::Access`); the request pipeline
//! (app.rs) enforces it after authentication and before the handler. These tests walk EVERY registered route.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::too_many_lines
)]

#[path = "common/kit.rs"]
mod kit;

use std::path::Path;

use ec_hub::api::deps::check_access;
use ec_hub::api::http::Fail;
use ec_hub::api::router::Access;
use ec_hub::api::security::{MUTATING, guarded, viewer_may};
use ec_hub::domain::errors::AppError;
use ec_hub::services::settings::Principal;
use kit::TestEnv;
use serde_json::{Value, json};

const GOLDEN: &str = include_str!("data/route_roles.json");
/// Query parameters that fail validation on every route that reads them: a role error must come first.
const BAD_QUERY: &str = "?limit=notanint&once=maybe&host=BAD!&window=bad&revision=a..b&cursor=x";

struct Row {
    method: String,
    path: String,
    role: String,
    guarded: bool,
}

fn golden() -> Vec<Row> {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    doc["routes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| Row {
            method: r["method"].as_str().unwrap().into(),
            path: r["path"].as_str().unwrap().into(),
            role: r["role"].as_str().unwrap().into(),
            guarded: r["guarded"].as_bool().unwrap(),
        })
        .collect()
}

/// The Rust route table with the SPA route (Python's golden is made with a frontend dir).
fn table() -> Vec<(&'static str, &'static str, Access)> {
    ec_hub::api::routes::router(Some(Path::new("/frontend"))).table()
}

/// A concrete request path for a template; `{x:path}` gets a repo id that fails validation ("--" is refused), so a
/// handler that ran before the role check would answer 422.
fn concrete(template: &str) -> String {
    let mut out = String::new();
    let mut rest = template;
    while let Some(i) = rest.find('{') {
        out.push_str(&rest[..i]);
        let j = rest[i..].find('}').unwrap() + i;
        out.push_str(if rest[i..j].ends_with(":path") {
            "org/bad--repo"
        } else {
            "x1"
        });
        rest = &rest[j + 1..];
    }
    out.push_str(rest);
    out
}

fn mutating(method: &str) -> bool {
    MUTATING.contains(&method)
}

#[test]
fn test_the_route_role_table_matches_the_python_app() {
    let rows = golden();
    let t = table();
    assert_eq!(t.len(), rows.len(), "route count");
    for (i, ((method, path, access), row)) in t.iter().zip(&rows).enumerate() {
        assert_eq!(
            (*method, *path, access.as_str()),
            (row.method.as_str(), row.path.as_str(), row.role.as_str()),
            "route #{i}"
        );
        assert_eq!(guarded(&concrete(path)), row.guarded, "{method} {path}");
    }
    // the admin-only set, spelled out once: a change here is a security change
    let admin: Vec<String> = t
        .iter()
        .filter(|r| r.2 == Access::Admin)
        .map(|r| format!("{} {}", r.0, r.1))
        .collect();
    assert_eq!(admin.len(), 14, "{admin:?}");
}

/// trust_loopback off, no key: every guarded route is 401, whatever its role, method or (invalid) parameters.
#[tokio::test]
async fn test_every_guarded_route_needs_a_key() {
    let mut env = TestEnv::with(|s| s.trust_loopback = false);
    for (method, path, _) in table() {
        let url = format!("{}{BAD_QUERY}", concrete(path));
        let r = env
            .req(method, &url)
            .header("content-type", "application/json")
            .raw("{bad json")
            .send()
            .await;
        if guarded(&concrete(path)) {
            assert_eq!(r.status, 401, "{method} {path}: {}", r.text());
            if method != "HEAD" {
                assert_eq!(r.code(), "unauthorized", "{method} {path}");
            }
        } else {
            assert_ne!(r.status, 401, "{method} {path} is open");
        }
    }
}

/// A viewer key: every admin-only route, and every mutating method a viewer may not use, is 403 (never 422, 404 or
/// 200), before any validation of the (invalid) path, query and body.
#[tokio::test]
async fn test_a_viewer_is_refused_on_every_admin_route_and_mutating_method() {
    let mut env = TestEnv::with(|s| s.trust_loopback = false);
    let key = env
        .container()
        .settings
        .create_key("v", "viewer")
        .unwrap()
        .secret
        .unwrap();
    let mut refused = 0;
    for (method, path, access) in table() {
        let p = concrete(path);
        let denied = access == Access::Admin || (mutating(method) && !viewer_may(method, &p));
        if !denied {
            continue;
        }
        let url = format!("{p}{BAD_QUERY}");
        let r = env
            .req(method, &url)
            .bearer(&key)
            .header("content-type", "application/json")
            .raw("{bad json")
            .send()
            .await;
        assert_eq!(r.status, 403, "{method} {path}: {}", r.text());
        let detail = r.json()["detail"].as_str().unwrap_or_default().to_string();
        assert!(
            [
                "admin role required",
                "this key has the viewer role (read-only)"
            ]
            .contains(&detail.as_str()),
            "{method} {path}: {detail}"
        );
        refused += 1;
    }
    assert_eq!(
        refused, 35,
        "admin-only routes + the mutating ones a viewer may not use"
    );
}

/// The pipeline's own check (the router dependencies), independent of the middleware's viewer rule: what it answers
/// for every route and every role.
#[test]
fn test_the_pipeline_check_runs_the_python_dependencies_for_every_route() {
    let roles = [
        Principal::new("k", "admin", None),
        Principal::new("k", "viewer", None),
        Principal::public(),
    ];
    let detail = |r: Result<(), Fail>| match r {
        Ok(()) => None,
        Err(Fail::App(AppError::Problem(p))) => {
            assert_eq!((p.status, p.code.as_str()), (403, "forbidden"));
            Some(p.detail)
        }
        Err(other) => panic!("unexpected {other:?}"),
    };
    for (method, path, access) in table() {
        let p = concrete(path);
        for who in &roles {
            let got = detail(check_access(access, method, &p, who));
            let safe = ["GET", "HEAD", "OPTIONS"].contains(&method);
            let roles_ok =
                safe || who.role == "admin" || (who.role == "viewer" && viewer_may(method, &p));
            let want = match access {
                Access::Open => None,
                Access::Caller if roles_ok => None,
                Access::Admin if roles_ok && who.role == "admin" => None,
                Access::Admin if roles_ok => Some("admin role required".to_string()),
                _ => Some("admin role required for this action".to_string()),
            };
            assert_eq!(got, want, "{method} {path} as {}", who.role);
        }
    }
}

/// The pipeline's refusal is exactly the dependency's problem body, and validation still runs once the role passes.
#[tokio::test]
async fn test_role_errors_come_before_validation_with_the_dependencys_body() {
    let mut env = TestEnv::with(|s| s.trust_loopback = false);
    let c = env.container();
    let viewer = c
        .settings
        .create_key("v", "viewer")
        .unwrap()
        .secret
        .unwrap();
    let admin = c.settings.create_key("a", "admin").unwrap().secret.unwrap();
    let r = env
        .get("/api/v1/hf/search?limit=notanint")
        .bearer(&viewer)
        .send()
        .await;
    assert_eq!(
        r.json(),
        json!({"type": "urn:engine-console:forbidden", "title": "forbidden", "status": 403, "code": "forbidden",
               "detail": "admin role required"})
    );
    assert_eq!(r.header("content-type"), Some("application/problem+json"));
    let r = env
        .get("/api/v1/hf/search?limit=notanint")
        .bearer(&admin)
        .send()
        .await;
    assert_eq!((r.status, r.code().as_str()), (422, "validation_error"));
    let r = env.get("/api/v1/keys").bearer(&viewer).send().await;
    assert_eq!(
        (r.status, r.json()["detail"].clone()),
        (403, json!("admin role required"))
    );
    // a refused request is still counted and audited like any other (the middleware's finally)
    let r = env.delete("/api/v1/keys/kx").bearer(&viewer).send().await;
    assert_eq!(r.status, 403);
    let rows = c.audit.list_page(10, None).unwrap();
    assert!(
        rows.items
            .iter()
            .any(|a| a["path"] == "/api/v1/keys/kx" && a["status"] == 403),
        "{:?}",
        rows.items
    );
}
