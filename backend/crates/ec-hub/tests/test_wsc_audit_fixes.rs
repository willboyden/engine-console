//! The HF client rows of backend/tests/core/test_audit_fixes.py: the HF client in direct mode ignores the proxy and CA
//! environment, and a redirect to plain http is refused before anything connects.
//!
//! Python reads the httpx client's `_trust_env`. Rust cannot set the environment of its own (multi-threaded) test
//! process safely, so the direct-mode request runs in a child copy of this test binary with HTTPS_PROXY & co. pointing
//! at a listener that counts connections and SSL_CERT_FILE / SSL_CERT_DIR naming the test CA: the request must neither
//! touch the proxy nor trust that CA (its TLS handshake with the test upstream fails).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

#[path = "common/wsb_egress.rs"]
mod wsb_egress;
#[path = "common/wsc_tls.rs"]
mod wsc_tls;

use std::sync::Arc;

use ec_hub::domain::errors::AppError;
use ec_hub::domain::ports::HubClient;
use ec_hub::services::hf_http::{EgressConfig, HfHttpClient};
use wsb_egress::{ConnectProxy, Counter, has_openssl, make_certs};
use wsc_tls::{TlsRecorder, response};

const CHILD: &str = "WSC_AUDIT_CHILD_HUB";

#[tokio::test]
async fn test_direct_client_ignores_proxy_env_and_ca_env() {
    if !has_openssl() {
        eprintln!("SKIP: no openssl CLI");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let certs = make_certs(&dir.path().join("pki"));
    let hub = TlsRecorder::start(&certs);
    let proxy = Counter::start().await;
    let proxy_url = format!("http://127.0.0.1:{}", proxy.port);
    let home = dir.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let out = tokio::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "child_direct_client_request",
            "--nocapture",
            "--test-threads=1",
        ])
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("HOME", &home)
        .env(CHILD, hub.endpoint())
        .env("HTTPS_PROXY", &proxy_url)
        .env("https_proxy", &proxy_url)
        .env("HTTP_PROXY", &proxy_url)
        .env("ALL_PROXY", &proxy_url)
        .env("all_proxy", &proxy_url)
        .env("SSL_CERT_FILE", &certs.ca)
        .env("SSL_CERT_DIR", certs.ca.parent().unwrap())
        .output()
        .await
        .unwrap();
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "{text}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(text.contains("child: refused:"), "{text}");
    assert_eq!(
        proxy.hits(),
        0,
        "the direct client used the proxy from the environment"
    );
    // the TLS handshake failed: the CA named by SSL_CERT_FILE was not trusted, so no request was ever sent
    assert!(hub.requests().is_empty(), "{:?}", hub.requests());
}

/// The child half of the test above: a no-op unless that test started this process.
#[tokio::test]
async fn child_direct_client_request() {
    let Ok(endpoint) = std::env::var(CHILD) else {
        return;
    };
    let c = HfHttpClient::new(
        &endpoint,
        vec!["127.0.0.1".into()],
        Arc::new(|| None),
        EgressConfig::default(),
    );
    let e = c.card("a/b", None).await.err();
    match e {
        Some(AppError::Problem(p)) => {
            println!("child: refused: {} {}", p.code, p.detail);
        }
        other => panic!("the direct client got through: {other:?}"),
    }
}

#[tokio::test]
async fn test_redirect_to_http_is_refused() {
    if !has_openssl() {
        eprintln!("SKIP: no openssl CLI");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let certs = make_certs(&dir.path().join("pki"));
    let hub = TlsRecorder::start(&certs);
    let proxy = ConnectProxy::start().await;
    let downgraded = Counter::start().await;
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
    hub.route(
        "/a/b/resolve/main/f",
        response(
            302,
            &[(
                "location",
                &format!("http://127.0.0.1:{}/a/b/f", downgraded.port),
            )],
            "",
        ),
    );
    let e = c.open_file("a/b", "main", "f", 0).await.err().unwrap();
    assert!(
        matches!(&e, AppError::Problem(p) if p.code == "egress_denied" && p.detail.contains("non-https")),
        "{e:?}"
    );
    assert_eq!(downgraded.hits(), 0);
    assert_eq!(
        proxy.targets(),
        [format!("127.0.0.1:{}", hub.port)],
        "only the hub was reached"
    );
}
