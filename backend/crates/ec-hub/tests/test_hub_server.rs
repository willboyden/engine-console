//! The real listener and the request pipeline's edges (hub hardening, findings 2, 8 and 10).
//! Rust-native: no Python test file (uvicorn's own behaviour is the reference where it has one).
//!
//! Every server here is `server::serve_on` on an ephemeral 127.0.0.1 port, with all state in temp dirs (the kit's
//! `TestEnv`: fake fleet, no background loops, OTLP off, Hugging Face on a closed port), stopped by its switch.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::too_many_lines
)]

#[path = "common/kit.rs"]
mod kit;

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body as AxBody;
use ec_hub::api::http::{Ctx, HResult};
use ec_hub::api::router::{Access, Router};
use ec_hub::app::{AppState, app_state_with, handle_peer, handle_request, latin1};
use ec_hub::server::{Limits, client_host, serve_on, stop_switch};
use http::HeaderValue;
use http_body_util::BodyExt;
use kit::TestEnv;
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

/// A running listener: its address, its stop switch and its task.
struct Running {
    addr: SocketAddr,
    stop: tokio::sync::watch::Sender<bool>,
    task: tokio::task::JoinHandle<()>,
}

async fn start(st: Arc<AppState>, limits: Limits) -> Running {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (stop, rx) = stop_switch();
    let task = tokio::spawn(serve_on(listener, st, limits, rx));
    Running { addr, stop, task }
}

impl Running {
    /// Flip the switch; the loop must return (idle connections closed, nothing left) within `within`.
    async fn shut(self, within: Duration) {
        let _ = self.stop.send(true);
        tokio::time::timeout(within, self.task)
            .await
            .expect("the server stops in time")
            .unwrap();
    }
}

fn fast() -> Limits {
    Limits {
        header_read: Duration::from_millis(400),
        keep_alive_idle: Duration::from_millis(400),
        body_read: Duration::from_millis(600),
        max_connections: 64,
        shutdown_grace: Duration::from_secs(2),
    }
}

/// Send raw bytes (a request ending in `Connection: close`) and read the answer until the server closes.
async fn exchange(addr: SocketAddr, raw: &[u8]) -> String {
    let mut s = TcpStream::connect(addr).await.unwrap();
    s.write_all(raw).await.unwrap();
    read_to_close(&mut s, Duration::from_secs(10)).await
}

async fn read_to_close(s: &mut TcpStream, within: Duration) -> String {
    let mut buf = Vec::new();
    let _ = tokio::time::timeout(within, s.read_to_end(&mut buf)).await;
    String::from_utf8_lossy(&buf).into_owned()
}

fn status(resp: &str) -> u16 {
    resp.split(' ')
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

fn body(resp: &str) -> &str {
    resp.split_once("\r\n\r\n").map_or("", |(_, b)| b)
}

fn get(path: &str, extra: &str) -> Vec<u8> {
    format!("GET {path} HTTP/1.1\r\nHost: testserver\r\n{extra}Connection: close\r\n\r\n")
        .into_bytes()
}

// ── finding 10: the real listener ─────────────────────────────────────────────────────────────────────────────
/// The real accept loop: a request is dispatched, Host problems are refused like uvicorn/h11 (400) and the security
/// middleware (421), loopback trust comes from the socket's address only, and the server stops cleanly.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_the_real_listener_dispatches_checks_the_host_and_trusts_only_the_peer_address() {
    let mut env = TestEnv::new(); // trust_loopback ON, Host "testserver" allowed
    let c = env.container();
    let run = start(
        app_state_with(Arc::clone(&c), Limits::default()),
        Limits::default(),
    )
    .await;
    let a = run.addr;

    let r = exchange(a, &get("/api/v1/health", "")).await;
    assert_eq!(status(&r), 200, "{r}");
    let v: Value = serde_json::from_str(body(&r)).unwrap();
    assert_eq!(v["status"], "ok");
    assert!(r.to_lowercase().contains("x-frame-options: deny"));

    // the peer is 127.0.0.1 and trust_loopback is on: admin without a key
    let r = exchange(a, &get("/api/v1/keys", "")).await;
    assert_eq!(status(&r), 200, "{r}");
    // a forwarding header never adds trust (and on loopback it takes trust away, as in Python)
    let r = exchange(a, &get("/api/v1/keys", "X-Forwarded-For: 127.0.0.1\r\n")).await;
    assert_eq!(status(&r), 401, "{r}");

    // HTTP/1.1 without a Host header, or with two: refused before the app, like uvicorn (h11)
    let r = exchange(
        a,
        b"GET /api/v1/health HTTP/1.1\r\nConnection: close\r\n\r\n",
    )
    .await;
    assert_eq!(
        (status(&r), body(&r)),
        (400, "Invalid HTTP request received."),
        "{r}"
    );
    let r = exchange(
        a,
        b"GET /api/v1/health HTTP/1.1\r\nHost: testserver\r\nHost: evil.example\r\nConnection: close\r\n\r\n",
    )
    .await;
    assert_eq!(status(&r), 400, "{r}");
    // a well-formed but unknown Host: the middleware's DNS-rebinding refusal
    let r = exchange(
        a,
        b"GET /api/v1/health HTTP/1.1\r\nHost: evil.example:8791\r\nConnection: close\r\n\r\n",
    )
    .await;
    assert_eq!(status(&r), 421, "{r}");
    // HTTP/1.0 may omit Host (h11): it then fails the allowlist
    let r = exchange(a, b"GET /api/v1/health HTTP/1.0\r\n\r\n").await;
    assert_eq!(status(&r), 421, "{r}");
    // a non-ASCII Origin is present (latin-1, finding 8): refused, never ignored
    let mut raw = b"POST /api/v1/keys HTTP/1.1\r\nHost: testserver\r\nX-Engine-Console: 1\r\nOrigin: http://\xe9vil.example\r\n".to_vec();
    raw.extend_from_slice(
        b"Content-Type: application/json\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}",
    );
    let r = exchange(a, &raw).await;
    assert_eq!(status(&r), 403, "{r}");
    assert!(body(&r).contains("cross_origin"), "{r}");

    run.shut(Duration::from_secs(5)).await;
    assert!(
        TcpStream::connect(a).await.is_err(),
        "nothing listens after the stop"
    );
}

/// The seam that receives the socket address: only the address decides loopback trust.
#[tokio::test]
async fn test_loopback_trust_comes_from_the_socket_address_never_a_header() {
    let mut env = TestEnv::new(); // trust_loopback ON
    let st = env.app();
    let req = |xff: Option<&str>| {
        let mut b = http::Request::builder()
            .method("GET")
            .uri("/api/v1/keys")
            .header("host", "testserver");
        if let Some(x) = xff {
            b = b.header("x-forwarded-for", x);
        }
        b.body(AxBody::empty()).unwrap()
    };
    let peer = |s: &str| s.parse::<SocketAddr>().unwrap();
    for (addr, xff, want) in [
        ("127.0.0.1:40000", None, 200),
        ("[::1]:40000", None, 200),
        ("127.0.0.1:40000", Some("127.0.0.1"), 401),
        ("203.0.113.7:40000", None, 401),
        ("203.0.113.7:40000", Some("127.0.0.1"), 401),
        // an IPv4-mapped address is not on the loopback list (uvicorn reports it the same way)
        ("[::ffff:127.0.0.1]:40000", None, 401),
    ] {
        let r = handle_peer(&st, peer(addr), req(xff)).await;
        assert_eq!(r.status().as_u16(), want, "{addr} {xff:?}");
    }
    assert_eq!(client_host(peer("[::1]:5")), "::1");
    assert_eq!(client_host(peer("203.0.113.7:5")), "203.0.113.7");
}

// ── finding 2: limits ─────────────────────────────────────────────────────────────────────────────────────────
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_idle_keep_alive_and_slow_heads_are_closed() {
    let mut env = TestEnv::new();
    let c = env.container();
    let limits = fast();
    let run = start(app_state_with(Arc::clone(&c), limits), limits).await;

    // keep-alive: one answered request, then silence: the server closes after keep_alive_idle
    let mut s = TcpStream::connect(run.addr).await.unwrap();
    s.write_all(b"GET /api/v1/health HTTP/1.1\r\nHost: testserver\r\n\r\n")
        .await
        .unwrap();
    let mut buf = vec![0u8; 65536];
    let n = s.read(&mut buf).await.unwrap();
    assert_eq!(status(&String::from_utf8_lossy(&buf[..n])), 200);
    let t = Instant::now();
    let rest = read_to_close(&mut s, Duration::from_secs(5)).await;
    assert_eq!(rest, "", "closed without another byte");
    let waited = t.elapsed();
    assert!(
        waited >= Duration::from_millis(300) && waited < Duration::from_secs(3),
        "{waited:?}"
    );

    // a connection that never sends anything is closed the same way
    let mut s = TcpStream::connect(run.addr).await.unwrap();
    let t = Instant::now();
    assert_eq!(read_to_close(&mut s, Duration::from_secs(5)).await, "");
    assert!(t.elapsed() < Duration::from_secs(3));

    // a head that never ends: dropped at the header-read timeout (never an answer)
    let mut s = TcpStream::connect(run.addr).await.unwrap();
    s.write_all(b"GET /api/v1/health HTTP/1.1\r\nHost: testserver\r\n")
        .await
        .unwrap();
    let t = Instant::now();
    let got = read_to_close(&mut s, Duration::from_secs(5)).await;
    assert!(got.is_empty() || status(&got) == 408, "{got}");
    assert!(t.elapsed() < Duration::from_secs(3), "{:?}", t.elapsed());

    run.shut(Duration::from_secs(5)).await;
}

/// A body that is still arriving at the deadline is a 408; a body that ends before its Content-Length (the client
/// closed its side) is a 400 and is never handed to the handler as complete: no key is created, and the refused
/// request is audited with its status.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_slow_and_truncated_bodies_never_reach_a_handler_as_complete() {
    let mut env = TestEnv::new(); // loopback admin
    let c = env.container();
    let limits = fast();
    let run = start(app_state_with(Arc::clone(&c), limits), limits).await;
    let head = |len: usize| {
        format!(
            "POST /api/v1/keys HTTP/1.1\r\nHost: testserver\r\nX-Engine-Console: 1\r\n\
             Content-Type: application/json\r\nContent-Length: {len}\r\n\r\n"
        )
    };

    // slow: 10 of 100 bytes, then nothing
    let mut s = TcpStream::connect(run.addr).await.unwrap();
    s.write_all(head(100).as_bytes()).await.unwrap();
    s.write_all(b"{\"name\":\"").await.unwrap();
    let t = Instant::now();
    let r = read_to_close(&mut s, Duration::from_secs(5)).await;
    assert_eq!(status(&r), 408, "{r}");
    assert!(body(&r).contains("not received in time"), "{r}");
    assert!(t.elapsed() < Duration::from_secs(3));

    // truncated: a complete-looking JSON object, but 29 of the 100 bytes announced, then the client shuts down
    let mut s = TcpStream::connect(run.addr).await.unwrap();
    let doc = br#"{"name":"cut","role":"admin"}"#;
    s.write_all(head(100).as_bytes()).await.unwrap();
    s.write_all(doc).await.unwrap();
    s.shutdown().await.unwrap();
    let r = read_to_close(&mut s, Duration::from_secs(5)).await;
    assert_eq!(status(&r), 400, "{r}");
    let p: Value = serde_json::from_str(body(&r)).unwrap();
    assert_eq!(
        p,
        json!({"type": "urn:engine-console:http_error", "title": "http error", "status": 400, "code": "http_error",
               "detail": "There was an error parsing the body"})
    );
    let keys = c.settings.list_keys().unwrap();
    assert!(
        keys.iter().all(|k| k.name != "cut"),
        "the partial body created a key"
    );
    let audit = c.audit.list_page(20, None).unwrap();
    assert!(
        audit
            .items
            .iter()
            .any(|a| a["path"] == "/api/v1/keys" && a["status"] == 400 && a["method"] == "POST"),
        "{:?}",
        audit.items
    );
    run.shut(Duration::from_secs(5)).await;
}

/// Past the connection cap a connection is answered 503 and closed at once (never queued); a freed slot serves again.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_connections_past_the_cap_are_refused_at_once() {
    let mut env = TestEnv::new();
    let c = env.container();
    let limits = Limits {
        max_connections: 2,
        keep_alive_idle: Duration::from_secs(30),
        header_read: Duration::from_secs(30),
        ..fast()
    };
    let run = start(app_state_with(Arc::clone(&c), limits), limits).await;
    let a = TcpStream::connect(run.addr).await.unwrap();
    let _b = TcpStream::connect(run.addr).await.unwrap();
    tokio::time::sleep(Duration::from_millis(200)).await; // both accepted and holding their slots
    let t = Instant::now();
    let mut over = TcpStream::connect(run.addr).await.unwrap();
    let r = read_to_close(&mut over, Duration::from_secs(5)).await;
    assert!(r.starts_with("HTTP/1.1 503 Service Unavailable\r\n"), "{r}");
    assert!(t.elapsed() < Duration::from_secs(2));
    drop(a);
    tokio::time::sleep(Duration::from_millis(300)).await;
    let r = exchange(run.addr, &get("/api/v1/health", "")).await;
    assert_eq!(status(&r), 200, "{r}");
    run.shut(Duration::from_secs(5)).await;
}

// ── finding 2: a panicking handler ────────────────────────────────────────────────────────────────────────────
async fn boom(mut ctx: Ctx) -> HResult {
    let _ = ctx.read_body().await;
    panic!("handler bug while holding ec_abcdefghijklmnopqrstuvwxyz");
}

/// A panic inside an async handler is the generic 500, and the request is still counted and audited exactly like a
/// normal mutating call (the middleware's `finally`), with its body redacted as usual.
#[tokio::test]
async fn test_a_panicking_handler_is_a_500_and_still_audited() {
    let mut env = TestEnv::new();
    let c = env.container();
    let mut router = Router::default();
    router.add("POST", "/api/v1/boom", Access::Caller, |ctx| {
        Box::pin(boom(ctx))
    });
    let st = Arc::new(AppState {
        hosts: ec_hub::api::security::allowed_hosts(&c.cfg),
        c: Arc::clone(&c),
        router,
        limits: Limits::default(),
    });
    let req = http::Request::builder()
        .method("POST")
        .uri("/api/v1/boom?x=1")
        .header("host", "testserver")
        .header("x-engine-console", "1")
        .header("content-type", "application/json")
        .body(AxBody::from(r#"{"api_key": "s3cret", "n": 1}"#))
        .unwrap();
    let r = handle_request(&st, "127.0.0.1", req).await;
    assert_eq!(r.status().as_u16(), 500);
    let b = r.into_body().collect().await.unwrap().to_bytes();
    let p: Value = serde_json::from_slice(&b).unwrap();
    assert_eq!(
        (p["code"].as_str(), p["detail"].as_str()),
        (Some("internal_error"), Some("unexpected server error"))
    );
    let audit = c.audit.list_page(10, None).unwrap();
    let row = audit
        .items
        .iter()
        .find(|a| a["path"] == "/api/v1/boom")
        .expect("an audit row for the panicking request");
    assert_eq!(
        (row["status"].clone(), row["method"].clone()),
        (json!(500), json!("POST"))
    );
    assert_eq!(row["actor"], "loopback");
    let params = row["params"].to_string();
    assert!(
        params.contains("[redacted]") && !params.contains("s3cret"),
        "{params}"
    );
    // the next request is served normally
    let ok = http::Request::builder()
        .method("POST")
        .uri("/api/v1/boom")
        .header("host", "evil.example")
        .body(AxBody::empty())
        .unwrap();
    assert_eq!(handle_request(&st, "127.0.0.1", ok).await.status(), 421);
}

// ── finding 8: header decoding ────────────────────────────────────────────────────────────────────────────────
/// Starlette decodes header values as latin-1: a non-ASCII byte is one character, so a header that is present is never
/// taken as absent, and a configured non-ASCII host name matches its latin-1 bytes.
#[tokio::test]
async fn test_headers_are_decoded_as_latin1() {
    assert_eq!(latin1(b"caf\xe9"), "café");
    assert_eq!(latin1(b"\xff\x00a"), "\u{ff}\u{0}a");
    let mut env = TestEnv::with(|s| s.allowed_hosts.push("café:8791".into()));
    let st = env.app();
    let send = |pairs: Vec<(&'static str, &'static [u8])>, method: &'static str| {
        let mut b = http::Request::builder().method(method).uri("/api/v1/keys");
        for (k, v) in pairs {
            b = b.header(k, HeaderValue::from_bytes(v).unwrap());
        }
        let req = b.body(AxBody::empty()).unwrap();
        let st = Arc::clone(&st);
        async move {
            handle_request(&st, "127.0.0.1", req)
                .await
                .status()
                .as_u16()
        }
    };
    // Host: the latin-1 spelling of an allowed name is allowed (UTF-8 lossy would have made it "caf\u{fffd}")
    assert_eq!(send(vec![("host", b"CAF\xc9:8791")], "GET").await, 200);
    assert_eq!(send(vec![("host", b"caf\xe9-x:8791")], "GET").await, 421);
    // Origin with a non-ASCII byte: present, so compared, so refused
    assert_eq!(
        send(
            vec![
                ("host", b"testserver"),
                ("x-engine-console", b"1"),
                ("origin", b"http://\xe9vil.example"),
            ],
            "POST"
        )
        .await,
        403
    );
    // X-Engine-Console must be exactly "1"
    assert_eq!(
        send(
            vec![("host", b"testserver"), ("x-engine-console", b"1\xa0")],
            "POST"
        )
        .await,
        403
    );
    // a bearer token with non-ASCII bytes is simply not a valid key
    assert_eq!(
        send(
            vec![
                ("host", b"testserver"),
                ("authorization", b"Bearer ec_\xe9\xe9")
            ],
            "GET"
        )
        .await,
        401
    );
}
