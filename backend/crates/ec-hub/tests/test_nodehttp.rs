//! services/nodehttp.rs through the fake fleet: the "helper" section and the M2 line-splitting tests of
//! backend/tests/core/test_nodehttp_chat_bench.py (the chat and bench tests of that file are in
//! tests/test_nodehttp_chat_bench.rs).
//!
//! Python gets a port the node relays by starting a ready instance (lifecycle); here the fake node simply knows an
//! engine endpoint on that port, which is what the node's port rule looks at. The M2 tests stream a hostile body from
//! the fake node instead of stubbing `HttpStream.chunks`.

#![allow(
    clippy::expect_used,
    clippy::panic,
    clippy::unwrap_used,
    clippy::too_many_lines,
    clippy::type_complexity,
    clippy::needless_pass_by_value,
    clippy::cast_possible_truncation
)]

#[path = "common/kit.rs"]
mod kit;

// the fake fleet is the kit's submodule (PORTING.md §A4): one copy of its types per test binary
use kit::fakefleet;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use bytes::Bytes;
use ec_hub::domain::errors::{AppError, NodeError, Obj};
use ec_hub::domain::fleet::{FleetPort, StreamItem};
use ec_hub::services::nodehttp::{HttpOpts, MAX_LINE, NodeHttp};
use fakefleet::{EndpointOpts, FakeNode, FakeStream, FleetEnv, HttpResp, http_handler};
use serde_json::{Value, json};

const PORT: i64 = 18000;

fn setup() -> (FleetEnv, NodeHttp) {
    let env = FleetEnv::new();
    env.node.add_endpoint(PORT, EndpointOpts::default());
    let http = NodeHttp::new(Arc::clone(&env.fleet) as Arc<dyn FleetPort>);
    (env, http)
}

fn problem(e: &AppError) -> (u16, String, String) {
    match e {
        AppError::Problem(p) => (p.status, p.code.clone(), p.detail.clone()),
        AppError::Node(n) => (0, n.code.clone(), n.message.clone()),
    }
}

/// A streaming `http.request` answer of the given chunks (after a 200 head); `pulled` counts the bytes produced.
fn chunked_answer(node: &FakeNode, chunks: Vec<Vec<u8>>, pulled: Arc<AtomicUsize>) {
    node.on_open(
        "http.request",
        Arc::new(move |_n: &FakeNode, _a: &Obj| {
            let chunks = chunks.clone();
            let pulled = Arc::clone(&pulled);
            Ok(FakeStream::new(
                "http.request",
                Box::new(move |_s| {
                    Box::pin(async_stream::stream! {
                        yield StreamItem::Event(json!({"status": 200, "headers": {}}).as_object().cloned().unwrap());
                        for c in chunks {
                            pulled.fetch_add(c.len(), Ordering::SeqCst);
                            yield StreamItem::Data(Bytes::from(c));
                        }
                    })
                }),
                Box::new(|| Ok(Obj::new())),
            ))
        }),
    );
}

// ---- the helper ------------------------------------------------------------------------------------------------
#[tokio::test]
async fn test_a_buffered_request_returns_status_headers_and_body() {
    let (env, http) = setup();
    env.node.serve(
        None,
        "/v1/models",
        http_handler(|_| HttpResp::json(200, &json!({"data": []})).header("X-Thing", "1")),
    );
    let r = http
        .request("local", PORT, "GET", "/v1/models", HttpOpts::default())
        .await
        .unwrap();
    assert_eq!(r.status, 200);
    assert_eq!(
        serde_json::from_slice::<Value>(&r.body).unwrap(),
        json!({"data": []})
    );
    assert_eq!(r.header("x-thing"), Some("1"));
    let sent = env.node.ops("http.request").last().cloned().unwrap();
    assert!(sent["port"] == PORT && sent["method"] == "GET" && !sent.contains_key("body_len"));
}

#[tokio::test]
async fn test_a_request_body_is_uploaded_in_bounded_chunks_and_arrives_whole() {
    let (env, http) = setup();
    let seen: Arc<std::sync::Mutex<Vec<Vec<u8>>>> = Arc::default();
    let s2 = Arc::clone(&seen);
    env.node.serve(
        None,
        "/v1/echo",
        http_handler(move |r| {
            s2.lock().unwrap().push(r.body.clone());
            HttpResp::text(200, "ok")
        }),
    );
    let body: Vec<u8> = (0..2800).flat_map(|_| 0u8..=255).collect(); // ~700 KiB: more than two upload chunks
    let opts = HttpOpts {
        body: Some(body.clone()),
        headers: vec![("content-type".into(), "application/json".into())],
        ..HttpOpts::default()
    };
    let r = http
        .request("local", PORT, "POST", "/v1/echo", opts)
        .await
        .unwrap();
    assert_eq!(r.status, 200);
    assert_eq!(*seen.lock().unwrap(), vec![body.clone()]);
    assert_eq!(
        env.node.ops("http.request").last().unwrap()["body_len"],
        json!(body.len())
    );
    assert!(env.node.streams().last().unwrap().ended.is_set());
}

#[tokio::test]
async fn test_a_response_over_the_cap_is_refused_not_buffered() {
    let (env, http) = setup();
    env.node.serve(
        None,
        "/v1/big",
        http_handler(|_| HttpResp::bytes(200, vec![b'x'; 5000])),
    );
    let e = http
        .request(
            "local",
            PORT,
            "GET",
            "/v1/big",
            HttpOpts {
                max_bytes: Some(1000),
                ..HttpOpts::default()
            },
        )
        .await
        .unwrap_err();
    assert_eq!(problem(&e).1, "upstream_error");
    assert!(env.node.streams().last().unwrap().is_cancelled()); // and the node is told to stop sending
}

#[tokio::test]
async fn test_the_nodes_refusals_are_problems_naming_the_host() {
    let (env, http) = setup();
    // a port that is neither a gateway nor an endpoint
    let (status, code, detail) = problem(
        &http
            .request("local", 9999, "GET", "/health", HttpOpts::default())
            .await
            .unwrap_err(),
    );
    assert!(
        status == 403 && code == "policy_denied" && detail.starts_with("testbox: "),
        "{detail}"
    );
    // a path the node does not relay
    assert_eq!(
        problem(
            &http
                .request("local", PORT, "GET", "/admin/secrets", HttpOpts::default())
                .await
                .unwrap_err()
        )
        .1,
        "policy_denied"
    );
    env.node.serve(
        None,
        "/v1/boom",
        Arc::new(|_: &fakefleet::HttpReq| {
            Err(NodeError::new("local", "http_error", "connection refused"))
        }),
    );
    let (status, code, _) = problem(
        &http
            .request("local", PORT, "GET", "/v1/boom", HttpOpts::default())
            .await
            .unwrap_err(),
    );
    assert_eq!((status, code.as_str()), (502, "http_error"));
    env.node.set_state("down", None);
    let (status, code, _) = problem(
        &http
            .request("local", PORT, "GET", "/health", HttpOpts::default())
            .await
            .unwrap_err(),
    );
    assert_eq!((status, code.as_str()), (503, "node_unavailable"));
    let (status, code, _) = problem(
        &http
            .request("nope", 1, "GET", "/health", HttpOpts::default())
            .await
            .unwrap_err(),
    );
    assert_eq!((status, code.as_str()), (404, "unknown_host"));
}

#[tokio::test]
async fn test_lines_are_split_across_chunk_boundaries_and_multibyte_characters_survive() {
    let (env, http) = setup();
    env.node.st().http_chunk = 3; // smaller than one "✓" (3 bytes) boundary
    env.node.serve(
        None,
        "/v1/lines",
        http_handler(|_| HttpResp::bytes(200, "héllo ✓\r\nsecond\nthird".as_bytes().to_vec())),
    );
    let mut resp = http
        .open("local", PORT, "GET", "/v1/lines", HttpOpts::default())
        .await
        .unwrap();
    let mut lines = Vec::new();
    while let Some(l) = resp.line().await.unwrap() {
        lines.push(l);
    }
    resp.close().await;
    assert_eq!(lines, ["héllo ✓", "second", "third"]);
}

#[tokio::test]
async fn test_closing_a_stream_early_cancels_the_request_on_the_node() {
    let (env, http) = setup();
    env.node.st().http_chunk = 4;
    env.node.serve(
        None,
        "/v1/long",
        http_handler(|_| HttpResp::bytes(200, vec![b'x'; 400])),
    );
    let mut resp = http
        .open("local", PORT, "GET", "/v1/long", HttpOpts::default())
        .await
        .unwrap();
    assert!(resp.chunk().await.unwrap().is_some());
    resp.close().await;
    assert!(env.node.streams().last().unwrap().is_cancelled());
}

/// security review 2026-10-04 M2: a hostile engine streaming bytes with no newline made lines() quadratic in CPU and
/// unbounded in memory. Now a line over MAX_LINE stops the stream.
#[tokio::test]
async fn test_m2_a_stream_with_no_newline_is_cut_off_not_buffered() {
    let (env, http) = setup();
    let pulled = Arc::new(AtomicUsize::new(0));
    chunked_answer(
        &env.node,
        vec![vec![b'A'; 512 * 1024]; 64],
        Arc::clone(&pulled),
    ); // 32 MiB, never a newline
    let mut resp = http
        .open("local", PORT, "GET", "/v1/stream", HttpOpts::default())
        .await
        .unwrap();
    let err = loop {
        match resp.line().await {
            Ok(Some(_)) => {}
            Ok(None) => panic!("the stream ended instead of being cut off"),
            Err(e) => break e,
        }
    };
    assert_eq!(problem(&err).1, "upstream_error");
    assert!(pulled.load(Ordering::SeqCst) <= MAX_LINE + 512 * 1024);
    assert!(env.node.streams().last().unwrap().is_cancelled());
}

#[tokio::test]
async fn test_lines_handle_crlf_split_across_frames_and_tiny_frames_stay_linear() {
    let (env, http) = setup();
    chunked_answer(
        &env.node,
        vec![
            b"a\r".to_vec(),
            b"\nb\r".to_vec(),
            b"c\n".to_vec(),
            Vec::new(),
            b"d".to_vec(),
        ],
        Arc::default(),
    );
    let mut resp = http
        .open("local", PORT, "GET", "/v1/stream", HttpOpts::default())
        .await
        .unwrap();
    let mut lines = Vec::new();
    while let Some(l) = resp.line().await.unwrap() {
        lines.push(l);
    }
    assert_eq!(lines, ["a", "b", "c", "d"]);
    let mut line = vec![b'x'; 200_000];
    line.push(b'\n');
    chunked_answer(
        &env.node,
        line.iter().map(|b| vec![*b]).collect(),
        Arc::default(),
    ); // one byte per frame
    let t0 = Instant::now();
    let mut resp = http
        .open("local", PORT, "GET", "/v1/stream", HttpOpts::default())
        .await
        .unwrap();
    let mut lens = Vec::new();
    while let Some(l) = resp.line().await.unwrap() {
        lens.push(l.len());
    }
    assert_eq!(lens, [200_000]);
    assert!(t0.elapsed() < Duration::from_secs(5));
}

#[tokio::test]
async fn test_nodehttp_open_times_out_as_a_504_problem_and_cancels() {
    let (env, http) = setup();
    // a node that never sends the response head
    env.node.on_open(
        "http.request",
        Arc::new(|_n: &FakeNode, _a: &Obj| {
            Ok(FakeStream::new(
                "http.request",
                Box::new(|_s| Box::pin(futures::stream::pending())),
                Box::new(|| Ok(Obj::new())),
            ))
        }),
    );
    let started = Instant::now();
    let fut = http.open(
        "local",
        PORT,
        "GET",
        "/v1/slow",
        HttpOpts {
            timeout_s: Some(-9.99),
            ..HttpOpts::default()
        },
    );
    let (status, code, detail) = problem(
        &tokio::time::timeout(Duration::from_secs(5), fut)
            .await
            .unwrap()
            .unwrap_err(),
    );
    assert_eq!((status, code.as_str()), (504, "timeout"));
    assert!(detail.contains("did not answer in"), "{detail}");
    assert!(started.elapsed() < Duration::from_secs(5));
    assert!(env.node.streams().last().unwrap().is_cancelled());
}
