//! node/tests/test_httpproxy.py: http.request: allowed ports and paths only, header hygiene, uploads, streaming,
//! no redirects, timeouts. Every request goes to a FakeEngine on an ephemeral loopback port (or a closed port).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    unreachable_pub
)]

#[path = "common/wsd_kit.rs"]
mod wsd_kit;

use std::collections::HashSet;
use std::time::Duration;

use ec_proto::{BAD_ARGS, HTTP_ERROR, Obj, POLICY_DENIED, TIMEOUT};
use ecnode::errors::HandlerError;
use ecnode::httpproxy::request;
use serde_json::{Value, json};
use wsd_kit::{FakeEngine, FakeStream, clean_args, closed_port, obj};

fn merged(engine: &FakeEngine, args: Value) -> Obj {
    let mut a = obj(json!({"port": engine.port, "method": "GET"}));
    a.extend(obj(args));
    clean_args("http.request", Value::Object(a))
}

/// run(engine, args, stream, ports): `ports` None means only the engine's own port is allowed.
fn run(
    engine: &FakeEngine,
    args: Value,
    stream: &FakeStream,
    ports: Option<HashSet<u16>>,
) -> Result<Obj, HandlerError> {
    let allowed = ports.unwrap_or_else(|| HashSet::from([engine.port]));
    request(
        &merged(engine, args),
        stream,
        &|_refresh| allowed.clone(),
        false,
    )
}

fn code(r: Result<Obj, HandlerError>) -> String {
    match r {
        Err(HandlerError::Op(e)) => e.code,
        other => panic!("expected an OpError, got {other:?}"),
    }
}

#[test]
fn test_get_returns_one_event_then_the_body() {
    let engine = FakeEngine::start();
    let s = FakeStream::empty();
    let res = run(&engine, json!({"path": "/v1/models"}), &s, None).unwrap();
    assert_eq!(res["status"], 200);
    assert_eq!(res["bytes"], json!(s.data().len()));
    let events = s.events();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0]["status"], 200);
    assert_eq!(events[0]["headers"]["content-type"], "application/json");
    assert!(String::from_utf8_lossy(&s.data()).contains("org/model-a"));
}

#[test]
fn test_post_body_arrives_as_upload_and_headers_are_cleaned() {
    let engine = FakeEngine::start();
    let body = br#"{"model": "m", "messages": []}"#.to_vec();
    let s = FakeStream::new(vec![body[..10].to_vec(), body[10..].to_vec()]);
    let res = run(
        &engine,
        json!({"method": "POST", "path": "/v1/chat/completions", "body_len": body.len(),
               "headers": {"Authorization": "Bearer engine-key-123456", "Host": "evil.invalid",
                           "Connection": "keep-alive, X-Drop", "X-Drop": "1", "Transfer-Encoding": "chunked",
                           "Content-Type": "application/json"}}),
        &s,
        None,
    )
    .unwrap();
    assert_eq!(res["status"], 200);
    let mut want = b"echo:".to_vec();
    want.extend_from_slice(&body);
    assert_eq!(s.data(), want);
    let seen = engine.seen().last().cloned().unwrap();
    assert_eq!(seen.body, body);
    assert_eq!(
        seen.header("Host").unwrap(),
        format!("127.0.0.1:{}", engine.port)
    );
    assert_eq!(
        seen.header("Authorization").unwrap(),
        "Bearer engine-key-123456"
    );
    assert!(seen.header("X-Drop").is_none() && seen.header("Transfer-Encoding").is_none());
}

#[test]
fn test_refusals() {
    let engine = FakeEngine::start();
    for (args, want) in [
        (json!({"path": "/admin"}), POLICY_DENIED),
        (json!({"path": "/v1/../admin"}), BAD_ARGS),
        (json!({"path": "/v1/%2e%2e/admin"}), BAD_ARGS),
        (json!({"path": "/v1/models with space"}), BAD_ARGS),
        (json!({"path": "v1/models"}), BAD_ARGS),
        (json!({"path": "/v1/x\r\nHost: evil"}), BAD_ARGS),
        (
            json!({"path": "/v1/models", "method": "PUT"}),
            POLICY_DENIED,
        ),
        (
            json!({"path": "/v1/models", "method": "DELETE"}),
            POLICY_DENIED,
        ),
        (
            json!({"path": "/v1/models", "headers": {"X-A": "a\r\nInjected: 1"}}),
            BAD_ARGS,
        ),
        (
            json!({"path": "/v1/models", "headers": {"Bad Name": "x"}}),
            BAD_ARGS,
        ),
        (json!({"path": "/v1/models", "body_len": 5}), BAD_ARGS),
    ] {
        let label = args.to_string();
        assert_eq!(
            code(run(&engine, args, &FakeStream::empty(), None)),
            want,
            "{label}"
        );
    }
    assert!(engine.seen().is_empty());
}

#[test]
fn test_only_engine_and_gateway_ports() {
    let engine = FakeEngine::start();
    let r = run(
        &engine,
        json!({"path": "/v1/models"}),
        &FakeStream::empty(),
        Some(HashSet::new()),
    );
    assert_eq!(code(r), POLICY_DENIED);
}

#[test]
fn test_body_length_must_match() {
    let engine = FakeEngine::start();
    for chunks in [
        vec![b"abc".to_vec()],
        vec![b"abcdefgh".to_vec()],
        vec![b"abcde".to_vec(), b"extra".to_vec()],
    ] {
        let s = FakeStream::new(chunks);
        let r = run(
            &engine,
            json!({"method": "POST", "path": "/v1/chat/completions", "body_len": 5}),
            &s,
            None,
        );
        assert_eq!(code(r), BAD_ARGS);
    }
}

#[test]
fn test_redirects_are_returned_not_followed() {
    let engine = FakeEngine::start();
    let s = FakeStream::empty();
    let res = run(&engine, json!({"path": "/v1/redirect"}), &s, None).unwrap();
    assert_eq!(res["status"], 302);
    assert_eq!(
        s.events()[0]["headers"]["location"],
        "http://example.invalid/"
    );
    assert!(engine.seen().iter().all(|r| r.path != "/"));
}

#[test]
fn test_chunked_streams_are_forwarded() {
    let engine = FakeEngine::start();
    let s = FakeStream::empty();
    let res = run(&engine, json!({"path": "/v1/stream"}), &s, None).unwrap();
    assert_eq!(s.data(), b"data: 0\n\ndata: 1\n\ndata: 2\n\n");
    assert_eq!(res["bytes"], 27);
}

#[test]
fn test_timeout_and_cancel() {
    let engine = FakeEngine::start();
    let r = run(
        &engine,
        json!({"path": "/v1/slow", "timeout_s": 1}),
        &FakeStream::empty(),
        None,
    );
    match r {
        Err(HandlerError::Op(e)) => {
            assert_eq!(e.code, TIMEOUT);
            assert_eq!(
                e.message,
                format!(
                    "the engine on port {} did not answer within 1 s",
                    engine.port
                )
            );
        }
        other => panic!("{other:?}"),
    }
    let s = FakeStream::empty();
    let r = std::thread::scope(|sc| {
        sc.spawn(|| {
            std::thread::sleep(Duration::from_millis(300));
            s.cancel();
        });
        run(&engine, json!({"path": "/v1/slow"}), &s, None)
    });
    assert!(matches!(r, Err(HandlerError::Cancelled)), "{r:?}");
}

#[test]
fn test_nothing_listening() {
    let closed = closed_port();
    let args = clean_args(
        "http.request",
        json!({"port": closed, "method": "GET", "path": "/health"}),
    );
    let r = request(
        &args,
        &FakeStream::empty(),
        &|_| HashSet::from([closed]),
        false,
    );
    match r {
        Err(HandlerError::Op(e)) => {
            assert_eq!(e.code, HTTP_ERROR);
            assert_eq!(
                e.message,
                format!("nothing is listening on 127.0.0.1:{closed} (is the engine running?)")
            );
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn test_m4_monitor_mode_forwards_post_only_to_inference() {
    // security review 2026-10-04 M4: Ollama's pull/create/delete change the machine; monitor mode stays read-only
    let engine = FakeEngine::start();
    for path in [
        "/api/pull",
        "/api/create",
        "/api/delete",
        "/api/blobs/sha256:00",
        "/v1/load_lora_adapter",
    ] {
        let s = FakeStream::new(vec![b"{}".to_vec()]);
        let args = clean_args(
            "http.request",
            json!({"port": engine.port, "method": "POST", "path": path, "body_len": 2}),
        );
        let r = request(&args, &s, &|_| HashSet::from([engine.port]), true);
        match r {
            Err(HandlerError::Op(e)) => {
                assert_eq!(e.code, POLICY_DENIED, "{path}");
                assert!(e.message.contains("monitor"), "{}", e.message);
            }
            other => panic!("{path}: {other:?}"),
        }
        assert!(!engine.seen().iter().any(|r| r.path == path));
    }
}

#[test]
fn test_monitor_mode_still_chats_and_reads() {
    let engine = FakeEngine::start();
    let body = br#"{"model": "m", "messages": []}"#.to_vec();
    let args = clean_args(
        "http.request",
        json!({"port": engine.port, "method": "POST", "path": "/v1/chat/completions", "body_len": body.len()}),
    );
    let res = request(
        &args,
        &FakeStream::new(vec![body]),
        &|_| HashSet::from([engine.port]),
        true,
    )
    .unwrap();
    assert_eq!(res["status"], 200);
    let get = clean_args(
        "http.request",
        json!({"port": engine.port, "method": "GET", "path": "/v1/models"}),
    );
    let res = request(
        &get,
        &FakeStream::empty(),
        &|_| HashSet::from([engine.port]),
        true,
    )
    .unwrap();
    assert_eq!(res["status"], 200);
}

#[test]
fn test_refusal_texts_and_method_case() {
    // Rust-only: the refusal texts the hub shows, word for word, and the method upper-cased like str.upper().
    let engine = FakeEngine::start();
    let msg = |r: Result<Obj, HandlerError>| match r {
        Err(HandlerError::Op(e)) => e.message,
        other => panic!("{other:?}"),
    };
    assert_eq!(
        msg(run(
            &engine,
            json!({"path": "/admin"}),
            &FakeStream::empty(),
            None
        )),
        "path '/admin' is not an engine API path this node forwards (/v1/, /health, /metrics, /version, /api/, \
         /get_model_info, /get_server_info, /server_info)"
    );
    assert_eq!(
        msg(run(
            &engine,
            json!({"path": "/v1/models", "method": "put"}),
            &FakeStream::empty(),
            None
        )),
        "only GET and POST are forwarded to engines"
    );
    assert_eq!(
        msg(run(
            &engine,
            json!({"path": "/v1/models", "headers": {"Bad Name": "x"}}),
            &FakeStream::empty(),
            None
        )),
        "header name 'Bad Name' is not a valid HTTP token"
    );
    // a lower-case method is upper-cased first, as Python's str.upper()
    let s = FakeStream::empty();
    let res = run(
        &engine,
        json!({"path": "/v1/models", "method": "get"}),
        &s,
        None,
    )
    .unwrap();
    assert_eq!(res["status"], 200);
}

#[test]
fn test_path_and_header_rules_match_python_goldens() {
    // Rust-only: 423 seeded paths and 11 header sets through check_path / clean_headers, against the answers of
    // node/src/ecnode/httpproxy.py (tests/gen/wsd_httpproxy_vectors.py writes tests/data/wsd_httpproxy_vectors.json)
    let text = include_str!("data/wsd_httpproxy_vectors.json");
    let v: Value = serde_json::from_str(text).unwrap();
    let mut n = 0;
    for case in v["paths"].as_array().unwrap() {
        let path = case[0].as_str().unwrap();
        let want = &case[1];
        let got = match ecnode::httpproxy::check_path(path) {
            Ok(p) => json!({"ok": p}),
            Err(e) => json!({"code": e.code, "message": e.message}),
        };
        assert_eq!(&got, want, "{path:?}");
        n += 1;
    }
    for case in v["headers"].as_array().unwrap() {
        let want = &case[1];
        let got = match ecnode::httpproxy::clean_headers(case[0].as_object()) {
            Ok(h) => json!({"ok": h.iter().map(|(k, v)| json!([k, v])).collect::<Vec<_>>()}),
            Err(e) => json!({"code": e.code, "message": e.message}),
        };
        assert_eq!(&got, want, "{}", case[0]);
        n += 1;
    }
    assert_eq!(n, 434);
}
