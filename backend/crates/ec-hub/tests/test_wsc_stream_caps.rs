//! The chat and fit rows of backend/tests/core/test_stream_caps.py (review finding L4): streamed chat replies take a
//! slot from the chat-stream caps (per key and in total), a slot comes back however a stream ends and the node is told
//! to stop the engine request, and a viewer's fit never spends the operator's Hugging Face token.
//!
//! Python drives the ASGI app by hand to hold a stream open or leave it (and runs the "client left" row for ASGI spec
//! 2.3 and 2.4: a disconnect reported through receive() or only by a failed send). Here kit's `Opened` holds a
//! response by not reading it and leaves by dropping its body, which is how hyper sees a client go away whatever the
//! spec: one Rust test covers both parametrized rows. The chat rows run against a ready instance of the lifecycle.

#![allow(clippy::panic, clippy::expect_used, clippy::unwrap_used)]

#[path = "common/kit.rs"]
mod kit;

use std::sync::Arc;
use std::time::Duration;

use ec_hub::domain::models::{Instance, InstanceCreate};
use kit::fakefleet::{HttpHandler, HttpResp, http_handler};
use kit::{Opened, TestEnv};
use serde_json::{Value, json};

const P: &str = "/api/v1";
const CHAT: &str = "/api/v1/chat/completions";
const REPO: &str = "Qwen/Qwen3-32B";

fn capped() -> TestEnv {
    let env: Vec<(String, String)> = [
        ("MAX_SSE_CONNECTIONS", "3"),
        ("MAX_SSE_PER_KEY", "2"),
        ("MAX_CHAT_STREAMS", "3"),
        ("MAX_CHAT_STREAMS_PER_KEY", "2"),
    ]
    .iter()
    .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
    .collect();
    let parsed = ec_hub::config::Settings::load(&env, None).unwrap();
    TestEnv::with(|c| {
        c.max_sse_connections = parsed.max_sse_connections;
        c.max_sse_per_key = parsed.max_sse_per_key;
        c.max_chat_streams = parsed.max_chat_streams;
        c.max_chat_streams_per_key = parsed.max_chat_streams_per_key;
    })
}

fn key(env: &mut TestEnv, name: &str, role: &str) -> String {
    env.container()
        .settings
        .create_key(name, role)
        .unwrap()
        .secret
        .unwrap()
}

/// test_nodehttp_chat_bench.engine
fn engine() -> HttpHandler {
    http_handler(|req| {
        let sent: Value = serde_json::from_slice(&req.body).unwrap_or(Value::Null);
        if sent["stream"] == json!(true) {
            let mut text: String = ["Hel", "lo", " there"]
                .iter()
                .map(|w| {
                    format!(
                        "data: {}\n\n",
                        json!({"choices": [{"delta": {"content": w}}]})
                    )
                })
                .collect();
            text.push_str(&format!(
                "data: {}\n\n",
                json!({"choices": [], "usage": {"prompt_tokens": 20, "completion_tokens": 3}})
            ));
            text.push_str("data: [DONE]\n\n");
            HttpResp::bytes(200, text.into_bytes()).header("content-type", "text/event-stream")
        } else {
            HttpResp::json(
                200,
                &json!({"choices": [{"message": {"content": "Hello there"}}],
                        "usage": {"prompt_tokens": 20, "completion_tokens": 3}}),
            )
        }
    })
}

async fn ready(env: &mut TestEnv) -> Instance {
    env.seed_local_model(REPO, None, 65_000_000_000, "local");
    env.engine_up();
    let c = env.container();
    let req: InstanceCreate = serde_json::from_value(
        json!({"engine": "fake", "repo_id": REPO, "params": {"max_model_len": 4096}}),
    )
    .unwrap();
    let inst = c.lifecycle.create(&req, None).await.unwrap();
    c.lifecycle.tick().await.unwrap();
    let got = c.lifecycle.get(&inst.id).unwrap();
    assert_eq!(got.state, "ready");
    got
}

fn body(inst: &Instance, stream: bool) -> Value {
    json!({"instance_id": inst.id, "messages": [{"role": "user", "content": "hi"}], "stream": stream})
}

/// A held chat request: opened, its first chunk read, then not read any further.
async fn hold(env: &mut TestEnv, key: &str, b: &Value) -> Opened {
    let mut o = env
        .post(CHAT)
        .client("192.0.2.7")
        .bearer(key)
        .json(b)
        .open()
        .await;
    if o.status == 200 {
        o.chunk(Duration::from_secs(5)).await;
    } else {
        o.rest(Duration::from_millis(500)).await;
    }
    o
}

async fn settle(mut cond: impl FnMut() -> bool) -> bool {
    for _ in 0..200 {
        if cond() {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    cond()
}

// ---- chat streams are capped ----------------------------------------------------------------------------------------
#[tokio::test]
async fn test_streamed_chat_replies_are_capped_per_key_and_the_non_streaming_path_is_not() {
    let mut env = capped();
    let inst = ready(&mut env).await;
    env.node.st().http_chunk = 8; // many small frames: a reply is still streaming while the test looks
    env.node.serve(None, "/v1/chat/completions", engine());
    let (a, b) = (key(&mut env, "a", "viewer"), key(&mut env, "b", "viewer"));
    let streamed = body(&inst, true);
    let mut held = vec![
        hold(&mut env, &a, &streamed).await,
        hold(&mut env, &a, &streamed).await,
    ];
    assert_eq!(
        held.iter().map(|c| c.status).collect::<Vec<_>>(),
        [200, 200]
    );
    let third = hold(&mut env, &a, &streamed).await;
    assert_eq!(third.status, 429);
    let p = third.json();
    assert_eq!(
        (p["code"].clone(), p["scope"].clone(), p["limit"].clone()),
        (json!("too_many_chat_streams"), json!("key"), json!(2))
    );
    // a non-streaming reply holds no stream slot
    let plain = env
        .post(CHAT)
        .client("192.0.2.7")
        .bearer(&a)
        .json(&body(&inst, false))
        .send()
        .await;
    assert_eq!(plain.status, 200, "{}", plain.text());
    let mut other = hold(&mut env, &b, &streamed).await;
    assert_eq!(other.status, 200);
    for c in &mut held {
        c.leave();
    }
    other.leave();
    let lim = Arc::clone(&env.container().chat_streams);
    assert!(settle(|| lim.active() == 0).await);
}

#[tokio::test]
async fn test_the_total_chat_stream_cap_holds_across_keys() {
    let mut env = capped();
    let inst = ready(&mut env).await;
    env.node.st().http_chunk = 8;
    env.node.serve(None, "/v1/chat/completions", engine());
    let streamed = body(&inst, true);
    let mut first = Vec::new();
    for i in 0..3 {
        let k = key(&mut env, &format!("k{i}"), "viewer");
        first.push(hold(&mut env, &k, &streamed).await);
    }
    assert_eq!(
        first.iter().map(|c| c.status).collect::<Vec<_>>(),
        [200, 200, 200]
    );
    let late_key = key(&mut env, "late", "viewer");
    let late = hold(&mut env, &late_key, &streamed).await;
    assert!(
        late.status == 429 && late.json()["scope"] == "total",
        "{}",
        late.json()
    );
    for c in &mut first {
        c.leave();
    }
}

/// Both parametrized rows (ASGI spec 2.3 and 2.4).
#[tokio::test]
async fn test_a_chat_slot_comes_back_when_the_client_leaves_and_the_engine_request_is_stopped() {
    let mut env = TestEnv::new();
    let inst = ready(&mut env).await;
    env.node.st().http_chunk = 8;
    env.node.serve(None, "/v1/chat/completions", engine());
    let k = key(&mut env, "k", "viewer");
    let mut c = hold(&mut env, &k, &body(&inst, true)).await;
    let ctr = env.container();
    assert!(c.status == 200 && ctr.chat_streams.active() == 1);
    c.leave();
    let lim = Arc::clone(&ctr.chat_streams);
    assert!(settle(|| lim.active() == 0).await);
    // the node was told to stop the engine request, and the partial reply is still accounted for
    let node = Arc::clone(&env.node);
    assert!(settle(|| node.streams().last().is_some_and(|s| s.is_cancelled())).await);
    let usage = Arc::clone(&ctr.usage);
    assert!(
        settle(|| usage
            .query("model", None, None)
            .is_ok_and(|u| u.first().is_some_and(|r| r.requests == 1)))
        .await
    );
}

#[tokio::test]
async fn test_a_chat_client_gone_before_the_first_byte_stops_the_engine_request() {
    let mut env = TestEnv::new();
    let inst = ready(&mut env).await;
    env.node.st().http_chunk = 8;
    env.node.serve(None, "/v1/chat/completions", engine());
    let k = key(&mut env, "k", "viewer");
    // the client goes away as soon as the response starts, before reading a byte of it
    let mut c = env
        .post(CHAT)
        .client("192.0.2.7")
        .bearer(&k)
        .json(&body(&inst, true))
        .open()
        .await;
    c.leave();
    let ctr = env.container();
    let lim = Arc::clone(&ctr.chat_streams);
    assert!(settle(|| lim.active() == 0).await);
    let node = Arc::clone(&env.node);
    assert!(
        settle(|| {
            let s = node.streams();
            !s.is_empty() && s.last().is_some_and(|x| x.is_cancelled())
        })
        .await
    );
}

#[tokio::test]
async fn test_a_chat_slot_comes_back_on_a_normal_end_and_on_a_refusal_before_streaming() {
    let mut env = TestEnv::new();
    let inst = ready(&mut env).await;
    env.node.serve(None, "/v1/chat/completions", engine());
    let k = key(&mut env, "k", "viewer");
    let streamed = body(&inst, true);
    let mut done = env
        .post(CHAT)
        .client("192.0.2.7")
        .bearer(&k)
        .json(&streamed)
        .open()
        .await;
    assert_eq!(done.status, 200);
    let all = done.rest(Duration::from_secs(5)).await;
    assert!(String::from_utf8_lossy(&all).contains("[DONE]"));
    // a server drops a body once it has sent its end (the kit keeps it until told)
    done.leave();
    let lim = Arc::clone(&env.container().chat_streams);
    assert!(settle(|| lim.active() == 0).await);
    let mut missing_body = streamed.clone();
    missing_body["instance_id"] = json!("nope");
    let missing = hold(&mut env, &k, &missing_body).await;
    assert_eq!(missing.status, 404);
    env.node.serve(
        None,
        "/v1/chat/completions",
        http_handler(|_| HttpResp::text(500, "engine fell over")),
    );
    let failed = hold(&mut env, &k, &streamed).await;
    assert_eq!(failed.status, 502);
    assert!(settle(|| lim.active() == 0).await);
}

// ---- /fit for a viewer: the operator's Hugging Face token is not spent ---------------------------------------------
#[tokio::test]
async fn test_a_viewer_fit_never_asks_hugging_face() {
    let mut env = TestEnv::new();
    env.seed_hub_model(REPO, None);
    let viewer = key(&mut env, "v", "viewer");
    let admin = key(&mut env, "adm", "admin");
    let fit = json!({"engine": "fake", "repo_id": REPO, "params": {"max_model_len": 8192}});
    let before = env.hub.calls.lock().unwrap().clone();
    let r = env
        .post(&format!("{P}/fit"))
        .client("192.0.2.7")
        .bearer(&viewer)
        .json(&fit)
        .send()
        .await;
    // not downloaded anywhere
    assert!(
        r.status == 422 && r.code() == "model_info_unavailable",
        "{}",
        r.text()
    );
    assert_eq!(*env.hub.calls.lock().unwrap(), before);
    let ok = env
        .post(&format!("{P}/fit"))
        .client("192.0.2.7")
        .bearer(&admin)
        .json(&fit)
        .send()
        .await;
    assert_eq!(ok.status, 200, "{}", ok.text());
    assert_ne!(*env.hub.calls.lock().unwrap(), before);
}
