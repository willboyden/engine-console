//! Port of backend/tests/core/test_nodehttp_chat_bench.py: chat (SSE passthrough) and the benchmark on top of the
//! node HTTP helper. The helper rows of the Python file (and its M2 rows) are in tests/test_nodehttp.rs under the same
//! names. The rows that need a ready instance get one from the lifecycle (`create` + `tick`).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

#[path = "common/kit.rs"]
mod kit;

use std::sync::Arc;
use std::time::Duration;

use ec_hub::domain::errors::{AppError, NodeError};
use ec_hub::domain::models::{ConversationIn, Instance, InstanceCreate};
use ec_hub::services::bench::{ReqStat, one_request};
use futures::StreamExt as _;
use kit::TestEnv;
use kit::fakefleet::{EndpointOpts, HelloOpts, HttpHandler, HttpResp, http_handler, ports};
use serde_json::{Value, json};

const CHAT: &str = "/v1/chat/completions";
const REPO: &str = "Qwen/Qwen3-32B";

fn sse_body(words: &[&str], usage: bool) -> Vec<u8> {
    let mut chunks: Vec<Value> = words
        .iter()
        .map(|w| json!({"choices": [{"delta": {"content": w}}]}))
        .collect();
    if usage {
        chunks.push(json!({"choices": [], "usage": {"prompt_tokens": 20, "completion_tokens": 3}}));
    }
    let mut out: String = chunks
        .iter()
        .map(|c| format!("data: {}\n\n", ec_hub::pyjson::dumps(c)))
        .collect();
    out.push_str("data: [DONE]\n\n");
    out.into_bytes()
}

/// The fake engine: SSE when asked to stream, else one JSON answer with reasoning.
fn engine() -> HttpHandler {
    http_handler(|req| {
        let sent: Value = serde_json::from_slice(&req.body).unwrap_or(Value::Null);
        if sent["stream"] == json!(true) {
            HttpResp::bytes(200, sse_body(&["Hel", "lo", " there"], true))
                .header("content-type", "text/event-stream")
        } else {
            HttpResp::json(
                200,
                &json!({"choices": [{"message": {"content": "Hello there", "reasoning_content": "hmm"}}],
                        "usage": {"prompt_tokens": 20, "completion_tokens": 3}}),
            )
        }
    })
}

fn create_req(host_id: &str) -> InstanceCreate {
    serde_json::from_value(
        json!({"engine": "fake", "repo_id": REPO, "params": {"max_model_len": 4096},
                                  "host_id": host_id}),
    )
    .unwrap()
}

/// `ready(env)`: a local instance of the fake engine, ticked to ready.
async fn ready(env: &mut TestEnv) -> Instance {
    env.seed_local_model(REPO, None, 65_000_000_000, "local");
    env.engine_up();
    let c = env.container();
    let inst = c
        .lifecycle
        .create(&create_req("local"), None)
        .await
        .unwrap();
    c.lifecycle.tick().await.unwrap();
    let got = c.lifecycle.get(&inst.id).unwrap();
    assert_eq!(got.state, "ready");
    got
}

fn messages() -> Value {
    json!([{"role": "user", "content": "hi"}])
}

/// The problem a client would see for this error (node errors labelled with the host).
fn problem(env: &mut TestEnv, e: &AppError) -> ec_hub::domain::errors::Problem {
    let c = env.container();
    let port: Arc<dyn ec_hub::domain::fleet::FleetPort> = c.fleet.clone();
    ec_hub::services::nodeerr::to_problem(e, Some(&port))
}

async fn until(what: &str, mut cond: impl FnMut() -> bool) {
    for _ in 0..200 {
        if cond() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("timed out waiting for {what}");
}

// ---- chat -----------------------------------------------------------------------------------------------------------
#[tokio::test]
async fn test_chat_streams_through_the_node_and_records_usage_and_the_conversation() {
    let mut env = TestEnv::new();
    let inst = ready(&mut env).await;
    env.node.serve(None, CHAT, engine());
    let chat = Arc::clone(&env.container().chat);
    let conv = chat
        .create_conversation(&ConversationIn {
            title: "t".into(),
            system_prompt: None,
            instance_id: Some(inst.id.clone()),
        })
        .unwrap();
    let stream = chat
        .open_stream(
            &json!({"instance_id": inst.id, "conversation_id": conv.id, "messages": messages(),
                             "stream": true}),
        )
        .await
        .unwrap();
    let text: String = stream.lines.collect::<Vec<_>>().await.concat();
    assert!(text.contains("Hel") && text.contains("[DONE]") && text.matches("\n\n").count() >= 3);
    let req = env.engine_requests().last().cloned().unwrap();
    let sent: Value = serde_json::from_slice(&req.body).unwrap();
    assert!(req.port == inst.port.unwrap() && req.path == CHAT && req.method == "POST");
    assert!(
        sent["model"] == REPO
            && sent.get("instance_id").is_none()
            && sent.get("conversation_id").is_none()
    );
    assert!(
        sent["stream"] == json!(true) && sent["stream_options"] == json!({"include_usage": true})
    );
    let op = env.node.ops("http.request").last().cloned().unwrap();
    assert_eq!(op["body_len"], json!(req.body.len()));
    assert_eq!(req.header("content-type"), Some("application/json"));
    let full = chat.get_conversation(&conv.id, true).unwrap();
    let msgs = full.messages.unwrap_or_default();
    assert_eq!(
        msgs.iter().map(|m| m["role"].clone()).collect::<Vec<_>>(),
        [json!("user"), json!("assistant")]
    );
    assert_eq!(msgs[1]["content"], "Hello there");
    let usage = env.container().usage.query("model", None, None).unwrap();
    assert!(usage[0].key == REPO && usage[0].requests == 1 && usage[0].prompt_tokens == 20);
}

#[tokio::test]
async fn test_chat_non_streaming_and_reasoning() {
    let mut env = TestEnv::new();
    let inst = ready(&mut env).await;
    env.node.serve(None, CHAT, engine());
    let c = env.container();
    let data = c
        .chat
        .complete(&json!({"instance_id": inst.id, "messages": messages()}))
        .await
        .unwrap();
    assert_eq!(data["choices"][0]["message"]["content"], "Hello there");
    let sent: Value = serde_json::from_slice(&env.engine_requests().last().unwrap().body).unwrap();
    assert_eq!(sent["stream"], json!(false));
    assert_eq!(
        c.usage.query("model", None, None).unwrap()[0].completion_tokens,
        3
    );
}

#[tokio::test]
async fn test_a_client_that_goes_away_mid_stream_cancels_the_node_request_and_still_records_usage()
{
    let mut env = TestEnv::new();
    let inst = ready(&mut env).await;
    env.node.st().http_chunk = 8;
    env.node.serve(None, CHAT, engine());
    let c = env.container();
    let mut stream = c
        .chat
        .open_stream(&json!({"instance_id": inst.id, "messages": messages()}))
        .await
        .unwrap();
    let first = stream.lines.next().await.unwrap();
    assert!(first.starts_with("data:"));
    drop(stream); // the browser closed the connection
    let node = Arc::clone(&env.node);
    until("the node request to be cancelled", || {
        node.streams().last().is_some_and(|s| s.is_cancelled())
    })
    .await;
    // the partial answer is still accounted for
    let usage = Arc::clone(&c.usage);
    until("the usage row", || {
        usage
            .query("model", None, None)
            .is_ok_and(|u| u.first().is_some_and(|r| r.requests == 1))
    })
    .await;
}

/// Audit F7: a stream dropped mid-reply no longer writes its record inside `Drop` (on the async worker, inside the
/// connection's poll): it hands it to the blocking pool. Dropping it returns at once even while something else holds
/// the database, and the partial reply (usage and the conversation's turn) is written once the database is free.
#[tokio::test]
async fn test_a_dropped_stream_hands_its_reply_off_and_never_writes_in_drop() {
    let mut env = TestEnv::new();
    let inst = ready(&mut env).await;
    env.node.st().http_chunk = 8;
    env.node.serve(None, CHAT, engine());
    let c = env.container();
    let conv = c
        .chat
        .create_conversation(&ConversationIn {
            title: "t".into(),
            system_prompt: None,
            instance_id: Some(inst.id.clone()),
        })
        .unwrap();
    let mut stream = c
        .chat
        .open_stream(
            &json!({"instance_id": inst.id, "conversation_id": conv.id, "messages": messages()}),
        )
        .await
        .unwrap();
    let first = stream.lines.next().await.unwrap();
    assert!(first.starts_with("data:"));
    // a slow statement holds the database for up to 5 s
    let store = Arc::clone(&c.store);
    let (held_tx, held_rx) = std::sync::mpsc::channel::<()>();
    let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
    let holder = std::thread::spawn(move || {
        let _ = store.transaction(|_c| {
            let _ = held_tx.send(());
            let _ = release_rx.recv_timeout(Duration::from_secs(5));
            Ok(())
        });
    });
    held_rx.recv_timeout(Duration::from_secs(10)).unwrap();
    let t = std::time::Instant::now();
    drop(stream); // the browser closed the connection
    let took = t.elapsed();
    let _ = release_tx.send(());
    holder.join().unwrap();
    assert!(
        took < Duration::from_secs(2),
        "dropping the stream waited {took:?} for the database"
    );
    // nothing of the reply is lost: the turn and the usage are written once the database is free
    let chat = Arc::clone(&c.chat);
    let cid = conv.id.clone();
    until("the conversation's turn", || {
        chat.get_conversation(&cid, true)
            .is_ok_and(|v| v.messages.is_some_and(|m| m.len() == 2))
    })
    .await;
    let msgs = chat
        .get_conversation(&conv.id, true)
        .unwrap()
        .messages
        .unwrap();
    assert_eq!(msgs[0]["role"], "user");
    assert_eq!(msgs[1]["role"], "assistant");
    let partial = msgs[1]["content"].as_str().unwrap();
    assert!(
        !partial.is_empty() && "Hello there".starts_with(partial),
        "{partial:?}"
    );
    let usage = c.usage.query("model", None, None).unwrap();
    assert!(usage[0].key == REPO && usage[0].requests == 1);
}

/// The lifespan's end waits for a reply that a dropped stream handed off: `flush` stays pending while the reply waits
/// for the database, `lifespan_stop` does not end before it is written, and once it returns the turn is there (read at
/// once, no polling). Multi-threaded, so the timer keeps running whatever waits for the database.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_the_lifespan_end_waits_for_a_reply_still_being_written() {
    let mut env = TestEnv::new();
    let inst = ready(&mut env).await;
    env.node.st().http_chunk = 8;
    env.node.serve(None, CHAT, engine());
    let c = env.container();
    let conv = c
        .chat
        .create_conversation(&ConversationIn {
            title: "t".into(),
            system_prompt: None,
            instance_id: Some(inst.id.clone()),
        })
        .unwrap();
    let mut stream = c
        .chat
        .open_stream(
            &json!({"instance_id": inst.id, "conversation_id": conv.id, "messages": messages()}),
        )
        .await
        .unwrap();
    assert!(stream.lines.next().await.unwrap().starts_with("data:"));
    let store = Arc::clone(&c.store);
    let (held_tx, held_rx) = std::sync::mpsc::channel::<()>();
    let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
    let holder = std::thread::spawn(move || {
        let _ = store.transaction(|_c| {
            let _ = held_tx.send(());
            let _ = release_rx.recv_timeout(Duration::from_secs(10));
            Ok(())
        });
    });
    held_rx.recv_timeout(Duration::from_secs(10)).unwrap();
    drop(stream);
    // the handed-off reply waits for the database, and so do the flush and the lifespan's end
    let pending = tokio::time::timeout(Duration::from_millis(300), c.chat.flush()).await;
    let c2 = Arc::clone(&c);
    let stop = tokio::spawn(async move { ec_hub::app::lifespan_stop(&c2, Vec::new()).await });
    tokio::time::sleep(Duration::from_millis(300)).await;
    let stopped_early = stop.is_finished();
    let _ = release_tx.send(());
    holder.join().unwrap();
    assert!(
        pending.is_err(),
        "flush returned while a reply was still waiting"
    );
    assert!(
        !stopped_early,
        "the lifespan ended while a reply was still waiting"
    );
    tokio::time::timeout(Duration::from_secs(10), stop)
        .await
        .unwrap()
        .unwrap();
    let msgs = c
        .chat
        .get_conversation(&conv.id, true)
        .unwrap()
        .messages
        .unwrap();
    assert_eq!(msgs.len(), 2, "{msgs:?}");
    assert_eq!(c.usage.query("model", None, None).unwrap()[0].requests, 1);
}

#[tokio::test]
async fn test_engine_error_statuses_are_mapped_and_unreachable_engines_are_502_problems() {
    let mut env = TestEnv::new();
    let inst = ready(&mut env).await;
    let chat = Arc::clone(&env.container().chat);
    let body = json!({"instance_id": inst.id, "messages": messages()});
    for (status, want) in [(400, 400), (404, 404), (422, 422), (500, 502), (503, 502)] {
        env.node.serve(
            None,
            CHAT,
            http_handler(move |_| HttpResp::text(status, "engine says no")),
        );
        let e = chat.complete(&body).await.unwrap_err();
        let p = problem(&mut env, &e);
        assert!(
            p.status == want && p.detail.contains("engine says no"),
            "{p:?}"
        );
        assert!(p.detail.contains(&format!("HTTP {status}")), "{p:?}");
        let mut streamed = body.clone();
        streamed["stream"] = json!(true);
        let e2 = chat.open_stream(&streamed).await.err().unwrap();
        assert_eq!(problem(&mut env, &e2).status, want);
    }
    env.node.fail(
        "http.request",
        NodeError::new("local", "http_error", "refused"),
    );
    let e3 = chat.complete(&body).await.unwrap_err();
    let p3 = problem(&mut env, &e3);
    assert!(
        p3.status == 502 && p3.code == "http_error" && p3.detail.contains("testbox"),
        "{p3:?}"
    );
    env.node.unfail("http.request");
    env.node.serve(
        None,
        CHAT,
        http_handler(|_| HttpResp::text(200, "<html>not json</html>")),
    );
    let e4 = chat.complete(&body).await.unwrap_err();
    assert_eq!(problem(&mut env, &e4).code, "upstream_error");
}

#[tokio::test]
async fn test_chat_to_an_instance_on_another_host_uses_that_hosts_node() {
    let mut env = TestEnv::new();
    let spark = env.add_host("spark", "Spark", HelloOpts::default());
    env.seed_local_model(REPO, None, 65_000_000_000, "spark");
    spark.serve(None, "/health", http_handler(|_| HttpResp::text(200, "ok")));
    spark.serve(None, CHAT, engine());
    let c = env.container();
    let inst = c
        .lifecycle
        .create(&create_req("spark"), None)
        .await
        .unwrap();
    c.lifecycle.tick().await.unwrap();
    let data = c
        .chat
        .complete(&json!({"instance_id": inst.id, "messages": messages()}))
        .await
        .unwrap();
    let on_spark = spark.http_requests();
    assert!(data["choices"].as_array().is_some_and(|a| !a.is_empty()));
    assert!(on_spark.last().is_some_and(|r| r.path == CHAT));
    assert!(!env.node.http_requests().iter().any(|r| r.path == CHAT));
}

#[tokio::test]
async fn test_chat_needs_a_ready_instance_and_a_known_one() {
    let mut env = TestEnv::new();
    env.seed_local_model(REPO, None, 65_000_000_000, "local");
    let c = env.container();
    let inst = c
        .lifecycle
        .create(&create_req("local"), None)
        .await
        .unwrap();
    let body = json!({"instance_id": inst.id, "messages": [{"role": "user", "content": "x"}]});
    let e1 = c.chat.complete(&body).await.unwrap_err();
    assert!(
        matches!(&e1, AppError::Problem(p) if p.code == "instance_not_ready" && p.status == 409),
        "{e1:?}"
    );
    let e2 = c
        .chat
        .complete(
            &json!({"instance_id": "inst_nope", "messages": [{"role": "user", "content": "x"}]}),
        )
        .await
        .unwrap_err();
    assert!(
        matches!(&e2, AppError::Problem(p) if p.status == 404),
        "{e2:?}"
    );
}

// ---- bench ----------------------------------------------------------------------------------------------------------
// one_request only needs a port the node relays to: a known engine endpoint here (Python takes a ready instance's).
const PORT: i64 = 18_000;

fn endpoint(env: &TestEnv) {
    env.node.add_endpoint(PORT, EndpointOpts::default());
}

#[tokio::test]
async fn test_one_request_measures_ttft_and_counts_tokens_through_the_node() {
    let mut env = TestEnv::new();
    endpoint(&env);
    env.node.serve(None, CHAT, engine());
    let http = Arc::clone(&env.container().nodehttp);
    let st = one_request(&http, "local", PORT, "m", "prompt", 8).await;
    assert!(st.error.is_none() && st.ttft.is_some(), "{st:?}");
    assert!(
        st.prompt_tokens == 20 && st.completion_tokens == 3 && st.e2e > 0.0,
        "{st:?}"
    );
    let req = env.engine_requests().last().cloned().unwrap();
    assert!(req.port == PORT && req.path == CHAT);
    let sent: Value = serde_json::from_slice(&req.body).unwrap();
    assert_eq!(sent["max_tokens"], 8);
    assert_eq!(sent["stream_options"], json!({"include_usage": true}));
}

#[tokio::test]
async fn test_one_request_reports_http_and_node_errors_instead_of_raising() {
    let mut env = TestEnv::new();
    endpoint(&env);
    let http = Arc::clone(&env.container().nodehttp);
    env.node
        .serve(None, CHAT, http_handler(|_| HttpResp::text(500, "x")));
    assert_eq!(
        one_request(&http, "local", PORT, "m", "p", 8)
            .await
            .error
            .as_deref(),
        Some("HTTP 500")
    );
    env.node
        .fail("http.request", NodeError::new("local", "timeout", "slow"));
    assert_eq!(
        one_request(&http, "local", PORT, "m", "p", 8)
            .await
            .error
            .as_deref(),
        Some("timeout")
    );
    assert!(ReqStat::default().error.is_none());
}

#[tokio::test]
async fn test_a_benchmark_runs_on_the_instances_host_and_stores_results() {
    let mut env = TestEnv::new();
    let inst = ready(&mut env).await;
    env.node.serve(None, CHAT, engine());
    let bench = Arc::clone(&env.container().bench);
    let run = bench
        .start(
            &inst.id,
            &json!({"concurrency": [1, 2], "prompt_tokens": 28, "max_tokens": 4, "rounds": 1, "prefix_cache": true}),
            false,
        )
        .await
        .unwrap();
    bench.wait(&run.id).await;
    let done = bench.get(&run.id).unwrap();
    assert_eq!(done.state, "completed", "{done:?}");
    let res = Value::Object(done.results);
    let levels: Vec<Value> = res["concurrency"]
        .as_array()
        .unwrap()
        .iter()
        .map(|l| l["concurrency"].clone())
        .collect();
    assert_eq!(levels, [json!(1), json!(2)]);
    assert_eq!(res["single"]["errors"], 0);
    assert!(res["concurrency"][1]["throughput_tps"].as_f64().unwrap() > 0.0);
    assert!(res["prefix_cache"].get("speedup").is_some());
    let used: std::collections::BTreeSet<i64> = env
        .engine_requests()
        .iter()
        .filter(|r| r.path == CHAT)
        .map(|r| r.port)
        .collect();
    assert_eq!(used.into_iter().collect::<Vec<_>>(), [inst.port.unwrap()]);
}

#[tokio::test]
async fn test_a_benchmark_with_a_failing_engine_completes_with_errors_counted() {
    let mut env = TestEnv::new();
    let inst = ready(&mut env).await;
    env.node
        .serve(None, CHAT, http_handler(|_| HttpResp::text(500, "x")));
    let bench = Arc::clone(&env.container().bench);
    let run = bench
        .start(
            &inst.id,
            &json!({"concurrency": [2], "prompt_tokens": 14, "max_tokens": 2, "rounds": 1}),
            false,
        )
        .await
        .unwrap();
    bench.wait(&run.id).await;
    let done = bench.get(&run.id).unwrap();
    let res = Value::Object(done.results.clone());
    assert!(
        done.state == "completed" && res["concurrency"][0]["errors"] == 2,
        "{done:?}"
    );
}

#[tokio::test]
async fn test_a_run_cut_off_by_shutdown_or_a_crash_is_closed_as_failed_not_left_running() {
    let mut env = TestEnv::new();
    let inst = ready(&mut env).await;
    env.node.serve(None, CHAT, engine());
    let c = env.container();
    let run = c
        .bench
        .start(
            &inst.id,
            &json!({"concurrency": [1], "prompt_tokens": 14, "max_tokens": 2, "rounds": 5}),
            false,
        )
        .await
        .unwrap();
    c.bench.shutdown().await; // cancelled before the task ever ran
    let done = c.bench.get(&run.id).unwrap();
    assert!(
        done.state == "failed"
            && done.finished_at.is_some()
            && done.error.as_deref().unwrap_or("").contains("interrupted"),
        "{done:?}"
    );
    c.store
        .execute(
            "UPDATE bench_runs SET state='running', error=NULL, finished_at=NULL WHERE id=?",
            &[&run.id],
        )
        .unwrap();
    c.bench.recover().unwrap(); // a crashed console leaves the same thing behind
    assert_eq!(c.bench.get(&run.id).unwrap().state, "failed");
}

#[test]
fn test_ports_helper_is_a_published_mapping() {
    assert_eq!(
        ports(18000, 8000, "0.0.0.0"),
        [json!({"host_ip": "0.0.0.0", "host_port": 18000, "container_port": 8000})]
    );
}
