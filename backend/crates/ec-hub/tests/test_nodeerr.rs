//! Port of backend/tests/core/test_nodeerr.py, the rows that need a fleet, a node, the log or the HTTP layer (the pure
//! table rows are unit tests in services/nodeerr.rs). `shielded` has no Rust counterpart: a cleanup that must outlive
//! a cancelled caller is spawned as its own task (backend/TEST-MAP.md marks those two rows N/A).

#![allow(clippy::panic, clippy::expect_used)]

#[path = "common/kit.rs"]
mod kit;

use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use ec_hub::api::http::{Fail, fail_response};
use ec_hub::domain::errors::{AppError, NodeError, Obj, Problem, not_found};
use ec_hub::domain::fleet::{FleetPort, HostConfig, LinkStatus, NodeClient, NodeStream};
use ec_hub::services::nodeerr::{NodeCaller, problem_from_node};
use http_body_util::BodyExt;
use kit::TestEnv;
use serde_json::{Value, json};

fn problem(r: Result<Obj, AppError>) -> Problem {
    match r {
        Err(AppError::Problem(p)) => p,
        other => panic!("expected a problem, got {other:?}"),
    }
}

/// A log sink for `tracing` (Python's caplog).
#[derive(Clone, Default)]
struct Captured(Arc<Mutex<Vec<u8>>>);

impl Write for Captured {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().expect("lock").extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
fn test_hub_bug_codes_are_logged_as_errors() {
    let log = Captured::default();
    let sink = log.clone();
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::ERROR)
        .with_ansi(false)
        .with_writer(move || sink.clone())
        .finish();
    tracing::subscriber::with_default(subscriber, || {
        problem_from_node(&NodeError::new("a", "bad_args", "missing field"), Some("A"));
        problem_from_node(&NodeError::new("a", "unknown_op", "no such op"), Some("A"));
        problem_from_node(
            &NodeError::new("a", "docker_error", "daemon down"),
            Some("A"),
        );
    });
    let text = String::from_utf8_lossy(&log.0.lock().expect("lock")).into_owned();
    assert_eq!(text.matches("hub bug").count(), 2, "{text}");
}

#[tokio::test]
async fn test_the_caller_translates_labels_and_resolves_hosts() {
    let env = TestEnv::new();
    let caller = NodeCaller::new(Arc::clone(&env.fleet) as Arc<dyn FleetPort>);
    assert_eq!(
        (
            caller.label("local").as_str(),
            caller.label("nope").as_str()
        ),
        ("testbox", "nope")
    );
    let fail = Arc::new(AtomicBool::new(true));
    let f = Arc::clone(&fail);
    env.node.on(
        "docker.images",
        Arc::new(move |_, _| {
            if f.load(Ordering::SeqCst) {
                Err(NodeError::new("local", "docker_error", "daemon down"))
            } else {
                Ok(json!({"images": []})
                    .as_object()
                    .cloned()
                    .unwrap_or_default())
            }
        }),
    );
    let p = problem(caller.call("local", "docker.images", None, 30.0).await);
    assert_eq!((p.status, p.detail.as_str()), (502, "testbox: daemon down"));
    let p = problem(caller.call("nope", "docker.images", None, 30.0).await);
    assert_eq!((p.status, p.code.as_str()), (404, "unknown_host"));
    fail.store(false, Ordering::SeqCst);
    assert_eq!(
        Value::Object(
            caller
                .call("local", "docker.images", None, 30.0)
                .await
                .expect("images")
        ),
        json!({"images": []})
    );
}

/// A node whose calls never answer.
struct NeverNode(HostConfig);

#[async_trait]
impl NodeClient for NeverNode {
    fn host(&self) -> &HostConfig {
        &self.0
    }
    fn link(&self) -> LinkStatus {
        LinkStatus::new("up", None)
    }
    fn latest(&self, _kind: &str) -> Option<Arc<Obj>> {
        None
    }
    async fn call(
        &self,
        _op: &str,
        _args: Option<Obj>,
        _timeout: Duration,
    ) -> Result<Obj, NodeError> {
        tokio::time::sleep(Duration::from_secs(3600)).await;
        Ok(Obj::new())
    }
    fn open(&self, _op: &str, _args: Option<Obj>) -> Box<dyn NodeStream> {
        panic!("not used")
    }
}

struct OneNode(Arc<NeverNode>);

impl FleetPort for OneNode {
    fn hosts(&self) -> Vec<HostConfig> {
        vec![self.0.0.clone()]
    }
    fn node(&self, host_id: &str) -> Result<Arc<dyn NodeClient>, Problem> {
        if host_id == self.0.0.id {
            Ok(Arc::clone(&self.0) as Arc<dyn NodeClient>)
        } else {
            Err(not_found(format!("no host '{host_id}' is configured")).code("unknown_host"))
        }
    }
}

#[tokio::test]
async fn test_a_client_that_never_answers_is_a_504_not_a_hang() {
    let host = HostConfig {
        id: "local".into(),
        label: "testbox".into(),
        ssh: None,
        port: None,
        ssh_config: false,
        enabled: true,
    };
    let caller = NodeCaller::new(Arc::new(OneNode(Arc::new(NeverNode(host)))));
    let t0 = std::time::Instant::now();
    let p = problem(
        caller
            .call_with_slack("local", "docker.images", None, 0.05, 0.05)
            .await,
    );
    assert_eq!((p.status, p.code.as_str()), (504, "timeout"));
    assert!(t0.elapsed() < Duration::from_secs(5));
}

#[tokio::test]
async fn test_the_global_handler_formats_untranslated_errors() {
    let mut env = TestEnv::new();
    let c = env.container();
    let body = |r: ec_hub::api::http::Resp| async move {
        let ct = r
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        let status = r.status().as_u16();
        let bytes = r.into_body().collect().await.expect("body").to_bytes();
        (
            status,
            ct,
            serde_json::from_slice::<Value>(&bytes).expect("json"),
        )
    };
    let (status, ct, j) = body(fail_response(
        &c,
        Fail::App(AppError::Node(NodeError::new(
            "local",
            "policy_denied",
            "not allowed here",
        ))),
    ))
    .await;
    assert_eq!(status, 403);
    assert!(ct.starts_with("application/problem+json"), "{ct}");
    assert_eq!(
        (&j["code"], &j["detail"]),
        (&json!("policy_denied"), &json!("testbox: not allowed here"))
    );
    assert_eq!(
        (&j["host_id"], &j["status"]),
        (&json!("local"), &json!(403))
    );
    // an unknown host shows its id
    let (status, _, j) = body(fail_response(
        &c,
        Fail::App(AppError::Node(NodeError::new(
            "ghost",
            "timeout",
            "no answer",
        ))),
    ))
    .await;
    assert_eq!((status, &j["detail"]), (504, &json!("ghost: no answer")));
}
