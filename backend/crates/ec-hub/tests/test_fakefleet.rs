//! Self-tests of the fake fleet (tests/common/fakefleet.rs, port of tests/core/fakefleet.py).
//! The Python fake has no tests of its own; these pin the behaviour the ported tests rely on.

#![allow(
    clippy::expect_used,
    clippy::panic,
    clippy::unwrap_used,
    clippy::needless_pass_by_value,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::double_ended_iterator_last,
    clippy::assertions_on_constants,
    clippy::type_complexity,
    clippy::too_many_lines,
    clippy::float_cmp
)]

#[path = "common/kit.rs"]
mod kit;

// the fake fleet is the kit's submodule (PORTING.md §A4): one copy of its types per test binary
use kit::fakefleet;

use std::sync::Arc;
use std::time::Duration;

use ec_hub::domain::fleet::{FleetPort, NodeClient, StreamItem};
use ec_hub::domain::refs::ModelRef;
use ec_hub::services::libport::LibraryPort;
use fakefleet::{
    ContainerOpts, EndpointOpts, FakeFleet, FakeLibrary, FakeNode, FleetEnv, Gate, GpuFree,
    HelloOpts, HttpResp, ScriptRun, big_model_script, http_handler,
};
use serde_json::{Value, json};

const T: Duration = Duration::from_secs(5);

fn o(v: Value) -> ec_hub::domain::errors::Obj {
    v.as_object().cloned().unwrap_or_default()
}

fn node() -> Arc<FakeNode> {
    FakeNode::new(
        FakeNode::host_config("local", "testbox", None),
        HelloOpts::default(),
    )
}

fn run_spec(name: &str) -> Value {
    json!({"spec": {"name": name, "image": "fake/engine:1.0", "argv": ["--port", "8000"],
                    "labels": {"engine-console.instance": "i1", "engine-console.engine": "fake"},
                    "gpu_uuids": ["GPU-uuid-0"], "mounts": [{"root": "hf", "rel": "hub", "dest": "/models", "ro": true}]}})
}

#[tokio::test]
async fn test_fake_hello_sample_and_link() {
    let n = node();
    let hello = n.latest("hello").expect("hello");
    assert_eq!(hello["host"]["hostname"], "testbox");
    assert_eq!(hello["gpus"].as_array().map(Vec::len), Some(2));
    n.set_gpu_free(GpuFree::Each(vec![90.0, 40.0]));
    let s = n.latest("sample").expect("sample");
    let g0 = &s["gpus"]["GPU-uuid-0"];
    assert_eq!(g0["mem_used_bytes"], json!(6 * 1_073_741_824_i64));
    assert_eq!(s["mem"]["total"], json!(246 * 1_073_741_824_i64));
    // a link that is down keeps showing the last sample it had
    n.set_state("down", Some("no route to host"));
    assert_eq!(n.link().state, "down");
    assert_eq!(n.link().connected_since, None);
    assert_eq!(n.latest("sample"), Some(s));
    let e = n.call("node.info", None, T).await.unwrap_err();
    assert_eq!(
        (e.code.as_str(), e.message.as_str()),
        ("node_unavailable", "no route to host")
    );
}

#[tokio::test]
async fn test_fake_gate_refuses_what_a_node_would_drop_and_monitor_mode() {
    let n = node();
    let e = n
        .call(
            "docker.inspect",
            Some(o(json!({"name": "x", "nmae": "y"}))),
            T,
        )
        .await
        .unwrap_err();
    assert_eq!(e.code, "bad_args");
    assert!(e.message.contains("would drop nmae"), "{}", e.message);
    let e = n
        .call("docker.stop", Some(o(json!({}))), T)
        .await
        .unwrap_err();
    assert_eq!(
        (e.code.as_str(), e.message.as_str()),
        ("bad_args", "docker.stop: missing name")
    );
    let e = n.call("no.such", None, T).await.unwrap_err();
    assert_eq!(e.code, "unknown_op");
    n.set_mode("monitor");
    let e = n
        .call("docker.run", Some(o(run_spec("ec-a"))), T)
        .await
        .unwrap_err();
    assert_eq!(e.code, "monitor_only");
    assert!(n.call("docker.images", None, T).await.is_ok());
    assert_eq!(n.ops("docker.run").len(), 1);
}

#[tokio::test]
async fn test_fake_run_gateway_stop_rm_and_inventory() {
    let n = node();
    let r = n
        .call("docker.run", Some(o(run_spec("ec-a"))), T)
        .await
        .unwrap();
    assert_eq!(r["id"].as_str().map(str::len), Some(12));
    let inv = n.latest("inventory").expect("inventory");
    let c = &inv["containers"][0];
    assert_eq!(
        (c["name"].as_str(), c["managed"].as_bool()),
        (Some("ec-a"), Some(true))
    );
    assert_eq!(c["mounts"][0]["source"], "/data/hf/hub");
    let again = n
        .call("docker.run", Some(o(run_spec("ec-a"))), T)
        .await
        .unwrap_err();
    assert_eq!(again.code, "conflict");
    let gw = json!({"name": "ec-a-gw", "instance": "i1", "host_port": 18001, "engine_host": "ec-a", "engine_port": 8000});
    n.call("gateway.start", Some(o(gw.clone())), T)
        .await
        .unwrap();
    let busy = n.call("gateway.start", Some(o(gw)), T).await.unwrap_err();
    assert_eq!(busy.code, "port_in_use");
    // ownership: a container the console did not create
    n.add_container("other", ContainerOpts::default());
    let e = n
        .call("docker.stop", Some(o(json!({"name": "other"}))), T)
        .await
        .unwrap_err();
    assert_eq!(e.code, "not_owned");
    n.call("docker.stop", Some(o(json!({"name": "ec-a-gw"}))), T)
        .await
        .unwrap();
    assert!(!n.st().port_taken.contains(&18001));
    n.call("docker.rm", Some(o(json!({"name": "ec-a-gw"}))), T)
        .await
        .unwrap();
    assert!(!n.st().containers.contains_key("ec-a-gw"));
    let names: Vec<String> = n.st().containers.keys().cloned().collect();
    assert_eq!(names, ["ec-a", "other"]);
}

#[tokio::test]
async fn test_fake_http_request_with_body_and_port_rules() {
    let n = node();
    let mut refused = n.open(
        "http.request",
        Some(o(json!({"port": 9999, "method": "GET", "path": "/health"}))),
    );
    assert!(refused.next().await.is_none());
    assert_eq!(refused.result().await.unwrap_err().code, "policy_denied");
    n.add_endpoint(8000, EndpointOpts::default());
    n.serve(
        None,
        "/v1/chat/completions",
        http_handler(|r| HttpResp::json(200, &json!({"echo": r.json()}))),
    );
    let body = br#"{"a": 1}"#;
    let mut s = n.open(
        "http.request",
        Some(o(
            json!({"port": 8000, "method": "POST", "path": "/v1/chat/completions",
                       "body_len": body.len()}),
        )),
    );
    s.write(body).await.unwrap();
    s.end().await.unwrap();
    let mut got = Vec::new();
    let mut status = None;
    while let Some(item) = s.next().await {
        match item {
            StreamItem::Event(e) => status = e.get("status").cloned(),
            StreamItem::Data(d) => got.extend_from_slice(&d),
        }
    }
    assert_eq!(status, Some(json!(200)));
    assert_eq!(
        serde_json::from_slice::<Value>(&got).unwrap(),
        json!({"echo": {"a": 1}})
    );
    assert_eq!(s.result().await.unwrap()["bytes"], json!(got.len()));
    assert_eq!(n.http_requests()[0].json(), json!({"a": 1}));
}

#[tokio::test]
async fn test_fake_scripts_run_gate_cancel_and_busy() {
    let n = FakeNode::new(
        FakeNode::host_config("local", "testbox", None),
        HelloOpts {
            scripts: vec![big_model_script()],
            ..HelloOpts::default()
        },
    );
    let gate = Gate::new();
    n.script_run(
        "big-model",
        "up",
        ScriptRun {
            gate: Some(gate.clone()),
            ..ScriptRun::lines(&[("out", "one"), ("out", "two")])
        },
    );
    let args = o(json!({"name": "big-model", "action": "up"}));
    let mut s = n.open("scripts.run", Some(args.clone()));
    assert!(matches!(s.next().await, Some(StreamItem::Event(_))));
    let mut second = n.open("scripts.run", Some(args.clone()));
    assert!(second.next().await.is_none());
    assert_eq!(second.result().await.unwrap_err().code, "busy");
    let waiting = tokio::time::timeout(Duration::from_millis(50), s.next()).await;
    assert!(waiting.is_err(), "the run waits at its gate");
    gate.set();
    assert!(matches!(s.next().await, Some(StreamItem::Event(_))));
    assert!(s.next().await.is_none());
    assert_eq!(s.result().await.unwrap()["rc"], json!(0));
    assert!(n.st().running_scripts.is_empty());
    let bad = n.open(
        "scripts.run",
        Some(o(
            json!({"name": "big-model", "action": "switch", "arg": "model-z"}),
        )),
    );
    drop(bad);
    assert_eq!(n.st().script_calls.len(), 1);
    // dropping a started run counts as a cancel and frees the script
    let mut s = n.open("scripts.run", Some(args));
    let _ = s.next().await;
    drop(s);
    assert!(n.streams().last().unwrap().is_cancelled());
    assert!(n.st().running_scripts.is_empty());
}

#[tokio::test]
async fn test_fake_fleet_and_library() {
    let a = node();
    let fleet = Arc::new(FakeFleet::new(vec![Arc::clone(&a)]));
    let e = fleet.node("ghost").err().expect("unknown");
    assert_eq!(
        (e.status, e.code.as_str(), e.detail.as_str()),
        (404, "unknown_host", "no host 'ghost' is configured")
    );
    let lib = FakeLibrary::new(Arc::clone(&fleet));
    let c = lib.add_hf("local", "Qwen/Qwen3-32B", &json!({}), 100, "hf", None, None);
    assert_eq!(c.path, "/data/hf/hub/models--Qwen--Qwen3-32B");
    assert_eq!(c.id.len(), 16);
    assert_eq!(lib.find("local", &ModelRef::hf("Qwen/Qwen3-32B")), Some(c));
    assert_eq!(lib.hf_root("local").map(|r| r.id), Some("hf".into()));
    assert_eq!(lib.ollama_root("local").map(|r| r.writable), Some(false));
    lib.remove("local", &ModelRef::hf("Qwen/Qwen3-32B"));
    assert!(lib.copies("local").is_empty());
}

#[tokio::test]
async fn test_fleet_env_builds_the_app_on_the_fake_fleet() {
    let mut env = FleetEnv::new();
    let spark = env.add_host(
        "spark",
        "Spark",
        HelloOpts {
            gpus: 1,
            ..HelloOpts::default()
        },
    );
    assert_eq!(spark.latest("hello").unwrap()["host"]["hostname"], "Spark");
    let c = env.container();
    assert_eq!(c.fleet.hosts().len(), 2);
    let r = env.get("/api/v1/health").send().await;
    assert_eq!(r.status, 200);
    env.engine_up();
    env.node.add_endpoint(8000, EndpointOpts::default());
    let mut s = env.node.open(
        "http.request",
        Some(o(json!({"port": 8000, "method": "GET", "path": "/health"}))),
    );
    while s.next().await.is_some() {}
    assert_eq!(s.result().await.unwrap()["status"], json!(200));
    assert_eq!(env.engine_requests().len(), 1);
}
