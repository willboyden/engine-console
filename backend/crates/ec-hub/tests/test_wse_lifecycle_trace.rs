//! Replays tests/data/wse_lifecycle_trace.json (backend/parity/gen/wse_lifecycle_trace.py): one managed instance's
//! whole life through the Python hub's LifecycleService over the Python fake node. The Rust service, driven through
//! the same steps over the Rust fake node, must send the node the same requests with the same arguments (the
//! docker.run spec, gateway.start, docker.stop/rm with role and instance, docker.logs, docker.render) and answer the
//! same instances, logs, snippets and lists. Normalised: the random instance id (and its 6-character tail in the
//! container name) and wall-clock times.

#![allow(
    clippy::expect_used,
    clippy::panic,
    clippy::unwrap_used,
    clippy::too_many_lines,
    clippy::needless_pass_by_value
)]

#[path = "common/kit.rs"]
mod kit;

use kit::fakefleet;

#[path = "common/wse_kit.rs"]
mod wse_kit;

use ec_hub::domain::models::{InstanceCreate, InstancePatch};
use ec_hub::domain::refs::ModelRef;
use fakefleet::{CopyOpts, FleetEnv, HttpResp, http_handler};
use serde_json::{Value, json};
use wse_kit::{SECRET, add_managed, set_logs, update_container};

const TIME_KEYS: &[&str] = &[
    "created_at",
    "started_at",
    "last_request_at",
    "uptime_s",
    "history_since",
];

fn normalise(v: &Value, iid: &str) -> Value {
    let tail = &iid[iid.len() - 6..];
    let text = serde_json::to_string(v)
        .unwrap()
        .replace(iid, "inst_X")
        .replace(tail, "TAIL");
    let mut v: Value = serde_json::from_str(&text).unwrap();
    blank_times(&mut v);
    v
}

fn blank_times(v: &mut Value) {
    match v {
        Value::Object(o) => {
            for (k, x) in o.iter_mut() {
                if TIME_KEYS.contains(&k.as_str()) && x.is_number() {
                    *x = json!("<ts>");
                } else {
                    blank_times(x);
                }
            }
        }
        Value::Array(a) => a.iter_mut().for_each(blank_times),
        _ => {}
    }
}

struct Recorder {
    seen: usize,
    steps: Vec<Value>,
}

impl Recorder {
    fn record(&mut self, env: &FleetEnv, name: &str, value: Value) {
        let calls = env.node.calls();
        let new: Vec<Value> = calls[self.seen..]
            .iter()
            .map(|(op, a)| json!([op, a]))
            .collect();
        self.seen = calls.len();
        self.steps
            .push(json!({"step": name, "value": value, "calls": new}));
    }
}

#[tokio::test]
async fn test_a_managed_instances_life_matches_python_request_for_request() {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/wse_lifecycle_trace.json"
    ))
    .expect("trace");
    let want: Vec<Value> = serde_json::from_str(&text).unwrap();
    let py_iid = want[0]["value"]["id"].as_str().unwrap().to_string();

    let mut env = FleetEnv::new();
    let c = env.container();
    let life = &c.lifecycle;
    let mut rec = Recorder {
        seen: 0,
        steps: Vec::new(),
    };
    env.library.add(
        "local",
        &ModelRef::hf("Qwen/Qwen3-32B"),
        CopyOpts::new("hf", "hub/models--Qwen--Qwen3-32B"),
    );
    let req: InstanceCreate = serde_json::from_value(json!({"engine": "fake", "repo_id": "Qwen/Qwen3-32B", "gpu_ids": [1],
        "name": "trace me", "params": {"max_model_len": 4096, "api_key": SECRET, "kv_dtype": "fp8"}}))
    .unwrap();
    let inst = life.create(&req, None).await.unwrap();
    rec.record(&env, "create", serde_json::to_value(&inst).unwrap());
    let cn = inst.container_name.clone().unwrap();
    set_logs(&env.node, &cn, "boot\nloading weights 55%\n");
    life.tick().await.unwrap();
    rec.record(
        &env,
        "tick-loading",
        serde_json::to_value(life.get(&inst.id).unwrap()).unwrap(),
    );
    env.engine_route("/health", http_handler(|_| HttpResp::text(200, "ok")));
    life.tick().await.unwrap();
    rec.record(
        &env,
        "tick-ready",
        serde_json::to_value(life.get(&inst.id).unwrap()).unwrap(),
    );
    set_logs(
        &env.node,
        &cn,
        &format!("key {SECRET} token hf_abcdefghijkl1234\nsecond\n"),
    );
    rec.record(&env, "logs", json!(life.logs(&inst.id, 50).await.unwrap()));
    rec.record(
        &env,
        "command",
        serde_json::to_value(life.command(&inst.id).await.unwrap()).unwrap(),
    );
    let patch = InstancePatch {
        pinned: Some(true),
        name: Some("renamed".into()),
        ttl_idle_s: Some(600),
    };
    rec.record(
        &env,
        "patch",
        serde_json::to_value(life.patch(&inst.id, &patch).unwrap()).unwrap(),
    );
    rec.record(
        &env,
        "stop",
        serde_json::to_value(life.stop(&inst.id).await.unwrap()).unwrap(),
    );
    rec.record(
        &env,
        "start",
        serde_json::to_value(life.start(&inst.id).await.unwrap()).unwrap(),
    );
    rec.record(
        &env,
        "restart",
        serde_json::to_value(life.restart(&inst.id).await.unwrap()).unwrap(),
    );
    set_logs(&env.node, &cn, "CUDA error: out of memory\n");
    update_container(&env.node, &cn, json!({"state": "exited", "exit_code": 137}));
    life.tick().await.unwrap();
    rec.record(
        &env,
        "crash",
        serde_json::to_value(life.get(&inst.id).unwrap()).unwrap(),
    );
    life.remove(&inst.id).await.unwrap();
    rec.record(&env, "remove", Value::Null);
    add_managed(
        &env.node,
        "ec-old-abc123",
        json!({"engine-console.instance": "inst_old", "engine-console.engine": "fake",
               "engine-console.model": "Qwen/Qwen3-32B", "engine-console.port": "18007", "engine-console.name": "old",
               "engine-console.gpus": "GPU-uuid-1", "engine-console.role": "engine"}),
        "fake/engine:1.0",
    );
    rec.record(&env, "adopt", json!(life.adopt(None).await.unwrap()));
    rec.record(
        &env,
        "adopted",
        serde_json::to_value(life.get("inst_old").unwrap()).unwrap(),
    );
    rec.record(
        &env,
        "list",
        serde_json::to_value(life.list_page(100, None, None).unwrap()).unwrap(),
    );

    assert_eq!(rec.steps.len(), want.len());
    for (got, want) in rec.steps.iter().zip(&want) {
        let (g, w) = (normalise(got, &inst.id), normalise(want, &py_iid));
        assert_eq!(g["step"], w["step"]);
        assert_eq!(
            g["calls"], w["calls"],
            "node requests of step {}",
            w["step"]
        );
        assert_eq!(g["value"], w["value"], "answer of step {}", w["step"]);
    }
}
