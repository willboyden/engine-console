//! PORTING §A5, the lifecycle's tests: an exception an adapter raises never stops a non-HTTP caller. Python's
//! adapters raise and nobody in the adapter catches: the lifecycle preflight falls back to the requested GPUs when
//! the fit estimator's `memory_model` raises, the metrics consumer logs a push it cannot parse and the pull fallback
//! swallows it, and the supervisor tick logs one instance's failure and goes on with the others.
//!
//! The raising adapter below raises with `pycompat::raise`, as the engine adapters do; the services call the `try_*`
//! forms everywhere a non-HTTP caller reaches an adapter.

#![allow(
    clippy::expect_used,
    clippy::panic,
    clippy::unwrap_used,
    clippy::needless_pass_by_value,
    clippy::too_many_lines
)]

#[path = "common/kit.rs"]
mod kit;

use kit::fakefleet;

#[path = "common/wse_kit.rs"]
mod wse_kit;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use ec_hub::adapters::base::{
    Compat, EngineAdapter, FloatMap, Hardware, LaunchSpec, ModelInfo, Obj, ParamSpec,
};
use ec_hub::domain::errors::Problem;
use fakefleet::{FleetEnv, HttpResp, http_handler};
use kit::FakeAdapter;
use serde_json::json;
use wse_kit::{obj, request, seed, set_logs};

use ec_hub::adapters::pycompat::raise;

/// The fake engine under another id, with some of its methods raising.
struct Raising {
    id: &'static str,
    memory_model: bool,
    parse_metrics: bool,
    parse_startup_log: bool,
    /// the exception `build_launch` raises while the switch is on
    build_launch: Option<(&'static str, Arc<AtomicBool>)>,
}

impl EngineAdapter for Raising {
    fn id(&self) -> &str {
        self.id
    }
    fn display_name(&self) -> &str {
        "Raising Engine"
    }
    fn default_image(&self) -> &str {
        "fake/engine:1.0"
    }
    fn param_catalog(&self) -> Vec<ParamSpec> {
        FakeAdapter.param_catalog()
    }
    fn presets(&self) -> Vec<(String, Obj)> {
        FakeAdapter.presets()
    }
    fn validate(&self, params: &Obj) -> Vec<String> {
        FakeAdapter.validate(params)
    }
    fn build_launch(
        &self,
        m: &str,
        p: &Obj,
        hw: &Hardware,
        s: &str,
        h: &str,
    ) -> Result<LaunchSpec, Problem> {
        if let Some((kind, on)) = &self.build_launch
            && on.load(Ordering::SeqCst)
        {
            raise(kind, "the launch cannot be built");
        }
        FakeAdapter.build_launch(m, p, hw, s, h)
    }
    fn compatibility(&self, m: &ModelInfo, p: &Obj, hw: &Hardware) -> Vec<Compat> {
        FakeAdapter.compatibility(m, p, hw)
    }
    fn memory_model(&self, m: &ModelInfo, p: &Obj) -> FloatMap {
        if self.memory_model {
            raise("ValueError", "cannot estimate this architecture");
        }
        FakeAdapter.memory_model(m, p)
    }
    fn parse_metrics(&self, text: &str) -> FloatMap {
        if self.parse_metrics {
            raise("ValueError", "could not convert string to float: 'x'");
        }
        FakeAdapter.parse_metrics(text)
    }
    fn parse_startup_log(&self, line: &str) -> Option<Obj> {
        if self.parse_startup_log {
            raise("TypeError", "expected string or bytes-like object");
        }
        FakeAdapter.parse_startup_log(line)
    }
}

fn raising(
    id: &'static str,
    memory_model: bool,
    parse_metrics: bool,
    parse_startup_log: bool,
) -> Arc<dyn EngineAdapter> {
    Arc::new(Raising {
        id,
        memory_model,
        parse_metrics,
        parse_startup_log,
        build_launch: None,
    })
}

/// Python: `_resolve_gpus` catches any exception from the adapter's memory model and uses the requested GPUs.
#[tokio::test]
async fn test_preflight_falls_back_to_the_requested_gpus_when_memory_model_raises() {
    // Python's `_resolve_gpus` catches the memory model's exception for a model it has no info on ("an adapter's
    // memory model may not cope with an unknown model: use the request"): the library has no copy of this repo
    // (`model_missing`), so there is no info and no fit estimate, and the GPUs are the request's.
    let mut env = FleetEnv::new();
    env.adapters.push(raising("raisy", true, false, false));
    seed(&env, "local");
    let c = env.container();
    let (report, port) = c
        .lifecycle
        .preflight_full(
            &request(json!({"engine": "raisy", "repo_id": "org/not-in-the-library"})),
            None,
            None,
        )
        .await
        .expect("the preflight answers");
    assert!(report.gpu_ids.is_empty() && port == Some(18000)); // the request asked for no GPU in particular
    assert!(report.checks.iter().any(|ch| ch.code == "port"));
    assert!(report.checks.iter().any(|ch| ch.code == "model_missing")); // no copy, so no info and no fit estimate
    // With the model's info at hand Python goes on to the fit estimate (`assess_info`), whose own call of the memory
    // model is not caught: the preflight is the generic 500, as in Python.
    let known = c
        .lifecycle
        .preflight_full(&request(json!({"engine": "raisy"})), None, None)
        .await;
    match known {
        Err(ec_hub::domain::errors::AppError::Problem(p)) => assert_eq!(p.status, 500),
        Err(other) => panic!("expected the generic 500, got {other:?}"),
        Ok(_) => panic!("expected the generic 500, got a report"),
    }
}

/// A raising `parse_metrics` drops that engine's scrape; the consumer and the poller go on with the others.
#[tokio::test]
async fn test_a_raising_parse_metrics_stops_neither_the_metrics_poller_nor_the_consumer() {
    let mut env = FleetEnv::new();
    env.adapters.push(raising("raisy", false, true, false));
    seed(&env, "local");
    env.engine_up();
    env.engine_route(
        "/metrics",
        http_handler(|_| HttpResp::text(200, "running 2\n")),
    );
    let c = env.container();
    let bad = c
        .lifecycle
        .create(&request(json!({"engine": "raisy", "gpu_ids": [0]})), None)
        .await
        .unwrap();
    let good = c
        .lifecycle
        .create(&request(json!({"name": "good", "gpu_ids": [1]})), None)
        .await
        .unwrap();
    c.lifecycle.tick().await.unwrap();
    // the pull fallback: both are ready and silent; the raising parser is swallowed, the other engine recorded
    c.metrics.scrape_once().await;
    assert!(c.metrics.latest(&bad.id).is_none());
    assert_eq!(
        c.metrics.latest(&good.id).unwrap().values["requests_running"],
        json!(2.0)
    );
    // the consumer: a push it cannot parse is logged, the next one is routed
    let push = |port: i64| {
        obj(json!({"type": "metrics", "port": port, "status": 200, "text": "running 5\n"}))
    };
    c.metrics.ingest("local", &push(bad.port.unwrap()));
    c.metrics.ingest("local", &push(good.port.unwrap()));
    assert!(c.metrics.latest(&bad.id).is_none());
    assert_eq!(
        c.metrics.latest(&good.id).unwrap().values["requests_running"],
        json!(5.0)
    );
}

/// A raising `parse_startup_log` fails that instance's supervisor round only: the tick goes on with the others.
#[tokio::test]
async fn test_a_raising_parse_startup_log_does_not_stop_the_supervisor_tick() {
    let mut env = FleetEnv::new();
    env.adapters.push(raising("raisy", false, false, true));
    seed(&env, "local");
    let c = env.container();
    let bad = c
        .lifecycle
        .create(&request(json!({"engine": "raisy", "gpu_ids": [0]})), None)
        .await
        .unwrap();
    let good = c
        .lifecycle
        .create(&request(json!({"name": "good", "gpu_ids": [1]})), None)
        .await
        .unwrap();
    for i in [&bad, &good] {
        set_logs(
            &env.node,
            i.container_name.as_deref().unwrap(),
            "loading weights 30%\n",
        );
    }
    c.lifecycle.sync_links().await; // the first look at the host adopts both (they run): `loading`
    c.lifecycle
        .tick()
        .await
        .expect("the tick itself never fails");
    let b = c.lifecycle.get(&bad.id).unwrap();
    assert!(b.state == "loading" && b.progress_pct.is_none()); // its round ended at the raise
    let g = c.lifecycle.get(&good.id).unwrap();
    assert!(g.state == "loading" && g.progress_pct == Some(30.0));
}

fn build_raising(
    id: &'static str,
    kind: &'static str,
    on: &Arc<AtomicBool>,
) -> Arc<dyn EngineAdapter> {
    Arc::new(Raising {
        id,
        memory_model: false,
        parse_metrics: false,
        parse_startup_log: false,
        build_launch: Some((kind, Arc::clone(on))),
    })
}

/// Python: `_build` turns a ValueError of the adapter into 422 invalid_params, which preflight shows as a check.
#[tokio::test]
async fn test_a_value_error_in_the_launch_build_is_an_invalid_params_check() {
    let mut env = FleetEnv::new();
    let on = Arc::new(AtomicBool::new(true));
    env.adapters.push(build_raising("raisy", "ValueError", &on));
    seed(&env, "local");
    let r = env
        .post("/api/v1/instances/preflight")
        .json(&json!({"engine": "raisy", "repo_id": wse_kit::QWEN, "gpu_ids": [0]}))
        .send()
        .await;
    assert_eq!(r.status, 200, "{}", r.text());
    let checks = r.json()["checks"].clone();
    let c = checks
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["code"] == "invalid_params")
        .unwrap()
        .clone();
    assert!(c["level"] == "block" && c["message"] == "the launch cannot be built");
}

/// Python: any other exception of the adapter propagates: over HTTP it is FastAPI's generic 500.
#[tokio::test]
async fn test_another_exception_in_the_launch_build_is_the_generic_500_over_http() {
    let mut env = FleetEnv::new();
    let on = Arc::new(AtomicBool::new(true));
    env.adapters.push(build_raising("raisy", "TypeError", &on));
    seed(&env, "local");
    for path in ["/api/v1/instances/preflight", "/api/v1/instances"] {
        let r = env
            .post(path)
            .json(&json!({"engine": "raisy", "repo_id": wse_kit::QWEN, "gpu_ids": [0]}))
            .send()
            .await;
        assert_eq!(r.status, 500, "{path}: {}", r.text());
        assert_eq!(r.json()["detail"], "unexpected server error");
        assert!(!r.text().contains("TypeError") && !r.text().contains("cannot be built")); // never echoed
    }
    assert!(env.node.ops("docker.run").is_empty());
}

/// A raise in the launch build the supervisor needs (the health path of an adopted engine) fails that instance's
/// round only (Python: `launch_for` does not catch a TypeError; tick logs it and supervises the others).
#[tokio::test]
async fn test_a_raising_launch_build_does_not_stop_the_supervisor_tick() {
    let mut env = FleetEnv::new();
    let on = Arc::new(AtomicBool::new(true));
    env.adapters.push(build_raising("raisy", "TypeError", &on));
    seed(&env, "local");
    env.engine_up();
    let labels = |iid: &str, engine: &str, port: &str| {
        json!({"engine-console.instance": iid, "engine-console.engine": engine, "engine-console.model": wse_kit::QWEN,
               "engine-console.port": port, "engine-console.name": iid, "engine-console.role": "engine",
               "engine-console.gpu_ids": "0"})
    };
    wse_kit::add_managed(
        &env.node,
        "ec-bad",
        labels("inst_bad", "raisy", "18010"),
        "fake/engine:1.0",
    );
    wse_kit::add_managed(
        &env.node,
        "ec-good",
        labels("inst_good", "fake", "18011"),
        "fake/engine:1.0",
    );
    // their gateways publish the ports the probe goes through
    for (name, iid, port) in [
        ("ec-bad-gw", "inst_bad", 18010),
        ("ec-good-gw", "inst_good", 18011),
    ] {
        env.node.add_container(
            name,
            fakefleet::ContainerOpts {
                managed: true,
                labels: obj(
                    json!({"engine-console.role": "gateway", "engine-console.instance": iid}),
                ),
                host_ports: fakefleet::ports(port, 8080, "127.0.0.1"),
                ..fakefleet::ContainerOpts::default()
            },
        );
    }
    let c = env.container();
    assert_eq!(c.lifecycle.adopt(None).await.unwrap(), 2);
    c.lifecycle
        .tick()
        .await
        .expect("the tick itself never fails");
    assert_eq!(c.lifecycle.get("inst_bad").unwrap().state, "loading"); // its round ended at the raise
    assert_eq!(c.lifecycle.get("inst_good").unwrap().state, "ready");
}
