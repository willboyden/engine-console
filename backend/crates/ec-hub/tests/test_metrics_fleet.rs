//! Port of backend/tests/core/test_metrics_fleet.py: nodes are configured on link-up, pushed scrapes are routed to
//! instances per host, per-host system series, the pull fallback, the bus, Prometheus gauges.

#![allow(
    clippy::redundant_clone,
    clippy::cloned_ref_to_slice_refs,
    clippy::uninlined_format_args,
    clippy::manual_let_else,
    clippy::expect_used,
    clippy::panic,
    clippy::unwrap_used,
    clippy::too_many_lines,
    clippy::needless_pass_by_value,
    clippy::float_cmp,
    clippy::too_many_arguments,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::type_complexity
)]

#[path = "common/kit.rs"]
mod kit;

use kit::fakefleet;

#[path = "common/wse_kit.rs"]
mod wse_kit;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use ec_hub::adapters::AdapterRegistry;
use ec_hub::adapters::base::{
    Compat, EngineAdapter, FloatMap, Hardware, LaunchSpec, ModelInfo, Obj, ParamSpec,
};
use ec_hub::container::Container;
use ec_hub::domain::errors::{NodeError, Problem};
use ec_hub::domain::models::Instance;
use ec_hub::services::metrics::{MetricsService, PREFIXES};
use fakefleet::{
    ContainerOpts, EndpointOpts, FleetEnv, HelloOpts, HttpResp, cgroup_default, http_handler,
    ports, until,
};
use kit::{FakeAdapter, FakeMultiAdapter};
use serde_json::{Value, json};
use wse_kit::{P, obj, request};

const TEXT1: &str = "running 2\nwaiting 1\nprompt_total 100\ngen_total 50\nkv 12.5\n";
const TEXT2: &str = "running 3\nwaiting 0\nprompt_total 300\ngen_total 150\nkv 20\nunknown_key 9\n";

type Clock = Arc<Mutex<f64>>;

fn use_clock(c: &Container) -> Clock {
    let clock = Arc::new(Mutex::new(10_000.0_f64));
    let k = Arc::clone(&clock);
    c.metrics.set_clock(Arc::new(move || *k.lock().unwrap()));
    clock
}

fn tick(clock: &Clock, dt: f64) {
    *clock.lock().unwrap() += dt;
}

fn push(port: Option<i64>, container: Option<&str>, text: &str, status: i64, engine: &str) -> Obj {
    obj(
        json!({"type": "metrics", "t": 1.0, "port": port, "container": container, "engine": engine, "status": status,
               "text": text}),
    )
}

fn push_port(port: i64) -> Obj {
    push(Some(port), None, TEXT1, 200, "fake")
}

/// `managed(env, name, host)`: a ready managed instance (its engine answers /health).
async fn managed(env: &mut FleetEnv, name: &str, host: &str) -> Instance {
    wse_kit::seed(env, host);
    env.fleet
        .fake(host)
        .serve(None, "/health", http_handler(|_| HttpResp::text(200, "ok")));
    let c = env.container();
    let inst = c
        .lifecycle
        .create(&request(json!({"name": name, "host_id": host})), None)
        .await
        .unwrap();
    c.lifecycle.tick().await.unwrap();
    c.lifecycle.get(&inst.id).unwrap()
}

/// The fake engine's parser under another engine id (Python: `fake = type(fake)(); fake.id = "vllm"`).
pub struct Renamed(pub &'static str);

impl EngineAdapter for Renamed {
    fn id(&self) -> &str {
        self.0
    }
    fn display_name(&self) -> &str {
        "Fake Engine"
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
        FakeAdapter.build_launch(m, p, hw, s, h)
    }
    fn compatibility(&self, m: &ModelInfo, p: &Obj, hw: &Hardware) -> Vec<Compat> {
        FakeAdapter.compatibility(m, p, hw)
    }
    fn memory_model(&self, m: &ModelInfo, p: &Obj) -> FloatMap {
        FakeAdapter.memory_model(m, p)
    }
    fn parse_metrics(&self, text: &str) -> FloatMap {
        FakeAdapter.parse_metrics(text)
    }
    fn parse_startup_log(&self, line: &str) -> Option<Obj> {
        FakeAdapter.parse_startup_log(line)
    }
}

fn values(p: &ec_hub::domain::models::MetricPoint) -> &Obj {
    &p.values
}

// ---- configuring the nodes ------------------------------------------------------------------------------------------------------
#[tokio::test]
async fn test_families_cover_every_engine_the_hub_knows_and_honour_an_adapters_own_prefixes() {
    let mut env = FleetEnv::new();
    env.adapters.push(Arc::new(FakeMultiAdapter));
    let fam = env.container().metrics.families();
    let get = |e: &str| {
        fam.iter()
            .find(|(k, _)| k == e)
            .map(|(_, p)| p.clone())
            .unwrap()
    };
    assert!(
        get("vllm") == ["vllm:"]
            && get("sglang") == ["sglang:"]
            && get("llamacpp") == ["llamacpp:"]
    );
    assert!(get("trtllm") == ["trtllm", "tensorrt_llm"] && get("dynamo") == ["dynamo_"]);
    assert_eq!(get("fakecpp"), ["fakecpp:"]); // declared by the adapter's `metric_prefixes`
    assert_eq!(get("fake"), ["fake:"]); // an adapter that declares none and is not in the table
    assert!(
        fam.iter()
            .all(|(_, ps)| ps.iter().all(|p| p.chars().count() <= 64))
            && fam.len() <= 16
    );
    assert!(
        PREFIXES
            .iter()
            .all(|(e, _)| fam.iter().any(|(k, _)| k == e))
    );
}

#[tokio::test]
async fn test_each_node_is_configured_once_per_connection_and_again_after_a_reconnect() {
    let mut env = FleetEnv::new();
    let spark = env.add_host("spark", "Spark", HelloOpts::default());
    let c = env.container();
    c.metrics.sync_hosts().await;
    c.metrics.sync_hosts().await;
    for node in [&env.node, &spark] {
        assert_eq!(node.ops("metrics.configure").len(), 1);
    }
    let cfg = env.node.ops("metrics.configure")[0].clone();
    assert_eq!(cfg["interval_s"], json!(env.cfg.scrape_interval_s));
    assert!(
        cfg["families"]["vllm"] == json!(["vllm:"]) && cfg["families"]["fake"] == json!(["fake:"])
    );
    spark.st().connected_since = 2000.0; // the link dropped and came back
    c.metrics.sync_hosts().await;
    assert!(
        spark.ops("metrics.configure").len() == 2 && env.node.ops("metrics.configure").len() == 1
    );
}

#[tokio::test]
async fn test_a_host_that_is_down_is_not_configured_until_it_comes_up() {
    let mut env = FleetEnv::new();
    let spark = env.add_host("spark", "Spark", HelloOpts::default());
    spark.set_state("down", None);
    let c = env.container();
    c.metrics.sync_hosts().await;
    assert!(spark.ops("metrics.configure").is_empty());
    spark.set_state("up", None);
    c.metrics.sync_hosts().await;
    assert_eq!(spark.ops("metrics.configure").len(), 1);
}

#[tokio::test]
async fn test_a_failed_configure_is_retried_on_the_next_round() {
    let mut env = FleetEnv::new();
    env.node.fail(
        "metrics.configure",
        NodeError::new("local", "timeout", "slow"),
    );
    let c = env.container();
    c.metrics.sync_hosts().await;
    env.node.unfail("metrics.configure");
    c.metrics.sync_hosts().await;
    assert!(env.node.ops("metrics.configure").len() == 2 && env.node.st().metrics_config.is_some());
}

// ---- routing a push --------------------------------------------------------------------------------------------------------------------
/// Security review 2026-10-04 L3: Prometheus text allows NaN/+Inf; SSE then sent `Infinity`, which is not JSON.
#[tokio::test]
async fn test_l3_non_finite_values_never_reach_the_series() {
    let mut env = FleetEnv::new();
    let c = env.container();
    let clock = use_clock(&c);
    let inst = managed(&mut env, "qwen", "local").await;
    c.metrics.ingest(
        "local",
        &push(
            inst.port,
            None,
            "running NaN\nwaiting +Inf\nkv 1e999\ngen_total 5\n",
            200,
            "fake",
        ),
    );
    tick(&clock, 1.0);
    let s = c.metrics.series(&inst.id, "15m").unwrap();
    let vals = s["points"].as_array().unwrap().last().unwrap()["values"].clone();
    for k in ["requests_running", "requests_waiting", "kv_cache_usage_pct"] {
        assert!(vals.get(k).is_none(), "{k}");
    }
    let text = serde_json::to_string(&s).unwrap(); // strict JSON, as the browser parses it
    assert!(!text.contains("NaN") && !text.contains("Infinity"));
}

#[tokio::test]
async fn test_a_push_is_routed_to_the_managed_instance_by_gateway_port_or_container() {
    let mut env = FleetEnv::new();
    let c = env.container();
    let clock = use_clock(&c);
    let inst = managed(&mut env, "qwen", "local").await;
    let cn = inst.container_name.clone().unwrap();
    c.metrics.ingest("local", &push_port(inst.port.unwrap()));
    tick(&clock, 1.0);
    c.metrics
        .ingest("local", &push(None, Some(&cn), TEXT2, 200, "fake"));
    tick(&clock, 1.0);
    c.metrics.ingest(
        "local",
        &push(None, Some(&format!("{cn}-gw")), TEXT2, 200, "fake"),
    ); // the gateway's name routes too
    let pts = c.metrics.series(&inst.id, "15m").unwrap()["points"]
        .as_array()
        .cloned()
        .unwrap();
    assert!(pts.len() == 3 && pts[0]["values"]["requests_running"] == json!(2.0));
    assert!(
        pts[1]["values"]["kv_cache_usage_pct"] == json!(20.0)
            && pts[1]["values"].get("unknown_key").is_none()
    );
}

#[tokio::test]
async fn test_pushes_are_routed_per_host_even_when_two_hosts_use_the_same_port_number() {
    let mut env = FleetEnv::new();
    let c = env.container();
    let clock = use_clock(&c);
    let a = managed(&mut env, "a", "local").await;
    env.add_host(
        "spark",
        "Spark",
        HelloOpts {
            port_range: (18000, 18099),
            ..HelloOpts::default()
        },
    );
    let b = managed(&mut env, "b", "spark").await;
    assert!(a.port == Some(18000) && b.port == Some(18000));
    c.metrics.ingest(
        "local",
        &push(Some(18000), None, "running 1\n", 200, "fake"),
    );
    tick(&clock, 1.0);
    c.metrics.ingest(
        "spark",
        &push(Some(18000), None, "running 7\n", 200, "fake"),
    );
    assert_eq!(
        values(&c.metrics.latest(&a.id).unwrap())["requests_running"],
        json!(1.0)
    );
    assert_eq!(
        values(&c.metrics.latest(&b.id).unwrap())["requests_running"],
        json!(7.0)
    );
}

#[tokio::test]
async fn test_pushes_for_externals_and_bare_endpoints_use_the_generic_map_or_their_adapter() {
    let mut env = FleetEnv::new();
    env.adapters.push(Arc::new(Renamed("vllm"))); // stands in for the real vllm adapter's parser
    env.node.add_container(
        "vllm-a",
        ContainerOpts {
            engine: Some("vllm".into()),
            host_ports: ports(18500, 8000, "0.0.0.0"),
            ..ContainerOpts::default()
        },
    );
    let ep = |c: Option<&str>, e: &str| EndpointOpts {
        container: c.map(str::to_string),
        engine: Some(e.into()),
        models: vec!["m".into()],
        ..EndpointOpts::default()
    };
    env.node.add_endpoint(18500, ep(Some("vllm-a"), "vllm"));
    env.node.add_container(
        "cpp-a",
        ContainerOpts {
            engine: Some("llamacpp".into()),
            image: "ghcr.io/ggml-org/llama.cpp:server".into(),
            host_ports: ports(18501, 8080, "0.0.0.0"),
            ..ContainerOpts::default()
        },
    );
    env.node.add_endpoint(18501, ep(Some("cpp-a"), "llamacpp"));
    env.node.add_endpoint(
        9100,
        EndpointOpts {
            models: vec!["bare".into()],
            ..ep(None, "vllm")
        },
    );
    let c = env.container();
    let clock = use_clock(&c);
    let m = &c.metrics;
    m.ingest("local", &push(None, Some("vllm-a"), TEXT1, 200, "vllm"));
    tick(&clock, 1.0);
    m.ingest(
        "local",
        &push(
            Some(18501),
            None,
            "llamacpp:requests_processing 4\nllamacpp:tokens_predicted_total 99\n",
            200,
            "llamacpp",
        ),
    );
    tick(&clock, 1.0);
    m.ingest("local", &push(Some(9100), None, "running 5\n", 200, "vllm"));
    assert_eq!(
        values(&m.latest("ext-local-vllm-a").unwrap())["requests_running"],
        json!(2.0)
    );
    assert_eq!(
        Value::Object(values(&m.latest("ext-local-cpp-a").unwrap()).clone()),
        json!({"requests_running": 4.0, "generation_tokens_total": 99.0})
    );
    assert_eq!(
        values(&m.latest("ext-local-127-0-0-1-9100").unwrap())["requests_running"],
        json!(5.0)
    );
}

#[tokio::test]
async fn test_unknown_ports_failed_scrapes_and_garbage_are_ignored() {
    let mut env = FleetEnv::new();
    let inst = managed(&mut env, "qwen", "local").await;
    let c = env.container();
    let m = &c.metrics;
    m.ingest("local", &push_port(31337)); // nobody's port
    m.ingest("local", &push(inst.port, None, TEXT1, 500, "fake")); // the scrape failed
    m.ingest(
        "local",
        &obj(json!({"type": "metrics", "port": "x", "container": 5, "text": 7})),
    ); // malformed
    m.ingest("nope", &push(inst.port, None, TEXT1, 200, "fake")); // a host that is not configured
    assert!(m.latest(&inst.id).is_none());
}

#[tokio::test]
async fn test_rates_are_derived_from_counter_deltas_and_touch_the_instance() {
    let mut env = FleetEnv::new();
    let c = env.container();
    let clock = use_clock(&c);
    let inst = managed(&mut env, "qwen", "local").await;
    c.metrics.ingest("local", &push_port(inst.port.unwrap()));
    tick(&clock, 2.0);
    c.metrics
        .ingest("local", &push(inst.port, None, TEXT2, 200, "fake"));
    let last = c.metrics.latest(&inst.id).unwrap().values;
    assert!((last["generation_tps"].as_f64().unwrap() - 50.0).abs() < 1e-9);
    assert!((last["prompt_tps"].as_f64().unwrap() - 100.0).abs() < 1e-9);
    let row = wse_kit::instance_row(&c, &inst.id).unwrap();
    assert!(!row["last_request_at"].is_null()); // requests_running > 0 keeps it from idling out
}

#[tokio::test]
async fn test_rollups_average_per_hour_and_window_semantics() {
    let mut env = FleetEnv::new();
    let c = env.container();
    let clock = use_clock(&c);
    let inst = managed(&mut env, "qwen", "local").await;
    c.metrics.ingest("local", &push_port(inst.port.unwrap()));
    tick(&clock, 1.0);
    c.metrics
        .ingest("local", &push(inst.port, None, TEXT2, 200, "fake"));
    let raw = c.metrics.series(&inst.id, "15m").unwrap();
    assert!(raw["resolution"] == "raw" && raw["points"].as_array().unwrap().len() == 2);
    let hourly = c.metrics.series(&inst.id, "1h").unwrap();
    assert!(
        hourly["resolution"] == "1h"
            && hourly["points"][0]["values"]["requests_running"] == json!(2.5)
    );
    assert_eq!(
        c.metrics.series(&inst.id, "7d").unwrap()["resolution"],
        "1d"
    );
}

// ---- host RAM per instance -----------------------------------------------------------------------------------------------------------
#[tokio::test]
async fn test_host_ram_keys_ride_along_with_engine_metrics() {
    let mut env = FleetEnv::new();
    let inst = managed(&mut env, "qwen", "local").await;
    env.node
        .st()
        .cg
        .insert(inst.container_name.clone().unwrap(), cgroup_default());
    let c = env.container();
    c.metrics.ingest("local", &push_port(inst.port.unwrap()));
    let v = c.metrics.latest(&inst.id).unwrap().values;
    let f = |k: &str| v[k].as_f64().unwrap();
    assert!((f("host_ram_gib") - 68.6).abs() < 1e-3 && (f("host_ram_anon_gib") - 3.3).abs() < 1e-3);
    assert!(
        (f("host_ram_shmem_gib") - 64.0).abs() < 1e-3
            && (f("host_ram_cache_gib") - 0.5).abs() < 1e-3
    );
    assert_eq!(f("requests_running"), 2.0);
}

#[tokio::test]
async fn test_an_engine_without_metrics_still_gets_a_host_ram_series() {
    let mut env = FleetEnv::new();
    env.node.add_container(
        "ollama-a",
        ContainerOpts {
            engine: Some("ollama".into()),
            image: "ollama/ollama:0.5".into(),
            host_ports: ports(18502, 11434, "0.0.0.0"),
            started_at: Some(1.0),
            ..ContainerOpts::default()
        },
    );
    env.node.add_endpoint(
        18502,
        EndpointOpts {
            container: Some("ollama-a".into()),
            engine: Some("ollama".into()),
            models: vec!["llama3".into()],
            health: None,
            ..EndpointOpts::default()
        },
    );
    env.node.st().cg.insert("ollama-a".into(), cgroup_default());
    let c = env.container();
    let clock = use_clock(&c);
    tick(&clock, 100.0);
    c.metrics.scrape_once().await;
    let pts = c.metrics.series("ext-local-ollama-a", "15m").unwrap()["points"]
        .as_array()
        .cloned()
        .unwrap();
    let mut keys: Vec<String> = pts[0]["values"]
        .as_object()
        .unwrap()
        .keys()
        .cloned()
        .collect();
    keys.sort();
    assert!(
        pts.len() == 1
            && keys
                == [
                    "host_ram_anon_gib",
                    "host_ram_cache_gib",
                    "host_ram_gib",
                    "host_ram_shmem_gib"
                ]
    );
}

#[tokio::test]
async fn test_instances_that_are_stopped_failed_or_on_a_down_host_get_no_points() {
    let mut env = FleetEnv::new();
    let c = env.container();
    let clock = use_clock(&c);
    let inst = managed(&mut env, "qwen", "local").await;
    env.node
        .st()
        .cg
        .insert(inst.container_name.clone().unwrap(), cgroup_default());
    env.node.set_state("down", None);
    tick(&clock, 100.0);
    c.metrics.scrape_once().await;
    assert!(c.metrics.latest(&inst.id).is_none());
}

// ---- the pull fallback ------------------------------------------------------------------------------------------------------------------
#[tokio::test]
async fn test_a_ready_managed_instance_nobody_pushed_for_is_read_through_the_node() {
    let mut env = FleetEnv::new();
    let c = env.container();
    let clock = use_clock(&c);
    let inst = managed(&mut env, "qwen", "local").await;
    let port = inst.port.unwrap();
    env.node.serve(
        Some(port),
        "/metrics",
        http_handler(|_| HttpResp::text(200, TEXT1)),
    );
    tick(&clock, 100.0);
    c.metrics.scrape_once().await;
    assert_eq!(
        c.metrics.latest(&inst.id).unwrap().values["requests_running"],
        json!(2.0)
    );
    assert!(
        env.node
            .http_requests()
            .iter()
            .any(|r| r.path == "/metrics" && r.port == port)
    );
    // pushes take over: while they keep arriving the hub does not poll
    c.metrics.ingest("local", &push_port(port));
    let n = env.node.http_requests().len();
    tick(&clock, env.cfg.scrape_interval_s);
    c.metrics.scrape_once().await;
    assert_eq!(env.node.http_requests().len(), n);
}

#[tokio::test]
async fn test_a_failing_metrics_endpoint_is_skipped_without_raising() {
    let mut env = FleetEnv::new();
    let c = env.container();
    let clock = use_clock(&c);
    let inst = managed(&mut env, "qwen", "local").await;
    env.node.serve(
        inst.port,
        "/metrics",
        http_handler(|_| HttpResp::text(503, "no")),
    );
    tick(&clock, 100.0);
    c.metrics.scrape_once().await;
    assert!(c.metrics.latest(&inst.id).is_none());
}

// ---- system series per host ---------------------------------------------------------------------------------------------------------------
#[tokio::test]
async fn test_system_series_are_kept_per_host_and_one_point_per_new_sample() {
    let mut env = FleetEnv::new();
    let spark = env.add_host("spark", "Spark", HelloOpts::default());
    spark.set_mem(&[
        ("total", 128.0),
        ("available", 100.0),
        ("shmem", 2.0),
        ("swap_total", 0.0),
        ("swap_free", 0.0),
    ]);
    let c = env.container();
    let clock = use_clock(&c);
    let m = &c.metrics;
    m.scrape_once().await;
    m.scrape_once().await; // the same samples again: nothing new, nothing recorded
    assert_eq!(
        m.system_series("15m", "local").unwrap()["points"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    tick(&clock, 2.0);
    env.node.tick_sample();
    m.scrape_once().await;
    let local = m.system_series("15m", "local").unwrap();
    assert!(local["points"].as_array().unwrap().len() == 2 && local["host_id"] == "local");
    let v = local["points"][0]["values"].clone();
    let mut keys: Vec<String> = v.as_object().unwrap().keys().cloned().collect();
    keys.sort();
    assert_eq!(
        keys,
        [
            "ram_available_gib",
            "ram_used_gib",
            "shmem_gib",
            "swap_used_gib"
        ]
    );
    assert!(v["ram_used_gib"] == json!(192.0) && v["ram_available_gib"] == json!(54.0));
    assert!(v["swap_used_gib"] == json!(24.0) && v["shmem_gib"] == json!(66.0));
    let sp = m.system_series("15m", "spark").unwrap()["points"][0]["values"].clone();
    assert!(sp["ram_used_gib"] == json!(28.0) && sp["swap_used_gib"] == json!(0.0));
    let hourly = m.system_series("1h", "spark").unwrap();
    assert!(
        hourly["resolution"] == "1h" && hourly["points"][0]["values"]["shmem_gib"] == json!(2.0)
    );
}

#[tokio::test]
async fn test_system_metrics_api_takes_a_host_and_validates_it() {
    let mut env = FleetEnv::new();
    env.container().metrics.scrape_once().await;
    let body = env
        .get(&format!("{P}/metrics/system?window=15m"))
        .send()
        .await
        .json();
    assert!(body["host_id"] == "local" && body["points"].as_array().unwrap().len() == 1);
    assert!(
        !env.get(&format!("{P}/metrics/system?host=local"))
            .send()
            .await
            .json()["points"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        env.get(&format!("{P}/metrics/system?host=nope"))
            .send()
            .await
            .status,
        404
    );
    assert_eq!(
        env.get(&format!("{P}/metrics/system?host=Bad%20Host"))
            .send()
            .await
            .status,
        422
    );
    assert_eq!(
        env.get(&format!("{P}/metrics/system?window=soon"))
            .send()
            .await
            .status,
        400
    );
}

#[tokio::test]
async fn test_the_sse_event_for_a_system_point_names_its_host() {
    let mut env = FleetEnv::new();
    env.add_host("spark", "Spark", HelloOpts::default());
    let c = env.container();
    let mut sub = c.bus.subscribe("metrics");
    c.metrics.scrape_once().await;
    let mut events = Vec::new();
    while events.len() < 2 {
        let ev = tokio::time::timeout(Duration::from_secs(5), sub.recv())
            .await
            .unwrap()
            .unwrap();
        events.push(ev);
    }
    let mut hosts: Vec<Value> = events.iter().map(|e| e["host_id"].clone()).collect();
    hosts.sort_by_key(ToString::to_string);
    assert_eq!(hosts, [json!("local"), json!("spark")]);
    assert!(events.iter().all(|e| e["instance_id"] == "system"));
}

// ---- the bus ----------------------------------------------------------------------------------------------------------------------------------------
#[tokio::test]
async fn test_consume_routes_what_the_fleet_publishes_on_the_bus() {
    let mut env = FleetEnv::new();
    let inst = managed(&mut env, "qwen", "local").await;
    let c = env.container();
    let m = Arc::clone(&c.metrics);
    let task = tokio::spawn(async move { m.consume().await });
    let bus = Arc::clone(&c.bus);
    until(
        move || bus.subscribers("node.metrics") > 0,
        5.0,
        "the consumer to subscribe",
    )
    .await;
    let port = inst.port.unwrap();
    c.bus.publish(
        "node.metrics",
        json!({"host_id": "local", "msg": Value::Object(push_port(port))}),
    );
    c.bus.publish(
        "node.metrics",
        json!({"host_id": "local", "msg": "not a dict"}),
    ); // must not kill the consumer
    c.bus.publish("node.metrics", json!({"host_id": "local", "msg": Value::Object(push(Some(port), None, TEXT2, 200, "fake"))}));
    let (c2, id) = (Arc::clone(&c), inst.id.clone());
    until(
        move || {
            c2.metrics.series(&id, "15m").unwrap()["points"]
                .as_array()
                .unwrap()
                .len()
                == 2
        },
        5.0,
        "both pushes to be recorded",
    )
    .await;
    task.abort();
}

#[tokio::test]
async fn test_run_configures_nodes_then_samples_until_cancelled() {
    let mut env = FleetEnv::new();
    let c = env.container();
    let m = Arc::clone(&c.metrics);
    let task = tokio::spawn(async move { m.run().await });
    let (node, c2) = (Arc::clone(&env.node), Arc::clone(&c));
    until(
        move || {
            !node.ops("metrics.configure").is_empty()
                && !c2.metrics.system_series("15m", "local").unwrap()["points"]
                    .as_array()
                    .unwrap()
                    .is_empty()
        },
        5.0,
        "the loop to configure the node and sample it",
    )
    .await;
    task.abort();
    let _ = task.await;
}

// ---- prometheus ------------------------------------------------------------------------------------------------------------------------------------
#[tokio::test]
async fn test_prometheus_gauges_for_hosts_gpus_and_instance_vram() {
    let mut env = FleetEnv::new();
    wse_kit::seed(&env, "local");
    env.engine_up();
    let spark = env.add_host("spark", "Spark", HelloOpts::default());
    spark.set_state("down", None);
    let inst = env
        .post(&format!("{P}/instances"))
        .json(&json!({"engine": "fake", "repo_id": wse_kit::QWEN, "gpu_ids": [1], "params": {"max_model_len": 4096}}))
        .send()
        .await
        .json();
    let (id, cn) = (
        inst["id"].as_str().unwrap().to_string(),
        inst["container_name"].as_str().unwrap().to_string(),
    );
    env.node
        .set_gpu_free(fakefleet::GpuFree::Each(vec![96.0, 36.0]));
    {
        let mut st = env.node.st();
        st.gpu_procs.insert(
            1,
            vec![json!({"pid": 1, "container": cn, "used_bytes": 60_i64 * (1 << 30)})],
        );
        st.cg.insert(cn.clone(), cgroup_default());
    }
    let text = env.get("/metrics").send().await.text();
    assert!(
        text.contains("engine_console_host_up{host=\"local\"} 1.0")
            && text.contains("engine_console_host_up{host=\"spark\"} 0.0")
    );
    assert!(text.contains("engine_console_host_gpu_used_gib{gpu=\"1\",host=\"local\"} 60.0"));
    assert!(text.contains("engine_console_host_gpu_used_gib{gpu=\"0\",host=\"local\"} 0.0"));
    let gpu_block = text
        .split("engine_console_host_gpu_used_gib")
        .nth(1)
        .unwrap()
        .split("# HELP")
        .next()
        .unwrap();
    assert!(!gpu_block.contains("host=\"spark\"")); // a down host reports no GPUs
    assert!(text.contains(&format!(
        "engine_console_instance_vram_gib{{gpu=\"1\",host=\"local\",instance=\"{id}\"}} 60.0"
    )));
    assert!(
        text.contains("engine_console_host_ram_used_gib 192.0")
            && text.contains("engine_console_host_swap_used_gib 24.0")
    );
    assert!(text.contains(&format!(
        "engine_console_instance_host_ram_gib{{instance=\"{id}\"}}"
    )));
    assert!(text.contains("engine_console_instances{state=\"starting\"} 1.0"));
    // label sets of things that are gone do not linger
    env.delete(&format!("{P}/instances/{id}")).send().await;
    assert!(
        !env.get("/metrics")
            .send()
            .await
            .text()
            .contains(&format!("instance=\"{id}\""))
    );
}

#[tokio::test]
async fn test_a_service_can_be_built_without_adapters() {
    let mut env = FleetEnv::new();
    let c = env.container();
    let m = MetricsService::new(
        Arc::clone(&c.store),
        Arc::clone(&c.lifecycle),
        AdapterRegistry::default(),
        c.fleet_port(),
        Arc::clone(&c.nodehttp),
        Arc::clone(&c.bus),
        2.0,
    );
    let mut fam: Vec<String> = m.families().into_iter().map(|(e, _)| e).collect();
    fam.sort();
    let mut want: Vec<String> = PREFIXES.iter().map(|(e, _)| (*e).to_string()).collect();
    want.sort();
    assert_eq!(fam, want);
}
