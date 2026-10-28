//! Port of backend/tests/core/test_metrics_prune.py. Security control (review M7): series keyed by names a node
//! chooses are capped per host, expire, and their rollups age out.

#![allow(
    clippy::redundant_clone,
    clippy::cloned_ref_to_slice_refs,
    clippy::uninlined_format_args,
    clippy::manual_let_else,
    clippy::expect_used,
    clippy::panic,
    clippy::unwrap_used,
    clippy::needless_pass_by_value,
    clippy::too_many_arguments,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    clippy::too_many_lines,
    clippy::type_complexity
)]

#[path = "common/kit.rs"]
mod kit;

use kit::fakefleet;

#[path = "common/wse_kit.rs"]
mod wse_kit;

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

use ec_hub::adapters::base::{
    Compat, EngineAdapter, FloatMap, Hardware, LaunchSpec, ModelInfo, Obj, ParamSpec,
};
use ec_hub::container::Container;
use ec_hub::domain::errors::Problem;
use ec_hub::services::metrics::{DAILY_KEEP_S, HOURLY_KEEP_S, KEY_TTL_S, PRUNE_EVERY_S};
use fakefleet::{ContainerOpts, EndpointOpts, FakeNode, FleetEnv, HelloOpts, ports};
use kit::FakeAdapter;
use serde_json::json;
use wse_kit::obj;

/// The fake engine's parser under the id "vllm" (Python's `vllm_parser` fixture).
struct Vllm;

impl EngineAdapter for Vllm {
    fn id(&self) -> &str {
        "vllm"
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

fn env_with_vllm() -> FleetEnv {
    let mut env = FleetEnv::new();
    env.adapters.push(Arc::new(Vllm));
    env
}

type Clock = Arc<Mutex<f64>>;

fn use_clock(c: &Container) -> Clock {
    let clock = Arc::new(Mutex::new(10_000.0_f64));
    let k = Arc::clone(&clock);
    c.metrics.set_clock(Arc::new(move || *k.lock().unwrap()));
    clock
}

fn now(clock: &Clock) -> f64 {
    *clock.lock().unwrap()
}

fn push(container: &str, text: &str) -> Obj {
    obj(
        json!({"type": "metrics", "t": 1.0, "port": null, "container": container, "engine": "vllm", "status": 200,
               "text": text}),
    )
}

/// n external engines on a node (their series are keyed by the container names the node chose).
fn engines(node: &FakeNode, n: i64, base: i64) -> Vec<String> {
    (0..n)
        .map(|i| {
            let name = format!("vllm-{}", base + i);
            node.add_container(
                &name,
                ContainerOpts {
                    engine: Some("vllm".into()),
                    host_ports: ports(base + i, 8000, "0.0.0.0"),
                    ..ContainerOpts::default()
                },
            );
            node.add_endpoint(
                base + i,
                EndpointOpts {
                    container: Some(name.clone()),
                    engine: Some("vllm".into()),
                    models: vec!["m".into()],
                    ..EndpointOpts::default()
                },
            );
            name
        })
        .collect()
}

fn rollup_ids(c: &Container) -> BTreeSet<String> {
    c.store
        .all("SELECT DISTINCT instance_id FROM metric_rollups", &[])
        .unwrap()
        .iter()
        .map(|r| r.str_or("instance_id", ""))
        .collect()
}

#[tokio::test]
async fn test_one_host_cannot_create_more_series_than_the_cap_and_the_stalest_goes_first() {
    let mut env = env_with_vllm();
    let c = env.container();
    c.metrics.set_max_keys_per_host(4);
    let clock = use_clock(&c);
    let names = engines(&env.node, 7, 18500);
    for n in &names {
        c.metrics.ingest("local", &push(n, "running 1\n"));
        *clock.lock().unwrap() += 1.0;
    }
    assert_eq!(c.metrics.keys_of("local"), Some(4));
    assert!(c.metrics.ring_len() <= 4);
    let want: BTreeSet<String> = names[3..]
        .iter()
        .map(|n| format!("ext-local-{n}"))
        .collect();
    assert_eq!(rollup_ids(&c), want); // the three oldest lost memory and rows
    assert!(
        c.metrics
            .latest(&format!("ext-local-{}", names[0]))
            .is_none()
    );
    assert!(
        c.metrics
            .latest(&format!("ext-local-{}", names[6]))
            .is_some()
    );
}

#[tokio::test]
async fn test_a_flooding_host_does_not_evict_another_hosts_series() {
    let mut env = env_with_vllm();
    let spark = env.add_host("spark", "Spark", HelloOpts::default());
    let c = env.container();
    c.metrics.set_max_keys_per_host(3);
    let clock = use_clock(&c);
    engines(&spark, 1, 19000);
    c.metrics
        .ingest("spark", &push("vllm-19000", "running 1\n"));
    for n in engines(&env.node, 8, 18500) {
        *clock.lock().unwrap() += 1.0;
        c.metrics.ingest("local", &push(&n, "running 1\n"));
    }
    assert!(
        c.metrics.latest("ext-spark-vllm-19000").is_some() && c.metrics.keys_of("spark") == Some(1)
    );
}

#[tokio::test]
async fn test_a_series_nobody_updates_expires_from_memory_and_its_rows_after_the_retention() {
    let mut env = env_with_vllm();
    let c = env.container();
    let clock = use_clock(&c);
    let name = engines(&env.node, 1, 18500).remove(0);
    c.metrics.ingest("local", &push(&name, "running 1\n"));
    let iid = format!("ext-local-{name}");
    let t = now(&clock);
    c.metrics.prune(Some(t + KEY_TTL_S - 1.0), true);
    assert!(c.metrics.latest(&iid).is_some()); // a short silence keeps the history
    c.metrics.prune(Some(t + KEY_TTL_S + 1.0), true);
    assert!(
        c.metrics.latest(&iid).is_none()
            && c.metrics.keys_of("local").is_none()
            && !c.metrics.has_counters(&iid)
    );
    assert!(rollup_ids(&c).contains(&iid)); // rows outlive the memory
    c.metrics.prune(Some(t + HOURLY_KEEP_S + 7200.0), true);
    let left: BTreeSet<String> = c
        .store
        .all(
            "SELECT res FROM metric_rollups WHERE instance_id=?",
            &[&iid],
        )
        .unwrap()
        .iter()
        .map(|r| r.str_or("res", ""))
        .collect();
    assert_eq!(left, BTreeSet::from(["d".to_string()])); // the hourly rows are gone, the daily stay
    c.metrics
        .prune(Some(t + DAILY_KEEP_S + 2.0 * 86400.0), true);
    assert!(!rollup_ids(&c).contains(&iid));
}

#[tokio::test]
async fn test_the_bookkeeping_of_a_host_that_left_hosts_json_is_forgotten() {
    let mut env = env_with_vllm();
    let c = env.container();
    c.metrics
        .seed_bookkeeping("gone", json!(5), (true, Some(1.0)));
    c.metrics.seed_warned("gone", Some(1.0));
    c.metrics.seed_warned("local", Some(-1.0));
    c.metrics.prune(None, true);
    let (seq, configured, warned) = c.metrics.bookkeeping();
    assert!(
        !seq.contains(&"gone".to_string())
            && !configured.contains(&"gone".to_string())
            && warned == 0
    );
}

#[tokio::test]
async fn test_the_prune_is_throttled_and_the_cap_survives_a_restart() {
    let mut env = env_with_vllm();
    let c = env.container();
    let clock = use_clock(&c);
    for n in engines(&env.node, 5, 18500) {
        c.metrics.ingest("local", &push(&n, "running 1\n"));
    }
    c.metrics.prune(None, false);
    c.metrics.forget_memory(); // a restart forgets memory
    c.metrics.set_max_keys_per_host(2);
    let t = now(&clock);
    c.metrics.prune(Some(t + 1.0), false); // inside PRUNE_EVERY_S: does nothing
    assert!(c.metrics.keys_of("local").is_none());
    c.metrics.prune(Some(t + PRUNE_EVERY_S + 1.0), false);
    assert!(c.metrics.keys_of("local") == Some(2) && rollup_ids(&c).len() == 2);
}
