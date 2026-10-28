//! Helpers for the lifecycle, externals, scripts and metrics tests, on top of the fake fleet
//! (common/fakefleet.rs, Python's fakefleet.py + conftest `Env`).
//! Include it after the kit and the fake fleet:
//! ```ignore
//! #[path = "common/kit.rs"] mod kit;
//! #[path = "common/fakefleet.rs"] mod fakefleet;
//! #[path = "common/wse_kit.rs"] mod wse_kit;
//! ```

#![allow(
    dead_code,
    unreachable_pub,
    clippy::expect_used,
    clippy::panic,
    clippy::needless_pass_by_value
)]

use std::collections::BTreeMap;
use std::sync::Arc;

use ec_hub::container::Container;
use ec_hub::domain::errors::{AppError, Obj, Problem};
use ec_hub::domain::models::{Instance, InstanceCreate};
use ec_hub::domain::refs::ModelRef;
use ec_hub::services::libport::LibraryPort;
use serde_json::{Value, json};

use crate::fakefleet::{ContainerOpts, FakeNode, FleetEnv};

pub const P: &str = "/api/v1";
pub const QWEN: &str = "Qwen/Qwen3-32B";
pub const SECRET: &str = "sk-live-abcdef123456";

pub fn obj(v: Value) -> Obj {
    v.as_object().cloned().unwrap_or_default()
}

/// `request(**kw)`: the fake engine, Qwen/Qwen3-32B, max_model_len 4096, with `extra` on top.
pub fn request(extra: Value) -> InstanceCreate {
    let mut b = json!({"engine": "fake", "repo_id": QWEN, "params": {"max_model_len": 4096}});
    for (k, v) in obj(extra) {
        b[k] = v;
    }
    serde_json::from_value(b).expect("InstanceCreate")
}

/// `env.seed_local_model(host_id=host)`
pub fn seed(env: &FleetEnv, host: &str) {
    env.seed_local_model(QWEN, None, 65_530_000_000, host);
}

/// `launch(env, **kw)`: seed the model on that host when it is not there yet, then create.
pub async fn launch(env: &mut FleetEnv, extra: Value) -> Instance {
    let req = request(extra);
    if env
        .library
        .find(&req.host_id, &ModelRef::hf(QWEN))
        .is_none()
    {
        seed(env, &req.host_id);
    }
    env.container()
        .lifecycle
        .create(&req, None)
        .await
        .unwrap_or_else(|e| panic!("create failed: {e:?}"))
}

pub fn problem(e: &AppError) -> Problem {
    e.problem()
        .cloned()
        .unwrap_or_else(|| panic!("not a problem: {e:?}"))
}

/// `codes(e)`: check code -> level of a `preflight_failed` problem.
pub fn codes(e: &AppError) -> BTreeMap<String, String> {
    let p = problem(e);
    p.extra["preflight"]["checks"]
        .as_array()
        .cloned()
        .unwrap_or_default()
        .iter()
        .map(|c| {
            (
                c["code"].as_str().unwrap_or_default().to_string(),
                c["level"].as_str().unwrap_or_default().to_string(),
            )
        })
        .collect()
}

/// `preflight_codes(env, **kw)`
pub async fn preflight_codes(c: &Arc<Container>, extra: Value) -> BTreeMap<String, String> {
    let (report, _) = c
        .lifecycle
        .preflight_full(&request(extra), None, None)
        .await
        .expect("preflight");
    report
        .checks
        .iter()
        .map(|c| (c.code.clone(), c.level.clone()))
        .collect()
}

pub fn run_specs(node: &FakeNode) -> Vec<Value> {
    node.st().run_specs.clone()
}

pub fn container(node: &FakeNode, name: &str) -> Option<Obj> {
    node.st()
        .containers
        .get(name)
        .and_then(Value::as_object)
        .cloned()
}

/// `node.containers[name].update(...)`
pub fn update_container(node: &FakeNode, name: &str, kv: Value) {
    let mut st = node.st();
    if let Some(Value::Object(c)) = st.containers.get_mut(name) {
        for (k, v) in obj(kv) {
            c.insert(k, v);
        }
    }
}

pub fn remove_container(node: &FakeNode, name: &str) {
    node.st().containers.remove(name);
}

pub fn set_logs(node: &FakeNode, name: &str, text: &str) {
    node.st().logs.insert(name.into(), text.into());
}

/// `node.add_container(name, managed=True, labels=..., image=...)`
pub fn add_managed(node: &FakeNode, name: &str, labels: Value, image: &str) {
    node.add_container(
        name,
        ContainerOpts {
            managed: true,
            labels: obj(labels),
            image: image.into(),
            ..ContainerOpts::default()
        },
    );
}

/// `node.add_container(name, engine=..., ...)`
pub fn add_engine(node: &FakeNode, name: &str, engine: &str, o: ContainerOpts) {
    node.add_container(
        name,
        ContainerOpts {
            engine: Some(engine.into()),
            ..o
        },
    );
}

/// A SQL row of `instances` as JSON (column -> value).
pub fn instance_row(c: &Container, iid: &str) -> Option<Obj> {
    c.store
        .one("SELECT * FROM instances WHERE id=?", &[&iid])
        .expect("db")
        .map(|r| r.to_map())
}

pub fn exec(c: &Container, sql: &str, iid: &str) {
    c.store.execute(sql, &[&iid]).expect("sql");
}

/// A fleet that can forget a host (Python tests `del env.fleet._nodes[host]`), delegating to the fake fleet.
pub struct HidingFleet {
    inner: Arc<crate::fakefleet::FakeFleet>,
    hidden: std::sync::Mutex<Vec<String>>,
}

impl HidingFleet {
    pub fn new(inner: Arc<crate::fakefleet::FakeFleet>) -> Self {
        Self {
            inner,
            hidden: std::sync::Mutex::new(Vec::new()),
        }
    }

    pub fn hide(&self, host_id: &str) {
        self.hidden.lock().expect("lock").push(host_id.into());
    }

    fn is_hidden(&self, host_id: &str) -> bool {
        self.hidden
            .lock()
            .expect("lock")
            .iter()
            .any(|h| h == host_id)
    }
}

impl ec_hub::domain::fleet::FleetPort for HidingFleet {
    fn hosts(&self) -> Vec<ec_hub::domain::fleet::HostConfig> {
        self.inner
            .hosts()
            .into_iter()
            .filter(|h| !self.is_hidden(&h.id))
            .collect()
    }
    fn node(&self, host_id: &str) -> Result<Arc<dyn ec_hub::domain::fleet::NodeClient>, Problem> {
        if self.is_hidden(host_id) {
            return Err(ec_hub::domain::errors::not_found(format!(
                "no host '{host_id}' is configured"
            ))
            .code("unknown_host"));
        }
        self.inner.node(host_id)
    }
}

#[async_trait::async_trait]
impl ec_hub::domain::fleet::Fleet for HidingFleet {
    async fn start(&self) {}
    async fn stop(&self) {}
    fn summaries(
        &self,
    ) -> ec_hub::domain::errors::AppResult<Vec<ec_hub::domain::fleet::HostSummary>> {
        self.inner.summaries()
    }
    fn summary(
        &self,
        host_id: &str,
    ) -> ec_hub::domain::errors::AppResult<ec_hub::domain::fleet::HostSummary> {
        self.inner.summary(host_id)
    }
    async fn detail(
        &self,
        host_id: &str,
    ) -> ec_hub::domain::errors::AppResult<ec_hub::domain::fleet::HostDetail> {
        self.inner.detail(host_id).await
    }
    fn reconnect(
        &self,
        host_id: &str,
    ) -> ec_hub::domain::errors::AppResult<ec_hub::domain::fleet::HostSummary> {
        self.inner.reconnect(host_id)
    }
    fn set_instance_resolver(&self, f: ec_hub::domain::fleet::InstanceResolver) {
        self.inner.set_instance_resolver(f);
    }
    fn set_counts_provider(&self, f: ec_hub::domain::fleet::CountsProvider) {
        self.inner.set_counts_provider(f);
    }
}

/// A container like FleetEnv's, over another fleet (the env's settings, adapters, hub and fake library).
pub fn container_with_fleet(
    env: &FleetEnv,
    fleet: Arc<dyn ec_hub::domain::fleet::Fleet>,
) -> Arc<Container> {
    let o = ec_hub::container::Overrides {
        adapters: Some(ec_hub::adapters::AdapterRegistry::new(env.adapters.clone())),
        fleet: Some(fleet),
        hub: Some(Arc::clone(&env.hub) as Arc<dyn ec_hub::domain::ports::HubClient>),
        library: Some(Arc::clone(&env.library) as Arc<dyn LibraryPort>),
        ..ec_hub::container::Overrides::default()
    };
    ec_hub::container::build_container(Arc::new(env.cfg.clone()), o).expect("container")
}
