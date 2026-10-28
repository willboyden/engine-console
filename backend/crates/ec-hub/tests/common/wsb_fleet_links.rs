//! test_fleet_service.py `FakeLink` and `make`: a link that never starts a process (state and data set by the test),
//! and a FleetService over such links. Needs `mod wsb_harness;` (and its `wsb_fake_node`) at the root.

#![allow(
    dead_code,
    unreachable_pub,
    clippy::expect_used,
    clippy::type_complexity
)]

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use async_trait::async_trait;
use ec_hub::config::Settings;
use ec_hub::domain::errors::{NodeError, Obj};
use ec_hub::domain::fleet::{HostConfig, LinkStatus, NodeClient, NodeStream};
use ec_hub::services::common::EventBus;
use ec_hub::services::fleet::{FailedStream, FleetService, LinkFactory, LinkLike};
use serde_json::json;

use crate::wsb_harness::{disks, hello, mount, obj, sample};

pub struct LinkState {
    pub state: String,
    pub error: Option<String>,
    pub data: HashMap<String, Obj>,
    /// (op, args, timeout in seconds)
    pub calls: Vec<(String, Option<Obj>, f64)>,
    pub results: HashMap<String, Result<Obj, NodeError>>,
    pub delay: f64,
    pub started: bool,
    pub stopped: bool,
    pub reconnects: u32,
}

/// A link that never starts a process: state and data are set by the test.
pub struct FakeLink {
    pub host: HostConfig,
    st: Mutex<LinkState>,
}

impl FakeLink {
    pub fn new(host: HostConfig, state: &str) -> Self {
        Self {
            host,
            st: Mutex::new(LinkState {
                state: state.into(),
                error: None,
                data: HashMap::new(),
                calls: Vec::new(),
                results: HashMap::new(),
                delay: 0.0,
                started: false,
                stopped: false,
                reconnects: 0,
            }),
        }
    }

    pub fn st(&self) -> MutexGuard<'_, LinkState> {
        self.st.lock().expect("lock")
    }

    pub fn set(&self, kind: &str, msg: Obj) {
        self.st().data.insert(kind.into(), msg);
    }

    pub fn result(&self, op: &str, r: Result<Obj, NodeError>) {
        self.st().results.insert(op.into(), r);
    }
}

#[async_trait]
impl NodeClient for FakeLink {
    fn host(&self) -> &HostConfig {
        &self.host
    }
    fn link(&self) -> LinkStatus {
        let st = self.st();
        LinkStatus::new(&st.state, st.error.clone())
    }
    fn latest(&self, kind: &str) -> Option<Arc<Obj>> {
        self.st().data.get(kind).cloned().map(Arc::new)
    }
    async fn call(&self, op: &str, args: Option<Obj>, timeout: Duration) -> Result<Obj, NodeError> {
        let delay = {
            let mut st = self.st();
            st.calls.push((op.into(), args, timeout.as_secs_f64()));
            st.delay
        };
        if delay > 0.0 {
            tokio::time::sleep(Duration::from_secs_f64(delay)).await;
        }
        self.st()
            .results
            .get(op)
            .cloned()
            .unwrap_or_else(|| Ok(Obj::new()))
    }
    fn open(&self, _op: &str, _args: Option<Obj>) -> Box<dyn NodeStream> {
        Box::new(FailedStream::new(NodeError::unavailable(
            &self.host.id,
            "fake",
        )))
    }
}

#[async_trait]
impl LinkLike for FakeLink {
    async fn start(&self) {
        self.st().started = true;
    }
    async fn stop(&self) {
        self.st().stopped = true;
    }
    fn reconnect(&self) {
        self.st().reconnects += 1;
    }
}

/// test_fleet_service.py `IMAGES`
pub fn images() -> Obj {
    obj(
        json!({"images": [{"ref": "vllm/vllm-openai:v0.1", "id": "sha256:abc", "size": 5, "created": 1.0, "arch": "amd64"}]}),
    )
}

pub type Links = Arc<Mutex<HashMap<String, Arc<FakeLink>>>>;

/// `make(cfg, hosts, **states)`: every link starts `up` (unless `states` says otherwise) with a hello, a sample, one
/// disk and the IMAGES answer.
pub fn make(
    cfg: Arc<Settings>,
    hosts: Vec<HostConfig>,
    states: &[(&str, &str)],
) -> (FleetService, HashMap<String, Arc<FakeLink>>, Arc<EventBus>) {
    make_with(cfg, hosts, states, |link| {
        link.set("hello", hello());
        link.set("sample", sample(json!({})));
        link.set("disks", disks(vec![mount("/", json!({}))]));
        link.result("docker.images", Ok(images()));
    })
}

/// `make` with the test's own data on every link.
pub fn make_with(
    cfg: Arc<Settings>,
    hosts: Vec<HostConfig>,
    states: &[(&str, &str)],
    init: impl Fn(&FakeLink) + Send + Sync + 'static,
) -> (FleetService, HashMap<String, Arc<FakeLink>>, Arc<EventBus>) {
    let bus = Arc::new(EventBus::new());
    let links: Links = Arc::default();
    let l2 = Arc::clone(&links);
    let states: HashMap<String, String> = states
        .iter()
        .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
        .collect();
    let factory: LinkFactory = Arc::new(move |h: &HostConfig| {
        let link = Arc::new(FakeLink::new(
            h.clone(),
            states.get(&h.id).map_or("up", String::as_str),
        ));
        init(&link);
        l2.lock()
            .expect("lock")
            .insert(h.id.clone(), Arc::clone(&link));
        link as Arc<dyn LinkLike>
    });
    let fleet =
        FleetService::with_links(cfg, Arc::clone(&bus), Some(hosts), Some(factory)).expect("fleet");
    let map = links.lock().expect("lock").clone();
    (fleet, map, bus)
}
