//! An in-process fleet for tests (port of backend/tests/core/fakefleet.py).
//!
//! `FakeNode` implements `NodeClient`, `FakeFleet` implements `Fleet`/`FleetPort`, `FakeLibrary` the `LibraryPort`
//! slice of the model library the runtime services use, and `FleetEnv` is Python's `Env` (tests/core/conftest.py)
//! built on them: the app container, the fake fleet behind it (one machine, `local`, by default), the fake library and
//! hub, and programmable fake engines.
//!
//! The fake node behaves like a real `ecnode` closely enough that the hub's runtime services are exercised end to end
//! without docker: `docker.run` creates a container that then shows up in the pushed `inventory`, `gateway.start`
//! publishes a port (or answers `port_in_use`), `docker.stop/rm` enforce ownership, `http.request` obeys the node's
//! port and path rules, `scripts.run` streams lines. Fidelity is enforced where it matters:
//!
//! * every request's arguments go through `ec_proto::clean_args` and must come out unchanged (a misspelt or
//!   over-long argument fails the request with `bad_args` instead of being silently dropped, as the real node would);
//! * monitor mode refuses every op outside `MONITOR_OPS`;
//! * every pushed message and every result is rebuilt with `sanitize_push` / `clean`, like the hub's client.
//!
//! Put test data on the node (`add_container`, `add_endpoint`, `set_gpu_free`, `st()` for everything else) and read
//! what the hub asked for from `node.calls()` / `node.ops(name)`.
//!
//! It is the core kit's submodule (it reuses the kit's fake hub, fake engine and `Resp`; PORTING.md §A4). Include
//! only the kit and reach it through it; never include this file as a module of its own as well (clippy
//! `duplicate_mod`, and two copies of every type):
//! ```ignore
//! #[path = "common/kit.rs"] mod kit;
//! use kit::fakefleet;
//! ```
//!
//! Differences from the Python fake, forced by Rust:
//! * Python tests assign attributes (`env.node.state = "down"`); here they lock the node's state: `node.st().state =
//!   "down".into()`, or use the setters.
//! * Python monkeypatches `_op_*` methods; here `node.on(op, ..)` / `node.on_async(op, ..)` / `node.on_open(op, ..)`
//!   replace (or add) an op's behaviour, and `st().hidden` makes `latest(kind)` answer None for a kind.
//! * `FakeLibrary::add_hf` builds the copy's ModelInfo with `services::hf::build_model_info` (as Python's fake does);
//!   `FakeLibrary::set_info_builder` replaces it.
//! * A `FakeStream` dropped before it finished counts as cancelled (Python's consumers always `aclose()`, which
//!   cancels; a dropped Rust future cannot run async cleanup).

#![allow(
    dead_code,
    unreachable_pub,
    clippy::panic,
    clippy::expect_used,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::needless_pass_by_value,
    clippy::type_complexity,
    clippy::too_many_arguments,
    clippy::missing_panics_doc
)]

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, Weak};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use axum::body::Body as AxBody;
use bytes::Bytes;
use ec_hub::adapters::AdapterRegistry;
use ec_hub::adapters::base::{EngineAdapter, ModelInfo};
use ec_hub::app::{AppState, app_state, handle_request};
use ec_hub::config::Settings;
use ec_hub::container::{Container, Overrides, build_container};
use ec_hub::domain::errors::{AppResult, NodeError, Obj, Problem, not_found, not_ported};
use ec_hub::domain::fleet::{
    CountsProvider, Fleet, FleetPort, HostConfig, HostDetail, HostSummary, InstanceResolver,
    LibraryCopy, LinkStatus, MountInfo, NodeClient, NodeStream, RootInfo, StreamItem,
};
use ec_hub::domain::ports::{HubClient, HubModel, RepoFile};
use ec_hub::domain::refs::ModelRef;
use ec_hub::services::libport::LibraryPort;
use ec_proto::specs;
use futures::StreamExt;
use futures::future::BoxFuture;
use futures::stream::BoxStream;
use http_body_util::BodyExt;
use serde_json::{Value, json};
use sha1::Digest as _;

use crate::kit::{FakeAdapter, FakeHub, Resp};

pub const GIB: f64 = 1_073_741_824.0;
/// The paths a node relays to an engine (the fake's version of the node's http.request path rule).
pub const HTTP_PREFIXES: &[&str] = &[
    "/v1/",
    "/health",
    "/metrics",
    "/version",
    "/api/",
    "/get_model_info",
    "/get_server_info",
    "/server_info",
];

/// fakefleet.py `DEFAULT_ROOTS`
pub fn default_roots() -> Vec<Value> {
    vec![
        json!({"id": "hf", "path": "/data/hf", "kind": "hf-home", "writable": true, "label": "HF cache"}),
        json!({"id": "gguf", "path": "/data/gguf", "kind": "gguf", "writable": true, "label": "GGUF files"}),
        json!({"id": "ollama", "path": "/data/ollama", "kind": "ollama", "writable": false, "label": "Ollama"}),
    ]
}

fn obj(v: Value) -> Obj {
    match v {
        Value::Object(o) => o,
        _ => Obj::new(),
    }
}

// ── hello ────────────────────────────────────────────────────────────────────────────────────────────────
/// fakefleet.py `make_hello` keyword arguments, with Python's defaults. `hostname` None: the host's label.
#[derive(Debug, Clone)]
pub struct HelloOpts {
    pub hostname: Option<String>,
    pub arch: String,
    pub gpus: usize,
    pub gpu_gib: f64,
    pub cc: String,
    pub unified: bool,
    pub mode: String,
    pub engine_images: Vec<String>,
    pub gateway_image: Option<String>,
    pub engine_network: String,
    pub port_range: (i64, i64),
    /// None: `default_roots()`
    pub roots: Option<Vec<Value>>,
    pub scripts: Vec<Value>,
    pub external_control: String,
    pub allow_ipc_host: bool,
    pub max_shm_gib: Option<f64>,
    pub addresses: Vec<String>,
    pub policy_path: String,
    pub ram_gib: f64,
    pub uuid_prefix: String,
    pub source: String,
}

impl Default for HelloOpts {
    fn default() -> Self {
        Self {
            hostname: None,
            arch: "x86_64".into(),
            gpus: 2,
            gpu_gib: 96.0,
            cc: "12.0".into(),
            unified: false,
            mode: "manage".into(),
            engine_images: vec!["fake/engine:*".into()],
            gateway_image: Some("fake/gateway:1".into()),
            engine_network: "engine-console-engines".into(),
            port_range: (18000, 18099),
            roots: None,
            scripts: Vec::new(),
            external_control: "none".into(),
            allow_ipc_host: true,
            max_shm_gib: Some(64.0),
            addresses: vec!["10.0.0.5".into()],
            policy_path: "/etc/ecnode/policy.json".into(),
            ram_gib: 246.0,
            uuid_prefix: "GPU-uuid-".into(),
            source: "nvml".into(),
        }
    }
}

/// A sanitized hello, as the hub's link would hold it.
pub fn make_hello(o: &HelloOpts) -> Obj {
    let gib = |x: f64| (x * GIB) as i64;
    let gpus: Vec<Value> = (0..o.gpus)
        .map(|i| {
            json!({"index": i, "uuid": format!("{}{i}", o.uuid_prefix), "name": "RTX PRO 6000 Blackwell",
                   "total_bytes": gib(o.gpu_gib), "compute_capability": o.cc, "unified_memory": o.unified})
        })
        .collect();
    let addresses: Vec<Value> = o
        .addresses
        .iter()
        .map(|a| json!({"iface": "eth0", "addr": a, "prefix": 24}))
        .collect();
    let msg = json!({
        "type": "hello", "proto": ec_proto::PROTO, "version": "0.1.0", "node_id": "n".repeat(16),
        "host": {"hostname": o.hostname.clone().unwrap_or_else(|| "testbox".into()), "arch": o.arch, "os": "Linux",
                 "kernel": "6.x", "cpu_model": "cpu", "cpu_cores": 16, "cpu_threads": 32,
                 "ram_total_bytes": gib(o.ram_gib), "boot_time": 1.0, "python": "3.12.0"},
        "gpu": {"driver": "595.91", "cuda": "13.0", "source": o.source},
        "gpus": gpus,
        "docker": {"available": true, "context": "rootless", "rootless": true, "version": "29.6", "nvidia_runtime": true},
        "addresses": addresses,
        "policy": {"path": o.policy_path, "mode": o.mode, "engine_images": o.engine_images,
                   "gateway_image": o.gateway_image, "engine_network": o.engine_network,
                   "port_range": [o.port_range.0, o.port_range.1],
                   "model_roots": o.roots.clone().unwrap_or_else(default_roots), "scripts": o.scripts,
                   "allow_pull": false, "allow_delete": false, "allow_ipc_host": o.allow_ipc_host,
                   "external_control": o.external_control, "max_shm_gib": o.max_shm_gib},
        "forced_cmd": null, "via_ssh": false});
    ec_proto::sanitize_push(&obj(msg))
        .expect("hello")
        .expect("known type")
}

/// fakefleet.py `BIG_MODEL_SCRIPT`
pub fn big_model_script() -> Value {
    json!({
        "name": "big-model", "label": "Big model (2 GPUs)", "engine": "sglang", "description": "Starts before the small one.",
        "containers": ["myproj-sglang-*"], "status_action": "status",
        "actions": [{"name": "up", "label": "Start", "confirm": true}, {"name": "down", "label": "Stop", "danger": true},
                    {"name": "status", "label": "Check status"},
                    {"name": "switch", "label": "Switch model", "args": ["model-a", "model-b"], "confirm": true}]})
}

/// A container's cgroup memory block in bytes (numbers measured on a real SGLang container by the v1 tests).
pub fn cgroup(current: f64, anon: f64, file: f64, shmem: f64, kernel: f64) -> Value {
    let b = |x: f64| (x * GIB) as i64;
    json!({"mem": {"current": b(current), "anon": b(anon), "file": b(file), "shmem": b(shmem), "kernel": b(kernel),
                   "source": "cgroup"}, "cpu_pct": 1.0})
}

/// `cgroup()` with Python's defaults.
pub fn cgroup_default() -> Value {
    cgroup(68.6, 3.3, 64.5, 64.0, 0.6)
}

pub fn ports(host_port: i64, container_port: i64, ip: &str) -> Vec<Value> {
    vec![json!({"host_ip": ip, "host_port": host_port, "container_port": container_port})]
}

// ── a one-shot event (asyncio.Event) ─────────────────────────────────────────────────────────────────────
#[derive(Clone, Default)]
pub struct Gate(Arc<GateInner>);

#[derive(Default)]
struct GateInner {
    set: AtomicBool,
    notify: tokio::sync::Notify,
}

impl std::fmt::Debug for Gate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Gate({})", self.is_set())
    }
}

impl Gate {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn set(&self) {
        self.0.set.store(true, Ordering::SeqCst);
        self.0.notify.notify_waiters();
    }
    pub fn clear(&self) {
        self.0.set.store(false, Ordering::SeqCst);
    }
    pub fn is_set(&self) -> bool {
        self.0.set.load(Ordering::SeqCst)
    }
    pub async fn wait(&self) {
        loop {
            let n = self.0.notify.notified();
            tokio::pin!(n);
            n.as_mut().enable();
            if self.is_set() {
                return;
            }
            n.await;
        }
    }
}

// ── engines reached through http.request ───────────────────────────────────────────────────────────────
/// What an engine received (Python: the `httpx.Request` a route handler gets).
#[derive(Debug, Clone, PartialEq)]
pub struct HttpReq {
    pub method: String,
    pub port: i64,
    pub path: String,
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl HttpReq {
    /// A header, case-insensitively.
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
    pub fn json(&self) -> Value {
        serde_json::from_slice(&self.body).unwrap_or(Value::Null)
    }
    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }
}

/// What an engine answers (Python: `httpx.Response`). Header names are lower-case, as `dict(resp.headers)` gives.
#[derive(Debug, Clone, PartialEq)]
pub struct HttpResp {
    pub status: i64,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl HttpResp {
    /// `httpx.Response(status, text=...)`
    pub fn text(status: i64, text: &str) -> Self {
        Self {
            status,
            headers: vec![
                ("content-length".into(), text.len().to_string()),
                ("content-type".into(), "text/plain; charset=utf-8".into()),
            ],
            body: text.as_bytes().to_vec(),
        }
    }
    /// `httpx.Response(status, json=...)` (compact separators, non-ASCII kept)
    pub fn json(status: i64, v: &Value) -> Self {
        let body = serde_json::to_vec(v).expect("json");
        Self {
            status,
            headers: vec![
                ("content-length".into(), body.len().to_string()),
                ("content-type".into(), "application/json".into()),
            ],
            body,
        }
    }
    /// `httpx.Response(status, content=...)`
    pub fn bytes(status: i64, body: impl Into<Vec<u8>>) -> Self {
        let body = body.into();
        Self {
            status,
            headers: vec![("content-length".into(), body.len().to_string())],
            body,
        }
    }
    #[must_use]
    pub fn header(mut self, k: &str, v: &str) -> Self {
        let k = k.to_lowercase();
        self.headers.retain(|(kk, _)| *kk != k);
        self.headers.push((k, v.into()));
        self
    }
}

pub type HttpHandler = Arc<dyn Fn(&HttpReq) -> Result<HttpResp, NodeError> + Send + Sync>;

/// `http_handler(|req| HttpResp::text(200, "ok"))`
pub fn http_handler(f: impl Fn(&HttpReq) -> HttpResp + Send + Sync + 'static) -> HttpHandler {
    Arc::new(move |r: &HttpReq| Ok(f(r)))
}

// ── streams ──────────────────────────────────────────────────────────────────────────────────────────────
/// The node's record of one streaming request (Python keeps the FakeStream itself in `node.streams`).
#[derive(Debug, Default)]
pub struct StreamRec {
    pub op: String,
    pub uploaded: Mutex<Vec<u8>>,
    pub ended: Gate,
    pub cancelled: AtomicBool,
    pub cancel_event: Gate,
    pub done: AtomicBool,
}

impl StreamRec {
    pub fn uploaded(&self) -> Vec<u8> {
        self.uploaded.lock().expect("lock").clone()
    }
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }
    pub fn is_done(&self) -> bool {
        self.done.load(Ordering::SeqCst)
    }
    fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
        self.cancel_event.set();
    }
}

pub type Producer = Box<dyn FnOnce(Arc<StreamRec>) -> BoxStream<'static, StreamItem> + Send>;
pub type Finish = Box<dyn Fn() -> Result<Obj, NodeError> + Send + Sync>;

/// One streaming request. `producer` yields the events and DATA the node would send (started on the first `next()`);
/// `finish` gives the res result (or the NodeError the node would answer with).
pub struct FakeStream {
    pub rec: Arc<StreamRec>,
    producer: Option<Producer>,
    items: Option<BoxStream<'static, StreamItem>>,
    finish: Finish,
}

impl FakeStream {
    pub fn new(op: &str, producer: Producer, finish: Finish) -> Self {
        Self {
            rec: Arc::new(StreamRec {
                op: op.into(),
                ..StreamRec::default()
            }),
            producer: Some(producer),
            items: None,
            finish,
        }
    }

    /// A stream of fixed items and a fixed result.
    pub fn fixed(op: &str, items: Vec<StreamItem>, result: Result<Obj, NodeError>) -> Self {
        Self::new(
            op,
            Box::new(move |_| futures::stream::iter(items).boxed()),
            Box::new(move || result.clone()),
        )
    }
}

#[async_trait]
impl NodeStream for FakeStream {
    async fn next(&mut self) -> Option<StreamItem> {
        if self.items.is_none() {
            let p = self.producer.take()?;
            self.items = Some(p(Arc::clone(&self.rec)));
        }
        let item = self.items.as_mut()?.next().await;
        if item.is_none() {
            self.rec.done.store(true, Ordering::SeqCst);
            self.items = None;
        }
        item
    }
    async fn write(&mut self, data: &[u8]) -> Result<(), NodeError> {
        self.rec
            .uploaded
            .lock()
            .expect("lock")
            .extend_from_slice(data);
        Ok(())
    }
    async fn end(&mut self) -> Result<(), NodeError> {
        self.rec.ended.set();
        Ok(())
    }
    async fn result(&mut self) -> Result<Obj, NodeError> {
        (self.finish)()
    }
    async fn cancel(&mut self) {
        self.rec.cancel();
    }
}

impl Drop for FakeStream {
    fn drop(&mut self) {
        // an iteration that was started and dropped before its end is a consumer that went away
        if self.items.is_some() && !self.rec.is_done() {
            self.rec.cancel();
        }
        self.items = None;
        self.rec.done.store(true, Ordering::SeqCst);
    }
}

/// What a real client returns from open() when the request cannot start: the error shows when the stream is used.
pub struct FailedStream {
    error: NodeError,
}

impl FailedStream {
    pub fn new(error: NodeError) -> Self {
        Self { error }
    }
}

#[async_trait]
impl NodeStream for FailedStream {
    async fn next(&mut self) -> Option<StreamItem> {
        None
    }
    async fn write(&mut self, _data: &[u8]) -> Result<(), NodeError> {
        Err(self.error.clone())
    }
    async fn end(&mut self) -> Result<(), NodeError> {
        Err(self.error.clone())
    }
    async fn result(&mut self) -> Result<Obj, NodeError> {
        Err(self.error.clone())
    }
    async fn cancel(&mut self) {}
}

/// What `scripts.run` of one (script, action) does on the fake node.
#[derive(Clone, Default)]
pub struct ScriptRun {
    /// (stream, line)
    pub lines: Vec<(String, String)>,
    pub rc: i64,
    /// when set, the run waits here after its first line (or until cancelled)
    pub gate: Option<Gate>,
    /// e.g. add the containers the script would start (only when rc == 0)
    pub effect: Option<Arc<dyn Fn(&FakeNode) + Send + Sync>>,
    pub error: Option<NodeError>,
}

impl ScriptRun {
    pub fn lines(lines: &[(&str, &str)]) -> Self {
        Self {
            lines: lines
                .iter()
                .map(|(s, l)| ((*s).to_string(), (*l).to_string()))
                .collect(),
            ..Self::default()
        }
    }
}

#[derive(Default)]
struct Outcome {
    result: Option<Obj>,
    error: Option<NodeError>,
    cancelled: bool,
}

fn finish_from(host_id: String, o: Arc<Mutex<Outcome>>) -> Finish {
    Box::new(move || {
        let o = o.lock().expect("lock");
        if let Some(e) = &o.error {
            return Err(e.clone());
        }
        if o.cancelled {
            return Err(NodeError::new(&host_id, "cancelled", "cancelled"));
        }
        Ok(o.result.clone().unwrap_or_default())
    })
}

// ── equality as Python's `==` on JSON values ────────────────────────────────────────────────────────────
fn py_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => match (x.as_i64(), y.as_i64()) {
            (Some(i), Some(j)) => i == j,
            _ => x.as_f64() == y.as_f64(),
        },
        (Value::Bool(x), Value::Number(n)) | (Value::Number(n), Value::Bool(x)) => {
            n.as_f64() == Some(if *x { 1.0 } else { 0.0 })
        }
        (Value::Array(x), Value::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| py_eq(p, q))
        }
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len() && x.iter().all(|(k, v)| y.get(k).is_some_and(|w| py_eq(v, w)))
        }
        _ => a == b,
    }
}

/// Where `kept` (after clean) differs from what the hub `sent`: those are fields the real node would drop.
pub fn lost(sent: &Value, kept: Option<&Value>, path: &str) -> Vec<String> {
    match sent {
        Value::Null => Vec::new(),
        Value::Object(o) => {
            let Some(Value::Object(k)) = kept else {
                return vec![if path.is_empty() {
                    "<root>".to_string()
                } else {
                    path.to_string()
                }];
            };
            o.iter()
                .flat_map(|(key, v)| {
                    let p = if path.is_empty() {
                        key.clone()
                    } else {
                        format!("{path}.{key}")
                    };
                    lost(v, k.get(key), &p)
                })
                .collect()
        }
        Value::Array(a) => {
            let Some(Value::Array(k)) = kept else {
                return vec![path.to_string()];
            };
            if k.len() != a.len() {
                return vec![path.to_string()];
            }
            a.iter()
                .zip(k)
                .enumerate()
                .flat_map(|(i, (x, y))| lost(x, Some(y), &format!("{path}[{i}]")))
                .collect()
        }
        _ => {
            if kept.is_some_and(|k| py_eq(sent, k)) {
                Vec::new()
            } else {
                vec![path.to_string()]
            }
        }
    }
}

pub fn proto_name_ok(name: &str) -> bool {
    static RE: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
        regex::Regex::new(r"\Aec-[a-z0-9][a-z0-9-]{0,58}\z").expect("re")
    });
    RE.is_match(name)
}

/// Every `host_port` a container publishes (Python: `for p in c.get("ports") or []: p["host_port"]`).
fn raw_ports(c: &Obj) -> Vec<i64> {
    c.get("ports")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|p| p.get("host_port").and_then(Value::as_i64))
        .collect()
}

fn truthy(v: Option<&Value>) -> bool {
    v.is_some_and(ec_proto::sanitize::truthy)
}

fn s_of(v: Option<&Value>) -> String {
    v.map(ec_proto::pyfmt::str_value).unwrap_or_default()
}

fn sha1_hex(s: &str) -> String {
    hex::encode(sha1::Sha1::digest(s.as_bytes()))
}

// ── the node ─────────────────────────────────────────────────────────────────────────────────────────────
/// `set_gpu_free`: None = idle (all free), one number for every GPU, or one per GPU index.
#[derive(Debug, Clone, PartialEq)]
pub enum GpuFree {
    Idle,
    All(f64),
    Each(Vec<f64>),
}

pub type OpFn = Arc<dyn Fn(&FakeNode, &Obj) -> Result<Obj, NodeError> + Send + Sync>;
pub type AsyncOpFn =
    Arc<dyn Fn(Arc<FakeNode>, Obj) -> BoxFuture<'static, Result<Obj, NodeError>> + Send + Sync>;
pub type OpenFn = Arc<dyn Fn(&FakeNode, &Obj) -> Result<FakeStream, NodeError> + Send + Sync>;

#[derive(Clone)]
enum OpOverride {
    Sync(OpFn),
    Async(AsyncOpFn),
}

/// Everything about a fake node a test may read or change (Python: the FakeNode's attributes).
pub struct NodeState {
    pub hello: Obj,
    pub state: String,
    pub link_error: Option<String>,
    pub connected_since: f64,
    pub skew_s: f64,
    // live numbers
    pub gpu_free: GpuFree,
    /// gpu index -> [{pid, container, used_bytes, name}]
    pub gpu_procs: BTreeMap<i64, Vec<Value>>,
    /// GiB: total, available, free, cached, shmem, swap_total, swap_free
    pub mem: Obj,
    /// container name -> cgroup block
    pub cg: Obj,
    pub seq: i64,
    // inventory
    pub containers: Obj,
    pub endpoints: Vec<Value>,
    pub script_status: Obj,
    pub disks: Vec<Value>,
    // behaviour
    pub logs: HashMap<String, String>,
    pub missing_images: HashSet<String>,
    pub port_taken: HashSet<i64>,
    pub network_internal: bool,
    pub fail: HashMap<String, NodeError>,
    pub http_routes: HashMap<(Option<i64>, String), HttpHandler>,
    pub http_requests: Vec<HttpReq>,
    pub http_chunk: usize,
    pub script_runs: HashMap<(String, String), ScriptRun>,
    pub running_scripts: HashSet<String>,
    pub script_calls: Vec<(String, String, Option<String>)>,
    /// kinds `latest()` pretends not to have (Python tests monkeypatch `latest`)
    pub hidden: HashSet<String>,
    // records
    pub calls: Vec<(String, Obj)>,
    pub run_specs: Vec<Value>,
    pub metrics_config: Option<Obj>,
    pub streams: Vec<Arc<StreamRec>>,
    stale_sample: Option<Arc<Obj>>,
    overrides: HashMap<String, OpOverride>,
    open_overrides: HashMap<String, OpenFn>,
}

/// One machine's node. Defaults: manage mode, two 96 GiB GPUs, an idle host, the `fake/engine:*` images allowed.
pub struct FakeNode {
    pub host: HostConfig,
    me: Weak<FakeNode>,
    state: Mutex<NodeState>,
}

impl std::fmt::Debug for FakeNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FakeNode")
            .field("host", &self.host)
            .finish_non_exhaustive()
    }
}

impl FakeNode {
    /// `FakeNode(host, **hello_kw)`: the hello's hostname is the host's label unless `o.hostname` says otherwise.
    pub fn new(host: HostConfig, o: HelloOpts) -> Arc<Self> {
        let mut o = o;
        if o.hostname.is_none() {
            o.hostname = Some(host.label.clone());
        }
        Self::with_hello(host, make_hello(&o))
    }

    /// `FakeNode(host, hello=...)`
    pub fn with_hello(host: HostConfig, hello: Obj) -> Arc<Self> {
        let mem = obj(
            json!({"total": 246.0, "available": 54.0, "free": 3.0, "cached": 90.0, "shmem": 66.0,
                             "swap_total": 32.0, "swap_free": 8.0}),
        );
        Arc::new_cyclic(|me| Self {
            host,
            me: me.clone(),
            state: Mutex::new(NodeState {
                hello,
                state: "up".into(),
                link_error: None,
                connected_since: 1000.0,
                skew_s: 0.0,
                gpu_free: GpuFree::Idle,
                gpu_procs: BTreeMap::new(),
                mem,
                cg: Obj::new(),
                seq: 1,
                containers: Obj::new(),
                endpoints: Vec::new(),
                script_status: Obj::new(),
                disks: Vec::new(),
                logs: HashMap::new(),
                missing_images: HashSet::new(),
                port_taken: HashSet::new(),
                network_internal: true,
                fail: HashMap::new(),
                http_routes: HashMap::new(),
                http_requests: Vec::new(),
                http_chunk: 64,
                script_runs: HashMap::new(),
                running_scripts: HashSet::new(),
                script_calls: Vec::new(),
                hidden: HashSet::new(),
                calls: Vec::new(),
                run_specs: Vec::new(),
                metrics_config: None,
                streams: Vec::new(),
                stale_sample: None,
                overrides: HashMap::new(),
                open_overrides: HashMap::new(),
            }),
        })
    }

    /// A host config (id, label; `ssh` for a remote one).
    pub fn host_config(id: &str, label: &str, ssh: Option<&str>) -> HostConfig {
        HostConfig {
            id: id.into(),
            label: label.into(),
            ssh: ssh.map(str::to_string),
            port: None,
            ssh_config: false,
            enabled: true,
        }
    }

    /// The node's state (lock it briefly: never across an await).
    pub fn st(&self) -> MutexGuard<'_, NodeState> {
        self.state.lock().expect("fake node lock")
    }

    fn me(&self) -> Arc<FakeNode> {
        self.me.upgrade().expect("the fake node is alive")
    }

    // -- test data -------------------------------------------------------------------------------------
    pub fn set_state(&self, state: &str, error: Option<&str>) {
        let mut st = self.st();
        st.state = state.into();
        st.link_error = error.map(str::to_string);
    }

    pub fn set_gpu_free(&self, free: GpuFree) {
        self.st().gpu_free = free;
    }

    /// `node.mem.update(total=..., ...)` (GiB)
    pub fn set_mem(&self, kv: &[(&str, f64)]) {
        let mut st = self.st();
        for (k, v) in kv {
            st.mem.insert((*k).to_string(), json!(v));
        }
    }

    /// The node pushed a newer sample (the hub's metrics service records one system point per new `seq`).
    pub fn tick_sample(&self) {
        self.st().seq += 1;
    }

    pub fn mode(&self) -> String {
        self.st().hello["policy"]["mode"]
            .as_str()
            .unwrap_or("monitor")
            .to_string()
    }

    pub fn set_mode(&self, mode: &str) {
        if let Some(Value::Object(p)) = self.st().hello.get_mut("policy") {
            p.insert("mode".into(), json!(mode));
        }
    }

    pub fn policy(&self) -> Obj {
        obj(self
            .st()
            .hello
            .get("policy")
            .cloned()
            .unwrap_or(Value::Null))
    }

    /// Change one policy field of the hello (`node.hello["policy"][k] = v`).
    pub fn set_policy(&self, key: &str, v: Value) {
        if let Some(Value::Object(p)) = self.st().hello.get_mut("policy") {
            p.insert(key.into(), v);
        }
    }

    pub fn root_path(&self, root_id: &str) -> Result<String, NodeError> {
        let pol = self.policy();
        for r in pol
            .get("model_roots")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            if r.get("id").and_then(Value::as_str) == Some(root_id) {
                return Ok(s_of(r.get("path")));
            }
        }
        Err(NodeError::new(
            &self.host.id,
            "bad_args",
            format!("{root_id} is not a model root of this policy"),
        ))
    }

    /// Options of `add_container` (Python keyword arguments with their defaults).
    pub fn add_container(&self, name: &str, o: ContainerOpts) -> Obj {
        let mut labels = o.labels.clone();
        if o.managed && !labels.contains_key("engine-console") {
            labels.insert("engine-console".into(), json!("1"));
        }
        let mut c = obj(json!({
            "name": name, "id": &sha1_hex(name)[..12], "image": o.image, "state": o.state, "status": o.state,
            "created": 50.0, "started_at": o.started_at, "labels": labels, "managed": o.managed,
            "role": labels.get("engine-console.role"), "instance": labels.get("engine-console.instance"),
            "ports": o.host_ports, "argv": o.argv, "params": o.params, "gpus": o.gpus, "all_gpus": o.all_gpus,
            "mounts": o.mounts, "exit_code": o.exit_code}));
        if let Some(e) = o.engine {
            c.insert("engine".into(), json!(e));
        }
        self.st()
            .containers
            .insert(name.into(), Value::Object(c.clone()));
        c
    }

    pub fn add_endpoint(&self, port: i64, o: EndpointOpts) -> Obj {
        let e = obj(
            json!({"port": port, "container": o.container, "engine": o.engine, "models": o.models,
                           "health_status": o.health, "models_status": o.models_status,
                           "metrics_status": o.metrics, "auth_required": o.auth_required}),
        );
        let mut st = self.st();
        st.endpoints
            .retain(|x| x.get("port").and_then(Value::as_i64) != Some(port));
        st.endpoints.push(Value::Object(e.clone()));
        e
    }

    /// Every call, in order: (op, args as the hub sent them).
    pub fn calls(&self) -> Vec<(String, Obj)> {
        self.st().calls.clone()
    }

    /// The arguments of every call of `op`, in order.
    pub fn ops(&self, op: &str) -> Vec<Obj> {
        self.st()
            .calls
            .iter()
            .filter(|(o, _)| o == op)
            .map(|(_, a)| a.clone())
            .collect()
    }

    pub fn streams(&self) -> Vec<Arc<StreamRec>> {
        self.st().streams.clone()
    }

    pub fn http_requests(&self) -> Vec<HttpReq> {
        self.st().http_requests.clone()
    }

    /// Answer GET/POST `path` on `port` (None: any port) like an engine would.
    pub fn serve(&self, port: Option<i64>, path: &str, handler: HttpHandler) {
        self.st().http_routes.insert((port, path.into()), handler);
    }

    /// Make `op` fail with this error (call and open), until `unfail`.
    pub fn fail(&self, op: &str, e: NodeError) {
        self.st().fail.insert(op.into(), e);
    }

    pub fn unfail(&self, op: &str) {
        self.st().fail.remove(op);
    }

    /// Replace (or add) a request/response op.
    pub fn on(&self, op: &str, f: OpFn) {
        self.st().overrides.insert(op.into(), OpOverride::Sync(f));
    }

    /// Replace (or add) a request/response op with an async handler (it may wait on a `Gate`).
    pub fn on_async(&self, op: &str, f: AsyncOpFn) {
        self.st().overrides.insert(op.into(), OpOverride::Async(f));
    }

    /// Replace (or add) a streaming op.
    pub fn on_open(&self, op: &str, f: OpenFn) {
        self.st().open_overrides.insert(op.into(), f);
    }

    pub fn script_run(&self, name: &str, action: &str, run: ScriptRun) {
        self.st()
            .script_runs
            .insert((name.into(), action.into()), run);
    }

    // -- the pushed messages ---------------------------------------------------------------------------
    fn sample_of(&self, st: &mut NodeState) -> Option<Arc<Obj>> {
        if !matches!(st.state.as_str(), "up" | "stale") {
            return st.stale_sample.clone();
        }
        let mut gpus = Obj::new();
        for g in st
            .hello
            .get("gpus")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default()
        {
            let total = g.get("total_bytes").and_then(Value::as_i64).unwrap_or(0);
            let idx = g.get("index").and_then(Value::as_i64).unwrap_or(0);
            let free_gib = match &st.gpu_free {
                GpuFree::Idle => total as f64 / GIB,
                GpuFree::All(f) => *f,
                GpuFree::Each(v) => v
                    .get(usize::try_from(idx).unwrap_or(usize::MAX))
                    .copied()
                    .expect("set_gpu_free: no number for that GPU index"),
            };
            let used = (total - (free_gib * GIB) as i64).max(0);
            let unified = truthy(g.get("unified_memory"));
            let uuid = s_of(g.get("uuid"));
            gpus.insert(
                uuid,
                json!({"util_pct": 5.0, "mem_used_bytes": if unified { Value::Null } else { json!(used) },
                       "mem_total_bytes": if unified { Value::Null } else { json!(total) },
                       "temp_c": 40.0, "power_w": 90.0, "power_limit_w": 300.0, "fan_pct": 30.0,
                       "procs": st.gpu_procs.get(&idx).cloned().unwrap_or_default()}),
            );
        }
        let mem: Obj = st
            .mem
            .iter()
            .map(|(k, v)| (k.clone(), json!((v.as_f64().unwrap_or(0.0) * GIB) as i64)))
            .collect();
        let raw = json!({"type": "sample", "seq": st.seq, "t": 1.0, "gpus": gpus, "mem": mem,
                         "cpu": {"load1": 0.5, "util_pct": 3.0}, "containers": st.cg});
        let out = ec_proto::sanitize_push(&obj(raw))
            .ok()
            .flatten()
            .map(Arc::new);
        st.stale_sample.clone_from(&out);
        out
    }

    fn inventory_of(st: &NodeState) -> Option<Arc<Obj>> {
        let raw = json!({"type": "inventory", "t": 1.0,
                         "containers": st.containers.values().cloned().collect::<Vec<_>>(),
                         "endpoints": st.endpoints,
                         "scripts": st.script_status.values().cloned().collect::<Vec<_>>()});
        ec_proto::sanitize_push(&obj(raw))
            .ok()
            .flatten()
            .map(Arc::new)
    }

    // -- request checks --------------------------------------------------------------------------------
    fn gate(&self, op: &str, args: &Obj) -> Result<(), NodeError> {
        let hid = &self.host.id;
        let (state, link_error, mode) = {
            let st = self.st();
            (
                st.state.clone(),
                st.link_error.clone(),
                st.hello["policy"]["mode"]
                    .as_str()
                    .unwrap_or("")
                    .to_string(),
            )
        };
        if state != "up" {
            return Err(NodeError::unavailable(
                hid,
                link_error.unwrap_or_else(|| format!("the node is {state}")),
            ));
        }
        if !specs::is_op(op) {
            return Err(NodeError::new(hid, "unknown_op", op));
        }
        let cleaned = ec_proto::clean_args(op, &Value::Object(args.clone()))
            .map_err(|e| NodeError::new(hid, "unknown_op", e.to_string()))?;
        let gone = lost(
            &Value::Object(args.clone()),
            Some(&Value::Object(cleaned.clone())),
            "",
        );
        if !gone.is_empty() {
            // the real node would silently drop these: a hub bug (misspelt, over-long or wrongly typed argument)
            return Err(NodeError::new(
                hid,
                "bad_args",
                format!("{op}: the node would drop {}", gone.join(", ")),
            ));
        }
        let missing = ec_proto::missing_args(op, &cleaned);
        if !missing.is_empty() {
            return Err(NodeError::new(
                hid,
                "bad_args",
                format!("{op}: missing {}", missing.join(", ")),
            ));
        }
        if mode != "manage" && !specs::is_monitor_op(op) {
            return Err(NodeError::new(
                hid,
                "monitor_only",
                format!("{op} needs the policy mode manage; this node is in monitor mode"),
            ));
        }
        Ok(())
    }

    fn err(&self, code: &str, msg: impl Into<String>) -> NodeError {
        NodeError::new(&self.host.id, code, msg)
    }

    fn owned(&self, st: &NodeState, name: &str, args: &Obj) -> Result<(), NodeError> {
        let Some(Value::Object(c)) = st.containers.get(name) else {
            return Err(self.err("not_found", format!("no container {name}")));
        };
        let labels = c.get("labels").cloned().unwrap_or(Value::Null);
        if !truthy(c.get("managed"))
            || labels.get("engine-console").and_then(Value::as_str) != Some("1")
        {
            return Err(self.err(
                "not_owned",
                format!("{name} is not a container this console created"),
            ));
        }
        if truthy(args.get("role")) && labels.get("engine-console.role") != args.get("role") {
            return Err(self.err(
                "not_owned",
                format!("{name} does not have role {}", s_of(args.get("role"))),
            ));
        }
        if truthy(args.get("instance"))
            && labels.get("engine-console.instance") != args.get("instance")
        {
            return Err(self.err("not_owned", format!("{name} belongs to another instance")));
        }
        Ok(())
    }

    fn check_spec(&self, spec: &Obj) -> Result<(), NodeError> {
        if !proto_name_ok(spec.get("name").and_then(Value::as_str).unwrap_or("")) {
            return Err(self.err("bad_args", "bad container name"));
        }
        let pol = self.policy();
        let images: Vec<String> = pol
            .get("engine_images")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect();
        let image = s_of(spec.get("image"));
        if !ec_hub::services::nodeview::image_allowed(&image, &images) {
            return Err(self.err(
                "policy_denied",
                format!("image {image} is not allowed by this node's policy"),
            ));
        }
        let known: HashSet<String> = self
            .st()
            .hello
            .get("gpus")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .map(|g| s_of(g.get("uuid")))
            .collect();
        if spec
            .get("gpu_uuids")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .any(|u| !known.contains(&s_of(Some(u))))
        {
            return Err(self.err("bad_args", "an unknown GPU uuid"));
        }
        if spec
            .get("labels")
            .and_then(Value::as_object)
            .into_iter()
            .flatten()
            .any(|(k, _)| !k.starts_with("engine-console."))
        {
            return Err(self.err("bad_args", "labels must start with engine-console."));
        }
        if truthy(spec.get("ipc_host")) && pol.get("allow_ipc_host") == Some(&Value::Bool(false)) {
            return Err(self.err(
                "policy_denied",
                "host IPC is not allowed by this node's policy",
            ));
        }
        for m in spec
            .get("mounts")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            self.root_path(&s_of(m.get("root")))?;
        }
        Ok(())
    }

    // -- node ops (request/response) -------------------------------------------------------------------
    fn builtin_op(&self, op: &str, args: &Obj) -> Result<Obj, NodeError> {
        match op {
            "node.info" => Ok(self.st().hello.clone()),
            "metrics.configure" => {
                self.st().metrics_config = Some(args.clone());
                Ok(Obj::new())
            }
            "net.ensure_internal" => {
                if !self.st().network_internal {
                    return Err(self.err(
                        "conflict",
                        "the engine network exists but is not --internal; refusing to use it",
                    ));
                }
                Ok(obj(json!({"network": self.policy().get("engine_network")})))
            }
            "docker.image_present" => {
                let image = s_of(args.get("image"));
                let present = !self.st().missing_images.contains(&image);
                Ok(obj(json!({"present": present, "id": "sha256:x"})))
            }
            "docker.images" => Ok(obj(json!({"images": []}))),
            "docker.render" => self.op_docker_render(args),
            "docker.run" => self.op_docker_run(args),
            "gateway.start" => self.op_gateway_start(args),
            "docker.stop" => {
                let name = s_of(args.get("name"));
                let mut st = self.st();
                self.owned(&st, &name, args)?;
                let mut freed = Vec::new();
                if let Some(Value::Object(c)) = st.containers.get_mut(&name) {
                    c.insert("state".into(), json!("exited"));
                    c.insert("status".into(), json!("exited"));
                    c.insert("exit_code".into(), json!(0));
                    freed = raw_ports(c);
                }
                for p in freed {
                    st.port_taken.remove(&p);
                }
                Ok(Obj::new())
            }
            "docker.rm" => {
                let name = s_of(args.get("name"));
                let mut st = self.st();
                self.owned(&st, &name, args)?;
                let freed = match st.containers.get(&name) {
                    Some(Value::Object(c)) => raw_ports(c),
                    _ => Vec::new(),
                };
                for p in freed {
                    st.port_taken.remove(&p);
                }
                st.containers.shift_remove(&name);
                Ok(Obj::new())
            }
            "docker.inspect" => {
                let name = s_of(args.get("name"));
                match self.st().containers.get(&name) {
                    Some(c) => Ok(obj(json!({"info": c}))),
                    None => Err(self.err("not_found", format!("no container {name}"))),
                }
            }
            "docker.logs" => {
                let name = s_of(args.get("name"));
                let st = self.st();
                // logs of containers the console did not create can hold secrets: refused
                self.owned(&st, &name, &Obj::new())?;
                Ok(obj(
                    json!({"text": st.logs.get(&name).cloned().unwrap_or_default(), "truncated": false}),
                ))
            }
            "docker.external" => self.op_docker_external(args),
            _ => Err(self.err(
                "unknown_op",
                format!("the fake node does not implement {op}"),
            )),
        }
    }

    fn op_docker_render(&self, args: &Obj) -> Result<Obj, NodeError> {
        let spec = obj(args.get("spec").cloned().unwrap_or(Value::Null));
        self.check_spec(&spec)?;
        let net = s_of(self.policy().get("engine_network"));
        let mut argv: Vec<String> = [
            "docker",
            "run",
            "-d",
            "--name",
            &s_of(spec.get("name")),
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",
            "--pids-limit",
            "4096",
            "--network",
            &net,
        ]
        .map(String::from)
        .to_vec();
        for (k, v) in spec
            .get("env")
            .and_then(Value::as_object)
            .into_iter()
            .flatten()
        {
            argv.push("-e".into());
            argv.push(format!("{k}={}", s_of(Some(v))));
        }
        for (k, _) in spec
            .get("secret_env")
            .and_then(Value::as_object)
            .into_iter()
            .flatten()
        {
            // a name only: the node never renders a secret value
            argv.push("-e".into());
            argv.push(k.clone());
        }
        argv.push(s_of(spec.get("image")));
        for a in spec
            .get("argv")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            argv.push(s_of(Some(a)));
        }
        Ok(obj(json!({"argv": argv})))
    }

    fn op_docker_run(&self, args: &Obj) -> Result<Obj, NodeError> {
        let spec = obj(args.get("spec").cloned().unwrap_or(Value::Null));
        self.check_spec(&spec)?;
        self.st().run_specs.push(Value::Object(spec.clone()));
        let mut labels = obj(json!({"engine-console": "1"}));
        for (k, v) in spec
            .get("labels")
            .and_then(Value::as_object)
            .into_iter()
            .flatten()
        {
            labels.insert(k.clone(), v.clone());
        }
        let mut mounts = Vec::new();
        for m in spec
            .get("mounts")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let base = self.root_path(&s_of(m.get("root")))?;
            let rel = if truthy(m.get("rel")) {
                s_of(m.get("rel"))
            } else {
                String::new()
            };
            let source = if rel.is_empty() {
                base
            } else {
                format!("{base}/{rel}")
            };
            mounts.push(json!({"source": source, "dest": m.get("dest"),
                               "ro": m.as_object().and_then(|o| o.get("ro")).cloned().unwrap_or(json!(true)),
                               "root": m.get("root"), "rel": rel}));
        }
        let name = s_of(spec.get("name"));
        let running = self
            .st()
            .containers
            .get(&name)
            .is_some_and(|c| c.get("state").and_then(Value::as_str) == Some("running"));
        if running {
            return Err(self.err(
                "conflict",
                format!("a container named {name} is already running"),
            ));
        }
        let strs = |k: &str| -> Vec<String> {
            spec.get(k)
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .map(|v| s_of(Some(v)))
                .collect()
        };
        self.add_container(
            &name,
            ContainerOpts {
                image: s_of(spec.get("image")),
                engine: labels
                    .get("engine-console.engine")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                managed: true,
                labels: labels.clone(),
                argv: strs("argv"),
                mounts,
                gpus: strs("gpu_uuids"),
                started_at: Some(1.0),
                ..ContainerOpts::default()
            },
        );
        Ok(obj(json!({"id": &sha1_hex(&name)[..12]})))
    }

    fn op_gateway_start(&self, args: &Obj) -> Result<Obj, NodeError> {
        let pol = self.policy();
        let pr: Vec<i64> = pol
            .get("port_range")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_i64)
            .collect();
        let (lo, hi) = (
            pr.first().copied().unwrap_or(0),
            pr.get(1).copied().unwrap_or(0),
        );
        let hp = args.get("host_port").and_then(Value::as_i64).unwrap_or(0);
        if !(lo <= hp && hp <= hi) {
            return Err(self.err("bad_args", "host_port is outside the policy's port range"));
        }
        if self.st().port_taken.contains(&hp) {
            return Err(self.err("port_in_use", format!("port {hp} is already in use")));
        }
        let eh = s_of(args.get("engine_host"));
        let managed = self
            .st()
            .containers
            .get(&eh)
            .is_some_and(|c| truthy(c.get("managed")));
        if !managed {
            return Err(self.err(
                "not_found",
                format!("{eh} is not a container this console created"),
            ));
        }
        if !truthy(pol.get("gateway_image")) {
            return Err(self.err("policy_denied", "the policy names no gateway_image"));
        }
        let name = s_of(args.get("name"));
        self.add_container(
            &name,
            ContainerOpts {
                image: s_of(pol.get("gateway_image")),
                managed: true,
                started_at: Some(1.0),
                labels: obj(json!({"engine-console.role": "gateway", "engine-console.instance": args.get("instance")})),
                host_ports: ports(hp, 8080, "127.0.0.1"),
                ..ContainerOpts::default()
            },
        );
        self.st().port_taken.insert(hp);
        let chars: Vec<char> = name.chars().collect();
        let tail: String = chars[chars.len().saturating_sub(6)..].iter().collect();
        Ok(obj(json!({"id": format!("gw{tail}")})))
    }

    fn op_docker_external(&self, args: &Obj) -> Result<Obj, NodeError> {
        let pol = self.policy();
        if pol.get("external_control").and_then(Value::as_str) != Some("stop_start") {
            return Err(self.err(
                "policy_denied",
                "this node's policy does not allow controlling other engines",
            ));
        }
        let name = s_of(args.get("name"));
        let mut st = self.st();
        let Some(Value::Object(c)) = st.containers.get(&name) else {
            return Err(self.err("not_found", format!("no container {name}")));
        };
        if truthy(c.get("managed")) {
            return Err(self.err(
                "not_owned",
                "that container is the console's own: use the instance routes",
            ));
        }
        let scripted = pol
            .get("scripts")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .any(|s| matches_script(&name, s));
        if scripted {
            return Err(self.err(
                "policy_denied",
                "that engine is managed by a script: run the script's action",
            ));
        }
        let action = s_of(args.get("action"));
        if action != "stop" && action != "start" {
            return Err(self.err("bad_args", "action must be stop or start"));
        }
        if let Some(Value::Object(c)) = st.containers.get_mut(&name) {
            c.insert(
                "state".into(),
                json!(if action == "stop" {
                    "exited"
                } else {
                    "running"
                }),
            );
        }
        Ok(Obj::new())
    }

    // -- node ops (streams) ----------------------------------------------------------------------------
    fn builtin_open(&self, op: &str, args: &Obj) -> Result<FakeStream, NodeError> {
        match op {
            "docker.logs_follow" => self.open_logs_follow(args),
            "http.request" => self.open_http_request(args),
            "scripts.run" => self.open_scripts_run(args),
            _ => Err(self.err(
                "unknown_op",
                format!("the fake node does not implement {op}"),
            )),
        }
    }

    fn open_logs_follow(&self, args: &Obj) -> Result<FakeStream, NodeError> {
        let name = s_of(args.get("name"));
        let lines: Vec<String> = {
            let st = self.st();
            self.owned(&st, &name, &Obj::new())?;
            let text = st.logs.get(&name).cloned().unwrap_or_default();
            ec_hub::services::common::py_splitlines(&text)
                .into_iter()
                .map(str::to_string)
                .collect()
        };
        let n = lines.len();
        Ok(FakeStream::new(
            "docker.logs_follow",
            Box::new(move |s: Arc<StreamRec>| {
                Box::pin(async_stream::stream! {
                    for ln in lines {
                        if s.is_cancelled() {
                            return;
                        }
                        yield StreamItem::Event(obj(json!({"line": ln})));
                        tokio::task::yield_now().await;
                    }
                })
            }),
            Box::new(move || Ok(obj(json!({"lines": n})))),
        ))
    }

    fn open_http_request(&self, args: &Obj) -> Result<FakeStream, NodeError> {
        let port = args.get("port").and_then(Value::as_i64).unwrap_or(0);
        let method = s_of(args.get("method"));
        let path = s_of(args.get("path"));
        let allowed: HashSet<i64> = {
            let st = self.st();
            let mut a: HashSet<i64> = st
                .containers
                .values()
                .filter_map(Value::as_object)
                .filter(|c| truthy(c.get("managed")))
                .flat_map(ec_hub::services::nodeview::host_ports)
                .collect();
            a.extend(
                st.endpoints
                    .iter()
                    .filter_map(|e| e.get("port").and_then(Value::as_i64)),
            );
            a
        };
        if !allowed.contains(&port) {
            return Err(self.err(
                "policy_denied",
                format!("port {port} is neither a gateway nor a known engine endpoint"),
            ));
        }
        if method != "GET" && method != "POST" {
            return Err(self.err("policy_denied", "only GET and POST"));
        }
        if !HTTP_PREFIXES.iter().any(|p| path.starts_with(p)) {
            let short: String = path.chars().take(40).collect();
            return Err(self.err("policy_denied", format!("path {short} is not allowed")));
        }
        let body_len = args.get("body_len").and_then(Value::as_i64).unwrap_or(0);
        let headers: Vec<(String, String)> = args
            .get("headers")
            .and_then(Value::as_object)
            .into_iter()
            .flatten()
            .map(|(k, v)| (k.clone(), s_of(Some(v))))
            .collect();
        let outcome = Arc::new(Mutex::new(Outcome::default()));
        let o2 = Arc::clone(&outcome);
        let node = self.me();
        let hid = self.host.id.clone();
        Ok(FakeStream::new(
            "http.request",
            Box::new(move |s: Arc<StreamRec>| {
                Box::pin(async_stream::stream! {
                    if body_len > 0 {
                        s.ended.wait().await;
                        if s.uploaded().len() as i64 != body_len {
                            o2.lock().expect("lock").error = Some(NodeError::new(&hid, "bad_args",
                                "uploaded body does not match body_len"));
                            return;
                        }
                    }
                    let req = HttpReq {
                        method: method.clone(),
                        port,
                        path: path.clone(),
                        url: format!("http://127.0.0.1:{port}{path}"),
                        headers: headers.clone(),
                        body: s.uploaded(),
                    };
                    let (route, chunk) = {
                        let mut st = node.st();
                        st.http_requests.push(req.clone());
                        let r = st.http_routes.get(&(Some(port), path.clone()))
                            .or_else(|| st.http_routes.get(&(None, path.clone())))
                            .cloned();
                        (r, st.http_chunk.max(1))
                    };
                    let resp = match route {
                        Some(f) => match f(&req) {
                            Ok(r) => r,
                            Err(e) => {
                                o2.lock().expect("lock").error = Some(e);
                                return;
                            }
                        },
                        None => HttpResp::text(503, "not up"),
                    };
                    let hdrs: Obj = resp.headers.iter().map(|(k, v)| (k.clone(), json!(v))).collect();
                    yield StreamItem::Event(obj(json!({"status": resp.status, "headers": hdrs})));
                    for piece in resp.body.chunks(chunk) {
                        if s.is_cancelled() {
                            o2.lock().expect("lock").cancelled = true;
                            return;
                        }
                        yield StreamItem::Data(Bytes::copy_from_slice(piece));
                        tokio::task::yield_now().await;
                    }
                    o2.lock().expect("lock").result = Some(obj(json!({"status": resp.status, "bytes": resp.body.len()})));
                })
            }),
            finish_from(self.host.id.clone(), outcome),
        ))
    }

    fn open_scripts_run(&self, args: &Obj) -> Result<FakeStream, NodeError> {
        let name = s_of(args.get("name"));
        let action = s_of(args.get("action"));
        let arg = args.get("arg").and_then(Value::as_str).map(str::to_string);
        let pol = self.policy();
        let script = pol
            .get("scripts")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .find(|s| s.get("name").and_then(Value::as_str) == Some(name.as_str()))
            .cloned()
            .ok_or_else(|| self.err("not_found", format!("no script {name}")))?;
        let act = script
            .get("actions")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .find(|a| a.get("name").and_then(Value::as_str) == Some(action.as_str()))
            .cloned()
            .ok_or_else(|| {
                self.err("not_found", format!("script {name} has no action {action}"))
            })?;
        let allowed: Vec<String> = act
            .get("args")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect();
        let has_args = truthy(act.get("args"));
        if (has_args && !arg.as_ref().is_some_and(|a| allowed.contains(a)))
            || (arg.is_some() && !has_args)
        {
            return Err(self.err("bad_args", "argument not allowed for this action"));
        }
        let run = {
            let mut st = self.st();
            if st.running_scripts.contains(&name) {
                return Err(self.err("busy", format!("script {name} is already running")));
            }
            st.script_calls.push((name.clone(), action.clone(), arg));
            st.running_scripts.insert(name.clone());
            st.script_runs
                .get(&(name.clone(), action))
                .cloned()
                .unwrap_or_default()
        };
        let outcome = Arc::new(Mutex::new(Outcome::default()));
        let o2 = Arc::clone(&outcome);
        let node = self.me();
        Ok(FakeStream::new(
            "scripts.run",
            Box::new(move |s: Arc<StreamRec>| {
                Box::pin(async_stream::stream! {
                    // `finally: running_scripts.discard(name)`, also when the consumer drops the stream
                    let _running = RunningGuard { node: Arc::clone(&node), name: name.clone() };
                    for (i, (stream, line)) in run.lines.iter().enumerate() {
                        if s.is_cancelled() {
                            o2.lock().expect("lock").cancelled = true;
                            return;
                        }
                        yield StreamItem::Event(obj(json!({"stream": stream, "line": line})));
                        if i == 0
                            && let Some(g) = &run.gate
                        {
                            tokio::select! {
                                () = g.wait() => {}
                                () = s.cancel_event.wait() => {}
                            }
                        }
                        tokio::task::yield_now().await;
                    }
                    if let Some(e) = &run.error {
                        o2.lock().expect("lock").error = Some(e.clone());
                        return;
                    }
                    if run.rc == 0 && let Some(effect) = &run.effect {
                        effect(&node);
                    }
                    o2.lock().expect("lock").result = Some(obj(json!({"rc": run.rc})));
                })
            }),
            finish_from(self.host.id.clone(), outcome),
        ))
    }
}

struct RunningGuard {
    node: Arc<FakeNode>,
    name: String,
}

impl Drop for RunningGuard {
    fn drop(&mut self) {
        if let Ok(mut st) = self.node.state.lock() {
            st.running_scripts.remove(&self.name);
        }
    }
}

/// externals.py `matches_script`: the container name matches one of the script's `containers` patterns.
pub fn matches_script(name: &str, script: &Value) -> bool {
    script
        .get("containers")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .any(|p| ec_hub::services::common::fnmatchcase(name, p))
}

/// `add_container` keyword arguments (Python's defaults).
#[derive(Debug, Clone)]
pub struct ContainerOpts {
    pub image: String,
    pub engine: Option<String>,
    pub state: String,
    pub managed: bool,
    pub labels: Obj,
    pub host_ports: Vec<Value>,
    pub argv: Vec<String>,
    pub params: Obj,
    pub mounts: Vec<Value>,
    pub gpus: Vec<String>,
    pub all_gpus: bool,
    pub started_at: Option<f64>,
    pub exit_code: Option<i64>,
}

impl Default for ContainerOpts {
    fn default() -> Self {
        Self {
            image: "vllm/vllm-openai:v0.23.0".into(),
            engine: None,
            state: "running".into(),
            managed: false,
            labels: Obj::new(),
            host_ports: Vec::new(),
            argv: Vec::new(),
            params: Obj::new(),
            mounts: Vec::new(),
            gpus: Vec::new(),
            all_gpus: false,
            started_at: Some(100.0),
            exit_code: None,
        }
    }
}

/// `add_endpoint` keyword arguments (Python's defaults).
#[derive(Debug, Clone)]
pub struct EndpointOpts {
    pub container: Option<String>,
    pub engine: Option<String>,
    pub models: Vec<String>,
    pub health: Option<i64>,
    pub models_status: Option<i64>,
    pub metrics: Option<i64>,
    pub auth_required: bool,
}

impl Default for EndpointOpts {
    fn default() -> Self {
        Self {
            container: None,
            engine: None,
            models: Vec::new(),
            health: Some(200),
            models_status: Some(200),
            metrics: None,
            auth_required: false,
        }
    }
}

#[async_trait]
impl NodeClient for FakeNode {
    fn host(&self) -> &HostConfig {
        &self.host
    }

    fn link(&self) -> LinkStatus {
        let st = self.st();
        let up = matches!(st.state.as_str(), "up" | "stale");
        let mut l = LinkStatus::new(&st.state, st.link_error.clone());
        l.connected_since = up.then_some(st.connected_since);
        l.skew_s = up.then_some(st.skew_s);
        l.last_seen = Some(1.0);
        l
    }

    fn latest(&self, kind: &str) -> Option<Arc<Obj>> {
        let mut st = self.st();
        if st.hidden.contains(kind) {
            return None;
        }
        match kind {
            "hello" => Some(Arc::new(st.hello.clone())),
            "sample" => self.sample_of(&mut st),
            "inventory" => Self::inventory_of(&st),
            "disks" => ec_proto::sanitize_push(&obj(
                json!({"type": "disks", "t": 1.0, "mounts": st.disks}),
            ))
            .ok()
            .flatten()
            .map(Arc::new),
            _ => None,
        }
    }

    async fn call(
        &self,
        op: &str,
        args: Option<Obj>,
        _timeout: Duration,
    ) -> Result<Obj, NodeError> {
        let args = args.unwrap_or_default();
        self.st().calls.push((op.into(), args.clone()));
        self.gate(op, &args)?;
        let (failed, over) = {
            let st = self.st();
            (st.fail.get(op).cloned(), st.overrides.get(op).cloned())
        };
        if let Some(e) = failed {
            return Err(e);
        }
        let out = match over {
            Some(OpOverride::Sync(f)) => f(self, &args)?,
            Some(OpOverride::Async(f)) => f(self.me(), args.clone()).await?,
            None => self.builtin_op(op, &args)?,
        };
        Ok(ec_proto::clean::clean_obj(
            &Value::Object(out),
            specs::results(op),
        ))
    }

    fn open(&self, op: &str, args: Option<Obj>) -> Box<dyn NodeStream> {
        let args = args.unwrap_or_default();
        self.st().calls.push((op.into(), args.clone()));
        let made = (|| {
            self.gate(op, &args)?;
            let (failed, over) = {
                let st = self.st();
                (st.fail.get(op).cloned(), st.open_overrides.get(op).cloned())
            };
            if let Some(e) = failed {
                return Err(e);
            }
            match over {
                Some(f) => f(self, &args),
                None => self.builtin_open(op, &args),
            }
        })();
        match made {
            Ok(s) => {
                self.st().streams.push(Arc::clone(&s.rec));
                Box::new(s)
            }
            Err(e) => Box::new(FailedStream::new(e)),
        }
    }
}

// ── the fleet ────────────────────────────────────────────────────────────────────────────────────────────
/// A Fleet over FakeNodes. `start`/`stop` and the `set_*` hooks exist because the app container drives them.
pub struct FakeFleet {
    nodes: Mutex<Vec<Arc<FakeNode>>>,
    pub started: AtomicBool,
    pub resolver: Mutex<Option<InstanceResolver>>,
    pub counts: Mutex<Option<CountsProvider>>,
}

impl FakeFleet {
    pub fn new(nodes: Vec<Arc<FakeNode>>) -> Self {
        Self {
            nodes: Mutex::new(nodes),
            started: AtomicBool::new(false),
            resolver: Mutex::new(None),
            counts: Mutex::new(None),
        }
    }

    /// Add a machine (one with the same id is replaced in place).
    pub fn add(&self, node: Arc<FakeNode>) -> Arc<FakeNode> {
        let mut nodes = self.nodes.lock().expect("lock");
        if let Some(slot) = nodes.iter_mut().find(|n| n.host.id == node.host.id) {
            *slot = Arc::clone(&node);
        } else {
            nodes.push(Arc::clone(&node));
        }
        node
    }

    /// The fake node of a host (panics for an unknown one: a test bug).
    pub fn fake(&self, host_id: &str) -> Arc<FakeNode> {
        self.nodes
            .lock()
            .expect("lock")
            .iter()
            .find(|n| n.host.id == host_id)
            .cloned()
            .unwrap_or_else(|| panic!("no fake node {host_id}"))
    }

    pub fn is_started(&self) -> bool {
        self.started.load(Ordering::SeqCst)
    }

    pub fn resolver(&self) -> Option<InstanceResolver> {
        self.resolver.lock().expect("lock").clone()
    }

    pub fn counts(&self) -> Option<CountsProvider> {
        self.counts.lock().expect("lock").clone()
    }
}

impl FleetPort for FakeFleet {
    fn hosts(&self) -> Vec<HostConfig> {
        self.nodes
            .lock()
            .expect("lock")
            .iter()
            .map(|n| n.host.clone())
            .collect()
    }

    fn node(&self, host_id: &str) -> Result<Arc<dyn NodeClient>, Problem> {
        self.nodes
            .lock()
            .expect("lock")
            .iter()
            .find(|n| n.host.id == host_id)
            .map(|n| Arc::clone(n) as Arc<dyn NodeClient>)
            .ok_or_else(|| {
                let shown: String = host_id.chars().take(40).collect();
                not_found(format!(
                    "no host {} is configured",
                    ec_proto::sanitize::py_repr(&shown)
                ))
                .code("unknown_host")
            })
    }
}

#[async_trait]
impl Fleet for FakeFleet {
    async fn start(&self) {
        self.started.store(true, Ordering::SeqCst);
    }
    async fn stop(&self) {
        self.started.store(false, Ordering::SeqCst);
    }
    fn summaries(&self) -> AppResult<Vec<HostSummary>> {
        Err(not_ported(
            "tests: FakeFleet (Python's FakeFleet has no summaries either)",
        ))
    }
    fn summary(&self, _host_id: &str) -> AppResult<HostSummary> {
        Err(not_ported("tests: FakeFleet"))
    }
    async fn detail(&self, _host_id: &str) -> AppResult<HostDetail> {
        Err(not_ported("tests: FakeFleet"))
    }
    fn reconnect(&self, _host_id: &str) -> AppResult<HostSummary> {
        Err(not_ported("tests: FakeFleet"))
    }
    fn set_instance_resolver(&self, f: InstanceResolver) {
        *self.resolver.lock().expect("lock") = Some(f);
    }
    fn set_counts_provider(&self, f: CountsProvider) {
        *self.counts.lock().expect("lock") = Some(f);
    }
}

// ── the library ──────────────────────────────────────────────────────────────────────────────────────────
pub type InfoBuilder = Arc<dyn Fn(&HubModel) -> ModelInfo + Send + Sync>;

/// The model library as the runtime services see it: copies registered by the test, roots read from each node's
/// policy.
pub struct FakeLibrary {
    fleet: Arc<FakeFleet>,
    copies: Mutex<Vec<((String, Vec<String>), LibraryCopy)>>,
    infos: Mutex<Vec<((String, Vec<String>), ModelInfo)>>,
    info_builder: Mutex<Option<InfoBuilder>>,
    pub started: AtomicBool,
}

/// `FakeLibrary.add` keyword arguments.
#[derive(Debug, Clone)]
pub struct CopyOpts {
    pub root_id: String,
    pub rel: String,
    pub size_bytes: i64,
    pub info: Option<ModelInfo>,
    pub drive: Option<MountInfo>,
    pub root_label: Option<String>,
}

impl CopyOpts {
    pub fn new(root_id: &str, rel: &str) -> Self {
        Self {
            root_id: root_id.into(),
            rel: rel.into(),
            size_bytes: 65_530_000_000,
            info: None,
            drive: None,
            root_label: None,
        }
    }
}

impl FakeLibrary {
    pub fn new(fleet: Arc<FakeFleet>) -> Self {
        Self {
            fleet,
            copies: Mutex::new(Vec::new()),
            infos: Mutex::new(Vec::new()),
            // Python's FakeLibrary builds the info with services.hf.build_model_info
            info_builder: Mutex::new(Some(Arc::new(|h: &HubModel| {
                ec_hub::services::hf::build_model_info(h, None).expect("build_model_info")
            }) as InfoBuilder)),
            started: AtomicBool::new(false),
        }
    }

    /// How `add_hf` turns the fake repo into a ModelInfo (Python: `services.hf.build_model_info`).
    pub fn set_info_builder(&self, f: InfoBuilder) {
        *self.info_builder.lock().expect("lock") = Some(f);
    }

    fn key(r: &ModelRef) -> Vec<String> {
        match r.format.as_str() {
            "hf" => vec!["hf".into(), r.repo_id.clone().unwrap_or_default()],
            "ollama" => vec!["ollama".into(), r.name.clone().unwrap_or_default()],
            f => vec![
                f.into(),
                r.root_id.clone().unwrap_or_default(),
                r.rel.clone().unwrap_or_default(),
            ],
        }
    }

    pub fn add(&self, host_id: &str, r: &ModelRef, o: CopyOpts) -> LibraryCopy {
        let node = self.fleet.fake(host_id);
        let base = node
            .root_path(&o.root_id)
            .expect("a root of the node's policy");
        let id = sha1_hex(&format!("{host_id}|{}|{}", o.root_id, o.rel))[..16].to_string();
        let c = LibraryCopy {
            id,
            host_id: host_id.into(),
            host_label: node.host.label.clone(),
            root_id: o.root_id.clone(),
            root_label: o.root_label.clone(),
            rel: o.rel.clone(),
            path: if o.rel.is_empty() {
                base
            } else {
                format!("{base}/{}", o.rel)
            },
            size_bytes: o.size_bytes,
            files: None,
            mtime: None,
            atime: None,
            drive: o.drive.clone(),
            visible_on: vec![host_id.into()],
            writable: false,
            deletable: false,
            in_use_by: Vec::new(),
        };
        let k = (host_id.to_string(), Self::key(r));
        upsert(&mut self.copies.lock().expect("lock"), k.clone(), c.clone());
        if let Some(info) = o.info {
            upsert(&mut self.infos.lock().expect("lock"), k, info);
        }
        c
    }

    /// A Hugging Face model on a host (rel default: `hub/models--<org>--<name>` in root `hf`).
    pub fn add_hf(
        &self,
        host_id: &str,
        repo_id: &str,
        cfg: &Value,
        size: i64,
        root_id: &str,
        rel: Option<&str>,
        drive: Option<MountInfo>,
    ) -> LibraryCopy {
        let hub = HubModel {
            repo_id: repo_id.into(),
            sha: Some("c0ffee".into()),
            gated: false,
            license: None,
            pipeline_tag: None,
            library_name: None,
            tags: Vec::new(),
            files: vec![RepoFile {
                path: "model.safetensors".into(),
                size,
                sha256: None,
            }],
            safetensors_params: Vec::new(),
            safetensors_total: None,
            config: cfg.as_object().cloned(),
            downloads: None,
            likes: None,
        };
        let info = self
            .info_builder
            .lock()
            .expect("lock")
            .clone()
            .map(|f| f(&hub));
        let rel = rel.map_or_else(
            || format!("hub/models--{}", repo_id.replace('/', "--")),
            str::to_string,
        );
        self.add(
            host_id,
            &ModelRef::hf(repo_id),
            CopyOpts {
                size_bytes: size,
                info,
                drive,
                ..CopyOpts::new(root_id, &rel)
            },
        )
    }

    pub fn remove(&self, host_id: &str, r: &ModelRef) {
        let k = (host_id.to_string(), Self::key(r));
        self.copies.lock().expect("lock").retain(|(kk, _)| *kk != k);
    }

    fn root(&self, host_id: &str, kinds: &[&str]) -> Option<RootInfo> {
        let node = self.fleet.fake(host_id);
        let pol = node.policy();
        for r in pol
            .get("model_roots")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let kind = s_of(r.get("kind"));
            if kinds.contains(&kind.as_str()) {
                return Some(RootInfo {
                    id: s_of(r.get("id")),
                    path: s_of(r.get("path")),
                    kind,
                    writable: truthy(r.get("writable")),
                    label: r.get("label").and_then(Value::as_str).map(str::to_string),
                    mount: None,
                    shared_from: None,
                    visible_on: Vec::new(),
                });
            }
        }
        None
    }
}

fn upsert<K: PartialEq, V>(v: &mut Vec<(K, V)>, k: K, val: V) {
    if let Some(slot) = v.iter_mut().find(|(kk, _)| *kk == k) {
        slot.1 = val;
    } else {
        v.push((k, val));
    }
}

#[async_trait]
impl LibraryPort for FakeLibrary {
    fn find(&self, host_id: &str, r: &ModelRef) -> Option<LibraryCopy> {
        let k = (host_id.to_string(), Self::key(r));
        self.copies
            .lock()
            .expect("lock")
            .iter()
            .find(|(kk, _)| *kk == k)
            .map(|(_, c)| c.clone())
    }
    fn locate(&self, host_id: &str, r: &ModelRef) -> Option<LibraryCopy> {
        self.find(host_id, r)
    }
    async fn model_info(&self, host_id: &str, r: &ModelRef) -> AppResult<Option<ModelInfo>> {
        let k = (host_id.to_string(), Self::key(r));
        Ok(self
            .infos
            .lock()
            .expect("lock")
            .iter()
            .find(|(kk, _)| *kk == k)
            .map(|(_, i)| i.clone()))
    }
    fn hf_root(&self, host_id: &str) -> Option<RootInfo> {
        self.root(host_id, &["hf-home", "hf-hub"])
    }
    fn ollama_root(&self, host_id: &str) -> Option<RootInfo> {
        self.root(host_id, &["ollama"])
    }
    fn copies(&self, host_id: &str) -> Vec<LibraryCopy> {
        self.copies
            .lock()
            .expect("lock")
            .iter()
            .filter(|((h, _), _)| h == host_id)
            .map(|(_, c)| c.clone())
            .collect()
    }
}

/// Poll `condition` until it holds. Tests wait on state, never on a guessed number of milliseconds.
pub async fn until(condition: impl Fn() -> bool, timeout_s: f64, what: &str) {
    let deadline = Instant::now() + Duration::from_secs_f64(timeout_s);
    while !condition() {
        assert!(Instant::now() <= deadline, "timed out waiting for {what}");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

// ── the environment (Python's `Env`) ─────────────────────────────────────────────────────────────────────
/// `QWEN_CFG` of tests/core/conftest.py
pub fn qwen_cfg() -> Value {
    json!({"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"], "hidden_size": 5120,
           "num_hidden_layers": 64, "num_attention_heads": 64, "num_key_value_heads": 8, "head_dim": 128,
           "max_position_embeddings": 40960, "torch_dtype": "bfloat16"})
}

/// Everything a test needs: the app container, the fake fleet behind it (one machine, `local`, label "testbox"),
/// the fake model library and hub, and programmable fake engines. Settings as tests/core/conftest.py `Env`.
pub struct FleetEnv {
    pub tmp: tempfile::TempDir,
    pub cfg: Settings,
    pub adapters: Vec<Arc<dyn EngineAdapter>>,
    pub learn: Option<ec_hub::domain::learn::LearnBundle>,
    pub hub: Arc<FakeHub>,
    pub node: Arc<FakeNode>,
    pub fleet: Arc<FakeFleet>,
    pub library: Arc<FakeLibrary>,
    state: Option<Arc<AppState>>,
}

impl Default for FleetEnv {
    fn default() -> Self {
        Self::new()
    }
}

impl FleetEnv {
    pub fn new() -> Self {
        Self::with(|_| {})
    }

    /// `Env(tmp)` with `edit` applied to the settings before the container is built (on first use).
    pub fn with(edit: impl FnOnce(&mut Settings)) -> Self {
        let tmp = tempfile::tempdir().expect("tempdir");
        let t = tmp.path();
        let mut cfg = Settings::defaults();
        cfg.data_dir = t.join("data");
        cfg.hf_cache_dir = t.join("hf");
        cfg.otlp_enabled = false;
        cfg.background_tasks = false;
        cfg.frontend_dir = Some(t.join("fe"));
        cfg.hf_token = None;
        cfg.hf_token_file = t.join("no-token");
        cfg.allowed_hosts = vec!["testserver".into()];
        cfg.trust_loopback = true;
        cfg.scrape_interval_s = 0.01;
        cfg.restart_settle_s = 0.0;
        cfg.engine_image_allowlist = vec!["fake/engine:2".into(), "fake/engine:2.0".into()];
        cfg.hosts_file = t.join("hosts.json");
        // Hermetic by construction (PORTING.md §A2): every outbound default points at a CLOSED loopback port or a fake,
        // so a test that forgets to inject a fake fails instead of reaching Hugging Face, an OTLP collector, a real
        // docker or the owner's ssh key. A test that needs Python's default value sets it explicitly in `edit`.
        cfg.hf_endpoint = crate::kit::CLOSED_URL.into();
        cfg.otlp_endpoint = crate::kit::CLOSED_HOSTPORT.into();
        cfg.egress_proxy = None;
        cfg.docker_bin = format!("{}/docker", crate::kit::FAKES);
        cfg.hub_key_file = t.join("ssh").join("hub-key");
        cfg.node_policy = None;
        cfg.node_command = None;
        edit(&mut cfg);
        std::fs::create_dir_all(t.join("fe").join("js")).expect("fe");
        std::fs::write(t.join("fe").join("index.html"), "<html>console</html>").expect("index");
        std::fs::write(t.join("fe").join("js").join("app.js"), "console.log(1)").expect("app.js");
        let node = FakeNode::new(
            FakeNode::host_config("local", "testbox", None),
            HelloOpts::default(),
        );
        let fleet = Arc::new(FakeFleet::new(vec![Arc::clone(&node)]));
        let library = Arc::new(FakeLibrary::new(Arc::clone(&fleet)));
        Self {
            tmp,
            cfg,
            adapters: vec![Arc::new(FakeAdapter) as Arc<dyn EngineAdapter>],
            learn: None,
            hub: Arc::new(FakeHub::default()),
            node,
            fleet,
            library,
            state: None,
        }
    }

    /// The app (built on first use). The fake library is the lifecycle's (and the fit estimator's) library.
    pub fn app(&mut self) -> Arc<AppState> {
        if let Some(s) = &self.state {
            return Arc::clone(s);
        }
        let o = Overrides {
            adapters: Some(AdapterRegistry::new(self.adapters.clone())),
            learn: self.learn.clone().map(Arc::new),
            fleet: Some(Arc::clone(&self.fleet) as Arc<dyn Fleet>),
            hub: Some(Arc::clone(&self.hub) as Arc<dyn HubClient>),
            library: Some(Arc::clone(&self.library) as Arc<dyn LibraryPort>),
            ..Overrides::default()
        };
        let c = build_container(Arc::new(self.cfg.clone()), o).expect("container");
        let st = app_state(c);
        self.state = Some(Arc::clone(&st));
        st
    }

    pub fn container(&mut self) -> Arc<Container> {
        Arc::clone(&self.app().c)
    }

    /// A second machine in the fleet (a remote one: ssh is irrelevant to the fake).
    pub fn add_host(&self, host_id: &str, label: &str, o: HelloOpts) -> Arc<FakeNode> {
        let node = FakeNode::new(
            FakeNode::host_config(host_id, label, Some(&format!("u@{host_id}.example"))),
            o,
        );
        self.fleet.add(node)
    }

    /// Put a Hugging Face model in a host's library (nothing is written to disk: the library is a fake).
    pub fn seed_local_model(&self, repo_id: &str, cfg: Option<Value>, size: i64, host_id: &str) {
        self.library.add_hf(
            host_id,
            repo_id,
            &cfg.unwrap_or_else(qwen_cfg),
            size,
            "hf",
            None,
            None,
        );
    }

    /// `env.engine_routes[path] = handler`: every port of the local node answers `path`.
    pub fn engine_route(&self, path: &str, h: HttpHandler) {
        self.node.serve(None, path, h);
    }

    /// `env.port_routes[(port, path)] = handler`
    pub fn port_route(&self, port: i64, path: &str, h: HttpHandler) {
        self.node.serve(Some(port), path, h);
    }

    pub fn engine_requests(&self) -> Vec<HttpReq> {
        self.node.http_requests()
    }

    pub fn engine_up(&self) {
        self.engine_route("/health", http_handler(|_| HttpResp::text(200, "ok")));
    }

    /// A repo on the (fake) Hugging Face hub, for the routes that describe models that are not downloaded yet.
    pub fn seed_hub_model(&self, repo_id: &str, cfg: Option<Value>) {
        let cfg = cfg.unwrap_or_else(qwen_cfg);
        let text = serde_json::to_vec(&cfg).expect("json");
        self.hub.add(
            repo_id,
            &[("config.json", &text), ("model.safetensors", &[b'x'; 100])],
            Some(cfg),
            &[("BF16", 32_762_000_000)],
        );
    }

    /// A request with the Python `client` fixture's defaults: Host testserver, client 127.0.0.1, X-Engine-Console: 1.
    pub fn req(&mut self, method: &str, path: &str) -> FReq {
        FReq {
            app: self.app(),
            method: method.into(),
            path: path.into(),
            headers: vec![
                ("host".into(), "testserver".into()),
                ("x-engine-console".into(), "1".into()),
            ],
            body: Vec::new(),
            client: "127.0.0.1".into(),
        }
    }
    pub fn get(&mut self, path: &str) -> FReq {
        self.req("GET", path)
    }
    pub fn post(&mut self, path: &str) -> FReq {
        self.req("POST", path)
    }
    pub fn put(&mut self, path: &str) -> FReq {
        self.req("PUT", path)
    }
    pub fn patch(&mut self, path: &str) -> FReq {
        self.req("PATCH", path)
    }
    pub fn delete(&mut self, path: &str) -> FReq {
        self.req("DELETE", path)
    }

    pub fn path(&self) -> PathBuf {
        self.tmp.path().to_path_buf()
    }
}

/// One request to the app (the kit's `Req`, for a FleetEnv).
pub struct FReq {
    app: Arc<AppState>,
    method: String,
    path: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
    client: String,
}

impl FReq {
    pub fn header(mut self, k: &str, v: &str) -> Self {
        let k = k.to_lowercase();
        self.headers.retain(|(kk, _)| *kk != k);
        self.headers.push((k, v.into()));
        self
    }
    pub fn without(mut self, k: &str) -> Self {
        let k = k.to_lowercase();
        self.headers.retain(|(kk, _)| *kk != k);
        self
    }
    pub fn bearer(self, key: &str) -> Self {
        self.header("authorization", &format!("Bearer {key}"))
    }
    pub fn json(mut self, v: &Value) -> Self {
        self.body = serde_json::to_vec(v).expect("json");
        self.header("content-type", "application/json")
    }
    pub fn raw(mut self, b: impl Into<Vec<u8>>) -> Self {
        self.body = b.into();
        self
    }
    pub fn client(mut self, host: &str) -> Self {
        self.client = host.into();
        self
    }

    fn build(self) -> (Arc<AppState>, String, http::Request<AxBody>) {
        let mut b = http::Request::builder()
            .method(self.method.as_str())
            .uri(self.path.as_str());
        let has_len = self.headers.iter().any(|(k, _)| k == "content-length");
        for (k, v) in &self.headers {
            b = b.header(k.as_str(), v.as_str());
        }
        if !self.body.is_empty() && !has_len {
            b = b.header("content-length", self.body.len().to_string());
        }
        let req = b.body(AxBody::from(self.body)).expect("request");
        (self.app, self.client, req)
    }

    pub async fn send(self) -> Resp {
        let (app, client, req) = self.build();
        let r = handle_request(&app, &client, req).await;
        let status = r.status().as_u16();
        let headers: Vec<(String, String)> = r
            .headers()
            .iter()
            .map(|(k, v)| {
                (
                    k.as_str().to_string(),
                    String::from_utf8_lossy(v.as_bytes()).into_owned(),
                )
            })
            .collect();
        let body = r
            .into_body()
            .collect()
            .await
            .map(|c| c.to_bytes())
            .unwrap_or_default();
        Resp {
            status,
            headers,
            body,
        }
    }

    /// Send and return as soon as the response head is there; the body is left to read (streams).
    pub async fn open(self) -> FOpened {
        let (app, client, req) = self.build();
        let r = handle_request(&app, &client, req).await;
        FOpened {
            status: r.status().as_u16(),
            body: Some(r.into_body()),
            received: Vec::new(),
        }
    }
}

/// A response whose body is read on demand.
pub struct FOpened {
    pub status: u16,
    body: Option<AxBody>,
    pub received: Vec<u8>,
}

impl FOpened {
    pub async fn chunk(&mut self, wait: Duration) -> Option<Bytes> {
        let body = self.body.as_mut()?;
        match tokio::time::timeout(wait, body.frame()).await {
            Ok(Some(Ok(f))) => {
                let d = f.into_data().ok()?;
                self.received.extend_from_slice(&d);
                Some(d)
            }
            _ => None,
        }
    }
    pub async fn rest(&mut self, wait: Duration) -> Vec<u8> {
        while self.chunk(wait).await.is_some() {}
        self.received.clone()
    }
    pub fn leave(&mut self) {
        self.body = None;
    }
    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.received).into_owned()
    }
}

/// `Pin<Box<..>>` alias kept for handlers written against the Python fake's async generators.
pub type ItemStream = Pin<Box<dyn futures::Stream<Item = StreamItem> + Send>>;
