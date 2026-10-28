//! Fleet: the hub's view of every machine ("host") and the ports services use to reach one (port of
//! domain/fleet.py). Additive changes only: the API serves these shapes.
//!
//! Every host, including the console's own (`local`), is reached the same way: a `NodeClient` speaking the ecnode
//! wire protocol (crate ec-proto) to an `ecnode --stdio` process. Services never run docker, read /proc or open a
//! model directory themselves; they ask the host's node and treat everything it says as untrusted data (already
//! through `ec_proto::sanitize_*`).
//!
//! Pushed node messages are re-published on the EventBus as topic `node.<type>` (sample, disks, inventory, metrics,
//! models_changed, hello) with payload `{"host_id": str, "msg": <sanitized message>}`; link changes go to topic
//! `fleet` as a HostSummary dump.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::domain::errors::{AppResult, NodeError, Obj, Problem};
use crate::domain::models::{HostAlert, HostMemInfo};

pub use crate::domain::models::HOST_ID_PATTERN;
pub use crate::domain::refs::{InstanceControls, InstanceVram, ModelLocation, ModelRef};

pub const LOCAL: &str = "local";
/// LinkState values
pub const LINK_STATES: &[&str] = &[
    "connecting",
    "up",
    "stale",
    "down",
    "incompatible",
    "disabled",
];
pub const LIBRARY_FORMATS: &[&str] = &["hf", "gguf", "ollama", "plain", "other"];
/// NodeKind values for `NodeClient::latest`
pub const NODE_KINDS: &[&str] = &["hello", "sample", "disks", "inventory"];

// ── reaching a node ──────────────────────────────────────────────────────────────────────────────────────
/// One machine from ~/.config/engine-console/hosts.json (validated by services/hosts.rs).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostConfig {
    pub id: String,
    pub label: String,
    #[serde(default)]
    pub ssh: Option<String>,
    #[serde(default)]
    pub port: Option<u16>,
    #[serde(default)]
    pub ssh_config: bool,
    #[serde(default = "yes")]
    pub enabled: bool,
}

fn yes() -> bool {
    true
}

/// One item of a streaming request: an event (sanitized with `ec_proto::specs::events(op)`) or DATA.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamItem {
    Event(Obj),
    Data(bytes::Bytes),
}

/// One streaming request: http.request, docker.logs_follow, docker.pull, fs.read, fs.write, scripts.run.
/// `next()` yields the node's events and DATA in order and ends (None) at the node's res or when the link drops;
/// then `result()` returns the result or the NodeError the request ended with. Dropping the stream before its res
/// cancels the request (the node still answers once).
#[async_trait]
pub trait NodeStream: Send {
    async fn next(&mut self) -> Option<StreamItem>;
    /// upload DATA; waits for credit (fs.write, an http.request body)
    async fn write(&mut self, data: &[u8]) -> Result<(), NodeError>;
    /// no more upload DATA
    async fn end(&mut self) -> Result<(), NodeError>;
    /// the sanitized res result; NodeError if ok=false
    async fn result(&mut self) -> Result<Obj, NodeError>;
    async fn cancel(&mut self);
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinkStatus {
    /// one of LINK_STATES
    pub state: String,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub retry_in_s: Option<i64>,
    #[serde(default)]
    pub notes: Vec<String>,
    #[serde(default)]
    pub last_seen: Option<f64>,
    #[serde(default)]
    pub connected_since: Option<f64>,
    #[serde(default)]
    pub rtt_ms: Option<i64>,
    #[serde(default)]
    pub skew_s: Option<f64>,
}

impl LinkStatus {
    pub fn new(state: &str, error: Option<String>) -> Self {
        Self {
            state: state.to_string(),
            error,
            retry_in_s: None,
            notes: Vec::new(),
            last_seen: None,
            connected_since: None,
            rtt_ms: None,
            skew_s: None,
        }
    }
}

/// One machine's node link (services/fleet.rs `NodeLink` / `DisabledLink`, or a test fake).
#[async_trait]
pub trait NodeClient: Send + Sync {
    fn host(&self) -> &HostConfig;
    fn link(&self) -> LinkStatus;
    /// last sanitized pushed message of that kind ("hello" | "sample" | "disks" | "inventory")
    fn latest(&self, kind: &str) -> Option<Arc<Obj>>;
    /// One request/response. NodeError "node_unavailable" when the link is not up; NodeError(code) when the node
    /// answers ok=false; code "timeout" when no res arrives in time (it may or may not have taken effect).
    async fn call(&self, op: &str, args: Option<Obj>, timeout: Duration) -> Result<Obj, NodeError>;
    /// A streaming request. A request that cannot start (link down, busy, bad args) yields a stream whose
    /// `result()` (and `write`/`end`) return the reason.
    fn open(&self, op: &str, args: Option<Obj>) -> Box<dyn NodeStream>;
}

/// `FleetPort`: every configured machine and its node.
pub trait FleetPort: Send + Sync {
    fn hosts(&self) -> Vec<HostConfig>;
    /// `not_found` with code "unknown_host" if the host is not configured.
    fn node(&self, host_id: &str) -> Result<Arc<dyn NodeClient>, Problem>;
}

/// What the routes, the CLI and the composition root need from the fleet on top of `FleetPort`
/// (services/fleet.rs `FleetService` implements it; tests may supply fakes).
#[async_trait]
pub trait Fleet: FleetPort {
    async fn start(&self);
    async fn stop(&self);
    fn summaries(&self) -> AppResult<Vec<HostSummary>>;
    /// 404 unknown_host for an unconfigured host
    fn summary(&self, host_id: &str) -> AppResult<HostSummary>;
    async fn detail(&self, host_id: &str) -> AppResult<HostDetail>;
    /// 409 host_disabled for a disabled host
    fn reconnect(&self, host_id: &str) -> AppResult<HostSummary>;
    /// fn(host_id, container_name) -> console instance id, to name the instance behind each GPU process
    fn set_instance_resolver(&self, f: InstanceResolver);
    /// fn(host_id) -> HostCounts shown in the host summary
    fn set_counts_provider(&self, f: CountsProvider);
}

pub type InstanceResolver = Arc<dyn Fn(&str, &str) -> Option<String> + Send + Sync>;
pub type CountsProvider = Arc<dyn Fn(&str) -> HostCounts + Send + Sync>;

// ── API shapes (GET /hosts, /hosts/{id}) ─────────────────────────────────────────────────────────────────────
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GpuProc {
    pub pid: i64,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub container: Option<String>,
    #[serde(default)]
    pub instance_id: Option<String>,
    #[serde(default)]
    pub used_gib: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GpuLive {
    pub index: i64,
    pub uuid: String,
    pub name: String,
    pub total_gib: f64,
    #[serde(default)]
    pub used_gib: Option<f64>,
    #[serde(default)]
    pub free_gib: Option<f64>,
    #[serde(default)]
    pub util_pct: Option<f64>,
    #[serde(default)]
    pub temp_c: Option<f64>,
    #[serde(default)]
    pub power_w: Option<f64>,
    #[serde(default)]
    pub power_limit_w: Option<f64>,
    #[serde(default)]
    pub fan_pct: Option<f64>,
    #[serde(default)]
    pub compute_capability: Option<String>,
    #[serde(default)]
    pub unified_memory: bool,
    #[serde(default)]
    pub procs: Vec<GpuProc>,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct NodeFacts {
    pub version: Option<String>,
    pub hostname: Option<String>,
    pub arch: Option<String>,
    pub os: Option<String>,
    pub kernel: Option<String>,
    pub cpu_model: Option<String>,
    pub cpu_cores: Option<i64>,
    pub cpu_threads: Option<i64>,
    pub python: Option<String>,
    pub driver: Option<String>,
    pub cuda: Option<String>,
    pub gpu_source: Option<String>,
    pub forced_cmd: Option<bool>,
    pub via_ssh: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct CpuFacts {
    pub load1: Option<f64>,
    pub load5: Option<f64>,
    pub load15: Option<f64>,
    pub util_pct: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct DockerFacts {
    #[serde(default)]
    pub available: bool,
    pub context: Option<String>,
    pub rootless: Option<bool>,
    pub version: Option<String>,
    pub nvidia_runtime: Option<bool>,
    pub error: Option<String>,
}

/// The node's own local policy, as it reported it. The hub can read it, never change it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolicyFacts {
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default = "monitor")]
    pub mode: String,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub engine_images: Vec<String>,
    #[serde(default)]
    pub gateway_image: Option<String>,
    #[serde(default)]
    pub engine_network: Option<String>,
    #[serde(default)]
    pub port_range: Vec<i64>,
    #[serde(default)]
    pub allow_pull: bool,
    #[serde(default)]
    pub allow_delete: bool,
    #[serde(default = "yes")]
    pub allow_ipc_host: bool,
    #[serde(default = "none_str")]
    pub external_control: String,
    #[serde(default)]
    pub max_shm_gib: Option<f64>,
}

fn monitor() -> String {
    "monitor".to_string()
}
fn none_str() -> String {
    "none".to_string()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MountInfo {
    pub mount: String,
    pub label: String,
    #[serde(default)]
    pub fstype: Option<String>,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub device: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub size_bytes: Option<i64>,
    #[serde(default)]
    pub used_bytes: Option<i64>,
    #[serde(default)]
    pub free_bytes: Option<i64>,
    #[serde(default)]
    pub readonly: bool,
    #[serde(default)]
    pub remote: bool,
    #[serde(default)]
    pub remote_host: Option<String>,
    #[serde(default)]
    pub remote_host_id: Option<String>,
    #[serde(default)]
    pub remote_path: Option<String>,
    #[serde(default)]
    pub roots: Vec<String>,
}

/// A model root from the node policy, plus where it physically lives.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RootInfo {
    pub id: String,
    pub path: String,
    pub kind: String,
    #[serde(default)]
    pub writable: bool,
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default)]
    pub mount: Option<MountInfo>,
    #[serde(default)]
    pub shared_from: Option<String>,
    #[serde(default)]
    pub visible_on: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageInfo {
    #[serde(rename = "ref")]
    pub ref_: String,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub size_bytes: Option<i64>,
    #[serde(default)]
    pub created: Option<f64>,
    #[serde(default)]
    pub arch: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScriptAction {
    pub name: String,
    pub label: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub confirm: bool,
    #[serde(default)]
    pub danger: bool,
}

/// A lifecycle script registered in a node's policy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScriptInfo {
    pub host_id: String,
    pub name: String,
    pub label: String,
    #[serde(default)]
    pub engine: Option<String>,
    #[serde(default)]
    pub description: String,
    /// "up" | "down" | "unknown" | "error"
    #[serde(default = "unknown")]
    pub status: String,
    #[serde(default)]
    pub detail: Option<String>,
    #[serde(default)]
    pub checked_at: Option<f64>,
    #[serde(default)]
    pub actions: Vec<ScriptAction>,
    #[serde(default)]
    pub containers: Vec<String>,
}

fn unknown() -> String {
    "unknown".to_string()
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct HostCounts {
    #[serde(default)]
    pub engines_running: i64,
    #[serde(default)]
    pub engines_total: i64,
    #[serde(default)]
    pub models: i64,
    #[serde(default)]
    pub models_bytes: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HostSummary {
    pub id: String,
    pub label: String,
    pub remote: bool,
    pub link: LinkStatus,
    #[serde(default)]
    pub node: Option<NodeFacts>,
    #[serde(default)]
    pub gpus: Vec<GpuLive>,
    #[serde(default)]
    pub memory: Option<HostMemInfo>,
    #[serde(default)]
    pub cpu: Option<CpuFacts>,
    #[serde(default)]
    pub docker: Option<DockerFacts>,
    #[serde(default)]
    pub policy: Option<PolicyFacts>,
    #[serde(default)]
    pub counts: HostCounts,
    #[serde(default)]
    pub alerts: Vec<HostAlert>,
}

/// `HostDetail(HostSummary)`: the summary's fields first, then the detail's.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HostDetail {
    #[serde(flatten)]
    pub summary: HostSummary,
    #[serde(default)]
    pub disks: Vec<MountInfo>,
    #[serde(default)]
    pub roots: Vec<RootInfo>,
    #[serde(default)]
    pub images: Vec<ImageInfo>,
    #[serde(default)]
    pub scripts: Vec<ScriptInfo>,
    #[serde(default)]
    pub addresses: Vec<String>,
}

// ── the library ─────────────────────────────────────────────────────────────────────────────────────────
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LibraryCopy {
    /// stable: first 16 hex of sha1("host_id|root_id|rel")
    pub id: String,
    pub host_id: String,
    pub host_label: String,
    pub root_id: String,
    #[serde(default)]
    pub root_label: Option<String>,
    pub rel: String,
    pub path: String,
    pub size_bytes: i64,
    #[serde(default)]
    pub files: Option<i64>,
    #[serde(default)]
    pub mtime: Option<f64>,
    #[serde(default)]
    pub atime: Option<f64>,
    #[serde(default)]
    pub drive: Option<MountInfo>,
    #[serde(default)]
    pub visible_on: Vec<String>,
    #[serde(default)]
    pub writable: bool,
    #[serde(default)]
    pub deletable: bool,
    #[serde(default)]
    pub in_use_by: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LibraryModel {
    /// "<format>:<name>[@<revision>]"
    pub key: String,
    pub name: String,
    pub format: String,
    #[serde(default)]
    pub revision: Option<String>,
    #[serde(default)]
    pub quant: Option<String>,
    #[serde(default)]
    pub params: Option<i64>,
    #[serde(default)]
    pub arch: Option<String>,
    #[serde(default)]
    pub context: Option<i64>,
    #[serde(default)]
    pub shards: Option<i64>,
    pub size_bytes: i64,
    #[serde(default)]
    pub engines: Vec<String>,
    #[serde(default)]
    pub copies: Vec<LibraryCopy>,
    #[serde(default)]
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CopyRequest {
    pub host_id: String,
    #[serde(default)]
    pub root_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ScriptRunRequest {
    #[serde(default)]
    pub arg: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScriptJob {
    pub id: String,
    pub host_id: String,
    pub name: String,
    pub action: String,
    #[serde(default)]
    pub arg: Option<String>,
    /// "running" | "completed" | "failed"
    pub state: String,
    #[serde(default)]
    pub rc: Option<i64>,
    pub started_at: f64,
    #[serde(default)]
    pub finished_at: Option<f64>,
    /// last ~20 kB, secrets masked
    #[serde(default)]
    pub output_tail: String,
}

/// Convenience used by every caller of `NodeClient::call`: the protocol's default request timeout.
pub const DEFAULT_CALL_TIMEOUT: Duration = Duration::from_secs(30);
