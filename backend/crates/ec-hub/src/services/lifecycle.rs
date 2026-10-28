//! Python: backend/src/engine_console/services/lifecycle.py
//! (tests/core/test_lifecycle_fleet.py, test_model_mounts.py, test_library_lifecycle_wiring.py).
//!
//! Instance lifecycle across the fleet: preflight -> node `docker.run` + `gateway.start` -> a supervised state machine
//! stopped -> starting -> loading -> ready -> stopping | failed.
//!
//! Contract v2 (docs/ARCHITECTURE.md §10): every docker action is an operation on the node of the host the instance
//! runs on; the hub never runs docker itself. Three classes of instance are served from here:
//!   * managed  - containers the console created (rows in the `instances` table, full control);
//!   * script   - an engine started by a script in the node's policy (derived from the inventory; start/stop run the
//!     script's `up`/`down`);
//!   * external - any other engine container (derived; monitor-only unless the node policy allows plain stop/start).
//!
//! Security controls kept here (each has a named test): engines join the node's internal network and publish nothing
//! (only the per-instance loopback gateway does); every container action names its instance (and `role: gateway`
//! for a gateway) so the node refuses foreign containers; secret parameter values never reach the DB, the API, argv,
//! plain env or any node request but `docker.run`'s `secret_env`; adopted container labels are validated before
//! they reach SQL rows, paths or the UI; derived instances are never controlled past the node's policy.
//!
//! `tick()` is the supervisor step. It reads container state from the host's pushed inventory and only asks the node
//! (`docker.inspect`) when a container is missing from an inventory that may not have caught up yet.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex, RwLock};

use futures::stream::BoxStream;
use regex::Regex;
use rusqlite::types::Value as SqlValue;
use serde_json::{Value, json};

use crate::adapters::AdapterRegistry;
use crate::adapters::base::{EngineAdapter, LaunchContext, LaunchSpec, ModelInfo};
use crate::adapters::pycompat::PyExc;
use crate::config::Settings;
use crate::domain::errors::{
    AppError, AppResult, Obj, Problem, bad_request, conflict, internal, not_found, unprocessable,
};
use crate::domain::fleet::{FleetPort, HostCounts, LibraryCopy, NodeClient, StreamItem};
use crate::domain::models::{
    CommandSnippets, FitReport, HostMemory, Instance, InstanceCreate, InstancePatch, Page,
    PreflightCheck, PreflightReport, ResidentUse,
};
use crate::domain::refs::{InstanceControls, InstanceVram, ModelLocation, ModelRef};
use crate::domain::repo_id::check_repo_id;
use crate::services::common::{EventBus, new_id, scrub};
use crate::services::discovery::{EXT, SCRIPT};
use crate::services::docker::{command_snippets, gateway_name};
use crate::services::externals::{
    HostCtx, Others, build_host_instances, external_controls, external_instance_id, matches_script,
    memory_for, script_instance_id, uuid_map, vram_for,
};
use crate::services::fitting::FitService;
use crate::services::hardware::HardwareService;
use crate::services::libport::{LibraryPort, NullLibrary};
use crate::services::nodeerr::NodeCaller;
use crate::services::nodehttp::{HttpOpts, NodeHttp};
use crate::services::nodeview as nv;
use crate::services::runspec::{
    LabelArgs, RunArgs, build_run_spec, check_ref, display_of, engine_labels, hf_mounts,
    model_mounts, rel_ok, split_env, validate_run_spec,
};
use crate::services::scripts::ScriptService;
use crate::services::secrets_store::{MARKER, SecretStore, split_secrets};
use crate::services::settings::SettingsService;
use crate::services::store::{SqlRow, Store, paginate};
use crate::sync::{Recover as _, RecoverRw as _};

/// Everything the lifecycle service is built from (the composition root fills it; tests use fakes).
pub struct LifecycleDeps {
    pub db: Arc<Store>,
    pub adapters: AdapterRegistry,
    pub fleet: Arc<dyn FleetPort>,
    pub hardware: Arc<HardwareService>,
    pub settings: Arc<SettingsService>,
    pub fit: Arc<FitService>,
    pub http: Arc<NodeHttp>,
    pub cfg: Arc<Settings>,
    pub bus: Arc<EventBus>,
    pub secrets: SecretStore,
}

/// compile/JIT caches survive container recreation; named volumes need no host path.
pub const NAMED_VOLUMES: &[(&str, &[(&str, &str)])] = &[(
    "vllm",
    &[
        ("ec-vllm-cache", "/root/.cache/vllm"),
        ("ec-flashinfer-cache", "/root/.cache/flashinfer"),
    ],
)];
pub const ACTIVE: &[&str] = &["starting", "loading", "ready", "stopping"];
pub const RUNNING: &[&str] = &["starting", "loading", "ready", "auth_required"];
/// the node could not act: the instance's state did not change
pub const UNAVAILABLE_CODES: &[&str] = &["node_unavailable", "timeout", "busy"];
pub const TOPIC: &str = "instances";
/// the node pushes its inventory on change and at least every 30 s: a younger gap is not proof
pub const INVENTORY_LAG_S: f64 = 40.0;
/// ports tried when gateway.start answers port_in_use
pub const GATEWAY_ATTEMPTS: usize = 8;
/// a vanished external engine shows `stopped` this long
pub const GONE_TTL_S: f64 = 30.0;

fn re(p: &str) -> Regex {
    Regex::new(p).unwrap_or_else(|_| unreachable!())
}

static ID_RE: LazyLock<Regex> = LazyLock::new(|| re(r"^[A-Za-z0-9_-]{1,64}$"));
static NAME_RE: LazyLock<Regex> = LazyLock::new(|| re(r"^[A-Za-z0-9][A-Za-z0-9 _.\-]{0,63}$"));
static ROOT_RE: LazyLock<Regex> = LazyLock::new(|| re(r"^[A-Za-z0-9_.\-]{1,32}$"));
static LABEL_MODEL: LazyLock<Regex> =
    LazyLock::new(|| re(r"^[A-Za-z0-9][A-Za-z0-9 _.:/@\-]{0,254}$"));
static NOT_SLUG: LazyLock<Regex> = LazyLock::new(|| re(r"[^a-z0-9]+"));
static NOT_NAME: LazyLock<Regex> = LazyLock::new(|| re(r"[^A-Za-z0-9 _.\-]+"));
static HF_DIR: LazyLock<Regex> = LazyLock::new(|| re(r"models--([^/]+?)--([^/]+)"));

/// Parsed command-line keys (inventory `params`, display-only parse of the redacted argv) that name weights an
/// engine loads: the model itself, a draft model for speculative decoding, a separate tokenizer or vision projector.
const MODEL_KEYS: &[&str] = &[
    "model",
    "model_path",
    "model_id",
    "model_name_or_path",
    "speculative_model",
    "speculative_draft_model_path",
    "draft_model",
    "tokenizer",
    "tokenizer_path",
    "mmproj",
];

fn take(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

/// `_slug`: the container-name part of an instance name.
fn slug40(s: &str) -> String {
    let out = take(
        NOT_SLUG
            .replace_all(&s.to_lowercase(), "-")
            .trim_matches('-'),
        40,
    );
    if out.is_empty() { "model".into() } else { out }
}

/// A derived instance name that satisfies InstanceName (it ends up in labels and is re-read at adoption).
fn safe_name(s: &str) -> String {
    let strip = [' ', '.', '-', '_'];
    let out = take(NOT_NAME.replace_all(s, "-").trim_matches(strip), 64)
        .trim_matches(strip)
        .to_string();
    if NAME_RE.is_match(&out) {
        out
    } else {
        "model".into()
    }
}

/// An Ollama server with no model named is a bare server; everything else must exist in the host's library.
fn needs_copy(r: &ModelRef) -> bool {
    r.format != "ollama" || r.name.as_deref().is_some_and(|n| !n.is_empty())
}

fn mask(text: &str, secrets: &[String]) -> String {
    let mut out = text.to_string();
    for sv in secrets {
        out = out.replace(sv.as_str(), "[redacted]");
    }
    out
}

fn sval(v: Option<&Value>) -> String {
    v.map_or_else(|| "None".to_string(), ec_proto::pyfmt::str_value)
}

fn truthy(v: Option<&Value>) -> bool {
    v.is_some_and(ec_proto::sanitize::truthy)
}

/// Does a container that mounts a folder holding `path` load that model? True when it cannot be told (no model
/// argument parsed), so a delete is refused rather than pulled from under a running engine.
fn loads_model(container: &Obj, path: &str) -> bool {
    let norm = |s: &str| s.trim().trim_end_matches('/').to_lowercase();
    let mut values: Vec<String> = Vec::new();
    if let Some(Value::Object(params)) = container.get("params") {
        for k in MODEL_KEYS {
            if let Some(v) = params.get(*k).filter(|v| ec_proto::sanitize::truthy(v)) {
                values.push(norm(&sval(Some(v))));
            }
        }
    }
    if let Some(Value::Object(labels)) = container.get("labels")
        && let Some(l) = labels
            .get("engine-console.model")
            .filter(|v| ec_proto::sanitize::truthy(v))
    {
        values.push(norm(&sval(Some(l))));
    }
    if values.is_empty() {
        return true;
    }
    let target = path.trim_end_matches('/').to_lowercase();
    let hf = HF_DIR.captures(&target);
    let (repo, folder) = match &hf {
        Some(c) => (Some(format!("{}/{}", &c[1], &c[2])), Some(c[0].to_string())),
        None => (None, None),
    };
    let base = target.rsplit('/').next().unwrap_or(&target).to_string();
    for v in values {
        if v == target || v.starts_with(&format!("{target}/")) {
            return true; // the same host path (mounted at the same place)
        }
        if let Some(repo) = &repo
            && (v == *repo
                || folder
                    .as_ref()
                    .is_some_and(|f| !f.is_empty() && v.contains(f.as_str())))
        {
            return true; // the repo id, or a path into that repo's cache folder
        }
        if repo.is_none() && v.rsplit('/').next().unwrap_or(&v) == base {
            return true; // a plain folder mounted somewhere else in the container
        }
    }
    false
}

/// The adapter's startup-log parser as the supervisor (a non-HTTP caller) must call it: the `try_*` form (PORTING
/// §A5), with a raised exception as `Kind: message`.
fn try_parse_startup_log(adapter: &dyn EngineAdapter, line: &str) -> Result<Option<Obj>, String> {
    adapter
        .try_parse_startup_log(line)
        .map_err(|e| e.to_string())
}

/// The problem a Python exception raised by an adapter becomes when nobody in the lifecycle catches its kind: the
/// generic 500 over HTTP (app.rs renders every `internal_error` the same way), and an error a non-HTTP caller logs.
/// The kind rides along (`exception`) so the callers that catch some kinds (`launch_for`, `_scrub`) can tell.
fn raised(e: &PyExc) -> AppError {
    AppError::Problem(internal(format!("{}: {}", e.kind, e.msg)).with("exception", e.kind.clone()))
}

/// Python's `except (ProblemError, NodeError, KeyError, ValueError)` around a launch build (`launch_for`, `_scrub`):
/// true for every error but an adapter exception of another kind.
fn caught_by_launch_for(e: &AppError) -> bool {
    match e
        .problem()
        .and_then(|p| p.extra.get("exception"))
        .and_then(Value::as_str)
    {
        Some(kind) => kind == "KeyError" || kind == "ValueError",
        None => true,
    }
}

/// A small insertion-ordered map with Python dict semantics (update keeps the position, delete then insert appends).
type OMap<V> = Vec<(String, V)>;

fn oget<'a, V>(m: &'a OMap<V>, k: &str) -> Option<&'a V> {
    m.iter().find(|(kk, _)| kk == k).map(|(_, v)| v)
}

fn oset<V>(m: &mut OMap<V>, k: &str, v: V) {
    if let Some(slot) = m.iter_mut().find(|(kk, _)| kk == k) {
        slot.1 = v;
    } else {
        m.push((k.to_string(), v));
    }
}

fn odel<V>(m: &mut OMap<V>, k: &str) {
    m.retain(|(kk, _)| kk != k);
}

/// Everything that identifies one launch of a managed instance (secrets resolved: memory only, never persisted).
#[derive(Debug, Clone)]
struct Plan {
    iid: String,
    name: String,
    engine: String,
    host_id: String,
    r: ModelRef,
    params: Obj,
    gpu_ids: Vec<i64>,
    port: i64,
    container: String,
    image_req: Option<String>,
}

impl Plan {
    fn display(&self) -> String {
        display_of(&self.r)
    }
}

#[derive(Debug, Clone)]
struct Built {
    /// the docker.run spec WITHOUT secret values
    run: Obj,
    /// name -> value; sent only inside the docker.run request
    secret_env: Vec<(String, String)>,
    launch: LaunchSpec,
    image: String,
    gpu_ids: Vec<i64>,
    gpu_uuids: Vec<String>,
    #[allow(dead_code)]
    location: Option<ModelLocation>,
    secret_values: Vec<String>,
}

impl Built {
    fn secret_env_obj(&self, values: bool) -> Obj {
        self.secret_env
            .iter()
            .map(|(k, v)| {
                let val = if values {
                    v.clone()
                } else {
                    MARKER.to_string()
                };
                (k.clone(), Value::String(val))
            })
            .collect()
    }
}

/// derived (script / external) instances, all keyed by host so one host's summary only rebuilds that host
#[derive(Default)]
struct Derived {
    /// when each derived id first appeared
    first_seen: HashMap<String, HashMap<String, f64>>,
    /// external engines seen running at the last look
    prev_ext: HashMap<String, OMap<Instance>>,
    /// vanished ones, shown stopped for a while
    gone: HashMap<String, OMap<(f64, Instance)>>,
    /// externals the hub itself stopped (so they can be started)
    stopped_ext: HashMap<String, OMap<Instance>>,
    /// container name -> derived instance id
    ext_by_container: HashMap<String, HashMap<String, String>>,
}

pub type Clock = Arc<dyn Fn() -> f64 + Send + Sync>;

pub struct LifecycleService {
    deps: LifecycleDeps,
    nodes: NodeCaller,
    library: RwLock<Arc<dyn LibraryPort>>,
    scripts: RwLock<Option<Arc<ScriptService>>>,
    locks: Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
    /// launch info per managed instance (in memory)
    built: Mutex<HashMap<String, Built>>,
    derived: Mutex<Derived>,
    links: Mutex<HashMap<String, (bool, Option<f64>)>>,
    clock: RwLock<Clock>,
}

impl std::fmt::Debug for LifecycleService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("LifecycleService")
    }
}

fn pc(code: &str, level: &str, message: impl Into<String>) -> PreflightCheck {
    PreflightCheck {
        code: code.into(),
        level: level.into(),
        message: message.into(),
    }
}

fn problem_of(e: &AppError) -> Option<&Problem> {
    e.problem()
}

impl LifecycleService {
    /// Python's constructor also registers itself as the fit estimator's resident provider and library user; the
    /// composition root does that wiring here (container.rs).
    pub fn new(deps: LifecycleDeps) -> Self {
        let nodes = NodeCaller::new(Arc::clone(&deps.fleet));
        Self {
            deps,
            nodes,
            library: RwLock::new(Arc::new(NullLibrary)),
            scripts: RwLock::new(None),
            locks: Mutex::new(HashMap::new()),
            built: Mutex::new(HashMap::new()),
            derived: Mutex::new(Derived::default()),
            links: Mutex::new(HashMap::new()),
            clock: RwLock::new(Arc::new(crate::services::common::now)),
        }
    }

    pub fn set_library(&self, library: Arc<dyn LibraryPort>) {
        self.deps.fit.set_library(Arc::clone(&library));
        *self.library.write_guard() = library;
    }

    pub fn set_scripts(&self, scripts: Arc<ScriptService>) {
        *self.scripts.write_guard() = Some(scripts);
    }

    /// Replace the clock (tests: Python monkeypatches `lifecycle.now`).
    pub fn set_clock(&self, clock: Clock) {
        *self.clock.write_guard() = clock;
    }

    fn now(&self) -> f64 {
        (*self.clock.read_guard())()
    }

    fn lib(&self) -> Arc<dyn LibraryPort> {
        Arc::clone(&self.library.read_guard())
    }

    fn scripts_svc(&self) -> Option<Arc<ScriptService>> {
        self.scripts.read_guard().clone()
    }

    fn cfg(&self) -> &Settings {
        &self.deps.cfg
    }

    // -- hosts ---------------------------------------------------------------------------------------------------
    fn host_lock(&self, host_id: &str) -> Arc<tokio::sync::Mutex<()>> {
        let mut locks = self.locks.guard();
        Arc::clone(locks.entry(host_id.to_string()).or_default())
    }

    /// Every host's label and addresses, to tell which configured machine an NFS export comes from.
    fn others(&self) -> Others {
        self.deps
            .fleet
            .hosts()
            .into_iter()
            .map(|h| {
                let addrs = self
                    .deps
                    .fleet
                    .node(&h.id)
                    .map(|n| nv::addresses(n.as_ref()))
                    .unwrap_or_default();
                (h.id, (h.label, addrs))
            })
            .collect()
    }

    fn ctx(&self, host_id: &str, others: Option<Others>) -> Result<HostCtx, Problem> {
        let node = self.nodes.node(host_id)?;
        Ok(HostCtx {
            host_id: host_id.to_string(),
            label: node.host().label.clone(),
            node,
            others: others.unwrap_or_else(|| self.others()),
        })
    }

    fn label(&self, host_id: &str) -> String {
        self.nodes.label(host_id)
    }

    // -- persistence ---------------------------------------------------------------------------------------------
    fn ref_of(r: &SqlRow) -> ModelRef {
        if let Some(m) = r.text("model").filter(|m| !m.is_empty())
            && let Ok(v) = serde_json::from_str::<ModelRef>(&m)
            && ["hf", "gguf", "ollama", "plain"].contains(&v.format.as_str())
        {
            return v;
        }
        ModelRef::hf(&r.str_or("repo_id", ""))
    }

    fn managed_controls(&self, host_id: &str) -> InstanceControls {
        let Ok(node) = self.nodes.node(host_id) else {
            return InstanceControls {
                reason: Some(format!("host {host_id} is not configured any more")),
                ..InstanceControls::default()
            };
        };
        let label = node.host().label.clone();
        if !nv::link_up(node.as_ref()) {
            let link = node.link();
            return InstanceControls {
                reason: Some(format!(
                    "{label} is not connected: {}",
                    link.error.filter(|e| !e.is_empty()).unwrap_or(link.state)
                )),
                ..InstanceControls::default()
            };
        }
        if nv::mode(node.as_ref()) != "manage" {
            return InstanceControls {
                logs: true,
                reason: Some(format!(
                    "{label}'s node is in monitor mode: engines there can be watched but not changed (its operator \
                     switches the policy mode to manage)"
                )),
                ..InstanceControls::default()
            };
        }
        InstanceControls {
            start: true,
            stop: true,
            restart: true,
            remove: true,
            logs: true,
            configure: true,
            reason: None,
        }
    }

    fn to_model(&self, r: &SqlRow) -> Instance {
        let host_id = r.str_or("host_id", "local");
        let row_state = r.str_or("state", "stopped");
        let mut state = row_state.clone();
        let mut reason: Option<String> = None;
        let cn = r.str_or("container_name", "");
        let names = [cn.clone()];
        let mut vram: Vec<InstanceVram> = Vec::new();
        let mut mem: Option<HostMemory> = None;
        let mut label = host_id.clone();
        match self.nodes.node(&host_id) {
            Ok(node) => {
                label = node.host().label.clone();
                if ACTIVE.contains(&state.as_str()) {
                    if nv::link_up(node.as_ref()) {
                        vram = vram_for(node.as_ref(), &names);
                        mem = memory_for(node.as_ref(), &names);
                    } else {
                        let link = node.link();
                        state = "unreachable".into();
                        reason = Some(format!(
                            "{label} is not connected: {}",
                            link.error.filter(|e| !e.is_empty()).unwrap_or(link.state)
                        ));
                    }
                }
            }
            Err(_) => {
                if ACTIVE.contains(&state.as_str()) {
                    state = "unreachable".into();
                    reason = Some(format!("host {host_id} is not configured any more"));
                }
            }
        }
        let started = r.f64("started_at");
        let up = started
            .filter(|s| *s != 0.0 && ["starting", "loading", "ready"].contains(&row_state.as_str()))
            .map(|s| self.now() - s);
        let rf = Self::ref_of(r);
        let port = r.i64("port");
        let container_port = r.i64("container_port");
        let obj_col = |c: &str| match r.json_text(c) {
            Value::Object(o) => o,
            _ => Obj::new(),
        };
        let ints = |c: &str| -> Vec<i64> {
            match r.json_text(c) {
                Value::Array(a) => a.iter().filter_map(Value::as_i64).collect(),
                _ => Vec::new(),
            }
        };
        let strs = |c: &str| -> Vec<String> {
            match r.json_text(c) {
                Value::Array(a) => a.iter().map(|x| sval(Some(x))).collect(),
                _ => Vec::new(),
            }
        };
        let fit = r
            .text("fit")
            .filter(|f| !f.is_empty())
            .and_then(|f| serde_json::from_str::<Value>(&f).ok())
            .and_then(|v| v.as_object().cloned());
        let location = self.location_of(&host_id, &rf);
        let controls = self.managed_controls(&host_id);
        Instance {
            id: r.str_or("id", ""),
            name: r.str_or("name", ""),
            engine: r.str_or("engine", ""),
            repo_id: r.str_or("repo_id", ""),
            params: obj_col("params"),
            gpu_ids: ints("gpu_ids"),
            gpu_uuids: strs("gpu_uuids"),
            port,
            container_name: Some(cn.clone()),
            container_port,
            internal_endpoint: container_port
                .filter(|p| *p != 0)
                .map(|p| format!("http://{cn}:{p}")),
            image: r.text("image"),
            state,
            phase: r.text("phase"),
            progress_pct: r.f64("progress_pct"),
            pinned: r.bool("pinned"),
            ttl_idle_s: r.i64("ttl_idle_s"),
            profile_id: r.text("profile_id"),
            error: r.text("error"),
            last_logs: r.text("last_logs"),
            fit,
            created_at: r.f64("created_at").unwrap_or(0.0),
            started_at: started,
            last_request_at: r.f64("last_request_at"),
            uptime_s: up,
            managed: true,
            source: "console".into(),
            endpoint: port
                .filter(|p| *p != 0)
                .map(|p| format!("http://127.0.0.1:{p}")),
            served_models: Vec::new(),
            state_reason: reason,
            history_since: None,
            host_memory: mem,
            host_id,
            host_label: Some(label),
            management: "managed".into(),
            script: None,
            vram,
            model: Some(rf),
            model_location: location,
            controls,
        }
    }

    fn location_of(&self, host_id: &str, r: &ModelRef) -> Option<ModelLocation> {
        if !needs_copy(r) {
            return None;
        }
        self.lib()
            .locate(host_id, r)
            .map(|c| Self::location_from_copy(&c))
    }

    fn location_from_copy(copy: &LibraryCopy) -> ModelLocation {
        let d = copy.drive.as_ref();
        let remote = d.is_some_and(|d| d.remote);
        ModelLocation {
            host_id: copy.host_id.clone(),
            root_id: copy.root_id.clone(),
            root_label: copy.root_label.clone(),
            path: copy.path.clone(),
            drive: d.map(|d| d.label.clone()),
            fstype: d.and_then(|d| d.fstype.clone()),
            remote,
            shared_from: d.filter(|_| remote).and_then(|d| {
                d.remote_host_id
                    .clone()
                    .filter(|s| !s.is_empty())
                    .or_else(|| d.remote_host.clone())
            }),
        }
    }

    fn row(&self, iid: &str) -> AppResult<Option<SqlRow>> {
        self.deps
            .db
            .one("SELECT * FROM instances WHERE id=?", &[&iid])
    }

    fn rows(&self, sql: &str, params: &[&dyn rusqlite::ToSql]) -> Vec<SqlRow> {
        self.deps.db.all(sql, params).unwrap_or_default()
    }

    // -- derived (script / external) instances --------------------------------------------------------------------
    fn busy(&self, host_id: &str) -> Vec<(String, String)> {
        self.scripts_svc()
            .map(|s| s.busy(host_id))
            .unwrap_or_default()
    }

    /// Script and external instances of every host (or one), from the pushed inventories. A vanished external
    /// engine stays listed as `stopped` for a short while, and one the hub itself stopped stays so it can be started.
    pub fn externals(&self, host_id: Option<&str>) -> Vec<Instance> {
        let t = self.now();
        let others = self.others();
        let mut out = Vec::new();
        for h in self.deps.fleet.hosts() {
            if host_id.is_none_or(|x| x == h.id)
                && let Ok(ctx) = self.ctx(&h.id, Some(others.clone()))
            {
                out.extend(self.externals_of(&ctx, t));
            }
        }
        out
    }

    fn externals_of(&self, ctx: &HostCtx, t: f64) -> Vec<Instance> {
        let hid = ctx.host_id.clone();
        let busy = self.busy(&hid);
        let mut d = self.derived.guard();
        let mut current: OMap<Instance> = Vec::new();
        let mut by_container: HashMap<String, String> = HashMap::new();
        {
            let seen = d.first_seen.entry(hid.clone()).or_default();
            for inst in build_host_instances(ctx, seen, t, &busy) {
                if let Some(cn) = &inst.container_name {
                    by_container.insert(cn.clone(), inst.id.clone());
                }
                let id = inst.id.clone();
                oset(
                    &mut current,
                    &id,
                    Self::overlay_link(inst, ctx.node.as_ref()),
                );
            }
        }
        let containers = nv::containers(ctx.node.as_ref());
        for s in nv::scripts(ctx.node.as_ref()) {
            // every container of a script's engine maps to the script's instance
            for c in &containers {
                let name = sval(c.get("name"));
                if matches_script(&name, &s) {
                    by_container.insert(name, script_instance_id(&hid, &sval(s.get("name"))));
                }
            }
        }
        d.ext_by_container.insert(hid.clone(), by_container);
        let prev = d.prev_ext.get(&hid).cloned().unwrap_or_default();
        if nv::link_up(ctx.node.as_ref()) {
            for (iid, inst) in &prev {
                // an external engine that was running and is not any more
                if oget(&current, iid).is_none() && inst.state != "stopped" {
                    let stopped = Self::as_stopped(inst, ctx);
                    let gone = d.gone.entry(hid.clone()).or_default();
                    oset(gone, iid, (t + GONE_TTL_S, stopped));
                }
            }
        }
        d.prev_ext.insert(
            hid.clone(),
            current
                .iter()
                .filter(|(_, v)| v.management == "external")
                .cloned()
                .collect(),
        );
        let gone = d.gone.entry(hid.clone()).or_default();
        gone.retain(|(i, (until, _))| !(*until < t || oget(&current, i).is_some()));
        let gone_now = gone.clone();
        let stopped = d.stopped_ext.entry(hid.clone()).or_default();
        stopped.retain(|(i, _)| oget(&current, i).is_none()); // the container is running again
        let stopped_now = stopped.clone();
        if let Some(seen) = d.first_seen.get_mut(&hid) {
            seen.retain(|i, _| {
                oget(&current, i).is_some()
                    || oget(&gone_now, i).is_some()
                    || oget(&stopped_now, i).is_some()
            });
        }
        let mut out: Vec<Instance> = current.into_iter().map(|(_, v)| v).collect();
        out.extend(gone_now.iter().map(|(_, (_, inst))| inst.clone()));
        out.extend(
            stopped_now
                .iter()
                .filter(|(iid, _)| oget(&gone_now, iid).is_none())
                .map(|(_, inst)| inst.clone()),
        );
        out
    }

    /// What was last pushed by a node we cannot reach any more is not live: show it as unreachable.
    fn overlay_link(mut inst: Instance, node: &dyn NodeClient) -> Instance {
        if nv::link_up(node) || inst.state == "stopped" || inst.state == "unreachable" {
            return inst;
        }
        let link = node.link();
        inst.state = "unreachable".into();
        inst.state_reason = Some(format!(
            "{} is not connected: {}",
            inst.host_label.clone().unwrap_or_else(|| "None".into()),
            link.error.filter(|e| !e.is_empty()).unwrap_or(link.state)
        ));
        inst
    }

    fn as_stopped(inst: &Instance, ctx: &HostCtx) -> Instance {
        let mut s = inst.clone();
        s.state = "stopped".into();
        s.state_reason = Some("container is no longer running".into());
        s.vram = Vec::new();
        s.host_memory = None;
        s.controls = external_controls(ctx, "stopped");
        s
    }

    // -- queries -------------------------------------------------------------------------------------------------
    /// 404 "no such instance: <iid>"
    pub fn get(&self, iid: &str) -> AppResult<Instance> {
        if iid.starts_with(EXT) || iid.starts_with(SCRIPT) {
            return self
                .externals(None)
                .into_iter()
                .find(|i| i.id == iid)
                .ok_or_else(|| not_found(format!("no such instance: {iid}")).into());
        }
        match self.row(iid)? {
            Some(r) => Ok(self.to_model(&r)),
            None => Err(not_found(format!("no such instance: {iid}")).into()),
        }
    }

    fn instances_on(&self, host_id: &str) -> Vec<Instance> {
        let mut out: Vec<Instance> = self
            .rows("SELECT * FROM instances WHERE host_id=?", &[&host_id])
            .iter()
            .map(|r| self.to_model(r))
            .collect();
        out.extend(self.externals(Some(host_id)));
        out
    }

    /// Managed, script and external instances of every machine (or one).
    pub fn list_page(
        &self,
        limit: i64,
        cursor: Option<&str>,
        host_id: Option<&str>,
    ) -> AppResult<Page<Instance>> {
        let (limit, off) = paginate(limit, cursor);
        let host = host_id.filter(|h| !h.is_empty());
        let rows = match host {
            Some(h) => self.deps.db.all(
                "SELECT * FROM instances WHERE host_id=? ORDER BY created_at DESC",
                &[&h],
            )?,
            None => self
                .deps
                .db
                .all("SELECT * FROM instances  ORDER BY created_at DESC", &[])?,
        };
        let mut items: Vec<Instance> = rows.iter().map(|r| self.to_model(r)).collect();
        items.extend(self.externals(host));
        let (lim, off) = (
            usize::try_from(limit).unwrap_or(0),
            usize::try_from(off).unwrap_or(0),
        );
        let next = (items.len() > off + lim).then(|| (off + lim).to_string());
        let page: Vec<Instance> = items.into_iter().skip(off).take(lim).collect();
        Ok(Page::new(page, next))
    }

    /// Every instance of every host (managed, script, external).
    pub fn all_instances(&self) -> Vec<Instance> {
        let mut out: Vec<Instance> = self
            .rows("SELECT * FROM instances", &[])
            .iter()
            .map(|r| self.to_model(r))
            .collect();
        out.extend(self.externals(None));
        out
    }

    pub fn all_active(&self) -> Vec<Instance> {
        self.rows(
            "SELECT * FROM instances WHERE state IN ('starting','loading','ready','stopping')",
            &[],
        )
        .iter()
        .map(|r| self.to_model(r))
        .collect()
    }

    /// The instance of a host that owns a gateway/published port or a container name (metrics routing).
    pub fn lookup(
        &self,
        host_id: &str,
        port: Option<i64>,
        container: Option<&str>,
    ) -> Option<Instance> {
        let container = container.filter(|c| !c.is_empty());
        for r in self.rows(
            "SELECT * FROM instances WHERE host_id=? AND state IN ('starting','loading','ready')",
            &[&host_id],
        ) {
            let cn = r.str_or("container_name", "");
            if (port.is_some() && r.i64("port") == port)
                || container.is_some_and(|c| c == cn || c == gateway_name(&cn))
            {
                return Some(self.to_model(&r));
            }
        }
        self.externals(Some(host_id)).into_iter().find(|i| {
            (port.is_some() && i.port == port)
                || container.is_some_and(|c| i.container_name.as_deref() == Some(c))
        })
    }

    /// The console instance id behind a container name on a host (to label GPU processes); None if unknown.
    pub fn instance_for_container(&self, host_id: &str, container: &str) -> Option<String> {
        if let Ok(Some(r)) = self.deps.db.one(
            "SELECT id FROM instances WHERE host_id=? AND container_name=?",
            &[&host_id, &container],
        ) {
            return r.text("id");
        }
        let known = self
            .derived
            .guard()
            .ext_by_container
            .get(host_id)
            .and_then(|m| m.get(container).cloned());
        if known.is_some() {
            return known;
        }
        let node = self.nodes.node(host_id).ok()?;
        if let Some(s) = nv::scripts(node.as_ref())
            .into_iter()
            .find(|s| matches_script(container, s))
        {
            return Some(script_instance_id(host_id, &sval(s.get("name"))));
        }
        let c = nv::container_named(node.as_ref(), container)?;
        (truthy(c.get("engine")) && !truthy(c.get("managed")))
            .then(|| external_instance_id(host_id, container))
    }

    pub fn counts(&self, host_id: &str) -> HostCounts {
        let insts = self.instances_on(host_id);
        let copies = self.lib().copies(host_id);
        HostCounts {
            engines_running: i64::try_from(
                insts
                    .iter()
                    .filter(|i| RUNNING.contains(&i.state.as_str()))
                    .count(),
            )
            .unwrap_or(0),
            engines_total: i64::try_from(insts.len()).unwrap_or(0),
            models: i64::try_from(copies.len()).unwrap_or(0),
            models_bytes: copies.iter().map(|c| c.size_bytes).sum(),
        }
    }

    /// 404 for an unknown id, 409 for an instance the console does not manage.
    pub fn require_managed(&self, iid: &str) -> AppResult<Instance> {
        let inst = self.get(iid)?;
        if inst.management != "managed" {
            let what = if inst.management == "script" {
                "a script's engine: use its start/stop/restart (they run the script)"
            } else {
                "an external engine the console does not own (monitor-only)"
            };
            return Err(conflict(format!("this is {what}"))
                .code("instance_not_managed")
                .into());
        }
        Ok(inst)
    }

    fn cn(inst: &Instance) -> Result<String, Problem> {
        inst.container_name
            .clone()
            .ok_or_else(|| conflict("this instance has no container").code("instance_not_managed"))
    }

    /// A request was seen: postpones the idle TTL.
    pub fn touch(&self, iid: &str) {
        let t = self.now();
        let _ = self.deps.db.execute(
            "UPDATE instances SET last_request_at=? WHERE id=?",
            &[&t, &iid],
        );
    }

    /// `_set`: update columns (column names are literals from this module) and publish the instance.
    #[allow(clippy::needless_pass_by_value)] // callers build the column list in place
    fn set(&self, iid: &str, cols: Vec<(&'static str, SqlValue)>) -> AppResult<()> {
        let sets: Vec<String> = cols.iter().map(|(k, _)| format!("{k}=?")).collect();
        let sql = format!("UPDATE instances SET {} WHERE id=?", sets.join(", "));
        let id = SqlValue::Text(iid.to_string());
        let mut params: Vec<&dyn rusqlite::ToSql> = cols
            .iter()
            .map(|(_, v)| v as &dyn rusqlite::ToSql)
            .collect();
        params.push(&id);
        self.deps.db.execute(&sql, &params)?;
        if let Ok(inst) = self.get(iid) {
            self.deps
                .bus
                .publish(TOPIC, serde_json::to_value(inst).unwrap_or(Value::Null));
        }
        Ok(())
    }

    // -- resident engines (for "fits if you stop X") and mounts in use ----------------------------------------------
    pub fn resident(&self, host_id: &str) -> Vec<ResidentUse> {
        let mut out = Vec::new();
        for i in self.instances_on(host_id) {
            if ["stopped", "failed", "stopping", "unreachable"].contains(&i.state.as_str()) {
                continue;
            }
            let known: Vec<&InstanceVram> =
                i.vram.iter().filter(|v| v.used_gib.is_some()).collect();
            if !known.is_empty() {
                #[allow(clippy::cast_precision_loss)]
                let per = known.iter().map(|v| v.used_gib.unwrap_or(0.0)).sum::<f64>()
                    / known.len() as f64;
                out.push(ResidentUse {
                    name: i.name.clone(),
                    gpu_ids: known.iter().map(|v| v.gpu_index).collect(),
                    gib_per_gpu: per,
                });
            } else if i.management == "managed" {
                let per = i
                    .fit
                    .as_ref()
                    .and_then(|f| f.get("per_gpu"))
                    .and_then(Value::as_array)
                    .and_then(|a| a.first())
                    .and_then(|g| g.get("total_gib"))
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0);
                out.push(ResidentUse {
                    name: i.name.clone(),
                    gpu_ids: i.gpu_ids.clone(),
                    gib_per_gpu: per,
                });
            }
        }
        out
    }

    /// Ids of the instances of a host using the model at `path`: their container mounts the model itself (or a
    /// part of it), or mounts a folder that holds it AND loads that very model. Engines usually mount a whole
    /// shared model cache; counting every model in it as in use would make none of them deletable.
    pub fn instances_using(&self, host_id: &str, path: &str) -> Vec<String> {
        let Ok(node) = self.nodes.node(host_id) else {
            return Vec::new();
        };
        let norm = path.trim_end_matches('/').to_string();
        let mut ids: Vec<String> = Vec::new();
        for c in nv::containers(node.as_ref()) {
            let st = c.get("state").and_then(Value::as_str).unwrap_or("");
            if !["running", "restarting", "paused", "created"].contains(&st) {
                continue;
            }
            let mounts = match c.get("mounts") {
                Some(Value::Array(a)) => a.clone(),
                _ => Vec::new(),
            };
            for m in mounts {
                let src = m
                    .get("source")
                    .filter(|v| ec_proto::sanitize::truthy(v))
                    .map_or_else(String::new, |v| sval(Some(v)));
                let src = src.trim_end_matches('/');
                if src.is_empty() {
                    continue;
                }
                let own = norm == src || src.starts_with(&format!("{norm}/"));
                let holder = norm.starts_with(&format!("{src}/"));
                if own || (holder && loads_model(&c, &norm)) {
                    if let Some(iid) = self.instance_for_container(host_id, &sval(c.get("name")))
                        && !ids.contains(&iid)
                    {
                        ids.push(iid);
                    }
                    break;
                }
            }
        }
        // a launch the inventory has not caught up with yet
        for r in self.rows(
            "SELECT * FROM instances WHERE host_id=? AND state IN ('starting','loading','ready','stopping')",
            &[&host_id],
        ) {
            let id = r.str_or("id", "");
            let Some(loc) = self.location_of(host_id, &Self::ref_of(&r)) else {
                continue;
            };
            if ids.contains(&id) {
                continue;
            }
            let base = loc.path.trim_end_matches('/');
            if norm == base
                || norm.starts_with(&format!("{base}/"))
                || base.starts_with(&format!("{norm}/"))
            {
                ids.push(id);
            }
        }
        ids
    }

    // -- build the launch ------------------------------------------------------------------------------------------
    /// Stored params with MARKER placeholders replaced by the real secret values (memory only).
    fn real_params(&self, inst: &Instance) -> Obj {
        let stored = self.deps.secrets.get(&inst.id);
        inst.params
            .iter()
            .map(|(k, v)| {
                let real = if v.as_str() == Some(MARKER) {
                    stored
                        .iter()
                        .find(|(sk, _)| sk == k)
                        .map_or_else(|| v.clone(), |(_, sv)| Value::String(sv.clone()))
                } else {
                    v.clone()
                };
                (k.clone(), real)
            })
            .collect()
    }

    fn secret_param_values(params: &Obj) -> Vec<String> {
        params
            .iter()
            .filter(|(k, _)| crate::services::common::is_sensitive_key(k))
            .filter_map(|(_, v)| v.as_str())
            .filter(|s| !s.is_empty() && *s != MARKER)
            .map(str::to_string)
            .collect()
    }

    fn param_errors(adapter: &dyn EngineAdapter, params: &Obj) -> Vec<String> {
        let mut errs: Vec<String> = adapter
            .param_catalog()
            .into_iter()
            .filter(|s| s.flag == "@image" && params.contains_key(&s.key))
            .map(|s| {
                format!(
                    "'{}' is a per-instance image override; choose the image in the launch form (or settings pins)",
                    s.key
                )
            })
            .collect();
        errs.extend(
            params
                .iter()
                .filter(|(_, v)| v.as_str() == Some(MARKER))
                .map(|(k, _)| {
                    format!("'{k}' still holds the [set] placeholder: supply the real value")
                }),
        );
        let real: Obj = params
            .iter()
            .filter(|(_, v)| v.as_str() != Some(MARKER))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        errs.extend(adapter.validate(&real));
        let secrets = Self::secret_param_values(params);
        errs.iter().map(|e| mask(e, &secrets)).collect()
    }

    /// InstanceCreate.image, else the settings pin for (host, engine) or engine, else the adapter's pinned image for
    /// the host's CPU architecture.
    fn pick_image(
        &self,
        adapter: &dyn EngineAdapter,
        node: &dyn NodeClient,
        host_id: &str,
        requested: Option<&str>,
    ) -> AppResult<String> {
        if let Some(r) = requested.filter(|r| !r.is_empty()) {
            return Ok(r.to_string());
        }
        if let Some(p) = self.deps.settings.image_pin(adapter.id(), host_id)? {
            return Ok(p);
        }
        Ok(adapter.image_for(&nv::arch(node)))
    }

    fn volumes(adapter: &dyn EngineAdapter) -> Vec<(String, String)> {
        NAMED_VOLUMES
            .iter()
            .find(|(id, _)| *id == adapter.id())
            .map(|(_, v)| {
                v.iter()
                    .map(|(n, d)| ((*n).to_string(), (*d).to_string()))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// The docker.run spec for a plan, validated the way the node will validate it.
    #[allow(clippy::too_many_lines)]
    fn build(&self, plan: &Plan) -> AppResult<Built> {
        let adapter = self.deps.adapters.get(&plan.engine)?;
        let node = self.nodes.node(&plan.host_id)?;
        check_ref(&plan.r)?;
        let r = &plan.r;
        let mut formats = adapter.model_formats();
        if !formats.contains(&r.format) {
            formats.sort();
            return Err(unprocessable(format!(
                "{} cannot load {} models (it loads: {})",
                adapter.display_name(),
                r.format,
                formats.join(", ")
            ))
            .code("format_unsupported")
            .into());
        }
        let ids = (!plan.gpu_ids.is_empty()).then_some(plan.gpu_ids.as_slice());
        let hw = self.deps.hardware.hardware(&plan.host_id, ids)?;
        let lib = self.lib();
        let copy = if needs_copy(r) {
            lib.locate(&plan.host_id, r)
        } else {
            None
        };
        if copy.is_none() && needs_copy(r) {
            return Err(unprocessable(format!(
                "{} is not in {}'s model library",
                plan.display(),
                self.label(&plan.host_id)
            ))
            .code("model_missing")
            .into());
        }
        let mut mounts: Vec<Obj>;
        let model_path: Option<String>;
        let mut model_arg = plan.display();
        match r.format.as_str() {
            "hf" => {
                model_arg = r
                    .repo_id
                    .clone()
                    .filter(|s| !s.is_empty())
                    .unwrap_or_else(|| plan.display());
                mounts = Vec::new();
                model_path = None;
            }
            "ollama" => {
                let root = lib.ollama_root(&plan.host_id);
                (mounts, model_path) = model_mounts(
                    r,
                    copy.as_ref().map(|c| c.rel.as_str()),
                    root.as_ref().map(|x| (x.id.as_str(), x.path.as_str())),
                );
                model_arg = r.name.clone().unwrap_or_default();
            }
            _ => {
                (mounts, model_path) = model_mounts(r, copy.as_ref().map(|c| c.rel.as_str()), None);
            }
        }
        let ctx = LaunchContext {
            model: model_arg,
            model_format: r.format.clone(),
            model_path,
            served_name: plan.display(),
            hf_cache_container_path: self.cfg().hf_cache_container_path.clone(),
            params: plan.params.clone(),
            hw: hw.clone(),
        };
        // Python: `except ValueError as e: raise Unprocessable(str(e), code="invalid_params")`; any other exception
        // propagates (the supervisor and the pull fallback call this too, so only the `try_*` form, §A5)
        let launch = match adapter.try_build_launch_ctx(&ctx) {
            Ok(r) => r?,
            Err(e) if e.kind == "ValueError" => {
                return Err(unprocessable(e.msg).code("invalid_params").into());
            }
            Err(e) => return Err(raised(&e)),
        };
        if launch.needs_hf_cache {
            mounts.extend(self.hf_mounts_for(node.as_ref(), &plan.host_id, r, copy.as_ref()));
        }
        let pol = nv::policy(node.as_ref());
        let image = self.pick_image(
            adapter.as_ref(),
            node.as_ref(),
            &plan.host_id,
            plan.image_req.as_deref(),
        )?;
        if !nv::image_allowed(&image, &nv::engine_images(node.as_ref())) {
            return Err(unprocessable(format!(
                "image {image} is not allowed by {}'s policy (engine_images)",
                self.label(&plan.host_id)
            ))
            .code("image_not_allowed")
            .into());
        }
        let mut secret_values = Self::secret_param_values(&plan.params);
        if launch
            .argv
            .iter()
            .any(|arg| secret_values.iter().any(|sv| arg.contains(sv.as_str())))
        {
            return Err(unprocessable(
                "this engine adapter would put a secret parameter (e.g. api_key) on the container command line, \
                 where any local user could read it; unset it or use an engine that takes it from an environment \
                 variable",
            )
            .code("secret_in_argv")
            .into());
        }
        let (env, secret_env) = split_env(&launch.env, self.cfg().engine_hf_offline);
        let ipc_host = launch.ipc_host
            && hw.gpu_uuids.len() > 1
            && pol.get("allow_ipc_host") != Some(&Value::Bool(false));
        let labels = engine_labels(&LabelArgs {
            instance: &plan.iid,
            engine: adapter.id(),
            display: &plan.display(),
            r,
            port: plan.port,
            name: &plan.name,
            gpu_ids: &hw.gpu_ids,
            gpu_uuids: &hw.gpu_uuids,
        });
        // secret names only for the validation; the values travel separately, in docker.run only
        let names_only: Obj = secret_env
            .iter()
            .map(|(k, _)| (k.clone(), json!("x")))
            .collect();
        let volumes = Self::volumes(adapter.as_ref());
        let mut run = build_run_spec(&RunArgs {
            name: &plan.container,
            instance: &plan.iid,
            image: &image,
            launch: &launch,
            gpu_uuids: &hw.gpu_uuids,
            labels: &labels,
            env: &env,
            secret_env: &names_only,
            mounts: &mounts,
            volumes: &volumes,
            ipc_host,
        });
        let errs = validate_run_spec(&run, nv::num(pol.get("max_shm_gib")));
        if !errs.is_empty() {
            return Err(unprocessable(errs.join("; ")).code("invalid_launch").into());
        }
        run.remove("secret_env");
        secret_values.extend(
            secret_env
                .iter()
                .map(|(_, v)| v.clone())
                .filter(|v| !v.is_empty()),
        );
        Ok(Built {
            run,
            secret_env,
            launch,
            image,
            gpu_ids: hw.gpu_ids,
            gpu_uuids: hw.gpu_uuids,
            location: copy.as_ref().map(Self::location_from_copy),
            secret_values,
        })
    }

    fn hf_mounts_for(
        &self,
        node: &dyn NodeClient,
        host_id: &str,
        r: &ModelRef,
        copy: Option<&LibraryCopy>,
    ) -> Vec<Obj> {
        let hf_path = &self.cfg().hf_cache_container_path;
        if r.format == "hf"
            && let Some(copy) = copy
        {
            let roots = nv::model_roots(node);
            let kind = roots
                .iter()
                .find(|(k, _)| *k == copy.root_id)
                .and_then(|(_, x)| x.get("kind").and_then(Value::as_str).map(str::to_string));
            return hf_mounts(&copy.root_id, kind.as_deref(), &copy.rel, hf_path);
        }
        // another format whose adapter still wants the HF cache next to it
        self.lib()
            .hf_root(host_id)
            .map(|hf| hf_mounts(&hf.id, Some(&hf.kind), "", hf_path))
            .unwrap_or_default()
    }

    fn plan_of(&self, inst: &Instance, port: Option<i64>) -> AppResult<Plan> {
        Ok(Plan {
            iid: inst.id.clone(),
            name: inst.name.clone(),
            engine: inst.engine.clone(),
            host_id: inst.host_id.clone(),
            r: inst
                .model
                .clone()
                .unwrap_or_else(|| ModelRef::hf(&inst.repo_id)),
            params: self.real_params(inst),
            gpu_ids: inst.gpu_ids.clone(),
            port: port
                .filter(|p| *p != 0)
                .or(inst.port.filter(|p| *p != 0))
                .unwrap_or(self.cfg().port_range_start),
            container: Self::cn(inst)?,
            image_req: self.row_image_req(&inst.id),
        })
    }

    fn row_image_req(&self, iid: &str) -> Option<String> {
        self.row(iid)
            .ok()
            .flatten()
            .and_then(|r| r.text("image_req"))
    }

    fn cached(&self, iid: &str) -> Option<Built> {
        self.built.guard().get(iid).cloned()
    }

    fn cache(&self, iid: &str, built: Built) {
        self.built.guard().insert(iid.to_string(), built);
    }

    /// The adapter's LaunchSpec for a managed instance (health/metrics paths); None when it cannot be built now.
    pub fn launch_for(&self, inst: &Instance) -> Option<LaunchSpec> {
        self.launch_for_checked(inst).ok().flatten()
    }

    /// `launch_for` with Python's exception semantics: problems, node errors and a raised KeyError/ValueError are
    /// None; an adapter exception of any other kind is the `Err` the caller's round fails with.
    pub fn launch_for_checked(&self, inst: &Instance) -> AppResult<Option<LaunchSpec>> {
        if let Some(b) = self.cached(&inst.id) {
            return Ok(Some(b.launch));
        }
        match self.plan_of(inst, None).and_then(|p| self.build(&p)) {
            Ok(built) => {
                let launch = built.launch.clone();
                self.cache(&inst.id, built);
                Ok(Some(launch))
            }
            Err(e) if caught_by_launch_for(&e) => Ok(None),
            Err(e) => Err(e),
        }
    }

    // -- ports -----------------------------------------------------------------------------------------------------
    /// Free ports of the host's gateway range: not used by another active instance there, not published by any
    /// container in its inventory (a stopped instance's own old gateway does not count against it).
    fn candidate_ports(&self, host_id: &str, exclude_id: Option<&str>) -> AppResult<Vec<i64>> {
        let node = self.nodes.node(host_id)?;
        let (lo, hi) = nv::port_range(
            node.as_ref(),
            (self.cfg().port_range_start, self.cfg().port_range_end),
        );
        let ex = exclude_id.unwrap_or("");
        let mut used: Vec<i64> = self
            .deps
            .db
            .all(
                "SELECT port FROM instances WHERE host_id=? AND state IN ('starting','loading','ready','stopping') AND id != ?",
                &[&host_id, &ex],
            )?
            .iter()
            .filter_map(|r| r.i64("port"))
            .filter(|p| *p != 0)
            .collect();
        let mut mine: Vec<String> = Vec::new();
        for c in nv::containers(node.as_ref()) {
            if exclude_id.is_some_and(|x| {
                !x.is_empty() && c.get("instance").and_then(Value::as_str) == Some(x)
            }) {
                mine.push(sval(c.get("name")));
                continue;
            }
            used.extend(nv::host_ports(&c));
        }
        for e in nv::endpoints(node.as_ref()) {
            let theirs = match e.get("container") {
                Some(Value::String(c)) => !mine.contains(c),
                _ => true,
            };
            if theirs && let Some(p) = e.get("port").and_then(nv::int_of) {
                used.push(p);
            }
        }
        Ok((lo..=hi).filter(|p| !used.contains(p)).collect())
    }

    fn alloc_port(&self, host_id: &str, exclude_id: Option<&str>) -> AppResult<Option<i64>> {
        Ok(self.candidate_ports(host_id, exclude_id)?.first().copied())
    }

    // -- preflight ---------------------------------------------------------------------------------------------------
    fn ref_for(req: &InstanceCreate) -> ModelRef {
        let mut r = req.model.clone().unwrap_or_else(|| ModelRef {
            repo_id: req.repo_id.clone(),
            ..ModelRef::hf("")
        });
        if r.format == "hf"
            && r.repo_id.as_deref().is_none_or(str::is_empty)
            && req.repo_id.as_deref().is_some_and(|x| !x.is_empty())
        {
            r.repo_id = req.repo_id.clone();
        }
        r
    }

    /// The checks POST /instances performs, as a dry run (`params`: the profile's params under the body's).
    pub async fn preflight(
        &self,
        req: &InstanceCreate,
        params: &Obj,
    ) -> AppResult<PreflightReport> {
        Ok(Box::pin(self.preflight_full(req, None, Some(params)))
            .await?
            .0)
    }

    /// Python's `preflight(req, exclude_id=, params=)`: the report and the port it would use.
    #[allow(clippy::too_many_lines)]
    pub async fn preflight_full(
        &self,
        req: &InstanceCreate,
        exclude_id: Option<&str>,
        params: Option<&Obj>,
    ) -> AppResult<(PreflightReport, Option<i64>)> {
        let host_id = req.host_id.as_str();
        let node = self.nodes.node(host_id)?; // 404 unknown_host
        let label = node.host().label.clone();
        let adapter = self.deps.adapters.get(&req.engine)?;
        let p = params.cloned().unwrap_or_else(|| req.params.clone());
        let r = Self::ref_for(req);
        check_ref(&r)?;
        let display = display_of(&r);
        let mut checks: Vec<PreflightCheck> = Vec::new();
        let mut fit: Option<FitReport> = None;

        if !nv::link_up(node.as_ref()) {
            let link = node.link();
            checks.push(pc(
                "host_down",
                "block",
                format!(
                    "{label} is not connected ({}): nothing can be started there until its link is up",
                    link.error.filter(|e| !e.is_empty()).unwrap_or(link.state)
                ),
            ));
            return Ok((
                PreflightReport {
                    ok: false,
                    gpu_ids: req.gpu_ids.clone(),
                    checks,
                    fit: None,
                },
                None,
            ));
        }
        if nv::mode(node.as_ref()) != "manage" {
            let pol = nv::policy(node.as_ref());
            let where_ = pol
                .get("path")
                .filter(|v| ec_proto::sanitize::truthy(v))
                .map_or_else(|| "its policy file".to_string(), |v| sval(Some(v)));
            checks.push(pc(
                "monitor_only",
                "block",
                format!(
                    "{label}'s node is in monitor mode ({where_}): it can be watched but not changed. Its operator \
                     must switch \"mode\" to \"manage\" in that policy file."
                ),
            ));
        }
        let errs = Self::param_errors(adapter.as_ref(), &p);
        if !errs.is_empty() {
            checks.push(pc("invalid_params", "block", errs.join("; ")));
        }
        let mut formats = adapter.model_formats();
        let format_ok = formats.contains(&r.format);
        if !format_ok {
            formats.sort();
            checks.push(pc(
                "format_unsupported",
                "block",
                format!(
                    "{} cannot load {} models (it loads: {})",
                    adapter.display_name(),
                    r.format,
                    formats.join(", ")
                ),
            ));
        }
        let copy = if needs_copy(&r) {
            self.lib().locate(host_id, &r)
        } else {
            None
        };
        if copy.is_none() && needs_copy(&r) {
            checks.push(pc(
                "model_missing",
                "block",
                format!(
                    "{display} is not in {label}'s model library: download or copy it to that machine first"
                ),
            ));
        }
        let model_ok = format_ok && (copy.is_some() || !needs_copy(&r));
        let mut info: Option<ModelInfo> = None;
        if model_ok && errs.is_empty() {
            match self.deps.fit.info_for(host_id, &r, None, true, false).await {
                Ok(i) => info = Some(i),
                Err(AppError::Problem(e)) => checks.push(pc(
                    "fit_unavailable",
                    "warn",
                    format!("no fit estimate: {}", e.detail),
                )),
                Err(e) => return Err(e),
            }
        }
        let gpu_ids = self.resolve_gpus(adapter.as_ref(), req, &p, info.as_ref(), &display);
        if let Some(info) = info.as_ref().filter(|_| errs.is_empty()) {
            let f = self
                .deps
                .fit
                .assess_info(&req.engine, info, &p, &gpu_ids, 1, host_id)?;
            let level = match f.verdict.as_str() {
                "fits" => "ok",
                "wont_fit" => "block",
                _ => "warn",
            };
            let mut msg = format!("fit verdict: {} ({} confidence)", f.verdict, f.confidence);
            if !f.fits_if_stop.is_empty() {
                msg.push_str(&format!("; fits if you stop {}", f.fits_if_stop.join(", ")));
            }
            checks.push(pc("fit", level, msg));
            for c in &f.compat {
                if c.level != "ok" {
                    checks.push(PreflightCheck {
                        code: c.code.clone(),
                        level: c.level.clone(),
                        message: c.message.clone(),
                    });
                }
            }
            fit = Some(f);
        }
        let trusted: Vec<String> = p
            .iter()
            .filter(|(k, v)| {
                k.contains("trust_remote")
                    && !matches!(v, Value::Null | Value::Bool(false))
                    && v.as_str() != Some("")
                    && v.as_str() != Some(MARKER)
                    && !v.as_f64().is_some_and(|x| x == 0.0)
            })
            .map(|(k, _)| k.clone())
            .collect();
        if !trusted.is_empty() {
            checks.push(pc(
                "trust_remote_code",
                "warn",
                format!(
                    "{} runs Python code from the model repo inside the container; it is off unless you set it \
                     explicitly",
                    trusted.join(", ")
                ),
            ));
        }
        let port = self.alloc_port(host_id, exclude_id)?;
        match port {
            None => {
                let (lo, hi) = nv::port_range(
                    node.as_ref(),
                    (self.cfg().port_range_start, self.cfg().port_range_end),
                );
                checks.push(pc(
                    "no_free_port",
                    "block",
                    format!("no free port in {lo}-{hi} on {label}"),
                ));
            }
            Some(port) => checks.push(pc("port", "ok", format!("host port {port}"))),
        }
        self.gpu_checks(
            &mut checks,
            node.as_ref(),
            host_id,
            &label,
            req,
            &gpu_ids,
            exclude_id,
        );
        let image = self.pick_image(
            adapter.as_ref(),
            node.as_ref(),
            host_id,
            req.image.as_deref(),
        )?;
        if nv::image_allowed(&image, &nv::engine_images(node.as_ref())) {
            let mut a = Obj::new();
            a.insert("image".into(), json!(image));
            match self
                .nodes
                .call(host_id, "docker.image_present", Some(a), 30.0)
                .await
            {
                Err(AppError::Problem(e)) => {
                    checks.push(pc("image_check_failed", "warn", e.detail));
                }
                Err(e) => return Err(e),
                Ok(present) => {
                    if !truthy(present.get("present")) {
                        checks.push(pc(
                            "image_missing",
                            "block",
                            format!(
                                "image {image} is not present on {label}: pull it on that machine (the console never \
                                 pulls)"
                            ),
                        ));
                    }
                }
            }
        } else {
            checks.push(pc(
                "image_not_allowed",
                "block",
                format!("image {image} is not allowed by {label}'s policy (engine_images)"),
            ));
        }
        if !truthy(nv::policy(node.as_ref()).get("gateway_image")) {
            checks.push(pc(
                "gateway_missing",
                "block",
                format!(
                    "{label}'s policy names no gateway_image, so engines there cannot get their loopback gateway"
                ),
            ));
        }
        let blocked_early = checks.iter().any(|c| {
            c.level == "block"
                && ["image_not_allowed", "no_gpu", "gpu_unknown"].contains(&c.code.as_str())
        });
        if model_ok && errs.is_empty() && !blocked_early {
            // the same build a real launch does: catches a secret on argv, a bad mount or an invalid spec
            let plan = Plan {
                iid: "preflight".into(),
                name: "preflight".into(),
                engine: req.engine.clone(),
                host_id: host_id.to_string(),
                r: r.clone(),
                params: p.clone(),
                gpu_ids: gpu_ids.clone(),
                port: port.unwrap_or(0),
                container: "ec-preflight".into(),
                image_req: req.image.clone(),
            };
            match self.build(&plan) {
                Ok(_) => {}
                // an adapter exception is not a check: it is the generic 500, as Python lets it propagate
                Err(AppError::Problem(e)) if !e.extra.contains_key("exception") => {
                    checks.push(PreflightCheck {
                        code: e.code.clone(),
                        level: "block".into(),
                        message: e.detail,
                    });
                }
                Err(e) => return Err(e),
            }
        }
        let ok = !checks.iter().any(|c| c.level == "block");
        Ok((
            PreflightReport {
                ok,
                gpu_ids,
                checks,
                fit,
            },
            port,
        ))
    }

    fn resolve_gpus(
        &self,
        adapter: &dyn EngineAdapter,
        req: &InstanceCreate,
        params: &Obj,
        info: Option<&ModelInfo>,
        display: &str,
    ) -> Vec<i64> {
        let fallback = ModelInfo {
            repo_id: display.to_string(),
            ..ModelInfo::default()
        };
        // an adapter's memory model may not cope with an unknown model: use the request
        self.deps
            .fit
            .resolve_gpus(
                adapter.id(),
                info.unwrap_or(&fallback),
                params,
                &req.gpu_ids,
                &req.host_id,
            )
            .unwrap_or_else(|_| req.gpu_ids.clone())
    }

    #[allow(clippy::too_many_arguments)]
    fn gpu_checks(
        &self,
        checks: &mut Vec<PreflightCheck>,
        node: &dyn NodeClient,
        host_id: &str,
        label: &str,
        req: &InstanceCreate,
        gpu_ids: &[i64],
        exclude_id: Option<&str>,
    ) {
        let mut have: Vec<i64> = nv::static_gpus(node)
            .iter()
            .filter_map(|g| g.get("index").and_then(nv::int_of))
            .collect();
        have.sort_unstable();
        have.dedup();
        if have.is_empty() {
            checks.push(pc("no_gpu", "block", format!("{label} reports no GPU")));
            return;
        }
        let missing: Vec<String> = req
            .gpu_ids
            .iter()
            .filter(|i| !have.contains(i))
            .map(ToString::to_string)
            .collect();
        if !missing.is_empty() {
            let has: Vec<String> = have.iter().map(ToString::to_string).collect();
            checks.push(pc(
                "gpu_unknown",
                "block",
                format!(
                    "{label} has no GPU {} (it has {})",
                    missing.join(", "),
                    has.join(", ")
                ),
            ));
            return;
        }
        let chosen: Vec<i64> = if gpu_ids.is_empty() {
            have
        } else {
            gpu_ids.to_vec()
        };
        let clash: Vec<String> = self
            .instances_on(host_id)
            .into_iter()
            .filter(|i| exclude_id != Some(i.id.as_str()) && RUNNING.contains(&i.state.as_str()))
            .filter(|i| {
                chosen
                    .iter()
                    .any(|g| i.gpu_ids.contains(g) || i.vram.iter().any(|v| v.gpu_index == *g))
            })
            .map(|i| i.name)
            .collect();
        if !clash.is_empty() {
            checks.push(pc(
                "gpu_shared",
                "warn",
                format!(
                    "GPU already used by {}; relying on free VRAM",
                    clash.join(", ")
                ),
            ));
        }
    }

    // -- commands --------------------------------------------------------------------------------------------------
    pub async fn create(
        &self,
        req: &InstanceCreate,
        profile_params: Option<&Obj>,
    ) -> AppResult<Instance> {
        let mut params = profile_params.cloned().unwrap_or_default();
        for (k, v) in &req.params {
            params.insert(k.clone(), v.clone());
        }
        let (public, secret_values) = split_secrets(&params);
        let lock = self.host_lock(&req.host_id);
        let guard = lock.lock().await;
        let (report, port) = Box::pin(self.preflight_full(req, None, Some(&params))).await?;
        let Some(port) = port.filter(|_| report.ok) else {
            return Err(conflict("preflight failed")
                .code("preflight_failed")
                .with(
                    "preflight",
                    serde_json::to_value(&report).unwrap_or(Value::Null),
                )
                .into());
        };
        let r = Self::ref_for(req);
        check_ref(&r)?;
        let iid = new_id("inst_");
        let display = display_of(&r);
        let name = req
            .name
            .clone()
            .filter(|n| !n.is_empty())
            .unwrap_or_else(|| safe_name(display.rsplit('/').next().unwrap_or(&display)));
        let tail: String = {
            let c: Vec<char> = iid.chars().collect();
            c[c.len().saturating_sub(6)..].iter().collect()
        };
        let plan = Plan {
            iid: iid.clone(),
            name: name.clone(),
            engine: req.engine.clone(),
            host_id: req.host_id.clone(),
            r: r.clone(),
            params: params.clone(),
            gpu_ids: report.gpu_ids.clone(),
            port,
            container: format!("ec-{}-{tail}", slug40(&name)),
            image_req: req.image.clone(),
        };
        let built = self.build(&plan)?;
        let fit_json = report
            .fit
            .as_ref()
            .map(|f| serde_json::to_string(f).unwrap_or_default());
        let model_json = serde_json::to_string(&r).unwrap_or_default();
        let created = self.now();
        let gpu_ids = crate::pyjson::dumps(&json!(built.gpu_ids));
        let gpu_uuids = crate::pyjson::dumps(&json!(built.gpu_uuids));
        let params_json = crate::pyjson::dumps(&Value::Object(public));
        self.deps.db.execute(
            "INSERT INTO instances(id,name,engine,repo_id,params,gpu_ids,gpu_uuids,port,container_name,image,state,\
             profile_id,ttl_idle_s,fit,created_at,host_id,model,image_req) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            &[
                &iid,
                &name,
                &req.engine,
                &display,
                &params_json,
                &gpu_ids,
                &gpu_uuids,
                &port,
                &plan.container,
                &built.image,
                &"starting",
                &req.profile_id,
                &req.ttl_idle_s,
                &fit_json,
                &created,
                &req.host_id,
                &model_json,
                &req.image,
            ],
        )?;
        self.deps
            .secrets
            .put(&iid, &secret_values)
            .map_err(|_| AppError::from(internal("unexpected server error")))?;
        Box::pin(self.launch(&plan, built)).await?;
        drop(guard);
        self.get(&iid)
    }

    /// Stop/remove are idempotent: a container that is already gone is the outcome we wanted.
    async fn call_ignoring_missing(
        &self,
        host_id: &str,
        op: &str,
        args: Obj,
        timeout: f64,
    ) -> AppResult<()> {
        match self.nodes.call(host_id, op, Some(args), timeout).await {
            Ok(_) => Ok(()),
            Err(e) if problem_of(&e).is_some_and(|p| p.code == "not_found") => Ok(()),
            Err(e) => Err(e),
        }
    }

    fn args(pairs: &[(&str, Value)]) -> Obj {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.clone()))
            .collect()
    }

    async fn remove_containers(&self, plan: &Plan) -> AppResult<()> {
        self.call_ignoring_missing(
            &plan.host_id,
            "docker.rm",
            Self::args(&[
                ("name", json!(gateway_name(&plan.container))),
                ("role", json!("gateway")),
                ("instance", json!(plan.iid)),
            ]),
            60.0,
        )
        .await?;
        self.call_ignoring_missing(
            &plan.host_id,
            "docker.rm",
            Self::args(&[
                ("name", json!(plan.container)),
                ("instance", json!(plan.iid)),
            ]),
            60.0,
        )
        .await
    }

    /// Start the loopback gateway; when its port is taken, try the next free one of the host's range.
    async fn gateway(&self, plan: &Plan, container_port: i64) -> AppResult<i64> {
        let mut order = vec![plan.port];
        order.extend(
            self.candidate_ports(&plan.host_id, Some(&plan.iid))?
                .into_iter()
                .filter(|p| *p != plan.port),
        );
        let mut last: Option<AppError> = None;
        for port in order.into_iter().take(GATEWAY_ATTEMPTS) {
            let a = Self::args(&[
                ("name", json!(gateway_name(&plan.container))),
                ("instance", json!(plan.iid)),
                ("host_port", json!(port)),
                ("engine_host", json!(plan.container)),
                ("engine_port", json!(container_port)),
            ]);
            match self
                .nodes
                .call(&plan.host_id, "gateway.start", Some(a), 60.0)
                .await
            {
                Ok(_) => return Ok(port),
                Err(e) if problem_of(&e).is_some_and(|p| p.code == "port_in_use") => last = Some(e),
                Err(e) => return Err(e),
            }
        }
        Err(last.unwrap_or_else(|| {
            Problem::new(409, "no_free_port", "no free port for the gateway").into()
        }))
    }

    async fn launch(&self, plan: &Plan, built: Built) -> AppResult<()> {
        let res: AppResult<()> = async {
            self.nodes
                .call(&plan.host_id, "net.ensure_internal", None, 30.0)
                .await?;
            // clear stale containers of the same name (the node verifies ownership)
            self.remove_containers(plan).await?;
            let mut spec = built.run.clone();
            spec.insert(
                "secret_env".into(),
                Value::Object(built.secret_env_obj(true)),
            );
            self.nodes
                .call(
                    &plan.host_id,
                    "docker.run",
                    Some(Self::args(&[("spec", Value::Object(spec))])),
                    180.0,
                )
                .await?;
            let port = match self.gateway(plan, built.launch.container_port).await {
                Ok(p) => p,
                Err(e) => {
                    if problem_of(&e).is_some() {
                        // never leave an engine running without its front door
                        let _ = self
                            .nodes
                            .call(
                                &plan.host_id,
                                "docker.rm",
                                Some(Self::args(&[
                                    ("name", json!(plan.container)),
                                    ("instance", json!(plan.iid)),
                                ])),
                                60.0,
                            )
                            .await;
                    }
                    return Err(e);
                }
            };
            self.cache(&plan.iid, built.clone());
            let t = self.now();
            self.set(
                &plan.iid,
                vec![
                    ("state", SqlValue::Text("starting".into())),
                    ("started_at", SqlValue::Real(t)),
                    ("error", SqlValue::Null),
                    ("phase", SqlValue::Null),
                    ("progress_pct", SqlValue::Null),
                    ("last_logs", SqlValue::Null),
                    (
                        "container_port",
                        SqlValue::Integer(built.launch.container_port),
                    ),
                    ("port", SqlValue::Integer(port)),
                ],
            )?;
            Ok(())
        }
        .await;
        if let Err(e) = &res
            && let Some(p) = problem_of(e)
        {
            let _ = self.set(
                &plan.iid,
                vec![
                    ("state", SqlValue::Text("failed".into())),
                    ("error", SqlValue::Text(p.detail.clone())),
                ],
            );
        }
        res
    }

    pub async fn start(&self, iid: &str) -> AppResult<Instance> {
        let inst = self.get(iid)?;
        if inst.management == "script" {
            return self.script_action(&inst, "up").await;
        }
        if inst.management == "external" {
            return self.external_action(&inst, "start").await;
        }
        if ["starting", "loading", "ready"].contains(&inst.state.as_str()) {
            return Err(conflict(format!("instance is already {}", inst.state)).into());
        }
        let lock = self.host_lock(&inst.host_id);
        let guard = lock.lock().await;
        // another request may have started it while this one waited for the host's lock
        let inst = self.get(iid)?;
        if ["starting", "loading", "ready"].contains(&inst.state.as_str()) {
            return Err(conflict(format!("instance is already {}", inst.state)).into());
        }
        let req = InstanceCreate {
            engine: inst.engine.clone(),
            repo_id: None,
            params: inst.params.clone(),
            gpu_ids: inst.gpu_ids.clone(), // a restart keeps its GPUs
            profile_id: None,
            name: None,
            ttl_idle_s: None,
            host_id: inst.host_id.clone(),
            model: inst.model.clone(),
            image: self.row_image_req(iid),
        };
        let real = self.real_params(&inst);
        let (report, port) = Box::pin(self.preflight_full(&req, Some(iid), Some(&real))).await?;
        let Some(port) = port.filter(|_| report.ok) else {
            return Err(conflict("preflight failed")
                .code("preflight_failed")
                .with(
                    "preflight",
                    serde_json::to_value(&report).unwrap_or(Value::Null),
                )
                .into());
        };
        let plan = self.plan_of(&inst, Some(port))?;
        let built = self.build(&plan)?;
        let fit_json = report.fit.as_ref().map_or(SqlValue::Null, |f| {
            SqlValue::Text(serde_json::to_string(f).unwrap_or_default())
        });
        self.set(
            iid,
            vec![
                ("port", SqlValue::Integer(port)),
                (
                    "gpu_uuids",
                    SqlValue::Text(crate::pyjson::dumps(&json!(built.gpu_uuids))),
                ),
                ("image", SqlValue::Text(built.image.clone())),
                ("state", SqlValue::Text("starting".into())),
                ("fit", fit_json),
            ],
        )?;
        Box::pin(self.launch(&plan, built)).await?;
        drop(guard);
        self.get(iid)
    }

    pub async fn stop(&self, iid: &str) -> AppResult<Instance> {
        let inst = self.get(iid)?;
        if inst.management == "script" {
            return self.script_action(&inst, "down").await;
        }
        if inst.management == "external" {
            return self.external_action(&inst, "stop").await;
        }
        if inst.state == "stopped" {
            return Ok(inst);
        }
        let before = self
            .row(iid)?
            .map_or_else(|| inst.state.clone(), |r| r.str_or("state", "stopped"));
        let cn = Self::cn(&inst)?;
        self.set(iid, vec![("state", SqlValue::Text("stopping".into()))])?;
        let res: AppResult<()> = async {
            self.call_ignoring_missing(
                &inst.host_id,
                "docker.stop",
                Self::args(&[
                    ("name", json!(gateway_name(&cn))),
                    ("role", json!("gateway")),
                    ("instance", json!(iid)),
                    ("timeout_s", json!(30)),
                ]),
                90.0,
            )
            .await?;
            self.call_ignoring_missing(
                &inst.host_id,
                "docker.stop",
                Self::args(&[
                    ("name", json!(cn)),
                    ("instance", json!(iid)),
                    ("timeout_s", json!(30)),
                ]),
                90.0,
            )
            .await
        }
        .await;
        if let Err(e) = res {
            if let Some(p) = problem_of(&e) {
                // a node that could not even be asked changed nothing: do not claim the engine failed
                let cols = if UNAVAILABLE_CODES.contains(&p.code.as_str()) {
                    vec![("state", SqlValue::Text(before))]
                } else {
                    vec![
                        ("state", SqlValue::Text("failed".into())),
                        ("error", SqlValue::Text(p.detail.clone())),
                    ]
                };
                let _ = self.set(iid, cols);
            }
            return Err(e);
        }
        self.set(
            iid,
            vec![
                ("state", SqlValue::Text("stopped".into())),
                ("phase", SqlValue::Null),
                ("progress_pct", SqlValue::Null),
            ],
        )?;
        self.get(iid)
    }

    pub async fn restart(&self, iid: &str) -> AppResult<Instance> {
        let inst = self.get(iid)?;
        if inst.management == "script" {
            let Some(scripts) = self.scripts_svc() else {
                return Err(conflict("scripts are not available")
                    .code("instance_not_managed")
                    .into());
            };
            if !inst.controls.restart {
                return Err(conflict(format!(
                    "cannot restart this engine: {}",
                    inst.controls
                        .reason
                        .clone()
                        .unwrap_or_else(|| "None".into())
                ))
                .code("instance_not_managed")
                .into());
            }
            scripts
                .restart(&inst.host_id, inst.script.as_deref().unwrap_or(""))
                .await?;
            return self.get(iid);
        }
        if inst.management == "external" {
            return Err(conflict(
                "an external engine can only be stopped or started, not restarted",
            )
            .code("instance_not_managed")
            .into());
        }
        Box::pin(self.stop(iid)).await?;
        if self.cfg().restart_settle_s > 0.0 {
            // the fit check reads the host's last pushed sample, which may still show the VRAM this engine just freed
            tokio::time::sleep(std::time::Duration::from_secs_f64(
                self.cfg().restart_settle_s,
            ))
            .await;
        }
        Box::pin(self.start(iid)).await
    }

    async fn script_action(&self, inst: &Instance, action: &str) -> AppResult<Instance> {
        let Some(scripts) = self.scripts_svc() else {
            return Err(conflict("scripts are not available")
                .code("instance_not_managed")
                .into());
        };
        let up = action == "up";
        if !(if up {
            inst.controls.start
        } else {
            inst.controls.stop
        }) {
            return Err(conflict(format!(
                "cannot {} this engine: {}",
                if up { "start" } else { "stop" },
                inst.controls
                    .reason
                    .clone()
                    .unwrap_or_else(|| "None".into())
            ))
            .code("instance_not_managed")
            .into());
        }
        scripts
            .run(
                &inst.host_id,
                inst.script.as_deref().unwrap_or(""),
                action,
                None,
            )
            .await?;
        self.get(&inst.id)
    }

    async fn external_action(&self, inst: &Instance, action: &str) -> AppResult<Instance> {
        let allowed = if action == "start" {
            inst.controls.start
        } else {
            inst.controls.stop
        };
        let Some(name) = inst
            .container_name
            .clone()
            .filter(|n| !n.is_empty() && allowed)
        else {
            return Err(conflict(format!(
                "cannot {action} this engine: {}",
                inst.controls
                    .reason
                    .clone()
                    .unwrap_or_else(|| "None".into())
            ))
            .code("instance_not_managed")
            .into());
        };
        self.nodes
            .call(
                &inst.host_id,
                "docker.external",
                Some(Self::args(&[
                    ("name", json!(name)),
                    ("action", json!(action)),
                ])),
                90.0,
            )
            .await?;
        if action == "stop" {
            let ctx = self.ctx(&inst.host_id, None)?;
            let stopped = Self::as_stopped(inst, &ctx);
            let mut d = self.derived.guard();
            oset(
                d.stopped_ext.entry(inst.host_id.clone()).or_default(),
                &inst.id,
                stopped.clone(),
            );
            if let Some(prev) = d.prev_ext.get_mut(&inst.host_id) {
                odel(prev, &inst.id);
            }
            return Ok(stopped);
        }
        {
            let mut d = self.derived.guard();
            if let Some(s) = d.stopped_ext.get_mut(&inst.host_id) {
                odel(s, &inst.id);
            }
            if let Some(g) = d.gone.get_mut(&inst.host_id) {
                odel(g, &inst.id);
            }
        }
        let mut started = inst.clone();
        started.state = "starting".into();
        started.state_reason = Some("start requested".into());
        Ok(started)
    }

    pub async fn remove(&self, iid: &str) -> AppResult<()> {
        let inst = self.require_managed(iid)?;
        let cn = Self::cn(&inst)?;
        self.call_ignoring_missing(
            &inst.host_id,
            "docker.rm",
            Self::args(&[
                ("name", json!(gateway_name(&cn))),
                ("role", json!("gateway")),
                ("instance", json!(iid)),
            ]),
            60.0,
        )
        .await?;
        self.call_ignoring_missing(
            &inst.host_id,
            "docker.rm",
            Self::args(&[("name", json!(cn)), ("instance", json!(iid))]),
            60.0,
        )
        .await?;
        self.deps.secrets.delete(iid);
        self.built.guard().remove(iid);
        self.deps
            .db
            .execute("DELETE FROM instances WHERE id=?", &[&iid])?;
        Ok(())
    }

    pub fn patch(&self, iid: &str, p: &InstancePatch) -> AppResult<Instance> {
        self.require_managed(iid)?;
        let mut cols: Vec<(&'static str, SqlValue)> = Vec::new();
        if let Some(pinned) = p.pinned {
            cols.push(("pinned", SqlValue::Integer(i64::from(pinned))));
        }
        if let Some(ttl) = p.ttl_idle_s {
            cols.push(("ttl_idle_s", SqlValue::Integer(ttl)));
        }
        if let Some(name) = &p.name {
            if name.trim().is_empty() {
                return Err(bad_request("name must not be empty").into());
            }
            cols.push(("name", SqlValue::Text(name.clone())));
        }
        if !cols.is_empty() {
            self.set(iid, cols)?;
        }
        self.get(iid)
    }

    // -- logs / command ----------------------------------------------------------------------------------------------
    /// The secret values a log line may echo (the instance's stored secrets and its launch's secret values).
    fn secret_values_of(&self, inst: &Instance) -> AppResult<Vec<String>> {
        let mut values: Vec<String> = self
            .deps
            .secrets
            .get(&inst.id)
            .into_iter()
            .map(|(_, v)| v)
            .collect();
        let built = match self.cached(&inst.id) {
            Some(b) => Some(b),
            // `contextlib.suppress(ProblemError, NodeError, KeyError, ValueError)`
            None => match self.plan_of(inst, None).and_then(|p| self.build(&p)) {
                Ok(b) => Some(b),
                Err(e) if caught_by_launch_for(&e) => None,
                Err(e) => return Err(e),
            },
        };
        if let Some(b) = built {
            values.extend(b.secret_values);
        }
        Ok(values)
    }

    fn scrub_with(values: &[String], text: &str) -> String {
        let mut out = text.to_string();
        for v in values {
            if v.chars().count() >= 6 {
                out = out.replace(v.as_str(), "[redacted]");
            }
        }
        scrub(&out)
    }

    /// Engines may echo their own args (api keys) into logs: mask known secret values, then token shapes.
    fn scrub_logs(&self, inst: &Instance, text: &str) -> AppResult<String> {
        Ok(Self::scrub_with(&self.secret_values_of(inst)?, text))
    }

    pub async fn logs(&self, iid: &str, tail: i64) -> AppResult<String> {
        let inst = self.require_managed(iid)?;
        let res = self
            .nodes
            .call(
                &inst.host_id,
                "docker.logs",
                Some(Self::args(&[
                    ("name", json!(Self::cn(&inst)?)),
                    ("tail", json!(tail.clamp(1, 5000))),
                ])),
                30.0,
            )
            .await?;
        let text = res
            .get("text")
            .filter(|v| ec_proto::sanitize::truthy(v))
            .map_or_else(String::new, |v| sval(Some(v)));
        self.scrub_logs(&inst, &text)
    }

    /// Live log lines (secrets masked); errors end the stream.
    pub async fn follow_logs(
        &self,
        iid: &str,
        tail: i64,
    ) -> AppResult<BoxStream<'static, AppResult<String>>> {
        let inst = self.require_managed(iid)?;
        let mut stream = self.nodes.open(
            &inst.host_id,
            "docker.logs_follow",
            Some(Self::args(&[
                ("name", json!(Self::cn(&inst)?)),
                ("tail", json!(tail.clamp(1, 5000))),
            ])),
        )?;
        let values = self.secret_values_of(&inst)?;
        let caller = self.nodes.clone();
        let s = async_stream::stream! {
            while let Some(item) = stream.next().await {
                if let StreamItem::Event(ev) = item
                    && let Some(Value::String(line)) = ev.get("line")
                {
                    yield Ok(Self::scrub_with(&values, line));
                }
            }
            match stream.result().await {
                Ok(_) => {}
                Err(e) => yield Err(AppError::from(caller.problem(&e))),
            }
            stream.cancel().await;
        };
        Ok(Box::pin(s))
    }

    /// "Copy as command": the node renders the exact `docker run` it would use; the compose file and the bare engine
    /// command are built here from the same spec. Secret values are never part of it: names only.
    pub async fn command(&self, iid: &str) -> AppResult<CommandSnippets> {
        let inst = self.require_managed(iid)?;
        let built = self.build(&self.plan_of(&inst, None)?)?;
        let mut spec = built.run.clone();
        spec.insert(
            "secret_env".into(),
            Value::Object(built.secret_env_obj(false)),
        );
        let rendered = self
            .nodes
            .call(
                &inst.host_id,
                "docker.render",
                Some(Self::args(&[("spec", Value::Object(spec.clone()))])),
                30.0,
            )
            .await?;
        let node = self.nodes.node(&inst.host_id)?;
        let roots: Vec<(String, String)> = nv::model_roots(node.as_ref())
            .into_iter()
            .filter(|(_, r)| truthy(r.get("path")))
            .map(|(id, r)| (id, sval(r.get("path"))))
            .collect();
        let argv: Vec<String> = match rendered.get("argv") {
            Some(Value::Array(a)) => a.iter().map(|x| sval(Some(x))).collect(),
            _ => Vec::new(),
        };
        let pol = nv::policy(node.as_ref());
        let network = pol
            .get("engine_network")
            .filter(|v| ec_proto::sanitize::truthy(v))
            .map_or_else(|| self.cfg().engine_network.clone(), |v| sval(Some(v)));
        Ok(command_snippets(&spec, Some(&argv), &roots, Some(&network)))
    }

    // -- supervisor ---------------------------------------------------------------------------------------------------
    /// Adopt each host's console-owned containers when its link comes up (and once at start-up).
    pub async fn sync_links(&self) {
        for h in self.deps.fleet.hosts() {
            let Ok(node) = self.deps.fleet.node(&h.id) else {
                continue;
            };
            let key = (nv::link_up(node.as_ref()), node.link().connected_since);
            let prev = self.links.guard().get(&h.id).copied();
            if key.0 && Some(key) != prev {
                let _ = self.adopt(Some(&h.id)).await;
            }
            self.links.guard().insert(h.id.clone(), key);
        }
    }

    /// One supervisor round (adoption per host as links come up, health/progress, idle TTL).
    pub async fn tick(&self) -> AppResult<()> {
        self.sync_links().await;
        for inst in self.all_active() {
            // one bad instance must not stop supervision of the rest
            if Box::pin(self.supervise(&inst)).await.is_err() {
                tracing::error!(target: "engine_console.services.lifecycle", "supervisor error for {}", inst.id);
            }
        }
        let _ = self.externals(None); // keeps the vanished-engine bookkeeping moving
        Ok(())
    }

    /// A container's facts: from the pushed inventory, asking the node only when it is not there and the inventory
    /// may simply not have caught up with something we just did.
    async fn container(
        &self,
        host_id: &str,
        name: &str,
        since: Option<f64>,
    ) -> AppResult<Option<Obj>> {
        let node = self.nodes.node(host_id)?;
        if let Some(c) = nv::container_named(node.as_ref(), name) {
            return Ok(Some(c));
        }
        let recent = since.is_some_and(|s| self.now() - s < INVENTORY_LAG_S);
        if !nv::inventory(node.as_ref()).is_empty() && !recent {
            return Ok(None);
        }
        match self
            .nodes
            .call(
                host_id,
                "docker.inspect",
                Some(Self::args(&[("name", json!(name))])),
                20.0,
            )
            .await
        {
            Ok(res) => Ok(match res.get("info") {
                Some(Value::Object(info)) if truthy(info.get("name")) => Some(info.clone()),
                _ => None,
            }),
            Err(e) if problem_of(&e).is_some_and(|p| p.code == "not_found") => Ok(None),
            Err(e) => Err(e),
        }
    }

    async fn logs_text(&self, host_id: &str, name: &str, tail: i64) -> String {
        match self
            .nodes
            .call(
                host_id,
                "docker.logs",
                Some(Self::args(&[("name", json!(name)), ("tail", json!(tail))])),
                30.0,
            )
            .await
        {
            Ok(r) => r
                .get("text")
                .filter(|v| ec_proto::sanitize::truthy(v))
                .map_or_else(String::new, |v| sval(Some(v))),
            Err(_) => String::new(),
        }
    }

    fn last_chars(s: &str, n: usize) -> Option<String> {
        let c: Vec<char> = s.chars().collect();
        let out: String = c[c.len().saturating_sub(n)..].iter().collect();
        (!out.is_empty()).then_some(out)
    }

    #[allow(clippy::too_many_lines)]
    async fn supervise(&self, inst: &Instance) -> AppResult<()> {
        if inst.state == "unreachable" || !self.deps.adapters.contains(&inst.engine) {
            return Ok(());
        }
        let Ok(node) = self.nodes.node(&inst.host_id) else {
            return Ok(());
        };
        if !nv::link_up(node.as_ref()) {
            return Ok(()); // cannot see that machine: keep the last known state, do not invent a failure
        }
        let cn = Self::cn(inst)?;
        if inst.state == "stopping" {
            // a stop whose request went away half way must not leave the instance stopping forever
            let gone = self.container(&inst.host_id, &cn, None).await?;
            if gone
                .as_ref()
                .is_none_or(|c| c.get("state").and_then(Value::as_str) != Some("running"))
            {
                self.set(
                    &inst.id,
                    vec![
                        ("state", SqlValue::Text("stopped".into())),
                        ("phase", SqlValue::Null),
                        ("progress_pct", SqlValue::Null),
                    ],
                )?;
            }
            return Ok(());
        }
        let info = self.container(&inst.host_id, &cn, inst.started_at).await?;
        let running = info
            .as_ref()
            .is_some_and(|c| c.get("state").and_then(Value::as_str) == Some("running"));
        if !running {
            let logs = if info.is_some() {
                self.logs_text(&inst.host_id, &cn, 200).await
            } else {
                String::new()
            };
            let error = match &info {
                Some(c) => format!("container exited with code {}", sval(c.get("exit_code"))),
                None => "container disappeared".into(),
            };
            let last = Self::last_chars(&self.scrub_logs(inst, &logs)?, 20000);
            self.set(
                &inst.id,
                vec![
                    ("state", SqlValue::Text("failed".into())),
                    ("error", SqlValue::Text(error)),
                    ("last_logs", last.map_or(SqlValue::Null, SqlValue::Text)),
                    ("phase", SqlValue::Null),
                ],
            )?;
            let _ = self
                .nodes
                .call(
                    &inst.host_id,
                    "docker.stop",
                    Some(Self::args(&[
                        ("name", json!(gateway_name(&cn))),
                        ("role", json!("gateway")),
                        ("instance", json!(inst.id)),
                    ])),
                    30.0,
                )
                .await;
            return Ok(());
        }
        let info = info.unwrap_or_default();
        if info.get("role").and_then(Value::as_str) == Some("engine") {
            // legacy (pre-gateway) containers publish their own port
            let gw = self
                .container(&inst.host_id, &gateway_name(&cn), inst.started_at)
                .await?;
            let ok = gw.as_ref().is_some_and(|g| {
                g.get("role").and_then(Value::as_str) == Some("gateway")
                    && g.get("instance").and_then(Value::as_str) == Some(inst.id.as_str())
                    && g.get("state").and_then(Value::as_str) == Some("running")
            });
            if !ok {
                self.set(
                    &inst.id,
                    vec![
                        ("state", SqlValue::Text("failed".into())),
                        ("phase", SqlValue::Null),
                        (
                            "error",
                            SqlValue::Text(
                                "gateway container missing or stopped; restart the instance".into(),
                            ),
                        ),
                    ],
                )?;
                return Ok(());
            }
        }
        let adapter = self.deps.adapters.get(&inst.engine)?;
        if inst.state == "starting" || inst.state == "loading" {
            Box::pin(self.progress(inst, adapter.as_ref())).await?;
        } else if inst.state == "ready" {
            let ttl = match inst.ttl_idle_s {
                Some(t) => t,
                None => self.deps.settings.idle_ttl_s()?,
            };
            let last = inst
                .last_request_at
                .filter(|t| *t != 0.0)
                .or(inst.started_at.filter(|t| *t != 0.0))
                .unwrap_or(0.0);
            #[allow(clippy::cast_precision_loss)]
            if ttl != 0 && !inst.pinned && self.now() - last > ttl as f64 {
                tracing::info!(target: "engine_console.services.lifecycle", "stopping idle instance {} after {}s", inst.id, ttl);
                Box::pin(self.stop(&inst.id)).await?;
            }
        }
        Ok(())
    }

    async fn progress(&self, inst: &Instance, adapter: &dyn EngineAdapter) -> AppResult<()> {
        let mut state = inst.state.clone();
        let (mut phase, mut pct) = (inst.phase.clone(), inst.progress_pct);
        let cn = Self::cn(inst)?;
        let text = self.logs_text(&inst.host_id, &cn, 40).await;
        for line in crate::services::common::py_splitlines(&text) {
            // an exception the parser raises ends this instance's round (Python: tick logs it and moves on)
            let parsed = try_parse_startup_log(adapter, line)
                .map_err(|e| AppError::from(internal(format!("parse_startup_log raised {e}"))))?;
            if let Some(parsed) = parsed.filter(|p| !p.is_empty()) {
                if let Some(ph) = parsed
                    .get("phase")
                    .filter(|v| ec_proto::sanitize::truthy(v))
                {
                    phase = Some(sval(Some(ph)));
                }
                match parsed.get("pct") {
                    None | Some(Value::Null) => {}
                    Some(Value::String(s)) => {
                        if let Some(v) = crate::services::discovery::py_float(s) {
                            pct = Some(v);
                        }
                    }
                    Some(v) => {
                        if let Some(x) = v.as_f64() {
                            pct = Some(x);
                        }
                    }
                }
            }
        }
        if phase
            .as_deref()
            .is_some_and(|p| !p.is_empty() && p != "ready")
            && state == "starting"
        {
            state = "loading".into();
        }
        if self.probe(inst).await? {
            state = "ready".into();
            phase = Some("ready".into());
            pct = Some(100.0);
        } else if let Some(s) = inst.started_at.filter(|s| *s != 0.0) {
            #[allow(clippy::cast_precision_loss)]
            if self.now() - s > self.cfg().startup_timeout_s as f64 {
                let logs = self.logs_text(&inst.host_id, &cn, 200).await;
                let last = Self::last_chars(&self.scrub_logs(inst, &logs)?, 20000);
                self.set(
                    &inst.id,
                    vec![
                        ("state", SqlValue::Text("failed".into())),
                        ("error", SqlValue::Text("startup timed out".into())),
                        ("last_logs", last.map_or(SqlValue::Null, SqlValue::Text)),
                    ],
                )?;
                return Ok(());
            }
        }
        if (state.as_str(), phase.as_deref(), pct)
            != (
                inst.state.as_str(),
                inst.phase.as_deref(),
                inst.progress_pct,
            )
        {
            self.set(
                &inst.id,
                vec![
                    ("state", SqlValue::Text(state)),
                    ("phase", phase.map_or(SqlValue::Null, SqlValue::Text)),
                    ("progress_pct", pct.map_or(SqlValue::Null, SqlValue::Real)),
                ],
            )?;
        }
        Ok(())
    }

    /// Readiness: GET the engine's health path through the host's node (the gateway's loopback port).
    async fn probe(&self, inst: &Instance) -> AppResult<bool> {
        let Some(port) = inst.port else {
            return Ok(false);
        };
        let path = self
            .launch_for_checked(inst)?
            .map_or_else(|| "/health".to_string(), |l| l.health_path);
        let opts = HttpOpts {
            timeout_s: Some(3.0),
            max_bytes: Some(64 * 1024),
            ..HttpOpts::default()
        };
        Ok(self
            .deps
            .http
            .request(&inst.host_id, port, "GET", &path, opts)
            .await
            .is_ok_and(|r| r.status == 200))
    }

    // -- adoption ------------------------------------------------------------------------------------------------------
    /// Re-attach to console-owned engine containers that outlived the console, on one host or every host.
    pub async fn adopt(&self, host_id: Option<&str>) -> AppResult<usize> {
        let ids: Vec<String> = match host_id {
            Some(h) => vec![h.to_string()],
            None => self.deps.fleet.hosts().into_iter().map(|h| h.id).collect(),
        };
        let mut n = 0;
        for h in ids {
            n += self.adopt_host(&h)?;
        }
        Ok(n)
    }

    #[allow(clippy::too_many_lines)]
    fn adopt_host(&self, host_id: &str) -> AppResult<usize> {
        let node = self.nodes.node(host_id)?;
        let mut n = 0;
        let (lo, hi) = nv::port_range(
            node.as_ref(),
            (self.cfg().port_range_start, self.cfg().port_range_end),
        );
        let idx = uuid_map(node.as_ref());
        for c in nv::containers(node.as_ref()) {
            if !truthy(c.get("managed")) {
                continue;
            }
            let lab: Vec<(String, String)> = match c.get("labels") {
                Some(Value::Object(l)) => {
                    l.iter().map(|(k, v)| (k.clone(), sval(Some(v)))).collect()
                }
                _ => Vec::new(),
            };
            let get = |k: &str| lab.iter().find(|(kk, _)| kk == k).map(|(_, v)| v.clone());
            if get("engine-console.role").as_deref() == Some("gateway")
                || c.get("role").and_then(Value::as_str) == Some("gateway")
            {
                continue; // gateways are only ever handled as part of their own instance
            }
            let cname = sval(c.get("name"));
            let running = c.get("state").and_then(Value::as_str) == Some("running");
            if let Some(row) = self.deps.db.one(
                "SELECT id, state FROM instances WHERE container_name=? AND host_id=?",
                &[&cname, &host_id],
            )? {
                let rid = row.str_or("id", "");
                if running {
                    // the supervisor promotes it to ready via the health probe
                    self.set(&rid, vec![("state", SqlValue::Text("loading".into()))])?;
                } else if ACTIVE.contains(&row.str_or("state", "").as_str())
                    && (matches!(c.get("exit_code"), None | Some(Value::Null))
                        || c.get("exit_code").and_then(nv::int_of) == Some(0))
                {
                    // a clean exit: someone stopped it while we were away
                    self.set(&rid, vec![("state", SqlValue::Text("stopped".into()))])?;
                }
                // any other exit is a crash: the supervisor reports it with the exit code and the engine's last logs
                continue;
            }
            let (Some(iid), Some(engine), Some(model)) = (
                get("engine-console.instance").filter(|s| !s.is_empty()),
                get("engine-console.engine").filter(|s| !s.is_empty()),
                get("engine-console.model").filter(|s| !s.is_empty()),
            ) else {
                continue;
            };
            if !self.deps.adapters.contains(&engine) {
                continue;
            }
            // labels are attacker-influenced strings: validate before they reach SQL rows, paths or UI.
            // Our ids are `inst_...`: anything else could shadow a derived instance (`ext-...`, `scr-...`).
            let name = get("engine-console.name").unwrap_or_else(|| cname.clone());
            let fmt = get("engine-console.format").unwrap_or_else(|| "hf".into());
            let valid = Self::ref_from_labels(&fmt, &model, &get).ok().filter(|_| {
                iid.starts_with("inst_")
                    && ID_RE.is_match(&iid)
                    && ID_RE.is_match(&engine)
                    && NAME_RE.is_match(&name)
            });
            let Some(rf) = valid else {
                tracing::warn!(target: "engine_console.services.lifecycle", "not adopting {}: invalid engine-console.* labels", cname);
                continue;
            };
            let port = Self::adopt_port(node.as_ref(), &cname, &iid, &get);
            let Some(port) = port.filter(|p| lo <= *p && *p <= hi) else {
                tracing::warn!(target: "engine_console.services.lifecycle", "not adopting {}: label port outside the console range", cname);
                continue;
            };
            let uuids: Vec<String> = get("engine-console.gpus")
                .unwrap_or_default()
                .split(',')
                .filter(|u| idx.iter().any(|(k, _)| k == u))
                .map(str::to_string)
                .collect();
            let mut gpu_ids: Vec<i64> = get("engine-console.gpu_ids")
                .unwrap_or_default()
                .split(',')
                .filter(|x| !x.is_empty() && x.chars().all(|ch| ch.is_ascii_digit()))
                .filter_map(|x| x.parse().ok())
                .collect();
            if gpu_ids.is_empty() {
                gpu_ids = uuids
                    .iter()
                    .filter_map(|u| idx.iter().find(|(k, _)| k == u).map(|(_, i)| *i))
                    .collect();
            }
            let image = c
                .get("image")
                .filter(|v| ec_proto::sanitize::truthy(v))
                .map_or_else(
                    || {
                        self.deps
                            .adapters
                            .get(&engine)
                            .map(|a| a.image_for(&nv::arch(node.as_ref())))
                            .unwrap_or_default()
                    },
                    |v| sval(Some(v)),
                );
            if self.row(&iid)?.is_some() {
                // a container claiming the id of an instance that already exists: not ours to adopt
                tracing::warn!(target: "engine_console.services.lifecycle", "not adopting {}: its instance id is already taken", cname);
                continue;
            }
            let t = self.now();
            let started: Option<f64> = running.then_some(t);
            let state = if running { "loading" } else { "stopped" };
            let ins = self.deps.db.execute(
                "INSERT INTO instances(id,name,engine,repo_id,params,gpu_ids,gpu_uuids,port,container_name,image,state,\
                 created_at,started_at,host_id,model) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                &[
                    &iid,
                    &name,
                    &engine,
                    &model,
                    &"{}",
                    &crate::pyjson::dumps(&json!(gpu_ids)),
                    &crate::pyjson::dumps(&json!(uuids)),
                    &port,
                    &cname,
                    &image,
                    &state,
                    &t,
                    &started,
                    &host_id,
                    &serde_json::to_string(&rf).unwrap_or_default(),
                ],
            );
            if ins.is_err() {
                tracing::warn!(target: "engine_console.services.lifecycle", "not adopting {}: its instance id is already taken", cname);
                continue;
            }
            n += 1;
        }
        Ok(n)
    }

    fn ref_from_labels(
        fmt: &str,
        model: &str,
        get: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ModelRef, &'static str> {
        if fmt == "hf" {
            check_repo_id(model)?;
            return Ok(ModelRef::hf(model));
        }
        if !LABEL_MODEL.is_match(model) {
            return Err("bad model label");
        }
        if fmt == "ollama" {
            return Ok(ModelRef {
                format: "ollama".into(),
                repo_id: None,
                root_id: None,
                rel: None,
                name: Some(model.to_string()),
            });
        }
        let (root, rel) = (get("engine-console.root"), get("engine-console.rel"));
        if (fmt == "gguf" || fmt == "plain")
            && let Some(root) = root.filter(|r| !r.is_empty() && ROOT_RE.is_match(r))
            && rel.as_deref().is_none_or(|r| r.is_empty() || rel_ok(r))
        {
            return Ok(ModelRef {
                format: fmt.to_string(),
                repo_id: None,
                root_id: Some(root),
                rel,
                name: None,
            });
        }
        Err("bad model location labels")
    }

    /// The gateway's real published port (a retry may have moved it off the label's), else the label's.
    fn adopt_port(
        node: &dyn NodeClient,
        cname: &str,
        iid: &str,
        get: &dyn Fn(&str) -> Option<String>,
    ) -> Option<i64> {
        if let Some(gw) = nv::container_named(node, &gateway_name(cname)) {
            let owner = gw.get("instance");
            let ours = matches!(owner, None | Some(Value::Null))
                || owner.and_then(Value::as_str) == Some(iid);
            if ours && let Some(p) = nv::host_ports(&gw).first() {
                return Some(*p);
            }
        }
        crate::services::store::py_int(&get("engine-console.port").unwrap_or_default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_named_volumes_follow_the_adapter() {
        // the vllm compile caches are named volumes the node accepts (ec-<name>)
        assert!(
            NAMED_VOLUMES
                .iter()
                .all(|(_, v)| v.iter().all(|(n, _)| n.starts_with("ec-")))
        );
        assert_eq!(NAMED_VOLUMES[0].1[0].0, "ec-vllm-cache");
    }

    #[test]
    fn test_safe_names_and_container_slugs() {
        assert_eq!(safe_name("Qwen3-32B"), "Qwen3-32B");
        assert_eq!(safe_name("  ..weird//name!!  "), "weird-name");
        assert_eq!(safe_name("!!!"), "model");
        assert_eq!(safe_name(&"a".repeat(80)).len(), 64);
        assert_eq!(slug40("My Model.v2"), "my-model-v2");
        assert_eq!(slug40("___"), "model");
        assert_eq!(slug40(&"x".repeat(50)).len(), 40);
    }

    fn container(params: Value, label: Option<&str>) -> Obj {
        let mut c = Obj::new();
        c.insert("params".into(), params);
        if let Some(l) = label {
            c.insert("labels".into(), json!({"engine-console.model": l}));
        }
        c
    }

    #[test]
    fn test_loads_model_tells_which_model_a_shared_mount_serves() {
        let target = "/data/hf/hub/models--Other--Small";
        assert!(loads_model(
            &container(json!({"model": "Other/Small"}), None),
            target
        ));
        assert!(loads_model(
            &container(
                json!({"model_path": "/root/.cache/huggingface/hub/models--Other--Small/snapshots/abc"}),
                None
            ),
            target
        ));
        assert!(!loads_model(
            &container(json!({"model": "Someone/Else"}), None),
            target
        ));
        assert!(loads_model(&container(json!({}), None), target)); // cannot tell: cautious
        assert!(loads_model(
            &container(json!({}), Some("other/small")),
            target
        ));
        // a plain folder mounted elsewhere in the container
        assert!(loads_model(
            &container(json!({"model": "/models/my-model"}), None),
            "/data/plain/my-model"
        ));
        assert!(!loads_model(
            &container(json!({"model": "/models/other"}), None),
            "/data/plain/my-model"
        ));
    }
}
