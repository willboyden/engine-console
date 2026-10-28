//! Offline fakes for the data side (port of backend/tests/data/fakes.py, conftest.py `World` and helpers.py).
//!
//! `DataNode` implements `NodeClient` closely enough that the library, downloads and copies run end to end with no
//! docker, no ssh and no network: every request's arguments go through `ec_proto::clean_args` and must come out
//! unchanged, results and pushed messages are rebuilt with the protocol's cleaners, monitor mode refuses every op
//! outside MONITOR_OPS, and `fs.*` does real file work in a temp directory following docs/FLEET-PROTOCOL.md (path
//! rules, writable roots only, a `.incomplete` partial resumed at its own size, size and sha256 checked at `end`,
//! relative symlinks inside the root, `allow_delete`, `in_use`). `DataHub` is an in-memory Hugging Face hub.
//!
//! Python tests assign attributes and monkeypatch methods; here the state is behind `node.st()` and the hooks
//! (`on`, `read_gate`, `read_error`, `model_override`, `fail_mid`) replace the patched methods.

#![allow(
    dead_code,
    unreachable_pub,
    clippy::needless_pass_by_value,
    clippy::expect_used,
    clippy::panic,
    clippy::missing_panics_doc,
    clippy::type_complexity,
    clippy::too_many_arguments,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use std::collections::{HashMap, HashSet};
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use async_trait::async_trait;
use base64::Engine as _;
use bytes::Bytes;
use ec_hub::adapters::AdapterRegistry;
use ec_hub::adapters::base::{
    Compat, EngineAdapter, FloatMap, Hardware, LaunchSpec, ModelInfo, Obj, ParamSpec,
};
use ec_hub::config::Settings;
use ec_hub::domain::errors::{AppResult, NodeError, Problem, gated_model, not_found};
use ec_hub::domain::fleet::{
    FleetPort, HostConfig, LinkStatus, NodeClient, NodeStream, StreamItem,
};
use ec_hub::domain::ports::{ByteStream, HubClient, HubModel, RepoFile};
use ec_hub::services::common::EventBus;
use ec_hub::services::downloads::DownloadService;
use ec_hub::services::hf::{HfService, gguf_quant};
use ec_hub::services::library::LibraryService;
use ec_hub::services::settings::SettingsService;
use ec_hub::services::store::Store;
use serde_json::{Value, json};
use sha2::Digest as _;

pub const COMMIT: &str = "c0ffee00000000000000000000000000000000ff";
pub const LFS_SUFFIXES: &[&str] = &[".safetensors", ".gguf", ".bin"];
pub const PAYLOAD: &[u8] = b"01234567890123456789012345678901234567890123456789";
pub const TOTAL: i64 = 2 + 50 + 1;

pub fn sha_hex(b: &[u8]) -> String {
    hex::encode(sha2::Sha256::digest(b))
}

/// sha256 of PAYLOAD
pub fn sha() -> String {
    sha_hex(PAYLOAD)
}

/// helpers.py FILES (insertion order kept)
pub fn files() -> Vec<(String, Vec<u8>)> {
    vec![
        ("config.json".into(), b"{}".to_vec()),
        ("w.safetensors".into(), PAYLOAD.to_vec()),
        ("notes.txt".into(), b"n".to_vec()),
    ]
}

pub fn obj(v: Value) -> Obj {
    v.as_object().cloned().unwrap_or_default()
}

// ---- adapters ---------------------------------------------------------------------------------------------------
/// Just enough of an engine for the library to ask which formats it loads.
pub struct StubAdapter {
    pub id: String,
    pub formats: Vec<String>,
    image: String,
    /// what memory_model() answers (empty: the estimator's defaults)
    pub mem: FloatMap,
}

impl StubAdapter {
    pub fn new(id: &str, formats: &[&str]) -> Self {
        Self {
            id: id.into(),
            formats: formats.iter().map(|s| (*s).to_string()).collect(),
            image: format!("fake/{id}:1"),
            mem: Vec::new(),
        }
    }

    /// The same stub with this memory model (e.g. a tiny `mem_fraction`: nothing fits).
    #[must_use]
    pub fn with_mem(mut self, mem: &[(&str, f64)]) -> Self {
        self.mem = mem.iter().map(|(k, v)| ((*k).to_string(), *v)).collect();
        self
    }
}

impl EngineAdapter for StubAdapter {
    fn id(&self) -> &str {
        &self.id
    }
    fn display_name(&self) -> &str {
        &self.id
    }
    fn default_image(&self) -> &str {
        &self.image
    }
    fn model_formats(&self) -> Vec<String> {
        self.formats.clone()
    }
    fn param_catalog(&self) -> Vec<ParamSpec> {
        Vec::new()
    }
    fn presets(&self) -> Vec<(String, Obj)> {
        Vec::new()
    }
    fn validate(&self, _params: &Obj) -> Vec<String> {
        Vec::new()
    }
    fn build_launch(
        &self,
        _m: &str,
        _p: &Obj,
        _hw: &Hardware,
        _s: &str,
        _h: &str,
    ) -> Result<LaunchSpec, Problem> {
        Err(Problem::new(500, "internal_error", "NotImplementedError"))
    }
    fn compatibility(&self, _m: &ModelInfo, _p: &Obj, _hw: &Hardware) -> Vec<Compat> {
        Vec::new()
    }
    fn memory_model(&self, _m: &ModelInfo, _p: &Obj) -> FloatMap {
        self.mem.clone()
    }
    fn parse_metrics(&self, _t: &str) -> FloatMap {
        Vec::new()
    }
    fn parse_startup_log(&self, _l: &str) -> Option<Obj> {
        None
    }
}

/// conftest.py `make_adapters`
pub fn make_adapters() -> AdapterRegistry {
    AdapterRegistry::new([
        Arc::new(StubAdapter::new("vllm", &["hf", "gguf"])) as Arc<dyn EngineAdapter>,
        Arc::new(StubAdapter::new("llamacpp", &["gguf", "plain"])),
        Arc::new(StubAdapter::new("ollama", &["ollama"])),
    ])
}

// ---- the node --------------------------------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct RootCfg {
    pub id: String,
    pub kind: String,
    pub writable: bool,
    /// None: <node dir>/<id>, created
    pub path: Option<PathBuf>,
    pub label: Option<String>,
}

impl RootCfg {
    pub fn new(id: &str, kind: &str) -> Self {
        Self {
            id: id.into(),
            kind: kind.into(),
            writable: true,
            path: None,
            label: None,
        }
    }
    pub fn ro(mut self) -> Self {
        self.writable = false;
        self
    }
    pub fn at(mut self, p: impl Into<PathBuf>) -> Self {
        self.path = Some(p.into());
        self
    }
}

/// `mount_dict(...)` keyword arguments with Python's defaults.
#[derive(Debug, Clone)]
pub struct MountOpts {
    pub fstype: Option<String>,
    pub source: Option<String>,
    pub model: Option<String>,
    pub size: i64,
    pub used: i64,
    pub free: i64,
    pub remote: bool,
    pub remote_host: Option<String>,
    pub remote_path: Option<String>,
    pub roots: Vec<String>,
    pub readonly: bool,
}

impl Default for MountOpts {
    fn default() -> Self {
        Self {
            fstype: Some("ext4".into()),
            source: None,
            model: Some("Test SSD".into()),
            size: 1_000_000_000_000,
            used: 100_000_000_000,
            free: 900_000_000_000,
            remote: false,
            remote_host: None,
            remote_path: None,
            roots: Vec::new(),
            readonly: false,
        }
    }
}

pub fn mount_dict(mount: &str, o: MountOpts) -> Value {
    json!({"mount": mount, "fstype": o.fstype, "source": o.source, "device": if o.remote { Value::Null } else { json!("sda1") },
           "model": o.model, "size": o.size, "used": o.used, "free": o.free, "readonly": o.readonly, "remote": o.remote,
           "remote_host": o.remote_host, "remote_path": o.remote_path, "roots": o.roots})
}

pub fn mount_with_roots(mount: &str, roots: &[&str]) -> Value {
    mount_dict(
        mount,
        MountOpts {
            roots: roots.iter().map(|s| (*s).to_string()).collect(),
            ..MountOpts::default()
        },
    )
}

pub type DOpFn = Arc<dyn Fn(&DataNode, &Obj) -> Result<Obj, NodeError> + Send + Sync>;

/// `FakeNode(...)` keyword arguments.
#[derive(Debug, Clone)]
pub struct NodeOpts {
    pub label: Option<String>,
    pub roots: Vec<RootCfg>,
    pub mode: String,
    pub allow_delete: bool,
    pub addresses: Vec<String>,
    pub hostname: Option<String>,
    pub mounts: Option<Vec<Value>>,
    pub state: String,
    pub scan_disk: bool,
    pub ssh: Option<String>,
    /// GPUs the hello reports (96 GiB each, compute capability 12.0; none by default)
    pub gpus: usize,
}

impl Default for NodeOpts {
    fn default() -> Self {
        Self {
            label: None,
            roots: Vec::new(),
            mode: "manage".into(),
            allow_delete: true,
            addresses: Vec::new(),
            hostname: None,
            mounts: None,
            state: "up".into(),
            scan_disk: true,
            ssh: None,
            gpus: 0,
        }
    }
}

/// Everything a test may read or change on a node (Python: its attributes).
pub struct DState {
    pub roots: Vec<RootCfg>,
    pub mode: String,
    pub allow_delete: bool,
    pub addresses: Vec<String>,
    pub hostname: String,
    pub mounts: Option<Vec<Value>>,
    pub state: String,
    pub scan_disk_enabled: bool,
    pub gpus: usize,
    pub forgotten: bool,
    pub free: i64,
    pub entries: Vec<Value>,
    pub generation: i64,
    pub scans: Vec<Obj>,
    pub configs: HashMap<(String, String), Value>,
    pub calls: Vec<(String, Obj)>,
    pub writes: Vec<Value>,
    pub reads: Vec<Value>,
    pub fail_after_bytes: Option<i64>,
    pub space_left: Option<i64>,
    pub in_use_paths: HashSet<String>,
    pub symlink_conflicts: bool,
    pub read_chunk: usize,
    pub list_generation_bump_once: bool,
    pub overrides: HashMap<String, DOpFn>,
    /// fs.read streams fail at their first await with this error (Python's DeadStream)
    pub read_error: Option<NodeError>,
    /// fs.read streams hold every item after the first `n` until the gate opens (Python's Slow)
    pub read_gate: Option<(usize, Gate)>,
}

pub struct DataNode {
    pub host: HostConfig,
    pub base: PathBuf,
    me: std::sync::Weak<DataNode>,
    bus: Mutex<Option<Arc<EventBus>>>,
    state: Mutex<DState>,
}

impl std::fmt::Debug for DataNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("DataNode")
    }
}

fn title(s: &str) -> String {
    // str.title() for a host id: the first letter of each alphanumeric run upper-cased
    let mut out = String::new();
    let mut prev_alpha = false;
    for c in s.chars() {
        if c.is_alphabetic() {
            if prev_alpha {
                out.extend(c.to_lowercase());
            } else {
                out.extend(c.to_uppercase());
            }
            prev_alpha = true;
        } else {
            out.push(c);
            prev_alpha = false;
        }
    }
    out
}

impl DataNode {
    pub fn new(base: &Path, host_id: &str, o: NodeOpts) -> Arc<Self> {
        let host = HostConfig {
            id: host_id.into(),
            label: o.label.clone().unwrap_or_else(|| title(host_id)),
            ssh: o.ssh.clone(),
            port: None,
            ssh_config: false,
            enabled: true,
        };
        let base = base.join(host_id);
        std::fs::create_dir_all(&base).expect("node dir");
        let mut roots = Vec::new();
        for mut r in o.roots {
            if r.path.is_none() {
                let p = base.join(&r.id);
                std::fs::create_dir_all(&p).expect("root dir");
                r.path = Some(p);
            }
            roots.push(r);
        }
        let node = Arc::new_cyclic(|me| Self {
            host,
            base,
            me: me.clone(),
            bus: Mutex::new(None),
            state: Mutex::new(DState {
                roots,
                mode: o.mode,
                allow_delete: o.allow_delete,
                addresses: o.addresses,
                hostname: o.hostname.unwrap_or_else(|| host_id.to_string()),
                mounts: o.mounts,
                state: o.state,
                scan_disk_enabled: o.scan_disk,
                gpus: o.gpus,
                forgotten: false,
                free: 1_000_000_000_000,
                entries: Vec::new(),
                generation: 1,
                scans: Vec::new(),
                configs: HashMap::new(),
                calls: Vec::new(),
                writes: Vec::new(),
                reads: Vec::new(),
                fail_after_bytes: None,
                space_left: None,
                in_use_paths: HashSet::new(),
                symlink_conflicts: false,
                read_chunk: 1024,
                list_generation_bump_once: false,
                overrides: HashMap::new(),
                read_error: None,
                read_gate: None,
            }),
        });
        if o.scan_disk {
            node.scan();
        }
        node
    }

    pub fn st(&self) -> MutexGuard<'_, DState> {
        self.state.lock().expect("node lock")
    }

    pub fn set_bus(&self, bus: Arc<EventBus>) {
        *self.bus.lock().expect("bus") = Some(bus);
    }

    /// A root's directory.
    pub fn root_path(&self, id: &str) -> PathBuf {
        self.st()
            .roots
            .iter()
            .find(|r| r.id == id)
            .and_then(|r| r.path.clone())
            .expect("root")
    }

    pub fn add_root(&self, r: RootCfg) {
        self.st().roots.push(r);
    }

    pub fn set_entries(&self, entries: Vec<Value>) {
        let mut st = self.st();
        st.entries = entries;
        st.generation += 1;
    }

    pub fn ops(&self, name: &str) -> Vec<Obj> {
        self.st()
            .calls
            .iter()
            .filter(|(o, _)| o == name)
            .map(|(_, a)| a.clone())
            .collect()
    }

    pub fn calls(&self) -> Vec<(String, Obj)> {
        self.st().calls.clone()
    }

    pub fn writes(&self) -> Vec<Value> {
        self.st().writes.clone()
    }

    pub fn reads(&self) -> Vec<Value> {
        self.st().reads.clone()
    }

    pub fn scans(&self) -> Vec<Obj> {
        self.st().scans.clone()
    }

    pub fn generation(&self) -> i64 {
        self.st().generation
    }

    /// Replace (or add) an op's behaviour (Python monkeypatches `_op_*`).
    pub fn on(&self, op: &str, f: DOpFn) {
        self.st().overrides.insert(op.into(), f);
    }

    /// What the hub knows of a machine it has not heard from since it started: no hello, no disks.
    pub fn forget(&self) {
        self.st().forgotten = true;
    }

    pub fn announce_models_changed(&self) {
        let generation = self.generation();
        if let Some(bus) = self.bus.lock().expect("bus").clone() {
            bus.publish(
                "node.models_changed",
                json!({"host_id": self.host.id, "msg": {"type": "models_changed", "generation": generation}}),
            );
        }
    }

    pub fn hello(&self) -> Obj {
        let st = self.st();
        let roots: Vec<Value> = st
            .roots
            .iter()
            .map(|r| {
                json!({"id": r.id, "path": r.path.as_ref().map(|p| p.display().to_string()), "kind": r.kind,
                       "writable": r.writable, "label": r.label.clone().unwrap_or_else(|| r.id.clone())})
            })
            .collect();
        let raw = json!({
            "type": "hello", "proto": 1, "version": "0.4.0", "node_id": "0123456789abcdef",
            "host": {"hostname": st.hostname, "arch": "x86_64", "os": "Linux", "kernel": "7", "cpu_model": "cpu",
                     "cpu_cores": 8, "cpu_threads": 16, "ram_total_bytes": 64_i64 * (1 << 30), "boot_time": 1.0, "python": "3.12"},
            "gpu": {"driver": "1", "cuda": "13", "source": "none"}, "gpus": (0..st.gpus).map(|i| json!({
                "index": i, "uuid": format!("GPU-{}-{i}", st.hostname), "name": "Test GPU",
                "total_bytes": 96_i64 * (1 << 30), "compute_capability": "12.0", "unified_memory": false}))
                .collect::<Vec<_>>(),
            "docker": {"available": false},
            "addresses": st.addresses.iter().map(|a| json!({"iface": "eth0", "addr": a, "prefix": 24})).collect::<Vec<_>>(),
            "policy": {"path": "/etc/ecnode/policy.json", "mode": st.mode, "allow_delete": st.allow_delete,
                       "model_roots": roots},
            "forced_cmd": true, "via_ssh": false});
        ec_proto::sanitize_push(raw.as_object().expect("obj"))
            .expect("hello")
            .expect("known type")
    }

    /// A live sample: every GPU idle (all of its memory free) and 64 GiB of RAM, mostly available.
    pub fn sample(&self) -> Obj {
        let st = self.st();
        let total = 96_i64 * (1 << 30);
        let gpus: Obj = (0..st.gpus)
            .map(|i| {
                (
                    format!("GPU-{}-{i}", st.hostname),
                    json!({"util_pct": 0.0, "mem_used_bytes": 0, "mem_total_bytes": total, "temp_c": 40.0,
                           "power_w": 50.0, "power_limit_w": 300.0, "fan_pct": 30.0, "procs": []}),
                )
            })
            .collect();
        let gib = 1_i64 << 30;
        let raw = json!({"type": "sample", "seq": 1, "t": 1.0, "gpus": gpus,
                         "mem": {"total": 64 * gib, "available": 60 * gib, "free": 50 * gib, "cached": 8 * gib,
                                 "shmem": 0, "swap_total": 0, "swap_free": 0},
                         "cpu": {"load1": 0.5, "util_pct": 3.0}, "containers": {}});
        ec_proto::sanitize_push(raw.as_object().expect("obj"))
            .expect("sample")
            .expect("known type")
    }

    pub fn disks(&self) -> Obj {
        let st = self.st();
        let mounts: Vec<Value> = match &st.mounts {
            None => {
                let mut ids: Vec<String> = st.roots.iter().map(|r| r.id.clone()).collect();
                ids.sort();
                vec![mount_dict(
                    "/",
                    MountOpts {
                        free: st.free,
                        roots: ids,
                        ..MountOpts::default()
                    },
                )]
            }
            Some(m) => {
                if st.free == 1_000_000_000_000 {
                    m.clone()
                } else {
                    m.iter()
                        .map(|x| {
                            let mut x = x.clone();
                            if x["mount"] == "/" {
                                x["free"] = json!(st.free);
                            }
                            x
                        })
                        .collect()
                }
            }
        };
        let raw = json!({"type": "disks", "t": 1.0, "mounts": mounts});
        ec_proto::sanitize_push(raw.as_object().expect("obj"))
            .expect("disks")
            .expect("known type")
    }

    pub fn set_mounts(&self, mounts: Vec<Value>) {
        self.st().mounts = Some(mounts);
    }

    fn err(&self, code: &str, message: impl Into<String>) -> NodeError {
        NodeError::new(&self.host.id, code, message)
    }

    fn gate(&self, op: &str, args: &Obj) -> Result<(), NodeError> {
        if !ec_proto::specs::is_op(op) {
            return Err(self.err(
                "unknown_op",
                format!("'{op}' is not an operation of the node protocol"),
            ));
        }
        let (state, mode) = {
            let st = self.st();
            (st.state.clone(), st.mode.clone())
        };
        if state != "up" {
            return Err(NodeError::unavailable(
                &self.host.id,
                format!("{} is not connected ({state})", self.host.label),
            ));
        }
        let cleaned = ec_proto::clean_args(op, &Value::Object(args.clone()))
            .map_err(|e| self.err("unknown_op", e.to_string()))?;
        let seen: Obj = cleaned
            .iter()
            .filter(|(_, v)| !v.is_null())
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        assert_eq!(
            Value::Object(seen),
            Value::Object(args.clone()),
            "{op}: the node would drop or change arguments"
        );
        let missing = ec_proto::missing_args(op, &cleaned);
        if !missing.is_empty() {
            return Err(self.err("bad_args", format!("{op}: missing {missing:?}")));
        }
        if mode != "manage" && !ec_proto::specs::is_monitor_op(op) {
            return Err(self.err(
                "monitor_only",
                format!("{op} needs the node policy to be in manage mode"),
            ));
        }
        Ok(())
    }

    // ---- paths ------------------------------------------------------------------------------------------------
    fn resolve(&self, root_id: &Value, rel: &Value, write: bool) -> Result<PathBuf, NodeError> {
        let root = root_id
            .as_str()
            .and_then(|id| self.st().roots.iter().find(|r| r.id == id).cloned());
        let Some(root) = root.filter(|r| r.path.is_some()) else {
            return Err(self.err("not_found", format!("no model root {root_id}")));
        };
        let rpath = root.path.clone().expect("path");
        // Roots given as literal paths exist only to test library arithmetic: file work must stay in the temp dir.
        assert!(
            rpath.starts_with(&self.base),
            "test would touch a real directory: {}",
            rpath.display()
        );
        if write && !root.writable {
            return Err(self.err("policy_denied", format!("root '{}' is read-only", root.id)));
        }
        let Some(rel) = rel.as_str().filter(|r| {
            !r.is_empty()
                && !r.starts_with('/')
                && !r.contains('\\')
                && !r.contains('\0')
                && r.split('/').all(|s| !matches!(s, "" | "." | ".."))
        }) else {
            return Err(self.err("bad_args", format!("bad path {rel}")));
        };
        let path = rpath.join(rel);
        let parent = realpath(path.parent().expect("parent"));
        let base = realpath(&rpath);
        if parent != base && !parent.starts_with(&base) {
            return Err(self.err("policy_denied", "the path leaves the root"));
        }
        Ok(path)
    }

    // ---- operations -------------------------------------------------------------------------------------------
    pub fn op_fs_stat(&self, a: &Obj) -> Result<Obj, NodeError> {
        let mut items = Vec::new();
        for rel in a["paths"].as_array().cloned().unwrap_or_default() {
            let p = self.resolve(&a["root"], &rel, false)?;
            let part = with_suffix(&p, ".incomplete");
            let exists = p.exists();
            items.push(json!({"rel": rel, "exists": exists, "is_dir": p.is_dir(),
                              "size": if exists && p.is_file() { file_size(&p) } else { 0 },
                              "partial": if part.is_file() { file_size(&part) } else { 0 }}));
        }
        Ok(obj(json!({"items": items})))
    }

    pub fn op_fs_symlink(&self, a: &Obj) -> Result<Obj, NodeError> {
        let link = self.resolve(&a["root"], &a["link"], true)?;
        let target = a["target"].as_str().unwrap_or("");
        if target.starts_with('/') {
            return Err(self.err("bad_args", "the target must be relative"));
        }
        let resolved = realpath(&link.parent().expect("parent").join(target));
        let base = realpath(&self.root_path(a["root"].as_str().unwrap_or("")));
        if resolved != base && !resolved.starts_with(&base) {
            return Err(self.err("policy_denied", "the link target leaves the root"));
        }
        std::fs::create_dir_all(link.parent().expect("parent")).expect("mkdir");
        if link.symlink_metadata().is_ok() {
            if self.st().symlink_conflicts {
                return Err(self.err("conflict", "the link exists"));
            }
            std::fs::remove_file(&link).expect("unlink");
        }
        std::os::unix::fs::symlink(target, &link).expect("symlink");
        Ok(Obj::new())
    }

    pub fn op_fs_put_small(&self, a: &Obj) -> Result<Obj, NodeError> {
        let path = self.resolve(&a["root"], &a["rel"], true)?;
        let data = base64::engine::general_purpose::STANDARD
            .decode(a["content_b64"].as_str().unwrap_or(""))
            .expect("b64");
        std::fs::create_dir_all(path.parent().expect("parent")).expect("mkdir");
        let tmp = with_suffix(&path, ".tmp");
        std::fs::write(&tmp, data).expect("write");
        std::fs::rename(&tmp, &path).expect("rename");
        Ok(Obj::new())
    }

    pub fn op_fs_delete(&self, a: &Obj) -> Result<Obj, NodeError> {
        let path = self.resolve(&a["root"], &a["rel"], true)?;
        if !self.st().allow_delete {
            return Err(self.err("policy_denied", "this node's policy does not allow deletes"));
        }
        let target = path.display().to_string();
        let busy = self.st().in_use_paths.iter().any(|u| {
            *u == target
                || u.starts_with(&format!("{target}/"))
                || target.starts_with(&format!("{u}/"))
        });
        if busy {
            return Err(self.err("in_use", "a running container mounts that path"));
        }
        let Ok(meta) = path.symlink_metadata() else {
            return Err(self.err("not_found", "no such path"));
        };
        let freed = if meta.is_dir() {
            let total = dir_size(&path);
            std::fs::remove_dir_all(&path).expect("rmtree");
            total
        } else {
            let n = if meta.is_file() { meta.len() } else { 0 };
            std::fs::remove_file(&path).expect("unlink");
            n
        };
        let rel = a["rel"].as_str().unwrap_or("").to_string();
        let root = a["root"].clone();
        let mut st = self.st();
        let before = st.entries.len();
        st.entries.retain(|e| {
            let er = e
                .get("rel")
                .map(|v| v.as_str().map_or_else(|| v.to_string(), str::to_string))
                .unwrap_or_default();
            !(e.get("root") == Some(&root) && (er == rel || er.starts_with(&format!("{rel}/"))))
        });
        if st.entries.len() != before {
            st.generation += 1;
        }
        Ok(obj(json!({"freed": freed})))
    }

    pub fn op_models_scan(&self, a: &Obj) -> Result<Obj, NodeError> {
        let enabled = {
            let mut st = self.st();
            st.scans.push(a.clone());
            st.scan_disk_enabled
        };
        if enabled {
            self.scan();
        }
        Ok(obj(json!({"generation": self.generation()})))
    }

    pub fn op_models_list(&self, a: &Obj) -> Result<Obj, NodeError> {
        let mut st = self.st();
        let limit = a
            .get("limit")
            .and_then(Value::as_i64)
            .unwrap_or(200)
            .clamp(1, 200) as usize;
        let cursor = a.get("cursor").and_then(Value::as_i64).unwrap_or(0) as usize;
        let mut generation = st.generation;
        if st.list_generation_bump_once && cursor != 0 {
            st.list_generation_bump_once = false;
            st.generation += 1; // a scan finished between two pages
            generation = st.generation;
        }
        let page: Vec<Value> = st
            .entries
            .iter()
            .skip(cursor)
            .take(limit)
            .cloned()
            .collect();
        let roots: Vec<Value> = st
            .roots
            .iter()
            .map(|r| {
                let n = st
                    .entries
                    .iter()
                    .filter(|e| e.get("root").and_then(Value::as_str) == Some(&r.id))
                    .count();
                json!({"root": r.id, "scanned_at": 1.0, "complete": true, "entries": n})
            })
            .collect();
        let mut out = obj(json!({"generation": generation, "entries": page, "roots": roots}));
        if cursor + limit < st.entries.len() {
            out.insert("next_cursor".into(), json!(cursor + limit));
        }
        Ok(out)
    }

    pub fn op_models_config(&self, a: &Obj) -> Result<Obj, NodeError> {
        let key = (
            a["root"].as_str().unwrap_or("").to_string(),
            a["rel"].as_str().unwrap_or("").to_string(),
        );
        match self.st().configs.get(&key) {
            Some(c) => Ok(obj(json!({"config": c}))),
            None => Err(self.err("not_found", "no config.json")),
        }
    }

    fn dispatch(&self, op: &str, a: &Obj) -> Result<Obj, NodeError> {
        let over = self.st().overrides.get(op).cloned();
        if let Some(f) = over {
            return f(self, a);
        }
        match op {
            "fs.stat" => self.op_fs_stat(a),
            "fs.symlink" => self.op_fs_symlink(a),
            "fs.put_small" => self.op_fs_put_small(a),
            "fs.delete" => self.op_fs_delete(a),
            "models.scan" => self.op_models_scan(a),
            "models.list" => self.op_models_list(a),
            "models.config" => self.op_models_config(a),
            _ => Err(self.err(
                "unsupported",
                format!("the fake node does not implement {op}"),
            )),
        }
    }

    // ---- a tiny scanner (hf-home, hf-hub and gguf roots) ---------------------------------------------------------
    pub fn scan(&self) {
        let roots = self.st().roots.clone();
        let mut entries: Vec<Value> = Vec::new();
        for r in &roots {
            let path = r.path.clone().expect("path");
            assert!(
                path.starts_with(&self.base),
                "test would scan a real directory"
            );
            if r.kind == "hf-home" || r.kind == "hf-hub" {
                let hub = if r.kind == "hf-home" {
                    path.join("hub")
                } else {
                    path.clone()
                };
                let prefix = if r.kind == "hf-home" { "hub/" } else { "" };
                let mut dirs: Vec<PathBuf> = std::fs::read_dir(&hub)
                    .map(|rd| rd.flatten().map(|e| e.path()).collect())
                    .unwrap_or_default();
                dirs.retain(|d| {
                    d.file_name()
                        .is_some_and(|n| n.to_string_lossy().starts_with("models--"))
                });
                dirs.sort();
                for d in dirs {
                    let rf = d.join("refs").join("main");
                    let Ok(rev) = std::fs::read_to_string(&rf) else {
                        continue;
                    };
                    let rev = rev.trim().to_string();
                    if !d.join("snapshots").join(&rev).is_dir() {
                        continue;
                    }
                    let dn = d.file_name().expect("name").to_string_lossy().to_string();
                    let rest = &dn["models--".len()..];
                    let (org, name) = rest.split_once("--").unwrap_or((rest, ""));
                    let blobs: Vec<PathBuf> = std::fs::read_dir(d.join("blobs"))
                        .map(|rd| rd.flatten().map(|e| e.path()).collect())
                        .unwrap_or_default();
                    let blobs: Vec<PathBuf> = blobs
                        .into_iter()
                        .filter(|f| f.is_file() && !f.to_string_lossy().ends_with(".incomplete"))
                        .collect();
                    let size: u64 = blobs.iter().map(|f| file_size(f)).sum();
                    entries.push(json!({"root": r.id, "rel": format!("{prefix}{dn}"), "format": "hf",
                                        "name": format!("{org}/{name}"), "revision": rev, "size": size,
                                        "files": blobs.len(), "mtime": 1.0, "atime": 2.0}));
                }
            } else if r.kind == "gguf" {
                let mut found = Vec::new();
                rglob(&path, &mut found, ".gguf");
                found.sort_by_key(|p| {
                    p.components()
                        .map(|c| c.as_os_str().to_string_lossy().into_owned())
                        .collect::<Vec<_>>()
                });
                for f in found {
                    let rel = f
                        .strip_prefix(&path)
                        .expect("rel")
                        .to_string_lossy()
                        .to_string();
                    let stem = f.file_stem().expect("stem").to_string_lossy().to_string();
                    entries.push(json!({"root": r.id, "rel": rel, "format": "gguf", "name": stem,
                                        "quant": gguf_quant(&rel), "size": file_size(&f), "files": 1,
                                        "mtime": 1.0, "atime": 2.0}));
                }
            }
        }
        let mut st = self.st();
        if entries != st.entries {
            st.entries = entries;
            st.generation += 1;
        }
    }
}

fn rglob(dir: &Path, out: &mut Vec<PathBuf>, suffix: &str) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            rglob(&p, out, suffix);
        } else if p.to_string_lossy().ends_with(suffix) {
            out.push(p);
        }
    }
}

fn file_size(p: &Path) -> u64 {
    std::fs::metadata(p).map(|m| m.len()).unwrap_or(0)
}

fn dir_size(p: &Path) -> u64 {
    let mut total = 0;
    if let Ok(rd) = std::fs::read_dir(p) {
        for e in rd.flatten() {
            let q = e.path();
            if let Ok(m) = q.symlink_metadata() {
                if m.is_dir() {
                    total += dir_size(&q);
                } else if m.is_file() {
                    total += m.len();
                }
            }
        }
    }
    total
}

pub fn with_suffix(p: &Path, suffix: &str) -> PathBuf {
    let name = p
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    p.with_file_name(format!("{name}{suffix}"))
}

/// `os.path.realpath`: the longest existing prefix resolved, the rest appended (normalised).
pub fn realpath(p: &Path) -> PathBuf {
    let mut done = PathBuf::new();
    let mut comps = p.components().peekable();
    let mut acc = PathBuf::new();
    while let Some(c) = comps.next() {
        acc.push(c.as_os_str());
        match std::fs::canonicalize(&acc) {
            Ok(r) => done = r,
            Err(_) => {
                let mut rest = done;
                match c {
                    Component::ParentDir => {
                        rest.pop();
                    }
                    Component::CurDir => {}
                    _ => rest.push(c.as_os_str()),
                }
                for c2 in comps.by_ref() {
                    match c2 {
                        Component::ParentDir => {
                            rest.pop();
                        }
                        Component::CurDir => {}
                        _ => rest.push(c2.as_os_str()),
                    }
                }
                return rest;
            }
        }
    }
    done
}

#[async_trait]
impl NodeClient for DataNode {
    fn host(&self) -> &HostConfig {
        &self.host
    }
    fn link(&self) -> LinkStatus {
        LinkStatus::new(&self.st().state, None)
    }
    fn latest(&self, kind: &str) -> Option<Arc<Obj>> {
        if self.st().forgotten {
            return None;
        }
        match kind {
            "hello" => Some(Arc::new(self.hello())),
            "disks" => Some(Arc::new(self.disks())),
            "sample" => Some(Arc::new(self.sample())),
            _ => None,
        }
    }
    async fn call(
        &self,
        op: &str,
        args: Option<Obj>,
        _timeout: Duration,
    ) -> Result<Obj, NodeError> {
        let a = args.unwrap_or_default();
        self.st().calls.push((op.into(), a.clone()));
        self.gate(op, &a)?;
        tokio::task::yield_now().await;
        let result = self.dispatch(op, &a)?;
        let res = obj(json!({"id": 1, "ok": true, "result": result}));
        Ok(ec_proto::sanitize_res(&res, op)
            .get("result")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default())
    }
    fn open(&self, op: &str, args: Option<Obj>) -> Box<dyn NodeStream> {
        let a = args.unwrap_or_default();
        self.st().calls.push((op.into(), a.clone()));
        match op {
            "fs.write" => Box::new(WriteStream::new(self.self_arc(), &a)),
            "fs.read" => Box::new(ReadStream::new(self.self_arc(), &a)),
            _ => panic!("the fake node cannot stream {op}"),
        }
    }
}

impl DataNode {
    fn self_arc(&self) -> Arc<DataNode> {
        self.me.upgrade().expect("the node is alive")
    }
}

/// `FakeNode(base, host_id, ...)`
pub fn node(base: &Path, host_id: &str, o: NodeOpts) -> Arc<DataNode> {
    DataNode::new(base, host_id, o)
}

/// An upload into `fs.write`: the node side of docs/FLEET-PROTOCOL.md for one file.
pub struct WriteStream {
    node: Arc<DataNode>,
    error: Option<NodeError>,
    result: Option<Obj>,
    ended: bool,
    sent: i64,
    final_: PathBuf,
    part: PathBuf,
    size: i64,
    sha: Option<String>,
}

impl WriteStream {
    fn new(node: Arc<DataNode>, args: &Obj) -> Self {
        let mut w = Self {
            node: Arc::clone(&node),
            error: None,
            result: None,
            ended: false,
            sent: 0,
            final_: PathBuf::new(),
            part: PathBuf::new(),
            size: 0,
            sha: None,
        };
        let r = (|| -> Result<(), NodeError> {
            node.gate("fs.write", args)?;
            w.final_ = node.resolve(&args["root"], &args["rel"], true)?;
            w.part = with_suffix(&w.final_, ".incomplete");
            w.size = args["size"].as_i64().unwrap_or(0);
            w.sha = args
                .get("sha256")
                .and_then(Value::as_str)
                .map(str::to_string);
            let offset = args.get("offset").and_then(Value::as_i64).unwrap_or(0);
            if offset == 0 {
                std::fs::create_dir_all(w.final_.parent().expect("parent")).expect("mkdir");
                std::fs::write(&w.part, b"").expect("partial");
            } else {
                let have = if w.part.is_file() {
                    file_size(&w.part) as i64
                } else {
                    -1
                };
                if have != offset {
                    return Err(node.err(
                        "conflict",
                        format!("offset {offset} is not the size of the partial ({have})"),
                    ));
                }
            }
            node.st().writes.push(
                json!({"rel": args["rel"], "offset": offset, "size": w.size, "sha256": w.sha}),
            );
            Ok(())
        })();
        if let Err(e) = r {
            w.error = Some(e);
        }
        w
    }
}

fn append(p: &Path, data: &[u8]) {
    use std::io::Write as _;
    let mut f = std::fs::OpenOptions::new()
        .append(true)
        .open(p)
        .expect("open partial");
    f.write_all(data).expect("append");
}

#[async_trait]
impl NodeStream for WriteStream {
    async fn next(&mut self) -> Option<StreamItem> {
        None
    }
    async fn write(&mut self, data: &[u8]) -> Result<(), NodeError> {
        if let Some(e) = &self.error {
            return Err(e.clone());
        }
        assert!(
            data.len() <= ec_proto::MAX_DATA,
            "a DATA frame over MAX_DATA"
        );
        let (space_left, fail_after) = {
            let st = self.node.st();
            (st.space_left, st.fail_after_bytes)
        };
        let n = data.len() as i64;
        if space_left.is_some_and(|s| n > s) {
            let e = self
                .node
                .err("no_space", "the filesystem would drop below 1 GiB free");
            self.error = Some(e.clone());
            return Err(e);
        }
        if let Some(fa) = fail_after
            && self.sent + n > fa
        {
            let keep = usize::try_from((fa - self.sent).max(0)).unwrap_or(0);
            append(&self.part, &data[..keep.min(data.len())]);
            self.node.st().fail_after_bytes = None;
            let e = self.node.err("io_error", "read error on the source disk");
            self.error = Some(e.clone());
            return Err(e);
        }
        append(&self.part, data);
        self.sent += n;
        if let Some(s) = self.node.st().space_left.as_mut() {
            *s -= n;
        }
        tokio::task::yield_now().await;
        Ok(())
    }
    async fn end(&mut self) -> Result<(), NodeError> {
        if let Some(e) = &self.error {
            return Err(e.clone());
        }
        self.ended = true;
        let data = std::fs::read(&self.part).expect("partial");
        let digest = sha_hex(&data);
        if data.len() as i64 != self.size {
            let _ = std::fs::remove_file(&self.part);
            self.error = Some(self.node.err(
                "size_mismatch",
                format!("got {} bytes, expected {}", data.len(), self.size),
            ));
        } else if self.sha.as_ref().is_some_and(|s| *s != digest) {
            let _ = std::fs::remove_file(&self.part);
            self.error = Some(self.node.err("hash_mismatch", "the sha256 does not match"));
        } else {
            std::fs::rename(&self.part, &self.final_).expect("rename");
            self.result = Some(obj(json!({"size": data.len(), "sha256": digest})));
        }
        Ok(())
    }
    async fn result(&mut self) -> Result<Obj, NodeError> {
        if let Some(e) = &self.error {
            return Err(e.clone());
        }
        assert!(self.ended, "result() before end()");
        Ok(self.result.clone().unwrap_or_default())
    }
    async fn cancel(&mut self) {}
}

pub struct ReadStream {
    node: Arc<DataNode>,
    error: Option<NodeError>,
    data: Vec<u8>,
    pos: usize,
    count: usize,
}

impl ReadStream {
    fn new(node: Arc<DataNode>, args: &Obj) -> Self {
        let mut s = Self {
            node: Arc::clone(&node),
            error: None,
            data: Vec::new(),
            pos: 0,
            count: 0,
        };
        if let Some(e) = node.st().read_error.clone() {
            s.error = Some(e);
            return s;
        }
        let r = (|| -> Result<(), NodeError> {
            node.gate("fs.read", args)?;
            let path = node.resolve(&args["root"], &args["rel"], false)?;
            let meta = path.symlink_metadata();
            if !path.is_file() || meta.is_ok_and(|m| m.file_type().is_symlink()) {
                return Err(node.err("not_found", "no such file"));
            }
            let off = args.get("offset").and_then(Value::as_i64).unwrap_or(0) as usize;
            let all = std::fs::read(&path).expect("read");
            s.data = all.get(off..).map(<[u8]>::to_vec).unwrap_or_default();
            node.st()
                .reads
                .push(json!({"rel": args["rel"], "offset": off}));
            Ok(())
        })();
        if let Err(e) = r {
            s.error = Some(e);
        }
        s
    }
}

#[async_trait]
impl NodeStream for ReadStream {
    async fn next(&mut self) -> Option<StreamItem> {
        if self.error.is_some() || self.pos >= self.data.len() {
            return None;
        }
        let gate = self.node.st().read_gate.clone();
        if let Some((n, g)) = gate
            && self.count >= n
        {
            g.wait().await;
        }
        self.count += 1;
        let step = self.node.st().read_chunk;
        let end = (self.pos + step).min(self.data.len());
        let chunk = Bytes::copy_from_slice(&self.data[self.pos..end]);
        self.pos = end;
        tokio::task::yield_now().await;
        Some(StreamItem::Data(chunk))
    }
    async fn write(&mut self, _data: &[u8]) -> Result<(), NodeError> {
        panic!("fs.read takes no upload")
    }
    async fn end(&mut self) -> Result<(), NodeError> {
        panic!("fs.read takes no upload")
    }
    async fn result(&mut self) -> Result<Obj, NodeError> {
        match &self.error {
            Some(e) => Err(e.clone()),
            None => Ok(obj(json!({"size": self.data.len()}))),
        }
    }
    async fn cancel(&mut self) {}
}

// ---- the fleet ---------------------------------------------------------------------------------------------------
/// `FleetPort` over data nodes. `set_state` also publishes the `fleet` bus topic like the real link does.
pub struct DataFleet {
    pub bus: Arc<EventBus>,
    pub nodes: Vec<Arc<DataNode>>,
}

impl DataFleet {
    pub fn new(bus: Arc<EventBus>, nodes: Vec<Arc<DataNode>>) -> Self {
        for n in &nodes {
            n.set_bus(Arc::clone(&bus));
        }
        Self { bus, nodes }
    }

    pub fn fake(&self, host_id: &str) -> Arc<DataNode> {
        Arc::clone(
            self.nodes
                .iter()
                .find(|n| n.host.id == host_id)
                .expect("node"),
        )
    }

    pub fn set_state(&self, host_id: &str, state: &str) {
        self.fake(host_id).st().state = state.into();
        self.bus
            .publish("fleet", json!({"id": host_id, "link": {"state": state}}));
    }
}

/// What the composition root needs on top of `FleetPort`, for building an app over data nodes (test_routers.rs).
/// Like the core FakeFleet: no summaries or details (the data routes do not use them).
#[async_trait]
impl ec_hub::domain::fleet::Fleet for DataFleet {
    async fn start(&self) {}
    async fn stop(&self) {}
    fn summaries(&self) -> AppResult<Vec<ec_hub::domain::fleet::HostSummary>> {
        Err(ec_hub::domain::errors::not_ported("tests: DataFleet"))
    }
    fn summary(&self, _host_id: &str) -> AppResult<ec_hub::domain::fleet::HostSummary> {
        Err(ec_hub::domain::errors::not_ported("tests: DataFleet"))
    }
    async fn detail(&self, _host_id: &str) -> AppResult<ec_hub::domain::fleet::HostDetail> {
        Err(ec_hub::domain::errors::not_ported("tests: DataFleet"))
    }
    fn reconnect(&self, _host_id: &str) -> AppResult<ec_hub::domain::fleet::HostSummary> {
        Err(ec_hub::domain::errors::not_ported("tests: DataFleet"))
    }
    fn set_instance_resolver(&self, _f: ec_hub::domain::fleet::InstanceResolver) {}
    fn set_counts_provider(&self, _f: ec_hub::domain::fleet::CountsProvider) {}
}

impl FleetPort for DataFleet {
    fn hosts(&self) -> Vec<HostConfig> {
        self.nodes.iter().map(|n| n.host.clone()).collect()
    }
    fn node(&self, host_id: &str) -> Result<Arc<dyn NodeClient>, Problem> {
        self.nodes
            .iter()
            .find(|n| n.host.id == host_id)
            .map(|n| Arc::clone(n) as Arc<dyn NodeClient>)
            .ok_or_else(|| {
                not_found(format!(
                    "no host {} is configured",
                    ec_proto::pyfmt::repr_str(&host_id.chars().take(40).collect::<String>())
                ))
                .code("unknown_host")
            })
    }
}

// ---- an asyncio.Event ------------------------------------------------------------------------------------------
#[derive(Clone)]
pub struct Gate(Arc<tokio::sync::watch::Sender<bool>>);

impl Default for Gate {
    fn default() -> Self {
        Self::new()
    }
}

impl Gate {
    pub fn new() -> Self {
        Self(Arc::new(tokio::sync::watch::channel(false).0))
    }
    pub fn set(&self) {
        self.0.send_replace(true);
    }
    pub fn is_set(&self) -> bool {
        *self.0.borrow()
    }
    pub async fn wait(&self) {
        let mut rx = self.0.subscribe();
        let _ = rx.wait_for(|v| *v).await;
    }
}

// ---- the Hugging Face hub ----------------------------------------------------------------------------------------
pub type ModelOverride = Arc<dyn Fn(&str) -> AppResult<HubModel> + Send + Sync>;

pub struct HubState {
    /// repo id -> (listing, files)
    pub repos: HashMap<String, (HubModel, Vec<(String, Vec<u8>)>)>,
    pub gated: HashSet<String>,
    pub corrupt: HashSet<String>,
    pub ignore_range: bool,
    pub opens: Vec<(String, String, String, i64)>,
    pub gate: Option<Gate>,
    pub pass_chunks: Option<usize>,
    pub listings: usize,
    pub model_override: Option<ModelOverride>,
    /// (path, fail at this chunk number (1-based), only while open calls are below this, the problem)
    pub fail_mid: Option<(String, usize, usize, Problem)>,
    pub open_calls: usize,
}

#[derive(Clone)]
pub struct DataHub(pub Arc<Mutex<HubState>>);

impl Default for DataHub {
    fn default() -> Self {
        Self::new()
    }
}

impl DataHub {
    pub fn new() -> Self {
        Self(Arc::new(Mutex::new(HubState {
            repos: HashMap::new(),
            gated: HashSet::new(),
            corrupt: HashSet::new(),
            ignore_range: false,
            opens: Vec::new(),
            gate: None,
            pass_chunks: None,
            listings: 0,
            model_override: None,
            fail_mid: None,
            open_calls: 0,
        })))
    }

    pub fn st(&self) -> MutexGuard<'_, HubState> {
        self.0.lock().expect("hub lock")
    }

    /// `FakeHub.add(repo_id, files, sha=, config=)`: large-looking files (weights) carry their sha256 like LFS.
    pub fn add_full(
        &self,
        repo_id: &str,
        files: &[(&str, &[u8])],
        sha: &str,
        config: Option<Value>,
    ) {
        let rf = files
            .iter()
            .map(|(p, b)| RepoFile {
                path: (*p).into(),
                size: b.len() as i64,
                sha256: LFS_SUFFIXES
                    .iter()
                    .any(|s| p.ends_with(s))
                    .then(|| sha_hex(b)),
            })
            .collect();
        let hm = HubModel {
            repo_id: repo_id.into(),
            sha: Some(sha.into()),
            gated: false,
            license: None,
            pipeline_tag: Some("text-generation".into()),
            library_name: None,
            tags: Vec::new(),
            files: rf,
            safetensors_params: Vec::new(),
            safetensors_total: None,
            config: config.and_then(|c| c.as_object().cloned()),
            downloads: None,
            likes: None,
        };
        let data = files
            .iter()
            .map(|(p, b)| ((*p).to_string(), b.to_vec()))
            .collect();
        self.st().repos.insert(repo_id.into(), (hm, data));
    }

    pub fn add(&self, repo_id: &str, files: &[(&str, &[u8])]) {
        self.add_full(repo_id, files, COMMIT, None);
    }

    pub fn add_files(&self, repo_id: &str, files: &[(String, Vec<u8>)], config: Option<Value>) {
        let f: Vec<(&str, &[u8])> = files
            .iter()
            .map(|(p, b)| (p.as_str(), b.as_slice()))
            .collect();
        self.add_full(repo_id, &f, COMMIT, config);
    }

    pub fn opens(&self) -> Vec<(String, String, String, i64)> {
        self.st().opens.clone()
    }

    /// Called before every chunk of every stream: how a test stops a transfer in the middle of a file.
    async fn turnstile(&self) {
        let gate = {
            let mut st = self.st();
            if let Some(n) = st.pass_chunks.as_mut()
                && *n > 0
            {
                *n -= 1;
                return;
            }
            st.gate.clone()
        };
        if let Some(g) = gate {
            g.wait().await;
        }
    }
}

pub struct HubStream {
    hub: DataHub,
    data: Vec<u8>,
    offset: i64,
    total: i64,
    pos: usize,
    n: usize,
    fail_at: Option<(usize, Problem)>,
    pub closed: bool,
}

#[async_trait]
impl ByteStream for HubStream {
    fn total(&self) -> Option<i64> {
        Some(self.total)
    }
    fn offset(&self) -> i64 {
        self.offset
    }
    async fn chunk(&mut self) -> AppResult<Option<Bytes>> {
        if self.pos >= self.data.len() {
            return Ok(None);
        }
        self.hub.turnstile().await;
        self.n += 1;
        if let Some((at, p)) = &self.fail_at
            && self.n == *at
        {
            return Err(p.clone().into());
        }
        let end = (self.pos + 16).min(self.data.len());
        let c = Bytes::copy_from_slice(&self.data[self.pos..end]);
        self.pos = end;
        tokio::task::yield_now().await;
        Ok(Some(c))
    }
    async fn close(&mut self) {
        self.closed = true;
    }
}

#[async_trait]
impl HubClient for DataHub {
    async fn search(
        &self,
        q: &str,
        _task: Option<&str>,
        _library: Option<&str>,
        _quant: Option<&str>,
        _sort: &str,
        limit: i64,
        _token_ok: bool,
    ) -> AppResult<Vec<HubModel>> {
        let st = self.st();
        let mut out: Vec<HubModel> = st
            .repos
            .values()
            .filter(|(m, _)| m.repo_id.to_lowercase().contains(&q.to_lowercase()))
            .map(|(m, _)| m.clone())
            .collect();
        out.truncate(usize::try_from(limit).unwrap_or(0));
        Ok(out)
    }
    async fn model(&self, repo_id: &str, _revision: Option<&str>) -> AppResult<HubModel> {
        let ov = {
            let mut st = self.st();
            st.listings += 1;
            st.model_override.clone()
        };
        if let Some(f) = ov {
            return f(repo_id);
        }
        let st = self.st();
        if st.gated.contains(repo_id) {
            return Err(gated_model(format!("gated: {repo_id}")).into());
        }
        st.repos
            .get(repo_id)
            .map(|(m, _)| m.clone())
            .ok_or_else(|| not_found(format!("not found on Hugging Face: {repo_id}")).into())
    }
    async fn card(&self, repo_id: &str, _revision: Option<&str>) -> AppResult<String> {
        Ok(format!("# {repo_id}"))
    }
    async fn open_file(
        &self,
        repo_id: &str,
        revision: &str,
        path: &str,
        start: i64,
    ) -> AppResult<Box<dyn ByteStream>> {
        let mut st = self.st();
        st.open_calls += 1;
        st.opens
            .push((repo_id.into(), revision.into(), path.into(), start));
        let mut data = st
            .repos
            .get(repo_id)
            .and_then(|(_, f)| f.iter().find(|(p, _)| p == path))
            .map(|(_, b)| b.clone())
            .expect("a file of the repo");
        if st.corrupt.contains(path) {
            data = vec![b'X'; data.len()];
        }
        let fail_at = match &st.fail_mid {
            Some((p, at, below, prob)) if p == path && st.open_calls < *below => {
                Some((*at, prob.clone()))
            }
            _ => None,
        };
        let total = data.len() as i64;
        let (data, offset) = if st.ignore_range && start != 0 {
            (data, 0)
        } else {
            let s = usize::try_from(start).unwrap_or(0).min(data.len());
            (data[s..].to_vec(), start)
        };
        Ok(Box::new(HubStream {
            hub: self.clone(),
            data,
            offset,
            total,
            pos: 0,
            n: 0,
            fail_at,
            closed: false,
        }))
    }
}

// ---- real HF cache directories, made without the code under test ------------------------------------------------
/// Write a Hugging Face cache directory the way huggingface_hub does: LFS blobs named by sha256, small files by their
/// git object id, relative snapshot links, refs/<name>.
pub fn seed_hf_cache(
    root: &Path,
    repo_id: &str,
    files: &[(String, Vec<u8>)],
    layout: &str,
    commit: &str,
    refs: &str,
) -> PathBuf {
    let hub = if layout == "hf-home" {
        root.join("hub")
    } else {
        root.to_path_buf()
    };
    let repo = hub.join(format!("models--{}", repo_id.replace('/', "--")));
    for (path, data) in files {
        let blob = if LFS_SUFFIXES.iter().any(|s| path.ends_with(s)) {
            sha_hex(data)
        } else {
            let mut h = sha1::Sha1::new();
            h.update(format!("blob {}\0", data.len()).as_bytes());
            h.update(data);
            hex::encode(h.finalize())
        };
        std::fs::create_dir_all(repo.join("blobs")).expect("blobs");
        std::fs::write(repo.join("blobs").join(&blob), data).expect("blob");
        let link = repo.join("snapshots").join(commit).join(path);
        std::fs::create_dir_all(link.parent().expect("parent")).expect("snap");
        let depth = path.matches('/').count() + 2;
        let target = format!("{}blobs/{blob}", "../".repeat(depth));
        std::os::unix::fs::symlink(target, &link).expect("symlink");
    }
    std::fs::create_dir_all(repo.join("refs")).expect("refs");
    std::fs::write(repo.join("refs").join(refs), commit).expect("ref");
    repo
}

// ---- entries as a node's scanner reports them ---------------------------------------------------------------------
pub fn hf_entry(root: &str, repo: &str, extra: Value) -> Value {
    hf_entry_at(root, repo, true, extra)
}

pub fn hf_entry_at(root: &str, repo: &str, home: bool, extra: Value) -> Value {
    let folder = format!("models--{}", repo.replace('/', "--"));
    let mut e = json!({"root": root, "rel": format!("{}{folder}", if home { "hub/" } else { "" }), "format": "hf",
                       "name": repo, "revision": COMMIT, "size": 1000, "files": 3, "mtime": 10.0, "atime": 20.0});
    merge(&mut e, extra);
    e
}

pub fn gguf_entry(root: &str, rel: &str, extra: Value) -> Value {
    let last = rel.rsplit('/').next().unwrap_or(rel);
    let stem = &last[..last.len().saturating_sub(5)];
    let mut e = json!({"root": root, "rel": rel, "format": "gguf", "name": stem, "quant": "Q4_K_M", "size": 4000,
                       "files": 1, "mtime": 10.0, "atime": 20.0});
    merge(&mut e, extra);
    e
}

pub fn entry(root: &str, rel: &str, fmt: &str, name: Option<&str>, extra: Value) -> Value {
    let mut e = json!({"root": root, "rel": rel, "format": fmt, "name": name, "size": 100, "files": 1, "mtime": 10.0});
    merge(&mut e, extra);
    e
}

fn merge(e: &mut Value, extra: Value) {
    if let (Value::Object(a), Value::Object(b)) = (e, extra) {
        for (k, v) in b {
            a.insert(k, v);
        }
    }
}

// ---- the world (conftest.py) ---------------------------------------------------------------------------------------
pub fn make_settings(tmp: &Path) -> Settings {
    let mut s = Settings::defaults();
    s.data_dir = tmp.join("data");
    s.hf_cache_dir = tmp.join("hf-unused");
    s.otlp_enabled = false;
    s.background_tasks = false;
    s.hf_token = None;
    s.hf_token_file = tmp.join("no-token");
    s.discovery_ports = Vec::new();
    s.fleet_enabled = false;
    s.hf_endpoint = "https://127.0.0.1:9".into();
    s
}

pub type InUse = Arc<dyn Fn(&str, &str) -> Vec<String> + Send + Sync>;

#[derive(Default)]
pub struct WorldOpts {
    pub in_use: Option<InUse>,
    pub store: Option<Arc<Store>>,
    pub page_limit: Option<i64>,
    pub hub: Option<DataHub>,
    pub clock: Option<ec_hub::services::downloads::Clock>,
    pub max_queue: Option<i64>,
}

/// A hub with a fleet of data nodes, the library over them and the download service.
pub struct World {
    pub bus: Arc<EventBus>,
    pub store: Arc<Store>,
    pub nodes: Vec<Arc<DataNode>>,
    pub fleet: Arc<DataFleet>,
    pub in_use_calls: Arc<Mutex<Vec<(String, String)>>>,
    pub busy: Arc<Mutex<HashMap<(String, String), Vec<String>>>>,
    pub library: Arc<LibraryService>,
    pub hub: DataHub,
    pub settings: Arc<SettingsService>,
    pub downloads: Arc<DownloadService>,
}

impl World {
    pub fn new(tmp: &Path, nodes: Vec<Arc<DataNode>>, o: WorldOpts) -> Self {
        let bus = Arc::new(EventBus::new());
        let store = o
            .store
            .unwrap_or_else(|| Arc::new(Store::memory().expect("store")));
        let fleet = Arc::new(DataFleet::new(Arc::clone(&bus), nodes.clone()));
        let in_use_calls: Arc<Mutex<Vec<(String, String)>>> = Arc::default();
        let busy: Arc<Mutex<HashMap<(String, String), Vec<String>>>> = Arc::default();
        let (calls2, busy2, user) = (
            Arc::clone(&in_use_calls),
            Arc::clone(&busy),
            o.in_use.clone(),
        );
        let in_use: InUse = Arc::new(move |host: &str, path: &str| {
            calls2
                .lock()
                .expect("calls")
                .push((host.into(), path.into()));
            match &user {
                Some(f) => f(host, path),
                None => busy2
                    .lock()
                    .expect("busy")
                    .get(&(host.to_string(), path.to_string()))
                    .cloned()
                    .unwrap_or_default(),
            }
        });
        let library = Arc::new(LibraryService::with_options(
            Arc::clone(&store),
            Arc::clone(&fleet) as Arc<dyn FleetPort>,
            make_adapters(),
            Arc::clone(&bus),
            in_use,
            o.page_limit.unwrap_or(200),
            &[0.01],
            30.0,
        ));
        let hub = o.hub.unwrap_or_default();
        let settings = Arc::new(SettingsService::new(
            Arc::clone(&store),
            Arc::new(make_settings(tmp)),
            Box::new(Vec::new),
        ));
        let downloads = Arc::new(DownloadService::new(
            Arc::clone(&store),
            Arc::new(hub.clone()) as Arc<dyn HubClient>,
            Arc::new(HfService::with_ttl(
                Arc::new(hub.clone()) as Arc<dyn HubClient>,
                Duration::ZERO,
            )),
            Arc::clone(&settings),
            Arc::clone(&bus),
            Arc::clone(&fleet) as Arc<dyn FleetPort>,
            Arc::clone(&library),
            2,
            o.max_queue.unwrap_or(50),
        ));
        if let Some(c) = o.clock {
            downloads.set_clock(c);
        }
        Self {
            bus,
            store,
            nodes,
            fleet,
            in_use_calls,
            busy,
            library,
            hub,
            settings,
            downloads,
        }
    }

    pub async fn start(self) -> Self {
        self.library.start().await;
        self.library.wait_idle(Duration::from_secs(10)).await;
        self
    }

    pub async fn settle(&self) {
        self.library.wait_idle(Duration::from_secs(10)).await;
    }

    pub async fn stop(&self) {
        self.library.stop().await;
        self.downloads.shutdown().await;
    }

    pub fn set_busy(&self, host: &str, path: &str, ids: &[&str]) {
        self.busy.lock().expect("busy").insert(
            (host.into(), path.into()),
            ids.iter().map(|s| (*s).to_string()).collect(),
        );
    }

    pub fn clear_busy(&self) {
        self.busy.lock().expect("busy").clear();
    }

    pub fn count(&self, sql: &str) -> i64 {
        self.store
            .one(sql, &[])
            .expect("query")
            .and_then(|r| r.i64("n"))
            .unwrap_or(-1)
    }
}

// ---- helpers.py ------------------------------------------------------------------------------------------------------
/// The machine that owns the disk: 10.0.0.1, an HF cache at /data/hf and a GGUF dir at /data/gguf.
pub fn nfs_alpha(tmp: &Path, addresses: Option<Vec<&str>>) -> Arc<DataNode> {
    node(
        tmp,
        "alpha",
        NodeOpts {
            roots: vec![
                RootCfg::new("hf", "hf-home").at("/data/hf"),
                RootCfg::new("gguf", "gguf").at("/data/gguf"),
            ],
            scan_disk: false,
            addresses: addresses
                .unwrap_or_else(|| vec!["10.0.0.1", "127.0.0.1"])
                .into_iter()
                .map(str::to_string)
                .collect(),
            hostname: Some("alpha.lan".into()),
            mounts: Some(vec![mount_with_roots("/", &["hf", "gguf"])]),
            ..NodeOpts::default()
        },
    )
}

pub fn nfs_mount(remote_host: &str, remote_path: &str, mount: &str, roots: &[&str]) -> Value {
    mount_dict(
        mount,
        MountOpts {
            fstype: Some("nfs4".into()),
            source: Some(format!("{remote_host}:{remote_path}")),
            model: None,
            remote: true,
            remote_host: Some(remote_host.into()),
            remote_path: Some(remote_path.into()),
            roots: roots.iter().map(|s| (*s).to_string()).collect(),
            ..MountOpts::default()
        },
    )
}

pub fn default_nfs_mount() -> Value {
    nfs_mount("10.0.0.1", "/data", "/mnt/alpha", &["shared", "shared_g"])
}

/// The machine that mounts it: /mnt/alpha, read-only roots `shared` (HF cache) and `shared_g` (GGUF).
pub fn nfs_beta(tmp: &Path, mount: Option<Value>) -> Arc<DataNode> {
    node(
        tmp,
        "beta",
        NodeOpts {
            roots: vec![
                RootCfg::new("shared", "hf-home").ro().at("/mnt/alpha/hf"),
                RootCfg::new("shared_g", "gguf").ro().at("/mnt/alpha/gguf"),
                RootCfg::new("own", "hf-hub").at("/srv/hub"),
            ],
            addresses: vec!["10.0.0.2".into()],
            hostname: Some("beta".into()),
            mounts: Some(vec![
                mount_with_roots("/", &["own"]),
                mount.unwrap_or_else(default_nfs_mount),
            ]),
            scan_disk: false,
            ..NodeOpts::default()
        },
    )
}

pub async fn until(cond: impl Fn() -> bool, what: &str) {
    for _ in 0..1500 {
        if cond() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    panic!("timed out waiting for {what}");
}

/// Let `chunks` chunks through (config.json is one, then two 16-byte chunks of the weights), then hold the rest.
pub fn stop_midway(w: &World, chunks: usize) -> Gate {
    let g = Gate::new();
    let mut st = w.hub.st();
    st.pass_chunks = Some(chunks);
    st.gate = Some(g.clone());
    g
}

pub fn release(w: &World) {
    let g = {
        let mut st = w.hub.st();
        st.pass_chunks = None;
        st.gate.clone()
    };
    g.expect("a gate").set();
}

pub fn partial_of(node: &DataNode, root: &str, rel: &str) -> PathBuf {
    node.root_path(root).join(format!("{rel}.incomplete"))
}

pub fn exists_with_size(p: &Path, size: u64) -> bool {
    p.is_file() && file_size(p) == size
}
