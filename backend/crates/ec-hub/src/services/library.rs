//! Python: backend/src/engine_console/services/library.py
//! (tests/data/test_library*.py).
//!
//! The cross-machine model library (contract v2, docs/ARCHITECTURE.md 10.5). Every host's node scans its own model
//! roots; the hub pages through each node's `models.list` and merges the answers:
//!
//! * One model on one host's root is a **copy**; copies of the same model on any machine group under one key.
//! * **Shared storage** is folded in: when host B's root sits on an NFS mount of an export of host A's disk, B's copy
//!   and A's are the same bytes, so the model lists A's copy once and says B can load it (`visible_on`).
//! * The last good snapshot of every host lives in SQLite (`library_snapshots`), so a machine that is down still
//!   shows its last-known models; they are marked stale and cannot be deleted from here.
//! * `in_use_by` and `deletable` are computed on every read (the `in_use` callback and the node's current policy).
//! * `find(host, ref)` and `visible_copies(host)` show a shared copy in `host`'s own terms (its root, its path, the
//!   mount it goes through); the copy's `id` stays the owner's.
//!
//! Every node is untrusted: entries were cleaned by the protocol and are cleaned again when read back from SQLite;
//! node strings are keys and display text only, never a hub-side path, SQL or HTML.
//!
//! Concurrency: the state sits behind one `std::sync::Mutex`, never held across an await nor while the `in_use`
//! callback runs (the lifecycle's callback may call back into `locate`).

use std::collections::{BTreeSet, HashMap, HashSet};
use std::net::IpAddr;
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use regex::Regex;
use serde_json::{Map, Value, json};
use sha1::Digest as _;
use tokio::task::JoinHandle;

use crate::adapters::AdapterRegistry;
use crate::adapters::base::ModelInfo;
use crate::domain::errors::{AppError, AppResult, NodeError, Obj, Problem, conflict, not_found};
use crate::domain::fleet::{FleetPort, LibraryCopy, LibraryModel, MountInfo, NodeClient, RootInfo};
use crate::domain::models::LocalModel;
use crate::domain::ports::HubModel;
use crate::domain::refs::ModelRef;
use crate::services::common::{EventBus, now};
use crate::services::gguf_info::{model_info_from_entry, normalize_quant, py};
use crate::services::hf::{build_model_info, gguf_quant, gguf_stem};
use crate::services::libport::LibraryPort;
use crate::services::nodeerr::problem_from_node;
use crate::services::store::Store;

/// The flag the routes check before serving the library (D10): it is ported.
pub const PORTED: bool = true;

pub const FORMATS: &[&str] = &["hf", "gguf", "ollama", "plain", "other"];
/// EventBus: {"host_id", "generation", "models"} after a host's snapshot changed
pub const LIBRARY_TOPIC: &str = "library";
/// per host: the node scans at most 50 000 directory entries per root
pub const MAX_ENTRIES: usize = 20_000;
pub const MAX_WARNINGS: usize = 10;
pub const GGUF_HEADER_KEYS: &[&str] = &[
    "block_count",
    "embedding_length",
    "head_count",
    "head_count_kv",
    "expert_count",
    "expert_used_count",
    "file_type",
    "size_label",
    "version",
];

static COPY_ID: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\A[0-9a-f]{16}\z").unwrap_or_else(|_| unreachable!()));
/// `(?:^|/)models--([^/]+?)--([^/]+?)(?:/|$)` (Python's `$` also matches before one trailing "\n")
static HF_DIR: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?:^|/)models--([^/]+?)--([^/]+?)(?:/|\n?\z)").unwrap_or_else(|_| unreachable!())
});

/// fn(host_id, path) -> ids of the instances whose container mounts that path (late-bound to the lifecycle).
pub type InUseFn = Arc<dyn Fn(&str, &str) -> Vec<String> + Send + Sync>;

/// Root kinds that naturally hold each format: when one machine reaches a copy through two of its roots, the natural
/// one is the root an engine should mount.
fn natural_kinds(fmt: &str) -> &'static [&'static str] {
    match fmt {
        "hf" => &["hf-home", "hf-hub"],
        "gguf" => &["gguf"],
        "plain" => &["plain"],
        "ollama" => &["ollama"],
        _ => &[],
    }
}

// ---- small helpers on untrusted values -----------------------------------------------------------------------
fn s_(v: Option<&Value>, cap: usize) -> Option<String> {
    match v {
        Some(Value::String(s)) if !s.is_empty() => Some(py::take(s, cap)),
        _ => None,
    }
}

fn i_(v: Option<&Value>) -> Option<i64> {
    v.and_then(py::as_int)
}

fn f_(v: Option<&Value>) -> Option<f64> {
    v.and_then(py::as_num)
}

/// `_norm`: posixpath.normpath, with a leading "//" folded to "/" (POSIX keeps it; both mean "/").
pub fn norm(path: &str) -> String {
    if path.is_empty() {
        return String::new();
    }
    let n = py::normpath(path);
    if n.starts_with("//") {
        format!("/{}", n.trim_start_matches('/'))
    } else {
        n
    }
}

/// `path` is `base` or inside it, by whole path components.
fn under(path: &str, base: &str) -> bool {
    base == "/" || path == base || path.starts_with(&format!("{base}/"))
}

fn overlap(a: &str, b: &str) -> bool {
    under(a, b) || under(b, a)
}

/// The node's own path rules for a relative path (docs/FLEET-PROTOCOL.md, "Path rules").
pub fn rel_ok(rel: &str) -> bool {
    if rel.is_empty() || rel.starts_with('/') || rel.contains('\\') || rel.contains('\0') {
        return false;
    }
    rel.split('/').all(|seg| !matches!(seg, "" | "." | ".."))
}

/// Stable id of one model on one host's root (docs/ARCHITECTURE.md 10.5).
pub fn copy_id(host_id: &str, root_id: &str, rel: &str) -> String {
    let d = sha1::Sha1::digest(format!("{host_id}|{root_id}|{rel}").as_bytes());
    hex::encode(d)[..16].to_string()
}

fn canon_addr(text: &str) -> String {
    let t = py::strip(text).trim_matches(['[', ']']);
    let t = t.split('%').next().unwrap_or("").to_lowercase();
    match t.parse::<IpAddr>() {
        Ok(a) => a.to_string(),
        Err(_) => t,
    }
}

fn is_loopback(addr: &str) -> bool {
    match addr.parse::<IpAddr>() {
        Ok(a) => a.is_loopback(),
        Err(_) => addr == "localhost" || addr == "localhost.localdomain",
    }
}

// ---- what the hub knows about one host's storage --------------------------------------------------------------
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RootSpec {
    pub id: String,
    pub path: String,
    pub kind: String,
    pub writable: bool,
    pub label: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Mount {
    pub mount: String,
    pub fstype: Option<String>,
    pub source: Option<String>,
    pub device: Option<String>,
    pub model: Option<String>,
    pub size: Option<i64>,
    pub used: Option<i64>,
    pub free: Option<i64>,
    pub readonly: bool,
    pub remote: bool,
    pub remote_host: Option<String>,
    pub remote_path: Option<String>,
    pub roots: Vec<String>,
}

impl Mount {
    pub fn nfs(&self) -> bool {
        self.fstype
            .as_deref()
            .is_some_and(|f| f.to_lowercase().starts_with("nfs"))
            && self.remote_host.is_some()
            && self.remote_path.is_some()
    }

    pub fn parse(d: &Value) -> Option<Mount> {
        let Value::Object(d) = d else {
            return None;
        };
        let mount = s_(d.get("mount"), 512)?;
        if !mount.starts_with('/') {
            return None;
        }
        let remote_path = s_(d.get("remote_path"), 512);
        let roots = match d.get("roots") {
            Some(Value::Array(a)) => a
                .iter()
                .filter_map(|r| r.as_str().map(str::to_string))
                .take(32)
                .collect(),
            _ => Vec::new(),
        };
        Some(Mount {
            mount: norm(&mount),
            fstype: s_(d.get("fstype"), 32),
            source: s_(d.get("source"), 512),
            device: s_(d.get("device"), 64),
            model: s_(d.get("model"), 128),
            size: i_(d.get("size")),
            used: i_(d.get("used")),
            free: i_(d.get("free")),
            readonly: d.get("readonly") == Some(&Value::Bool(true)),
            remote: d.get("remote") == Some(&Value::Bool(true)),
            remote_host: s_(d.get("remote_host"), 253),
            remote_path: remote_path.map(|p| norm(&p)),
            roots,
        })
    }
}

/// The name a person recognises a drive by: "tank (zfs)", "NFS from gpu-box-2", "/ (ext4, Samsung 990)".
pub fn mount_label(m: &Mount, owner_label: Option<&str>) -> String {
    let ft = m.fstype.clone().unwrap_or_default().to_lowercase();
    if ft == "zfs" {
        let src = m.source.clone().unwrap_or_default();
        let pool = src
            .split('/')
            .next()
            .unwrap_or("")
            .split('@')
            .next()
            .unwrap_or("")
            .to_string();
        return format!("{} (zfs)", if pool.is_empty() { &m.mount } else { &pool });
    }
    if ft.starts_with("nfs") || m.remote {
        let kind = if ft.starts_with("nfs") {
            "NFS".to_string()
        } else if matches!(ft.as_str(), "cifs" | "smb3" | "smbfs") {
            "CIFS".to_string()
        } else if ft.is_empty() {
            "Network".to_string()
        } else {
            ft.to_uppercase()
        };
        let who = owner_label
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .or_else(|| m.remote_host.clone())
            .or_else(|| m.source.clone())
            .unwrap_or_else(|| "another machine".into());
        return format!("{kind} from {who}");
    }
    if let Some(model) = &m.model {
        return match &m.fstype {
            Some(f) => format!("{} ({f}, {model})", m.mount),
            None => format!("{} ({model})", m.mount),
        };
    }
    match &m.fstype {
        Some(f) => format!("{} ({f})", m.mount),
        None => m.mount.clone(),
    }
}

/// The mount a path lives on: the one with the longest mount point that contains it.
pub fn mount_for<'a>(mounts: &'a [Mount], path: &str) -> Option<&'a Mount> {
    let mut best: Option<&Mount> = None;
    for m in mounts {
        if under(path, &m.mount) && best.is_none_or(|b| m.mount.len() > b.mount.len()) {
            best = Some(m);
        }
    }
    best
}

/// The mount a model root lives on: the one the node says holds it, else the longest mount path above it.
fn mount_of_root<'a>(spec: &RootSpec, mounts: &'a [Mount]) -> Option<&'a Mount> {
    let claimed: Vec<&Mount> = mounts
        .iter()
        .filter(|m| m.roots.contains(&spec.id))
        .collect();
    if claimed.len() == 1 {
        Some(claimed[0])
    } else {
        mount_for(mounts, &spec.path)
    }
}

type MountSig = (
    String,
    Option<String>,
    Option<String>,
    Option<String>,
    Vec<String>,
);
/// What a copy's place in the library depends on (free space is not part of it).
pub type Signature = (
    Vec<(String, String, String)>,
    Vec<String>,
    Option<String>,
    Vec<MountSig>,
);

/// What the hub needs from a node's `hello` and `disks` to build that host's copies; stored beside the entries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostFacts {
    pub roots: Vec<RootSpec>,
    pub mode: String,
    pub allow_delete: bool,
    pub addresses: Vec<String>,
    pub hostname: Option<String>,
    pub mounts: Vec<Mount>,
}

impl Default for HostFacts {
    fn default() -> Self {
        Self {
            roots: Vec::new(),
            mode: "monitor".into(),
            allow_delete: false,
            addresses: Vec::new(),
            hostname: None,
            mounts: Vec::new(),
        }
    }
}

impl HostFacts {
    pub fn root(&self, root_id: &str) -> Option<&RootSpec> {
        self.roots.iter().find(|r| r.id == root_id)
    }

    pub fn signature(&self) -> Signature {
        let mut mounts: Vec<MountSig> = self
            .mounts
            .iter()
            .map(|m| {
                (
                    m.mount.clone(),
                    m.fstype.clone(),
                    m.remote_host.clone(),
                    m.remote_path.clone(),
                    m.roots.clone(),
                )
            })
            .collect();
        mounts.sort();
        let mut addrs = self.addresses.clone();
        addrs.sort();
        (
            self.roots
                .iter()
                .map(|r| (r.id.clone(), r.path.clone(), r.kind.clone()))
                .collect(),
            addrs,
            self.hostname.clone(),
            mounts,
        )
    }

    pub fn to_json(&self) -> Value {
        json!({"mode": self.mode, "allow_delete": self.allow_delete, "addresses": self.addresses,
               "hostname": self.hostname, "mounts": self.mounts.iter().map(mount_json).collect::<Vec<_>>()})
    }

    pub fn roots_json(&self) -> Value {
        Value::Array(
            self.roots
                .iter()
                .map(|r| json!({"id": r.id, "path": r.path, "kind": r.kind, "writable": r.writable, "label": r.label}))
                .collect(),
        )
    }

    pub fn from_json(roots: &Value, facts: &Value) -> HostFacts {
        let mut out = HostFacts {
            roots: parse_roots(roots),
            ..HostFacts::default()
        };
        if let Value::Object(f) = facts {
            out.mode = if f.get("mode").and_then(Value::as_str) == Some("manage") {
                "manage".into()
            } else {
                "monitor".into()
            };
            out.allow_delete = f.get("allow_delete") == Some(&Value::Bool(true));
            out.addresses = match f.get("addresses") {
                Some(Value::Array(a)) => a
                    .iter()
                    .filter_map(|x| x.as_str().map(str::to_string))
                    .take(64)
                    .collect(),
                _ => Vec::new(),
            };
            out.hostname = s_(f.get("hostname"), 253);
            out.mounts = match f.get("mounts") {
                Some(Value::Array(a)) => a.iter().take(64).filter_map(Mount::parse).collect(),
                _ => Vec::new(),
            };
        }
        out
    }
}

fn mount_json(m: &Mount) -> Value {
    json!({"mount": m.mount, "fstype": m.fstype, "source": m.source, "device": m.device, "model": m.model,
           "size": m.size, "used": m.used, "free": m.free, "readonly": m.readonly, "remote": m.remote,
           "remote_host": m.remote_host, "remote_path": m.remote_path, "roots": m.roots})
}

fn parse_roots(raw: &Value) -> Vec<RootSpec> {
    let mut roots: Vec<RootSpec> = Vec::new();
    let Value::Array(items) = raw else {
        return roots;
    };
    for r in items.iter().take(32) {
        let Value::Object(r) = r else {
            continue;
        };
        let (Some(rid), Some(path)) = (s_(r.get("id"), 32), s_(r.get("path"), 512)) else {
            continue;
        };
        if !path.starts_with('/') || roots.iter().any(|x| x.id == rid) {
            continue;
        }
        roots.push(RootSpec {
            id: rid,
            path: norm(&path),
            kind: s_(r.get("kind"), 16).unwrap_or_else(|| "auto".into()),
            writable: r.get("writable") == Some(&Value::Bool(true)),
            label: s_(r.get("label"), 64),
        });
    }
    roots
}

/// Facts from a node's latest `hello` and `disks`; the stored ones when the node has not spoken since the hub
/// started. None when there is neither.
pub fn parse_facts(
    hello: Option<&Obj>,
    disks: Option<&Obj>,
    fallback: Option<&HostFacts>,
) -> Option<HostFacts> {
    let Some(hello) = hello else {
        return fallback.cloned();
    };
    let empty = Map::new();
    let pol = match hello.get("policy") {
        Some(Value::Object(p)) => p,
        _ => &empty,
    };
    let addrs: Vec<String> = match hello.get("addresses") {
        Some(Value::Array(a)) => a
            .iter()
            .filter_map(|x| match x {
                Value::Object(o) => o
                    .get("addr")
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty()),
                _ => None,
            })
            .map(str::to_string)
            .take(64)
            .collect(),
        _ => Vec::new(),
    };
    let mounts = match disks.and_then(|d| d.get("mounts")) {
        Some(Value::Array(a)) => a.iter().take(64).filter_map(Mount::parse).collect(),
        _ => fallback.map(|f| f.mounts.clone()).unwrap_or_default(),
    };
    Some(HostFacts {
        roots: parse_roots(pol.get("model_roots").unwrap_or(&Value::Null)),
        mode: if pol.get("mode").and_then(Value::as_str) == Some("manage") {
            "manage".into()
        } else {
            "monitor".into()
        },
        allow_delete: pol.get("allow_delete") == Some(&Value::Bool(true)),
        addresses: addrs,
        hostname: match hello.get("host") {
            Some(Value::Object(h)) => s_(h.get("hostname"), 253),
            _ => None,
        },
        mounts,
    })
}

/// One host's last good `models.list`, plus the facts that were true when it was taken.
#[derive(Debug, Clone)]
pub struct HostSnapshot {
    pub generation: Option<i64>,
    pub entries: Vec<Obj>,
    pub scans: Vec<Obj>,
    pub facts: HostFacts,
    pub updated_at: f64,
}

/// What the last scan found in one root: enough to tell whether an `auto` root holds an HF cache.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RootUsage {
    pub entries: i64,
    /// an hf entry under hub/
    pub hf_home: bool,
    /// an hf entry directly in the root (models--org--name)
    pub hf_hub: bool,
    /// gguf files or plain model directories
    pub loose: bool,
    pub ollama: bool,
    pub other: bool,
}

/// What a copy of a library copy needs to know about its source (the entry as the node scanned it).
#[derive(Debug, Clone, PartialEq)]
pub struct CopySource {
    pub copy: LibraryCopy,
    pub format: String,
    pub name: String,
    pub revision: Option<String>,
    pub quant: Option<String>,
    pub shards: Option<i64>,
    /// every (host id, root id) that reaches these bytes: the owner and NFS aliases
    pub locations: Vec<(String, String)>,
    /// the node's own warnings for this entry
    pub warnings: Vec<String>,
}

// ---- who owns an address ----------------------------------------------------------------------------------------
#[derive(Debug, Clone)]
struct HostView {
    id: String,
    label: String,
    state: String,
    facts: HostFacts,
}

/// Which configured host owns an address or a name. An address two hosts both report identifies nobody.
struct AddressBook {
    addr: HashMap<String, String>,
    name: HashMap<String, String>,
}

impl AddressBook {
    fn new(views: &[(String, HostView)]) -> Self {
        let mut by_addr: HashMap<String, BTreeSet<String>> = HashMap::new();
        let mut by_name: HashMap<String, BTreeSet<String>> = HashMap::new();
        for (hid, v) in views {
            for a in &v.facts.addresses {
                let n = canon_addr(a);
                if !n.is_empty() && !is_loopback(&n) {
                    by_addr.entry(n).or_default().insert(hid.clone());
                }
            }
            let name = v.facts.hostname.clone().unwrap_or_default().to_lowercase();
            if !name.is_empty() && !is_loopback(&name) {
                by_name.entry(name.clone()).or_default().insert(hid.clone());
                let short = name.split('.').next().unwrap_or("").to_string();
                by_name.entry(short).or_default().insert(hid.clone());
            }
        }
        let single = |m: HashMap<String, BTreeSet<String>>| -> HashMap<String, String> {
            m.into_iter()
                .filter(|(_, v)| v.len() == 1)
                .filter_map(|(k, v)| v.into_iter().next().map(|h| (k, h)))
                .collect()
        };
        Self {
            addr: single(by_addr),
            name: single(by_name),
        }
    }

    /// The configured host an NFS server address or name belongs to (never `exclude`).
    fn find(&self, server: Option<&str>, exclude: &str) -> Option<String> {
        let server = server.filter(|s| !s.is_empty())?;
        let key = canon_addr(server);
        if key.is_empty() || is_loopback(&key) {
            return None;
        }
        let hid = self.addr.get(&key).or_else(|| self.name.get(&key))?;
        (hid != exclude).then(|| hid.clone())
    }
}

#[derive(Debug, Clone)]
struct RootLoc {
    host_id: String,
    root_id: String,
    owner: String,
    path: String,
    imported: bool,
}

fn root_locations(views: &[(String, HostView)], book: &AddressBook) -> Vec<RootLoc> {
    let mut locs = Vec::new();
    for (hid, v) in views {
        for spec in &v.facts.roots {
            let m = mount_of_root(spec, &v.facts.mounts);
            let (mut owner, mut path, mut imported) = (hid.clone(), spec.path.clone(), false);
            if let Some(m) = m
                && m.nfs()
                && let Some(rp) = &m.remote_path
                && under(&spec.path, &m.mount)
                && let Some(server) = book.find(m.remote_host.as_deref(), hid)
            {
                let rel = py::relpath(&spec.path, &m.mount);
                owner = server;
                imported = true;
                path = if rel == "." {
                    rp.clone()
                } else {
                    py::normpath(&py::join(rp, &rel))
                };
            }
            locs.push(RootLoc {
                host_id: hid.clone(),
                root_id: spec.id.clone(),
                owner,
                path,
                imported,
            });
        }
    }
    locs
}

type Shares = HashMap<(String, String), (Option<String>, Vec<String>)>;

/// (host_id, root_id) -> (shared_from, visible_on). Two roots are the same storage when they resolve to the same
/// owner host and their paths on it overlap.
fn shares(
    views: &[(String, HostView)],
    book: &AddressBook,
    order: &HashMap<String, usize>,
) -> Shares {
    let locs = root_locations(views, book);
    let mut out = HashMap::new();
    for loc in &locs {
        let peers: Vec<&RootLoc> = locs
            .iter()
            .filter(|o| o.owner == loc.owner && overlap(&o.path, &loc.path))
            .collect();
        let mut seen: BTreeSet<String> = peers.iter().map(|o| o.host_id.clone()).collect();
        seen.insert(loc.host_id.clone());
        let mut visible = vec![loc.host_id.clone()];
        if seen.contains(&loc.owner) && loc.owner != loc.host_id {
            visible.push(loc.owner.clone());
        }
        let mut rest: Vec<String> = seen.into_iter().filter(|h| !visible.contains(h)).collect();
        rest.sort_by_key(|h| order.get(h).copied().unwrap_or(1 << 30));
        visible.extend(rest);
        let shared = if loc.imported && peers.iter().any(|o| o.host_id == loc.owner && !o.imported)
        {
            Some(loc.owner.clone())
        } else {
            None
        };
        out.insert(
            (loc.host_id.clone(), loc.root_id.clone()),
            (shared, visible),
        );
    }
    out
}

// ---- the merged view ------------------------------------------------------------------------------------------
#[derive(Debug, Clone, PartialEq)]
struct Alias {
    host_id: String,
    root_id: String,
    rel: String,
    path: String,
}

#[derive(Debug, Clone)]
struct Rec {
    id: String,
    host_id: String,
    root_id: String,
    rel: String,
    path: String,
    fmt: String,
    name: String,
    key: String,
    entry: Obj,
    size: i64,
    mount: Option<Mount>,
    base: LibraryCopy,
    visible_on: Vec<String>,
    aliases: Vec<Alias>,
}

#[derive(Debug, Default)]
struct Snapshot {
    /// insertion order (Python's dict)
    recs: Vec<Rec>,
    rec_idx: HashMap<String, usize>,
    models: Vec<(String, LibraryModel)>,
    model_idx: HashMap<String, usize>,
    members: HashMap<String, Vec<String>>,
    alias: HashMap<(String, String, String), String>,
    by_host: HashMap<String, Vec<String>>,
    hf_by_name: HashMap<String, Vec<String>>,
    ollama_by_name: HashMap<String, Vec<String>>,
}

impl Snapshot {
    fn rec(&self, id: &str) -> Option<&Rec> {
        self.rec_idx.get(id).map(|i| &self.recs[*i])
    }
}

struct ReadCtx {
    views: Vec<(String, HostView)>,
    order: HashMap<String, usize>,
    book: AddressBook,
    drives: Mutex<HashMap<(String, String), Option<MountInfo>>>,
}

impl ReadCtx {
    fn view(&self, host_id: &str) -> Option<&HostView> {
        self.views
            .iter()
            .find(|(h, _)| h == host_id)
            .map(|(_, v)| v)
    }
}

fn loadable_hf(entry: &Obj) -> bool {
    entry.get("revision").is_some_and(|v| !v.is_null()) || i_(entry.get("files")).unwrap_or(0) > 0
}

fn fmt_of(entry: &Obj) -> String {
    let f = s_(entry.get("format"), 16);
    match f {
        Some(f) if FORMATS.contains(&f.as_str()) => f,
        _ => "other".into(),
    }
}

fn derive_name(fmt: &str, entry: &Obj, rel: &str) -> String {
    if let Some(name) = s_(entry.get("name"), 256) {
        return name;
    }
    let last = if rel.is_empty() {
        ""
    } else {
        rel.rsplit('/').next().unwrap_or("")
    };
    let or_model = |s: &str| {
        if s.is_empty() {
            "model".to_string()
        } else {
            s.to_string()
        }
    };
    match fmt {
        "hf" => match HF_DIR.captures(rel) {
            Some(c) => format!(
                "{}/{}",
                c.get(1).map_or("", |m| m.as_str()),
                c.get(2).map_or("", |m| m.as_str())
            ),
            None => or_model(last),
        },
        "gguf" => or_model(&gguf_stem(last)),
        _ => or_model(last),
    }
}

/// (key, quant). hf: the repo id; gguf: repo id or file stem plus ":<quant>"; ollama: name:tag; plain: the
/// directory name; anything else "other:<dir>".
fn entry_key(fmt: &str, name: &str, entry: &Obj, rel: &str) -> (String, Option<String>) {
    let quant = s_(entry.get("quant"), 32);
    if fmt == "gguf" {
        let quant = quant.or_else(|| {
            let q = gguf_quant(rel);
            (!q.is_empty()).then_some(q)
        });
        let mut base = name.to_string();
        if let Some(q) = &quant
            && !base
                .to_lowercase()
                .ends_with(&format!(":{}", q.to_lowercase()))
        {
            base = format!("{base}:{}", q.to_uppercase());
        }
        return (format!("gguf:{base}"), quant);
    }
    if fmt == "other" {
        return (format!("other:{name}"), quant);
    }
    (format!("{fmt}:{name}"), quant)
}

/// A GGUF-only HF repo arrives as ONE entry whose rel is the repo directory, with every quant (or split set) in
/// meta.gguf_files. The library lists each quant as a model of its own: rel = <repo dir>/<path of its first file>.
pub fn expand_gguf_repo(entry: &Obj) -> Vec<Obj> {
    let meta = match entry.get("meta") {
        Some(Value::Object(m)) => Some(m),
        _ => None,
    };
    let files = meta.and_then(|m| m.get("gguf_files"));
    let base_rel = entry.get("rel").and_then(Value::as_str);
    let (Some(meta), Some(Value::Array(files)), Some(base_rel)) = (meta, files, base_rel) else {
        return vec![entry.clone()];
    };
    if fmt_of(entry) != "gguf"
        || files.is_empty()
        || base_rel.is_empty()
        || py::ends_with_ci(base_rel, ".gguf")
    {
        return vec![entry.clone()];
    }
    let projector = files
        .iter()
        .any(|f| f.get("role").and_then(Value::as_str) == Some("mmproj"));
    let mut warnings: Vec<Value> = match entry.get("warnings") {
        Some(Value::Array(w)) => w
            .iter()
            .filter(|x| x.is_string())
            .take(7)
            .cloned()
            .collect(),
        _ => Vec::new(),
    };
    if projector {
        warnings.push(Value::String(
            "the repo also has a vision projector (mmproj) file; image input is not wired up for it yet".into(),
        ));
    }
    let mut out = Vec::new();
    for f in files {
        let Value::Object(f) = f else {
            continue;
        };
        if f.get("role").is_some_and(|r| !r.is_null()) {
            continue;
        }
        let Some(rel) = f
            .get("rel")
            .and_then(Value::as_str)
            .filter(|r| !r.is_empty())
        else {
            continue;
        };
        let mut e = entry.clone();
        e.insert("rel".into(), Value::String(format!("{base_rel}/{rel}")));
        for k in ["size", "files", "shards", "quant"] {
            e.insert(k.into(), f.get(k).cloned().unwrap_or(Value::Null));
        }
        e.insert("params".into(), Value::Null);
        e.insert("warnings".into(), Value::Array(warnings.clone()));
        e.insert(
            "meta".into(),
            json!({"hf_repo": meta.get("hf_repo").cloned().unwrap_or(Value::Null), "repo_rel": base_rel}),
        );
        out.push(e);
    }
    if out.is_empty() {
        vec![entry.clone()]
    } else {
        out
    }
}

fn mount_info(m: &Mount, owner: Option<String>, views: &[(String, HostView)]) -> MountInfo {
    let owner_label = owner.as_ref().and_then(|o| {
        views
            .iter()
            .find(|(h, _)| h == o)
            .map(|(_, v)| v.label.clone())
    });
    MountInfo {
        mount: m.mount.clone(),
        label: mount_label(m, owner_label.as_deref()),
        fstype: m.fstype.clone(),
        source: m.source.clone(),
        device: m.device.clone(),
        model: m.model.clone(),
        size_bytes: m.size,
        used_bytes: m.used,
        free_bytes: m.free,
        readonly: m.readonly,
        remote: m.remote,
        remote_host: m.remote_host.clone(),
        remote_host_id: owner,
        remote_path: m.remote_path.clone(),
        roots: m.roots.clone(),
    }
}

// ---- the service ------------------------------------------------------------------------------------------------
struct State {
    snaps: HashMap<String, Arc<HostSnapshot>>,
    view: Arc<Snapshot>,
    dirty: HashSet<String>,
    workers: HashMap<String, JoinHandle<()>>,
    up: HashSet<String>,
    errors: HashMap<String, String>,
    sigs: HashMap<String, Signature>,
    info_cache: HashMap<(String, Option<i64>, String), ModelInfo>,
    consumers: Vec<JoinHandle<()>>,
}

struct Inner {
    db: Arc<Store>,
    fleet: Arc<dyn FleetPort>,
    adapters: AdapterRegistry,
    bus: Arc<EventBus>,
    in_use: InUseFn,
    page_limit: i64,
    retry_delays: Vec<f64>,
    call_timeout: Duration,
    state: Mutex<State>,
    host_locks: Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
}

pub struct LibraryService {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for LibraryService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("LibraryService")
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

impl LibraryService {
    pub fn new(
        db: Arc<Store>,
        fleet: Arc<dyn FleetPort>,
        adapters: AdapterRegistry,
        bus: Arc<EventBus>,
        in_use: InUseFn,
    ) -> Self {
        Self::with_options(
            db,
            fleet,
            adapters,
            bus,
            in_use,
            200,
            &[5.0, 15.0, 60.0],
            30.0,
        )
    }

    /// `LibraryService(..., page_limit=, retry_delays=, call_timeout=)`.
    #[allow(clippy::too_many_arguments)]
    pub fn with_options(
        db: Arc<Store>,
        fleet: Arc<dyn FleetPort>,
        adapters: AdapterRegistry,
        bus: Arc<EventBus>,
        in_use: InUseFn,
        page_limit: i64,
        retry_delays: &[f64],
        call_timeout: f64,
    ) -> Self {
        Self {
            inner: Arc::new(Inner {
                db,
                fleet,
                adapters,
                bus,
                in_use,
                page_limit: page_limit.clamp(1, 200),
                retry_delays: if retry_delays.is_empty() {
                    vec![5.0]
                } else {
                    retry_delays.to_vec()
                },
                call_timeout: Duration::from_secs_f64(call_timeout),
                state: Mutex::new(State {
                    snaps: HashMap::new(),
                    view: Arc::new(Snapshot::default()),
                    dirty: HashSet::new(),
                    workers: HashMap::new(),
                    up: HashSet::new(),
                    errors: HashMap::new(),
                    sigs: HashMap::new(),
                    info_cache: HashMap::new(),
                    consumers: Vec::new(),
                }),
                host_locks: Mutex::new(HashMap::new()),
            }),
        }
    }

    /// Load the stored snapshots, subscribe to the node and fleet topics, and fetch every host that is up (in the
    /// background: a slow or dead machine must not hold the console's start-up).
    pub async fn start(&self) {
        let inner = &self.inner;
        if !lock(&inner.state).consumers.is_empty() {
            return;
        }
        let known: HashSet<String> = inner.fleet.hosts().into_iter().map(|h| h.id).collect();
        inner.load(&known);
        inner.rebuild();
        let mut consumers = Vec::new();
        for topic in ["node.models_changed", "fleet", "node.disks", "node.hello"] {
            // subscribed here, before start() returns: an event published right after is not lost
            let mut sub = inner.bus.subscribe(topic);
            let me = Arc::clone(inner);
            consumers.push(tokio::spawn(async move {
                while let Some(ev) = sub.recv().await {
                    // one event's panic is logged (redacted) and costs that event, never the consumer
                    crate::panics::guarded_sync("library: event", || match topic {
                        "node.models_changed" => me.on_changed(&ev),
                        "fleet" => me.on_fleet(&ev),
                        _ => me.on_facts(&ev),
                    });
                }
            }));
        }
        lock(&inner.state).consumers = consumers;
        for h in inner.fleet.hosts() {
            if inner.link_state(&h.id) == "up" {
                lock(&inner.state).up.insert(h.id.clone());
                Inner::schedule(inner, &h.id);
            }
        }
    }

    pub async fn stop(&self) {
        let tasks: Vec<JoinHandle<()>> = {
            let mut st = lock(&self.inner.state);
            let mut t: Vec<JoinHandle<()>> = std::mem::take(&mut st.consumers);
            t.extend(st.workers.drain().map(|(_, h)| h));
            t
        };
        for t in &tasks {
            t.abort();
        }
        for t in tasks {
            let _ = t.await;
        }
    }

    /// Test and ops helper: block until no host is being fetched.
    pub async fn wait_idle(&self, timeout: Duration) {
        let end = Instant::now() + timeout;
        loop {
            let busy = lock(&self.inner.state)
                .workers
                .values()
                .any(|h| !h.is_finished());
            if !busy || Instant::now() > end {
                return;
            }
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
    }

    /// Fetch one host's library now (errors propagate).
    pub async fn refresh(&self, host_id: &str) -> AppResult<()> {
        Inner::fetch(&self.inner, host_id).await
    }

    /// GET /library: every model on every machine, grouped; `host` keeps what that machine holds or can load.
    pub fn models(
        &self,
        host: Option<&str>,
        format: Option<&str>,
        q: Option<&str>,
    ) -> AppResult<Vec<LibraryModel>> {
        let inner = &self.inner;
        let ctx = inner.ctx();
        let snap = inner.view();
        let ql = q.unwrap_or("").to_lowercase();
        let words = py::split_ws(&ql);
        let mut out = Vec::new();
        for (key, m) in &snap.models {
            if format.is_some_and(|f| !f.is_empty() && m.format != f) {
                continue;
            }
            if let Some(h) = host.filter(|h| !h.is_empty())
                && !snap.members[key].iter().any(|i| {
                    snap.rec(i)
                        .is_some_and(|r| r.visible_on.iter().any(|v| v == h))
                })
            {
                continue;
            }
            if !words.is_empty() {
                let parts: Vec<&str> = [
                    Some(m.name.as_str()),
                    Some(m.key.as_str()),
                    m.quant.as_deref(),
                    m.arch.as_deref(),
                    Some(m.format.as_str()),
                ]
                .into_iter()
                .flatten()
                .filter(|x| !x.is_empty())
                .collect();
                let hay = parts.join(" ").to_lowercase();
                if !words.iter().all(|w| hay.contains(w)) {
                    continue;
                }
            }
            out.push(inner.model_view(&snap, key, &ctx));
        }
        out.sort_by(|a, b| {
            a.name
                .to_lowercase()
                .cmp(&b.name.to_lowercase())
                .then_with(|| a.key.cmp(&b.key))
        });
        Ok(out)
    }

    pub fn model(&self, key: &str) -> Option<LibraryModel> {
        let snap = self.inner.view();
        snap.model_idx
            .contains_key(key)
            .then(|| self.inner.model_view(&snap, key, &self.inner.ctx()))
    }

    pub fn copy(&self, copy_id: &str) -> Option<LibraryCopy> {
        let snap = self.inner.view();
        let rec = self.inner.rec(&snap, copy_id)?;
        Some(self.inner.copy_view(rec, &self.inner.ctx(), None, true))
    }

    /// The copy of `r` on `host_id` (`own`: on its own roots only, not shared storage).
    pub fn find_copy(&self, host_id: &str, r: &ModelRef, own: bool) -> Option<LibraryCopy> {
        let snap = self.inner.view();
        let rec = self.inner.find_rec(&snap, host_id, r, own)?;
        let ctx = self.inner.ctx();
        let alias = alias_for(rec, host_id, r, &ctx);
        Some(self.inner.copy_view(rec, &ctx, Some(&alias), true))
    }

    /// Every copy `host_id` can load, each expressed in `host_id`'s own terms.
    pub fn visible_copies(&self, host_id: &str) -> Vec<LibraryCopy> {
        let snap = self.inner.view();
        let ctx = self.inner.ctx();
        snap.recs
            .iter()
            .filter(|r| r.visible_on.iter().any(|h| h == host_id))
            .map(|r| {
                let a = pick_alias(r, host_id, &ctx);
                self.inner.copy_view(r, &ctx, Some(&a), true)
            })
            .collect()
    }

    pub fn source(&self, copy_id: &str) -> Option<CopySource> {
        let snap = self.inner.view();
        let rec = self.inner.rec(&snap, copy_id)?;
        let e = &rec.entry;
        Some(CopySource {
            copy: self.inner.copy_view(rec, &self.inner.ctx(), None, true),
            format: rec.fmt.clone(),
            name: rec.name.clone(),
            revision: s_(e.get("revision"), 64),
            quant: s_(e.get("quant"), 32),
            shards: i_(e.get("shards")),
            locations: rec
                .aliases
                .iter()
                .map(|a| (a.host_id.clone(), a.root_id.clone()))
                .collect(),
            warnings: match e.get("warnings") {
                Some(Value::Array(w)) => w
                    .iter()
                    .filter_map(|x| x.as_str().map(str::to_string))
                    .collect(),
                _ => Vec::new(),
            },
        })
    }

    /// The v1 /models list: the Hugging Face models of every machine.
    pub fn local_models(&self) -> AppResult<Vec<LocalModel>> {
        let snap = self.inner.view();
        let mut out: Vec<LocalModel> = snap
            .recs
            .iter()
            .filter(|r| r.fmt == "hf")
            .map(|r| LocalModel {
                repo_id: r.name.clone(),
                size_bytes: r.size,
                revisions: r
                    .entry
                    .get("revision")
                    .filter(|v| py::truthy(v))
                    .map(|v| vec![ec_proto::pyfmt::str_value(v)])
                    .unwrap_or_default(),
                last_used: f_(r.entry.get("atime")),
                path: r.path.clone(),
                engines_that_fit: Vec::new(),
                host_id: r.host_id.clone(),
            })
            .collect();
        out.sort_by(|a, b| {
            a.repo_id
                .cmp(&b.repo_id)
                .then_with(|| a.host_id.cmp(&b.host_id))
        });
        Ok(out)
    }

    /// Per host: when its library was last read, which generation, whether it is stale, the last error.
    pub fn status(&self) -> Map<String, Value> {
        let inner = &self.inner;
        let mut out = Map::new();
        for h in inner.fleet.hosts() {
            let (snap, err) = {
                let st = lock(&inner.state);
                (st.snaps.get(&h.id).cloned(), st.errors.get(&h.id).cloned())
            };
            out.insert(
                h.id.clone(),
                json!({"generation": snap.as_ref().and_then(|s| s.generation),
                       "updated_at": snap.as_ref().map(|s| s.updated_at),
                       "entries": snap.as_ref().map_or(0, |s| s.entries.len()),
                       "stale": inner.link_state(&h.id) != "up", "error": err}),
            );
        }
        out
    }

    /// The host's policy roots, delete switch, mode and mounts: live when its node has spoken, else stored.
    pub fn host_facts(&self, host_id: &str) -> Option<HostFacts> {
        self.inner.host_facts(host_id)
    }

    pub fn host_label(&self, host_id: &str) -> String {
        self.inner
            .fleet
            .node(host_id)
            .map_or_else(|_| host_id.to_string(), |n| n.host().label.clone())
    }

    pub fn root_usage(&self, host_id: &str, root_id: &str) -> RootUsage {
        self.inner.root_usage(host_id, root_id)
    }

    /// Free bytes on the drive holding `path`, from the node's `disks` (None when it has not reported).
    pub fn free_bytes(&self, host_id: &str, path: &str) -> Option<i64> {
        let facts = self.inner.host_facts(host_id)?;
        mount_for(&facts.mounts, &norm(path)).and_then(|m| m.free)
    }

    pub fn roots(&self, host_id: &str) -> Vec<RootInfo> {
        self.inner.roots(host_id)
    }

    pub fn root(&self, host_id: &str, root_id: &str) -> Option<RootInfo> {
        self.roots(host_id).into_iter().find(|r| r.id == root_id)
    }

    /// Ask the nodes to rescan now; the answer is the machines asked. For one named host a refusal is an error;
    /// for all hosts, unreachable ones are skipped.
    pub async fn rescan(&self, host_id: Option<&str>) -> AppResult<Vec<String>> {
        let inner = &self.inner;
        let targets: Vec<String> = match host_id {
            Some(h) => {
                inner.fleet.node(h)?;
                vec![h.to_string()]
            }
            None => inner
                .fleet
                .hosts()
                .into_iter()
                .map(|h| h.id)
                .filter(|h| inner.link_state(h) == "up")
                .collect(),
        };
        let mut done = Vec::new();
        for hid in targets {
            let node = inner.fleet.node(&hid)?;
            let mut args = Obj::new();
            args.insert("force".into(), Value::Bool(true));
            if let Err(e) = node
                .call("models.scan", Some(args), inner.call_timeout)
                .await
            {
                if host_id.is_some() {
                    return Err(problem_from_node(&e, Some(&node.host().label)).into());
                }
                tracing::info!(target: "engine_console.services.library", "library: rescan skipped {} ({})", hid, e.code);
                continue;
            }
            done.push(hid.clone());
            Inner::schedule(inner, &hid);
        }
        Ok(done)
    }

    /// Delete one copy (409 while an engine uses it, 403 when that node's policy forbids, 503 when it is down).
    pub async fn delete_copy(&self, copy_id: &str) -> AppResult<()> {
        let inner = &self.inner;
        let snap = inner.view();
        let Some(rec) = inner.rec(&snap, copy_id).cloned() else {
            return Err(not_found("no such model copy")
                .code("copy_not_found")
                .into());
        };
        let ctx = inner.ctx();
        let view = ctx.view(&rec.host_id).cloned();
        let label = view
            .as_ref()
            .map_or_else(|| rec.host_id.clone(), |v| v.label.clone());
        let deny = |code: &str, msg: String| -> AppError {
            problem_from_node(&NodeError::new(&rec.host_id, code, msg), Some(&label)).into()
        };
        let Some(view) = view.filter(|v| v.state == "up") else {
            let state = ctx
                .view(&rec.host_id)
                .map_or_else(|| "unknown".to_string(), |v| v.state.clone());
            return Err(deny(
                "node_unavailable",
                format!("the machine is not connected ({state})"),
            ));
        };
        let spec = view.facts.root(&rec.root_id);
        if !view.facts.allow_delete {
            return Err(deny(
                "policy_denied",
                "this machine's policy does not allow deleting models".into(),
            ));
        }
        if view.facts.mode != "manage" {
            return Err(deny(
                "monitor_only",
                "this machine's policy is monitor-only".into(),
            ));
        }
        if !spec.is_some_and(|s| s.writable) {
            return Err(deny(
                "policy_denied",
                "that model root is read-only in this machine's policy".into(),
            ));
        }
        let users = inner.users(&rec);
        if !users.is_empty() {
            return Err(conflict(format!(
                "the model is used by {}; stop it first",
                users.join(", ")
            ))
            .code("model_in_use")
            .into());
        }
        if rec.rel.is_empty() {
            return Err(conflict(
                "this entry is a whole model root; the node never deletes a root",
            )
            .code("root_not_deletable")
            .into());
        }
        let node = inner.fleet.node(&rec.host_id)?;
        let mut args = Obj::new();
        args.insert("root".into(), Value::String(rec.root_id.clone()));
        args.insert("rel".into(), Value::String(rec.rel.clone()));
        node.call("fs.delete", Some(args), Duration::from_secs(300))
            .await
            .map_err(|e| AppError::from(problem_from_node(&e, Some(&label))))?;
        inner.forget(&rec);
        // Everyone who could see those bytes has to look again: the owner's node and the machines mounting its
        // disk. No refetch is queued here: the node pushes `models_changed` when its scan sees the change.
        let configured: HashSet<String> = inner.fleet.hosts().into_iter().map(|h| h.id).collect();
        for hid in &rec.visible_on {
            if configured.contains(hid)
                && let Ok(n) = inner.fleet.node(hid)
            {
                let mut a = Obj::new();
                a.insert("force".into(), Value::Bool(true));
                let _ = n.call("models.scan", Some(a), inner.call_timeout).await;
            }
        }
        Ok(())
    }

    /// The facts the fit estimator needs, from the copy `host_id` would load (see `LibraryPort::model_info`).
    pub async fn model_info_of(&self, host_id: &str, r: &ModelRef) -> AppResult<Option<ModelInfo>> {
        Inner::model_info(&self.inner, host_id, r).await
    }

    /// `hf_root` (also on the LibraryPort).
    pub fn hf_root_of(&self, host_id: &str) -> Option<RootInfo> {
        for r in self.roots(host_id) {
            if r.kind == "hf-home" || r.kind == "hf-hub" {
                return Some(r);
            }
            if r.kind == "auto" {
                let u = self.root_usage(host_id, &r.id);
                if u.hf_home || u.hf_hub {
                    return Some(r);
                }
            }
        }
        None
    }

    pub fn ollama_root_of(&self, host_id: &str) -> Option<RootInfo> {
        self.roots(host_id).into_iter().find(|r| {
            r.kind == "ollama" || (r.kind == "auto" && self.root_usage(host_id, &r.id).ollama)
        })
    }

    /// The copies physically on `host_id`'s disks (a shared copy belongs to the machine that owns the disk).
    pub fn copies_of(&self, host_id: &str) -> Vec<LibraryCopy> {
        let snap = self.inner.view();
        let ctx = self.inner.ctx();
        snap.by_host
            .get(host_id)
            .map(|ids| {
                ids.iter()
                    .filter_map(|i| snap.rec(i))
                    .map(|r| self.inner.copy_view(r, &ctx, None, true))
                    .collect()
            })
            .unwrap_or_default()
    }
}

/// The way `host_id` reaches this copy: normally one; with two roots on the same storage, the root whose kind
/// naturally holds this format wins, then an `auto` root, then the first.
fn pick_alias(rec: &Rec, host_id: &str, ctx: &ReadCtx) -> Alias {
    let mine: Vec<&Alias> = rec
        .aliases
        .iter()
        .filter(|a| a.host_id == host_id)
        .collect();
    if mine.len() <= 1 {
        return mine
            .first()
            .map_or_else(|| rec.aliases[0].clone(), |a| (*a).clone());
    }
    let view = ctx.view(host_id);
    let rank = |a: &Alias| -> u8 {
        let kind = view
            .and_then(|v| v.facts.root(&a.root_id))
            .map_or("", |s| s.kind.as_str());
        if natural_kinds(&rec.fmt).contains(&kind) {
            0
        } else if kind == "auto" {
            1
        } else {
            2
        }
    };
    let mut best = mine[0];
    for a in &mine[1..] {
        if rank(a) < rank(best) {
            best = a;
        }
    }
    best.clone()
}

/// The way `host_id` reaches `rec`: the root a gguf or plain ref names, else the natural one.
fn alias_for(rec: &Rec, host_id: &str, r: &ModelRef, ctx: &ReadCtx) -> Alias {
    if (r.format == "gguf" || r.format == "plain")
        && let (Some(root), Some(rel)) = (
            r.root_id.as_deref().filter(|x| !x.is_empty()),
            r.rel.as_deref(),
        )
    {
        for a in &rec.aliases {
            if a.host_id == host_id && a.root_id == root && a.rel == rel {
                return a.clone();
            }
        }
    }
    pick_alias(rec, host_id, ctx)
}

impl Inner {
    fn view(&self) -> Arc<Snapshot> {
        Arc::clone(&lock(&self.state).view)
    }

    fn rec<'a>(&self, snap: &'a Snapshot, copy_id: &str) -> Option<&'a Rec> {
        if COPY_ID.is_match(copy_id) {
            snap.rec(copy_id)
        } else {
            None
        }
    }

    fn link_state(&self, host_id: &str) -> String {
        self.fleet
            .node(host_id)
            .map_or_else(|_| "disabled".to_string(), |n| n.link().state)
    }

    fn known(&self, host_id: Option<&Value>) -> Option<String> {
        let h = host_id?.as_str()?;
        self.fleet
            .hosts()
            .iter()
            .any(|x| x.id == h)
            .then(|| h.to_string())
    }

    fn on_changed(self: &Arc<Self>, ev: &Value) {
        if let Some(h) = self.known(ev.get("host_id"))
            && self.link_state(&h) == "up"
        {
            Self::schedule(self, &h);
        }
    }

    fn on_fleet(self: &Arc<Self>, ev: &Value) {
        let Some(h) = self.known(ev.get("id")) else {
            return;
        };
        let state = ev
            .get("link")
            .and_then(|l| l.get("state"))
            .and_then(Value::as_str);
        if state == Some("up") {
            let first = lock(&self.state).up.insert(h.clone());
            if first {
                Self::schedule(self, &h);
            }
        } else {
            lock(&self.state).up.remove(&h); // shown stale at read time; nothing to rebuild
        }
    }

    /// A host's `hello` or `disks` changed in a way that moves copies between drives or changes who shares them.
    fn on_facts(&self, ev: &Value) {
        let Some(h) = self.known(ev.get("host_id")) else {
            return;
        };
        if !lock(&self.state).snaps.contains_key(&h) {
            return;
        }
        let views = self.views();
        let Some((_, view)) = views.iter().find(|(id, _)| *id == h) else {
            return;
        };
        let sig = view.facts.signature();
        if lock(&self.state).sigs.get(&h) != Some(&sig) {
            self.rebuild();
        }
    }

    /// Ask for a refetch of one host. Requests that arrive while a fetch runs collapse into one more fetch.
    fn schedule(me: &Arc<Self>, host_id: &str) {
        let mut st = lock(&me.state);
        st.dirty.insert(host_id.to_string());
        let running = st.workers.get(host_id).is_some_and(|h| !h.is_finished());
        if !running {
            let me2 = Arc::clone(me);
            let h = host_id.to_string();
            let handle = tokio::spawn(async move { Self::worker(me2, h).await });
            st.workers.insert(host_id.to_string(), handle);
        }
    }

    async fn worker(me: Arc<Self>, host_id: String) {
        let mut failures = 0usize;
        loop {
            if !lock(&me.state).dirty.remove(&host_id) {
                return;
            }
            // a panicking fetch is logged (redacted) and counts as an unexpected error, as Python's except does
            let fetched = crate::panics::guarded("library: fetch", Self::fetch(&me, &host_id))
                .await
                .unwrap_or_else(|| {
                    Err(AppError::Problem(crate::domain::errors::internal(
                        "unexpected server error",
                    )))
                });
            match fetched {
                Ok(()) => {
                    failures = 0;
                    continue;
                }
                Err(AppError::Node(e)) => {
                    let msg = py::take(&format!("{}: {}", e.code, e.message), 300);
                    lock(&me.state).errors.insert(host_id.clone(), msg);
                    tracing::warn!(target: "engine_console.services.library", "library: fetching {} failed: {}", host_id, e.code);
                }
                Err(AppError::Problem(_)) => {
                    lock(&me.state).errors.insert(
                        host_id.clone(),
                        "unexpected error while reading the node's library".into(),
                    );
                    tracing::error!(target: "engine_console.services.library", "library: fetching {} failed", host_id);
                }
            }
            failures += 1;
            if me.link_state(&host_id) != "up" {
                return; // fetched again when the link comes back up
            }
            let d = me.retry_delays[(failures - 1).min(me.retry_delays.len() - 1)];
            tokio::time::sleep(Duration::from_secs_f64(d.max(0.0))).await;
            lock(&me.state).dirty.insert(host_id.clone());
        }
    }

    fn host_lock(&self, host_id: &str) -> Arc<tokio::sync::Mutex<()>> {
        Arc::clone(
            lock(&self.host_locks)
                .entry(host_id.to_string())
                .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(()))),
        )
    }

    async fn fetch(me: &Arc<Self>, host_id: &str) -> AppResult<()> {
        let node = me.fleet.node(host_id)?;
        let hl = me.host_lock(host_id);
        let (generation, count) = {
            let _g = hl.lock().await;
            let (entries, scans, generation) = me.list_all(node.as_ref(), host_id).await?;
            let prev = lock(&me.state).snaps.get(host_id).cloned();
            let facts = parse_facts(
                node.latest("hello").as_deref(),
                node.latest("disks").as_deref(),
                prev.as_ref().map(|p| &p.facts),
            )
            .unwrap_or_default();
            let snap = Arc::new(HostSnapshot {
                generation,
                entries,
                scans,
                facts,
                updated_at: now(),
            });
            let count = snap.entries.len();
            {
                let mut st = lock(&me.state);
                st.snaps.insert(host_id.to_string(), Arc::clone(&snap));
                st.errors.remove(host_id);
                let prefix = format!("{host_id}|");
                st.info_cache.retain(|k, _| !k.0.starts_with(&prefix));
            }
            me.persist(host_id, &snap)?;
            me.rebuild();
            (generation, count)
        };
        me.bus.publish(
            LIBRARY_TOPIC,
            json!({"host_id": host_id, "generation": generation, "models": count}),
        );
        Ok(())
    }

    /// Every page of `models.list`. A scan finishing between two pages would mix two generations, so a pass that
    /// sees the generation change starts over (twice at most, then the mix is accepted).
    async fn list_all(
        &self,
        node: &dyn NodeClient,
        host_id: &str,
    ) -> AppResult<(Vec<Obj>, Vec<Obj>, Option<i64>)> {
        for attempt in 0..3 {
            if let Some(out) = self.one_pass(node, host_id, attempt == 2).await? {
                return Ok(out);
            }
        }
        Err(crate::domain::errors::internal("unexpected server error").into())
    }

    async fn one_pass(
        &self,
        node: &dyn NodeClient,
        host_id: &str,
        accept_change: bool,
    ) -> AppResult<Option<(Vec<Obj>, Vec<Obj>, Option<i64>)>> {
        let mut entries: Vec<Obj> = Vec::new();
        let mut scans: Vec<Obj> = Vec::new();
        let mut generation: Option<i64> = None;
        let mut cursor: i64 = 0;
        let pages = usize::try_from(i64::try_from(MAX_ENTRIES).unwrap_or(0) / self.page_limit + 2)
            .unwrap_or(2);
        for _ in 0..pages {
            let mut args = Obj::new();
            args.insert("limit".into(), Value::from(self.page_limit));
            if cursor != 0 {
                args.insert("cursor".into(), Value::from(cursor));
            }
            let res = node
                .call("models.list", Some(args), self.call_timeout)
                .await?;
            let gen_ = i_(res.get("generation"));
            if generation.is_none() {
                generation = gen_;
            } else if gen_.is_some() && gen_ != generation && !accept_change {
                return Ok(None);
            }
            if let Some(Value::Array(es)) = res.get("entries") {
                entries.extend(es.iter().filter_map(|e| e.as_object().cloned()));
            }
            if scans.is_empty()
                && let Some(Value::Array(rs)) = res.get("roots")
            {
                scans = rs.iter().filter_map(|e| e.as_object().cloned()).collect();
            }
            let nxt_raw = res.get("next_cursor");
            if nxt_raw.is_none_or(Value::is_null) {
                return Ok(Some((entries, scans, generation)));
            }
            match i_(nxt_raw) {
                Some(n) if n > cursor => cursor = n,
                _ => {
                    tracing::warn!(target: "engine_console.services.library", "library: {} sent a page cursor that does not advance; keeping what it sent", host_id);
                    return Ok(Some((entries, scans, generation)));
                }
            }
            if entries.len() >= MAX_ENTRIES {
                break;
            }
        }
        tracing::warn!(target: "engine_console.services.library", "library: {} has more than {} entries; the rest are not listed", host_id, MAX_ENTRIES);
        entries.truncate(MAX_ENTRIES);
        Ok(Some((entries, scans, generation)))
    }

    // ---- persistence ------------------------------------------------------------------------------------------
    fn persist(&self, host_id: &str, snap: &HostSnapshot) -> AppResult<()> {
        let entries = Value::Array(snap.entries.iter().cloned().map(Value::Object).collect());
        let scans = Value::Array(snap.scans.iter().cloned().map(Value::Object).collect());
        self.db.execute(
            "INSERT INTO library_snapshots(host_id,generation,roots_json,entries_json,scans_json,facts_json,updated_at) \
             VALUES(?,?,?,?,?,?,?) ON CONFLICT(host_id) DO UPDATE SET generation=excluded.generation, \
             roots_json=excluded.roots_json, entries_json=excluded.entries_json, scans_json=excluded.scans_json, \
             facts_json=excluded.facts_json, updated_at=excluded.updated_at",
            &[
                &host_id,
                &snap.generation,
                &crate::pyjson::dumps(&snap.facts.roots_json()),
                &crate::pyjson::dumps(&entries),
                &crate::pyjson::dumps(&scans),
                &crate::pyjson::dumps(&snap.facts.to_json()),
                &snap.updated_at,
            ],
        )?;
        Ok(())
    }

    fn load(&self, known: &HashSet<String>) {
        let Ok(rows) = self.db.all("SELECT * FROM library_snapshots", &[]) else {
            return;
        };
        for r in rows {
            let host_id = r.str_or("host_id", "");
            if !known.contains(&host_id) {
                let _ = self
                    .db
                    .execute("DELETE FROM library_snapshots WHERE host_id=?", &[&host_id]);
                continue;
            }
            let parse = |col: &str| -> Option<Value> { serde_json::from_str(&r.text(col)?).ok() };
            let (Some(e), Some(s), Some(roots), Some(facts)) = (
                parse("entries_json"),
                parse("scans_json"),
                parse("roots_json"),
                parse("facts_json"),
            ) else {
                tracing::warn!(target: "engine_console.services.library", "library: dropping an unreadable stored snapshot of {}", host_id);
                let _ = self
                    .db
                    .execute("DELETE FROM library_snapshots WHERE host_id=?", &[&host_id]);
                continue;
            };
            let objs = |v: Value| -> Vec<Obj> {
                match v {
                    Value::Array(a) => a
                        .into_iter()
                        .filter_map(|x| x.as_object().cloned())
                        .collect(),
                    _ => Vec::new(),
                }
            };
            let entries = objs(ec_proto::clean(
                &e,
                &ec_proto::clean::l(ec_proto::specs::model_entry(), MAX_ENTRIES),
            ));
            let scans = objs(ec_proto::clean(
                &s,
                &ec_proto::clean::l(ec_proto::specs::root_scan(), 32),
            ));
            let facts = HostFacts::from_json(&roots, &facts);
            let snap = HostSnapshot {
                generation: r.get("generation").and_then(|v| match v {
                    rusqlite::types::Value::Integer(i) => Some(*i),
                    _ => None,
                }),
                entries,
                scans,
                facts,
                updated_at: r.f64("updated_at").unwrap_or(0.0),
            };
            lock(&self.state).snaps.insert(host_id, Arc::new(snap));
        }
    }

    // ---- building the merged view -------------------------------------------------------------------------------
    fn views(&self) -> Vec<(String, HostView)> {
        let snaps = lock(&self.state).snaps.clone();
        let mut views = Vec::new();
        for h in self.fleet.hosts() {
            let Ok(node) = self.fleet.node(&h.id) else {
                continue;
            };
            let snap = snaps.get(&h.id);
            let Some(facts) = parse_facts(
                node.latest("hello").as_deref(),
                node.latest("disks").as_deref(),
                snap.map(|s| &s.facts),
            ) else {
                continue;
            };
            views.push((
                h.id.clone(),
                HostView {
                    id: h.id.clone(),
                    label: h.label.clone(),
                    state: node.link().state,
                    facts,
                },
            ));
        }
        views
    }

    fn host_facts(&self, host_id: &str) -> Option<HostFacts> {
        let node = self.fleet.node(host_id).ok()?;
        let snap = lock(&self.state).snaps.get(host_id).cloned();
        parse_facts(
            node.latest("hello").as_deref(),
            node.latest("disks").as_deref(),
            snap.as_ref().map(|s| &s.facts),
        )
    }

    fn make_rec(view: &HostView, entry: Obj) -> Option<Rec> {
        let root_id = s_(entry.get("root"), 32)?;
        let spec = view.facts.root(&root_id)?;
        let rel = entry.get("rel").and_then(Value::as_str)?.to_string();
        let rel = if rel.is_empty() || rel == "." {
            String::new()
        } else {
            rel
        };
        if !rel.is_empty() && !rel_ok(&rel) {
            return None; // a node-supplied path that breaks the node's own rules is not listed
        }
        let fmt = fmt_of(&entry);
        let name = derive_name(&fmt, &entry, &rel);
        let (key, _) = entry_key(&fmt, &name, &entry, &rel);
        let path = if rel.is_empty() {
            spec.path.clone()
        } else {
            norm(&format!("{}/{rel}", spec.path))
        };
        let size = i_(entry.get("size")).unwrap_or(0).max(0);
        let cid = copy_id(&view.id, &root_id, &rel);
        let base = LibraryCopy {
            id: cid.clone(),
            host_id: view.id.clone(),
            host_label: view.label.clone(),
            root_id: root_id.clone(),
            root_label: spec.label.clone(),
            rel: rel.clone(),
            path: path.clone(),
            size_bytes: size,
            files: i_(entry.get("files")),
            mtime: f_(entry.get("mtime")),
            atime: f_(entry.get("atime")),
            drive: None,
            visible_on: vec![view.id.clone()],
            writable: spec.writable,
            deletable: false,
            in_use_by: Vec::new(),
        };
        let mount = mount_for(&view.facts.mounts, &path).cloned();
        Some(Rec {
            id: cid,
            host_id: view.id.clone(),
            root_id: root_id.clone(),
            rel: rel.clone(),
            path: path.clone(),
            fmt,
            name,
            key,
            entry,
            size,
            mount,
            base,
            visible_on: vec![view.id.clone()],
            aliases: vec![Alias {
                host_id: view.id.clone(),
                root_id,
                rel,
                path,
            }],
        })
    }

    /// Fold copies on an NFS mount into the copy on the disk that exports it: the NFS server must be one of a
    /// configured host's addresses, and the copy's path on that host must be a copy that host lists. Anything else
    /// stays a copy of its own (in its place).
    fn merge_shared(mut recs: Vec<Rec>, book: &AddressBook) -> Vec<Rec> {
        let nfs = |r: &Rec| r.mount.as_ref().is_some_and(Mount::nfs);
        let mut owners: HashMap<(String, String), usize> = HashMap::new();
        for (i, r) in recs.iter().enumerate() {
            if !nfs(r) {
                owners.insert((r.host_id.clone(), norm(&r.path)), i); // a later one wins, as in a dict
            }
        }
        let mut merged_away: HashSet<usize> = HashSet::new();
        for j in 0..recs.len() {
            if !nfs(&recs[j]) {
                continue;
            }
            let r = &recs[j];
            let m = r.mount.clone().unwrap_or_default();
            let mut target: Option<usize> = None;
            if let (Some(owner), Some(rp)) = (
                book.find(m.remote_host.as_deref(), &r.host_id),
                m.remote_path.as_ref(),
            ) && under(&r.path, &m.mount)
            {
                let sub = py::relpath(&r.path, &m.mount);
                let equivalent = if sub == "." {
                    rp.clone()
                } else {
                    py::normpath(&py::join(rp, &sub))
                };
                target = owners.get(&(owner, equivalent)).copied();
            }
            let Some(t) = target else {
                continue; // the owner is not a configured host (or does not list it): this copy stays
            };
            let (host, alias) = (
                r.host_id.clone(),
                Alias {
                    host_id: r.host_id.clone(),
                    root_id: r.root_id.clone(),
                    rel: r.rel.clone(),
                    path: r.path.clone(),
                },
            );
            let tr = &mut recs[t];
            if !tr.visible_on.contains(&host) {
                tr.visible_on.push(host);
            }
            tr.aliases.push(alias);
            merged_away.insert(j);
        }
        recs.into_iter()
            .enumerate()
            .filter(|(i, _)| !merged_away.contains(i))
            .map(|(_, r)| r)
            .collect()
    }

    fn rebuild(&self) {
        let views = self.views();
        let order: HashMap<String, usize> = self
            .fleet
            .hosts()
            .into_iter()
            .enumerate()
            .map(|(n, h)| (h.id, n))
            .collect();
        let book = AddressBook::new(&views);
        let snaps = lock(&self.state).snaps.clone();
        let mut sigs: Vec<(String, Signature)> = Vec::new();
        let mut recs: Vec<Rec> = Vec::new();
        for (host_id, snap) in &snaps {
            let Some((_, view)) = views.iter().find(|(h, _)| h == host_id) else {
                continue;
            };
            sigs.push((host_id.clone(), view.facts.signature()));
            for raw in &snap.entries {
                for entry in expand_gguf_repo(raw) {
                    if let Some(rec) = Self::make_rec(view, entry) {
                        recs.push(rec);
                    }
                }
            }
        }
        recs.sort_by(|a, b| {
            let oa = order.get(&a.host_id).copied().unwrap_or(1 << 30);
            let ob = order.get(&b.host_id).copied().unwrap_or(1 << 30);
            oa.cmp(&ob)
                .then_with(|| a.root_id.cmp(&b.root_id))
                .then_with(|| a.rel.cmp(&b.rel))
        });
        let kept = Self::merge_shared(recs, &book);

        let mut merged = Snapshot::default();
        let mut by_key: Vec<(String, Vec<usize>)> = Vec::new();
        let mut key_pos: HashMap<String, usize> = HashMap::new();
        for r in kept {
            let idx = merged.recs.len();
            merged.rec_idx.insert(r.id.clone(), idx);
            merged
                .by_host
                .entry(r.host_id.clone())
                .or_default()
                .push(r.id.clone());
            for a in &r.aliases {
                merged.alias.insert(
                    (a.host_id.clone(), a.root_id.clone(), a.rel.clone()),
                    r.id.clone(),
                );
            }
            if r.fmt == "hf" {
                merged
                    .hf_by_name
                    .entry(r.name.clone())
                    .or_default()
                    .push(r.id.clone());
            } else if r.fmt == "ollama" {
                merged
                    .ollama_by_name
                    .entry(r.name.clone())
                    .or_default()
                    .push(r.id.clone());
            }
            match key_pos.get(&r.key) {
                Some(p) => by_key[*p].1.push(idx),
                None => {
                    key_pos.insert(r.key.clone(), by_key.len());
                    by_key.push((r.key.clone(), vec![idx]));
                }
            }
            merged.recs.push(r);
        }
        for (key, members) in by_key {
            let model = self.model_of(&key, &members, &merged, &snaps, &views);
            merged.model_idx.insert(key.clone(), merged.models.len());
            merged.members.insert(
                key.clone(),
                members.iter().map(|i| merged.recs[*i].id.clone()).collect(),
            );
            merged.models.push((key, model));
        }
        let mut st = lock(&self.state);
        for (h, s) in sigs {
            st.sigs.insert(h, s);
        }
        st.view = Arc::new(merged);
    }

    fn engines_for(&self, fmt: &str) -> Vec<String> {
        if !matches!(fmt, "hf" | "gguf" | "ollama" | "plain") {
            return Vec::new();
        }
        self.adapters
            .list()
            .iter()
            .filter(|a| a.model_formats().iter().any(|f| f == fmt))
            .map(|a| a.id().to_string())
            .collect()
    }

    fn model_of(
        &self,
        key: &str,
        members: &[usize],
        snap: &Snapshot,
        snaps: &HashMap<String, Arc<HostSnapshot>>,
        views: &[(String, HostView)],
    ) -> LibraryModel {
        let recs: Vec<&Rec> = members.iter().map(|i| &snap.recs[*i]).collect();
        let first = recs[0];
        let pick_str = |field: &str| -> Option<String> {
            recs.iter().find_map(|r| match r.entry.get(field) {
                Some(Value::String(s)) if !s.is_empty() => Some(s.clone()),
                _ => None,
            })
        };
        let pick_int = |field: &str| -> Option<i64> {
            recs.iter()
                .find_map(|r| r.entry.get(field).and_then(py::as_int).filter(|v| *v != 0))
        };
        let mut warnings: Vec<String> = Vec::new();
        for r in &recs {
            if let Some(Value::Array(ws)) = r.entry.get("warnings") {
                for w in ws {
                    if let Some(w) = w.as_str().filter(|w| !w.is_empty())
                        && !warnings.iter().any(|x| x == w)
                    {
                        warnings.push(w.to_string());
                    }
                }
            }
        }
        let revisions: BTreeSet<String> = recs
            .iter()
            .filter_map(|r| r.entry.get("revision").filter(|v| py::truthy(v)))
            .map(ec_proto::pyfmt::str_value)
            .collect();
        if revisions.len() > 1 {
            let mut short: Vec<String> = revisions.iter().map(|v| py::take(v, 12)).collect();
            short.sort();
            warnings.push(format!(
                "copies are at different revisions: {}",
                short.join(", ")
            ));
        }
        for r in &recs {
            let Some(hs) = snaps.get(&r.host_id) else {
                continue;
            };
            for scan in &hs.scans {
                let err = scan.get("error").filter(|e| py::truthy(e));
                if scan.get("root").and_then(Value::as_str) == Some(r.root_id.as_str())
                    && (scan.get("complete") == Some(&Value::Bool(false)) || err.is_some())
                {
                    let label = views
                        .iter()
                        .find(|(h, _)| *h == r.host_id)
                        .map_or_else(|| r.host_id.clone(), |(_, v)| v.label.clone());
                    let why = err
                        .map(|e| format!(": {}", ec_proto::pyfmt::str_value(e)))
                        .unwrap_or_default();
                    let w = format!(
                        "{label}: the scan of root {} did not finish{why}",
                        r.root_id
                    );
                    if !warnings.contains(&w) {
                        warnings.push(w);
                    }
                }
            }
        }
        warnings.truncate(MAX_WARNINGS);
        LibraryModel {
            key: key.to_string(),
            name: first.name.clone(),
            format: first.fmt.clone(),
            revision: pick_str("revision"),
            quant: entry_key(&first.fmt, &first.name, &first.entry, &first.rel).1,
            params: pick_int("params"),
            arch: pick_str("arch"),
            context: pick_int("context"),
            shards: pick_int("shards"),
            size_bytes: recs.iter().map(|r| r.size).max().unwrap_or(0),
            engines: self.engines_for(&first.fmt),
            copies: Vec::new(),
            warnings,
        }
    }

    // ---- reading (live overlay) --------------------------------------------------------------------------------
    fn ctx(&self) -> ReadCtx {
        let views = self.views();
        let order = self
            .fleet
            .hosts()
            .into_iter()
            .enumerate()
            .map(|(n, h)| (h.id, n))
            .collect();
        let book = AddressBook::new(&views);
        ReadCtx {
            views,
            order,
            book,
            drives: Mutex::new(HashMap::new()),
        }
    }

    /// Instances using this copy on any machine that can see it (the owner's path and every NFS alias).
    fn users(&self, rec: &Rec) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for a in &rec.aliases {
            let f = Arc::clone(&self.in_use);
            let ids = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(&a.host_id, &a.path)))
                .unwrap_or_else(|_| {
                    tracing::error!(target: "engine_console.services.library", "library: the in_use callback failed for {}", a.host_id);
                    Vec::new()
                });
            for i in ids {
                if !out.contains(&i) {
                    out.push(i);
                }
            }
        }
        out
    }

    /// The drive a path lives on, with the free space the node last reported.
    fn drive_of(host_id: &str, mount: Option<&Mount>, ctx: &ReadCtx) -> Option<MountInfo> {
        let mount = mount?;
        let key = (host_id.to_string(), mount.mount.clone());
        if let Some(d) = lock(&ctx.drives).get(&key) {
            return d.clone();
        }
        let live = ctx
            .view(host_id)
            .and_then(|v| v.facts.mounts.iter().find(|m| m.mount == mount.mount));
        let m = live.unwrap_or(mount);
        let owner = if m.remote {
            ctx.book.find(m.remote_host.as_deref(), host_id)
        } else {
            None
        };
        let info = Some(mount_info(m, owner, &ctx.views));
        lock(&ctx.drives).insert(key, info.clone());
        info
    }

    /// The copy as the API shows it; with `alias`, in that machine's terms (the id stays the owner's).
    fn copy_view(
        &self,
        rec: &Rec,
        ctx: &ReadCtx,
        alias: Option<&Alias>,
        with_users: bool,
    ) -> LibraryCopy {
        let view = ctx.view(&rec.host_id);
        let spec = view.and_then(|v| v.facts.root(&rec.root_id));
        // with_users=false is for the lifecycle, which only needs where the copy is
        let users = if with_users {
            self.users(rec)
        } else {
            Vec::new()
        };
        let writable = spec.is_some_and(|s| s.writable);
        let deletable = view.is_some_and(|v| {
            v.state == "up"
                && writable
                && v.facts.allow_delete
                && v.facts.mode == "manage"
                && users.is_empty()
                && !rec.rel.is_empty()
        });
        let mut out = rec.base.clone();
        out.writable = writable;
        out.deletable = deletable;
        out.in_use_by = users;
        out.drive = Self::drive_of(&rec.host_id, rec.mount.as_ref(), ctx);
        out.visible_on = rec.visible_on.clone();
        let Some(alias) = alias.filter(|a| a.host_id != rec.host_id) else {
            return out;
        };
        let there = ctx.view(&alias.host_id);
        let there_spec = there.and_then(|t| t.facts.root(&alias.root_id));
        let there_mount = there.and_then(|t| mount_for(&t.facts.mounts, &alias.path));
        out.host_id = alias.host_id.clone();
        out.host_label = there.map_or_else(|| alias.host_id.clone(), |t| t.label.clone());
        out.root_id = alias.root_id.clone();
        out.root_label = there_spec.and_then(|s| s.label.clone());
        out.rel = alias.rel.clone();
        out.path = alias.path.clone();
        out.writable = there_spec.is_some_and(|s| s.writable);
        out.drive = Self::drive_of(&alias.host_id, there_mount, ctx);
        out
    }

    fn model_view(&self, snap: &Snapshot, key: &str, ctx: &ReadCtx) -> LibraryModel {
        let Some(i) = snap.model_idx.get(key) else {
            return LibraryModel {
                key: key.into(),
                name: String::new(),
                format: "other".into(),
                revision: None,
                quant: None,
                params: None,
                arch: None,
                context: None,
                shards: None,
                size_bytes: 0,
                engines: Vec::new(),
                copies: Vec::new(),
                warnings: Vec::new(),
            };
        };
        let base = &snap.models[*i].1;
        let copies: Vec<LibraryCopy> = snap.members[key]
            .iter()
            .filter_map(|id| snap.rec(id))
            .map(|r| self.copy_view(r, ctx, None, true))
            .collect();
        let mut warnings = base.warnings.clone();
        let mut hosts: Vec<String> = Vec::new();
        for c in &copies {
            if !hosts.contains(&c.host_id) {
                hosts.push(c.host_id.clone());
            }
        }
        hosts.sort_by_key(|h| ctx.order.get(h).copied().unwrap_or(1 << 30));
        for h in hosts {
            if let Some(v) = ctx.view(&h)
                && v.state != "up"
                && v.state != "stale"
            {
                warnings.push(format!(
                    "{} is {}: showing its last-known library, which cannot be changed until it reconnects",
                    v.label, v.state
                ));
            }
        }
        warnings.truncate(MAX_WARNINGS);
        let mut m = base.clone();
        m.copies = copies;
        m.warnings = warnings;
        m
    }

    /// The copy `host_id` would load for `ref`. `own` keeps only copies on `host_id`'s own disks.
    fn find_rec<'a>(
        &self,
        snap: &'a Snapshot,
        host_id: &str,
        r: &ModelRef,
        own: bool,
    ) -> Option<&'a Rec> {
        let mut ids: Vec<String> = Vec::new();
        let nonempty =
            |s: &Option<String>| s.as_deref().filter(|x| !x.is_empty()).map(str::to_string);
        if r.format == "hf"
            && let Some(repo) = nonempty(&r.repo_id)
        {
            // a repo directory with no snapshot is listed, with the node's warning, but is not loadable
            ids = snap
                .hf_by_name
                .get(&repo)
                .map(|v| {
                    v.iter()
                        .filter(|i| snap.rec(i).is_some_and(|x| loadable_hf(&x.entry)))
                        .cloned()
                        .collect()
                })
                .unwrap_or_default();
        } else if r.format == "ollama"
            && let Some(name) = nonempty(&r.name)
        {
            ids = snap.ollama_by_name.get(&name).cloned().unwrap_or_default();
        } else if (r.format == "gguf" || r.format == "plain")
            && let (Some(root), Some(rel)) = (nonempty(&r.root_id), r.rel.as_ref())
        {
            match snap
                .alias
                .get(&(host_id.to_string(), root.clone(), rel.clone()))
            {
                Some(id) => ids = vec![id.clone()],
                // the owner's coordinates, for a host that reaches the copy over shared storage
                None => {
                    ids = snap
                        .recs
                        .iter()
                        .filter(|x| x.root_id == root && &x.rel == rel)
                        .map(|x| x.id.clone())
                        .collect();
                }
            }
            ids.retain(|i| snap.rec(i).is_some_and(|x| x.fmt == r.format));
        }
        let cands: Vec<&Rec> = ids
            .iter()
            .filter_map(|i| snap.rec(i))
            .filter(|x| x.visible_on.iter().any(|h| h == host_id) && (!own || x.host_id == host_id))
            .collect();
        cands
            .iter()
            .find(|x| x.host_id == host_id)
            .or_else(|| cands.first())
            .copied()
    }

    fn root_usage(&self, host_id: &str, root_id: &str) -> RootUsage {
        let snap = lock(&self.state).snaps.get(host_id).cloned();
        let mut u = RootUsage::default();
        for e in snap.as_ref().map(|s| s.entries.as_slice()).unwrap_or(&[]) {
            if e.get("root").and_then(Value::as_str) != Some(root_id) {
                continue;
            }
            u.entries += 1;
            let rel = e.get("rel").and_then(Value::as_str).unwrap_or("");
            match fmt_of(e).as_str() {
                "hf" => {
                    u.hf_home |= rel.starts_with("hub/");
                    u.hf_hub |= rel.starts_with("models--");
                }
                "gguf" | "plain" => u.loose = true,
                "ollama" => u.ollama = true,
                _ => u.other = true,
            }
        }
        u
    }

    fn roots(&self, host_id: &str) -> Vec<RootInfo> {
        let ctx = self.ctx();
        let Some(view) = ctx.view(host_id) else {
            return Vec::new();
        };
        let sh = shares(&ctx.views, &ctx.book, &ctx.order);
        let mut out = Vec::new();
        for spec in &view.facts.roots {
            let m = mount_of_root(spec, &view.facts.mounts);
            let (shared_from, visible) = sh
                .get(&(host_id.to_string(), spec.id.clone()))
                .cloned()
                .unwrap_or_else(|| (None, vec![host_id.to_string()]));
            let owner = match m {
                Some(m) if m.remote => ctx.book.find(m.remote_host.as_deref(), host_id),
                _ => None,
            };
            out.push(RootInfo {
                id: spec.id.clone(),
                path: spec.path.clone(),
                kind: spec.kind.clone(),
                writable: spec.writable,
                label: spec.label.clone(),
                mount: m.map(|m| mount_info(m, owner, &ctx.views)),
                shared_from,
                visible_on: visible,
            });
        }
        out
    }

    // ---- model facts ---------------------------------------------------------------------------------------------
    async fn model_info(
        me: &Arc<Self>,
        host_id: &str,
        r: &ModelRef,
    ) -> AppResult<Option<ModelInfo>> {
        let snap = me.view();
        let Some(rec) = me.find_rec(&snap, host_id, r, false).cloned() else {
            return Ok(None);
        };
        if rec.fmt == "other" {
            return Ok(None);
        }
        let generation = lock(&me.state)
            .snaps
            .get(&rec.host_id)
            .and_then(|s| s.generation);
        let ck = (
            format!("{}|{}", rec.host_id, rec.id),
            generation,
            rec.fmt.clone(),
        );
        if let Some(c) = lock(&me.state).info_cache.get(&ck) {
            return Ok(Some(c.clone()));
        }
        let revision = s_(rec.entry.get("revision"), 64);
        let info = if rec.fmt == "gguf" || rec.fmt == "ollama" {
            let mut entry = rec.entry.clone();
            let meta = match entry.get("meta") {
                Some(Value::Object(m)) => m.clone(),
                _ => Map::new(),
            };
            if rec.fmt == "gguf" && !meta.contains_key("block_count") {
                // A quant expanded from an HF repo carries no header facts of its own: read that file's header
                // through the node (models.config on a .gguf answers {"gguf": {...}}).
                let cfg = me.config_of(&rec).await;
                if let Some(Value::Object(header)) = cfg.as_ref().and_then(|c| c.get("gguf")) {
                    let or = |k: &str| -> Value {
                        match entry.get(k) {
                            Some(v) if py::truthy(v) => v.clone(),
                            _ => header.get(k).cloned().unwrap_or(Value::Null),
                        }
                    };
                    let (arch, context, params) = (or("arch"), or("context"), or("params"));
                    let mut m2 = meta.clone();
                    for k in GGUF_HEADER_KEYS {
                        if let Some(v) = header.get(*k).filter(|v| !v.is_null()) {
                            m2.insert((*k).to_string(), v.clone());
                        }
                    }
                    entry.insert("arch".into(), arch);
                    entry.insert("context".into(), context);
                    entry.insert("params".into(), params);
                    entry.insert("meta".into(), Value::Object(m2));
                }
            }
            model_info_from_entry(&rec.name, &entry, &rec.fmt, Some(rec.size))
        } else {
            let cfg = me.config_of(&rec).await;
            let hub = HubModel {
                repo_id: rec.name.clone(),
                sha: revision.clone(),
                gated: false,
                license: None,
                pipeline_tag: None,
                library_name: None,
                tags: Vec::new(),
                files: Vec::new(),
                safetensors_params: Vec::new(),
                safetensors_total: i_(rec.entry.get("params")),
                config: cfg,
                downloads: None,
                likes: None,
            };
            let mut info = build_model_info(&hub, revision.as_deref())?;
            if info.weight_bytes.is_none() && rec.size != 0 {
                info.weight_bytes = Some(rec.size);
            }
            if info.quantization.is_none()
                && let Some(q) = normalize_quant(s_(rec.entry.get("quant"), 32).as_deref())
            {
                info.quantization = Some(q);
            }
            info
        };
        lock(&me.state).info_cache.insert(ck, info.clone());
        Ok(Some(info))
    }

    /// models.config from the owner's node, else from a machine that reaches the copy over shared storage.
    async fn config_of(&self, rec: &Rec) -> Option<Obj> {
        for a in &rec.aliases {
            let Ok(node) = self.fleet.node(&a.host_id) else {
                continue;
            };
            if node.link().state != "up" {
                continue;
            }
            let mut args = Obj::new();
            args.insert("root".into(), Value::String(a.root_id.clone()));
            args.insert("rel".into(), Value::String(a.rel.clone()));
            match node
                .call("models.config", Some(args), self.call_timeout)
                .await
            {
                Ok(res) => {
                    if let Some(Value::Object(cfg)) = res.get("config") {
                        return Some(cfg.clone());
                    }
                }
                Err(e) => {
                    if e.code != "not_found" {
                        tracing::info!(target: "engine_console.services.library", "library: models.config on {} answered {}", a.host_id, e.code);
                    }
                }
            }
        }
        None
    }

    /// Drop a deleted copy from every host snapshot that listed it, so the library shows the deletion at once.
    fn forget(&self, rec: &Rec) {
        let mut touched: Vec<(String, Arc<HostSnapshot>)> = Vec::new();
        {
            let mut st = lock(&self.state);
            for a in &rec.aliases {
                let Some(snap) = st.snaps.get(&a.host_id).cloned() else {
                    continue;
                };
                let kept: Vec<Obj> = snap
                    .entries
                    .iter()
                    .filter(|e| {
                        let rel = match e.get("rel") {
                            None | Some(Value::Null) => String::new(),
                            Some(Value::String(s)) if s == "." => String::new(),
                            Some(Value::String(s)) => s.clone(),
                            Some(other) => other.to_string(),
                        };
                        !(e.get("root").and_then(Value::as_str) == Some(a.root_id.as_str())
                            && rel == a.rel)
                    })
                    .cloned()
                    .collect();
                if kept.len() != snap.entries.len() {
                    let mut s2 = (*snap).clone();
                    s2.entries = kept;
                    let s2 = Arc::new(s2);
                    st.snaps.insert(a.host_id.clone(), Arc::clone(&s2));
                    touched.push((a.host_id.clone(), s2));
                }
            }
        }
        if !touched.is_empty() {
            self.rebuild();
            for (h, s) in touched {
                let _ = self.persist(&h, &s);
            }
        }
    }
}

#[async_trait]
impl LibraryPort for LibraryService {
    fn find(&self, host_id: &str, r: &ModelRef) -> Option<LibraryCopy> {
        self.find_copy(host_id, r, false)
    }
    /// `find` without `in_use_by` (empty) and with `deletable` not to be trusted: where the copy is and how
    /// `host_id` reaches it, for the lifecycle's launches and locations.
    fn locate(&self, host_id: &str, r: &ModelRef) -> Option<LibraryCopy> {
        let snap = self.inner.view();
        let rec = self.inner.find_rec(&snap, host_id, r, false)?;
        let ctx = self.inner.ctx();
        let alias = alias_for(rec, host_id, r, &ctx);
        Some(self.inner.copy_view(rec, &ctx, Some(&alias), false))
    }
    async fn model_info(&self, host_id: &str, r: &ModelRef) -> AppResult<Option<ModelInfo>> {
        self.model_info_of(host_id, r).await
    }
    fn hf_root(&self, host_id: &str) -> Option<RootInfo> {
        self.hf_root_of(host_id)
    }
    fn ollama_root(&self, host_id: &str) -> Option<RootInfo> {
        self.ollama_root_of(host_id)
    }
    fn copies(&self, host_id: &str) -> Vec<LibraryCopy> {
        self.copies_of(host_id)
    }
}

/// The problem a missing library copy is (tests and the downloads service share it).
pub fn copy_not_found() -> Problem {
    not_found("no such model copy").code("copy_not_found")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pure_helpers() {
        assert_eq!(norm("//a//b/"), "/a/b");
        assert_eq!(norm(""), "");
        assert!(under("/data/hf/x", "/data/hf") && !under("/data/hf2", "/data/hf"));
        assert!(rel_ok("a/b") && !rel_ok("a//b") && !rel_ok("../x") && !rel_ok("/x"));
        assert_eq!(canon_addr(" [FE80::1%eth0] "), "fe80::1");
        assert_eq!(canon_addr("Alpha.LAN"), "alpha.lan");
        assert!(is_loopback("127.0.0.2") && is_loopback("localhost") && !is_loopback("10.0.0.1"));
        let e: Obj = json!({"rel": "x/models--a--b\n"})
            .as_object()
            .cloned()
            .unwrap_or_default();
        assert_eq!(derive_name("hf", &e, "x/models--a--b\n"), "a/b");
    }
}
