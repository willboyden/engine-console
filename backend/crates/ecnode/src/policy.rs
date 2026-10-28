//! Python: node/src/ecnode/policy.py.
//!
//! The node's local policy file (docs/FLEET-PROTOCOL.md §6): the hard boundary for everything a hub asks.
//!
//! Lookup: `--policy PATH`, else /etc/ecnode/policy.json if it exists, else ~/.config/ecnode/policy.json. The file is
//! REFUSED (and the node runs in monitor mode, saying why in `error`) when it is a symlink or not a regular file,
//! group/world-writable, owned by anyone but root or this user, under /etc and not owned by root, in a directory
//! others can write to, too large, not JSON, or not valid against the schema below (unknown keys included, so a
//! typo in a safety switch is never silently ignored; keys starting with "_" are comments). No file at all also
//! means monitor mode. A hub can read the summary (it is in hello) but never change the file.

use std::os::fd::AsFd;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::Instant;

use ec_proto::Obj;
use nix::errno::Errno;
use nix::fcntl::OFlag;
use nix::sys::stat::Mode;
use regex::Regex;
use serde_json::{Map, Value, json};

use crate::util::py;

pub const ETC_POLICY: &str = "/etc/ecnode/policy.json";
pub const USER_POLICY: &str = ".config/ecnode/policy.json";
pub const MAX_POLICY_BYTES: u64 = 256 * 1024;
pub const SYSTEM_DIRS: &[&str] = &["/etc/"];
pub const MODES: &[&str] = &["monitor", "manage"];
pub const KINDS: &[&str] = &[
    "auto", "hf-home", "hf-hub", "gguf", "plain", "ollama", "other",
];
pub const EXTERNAL: &[&str] = &["none", "stop_start"];
/// The starter template's engine patterns (cli.IMAGES_NOTE says why llama.cpp is by digest).
pub const DEFAULT_IMAGES: &[&str] = &[
    "vllm/vllm-openai:*",
    "lmsysorg/sglang:*",
    "nvcr.io/nvidia/tensorrt-llm/release:*",
    "ghcr.io/ggml-org/llama.cpp@sha256:*",
    "ollama/ollama:*",
];
pub const DEFAULT_NETWORK: &str = "engine-console-engines";
pub const DEFAULT_PORTS: (u16, u16) = (18000, 18099);
pub const DEFAULT_DISCOVERY: &[u16] = &[8000, 8080, 11434, 30000];

// Image references (docker's grammar, slightly stricter): [domain[:port]/]path[:tag][@sha256:digest]
const COMP: &str = r"[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*";
const DOMAIN: &str = r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)*(?::[0-9]{1,5})?/)";
const TAG: &str = r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}";
const DIGEST: &str = r"sha256:[a-f0-9]{64}";

fn repo_re() -> String {
    format!("{DOMAIN}?{COMP}(?:/{COMP})*")
}

/// A Python `re.fullmatch` pattern for the regex crate.
pub(crate) fn full(p: &str) -> Regex {
    Regex::new(&format!(r"\A(?:{p})\z")).unwrap_or_else(|_| std::process::abort())
}

static IMAGE_REF: LazyLock<Regex> = LazyLock::new(|| {
    full(&format!(
        r"(?P<repo>{})(?::(?P<tag>{TAG}))?(?:@(?P<digest>{DIGEST}))?",
        repo_re()
    ))
});
static PAT_TAG: LazyLock<Regex> = LazyLock::new(|| full(&format!(r"(?P<repo>{}):\*", repo_re())));
static PAT_DIGEST: LazyLock<Regex> =
    LazyLock::new(|| full(&format!(r"(?P<repo>{})@sha256:\*", repo_re())));
static ROOT_ID: LazyLock<Regex> = LazyLock::new(|| full("[a-z0-9][a-z0-9_-]{0,31}"));
static SCRIPT_NAME: LazyLock<Regex> = LazyLock::new(|| full("[a-z0-9][a-z0-9_-]{0,31}"));
static NETWORK: LazyLock<Regex> = LazyLock::new(|| full("[A-Za-z0-9][A-Za-z0-9_.-]{0,63}"));
static CONTEXT: LazyLock<Regex> = LazyLock::new(|| full("[A-Za-z0-9][A-Za-z0-9_.+-]{0,63}"));
static BINARY: LazyLock<Regex> = LazyLock::new(|| full("/[A-Za-z0-9_./+-]{1,255}"));
/// Directories that must never be a model root themselves (a root below them is fine, e.g. /var/lib/ollama).
pub const FORBIDDEN_ROOTS: &[&str] = &[
    "/",
    "/etc",
    "/proc",
    "/sys",
    "/dev",
    "/boot",
    "/root",
    "/run",
    "/var",
    "/usr",
    "/bin",
    "/sbin",
    "/lib",
    "/lib32",
    "/lib64",
    "/libx32",
    "/home",
    "/tmp",
    "/opt",
    "/srv",
    "/mnt",
    "/media",
    "/snap",
    "/var/run",
    "/var/lib",
    "/usr/local",
];

/// The policy file is not acceptable; the message says which field and why (policy.py `PolicyError`).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct PolicyError(pub String);

fn err<T>(msg: impl Into<String>) -> Result<T, PolicyError> {
    Err(PolicyError(msg.into()))
}

#[derive(Debug, Clone, PartialEq)]
pub struct Root {
    pub id: String,
    pub path: String,
    pub kind: String,
    pub writable: bool,
    pub label: String,
}

impl Root {
    /// `Root(id, path)` with the dataclass defaults (kind auto, read-only, no label).
    pub fn new(id: &str, path: &str) -> Self {
        Self {
            id: id.into(),
            path: path.into(),
            kind: "auto".into(),
            writable: false,
            label: String::new(),
        }
    }

    pub fn summary(&self) -> Obj {
        let label = if self.label.is_empty() {
            &self.id
        } else {
            &self.label
        };
        obj(
            json!({"id": self.id, "path": self.path, "kind": self.kind, "writable": self.writable, "label": label}),
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScriptAction {
    pub name: String,
    pub label: String,
    pub argv: Vec<String>,
    pub args: Vec<String>,
    pub confirm: bool,
    pub danger: bool,
}

impl ScriptAction {
    /// argv stays on the node
    pub fn summary(&self) -> Obj {
        obj(
            json!({"name": self.name, "label": self.label, "args": self.args, "confirm": self.confirm,
                   "danger": self.danger}),
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Script {
    pub name: String,
    pub label: String,
    pub engine: Option<String>,
    pub description: String,
    pub containers: Vec<String>,
    pub status_action: Option<String>,
    pub timeout_s: u64,
    pub actions: Vec<ScriptAction>,
}

impl Script {
    pub fn action(&self, name: &str) -> Option<&ScriptAction> {
        self.actions.iter().find(|a| a.name == name)
    }

    /// `any(fnmatch.fnmatchcase(container_name, p) for p in self.containers)`
    pub fn matches(&self, container_name: &str) -> bool {
        self.containers
            .iter()
            .any(|p| py::fnmatchcase(container_name, p))
    }

    pub fn summary(&self) -> Obj {
        let actions: Vec<Value> = self
            .actions
            .iter()
            .map(|a| Value::Object(a.summary()))
            .collect();
        obj(
            json!({"name": self.name, "label": self.label, "engine": self.engine, "description": self.description,
                   "actions": actions, "containers": self.containers, "status_action": self.status_action}),
        )
    }
}

/// The node's local policy (policy.py `Policy`, same fields).
#[derive(Debug, Clone, PartialEq)]
pub struct Policy {
    pub path: Option<String>,
    pub mode: String,
    pub error: Option<String>,
    pub docker_binary: String,
    pub docker_context: Option<String>,
    pub engine_images: Vec<String>,
    pub gateway_image: Option<String>,
    pub engine_network: String,
    pub port_range: (u16, u16),
    pub discovery_ports: Vec<u16>,
    pub model_roots: Vec<Root>,
    pub scripts: Vec<Script>,
    pub allow_pull: bool,
    pub allow_delete: bool,
    pub allow_ipc_host: bool,
    pub external_control: String,
    pub max_shm_gib: f64,
}

/// An INERT policy for tests and in-memory tools: Python's `Policy()` defaults except `discovery_ports`, which is
/// empty, so an agent built on it probes no port of the machine it runs on. The node's own fallback for a missing or
/// refused file, `monitor_policy()`, is just as inert (no discovery ports); only a VALID policy that omits
/// `discovery_ports` gets Python's defaults (8000, 8080, 11434, 30000).
impl Default for Policy {
    fn default() -> Self {
        Self {
            discovery_ports: Vec::new(),
            ..python_default()
        }
    }
}

/// Python's `Policy()`: every dataclass default.
pub fn python_default() -> Policy {
    Policy {
        path: None,
        mode: "monitor".into(),
        error: None,
        docker_binary: "docker".into(),
        docker_context: None,
        engine_images: Vec::new(),
        gateway_image: None,
        engine_network: DEFAULT_NETWORK.into(),
        port_range: DEFAULT_PORTS,
        discovery_ports: DEFAULT_DISCOVERY.to_vec(),
        model_roots: Vec::new(),
        scripts: Vec::new(),
        allow_pull: false,
        allow_delete: false,
        allow_ipc_host: false,
        external_control: "none".into(),
        max_shm_gib: 64.0,
    }
}

impl Policy {
    pub fn manage(&self) -> bool {
        self.mode == "manage"
    }

    pub fn root(&self, root_id: &str) -> Option<&Root> {
        self.model_roots.iter().find(|r| r.id == root_id)
    }

    /// `policy.root(v)` for a JSON value (only a string can name a root).
    pub fn root_of(&self, root_id: &Value) -> Option<&Root> {
        root_id.as_str().and_then(|r| self.root(r))
    }

    pub fn script(&self, name: &str) -> Option<&Script> {
        self.scripts.iter().find(|s| s.name == name)
    }

    pub fn image_allowed(&self, image: &str) -> bool {
        self.engine_images.iter().any(|p| image_matches(p, image))
    }

    /// `policy.where()`: the file, or /etc/ecnode/policy.json when there is none.
    pub fn where_(&self) -> String {
        self.path.clone().unwrap_or_else(|| ETC_POLICY.to_string())
    }

    /// What `hello` reports (POLICY spec keys, policy.py order).
    pub fn summary(&self) -> Obj {
        let roots: Vec<Value> = self
            .model_roots
            .iter()
            .map(|r| Value::Object(r.summary()))
            .collect();
        let scripts: Vec<Value> = self
            .scripts
            .iter()
            .map(|s| Value::Object(s.summary()))
            .collect();
        obj(json!({
            "path": self.path, "mode": self.mode, "error": self.error, "engine_images": self.engine_images,
            "gateway_image": self.gateway_image, "engine_network": self.engine_network,
            "port_range": [self.port_range.0, self.port_range.1], "discovery_ports": self.discovery_ports,
            "model_roots": roots, "scripts": scripts, "allow_pull": self.allow_pull,
            "allow_delete": self.allow_delete, "allow_ipc_host": self.allow_ipc_host,
            "external_control": self.external_control, "max_shm_gib": self.max_shm_gib,
        }))
    }
}

fn obj(v: Value) -> Obj {
    match v {
        Value::Object(o) => o,
        _ => Obj::new(),
    }
}

// ── image references ─────────────────────────────────────────────────────────────────────────────────────
/// (repository, tag, digest) of a valid image reference, else None.
pub fn split_image(r: &str) -> Option<(String, Option<String>, Option<String>)> {
    if py::len(r) > 256 {
        return None;
    }
    let c = IMAGE_REF.captures(r)?;
    Some((
        c.name("repo")?.as_str().to_string(),
        c.name("tag").map(|m| m.as_str().to_string()),
        c.name("digest").map(|m| m.as_str().to_string()),
    ))
}

pub fn valid_image(r: &str) -> bool {
    split_image(r).is_some()
}

/// `valid_image(v)` for a JSON value.
pub fn valid_image_value(v: &Value) -> bool {
    v.as_str().is_some_and(valid_image)
}

/// Exact reference, `repo:*` (any tag of exactly that repository) or `repo@sha256:*` (any digest of it). The
/// repository is compared literally.
pub fn image_matches(pattern: &str, image: &str) -> bool {
    let Some((repo, tag, digest)) = split_image(image) else {
        return false;
    };
    if let Some(m) = PAT_TAG.captures(pattern) {
        return m.name("repo").is_some_and(|r| r.as_str() == repo)
            && tag.is_some()
            && digest.is_none();
    }
    if let Some(m) = PAT_DIGEST.captures(pattern) {
        return m.name("repo").is_some_and(|r| r.as_str() == repo)
            && digest.is_some()
            && tag.is_none();
    }
    valid_image(pattern) && image == pattern
}

pub fn valid_image_pattern(p: &str) -> bool {
    PAT_TAG.is_match(p) || PAT_DIGEST.is_match(p) || valid_image(p)
}

// ── schema ───────────────────────────────────────────────────────────────────────────────────────────────
fn keys(obj: &Map<String, Value>, allowed: &[&str], where_: &str) -> Result<(), PolicyError> {
    for k in obj.keys() {
        if !k.starts_with('_') && !allowed.contains(&k.as_str()) {
            let mut sorted: Vec<&str> = allowed.to_vec();
            sorted.sort_unstable();
            return err(format!(
                "{where_}: unknown key {} (allowed: {})",
                py::repr(k),
                sorted.join(", ")
            ));
        }
    }
    Ok(())
}

fn bool_(
    obj: &Map<String, Value>,
    key: &str,
    where_: &str,
    default: bool,
) -> Result<bool, PolicyError> {
    match obj.get(key) {
        None => Ok(default),
        Some(Value::Bool(b)) => Ok(*b),
        Some(_) => err(format!("{where_}.{key} must be true or false")),
    }
}

fn str_(
    obj: &Map<String, Value>,
    key: &str,
    where_: &str,
    limit: usize,
    default: Option<&str>,
    pattern: Option<&Regex>,
    required: bool,
) -> Result<Option<String>, PolicyError> {
    let v = match obj.get(key) {
        None | Some(Value::Null) => {
            if required {
                return err(format!("{where_}.{key} is required"));
            }
            return Ok(default.map(str::to_string));
        }
        Some(v) => v,
    };
    let s = match v {
        Value::String(s) if !s.is_empty() && py::len(s) <= limit && !s.contains('\0') => s,
        _ => {
            return err(format!(
                "{where_}.{key} must be a non-empty string of at most {limit} characters"
            ));
        }
    };
    if let Some(p) = pattern
        && !p.is_match(s)
    {
        return err(format!(
            "{where_}.{key}: {} is not a valid value",
            py::repr(s)
        ));
    }
    Ok(Some(s.clone()))
}

fn int_(v: &Value, where_: &str, lo: i64, hi: i64) -> Result<i64, PolicyError> {
    match v {
        Value::Number(n) if !n.is_f64() => match n.as_i64() {
            Some(i) if lo <= i && i <= hi => Ok(i),
            _ => err(format!("{where_} must be a whole number from {lo} to {hi}")),
        },
        _ => err(format!("{where_} must be a whole number from {lo} to {hi}")),
    }
}

fn list<'a>(
    obj: &'a Map<String, Value>,
    key: &str,
    where_: &str,
    cap: usize,
) -> Result<Vec<&'a Value>, PolicyError> {
    match obj.get(key) {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::Array(a)) if a.len() <= cap => Ok(a.iter().collect()),
        Some(_) => err(format!(
            "{where_}.{key} must be a list of at most {cap} items"
        )),
    }
}

/// The node's own files and the login user's keys, which no model root may be, contain or lie inside (security
/// audit F2, beyond Python): (what, directory). A hub that can write in a root that holds the user policy could add a
/// `scripts` entry and run anything as this user. Ancestors of the home directory are refused separately (they hold
/// every rc file).
fn protected_dirs(policy_path: Option<&str>, home: Option<&str>) -> Vec<(&'static str, PathBuf)> {
    let mut out = vec![(
        "the node's system policy directory",
        PathBuf::from("/etc/ecnode"),
    )];
    if let Some(p) = policy_path.filter(|p| !p.is_empty()) {
        out.push((
            "the policy file's directory",
            py::dirname(&py::abspath(Path::new(p))),
        ));
    }
    if let Some(h) = home.filter(|h| !h.is_empty()) {
        for (what, rel) in [
            ("the login user's SSH directory", ".ssh"),
            ("the node's user policy directory", ".config/ecnode"),
            ("the node's state directory", ".local/state/ecnode"),
        ] {
            out.push((what, py::join(Path::new(h), &[rel])));
        }
    }
    out
}

/// A path as written (normalised) and, when it resolves elsewhere, its realpath: a root reached through a symlink is
/// compared where it really leads too.
fn path_forms(p: &str) -> Vec<PathBuf> {
    let norm = PathBuf::from(py::normpath_str(p));
    let real = py::realpath(Path::new(p));
    if real == norm {
        vec![norm]
    } else {
        vec![norm, real]
    }
}

/// Refuse a root that is, holds or lies inside a protected directory, or that holds the home directory (security
/// audit F2).
fn check_root_overlap(
    norm: &str,
    where_: &str,
    policy_path: Option<&str>,
    home: Option<&str>,
) -> Result<(), PolicyError> {
    use crate::confine::inside;
    let roots = path_forms(norm);
    if let Some(h) = home.filter(|h| !h.is_empty()) {
        let homes = path_forms(h);
        if roots.iter().any(|r| homes.iter().any(|hh| inside(hh, r))) {
            return err(format!(
                "{where_}.path: {} holds the home directory {}; name the directory that holds the models (for \
                 example /data/hf)",
                py::repr(norm),
                py::repr(h)
            ));
        }
    }
    for (what, dir) in protected_dirs(policy_path, home) {
        let dirs = path_forms(&py::path_str(&dir));
        if roots
            .iter()
            .any(|r| dirs.iter().any(|d| inside(r, d) || inside(d, r)))
        {
            return err(format!(
                "{where_}.path: {} overlaps {what} ({}); a model root must not be, contain or lie inside it",
                py::repr(norm),
                py::path_str(&dir)
            ));
        }
    }
    Ok(())
}

fn root_path(p: Option<&Value>, where_: &str, home: Option<&str>) -> Result<String, PolicyError> {
    let p = match p {
        Some(Value::String(s)) if s.starts_with('/') && py::len(s) <= 512 => s,
        _ => {
            return err(format!(
                "{where_}.path must be an absolute path of at most 512 characters"
            ));
        }
    };
    if p.chars().any(|c| ":,\0\n\r\\".contains(c))
        || p.split('/').any(|seg| seg == "." || seg == "..")
    {
        return err(format!(
            "{where_}.path: {} must not contain ':', ',', '\\', '.' or '..' segments",
            py::repr(p)
        ));
    }
    let norm = py::normpath_str(p);
    let home_hit = home
        .filter(|h| !h.is_empty())
        .is_some_and(|h| norm == py::normpath_str(h));
    if FORBIDDEN_ROOTS.contains(&norm.as_str()) || home_hit {
        return err(format!(
            "{where_}.path: {} is too broad for a model root; name the directory that holds the models (for \
             example /data/hf)",
            py::repr(&norm)
        ));
    }
    Ok(norm)
}

/// `value == 1` in Python (True and 1.0 count).
fn py_eq_one(v: &Value) -> bool {
    match v {
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64() == Some(1.0),
        _ => false,
    }
}

fn str_in(v: Option<&Value>, set: &[&str]) -> bool {
    v.and_then(Value::as_str).is_some_and(|s| set.contains(&s))
}

/// Validate a decoded policy document; PolicyError says exactly what is wrong (policy.py `parse`).
pub fn parse(data: &Value, path: Option<&str>, home: Option<&str>) -> Result<Policy, PolicyError> {
    let path_arg = path;
    let Value::Object(data) = data else {
        return err("the policy must be one JSON object");
    };
    keys(
        data,
        &[
            "version",
            "mode",
            "docker",
            "engine_images",
            "gateway_image",
            "engine_network",
            "port_range",
            "discovery_ports",
            "model_roots",
            "scripts",
            "allow_pull",
            "allow_delete",
            "allow_ipc_host",
            "external_control",
            "max_shm_gib",
        ],
        "policy",
    )?;
    if !data.get("version").is_none_or(py_eq_one) {
        return err("policy.version must be 1 (this node understands policy version 1)");
    }
    let mode = match data.get("mode") {
        None => "monitor".to_string(),
        Some(v) if str_in(Some(v), MODES) => v.as_str().unwrap_or("monitor").to_string(),
        Some(_) => return err("policy.mode must be \"monitor\" or \"manage\""),
    };

    let empty = Map::new();
    let dk = match data.get("docker") {
        Some(v) if py::truthy(v) => match v {
            Value::Object(m) => m,
            _ => {
                return err(
                    "policy.docker must be an object like {\"binary\": \"docker\", \"context\": null}",
                );
            }
        },
        _ => &empty,
    };
    keys(dk, &["binary", "context"], "policy.docker")?;
    let binary = str_(
        dk,
        "binary",
        "policy.docker",
        256,
        Some("docker"),
        None,
        false,
    )?
    .filter(|b| !b.is_empty())
    .unwrap_or_else(|| "docker".into());
    if binary != "docker" && (!BINARY.is_match(&binary) || binary.split('/').any(|s| s == "..")) {
        return err(
            "policy.docker.binary must be \"docker\" or an absolute path to the docker CLI",
        );
    }
    let context = str_(
        dk,
        "context",
        "policy.docker",
        64,
        None,
        Some(&CONTEXT),
        false,
    )?;

    let images = list(data, "engine_images", "policy", 64)?;
    let mut engine_images = Vec::new();
    for (i, p) in images.iter().enumerate() {
        match p {
            Value::String(s) if valid_image_pattern(s) => engine_images.push(s.clone()),
            _ => {
                return err(format!(
                    "policy.engine_images[{i}]: {} must be an exact image reference, 'repository:*' or \
                     'repository@sha256:*' with a literal repository",
                    py::repr_of(p)
                ));
            }
        }
    }
    let gateway = match data.get("gateway_image") {
        None | Some(Value::Null) => None,
        Some(v) if valid_image_value(v) => v.as_str().map(str::to_string),
        Some(_) => {
            return err(
                "policy.gateway_image must be an exact image reference (best: nginx@sha256:<digest>)",
            );
        }
    };
    let network = str_(
        data,
        "engine_network",
        "policy",
        64,
        Some(DEFAULT_NETWORK),
        Some(&NETWORK),
        false,
    )?;

    let default_pr = json!([DEFAULT_PORTS.0, DEFAULT_PORTS.1]);
    let pr = data.get("port_range").unwrap_or(&default_pr);
    let pr = match pr {
        Value::Array(a) if a.len() == 2 => a,
        _ => return err("policy.port_range must be [first, last], e.g. [18000, 18099]"),
    };
    let lo = int_(&pr[0], "policy.port_range[0]", 1024, 65535)?;
    let hi = int_(&pr[1], "policy.port_range[1]", lo, 65535.min(lo + 1000))?;
    let mut dports: Vec<u16> = Vec::new();
    for (i, p) in list(data, "discovery_ports", "policy", 128)?
        .into_iter()
        .enumerate()
    {
        let v = int_(p, &format!("policy.discovery_ports[{i}]"), 1, 65535)?;
        let v = u16::try_from(v).unwrap_or(0);
        if !dports.contains(&v) {
            dports.push(v);
        }
    }
    if !data.contains_key("discovery_ports") {
        dports = DEFAULT_DISCOVERY.to_vec();
    }

    let mut roots: Vec<Root> = Vec::new();
    for (i, r) in list(data, "model_roots", "policy", 32)?
        .into_iter()
        .enumerate()
    {
        let where_ = format!("policy.model_roots[{i}]");
        let Value::Object(r) = r else {
            return err(format!("{where_} must be an object"));
        };
        keys(r, &["id", "path", "kind", "writable", "label"], &where_)?;
        let rid = str_(r, "id", &where_, 32, None, Some(&ROOT_ID), true)?.unwrap_or_default();
        if roots.iter().any(|x| x.id == rid) {
            return err(format!("{where_}.id: {} is used twice", py::repr(&rid)));
        }
        let kind = str_(r, "kind", &where_, 16, Some("auto"), None, false)?
            .filter(|k| !k.is_empty())
            .unwrap_or_else(|| "auto".into());
        if !KINDS.contains(&kind.as_str()) {
            return err(format!("{where_}.kind must be one of {}", KINDS.join(", ")));
        }
        let path = root_path(r.get("path"), &where_, home)?;
        check_root_overlap(&path, &where_, path_arg, home)?;
        let writable = bool_(r, "writable", &where_, false)?;
        let label = str_(r, "label", &where_, 64, Some(&rid), None, false)?
            .filter(|l| !l.is_empty())
            .unwrap_or_else(|| rid.clone());
        roots.push(Root {
            id: rid,
            path,
            kind,
            writable,
            label,
        });
    }

    let mut scripts: Vec<Script> = Vec::new();
    for (i, s) in list(data, "scripts", "policy", 32)?.into_iter().enumerate() {
        let sc = script(s, &format!("policy.scripts[{i}]"), &scripts)?;
        scripts.push(sc);
    }

    let ext = match data.get("external_control") {
        None => "none".to_string(),
        Some(v) if str_in(Some(v), EXTERNAL) => v.as_str().unwrap_or("none").to_string(),
        Some(_) => return err("policy.external_control must be \"none\" or \"stop_start\""),
    };
    let shm = match data.get("max_shm_gib") {
        None => 64.0,
        Some(Value::Number(n)) => match n.as_f64() {
            Some(f) if f > 0.0 && f <= 4096.0 => f,
            _ => return err("policy.max_shm_gib must be a number above 0 and at most 4096"),
        },
        Some(_) => return err("policy.max_shm_gib must be a number above 0 and at most 4096"),
    };
    let allow_pull = bool_(data, "allow_pull", "policy", false)?;
    let allow_delete = bool_(data, "allow_delete", "policy", false)?;
    let allow_ipc_host = bool_(data, "allow_ipc_host", "policy", false)?;
    Ok(Policy {
        path: path.map(str::to_string),
        mode,
        error: None,
        docker_binary: binary,
        docker_context: context,
        engine_images,
        gateway_image: gateway,
        engine_network: network
            .filter(|n| !n.is_empty())
            .unwrap_or_else(|| DEFAULT_NETWORK.into()),
        port_range: (
            u16::try_from(lo).unwrap_or(0),
            u16::try_from(hi).unwrap_or(0),
        ),
        discovery_ports: dports,
        model_roots: roots,
        scripts,
        allow_pull,
        allow_delete,
        allow_ipc_host,
        external_control: ext,
        max_shm_gib: shm,
    })
}

fn script(s: &Value, where_: &str, taken: &[Script]) -> Result<Script, PolicyError> {
    let Value::Object(s) = s else {
        return err(format!("{where_} must be an object"));
    };
    keys(
        s,
        &[
            "name",
            "label",
            "engine",
            "description",
            "containers",
            "status_action",
            "timeout_s",
            "actions",
        ],
        where_,
    )?;
    let name = str_(s, "name", where_, 32, None, Some(&SCRIPT_NAME), true)?.unwrap_or_default();
    if taken.iter().any(|x| x.name == name) {
        return err(format!("{where_}.name: {} is used twice", py::repr(&name)));
    }
    let mut containers = Vec::new();
    for (j, c) in list(s, "containers", where_, 16)?.into_iter().enumerate() {
        match c {
            Value::String(c) if !c.is_empty() && py::len(c) <= 128 && !c.contains('\0') => {
                containers.push(c.clone());
            }
            _ => {
                return err(format!(
                    "{where_}.containers[{j}] must be a container name pattern like 'myproj-sglang-*'"
                ));
            }
        }
    }
    let mut actions: Vec<ScriptAction> = Vec::new();
    for (j, a) in list(s, "actions", where_, 16)?.into_iter().enumerate() {
        let aw = format!("{where_}.actions[{j}]");
        let Value::Object(a) = a else {
            return err(format!("{aw} must be an object"));
        };
        keys(
            a,
            &["name", "label", "argv", "args", "confirm", "danger"],
            &aw,
        )?;
        let an = str_(a, "name", &aw, 32, None, Some(&SCRIPT_NAME), true)?.unwrap_or_default();
        if actions.iter().any(|x| x.name == an) {
            return err(format!("{aw}.name: {} is used twice", py::repr(&an)));
        }
        let argv: Vec<String> = match a.get("argv") {
            Some(Value::Array(v))
                if (1..=64).contains(&v.len())
                    && v.iter().all(|x| {
                        x.as_str().is_some_and(|x| {
                            !x.is_empty() && py::len(x) <= 4096 && !x.contains('\0')
                        })
                    }) =>
            {
                v.iter()
                    .filter_map(|x| x.as_str().map(str::to_string))
                    .collect()
            }
            _ => {
                return err(format!(
                    "{aw}.argv must be a list of 1-64 non-empty strings (the program, then its arguments)"
                ));
            }
        };
        if !argv[0].starts_with('/') || argv[0].contains('\n') {
            return err(format!(
                "{aw}.argv[0] must be an absolute path to the program (no shell, no PATH lookup)"
            ));
        }
        let args_v = list(a, "args", &aw, 32)?;
        if !args_v.iter().all(|x| {
            x.as_str()
                .is_some_and(|x| !x.is_empty() && py::len(x) <= 64 && !x.contains('\0'))
        }) {
            return err(format!(
                "{aw}.args must be a list of allowed argument values (each 1-64 characters)"
            ));
        }
        let args: Vec<String> = args_v
            .iter()
            .filter_map(|x| x.as_str().map(str::to_string))
            .collect();
        let label = str_(a, "label", &aw, 64, Some(&an), None, false)?
            .filter(|l| !l.is_empty())
            .unwrap_or_else(|| an.clone());
        let confirm = bool_(a, "confirm", &aw, false)?;
        let danger = bool_(a, "danger", &aw, false)?;
        actions.push(ScriptAction {
            name: an,
            label,
            argv,
            args,
            confirm,
            danger,
        });
    }
    let status = str_(s, "status_action", where_, 32, None, None, false)?;
    if let Some(st) = &status
        && !actions.iter().any(|x| &x.name == st)
    {
        return err(format!(
            "{where_}.status_action must name one of the script's actions"
        ));
    }
    let timeout = int_(
        s.get("timeout_s").unwrap_or(&json!(900)),
        &format!("{where_}.timeout_s"),
        1,
        86400,
    )?;
    let label = str_(s, "label", where_, 64, Some(&name), None, false)?
        .filter(|l| !l.is_empty())
        .unwrap_or_else(|| name.clone());
    let engine = str_(s, "engine", where_, 32, None, None, false)?;
    let description =
        str_(s, "description", where_, 500, Some(""), None, false)?.unwrap_or_default();
    Ok(Script {
        name,
        label,
        engine,
        description,
        containers,
        status_action: status,
        timeout_s: u64::try_from(timeout).unwrap_or(900),
        actions,
    })
}

// ── loading ──────────────────────────────────────────────────────────────────────────────────────────────
/// What the node runs with when there is no usable policy (no file, or a refused one; `error` says which): monitor
/// mode, no roots, no scripts and NO discovery ports (Python's defaults otherwise). Security audit F1, beyond Python:
/// a node nobody configured probes no loopback port of its own (Python probes 8000, 8080, 11434 and 30000 every
/// inventory round), and the agent then neither guesses ports of host-network containers nor lets `http.request`
/// reach any port but one a container publishes (agent.rs `refresh` / `allowed_ports`).
pub fn monitor_policy(path: Option<String>, error: Option<String>) -> Policy {
    Policy {
        path,
        error,
        discovery_ports: Vec::new(),
        ..python_default()
    }
}

/// `os.path.expanduser("~")`
pub fn home_dir() -> String {
    match std::env::var_os("HOME") {
        Some(h) => {
            let h = h.to_string_lossy();
            let h = h.trim_end_matches('/');
            if h.is_empty() {
                "/".into()
            } else {
                h.to_string()
            }
        }
        None => nix::unistd::User::from_uid(nix::unistd::getuid())
            .ok()
            .flatten()
            .map_or_else(|| "/".into(), |u| u.dir.to_string_lossy().into_owned()),
    }
}

/// (path to load, whether it was named explicitly), as policy.py `candidate`.
pub fn candidate(
    explicit: Option<&str>,
    home: Option<&str>,
    etc_path: &str,
) -> (Option<String>, bool) {
    if let Some(e) = explicit.filter(|e| !e.is_empty()) {
        return (Some(e.to_string()), true);
    }
    if Path::new(etc_path).symlink_metadata().is_ok() {
        return (Some(etc_path.to_string()), false);
    }
    let home = home
        .filter(|h| !h.is_empty())
        .map_or_else(home_dir, str::to_string);
    let user = py::join(Path::new(&home), &[USER_POLICY]);
    if user.symlink_metadata().is_ok() {
        return (Some(user.to_string_lossy().into_owned()), false);
    }
    (None, false)
}

fn under(path: &str, prefixes: &[String]) -> bool {
    let ap = py::path_str(&py::abspath(Path::new(path)));
    prefixes.iter().any(|p| ap.starts_with(p.as_str()))
}

/// Identity of whatever is at `path` now (not following a symlink): (dev, ino, mtime_ns, size, mode, uid).
pub fn file_stamp(path: &str) -> Vec<i64> {
    match std::fs::symlink_metadata(path) {
        Ok(m) => vec![
            i64::try_from(m.dev()).unwrap_or(-1),
            i64::try_from(m.ino()).unwrap_or(-1),
            m.mtime() * 1_000_000_000 + m.mtime_nsec(),
            i64::try_from(m.size()).unwrap_or(-1),
            i64::from(m.mode()),
            i64::from(m.uid()),
        ],
        Err(_) => Vec::new(),
    }
}

/// Where load() looks and what it trusts (tests point it at temp dirs).
#[derive(Debug, Clone)]
pub struct LoadOptions {
    pub explicit: Option<String>,
    pub home: Option<String>,
    pub etc_path: String,
    pub system_dirs: Vec<String>,
    /// the uid that may own the file besides root (None: this process's)
    pub uid: Option<u32>,
}

impl LoadOptions {
    pub fn new(explicit: Option<&str>, home: Option<&str>, etc_path: &str) -> Self {
        Self {
            explicit: explicit.map(str::to_string),
            home: home.map(str::to_string),
            etc_path: etc_path.to_string(),
            system_dirs: SYSTEM_DIRS.iter().map(|s| (*s).to_string()).collect(),
            uid: None,
        }
    }
}

/// Read and validate the policy; never fails. Any problem -> monitor mode with `error` set.
pub fn load(explicit: Option<&str>, home: Option<&str>, etc_path: &str) -> Policy {
    load_with(&LoadOptions::new(explicit, home, etc_path)).0
}

/// load() and the file stamp it read (for PolicyStore's change detection).
pub fn load_with(o: &LoadOptions) -> (Policy, Vec<i64>) {
    let (path, named) = candidate(o.explicit.as_deref(), o.home.as_deref(), &o.etc_path);
    let Some(path) = path else {
        return (
            monitor_policy(
                None,
                Some(format!(
                    "no policy file, so this node runs in monitor mode (write {}, for example from `ecnode \
                     policy-template`)",
                    o.etc_path
                )),
            ),
            Vec::new(),
        );
    };
    let me = o.uid.unwrap_or_else(|| nix::unistd::getuid().as_raw());
    let stamp = file_stamp(&path);
    let refused = |why: String| {
        monitor_policy(
            Some(path.clone()),
            Some(format!("policy file {path} {why}; running in monitor mode")),
        )
    };
    let fd = match nix::fcntl::open(
        path.as_str(),
        OFlag::O_RDONLY | OFlag::O_NOFOLLOW | OFlag::O_CLOEXEC | OFlag::O_NONBLOCK,
        Mode::empty(),
    ) {
        Ok(fd) => fd,
        Err(e) => {
            let why = match e {
                Errno::ELOOP => "is a symlink; point --policy at the real file".to_string(),
                Errno::ENOENT if named => "does not exist".to_string(),
                Errno::ENOENT => "disappeared while being read".to_string(),
                e => format!("cannot be read ({})", py::strerror(e as i32)),
            };
            return (refused(why), stamp);
        }
    };
    let st = match nix::sys::stat::fstat(fd.as_fd()) {
        Ok(st) => st,
        Err(e) => {
            return (
                refused(format!("cannot be read ({})", py::strerror(e as i32))),
                stamp,
            );
        }
    };
    if let Some(problem) = file_problem(&path, &st, me, &o.system_dirs) {
        return (refused(problem), stamp);
    }
    let mut raw: Vec<u8> = Vec::new();
    let mut file = std::fs::File::from(fd);
    let max = usize::try_from(MAX_POLICY_BYTES).unwrap_or(usize::MAX);
    let mut buf = vec![0u8; 65536];
    while raw.len() <= max {
        let want = (max + 1 - raw.len()).min(buf.len());
        match std::io::Read::read(&mut file, &mut buf[..want]) {
            Ok(0) => break,
            Ok(n) => raw.extend_from_slice(&buf[..n]),
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => {
                return (
                    refused(format!("cannot be read ({})", py::io_strerror(&e))),
                    stamp,
                );
            }
        }
    }
    if raw.len() > max {
        return (
            refused(format!("is larger than {MAX_POLICY_BYTES} bytes")),
            stamp,
        );
    }
    let text = match String::from_utf8(raw) {
        Ok(t) => t,
        Err(e) => {
            let pos = e.utf8_error().valid_up_to();
            let byte = e.as_bytes().get(pos).copied().unwrap_or(0);
            let why = format!(
                "'utf-8' codec can't decode byte 0x{byte:02x} in position {pos}: invalid start byte"
            );
            return (
                refused(format!("is not valid JSON ({})", py::take(&why, 160))),
                stamp,
            );
        }
    };
    let data: Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(e) => {
            let why = json_error_text(&text, &e);
            return (
                refused(format!("is not valid JSON ({})", py::take(&why, 160))),
                stamp,
            );
        }
    };
    let home = o
        .home
        .clone()
        .filter(|h| !h.is_empty())
        .unwrap_or_else(home_dir);
    match parse(&data, Some(&path), Some(&home)) {
        Ok(p) => (p, stamp),
        Err(PolicyError(e)) => (
            monitor_policy(
                Some(path.clone()),
                Some(format!("policy file {path}: {e}; running in monitor mode")),
            ),
            stamp,
        ),
    }
}

/// The detail of a JSON syntax error, in the words of Python's json module where serde can tell (the position
/// is serde's: PORTING.md DV19).
fn json_error_text(text: &str, e: &serde_json::Error) -> String {
    let (line, col) = (e.line(), e.column());
    // char offset of (line, col), 1-based column as Python counts it
    let mut pos = 0usize;
    for (i, l) in text.split('\n').enumerate() {
        if i + 1 == line {
            pos += py::len(py::take(l, col.saturating_sub(1)));
            break;
        }
        pos += py::len(l) + 1;
    }
    let msg = e.to_string();
    let what = if msg.starts_with("key must be a string") {
        "Expecting property name enclosed in double quotes"
    } else if msg.starts_with("expected `:`") {
        "Expecting ':' delimiter"
    } else if msg.starts_with("expected `,` or") {
        "Expecting ',' delimiter"
    } else if msg.starts_with("trailing characters") {
        "Extra data"
    } else if msg.starts_with("EOF while parsing a string") {
        "Unterminated string starting at"
    } else if msg.starts_with("control character") {
        "Invalid control character at"
    } else if msg.starts_with("invalid escape") {
        "Invalid \\escape"
    } else {
        "Expecting value"
    };
    format!("{what}: line {line} column {col} (char {pos})")
}

fn file_problem(
    path: &str,
    st: &nix::sys::stat::FileStat,
    me: u32,
    system_dirs: &[String],
) -> Option<String> {
    let mode = st.st_mode;
    if mode & nix::libc::S_IFMT != nix::libc::S_IFREG {
        return Some("is not a regular file".into());
    }
    if mode & 0o022 != 0 {
        return Some(format!(
            "is group- or world-writable (run: sudo chmod 644 {path})"
        ));
    }
    let system = under(path, system_dirs);
    if system && st.st_uid != 0 {
        return Some(format!(
            "must be owned by root (run: sudo chown root:root {path})"
        ));
    }
    if st.st_uid != 0 && st.st_uid != me {
        return Some("is owned by another user".into());
    }
    // The directory matters too: whoever can write to it can swap the file. Under /etc it must be root's alone;
    // elsewhere a group-writable directory is tolerated, a world-writable one or one owned by another user is not.
    let parent = py::dirname(&py::abspath(Path::new(path)));
    let Ok(pst) = std::fs::metadata(&parent) else {
        return Some("has a directory that cannot be checked".into());
    };
    let parent = py::path_str(&parent);
    if pst.mode() & (if system { 0o022 } else { 0o002 }) != 0 {
        return Some(format!("is in a directory others can write to ({parent})"));
    }
    if (system && pst.uid() != 0) || (pst.uid() != 0 && pst.uid() != me) {
        return Some(format!(
            "is in a directory owned by another user ({parent})"
        ));
    }
    None
}

pub type Clock = Arc<dyn Fn() -> f64 + Send + Sync>;

pub fn monotonic() -> Clock {
    let start = Instant::now();
    Arc::new(move || start.elapsed().as_secs_f64())
}

struct StoreState {
    policy: Arc<Policy>,
    stamp: Vec<i64>,
    checked: f64,
}

/// The current policy (policy.py `PolicyStore`). Re-checks the file at most every `recheck_s` and reloads it when
/// it changed (or when a file appears where there was none), so an administrator tightening the policy need not
/// restart the node.
pub struct PolicyStore {
    opts: LoadOptions,
    recheck_s: f64,
    clock: Clock,
    fixed: bool,
    state: Mutex<StoreState>,
}

impl PolicyStore {
    pub fn new(explicit: Option<String>) -> Self {
        Self::with_paths(explicit, None, ETC_POLICY.to_string())
    }

    #[allow(clippy::needless_pass_by_value)] // the signature the callers use
    pub fn with_paths(explicit: Option<String>, home: Option<String>, etc_path: String) -> Self {
        Self::with_options(
            LoadOptions::new(explicit.as_deref(), home.as_deref(), &etc_path),
            2.0,
            monotonic(),
        )
    }

    /// Every knob of Python's constructor (system_dirs, recheck_s, clock).
    pub fn with_options(opts: LoadOptions, recheck_s: f64, clock: Clock) -> Self {
        let (p, stamp) = load_with(&opts);
        let checked = clock();
        Self {
            opts,
            recheck_s,
            clock,
            fixed: false,
            state: Mutex::new(StoreState {
                policy: Arc::new(p),
                stamp,
                checked,
            }),
        }
    }

    /// A store that always answers `policy` (tests, and tools that build a policy in memory).
    pub fn from_policy(policy: Policy) -> Self {
        Self {
            opts: LoadOptions::new(None, None, ETC_POLICY),
            recheck_s: f64::INFINITY,
            clock: monotonic(),
            fixed: true,
            state: Mutex::new(StoreState {
                policy: Arc::new(policy),
                stamp: Vec::new(),
                checked: 0.0,
            }),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, StoreState> {
        crate::util::lock(&self.state)
    }

    fn identity(&self) -> (Option<String>, Vec<i64>) {
        let (path, _) = candidate(
            self.opts.explicit.as_deref(),
            self.opts.home.as_deref(),
            &self.opts.etc_path,
        );
        let stamp = path.as_deref().map(file_stamp).unwrap_or_default();
        (path, stamp)
    }

    pub fn current(&self) -> Arc<Policy> {
        if self.fixed {
            return Arc::clone(&self.lock().policy);
        }
        let now = (self.clock)();
        let due = {
            let mut st = self.lock();
            if now - st.checked >= self.recheck_s {
                st.checked = now;
                true
            } else {
                false
            }
        };
        if due {
            let ident = self.identity();
            let changed = {
                let st = self.lock();
                ident != (st.policy.path.clone(), st.stamp.clone())
            };
            if changed {
                let (p, stamp) = load_with(&self.opts);
                let mut st = self.lock();
                st.policy = Arc::new(p);
                st.stamp = stamp;
            }
        }
        Arc::clone(&self.lock().policy)
    }

    pub fn reload(&self) -> Arc<Policy> {
        if self.fixed {
            return Arc::clone(&self.lock().policy);
        }
        let (p, stamp) = load_with(&self.opts);
        let p = Arc::new(p);
        let mut st = self.lock();
        st.policy = Arc::clone(&p);
        st.stamp = stamp;
        st.checked = (self.clock)();
        p
    }

    /// The home this store resolves `~` against (None: the process's).
    pub fn home(&self) -> Option<&str> {
        self.opts.home.as_deref()
    }
}
