//! A hub-side mirror of the node's policy validator (`ecnode::policy::parse`, with the size cap and JSON decoding
//! of `ecnode::policy::load_with`), for the config editor's "Validate" button: it runs the node's rules on pasted
//! policy text, writes nothing and needs no `ecnode` at runtime (`ecnode` is a dev-dependency of the hub only).
//!
//! Interface (`config_check` and the tests use it): `MAX_POLICY_BYTES`, the key lists `TOP_KEYS` / `DOCKER_KEYS` /
//! `ROOT_KEYS` / `SCRIPT_KEYS` / `ACTION_KEYS`, `PolicyOutline` and `check_policy_text`.
//!
//! The checks, their order and their error strings are the node's; `tests/test_policy_check.rs` proves that against
//! the real `ecnode::policy::parse(data, None, None)` (and `load_with` for the size cap and JSON errors). Keep the two
//! in step: a change to the node's rules needs the same change here, and the differential test fails until it has.
//!
//! What the hub cannot check, deliberately (`ecnode check --policy <file>` on the node is the final word):
//! * The node's home directory and the policy file's own path are unknown here, so the mirror validates as
//!   `parse(data, None, None)`: a model root that is the home directory, holds it, or overlaps `~/.ssh`,
//!   `~/.config/ecnode`, `~/.local/state/ecnode` or the policy file's directory is refused only on the node.
//! * Symlinks: the node also compares a root's realpath, which depends on the node's filesystem. The hub compares
//!   the normalised (lexical) path only, so a root that reaches `/etc/ecnode`, a forbidden root such as `/` or
//!   `/etc`, or the home directory through a symlink is refused only on the node. For a path with no symlinks on it this is exactly the node's answer: the node keeps the realpath only
//!   when it differs from the normalised path, and a root written with a leading `//` (the one lexical form realpath
//!   would fold) is refused before either comparison.
//! * File-level checks (a symlink, not a regular file, group/world-writable, the owner, the directory's owner and
//!   mode) concern the file on the node and happen only there.
//! * Wording of the file-level errors: the node says "policy file <path> is larger than 262144 bytes; running in
//!   monitor mode" and "policy file <path> is not valid JSON (...); running in monitor mode"; the hub says "the
//!   policy is larger than 262144 bytes" and "the policy is not valid JSON (...)" (same detail inside the brackets).
//!   The size cap counts the text's UTF-8 bytes, as the node counts the file's bytes; the node's invalid-UTF-8 error
//!   cannot arise because the hub is handed text. Validation errors are the node's strings exactly.

use std::sync::LazyLock;

use ec_proto::pyfmt::{repr_str, repr_value, take_chars};
use ec_proto::sanitize::truthy;
use regex::Regex;
use serde_json::{Map, Value};

/// The node's cap on the policy file (`ecnode::policy::MAX_POLICY_BYTES`).
pub const MAX_POLICY_BYTES: usize = 256 * 1024;
/// Allowed top-level keys (keys starting with "_" are comments), in the order `parse` lists them.
pub const TOP_KEYS: &[&str] = &[
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
];
/// Allowed keys of `docker`.
pub const DOCKER_KEYS: &[&str] = &["binary", "context"];
/// Allowed keys of a `model_roots` entry.
pub const ROOT_KEYS: &[&str] = &["id", "path", "kind", "writable", "label"];
/// Allowed keys of a `scripts` entry.
pub const SCRIPT_KEYS: &[&str] = &[
    "name",
    "label",
    "engine",
    "description",
    "containers",
    "status_action",
    "timeout_s",
    "actions",
];
/// Allowed keys of a script's `actions` entry.
pub const ACTION_KEYS: &[&str] = &["name", "label", "argv", "args", "confirm", "danger"];

const MODES: &[&str] = &["monitor", "manage"];
const KINDS: &[&str] = &[
    "auto", "hf-home", "hf-hub", "gguf", "plain", "ollama", "other",
];
const EXTERNAL: &[&str] = &["none", "stop_start"];
/// The only protected directory the hub can know (the others depend on the node's home and policy path).
const SYSTEM_POLICY_DIR: &str = "/etc/ecnode";
/// Directories that must never be a model root themselves (the node's `FORBIDDEN_ROOTS`).
const FORBIDDEN_ROOTS: &[&str] = &[
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

// Image references, the node's patterns verbatim: [domain[:port]/]path[:tag][@sha256:digest]
const COMP: &str = r"[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*";
const DOMAIN: &str = r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)*(?::[0-9]{1,5})?/)";
const TAG: &str = r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}";
const DIGEST: &str = r"sha256:[a-f0-9]{64}";

fn repo_re() -> String {
    format!("{DOMAIN}?{COMP}(?:/{COMP})*")
}

/// A Python `re.fullmatch` pattern. None only if the pattern does not compile, which then refuses every value
/// (fail closed; the differential test would show it at once).
fn full(p: &str) -> Option<Regex> {
    Regex::new(&format!(r"\A(?:{p})\z")).ok()
}

static IMAGE_REF: LazyLock<Option<Regex>> = LazyLock::new(|| {
    full(&format!(
        r"(?P<repo>{})(?::(?P<tag>{TAG}))?(?:@(?P<digest>{DIGEST}))?",
        repo_re()
    ))
});
static PAT_TAG: LazyLock<Option<Regex>> =
    LazyLock::new(|| full(&format!(r"(?P<repo>{}):\*", repo_re())));
static PAT_DIGEST: LazyLock<Option<Regex>> =
    LazyLock::new(|| full(&format!(r"(?P<repo>{})@sha256:\*", repo_re())));
static ROOT_ID: LazyLock<Option<Regex>> = LazyLock::new(|| full("[a-z0-9][a-z0-9_-]{0,31}"));
static SCRIPT_NAME: LazyLock<Option<Regex>> = LazyLock::new(|| full("[a-z0-9][a-z0-9_-]{0,31}"));
static NETWORK: LazyLock<Option<Regex>> = LazyLock::new(|| full("[A-Za-z0-9][A-Za-z0-9_.-]{0,63}"));
static CONTEXT: LazyLock<Option<Regex>> =
    LazyLock::new(|| full("[A-Za-z0-9][A-Za-z0-9_.+-]{0,63}"));
static BINARY: LazyLock<Option<Regex>> = LazyLock::new(|| full("/[A-Za-z0-9_./+-]{1,255}"));

fn is_match(re: &Option<Regex>, s: &str) -> bool {
    re.as_ref().is_some_and(|r| r.is_match(s))
}

/// What the node would run with, in brief (for the editor's summary line).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyOutline {
    pub mode: String,
    pub engine_images: usize,
    pub model_roots: usize,
    pub writable_roots: usize,
    pub scripts: usize,
    pub allow_pull: bool,
    pub allow_delete: bool,
    pub allow_ipc_host: bool,
    pub external_control: String,
}

type Check<T> = Result<T, String>;

/// Decode `text` as the node decodes its policy file, then validate it as `ecnode::policy::parse` does.
pub fn check_policy_text(text: &str) -> Result<PolicyOutline, String> {
    if text.len() > MAX_POLICY_BYTES {
        return Err(format!(
            "the policy is larger than {MAX_POLICY_BYTES} bytes"
        ));
    }
    let data: Value = serde_json::from_str(text).map_err(|e| {
        format!(
            "the policy is not valid JSON ({})",
            take_chars(&json_error_text(text, &e), 160)
        )
    })?;
    check_policy(&data)
}

/// The node's `json_error_text`: a JSON syntax error in the words of Python's json module, at serde's position.
fn json_error_text(text: &str, e: &serde_json::Error) -> String {
    let (line, col) = (e.line(), e.column());
    // char offset of (line, col), 1-based column as Python counts it
    let mut pos = 0usize;
    for (i, l) in text.split('\n').enumerate() {
        if i + 1 == line {
            pos += l.chars().take(col.saturating_sub(1)).count();
            break;
        }
        pos += l.chars().count() + 1;
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

// ── small Python helpers (the node's util::py, only what the checks need) ─────────────────────────────────────

/// `len(s)` in characters.
fn len(s: &str) -> usize {
    s.chars().count()
}

/// `posixpath.normpath` of text.
fn normpath(path: &str) -> String {
    if path.is_empty() {
        return ".".into();
    }
    let initial = if path.starts_with('/') {
        if path.starts_with("//") && !path.starts_with("///") {
            "//"
        } else {
            "/"
        }
    } else {
        ""
    };
    let mut comps: Vec<&str> = Vec::new();
    for comp in path.split('/') {
        if comp.is_empty() || comp == "." {
            continue;
        }
        if comp != ".." || (initial.is_empty() && comps.is_empty()) || comps.last() == Some(&"..") {
            comps.push(comp);
        } else {
            comps.pop();
        }
    }
    let out = format!("{initial}{}", comps.join("/"));
    if out.is_empty() { ".".into() } else { out }
}

/// The node's `confine::inside`: `child` is `root` or lies below it (whole segments).
fn inside(child: &str, root: &str) -> bool {
    if child == root {
        return true;
    }
    let root = root.trim_end_matches('/');
    child.len() > root.len()
        && child.starts_with(root)
        && child.as_bytes().get(root.len()) == Some(&b'/')
}

// ── schema (the node's helpers, same messages) ──────────────────────────────────────────────────────────────

fn keys(obj: &Map<String, Value>, allowed: &[&str], where_: &str) -> Check<()> {
    for k in obj.keys() {
        if !k.starts_with('_') && !allowed.contains(&k.as_str()) {
            let mut sorted: Vec<&str> = allowed.to_vec();
            sorted.sort_unstable();
            return Err(format!(
                "{where_}: unknown key {} (allowed: {})",
                repr_str(k),
                sorted.join(", ")
            ));
        }
    }
    Ok(())
}

fn bool_(obj: &Map<String, Value>, key: &str, where_: &str) -> Check<bool> {
    match obj.get(key) {
        None => Ok(false),
        Some(Value::Bool(b)) => Ok(*b),
        Some(_) => Err(format!("{where_}.{key} must be true or false")),
    }
}

/// An optional (or, with `required`, mandatory) string of at most `limit` characters, matching `pattern` if given.
/// Ok(None) when absent or null and not required.
fn str_<'a>(
    obj: &'a Map<String, Value>,
    key: &str,
    where_: &str,
    limit: usize,
    pattern: Option<&Option<Regex>>,
    required: bool,
) -> Check<Option<&'a str>> {
    let v = match obj.get(key) {
        None | Some(Value::Null) => {
            if required {
                return Err(format!("{where_}.{key} is required"));
            }
            return Ok(None);
        }
        Some(v) => v,
    };
    let s = match v {
        Value::String(s) if !s.is_empty() && len(s) <= limit && !s.contains('\0') => s,
        _ => {
            return Err(format!(
                "{where_}.{key} must be a non-empty string of at most {limit} characters"
            ));
        }
    };
    if let Some(p) = pattern
        && !is_match(p, s)
    {
        return Err(format!(
            "{where_}.{key}: {} is not a valid value",
            repr_str(s)
        ));
    }
    Ok(Some(s))
}

fn int_(v: &Value, where_: &str, lo: i64, hi: i64) -> Check<i64> {
    match v {
        Value::Number(n) if !n.is_f64() => match n.as_i64() {
            Some(i) if lo <= i && i <= hi => Ok(i),
            _ => Err(format!("{where_} must be a whole number from {lo} to {hi}")),
        },
        _ => Err(format!("{where_} must be a whole number from {lo} to {hi}")),
    }
}

fn list<'a>(
    obj: &'a Map<String, Value>,
    key: &str,
    where_: &str,
    cap: usize,
) -> Check<&'a [Value]> {
    match obj.get(key) {
        None | Some(Value::Null) => Ok(&[]),
        Some(Value::Array(a)) if a.len() <= cap => Ok(a.as_slice()),
        Some(_) => Err(format!(
            "{where_}.{key} must be a list of at most {cap} items"
        )),
    }
}

/// `value == 1` in Python (True and 1.0 count).
fn py_eq_one(v: &Value) -> bool {
    match v {
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64() == Some(1.0),
        _ => false,
    }
}

/// The string in `set` that `v` is, if it is one.
fn str_in<'a>(v: &'a Value, set: &[&str]) -> Option<&'a str> {
    v.as_str().filter(|s| set.contains(s))
}

// ── image references ────────────────────────────────────────────────────────────────────────────────────────

fn valid_image(r: &str) -> bool {
    len(r) <= 256 && is_match(&IMAGE_REF, r)
}

fn valid_image_pattern(p: &str) -> bool {
    is_match(&PAT_TAG, p) || is_match(&PAT_DIGEST, p) || valid_image(p)
}

// ── model roots ─────────────────────────────────────────────────────────────────────────────────────────────

/// The node's `check_root_overlap` with no policy path and no home: only /etc/ecnode is known here.
fn check_root_overlap(norm: &str, where_: &str) -> Check<()> {
    // The node's `path_forms` of a symlink-free path is the normalised path alone (see the module doc).
    if inside(norm, SYSTEM_POLICY_DIR) || inside(SYSTEM_POLICY_DIR, norm) {
        return Err(format!(
            "{where_}.path: {} overlaps the node's system policy directory ({SYSTEM_POLICY_DIR}); a model root must \
             not be, contain or lie inside it",
            repr_str(norm)
        ));
    }
    Ok(())
}

/// The node's `root_path` with no home.
fn root_path(p: Option<&Value>, where_: &str) -> Check<String> {
    let p = match p {
        Some(Value::String(s)) if s.starts_with('/') && len(s) <= 512 => s,
        _ => {
            return Err(format!(
                "{where_}.path must be an absolute path of at most 512 characters"
            ));
        }
    };
    // POSIX keeps a leading `//` as written (normpath does too), so `//etc` would not match the forbidden `/etc`.
    if p.starts_with("//") {
        return Err(format!(
            "{where_}.path: {} must start with a single '/'",
            repr_str(p)
        ));
    }
    if p.chars().any(|c| ":,\0\n\r\\".contains(c))
        || p.split('/').any(|seg| seg == "." || seg == "..")
    {
        return Err(format!(
            "{where_}.path: {} must not contain ':', ',', '\\', '.' or '..' segments",
            repr_str(p)
        ));
    }
    let norm = normpath(p);
    if FORBIDDEN_ROOTS.contains(&norm.as_str()) {
        return Err(format!(
            "{where_}.path: {} is too broad for a model root; name the directory that holds the models (for \
             example /data/hf)",
            repr_str(&norm)
        ));
    }
    Ok(norm)
}

/// One `model_roots` entry; returns (id, writable).
fn model_root(r: &Value, where_: &str, taken: &[String]) -> Check<(String, bool)> {
    let Value::Object(r) = r else {
        return Err(format!("{where_} must be an object"));
    };
    keys(r, ROOT_KEYS, where_)?;
    let rid = str_(r, "id", where_, 32, Some(&ROOT_ID), true)?.unwrap_or_default();
    if taken.iter().any(|x| x == rid) {
        return Err(format!("{where_}.id: {} is used twice", repr_str(rid)));
    }
    let kind = str_(r, "kind", where_, 16, None, false)?.unwrap_or("auto");
    if !KINDS.contains(&kind) {
        return Err(format!("{where_}.kind must be one of {}", KINDS.join(", ")));
    }
    let path = root_path(r.get("path"), where_)?;
    check_root_overlap(&path, where_)?;
    let writable = bool_(r, "writable", where_)?;
    str_(r, "label", where_, 64, None, false)?;
    Ok((rid.to_string(), writable))
}

// ── scripts ─────────────────────────────────────────────────────────────────────────────────────────────────

/// One action of a script; returns its name.
fn action(a: &Value, aw: &str, taken: &[String]) -> Check<String> {
    let Value::Object(a) = a else {
        return Err(format!("{aw} must be an object"));
    };
    keys(a, ACTION_KEYS, aw)?;
    let an = str_(a, "name", aw, 32, Some(&SCRIPT_NAME), true)?.unwrap_or_default();
    if taken.iter().any(|x| x == an) {
        return Err(format!("{aw}.name: {} is used twice", repr_str(an)));
    }
    let argv = match a.get("argv") {
        Some(Value::Array(v))
            if (1..=64).contains(&v.len())
                && v.iter().all(|x| {
                    x.as_str()
                        .is_some_and(|x| !x.is_empty() && len(x) <= 4096 && !x.contains('\0'))
                }) =>
        {
            v
        }
        _ => {
            return Err(format!(
                "{aw}.argv must be a list of 1-64 non-empty strings (the program, then its arguments)"
            ));
        }
    };
    let program = argv.first().and_then(Value::as_str).unwrap_or_default();
    if !program.starts_with('/') || program.contains('\n') {
        return Err(format!(
            "{aw}.argv[0] must be an absolute path to the program (no shell, no PATH lookup)"
        ));
    }
    let args = list(a, "args", aw, 32)?;
    if !args.iter().all(|x| {
        x.as_str()
            .is_some_and(|x| !x.is_empty() && len(x) <= 64 && !x.contains('\0'))
    }) {
        return Err(format!(
            "{aw}.args must be a list of allowed argument values (each 1-64 characters)"
        ));
    }
    str_(a, "label", aw, 64, None, false)?;
    bool_(a, "confirm", aw)?;
    bool_(a, "danger", aw)?;
    Ok(an.to_string())
}

/// One `scripts` entry; returns its name.
fn script(s: &Value, where_: &str, taken: &[String]) -> Check<String> {
    let Value::Object(s) = s else {
        return Err(format!("{where_} must be an object"));
    };
    keys(s, SCRIPT_KEYS, where_)?;
    let name = str_(s, "name", where_, 32, Some(&SCRIPT_NAME), true)?.unwrap_or_default();
    if taken.iter().any(|x| x == name) {
        return Err(format!("{where_}.name: {} is used twice", repr_str(name)));
    }
    for (j, c) in list(s, "containers", where_, 16)?.iter().enumerate() {
        if !c
            .as_str()
            .is_some_and(|c| !c.is_empty() && len(c) <= 128 && !c.contains('\0'))
        {
            return Err(format!(
                "{where_}.containers[{j}] must be a container name pattern like 'myproj-sglang-*'"
            ));
        }
    }
    let mut actions: Vec<String> = Vec::new();
    for (j, a) in list(s, "actions", where_, 16)?.iter().enumerate() {
        let an = action(a, &format!("{where_}.actions[{j}]"), &actions)?;
        actions.push(an);
    }
    if let Some(st) = str_(s, "status_action", where_, 32, None, false)?
        && !actions.iter().any(|x| x == st)
    {
        return Err(format!(
            "{where_}.status_action must name one of the script's actions"
        ));
    }
    if let Some(t) = s.get("timeout_s") {
        int_(t, &format!("{where_}.timeout_s"), 1, 86400)?;
    }
    str_(s, "label", where_, 64, None, false)?;
    str_(s, "engine", where_, 32, None, false)?;
    str_(s, "description", where_, 500, None, false)?;
    Ok(name.to_string())
}

// ── the policy ──────────────────────────────────────────────────────────────────────────────────────────────

/// `ecnode::policy::parse(data, None, None)`, returning the outline instead of the policy.
fn check_policy(data: &Value) -> Check<PolicyOutline> {
    let Value::Object(data) = data else {
        return Err("the policy must be one JSON object".into());
    };
    keys(data, TOP_KEYS, "policy")?;
    if !data.get("version").is_none_or(py_eq_one) {
        return Err("policy.version must be 1 (this node understands policy version 1)".into());
    }
    let mode = match data.get("mode") {
        None => "monitor",
        Some(v) => str_in(v, MODES).ok_or("policy.mode must be \"monitor\" or \"manage\"")?,
    };

    let empty = Map::new();
    let dk = match data.get("docker") {
        Some(v) if truthy(v) => match v {
            Value::Object(m) => m,
            _ => {
                return Err(
                    "policy.docker must be an object like {\"binary\": \"docker\", \"context\": null}".into(),
                );
            }
        },
        _ => &empty,
    };
    keys(dk, DOCKER_KEYS, "policy.docker")?;
    let binary = str_(dk, "binary", "policy.docker", 256, None, false)?.unwrap_or("docker");
    if binary != "docker" && (!is_match(&BINARY, binary) || binary.split('/').any(|s| s == "..")) {
        return Err(
            "policy.docker.binary must be \"docker\" or an absolute path to the docker CLI".into(),
        );
    }
    str_(dk, "context", "policy.docker", 64, Some(&CONTEXT), false)?;

    let images = list(data, "engine_images", "policy", 64)?;
    for (i, p) in images.iter().enumerate() {
        if !p.as_str().is_some_and(valid_image_pattern) {
            return Err(format!(
                "policy.engine_images[{i}]: {} must be an exact image reference, 'repository:*' or \
                 'repository@sha256:*' with a literal repository",
                repr_value(p)
            ));
        }
    }
    match data.get("gateway_image") {
        None | Some(Value::Null) => {}
        Some(v) if v.as_str().is_some_and(valid_image) => {}
        Some(_) => {
            return Err(
                "policy.gateway_image must be an exact image reference (best: nginx@sha256:<digest>)".into(),
            );
        }
    }
    str_(data, "engine_network", "policy", 64, Some(&NETWORK), false)?;

    // A missing port_range is the node's default [18000, 18099], which passes.
    if let Some(pr) = data.get("port_range") {
        let Some([first, last]) = pr.as_array().map(Vec::as_slice) else {
            return Err("policy.port_range must be [first, last], e.g. [18000, 18099]".into());
        };
        let lo = int_(first, "policy.port_range[0]", 1024, 65535)?;
        int_(last, "policy.port_range[1]", lo, 65535.min(lo + 1000))?;
    }
    for (i, p) in list(data, "discovery_ports", "policy", 128)?
        .iter()
        .enumerate()
    {
        int_(p, &format!("policy.discovery_ports[{i}]"), 1, 65535)?;
    }

    let mut roots: Vec<String> = Vec::new();
    let mut writable_roots = 0usize;
    for (i, r) in list(data, "model_roots", "policy", 32)?.iter().enumerate() {
        let (rid, writable) = model_root(r, &format!("policy.model_roots[{i}]"), &roots)?;
        roots.push(rid);
        writable_roots += usize::from(writable);
    }

    let mut scripts: Vec<String> = Vec::new();
    for (i, s) in list(data, "scripts", "policy", 32)?.iter().enumerate() {
        let name = script(s, &format!("policy.scripts[{i}]"), &scripts)?;
        scripts.push(name);
    }

    let external_control = match data.get("external_control") {
        None => "none",
        Some(v) => str_in(v, EXTERNAL)
            .ok_or("policy.external_control must be \"none\" or \"stop_start\"")?,
    };
    if let Some(v) = data.get("max_shm_gib")
        && !v.as_f64().is_some_and(|f| f > 0.0 && f <= 4096.0)
    {
        return Err("policy.max_shm_gib must be a number above 0 and at most 4096".into());
    }
    let allow_pull = bool_(data, "allow_pull", "policy")?;
    let allow_delete = bool_(data, "allow_delete", "policy")?;
    let allow_ipc_host = bool_(data, "allow_ipc_host", "policy")?;
    Ok(PolicyOutline {
        mode: mode.to_string(),
        engine_images: images.len(),
        model_roots: roots.len(),
        writable_roots,
        scripts: scripts.len(),
        allow_pull,
        allow_delete,
        allow_ipc_host,
        external_control: external_control.to_string(),
    })
}
