//! Python: backend/src/engine_console/services/hosts.py (and tests/fleet/test_hosts.py).
//!
//! Which machines the console manages and exactly how each one is reached (contract v2, docs/ARCHITECTURE.md §10.3).
//!
//! hosts.json decides what this process runs `ssh` against, so it is read once at start-up, from disk only, and only
//! if it could not have been planted by someone else: a plain file (no symlink), owned by the running user, writable
//! by that user alone. Nothing a browser or a node sends can change it or reach an ssh argument.
//!
//! Everything that ends up on a command line is validated here with whole-string matches (`\A...\z`: Python's
//! `fullmatch`, never a `$` that would accept a trailing newline) and is passed as its own argv element: no shell is
//! ever involved.
//!
//! The LOCAL node argv is decided by PORTING.md D8: NODE_COMMAND when set (and not empty), else the `ecnode` binary
//! next to the running `engine-console` executable, with `--policy NODE_POLICY` when set.

use std::collections::HashSet;
use std::io::Read as _;
use std::os::unix::fs::{DirBuilderExt as _, MetadataExt as _, OpenOptionsExt as _};
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use std::time::Duration;

use regex::Regex;
use serde_json::Value;

use crate::config::Settings;
use crate::domain::fleet::{HOST_ID_PATTERN, HostConfig, LOCAL};

pub use crate::config::DEFAULT_REMOTE_COMMAND;
pub const MAX_HOSTS: usize = 32;
pub const MAX_FILE_BYTES: usize = 64 * 1024;
pub const LABEL_MAX: usize = 64;
pub const KEY_COMMENT: &str = "engine-console-hub";
pub const HOST_KEYS: &[&str] = &["id", "label", "ssh", "port", "ssh_config", "enabled"];
/// The variables the local node may inherit (hosts.py `_LOCAL_ENV_KEYS`).
pub const LOCAL_ENV_KEYS: &[&str] = &[
    "HOME",
    "PATH",
    "USER",
    "LOGNAME",
    "LANG",
    "XDG_RUNTIME_DIR",
    "DOCKER_HOST",
    "DOCKER_CONTEXT",
    "DOCKER_CONFIG",
];

fn whole(pat: &str) -> Regex {
    Regex::new(&format!(r"\A(?:{pat})\z")).unwrap_or_else(|_| unreachable!())
}

static HOST_ID: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(HOST_ID_PATTERN).unwrap_or_else(|_| unreachable!()));
// [user@]host. No leading '-' (ssh would read it as an option, e.g. -oProxyCommand=...), no '%' (ssh token
// expansion), no whitespace or shell metacharacters. An ssh_config alias has the same shape.
static SSH_DEST: LazyLock<Regex> = LazyLock::new(|| {
    whole(r"(?:[A-Za-z0-9_][A-Za-z0-9._-]{0,31}@)?[A-Za-z0-9][A-Za-z0-9._-]{0,252}")
});
// What sshd accepts in `from="..."`: comma-separated address/host patterns (CIDR, '*', '?', '!'). No quote or space,
// so the value can never close the option string it is written into.
static FROM: LazyLock<Regex> = LazyLock::new(|| whole(r"[0-9A-Za-z.:/*?!,-]{1,256}"));
// An absolute path and plain arguments: this string is both the remote command and the `command="..."` of the
// authorized_keys line, so it must not be able to break out of either.
static REMOTE_CMD: LazyLock<Regex> =
    LazyLock::new(|| whole(r"/[A-Za-z0-9_./+-]{1,200}(?: [A-Za-z0-9_./=:+-]{1,64}){0,8}"));
static PUBKEY_TYPE: LazyLock<Regex> = LazyLock::new(|| {
    whole(
        r"ssh-ed25519|ssh-rsa|ecdsa-sha2-nistp(?:256|384|521)|sk-ssh-ed25519@openssh\.com|sk-ecdsa-sha2-nistp256@openssh\.com",
    )
});
static PUBKEY_B64: LazyLock<Regex> = LazyLock::new(|| whole(r"[A-Za-z0-9+/]{16,4096}={0,3}"));

/// hosts.json (or a key file, or an ssh argument) is not acceptable. The text says what to fix.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum HostsError {
    #[error("{0}")]
    Invalid(String),
    /// The console's own key does not exist yet (`engine-console hosts init-key` creates it).
    #[error("{0}")]
    NoHubKey(String),
}

fn invalid(msg: impl Into<String>) -> HostsError {
    HostsError::Invalid(msg.into())
}

// ── Python string predicates ────────────────────────────────────────────────────────────────────────────
/// Python's `str.isprintable()` for one character: false for the Unicode categories Cc, Cf, Cs, Co, Zl, Zp and Zs
/// (except the ASCII space). Unassigned code points (Cn) other than the noncharacters are taken as printable: there
/// is no Unicode table in this build, and an unassigned character in a label or a node message is harmless.
pub fn py_isprintable(c: char) -> bool {
    let n = c as u32;
    if c == ' ' {
        return true;
    }
    let cc = n < 0x20 || (0x7f..=0x9f).contains(&n);
    let zs =
        matches!(n, 0xa0 | 0x1680 | 0x202f | 0x205f | 0x3000) || (0x2000..=0x200a).contains(&n);
    let zl_zp = n == 0x2028 || n == 0x2029;
    let cf = matches!(
        n,
        0xad | 0x61c | 0x6dd | 0x70f | 0x8e2 | 0x180e | 0xfeff | 0x110bd | 0x110cd | 0xe0001
    ) || (0x600..=0x605).contains(&n)
        || (0x890..=0x891).contains(&n)
        || (0x200b..=0x200f).contains(&n)
        || (0x202a..=0x202e).contains(&n)
        || (0x2060..=0x2064).contains(&n)
        || (0x2066..=0x206f).contains(&n)
        || (0xfff9..=0xfffb).contains(&n)
        || (0x13430..=0x1343f).contains(&n)
        || (0x1bca0..=0x1bca3).contains(&n)
        || (0x1d173..=0x1d17a).contains(&n)
        || (0xe0020..=0xe007f).contains(&n);
    let co = (0xe000..=0xf8ff).contains(&n)
        || (0xf0000..=0xffffd).contains(&n)
        || (0x10_0000..=0x10_fffd).contains(&n);
    let nonchar = (0xfdd0..=0xfdef).contains(&n) || (n & 0xfffe) == 0xfffe;
    !(cc || zs || zl_zp || cf || co || nonchar)
}

/// Python's `str.isspace()` for one character (Rust's White_Space plus the ASCII separators 0x1c-0x1f).
pub fn py_isspace(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// `str.split()` with no arguments.
pub fn py_split(s: &str) -> Vec<&str> {
    s.split(py_isspace).filter(|p| !p.is_empty()).collect()
}

/// `str.strip()` with no arguments.
pub fn py_strip(s: &str) -> &str {
    s.trim_matches(py_isspace)
}

/// The text of an OS error the way Python's `e.strerror` gives it ("No such file or directory").
pub fn strerror(e: &std::io::Error) -> String {
    match e.raw_os_error() {
        Some(code) => nix::errno::Errno::from_raw(code).desc().to_string(),
        None => e.to_string(),
    }
}

// ── hosts.json ───────────────────────────────────────────────────────────────────────────────────────────
/// No hosts.json: the console manages the machine it runs on and nothing else (label = printable hostname).
pub fn default_hosts() -> Vec<HostConfig> {
    let name = nix::unistd::gethostname()
        .map(|h| h.to_string_lossy().into_owned())
        .unwrap_or_default();
    default_hosts_named(&name)
}

/// `default_hosts()` for a given hostname (the test seam Python gets by patching `socket.gethostname`).
pub fn default_hosts_named(hostname: &str) -> Vec<HostConfig> {
    let name: String = hostname
        .chars()
        .filter(|c| py_isprintable(*c))
        .take(LABEL_MAX)
        .collect();
    vec![HostConfig {
        id: LOCAL.into(),
        label: if name.is_empty() {
            "This machine".into()
        } else {
            name
        },
        ssh: None,
        port: None,
        ssh_config: false,
        enabled: true,
    }]
}

/// Validate a decoded hosts.json document. Strict on purpose: unknown keys are errors, not ignored, so a typo such
/// as "sshconfig" can never silently leave a safer default in place.
pub fn parse_hosts(doc: &Value) -> Result<Vec<HostConfig>, HostsError> {
    let Some(obj) = doc
        .as_object()
        .filter(|o| o.len() == 1 && o.contains_key("hosts"))
    else {
        return Err(invalid(
            r#"the file must be {"hosts": [ ... ]} and nothing else"#,
        ));
    };
    let raw = match obj.get("hosts") {
        Some(Value::Array(a)) if (1..=MAX_HOSTS).contains(&a.len()) => a,
        _ => {
            return Err(invalid(format!(
                "hosts must be a list of 1-{MAX_HOSTS} machines"
            )));
        }
    };
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for (i, h) in raw.iter().enumerate() {
        let cfg = parse_host(h, i)?;
        if !seen.insert(cfg.id.clone()) {
            return Err(invalid(format!(
                "hosts[{i}]: id {} is used twice",
                ec_proto::pyfmt::repr_str(&cfg.id)
            )));
        }
        out.push(cfg);
    }
    Ok(out)
}

fn port_error(where_: &str) -> HostsError {
    invalid(format!(
        "{where_}: port must be a whole number from 1 to 65535"
    ))
}

fn parse_host(h: &Value, i: usize) -> Result<HostConfig, HostsError> {
    let where_ = format!("hosts[{i}]");
    let Some(h) = h.as_object() else {
        return Err(invalid(format!("{where_} must be an object")));
    };
    let mut extra: Vec<&str> = h
        .keys()
        .map(String::as_str)
        .filter(|k| !HOST_KEYS.contains(k))
        .collect();
    if !extra.is_empty() {
        extra.sort_unstable();
        return Err(invalid(format!(
            "{where_}: unknown key(s) {}",
            extra.join(", ")
        )));
    }
    let hid = match h.get("id") {
        Some(Value::String(s)) if HOST_ID.is_match(s) => s.clone(),
        _ => {
            return Err(invalid(format!(
                "{where_}: id must be lowercase letters, digits and '-', 1-32 long, not starting with '-'"
            )));
        }
    };
    let where_ = format!("host {}", ec_proto::pyfmt::repr_str(&hid));
    // a label is a convenience: the id is shown when none is given
    let label = match h.get("label") {
        None => Some(hid.clone()),
        Some(Value::String(s)) => Some(s.clone()),
        Some(_) => None,
    }
    .filter(|l| {
        let n = l.chars().count();
        (1..=LABEL_MAX).contains(&n) && l.chars().all(py_isprintable)
    });
    let Some(label) = label else {
        return Err(invalid(format!(
            "{where_}: label must be 1-{LABEL_MAX} printable characters"
        )));
    };
    let dest = h.get("ssh").filter(|v| !v.is_null());
    let port = h.get("port").filter(|v| !v.is_null());
    let ssh_config = h.get("ssh_config").cloned().unwrap_or(Value::Bool(false));
    let enabled = h.get("enabled").cloned().unwrap_or(Value::Bool(true));
    let (Value::Bool(ssh_config), Value::Bool(enabled)) = (ssh_config, enabled) else {
        return Err(invalid(format!(
            "{where_}: ssh_config and enabled must be true or false"
        )));
    };
    if hid == LOCAL {
        if dest.is_some() || port.is_some() || ssh_config {
            return Err(invalid(format!(
                "{where_}: \"local\" is the machine the console runs on; it takes no ssh, port or ssh_config"
            )));
        }
        return Ok(HostConfig {
            id: hid,
            label,
            ssh: None,
            port: None,
            ssh_config,
            enabled,
        });
    }
    let dest = match dest {
        Some(Value::String(d)) if SSH_DEST.is_match(d) => d.clone(),
        _ => {
            return Err(invalid(format!(
                "{where_}: ssh must be user@address or an ssh_config alias (letters, digits, '.', '_', '-'; not starting with '-')"
            )));
        }
    };
    let port = match port {
        None => None,
        // a JSON integer (Python int, never bool or float) from 1 to 65535
        Some(Value::Number(n)) if n.is_i64() || n.is_u64() => Some(
            n.as_u64()
                .and_then(|p| u16::try_from(p).ok())
                .filter(|p| *p >= 1)
                .ok_or_else(|| port_error(&where_))?,
        ),
        Some(_) => return Err(port_error(&where_)),
    };
    Ok(HostConfig {
        id: hid,
        label,
        ssh: Some(dest),
        port,
        ssh_config,
        enabled,
    })
}

/// The configured machines, or just this one when the file does not exist. Errors start with the path.
pub fn load_hosts(path: &Path) -> Result<Vec<HostConfig>, HostsError> {
    load_hosts_as(path, nix::unistd::geteuid().as_raw())
}

/// `load_hosts` as the user `euid` (the test seam Python gets by patching `os.geteuid`).
pub fn load_hosts_as(path: &Path, euid: u32) -> Result<Vec<HostConfig>, HostsError> {
    let p = path.display();
    // O_NOFOLLOW: a symlink in the last position is refused, so the file cannot be redirected somewhere the owner
    // checks below would not cover. O_NONBLOCK: opening a FIFO must not hang start-up. (Rust opens with O_CLOEXEC.)
    let opened = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(nix::fcntl::OFlag::O_NOFOLLOW.bits() | nix::fcntl::OFlag::O_NONBLOCK.bits())
        .open(path);
    let mut f = match opened {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(default_hosts()),
        Err(e) if e.raw_os_error() == Some(nix::errno::Errno::ELOOP as i32) => {
            return Err(invalid(format!(
                "{p}: is a symlink; it decides what the console runs ssh against, so it must be a plain file"
            )));
        }
        Err(e) => {
            return Err(invalid(format!("{p}: cannot open it: {}", strerror(&e))));
        }
    };
    // Every check is made on the descriptor we read from, not on the path again, so the file cannot be swapped
    // between the check and the read.
    let st = f
        .metadata()
        .map_err(|e| invalid(format!("{p}: cannot read it: {}", strerror(&e))))?;
    if !st.file_type().is_file() {
        return Err(invalid(format!("{p}: is not a regular file")));
    }
    if st.uid() != euid {
        return Err(invalid(format!(
            "{p}: must be owned by the user running the console: it decides what the console runs ssh against"
        )));
    }
    if st.mode() & 0o022 != 0 {
        return Err(invalid(format!(
            "{p}: must not be group- or world-writable (chmod 600)"
        )));
    }
    let mut data = Vec::new();
    (&mut f)
        .take((MAX_FILE_BYTES + 1) as u64)
        .read_to_end(&mut data)
        .map_err(|e| invalid(format!("{p}: cannot read it: {}", strerror(&e))))?;
    drop(f);
    if data.len() > MAX_FILE_BYTES {
        return Err(invalid(format!(
            "{p}: is over {} KiB",
            MAX_FILE_BYTES / 1024
        )));
    }
    let doc = match decode_strict(&data) {
        Ok(v) => v,
        Err(StrictError::Hosts(msg)) => return Err(invalid(format!("{p}: {msg}"))),
        Err(StrictError::Json(msg)) => {
            let short: String = msg.chars().take(120).collect();
            return Err(invalid(format!("{p}: not valid JSON: {short}")));
        }
    };
    parse_hosts(&doc).map_err(|e| invalid(format!("{p}: {e}")))
}

// ── strict JSON (hosts.py: parse_constant, parse_float, object_pairs_hook) ─────────────────────────────────
enum StrictError {
    /// a HostsError raised by one of the decoder hooks (duplicate key, NaN/Infinity, a non-finite float)
    Hosts(String),
    /// anything else (JSONDecodeError, bad UTF-8, absurd numbers, deep nesting)
    Json(String),
}

const DUP_MARK: &str = "\u{1}hosts-dup\u{1}";

struct StrictValue(Value);

impl<'de> serde::Deserialize<'de> for StrictValue {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        d.deserialize_any(StrictVisitor)
    }
}

struct StrictVisitor;

impl<'de> serde::de::Visitor<'de> for StrictVisitor {
    type Value = StrictValue;

    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("any JSON value")
    }
    fn visit_bool<E>(self, v: bool) -> Result<StrictValue, E> {
        Ok(StrictValue(Value::Bool(v)))
    }
    fn visit_i64<E>(self, v: i64) -> Result<StrictValue, E> {
        Ok(StrictValue(Value::from(v)))
    }
    fn visit_u64<E>(self, v: u64) -> Result<StrictValue, E> {
        Ok(StrictValue(Value::from(v)))
    }
    fn visit_f64<E>(self, v: f64) -> Result<StrictValue, E> {
        Ok(StrictValue(Value::from(v)))
    }
    fn visit_str<E>(self, v: &str) -> Result<StrictValue, E> {
        Ok(StrictValue(Value::String(v.to_string())))
    }
    fn visit_string<E>(self, v: String) -> Result<StrictValue, E> {
        Ok(StrictValue(Value::String(v)))
    }
    fn visit_unit<E>(self) -> Result<StrictValue, E> {
        Ok(StrictValue(Value::Null))
    }
    fn visit_none<E>(self) -> Result<StrictValue, E> {
        Ok(StrictValue(Value::Null))
    }
    fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut seq: A) -> Result<StrictValue, A::Error> {
        let mut out = Vec::new();
        while let Some(StrictValue(v)) = seq.next_element::<StrictValue>()? {
            out.push(v);
        }
        Ok(StrictValue(Value::Array(out)))
    }
    fn visit_map<A: serde::de::MapAccess<'de>>(self, mut map: A) -> Result<StrictValue, A::Error> {
        let mut out = serde_json::Map::new();
        while let Some(k) = map.next_key::<String>()? {
            if out.contains_key(&k) {
                return Err(serde::de::Error::custom(format!(
                    "{DUP_MARK}the key {} appears twice in one object{DUP_MARK}",
                    ec_proto::pyfmt::repr_str(&k)
                )));
            }
            let StrictValue(v) = map.next_value::<StrictValue>()?;
            out.insert(k, v);
        }
        Ok(StrictValue(Value::Object(out)))
    }
}

/// The first token Python's `json` would hand to `parse_constant` (NaN, Infinity, -Infinity; `constant` true) or
/// that `float()` turns into an infinity (`constant` false), with its byte offset, scanning outside strings.
pub fn first_non_finite(text: &str) -> Option<(usize, String, bool)> {
    let b = text.as_bytes();
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'"' => {
                i += 1;
                while i < b.len() && b[i] != b'"' {
                    if b[i] == b'\\' {
                        i += 1;
                    }
                    i += 1;
                }
                i += 1;
            }
            b'N' if text[i..].starts_with("NaN") => return Some((i, "NaN".into(), true)),
            b'I' if text[i..].starts_with("Infinity") => {
                return Some((i, "Infinity".into(), true));
            }
            b'-' if text[i..].starts_with("-Infinity") => {
                return Some((i, "-Infinity".into(), true));
            }
            b'-' | b'0'..=b'9' => {
                let start = i;
                i += 1;
                while i < b.len() && matches!(b[i], b'0'..=b'9' | b'.' | b'e' | b'E' | b'+' | b'-')
                {
                    i += 1;
                }
                let tok = &text[start..i];
                if tok.contains(['.', 'e', 'E']) && tok.parse::<f64>().is_ok_and(f64::is_infinite) {
                    return Some((start, tok.to_string(), false));
                }
            }
            _ => i += 1,
        }
    }
    None
}

/// Byte offset of a serde_json error position (line and column are 1-based; the column counts bytes).
pub fn serde_offset(text: &str, e: &serde_json::Error) -> usize {
    let mut start = 0;
    for (n, line) in text.split_inclusive('\n').enumerate() {
        if n + 1 == e.line() {
            break;
        }
        start += line.len();
    }
    start + e.column().saturating_sub(1)
}

fn decode_strict(data: &[u8]) -> Result<Value, StrictError> {
    let text = std::str::from_utf8(data).map_err(|e| StrictError::Json(e.to_string()))?;
    match serde_json::from_str::<StrictValue>(text) {
        Ok(StrictValue(v)) => Ok(v),
        Err(e) => {
            let msg = e.to_string();
            if let Some(rest) = msg.split(DUP_MARK).nth(1) {
                return Err(StrictError::Hosts(rest.to_string()));
            }
            // Python's decoder reaches a NaN/Infinity/overflowing float before any later syntax error
            if let Some((at, tok, constant)) = first_non_finite(text)
                && at <= serde_offset(text, &e)
            {
                return Err(StrictError::Hosts(if constant {
                    format!(
                        "{tok} is not allowed: the file holds text, booleans and whole numbers only"
                    )
                } else {
                    let t: String = tok.chars().take(20).collect();
                    format!("{t} is not a finite number")
                }));
            }
            Err(StrictError::Json(msg))
        }
    }
}

// ── how each machine is reached ─────────────────────────────────────────────────────────────────────────
/// An absolute path followed by plain arguments (it is both the remote command and the forced command).
pub fn check_remote_command(cmd: &str) -> Result<String, HostsError> {
    if !REMOTE_CMD.is_match(cmd) {
        return Err(invalid(format!(
            "the remote node command must be an absolute path followed by plain arguments, for example {}",
            ec_proto::pyfmt::repr_str(DEFAULT_REMOTE_COMMAND)
        )));
    }
    Ok(cmd.to_string())
}

const SSH_OPTIONS: &[&str] = &[
    "BatchMode=yes",
    "StrictHostKeyChecking=yes",
    "UpdateHostKeys=no",
    "VerifyHostKeyDNS=no",
    "IdentityAgent=none",
    "PKCS11Provider=none",
    "PasswordAuthentication=no",
    "KbdInteractiveAuthentication=no",
    "ControlMaster=no",
    "ControlPath=none",
    "ForwardAgent=no",
    "ForwardX11=no",
    "ClearAllForwardings=yes",
    "PermitLocalCommand=no",
    "ConnectTimeout=5",
    "ServerAliveInterval=5",
    "ServerAliveCountMax=3",
];

/// The exact ssh command for one remote host (docs/ARCHITECTURE.md §10.3).
///
/// Only the hub's own key is offered (`-F none` unless the host opts into the user's ssh config, IdentitiesOnly, no
/// agent, no passwords; `NoHubKey` rather than a command ssh would complete with other keys); known_hosts is read,
/// never written (StrictHostKeyChecking=yes, UpdateHostKeys=no), so an unknown or changed host key is an error and
/// not a silent trust. `--` ends option parsing, so the destination can never be taken for an option.
pub fn ssh_argv(
    host: &HostConfig,
    key_file: &Path,
    remote_cmd: &str,
) -> Result<Vec<String>, HostsError> {
    let hid = ec_proto::pyfmt::repr_str(&host.id);
    let dest = match &host.ssh {
        Some(d) if host.id != LOCAL => d,
        _ => return Err(invalid(format!("host {hid} is reached without ssh"))),
    };
    if !SSH_DEST.is_match(dest) {
        return Err(invalid(format!(
            "host {hid}: ssh must be user@address or an ssh_config alias"
        )));
    }
    let cmd = check_remote_command(remote_cmd)?;
    let Some(key) = key_file.to_str() else {
        return Err(invalid("the hub key path is not usable"));
    };
    if key.is_empty() || key.starts_with('-') || key.contains('\0') || key.contains('\n') {
        return Err(invalid("the hub key path is not usable"));
    }
    if key.contains([' ', '\t', '"', '\'', '%']) {
        // The key is passed as `-o IdentityFile=<path>`, which ssh parses like a config line: whitespace and quotes
        // would split or quote it and `%` would be expanded as a token.
        return Err(invalid(
            "the hub key path must not contain spaces, quotes or '%'",
        ));
    }
    // Checked here, not left to ssh: when the identity file does not exist ssh only warns and then offers the user's
    // default identity files instead, even with IdentitiesOnly=yes. That would log in with the user's own
    // unrestricted key instead of the console's restricted one.
    if !Path::new(key).is_file() {
        return Err(HostsError::NoHubKey(
            "the console has no ssh key yet: run `engine-console hosts init-key --from <this machine's address>` \
             and add the line it prints to authorized_keys on each machine"
                .into(),
        ));
    }
    let mut argv: Vec<String> = ["ssh", "-T", "-e", "none"].map(String::from).to_vec();
    if !host.ssh_config {
        argv.extend(["-F".into(), "none".into()]);
    }
    for opt in SSH_OPTIONS {
        argv.extend(["-o".into(), (*opt).to_string()]);
    }
    // `-o IdentityFile=` rather than `-i`: ssh registers it without checking that the file exists, so if the key
    // vanishes between the check above and the spawn, the login fails instead of falling back to default identities.
    argv.extend([
        "-o".into(),
        format!("IdentityFile={key}"),
        "-o".into(),
        "IdentitiesOnly=yes".into(),
    ]);
    if let Some(p) = host.port.filter(|p| *p != 0) {
        argv.extend(["-p".into(), p.to_string()]);
    }
    argv.extend(["--".into(), dest.clone(), cmd]);
    Ok(argv)
}

fn passwd_entry() -> Option<nix::unistd::User> {
    nix::unistd::User::from_uid(nix::unistd::getuid())
        .ok()
        .flatten()
}

fn username_from(env: &dyn Fn(&str) -> Option<String>) -> String {
    // getpass.getuser(): LOGNAME, USER, LNAME, USERNAME, then the password database
    for k in ["LOGNAME", "USER", "LNAME", "USERNAME"] {
        if let Some(v) = env(k).filter(|v| !v.is_empty()) {
            return v;
        }
    }
    passwd_entry().map(|u| u.name).unwrap_or_default()
}

/// `os.environ.get("HOME") or str(Path.home())`
fn home_from(env: &dyn Fn(&str) -> Option<String>) -> String {
    env("HOME").filter(|h| !h.is_empty()).unwrap_or_else(|| {
        passwd_entry().map_or_else(|| "/".to_string(), |u| u.dir.to_string_lossy().into_owned())
    })
}

fn process_env(k: &str) -> Option<String> {
    std::env::var(k).ok()
}

/// The scrubbed environment ssh gets: HOME, PATH, LANG=C.UTF-8, USER. No SSH_AUTH_SOCK (so a terminal and a
/// service authenticate identically, with the hub's own key), no proxies, nothing else of the console's. PATH is the
/// hub's fixed tool path (`tools::tool_path`), never the caller's (hardening deviation from Python, which passes it).
pub fn ssh_env() -> Vec<(String, String)> {
    ssh_env_from(&process_env)
}

/// `ssh_env()` over a given environment (`env(name)`), for tests.
pub fn ssh_env_from(env: &dyn Fn(&str) -> Option<String>) -> Vec<(String, String)> {
    let mut out = vec![
        ("HOME".to_string(), home_from(env)),
        ("PATH".to_string(), crate::tools::tool_path_from(env)),
        ("LANG".to_string(), "C.UTF-8".to_string()),
    ];
    let user = env("USER")
        .filter(|u| !u.is_empty())
        .unwrap_or_else(|| username_from(env));
    if !user.is_empty() {
        out.push(("USER".into(), user));
    }
    out
}

/// Environment of the local node process: the few variables a node needs to find its home and a rootless docker.
/// The console's own environment (HF_TOKEN, engine API keys, ...) must never reach it: the node starts docker and
/// scripts, and secrets travel to docker only as the explicit `secret_env` of one request. Its PATH is the hub's
/// fixed tool path (`tools::tool_path`), not the caller's: the node finds docker, nvidia-smi and ip there.
pub fn local_env() -> Vec<(String, String)> {
    local_env_from(&process_env)
}

/// `local_env()` over a given environment, for tests.
pub fn local_env_from(env: &dyn Fn(&str) -> Option<String>) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = LOCAL_ENV_KEYS
        .iter()
        .filter_map(|k| {
            if *k == "PATH" {
                return Some(("PATH".to_string(), crate::tools::tool_path_from(env)));
            }
            env(k)
                .filter(|v| !v.is_empty())
                .map(|v| ((*k).to_string(), v))
        })
        .collect();
    let has = |out: &Vec<(String, String)>, k: &str| out.iter().any(|(kk, _)| kk == k);
    if !has(&out, "HOME") {
        out.push(("HOME".into(), home_from(env)));
    }
    if !has(&out, "LANG") {
        out.push(("LANG".into(), "C.UTF-8".into()));
    }
    if !has(&out, "USER") {
        let user = username_from(env);
        if !user.is_empty() {
            out.push(("USER".into(), user));
        }
    }
    out
}

/// argv of the local node: `node_command` (when set and not empty) replaces the whole thing; else the sibling
/// `ecnode` binary (PORTING.md D8) with `--stdio` and `--policy <node_policy>` when set. Only the default command
/// takes `--policy`, because an override is the complete command line by definition.
pub fn local_argv(cfg: &Settings) -> Vec<String> {
    if let Some(cmd) = cfg.node_command.as_ref().filter(|c| !c.is_empty()) {
        return cmd.clone();
    }
    let mut argv = vec![
        sibling_ecnode().to_string_lossy().into_owned(),
        "--stdio".to_string(),
    ];
    if let Some(p) = &cfg.node_policy {
        argv.push("--policy".into());
        argv.push(p.to_string_lossy().into_owned());
    }
    argv
}

/// `<dir of the running executable>/ecnode`.
pub fn sibling_ecnode() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join("ecnode")))
        .unwrap_or_else(|| PathBuf::from("ecnode"))
}

// ── the hub's ssh key ───────────────────────────────────────────────────────────────────────────────────
/// The one line to add to the login user's authorized_keys on each managed machine. `restrict` removes every
/// capability (shell, pty, forwarding, agent), `command` pins the key to the node, `from` pins it to the hub's
/// address: a stolen key can start a node, from the hub, and nothing else. The public key's own comment is dropped,
/// so nothing in it can add options or lines.
pub fn key_line(pubkey: &str, from: Option<&str>, remote_cmd: &str) -> Result<String, HostsError> {
    let text = py_strip(pubkey);
    let parts = py_split(text);
    if text.contains('\n')
        || text.contains('\r')
        || parts.len() < 2
        || !PUBKEY_TYPE.is_match(parts[0])
        || !PUBKEY_B64.is_match(parts[1])
    {
        return Err(invalid("that is not a single OpenSSH public key line"));
    }
    let cmd = check_remote_command(remote_cmd)?;
    let mut opts = vec!["restrict".to_string()];
    if let Some(f) = from {
        if !FROM.is_match(f) {
            return Err(invalid(
                "--from must be addresses or patterns like 10.0.0.5 or 10.0.0.0/24 (comma-separated)",
            ));
        }
        opts.push(format!("from=\"{f}\""));
    }
    opts.push(format!("command=\"{cmd}\""));
    Ok(format!(
        "{} {} {} {KEY_COMMENT}",
        opts.join(","),
        parts[0],
        parts[1]
    ))
}

/// Create the hub's ed25519 key unless one exists, and return that line (never replaces an existing key).
pub fn init_key(
    key_file: &Path,
    from: Option<&str>,
    remote_cmd: &str,
) -> Result<String, HostsError> {
    init_key_with(key_file, from, remote_cmd, "ssh-keygen", 30.0)
}

/// Restores the process umask when dropped (also on an early return).
struct UmaskGuard(nix::sys::stat::Mode);

impl Drop for UmaskGuard {
    fn drop(&mut self) {
        nix::sys::stat::umask(self.0);
    }
}

/// `init_key` with Python's keyword arguments `keygen` and `timeout`.
///
/// The key has no passphrase: the console uses it unattended; what limits it is the authorized_keys line. An
/// existing key is NEVER replaced: its public half is read (or re-derived) and the same line is returned. Missing
/// directories are created 0700 and the key 0600 (umask 077 while ssh-keygen runs). The key is never printed or
/// returned; only its public half is.
pub fn init_key_with(
    key_file: &Path,
    from: Option<&str>,
    remote_cmd: &str,
    keygen: &str,
    timeout_s: f64,
) -> Result<String, HostsError> {
    let key = key_file.display().to_string();
    let pub_path = PathBuf::from(format!("{key}.pub"));
    // validate the arguments before touching disk
    key_line("ssh-ed25519 AAAAAAAAAAAAAAAAAAAAAA", from, remote_cmd)?;
    if key_file.symlink_metadata().is_ok() {
        if !key_file.exists() {
            return Err(invalid(format!(
                "{key} is a dangling symlink: remove it first"
            )));
        }
    } else {
        if pub_path.symlink_metadata().is_ok() {
            return Err(invalid(format!(
                "{} exists without its private key: move it away so it is not overwritten",
                pub_path.display()
            )));
        }
        {
            let _restore = UmaskGuard(nix::sys::stat::umask(
                nix::sys::stat::Mode::from_bits_truncate(0o077),
            ));
            let dir = key_file
                .parent()
                .filter(|d| !d.as_os_str().is_empty())
                .unwrap_or_else(|| Path::new("."));
            std::fs::DirBuilder::new()
                .recursive(true)
                .mode(0o700)
                .create(dir)
                .map_err(|e| {
                    invalid(format!("cannot create {}: {}", dir.display(), strerror(&e)))
                })?;
            run_keygen(
                &[
                    keygen,
                    "-t",
                    "ed25519",
                    "-N",
                    "",
                    "-C",
                    KEY_COMMENT,
                    "-q",
                    "-f",
                    &key,
                ],
                timeout_s,
            )?;
        }
        let mode = std::fs::metadata(key_file)
            .map(|m| m.mode())
            .map_err(|e| invalid(format!("cannot read {key}: {}", strerror(&e))))?;
        if mode & 0o077 != 0 {
            return Err(invalid(format!(
                "{key} was created readable by others; delete it and check the umask"
            )));
        }
    }
    // followed like Python's `read_text`, but never a blocking open (a FIFO) and never a special file
    let text = match crate::fsutil::open_regular(&pub_path, crate::fsutil::Follow::Yes) {
        Ok(f) => {
            let mut buf = Vec::new();
            f.take(8192).read_to_end(&mut buf).map_err(|e| {
                invalid(format!(
                    "cannot read {}: {}",
                    pub_path.display(),
                    strerror(&e)
                ))
            })?;
            String::from_utf8_lossy(&buf).into_owned()
        }
        // the public half can always be derived again
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            run_keygen(&[keygen, "-y", "-f", &key], timeout_s)?
        }
        Err(e) => {
            return Err(invalid(format!(
                "cannot read {}: {}",
                pub_path.display(),
                strerror(&e)
            )));
        }
    };
    key_line(&text, from, remote_cmd)
}

/// The most of ssh-keygen's stdout and of its stderr kept (it prints one key line or one error line).
pub const KEYGEN_OUTPUT_CAP: usize = 64 * 1024;

/// Run ssh-keygen (argv only, scrubbed environment, no stdin) and return its stdout. The program is found on the
/// hub's tool path, never the caller's PATH; its output is capped while it is read (`tools::run_capped`).
fn run_keygen(argv: &[&str], timeout_s: f64) -> Result<String, HostsError> {
    let env = ssh_env();
    let search = env
        .iter()
        .find(|(k, _)| k == "PATH")
        .map_or_else(crate::tools::tool_path, |(_, v)| v.clone());
    let Some(prog) = crate::tools::find_in(argv[0], &search) else {
        return Err(invalid(
            "ssh-keygen was not found: install the OpenSSH client",
        ));
    };
    let timeout = Duration::from_secs_f64(timeout_s.max(0.0));
    let out = match crate::tools::run_capped(&prog, &argv[1..], &env, timeout, KEYGEN_OUTPUT_CAP) {
        Ok(o) => o,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(invalid(
                "ssh-keygen was not found: install the OpenSSH client",
            ));
        }
        Err(e) => {
            return Err(invalid(format!(
                "could not run ssh-keygen: {}",
                strerror(&e)
            )));
        }
    };
    let Some(status) = out.status else {
        return Err(invalid("ssh-keygen did not finish in time"));
    };
    if !status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        let short: String = py_strip(&stderr).chars().take(200).collect();
        let why = if short.is_empty() {
            match status.code() {
                Some(c) => format!("exit {c}"),
                None => format!(
                    "exit -{}",
                    std::os::unix::process::ExitStatusExt::signal(&status).unwrap_or(0)
                ),
            }
        } else {
            short
        };
        return Err(invalid(format!("ssh-keygen failed: {why}")));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_py_predicates() {
        assert!(py_isprintable('a') && py_isprintable(' ') && py_isprintable('é'));
        assert!(!py_isprintable('\x07') && !py_isprintable('\n') && !py_isprintable('\u{a0}'));
        assert!(!py_isprintable('\u{200b}') && !py_isprintable('\u{e000}'));
        assert_eq!(py_split(" a\x1cb  c\n"), ["a", "b", "c"]);
        assert_eq!(py_strip("\x1f x \t"), "x");
    }

    #[test]
    fn test_first_non_finite() {
        let t = |s: &str| first_non_finite(s).map(|t| t.1);
        assert_eq!(t(r#"{"a": "NaN", "b": NaN}"#), Some("NaN".into()));
        assert_eq!(t(r#"{"p": 1e999}"#), Some("1e999".into()));
        assert_eq!(t(r#"{"p": -Infinity}"#), Some("-Infinity".into()));
        assert_eq!(t(r#"{"p": 99999999999999999999999}"#), None);
        assert_eq!(t(r#"{"p": "a\"NaN"}"#), None);
    }
}
