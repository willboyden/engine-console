//! Process configuration (port of config.py).
//!
//! Environment variables, with an optional per-user env file (dotenv syntax) so settings survive restarts. Real
//! environment variables win over the file. Names are case-insensitive and carry no prefix (HF_TOKEN, EGRESS_PROXY,
//! ...), except the generic ones in [`PREFIXED`], which are read ONLY as `ENGINE_CONSOLE_<NAME>` (review finding L5).
//! Value parsing follows pydantic-settings: list fields are JSON, booleans accept 1/0/true/false/yes/no/on/off/t/f/
//! y/n, numbers are trimmed. A value that does not parse is a startup error naming the field (never echoing it).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde_json::Value;

pub const GATEWAY_IMAGE: &str =
    "nginx@sha256:62ff2089abf5a9ed33bd232895bef5e22f7bb4b200675cec49a5ebc48e3d4ac8";
pub const PREFIX: &str = "ENGINE_CONSOLE_";
pub const PREFIXED: &[&str] = &[
    "host",
    "port",
    "allowed_hosts",
    "trust_loopback",
    "data_dir",
];
pub const DEFAULT_REMOTE_COMMAND: &str = "/usr/local/lib/ecnode/ecnode --stdio";

/// A setting that cannot be used. `field` is the setting's name; the message never contains its value.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{field}: {message}")]
pub struct SettingsError {
    pub field: String,
    pub message: String,
}

fn err(field: &str, message: impl Into<String>) -> SettingsError {
    SettingsError {
        field: field.to_string(),
        message: message.into(),
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Settings {
    pub host: String,
    pub port: u16,
    pub data_dir: PathBuf,
    pub hf_cache_dir: PathBuf,
    pub hf_cache_container_path: String,
    pub hf_endpoint: String,
    /// ADR 7: never logged or returned
    pub hf_token: Option<String>,
    pub hf_token_file: PathBuf,
    pub hf_allowed_hosts: Vec<String>,
    pub docker_bin: String,
    pub docker_context: String,
    pub engine_network: String,
    pub gateway_image: String,
    pub egress_proxy: Option<String>,
    pub egress_ca_bundle: Option<PathBuf>,
    pub require_egress_proxy: bool,
    pub discovery_enabled: bool,
    pub discovery_interval_s: f64,
    pub discovery_ports: Vec<i64>,
    pub port_range_start: i64,
    pub port_range_end: i64,
    pub engine_image_allowlist: Vec<String>,
    pub gateway_image_allowlist: Vec<String>,
    pub enable_api_docs: bool,
    pub allowed_hosts: Vec<String>,
    pub body_cap_bytes: u64,
    pub chat_body_cap_bytes: u64,
    pub import_body_cap_bytes: u64,
    pub max_sse_connections: i64,
    pub max_sse_per_key: i64,
    pub max_chat_streams: i64,
    pub max_chat_streams_per_key: i64,
    pub sse_idle_timeout_s: f64,
    pub max_download_queue: i64,
    pub startup_timeout_s: i64,
    pub engine_hf_offline: bool,
    pub otlp_endpoint: String,
    pub otlp_enabled: bool,
    pub frontend_dir: Option<PathBuf>,
    pub background_tasks: bool,
    pub scrape_interval_s: f64,
    pub supervisor_interval_s: f64,
    pub download_concurrency: i64,
    pub trust_loopback: bool,
    pub hosts_file: PathBuf,
    pub hub_key_file: PathBuf,
    pub node_policy: Option<PathBuf>,
    pub node_command: Option<Vec<String>>,
    pub remote_node_command: String,
    pub fleet_enabled: bool,
    pub restart_settle_s: f64,
}

/// The user's home (`Path.home()`: $HOME, else the password database).
pub fn home() -> PathBuf {
    if let Some(h) = std::env::var_os("HOME").filter(|h| !h.is_empty()) {
        return PathBuf::from(h);
    }
    nix::unistd::User::from_uid(nix::unistd::getuid())
        .ok()
        .flatten()
        .map_or_else(|| PathBuf::from("/"), |u| u.dir)
}

/// `ENV_FILE`: $ENGINE_CONSOLE_ENV_FILE (when set and non-empty), else ~/.config/engine-console/env.
pub fn env_file() -> PathBuf {
    match std::env::var("ENGINE_CONSOLE_ENV_FILE") {
        Ok(v) if !v.is_empty() => PathBuf::from(v),
        _ => home().join(".config").join("engine-console").join("env"),
    }
}

/// ~/.cache/huggingface, or its target when it is a symlink (the validator refuses symlinked cache dirs).
fn default_cache_dir(home: &Path) -> PathBuf {
    let p = home.join(".cache").join("huggingface");
    if p.is_symlink() {
        return std::fs::canonicalize(&p).unwrap_or(p);
    }
    p
}

impl Settings {
    /// All defaults (as `Settings()` with an empty environment and no env file).
    pub fn defaults() -> Self {
        let h = home();
        let cfg = h.join(".config").join("engine-console");
        Self {
            host: "127.0.0.1".into(),
            port: 8791,
            data_dir: h.join(".local").join("share").join("engine-console"),
            hf_cache_dir: default_cache_dir(&h),
            hf_cache_container_path: "/root/.cache/huggingface".into(),
            hf_endpoint: "https://huggingface.co".into(),
            hf_token: None,
            hf_token_file: cfg.join("hf_token"),
            hf_allowed_hosts: [
                "huggingface.co",
                "cdn-lfs*.huggingface.co",
                "cas-bridge.xethub.hf.co",
                "*.hf.co",
            ]
            .map(String::from)
            .to_vec(),
            docker_bin: "docker".into(),
            docker_context: "rootless".into(),
            engine_network: "engine-console-engines".into(),
            gateway_image: GATEWAY_IMAGE.into(),
            egress_proxy: None,
            egress_ca_bundle: None,
            require_egress_proxy: false,
            discovery_enabled: true,
            discovery_interval_s: 10.0,
            discovery_ports: [8000, 30000, 11434, 8001]
                .into_iter()
                .chain(18000..18100)
                .collect(),
            port_range_start: 18000,
            port_range_end: 18099,
            engine_image_allowlist: Vec::new(),
            gateway_image_allowlist: vec![GATEWAY_IMAGE.into()],
            enable_api_docs: false,
            allowed_hosts: Vec::new(),
            body_cap_bytes: 1024 * 1024,
            chat_body_cap_bytes: 32 * 1024 * 1024,
            import_body_cap_bytes: 256 * 1024,
            max_sse_connections: 8,
            max_sse_per_key: 4,
            max_chat_streams: 16,
            max_chat_streams_per_key: 4,
            sse_idle_timeout_s: 300.0,
            max_download_queue: 50,
            startup_timeout_s: 1800,
            engine_hf_offline: true,
            otlp_endpoint: "localhost:4317".into(),
            otlp_enabled: true,
            frontend_dir: None,
            background_tasks: true,
            scrape_interval_s: 2.0,
            supervisor_interval_s: 2.0,
            download_concurrency: 2,
            trust_loopback: false,
            hosts_file: cfg.join("hosts.json"),
            hub_key_file: cfg.join("ssh").join("id_ed25519"),
            node_policy: None,
            node_command: None,
            remote_node_command: DEFAULT_REMOTE_COMMAND.into(),
            fleet_enabled: true,
            restart_settle_s: 4.0,
        }
    }

    /// `Settings()`: the process environment over the env file over the defaults.
    pub fn from_env() -> Result<Self, SettingsError> {
        let environ: Vec<(String, String)> = std::env::vars_os()
            .filter_map(|(k, v)| Some((k.into_string().ok()?, v.into_string().ok()?)))
            .collect();
        Self::load(&environ, Some(&env_file()))
    }

    /// The same, from an explicit environment and env file (tests).
    pub fn load(
        environ: &[(String, String)],
        env_file: Option<&Path>,
    ) -> Result<Self, SettingsError> {
        let file_vars = env_file
            .map(|p| read_env_file(p, environ))
            .unwrap_or_default();
        // env-file first, then the environment over it (env wins); names lower-cased, prefix rule applied to each
        let mut vars: HashMap<String, String> = HashMap::new();
        for (k, v) in prefix_only(&file_vars) {
            vars.insert(k, v);
        }
        for (k, v) in prefix_only(environ) {
            vars.insert(k, v);
        }
        let mut s = Self::defaults();
        s.apply(&vars)?;
        s.validate()?;
        Ok(s)
    }

    fn apply(&mut self, vars: &HashMap<String, String>) -> Result<(), SettingsError> {
        let get = |name: &str| vars.get(name).map(String::as_str);
        macro_rules! set {
            ($field:ident, $parse:expr) => {
                if let Some(v) = get(stringify!($field)) {
                    self.$field = $parse(stringify!($field), v)?;
                }
            };
        }
        set!(host, p_str);
        set!(port, p_port);
        set!(data_dir, p_path);
        set!(hf_cache_dir, p_path);
        set!(hf_cache_container_path, p_str);
        set!(hf_endpoint, p_str);
        set!(hf_token, p_opt_str);
        set!(hf_token_file, p_path);
        set!(hf_allowed_hosts, p_str_list);
        set!(docker_bin, p_str);
        set!(docker_context, p_str);
        set!(engine_network, p_str);
        set!(gateway_image, p_str);
        set!(egress_proxy, p_opt_str);
        set!(egress_ca_bundle, p_opt_path);
        set!(require_egress_proxy, p_bool);
        set!(discovery_enabled, p_bool);
        set!(discovery_interval_s, p_float);
        set!(discovery_ports, p_int_list);
        set!(port_range_start, p_int);
        set!(port_range_end, p_int);
        set!(engine_image_allowlist, p_str_list);
        set!(gateway_image_allowlist, p_str_list);
        set!(enable_api_docs, p_bool);
        set!(allowed_hosts, p_str_list);
        set!(body_cap_bytes, p_u64);
        set!(chat_body_cap_bytes, p_u64);
        set!(import_body_cap_bytes, p_u64);
        set!(max_sse_connections, p_int);
        set!(max_sse_per_key, p_int);
        set!(max_chat_streams, p_int);
        set!(max_chat_streams_per_key, p_int);
        set!(sse_idle_timeout_s, p_float);
        set!(max_download_queue, p_int);
        set!(startup_timeout_s, p_int);
        set!(engine_hf_offline, p_bool);
        set!(otlp_endpoint, p_str);
        set!(otlp_enabled, p_bool);
        set!(frontend_dir, p_opt_path);
        set!(background_tasks, p_bool);
        set!(scrape_interval_s, p_float);
        set!(supervisor_interval_s, p_float);
        set!(download_concurrency, p_int);
        set!(trust_loopback, p_bool);
        set!(hosts_file, p_path);
        set!(hub_key_file, p_path);
        set!(node_policy, p_opt_path);
        set!(node_command, p_opt_str_list);
        set!(remote_node_command, p_str);
        set!(fleet_enabled, p_bool);
        set!(restart_settle_s, p_float);
        Ok(())
    }

    /// The field validators of config.py.
    pub fn validate(&self) -> Result<(), SettingsError> {
        if self.host.trim().is_empty() {
            return Err(err(
                "host",
                "ENGINE_CONSOLE_HOST must not be empty: an empty address listens on every interface",
            ));
        }
        let v = &self.hf_cache_dir;
        let s = v.to_string_lossy();
        if !v.is_absolute() {
            return Err(err("hf_cache_dir", "HF_CACHE_DIR must be an absolute path"));
        }
        if s.contains([':', ',', '\n', '\r', '\0']) {
            return Err(err(
                "hf_cache_dir",
                "HF_CACHE_DIR must not contain ':' ',' or newlines (docker -v syntax injection)",
            ));
        }
        if v.is_symlink() {
            return Err(err("hf_cache_dir", "HF_CACHE_DIR must not be a symlink"));
        }
        Ok(())
    }

    pub fn db_path(&self) -> PathBuf {
        self.data_dir.join("console.db")
    }

    pub fn hub_dir(&self) -> PathBuf {
        self.hf_cache_dir.join("hub")
    }
}

/// A source's variables (lower-case names) with each bare PREFIXED name dropped and ENGINE_CONSOLE_<NAME> moved to
/// its place, so only the prefixed value is found.
fn prefix_only(env: &[(String, String)]) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    let mut prefixed: Vec<(String, String)> = Vec::new();
    let pl = PREFIX.to_ascii_lowercase();
    for (k, v) in env {
        let lk = k.to_ascii_lowercase();
        if PREFIXED.contains(&lk.as_str()) {
            continue;
        }
        if let Some(name) = lk.strip_prefix(&pl)
            && PREFIXED.contains(&name)
        {
            prefixed.push((name.to_string(), v.clone()));
            continue;
        }
        out.push((lk, v.clone()));
    }
    out.extend(prefixed);
    out
}

/// The bare PREFIXED names (HOST, PORT, ...) set in `environ` or in the env file, which are therefore ignored.
/// Names only: their values are never returned.
pub fn legacy_env_names(environ: &[(String, String)], env_file: Option<&Path>) -> Vec<String> {
    let mut found: Vec<String> = environ
        .iter()
        .map(|(k, _)| k.to_ascii_lowercase())
        .collect();
    if let Some(p) = env_file {
        found.extend(
            read_env_file(p, environ)
                .into_iter()
                .map(|(k, _)| k.to_ascii_lowercase()),
        );
    }
    PREFIXED
        .iter()
        .filter(|n| found.iter().any(|f| f == *n))
        .map(|n| n.to_ascii_uppercase())
        .collect()
}

/// main.py `warn_legacy_env`: the one start-up warning when an old unprefixed name is still set (it is ignored now, so
/// the console may not be listening where the operator expects), or None when nothing is stale. Names only, never
/// values.
pub fn legacy_env_warning(environ: &[(String, String)], env_file: Option<&Path>) -> Option<String> {
    let old = legacy_env_names(environ, env_file);
    if old.is_empty() {
        return None;
    }
    let new: Vec<String> = old.iter().map(|n| format!("{PREFIX}{n}")).collect();
    let file = env_file.map_or_else(|| "None".to_string(), |p| p.display().to_string());
    Some(format!(
        "ignoring {} (set in the environment or {file}): these settings are read only with the {PREFIX} prefix now; use {}",
        old.join(", "),
        new.join(", ")
    ))
}

// ── value parsing (pydantic-settings) ──────────────────────────────────────────────────────────────────────
fn p_str(_f: &str, v: &str) -> Result<String, SettingsError> {
    Ok(v.to_string())
}

fn p_opt_str(_f: &str, v: &str) -> Result<Option<String>, SettingsError> {
    Ok(Some(v.to_string()))
}

fn p_path(_f: &str, v: &str) -> Result<PathBuf, SettingsError> {
    Ok(py_path(v))
}

fn p_opt_path(_f: &str, v: &str) -> Result<Option<PathBuf>, SettingsError> {
    Ok(Some(py_path(v)))
}

/// `pathlib.Path(v)`: "" is ".", trailing slashes and "//" are folded.
fn py_path(v: &str) -> PathBuf {
    if v.is_empty() {
        return PathBuf::from(".");
    }
    PathBuf::from(v).components().collect()
}

fn p_bool(f: &str, v: &str) -> Result<bool, SettingsError> {
    match v.trim().to_ascii_lowercase().as_str() {
        "1" | "on" | "t" | "true" | "y" | "yes" => Ok(true),
        "0" | "off" | "f" | "false" | "n" | "no" => Ok(false),
        _ => Err(err(
            f,
            "Input should be a valid boolean, unable to interpret input",
        )),
    }
}

fn p_int(f: &str, v: &str) -> Result<i64, SettingsError> {
    let t = v.trim().replace('_', "");
    if let Ok(i) = t.parse::<i64>() {
        return Ok(i);
    }
    match t.parse::<f64>() {
        Ok(x) if x.is_finite() && x.fract() == 0.0 => Ok(float_to_i64(x)),
        Ok(_) => Err(err(
            f,
            "Input should be a valid integer, got a number with a fractional part",
        )),
        Err(_) => Err(err(
            f,
            "Input should be a valid integer, unable to parse string as an integer",
        )),
    }
}

#[allow(clippy::cast_possible_truncation)]
fn float_to_i64(x: f64) -> i64 {
    x as i64
}

fn p_port(f: &str, v: &str) -> Result<u16, SettingsError> {
    let i = p_int(f, v)?;
    // pydantic accepts any int here; uvicorn then fails to bind outside 0-65535. Refused up front instead.
    u16::try_from(i).map_err(|_| err(f, "port must be 0-65535"))
}

fn p_u64(f: &str, v: &str) -> Result<u64, SettingsError> {
    let i = p_int(f, v)?;
    Ok(u64::try_from(i).unwrap_or(0))
}

fn p_float(f: &str, v: &str) -> Result<f64, SettingsError> {
    let t = v.trim().to_ascii_lowercase();
    let x = match t.as_str() {
        "inf" | "+inf" | "infinity" | "+infinity" => f64::INFINITY,
        "-inf" | "-infinity" => f64::NEG_INFINITY,
        "nan" | "+nan" | "-nan" => f64::NAN,
        _ => t.replace('_', "").parse::<f64>().map_err(|_| {
            err(
                f,
                "Input should be a valid number, unable to parse string as a number",
            )
        })?,
    };
    Ok(x)
}

fn json_of(f: &str, v: &str) -> Result<Value, SettingsError> {
    serde_json::from_str(v).map_err(|_| {
        err(
            f,
            format!("error parsing value for field \"{f}\" from source"),
        )
    })
}

fn p_str_list(f: &str, v: &str) -> Result<Vec<String>, SettingsError> {
    match json_of(f, v)? {
        Value::Array(a) => a
            .into_iter()
            .map(|x| match x {
                Value::String(s) => Ok(s),
                _ => Err(err(f, "Input should be a valid string")),
            })
            .collect(),
        _ => Err(err(f, "Input should be a valid list")),
    }
}

fn p_opt_str_list(f: &str, v: &str) -> Result<Option<Vec<String>>, SettingsError> {
    match json_of(f, v)? {
        Value::Null => Ok(None),
        other => p_str_list(f, &other.to_string()).map(Some),
    }
}

fn p_int_list(f: &str, v: &str) -> Result<Vec<i64>, SettingsError> {
    match json_of(f, v)? {
        Value::Array(a) => a
            .into_iter()
            .map(|x| match &x {
                Value::Number(n) if n.is_i64() => Ok(n.as_i64().unwrap_or(0)),
                Value::Number(n) if n.as_f64().is_some_and(|y| y.fract() == 0.0) => {
                    Ok(float_to_i64(n.as_f64().unwrap_or(0.0)))
                }
                Value::String(s) => p_int(f, s),
                Value::Bool(b) => Ok(i64::from(*b)),
                _ => Err(err(f, "Input should be a valid integer")),
            })
            .collect(),
        _ => Err(err(f, "Input should be a valid list")),
    }
}

// ── the env file (python-dotenv's dotenv_values, the subset pydantic-settings relies on) ─────────────────────
/// KEY=VALUE lines; `export ` prefix; `#` comments (whole-line, and after an unquoted value when preceded by a
/// space); single quotes literal; double quotes with \n \t \" \\ escapes; `${VAR}` / `${VAR:-default}` expansion
/// in unquoted and double-quoted values (from earlier keys of the file, then the environment). A missing or
/// unreadable file is empty. Key order kept. Hardening: the file is opened without blocking and must be a regular
/// file (a FIFO there would hang start-up) of at most `MAX_ENV_FILE` bytes; otherwise it counts as unreadable.
pub fn read_env_file(path: &Path, environ: &[(String, String)]) -> Vec<(String, String)> {
    let Ok(text) = crate::fsutil::read_text(path, crate::fsutil::Follow::Yes, MAX_ENV_FILE) else {
        return Vec::new();
    };
    parse_dotenv(&text, environ)
}

/// The largest env file read (a few dozen settings; JSON lists included, far below this).
pub const MAX_ENV_FILE: usize = 1024 * 1024;

pub fn parse_dotenv(text: &str, environ: &[(String, String)]) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    for raw in text.lines() {
        let line = raw.trim_start();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let line = line.strip_prefix("export ").map_or(line, str::trim_start);
        let Some((k, rest)) = line.split_once('=') else {
            continue; // a bare key has no value: ignored by pydantic-settings
        };
        let key = k.trim().to_string();
        if key.is_empty() {
            continue;
        }
        let rest = rest.trim_start();
        let (value, expand) = if let Some(body) = rest.strip_prefix('\'') {
            (
                body.split_once('\'').map_or(body, |(v, _)| v).to_string(),
                false,
            )
        } else if let Some(body) = rest.strip_prefix('"') {
            (unescape_double(body), true)
        } else {
            let v = match rest.find(" #") {
                Some(i) => &rest[..i],
                None => rest,
            };
            (v.trim_end().to_string(), true)
        };
        let value = if expand {
            expand_vars(&value, &out, environ)
        } else {
            value
        };
        if let Some(slot) = out.iter_mut().find(|(kk, _)| *kk == key) {
            slot.1 = value;
        } else {
            out.push((key, value));
        }
    }
    out
}

fn unescape_double(body: &str) -> String {
    let mut out = String::new();
    let mut chars = body.chars();
    while let Some(c) = chars.next() {
        match c {
            '"' => break,
            '\\' => match chars.next() {
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some('r') => out.push('\r'),
                Some('"') => out.push('"'),
                Some('\'') => out.push('\''),
                Some('\\') | None => out.push('\\'),
                Some(o) => {
                    out.push('\\');
                    out.push(o);
                }
            },
            c => out.push(c),
        }
    }
    out
}

fn expand_vars(v: &str, file: &[(String, String)], environ: &[(String, String)]) -> String {
    let mut out = String::new();
    let mut rest = v;
    while let Some(i) = rest.find("${") {
        out.push_str(&rest[..i]);
        let after = &rest[i + 2..];
        let Some(j) = after.find('}') else {
            out.push_str(&rest[i..]);
            return out;
        };
        let inner = &after[..j];
        let (name, default) = inner.split_once(":-").map_or((inner, ""), |(n, d)| (n, d));
        let found = environ
            .iter()
            .find(|(k, _)| k == name)
            .or_else(|| file.iter().find(|(k, _)| k == name))
            .map(|(_, v)| v.as_str());
        out.push_str(found.unwrap_or(default));
        rest = &after[j + 1..];
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect()
    }

    // Port of tests/core/test_env_prefix.py (names mirror the Python tests).
    const BARE: &[(&str, &str)] = &[
        ("HOST", "0.0.0.0"),
        ("PORT", "9"),
        ("ALLOWED_HOSTS", "[\"evil.example\"]"),
        ("TRUST_LOOPBACK", "true"),
        ("DATA_DIR", "/tmp/somewhere-else"),
    ];

    fn defaults(s: &Settings) -> bool {
        let d = Settings::defaults();
        (
            s.host.as_str(),
            s.port,
            s.allowed_hosts.is_empty(),
            s.trust_loopback,
            &s.data_dir,
        ) == ("127.0.0.1", 8791, true, false, &d.data_dir)
    }

    #[test]
    fn test_bare_generic_names_in_the_environment_are_ignored() {
        assert!(defaults(&Settings::load(&env(BARE), None).unwrap()));
    }

    #[test]
    fn test_bare_generic_names_in_the_env_file_are_ignored() {
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("env");
        let text: String = BARE.iter().map(|(k, v)| format!("{k}={v}\n")).collect();
        std::fs::write(&f, text).unwrap();
        assert!(defaults(&Settings::load(&env(&[]), Some(&f)).unwrap()));
    }

    #[test]
    fn test_prefixed_names_work_from_the_env_file_and_the_environment_wins() {
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("env");
        let data = dir.path().join("data");
        std::fs::write(
            &f,
            format!(
                "ENGINE_CONSOLE_HOST=10.0.0.1\nENGINE_CONSOLE_PORT=9000\nENGINE_CONSOLE_ALLOWED_HOSTS=[\"console.example\"]\n\
                 ENGINE_CONSOLE_TRUST_LOOPBACK=true\nENGINE_CONSOLE_DATA_DIR={}\nHOST=0.0.0.0\n",
                data.display()
            ),
        )
        .unwrap();
        let s = Settings::load(&env(&[]), Some(&f)).unwrap();
        assert_eq!(
            (
                s.host.as_str(),
                s.port,
                s.allowed_hosts.clone(),
                s.trust_loopback,
                &s.data_dir
            ),
            (
                "10.0.0.1",
                9000,
                vec!["console.example".to_string()],
                true,
                &data
            )
        );
        // bare in the environment: still ignored
        let s = Settings::load(
            &env(&[("ENGINE_CONSOLE_PORT", "9100"), ("HOST", "0.0.0.0")]),
            Some(&f),
        )
        .unwrap();
        assert_eq!((s.port, s.host.as_str()), (9100, "10.0.0.1"));
    }

    #[test]
    fn test_every_other_setting_keeps_its_documented_name() {
        let s = Settings::load(
            &env(&[
                ("EGRESS_PROXY", "http://127.0.0.1:8082"),
                ("MAX_SSE_PER_KEY", "3"),
                ("MAX_CHAT_STREAMS", "5"),
            ]),
            None,
        )
        .unwrap();
        assert_eq!(
            (
                s.egress_proxy.as_deref(),
                s.max_sse_per_key,
                s.max_chat_streams
            ),
            (Some("http://127.0.0.1:8082"), 3, 5)
        );
        // only the five generic names take the prefix
        let s = Settings::load(
            &env(&[("ENGINE_CONSOLE_EGRESS_PROXY", "http://127.0.0.1:9")]),
            None,
        )
        .unwrap();
        assert_eq!(s.egress_proxy, None);
    }

    #[test]
    fn test_an_empty_host_is_refused_rather_than_listening_everywhere() {
        let e = Settings::load(&env(&[("ENGINE_CONSOLE_HOST", "")]), None).unwrap_err();
        assert_eq!(e.field, "host");
        assert!(
            e.message.contains("ENGINE_CONSOLE_HOST must not be empty"),
            "{e}"
        );
    }

    #[test]
    fn test_a_stale_name_gives_one_warning_with_the_new_names_and_no_values() {
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("env");
        std::fs::write(
            &f,
            "PORT=4242\nDATA_DIR=/srv/private-place\nHF_ENDPOINT=https://huggingface.co\n",
        )
        .unwrap();
        let environ = env(&[
            ("HOST", "203.0.113.9"),
            ("PATH", "/usr/bin"),
            ("ENGINE_CONSOLE_PORT", "8791"),
        ]);
        assert_eq!(
            legacy_env_names(&environ, Some(&f)),
            vec!["HOST", "PORT", "DATA_DIR"]
        );
        assert!(
            legacy_env_names(
                &env(&[("PATH", "/usr/bin")]),
                Some(&dir.path().join("missing"))
            )
            .is_empty()
        );
        assert_eq!(
            legacy_env_warning(&env(&[("PATH", "/usr/bin")]), None),
            None
        ); // nothing stale: no warning
        let msg = legacy_env_warning(&environ, Some(&f)).unwrap();
        for name in [
            "ENGINE_CONSOLE_HOST",
            "ENGINE_CONSOLE_PORT",
            "ENGINE_CONSOLE_DATA_DIR",
        ] {
            assert!(msg.contains(name), "{msg}");
        }
        for value in ["203.0.113.9", "4242", "private-place"] {
            assert!(!msg.contains(value), "{msg}");
        }
    }

    #[test]
    fn test_a_stray_exported_host_no_longer_widens_the_bind() {
        // The finding end to end at the settings layer (the server itself is never started on the console's port):
        // `engine-console` started with HOST=0.0.0.0 and PORT=80 in its environment still binds 127.0.0.1:8791, and the
        // start-up warning names ENGINE_CONSOLE_HOST.
        let dir = tempfile::tempdir().unwrap();
        let environ = env(&[("HOST", "0.0.0.0"), ("PORT", "80")]);
        let no_file = dir.path().join("no-env-file");
        let s = Settings::load(&environ, Some(&no_file)).unwrap();
        assert_eq!((s.host.as_str(), s.port), ("127.0.0.1", 8791));
        assert!(
            legacy_env_warning(&environ, Some(&no_file))
                .unwrap()
                .contains("ENGINE_CONSOLE_HOST")
        );
    }

    #[test]
    fn test_validators() {
        assert_eq!(
            Settings::load(&env(&[("ENGINE_CONSOLE_HOST", "  ")]), None)
                .unwrap_err()
                .field,
            "host"
        );
        assert_eq!(
            Settings::load(&env(&[("HF_CACHE_DIR", "rel/x")]), None)
                .unwrap_err()
                .field,
            "hf_cache_dir"
        );
        assert_eq!(
            Settings::load(&env(&[("HF_CACHE_DIR", "/a:b")]), None)
                .unwrap_err()
                .field,
            "hf_cache_dir"
        );
    }

    #[test]
    fn test_values_parse_like_pydantic_settings() {
        let s = Settings::load(
            &env(&[
                ("otlp_enabled", "no"),
                ("FLEET_ENABLED", "0"),
                ("NODE_COMMAND", "[\"/bin/ecnode\", \"--stdio\"]"),
                ("DISCOVERY_PORTS", "[1, 2]"),
                ("SCRAPE_INTERVAL_S", " 0.5 "),
            ]),
            None,
        )
        .unwrap();
        assert!(!s.otlp_enabled && !s.fleet_enabled);
        assert_eq!(
            s.node_command,
            Some(vec!["/bin/ecnode".to_string(), "--stdio".to_string()])
        );
        assert_eq!(s.discovery_ports, vec![1, 2]);
        assert!((s.scrape_interval_s - 0.5).abs() < f64::EPSILON);
        assert!(Settings::load(&env(&[("PORT_RANGE_START", "x")]), None).is_err());
        assert!(Settings::load(&env(&[("ALLOWED_HOSTS", "not json")]), None).is_ok()); // bare name: ignored
        assert!(
            Settings::load(&env(&[("ENGINE_CONSOLE_ALLOWED_HOSTS", "not json")]), None).is_err()
        );
    }

    #[test]
    fn test_dotenv_quotes_comments_and_expansion() {
        let got = parse_dotenv(
            "export A=1\nB='x # y'\nC=\"q\\n\" \nD=${A}-${MISSING:-z} # trailing\n#E=2\n",
            &env(&[]),
        );
        assert_eq!(
            got,
            env(&[("A", "1"), ("B", "x # y"), ("C", "q\n"), ("D", "1-z")])
                .into_iter()
                .collect::<Vec<_>>()
        );
    }
}
