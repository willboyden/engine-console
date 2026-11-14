//! The hub's env file (dotenv) and its settings' value parsers, moved out of `ec-hub/src/config.rs` unchanged (ADR
//! 0023) so the browser's Config files page can check an env draft with the same code, without sending it (an env
//! draft may hold a token). `ec_hub::config` re-exports everything here; `Settings` itself (paths derived from the
//! user's home, the file reader) stays in the hub.
//!
//! [`check_env_draft`] is the draft check without `Settings`: every assignment parsed by its setting's parser in
//! `apply`'s order, then the value checks of `Settings::validate_values`. `ec-hub`'s `test_env_draft_check_matches`
//! proves it gives the hub's `config::check_env_text` answer.

use std::collections::HashMap;
use std::path::PathBuf;

use serde_json::Value;

pub const PREFIX: &str = "ENGINE_CONSOLE_";
pub const PREFIXED: &[&str] = &[
    "host",
    "port",
    "allowed_hosts",
    "trust_loopback",
    "data_dir",
];
/// Every setting `apply` reads, in its order, with the parser that reads it (`p_<kind>`): the env-file editor's field
/// list and the learn bundle's `config.env.*` coverage come from here. `test_settings_table_matches_apply` keeps it
/// equal to the `set!` lines below.
pub const SETTINGS: &[(&str, &str)] = &[
    ("host", "str"),
    ("port", "port"),
    ("data_dir", "path"),
    ("hf_cache_dir", "path"),
    ("hf_cache_container_path", "str"),
    ("hf_endpoint", "str"),
    ("hf_token", "opt_str"),
    ("hf_token_file", "path"),
    ("hf_allowed_hosts", "str_list"),
    ("docker_bin", "str"),
    ("docker_context", "str"),
    ("engine_network", "str"),
    ("gateway_image", "str"),
    ("egress_proxy", "opt_str"),
    ("egress_ca_bundle", "opt_path"),
    ("require_egress_proxy", "bool"),
    ("discovery_enabled", "bool"),
    ("discovery_interval_s", "float"),
    ("discovery_ports", "int_list"),
    ("port_range_start", "int"),
    ("port_range_end", "int"),
    ("engine_image_allowlist", "str_list"),
    ("gateway_image_allowlist", "str_list"),
    ("enable_api_docs", "bool"),
    ("allowed_hosts", "str_list"),
    ("body_cap_bytes", "u64"),
    ("chat_body_cap_bytes", "u64"),
    ("import_body_cap_bytes", "u64"),
    ("max_sse_connections", "int"),
    ("max_sse_per_key", "int"),
    ("max_chat_streams", "int"),
    ("max_chat_streams_per_key", "int"),
    ("sse_idle_timeout_s", "float"),
    ("max_download_queue", "int"),
    ("startup_timeout_s", "int"),
    ("engine_hf_offline", "bool"),
    ("otlp_endpoint", "str"),
    ("otlp_enabled", "bool"),
    ("frontend_dir", "opt_path"),
    ("background_tasks", "bool"),
    ("scrape_interval_s", "float"),
    ("supervisor_interval_s", "float"),
    ("download_concurrency", "int"),
    ("trust_loopback", "bool"),
    ("hosts_file", "path"),
    ("hub_key_file", "path"),
    ("node_policy", "opt_path"),
    ("node_command", "opt_str_list"),
    ("remote_node_command", "str"),
    ("fleet_enabled", "bool"),
    ("restart_settle_s", "float"),
];

/// The variable name a setting is read as: `ENGINE_CONSOLE_<NAME>` for the five generic ones, else `<NAME>`.
pub fn env_var_name(field: &str) -> String {
    let up = field.to_ascii_uppercase();
    if PREFIXED.contains(&field) {
        format!("{PREFIX}{up}")
    } else {
        up
    }
}

/// A setting that cannot be used. `field` is the setting's name; the message never contains its value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SettingsError {
    pub field: String,
    pub message: String,
}

impl std::fmt::Display for SettingsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}

impl std::error::Error for SettingsError {}

/// A [`SettingsError`] for `field`.
pub fn err(field: &str, message: impl Into<String>) -> SettingsError {
    SettingsError {
        field: field.to_string(),
        message: message.into(),
    }
}

/// A source's variables (lower-case names) with each bare PREFIXED name dropped and ENGINE_CONSOLE_<NAME> moved to
/// its place, so only the prefixed value is found.
pub fn prefix_only(env: &[(String, String)]) -> Vec<(String, String)> {
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

/// Something about a value in an env-file draft that loads but probably does not do what was meant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueNote {
    /// A path setting starting with `~`: the hub never expands it, so it names a directory called `~`.
    Tilde,
    /// A negative byte cap (`p_u64`), which the loader turns into 0.
    Negative,
}

/// The env-file editor's value notes: (name as written, setting, note) for each assignment of `text` that loads but
/// is probably a mistake. Values never leave this function.
pub fn env_value_notes(text: &str) -> Vec<(String, String, ValueNote)> {
    let pl = PREFIX.to_ascii_lowercase();
    let mut out = Vec::new();
    for (name, v) in parse_dotenv(text, &[]) {
        let lk = name.to_ascii_lowercase();
        if PREFIXED.contains(&lk.as_str()) {
            continue;
        }
        let field = match lk.strip_prefix(&pl) {
            Some(f) if PREFIXED.contains(&f) => f.to_string(),
            _ => lk,
        };
        let Some((_, kind)) = SETTINGS.iter().find(|(n, _)| *n == field) else {
            continue;
        };
        let note = match *kind {
            "path" | "opt_path" if v.starts_with('~') => ValueNote::Tilde,
            "u64" if p_int(&field, &v).is_ok_and(|i| i < 0) => ValueNote::Negative,
            _ => continue,
        };
        out.push((name, field, note));
    }
    out
}

// ── value parsing (pydantic-settings) ──────────────────────────────────────────────────────────────────────
pub fn p_str(_f: &str, v: &str) -> Result<String, SettingsError> {
    Ok(v.to_string())
}

pub fn p_opt_str(_f: &str, v: &str) -> Result<Option<String>, SettingsError> {
    Ok(Some(v.to_string()))
}

pub fn p_path(_f: &str, v: &str) -> Result<PathBuf, SettingsError> {
    Ok(py_path(v))
}

pub fn p_opt_path(_f: &str, v: &str) -> Result<Option<PathBuf>, SettingsError> {
    Ok(Some(py_path(v)))
}

/// `pathlib.Path(v)`: "" is ".", trailing slashes and "//" are folded.
fn py_path(v: &str) -> PathBuf {
    if v.is_empty() {
        return PathBuf::from(".");
    }
    PathBuf::from(v).components().collect()
}

pub fn p_bool(f: &str, v: &str) -> Result<bool, SettingsError> {
    match v.trim().to_ascii_lowercase().as_str() {
        "1" | "on" | "t" | "true" | "y" | "yes" => Ok(true),
        "0" | "off" | "f" | "false" | "n" | "no" => Ok(false),
        _ => Err(err(
            f,
            "Input should be a valid boolean, unable to interpret input",
        )),
    }
}

pub fn p_int(f: &str, v: &str) -> Result<i64, SettingsError> {
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

pub fn p_port(f: &str, v: &str) -> Result<u16, SettingsError> {
    let i = p_int(f, v)?;
    // pydantic accepts any int here; uvicorn then fails to bind outside 0-65535. Refused up front instead.
    u16::try_from(i).map_err(|_| err(f, "port must be 0-65535"))
}

pub fn p_u64(f: &str, v: &str) -> Result<u64, SettingsError> {
    let i = p_int(f, v)?;
    Ok(u64::try_from(i).unwrap_or(0))
}

/// The largest value a float setting takes. Every float setting is a number of seconds that becomes a `Duration`,
/// and `Duration::from_secs_f64` panics on infinity and on anything past about 1.8e19 (deviation DV51).
pub const MAX_FLOAT_SETTING: f64 = 1e15;

pub fn p_float(f: &str, v: &str) -> Result<f64, SettingsError> {
    let t = v.trim().to_ascii_lowercase();
    // pydantic also accepts inf/infinity/nan; they are refused here (DV51), as is a number Rust's own parser reads as
    // infinite or NaN ("1e999", "infinity" in any case).
    let x = t.replace('_', "").parse::<f64>().map_err(|_| {
        err(
            f,
            "Input should be a valid number, unable to parse string as a number",
        )
    })?;
    if !x.is_finite() {
        return Err(err(
            f,
            "Input should be a finite number (inf and nan are refused: every float setting is a number of seconds)",
        ));
    }
    if x > MAX_FLOAT_SETTING {
        return Err(err(
            f,
            format!("Input should be at most {MAX_FLOAT_SETTING:e} seconds"),
        ));
    }
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

pub fn p_str_list(f: &str, v: &str) -> Result<Vec<String>, SettingsError> {
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

pub fn p_opt_str_list(f: &str, v: &str) -> Result<Option<Vec<String>>, SettingsError> {
    match json_of(f, v)? {
        Value::Null => Ok(None),
        other => p_str_list(f, &other.to_string()).map(Some),
    }
}

pub fn p_int_list(f: &str, v: &str) -> Result<Vec<i64>, SettingsError> {
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

/// The largest env file read (a few dozen settings; JSON lists included, far below this).
pub const MAX_ENV_FILE: usize = 1024 * 1024;
/// The most bytes all values of one env file may add up to after `${VAR}` expansion. A value can only outgrow the
/// file through expansion, and `A=${B}${B}` lines double at each step: a file past this counts as unreadable, like
/// one past `MAX_ENV_FILE` (DV39), instead of exhausting memory.
pub const MAX_ENV_EXPANDED: usize = 8 * 1024 * 1024;

/// The env file's `${VAR}` expansions add up to more than [`MAX_ENV_EXPANDED`] bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpansionTooLarge;

/// [`try_parse_dotenv`], with a file whose expansions grow too large read as empty (unreadable).
pub fn parse_dotenv(text: &str, environ: &[(String, String)]) -> Vec<(String, String)> {
    try_parse_dotenv(text, environ).unwrap_or_default()
}

/// The dotenv reader (see [`read_env_file`]); Err when the values add up to more than [`MAX_ENV_EXPANDED`] bytes.
pub fn try_parse_dotenv(
    text: &str,
    environ: &[(String, String)],
) -> Result<Vec<(String, String)>, ExpansionTooLarge> {
    let mut budget = MAX_ENV_EXPANDED;
    let mut out: Vec<(String, String)> = Vec::new();
    // Indexes, so a file of n lines costs O(n) lookups, not O(n^2) scans (the Config files page checks drafts of up
    // to a megabyte with this function). `env` keeps the FIRST entry of a name, as the linear `find` it replaces did;
    // `index` maps a key of the file to its slot in `out`.
    let mut env: HashMap<&str, &str> = HashMap::with_capacity(environ.len());
    for (k, v) in environ {
        env.entry(k.as_str()).or_insert(v.as_str());
    }
    let mut index: HashMap<String, usize> = HashMap::new();
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
            expand_vars(
                &value,
                |name| {
                    env.get(name)
                        .copied()
                        .or_else(|| index.get(name).map(|&i| out[i].1.as_str()))
                },
                budget,
            )
            .ok_or(ExpansionTooLarge)?
        } else {
            value
        };
        budget = budget.checked_sub(value.len()).ok_or(ExpansionTooLarge)?;
        if let Some(&i) = index.get(&key) {
            out[i].1 = value;
        } else {
            index.insert(key.clone(), out.len());
            out.push((key, value));
        }
    }
    Ok(out)
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

/// `${NAME}` / `${NAME:-default}` replaced by `lookup(NAME)` (the environment, then earlier keys of the file), else
/// the default. None as soon as the result would pass `max` bytes.
fn expand_vars<'a>(
    v: &str,
    lookup: impl Fn(&str) -> Option<&'a str>,
    max: usize,
) -> Option<String> {
    let mut out = String::new();
    let mut rest = v;
    let push = |out: &mut String, s: &str| (out.len() + s.len() <= max).then(|| out.push_str(s));
    while let Some(i) = rest.find("${") {
        push(&mut out, &rest[..i])?;
        let after = &rest[i + 2..];
        let Some(j) = after.find('}') else {
            push(&mut out, &rest[i..])?;
            return Some(out);
        };
        let inner = &after[..j];
        let (name, default) = inner.split_once(":-").map_or((inner, ""), |(n, d)| (n, d));
        push(&mut out, lookup(name).unwrap_or(default))?;
        rest = &after[j + 1..];
    }
    push(&mut out, rest)?;
    Some(out)
}

// ── the draft check without `Settings` (the browser's) ─────────────────────────────────────────────────────
/// The address the hub listens on when ENGINE_CONSOLE_HOST is not set.
pub const DEFAULT_HOST: &str = "127.0.0.1";
/// The addresses that count as loopback (`api::security`: the trust-loopback rule, and the env check's warning).
pub const LOOPBACK: &[&str] = &["127.0.0.1", "::1", "localhost"];

/// `Settings::validate_values` on the listen address.
pub fn check_host(host: &str) -> Result<(), SettingsError> {
    if host.trim().is_empty() {
        return Err(err(
            "host",
            "ENGINE_CONSOLE_HOST must not be empty: an empty address listens on every interface",
        ));
    }
    Ok(())
}

/// `Settings::validate_values` on HF_CACHE_DIR, given as the path's text. Absolute means "starts with /" (what
/// `Path::is_absolute` answers on Unix, also on wasm32, where std's answer differs).
pub fn check_hf_cache_dir(path: &str) -> Result<(), SettingsError> {
    if !path.starts_with('/') {
        return Err(err("hf_cache_dir", "HF_CACHE_DIR must be an absolute path"));
    }
    if path.contains([':', ',', '\n', '\r', '\0']) {
        return Err(err(
            "hf_cache_dir",
            "HF_CACHE_DIR must not contain ':' ',' or newlines (docker -v syntax injection)",
        ));
    }
    Ok(())
}

/// Parse `v` with the parser of `kind` (a [`SETTINGS`] kind, `p_<kind>`), keeping only the verdict.
pub fn check_kind(kind: &str, field: &str, v: &str) -> Result<(), SettingsError> {
    match kind {
        "port" => p_port(field, v).map(drop),
        "bool" => p_bool(field, v).map(drop),
        "int" => p_int(field, v).map(drop),
        "u64" => p_u64(field, v).map(drop),
        "float" => p_float(field, v).map(drop),
        "str_list" => p_str_list(field, v).map(drop),
        "opt_str_list" => p_opt_str_list(field, v).map(drop),
        "int_list" => p_int_list(field, v).map(drop),
        // str, opt_str, path, opt_path: every text parses
        _ => Ok(()),
    }
}

/// What the env check's warnings need of a draft that loads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnvSummary {
    pub host: String,
    pub trust_loopback: bool,
}

/// The env-file editor's check of a draft, as `ec_hub::config::check_env_text` gives it but without `Settings`:
/// `text` read as the hub reads its env file with an EMPTY environment, every assignment parsed by its setting's
/// parser in `apply`'s order (the first error wins), then the value checks. Returns the names the file sets, as
/// written and in order, and the first error. Values never leave this function.
pub fn check_env_draft(text: &str) -> (Vec<String>, Result<EnvSummary, SettingsError>) {
    let vars = parse_dotenv(text, &[]);
    let names = vars.iter().map(|(k, _)| k.clone()).collect();
    let mut map: HashMap<String, String> = HashMap::new();
    for (k, v) in prefix_only(&vars) {
        map.insert(k, v);
    }
    let check = || {
        let mut out = EnvSummary {
            host: DEFAULT_HOST.to_string(),
            trust_loopback: false,
        };
        for (field, kind) in SETTINGS {
            let Some(v) = map.get(*field) else { continue };
            check_kind(kind, field, v)?;
            match *field {
                "host" => out.host.clone_from(v),
                "trust_loopback" => out.trust_loopback = p_bool(field, v)?,
                _ => {}
            }
        }
        check_host(&out.host)?;
        if let Some(v) = map.get("hf_cache_dir") {
            check_hf_cache_dir(v)?;
        }
        Ok(out)
    };
    (names, check())
}
