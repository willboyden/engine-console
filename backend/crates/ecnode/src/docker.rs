//! Python: node/src/ecnode/docker.py.
//!
//! Docker through the `docker` CLI only: argv lists, timeouts, `--context` from the policy, never a shell.
//!
//! What this module builds is what the node runs (docker.run) and what it shows (docker.render): the same argv, never
//! a secret value in it. Every engine container gets `--cap-drop ALL`, `no-new-privileges`, `--pids-limit 4096`,
//! `--restart no`, `--pull never` (pulls are a separate, policy-gated op) and the policy's internal engine network.
//! The node changes only containers labelled `engine-console=1` (docs/FLEET-PROTOCOL.md §4 and §5).

use std::collections::HashSet;
use std::io::Write;
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock};

use ec_proto::{
    BAD_ARGS, CONFLICT, DOCKER_ERROR, IO_ERROR, NOT_FOUND, NOT_OWNED, POLICY_DENIED, PORT_IN_USE,
    TIMEOUT,
};
use serde_json::{Map, Value, json};

use crate::confine::{open_root, resolve_existing, split_rel};
use crate::errors::OpError;
use crate::policy::{Policy, full, valid_image};
use crate::util::{CmdResult, EnvMap, Runner, clean_text, is_secret_name, py, scrub};

pub const LABEL: &str = "engine-console";
pub const ROLE_LABEL: &str = "engine-console.role";
pub const INSTANCE_LABEL: &str = "engine-console.instance";
pub const ENGINE_LABEL: &str = "engine-console.engine";
pub const GATEWAY_LISTEN: u16 = 8080;
/// constant shell text: the only variable input (the nginx config) travels in an env var, never in this string
pub const GATEWAY_SH: &str =
    r#"printf "%s\n" "$NGINX_CONF" > /tmp/nginx.conf && exec nginx -c /tmp/nginx.conf"#;

pub static CONTAINER_REF: LazyLock<regex::Regex> =
    LazyLock::new(|| full("[A-Za-z0-9][A-Za-z0-9_.-]{0,127}"));
pub static RUN_NAME: LazyLock<regex::Regex> = LazyLock::new(|| full("ec-[a-z0-9][a-z0-9-]{0,58}"));
pub static GATEWAY_NAME: LazyLock<regex::Regex> =
    LazyLock::new(|| full("ec-[a-z0-9][a-z0-9-]{0,60}"));
pub static INSTANCE: LazyLock<regex::Regex> = LazyLock::new(|| full("[A-Za-z0-9_-]{1,64}"));
pub static ENV_NAME: LazyLock<regex::Regex> =
    LazyLock::new(|| full("[A-Za-z_][A-Za-z0-9_]{0,127}"));
pub static VOLUME: LazyLock<regex::Regex> = LazyLock::new(|| full("ec-[a-z0-9][a-z0-9_.-]{0,60}"));
pub static SHM: LazyLock<regex::Regex> = LazyLock::new(|| full("([0-9]{1,4})([kmg])"));
pub static LABEL_KEY: LazyLock<regex::Regex> =
    LazyLock::new(|| full(r"engine-console\.[a-z0-9][a-z0-9_.-]{0,62}"));
pub static NGINX_HOST: LazyLock<regex::Regex> = LazyLock::new(|| full("[a-z0-9][a-z0-9_.-]{0,62}"));
/// A plain `env` name that looks secret would put its value on the docker command line: it belongs in secret_env.
static ENV_LOOKS_SECRET: LazyLock<regex::Regex> = LazyLock::new(|| {
    regex::Regex::new("(?i)(key|token|secret|password)").unwrap_or_else(|_| std::process::abort())
});
/// secret_env names that would change what the docker CLI process (its loader, Go runtime or OpenTelemetry
/// exporter) does are refused.
static CHILD_ENV_DENY: LazyLock<regex::Regex> = LazyLock::new(|| {
    regex::Regex::new(
        r"(?i)^(?:LD_|DYLD_|DOCKER_|XDG_|LC_|BUILDKIT_|COMPOSE_|PYTHON|NOTARY_|SSL_|OPENSSL_|GCONV_|MALLOC_|GLIBC_|GIT_|SSH_|KRB5|NIX_|OTEL_|GRPC_)|^(?:PATH|HOME|USER|LOGNAME|SHELL|TMPDIR|TMP|TEMP|PWD|OLDPWD|IFS|ENV|BASH_ENV|CDPATH|LANG|LANGUAGE|TZ|TZDIR|TERM|HTTP_PROXY|HTTPS_PROXY|NO_PROXY|ALL_PROXY|FTP_PROXY|HOSTALIASES|LOCALDOMAIN|RES_OPTIONS|NLSPATH|LOCPATH|SSLKEYLOGFILE|ZONEINFO|GODEBUG|GOGC|GOMAXPROCS|GOMEMLIMIT|GOTRACEBACK|GORACE|GOFLAGS|GOROOT|GOPATH|GOENV|GOTOOLCHAIN|GOCOVERDIR)$",
    )
    .unwrap_or_else(|_| std::process::abort())
});
pub const MAX_FILES: usize = 4;
pub const DEFAULT_SOCKET: &str = "/var/run/docker.sock";

fn sv(args: &[&str]) -> Vec<String> {
    args.iter().map(|s| (*s).to_string()).collect()
}

/// The `--context` to pass. The policy's context when it names one. Otherwise the CLI's own choice, with one
/// exception: a node started with a scrubbed environment has no DOCKER_HOST, so on a rootless-only machine
/// "default" means a rootful socket this user cannot use; then an existing `rootless` context is used instead.
/// The default socket is only checked for access, never connected to.
pub fn effective_context(
    binary: &str,
    configured: Option<&str>,
    runner: &dyn Runner,
    env: &EnvMap,
    default_socket: &str,
) -> Option<String> {
    if let Some(c) = configured.filter(|c| !c.is_empty()) {
        return Some(c.to_string());
    }
    let set = |k: &str| env.get(k).is_some_and(|v| !v.is_empty());
    if set("DOCKER_HOST") || set("DOCKER_CONTEXT") {
        return None; // the operator pointed the CLI somewhere: follow it
    }
    let shown = runner.run(&sv(&[binary, "context", "show"]), 5.0, None);
    if !shown.ok() || py::strip(&shown.out) != "default" {
        return None; // no CLI, or a current context chosen in the CLI config
    }
    if nix::unistd::access(default_socket, nix::unistd::AccessFlags::W_OK).is_ok() {
        return None; // a default daemon this user may use
    }
    let listed = runner.run(
        &sv(&[binary, "context", "ls", "--format", "{{.Name}}"]),
        5.0,
        None,
    );
    let names: HashSet<String> = if listed.ok() {
        py::splitlines(&listed.out)
            .into_iter()
            .map(|l| py::rstrip_chars(py::strip(l), " *").to_string())
            .collect()
    } else {
        HashSet::new()
    };
    names.contains("rootless").then(|| "rootless".to_string())
}

/// One `docker ps` row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PsRow {
    pub id: String,
    pub name: String,
    pub state: String,
    pub status: String,
}

#[derive(Clone)]
pub struct Docker {
    pub binary: String,
    pub context: Option<String>,
    runner: Arc<dyn Runner>,
}

impl std::fmt::Debug for Docker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Docker")
            .field("binary", &self.binary)
            .field("context", &self.context)
            .finish_non_exhaustive()
    }
}

impl Docker {
    pub fn new(binary: &str, context: Option<String>, runner: Arc<dyn Runner>) -> Self {
        Self {
            binary: binary.to_string(),
            context,
            runner,
        }
    }

    pub fn runner(&self) -> &Arc<dyn Runner> {
        &self.runner
    }

    pub fn argv(&self, args: &[&str]) -> Vec<String> {
        let mut out = vec![self.binary.clone()];
        if let Some(c) = &self.context {
            out.push("--context".into());
            out.push(c.clone());
        }
        out.extend(args.iter().map(|s| (*s).to_string()));
        out
    }

    pub fn call(&self, args: &[&str], timeout: f64, env: Option<&EnvMap>) -> CmdResult {
        self.runner.run(&self.argv(args), timeout, env)
    }

    pub fn ok(
        &self,
        args: &[&str],
        timeout: f64,
        env: Option<&EnvMap>,
    ) -> Result<CmdResult, OpError> {
        let res = self.call(args, timeout, env);
        if !res.ok() {
            return Err(docker_error(args.first().copied().unwrap_or(""), &res));
        }
        Ok(res)
    }

    // ── reads ────────────────────────────────────────────────────────────────────────────────────────────
    /// Raw `docker container inspect` documents (untrusted; never keep Config.Env). Missing ones are skipped.
    pub fn inspect(&self, refs: &[String]) -> Result<Vec<Value>, OpError> {
        if refs.is_empty() {
            return Ok(Vec::new());
        }
        let mut args = vec!["container", "inspect"];
        args.extend(refs.iter().map(String::as_str));
        let res = self.call(&args, 30.0, None);
        if res.truncated {
            // the output was cut while it was read (util::RUN_MAX_OUT): never parse half a document (node hardening)
            return Err(OpError::new(
                DOCKER_ERROR,
                format!(
                    "docker inspect printed more than {} MiB; refusing to read it",
                    crate::util::RUN_MAX_OUT >> 20
                ),
            ));
        }
        let docs: Value = if py::strip(&res.out).is_empty() {
            Value::Array(Vec::new())
        } else {
            serde_json::from_str(&res.out).unwrap_or(Value::Array(Vec::new()))
        };
        let empty = match &docs {
            Value::Array(a) => a.is_empty(),
            other => !py::truthy(other),
        };
        if !res.ok() && empty && !missing(&res) {
            return Err(docker_error("inspect", &res));
        }
        Ok(match docs {
            Value::Array(a) => a.into_iter().filter(Value::is_object).collect(),
            _ => Vec::new(),
        })
    }

    pub fn inspect_one(&self, r: &Value) -> Result<Option<Value>, OpError> {
        let s = r.as_str().unwrap_or("");
        if !CONTAINER_REF.is_match(s) {
            return Err(OpError::new(
                BAD_ARGS,
                format!(
                    "{} is not a container name",
                    py::repr_take(&py::str_of(r), 130)
                ),
            ));
        }
        Ok(self.inspect(&[s.to_string()])?.into_iter().next())
    }

    /// [{id, name, state, status}] from `docker ps` (no command column: it can hold secrets).
    pub fn ps(&self, all: bool, label: Option<&str>) -> Result<Vec<PsRow>, OpError> {
        let mut args: Vec<String> = sv(&[
            "ps",
            "--no-trunc",
            "--format",
            "{{.ID}}\t{{.Names}}\t{{.State}}\t{{.Status}}",
        ]);
        if all {
            args.insert(1, "-a".into());
        }
        if let Some(l) = label {
            args.splice(1..1, ["--filter".to_string(), format!("label={l}")]);
        }
        let refs: Vec<&str> = args.iter().map(String::as_str).collect();
        let res = self.ok(&refs, 20.0, None)?;
        let mut out = Vec::new();
        for line in py::splitlines(&res.out) {
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() >= 4 && !parts[0].is_empty() {
                out.push(PsRow {
                    id: parts[0].into(),
                    name: parts[1].split(',').next().unwrap_or("").into(),
                    state: parts[2].into(),
                    status: parts[3].into(),
                });
            }
        }
        Ok(out)
    }

    /// hello.docker: {available, context, rootless, version, nvidia_runtime, error}.
    pub fn facts(&self) -> Map<String, Value> {
        let res = self.call(&["info", "--format", "{{json .}}"], 8.0, None);
        let mut out = Map::new();
        out.insert("available".into(), json!(false));
        out.insert("context".into(), json!(self.context));
        out.insert("rootless".into(), Value::Null);
        out.insert("version".into(), Value::Null);
        out.insert("nvidia_runtime".into(), Value::Null);
        out.insert("error".into(), Value::Null);
        if res.rc == 127 {
            out.insert(
                "error".into(),
                json!(format!(
                    "the docker CLI ({}) is not installed or not on PATH",
                    self.binary
                )),
            );
            return out;
        }
        let info: Option<Value> = if py::strip(&res.out).is_empty() {
            None
        } else {
            serde_json::from_str(&res.out).ok()
        };
        let Some(Value::Object(info)) = info else {
            let fl = first_line(&res);
            out.insert(
                "error".into(),
                json!(if fl.is_empty() {
                    "docker info gave no answer".to_string()
                } else {
                    fl
                }),
            );
            return out;
        };
        if let Some(Value::Array(errors)) = info.get("ServerErrors")
            && let Some(first) = errors.first()
        {
            out.insert(
                "error".into(),
                json!(clean_text(&scrub(&py::str_of(first)), 300)),
            );
            return out;
        }
        let sec = info.get("SecurityOptions").filter(|v| py::truthy(v));
        let runtimes = info.get("Runtimes").filter(|v| py::truthy(v));
        let version = match info.get("ServerVersion").filter(|v| py::truthy(v)) {
            Some(v) => clean_text(&py::str_of(v), 64),
            None => String::new(),
        };
        out.insert("available".into(), json!(res.ok()));
        out.insert(
            "version".into(),
            if version.is_empty() {
                Value::Null
            } else {
                json!(version)
            },
        );
        out.insert(
            "rootless".into(),
            match sec {
                None => json!(false),
                Some(Value::Array(a)) => {
                    json!(a.iter().any(|s| py::str_of(s).contains("rootless")))
                }
                Some(_) => Value::Null,
            },
        );
        out.insert(
            "nvidia_runtime".into(),
            match runtimes {
                None => json!(false),
                Some(Value::Object(m)) => json!(m.contains_key("nvidia")),
                Some(_) => Value::Null,
            },
        );
        if !res.ok() {
            out.insert("error".into(), json!(first_line(&res)));
        }
        if self.context.is_none() {
            let ctx = self.call(&["context", "show"], 5.0, None);
            let c = clean_text(py::strip(&ctx.out), 64);
            out.insert(
                "context".into(),
                if ctx.ok() && !c.is_empty() {
                    json!(c)
                } else {
                    Value::Null
                },
            );
        }
        out
    }
}

fn first_line(res: &CmdResult) -> String {
    let src = if res.err.is_empty() {
        &res.out
    } else {
        &res.err
    };
    let text = py::strip(src);
    let lines = py::splitlines(text);
    let line = match lines.first() {
        Some(l) => (*l).to_string(),
        None => format!("exit status {}", res.rc),
    };
    clean_text(&scrub(&line), 300)
}

fn missing(res: &CmdResult) -> bool {
    let t = format!("{}{}", res.err, res.out).to_lowercase();
    t.contains("no such container") || t.contains("no such object") || t.contains("not found")
}

/// A docker failure as a protocol error with a sentence a person can act on (never a secret).
pub fn docker_error(verb: &str, res: &CmdResult) -> OpError {
    let text = format!("{}\n{}", res.err, res.out).to_lowercase();
    let line = first_line(res);
    let (code, msg) = if res.rc == 124 {
        (
            TIMEOUT,
            format!("docker {verb} did not finish in time; it may or may not have taken effect"),
        )
    } else if res.rc == 127 {
        (
            DOCKER_ERROR,
            "the docker CLI is not installed on this machine (or not on PATH)".to_string(),
        )
    } else if text.contains("cannot connect to the docker daemon")
        || text.contains("permission denied while trying to connect")
    {
        (
            DOCKER_ERROR,
            format!("docker {verb}: cannot reach the docker daemon ({line})"),
        )
    } else if text.contains("no such container") || text.contains("no such object") {
        (
            NOT_FOUND,
            format!("docker {verb}: no such container ({line})"),
        )
    } else if text.contains("no such image")
        || text.contains("unable to find image")
        || text.contains("manifest unknown")
    {
        (
            NOT_FOUND,
            format!("docker {verb}: the image is not on this machine; pull it first ({line})"),
        )
    } else if text.contains("port is already allocated") || text.contains("address already in use")
    {
        (
            PORT_IN_USE,
            format!("docker {verb}: the host port is already in use ({line})"),
        )
    } else if text.contains("is already in use by container") || text.contains("conflict") {
        (
            CONFLICT,
            format!("docker {verb}: a container with that name already exists ({line})"),
        )
    } else {
        (DOCKER_ERROR, format!("docker {verb} failed: {line}"))
    };
    OpError::new(code, msg)
}

/// `{str(k): str(v) for k, v in Config.Labels.items()}`
pub fn labels_of(doc: &Value) -> Vec<(String, String)> {
    match doc.get("Config").and_then(|c| c.get("Labels")) {
        Some(Value::Object(m)) => m.iter().map(|(k, v)| (k.clone(), py::str_of(v))).collect(),
        _ => Vec::new(),
    }
}

pub fn label<'a>(labels: &'a [(String, String)], key: &str) -> Option<&'a str> {
    labels
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.as_str())
}

/// The console touches only its own containers, and with role/instance exactly that role and instance.
pub fn check_owned(
    doc: &Value,
    name: &str,
    role: Option<&str>,
    instance: Option<&str>,
) -> Result<(), OpError> {
    let labels = labels_of(doc);
    if label(&labels, LABEL) != Some("1") {
        return Err(OpError::new(
            NOT_OWNED,
            format!(
                "container {} was not created by Engine Console; this node will not touch it",
                py::repr(name)
            ),
        ));
    }
    if let Some(r) = role.filter(|r| !r.is_empty())
        && label(&labels, ROLE_LABEL) != Some(r)
    {
        return Err(OpError::new(
            NOT_OWNED,
            format!("container {} is not a console {r}", py::repr(name)),
        ));
    }
    if let Some(i) = instance.filter(|i| !i.is_empty())
        && label(&labels, INSTANCE_LABEL) != Some(i)
    {
        return Err(OpError::new(
            NOT_OWNED,
            format!(
                "container {} belongs to another console instance",
                py::repr(name)
            ),
        ));
    }
    Ok(())
}

pub fn container_name(r: &Value) -> Result<String, OpError> {
    match r.as_str() {
        Some(s) if CONTAINER_REF.is_match(s) => Ok(s.to_string()),
        _ => Err(OpError::new(
            BAD_ARGS,
            format!(
                "{} is not a valid container name",
                py::repr_take(&py::str_of(r), 130)
            ),
        )),
    }
}

// ── run specs (docs/FLEET-PROTOCOL.md §5) ───────────────────────────────────────────────────────────────
#[derive(Debug, Clone, PartialEq)]
pub struct RunPlan {
    pub name: String,
    pub instance: String,
    pub image: String,
    pub argv: Vec<String>,
    pub container_port: i64,
    pub network: String,
    pub alias: String,
    pub gpu_uuids: Vec<String>,
    pub labels: Vec<(String, String)>,
    pub env: Vec<(String, String)>,
    /// values: child environment only
    pub secret_env: Vec<(String, String)>,
    pub mounts: Vec<(PathBuf, String, bool)>,
    pub volumes: Vec<(String, String)>,
    /// (container path, host path, content)
    pub files: Vec<(String, PathBuf, String)>,
    pub shm_size: String,
    pub ipc_host: bool,
    pub ulimits: Vec<(String, String)>,
}

/// Container ulimits an engine may ask for (`memlock` deliberately absent).
pub const ULIMITS: &[&str] = &["nofile", "stack"];
static ULIMIT_VALUE: LazyLock<regex::Regex> =
    LazyLock::new(|| full("(?:-1|[0-9]{1,15})(?::(?:-1|[0-9]{1,15}))?"));

fn bad(msg: impl Into<String>) -> OpError {
    OpError::new(BAD_ARGS, msg)
}

fn has_ctrl(s: &str) -> bool {
    s.chars().any(|c| (c as u32) < 32 || c as u32 == 127)
}

pub fn check_dest(dest: &Value, what: &str) -> Result<String, OpError> {
    let d = match dest.as_str() {
        Some(d) if d.starts_with('/') && py::len(d) <= 512 && d != "/" => d,
        _ => {
            return Err(bad(format!(
                "{what}: the container path must be absolute (and not '/')"
            )));
        }
    };
    if d.chars().any(|c| ":,\\".contains(c)) || has_ctrl(d) {
        return Err(bad(format!(
            "{what}: the container path {} may not contain ':', ',', '\\' or control characters",
            py::repr_take(d, 200)
        )));
    }
    if d.split('/').any(|s| s == "." || s == "..") {
        return Err(bad(format!(
            "{what}: the container path {} may not contain '.' or '..' segments",
            py::repr_take(d, 200)
        )));
    }
    let t = d.trim_end_matches('/');
    Ok(if t.is_empty() {
        d.to_string()
    } else {
        t.to_string()
    })
}

pub fn shm_gib(value: &str) -> Result<f64, OpError> {
    let Some(c) = SHM.captures(value) else {
        return Err(bad(format!(
            "shm_size {} must look like 16g, 512m or 1024k",
            py::repr_take(value, 20)
        )));
    };
    let n: f64 = c.get(1).map_or(0.0, |m| m.as_str().parse().unwrap_or(0.0));
    let unit = match c.get(2).map(|m| m.as_str()) {
        Some("k") => 1.0 / (1024.0 * 1024.0),
        Some("m") => 1.0 / 1024.0,
        _ => 1.0,
    };
    Ok(n * unit)
}

/// How far below a mount source the secret scan looks, and how many entries it reads before refusing.
pub const SECRET_SCAN_DEPTH: usize = 4;
pub const SECRET_SCAN_MAX_ENTRIES: usize = 20000;
static MODEL_TREE: LazyLock<regex::Regex> =
    LazyLock::new(|| full(r"(?:models|datasets|spaces)--.+|blobs|snapshots|refs"));

/// The first key- or token-looking name in a mount source, down to SECRET_SCAN_DEPTH levels, relative to the
/// source; None when there is none. HF model trees are not walked. A source too big to check within `max_entries`
/// is refused rather than waved through.
pub fn secret_near_top(src: &Path, max_entries: usize) -> Result<Option<String>, OpError> {
    let base = py::basename(src).to_string_lossy().into_owned();
    if is_secret_name(&base) {
        return Ok(Some(base));
    }
    let shown = || py::repr_take(&py::path_str(src), 200);
    let mut seen = 0usize;
    let mut level: Vec<String> = vec![String::new()];
    for depth in 0..SECRET_SCAN_DEPTH {
        let mut next: Vec<String> = Vec::new();
        for rel in &level {
            let dir = if rel.is_empty() {
                src.to_path_buf()
            } else {
                py::join(src, &[rel.as_str()])
            };
            let entries: Vec<(String, bool)> = match std::fs::read_dir(&dir) {
                Ok(it) => it
                    .filter_map(Result::ok)
                    .map(|e| {
                        let is_dir = e.file_type().is_ok_and(|t| t.is_dir());
                        (e.file_name().to_string_lossy().into_owned(), is_dir)
                    })
                    .collect(),
                Err(e) if e.raw_os_error() == Some(nix::libc::ENOTDIR) => continue,
                Err(e) => {
                    if depth == 0 {
                        return Err(OpError::new(
                            IO_ERROR,
                            format!(
                                "cannot list {} to check it before mounting: {}",
                                shown(),
                                py::io_strerror(&e)
                            ),
                        ));
                    }
                    continue;
                }
            };
            seen += entries.len();
            if seen > max_entries {
                return Err(OpError::new(
                    POLICY_DENIED,
                    format!(
                        "{} holds more than {max_entries} entries near its top, too many to check for keys before \
                         mounting; mount a narrower folder",
                        shown()
                    ),
                ));
            }
            for (name, is_dir) in entries {
                let path = if rel.is_empty() {
                    name.clone()
                } else {
                    format!("{rel}/{name}")
                };
                if is_secret_name(&name) {
                    return Ok(Some(path));
                }
                if is_dir && !MODEL_TREE.is_match(&name) {
                    next.push(path);
                }
            }
        }
        level = next;
    }
    Ok(None)
}

fn str_map(v: Option<&Value>) -> Vec<(String, Value)> {
    match v {
        Some(Value::Object(m)) => m.iter().map(|(k, x)| (k.clone(), x.clone())).collect(),
        _ => Vec::new(),
    }
}

fn list(v: Option<&Value>) -> Vec<Value> {
    match v {
        Some(Value::Array(a)) => a.clone(),
        _ => Vec::new(),
    }
}

/// Validate a cleaned RUN_SPEC against §5 and this node's policy; OpError says which field and why.
pub fn plan_run(
    spec: &Value,
    policy: &Policy,
    known_gpus: &HashSet<String>,
    state_dir: &Path,
    secret_scan_max_entries: usize,
) -> Result<RunPlan, OpError> {
    let Value::Object(spec) = spec else {
        return Err(bad("spec is missing"));
    };
    let name = match spec.get("name").and_then(Value::as_str) {
        Some(n) if RUN_NAME.is_match(n) => n.to_string(),
        _ => {
            return Err(bad(
                "spec.name must look like ec-<lowercase letters, digits, hyphens> (at most 61 characters)",
            ));
        }
    };
    let instance = match spec.get("instance").and_then(Value::as_str) {
        Some(i) if INSTANCE.is_match(i) => i.to_string(),
        _ => {
            return Err(bad(
                "spec.instance must be 1-64 letters, digits, '_' or '-'",
            ));
        }
    };
    let image = match spec.get("image").and_then(Value::as_str) {
        Some(i) if valid_image(i) => i.to_string(),
        _ => return Err(bad("spec.image is not a valid image reference")),
    };
    if !policy.image_allowed(&image) {
        return Err(OpError::new(
            POLICY_DENIED,
            format!(
                "image {} is not allowed by this node's policy (engine_images in {})",
                py::repr(&image),
                policy.where_()
            ),
        ));
    }
    let argv: Vec<String> = list(spec.get("argv"))
        .iter()
        .filter_map(|a| a.as_str().map(str::to_string))
        .collect();
    if argv.len() > 512 || argv.iter().any(|a| py::len(a) > 4096 || a.contains('\0')) {
        return Err(bad(
            "spec.argv: at most 512 arguments of at most 4096 characters, no NUL",
        ));
    }
    let port = match spec.get("container_port") {
        Some(Value::Number(n))
            if !n.is_f64() && n.as_i64().is_some_and(|p| (1..=65535).contains(&p)) =>
        {
            n.as_i64().unwrap_or(0)
        }
        _ => return Err(bad("spec.container_port must be 1-65535")),
    };
    let alias = match spec.get("network_alias").and_then(Value::as_str) {
        Some(a) if !a.is_empty() => a.to_string(),
        _ => name.clone(),
    };
    if !RUN_NAME.is_match(&alias) {
        return Err(bad(
            "spec.network_alias must look like ec-<lowercase letters, digits, hyphens>",
        ));
    }
    let mut gpus: Vec<String> = Vec::new();
    for u in list(spec.get("gpu_uuids")) {
        let u = u.as_str().unwrap_or("").to_string();
        if !known_gpus.contains(&u) {
            return Err(bad(format!(
                "GPU {} is not a GPU of this machine (see node.info)",
                py::repr_take(&u, 96)
            )));
        }
        if !gpus.contains(&u) {
            gpus.push(u);
        }
    }

    let mut labels: Vec<(String, String)> = vec![
        (LABEL.into(), "1".into()),
        (ROLE_LABEL.into(), "engine".into()),
        (INSTANCE_LABEL.into(), instance.clone()),
    ];
    for (k, v) in str_map(spec.get("labels")) {
        let v = v.as_str().unwrap_or("").to_string();
        if !LABEL_KEY.is_match(&k) {
            return Err(bad(format!(
                "label {}: keys must start with 'engine-console.' (lowercase, digits, . _ -)",
                py::repr_take(&k, 80)
            )));
        }
        if has_ctrl(&v) {
            return Err(bad(format!(
                "label {} has a control character in its value",
                py::repr(&k)
            )));
        }
        if let Some((_, have)) = labels.iter().find(|(x, _)| *x == k) {
            if *have != v {
                return Err(bad(format!(
                    "label {} must be {} for this container",
                    py::repr(&k),
                    py::repr(have)
                )));
            }
        } else {
            labels.push((k, v));
        }
    }

    let mut env: Vec<(String, String)> = Vec::new();
    for (k, v) in str_map(spec.get("env")) {
        if !ENV_NAME.is_match(&k) {
            return Err(bad(format!(
                "env name {} must be letters, digits and '_' (not starting with a digit)",
                py::repr_take(&k, 80)
            )));
        }
        if ENV_LOOKS_SECRET.is_match(&k) {
            return Err(bad(format!(
                "env {} looks like a secret: send it in secret_env so its value never reaches a command line",
                py::repr(&k)
            )));
        }
        env.push((k, v.as_str().unwrap_or("").to_string()));
    }
    let mut secret_env: Vec<(String, String)> = Vec::new();
    for (k, v) in str_map(spec.get("secret_env")) {
        if !ENV_NAME.is_match(&k) {
            return Err(bad(format!(
                "secret_env name {} must be letters, digits and '_' (not starting with a digit)",
                py::repr_take(&k, 80)
            )));
        }
        if CHILD_ENV_DENY.is_match(&k) {
            return Err(OpError::new(
                POLICY_DENIED,
                format!(
                    "secret_env name {} would change how the docker CLI itself runs on this machine; use another \
                     name",
                    py::repr(&k)
                ),
            ));
        }
        if !ENV_LOOKS_SECRET.is_match(&k) {
            return Err(bad(format!(
                "secret_env name {} does not look like a secret (key, token, secret or password): send it in env",
                py::repr(&k)
            )));
        }
        if env.iter().any(|(e, _)| *e == k) {
            return Err(bad(format!(
                "{} is in both env and secret_env",
                py::repr(&k)
            )));
        }
        secret_env.push((k, v.as_str().unwrap_or("").to_string()));
    }

    let mut dests: HashSet<String> = HashSet::new();
    let mut claim = |dest: String, what: &str| -> Result<String, OpError> {
        if !dests.insert(dest.clone()) {
            return Err(bad(format!(
                "{what}: container path {} is used twice",
                py::repr(&dest)
            )));
        }
        Ok(dest)
    };

    let mut mounts: Vec<(PathBuf, String, bool)> = Vec::new();
    for (i, m) in list(spec.get("mounts")).iter().enumerate() {
        let what = format!("mounts[{i}]");
        let h = open_root(policy, m.get("root").unwrap_or(&Value::Null), false)?;
        let rel = m.get("rel").and_then(Value::as_str);
        if m.get("ro") == Some(&Value::Bool(false)) {
            // A read-write mount lets the container plant links and swap directories inside the root (and, on a
            // rootful daemon, setuid files). Engines only read weights, so every mount is read-only.
            return Err(OpError::new(
                POLICY_DENIED,
                format!("{what}: model roots are only ever mounted read-only on this node"),
            ));
        }
        // mounts only: an empty rel is the root directory itself (still realpath-confined)
        let src = match rel {
            None | Some("") => h.real.clone(),
            Some(r) => resolve_existing(&h, &split_rel(Some(r), &format!("{what}.rel"))?)?,
        };
        let src_s = py::path_str(&src);
        if src_s.contains(':') || src_s.contains(',') {
            return Err(bad(format!(
                "{what}: host path {} contains ':' or ',' and cannot be mounted",
                py::repr_take(&src_s, 200)
            )));
        }
        if let Some(secret) = secret_near_top(&src, secret_scan_max_entries)? {
            return Err(OpError::new(
                POLICY_DENIED,
                format!(
                    "{what}: {} looks like a key or token file, and a container would see it; mount a folder \
                     without it (for Hugging Face, a root of kind hf-hub on the hub/ folder, not the whole HF_HOME)",
                    py::repr_take(&secret, 200)
                ),
            ));
        }
        let dest = claim(
            check_dest(m.get("dest").unwrap_or(&Value::Null), &what)?,
            &what,
        )?;
        mounts.push((src, dest, true));
    }
    let mut volumes: Vec<(String, String)> = Vec::new();
    for (i, v) in list(spec.get("volumes")).iter().enumerate() {
        let what = format!("volumes[{i}]");
        let vname = match v.get("name").and_then(Value::as_str) {
            Some(n) if VOLUME.is_match(n) => n.to_string(),
            _ => {
                return Err(bad(format!(
                    "{what}: named volumes must be called ec-<name> (lowercase, digits, . _ -)"
                )));
            }
        };
        let dest = claim(
            check_dest(v.get("dest").unwrap_or(&Value::Null), &what)?,
            &what,
        )?;
        volumes.push((vname, dest));
    }
    let fdir = state_dir.join("instances").join(&instance).join("files");
    let mut files: Vec<(String, PathBuf, String)> = Vec::new();
    for (i, (dest, content)) in str_map(spec.get("files"))
        .into_iter()
        .take(MAX_FILES)
        .enumerate()
    {
        let what = format!("files[{i}]");
        let dest = claim(check_dest(&Value::String(dest), &what)?, &what)?;
        let last = dest.rsplit('/').next().unwrap_or("");
        let cleaned: String = last
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || "._-".contains(c) {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        let base = py::take(&cleaned, 100).trim_start_matches('.').to_string();
        let base = if base.is_empty() {
            "file".to_string()
        } else {
            base
        };
        files.push((
            dest,
            fdir.join(format!("{i}-{base}")),
            content.as_str().unwrap_or("").to_string(),
        ));
    }
    let fdir_s = py::path_str(&fdir);
    if fdir_s.contains(':') || fdir_s.contains(',') {
        return Err(bad(
            "the node's state directory contains ':' or ',' and cannot hold mounted files",
        ));
    }

    let shm = match spec.get("shm_size") {
        Some(Value::String(s)) => s.clone(),
        _ if policy.max_shm_gib >= 16.0 => "16g".to_string(),
        _ => format!("{}m", 1.max(py::int_of_float(policy.max_shm_gib * 1024.0))),
    };
    if shm_gib(&shm)? > policy.max_shm_gib {
        return Err(OpError::new(
            POLICY_DENIED,
            format!(
                "shm_size {shm} is above this node's policy limit of {} GiB (max_shm_gib)",
                ec_proto::pyfmt::fmt_g(policy.max_shm_gib)
            ),
        ));
    }
    let ipc_host = spec.get("ipc_host") == Some(&Value::Bool(true));
    if ipc_host && !policy.allow_ipc_host {
        return Err(OpError::new(
            POLICY_DENIED,
            "this node's policy does not allow --ipc host (allow_ipc_host); multi-GPU engines need it for NCCL",
        ));
    }
    let mut ulimits: Vec<(String, String)> = Vec::new();
    for (k, v) in str_map(spec.get("ulimits")) {
        let v = v.as_str().unwrap_or("").to_string();
        if !ULIMITS.contains(&k.as_str()) {
            return Err(bad(format!(
                "ulimit {} is not one an engine may set here (allowed: {})",
                py::repr_take(&k, 40),
                ULIMITS.join(", ")
            )));
        }
        if !ULIMIT_VALUE.is_match(&v) {
            return Err(bad(format!(
                "ulimit {k} value {} must be -1 (unlimited), a number, or soft:hard numbers",
                py::repr_take(&v, 40)
            )));
        }
        ulimits.push((k, v));
    }
    Ok(RunPlan {
        name,
        instance,
        image,
        argv,
        container_port: port,
        network: policy.engine_network.clone(),
        alias,
        gpu_uuids: gpus,
        labels,
        env,
        secret_env,
        mounts,
        volumes,
        files,
        shm_size: shm,
        ipc_host,
        ulimits,
    })
}

/// docker parses --gpus as CSV: several devices need the value wrapped in literal double quotes
pub fn gpus_arg(uuids: &[String]) -> String {
    if uuids.len() > 1 {
        format!("\"device={}\"", uuids.join(","))
    } else {
        format!("device={}", uuids.first().map_or("", String::as_str))
    }
}

/// The exact argv docker.run executes. secret_env appears as bare `-e NAME`: values never reach argv.
pub fn run_argv(d: &Docker, p: &RunPlan) -> Vec<String> {
    let mut a: Vec<String> = sv(&[
        "run",
        "-d",
        "--pull",
        "never",
        "--name",
        &p.name,
        "--network",
        &p.network,
        "--network-alias",
        &p.alias,
        "--restart",
        "no",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges:true",
        "--pids-limit",
        "4096",
    ]);
    if !p.gpu_uuids.is_empty() {
        a.push("--gpus".into());
        a.push(gpus_arg(&p.gpu_uuids));
    }
    a.extend(sv(&[
        "--shm-size",
        &p.shm_size,
        "--ipc",
        if p.ipc_host { "host" } else { "private" },
    ]));
    for (k, v) in &p.ulimits {
        a.push("--ulimit".into());
        a.push(format!("{k}={v}"));
    }
    for (k, v) in &p.labels {
        a.push("--label".into());
        a.push(format!("{k}={v}"));
    }
    for (src, dest, ro) in &p.mounts {
        a.push("-v".into());
        a.push(format!(
            "{}:{dest}{}",
            py::path_str(src),
            if *ro { ":ro" } else { "" }
        ));
    }
    for (vname, dest) in &p.volumes {
        a.push("-v".into());
        a.push(format!("{vname}:{dest}"));
    }
    for (dest, host, _) in &p.files {
        a.push("-v".into());
        a.push(format!("{}:{dest}:ro", py::path_str(host)));
    }
    for (k, v) in &p.env {
        a.push("-e".into());
        a.push(format!("{k}={v}"));
    }
    for (k, _) in &p.secret_env {
        a.push("-e".into());
        a.push(k.clone());
    }
    a.push(p.image.clone());
    a.extend(p.argv.iter().cloned());
    let refs: Vec<&str> = a.iter().map(String::as_str).collect();
    d.argv(&refs)
}

fn random_hex(n: usize) -> String {
    let mut buf = vec![0u8; n];
    if let Ok(mut f) = std::fs::File::open("/dev/urandom") {
        let _ = std::io::Read::read_exact(&mut f, &mut buf);
    }
    hex::encode(buf)
}

/// Write the spec's generated files under <state>/instances/<instance>/files/ (directories 0700, files 0644, each
/// replaced atomically); stale files of earlier runs of that instance are removed.
pub fn write_files(p: &RunPlan, state_dir: &Path) -> Result<(), OpError> {
    let fail = |e: &std::io::Error| {
        OpError::new(
            IO_ERROR,
            format!(
                "cannot write the engine's config files under {}: {}",
                py::path_str(state_dir),
                py::io_strerror(e)
            ),
        )
    };
    let fdir = state_dir.join("instances").join(&p.instance).join("files");
    let inst = state_dir.join("instances").join(&p.instance);
    for d in [
        state_dir.to_path_buf(),
        state_dir.join("instances"),
        inst,
        fdir.clone(),
    ] {
        match std::fs::DirBuilder::new().mode(0o700).create(&d) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists && d.is_dir() => {}
            Err(e) => return Err(fail(&e)),
        }
        let is_link = std::fs::symlink_metadata(&d).is_ok_and(|m| m.file_type().is_symlink());
        if is_link || !d.is_dir() {
            return Err(OpError::new(
                IO_ERROR,
                format!("{} is not a plain directory", py::path_str(&d)),
            ));
        }
        std::fs::set_permissions(&d, std::fs::Permissions::from_mode(0o700))
            .map_err(|e| fail(&e))?;
    }
    let keep: HashSet<std::ffi::OsString> = p
        .files
        .iter()
        .map(|(_, host, _)| py::basename(host))
        .collect();
    for old in std::fs::read_dir(&fdir).map_err(|e| fail(&e))? {
        let old = old.map_err(|e| fail(&e))?;
        if !keep.contains(&old.file_name()) {
            match std::fs::remove_file(old.path()) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(fail(&e)),
            }
        }
    }
    for (_, host, content) in &p.files {
        let mut tmp = host.clone().into_os_string();
        tmp.push(format!(".{}.tmp", random_hex(4)));
        let tmp = PathBuf::from(tmp);
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .custom_flags(nix::libc::O_NOFOLLOW | nix::libc::O_CLOEXEC)
            .mode(0o644)
            .open(&tmp)
            .map_err(|e| fail(&e))?;
        f.write_all(content.as_bytes()).map_err(|e| fail(&e))?;
        drop(f);
        std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o644))
            .map_err(|e| fail(&e))?;
        std::fs::rename(&tmp, host).map_err(|e| fail(&e))?;
    }
    Ok(())
}

/// Remove an instance's generated files (never through a symlinked instance directory).
pub fn remove_files(state_dir: &Path, instance: &str) {
    if !INSTANCE.is_match(instance) {
        return;
    }
    let d = state_dir.join("instances").join(instance);
    match std::fs::symlink_metadata(&d) {
        Ok(m) if m.file_type().is_symlink() => {}
        Ok(_) => {
            let _ = std::fs::remove_dir_all(&d);
        }
        Err(_) => {}
    }
}

// ── network and gateway ──────────────────────────────────────────────────────────────────────────────────
/// Create the engine network with --internal, or verify an existing one really is internal.
pub fn ensure_network(d: &Docker, name: &str) -> Result<(), OpError> {
    let res = d.call(
        &["network", "inspect", "--format", "{{.Internal}}", name],
        15.0,
        None,
    );
    if res.ok() {
        if py::strip(&res.out).to_lowercase() != "true" {
            return Err(OpError::new(
                CONFLICT,
                format!(
                    "docker network {} exists but is not --internal; refusing to attach engines to it (remove it or \
                     name another engine_network in the policy)",
                    py::repr(name)
                ),
            ));
        }
        return Ok(());
    }
    if !missing(&res) {
        return Err(docker_error("network inspect", &res));
    }
    d.ok(
        &[
            "network",
            "create",
            "--internal",
            "--label",
            &format!("{LABEL}=1"),
            name,
        ],
        30.0,
        None,
    )?;
    Ok(())
}

/// nginx.conf for a pure TCP forward. Only a checked name and an int reach the template.
pub fn gateway_config(engine_host: &str, engine_port: i64) -> Result<String, OpError> {
    if !NGINX_HOST.is_match(engine_host) {
        return Err(bad("engine_host is not a safe container name"));
    }
    if !(1..=65535).contains(&engine_port) {
        return Err(bad("engine_port must be 1-65535"));
    }
    // daemon off: nginx forks by default, which ends the container's PID 1 (found by live test, not mocks).
    Ok(format!(
        "daemon off;\nworker_processes 1;\npid /tmp/nginx.pid;\nerror_log /dev/stderr warn;\n\
         events {{ worker_connections 1024; }}\n\
         stream {{\n  server {{\n    listen {GATEWAY_LISTEN};\n    proxy_pass {engine_host}:{engine_port};\n\
         \x20   proxy_connect_timeout 5s;\n    proxy_timeout 1h;\n  }}\n}}\n"
    ))
}

/// `docker create` on the default bridge (so it can publish on loopback); the engine network is attached afterwards
/// with `docker network connect`, since one create accepts only one --network.
pub fn gateway_argv(
    d: &Docker,
    image: &str,
    name: &str,
    instance: &str,
    host_port: i64,
    engine_host: &str,
    engine_port: i64,
) -> Result<Vec<String>, OpError> {
    let conf = gateway_config(engine_host, engine_port)?;
    let publish = format!("127.0.0.1:{host_port}:{GATEWAY_LISTEN}");
    let inst = format!("{INSTANCE_LABEL}={instance}");
    let nginx = format!("NGINX_CONF={conf}");
    Ok(d.argv(&[
        "create",
        "--pull",
        "never",
        "--name",
        name,
        "--network",
        "bridge",
        "--restart",
        "no",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges:true",
        "--pids-limit",
        "256",
        "--user",
        "101:101",
        "--tmpfs",
        "/var/cache/nginx:rw,mode=1777,size=4m",
        "--tmpfs",
        "/var/run:rw,mode=1777,size=1m",
        "--tmpfs",
        "/tmp:rw,mode=1777,size=1m",
        "-p",
        &publish,
        "--label",
        "engine-console=1",
        "--label",
        "engine-console.role=gateway",
        "--label",
        &inst,
        "-e",
        &nginx,
        "--entrypoint",
        "/bin/sh",
        image,
        "-c",
        GATEWAY_SH,
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gateway_config_text_is_pythons() {
        assert_eq!(
            gateway_config("ec-owned-1", 8000).unwrap_or_default(),
            "daemon off;\nworker_processes 1;\npid /tmp/nginx.pid;\nerror_log /dev/stderr warn;\n\
             events { worker_connections 1024; }\nstream {\n  server {\n    listen 8080;\n    proxy_pass \
             ec-owned-1:8000;\n    proxy_connect_timeout 5s;\n    proxy_timeout 1h;\n  }\n}\n"
        );
        assert!(GATEWAY_SH.contains(r#""%s\n""#));
    }

    #[test]
    fn gpus_arg_quotes_several_devices() {
        assert_eq!(gpus_arg(&["a".into()]), "device=a");
        assert_eq!(gpus_arg(&["a".into(), "b".into()]), "\"device=a,b\"");
    }
}
