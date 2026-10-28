//! Python: node/src/ecnode/discovery.py.
//!
//! What runs on this machine: containers (sanitized `docker inspect`), engine classification and endpoints.
//!
//! Everything read from a container or an endpoint is untrusted data: capped, control-character-stripped, never
//! executed or used to build a shell command or a path. A container's argv is REDACTED here, on the node, before it
//! goes anywhere; `Config.Env` is never read at all. Endpoint probes are GETs to 127.0.0.1 only, with short
//! timeouts, a 1 MB cap and no redirects.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};

use ec_proto::Obj;
use regex::Regex;
use serde_json::{Map, Value, json};

use crate::httpproxy::client;
use crate::policy::{Policy, Root, full};
use crate::util::{clean_text, parse_time, py, redact_argv, scrub};

pub const MAX_BODY: usize = 1024 * 1024;
pub const MAX_CMDLINE: usize = 8192;
pub const MAX_PARAMS: usize = 60;
pub const PROBE_CONNECT_S: f64 = 1.0;
pub const PROBE_READ_S: f64 = 2.0;
pub const PROBE_TOTAL_S: f64 = 5.0;

/// Engines whose /metrics answers slowly and whose every read drains the engine's statistics queue (TensorRT-LLM's
/// iteration stats): discovery never probes their /metrics, so only the metrics scraper reads it.
pub const SLOW_METRICS: &[&str] = &["trtllm"];
pub const SLOW_METRICS_READ_S: f64 = 10.0;
pub const SLOW_METRICS_TOTAL_S: f64 = 15.0;

pub const LABEL: &str = "engine-console";
pub const ROLE_LABEL: &str = "engine-console.role";
pub const INSTANCE_LABEL: &str = "engine-console.instance";
pub const ENGINE_LABEL: &str = "engine-console.engine";

// ── data-driven engine signatures ──────────────────────────────────────────────────────────────────────────
/// One engine: regexes on the lower-cased image reference and on the (redacted) command line, default ports.
pub struct Signature {
    pub engine: &'static str,
    pub image: &'static [&'static str],
    pub cmd: &'static [&'static str],
    pub ports: &'static [u16],
}

pub const SIGNATURES: &[Signature] = &[
    Signature {
        engine: "vllm",
        image: &[r"(^|/)vllm([-/:@]|$)", r"(^|/)vllm-openai"],
        cmd: &[r"\bvllm\s+serve\b", r"vllm\.entrypoints"],
        ports: &[8000],
    },
    Signature {
        engine: "sglang",
        image: &[r"(^|/)sglang([-/:@]|$)"],
        cmd: &[r"sglang\.launch_server", r"\bsglang\s+serve\b"],
        ports: &[30000],
    },
    Signature {
        engine: "ollama",
        image: &[r"(^|/)ollama([-/:@]|$)"],
        cmd: &[r"\bollama\s+serve\b"],
        ports: &[11434],
    },
    Signature {
        engine: "llamacpp",
        image: &[r"llama[.-]?cpp", r"(^|/)llama-server"],
        cmd: &[r"(^|/|\s)llama-server\b"],
        ports: &[8080],
    },
    Signature {
        engine: "trtllm",
        image: &[r"tensorrt[-_]?llm", r"trtllm"],
        cmd: &[r"\btrtllm-serve\b", r"tensorrt_llm"],
        ports: &[8000],
    },
    Signature {
        engine: "dynamo",
        image: &[r"(^|/)dynamo([-/:@]|$)", r"ai-dynamo"],
        cmd: &[r"dynamo\.frontend", r"\bdynamo-run\b"],
        ports: &[8000],
    },
];

fn re(p: &str) -> Regex {
    Regex::new(p).unwrap_or_else(|_| std::process::abort())
}

static SIG_RES: LazyLock<Vec<(Vec<Regex>, Vec<Regex>)>> = LazyLock::new(|| {
    SIGNATURES
        .iter()
        .map(|s| {
            (
                s.image.iter().map(|p| re(p)).collect(),
                s.cmd.iter().map(|p| re(p)).collect(),
            )
        })
        .collect()
});

/// Images that are never engines (routers, UIs, proxies, databases, telemetry), checked BEFORE the signatures.
static NON_ENGINE_IMAGE: LazyLock<Regex> = LazyLock::new(|| {
    re(
        r"(?i)litellm|open-?webui|mcpo|mitmproxy|grafana|prometheus|loki|promtail|otel|opentelemetry|langfuse|phoenix|clickhouse|postgres|redis|searxng|nginx|traefik|caddy|haproxy|envoy|router|relay|switchyard|gateway|cadvisor|exporter|minio|mysql|mongo|rabbitmq|kafka|elasticsearch|jaeger|tempo|nats|etcd",
    )
});

pub const METRIC_MARKERS: &[(&str, &str)] = &[
    ("vllm:", "vllm"),
    ("sglang:", "sglang"),
    ("llamacpp:", "llamacpp"),
    ("trtllm", "trtllm"),
    ("tensorrt_llm", "trtllm"),
    ("dynamo_", "dynamo"),
];

static ENGINE_ID: LazyLock<Regex> = LazyLock::new(|| full("[a-z0-9_]{1,32}"));
static PARAM_KEY: LazyLock<Regex> = LazyLock::new(|| re("[^a-z0-9_]"));

/// Engine id of a container by image/command signature; None if unknown or explicitly not an engine.
pub fn classify(image: &str, cmdline: &str) -> Option<String> {
    let lower = image.to_lowercase();
    let img = py::take(&lower, 300);
    let cmd = py::take(cmdline, MAX_CMDLINE);
    if NON_ENGINE_IMAGE.is_match(img) {
        return None;
    }
    for (sig, (img_res, cmd_res)) in SIGNATURES.iter().zip(SIG_RES.iter()) {
        if img_res.iter().any(|r| r.is_match(img)) || cmd_res.iter().any(|r| r.is_match(cmd)) {
            return Some(sig.engine.to_string());
        }
    }
    None
}

pub fn default_ports(engine: Option<&str>) -> Vec<u16> {
    SIGNATURES
        .iter()
        .find(|s| Some(s.engine) == engine)
        .map(|s| s.ports.to_vec())
        .unwrap_or_default()
}

/// Best-effort `--flag value` parse of an ALREADY redacted argv. Display only; never used to launch.
pub fn parse_params(argv: &[String]) -> Obj {
    let mut params = Obj::new();
    let mut i = 0;
    while i < argv.len() && params.len() < MAX_PARAMS {
        let tok = &argv[i];
        if tok == "serve" && i + 1 < argv.len() && !argv[i + 1].starts_with('-') {
            params.insert("model".into(), json!(clean_text(&argv[i + 1], 200)));
            i += 2;
            continue;
        }
        if tok.starts_with("--") && py::len(tok) > 2 {
            let rest = &tok[2..];
            let (name, mut eq, mut val) = match rest.split_once('=') {
                Some((n, v)) => (n, true, v.to_string()),
                None => (rest, false, String::new()),
            };
            let lowered = name.to_lowercase();
            let replaced = PARAM_KEY.replace_all(&lowered, "_");
            let key = py::take(&replaced, 64).to_string();
            if !eq && i + 1 < argv.len() && !argv[i + 1].starts_with("--") {
                val.clone_from(&argv[i + 1]);
                i += 1;
                eq = true;
            }
            if !key.is_empty() {
                let v = if eq {
                    clean_text(&val, 200)
                } else {
                    "true".to_string()
                };
                params.insert(key, json!(v));
            }
        }
        i += 1;
    }
    params
}

// ── one container ──────────────────────────────────────────────────────────────────────────────────────────
/// A sanitized container plus the node-internal facts the hub never sees.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct View {
    /// proto CONTAINER
    pub info: Obj,
    pub pid: i64,
    pub running: bool,
    pub host_network: bool,
    /// host ports reachable on 127.0.0.1
    pub loopback_ports: Vec<u16>,
    /// container port -> host port (first binding wins), in docker's order
    pub published: Vec<(i64, u16)>,
}

impl View {
    /// `published.get(cport)`
    pub fn published_port(&self, cport: i64) -> Option<u16> {
        self.published
            .iter()
            .find(|(c, _)| *c == cport)
            .map(|(_, h)| *h)
    }

    fn s(&self, key: &str) -> Option<&str> {
        self.info.get(key).and_then(Value::as_str)
    }

    fn managed(&self) -> bool {
        self.info.get("managed") == Some(&Value::Bool(true))
    }
}

/// (root id, rel) of a host path that lies inside a policy model root ("" = the root itself).
fn map_root(source: &str, roots: &[(Root, PathBuf)]) -> (Option<String>, Option<String>) {
    let src = py::normpath_str(source);
    let mut best: (Option<String>, Option<String>, i64) = (None, None, -1);
    for (root, real) in roots {
        let mut bases = vec![py::normpath_str(&root.path)];
        let real = py::path_str(real);
        if !bases.contains(&real) {
            bases.push(real);
        }
        for base in bases {
            let prefix = format!("{}/", base.trim_end_matches('/'));
            let n = i64::try_from(py::len(&base)).unwrap_or(i64::MAX);
            if (src == base || src.starts_with(&prefix)) && n > best.2 {
                let rel = src.get(base.len()..).unwrap_or("").trim_start_matches('/');
                best = (Some(root.id.clone()), Some(rel.to_string()), n);
            }
        }
    }
    (best.0, best.1)
}

fn obj_of(v: Option<&Value>) -> Option<&Map<String, Value>> {
    match v {
        Some(Value::Object(m)) => Some(m),
        _ => None,
    }
}

/// `x or ""` then str(): the text of a value that is truthy, else "".
fn text_or_empty(v: Option<&Value>) -> String {
    match v {
        Some(v) if py::truthy(v) => py::str_of(v),
        _ => String::new(),
    }
}

/// `int(f)` of a finite float (saturating outside i64: such a value is no port anyway).
#[allow(clippy::cast_possible_truncation)]
fn trunc(f: f64) -> i64 {
    f.trunc() as i64
}

/// `int(x or "")`, None where Python raises TypeError/ValueError.
fn int_or_none(v: Option<&Value>) -> Option<i64> {
    let v = v.filter(|v| py::truthy(v))?;
    match v {
        Value::String(s) => py::parse_int(s),
        Value::Bool(true) => Some(1),
        Value::Number(n) if n.is_f64() => n.as_f64().filter(|f| f.is_finite()).map(trunc),
        Value::Number(n) => n.as_i64(),
        _ => None,
    }
}

fn gpus_of(doc: &Value, gpu_by_index: &HashMap<String, String>) -> (Vec<String>, bool) {
    let mut uuids: Vec<String> = Vec::new();
    let mut all_gpus = false;
    let reqs = obj_of(doc.get("HostConfig"))
        .and_then(|hc| hc.get("DeviceRequests"))
        .and_then(Value::as_array);
    for req in reqs.map(Vec::as_slice).unwrap_or_default() {
        let Value::Object(req) = req else { continue };
        let ids = req.get("DeviceIDs").filter(|v| py::truthy(v));
        let count_is_minus_one = req.get("Count").and_then(Value::as_f64) == Some(-1.0)
            && !matches!(req.get("Count"), Some(Value::Bool(_)));
        if count_is_minus_one && ids.is_none() {
            all_gpus = true;
        }
        for dev in ids
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or_default()
        {
            let s = py::str_of(dev);
            let mut dev = py::strip(&s);
            if let Some(rest) = dev.strip_prefix("nvidia.com/gpu=") {
                dev = rest;
            }
            if dev == "all" {
                all_gpus = true;
            } else if let Some(u) = gpu_by_index.get(dev) {
                uuids.push(u.clone());
            } else if dev.starts_with("GPU-") || dev.starts_with("MIG-") {
                uuids.push(py::take(dev, 96).to_string());
            }
        }
    }
    let mut seen = HashSet::new();
    uuids.retain(|u| seen.insert(u.clone()));
    uuids.truncate(16);
    (uuids, all_gpus)
}

/// proto CONTAINER from an inspect document. Reads NO environment values; argv is redacted first.
pub fn container_view(
    doc: &Value,
    roots: &[(Root, PathBuf)],
    gpu_by_index: &HashMap<String, String>,
    status: Option<&str>,
) -> Option<View> {
    let empty = Map::new();
    let cfg = obj_of(doc.get("Config")).unwrap_or(&empty);
    let name_raw = doc.get("Name").map_or_else(String::new, py::str_of);
    let name = clean_text(name_raw.trim_start_matches('/'), 128);
    if name.is_empty() {
        return None;
    }
    let mut raw: Vec<String> = Vec::new();
    if let Some(Value::String(p)) = doc.get("Path") {
        raw.push(p.clone());
    }
    if let Some(Value::Array(a)) = doc.get("Args") {
        raw.extend(a.iter().map(py::str_of));
    }
    if raw.is_empty() {
        for k in ["Entrypoint", "Cmd"] {
            match cfg.get(k) {
                Some(Value::Array(a)) => raw.extend(a.iter().map(py::str_of)),
                Some(Value::String(s)) => raw.push(s.clone()),
                _ => {}
            }
        }
    }
    let argv = redact_argv(&raw, 400, 1024);
    let mut labels = Obj::new();
    if let Some(Value::Object(raw_labels)) = cfg.get("Labels") {
        for (k, v) in raw_labels.iter().take(48) {
            labels.insert(
                clean_text(k, 256),
                json!(clean_text(&scrub(&py::str_of(v)), 256)),
            );
        }
    }
    let label = |k: &str| labels.get(k).and_then(Value::as_str).map(str::to_string);
    let st = obj_of(doc.get("State")).unwrap_or(&empty);
    let hc = obj_of(doc.get("HostConfig")).unwrap_or(&empty);
    let net_mode = clean_text(&text_or_empty(hc.get("NetworkMode")), 64);
    let image = clean_text(&text_or_empty(cfg.get("Image")), 256);
    let managed = label(LABEL).as_deref() == Some("1");
    let role = if managed { label(ROLE_LABEL) } else { None };
    let engine_label = label(ENGINE_LABEL).unwrap_or_default();
    let engine =
        if managed && role.as_deref() == Some("engine") && ENGINE_ID.is_match(&engine_label) {
            Some(engine_label)
        } else if role.as_deref() != Some("gateway") {
            classify(&image, &argv.join(" "))
        } else {
            None
        };

    let mut ports: Vec<Value> = Vec::new();
    let mut loopback: Vec<u16> = Vec::new();
    let mut published: Vec<(i64, u16)> = Vec::new();
    let mut bindings = obj_of(doc.get("NetworkSettings")).and_then(|ns| ns.get("Ports"));
    if !bindings.is_some_and(py::truthy) {
        bindings = hc.get("PortBindings");
    }
    if let Some(Value::Object(bindings)) = bindings {
        for (cport, binds) in bindings {
            let Some(cp) = py::parse_int(cport.split('/').next().unwrap_or("")) else {
                continue;
            };
            let binds = match binds {
                Value::Array(a) => a.as_slice(),
                _ => &[],
            };
            for b in binds {
                let Value::Object(b) = b else { continue };
                let Some(hp) = int_or_none(b.get("HostPort")) else {
                    continue;
                };
                let hip = clean_text(&text_or_empty(b.get("HostIp")), 64);
                if !(1..=65535).contains(&hp) || ports.len() >= 20 {
                    continue;
                }
                let hp16 = u16::try_from(hp).unwrap_or(0);
                ports.push(json!({"host_ip": hip, "host_port": hp, "container_port": cp}));
                if !published.iter().any(|(c, _)| *c == cp) {
                    published.push((cp, hp16));
                }
                if ["", "0.0.0.0", "127.0.0.1", "::", "::1"].contains(&hip.as_str())
                    && !loopback.contains(&hp16)
                {
                    loopback.push(hp16);
                }
            }
        }
    }
    let mut mounts: Vec<Value> = Vec::new();
    if let Some(Value::Array(ms)) = doc.get("Mounts") {
        for m in ms.iter().take(32) {
            let Value::Object(m) = m else { continue };
            let src = text_or_empty(m.get("Source"));
            let (rid, rel) = if src.starts_with('/') {
                map_root(&src, roots)
            } else {
                (None, None)
            };
            let ro = m.get("RW").is_some_and(|v| !py::truthy(v));
            mounts.push(json!({"source": clean_text(&src, 512),
                               "dest": clean_text(&text_or_empty(m.get("Destination")), 512),
                               "ro": ro, "root": rid, "rel": rel}));
        }
    }
    let (gpus, all_gpus) = gpus_of(doc, gpu_by_index);
    let pid = match st.get("Pid") {
        Some(Value::Number(n)) if !n.is_f64() => n.as_i64().filter(|p| *p > 0).unwrap_or(0),
        _ => 0,
    };
    let state = clean_text(&text_or_empty(st.get("Status")), 16);
    let exit_code = st.get("ExitCode");
    let status_text = match status.filter(|s| !s.is_empty()) {
        Some(s) => s.to_string(),
        None if state == "exited" => format!(
            "{state} (exit {})",
            py::str_of(exit_code.unwrap_or(&Value::Null))
        ),
        None => state.clone(),
    };
    let exit_int = match exit_code {
        Some(Value::Number(n)) if !n.is_f64() => Some(Value::Number(n.clone())),
        _ => None,
    };
    let mut info = Obj::new();
    info.insert("name".into(), json!(name));
    info.insert(
        "id".into(),
        json!(clean_text(&text_or_empty(doc.get("Id")), 64)),
    );
    info.insert("image".into(), json!(image));
    info.insert(
        "image_id".into(),
        json!(clean_text(&text_or_empty(doc.get("Image")), 80)),
    );
    info.insert("state".into(), json!(state));
    info.insert("status".into(), json!(clean_text(&status_text, 128)));
    info.insert(
        "created".into(),
        json!(parse_time(doc.get("Created").unwrap_or(&Value::Null))),
    );
    info.insert(
        "started_at".into(),
        json!(parse_time(st.get("StartedAt").unwrap_or(&Value::Null))),
    );
    info.insert("pid".into(), if pid > 0 { json!(pid) } else { Value::Null });
    info.insert("exit_code".into(), exit_int.unwrap_or(Value::Null));
    let params = parse_params(&argv);
    let instance = if managed { label(INSTANCE_LABEL) } else { None };
    let (project, service) = (
        label("com.docker.compose.project"),
        label("com.docker.compose.service"),
    );
    info.insert("labels".into(), Value::Object(labels));
    info.insert("engine".into(), json!(engine));
    info.insert("managed".into(), json!(managed));
    info.insert("role".into(), json!(role));
    info.insert("instance".into(), json!(instance));
    info.insert("ports".into(), Value::Array(ports));
    info.insert("argv".into(), json!(argv));
    info.insert("params".into(), Value::Object(params));
    info.insert("gpus".into(), json!(gpus));
    info.insert("all_gpus".into(), json!(all_gpus));
    info.insert("mounts".into(), Value::Array(mounts));
    info.insert("network_mode".into(), json!(net_mode));
    info.insert("compose_project".into(), json!(project));
    info.insert("compose_service".into(), json!(service));
    Some(View {
        info,
        pid,
        running: st.get("Running") == Some(&Value::Bool(true)),
        host_network: net_mode == "host",
        loopback_ports: loopback,
        published,
    })
}

// ── probing ──────────────────────────────────────────────────────────────────────────────────────────────
/// One GET's outcome (status None: no answer).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Probe {
    pub status: Option<i64>,
    pub body: Vec<u8>,
}

impl Probe {
    pub fn ok(&self) -> bool {
        self.status.is_some_and(|s| (200..300).contains(&s))
    }

    pub fn auth(&self) -> bool {
        matches!(self.status, Some(401 | 403))
    }

    pub fn answered(&self) -> bool {
        self.ok() || self.auth()
    }
}

/// http_get's bounds (discovery.py keyword arguments).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GetOpts {
    pub max_body: usize,
    pub connect_s: f64,
    pub read_s: f64,
    pub total_s: f64,
}

impl Default for GetOpts {
    fn default() -> Self {
        Self {
            max_body: MAX_BODY,
            connect_s: PROBE_CONNECT_S,
            read_s: PROBE_READ_S,
            total_s: PROBE_TOTAL_S,
        }
    }
}

/// GET http://127.0.0.1:<port><path>: loopback only, no redirects, body capped, bounded in time.
pub fn http_get(port: u16, path: &str, o: &GetOpts) -> Probe {
    let deadline = Instant::now() + Duration::from_secs_f64(o.total_s.max(0.0));
    // http.client refuses a URL with spaces or control characters (InvalidURL, a ValueError)
    if path.chars().any(|c| u32::from(c) <= 0x20 || c == '\u{7f}') {
        return Probe::default();
    }
    let attempt = || -> Result<Probe, client::HttpError> {
        let mut conn = client::Conn::connect(port, o.connect_s)?;
        conn.set_timeout(o.read_s)?;
        let host = if port == 80 {
            "127.0.0.1".to_string()
        } else {
            format!("127.0.0.1:{port}")
        };
        let head = vec![
            ("Host".to_string(), host),
            ("Accept-Encoding".to_string(), "identity".to_string()),
            ("Accept".to_string(), "*/*".to_string()),
            (
                "User-Agent".to_string(),
                format!("ecnode/{}", crate::agent::VERSION),
            ),
        ];
        conn.send(&client::request_head("GET", path, &head))?;
        let mut r = conn.get_response(false)?;
        let mut body: Vec<u8> = Vec::new();
        while body.len() < o.max_body && Instant::now() < deadline {
            let chunk = r.read1(65536.min(o.max_body - body.len()))?;
            if chunk.is_empty() {
                break;
            }
            body.extend_from_slice(&chunk);
        }
        body.truncate(o.max_body);
        Ok(Probe {
            status: Some(r.status),
            body,
        })
    };
    attempt().unwrap_or_default()
}

/// `json.loads(body)` (a UTF-8 BOM is allowed, as Python's byte decoding allows it).
fn loads(body: &[u8]) -> Option<Value> {
    let b = body.strip_prefix(b"\xef\xbb\xbf").unwrap_or(body);
    serde_json::from_slice(b).ok()
}

fn head_text(body: &[u8]) -> String {
    String::from_utf8_lossy(&body[..body.len().min(65536)]).into_owned()
}

/// The three GETs of one port.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PortProbe {
    pub port: u16,
    pub models: Probe,
    pub health: Probe,
    pub metrics: Probe,
}

impl PortProbe {
    pub fn any_answer(&self) -> bool {
        self.models.answered() || self.health.answered() || self.metrics.answered()
    }

    pub fn model_names(&self) -> Vec<String> {
        let Some(Value::Object(doc)) = loads(&self.models.body) else {
            return Vec::new();
        };
        let Some(Value::Array(data)) = doc.get("data") else {
            return Vec::new();
        };
        data.iter()
            .take(20)
            .filter_map(|m| match m.get("id") {
                Some(Value::String(id)) if m.is_object() => Some(clean_text(id, 200)),
                _ => None,
            })
            .collect()
    }

    pub fn hint(&self) -> Option<String> {
        let text = head_text(&self.metrics.body);
        for (marker, engine) in METRIC_MARKERS {
            if text.contains(marker) {
                return Some((*engine).to_string());
            }
        }
        let data = match loads(&self.models.body) {
            Some(Value::Object(doc)) => doc.get("data").filter(|d| py::truthy(d)).cloned(),
            _ => None,
        };
        if let Some(Value::Array(data)) = data
            && data
                .iter()
                .take(20)
                .any(|m| m.get("owned_by") == Some(&json!("library")))
        {
            return Some("ollama".into());
        }
        (self.port == 11434 && self.models.ok()).then(|| "ollama".to_string())
    }

    /// /v1/models answers like an OpenAI-style server: a JSON object with a `data` list, or 401/403 from an engine
    /// with an API key (counted only when /metrics names a known engine).
    pub fn models_listing(&self) -> bool {
        if self.models.auth() {
            let text = if self.metrics.ok() {
                head_text(&self.metrics.body)
            } else {
                String::new()
            };
            return METRIC_MARKERS.iter().any(|(m, _)| text.contains(m));
        }
        if !self.models.ok() {
            return false;
        }
        matches!(loads(&self.models.body), Some(Value::Object(d)) if matches!(d.get("data"), Some(Value::Array(_))))
    }

    /// A bare endpoint counts as an engine only if it looks like one: a /v1/models listing AND (/metrics or
    /// /health).
    pub fn qualifies(&self) -> bool {
        self.models_listing() && (self.metrics.answered() || self.health.answered())
    }

    /// Whether a bare (non-container) port becomes an endpoint: always a real /v1/models listing (the `http.request`
    /// row of docs/FLEET-PROTOCOL.md, docs/SECURITY.md), plus either `qualifies()` or an engine hint (an Ollama on
    /// 11434 has neither /health nor /metrics). Python also took a hint ALONE, so any page whose /metrics merely
    /// mentions "vllm:" or "dynamo_" became an `http.request` target (security audit F3): the hint now only names the
    /// engine.
    pub fn bare_engine(&self) -> bool {
        self.models_listing() && (self.qualifies() || self.hint().is_some())
    }
}

pub type Prober = Arc<dyn Fn(u16) -> PortProbe + Send + Sync>;
pub type MetricsGet = Arc<dyn Fn(u16, &str, &GetOpts) -> Probe + Send + Sync>;

/// `probe_port(port, get)`
pub fn probe_port_with(port: u16, get: &dyn Fn(u16, &str) -> Probe) -> PortProbe {
    PortProbe {
        port,
        models: get(port, "/v1/models"),
        health: get(port, "/health"),
        metrics: get(port, "/metrics"),
    }
}

pub fn probe_port(port: u16) -> PortProbe {
    probe_port_with(port, &|p, path| http_get(p, path, &GetOpts::default()))
}

/// probe_port without the /metrics GET, for engines in SLOW_METRICS.
pub fn probe_port_quiet(port: u16) -> PortProbe {
    let get = |p: u16, path: &str| http_get(p, path, &GetOpts::default());
    PortProbe {
        port,
        models: get(port, "/v1/models"),
        health: get(port, "/health"),
        metrics: Probe::default(),
    }
}

fn endpoint(p: &PortProbe, container: Option<&str>, engine: Option<&str>) -> Obj {
    let mut o = Obj::new();
    o.insert("port".into(), json!(p.port));
    o.insert("container".into(), json!(container));
    o.insert("engine".into(), json!(engine));
    o.insert("models".into(), json!(p.model_names()));
    o.insert("health_status".into(), json!(p.health.status));
    o.insert("models_status".into(), json!(p.models.status));
    o.insert("metrics_status".into(), json!(p.metrics.status));
    o.insert(
        "auth_required".into(),
        json!(p.models.auth() || p.health.auth()),
    );
    o
}

struct Plan {
    container: Option<String>,
    engine: Option<String>,
    ports: Vec<u16>,
    bare: bool,
}

/// Endpoints of running engine containers, of console gateways and of bare loopback ports.
pub fn build_endpoints(
    views: &[View],
    discovery_ports: &[u16],
    prober: &Prober,
    workers: usize,
    quiet_prober: &Prober,
) -> Vec<Obj> {
    build_endpoints_with(views, discovery_ports, prober, workers, quiet_prober, false)
}

/// build_endpoints; `published_only`: probe only ports a container publishes (security audit F1, for a node without
/// a usable policy): a host-network engine's ports are guessed from its argv or the engine's defaults, so they are
/// left out (the container is still listed, without an endpoint).
pub fn build_endpoints_with(
    views: &[View],
    discovery_ports: &[u16],
    prober: &Prober,
    workers: usize,
    quiet_prober: &Prober,
    published_only: bool,
) -> Vec<Obj> {
    let running: Vec<&View> = views.iter().filter(|v| v.running).collect();
    let claimed: HashSet<u16> = running
        .iter()
        .flat_map(|v| v.loopback_ports.iter().copied())
        .collect();
    let mut plans: Vec<Plan> = Vec::new();
    let mut engines_by_instance: HashMap<String, &View> = HashMap::new();
    for v in &running {
        if v.managed()
            && v.s("role") == Some("engine")
            && let Some(inst) = v.s("instance").filter(|i| !i.is_empty())
        {
            engines_by_instance.insert(inst.to_string(), v);
        }
    }
    for v in &running {
        if v.managed() && v.s("role") == Some("gateway") {
            let eng = v
                .s("instance")
                .and_then(|i| engines_by_instance.get(i))
                .copied();
            let port = v
                .published_port(8080)
                .filter(|p| *p != 0)
                .or_else(|| v.loopback_ports.first().copied());
            if let Some(port) = port.filter(|p| *p != 0) {
                plans.push(Plan {
                    container: eng.unwrap_or(v).s("name").map(str::to_string),
                    engine: eng.and_then(|e| e.s("engine")).map(str::to_string),
                    ports: vec![port],
                    bare: false,
                });
            }
            continue;
        }
        let Some(engine) = v.s("engine").filter(|e| !e.is_empty()) else {
            continue;
        };
        let mut ports = v.loopback_ports.clone();
        if v.host_network && ports.is_empty() && !published_only {
            let argv: Vec<String> = v
                .info
                .get("argv")
                .and_then(Value::as_array)
                .map(|a| a.iter().map(py::str_of).collect())
                .unwrap_or_default();
            for (i, tok) in argv.iter().enumerate().take(argv.len().saturating_sub(1)) {
                let next = &argv[i + 1];
                if (tok == "--port" || tok == "-p")
                    && py::isdigit(next)
                    && let Ok(p) = next.parse::<u32>()
                    && (1..=65535).contains(&p)
                {
                    ports.push(u16::try_from(p).unwrap_or(0));
                }
            }
            if ports.is_empty() {
                ports = default_ports(Some(engine));
            }
        }
        if !ports.is_empty() {
            ports.truncate(5);
            plans.push(Plan {
                container: v.s("name").map(str::to_string),
                engine: Some(engine.to_string()),
                ports,
                bare: false,
            });
        }
    }
    let taken: HashSet<u16> = plans.iter().flat_map(|p| p.ports.iter().copied()).collect();
    let mut seen = HashSet::new();
    for p in discovery_ports {
        if seen.insert(*p) && !claimed.contains(p) && !taken.contains(p) && *p >= 1 {
            plans.push(Plan {
                container: None,
                engine: None,
                ports: vec![*p],
                bare: true,
            });
        }
    }
    let mut all_ports: Vec<u16> = Vec::new();
    for p in plans.iter().flat_map(|p| p.ports.iter().copied()) {
        if !all_ports.contains(&p) {
            all_ports.push(p);
        }
    }
    let quiet: HashSet<u16> = plans
        .iter()
        .filter(|p| {
            p.engine
                .as_deref()
                .is_some_and(|e| SLOW_METRICS.contains(&e))
        })
        .flat_map(|p| p.ports.iter().copied())
        .collect();
    let probes = probe_all(&all_ports, &quiet, prober, quiet_prober, workers);
    let mut out = Vec::new();
    for plan in &plans {
        let candidates: Vec<&PortProbe> = plan.ports.iter().filter_map(|p| probes.get(p)).collect();
        if plan.bare {
            let Some(pr) = candidates.first() else {
                continue;
            };
            if pr.any_answer() && pr.bare_engine() {
                let engine = pr.hint().unwrap_or_else(|| "openai_compatible".into());
                out.push(endpoint(pr, None, Some(&engine)));
            }
            continue;
        }
        // prefer a port that reports healthy, then any that answers at all, then the first one
        let key = |p: &PortProbe| (!(p.models.ok() || p.health.ok()), !p.any_answer());
        let mut best: Option<&PortProbe> = None;
        for c in &candidates {
            if best.is_none_or(|b| key(c) < key(b)) {
                best = Some(c);
            }
        }
        let Some(best) = best else { continue };
        let engine = plan
            .engine
            .clone()
            .filter(|e| !e.is_empty())
            .or_else(|| best.hint());
        out.push(endpoint(best, plan.container.as_deref(), engine.as_deref()));
    }
    out.truncate(64);
    out
}

fn probe_all(
    ports: &[u16],
    quiet: &HashSet<u16>,
    prober: &Prober,
    quiet_prober: &Prober,
    workers: usize,
) -> HashMap<u16, PortProbe> {
    let n = ports.len();
    let workers = workers.min(n.max(1)).max(1);
    let next = AtomicUsize::new(0);
    let results: Mutex<HashMap<u16, PortProbe>> = Mutex::new(HashMap::new());
    std::thread::scope(|s| {
        for _ in 0..workers {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::SeqCst);
                    if i >= n {
                        return;
                    }
                    let p = ports[i];
                    // a probe that panics costs that port's answer, not the inventory round
                    let pr = crate::util::panics::contain("probe", || {
                        if quiet.contains(&p) {
                            quiet_prober(p)
                        } else {
                            prober(p)
                        }
                    });
                    if let Some(pr) = pr {
                        crate::util::lock(&results).insert(p, pr);
                    }
                }
            });
        }
    });
    crate::util::sync::into_inner(results)
}

/// `up` when a running container matches one of a script's patterns, else `down` (§6.3).
pub fn script_status(policy: &Policy, running_names: &[String], now: f64) -> Vec<Obj> {
    let mut out = Vec::new();
    for s in &policy.scripts {
        let mut o = Obj::new();
        o.insert("name".into(), json!(s.name));
        if s.containers.is_empty() {
            o.insert("status".into(), json!("unknown"));
            o.insert("checked_at".into(), json!(now));
            o.insert(
                "detail".into(),
                json!("this script lists no container patterns; use its status action"),
            );
            out.push(o);
            continue;
        }
        let hits: Vec<&String> = running_names.iter().filter(|n| s.matches(n)).collect();
        let detail = if hits.is_empty() {
            format!(
                "no running container matches {}",
                s.containers
                    .iter()
                    .take(4)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        } else {
            format!(
                "running: {}",
                hits.iter()
                    .take(5)
                    .map(|h| h.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        o.insert(
            "status".into(),
            json!(if hits.is_empty() { "down" } else { "up" }),
        );
        o.insert("checked_at".into(), json!(now));
        o.insert("detail".into(), json!(clean_text(&detail, 500)));
        out.push(o);
    }
    out
}
