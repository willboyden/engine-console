//! Python: backend/src/engine_console/services/externals.py.
//!
//! Engines the console did not create, read from each host's pushed inventory (contract v2, §10.4). Two classes,
//! both derived (never stored):
//!   * `script` - containers matched by a script registered in the node's policy become ONE instance per script
//!     (id `scr-<host>-<script>`, each '-' of the host doubled); start/stop/restart map to that script's `up`/`down`.
//!   * `external` - any other running engine container (id `ext-<host>-<slug>`, same rule), monitor-only unless the
//!     node policy says `external_control: "stop_start"`.
//!
//! Pure functions over a `NodeClient`'s latest messages: no I/O, no clocks (the caller passes `now`).

use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

use regex::Regex;
use serde_json::Value;
use sha1::Digest as _;

use crate::domain::errors::Obj;
use crate::domain::fleet::NodeClient;
use crate::domain::models::{HostMemory, Instance};
use crate::domain::refs::{InstanceControls, InstanceVram, ModelLocation};
use crate::services::common::{fnmatchcase, scrub};
use crate::services::discovery::{EXT, SCRIPT, clean_text, clean_text200, display_name, slug};
use crate::services::hostmem::{container_memory, sum_memory};
use crate::services::nodeview as nv;

static SCRIPT_NAME: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,31}$").unwrap_or_else(|_| unreachable!())
});
pub const GONE_CAP: usize = 64;

/// host id -> (label, addresses) of every configured host.
pub type Others = Vec<(String, (String, Vec<String>))>;

/// What the builders need to know about one host, plus how to recognise the other hosts (shared storage).
#[derive(Clone)]
pub struct HostCtx {
    pub host_id: String,
    pub label: String,
    pub node: Arc<dyn NodeClient>,
    pub others: Others,
}

impl std::fmt::Debug for HostCtx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostCtx")
            .field("host_id", &self.host_id)
            .finish_non_exhaustive()
    }
}

fn truthy(v: Option<&Value>) -> bool {
    v.is_some_and(ec_proto::sanitize::truthy)
}

fn s_of(v: Option<&Value>) -> String {
    v.map_or_else(|| "None".to_string(), ec_proto::pyfmt::str_value)
}

/// A str field, or None (what `x.get(k)` gives a pydantic `str | None` field after the sanitizer).
fn opt_str(v: Option<&Value>) -> Option<String> {
    match v {
        Some(Value::String(s)) => Some(s.clone()),
        Some(Value::Null) | None => None,
        Some(other) => Some(ec_proto::pyfmt::str_value(other)),
    }
}

/// Host ids may hold '-', and so may the slug after them, so `ext-gpu-2-vllm` could be host `gpu` running `2-vllm`
/// or host `gpu-2` running `vllm`; an action on one row would reach the other host (security review M1). Every '-'
/// of the host id is doubled. A slug never holds '--' and never starts with '-', so the first single '-' after the
/// host part is the separator and two hosts can never produce the same id.
fn host_part(host_id: &str) -> String {
    host_id.replace('-', "--")
}

pub fn script_instance_id(host_id: &str, name: &str) -> String {
    format!("{SCRIPT}{}-{}", host_part(host_id), slug(name))
}

pub fn external_instance_id(host_id: &str, key: &str) -> String {
    format!("{EXT}{}-{}", host_part(host_id), slug(key))
}

/// `nv.uuid_index` as Python's dict (a repeated uuid keeps its first position, the last index wins).
pub(crate) fn uuid_map(node: &dyn NodeClient) -> Vec<(String, i64)> {
    let mut out: Vec<(String, i64)> = Vec::new();
    for (u, i) in nv::uuid_index(node) {
        if let Some(slot) = out.iter_mut().find(|(k, _)| *k == u) {
            slot.1 = i;
        } else {
            out.push((u, i));
        }
    }
    out
}

fn idx_of(idx: &[(String, i64)], uuid: &str) -> Option<i64> {
    idx.iter().find(|(u, _)| u == uuid).map(|(_, i)| *i)
}

// ---- measurements -------------------------------------------------------------------------------------------------
/// Per GPU, the VRAM the driver attributes to processes of these containers. None where the driver cannot say.
pub fn vram_for(node: &dyn NodeClient, names: &[String]) -> Vec<InstanceVram> {
    let idx = uuid_map(node);
    let sample = nv::sample(node);
    let Some(Value::Object(live)) = sample.get("gpus") else {
        return Vec::new();
    };
    let mut out: Vec<InstanceVram> = Vec::new();
    for (uuid, g) in live {
        let (Some(index), Value::Object(g)) = (idx_of(&idx, uuid), g) else {
            continue;
        };
        let procs: Vec<&Obj> = match g.get("procs") {
            Some(Value::Array(a)) => a
                .iter()
                .filter_map(Value::as_object)
                .filter(|p| {
                    p.get("container")
                        .and_then(Value::as_str)
                        .is_some_and(|c| names.iter().any(|n| n == c))
                })
                .collect(),
            _ => Vec::new(),
        };
        if procs.is_empty() {
            continue;
        }
        let known: Vec<i64> = procs
            .iter()
            .filter_map(|p| p.get("used_bytes").and_then(nv::int_of))
            .collect();
        #[allow(clippy::cast_precision_loss)]
        let used = (!known.is_empty()).then(|| known.iter().sum::<i64>() as f64 / nv::GIB);
        out.push(InstanceVram {
            gpu_index: index,
            gpu_uuid: uuid.clone(),
            used_gib: used,
        });
    }
    out.sort_by_key(|v| v.gpu_index);
    out
}

pub fn memory_for(node: &dyn NodeClient, names: &[String]) -> Option<HostMemory> {
    let sample = nv::sample(node);
    let Some(Value::Object(per)) = sample.get("containers") else {
        return None;
    };
    let mut sorted: Vec<&String> = names.iter().collect();
    sorted.sort();
    sorted.dedup();
    let parts: Vec<HostMemory> = sorted
        .into_iter()
        .filter_map(|n| match per.get(n.as_str()) {
            Some(Value::Object(c)) => container_memory(c.get("mem")),
            _ => None,
        })
        .collect();
    sum_memory(&parts)
}

/// The configured host (other than this one) that owns an NFS server address, if any.
fn owner_of(address: &str, ctx: &HostCtx) -> Option<String> {
    ctx.others
        .iter()
        .find(|(h, (_, addrs))| *h != ctx.host_id && addrs.iter().any(|a| a == address))
        .map(|(h, _)| h.clone())
}

/// Human label of a filesystem: "tank (zfs)", "NFS from gpu-box-2", "/ (ext4, Samsung 990)".
pub fn drive_label(m: &Obj, ctx: &HostCtx) -> String {
    if truthy(m.get("remote")) && truthy(m.get("remote_host")) {
        let rh = s_of(m.get("remote_host"));
        let who = owner_of(&rh, ctx)
            .and_then(|o| {
                ctx.others
                    .iter()
                    .find(|(h, _)| *h == o)
                    .map(|(_, (l, _))| l.clone())
            })
            .unwrap_or(rh);
        return format!("NFS from {who}");
    }
    let kind: Vec<String> = [m.get("fstype"), m.get("model")]
        .into_iter()
        .filter(|x| truthy(*x))
        .map(s_of)
        .collect();
    let mount = m
        .get("mount")
        .map_or_else(|| "?".to_string(), |v| s_of(Some(v)));
    if kind.is_empty() {
        mount
    } else {
        format!("{mount} ({})", kind.join(", "))
    }
}

/// Where an engine's model lives, from its container mounts that fall inside a policy model root (the node
/// reports `root`/`rel` per mount) joined with that root's filesystem from the node's `disks`.
pub fn location_for(ctx: &HostCtx, mounts: &[Obj]) -> Option<ModelLocation> {
    let cands: Vec<&Obj> = mounts
        .iter()
        .filter(|m| truthy(m.get("root")) && truthy(m.get("source")))
        .collect();
    let first = *cands.first()?;
    let pick = cands
        .iter()
        .copied()
        .find(|m| s_of(m.get("dest")).starts_with("/models") && m.get("dest").is_some())
        .unwrap_or(first);
    let root_id = s_of(pick.get("root"));
    let roots = nv::model_roots(ctx.node.as_ref());
    let root = roots
        .iter()
        .find(|(k, _)| *k == root_id)
        .map(|(_, r)| r.clone())
        .unwrap_or_default();
    let drive = nv::disk_mounts(ctx.node.as_ref()).into_iter().find(|d| {
        matches!(d.get("roots"), Some(Value::Array(a)) if a.iter().any(|r| r.as_str() == Some(root_id.as_str())))
    });
    let owner = drive
        .as_ref()
        .filter(|d| truthy(d.get("remote_host")))
        .and_then(|d| owner_of(&s_of(d.get("remote_host")), ctx));
    let remote = drive.as_ref().is_some_and(|d| truthy(d.get("remote")));
    let shared_from = owner.or_else(|| {
        drive
            .as_ref()
            .filter(|_| remote)
            .and_then(|d| opt_str(d.get("remote_host")))
    });
    Some(ModelLocation {
        host_id: ctx.host_id.clone(),
        root_id,
        root_label: opt_str(root.get("label")).filter(|s| !s.is_empty()),
        path: s_of(pick.get("source")),
        drive: drive.as_ref().map(|d| drive_label(d, ctx)),
        fstype: drive
            .as_ref()
            .and_then(|d| opt_str(d.get("fstype")))
            .filter(|s| !s.is_empty()),
        remote,
        shared_from: shared_from.filter(|s| !s.is_empty()),
    })
}

// ---- endpoint reading ---------------------------------------------------------------------------------------------
fn answered(e: &Obj) -> bool {
    ["health_status", "models_status", "metrics_status"]
        .iter()
        .any(|k| nv::ok_status(e.get(*k)) || nv::auth_status(e.get(*k)))
}

/// The probed endpoint that belongs to a container: by name, else by one of its published ports.
pub fn endpoint_of(c: &Obj, eps: &[Obj]) -> Option<Obj> {
    let named: Vec<&Obj> = eps
        .iter()
        .filter(|e| e.get("container") == c.get("name") && e.get("container").is_some())
        .collect();
    let ports = nv::host_ports(c);
    let cands: Vec<&Obj> = if named.is_empty() {
        eps.iter()
            .filter(|e| {
                e.get("port")
                    .and_then(nv::int_of)
                    .is_some_and(|p| ports.contains(&p))
            })
            .collect()
    } else {
        named
    };
    // prefer the one that reports healthy, then any that answers at all (the first of equals wins, like min())
    let key = |e: &Obj| {
        (
            !(nv::ok_status(e.get("health_status")) || nv::ok_status(e.get("models_status"))),
            !answered(e),
        )
    };
    let mut best: Option<&Obj> = None;
    for e in cands {
        if best.is_none_or(|b| key(e) < key(b)) {
            best = Some(e);
        }
    }
    best.cloned()
}

/// (state, reason, port, served models) for a running engine from its probed endpoint.
pub fn endpoint_state(
    ep: Option<&Obj>,
    has_ports: bool,
) -> (String, Option<String>, Option<i64>, Vec<String>) {
    let Some(ep) = ep else {
        if !has_ports {
            return (
                "unreachable".into(),
                Some(
                    "no published host port: the console reaches engines on their machine's loopback \
                     (publish a port on 127.0.0.1 to monitor it)"
                        .into(),
                ),
                None,
                Vec::new(),
            );
        }
        return (
            "unreachable".into(),
            Some("its published port(s) were not probed or did not answer yet".into()),
            None,
            Vec::new(),
        );
    };
    let port = ep.get("port").and_then(nv::int_of).unwrap_or(0);
    let models: Vec<String> = match ep.get("models") {
        Some(Value::Array(a)) => a
            .iter()
            .take(20)
            .filter_map(Value::as_str)
            .map(clean_text200)
            .collect(),
        _ => Vec::new(),
    };
    if nv::ok_status(ep.get("health_status")) || nv::ok_status(ep.get("models_status")) {
        let reason = (nv::auth_status(ep.get("models_status")) || truthy(ep.get("auth_required")))
            .then(|| {
                "/v1 endpoints need an API key (HTTP 401/403): metrics work, chat does not"
                    .to_string()
            });
        return ("ready".into(), reason, Some(port), models);
    }
    if answered(ep) || truthy(ep.get("auth_required")) {
        return (
            "auth_required".into(),
            Some(
                "the engine answered HTTP 401/403: an API key is required, so metrics and chat are unavailable"
                    .into(),
            ),
            Some(port),
            models,
        );
    }
    let is_503 = |k: &str| ep.get(k).and_then(nv::int_of) == Some(503);
    if is_503("health_status") || is_503("models_status") {
        // up, but not ready to serve yet
        return (
            "loading".into(),
            Some("the engine answers HTTP 503: it is still loading its model".into()),
            Some(port),
            models,
        );
    }
    (
        "unreachable".into(),
        Some(format!(
            "port {port} did not answer /v1/models, /health or /metrics"
        )),
        Some(port),
        models,
    )
}

// ---- instances ----------------------------------------------------------------------------------------------------
fn gpu_uuids_of(ctx: &HostCtx, c: &Obj) -> (Vec<i64>, Vec<String>) {
    let idx = uuid_map(ctx.node.as_ref());
    let uuids: Vec<String> = if truthy(c.get("all_gpus")) {
        let mut all = idx.clone();
        all.sort_by_key(|(_, i)| *i);
        all.into_iter().map(|(u, _)| u).collect()
    } else {
        match c.get("gpus") {
            Some(Value::Array(a)) => a
                .iter()
                .filter_map(Value::as_str)
                .filter(|u| idx_of(&idx, u).is_some())
                .map(str::to_string)
                .collect(),
            _ => Vec::new(),
        }
    };
    let ids = uuids.iter().filter_map(|u| idx_of(&idx, u)).collect();
    (ids, uuids)
}

/// The node's display-only parse of an engine's (already redacted) command line. Node text is untrusted, so it is
/// scrubbed again here: a token-shaped value never reaches the API, even if a node forgot to redact it.
fn params_of(c: &Obj) -> Obj {
    let Some(Value::Object(raw)) = c.get("params") else {
        return Obj::new();
    };
    let mut out = Obj::new();
    for (k, v) in raw {
        out.insert(
            clean_text(k, 64),
            Value::String(scrub(&clean_text200(&ec_proto::pyfmt::str_value(v)))),
        );
    }
    out
}

fn started_of(ctx: &HostCtx, c: &Obj) -> Option<f64> {
    nv::num(c.get("started_at")).map(|t| t - ctx.node.link().skew_s.unwrap_or(0.0))
}

pub fn external_controls(ctx: &HostCtx, state: &str) -> InstanceControls {
    let node = ctx.node.as_ref();
    let policy_ok = nv::external_control(node) == "stop_start";
    let manage = nv::mode(node) == "manage" && nv::link_up(node);
    if policy_ok && manage {
        let running = state != "stopped";
        return InstanceControls {
            start: !running,
            stop: running,
            restart: false,
            remove: false,
            logs: false,
            configure: true,
            reason: Some(
                "not created by the console: only stop and start are allowed on this machine"
                    .into(),
            ),
        };
    }
    let why = if !nv::link_up(node) {
        "this machine is not connected"
    } else if nv::mode(node) != "manage" {
        "this machine's node is in monitor mode"
    } else {
        "this machine's policy does not allow controlling engines the console did not create"
    };
    InstanceControls {
        configure: true,
        reason: Some(format!(
            "monitor-only: {why}. Clone it as a managed instance to control it."
        )),
        ..InstanceControls::default()
    }
}

/// An Instance with pydantic's defaults for everything but its identity.
pub fn blank_instance(
    id: &str,
    name: &str,
    engine: &str,
    repo_id: &str,
    created_at: f64,
) -> Instance {
    Instance {
        id: id.into(),
        name: name.into(),
        engine: engine.into(),
        repo_id: repo_id.into(),
        params: Obj::new(),
        gpu_ids: Vec::new(),
        gpu_uuids: Vec::new(),
        port: None,
        container_name: None,
        container_port: None,
        internal_endpoint: None,
        image: None,
        state: "stopped".into(),
        phase: None,
        progress_pct: None,
        pinned: false,
        ttl_idle_s: None,
        profile_id: None,
        error: None,
        last_logs: None,
        fit: None,
        created_at,
        started_at: None,
        last_request_at: None,
        uptime_s: None,
        managed: true,
        source: "console".into(),
        endpoint: None,
        served_models: Vec::new(),
        state_reason: None,
        history_since: None,
        host_memory: None,
        host_id: "local".into(),
        host_label: None,
        management: "managed".into(),
        script: None,
        vram: Vec::new(),
        model: None,
        model_location: None,
        controls: InstanceControls::default(),
    }
}

fn mounts_of(c: &Obj) -> Vec<Obj> {
    match c.get("mounts") {
        Some(Value::Array(a)) => a.iter().filter_map(Value::as_object).cloned().collect(),
        _ => Vec::new(),
    }
}

fn ext_instance(
    ctx: &HostCtx,
    c: &Obj,
    ep: Option<&Obj>,
    iid: &str,
    first_seen: f64,
    now: f64,
) -> Instance {
    let (state, reason, port, models) = endpoint_state(ep, !nv::host_ports(c).is_empty());
    let name = s_of(c.get("name"));
    let (gpu_ids, uuids) = gpu_uuids_of(ctx, c);
    let started = started_of(ctx, c);
    let dn = display_name(&name);
    let repo = models.first().cloned().unwrap_or_else(|| dn.clone());
    let engine = c
        .get("engine")
        .filter(|v| ec_proto::sanitize::truthy(v))
        .map_or_else(|| "openai_compatible".to_string(), |v| s_of(Some(v)));
    let mut i = blank_instance(iid, &dn, &engine, &repo, first_seen);
    i.params = params_of(c);
    i.gpu_ids = gpu_ids;
    i.gpu_uuids = uuids;
    i.port = port;
    i.container_name = Some(name.clone());
    i.image = truthy(c.get("image")).then(|| s_of(c.get("image")));
    i.started_at = started;
    i.uptime_s = started.filter(|s| *s != 0.0).map(|s| (now - s).max(0.0));
    i.managed = false;
    i.source = "external".into();
    i.endpoint = port
        .map(|p| format!("http://127.0.0.1:{p}"))
        .filter(|_| port.is_some_and(|p| p != 0));
    i.served_models = models;
    i.state_reason = reason;
    i.history_since = Some(first_seen);
    i.host_id = ctx.host_id.clone();
    i.host_label = Some(ctx.label.clone());
    i.management = "external".into();
    let names = [name];
    i.vram = vram_for(ctx.node.as_ref(), &names);
    i.host_memory = memory_for(ctx.node.as_ref(), &names);
    i.model_location = location_for(ctx, &mounts_of(c));
    i.controls = external_controls(ctx, &state);
    i.state = state;
    i
}

fn bare_instance(ctx: &HostCtx, ep: &Obj, iid: &str, first_seen: f64) -> Instance {
    let (state, reason, port, models) = endpoint_state(Some(ep), true);
    let engine = ep
        .get("engine")
        .filter(|v| ec_proto::sanitize::truthy(v))
        .map_or_else(|| "openai_compatible".to_string(), |v| s_of(Some(v)));
    let name = format!("127.0.0.1:{}", s_of(ep.get("port")));
    let dn = display_name(&name);
    let repo = models.first().cloned().unwrap_or_else(|| dn.clone());
    let mut i = blank_instance(iid, &dn, &engine, &repo, first_seen);
    i.port = port;
    i.managed = false;
    i.source = "external".into();
    i.endpoint = port
        .filter(|p| *p != 0)
        .map(|p| format!("http://127.0.0.1:{p}"));
    i.served_models = models;
    i.state_reason = reason;
    i.history_since = Some(first_seen);
    i.host_id = ctx.host_id.clone();
    i.host_label = Some(ctx.label.clone());
    i.management = "external".into();
    i.controls = external_controls(ctx, &state);
    i.state = state;
    i
}

pub fn script_controls(
    ctx: &HostCtx,
    script: &Obj,
    running: bool,
    busy: Option<&str>,
) -> InstanceControls {
    let label = script
        .get("label")
        .filter(|v| ec_proto::sanitize::truthy(v))
        .or_else(|| script.get("name"));
    let label = s_of(label);
    let actions: Vec<String> = match script.get("actions") {
        Some(Value::Array(a)) => a
            .iter()
            .filter_map(Value::as_object)
            .map(|x| s_of(x.get("name")))
            .collect(),
        _ => Vec::new(),
    };
    let reason = format!("managed by the script '{label}' registered in this machine's policy");
    let node = ctx.node.as_ref();
    let only = |r: String| InstanceControls {
        reason: Some(r),
        ..InstanceControls::default()
    };
    if !nv::link_up(node) {
        return only(format!("{reason}; this machine is not connected"));
    }
    if nv::mode(node) != "manage" {
        return only(format!(
            "{reason}; this machine's node is in monitor mode, so scripts cannot run"
        ));
    }
    if let Some(b) = busy.filter(|b| !b.is_empty()) {
        return only(format!("{reason}; its '{b}' action is running"));
    }
    let up = actions.iter().any(|a| a == "up");
    let down = actions.iter().any(|a| a == "down");
    InstanceControls {
        start: up && !running,
        stop: down && running,
        restart: up && down,
        reason: Some(reason),
        ..InstanceControls::default()
    }
}

fn script_instance(
    ctx: &HostCtx,
    script: &Obj,
    matched: &[Obj],
    eps: &[Obj],
    first_seen: f64,
    now: f64,
    busy: Option<&str>,
) -> Instance {
    let name = s_of(script.get("name"));
    let main = matched
        .iter()
        .find(|c| truthy(c.get("engine")))
        .or_else(|| matched.first());
    let ep = main.and_then(|m| endpoint_of(m, eps));
    let running = !matched.is_empty();
    let mut models: Vec<String> = Vec::new();
    let mut port: Option<i64> = None;
    let mut reason: Option<String> = None;
    let mut state = "stopped".to_string();
    if running {
        let has = main.is_some_and(|m| !nv::host_ports(m).is_empty());
        (state, reason, port, models) = endpoint_state(ep.as_ref(), has);
        if state == "unreachable" {
            // the container is up and simply not answering yet: it is loading
            state = "loading".into();
            reason = Some("the engine's container is running but does not answer yet".into());
        }
    }
    let busy = busy.filter(|b| !b.is_empty());
    if busy == Some("down") && running {
        state = "stopping".into();
    } else if busy == Some("up") && !running {
        state = "starting".into();
    }
    let names: Vec<String> = matched.iter().map(|c| s_of(c.get("name"))).collect();
    let mut gpu_ids: Vec<i64> = Vec::new();
    let mut uuids: Vec<String> = Vec::new();
    for c in matched {
        let (g, u) = gpu_uuids_of(ctx, c);
        for x in g {
            if !gpu_ids.contains(&x) {
                gpu_ids.push(x);
            }
        }
        for x in u {
            if !uuids.contains(&x) {
                uuids.push(x);
            }
        }
    }
    gpu_ids.sort_unstable();
    let started = matched
        .iter()
        .filter_map(|c| started_of(ctx, c))
        .reduce(f64::min);
    // a human label: only control characters and length are cleaned
    let raw_label = script
        .get("label")
        .filter(|v| ec_proto::sanitize::truthy(v))
        .map_or_else(|| name.clone(), |v| s_of(Some(v)));
    let label = clean_text(&raw_label, 64)
        .trim_matches(crate::services::discovery::py_is_space)
        .to_string();
    let shown = if label.is_empty() {
        name.clone()
    } else {
        label
    };
    let engine = script
        .get("engine")
        .filter(|v| ec_proto::sanitize::truthy(v))
        .or_else(|| {
            main.and_then(|m| m.get("engine"))
                .filter(|v| ec_proto::sanitize::truthy(v))
        })
        .map_or_else(|| "script".to_string(), |v| s_of(Some(v)));
    let repo = models.first().cloned().unwrap_or_else(|| shown.clone());
    let mut i = blank_instance(
        &script_instance_id(&ctx.host_id, &name),
        &shown,
        &engine,
        &repo,
        first_seen,
    );
    i.params = main.map(params_of).unwrap_or_default();
    i.gpu_ids = gpu_ids;
    i.gpu_uuids = uuids;
    i.port = port;
    i.container_name = main.map(|m| s_of(m.get("name")));
    i.image = main
        .filter(|m| truthy(m.get("image")))
        .map(|m| s_of(m.get("image")));
    i.phase = busy.map(|b| format!("script: {b} is running"));
    i.started_at = started;
    i.uptime_s = started
        .filter(|s| *s != 0.0 && running)
        .map(|s| (now - s).max(0.0));
    i.managed = false;
    i.source = "external".into();
    i.endpoint = port
        .filter(|p| *p != 0)
        .map(|p| format!("http://127.0.0.1:{p}"));
    i.served_models = models;
    i.state_reason = reason;
    i.history_since = Some(first_seen);
    i.host_id = ctx.host_id.clone();
    i.host_label = Some(ctx.label.clone());
    i.management = "script".into();
    i.script = Some(name);
    i.vram = vram_for(ctx.node.as_ref(), &names);
    i.host_memory = memory_for(ctx.node.as_ref(), &names);
    let all_mounts: Vec<Obj> = matched.iter().flat_map(mounts_of).collect();
    i.model_location = location_for(ctx, &all_mounts);
    i.controls = script_controls(ctx, script, running, busy);
    i.state = state;
    i
}

pub fn matches_script(name: &str, script: &Obj) -> bool {
    match script.get("containers") {
        Some(Value::Array(pats)) => pats
            .iter()
            .filter_map(Value::as_str)
            .any(|p| fnmatchcase(name, p)),
        _ => false,
    }
}

fn sha6(s: &str) -> String {
    hex::encode(sha1::Sha1::digest(s.as_bytes()))[..6].to_string()
}

/// Script instances (one per policy script) and external instances of one host, from its latest inventory.
///
/// `first_seen` is updated in place with the first time each id was seen (the history of its metrics starts there).
/// `busy` maps a script name to the action currently running for it.
pub fn build_host_instances(
    ctx: &HostCtx,
    first_seen: &mut HashMap<String, f64>,
    now: f64,
    busy: &[(String, String)],
) -> Vec<Instance> {
    let node = ctx.node.as_ref();
    let scripts: Vec<Obj> = nv::scripts(node)
        .into_iter()
        .filter(|s| SCRIPT_NAME.is_match(&s_of(s.get("name"))))
        .collect();
    let eps = nv::endpoints(node);
    let all = nv::containers(node);
    let conts: Vec<&Obj> = all
        .iter()
        .filter(|c| c.get("state").and_then(Value::as_str) == Some("running"))
        .collect();
    let owned_ports: Vec<i64> = all
        .iter()
        .filter(|c| truthy(c.get("managed")))
        .flat_map(nv::host_ports)
        .collect();
    let mut out: Vec<Instance> = Vec::new();
    let mut claimed: Vec<String> = Vec::new();
    for s in &scripts {
        let matched: Vec<Obj> = conts
            .iter()
            .filter(|c| !truthy(c.get("managed")) && matches_script(&s_of(c.get("name")), s))
            .map(|c| (*c).clone())
            .collect();
        claimed.extend(matched.iter().map(|c| s_of(c.get("name"))));
        let sname = s_of(s.get("name"));
        let iid = script_instance_id(&ctx.host_id, &sname);
        let fs = *first_seen.entry(iid).or_insert(now);
        let b = busy
            .iter()
            .find(|(n, _)| *n == sname)
            .map(|(_, a)| a.as_str());
        out.push(script_instance(ctx, s, &matched, &eps, fs, now, b));
    }
    let mut taken: Vec<String> = out.iter().map(|i| i.id.clone()).collect();
    let mut covered_ports: Vec<i64> = owned_ports;
    let mut sorted = conts;
    sorted.sort_by_key(|c| s_of(c.get("name")));
    for c in sorted {
        let name = s_of(c.get("name"));
        covered_ports.extend(nv::host_ports(c));
        if truthy(c.get("managed")) || claimed.contains(&name) || !truthy(c.get("engine")) {
            continue;
        }
        let mut iid = external_instance_id(&ctx.host_id, &name);
        if taken.contains(&iid) {
            // slug collision: disambiguate deterministically from the raw name
            iid = format!("{iid}-{}", sha6(&name));
        }
        taken.push(iid.clone());
        let fs = *first_seen.entry(iid.clone()).or_insert(now);
        let ep = endpoint_of(c, &eps);
        out.push(ext_instance(ctx, c, ep.as_ref(), &iid, fs, now));
    }
    let mut eps_sorted: Vec<&Obj> = eps.iter().collect();
    eps_sorted.sort_by_key(|e| e.get("port").and_then(nv::int_of).unwrap_or(0));
    for ep in eps_sorted {
        let port = ep.get("port").and_then(nv::int_of).unwrap_or(0);
        if truthy(ep.get("container")) || covered_ports.contains(&port) {
            continue;
        }
        let recognised = truthy(ep.get("engine"))
            || (answered(ep)
                && nv::ok_status(ep.get("models_status"))
                && (nv::ok_status(ep.get("metrics_status"))
                    || nv::ok_status(ep.get("health_status"))));
        if !recognised {
            continue; // not an engine we can recognise: never listed
        }
        let mut iid = external_instance_id(&ctx.host_id, &format!("127.0.0.1:{port}"));
        if taken.contains(&iid) {
            iid = format!("{iid}-{}", sha6(&port.to_string()));
        }
        taken.push(iid.clone());
        let fs = *first_seen.entry(iid.clone()).or_insert(now);
        out.push(bare_instance(ctx, ep, &iid, fs));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_m1_ids_of_hyphenated_hosts_never_collide() {
        // security review M1: host `gpu` running `2-vllm` and host `gpu-2` running `vllm`
        assert_eq!(external_instance_id("gpu", "2-vllm"), "ext-gpu-2-vllm");
        assert_eq!(external_instance_id("gpu-2", "vllm"), "ext-gpu--2-vllm");
        assert_ne!(
            external_instance_id("gpu", "2-vllm"),
            external_instance_id("gpu-2", "vllm")
        );
        assert_eq!(
            script_instance_id("local", "big-model"),
            "scr-local-big-model"
        );
        assert_eq!(
            external_instance_id("local", "127.0.0.1:9100"),
            "ext-local-127-0-0-1-9100"
        );
    }

    proptest::proptest! {
        // M1 as a property: two different (host, name) pairs never give the same external id unless the names slug alike
        #[test]
        fn prop_ext_ids_identify_the_host(h1 in "[a-z0-9][a-z0-9-]{0,8}", h2 in "[a-z0-9][a-z0-9-]{0,8}",
                                          n1 in "[a-zA-Z0-9 ._-]{0,12}", n2 in "[a-zA-Z0-9 ._-]{0,12}") {
            if h1 != h2 {
                proptest::prop_assert_ne!(external_instance_id(&h1, &n1), external_instance_id(&h2, &n2));
            }
        }
    }

    #[test]
    fn test_endpoint_state_classifies_like_python() {
        let ep = |v: Value| v.as_object().cloned().unwrap_or_default();
        let (s, r, p, _) = endpoint_state(None, false);
        assert_eq!((s.as_str(), p), ("unreachable", None));
        assert!(r.unwrap_or_default().contains("no published host port"));
        let ready = ep(serde_json::json!({"port": 5, "health_status": 200, "models_status": 401}));
        let (s, r, p, _) = endpoint_state(Some(&ready), true);
        assert_eq!((s.as_str(), p), ("ready", Some(5)));
        assert!(r.unwrap_or_default().contains("API key"));
        let auth = ep(serde_json::json!({"port": 5, "health_status": 401}));
        assert_eq!(endpoint_state(Some(&auth), true).0, "auth_required");
        let warm = ep(serde_json::json!({"port": 5, "health_status": 503}));
        assert_eq!(endpoint_state(Some(&warm), true).0, "loading");
        let dead = ep(serde_json::json!({"port": 5}));
        let (s, r, _, _) = endpoint_state(Some(&dead), true);
        assert_eq!(s, "unreachable");
        assert_eq!(
            r.as_deref(),
            Some("port 5 did not answer /v1/models, /health or /metrics")
        );
    }
}
