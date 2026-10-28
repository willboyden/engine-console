//! Python: backend/src/engine_console/services/fleet_status.py
//! (tests/fleet/test_fleet_status.py).
//!
//! Pure builders: what a host's node last said, plus its link state, become the API shapes HostSummary / HostDetail.
//!
//! Nothing here talks to a node, reads a file or keeps state. Every input is untrusted data that has already been
//! through `ec_proto::sanitize_*` (unknown fields dropped, strings capped, wrong types turned into null), so any value
//! can still be missing or null: every accessor tolerates that and degrades to "unknown" instead of failing.
//!
//! Some facts only exist across hosts. Whether host B's model root is really host A's disk (an NFS mount of A) can
//! only be told by comparing B's mounts with A's addresses, so `build_detail` takes every host's data.

use std::collections::HashSet;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use serde_json::Value;

use crate::domain::errors::Obj;
use crate::domain::fleet::{
    CountsProvider, CpuFacts, DockerFacts, GpuLive, GpuProc, HostConfig, HostCounts, HostDetail,
    HostSummary, ImageInfo, InstanceResolver, LinkStatus, MountInfo, NodeFacts, PolicyFacts,
    RootInfo, ScriptAction, ScriptInfo,
};
use crate::domain::models::HostMemInfo;
use crate::services::hostmem::alerts_for;

pub const GIB: f64 = 1_073_741_824.0;
/// link states in which a sample still describes the machine
pub const LIVE_STATES: &[&str] = &["up", "stale"];
pub const SCRIPT_STATUSES: &[&str] = &["up", "down", "unknown", "error"];

/// Everything the hub knows about one host at one moment.
#[derive(Debug, Clone)]
pub struct HostData {
    pub host: HostConfig,
    pub link: LinkStatus,
    pub hello: Option<Arc<Obj>>,
    pub sample: Option<Arc<Obj>>,
    pub disks: Option<Arc<Obj>>,
    pub inventory: Option<Arc<Obj>>,
}

// ── tolerant accessors (every one of them takes untrusted, possibly missing data) ───────────────────────────
static EMPTY: std::sync::LazyLock<Obj> = std::sync::LazyLock::new(Obj::new);

fn d(v: Option<&Value>) -> &Obj {
    match v {
        Some(Value::Object(o)) => o,
        _ => &EMPTY,
    }
}

fn opt(o: Option<&Arc<Obj>>) -> &Obj {
    o.map_or(&EMPTY, |a| a.as_ref())
}

fn l(v: Option<&Value>) -> &[Value] {
    match v {
        Some(Value::Array(a)) => a,
        _ => &[],
    }
}

/// a non-empty string
fn s(v: Option<&Value>) -> Option<String> {
    v.and_then(Value::as_str)
        .filter(|x| !x.is_empty())
        .map(str::to_string)
}

/// an int (never a bool or a float)
fn i(v: Option<&Value>) -> Option<i64> {
    match v {
        Some(Value::Number(n)) if n.is_i64() || n.is_u64() => n.as_i64(),
        _ => None,
    }
}

/// an int or a float, as a float (never a bool)
fn f(v: Option<&Value>) -> Option<f64> {
    match v {
        Some(Value::Number(n)) => n.as_f64(),
        _ => None,
    }
}

fn b(v: Option<&Value>) -> Option<bool> {
    v.and_then(Value::as_bool)
}

fn is_true(v: Option<&Value>) -> bool {
    v == Some(&Value::Bool(true))
}

fn gib(v: Option<&Value>) -> Option<f64> {
    i(v).map(|n| n as f64 / GIB)
}

/// Python's `x or y` for floats: 0.0 (and None) fall through.
fn or_f(x: Option<f64>, y: impl FnOnce() -> Option<f64>) -> Option<f64> {
    match x {
        Some(v) if v != 0.0 => Some(v),
        _ => y(),
    }
}

/// `posixpath.normpath`
pub fn normpath(p: &str) -> String {
    if p.is_empty() {
        return ".".into();
    }
    let initial = if p.starts_with('/') {
        if p.starts_with("//") && !p.starts_with("///") {
            2
        } else {
            1
        }
    } else {
        0
    };
    let mut comps: Vec<&str> = Vec::new();
    for c in p.split('/') {
        if c.is_empty() || c == "." {
            continue;
        }
        if c != ".." || (initial == 0 && comps.is_empty()) || comps.last() == Some(&"..") {
            comps.push(c);
        } else if !comps.is_empty() {
            comps.pop();
        }
    }
    let out = format!("{}{}", "/".repeat(initial), comps.join("/"));
    if out.is_empty() { ".".into() } else { out }
}

/// A normalised absolute POSIX path, or None. Used only to compare paths, never to open anything.
fn abs(v: Option<&Value>) -> Option<String> {
    s(v).filter(|x| x.starts_with('/')).map(|x| normpath(&x))
}

// ── one host ─────────────────────────────────────────────────────────────────────────────────────────────
pub fn node_facts(hello: Option<&Obj>) -> Option<NodeFacts> {
    let hello = hello.filter(|h| !h.is_empty())?;
    let host = d(hello.get("host"));
    let gpu = d(hello.get("gpu"));
    Some(NodeFacts {
        version: s(hello.get("version")),
        hostname: s(host.get("hostname")),
        arch: s(host.get("arch")),
        os: s(host.get("os")),
        kernel: s(host.get("kernel")),
        cpu_model: s(host.get("cpu_model")),
        cpu_cores: i(host.get("cpu_cores")),
        cpu_threads: i(host.get("cpu_threads")),
        python: s(host.get("python")),
        driver: s(gpu.get("driver")),
        cuda: s(gpu.get("cuda")),
        gpu_source: s(gpu.get("source")),
        forced_cmd: b(hello.get("forced_cmd")),
        via_ssh: b(hello.get("via_ssh")),
    })
}

pub fn docker_facts(hello: Option<&Obj>) -> Option<DockerFacts> {
    let hello = hello.filter(|h| !h.is_empty())?;
    let dk = d(hello.get("docker"));
    Some(DockerFacts {
        available: is_true(dk.get("available")),
        context: s(dk.get("context")),
        rootless: b(dk.get("rootless")),
        version: s(dk.get("version")),
        nvidia_runtime: b(dk.get("nvidia_runtime")),
        error: s(dk.get("error")),
    })
}

pub fn policy_facts(hello: Option<&Obj>) -> Option<PolicyFacts> {
    let hello = hello.filter(|h| !h.is_empty())?;
    let p = d(hello.get("policy"));
    let pr: Vec<i64> = l(p.get("port_range"))
        .iter()
        .filter_map(|x| i(Some(x)))
        .collect();
    let ext = s(p.get("external_control"));
    Some(PolicyFacts {
        path: s(p.get("path")),
        // anything unexpected means read-only
        mode: if p.get("mode").and_then(Value::as_str) == Some("manage") {
            "manage".into()
        } else {
            "monitor".into()
        },
        error: s(p.get("error")),
        engine_images: l(p.get("engine_images"))
            .iter()
            .filter_map(|x| x.as_str().map(str::to_string))
            .collect(),
        gateway_image: s(p.get("gateway_image")),
        engine_network: s(p.get("engine_network")),
        port_range: if pr.len() == 2 { pr } else { Vec::new() },
        allow_pull: is_true(p.get("allow_pull")),
        allow_delete: is_true(p.get("allow_delete")),
        allow_ipc_host: b(p.get("allow_ipc_host")).unwrap_or(true),
        external_control: if ext.as_deref() == Some("stop_start") {
            "stop_start".into()
        } else {
            "none".into()
        },
        max_shm_gib: f(p.get("max_shm_gib")),
    })
}

pub fn cpu_facts(sample: Option<&Obj>) -> Option<CpuFacts> {
    let sample = sample.filter(|x| !x.is_empty())?;
    let c = d(sample.get("cpu"));
    Some(CpuFacts {
        load1: f(c.get("load1")),
        load5: f(c.get("load5")),
        load15: f(c.get("load15")),
        util_pct: f(c.get("util_pct")),
    })
}

/// HostMemInfo from the sample's /proc/meminfo numbers (bytes). `used` is total - available, the same rule as the
/// local /hardware report, so alerts mean the same thing on every machine.
pub fn host_memory(sample: Option<&Obj>, link: &LinkStatus) -> Option<HostMemInfo> {
    let sample = sample.filter(|x| !x.is_empty())?;
    let m = d(sample.get("mem"));
    let total = i(m.get("total")).filter(|t| *t > 0)?;
    let avail = i(m.get("available"))?;
    let swap_total = i(m.get("swap_total")).unwrap_or(0);
    let swap_free = i(m.get("swap_free")).unwrap_or(0);
    // the node's clock minus the measured skew is the hub's clock, which is what a browser compares against
    let updated = match f(sample.get("t")) {
        Some(t) => t - link.skew_s.unwrap_or(0.0),
        None => link.last_seen.unwrap_or(0.0),
    };
    let g = |n: i64| n as f64 / GIB;
    Some(HostMemInfo {
        total_gib: g(total),
        used_gib: g((total - avail).max(0)),
        available_gib: g(avail),
        free_gib: g(i(m.get("free")).unwrap_or(0)),
        cached_gib: g(i(m.get("cached")).unwrap_or(0)),
        shmem_gib: g(i(m.get("shmem")).unwrap_or(0)),
        swap_total_gib: g(swap_total),
        swap_used_gib: g((swap_total - swap_free).max(0)),
        updated_at: updated,
    })
}

fn instance_of(
    resolver: Option<&InstanceResolver>,
    host_id: &str,
    container: Option<&str>,
) -> Option<String> {
    let (r, c) = (resolver?, container?);
    // a resolver bug must not take the host list down
    if let Ok(v) = catch_unwind(AssertUnwindSafe(|| r(host_id, c))) {
        v
    } else {
        tracing::warn!(target: "engine_console.services.fleet_status", "instance resolver failed for host {host_id}");
        None
    }
}

/// Static facts from hello, live numbers from the sample (keyed by UUID), bytes -> GiB. Without a hello there is
/// nothing to list: the hub only ever shows GPUs the node announced.
pub fn build_gpus(
    data: &HostData,
    sample: Option<&Obj>,
    resolver: Option<&InstanceResolver>,
) -> Vec<GpuLive> {
    let live = d(sample.and_then(|x| x.get("gpus")));
    let mut gpus: Vec<&Obj> = l(opt(data.hello.as_ref()).get("gpus"))
        .iter()
        .map(|g| d(Some(g)))
        .collect();
    // a GPU with no index goes last (a stable sort, as Python's)
    gpus.sort_by_key(|g| i(g.get("index")).unwrap_or(1_000_000));
    let mut out = Vec::new();
    for g in gpus {
        let Some(uuid) = s(g.get("uuid")) else {
            continue;
        };
        let sm = d(live.get(&uuid));
        let total =
            or_f(gib(g.get("total_bytes")), || gib(sm.get("mem_total_bytes"))).unwrap_or(0.0);
        let used = gib(sm.get("mem_used_bytes"));
        let procs = l(sm.get("procs"))
            .iter()
            .map(|p| d(Some(p)))
            .filter_map(|p| {
                let pid = i(p.get("pid"))?;
                let container = s(p.get("container"));
                Some(GpuProc {
                    pid,
                    name: s(p.get("name")),
                    instance_id: instance_of(resolver, &data.host.id, container.as_deref()),
                    container,
                    used_gib: gib(p.get("used_bytes")),
                })
            })
            .collect();
        out.push(GpuLive {
            index: i(g.get("index")).unwrap_or(0),
            uuid,
            name: s(g.get("name")).unwrap_or_else(|| "GPU".into()),
            total_gib: total,
            used_gib: used,
            free_gib: used.map(|u| (total - u).max(0.0)),
            util_pct: f(sm.get("util_pct")),
            temp_c: f(sm.get("temp_c")),
            power_w: f(sm.get("power_w")),
            power_limit_w: f(sm.get("power_limit_w")),
            fan_pct: f(sm.get("fan_pct")),
            compute_capability: s(g.get("compute_capability")),
            unified_memory: is_true(g.get("unified_memory")),
            procs,
        });
    }
    out
}

fn counts_of(provider: Option<&CountsProvider>, host_id: &str) -> HostCounts {
    let Some(p) = provider else {
        return HostCounts::default();
    };
    if let Ok(c) = catch_unwind(AssertUnwindSafe(|| p(host_id))) {
        c
    } else {
        tracing::warn!(target: "engine_console.services.fleet_status", "counts provider failed for host {host_id}");
        HostCounts::default()
    }
}

pub fn build_summary(
    data: &HostData,
    resolver: Option<&InstanceResolver>,
    counts: Option<&CountsProvider>,
) -> HostSummary {
    // A sample from before a link drop describes a machine we can no longer see: show it as unknown, not as live.
    let sample = if LIVE_STATES.contains(&data.link.state.as_str()) {
        data.sample.as_deref()
    } else {
        None
    };
    let hello = data.hello.as_deref();
    let mem = host_memory(sample, &data.link);
    HostSummary {
        id: data.host.id.clone(),
        label: data.host.label.clone(),
        remote: data.host.ssh.is_some(),
        link: data.link.clone(),
        node: node_facts(hello),
        gpus: build_gpus(data, sample, resolver),
        alerts: mem.as_ref().map(alerts_for).unwrap_or_default(),
        memory: mem,
        cpu: cpu_facts(sample),
        docker: docker_facts(hello),
        policy: policy_facts(hello),
        counts: counts_of(counts, &data.host.id),
    }
}

// ── disks, roots and storage shared between hosts ────────────────────────────────────────────────────────
/// `path` is `base` or inside it, compared by whole path components ("/data/hf2" is not under "/data/hf").
fn under(path: &str, base: &str) -> bool {
    if base == "/" {
        return true;
    }
    path == base || path.starts_with(&format!("{base}/"))
}

fn overlap(a: &str, b: &str) -> bool {
    under(a, b) || under(b, a)
}

fn norm_addr(v: Option<&Value>) -> Option<String> {
    let raw = s(v)?;
    let x = crate::services::hosts::py_strip(&raw).trim_matches(['[', ']']);
    let x = x.split('%').next().unwrap_or("").to_lowercase();
    Some(match x.parse::<std::net::IpAddr>() {
        Ok(ip) => ip.to_string(),
        Err(_) => x,
    })
}

fn is_loopback(addr: &str) -> bool {
    match addr.parse::<std::net::IpAddr>() {
        Ok(ip) => ip.is_loopback(),
        Err(_) => addr == "localhost" || addr == "localhost.localdomain",
    }
}

/// Which configured host owns an address or name. An address that two hosts both report (a docker bridge such as
/// 172.17.0.1 exists on every machine with docker) identifies nobody, so it is left out.
pub struct AddressBook {
    addr: Vec<(String, String)>,
    name: Vec<(String, String)>,
}

fn add_owner(map: &mut Vec<(String, Vec<String>)>, key: String, hid: &str) {
    match map.iter_mut().find(|(k, _)| *k == key) {
        Some((_, owners)) => {
            if !owners.iter().any(|o| o == hid) {
                owners.push(hid.to_string());
            }
        }
        None => map.push((key, vec![hid.to_string()])),
    }
}

impl AddressBook {
    pub fn new(fleet: &[HostData]) -> Self {
        let mut by_addr: Vec<(String, Vec<String>)> = Vec::new();
        let mut by_name: Vec<(String, Vec<String>)> = Vec::new();
        for x in fleet {
            let hid = &x.host.id;
            let hello = opt(x.hello.as_ref());
            for a in l(hello.get("addresses")) {
                if let Some(addr) = norm_addr(d(Some(a)).get("addr"))
                    && !addr.is_empty()
                    && !is_loopback(&addr)
                {
                    add_owner(&mut by_addr, addr, hid);
                }
            }
            let name = s(d(hello.get("host")).get("hostname"))
                .unwrap_or_default()
                .to_lowercase();
            if !name.is_empty() && !is_loopback(&name) {
                let short = name.split('.').next().unwrap_or("").to_string();
                add_owner(&mut by_name, name, hid);
                add_owner(&mut by_name, short, hid);
            }
        }
        let single = |m: Vec<(String, Vec<String>)>| -> Vec<(String, String)> {
            m.into_iter()
                .filter(|(_, v)| v.len() == 1)
                .map(|(k, mut v)| (k, v.remove(0)))
                .collect()
        };
        Self {
            addr: single(by_addr),
            name: single(by_name),
        }
    }

    /// The configured host an NFS server address or name belongs to (never `exclude`: a machine mounting its own
    /// export is not borrowing someone else's disk).
    pub fn find(&self, server: Option<&str>, exclude: &str) -> Option<String> {
        let key = norm_addr(server.map(|x| Value::String(x.into())).as_ref())?;
        if key.is_empty() || is_loopback(&key) {
            return None;
        }
        let hid = self
            .addr
            .iter()
            .find(|(k, _)| *k == key)
            .or_else(|| self.name.iter().find(|(k, _)| *k == key))
            .map(|(_, h)| h.clone())?;
        (hid != exclude).then_some(hid)
    }
}

/// The human name of a filesystem: "tank (zfs)", "NFS from gpu-box-2", "/ (ext4, Samsung 990)".
pub fn mount_label(
    mount: &str,
    fstype: Option<&str>,
    source: Option<&str>,
    model: Option<&str>,
    remote: bool,
    server: Option<&str>,
    server_label: Option<&str>,
) -> String {
    fn nonempty(x: Option<&str>) -> Option<&str> {
        x.filter(|v| !v.is_empty())
    }
    let ft = fstype.unwrap_or("").to_lowercase();
    if ft == "zfs" {
        let src = source.unwrap_or("");
        let pool = src
            .split('/')
            .next()
            .unwrap_or("")
            .split('@')
            .next()
            .unwrap_or("");
        return format!("{} (zfs)", if pool.is_empty() { mount } else { pool });
    }
    if ft.starts_with("nfs") || remote {
        let kind = if ft.starts_with("nfs") {
            "NFS".to_string()
        } else if ["cifs", "smb3", "smbfs"].contains(&ft.as_str()) {
            "CIFS".to_string()
        } else if ft.is_empty() {
            "Network".to_string()
        } else {
            ft.to_uppercase()
        };
        let who = nonempty(server_label)
            .or(nonempty(server))
            .or(nonempty(source))
            .unwrap_or("another machine");
        return format!("{kind} from {who}");
    }
    match (nonempty(fstype), nonempty(model)) {
        (Some(ft), Some(m)) => format!("{mount} ({ft}, {m})"),
        (None, Some(m)) => format!("{mount} ({m})"),
        (Some(ft), None) => format!("{mount} ({ft})"),
        (None, None) => mount.to_string(),
    }
}

fn mount_info(
    m: &Obj,
    data: &HostData,
    book: &AddressBook,
    labels: &[(String, String)],
    roots: Vec<String>,
) -> MountInfo {
    let mount = s(m.get("mount")).unwrap_or_else(|| "?".into());
    let (fstype, source, model) = (s(m.get("fstype")), s(m.get("source")), s(m.get("model")));
    let remote = is_true(m.get("remote"));
    let server = s(m.get("remote_host"));
    let owner = if remote {
        book.find(server.as_deref(), &data.host.id)
    } else {
        None
    };
    let owner_label = labels
        .iter()
        .find(|(h, _)| Some(h) == owner.as_ref())
        .map(|(_, lab)| lab.as_str());
    MountInfo {
        label: mount_label(
            &mount,
            fstype.as_deref(),
            source.as_deref(),
            model.as_deref(),
            remote,
            server.as_deref(),
            owner_label,
        ),
        mount,
        fstype,
        source,
        device: s(m.get("device")),
        model,
        size_bytes: i(m.get("size")),
        used_bytes: i(m.get("used")),
        free_bytes: i(m.get("free")),
        readonly: is_true(m.get("readonly")),
        remote,
        remote_host: server,
        remote_host_id: owner,
        remote_path: s(m.get("remote_path")),
        roots,
    }
}

/// Index of the mount a model root lives on: the one the node says holds it, else the longest mount path that is a
/// prefix of the root's path.
fn mount_of_root(root_id: &str, root_path: &str, mounts: &[&Obj]) -> Option<usize> {
    let claimed: Vec<usize> = mounts
        .iter()
        .enumerate()
        .filter(|(_, m)| {
            l(m.get("roots"))
                .iter()
                .any(|r| r.as_str() == Some(root_id))
        })
        .map(|(n, _)| n)
        .collect();
    if claimed.len() == 1 {
        return Some(claimed[0]);
    }
    let mut best = None;
    let mut best_len: i64 = -1;
    for (n, m) in mounts.iter().enumerate() {
        if let Some(mp) = abs(m.get("mount"))
            && under(root_path, &mp)
        {
            let depth = if mp == "/" {
                0
            } else {
                i64::try_from(mp.matches('/').count()).unwrap_or(i64::MAX)
            };
            if depth > best_len {
                best = Some(n);
                best_len = depth;
            }
        }
    }
    best
}

struct RootLoc {
    host_id: String,
    root_id: String,
    /// the host whose disk this really is
    owner: String,
    /// the root's path expressed on the owner (a root that is an NFS mount is translated)
    path: String,
    imported: bool,
}

/// `posixpath.relpath(path, start)` for a `path` known to be `start` or under it.
fn relpath_under(path: &str, start: &str) -> String {
    let rest = if start == "/" {
        path.get(1..).unwrap_or("")
    } else {
        path.get(start.len()..).unwrap_or("")
    };
    let rest = rest.trim_start_matches('/');
    if rest.is_empty() {
        ".".into()
    } else {
        rest.to_string()
    }
}

fn root_locations(fleet: &[HostData], book: &AddressBook) -> Vec<RootLoc> {
    let mut locs = Vec::new();
    for x in fleet {
        let hid = &x.host.id;
        let mounts: Vec<&Obj> = l(opt(x.disks.as_ref()).get("mounts"))
            .iter()
            .map(|m| d(Some(m)))
            .collect();
        let hello = opt(x.hello.as_ref());
        for root in l(d(hello.get("policy")).get("model_roots"))
            .iter()
            .map(|r| d(Some(r)))
        {
            let (Some(rid), Some(rpath)) = (s(root.get("id")), abs(root.get("path"))) else {
                continue;
            };
            let (mut owner, mut path, mut imported) = (hid.clone(), rpath.clone(), false);
            let m = mount_of_root(&rid, &rpath, &mounts).map_or(&*EMPTY, |n| mounts[n]);
            let (mp, export) = (abs(m.get("mount")), abs(m.get("remote_path")));
            let server = if s(m.get("fstype"))
                .unwrap_or_default()
                .to_lowercase()
                .starts_with("nfs")
            {
                book.find(s(m.get("remote_host")).as_deref(), hid)
            } else {
                None
            };
            if let (Some(server), Some(mp), Some(export)) = (server, mp, export)
                && under(&rpath, &mp)
            {
                let rel = relpath_under(&rpath, &mp);
                owner = server;
                imported = true;
                path = if rel == "." {
                    export
                } else {
                    normpath(&format!("{}/{rel}", export.trim_end_matches('/')))
                };
            }
            locs.push(RootLoc {
                host_id: hid.clone(),
                root_id: rid,
                owner,
                path,
                imported,
            });
        }
    }
    locs
}

/// (host_id, root_id) -> (shared_from, visible_on), in first-seen order.
type Shares = Vec<((String, String), (Option<String>, Vec<String>))>;

/// Two roots are the same storage when they resolve to the same owner host and their paths on it overlap.
/// `visible_on` lists this host first, then the owner, then the rest in configuration order; `shared_from` is set
/// only for a root that is another configured host's disk AND overlaps one of that host's own roots (otherwise that
/// host's library would not list what this one sees).
fn shares(fleet: &[HostData], book: &AddressBook) -> Shares {
    let order: Vec<&String> = fleet.iter().map(|x| &x.host.id).collect();
    let locs = root_locations(fleet, book);
    let mut out: Shares = Vec::new();
    for loc in &locs {
        let peers: Vec<&RootLoc> = locs
            .iter()
            .filter(|o| o.owner == loc.owner && overlap(&o.path, &loc.path))
            .collect();
        let mut seen: HashSet<&str> = peers.iter().map(|o| o.host_id.as_str()).collect();
        seen.insert(&loc.host_id);
        let mut visible = vec![loc.host_id.clone()];
        if seen.contains(loc.owner.as_str()) && loc.owner != loc.host_id {
            visible.push(loc.owner.clone());
        }
        for h in &order {
            if seen.contains(h.as_str()) && !visible.contains(h) {
                visible.push((*h).clone());
            }
        }
        let shared = (loc.imported && peers.iter().any(|o| o.host_id == loc.owner && !o.imported))
            .then(|| loc.owner.clone());
        let key = (loc.host_id.clone(), loc.root_id.clone());
        match out.iter_mut().find(|(k, _)| *k == key) {
            Some(slot) => slot.1 = (shared, visible),
            None => out.push((key, (shared, visible))),
        }
    }
    out
}

// ── scripts, images, addresses ───────────────────────────────────────────────────────────────────────────
/// Scripts from the node's policy (names, labels, allowed args: never the argv) with the last status the node
/// reported for each in its inventory.
pub fn build_scripts(data: &HostData) -> Vec<ScriptInfo> {
    let mut status: Vec<(String, &Obj)> = Vec::new();
    for st in l(opt(data.inventory.as_ref()).get("scripts"))
        .iter()
        .map(|x| d(Some(x)))
    {
        if let Some(n) = s(st.get("name")) {
            match status.iter_mut().find(|(k, _)| *k == n) {
                Some(slot) => slot.1 = st,
                None => status.push((n, st)),
            }
        }
    }
    let hello = opt(data.hello.as_ref());
    let mut out = Vec::new();
    for sc in l(d(hello.get("policy")).get("scripts"))
        .iter()
        .map(|x| d(Some(x)))
    {
        let Some(name) = s(sc.get("name")) else {
            continue;
        };
        let st = status
            .iter()
            .find(|(k, _)| *k == name)
            .map_or(&*EMPTY, |(_, v)| *v);
        let raw_status = s(st.get("status"));
        let actions = l(sc.get("actions"))
            .iter()
            .map(|a| d(Some(a)))
            .filter_map(|a| {
                let an = s(a.get("name"))?;
                Some(ScriptAction {
                    label: s(a.get("label")).unwrap_or_else(|| an.clone()),
                    name: an,
                    args: l(a.get("args"))
                        .iter()
                        .filter_map(|x| x.as_str().map(str::to_string))
                        .collect(),
                    confirm: is_true(a.get("confirm")),
                    danger: is_true(a.get("danger")),
                })
            })
            .collect();
        out.push(ScriptInfo {
            host_id: data.host.id.clone(),
            label: s(sc.get("label")).unwrap_or_else(|| name.clone()),
            name,
            engine: s(sc.get("engine")),
            description: s(sc.get("description")).unwrap_or_default(),
            status: match raw_status {
                Some(x) if SCRIPT_STATUSES.contains(&x.as_str()) => x,
                _ => "unknown".into(),
            },
            detail: s(st.get("detail")),
            checked_at: f(st.get("checked_at")),
            actions,
            containers: l(sc.get("containers"))
                .iter()
                .filter_map(|x| x.as_str().map(str::to_string))
                .collect(),
        });
    }
    out
}

/// ImageInfo list from a sanitized `docker.images` result.
pub fn images_from_result(result: &Obj) -> Vec<ImageInfo> {
    l(result.get("images"))
        .iter()
        .map(|x| d(Some(x)))
        .filter_map(|x| {
            Some(ImageInfo {
                ref_: s(x.get("ref"))?,
                id: s(x.get("id")),
                size_bytes: i(x.get("size")),
                created: f(x.get("created")),
                arch: s(x.get("arch")),
            })
        })
        .collect()
}

/// "10.0.0.5/24 (eth0)" for every non-loopback interface address the node reported.
pub fn addresses_of(hello: Option<&Obj>) -> Vec<String> {
    let mut out = Vec::new();
    for a in l(hello.unwrap_or(&EMPTY).get("addresses"))
        .iter()
        .map(|x| d(Some(x)))
    {
        let (Some(addr), Some(norm)) = (s(a.get("addr")), norm_addr(a.get("addr"))) else {
            continue;
        };
        if is_loopback(&norm) {
            continue;
        }
        let p = i(a.get("prefix"))
            .map(|p| format!("/{p}"))
            .unwrap_or_default();
        let ifc = s(a.get("iface"))
            .map(|x| format!(" ({x})"))
            .unwrap_or_default();
        out.push(format!("{addr}{p}{ifc}"));
    }
    out
}

// ── the detail view ──────────────────────────────────────────────────────────────────────────────────────
/// The detail of `host_id`; `fleet` holds every configured host's data (including this one, in configuration
/// order), which is how a mount is matched to the host that exports it. `images` come from a separate (cached)
/// `docker.images` call: they are not part of the node's pushed inventory. None when `host_id` is not in `fleet`.
pub fn build_detail(
    host_id: &str,
    fleet: &[HostData],
    images: Vec<ImageInfo>,
    resolver: Option<&InstanceResolver>,
    counts: Option<&CountsProvider>,
) -> Option<HostDetail> {
    let data = fleet.iter().find(|x| x.host.id == host_id)?;
    let summary = build_summary(data, resolver, counts);
    let book = AddressBook::new(fleet);
    let labels: Vec<(String, String)> = fleet
        .iter()
        .map(|x| (x.host.id.clone(), x.host.label.clone()))
        .collect();
    let raw_mounts: Vec<&Obj> = l(opt(data.disks.as_ref()).get("mounts"))
        .iter()
        .map(|m| d(Some(m)))
        .collect();
    let hello = opt(data.hello.as_ref());
    let root_specs: Vec<&Obj> = l(d(hello.get("policy")).get("model_roots"))
        .iter()
        .map(|r| d(Some(r)))
        .filter(|r| s(r.get("id")).is_some() && abs(r.get("path")).is_some())
        .collect();
    // a dict: the first occurrence of an id keeps its place, the last one's mount wins
    let mut root_mount: Vec<(String, Option<usize>)> = Vec::new();
    for r in &root_specs {
        let rid = s(r.get("id")).unwrap_or_default();
        let mi = mount_of_root(&rid, &abs(r.get("path")).unwrap_or_default(), &raw_mounts);
        match root_mount.iter_mut().find(|(k, _)| *k == rid) {
            Some(slot) => slot.1 = mi,
            None => root_mount.push((rid, mi)),
        }
    }
    let mounts: Vec<MountInfo> = raw_mounts
        .iter()
        .enumerate()
        .map(|(n, m)| {
            let roots = root_mount
                .iter()
                .filter(|(_, mi)| *mi == Some(n))
                .map(|(rid, _)| rid.clone())
                .collect();
            mount_info(m, data, &book, &labels, roots)
        })
        .collect();
    let shares = shares(fleet, &book);
    let mut roots = Vec::new();
    for r in &root_specs {
        let rid = s(r.get("id")).unwrap_or_default();
        let mi = root_mount
            .iter()
            .find(|(k, _)| *k == rid)
            .and_then(|(_, mi)| *mi);
        let (shared_from, visible) = shares
            .iter()
            .find(|((h, k), _)| h == host_id && *k == rid)
            .map_or((None, vec![host_id.to_string()]), |(_, v)| v.clone());
        roots.push(RootInfo {
            id: rid,
            path: r
                .get("path")
                .map(ec_proto::pyfmt::str_value)
                .unwrap_or_default(),
            kind: s(r.get("kind")).unwrap_or_else(|| "auto".into()),
            writable: is_true(r.get("writable")),
            label: s(r.get("label")),
            mount: mi.and_then(|n| mounts.get(n).cloned()),
            shared_from,
            visible_on: visible,
        });
    }
    Some(HostDetail {
        summary,
        disks: mounts,
        roots,
        images,
        scripts: build_scripts(data),
        addresses: addresses_of(data.hello.as_deref()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normpath_matches_posixpath() {
        for (p, want) in [
            ("/a//b/./c/", "/a/b/c"),
            ("/a/b/../c", "/a/c"),
            ("/../a", "/a"),
            ("//a", "//a"),
            ("///a", "/a"),
            ("/", "/"),
            ("a/../..", ".."),
            ("", "."),
        ] {
            assert_eq!(normpath(p), want, "{p}");
        }
    }

    #[test]
    fn test_relpath_under() {
        assert_eq!(relpath_under("/data/hf", "/data/hf"), ".");
        assert_eq!(relpath_under("/data/hf/x", "/data"), "hf/x");
        assert_eq!(relpath_under("/x/y", "/"), "x/y");
    }
}
