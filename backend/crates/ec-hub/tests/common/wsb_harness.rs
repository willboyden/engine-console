//! Shared helpers for the fleet tests (port of backend/tests/fleet/harness.py, conftest.py and builders.py).
//! Needs `mod wsb_fake_node;` at the test crate's root (and `fake_node_entry!()` there).

#![allow(
    dead_code,
    unreachable_pub,
    clippy::panic,
    clippy::type_complexity,
    clippy::expect_used,
    clippy::needless_pass_by_value
)]

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ec_hub::config::Settings;
use ec_hub::domain::fleet::{HostConfig, LinkStatus, NodeClient, NodeStream, StreamItem};
use ec_hub::services::common::EventBus;
use ec_hub::services::fleet::{ArgvFn, LinkLike, LinkTiming, NodeLink};
use ec_hub::services::fleet_status::HostData;
use ec_proto::Obj;
use serde_json::{Value, json};

use crate::wsb_fake_node::{node_argv, read_log};

pub const GIB: i64 = 1024 * 1024 * 1024;
pub const MIB: usize = 1024 * 1024;
pub const TEST_TIMEOUT_S: f64 = 45.0;

pub fn host(id: &str, label: &str, ssh: Option<&str>) -> HostConfig {
    HostConfig {
        id: id.into(),
        label: label.into(),
        ssh: ssh.map(str::to_string),
        port: None,
        ssh_config: false,
        enabled: true,
    }
}

pub fn local_host() -> HostConfig {
    host("local", "This machine", None)
}

pub fn remote_host() -> HostConfig {
    host("gpu-box-2", "GPU box 2", Some("ops@10.0.0.5"))
}

pub fn env() -> Vec<(String, String)> {
    vec![
        ("PATH".into(), "/usr/bin:/bin".into()),
        ("LANG".into(), "C.UTF-8".into()),
    ]
}

/// Short everything, so a drop, a silence or a backoff takes milliseconds. `fix_retry_s` is deliberately unlike any
/// backoff value (<= 0.4) so a test can tell "needs a person" waits from ordinary ones.
pub fn fast() -> LinkTiming {
    LinkTiming {
        hello_timeout_s: 5.0,
        ping_s: 0.05,
        silent_s: 2.0,
        stale_s: 3.0,
        backoff_min_s: 0.05,
        backoff_max_s: 0.4,
        stable_s: 30.0,
        fix_retry_s: 9.0,
        flood_bytes: 500 * 1024 * 1024,
        flood_window_s: 10.0,
        flood_frames: 1_000_000,
        exit_grace_s: 1.0,
        max_stream_events: 10_000,
    }
}

pub fn obj(v: Value) -> Obj {
    v.as_object().cloned().unwrap_or_default()
}

/// Poll `pred` until it is true; fail with `what` after `timeout_s`.
pub async fn wait_until(pred: impl Fn() -> bool, timeout_s: f64, what: &str) {
    let end = Instant::now() + Duration::from_secs_f64(timeout_s);
    while !pred() {
        assert!(
            Instant::now() <= end,
            "timed out after {timeout_s} s waiting for {what}"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// Poll `get` until it returns Some.
pub async fn wait_for<T>(get: impl Fn() -> Option<T>, timeout_s: f64, what: &str) -> T {
    let end = Instant::now() + Duration::from_secs_f64(timeout_s);
    loop {
        if let Some(v) = get() {
            return v;
        }
        assert!(
            Instant::now() <= end,
            "timed out after {timeout_s} s waiting for {what}"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// Is that process still running? A zombie (exited, not yet reaped) still answers signal 0, so look at its state.
pub fn alive(pid: i64) -> bool {
    let Ok(p) = i32::try_from(pid) else {
        return false;
    };
    if nix::sys::signal::kill(nix::unistd::Pid::from_raw(p), None).is_err() {
        return false;
    }
    std::fs::read_to_string(format!("/proc/{pid}/stat"))
        .ok()
        .and_then(|s| {
            s.rsplit_once(')')
                .and_then(|(_, rest)| rest.split_whitespace().next().map(str::to_string))
        })
        .is_some_and(|st| st != "Z")
}

/// Every item of a stream, to its end.
pub async fn items(s: &mut Box<dyn NodeStream>) -> Vec<StreamItem> {
    let mut out = Vec::new();
    while let Some(i) = s.next().await {
        out.push(i);
    }
    out
}

pub fn event(i: &StreamItem) -> Option<&Obj> {
    match i {
        StreamItem::Event(e) => Some(e),
        StreamItem::Data(_) => None,
    }
}

/// A started NodeLink, its bus and the fake node's log.
pub struct Rig {
    pub link: Arc<NodeLink>,
    pub bus: Arc<EventBus>,
    pub log_path: PathBuf,
    /// (state, the wait chosen before the next attempt, or None when none is pending) per change
    pub changes: Arc<Mutex<Vec<(String, Option<f64>)>>>,
}

impl Rig {
    pub fn log(&self) -> Vec<Value> {
        read_log(&self.log_path)
    }

    pub fn events(&self, ev: &str) -> Vec<Value> {
        self.log().into_iter().filter(|e| e["ev"] == ev).collect()
    }

    pub fn st(&self) -> LinkStatus {
        self.link.link()
    }

    pub fn error(&self) -> String {
        self.st().error.unwrap_or_default()
    }

    pub async fn up(&self) {
        let l = Arc::clone(&self.link);
        wait_until(move || l.link().state == "up", 8.0, "the link to come up").await;
    }

    /// Wait for a link state. For `down` that means the SETTLED one, which carries the retry delay.
    pub async fn state(&self, state: &str) {
        self.state_within(state, 8.0).await;
    }

    pub async fn state_within(&self, state: &str, timeout_s: f64) {
        let l = Arc::clone(&self.link);
        let want = state.to_string();
        wait_until(
            move || {
                let st = l.link();
                st.state == want && (want != "down" || st.retry_in_s.is_some())
            },
            timeout_s,
            &format!("state {state} (is {:?})", self.link.link().state),
        )
        .await;
    }

    pub fn waits(&self) -> Vec<f64> {
        self.changes
            .lock()
            .expect("lock")
            .iter()
            .filter(|(s, w)| s == "down" && w.is_some())
            .filter_map(|(_, w)| *w)
            .collect()
    }

    pub fn pids(&self) -> Vec<i64> {
        self.events("start")
            .iter()
            .filter_map(|e| e["pid"].as_i64())
            .collect()
    }

    pub async fn call(
        &self,
        op: &str,
        args: Value,
    ) -> Result<Obj, ec_hub::domain::errors::NodeError> {
        self.link
            .call(op, Some(obj(args)), Duration::from_secs(30))
            .await
    }
}

/// What `make_rig` takes (Python keyword arguments).
pub struct RigOpts {
    pub host: HostConfig,
    pub timing: LinkTiming,
    pub argv: Option<ArgvFn>,
    pub env: Option<Vec<(String, String)>>,
    pub start: bool,
}

impl Default for RigOpts {
    fn default() -> Self {
        Self {
            host: local_host(),
            timing: fast(),
            argv: None,
            env: Some(env()),
            start: true,
        }
    }
}

/// Starts NodeLinks against the fake node (the real subprocess path) and stops them all at the end.
pub struct Harness {
    pub tmp: tempfile::TempDir,
    links: Vec<Arc<NodeLink>>,
}

impl Default for Harness {
    fn default() -> Self {
        Self::new()
    }
}

impl Harness {
    pub fn new() -> Self {
        Self {
            tmp: tempfile::tempdir().expect("tempdir"),
            links: Vec::new(),
        }
    }

    pub async fn rig(&mut self, scenario: Value) -> Rig {
        self.rig_with(scenario, RigOpts::default()).await
    }

    pub async fn rig_with(&mut self, scenario: Value, o: RigOpts) -> Rig {
        let n = self.links.len();
        let log_path = self.tmp.path().join(format!("node-{n}.jsonl"));
        let mut sc = json!({"log": log_path.to_string_lossy()});
        if let (Value::Object(m), Value::Object(extra)) = (&mut sc, scenario) {
            for (k, v) in extra {
                m.insert(k, v);
            }
        }
        let bus = Arc::new(EventBus::new());
        let argv = o
            .argv
            .unwrap_or_else(|| NodeLink::fixed_argv(node_argv(&sc)));
        let link = Arc::new(NodeLink::new(
            o.host,
            argv,
            o.env,
            Arc::clone(&bus),
            o.timing,
            None,
            Vec::new(),
        ));
        let changes = Arc::new(Mutex::new(Vec::new()));
        let weak = Arc::downgrade(&link);
        let ch = Arc::clone(&changes);
        link.set_on_change(Some(Arc::new(move || {
            if let Some(l) = weak.upgrade() {
                let wait = if l.retry_pending() {
                    l.last_wait_s()
                } else {
                    None
                };
                ch.lock().expect("lock").push((l.link().state, wait));
            }
        })));
        self.links.push(Arc::clone(&link));
        if o.start {
            link.start().await;
        }
        Rig {
            link,
            bus,
            log_path,
            changes,
        }
    }

    pub fn path(&self, name: &str) -> PathBuf {
        self.tmp.path().join(name)
    }

    pub async fn stop(&mut self) {
        for l in std::mem::take(&mut self.links) {
            l.stop().await;
        }
    }
}

impl Drop for Harness {
    fn drop(&mut self) {
        // a test that failed half-way must not leave nodes behind
        for l in &self.links {
            if let Some(pid) = l.pid() {
                let p = nix::unistd::Pid::from_raw(pid);
                let _ = nix::sys::signal::killpg(p, nix::sys::signal::Signal::SIGKILL);
            }
        }
    }
}

/// Run a test body with Python's 45 s hard limit (a regression in flow control must fail, not hang).
pub async fn limited<F: std::future::Future<Output = ()>>(f: F) {
    if tokio::time::timeout(Duration::from_secs_f64(TEST_TIMEOUT_S), f)
        .await
        .is_err()
    {
        panic!("the test did not finish within {TEST_TIMEOUT_S} s (hung?)");
    }
}

/// Settings whose every path is inside `dir`: nothing under the real home is ever touched (conftest `make_settings`).
pub fn make_settings(dir: &std::path::Path) -> Settings {
    let mut s = Settings::defaults();
    s.data_dir = dir.join("data");
    s.hf_cache_dir = dir.join("hf");
    s.hf_token_file = dir.join("no-token");
    s.hosts_file = dir.join("hosts.json");
    s.hub_key_file = dir.join("ssh").join("hub-key");
    s.node_policy = None;
    s.node_command = None;
    s.fleet_enabled = true;
    s.otlp_enabled = false;
    s.background_tasks = false;
    s.discovery_ports = Vec::new();
    s.hf_token = None;
    s.hf_endpoint = "https://127.0.0.1:9".into();
    s.remote_node_command = "/usr/local/lib/ecnode/ecnode --stdio".into();
    s
}

// ── builders.py: realistic (already sanitized) node messages and host data ─────────────────────────────────
fn sanitized(msg: Value) -> Obj {
    ec_proto::sanitize_push(&obj(msg))
        .expect("sanitize")
        .expect("known type")
}

/// builders.py `hello(addresses=, roots=, hostname=, **policy)`.
pub fn b_hello(addresses: &[&str], roots: Vec<Value>, hostname: &str, policy: Value) -> Obj {
    let mut pol = json!({"path": "/etc/ecnode/policy.json", "mode": "monitor", "engine_images": ["vllm/vllm-openai:*"],
           "gateway_image": "nginx@sha256:00", "engine_network": "engine-console-engines", "port_range": [18000, 18099],
           "discovery_ports": [8000], "model_roots": roots, "scripts": [], "allow_pull": false, "allow_delete": false,
           "allow_ipc_host": true, "external_control": "none", "max_shm_gib": 64});
    if let (Value::Object(p), Value::Object(extra)) = (&mut pol, policy) {
        for (k, v) in extra {
            p.insert(k, v);
        }
    }
    let addrs: Vec<&str> = if addresses.is_empty() {
        vec!["10.0.0.5"]
    } else {
        addresses.to_vec()
    };
    sanitized(json!({
        "type": "hello", "proto": 1, "version": "0.4.0", "node_id": "abc",
        "host": {"hostname": hostname, "arch": "aarch64", "os": "Ubuntu 24.04", "kernel": "6.8", "cpu_model": "Grace",
                 "cpu_cores": 20, "cpu_threads": 20, "ram_total_bytes": 128 * GIB, "boot_time": 1.0, "python": "3.12.3"},
        "gpu": {"driver": "570.1", "cuda": "12.8", "source": "nvml"},
        "gpus": [{"index": 1, "uuid": "GPU-B", "name": "RTX B", "total_bytes": 96 * GIB, "compute_capability": "12.0"},
                 {"index": 0, "uuid": "GPU-A", "name": "RTX A", "total_bytes": 48 * GIB, "compute_capability": "8.9"}],
        "docker": {"available": true, "context": "rootless", "rootless": true, "version": "27", "nvidia_runtime": true},
        "addresses": addrs.iter().map(|a| json!({"iface": "eth0", "addr": a, "prefix": 24})).collect::<Vec<_>>(),
        "policy": pol, "forced_cmd": true, "via_ssh": true}))
}

/// `hello()` with the defaults.
pub fn hello() -> Obj {
    b_hello(&[], Vec::new(), "gpu-box-2", json!({}))
}

/// builders.py `sample(**over)`.
pub fn sample(over: Value) -> Obj {
    let mut msg = json!({"type": "sample", "seq": 5, "t": 1000.0,
           "gpus": {"GPU-A": {"util_pct": 80.0, "mem_used_bytes": 40 * GIB, "mem_total_bytes": 48 * GIB, "temp_c": 61.0,
                              "power_w": 250.0, "power_limit_w": 300.0, "fan_pct": 44.0,
                              "procs": [{"pid": 11, "used_bytes": 30 * GIB, "container": "ec-engine-1", "name": "python3"},
                                        {"pid": 12, "used_bytes": 10 * GIB, "container": "other", "name": "x"},
                                        {"pid": 13, "name": "no-container"}]},
                    "GPU-B": {"util_pct": 0.0, "mem_used_bytes": GIB, "temp_c": 30.0}},
           "mem": {"total": 256 * GIB, "available": 200 * GIB, "free": 150 * GIB, "cached": 40 * GIB, "shmem": GIB,
                   "swap_total": 8 * GIB, "swap_free": 8 * GIB},
           "cpu": {"load1": 1.5, "load5": 1.0, "load15": 0.5, "util_pct": 12.0}, "containers": {}});
    if let (Value::Object(m), Value::Object(extra)) = (&mut msg, over) {
        for (k, v) in extra {
            m.insert(k, v);
        }
    }
    sanitized(msg)
}

pub fn disks(mounts: Vec<Value>) -> Obj {
    sanitized(json!({"type": "disks", "t": 1.0, "mounts": mounts}))
}

/// builders.py `mount(path, **kw)`.
pub fn mount(path: &str, kw: Value) -> Value {
    let mut m = json!({"mount": path, "fstype": "ext4", "size": 100 * GIB, "used": 40 * GIB, "free": 60 * GIB});
    if let (Value::Object(o), Value::Object(extra)) = (&mut m, kw) {
        for (k, v) in extra {
            o.insert(k, v);
        }
    }
    m
}

/// builders.py `root(rid, path, **kw)`.
pub fn root(rid: &str, path: &str, kw: Value) -> Value {
    let mut r = json!({"id": rid, "path": path, "kind": "hf-home", "writable": true, "label": format!("{rid} label")});
    if let (Value::Object(o), Value::Object(extra)) = (&mut r, kw) {
        for (k, v) in extra {
            o.insert(k, v);
        }
    }
    r
}

/// `str.title()` of an ASCII id ("gpu-box-2" -> "Gpu-Box-2").
fn title(s: &str) -> String {
    let mut out = String::new();
    let mut prev_alpha = false;
    for c in s.chars() {
        out.push(if prev_alpha {
            c.to_ascii_lowercase()
        } else {
            c.to_ascii_uppercase()
        });
        prev_alpha = c.is_alphabetic();
    }
    out
}

/// builders.py `data(hid, h=, s=, d=, inv=, state=, ssh=, label=, **link)`.
pub struct DataOpts {
    pub h: Option<Obj>,
    pub s: Option<Obj>,
    pub d: Option<Obj>,
    pub inv: Option<Obj>,
    pub state: String,
    pub ssh: Option<String>,
    pub label: Option<String>,
    pub error: Option<String>,
    pub skew_s: Option<f64>,
}

impl Default for DataOpts {
    fn default() -> Self {
        Self {
            h: None,
            s: None,
            d: None,
            inv: None,
            state: "up".into(),
            ssh: Some("ops@10.0.0.5".into()),
            label: None,
            error: None,
            skew_s: None,
        }
    }
}

pub fn data(hid: &str, o: DataOpts) -> HostData {
    let mut link = LinkStatus::new(&o.state, o.error);
    link.skew_s = o.skew_s;
    HostData {
        host: HostConfig {
            id: hid.into(),
            label: o.label.unwrap_or_else(|| title(hid)),
            ssh: if hid == "local" { None } else { o.ssh },
            port: None,
            ssh_config: false,
            enabled: true,
        },
        link,
        hello: o.h.map(Arc::new),
        sample: o.s.map(Arc::new),
        disks: o.d.map(Arc::new),
        inventory: o.inv.map(Arc::new),
    }
}

/// Keep the trait in scope for `link.start()` callers.
pub fn as_link(l: &Arc<NodeLink>) -> &dyn LinkLike {
    l.as_ref()
}
