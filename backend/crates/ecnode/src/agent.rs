//! Python: node/src/ecnode/agent.py.
//!
//! The node's own view of its machine, independent of the wire: hello, sample, disks, inventory, metrics. The stdio
//! server (server.rs) pushes these on their schedules and calls the op handlers (ops.rs) with this agent. Every
//! collector degrades instead of failing: no GPUs, no docker or a dead mount give empty or None fields and a human
//! `error`, never a panic out of a push loop.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use ec_proto::Obj;
use serde_json::{Value, json};

use crate::discovery::{
    self, GetOpts, MetricsGet, Prober, SLOW_METRICS, SLOW_METRICS_READ_S, SLOW_METRICS_TOTAL_S,
    View, build_endpoints_with, container_view, script_status,
};
use crate::disks::{Disks, StatvfsFn};
use crate::docker::{Docker, effective_context};
use crate::errors::OpError;
use crate::fsops::WriteLocks;
use crate::gpus::Gpus;
use crate::hostinfo::{Paths, addresses, host_facts};
use crate::memory::Mem;
use crate::models::{Scanner, ScannerOptions};
use crate::policy::{Policy, PolicyStore};
use crate::scripts::Scripts;
use crate::util::sync::{lock, wait_timeout_while};
use crate::util::{EnvMap, ProcRunner, Runner, json_size, node_env, py};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const METRICS_RAW_MAX: usize = 8 * 1024 * 1024;
/// bytes of JSON: one inventory must fit one frame (MAX_JSON is 1 MiB)
pub const INVENTORY_BUDGET: usize = 900 * 1024;

/// `metrics.configure` state (agent.py `MetricsConf`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MetricsConf {
    /// None: not configured, nothing is scraped
    pub interval_s: Option<f64>,
    pub families: Vec<(String, Vec<String>)>,
}

/// Callback the model scanner calls with each new generation (server.rs pushes `models_changed`).
pub type ScanCallback = Box<dyn Fn(i64) + Send + Sync>;

/// The latest inventory and the node-internal facts derived from it (agent.py `Inventory`).
#[derive(Debug, Clone, Default)]
pub struct Inventory {
    /// proto INVENTORY without "type"
    pub msg: Obj,
    pub views: Vec<View>,
    pub names_by_id: HashMap<String, String>,
    /// (name, init pid) of running engine containers
    pub engines: Vec<(String, i64)>,
    pub endpoint_ports: HashSet<u16>,
    pub gateway_ports: HashSet<u16>,
    /// host ports that running containers publish on a loopback-reachable address (all of their `loopback_ports`)
    pub published_ports: HashSet<u16>,
    pub error: Option<String>,
}

impl Inventory {
    pub fn empty() -> Self {
        let mut msg = Obj::new();
        msg.insert("t".into(), Value::Null);
        msg.insert("containers".into(), json!([]));
        msg.insert("endpoints".into(), json!([]));
        msg.insert("scripts".into(), json!([]));
        Self {
            msg,
            ..Self::default()
        }
    }
}

/// Keep the samples (and the # HELP / # TYPE lines) of metric families whose name starts with a prefix.
pub fn filter_metrics(text: &str, prefixes: &[String], cap: usize) -> String {
    let mut keep: Vec<&str> = Vec::new();
    let mut size = 0usize;
    for line in py::splitlines(text) {
        let name: String = if line.starts_with("# HELP ") || line.starts_with("# TYPE ") {
            line.split(' ').nth(2).unwrap_or("").to_string()
        } else if py::strip(line).is_empty() || line.starts_with('#') {
            continue;
        } else {
            let before_brace = line.split('{').next().unwrap_or("");
            before_brace.split(' ').next().unwrap_or("").to_string()
        };
        if prefixes.iter().any(|p| name.starts_with(p.as_str())) {
            let n = py::len(line);
            if size + n + 1 > cap {
                break;
            }
            keep.push(line);
            size += n + 1;
        }
    }
    if keep.is_empty() {
        String::new()
    } else {
        format!("{}\n", keep.join("\n"))
    }
}

fn truthy(v: Option<&Value>) -> bool {
    v.is_some_and(py::truthy)
}

/// An inventory that fits one frame: on a machine with very many or very verbose containers, first the argv and
/// labels of containers that are not engines are trimmed, then those containers are left out, then every argv is cut
/// short. Engines and console containers stay listed.
pub fn shrink_inventory(mut msg: Obj, budget: usize) -> Obj {
    let size = |m: &Obj| json_size(&Value::Object(m.clone()));
    if size(&msg) <= budget {
        return msg;
    }
    let is_kept = |c: &Value| truthy(c.get("engine")) || truthy(c.get("managed"));
    if let Some(Value::Array(cs)) = msg.get_mut("containers") {
        for c in cs.iter_mut() {
            if is_kept(c) {
                continue;
            }
            if let Value::Object(o) = c {
                let argv: Vec<Value> = o
                    .get("argv")
                    .and_then(Value::as_array)
                    .map(|a| a.iter().take(8).cloned().collect())
                    .unwrap_or_default();
                o.insert("argv".into(), Value::Array(argv));
                o.insert("params".into(), json!({}));
                o.insert("labels".into(), json!({}));
            }
        }
    }
    if size(&msg) > budget
        && let Some(Value::Array(cs)) = msg.get_mut("containers")
    {
        cs.retain(is_kept);
    }
    for cap in [100usize, 20] {
        if size(&msg) <= budget {
            break;
        }
        if let Some(Value::Array(cs)) = msg.get_mut("containers") {
            for c in cs.iter_mut() {
                if let Some(Value::Array(argv)) = c.get_mut("argv") {
                    let cut: Vec<Value> = argv
                        .iter()
                        .take(cap)
                        .map(|a| json!(py::take(a.as_str().unwrap_or(""), 256)))
                        .collect();
                    *argv = cut;
                }
            }
        }
    }
    msg
}

/// How an agent reaches its machine (agent.py constructor arguments). `Default` is the real machine: /proc, /sys,
/// /etc, the node's own environment, NVML-then-nvidia-smi, real loopback probes.
pub struct AgentParts {
    pub paths: Paths,
    /// the node's own environment (children get `child_env` of it)
    pub env: EnvMap,
    pub runner: Arc<dyn Runner>,
    pub gpus: Option<Gpus>,
    pub statvfs: Option<StatvfsFn>,
    pub prober: Prober,
    pub quiet_prober: Prober,
    pub metrics_get: MetricsGet,
    pub port_free: Arc<dyn Fn(u16) -> bool + Send + Sync>,
    pub default_socket: String,
    pub secret_scan_max_entries: usize,
    pub scanner: ScannerOptions,
}

/// ops.py `port_free`: nothing listens on 127.0.0.1:<port>.
pub fn port_free(port: u16) -> bool {
    std::net::TcpListener::bind(("127.0.0.1", port)).is_ok()
}

impl Default for AgentParts {
    fn default() -> Self {
        let env = node_env();
        Self {
            paths: Paths::default(),
            runner: Arc::new(ProcRunner::new(env.clone())),
            env,
            gpus: None,
            statvfs: None,
            prober: Arc::new(discovery::probe_port),
            quiet_prober: Arc::new(discovery::probe_port_quiet),
            metrics_get: Arc::new(|port, path, o: &GetOpts| discovery::http_get(port, path, o)),
            port_free: Arc::new(port_free),
            default_socket: crate::docker::DEFAULT_SOCKET.into(),
            secret_scan_max_entries: crate::docker::SECRET_SCAN_MAX_ENTRIES,
            scanner: ScannerOptions::default(),
        }
    }
}

/// (docker binary, configured context) -> the effective context
type ContextCache = HashMap<(String, Option<String>), Option<String>>;

pub struct Agent {
    pub policies: Arc<PolicyStore>,
    pub paths: Paths,
    pub env: EnvMap,
    pub runner: Arc<dyn Runner>,
    pub gpus: Gpus,
    pub mem: Mem,
    pub disk: Disks,
    pub prober: Prober,
    pub quiet_prober: Prober,
    pub metrics_get: MetricsGet,
    pub scripts: Scripts,
    pub write_locks: WriteLocks,
    pub port_free: Arc<dyn Fn(u16) -> bool + Send + Sync>,
    pub default_socket: String,
    pub secret_scan_max_entries: usize,
    scanner_opts: ScannerOptions,
    scanner: Mutex<Option<Arc<Scanner>>>,
    metrics_conf: Mutex<MetricsConf>,
    gpu_static: Mutex<Vec<Value>>,
    inventory: Mutex<Arc<Inventory>>,
    first_inventory: (Mutex<bool>, Condvar),
    cpu_prev: Mutex<HashMap<String, (Instant, i64)>>,
    contexts: Mutex<ContextCache>,
}

impl Agent {
    /// The node on this machine.
    pub fn new(policies: PolicyStore) -> Self {
        Self::with_parts(policies, AgentParts::default())
    }

    pub fn with_parts(policies: PolicyStore, parts: AgentParts) -> Self {
        let gpus = parts
            .gpus
            .unwrap_or_else(|| Gpus::for_env(&parts.env, Arc::clone(&parts.runner)));
        let disk = match parts.statvfs {
            Some(s) => Disks::with_statvfs(parts.paths.clone(), s),
            None => Disks::new(parts.paths.clone()),
        };
        Self {
            policies: Arc::new(policies),
            mem: Mem::new(parts.paths.clone()),
            disk,
            paths: parts.paths,
            env: parts.env,
            runner: parts.runner,
            gpus,
            prober: parts.prober,
            quiet_prober: parts.quiet_prober,
            metrics_get: parts.metrics_get,
            scripts: Scripts::new(),
            write_locks: WriteLocks::new(),
            port_free: parts.port_free,
            default_socket: parts.default_socket,
            secret_scan_max_entries: parts.secret_scan_max_entries,
            scanner_opts: parts.scanner,
            scanner: Mutex::new(None),
            metrics_conf: Mutex::new(MetricsConf::default()),
            gpu_static: Mutex::new(Vec::new()),
            inventory: Mutex::new(Arc::new(Inventory::empty())),
            first_inventory: (Mutex::new(false), Condvar::new()),
            cpu_prev: Mutex::new(HashMap::new()),
            contexts: Mutex::new(HashMap::new()),
        }
    }

    pub fn policy(&self) -> Arc<Policy> {
        self.policies.current()
    }

    /// A docker CLI for the policy (its context resolved once per binary and configured context).
    pub fn docker_for(&self, policy: &Policy) -> Docker {
        let key = (policy.docker_binary.clone(), policy.docker_context.clone());
        let cached = lock(&self.contexts).get(&key).cloned();
        let ctx = match cached {
            Some(c) => c,
            None => {
                let c = effective_context(
                    &key.0,
                    key.1.as_deref(),
                    self.runner.as_ref(),
                    &self.env,
                    &self.default_socket,
                );
                lock(&self.contexts).insert(key, c.clone());
                c
            }
        };
        Docker::new(&policy.docker_binary, ctx, Arc::clone(&self.runner))
    }

    pub fn docker(&self) -> Docker {
        self.docker_for(&self.policy())
    }

    pub fn gpu_uuids(&self) -> HashSet<String> {
        lock(&self.gpu_static)
            .iter()
            .filter_map(|g| g.get("uuid").and_then(Value::as_str))
            .filter(|u| !u.is_empty())
            .map(str::to_string)
            .collect()
    }

    pub fn gpu_by_index(&self) -> HashMap<String, String> {
        lock(&self.gpu_static)
            .iter()
            .filter_map(|g| {
                let idx = g.get("index").filter(|i| !i.is_null())?;
                let uuid = g
                    .get("uuid")
                    .and_then(Value::as_str)
                    .filter(|u| !u.is_empty())?;
                Some((py::str_of(idx), uuid.to_string()))
            })
            .collect()
    }

    pub fn inventory(&self) -> Arc<Inventory> {
        Arc::clone(&lock(&self.inventory))
    }

    // ── hello ────────────────────────────────────────────────────────────────────────────────────────────
    /// The `hello` message, `"type": "hello"` included (`reload`: re-read the policy file first).
    pub fn hello(&self, reload: bool) -> Obj {
        let policy = if reload {
            self.policies.reload()
        } else {
            self.policy()
        };
        let d = self.docker_for(&policy);
        let ((gpus, gpu_info), docker, addrs, host) = std::thread::scope(|s| {
            let g = s.spawn(|| self.gpus.static_info());
            let dk = s.spawn(|| d.facts());
            let ad = s.spawn(|| addresses(self.runner.as_ref(), &self.paths));
            let host = host_facts(&self.paths, self.runner.as_ref());
            (
                g.join().unwrap_or_else(|_| {
                    (
                        Vec::new(),
                        json!({"driver": null, "cuda": null, "source": "none", "error": "panic"}),
                    )
                }),
                dk.join().unwrap_or_default(),
                ad.join().unwrap_or_default(),
                host,
            )
        });
        *lock(&self.gpu_static) = gpus.clone();
        let mut m = Obj::new();
        m.insert("type".into(), json!("hello"));
        m.insert("proto".into(), json!(ec_proto::PROTO));
        m.insert("version".into(), json!(VERSION));
        m.insert(
            "node_id".into(),
            json!(crate::hostinfo::node_id(&self.paths)),
        );
        m.insert("host".into(), Value::Object(host));
        m.insert("gpu".into(), gpu_info);
        m.insert("gpus".into(), Value::Array(gpus));
        m.insert("docker".into(), Value::Object(docker));
        m.insert("addresses".into(), Value::Array(addrs));
        m.insert("policy".into(), Value::Object(policy.summary()));
        m.insert(
            "forced_cmd".into(),
            json!(self.env.contains_key("SSH_ORIGINAL_COMMAND")),
        );
        m.insert(
            "via_ssh".into(),
            json!(self.env.contains_key("SSH_CONNECTION")),
        );
        m
    }

    // ── sample ───────────────────────────────────────────────────────────────────────────────────────────
    /// One `sample` message (`"type"` included).
    pub fn sample(&self, seq: i64) -> Obj {
        let (stats, procs) = self.gpus.sample();
        let inv = self.inventory();
        let mut gpus = Obj::new();
        for (uuid, s) in stats {
            let mut g = s;
            g.insert("procs".into(), json!([]));
            gpus.insert(uuid, Value::Object(g));
        }
        for (uuid, pid, used) in procs {
            let Some(Value::Object(g)) = gpus.get_mut(&uuid) else {
                continue;
            };
            let Some(Value::Array(list)) = g.get_mut("procs") else {
                continue;
            };
            if list.len() >= 32 {
                continue;
            }
            let cid = self.mem.container_id_of(pid);
            let container = cid.and_then(|c| inv.names_by_id.get(&c).cloned());
            list.push(json!({"pid": pid, "used_bytes": used, "name": self.mem.comm(pid), "container": container}));
        }
        for g in gpus.values_mut() {
            if let Some(Value::Array(list)) = g.get_mut("procs") {
                list.sort_by_key(|p| {
                    std::cmp::Reverse(p.get("used_bytes").and_then(Value::as_i64).unwrap_or(0))
                });
            }
        }
        let (l1, l5, l15) = self.mem.loadavg();
        let mut containers = Obj::new();
        let now_m = Instant::now();
        {
            let mut prev_map = lock(&self.cpu_prev);
            for (name, pid) in inv.engines.iter().take(256) {
                let usage = self.mem.container_cpu_usec(*pid);
                let prev = prev_map.get(name).copied();
                let mut cpu_pct: Option<f64> = None;
                if let (Some(u), Some((pt, pu))) = (usage, prev)
                    && now_m > pt
                    && u >= pu
                {
                    let dt = now_m.duration_since(pt).as_secs_f64();
                    cpu_pct = Some(py::round((u - pu) as f64 / 1e6 / dt * 100.0, 1));
                }
                if let Some(u) = usage {
                    prev_map.insert(name.clone(), (now_m, u));
                }
                containers.insert(
                    name.clone(),
                    json!({"mem": self.mem.container_mem(*pid), "cpu_pct": cpu_pct}),
                );
            }
            prev_map.retain(|k, _| containers.contains_key(k));
        }
        let mut m = Obj::new();
        m.insert("type".into(), json!("sample"));
        m.insert("seq".into(), json!(seq));
        m.insert("t".into(), json!(now3()));
        m.insert("gpus".into(), Value::Object(gpus));
        m.insert("mem".into(), Value::Object(self.mem.meminfo()));
        m.insert(
            "cpu".into(),
            json!({"load1": l1, "load5": l5, "load15": l15, "util_pct": self.mem.cpu_util()}),
        );
        m.insert("containers".into(), Value::Object(containers));
        m
    }

    // ── disks ────────────────────────────────────────────────────────────────────────────────────────────
    /// One `disks` message (`"type"` included).
    pub fn disks(&self) -> Obj {
        let mut m = Obj::new();
        m.insert("type".into(), json!("disks"));
        m.extend(self.disk.report(&self.policy().model_roots));
        m
    }

    // ── inventory ────────────────────────────────────────────────────────────────────────────────────────
    /// A fresh inventory (agent.py `refresh_inventory`). Marks the first inventory as known.
    pub fn refresh(&self) -> Arc<Inventory> {
        let policy = self.policy();
        let d = self.docker_for(&policy);
        let mut views: Vec<View> = Vec::new();
        let mut error: Option<String> = None;
        let collect = || -> Result<Vec<View>, OpError> {
            let mut rows: Vec<(String, String)> = Vec::new(); // id -> status, first position kept
            for r in d
                .ps(false, None)?
                .into_iter()
                .chain(d.ps(true, Some("engine-console=1"))?)
            {
                match rows.iter_mut().find(|(id, _)| *id == r.id) {
                    Some(slot) => slot.1 = r.status,
                    None => rows.push((r.id, r.status)),
                }
            }
            let roots: Vec<_> = policy
                .model_roots
                .iter()
                .map(|r| {
                    (
                        r.clone(),
                        std::path::PathBuf::from(self.disk.resolve(&r.path)),
                    )
                })
                .collect();
            let by_index = self.gpu_by_index();
            let ids: Vec<String> = rows.iter().take(256).map(|(id, _)| id.clone()).collect();
            let mut out = Vec::new();
            for doc in d.inspect(&ids)? {
                let id = match doc.get("Id") {
                    Some(v) if py::truthy(v) => py::str_of(v),
                    _ => String::new(),
                };
                let status = rows.iter().find(|(i, _)| *i == id).map(|(_, s)| s.as_str());
                if let Some(v) = container_view(&doc, &roots, &by_index, status) {
                    out.push(v);
                }
            }
            Ok(out)
        };
        match collect() {
            Ok(v) => views = v,
            Err(e) => error = Some(e.message),
        }
        // no usable policy (no file, or a refused one): probe only ports containers publish (security audit F1)
        let endpoints = build_endpoints_with(
            &views,
            &policy.discovery_ports,
            &self.prober,
            8,
            &self.quiet_prober,
            policy.error.is_some(),
        );
        let now = now3();
        let running: Vec<String> = views
            .iter()
            .filter(|v| v.running)
            .map(|v| info_str(&v.info, "name"))
            .collect();
        let mut msg = Obj::new();
        msg.insert("t".into(), json!(now));
        msg.insert(
            "containers".into(),
            Value::Array(
                views
                    .iter()
                    .map(|v| Value::Object(v.info.clone()))
                    .collect(),
            ),
        );
        msg.insert(
            "endpoints".into(),
            Value::Array(endpoints.iter().cloned().map(Value::Object).collect()),
        );
        msg.insert(
            "scripts".into(),
            Value::Array(
                script_status(&policy, &running, now)
                    .into_iter()
                    .map(Value::Object)
                    .collect(),
            ),
        );
        let msg = shrink_inventory(msg, INVENTORY_BUDGET);
        let mut gateway_ports = HashSet::new();
        for v in &views {
            if v.running
                && truthy(v.info.get("managed"))
                && v.info.get("role").and_then(Value::as_str) == Some("gateway")
            {
                let p = v
                    .published_port(8080)
                    .or_else(|| v.loopback_ports.first().copied())
                    .unwrap_or(0);
                if p != 0 {
                    gateway_ports.insert(p);
                }
            }
        }
        let names_by_id = views
            .iter()
            .filter_map(|v| {
                let id = info_str(&v.info, "id");
                (!id.is_empty()).then(|| (id, info_str(&v.info, "name")))
            })
            .collect();
        // cgroup memory is sampled for every running engine-classified or console-owned container
        let engines = views
            .iter()
            .filter(|v| {
                v.running
                    && v.pid != 0
                    && (truthy(v.info.get("engine")) || truthy(v.info.get("managed")))
            })
            .map(|v| (info_str(&v.info, "name"), v.pid))
            .collect();
        let endpoint_ports = endpoints
            .iter()
            .filter_map(|e| e.get("port").and_then(Value::as_i64))
            .filter_map(|p| u16::try_from(p).ok())
            .collect();
        let published_ports = views
            .iter()
            .filter(|v| v.running)
            .flat_map(|v| v.loopback_ports.iter().copied())
            .collect();
        let inv = Arc::new(Inventory {
            msg,
            views,
            names_by_id,
            engines,
            endpoint_ports,
            gateway_ports,
            published_ports,
            error,
        });
        *lock(&self.inventory) = Arc::clone(&inv);
        let (l, cv) = &self.first_inventory;
        *lock(l) = true;
        cv.notify_all();
        inv
    }

    /// A fresh inventory: the `inventory` message WITHOUT "type" (server.rs adds it).
    pub fn refresh_inventory(&self) -> Obj {
        self.refresh().msg.clone()
    }

    /// Wait (at most `timeout`) until the first inventory is known.
    pub fn wait_first_inventory(&self, timeout: Duration) {
        let (l, cv) = &self.first_inventory;
        drop(wait_timeout_while(cv, lock(l), timeout, |ready| !*ready));
    }

    /// The ports `http.request` may reach: engine endpoints and console gateways of the inventory. Without a usable
    /// policy only those a running container publishes (security audit F1): never a bare port that merely answered,
    /// also when the inventory at hand was built while the policy was still valid.
    pub fn allowed_ports(&self, refresh: bool) -> HashSet<u16> {
        let inv = if refresh {
            self.refresh()
        } else {
            self.inventory()
        };
        let published_only = self.policy().error.is_some();
        inv.endpoint_ports
            .union(&inv.gateway_ports)
            .copied()
            .filter(|p| !published_only || inv.published_ports.contains(p))
            .collect()
    }

    /// Host paths that running containers mount, read fresh from docker (for fs.delete's in_use check).
    pub fn running_mounts(&self) -> Result<Vec<String>, OpError> {
        let d = self.docker();
        let res = d.call(&["ps", "-q", "--no-trunc"], 20.0, None);
        if res.rc == 127 {
            return Ok(Vec::new()); // no docker CLI on this machine: nothing can be running in a container
        }
        if !res.ok() {
            return Err(OpError::new(
                ec_proto::DOCKER_ERROR,
                "cannot ask docker which containers are running, so cannot tell whether the model is in use; try \
                 again",
            ));
        }
        let ids: Vec<String> = py::split_ws(&res.out)
            .into_iter()
            .take(512)
            .map(str::to_string)
            .collect();
        let mut out = Vec::new();
        for doc in d.inspect(&ids)? {
            for m in doc
                .get("Mounts")
                .and_then(Value::as_array)
                .map(Vec::as_slice)
                .unwrap_or_default()
            {
                if let Some(src) = m.get("Source").and_then(Value::as_str)
                    && src.starts_with('/')
                {
                    out.push(src.to_string());
                }
            }
        }
        Ok(out)
    }

    // ── metrics ──────────────────────────────────────────────────────────────────────────────────────────
    /// `metrics.configure`: interval clamped 1-60 s (10 s when not given), empty prefixes dropped.
    pub fn configure_metrics(&self, interval_s: Option<f64>, families: Vec<(String, Vec<String>)>) {
        let interval = interval_s.map_or(10.0, |v| v.clamp(1.0, 60.0));
        let families = families
            .into_iter()
            .map(|(k, v)| (k, v.into_iter().filter(|p| !p.is_empty()).collect()))
            .collect();
        *lock(&self.metrics_conf) = MetricsConf {
            interval_s: Some(interval),
            families,
        };
    }

    pub fn metrics_conf(&self) -> MetricsConf {
        lock(&self.metrics_conf).clone()
    }

    /// The metrics loop's period source: the configured interval, None when not configured.
    pub fn metrics_interval(&self) -> Option<f64> {
        self.metrics_conf().interval_s
    }

    /// One `metrics` message per scraped engine endpoint.
    pub fn scrape(&self) -> Vec<Obj> {
        let conf = self.metrics_conf();
        if conf.interval_s.is_none() {
            return Vec::new();
        }
        let family = |k: &str| {
            conf.families
                .iter()
                .find(|(n, _)| n == k)
                .map(|(_, p)| p.clone())
                .filter(|p| !p.is_empty())
        };
        let inv = self.inventory();
        let mut out = Vec::new();
        for ep in inv
            .msg
            .get("endpoints")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or_default()
        {
            let engine = ep.get("engine").and_then(Value::as_str).unwrap_or("");
            let Some(prefixes) = family(engine).or_else(|| family("*")) else {
                continue;
            };
            let slow = SLOW_METRICS.contains(&engine);
            let status_key = if slow {
                "health_status"
            } else {
                "metrics_status"
            };
            if ep.get(status_key).and_then(Value::as_i64) != Some(200) {
                continue;
            }
            let Some(port) = ep
                .get("port")
                .and_then(Value::as_i64)
                .and_then(|p| u16::try_from(p).ok())
            else {
                continue;
            };
            let mut o = GetOpts {
                max_body: METRICS_RAW_MAX,
                ..GetOpts::default()
            };
            if slow {
                o.read_s = SLOW_METRICS_READ_S;
                o.total_s = SLOW_METRICS_TOTAL_S;
            }
            let probe = (self.metrics_get)(port, "/metrics", &o);
            let text = if probe.ok() {
                String::from_utf8_lossy(&probe.body).into_owned()
            } else {
                String::new()
            };
            let mut m = Obj::new();
            m.insert("type".into(), json!("metrics"));
            m.insert("t".into(), json!(now3()));
            m.insert(
                "port".into(),
                ep.get("port").cloned().unwrap_or(Value::Null),
            );
            m.insert(
                "container".into(),
                ep.get("container").cloned().unwrap_or(Value::Null),
            );
            m.insert(
                "engine".into(),
                ep.get("engine").cloned().unwrap_or(Value::Null),
            );
            m.insert("status".into(), json!(probe.status));
            m.insert(
                "text".into(),
                json!(filter_metrics(&text, &prefixes, ec_proto::MAX_METRICS_TEXT)),
            );
            out.push(m);
        }
        out
    }

    // ── the model scanner ────────────────────────────────────────────────────────────────────────────────
    /// Start the model scanner (server.rs; Python: `agent.scanner = Scanner(...); scanner.start()`).
    pub fn start_scanner(&self, on_change: ScanCallback) -> bool {
        let store = Arc::clone(&self.policies);
        let sc = Scanner::new(
            Arc::new(move || store.current()),
            Arc::from(on_change),
            self.scanner_opts.clone(),
        );
        sc.start();
        *lock(&self.scanner) = Some(sc);
        true
    }

    pub fn stop_scanner(&self) {
        if let Some(sc) = lock(&self.scanner).as_ref() {
            sc.stop();
        }
    }

    /// The running scanner (ops' `_scanner(a)`), None when there is none.
    pub fn scanner(&self) -> Option<Arc<Scanner>> {
        lock(&self.scanner).clone()
    }

    /// Install a scanner without starting it (tests).
    pub fn set_scanner(&self, sc: Option<Arc<Scanner>>) {
        *lock(&self.scanner) = sc;
    }
}

fn info_str(info: &Obj, k: &str) -> String {
    info.get(k)
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string()
}

/// `round(time.time(), 3)`
pub fn now3() -> f64 {
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);
    py::round(t, 3)
}

/// First 16 hex of sha256(/etc/machine-id) (hostinfo.py `node_id` on the real /etc).
pub fn node_id() -> String {
    crate::hostinfo::node_id(&Paths::default())
}

pub fn hostname() -> String {
    crate::hostinfo::hostname()
}

/// `platform.machine()` names.
pub fn arch() -> &'static str {
    std::env::consts::ARCH
}
