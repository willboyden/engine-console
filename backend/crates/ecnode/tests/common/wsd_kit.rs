//! Node test kit: the Rust side of node/tests/conftest.py. Nothing here touches the real docker, GPUs, /proc, /sys
//! or network: the node gets an environment whose PATH holds only fake `docker`, `nvidia-smi` and `ip` programs
//! (tests/wsd_fakes/*.cjs, Node 22), machine trees live in temp dirs, and endpoint probes are refused for any port
//! that is not a fake engine's.
//!
//! Include with `#[path = "common/wsd_kit.rs"] mod wsd_kit;`.
#![allow(
    dead_code,
    unreachable_pub,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::needless_pass_by_value
)]

use std::collections::{HashSet, VecDeque};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use ec_proto::Obj;
use ecnode::discovery::{self, GetOpts, PortProbe, Probe};
use ecnode::disks::{StatvfsFn, VfsStat};
use ecnode::errors::{Cancelled, HandlerError, Stream};
use ecnode::hostinfo::Paths;
use ecnode::util::EnvMap;
use serde_json::{Value, json};

pub const GPU_A: &str = "GPU-aaaaaaaa-1111-2222-3333-444444444444";
pub const GPU_B: &str = "GPU-bbbbbbbb-1111-2222-3333-444444444444";
pub const CID_EXT: &str = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";
pub const GATEWAY_IMAGE: &str =
    "nginx@sha256:0000000000000000000000000000000000000000000000000000000000000000";
/// in every fake container's Config.Env: must never be reported
pub const ENV_VALUE: &str = "NEVER-LEAK-THIS-ENV-VALUE";
/// in a fake container's argv: must be redacted
pub const ARGV_SECRET: &str = "sk-test-not-a-real-key-0000";

pub fn cid_engine() -> String {
    "e".repeat(64)
}

pub fn obj(v: Value) -> Obj {
    match v {
        Value::Object(o) => o,
        other => panic!("not an object: {other}"),
    }
}

// ── the Node runtime for the fakes ─────────────────────────────────────────────────────────────────────
/// The `node` binary (Node 22): $ECNODE_TEST_NODE, else the first `node` on PATH, else ~/.nvm's newest.
pub fn node_bin() -> PathBuf {
    if let Some(p) = std::env::var_os("ECNODE_TEST_NODE") {
        return PathBuf::from(p);
    }
    if let Some(path) = std::env::var_os("PATH") {
        for d in std::env::split_paths(&path) {
            let c = d.join("node");
            if c.is_file() {
                return c;
            }
        }
    }
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_default();
    let nvm = home.join(".nvm").join("versions").join("node");
    let mut found: Vec<PathBuf> = std::fs::read_dir(&nvm)
        .map(|d| {
            d.filter_map(|e| e.ok().map(|e| e.path().join("bin").join("node")))
                .collect()
        })
        .unwrap_or_default();
    found.retain(|p| p.is_file());
    found.sort();
    if let Some(p) = found.pop() {
        return p;
    }
    for p in ["/usr/local/bin/node", "/usr/bin/node"] {
        if Path::new(p).is_file() {
            return PathBuf::from(p);
        }
    }
    panic!("the node tests need Node 22 for their fake docker/nvidia-smi/ip: set ECNODE_TEST_NODE")
}

fn fakes_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("wsd_fakes")
}

fn wrapper(bin: &Path, name: &str, module: &str, settings: &[(&str, String)]) {
    let set: String = settings
        .iter()
        .map(|(k, v)| format!("process.env[{}] = {};\n", json!(k), json!(v)))
        .collect();
    let text = format!(
        "#!{}\n'use strict';\nconst child = {{ ...process.env }};\n{set}const code = require({}).main(child);\n\
         if (code !== null && code !== undefined) process.exitCode = code;\n",
        node_bin().display(),
        json!(fakes_dir().join(module).to_string_lossy())
    );
    let p = bin.join(name);
    std::fs::write(&p, text).unwrap();
    std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
}

// ── docker / nvidia-smi / ip ───────────────────────────────────────────────────────────────────────────
pub struct FakeBin {
    pub dir: PathBuf,
    pub home: PathBuf,
    pub docker_state: PathBuf,
    pub docker_log: PathBuf,
    pub smi_state: PathBuf,
    pub ip_state: PathBuf,
}

/// conftest.container
#[derive(Default)]
pub struct C {
    pub labels: Option<Value>,
    pub running: Option<bool>,
    pub pid: Option<i64>,
    pub ports: Option<Value>,
    pub mounts: Option<Value>,
    pub network: Option<&'static str>,
    pub device_requests: Option<Value>,
}

pub fn container(name: &str, cid: &str, image: &str, args: &[&str], c: C) -> Value {
    let running = c.running.unwrap_or(true);
    let pid = c.pid.unwrap_or(4242);
    json!({
        "Id": cid, "Name": format!("/{name}"), "Image": format!("sha256:{}", "ab".repeat(32)),
        "Created": "2026-09-01T10:00:00.123456789Z", "Path": args[0], "Args": &args[1..],
        "State": {"Status": if running { "running" } else { "exited" }, "Running": running,
                  "Pid": if running { pid } else { 0 }, "ExitCode": 0, "StartedAt": "2026-09-01T10:00:01.5Z"},
        "Config": {"Image": image, "Labels": c.labels.unwrap_or_else(|| json!({})),
                   "Env": [format!("API_TOKEN={ENV_VALUE}"), "PATH=/usr/bin"]},
        "HostConfig": {"NetworkMode": c.network.unwrap_or("bridge"),
                       "DeviceRequests": c.device_requests.unwrap_or_else(|| json!([]))},
        "NetworkSettings": {"Ports": c.ports.unwrap_or_else(|| json!({}))},
        "Mounts": c.mounts.unwrap_or_else(|| json!([])),
    })
}

pub fn default_docker_state() -> Value {
    let engine_labels = json!({"engine-console": "1", "engine-console.role": "engine",
                               "engine-console.instance": "inst_1", "engine-console.engine": "vllm"});
    json!({
        "context": "testctx",
        "info": {"ServerVersion": "29.0.0", "SecurityOptions": ["name=seccomp,profile=builtin", "name=rootless"],
                 "Runtimes": {"runc": {}, "nvidia": {}}, "HttpProxy": "http://user:pw@proxy.invalid:3128"},
        "containers": [
            container("ec-owned-1", &cid_engine(), "vllm/vllm-openai:v0.11.0",
                      &["vllm", "serve", "org/model-a", "--api-key", ARGV_SECRET, "--port", "8000"],
                      C { labels: Some(engine_labels), ..C::default() }),
            container("other-engine", CID_EXT, "lmsysorg/sglang:v0.5.0",
                      &["python3", "-m", "sglang.launch_server", "--model-path", "org/model-b",
                        "--api-key=topsecretvalue", "--port", "30000"],
                      C { pid: Some(4343),
                          labels: Some(json!({"com.docker.compose.project": "lab",
                                              "com.docker.compose.service": "sglang"})),
                          ..C::default() }),
            container("grafana", &"a".repeat(64), "grafana/grafana:11", &["/run.sh"],
                      C { pid: Some(4444), ..C::default() }),
        ],
        "images": [
            {"Id": format!("sha256:{}", "11".repeat(32)), "RepoTags": ["vllm/vllm-openai:v0.11.0"],
             "Size": 123_456_789, "Created": "2026-08-01T00:00:00Z", "Architecture": "amd64"},
            {"Id": format!("sha256:{}", "22".repeat(32)), "RepoTags": [], "RepoDigests": [GATEWAY_IMAGE],
             "Size": 1234, "Created": "2026-08-02T00:00:00Z", "Architecture": "amd64"},
        ],
        "networks": {},
        "logs": {"ec-owned-1": ["INFO starting", format!("non-default args: {{'api_key': ['{ARGV_SECRET}']}}"),
                                "INFO ready"]},
    })
}

pub fn default_smi_state() -> Value {
    let gpu = |index: i64, uuid: &str, bus: &str| {
        json!({"memory.total": "97887", "compute_cap": "12.0", "driver_version": "595.0",
               "utilization.gpu": "7", "memory.used": "1024", "temperature.gpu": "41", "power.draw": "71.50",
               "power.limit": "300.00", "fan.speed": "30", "index": index, "uuid": uuid,
               "name": "NVIDIA Test GPU", "pci.bus_id": bus})
    };
    json!({"driver": "595.0", "cuda": "13.0",
           "gpus": [gpu(0, GPU_A, "00000000:01:00.0"), gpu(1, GPU_B, "00000000:02:00.0")],
           "apps": [{"pid": 4242, "used_memory": "2048", "gpu_uuid": GPU_A},
                    {"pid": 999_999, "used_memory": "10", "gpu_uuid": GPU_B}]})
}

impl FakeBin {
    pub fn new(tmp: &Path) -> Self {
        let dir = tmp.join("bin");
        let home = tmp.join("home");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::create_dir_all(&home).unwrap();
        let fb = Self {
            dir: dir.clone(),
            home,
            docker_state: tmp.join("docker.json"),
            docker_log: tmp.join("docker.log"),
            smi_state: tmp.join("smi.json"),
            ip_state: tmp.join("ip.json"),
        };
        let s = |p: &Path| p.to_string_lossy().into_owned();
        let settings = [
            ("FAKE_DOCKER_STATE", s(&fb.docker_state)),
            ("FAKE_DOCKER_LOG", s(&fb.docker_log)),
            ("FAKE_SMI_STATE", s(&fb.smi_state)),
            ("FAKE_IP_STATE", s(&fb.ip_state)),
        ];
        for (name, module) in [
            ("docker", "fake_docker.cjs"),
            ("nvidia-smi", "fake_smi.cjs"),
            ("ip", "fake_ip.cjs"),
        ] {
            wrapper(&dir, name, module, &settings);
        }
        fb.set_state(&default_docker_state());
        std::fs::write(&fb.smi_state, default_smi_state().to_string()).unwrap();
        fb
    }

    /// The node's own environment in these tests: the fakes only, a temp HOME. The node finds its tools in
    /// ECNODE_TOOL_PATH (never PATH: node hardening), so both point at the fakes; a test that wants "no such tool"
    /// points ECNODE_TOOL_PATH at an empty directory (`without_tools`), never just PATH, or the real ones would run.
    pub fn env(&self) -> EnvMap {
        let fakes = self.dir.to_string_lossy().into_owned();
        let mut e = EnvMap::new();
        e.insert("PATH".into(), fakes.clone());
        e.insert(ecnode::util::TOOL_PATH_ENV.into(), fakes);
        e.insert("HOME".into(), self.home.to_string_lossy().into_owned());
        e
    }

    /// `env()` with no tools at all: both PATH and ECNODE_TOOL_PATH name `empty` (an empty directory).
    pub fn without_tools(&self, empty: &Path) -> EnvMap {
        let mut e = self.env();
        let d = empty.to_string_lossy().into_owned();
        e.insert("PATH".into(), d.clone());
        e.insert(ecnode::util::TOOL_PATH_ENV.into(), d);
        e
    }

    pub fn state(&self) -> Value {
        serde_json::from_str(&std::fs::read_to_string(&self.docker_state).unwrap()).unwrap()
    }

    pub fn set_state(&self, state: &Value) {
        std::fs::write(&self.docker_state, state.to_string()).unwrap();
    }

    pub fn update(&self, kv: Value) {
        let mut s = self.state();
        for (k, v) in obj(kv) {
            s[k] = v;
        }
        self.set_state(&s);
    }

    pub fn calls(&self) -> Vec<Value> {
        std::fs::read_to_string(&self.docker_log)
            .unwrap_or_default()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    }

    /// argv lists of every docker call so far
    pub fn argvs(&self) -> Vec<Vec<String>> {
        self.calls()
            .iter()
            .map(|c| {
                c["argv"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|a| a.as_str().unwrap().to_string())
                    .collect()
            })
            .collect()
    }

    pub fn smi(&self, kv: Value) {
        let mut s: Value =
            serde_json::from_str(&std::fs::read_to_string(&self.smi_state).unwrap()).unwrap();
        for (k, v) in obj(kv) {
            s[k] = v;
        }
        std::fs::write(&self.smi_state, s.to_string()).unwrap();
    }

    pub fn smi_log(&self) -> PathBuf {
        let mut p = self.smi_state.clone().into_os_string();
        p.push(".log");
        PathBuf::from(p)
    }
}

// ── a fake machine: /proc, /sys, /etc and the mounts it reports ────────────────────────────────────────
pub struct Machine {
    pub root: PathBuf,
    pub paths: Paths,
    /// an ext4 "disk" holding model roots
    pub data: PathBuf,
    /// an NFS mount
    pub nfs: PathBuf,
}

pub fn w(path: &Path, text: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, text).unwrap();
}

/// 4 TiB disk, 1 TiB free, in 4 KiB blocks.
pub fn machine_statvfs() -> StatvfsFn {
    Arc::new(|_p: &Path| {
        Ok(VfsStat {
            frsize: 4096,
            blocks: 1 << 30,
            bfree: 1 << 28,
            bavail: 1 << 28,
        })
    })
}

impl Machine {
    pub fn new(tmp: &Path) -> Self {
        let root = tmp.join("machine");
        let (proc, sysd, etc) = (root.join("proc"), root.join("sys"), root.join("etc"));
        let (data, nfs) = (tmp.join("data"), tmp.join("nfs"));
        std::fs::create_dir_all(&data).unwrap();
        std::fs::create_dir_all(&nfs).unwrap();
        w(
            &etc.join("machine-id"),
            "0123456789abcdef0123456789abcdef\n",
        );
        w(
            &etc.join("os-release"),
            "NAME=\"Test Linux\"\nPRETTY_NAME=\"Test Linux 1.0\"\n",
        );
        let cpuinfo: String = (0..4)
            .map(|i| format!("processor\t: {i}\nmodel name\t: Test CPU 9000\nphysical id\t: 0\ncore id\t\t: {}\n\n", i / 2))
            .collect();
        w(&proc.join("cpuinfo"), &cpuinfo);
        w(
            &proc.join("meminfo"),
            "MemTotal:       263000000 kB\nMemFree:         1000000 kB\nMemAvailable:   200000000 kB\n\
             Cached:          5000000 kB\nShmem:           2000000 kB\nSwapTotal:       8000000 kB\n\
             SwapFree:        8000000 kB\n",
        );
        w(
            &proc.join("stat"),
            "cpu  100 0 100 800 0 0 0 0 0 0\ncpu0 1 0 1 8 0 0 0 0 0 0\nbtime 1700000000\n",
        );
        w(&proc.join("loadavg"), "0.50 0.40 0.30 1/100 1234\n");
        w(
            &proc.join("self").join("mountinfo"),
            &format!(
                "1 0 259:2 / / rw,relatime shared:1 - ext4 /dev/nvme0n1p2 rw\n\
                 2 1 259:3 / {} rw,noatime shared:2 - ext4 /dev/nvme1n1 rw\n\
                 3 1 0:50 / {} ro,relatime shared:3 - nfs4 10.0.0.9:/export/models ro,vers=4.2\n\
                 4 1 0:5 / /proc rw - proc proc rw\n\
                 5 1 7:1 / /snap/core/1 ro - squashfs /dev/loop1 ro\n",
                data.display(),
                nfs.display()
            ),
        );
        // a container's init process (pid 4242) in a rootless-docker cgroup, with memory and CPU counters
        let cg = format!(
            "/user.slice/user-1000.slice/user@1000.service/user.slice/docker-{}.scope",
            cid_engine()
        );
        w(&proc.join("4242").join("cgroup"), &format!("0::{cg}\n"));
        w(&proc.join("4242").join("comm"), "vllm\n");
        w(
            &proc.join("4242").join("stat"),
            "4242 (vllm) S 1 4242 4242 0 -1 0\n",
        );
        w(
            &proc.join("4242").join("status"),
            "Name:\tvllm\nVmRSS:\t  1024 kB\n",
        );
        let cgdir = sysd
            .join("fs")
            .join("cgroup")
            .join(cg.trim_start_matches('/'));
        w(&cgdir.join("memory.current"), "3000000000\n");
        w(
            &cgdir.join("memory.stat"),
            "anon 1000000000\nfile 1500000000\nshmem 500000000\nkernel 100000000\n",
        );
        w(
            &cgdir.join("cpu.stat"),
            "usage_usec 5000000\nuser_usec 4000000\n",
        );
        // pid 4343: no cgroup v2 dir -> RSS fallback over its process tree
        w(
            &proc.join("4343").join("cgroup"),
            &format!("0::/docker/{CID_EXT}\n"),
        );
        w(&proc.join("4343").join("comm"), "python3\n");
        w(
            &proc.join("4343").join("stat"),
            "4343 (python3) S 1 4343 4343 0 -1 0\n",
        );
        w(&proc.join("4343").join("status"), "VmRSS:\t  2048 kB\n");
        w(
            &proc.join("4344").join("stat"),
            "4344 (worker) S 4343 4343 4343 0 -1 0\n",
        );
        w(&proc.join("4344").join("status"), "VmRSS:\t  1024 kB\n");
        // block devices: a partition of nvme0n1, and a whole disk nvme1n1
        let dev0 = sysd
            .join("devices")
            .join("pci0")
            .join("nvme")
            .join("nvme0")
            .join("nvme0n1");
        w(&dev0.join("nvme0n1p2").join("partition"), "2\n");
        w(
            &sysd
                .join("block")
                .join("nvme0n1")
                .join("device")
                .join("model"),
            "Test SSD 4TB          \n",
        );
        let dev1 = sysd
            .join("devices")
            .join("pci0")
            .join("nvme")
            .join("nvme1")
            .join("nvme1n1");
        std::fs::create_dir_all(&dev1).unwrap();
        w(
            &sysd
                .join("block")
                .join("nvme1n1")
                .join("device")
                .join("model"),
            "Data Drive 8TB\n",
        );
        std::fs::create_dir_all(sysd.join("dev").join("block")).unwrap();
        std::os::unix::fs::symlink(
            dev0.join("nvme0n1p2"),
            sysd.join("dev").join("block").join("259:2"),
        )
        .unwrap();
        std::os::unix::fs::symlink(&dev1, sysd.join("dev").join("block").join("259:3")).unwrap();
        let paths = Paths {
            proc,
            sys: sysd,
            etc,
            state: tmp.join("state"),
        };
        Self {
            root,
            paths,
            data,
            nfs,
        }
    }
}

// ── policies ───────────────────────────────────────────────────────────────────────────────────────────
pub fn write_policy(path: &Path, data: &Value, mode: u32) -> PathBuf {
    let dir = path.parent().unwrap();
    std::fs::create_dir_all(dir).unwrap();
    std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700)).unwrap();
    std::fs::write(path, data.to_string()).unwrap();
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode)).unwrap();
    path.to_path_buf()
}

/// conftest.manage_policy: `over` keys replace the defaults.
pub fn manage_policy(machine: &Machine, over: Value) -> Value {
    for d in ["hf", "gguf", "plain", "ro"] {
        let _ = std::fs::create_dir(machine.data.join(d));
    }
    let p = |d: &str| machine.data.join(d).to_string_lossy().into_owned();
    let mut data = json!({
        "version": 1, "mode": "manage", "docker": {"binary": "docker", "context": "testctx"},
        "engine_images": ["vllm/vllm-openai:*", "lmsysorg/sglang@sha256:*", "ghcr.io/ggml-org/llama.cpp:server-cuda"],
        "gateway_image": GATEWAY_IMAGE, "engine_network": "engine-console-engines", "port_range": [18000, 18099],
        "discovery_ports": [],
        "model_roots": [
            {"id": "hf", "path": p("hf"), "kind": "hf-home", "writable": true, "label": "HF"},
            {"id": "gguf", "path": p("gguf"), "kind": "gguf", "writable": true},
            {"id": "plain", "path": p("plain"), "kind": "plain", "writable": true},
            {"id": "ro", "path": p("ro"), "kind": "auto", "writable": false},
            {"id": "shared", "path": machine.nfs.to_string_lossy(), "kind": "auto", "writable": false},
        ],
        "scripts": [],
        "allow_pull": true, "allow_delete": true, "allow_ipc_host": true, "external_control": "stop_start",
        "max_shm_gib": 64,
    });
    for (k, v) in obj(over) {
        data[k] = v;
    }
    data
}

/// A PolicyStore over a policy file in tmp (or no file at all), never looking at the real /etc; re-checked on
/// every read (conftest.store_for, recheck_s=0).
pub fn store_for(tmp: &Path, data: Option<&Value>) -> ecnode::policy::PolicyStore {
    let path = data.map(|d| {
        write_policy(&tmp.join("pol").join("policy.json"), d, 0o644)
            .to_string_lossy()
            .into_owned()
    });
    let opts = ecnode::policy::LoadOptions::new(
        path.as_deref(),
        Some(&tmp.join("home").to_string_lossy()),
        &tmp.join("no-etc").join("policy.json").to_string_lossy(),
    );
    ecnode::policy::PolicyStore::with_options(opts, 0.0, ecnode::policy::monotonic())
}

// ── a fake op stream ───────────────────────────────────────────────────────────────────────────────────
type Hook = Box<dyn FnOnce() + Send>;

/// test_fsops.FakeStream
#[derive(Default)]
pub struct FakeStream {
    chunks: Mutex<VecDeque<Vec<u8>>>,
    cancel_after: Option<usize>,
    reads: Mutex<usize>,
    pub events: Mutex<Vec<Obj>>,
    pub data: Mutex<Vec<Vec<u8>>>,
    cancelled: AtomicBool,
    hooks: Mutex<Vec<Hook>>,
    /// called at the start of every read_upload (the "Slow" stream of test_two_uploads_of_one_file_conflict)
    pub before_read: Mutex<Option<Box<dyn Fn() + Send + Sync>>>,
}

impl FakeStream {
    pub fn new(chunks: Vec<Vec<u8>>) -> Self {
        Self {
            chunks: Mutex::new(chunks.into()),
            ..Self::default()
        }
    }

    pub fn empty() -> Self {
        Self::default()
    }

    pub fn cancel_after(chunks: Vec<Vec<u8>>, n: usize) -> Self {
        Self {
            chunks: Mutex::new(chunks.into()),
            cancel_after: Some(n),
            ..Self::default()
        }
    }

    pub fn events(&self) -> Vec<Obj> {
        self.events.lock().unwrap().clone()
    }

    pub fn data(&self) -> Vec<u8> {
        self.data.lock().unwrap().concat()
    }

    pub fn chunk_sizes(&self) -> Vec<usize> {
        self.data.lock().unwrap().iter().map(Vec::len).collect()
    }

    /// `s._cancelled = True; [h() for h in s.hooks]`
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
        let hooks: Vec<Hook> = std::mem::take(&mut *self.hooks.lock().unwrap());
        for h in hooks {
            h();
        }
    }

    pub fn set_cancelled(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    pub fn hook_count(&self) -> usize {
        self.hooks.lock().unwrap().len()
    }
}

impl Stream for FakeStream {
    fn cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    fn check(&self) -> Result<(), Cancelled> {
        if self.cancelled() {
            Err(Cancelled)
        } else {
            Ok(())
        }
    }

    fn event(&self, ev: Obj) {
        self.events.lock().unwrap().push(ev);
    }

    fn send_data(&self, data: &[u8]) -> Result<(), HandlerError> {
        self.data.lock().unwrap().push(data.to_vec());
        Ok(())
    }

    fn read_upload(&self) -> Result<Vec<u8>, HandlerError> {
        if let Some(f) = self.before_read.lock().unwrap().as_ref() {
            f();
        }
        let mut reads = self.reads.lock().unwrap();
        if let Some(n) = self.cancel_after
            && *reads >= n
        {
            self.cancelled.store(true, Ordering::SeqCst);
            return Err(HandlerError::Cancelled);
        }
        *reads += 1;
        Ok(self.chunks.lock().unwrap().pop_front().unwrap_or_default())
    }

    fn drain_upload(&self) {
        self.chunks.lock().unwrap().clear();
    }

    fn on_cancel(&self, f: Box<dyn FnOnce() + Send>) {
        self.hooks.lock().unwrap().push(f);
    }
}

// ── probes and a fake engine ───────────────────────────────────────────────────────────────────────────
/// Probes only the ports a test allows (fake engines); anything else is refused without a connection.
#[derive(Clone, Default)]
pub struct SafeProber {
    pub allowed: Arc<Mutex<HashSet<u16>>>,
    pub tried: Arc<Mutex<Vec<u16>>>,
}

impl SafeProber {
    pub fn allow(&self, port: u16) {
        self.allowed.lock().unwrap().insert(port);
    }

    pub fn tried(&self) -> Vec<u16> {
        self.tried.lock().unwrap().clone()
    }

    pub fn probe(&self, port: u16) -> PortProbe {
        self.tried.lock().unwrap().push(port);
        if !self.allowed.lock().unwrap().contains(&port) {
            return PortProbe {
                port,
                ..PortProbe::default()
            };
        }
        discovery::probe_port(port)
    }

    pub fn get(&self, port: u16, path: &str, o: &GetOpts) -> Probe {
        if !self.allowed.lock().unwrap().contains(&port) {
            return Probe::default();
        }
        discovery::http_get(port, path, o)
    }

    pub fn prober(&self) -> discovery::Prober {
        let me = self.clone();
        Arc::new(move |p| me.probe(p))
    }

    /// probe_port_quiet, through the same allowlist
    pub fn quiet_prober(&self) -> discovery::Prober {
        let me = self.clone();
        Arc::new(move |p| {
            me.tried.lock().unwrap().push(p);
            if !me.allowed.lock().unwrap().contains(&p) {
                return PortProbe {
                    port: p,
                    ..PortProbe::default()
                };
            }
            discovery::probe_port_quiet(p)
        })
    }

    pub fn metrics_get(&self) -> discovery::MetricsGet {
        let me = self.clone();
        Arc::new(move |port, path, o| me.get(port, path, o))
    }
}

pub const METRICS_TEXT: &str = "# HELP vllm:num_requests_running Requests running.\n\
# TYPE vllm:num_requests_running gauge\n\
vllm:num_requests_running{model_name=\"org/model-a\"} 1.0\n\
# HELP process_cpu_seconds_total CPU.\n\
# TYPE process_cpu_seconds_total counter\n\
process_cpu_seconds_total 5.0\n";

/// One request the fake engine saw.
#[derive(Debug, Clone, Default)]
pub struct Seen {
    pub method: String,
    pub path: String,
    /// as sent (name case kept); `header()` looks one up like Python's dict(self.headers)
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Seen {
    /// dict(self.headers)[name]: exact name, the last value
    pub fn header(&self, name: &str) -> Option<String> {
        self.headers
            .iter()
            .rev()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.clone())
    }
}

/// conftest's fake engine (an HTTP/1.1 server on 127.0.0.1).
pub struct FakeEngine {
    pub port: u16,
    pub seen: Arc<Mutex<Vec<Seen>>>,
    stop: Arc<AtomicBool>,
}

fn respond(
    s: &mut TcpStream,
    status: u16,
    reason: &str,
    body: &[u8],
    ctype: &str,
    extra: &[(&str, &str)],
) {
    let mut head = format!(
        "HTTP/1.1 {status} {reason}\r\nServer: FakeEngine/1.0\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\n",
        body.len()
    );
    for (k, v) in extra {
        head.push_str(&format!("{k}: {v}\r\n"));
    }
    head.push_str("\r\n");
    let _ = s.write_all(head.as_bytes());
    let _ = s.write_all(body);
    let _ = s.flush();
}

fn serve(mut s: TcpStream, seen: &Arc<Mutex<Vec<Seen>>>) {
    let mut r = BufReader::new(match s.try_clone() {
        Ok(c) => c,
        Err(_) => return,
    });
    loop {
        let mut line = String::new();
        if r.read_line(&mut line).unwrap_or(0) == 0 {
            return;
        }
        let parts: Vec<&str> = line.trim_end().splitn(3, ' ').collect();
        if parts.len() < 2 {
            return;
        }
        let (method, path) = (parts[0].to_string(), parts[1].to_string());
        let mut headers = Vec::new();
        loop {
            let mut h = String::new();
            if r.read_line(&mut h).unwrap_or(0) == 0 {
                return;
            }
            let h = h.trim_end_matches(['\r', '\n']);
            if h.is_empty() {
                break;
            }
            if let Some((k, v)) = h.split_once(':') {
                headers.push((k.to_string(), v.trim_start().to_string()));
            }
        }
        let mut body = Vec::new();
        if method == "POST" {
            let n: usize = headers
                .iter()
                .find(|(k, _)| k.eq_ignore_ascii_case("content-length"))
                .and_then(|(_, v)| v.trim().parse().ok())
                .unwrap_or(0);
            body = vec![0u8; n];
            if r.read_exact(&mut body).is_err() {
                return;
            }
        }
        seen.lock().unwrap().push(Seen {
            method: method.clone(),
            path: path.clone(),
            headers,
            body: body.clone(),
        });
        if method == "POST" {
            let mut out = b"echo:".to_vec();
            out.extend_from_slice(&body);
            respond(&mut s, 200, "OK", &out, "application/json", &[]);
            continue;
        }
        match path.as_str() {
            "/v1/models" => respond(
                &mut s,
                200,
                "OK",
                br#"{"object": "list", "data": [{"id": "org/model-a", "owned_by": "vllm"}]}"#,
                "application/json",
                &[],
            ),
            "/health" => respond(&mut s, 200, "OK", b"", "application/json", &[]),
            "/metrics" => respond(
                &mut s,
                200,
                "OK",
                METRICS_TEXT.as_bytes(),
                "text/plain",
                &[],
            ),
            "/v1/redirect" => respond(
                &mut s,
                302,
                "Found",
                b"",
                "application/json",
                &[("Location", "http://example.invalid/")],
            ),
            "/v1/stream" => {
                let _ = s.write_all(
                    b"HTTP/1.1 200 OK\r\nServer: FakeEngine/1.0\r\nContent-Type: text/event-stream\r\n\
                      Transfer-Encoding: chunked\r\n\r\n",
                );
                for i in 0..3 {
                    let chunk = format!("data: {i}\n\n");
                    let _ = s.write_all(format!("{:x}\r\n{chunk}\r\n", chunk.len()).as_bytes());
                    let _ = s.flush();
                    thread::sleep(Duration::from_millis(20));
                }
                let _ = s.write_all(b"0\r\n\r\n");
            }
            "/v1/slow" => {
                thread::sleep(Duration::from_secs(3));
                respond(&mut s, 200, "OK", b"late", "application/json", &[]);
            }
            _ => respond(&mut s, 404, "Not Found", b"{}", "application/json", &[]),
        }
    }
}

impl FakeEngine {
    pub fn start() -> Self {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = l.local_addr().unwrap().port();
        let seen: Arc<Mutex<Vec<Seen>>> = Arc::default();
        let stop = Arc::new(AtomicBool::new(false));
        {
            let (seen, stop) = (Arc::clone(&seen), Arc::clone(&stop));
            thread::spawn(move || {
                for conn in l.incoming() {
                    if stop.load(Ordering::SeqCst) {
                        return;
                    }
                    let Ok(c) = conn else { continue };
                    let seen = Arc::clone(&seen);
                    thread::spawn(move || serve(c, &seen));
                }
            });
        }
        Self { port, seen, stop }
    }

    pub fn seen(&self) -> Vec<Seen> {
        self.seen.lock().unwrap().clone()
    }
}

impl Drop for FakeEngine {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        let _ = TcpStream::connect(("127.0.0.1", self.port));
    }
}

/// A loopback port nobody listens on (bound, then released).
pub fn closed_port() -> u16 {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    l.local_addr().unwrap().port()
}

/// `proto.clean_args(op, args)` as an object.
pub fn clean_args(op: &str, args: Value) -> Obj {
    ec_proto::clean_args(op, &args).unwrap()
}
