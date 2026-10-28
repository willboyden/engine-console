//! A frame-level fake `ecnode --stdio` for the hub's link tests (port of backend/tests/fleet/fake_node.py).
//!
//! It runs as THIS test binary: `node_argv(scenario)` is a command that re-executes the current test executable,
//! filtered to its `fake_node_main` test (every test file that uses it declares one with `fake_node_entry!()`), with
//! the scenario as an extra argument. libtest prints a banner on stdout before the test runs, so the command first
//! moves the pipe the hub reads to fd 3 and points fd 1 at /dev/null (`/bin/sh -c 'exec ... 3>&1 1>/dev/null'`); the
//! fake node then puts the pipe back on fd 1 and closes fd 3. `env -u PWD` keeps the shell's PWD out of the node's
//! environment, so the environment the hub gave is exactly what the node sees.
//!
//! It speaks the real wire protocol with the real `ec_proto` framing, so the hub's `NodeLink` exercises its true
//! subprocess path: spawn, preamble, hello, pushes, requests, DATA with credit, cancel, silence, flood, crash. The
//! scenario (a JSON object, see `defaults()`) scripts what it does; `log` names a JSONL file the node appends
//! everything it saw and did to, so a test can assert on the far side. Requests are served in one thread each.

#![allow(
    dead_code,
    unreachable_pub,
    clippy::panic,
    clippy::type_complexity,
    clippy::expect_used,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::needless_pass_by_value,
    clippy::print_stderr,
    clippy::too_many_lines
)]

use std::collections::HashMap;
use std::io::Write as _;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use ec_proto::Obj;
use serde_json::{Value, json};
use sha2::Digest as _;

pub const GIB: i64 = 1024 * 1024 * 1024;
const MARK: &str = "WSB_FAKE_NODE_SCENARIO=";
const ENTRY: &str = "fake_node_main";

/// Declare the entry point the fake node runs as, in a test file that starts fake nodes.
#[macro_export]
macro_rules! fake_node_entry {
    () => {
        #[test]
        fn fake_node_main() {
            wsb_fake_node::main_if_asked();
        }
    };
}

/// The command that runs the fake node with this scenario.
pub fn node_argv(scenario: &Value) -> Vec<String> {
    let exe = std::env::current_exe()
        .expect("current exe")
        .to_string_lossy()
        .into_owned();
    vec![
        "/bin/sh".into(),
        "-c".into(),
        "exec /usr/bin/env -u PWD \"$0\" \"$@\" 3>&1 1>/dev/null".into(),
        exe,
        "--exact".into(),
        ENTRY.into(),
        "--nocapture".into(),
        "--test-threads=1".into(),
        "-q".into(),
        format!("{MARK}{scenario}"),
    ]
}

/// Bytes offset..offset+n of the node's deterministic test file: each 4 KiB block carries its own index.
pub fn pattern(offset: usize, n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n);
    for pos in offset..offset + n {
        out.push(((pos / 4096) % 251) as u8);
    }
    out
}

/// A stand-in `ssh` in `bindir`: a script that records its arguments and becomes this test binary in fake-node mode,
/// which logs the call to `ssh_log` and then fails like ssh does (`mode` "denied" | "badkey") or behaves like a node.
pub fn install_fake_ssh(
    bindir: &Path,
    ssh_log: &Path,
    mode: &str,
    scenario: &Value,
    key: Option<&Path>,
) {
    std::fs::create_dir_all(bindir).expect("bindir");
    let args_file = bindir.join("ssh-args");
    let mut sc = scenario.clone();
    if let Value::Object(m) = &mut sc {
        m.insert("ssh_log".into(), json!(ssh_log.to_string_lossy()));
        m.insert("ssh_args_file".into(), json!(args_file.to_string_lossy()));
        m.insert("ssh_mode".into(), json!(mode));
        m.insert(
            "ssh_key".into(),
            json!(key.map(|k| k.to_string_lossy().into_owned())),
        );
    }
    let argv = node_argv(&sc);
    let quote = |s: &str| {
        assert!(!s.contains('\''), "no single quotes in a fake ssh argument");
        format!("'{s}'")
    };
    let script = format!(
        "#!/bin/sh\nprintf '%s\\0' \"$@\" > {}\nexec /usr/bin/env -u PWD {} 3>&1 1>/dev/null\n",
        quote(&args_file.to_string_lossy()),
        argv[3..]
            .iter()
            .map(|a| quote(a))
            .collect::<Vec<_>>()
            .join(" ")
    );
    let path = bindir.join("ssh");
    std::fs::write(&path, script).expect("script");
    let mut perms = std::fs::metadata(&path).expect("meta").permissions();
    std::os::unix::fs::PermissionsExt::set_mode(&mut perms, 0o755);
    std::fs::set_permissions(&path, perms).expect("chmod");
}

pub fn read_log(path: &Path) -> Vec<Value> {
    std::fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect()
}

fn defaults() -> Obj {
    let images = json!([{"ref": "vllm/vllm-openai:v0.1", "id": "sha256:abc", "size": 10 * GIB, "created": 1.7e9,
                         "arch": "amd64"}]);
    json!({
        "log": null, "junk": "", "junk_lines": 0, "stderr": "", "stderr_noise": 0, "exit_before_hello": null,
        "hello_delay_s": 0.0, "first_frame": "hello", "proto": ec_proto::PROTO, "version": "0.4.0-test",
        "forced_cmd": true, "gpus": 2, "hello_patch": {}, "hello_extra": {}, "info_mode": null, "samples": true,
        "sample_interval_s": 0.05, "samples_stop_after_s": null, "rogue_gpu": true, "silent_after_hello": false,
        "no_pong": false, "exit_after_s": null, "exit_code": 3, "pushes": [], "push_disks": null,
        "push_inventory": null, "flood": null, "ignore_cancel": false, "ignore_credit": false,
        "data_chunk": 256 * 1024, "upload_hold_s": 0.0, "upload_consume_delay_s": 0.0, "credit_batch": 1024 * 1024,
        "crash_after_bytes": null, "follow_delay_s": 0.0, "images": images, "hostile_images": false,
        "spawn_child": false})
    .as_object()
    .cloned()
    .unwrap_or_default()
}

/// Run the fake node when this process was started as one (never returns then); a no-op otherwise.
pub fn main_if_asked() {
    let Some(arg) = std::env::args().find(|a| a.starts_with(MARK)) else {
        return;
    };
    let sc: Obj = serde_json::from_str(&arg[MARK.len()..]).unwrap_or_default();
    // the pipe the hub reads is fd 3 (fd 1 is /dev/null while libtest prints its banner): put it back on fd 1
    let out = std::fs::OpenOptions::new()
        .write(true)
        .open("/proc/self/fd/3")
        .expect("fd 3");
    let _ = nix::unistd::dup2_stdout(&out);
    let _ = nix::unistd::close(3);
    let node = Arc::new(Node::new(sc, out));
    node.main();
    std::process::exit(0);
}

fn merge(base: &mut Value, patch: &Value) {
    if let (Value::Object(b), Value::Object(p)) = (base, patch) {
        for (k, v) in p {
            match b.get_mut(k) {
                Some(bv @ Value::Object(_)) if v.is_object() => merge(bv, v),
                _ => {
                    b.insert(k.clone(), v.clone());
                }
            }
        }
    }
}

fn sleep(s: f64) {
    if s > 0.0 {
        std::thread::sleep(Duration::from_secs_f64(s));
    }
}

fn now_wall() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64())
}

/// One request being served: its cancel flag, the node's send credit for it, and the hub's upload.
struct Req {
    rid: i64,
    op: String,
    args: Obj,
    cancel: AtomicBool,
    credit: Mutex<i64>,
    cv: Condvar,
    inbox: Mutex<Option<mpsc::Sender<(String, Vec<u8>)>>>,
    received: AtomicI64,
    granted: AtomicI64,
}

impl Req {
    fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::SeqCst)
    }
}

struct Node {
    sc: Obj,
    out: Mutex<std::fs::File>,
    log_lock: Mutex<()>,
    reqs: Mutex<HashMap<i64, Arc<Req>>>,
    t0: Instant,
}

impl Node {
    fn new(sc: Obj, out: std::fs::File) -> Self {
        let mut all = defaults();
        for (k, v) in sc {
            all.insert(k, v);
        }
        Self {
            sc: all,
            out: Mutex::new(out),
            log_lock: Mutex::new(()),
            reqs: Mutex::new(HashMap::new()),
            t0: Instant::now(),
        }
    }

    fn f(&self, k: &str) -> f64 {
        self.sc.get(k).and_then(Value::as_f64).unwrap_or(0.0)
    }

    fn b(&self, k: &str) -> bool {
        self.sc.get(k).and_then(Value::as_bool).unwrap_or(false)
    }

    fn i(&self, k: &str) -> i64 {
        self.sc.get(k).and_then(Value::as_i64).unwrap_or(0)
    }

    fn s(&self, k: &str) -> String {
        self.sc
            .get(k)
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string()
    }

    // ── output ───────────────────────────────────────────────────────────────────────────────────────────
    fn log(&self, ev: &str, kw: Value) {
        let Some(path) = self.sc.get("log").and_then(Value::as_str) else {
            return;
        };
        let mut line =
            json!({"ev": ev, "t": (self.t0.elapsed().as_secs_f64() * 1000.0).round() / 1000.0});
        merge(&mut line, &kw);
        let _g = self.log_lock.lock();
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        {
            let _ = writeln!(f, "{line}");
        }
    }

    fn write(&self, data: &[u8]) {
        let mut out = self
            .out
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if out.write_all(data).and_then(|()| out.flush()).is_err() {
            std::process::exit(0); // the hub is gone
        }
    }

    fn send(&self, msg: &Value) {
        if let Ok(frame) = ec_proto::encode_value(msg) {
            self.write(&frame);
        }
    }

    fn send_raw_json(&self, text: &str) {
        let body = text.as_bytes();
        let mut frame = ((body.len() + 1) as u32).to_be_bytes().to_vec();
        frame.push(ec_proto::KIND_JSON);
        frame.extend_from_slice(body);
        self.write(&frame);
    }

    fn send_data(&self, stream: i64, payload: &[u8]) {
        if let Ok(frame) = ec_proto::encode_data(stream, payload) {
            self.write(&frame);
        }
    }

    fn res_ok(&self, rid: i64, result: Value) {
        self.send(&Value::Object(ec_proto::msg::res_ok(
            rid,
            result.as_object().cloned(),
        )));
    }

    fn res_err(&self, rid: i64, code: &str, message: &str) {
        self.send(&Value::Object(ec_proto::msg::res_err(rid, code, message)));
    }

    fn event(&self, rid: i64, ev: Value) {
        self.send(&Value::Object(ec_proto::msg::event(
            rid,
            ev.as_object().cloned().unwrap_or_default(),
        )));
    }

    // ── messages ─────────────────────────────────────────────────────────────────────────────────────────
    fn hello(&self) -> Value {
        let gpus: Vec<Value> = (0..self.i("gpus"))
            .map(|i| {
                json!({"index": i, "uuid": format!("GPU-{i:04}"), "name": "Test GPU 96GB", "total_bytes": 96 * GIB,
                       "compute_capability": "12.0", "unified_memory": false, "pci_bus_id": format!("0000:0{i}:00.0")})
            })
            .collect();
        let mut msg = json!({
            "type": "hello", "proto": self.sc.get("proto").cloned().unwrap_or(Value::Null),
            "version": self.sc.get("version").cloned().unwrap_or(Value::Null), "node_id": "feedc0de12345678",
            "host": {"hostname": "gpu-box-2", "arch": "x86_64", "os": "Ubuntu 24.04", "kernel": "6.8.0-test",
                     "cpu_model": "Test CPU", "cpu_cores": 16, "cpu_threads": 32, "ram_total_bytes": 256 * GIB,
                     "boot_time": 1.7e9, "python": "3.12.3"},
            "gpu": {"driver": "570.00", "cuda": "12.8", "source": "nvml"},
            "gpus": gpus,
            "docker": {"available": true, "context": "default", "rootless": true, "version": "27.0.0",
                       "nvidia_runtime": true},
            "addresses": [{"iface": "eth0", "addr": "10.0.0.5", "prefix": 24}],
            "policy": {"path": "/etc/ecnode/policy.json", "mode": "monitor", "engine_images": ["vllm/vllm-openai:*"],
                       "gateway_image": "nginx@sha256:00", "engine_network": "engine-console-engines",
                       "port_range": [18000, 18099], "discovery_ports": [8000],
                       "model_roots": [{"id": "hf", "path": "/data/hf", "kind": "hf-home", "writable": true,
                                        "label": "HF cache"}],
                       "scripts": [], "allow_pull": false, "allow_delete": false, "allow_ipc_host": true,
                       "external_control": "none", "max_shm_gib": 64},
            "forced_cmd": self.sc.get("forced_cmd").cloned().unwrap_or(Value::Null), "via_ssh": false});
        merge(&mut msg, self.sc.get("hello_patch").unwrap_or(&Value::Null));
        if let (Value::Object(m), Some(Value::Object(extra))) =
            (&mut msg, self.sc.get("hello_extra"))
        {
            for (k, v) in extra {
                m.insert(k.clone(), v.clone());
            }
        }
        msg
    }

    fn sample(&self, seq: i64) -> Value {
        let mut gpus = serde_json::Map::new();
        for i in 0..self.i("gpus") {
            gpus.insert(
                format!("GPU-{i:04}"),
                json!({"util_pct": 12.5, "mem_used_bytes": 10 * GIB, "mem_total_bytes": 96 * GIB, "temp_c": 41.0,
                       "power_w": 123.0, "power_limit_w": 300.0, "fan_pct": 30.0,
                       "procs": [{"pid": 1000 + i, "used_bytes": 8 * GIB, "container": "ec-engine-1", "name": "python3"}]}),
            );
        }
        if self.b("rogue_gpu") {
            gpus.insert("GPU-ROGUE".into(), json!({"util_pct": 99.0}));
        }
        json!({"type": "sample", "seq": seq, "t": now_wall(), "gpus": gpus,
               "mem": {"total": 256 * GIB, "available": 200 * GIB, "free": 150 * GIB, "cached": 40 * GIB,
                       "shmem": GIB, "swap_total": 8 * GIB, "swap_free": 8 * GIB},
               "cpu": {"load1": 1.0, "load5": 0.8, "load15": 0.5, "util_pct": 7.5},
               "containers": {"ec-engine-1": {"mem": {"current": 30 * GIB, "anon": 20 * GIB, "file": 9 * GIB,
                                                     "shmem": GIB, "kernel": GIB, "source": "cgroup"}, "cpu_pct": 55.0}}})
    }

    // ── background behaviours ────────────────────────────────────────────────────────────────────────────
    fn sampler(self: Arc<Self>) {
        let stop_after = self.sc.get("samples_stop_after_s").and_then(Value::as_f64);
        let mut seq = 0;
        loop {
            if stop_after.is_some_and(|s| self.t0.elapsed().as_secs_f64() > s) {
                return;
            }
            seq += 1;
            self.send(&self.sample(seq));
            sleep(self.f("sample_interval_s"));
        }
    }

    fn scheduled(self: Arc<Self>) {
        let mut pushes: Vec<Value> = self
            .sc
            .get("pushes")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let at = |p: &Value| p.get("after_s").and_then(Value::as_f64).unwrap_or(0.0);
        pushes.sort_by(|a, b| at(a).total_cmp(&at(b)));
        for p in pushes {
            sleep(at(&p) - self.t0.elapsed().as_secs_f64());
            if let Some(raw) = p.get("raw").and_then(Value::as_str) {
                self.send_raw_json(raw);
            } else if let Some(stream) = p.get("data_stream").and_then(Value::as_i64) {
                let n = p.get("bytes").and_then(Value::as_u64).unwrap_or(10) as usize;
                self.send_data(stream, &vec![b'x'; n]);
            } else if let Some(m) = p.get("msg") {
                self.send(m);
            }
        }
    }

    fn flooder(self: Arc<Self>) {
        let f = self.sc.get("flood").cloned().unwrap_or(Value::Null);
        sleep(f.get("after_s").and_then(Value::as_f64).unwrap_or(0.0));
        let kb = f.get("frame_kb").and_then(Value::as_u64).unwrap_or(1) as usize;
        let body = json!({"type": "log", "level": "info", "msg": "x".repeat(kb * 1024)});
        for _ in 0..f.get("count").and_then(Value::as_u64).unwrap_or(100) {
            self.send(&body);
        }
    }

    fn exiter(self: Arc<Self>) {
        sleep(self.f("exit_after_s"));
        eprintln!("fake node crashing on purpose");
        let code = self.i("exit_code");
        self.log("exit", json!({"code": code}));
        std::process::exit(code as i32);
    }

    // ── requests ─────────────────────────────────────────────────────────────────────────────────────────
    fn serve(self: Arc<Self>, r: Arc<Req>, inbox: mpsc::Receiver<(String, Vec<u8>)>) {
        match r.op.as_str() {
            "node.info" => self.op_node_info(&r),
            "metrics.configure" | "docker.stop" => self.res_ok(r.rid, json!({})),
            "docker.images" => self.op_docker_images(&r),
            "docker.image_present" => {
                if r.args.get("image").and_then(Value::as_str) == Some("missing") {
                    self.res_err(r.rid, ec_proto::NOT_FOUND, "no such image");
                } else {
                    self.res_ok(r.rid, json!({"present": true, "id": "sha256:abc"}));
                }
            }
            "docker.inspect" => self.op_docker_inspect(&r),
            "models.scan" => {
                // the op the tests use to "hang"
                while !r.cancelled() {
                    std::thread::sleep(Duration::from_millis(50));
                }
                self.cancelled(&r);
            }
            "docker.logs_follow" => self.op_logs_follow(&r),
            "docker.pull" => {
                for i in 0..3 {
                    self.event(r.rid, json!({"line": format!("layer {i}")}));
                }
                // an event the hub must drop
                self.send(&json!({"type": "event", "id": r.rid, "event": {"unknown_field": 1}}));
                self.res_err(r.rid, ec_proto::DOCKER_ERROR, "pull failed (scripted)");
            }
            "scripts.run" => {
                for i in 0..2 {
                    self.event(r.rid, json!({"stream": "out", "line": format!("step {i}")}));
                }
                self.res_ok(r.rid, json!({"rc": 0}));
            }
            "http.request" => self.op_http_request(&r, &inbox),
            "fs.read" => self.op_fs_read(&r),
            "fs.write" => self.op_fs_write(&r, &inbox),
            op => self.res_err(
                r.rid,
                ec_proto::UNKNOWN_OP,
                &format!("the fake node does not do {op}"),
            ),
        }
        self.reqs
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&r.rid);
    }

    fn cancelled(&self, r: &Req) {
        if !self.b("ignore_cancel") {
            self.res_err(r.rid, ec_proto::CANCELLED, "cancelled");
        }
    }

    fn op_node_info(&self, r: &Req) {
        let mut h = self.hello();
        if let Value::Object(m) = &mut h {
            m.remove("type");
            if let Some(mode) = self.sc.get("info_mode").filter(|v| !v.is_null())
                && let Some(Value::Object(p)) = m.get_mut("policy")
            {
                p.insert("mode".into(), mode.clone());
            }
        }
        self.res_ok(r.rid, h);
    }

    fn op_docker_images(&self, r: &Req) {
        if self.b("hostile_images") {
            self.res_ok(
                r.rid,
                json!({"images": [{"ref": "r".repeat(5000), "id": "i\u{0}d", "size": 5, "evil": "x"}],
                       "evil_top": {"deep": [1, 2]}}),
            );
        } else {
            self.res_ok(
                r.rid,
                json!({"images": self.sc.get("images").cloned().unwrap_or(Value::Null)}),
            );
        }
    }

    fn op_docker_inspect(&self, r: &Req) {
        let name = r
            .args
            .get("name")
            .map(ec_proto::pyfmt::str_value)
            .unwrap_or_else(|| "None".into());
        if let Some(code) = name.strip_prefix("err-") {
            self.res_err(r.rid, code, "scripted refusal");
        } else if name == "no-error-object" {
            self.send(&json!({"type": "res", "id": r.rid, "ok": false}));
        } else if name == "ctl" {
            self.res_err(
                r.rid,
                ec_proto::IO_ERROR,
                "disk\u{7} full\nsecond line hf_abcdefghijklmnop",
            );
        } else {
            if name.starts_with("slow-") {
                sleep(0.4);
            }
            self.res_ok(r.rid, json!({"info": {"name": name, "state": "running"}}));
        }
    }

    fn op_logs_follow(&self, r: &Req) {
        let total = r.args.get("tail").and_then(Value::as_i64).unwrap_or(0); // 0: follow until cancelled
        let ignore = self.b("ignore_cancel");
        let mut n = 0;
        // a node that ignores cancel keeps sending events after it, which the hub must not keep
        while !(r.cancelled() && !ignore) && (total == 0 || n < total) {
            n += 1;
            self.event(r.rid, json!({"line": format!("line {n}")}));
            self.log("follow", json!({"id": r.rid, "n": n}));
            sleep(self.f("follow_delay_s"));
        }
        if r.cancelled() {
            self.cancelled(r);
        } else {
            self.res_ok(r.rid, json!({"lines": n}));
        }
    }

    fn op_http_request(&self, r: &Req, inbox: &mpsc::Receiver<(String, Vec<u8>)>) {
        let body_len = r.args.get("body_len").and_then(Value::as_i64).unwrap_or(0);
        let mut got = 0i64;
        while got < body_len {
            match inbox.recv() {
                Ok((kind, _)) if kind == "end" => break,
                Ok((_, data)) => got += data.len() as i64,
                Err(_) => break,
            }
        }
        self.event(
            r.rid,
            json!({"status": 200, "headers": {"content-type": "text/plain"}}),
        );
        let body = format!("engine answer, request body was {got} bytes");
        self.send_data(r.rid, body.as_bytes());
        self.res_ok(r.rid, json!({"status": 200, "bytes": body.len()}));
    }

    fn op_fs_read(&self, r: &Req) {
        let length = r.args.get("length").and_then(Value::as_i64).unwrap_or(0);
        let offset = r.args.get("offset").and_then(Value::as_i64).unwrap_or(0);
        let chunk = self.i("data_chunk");
        let mut sent = 0i64;
        while sent < length && !r.cancelled() {
            let n = chunk.min(length - sent);
            if !self.b("ignore_credit") {
                let mut credit = r
                    .credit
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if *credit < n {
                    self.log(
                        "stalled",
                        json!({"id": r.rid, "sent": sent, "credit": *credit}),
                    );
                }
                while *credit < n && !r.cancelled() {
                    credit =
                        r.cv.wait_timeout(credit, Duration::from_millis(100))
                            .map_or_else(|e| e.into_inner().0, |(g, _)| g);
                }
                *credit -= n;
            }
            if r.cancelled() {
                break;
            }
            self.send_data(r.rid, &pattern((offset + sent) as usize, n as usize));
            sent += n;
            self.log("sent", json!({"id": r.rid, "sent": sent}));
            let crash = self.sc.get("crash_after_bytes").and_then(Value::as_i64);
            if crash.is_some_and(|c| c > 0 && sent >= c) {
                std::process::exit(7);
            }
        }
        if r.cancelled() {
            self.cancelled(r);
        } else {
            self.res_ok(r.rid, json!({"size": sent}));
        }
    }

    fn op_fs_write(&self, r: &Req, inbox: &mpsc::Receiver<(String, Vec<u8>)>) {
        let deadline = Instant::now() + Duration::from_secs_f64(self.f("upload_hold_s").max(0.0));
        while Instant::now() < deadline && !r.cancelled() {
            std::thread::sleep(Duration::from_millis(20));
        }
        let mut digest = sha2::Sha256::new();
        let (mut total, mut since) = (0i64, 0i64);
        let batch = self.i("credit_batch");
        while !r.cancelled() {
            let Ok((kind, data)) = inbox.recv_timeout(Duration::from_millis(50)) else {
                continue;
            };
            if kind == "end" {
                self.res_ok(
                    r.rid,
                    json!({"size": total, "sha256": hex::encode(digest.finalize())}),
                );
                return;
            }
            sleep(self.f("upload_consume_delay_s"));
            digest.update(&data);
            total += data.len() as i64;
            since += data.len() as i64;
            self.log("consumed", json!({"id": r.rid, "total": total}));
            if since >= batch {
                r.granted.fetch_add(since, Ordering::SeqCst);
                self.send(&Value::Object(ec_proto::msg::credit(r.rid, since as u64)));
                since = 0;
            }
        }
        self.cancelled(r);
    }

    // ── the main loop ────────────────────────────────────────────────────────────────────────────────────
    fn dispatch(self: &Arc<Self>, msg: &Obj) {
        let t = msg.get("type").and_then(Value::as_str).unwrap_or("");
        let rid_v = msg.get("id").cloned().unwrap_or(Value::Null);
        let rid = rid_v.as_i64().unwrap_or(0);
        let req = || {
            self.reqs
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .get(&rid)
                .cloned()
        };
        match t {
            "ping" => {
                self.log("ping", json!({"id": rid_v}));
                if !self.b("no_pong") {
                    self.send(&json!({"type": "pong", "id": rid_v}));
                }
            }
            "req" => {
                let op = msg
                    .get("op")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let args = msg.get("args").cloned().unwrap_or(json!({}));
                self.log("req", json!({"id": rid_v, "op": op, "args": args}));
                let (tx, rx) = mpsc::channel();
                let r = Arc::new(Req {
                    rid,
                    op,
                    args: args.as_object().cloned().unwrap_or_default(),
                    cancel: AtomicBool::new(false),
                    credit: Mutex::new(ec_proto::WINDOW as i64),
                    cv: Condvar::new(),
                    inbox: Mutex::new(Some(tx)),
                    received: AtomicI64::new(0),
                    granted: AtomicI64::new(0),
                });
                self.reqs
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .insert(rid, Arc::clone(&r));
                let me = Arc::clone(self);
                std::thread::spawn(move || me.serve(r, rx));
            }
            "cancel" => {
                self.log("cancel", json!({"id": rid_v}));
                if let Some(r) = req() {
                    r.cancel.store(true, Ordering::SeqCst);
                    r.cv.notify_all();
                }
            }
            "credit" => {
                let bytes = msg.get("bytes").cloned().unwrap_or(Value::Null);
                self.log("credit", json!({"id": rid_v, "bytes": bytes}));
                if let Some(r) = req() {
                    let mut c = r
                        .credit
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    *c += bytes.as_i64().unwrap_or(0);
                    r.cv.notify_all();
                }
            }
            "end" => {
                self.log("end", json!({"id": rid_v}));
                if let Some(r) = req()
                    && let Some(tx) = r.inbox.lock().ok().and_then(|g| g.clone())
                {
                    let _ = tx.send(("end".into(), Vec::new()));
                }
            }
            other => self.log("other", json!({"type": other})),
        }
    }

    /// test_fleet_service.py `install_fake_ssh`: the fake node started in place of `ssh` records its argv (written
    /// by the wrapper script) and the names of its environment, then fails like ssh does or carries on as a node.
    fn as_ssh(&self) {
        let Some(ssh_log) = self.sc.get("ssh_log").and_then(Value::as_str) else {
            return;
        };
        let raw = std::fs::read(self.s("ssh_args_file")).unwrap_or_default();
        let mut argv: Vec<String> = raw
            .split(|b| *b == 0)
            .map(|a| String::from_utf8_lossy(a).into_owned())
            .collect();
        argv.pop(); // the empty piece after the last NUL
        let mut env: Vec<String> = std::env::vars_os()
            .map(|(k, _)| k.to_string_lossy().into_owned())
            .collect();
        env.sort();
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(ssh_log)
        {
            let _ = writeln!(f, "{}", json!({"argv": argv, "env": env}));
        }
        match self.s("ssh_mode").as_str() {
            "denied" => {
                eprintln!("ops@10.0.0.5: Permission denied (publickey).");
                std::process::exit(255);
            }
            "badkey" => {
                eprintln!("Load key \"{}\": error in libcrypto", self.s("ssh_key"));
                std::process::exit(255);
            }
            _ => {}
        }
    }

    fn main(self: Arc<Self>) {
        self.as_ssh();
        let noise = self.i("stderr_noise");
        if noise > 0 {
            eprint!("{}", "noise line\n".repeat(noise as usize));
        }
        let stderr = self.s("stderr");
        if !stderr.is_empty() {
            eprintln!("{stderr}");
        }
        let _ = std::io::stderr().flush();
        let mut env: Vec<String> = std::env::vars_os()
            .map(|(k, _)| k.to_string_lossy().into_owned())
            .collect();
        env.sort();
        self.log("start", json!({"pid": std::process::id(), "env": env}));
        if let Some(code) = self.sc.get("exit_before_hello").and_then(Value::as_i64) {
            std::process::exit(code as i32);
        }
        let junk = self.s("junk");
        if !junk.is_empty() {
            self.write(junk.as_bytes());
        }
        let junk_lines = self.i("junk_lines");
        if junk_lines > 0 {
            let mut line = vec![b'j'; 1023];
            line.push(b'\n');
            for _ in 0..junk_lines {
                self.write(&line);
            }
            sleep(30.0); // never speaks the protocol
            return;
        }
        sleep(self.f("hello_delay_s"));
        self.write(ec_proto::MAGIC);
        match self.s("first_frame").as_str() {
            "none" => {
                sleep(30.0);
                return;
            }
            "fatal" => {
                self.send(&json!({"type": "fatal", "error": "scripted fatal: no GPU support\nsecond line"}));
                std::process::exit(1);
            }
            "sample" => self.send(&self.sample(0)),
            _ => self.send(&self.hello()),
        }
        if self.b("spawn_child")
            && let Ok(child) = std::process::Command::new("sleep").arg("120").spawn()
        {
            self.log("child", json!({"pid": child.id()}));
        }
        if self.b("silent_after_hello") {
            sleep(3600.0);
        }
        if let Some(Value::Object(d)) = self.sc.get("push_disks") {
            let mut m = json!({"type": "disks", "t": now_wall()});
            merge(&mut m, &Value::Object(d.clone()));
            self.send(&m);
        }
        if let Some(Value::Object(inv)) = self.sc.get("push_inventory") {
            let mut m = json!({"type": "inventory", "t": now_wall()});
            merge(&mut m, &Value::Object(inv.clone()));
            self.send(&m);
        }
        if self.b("samples") {
            let me = Arc::clone(&self);
            std::thread::spawn(move || me.sampler());
        }
        if self
            .sc
            .get("pushes")
            .and_then(Value::as_array)
            .is_some_and(|p| !p.is_empty())
        {
            let me = Arc::clone(&self);
            std::thread::spawn(move || me.scheduled());
        }
        if self.sc.get("flood").is_some_and(|f| !f.is_null()) {
            let me = Arc::clone(&self);
            std::thread::spawn(move || me.flooder());
        }
        if self.sc.get("exit_after_s").is_some_and(|f| !f.is_null()) {
            let me = Arc::clone(&self);
            std::thread::spawn(move || me.exiter());
        }
        let stdin = std::io::stdin();
        let mut input = stdin.lock();
        loop {
            match ec_proto::read_frame_sync(&mut input) {
                Ok(None) => {
                    self.log("eof", json!({}));
                    std::process::exit(0);
                }
                Ok(Some(ec_proto::Frame::Data { stream, data })) => {
                    let r = self
                        .reqs
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .get(&i64::from(stream))
                        .cloned();
                    if let Some(r) = r {
                        let total = r.received.fetch_add(data.len() as i64, Ordering::SeqCst)
                            + data.len() as i64;
                        let granted = r.granted.load(Ordering::SeqCst);
                        if total > ec_proto::WINDOW as i64 + granted {
                            self.log(
                                "violation",
                                json!({"id": r.rid, "received": total, "granted": granted}),
                            );
                        }
                        self.log("data", json!({"id": r.rid, "total": total}));
                        if let Some(tx) = r.inbox.lock().ok().and_then(|g| g.clone()) {
                            let _ = tx.send(("data".into(), data));
                        }
                    }
                }
                Ok(Some(ec_proto::Frame::Json(msg))) => self.dispatch(&msg),
                Err(e) => {
                    eprintln!("fake node: {e}");
                    std::process::exit(1);
                }
            }
        }
    }
}
