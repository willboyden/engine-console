//! Port of node/tests/test_server.py: the real node server loop over OS pipes, driven by a hub that speaks real
//! proto frames. The op handlers the Python tests exercise (fs.write, fs.read, docker.logs_follow, docker.images)
//! replaced here by small fakes with the same stream behaviour (credit, uploads, cancel), injected through
//! `ServerOptions::handler`, so these tests pin the SERVER's mechanics; the real handlers have their own tests.
#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    unreachable_pub,
    clippy::cast_possible_truncation
)]

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, channel};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use ec_proto::{Frame, MAX_DATA, Obj, WINDOW, msg, specs};
use ecnode::agent::{Agent, AgentParts};
use ecnode::discovery::{PortProbe, Probe};
use ecnode::errors::{Call, Cancelled, HandlerError, OpError};
use ecnode::gpus::Gpus;
use ecnode::policy::{Policy, PolicyStore};
use ecnode::server::{Handler, Server, ServerOptions, Timing};
use ecnode::util::{EnvMap, ProcRunner, Runner};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const STEP: usize = 1024 * 1024;
const ARGV_SECRET: &str = "sk-test-not-a-real-key-0000";

fn fast() -> Timing {
    Timing {
        sample_s: 0.2,
        disks_s: 0.5,
        inventory_s: 1.0,
        inventory_poll_s: 0.3,
        hub_silence_s: 30.0,
        write_stall_s: 10.0,
        watchdog_s: 0.05,
    }
}

fn manage_policy() -> Policy {
    Policy {
        path: Some("/test/policy.json".into()),
        mode: "manage".into(),
        ..Policy::default()
    }
}

fn monitor_policy() -> Policy {
    Policy {
        path: Some("/test/policy.json".into()),
        ..Policy::default()
    }
}

fn obj(v: Value) -> Obj {
    match v {
        Value::Object(o) => o,
        _ => panic!("not an object"),
    }
}

// ── a node that cannot reach anything real (PORTING.md §A2) ────────────────────────────────────────────────
const FAKES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../parity/fakes/bin");

/// The agent of every server test: its tools are found in ECNODE_TOOL_PATH = the parity fakes then /usr/bin:/bin
/// (the fake `docker`, `nvidia-smi` and `ip`; the node never searches PATH), `ECNODE_NO_NVML=1` and a HOME that does
/// not exist; NVML is off; every port probe and every metrics GET is refused without opening a socket (the default
/// policy's discovery ports include 8000, 11434 and 30000, where real engines may listen); the docker socket default
/// points nowhere. Nothing of this test process's own environment reaches the node.
fn hermetic_agent(store: PolicyStore) -> Agent {
    let env: EnvMap = [
        ("PATH", format!("{FAKES}:/usr/bin:/bin")),
        ("ECNODE_TOOL_PATH", format!("{FAKES}:/usr/bin:/bin")),
        ("HOME", "/nonexistent/ecnode-test-home".to_string()),
        ("LANG", "C.UTF-8".to_string()),
        ("ECNODE_NO_NVML", "1".to_string()),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();
    let runner: Arc<dyn Runner> = Arc::new(ProcRunner::new(env.clone()));
    let refuse = |port: u16| PortProbe {
        port,
        ..PortProbe::default()
    };
    Agent::with_parts(
        store,
        AgentParts {
            gpus: Some(Gpus::new(false, Arc::clone(&runner))),
            runner,
            env,
            prober: Arc::new(refuse),
            quiet_prober: Arc::new(refuse),
            metrics_get: Arc::new(|_, _, _| Probe::default()),
            default_socket: "/nonexistent/ecnode-test/docker.sock".into(),
            ..AgentParts::default()
        },
    )
}

// ── fake handlers (the shapes of ops.py's, without the filesystem or docker) ───────────────────────────────
type Files = Arc<Mutex<HashMap<String, Vec<u8>>>>;

fn fake_handler(files: Files) -> Handler {
    Arc::new(
        move |agent: &Agent, call: &Call<'_>| -> Result<Obj, HandlerError> {
            let a = call.args;
            let s = |k: &str| a.get(k).and_then(Value::as_str).unwrap_or("").to_string();
            match call.op {
                "docker.images" => Ok(obj(json!({"images": []}))),
                // a handler bug (node hardening): it must cost this request only
                "docker.image_present" if s("image") == "panic/now:1" => {
                    panic!("handler bug while running docker --api-key {ARGV_SECRET}")
                }
                "fs.write" => {
                    if s("root") == "ro" {
                        return Err(OpError::new(
                            ec_proto::POLICY_DENIED,
                            "model root 'ro' is read-only in this node's policy",
                        )
                        .into());
                    }
                    if s("rel") == "greedy.bin" {
                        thread::sleep(Duration::from_secs(1)); // the node grants nothing while the hub floods it
                    }
                    let size = a.get("size").and_then(Value::as_u64).unwrap_or(0);
                    let mut buf = Vec::new();
                    loop {
                        let chunk = call.stream.read_upload()?;
                        if chunk.is_empty() {
                            break;
                        }
                        buf.extend_from_slice(&chunk);
                    }
                    if buf.len() as u64 != size {
                        return Err(OpError::new(ec_proto::SIZE_MISMATCH, "size").into());
                    }
                    let sha = hex::encode(Sha256::digest(&buf));
                    files.lock().unwrap().insert(s("rel"), buf);
                    Ok(obj(json!({"size": size, "sha256": sha})))
                }
                "fs.read" => {
                    let data = files
                        .lock()
                        .unwrap()
                        .get(&s("rel"))
                        .cloned()
                        .unwrap_or_default();
                    let mut sent = 0;
                    for chunk in data.chunks(512 * 1024) {
                        call.stream.check()?;
                        call.stream.send_data(chunk)?;
                        sent += chunk.len();
                    }
                    Ok(obj(json!({"size": sent})))
                }
                "docker.logs_follow" => {
                    let stop = Arc::new(AtomicBool::new(false));
                    let s2 = Arc::clone(&stop);
                    call.stream
                        .on_cancel(Box::new(move || s2.store(true, Ordering::SeqCst)));
                    for line in [
                        "INFO starting",
                        &format!("non-default args: {{'api_key': ['{ARGV_SECRET}']}}"),
                        "INFO ready",
                    ] {
                        call.stream
                            .event(obj(json!({"line": ecnode::util::scrub(line)})));
                    }
                    while !stop.load(Ordering::SeqCst) && !call.stream.cancelled() {
                        thread::sleep(Duration::from_millis(50));
                    }
                    Err(Cancelled.into())
                }
                _ => ecnode::ops::dispatch(agent, call),
            }
        },
    )
}

// ── the hub's end of the pipes ───────────────────────────────────────────────────────────────────────────
struct Hub {
    to: Option<File>,
    rx: Receiver<Option<Frame>>,
    pushed: Vec<Obj>,
    skipped: Arc<Mutex<Option<usize>>>,
    rid: i64,
}

#[derive(Default)]
struct CallOut {
    events: Vec<Obj>,
    data: Vec<u8>,
    credit_frames: usize,
    credit_bytes: u64,
    res: Obj,
}

impl Hub {
    fn new(to: File, from: File) -> Self {
        let (tx, rx) = channel();
        let skipped = Arc::new(Mutex::new(None));
        let sk = Arc::clone(&skipped);
        thread::spawn(move || {
            let mut r = BufReader::new(from);
            if let Ok(n) = ec_proto::find_magic_sync(&mut r) {
                *sk.lock().unwrap() = Some(n);
                while let Ok(Some(f)) = ec_proto::read_frame_sync(&mut r) {
                    if tx.send(Some(f)).is_err() {
                        return;
                    }
                }
            }
            let _ = tx.send(None);
        });
        Self {
            to: Some(to),
            rx,
            pushed: Vec::new(),
            skipped,
            rid: 100,
        }
    }

    fn rid(&mut self) -> i64 {
        self.rid += 1;
        self.rid
    }

    fn raw(&mut self, b: &[u8]) {
        let to = self.to.as_mut().unwrap();
        to.write_all(b).unwrap();
        to.flush().unwrap();
    }

    fn send(&mut self, m: &Obj) {
        self.raw(&ec_proto::encode_json(m).unwrap());
    }

    fn next(&self, timeout: Duration) -> Result<Option<Frame>, RecvTimeoutError> {
        self.rx.recv_timeout(timeout)
    }

    fn push_of(&mut self, kind: &str) -> Obj {
        if let Some(i) = self
            .pushed
            .iter()
            .position(|m| m.get("type").and_then(Value::as_str) == Some(kind))
        {
            return self.pushed.remove(i);
        }
        let deadline = Instant::now() + Duration::from_secs(10);
        while Instant::now() < deadline {
            match self.next(deadline.saturating_duration_since(Instant::now())) {
                Ok(Some(Frame::Json(m))) => {
                    if m.get("type").and_then(Value::as_str) == Some(kind) {
                        return m;
                    }
                    self.pushed.push(m);
                }
                Ok(Some(Frame::Data { .. })) | Err(RecvTimeoutError::Timeout) => {}
                Ok(None) | Err(RecvTimeoutError::Disconnected) => {
                    panic!("the node closed the stream")
                }
            }
        }
        panic!("no {kind} within 10 s");
    }

    /// One request. Uploads respect the node's credit; downloads are credited back as they are consumed, and the
    /// test fails the moment the node sends more than it was granted.
    fn call(
        &mut self,
        op: &str,
        args: Value,
        upload: Option<&[u8]>,
        cancel_after: Option<usize>,
    ) -> CallOut {
        let rid = self.rid();
        self.send(&msg::req(rid, op, Some(obj(args))));
        let (mut credit, mut sent, mut ended) = (WINDOW as usize, 0usize, upload.is_none());
        let (mut window, mut got, mut acked) = (WINDOW as usize, 0usize, 0usize);
        let mut cancel_after = cancel_after;
        let mut out = CallOut::default();
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            if !ended {
                let up = upload.unwrap();
                while sent < up.len() && credit > 0 {
                    let n = MAX_DATA.min(credit).min(up.len() - sent);
                    let f = ec_proto::encode_data(rid, &up[sent..sent + n]).unwrap();
                    self.raw(&f);
                    sent += n;
                    credit -= n;
                }
                if sent == up.len() {
                    self.send(&msg::end(rid));
                    ended = true;
                }
            }
            if let Some(n) = cancel_after
                && out.events.len() >= n
            {
                self.send(&msg::cancel(rid));
                cancel_after = None;
            }
            assert!(Instant::now() < deadline, "timed out");
            let f = match self.next(Duration::from_millis(100)) {
                Ok(Some(f)) => f,
                Err(RecvTimeoutError::Timeout) => continue,
                _ => panic!("the node closed the stream"),
            };
            match f {
                Frame::Data { stream, data } => {
                    assert_eq!(i64::from(stream), rid);
                    got += data.len();
                    assert!(
                        got <= window,
                        "node sent {got} bytes with only {window} granted"
                    );
                    out.data.extend_from_slice(&data);
                    if got - acked >= STEP {
                        window += got - acked;
                        self.send(&msg::credit(rid, (got - acked) as u64));
                        acked = got;
                    }
                }
                Frame::Json(m) => {
                    let t = m
                        .get("type")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();
                    if m.get("id").and_then(Value::as_i64) != Some(rid)
                        || !["credit", "event", "res"].contains(&t.as_str())
                    {
                        self.pushed.push(m);
                    } else if t == "credit" {
                        credit += m["bytes"].as_u64().unwrap() as usize;
                        out.credit_frames += 1;
                        out.credit_bytes += m["bytes"].as_u64().unwrap();
                    } else if t == "event" {
                        out.events.push(m["event"].as_object().unwrap().clone());
                    } else {
                        out.res = m;
                        return out;
                    }
                }
            }
        }
    }
}

struct Node {
    server: Arc<Server>,
    hub: Hub,
    done: Receiver<i32>,
}

impl Node {
    fn start(store: PolicyStore, opts: ServerOptions) -> Self {
        let (r1, w1) = nix::unistd::pipe().unwrap();
        let (r2, w2) = nix::unistd::pipe().unwrap();
        let server = Arc::new(Server::new(Arc::new(hermetic_agent(store)), opts));
        let (tx, done) = channel();
        let s = Arc::clone(&server);
        let (inp, out) = (File::from(r1), File::from(w2));
        thread::spawn(move || {
            let code = s.run(Box::new(inp), Box::new(out));
            let _ = tx.send(code);
        });
        Self {
            server,
            hub: Hub::new(File::from(w1), File::from(r2)),
            done,
        }
    }

    fn with(policy: Policy, scanner: bool) -> (Self, Files) {
        let files: Files = Arc::default();
        let opts = ServerOptions {
            timing: fast(),
            on_stall: Arc::new(|| {}),
            scanner,
            handler: fake_handler(Arc::clone(&files)),
            ..ServerOptions::default()
        };
        (Self::start(PolicyStore::from_policy(policy), opts), files)
    }

    /// Close stdin and wait for the server to exit; its exit code.
    fn close(mut self) -> i32 {
        self.hub.to = None;
        self.done
            .recv_timeout(Duration::from_secs(15))
            .expect("the node did not exit after stdin closed")
    }
}

#[test]
fn test_session_hello_ping_sample_requests_and_eof() {
    let (mut n, _) = Node::with(manage_policy(), true);
    let hello = n.hub.push_of("hello");
    assert_eq!(*n.hub.skipped.lock().unwrap(), Some(0));
    assert_eq!(hello["proto"], json!(ec_proto::PROTO));
    assert_eq!(hello["policy"]["mode"], "manage");
    assert!(ec_proto::sanitize_push(&hello).unwrap().is_some());
    n.hub.send(&msg::ping(7));
    assert_eq!(
        Value::Object(n.hub.push_of("pong")),
        json!({"type": "pong", "id": 7})
    );
    let sample = ec_proto::sanitize_push(&n.hub.push_of("sample"))
        .unwrap()
        .unwrap();
    assert_eq!(sample["seq"], 1);
    n.hub.push_of("disks");
    n.hub.push_of("inventory");
    let info = n.hub.call("node.info", json!({}), None, None).res;
    assert_eq!(info["ok"], true);
    assert_eq!(info["result"]["node_id"], hello["node_id"]);
    assert!(info["result"].get("type").is_none());
    // (Python also checks GPUs, memory, containers and models_changed: the collectors' and scanner's own tests.)
    let code = n.close();
    assert_eq!(code, 0);
}

#[test]
fn test_upload_bigger_than_the_window_flows_on_credit() {
    let (mut n, files) = Node::with(manage_policy(), false);
    n.hub.push_of("hello");
    let payload: Vec<u8> = (0..(WINDOW as usize + 3 * STEP + 123))
        .map(|i| (i * 31 + 7) as u8)
        .collect();
    let sha = hex::encode(Sha256::digest(&payload));
    let out = n.hub.call(
        "fs.write",
        json!({"root": "gguf", "rel": "big/model.gguf", "size": payload.len(), "sha256": sha}),
        Some(&payload),
        None,
    );
    assert_eq!(out.res["ok"], true, "{:?}", out.res);
    assert_eq!(
        out.res["result"],
        json!({"size": payload.len(), "sha256": sha})
    );
    assert!(out.credit_frames >= 1 && out.credit_bytes >= payload.len() as u64 - WINDOW);
    assert_eq!(files.lock().unwrap()["big/model.gguf"], payload);
    // the same bytes back: a download that respects the hub's credit
    let down = n.hub.call(
        "fs.read",
        json!({"root": "gguf", "rel": "big/model.gguf"}),
        None,
        None,
    );
    assert_eq!(down.res["ok"], true);
    assert_eq!(down.data, payload);
    assert_eq!(down.res["result"]["size"], json!(payload.len()));
}

#[test]
fn test_download_stops_at_the_credit_until_more_is_granted() {
    let (mut n, files) = Node::with(manage_policy(), false);
    n.hub.push_of("hello");
    let blob: Vec<u8> = (0..(WINDOW as usize + 2 * STEP))
        .map(|i| (i % 251) as u8)
        .collect();
    files.lock().unwrap().insert("x.gguf".into(), blob.clone());
    let rid = n.hub.rid();
    n.hub.send(&msg::req(
        rid,
        "fs.read",
        Some(obj(json!({"root": "gguf", "rel": "x.gguf"}))),
    ));
    let mut got = 0usize;
    let deadline = Instant::now() + Duration::from_secs(15);
    while got < WINDOW as usize && Instant::now() < deadline {
        if let Ok(Some(Frame::Data { data, .. })) = n.hub.next(Duration::from_secs(5)) {
            got += data.len();
        }
    }
    assert_eq!(got, WINDOW as usize);
    let quiet_until = Instant::now() + Duration::from_secs(1); // for a second: pushes yes, DATA no
    while Instant::now() < quiet_until {
        match n.hub.next(Duration::from_millis(100)) {
            Ok(Some(Frame::Json(_))) | Err(RecvTimeoutError::Timeout) => {}
            other => panic!("DATA beyond the granted credit: {other:?}"),
        }
    }
    n.hub.send(&msg::credit(rid, blob.len() as u64));
    loop {
        match n.hub.next(Duration::from_secs(10)).unwrap().unwrap() {
            Frame::Data { data, .. } => got += data.len(),
            Frame::Json(m) if m["type"] == "res" && m["id"] == rid => {
                assert_eq!(m["ok"], true);
                assert_eq!(got, blob.len());
                break;
            }
            Frame::Json(_) => {}
        }
    }
}

#[test]
fn test_cancel_stops_a_stream_and_answers_cancelled() {
    let (mut n, _) = Node::with(manage_policy(), false);
    n.hub.push_of("hello");
    let out = n.hub.call(
        "docker.logs_follow",
        json!({"name": "ec-owned-1", "tail": 10}),
        None,
        Some(3),
    );
    assert_eq!(out.res["ok"], false);
    assert_eq!(out.res["error"]["code"], ec_proto::CANCELLED);
    let lines: Vec<&str> = out
        .events
        .iter()
        .map(|e| e["line"].as_str().unwrap())
        .collect();
    assert!(lines.contains(&"INFO ready"));
    assert!(!lines.iter().any(|l| l.contains(ARGV_SECRET)));
}

/// Node hardening: a handler that panics answers `internal` (no panic text, no secret in it), frees its slot, and
/// the node serves the next requests as before, the push loops included.
#[test]
fn test_a_panicking_handler_answers_internal_and_the_next_request_works() {
    let (mut n, _) = Node::with(manage_policy(), false);
    n.hub.push_of("hello");
    for _ in 0..3 {
        let res = n
            .hub
            .call(
                "docker.image_present",
                json!({"image": "panic/now:1"}),
                None,
                None,
            )
            .res;
        assert_eq!(res["ok"], false);
        assert_eq!(res["error"]["code"], ec_proto::INTERNAL);
        assert_eq!(
            res["error"]["message"],
            "the node hit an unexpected error in docker.image_present (panic); see the node's stderr"
        );
        assert!(!Value::Object(res).to_string().contains(ARGV_SECRET));
    }
    let res = n.hub.call("docker.images", json!({}), None, None).res;
    assert_eq!(res["ok"], true, "{res:?}");
    n.hub.send(&msg::ping(9));
    assert_eq!(n.hub.push_of("pong")["id"], 9);
    n.hub.push_of("sample");
    assert_eq!(n.close(), 0);
}

#[test]
fn test_unknown_op_bad_args_and_unknown_messages() {
    let (mut n, _) = Node::with(manage_policy(), false);
    n.hub.push_of("hello");
    let res = n
        .hub
        .call("shell.exec", json!({"cmd": "id"}), None, None)
        .res;
    assert_eq!(res["error"]["code"], ec_proto::UNKNOWN_OP);
    assert_eq!(
        res["error"]["message"],
        "this node does not know the op 'shell.exec' (ecnode protocol 1)"
    );
    let res = n
        .hub
        .call("fs.read", json!({"root": "gguf"}), None, None)
        .res;
    assert_eq!(res["error"]["code"], ec_proto::BAD_ARGS);
    assert_eq!(res["error"]["message"], "fs.read needs rel");
    n.hub.send(&obj(json!({"type": "who-knows", "id": 5})));
    n.hub.send(&msg::cancel(999_999)); // cancel / credit / end for unknown ids are ignored
    n.hub.send(&msg::credit(999_999, 5));
    n.hub.send(&msg::end(999_999));
    n.hub.send(&msg::ping(8));
    assert_eq!(n.hub.push_of("pong")["id"], 8);
}

#[test]
fn test_audit_f5_a_path_name_image_or_argv_that_cleaning_would_cut_is_refused() {
    // security audit F5: proto `clean` caps `rel` at 1024 characters, so Python acted on the first 1024 characters of
    // the path the hub named (another file); the node refuses such a request before any handler sees it
    let (mut n, files) = Node::with(manage_policy(), false);
    n.hub.push_of("hello");
    let refused = |n: &mut Node, op: &str, args: Value, upload: Option<&[u8]>| -> String {
        let res = n.hub.call(op, args, upload, None).res;
        assert_eq!(res["ok"], false, "{op}: {res:?}");
        assert_eq!(res["error"]["code"], ec_proto::BAD_ARGS, "{op}: {res:?}");
        res["error"]["message"].as_str().unwrap().to_string()
    };
    let long = format!("{}/x.bin", "d".repeat(1100));
    let m = refused(
        &mut n,
        "fs.write",
        json!({"root": "gguf", "rel": long, "size": 3}),
        Some(b"abc"),
    );
    assert_eq!(
        m,
        "fs.write: rel is longer than 1024 characters; the node refuses it rather than cut it short"
    );
    assert!(files.lock().unwrap().is_empty());
    let m = refused(
        &mut n,
        "fs.read",
        json!({"root": "gguf", "rel": "a\u{0}/b"}),
        None,
    );
    assert_eq!(
        m,
        "fs.read: rel contains a NUL character; the node refuses it rather than cut it short"
    );
    let m = refused(
        &mut n,
        "docker.stop",
        json!({"name": "ec-".repeat(43)}),
        None,
    );
    assert!(
        m.starts_with("docker.stop: name is longer than 128 characters"),
        "{m}"
    );
    let argv: Vec<String> = (0..513).map(|i| format!("--a{i}")).collect();
    let m = refused(
        &mut n,
        "docker.render",
        json!({"spec": {"name": "ec-x", "image": "vllm/vllm-openai:v1", "argv": argv}}),
        None,
    );
    assert!(
        m.starts_with("docker.render: spec.argv has more than 512 items"),
        "{m}"
    );
    let m = refused(
        &mut n,
        "docker.render",
        json!({"spec": {"name": "ec-x", "argv": ["vllm", "x".repeat(4097)]}}),
        None,
    );
    assert!(
        m.starts_with("docker.render: spec.argv[1] is longer than 4096 characters"),
        "{m}"
    );
    let m = refused(
        &mut n,
        "docker.render",
        json!({"spec": {"name": "ec-x", "mounts": [{"root": "hf", "rel": "r".repeat(1025), "dest": "/m"}]}}),
        None,
    );
    assert!(
        m.starts_with("docker.render: spec.mounts[0].rel is longer than 1024 characters"),
        "{m}"
    );
    let m = refused(
        &mut n,
        "docker.pull",
        json!({"image": format!("vllm/vllm-openai:{}", "t".repeat(250))}),
        None,
    );
    assert!(
        m.starts_with("docker.pull: image is longer than 256 characters"),
        "{m}"
    );
    // exactly at the cap is fine, and the handler gets the value whole
    let at_cap = "e".repeat(1024);
    let res = n
        .hub
        .call(
            "fs.write",
            json!({"root": "gguf", "rel": at_cap, "size": 3}),
            Some(b"abc"),
            None,
        )
        .res;
    assert_eq!(res["ok"], true, "{res:?}");
    assert!(files.lock().unwrap().contains_key(&at_cap));
    n.hub.send(&msg::ping(11)); // the link is still fine
    assert_eq!(n.hub.push_of("pong")["id"], 11);
}

#[test]
fn test_monitor_mode_refuses_every_op_that_changes_the_machine() {
    let (mut n, _) = Node::with(monitor_policy(), false);
    n.hub.push_of("hello");
    let mut ops: Vec<&str> = specs::OPS
        .iter()
        .copied()
        .filter(|o| !specs::is_monitor_op(o))
        .collect();
    ops.sort_unstable();
    for op in ops {
        let res = n.hub.call(op, json!({}), None, None).res;
        assert_eq!(res["ok"], false, "{op}");
        assert_eq!(res["error"]["code"], ec_proto::MONITOR_ONLY, "{op}");
        assert_eq!(
            res["error"]["message"],
            format!(
                "this node's policy is in monitor mode, so {op}, which would change the machine, is refused; an \
                 administrator can set \"mode\": \"manage\" in /test/policy.json"
            )
        );
    }
    assert_eq!(
        n.hub.call("node.info", json!({}), None, None).res["ok"],
        true
    );
    assert_eq!(
        n.hub.call("docker.images", json!({}), None, None).res["ok"],
        true
    );
}

#[test]
fn test_no_policy_file_is_monitor_mode_too() {
    let tmp = tempfile::tempdir().unwrap();
    let store = PolicyStore::with_paths(
        None,
        Some(tmp.path().to_string_lossy().into_owned()),
        tmp.path().join("none").to_string_lossy().into_owned(),
    );
    let opts = ServerOptions {
        timing: fast(),
        on_stall: Arc::new(|| {}),
        scanner: false,
        ..ServerOptions::default()
    };
    let mut n = Node::start(store, opts);
    let hello = n.hub.push_of("hello");
    assert_eq!(hello["policy"]["mode"], "monitor");
    assert!(
        hello["policy"]["error"]
            .as_str()
            .unwrap()
            .contains("no policy file")
    );
    let res = n
        .hub
        .call("docker.run", json!({"spec": {"name": "ec-x"}}), None, None)
        .res;
    assert_eq!(res["error"]["code"], ec_proto::MONITOR_ONLY);
}

#[test]
fn test_busy_when_every_slot_is_taken() {
    let files: Files = Arc::default();
    let opts = ServerOptions {
        timing: fast(),
        on_stall: Arc::new(|| {}),
        scanner: false,
        handler: fake_handler(files),
        max_inflight: 1,
    };
    let mut n = Node::start(PolicyStore::from_policy(manage_policy()), opts);
    n.hub.push_of("hello");
    let rid = n.hub.rid();
    n.hub.send(&msg::req(
        rid,
        "docker.logs_follow",
        Some(obj(json!({"name": "ec-owned-1"}))),
    ));
    thread::sleep(Duration::from_millis(500));
    let res = n.hub.call("node.info", json!({}), None, None).res;
    assert_eq!(res["error"]["code"], ec_proto::BUSY);
    assert_eq!(
        res["error"]["message"],
        "this node is already running 1 requests; try again shortly"
    );
    n.hub.send(&msg::cancel(rid));
}

#[test]
fn test_duplicate_in_flight_id_is_ignored_with_a_warning() {
    // server.py `_on_req`: the dup branch (node/tests have no test of its own for it)
    let (mut n, _) = Node::with(manage_policy(), false);
    n.hub.push_of("hello");
    let rid = n.hub.rid();
    let req = msg::req(
        rid,
        "docker.logs_follow",
        Some(obj(json!({"name": "ec-owned-1"}))),
    );
    n.hub.send(&req);
    thread::sleep(Duration::from_millis(300));
    n.hub.send(&req);
    let log = n.hub.push_of("log");
    assert_eq!(
        log["msg"],
        format!("ignored request {rid}: that id is already in flight")
    );
    assert_eq!(log["level"], "warn");
    n.hub.send(&msg::cancel(rid));
}

#[test]
fn test_refused_upload_answers_at_once_and_drops_the_rest() {
    let (mut n, _) = Node::with(manage_policy(), false);
    n.hub.push_of("hello");
    let up = vec![b'z'; 3 * STEP];
    let out = n.hub.call(
        "fs.write",
        json!({"root": "ro", "rel": "x.bin", "size": 3 * STEP}),
        Some(&up),
        None,
    );
    assert_eq!(out.res["error"]["code"], ec_proto::POLICY_DENIED);
    n.hub.send(&msg::ping(9)); // the link is still fine
    assert_eq!(n.hub.push_of("pong")["id"], 9);
}

#[test]
fn test_upload_beyond_credit_fails_that_request_only() {
    let (mut n, files) = Node::with(manage_policy(), false);
    n.hub.push_of("hello");
    let rid = n.hub.rid();
    let size = WINDOW as usize + 4 * STEP;
    n.hub.send(&msg::req(
        rid,
        "fs.write",
        Some(obj(
            json!({"root": "gguf", "rel": "greedy.bin", "size": size}),
        )),
    ));
    let chunk = vec![b'q'; MAX_DATA];
    for _ in 0..(size / MAX_DATA) {
        // ignores credit on purpose
        n.hub.raw(&ec_proto::encode_data(rid, &chunk).unwrap());
    }
    n.hub.send(&msg::end(rid));
    let deadline = Instant::now() + Duration::from_secs(15);
    let mut res = None;
    while res.is_none() && Instant::now() < deadline {
        if let Ok(Some(Frame::Json(m))) = n.hub.next(Duration::from_secs(5))
            && m["type"] == "res"
            && m["id"] == rid
        {
            res = Some(m);
        }
    }
    let res = res.expect("no res");
    assert_eq!(res["ok"], false);
    assert_eq!(res["error"]["code"], ec_proto::BAD_ARGS);
    assert!(res["error"]["message"].as_str().unwrap().contains("credit"));
    assert!(!files.lock().unwrap().contains_key("greedy.bin"));
    n.hub.send(&msg::ping(10));
    assert_eq!(n.hub.push_of("pong")["id"], 10);
}

#[test]
fn test_hub_silence_ends_the_session() {
    let opts = ServerOptions {
        timing: Timing {
            sample_s: 0.2,
            hub_silence_s: 0.6,
            watchdog_s: 0.05,
            ..Timing::default()
        },
        on_stall: Arc::new(|| {}),
        scanner: false,
        ..ServerOptions::default()
    };
    let mut n = Node::start(PolicyStore::from_policy(manage_policy()), opts);
    n.hub.push_of("hello");
    let code = n
        .done
        .recv_timeout(Duration::from_secs(10))
        .expect("silence did not end the session");
    assert_eq!(code, 0);
}

#[test]
fn test_protocol_error_from_the_hub_drops_the_link() {
    let (mut n, _) = Node::with(manage_policy(), false);
    n.hub.push_of("hello");
    n.hub.raw(b"\x00\x00\x00\x05\x09junk"); // unknown frame kind
    let code = n
        .done
        .recv_timeout(Duration::from_secs(10))
        .expect("the node did not exit");
    assert_eq!(code, 2);
    assert_eq!(n.server.exit_code(), 2);
}

/// A writer that blocks until released (server.py's watchdog test uses the same trick).
struct Stuck(Arc<(Mutex<bool>, Condvar)>);

impl Write for Stuck {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        let (m, cv) = &*self.0;
        let g = m.lock().unwrap();
        let _ = cv
            .wait_timeout_while(g, Duration::from_secs(10), |open| !*open)
            .unwrap();
        Ok(b.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
fn test_blocked_writer_trips_the_watchdog() {
    let gate = Arc::new((Mutex::new(false), Condvar::new()));
    let stalled = Arc::new((Mutex::new(false), Condvar::new()));
    let st = Arc::clone(&stalled);
    let opts = ServerOptions {
        timing: Timing {
            write_stall_s: 0.3,
            watchdog_s: 0.05,
            ..Timing::default()
        },
        on_stall: Arc::new(move || {
            *st.0.lock().unwrap() = true;
            st.1.notify_all();
        }),
        scanner: false,
        ..ServerOptions::default()
    };
    let (r, w) = nix::unistd::pipe().unwrap();
    let server = Arc::new(Server::new(
        Arc::new(hermetic_agent(PolicyStore::from_policy(manage_policy()))),
        opts,
    ));
    let s = Arc::clone(&server);
    let out = Stuck(Arc::clone(&gate));
    let th = thread::spawn(move || s.run(Box::new(File::from(r)), Box::new(out)));
    let tripped = {
        let g = stalled.0.lock().unwrap();
        *stalled
            .1
            .wait_timeout_while(g, Duration::from_secs(5), |s| !*s)
            .unwrap()
            .0
    };
    *gate.0.lock().unwrap() = true;
    gate.1.notify_all();
    drop(w);
    assert!(tripped, "a blocked write did not trip the watchdog");
    th.join().unwrap();
}
