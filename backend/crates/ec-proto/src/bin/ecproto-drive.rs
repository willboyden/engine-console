//! `ecproto-drive`: the hub-side protocol driver in Rust (twin of backend/parity/proto/drive.py, same scenario and
//! transcript formats). It proves the Rust framing, encoding and credit handling against a real node:
//! backend/parity/proto/matrix.mjs runs it against the Python node and checks its transcript equals the golden one.
//!
//!   ecproto-drive --node '<argv as JSON>' --scenario <file.json>
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::collections::HashMap;
use std::io::{BufReader, Read, Write};
use std::process::{Child, ChildStdin, Command, ExitStatus, Stdio};
use std::sync::mpsc::{Receiver, RecvTimeoutError, channel};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ec_proto::{Frame, MAX_DATA, Obj, WINDOW, msg};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const CREDIT_BATCH: u64 = WINDOW / 4;

/// byte i = (i * 31 + 7) % 251
fn pattern(n: usize) -> Vec<u8> {
    (0..n)
        .map(|i| u8::try_from((i % 251 * 31 + 7) % 251).unwrap_or(0))
        .collect()
}

struct Req {
    rid: i64,
    events: Vec<Value>,
    res: Option<Obj>,
    hasher: Sha256,
    got: u64,
    recv_window: i64,
    unacked: u64,
    violation: bool,
    credit_from_node: u64,
    upload: Option<Vec<u8>>,
    sent: usize,
    send_credit: u64,
    ended: bool,
    grant: bool,
    cancel_after: Option<u64>,
    cancel_after_events: Option<usize>,
    cancel_sent: bool,
}

impl Req {
    fn new(
        rid: i64,
        upload: Option<Vec<u8>>,
        grant: bool,
        cancel_after: Option<u64>,
        cae: Option<usize>,
    ) -> Self {
        let ended = upload.is_none();
        Self {
            rid,
            events: Vec::new(),
            res: None,
            hasher: Sha256::new(),
            got: 0,
            recv_window: i64::try_from(WINDOW).unwrap_or(i64::MAX),
            unacked: 0,
            violation: false,
            credit_from_node: 0,
            upload,
            sent: 0,
            send_credit: WINDOW,
            ended,
            grant,
            cancel_after,
            cancel_after_events: cae,
            cancel_sent: false,
        }
    }
}

struct Driver {
    child: Child,
    stdin: Option<ChildStdin>,
    rx: Receiver<Option<Frame>>,
    ended: bool,
    skipped: Arc<Mutex<Option<usize>>>,
    err: Arc<Mutex<Vec<u8>>>,
    hello: Option<Obj>,
    first: HashMap<String, Obj>,
    counts: HashMap<String, usize>,
    last: HashMap<String, Obj>,
    pongs: HashMap<i64, Obj>,
    active: HashMap<i64, Req>,
}

impl Driver {
    fn spawn(argv: &[String]) -> std::io::Result<Self> {
        let mut child = Command::new(&argv[0])
            .args(&argv[1..])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        let stdin = child.stdin.take();
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        let (tx, rx) = channel();
        let skipped = Arc::new(Mutex::new(None));
        let sk = Arc::clone(&skipped);
        std::thread::spawn(move || {
            if let Some(out) = stdout {
                let mut r = BufReader::with_capacity(MAX_DATA + 4096, out);
                if let Ok(n) = ec_proto::find_magic_sync(&mut r) {
                    if let Ok(mut g) = sk.lock() {
                        *g = Some(n);
                    }
                    while let Ok(Some(f)) = ec_proto::read_frame_sync(&mut r) {
                        if tx.send(Some(f)).is_err() {
                            return;
                        }
                    }
                }
            }
            let _ = tx.send(None);
        });
        let err = Arc::new(Mutex::new(Vec::new()));
        let e2 = Arc::clone(&err);
        std::thread::spawn(move || {
            if let Some(mut s) = stderr {
                let mut buf = [0u8; 4096];
                while let Ok(n) = s.read(&mut buf) {
                    if n == 0 {
                        break;
                    }
                    if let Ok(mut g) = e2.lock() {
                        g.extend_from_slice(&buf[..n]);
                    }
                }
            }
        });
        Ok(Self {
            child,
            stdin,
            rx,
            ended: false,
            skipped,
            err,
            hello: None,
            first: HashMap::new(),
            counts: HashMap::new(),
            last: HashMap::new(),
            pongs: HashMap::new(),
            active: HashMap::new(),
        })
    }

    fn raw(&mut self, b: &[u8]) {
        if let Some(s) = self.stdin.as_mut()
            && (s.write_all(b).is_err() || s.flush().is_err())
        {
            self.stdin = None;
        }
    }

    fn send(&mut self, m: &Obj) {
        if let Ok(f) = ec_proto::encode_json(m) {
            self.raw(&f);
        }
    }

    fn send_value(&mut self, v: &Value) {
        if let Ok(f) = ec_proto::encode_value(v) {
            self.raw(&f);
        }
    }

    fn feed_uploads(&mut self) {
        let mut out: Vec<Vec<u8>> = Vec::new();
        for r in self.active.values_mut() {
            if r.ended || r.res.is_some() {
                continue;
            }
            let Some(up) = r.upload.as_ref() else {
                continue;
            };
            while r.sent < up.len() && r.send_credit > 0 {
                let credit = usize::try_from(r.send_credit).unwrap_or(usize::MAX);
                let n = MAX_DATA.min(credit).min(up.len() - r.sent);
                if let Ok(f) = ec_proto::encode_data(r.rid, &up[r.sent..r.sent + n]) {
                    out.push(f);
                }
                r.sent += n;
                r.send_credit -= n as u64;
            }
            if r.sent == up.len() {
                if let Ok(f) = ec_proto::encode_json(&msg::end(r.rid)) {
                    out.push(f);
                }
                r.ended = true;
            }
        }
        for f in out {
            self.raw(&f);
        }
    }

    /// Handle one frame (or none within `timeout`); false once the node's stream ended.
    fn pump(&mut self, timeout: Duration) -> bool {
        self.feed_uploads();
        if self.ended {
            std::thread::sleep(timeout);
            return false;
        }
        let f = match self.rx.recv_timeout(timeout) {
            Ok(Some(f)) => f,
            Ok(None) | Err(RecvTimeoutError::Disconnected) => {
                self.ended = true;
                return false;
            }
            Err(RecvTimeoutError::Timeout) => return true,
        };
        match f {
            Frame::Data { stream, data } => {
                let mut to_send: Vec<Obj> = Vec::new();
                if let Some(r) = self.active.get_mut(&i64::from(stream)) {
                    let n = data.len() as u64;
                    r.got += n;
                    r.hasher.update(&data);
                    if i64::try_from(n).unwrap_or(i64::MAX) > r.recv_window {
                        r.violation = true;
                    }
                    r.recv_window -= i64::try_from(n).unwrap_or(0);
                    r.unacked += n;
                    if let Some(ca) = r.cancel_after
                        && !r.cancel_sent
                        && r.got >= ca
                    {
                        r.cancel_sent = true;
                        to_send.push(msg::cancel(r.rid));
                    }
                    if r.grant && r.unacked >= CREDIT_BATCH && r.res.is_none() && !r.cancel_sent {
                        to_send.push(msg::credit(r.rid, r.unacked));
                        r.recv_window += i64::try_from(r.unacked).unwrap_or(0);
                        r.unacked = 0;
                    }
                }
                for m in to_send {
                    self.send(&m);
                }
            }
            Frame::Json(m) => self.on_json(m),
        }
        true
    }

    fn on_json(&mut self, m: Obj) {
        let t = m
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let rid = m.get("id").and_then(|v| {
            if v.is_i64() || v.is_u64() {
                v.as_i64()
            } else {
                None
            }
        });
        if ["res", "event", "credit"].contains(&t.as_str())
            && let Some(rid) = rid
            && let Some(r) = self.active.get_mut(&rid)
        {
            let mut cancel = None;
            match t.as_str() {
                "res" => r.res = Some(m),
                "event" => {
                    r.events
                        .push(m.get("event").cloned().unwrap_or(Value::Null));
                    if let Some(n) = r.cancel_after_events
                        && !r.cancel_sent
                        && r.events.len() >= n
                    {
                        r.cancel_sent = true;
                        cancel = Some(r.rid);
                    }
                }
                _ => {
                    let b = m.get("bytes").and_then(Value::as_u64).unwrap_or(0);
                    r.credit_from_node += b;
                    r.send_credit += b;
                }
            }
            if let Some(rid) = cancel {
                self.send(&msg::cancel(rid));
            }
            return;
        }
        if t == "pong" {
            if let Some(rid) = rid {
                self.pongs.insert(rid, m);
            }
            return;
        }
        if t == "hello" && self.hello.is_none() {
            self.hello = Some(m);
            return;
        }
        if !t.is_empty() {
            *self.counts.entry(t.clone()).or_default() += 1;
            self.first.entry(t.clone()).or_insert_with(|| m.clone());
            self.last.insert(t, m);
        }
    }

    fn wait(&mut self, pred: &dyn Fn(&Self) -> bool, timeout: f64) -> bool {
        let deadline = Instant::now() + Duration::from_secs_f64(timeout);
        while !pred(self) {
            if Instant::now() > deadline {
                return false;
            }
            if !self.pump(Duration::from_millis(50)) {
                return pred(self);
            }
        }
        true
    }

    fn try_wait(&mut self) -> Option<ExitStatus> {
        self.child.try_wait().ok().flatten()
    }
}

fn record_req(r: &Req, timed_out: bool) -> Obj {
    let mut rec = Obj::new();
    rec.insert(
        "res".into(),
        r.res.clone().map_or(Value::Null, Value::Object),
    );
    rec.insert("events".into(), Value::Array(r.events.clone()));
    rec.insert("credit_from_node".into(), json!(r.credit_from_node));
    rec.insert("credit_violation".into(), json!(r.violation));
    if timed_out {
        rec.insert("timeout".into(), json!(true));
    }
    let data = if let Some(ca) = r.cancel_after {
        json!({"bytes_ge_cancel": r.got >= ca})
    } else if let Some(n) = r.cancel_after_events {
        rec.insert(
            "events".into(),
            Value::Array(r.events.iter().take(n).cloned().collect()),
        );
        Value::Null
    } else if r.got > 0 {
        json!({"bytes": r.got, "sha256": hex::encode(r.hasher.clone().finalize())})
    } else {
        Value::Null
    };
    rec.insert("data".into(), data);
    rec
}

fn num(step: &Value, k: &str) -> Option<f64> {
    step.get(k).and_then(Value::as_f64)
}

fn run(node: &[String], scenario: &Value) -> Result<Value, String> {
    let mut d = Driver::spawn(node).map_err(|e| format!("cannot start the node: {e}"))?;
    let mut out = Obj::new();
    out.insert("scenario".into(), scenario["name"].clone());
    d.wait(&|d| d.hello.is_some(), 25.0);
    out.insert(
        "skipped".into(),
        d.skipped
            .lock()
            .ok()
            .and_then(|g| *g)
            .map_or(Value::Null, |n| json!(n)),
    );
    out.insert(
        "hello".into(),
        d.hello.clone().map_or(Value::Null, Value::Object),
    );
    d.wait(
        &|d| {
            ["sample", "disks", "inventory"]
                .iter()
                .all(|k| d.first.contains_key(*k))
        },
        15.0,
    );
    let mut initial = Obj::new();
    for k in ["sample", "disks", "inventory"] {
        initial.insert(
            k.into(),
            d.first.get(k).cloned().map_or(Value::Null, Value::Object),
        );
    }
    out.insert("initial".into(), Value::Object(initial));
    let mut steps = Vec::new();
    let mut next_id: i64 = 1;
    let empty = Vec::new();
    for step in scenario["steps"].as_array().unwrap_or(&empty) {
        let kind = step["do"].as_str().unwrap_or("").to_string();
        let mut rec = Obj::new();
        rec.insert("do".into(), json!(kind));
        match kind.as_str() {
            "ping" => {
                let id = step["id"].as_i64().unwrap_or(0);
                d.send(&msg::ping(id));
                let ok = d.wait(&|d| d.pongs.contains_key(&id), 10.0);
                rec.insert(
                    "pong".into(),
                    if ok {
                        Value::Object(d.pongs[&id].clone())
                    } else {
                        Value::Null
                    },
                );
            }
            "req" => {
                let rid = step.get("id").and_then(Value::as_i64).unwrap_or(next_id);
                next_id = next_id.max(rid) + 1;
                let upload = step
                    .get("upload")
                    .and_then(Value::as_u64)
                    .map(|n| pattern(usize::try_from(n).unwrap_or(0)));
                let mut m = Obj::new();
                m.insert("type".into(), json!("req"));
                m.insert("id".into(), json!(rid));
                if step.get("op_missing") != Some(&json!(true)) {
                    m.insert("op".into(), step["op"].clone());
                }
                if step.get("args_missing") != Some(&json!(true)) {
                    let mut args = step.get("args").cloned().unwrap_or_else(|| json!({}));
                    if args.get("sha256") == Some(&json!("$upload_sha256"))
                        && let Some(up) = &upload
                        && let Some(o) = args.as_object_mut()
                    {
                        o.insert("sha256".into(), json!(hex::encode(Sha256::digest(up))));
                    }
                    m.insert("args".into(), args);
                }
                let grant = step.get("grant").and_then(Value::as_bool).unwrap_or(true);
                let ca = step.get("cancel_after_bytes").and_then(Value::as_u64);
                let cae = step
                    .get("cancel_after_events")
                    .and_then(Value::as_u64)
                    .map(|n| usize::try_from(n).unwrap_or(0));
                d.active.insert(rid, Req::new(rid, upload, grant, ca, cae));
                d.send(&m);
                let done = d.wait(
                    &|d| d.active.get(&rid).is_some_and(|r| r.res.is_some()),
                    num(step, "timeout").unwrap_or(30.0),
                );
                if let Some(r) = d.active.remove(&rid) {
                    rec.insert("id".into(), json!(rid));
                    rec.extend(record_req(&r, !done));
                }
            }
            "send_reqs" => {
                let from = step["from"].as_i64().unwrap_or(0);
                let count = step["count"].as_i64().unwrap_or(0);
                let op = step["op"].as_str().unwrap_or("").to_string();
                for i in 0..count {
                    let rid = from + i;
                    let text = step["args"].to_string().replace("{i}", &i.to_string());
                    let args: Obj = serde_json::from_str(&text).unwrap_or_default();
                    d.active.insert(rid, Req::new(rid, None, true, None, None));
                    d.send(&msg::req(rid, &op, Some(args)));
                }
                rec.insert("sent".into(), json!(count));
            }
            "send_req" => {
                let args = step
                    .get("args")
                    .and_then(Value::as_object)
                    .cloned()
                    .unwrap_or_default();
                d.send(&msg::req(
                    step["id"].as_i64().unwrap_or(0),
                    step["op"].as_str().unwrap_or(""),
                    Some(args),
                ));
            }
            "cancel_all" => {
                let from = step["from"].as_i64().unwrap_or(0);
                let ids: Vec<i64> = (0..step["count"].as_i64().unwrap_or(0))
                    .map(|i| from + i)
                    .collect();
                for rid in &ids {
                    d.send(&msg::cancel(*rid));
                }
                let ids2 = ids.clone();
                d.wait(
                    &move |d| {
                        ids2.iter()
                            .all(|i| d.active.get(i).is_some_and(|r| r.res.is_some()))
                    },
                    30.0,
                );
                let res: Vec<Value> = ids
                    .iter()
                    .map(|i| {
                        d.active
                            .remove(i)
                            .and_then(|r| r.res)
                            .map_or(Value::Null, Value::Object)
                    })
                    .collect();
                rec.insert("res".into(), Value::Array(res));
            }
            "send" => d.send_value(&step["msg"]),
            "raw" => {
                let bytes = hex::decode(step["hex"].as_str().unwrap_or("")).unwrap_or_default();
                d.raw(&bytes);
            }
            "sleep" => {
                let end = Instant::now() + Duration::from_secs_f64(num(step, "s").unwrap_or(0.0));
                while Instant::now() < end {
                    d.pump(Duration::from_millis(50));
                }
            }
            "await_push" => {
                let t = step["type"].as_str().unwrap_or("").to_string();
                let n = usize::try_from(step.get("count").and_then(Value::as_u64).unwrap_or(1))
                    .unwrap_or(1);
                let t2 = t.clone();
                let ok = d.wait(
                    &move |d| d.counts.get(&t2).copied().unwrap_or(0) >= n,
                    num(step, "timeout").unwrap_or(10.0),
                );
                rec.insert(
                    "msg".into(),
                    if ok {
                        d.last.get(&t).cloned().map_or(Value::Null, Value::Object)
                    } else {
                        Value::Null
                    },
                );
                rec.insert("timeout".into(), json!(!ok));
            }
            "close" | "wait_exit" => {
                if kind == "close" {
                    d.stdin = None; // EOF
                }
                let deadline =
                    Instant::now() + Duration::from_secs_f64(num(step, "timeout").unwrap_or(20.0));
                let mut status = d.try_wait();
                while status.is_none() && Instant::now() < deadline {
                    d.pump(Duration::from_millis(50));
                    status = d.try_wait();
                }
                if status.is_none() {
                    let _ = d.child.kill();
                    let _ = d.child.wait();
                }
                std::thread::sleep(Duration::from_millis(200)); // let the stderr reader catch the last words
                let text = d
                    .err
                    .lock()
                    .map(|g| String::from_utf8_lossy(&g).into_owned())
                    .unwrap_or_default();
                rec.insert(
                    "exit".into(),
                    status
                        .and_then(|s| s.code())
                        .map_or(json!("timeout"), |c| json!(c)),
                );
                let mut contains = Obj::new();
                for s in step
                    .get("stderr_contains")
                    .and_then(Value::as_array)
                    .unwrap_or(&empty)
                {
                    let s = s.as_str().unwrap_or("");
                    contains.insert(s.into(), json!(text.contains(s)));
                }
                rec.insert("stderr_contains".into(), Value::Object(contains));
            }
            other => return Err(format!("unknown step {other:?}")),
        }
        steps.push(Value::Object(rec));
    }
    out.insert("steps".into(), Value::Array(steps));
    if d.try_wait().is_none() {
        let _ = d.child.kill();
        let _ = d.child.wait();
    }
    Ok(Value::Object(out))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut node = None;
    let mut scenario = None;
    let mut i = 0;
    while i + 1 < args.len() {
        match args[i].as_str() {
            "--node" => node = Some(args[i + 1].clone()),
            "--scenario" => scenario = Some(args[i + 1].clone()),
            _ => {}
        }
        i += 2;
    }
    let (Some(node), Some(scenario)) = (node, scenario) else {
        eprintln!("usage: ecproto-drive --node '<argv JSON>' --scenario <file.json>");
        std::process::exit(2);
    };
    let node: Vec<String> = serde_json::from_str(&node).unwrap_or_default();
    let scenario: Value = std::fs::read_to_string(&scenario)
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or(Value::Null);
    if node.is_empty() || scenario.is_null() {
        eprintln!("ecproto-drive: bad --node or --scenario");
        std::process::exit(2);
    }
    match run(&node, &scenario) {
        Ok(t) => println!("{}", serde_json::to_string_pretty(&t).unwrap_or_default()),
        Err(e) => {
            eprintln!("ecproto-drive: {e}");
            std::process::exit(1);
        }
    }
}
