//! Port of node/src/ecnode/server.py: the stdio protocol server (docs/FLEET-PROTOCOL.md section 2):
//! MAGIC + hello, then frames both ways.
//!
//! Threads: one reader (frames from the hub), one writer (every frame to the hub goes through its queue, so frames
//! never interleave), a watchdog (a write blocked for `write_stall_s` ends the process; `hub_silence_s` without any
//! hub frame ends the session), the push loops (sample, disks, inventory, metrics), the model scanner, and one
//! thread per running request (at most `max_inflight`; more get `busy`).
//!
//! Flow control: every stream starts with WINDOW bytes of credit each way. Downloads (node -> hub) never put more
//! DATA on a stream than the hub granted; uploads (hub -> node) are buffered only up to the credit this node granted,
//! and credit is handed back as the handler consumes, so a slow disk slows only its own upload. Bulk JSON (events,
//! inventory, metrics, disks) is rate-limited well under the hub's flood limit; control frames (res, pong, credit,
//! sample) are never held back.

use std::collections::{HashMap, VecDeque};
use std::io::{BufReader, Read, Write};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicI64, Ordering};
use std::sync::{Arc, Condvar, Mutex, Weak};
use std::thread;
use std::time::{Duration, Instant};

use ec_proto::clean::clean_obj;
use ec_proto::pyfmt::{fmt_g, repr_str, str_value, take_chars};
use ec_proto::{Frame, Obj, ProtoError, ReadError, msg, specs};
use serde_json::{Value, json};

use crate::agent::Agent;
use crate::errors::{Call, Cancelled, HandlerError, OpError, Stream};
use crate::util::panics::{contain, report};
use crate::util::scrub;
use crate::util::sync::{lock, wait, wait_timeout, wait_timeout_while};

pub const DATA_QUEUE_MAX: usize = 4 * 1024 * 1024;
pub const JSON_QUEUE_MAX: usize = 64 * 1024 * 1024;
pub const WRITE_BATCH: usize = 1024 * 1024;
pub const GRANT_STEP: i64 = 1024 * 1024;
pub const UPLOAD_IDLE_S: f64 = 300.0;
pub const CREDIT_WAIT_S: f64 = 300.0;
pub const BULK_WINDOW_S: f64 = 10.0;
pub const BULK_BYTES: usize = 8 * 1024 * 1024;
pub const BULK_FRAMES: usize = 20_000;
pub const LINE_OPS: &[&str] = &["docker.logs_follow", "docker.pull", "scripts.run"];
pub const FIRST_INVENTORY_WAIT_S: f64 = 5.0;
const POLL: Duration = Duration::from_millis(250);
const WINDOW: i64 = ec_proto::WINDOW as i64;

/// Every period of the server (server.py `Timing`); tests shrink them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Timing {
    pub sample_s: f64,
    pub disks_s: f64,
    pub inventory_s: f64,
    pub inventory_poll_s: f64,
    pub hub_silence_s: f64,
    pub write_stall_s: f64,
    pub watchdog_s: f64,
}

impl Default for Timing {
    fn default() -> Self {
        Self {
            sample_s: ec_proto::SAMPLE_S,
            disks_s: ec_proto::DISKS_S,
            inventory_s: ec_proto::INVENTORY_S,
            inventory_poll_s: 10.0,
            hub_silence_s: ec_proto::HUB_SILENCE_S,
            write_stall_s: 10.0,
            watchdog_s: 1.0,
        }
    }
}

/// The op handler the server calls (ops::dispatch in production; tests inject fakes).
pub type Handler = Arc<dyn Fn(&Agent, &Call<'_>) -> Result<Obj, HandlerError> + Send + Sync>;

pub struct ServerOptions {
    pub timing: Timing,
    /// What a blocked writer does (Python: `os._exit(3)`).
    pub on_stall: Arc<dyn Fn() + Send + Sync>,
    /// Start the agent's model scanner.
    pub scanner: bool,
    pub handler: Handler,
    pub max_inflight: usize,
}

impl Default for ServerOptions {
    fn default() -> Self {
        Self {
            timing: Timing::default(),
            on_stall: Arc::new(|| std::process::exit(3)),
            scanner: true,
            handler: Arc::new(crate::ops::dispatch),
            max_inflight: ec_proto::MAX_INFLIGHT,
        }
    }
}

#[allow(clippy::print_stderr)]
pub fn say(text: &str) {
    eprintln!("ecnode: {text}");
}

fn secs(s: f64) -> Duration {
    Duration::from_secs_f64(s.max(0.0))
}

// ── a settable event ────────────────────────────────────────────────────────────────────────────────────
#[derive(Default)]
struct Flag {
    set: Mutex<bool>,
    cv: Condvar,
}

impl Flag {
    fn set(&self) {
        *lock(&self.set) = true;
        self.cv.notify_all();
    }

    fn is_set(&self) -> bool {
        *lock(&self.set)
    }

    /// `threading.Event.wait(timeout)`: true when set.
    fn wait(&self, d: Duration) -> bool {
        *wait_timeout_while(&self.cv, lock(&self.set), d, |s| !*s)
    }

    fn wait_forever(&self) {
        let mut g = lock(&self.set);
        while !*g {
            g = wait(&self.cv, g);
        }
    }
}

// ── the writer ───────────────────────────────────────────────────────────────────────────────────────────
struct WState {
    q: VecDeque<(Vec<u8>, bool)>,
    data: usize,
    json: usize,
    closing: bool,
    closed: bool,
    win_start: Instant,
    win_bytes: usize,
    win_frames: usize,
}

/// Single serialized writer. put() never interleaves frames; DATA and bulk JSON wait for room.
struct Writer {
    st: Mutex<WState>,
    cv: Condvar,
    writing_since: Mutex<Option<Instant>>,
    bulk_bytes: usize,
    bulk_frames: usize,
}

impl Writer {
    fn new() -> Self {
        Self {
            st: Mutex::new(WState {
                q: VecDeque::new(),
                data: 0,
                json: 0,
                closing: false,
                closed: false,
                win_start: Instant::now(),
                win_bytes: 0,
                win_frames: 0,
            }),
            cv: Condvar::new(),
            writing_since: Mutex::new(None),
            bulk_bytes: BULK_BYTES,
            bulk_frames: BULK_FRAMES,
        }
    }

    fn put(
        &self,
        frame: Vec<u8>,
        data: bool,
        bulk: bool,
        abort: Option<&dyn Fn() -> bool>,
    ) -> bool {
        let mut st = lock(&self.st);
        while !st.closed && !st.closing && self.must_wait(&mut st, frame.len(), data, bulk) {
            if let Some(a) = abort
                && a()
            {
                return false;
            }
            st = wait_timeout(&self.cv, st, POLL);
        }
        if st.closed || st.closing {
            return false;
        }
        if bulk {
            st.win_bytes += frame.len();
            st.win_frames += 1;
        }
        if data {
            st.data += frame.len();
        } else {
            st.json += frame.len();
        }
        st.q.push_back((frame, data));
        self.cv.notify_all();
        true
    }

    fn must_wait(&self, st: &mut WState, n: usize, data: bool, bulk: bool) -> bool {
        if data {
            return st.data > 0 && st.data + n > DATA_QUEUE_MAX;
        }
        if st.json > 0 && st.json + n > JSON_QUEUE_MAX {
            return true;
        }
        bulk && !self.bulk_room(st, n)
    }

    fn bulk_room(&self, st: &mut WState, n: usize) -> bool {
        let now = Instant::now();
        if now.duration_since(st.win_start) >= secs(BULK_WINDOW_S) {
            st.win_start = now;
            st.win_bytes = 0;
            st.win_frames = 0;
        }
        st.win_frames < self.bulk_frames
            && (st.win_bytes + n <= self.bulk_bytes || st.win_bytes == 0)
    }

    fn run(&self, out: &mut dyn Write, on_closed: &dyn Fn()) {
        loop {
            let batch = {
                let mut st = lock(&self.st);
                while st.q.is_empty() && !st.closed && !st.closing {
                    st = wait_timeout(&self.cv, st, Duration::from_millis(500));
                }
                if st.q.is_empty() {
                    st.closed = true;
                    self.cv.notify_all();
                    return;
                }
                let mut batch = Vec::new();
                let mut size = 0;
                while size < WRITE_BATCH {
                    let Some((frame, is_data)) = st.q.pop_front() else {
                        break;
                    };
                    size += frame.len();
                    if is_data {
                        st.data -= frame.len();
                    } else {
                        st.json -= frame.len();
                    }
                    batch.extend_from_slice(&frame);
                }
                self.cv.notify_all();
                batch
            };
            *lock(&self.writing_since) = Some(Instant::now());
            let ok = out.write_all(&batch).and_then(|()| out.flush()).is_ok();
            *lock(&self.writing_since) = None;
            if !ok {
                {
                    let mut st = lock(&self.st);
                    st.closed = true;
                    st.q.clear();
                    st.data = 0;
                    st.json = 0;
                    self.cv.notify_all();
                }
                on_closed();
                return;
            }
        }
    }

    /// Stop accepting frames, write what is queued (for at most `timeout`), stop.
    fn close(&self, timeout: Duration) {
        let deadline = Instant::now() + timeout;
        let mut st = lock(&self.st);
        st.closing = true;
        self.cv.notify_all();
        while !st.q.is_empty() && !st.closed && Instant::now() < deadline {
            st = wait_timeout(&self.cv, st, Duration::from_millis(100));
        }
        st.closed = true;
        self.cv.notify_all();
    }

    fn writing_since(&self) -> Option<Instant> {
        *lock(&self.writing_since)
    }
}

// ── one request ──────────────────────────────────────────────────────────────────────────────────────────
struct RState {
    hooks: Vec<Box<dyn FnOnce() + Send>>,
    send_credit: i64,
    chunks: VecDeque<Vec<u8>>,
    received: i64,
    granted: i64,
    consumed: i64,
    ended: bool,
    violation: Option<String>,
}

/// A request in flight: its cancel state, both credit windows and its upload buffer (errors.rs `Stream`).
struct Request {
    shared: Weak<Shared>,
    rid: i64,
    op: String,
    upload: bool,
    cancelled_flag: AtomicBool,
    st: Mutex<RState>,
    cv: Condvar,
}

impl Request {
    fn new(shared: &Arc<Shared>, rid: i64, op: &str, upload: bool) -> Self {
        Self {
            shared: Arc::downgrade(shared),
            rid,
            op: op.to_string(),
            upload,
            cancelled_flag: AtomicBool::new(false),
            st: Mutex::new(RState {
                hooks: Vec::new(),
                send_credit: WINDOW,
                chunks: VecDeque::new(),
                received: 0,
                granted: WINDOW,
                consumed: 0,
                ended: false,
                violation: None,
            }),
            cv: Condvar::new(),
        }
    }

    fn server_done(&self) -> bool {
        self.shared.upgrade().is_none_or(|s| s.done.is_set())
    }

    /// The reader's side: cancel, and run the cancel hooks on their own thread (they may block).
    fn cancel(&self) -> Option<Arc<Flag>> {
        let hooks = {
            let mut st = lock(&self.st);
            if self.cancelled_flag.swap(true, Ordering::SeqCst) {
                return None;
            }
            let hooks = std::mem::take(&mut st.hooks);
            self.cv.notify_all();
            hooks
        };
        if hooks.is_empty() {
            return None;
        }
        let done = Arc::new(Flag::default());
        let d2 = Arc::clone(&done);
        let spawned = thread::Builder::new()
            .name("ecnode-cancel".into())
            .spawn(move || {
                for f in hooks {
                    if let Err(p) = catch_unwind(AssertUnwindSafe(f)) {
                        report("cancel hook failed", p.as_ref());
                    }
                }
                d2.set();
            });
        if spawned.is_err() {
            done.set();
        }
        Some(done)
    }

    fn add_credit(&self, n: i64) {
        let mut st = lock(&self.st);
        st.send_credit = (st.send_credit + n).min(1 << 40);
        self.cv.notify_all();
    }

    fn add_data(&self, data: Vec<u8>) {
        let mut st = lock(&self.st);
        if !self.upload || st.ended {
            return;
        }
        st.received += i64::try_from(data.len()).unwrap_or(i64::MAX);
        if st.received > st.granted {
            st.violation =
                Some("the hub sent more upload data than the credit this node granted".into());
            st.chunks.clear();
        } else if st.violation.is_none() && !data.is_empty() {
            st.chunks.push_back(data);
        }
        self.cv.notify_all();
    }

    fn end(&self) {
        lock(&self.st).ended = true;
        self.cv.notify_all();
    }

    fn send(&self, m: &Obj, bulk: bool, abort: Option<&dyn Fn() -> bool>) -> bool {
        self.shared
            .upgrade()
            .is_some_and(|s| s.send(m, bulk, abort))
    }
}

impl Stream for Request {
    fn cancelled(&self) -> bool {
        self.cancelled_flag.load(Ordering::SeqCst) || self.server_done()
    }

    fn check(&self) -> Result<(), Cancelled> {
        if self.cancelled() {
            Err(Cancelled)
        } else {
            Ok(())
        }
    }

    fn event(&self, ev: Obj) {
        let cleaned = clean_obj(&Value::Object(ev), specs::events(&self.op));
        let abort = || self.cancelled();
        self.send(
            &msg::event(self.rid, cleaned),
            LINE_OPS.contains(&self.op.as_str()),
            Some(&abort),
        );
    }

    fn send_data(&self, data: &[u8]) -> Result<(), HandlerError> {
        let mut view = data;
        while !view.is_empty() {
            let n = {
                let mut st = lock(&self.st);
                let deadline = Instant::now() + secs(CREDIT_WAIT_S);
                while st.send_credit <= 0 && !self.cancelled() {
                    if Instant::now() > deadline {
                        return Err(OpError::new(
                            ec_proto::TIMEOUT,
                            "the hub stopped granting credit for this stream",
                        )
                        .into());
                    }
                    st = wait_timeout(&self.cv, st, POLL);
                }
                self.check()?;
                let credit = usize::try_from(st.send_credit).unwrap_or(0);
                let n = view.len().min(credit).min(ec_proto::MAX_DATA);
                st.send_credit -= i64::try_from(n).unwrap_or(0);
                n
            };
            let frame = ec_proto::encode_data(self.rid, &view[..n])
                .map_err(|e| HandlerError::internal(format!("ProtoError: {e}")))?;
            let Some(shared) = self.shared.upgrade() else {
                return Err(HandlerError::Cancelled);
            };
            let abort = || self.cancelled();
            if !shared.writer.put(frame, true, false, Some(&abort)) {
                return Err(HandlerError::Cancelled);
            }
            view = &view[n..];
        }
        Ok(())
    }

    fn read_upload(&self) -> Result<Vec<u8>, HandlerError> {
        let (chunk, grant) = {
            let mut st = lock(&self.st);
            let deadline = Instant::now() + secs(UPLOAD_IDLE_S);
            while st.chunks.is_empty() && !st.ended && !self.cancelled() && st.violation.is_none() {
                if Instant::now() > deadline {
                    return Err(OpError::new(
                        ec_proto::TIMEOUT,
                        format!("no upload data from the hub for {} s", fmt_g(UPLOAD_IDLE_S)),
                    )
                    .into());
                }
                st = wait_timeout(&self.cv, st, POLL);
            }
            self.check()?;
            if let Some(v) = &st.violation {
                return Err(OpError::new(ec_proto::BAD_ARGS, v.clone()).into());
            }
            let Some(chunk) = st.chunks.pop_front() else {
                return Ok(Vec::new());
            };
            st.consumed += i64::try_from(chunk.len()).unwrap_or(0);
            let mut grant = st.consumed + WINDOW - st.granted;
            if grant >= GRANT_STEP && !st.ended {
                st.granted += grant;
            } else {
                grant = 0;
            }
            (chunk, grant)
        };
        if grant > 0 {
            self.send(
                &msg::credit(self.rid, u64::try_from(grant).unwrap_or(0)),
                false,
                None,
            );
        }
        Ok(chunk)
    }

    fn drain_upload(&self) {
        while let Ok(chunk) = self.read_upload() {
            if chunk.is_empty() {
                break;
            }
        }
    }

    fn on_cancel(&self, f: Box<dyn FnOnce() + Send>) {
        {
            let mut st = lock(&self.st);
            if !self.cancelled() {
                st.hooks.push(f);
                return;
            }
        }
        f();
    }
}

// ── the server ───────────────────────────────────────────────────────────────────────────────────────────
struct Shared {
    agent: Arc<Agent>,
    opts: ServerOptions,
    writer: Writer,
    done: Flag,
    exit_code: AtomicI32,
    requests: Mutex<HashMap<i64, Arc<Request>>>,
    last_rx: Mutex<Instant>,
    seq: AtomicI64,
    inventory: Mutex<(Option<Value>, Option<Instant>)>,
}

impl Shared {
    fn send(&self, m: &Obj, bulk: bool, abort: Option<&dyn Fn() -> bool>) -> bool {
        let frame = match ec_proto::encode_json(m) {
            Ok(f) => f,
            Err(e) => {
                let t = m.get("type").map(str_value).unwrap_or_default();
                say(&format!("dropped an unsendable {t} message: {e}"));
                if m.get("type").and_then(Value::as_str) == Some("res") {
                    let rid = m.get("id").and_then(Value::as_i64).unwrap_or(0);
                    match ec_proto::encode_json(&msg::res_err(
                        rid,
                        ec_proto::INTERNAL,
                        "the result was too large or malformed to send",
                    )) {
                        Ok(f) => f,
                        Err(_) => return false,
                    }
                } else {
                    return false;
                }
            }
        };
        self.writer.put(frame, false, bulk, abort)
    }

    /// A pushed message, rebuilt from its whitelist (`"type"` last, as Python's dict update leaves it).
    fn push(&self, m: &Obj, bulk: bool) -> bool {
        let Some(t) = m.get("type").and_then(Value::as_str) else {
            return false;
        };
        let Some(spec) = specs::push(t) else {
            return false;
        };
        let mut out = clean_obj(&Value::Object(m.clone()), spec);
        out.insert("type".into(), json!(t));
        self.send(&out, bulk, None)
    }

    fn log(&self, level: &str, text: &str) {
        say(text);
        let mut m = Obj::new();
        m.insert("type".into(), json!("log"));
        m.insert("level".into(), json!(level));
        m.insert("msg".into(), json!(take_chars(&scrub(text), 1000)));
        self.push(&m, false);
    }

    fn touch_rx(&self) {
        *lock(&self.last_rx) = Instant::now();
    }

    fn request(&self, rid: i64) -> Option<Arc<Request>> {
        lock(&self.requests).get(&rid).cloned()
    }
}

/// The node's protocol server over one input and one output stream.
pub struct Server {
    shared: Arc<Shared>,
}

impl Server {
    pub fn new(agent: Arc<Agent>, opts: ServerOptions) -> Self {
        Self {
            shared: Arc::new(Shared {
                agent,
                opts,
                writer: Writer::new(),
                done: Flag::default(),
                exit_code: AtomicI32::new(0),
                requests: Mutex::new(HashMap::new()),
                last_rx: Mutex::new(Instant::now()),
                seq: AtomicI64::new(0),
                inventory: Mutex::new((None, None)),
            }),
        }
    }

    pub fn exit_code(&self) -> i32 {
        self.shared.exit_code.load(Ordering::SeqCst)
    }

    /// Speak the protocol until stdin closes, the hub goes silent, the hub breaks the protocol or the output fails.
    /// Returns the process exit code (0, 1 when the machine cannot be described, 2 on a protocol error).
    pub fn run(&self, inp: Box<dyn Read + Send>, out: Box<dyn Write + Send>) -> i32 {
        let sh = &self.shared;
        sh.writer.put(ec_proto::MAGIC.to_vec(), false, false, None);
        {
            let s = Arc::clone(sh);
            let mut out = out;
            spawn("writer", move || {
                // a writer that died would leave the node mute: a panic ends the session instead
                if contain("writer", || s.writer.run(&mut *out, &|| s.done.set())).is_none() {
                    s.done.set();
                }
            });
        }
        let agent = Arc::clone(&sh.agent);
        let hello = catch_unwind(AssertUnwindSafe(|| agent.hello(false)));
        let Ok(hello) = hello else {
            let mut m = Obj::new();
            m.insert("type".into(), json!("fatal"));
            m.insert(
                "error".into(),
                json!("the node could not describe this machine: panic"),
            );
            sh.push(&m, false);
            sh.writer.close(Duration::from_secs(2));
            return 1;
        };
        sh.push(&hello, false);
        if sh.opts.scanner {
            let weak = Arc::downgrade(sh);
            sh.agent.start_scanner(Box::new(move |generation| {
                if let Some(s) = weak.upgrade() {
                    let mut m = Obj::new();
                    m.insert("type".into(), json!("models_changed"));
                    m.insert("generation".into(), json!(generation));
                    m.insert("t".into(), json!(crate::agent::now3()));
                    s.push(&m, false);
                }
            }));
        }
        sh.touch_rx();
        {
            let s = Arc::clone(sh);
            spawn("reader", move || reader(&s, inp));
        }
        for (name, f) in [
            ("watchdog", watchdog as fn(&Arc<Shared>)),
            ("sampler", sampler),
            ("disks", disks),
            ("inventory", inventory),
            ("metrics", metrics),
        ] {
            let s = Arc::clone(sh);
            spawn(name, move || f(&s));
        }
        sh.done.wait_forever();
        self.shutdown();
        self.exit_code()
    }

    fn shutdown(&self) {
        let sh = &self.shared;
        sh.done.set();
        let reqs: Vec<Arc<Request>> = lock(&sh.requests).values().cloned().collect();
        let hooks: Vec<Arc<Flag>> = reqs.iter().filter_map(|r| r.cancel()).collect();
        if sh.opts.scanner {
            sh.agent.stop_scanner();
        }
        // child processes (logs -f, pulls, scripts) are stopped, not orphaned
        let deadline = Instant::now() + Duration::from_secs(8);
        for h in hooks {
            h.wait(deadline.saturating_duration_since(Instant::now()));
        }
        sh.writer.close(Duration::from_secs(2));
    }
}

fn spawn(name: &str, f: impl FnOnce() + Send + 'static) {
    if thread::Builder::new()
        .name(format!("ecnode-{name}"))
        .spawn(f)
        .is_err()
    {
        say(&format!("could not start the {name} thread"));
    }
}

// ── reading ──────────────────────────────────────────────────────────────────────────────────────────────
fn reader(sh: &Arc<Shared>, inp: Box<dyn Read + Send>) {
    let mut inp = BufReader::with_capacity(ec_proto::MAX_DATA + 4096, inp);
    while !sh.done.is_set() {
        match ec_proto::read_frame_sync(&mut inp) {
            Ok(None) => break,
            Ok(Some(frame)) => {
                sh.touch_rx();
                // one frame that trips a bug costs that frame, never the reader (and with it the session)
                let _ = contain("reader", || match frame {
                    Frame::Data { stream, data } => {
                        if let Some(req) = sh.request(i64::from(stream)) {
                            req.add_data(data);
                        }
                    }
                    Frame::Json(m) => on_message(sh, &m),
                });
            }
            Err(ReadError::Proto(e)) => {
                say(&format!(
                    "protocol error from the hub, closing: {}",
                    proto_text(&e)
                ));
                sh.exit_code.store(2, Ordering::SeqCst);
                break;
            }
            Err(ReadError::Io(e)) => {
                say(&format!(
                    "reading from the hub failed, closing: {:?}",
                    e.kind()
                ));
                break;
            }
        }
    }
    sh.done.set();
}

fn proto_text(e: &ProtoError) -> String {
    match e {
        ProtoError::Invalid(s) | ProtoError::Truncated(s) => s.clone(),
    }
}

fn int_of(v: Option<&Value>) -> Option<i64> {
    v.and_then(|v| {
        if v.is_i64() || v.is_u64() {
            v.as_i64()
        } else {
            None
        }
    })
}

fn on_message(sh: &Arc<Shared>, m: &Obj) {
    let t = m.get("type").and_then(Value::as_str);
    if t == Some("req") {
        on_req(sh, m);
        return;
    }
    let Some(rid) = int_of(m.get("id")).filter(|r| ec_proto::valid_id_i64(*r)) else {
        return;
    };
    if t == Some("ping") {
        sh.send(&msg::pong(rid), false, None);
        return;
    }
    let Some(req) = sh.request(rid) else { return };
    match t {
        Some("cancel") => {
            req.cancel();
        }
        Some("credit") => {
            if let Some(n) = int_of(m.get("bytes")).filter(|n| *n > 0 && *n <= 1 << 40) {
                req.add_credit(n);
            }
        }
        Some("end") => req.end(),
        _ => {}
    }
}

fn on_req(sh: &Arc<Shared>, m: &Obj) {
    let Some(rid) = int_of(m.get("id")).filter(|r| ec_proto::valid_id_i64(*r)) else {
        return;
    };
    let op = match m.get("op") {
        Some(Value::String(op)) if specs::is_op(op) => op.clone(),
        other => {
            let shown = take_chars(&str_value(other.unwrap_or(&Value::Null)), 64);
            sh.send(
                &msg::res_err(
                    rid,
                    ec_proto::UNKNOWN_OP,
                    &format!(
                        "this node does not know the op {} (ecnode protocol {})",
                        repr_str(&shown),
                        ec_proto::PROTO
                    ),
                ),
                false,
                None,
            );
            return;
        }
    };
    let policy = sh.agent.policy();
    if !specs::is_monitor_op(&op) && !policy.manage() {
        let why = policy
            .error
            .as_ref()
            .map(|e| format!(" ({e})"))
            .unwrap_or_default();
        sh.send(
            &msg::res_err(
                rid,
                ec_proto::MONITOR_ONLY,
                &format!(
                    "this node's policy is in monitor mode{why}, so {op}, which would change the machine, is \
                     refused; an administrator can set \"mode\": \"manage\" in {}",
                    policy.where_()
                ),
            ),
            false,
            None,
        );
        return;
    }
    let raw_args = m.get("args").unwrap_or(&Value::Null);
    let Ok(args) = ec_proto::clean_args(&op, raw_args) else {
        return;
    };
    // a path, container, image or argv that cleaning would cut short is refused, never acted on cut (audit F5)
    if let Some(why) = ec_proto::overlong_args(&op, raw_args) {
        sh.send(&msg::res_err(rid, ec_proto::BAD_ARGS, &why), false, None);
        return;
    }
    let missing = ec_proto::missing_args(&op, &args);
    if !missing.is_empty() {
        sh.send(
            &msg::res_err(
                rid,
                ec_proto::BAD_ARGS,
                &format!("{op} needs {}", missing.join(", ")),
            ),
            false,
            None,
        );
        return;
    }
    let upload =
        op == "fs.write" || (op == "http.request" && int_of(args.get("body_len")).unwrap_or(0) > 0);
    let req = Arc::new(Request::new(sh, rid, &op, upload));
    let (dup, full) = {
        let mut reqs = lock(&sh.requests);
        let dup = reqs.contains_key(&rid);
        let full = !dup && reqs.len() >= sh.opts.max_inflight;
        if !dup && !full {
            reqs.insert(rid, Arc::clone(&req));
        }
        (dup, full)
    };
    if dup {
        sh.log(
            "warn",
            &format!("ignored request {rid}: that id is already in flight"),
        );
    } else if full {
        sh.send(
            &msg::res_err(
                rid,
                ec_proto::BUSY,
                &format!(
                    "this node is already running {} requests; try again shortly",
                    sh.opts.max_inflight
                ),
            ),
            false,
            None,
        );
    } else {
        let s = Arc::clone(sh);
        let started = thread::Builder::new()
            .name(format!("ecnode-req-{op}"))
            .spawn(move || handle(&s, &req, &args));
        if started.is_err() {
            lock(&sh.requests).remove(&rid);
            sh.send(
                &msg::res_err(
                    rid,
                    ec_proto::INTERNAL,
                    "the node could not start a thread for this request",
                ),
                false,
                None,
            );
        }
    }
}

fn handle(sh: &Arc<Shared>, req: &Arc<Request>, args: &Obj) {
    let call = Call {
        op: &req.op,
        args,
        stream: &**req,
    };
    let outcome = catch_unwind(AssertUnwindSafe(|| (sh.opts.handler)(&sh.agent, &call)));
    let op = req.op.as_str();
    let internal = |kind: &str, location: Option<&str>| {
        let loc = location.map(|l| format!(" at {l}")).unwrap_or_default();
        say(&format!("internal error in {op}: {kind}{loc}"));
        msg::res_err(
            req.rid,
            ec_proto::INTERNAL,
            &format!("the node hit an unexpected error in {op} ({kind}); see the node's stderr"),
        )
    };
    let mut res = match outcome {
        Ok(Ok(result)) => msg::res_ok(
            req.rid,
            Some(clean_obj(&Value::Object(result), specs::results(op))),
        ),
        Ok(Err(HandlerError::Cancelled)) => msg::res_err(req.rid, ec_proto::CANCELLED, "cancelled"),
        Ok(Err(HandlerError::Op(e))) => msg::res_err(req.rid, &e.code, &scrub(&e.message)),
        Ok(Err(HandlerError::Internal { kind, location })) => internal(&kind, location.as_deref()),
        Err(_) => internal("panic", None),
    };
    let ok = res.get("ok") == Some(&Value::Bool(true));
    let code = res
        .get("error")
        .and_then(|e| e.get("code"))
        .and_then(Value::as_str)
        .unwrap_or("");
    if req.cancelled() && !ok && code != ec_proto::CANCELLED {
        res = msg::res_err(req.rid, ec_proto::CANCELLED, "cancelled");
    }
    sh.send(&res, false, None);
    lock(&sh.requests).remove(&req.rid);
}

// ── background loops ─────────────────────────────────────────────────────────────────────────────────────
/// One round of `f` every `period()` until the session ends. A round that panics is reported on stderr (redacted,
/// `util::panics::report`, as Python's `_say(f"{what}: {type}: {scrub(str(e))[:200]}")`) and the loop goes on.
fn every(sh: &Arc<Shared>, period: &dyn Fn() -> f64, f: &dyn Fn(), what: &str, first: f64) {
    let mut nxt = Instant::now() + secs(first);
    while !sh.done.wait(nxt.saturating_duration_since(Instant::now())) {
        let started = Instant::now();
        let _ = contain(what, f);
        nxt = (started + secs(period())).max(Instant::now());
    }
}

fn watchdog(sh: &Arc<Shared>) {
    let t = sh.opts.timing;
    while !sh.done.wait(secs(t.watchdog_s)) {
        let now = Instant::now();
        let round = contain("watchdog", || {
            if let Some(since) = sh.writer.writing_since()
                && now.duration_since(since) > secs(t.write_stall_s)
            {
                say("a write to the hub has been blocked too long; exiting");
                (sh.opts.on_stall)();
                return true;
            }
            if now.duration_since(*lock(&sh.last_rx)) > secs(t.hub_silence_s) {
                say(&format!(
                    "nothing from the hub for {} s; exiting",
                    fmt_g(t.hub_silence_s)
                ));
                return true;
            }
            false
        });
        if round == Some(true) {
            sh.done.set();
            return;
        }
    }
}

fn sampler(sh: &Arc<Shared>) {
    // The first samples should already name the container of each GPU process: wait (briefly) for the first
    // inventory, which knows the containers.
    sh.agent.wait_first_inventory(secs(FIRST_INVENTORY_WAIT_S));
    let one = || {
        let seq = sh.seq.fetch_add(1, Ordering::SeqCst) + 1;
        stray_print_for_tests();
        panic_for_tests(seq);
        sh.push(&sh.agent.sample(seq), false);
    };
    every(sh, &|| sh.opts.timing.sample_s, &one, "sampler", 0.0);
}

/// Test seam (debug builds only): crates/ecnode/tests/test_cli.rs checks that a stray print inside the node lands on
/// stderr, never in the frame stream, as node/tests/test_cli.py does by patching Agent.sample.
#[allow(clippy::print_stdout)]
fn stray_print_for_tests() {
    if cfg!(debug_assertions) && std::env::var_os("ECNODE_TEST_STRAY_PRINT").is_some() {
        println!("a stray print from inside the node");
    }
}

/// Test seam (debug builds only): with ECNODE_TEST_PANIC set, the first sample panics with a message that holds the
/// variable's value as an API key. crates/ecnode/tests/test_cli.rs checks that the node keeps sampling and that
/// stderr shows the panic redacted (the panic hook and the loop's report), never the value.
#[allow(clippy::panic)]
fn panic_for_tests(seq: i64) {
    if cfg!(debug_assertions)
        && seq == 1
        && let Some(v) = std::env::var_os("ECNODE_TEST_PANIC")
    {
        panic!(
            "test panic in the sampler: vllm serve --api-key {}",
            v.to_string_lossy()
        );
    }
}

fn disks(sh: &Arc<Shared>) {
    let one = || {
        sh.push(&sh.agent.disks(), true);
    };
    every(sh, &|| sh.opts.timing.disks_s, &one, "disks", 0.0);
}

/// What counts as an inventory change: everything except timestamps and docker's "Up 5 minutes" text.
fn inventory_key(m: &Obj) -> Value {
    let strip = |k: &str, drop: &str| -> Value {
        Value::Array(
            m.get(k)
                .and_then(Value::as_array)
                .map(|a| {
                    a.iter()
                        .map(|c| match c {
                            Value::Object(o) => {
                                let mut o = o.clone();
                                o.remove(drop);
                                Value::Object(o)
                            }
                            other => other.clone(),
                        })
                        .collect()
                })
                .unwrap_or_default(),
        )
    };
    json!([
        strip("containers", "status"),
        m.get("endpoints").cloned().unwrap_or(Value::Null),
        strip("scripts", "checked_at")
    ])
}

fn inventory(sh: &Arc<Shared>) {
    let one = || {
        let inv = sh.agent.refresh_inventory();
        let key = inventory_key(&inv);
        let now = Instant::now();
        let due = {
            let st = lock(&sh.inventory);
            st.0.as_ref() != Some(&key)
                || st
                    .1
                    .is_none_or(|t| now.duration_since(t) >= secs(sh.opts.timing.inventory_s))
        };
        if due {
            let mut m = Obj::new();
            m.insert("type".into(), json!("inventory"));
            m.extend(inv);
            if sh.push(&m, true) {
                *lock(&sh.inventory) = (Some(key), Some(now));
            }
        }
    };
    every(
        sh,
        &|| sh.opts.timing.inventory_poll_s,
        &one,
        "inventory",
        0.0,
    );
}

fn metrics(sh: &Arc<Shared>) {
    let one = || {
        for m in sh.agent.scrape() {
            sh.push(&m, true);
        }
    };
    let period = || {
        sh.agent
            .metrics_interval()
            .filter(|v| *v != 0.0)
            .unwrap_or(1.0)
    };
    every(sh, &period, &one, "metrics", 1.0);
}
