//! Python: backend/src/engine_console/services/fleet.py (tests/fleet/test_fleet_service.py,
//! test_link.py, test_link_failures.py, test_link_streams.py).
//!
//! The hub's side of the node protocol: one `NodeLink` per machine, owned by the `FleetService`.
//!
//! A node is an `ecnode --stdio` process: a local subprocess for host "local", `ssh` with the hub's restricted key for
//! every other machine. The hub talks to it in length-prefixed frames (crate ec-proto, the locked wire contract) and
//! treats everything it says as untrusted: each pushed message is rebuilt from a whitelist (`sanitize_*`), a node that
//! floods, goes silent, breaks the framing or sends more data than the credit it was granted is dropped, and whatever
//! goes wrong the pending requests fail with `node_unavailable` instead of hanging.
//!
//! Everything runs on the tokio runtime; nothing here blocks it. Every wait has a timeout, no shell is involved, and
//! the ssh command line and the key path are never logged above debug level.
//!
//! The interface: `FleetService::new` and the `Fleet`/`FleetPort` trait impls; also public, for the CLI and tests:
//! `LinkTiming`, `NodeLink` and its stream `LinkStream`, `LinkLike`, `explain_exit`, `FleetService::with_links`.

use std::collections::{HashMap, VecDeque};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Mutex, MutexGuard, RwLock, Weak};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use bytes::Bytes;
use serde_json::{Value, json};
use tokio::io::{AsyncRead, AsyncReadExt as _, AsyncWriteExt as _, BufReader};
use tokio::sync::{Notify, mpsc, oneshot};
use tokio::task::JoinHandle;

use crate::config::Settings;
use crate::domain::errors::{AppResult, NodeError, Obj, Problem, conflict, not_found};
use crate::domain::fleet::{
    CountsProvider, Fleet, FleetPort, HostConfig, HostDetail, HostSummary, ImageInfo,
    InstanceResolver, LOCAL, LinkStatus, NodeClient, NodeStream, StreamItem,
};
use crate::services::common::{EventBus, py_splitlines, scrub};
use crate::services::fleet_status::{self, HostData};
use crate::services::hosts::{
    HostsError, default_hosts, load_hosts, local_argv, local_env, py_isprintable, ssh_argv,
    ssh_env, strerror,
};
use crate::sync::{Recover as _, RecoverRw as _};

const LOG: &str = "engine_console.fleet";
/// buffer of the node's stdout: a whole DATA frame fits, so reading it never stalls
pub const READ_LIMIT: usize = ec_proto::MAX_DATA + 4096;
/// bytes of the node's stderr kept to explain why a link dropped
pub const STDERR_KEEP: usize = 2048;
/// events queued for a consumer that is not reading; more are dropped (`LinkTiming::max_stream_events`)
pub const MAX_STREAM_EVENTS: usize = 10_000;
/// credit goes back to the node in chunks this big, as the consumer reads
pub const CREDIT_BATCH: u64 = ec_proto::WINDOW / 4;
/// a node cannot make the hub's send window absurd
pub const MAX_CREDIT: i64 = 1 << 30;
/// a cancelled request the node never answered is forgotten after this
pub const ORPHAN_MAX_S: f64 = 600.0;
/// node `log` messages written to the hub's log per 10 s, the rest are dropped
pub const NODE_LOG_BURST: u32 = 20;
pub const IMAGES_TTL_S: f64 = 60.0;
const WINDOW: i64 = 8 * 1024 * 1024;

/// Every duration of the link. The defaults are the contract's; tests shrink them so a drop, a silence or a backoff
/// does not take half a minute. (`max_stream_events` is Python's module constant `MAX_STREAM_EVENTS`.)
#[derive(Debug, Clone, PartialEq)]
pub struct LinkTiming {
    pub hello_timeout_s: f64,
    pub ping_s: f64,
    pub silent_s: f64,
    /// no sample for this long: the host shows as `stale`
    pub stale_s: f64,
    pub backoff_min_s: f64,
    pub backoff_max_s: f64,
    /// up this long: the next failure starts the backoff over
    pub stable_s: f64,
    /// needs a person (login, host key, install, version): rarely retried
    pub fix_retry_s: f64,
    /// of JSON messages from one node ...
    pub flood_bytes: u64,
    /// ... in this window
    pub flood_window_s: f64,
    /// JSON messages (and empty DATA frames) in that window
    pub flood_frames: u64,
    /// after the node's pipes close: time to exit on its own
    pub exit_grace_s: f64,
    pub max_stream_events: usize,
}

impl Default for LinkTiming {
    fn default() -> Self {
        Self {
            hello_timeout_s: ec_proto::HELLO_TIMEOUT_S,
            ping_s: ec_proto::PING_S,
            silent_s: ec_proto::SILENT_S,
            stale_s: 3.0 * ec_proto::SAMPLE_S,
            backoff_min_s: 1.0,
            backoff_max_s: 60.0,
            stable_s: 60.0,
            fix_retry_s: 300.0,
            flood_bytes: 20 * 1024 * 1024,
            flood_window_s: 10.0,
            flood_frames: 50_000,
            exit_grace_s: 3.0,
            max_stream_events: MAX_STREAM_EVENTS,
        }
    }
}

/// Python's `round(x)` to an int (ties to even); the values here are small durations.
#[allow(clippy::cast_possible_truncation)]
fn round_i64(x: f64) -> i64 {
    x.round_ties_even() as i64
}

fn secs(s: f64) -> Duration {
    Duration::from_secs_f64(if s.is_finite() { s.max(0.0) } else { 0.0 })
}

fn g(x: f64) -> String {
    ec_proto::pyfmt::fmt_g(x)
}

fn wall() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64())
}

/// Untrusted text made safe for a message or a log line: secrets masked, control characters (newlines included)
/// turned into spaces, whitespace collapsed, length capped (characters).
pub fn text(v: &str, limit: usize) -> String {
    let s: String = scrub(v)
        .chars()
        .map(|c| if py_isprintable(c) { c } else { ' ' })
        .collect();
    s.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .chars()
        .take(limit)
        .collect()
}

/// `_text(v)` of a JSON value (`str(v)` first: None is "None").
fn text_value(v: Option<&Value>, limit: usize) -> String {
    text(
        &v.map_or_else(|| "None".into(), ec_proto::pyfmt::str_value),
        limit,
    )
}

/// A node error code. Well-formed codes pass (a newer node may add some); anything else is `internal`.
pub fn code(v: Option<&Value>) -> String {
    static RE: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
        regex::Regex::new(r"\A[a-z][a-z0-9_]{0,31}\z").unwrap_or_else(|_| unreachable!())
    });
    match v.and_then(Value::as_str) {
        Some(c) if RE.is_match(c) => c.to_string(),
        _ => ec_proto::INTERNAL.to_string(),
    }
}

/// SIGKILL the child's whole process group (it is its own group, so ssh and anything it spawned go too). A group
/// that is not the child's own (that could be the hub's) is never signalled.
fn kill_proc(pid: Option<i32>) {
    let Some(pid) = pid.filter(|p| *p > 0) else {
        return;
    };
    let p = nix::unistd::Pid::from_raw(pid);
    if nix::unistd::getpgid(Some(p)).ok() == Some(p) {
        let _ = nix::sys::signal::killpg(p, nix::sys::signal::Signal::SIGKILL);
    } else {
        let _ = nix::sys::signal::kill(p, nix::sys::signal::Signal::SIGKILL);
    }
}

/// Why a node process ended, in words a person can act on, and a forced retry delay (None: ordinary backoff).
/// Failures that only a person can fix (login refused, host key, node not installed) retry rarely. Anything in
/// `mask` (the hub key's path: ssh quotes it in its complaints) is replaced before the text goes anywhere.
pub fn explain_exit(
    host: &HostConfig,
    rc: Option<i32>,
    stderr: &str,
    remote_cmd: Option<&str>,
    fix_after_s: f64,
    mask: &[String],
) -> (String, Option<f64>) {
    let mut stderr = stderr.to_string();
    for secret in mask {
        if !secret.is_empty() {
            stderr = stderr.replace(secret.as_str(), "<hub key>");
        }
    }
    let t = crate::services::hosts::py_strip(&stderr);
    let last = py_splitlines(t)
        .last()
        .map(|x| text(x, 300))
        .unwrap_or_default();
    let with_last = |prefix: String| {
        if last.is_empty() {
            prefix
        } else {
            format!("{prefix}: {last}")
        }
    };
    if let Some(dest) = &host.ssh
        && rc == Some(255)
    {
        if t.contains("Permission denied") {
            return (
                format!(
                    "ssh refused the login to {dest}: the hub's key is not authorized there. Run \
                     `engine-console hosts init-key --from <this machine's address>` and add the line it prints to \
                     authorized_keys of the login user on {}",
                    host.label
                ),
                Some(fix_after_s),
            );
        }
        if t.contains("UNPROTECTED PRIVATE KEY") || t.contains("bad permissions") {
            return (
                "the hub's ssh key file is readable by other users, so ssh refuses it: chmod 600 the key".into(),
                Some(fix_after_s),
            );
        }
        if t.contains("Host key verification failed") || t.contains("IDENTIFICATION HAS CHANGED") {
            return (
                format!(
                    "the host key of {dest} is unknown or has changed. Check it, then connect once by hand to trust \
                     it: ssh {dest} true"
                ),
                Some(fix_after_s),
            );
        }
        if t.contains("Could not resolve hostname") {
            return (
                format!("cannot resolve {dest}: check the name in hosts.json"),
                None,
            );
        }
        if [
            "No route to host",
            "Connection timed out",
            "Connection refused",
            "Network is unreachable",
            "Operation timed out",
        ]
        .iter()
        .any(|x| t.contains(x))
        {
            return (format!("cannot reach {dest}: {last}"), None);
        }
        let why = if last.is_empty() {
            "exit 255".to_string()
        } else {
            last.clone()
        };
        return (format!("ssh to {dest} failed: {why}"), None);
    }
    if host.ssh.is_some() && rc == Some(127) {
        // the remote shell's "command not found"; matching text would also match a node's own errors
        let where_ = remote_cmd
            .filter(|c| !c.is_empty())
            .and_then(|c| crate::services::hosts::py_split(c).first().copied())
            .map(|c| format!(" (expected {c})"))
            .unwrap_or_default();
        return (
            format!(
                "the ecnode node is not installed on {}{where_}: run its install.sh there",
                host.label
            ),
            Some(fix_after_s),
        );
    }
    if host.ssh.is_none() && t.contains("No module named") {
        return (
            format!("the ecnode package is missing from the console's Python environment: {last}"),
            None,
        );
    }
    match rc {
        Some(rc) if rc > 0 => (with_last(format!("the node exited ({rc})")), None),
        Some(rc) if rc < 0 && rc != -9 => (
            with_last(format!("the node was stopped by signal {}", -rc)),
            None,
        ),
        _ => (
            if last.is_empty() {
                "the connection closed".into()
            } else {
                last
            },
            None,
        ),
    }
}

// ── frames ───────────────────────────────────────────────────────────────────────────────────────────────
enum FrameErr {
    /// the stream ended inside a frame, or reading failed: the far side died; its exit status says why
    Closed,
    Proto(String),
}

/// `ec_proto::decode_json` with Python's text for a non-finite number ("non-finite number NaN in message").
fn decode_json_py(body: &[u8]) -> Result<Obj, String> {
    match ec_proto::decode_json(body) {
        Ok(o) => Ok(o),
        Err(e) => {
            if let Ok(t) = std::str::from_utf8(body)
                && let Err(se) = serde_json::from_str::<Value>(t)
                && let Some((at, tok, _)) = crate::services::hosts::first_non_finite(t)
                && at <= crate::services::hosts::serde_offset(t, &se)
            {
                let tok: String = tok.chars().take(20).collect();
                return Err(format!("non-finite number {tok} in message"));
            }
            Err(e.to_string())
        }
    }
}

/// One frame and its size on the wire (the flood limit needs it); None at a clean end between frames.
async fn read_frame<R: AsyncRead + Unpin>(
    r: &mut R,
) -> Result<Option<(ec_proto::Frame, usize)>, FrameErr> {
    let mut hdr = [0u8; 5];
    let mut got = 0;
    while got < 5 {
        match r.read(&mut hdr[got..]).await {
            Ok(0) if got == 0 => return Ok(None),
            Ok(0) | Err(_) => return Err(FrameErr::Closed),
            Ok(n) => got += n,
        }
    }
    let (kind, n) = ec_proto::parse_header(&hdr).map_err(|e| FrameErr::Proto(e.to_string()))?;
    let mut body = vec![0u8; n];
    if r.read_exact(&mut body).await.is_err() {
        return Err(FrameErr::Closed);
    }
    let frame = if kind == ec_proto::KIND_JSON {
        ec_proto::Frame::Json(decode_json_py(&body).map_err(FrameErr::Proto)?)
    } else {
        ec_proto::decode_body(kind, &body).map_err(|e| FrameErr::Proto(e.to_string()))?
    };
    Ok(Some((frame, 5 + n)))
}

// ── requests ─────────────────────────────────────────────────────────────────────────────────────────────
struct ReqState {
    finished: bool,
    res: Option<Obj>,
    /// set when the link dropped before the node answered
    unavailable: Option<String>,
    /// None = end of stream marker
    queue: VecDeque<Option<StreamItem>>,
    queued_events: usize,
    cancel_sent: bool,
    cancel_at: f64,
    uploading: bool,
    ended: bool,
    /// DATA the hub may still send (the node's credit to us)
    send_credit: i64,
    /// DATA the node may still send (our credit to it)
    recv_window: i64,
    /// bytes the consumer read that the node has not been credited for
    unacked: i64,
}

/// One request in flight, seen from the hub: its answer, its stream of events and DATA, and both credit windows.
struct Req {
    rid: i64,
    op: String,
    streaming: bool,
    st: Mutex<ReqState>,
    changed: Notify,
}

impl Req {
    fn new(rid: i64, op: &str, streaming: bool) -> Self {
        Self {
            rid,
            op: op.to_string(),
            streaming,
            st: Mutex::new(ReqState {
                finished: false,
                res: None,
                unavailable: None,
                queue: VecDeque::new(),
                queued_events: 0,
                cancel_sent: false,
                cancel_at: 0.0,
                uploading: false,
                ended: false,
                send_credit: WINDOW,
                recv_window: WINDOW,
                unacked: 0,
            }),
            changed: Notify::new(),
        }
    }

    fn lock(&self) -> MutexGuard<'_, ReqState> {
        self.st.guard()
    }

    fn finished(&self) -> bool {
        self.lock().finished
    }

    fn finish(&self, res: Option<Obj>, unavailable: Option<String>) {
        {
            let mut st = self.lock();
            if st.finished {
                return;
            }
            st.res = res;
            st.unavailable = unavailable;
            st.finished = true;
            if self.streaming {
                st.queue.push_back(None); // end-of-stream marker, behind everything already queued
            }
        }
        self.changed.notify_waiters();
    }

    /// Wait until `pred` holds on the request's state.
    async fn wait_for(&self, pred: impl Fn(&ReqState) -> bool) {
        loop {
            let n = self.changed.notified();
            tokio::pin!(n);
            n.as_mut().enable();
            if pred(&self.lock()) {
                return;
            }
            n.await;
        }
    }

    /// The result, or the NodeError / unavailable error the request ended with.
    fn outcome(&self, host_id: &str) -> Result<Obj, NodeError> {
        let st = self.lock();
        if let Some(why) = &st.unavailable {
            return Err(NodeError::unavailable(host_id, why.clone()));
        }
        let empty = Obj::new();
        let res = st.res.as_ref().unwrap_or(&empty);
        if res.get("ok") == Some(&Value::Bool(true)) {
            return Ok(match res.get("result") {
                Some(Value::Object(o)) => o.clone(),
                _ => Obj::new(),
            });
        }
        let err = match res.get("error") {
            Some(Value::Object(o)) => o.clone(),
            _ => Obj::new(),
        };
        let msg = match err.get("message") {
            Some(v) if ec_proto::sanitize::truthy(v) => ec_proto::pyfmt::str_value(v),
            _ => "the node refused".into(),
        };
        Err(NodeError::new(
            host_id,
            &code(err.get("code")),
            text(&msg, 1000),
        ))
    }
}

// ── the link ─────────────────────────────────────────────────────────────────────────────────────────────
/// The command of one connection attempt: called on every attempt, so a key created after the first failure is
/// picked up; a `HostsError` is shown as the link's error and retried rarely because only a person can fix it.
pub type ArgvFn = Arc<dyn Fn() -> Result<Vec<String>, HostsError> + Send + Sync>;
pub type Listener = Arc<dyn Fn() + Send + Sync>;

enum WriteCmd {
    Frame(Vec<u8>),
    /// a DATA frame; the sender learns whether it was written (Python: `write` + `drain`)
    Data(Vec<u8>, oneshot::Sender<bool>),
}

struct State {
    state: String,
    error: Option<String>,
    notes: Vec<String>,
    retry_at: Option<f64>,
    up_mono: Option<f64>,
    connected_wall: Option<f64>,
    last_seen: Option<f64>,
    last_rx: f64,
    last_sample_mono: Option<f64>,
    rtt_ms: Option<i64>,
    skew_s: Option<f64>,
    latest: HashMap<String, Arc<Obj>>,
    pending: HashMap<i64, Arc<Req>>,
    next_id: i64,
    pings: HashMap<i64, f64>,
    next_ping: i64,
    /// the node process, while it has not been reaped
    pid: Option<i32>,
    session: u64,
    writer: Option<mpsc::UnboundedSender<WriteCmd>>,
    stopping: bool,
    drop_reason: Option<String>,
    backoff: f64,
    log_window: f64,
    log_count: u32,
    last_wait_s: Option<f64>,
    wake: bool,
}

struct Link {
    host: HostConfig,
    mask: Vec<String>,
    argv: ArgvFn,
    env: Option<Vec<(String, String)>>,
    bus: Arc<EventBus>,
    t: LinkTiming,
    on_change: RwLock<Option<Listener>>,
    remote_cmd: RwLock<Option<String>>,
    epoch: Instant,
    st: Mutex<State>,
    wake: Notify,
    task: Mutex<Option<JoinHandle<()>>>,
}

/// How one connection attempt ended.
struct End {
    error: String,
    /// forced retry delay; None = ordinary exponential backoff
    wait: Option<f64>,
    incompatible: bool,
    /// how long the link was up
    up_s: f64,
}

#[derive(Default)]
struct Talk {
    /// why the conversation ended, when we know (None: the node closed its pipes)
    err: Option<String>,
    wait: Option<f64>,
    incompatible: bool,
    up_s: f64,
    was_up: bool,
}

enum Handshake {
    Closed,
    Fatal(String),
    Proto(String),
}

impl Link {
    fn clock(&self) -> f64 {
        self.epoch.elapsed().as_secs_f64()
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.st.guard()
    }

    fn label(&self) -> &str {
        &self.host.label
    }

    fn link(&self) -> LinkStatus {
        let now = self.clock();
        let st = self.lock();
        let mut state = st.state.clone();
        if state == "up" {
            let reference = st.last_sample_mono.or(st.up_mono);
            if reference.is_some_and(|r| now - r > self.t.stale_s) {
                state = "stale".into();
            }
        }
        let live = state == "up" || state == "stale";
        let mut l = LinkStatus::new(&state, st.error.clone());
        l.retry_in_s = st.retry_at.map(|at| round_i64((at - now).max(0.0)));
        l.notes.clone_from(&st.notes);
        l.last_seen = st.last_seen;
        l.connected_since = st.connected_wall.filter(|_| live);
        l.rtt_ms = st.rtt_ms.filter(|_| live);
        l.skew_s = st.skew_s.filter(|_| live);
        l
    }

    fn changed(&self) {
        let f = self.on_change.read_guard().as_ref().map(Arc::clone);
        if let Some(f) = f
            && catch_unwind(AssertUnwindSafe(|| f())).is_err()
        {
            // a listener bug must not break the link
            tracing::error!(target: LOG, "link {}: change listener failed", self.host.id);
        }
    }

    /// Move to `state`; `error` replaces the error unless None (and "up" clears it); `wait` sets the retry clock.
    fn to(&self, state: &str, error: Option<String>, wait: Option<f64>) {
        let now = self.clock();
        let changed = {
            let mut st = self.lock();
            let before = (st.state.clone(), st.error.clone(), st.notes.clone());
            st.state = state.to_string();
            if state == "up" {
                st.error = None;
            } else if error.is_some() {
                st.error = error;
            }
            st.retry_at = wait.map(|w| now + w);
            if state != "up" {
                st.up_mono = None;
            }
            (st.state.clone(), st.error.clone(), st.notes.clone()) != before
        };
        if changed {
            self.changed();
        }
    }

    fn publish(&self, kind: &str, msg: &Obj) {
        self.bus.publish(
            &format!("node.{kind}"),
            json!({"host_id": self.host.id, "msg": Value::Object(msg.clone())}),
        );
    }

    fn kill(&self, reason: Option<&str>) {
        let pid = {
            let mut st = self.lock();
            if let Some(r) = reason
                && st.drop_reason.is_none()
            {
                st.drop_reason = Some(r.to_string());
            }
            st.pid
        };
        kill_proc(pid);
    }

    fn fail_all(&self, reason: &str) {
        let pending: Vec<Arc<Req>> = {
            let mut st = self.lock();
            st.pings.clear();
            st.pending.drain().map(|(_, r)| r).collect()
        };
        for req in pending {
            req.finish(None, Some(reason.to_string()));
        }
    }

    /// Wait for a reconnect request (true) or the timeout (false).
    async fn wait_wake(&self, d: Duration) -> bool {
        let deadline = tokio::time::Instant::now() + d;
        loop {
            let n = self.wake.notified();
            tokio::pin!(n);
            n.as_mut().enable();
            if self.lock().wake {
                return true;
            }
            if tokio::time::timeout_at(deadline, n).await.is_err() {
                return self.lock().wake;
            }
        }
    }

    // ── writing ──────────────────────────────────────────────────────────────────────────────────────────
    fn writer(&self) -> Option<mpsc::UnboundedSender<WriteCmd>> {
        self.lock().writer.clone().filter(|w| !w.is_closed())
    }

    /// Queue one small control frame (ping, cancel, end, credit). Never waits: control frames must not sit behind a
    /// bulk transfer's back-pressure. A dead link is ignored: its requests are failed elsewhere.
    fn send_nowait(&self, msg: &Obj) {
        let Some(w) = self.writer() else {
            return;
        };
        match ec_proto::encode_json(msg) {
            Ok(frame) => {
                let _ = w.send(WriteCmd::Frame(frame));
            }
            Err(_) => {
                let kind = msg.get("type").and_then(Value::as_str).unwrap_or("?");
                tracing::error!(target: LOG, "link {}: could not encode a {kind} frame", self.host.id);
            }
        }
    }

    async fn send_data(&self, frame: Vec<u8>) -> Result<(), NodeError> {
        let dropped = || NodeError::unavailable(&self.host.id, "the connection dropped");
        let w = self.writer().ok_or_else(dropped)?;
        let (tx, rx) = oneshot::channel();
        w.send(WriteCmd::Data(frame, tx)).map_err(|_| dropped())?;
        match rx.await {
            Ok(true) => Ok(()),
            _ => Err(dropped()),
        }
    }

    /// Ask the node to stop a request. It still answers with exactly one res, so the request stays registered (its id
    /// cannot be reused, and it still counts toward the node's limit) until that answer arrives.
    fn cancel_request(&self, req: &Req) {
        let now = self.clock();
        {
            let mut st = req.lock();
            if st.finished || st.cancel_sent {
                return;
            }
            st.cancel_sent = true;
            st.cancel_at = now;
        }
        self.send_nowait(&ec_proto::msg::cancel(req.rid));
    }

    fn begin(&self, op: &str, args: Option<Obj>, streaming: bool) -> Result<Arc<Req>, NodeError> {
        let hid = &self.host.id;
        if !ec_proto::specs::is_op(op) {
            return Err(NodeError::new(
                hid,
                ec_proto::UNKNOWN_OP,
                format!(
                    "{} is not an operation of the node protocol",
                    ec_proto::pyfmt::repr_str(op)
                ),
            ));
        }
        let mut st = self.lock();
        let w = st.writer.clone().filter(|w| !w.is_closed());
        let Some(w) = w.filter(|_| st.state == "up") else {
            let why = st.error.clone().unwrap_or_else(|| st.state.clone());
            return Err(NodeError::unavailable(
                hid,
                format!("{} is not connected ({})", self.label(), text(&why, 200)),
            ));
        };
        if st.pending.len() >= ec_proto::MAX_INFLIGHT {
            return Err(NodeError::new(
                hid,
                ec_proto::BUSY,
                format!(
                    "{} already has {} requests in flight",
                    self.label(),
                    ec_proto::MAX_INFLIGHT
                ),
            ));
        }
        let rid = loop {
            let rid = st.next_id;
            st.next_id = rid % ec_proto::MAX_ID + 1;
            if !st.pending.contains_key(&rid) {
                break rid; // never reuse an id that is still in flight
            }
        };
        let frame = ec_proto::encode_json(&ec_proto::msg::req(rid, op, args)).map_err(|e| {
            NodeError::new(
                hid,
                ec_proto::BAD_ARGS,
                format!("cannot send {op}: {}", text(&e.to_string(), 200)),
            )
        })?;
        let req = Arc::new(Req::new(rid, op, streaming));
        st.pending.insert(rid, Arc::clone(&req));
        let _ = w.send(WriteCmd::Frame(frame));
        Ok(req)
    }

    // ── liveness ─────────────────────────────────────────────────────────────────────────────────────────
    fn ping(&self) {
        let now = self.clock();
        let pid = {
            let mut st = self.lock();
            if st.pings.len() > 16 {
                st.pings.clear(); // a node that never answers cannot grow this
            }
            let pid = st.next_ping;
            st.next_ping = pid % ec_proto::MAX_ID + 1;
            st.pings.insert(pid, now);
            pid
        };
        self.send_nowait(&ec_proto::msg::ping(pid));
    }

    async fn pinger(self: Arc<Self>) {
        loop {
            // the first one at once, so the page has an rtt right away; a panicking ping is logged, the next one runs
            crate::panics::guarded_sync("link pinger", || self.ping());
            tokio::time::sleep(secs(self.t.ping_s)).await;
        }
    }

    async fn watchdog(self: Arc<Self>) {
        let tick = (self.t.silent_s / 5.0).clamp(0.02, 1.0);
        loop {
            tokio::time::sleep(secs(tick)).await;
            // a tick that panics is logged and the next one runs: the watchdog never silently stops watching
            if crate::panics::guarded_sync("link watchdog", || self.watch_once()) == Some(true) {
                return;
            }
        }
    }

    /// One watchdog tick; true once it has killed a silent node (the watchdog's work is done).
    fn watch_once(&self) -> bool {
        let now = self.clock();
        if now - self.lock().last_rx > self.t.silent_s {
            self.kill(Some(&format!(
                "no message from the node for {} s",
                g(self.t.silent_s)
            )));
            return true;
        }
        // cancelled long ago and never answered: forget it
        let orphans: Vec<Arc<Req>> = {
            let mut st = self.lock();
            let ids: Vec<i64> = st
                .pending
                .iter()
                .filter(|(_, r)| {
                    let rs = r.lock();
                    rs.cancel_sent && now - rs.cancel_at > ORPHAN_MAX_S
                })
                .map(|(id, _)| *id)
                .collect();
            ids.iter().filter_map(|id| st.pending.remove(id)).collect()
        };
        for r in orphans {
            r.finish(
                None,
                Some("the node never answered a cancelled request".into()),
            );
        }
        false
    }

    // ── reading ──────────────────────────────────────────────────────────────────────────────────────────
    fn request(&self, rid: Option<&Value>) -> Option<Arc<Req>> {
        let rid = rid.filter(|v| ec_proto::valid_id(v))?.as_i64()?;
        self.lock().pending.get(&rid).cloned()
    }

    fn mismatch(&self, hello: &Obj) -> String {
        let theirs = hello.get("proto").cloned().unwrap_or(Value::Null);
        let newer = theirs.as_i64().is_some_and(|p| p > ec_proto::PROTO) && !theirs.is_f64();
        let side = if newer {
            "the Engine Console hub".to_string()
        } else {
            format!("ecnode on {}", self.label())
        };
        let version = match hello.get("version") {
            Some(v) if ec_proto::sanitize::truthy(v) => ec_proto::pyfmt::str_value(v),
            _ => "?".into(),
        };
        format!(
            "{} runs ecnode {} (protocol {}); this Engine Console speaks protocol {}. Update {side}.",
            self.label(),
            text(&version, 32),
            ec_proto::pyfmt::str_value(&theirs),
            ec_proto::PROTO
        )
    }

    fn node_log(&self, out: &Obj) {
        let now = self.clock();
        let ok = {
            let mut st = self.lock();
            if now - st.log_window > 10.0 {
                st.log_window = now;
                st.log_count = 0;
            }
            st.log_count += 1;
            st.log_count <= NODE_LOG_BURST
        };
        if ok {
            let line = text_value(out.get("msg").filter(|v| !v.is_null()), 300);
            let line = if out.get("msg").is_none_or(Value::is_null) {
                String::new()
            } else {
                line
            };
            match out.get("level").and_then(Value::as_str) {
                Some("warn" | "warning" | "error") => {
                    tracing::warn!(target: LOG, "node {}: {line}", self.host.id);
                }
                _ => tracing::debug!(target: LOG, "node {}: {line}", self.host.id),
            }
        }
    }

    /// Handle one message; returns a reason to drop the link, or None.
    fn on_json(&self, msg: &Obj, now: f64) -> Option<String> {
        let kind = msg.get("type").and_then(Value::as_str);
        if kind == Some("res") {
            if let Some(req) = self.request(msg.get("id")) {
                // unknown ids (late answers to forgotten requests) are ignored
                self.lock().pending.remove(&req.rid);
                req.finish(Some(ec_proto::sanitize_res(msg, &req.op)), None);
            }
            return None;
        }
        if kind == Some("event") {
            let req = self.request(msg.get("id"))?;
            if !req.streaming {
                return None;
            }
            let ev = match ec_proto::sanitize_event(msg, &req.op).remove("event") {
                Some(Value::Object(o)) => o,
                _ => Obj::new(),
            };
            let queued = {
                let mut st = req.lock();
                if st.cancel_sent {
                    return None;
                }
                if ev.values().any(|v| !v.is_null()) && st.queued_events < self.t.max_stream_events
                {
                    st.queued_events += 1;
                    st.queue.push_back(Some(StreamItem::Event(ev)));
                    true
                } else {
                    false
                }
            };
            if queued {
                req.changed.notify_waiters();
            }
            return None;
        }
        let out = match ec_proto::sanitize_push(msg) {
            Err(e) => {
                return Some(format!(
                    "protocol error from the node: {}",
                    text(&e.to_string(), 300)
                ));
            }
            Ok(None) => return None, // a message type this hub does not know: ignore it
            Ok(Some(o)) => o,
        };
        match out.get("type").and_then(Value::as_str).unwrap_or("") {
            "sample" => {
                let mut out = out;
                let announced: Vec<String> = {
                    let st = self.lock();
                    st.latest
                        .get("hello")
                        .and_then(|h| h.get("gpus"))
                        .and_then(Value::as_array)
                        .map(|gs| {
                            gs.iter()
                                .filter_map(|g| g.get("uuid").and_then(Value::as_str))
                                .map(str::to_string)
                                .collect()
                        })
                        .unwrap_or_default()
                };
                // no new GPUs from a sample
                let gpus: Obj = match out.get("gpus") {
                    Some(Value::Object(m)) => m
                        .iter()
                        .filter(|(u, _)| announced.contains(u))
                        .map(|(u, v)| (u.clone(), v.clone()))
                        .collect(),
                    _ => Obj::new(),
                };
                out.insert("gpus".into(), Value::Object(gpus));
                let skew = out
                    .get("t")
                    .and_then(Value::as_f64)
                    .map(|t| ((t - wall()) * 10.0).round_ties_even() / 10.0);
                {
                    let mut st = self.lock();
                    st.latest.insert("sample".into(), Arc::new(out.clone()));
                    st.last_sample_mono = Some(now);
                    if skew.is_some() {
                        st.skew_s = skew;
                    }
                }
                self.publish("sample", &out);
            }
            t @ ("disks" | "inventory") => {
                self.lock().latest.insert(t.into(), Arc::new(out.clone()));
                self.publish(t, &out);
            }
            "hello" => {
                // a re-sent hello: same rules as the first
                if !proto_ok(out.get("proto")) {
                    return Some(self.mismatch(&out));
                }
                self.lock()
                    .latest
                    .insert("hello".into(), Arc::new(out.clone()));
                self.publish("hello", &out);
                self.changed();
            }
            t @ ("metrics" | "models_changed") => self.publish(t, &out),
            "pong" => {
                let mut st = self.lock();
                let t0 = out
                    .get("id")
                    .filter(|v| v.is_i64() || v.is_u64())
                    .and_then(Value::as_i64)
                    .and_then(|id| st.pings.remove(&id));
                if let Some(t0) = t0 {
                    st.rtt_ms = Some(round_i64((now - t0) * 1000.0));
                }
            }
            "credit" => {
                let n = out
                    .get("bytes")
                    .filter(|v| v.is_i64() || v.is_u64())
                    .and_then(Value::as_i64);
                if let (Some(req), Some(n)) = (self.request(out.get("id")), n)
                    && n > 0
                {
                    {
                        let mut st = req.lock();
                        st.send_credit = (st.send_credit.saturating_add(n)).min(MAX_CREDIT);
                    }
                    req.changed.notify_waiters();
                }
            }
            "log" => self.node_log(&out),
            "fatal" => {
                let why = match out.get("error") {
                    Some(v) if ec_proto::sanitize::truthy(v) => ec_proto::pyfmt::str_value(v),
                    _ => "no reason given".into(),
                };
                return Some(format!("the node gave up: {}", text(&why, 300)));
            }
            _ => {}
        }
        None
    }

    fn on_data(&self, stream: u32, data: Vec<u8>) -> Option<String> {
        let req = self.lock().pending.get(&i64::from(stream)).cloned()?;
        if !req.streaming {
            return None; // not one the hub opened for DATA: drop
        }
        let n = i64::try_from(data.len()).unwrap_or(i64::MAX);
        let queued = {
            let mut st = req.lock();
            if n > st.recv_window {
                return Some(format!(
                    "the node sent more data on stream {stream} than the credit this hub granted it"
                ));
            }
            st.recv_window -= n;
            if !data.is_empty() && !st.cancel_sent {
                st.queue
                    .push_back(Some(StreamItem::Data(Bytes::from(data))));
                true
            } else {
                false
            }
        };
        if queued {
            req.changed.notify_waiters();
        }
        None
    }

    /// Route frames until the node closes (None) or misbehaves (the reason, after killing it).
    async fn read_loop<R: AsyncRead + Unpin>(&self, r: &mut R) -> Option<String> {
        let t = &self.t;
        let (mut win_start, mut win_bytes, mut win_frames, mut since_yield) =
            (self.clock(), 0u64, 0u64, 0u32);
        loop {
            let (frame, size) = match read_frame(r).await {
                Err(FrameErr::Closed) | Ok(None) => return None,
                Err(FrameErr::Proto(e)) => {
                    self.kill(None);
                    return Some(format!("protocol error from the node: {}", text(&e, 300)));
                }
                Ok(Some(x)) => x,
            };
            let now = self.clock();
            {
                let mut st = self.lock();
                st.last_rx = now;
                st.last_seen = Some(wall());
            }
            // The flood limit is for JSON messages, which nothing else bounds. DATA is bounded by the credit this hub
            // grants (a model copy legitimately moves far more than 20 MB in 10 s).
            let counts = match &frame {
                ec_proto::Frame::Json(_) => true,
                ec_proto::Frame::Data { data, .. } => data.is_empty(),
            };
            if counts {
                if now - win_start > t.flood_window_s {
                    (win_start, win_bytes, win_frames) = (now, 0, 0);
                }
                win_bytes += size as u64;
                win_frames += 1;
                if win_bytes > t.flood_bytes || win_frames > t.flood_frames {
                    self.kill(None);
                    return Some(format!(
                        "the node sent over {} MB or {} messages in {} s; dropped it",
                        g(t.flood_bytes as f64 / 1_048_576.0),
                        t.flood_frames,
                        g(t.flood_window_s)
                    ));
                }
            }
            let err = match frame {
                ec_proto::Frame::Json(msg) => self.on_json(&msg, now),
                ec_proto::Frame::Data { stream, data } => self.on_data(stream, data),
            };
            if err.is_some() {
                self.kill(None);
                return err;
            }
            since_yield += 1;
            if since_yield >= 64 {
                // a node writing flat out must not starve the runtime
                since_yield = 0;
                tokio::task::yield_now().await;
            }
        }
    }

    async fn handshake<R: tokio::io::AsyncBufRead + Unpin>(
        &self,
        r: &mut R,
    ) -> Result<(Obj, usize), Handshake> {
        let skipped = match ec_proto::find_magic_async(r, READ_LIMIT).await {
            Ok(n) => n,
            Err(ec_proto::ReadError::Io(_)) => return Err(Handshake::Closed),
            Err(ec_proto::ReadError::Proto(e)) => {
                let m = e.to_string();
                // the far side simply closed (ssh failing, a shell exiting): its exit status explains it
                if m.contains("closed before") {
                    return Err(Handshake::Closed);
                }
                return Err(Handshake::Proto(m));
            }
        };
        let frame = match read_frame(r).await {
            Ok(Some((f, _))) => f,
            Ok(None) | Err(FrameErr::Closed) => return Err(Handshake::Closed),
            Err(FrameErr::Proto(e)) => return Err(Handshake::Proto(e)),
        };
        let ec_proto::Frame::Json(msg) = frame else {
            return Err(Handshake::Proto(
                "the node's first frame was not a message".into(),
            ));
        };
        let t = msg.get("type");
        if t.and_then(Value::as_str) == Some("fatal") {
            let why = match ec_proto::clean::clean(
                msg.get("error").unwrap_or(&Value::Null),
                &ec_proto::clean::s(1000),
            ) {
                Value::String(s) if !s.is_empty() => s,
                _ => "no reason given".into(),
            };
            return Err(Handshake::Fatal(format!(
                "the node gave up: {}",
                text(&why, 300)
            )));
        }
        if t.and_then(Value::as_str) != Some("hello") {
            return Err(Handshake::Proto(format!(
                "the node's first message was {}, not hello",
                ec_proto::pyfmt::repr_str(&text_value(t, 40))
            )));
        }
        match ec_proto::sanitize_push(&msg) {
            Ok(Some(hello)) => Ok((hello, skipped)),
            Ok(None) => Err(Handshake::Proto("the node sent no hello".into())),
            Err(e) => Err(Handshake::Proto(e.to_string())),
        }
    }

    fn connected(&self, hello: &Obj, skipped: usize) {
        let mut notes = Vec::new();
        if skipped > 0 {
            notes.push(format!(
                "ignored {skipped} byte(s) the far side printed before the node started (a shell startup file \
                 writing to stdout?)"
            ));
        }
        if self.host.ssh.is_some() && hello.get("forced_cmd") != Some(&Value::Bool(true)) {
            notes.push(
                "reached with an UNRESTRICTED ssh key: restrict it to the node command in authorized_keys \
                 (`engine-console hosts init-key` prints the line)"
                    .into(),
            );
        }
        let now = self.clock();
        {
            let mut st = self.lock();
            st.latest.insert("hello".into(), Arc::new(hello.clone()));
            st.notes = notes;
            st.up_mono = Some(now);
            st.last_rx = now;
            st.last_sample_mono = None;
            let w = wall();
            st.connected_wall = Some(w);
            st.last_seen = Some(w);
            st.rtt_ms = None;
            st.skew_s = None;
            st.pings.clear();
            st.wake = false; // a reconnect request has been served
        }
        self.to("up", None, None);
        self.publish("hello", hello);
        let version = match hello.get("version") {
            Some(v) if ec_proto::sanitize::truthy(v) => text_value(Some(v), 32),
            _ => "?".to_string(),
        };
        let ngpus = hello
            .get("gpus")
            .and_then(Value::as_array)
            .map_or(0, Vec::len);
        tracing::info!(target: LOG, "link {}: up (ecnode {version}, {ngpus} GPU(s))", self.host.id);
    }

    /// `node.info` re-reads the node's policy: show the fresh facts without waiting for a reconnect.
    fn refresh_hello(&self, result: &Obj) {
        if !proto_ok(result.get("proto")) || !self.lock().latest.contains_key("hello") {
            return;
        }
        let mut hello = result.clone();
        hello.insert("type".into(), json!("hello"));
        let gpus: Vec<Value> = result
            .get("gpus")
            .and_then(Value::as_array)
            .map(|gs| {
                gs.iter()
                    .filter(|g| g.get("uuid").is_some_and(ec_proto::sanitize::truthy))
                    .cloned()
                    .collect()
            })
            .unwrap_or_default();
        hello.insert("gpus".into(), Value::Array(gpus));
        self.lock()
            .latest
            .insert("hello".into(), Arc::new(hello.clone()));
        self.publish("hello", &hello);
        self.changed();
    }

    async fn converse<R: AsyncRead + Unpin + Send + 'static>(self: &Arc<Self>, out: R) -> Talk {
        let t = &self.t;
        let mut r = BufReader::with_capacity(READ_LIMIT, out);
        let hs = tokio::time::timeout(secs(t.hello_timeout_s), self.handshake(&mut r)).await;
        let (hello, skipped) = match hs {
            Err(_) => {
                self.kill(None);
                return Talk {
                    err: Some(format!(
                        "no answer from the node within {} s",
                        g(t.hello_timeout_s)
                    )),
                    ..Talk::default()
                };
            }
            Ok(Err(Handshake::Fatal(m))) => {
                return Talk {
                    err: Some(m),
                    ..Talk::default()
                };
            }
            // The far side simply closed (ssh failing, a shell exiting). Do not kill it: its exit status and its last
            // words on stderr explain why far better than "closed before the protocol".
            Ok(Err(Handshake::Closed)) => return Talk::default(),
            Ok(Err(Handshake::Proto(e))) => {
                self.kill(None);
                return Talk {
                    err: Some(format!("protocol error from the node: {}", text(&e, 300))),
                    ..Talk::default()
                };
            }
            Ok(Ok(x)) => x,
        };
        if !proto_ok(hello.get("proto")) {
            let msg = self.mismatch(&hello);
            // so the page can show which version the far side runs
            self.lock().latest.insert("hello".into(), Arc::new(hello));
            return Talk {
                err: Some(msg),
                wait: Some(t.fix_retry_s),
                incompatible: true,
                ..Talk::default()
            };
        }
        self.connected(&hello, skipped);
        let up_at = self.clock();
        let watch = tokio::spawn(Arc::clone(self).watchdog());
        let pinger = tokio::spawn(Arc::clone(self).pinger());
        let err = self.read_loop(&mut r).await;
        watch.abort();
        pinger.abort();
        let _ = watch.await;
        let _ = pinger.await;
        Talk {
            err,
            up_s: self.clock() - up_at,
            was_up: true,
            ..Talk::default()
        }
    }

    async fn session(self: &Arc<Self>) -> End {
        let t = &self.t;
        let argv = match (self.argv)() {
            Ok(a) => a,
            // e.g. no hub key yet: only a person can fix it
            Err(e) => {
                return End {
                    error: e.to_string(),
                    wait: Some(t.fix_retry_s),
                    incompatible: false,
                    up_s: 0.0,
                };
            }
        };
        let Some(program) = argv.first() else {
            return End {
                error: "there is no command to run for this host".into(),
                wait: None,
                incompatible: false,
                up_s: 0.0,
            };
        };
        let base = std::path::Path::new(program)
            .file_name()
            .map_or_else(|| program.clone(), |n| n.to_string_lossy().into_owned());
        // A bare name (`ssh`, a relative NODE_COMMAND) is looked up on the PATH the child gets, which is the hub's
        // fixed tool path (hosts.rs ssh_env/local_env), never the caller's PATH; argv[0] stays as given.
        let search = self
            .env
            .as_ref()
            .and_then(|e| e.iter().find(|(k, _)| k == "PATH").map(|(_, v)| v.clone()))
            .unwrap_or_else(crate::tools::tool_path);
        let Some(exe) = crate::tools::find_in(program, &search) else {
            let e = std::io::Error::from_raw_os_error(nix::libc::ENOENT);
            return End {
                error: format!("could not start {base}: {}", strerror(&e)),
                wait: None,
                incompatible: false,
                up_s: 0.0,
            };
        };
        let mut cmd = tokio::process::Command::new(&exe);
        cmd.arg0(program)
            .args(&argv[1..])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .process_group(0)
            .kill_on_drop(false);
        if let Some(env) = &self.env {
            cmd.env_clear().envs(env.iter().map(|(k, v)| (k, v)));
        }
        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                return End {
                    error: format!("could not start {base}: {}", strerror(&e)),
                    wait: None,
                    incompatible: false,
                    up_s: 0.0,
                };
            }
        };
        let pid = child.id().and_then(|p| i32::try_from(p).ok());
        tracing::debug!(target: LOG, "link {}: started {base} (pid {pid:?})", self.host.id);
        let (Some(stdin), Some(stdout), Some(mut stderr)) =
            (child.stdin.take(), child.stdout.take(), child.stderr.take())
        else {
            kill_proc(pid);
            return End {
                error: "could not start the node: no pipes".into(),
                wait: None,
                incompatible: false,
                up_s: 0.0,
            };
        };
        let (wtx, wrx) = mpsc::unbounded_channel();
        let writer_task = tokio::spawn(write_loop(stdin, wrx));
        let stopping = {
            let mut st = self.lock();
            st.pid = pid;
            st.session += 1;
            st.writer = Some(wtx);
            st.drop_reason = None;
            st.stopping
        };
        let guard = ProcGuard {
            link: Arc::clone(self),
            pid,
        };
        if stopping {
            kill_proc(pid); // stop() ran while the child was being spawned
        }
        let tail = Arc::new(Mutex::new(Vec::<u8>::new()));
        let tail2 = Arc::clone(&tail);
        let drain = tokio::spawn(async move {
            // losing the diagnostics must never take the link down
            let mut buf = vec![0u8; 4096];
            while let Ok(n) = stderr.read(&mut buf).await {
                if n == 0 {
                    break;
                }
                // bounded while reading: only the last STDERR_KEEP bytes are ever held
                let mut t = tail2.guard();
                t.extend_from_slice(&buf[..n]);
                let extra = t.len().saturating_sub(STDERR_KEEP);
                t.drain(..extra);
            }
        });
        let talk = self.converse(stdout).await;
        let reason = {
            let st = self.lock();
            talk.err
                .clone()
                .or_else(|| st.drop_reason.clone())
                .unwrap_or_else(|| "the node closed the connection".into())
        };
        // From here nothing may be sent, and nobody should wait for the process to be reaped below.
        self.lock().writer = None;
        self.fail_all(&reason);
        self.lock().latest.remove("sample");
        if talk.was_up {
            self.to("down", Some(reason), None);
        }
        let status = self.reap(&mut child, pid, drain).await;
        writer_task.abort();
        drop(guard);
        {
            let mut st = self.lock();
            st.pid = None;
            st.writer = None;
        }
        let err = talk.err.or_else(|| self.lock().drop_reason.clone());
        if let Some(error) = err {
            return End {
                error,
                wait: talk.wait,
                incompatible: talk.incompatible,
                up_s: talk.up_s,
            };
        }
        let rc = status.and_then(|s| {
            s.code()
                .or_else(|| std::os::unix::process::ExitStatusExt::signal(&s).map(|n| -n))
        });
        let stderr_text = String::from_utf8_lossy(&tail.guard()).into_owned();
        let remote_cmd = self.remote_cmd.read_guard().clone();
        let (msg, wait) = explain_exit(
            &self.host,
            rc,
            &stderr_text,
            remote_cmd.as_deref(),
            t.fix_retry_s,
            &self.mask,
        );
        End {
            error: msg,
            wait,
            incompatible: false,
            up_s: talk.up_s,
        }
    }

    /// Let a node that closed its pipes exit by itself, so its real exit status (ssh's 255, a shell's 127) and its
    /// last words on stderr can explain the drop; kill it if it does not.
    async fn reap(
        &self,
        child: &mut tokio::process::Child,
        pid: Option<i32>,
        drain: JoinHandle<()>,
    ) -> Option<std::process::ExitStatus> {
        let status = match tokio::time::timeout(secs(self.t.exit_grace_s), child.wait()).await {
            Ok(s) => s.ok(),
            Err(_) => {
                kill_proc(pid);
                tokio::time::timeout(Duration::from_secs(5), child.wait())
                    .await
                    .ok()
                    .and_then(Result::ok)
            }
        };
        if status.is_some() {
            self.lock().pid = None;
        }
        let mut drain = drain;
        if tokio::time::timeout(Duration::from_secs(2), &mut drain)
            .await
            .is_err()
        {
            drain.abort();
        }
        status
    }

    async fn run(self: Arc<Self>) {
        let t = self.t.clone();
        loop {
            if self.lock().stopping {
                return;
            }
            self.lock().wake = false;
            self.to("connecting", None, None);
            let end = match futures::FutureExt::catch_unwind(AssertUnwindSafe(self.session())).await
            {
                Ok(e) => e,
                Err(_) => {
                    tracing::error!(target: LOG, "link {}: internal error", self.host.id);
                    End {
                        error: "internal error in the link (panic)".into(),
                        wait: None,
                        incompatible: false,
                        up_s: 0.0,
                    }
                }
            };
            let wait = {
                let mut st = self.lock();
                if st.stopping {
                    return;
                }
                if st.wake {
                    continue; // reconnect() asked for a retry now: not a failure to back off from
                }
                let wait = match end.wait {
                    Some(w) => w,
                    None => {
                        if end.up_s >= t.stable_s {
                            st.backoff = t.backoff_min_s;
                        }
                        let w = st.backoff;
                        st.backoff = (st.backoff * 2.0).min(t.backoff_max_s);
                        w
                    }
                };
                st.last_wait_s = Some(wait);
                wait
            };
            let state = if end.incompatible {
                "incompatible"
            } else {
                "down"
            };
            self.to(state, Some(end.error.clone()), Some(wait));
            tracing::warn!(
                target: LOG,
                "link {}: {state}: {} (retry in {wait:.0} s)",
                self.host.id,
                text(&end.error, 300)
            );
            self.wait_wake(secs(wait)).await;
        }
    }
}

fn proto_ok(v: Option<&Value>) -> bool {
    v.and_then(Value::as_f64) == Some(ec_proto::PROTO as f64)
}

/// Kills the node's process group if the session ends without reaping it (a cancelled session task).
struct ProcGuard {
    link: Arc<Link>,
    pid: Option<i32>,
}

impl Drop for ProcGuard {
    fn drop(&mut self) {
        let still = self.link.lock().pid;
        if still.is_some() && still == self.pid {
            kill_proc(self.pid);
        }
    }
}

async fn write_loop(
    mut stdin: tokio::process::ChildStdin,
    mut rx: mpsc::UnboundedReceiver<WriteCmd>,
) {
    while let Some(cmd) = rx.recv().await {
        let (frame, ack) = match cmd {
            WriteCmd::Frame(f) => (f, None),
            WriteCmd::Data(f, a) => (f, Some(a)),
        };
        let ok = stdin.write_all(&frame).await.is_ok() && stdin.flush().await.is_ok();
        if let Some(a) = ack {
            let _ = a.send(ok);
        }
        if !ok {
            break;
        }
    }
    // dropping stdin closes the node's input: it exits on EOF
}

/// One machine's node, kept connected: spawn, handshake, read, ping, and reconnect with backoff, forever.
pub struct NodeLink {
    inner: Arc<Link>,
}

impl std::fmt::Debug for NodeLink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NodeLink")
            .field("host", &self.inner.host.id)
            .finish_non_exhaustive()
    }
}

impl NodeLink {
    /// `env` None: the node inherits the console's environment (only tests do that). `mask`: text never shown.
    pub fn new(
        host: HostConfig,
        argv: ArgvFn,
        env: Option<Vec<(String, String)>>,
        bus: Arc<EventBus>,
        timing: LinkTiming,
        remote_cmd: Option<String>,
        mask: Vec<String>,
    ) -> Self {
        let backoff = timing.backoff_min_s;
        let inner = Arc::new(Link {
            host,
            mask,
            argv,
            env,
            bus,
            t: timing,
            on_change: RwLock::new(None),
            remote_cmd: RwLock::new(remote_cmd),
            epoch: Instant::now(),
            st: Mutex::new(State {
                state: "connecting".into(),
                error: None,
                notes: Vec::new(),
                retry_at: None,
                up_mono: None,
                connected_wall: None,
                last_seen: None,
                last_rx: 0.0,
                last_sample_mono: None,
                rtt_ms: None,
                skew_s: None,
                latest: HashMap::new(),
                pending: HashMap::new(),
                next_id: 1,
                pings: HashMap::new(),
                next_ping: 1,
                pid: None,
                session: 0,
                writer: None,
                stopping: false,
                drop_reason: None,
                backoff,
                log_window: 0.0,
                log_count: 0,
                last_wait_s: None,
                wake: false,
            }),
            wake: Notify::new(),
            task: Mutex::new(None),
        });
        Self { inner }
    }

    /// A fixed command.
    pub fn fixed_argv(argv: Vec<String>) -> ArgvFn {
        Arc::new(move || Ok(argv.clone()))
    }

    /// Called (outside every lock) when the link's state, error or notes change, and on a new hello.
    pub fn set_on_change(&self, f: Option<Listener>) {
        *self.inner.on_change.write_guard() = f;
    }

    pub fn set_remote_cmd(&self, cmd: Option<String>) {
        *self.inner.remote_cmd.write_guard() = cmd;
    }

    /// Requests the node is (as far as the hub knows) still working on, cancelled ones included.
    pub fn inflight(&self) -> usize {
        self.inner.lock().pending.len()
    }

    /// The delay chosen before the next attempt (diagnostics, tests).
    pub fn last_wait_s(&self) -> Option<f64> {
        self.inner.lock().last_wait_s
    }

    /// Whether a retry is pending (the retry clock is set).
    pub fn retry_pending(&self) -> bool {
        self.inner.lock().retry_at.is_some()
    }

    /// The next request id to try (tests: id wrap-around).
    pub fn set_next_id(&self, id: i64) {
        self.inner.lock().next_id = id;
    }

    pub fn host_config(&self) -> &HostConfig {
        &self.inner.host
    }

    /// The node process, while it runs (tests: cleanup after a failed assertion).
    pub fn pid(&self) -> Option<i32> {
        self.inner.lock().pid
    }

    /// `open` with the concrete stream (tests reach its queue and cancel it from another task).
    pub fn open_stream(&self, op: &str, args: Option<Obj>) -> Result<LinkStream, NodeError> {
        let req = self.inner.begin(op, args, true)?;
        Ok(LinkStream {
            link: Arc::clone(&self.inner),
            req,
            iterated: false,
        })
    }
}

#[async_trait]
impl NodeClient for NodeLink {
    fn host(&self) -> &HostConfig {
        &self.inner.host
    }

    fn link(&self) -> LinkStatus {
        self.inner.link()
    }

    fn latest(&self, kind: &str) -> Option<Arc<Obj>> {
        self.inner.lock().latest.get(kind).cloned()
    }

    async fn call(&self, op: &str, args: Option<Obj>, timeout: Duration) -> Result<Obj, NodeError> {
        let link = &self.inner;
        let req = link.begin(op, args, false)?;
        // the caller went away (its future was dropped): the node should stop working for it
        let mut guard = CancelOnDrop {
            link: Arc::clone(link),
            req: Some(Arc::clone(&req)),
        };
        let done = tokio::time::timeout(timeout, req.wait_for(|s| s.finished)).await;
        guard.req = None;
        if done.is_err() {
            link.cancel_request(&req);
            return Err(NodeError::new(
                &link.host.id,
                ec_proto::TIMEOUT,
                format!(
                    "{op} did not answer within {} s; it may or may not have taken effect",
                    g(timeout.as_secs_f64())
                ),
            ));
        }
        let result = req.outcome(&link.host.id)?;
        if op == "node.info" {
            link.refresh_hello(&result);
        }
        Ok(result)
    }

    fn open(&self, op: &str, args: Option<Obj>) -> Box<dyn NodeStream> {
        match self.open_stream(op, args) {
            Ok(s) => Box::new(s),
            Err(e) => Box::new(FailedStream::new(e)),
        }
    }
}

struct CancelOnDrop {
    link: Arc<Link>,
    req: Option<Arc<Req>>,
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if let Some(r) = self.req.take() {
            self.link.cancel_request(&r);
        }
    }
}

/// What the FleetService needs from a link: a NodeClient that can be started, stopped and kicked.
#[async_trait]
pub trait LinkLike: NodeClient {
    async fn start(&self);
    async fn stop(&self);
    fn reconnect(&self);
}

#[async_trait]
impl LinkLike for NodeLink {
    async fn start(&self) {
        let mut task = self.inner.task.guard();
        if task.is_none() {
            *task = Some(tokio::spawn(Arc::clone(&self.inner).run()));
        }
    }

    /// Close the node's stdin (it exits on EOF, and a script it is running is not killed under it), give it a
    /// moment, then kill what is left.
    async fn stop(&self) {
        let link = &self.inner;
        let pid = {
            let mut st = link.lock();
            st.stopping = true;
            st.wake = true;
            st.writer = None; // closes the node's stdin
            st.pid
        };
        link.wake.notify_waiters();
        let task = link.task.guard().take();
        if let Some(mut task) = task
            && tokio::time::timeout(secs(link.t.exit_grace_s), &mut task)
                .await
                .is_err()
        {
            kill_proc(pid.or_else(|| link.lock().pid));
            if tokio::time::timeout(Duration::from_secs(5), &mut task)
                .await
                .is_err()
            {
                task.abort();
                let _ = task.await;
            }
        }
        kill_proc(link.lock().pid);
        link.fail_all("the console is shutting down");
        link.lock().latest.remove("sample");
        link.to("down", Some("stopped".into()), None);
    }

    /// Reconnect now: skip the wait, forget the backoff, and drop a connection that is up.
    fn reconnect(&self) {
        let link = &self.inner;
        let (pid, state, session) = {
            let mut st = link.lock();
            st.backoff = link.t.backoff_min_s;
            st.wake = true;
            (st.pid, st.state.clone(), st.session)
        };
        link.wake.notify_waiters();
        if pid.is_none() {
            return;
        }
        if state == "up" {
            {
                let mut st = link.lock();
                st.drop_reason = Some("reconnecting at your request".into());
                st.writer = None; // a node exits on EOF; the timer below catches one that does not
            }
            let l = Arc::clone(link);
            let grace = secs(link.t.exit_grace_s);
            if let Ok(rt) = tokio::runtime::Handle::try_current() {
                rt.spawn(async move {
                    tokio::time::sleep(grace).await;
                    let (now_pid, now_session) = {
                        let st = l.lock();
                        (st.pid, st.session)
                    };
                    if now_session == session && now_pid == pid {
                        kill_proc(pid);
                    }
                });
            }
        } else if state == "connecting" {
            // no hello yet: nothing to be graceful about, start over
            link.kill(Some("reconnecting at your request"));
        }
    }
}

// ── streams ──────────────────────────────────────────────────────────────────────────────────────────────
/// `NodeStream` over one request. Reading DATA hands credit back to the node, so a slow consumer slows only its own
/// stream: nothing here ever blocks the link's reader.
///
/// `next()` yields the node's events and DATA in order and simply ends (None) when the request does: at the node's
/// res (whether it succeeded or not) or when the link drops. Call `result()` afterwards. Dropping a stream whose
/// iteration started, or whose upload was started and not ended, before the res cancels the request.
pub struct LinkStream {
    link: Arc<Link>,
    req: Arc<Req>,
    iterated: bool,
}

impl std::fmt::Debug for LinkStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LinkStream")
            .field("id", &self.req.rid)
            .finish_non_exhaustive()
    }
}

/// Cancels a stream's request from anywhere (a writer blocked in `write` holds the stream itself).
#[derive(Clone)]
pub struct StreamCanceller {
    link: Arc<Link>,
    req: Arc<Req>,
}

impl StreamCanceller {
    pub fn cancel(&self) {
        self.link.cancel_request(&self.req);
    }
}

impl LinkStream {
    pub fn id(&self) -> i64 {
        self.req.rid
    }

    /// Items queued for the consumer right now (the end marker included).
    pub fn queued(&self) -> usize {
        self.req.lock().queue.len()
    }

    pub fn canceller(&self) -> StreamCanceller {
        StreamCanceller {
            link: Arc::clone(&self.link),
            req: Arc::clone(&self.req),
        }
    }

    fn consumed(&self, n: usize) {
        let grant = {
            let mut st = self.req.lock();
            st.unacked += i64::try_from(n).unwrap_or(0);
            if st.unacked >= CREDIT_BATCH as i64 && !st.finished {
                let g = st.unacked;
                st.unacked = 0;
                st.recv_window += g;
                Some(g)
            } else {
                None
            }
        };
        if let Some(g) = grant {
            self.link
                .send_nowait(&ec_proto::msg::credit(self.req.rid, g.unsigned_abs()));
        }
    }
}

#[async_trait]
impl NodeStream for LinkStream {
    async fn next(&mut self) -> Option<StreamItem> {
        self.iterated = true;
        self.req.wait_for(|s| !s.queue.is_empty()).await;
        let item = {
            let mut st = self.req.lock();
            match st.queue.pop_front() {
                Some(None) => {
                    // a second iteration ends at once instead of waiting forever
                    st.queue.push_front(None);
                    None
                }
                Some(Some(item)) => {
                    if matches!(item, StreamItem::Event(_)) {
                        st.queued_events = st.queued_events.saturating_sub(1);
                    }
                    Some(item)
                }
                None => None,
            }
        };
        if let Some(StreamItem::Data(d)) = &item {
            self.consumed(d.len());
        }
        item
    }

    /// Upload bytes as DATA frames of at most MAX_DATA, waiting for the node's credit when its window is used up.
    async fn write(&mut self, data: &[u8]) -> Result<(), NodeError> {
        let hid = self.link.host.id.clone();
        {
            let mut st = self.req.lock();
            if st.ended {
                return Err(NodeError::new(
                    &hid,
                    ec_proto::BAD_ARGS,
                    "write after end()",
                ));
            }
            st.uploading = true;
        }
        let mut pos = 0;
        while pos < data.len() {
            if self.req.finished() {
                // the node answered (usually refused) before the upload was done
                self.req.outcome(&hid)?;
                return Err(NodeError::new(
                    &hid,
                    ec_proto::CONFLICT,
                    "the node already finished this request",
                ));
            }
            let n = {
                let mut st = self.req.lock();
                if st.send_credit <= 0 {
                    None
                } else {
                    let n = (data.len() - pos)
                        .min(ec_proto::MAX_DATA)
                        .min(usize::try_from(st.send_credit).unwrap_or(usize::MAX));
                    st.send_credit -= i64::try_from(n).unwrap_or(0);
                    Some(n)
                }
            };
            let Some(n) = n else {
                self.req.wait_for(|s| s.send_credit > 0 || s.finished).await;
                continue;
            };
            let frame = ec_proto::encode_data(self.req.rid, &data[pos..pos + n])
                .map_err(|e| NodeError::new(&hid, ec_proto::BAD_ARGS, e.to_string()))?;
            self.link.send_data(frame).await?;
            pos += n;
        }
        Ok(())
    }

    async fn end(&mut self) -> Result<(), NodeError> {
        let (ended, finished) = {
            let st = self.req.lock();
            (st.ended, st.finished)
        };
        if ended {
            return Ok(());
        }
        if finished {
            self.req.outcome(&self.link.host.id)?;
            return Ok(());
        }
        self.req.lock().ended = true;
        self.link.send_nowait(&ec_proto::msg::end(self.req.rid));
        Ok(())
    }

    async fn result(&mut self) -> Result<Obj, NodeError> {
        self.req.wait_for(|s| s.finished).await;
        self.req.outcome(&self.link.host.id)
    }

    async fn cancel(&mut self) {
        self.link.cancel_request(&self.req);
    }
}

impl Drop for LinkStream {
    fn drop(&mut self) {
        // An upload that was started and dropped without end() or cancel() would hold a request slot on the node
        // (and an open partial file) until the link drops; a consumer that stopped iterating early is gone too. A
        // request whose consumer simply never read it is still wanted.
        let (finished, abandoned_upload) = {
            let st = self.req.lock();
            (st.finished, st.uploading && !st.ended)
        };
        if !finished && (self.iterated || abandoned_upload) {
            self.link.cancel_request(&self.req);
        }
    }
}

/// A stream that never started (link down, too many requests, bad arguments): yields nothing; `result()`,
/// `write()` and `end()` return the reason.
pub struct FailedStream {
    error: NodeError,
}

impl FailedStream {
    pub fn new(error: NodeError) -> Self {
        Self { error }
    }
}

#[async_trait]
impl NodeStream for FailedStream {
    async fn next(&mut self) -> Option<StreamItem> {
        None
    }
    async fn write(&mut self, _data: &[u8]) -> Result<(), NodeError> {
        Err(self.error.clone())
    }
    async fn end(&mut self) -> Result<(), NodeError> {
        Err(self.error.clone())
    }
    async fn result(&mut self) -> Result<Obj, NodeError> {
        Err(self.error.clone())
    }
    async fn cancel(&mut self) {}
}

// ── disabled hosts ───────────────────────────────────────────────────────────────────────────────────────
/// The link of a host that is switched off (`enabled: false`, or the whole fleet is): it never connects.
pub struct DisabledLink {
    host: HostConfig,
    reason: String,
}

impl DisabledLink {
    pub fn new(host: HostConfig, reason: &str) -> Self {
        Self {
            host,
            reason: reason.to_string(),
        }
    }

    fn unavailable(&self) -> NodeError {
        NodeError::unavailable(
            &self.host.id,
            format!("{} is disabled: {}", self.host.label, self.reason),
        )
    }
}

#[async_trait]
impl NodeClient for DisabledLink {
    fn host(&self) -> &HostConfig {
        &self.host
    }
    fn link(&self) -> LinkStatus {
        LinkStatus::new("disabled", Some(self.reason.clone()))
    }
    fn latest(&self, _kind: &str) -> Option<Arc<Obj>> {
        None
    }
    async fn call(
        &self,
        _op: &str,
        _args: Option<Obj>,
        _timeout: Duration,
    ) -> Result<Obj, NodeError> {
        Err(self.unavailable())
    }
    fn open(&self, _op: &str, _args: Option<Obj>) -> Box<dyn NodeStream> {
        Box::new(FailedStream::new(self.unavailable()))
    }
}

#[async_trait]
impl LinkLike for DisabledLink {
    async fn start(&self) {}
    async fn stop(&self) {}
    fn reconnect(&self) {}
}

// ── the fleet ────────────────────────────────────────────────────────────────────────────────────────────
pub type LinkFactory = Arc<dyn Fn(&HostConfig) -> Arc<dyn LinkLike> + Send + Sync>;
/// `time.monotonic` of the image cache (tests move it).
pub type Clock = Arc<dyn Fn() -> f64 + Send + Sync>;

struct FleetInner {
    cfg: Arc<Settings>,
    bus: Arc<EventBus>,
    hosts: Vec<HostConfig>,
    links: Vec<(String, Arc<dyn LinkLike>)>,
    resolver: RwLock<Option<InstanceResolver>>,
    counts: RwLock<Option<CountsProvider>>,
    images: Mutex<HashMap<String, (f64, Vec<ImageInfo>)>>,
    image_locks: Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
    seen: Mutex<HashMap<String, (String, Option<String>)>>,
    watcher: Mutex<Option<JoinHandle<()>>>,
    watch_s: RwLock<f64>,
    clock: RwLock<Clock>,
}

/// Every configured machine's link, and the hub's one picture of the fleet.
pub struct FleetService {
    inner: Arc<FleetInner>,
}

impl std::fmt::Debug for FleetService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FleetService")
            .field("hosts", &self.inner.hosts)
            .finish_non_exhaustive()
    }
}

/// How often the watcher looks for link changes nobody announced.
pub const WATCH_S: f64 = 1.0;

impl FleetService {
    /// Errors when hosts.json is unacceptable: the console must refuse to start rather than guess which machines it
    /// may reach. With `cfg.fleet_enabled` false no link ever connects (every host is `disabled`).
    pub fn new(cfg: Arc<Settings>, bus: Arc<EventBus>) -> Result<Self, HostsError> {
        Self::with_links(cfg, bus, None, None)
    }

    /// Python's keyword arguments `hosts=` and `link_factory=` (tests). Two hosts with one id are refused.
    #[allow(clippy::needless_pass_by_value)]
    pub fn with_links(
        cfg: Arc<Settings>,
        bus: Arc<EventBus>,
        hosts: Option<Vec<HostConfig>>,
        factory: Option<LinkFactory>,
    ) -> Result<Self, HostsError> {
        let hosts = match hosts {
            Some(h) => h,
            None if cfg.fleet_enabled => load_hosts(&cfg.hosts_file)?,
            None => default_hosts(),
        };
        let mut ids: Vec<&str> = hosts.iter().map(|h| h.id.as_str()).collect();
        ids.sort_unstable();
        ids.dedup();
        if ids.len() != hosts.len() {
            return Err(HostsError::Invalid("two hosts share an id".into()));
        }
        let start = Instant::now();
        let inner = Arc::new_cyclic(|weak: &Weak<FleetInner>| {
            let links = hosts
                .iter()
                .map(|h| {
                    let link = match &factory {
                        Some(f) => f(h),
                        None => make_link(&cfg, &bus, h, weak),
                    };
                    (h.id.clone(), link)
                })
                .collect();
            FleetInner {
                cfg: Arc::clone(&cfg),
                bus: Arc::clone(&bus),
                hosts: hosts.clone(),
                links,
                resolver: RwLock::new(None),
                counts: RwLock::new(None),
                images: Mutex::new(HashMap::new()),
                image_locks: Mutex::new(HashMap::new()),
                seen: Mutex::new(HashMap::new()),
                watcher: Mutex::new(None),
                watch_s: RwLock::new(WATCH_S),
                clock: RwLock::new(Arc::new(move || start.elapsed().as_secs_f64())),
            }
        });
        Ok(Self { inner })
    }

    /// How often the watcher looks for changes nobody announced (tests shorten it; set before `start`).
    pub fn set_watch_s(&self, s: f64) {
        *self.inner.watch_s.write_guard() = s;
    }

    /// The monotonic clock of the image cache (tests).
    pub fn set_clock(&self, clock: Clock) {
        *self.inner.clock.write_guard() = clock;
    }

    pub fn bus(&self) -> Arc<EventBus> {
        Arc::clone(&self.inner.bus)
    }

    /// The link of a host (tests and the CLI).
    pub fn link_of(&self, host_id: &str) -> Option<Arc<dyn LinkLike>> {
        self.inner.link(host_id).ok()
    }
}

fn make_link(
    cfg: &Arc<Settings>,
    bus: &Arc<EventBus>,
    host: &HostConfig,
    fleet: &Weak<FleetInner>,
) -> Arc<dyn LinkLike> {
    if !cfg.fleet_enabled {
        return Arc::new(DisabledLink::new(
            host.clone(),
            "the fleet is switched off (FLEET_ENABLED=false)",
        ));
    }
    if !host.enabled {
        return Arc::new(DisabledLink::new(host.clone(), "disabled in hosts.json"));
    }
    let w = Weak::clone(fleet);
    let hid = host.id.clone();
    let notify: Listener = Arc::new(move || {
        if let Some(f) = w.upgrade() {
            f.publish(&hid);
        }
    });
    let link = if host.id == LOCAL {
        let c = Arc::clone(cfg);
        NodeLink::new(
            host.clone(),
            Arc::new(move || Ok(local_argv(&c))),
            Some(local_env()),
            Arc::clone(bus),
            LinkTiming::default(),
            None,
            Vec::new(),
        )
    } else {
        // Built on every attempt (and refusing while there is no key), so running `hosts init-key` fixes the link by
        // itself at its next retry or on a manual reconnect.
        let c = Arc::clone(cfg);
        let h = host.clone();
        NodeLink::new(
            host.clone(),
            Arc::new(move || ssh_argv(&h, &c.hub_key_file, &c.remote_node_command)),
            Some(ssh_env()),
            Arc::clone(bus),
            LinkTiming::default(),
            Some(cfg.remote_node_command.clone()),
            vec![cfg.hub_key_file.to_string_lossy().into_owned()],
        )
    };
    link.set_on_change(Some(notify));
    Arc::new(link)
}

impl FleetInner {
    fn link(&self, host_id: &str) -> Result<Arc<dyn LinkLike>, Problem> {
        self.links
            .iter()
            .find(|(id, _)| id == host_id)
            .map(|(_, l)| Arc::clone(l))
            .ok_or_else(|| {
                let shown: String = host_id.chars().take(40).collect();
                not_found(format!(
                    "no host {} is configured",
                    ec_proto::pyfmt::repr_str(&shown)
                ))
                .code("unknown_host")
            })
    }

    fn data(&self, host_id: &str) -> Result<HostData, Problem> {
        let link = self.link(host_id)?;
        Ok(HostData {
            host: link.host().clone(),
            link: link.link(),
            hello: link.latest("hello"),
            sample: link.latest("sample"),
            disks: link.latest("disks"),
            inventory: link.latest("inventory"),
        })
    }

    fn resolver(&self) -> Option<InstanceResolver> {
        self.resolver.read_guard().clone()
    }

    fn counts(&self) -> Option<CountsProvider> {
        self.counts.read_guard().clone()
    }

    fn summary(&self, host_id: &str) -> Result<HostSummary, Problem> {
        let d = self.data(host_id)?;
        Ok(fleet_status::build_summary(
            &d,
            self.resolver().as_ref(),
            self.counts().as_ref(),
        ))
    }

    fn publish(&self, host_id: &str) {
        let Ok(s) = self.summary(host_id) else {
            tracing::error!(target: LOG, "could not build the summary of host {host_id}");
            return;
        };
        self.seen.guard().insert(
            host_id.to_string(),
            (s.link.state.clone(), s.link.error.clone()),
        );
        self.bus
            .publish("fleet", serde_json::to_value(&s).unwrap_or(Value::Null));
    }

    fn now(&self) -> f64 {
        (*self.clock.read_guard())()
    }

    /// Local images are not part of a node's pushed inventory: ask (at most once a minute per host).
    async fn images_of(&self, host_id: &str) -> Result<Vec<ImageInfo>, Problem> {
        let link = self.link(host_id)?;
        let cached =
            |s: &Self| -> Option<(f64, Vec<ImageInfo>)> { s.images.guard().get(host_id).cloned() };
        if let Some((exp, list)) = cached(self)
            && self.now() < exp
        {
            return Ok(list);
        }
        let facts = fleet_status::docker_facts(link.latest("hello").as_deref());
        let state = link.link().state;
        if !(state == "up" || state == "stale") || !facts.is_some_and(|f| f.available) {
            return Ok(cached(self).map(|c| c.1).unwrap_or_default());
        }
        let lock = {
            let mut locks = self.image_locks.guard();
            Arc::clone(locks.entry(host_id.to_string()).or_default())
        };
        let _held = lock.lock().await;
        let old = cached(self);
        if let Some((exp, list)) = &old
            && self.now() < *exp
        {
            return Ok(list.clone());
        }
        let images = match link
            .call("docker.images", None, Duration::from_secs(5))
            .await
        {
            Ok(r) => {
                let images = fleet_status::images_from_result(&r);
                let until = self.now() + IMAGES_TTL_S;
                self.images
                    .guard()
                    .insert(host_id.to_string(), (until, images.clone()));
                images
            }
            Err(_) => {
                let old = old.map(|c| c.1).unwrap_or_default();
                // do not hammer a node that cannot answer
                let until = self.now() + 10.0;
                self.images
                    .guard()
                    .insert(host_id.to_string(), (until, old.clone()));
                old
            }
        };
        Ok(images)
    }

    async fn watch(self: Arc<Self>) {
        loop {
            let every = *self.watch_s.read_guard();
            tokio::time::sleep(secs(every)).await;
            // one round that panics (a resolver or counts provider bug) is logged; the watcher goes on
            crate::panics::guarded_sync("fleet watcher", || {
                for (hid, link) in &self.links {
                    let st = link.link();
                    let seen = self.seen.guard().get(hid).cloned();
                    if seen != Some((st.state.clone(), st.error.clone())) {
                        self.publish(hid);
                    }
                }
            });
        }
    }
}

impl FleetPort for FleetService {
    fn hosts(&self) -> Vec<HostConfig> {
        self.inner.hosts.clone()
    }

    fn node(&self, host_id: &str) -> Result<Arc<dyn NodeClient>, Problem> {
        let link = self.inner.link(host_id)?;
        Ok(link as Arc<dyn NodeClient>)
    }
}

#[async_trait]
impl Fleet for FleetService {
    async fn start(&self) {
        for (_, link) in &self.inner.links {
            link.start().await;
        }
        let mut w = self.inner.watcher.guard();
        if w.is_none() {
            *w = Some(tokio::spawn(Arc::clone(&self.inner).watch()));
        }
    }

    async fn stop(&self) {
        let watcher = self.inner.watcher.guard().take();
        if let Some(w) = watcher {
            w.abort();
            let _ = w.await;
        }
        futures::future::join_all(self.inner.links.iter().map(|(_, l)| l.stop())).await;
    }

    fn summaries(&self) -> AppResult<Vec<HostSummary>> {
        self.inner
            .hosts
            .iter()
            .map(|h| self.inner.summary(&h.id).map_err(Into::into))
            .collect()
    }

    fn summary(&self, host_id: &str) -> AppResult<HostSummary> {
        Ok(self.inner.summary(host_id)?)
    }

    async fn detail(&self, host_id: &str) -> AppResult<HostDetail> {
        self.inner.link(host_id)?;
        let images = self.inner.images_of(host_id).await?;
        let fleet: Vec<HostData> = self
            .inner
            .hosts
            .iter()
            .map(|h| self.inner.data(&h.id))
            .collect::<Result<_, _>>()?;
        let resolver = self.inner.resolver();
        let counts = self.inner.counts();
        fleet_status::build_detail(host_id, &fleet, images, resolver.as_ref(), counts.as_ref())
            .ok_or_else(|| {
                not_found(format!(
                    "no host {} is configured",
                    ec_proto::pyfmt::repr_str(host_id)
                ))
                .code("unknown_host")
                .into()
            })
    }

    fn reconnect(&self, host_id: &str) -> AppResult<HostSummary> {
        let link = self.inner.link(host_id)?;
        if link.link().state == "disabled" {
            return Err(conflict(format!("{} is disabled", link.host().label))
                .code("host_disabled")
                .into());
        }
        link.reconnect();
        Ok(self.inner.summary(host_id)?)
    }

    fn set_instance_resolver(&self, f: InstanceResolver) {
        *self.inner.resolver.write_guard() = Some(f);
    }

    fn set_counts_provider(&self, f: CountsProvider) {
        *self.inner.counts.write_guard() = Some(f);
    }
}

impl FleetService {
    /// `set_instance_resolver(None)` (Python tests unplug a provider).
    pub fn clear_providers(&self) {
        *self.inner.resolver.write_guard() = None;
        *self.inner.counts.write_guard() = None;
    }

    pub fn settings(&self) -> Arc<Settings> {
        Arc::clone(&self.inner.cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_makes_untrusted_text_safe() {
        assert_eq!(
            text("disk\x07 full\nsecond line hf_abcdefghijklmnop", 1000),
            "disk full second line [redacted]"
        );
        assert_eq!(text(&"x".repeat(500), 300).len(), 300);
    }

    #[test]
    fn test_code_passes_well_formed_codes_only() {
        assert_eq!(code(Some(&json!("not_found"))), "not_found");
        for bad in ["BAD CODE!", "../../etc", "Not_Snake", "9lives"] {
            assert_eq!(code(Some(&json!(bad))), "internal");
        }
        assert_eq!(code(Some(&json!("x".repeat(60)))), "internal");
        assert_eq!(code(None), "internal");
    }

    #[test]
    fn test_decode_json_py_names_non_finite_numbers() {
        assert_eq!(
            decode_json_py(br#"{"type":"sample","seq":1,"t":NaN}"#).err(),
            Some("non-finite number NaN in message".into())
        );
        assert_eq!(
            decode_json_py(br#"{"t": 1e999}"#).err(),
            Some("non-finite number 1e999 in message".into())
        );
        assert!(
            decode_json_py(b"{not json at all")
                .err()
                .is_some_and(|e| e.starts_with("not JSON"))
        );
    }
}
