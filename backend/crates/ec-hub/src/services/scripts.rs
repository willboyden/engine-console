//! Python: backend/src/engine_console/services/scripts.py (tests/core/test_scripts.py).
//!
//! Lifecycle scripts registered in a node's policy (contract v2, management class `script`).
//!
//! A lab often already starts its engines with its own script that encodes ordering rules (start A before B); the
//! console must use that script and never bypass it. The hub sees only a script's name, label and the NAMES of its
//! actions (argv stays on the node); running one is `scripts.run` on that host's node, whose own policy decides.
//!
//! Rules enforced here before anything is sent (security control, each with a named test): the script and action
//! exist in the node-reported policy, the one argument (if any) is from the action's allowlist, the node is connected
//! and in manage mode, and only one job per (host, script) runs at a time. Output lines are masked with
//! `common.scrub` before they are stored or streamed.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::Duration;

use futures::stream::BoxStream;
use regex::Regex;
use serde_json::{Value, json};
use tokio::sync::watch;

use crate::domain::errors::{
    AppError, AppResult, NodeError, Obj, bad_request, conflict, not_found,
};
use crate::domain::fleet::{FleetPort, ScriptAction, ScriptInfo, ScriptJob, StreamItem};
use crate::services::common::{new_id, scrub};
use crate::services::nodeerr::NodeCaller;
use crate::services::nodeview as nv;
use crate::sync::Recover as _;

/// ScriptJob.output_tail: the last ~20 kB
pub const TAIL_CHARS: usize = 20_000;
/// lines kept for late SSE subscribers
pub const MAX_LINES: usize = 2000;
/// finished jobs kept in memory
pub const MAX_JOBS: usize = 50;
pub const LINE_CAP: usize = 4000;
/// a job that outlives this is failed (the node's own per-script timeout normally ends it long before)
pub const JOB_MAX_S: f64 = 4.0 * 3600.0;

/// everything but tab and newline: a line is one line
static CONTROL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"[\x00-\x08\x0b-\x1f\x7f-\x9f\u{2028}\u{2029}]").unwrap_or_else(|_| unreachable!())
});

pub type Clock = Arc<dyn Fn() -> f64 + Send + Sync>;

#[derive(Default)]
struct Lines {
    lines: VecDeque<String>,
    /// lines dropped from the front of `lines`
    base: usize,
}

struct Job {
    info: Mutex<ScriptJob>,
    steps: Vec<(String, Option<String>)>,
    lines: Mutex<Lines>,
    /// the action running right now
    step: Mutex<String>,
    /// bumped on every new line and at the end (SSE subscribers wait on it)
    changed: watch::Sender<u64>,
    /// true once the job reached its terminal state
    done: watch::Sender<bool>,
}

impl Job {
    fn finished(&self) -> bool {
        self.info.guard().state != "running"
    }

    fn tail(&self) -> String {
        let l = self.lines.guard();
        let joined = l.lines.iter().cloned().collect::<Vec<_>>().join("\n");
        let chars: Vec<char> = joined.chars().collect();
        chars[chars.len().saturating_sub(TAIL_CHARS)..]
            .iter()
            .collect()
    }

    fn snapshot(&self) -> ScriptJob {
        let mut info = self.info.guard().clone();
        info.output_tail = self.tail();
        info
    }

    fn emit(&self, line: &str) {
        let text: String = CONTROL
            .replace_all(&scrub(line), " ")
            .chars()
            .take(LINE_CAP)
            .collect();
        {
            let mut l = self.lines.guard();
            if l.lines.len() == MAX_LINES {
                l.lines.pop_front();
                l.base += 1;
            }
            l.lines.push_back(text);
        }
        self.changed.send_modify(|v| *v += 1);
    }
}

#[derive(Default)]
struct Inner {
    /// insertion-ordered job id -> job
    jobs: Vec<(String, Arc<Job>)>,
    /// (host, script) -> job id
    running: HashMap<(String, String), String>,
}

pub struct ScriptService {
    fleet: Arc<dyn FleetPort>,
    nodes: NodeCaller,
    inner: Arc<Mutex<Inner>>,
    clock: Mutex<Clock>,
    job_max_s: Mutex<f64>,
    /// set on shutdown: every running job stops and is failed
    cancel: watch::Sender<bool>,
}

impl std::fmt::Debug for ScriptService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ScriptService")
    }
}

fn s_of(v: Option<&Value>) -> String {
    v.map_or_else(|| "None".to_string(), ec_proto::pyfmt::str_value)
}

fn truthy(v: Option<&Value>) -> bool {
    v.is_some_and(ec_proto::sanitize::truthy)
}

fn job_404(job_id: &str) -> AppError {
    not_found(format!("no such script job: {job_id}"))
        .code("unknown_job")
        .into()
}

impl ScriptService {
    pub fn new(fleet: Arc<dyn FleetPort>) -> Self {
        let nodes = NodeCaller::new(Arc::clone(&fleet));
        Self {
            fleet,
            nodes,
            inner: Arc::new(Mutex::new(Inner::default())),
            clock: Mutex::new(Arc::new(crate::services::common::now)),
            job_max_s: Mutex::new(JOB_MAX_S),
            cancel: watch::Sender::new(false),
        }
    }

    /// Replace the clock (tests: Python passes `clock=`).
    pub fn set_clock(&self, clock: Clock) {
        *self.clock.guard() = clock;
    }

    /// The upper bound of one job (tests: Python monkeypatches `scripts.JOB_MAX_S`).
    pub fn set_job_max_s(&self, s: f64) {
        *self.job_max_s.guard() = s;
    }

    fn clock(&self) -> Clock {
        Arc::clone(&self.clock.guard())
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.guard()
    }

    // -- what the policies register ----------------------------------------------------------------------------
    /// The scripts of every host (or one), as their nodes reported them.
    pub fn scripts(&self, host_id: Option<&str>) -> AppResult<Vec<ScriptInfo>> {
        let mut out = Vec::new();
        for h in self.fleet.hosts() {
            if host_id.is_some_and(|x| x != h.id) {
                continue;
            }
            let node = self.fleet.node(&h.id)?;
            let connected = nv::link_up(node.as_ref());
            let inv = nv::inventory(node.as_ref());
            let mut status: Vec<(String, Obj)> = Vec::new();
            if let Some(Value::Array(a)) = inv.get("scripts") {
                for s in a.iter().filter_map(Value::as_object) {
                    let name = s_of(s.get("name"));
                    status.retain(|(n, _)| *n != name);
                    status.push((name, s.clone()));
                }
            }
            for s in nv::scripts(node.as_ref()) {
                let name = s_of(s.get("name"));
                let st = if connected {
                    status
                        .iter()
                        .find(|(n, _)| *n == name)
                        .map(|(_, x)| x.clone())
                } else {
                    None
                };
                let code = st
                    .as_ref()
                    .and_then(|x| x.get("status"))
                    .and_then(Value::as_str);
                let status = match code {
                    Some(c @ ("up" | "down" | "unknown" | "error")) => c.to_string(),
                    _ => "unknown".to_string(),
                };
                let detail = st
                    .as_ref()
                    .and_then(|x| x.get("detail"))
                    .filter(|v| ec_proto::sanitize::truthy(v))
                    .map(|v| s_of(Some(v)))
                    .or_else(|| (!connected).then(|| "this machine is not connected".to_string()));
                let actions = match s.get("actions") {
                    Some(Value::Array(a)) => a
                        .iter()
                        .filter_map(Value::as_object)
                        .filter(|a| truthy(a.get("name")))
                        .map(|a| ScriptAction {
                            name: s_of(a.get("name")),
                            label: s_of(
                                a.get("label")
                                    .filter(|v| ec_proto::sanitize::truthy(v))
                                    .or(a.get("name")),
                            ),
                            args: match a.get("args") {
                                Some(Value::Array(x)) => x.iter().map(|v| s_of(Some(v))).collect(),
                                _ => Vec::new(),
                            },
                            confirm: truthy(a.get("confirm")),
                            danger: truthy(a.get("danger")),
                        })
                        .collect(),
                    _ => Vec::new(),
                };
                out.push(ScriptInfo {
                    host_id: h.id.clone(),
                    name: name.clone(),
                    label: s_of(
                        s.get("label")
                            .filter(|v| ec_proto::sanitize::truthy(v))
                            .or(s.get("name")),
                    ),
                    engine: s
                        .get("engine")
                        .filter(|v| ec_proto::sanitize::truthy(v))
                        .map(|v| s_of(Some(v))),
                    description: s
                        .get("description")
                        .filter(|v| ec_proto::sanitize::truthy(v))
                        .map_or_else(String::new, |v| s_of(Some(v))),
                    status,
                    detail,
                    checked_at: st.as_ref().and_then(|x| nv::num(x.get("checked_at"))),
                    actions,
                    containers: match s.get("containers") {
                        Some(Value::Array(c)) => c.iter().map(|v| s_of(Some(v))).collect(),
                        _ => Vec::new(),
                    },
                });
            }
        }
        Ok(out)
    }

    pub fn script(&self, host_id: &str, name: &str) -> AppResult<ScriptInfo> {
        self.scripts(Some(host_id))?
            .into_iter()
            .find(|s| s.name == name)
            .ok_or_else(|| {
                not_found(format!(
                    "no script '{name}' is registered in the policy of host {host_id}"
                ))
                .code("unknown_script")
                .into()
            })
    }

    /// script name -> the action running now, on one host.
    pub fn busy(&self, host_id: &str) -> Vec<(String, String)> {
        let inner = self.lock();
        inner
            .running
            .iter()
            .filter(|((h, _), _)| h == host_id)
            .filter_map(|((_, n), j)| {
                inner
                    .jobs
                    .iter()
                    .find(|(id, _)| id == j)
                    .map(|(_, job)| (n.clone(), job.step.guard().clone()))
            })
            .collect()
    }

    // -- running -------------------------------------------------------------------------------------------------
    fn validate(
        &self,
        host_id: &str,
        name: &str,
        action: &str,
        arg: Option<&str>,
    ) -> AppResult<()> {
        let node = self.nodes.node(host_id)?;
        let info = self.script(host_id, name)?;
        let Some(act) = info.actions.iter().find(|a| a.name == action) else {
            let offers: Vec<&str> = info.actions.iter().map(|a| a.name.as_str()).collect();
            let offers = if offers.is_empty() {
                "none".to_string()
            } else {
                offers.join(", ")
            };
            return Err(not_found(format!(
                "script '{name}' has no action '{action}' (it offers: {offers})"
            ))
            .code("unknown_action")
            .into());
        };
        if !act.args.is_empty() && arg.is_none() {
            return Err(bad_request(format!(
                "action '{action}' needs one of: {}",
                act.args.join(", ")
            ))
            .code("arg_required")
            .into());
        }
        if let Some(a) = arg
            && !act.args.iter().any(|x| x == a)
        {
            let tail = if act.args.is_empty() {
                " (it takes none)".to_string()
            } else {
                format!(" (allowed: {})", act.args.join(", "))
            };
            return Err(bad_request(format!(
                "'{a}' is not an allowed argument for '{action}'{tail}"
            ))
            .code("invalid_arg")
            .into());
        }
        if !nv::link_up(node.as_ref()) {
            let link = node.link();
            let msg = link
                .error
                .filter(|e| !e.is_empty())
                .unwrap_or_else(|| format!("the node is {}", link.state));
            return Err(self
                .nodes
                .problem(&NodeError::new(host_id, "node_unavailable", msg))
                .into());
        }
        if nv::mode(node.as_ref()) != "manage" {
            return Err(self
                .nodes
                .problem(&NodeError::new(
                    host_id,
                    "monitor_only",
                    "this machine's node policy is in monitor mode: scripts cannot run until its operator \
                     switches the mode to manage",
                ))
                .into());
        }
        Ok(())
    }

    fn start(
        &self,
        host_id: &str,
        name: &str,
        label: &str,
        arg: Option<&str>,
        steps: Vec<(String, Option<String>)>,
    ) -> AppResult<ScriptJob> {
        let key = (host_id.to_string(), name.to_string());
        let clock = self.clock();
        let job = {
            let mut inner = self.lock();
            if let Some(running) = inner.running.get(&key) {
                return Err(conflict(format!(
                    "script '{name}' on host {host_id} is already running (job {running}); wait for it to finish"
                ))
                .code("script_busy")
                .into());
            }
            let info = ScriptJob {
                id: new_id("job_"),
                host_id: host_id.to_string(),
                name: name.to_string(),
                action: label.to_string(),
                arg: arg.map(str::to_string),
                state: "running".into(),
                rc: None,
                started_at: clock(),
                finished_at: None,
                output_tail: String::new(),
            };
            let first = steps.first().map(|s| s.0.clone()).unwrap_or_default();
            let job = Arc::new(Job {
                info: Mutex::new(info),
                steps,
                lines: Mutex::new(Lines::default()),
                step: Mutex::new(first),
                changed: watch::Sender::new(0),
                done: watch::Sender::new(false),
            });
            let id = job.snapshot().id;
            inner.jobs.push((id.clone(), Arc::clone(&job)));
            inner.running.insert(key, id);
            job
        };
        let out = job.snapshot();
        let run = Runner {
            nodes: self.nodes.clone(),
            inner: Arc::clone(&self.inner),
            clock,
            job_max_s: *self.job_max_s.guard(),
            cancel: self.cancel.subscribe(),
        };
        tokio::spawn(async move { run.execute(job).await });
        self.evict();
        Ok(out)
    }

    pub async fn run(
        &self,
        host_id: &str,
        name: &str,
        action: &str,
        arg: Option<&str>,
    ) -> AppResult<ScriptJob> {
        self.validate(host_id, name, action, arg)?;
        self.start(
            host_id,
            name,
            action,
            arg,
            vec![(action.to_string(), arg.map(str::to_string))],
        )
    }

    /// `down` then `up` as ONE job (one id to follow). Stops at the first failure: never starts what failed to stop.
    pub async fn restart(&self, host_id: &str, name: &str) -> AppResult<ScriptJob> {
        self.validate(host_id, name, "down", None)?;
        self.validate(host_id, name, "up", None)?;
        self.start(
            host_id,
            name,
            "restart",
            None,
            vec![("down".into(), None), ("up".into(), None)],
        )
    }

    // -- reading -------------------------------------------------------------------------------------------------
    fn find(&self, job_id: &str) -> Option<Arc<Job>> {
        self.lock()
            .jobs
            .iter()
            .find(|(id, _)| id == job_id)
            .map(|(_, j)| Arc::clone(j))
    }

    /// 404 code unknown_job "no such script job: <id>"
    pub fn job(&self, job_id: &str) -> AppResult<ScriptJob> {
        self.find(job_id)
            .map(|j| j.snapshot())
            .ok_or_else(|| job_404(job_id))
    }

    pub fn jobs(&self) -> Vec<ScriptJob> {
        let jobs: Vec<Arc<Job>> = self
            .lock()
            .jobs
            .iter()
            .map(|(_, j)| Arc::clone(j))
            .collect();
        jobs.iter().rev().map(|j| j.snapshot()).collect()
    }

    /// (event, data) pairs for SSE: every `line` (history, then live), then one `end` with the job as JSON.
    pub fn stream(
        &self,
        job_id: &str,
        once: bool,
    ) -> AppResult<BoxStream<'static, (String, String)>> {
        let job = self.find(job_id).ok_or_else(|| job_404(job_id))?;
        let s = async_stream::stream! {
            let mut rx = job.changed.subscribe();
            let mut pos: usize = 0; // absolute line number
            loop {
                rx.borrow_and_update();
                let finished = job.finished();
                let pending: Vec<String> = {
                    let l = job.lines.guard();
                    pos = pos.max(l.base);
                    l.lines.iter().skip(pos - l.base).cloned().collect()
                };
                if pending.is_empty() && !finished && !once {
                    if rx.changed().await.is_err() {
                        break;
                    }
                    continue;
                }
                pos += pending.len();
                for line in pending {
                    yield ("line".to_string(), line);
                }
                if finished || once {
                    break;
                }
            }
            yield ("end".to_string(), serde_json::to_string(&job.snapshot()).unwrap_or_default());
        };
        Ok(Box::pin(s))
    }

    /// Wait until a job reached its terminal state (tests and the API smoke tests).
    pub async fn wait(&self, job_id: &str) {
        let Some(job) = self.find(job_id) else { return };
        let mut rx = job.done.subscribe();
        let _ = rx.wait_for(|d| *d).await;
    }

    fn evict(&self) {
        let mut inner = self.lock();
        let done: Vec<String> = inner
            .jobs
            .iter()
            .filter(|(_, j)| j.finished())
            .map(|(id, _)| id.clone())
            .collect();
        let n = done.len().saturating_sub(MAX_JOBS);
        let drop_ids: Vec<String> = done.into_iter().take(n).collect();
        inner.jobs.retain(|(id, _)| !drop_ids.contains(id));
    }

    pub async fn shutdown(&self) {
        self.cancel.send_replace(true);
        let jobs: Vec<Arc<Job>> = self
            .lock()
            .jobs
            .iter()
            .map(|(_, j)| Arc::clone(j))
            .collect();
        for j in jobs {
            let mut rx = j.done.subscribe();
            let _ = tokio::time::timeout(Duration::from_secs(5), rx.wait_for(|d| *d)).await;
        }
    }
}

/// What a job's task needs (owned: the task outlives the request that started it).
struct Runner {
    nodes: NodeCaller,
    inner: Arc<Mutex<Inner>>,
    clock: Clock,
    job_max_s: f64,
    cancel: watch::Receiver<bool>,
}

enum Outcome {
    Rc(Option<i64>),
    Node(NodeError),
    Internal(String),
    Cancelled,
}

impl Runner {
    async fn execute(mut self, job: Arc<Job>) {
        let mut cancel = self.cancel.clone();
        let mut rc: Option<i64> = Some(0);
        let mut outcome = Outcome::Rc(rc);
        let many = job.steps.len() > 1;
        for (action, arg) in job.steps.clone() {
            job.step.guard().clone_from(&action);
            if many {
                job.emit(&format!("== {action} =="));
            }
            // a panic in one step ends the job like Python's `except Exception` ("script job crashed"): the job is
            // failed, its script freed, and the console goes on
            let res = tokio::select! {
                r = crate::panics::guarded("script job", self.one(&job, &action, arg.as_deref())) => {
                    r.unwrap_or_else(|| Outcome::Internal("panic".into()))
                }
                _ = cancel.wait_for(|c| *c) => Outcome::Cancelled,
            };
            match res {
                Outcome::Rc(r) => {
                    rc = r;
                    outcome = Outcome::Rc(r);
                    if r != Some(0) {
                        break;
                    }
                }
                other => {
                    outcome = other;
                    break;
                }
            }
        }
        let state;
        match outcome {
            Outcome::Rc(_) => {
                state = if rc == Some(0) { "completed" } else { "failed" };
                job.info.guard().rc = rc;
            }
            Outcome::Node(e) => {
                job.emit(&format!("[{}] {}", e.code, e.message));
                state = "failed";
            }
            Outcome::Internal(kind) => {
                tracing::error!(target: "engine_console.services.scripts", "script job crashed");
                job.emit(&format!("[internal] {kind}"));
                state = "failed";
            }
            Outcome::Cancelled => {
                job.emit("[cancelled] the console is shutting down");
                state = "failed";
            }
        }
        let tail = job.tail();
        let key = {
            let mut i = job.info.guard();
            i.state = state.into();
            i.finished_at = Some((self.clock)());
            i.output_tail = tail;
            (i.host_id.clone(), i.name.clone())
        };
        self.inner.guard().running.remove(&key);
        self.cancel = cancel;
        job.changed.send_modify(|v| *v += 1);
        job.done.send_replace(true);
    }

    async fn one(&self, job: &Arc<Job>, action: &str, arg: Option<&str>) -> Outcome {
        let (host, name) = {
            let i = job.info.guard();
            (i.host_id.clone(), i.name.clone())
        };
        let mut args = Obj::new();
        args.insert("name".into(), json!(name));
        args.insert("action".into(), json!(action));
        if let Some(a) = arg {
            args.insert("arg".into(), json!(a));
        }
        let mut stream = match self.nodes.open(&host, "scripts.run", Some(args)) {
            Ok(s) => s,
            Err(p) => return Outcome::Internal(problem_class(&p.code)),
        };
        let max = Duration::from_secs_f64(self.job_max_s.max(0.0));
        let read = async {
            while let Some(item) = stream.next().await {
                if let StreamItem::Event(ev) = item
                    && let Some(Value::String(line)) = ev.get("line")
                {
                    job.emit(line);
                    let tail = job.tail();
                    job.info.guard().output_tail = tail;
                }
            }
            stream.result().await
        };
        let res = tokio::time::timeout(max, read).await;
        match res {
            Err(_) => {
                stream.cancel().await;
                let hours = (self.job_max_s / 3600.0).floor();
                #[allow(clippy::cast_possible_truncation)]
                let h = if self.job_max_s.fract() == 0.0 {
                    (hours as i64).to_string()
                } else {
                    crate::pyjson::float_repr(hours)
                };
                Outcome::Node(NodeError::new(
                    &host,
                    "timeout",
                    format!("the script ran longer than {h} hours"),
                ))
            }
            Ok(Err(e)) => {
                stream.cancel().await;
                Outcome::Node(e)
            }
            Ok(Ok(res)) => Outcome::Rc(res.get("rc").and_then(nv::int_of)),
        }
    }
}

/// `type(e).__name__` of the problem a NodeCaller raises before a stream opens (only NotFound: an unknown host).
fn problem_class(code: &str) -> String {
    if code == "unknown_host" || code == "not_found" {
        "NotFound".into()
    } else {
        "ProblemError".into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_control_characters_are_spaces_but_tab_stays() {
        let job = Job {
            info: Mutex::new(ScriptJob {
                id: "job_x".into(),
                host_id: "h".into(),
                name: "n".into(),
                action: "up".into(),
                arg: None,
                state: "running".into(),
                rc: None,
                started_at: 0.0,
                finished_at: None,
                output_tail: String::new(),
            }),
            steps: Vec::new(),
            lines: Mutex::new(Lines::default()),
            step: Mutex::new(String::new()),
            changed: watch::Sender::new(0),
            done: watch::Sender::new(false),
        };
        job.emit("a\x1b[31mred\x0cfeed\tsep token hf_abcdefghijkl1234");
        assert_eq!(job.tail(), "a [31mred feed\tsep token [redacted]");
        for i in 0..(MAX_LINES + 5) {
            job.emit(&format!("{i}"));
        }
        let l = job.lines.guard();
        assert_eq!(l.lines.len(), MAX_LINES);
        assert_eq!(l.base, 6);
    }
}
