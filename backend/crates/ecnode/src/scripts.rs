//! Python: node/src/ecnode/scripts.py.
//!
//! scripts.run: an operator's own lifecycle script, registered in the node policy (docs/FLEET-PROTOCOL.md §6.3).
//!
//! argv comes from the policy only (the hub names a script and an action, and may pick one argument from that
//! action's allowlist), never a shell. The environment is a minimal one built from the node's own (HOME, USER,
//! LOGNAME, LANG, XDG_RUNTIME_DIR, DOCKER_HOST, DOCKER_CONTEXT) plus PATH = the node's fixed tool PATH
//! (`util::tool_path`; Python passes the node's own PATH: node hardening): nothing from the hub and nothing else of
//! the node's. One run per script at a time; a run that outlives its timeout_s, or whose request is cancelled (or
//! whose hub link drops), is stopped with its whole process group.

use std::collections::HashSet;
use std::io::Read;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use ec_proto::{BAD_ARGS, BUSY, IO_ERROR, NOT_FOUND, Obj, POLICY_DENIED, TIMEOUT};
use serde_json::{Value, json};

use crate::errors::{HandlerError, OpError, Stream};
use crate::policy::Policy;
use crate::util::proc::{Timer, kill_group};
use crate::util::{EnvMap, iter_lines, lock, py, scrub, spawn, tool_path};

pub const ENV_KEEP: &[&str] = &[
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "LANG",
    "XDG_RUNTIME_DIR",
    "DOCKER_HOST",
    "DOCKER_CONTEXT",
];
pub const OUTPUT_GRACE_S: f64 = 2.0;

/// The environment of a script (from the node's own `base`).
pub fn script_env(base: &EnvMap) -> EnvMap {
    let mut env: EnvMap = ENV_KEEP
        .iter()
        .filter_map(|k| base.get(*k).map(|v| ((*k).to_string(), v.clone())))
        .collect();
    // the fixed tool PATH, never the node's own (scripts run in HOME: an empty or relative entry would search it)
    env.insert("PATH".into(), tool_path(base));
    env
}

fn op(code: &str, msg: impl Into<String>) -> HandlerError {
    OpError::new(code, msg).into()
}

#[derive(Default)]
pub struct Scripts {
    running: Mutex<HashSet<String>>,
}

impl Scripts {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn run(
        &self,
        policy: &Policy,
        args: &Obj,
        stream: &dyn Stream,
        base: &EnvMap,
    ) -> Result<Obj, HandlerError> {
        let name_v = args.get("name").unwrap_or(&Value::Null);
        let Some(s) = name_v.as_str().and_then(|n| policy.script(n)) else {
            return Err(op(
                NOT_FOUND,
                format!(
                    "no script named {} in this node's policy",
                    py::repr_take(&py::str_of(name_v), 32)
                ),
            ));
        };
        let action_v = args.get("action").unwrap_or(&Value::Null);
        let Some(a) = action_v.as_str().and_then(|n| s.action(n)) else {
            return Err(op(
                NOT_FOUND,
                format!(
                    "script {} has no action {}",
                    py::repr(&s.name),
                    py::repr_take(&py::str_of(action_v), 32)
                ),
            ));
        };
        let arg = args
            .get("arg")
            .and_then(Value::as_str)
            .filter(|a| !a.is_empty())
            .map(str::to_string);
        if a.args.is_empty() {
            if arg.is_some() {
                return Err(op(
                    BAD_ARGS,
                    format!("{}/{} takes no argument", s.name, a.name),
                ));
            }
        } else {
            let Some(x) = &arg else {
                return Err(op(
                    BAD_ARGS,
                    format!(
                        "{}/{} needs an argument: one of {}",
                        s.name,
                        a.name,
                        a.args.join(", ")
                    ),
                ));
            };
            if !a.args.contains(x) {
                return Err(op(
                    POLICY_DENIED,
                    format!(
                        "{} is not an allowed argument of {}/{} (allowed: {})",
                        py::repr_take(x, 64),
                        s.name,
                        a.name,
                        a.args.join(", ")
                    ),
                ));
            }
        }
        let mut argv = a.argv.clone();
        if let Some(x) = arg {
            argv.push(x);
        }
        {
            let mut running = lock(&self.running);
            if !running.insert(s.name.clone()) {
                return Err(op(
                    BUSY,
                    format!(
                        "script {} is already running; wait for it to finish",
                        py::repr(&s.name)
                    ),
                ));
            }
        }
        // the slot is given back even when the run panics (the server then answers that request `internal`)
        let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_script(&argv, s.timeout_s, stream, base)
        }));
        lock(&self.running).remove(&s.name);
        out.unwrap_or_else(|p| std::panic::resume_unwind(p))
    }
}

type Line = Option<(&'static str, String)>;

fn pump(
    r: Box<dyn Read + Send>,
    name: &'static str,
    tx: &mpsc::SyncSender<Line>,
    finished: &AtomicBool,
) {
    // After the run is over, keep draining (and dropping) output: a background child the script left running may
    // still hold the pipe, and must not block on a full one. Each line is capped (iter_lines) and handed on at once.
    let _ = crate::util::panics::contain("script output", || {
        for line in iter_lines(r, 8192) {
            if !finished.load(Ordering::SeqCst) {
                let _ = tx.send(Some((name, line)));
            }
        }
    });
    if !finished.load(Ordering::SeqCst) {
        let _ = tx.send(None);
    }
}

fn run_script(
    argv: &[String],
    timeout_s: u64,
    stream: &dyn Stream,
    base: &EnvMap,
) -> Result<Obj, HandlerError> {
    let env = script_env(base);
    let cwd = env
        .get("HOME")
        .filter(|h| !h.is_empty())
        .cloned()
        .unwrap_or_else(|| "/".into());
    let sp = match spawn(argv, &env, false, Some(&cwd)) {
        Ok(sp) => sp,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(op(
                NOT_FOUND,
                format!(
                    "the script program {} does not exist on this machine",
                    py::repr(argv.first().map_or("", String::as_str))
                ),
            ));
        }
        Err(e) => {
            return Err(op(
                IO_ERROR,
                format!(
                    "cannot start {}: {}",
                    py::repr(argv.first().map_or("", String::as_str)),
                    py::io_strerror(&e)
                ),
            ));
        }
    };
    let proc = Arc::clone(&sp.proc);
    let timed_out = Arc::new(AtomicBool::new(false));
    let mut timer = {
        let (p, t) = (Arc::clone(&proc), Arc::clone(&timed_out));
        Timer::start(Duration::from_secs(timeout_s), move || {
            // an exited script waiting out its output grace is not late
            if p.poll().is_none() {
                t.store(true, Ordering::SeqCst);
                kill_group(&p, Duration::from_secs(5));
            }
        })
    };
    {
        let p = Arc::clone(&proc);
        stream.on_cancel(Box::new(move || kill_group(&p, Duration::from_secs(5))));
    }
    let (tx, rx) = mpsc::sync_channel::<Line>(4096);
    let finished = Arc::new(AtomicBool::new(false));
    let mut readers = 0;
    for (r, name) in [(Some(sp.stdout), "out"), (sp.stderr, "err")] {
        let Some(r) = r else { continue };
        let (tx, f) = (tx.clone(), Arc::clone(&finished));
        if thread::Builder::new()
            .name("ecnode-script-pipe".into())
            .spawn(move || pump(r, name, &tx, &f))
            .is_ok()
        {
            readers += 1;
        }
    }
    drop(tx);
    let mut done = 0;
    let mut exited_at: Option<Instant> = None;
    while done < readers {
        match rx.recv_timeout(Duration::from_millis(500)) {
            Ok(None) => done += 1,
            Ok(Some((name, line))) => {
                if !stream.cancelled() {
                    let mut ev = Obj::new();
                    ev.insert("stream".into(), json!(name));
                    ev.insert("line".into(), json!(scrub(&line)));
                    stream.event(ev);
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                if proc.poll().is_some() {
                    let at = *exited_at.get_or_insert_with(Instant::now);
                    if at.elapsed().as_secs_f64() > OUTPUT_GRACE_S {
                        break; // the script exited; something it started in the background holds the pipe
                    }
                }
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    let rc = proc.wait();
    finished.store(true, Ordering::SeqCst);
    timer.cancel();
    if proc.poll().is_none() {
        // only a run that is still going is stopped (timeout, cancel, link loss)
        kill_group(&proc, Duration::from_secs(2));
    }
    drop(rx);
    if timed_out.load(Ordering::SeqCst) {
        return Err(op(
            TIMEOUT,
            format!("the script ran longer than its timeout_s ({timeout_s} s) and was stopped"),
        ));
    }
    if stream.cancelled() {
        return Err(HandlerError::Cancelled);
    }
    let mut out = Obj::new();
    out.insert("rc".into(), json!(rc));
    Ok(out)
}
