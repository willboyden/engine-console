//! util.py's subprocess half: argv lists only (never a shell), always a timeout, stdin from /dev/null,
//! and an environment built from scratch (`child_env`) so nothing in the node's own environment steers a child.
//!
//! The node's own environment is passed in (`EnvMap`, a snapshot taken at start: `node_env()`), never read here,
//! so tests give each node a hermetic environment without touching the process's.
//!
//! Hardening beyond Python (PORTING.md deviations, node hardening): tools are found in a FIXED PATH, never the
//! caller's (`tool_path`: DEFAULT_PATH, or in debug builds `ECNODE_TOOL_PATH` for tests and development), resolved
//! here to an absolute path before the exec (`resolve_program`), and every child gets that same PATH. A child's
//! output is capped WHILE it is read (`run` keeps at most RUN_MAX_OUT / RUN_MAX_ERR bytes and drains the rest;
//! streaming readers cap each line), so a flooding child neither blocks on a full pipe nor grows the node's memory.

use std::collections::{BTreeMap, HashSet};
use std::io::{BufRead, BufReader, Read};
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use super::py;
use super::sync::lock as guard;

pub type EnvMap = BTreeMap<String, String>;

/// `os.environ` (names and values that are not UTF-8 are left out).
pub fn node_env() -> EnvMap {
    std::env::vars_os()
        .filter_map(|(k, v)| Some((k.into_string().ok()?, v.into_string().ok()?)))
        .collect()
}

/// Copied from the node's environment when set; nothing else is (util.py CHILD_ENV_KEEP).
pub const CHILD_ENV_KEEP: &[&str] = &["HOME", "XDG_RUNTIME_DIR", "DOCKER_HOST", "DOCKER_CONTEXT"];
pub const DEFAULT_PATH: &str = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin";
pub const CHILD_LANG: &str = "C.UTF-8";
/// The one override of the fixed tool PATH, for tests and development (fake `docker`, `nvidia-smi`, `ip`): a
/// colon-separated list of absolute directories, read from the node's own environment. Honoured only by DEBUG builds
/// (`TOOL_PATH_OVERRIDE`, security audit F10): a release node always uses DEFAULT_PATH, whatever its environment
/// says; a machine whose docker CLI lives outside DEFAULT_PATH names it in the policy (`docker.binary`, absolute).
pub const TOOL_PATH_ENV: &str = "ECNODE_TOOL_PATH";
/// Whether this build reads TOOL_PATH_ENV at all: debug builds (`cargo test`, `cargo build`, the parity and e2e
/// harnesses) do; release builds (`make release`, the published node binaries) compile the override out.
pub const TOOL_PATH_OVERRIDE: bool = cfg!(debug_assertions);
/// Bytes of a short command's stdout the node keeps (256 `docker inspect` documents fit many times over).
pub const RUN_MAX_OUT: usize = 32 * 1024 * 1024;
/// Bytes of a short command's stderr the node keeps (error texts; only their first line is ever shown).
pub const RUN_MAX_ERR: usize = 1024 * 1024;

/// PATH with absolute entries only (an empty or relative entry searches the current directory), deduplicated,
/// else the system default.
pub fn safe_path(value: Option<&str>) -> String {
    let mut seen = HashSet::new();
    let entries: Vec<&str> = value
        .unwrap_or("")
        .split(':')
        .filter(|p| p.starts_with('/') && seen.insert(*p))
        .collect();
    if entries.is_empty() {
        DEFAULT_PATH.to_string()
    } else {
        entries.join(":")
    }
}

/// The PATH the node finds its tools in (docker, nvidia-smi, ip, uname, date) and the PATH every child gets:
/// DEFAULT_PATH, or (debug builds only) the absolute entries of `ECNODE_TOOL_PATH` when the node's environment
/// `base` sets it. The node's own PATH is never consulted (Python's node uses it: PORTING.md deviation, node
/// hardening).
pub fn tool_path(base: &EnvMap) -> String {
    tool_path_with(base, TOOL_PATH_OVERRIDE)
}

/// tool_path with the override switch explicit (`honour_override`: whether TOOL_PATH_ENV is read at all).
pub fn tool_path_with(base: &EnvMap, honour_override: bool) -> String {
    match base
        .get(TOOL_PATH_ENV)
        .filter(|v| honour_override && !v.is_empty())
    {
        Some(v) => safe_path(Some(v)),
        None => DEFAULT_PATH.to_string(),
    }
}

/// A child's whole environment: CHILD_ENV_KEEP from `base`, PATH (`tool_path`), LANG, then `extra` (docker.run's
/// secret values). An `extra` name never replaces one of the others.
pub fn child_env(extra: Option<&EnvMap>, base: &EnvMap) -> EnvMap {
    let mut env: EnvMap = extra.cloned().unwrap_or_default();
    for k in CHILD_ENV_KEEP {
        if let Some(v) = base.get(*k).filter(|v| !v.is_empty()) {
            env.insert((*k).to_string(), v.clone());
        }
    }
    env.insert("PATH".into(), tool_path(base));
    env.insert("LANG".into(), CHILD_LANG.into());
    env
}

/// Where `program` lives: itself when it is an absolute path, else the first executable regular file of that name
/// in the absolute entries of `path` (execvp's search, done here so that the exec itself never searches: the kernel
/// gets an absolute path). A relative path with a '/' depends on the current directory and is refused (not found).
/// Not found: ENOENT; found but not executable: EACCES (as execvp reports them).
pub fn resolve_program(program: &str, path: &str) -> std::io::Result<PathBuf> {
    let not_found = || std::io::Error::from_raw_os_error(nix::libc::ENOENT);
    if program.contains('/') {
        return if program.starts_with('/') {
            Ok(PathBuf::from(program))
        } else {
            Err(not_found())
        };
    }
    if program.is_empty() || program.contains('\0') {
        return Err(not_found());
    }
    let mut denied = false;
    for dir in path.split(':').filter(|d| d.starts_with('/')) {
        let cand = Path::new(dir).join(program);
        if std::fs::metadata(&cand).is_ok_and(|m| m.is_file()) {
            if nix::unistd::access(&cand, nix::unistd::AccessFlags::X_OK).is_ok() {
                return Ok(cand);
            }
            denied = true;
        }
    }
    Err(std::io::Error::from_raw_os_error(if denied {
        nix::libc::EACCES
    } else {
        nix::libc::ENOENT
    }))
}

/// A `Command` for argv[0] resolved in the PATH of the child's own environment `env` (built by `child_env` or
/// `script_env`: the fixed tool PATH; DEFAULT_PATH when `env` has none), stdin /dev/null, the environment exactly
/// `env`. The child still sees argv[0] as it was given, as after Python's PATH lookup.
fn command_for(argv0: &str, env: &EnvMap) -> std::io::Result<Command> {
    let path = safe_path(env.get("PATH").map(String::as_str));
    let program = resolve_program(argv0, &path)?;
    let mut cmd = Command::new(program);
    cmd.arg0(argv0).env_clear().envs(env).stdin(Stdio::null());
    Ok(cmd)
}

/// rc 124: timed out, 126: could not start, 127: not found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CmdResult {
    pub rc: i32,
    pub out: String,
    pub err: String,
    /// stdout went past its cap or stderr past its cap: `out` / `err` hold only their first bytes
    pub truncated: bool,
}

impl CmdResult {
    pub fn ok(&self) -> bool {
        self.rc == 0
    }

    pub fn new(rc: i32, out: impl Into<String>, err: impl Into<String>) -> Self {
        Self {
            rc,
            out: out.into(),
            err: err.into(),
            truncated: false,
        }
    }
}

/// util.py `Runner`: how the node runs a short command. `env` is the child's whole environment; None means
/// `child_env(None, <the node's environment>)`.
pub trait Runner: Send + Sync {
    fn run(&self, argv: &[String], timeout: f64, env: Option<&EnvMap>) -> CmdResult;
}

impl<F> Runner for F
where
    F: Fn(&[String], f64, Option<&EnvMap>) -> CmdResult + Send + Sync,
{
    fn run(&self, argv: &[String], timeout: f64, env: Option<&EnvMap>) -> CmdResult {
        self(argv, timeout, env)
    }
}

/// The real runner over a node environment.
#[derive(Debug, Clone, Default)]
pub struct ProcRunner {
    pub base: EnvMap,
}

impl ProcRunner {
    pub fn new(base: EnvMap) -> Self {
        Self { base }
    }
}

impl Runner for ProcRunner {
    fn run(&self, argv: &[String], timeout: f64, env: Option<&EnvMap>) -> CmdResult {
        let env = env.cloned().unwrap_or_else(|| child_env(None, &self.base));
        run(argv, timeout, &env)
    }
}

/// Python's returncode: the exit status, or minus the signal that ended the process.
pub fn rc_of(st: ExitStatus) -> i32 {
    st.code().unwrap_or_else(|| -st.signal().unwrap_or(0))
}

fn start_error(argv0: &str, e: &std::io::Error) -> (i32, String) {
    if e.kind() == std::io::ErrorKind::NotFound {
        (127, format!("{argv0}: command not found"))
    } else {
        (126, format!("{argv0}: {}", py::io_strerror(e)))
    }
}

/// Read `r` to EOF keeping at most `cap` bytes; everything after that is read and dropped, so the writer never
/// blocks on a full pipe and memory stays bounded. (kept bytes, whether anything was dropped)
pub fn drain_capped(r: &mut impl Read, cap: usize) -> (Vec<u8>, bool) {
    let mut kept = Vec::new();
    let mut truncated = false;
    let mut chunk = vec![0u8; 65536];
    loop {
        match r.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => {
                let room = cap.saturating_sub(kept.len());
                if n > room {
                    truncated = true;
                }
                kept.extend_from_slice(&chunk[..n.min(room)]);
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(_) => break,
        }
    }
    (kept, truncated)
}

fn reader_thread(mut r: impl Read + Send + 'static, cap: usize) -> mpsc::Receiver<(Vec<u8>, bool)> {
    let (tx, rx) = mpsc::channel();
    let _ = thread::Builder::new()
        .name("ecnode-pipe".into())
        .spawn(move || {
            let got = super::panics::contain("pipe reader", || drain_capped(&mut r, cap))
                .unwrap_or_default();
            let _ = tx.send(got);
        });
    rx
}

/// util.py `run`: run argv with a timeout; never fails for a missing binary or a timeout (see CmdResult.rc).
/// stdout is capped at RUN_MAX_OUT and stderr at RUN_MAX_ERR bytes while they are read (`run_capped`).
pub fn run(argv: &[String], timeout: f64, env: &EnvMap) -> CmdResult {
    run_capped(argv, timeout, env, RUN_MAX_OUT, RUN_MAX_ERR)
}

/// `run` with explicit caps: at most `max_out` bytes of stdout and `max_err` of stderr are kept; the rest is
/// drained and dropped while the child runs (`CmdResult::truncated` says so), and the timeout still holds.
pub fn run_capped(
    argv: &[String],
    timeout: f64,
    env: &EnvMap,
    max_out: usize,
    max_err: usize,
) -> CmdResult {
    let Some(argv0) = argv.first() else {
        return CmdResult::new(126, "", ": no command");
    };
    let spawned = command_for(argv0, env).and_then(|mut cmd| {
        cmd.args(&argv[1..])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
    });
    let mut child = match spawned {
        Ok(c) => c,
        Err(e) => {
            let (rc, msg) = start_error(argv0, &e);
            return CmdResult::new(rc, "", msg);
        }
    };
    let out_rx = child.stdout.take().map(|o| reader_thread(o, max_out));
    let err_rx = child.stderr.take().map(|e| reader_thread(e, max_err));
    let deadline = Instant::now() + Duration::from_secs_f64(timeout.max(0.0));
    let timed_out = || {
        CmdResult::new(
            124,
            "",
            format!(
                "{argv0} did not finish within {} s",
                ec_proto::pyfmt::fmt_g(timeout)
            ),
        )
    };
    // communicate(): both pipes to EOF, then the exit status, all within the timeout
    let collect = |rx: Option<mpsc::Receiver<(Vec<u8>, bool)>>| -> Option<(Vec<u8>, bool)> {
        match rx {
            None => Some((Vec::new(), false)),
            Some(rx) => rx
                .recv_timeout(deadline.saturating_duration_since(Instant::now()))
                .ok(),
        }
    };
    let out = collect(out_rx);
    let err = out.as_ref().and_then(|_| collect(err_rx));
    let (Some((out, out_cut)), Some((err, err_cut))) = (out, err) else {
        let _ = child.kill();
        let _ = child.wait();
        return timed_out();
    };
    loop {
        match child.try_wait() {
            Ok(Some(st)) => {
                return CmdResult {
                    rc: rc_of(st),
                    out: String::from_utf8_lossy(&out).into_owned(),
                    err: String::from_utf8_lossy(&err).into_owned(),
                    truncated: out_cut || err_cut,
                };
            }
            Ok(None) if Instant::now() < deadline => thread::sleep(Duration::from_millis(5)),
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
                return timed_out();
            }
            Err(e) => return CmdResult::new(126, "", format!("{argv0}: {}", py::io_strerror(&e))),
        }
    }
}

/// A streaming child in its own process group (util.py `spawn`), shareable between the thread reading it and
/// a cancel hook.
#[derive(Debug)]
pub struct Proc {
    pid: i32,
    child: Mutex<Child>,
    status: Mutex<Option<i32>>,
}

/// What `spawn` returns: the process and its output pipe(s).
pub struct Spawned {
    pub proc: Arc<Proc>,
    /// stdout (stdout and stderr together when merged)
    pub stdout: Box<dyn Read + Send>,
    pub stderr: Option<Box<dyn Read + Send>>,
}

impl Proc {
    pub fn pid(&self) -> i32 {
        self.pid
    }

    /// `proc.poll()`: the return code once it exited.
    pub fn poll(&self) -> Option<i32> {
        let mut status = guard(&self.status);
        if status.is_none()
            && let Ok(Some(st)) = guard(&self.child).try_wait()
        {
            *status = Some(rc_of(st));
        }
        *status
    }

    /// `proc.wait(timeout)`: None when it is still running at the deadline.
    pub fn wait_timeout(&self, d: Duration) -> Option<i32> {
        let deadline = Instant::now() + d;
        loop {
            if let Some(rc) = self.poll() {
                return Some(rc);
            }
            if Instant::now() >= deadline {
                return None;
            }
            thread::sleep(Duration::from_millis(10));
        }
    }

    /// `proc.wait()`
    pub fn wait(&self) -> i32 {
        loop {
            if let Some(rc) = self.wait_timeout(Duration::from_secs(3600)) {
                return rc;
            }
        }
    }
}

/// util.py `spawn`: stdin /dev/null, stdout piped (stderr into it when `merge_stderr`), `env` the whole
/// environment (argv[0] resolved in its PATH: `command_for`), a new process group (kill_group reaches everything it
/// started). Callers read the output with a bounded reader (`iter_lines`, `run_tail`).
pub fn spawn(
    argv: &[String],
    env: &EnvMap,
    merge_stderr: bool,
    cwd: Option<&str>,
) -> std::io::Result<Spawned> {
    let argv0 = argv
        .first()
        .ok_or_else(|| std::io::Error::from_raw_os_error(nix::libc::ENOENT))?;
    let mut cmd = command_for(argv0, env)?;
    cmd.args(&argv[1..]).process_group(0);
    if let Some(d) = cwd {
        cmd.current_dir(d);
    }
    let merged = if merge_stderr {
        let (r, w) = std::io::pipe()?;
        cmd.stdout(w.try_clone()?).stderr(w);
        Some(r)
    } else {
        cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
        None
    };
    let mut child = cmd.spawn()?;
    drop(cmd); // our copies of the pipe's write end: the reader sees EOF once the child's are gone
    let pid = i32::try_from(child.id()).unwrap_or(0);
    let (stdout, stderr): (Box<dyn Read + Send>, Option<Box<dyn Read + Send>>) = match merged {
        Some(r) => (Box::new(r), None),
        None => {
            let out = child.stdout.take().map_or_else(
                || Box::new(std::io::empty()) as Box<dyn Read + Send>,
                |o| Box::new(o) as _,
            );
            let err = child
                .stderr
                .take()
                .map(|e| Box::new(e) as Box<dyn Read + Send>);
            (out, err)
        }
    };
    Ok(Spawned {
        proc: Arc::new(Proc {
            pid,
            child: Mutex::new(child),
            status: Mutex::new(None),
        }),
        stdout,
        stderr,
    })
}

fn killpg(pid: i32, sig: nix::sys::signal::Signal) {
    if pid > 0 {
        let _ = nix::sys::signal::killpg(nix::unistd::Pid::from_raw(pid), sig);
    }
}

/// util.py `kill_group`: SIGTERM the child's process group, SIGKILL it if it is still there after `grace`.
pub fn kill_group(proc: &Proc, grace: Duration) {
    if proc.poll().is_some() {
        return;
    }
    killpg(proc.pid, nix::sys::signal::Signal::SIGTERM);
    if proc.wait_timeout(grace).is_none() {
        killpg(proc.pid, nix::sys::signal::Signal::SIGKILL);
        let _ = proc.wait_timeout(grace);
    }
}

/// A one-shot timer that can be cancelled (threading.Timer). A panic in `f` is contained (reported on stderr).
pub struct Timer {
    cancel: Option<mpsc::Sender<()>>,
}

impl Timer {
    pub fn start(after: Duration, f: impl FnOnce() + Send + 'static) -> Self {
        let (tx, rx) = mpsc::channel::<()>();
        let _ = thread::Builder::new()
            .name("ecnode-timer".into())
            .spawn(move || {
                if let Err(mpsc::RecvTimeoutError::Timeout) = rx.recv_timeout(after) {
                    let _ = super::panics::contain("timer", f);
                }
            });
        Self { cancel: Some(tx) }
    }

    pub fn cancel(&mut self) {
        if let Some(tx) = self.cancel.take() {
            let _ = tx.send(());
        }
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        self.cancel();
    }
}

/// util.py `run_tail`: (rc, the last `max_bytes` of stdout+stderr, truncated). Memory stays bounded however much
/// the child prints.
pub fn run_tail(
    argv: &[String],
    timeout: f64,
    max_bytes: usize,
    env: &EnvMap,
) -> (i32, Vec<u8>, bool) {
    let argv0 = argv.first().cloned().unwrap_or_default();
    let sp = match spawn(argv, env, true, None) {
        Ok(s) => s,
        Err(e) => {
            let (rc, msg) = start_error(&argv0, &e);
            return (rc, msg.into_bytes(), false);
        }
    };
    let proc = Arc::clone(&sp.proc);
    let mut timer = {
        let p = Arc::clone(&proc);
        Timer::start(Duration::from_secs_f64(timeout.max(0.0)), move || {
            kill_group(&p, Duration::from_secs(2));
        })
    };
    let mut out = sp.stdout;
    let mut buf: Vec<u8> = Vec::new();
    let mut truncated = false;
    let mut chunk = vec![0u8; 65536];
    loop {
        match out.read(&mut chunk) {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                buf.extend_from_slice(&chunk[..n]);
                if buf.len() > max_bytes {
                    let cut = buf.len() - max_bytes;
                    buf.drain(..cut);
                    truncated = true;
                }
            }
        }
    }
    let rc = proc.wait();
    timer.cancel();
    kill_group(&proc, Duration::from_secs(1));
    (rc, buf, truncated)
}

/// util.py `iter_lines`: lines of a child's output (newline stripped, capped at `limit` characters, at most
/// `LINE_READ_MAX` bytes read per line, a longer line coming out in pieces as Python's `readline(1 MiB)` does);
/// stops at EOF. Lines are yielded one at a time, never accumulated, so a long-running stream stays bounded.
pub struct Lines<R: Read> {
    r: BufReader<R>,
    limit: usize,
}

/// Bytes `iter_lines` reads for one line at most.
pub const LINE_READ_MAX: usize = 1024 * 1024;

pub fn iter_lines<R: Read>(r: R, limit: usize) -> Lines<R> {
    Lines {
        r: BufReader::with_capacity(65536, r),
        limit,
    }
}

impl<R: Read> Iterator for Lines<R> {
    type Item = String;

    fn next(&mut self) -> Option<String> {
        let mut line: Vec<u8> = Vec::new();
        loop {
            let avail = match self.r.fill_buf() {
                Ok(b) => b,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(_) => &[],
            };
            if avail.is_empty() {
                break;
            }
            let room = LINE_READ_MAX - line.len();
            let window = &avail[..avail.len().min(room)];
            if let Some(i) = window.iter().position(|c| *c == b'\n') {
                line.extend_from_slice(&window[..=i]);
                self.r.consume(i + 1);
                break;
            }
            let n = window.len();
            line.extend_from_slice(window);
            self.r.consume(n);
            if line.len() >= LINE_READ_MAX {
                break;
            }
        }
        if line.is_empty() {
            return None;
        }
        let text = String::from_utf8_lossy(&line);
        let text = py::rstrip_chars(&text, "\r\n");
        Some(py::take(text, self.limit).to_string())
    }
}

/// util.py `Guarded`: a blocking filesystem call (statvfs, realpath on a network mount) with a timeout. A call
/// that does not return is abandoned (its thread stays blocked) and its key is skipped until it does return.
#[derive(Debug, Clone, Default)]
pub struct Guarded {
    busy: Arc<Mutex<HashSet<String>>>,
}

impl Guarded {
    pub fn new() -> Self {
        Self::default()
    }

    /// None when the key is still busy, the call did not finish in time, or it failed (panicked).
    pub fn call<T: Send + 'static>(
        &self,
        key: &str,
        f: impl FnOnce() -> T + Send + 'static,
        timeout: Duration,
    ) -> Option<T> {
        {
            let mut busy = guard(&self.busy);
            if !busy.insert(key.to_string()) {
                return None;
            }
        }
        let (tx, rx) = mpsc::channel();
        let busy = Arc::clone(&self.busy);
        let k = key.to_string();
        let started = thread::Builder::new()
            .name("ecnode-guarded".into())
            .spawn(move || {
                let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).ok();
                guard(&busy).remove(&k);
                let _ = tx.send(res);
            });
        if started.is_err() {
            guard(&self.busy).remove(key);
            return None;
        }
        rx.recv_timeout(timeout).ok().flatten()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safe_path_keeps_absolute_entries_once() {
        assert_eq!(
            safe_path(Some(":relative/bin:/usr/bin:/usr/bin:/opt/tools/bin")),
            "/usr/bin:/opt/tools/bin"
        );
        assert_eq!(safe_path(None), DEFAULT_PATH);
        assert_eq!(safe_path(Some("relative:")), DEFAULT_PATH);
    }

    #[test]
    fn tool_path_ignores_the_callers_path() {
        let mut base = EnvMap::new();
        base.insert("PATH".into(), "/opt/evil/bin:/usr/bin".into());
        assert_eq!(tool_path_with(&base, true), DEFAULT_PATH);
        base.insert(
            TOOL_PATH_ENV.into(),
            "rel:/opt/fakes:/opt/fakes:/usr/bin".into(),
        );
        assert_eq!(tool_path_with(&base, true), "/opt/fakes:/usr/bin");
        base.insert(TOOL_PATH_ENV.into(), "only/relative".into());
        assert_eq!(tool_path_with(&base, true), DEFAULT_PATH);
        assert_eq!(child_env(None, &base)["PATH"], DEFAULT_PATH);
    }

    #[test]
    fn test_audit_f10_release_builds_ignore_the_tool_path_override() {
        // security audit F10: the override exists in debug builds only (every test and harness runs one)
        assert_eq!(TOOL_PATH_OVERRIDE, cfg!(debug_assertions));
        let mut base = EnvMap::new();
        base.insert(TOOL_PATH_ENV.into(), "/opt/fakes:/usr/bin".into());
        // with the override compiled out, the variable changes nothing: not the tools, not a child's PATH
        assert_eq!(tool_path_with(&base, false), DEFAULT_PATH);
        if cfg!(debug_assertions) {
            assert_eq!(tool_path(&base), "/opt/fakes:/usr/bin");
            assert_eq!(child_env(None, &base)["PATH"], "/opt/fakes:/usr/bin");
        } else {
            // `cargo test --release`: what a published node does
            assert_eq!(tool_path(&base), DEFAULT_PATH);
            assert_eq!(child_env(None, &base)["PATH"], DEFAULT_PATH);
        }
    }

    #[test]
    fn resolve_program_searches_only_the_given_path() {
        let d = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        let exe = d.path().join("tool");
        std::fs::write(&exe, "#!/bin/sh\n").unwrap_or_else(|e| panic!("{e}"));
        let plain = d.path().join("plain");
        std::fs::write(&plain, "x").unwrap_or_else(|e| panic!("{e}"));
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&exe, std::fs::Permissions::from_mode(0o755))
            .unwrap_or_else(|e| panic!("{e}"));
        std::fs::set_permissions(&plain, std::fs::Permissions::from_mode(0o644))
            .unwrap_or_else(|e| panic!("{e}"));
        let dir = d.path().to_string_lossy().into_owned();
        assert_eq!(resolve_program("tool", &dir).ok(), Some(exe));
        let kind = |r: std::io::Result<PathBuf>| r.err().and_then(|e| e.raw_os_error());
        assert_eq!(
            kind(resolve_program("tool", "/nonexistent")),
            Some(nix::libc::ENOENT)
        );
        assert_eq!(
            kind(resolve_program("plain", &dir)),
            Some(nix::libc::EACCES)
        );
        assert_eq!(
            kind(resolve_program("./tool", &dir)),
            Some(nix::libc::ENOENT)
        );
        assert_eq!(kind(resolve_program("", &dir)), Some(nix::libc::ENOENT));
        assert_eq!(
            resolve_program("/abs/anything", "").ok(),
            Some(PathBuf::from("/abs/anything"))
        );
        // a relative PATH entry is never searched
        assert_eq!(
            kind(resolve_program("tool", "relative")),
            Some(nix::libc::ENOENT)
        );
    }
}
