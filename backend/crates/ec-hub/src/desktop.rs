//! `engine-console open` / `engine-console stop`: the desktop launcher's entry points (port of desktop.py).
//!
//! `open` starts the console in the background if nothing is serving yet, waits for /api/v1/health, then opens the
//! UI with `xdg-open`, only on a loopback address and only when every socket listening on the port belongs to the
//! current user and the answer looks like Engine Console. It never reads or passes a key. Concurrent launches are
//! serialised by `<data dir>/console.lock`. `stop` stops the console that `open` started (tracked in
//! `<data dir>/console.pid`), never one started another way: the recorded process must be this user's and its argv
//! must be exactly `[<this executable>, "--from-launcher"]` (PORTING.md D9). Failures also go to `notify-send`.
//!
//! Test seams (the Python tests monkeypatch the module): every side effect goes through [`Hooks`]
//! (`open_console_with` / `stop_console_with`), the background start through `start_background_with` (an explicit
//! spawn function and log-rotation size), the ownership check through `check_with`, the probe through `probe_with`.

use std::io::{Read, Write};
use std::net::{IpAddr, TcpStream, ToSocketAddrs};
use std::os::fd::AsRawFd;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use crate::config::{Settings, SettingsError};

pub const START_TIMEOUT_S: f64 = 60.0;
pub const STOP_TIMEOUT_S: f64 = 15.0;
pub const POLL_S: f64 = 0.5;
pub const PROBE_TIMEOUT_S: f64 = 2.0;
pub const PROBE_MAX_BYTES: usize = 64 * 1024;
pub const LOG_ROTATE_BYTES: u64 = 10 * 1024 * 1024;
pub const HEALTH_KEYS: &[&str] = &[
    "status",
    "version",
    "engines",
    "docker_context",
    "egress_mode",
];
pub const PROC_NET_TABLES: &[&str] = &["/proc/net/tcp", "/proc/net/tcp6"];
pub const LOG_NAME: &str = "console.log";
pub const PID_NAME: &str = "console.pid";
pub const LOCK_NAME: &str = "console.lock";
pub const WILDCARD_HOSTS: &[&str] = &["", "0.0.0.0", "::", "[::]"];
/// the marker argument of a console started by `open`
pub const LAUNCHER_ARG: &str = "--from-launcher";

/// A refusal the launcher reports (notification + exit status 1).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct LauncherError(pub String);

fn err(m: impl Into<String>) -> LauncherError {
    LauncherError(m.into())
}

/// The console's URL, loopback only: neither the probe nor the browser may be sent off the box.
pub fn console_url(cfg: &Settings) -> Result<String, LauncherError> {
    let mut host = cfg.host.trim().to_string();
    if host.is_empty() || host == "0.0.0.0" {
        host = "127.0.0.1".into();
    } else if host == "::" || host == "[::]" {
        host = "::1".into();
    }
    if host != "localhost" {
        let bare = host.strip_prefix('[').unwrap_or(&host);
        let bare = bare.strip_suffix(']').unwrap_or(bare);
        let ip: Option<IpAddr> = bare.parse().ok();
        match ip {
            Some(ip) if ip.is_loopback() => {
                host = if ip.is_ipv6() {
                    format!("[{ip}]")
                } else {
                    ip.to_string()
                };
            }
            _ => {
                return Err(err(format!(
                    "host {} is not a loopback address; the launcher only opens a local console",
                    ec_proto::sanitize::py_repr(&cfg.host)
                )));
            }
        }
    }
    if cfg.port == 0 {
        // Settings holds a u16, so above 65535 is refused when the setting is read; 0 is the one value left
        return Err(err(format!("port {} is out of range", cfg.port)));
    }
    Ok(format!("http://{host}:{}/", cfg.port))
}

/// The largest /proc/net/tcp* table read (~150 bytes a socket: over 400,000 sockets); a bigger one is unreadable,
/// which `held_by` treats as "not ours" (the launcher then refuses, never guesses).
pub const MAX_PROC_TABLE: usize = 64 * 1024 * 1024;

/// Owners of every LISTEN socket on `port`, any address, IPv4 and IPv6. None if a table is unreadable.
pub fn listen_uids(port: u16, tables: &[&str]) -> Option<Vec<u32>> {
    let mut uids = Vec::new();
    for table in tables {
        let text = match crate::fsutil::read_text(
            Path::new(table),
            crate::fsutil::Follow::Yes,
            MAX_PROC_TABLE,
        ) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue, // no IPv6
            Err(_) => return None,
        };
        for row in text.lines().skip(1) {
            let f: Vec<&str> = row.split_whitespace().collect();
            if f.len() > 7 && f[3] == "0A" {
                let p = f[1]
                    .rsplit_once(':')
                    .and_then(|(_, p)| u16::from_str_radix(p, 16).ok());
                if p == Some(port)
                    && let Ok(uid) = f[7].parse::<u32>()
                    && !uids.contains(&uid)
                {
                    uids.push(uid);
                }
            }
        }
    }
    Some(uids)
}

/// Every listener on the port is ours (`uids` is `listen_uids` of the port; None = unreadable = not ours).
pub fn held_by(uids: Option<&[u32]>, me: u32) -> bool {
    uids.is_some_and(|u| !u.is_empty() && u.iter().all(|x| *x == me))
}

pub fn held_by_me(port: u16) -> bool {
    held_by(
        listen_uids(port, PROC_NET_TABLES).as_deref(),
        nix::unistd::getuid().as_raw(),
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Check {
    Console,
    OtherUser,
    Foreign,
    Down,
}

/// `console` if /health answers with the console's shape, `foreign` for any other answer, `down` for none. The
/// body is capped in size and in total read time, so a listener that drips bytes cannot hold the launcher.
pub fn probe(url: &str) -> Check {
    probe_with(url, PROBE_TIMEOUT_S)
}

/// [`probe`] with an explicit time budget (the whole read, not per socket operation).
pub fn probe_with(url: &str, timeout_s: f64) -> Check {
    let budget = Duration::from_secs_f64(timeout_s.max(0.001));
    let deadline = Instant::now() + budget;
    let Some(rest) = url.strip_prefix("http://") else {
        return Check::Down;
    };
    let hostport = rest.trim_end_matches('/');
    let Ok(mut addrs) = hostport.to_socket_addrs() else {
        return Check::Down;
    };
    let Some(addr) = addrs.next() else {
        return Check::Down;
    };
    let Ok(mut s) = TcpStream::connect_timeout(&addr, budget) else {
        return Check::Down;
    };
    let _ = s.set_read_timeout(Some(budget.min(Duration::from_millis(500))));
    let req = format!(
        "GET /api/v1/health HTTP/1.1\r\nHost: {hostport}\r\nAccept: */*\r\nConnection: close\r\n\r\n"
    );
    if s.write_all(req.as_bytes()).is_err() {
        return Check::Down;
    }
    let mut raw = Vec::new();
    let mut buf = [0u8; 8192];
    let mut last = Instant::now();
    // like httpx: a stall of the whole budget (no answer, or one that stops) is down, any error is down, and bytes still
    // arriving after the deadline (a drip) or past the size cap are not the console
    loop {
        match s.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                raw.extend_from_slice(&buf[..n]);
                last = Instant::now();
                if raw.len() > PROBE_MAX_BYTES + 16 * 1024 || last > deadline {
                    return Check::Foreign;
                }
            }
            Err(e)
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                ) =>
            {
                if last.elapsed() >= budget {
                    return Check::Down;
                }
            }
            Err(_) => return Check::Down,
        }
    }
    let Some(split) = raw.windows(4).position(|w| w == b"\r\n\r\n") else {
        return Check::Down; // not an HTTP answer (httpx: a protocol error)
    };
    let head = String::from_utf8_lossy(&raw[..split]).to_lowercase();
    let mut body = raw[split + 4..].to_vec();
    if head.contains("transfer-encoding: chunked") {
        body = dechunk(&body);
    }
    if body.len() > PROBE_MAX_BYTES {
        return Check::Foreign;
    }
    match serde_json::from_slice::<serde_json::Value>(&body) {
        Ok(serde_json::Value::Object(m)) if HEALTH_KEYS.iter().all(|k| m.contains_key(*k)) => {
            Check::Console
        }
        _ => Check::Foreign,
    }
}

fn dechunk(b: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut i = 0;
    while let Some(nl) = b[i..].windows(2).position(|w| w == b"\r\n") {
        let size =
            usize::from_str_radix(String::from_utf8_lossy(&b[i..i + nl]).trim(), 16).unwrap_or(0);
        i += nl + 2;
        if size == 0 || i + size > b.len() {
            break;
        }
        out.extend_from_slice(&b[i..i + size]);
        i += size + 2;
        if i >= b.len() {
            break;
        }
    }
    out
}

/// `check`: the probe, then "every listener is ours" for a console-shaped answer.
pub fn check(url: &str, port: u16) -> Check {
    check_with(url, port, probe, |p| listen_uids(p, PROC_NET_TABLES))
}

/// [`check`] with the probe and the listener lookup given (tests).
pub fn check_with(
    url: &str,
    port: u16,
    probe: impl FnOnce(&str) -> Check,
    uids: impl FnOnce(u16) -> Option<Vec<u32>>,
) -> Check {
    let state = probe(url);
    if state == Check::Console && !held_by(uids(port).as_deref(), nix::unistd::getuid().as_raw()) {
        return Check::OtherUser; // answers like the console, but not every listener is ours
    }
    state
}

fn private_data_dir(dir: &Path) -> Result<(), LauncherError> {
    crate::services::store::create_dir_0700(dir).map_err(|e| err(e.to_string()))?;
    let st = std::fs::symlink_metadata(dir).map_err(|e| err(e.to_string()))?;
    if !st.is_dir() || st.uid() != nix::unistd::getuid().as_raw() {
        return Err(err(format!(
            "data dir {} is not a directory you own",
            dir.display()
        )));
    }
    std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))
        .map_err(|e| err(e.to_string()))
}

/// Open a regular file we own for writing (`read` adds read access, `append` appends), never through a symlink or a
/// FIFO, and force mode 0600.
fn open_private(path: &Path, read: bool, append: bool) -> Result<std::fs::File, LauncherError> {
    let f = std::fs::OpenOptions::new()
        .read(read)
        .write(!append)
        .append(append)
        .create(true)
        .mode(0o600)
        .custom_flags(nix::libc::O_NOFOLLOW | nix::libc::O_NONBLOCK | nix::libc::O_CLOEXEC)
        .open(path)
        .map_err(|e| err(format!("{}: {e}", path.display())))?;
    let st = f.metadata().map_err(|e| err(e.to_string()))?;
    if !st.is_file() || st.uid() != nix::unistd::getuid().as_raw() {
        return Err(err(format!(
            "{} is not a regular file you own",
            path.display()
        )));
    }
    f.set_permissions(std::fs::Permissions::from_mode(0o600))
        .map_err(|e| err(e.to_string()))?;
    Ok(f)
}

fn rotate(log: &Path, rotate_bytes: u64) {
    if let Ok(st) = std::fs::symlink_metadata(log)
        && st.is_file()
        && st.len() > rotate_bytes
    {
        let mut to = log.as_os_str().to_owned();
        to.push(".1");
        let _ = std::fs::rename(log, PathBuf::from(to));
    }
}

/// argv of a console started by `open`.
pub fn server_argv() -> Vec<String> {
    let exe = std::env::current_exe()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_default();
    vec![exe, LAUNCHER_ARG.to_string()]
}

/// A started console, as the launcher sees it (a real child, or a test fake).
pub trait Proc: Send {
    fn pid(&self) -> u32;
    /// Some(exit status) once it has exited.
    fn poll(&mut self) -> Option<i32>;
    /// SIGTERM, wait up to 5 s, then SIGKILL.
    fn terminate(&mut self);
}

impl Proc for std::process::Child {
    fn pid(&self) -> u32 {
        self.id()
    }
    fn poll(&mut self) -> Option<i32> {
        self.try_wait()
            .ok()
            .flatten()
            .map(|s| s.code().unwrap_or(-1))
    }
    fn terminate(&mut self) {
        // never kill(0, ...): that would signal our whole process group
        if let Ok(raw) = i32::try_from(self.id())
            && raw > 0
        {
            let _ = nix::sys::signal::kill(
                nix::unistd::Pid::from_raw(raw),
                nix::sys::signal::Signal::SIGTERM,
            );
        }
        let t = Instant::now();
        while t.elapsed() < Duration::from_secs(5) {
            if matches!(self.try_wait(), Ok(Some(_))) {
                return;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        let _ = self.kill();
        let _ = self.wait();
    }
}

/// How a console is spawned: (argv, stdout, stderr) -> the process. The real one puts it in a new process group.
pub type SpawnFn<'a> =
    dyn FnMut(&[String], std::fs::File, std::fs::File) -> std::io::Result<Box<dyn Proc>> + 'a;

/// Spawn `argv` in its own process group, stdin from /dev/null, stdout/stderr to the log.
pub fn spawn_detached(
    argv: &[String],
    out: std::fs::File,
    errf: std::fs::File,
) -> std::io::Result<Box<dyn Proc>> {
    let (prog, args) = argv
        .split_first()
        .ok_or_else(|| std::io::Error::other("empty argv"))?;
    let child = Command::new(prog)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::from(out))
        .stderr(Stdio::from(errf))
        .process_group(0)
        .spawn()?;
    Ok(Box::new(child))
}

pub fn start_background(cfg: &Settings) -> Result<Box<dyn Proc>, LauncherError> {
    start_background_with(cfg, LOG_ROTATE_BYTES, &mut spawn_detached)
}

/// The background start: the data dir made private, the log rotated past `rotate_bytes`, the log and the pid file
/// opened (no symlinks, ours, 0600) BEFORE the spawn, so a refused file never leaves an unrecorded console running.
pub fn start_background_with(
    cfg: &Settings,
    rotate_bytes: u64,
    spawn: &mut SpawnFn<'_>,
) -> Result<Box<dyn Proc>, LauncherError> {
    private_data_dir(&cfg.data_dir)?;
    let log = cfg.data_dir.join(LOG_NAME);
    rotate(&log, rotate_bytes);
    let log_f = open_private(&log, false, true)?;
    let mut pid_f = open_private(&cfg.data_dir.join(PID_NAME), false, false)?;
    let err_f = log_f.try_clone().map_err(|e| err(e.to_string()))?;
    let child = spawn(&server_argv(), log_f, err_f)
        .map_err(|e| err(format!("cannot start the console: {e}")))?;
    let _ = pid_f.set_len(0);
    let _ = pid_f.write_all(format!("{}\n", child.pid()).as_bytes());
    Ok(child)
}

/// `shutil.which(name)`, on the hub's fixed tool path instead of the caller's PATH (tools.rs; hardening deviation).
fn which(name: &str) -> Option<PathBuf> {
    crate::tools::resolve(name)
}

pub fn open_browser(url: &str) -> bool {
    let Some(opener) = which("xdg-open") else {
        return false;
    };
    // the session's environment (DISPLAY, DBUS_SESSION_BUS_ADDRESS, ...) is kept, but not the caller's PATH
    Command::new(opener)
        .arg(url)
        .env("PATH", crate::tools::tool_path())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .process_group(0)
        .spawn()
        .is_ok()
}

/// `html.escape(message, quote=False)`: notify-send may parse the body as markup.
pub fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

pub fn notify(message: &str) {
    let _ = writeln!(std::io::stderr(), "engine-console: {message}");
    if let Some(tool) = which("notify-send") {
        let child = Command::new(tool)
            .args(["--", "Engine Console", &html_escape(message)])
            .env("PATH", crate::tools::tool_path())
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn();
        if let Ok(mut c) = child {
            let t = Instant::now();
            while t.elapsed() < Duration::from_secs(5) {
                if matches!(c.try_wait(), Ok(Some(_))) {
                    return;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            let _ = c.kill();
        }
    }
}

/// (owner uid, argv) of a live process, or None if it is gone.
pub fn pid_cmdline(pid: i32) -> Option<(u32, Vec<String>)> {
    let uid = std::fs::metadata(format!("/proc/{pid}")).ok()?.uid();
    // capped: the argv compared is two short strings; a huge cmdline is simply not ours
    let raw = crate::fsutil::read_file(
        Path::new(&format!("/proc/{pid}/cmdline")),
        crate::fsutil::Follow::Yes,
        1024 * 1024,
    )
    .ok()?;
    Some((
        uid,
        raw.split(|b| *b == 0)
            .filter(|a| !a.is_empty())
            .map(|a| String::from_utf8_lossy(a).into_owned())
            .collect(),
    ))
}

/// `check(url, port)`
pub type CheckFn = dyn FnMut(&str, u16) -> Check + Send;
/// `start_background(cfg)`
pub type StartFn = dyn FnMut(&Settings) -> Result<Box<dyn Proc>, LauncherError> + Send;
/// `pid_cmdline(pid)`: (owner uid, argv) of a live process
pub type PidCmdlineFn = dyn FnMut(i32) -> Option<(u32, Vec<String>)> + Send;

/// Every side effect of `open` / `stop`, replaceable by tests (the Python tests monkeypatch these names).
pub struct Hooks {
    pub check: Box<CheckFn>,
    pub start: Box<StartFn>,
    pub open_browser: Box<dyn FnMut(&str) -> bool + Send>,
    pub notify: Box<dyn FnMut(&str) + Send>,
    pub pid_cmdline: Box<PidCmdlineFn>,
    /// SIGTERM to a pid
    pub signal: Box<dyn FnMut(i32) + Send>,
    pub poll_s: f64,
    pub start_timeout_s: f64,
    pub stop_timeout_s: f64,
}

impl Hooks {
    /// The real launcher.
    pub fn real() -> Self {
        Self {
            check: Box::new(check),
            start: Box::new(start_background),
            open_browser: Box::new(open_browser),
            notify: Box::new(notify),
            pid_cmdline: Box::new(pid_cmdline),
            signal: Box::new(|pid| {
                if pid > 0 {
                    // never kill(0 or -n, ...): those signal whole process groups
                    let _ = nix::sys::signal::kill(
                        nix::unistd::Pid::from_raw(pid),
                        nix::sys::signal::Signal::SIGTERM,
                    );
                }
            }),
            poll_s: POLL_S,
            start_timeout_s: START_TIMEOUT_S,
            stop_timeout_s: STOP_TIMEOUT_S,
        }
    }

    fn sleep(&self) {
        std::thread::sleep(Duration::from_secs_f64(self.poll_s.max(0.0)));
    }
}

fn other_user(url: &str) -> LauncherError {
    err(format!(
        "another user's process is listening on the console's port ({url}); not opening it"
    ))
}

fn do_open(cfg: &Settings, h: &mut Hooks) -> Result<(), LauncherError> {
    let url = console_url(cfg)?;
    private_data_dir(&cfg.data_dir)?;
    {
        let lock = open_private(&cfg.data_dir.join(LOCK_NAME), true, false)?;
        let guard = nix::fcntl::Flock::lock(lock, nix::fcntl::FlockArg::LockExclusive)
            .map_err(|(_, e)| err(format!("cannot lock: {e}")))?;
        let _held = guard.as_raw_fd(); // closing the last descriptor (dropping the guard) releases the lock
        match (h.check)(&url, cfg.port) {
            Check::OtherUser => return Err(other_user(&url)),
            Check::Foreign => {
                return Err(err(format!(
                    "something other than Engine Console answers on {url}; not opening it"
                )));
            }
            Check::Down => {
                if WILDCARD_HOSTS.contains(&cfg.host.trim()) {
                    return Err(err(format!(
                        "ENGINE_CONSOLE_HOST {} binds every interface; the launcher only starts a loopback-bound console",
                        ec_proto::sanitize::py_repr(&cfg.host)
                    )));
                }
                start_and_wait(cfg, &url, h)?;
            }
            Check::Console => {}
        }
    }
    if !(h.open_browser)(&url) {
        return Err(err(format!("xdg-open not found; open {url} in a browser")));
    }
    Ok(())
}

fn start_and_wait(cfg: &Settings, url: &str, h: &mut Hooks) -> Result<(), LauncherError> {
    let mut child = (h.start)(cfg)?;
    let log = cfg.data_dir.join(LOG_NAME);
    let deadline = Instant::now() + Duration::from_secs_f64(h.start_timeout_s.max(0.0));
    loop {
        let state = (h.check)(url, cfg.port); // check first: another console may have won the port
        if state == Check::Console {
            break;
        }
        if state == Check::OtherUser {
            child.terminate();
            return Err(other_user(url));
        }
        if child.poll().is_some() {
            return Err(err(format!(
                "the console did not start; see {}",
                log.display()
            )));
        }
        if Instant::now() >= deadline {
            child.terminate();
            return Err(err(format!(
                "the console did not become healthy in {:.0} s; see {}",
                h.start_timeout_s,
                log.display()
            )));
        }
        h.sleep();
    }
    if child.poll().is_some() {
        // a console of ours answers, but our child lost the port to it: no dead pid kept
        let _ = std::fs::remove_file(cfg.data_dir.join(PID_NAME));
        let _ = writeln!(
            std::io::stdout(),
            "Engine Console was started by something else; log of the failed start: {}",
            log.display()
        );
        return Ok(());
    }
    let _ = writeln!(
        std::io::stdout(),
        "started Engine Console in the background (pid {}); log: {}",
        child.pid(),
        log.display()
    );
    Ok(())
}

fn do_stop(cfg: &Settings, h: &mut Hooks) -> Result<(), LauncherError> {
    let pid_file = cfg.data_dir.join(PID_NAME);
    let text = match std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(nix::libc::O_NOFOLLOW | nix::libc::O_NONBLOCK | nix::libc::O_CLOEXEC)
        .open(&pid_file)
    {
        Ok(mut f) => {
            let mut buf = [0u8; 64];
            let n = f.read(&mut buf).unwrap_or(0);
            String::from_utf8_lossy(&buf[..n]).trim().to_string()
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(err(
                "no console started by `engine-console open` is recorded",
            ));
        }
        Err(e) => return Err(err(e.to_string())),
    };
    if text.is_empty() || !text.chars().all(|c| c.is_ascii_digit()) {
        return Err(err(format!("{} does not hold a pid", pid_file.display())));
    }
    let pid: i32 = text
        .parse()
        .map_err(|_| err(format!("{} does not hold a pid", pid_file.display())))?;
    let Some((uid, argv)) = (h.pid_cmdline)(pid) else {
        let _ = std::fs::remove_file(&pid_file);
        return Err(err(
            "the console started by `engine-console open` is no longer running",
        ));
    };
    if uid != nix::unistd::getuid().as_raw() || argv != server_argv() {
        let _ = std::fs::remove_file(&pid_file); // a recycled pid: never signal a stranger
        return Err(err(format!(
            "pid {pid} is not a console started by `engine-console open`; nothing stopped"
        )));
    }
    (h.signal)(pid);
    let deadline = Instant::now() + Duration::from_secs_f64(h.stop_timeout_s.max(0.0));
    while (h.pid_cmdline)(pid).is_some() {
        if Instant::now() > deadline {
            return Err(err(format!(
                "pid {pid} did not exit within {:.0} s",
                h.stop_timeout_s
            )));
        }
        h.sleep();
    }
    let _ = std::fs::remove_file(&pid_file);
    let _ = writeln!(std::io::stdout(), "stopped Engine Console (pid {pid})");
    Ok(())
}

fn run(
    command: &str,
    argv: &[String],
    load: impl FnOnce() -> Result<Settings, SettingsError>,
    h: &mut Hooks,
) -> i32 {
    if !argv.is_empty() {
        let _ = writeln!(std::io::stderr(), "usage: engine-console {command}");
        return 2;
    }
    let cfg = match load() {
        Ok(c) => c,
        Err(e) => {
            // names and reasons only: never the offending value
            (h.notify)(&format!("invalid setting: {}: {}", e.field, e.message));
            return 1;
        }
    };
    let res = if command == "open" {
        do_open(&cfg, h)
    } else {
        do_stop(&cfg, h)
    };
    match res {
        Ok(()) => 0,
        Err(e) => {
            (h.notify)(&e.0);
            1
        }
    }
}

pub fn open_console(argv: &[String]) -> i32 {
    run("open", argv, Settings::from_env, &mut Hooks::real())
}

pub fn stop_console(argv: &[String]) -> i32 {
    run("stop", argv, Settings::from_env, &mut Hooks::real())
}

/// `open_console(argv, cfg)` with the settings source and the side effects given (tests).
pub fn open_console_with(
    argv: &[String],
    load: impl FnOnce() -> Result<Settings, SettingsError>,
    h: &mut Hooks,
) -> i32 {
    run("open", argv, load, h)
}

/// `stop_console(argv, cfg)` with the settings source and the side effects given (tests).
pub fn stop_console_with(
    argv: &[String],
    load: impl FnOnce() -> Result<Settings, SettingsError>,
    h: &mut Hooks,
) -> i32 {
    run("stop", argv, load, h)
}
