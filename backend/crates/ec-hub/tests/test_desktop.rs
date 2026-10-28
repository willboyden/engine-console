//! Port of backend/tests/core/test_desktop.py: `engine-console open` / `stop`: URL, ownership check, background start
//! and stop. Everything is faked through `desktop::Hooks` (nothing is started on the console's port, opened in a
//! browser or signalled), except two real but harmless processes this test starts and stops itself.

#![allow(clippy::panic, clippy::expect_used, clippy::unwrap_used)]

use std::io::{Read, Write};
use std::os::unix::fs::{PermissionsExt, symlink};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ec_hub::config::Settings;
use ec_hub::desktop::{
    self, Check, Hooks, LauncherError, Proc, check_with, console_url, listen_uids,
    open_console_with, probe_with, server_argv, start_background_with, stop_console_with,
};

const URL: &str = "http://127.0.0.1:8791/";
const HEALTH: &str = r#"{"status": "ok", "version": "0.4.0", "engines": ["vllm", "sglang"], "docker_context": "rootless", "egress_mode": "proxied", "warnings": []}"#;
const TCP_HEADER: &str = "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n";

fn me() -> u32 {
    nix::unistd::getuid().as_raw()
}

fn cfg(host: &str, port: u16, data: &Path) -> Settings {
    let mut c = Settings::defaults();
    c.host = host.into();
    c.port = port;
    c.data_dir = data.to_path_buf();
    c
}

// ── the launcher fixture ──────────────────────────────────────────────────────────────────────────────────
struct FakeProc {
    exit_code: Option<i32>,
    terminated: Arc<AtomicBool>,
}

impl Proc for FakeProc {
    fn pid(&self) -> u32 {
        4242
    }
    fn poll(&mut self) -> Option<i32> {
        self.exit_code
    }
    fn terminate(&mut self) {
        self.terminated.store(true, Ordering::SeqCst);
    }
}

#[derive(Default)]
struct Rec {
    checks: Vec<Check>,
    started: u32,
    opened: Vec<String>,
    notes: Vec<String>,
    exit_code: Option<i32>,
    terminated: Arc<AtomicBool>,
}

fn launcher(rec: &Arc<Mutex<Rec>>) -> Hooks {
    let (r1, r2, r3, r4) = (
        Arc::clone(rec),
        Arc::clone(rec),
        Arc::clone(rec),
        Arc::clone(rec),
    );
    Hooks {
        check: Box::new(move |url, port| {
            assert!(url == URL || url == "http://[::1]:8791/", "{url}");
            assert_eq!(port, 8791);
            let mut r = r1.lock().unwrap();
            if r.checks.len() > 1 {
                r.checks.remove(0)
            } else {
                r.checks[0]
            }
        }),
        start: Box::new(
            move |cfg: &Settings| -> Result<Box<dyn Proc>, LauncherError> {
                let mut r = r2.lock().unwrap();
                r.started += 1;
                std::fs::write(cfg.data_dir.join("console.pid"), "4242\n").unwrap();
                Ok(Box::new(FakeProc {
                    exit_code: r.exit_code,
                    terminated: Arc::clone(&r.terminated),
                }))
            },
        ),
        open_browser: Box::new(move |url| {
            r3.lock().unwrap().opened.push(url.to_string());
            true
        }),
        notify: Box::new(move |m| r4.lock().unwrap().notes.push(m.to_string())),
        pid_cmdline: Box::new(|_| panic!("open must not look at processes")),
        signal: Box::new(|_| panic!("open must not signal")),
        poll_s: 0.0,
        start_timeout_s: desktop::START_TIMEOUT_S,
        stop_timeout_s: desktop::STOP_TIMEOUT_S,
    }
}

fn setup(checks: &[Check]) -> (tempfile::TempDir, Settings, Arc<Mutex<Rec>>) {
    let tmp = tempfile::tempdir().unwrap();
    let c = cfg("127.0.0.1", 8791, tmp.path());
    let rec = Arc::new(Mutex::new(Rec {
        checks: checks.to_vec(),
        ..Rec::default()
    }));
    (tmp, c, rec)
}

fn open(c: &Settings, rec: &Arc<Mutex<Rec>>) -> i32 {
    let c = c.clone();
    open_console_with(&[], move || Ok(c), &mut launcher(rec))
}

// --- URL -------------------------------------------------------------------------------------------------------
#[test]
fn test_console_url_maps_wildcard_binds_to_loopback() {
    let d = Path::new("/nonexistent");
    assert_eq!(console_url(&cfg("127.0.0.1", 8791, d)).unwrap(), URL);
    assert_eq!(
        console_url(&cfg("0.0.0.0", 9000, d)).unwrap(),
        "http://127.0.0.1:9000/"
    );
    assert_eq!(
        console_url(&cfg("::", 8791, d)).unwrap(),
        "http://[::1]:8791/"
    );
    assert_eq!(
        console_url(&cfg("::1", 8791, d)).unwrap(),
        "http://[::1]:8791/"
    );
    assert_eq!(
        console_url(&cfg("localhost", 8791, d)).unwrap(),
        "http://localhost:8791/"
    );
}

#[test]
fn test_console_url_refuses_anything_off_the_box() {
    for host in [
        "127.0.0.1@evil.example",
        "evil.example/#",
        "evil.example",
        "203.0.113.7",
        "10.0.0.1",
    ] {
        let e = console_url(&cfg(host, 8791, Path::new("/x"))).unwrap_err();
        assert!(e.0.contains("not a loopback address"), "{host}: {e}");
    }
}

#[test]
fn test_console_url_refuses_an_out_of_range_port() {
    // the setting cannot even hold 70000 (it is read as a u16), and 0 is refused by the launcher
    let e = Settings::load(&[("ENGINE_CONSOLE_PORT".into(), "70000".into())], None).unwrap_err();
    assert_eq!(e.field, "port");
    assert!(
        console_url(&cfg("127.0.0.1", 0, Path::new("/x")))
            .unwrap_err()
            .0
            .contains("out of range")
    );
}

// --- who holds the port ----------------------------------------------------------------------------------------
#[test]
fn test_listen_uids_reads_every_listener_on_the_port() {
    let tmp = tempfile::tempdir().unwrap();
    let (tcp, tcp6) = (tmp.path().join("tcp"), tmp.path().join("tcp6"));
    std::fs::write(
        &tcp,
        format!(
            "{TCP_HEADER}   0: 0100007F:2257 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 11\n   1: 0100007F:2257 0100007F:9C40 01 00000000:00000000 00:00000000 00000000     0        0 12\n   2: 00000000:1F90 00000000:0000 0A 00000000:00000000 00:00000000 00000000     0        0 13\n"
        ),
    )
    .unwrap();
    std::fs::write(
        &tcp6,
        format!(
            "{TCP_HEADER}   0: 00000000000000000000000000000000:2257 00000000000000000000000000000000:0000 0A 00000000:00000000 00:00000000 00000000   999        0 14\n"
        ),
    )
    .unwrap();
    let (t4, t6) = (
        tcp.to_string_lossy().into_owned(),
        tcp6.to_string_lossy().into_owned(),
    );
    assert_eq!(listen_uids(8791, &[t4.as_str()]), Some(vec![1000])); // ESTABLISHED and other ports ignored
    let mut both = listen_uids(8791, &[t4.as_str(), t6.as_str()]).unwrap();
    both.sort_unstable();
    assert_eq!(both, vec![999, 1000]); // a wildcard IPv6 squatter counts
    let missing = tmp.path().join("missing").to_string_lossy().into_owned();
    assert_eq!(listen_uids(8791, &[missing.as_str()]), Some(vec![]));
}

#[test]
fn test_check_flags_a_console_shaped_answer_from_another_users_socket() {
    let console = |_: &str| Check::Console;
    assert_eq!(
        check_with(URL, 8791, console, |_| Some(vec![0])),
        Check::OtherUser
    );
    assert_eq!(
        check_with(URL, 8791, console, |_| Some(vec![me(), 0])),
        Check::OtherUser
    );
    assert_eq!(check_with(URL, 8791, console, |_| None), Check::OtherUser); // unreadable: fail closed
    assert_eq!(
        check_with(URL, 8791, console, |_| Some(vec![me()])),
        Check::Console
    );
}

/// A one-shot HTTP server on an ephemeral loopback port: reads the request (sent back on the channel), then writes
/// `chunks` (each after `delay`), then closes.
fn serve_once(chunks: Vec<Vec<u8>>, delay: Duration) -> (u16, std::sync::mpsc::Receiver<String>) {
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = l.local_addr().unwrap().port();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        if let Ok((mut s, _)) = l.accept() {
            let mut buf = [0u8; 4096];
            let n = s.read(&mut buf).unwrap_or(0);
            let _ = tx.send(String::from_utf8_lossy(&buf[..n]).into_owned());
            for c in chunks {
                std::thread::sleep(delay);
                if s.write_all(&c).is_err() {
                    return;
                }
            }
        }
    });
    (port, rx)
}

fn http_answer(body: &[u8]) -> Vec<u8> {
    let mut v = format!("HTTP/1.1 200 OK\r\ncontent-length: {}\r\n\r\n", body.len()).into_bytes();
    v.extend_from_slice(body);
    v
}

#[test]
fn test_probe_tells_console_foreign_and_down_apart() {
    let redirect =
        b"HTTP/1.1 302 Found\r\nlocation: http://127.0.0.1:1/\r\ncontent-length: 0\r\n\r\n"
            .to_vec();
    for (answer, want) in [
        (http_answer(HEALTH.as_bytes()), Check::Console),
        (
            http_answer(b"<html>some other service</html>"),
            Check::Foreign,
        ),
        (http_answer(br#"{"status": "ok"}"#), Check::Foreign),
        (redirect, Check::Foreign), // never followed
    ] {
        let (port, req) = serve_once(vec![answer], Duration::ZERO);
        assert_eq!(probe_with(&format!("http://127.0.0.1:{port}/"), 2.0), want);
        assert!(
            req.recv()
                .unwrap()
                .starts_with("GET /api/v1/health HTTP/1.1\r\n")
        );
    }
    let closed = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = closed.local_addr().unwrap().port();
    drop(closed);
    assert_eq!(
        probe_with(&format!("http://127.0.0.1:{port}/"), 2.0),
        Check::Down
    ); // connection refused
}

#[test]
fn test_probe_caps_the_body_and_the_total_read_time() {
    // 100 KiB of padding before any JSON
    let mut head = b"HTTP/1.1 200 OK\r\ncontent-length: 102400\r\n\r\n".to_vec();
    head.extend(std::iter::repeat_n(b' ', 1024));
    let mut chunks = vec![head];
    chunks.extend((0..99).map(|_| vec![b' '; 1024]));
    let (port, _req) = serve_once(chunks, Duration::ZERO);
    assert_eq!(
        probe_with(&format!("http://127.0.0.1:{port}/"), 2.0),
        Check::Foreign
    );
    // a drip that would take 10 s
    let mut drip = vec![b"HTTP/1.1 200 OK\r\ncontent-length: 1000\r\n\r\n".to_vec()];
    drip.extend((0..1000).map(|_| vec![b' ']));
    let (port, _req) = serve_once(drip, Duration::from_millis(10));
    let t0 = Instant::now();
    assert_eq!(
        probe_with(&format!("http://127.0.0.1:{port}/"), 0.05),
        Check::Foreign
    );
    assert!(t0.elapsed() < Duration::from_secs(1));
}

// --- open ------------------------------------------------------------------------------------------------------
#[test]
fn test_open_when_already_running_starts_nothing() {
    let (_t, c, rec) = setup(&[Check::Console]);
    assert_eq!(open(&c, &rec), 0);
    let r = rec.lock().unwrap();
    assert_eq!(
        (r.started, r.opened.clone(), r.notes.len()),
        (0, vec![URL.to_string()], 0)
    );
}

#[test]
fn test_open_starts_the_console_then_waits_for_health() {
    let (_t, c, rec) = setup(&[Check::Down, Check::Down, Check::Console]);
    assert_eq!(open(&c, &rec), 0);
    let r = rec.lock().unwrap();
    assert_eq!((r.started, r.opened.clone()), (1, vec![URL.to_string()]));
}

#[test]
fn test_open_refuses_a_port_it_cannot_trust() {
    for (state, note) in [
        (Check::Foreign, "something other than Engine Console"),
        (Check::OtherUser, "another user's process"),
    ] {
        let (_t, c, rec) = setup(&[state]);
        assert_eq!(open(&c, &rec), 1);
        let r = rec.lock().unwrap();
        assert_eq!((r.started, r.opened.len()), (0, 0));
        assert!(r.notes[0].contains(note), "{}", r.notes[0]);
    }
}

#[test]
fn test_open_refuses_a_squatter_that_wins_the_port_during_start() {
    let (_t, c, rec) = setup(&[Check::Down, Check::OtherUser]); // our child lost the bind; the squatter answers
    rec.lock().unwrap().exit_code = Some(1);
    assert_eq!(open(&c, &rec), 1);
    let r = rec.lock().unwrap();
    assert!(r.opened.is_empty() && r.notes[0].contains("another user's process"));
}

#[test]
fn test_open_reports_a_console_that_exits_during_start() {
    let (_t, c, rec) = setup(&[Check::Down]);
    rec.lock().unwrap().exit_code = Some(1);
    assert_eq!(open(&c, &rec), 1);
    let r = rec.lock().unwrap();
    assert!(
        r.opened.is_empty() && r.notes[0].contains("console.log"),
        "{}",
        r.notes[0]
    );
}

#[test]
fn test_open_stops_its_child_after_the_start_timeout() {
    let (_t, c, rec) = setup(&[Check::Down]);
    let mut h = launcher(&rec);
    h.start_timeout_s = 0.0;
    assert_eq!(open_console_with(&[], move || Ok(c), &mut h), 1);
    let r = rec.lock().unwrap();
    assert!(r.opened.is_empty() && r.terminated.load(Ordering::SeqCst));
}

#[test]
fn test_open_refuses_a_non_loopback_host_before_probing() {
    let (t, _c, rec) = setup(&[Check::Console]);
    let bad = cfg("evil.example", 8791, t.path());
    assert_eq!(open(&bad, &rec), 1);
    let r = rec.lock().unwrap();
    assert!(r.started == 0 && r.notes[0].contains("not a loopback address"));
}

#[test]
fn test_open_does_not_record_a_child_that_lost_the_port_to_another_console() {
    let (_t, c, rec) = setup(&[Check::Down, Check::Console]); // a console of ours answers, but our child exited
    rec.lock().unwrap().exit_code = Some(1);
    assert_eq!(open(&c, &rec), 0);
    assert_eq!(rec.lock().unwrap().opened, vec![URL.to_string()]);
    assert!(!c.data_dir.join("console.pid").exists());
}

#[test]
fn test_concurrent_launches_wait_for_each_other() {
    let (_t, c, rec) = setup(&[Check::Console]);
    std::fs::set_permissions(&c.data_dir, std::fs::Permissions::from_mode(0o700)).unwrap();
    let f = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(c.data_dir.join("console.lock"))
        .unwrap();
    let held = nix::fcntl::Flock::lock(f, nix::fcntl::FlockArg::LockExclusive).unwrap(); // another launch is mid-start
    let rec2 = Arc::clone(&rec);
    let t = std::thread::spawn(move || open(&c, &rec2));
    std::thread::sleep(Duration::from_millis(300));
    assert!(rec.lock().unwrap().opened.is_empty()); // blocked on the lock, has not even probed
    drop(held);
    assert_eq!(t.join().unwrap(), 0);
    let r = rec.lock().unwrap();
    assert_eq!((r.opened.clone(), r.started), (vec![URL.to_string()], 0));
}

#[test]
fn test_open_never_starts_a_wildcard_bound_console() {
    for host in ["0.0.0.0", "::"] {
        let (t, _c, rec) = setup(&[Check::Down]);
        let wild = cfg(host, 8791, t.path());
        assert_eq!(open(&wild, &rec), 1);
        let r = rec.lock().unwrap();
        assert!(
            r.started == 0 && r.notes[0].contains("binds every interface"),
            "{host}"
        );
    }
}

#[test]
fn test_open_and_stop_reject_arguments() {
    let (_t, c, rec) = setup(&[Check::Console]);
    let (c1, c2) = (c.clone(), c);
    assert_eq!(
        open_console_with(
            &["--port".into(), "1".into()],
            move || Ok(c1),
            &mut launcher(&rec)
        ),
        2
    );
    assert_eq!(
        stop_console_with(&["now".into()], move || Ok(c2), &mut launcher(&rec)),
        2
    );
    assert_eq!(rec.lock().unwrap().started, 0);
}

// --- background start ------------------------------------------------------------------------------------------
type Spawned = Arc<Mutex<Vec<Vec<String>>>>;

fn fake_spawn(
    calls: &Spawned,
) -> impl FnMut(&[String], std::fs::File, std::fs::File) -> std::io::Result<Box<dyn Proc>> {
    let calls = Arc::clone(calls);
    move |argv: &[String], _o, _e| {
        calls.lock().unwrap().push(argv.to_vec());
        Ok(Box::new(FakeProc {
            exit_code: None,
            terminated: Arc::new(AtomicBool::new(false)),
        }) as Box<dyn Proc>)
    }
}

fn mode(p: &Path) -> u32 {
    std::fs::metadata(p).unwrap().permissions().mode() & 0o777
}

#[test]
fn test_start_background_is_isolated_detached_and_private() {
    let tmp = tempfile::tempdir().unwrap();
    let state = tmp.path().join("state");
    let calls: Spawned = Arc::default();
    start_background_with(
        &cfg("127.0.0.1", 8791, &state),
        1 << 20,
        &mut fake_spawn(&calls),
    )
    .unwrap();
    assert_eq!(calls.lock().unwrap()[0], server_argv());
    assert_eq!(server_argv()[1], "--from-launcher");
    assert_eq!(mode(&state), 0o700);
    for name in ["console.log", "console.pid"] {
        assert_eq!(mode(&state.join(name)), 0o600, "{name}");
    }
    assert_eq!(
        std::fs::read_to_string(state.join("console.pid")).unwrap(),
        "4242\n"
    );
    // the real spawn detaches the console into its own process group (a harmless child we start and stop here)
    let (o, e) = (tempfile::tempfile().unwrap(), tempfile::tempfile().unwrap());
    let mut child = desktop::spawn_detached(&["/bin/sleep".into(), "30".into()], o, e).unwrap();
    let stat = std::fs::read_to_string(format!("/proc/{}/stat", child.pid())).unwrap();
    let pgrp: u32 = stat
        .rsplit(')')
        .next()
        .unwrap()
        .split_whitespace()
        .nth(2)
        .unwrap()
        .parse()
        .unwrap();
    assert_eq!(pgrp, child.pid());
    child.terminate();
}

#[test]
fn test_start_background_refuses_a_symlinked_log() {
    let tmp = tempfile::tempdir().unwrap();
    let (state, target) = (tmp.path().join("state"), tmp.path().join("elsewhere"));
    std::fs::create_dir(&state).unwrap();
    std::fs::set_permissions(&state, std::fs::Permissions::from_mode(0o700)).unwrap();
    std::fs::write(&target, "keep\n").unwrap();
    symlink(&target, state.join("console.log")).unwrap();
    let calls: Spawned = Arc::default();
    assert!(
        start_background_with(
            &cfg("127.0.0.1", 8791, &state),
            1 << 20,
            &mut fake_spawn(&calls)
        )
        .is_err()
    );
    assert!(calls.lock().unwrap().is_empty());
    assert_eq!(std::fs::read_to_string(&target).unwrap(), "keep\n");
}

#[test]
fn test_start_background_refuses_a_symlinked_pid_file_before_spawning() {
    let tmp = tempfile::tempdir().unwrap();
    let (state, target) = (tmp.path().join("state"), tmp.path().join("elsewhere"));
    std::fs::create_dir(&state).unwrap();
    std::fs::set_permissions(&state, std::fs::Permissions::from_mode(0o700)).unwrap();
    std::fs::write(&target, "keep\n").unwrap();
    symlink(&target, state.join("console.pid")).unwrap();
    let calls: Spawned = Arc::default();
    assert!(
        start_background_with(
            &cfg("127.0.0.1", 8791, &state),
            1 << 20,
            &mut fake_spawn(&calls)
        )
        .is_err()
    );
    assert!(calls.lock().unwrap().is_empty());
    assert_eq!(std::fs::read_to_string(&target).unwrap(), "keep\n");
}

#[test]
fn test_start_background_tightens_an_existing_log_and_rotates_a_big_one() {
    let tmp = tempfile::tempdir().unwrap();
    let state = tmp.path().join("state");
    std::fs::create_dir(&state).unwrap();
    std::fs::set_permissions(&state, std::fs::Permissions::from_mode(0o755)).unwrap();
    let log = state.join("console.log");
    std::fs::write(&log, "x".repeat(20)).unwrap();
    std::fs::set_permissions(&log, std::fs::Permissions::from_mode(0o644)).unwrap();
    let calls: Spawned = Arc::default();
    start_background_with(&cfg("127.0.0.1", 8791, &state), 10, &mut fake_spawn(&calls)).unwrap();
    assert_eq!(
        std::fs::read_to_string(state.join("console.log.1")).unwrap(),
        "x".repeat(20)
    );
    assert_eq!((mode(&log), mode(&state)), (0o600, 0o700));
}

// --- stop ------------------------------------------------------------------------------------------------------
#[derive(Default)]
struct StopRec {
    alive: Vec<(u32, Vec<String>)>,
    signals: Vec<i32>,
    notes: Vec<String>,
}

fn stopper(tmp: &Path) -> (Settings, Arc<Mutex<StopRec>>, Hooks) {
    std::fs::write(tmp.join("console.pid"), "4242\n").unwrap();
    let rec = Arc::new(Mutex::new(StopRec::default()));
    let (r1, r2, r3) = (Arc::clone(&rec), Arc::clone(&rec), Arc::clone(&rec));
    let h = Hooks {
        check: Box::new(|_, _| panic!("stop must not probe")),
        start: Box::new(|_| panic!("stop must not start")),
        open_browser: Box::new(|_| panic!("stop must not open a browser")),
        notify: Box::new(move |m| r3.lock().unwrap().notes.push(m.to_string())),
        pid_cmdline: Box::new(move |pid| {
            assert_eq!(pid, 4242);
            let mut r = r1.lock().unwrap();
            if r.alive.is_empty() {
                None
            } else {
                Some(r.alive.remove(0))
            }
        }),
        signal: Box::new(move |pid| r2.lock().unwrap().signals.push(pid)),
        poll_s: 0.0,
        start_timeout_s: desktop::START_TIMEOUT_S,
        stop_timeout_s: desktop::STOP_TIMEOUT_S,
    };
    (cfg("127.0.0.1", 8791, tmp), rec, h)
}

#[test]
fn test_stop_terminates_the_console_it_started() {
    let tmp = tempfile::tempdir().unwrap();
    let (c, rec, mut h) = stopper(tmp.path());
    let me_proc = (me(), server_argv());
    rec.lock().unwrap().alive = vec![me_proc.clone(), me_proc]; // identity check, then one poll before it exits
    let cc = c.clone();
    assert_eq!(stop_console_with(&[], move || Ok(cc), &mut h), 0);
    assert_eq!(rec.lock().unwrap().signals, vec![4242]); // SIGTERM (the hook is the real one's stand-in)
    assert!(!c.data_dir.join("console.pid").exists());
}

#[test]
fn test_stop_never_signals_a_process_it_did_not_start() {
    for proc in [
        (0, server_argv()), // another user
        (
            me(),
            vec![
                "/usr/bin/vim".to_string(),
                "src/engine_console/main.py".to_string(),
            ],
        ), // recycled pid
        (
            me(),
            vec![
                "/usr/local/bin/engine-console".to_string(),
                "--from-launcher".into(),
            ],
        ), // another binary
    ] {
        let tmp = tempfile::tempdir().unwrap();
        let (c, rec, mut h) = stopper(tmp.path());
        rec.lock().unwrap().alive = vec![proc];
        assert_eq!(stop_console_with(&[], move || Ok(c), &mut h), 1);
        let r = rec.lock().unwrap();
        assert!(
            r.signals.is_empty() && r.notes[0].contains("nothing stopped"),
            "{:?}",
            r.notes
        );
    }
}

#[test]
fn test_stop_reports_a_console_that_is_already_gone() {
    let tmp = tempfile::tempdir().unwrap();
    let (c, rec, mut h) = stopper(tmp.path());
    let cc = c.clone();
    assert_eq!(stop_console_with(&[], move || Ok(cc), &mut h), 1);
    assert!(rec.lock().unwrap().signals.is_empty());
    assert!(!c.data_dir.join("console.pid").exists());
}

// --- entry point -----------------------------------------------------------------------------------------------
#[test]
fn test_a_malformed_setting_is_reported_not_raised() {
    let (_t, _c, rec) = setup(&[Check::Console]);
    let load = || {
        Settings::load(
            &[(
                "ENGINE_CONSOLE_PORT".into(),
                "value-that-must-not-be-echoed".into(),
            )],
            None,
        )
    };
    assert_eq!(open_console_with(&[], load, &mut launcher(&rec)), 1);
    let r = rec.lock().unwrap();
    assert_eq!(r.notes.len(), 1);
    assert!(
        r.notes[0].contains("port") && !r.notes[0].contains("value-that-must-not-be-echoed"),
        "{}",
        r.notes[0]
    );
}

#[test]
fn test_engine_console_dispatches_open_and_stop_to_the_launcher() {
    // The real binary: `open`/`stop` reach the launcher (which refuses the extra argument with its own usage line,
    // exit 2) and never serve in-process; nothing is started, opened or signalled.
    let home = tempfile::tempdir().unwrap();
    for cmd in ["open", "stop"] {
        let out = std::process::Command::new(env!("CARGO_BIN_EXE_engine-console"))
            .args([cmd, "now"])
            .env_clear()
            .env("HOME", home.path())
            .env("PATH", "/usr/bin:/bin")
            .output()
            .unwrap();
        assert_eq!(out.status.code(), Some(2), "{cmd}");
        assert!(
            String::from_utf8_lossy(&out.stderr).contains(&format!("usage: engine-console {cmd}"))
        );
    }
}

// ── the real launcher, end to end, hermetic (Rust-native extra; no Python test starts a real console) ─────────
// `engine-console open` / `stop` as the desktop entry runs them: a real background console on an ephemeral loopback
// port, every path in a temp dir, nothing inherited from this shell (env_clear), and a fake `xdg-open` and
// `notify-send` FIRST on the hub's tool path (`ENGINE_CONSOLE_TOOL_PATH`, then the parity fakes for
// docker/nvidia-smi/ip, then /usr/bin:/bin), so no browser opens and no desktop notification is sent. The launcher
// never looks at PATH for its tools (tools.rs): step 3 puts the fake `xdg-open` first on PATH and leaves it off the
// tool path. A drop guard stops the console this test started if an assertion fails.

const FAKES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../parity/fakes/bin");
const CONSOLE_BIN: &str = env!("CARGO_BIN_EXE_engine-console");

/// A fake program at `dir/name` that appends its arguments (one per line, then "--end--") to `log`.
fn fake_tool(dir: &Path, name: &str, log: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    let p = dir.join(name);
    std::fs::write(
        &p,
        format!(
            "#!/bin/sh\n# fake {name}: records its argv, opens nothing\nprintf '%s\\n' \"$@\" --end-- >> '{}'\n",
            log.display()
        ),
    )
    .unwrap();
    std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
}

struct Launch {
    root: tempfile::TempDir,
    port: u16,
}

impl Launch {
    fn new() -> Self {
        let root = tempfile::tempdir().unwrap();
        let t = root.path();
        fake_tool(
            &t.join("bin"),
            "xdg-open",
            &t.join("rec").join("xdg-open.log"),
        );
        fake_tool(
            &t.join("bin"),
            "notify-send",
            &t.join("rec").join("notify.log"),
        );
        // a PATH with a notify-send but no xdg-open at all (not even /usr/bin's)
        fake_tool(
            &t.join("bin-noopen"),
            "notify-send",
            &t.join("rec").join("notify.log"),
        );
        std::fs::create_dir_all(t.join("rec")).unwrap();
        std::fs::create_dir_all(t.join("hf")).unwrap();
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        Self { root, port }
    }

    fn t(&self) -> &Path {
        self.root.path()
    }

    fn data(&self) -> std::path::PathBuf {
        self.t().join("data")
    }

    fn rec(&self, name: &str) -> String {
        std::fs::read_to_string(self.t().join("rec").join(name)).unwrap_or_default()
    }

    /// `engine-console <cmd>` with nothing inherited. PATH is always the fake-first one; `tools` replaces the tool
    /// path (`ENGINE_CONSOLE_TOOL_PATH`, which defaults to that same fake-first path).
    fn run(&self, cmd: &str, tools: Option<&str>) -> (i32, String, String) {
        let t = self.t();
        let default_path = format!("{}:{FAKES}:/usr/bin:/bin", t.join("bin").display());
        let mut c = std::process::Command::new(CONSOLE_BIN);
        c.arg(cmd)
            .env_clear()
            .env("PATH", &default_path)
            .env("ENGINE_CONSOLE_TOOL_PATH", tools.unwrap_or(&default_path))
            .env("HOME", t)
            .env("LANG", "C.UTF-8")
            .env("XDG_CONFIG_HOME", t.join("xdg-config"))
            .env("XDG_DATA_HOME", t.join("xdg-data"))
            .env("XDG_STATE_HOME", t.join("xdg-state"))
            .env("XDG_CACHE_HOME", t.join("xdg-cache"))
            .env("XDG_RUNTIME_DIR", t.join("xdg-run"))
            .env("ENGINE_CONSOLE_ENV_FILE", t.join("no-env-file"))
            .env("ENGINE_CONSOLE_HOST", "127.0.0.1")
            .env("ENGINE_CONSOLE_PORT", self.port.to_string())
            .env("ENGINE_CONSOLE_DATA_DIR", self.data())
            .env("HF_CACHE_DIR", t.join("hf"))
            .env("HF_TOKEN_FILE", t.join("no-token"))
            .env("HF_ENDPOINT", "https://127.0.0.1:9")
            .env("HOSTS_FILE", t.join("hosts.json"))
            .env("HUB_KEY_FILE", t.join("no-key"))
            .env("FLEET_ENABLED", "false")
            .env("DISCOVERY_ENABLED", "false")
            .env("BACKGROUND_TASKS", "false")
            .env("OTLP_ENABLED", "false")
            .env("OTLP_ENDPOINT", "127.0.0.1:9")
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
        let mut child = c.spawn().unwrap();
        let deadline = Instant::now() + Duration::from_secs(90);
        while child.try_wait().unwrap().is_none() {
            if Instant::now() > deadline {
                let _ = child.kill();
                let _ = child.wait();
                panic!("engine-console {cmd} did not finish in 90 s");
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        let out = child.wait_with_output().unwrap();
        (
            out.status.code().unwrap_or(-1),
            String::from_utf8_lossy(&out.stdout).into_owned(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        )
    }

    fn recorded_pid(&self) -> Option<i32> {
        std::fs::read_to_string(self.data().join(desktop::PID_NAME))
            .ok()?
            .trim()
            .parse()
            .ok()
    }

    fn log_tail(&self) -> String {
        let s = std::fs::read_to_string(self.data().join(desktop::LOG_NAME)).unwrap_or_default();
        s[s.len().saturating_sub(2000)..].to_string()
    }
}

/// Our console: this user's process whose argv is exactly `[<engine-console>, "--from-launcher"]`.
fn is_our_console(pid: i32) -> bool {
    let want = std::fs::canonicalize(CONSOLE_BIN).unwrap();
    desktop::pid_cmdline(pid).is_some_and(|(uid, argv)| {
        uid == me()
            && argv.len() == 2
            && argv[1] == desktop::LAUNCHER_ARG
            && std::fs::canonicalize(&argv[0]).is_ok_and(|p| p == want)
    })
}

impl Drop for Launch {
    /// Never leave a console behind: stop the one this test started (checked by uid and argv, never a pid group).
    fn drop(&mut self) {
        if let Some(pid) = self.recorded_pid().filter(|p| *p > 0)
            && is_our_console(pid)
        {
            let p = nix::unistd::Pid::from_raw(pid);
            let _ = nix::sys::signal::kill(p, nix::sys::signal::Signal::SIGTERM);
            let t = Instant::now();
            while is_our_console(pid) && t.elapsed() < Duration::from_secs(5) {
                std::thread::sleep(Duration::from_millis(50));
            }
            if is_our_console(pid) {
                let _ = nix::sys::signal::kill(p, nix::sys::signal::Signal::SIGKILL);
            }
        }
    }
}

#[test]
fn test_open_then_stop_end_to_end_with_a_fake_xdg_open() {
    let l = Launch::new();
    let url = format!("http://127.0.0.1:{}/", l.port);
    let data = l.data();
    let log = data.join(desktop::LOG_NAME);

    // 1. nothing serves yet: open starts the console in the background, waits for it, then opens the URL
    let (code, out, err) = l.run("open", None);
    assert_eq!(
        code,
        0,
        "stdout: {out}\nstderr: {err}\nlog: {}",
        l.log_tail()
    );
    let pid = l.recorded_pid().expect("console.pid holds the pid");
    assert_eq!(
        out,
        format!(
            "started Engine Console in the background (pid {pid}); log: {}\n",
            log.display()
        )
    );
    assert_eq!(err, "");
    assert!(
        is_our_console(pid),
        "the recorded pid is the console started by open"
    );
    assert_eq!(l.rec("xdg-open.log"), format!("{url}\n--end--\n"));
    assert_eq!(l.rec("notify.log"), "", "no notification on success");
    assert_eq!(mode(&data), 0o700);
    assert_eq!(mode(&log), 0o600);
    assert_eq!(mode(&data.join(desktop::PID_NAME)), 0o600);
    assert_eq!(mode(&data.join(desktop::LOCK_NAME)), 0o600);
    assert_eq!(desktop::check(&url, l.port), Check::Console);

    // 2. it already serves: open only opens the browser again (nothing started, the pid file unchanged)
    let (code, out, err) = l.run("open", None);
    assert_eq!((code, out.as_str(), err.as_str()), (0, "", ""));
    assert_eq!(
        l.rec("xdg-open.log"),
        format!("{url}\n--end--\n{url}\n--end--\n")
    );
    assert_eq!(l.recorded_pid(), Some(pid));

    // 3. no xdg-open on the tool path (the fake one is still first on PATH, which the launcher never searches):
    // refused with the URL to open by hand, also sent to notify-send
    let noopen = l.t().join("bin-noopen");
    let (code, out, err) = l.run("open", Some(&noopen.to_string_lossy()));
    let msg = format!("xdg-open not found; open {url} in a browser");
    assert_eq!((code, out.as_str()), (1, ""));
    assert_eq!(err, format!("engine-console: {msg}\n"));
    assert_eq!(
        l.rec("notify.log"),
        format!("--\nEngine Console\n{msg}\n--end--\n")
    );
    assert_eq!(
        l.rec("xdg-open.log").lines().count(),
        4,
        "the fake on PATH was not used: it is not on the tool path"
    );

    // 4. stop: the recorded console is signalled, waited for, and its pid file removed
    let (code, out, err) = l.run("stop", None);
    assert_eq!(code, 0, "stdout: {out}\nstderr: {err}");
    assert_eq!(out, format!("stopped Engine Console (pid {pid})\n"));
    assert_eq!(err, "");
    assert!(!is_our_console(pid));
    assert!(!data.join(desktop::PID_NAME).exists());
    assert!(
        std::net::TcpStream::connect(("127.0.0.1", l.port)).is_err(),
        "nothing listens on the port any more"
    );

    // 5. stop again: nothing recorded, refused (exit 1) through stderr and notify-send
    let (code, out, err) = l.run("stop", None);
    let msg = "no console started by `engine-console open` is recorded";
    assert_eq!((code, out.as_str()), (1, ""));
    assert_eq!(err, format!("engine-console: {msg}\n"));
    assert!(
        l.rec("notify.log")
            .ends_with(&format!("--\nEngine Console\n{msg}\n--end--\n")),
        "{}",
        l.rec("notify.log")
    );
}
