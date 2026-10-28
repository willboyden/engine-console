//! Port of backend/tests/core/test_wiring.py: the composition root, the lifespan and the CLI entry point. Python
//! monkeypatches `create_app`/`uvicorn.run` to test `run()`; here the real binary runs with its home, data dir, HF
//! cache and hosts file in a temp dir, the fleet and background loops off, and an ephemeral port (never 8791).
#![allow(clippy::panic, clippy::expect_used, clippy::unwrap_used)] // test helpers outside #[test] fns

#[path = "common/kit.rs"]
mod kit;

use std::io::{Read as _, Write as _};
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ec_hub::domain::errors::internal;
use ec_hub::domain::fleet::{Fleet, HostCounts};
use kit::TestEnv;
use serde_json::json;

const P: &str = "/api/v1";
const BIN: &str = env!("CARGO_BIN_EXE_engine-console");

#[tokio::test]
async fn test_the_fleet_is_told_how_to_name_instances_and_count_engines_and_models() {
    let mut env = TestEnv::new();
    let c = env.container();
    // H1's router reads `Container.fleet`: the one the app was built with
    let built = Arc::clone(&env.fleet) as Arc<dyn Fleet>;
    assert!(Arc::ptr_eq(&c.fleet, &built));
    let resolver = env
        .fleet
        .resolver
        .lock()
        .unwrap()
        .clone()
        .expect("the instance resolver is set");
    let counts = env
        .fleet
        .counts
        .lock()
        .unwrap()
        .clone()
        .expect("the counts provider is set");
    // both answer from the lifecycle (Python also checks models == 1 after seeding a model)
    assert_eq!(
        resolver("local", "nobody"),
        c.lifecycle.instance_for_container("local", "nobody")
    );
    let got: HostCounts = counts("local");
    assert_eq!(got, c.lifecycle.counts("local"));
    assert_eq!(got.engines_total, 0);
}

/// Python looks for `env.fleet` among each service's attributes. The services keep their fleet private here, so the
/// check is that they all hold the one fleet the container was given: the test env, the container and at least
/// lifecycle, scripts, hardware, nodehttp, metrics and library share that one Arc.
#[tokio::test]
async fn test_every_service_that_talks_to_a_machine_goes_through_the_one_fleet() {
    let mut env = TestEnv::new();
    let c = env.container();
    assert!(Arc::ptr_eq(
        &c.fleet,
        &(Arc::clone(&env.fleet) as Arc<dyn Fleet>)
    ));
    let holders = Arc::strong_count(&env.fleet);
    assert!(holders >= 2 + 6, "only {holders} holders of the fleet");
}

#[tokio::test]
async fn test_the_lifespan_starts_and_stops_the_fleet_and_the_library_and_closes_what_the_last_run_left_open()
 {
    let mut env = TestEnv::new();
    let c = env.container();
    // the library's start/stop and the bench recovery (b1 -> failed) are not checked here
    assert!(!env.fleet.is_started());
    let tasks = ec_hub::app::lifespan_start(&c).await;
    assert!(env.fleet.is_started());
    assert_eq!(env.get(&format!("{P}/health")).send().await.status, 200);
    ec_hub::app::lifespan_stop(&c, tasks).await;
    assert!(!env.fleet.is_started());
}

#[tokio::test]
async fn test_health_lists_engines_and_the_loopback_warning_follows_the_setting() {
    let mut env = TestEnv::new();
    let body = env.get(&format!("{P}/health")).send().await.json();
    assert_eq!(body["engines"], json!(["fake"]));
    assert!(
        body["warnings"]
            .as_array()
            .unwrap()
            .iter()
            .any(|w| w.as_str().unwrap().contains("trust_loopback"))
    );
}

#[derive(Clone, Default)]
struct Buf(Arc<Mutex<Vec<u8>>>);

impl std::io::Write for Buf {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(b);
        Ok(b.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
fn test_a_supervisor_round_that_raises_does_not_end_the_loop() {
    let buf = Buf::default();
    let w = buf.clone();
    let sub = tracing_subscriber::fmt()
        .with_ansi(false)
        .with_writer(move || w.clone())
        .finish();
    let _guard = tracing::subscriber::set_default(sub);
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let n = Arc::clone(&calls);
    rt.block_on(async move {
        let flaky = move || {
            let n = Arc::clone(&n);
            async move {
                if n.fetch_add(1, Ordering::SeqCst) == 0 {
                    return Err(internal("boom").into());
                }
                Ok(())
            }
        };
        let task = tokio::spawn(ec_hub::app::run_loop(flaky, Duration::from_millis(10)));
        let deadline = Instant::now() + Duration::from_secs(5);
        while calls.load(Ordering::SeqCst) <= 2 && Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        task.abort();
        let _ = task.await;
        assert!(
            calls.load(Ordering::SeqCst) > 2,
            "the loop stopped after a failed round"
        );
    });
    let text = String::from_utf8(buf.0.lock().unwrap().clone()).unwrap();
    assert!(text.contains("background loop error"), "{text}");
}

// ---- the entry point -------------------------------------------------------------------------------------------------------------------
const FAKES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../parity/fakes/bin");

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

/// The binary with every path in `t`, nothing inherited (PATH = the parity fakes, then /usr/bin:/bin), the
/// fleet/discovery/loops/OTLP off, Hugging Face and OTLP on a closed port and an ephemeral port, so a run that
/// unexpectedly serves can never bind a live console's port or reach a live service.
fn console(t: &Path, port: u16) -> Command {
    let mut cmd = Command::new(BIN);
    std::fs::create_dir_all(t.join("hf")).unwrap();
    cmd.env_clear()
        .env("PATH", format!("{FAKES}:/usr/bin:/bin"))
        .env("HOME", t)
        .env("XDG_CONFIG_HOME", t.join("xdg-config"))
        .env("XDG_DATA_HOME", t.join("xdg-data"))
        .env("XDG_STATE_HOME", t.join("xdg-state"))
        .env("XDG_CACHE_HOME", t.join("xdg-cache"))
        .env("XDG_RUNTIME_DIR", t.join("xdg-run"))
        .env("ENGINE_CONSOLE_ENV_FILE", t.join("no-env-file"))
        .env("ENGINE_CONSOLE_HOST", "127.0.0.1")
        .env("ENGINE_CONSOLE_PORT", port.to_string())
        .env("ENGINE_CONSOLE_DATA_DIR", t.join("data"))
        .env("HF_CACHE_DIR", t.join("hf"))
        .env("HF_TOKEN_FILE", t.join("no-token"))
        .env("HOSTS_FILE", t.join("hosts.json"))
        .env("HUB_KEY_FILE", t.join("no-key"))
        .env("FLEET_ENABLED", "false")
        .env("DISCOVERY_ENABLED", "false")
        .env("BACKGROUND_TASKS", "false")
        .env("OTLP_ENABLED", "false")
        .env("OTLP_ENDPOINT", "127.0.0.1:9")
        .env("HF_ENDPOINT", "https://127.0.0.1:9")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    cmd
}

/// Run to completion (killed after 20 s: a run that serves instead of exiting fails the test, it does not hang it).
fn run(mut cmd: Command) -> (i32, String, String) {
    let mut child = cmd.spawn().unwrap();
    let deadline = Instant::now() + Duration::from_secs(20);
    let status = loop {
        if let Some(s) = child.try_wait().unwrap() {
            break s;
        }
        if Instant::now() > deadline {
            let _ = child.kill();
            let _ = child.wait();
            panic!("the console did not exit");
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    let (mut out, mut err) = (String::new(), String::new());
    child
        .stdout
        .take()
        .unwrap()
        .read_to_string(&mut out)
        .unwrap();
    child
        .stderr
        .take()
        .unwrap()
        .read_to_string(&mut err)
        .unwrap();
    (status.code().unwrap_or(-1), out, err)
}

#[test]
fn test_the_hosts_subcommand_goes_to_the_hosts_cli_with_its_own_arguments() {
    let t = tempfile::tempdir().unwrap();
    // the hosts CLI parses what follows `hosts`: its own usage error (argparse's exit 2) proves the hand-off
    let mut cmd = console(t.path(), free_port());
    cmd.args(["hosts", "check", "--timeout", "x"]);
    let (code, _, err) = run(cmd);
    assert_eq!(code, 2);
    assert!(
        err.contains("usage: engine-console hosts check [-h] [--timeout TIMEOUT]"),
        "{err}"
    );
    let mut cmd = console(t.path(), free_port());
    cmd.args(["hosts", "list"]);
    let (code, out, _) = run(cmd);
    assert_eq!(code, 0);
    assert!(
        out.contains("(not present: only this machine is managed)"),
        "{out}"
    );
    assert!(
        !t.path().join("data").join("console.db").exists(),
        "the hosts CLI must not start the console"
    );
}

#[test]
fn test_a_hosts_file_the_console_refuses_stops_it_with_a_message_and_status_2() {
    let t = tempfile::tempdir().unwrap();
    let hosts = t.path().join("hosts.json");
    std::fs::write(&hosts, r#"{"hosts": []}"#).unwrap();
    std::fs::set_permissions(&hosts, std::fs::Permissions::from_mode(0o664)).unwrap();
    // the fleet reads hosts.json only when it is on (Python's FleetService too); the file is refused before any link
    let mut cmd = console(t.path(), free_port());
    cmd.env("FLEET_ENABLED", "true");
    let (code, _, err) = run(cmd);
    assert_eq!(code, 2, "{err}");
    // Python's message names the reason ("group-writable"): the hosts.json checks have their own tests
    assert!(err.starts_with("engine-console: "), "{err}");
}

fn health(port: u16) -> Option<String> {
    let mut s = std::net::TcpStream::connect(("127.0.0.1", port)).ok()?;
    s.set_read_timeout(Some(Duration::from_secs(2))).ok()?;
    let req =
        format!("GET {P}/health HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n");
    s.write_all(req.as_bytes()).ok()?;
    let mut out = String::new();
    s.read_to_string(&mut out).ok()?;
    Some(out)
}

#[test]
fn test_serving_starts_uvicorn_with_the_configured_address() {
    let d = ec_hub::config::Settings::defaults();
    assert_eq!((d.host.as_str(), d.port), ("127.0.0.1", 8791));
    let t = tempfile::tempdir().unwrap();
    let port = free_port();
    let mut child = console(t.path(), port)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(20);
    let mut answer = None;
    while Instant::now() < deadline {
        if let Some(a) = health(port).filter(|a| a.starts_with("HTTP/1.1 200")) {
            answer = Some(a);
            break;
        }
        if child.try_wait().unwrap().is_some() {
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    // stop our own child (SIGTERM, the graceful path), then make sure it is gone
    let pid = nix::unistd::Pid::from_raw(i32::try_from(child.id()).unwrap());
    let _ = nix::sys::signal::kill(pid, nix::sys::signal::Signal::SIGTERM);
    let stop = Instant::now() + Duration::from_secs(10);
    let status = loop {
        if let Some(s) = child.try_wait().unwrap() {
            break Some(s);
        }
        if Instant::now() > stop {
            let _ = child.kill();
            let _ = child.wait();
            break None;
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    let answer = answer.expect("the console answered /api/v1/health on the configured address");
    assert!(answer.contains("\"status\":\"ok\""), "{answer}");
    assert_eq!(
        status.and_then(|s| s.code()),
        Some(0),
        "SIGTERM is a clean shutdown"
    );
}
