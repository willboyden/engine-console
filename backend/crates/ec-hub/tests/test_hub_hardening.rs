//! Hub hardening (findings 3 to 6): background loops survive panics, the panic hook masks secrets, child output is
//! bounded while it is read, tools come from the fixed tool path, and the hub's own file readers never hang on a FIFO
//! or follow a symlink they should not. Rust-native: no Python test file.
//!
//! Every process this file starts is a shell script in a temp dir, the real `ssh-keygen` writing into a temp dir, or
//! this test binary itself (child mode, selected by `EC_HARDENING_CHILD`); nothing reaches the network or a lab port.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::too_many_lines
)]

#[path = "common/kit.rs"]
mod kit;

use std::os::unix::fs::PermissionsExt as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use ec_hub::domain::fleet::{HostConfig, NodeClient};
use ec_hub::services::common::EventBus;
use ec_hub::services::fleet::{LinkLike, LinkTiming, NodeLink};
use kit::TestEnv;

const CHILD: &str = "EC_HARDENING_CHILD";

/// Tests that write a script and run it, or start processes, take turns: a script written while another thread forks
/// is still open for writing in that fork for a moment, and exec'ing it then fails with ETXTBSY ("Text file busy").
static SERIAL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

fn mkfifo(p: &Path) {
    nix::unistd::mkfifo(p, nix::sys::stat::Mode::from_bits_truncate(0o600)).unwrap();
}

fn script(dir: &Path, name: &str, body: &str) -> PathBuf {
    std::fs::create_dir_all(dir).unwrap();
    let p = dir.join(name);
    std::fs::write(&p, format!("#!/bin/sh\n{body}\n")).unwrap();
    std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
    p
}

/// Run one test of this binary in a child process with `env` added; its stdout and stderr together.
fn run_child(name: &str, env: &[(&str, String)]) -> (bool, String) {
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", name, "--nocapture", "--test-threads=1"])
        .env(CHILD, "1")
        .env_remove("ENGINE_CONSOLE_TOOL_PATH")
        .envs(env.iter().map(|(k, v)| (*k, v.as_str())))
        .stdin(std::process::Stdio::null())
        .output()
        .unwrap();
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    (out.status.success() && text.contains("1 passed"), text)
}

fn is_child() -> bool {
    std::env::var_os(CHILD).is_some()
}

// ── finding 3: panics ─────────────────────────────────────────────────────────────────────────────────────────
/// The supervisor's loop (main.py `_loop`): a tick that panics is logged and the next tick runs.
#[tokio::test]
async fn test_a_panicking_tick_does_not_stop_the_next_one() {
    let n = Arc::new(AtomicUsize::new(0));
    let n2 = Arc::clone(&n);
    let task = tokio::spawn(ec_hub::app::run_loop(
        move || {
            let n = Arc::clone(&n2);
            async move {
                let i = n.fetch_add(1, Ordering::SeqCst);
                if i.is_multiple_of(2) {
                    panic!("tick {i} failed holding ec_abcdefghijklmnopqrstu");
                }
                Ok(())
            }
        },
        Duration::from_millis(5),
    ));
    let t = Instant::now();
    while n.load(Ordering::SeqCst) < 6 {
        assert!(t.elapsed() < Duration::from_secs(10), "the loop stopped");
        assert!(!task.is_finished(), "the loop task died");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    task.abort();
}

/// The process hook writes a panic with its secrets masked, and keeps an adapter's Python exception off stderr.
#[test]
fn test_the_panic_hook_masks_secrets_and_keeps_python_exceptions_quiet() {
    let _turn = SERIAL.blocking_lock();
    if is_child() {
        ec_hub::panics::install_panic_hook();
        let _ = std::thread::Builder::new()
            .name("worker".into())
            .spawn(|| {
                panic!("upstream refused Bearer abcdefgh12345678 and hf_abcdefghijklmnop for sk-abcdefghijklmnopqr")
            })
            .unwrap()
            .join();
        let _ = std::thread::spawn(|| {
            ec_hub::adapters::pycompat::raise("ValueError", "quiet-python-exception")
        })
        .join();
        return;
    }
    let (ok, text) = run_child(
        "test_the_panic_hook_masks_secrets_and_keeps_python_exceptions_quiet",
        &[],
    );
    assert!(ok, "{text}");
    assert!(
        text.contains("thread 'worker' panicked at ") && text.contains("[redacted]"),
        "{text}"
    );
    for secret in [
        "abcdefgh12345678",
        "hf_abcdefghijklmnop",
        "sk-abcdefghijklmnopqr",
    ] {
        assert!(!text.contains(secret), "{secret} reached stderr:\n{text}");
    }
    assert!(!text.contains("quiet-python-exception"), "{text}");
}

// ── finding 5: child output ───────────────────────────────────────────────────────────────────────────────────
fn host() -> HostConfig {
    HostConfig {
        id: "local".into(),
        label: "This machine".into(),
        ssh: None,
        port: None,
        ssh_config: false,
        enabled: true,
    }
}

fn timing() -> LinkTiming {
    LinkTiming {
        hello_timeout_s: 10.0,
        backoff_min_s: 30.0,
        backoff_max_s: 60.0,
        exit_grace_s: 2.0,
        ..LinkTiming::default()
    }
}

/// Start a link running `argv` with `env`; return its first error once it is down.
async fn first_failure(argv: Vec<String>, env: Option<Vec<(String, String)>>) -> String {
    let link = NodeLink::new(
        host(),
        NodeLink::fixed_argv(argv),
        env,
        Arc::new(EventBus::new()),
        timing(),
        None,
        Vec::new(),
    );
    link.start().await;
    let t = Instant::now();
    let err = loop {
        let l = link.link();
        if l.state == "down" {
            break l.error.unwrap_or_default();
        }
        assert!(t.elapsed() < Duration::from_secs(30), "still {}", l.state);
        tokio::time::sleep(Duration::from_millis(20)).await;
    };
    link.stop().await;
    err
}

fn sh(body: &str) -> Vec<String> {
    vec!["/bin/sh".into(), "-c".into(), body.into()]
}

fn plain_env() -> Option<Vec<(String, String)>> {
    Some(vec![("PATH".into(), "/usr/bin:/bin".into())])
}

/// 64 MiB on the node's stderr: only the last 2048 bytes are ever held (the tail is cut as the bytes arrive), and its
/// last line still explains the drop.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_a_node_flooding_stderr_is_bounded_while_read() {
    let _turn = SERIAL.lock().await;
    let err = first_failure(
        sh("head -c 67108864 /dev/zero | tr '\\000' x >&2; echo >&2; echo 'the actual reason' >&2; exit 4"),
        plain_env(),
    )
    .await;
    assert_eq!(err, "the node exited (4): the actual reason");
}

/// Endless shell noise on stdout before the protocol starts: the preamble limit ends it (never read whole).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_a_node_flooding_stdout_before_the_protocol_is_cut_off() {
    let _turn = SERIAL.lock().await;
    let t = Instant::now();
    let err = first_failure(sh("exec yes"), plain_env()).await;
    assert!(err.starts_with("protocol error from the node"), "{err}");
    assert!(t.elapsed() < Duration::from_secs(15));
}

/// A frame header announcing 2 GiB, then endless bytes: refused from the header, before any of it is buffered.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_an_oversized_frame_is_refused_from_its_header() {
    let _turn = SERIAL.lock().await;
    let t = Instant::now();
    let err = first_failure(
        sh("printf 'ecnode-proto 1\\n\\177\\377\\377\\377\\001'; exec cat /dev/zero"),
        plain_env(),
    )
    .await;
    assert!(err.starts_with("protocol error from the node"), "{err}");
    assert!(t.elapsed() < Duration::from_secs(15));
}

/// ssh-keygen flooding its output: `init-key` reads at most the cap of each stream and still answers.
#[test]
fn test_a_flooding_ssh_keygen_is_capped_while_read() {
    let _turn = SERIAL.blocking_lock();
    let d = tempfile::tempdir().unwrap();
    let keygen = script(
        &d.path().join("bin"),
        "keygen",
        "head -c 33554432 /dev/zero | tr '\\000' y; head -c 33554432 /dev/zero >&2; exit 1",
    );
    let t = Instant::now();
    let e = ec_hub::services::hosts::init_key_with(
        &d.path().join("ssh").join("hub-key"),
        Some("203.0.113.5"),
        ec_hub::services::hosts::DEFAULT_REMOTE_COMMAND,
        &keygen.to_string_lossy(),
        30.0,
    )
    .unwrap_err();
    assert!(e.to_string().starts_with("ssh-keygen failed: "), "{e}");
    assert!(
        e.to_string().len() < 400,
        "the message is cut: {}",
        e.to_string().len()
    );
    assert!(t.elapsed() < Duration::from_secs(20));
}

// ── finding 6: tools come from the tool path ─────────────────────────────────────────────────────────────────
/// In a child whose PATH holds fake `ssh-keygen` and a fake node (and no ENGINE_CONSOLE_TOOL_PATH), the hub uses
/// neither: the node command is looked up on the fixed tool path, ssh-keygen too.
#[test]
fn test_the_callers_path_is_never_searched_for_tools() {
    let _turn = SERIAL.blocking_lock();
    let d = tempfile::tempdir().unwrap();
    let bin = d.path().join("bin");
    let marker = d.path().join("ran.log");
    script(
        &bin,
        "ssh-keygen",
        &format!("echo keygen >> '{}'; exit 1", marker.display()),
    );
    script(
        &bin,
        "ec-fake-node",
        &format!("echo node >> '{}'; exit 3", marker.display()),
    );
    let (ok, text) = run_child(
        "child_tools_from_the_tool_path",
        &[
            ("PATH", format!("{}:/usr/bin:/bin", bin.display())),
            ("EC_HARDENING_DIR", d.path().display().to_string()),
        ],
    );
    assert!(ok, "{text}");
    assert!(
        !marker.exists(),
        "a tool from PATH ran: {:?}",
        std::fs::read_to_string(&marker)
    );
}

#[test]
fn child_tools_from_the_tool_path() {
    if !is_child() {
        return;
    }
    let dir = PathBuf::from(std::env::var("EC_HARDENING_DIR").unwrap());
    assert_eq!(ec_hub::tools::tool_path(), ec_hub::tools::SAFE_PATH);
    assert_eq!(ec_hub::tools::resolve("ec-fake-node"), None);
    // the local node's environment carries the tool path, not this process's PATH
    let env = ec_hub::services::hosts::local_env();
    assert!(env.contains(&("PATH".to_string(), ec_hub::tools::SAFE_PATH.to_string())));
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    // a link with no explicit environment: the bare name is not found on the tool path
    let err = rt.block_on(first_failure(vec!["ec-fake-node".into()], None));
    assert_eq!(
        err,
        "could not start ec-fake-node: No such file or directory"
    );
    // with the hub's own environment (local_env): the same
    let err = rt.block_on(first_failure(vec!["ec-fake-node".into()], Some(env)));
    assert!(err.starts_with("could not start ec-fake-node"), "{err}");
    // ssh-keygen by name: the fake on PATH never runs (the real one on the tool path writes into the temp dir)
    let _ = ec_hub::services::hosts::init_key(
        &dir.join("ssh").join("hub-key"),
        Some("203.0.113.5"),
        ec_hub::services::hosts::DEFAULT_REMOTE_COMMAND,
    );
}

/// An explicit environment (a test rig, or ENGINE_CONSOLE_TOOL_PATH through local_env) is where a bare name is
/// found: the child's own PATH, which in production is the tool path.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_a_bare_node_command_is_found_on_the_childs_path() {
    let _turn = SERIAL.lock().await;
    let d = tempfile::tempdir().unwrap();
    let bin = d.path().join("bin");
    script(&bin, "ec-fake-node", "echo 'fake node here' >&2; exit 3");
    let env = Some(vec![(
        "PATH".to_string(),
        format!("{}:/usr/bin:/bin", bin.display()),
    )]);
    let err = first_failure(vec!["ec-fake-node".into()], env).await;
    assert_eq!(err, "the node exited (3): fake node here");
}

// ── finding 4: special files ──────────────────────────────────────────────────────────────────────────────────
#[test]
fn test_the_env_file_is_never_a_fifo_or_a_huge_file() {
    let d = tempfile::tempdir().unwrap();
    let fifo = d.path().join("env");
    mkfifo(&fifo);
    let t = Instant::now();
    assert!(ec_hub::config::read_env_file(&fifo, &[]).is_empty());
    assert!(t.elapsed() < Duration::from_secs(2));
    // a symlink to a real env file is followed (Python's dotenv opens the path)
    let real = d.path().join("real-env");
    std::fs::write(&real, "HF_ENDPOINT=https://example.invalid\n").unwrap();
    let link = d.path().join("link-env");
    std::os::unix::fs::symlink(&real, &link).unwrap();
    assert_eq!(
        ec_hub::config::read_env_file(&link, &[]),
        [(
            "HF_ENDPOINT".to_string(),
            "https://example.invalid".to_string()
        )]
    );
    let big = d.path().join("big-env");
    std::fs::write(&big, "A=1\n".repeat(ec_hub::config::MAX_ENV_FILE / 4 + 1)).unwrap();
    assert!(ec_hub::config::read_env_file(&big, &[]).is_empty());
}

#[test]
fn test_secret_files_are_never_followed_or_read_from_a_fifo() {
    use ec_hub::services::secrets_store::SecretStore;
    let d = tempfile::tempdir().unwrap();
    let dir = d.path().join("secrets");
    let store = SecretStore::new(dir.clone());
    store
        .put("inst_a", &[("api_key".into(), "s1".into())])
        .unwrap();
    assert_eq!(
        store.get("inst_a"),
        [("api_key".to_string(), "s1".to_string())]
    );
    // a planted symlink: neither read through nor written through
    let elsewhere = d.path().join("elsewhere.json");
    std::fs::write(&elsewhere, r#"{"api_key": "planted"}"#).unwrap();
    std::os::unix::fs::symlink(&elsewhere, dir.join("inst_b.json")).unwrap();
    assert!(store.get("inst_b").is_empty());
    assert!(
        store
            .put("inst_b", &[("api_key".into(), "s2".into())])
            .is_err()
    );
    assert_eq!(
        std::fs::read_to_string(&elsewhere).unwrap(),
        r#"{"api_key": "planted"}"#
    );
    // a FIFO: refused at once both ways
    mkfifo(&dir.join("inst_c.json"));
    let t = Instant::now();
    assert!(store.get("inst_c").is_empty());
    assert!(
        store
            .put("inst_c", &[("api_key".into(), "s3".into())])
            .is_err()
    );
    assert!(t.elapsed() < Duration::from_secs(2));
}

#[test]
fn test_the_hf_token_file_is_never_a_fifo_but_may_be_a_symlink() {
    let d = tempfile::tempdir().unwrap();
    let token = d.path().join("hf_token");
    let mut env = TestEnv::with(|s| s.hf_token_file = token.clone());
    let c = env.container();
    mkfifo(&token);
    let t = Instant::now();
    assert_eq!(c.settings.hf_token(), None);
    assert_eq!(c.settings.token_status(), "[unset]");
    assert!(t.elapsed() < Duration::from_secs(2));
    std::fs::remove_file(&token).unwrap();
    let real = d.path().join("real-token");
    std::fs::write(&real, "hf_abcdefghijkl\n").unwrap();
    std::fs::set_permissions(&real, std::fs::Permissions::from_mode(0o600)).unwrap();
    std::os::unix::fs::symlink(&real, &token).unwrap();
    assert_eq!(c.settings.hf_token().as_deref(), Some("hf_abcdefghijkl"));
}

#[test]
fn test_a_fifo_as_the_hub_keys_public_half_is_refused_without_hanging() {
    let d = tempfile::tempdir().unwrap();
    let key = d.path().join("ssh").join("hub-key");
    std::fs::create_dir_all(key.parent().unwrap()).unwrap();
    std::fs::write(&key, "stand-in private key\n").unwrap();
    std::fs::set_permissions(&key, std::fs::Permissions::from_mode(0o600)).unwrap();
    mkfifo(&PathBuf::from(format!("{}.pub", key.display())));
    let t = Instant::now();
    let e = ec_hub::services::hosts::init_key_with(
        &key,
        Some("203.0.113.5"),
        ec_hub::services::hosts::DEFAULT_REMOTE_COMMAND,
        "/nonexistent/ssh-keygen",
        5.0,
    )
    .unwrap_err();
    assert!(e.to_string().starts_with("cannot read "), "{e}");
    assert!(t.elapsed() < Duration::from_secs(2));
}
