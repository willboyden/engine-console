//! Port of backend/tests/fleet/test_cli.py: `engine-console hosts ...`: list, check (against the fake node, real
//! subprocess) and init-key (real ssh-keygen).
//!
//! Python calls `hosts_cli(argv, cfg=, out=)`; here the real `engine-console` binary runs with a throwaway HOME and
//! the settings as environment variables, so its exit code, stdout and stderr are what the fleet installer sees.

#![allow(
    clippy::type_complexity,
    clippy::expect_used,
    clippy::panic,
    clippy::unwrap_used,
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::double_ended_iterator_last,
    clippy::assertions_on_constants,
    clippy::needless_pass_by_value,
    clippy::float_cmp
)]

#[path = "common/wsb_fake_node.rs"]
mod wsb_fake_node;

#[path = "common/wsb_harness.rs"]
mod wsb_harness;

use std::os::unix::fs::PermissionsExt as _;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use ec_hub::services::hosts::key_line;
use serde_json::{Value, json};
use wsb_fake_node::{node_argv, read_log};
use wsb_harness::alive;

fake_node_entry!();

const BIN: &str = env!("CARGO_BIN_EXE_engine-console");

struct Env {
    dir: tempfile::TempDir,
    vars: Vec<(String, String)>,
}

impl Env {
    /// conftest `make_settings`: every path inside the temp dir.
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();
        std::fs::create_dir_all(p.join("home")).unwrap();
        let vars = vec![
            (
                "PATH".to_string(),
                std::env::var("PATH").unwrap_or_else(|_| "/usr/bin:/bin".into()),
            ),
            ("HOME".into(), p.join("home").display().to_string()),
            ("LANG".into(), "C.UTF-8".into()),
            (
                "ENGINE_CONSOLE_ENV_FILE".into(),
                p.join("no-env-file").display().to_string(),
            ),
            (
                "ENGINE_CONSOLE_DATA_DIR".into(),
                p.join("data").display().to_string(),
            ),
            ("HF_CACHE_DIR".into(), p.join("hf").display().to_string()),
            (
                "HF_TOKEN_FILE".into(),
                p.join("no-token").display().to_string(),
            ),
            (
                "HOSTS_FILE".into(),
                p.join("hosts.json").display().to_string(),
            ),
            (
                "HUB_KEY_FILE".into(),
                p.join("ssh").join("hub-key").display().to_string(),
            ),
            (
                "REMOTE_NODE_COMMAND".into(),
                "/usr/local/lib/ecnode/ecnode --stdio".into(),
            ),
            ("FLEET_ENABLED".into(), "true".into()),
            ("OTLP_ENABLED".into(), "false".into()),
        ];
        Self { dir, vars }
    }

    fn set(&mut self, k: &str, v: impl Into<String>) {
        self.vars.retain(|(kk, _)| kk != k);
        self.vars.push((k.into(), v.into()));
    }

    fn path(&self, name: &str) -> PathBuf {
        self.dir.path().join(name)
    }

    fn var(&self, k: &str) -> String {
        self.vars
            .iter()
            .find(|(kk, _)| kk == k)
            .map(|(_, v)| v.clone())
            .unwrap_or_default()
    }

    fn hub_key(&self) -> PathBuf {
        PathBuf::from(self.var("HUB_KEY_FILE"))
    }

    fn hosts_file(&self) -> PathBuf {
        PathBuf::from(self.var("HOSTS_FILE"))
    }

    fn node(&mut self, scenario: &Value) {
        self.set(
            "NODE_COMMAND",
            serde_json::to_string(&node_argv(scenario)).unwrap(),
        );
    }

    /// `engine-console hosts <args>`: (exit code, stdout, stderr).
    fn run(&self, args: &[&str]) -> (i32, String, String) {
        let out = std::process::Command::new(BIN)
            .arg("hosts")
            .args(args)
            .env_clear()
            .envs(self.vars.iter().map(|(k, v)| (k.as_str(), v.as_str())))
            .stdin(std::process::Stdio::null())
            .output()
            .unwrap();
        (
            out.status.code().unwrap_or(-1),
            String::from_utf8_lossy(&out.stdout).into_owned(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        )
    }
}

fn write_hosts(path: &Path, hosts: &Value) -> PathBuf {
    std::fs::write(path, json!({"hosts": hosts}).to_string()).unwrap();
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600)).unwrap();
    path.to_path_buf()
}

fn words(line: &str) -> Vec<&str> {
    line.split_whitespace().collect()
}

/// The hub finds ssh-keygen on its fixed tool path, never on PATH (tools.rs): so does this check.
fn has_keygen() -> bool {
    ec_hub::tools::resolve("ssh-keygen").is_some()
}

// ── list ───────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn test_list_without_a_file_shows_this_machine() {
    let env = Env::new();
    let (code, out, _) = env.run(&["list"]);
    assert!(
        code == 0 && out.contains("not present: only this machine is managed"),
        "{out}"
    );
    let rows: Vec<&str> = out.lines().collect();
    assert_eq!(words(rows[1]), ["ID", "LABEL", "REACHED", "BY", "ENABLED"]);
    assert!(
        rows[2].starts_with("local")
            && rows[2].contains("this machine")
            && rows[2].trim_end().ends_with("yes")
    );
}

#[test]
fn test_list_shows_how_each_host_is_reached() {
    let mut env = Env::new();
    let hf = write_hosts(
        &env.path("hosts.json"),
        &json!([
        {"id": "local", "label": "Workstation"},
        {"id": "gpu-box-2", "label": "GPU box 2", "ssh": "ops@10.0.0.5", "port": 2222},
        {"id": "edge", "ssh": "edge-alias", "ssh_config": true, "enabled": false}]),
    );
    env.set("HOSTS_FILE", hf.display().to_string());
    let (code, out, _) = env.run(&["list"]);
    assert!(code == 0 && !out.contains("not present"));
    let lines: Vec<&str> = out.lines().collect();
    assert!(lines.iter().any(|l| l.starts_with("gpu-box-2")
        && l.contains("ssh ops@10.0.0.5 -p 2222")
        && l.trim_end().ends_with("yes")));
    assert!(lines.iter().any(|l| l.starts_with("edge")
        && l.contains("(ssh_config)")
        && l.trim_end().ends_with("no")));
}

#[test]
fn test_a_bad_hosts_file_is_an_error_on_stderr_and_a_failing_exit_code() {
    let mut env = Env::new();
    let bad = write_hosts(&env.path("hosts.json"), &json!([]));
    env.set("HOSTS_FILE", bad.display().to_string());
    let (code, out, err) = env.run(&["list"]);
    assert!(code == 1 && out.is_empty());
    assert!(
        err.starts_with("engine-console: ")
            && err.contains(&bad.display().to_string())
            && err.contains("1-32"),
        "{err}"
    );
}

#[test]
fn test_usage_errors_return_two_and_help_returns_zero() {
    let env = Env::new();
    assert_eq!(env.run(&[]).0, 2);
    assert_eq!(env.run(&["frobnicate"]).0, 2);
    assert_eq!(env.run(&["init-key"]).0, 2); // --from is required
    let (code, out, _) = env.run(&["--help"]);
    assert!(code == 0 && out.contains("init-key"));
}

// ── init-key ───────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn test_init_key_creates_the_key_and_prints_the_one_line() {
    if !has_keygen() {
        return;
    }
    let env = Env::new();
    let key = env.hub_key();
    let (code, out, err) = env.run(&["init-key", "--from", "10.0.0.5"]);
    assert!(
        code == 0
            && out.contains(&format!(
                "created {} (private half 0600, never printed)",
                key.display()
            )),
        "{out}{err}"
    );
    let pubtext = std::fs::read_to_string(format!("{}.pub", key.display())).unwrap();
    let line = key_line(&pubtext, Some("10.0.0.5"), &env.var("REMOTE_NODE_COMMAND")).unwrap();
    assert!(out.lines().any(|l| l == line));
    assert!(line.contains(
        "restrict,from=\"10.0.0.5\",command=\"/usr/local/lib/ecnode/ecnode --stdio\" ssh-ed25519 "
    ));
    // a throwaway key made by this test in its own temp dir: no part of it may be printed
    let private = std::fs::read_to_string(&key).unwrap();
    assert!(!out.contains("PRIVATE KEY") && !out.contains(private.lines().nth(1).unwrap()));
    assert_eq!(
        std::fs::metadata(&key).unwrap().permissions().mode() & 0o777,
        0o600
    );
    assert!(
        out.contains(&format!(
            "Then list the machines in {}",
            env.hosts_file().display()
        )) && !out.contains("WARNING")
    );
}

#[test]
fn test_init_key_twice_reuses_the_key() {
    if !has_keygen() {
        return;
    }
    let env = Env::new();
    let (_, first, _) = env.run(&["init-key", "--from", "10.0.0.5"]);
    let before = std::fs::read(env.hub_key()).unwrap();
    let (code, second, _) = env.run(&["init-key", "--from", "10.0.0.5"]);
    assert!(
        code == 0
            && second.contains(&format!(
                "using the existing key {}",
                env.hub_key().display()
            ))
    );
    assert_eq!(std::fs::read(env.hub_key()).unwrap(), before);
    let restrict = |t: &str| {
        t.lines()
            .filter(|l| l.starts_with("restrict"))
            .map(str::to_string)
            .collect::<Vec<_>>()
    };
    assert_eq!(restrict(&first), restrict(&second));
}

#[test]
fn test_init_key_honours_the_key_file_option_and_the_configured_remote_command() {
    if !has_keygen() {
        return;
    }
    let mut env = Env::new();
    env.set("REMOTE_NODE_COMMAND", "/opt/ecnode/bin/ecnode --stdio");
    let other = env.path("elsewhere").join("key");
    let (code, out, _) = env.run(&[
        "init-key",
        "--from",
        "10.0.0.0/24,10.1.0.5",
        "--key-file",
        &other.display().to_string(),
    ]);
    assert!(code == 0 && other.exists() && !env.hub_key().exists());
    assert!(
        out.contains("from=\"10.0.0.0/24,10.1.0.5\",command=\"/opt/ecnode/bin/ecnode --stdio\"")
    );
}

#[test]
fn test_a_wildcard_from_is_allowed_but_warned_about() {
    if !has_keygen() {
        return;
    }
    let env = Env::new();
    let (code, out, _) = env.run(&["init-key", "--from", "*"]);
    assert!(code == 0 && out.contains("WARNING") && out.contains("from=\"*\""));
}

#[test]
fn test_init_key_refuses_a_from_that_could_break_out_of_its_quotes() {
    let env = Env::new();
    let (code, out, err) = env.run(&["init-key", "--from", "x\",command=\"evil"]);
    assert!(
        code == 1 && out.is_empty() && err.contains("--from"),
        "{err}"
    );
    // nothing was created
    assert!(!env.hub_key().exists() && !env.hub_key().parent().unwrap().exists());
}

// ── check ──────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn test_check_reports_a_working_node() {
    let mut env = Env::new();
    let hf = write_hosts(
        &env.path("hosts.json"),
        &json!([{"id": "local", "label": "Workstation"}, {"id": "off", "ssh": "ops@10.0.0.9", "enabled": false}]),
    );
    env.set("HOSTS_FILE", hf.display().to_string());
    env.node(&json!({"hello_patch": {"policy": {"mode": "manage"}}}));
    let t0 = Instant::now();
    let (code, out, err) = env.run(&["check"]);
    assert!(
        code == 0 && t0.elapsed() < Duration::from_secs(15),
        "{out}{err}"
    );
    let rows: Vec<Vec<&str>> = out.lines().map(words).collect();
    assert_eq!(rows[0], ["HOST", "STATE", "NODE", "ARCH", "GPUS", "POLICY"]);
    assert_eq!(
        rows[1],
        ["local", "up", "0.4.0-test", "x86_64", "2", "manage"]
    );
    assert_eq!(rows[2][..2], ["off", "disabled"]); // a disabled host is not a failure
    assert!(out.contains("off: disabled in hosts.json"));
}

#[test]
fn test_check_reports_a_node_that_fails_and_exits_nonzero() {
    let mut env = Env::new();
    let hf = write_hosts(&env.path("hosts.json"), &json!([{"id": "local"}]));
    env.set("HOSTS_FILE", hf.display().to_string());
    env.node(&json!({"stderr": "Traceback...\nValueError: boom", "exit_before_hello": 1}));
    let (code, out, _) = env.run(&["check"]);
    assert_eq!(code, 1);
    assert_eq!(words(out.lines().nth(1).unwrap())[..2], ["local", "down"]);
    assert!(
        out.contains("local: the node exited (1): ValueError: boom"),
        "{out}"
    );
}

#[test]
fn test_check_reports_a_version_mismatch_in_words() {
    let mut env = Env::new();
    let hf = write_hosts(
        &env.path("hosts.json"),
        &json!([{"id": "local", "label": "Workstation"}]),
    );
    env.set("HOSTS_FILE", hf.display().to_string());
    env.node(&json!({"proto": 2}));
    let (code, out, _) = env.run(&["check"]);
    assert!(
        code == 1 && out.contains("incompatible") && out.contains("Update the Engine Console hub"),
        "{out}"
    );
}

#[test]
fn test_check_prints_a_policy_problem_and_notes() {
    let mut env = Env::new();
    let hf = write_hosts(&env.path("hosts.json"), &json!([{"id": "local"}]));
    env.set("HOSTS_FILE", hf.display().to_string());
    env.node(&json!({"junk": "Welcome\n", "hello_patch": {"policy": {"error": "policy.json is group-writable"}}}));
    let (code, out, _) = env.run(&["check"]);
    assert!(
        code == 0 && out.contains("node policy: policy.json is group-writable"),
        "{out}"
    );
    assert!(out.contains("local: note: ignored 8 byte(s)"));
    assert_eq!(
        *words(out.lines().nth(1).unwrap()).last().unwrap(),
        "monitor"
    );
}

#[test]
fn test_check_gives_up_waiting_after_the_timeout_and_cleans_up() {
    let mut env = Env::new();
    let hf = write_hosts(&env.path("hosts.json"), &json!([{"id": "local"}]));
    env.set("HOSTS_FILE", hf.display().to_string());
    let log = env.path("slow.jsonl");
    env.node(&json!({"hello_delay_s": 30, "log": log.display().to_string()}));
    let t0 = Instant::now();
    let (code, out, _) = env.run(&["check", "--timeout", "1"]);
    assert!(code == 1 && t0.elapsed() < Duration::from_secs(12));
    assert_eq!(
        words(out.lines().nth(1).unwrap())[..2],
        ["local", "connecting"]
    );
    assert!(out.contains("local: no answer yet"), "{out}");
    let pid = read_log(&log)
        .iter()
        .find(|e| e["ev"] == "start")
        .and_then(|e| e["pid"].as_i64())
        .unwrap();
    assert!(!alive(pid)); // the silent node was killed on the way out
}

#[test]
fn test_check_with_a_bad_hosts_file_fails_cleanly() {
    let mut env = Env::new();
    let bad = write_hosts(&env.path("hosts.json"), &json!([]));
    env.set("HOSTS_FILE", bad.display().to_string());
    let (code, _, err) = env.run(&["check"]);
    assert!(code == 1 && err.contains("engine-console: "));
}

#[test]
fn test_check_with_the_fleet_disabled_reports_every_host_disabled() {
    let mut env = Env::new();
    env.set("FLEET_ENABLED", "false");
    let (code, out, _) = env.run(&["check"]);
    assert_eq!(code, 0);
    assert_eq!(
        words(out.lines().nth(1).unwrap())[..2],
        ["local", "disabled"]
    );
}
