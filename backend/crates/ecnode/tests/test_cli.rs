//! The `__main__.py` parts of node/tests/test_cli.py (`check` and `policy-template` are in tests/test_cli_check.rs),
//! plus the argparse-compatibility cases of the Rust command line.
#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    unreachable_pub
)]

use std::io::{BufReader, Write};
use std::os::unix::fs::PermissionsExt;
use std::process::{Command, Stdio};
use std::time::Duration;

use ec_proto::{Frame, msg};
use ecnode::entry::{HELP, main_with};

fn run(args: &[&str]) -> (i32, String, String) {
    let args: Vec<String> = args.iter().map(|s| (*s).to_string()).collect();
    let (mut out, mut err) = (Vec::new(), Vec::new());
    let code = main_with(&args, &mut out, &mut err);
    (
        code,
        String::from_utf8(out).unwrap(),
        String::from_utf8(err).unwrap(),
    )
}

const USAGE: &str = "usage: ecnode [-h] [--stdio] [--policy PATH]\n              {check,policy-template,version} ...\n";

/// The parity fakes (`docker`, `nvidia-smi` with two fixed GPUs, `ip`) then /usr/bin:/bin.
const FAKE_TOOLS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../parity/fakes/bin:/usr/bin:/bin"
);

/// A hermetic monitor policy (no discovery probes, no docker), written 0644 explicitly (§A6: under a 002 umask a
/// plain write is group-writable and the node would fall back to the default policy, probes included).
fn hermetic_policy(dir: &std::path::Path, docker: &str) -> std::path::PathBuf {
    let policy = dir.join("policy.json");
    std::fs::write(
        &policy,
        serde_json::json!({"version": 1, "mode": "monitor", "discovery_ports": [],
            "docker": {"binary": docker}})
        .to_string(),
    )
    .unwrap();
    std::fs::set_permissions(&policy, std::fs::Permissions::from_mode(0o644)).unwrap();
    policy
}

/// `ecnode --stdio` with exactly `env` (nothing of this test's own environment).
fn start_node(policy: &std::path::Path, env: &[(&str, String)]) -> std::process::Child {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_ecnode"));
    cmd.arg("--stdio")
        .arg("--policy")
        .arg(policy)
        .env_clear()
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for (k, v) in env {
        cmd.env(k, v);
    }
    cmd.spawn().unwrap()
}

/// The next JSON frame of `kind` (other frames are skipped), within the node's own pace.
fn next_of(r: &mut BufReader<std::process::ChildStdout>, kind: &str) -> serde_json::Value {
    loop {
        let Some(Frame::Json(m)) = ec_proto::read_frame_sync(r).unwrap() else {
            panic!("the node closed its stream before a {kind}")
        };
        if m["type"] == kind {
            return serde_json::Value::Object(m);
        }
    }
}

/// End the node (EOF on stdin) and return its stderr.
fn finish(mut child: std::process::Child, stdin: std::process::ChildStdin) -> String {
    drop(stdin);
    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    while child.try_wait().unwrap().is_none() {
        assert!(
            std::time::Instant::now() < deadline,
            "the node did not exit on EOF"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
    let mut err = String::new();
    std::io::Read::read_to_string(&mut child.stderr.take().unwrap(), &mut err).unwrap();
    err
}

#[test]
fn test_version_and_usage() {
    let (code, out, _) = run(&["version"]);
    assert_eq!(
        (code, out.trim()),
        (0, concat!("ecnode ", env!("CARGO_PKG_VERSION")))
    );
    let (code, _, err) = run(&[]);
    assert_eq!(code, 2);
    assert_eq!(err, HELP);
    let (code, _, err) = run(&["--stdio", "version"]);
    assert_eq!(code, 2);
    assert_eq!(
        err,
        format!("{USAGE}ecnode: error: --stdio takes no command\n")
    );
}

#[test]
fn test_argparse_compatible_help_and_errors() {
    // texts captured from `python -m ecnode ...` (Python 3.12 argparse, 80 columns)
    assert_eq!(run(&["--help"]), (0, HELP.to_string(), String::new()));
    assert_eq!(run(&["-h", "version"]).0, 0);
    assert_eq!(run(&["--he"]).0, 0);
    assert_eq!(
        run(&["bogus"]),
        (
            2,
            String::new(),
            format!(
                "{USAGE}ecnode: error: argument cmd: invalid choice: 'bogus' (choose from check, policy-template, \
                 version)\n"
            )
        )
    );
    assert_eq!(
        run(&["version", "extra"]).2,
        format!("{USAGE}ecnode: error: unrecognized arguments: extra\n")
    );
    assert_eq!(
        run(&["version", "--stdio"]).2,
        format!("{USAGE}ecnode: error: unrecognized arguments: --stdio\n")
    );
    assert_eq!(
        run(&["check", "-x"]).2,
        format!("{USAGE}ecnode: error: unrecognized arguments: -x\n")
    );
    assert_eq!(
        run(&["--stdio", "--policy"]).2,
        format!("{USAGE}ecnode: error: argument --policy: expected one argument\n")
    );
    assert_eq!(
        run(&["check", "--policy"]).2,
        "usage: ecnode check [-h] [--policy PATH]\necnode check: error: argument --policy: expected one argument\n"
    );
    assert_eq!(
        run(&["check", "--help"]).1,
        "usage: ecnode check [-h] [--policy PATH]\n\noptions:\n  -h, --help     show this help message and exit\n  --policy PATH\n"
    );
    assert_eq!(
        run(&["policy-template", "--help"]).1,
        "usage: ecnode policy-template [-h]\n\noptions:\n  -h, --help  show this help message and exit\n"
    );
    assert_eq!(
        run(&["--pol=x", "version"]).1,
        concat!("ecnode ", env!("CARGO_PKG_VERSION"), "\n")
    );
    assert_eq!(
        run(&["--p", "x", "version"]).1,
        concat!("ecnode ", env!("CARGO_PKG_VERSION"), "\n")
    );
    assert_eq!(
        run(&["--std", "version"]).2,
        format!("{USAGE}ecnode: error: --stdio takes no command\n")
    );
}

#[test]
fn test_stdio_child_process_protects_its_stream() {
    let tmp = tempfile::tempdir().unwrap();
    let policy = tmp.path().join("policy.json");
    // hermetic (PORTING.md §A2): no discovery probes (the defaults include 8000, 11434 and 30000), no docker. 0644
    // explicitly: under a 002 umask the file would be group-writable, the node would refuse it and fall back to the
    // default policy, probes included (the hello assertions below make that fail loudly).
    std::fs::write(
        &policy,
        r#"{"version": 1, "mode": "monitor", "discovery_ports": [], "docker": {"binary": "/nonexistent/docker"}}"#,
    )
    .unwrap();
    std::fs::set_permissions(&policy, std::fs::Permissions::from_mode(0o644)).unwrap();
    let bin = env!("CARGO_BIN_EXE_ecnode");
    // a login shell's noise before the node starts, then the node itself
    let mut child = Command::new("/bin/sh")
        .args([
            "-c",
            r#"echo "login banner noise before the node starts"; exec "$0" --stdio --policy "$1""#,
            bin,
        ])
        .arg(&policy)
        .env_clear() // hermetic: nothing of this shell's environment (keys, PATH) reaches the node
        .env("PATH", FAKE_TOOLS)
        .env("ECNODE_TOOL_PATH", FAKE_TOOLS) // the node's tools: the fakes (it never searches PATH)
        .env("LANG", "C.UTF-8")
        .env("ECNODE_NO_NVML", "1") // PORTING.md §A2: GPU data from the fake nvidia-smi, never the real NVML
        .env("ECNODE_TEST_STRAY_PRINT", "1") // debug builds print to stdout on each sample: must land on stderr
        .env("HOME", tmp.path())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());
    let skipped = ec_proto::find_magic_sync(&mut stdout).unwrap();
    assert_eq!(
        skipped,
        b"login banner noise before the node starts\n".len()
    );
    let Some(Frame::Json(hello)) = ec_proto::read_frame_sync(&mut stdout).unwrap() else {
        panic!("no hello")
    };
    assert_eq!(hello["type"], "hello");
    assert_eq!(hello["policy"]["mode"], "monitor");
    // the policy file was accepted (a refused one means the default discovery ports: never in a test)
    assert!(hello["policy"]["error"].is_null(), "{}", hello["policy"]);
    assert_eq!(hello["policy"]["discovery_ports"], serde_json::json!([]));
    stdin
        .write_all(&ec_proto::encode_json(&msg::req(1, "node.info", None)).unwrap())
        .unwrap();
    stdin.flush().unwrap();
    let mut kinds = Vec::new();
    loop {
        let Some(Frame::Json(m)) = ec_proto::read_frame_sync(&mut stdout).unwrap() else {
            panic!("closed")
        };
        let t = m["type"].as_str().unwrap().to_string();
        kinds.push(t.clone());
        if t == "res" {
            assert_eq!(m["ok"], true);
            assert_eq!(m["id"], 1);
            break;
        }
    }
    while !kinds.iter().any(|k| k == "sample") {
        // a sample (with its stray print) arrives intact
        let Some(Frame::Json(m)) = ec_proto::read_frame_sync(&mut stdout).unwrap() else {
            panic!("closed")
        };
        kinds.push(m["type"].as_str().unwrap().to_string());
    }
    drop(stdin); // EOF: the node exits by itself
    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    let status = loop {
        if let Some(s) = child.try_wait().unwrap() {
            break s;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the node did not exit on EOF"
        );
        std::thread::sleep(Duration::from_millis(50));
    };
    assert_eq!(status.code(), Some(0));
    let mut err = String::new();
    std::io::Read::read_to_string(&mut child.stderr.take().unwrap(), &mut err).unwrap();
    assert!(err.contains("a stray print from inside the node"), "{err}");
}

// ── node hardening ───────────────────────────────────────────────────────────────────────────────────────

/// The node never runs a tool from the PATH it was started with: `docker`, `nvidia-smi`, `ip` and `uname` traps
/// first on PATH stay untouched; the tools come from ECNODE_TOOL_PATH (here the parity fakes).
#[test]
fn test_the_node_never_runs_tools_from_its_path() {
    let tmp = tempfile::tempdir().unwrap();
    let traps = tmp.path().join("traps");
    std::fs::create_dir_all(&traps).unwrap();
    let marker = tmp.path().join("trapped");
    for tool in ["docker", "nvidia-smi", "ip", "uname", "date"] {
        let p = traps.join(tool);
        std::fs::write(
            &p,
            format!("#!/bin/sh\necho {tool} >> '{}'\nexit 1\n", marker.display()),
        )
        .unwrap();
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    // a bare "docker" in the policy: resolved in the tool PATH, never PATH
    let policy = hermetic_policy(tmp.path(), "docker");
    let mut child = start_node(
        &policy,
        &[
            ("PATH", format!("{}:/usr/bin:/bin", traps.display())),
            ("ECNODE_TOOL_PATH", FAKE_TOOLS.to_string()),
            ("ECNODE_NO_NVML", "1".into()),
            ("LANG", "C.UTF-8".into()),
            ("HOME", tmp.path().display().to_string()),
        ],
    );
    let stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());
    ec_proto::find_magic_sync(&mut stdout).unwrap();
    let hello = next_of(&mut stdout, "hello");
    assert!(hello["policy"]["error"].is_null(), "{}", hello["policy"]);
    // the fakes answered: two GPUs (fake nvidia-smi), the documentation address (fake ip), the fake daemon
    assert_eq!(hello["gpus"].as_array().map(Vec::len), Some(2), "{hello}");
    assert_eq!(hello["addresses"][0]["addr"], "192.0.2.10", "{hello}");
    assert_eq!(hello["docker"]["version"], "29.6.0-fake", "{hello}");
    // a sample and an inventory went by (in either order: the sampler waits for the first inventory)
    let mut seen = std::collections::BTreeSet::new();
    while seen.len() < 2 {
        let Some(Frame::Json(m)) = ec_proto::read_frame_sync(&mut stdout).unwrap() else {
            panic!("the node closed its stream")
        };
        if let Some(t @ ("sample" | "inventory")) = m["type"].as_str() {
            seen.insert(t.to_string());
        }
    }
    let _ = finish(child, stdin);
    assert!(
        !marker.exists(),
        "a tool ran from PATH: {}",
        std::fs::read_to_string(&marker).unwrap_or_default()
    );
}

/// A panic in a push loop costs one round: the node keeps sampling, and stderr shows the panic redacted (the hook
/// and the loop's report), never the secret in its message.
#[test]
fn test_a_panicking_loop_round_is_redacted_and_the_node_goes_on() {
    const SECRET: &str = "ZQ9VLMT4PR2SXW8KABCDEF";
    let tmp = tempfile::tempdir().unwrap();
    let policy = hermetic_policy(tmp.path(), "/nonexistent/docker");
    let mut child = start_node(
        &policy,
        &[
            ("PATH", FAKE_TOOLS.to_string()),
            ("ECNODE_TOOL_PATH", FAKE_TOOLS.to_string()),
            ("ECNODE_NO_NVML", "1".into()),
            ("ECNODE_TEST_PANIC", SECRET.into()), // debug builds: the first sample panics with it in the message
            ("LANG", "C.UTF-8".into()),
            ("HOME", tmp.path().display().to_string()),
        ],
    );
    let stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());
    ec_proto::find_magic_sync(&mut stdout).unwrap();
    next_of(&mut stdout, "hello");
    // sample 1 panicked; the loop goes on
    let s = next_of(&mut stdout, "sample");
    assert!(s["seq"].as_i64().unwrap() >= 2, "{s}");
    let s2 = next_of(&mut stdout, "sample");
    assert!(s2["seq"].as_i64() > s["seq"].as_i64());
    let err = finish(child, stdin);
    assert!(!err.contains(SECRET), "the secret reached stderr:\n{err}");
    assert!(
        err.contains("ecnode: panic in thread 'ecnode-sampler'"),
        "{err}"
    );
    assert!(
        err.contains(
            "ecnode: sampler: panic: test panic in the sampler: vllm serve --api-key [redacted]"
        ),
        "{err}"
    );
}
