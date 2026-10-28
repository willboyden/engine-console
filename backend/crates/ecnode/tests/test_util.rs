//! node/tests/test_util.py: secret masking, argv redaction, secret-looking names, subprocesses and their
//! environment, time helpers.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::needless_pass_by_value,
    clippy::double_ended_iterator_last,
    clippy::redundant_clone,
    unreachable_pub
)]

#[path = "common/wsd_kit.rs"]
mod wsd_kit;

use std::collections::BTreeSet;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use ecnode::util::{
    self, CmdResult, EnvMap, Guarded, REDACTED, child_env, is_secret_name, parse_time, redact_argv,
    scrub,
};
use serde_json::{Value, json};

const SECRET: &str = "s3cr3tVALUE_abc123";

fn fake_hf() -> String {
    format!("hf_{}", "Fake0".repeat(5))
}

fn fake_nv() -> String {
    format!("nvapi-{}", "Fake0".repeat(5))
}

fn sv(a: &[&str]) -> Vec<String> {
    a.iter().map(|s| (*s).to_string()).collect()
}

#[test]
fn test_redact_argv_masks_every_form() {
    let cases: Vec<Vec<String>> = vec![
        sv(&["vllm", "serve", "m", "--api-key", SECRET]),
        vec![
            "vllm".into(),
            "serve".into(),
            "m".into(),
            format!("--api-key={SECRET}"),
        ],
        sv(&["x", "--hf-token", SECRET]),
        vec!["x".into(), format!("--token={SECRET}")],
        vec!["env".into(), format!("HF_TOKEN={SECRET}"), "python".into()],
        vec!["env".into(), format!("MY_PASSWORD={SECRET}")],
        vec![
            "sh".into(),
            "-c".into(),
            format!("vllm serve m --api-key {SECRET} --port 8000"),
        ],
        vec![
            "sh".into(),
            "-c".into(),
            format!("VLLM_API_KEY={SECRET} exec vllm serve m"),
        ],
        sv(&["x", "--client-secret", SECRET]),
        vec![
            "x".into(),
            format!("--db-url=postgres://admin:{SECRET}@db:5432/x"),
        ],
    ];
    for argv in cases {
        let out = redact_argv(&argv, 400, 1000).join(" ");
        assert!(!out.contains(SECRET), "{out}");
        assert!(out.contains(REDACTED), "{out}");
    }
}

#[test]
fn test_redact_argv_keeps_ordinary_flags_and_values() {
    let argv = sv(&[
        "vllm",
        "serve",
        "org/m",
        "--tokenizer",
        "org/tok",
        "--max-num-batched-tokens",
        "8192",
        "--port",
        "8000",
        "--tensor-parallel-size",
        "2",
    ]);
    assert_eq!(redact_argv(&argv, 400, 1000), argv);
}

#[test]
fn test_redact_argv_bounds_items_and_length() {
    let argv = vec!["x".repeat(5000); 1000];
    let out = redact_argv(&argv, 400, 1024);
    assert_eq!(out.len(), 400);
    assert!(out.iter().all(|t| t.chars().count() <= 1024));
}

#[test]
fn test_scrub_masks_secret_values_in_free_text() {
    for text in [
        "Authorization: Bearer abcdefghijklmnop1234".to_string(),
        format!("non-default args: {{'api_key': ['{SECRET}']}}"),
        format!("{{\"api_key\": \"{SECRET}\"}}"),
        format!("api_key={SECRET}"),
        format!("API_KEY: {SECRET}"),
        format!("--api-key {SECRET}"),
        format!("token {}", fake_hf()),
        format!("using key {}", fake_nv()),
        format!("https://user:{SECRET}@example.invalid/x"),
    ] {
        let out = scrub(&text);
        for s in [
            SECRET.to_string(),
            "abcdefghijklmnop1234".into(),
            fake_hf(),
            fake_nv(),
        ] {
            assert!(!out.contains(&s), "{out}");
        }
    }
}

#[test]
fn test_scrub_leaves_ordinary_log_lines_alone() {
    let line = "INFO 10-04 12:00:00 engine.py:300 max_num_batched_tokens=8192, tokenizer=org/tok, port=8000";
    assert_eq!(scrub(line), line);
}

#[test]
fn test_scrub_is_fast_on_hostile_lines() {
    let t0 = Instant::now();
    scrub(&format!("{}={}", "a-".repeat(4000), "b".repeat(8000)));
    scrub(&format!("{}.", "x".repeat(30)).repeat(300));
    assert!(t0.elapsed() < Duration::from_secs(2));
}

#[test]
fn test_secret_looking_names() {
    for name in [
        "id_rsa",
        "id_ed25519",
        "id_ed25519.pub",
        "server.pem",
        "tls.key",
        ".env",
        ".env.local",
        "token",
        "TOKEN",
        "tokens.json",
        "token.txt",
        "credentials",
        "credentials.json",
        "stored_tokens",
        ".netrc",
        ".ssh",
        ".config",
        ".gnupg",
        "hf_token",
        "HF_TOKEN.txt",
        "my-secret.txt",
        "apikey",
        "openai_api_key",
        "github-token",
        "aws-credentials.ini",
        "db_password",
        "private_key",
    ] {
        assert!(is_secret_name(name), "{name}");
    }
}

#[test]
fn test_model_files_are_not_secret_looking() {
    for name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "config.json",
        "model-00001-of-00004.safetensors",
        "special_tokens_map.json",
        "README.md",
        "keymap.txt",
        "blobs",
        "snapshots",
        "added_tokens.json",
        "tokenizer.ggml.tokens",
        "models--org--tokenizer-model",
    ] {
        assert!(!is_secret_name(name), "{name}");
    }
}

#[test]
fn test_parse_time_handles_docker_timestamps() {
    assert_eq!(
        parse_time(&json!("2026-09-01T10:00:00Z")),
        Some(1_788_256_800.0)
    );
    assert_eq!(
        parse_time(&json!("2026-09-01T10:00:00.5Z")),
        Some(1_788_256_800.5)
    );
    let t = parse_time(&json!("2026-09-01T10:00:00.123456789Z")).unwrap();
    assert!((t - 1_788_256_800.123_457).abs() < 1e-6);
    assert_eq!(
        parse_time(&json!("2026-09-01T12:00:00+02:00")),
        Some(1_788_256_800.0)
    );
    for bad in [
        json!("0001-01-01T00:00:00Z"),
        json!(""),
        Value::Null,
        json!("garbage"),
        json!(5),
    ] {
        assert_eq!(parse_time(&bad), None, "{bad}");
    }
}

fn base() -> EnvMap {
    let mut e = EnvMap::new();
    e.insert("PATH".into(), "/usr/bin:/bin".into());
    e
}

fn sh(script: &str) -> Vec<String> {
    vec!["/bin/sh".into(), "-c".into(), script.into()]
}

#[test]
fn test_run_reports_missing_binaries_and_timeouts() {
    let tmp = tempfile::tempdir().unwrap();
    let env = child_env(None, &base());
    let missing = tmp.path().join("nope").to_string_lossy().into_owned();
    assert_eq!(util::run(&[missing], 5.0, &env).rc, 127);
    let res = util::run(&sh("sleep 5"), 0.3, &env);
    assert_eq!(res.rc, 124);
    let ok = util::run(&sh("echo hi"), 10.0, &env);
    assert!(ok.ok() && ok.out.trim() == "hi", "{ok:?}");
}

#[test]
fn test_run_tail_keeps_only_the_last_bytes() {
    let env = child_env(None, &base());
    let (rc, data, truncated) = util::run_tail(
        &sh("head -c 100000 /dev/zero | tr '\\0' x; echo; echo END"),
        10.0,
        1000,
        &env,
    );
    assert!(
        rc == 0 && truncated && data.len() <= 1000,
        "{rc} {truncated} {}",
        data.len()
    );
    assert!(String::from_utf8_lossy(&data).trim_end().ends_with("END"));
}

#[test]
fn test_guarded_call_times_out_and_skips_a_stuck_key() {
    let g = Guarded::new();
    let (gate_tx, gate_rx) = mpsc::channel::<()>();
    let stuck = g.call(
        "k",
        move || {
            let _ = gate_rx.recv_timeout(Duration::from_secs(5));
            1
        },
        Duration::from_millis(100),
    );
    assert_eq!(stuck, None); // stuck
    assert_eq!(g.call("k", || 2, Duration::from_millis(500)), None); // still stuck: skipped, no new thread
    gate_tx.send(()).unwrap();
    std::thread::sleep(Duration::from_millis(200));
    assert_eq!(g.call("k", || 3, Duration::from_millis(500)), Some(3));
    #[allow(unreachable_code)]
    let failing = g.call(
        "other",
        || -> i32 { panic!("an exception is just no answer") },
        Duration::from_millis(500),
    );
    assert_eq!(failing, None);
}

#[test]
fn test_fit_tail_respects_the_encoded_size() {
    let text = format!("{}THE END", "\u{1b}[0m".repeat(100_000));
    let (tail, trimmed) = util::fit_tail(&text, 10_000);
    assert!(trimmed && tail.ends_with("THE END"));
    assert!(util::json_size(&Value::String(tail)) <= 10_000);
    assert_eq!(util::fit_tail("short", 100), ("short".to_string(), false));
}

// ── child environments (2026-10-04 review, L1) ───────────────────────────────────────────────────────────
// Spelled out here, not imported, so the test fails if the allowlist grows by accident.
const CHILD_NAMES: &[&str] = &[
    "PATH",
    "HOME",
    "LANG",
    "XDG_RUNTIME_DIR",
    "DOCKER_HOST",
    "DOCKER_CONTEXT",
];

fn hostile() -> EnvMap {
    [
        ("LD_PRELOAD", "/nonexistent/evil.so"),
        ("LD_AUDIT", "/nonexistent/audit.so"),
        ("DOCKER_CONFIG", "/nonexistent/docker-config"),
        ("DOCKER_CERT_PATH", "/nonexistent/certs"),
        ("OTEL_EXPORTER_OTLP_ENDPOINT", "http://198.51.100.7:4318"),
        ("OTEL_TRACES_EXPORTER", "otlp"),
        ("PYTHONPATH", "/nonexistent/py"),
        ("GCONV_PATH", "/nonexistent/gconv"),
        ("GODEBUG", "http2debug=2"),
        ("HTTPS_PROXY", "http://198.51.100.7:3128"),
        ("LC_ALL", "C"),
        ("SSH_AUTH_SOCK", "/nonexistent/agent"),
        ("CANARY_SECRET", "must-not-reach-a-child"),
    ]
    .iter()
    .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
    .collect()
}

fn node_own() -> EnvMap {
    [
        ("PATH", ":relative/bin:/usr/bin:/usr/bin:/opt/tools/bin"),
        ("HOME", "/home/ops"),
        ("XDG_RUNTIME_DIR", "/run/user/1000"),
        ("DOCKER_HOST", "unix:///run/user/1000/docker.sock"),
        ("DOCKER_CONTEXT", "rootless"),
        ("LANG", "de_DE.UTF-8"),
        ("USER", "ops"),
    ]
    .iter()
    .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
    .collect()
}

fn em(pairs: &[(&str, &str)]) -> EnvMap {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
        .collect()
}

#[test]
fn test_child_env_is_built_from_scratch() {
    // Node hardening (PORTING.md deviation): the child's PATH is the node's fixed tool PATH, never the node's own
    // PATH (Python sanitises and passes the node's PATH); ECNODE_TOOL_PATH overrides it for tests, sanitised the same
    // way (absolute entries only, once).
    let mut b = hostile();
    b.extend(node_own());
    assert_eq!(
        child_env(None, &b),
        em(&[
            ("PATH", util::DEFAULT_PATH),
            ("HOME", "/home/ops"),
            ("XDG_RUNTIME_DIR", "/run/user/1000"),
            ("DOCKER_HOST", "unix:///run/user/1000/docker.sock"),
            ("DOCKER_CONTEXT", "rootless"),
            ("LANG", "C.UTF-8"),
        ])
    );
    let mut over = node_own();
    over.insert(
        util::TOOL_PATH_ENV.into(),
        ":relative/bin:/usr/bin:/usr/bin:/opt/tools/bin".into(),
    );
    assert_eq!(child_env(None, &over)["PATH"], "/usr/bin:/opt/tools/bin");
    assert!(!child_env(None, &over).contains_key(util::TOOL_PATH_ENV));
    // docker.run's secret values ride along, and never replace one of the names above
    let extra = em(&[
        ("VLLM_API_KEY", SECRET),
        ("HOME", "/tmp/evil"),
        ("PATH", "/tmp"),
    ]);
    let env = child_env(Some(&extra), &over);
    assert_eq!(env["VLLM_API_KEY"], SECRET);
    assert_eq!(env["HOME"], "/home/ops");
    assert_eq!(env["PATH"], "/usr/bin:/opt/tools/bin");
    assert_eq!(
        child_env(None, &EnvMap::new()),
        em(&[("PATH", util::DEFAULT_PATH), ("LANG", "C.UTF-8")])
    );
    assert_eq!(
        child_env(None, &em(&[("PATH", "relative:"), ("HOME", "")]))["PATH"],
        util::DEFAULT_PATH
    );
}

// ── node hardening: tools from a fixed PATH, output capped while it is read ─────────────────────────────

fn tool_dir(dir: &std::path::Path, tools: &[(&str, &str)]) {
    use std::os::unix::fs::PermissionsExt;
    std::fs::create_dir_all(dir).unwrap();
    for (name, body) in tools {
        let p = dir.join(name);
        std::fs::write(&p, format!("#!/bin/sh\n{body}\n")).unwrap();
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
}

#[test]
fn test_tools_resolve_in_the_tool_path_never_the_node_path() {
    let tmp = tempfile::tempdir().unwrap();
    let (on_path, tools) = (tmp.path().join("on-path"), tmp.path().join("tools"));
    tool_dir(&on_path, &[("ectool", "echo FROM-PATH")]);
    tool_dir(&tools, &[("ectool", "echo \"FROM-TOOLS $0 $PATH\"")]);
    let node_path = format!("{}:/usr/bin:/bin", on_path.display());
    let mut node = em(&[("PATH", node_path.as_str())]);
    let argv = vec!["ectool".to_string()];
    // no override: DEFAULT_PATH only, so the tool on the node's PATH is not found
    let res = util::Runner::run(&util::ProcRunner::new(node.clone()), &argv, 10.0, None);
    assert_eq!(res.rc, 127, "{res:?}");
    assert_eq!(res.err, "ectool: command not found");
    // with the override, the tool path's copy runs (a script sees the path it was executed by as $0, as after
    // Python's PATH search) and its PATH is the tool PATH
    node.insert(util::TOOL_PATH_ENV.into(), tools.display().to_string());
    let res = util::Runner::run(&util::ProcRunner::new(node.clone()), &argv, 10.0, None);
    assert!(res.ok(), "{res:?}");
    assert_eq!(
        res.out.trim(),
        format!(
            "FROM-TOOLS {} {}",
            tools.join("ectool").display(),
            tools.display()
        )
    );
    // the streaming runner resolves the same way
    let mut sp = util::spawn(&argv, &child_env(None, &node), true, None).unwrap();
    let mut out = String::new();
    std::io::Read::read_to_string(&mut sp.stdout, &mut out).unwrap();
    sp.proc.wait();
    assert!(out.starts_with("FROM-TOOLS"), "{out}");
    // a program found by name still sees that name as argv[0] (the shell prints its own argv[0] as $0)
    let base_env = child_env(None, &base());
    let res = util::run(
        &["sh".to_string(), "-c".to_string(), "echo $0".to_string()],
        5.0,
        &base_env,
    );
    assert_eq!(res.out.trim(), "sh", "{res:?}");
    // a relative program with a '/' would depend on the current directory: never run
    assert_eq!(
        util::run(&["./ectool".to_string()], 5.0, &child_env(None, &node)).rc,
        127
    );
}

#[test]
fn test_child_output_is_capped_while_it_is_read() {
    let env = child_env(None, &base());
    // 40 MB on each pipe, both at once: with caps of 1000 / 500 bytes the child is drained (never blocked on a full
    // pipe), finishes, and only the first bytes are kept
    let flood = sh(
        "head -c 40000000 /dev/zero | tr '\\0' o & head -c 40000000 /dev/zero | tr '\\0' e >&2; wait",
    );
    let started = Instant::now();
    let res = util::run_capped(&flood, 60.0, &env, 1000, 500);
    assert_eq!(res.rc, 0, "{}", res.err);
    assert!(res.truncated);
    assert_eq!(res.out, "o".repeat(1000));
    assert_eq!(res.err, "e".repeat(500));
    assert!(started.elapsed() < Duration::from_secs(60));
    // the default caps hold for the plain runner too
    let big = sh(&format!(
        "head -c {} /dev/zero | tr '\\0' x",
        util::RUN_MAX_OUT + 4096
    ));
    let res = util::run(&big, 60.0, &env);
    assert!(res.ok() && res.truncated && res.out.len() == util::RUN_MAX_OUT);
    // a quiet child is untouched
    let ok = util::run(&sh("echo hi; echo there >&2"), 10.0, &env);
    assert_eq!(
        (ok.out.as_str(), ok.err.as_str(), ok.truncated),
        ("hi\n", "there\n", false)
    );
    // the timeout still holds for a child that floods and never ends
    let res = util::run_capped(&sh("yes"), 0.5, &env, 100, 100);
    assert_eq!(res.rc, 124);
}

#[test]
fn test_iter_lines_caps_each_line() {
    // a 3 MiB line comes out in 1 MiB pieces, then its newline alone (Python's readline(1 MiB) does the same),
    // each piece cut to `limit` characters
    let data = format!("{}\nshort\n", "z".repeat(3 * 1024 * 1024));
    let lines: Vec<String> = util::iter_lines(data.as_bytes(), 100).collect();
    assert_eq!(
        lines.iter().map(String::len).collect::<Vec<_>>(),
        vec![100, 100, 100, 0, 5]
    );
    assert!(lines[..3].iter().all(|l| l == &"z".repeat(100)));
    assert_eq!(lines[4], "short");
}

fn names_of(out: &[u8]) -> BTreeSet<String> {
    String::from_utf8_lossy(out)
        .lines()
        .filter_map(|l| l.split_once('=').map(|(k, _)| k.to_string()))
        .collect()
}

fn set(a: &[&str]) -> BTreeSet<String> {
    a.iter().map(|s| (*s).to_string()).collect()
}

#[test]
fn test_children_never_inherit_the_node_environment() {
    // the node's own environment: hostile names plus HOME and DOCKER_HOST (no XDG_RUNTIME_DIR, DOCKER_CONTEXT)
    let mut node = hostile();
    node.insert("HOME".into(), "/home/ops".into());
    node.insert(
        "DOCKER_HOST".into(),
        "unix:///run/user/1000/docker.sock".into(),
    );
    node.insert("PATH".into(), "/usr/bin:/bin".into());
    let runner = util::ProcRunner::new(node.clone());
    let show = vec!["/usr/bin/env".to_string()];
    let res: CmdResult = util::Runner::run(&runner, &show, 20.0, None);
    assert!(res.ok());
    assert_eq!(
        names_of(res.out.as_bytes()),
        set(&["PATH", "HOME", "LANG", "DOCKER_HOST"]),
        "{}",
        res.out
    );
    let (rc, data, _) = util::run_tail(&show, 20.0, 100_000, &child_env(None, &node));
    assert_eq!(rc, 0);
    assert!(names_of(&data).is_subset(&set(CHILD_NAMES)));
    let mut sp = util::spawn(&show, &child_env(None, &node), false, None).unwrap();
    let mut out = Vec::new();
    std::io::Read::read_to_end(&mut sp.stdout, &mut out).unwrap();
    sp.proc.wait();
    assert!(names_of(&out).is_subset(&set(CHILD_NAMES)));
    // an explicit env is the child's whole environment, as before
    let res = util::Runner::run(
        &runner,
        &show,
        20.0,
        Some(&em(&[
            ("PATH", "/usr/bin"),
            ("LANG", "C.UTF-8"),
            ("ONLY_THIS", "1"),
        ])),
    );
    assert_eq!(
        names_of(res.out.as_bytes()),
        set(&["PATH", "LANG", "ONLY_THIS"])
    );
    let _ = wsd_kit::GPU_A;
}
