//! node/tests/test_scripts.py: scripts.run: policy argv only, one allowlisted argument, minimal environment,
//! streaming, busy, timeouts. (The Python test's helper script is Python; here the actions are /usr/bin/env,
//! /usr/bin/printenv and /bin/sh one-liners, which add nothing to the environment they print.)
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
use std::sync::Arc;
use std::time::{Duration, Instant};

use ec_proto::{BAD_ARGS, BUSY, NOT_FOUND, Obj, POLICY_DENIED, TIMEOUT};
use ecnode::errors::HandlerError;
use ecnode::policy::{Policy, parse};
use ecnode::scripts::{ENV_KEEP, Scripts};
use ecnode::util::EnvMap;
use serde_json::{Value, json};
use wsd_kit::{FakeStream, clean_args};

struct Fixture {
    tmp: tempfile::TempDir,
    policy: Policy,
    env: EnvMap,
}

fn fixture() -> Fixture {
    let tmp = tempfile::tempdir().unwrap();
    let nope = tmp.path().join("nope.sh").to_string_lossy().into_owned();
    let policy = parse(
        &json!({"mode": "manage", "scripts": [{
            "name": "lab", "containers": ["lab-*"], "timeout_s": 2, "status_action": "env",
            "actions": [
                {"name": "env", "argv": ["/usr/bin/env"]},
                {"name": "path", "argv": ["/usr/bin/printenv", "PATH"]},
                {"name": "talk", "argv": ["/bin/sh", "-c",
                    "echo 'out line 1'; echo 'api_key=sk-test-not-a-real-key-0000'; echo 'err line' >&2; exit 3"]},
                {"name": "sleep", "argv": ["/bin/sh", "-c", "sleep \"$1\"; echo \"args: ['sleep', '$1']\"", "lab"],
                 "args": ["0.1", "10"]},
                {"name": "bg", "argv": ["/bin/sh", "-c", "sleep 4 & echo started"]},
                {"name": "missing", "argv": [nope]}]}]}),
        Some("/x"),
        None,
    )
    .unwrap();
    let env: EnvMap = [
        ("LANG", "C.UTF-8".to_string()),
        ("CANARY_SECRET", "must-not-reach-scripts".into()),
        ("HOME", tmp.path().to_string_lossy().into_owned()),
        ("PATH", "/usr/bin:/bin".into()),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();
    Fixture { tmp, policy, env }
}

fn args(action: &str, arg: Option<&str>) -> Obj {
    clean_args(
        "scripts.run",
        json!({"name": "lab", "action": action, "arg": arg}),
    )
}

fn run(
    f: &Fixture,
    action: &str,
    arg: Option<&str>,
    s: &FakeStream,
    runner: &Scripts,
) -> Result<Obj, HandlerError> {
    runner.run(&f.policy, &args(action, arg), s, &f.env)
}

fn code(r: Result<Obj, HandlerError>) -> String {
    match r {
        Err(HandlerError::Op(e)) => e.code,
        other => panic!("expected an OpError, got {other:?}"),
    }
}

fn lines(s: &FakeStream) -> Vec<(String, String)> {
    s.events()
        .iter()
        .map(|e| {
            (
                e["stream"].as_str().unwrap().to_string(),
                e["line"].as_str().unwrap().to_string(),
            )
        })
        .collect()
}

#[test]
fn test_environment_is_minimal() {
    let f = fixture();
    let s = FakeStream::empty();
    let res = run(&f, "env", None, &s, &Scripts::new()).unwrap();
    let keys: BTreeSet<String> = lines(&s)
        .into_iter()
        .filter_map(|(_, l)| l.split_once('=').map(|(k, _)| k.to_string()))
        .collect();
    let allowed: BTreeSet<String> = ENV_KEEP.iter().map(|k| (*k).to_string()).collect();
    assert!(
        keys.is_subset(&allowed) && !keys.contains("CANARY_SECRET"),
        "{keys:?}"
    );
    assert_eq!(Value::Object(res), json!({"rc": 0}));
}

#[test]
fn test_path_keeps_absolute_entries_only() {
    // Scripts run in HOME: an empty or relative PATH entry would make `docker` in a script resolve there first.
    // Node hardening: a script's PATH is the node's tool PATH (ECNODE_TOOL_PATH here), never the node's own PATH;
    // the same absolute-entries-only rule applies to it.
    let mut f = fixture();
    let t = f.tmp.path().display().to_string();
    f.env.insert(
        ecnode::util::TOOL_PATH_ENV.into(),
        format!(":relative/bin:{t}/bin::/usr/bin:{t}/bin"),
    );
    let s = FakeStream::empty();
    run(&f, "path", None, &s, &Scripts::new()).unwrap();
    assert_eq!(lines(&s)[0].1, format!("{t}/bin:/usr/bin"));
}

#[test]
fn test_scripts_never_get_the_node_path() {
    let mut f = fixture();
    f.env
        .insert("PATH".into(), "/opt/caller/bin:/usr/bin".into());
    let s = FakeStream::empty();
    run(&f, "path", None, &s, &Scripts::new()).unwrap();
    assert_eq!(lines(&s)[0].1, ecnode::util::DEFAULT_PATH);
}

#[test]
fn test_lines_are_streamed_scrubbed_and_rc_returned() {
    let f = fixture();
    let s = FakeStream::empty();
    let res = run(&f, "talk", None, &s, &Scripts::new()).unwrap();
    assert_eq!(Value::Object(res), json!({"rc": 3}));
    let l = lines(&s);
    assert!(
        l.contains(&("out".into(), "out line 1".into()))
            && l.contains(&("err".into(), "err line".into()))
    );
    assert!(
        l.iter()
            .all(|(_, line)| !line.contains("sk-test-not-a-real-key"))
    );
}

#[test]
fn test_arguments_come_from_the_allowlist() {
    let f = fixture();
    let s = FakeStream::empty();
    let res = run(&f, "sleep", Some("0.1"), &s, &Scripts::new()).unwrap();
    assert_eq!(Value::Object(res), json!({"rc": 0}));
    assert!(lines(&s).iter().any(|(_, l)| l.contains("'0.1'")));
    for (action, arg, c) in [
        ("sleep", Some("rm -rf /"), POLICY_DENIED),
        ("sleep", None, BAD_ARGS),
        ("env", Some("x"), BAD_ARGS),
        ("nope", None, NOT_FOUND),
        ("missing", None, NOT_FOUND),
    ] {
        assert_eq!(
            code(run(&f, action, arg, &FakeStream::empty(), &Scripts::new())),
            c,
            "{action} {arg:?}"
        );
    }
    let other = clean_args(
        "scripts.run",
        json!({"name": "other", "action": "env", "arg": null}),
    );
    assert_eq!(
        code(Scripts::new().run(&f.policy, &other, &FakeStream::empty(), &f.env)),
        NOT_FOUND
    );
}

#[test]
fn test_one_run_per_script() {
    let f = Arc::new(fixture());
    let runner = Arc::new(Scripts::new());
    let long = Arc::new(FakeStream::empty());
    let t = {
        let (f, runner, long) = (Arc::clone(&f), Arc::clone(&runner), Arc::clone(&long));
        std::thread::spawn(move || {
            let _ = run(&f, "sleep", Some("10"), &long, &runner);
        })
    };
    std::thread::sleep(Duration::from_millis(300));
    let s = FakeStream::empty();
    assert_eq!(code(run(&f, "env", None, &s, &runner)), BUSY);
    long.cancel();
    t.join().unwrap();
    assert!(s.events().is_empty());
}

#[test]
fn test_timeout_stops_the_run() {
    let f = fixture();
    let t0 = Instant::now();
    assert_eq!(
        code(run(
            &f,
            "sleep",
            Some("10"),
            &FakeStream::empty(),
            &Scripts::new()
        )),
        TIMEOUT
    );
    assert!(t0.elapsed() < Duration::from_secs(8));
}

#[test]
fn test_cancel_stops_the_run() {
    let f = fixture();
    let s = Arc::new(FakeStream::empty());
    {
        let s = Arc::clone(&s);
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(300));
            s.cancel();
        });
    }
    let t0 = Instant::now();
    let r = run(&f, "sleep", Some("10"), &s, &Scripts::new());
    assert!(matches!(r, Err(HandlerError::Cancelled)), "{r:?}");
    assert!(t0.elapsed() < Duration::from_secs(5));
}

#[test]
fn test_a_background_child_holding_the_pipe_does_not_hang_the_run() {
    let f = fixture();
    let s = FakeStream::empty();
    let t0 = Instant::now();
    let res = run(&f, "bg", None, &s, &Scripts::new()).unwrap();
    assert_eq!(Value::Object(res), json!({"rc": 0}));
    assert!(lines(&s).iter().any(|(_, l)| l == "started"));
    assert!(t0.elapsed() < Duration::from_secs(6));
}

#[test]
fn test_status_patterns() {
    let f = fixture();
    let up = ecnode::discovery::script_status(&f.policy, &["lab-sglang-1".to_string()], 1.0);
    assert_eq!(up[0]["status"], "up");
    let down = ecnode::discovery::script_status(&f.policy, &["other".to_string()], 1.0);
    assert_eq!(down[0]["status"], "down");
}
