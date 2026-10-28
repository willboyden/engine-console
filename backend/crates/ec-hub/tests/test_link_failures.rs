//! Port of backend/tests/fleet/test_link_failures.py: how a link fails and recovers: backoff, reconnect, ssh failures
//! in words, crashes and cleanup.

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

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use ec_hub::domain::fleet::{HostConfig, NodeClient};
use ec_hub::services::fleet::{LinkLike, LinkTiming, NodeLink, explain_exit};
use ec_hub::services::hosts::HostsError;
use serde_json::json;
use wsb_harness::{
    Harness, RigOpts, alive, fast, host, limited, remote_host, wait_for, wait_until,
};

fake_node_entry!();

const CMD: &str = "/usr/local/lib/ecnode/ecnode --stdio";

fn timing(edit: impl FnOnce(&mut LinkTiming)) -> LinkTiming {
    let mut t = fast();
    edit(&mut t);
    t
}

fn remote() -> HostConfig {
    host("gpu-box-2", "GPU box 2", Some("ops@10.0.0.5"))
}

fn local() -> HostConfig {
    host("local", "This machine", None)
}

// ── explain_exit, case by case ─────────────────────────────────────────────────────────────────────────
#[test]
fn test_explain_exit() {
    let cases: Vec<(HostConfig, Option<i32>, &str, &str, bool)> = vec![
        (
            remote(),
            Some(255),
            "ops@10.0.0.5: Permission denied (publickey).",
            "hosts init-key",
            true,
        ),
        (
            remote(),
            Some(255),
            "Warning: x\nops@10.0.0.5: Permission denied (publickey,password).",
            "authorized_keys",
            true,
        ),
        (
            remote(),
            Some(255),
            "No ED25519 host key is known for 10.0.0.5 and you have requested strict checking.\nHost key verification failed.",
            "connect once by hand",
            true,
        ),
        (
            remote(),
            Some(255),
            "@@@ WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED! @@@",
            "ssh ops@10.0.0.5 true",
            true,
        ),
        (
            remote(),
            Some(255),
            "@ WARNING: UNPROTECTED PRIVATE KEY FILE! @\nPermissions 0644 are too open.",
            "chmod 600",
            true,
        ),
        (
            remote(),
            Some(255),
            "ssh: Could not resolve hostname gpu-box-9: Name or service not known",
            "cannot resolve",
            false,
        ),
        (
            remote(),
            Some(255),
            "ssh: connect to host 10.0.0.5 port 22: No route to host",
            "cannot reach ops@10.0.0.5",
            false,
        ),
        (
            remote(),
            Some(255),
            "ssh: connect to host 10.0.0.5 port 22: Connection refused",
            "cannot reach",
            false,
        ),
        (
            remote(),
            Some(255),
            "ssh: connect to host 10.0.0.5 port 22: Connection timed out",
            "cannot reach",
            false,
        ),
        (
            remote(),
            Some(255),
            "something odd",
            "ssh to ops@10.0.0.5 failed: something odd",
            false,
        ),
        (
            remote(),
            Some(255),
            "",
            "ssh to ops@10.0.0.5 failed: exit 255",
            false,
        ),
        (
            remote(),
            Some(127),
            "bash: line 1: /usr/local/lib/ecnode/ecnode: No such file or directory",
            "not installed on GPU box 2 (expected /usr/local/lib/ecnode/ecnode)",
            true,
        ),
        (
            remote(),
            Some(1),
            "FileNotFoundError: No such file or directory: '/etc/x'",
            "the node exited (1)",
            false,
        ), // not "not installed"
        (
            local(),
            Some(255),
            "Permission denied",
            "the node exited (255): Permission denied",
            false,
        ), // no ssh, no ssh advice
        (
            local(),
            Some(1),
            "/usr/bin/python3: No module named ecnode",
            "missing from the console's Python environment",
            false,
        ),
        (local(), Some(2), "", "the node exited (2)", false),
        (local(), Some(-15), "", "stopped by signal 15", false),
        (local(), Some(-9), "", "the connection closed", false),
        (local(), Some(0), "", "the connection closed", false),
        (local(), None, "last words", "last words", false),
    ];
    for (h, rc, stderr, needle, fix) in cases {
        let (msg, wait) = explain_exit(&h, rc, stderr, Some(CMD), 300.0, &[]);
        assert!(msg.contains(needle), "{msg:?} lacks {needle:?}");
        assert_eq!(wait == Some(300.0), fix, "{msg}");
        assert_eq!(wait.is_none(), !fix, "{msg}");
        assert!(!msg.contains('\n'));
    }
}

#[test]
fn test_explain_exit_masks_the_hub_key_path() {
    let key = "/home/ops/.config/engine-console/ssh/hub-key";
    let (msg, wait) = explain_exit(
        &remote(),
        Some(255),
        &format!("Load key \"{key}\": error in libcrypto"),
        Some(CMD),
        300.0,
        &[key.to_string()],
    );
    assert_eq!(
        msg,
        "ssh to ops@10.0.0.5 failed: Load key \"<hub key>\": error in libcrypto"
    );
    assert!(wait.is_none() && !msg.contains(key));
    let (again, _) = explain_exit(
        &remote(),
        Some(255),
        &format!("Load key \"{key}\": bad\nand again {key}"),
        Some(CMD),
        300.0,
        &[String::new(), key.to_string()],
    );
    assert!(!again.contains(key) && again.contains("<hub key>"));
}

#[test]
fn test_explain_exit_makes_untrusted_text_safe() {
    let (msg, _) = explain_exit(
        &local(),
        Some(1),
        "boom hf_abcdefghijklmnop\u{1b}[31m red\r\n",
        Some(CMD),
        300.0,
        &[],
    );
    assert!(!msg.contains("hf_abcdefghijklmnop") && msg.contains("[redacted]"));
    assert!(!msg.contains('\u{1b}') && !msg.contains('\r') && !msg.contains('\n'));
    let (long_msg, _) = explain_exit(&local(), Some(1), &"x".repeat(5000), Some(CMD), 300.0, &[]);
    assert!(long_msg.chars().count() < 400);
}

// ── backoff ────────────────────────────────────────────────────────────────────────────────────────────
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_backoff_doubles_and_is_capped() {
    limited(async {
        let mut h = Harness::new();
        let rig = h
            .rig_with(
                json!({"exit_after_s": 0.05}),
                RigOpts {
                    timing: timing(|t| {
                        t.backoff_min_s = 0.02;
                        t.backoff_max_s = 0.16;
                    }),
                    ..RigOpts::default()
                },
            )
            .await;
        let ch = Arc::clone(&rig.changes);
        wait_until(
            move || {
                ch.lock()
                    .unwrap()
                    .iter()
                    .filter(|(s, w)| s == "down" && w.is_some())
                    .count()
                    >= 6
            },
            15.0,
            "six failures",
        )
        .await;
        assert_eq!(rig.waits()[..6], [0.02, 0.04, 0.08, 0.16, 0.16, 0.16]);
        assert!(
            rig.error()
                .contains("the node exited (3): fake node crashing on purpose"),
            "{}",
            rig.error()
        );
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_backoff_starts_over_after_a_long_enough_connection() {
    limited(async {
        let mut h = Harness::new();
        let rig = h
            .rig_with(
                json!({"exit_after_s": 0.3}),
                RigOpts {
                    timing: timing(|t| {
                        t.backoff_min_s = 0.02;
                        t.backoff_max_s = 0.5;
                        t.stable_s = 0.15;
                    }),
                    ..RigOpts::default()
                },
            )
            .await;
        let ch = Arc::clone(&rig.changes);
        wait_until(
            move || {
                ch.lock()
                    .unwrap()
                    .iter()
                    .filter(|(s, w)| s == "down" && w.is_some())
                    .count()
                    >= 3
            },
            15.0,
            "three failures",
        )
        .await;
        // each session outlived stable_s, so the wait never grew
        assert_eq!(rig.waits()[..3], [0.02, 0.02, 0.02]);
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_a_session_that_never_got_up_backs_off_too() {
    limited(async {
        let mut h = Harness::new();
        let rig = h
            .rig_with(
                json!({"exit_before_hello": 1}),
                RigOpts {
                    timing: timing(|t| {
                        t.backoff_min_s = 0.02;
                        t.backoff_max_s = 0.16;
                    }),
                    ..RigOpts::default()
                },
            )
            .await;
        let ch = Arc::clone(&rig.changes);
        wait_until(
            move || {
                ch.lock()
                    .unwrap()
                    .iter()
                    .filter(|(s, w)| s == "down" && w.is_some())
                    .count()
                    >= 5
            },
            15.0,
            "five failures",
        )
        .await;
        assert_eq!(rig.waits()[..5], [0.02, 0.04, 0.08, 0.16, 0.16]);
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_the_link_shows_the_retry_countdown_while_down() {
    limited(async {
        let mut h = Harness::new();
        let rig = h
            .rig_with(
                json!({"exit_before_hello": 2}),
                RigOpts {
                    timing: timing(|t| {
                        t.backoff_min_s = 5.0;
                        t.backoff_max_s = 5.0;
                    }),
                    ..RigOpts::default()
                },
            )
            .await;
        rig.state("down").await;
        let st = rig.st();
        assert!(matches!(st.retry_in_s, Some(4 | 5)), "{:?}", st.retry_in_s);
        assert!(st.connected_since.is_none() && st.rtt_ms.is_none());
        h.stop().await;
    })
    .await;
}

// ── ssh-style failures end to end (the fake exits like ssh does) ───────────────────────────────────────
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_login_refused_names_the_fix_and_retries_rarely() {
    limited(async {
        let mut h = Harness::new();
        let rig = h
            .rig_with(json!({"stderr": "ops@10.0.0.5: Permission denied (publickey).", "exit_before_hello": 255}), RigOpts { host: remote_host(), ..RigOpts::default() })
            .await;
        rig.state("down").await;
        assert!(rig.error().contains("hosts init-key"), "{}", rig.error());
        assert_eq!(rig.link.last_wait_s(), Some(fast().fix_retry_s));
        assert_eq!(rig.st().retry_in_s, Some(9));
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_host_key_trouble_says_to_connect_once_by_hand() {
    limited(async {
        let mut h = Harness::new();
        let rig = h
            .rig_with(
                json!({"stderr": "Host key verification failed.", "exit_before_hello": 255}),
                RigOpts {
                    host: remote_host(),
                    ..RigOpts::default()
                },
            )
            .await;
        rig.state("down").await;
        assert!(rig.error().contains("connect once by hand"));
        assert_eq!(rig.link.last_wait_s(), Some(fast().fix_retry_s));
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_node_not_installed_says_where_it_expected_it() {
    limited(async {
        let mut h = Harness::new();
        let rig = h
            .rig_with(
                json!({"stderr": "sh: 1: /usr/local/lib/ecnode/ecnode: not found", "exit_before_hello": 127}),
                RigOpts { host: remote_host(), start: false, ..RigOpts::default() },
            )
            .await;
        rig.link.set_remote_cmd(Some(CMD.into()));
        rig.link.start().await;
        rig.state("down").await;
        assert!(rig.error().contains("not installed on GPU box 2"), "{}", rig.error());
        assert_eq!(rig.link.last_wait_s(), Some(fast().fix_retry_s));
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_no_route_is_an_ordinary_failure_with_ordinary_backoff() {
    limited(async {
        let mut h = Harness::new();
        let rig = h
            .rig_with(json!({"stderr": "ssh: connect to host 10.0.0.5 port 22: No route to host", "exit_before_hello": 255}), RigOpts { host: remote_host(), ..RigOpts::default() })
            .await;
        rig.state("down").await;
        assert!(rig.error().contains("cannot reach ops@10.0.0.5"));
        assert_eq!(rig.link.last_wait_s(), Some(fast().backoff_min_s));
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_reconnect_skips_a_long_wait_and_forgets_the_backoff() {
    limited(async {
        let mut h = Harness::new();
        let rig = h
            .rig_with(
                json!({"stderr": "Permission denied", "exit_before_hello": 255}),
                RigOpts {
                    host: remote_host(),
                    ..RigOpts::default()
                },
            )
            .await;
        rig.state("down").await;
        assert_eq!(rig.events("start").len(), 1);
        assert_eq!(rig.link.last_wait_s(), Some(fast().fix_retry_s));
        rig.link.reconnect();
        let path = rig.log_path.clone();
        wait_until(
            move || {
                wsb_fake_node::read_log(&path)
                    .iter()
                    .filter(|e| e["ev"] == "start")
                    .count()
                    >= 2
            },
            3.0,
            "an immediate second attempt",
        )
        .await;
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_incompatible_waits_are_cut_short_by_reconnect_too() {
    limited(async {
        let mut h = Harness::new();
        let rig = h.rig(json!({"proto": 2})).await;
        rig.state("incompatible").await;
        rig.link.reconnect();
        let path = rig.log_path.clone();
        wait_until(
            move || {
                wsb_fake_node::read_log(&path)
                    .iter()
                    .filter(|e| e["ev"] == "start")
                    .count()
                    >= 2
            },
            3.0,
            "a new attempt",
        )
        .await;
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_a_command_that_cannot_be_built_is_shown_and_retried_rarely_then_fixed() {
    limited(async {
        let mut h = Harness::new();
        let attempts = Arc::new(AtomicUsize::new(0));
        let a2 = Arc::clone(&attempts);
        let late = h.path("late.jsonl");
        let argv: ec_hub::services::fleet::ArgvFn = Arc::new(move || {
            if a2.fetch_add(1, Ordering::SeqCst) == 0 {
                return Err(HostsError::Invalid(
                    "the console has no ssh key yet: run `engine-console hosts init-key`".into(),
                ));
            }
            Ok(wsb_fake_node::node_argv(
                &json!({"log": late.to_string_lossy()}),
            ))
        });
        let rig = h
            .rig_with(
                json!({}),
                RigOpts {
                    argv: Some(argv),
                    ..RigOpts::default()
                },
            )
            .await;
        rig.state("down").await;
        assert_eq!(
            rig.error(),
            "the console has no ssh key yet: run `engine-console hosts init-key`"
        );
        assert_eq!(rig.link.last_wait_s(), Some(fast().fix_retry_s));
        rig.link.reconnect(); // the person did what it said
        rig.up().await;
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
        h.stop().await;
    })
    .await;
}

// ── crashes and recovery ───────────────────────────────────────────────────────────────────────────────
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_the_link_recovers_after_the_node_dies_and_keeps_what_it_knew() {
    limited(async {
        let mut h = Harness::new();
        let rig = h
            .rig_with(
                json!({"exit_after_s": 0.5}),
                RigOpts {
                    timing: timing(|t| t.backoff_min_s = 0.05),
                    ..RigOpts::default()
                },
            )
            .await;
        rig.up().await;
        let first = rig.st().connected_since;
        let l = Arc::clone(&rig.link);
        wait_for(move || l.latest("sample"), 8.0, "a sample").await;
        rig.state("down").await;
        // the last hello is kept: the page can still show the machine; a live sample from a dead link is gone
        assert!(rig.link.latest("hello").is_some());
        assert!(rig.link.latest("sample").is_none());
        assert!(rig.st().connected_since.is_none() && rig.st().rtt_ms.is_none());
        rig.up().await;
        assert!(rig.st().connected_since.is_some() && rig.st().connected_since != first);
        assert!(rig.events("start").len() >= 2);
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_pending_calls_fail_at_once_when_the_node_dies() {
    limited(async {
        let mut h = Harness::new();
        let rig = h.rig(json!({"exit_after_s": 0.4})).await;
        rig.up().await;
        let t0 = Instant::now();
        let e = rig
            .link
            .call("models.scan", None, Duration::from_secs(30))
            .await
            .unwrap_err();
        assert!(t0.elapsed() < Duration::from_secs(3));
        assert!(
            e.code == "node_unavailable" && e.message.contains("closed the connection"),
            "{e}"
        );
        assert_eq!(rig.link.inflight(), 0);
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_a_fatal_after_the_hello_ends_the_link_with_the_nodes_words() {
    limited(async {
        let mut h = Harness::new();
        let rig = h.rig(json!({"pushes": [{"after_s": 0.2, "msg": {"type": "fatal", "error": "docker socket vanished"}}]})).await;
        rig.up().await;
        rig.state("down").await;
        assert!(rig.error().starts_with("the node gave up: docker socket vanished"), "{}", rig.error());
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_reconnect_on_a_live_link_drops_and_re_establishes_it() {
    limited(async {
        let mut h = Harness::new();
        let rig = h.rig(json!({})).await;
        rig.up().await;
        let first = rig.st().connected_since;
        let l = Arc::clone(&rig.link);
        let call =
            tokio::spawn(async move { l.call("models.scan", None, Duration::from_secs(30)).await });
        let l = Arc::clone(&rig.link);
        wait_until(move || l.inflight() == 1, 8.0, "a pending call").await;
        tokio::time::sleep(Duration::from_millis(10)).await;
        rig.link.reconnect();
        let e = call.await.unwrap().unwrap_err();
        assert!(
            e.code == "node_unavailable" && e.message.contains("reconnecting at your request"),
            "{e}"
        );
        let (path, l) = (rig.log_path.clone(), Arc::clone(&rig.link));
        wait_until(
            move || {
                wsb_fake_node::read_log(&path)
                    .iter()
                    .filter(|e| e["ev"] == "start")
                    .count()
                    == 2
                    && l.link().state == "up"
            },
            5.0,
            "a new connection",
        )
        .await;
        assert_ne!(rig.st().connected_since, first);
        assert!(
            !rig.call("docker.images", json!({})).await.unwrap()["images"]
                .as_array()
                .unwrap()
                .is_empty()
        );
        // a requested reconnect is not a failure to back off from
        assert!(rig.waits().is_empty(), "{:?}", rig.waits());
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_reconnect_while_connecting_does_not_leave_a_stale_request_behind() {
    limited(async {
        let mut h = Harness::new();
        let rig = h.rig(json!({"hello_delay_s": 1.5})).await;
        let path = rig.log_path.clone();
        wait_until(
            move || {
                wsb_fake_node::read_log(&path)
                    .iter()
                    .any(|e| e["ev"] == "start")
            },
            8.0,
            "the first attempt to be running",
        )
        .await;
        rig.link.reconnect();
        rig.up().await;
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(rig.st().state, "up");
        // the slow first attempt was abandoned at once (no hello yet, nothing graceful to do) and replaced; the link that
        // came up has a working writer, and nothing was counted as a failure to back off from
        assert_eq!(rig.events("start").len(), 2);
        assert_eq!(rig.link.last_wait_s(), None);
        assert!(
            !rig.call("docker.images", json!({})).await.unwrap()["images"]
                .as_array()
                .unwrap()
                .is_empty()
        );
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_a_listener_that_raises_does_not_break_the_link() {
    limited(async {
        let mut h = Harness::new();
        let rig = h
            .rig_with(
                json!({}),
                RigOpts {
                    start: false,
                    ..RigOpts::default()
                },
            )
            .await;
        rig.link
            .set_on_change(Some(Arc::new(|| panic!("listener bug"))));
        rig.link.start().await;
        rig.up().await;
        assert!(
            !rig.call("docker.images", json!({})).await.unwrap()["images"]
                .as_array()
                .unwrap()
                .is_empty()
        );
        h.stop().await;
    })
    .await;
}

// ── cleanup ────────────────────────────────────────────────────────────────────────────────────────────
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_every_node_process_is_gone_after_many_restarts_and_stop() {
    limited(async {
        let mut h = Harness::new();
        let rig = h
            .rig_with(
                json!({"exit_after_s": 0.05}),
                RigOpts {
                    timing: timing(|t| {
                        t.backoff_min_s = 0.01;
                        t.backoff_max_s = 0.02;
                    }),
                    ..RigOpts::default()
                },
            )
            .await;
        let path = rig.log_path.clone();
        wait_until(
            move || {
                wsb_fake_node::read_log(&path)
                    .iter()
                    .filter(|e| e["ev"] == "start")
                    .count()
                    >= 5
            },
            15.0,
            "five restarts",
        )
        .await;
        rig.link.stop().await;
        let pids = rig.pids();
        assert!(pids.len() >= 5);
        assert!(pids.iter().all(|p| !alive(*p)), "{pids:?}");
        h.stop().await;
    })
    .await;
}

/// ssh may spawn helpers (a ProxyCommand); a node may spawn children. Each node is its own process group, and the
/// hub kills the group, so nothing a dropped node started is left running.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_the_whole_process_group_is_killed_not_just_the_child() {
    limited(async {
        let mut h = Harness::new();
        let rig = h
            .rig_with(
                json!({"spawn_child": true, "silent_after_hello": true}),
                RigOpts {
                    timing: timing(|t| {
                        t.silent_s = 0.4;
                        t.exit_grace_s = 0.3;
                    }),
                    ..RigOpts::default()
                },
            )
            .await;
        let path = rig.log_path.clone();
        let child = wait_for(
            move || {
                wsb_fake_node::read_log(&path)
                    .into_iter()
                    .find(|e| e["ev"] == "child")
                    .and_then(|e| e["pid"].as_i64())
            },
            8.0,
            "the grandchild",
        )
        .await;
        assert!(alive(child));
        rig.state("down").await; // the silent node was dropped and killed
        wait_until(
            move || !alive(child),
            8.0,
            "the grandchild to be killed with its parent",
        )
        .await;
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_stop_kills_the_whole_group_too() {
    limited(async {
        let mut h = Harness::new();
        let rig = h
            .rig_with(
                json!({"spawn_child": true, "silent_after_hello": true}),
                RigOpts {
                    timing: timing(|t| {
                        t.silent_s = 60.0;
                        t.exit_grace_s = 0.3;
                    }),
                    ..RigOpts::default()
                },
            )
            .await;
        let path = rig.log_path.clone();
        let child = wait_for(
            move || {
                wsb_fake_node::read_log(&path)
                    .into_iter()
                    .find(|e| e["ev"] == "child")
                    .and_then(|e| e["pid"].as_i64())
            },
            8.0,
            "the grandchild",
        )
        .await;
        rig.up().await;
        rig.link.stop().await;
        wait_until(
            move || !alive(child),
            8.0,
            "the grandchild to die with its parent's group",
        )
        .await;
        h.stop().await;
    })
    .await;
}

/// Python starts each node in its own session; the Rust hub gives it its own process group (DV7: setsid would need
/// unsafe), which is what this test checks: the group is the node's, not the hub's.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_the_node_is_its_own_session_so_a_signal_to_the_hub_group_misses_it() {
    limited(async {
        let mut h = Harness::new();
        let rig = h.rig(json!({})).await;
        rig.up().await;
        let pids = rig.pids();
        assert_eq!(pids.len(), 1);
        let pid = nix::unistd::Pid::from_raw(i32::try_from(pids[0]).unwrap());
        let group = nix::unistd::getpgid(Some(pid)).unwrap();
        assert_eq!(group, pid);
        assert_ne!(group, nix::unistd::getpgid(None).unwrap());
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_huge_stderr_is_bounded_and_the_last_line_still_explains() {
    limited(async {
        let mut h = Harness::new();
        let rig = h.rig(json!({"stderr": "the actual reason", "stderr_noise": 20000, "exit_before_hello": 4})).await;
        rig.state("down").await;
        assert_eq!(rig.error(), "the node exited (4): the actual reason");
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_signal_exit_is_reported() {
    limited(async {
        let mut h = Harness::new();
        let rig = h
            .rig_with(
                json!({"stderr": "about to be killed", "silent_after_hello": true}),
                RigOpts {
                    timing: timing(|t| t.silent_s = 60.0),
                    ..RigOpts::default()
                },
            )
            .await;
        rig.up().await;
        let pids = rig.pids();
        assert_eq!(pids.len(), 1);
        nix::sys::signal::kill(
            nix::unistd::Pid::from_raw(i32::try_from(pids[0]).unwrap()),
            nix::sys::signal::Signal::SIGTERM,
        )
        .unwrap();
        rig.state("down").await;
        assert!(
            rig.error().contains("stopped by signal 15"),
            "{}",
            rig.error()
        );
        h.stop().await;
    })
    .await;
}

#[test]
fn test_fixed_argv_is_reused_on_every_attempt() {
    let f = NodeLink::fixed_argv(vec!["a".into()]);
    assert_eq!(f().unwrap(), f().unwrap());
}
