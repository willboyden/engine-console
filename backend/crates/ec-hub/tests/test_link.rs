//! Port of backend/tests/fleet/test_link.py: NodeLink against the fake node over a real subprocess: handshake,
//! sanitizing, requests, timeouts and limits.

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

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ec_hub::domain::fleet::{NodeClient, StreamItem};
use ec_hub::services::fleet::{LinkLike, LinkTiming};
use serde_json::{Value, json};
use wsb_harness::{
    GIB, Harness, RigOpts, fast, items, limited, obj, remote_host, wait_for, wait_until,
};

fake_node_entry!();

fn timing(edit: impl FnOnce(&mut LinkTiming)) -> LinkTiming {
    let mut t = fast();
    edit(&mut t);
    t
}

// ── handshake ──────────────────────────────────────────────────────────────────────────────────────────
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_handshake_brings_the_link_up() {
    limited(async {
        let mut h = Harness::new();
        let rig = h.rig(json!({})).await;
        assert_eq!(rig.st().state, "connecting");
        rig.up().await;
        let st = rig.st();
        assert!(st.error.is_none() && st.retry_in_s.is_none() && st.notes.is_empty());
        assert!(st.connected_since.is_some() && st.last_seen.is_some());
        let hello = rig.link.latest("hello").unwrap();
        assert_eq!(hello["version"], "0.4.0-test");
        assert_eq!(hello["proto"], json!(ec_proto::PROTO));
        let uuids: Vec<&str> = hello["gpus"]
            .as_array()
            .unwrap()
            .iter()
            .map(|g| g["uuid"].as_str().unwrap())
            .collect();
        assert_eq!(uuids, ["GPU-0000", "GPU-0001"]);
        let l = Arc::clone(&rig.link);
        let sample = wait_for(move || l.latest("sample"), 8.0, "a sample").await;
        assert!(sample["mem"]["total"].as_i64().unwrap() > 0 && sample["type"] == "sample");
        assert!(rig.link.latest("disks").is_none() && rig.link.latest("inventory").is_none());
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_ping_pong_measures_rtt_and_the_clock_skew() {
    limited(async {
        let mut h = Harness::new();
        let rig = h.rig(json!({})).await;
        rig.up().await;
        let l = Arc::clone(&rig.link);
        wait_until(move || l.link().rtt_ms.is_some(), 8.0, "a pong").await;
        let st = rig.st();
        let rtt = st.rtt_ms.unwrap();
        assert!((0..1000).contains(&rtt));
        // same machine, same clock (the first sample sets it)
        let l = Arc::clone(&rig.link);
        wait_until(move || l.link().skew_s.is_some(), 8.0, "a skew").await;
        assert!(rig.st().skew_s.unwrap().abs() < 2.0);
        assert!(!rig.events("ping").is_empty());
        h.stop().await;
    })
    .await;
}

/// Subscribe to `topics` and keep the first event of each.
fn collect(
    bus: &Arc<ec_hub::services::common::EventBus>,
    topics: &[&str],
) -> (
    Arc<Mutex<HashMap<String, Value>>>,
    Vec<tokio::task::JoinHandle<()>>,
) {
    let seen: Arc<Mutex<HashMap<String, Value>>> = Arc::default();
    let mut tasks = Vec::new();
    for t in topics {
        let mut sub = bus.subscribe(t);
        let seen = Arc::clone(&seen);
        let topic = (*t).to_string();
        tasks.push(tokio::spawn(async move {
            while let Some(ev) = sub.recv().await {
                seen.lock().unwrap().entry(topic.clone()).or_insert(ev);
            }
        }));
    }
    (seen, tasks)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_pushed_messages_are_republished_on_the_bus() {
    limited(async {
        let mut h = Harness::new();
        let rig = h
            .rig_with(
                json!({"push_disks": {"mounts": [{"mount": "/", "fstype": "ext4"}]},
                       "push_inventory": {"containers": [], "endpoints": [], "scripts": []},
                       "pushes": [{"after_s": 0.05, "msg": {"type": "models_changed", "generation": 4, "t": 1.0}},
                                  {"after_s": 0.06, "msg": {"type": "metrics", "t": 1.0, "port": 18000,
                                                           "container": "ec-a", "engine": "vllm", "status": 200,
                                                           "text": "vllm:num_requests_running 1\n"}}]}),
                RigOpts { start: false, ..RigOpts::default() },
            )
            .await;
        let topics = ["node.hello", "node.sample", "node.disks", "node.inventory", "node.metrics", "node.models_changed"];
        let (seen, tasks) = collect(&rig.bus, &topics);
        rig.link.start().await;
        let s2 = Arc::clone(&seen);
        wait_until(move || s2.lock().unwrap().len() == 6, 8.0, "all six topics").await;
        for t in tasks {
            t.abort();
        }
        let seen = seen.lock().unwrap().clone();
        assert!(seen.values().all(|ev| ev["host_id"] == "local"));
        assert_eq!(seen["node.models_changed"]["msg"]["generation"], 4);
        assert_eq!(seen["node.metrics"]["msg"]["port"], 18000);
        assert!(rig.link.latest("disks").is_some() && rig.link.latest("inventory").is_some());
        // metrics and models_changed are events only: the hub keeps no copy of them
        assert!(rig.link.latest("metrics").is_none());
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_pong_and_credit_and_log_are_not_republished() {
    limited(async {
        let mut h = Harness::new();
        let rig = h
            .rig_with(
                json!({"pushes": [{"after_s": 0.05, "msg": {"type": "log", "level": "info", "msg": "hello"}}]}),
                RigOpts { start: false, ..RigOpts::default() },
            )
            .await;
        let (seen, tasks) = collect(&rig.bus, &["node.pong", "node.credit", "node.log", "node.fatal"]);
        rig.link.start().await;
        rig.up().await;
        tokio::time::sleep(Duration::from_millis(400)).await;
        for t in tasks {
            t.abort();
        }
        assert!(seen.lock().unwrap().is_empty());
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_junk_before_the_magic_line_is_skipped_and_noted() {
    limited(async {
        let junk = "Welcome to gpu-box-2\nLast login: yesterday\n";
        let mut h = Harness::new();
        let rig = h.rig(json!({"junk": junk})).await;
        rig.up().await;
        let notes = rig.st().notes;
        assert_eq!(notes.len(), 1);
        assert!(
            notes[0].contains(&format!("ignored {} byte(s)", junk.len()))
                && notes[0].contains("shell startup file")
        );
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_too_much_junk_is_refused() {
    limited(async {
        let mut h = Harness::new();
        let rig = h.rig(json!({"junk_lines": 80})).await; // 80 KiB of lines and never the protocol
        rig.state("down").await;
        assert!(
            rig.error()
                .contains("something other than the ecnode protocol")
        );
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_the_unrestricted_key_note_is_for_remote_hosts_only() {
    limited(async {
        let mut h = Harness::new();
        let remote = h
            .rig_with(
                json!({"forced_cmd": false}),
                RigOpts {
                    host: remote_host(),
                    ..RigOpts::default()
                },
            )
            .await;
        let local = h.rig(json!({"forced_cmd": false})).await;
        let restricted = h
            .rig_with(
                json!({"forced_cmd": true}),
                RigOpts {
                    host: remote_host(),
                    ..RigOpts::default()
                },
            )
            .await;
        for r in [&remote, &local, &restricted] {
            r.up().await;
        }
        let notes = remote.st().notes;
        assert_eq!(notes.len(), 1);
        assert!(notes[0].contains("UNRESTRICTED ssh key") && notes[0].contains("hosts init-key"));
        assert!(local.st().notes.is_empty() && restricted.st().notes.is_empty());
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_a_hello_without_the_forced_command_flag_counts_as_unrestricted() {
    limited(async {
        let mut h = Harness::new();
        // not a bool: cleaned to None
        let rig = h
            .rig_with(
                json!({"hello_extra": {"forced_cmd": "yes"}}),
                RigOpts {
                    host: remote_host(),
                    ..RigOpts::default()
                },
            )
            .await;
        rig.up().await;
        assert!(rig.st().notes.iter().any(|n| n.contains("UNRESTRICTED")));
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_a_protocol_mismatch_is_incompatible_and_says_which_side_to_update() {
    for (theirs, update) in [
        (2, "Update the Engine Console hub"),
        (0, "Update ecnode on This machine"),
    ] {
        limited(async {
            let mut h = Harness::new();
            let rig = h.rig(json!({"proto": theirs})).await;
            rig.state("incompatible").await;
            let st = rig.st();
            let err = st.error.clone().unwrap_or_default();
            assert!(
                err.contains(update)
                    && err.contains(&format!("protocol {theirs}"))
                    && err.contains("0.4.0-test"),
                "{err}"
            );
            assert_eq!(rig.link.last_wait_s(), Some(fast().fix_retry_s));
            assert_eq!(st.retry_in_s, Some(9));
            // the page can still show which version it is
            assert!(rig.link.latest("hello").is_some());
            let e = rig.call("docker.images", json!({})).await.unwrap_err();
            assert_eq!(e.code, "node_unavailable");
            h.stop().await;
        })
        .await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_no_hello_within_the_timeout_drops_and_kills_the_node() {
    limited(async {
        let mut h = Harness::new();
        let rig = h
            .rig_with(
                json!({"hello_delay_s": 30}),
                RigOpts {
                    timing: timing(|t| t.hello_timeout_s = 0.3),
                    ..RigOpts::default()
                },
            )
            .await;
        rig.state("down").await;
        assert!(
            rig.error().contains("no answer from the node within 0.3 s"),
            "{}",
            rig.error()
        );
        let pid = rig.pids()[0];
        wait_until(
            move || !wsb_harness::alive(pid),
            8.0,
            "the silent node to be killed",
        )
        .await;
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_a_fatal_before_the_hello_is_reported() {
    limited(async {
        let mut h = Harness::new();
        let rig = h.rig(json!({"first_frame": "fatal"})).await;
        rig.state("down").await;
        let err = rig.error();
        assert!(
            err.starts_with("the node gave up: scripted fatal: no GPU support")
                && !err.contains('\n'),
            "{err}"
        );
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_a_first_message_that_is_not_a_hello_is_a_protocol_error() {
    limited(async {
        let mut h = Harness::new();
        let rig = h.rig(json!({"first_frame": "sample"})).await;
        rig.state("down").await;
        assert!(
            rig.error()
                .contains("first message was 'sample', not hello"),
            "{}",
            rig.error()
        );
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_a_hello_without_a_protocol_version_is_refused() {
    limited(async {
        let mut h = Harness::new();
        let rig = h.rig(json!({"proto": null})).await;
        rig.state("down").await;
        assert!(rig.error().contains("protocol"), "{}", rig.error());
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_the_node_exiting_before_hello_is_explained_by_its_stderr() {
    limited(async {
        let mut h = Harness::new();
        let rig = h.rig(json!({"stderr": "Traceback (most recent call last):\nValueError: boom", "exit_before_hello": 1})).await;
        rig.state("down").await;
        assert_eq!(rig.error(), "the node exited (1): ValueError: boom");
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_a_missing_command_is_a_clear_error_and_retries() {
    limited(async {
        let mut h = Harness::new();
        let argv = ec_hub::services::fleet::NodeLink::fixed_argv(vec![
            "/nonexistent/path/to/ecnode".into(),
            "--stdio".into(),
        ]);
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
            "could not start ecnode: No such file or directory"
        );
        assert_eq!(rig.link.last_wait_s(), Some(fast().backoff_min_s));
        h.stop().await;
    })
    .await;
}

// ── sanitizing ─────────────────────────────────────────────────────────────────────────────────────────
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_hostile_fields_are_dropped_and_values_are_capped() {
    limited(async {
        let mut h = Harness::new();
        let name = format!("box\u{0}name{}", "h".repeat(400));
        let rig = h
            .rig(json!({"hello_extra": {"evil": "x", "shell": ["rm", "-rf"]},
                        "hello_patch": {"host": {"hostname": name, "ssh_agent": "/tmp/agent"},
                                        "gpu": {"driver": "570\u{0}.00"}}}))
            .await;
        rig.up().await;
        let hello = rig.link.latest("hello").unwrap();
        assert!(!hello.contains_key("evil") && !hello.contains_key("shell"));
        assert!(hello["host"].get("ssh_agent").is_none());
        let name = hello["host"]["hostname"].as_str().unwrap();
        let n = name.chars().count();
        assert!(name.starts_with("boxname") && !name.contains('\0') && 200 < n && n <= 253); // capped (253), NUL removed
        assert_eq!(hello["gpu"]["driver"], "570.00");
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_a_sample_cannot_add_gpus_the_hello_did_not_announce() {
    limited(async {
        let mut h = Harness::new();
        let rig = h.rig(json!({})).await;
        let l = Arc::clone(&rig.link);
        let sample = wait_for(move || l.latest("sample"), 8.0, "a sample").await;
        let mut keys: Vec<&String> = sample["gpus"].as_object().unwrap().keys().collect();
        keys.sort();
        assert_eq!(keys, ["GPU-0000", "GPU-0001"]); // the node also sent "GPU-ROGUE"
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_unknown_message_types_and_unknown_ids_are_ignored() {
    limited(async {
        let mut h = Harness::new();
        let rig = h
            .rig(json!({"pushes": [
                {"after_s": 0.05, "msg": {"type": "weird", "payload": {"deep": [1, 2, 3]}}},
                {"after_s": 0.06, "msg": {"type": "res", "id": 999, "ok": true, "result": {}}},
                {"after_s": 0.07, "msg": {"type": "event", "id": 999, "event": {"line": "x"}}},
                {"after_s": 0.08, "msg": {"type": "res", "id": "7", "ok": true}},
                {"after_s": 0.09, "msg": {"type": "credit", "id": 424_242, "bytes": 5}},
                {"after_s": 0.10, "msg": {"type": "credit", "id": 1, "bytes": -5}},
                {"after_s": 0.11, "msg": {"type": "pong", "id": 5555}},
                {"after_s": 0.12, "data_stream": 999, "bytes": 100},
                {"after_s": 0.13, "msg": {"type": 7}}]}))
            .await;
        rig.up().await;
        tokio::time::sleep(Duration::from_millis(600)).await;
        assert!(rig.st().state == "up" && rig.link.inflight() == 0);
        // and the link still works
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
async fn test_a_non_finite_number_drops_the_link() {
    limited(async {
        let mut h = Harness::new();
        let rig = h.rig(json!({"pushes": [{"after_s": 0.1, "raw": "{\"type\":\"sample\",\"seq\":1,\"t\":NaN}"}]})).await;
        rig.up().await;
        rig.state("down").await;
        let err = rig.error();
        assert!(err.contains("protocol error from the node") && err.contains("non-finite"), "{err}");
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_invalid_json_and_a_broken_frame_drop_the_link() {
    limited(async {
        let mut h = Harness::new();
        let rig = h
            .rig(json!({"pushes": [{"after_s": 0.1, "raw": "{not json at all"}]}))
            .await;
        rig.up().await;
        rig.state("down").await;
        assert!(
            rig.error()
                .contains("protocol error from the node: not JSON"),
            "{}",
            rig.error()
        );
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_wrongly_typed_fields_become_none_not_errors() {
    limited(async {
        let mut h = Harness::new();
        let rig = h
            .rig(json!({"pushes": [{"after_s": 0.05, "msg": {"type": "disks", "t": "yesterday", "mounts": [
                {"mount": 5, "size": "big", "roots": "all", "remote": "yes"}, {"mount": "/data", "size": 10}]}}]}))
            .await;
        rig.up().await;
        let l = Arc::clone(&rig.link);
        let disks = wait_for(move || l.latest("disks"), 8.0, "the disks message").await;
        assert!(disks["t"].is_null());
        assert_eq!(disks["mounts"][1]["mount"], "/data");
        assert_eq!(disks["mounts"][1]["size"], 10);
        let m0 = &disks["mounts"][0];
        assert!(m0["mount"].is_null() && m0["size"].is_null() && m0["remote"].is_null());
        h.stop().await;
    })
    .await;
}

// ── requests ───────────────────────────────────────────────────────────────────────────────────────────
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_call_returns_the_sanitized_result() {
    limited(async {
        let mut h = Harness::new();
        let rig = h.rig(json!({})).await;
        rig.up().await;
        let res = rig.call("docker.images", json!({})).await.unwrap();
        assert_eq!(
            Value::Object(res),
            json!({"images": [{"ref": "vllm/vllm-openai:v0.1", "id": "sha256:abc", "size": 10 * GIB,
                               "created": 1.7e9, "arch": "amd64"}]})
        );
        let sent: Vec<Value> = rig
            .events("req")
            .into_iter()
            .filter(|e| e["op"] == "docker.images")
            .collect();
        assert_eq!(sent.len(), 1);
        assert_eq!(sent[0]["args"], json!({}));
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_results_are_rebuilt_from_the_whitelist() {
    limited(async {
        let mut h = Harness::new();
        let rig = h.rig(json!({"hostile_images": true})).await;
        rig.up().await;
        let res = rig.call("docker.images", json!({})).await.unwrap();
        assert_eq!(res.keys().collect::<Vec<_>>(), ["images"]); // "evil_top" is gone
        let imgs = res["images"].as_array().unwrap();
        assert_eq!(imgs.len(), 1);
        assert_eq!(imgs[0]["ref"].as_str().unwrap().len(), 256);
        assert_eq!(imgs[0]["id"], "id");
        assert!(imgs[0].get("evil").is_none());
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_args_reach_the_node_unchanged() {
    limited(async {
        let mut h = Harness::new();
        let rig = h.rig(json!({})).await;
        rig.up().await;
        rig.call(
            "docker.image_present",
            json!({"image": "vllm/vllm-openai:v0.1"}),
        )
        .await
        .unwrap();
        let reqs: Vec<Value> = rig
            .events("req")
            .into_iter()
            .filter(|e| e["op"] == "docker.image_present")
            .collect();
        assert_eq!(reqs.len(), 1);
        assert_eq!(reqs[0]["args"], json!({"image": "vllm/vllm-openai:v0.1"}));
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_an_error_res_becomes_a_node_error_with_the_nodes_code() {
    limited(async {
        let mut h = Harness::new();
        let rig = h.rig(json!({})).await;
        rig.up().await;
        for code in [
            "not_found",
            "policy_denied",
            "monitor_only",
            "no_space",
            "timeout",
            "future_code_x",
        ] {
            let e = rig
                .call("docker.inspect", json!({"name": format!("err-{code}")}))
                .await
                .unwrap_err();
            assert_eq!(
                (e.code.as_str(), e.message.as_str(), e.host_id.as_str()),
                (code, "scripted refusal", "local")
            );
            assert_ne!(e.code, "node_unavailable");
        }
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_a_malformed_error_code_becomes_internal() {
    limited(async {
        let mut h = Harness::new();
        let rig = h.rig(json!({})).await;
        rig.up().await;
        for code in [
            "BAD CODE!".to_string(),
            "../../etc".into(),
            "x".repeat(60),
            "Not_Snake".into(),
            "9lives".into(),
        ] {
            let e = rig
                .call("docker.inspect", json!({"name": format!("err-{code}")}))
                .await
                .unwrap_err();
            assert_eq!(e.code, "internal", "{code}");
        }
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_an_error_res_without_an_error_object_is_internal_with_a_message() {
    limited(async {
        let mut h = Harness::new();
        let rig = h.rig(json!({})).await;
        rig.up().await;
        let e = rig
            .call("docker.inspect", json!({"name": "no-error-object"}))
            .await
            .unwrap_err();
        assert!(e.code == "internal" && e.message.contains("without saying why"));
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_node_error_text_is_made_safe() {
    limited(async {
        let mut h = Harness::new();
        let rig = h.rig(json!({})).await;
        rig.up().await;
        // a node's message can carry anything: control characters, newlines and token-looking text must not reach
        // logs and problem documents as they are
        let e = rig
            .call("docker.inspect", json!({"name": "ctl"}))
            .await
            .unwrap_err();
        assert_eq!(e.code, "io_error");
        assert_eq!(e.message, "disk full second line [redacted]");
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_calls_are_refused_until_the_link_is_up() {
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
        let e = rig.call("docker.images", json!({})).await.unwrap_err();
        assert!(
            e.code == "node_unavailable" && e.message.contains("not connected"),
            "{e}"
        );
        let mut stream = rig.link.open("docker.images", None);
        assert!(items(&mut stream).await.is_empty());
        assert_eq!(stream.result().await.unwrap_err().code, "node_unavailable");
        assert_eq!(
            stream.write(b"x").await.unwrap_err().code,
            "node_unavailable"
        );
        assert_eq!(stream.end().await.unwrap_err().code, "node_unavailable");
        stream.cancel().await; // harmless
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_an_unknown_op_is_refused_locally_and_nothing_is_sent() {
    limited(async {
        let mut h = Harness::new();
        let rig = h.rig(json!({})).await;
        rig.up().await;
        let e = rig.call("docker.rm_rf", json!({})).await.unwrap_err();
        assert_eq!(e.code, "unknown_op");
        assert!(rig.events("req").iter().all(|r| r["op"] != "docker.rm_rf"));
        let mut stream = rig.link.open("nope", None);
        assert_eq!(stream.result().await.unwrap_err().code, "unknown_op");
        h.stop().await;
    })
    .await;
}

/// Python also sends a NaN and a `bytes` value: neither can exist in a serde_json value, so only the message that is
/// too big for one frame is left to refuse here.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_arguments_json_cannot_carry_are_bad_args_and_nothing_is_sent() {
    limited(async {
        let mut h = Harness::new();
        let rig = h.rig(json!({})).await;
        rig.up().await;
        let e = rig
            .call("docker.stop", json!({"name": "x".repeat(2 * 1024 * 1024)}))
            .await
            .unwrap_err();
        assert_eq!(e.code, "bad_args");
        assert!(rig.events("req").is_empty());
        assert_eq!(rig.link.inflight(), 0);
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_node_info_refreshes_the_stored_hello() {
    limited(async {
        let mut h = Harness::new();
        let rig = h.rig(json!({"info_mode": "manage"})).await;
        rig.up().await;
        let (seen, tasks) = collect(&rig.bus, &["node.hello"]);
        tokio::time::sleep(Duration::from_millis(50)).await;
        let before = rig.link.latest("hello").unwrap();
        assert_eq!(before["policy"]["mode"], "monitor");
        rig.call("node.info", json!({})).await.unwrap();
        let after = rig.link.latest("hello").unwrap();
        assert!(after["policy"]["mode"] == "manage" && after["type"] == "hello");
        let uuids: Vec<&str> = after["gpus"]
            .as_array()
            .unwrap()
            .iter()
            .map(|g| g["uuid"].as_str().unwrap())
            .collect();
        assert_eq!(uuids, ["GPU-0000", "GPU-0001"]);
        let s2 = Arc::clone(&seen);
        wait_until(
            move || !s2.lock().unwrap().is_empty(),
            8.0,
            "a node.hello event",
        )
        .await;
        for t in tasks {
            t.abort();
        }
        h.stop().await;
    })
    .await;
}

// ── timeouts, cancel, limits ───────────────────────────────────────────────────────────────────────────
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_a_timeout_says_so_and_sends_cancel() {
    limited(async {
        let mut h = Harness::new();
        let rig = h.rig(json!({})).await;
        rig.up().await;
        let e = rig
            .link
            .call("models.scan", None, Duration::from_secs_f64(0.2))
            .await
            .unwrap_err();
        assert!(
            e.code == "timeout" && e.message.contains("may or may not have taken effect"),
            "{e}"
        );
        assert!(
            e.message
                .starts_with("models.scan did not answer within 0.2 s"),
            "{}",
            e.message
        );
        let reqs: Vec<Value> = rig
            .events("req")
            .into_iter()
            .filter(|r| r["op"] == "models.scan")
            .collect();
        assert_eq!(reqs.len(), 1);
        let id = reqs[0]["id"].clone();
        let path = rig.log_path.clone();
        wait_until(
            move || {
                wsb_fake_node::read_log(&path)
                    .iter()
                    .any(|c| c["ev"] == "cancel" && c["id"] == id)
            },
            8.0,
            "the cancel frame",
        )
        .await;
        // the node answers the cancel with a res, which settles the request: nothing is left in flight
        let l = Arc::clone(&rig.link);
        wait_until(
            move || l.inflight() == 0,
            8.0,
            "the cancelled request to settle",
        )
        .await;
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_a_request_the_node_never_answers_after_cancel_keeps_its_slot() {
    limited(async {
        let mut h = Harness::new();
        let rig = h.rig(json!({"ignore_cancel": true})).await;
        rig.up().await;
        let e = rig
            .link
            .call("models.scan", None, Duration::from_secs_f64(0.1))
            .await
            .unwrap_err();
        assert_eq!(e.code, "timeout");
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert_eq!(rig.link.inflight(), 1); // still counts: the node still counts it too
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_cancelling_the_caller_cancels_the_request_on_the_node() {
    limited(async {
        let mut h = Harness::new();
        let rig = h.rig(json!({})).await;
        rig.up().await;
        let l = Arc::clone(&rig.link);
        let task =
            tokio::spawn(async move { l.call("models.scan", None, Duration::from_secs(30)).await });
        let path = rig.log_path.clone();
        wait_until(
            move || {
                !wsb_fake_node::read_log(&path)
                    .iter()
                    .all(|e| e["ev"] != "req")
            },
            8.0,
            "the request to reach the node",
        )
        .await;
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        let path = rig.log_path.clone();
        wait_until(
            move || {
                wsb_fake_node::read_log(&path)
                    .iter()
                    .any(|e| e["ev"] == "cancel")
            },
            8.0,
            "the cancel frame",
        )
        .await;
        let l = Arc::clone(&rig.link);
        wait_until(
            move || l.inflight() == 0,
            8.0,
            "the cancelled request to settle",
        )
        .await;
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_at_most_64_requests_in_flight() {
    limited(async {
        let mut h = Harness::new();
        let rig = h.rig(json!({})).await;
        rig.up().await;
        let tasks: Vec<_> = (0..ec_proto::MAX_INFLIGHT)
            .map(|_| {
                let l = Arc::clone(&rig.link);
                tokio::spawn(
                    async move { l.call("models.scan", None, Duration::from_secs(30)).await },
                )
            })
            .collect();
        let l = Arc::clone(&rig.link);
        wait_until(
            move || l.inflight() == ec_proto::MAX_INFLIGHT,
            8.0,
            "64 requests in flight",
        )
        .await;
        let e = rig.call("docker.images", json!({})).await.unwrap_err();
        assert_eq!(e.code, "busy");
        let mut stream = rig.link.open("docker.images", None);
        assert_eq!(stream.result().await.unwrap_err().code, "busy");
        let path = rig.log_path.clone();
        wait_until(
            move || {
                wsb_fake_node::read_log(&path)
                    .iter()
                    .filter(|e| e["ev"] == "req")
                    .count()
                    == ec_proto::MAX_INFLIGHT
            },
            8.0,
            "the node to read all 64 requests",
        )
        .await;
        let mut ids: Vec<i64> = rig
            .events("req")
            .iter()
            .map(|r| r["id"].as_i64().unwrap())
            .collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), ec_proto::MAX_INFLIGHT);
        let mut tasks = tasks.into_iter();
        for t in tasks.by_ref().take(3) {
            // free a few slots: new requests work again
            t.abort();
            let _ = t.await;
        }
        let l = Arc::clone(&rig.link);
        wait_until(
            move || l.inflight() <= ec_proto::MAX_INFLIGHT - 3,
            8.0,
            "slots to free up",
        )
        .await;
        assert!(
            !rig.call("docker.images", json!({})).await.unwrap()["images"]
                .as_array()
                .unwrap()
                .is_empty()
        );
        for t in tasks {
            t.abort();
            let _ = t.await;
        }
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_request_ids_wrap_and_never_collide_with_one_in_flight() {
    limited(async {
        let mut h = Harness::new();
        let rig = h.rig(json!({})).await;
        rig.up().await;
        let max = ec_proto::MAX_ID;
        rig.link.set_next_id(max - 1);
        let l = Arc::clone(&rig.link);
        let hang =
            tokio::spawn(async move { l.call("models.scan", None, Duration::from_secs(30)).await }); // takes MAX_ID - 1
        let l = Arc::clone(&rig.link);
        wait_until(move || l.inflight() == 1, 8.0, "the hanging request").await;
        for _ in 0..4 {
            rig.call("docker.images", json!({})).await.unwrap(); // MAX_ID, then 1, 2, 3
        }
        rig.link.set_next_id(max - 1); // wrap onto the id still in flight
        rig.call("docker.images", json!({})).await.unwrap();
        let ids: Vec<i64> = rig
            .events("req")
            .iter()
            .map(|r| r["id"].as_i64().unwrap())
            .collect();
        assert_eq!(ids[..5], [max - 1, max, 1, 2, 3]);
        assert_eq!(ids[5], max); // skipped the in-flight id
        hang.abort();
        let _ = hang.await;
        h.stop().await;
    })
    .await;
}

// ── liveness ───────────────────────────────────────────────────────────────────────────────────────────
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_a_silent_node_is_dropped_and_pending_calls_fail() {
    limited(async {
        let mut h = Harness::new();
        let rig = h
            .rig_with(
                json!({"silent_after_hello": true}),
                RigOpts {
                    timing: timing(|t| t.silent_s = 0.5),
                    ..RigOpts::default()
                },
            )
            .await;
        rig.up().await;
        let l = Arc::clone(&rig.link);
        let call =
            tokio::spawn(
                async move { l.call("docker.images", None, Duration::from_secs(30)).await },
            );
        rig.state("down").await;
        assert!(
            rig.error().contains("no message from the node for 0.5 s"),
            "{}",
            rig.error()
        );
        let e = call.await.unwrap().unwrap_err();
        assert!(
            e.code == "node_unavailable" && e.message.contains("no message from the node"),
            "{e}"
        );
        let pid = rig.pids()[0];
        wait_until(
            move || !wsb_harness::alive(pid),
            8.0,
            "the silent node to be killed",
        )
        .await;
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_a_node_that_stops_sending_samples_turns_stale_but_stays_usable() {
    limited(async {
        let mut h = Harness::new();
        let rig = h
            .rig_with(
                json!({"samples_stop_after_s": 0.3}),
                RigOpts {
                    timing: timing(|t| t.stale_s = 0.5),
                    ..RigOpts::default()
                },
            )
            .await;
        rig.up().await;
        assert_eq!(rig.st().state, "up");
        rig.state("stale").await;
        assert!(rig.st().error.is_none() && rig.st().connected_since.is_some());
        // stale is not unavailable: pings still prove it is alive
        assert!(
            !rig.call("docker.images", json!({})).await.unwrap()["images"]
                .as_array()
                .unwrap()
                .is_empty()
        );
        assert!(rig.link.latest("sample").is_some());
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_no_samples_at_all_is_stale_measured_from_the_hello() {
    limited(async {
        let mut h = Harness::new();
        let rig = h
            .rig_with(
                json!({"samples": false}),
                RigOpts {
                    timing: timing(|t| t.stale_s = 0.4),
                    ..RigOpts::default()
                },
            )
            .await;
        rig.up().await;
        assert_eq!(rig.st().state, "up");
        rig.state("stale").await;
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_a_flood_of_json_messages_drops_the_link() {
    limited(async {
        let mut h = Harness::new();
        let rig = h
            .rig_with(
                json!({"flood": {"after_s": 0.1, "frame_kb": 64, "count": 400}}),
                RigOpts {
                    timing: timing(|t| t.flood_bytes = 2 * 1024 * 1024),
                    ..RigOpts::default()
                },
            )
            .await;
        rig.up().await;
        rig.state("down").await;
        let err = rig.error();
        assert!(
            err.contains("the node sent over 2 MB") && err.contains("dropped it"),
            "{err}"
        );
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_a_flood_of_tiny_messages_drops_the_link() {
    limited(async {
        let mut h = Harness::new();
        let rig = h
            .rig_with(
                json!({"flood": {"after_s": 0.1, "frame_kb": 0, "count": 5000}}),
                RigOpts {
                    timing: timing(|t| t.flood_frames = 300),
                    ..RigOpts::default()
                },
            )
            .await;
        rig.up().await;
        rig.state("down").await;
        assert!(rig.error().contains("messages in"), "{}", rig.error());
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_a_busy_but_honest_node_is_not_a_flood() {
    limited(async {
        let mut h = Harness::new();
        let rig = h
            .rig_with(
                json!({"sample_interval_s": 0.005}),
                RigOpts {
                    timing: timing(|t| {
                        t.flood_bytes = 10 * 1024 * 1024;
                        t.flood_frames = 100_000;
                    }),
                    ..RigOpts::default()
                },
            )
            .await;
        rig.up().await;
        tokio::time::sleep(Duration::from_secs(1)).await;
        assert_eq!(rig.st().state, "up");
        h.stop().await;
    })
    .await;
}

// ── the child process ──────────────────────────────────────────────────────────────────────────────────
/// Python plants HF_TOKEN in the console's environment; Rust code cannot change its own environment (no unsafe), so
/// every variable this test process has (cargo gives it plenty) stands in for it: none may reach the node.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_the_node_gets_only_the_environment_it_was_given() {
    limited(async {
        let mut h = Harness::new();
        let env = vec![
            (
                "PATH".to_string(),
                std::env::var("PATH").unwrap_or_else(|_| "/usr/bin:/bin".into()),
            ),
            ("LANG".to_string(), "C.UTF-8".to_string()),
        ];
        let rig = h
            .rig_with(
                json!({}),
                RigOpts {
                    env: Some(env),
                    ..RigOpts::default()
                },
            )
            .await;
        rig.up().await;
        let starts = rig.events("start");
        assert_eq!(starts.len(), 1);
        let got: Vec<String> = starts[0]["env"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        assert!(!got.contains(&"HF_TOKEN".to_string()));
        for (k, _) in std::env::vars_os() {
            let k = k.to_string_lossy().into_owned();
            if k != "PATH" && k != "LANG" {
                assert!(!got.contains(&k), "{k} reached the node");
            }
        }
        assert!(
            got.contains(&"PATH".to_string())
                && got.contains(&"LANG".to_string())
                && got.len() <= 6,
            "{got:?}"
        );
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_stop_closes_stdin_so_a_node_exits_cleanly_and_is_fast() {
    limited(async {
        let mut h = Harness::new();
        let rig = h.rig(json!({})).await;
        rig.up().await;
        let t0 = Instant::now();
        rig.link.stop().await;
        assert!(t0.elapsed() < Duration::from_secs(1));
        assert!(!rig.events("eof").is_empty()); // it saw the pipe close and left by itself
        assert_eq!(rig.st().state, "down");
        assert_eq!(rig.error(), "stopped");
        assert!(rig.link.latest("sample").is_none());
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_stop_kills_a_node_that_ignores_eof_and_fails_pending_calls() {
    limited(async {
        let mut h = Harness::new();
        let rig = h
            .rig_with(
                json!({"silent_after_hello": true}),
                RigOpts {
                    timing: timing(|t| {
                        t.exit_grace_s = 0.3;
                        t.silent_s = 60.0;
                    }),
                    ..RigOpts::default()
                },
            )
            .await;
        rig.up().await;
        let l = Arc::clone(&rig.link);
        let call =
            tokio::spawn(
                async move { l.call("docker.images", None, Duration::from_secs(30)).await },
            );
        tokio::time::sleep(Duration::from_millis(100)).await;
        rig.link.stop().await;
        assert_eq!(call.await.unwrap().unwrap_err().code, "node_unavailable");
        let pid = rig.pids()[0];
        assert!(!wsb_harness::alive(pid));
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_stop_before_the_first_connection_finishes() {
    limited(async {
        let mut h = Harness::new();
        let rig = h.rig(json!({"hello_delay_s": 30})).await;
        let path = rig.log_path.clone();
        wait_until(
            move || {
                wsb_fake_node::read_log(&path)
                    .iter()
                    .any(|e| e["ev"] == "start")
            },
            8.0,
            "the node to start",
        )
        .await;
        tokio::time::timeout(Duration::from_secs(5), rig.link.stop())
            .await
            .expect("stop in time");
        let pid = rig.pids()[0];
        assert!(!wsb_harness::alive(pid));
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_stop_on_a_link_that_never_started() {
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
        rig.link.stop().await;
        assert_eq!(rig.st().state, "down");
        h.stop().await;
    })
    .await;
}

#[test]
fn test_obj_helper() {
    assert_eq!(obj(json!({"a": 1})).len(), 1);
    let _ = StreamItem::Data(bytes::Bytes::new());
}
