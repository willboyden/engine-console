//! Port of backend/tests/core/test_scripts.py: lifecycle scripts registered in node policies (listing, validation
//! against the node-reported policy, jobs, masking, one-job-at-a-time, SSE) and the script instances' start/stop/
//! restart that map onto them.

#![allow(
    clippy::redundant_clone,
    clippy::cloned_ref_to_slice_refs,
    clippy::uninlined_format_args,
    clippy::manual_let_else,
    clippy::expect_used,
    clippy::panic,
    clippy::unwrap_used,
    clippy::too_many_lines,
    clippy::needless_pass_by_value,
    clippy::too_many_arguments,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    clippy::type_complexity
)]

#[path = "common/kit.rs"]
mod kit;

use kit::fakefleet;

#[path = "common/wse_kit.rs"]
mod wse_kit;

use std::sync::Arc;
use std::time::Duration;

use ec_hub::domain::errors::NodeError;
use ec_hub::services::scripts::{MAX_JOBS, ScriptService};
use fakefleet::{
    ContainerOpts, EndpointOpts, FakeNode, FleetEnv, Gate, HelloOpts, ScriptRun, big_model_script,
    ports, until,
};
use futures::StreamExt;
use serde_json::{Value, json};
use wse_kit::{P, problem};

fn register(node: &FakeNode, kw: Value) {
    let mut s = big_model_script();
    for (k, v) in kw.as_object().cloned().unwrap_or_default() {
        s[k] = v;
    }
    node.set_policy("scripts", json!([s]));
}

fn script_calls(node: &FakeNode) -> Vec<(String, String, Option<String>)> {
    node.st().script_calls.clone()
}

async fn finished(s: &ScriptService, job_id: &str) -> ec_hub::domain::fleet::ScriptJob {
    tokio::time::timeout(Duration::from_secs(5), s.wait(job_id))
        .await
        .expect("the job ends");
    s.job(job_id).unwrap()
}

fn run(lines: &[(&str, &str)]) -> ScriptRun {
    ScriptRun::lines(lines)
}

// ---- what the policies register ------------------------------------------------------------------------------------------------
#[tokio::test]
async fn test_scripts_are_listed_per_host_with_actions_status_and_container_patterns() {
    let mut env = FleetEnv::new();
    register(&env.node, json!({}));
    let spark = env.add_host("spark", "Spark", HelloOpts::default());
    register(&spark, json!({"name": "other", "label": "Other"}));
    env.node.st().script_status.insert(
        "big-model".into(),
        json!({"name": "big-model", "status": "up", "detail": "1 container", "checked_at": 5.0}),
    );
    let c = env.container();
    let items = c.scripts.scripts(None).unwrap();
    let ids: Vec<(&str, &str)> = items
        .iter()
        .map(|s| (s.host_id.as_str(), s.name.as_str()))
        .collect();
    assert_eq!(ids, [("local", "big-model"), ("spark", "other")]);
    let s = &items[0];
    assert_eq!(
        (
            s.label.as_str(),
            s.engine.as_deref(),
            s.status.as_str(),
            s.detail.as_deref(),
            s.checked_at
        ),
        (
            "Big model (2 GPUs)",
            Some("sglang"),
            "up",
            Some("1 container"),
            Some(5.0)
        )
    );
    assert_eq!(s.containers, ["myproj-sglang-*"]);
    let actions: Vec<(&str, bool, bool, Vec<String>)> = s
        .actions
        .iter()
        .map(|a| (a.name.as_str(), a.confirm, a.danger, a.args.clone()))
        .collect();
    assert_eq!(
        actions,
        [
            ("up", true, false, vec![]),
            ("down", false, true, vec![]),
            ("status", false, false, vec![]),
            (
                "switch",
                true,
                false,
                vec!["model-a".to_string(), "model-b".to_string()]
            )
        ]
    );
    assert_eq!(items[1].status, "unknown"); // no status pushed yet
    assert_eq!(
        c.scripts
            .scripts(Some("spark"))
            .unwrap()
            .iter()
            .map(|i| i.name.as_str())
            .collect::<Vec<_>>(),
        ["other"]
    );
    assert!(!serde_json::to_string(&items).unwrap().contains("argv")); // the hub never sees a script's argv
}

#[tokio::test]
async fn test_an_unreachable_hosts_script_status_is_unknown_not_last_seen() {
    let mut env = FleetEnv::new();
    register(&env.node, json!({}));
    env.node.st().script_status.insert(
        "big-model".into(),
        json!({"name": "big-model", "status": "up", "detail": "x"}),
    );
    env.node.set_state("down", None);
    let s = env.container().scripts.scripts(None).unwrap()[0].clone();
    assert!(s.status == "unknown" && s.detail.unwrap_or_default().contains("not connected"));
}

#[tokio::test]
async fn test_api_lists_scripts_as_a_page() {
    let mut env = FleetEnv::new();
    register(&env.node, json!({}));
    let body = env.get(&format!("{P}/scripts")).send().await.json();
    assert!(
        body["items"]
            .as_array()
            .unwrap()
            .iter()
            .map(|i| i["name"].clone())
            .collect::<Vec<_>>()
            == [json!("big-model")]
    );
    assert_eq!(body["next_cursor"], Value::Null);
    assert_eq!(
        env.get(&format!("{P}/scripts?host=spark"))
            .send()
            .await
            .json()["items"],
        json!([])
    );
    assert_eq!(
        env.get(&format!("{P}/scripts?host=Bad%20Host"))
            .send()
            .await
            .status,
        422
    );
}

// ---- validation before anything is sent ------------------------------------------------------------------------------------------
/// Security control: host, script, action and the one argument are checked against the reported policy first.
#[tokio::test]
async fn test_run_validates_host_script_action_and_argument_against_the_reported_policy() {
    let mut env = FleetEnv::new();
    register(&env.node, json!({}));
    let c = env.container();
    let s = &c.scripts;
    let code = |e: ec_hub::domain::errors::AppError| problem(&e);
    let e1 = code(s.run("nope", "big-model", "up", None).await.unwrap_err());
    assert_eq!((e1.status, e1.code.as_str()), (404, "unknown_host"));
    let e2 = code(
        s.run("local", "not-a-script", "up", None)
            .await
            .unwrap_err(),
    );
    assert_eq!((e2.status, e2.code.as_str()), (404, "unknown_script"));
    let e3 = code(
        s.run("local", "big-model", "format-disk", None)
            .await
            .unwrap_err(),
    );
    assert!(
        e3.status == 404
            && e3.code == "unknown_action"
            && e3.detail.contains("up, down, status, switch")
    );
    let e4 = code(
        s.run("local", "big-model", "switch", None)
            .await
            .unwrap_err(),
    );
    assert_eq!((e4.status, e4.code.as_str()), (400, "arg_required"));
    let e5 = code(
        s.run("local", "big-model", "switch", Some("model-c; rm -rf /"))
            .await
            .unwrap_err(),
    );
    assert_eq!((e5.status, e5.code.as_str()), (400, "invalid_arg"));
    let e6 = code(
        s.run("local", "big-model", "up", Some("surprise"))
            .await
            .unwrap_err(),
    );
    assert!(e6.status == 400 && e6.code == "invalid_arg" && e6.detail.contains("takes none"));
    assert!(script_calls(&env.node).is_empty() && env.node.ops("scripts.run").is_empty()); // nothing reached the node
}

/// Security control: a node that is down or in monitor mode is never asked to run a script.
#[tokio::test]
async fn test_run_refuses_when_the_node_is_down_or_in_monitor_mode() {
    let mut env = FleetEnv::new();
    register(&env.node, json!({}));
    env.node.set_mode("monitor");
    let c = env.container();
    let e1 = problem(
        &c.scripts
            .run("local", "big-model", "up", None)
            .await
            .unwrap_err(),
    );
    assert!(e1.status == 409 && e1.code == "monitor_only" && e1.detail.contains("testbox"));
    env.node.set_mode("manage");
    env.node.set_state("down", Some("no route to host"));
    let e2 = problem(
        &c.scripts
            .run("local", "big-model", "up", None)
            .await
            .unwrap_err(),
    );
    assert!(e2.status == 503 && e2.detail.contains("no route to host"));
    assert!(script_calls(&env.node).is_empty());
}

// ---- jobs --------------------------------------------------------------------------------------------------------------------------
#[tokio::test]
async fn test_a_job_streams_the_scripts_lines_and_completes_with_the_return_code() {
    let mut env = FleetEnv::new();
    register(&env.node, json!({}));
    env.node.script_run(
        "big-model",
        "up",
        run(&[
            ("out", "starting engine"),
            ("err", "warming up"),
            ("out", "done"),
        ]),
    );
    let c = env.container();
    let job = c
        .scripts
        .run("local", "big-model", "up", None)
        .await
        .unwrap();
    assert!(
        job.state == "running"
            && job.host_id == "local"
            && job.name == "big-model"
            && job.action == "up"
    );
    let done = finished(&c.scripts, &job.id).await;
    assert!(done.state == "completed" && done.rc == Some(0) && done.finished_at.is_some());
    assert_eq!(
        done.output_tail.lines().collect::<Vec<_>>(),
        ["starting engine", "warming up", "done"]
    );
    assert_eq!(
        script_calls(&env.node),
        [("big-model".to_string(), "up".to_string(), None)]
    );
    assert_eq!(
        env.node.ops("scripts.run"),
        [wse_kit::obj(json!({"name": "big-model", "action": "up"}))]
    ); // no `arg` key
}

#[tokio::test]
async fn test_the_one_allowed_argument_is_passed_on() {
    let mut env = FleetEnv::new();
    register(&env.node, json!({}));
    let c = env.container();
    let job = c
        .scripts
        .run("local", "big-model", "switch", Some("model-b"))
        .await
        .unwrap();
    finished(&c.scripts, &job.id).await;
    assert_eq!(
        env.node.ops("scripts.run"),
        [wse_kit::obj(
            json!({"name": "big-model", "action": "switch", "arg": "model-b"})
        )]
    );
    assert_eq!(
        c.scripts.job(&job.id).unwrap().arg.as_deref(),
        Some("model-b")
    );
}

#[tokio::test]
async fn test_a_nonzero_return_code_is_a_failed_job() {
    let mut env = FleetEnv::new();
    register(&env.node, json!({}));
    env.node.script_run(
        "big-model",
        "up",
        ScriptRun {
            rc: 3,
            ..run(&[("err", "no GPU memory")])
        },
    );
    let c = env.container();
    let job = c
        .scripts
        .run("local", "big-model", "up", None)
        .await
        .unwrap();
    let done = finished(&c.scripts, &job.id).await;
    assert!(
        done.state == "failed" && done.rc == Some(3) && done.output_tail.contains("no GPU memory")
    );
}

#[tokio::test]
async fn test_a_node_error_mid_run_is_a_failed_job_with_the_error_in_its_output() {
    let mut env = FleetEnv::new();
    register(&env.node, json!({}));
    env.node.script_run(
        "big-model",
        "up",
        ScriptRun {
            error: Some(NodeError::new("local", "timeout", "script ran too long")),
            ..run(&[("out", "half way")])
        },
    );
    let c = env.container();
    let job = c
        .scripts
        .run("local", "big-model", "up", None)
        .await
        .unwrap();
    let done = finished(&c.scripts, &job.id).await;
    assert!(done.state == "failed" && done.rc.is_none());
    assert!(
        done.output_tail.contains("half way")
            && done.output_tail.contains("[timeout] script ran too long")
    );
}

#[tokio::test]
async fn test_a_refusal_by_the_node_at_open_time_is_a_failed_job() {
    let mut env = FleetEnv::new();
    register(&env.node, json!({}));
    env.node.fail(
        "scripts.run",
        NodeError::new("local", "busy", "the node is busy"),
    );
    let c = env.container();
    let job = c
        .scripts
        .run("local", "big-model", "up", None)
        .await
        .unwrap();
    let done = finished(&c.scripts, &job.id).await;
    assert!(done.state == "failed" && done.output_tail.contains("[busy]"));
}

/// Security control: script output is masked in the tail and in what late subscribers see.
#[tokio::test]
async fn test_output_is_masked_in_the_tail_and_in_what_late_subscribers_see() {
    let mut env = FleetEnv::new();
    register(&env.node, json!({}));
    let leak = "hf_abcdefghijkl1234";
    let l1 = format!("token {leak}");
    env.node.script_run(
        "big-model",
        "up",
        run(&[
            ("out", l1.as_str()),
            ("out", "Authorization: Bearer abcdefghijklmnop"),
            ("err", "key ec_abcdefghijklmnopqrstuv"),
        ]),
    );
    let c = env.container();
    let job = c
        .scripts
        .run("local", "big-model", "up", None)
        .await
        .unwrap();
    let done = finished(&c.scripts, &job.id).await;
    assert!(
        !done.output_tail.contains(leak)
            && !done.output_tail.contains("abcdefghijklmnop")
            && !done.output_tail.contains("ec_abcdef")
    );
    assert_eq!(done.output_tail.matches("[redacted]").count(), 3);
    let events: Vec<(String, String)> = c.scripts.stream(&job.id, false).unwrap().collect().await;
    assert!(!format!("{events:?}").contains(leak) && events.last().unwrap().0 == "end");
}

#[tokio::test]
async fn test_the_tail_is_bounded_and_keeps_the_end() {
    let mut env = FleetEnv::new();
    register(&env.node, json!({}));
    let lines: Vec<String> = (0..1000)
        .map(|i| format!("line {i:05} {}", "x".repeat(80)))
        .collect();
    let pairs: Vec<(&str, &str)> = lines.iter().map(|l| ("out", l.as_str())).collect();
    env.node.script_run("big-model", "up", run(&pairs));
    let c = env.container();
    let job = c
        .scripts
        .run("local", "big-model", "up", None)
        .await
        .unwrap();
    let done = finished(&c.scripts, &job.id).await;
    assert!(
        done.output_tail.chars().count() <= 20_000
            && done
                .output_tail
                .lines()
                .last()
                .unwrap()
                .starts_with("line 00999")
    );
}

#[tokio::test]
async fn test_control_characters_never_split_one_line_into_two() {
    let mut env = FleetEnv::new();
    register(&env.node, json!({}));
    env.node.script_run(
        "big-model",
        "up",
        run(&[("out", "a\x1b[31mred\x0cfeed sep")]),
    );
    let c = env.container();
    let job = c
        .scripts
        .run("local", "big-model", "up", None)
        .await
        .unwrap();
    let done = finished(&c.scripts, &job.id).await;
    assert!(
        ec_hub::services::common::py_splitlines(&done.output_tail).len() == 1
            && !done.output_tail.contains('\x1b')
    );
}

// ---- one job per (host, script) -----------------------------------------------------------------------------------------------------
/// Security control: one job per (host, script) at a time.
#[tokio::test]
async fn test_a_second_run_of_the_same_script_is_refused_until_the_first_ends() {
    let mut env = FleetEnv::new();
    register(&env.node, json!({}));
    let gate = Gate::new();
    env.node.script_run(
        "big-model",
        "up",
        ScriptRun {
            gate: Some(gate.clone()),
            ..run(&[("out", "started")])
        },
    );
    let c = env.container();
    let first = c
        .scripts
        .run("local", "big-model", "up", None)
        .await
        .unwrap();
    let node = Arc::clone(&env.node);
    until(
        move || !node.st().script_calls.is_empty(),
        5.0,
        "the first run to reach the node",
    )
    .await;
    let e = problem(
        &c.scripts
            .run("local", "big-model", "down", None)
            .await
            .unwrap_err(),
    );
    assert!(e.code == "script_busy" && e.detail.contains(&first.id));
    assert_eq!(
        c.scripts.busy("local"),
        [("big-model".to_string(), "up".to_string())]
    );
    gate.set();
    assert_eq!(finished(&c.scripts, &first.id).await.state, "completed");
    assert!(c.scripts.busy("local").is_empty());
    let second = c
        .scripts
        .run("local", "big-model", "down", None)
        .await
        .unwrap(); // free again
    assert_eq!(finished(&c.scripts, &second.id).await.state, "completed");
}

#[tokio::test]
async fn test_other_scripts_and_other_hosts_run_in_parallel() {
    let mut env = FleetEnv::new();
    let mut small = big_model_script();
    small["name"] = json!("small-model");
    small["label"] = json!("Small");
    env.node
        .set_policy("scripts", json!([big_model_script(), small]));
    let spark = env.add_host("spark", "Spark", HelloOpts::default());
    register(&spark, json!({}));
    let gate = Gate::new();
    for node in [&env.node, &spark] {
        for name in ["big-model", "small-model"] {
            node.script_run(
                name,
                "up",
                ScriptRun {
                    gate: Some(gate.clone()),
                    ..run(&[("out", "x")])
                },
            );
        }
    }
    let c = env.container();
    let jobs = vec![
        c.scripts
            .run("local", "big-model", "up", None)
            .await
            .unwrap(),
        c.scripts
            .run("local", "small-model", "up", None)
            .await
            .unwrap(),
        c.scripts
            .run("spark", "big-model", "up", None)
            .await
            .unwrap(),
    ];
    let (n1, n2) = (Arc::clone(&env.node), Arc::clone(&spark));
    until(
        move || n1.st().script_calls.len() == 2 && n2.st().script_calls.len() == 1,
        5.0,
        "all three runs to reach their nodes",
    )
    .await;
    assert!(
        jobs.iter()
            .all(|j| c.scripts.job(&j.id).unwrap().state == "running")
    );
    gate.set();
    for j in &jobs {
        assert_eq!(finished(&c.scripts, &j.id).await.state, "completed");
    }
}

#[tokio::test]
async fn test_finished_jobs_are_kept_up_to_a_bound_and_running_ones_never_evicted() {
    let mut env = FleetEnv::new();
    register(&env.node, json!({}));
    let c = env.container();
    for _ in 0..(MAX_JOBS + 5) {
        let j = c
            .scripts
            .run("local", "big-model", "status", None)
            .await
            .unwrap();
        finished(&c.scripts, &j.id).await;
    }
    assert!(c.scripts.jobs().len() <= MAX_JOBS + 1);
}

// ---- restart = down then up ------------------------------------------------------------------------------------------------------------
#[tokio::test]
async fn test_restart_is_down_then_up_as_one_job() {
    let mut env = FleetEnv::new();
    register(&env.node, json!({}));
    let c = env.container();
    let job = c.scripts.restart("local", "big-model").await.unwrap();
    assert_eq!(job.action, "restart");
    let done = finished(&c.scripts, &job.id).await;
    let actions: Vec<String> = script_calls(&env.node).into_iter().map(|c| c.1).collect();
    assert!(done.state == "completed" && actions == ["down", "up"]);
    assert_eq!(done.output_tail.lines().next(), Some("== down =="));
}

/// Security control: a restart never starts what failed to stop.
#[tokio::test]
async fn test_restart_never_starts_what_failed_to_stop() {
    let mut env = FleetEnv::new();
    register(&env.node, json!({}));
    env.node.script_run(
        "big-model",
        "down",
        ScriptRun {
            rc: 1,
            ..run(&[("err", "cannot stop")])
        },
    );
    let c = env.container();
    let job = c.scripts.restart("local", "big-model").await.unwrap();
    let done = finished(&c.scripts, &job.id).await;
    let actions: Vec<String> = script_calls(&env.node).into_iter().map(|c| c.1).collect();
    assert!(done.state == "failed" && done.rc == Some(1) && actions == ["down"]);
}

// ---- SSE ----------------------------------------------------------------------------------------------------------------------------------
#[tokio::test]
async fn test_stream_replays_history_for_a_late_subscriber_then_ends() {
    let mut env = FleetEnv::new();
    register(&env.node, json!({}));
    env.node
        .script_run("big-model", "up", run(&[("out", "one"), ("out", "two")]));
    let c = env.container();
    let job = c
        .scripts
        .run("local", "big-model", "up", None)
        .await
        .unwrap();
    finished(&c.scripts, &job.id).await;
    let events: Vec<(String, String)> = c.scripts.stream(&job.id, false).unwrap().collect().await;
    assert_eq!(
        events.iter().map(|e| e.0.as_str()).collect::<Vec<_>>(),
        ["line", "line", "end"]
    );
    assert_eq!(
        events[..2].iter().map(|e| e.1.as_str()).collect::<Vec<_>>(),
        ["one", "two"]
    );
    assert_eq!(
        serde_json::from_str::<Value>(&events[2].1).unwrap()["state"],
        "completed"
    );
}

#[tokio::test]
async fn test_stream_follows_a_running_job_live() {
    let mut env = FleetEnv::new();
    register(&env.node, json!({}));
    let gate = Gate::new();
    env.node.script_run(
        "big-model",
        "up",
        ScriptRun {
            gate: Some(gate.clone()),
            ..run(&[("out", "first"), ("out", "second")])
        },
    );
    let c = env.container();
    let job = c
        .scripts
        .run("local", "big-model", "up", None)
        .await
        .unwrap();
    let seen = Arc::new(std::sync::Mutex::new(Vec::<(String, String)>::new()));
    let mut stream = c.scripts.stream(&job.id, false).unwrap();
    let s2 = Arc::clone(&seen);
    let task = tokio::spawn(async move {
        while let Some(ev) = stream.next().await {
            s2.lock().unwrap().push(ev);
        }
    });
    let s3 = Arc::clone(&seen);
    until(
        move || !s3.lock().unwrap().is_empty(),
        5.0,
        "the first line",
    )
    .await;
    assert_eq!(
        *seen.lock().unwrap(),
        [("line".to_string(), "first".to_string())]
    ); // live: the job is still running
    gate.set();
    tokio::time::timeout(Duration::from_secs(2), task)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        seen.lock()
            .unwrap()
            .iter()
            .map(|e| e.0.clone())
            .collect::<Vec<_>>(),
        ["line", "line", "end"]
    );
}

#[tokio::test]
async fn test_stream_once_sends_what_there_is_and_closes() {
    let mut env = FleetEnv::new();
    register(&env.node, json!({}));
    let gate = Gate::new();
    env.node.script_run(
        "big-model",
        "up",
        ScriptRun {
            gate: Some(gate.clone()),
            ..run(&[("out", "first"), ("out", "second")])
        },
    );
    let c = env.container();
    let job = c
        .scripts
        .run("local", "big-model", "up", None)
        .await
        .unwrap();
    let (s, id) = (Arc::clone(&c), job.id.clone());
    until(
        move || !s.scripts.job(&id).unwrap().output_tail.is_empty(),
        5.0,
        "the first line",
    )
    .await;
    let events: Vec<(String, String)> = c.scripts.stream(&job.id, true).unwrap().collect().await;
    assert_eq!(
        events.iter().map(|e| e.0.as_str()).collect::<Vec<_>>(),
        ["line", "end"]
    );
    assert_eq!(
        serde_json::from_str::<Value>(&events[1].1).unwrap()["state"],
        "running"
    );
    gate.set();
    finished(&c.scripts, &job.id).await;
}

#[tokio::test]
async fn test_api_run_job_and_stream() {
    let mut env = FleetEnv::new();
    register(&env.node, json!({}));
    env.node.script_run(
        "big-model",
        "up",
        run(&[("out", "hello"), ("out", "token hf_abcdefghijkl1234")]),
    );
    let r = env
        .post(&format!("{P}/scripts/local/big-model/up"))
        .send()
        .await;
    assert_eq!(r.status, 202, "{}", r.text());
    let job = r.json();
    assert!(job["state"] == "running" && job["action"] == "up");
    let id = job["id"].as_str().unwrap().to_string();
    let c = env.container();
    finished(&c.scripts, &id).await;
    let got = env
        .get(&format!("{P}/scripts/jobs/{id}"))
        .send()
        .await
        .json();
    let tail = got["output_tail"].as_str().unwrap();
    assert!(got["state"] == "completed" && tail.contains("hello") && !tail.contains("hf_abcdef"));
    let stream = env
        .get(&format!("{P}/scripts/jobs/{id}/stream"))
        .send()
        .await;
    assert!(
        stream
            .header("content-type")
            .unwrap()
            .starts_with("text/event-stream")
    );
    let text = stream.text();
    assert!(
        text.matches("event: line").count() == 2
            && text.contains("event: end")
            && !text.contains("hf_abcdef")
    );
    let once = env
        .get(&format!("{P}/scripts/jobs/{id}/stream?once=true"))
        .send()
        .await
        .text();
    assert!(once.contains("event: end"));
    assert_eq!(
        env.get(&format!("{P}/scripts/jobs/nope"))
            .send()
            .await
            .status,
        404
    );
    assert_eq!(
        env.get(&format!("{P}/scripts/jobs/nope/stream"))
            .send()
            .await
            .code(),
        "unknown_job"
    );
}

/// Security control: hostile script/action names and arguments are refused before any job exists.
#[tokio::test]
async fn test_api_validation_errors_are_problems() {
    let mut env = FleetEnv::new();
    register(&env.node, json!({}));
    assert_eq!(
        env.post(&format!("{P}/scripts/local/nope/up"))
            .send()
            .await
            .code(),
        "unknown_script"
    );
    assert_eq!(
        env.post(&format!("{P}/scripts/local/big-model/nope"))
            .send()
            .await
            .code(),
        "unknown_action"
    );
    assert_eq!(
        env.post(&format!("{P}/scripts/nope/big-model/up"))
            .send()
            .await
            .code(),
        "unknown_host"
    );
    let bad = env
        .post(&format!("{P}/scripts/local/big-model/switch"))
        .json(&json!({"arg": "model-c"}))
        .send()
        .await;
    assert!(bad.status == 400 && bad.code() == "invalid_arg");
    let ok = env
        .post(&format!("{P}/scripts/local/big-model/switch"))
        .json(&json!({"arg": "model-a"}))
        .send()
        .await;
    assert!(ok.status == 202 && ok.json()["arg"] == "model-a");
    let c = env.container();
    finished(&c.scripts, ok.json()["id"].as_str().unwrap()).await;
    for evil in ["..%2Fx", "a%20b", &"x".repeat(40), "-x"] {
        let st = env
            .post(&format!("{P}/scripts/local/{evil}/up"))
            .send()
            .await
            .status;
        assert!([404, 405, 422].contains(&st), "{evil} {st}"); // never a job
    }
    assert_eq!(script_calls(&env.node).len(), 1);
}

/// Security control: running a script is admin-only; reading is fine for a viewer.
#[tokio::test]
async fn test_running_a_script_is_admin_only() {
    let mut env = FleetEnv::new();
    register(&env.node, json!({}));
    let c = env.container();
    c.settings.ensure_bootstrap_key(&env.cfg.data_dir).unwrap();
    let boot = std::fs::read_to_string(env.cfg.data_dir.join("bootstrap-admin.key"))
        .unwrap()
        .trim()
        .to_string();
    let remote = "203.0.113.9";
    let viewer = env
        .post(&format!("{P}/keys"))
        .client(remote)
        .bearer(&boot)
        .json(&json!({"name": "v", "role": "viewer"}))
        .send()
        .await
        .json()["secret"]
        .as_str()
        .unwrap()
        .to_string();
    assert_eq!(
        env.get(&format!("{P}/scripts"))
            .client(remote)
            .bearer(&viewer)
            .send()
            .await
            .status,
        200
    );
    assert_eq!(
        env.post(&format!("{P}/scripts/local/big-model/up"))
            .client(remote)
            .bearer(&viewer)
            .send()
            .await
            .status,
        403
    );
    assert!(script_calls(&env.node).is_empty());
    let r = env
        .post(&format!("{P}/scripts/local/big-model/up"))
        .client(remote)
        .bearer(&boot)
        .send()
        .await;
    assert_eq!(r.status, 202);
    finished(&c.scripts, r.json()["id"].as_str().unwrap()).await;
}

// ---- script instances: start / stop / restart map to up / down ---------------------------------------------------------------------------------
#[tokio::test]
async fn test_script_instance_start_and_stop_run_the_scripts_up_and_down() {
    let mut env = FleetEnv::new();
    register(&env.node, json!({}));
    let inst = env
        .get(&format!("{P}/instances/scr-local-big-model"))
        .send()
        .await
        .json();
    assert!(inst["state"] == "stopped" && inst["controls"]["start"] == json!(true));
    let started = env
        .post(&format!("{P}/instances/scr-local-big-model/start"))
        .send()
        .await;
    let body = started.json();
    assert!(
        started.status == 200
            && body["state"] == "starting"
            && body["phase"]
                .as_str()
                .unwrap()
                .contains("script: up is running")
    );
    let c = env.container();
    finished(&c.scripts, &c.scripts.jobs()[0].id).await;
    assert_eq!(
        script_calls(&env.node),
        [("big-model".to_string(), "up".to_string(), None)]
    );
    env.node.add_container(
        "myproj-sglang-a",
        ContainerOpts {
            engine: Some("sglang".into()),
            host_ports: ports(18510, 30000, "0.0.0.0"),
            ..ContainerOpts::default()
        },
    );
    env.node.add_endpoint(
        18510,
        EndpointOpts {
            container: Some("myproj-sglang-a".into()),
            engine: Some("sglang".into()),
            models: vec!["m".into()],
            ..EndpointOpts::default()
        },
    );
    assert_eq!(
        env.get(&format!("{P}/instances/scr-local-big-model"))
            .send()
            .await
            .json()["state"],
        "ready"
    );
    assert_eq!(
        env.post(&format!("{P}/instances/scr-local-big-model/start"))
            .send()
            .await
            .status,
        409
    ); // no `start`
    let stopped = env
        .post(&format!("{P}/instances/scr-local-big-model/stop"))
        .send()
        .await;
    assert!(stopped.status == 200 && stopped.json()["state"] == "stopping");
    finished(&c.scripts, &c.scripts.jobs()[0].id).await;
    let actions: Vec<String> = script_calls(&env.node).into_iter().map(|c| c.1).collect();
    assert_eq!(actions, ["up", "down"]);
}

/// Security control: a script's engine is restarted through the script only, never through docker.
#[tokio::test]
async fn test_script_instance_restart_is_down_then_up_and_never_touches_docker() {
    let mut env = FleetEnv::new();
    register(&env.node, json!({}));
    let r = env
        .post(&format!("{P}/instances/scr-local-big-model/restart"))
        .send()
        .await;
    assert_eq!(r.status, 200, "{}", r.text());
    let c = env.container();
    finished(&c.scripts, &c.scripts.jobs()[0].id).await;
    let actions: Vec<String> = script_calls(&env.node).into_iter().map(|c| c.1).collect();
    assert_eq!(actions, ["down", "up"]);
    assert!(!env.node.calls().iter().any(|(o, _)| {
        ["docker.stop", "docker.rm", "docker.run", "docker.external"].contains(&o.as_str())
    }));
}

#[tokio::test]
async fn test_script_instance_without_the_action_or_in_monitor_mode_refuses_with_the_reason() {
    let mut env = FleetEnv::new();
    register(&env.node, json!({}));
    env.node.set_mode("monitor");
    let r = env
        .post(&format!("{P}/instances/scr-local-big-model/start"))
        .send()
        .await;
    assert!(
        r.status == 409
            && r.code() == "instance_not_managed"
            && r.json()["detail"]
                .as_str()
                .unwrap()
                .contains("monitor mode")
    );
    env.node.set_mode("manage");
    let mut s = big_model_script();
    let actions: Vec<Value> = s["actions"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|a| a["name"] != "up")
        .cloned()
        .collect();
    s["actions"] = Value::Array(actions);
    env.node.set_policy("scripts", json!([s]));
    assert_eq!(
        env.post(&format!("{P}/instances/scr-local-big-model/start"))
            .send()
            .await
            .status,
        409
    );
    for path in ["logs", "command"] {
        assert_eq!(
            env.get(&format!("{P}/instances/scr-local-big-model/{path}"))
                .send()
                .await
                .status,
            409
        );
    }
    assert_eq!(
        env.delete(&format!("{P}/instances/scr-local-big-model"))
            .send()
            .await
            .status,
        409
    );
    assert!(script_calls(&env.node).is_empty());
}
