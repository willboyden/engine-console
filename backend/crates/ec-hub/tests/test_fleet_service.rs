//! Port of backend/tests/fleet/test_fleet_service.py: FleetService: its hosts and links, summaries and details, the
//! "fleet" topic, and the default link factories.
//!
//! The tests that plant variables in the console's environment (secrets that must not reach a node or ssh, a fake
//! `ssh` first on PATH) run their body in a child process: this test binary again, started with that environment
//! (Rust code cannot change its own environment without `unsafe`).

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

#[path = "common/wsb_fleet_links.rs"]
mod fleet_links;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use ec_hub::config::Settings;
use ec_hub::domain::fleet::{Fleet, FleetPort, HostConfig, HostCounts, InstanceResolver};
use ec_hub::services::common::EventBus;
use ec_hub::services::fleet::FleetService;
use ec_hub::services::hosts::{LOCAL_ENV_KEYS, ssh_argv};
use fleet_links::{FakeLink, images, make};
use serde_json::{Value, json};
use wsb_fake_node::{install_fake_ssh, node_argv, read_log};
use wsb_harness::{
    alive, b_hello, disks, host, limited, make_settings, mount, obj, root, wait_until,
};

fake_node_entry!();

fn a() -> HostConfig {
    host("gpu-box-2", "GPU box 2", Some("ops@10.0.0.5"))
}

fn l() -> HostConfig {
    host("local", "Workstation", None)
}

fn settings(dir: &Path, edit: impl FnOnce(&mut Settings)) -> Arc<Settings> {
    let mut s = make_settings(dir);
    edit(&mut s);
    Arc::new(s)
}

// ── hosts and nodes ────────────────────────────────────────────────────────────────────────────────────
#[test]
fn test_hosts_and_nodes() {
    let d = tempfile::tempdir().unwrap();
    let (fleet, links, _) = make(settings(d.path(), |_| {}), vec![l(), a()], &[]);
    let mut got = fleet.hosts();
    assert_eq!(
        got.iter().map(|h| h.id.as_str()).collect::<Vec<_>>(),
        ["local", "gpu-box-2"]
    );
    got.clear();
    assert_eq!(fleet.hosts().len(), 2); // a copy: callers cannot edit the fleet
    let node = fleet.node("gpu-box-2").unwrap();
    assert_eq!(
        Arc::as_ptr(&node).cast::<()>(),
        Arc::as_ptr(&links["gpu-box-2"]).cast::<()>()
    );
}

#[test]
fn test_an_unknown_host_is_not_found_with_a_stable_code() {
    let d = tempfile::tempdir().unwrap();
    let (fleet, _, _) = make(settings(d.path(), |_| {}), vec![l(), a()], &[]);
    let e = fleet.node("nope").err().unwrap();
    assert_eq!((e.code.as_str(), e.status), ("unknown_host", 404));
    let long = fleet.node(&"x".repeat(500)).err().unwrap();
    assert!(long.detail.len() < 100); // an attacker-sized id is not echoed back whole
    assert_eq!(fleet.summary("nope").unwrap_err().code(), "unknown_host");
    assert_eq!(fleet.reconnect("nope").unwrap_err().code(), "unknown_host");
}

#[test]
fn test_two_hosts_with_one_id_are_refused() {
    let d = tempfile::tempdir().unwrap();
    let e = FleetService::with_links(
        settings(d.path(), |_| {}),
        Arc::new(EventBus::new()),
        Some(vec![a(), a()]),
        None,
    )
    .unwrap_err();
    assert!(e.to_string().contains("share an id"));
}

#[test]
fn test_summaries_follow_the_configuration_order() {
    let d = tempfile::tempdir().unwrap();
    let (fleet, _, _) = make(settings(d.path(), |_| {}), vec![a(), l()], &[]);
    assert_eq!(
        fleet
            .summaries()
            .unwrap()
            .iter()
            .map(|s| s.id.as_str())
            .collect::<Vec<_>>(),
        ["gpu-box-2", "local"]
    );
    let s = fleet.summary("gpu-box-2").unwrap();
    assert!(s.remote && s.link.state == "up" && !s.gpus.is_empty());
    assert_eq!(s.node.unwrap().version.as_deref(), Some("0.4.0"));
}

#[test]
fn test_other_services_plug_in_a_resolver_and_a_counts_provider() {
    let d = tempfile::tempdir().unwrap();
    let (fleet, _, _) = make(settings(d.path(), |_| {}), vec![l(), a()], &[]);
    let asked: Arc<Mutex<Vec<(String, String)>>> = Arc::default();
    let a2 = Arc::clone(&asked);
    let resolver: InstanceResolver = Arc::new(move |hid: &str, c: &str| {
        a2.lock().unwrap().push((hid.into(), c.into()));
        Some(format!("inst_{hid}"))
    });
    fleet.set_instance_resolver(resolver);
    fleet.set_counts_provider(Arc::new(|_: &str| HostCounts {
        engines_running: 1,
        models: 4,
        ..HostCounts::default()
    }));
    let s = fleet.summary("local").unwrap();
    assert!(
        s.gpus[0].procs[0].instance_id.as_deref() == Some("inst_local") && s.counts.models == 4
    );
    assert!(
        asked
            .lock()
            .unwrap()
            .contains(&("local".into(), "ec-engine-1".into()))
    );
    fleet.clear_providers(); // set_instance_resolver(None), set_counts_provider(None)
    let s = fleet.summary("local").unwrap();
    assert!(s.gpus[0].procs[0].instance_id.is_none() && s.counts == HostCounts::default());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_start_and_stop_reach_every_link() {
    let d = tempfile::tempdir().unwrap();
    let (fleet, links, _) = make(settings(d.path(), |_| {}), vec![l(), a()], &[]);
    fleet.start().await;
    assert!(links.values().all(|k| k.st().started));
    fleet.start().await; // idempotent for the watcher
    fleet.stop().await;
    assert!(links.values().all(|k| k.st().stopped));
    fleet.stop().await; // and stopping twice is harmless
}

// ── detail and images ──────────────────────────────────────────────────────────────────────────────────
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_detail_adds_disks_roots_scripts_addresses_and_images() {
    let d = tempfile::tempdir().unwrap();
    let (fleet, links, _) = make(settings(d.path(), |_| {}), vec![l(), a()], &[]);
    links["gpu-box-2"].set(
        "hello",
        b_hello(
            &[],
            vec![root("hf", "/data/hf", json!({}))],
            "gpu-box-2",
            json!({}),
        ),
    );
    links["gpu-box-2"].set(
        "disks",
        disks(vec![
            mount("/", json!({})),
            mount("/data", json!({"fstype": "zfs", "source": "pool/data"})),
        ]),
    );
    let det = fleet.detail("gpu-box-2").await.unwrap();
    assert_eq!(
        det.disks
            .iter()
            .map(|m| m.label.as_str())
            .collect::<Vec<_>>(),
        ["/ (ext4)", "pool (zfs)"]
    );
    assert_eq!(det.roots[0].mount.as_ref().unwrap().mount, "/data");
    assert_eq!(det.addresses, ["10.0.0.5/24 (eth0)"]);
    assert_eq!(
        det.images
            .iter()
            .map(|i| i.ref_.as_str())
            .collect::<Vec<_>>(),
        ["vllm/vllm-openai:v0.1"]
    );
    assert!(det.summary.id == "gpu-box-2" && !det.summary.gpus.is_empty());
    assert_eq!(
        fleet.detail("nope").await.unwrap_err().code(),
        "unknown_host"
    );
}

fn clock(now: &Arc<Mutex<f64>>) -> ec_hub::services::fleet::Clock {
    let n = Arc::clone(now);
    Arc::new(move || *n.lock().unwrap())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_images_are_asked_for_once_a_minute_not_on_every_detail() {
    let d = tempfile::tempdir().unwrap();
    let now = Arc::new(Mutex::new(1000.0));
    let (fleet, links, _) = make(settings(d.path(), |_| {}), vec![l(), a()], &[]);
    fleet.set_clock(clock(&now));
    let link = &links["local"];
    for _ in 0..3 {
        let det = fleet.detail("local").await.unwrap();
        assert_eq!(
            det.images
                .iter()
                .map(|i| i.ref_.as_str())
                .collect::<Vec<_>>(),
            ["vllm/vllm-openai:v0.1"]
        );
    }
    let calls = link.st().calls.clone();
    assert_eq!(
        calls.iter().map(|c| c.0.as_str()).collect::<Vec<_>>(),
        ["docker.images"]
    );
    assert!(calls[0].2 <= 5.0); // a slow node cannot hold the page for long
    *now.lock().unwrap() += 59.0;
    fleet.detail("local").await.unwrap();
    assert_eq!(link.st().calls.len(), 1);
    *now.lock().unwrap() += 2.0;
    link.result("docker.images", Ok(obj(json!({"images": []}))));
    assert!(fleet.detail("local").await.unwrap().images.is_empty());
    assert_eq!(link.st().calls.len(), 2);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_a_failing_image_call_keeps_the_old_list_and_is_not_retried_at_once() {
    let d = tempfile::tempdir().unwrap();
    let now = Arc::new(Mutex::new(1000.0));
    let (fleet, links, _) = make(settings(d.path(), |_| {}), vec![l(), a()], &[]);
    fleet.set_clock(clock(&now));
    let link = &links["local"];
    fleet.detail("local").await.unwrap();
    *now.lock().unwrap() += 61.0;
    link.result(
        "docker.images",
        Err(ec_hub::domain::errors::NodeError::new(
            "local", "timeout", "slow",
        )),
    );
    let refs = |det: &ec_hub::domain::fleet::HostDetail| {
        det.images
            .iter()
            .map(|i| i.ref_.clone())
            .collect::<Vec<_>>()
    };
    assert_eq!(
        refs(&fleet.detail("local").await.unwrap()),
        ["vllm/vllm-openai:v0.1"]
    ); // last good list
    assert_eq!(
        refs(&fleet.detail("local").await.unwrap()),
        ["vllm/vllm-openai:v0.1"]
    );
    assert_eq!(link.st().calls.len(), 2); // the failure was remembered for a few seconds
    *now.lock().unwrap() += 11.0;
    link.result("docker.images", Ok(images()));
    fleet.detail("local").await.unwrap();
    assert_eq!(link.st().calls.len(), 3);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_no_image_call_for_a_host_that_is_down_or_has_no_docker() {
    let d = tempfile::tempdir().unwrap();
    let (fleet, links, _) = make(
        settings(d.path(), |_| {}),
        vec![l(), a()],
        &[("local", "down")],
    );
    assert!(fleet.detail("local").await.unwrap().images.is_empty());
    assert!(links["local"].st().calls.is_empty());
    let mut h = wsb_harness::hello();
    h.get_mut("docker").unwrap()["available"] = json!(false);
    links["gpu-box-2"].set("hello", h);
    assert!(fleet.detail("gpu-box-2").await.unwrap().images.is_empty());
    assert!(links["gpu-box-2"].st().calls.is_empty());
    links["gpu-box-2"].st().data.remove("hello");
    assert!(fleet.detail("gpu-box-2").await.unwrap().images.is_empty());
    assert!(links["gpu-box-2"].st().calls.is_empty());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_concurrent_details_share_one_image_call() {
    let d = tempfile::tempdir().unwrap();
    let (fleet, links, _) = make(settings(d.path(), |_| {}), vec![l(), a()], &[]);
    links["local"].st().delay = 0.1;
    let fleet = Arc::new(fleet);
    let all = (0..8).map(|_| {
        let f = Arc::clone(&fleet);
        async move { f.detail("local").await.unwrap() }
    });
    futures::future::join_all(all).await;
    assert_eq!(links["local"].st().calls.len(), 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_detail_of_a_stale_host_still_asks_for_images() {
    let d = tempfile::tempdir().unwrap();
    let (fleet, links, _) = make(
        settings(d.path(), |_| {}),
        vec![l(), a()],
        &[("local", "stale")],
    );
    assert!(!fleet.detail("local").await.unwrap().images.is_empty());
    assert_eq!(links["local"].st().calls.len(), 1);
}

// ── reconnect ──────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn test_reconnect_kicks_the_link_and_returns_the_summary() {
    let d = tempfile::tempdir().unwrap();
    let (fleet, links, _) = make(
        settings(d.path(), |_| {}),
        vec![l(), a()],
        &[("gpu-box-2", "down")],
    );
    let s = fleet.reconnect("gpu-box-2").unwrap();
    assert!(
        links["gpu-box-2"].st().reconnects == 1 && s.id == "gpu-box-2" && s.link.state == "down"
    );
}

#[test]
fn test_reconnecting_a_disabled_host_is_a_conflict() {
    let d = tempfile::tempdir().unwrap();
    let (fleet, links, _) = make(
        settings(d.path(), |_| {}),
        vec![l(), a()],
        &[("local", "disabled")],
    );
    let e = fleet.reconnect("local").unwrap_err();
    assert!(e.code() == "host_disabled" && e.status() == 409);
    assert_eq!(links["local"].st().reconnects, 0);
}

// ── the "fleet" topic ──────────────────────────────────────────────────────────────────────────────────
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_state_changes_are_published_as_host_summaries() {
    limited(async {
        let d = tempfile::tempdir().unwrap();
        let (fleet, links, bus) = make(settings(d.path(), |_| {}), vec![l(), a()], &[]);
        fleet.set_watch_s(0.05);
        let got: Arc<Mutex<Vec<Value>>> = Arc::default();
        let mut sub = bus.subscribe("fleet");
        let g2 = Arc::clone(&got);
        let task = tokio::spawn(async move {
            while let Some(ev) = sub.recv().await {
                g2.lock().unwrap().push(ev);
            }
        });
        fleet.start().await;
        let g = Arc::clone(&got);
        wait_until(
            move || {
                let ids: std::collections::HashSet<String> = g
                    .lock()
                    .unwrap()
                    .iter()
                    .map(|e| e["id"].as_str().unwrap().to_string())
                    .collect();
                ids.len() == 2
            },
            8.0,
            "the first snapshot of both hosts",
        )
        .await;
        let n = got.lock().unwrap().len();
        {
            let mut st = links["gpu-box-2"].st();
            st.state = "down".into();
            st.error = Some("the node exited (3)".into());
        }
        let g = Arc::clone(&got);
        wait_until(
            move || {
                g.lock()
                    .unwrap()
                    .iter()
                    .any(|e| e["id"] == "gpu-box-2" && e["link"]["state"] == "down")
            },
            8.0,
            "the down event",
        )
        .await;
        let down = got
            .lock()
            .unwrap()
            .iter()
            .find(|e| e["link"]["state"] == "down")
            .cloned()
            .unwrap();
        assert!(
            down["link"]["error"] == "the node exited (3)"
                && down["label"] == "GPU box 2"
                && down.get("gpus").is_some()
        );
        assert!(got.lock().unwrap().len() > n);
        let quiet = got.lock().unwrap().len();
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert_eq!(got.lock().unwrap().len(), quiet); // nothing changed, nothing published
        fleet.stop().await;
        task.abort();
    })
    .await;
}

// ── switched off ───────────────────────────────────────────────────────────────────────────────────────
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_with_the_fleet_disabled_nothing_connects_and_the_hosts_file_is_not_read() {
    use std::os::unix::fs::PermissionsExt as _;
    let d = tempfile::tempdir().unwrap();
    let bad = d.path().join("hosts.json");
    std::fs::write(&bad, "{not even json").unwrap();
    std::fs::set_permissions(&bad, std::fs::Permissions::from_mode(0o666)).unwrap(); // would be refused if it were read
    let fleet = FleetService::new(
        settings(d.path(), |s| {
            s.fleet_enabled = false;
            s.hosts_file.clone_from(&bad);
        }),
        Arc::new(EventBus::new()),
    )
    .unwrap();
    fleet.start().await;
    let hosts = fleet.hosts();
    assert!(hosts.len() == 1 && hosts[0].id == "local");
    let s = fleet.summary("local").unwrap();
    assert!(
        s.link.state == "disabled" && s.link.error.unwrap_or_default().contains("FLEET_ENABLED")
    );
    let node = fleet.node("local").unwrap();
    assert_eq!(
        node.call("docker.images", None, Duration::from_secs(30))
            .await
            .unwrap_err()
            .code,
        "node_unavailable"
    );
    let mut stream = node.open("docker.images", None);
    assert!(wsb_harness::items(&mut stream).await.is_empty());
    assert_eq!(stream.result().await.unwrap_err().code, "node_unavailable");
    assert!(node.latest("hello").is_none());
    fleet.stop().await;
}

fn write_hosts(path: &Path, doc: &Value) -> PathBuf {
    use std::os::unix::fs::PermissionsExt as _;
    std::fs::write(path, doc.to_string()).unwrap();
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600)).unwrap();
    path.to_path_buf()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_a_host_disabled_in_the_file_gets_a_disabled_link_and_the_rest_connect() {
    limited(async {
        let d = tempfile::tempdir().unwrap();
        let hosts_file = write_hosts(&d.path().join("hosts.json"), &json!({"hosts": [{"id": "local"}, {"id": "off", "ssh": "ops@10.0.0.9", "enabled": false}]}));
        let cfg = settings(d.path(), |s| { s.hosts_file = hosts_file; s.node_command = Some(node_argv(&json!({}))); });
        let fleet = FleetService::new(cfg, Arc::new(EventBus::new())).unwrap();
        assert_eq!(fleet.node("off").unwrap().link().state, "disabled");
        assert_ne!(fleet.node("local").unwrap().link().state, "disabled");
        fleet.start().await;
        let f = &fleet;
        wait_until(|| f.summary("local").unwrap().link.state == "up", 15.0, "the local node").await;
        let off = fleet.summary("off").unwrap();
        assert!(off.link.state == "disabled" && off.link.error.as_deref() == Some("disabled in hosts.json") && off.remote);
        fleet.stop().await;
    })
    .await;
}

#[test]
fn test_a_bad_hosts_file_stops_the_console_at_construction() {
    let d = tempfile::tempdir().unwrap();
    let hosts_file = write_hosts(&d.path().join("hosts.json"), &json!({"hosts": []}));
    let e = FleetService::new(
        settings(d.path(), |s| s.hosts_file = hosts_file),
        Arc::new(EventBus::new()),
    )
    .unwrap_err();
    assert!(e.to_string().contains("1-32"));
}

#[test]
fn test_no_hosts_file_means_the_local_machine() {
    let d = tempfile::tempdir().unwrap();
    let fleet = FleetService::new(settings(d.path(), |_| {}), Arc::new(EventBus::new())).unwrap();
    assert_eq!(
        fleet
            .hosts()
            .iter()
            .map(|h| h.id.as_str())
            .collect::<Vec<_>>(),
        ["local"]
    );
}

// ── tests that need a different process environment run in a child process ─────────────────────────────
const CHILD_DIR: &str = "WSB_CHILD_DIR";

/// Run the test `name` of this binary in a child process with `env` added; it fails when the child fails.
fn run_child(name: &str, dir: &Path, env: &[(&str, String)]) {
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", name, "--nocapture", "--test-threads=1"])
        .env(CHILD_DIR, dir)
        .envs(env.iter().map(|(k, v)| (*k, v.as_str())))
        .stdin(std::process::Stdio::null())
        .output()
        .unwrap();
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        out.status.success() && text.contains("1 passed"),
        "{name} failed in its child process:\n{text}"
    );
}

/// The work of a child: None when this is not the child process.
fn child_dir() -> Option<PathBuf> {
    std::env::var_os(CHILD_DIR).map(PathBuf::from)
}

fn rt() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap()
}

// ── the default local link: a real subprocess ──────────────────────────────────────────────────────────
#[test]
fn test_the_default_local_link_runs_the_configured_command_with_a_scrubbed_environment() {
    let d = tempfile::tempdir().unwrap();
    run_child(
        "child_default_local_link",
        d.path(),
        &[
            ("HF_TOKEN", "hf_this_must_stay_in_the_console".into()),
            ("VLLM_API_KEY", "also-secret".into()),
        ],
    );
}

#[test]
fn child_default_local_link() {
    let Some(dir) = child_dir() else { return };
    rt().block_on(limited(async move {
        let log = dir.join("local.jsonl");
        let cfg = settings(&dir, |s| {
            s.node_command = Some(node_argv(&json!({"log": log.to_string_lossy()})));
        });
        let fleet = FleetService::with_links(cfg, Arc::new(EventBus::new()), Some(vec![l()]), None)
            .unwrap();
        let events: Arc<Mutex<Vec<Value>>> = Arc::default();
        let mut sub = fleet.bus().subscribe("fleet");
        let e2 = Arc::clone(&events);
        let task = tokio::spawn(async move {
            while let Some(ev) = sub.recv().await {
                e2.lock().unwrap().push(ev);
            }
        });
        fleet.start().await;
        let ev = Arc::clone(&events);
        wait_until(
            move || {
                ev.lock()
                    .unwrap()
                    .iter()
                    .any(|e| e["link"]["state"] == "up")
            },
            15.0,
            "an 'up' event on the fleet topic",
        )
        .await;
        let up = events
            .lock()
            .unwrap()
            .iter()
            .find(|e| e["link"]["state"] == "up")
            .cloned()
            .unwrap();
        assert!(
            up["id"] == "local"
                && up["node"]["version"] == "0.4.0-test"
                && up["gpus"].as_array().unwrap().len() == 2
        );
        let s = fleet.summary("local").unwrap();
        assert!(s.link.state == "up" && s.docker.is_some_and(|d| d.available));
        let det = fleet.detail("local").await.unwrap();
        assert_eq!(
            det.images
                .iter()
                .map(|i| i.ref_.as_str())
                .collect::<Vec<_>>(),
            ["vllm/vllm-openai:v0.1"]
        );
        fleet.detail("local").await.unwrap();
        let reqs = read_log(&log)
            .iter()
            .filter(|e| e["ev"] == "req" && e["op"] == "docker.images")
            .count();
        assert_eq!(reqs, 1); // cached
        let starts: Vec<Value> = read_log(&log)
            .into_iter()
            .filter(|e| e["ev"] == "start")
            .collect();
        assert_eq!(starts.len(), 1);
        let env: Vec<&str> = starts[0]["env"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert!(!env.contains(&"HF_TOKEN") && !env.contains(&"VLLM_API_KEY"));
        assert!(env.iter().all(|k| LOCAL_ENV_KEYS.contains(k)), "{env:?}");
        assert!(fleet.node("local").unwrap().latest("hello").is_some());
        fleet.stop().await;
        task.abort();
        let pid = starts[0]["pid"].as_i64().unwrap();
        assert!(!alive(pid));
    }));
}

// ── the default remote link: a fake `ssh` on PATH ──────────────────────────────────────────────────────
fn key_file(dir: &Path) -> PathBuf {
    let key = dir.join("keys").join("hub-key");
    std::fs::create_dir_all(key.parent().unwrap()).unwrap();
    std::fs::write(&key, "stand-in, not a real key\n").unwrap();
    key
}

fn path_with(bindir: &Path) -> String {
    format!(
        "{}:{}",
        bindir.display(),
        std::env::var("PATH").unwrap_or_default()
    )
}

/// The child's environment for a fake `ssh` in `bindir`: the hub looks for ssh on its tool path
/// (`ENGINE_CONSOLE_TOOL_PATH`), never on PATH (tools.rs), so both carry the fake first.
fn fake_tools(bindir: &Path) -> [(&'static str, String); 2] {
    [
        ("PATH", path_with(bindir)),
        ("ENGINE_CONSOLE_TOOL_PATH", path_with(bindir)),
    ]
}

#[test]
fn test_the_default_remote_link_runs_the_documented_ssh_command() {
    let d = tempfile::tempdir().unwrap();
    install_fake_ssh(
        &d.path().join("bin"),
        &d.path().join("ssh.jsonl"),
        "node",
        &json!({"forced_cmd": true}),
        None,
    );
    run_child(
        "child_remote_documented_ssh_command",
        d.path(),
        &[
            fake_tools(&d.path().join("bin")),
            [
                ("SSH_AUTH_SOCK", "/run/some-agent".into()),
                ("HF_TOKEN", "hf_must_not_reach_ssh".into()),
            ],
        ]
        .concat(),
    );
}

#[test]
fn child_remote_documented_ssh_command() {
    let Some(dir) = child_dir() else { return };
    rt().block_on(limited(async move {
        let cfg = settings(&dir, |s| s.hub_key_file = key_file(&dir));
        let mut h = a();
        h.port = Some(2222);
        let fleet = FleetService::with_links(
            Arc::clone(&cfg),
            Arc::new(EventBus::new()),
            Some(vec![h.clone()]),
            None,
        )
        .unwrap();
        fleet.start().await;
        let f = &fleet;
        wait_until(
            || f.summary("gpu-box-2").unwrap().link.state == "up",
            15.0,
            "the remote node",
        )
        .await;
        let calls = read_log(&dir.join("ssh.jsonl"));
        assert_eq!(calls.len(), 1);
        let mut argv = vec!["ssh".to_string()];
        argv.extend(
            calls[0]["argv"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap().to_string()),
        );
        assert_eq!(
            argv,
            ssh_argv(&h, &cfg.hub_key_file, &cfg.remote_node_command).unwrap()
        );
        let env: Vec<&str> = calls[0]["env"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        // nothing of the console's environment
        assert!(
            env.iter()
                .all(|k| ["HOME", "PATH", "LANG", "USER"].contains(k)),
            "{env:?}"
        );
        let s = fleet.summary("gpu-box-2").unwrap();
        assert!(s.remote && s.link.notes.is_empty());
        assert_eq!(s.node.unwrap().version.as_deref(), Some("0.4.0-test"));
        fleet.stop().await;
    }));
}

#[test]
fn test_a_login_the_key_cannot_make_names_the_fix_and_waits_five_minutes() {
    let d = tempfile::tempdir().unwrap();
    install_fake_ssh(
        &d.path().join("bin"),
        &d.path().join("ssh.jsonl"),
        "denied",
        &json!({}),
        None,
    );
    run_child(
        "child_login_refused",
        d.path(),
        &fake_tools(&d.path().join("bin")),
    );
}

#[test]
fn child_login_refused() {
    let Some(dir) = child_dir() else { return };
    rt().block_on(limited(async move {
        let fleet = FleetService::with_links(
            settings(&dir, |s| s.hub_key_file = key_file(&dir)),
            Arc::new(EventBus::new()),
            Some(vec![a()]),
            None,
        )
        .unwrap();
        fleet.start().await;
        let f = &fleet;
        wait_until(
            || {
                let l = f.summary("gpu-box-2").unwrap().link;
                l.state == "down" && l.retry_in_s.is_some()
            },
            15.0,
            "the refused login",
        )
        .await;
        let link = fleet.summary("gpu-box-2").unwrap().link;
        let err = link.error.unwrap_or_default();
        assert!(
            err.contains("hosts init-key") && err.contains("GPU box 2"),
            "{err}"
        );
        assert!(link.retry_in_s.is_some_and(|r| (295..=300).contains(&r)));
        fleet.stop().await;
    }));
}

#[test]
fn test_the_hub_key_path_never_reaches_the_link_error() {
    let d = tempfile::tempdir().unwrap();
    let key = key_file(d.path());
    install_fake_ssh(
        &d.path().join("bin"),
        &d.path().join("ssh.jsonl"),
        "badkey",
        &json!({}),
        Some(&key),
    );
    run_child(
        "child_hub_key_masked",
        d.path(),
        &fake_tools(&d.path().join("bin")),
    );
}

#[test]
fn child_hub_key_masked() {
    let Some(dir) = child_dir() else { return };
    rt().block_on(limited(async move {
        let key = dir.join("keys").join("hub-key");
        let fleet = FleetService::with_links(
            settings(&dir, |s| s.hub_key_file.clone_from(&key)),
            Arc::new(EventBus::new()),
            Some(vec![a()]),
            None,
        )
        .unwrap();
        fleet.start().await;
        let f = &fleet;
        wait_until(
            || {
                let l = f.summary("gpu-box-2").unwrap().link;
                l.state == "down" && l.retry_in_s.is_some()
            },
            15.0,
            "the failed login",
        )
        .await;
        let err = fleet
            .summary("gpu-box-2")
            .unwrap()
            .link
            .error
            .unwrap_or_default();
        assert_eq!(
            err,
            "ssh to ops@10.0.0.5 failed: Load key \"<hub key>\": error in libcrypto"
        );
        assert!(!err.contains(&key.display().to_string()));
        fleet.stop().await;
    }));
}

/// ssh would otherwise try the user's own default keys when the key named with -i is missing.
#[test]
fn test_without_a_hub_key_ssh_is_never_started() {
    let d = tempfile::tempdir().unwrap();
    install_fake_ssh(
        &d.path().join("bin"),
        &d.path().join("ssh.jsonl"),
        "node",
        &json!({}),
        None,
    );
    run_child(
        "child_no_hub_key",
        d.path(),
        &fake_tools(&d.path().join("bin")),
    );
}

#[test]
fn child_no_hub_key() {
    let Some(dir) = child_dir() else { return };
    rt().block_on(limited(async move {
        let key = dir.join("keys").join("does-not-exist");
        let fleet = FleetService::with_links(
            settings(&dir, |s| s.hub_key_file.clone_from(&key)),
            Arc::new(EventBus::new()),
            Some(vec![a()]),
            None,
        )
        .unwrap();
        fleet.start().await;
        let f = &fleet;
        wait_until(
            || f.summary("gpu-box-2").unwrap().link.error.is_some(),
            15.0,
            "the missing-key error",
        )
        .await;
        let link = fleet.summary("gpu-box-2").unwrap().link;
        let err = link.error.unwrap_or_default();
        assert!(
            ["down", "connecting"].contains(&link.state.as_str())
                && err.contains("no ssh key yet")
                && err.contains("hosts init-key")
        );
        tokio::time::sleep(Duration::from_millis(300)).await;
        let log = dir.join("ssh.jsonl");
        assert!(!log.exists()); // ssh was never run
        std::fs::create_dir_all(key.parent().unwrap()).unwrap(); // create the key where the settings point...
        std::fs::write(&key, "stand-in, not a real key\n").unwrap();
        fleet.reconnect("gpu-box-2").unwrap(); // ...and the next attempt uses it
        wait_until(
            move || log.exists(),
            10.0,
            "ssh to be started now that the key exists",
        )
        .await;
        fleet.stop().await;
    }));
}

#[test]
fn test_fleet_links_module_is_used() {
    let _ = FakeLink::new(host("x", "x", None), "up");
    let _: HashMap<String, Value> = HashMap::new();
}
