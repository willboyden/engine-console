//! Port of backend/tests/fleet/test_router.py: the hosts routes and the live /fleet/stream, through the real app (its
//! security middleware, error handlers and role checks) over a FleetService of FakeLinks.
//!
//! Python puts a principal on each request from an `X-Test-Role` header; here the roles come the way they do in
//! production: admin = the trusted loopback caller, viewer/public = a bearer key with that role.

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
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body as AxBody;
use ec_hub::api::routes::hosts::{STREAM_POLL_S, STREAM_REFRESH_S, fleet_events};
use ec_hub::app::{AppState, app_state, handle_request};
use ec_hub::container::{Overrides, build_container};
use ec_hub::domain::fleet::{Fleet, HostConfig};
use ec_hub::services::common::EventBus;
use ec_hub::services::fleet::FleetService;
use fleet_links::{FakeLink, make_with};
use futures::StreamExt as _;
use futures::stream::BoxStream;
use http_body_util::BodyExt;
use serde_json::{Value, json};
use wsb_harness::{disks, host, make_settings, mount, obj, root, sample, wait_until};

fake_node_entry!();

const P: &str = "/api/v1";

fn a() -> HostConfig {
    host("gpu-box-2", "GPU box 2", Some("ops@10.0.0.5"))
}

fn l() -> HostConfig {
    host("local", "Workstation", None)
}

struct Rig {
    _tmp: tempfile::TempDir,
    st: Arc<AppState>,
    fleet: Arc<FleetService>,
    links: HashMap<String, Arc<FakeLink>>,
    bus: Arc<EventBus>,
    keys: HashMap<&'static str, String>,
}

fn rig(limit: i64) -> Rig {
    let tmp = tempfile::tempdir().unwrap();
    let mut cfg = make_settings(tmp.path());
    cfg.allowed_hosts = vec!["testserver".into()];
    cfg.trust_loopback = true;
    cfg.max_sse_connections = limit;
    cfg.sse_idle_timeout_s = 5.0;
    let cfg = Arc::new(cfg);
    let (fleet, links, bus) = make_with(Arc::clone(&cfg), vec![l(), a()], &[], |link| {
        link.set(
            "hello",
            wsb_harness::b_hello(
                &[],
                vec![root("hf", "/data/hf", json!({}))],
                "gpu-box-2",
                json!({}),
            ),
        );
        link.set("sample", sample(json!({})));
        link.set(
            "disks",
            disks(vec![mount("/", json!({})), mount("/data", json!({}))]),
        );
        link.result(
            "docker.images",
            Ok(obj(
                json!({"images": [{"ref": "vllm/vllm-openai:v0.1", "size": 5}]}),
            )),
        );
    });
    let fleet = Arc::new(fleet);
    let c = build_container(
        cfg,
        Overrides {
            fleet: Some(Arc::clone(&fleet) as Arc<dyn Fleet>),
            frontend: Some(None),
            ..Overrides::default()
        },
    )
    .unwrap();
    let mut keys = HashMap::new();
    for role in ["viewer", "public"] {
        keys.insert(
            role,
            c.settings.create_key(role, role).unwrap().secret.unwrap(),
        );
    }
    Rig {
        _tmp: tmp,
        st: app_state(c),
        fleet,
        links,
        bus,
        keys,
    }
}

struct Resp {
    status: u16,
    headers: HashMap<String, String>,
    body: Vec<u8>,
}

impl Resp {
    fn json(&self) -> Value {
        serde_json::from_slice(&self.body).unwrap_or(Value::Null)
    }
    fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }
    /// The problem document (and its content type and `status` member checked).
    fn problem(&self) -> Value {
        assert!(
            self.headers["content-type"].starts_with("application/problem+json"),
            "{}",
            self.text()
        );
        let body = self.json();
        assert_eq!(body["status"], json!(self.status));
        body
    }
}

impl Rig {
    async fn req(&self, method: &str, path: &str, role: &str) -> Resp {
        let mut b = http::Request::builder()
            .method(method)
            .uri(path)
            .header("host", "testserver")
            .header("x-engine-console", "1");
        if role != "admin" {
            b = b.header("authorization", format!("Bearer {}", self.keys[role]));
        }
        let r = handle_request(&self.st, "127.0.0.1", b.body(AxBody::empty()).unwrap()).await;
        let status = r.status().as_u16();
        let headers = r
            .headers()
            .iter()
            .map(|(k, v)| {
                (
                    k.as_str().to_string(),
                    String::from_utf8_lossy(v.as_bytes()).into_owned(),
                )
            })
            .collect();
        let body = r
            .into_body()
            .collect()
            .await
            .map(|c| c.to_bytes().to_vec())
            .unwrap_or_default();
        Resp {
            status,
            headers,
            body,
        }
    }

    async fn get(&self, path: &str) -> Resp {
        self.req("GET", path, "admin").await
    }
}

// ── GET /hosts, /hosts/{id} ────────────────────────────────────────────────────────────────────────────
#[tokio::test]
async fn test_list_hosts_is_a_page_of_summaries() {
    let r = rig(4);
    let resp = r.get(&format!("{P}/hosts")).await;
    assert_eq!(resp.status, 200);
    let body = resp.json();
    assert!(body["next_cursor"].is_null());
    assert_eq!(
        body["items"]
            .as_array()
            .unwrap()
            .iter()
            .map(|h| h["id"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["local", "gpu-box-2"]
    );
    let first = &body["items"][0];
    assert!(
        first["label"] == "Workstation"
            && first["remote"] == false
            && first["link"]["state"] == "up"
    );
    assert!(
        first["gpus"][0]["total_gib"] == 48.0
            && first["memory"]["total_gib"] == 256.0
            && first["counts"]["models"] == 0
    );
    assert_eq!(body["items"][1]["remote"], true);
    assert!(first.get("disks").is_none()); // the list carries summaries, not details
}

#[tokio::test]
async fn test_host_detail() {
    let r = rig(4);
    let resp = r.get(&format!("{P}/hosts/gpu-box-2")).await;
    assert_eq!(resp.status, 200);
    let d = resp.json();
    assert_eq!(d["id"], "gpu-box-2");
    assert_eq!(
        d["disks"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["mount"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["/", "/data"]
    );
    assert!(
        d["roots"][0]["id"] == "hf"
            && d["roots"][0]["mount"]["mount"] == "/data"
            && d["roots"][0]["visible_on"] == json!(["gpu-box-2"])
    );
    assert!(
        d["images"][0]["ref"] == "vllm/vllm-openai:v0.1"
            && d["addresses"] == json!(["10.0.0.5/24 (eth0)"])
            && d["scripts"] == json!([])
    );
}

#[tokio::test]
async fn test_an_unknown_host_is_a_404_problem() {
    let r = rig(4);
    let resp = r.get(&format!("{P}/hosts/nope")).await;
    assert_eq!(resp.status, 404);
    assert_eq!(resp.problem()["code"], "unknown_host");
    assert_eq!(
        r.req("POST", &format!("{P}/hosts/nope/reconnect"), "admin")
            .await
            .problem()["code"],
        "unknown_host"
    );
}

#[tokio::test]
async fn test_the_link_error_and_notes_are_visible_to_the_page() {
    let r = rig(4);
    {
        let mut st = r.links["gpu-box-2"].st();
        st.state = "down".into();
        st.error = Some("ssh refused the login to ops@10.0.0.5".into());
    }
    let h = r.get(&format!("{P}/hosts/gpu-box-2")).await.json();
    assert!(
        h["link"]["state"] == "down"
            && h["link"]["error"] == "ssh refused the login to ops@10.0.0.5"
    );
    assert!(h["gpus"][0]["used_gib"].is_null() && h["memory"].is_null()); // nothing live from a host that is down
}

// ── POST /hosts/{id}/reconnect ─────────────────────────────────────────────────────────────────────────
#[tokio::test]
async fn test_reconnect_is_admin_only_and_returns_the_summary() {
    let r = rig(4);
    let resp = r
        .req("POST", &format!("{P}/hosts/gpu-box-2/reconnect"), "admin")
        .await;
    assert!(
        resp.status == 200
            && resp.json()["id"] == "gpu-box-2"
            && r.links["gpu-box-2"].st().reconnects == 1
    );
    for role in ["viewer", "public"] {
        let denied = r
            .req("POST", &format!("{P}/hosts/gpu-box-2/reconnect"), role)
            .await;
        assert_eq!(denied.status, 403);
        assert_eq!(denied.problem()["code"], "forbidden");
    }
    assert_eq!(r.links["gpu-box-2"].st().reconnects, 1); // the refused requests did nothing
}

/// The real app also enforces the router-level role check; the route must not depend on that (core's
/// `reconnect_host` calls `require_admin` itself): every role but admin is refused, admin passes.
#[tokio::test]
async fn test_the_route_checks_for_admin_itself() {
    let r = rig(4);
    assert_eq!(
        r.req("POST", &format!("{P}/hosts/gpu-box-2/reconnect"), "viewer")
            .await
            .status,
        403
    );
    assert_eq!(
        r.req("POST", &format!("{P}/hosts/gpu-box-2/reconnect"), "public")
            .await
            .status,
        403
    );
    assert_eq!(r.links["gpu-box-2"].st().reconnects, 0);
    assert_eq!(
        r.req("POST", &format!("{P}/hosts/gpu-box-2/reconnect"), "admin")
            .await
            .status,
        200
    );
}

#[tokio::test]
async fn test_reconnecting_a_disabled_host_is_a_409() {
    let r = rig(4);
    r.links["local"].st().state = "disabled".into();
    let resp = r
        .req("POST", &format!("{P}/hosts/local/reconnect"), "admin")
        .await;
    assert_eq!(resp.status, 409);
    assert_eq!(resp.problem()["code"], "host_disabled");
}

#[tokio::test]
async fn test_viewers_may_read_hosts() {
    let r = rig(4);
    assert_eq!(
        r.req("GET", &format!("{P}/hosts"), "viewer").await.status,
        200
    );
    assert_eq!(
        r.req("GET", &format!("{P}/hosts/local"), "viewer")
            .await
            .status,
        200
    );
}

// ── GET /fleet/stream ──────────────────────────────────────────────────────────────────────────────────
fn parse_sse(text: &str) -> Vec<(String, Value)> {
    text.trim()
        .split("\n\n")
        .map(|block| {
            let lines: Vec<&str> = block.lines().collect();
            assert!(
                lines.len() == 2
                    && lines[0].starts_with("event: ")
                    && lines[1].starts_with("data: "),
                "{block}"
            );
            (
                lines[0][7..].to_string(),
                serde_json::from_str(&lines[1][6..]).unwrap(),
            )
        })
        .collect()
}

#[tokio::test]
async fn test_stream_once_sends_every_host_and_closes() {
    let r = rig(4);
    let resp = r.get(&format!("{P}/fleet/stream?once=true")).await;
    assert!(resp.status == 200 && resp.headers["content-type"].starts_with("text/event-stream"));
    // the route answers `no-cache`; the real app's security middleware (Python's too) replaces it with `no-store` on
    // every /api/ answer, which is what a browser gets (the Python test mounts the router without that middleware)
    assert_eq!(resp.headers["cache-control"], "no-store");
    let events = parse_sse(&resp.text());
    assert_eq!(
        events
            .iter()
            .map(|(e, d)| (e.as_str(), d["id"].as_str().unwrap()))
            .collect::<Vec<_>>(),
        [("host", "local"), ("host", "gpu-box-2")]
    );
    assert!(
        events[0].1["link"]["state"] == "up" && !events[0].1["gpus"].as_array().unwrap().is_empty()
    );
}

#[tokio::test]
async fn test_the_stream_counts_against_the_connection_cap() {
    let capped = rig(0);
    let resp = capped.get(&format!("{P}/fleet/stream?once=true")).await;
    assert_eq!(resp.status, 429);
    assert_eq!(resp.problem()["code"], "too_many_streams");
}

#[tokio::test]
async fn test_the_slot_is_given_back_when_the_stream_ends() {
    let one = rig(1);
    for _ in 0..3 {
        assert_eq!(
            one.get(&format!("{P}/fleet/stream?once=true")).await.status,
            200
        );
    }
    assert_eq!(one.st.c.sse.active(), 0);
}

// ── the event generator itself ─────────────────────────────────────────────────────────────────────────
async fn next_chunk(stream: &mut BoxStream<'static, String>, timeout_s: f64) -> (String, Value) {
    let chunk = tokio::time::timeout(Duration::from_secs_f64(timeout_s), stream.next())
        .await
        .expect("a chunk in time")
        .expect("not the end");
    let mut events = parse_sse(&chunk);
    assert_eq!(events.len(), 1);
    events.remove(0)
}

fn events_of(r: &Rig, once: bool, poll: f64, refresh: f64) -> BoxStream<'static, String> {
    fleet_events(
        Arc::clone(&r.fleet) as Arc<dyn Fleet>,
        &r.bus,
        once,
        Duration::from_secs_f64(poll),
        Duration::from_secs_f64(refresh),
    )
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_events_start_with_a_snapshot_then_report_changes() {
    let r = rig(4);
    let mut g = events_of(&r, false, 0.05, 60.0);
    let first = [next_chunk(&mut g, 5.0).await, next_chunk(&mut g, 5.0).await];
    assert_eq!(
        first
            .iter()
            .map(|(_, d)| d["id"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["local", "gpu-box-2"]
    );
    // nothing changed: nothing is sent
    assert!(
        tokio::time::timeout(Duration::from_millis(300), g.next())
            .await
            .is_err()
    );
    {
        let mut st = r.links["gpu-box-2"].st();
        st.state = "down".into();
        st.error = Some("the node exited (3)".into());
    }
    let (event, d) = next_chunk(&mut g, 5.0).await;
    assert!(
        event == "host"
            && d["id"] == "gpu-box-2"
            && d["link"]["state"] == "down"
            && d["link"]["error"] == "the node exited (3)"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_a_change_to_live_numbers_is_sent_without_any_link_event() {
    let r = rig(4);
    let mut g = events_of(&r, false, 0.05, 60.0);
    next_chunk(&mut g, 5.0).await;
    next_chunk(&mut g, 5.0).await;
    r.links["local"].set(
        "sample",
        sample(json!({"gpus": {"GPU-A": {"util_pct": 99.0, "mem_used_bytes": 1}, "GPU-B": {}}})),
    );
    let (_, d) = next_chunk(&mut g, 5.0).await;
    assert!(d["id"] == "local" && d["gpus"][0]["util_pct"] == 99.0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_every_host_is_resent_at_least_every_refresh_interval() {
    let r = rig(4);
    let mut g = events_of(&r, false, 0.05, 0.2);
    let mut ids = Vec::new();
    for _ in 0..6 {
        ids.push(
            next_chunk(&mut g, 5.0).await.1["id"]
                .as_str()
                .unwrap()
                .to_string(),
        );
    }
    assert_eq!(ids[..2], ["local", "gpu-box-2"]);
    let mut mid = ids[2..4].to_vec();
    mid.sort();
    assert_eq!(mid, ["gpu-box-2", "local"]); // unchanged hosts, resent
    assert_eq!(ids.len(), 6);
}

#[test]
fn test_the_default_refresh_is_five_seconds() {
    assert_eq!(STREAM_REFRESH_S, 5.0);
    assert!(STREAM_POLL_S <= 1.0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_a_bus_event_wakes_the_stream_before_the_next_poll() {
    let r = rig(4);
    let mut g = events_of(&r, false, 30.0, 300.0);
    next_chunk(&mut g, 5.0).await;
    next_chunk(&mut g, 5.0).await;
    let waiting = tokio::spawn(async move { (next_chunk(&mut g, 5.0).await, g) });
    tokio::time::sleep(Duration::from_millis(100)).await; // the generator is now parked on the bus wake-up
    r.links["gpu-box-2"].st().state = "stale".into();
    r.bus.publish("fleet", json!({"id": "gpu-box-2"})); // what a link does on a state change
    let ((_, d), _g) = tokio::time::timeout(Duration::from_secs(2), waiting)
        .await
        .unwrap()
        .unwrap();
    assert!(d["id"] == "gpu-box-2" && d["link"]["state"] == "stale");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_closing_the_stream_unsubscribes_from_the_bus() {
    let r = rig(4);
    let mut g = events_of(&r, false, 0.05, 60.0);
    next_chunk(&mut g, 5.0).await;
    next_chunk(&mut g, 5.0).await;
    tokio::time::sleep(Duration::from_millis(150)).await; // parked in its wait: the subscription exists
    assert_eq!(r.bus.subscribers("fleet"), 1);
    drop(g);
    let bus = Arc::clone(&r.bus);
    wait_until(
        move || bus.subscribers("fleet") == 0,
        8.0,
        "the subscription to go away",
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_cancelling_the_reader_cleans_up_too() {
    let r = rig(4);
    let mut g = events_of(&r, false, 0.05, 60.0);
    let task = tokio::spawn(async move { while g.next().await.is_some() {} });
    let bus = Arc::clone(&r.bus);
    wait_until(
        move || bus.subscribers("fleet") > 0,
        8.0,
        "the subscription",
    )
    .await;
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    let bus = Arc::clone(&r.bus);
    wait_until(
        move || bus.subscribers("fleet") == 0,
        8.0,
        "the subscription to go away",
    )
    .await;
}

#[tokio::test]
async fn test_once_never_subscribes() {
    let r = rig(4);
    let chunks: Vec<String> = events_of(&r, true, 1.0, 5.0).collect().await;
    assert_eq!(chunks.len(), 2);
    assert_eq!(r.bus.subscribers("fleet"), 0);
}
