//! Port of backend/tests/core/test_stream_caps.py (review finding L4): the core rows. The stream caps count per API
//! key as well as in total, a slot is given back however a stream ends, and the /hf routes are admin-only. The
//! streamed-chat rows are in tests/test_wsc_stream_caps.rs. Python's long-lived SSE vehicle is /downloads/stream;
//! here it is /metrics/stream without `once`, which stays open the same way (until the client leaves or goes idle).
#![allow(clippy::panic, clippy::expect_used, clippy::unwrap_used)] // test helpers outside #[test] fns

#[path = "common/kit.rs"]
mod kit;

use std::time::Duration;

use ec_hub::services::common::SseLimiter;
use kit::{Opened, TestEnv};
use serde_json::{Value, json};

const P: &str = "/api/v1";
const SSE: &str = "/api/v1/metrics/stream";

fn capped() -> TestEnv {
    // set the way an operator sets them (the environment), as Python's `capped` fixture does
    let env: Vec<(String, String)> = [
        ("MAX_SSE_CONNECTIONS", "3"),
        ("MAX_SSE_PER_KEY", "2"),
        ("MAX_CHAT_STREAMS", "3"),
        ("MAX_CHAT_STREAMS_PER_KEY", "2"),
    ]
    .iter()
    .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
    .collect();
    let parsed = ec_hub::config::Settings::load(&env, None).unwrap();
    TestEnv::with(|c| {
        c.max_sse_connections = parsed.max_sse_connections;
        c.max_sse_per_key = parsed.max_sse_per_key;
        c.max_chat_streams = parsed.max_chat_streams;
        c.max_chat_streams_per_key = parsed.max_chat_streams_per_key;
    })
}

fn key(env: &mut TestEnv, name: &str) -> String {
    env.container()
        .settings
        .create_key(name, "viewer")
        .unwrap()
        .secret
        .unwrap()
}

async fn sse(env: &mut TestEnv, key: Option<&str>) -> Opened {
    let mut r = env.get(SSE);
    if let Some(k) = key {
        r = r.bearer(k);
    }
    let mut o = r.open().await;
    if o.status != 200 {
        o.rest(Duration::from_millis(500)).await;
    }
    o
}

// ---- the limiter itself ------------------------------------------------------------------------------------------
#[test]
fn test_limiter_counts_per_key_and_in_total_and_forgets_empty_keys() {
    let lim = SseLimiter::new(3, Some(2));
    assert!(lim.acquire("a") && lim.acquire("a") && !lim.acquire("a")); // a's share is full
    assert!(lim.key_full("a") && !lim.key_full("b"));
    assert!(lim.acquire("b") && !lim.acquire("c")); // the total is full
    lim.release("a");
    assert!(lim.acquire("c") && lim.active() == 3);
    for k in ["a", "b", "c"] {
        lim.release(k);
    }
    assert_eq!((lim.active(), lim.tracked_keys()), (0, 0)); // no empty buckets kept per key id
    assert_eq!(SseLimiter::new(8, None).per_key, 8); // no per-key share given: only the total
}

// ---- SSE: one key cannot hold every slot ---------------------------------------------------------------------------
#[tokio::test]
async fn test_one_key_cannot_hold_every_sse_slot() {
    let mut env = capped();
    let a = key(&mut env, "a");
    let b = key(&mut env, "b");
    let mut held = vec![sse(&mut env, Some(&a)).await, sse(&mut env, Some(&a)).await];
    assert_eq!(
        held.iter().map(|c| c.status).collect::<Vec<_>>(),
        [200, 200]
    );
    let third = sse(&mut env, Some(&a)).await;
    assert_eq!(third.status, 429);
    let p = third.json();
    // the same code the UI already treats as "do not retry", plus which cap is full
    assert_eq!(
        (p["code"].clone(), p["scope"].clone(), p["limit"].clone()),
        (json!("too_many_streams"), json!("key"), json!(2))
    );
    assert!(p["detail"].as_str().unwrap().contains("this API key"));
    let mut other = sse(&mut env, Some(&b)).await;
    assert_eq!(other.status, 200); // another key still gets its streams
    for c in &mut held {
        c.leave();
    }
    other.leave();
    assert_eq!(env.container().sse.active(), 0);
    let mut again = sse(&mut env, Some(&a)).await;
    assert_eq!(again.status, 200); // a's slots came back
    again.leave();
}

#[tokio::test]
async fn test_the_total_sse_cap_still_holds_across_keys() {
    let mut env = capped();
    let mut held = Vec::new();
    for i in 0..3 {
        let k = key(&mut env, &format!("k{i}"));
        held.push(sse(&mut env, Some(&k)).await);
    }
    assert_eq!(
        held.iter().map(|c| c.status).collect::<Vec<_>>(),
        [200, 200, 200]
    );
    let late_key = key(&mut env, "late");
    let late = sse(&mut env, Some(&late_key)).await;
    assert_eq!(late.status, 429);
    assert_eq!(
        (late.json()["code"].clone(), late.json()["scope"].clone()),
        (json!("too_many_streams"), json!("total"))
    );
    for c in &mut held {
        c.leave();
    }
}

#[tokio::test]
async fn test_loopback_requests_without_a_key_share_one_bucket() {
    let mut env = capped();
    assert!(env.cfg.trust_loopback); // the core test env trusts loopback: no key needed
    let mut first = vec![sse(&mut env, None).await, sse(&mut env, None).await];
    assert_eq!(
        first.iter().map(|c| c.status).collect::<Vec<_>>(),
        [200, 200]
    );
    let third = sse(&mut env, None).await;
    assert_eq!(
        (third.status, third.json()["scope"].clone()),
        (429, json!("key"))
    );
    let k = key(&mut env, "k");
    let mut keyed = sse(&mut env, Some(&k)).await;
    assert_eq!(keyed.status, 200); // a key has its own share
    for c in &mut first {
        c.leave();
    }
    keyed.leave();
}

/// Python's two ASGI spec variants (2.3: disconnect through receive(); 2.4: a failing send) are one case here: the
/// client going away drops the response body, which holds the slot.
#[tokio::test]
async fn test_an_sse_slot_comes_back_when_the_client_leaves_mid_stream() {
    let mut env = TestEnv::new();
    let k = key(&mut env, "k");
    let c = env.container();
    let mut s = sse(&mut env, Some(&k)).await;
    assert_eq!((s.status, c.sse.active()), (200, 1));
    c.bus.publish(
        "metrics",
        json!({"instance_id": "i1", "t": 1.0, "values": {}}),
    );
    assert!(s.chunk(Duration::from_secs(2)).await.is_some());
    s.leave();
    assert_eq!(c.sse.active(), 0);
}

#[tokio::test]
async fn test_an_sse_slot_comes_back_when_the_client_leaves_before_the_first_byte() {
    let mut env = TestEnv::new();
    let k = key(&mut env, "k");
    let c = env.container();
    let mut s = env.get(SSE).bearer(&k).open().await;
    s.leave(); // gone before reading anything
    assert_eq!(c.sse.active(), 0);
}

// ---- /hf/* spends the operator's token: admin only --------------------------------------------------------------------
#[tokio::test]
async fn test_the_hf_routes_are_admin_only() {
    let mut env = TestEnv::new();
    env.hub.add(
        "Qwen/Qwen3-32B",
        &[("config.json", b"{}"), ("model.safetensors", &[b'x'; 100])],
        None,
        &[("BF16", 32_762_000_000)],
    );
    let viewer = key(&mut env, "v");
    let admin = env
        .container()
        .settings
        .create_key("adm", "admin")
        .unwrap()
        .secret
        .unwrap();
    let before = env.hub.calls.lock().unwrap().clone();
    for path in [
        format!("{P}/hf/search?q=qwen"),
        format!("{P}/hf/models/Qwen/Qwen3-32B?q=qwen"),
    ] {
        let r = env
            .get(&path)
            .client("192.0.2.7")
            .bearer(&viewer)
            .send()
            .await; // not loopback: only keys count
        assert_eq!((r.status, r.code().as_str()), (403, "forbidden"), "{path}");
    }
    assert_eq!(*env.hub.calls.lock().unwrap(), before); // nothing reached Hugging Face for the viewer
    // an admin passes the role check (the answer itself is the HF service's)
    let ok = env
        .get(&format!("{P}/hf/search?q=qwen"))
        .client("192.0.2.7")
        .bearer(&admin)
        .send()
        .await;
    assert!(ok.status != 403 && ok.status != 401);
    let ok = env
        .get(&format!("{P}/hf/models/Qwen/Qwen3-32B"))
        .client("192.0.2.7")
        .bearer(&admin)
        .send()
        .await;
    assert!(ok.status != 403 && ok.status != 401);
    // every /hf route, enumerated from the OpenAPI document so a new one cannot skip the check
    let doc: Value = env.get(&format!("{P}/openapi.json")).send().await.json();
    let hf: Vec<String> = doc["paths"]
        .as_object()
        .unwrap()
        .keys()
        .filter(|p| p.starts_with(&format!("{P}/hf/")))
        .cloned()
        .collect();
    assert_eq!(hf.len(), 2);
    for path in hf {
        let concrete = path.replace("{repo_id}", "a/b");
        assert_eq!(
            env.get(&concrete)
                .client("192.0.2.7")
                .bearer(&viewer)
                .send()
                .await
                .status,
            403,
            "{path}"
        );
    }
}
