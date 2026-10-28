//! Port of backend/tests/core/test_network.py: engine isolation (internal network + per-instance gateway) and the
//! console's egress failing closed.
//!
//! Isolation is a protocol between the hub's lifecycle and each host's node: these tests drive it through the fake
//! fleet. The egress tests run the real HfHttpClient end to end against a throwaway CA, `openssl s_server` and an
//! in-test CONNECT proxy (Python unit-tests the client's private helpers instead; the behaviour checked is the same).
//! The settings, health and allowlist rows run now.

#![allow(
    clippy::expect_used,
    clippy::panic,
    clippy::unwrap_used,
    clippy::too_many_lines,
    clippy::type_complexity,
    clippy::needless_pass_by_value
)]

#[path = "common/kit.rs"]
mod kit;

// the fake fleet is the kit's submodule (PORTING.md §A4): one copy of its types per test binary
use kit::fakefleet;

#[path = "common/wsb_egress.rs"]
mod wsb_egress;

use std::path::Path;
use std::sync::Arc;

use axum::body::Body as AxBody;
use ec_hub::app::{app_state, handle_request};
use ec_hub::config::{GATEWAY_IMAGE, Settings};
use ec_hub::container::{Overrides, build_container};
use ec_hub::domain::errors::AppError;
use ec_hub::domain::models::InstanceCreate;
use ec_hub::domain::ports::HubClient;
use ec_hub::domain::refs::ModelRef;
use ec_hub::services::common::fnmatchcase;
use ec_hub::services::hf_http::{EgressConfig, HfHttpClient, host_allowed};
use ec_hub::services::settings::public_proxy;
use fakefleet::{CopyOpts, FleetEnv};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use wsb_egress::{BLOCK_PAGE, ConnectProxy, Counter, TlsServer, has_openssl, make_certs};

const P: &str = "/api/v1";

/// A throwaway self-signed certificate (no private key kept): only used so an SSL context can be built.
const TEST_CA: &str = "-----BEGIN CERTIFICATE-----
MIIDBTCCAe2gAwIBAgIUCtZZanOajHEtt8R6GS0GjwC+x+EwDQYJKoZIhvcNAQEL
BQAwEjEQMA4GA1UEAwwHdGVzdC1jYTAeFw0yNjA5MjQwMzU2MjRaFw0zNjA5MjEw
MzU2MjRaMBIxEDAOBgNVBAMMB3Rlc3QtY2EwggEiMA0GCSqGSIb3DQEBAQUAA4IB
DwAwggEKAoIBAQDmD6fL5c6Fa8/gwfZ8ICOeTApiS5P3dTnCV5ShZCFjAeMslalB
FBHUb6F5GvTfDtKUW1JYHh+wJSNdvFJr1N1UKJcYhYkRYpEcvJ+WYVF49EyeTe43
PXRQwtj3jP52cEJDvQ6LqLQUIIPyk55xmqJlrRPjFYshYl7FoN1VMdV2lAmkIorN
ESLRiudH7aw1KqUFCKVW8B6GuMnDrUQvZa9gqbuKKqn3EhVn1AiqaWgjOjDCL/NL
bqZUZ0Li7DiUUvW4/RoHXMzNoOZQlBCG+lqJkR2rOS0An1dYaV4GGbPFolpsZUdw
KNMyA1flIqYNnZXpWEuw+CzCdojyd/ouRW5/AgMBAAGjUzBRMB0GA1UdDgQWBBQF
NzdJOmBeCM6T+H+IGRckrKvhqjAfBgNVHSMEGDAWgBQFNzdJOmBeCM6T+H+IGRck
rKvhqjAPBgNVHRMBAf8EBTADAQH/MA0GCSqGSIb3DQEBCwUAA4IBAQB/OOWMxOGA
50BkyxkR93rEi4dFDoRDDqPiX93ywBkPBOjbCVnThTp10qv28iaCufdd2saob+f4
XfIQyMFTNYv6rO07pAlG4voz0zbDbMbhOGaUh7+lv2T5cgTndKcbwF5yzCi1TKGV
s9yNPfTD167nUNz8jlgpmoRfVrZ18WC7ibCzniL3etSSV7e6Y4Fnk6t1+x+1+65d
xSOlZHyI1qHImdkLD5GeWeEHEq+lKU7phP1YxAYzhOvQgLgp12K/M9VnAQYWYx2H
6PEuWxrdG9Igd9QX1Qo9DgsrCADr2nzPyBuEQx/vQch6O3MGS3CA2AtkeS1lzTgC
nL32Ohbc45Ve
-----END CERTIFICATE-----
";

fn create_engine_body(extra: Value) -> Value {
    let mut b =
        json!({"engine": "fake", "repo_id": "Qwen/Qwen3-32B", "params": {"max_model_len": 4096}});
    for (k, v) in extra.as_object().cloned().unwrap_or_default() {
        b[k] = v;
    }
    b
}

async fn create_engine(env: &mut FleetEnv) -> Value {
    env.seed_local_model("Qwen/Qwen3-32B", None, 65_530_000_000, "local");
    let r = env
        .post(&format!("{P}/instances"))
        .json(&create_engine_body(json!({})))
        .send()
        .await;
    assert_eq!(r.status, 201, "{}", r.text());
    r.json()
}

async fn tick(env: &mut FleetEnv) {
    env.container().lifecycle.tick().await.unwrap();
}

// ── engine on the internal net, gateway on loopback ────────────────────────────────────────────────────
#[tokio::test]
async fn test_engine_is_internal_only_and_the_gateway_is_the_only_published_port() {
    let mut env = FleetEnv::new();
    let inst = create_engine(&mut env).await;
    let ops: Vec<String> = env.node.calls().into_iter().map(|(o, _)| o).collect();
    let pos = |op: &str| ops.iter().position(|o| o == op).unwrap();
    assert!(
        pos("net.ensure_internal") < pos("docker.run") && pos("docker.run") < pos("gateway.start")
    );
    let spec = env.node.st().run_specs[0].clone();
    let gw = env.node.ops("gateway.start")[0].clone();
    assert_eq!(spec["network_alias"], inst["container_name"]);
    for k in ["host_port", "ports", "publish", "network", "network_mode"] {
        assert!(spec.get(k).is_none(), "{k}"); // the engine has no way to publish a port
    }
    assert_eq!(
        Value::Object(gw.clone()),
        json!({"name": format!("{}-gw", inst["container_name"].as_str().unwrap()), "instance": inst["id"],
               "host_port": inst["port"], "engine_host": inst["container_name"], "engine_port": 8000})
    );
    let st = env.node.st();
    assert_eq!(
        st.containers[gw["name"].as_str().unwrap()]["ports"],
        json!([{"host_ip": "127.0.0.1", "host_port": inst["port"], "container_port": 8080}])
    );
    assert_eq!(
        st.containers[inst["container_name"].as_str().unwrap()]["ports"],
        json!([])
    ); // loopback only, and only the gateway's
}

#[tokio::test]
async fn test_the_gateway_forwards_to_the_engines_container_name_and_the_adapters_port() {
    let mut env = FleetEnv::new();
    let inst = create_engine(&mut env).await;
    let gw = env.node.ops("gateway.start")[0].clone();
    assert!(
        gw["engine_host"] == inst["container_name"]
            && gw["engine_port"] == inst["container_port"]
            && gw["engine_port"] == 8000
    );
    assert!(
        gw["name"] == json!(format!("{}-gw", inst["container_name"].as_str().unwrap()))
            && gw["instance"] == inst["id"]
    );
}

#[test]
fn test_default_gateway_image_is_the_pinned_digest_and_allowlisted() {
    let s = Settings::defaults();
    assert!(s.gateway_image == GATEWAY_IMAGE && GATEWAY_IMAGE.starts_with("nginx@sha256:62ff2089"));
    assert!(s.gateway_image_allowlist.iter().any(|i| i == GATEWAY_IMAGE));
    assert!(!s.engine_image_allowlist.iter().any(|i| i == GATEWAY_IMAGE));
}

#[tokio::test]
async fn test_internal_endpoint_field() {
    let mut env = FleetEnv::new();
    let inst = create_engine(&mut env).await;
    let got = env
        .get(&format!("{P}/instances/{}", inst["id"].as_str().unwrap()))
        .send()
        .await
        .json();
    assert_eq!(
        got["internal_endpoint"],
        json!(format!(
            "http://{}:8000",
            inst["container_name"].as_str().unwrap()
        ))
    );
    assert_eq!(got["container_port"], 8000);
}

#[tokio::test]
async fn test_the_engine_network_is_ensured_on_that_host_before_every_launch() {
    let mut env = FleetEnv::new();
    create_engine(&mut env).await;
    assert_eq!(
        env.node.ops("net.ensure_internal"),
        vec![serde_json::Map::new()]
    );
}

#[tokio::test]
async fn test_existing_non_internal_network_is_refused() {
    let mut env = FleetEnv::new();
    env.node.st().network_internal = false;
    env.seed_local_model("Qwen/Qwen3-32B", None, 65_530_000_000, "local");
    let r = env
        .post(&format!("{P}/instances"))
        .json(&create_engine_body(json!({})))
        .send()
        .await;
    assert!(
        r.status == 409
            && r.code() == "conflict"
            && r.json()["detail"]
                .as_str()
                .unwrap()
                .contains("not --internal")
    );
    assert!(env.node.ops("docker.run").is_empty() && env.node.ops("gateway.start").is_empty());
}

#[tokio::test]
async fn test_gateway_failure_tears_the_engine_down() {
    let mut env = FleetEnv::new();
    env.node.fail(
        "gateway.start",
        ec_hub::domain::errors::NodeError::new("local", "docker_error", "network connect failed"),
    );
    env.seed_local_model("Qwen/Qwen3-32B", None, 65_530_000_000, "local");
    let r = env
        .post(&format!("{P}/instances"))
        .json(&create_engine_body(json!({})))
        .send()
        .await;
    assert_eq!(r.status, 502);
    assert!(
        !env.node
            .st()
            .containers
            .values()
            .any(|c| c["state"] == "running")
    );
    assert_eq!(
        env.get(&format!("{P}/instances")).send().await.json()["items"][0]["state"],
        "failed"
    );
}

#[tokio::test]
async fn test_stop_and_remove_handle_both_containers() {
    let mut env = FleetEnv::new();
    let inst = create_engine(&mut env).await;
    let (id, name) = (
        inst["id"].as_str().unwrap().to_string(),
        inst["container_name"].as_str().unwrap().to_string(),
    );
    let n0 = env.node.calls().len();
    env.post(&format!("{P}/instances/{id}/stop")).send().await;
    let stops: Vec<String> = env.node.calls()[n0..]
        .iter()
        .filter(|(o, _)| o == "docker.stop")
        .map(|(_, a)| a["name"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(stops, [format!("{name}-gw"), name.clone()]); // gateway first
    env.delete(&format!("{P}/instances/{id}")).send().await;
    let st = env.node.st();
    assert!(
        !st.containers.contains_key(&name) && !st.containers.contains_key(&format!("{name}-gw"))
    );
}

#[tokio::test]
async fn test_supervisor_fails_instance_whose_gateway_is_gone_or_foreign() {
    let mut env = FleetEnv::new();
    let inst = create_engine(&mut env).await;
    let (id, name) = (
        inst["id"].as_str().unwrap().to_string(),
        inst["container_name"].as_str().unwrap().to_string(),
    );
    env.engine_up();
    tick(&mut env).await;
    assert_eq!(
        env.get(&format!("{P}/instances/{id}")).send().await.json()["state"],
        "ready"
    );
    {
        let mut st = env.node.st();
        let gw = st.containers.get_mut(&format!("{name}-gw")).unwrap();
        gw["labels"]["engine-console.instance"] = json!("inst_someone_else");
        gw["instance"] = json!("inst_someone_else");
    }
    tick(&mut env).await;
    let got = env.get(&format!("{P}/instances/{id}")).send().await.json();
    assert!(got["state"] == "failed" && got["error"].as_str().unwrap().contains("gateway"));
    assert!(!env.node.calls().iter().any(|(o, a)| o == "docker.stop"
        && a["name"] == json!(format!("{name}-gw"))
        && a.get("instance") != Some(&json!(id))));
}

/// Python calls `LifecycleService.adopt()` (locked by §A4): a gateway alone is not adopted, its engine is.
#[tokio::test]
async fn test_adoption_ignores_gateways_and_only_adopts_engines() {
    let mut env = FleetEnv::new();
    let lab = json!({"engine-console.instance": "inst_x", "engine-console.engine": "fake", "engine-console.model": "a/b", "engine-console.port": "18020"});
    let mut gw_labels = lab.as_object().cloned().unwrap();
    gw_labels.insert("engine-console.role".into(), json!("gateway"));
    env.node.add_container(
        "ec-x-gw",
        fakefleet::ContainerOpts {
            managed: true,
            labels: gw_labels,
            ..fakefleet::ContainerOpts::default()
        },
    );
    let c = env.container();
    assert_eq!(c.lifecycle.adopt(None).await.unwrap(), 0);
    assert!(c.lifecycle.all_instances().iter().all(|i| i.id != "inst_x"));
    let mut eng_labels = lab.as_object().cloned().unwrap();
    eng_labels.insert("engine-console.role".into(), json!("engine"));
    env.node.add_container(
        "ec-x",
        fakefleet::ContainerOpts {
            managed: true,
            labels: eng_labels,
            image: "fake/engine:1.0".into(),
            ..fakefleet::ContainerOpts::default()
        },
    );
    assert_eq!(c.lifecycle.adopt(None).await.unwrap(), 1);
    assert_eq!(
        c.lifecycle
            .all_instances()
            .iter()
            .filter(|i| i.id == "inst_x")
            .count(),
        1
    );
}

// ── hostile text can never become the name the gateway forwards to or the engine is called ─────────────
#[tokio::test]
async fn test_hostile_model_file_names_become_safe_container_names() {
    let name_re = regex::Regex::new(r"\Aec-[a-z0-9][a-z0-9-]{0,58}\z").unwrap();
    let label_re = regex::Regex::new(r"\A[A-Za-z0-9][A-Za-z0-9 _.\-]{0,63}\z").unwrap();
    let long = format!("a/{}.gguf", "x".repeat(150));
    for rel in [
        "a/we'ird $name;rm -rf x.gguf",
        "a/` id `.gguf",
        "a/ünï cödé.gguf",
        "a/x y z.gguf",
        "a/{brace}.gguf",
        "a/#hash?q=1.gguf",
        long.as_str(),
        "a/-leading-dash.gguf",
        "a/..dots...gguf",
        "a/UPPER_case.GGUF.gguf",
    ] {
        let mut env = FleetEnv::new();
        env.adapters.push(Arc::new(kit::FakeMultiAdapter));
        let r = ModelRef {
            format: "gguf".into(),
            repo_id: None,
            root_id: Some("gguf".into()),
            rel: Some(rel.into()),
            name: None,
        };
        env.library.add("local", &r, CopyOpts::new("gguf", rel));
        let req: InstanceCreate =
            serde_json::from_value(json!({"engine": "fakecpp", "model": r})).unwrap();
        let inst = env.container().lifecycle.create(&req, None).await.unwrap();
        let spec = env.node.st().run_specs.last().cloned().unwrap();
        let gw = env.node.ops("gateway.start").last().cloned().unwrap();
        let sname = spec["name"].as_str().unwrap();
        assert!(
            name_re.is_match(sname) && gw["engine_host"] == json!(sname),
            "{rel}"
        );
        assert!(
            name_re.is_match(gw["name"].as_str().unwrap())
                && gw["name"] == json!(format!("{sname}-gw"))
        );
        assert!(label_re.is_match(&inst.name)); // the label adoption re-reads is boring too
        let port = gw["host_port"].as_i64().unwrap();
        assert!(
            spec["labels"]["engine-console.name"] == json!(inst.name)
                && (18000..=18099).contains(&port)
        );
    }
}

// ── console egress ─────────────────────────────────────────────────────────────────────────────────────
fn egress_client(
    dir: &Path,
    endpoint: &str,
    proxy: Option<String>,
    ca: Option<&str>,
    require: bool,
) -> HfHttpClient {
    let bundle = dir.join("ca.pem");
    if let Some(text) = ca {
        std::fs::write(&bundle, text).unwrap();
    }
    let host = url::Url::parse(endpoint)
        .ok()
        .and_then(|u| u.host_str().map(str::to_string))
        .unwrap_or_default();
    HfHttpClient::new(
        endpoint,
        vec![host],
        Arc::new(|| None),
        EgressConfig {
            proxy,
            ca_bundle: Some(bundle),
            require_proxy: require,
        },
    )
}

fn code_of(e: &AppError) -> (u16, String, String) {
    match e {
        AppError::Problem(p) => (p.status, p.code.clone(), p.detail.clone()),
        AppError::Node(n) => (0, n.code.clone(), n.message.clone()),
    }
}

#[tokio::test]
async fn test_missing_ca_fails_closed_without_any_connection() {
    let d = tempfile::tempdir().unwrap();
    let (proxy, direct) = (Counter::start().await, Counter::start().await);
    let c = egress_client(
        d.path(),
        &format!("https://127.0.0.1:{}", direct.port),
        Some(format!("http://127.0.0.1:{}", proxy.port)),
        None,
        false,
    );
    let e = c
        .search("x", None, None, None, "downloads", 1, true)
        .await
        .unwrap_err();
    let (status, code, _) = code_of(&e);
    assert_eq!((status, code.as_str()), (503, "egress_proxy_unavailable"));
    assert_eq!((proxy.hits(), direct.hits()), (0, 0));
}

#[tokio::test]
async fn test_unreachable_proxy_fails_closed_never_direct() {
    let d = tempfile::tempdir().unwrap();
    let direct = Counter::start().await;
    // nothing listens on port 1
    let c = egress_client(
        d.path(),
        &format!("https://127.0.0.1:{}", direct.port),
        Some("http://127.0.0.1:1".into()),
        Some(TEST_CA),
        false,
    );
    assert_eq!(
        code_of(&c.model("org/m", None).await.unwrap_err()).1,
        "egress_proxy_unavailable"
    );
    assert_eq!(
        code_of(&c.open_file("org/m", "main", "f", 0).await.err().unwrap()).1,
        "egress_proxy_unavailable"
    );
    assert_eq!(
        direct.hits(),
        0,
        "the endpoint must never be reached directly when a proxy is configured"
    );
}

#[tokio::test]
async fn test_require_proxy_without_proxy_refuses() {
    let d = tempfile::tempdir().unwrap();
    let direct = Counter::start().await;
    let c = egress_client(
        d.path(),
        &format!("https://127.0.0.1:{}", direct.port),
        None,
        Some(TEST_CA),
        true,
    );
    assert_eq!(
        code_of(&c.card("org/m", None).await.unwrap_err()).1,
        "egress_proxy_unavailable"
    );
    assert_eq!(direct.hits(), 0);
}

#[tokio::test]
async fn test_bad_ca_file_fails_closed() {
    let d = tempfile::tempdir().unwrap();
    let c = egress_client(
        d.path(),
        "https://127.0.0.1:9",
        Some("http://127.0.0.1:1".into()),
        Some("not a certificate"),
        false,
    );
    assert_eq!(
        code_of(&c.card("org/m", None).await.unwrap_err()).1,
        "egress_proxy_unavailable"
    );
}

/// Python checks the client was built with `trust_env=False`; here: proxy variables in the console's environment
/// (set for a child process of this test binary) are ignored and the configured proxy is the one used.
#[test]
fn test_proxied_client_ignores_env_and_uses_the_proxy() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let ambient = rt.block_on(Counter::start());
    let url = format!("http://127.0.0.1:{}", ambient.port);
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "child_proxied_client_ignores_env",
            "--nocapture",
            "--test-threads=1",
            "--include-ignored",
        ])
        .env("WSB_CHILD", "1")
        .envs([
            ("HTTPS_PROXY", &url),
            ("HTTP_PROXY", &url),
            ("ALL_PROXY", &url),
            ("https_proxy", &url),
            ("http_proxy", &url),
            ("all_proxy", &url),
        ])
        .stdin(std::process::Stdio::null())
        .output()
        .unwrap();
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(out.status.success() && text.contains("1 passed"), "{text}");
    assert_eq!(ambient.hits(), 0, "an ambient proxy variable was used");
}

#[test]
fn child_proxied_client_ignores_env() {
    if std::env::var("WSB_CHILD").is_err() {
        return;
    }
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        let d = tempfile::tempdir().unwrap();
        let proxy = ConnectProxy::start().await;
        let target = Counter::start().await;
        let c = egress_client(
            d.path(),
            &format!("https://127.0.0.1:{}", target.port),
            Some(proxy.url()),
            Some(TEST_CA),
            false,
        );
        let _ = c.card("org/m", None).await;
        assert_eq!(proxy.targets(), [format!("127.0.0.1:{}", target.port)]);
    });
}

/// Python maps the response with `HfHttpClient._raise`; here the block page really comes back through a proxy
/// tunnel from a TLS server the client trusts through EGRESS_CA_BUNDLE.
#[tokio::test]
async fn test_proxy_block_page_maps_to_egress_denied() {
    if !has_openssl() {
        return;
    }
    let d = tempfile::tempdir().unwrap();
    let certs = make_certs(&d.path().join("pki"));
    let srv = TlsServer::start(&certs, &d.path().join("www"));
    srv.serve("/org/m/raw/main/README.md", BLOCK_PAGE);
    let proxy = ConnectProxy::start().await;
    let ca = std::fs::read_to_string(&certs.ca).unwrap();
    let c = egress_client(
        d.path(),
        &format!("https://127.0.0.1:{}", srv.port),
        Some(proxy.url()),
        Some(&ca),
        false,
    );
    let (status, code, _) = code_of(&c.card("org/m", None).await.unwrap_err());
    assert_eq!((status, code.as_str()), (403, "egress_denied"));
}

#[tokio::test]
async fn test_health_and_settings_show_egress_mode() {
    let mut env = FleetEnv::new();
    let h = env.get(&format!("{P}/health")).send().await.json();
    assert_eq!(h["egress_mode"], "direct");
    assert!(
        h["warnings"]
            .as_array()
            .unwrap()
            .iter()
            .any(|w| w.as_str().unwrap().contains("DIRECT"))
    );
    assert_eq!(
        env.get(&format!("{P}/settings")).send().await.json()["egress_mode"],
        "direct"
    );
    // Python changes the live settings object; the Rust settings are immutable, so a second console starts with it
    let mut env2 =
        FleetEnv::with(|c| c.egress_proxy = Some("http://user:hunter2@127.0.0.1:8082".into()));
    let h2 = env2.get(&format!("{P}/health")).send().await.json();
    assert_eq!(h2["egress_mode"], "proxied");
    assert!(
        !h2["warnings"]
            .as_array()
            .unwrap()
            .iter()
            .any(|w| w.as_str().unwrap().contains("DIRECT"))
    );
    let s = env2.get(&format!("{P}/settings")).send().await.json();
    assert!(
        s["egress_proxy"] == "http://127.0.0.1:8082"
            && !s.to_string().contains("hunter2")
            && s["require_egress_proxy"] == false
    );
}

#[test]
fn test_public_proxy_strips_credentials() {
    assert_eq!(
        public_proxy(Some("http://u:p@h:1")).as_deref(),
        Some("http://h:1")
    );
    assert_eq!(public_proxy(None), None);
}

#[tokio::test]
async fn test_egress_failure_is_a_503_problem_over_the_api() {
    let d = tempfile::tempdir().unwrap();
    let mut cfg = Settings::defaults();
    cfg.data_dir = d.path().join("data");
    cfg.hf_cache_dir = d.path().join("hf");
    cfg.hf_token_file = d.path().join("no-token");
    cfg.hosts_file = d.path().join("hosts.json");
    cfg.allowed_hosts = vec!["testserver".into()];
    cfg.trust_loopback = true;
    cfg.otlp_enabled = false;
    cfg.fleet_enabled = false;
    let hub = egress_client(
        d.path(),
        "https://127.0.0.1:9",
        Some("http://127.0.0.1:1".into()),
        None,
        false,
    );
    let c = build_container(
        Arc::new(cfg),
        Overrides {
            hub: Some(Arc::new(hub) as Arc<dyn HubClient>),
            frontend: Some(None),
            ..Overrides::default()
        },
    )
    .unwrap();
    let st = app_state(c);
    let req = http::Request::builder()
        .method("GET")
        .uri(format!("{P}/hf/models/org/m"))
        .header("host", "testserver")
        .header("x-engine-console", "1")
        .body(AxBody::empty())
        .unwrap();
    let r = handle_request(&st, "127.0.0.1", req).await;
    assert_eq!(r.status().as_u16(), 503);
    let body: Value =
        serde_json::from_slice(&r.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(body["code"], "egress_proxy_unavailable");
}

#[test]
fn test_defaults_are_standalone() {
    let d = Settings::defaults();
    assert!(d.egress_ca_bundle.is_none() && d.egress_proxy.is_none() && !d.require_egress_proxy);
    assert!(
        (d.hf_cache_dir
            .file_name()
            .is_some_and(|n| n == "huggingface")
            && d.hf_cache_dir
                .components()
                .any(|c| c.as_os_str() == ".cache"))
            || d.hf_cache_dir.is_absolute()
    );
    assert!(!d.hf_cache_dir.display().to_string().contains("/fast"));
}

/// Python checks the proxied client was built; here: with no CA bundle the client still goes through the proxy (and
/// a certificate the system does not trust then fails as a CA problem, never as a direct connection).
#[tokio::test]
async fn test_proxy_without_ca_bundle_uses_system_trust_store() {
    let proxy = ConnectProxy::start().await;
    let target = Counter::start().await;
    let endpoint = format!("https://127.0.0.1:{}", target.port);
    let c = HfHttpClient::new(
        &endpoint,
        vec!["127.0.0.1".into()],
        Arc::new(|| None),
        EgressConfig {
            proxy: Some(proxy.url()),
            ca_bundle: None,
            require_proxy: false,
        },
    );
    let _ = c.card("org/m", None).await;
    assert_eq!(proxy.targets(), [format!("127.0.0.1:{}", target.port)]);
}

// ── host allowlist matching (self-contained reference implementation of a proxy allowlist) ─────────────
const SAMPLE_ALLOWLIST: &str = "
# console egress allowlist (glob per line)
huggingface.co
*.huggingface.co
cdn-lfs*.hf.co
*.hf.co
";

/// egress_allowlist_ref.py `load_patterns`
fn load_patterns(text: &str) -> Vec<String> {
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(str::to_string)
        .collect()
}

/// egress_allowlist_ref.py `allowed` (fnmatch: POSIX case-sensitive)
fn allowed(host: &str, patterns: &[String]) -> bool {
    patterns.iter().any(|p| fnmatchcase(host, p))
}

#[test]
fn test_console_hosts_match_a_typical_proxy_allowlist() {
    let pats = load_patterns(SAMPLE_ALLOWLIST);
    let hf = Settings::defaults().hf_allowed_hosts;
    for host in [
        "huggingface.co",
        "cdn-lfs.huggingface.co",
        "cdn-lfs-us-1.hf.co",
        "cdn-lfs.hf.co",
        "cas-bridge.xethub.hf.co",
        "transfer.xethub.hf.co",
        "x.hf.co",
    ] {
        assert!(allowed(host, &pats), "{host}");
        assert!(host_allowed(host, &hf), "{host}"); // the in-app list agrees
    }
}

#[test]
fn test_allowlist_matching_blocks_lookalikes() {
    let pats = load_patterns(SAMPLE_ALLOWLIST);
    let hf = Settings::defaults().hf_allowed_hosts;
    for host in [
        "evil.example",
        "huggingface.co.evil.example",
        "nothf.co",
        "hf.co.evil.example",
    ] {
        assert!(!allowed(host, &pats), "{host}");
        assert!(!host_allowed(host, &hf), "{host}");
    }
}

/// Python unit-tests `_transport_error`; here a real TLS verification failure through the proxy (a server
/// certificate from a CA the configured bundle does not hold) must be reported as a CA problem, not "unreachable".
#[tokio::test]
async fn test_tls_verify_failure_is_reported_as_a_ca_problem_not_unreachable() {
    if !has_openssl() {
        return;
    }
    let d = tempfile::tempdir().unwrap();
    let certs = make_certs(&d.path().join("pki"));
    let other = make_certs(&d.path().join("other-pki"));
    let srv = TlsServer::start(&certs, &d.path().join("www"));
    srv.serve(
        "/org/m/raw/main/README.md",
        "HTTP/1.0 200 OK\r\nContent-Type: text/plain\r\n\r\n# card\n",
    );
    let proxy = ConnectProxy::start().await;
    let wrong_ca = std::fs::read_to_string(&other.ca).unwrap();
    let c = egress_client(
        d.path(),
        &format!("https://127.0.0.1:{}", srv.port),
        Some(proxy.url()),
        Some(&wrong_ca),
        false,
    );
    let (_, code, detail) = code_of(&c.card("org/m", None).await.unwrap_err());
    assert_eq!(code, "egress_proxy_unavailable");
    assert!(detail.contains("EGRESS_CA_BUNDLE"), "{detail}");
}

#[test]
fn test_env_file_supplies_settings_and_real_env_wins() {
    let d = tempfile::tempdir().unwrap();
    let f = d.path().join("env");
    std::fs::write(&f, "HF_CACHE_DIR=/data/hf\nDISCOVERY_ENABLED=false\n").unwrap();
    let s = Settings::load(&[("HOME".into(), d.path().display().to_string())], Some(&f)).unwrap();
    assert!(s.hf_cache_dir == Path::new("/data/hf") && !s.discovery_enabled);
    let s2 = Settings::load(&[("HF_CACHE_DIR".into(), "/other".into())], Some(&f)).unwrap();
    assert_eq!(s2.hf_cache_dir, Path::new("/other"));
}

/// The egress fixtures themselves (curl through the CONNECT proxy to the TLS server, trusting the throwaway CA).
#[tokio::test]
async fn test_egress_fixtures_serve_a_block_page_through_the_proxy() {
    if !has_openssl()
        || std::process::Command::new(sys_tool("curl"))
            .arg("--version")
            .output()
            .is_err()
    {
        return;
    }
    let d = tempfile::tempdir().unwrap();
    let certs = make_certs(&d.path().join("pki"));
    let srv = TlsServer::start(&certs, &d.path().join("www"));
    srv.serve("/org/m/raw/main/README.md", BLOCK_PAGE);
    let proxy = ConnectProxy::start().await;
    let out = tokio::process::Command::new(sys_tool("curl"))
        .args(["-s", "--noproxy", "", "-w", "\n%{http_code}", "--cacert"])
        .arg(&certs.ca)
        .args(["--proxy", &proxy.url()])
        .arg(format!(
            "https://127.0.0.1:{}/org/m/raw/main/README.md",
            srv.port
        ))
        .env_clear()
        .env("PATH", std::env::var("PATH").unwrap_or_default())
        .output()
        .await
        .unwrap();
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(
        text.contains("egress blocked by addon_guard") && text.trim_end().ends_with("403"),
        "{text}"
    );
    assert_eq!(proxy.targets(), [format!("127.0.0.1:{}", srv.port)]);
}

/// A system tool the test needs, from the fixed safe PATH (never the caller's PATH, PORTING.md §A12): an absolute path,
/// or one that does not exist (the spawn then fails as it would without the tool).
fn sys_tool(name: &str) -> std::path::PathBuf {
    ec_hub::tools::find_in(name, ec_hub::tools::SAFE_PATH)
        .unwrap_or_else(|| std::path::PathBuf::from(format!("/nonexistent/{name}")))
}
