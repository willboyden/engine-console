//! Port of backend/tests/core/test_externals.py: engines the console did not create (external and script instances),
//! read from each host's pushed inventory: ids, states, who may control them, VRAM and RAM attribution, model
//! location, and every mutating route refused when the node policy says monitor-only.

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

use std::collections::BTreeMap;
use std::sync::Arc;

use ec_hub::domain::errors::NodeError;
use ec_hub::domain::models::Instance;
use fakefleet::{
    ContainerOpts, EndpointOpts, FakeNode, FleetEnv, HelloOpts, big_model_script, cgroup,
    cgroup_default, ports,
};
use serde_json::{Value, json};
use wse_kit::{P, obj};

const LEAKS: &[&str] = &["hf_supersecret", "argvsecret", "s3cr3t-key", "Bearer abc"];

fn add(node: &FakeNode, name: &str, image: &str, engine: Option<&str>, host_ports: Vec<Value>) {
    node.add_container(
        name,
        ContainerOpts {
            image: image.into(),
            engine: engine.map(str::to_string),
            host_ports,
            ..ContainerOpts::default()
        },
    );
}

fn ep(
    node: &FakeNode,
    port: i64,
    container: Option<&str>,
    engine: Option<&str>,
    models: &[&str],
    health: Option<i64>,
    models_status: Option<i64>,
    metrics: Option<i64>,
    auth: bool,
) {
    node.add_endpoint(
        port,
        EndpointOpts {
            container: container.map(str::to_string),
            engine: engine.map(str::to_string),
            models: models.iter().map(|m| (*m).to_string()).collect(),
            health,
            models_status,
            metrics,
            auth_required: auth,
        },
    );
}

fn seed_world(node: &FakeNode) {
    node.add_container(
        "vllm-a",
        ContainerOpts {
            image: "vllm/vllm-openai:v0.23.0".into(),
            engine: Some("vllm".into()),
            host_ports: ports(18500, 8000, "0.0.0.0"),
            params: obj(json!({"model": "org/big", "api_key": "[redacted]", "port": "8000"})),
            mounts: vec![json!({"source": "/data/hf", "dest": "/root/.cache/huggingface", "ro": true, "root": "hf", "rel": ""})],
            gpus: vec!["GPU-uuid-0".into()],
            ..ContainerOpts::default()
        },
    );
    ep(
        node,
        18500,
        Some("vllm-a"),
        Some("vllm"),
        &["org/big"],
        Some(200),
        Some(200),
        Some(200),
        false,
    );
    add(
        node,
        "sgl-b",
        "lmsysorg/sglang:v0.5.14",
        Some("sglang"),
        ports(18501, 30000, "0.0.0.0"),
    );
    ep(
        node,
        18501,
        Some("sgl-b"),
        Some("sglang"),
        &[],
        Some(401),
        Some(401),
        None,
        true,
    );
    add(
        node,
        "ollama-c",
        "ollama/ollama:0.5",
        Some("ollama"),
        ports(18502, 11434, "0.0.0.0"),
    );
    ep(
        node,
        18502,
        Some("ollama-c"),
        Some("ollama"),
        &["llama3"],
        None,
        Some(200),
        None,
        false,
    );
    add(
        node,
        "router",
        "ghcr.io/berriai/litellm:main",
        None,
        ports(4000, 4000, "0.0.0.0"),
    ); // not an engine
    add(
        node,
        "vllm-noport",
        "vllm/vllm-openai:v0.23.0",
        Some("vllm"),
        Vec::new(),
    );
    add(
        node,
        "postgres",
        "postgres:16",
        None,
        ports(5432, 5432, "0.0.0.0"),
    );
    node.add_container(
        "mine-gw",
        ContainerOpts {
            managed: true,
            labels: obj(json!({"engine-console.role": "gateway"})),
            host_ports: ports(18503, 8080, "127.0.0.1"),
            ..ContainerOpts::default()
        },
    );
    ep(
        node,
        18503,
        Some("mine-gw"),
        Some("vllm"),
        &["owned"],
        Some(200),
        Some(200),
        None,
        false,
    );
    ep(
        node,
        9100,
        None,
        Some("vllm"),
        &["bare-model"],
        None,
        Some(200),
        Some(200),
        false,
    ); // no container
    ep(node, 9101, None, None, &[], Some(200), None, None, false); // health only: not an engine
}

fn by_name(items: Vec<Instance>) -> BTreeMap<String, Instance> {
    items.into_iter().map(|i| (i.name.clone(), i)).collect()
}

fn externals(env: &mut FleetEnv, host: Option<&str>) -> Vec<Instance> {
    env.container()
        .lifecycle
        .list_page(500, None, host)
        .unwrap()
        .items
        .into_iter()
        .filter(|i| i.management != "managed")
        .collect()
}

// ---- external instances ---------------------------------------------------------------------------------------------------------
/// Security control: what a node reports is listed with neutral fields, and no secret reaches the API.
#[tokio::test]
async fn test_external_engines_are_listed_with_ids_states_and_fields_and_everything_else_is_ignored()
 {
    let mut env = FleetEnv::new();
    seed_world(&env.node);
    let got = by_name(externals(&mut env, None));
    assert_eq!(
        got.keys().map(String::as_str).collect::<Vec<_>>(),
        [
            "127.0.0.1:9100",
            "ollama-c",
            "sgl-b",
            "vllm-a",
            "vllm-noport"
        ]
    );
    let a = &got["vllm-a"];
    assert_eq!(
        (
            a.id.as_str(),
            a.managed,
            a.source.as_str(),
            a.engine.as_str(),
            a.state.as_str(),
            a.management.as_str()
        ),
        (
            "ext-local-vllm-a",
            false,
            "external",
            "vllm",
            "ready",
            "external"
        )
    );
    assert!(
        a.endpoint.as_deref() == Some("http://127.0.0.1:18500")
            && a.port == Some(18500)
            && a.served_models == ["org/big"]
    );
    assert!(
        a.container_name.as_deref() == Some("vllm-a")
            && a.image.as_deref() == Some("vllm/vllm-openai:v0.23.0")
            && a.repo_id == "org/big"
    );
    assert!(
        a.params["model"] == "org/big"
            && a.params["api_key"] == "[redacted]"
            && a.params["port"] == "8000"
    );
    assert!(
        a.gpu_ids == [0]
            && a.gpu_uuids == ["GPU-uuid-0"]
            && a.host_id == "local"
            && a.host_label.as_deref() == Some("testbox")
    );
    assert!(a.history_since.is_some() && a.history_since == Some(a.created_at));
    assert!(
        got["sgl-b"].state == "auth_required"
            && got["sgl-b"]
                .state_reason
                .clone()
                .unwrap_or_default()
                .contains("API key")
    );
    assert!(
        got["ollama-c"].engine == "ollama"
            && got["ollama-c"].served_models == ["llama3"]
            && got["ollama-c"].state == "ready"
    );
    let np = &got["vllm-noport"];
    assert!(
        np.state == "unreachable"
            && np
                .state_reason
                .clone()
                .unwrap_or_default()
                .contains("no published host port")
    );
    assert!(np.port.is_none() && np.endpoint.is_none());
    let bare = &got["127.0.0.1:9100"];
    assert!(
        bare.engine == "vllm"
            && bare.container_name.is_none()
            && bare.id == "ext-local-127-0-0-1-9100"
    );
    let blob = serde_json::to_string(&got.values().collect::<Vec<_>>()).unwrap();
    for leak in LEAKS
        .iter()
        .chain(&["HF_TOKEN", "VLLM_API_KEY", "sk-live", "hf_super"])
    {
        assert!(!blob.contains(leak), "{leak}");
    }
    // routers, databases, our own gateways
    assert!(!blob.contains("router") && !blob.contains("postgres") && !blob.contains("owned"));
}

/// Security control: a token-shaped value in a node's params is masked even if the node forgot.
#[tokio::test]
async fn test_a_token_shaped_value_in_a_nodes_params_is_masked_even_if_the_node_forgot() {
    let mut env = FleetEnv::new();
    env.node.add_container(
        "leaky",
        ContainerOpts {
            engine: Some("vllm".into()),
            host_ports: ports(18600, 8000, "0.0.0.0"),
            params: obj(json!({"hf_note": "token hf_abcdefghijkl1234 here"})),
            ..ContainerOpts::default()
        },
    );
    ep(
        &env.node,
        18600,
        Some("leaky"),
        Some("vllm"),
        &["m"],
        Some(200),
        Some(200),
        None,
        false,
    );
    let inst = by_name(externals(&mut env, None)).remove("leaky").unwrap();
    let p = Value::Object(inst.params).to_string();
    assert!(!p.contains("hf_abcdefgh") && p.contains("[redacted]"));
}

#[tokio::test]
async fn test_ids_are_per_host_and_stable_and_collisions_get_distinct_suffixes() {
    let mut env = FleetEnv::new();
    let spark = env.add_host("spark", "Spark", HelloOpts::default());
    for node in [&env.node, &spark] {
        for n in ["My Engine", "my-engine"] {
            let port = if n == "My Engine" { 18700 } else { 18701 };
            add(
                node,
                n,
                "vllm/vllm-openai:v0.23.0",
                Some("vllm"),
                ports(port, 8000, "0.0.0.0"),
            );
        }
    }
    let first: Vec<String> = externals(&mut env, None)
        .into_iter()
        .map(|i| i.id)
        .collect();
    let mut uniq = first.clone();
    uniq.sort();
    uniq.dedup();
    assert!(first.len() == 4 && uniq.len() == 4);
    assert_eq!(
        first,
        externals(&mut env, None)
            .into_iter()
            .map(|i| i.id)
            .collect::<Vec<_>>()
    ); // stable
    assert_eq!(
        first.iter().filter(|i| i.starts_with("ext-spark-")).count(),
        2
    );
    assert_eq!(
        first.iter().filter(|i| i.starts_with("ext-local-")).count(),
        2
    );
}

/// Security review 2026-10-04 M1: host `gpu` running `2-vllm` and host `gpu-2` running `vllm` both gave
/// `ext-gpu-2-vllm`, and a stop on the compromised host's row was sent to the honest host.
#[tokio::test]
async fn test_m1_hyphenated_host_ids_cannot_collide_and_stop_reaches_only_the_row_clicked() {
    let mut env = FleetEnv::new();
    let o = |addr: &str| HelloOpts {
        external_control: "stop_start".into(),
        addresses: vec![addr.into()],
        ..HelloOpts::default()
    };
    let honest = env.add_host("gpu-2", "Honest", o("10.0.0.22"));
    let evil = env.add_host("gpu", "Compromised", o("10.0.0.21"));
    add(
        &honest,
        "vllm",
        "vllm/vllm-openai:v0.23.0",
        Some("vllm"),
        ports(8000, 8000, "127.0.0.1"),
    );
    ep(
        &honest,
        8000,
        Some("vllm"),
        Some("vllm"),
        &["honest-model"],
        Some(200),
        Some(200),
        None,
        false,
    );
    add(
        &evil,
        "2-vllm",
        "vllm/vllm-openai:v0.23.0",
        Some("vllm"),
        ports(8000, 8000, "127.0.0.1"),
    );
    ep(
        &evil,
        8000,
        Some("2-vllm"),
        Some("vllm"),
        &["evil-model"],
        Some(200),
        Some(200),
        None,
        false,
    );
    let c = env.container();
    let rows: BTreeMap<String, String> = c
        .lifecycle
        .externals(None)
        .into_iter()
        .filter(|i| i.host_id == "gpu" || i.host_id == "gpu-2")
        .map(|i| (i.id, i.host_id))
        .collect();
    assert_eq!(rows.len(), 2);
    let evil_id = rows.iter().find(|(_, h)| *h == "gpu").unwrap().0.clone();
    assert_eq!(c.lifecycle.get(&evil_id).unwrap().host_id, "gpu");
    c.lifecycle.stop(&evil_id).await.unwrap();
    assert!(!honest.calls().iter().any(|(o, _)| o.starts_with("docker.")));
    assert!(evil.calls().iter().any(|(o, _)| o.starts_with("docker.")));
}

/// Security control: hostile container and model names are neutralised before they reach ids and the UI.
#[tokio::test]
async fn test_hostile_container_and_model_names_are_neutralised() {
    let mut env = FleetEnv::new();
    let hostile = "../../etc; rm -rf $(id) `x`";
    add(
        &env.node,
        hostile,
        "vllm/vllm-openai:v0.23.0",
        Some("vllm"),
        ports(18600, 8000, "0.0.0.0"),
    );
    let long_model = format!("m\x1b[31m{}", "x".repeat(150));
    ep(
        &env.node,
        18600,
        Some(hostile),
        Some("vllm"),
        &[long_model.as_str()],
        Some(200),
        Some(200),
        None,
        false,
    );
    let ext = externals(&mut env, None)
        .into_iter()
        .find(|i| i.container_name.as_deref() == Some(hostile))
        .unwrap();
    assert!(
        regex::Regex::new(r"\Aext-local-[a-z0-9-]{1,80}\z")
            .unwrap()
            .is_match(&ext.id)
    );
    assert!(!ext.id.contains("..") && !ext.id.contains('/'));
    assert!(
        regex::Regex::new(r"\A[A-Za-z0-9 _.:\-]{1,64}\z")
            .unwrap()
            .is_match(&ext.name)
    );
    assert!(
        ext.served_models
            .iter()
            .all(|m| !m.contains('\x1b') && m.chars().count() <= 200)
    );
}

#[tokio::test]
async fn test_a_vanished_external_shows_stopped_for_a_while_then_disappears() {
    let mut env = FleetEnv::new();
    let clock = Arc::new(std::sync::Mutex::new(1000.0_f64));
    let c = env.container();
    let k = Arc::clone(&clock);
    c.lifecycle.set_clock(Arc::new(move || *k.lock().unwrap()));
    seed_world(&env.node);
    assert!(by_name(externals(&mut env, None)).contains_key("vllm-a"));
    wse_kit::remove_container(&env.node, "vllm-a");
    let got = by_name(externals(&mut env, None));
    let a = &got["vllm-a"];
    assert!(
        a.state == "stopped"
            && a.state_reason
                .clone()
                .unwrap_or_default()
                .contains("no longer running")
    );
    assert!(a.vram.is_empty() && a.host_memory.is_none());
    *clock.lock().unwrap() += ec_hub::services::lifecycle::GONE_TTL_S + 1.0;
    assert!(!by_name(externals(&mut env, None)).contains_key("vllm-a"));
}

#[tokio::test]
async fn test_a_vanished_external_does_not_show_as_stopped_when_its_host_just_went_away() {
    let mut env = FleetEnv::new();
    seed_world(&env.node);
    externals(&mut env, None);
    env.node.set_state("down", None);
    let states: Vec<String> = externals(&mut env, None)
        .into_iter()
        .map(|i| i.state)
        .collect();
    assert!(!states.is_empty() && states.iter().all(|s| s == "unreachable")); // not live, not "stopped"
    env.node.set_state("up", None);
    assert_eq!(by_name(externals(&mut env, None))["vllm-a"].state, "ready");
}

#[tokio::test]
async fn test_first_seen_is_kept_and_history_since_matches_it() {
    let mut env = FleetEnv::new();
    seed_world(&env.node);
    let t0 = by_name(externals(&mut env, None))["vllm-a"].history_since;
    assert!(t0.is_some() && by_name(externals(&mut env, None))["vllm-a"].history_since == t0);
}

#[tokio::test]
async fn test_get_one_external_instance_and_404_for_unknown() {
    let mut env = FleetEnv::new();
    seed_world(&env.node);
    let one = env
        .get(&format!("{P}/instances/ext-local-vllm-a"))
        .send()
        .await
        .json();
    assert!(
        one["state"] == "ready"
            && one["managed"] == json!(false)
            && one["management"] == "external"
    );
    assert_eq!(
        env.get(&format!("{P}/instances/ext-local-nope"))
            .send()
            .await
            .status,
        404
    );
    let rows = env
        .container()
        .store
        .all("SELECT * FROM instances", &[])
        .unwrap();
    assert!(!format!("{rows:?}").contains("vllm-a")); // nothing external is persisted
}

#[tokio::test]
async fn test_controls_and_clone_as_managed_for_externals_without_permission() {
    let mut env = FleetEnv::new();
    seed_world(&env.node);
    let c = by_name(externals(&mut env, None))["vllm-a"]
        .controls
        .clone();
    assert!(!(c.start || c.stop || c.restart || c.remove || c.logs) && c.configure); // clone as managed is always offered
    assert!(c.reason.unwrap_or_default().contains("monitor-only"));
}

#[tokio::test]
async fn test_controls_follow_the_nodes_external_control_policy_and_mode() {
    let mut env = FleetEnv::new();
    seed_world(&env.node);
    env.node.set_policy("external_control", json!("stop_start"));
    let c = by_name(externals(&mut env, None))["vllm-a"]
        .controls
        .clone();
    assert!(c.stop && !c.start && !c.restart && !c.remove && !c.logs && c.configure);
    env.node.set_mode("monitor");
    let c = by_name(externals(&mut env, None))["vllm-a"]
        .controls
        .clone();
    assert!(!c.stop && c.reason.unwrap_or_default().contains("monitor mode"));
    env.node.set_mode("manage");
    env.node.set_state("down", None);
    let c = by_name(externals(&mut env, None))["vllm-a"]
        .controls
        .clone();
    assert!(!c.stop && c.reason.unwrap_or_default().contains("not connected"));
}

// ---- monitor-only enforcement ---------------------------------------------------------------------------------------------------
/// Security control: every mutating instance route is 409 for external engines by default, before any node call.
#[tokio::test]
async fn test_every_mutating_instance_route_is_409_for_external_engines_by_default() {
    let mut env = FleetEnv::new();
    seed_world(&env.node);
    let spec: Value = serde_json::from_str(include_str!("../assets/openapi.json")).unwrap();
    let mut paths: Vec<(String, String)> = Vec::new();
    for (p, ops) in spec["paths"].as_object().unwrap() {
        if !p.contains("/instances/{iid}") {
            continue;
        }
        for m in ops.as_object().unwrap().keys() {
            let m = m.to_uppercase();
            if ["POST", "PUT", "PATCH", "DELETE"].contains(&m.as_str()) {
                paths.push((m, p.clone()));
            }
        }
    }
    let mut methods: Vec<&str> = paths.iter().map(|(m, _)| m.as_str()).collect();
    methods.sort_unstable();
    methods.dedup();
    assert!(methods == ["DELETE", "PATCH", "POST"] && paths.len() >= 5);
    for (method, path) in &paths {
        let r = env
            .req(method, &path.replace("{iid}", "ext-local-vllm-a"))
            .json(&json!({"pinned": true}))
            .send()
            .await;
        assert!(
            r.status == 409 && r.code() == "instance_not_managed",
            "{method} {path} {}",
            r.text()
        );
    }
    for path in ["logs", "logs/stream", "command"] {
        let r = env
            .get(&format!("{P}/instances/ext-local-vllm-a/{path}"))
            .send()
            .await;
        assert!(
            r.status == 409 && r.code() == "instance_not_managed",
            "{path}"
        );
    }
    assert!(!env.node.calls().iter().any(|(o, _)| {
        [
            "docker.stop",
            "docker.rm",
            "docker.run",
            "docker.start",
            "docker.external",
            "gateway.start",
        ]
        .contains(&o.as_str())
    })); // the node was never even asked
}

#[tokio::test]
async fn test_stop_and_start_work_when_the_policy_allows_and_go_through_docker_external() {
    let mut env = FleetEnv::new();
    seed_world(&env.node);
    env.node.set_policy("external_control", json!("stop_start"));
    let r = env
        .post(&format!("{P}/instances/ext-local-vllm-a/stop"))
        .send()
        .await;
    assert!(r.status == 200 && r.json()["state"] == "stopped");
    assert_eq!(
        env.node.ops("docker.external"),
        [obj(json!({"name": "vllm-a", "action": "stop"}))]
    );
    // the engine is no longer running, yet it stays listed so it can be started again
    let listed = env.get(&format!("{P}/instances")).send().await.json();
    let row = listed["items"]
        .as_array()
        .unwrap()
        .iter()
        .find(|i| i["id"] == "ext-local-vllm-a")
        .unwrap()
        .clone();
    assert!(row["state"] == "stopped" && row["controls"]["start"] == json!(true));
    assert_eq!(
        env.post(&format!("{P}/instances/ext-local-vllm-a/restart"))
            .send()
            .await
            .status,
        409
    ); // not offered
    let again = env
        .post(&format!("{P}/instances/ext-local-vllm-a/start"))
        .send()
        .await;
    assert_eq!(again.status, 200);
    assert_eq!(
        env.node.ops("docker.external").last().cloned().unwrap(),
        obj(json!({"name": "vllm-a", "action": "start"}))
    );
    // once it runs again it is a normal external instance
    assert_eq!(
        env.get(&format!("{P}/instances/ext-local-vllm-a"))
            .send()
            .await
            .json()["state"],
        "ready"
    );
    assert_eq!(
        env.delete(&format!("{P}/instances/ext-local-vllm-a"))
            .send()
            .await
            .status,
        409
    ); // never removable
}

/// Security control: the node's refusal is a 403 problem naming the host; a policy flip the hub sees is refused first.
#[tokio::test]
async fn test_a_refusal_by_the_node_is_a_403_problem_naming_the_host_and_the_instance_is_left_alone()
 {
    let mut env = FleetEnv::new();
    seed_world(&env.node);
    env.node.set_policy("external_control", json!("stop_start"));
    // the operator edited the node's policy after its last hello: the node refuses, and its words reach the user
    env.node.fail(
        "docker.external",
        NodeError::new(
            "local",
            "policy_denied",
            "this node no longer allows controlling engines",
        ),
    );
    let r = env
        .post(&format!("{P}/instances/ext-local-vllm-a/stop"))
        .send()
        .await;
    assert!(
        r.status == 403
            && r.code() == "policy_denied"
            && r.json()["detail"]
                .as_str()
                .unwrap()
                .starts_with("testbox: ")
    );
    assert_eq!(
        env.get(&format!("{P}/instances/ext-local-vllm-a"))
            .send()
            .await
            .json()["state"],
        "ready"
    );
    env.node.unfail("docker.external");
    env.node.set_policy("external_control", json!("none")); // and a flip the hub already sees is refused before the node
    let n = env.node.calls().len();
    assert_eq!(
        env.post(&format!("{P}/instances/ext-local-vllm-a/stop"))
            .send()
            .await
            .status,
        409
    );
    assert!(env.node.ops("docker.external").len() == 1 && env.node.calls().len() >= n);
}

/// Security control: a viewer key cannot force discovery (admin-only route).
#[tokio::test]
async fn test_viewer_cannot_force_discovery() {
    let mut env = FleetEnv::new();
    let c = env.container();
    c.settings.ensure_bootstrap_key(&env.cfg.data_dir).unwrap();
    let boot = std::fs::read_to_string(env.cfg.data_dir.join("bootstrap-admin.key"))
        .unwrap()
        .trim()
        .to_string();
    let v = env
        .post(&format!("{P}/keys"))
        .client("203.0.113.9")
        .bearer(&boot)
        .json(&json!({"name": "v", "role": "viewer"}))
        .send()
        .await
        .json()["secret"]
        .as_str()
        .unwrap()
        .to_string();
    let as_viewer = env
        .post(&format!("{P}/instances/discover"))
        .client("203.0.113.9")
        .bearer(&v)
        .send()
        .await;
    assert_eq!(as_viewer.status, 403);
    let as_admin = env
        .post(&format!("{P}/instances/discover"))
        .client("203.0.113.9")
        .bearer(&boot)
        .send()
        .await;
    assert_eq!(as_admin.status, 200);
}

// ---- VRAM, memory, model location ------------------------------------------------------------------------------------------------
#[tokio::test]
async fn test_vram_host_memory_and_model_location_of_an_external() {
    let mut env = FleetEnv::new();
    seed_world(&env.node);
    {
        let mut st = env.node.st();
        st.gpu_procs.insert(0, vec![json!({"pid": 7, "container": "vllm-a", "used_bytes": 60_i64 * (1 << 30), "name": "python"})]);
        st.cg.insert("vllm-a".into(), cgroup_default());
        st.disks = vec![
            json!({"mount": "/data", "fstype": "zfs", "source": "tank", "size": 1_000_000_000_000_i64,
                               "used": 100_000_000_000_i64, "free": 900_000_000_000_i64, "roots": ["hf"]}),
        ];
    }
    let got = by_name(externals(&mut env, None));
    let a = &got["vllm-a"];
    let v: Vec<(i64, String, i64)> = a
        .vram
        .iter()
        .map(|v| {
            (
                v.gpu_index,
                v.gpu_uuid.clone(),
                v.used_gib.unwrap().round() as i64,
            )
        })
        .collect();
    assert_eq!(v, [(0, "GPU-uuid-0".to_string(), 60)]);
    let m = a.host_memory.clone().unwrap();
    assert!(m.source == "cgroup" && (m.shmem_gib - 64.0).abs() < 1e-3);
    let loc = a.model_location.clone().unwrap();
    assert_eq!(
        (
            loc.host_id.as_str(),
            loc.root_id.as_str(),
            loc.root_label.as_deref(),
            loc.path.as_str()
        ),
        ("local", "hf", Some("HF cache"), "/data/hf")
    );
    assert!(
        loc.drive.as_deref() == Some("/data (zfs)")
            && loc.fstype.as_deref() == Some("zfs")
            && !loc.remote
            && loc.shared_from.is_none()
    );
    assert!(got["sgl-b"].vram.is_empty() && got["sgl-b"].host_memory.is_none());
}

#[tokio::test]
async fn test_model_location_on_nfs_names_the_machine_that_owns_the_disk() {
    let mut env = FleetEnv::new();
    env.add_host(
        "spark",
        "Spark",
        HelloOpts {
            addresses: vec!["10.0.0.7".into()],
            ..HelloOpts::default()
        },
    );
    env.node.add_container(
        "vllm-n",
        ContainerOpts {
            engine: Some("vllm".into()),
            host_ports: ports(18500, 8000, "0.0.0.0"),
            mounts: vec![json!({"source": "/mnt/shared/hf", "dest": "/root/.cache/huggingface", "ro": true, "root": "shared", "rel": "hf"})],
            ..ContainerOpts::default()
        },
    );
    let mut roots = env.node.policy()["model_roots"]
        .as_array()
        .cloned()
        .unwrap();
    roots.push(json!({"id": "shared", "path": "/mnt/shared", "kind": "auto", "writable": false, "label": "Shared (NFS)"}));
    env.node.set_policy("model_roots", Value::Array(roots));
    ep(
        &env.node,
        18500,
        Some("vllm-n"),
        Some("vllm"),
        &["m"],
        Some(200),
        Some(200),
        None,
        false,
    );
    env.node.st().disks = vec![
        json!({"mount": "/mnt/shared", "fstype": "nfs4", "source": "10.0.0.7:/models", "remote": true,
                                      "remote_host": "10.0.0.7", "remote_path": "/models", "roots": ["shared"]}),
    ];
    let loc = by_name(externals(&mut env, Some("local")))["vllm-n"]
        .model_location
        .clone()
        .unwrap();
    assert!(
        loc.remote
            && loc.shared_from.as_deref() == Some("spark")
            && loc.drive.as_deref() == Some("NFS from Spark")
    );
    env.node.st().disks[0]["remote_host"] = json!("10.9.9.9"); // an export from a machine that is not ours
    let loc = by_name(externals(&mut env, Some("local")))["vllm-n"]
        .model_location
        .clone()
        .unwrap();
    assert_eq!(loc.shared_from.as_deref(), Some("10.9.9.9"));
}

// ---- chat, bench on externals ------------------------------------------------------------------------------------------------------
#[tokio::test]
async fn test_chat_and_bench_can_target_external_with_confirmation() {
    let mut env = FleetEnv::new();
    seed_world(&env.node);
    env.port_route(18500, "/v1/chat/completions", sse_engine());
    let r = env
        .post(&format!("{P}/chat/completions"))
        .json(&json!({"instance_id": "ext-local-vllm-a", "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await;
    let last = env.engine_requests().last().cloned().unwrap();
    assert!(r.status == 200 && last.json()["model"] == "org/big" && last.port == 18500);
    let sgl = env
        .post(&format!("{P}/chat/completions"))
        .json(&json!({"instance_id": "ext-local-sgl-b", "messages": [{"role": "user", "content": "x"}]}))
        .send()
        .await;
    assert_eq!(sgl.status, 409);
    let no = env
        .post(&format!("{P}/bench"))
        .json(&json!({"instance_id": "ext-local-vllm-a"}))
        .send()
        .await;
    assert!(no.status == 400 && no.code() == "confirm_external_required");
    let not_ready = env
        .post(&format!("{P}/bench"))
        .json(&json!({"instance_id": "ext-local-sgl-b", "confirm_external": true}))
        .send()
        .await;
    assert_eq!(not_ready.status, 409);
    let ok = env
        .post(&format!("{P}/bench"))
        .json(&json!({"instance_id": "ext-local-vllm-a", "confirm_external": true,
                      "suite": {"concurrency": [1], "prompt_tokens": 14, "max_tokens": 2, "rounds": 1}}))
        .send()
        .await;
    assert!(
        ok.status == 202
            && ok.json()["suite"]["note"]
                .as_str()
                .unwrap()
                .contains("external engine")
    );
}

#[tokio::test]
async fn test_an_external_on_another_host_is_chatted_through_that_hosts_node() {
    let mut env = FleetEnv::new();
    let spark = env.add_host("spark", "Spark", HelloOpts::default());
    add(
        &spark,
        "vllm-s",
        "vllm/vllm-openai:v0.23.0",
        Some("vllm"),
        ports(18500, 8000, "0.0.0.0"),
    );
    ep(
        &spark,
        18500,
        Some("vllm-s"),
        Some("vllm"),
        &["org/spark-model"],
        Some(200),
        Some(200),
        None,
        false,
    );
    spark.serve(Some(18500), "/v1/chat/completions", sse_engine());
    let r = env
        .post(&format!("{P}/chat/completions"))
        .json(&json!({"instance_id": "ext-spark-vllm-s", "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await;
    assert_eq!(r.status, 200);
    assert!(spark.http_requests().len() == 1 && env.node.http_requests().is_empty());
    assert_eq!(spark.http_requests()[0].json()["model"], "org/spark-model");
}

/// test_api_smoke.py `sse_engine`: an OpenAI-compatible streaming answer.
/// test_api_smoke.py `sse_engine`: an SSE stream when the request asks for one (`stream: true`), else one JSON
/// completion with its usage (the bench and a non-streamed chat).
fn sse_engine() -> fakefleet::HttpHandler {
    fakefleet::http_handler(|req| {
        let streamed = req.json().get("stream").and_then(Value::as_bool) == Some(true);
        if streamed {
            let mut text = String::new();
            for w in ["Hel", "lo", " there"] {
                text.push_str(&format!(
                    "data: {}\n\n",
                    json!({"choices": [{"delta": {"content": w}}]})
                ));
            }
            text.push_str(&format!(
                "data: {}\n\n",
                json!({"choices": [], "usage": {"prompt_tokens": 20, "completion_tokens": 3}})
            ));
            text.push_str("data: [DONE]\n\n");
            return fakefleet::HttpResp::text(200, &text)
                .header("content-type", "text/event-stream");
        }
        fakefleet::HttpResp::json(
            200,
            &json!({"choices": [{"message": {"content": "Hello there", "reasoning_content": "hmm"}}],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 3}}),
        )
    })
}

// ---- script instances --------------------------------------------------------------------------------------------------------------------
fn with_script(node: &FakeNode) {
    node.set_policy("scripts", json!([big_model_script()]));
}

fn script_instance(env: &mut FleetEnv, host: &str) -> Instance {
    externals(env, Some(host))
        .into_iter()
        .find(|i| i.management == "script")
        .unwrap()
}

#[tokio::test]
async fn test_a_script_is_one_instance_even_when_nothing_runs_and_its_containers_are_not_externals()
{
    let mut env = FleetEnv::new();
    with_script(&env.node);
    let inst = script_instance(&mut env, "local");
    assert!(
        inst.id == "scr-local-big-model"
            && inst.script.as_deref() == Some("big-model")
            && inst.state == "stopped"
    );
    assert!(
        inst.name == "Big model (2 GPUs)"
            && inst.engine == "sglang"
            && inst.container_name.is_none()
    );
    assert!(!inst.managed && inst.management == "script");
    // two containers of the script's engine start: still ONE instance, and neither is listed as an external
    env.node.add_container(
        "myproj-sglang-a",
        ContainerOpts {
            engine: Some("sglang".into()),
            image: "lmsysorg/sglang:v0.5.14".into(),
            host_ports: ports(18510, 30000, "0.0.0.0"),
            gpus: vec!["GPU-uuid-0".into(), "GPU-uuid-1".into()],
            ..ContainerOpts::default()
        },
    );
    add(
        &env.node,
        "myproj-sglang-helper",
        "busybox",
        None,
        Vec::new(),
    );
    ep(
        &env.node,
        18510,
        Some("myproj-sglang-a"),
        Some("sglang"),
        &["qwen-big"],
        Some(200),
        Some(200),
        None,
        false,
    );
    let items = externals(&mut env, None);
    assert_eq!(
        items.iter().map(|i| i.id.as_str()).collect::<Vec<_>>(),
        ["scr-local-big-model"]
    );
    let inst = &items[0];
    assert!(
        inst.state == "ready"
            && inst.port == Some(18510)
            && inst.container_name.as_deref() == Some("myproj-sglang-a")
    );
    assert!(
        inst.served_models == ["qwen-big"] && inst.repo_id == "qwen-big" && inst.gpu_ids == [0, 1]
    );
}

#[tokio::test]
async fn test_a_script_engine_that_is_running_but_not_answering_yet_is_loading() {
    let mut env = FleetEnv::new();
    with_script(&env.node);
    add(
        &env.node,
        "myproj-sglang-a",
        "vllm/vllm-openai:v0.23.0",
        Some("sglang"),
        ports(18510, 30000, "0.0.0.0"),
    );
    assert_eq!(script_instance(&mut env, "local").state, "loading");
    ep(
        &env.node,
        18510,
        Some("myproj-sglang-a"),
        Some("sglang"),
        &[],
        Some(401),
        Some(401),
        None,
        false,
    );
    assert_eq!(script_instance(&mut env, "local").state, "auth_required");
}

#[tokio::test]
async fn test_script_controls_map_to_the_scripts_actions_and_name_the_script() {
    let mut env = FleetEnv::new();
    with_script(&env.node);
    let c = script_instance(&mut env, "local").controls;
    assert!(c.start && !c.stop && c.restart && !c.remove && !c.logs && !c.configure);
    assert!(c.reason.unwrap_or_default().contains("Big model"));
    add(
        &env.node,
        "myproj-sglang-a",
        "vllm/vllm-openai:v0.23.0",
        Some("sglang"),
        ports(18510, 30000, "0.0.0.0"),
    );
    let c = script_instance(&mut env, "local").controls;
    assert!(c.stop && !c.start && c.restart);
    let mut s = big_model_script();
    let actions: Vec<Value> = s["actions"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|a| a["name"] != "down")
        .cloned()
        .collect();
    s["actions"] = Value::Array(actions);
    env.node.set_policy("scripts", json!([s]));
    let c = script_instance(&mut env, "local").controls;
    assert!(!c.stop && !c.restart); // no `down` action: nothing to map stop to
    env.node.set_mode("monitor");
    let c = script_instance(&mut env, "local").controls;
    assert!(
        !(c.start || c.stop || c.restart) && c.reason.unwrap_or_default().contains("monitor mode")
    );
}

#[tokio::test]
async fn test_script_vram_and_memory_aggregate_over_the_scripts_containers() {
    let mut env = FleetEnv::new();
    with_script(&env.node);
    add(
        &env.node,
        "myproj-sglang-a",
        "vllm/vllm-openai:v0.23.0",
        Some("sglang"),
        ports(18510, 30000, "0.0.0.0"),
    );
    add(
        &env.node,
        "myproj-sglang-b",
        "vllm/vllm-openai:v0.23.0",
        Some("sglang"),
        Vec::new(),
    );
    {
        let mut st = env.node.st();
        st.gpu_procs.insert(
            0,
            vec![
                json!({"pid": 1, "container": "myproj-sglang-a", "used_bytes": 30_i64 * (1 << 30)}),
                json!({"pid": 2, "container": "myproj-sglang-b", "used_bytes": 10_i64 * (1 << 30)}),
            ],
        );
        st.cg
            .insert("myproj-sglang-a".into(), cgroup(10.0, 1.0, 9.0, 8.0, 0.1));
        st.cg
            .insert("myproj-sglang-b".into(), cgroup(5.0, 1.0, 4.0, 3.0, 0.1));
    }
    let inst = script_instance(&mut env, "local");
    assert_eq!(
        inst.vram
            .iter()
            .map(|v| v.used_gib.unwrap().round() as i64)
            .collect::<Vec<_>>(),
        [40]
    );
    assert!((inst.host_memory.unwrap().total_gib - 15.0).abs() < 1e-3);
}

/// Security control: a policy script with an odd name is not listed rather than trusted.
#[tokio::test]
async fn test_a_policy_script_with_an_odd_name_is_not_listed_rather_than_trusted() {
    let mut env = FleetEnv::new();
    let mut s = big_model_script();
    s["name"] = json!("../evil name");
    env.node.set_policy("scripts", json!([s]));
    assert!(
        !externals(&mut env, None)
            .iter()
            .any(|i| i.management == "script")
    );
}

#[tokio::test]
async fn test_script_instance_ids_are_per_host() {
    let mut env = FleetEnv::new();
    let spark = env.add_host("spark", "Spark", HelloOpts::default());
    with_script(&env.node);
    with_script(&spark);
    let mut ids: Vec<String> = externals(&mut env, None)
        .into_iter()
        .filter(|i| i.management == "script")
        .map(|i| i.id)
        .collect();
    ids.sort();
    assert_eq!(ids, ["scr-local-big-model", "scr-spark-big-model"]);
}

#[tokio::test]
async fn test_instance_for_container_resolves_managed_script_and_external_engines() {
    let mut env = FleetEnv::new();
    with_script(&env.node);
    seed_world(&env.node);
    add(
        &env.node,
        "myproj-sglang-a",
        "vllm/vllm-openai:v0.23.0",
        Some("sglang"),
        Vec::new(),
    );
    let c = env.container();
    let life = &c.lifecycle;
    life.externals(None);
    assert_eq!(
        life.instance_for_container("local", "vllm-a").as_deref(),
        Some("ext-local-vllm-a")
    );
    assert_eq!(
        life.instance_for_container("local", "myproj-sglang-a")
            .as_deref(),
        Some("scr-local-big-model")
    );
    assert_eq!(life.instance_for_container("local", "postgres"), None);
}

#[tokio::test]
async fn test_an_external_engine_answering_503_is_loading_not_unreachable() {
    let mut env = FleetEnv::new();
    add(
        &env.node,
        "sgl-warm",
        "vllm/vllm-openai:v0.23.0",
        Some("sglang"),
        ports(18510, 30000, "0.0.0.0"),
    );
    ep(
        &env.node,
        18510,
        Some("sgl-warm"),
        Some("sglang"),
        &[],
        Some(503),
        None,
        None,
        false,
    );
    let inst = by_name(externals(&mut env, None))
        .remove("sgl-warm")
        .unwrap();
    assert!(
        inst.state == "loading"
            && inst.state_reason.unwrap_or_default().contains("503")
            && inst.port == Some(18510)
    );
    ep(
        &env.node,
        18510,
        Some("sgl-warm"),
        Some("sglang"),
        &[],
        Some(200),
        Some(200),
        None,
        false,
    );
    assert_eq!(
        by_name(externals(&mut env, None))["sgl-warm"].state,
        "ready"
    );
    ep(
        &env.node,
        18510,
        Some("sgl-warm"),
        Some("sglang"),
        &[],
        None,
        None,
        None,
        false,
    );
    assert_eq!(
        by_name(externals(&mut env, None))["sgl-warm"].state,
        "unreachable"
    );
}
