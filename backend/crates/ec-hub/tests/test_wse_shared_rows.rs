//! The lifecycle rows of shared Python test files (the rest of each file is ported next to its other rows):
//! test_api_smoke.py, test_audit_fixes.py, test_security.py, test_wiring.py, test_library_lifecycle_wiring.py and
//! test_runspec_ulimits.py.

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

use std::time::Duration;

use ec_hub::adapters::base::{Hardware, LaunchContext};
use ec_hub::adapters::load_default_adapters;
use ec_hub::domain::errors::NodeError;
use ec_hub::domain::fleet::NodeClient;
use ec_hub::domain::refs::ModelRef;
use ec_hub::services::libport::LibraryPort;
use fakefleet::{FleetEnv, GpuFree, HttpResp, cgroup_default, http_handler};
use serde_json::{Value, json};
use wse_kit::{P, QWEN, SECRET, run_specs, seed, set_logs, update_container};

fn tick_sync(env: &mut FleetEnv) -> impl std::future::Future<Output = ()> + use<> {
    let c = env.container();
    async move {
        c.lifecycle.tick().await.unwrap();
    }
}

/// test_api_smoke.py `make_ready_instance`
async fn make_ready_instance(env: &mut FleetEnv, name: &str) -> Value {
    if env.library.find("local", &ModelRef::hf(QWEN)).is_none() {
        seed(env, "local");
    }
    env.engine_up();
    let gpus = if name == "qwen" {
        json!([0])
    } else {
        json!([1])
    };
    let r = env
        .post(&format!("{P}/instances"))
        .json(&json!({"engine": "fake", "repo_id": QWEN, "name": name, "params": {"max_model_len": 8192}, "gpu_ids": gpus}))
        .send()
        .await;
    assert_eq!(r.status, 201, "{}", r.text());
    let inst = r.json();
    set_logs(
        &env.node,
        inst["container_name"].as_str().unwrap(),
        "loading weights 40%\n",
    );
    tick_sync(env).await;
    let id = inst["id"].as_str().unwrap();
    let got = env.get(&format!("{P}/instances/{id}")).send().await.json();
    assert_eq!(got["state"], "ready");
    got
}

fn problem_code(r: &kit::Resp) -> String {
    assert!(
        r.header("content-type")
            .unwrap()
            .starts_with("application/problem+json"),
        "{}",
        r.text()
    );
    assert_eq!(r.json()["status"], json!(r.status));
    r.code()
}

// ---- test_api_smoke.py --------------------------------------------------------------------------------------------
#[tokio::test]
async fn test_instance_lifecycle_end_to_end() {
    let mut env = FleetEnv::new();
    seed(&env, "local");
    let r = env
        .post(&format!("{P}/instances"))
        .json(&json!({"engine": "fake", "repo_id": QWEN, "params": {"max_model_len": 8192}}))
        .send()
        .await;
    assert_eq!(r.status, 201, "{}", r.text());
    let inst = r.json();
    let port = inst["port"].as_i64().unwrap();
    assert!(
        inst["state"] == "starting"
            && (18000..=18099).contains(&port)
            && inst["gpu_uuids"] == json!(["GPU-uuid-0"])
    );
    assert!(
        inst["host_id"] == "local"
            && inst["management"] == "managed"
            && inst["controls"]["stop"] == json!(true)
    );
    let spec = run_specs(&env.node)[0].clone();
    assert!(
        spec["gpu_uuids"] == json!(["GPU-uuid-0"])
            && spec["network_alias"] == inst["container_name"]
    );
    assert!(
        spec.get("host_port").is_none()
            && env.node.ops("gateway.start")[0]["host_port"] == json!(port)
    );
    assert_eq!(
        spec["secret_env"],
        json!({"FAKE_API_KEY": "s3cr3t-key-value"})
    );
    let mut rest = spec.as_object().unwrap().clone();
    rest.remove("secret_env");
    assert!(!Value::Object(rest).to_string().contains("s3cr3t-key-value"));
    assert_eq!(spec["env"]["HF_HUB_OFFLINE"], "1");
    let name = inst["container_name"].as_str().unwrap().to_string();
    let id = inst["id"].as_str().unwrap().to_string();
    set_logs(&env.node, &name, "boot\nloading weights 55%\n");
    tick_sync(&mut env).await;
    let got = env.get(&format!("{P}/instances/{id}")).send().await.json();
    assert!(got["state"] == "loading" && got["progress_pct"] == json!(55.0));
    env.engine_up();
    tick_sync(&mut env).await;
    assert_eq!(
        env.get(&format!("{P}/instances/{id}")).send().await.json()["state"],
        "ready"
    );
    assert_eq!(
        env.get(&format!("{P}/instances")).send().await.json()["items"][0]["state"],
        "ready"
    );
    set_logs(
        &env.node,
        &name,
        "args: api_key=s3cr3t-key-value token hf_abcdefghijkl1234\n",
    );
    let logs = env
        .get(&format!("{P}/instances/{id}/logs"))
        .send()
        .await
        .text();
    assert!(!logs.contains("s3cr3t") && !logs.contains("hf_abcdef") && logs.contains("[redacted]"));
    let once = env
        .get(&format!("{P}/instances/{id}/logs/stream?once=true"))
        .send()
        .await
        .text();
    assert!(once.contains("event: log") && !once.contains("s3cr3t"));
    let cmd = env
        .get(&format!("{P}/instances/{id}/command"))
        .send()
        .await
        .json();
    assert!(
        cmd["docker_run"].as_str().unwrap().contains("docker run")
            && !cmd.to_string().contains("s3cr3t")
    );
    assert!(
        cmd["engine_cli"].as_str().unwrap().contains("--model")
            && cmd["compose_yaml"].as_str().unwrap().contains("GPU-uuid-0")
    );
    assert_eq!(
        env.patch(&format!("{P}/instances/{id}"))
            .json(&json!({"pinned": true}))
            .send()
            .await
            .json()["pinned"],
        json!(true)
    );
    assert_eq!(
        env.post(&format!("{P}/instances/{id}/stop"))
            .send()
            .await
            .json()["state"],
        "stopped"
    );
    assert_eq!(
        env.post(&format!("{P}/instances/{id}/start"))
            .send()
            .await
            .json()["state"],
        "starting"
    );
    assert_eq!(
        env.post(&format!("{P}/instances/{id}/restart"))
            .send()
            .await
            .json()["state"],
        "starting"
    );
    set_logs(&env.node, &name, "CUDA error: out of memory\n");
    update_container(
        &env.node,
        &name,
        json!({"state": "exited", "exit_code": 137}),
    );
    tick_sync(&mut env).await;
    let failed = env.get(&format!("{P}/instances/{id}")).send().await.json();
    assert!(failed["state"] == "failed" && failed["error"].as_str().unwrap().contains("code 137"));
    assert!(
        failed["last_logs"]
            .as_str()
            .unwrap()
            .contains("out of memory")
    );
    assert_eq!(
        env.delete(&format!("{P}/instances/{id}"))
            .send()
            .await
            .status,
        204
    );
    assert_eq!(
        problem_code(&env.get(&format!("{P}/instances/{id}")).send().await),
        "not_found"
    );
}

#[tokio::test]
async fn test_instance_preflight_blocks() {
    let mut env = FleetEnv::new();
    let body = json!({"engine": "fake", "repo_id": QWEN, "params": {}});
    let r = env.post(&format!("{P}/instances")).json(&body).send().await;
    assert!(r.status == 409 && problem_code(&r) == "preflight_failed");
    let codes: Vec<(Value, Value)> = r.json()["preflight"]["checks"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| (c["code"].clone(), c["level"].clone()))
        .collect();
    assert!(codes.contains(&(json!("model_missing"), json!("block"))));
    seed(&env, "local");
    env.node
        .st()
        .missing_images
        .insert("fake/engine:1.0".into());
    let pf = env
        .post(&format!("{P}/instances/preflight"))
        .json(&body)
        .send()
        .await
        .json();
    let blocks: Vec<Value> = pf["checks"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["level"] == "block")
        .map(|c| c["code"].clone())
        .collect();
    assert!(pf["ok"] == json!(false) && blocks == [json!("image_missing")]);
    env.node.st().missing_images.clear();
    let ok = env
        .post(&format!("{P}/instances/preflight"))
        .json(
            &json!({"engine": "fake", "repo_id": QWEN, "params": {"tp": 1, "max_model_len": 4096}}),
        )
        .send()
        .await
        .json();
    assert_eq!(ok["ok"], json!(true));
    env.node.set_gpu_free(GpuFree::All(10.0)); // other tenants ate the VRAM
    let pf = env
        .post(&format!("{P}/instances/preflight"))
        .json(&body)
        .send()
        .await
        .json();
    assert!(pf["ok"] == json!(false) && pf["fit"]["verdict"] == "wont_fit");
    let bad = env
        .post(&format!("{P}/instances"))
        .json(&json!({"engine": "fake", "repo_id": QWEN, "params": {"nope": 1}}))
        .send()
        .await;
    assert_eq!(problem_code(&bad), "preflight_failed");
}

#[tokio::test]
async fn test_docker_failure_marks_failed() {
    let mut env = FleetEnv::new();
    seed(&env, "local");
    env.node.fail(
        "docker.run",
        NodeError::new("local", "docker_error", "no such gpu"),
    );
    let r = env
        .post(&format!("{P}/instances"))
        .json(&json!({"engine": "fake", "repo_id": QWEN}))
        .send()
        .await;
    assert!(r.status == 502 && problem_code(&r) == "docker_error");
    assert_eq!(
        env.get(&format!("{P}/instances")).send().await.json()["items"][0]["state"],
        "failed"
    );
}

#[tokio::test]
async fn test_two_instances_get_distinct_ports_and_ttl_stop() {
    let mut env = FleetEnv::new();
    let a = make_ready_instance(&mut env, "qwen").await;
    let b = make_ready_instance(&mut env, "qwen-b").await;
    assert!(a["port"] != b["port"] && a["gpu_ids"] == json!([0]) && b["gpu_ids"] == json!([1]));
    let (ia, ib) = (
        a["id"].as_str().unwrap().to_string(),
        b["id"].as_str().unwrap().to_string(),
    );
    env.patch(&format!("{P}/instances/{ia}"))
        .json(&json!({"ttl_idle_s": 1}))
        .send()
        .await;
    let past = ec_hub::services::common::now() - 100.0;
    env.container()
        .store
        .execute(
            "UPDATE instances SET last_request_at=? WHERE id=?",
            &[&past, &ia],
        )
        .unwrap();
    tick_sync(&mut env).await;
    assert_eq!(
        env.get(&format!("{P}/instances/{ia}")).send().await.json()["state"],
        "stopped"
    );
    assert_eq!(
        env.get(&format!("{P}/instances/{ib}")).send().await.json()["state"],
        "ready"
    );
}

#[tokio::test]
async fn test_instance_from_profile() {
    let mut env = FleetEnv::new();
    seed(&env, "local");
    let p = env
        .post(&format!("{P}/profiles"))
        .json(&json!({"name": "p", "engine": "fake", "params": {"max_model_len": 4096}}))
        .send()
        .await
        .json();
    let inst = env
        .post(&format!("{P}/instances"))
        .json(&json!({"engine": "fake", "repo_id": QWEN, "profile_id": p["id"]}))
        .send()
        .await
        .json();
    assert!(inst["params"] == json!({"max_model_len": 4096}) && inst["profile_id"] == p["id"]);
}

#[tokio::test]
async fn test_metrics_series_stream_and_prometheus() {
    let mut env = FleetEnv::new();
    let inst = make_ready_instance(&mut env, "qwen").await;
    // nothing was pushed for this engine, so the hub reads its /metrics itself through the node (the pull fallback)
    let bodies = std::sync::Arc::new(std::sync::Mutex::new(vec![
        "running 3\nwaiting 0\nprompt_total 300\ngen_total 150\nkv 20\nunknown_key 9\n",
        "running 2\nwaiting 1\nprompt_total 100\ngen_total 50\nkv 12.5\n",
    ]));
    env.engine_route(
        "/metrics",
        http_handler(move |_| HttpResp::text(200, bodies.lock().unwrap().pop().unwrap_or(""))),
    );
    let c = env.container();
    c.metrics.scrape_once().await;
    tokio::time::sleep(Duration::from_millis(50)).await;
    c.metrics.scrape_once().await;
    let id = inst["id"].as_str().unwrap();
    let ts = env
        .get(&format!("{P}/metrics/instances/{id}?window=15m"))
        .send()
        .await
        .json();
    assert!(ts["resolution"] == "raw" && ts["points"].as_array().unwrap().len() == 2);
    let last = ts["points"].as_array().unwrap().last().unwrap()["values"].clone();
    assert!(last["requests_running"] == json!(3.0) && last["kv_cache_usage_pct"] == json!(20.0));
    assert!(last["generation_tps"].as_f64().unwrap() > 0.0 && last.get("unknown_key").is_none()); // counter deltas
    let hourly = env
        .get(&format!("{P}/metrics/instances/{id}?window=1h"))
        .send()
        .await
        .json();
    assert!(
        hourly["resolution"] == "1h"
            && hourly["points"][0]["values"]["requests_running"] == json!(2.5)
    );
    assert_eq!(
        problem_code(
            &env.get(&format!("{P}/metrics/instances/{id}?window=soon"))
                .send()
                .await
        ),
        "invalid_window"
    );
    assert!(
        env.get(&format!("{P}/metrics/stream?once=true"))
            .send()
            .await
            .text()
            .contains("event: metrics")
    );
    let prom = env.get("/metrics").send().await;
    assert!(
        prom.status == 200
            && prom
                .text()
                .contains("engine_console_instances{state=\"ready\"} 1.0")
    );
    assert!(prom.text().contains("engine_console_http_requests_total"));
}

// ---- test_audit_fixes.py ------------------------------------------------------------------------------------------
/// Security control: the gateway's image is the node policy's, and a policy without one blocks the launch.
#[tokio::test]
async fn test_the_gateway_image_is_the_nodes_policy_and_a_policy_without_one_blocks_the_launch() {
    let mut env = FleetEnv::new();
    seed(&env, "local");
    let body = json!({"engine": "fake", "repo_id": QWEN, "params": {"max_model_len": 4096}});
    assert!(!format!("{:?}", env.node.ops("gateway.start")).contains("gateway_image")); // the hub never chooses it
    env.node.set_policy("gateway_image", Value::Null);
    let r = env.post(&format!("{P}/instances")).json(&body).send().await;
    assert!(r.status == 409 && problem_code(&r) == "preflight_failed");
    assert!(
        r.json()["preflight"]["checks"]
            .as_array()
            .unwrap()
            .iter()
            .any(|c| c["code"] == "gateway_missing")
    );
    assert!(env.node.ops("docker.run").is_empty() && env.node.ops("gateway.start").is_empty());
}

async fn run_spec(env: &mut FleetEnv, params: Value) -> Value {
    if env.library.find("local", &ModelRef::hf(QWEN)).is_none() {
        seed(env, "local");
    }
    let r = env
        .post(&format!("{P}/instances"))
        .json(&json!({"engine": "fake", "repo_id": QWEN, "params": params}))
        .send()
        .await;
    assert_eq!(r.status, 201, "{}", r.text());
    run_specs(&env.node).last().cloned().unwrap()
}

#[tokio::test]
async fn test_single_gpu_uses_private_ipc_and_multi_gpu_uses_host() {
    let mut env = FleetEnv::new();
    let a = run_spec(&mut env, json!({"max_model_len": 4096})).await;
    assert!(a["ipc_host"] == json!(false) && a["shm_size"] == "16g");
    let b = run_spec(&mut env, json!({"max_model_len": 4096, "tp": 2})).await;
    assert_eq!(b["ipc_host"], json!(true));
}

/// Security control: trust_remote_code is off unless set, and warns when on.
#[tokio::test]
async fn test_trust_remote_code_is_off_by_default_and_warns_when_on() {
    let mut env = FleetEnv::new();
    seed(&env, "local");
    let body = json!({"engine": "fake", "repo_id": QWEN, "params": {"max_model_len": 4096}});
    let checks = env
        .post(&format!("{P}/instances/preflight"))
        .json(&body)
        .send()
        .await
        .json()["checks"]
        .clone();
    assert!(
        !checks
            .as_array()
            .unwrap()
            .iter()
            .any(|c| c["code"] == "trust_remote_code")
    );
    let on = json!({"engine": "fake", "repo_id": QWEN, "params": {"max_model_len": 4096, "trust_remote_code": true}});
    let checks = env
        .post(&format!("{P}/instances/preflight"))
        .json(&on)
        .send()
        .await
        .json()["checks"]
        .clone();
    assert!(
        checks
            .as_array()
            .unwrap()
            .iter()
            .any(|c| c["code"] == "trust_remote_code" && c["level"] == "warn")
    );
    run_spec(&mut env, json!({"max_model_len": 4096})).await;
    assert!(!run_specs(&env.node).iter().any(|s| {
        s["argv"]
            .as_array()
            .unwrap()
            .contains(&json!("--trust-remote-code"))
    }));
}

// ---- test_security.py ---------------------------------------------------------------------------------------------
/// Security control: the HF cache dir is not settable through the API and the engine mount is read-only.
#[tokio::test]
async fn test_cache_dir_not_settable_via_api_and_mount_is_readonly() {
    let mut env = FleetEnv::new();
    let before = env.get(&format!("{P}/settings")).send().await.json()["hf_cache_dir"].clone();
    let r = env
        .put(&format!("{P}/settings"))
        .json(&json!({"hf_cache_dir": "/etc"}))
        .send()
        .await;
    assert!(r.status == 200 && r.json()["hf_cache_dir"] == before);
    seed(&env, "local");
    env.post(&format!("{P}/instances"))
        .json(&json!({"engine": "fake", "repo_id": QWEN, "params": {"max_model_len": 4096}}))
        .send()
        .await;
    let mount = run_specs(&env.node)[0]["mounts"][0].clone();
    assert!(
        mount["ro"] == json!(true)
            && mount["dest"] == "/root/.cache/huggingface/hub"
            && mount["root"] == "hf"
    );
}

/// Security control: an API key parameter is never stored or returned: DB/API/audit carry `[set]`, the value lives
/// in a 0600 file and rides only in docker.run's secret_env.
#[tokio::test]
async fn test_api_key_never_stored_or_returned() {
    use std::os::unix::fs::PermissionsExt;
    let mut env = FleetEnv::new();
    seed(&env, "local");
    env.engine_up();
    let r = env
        .post(&format!("{P}/instances"))
        .json(&json!({"engine": "fake", "repo_id": QWEN, "params": {"max_model_len": 4096, "api_key": SECRET}}))
        .send()
        .await;
    assert_eq!(r.status, 201, "{}", r.text());
    let inst = r.json();
    assert!(inst["params"]["api_key"] == "[set]" && !r.text().contains(SECRET));
    let rows = env
        .container()
        .store
        .all("SELECT * FROM instances", &[])
        .unwrap();
    assert!(!format!("{rows:?}").contains(SECRET));
    let id = inst["id"].as_str().unwrap().to_string();
    let sf = env.cfg.data_dir.join("secrets").join(format!("{id}.json"));
    assert_eq!(
        std::fs::metadata(&sf).unwrap().permissions().mode() & 0o777,
        0o600
    );
    assert_eq!(
        serde_json::from_str::<Value>(&std::fs::read_to_string(&sf).unwrap()).unwrap(),
        json!({"api_key": SECRET})
    );
    let run = run_specs(&env.node)[0].clone();
    assert_eq!(run["secret_env"], json!({"FAKE_API_KEY": SECRET}));
    let mut rest = run.as_object().unwrap().clone();
    rest.remove("secret_env");
    assert!(!Value::Object(rest).to_string().contains(SECRET));
    for (op, args) in env.node.calls() {
        if op != "docker.run" {
            assert!(!Value::Object(args).to_string().contains(SECRET), "{op}");
        }
    }
    let cmd = env.get(&format!("{P}/instances/{id}/command")).send().await;
    assert!(
        !cmd.text().contains(SECRET)
            && !env
                .get(&format!("{P}/instances"))
                .send()
                .await
                .text()
                .contains(SECRET)
    );
    set_logs(
        &env.node,
        inst["container_name"].as_str().unwrap(),
        &format!("starting with key {SECRET}\n"),
    );
    assert!(
        !env.get(&format!("{P}/instances/{id}/logs"))
            .send()
            .await
            .text()
            .contains(SECRET)
    );
    let audit = env.get(&format!("{P}/audit")).send().await.text();
    assert!(!audit.contains(SECRET) && audit.contains("[redacted]"));
    // a stop/start cycle re-reads the secret file rather than needing the API to hold the value
    env.post(&format!("{P}/instances/{id}/stop")).send().await;
    assert_eq!(
        env.post(&format!("{P}/instances/{id}/start"))
            .send()
            .await
            .status,
        200
    );
    assert_eq!(
        run_specs(&env.node).last().unwrap()["secret_env"],
        json!({"FAKE_API_KEY": SECRET})
    );
    env.delete(&format!("{P}/instances/{id}")).send().await;
    assert!(!sf.exists());
}

/// Security control: a secret that would land on the container's argv is refused.
#[tokio::test]
async fn test_secret_that_would_land_in_argv_is_refused() {
    let mut env = FleetEnv::new();
    seed(&env, "local");
    let r = env
        .post(&format!("{P}/instances"))
        .json(&json!({"engine": "fake", "repo_id": QWEN, "params": {"leaky_key": SECRET}}))
        .send()
        .await;
    assert!(r.status == 409 && !r.text().contains(SECRET));
    assert!(
        r.json()["preflight"]["checks"]
            .as_array()
            .unwrap()
            .iter()
            .any(|c| c["code"] == "secret_in_argv" && c["level"] == "block")
    );
    assert!(env.node.ops("docker.run").is_empty());
}

/// Security control: a per-instance image parameter is rejected (images come from the launch form or pins).
#[tokio::test]
async fn test_per_instance_image_param_is_rejected() {
    let mut env = FleetEnv::new();
    seed(&env, "local");
    let r = env
        .post(&format!("{P}/instances"))
        .json(&json!({"engine": "fake", "repo_id": QWEN, "params": {"image": "evil/miner:1"}}))
        .send()
        .await;
    assert!(r.status == 409 && r.json()["preflight"].to_string().contains("image override"));
    let p = env
        .post(&format!("{P}/profiles"))
        .json(&json!({"name": "i", "engine": "fake", "params": {"image": "x/y:1"}}))
        .send()
        .await;
    assert_eq!(p.status, 422);
    assert!(env.node.ops("docker.run").is_empty());
}

/// Security control: the per-instance image part of the row above (it needs no profile).
#[tokio::test]
async fn test_per_instance_image_param_is_rejected_by_preflight() {
    let mut env = FleetEnv::new();
    seed(&env, "local");
    let r = env
        .post(&format!("{P}/instances"))
        .json(&json!({"engine": "fake", "repo_id": QWEN, "params": {"image": "evil/miner:1"}}))
        .send()
        .await;
    assert!(r.status == 409 && r.json()["preflight"].to_string().contains("image override"));
    assert!(env.node.ops("docker.run").is_empty());
}

/// Security control: the image must be allowed by the node's policy for pins, defaults and explicit choices.
#[tokio::test]
async fn test_the_image_must_be_allowed_by_the_nodes_policy_for_pins_defaults_and_explicit_choices()
{
    let mut env = FleetEnv::new();
    let r = env
        .put(&format!("{P}/settings"))
        .json(&json!({"image_pins": {"fake": "evil/miner:1"}}))
        .send()
        .await;
    assert!(r.status == 400 && problem_code(&r) == "image_not_allowed");
    assert_eq!(
        env.put(&format!("{P}/settings"))
            .json(&json!({"image_pins": {"fake": "fake/engine:2"}}))
            .send()
            .await
            .status,
        200
    );
    seed(&env, "local");
    let body = json!({"engine": "fake", "repo_id": QWEN});
    assert_eq!(
        env.post(&format!("{P}/instances/preflight"))
            .json(&body)
            .send()
            .await
            .json()["ok"],
        json!(true)
    );
    env.node.set_policy("engine_images", json!(["only/this:*"]));
    let pf = env
        .post(&format!("{P}/instances/preflight"))
        .json(&body)
        .send()
        .await
        .json();
    assert!(
        pf["checks"]
            .as_array()
            .unwrap()
            .iter()
            .any(|c| c["code"] == "image_not_allowed" && c["level"] == "block")
    );
    assert_eq!(
        env.post(&format!("{P}/instances"))
            .json(&body)
            .send()
            .await
            .status,
        409
    );
    env.node
        .set_policy("engine_images", json!(["fake/engine:*"]));
    let pf = env
        .post(&format!("{P}/instances/preflight"))
        .json(&json!({"engine": "fake", "repo_id": QWEN, "image": "evil/miner:1"}))
        .send()
        .await
        .json();
    assert!(
        pf["checks"]
            .as_array()
            .unwrap()
            .iter()
            .any(|c| c["code"] == "image_not_allowed")
    );
    assert!(env.node.ops("docker.run").is_empty());
}

/// Security control: every container action names the instance it belongs to (and a gateway's role).
#[tokio::test]
async fn test_every_container_action_names_the_instance_it_belongs_to() {
    let mut env = FleetEnv::new();
    seed(&env, "local");
    let inst = env
        .post(&format!("{P}/instances"))
        .json(&json!({"engine": "fake", "repo_id": QWEN, "params": {"max_model_len": 4096}}))
        .send()
        .await
        .json();
    let id = inst["id"].as_str().unwrap().to_string();
    env.post(&format!("{P}/instances/{id}/stop")).send().await;
    env.post(&format!("{P}/instances/{id}/start")).send().await;
    set_logs(&env.node, inst["container_name"].as_str().unwrap(), "x\n");
    env.get(&format!("{P}/instances/{id}/logs")).send().await;
    env.delete(&format!("{P}/instances/{id}")).send().await;
    let ownership: Vec<_> = env
        .node
        .calls()
        .into_iter()
        .filter(|(o, _)| o == "docker.stop" || o == "docker.rm")
        .collect();
    assert!(ownership.len() >= 6);
    for (op, args) in ownership {
        assert_eq!(args["instance"], json!(id), "{op} {args:?}"); // the node refuses another instance's containers
        if args["name"].as_str().unwrap().ends_with("-gw") {
            assert_eq!(args["role"], "gateway", "{op} {args:?}"); // a gateway is only touched as a gateway
        }
    }
}

/// Security control: the node's refusal to touch a foreign container is a 403 and the container is left alone.
#[tokio::test]
async fn test_the_nodes_refusal_to_touch_a_foreign_container_is_a_403_and_leaves_it_alone() {
    let mut env = FleetEnv::new();
    seed(&env, "local");
    let inst = env
        .post(&format!("{P}/instances"))
        .json(&json!({"engine": "fake", "repo_id": QWEN, "params": {"max_model_len": 4096}}))
        .send()
        .await
        .json();
    let cn = inst["container_name"].as_str().unwrap();
    update_container(&env.node, cn, json!({"labels": {}, "managed": false})); // someone else wears our name now
    let r = env
        .post(&format!(
            "{P}/instances/{}/stop",
            inst["id"].as_str().unwrap()
        ))
        .send()
        .await;
    assert!(r.status == 403 && problem_code(&r) == "not_owned");
    assert_eq!(
        wse_kit::container(&env.node, cn).unwrap()["state"],
        "running"
    );
}

/// Security control: the run spec cannot express anything that weakens the container.
#[tokio::test]
async fn test_the_run_spec_cannot_express_anything_that_weakens_the_container() {
    let mut env = FleetEnv::new();
    seed(&env, "local");
    env.post(&format!("{P}/instances"))
        .json(
            &json!({"engine": "fake", "repo_id": QWEN, "params": {"max_model_len": 4096, "tp": 2}}),
        )
        .send()
        .await;
    let spec = run_specs(&env.node)[0].clone();
    let allowed: Vec<&str> = ec_proto::specs::run_spec()
        .fields()
        .iter()
        .map(|(k, _)| *k)
        .collect();
    assert!(
        spec.as_object()
            .unwrap()
            .keys()
            .all(|k| allowed.contains(&k.as_str()))
    ); // nothing the node would drop
    for forbidden in [
        "privileged",
        "cap_add",
        "devices",
        "network_mode",
        "pid",
        "user",
        "security_opt",
        "volumes_from",
    ] {
        assert!(spec.get(forbidden).is_none(), "{forbidden}");
    }
    assert!(
        spec["mounts"]
            .as_array()
            .unwrap()
            .iter()
            .all(|m| m["ro"] == json!(true))
    ); // models are read-only
    // the caps, no-new-privileges and the pids limit are the node's, always: its rendering of this very spec shows them
    let rendered = env
        .node
        .call(
            "docker.render",
            Some(wse_kit::obj(json!({"spec": spec}))),
            Duration::from_secs(5),
        )
        .await
        .unwrap()["argv"]
        .clone();
    let argv: Vec<&str> = rendered
        .as_array()
        .unwrap()
        .iter()
        .map(|a| a.as_str().unwrap())
        .collect();
    let at = |flag: &str| argv[argv.iter().position(|a| *a == flag).unwrap() + 1];
    assert!(
        at("--cap-drop") == "ALL"
            && argv.contains(&"no-new-privileges:true")
            && at("--pids-limit") == "4096"
    );
}

// ---- test_wiring.py -----------------------------------------------------------------------------------------------
#[tokio::test]
async fn test_the_background_supervisor_and_metrics_loops_run_when_enabled() {
    let mut env = FleetEnv::with(|c| {
        c.background_tasks = true;
        c.supervisor_interval_s = 0.02;
        c.scrape_interval_s = 0.02;
    });
    seed(&env, "local");
    env.engine_up();
    let c = env.container();
    let tasks = ec_hub::app::lifespan_start(&c).await;
    let inst = env
        .post(&format!("{P}/instances"))
        .json(&json!({"engine": "fake", "repo_id": QWEN, "params": {"max_model_len": 4096}}))
        .send()
        .await
        .json();
    env.node.st().cg.insert(
        inst["container_name"].as_str().unwrap().into(),
        cgroup_default(),
    );
    let id = inst["id"].as_str().unwrap().to_string();
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while std::time::Instant::now() < deadline
        && env.get(&format!("{P}/instances/{id}")).send().await.json()["state"] != "ready"
    {
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(
        env.get(&format!("{P}/instances/{id}")).send().await.json()["state"],
        "ready"
    ); // the supervisor did it
    while std::time::Instant::now() < deadline
        && env.get(&format!("{P}/metrics/system")).send().await.json()["points"]
            .as_array()
            .unwrap()
            .is_empty()
    {
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(
        !env.get(&format!("{P}/metrics/system")).send().await.json()["points"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    ec_hub::app::lifespan_stop(&c, tasks).await;
    assert!(!env.node.ops("metrics.configure").is_empty()); // nodes are configured when the loop starts
}

// ---- test_runspec_ulimits.py --------------------------------------------------------------------------------------
#[test]
fn test_trtllm_asks_for_the_documented_stack_size() {
    let trt = load_default_adapters().get("trtllm").unwrap();
    let hw = Hardware {
        gpu_ids: vec![0],
        gpu_uuids: vec!["GPU-a".into()],
        gpu_names: vec!["x".into()],
        gpu_total_gib: vec![96.0],
        gpu_free_gib: vec![90.0],
        compute_capability: "12.0".into(),
        host_ram_gib: 256.0,
        host_id: "local".into(),
        arch: "x86_64".into(),
        unified_memory: false,
    };
    let ctx = LaunchContext {
        model: "org/m".into(),
        model_format: "hf".into(),
        model_path: None,
        served_name: "org/m".into(),
        hf_cache_container_path: "/root/.cache/huggingface".into(),
        params: ec_hub::adapters::base::Obj::new(),
        hw,
    };
    let spec = trt.build_launch_ctx(&ctx).unwrap();
    assert_eq!(Value::Object(spec.ulimits), json!({"stack": "67108864"}));
}
