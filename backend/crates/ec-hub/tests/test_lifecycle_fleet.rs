//! Port of backend/tests/core/test_lifecycle_fleet.py: managed instances across the fleet, what the hub asks each
//! host's node, in what order, and what it shows.
//!
//! Python mutates the container's settings and adapter registry after building it; here they are set on the
//! FleetEnv before its container is built (the Rust container owns copies).

#![allow(
    clippy::redundant_clone,
    clippy::cloned_ref_to_slice_refs,
    clippy::uninlined_format_args,
    clippy::manual_let_else,
    clippy::expect_used,
    clippy::panic,
    clippy::unwrap_used,
    clippy::too_many_lines,
    clippy::float_cmp,
    clippy::needless_pass_by_value,
    clippy::too_many_arguments,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::type_complexity
)]

#[path = "common/kit.rs"]
mod kit;

use kit::fakefleet;

#[path = "common/wse_kit.rs"]
mod wse_kit;

use std::sync::Arc;

use ec_hub::domain::errors::NodeError;
use ec_hub::domain::models::{InstancePatch, SettingsPatch};
use ec_hub::domain::refs::ModelRef;
use ec_hub::services::lifecycle::NAMED_VOLUMES;
use ec_hub::services::nodehttp::HttpOpts;
use fakefleet::{
    ContainerOpts, CopyOpts, EndpointOpts, FleetEnv, GpuFree, HelloOpts, cgroup_default, ports,
};
use serde_json::{Value, json};
use wse_kit::{
    P, QWEN, SECRET, add_managed, codes, container, exec, instance_row, launch, obj,
    preflight_codes, problem, remove_container, request, run_specs, seed, set_logs,
    update_container,
};

fn gguf(rel: &str) -> ModelRef {
    ModelRef {
        format: "gguf".into(),
        repo_id: None,
        root_id: Some("gguf".into()),
        rel: Some(rel.into()),
        name: None,
    }
}

// ---- the happy path -----------------------------------------------------------------------------------------------
#[tokio::test]
async fn test_create_runs_engine_then_gateway_in_order_and_publishes_the_gateway_port() {
    let mut env = FleetEnv::new();
    let inst = launch(&mut env, json!({})).await;
    let ops: Vec<String> = env
        .node
        .calls()
        .into_iter()
        .map(|(o, _)| o)
        .filter(|o| {
            [
                "net.ensure_internal",
                "docker.rm",
                "docker.run",
                "gateway.start",
                "docker.image_present",
            ]
            .contains(&o.as_str())
        })
        .collect();
    let at = |op: &str| ops.iter().position(|o| o == op).unwrap();
    assert!(at("net.ensure_internal") < at("docker.run") && at("docker.run") < at("gateway.start"));
    let run = run_specs(&env.node)[0].clone();
    let gw = env.node.ops("gateway.start")[0].clone();
    assert!(
        inst.state == "starting"
            && inst.host_id == "local"
            && inst.host_label.as_deref() == Some("testbox")
    );
    let port = inst.port.unwrap();
    assert!(
        (18000..=18099).contains(&port)
            && gw["host_port"] == json!(port)
            && gw["engine_host"] == run["name"]
    );
    assert_eq!(
        gw["name"],
        json!(format!("{}-gw", run["name"].as_str().unwrap()))
    );
    assert!(gw["instance"] == json!(inst.id) && gw["engine_port"] == json!(8000));
    assert!(
        inst.container_name.as_deref() == run["name"].as_str() && run["instance"] == json!(inst.id)
    );
    assert!(inst.gpu_uuids == ["GPU-uuid-0"] && run["gpu_uuids"] == json!(["GPU-uuid-0"]));
    assert_eq!(inst.model, Some(ModelRef::hf(QWEN)));
    assert_eq!(inst.model_location.as_ref().unwrap().root_id, "hf");
    assert!(
        inst.controls.start && inst.controls.stop && inst.controls.remove && inst.controls.logs
    );
}

/// Security control: secrets ride only in docker.run's secret_env, never argv/env/labels/mounts or other requests.
#[tokio::test]
async fn test_run_spec_carries_labels_mounts_env_and_never_a_secret_in_argv_or_plain_env() {
    let mut env = FleetEnv::new();
    let inst = launch(
        &mut env,
        json!({"params": {"max_model_len": 4096, "api_key": SECRET}}),
    )
    .await;
    let spec = run_specs(&env.node)[0].clone();
    let labels = spec["labels"].as_object().unwrap();
    for k in [
        "engine-console.role",
        "engine-console.instance",
        "engine-console.engine",
        "engine-console.model",
        "engine-console.port",
        "engine-console.name",
        "engine-console.format",
    ] {
        assert!(labels.contains_key(k), "{k}");
    }
    assert!(
        labels["engine-console.instance"] == json!(inst.id)
            && labels["engine-console.role"] == "engine"
    );
    assert!(
        labels.keys().all(|k| k.starts_with("engine-console."))
            && !labels.contains_key("engine-console")
    );
    assert_eq!(
        spec["mounts"],
        json!([{"root": "hf", "rel": "hub", "dest": "/root/.cache/huggingface/hub", "ro": true}])
    );
    assert!(
        spec["env"]["HF_HUB_OFFLINE"] == "1"
            && spec["env"]["HF_HOME"] == "/root/.cache/huggingface"
    );
    assert_eq!(spec["secret_env"], json!({"FAKE_API_KEY": SECRET}));
    for field in ["argv", "env", "labels", "mounts", "volumes", "files"] {
        assert!(
            !spec
                .get(field)
                .map(ToString::to_string)
                .unwrap_or_default()
                .contains(SECRET),
            "{field}"
        );
    }
    assert!(spec["network_alias"] == spec["name"] && spec["container_port"] == json!(8000));
    assert!(spec["shm_size"] == "16g" && spec["ipc_host"] == json!(false)); // one GPU: a private IPC namespace
    assert!(spec["env"].get("image").is_none() && spec["image"] == "fake/engine:1.0");
    for (op, args) in env.node.calls() {
        if op != "docker.run" {
            assert!(!Value::Object(args).to_string().contains(SECRET), "{op}");
        }
    }
    assert!(!serde_json::to_string(&inst).unwrap().contains(SECRET));
}

/// Security control: the hub's HF token never reaches a node.
#[tokio::test]
async fn test_hf_token_never_reaches_a_node() {
    let mut env = FleetEnv::with(|c| {
        c.hf_token = Some("hf_supersecrettokenvalue123".into());
        c.engine_hf_offline = false;
    });
    launch(&mut env, json!({})).await;
    let blob = serde_json::to_string(
        &env.node
            .calls()
            .into_iter()
            .map(|(o, a)| (o, Value::Object(a)))
            .collect::<Vec<_>>(),
    )
    .unwrap();
    assert!(!blob.contains("hf_supersecret") && !blob.contains("HF_TOKEN"));
}

#[tokio::test]
async fn test_multi_gpu_uses_host_ipc_only_when_the_policy_allows_it() {
    let mut env = FleetEnv::new();
    let inst = launch(
        &mut env,
        json!({"params": {"max_model_len": 4096, "tp": 2}}),
    )
    .await;
    assert!(run_specs(&env.node)[0]["ipc_host"] == json!(true) && inst.gpu_ids == [0, 1]);
    let other = env.add_host(
        "nohostipc",
        "No host IPC",
        HelloOpts {
            allow_ipc_host: false,
            ..HelloOpts::default()
        },
    );
    seed(&env, "nohostipc");
    env.container()
        .lifecycle
        .create(
            &request(json!({"host_id": "nohostipc", "params": {"max_model_len": 4096, "tp": 2}})),
            None,
        )
        .await
        .unwrap();
    assert_eq!(run_specs(&other)[0]["ipc_host"], json!(false));
}

#[tokio::test]
async fn test_named_volumes_follow_the_adapter() {
    let mut env = FleetEnv::new();
    launch(&mut env, json!({})).await;
    assert_eq!(run_specs(&env.node)[0]["volumes"], json!([])); // the fake engine declares none
    assert!(
        NAMED_VOLUMES.iter().find(|(e, _)| *e == "vllm").unwrap().1[0]
            .0
            .starts_with("ec-")
    );
}

// ---- image choice ----------------------------------------------------------------------------------------------------
#[tokio::test]
async fn test_image_choice_order_request_then_host_pin_then_engine_pin_then_arch_default() {
    let mut env = FleetEnv::with(|c| {
        c.engine_image_allowlist = vec![
            "fake/engine:2".into(),
            "fake/engine:2.0".into(),
            "fake/engine:3".into(),
        ];
    });
    env.node
        .set_policy("engine_images", json!(["fake/engine:*"]));
    seed(&env, "local");
    let c = env.container();
    let life = &c.lifecycle;
    let first = life.create(&request(json!({})), None).await.unwrap();
    assert_eq!(first.image.as_deref(), Some("fake/engine:1.0")); // the adapter's pinned default
    c.settings
        .update(
            &serde_json::from_value::<SettingsPatch>(
                json!({"image_pins": {"fake": "fake/engine:2"}}),
            )
            .unwrap(),
        )
        .unwrap();
    let second = life
        .create(&request(json!({"name": "second"})), None)
        .await
        .unwrap();
    assert_eq!(second.image.as_deref(), Some("fake/engine:2")); // the engine pin
    c.settings
        .update(
            &serde_json::from_value::<SettingsPatch>(
                json!({"host_image_pins": {"local": {"fake": "fake/engine:3"}}}),
            )
            .unwrap(),
        )
        .unwrap();
    let third = life
        .create(&request(json!({"name": "third"})), None)
        .await
        .unwrap();
    assert_eq!(third.image.as_deref(), Some("fake/engine:3")); // the host pin wins over the engine pin
    let fourth = life
        .create(
            &request(json!({"name": "fourth", "image": "fake/engine:9"})),
            None,
        )
        .await
        .unwrap();
    assert_eq!(fourth.image.as_deref(), Some("fake/engine:9")); // the request wins over everything
    let images: Vec<Value> = run_specs(&env.node)
        .iter()
        .map(|s| s["image"].clone())
        .collect();
    assert_eq!(
        images,
        [
            json!("fake/engine:1.0"),
            json!("fake/engine:2"),
            json!("fake/engine:3"),
            json!("fake/engine:9")
        ]
    );
}

#[tokio::test]
async fn test_arm_hosts_get_the_arm_image_of_an_engine_that_has_one() {
    let mut env = FleetEnv::new();
    env.adapters.push(Arc::new(kit::FakeMultiAdapter));
    let arm = env.add_host(
        "arm",
        "ARM box",
        HelloOpts {
            arch: "aarch64".into(),
            ..HelloOpts::default()
        },
    );
    seed(&env, "arm");
    let inst = env
        .container()
        .lifecycle
        .create(
            &request(json!({"engine": "fakecpp", "host_id": "arm"})),
            None,
        )
        .await
        .unwrap();
    assert!(
        inst.image.as_deref() == Some("fake/engine:1.0-arm64")
            && run_specs(&arm)[0]["image"] == "fake/engine:1.0-arm64"
    );
}

// ---- preflight, one block per rule ---------------------------------------------------------------------------------------
#[tokio::test]
async fn test_preflight_ok_reports_every_check_and_the_chosen_gpus() {
    let mut env = FleetEnv::new();
    seed(&env, "local");
    let (report, port) = env
        .container()
        .lifecycle
        .preflight_full(&request(json!({})), None, None)
        .await
        .unwrap();
    assert!(report.ok && port == Some(18000) && !report.gpu_ids.is_empty() && report.fit.is_some());
    let codes: Vec<&str> = report.checks.iter().map(|c| c.code.as_str()).collect();
    assert!(codes.contains(&"fit") && codes.contains(&"port"));
    assert!(!report.checks.iter().any(|c| c.level == "block"));
}

#[tokio::test]
async fn test_host_down_blocks_with_the_reason() {
    let mut env = FleetEnv::new();
    seed(&env, "local");
    env.node.set_state("down", Some("no route to host"));
    let c = env.container();
    let (report, port) = c
        .lifecycle
        .preflight_full(&request(json!({})), None, None)
        .await
        .unwrap();
    assert!(!report.ok && port.is_none());
    assert_eq!(
        report
            .checks
            .iter()
            .map(|c| c.code.as_str())
            .collect::<Vec<_>>(),
        ["host_down"]
    );
    assert!(report.checks[0].message.contains("no route to host"));
    let e = c
        .lifecycle
        .create(&request(json!({})), None)
        .await
        .unwrap_err();
    assert_eq!(problem(&e).code, "preflight_failed");
    assert!(env.node.ops("docker.run").is_empty());
}

#[tokio::test]
async fn test_monitor_mode_blocks_with_the_policy_path_and_the_advice() {
    let mut env = FleetEnv::new();
    seed(&env, "local");
    env.node.set_mode("monitor");
    let c = env.container();
    let (report, _) = c
        .lifecycle
        .preflight_full(&request(json!({})), None, None)
        .await
        .unwrap();
    let ch = report
        .checks
        .iter()
        .find(|c| c.code == "monitor_only")
        .unwrap();
    assert!(
        ch.level == "block"
            && ch.message.contains("/etc/ecnode/policy.json")
            && ch.message.contains("manage")
    );
    let e = c
        .lifecycle
        .create(&request(json!({})), None)
        .await
        .unwrap_err();
    assert_eq!(problem(&e).status, 409);
    assert!(env.node.ops("docker.run").is_empty());
}

/// Security control: the node's engine_images policy is applied before any check of the image.
#[tokio::test]
async fn test_image_not_allowed_by_the_node_policy_blocks_before_any_check_of_the_image() {
    let mut env = FleetEnv::new();
    seed(&env, "local");
    let c = env.container();
    let got = preflight_codes(&c, json!({"image": "evil/miner:1"})).await;
    assert_eq!(got["image_not_allowed"], "block");
    assert!(env.node.ops("docker.image_present").is_empty());
    env.node.set_policy("engine_images", json!(["only/this:*"]));
    assert_eq!(
        preflight_codes(&c, json!({})).await["image_not_allowed"],
        "block"
    );
}

/// Security control: a policy pattern is a literal repository, never a prefix.
#[tokio::test]
async fn test_node_policy_patterns_are_a_literal_repository_not_a_prefix() {
    let mut env = FleetEnv::new();
    seed(&env, "local");
    let c = env.container();
    for image in [
        "fake/engine:evil/x",
        "fake/engine-evil:1",
        "other/fake/engine:1",
    ] {
        assert_eq!(
            preflight_codes(&c, json!({"image": image})).await["image_not_allowed"],
            "block",
            "{image}"
        );
    }
    assert!(
        !preflight_codes(&c, json!({"image": "fake/engine:7"}))
            .await
            .contains_key("image_not_allowed")
    );
}

#[tokio::test]
async fn test_missing_image_blocks_and_says_to_pull_it_on_that_machine() {
    let mut env = FleetEnv::new();
    seed(&env, "local");
    env.node
        .st()
        .missing_images
        .insert("fake/engine:1.0".into());
    let (report, _) = env
        .container()
        .lifecycle
        .preflight_full(&request(json!({})), None, None)
        .await
        .unwrap();
    let ch = report
        .checks
        .iter()
        .find(|c| c.code == "image_missing")
        .unwrap();
    assert!(
        ch.level == "block"
            && ch.message.contains("pull it on that machine")
            && ch.message.contains("testbox")
    );
}

#[tokio::test]
async fn test_model_missing_blocks_per_host() {
    let mut env = FleetEnv::new();
    seed(&env, "local");
    let c = env.container();
    assert!(
        !preflight_codes(&c, json!({}))
            .await
            .contains_key("model_missing")
    );
    env.add_host("spark", "Spark", HelloOpts::default());
    let got = preflight_codes(&c, json!({"host_id": "spark"})).await; // the model is only on `local`
    assert_eq!(got["model_missing"], "block");
}

#[tokio::test]
async fn test_format_unsupported_blocks_when_the_engine_cannot_load_the_format() {
    let mut env = FleetEnv::new();
    let r = gguf("a/model-Q4.gguf");
    env.library
        .add("local", &r, CopyOpts::new("gguf", "a/model-Q4.gguf"));
    let c = env.container();
    let got = preflight_codes(&c, json!({"repo_id": null, "model": r})).await;
    assert_eq!(got["format_unsupported"], "block");
}

#[tokio::test]
async fn test_fit_blocks_against_that_hosts_free_vram() {
    let mut env = FleetEnv::new();
    seed(&env, "local");
    env.node.set_gpu_free(GpuFree::All(10.0));
    let c = env.container();
    let (report, _) = c
        .lifecycle
        .preflight_full(&request(json!({})), None, None)
        .await
        .unwrap();
    assert!(!report.ok && report.fit.as_ref().unwrap().verdict == "wont_fit");
    env.add_host("spark", "Spark", HelloOpts::default()); // another machine, nothing resident
    seed(&env, "spark");
    let (ok, _) = c
        .lifecycle
        .preflight_full(&request(json!({"host_id": "spark"})), None, None)
        .await
        .unwrap();
    assert!(ok.ok && ["fits", "tight"].contains(&ok.fit.unwrap().verdict.as_str()));
}

#[tokio::test]
async fn test_unified_memory_hosts_budget_vram_from_ram() {
    let mut env = FleetEnv::new();
    let gb10 = env.add_host(
        "gb10",
        "GB10",
        HelloOpts {
            gpus: 1,
            gpu_gib: 119.0,
            unified: true,
            ram_gib: 119.0,
            ..HelloOpts::default()
        },
    );
    gb10.set_mem(&[("total", 119.0), ("available", 30.0)]);
    seed(&env, "gb10");
    let (report, _) = env
        .container()
        .lifecycle
        .preflight_full(&request(json!({"host_id": "gb10"})), None, None)
        .await
        .unwrap();
    let fit = report.fit.unwrap();
    assert!(fit.per_gpu[0].free_gib < 30.0);
    assert!(fit.notes.iter().any(|n| n.contains("unified memory")));
    assert_eq!(fit.verdict, "wont_fit");
}

#[tokio::test]
async fn test_gpu_clash_warns_but_unknown_gpu_blocks() {
    let mut env = FleetEnv::new();
    let first = launch(&mut env, json!({"gpu_ids": [0]})).await;
    assert_eq!(first.gpu_ids, [0]);
    update_container(
        &env.node,
        first.container_name.as_deref().unwrap(),
        json!({"state": "running"}),
    );
    let c = env.container();
    let (report, _) = c
        .lifecycle
        .preflight_full(&request(json!({"gpu_ids": [0], "name": "b"})), None, None)
        .await
        .unwrap();
    let clash = report
        .checks
        .iter()
        .find(|c| c.code == "gpu_shared")
        .unwrap();
    assert!(clash.level == "warn" && clash.message.contains(&first.name));
    assert_eq!(
        preflight_codes(&c, json!({"gpu_ids": [7]})).await["gpu_unknown"],
        "block"
    );
}

#[tokio::test]
async fn test_a_host_without_gpus_blocks() {
    let mut env = FleetEnv::new();
    env.add_host(
        "cpu-box",
        "CPU box",
        HelloOpts {
            gpus: 0,
            ..HelloOpts::default()
        },
    );
    seed(&env, "cpu-box");
    let c = env.container();
    assert_eq!(
        preflight_codes(&c, json!({"host_id": "cpu-box"})).await["no_gpu"],
        "block"
    );
}

#[tokio::test]
async fn test_a_policy_without_a_gateway_image_blocks() {
    let mut env = FleetEnv::new();
    seed(&env, "local");
    env.node.set_policy("gateway_image", Value::Null);
    let c = env.container();
    assert_eq!(
        preflight_codes(&c, json!({})).await["gateway_missing"],
        "block"
    );
}

#[tokio::test]
async fn test_shm_above_the_policy_limit_blocks_instead_of_being_clamped() {
    let mut env = FleetEnv::new();
    seed(&env, "local");
    env.node.set_policy("max_shm_gib", json!(8.0));
    let c = env.container();
    assert_eq!(
        preflight_codes(&c, json!({})).await["invalid_launch"],
        "block"
    );
}

/// Security control: a secret that would land on the container command line is refused before anything runs.
#[tokio::test]
async fn test_secret_on_argv_is_refused_before_anything_runs() {
    let mut env = FleetEnv::new();
    seed(&env, "local");
    let e = env
        .container()
        .lifecycle
        .create(&request(json!({"params": {"leaky_key": SECRET}})), None)
        .await
        .unwrap_err();
    assert_eq!(codes(&e)["secret_in_argv"], "block");
    assert!(
        !Value::Object(problem(&e).extra)
            .to_string()
            .contains(SECRET)
    );
    assert!(env.node.ops("docker.run").is_empty());
}

// ---- ports -------------------------------------------------------------------------------------------------------------------
#[tokio::test]
async fn test_ports_are_allocated_per_host_from_that_hosts_range() {
    let mut env = FleetEnv::new();
    let a = launch(&mut env, json!({})).await;
    let other = env.add_host(
        "spark",
        "Spark",
        HelloOpts {
            port_range: (19000, 19010),
            ..HelloOpts::default()
        },
    );
    seed(&env, "spark");
    let b = env
        .container()
        .lifecycle
        .create(&request(json!({"host_id": "spark"})), None)
        .await
        .unwrap();
    let c = launch(&mut env, json!({"name": "c"})).await;
    assert!((a.port, c.port) == (Some(18000), Some(18001)) && b.port == Some(19000));
    assert_eq!(other.ops("gateway.start")[0]["host_port"], json!(19000));
    assert!(
        b.host_id == "spark"
            && b.host_label.as_deref() == Some("Spark")
            && b.gpu_uuids == ["GPU-uuid-0"]
    );
}

#[tokio::test]
async fn test_ports_published_by_other_containers_are_skipped() {
    let mut env = FleetEnv::new();
    env.node.add_container(
        "someone",
        ContainerOpts {
            image: "redis".into(),
            host_ports: ports(18000, 6379, "0.0.0.0"),
            ..ContainerOpts::default()
        },
    );
    env.node.add_endpoint(
        18001,
        EndpointOpts {
            container: Some("other".into()),
            ..EndpointOpts::default()
        },
    );
    let inst = launch(&mut env, json!({})).await;
    assert_eq!(inst.port, Some(18002));
}

#[tokio::test]
async fn test_gateway_port_in_use_retries_the_next_free_port() {
    let mut env = FleetEnv::new();
    env.node.st().port_taken.extend([18000, 18001]); // something listens there that the inventory does not show
    let inst = launch(&mut env, json!({})).await;
    let tried: Vec<Value> = env
        .node
        .ops("gateway.start")
        .iter()
        .map(|a| a["host_port"].clone())
        .collect();
    assert_eq!(tried, [json!(18000), json!(18001), json!(18002)]);
    assert!(
        inst.port == Some(18002)
            && env.container().lifecycle.get(&inst.id).unwrap().port == Some(18002)
    );
    let gw = container(&env.node, &format!("{}-gw", inst.container_name.unwrap())).unwrap();
    assert_eq!(gw["ports"][0]["host_port"], json!(18002));
}

#[tokio::test]
async fn test_gateway_gives_up_after_a_bounded_number_of_ports_and_leaves_no_engine_behind() {
    let mut env = FleetEnv::new();
    env.node.st().port_taken.extend(18000..18100);
    seed(&env, "local");
    let c = env.container();
    let e = c
        .lifecycle
        .create(&request(json!({})), None)
        .await
        .unwrap_err();
    let p = problem(&e);
    assert!(p.code == "port_in_use" && p.status == 409); // the node's own words, mapped to 409
    assert_eq!(env.node.ops("gateway.start").len(), 8); // bounded: not every port of the range
    assert!(
        !env.node
            .st()
            .containers
            .values()
            .any(|c| c["state"] == "running" && c["managed"] == json!(true))
    );
    assert_eq!(
        c.lifecycle.list_page(100, None, None).unwrap().items[0].state,
        "failed"
    );
}

#[tokio::test]
async fn test_a_gateway_that_fails_otherwise_tears_the_engine_down() {
    let mut env = FleetEnv::new();
    env.node.fail(
        "gateway.start",
        NodeError::new("local", "docker_error", "cannot connect the network"),
    );
    seed(&env, "local");
    let c = env.container();
    let p = problem(
        &c.lifecycle
            .create(&request(json!({})), None)
            .await
            .unwrap_err(),
    );
    assert!(p.status == 502 && p.code == "docker_error");
    assert!(
        !env.node
            .st()
            .containers
            .values()
            .any(|c| c["managed"] == json!(true))
    );
    assert_eq!(
        c.lifecycle.list_page(100, None, None).unwrap().items[0].state,
        "failed"
    );
}

#[tokio::test]
async fn test_docker_run_failure_marks_the_instance_failed_with_the_nodes_words() {
    let mut env = FleetEnv::new();
    env.node.fail(
        "docker.run",
        NodeError::new("local", "docker_error", "no such gpu"),
    );
    seed(&env, "local");
    let c = env.container();
    let p = problem(
        &c.lifecycle
            .create(&request(json!({})), None)
            .await
            .unwrap_err(),
    );
    assert!(p.status == 502 && p.detail.contains("testbox") && p.detail.contains("no such gpu"));
    let inst = c.lifecycle.list_page(100, None, None).unwrap().items[0].clone();
    assert!(inst.state == "failed" && inst.error.unwrap_or_default().contains("no such gpu"));
}

/// Security control: an engine network that exists but is not --internal is refused and nothing starts.
#[tokio::test]
async fn test_existing_non_internal_network_is_refused_and_nothing_is_started() {
    let mut env = FleetEnv::new();
    env.node.st().network_internal = false;
    seed(&env, "local");
    let e = env
        .container()
        .lifecycle
        .create(&request(json!({})), None)
        .await
        .unwrap_err();
    assert_eq!(problem(&e).status, 409);
    assert!(env.node.ops("docker.run").is_empty());
}

// ---- stop / start / remove -------------------------------------------------------------------------------------------------
/// Security control: every container action names its instance, and a gateway is touched only as a gateway.
#[tokio::test]
async fn test_stop_stops_the_gateway_first_then_the_engine_with_role_and_instance() {
    let mut env = FleetEnv::new();
    let inst = launch(&mut env, json!({})).await;
    let n0 = env.node.calls().len();
    let c = env.container();
    let stopped = c.lifecycle.stop(&inst.id).await.unwrap();
    assert_eq!(stopped.state, "stopped");
    let stops: Vec<_> = env.node.calls()[n0..]
        .iter()
        .filter(|(o, _)| o == "docker.stop")
        .map(|(_, a)| a.clone())
        .collect();
    let cn = inst.container_name.clone().unwrap();
    assert_eq!(
        stops.iter().map(|s| s["name"].clone()).collect::<Vec<_>>(),
        [json!(format!("{cn}-gw")), json!(cn)]
    );
    assert!(
        stops[0]["role"] == "gateway"
            && stops[0]["instance"] == json!(inst.id)
            && stops[1]["instance"] == json!(inst.id)
    );
    assert_eq!(c.lifecycle.get(&inst.id).unwrap().phase, None);
}

/// Security control: removal deletes both containers, the 0600 secrets file and the row.
#[tokio::test]
async fn test_remove_removes_both_containers_the_secrets_and_the_row() {
    let mut env = FleetEnv::new();
    let inst = launch(
        &mut env,
        json!({"params": {"max_model_len": 4096, "api_key": SECRET}}),
    )
    .await;
    let sf = env
        .cfg
        .data_dir
        .join("secrets")
        .join(format!("{}.json", inst.id));
    assert!(sf.exists());
    let c = env.container();
    c.lifecycle.remove(&inst.id).await.unwrap();
    let cn = inst.container_name.clone().unwrap();
    assert!(
        container(&env.node, &cn).is_none() && container(&env.node, &format!("{cn}-gw")).is_none()
    );
    assert!(!sf.exists());
    assert!(c.lifecycle.get(&inst.id).is_err());
}

#[tokio::test]
async fn test_stop_and_remove_tolerate_containers_that_are_already_gone() {
    let mut env = FleetEnv::new();
    let inst = launch(&mut env, json!({})).await;
    let cn = inst.container_name.clone().unwrap();
    remove_container(&env.node, &cn);
    remove_container(&env.node, &format!("{cn}-gw"));
    let c = env.container();
    assert_eq!(c.lifecycle.stop(&inst.id).await.unwrap().state, "stopped");
    c.lifecycle.remove(&inst.id).await.unwrap();
}

#[tokio::test]
async fn test_stop_on_an_unreachable_host_changes_nothing_and_says_503() {
    let mut env = FleetEnv::new();
    let inst = launch(&mut env, json!({})).await;
    env.node.set_state("down", None);
    let c = env.container();
    let p = problem(&c.lifecycle.stop(&inst.id).await.unwrap_err());
    assert!(p.status == 503 && p.code == "node_unavailable");
    assert_eq!(instance_row(&c, &inst.id).unwrap()["state"], "starting"); // not "failed", not "stopping"
    assert_eq!(c.lifecycle.get(&inst.id).unwrap().state, "unreachable"); // what the API shows meanwhile
    env.node.set_state("up", None);
    assert_eq!(c.lifecycle.get(&inst.id).unwrap().state, "starting");
}

#[tokio::test]
async fn test_start_recreates_the_containers_and_keeps_the_gpus() {
    let mut env = FleetEnv::new();
    let inst = launch(&mut env, json!({"gpu_ids": [1]})).await;
    let c = env.container();
    c.lifecycle.stop(&inst.id).await.unwrap();
    let again = c.lifecycle.start(&inst.id).await.unwrap();
    assert!(
        again.state == "starting"
            && again.gpu_ids == [1]
            && again.container_name == inst.container_name
    );
    let specs = run_specs(&env.node);
    assert!(specs.len() == 2 && specs[1]["gpu_uuids"] == json!(["GPU-uuid-1"]));
    let e = c.lifecycle.start(&inst.id).await.unwrap_err(); // already starting
    assert_eq!(problem(&e).status, 409);
}

/// Security control: a launch never removes an unlabelled container that wears the instance's name.
#[tokio::test]
async fn test_launch_will_not_remove_an_unlabelled_namesake() {
    let mut env = FleetEnv::new();
    let inst = launch(&mut env, json!({})).await;
    let c = env.container();
    exec(
        &c,
        "UPDATE instances SET state='stopped' WHERE id=?",
        &inst.id,
    );
    let cn = inst.container_name.clone().unwrap();
    update_container(&env.node, &cn, json!({"labels": {}, "managed": false}));
    let p = problem(&c.lifecycle.start(&inst.id).await.unwrap_err());
    assert!(p.status == 403 && p.code == "not_owned");
    assert_eq!(container(&env.node, &cn).unwrap()["state"], "running");
    assert_eq!(c.lifecycle.get(&inst.id).unwrap().state, "failed");
}

// ---- the supervisor -----------------------------------------------------------------------------------------------------------
#[tokio::test]
async fn test_supervisor_walks_starting_loading_ready_from_logs_and_the_health_probe() {
    let mut env = FleetEnv::new();
    let inst = launch(&mut env, json!({})).await;
    let c = env.container();
    set_logs(
        &env.node,
        inst.container_name.as_deref().unwrap(),
        "boot\nloading weights 55%\n",
    );
    c.lifecycle.tick().await.unwrap();
    let got = c.lifecycle.get(&inst.id).unwrap();
    assert!(
        got.state == "loading"
            && got.progress_pct == Some(55.0)
            && got.phase.as_deref() == Some("loading_weights")
    );
    env.engine_up();
    c.lifecycle.tick().await.unwrap();
    assert_eq!(c.lifecycle.get(&inst.id).unwrap().state, "ready");
    let probes: Vec<_> = env
        .node
        .http_requests()
        .into_iter()
        .filter(|r| r.path == "/health")
        .collect();
    let last = probes.last().unwrap();
    assert!(last.port == inst.port.unwrap() && last.method == "GET");
}

#[tokio::test]
async fn test_supervisor_reads_the_inventory_and_only_asks_the_node_when_it_has_not_caught_up() {
    let mut env = FleetEnv::new();
    let inst = launch(&mut env, json!({})).await;
    let c = env.container();
    let cn = inst.container_name.clone().unwrap();
    set_logs(&env.node, &cn, "x\n");
    c.lifecycle.tick().await.unwrap();
    assert!(env.node.ops("docker.inspect").is_empty()); // the container is in the inventory
    remove_container(&env.node, &cn);
    c.lifecycle.tick().await.unwrap(); // just launched: ask the node, it says gone
    assert!(!env.node.ops("docker.inspect").is_empty());
    assert_eq!(
        c.lifecycle.get(&inst.id).unwrap().error.as_deref(),
        Some("container disappeared")
    );
}

#[tokio::test]
async fn test_supervisor_trusts_a_missing_container_once_the_inventory_is_older_than_a_launch() {
    let mut env = FleetEnv::new();
    let inst = launch(&mut env, json!({})).await;
    let c = env.container();
    exec(&c, "UPDATE instances SET started_at=1 WHERE id=?", &inst.id);
    remove_container(&env.node, inst.container_name.as_deref().unwrap());
    c.lifecycle.tick().await.unwrap();
    assert!(env.node.ops("docker.inspect").is_empty());
    assert_eq!(
        c.lifecycle.get(&inst.id).unwrap().error.as_deref(),
        Some("container disappeared")
    );
}

/// Security control: a crashed engine's captured logs are scrubbed of its secret values and token shapes.
#[tokio::test]
async fn test_crash_detection_captures_the_exit_code_and_scrubbed_logs() {
    let mut env = FleetEnv::new();
    let inst = launch(
        &mut env,
        json!({"params": {"max_model_len": 4096, "api_key": SECRET}}),
    )
    .await;
    let cn = inst.container_name.clone().unwrap();
    set_logs(
        &env.node,
        &cn,
        &format!("CUDA error: out of memory\nkey {SECRET} hf_abcdefghijkl1234\n"),
    );
    update_container(&env.node, &cn, json!({"state": "exited", "exit_code": 137}));
    let c = env.container();
    c.lifecycle.tick().await.unwrap();
    let got = c.lifecycle.get(&inst.id).unwrap();
    let logs = got.last_logs.clone().unwrap_or_default();
    assert!(
        got.state == "failed"
            && got.error.unwrap_or_default().contains("code 137")
            && logs.contains("out of memory")
    );
    assert!(!logs.contains(SECRET) && !logs.contains("hf_abcdef"));
    assert!(
        env.node
            .ops("docker.stop")
            .iter()
            .any(|a| a["name"].as_str().unwrap().ends_with("-gw"))
    );
}

#[tokio::test]
async fn test_startup_timeout_and_gateway_checks() {
    let mut env = FleetEnv::new();
    let inst = launch(&mut env, json!({})).await;
    let c = env.container();
    exec(&c, "UPDATE instances SET started_at=1 WHERE id=?", &inst.id);
    c.lifecycle.tick().await.unwrap();
    assert_eq!(
        c.lifecycle.get(&inst.id).unwrap().error.as_deref(),
        Some("startup timed out")
    );
    let second = launch(&mut env, json!({"name": "two"})).await;
    let gw = format!("{}-gw", second.container_name.clone().unwrap());
    {
        let mut st = env.node.st();
        let g = st.containers.get_mut(&gw).unwrap().as_object_mut().unwrap();
        g["labels"]["engine-console.instance"] = json!("inst_someone_else");
        g.insert("instance".into(), json!("inst_someone_else"));
    }
    c.lifecycle.tick().await.unwrap();
    let got = c.lifecycle.get(&second.id).unwrap();
    assert!(got.state == "failed" && got.error.unwrap_or_default().contains("gateway"));
    assert!(
        !env.node
            .ops("docker.stop")
            .iter()
            .any(|a| a["name"] == json!(gw) && a["instance"] == "inst_someone_else")
    );
}

#[tokio::test]
async fn test_idle_ttl_stops_a_ready_instance_unless_it_is_pinned() {
    let mut env = FleetEnv::new();
    env.engine_up();
    let a = launch(&mut env, json!({})).await;
    let b = launch(&mut env, json!({"name": "b", "gpu_ids": [1]})).await;
    let c = env.container();
    for i in [&a, &b] {
        c.lifecycle.tick().await.unwrap();
        assert_eq!(c.lifecycle.get(&i.id).unwrap().state, "ready");
    }
    c.lifecycle
        .patch(
            &a.id,
            &InstancePatch {
                ttl_idle_s: Some(1),
                ..InstancePatch::default()
            },
        )
        .unwrap();
    let past = ec_hub::services::common::now() - 100.0;
    c.store
        .execute(
            "UPDATE instances SET last_request_at=? WHERE id=?",
            &[&past, &a.id],
        )
        .unwrap();
    c.lifecycle.tick().await.unwrap();
    assert!(
        c.lifecycle.get(&a.id).unwrap().state == "stopped"
            && c.lifecycle.get(&b.id).unwrap().state == "ready"
    );
}

#[tokio::test]
async fn test_a_host_that_is_down_is_not_supervised_and_nothing_is_marked_failed() {
    let mut env = FleetEnv::new();
    let inst = launch(&mut env, json!({})).await;
    env.node.set_state("down", None);
    remove_container(&env.node, inst.container_name.as_deref().unwrap());
    let c = env.container();
    exec(&c, "UPDATE instances SET started_at=1 WHERE id=?", &inst.id);
    c.lifecycle.tick().await.unwrap();
    env.node.set_state("up", None);
    assert_eq!(instance_row(&c, &inst.id).unwrap()["state"], "starting");
}

// ---- adoption ---------------------------------------------------------------------------------------------------------------------
fn good() -> Value {
    json!({"engine-console": "1", "engine-console.instance": "inst_ok", "engine-console.engine": "fake",
           "engine-console.model": "a/b", "engine-console.port": "18010", "engine-console.name": "ok",
           "engine-console.role": "engine"})
}

/// Security control (12 parametrised ids): adopted labels are attacker-influenced and validated before use.
#[tokio::test]
async fn test_adopt_skips_invalid_labels_without_raising() {
    let long_name = "x".repeat(200);
    let long_id = "i".repeat(100);
    let cases = [
        json!({"engine-console.model": "a/b;rm -rf"}),
        json!({"engine-console.model": "a--b/c"}),
        json!({"engine-console.instance": "inst x"}),
        json!({"engine-console.instance": "../x"}),
        json!({"engine-console.name": "a\nb"}),
        json!({"engine-console.name": long_name}),
        json!({"engine-console.instance": long_id}),
        json!({"engine-console.port": "22"}),
        json!({"engine-console.port": "99999"}),
        json!({"engine-console.port": "abc"}),
        json!({"engine-console.engine": "unknown"}),
        json!({"engine-console.format": "gguf"}),
    ];
    for bad in cases {
        let mut env = FleetEnv::new();
        let mut labels = good();
        for (k, v) in obj(bad.clone()) {
            labels[k] = v;
        }
        add_managed(&env.node, "ec-bad", labels, "fake/engine:1.0");
        let c = env.container();
        assert_eq!(c.lifecycle.adopt(None).await.unwrap(), 0, "{bad}");
        update_container(&env.node, "ec-bad", json!({"labels": good()}));
        assert_eq!(c.lifecycle.adopt(Some("local")).await.unwrap(), 1, "{bad}");
    }
}

#[tokio::test]
async fn test_adopt_reattaches_labelled_containers_and_is_idempotent() {
    let mut env = FleetEnv::new();
    add_managed(
        &env.node,
        "ec-old-abc123",
        json!({"engine-console.instance": "inst_old", "engine-console.engine": "fake",
               "engine-console.model": QWEN, "engine-console.port": "18007", "engine-console.name": "old",
               "engine-console.gpu_ids": "1", "engine-console.role": "engine"}),
        "fake/engine:1.0",
    );
    add_managed(
        &env.node,
        "ec-stranger",
        json!({"engine-console.role": "engine"}),
        "vllm/vllm-openai:v0.23.0",
    );
    let c = env.container();
    let life = &c.lifecycle;
    assert_eq!(life.adopt(None).await.unwrap(), 1);
    let inst = life.get("inst_old").unwrap();
    assert_eq!(
        (
            inst.state.as_str(),
            inst.port,
            inst.gpu_ids.clone(),
            inst.name.as_str(),
            inst.host_id.as_str()
        ),
        ("loading", Some(18007), vec![1], "old", "local")
    );
    assert_eq!(inst.model, Some(ModelRef::hf(QWEN)));
    assert_eq!(life.adopt(None).await.unwrap(), 0);
    update_container(&env.node, "ec-old-abc123", json!({"state": "exited"}));
    life.adopt(None).await.unwrap();
    assert_eq!(life.get("inst_old").unwrap().state, "stopped");
}

#[tokio::test]
async fn test_adopt_takes_the_real_port_from_the_gateway_and_ignores_gateways_as_instances() {
    let mut env = FleetEnv::new();
    let mut lab = good();
    lab["engine-console.instance"] = json!("inst_x");
    lab["engine-console.port"] = json!("18020");
    let mut gw = lab.clone();
    gw["engine-console.role"] = json!("gateway");
    env.node.add_container(
        "ec-x-gw",
        ContainerOpts {
            managed: true,
            labels: obj(gw),
            host_ports: ports(18033, 8080, "127.0.0.1"),
            ..ContainerOpts::default()
        },
    );
    let c = env.container();
    assert_eq!(c.lifecycle.adopt(None).await.unwrap(), 0); // a gateway alone is not an instance
    add_managed(&env.node, "ec-x", lab, "fake/engine:1.0");
    assert_eq!(c.lifecycle.adopt(None).await.unwrap(), 1);
    assert_eq!(c.lifecycle.get("inst_x").unwrap().port, Some(18033)); // a retried gateway moved off the label's port
}

#[tokio::test]
async fn test_adopt_runs_per_host_when_its_link_comes_up() {
    let mut env = FleetEnv::new();
    let spark = env.add_host("spark", "Spark", HelloOpts::default());
    let mut lab = good();
    lab["engine-console.instance"] = json!("inst_s");
    add_managed(&spark, "ec-s", lab, "fake/engine:1.0");
    spark.set_state("down", None);
    let c = env.container();
    c.lifecycle.tick().await.unwrap();
    assert!(
        c.store
            .one("SELECT 1 FROM instances WHERE id='inst_s'", &[])
            .unwrap()
            .is_none()
    );
    spark.set_state("up", None);
    c.lifecycle.tick().await.unwrap();
    let row = c
        .store
        .one("SELECT host_id FROM instances WHERE id='inst_s'", &[])
        .unwrap()
        .unwrap();
    assert_eq!(row.text("host_id").as_deref(), Some("spark"));
}

#[tokio::test]
async fn test_adopt_validates_a_gguf_models_root_and_path_labels() {
    let mut env = FleetEnv::new();
    env.adapters.push(Arc::new(kit::FakeMultiAdapter));
    let mut lab = good();
    for (k, v) in [
        ("engine-console.engine", "fakecpp"),
        ("engine-console.format", "gguf"),
        ("engine-console.model", "m-Q4"),
        ("engine-console.root", "gguf"),
        ("engine-console.rel", "a/m-Q4.gguf"),
    ] {
        lab[k] = json!(v);
    }
    add_managed(&env.node, "ec-g", lab, "fake/engine:1.0");
    let c = env.container();
    assert_eq!(c.lifecycle.adopt(None).await.unwrap(), 1);
    assert_eq!(
        c.lifecycle.get("inst_ok").unwrap().model,
        Some(gguf("a/m-Q4.gguf"))
    );
}

// ---- several machines ---------------------------------------------------------------------------------------------------------------
#[tokio::test]
async fn test_instances_of_several_hosts_are_listed_together_and_filtered_by_host() {
    let mut env = FleetEnv::new();
    let spark = env.add_host("spark", "Spark", HelloOpts::default());
    let a = launch(&mut env, json!({})).await;
    let b = launch(&mut env, json!({"host_id": "spark"})).await;
    spark.add_container(
        "vllm-ext",
        ContainerOpts {
            engine: Some("vllm".into()),
            host_ports: ports(18500, 8000, "0.0.0.0"),
            ..ContainerOpts::default()
        },
    );
    spark.add_endpoint(
        18500,
        EndpointOpts {
            container: Some("vllm-ext".into()),
            engine: Some("vllm".into()),
            models: vec!["org/big".into()],
            ..EndpointOpts::default()
        },
    );
    env.node.add_container(
        "sgl-ext",
        ContainerOpts {
            engine: Some("sglang".into()),
            image: "lmsysorg/sglang:v0.5".into(),
            host_ports: ports(18501, 30000, "0.0.0.0"),
            ..ContainerOpts::default()
        },
    );
    env.node.add_endpoint(
        18501,
        EndpointOpts {
            container: Some("sgl-ext".into()),
            engine: Some("sglang".into()),
            models: vec!["m".into()],
            ..EndpointOpts::default()
        },
    );
    let c = env.container();
    let items = c.lifecycle.list_page(100, None, None).unwrap().items;
    let mut ids: Vec<&str> = items.iter().map(|i| i.id.as_str()).collect();
    ids.sort_unstable();
    let mut want = vec![
        a.id.as_str(),
        b.id.as_str(),
        "ext-spark-vllm-ext",
        "ext-local-sgl-ext",
    ];
    want.sort_unstable();
    assert_eq!(ids, want);
    let spark_items = c
        .lifecycle
        .list_page(100, None, Some("spark"))
        .unwrap()
        .items;
    let managed: Vec<&str> = spark_items
        .iter()
        .filter(|i| i.management == "managed")
        .map(|i| i.id.as_str())
        .collect();
    assert_eq!(managed, [b.id.as_str()]);
    let mut sp: Vec<&str> = spark_items.iter().map(|i| i.id.as_str()).collect();
    sp.sort_unstable();
    let mut want_sp = vec![b.id.as_str(), "ext-spark-vllm-ext"];
    want_sp.sort_unstable();
    assert_eq!(sp, want_sp);
    let by = |id: &str| items.iter().find(|i| i.id == id).unwrap().clone();
    assert!(
        by("ext-spark-vllm-ext").host_label.as_deref() == Some("Spark")
            && by(&a.id).host_label.as_deref() == Some("testbox")
    );
}

#[tokio::test]
async fn test_a_host_that_disappears_from_the_fleet_does_not_break_listing() {
    let env = FleetEnv::new();
    env.add_host("spark", "Spark", HelloOpts::default());
    seed(&env, "spark");
    let fleet = Arc::new(wse_kit::HidingFleet::new(Arc::clone(&env.fleet)));
    let c = wse_kit::container_with_fleet(
        &env,
        Arc::clone(&fleet) as Arc<dyn ec_hub::domain::fleet::Fleet>,
    );
    let inst = c
        .lifecycle
        .create(&request(json!({"host_id": "spark"})), None)
        .await
        .unwrap();
    fleet.hide("spark"); // Python: `del env.fleet._nodes["spark"]`
    let got = c.lifecycle.get(&inst.id).unwrap();
    assert!(
        got.state == "unreachable"
            && got
                .state_reason
                .unwrap_or_default()
                .contains("not configured")
    );
    c.lifecycle.tick().await.unwrap(); // supervising a host that is gone must not raise
}

// ---- VRAM, memory and counts ----------------------------------------------------------------------------------------------------------
#[tokio::test]
async fn test_vram_and_host_memory_come_from_the_hosts_sample_for_the_instances_container() {
    let mut env = FleetEnv::new();
    let inst = launch(&mut env, json!({"gpu_ids": [1]})).await;
    let cn = inst.container_name.clone().unwrap();
    {
        let mut st = env.node.st();
        st.gpu_procs.insert(
            1,
            vec![
                json!({"pid": 11, "container": cn, "used_bytes": 40_i64 * (1 << 30), "name": "python"}),
                json!({"pid": 12, "container": "someone-else", "used_bytes": 5_i64 * (1 << 30), "name": "python"}),
            ],
        );
        st.cg.insert(cn.clone(), cgroup_default());
    }
    let c = env.container();
    let got = c.lifecycle.get(&inst.id).unwrap();
    let v: Vec<(i64, String, i64)> = got
        .vram
        .iter()
        .map(|v| {
            (
                v.gpu_index,
                v.gpu_uuid.clone(),
                v.used_gib.unwrap_or(0.0).round() as i64,
            )
        })
        .collect();
    assert_eq!(v, [(1, "GPU-uuid-1".to_string(), 40)]);
    let m = got.host_memory.unwrap();
    assert!(
        m.source == "cgroup"
            && (m.total_gib - 68.6).abs() < 1e-3
            && (m.cache_gib - 0.5).abs() < 1e-3
    );
    env.node.st().gpu_procs.insert(
        1,
        vec![json!({"pid": 11, "container": cn, "name": "python"})],
    ); // the driver cannot say
    assert_eq!(c.lifecycle.get(&inst.id).unwrap().vram[0].used_gib, None);
}

#[tokio::test]
async fn test_counts_resident_and_instances_using_a_path() {
    let mut env = FleetEnv::new();
    let inst = launch(&mut env, json!({})).await;
    let c = env.container();
    let life = &c.lifecycle;
    let n = life.counts("local");
    assert!(
        n.engines_total == 1
            && n.engines_running == 1
            && n.models == 1
            && n.models_bytes == 65_530_000_000
    );
    let cn = inst.container_name.clone().unwrap();
    env.node.st().gpu_procs.insert(
        0,
        vec![json!({"pid": 1, "container": cn, "used_bytes": 50_i64 * (1 << 30)})],
    );
    let res: Vec<(String, Vec<i64>, i64)> = life
        .resident("local")
        .into_iter()
        .map(|r| (r.name, r.gpu_ids, r.gib_per_gpu.round() as i64))
        .collect();
    assert_eq!(res, [(inst.name.clone(), vec![0], 50)]);
    assert_eq!(
        life.instances_using("local", "/data/hf/hub/models--Qwen--Qwen3-32B"),
        [inst.id.clone()]
    ); // the HF root is mounted
    assert_eq!(life.instances_using("local", "/data/hf"), [inst.id.clone()]);
    assert!(life.instances_using("local", "/elsewhere/model").is_empty());
    assert_eq!(
        life.instance_for_container("local", &cn),
        Some(inst.id.clone())
    );
    assert_eq!(life.instance_for_container("local", "nobody"), None);
}

/// Engines mount a whole HF cache; only the model they load (or a mount of the model itself) is in use.
#[tokio::test]
async fn test_a_shared_cache_mount_only_counts_for_the_model_the_engine_loads() {
    let mut env = FleetEnv::new();
    let inst = launch(&mut env, json!({})).await; // managed, loads Qwen/Qwen3-32B from /data/hf
    let other = "/data/hf/hub/models--Other--Small";
    let c = env.container();
    let life = &c.lifecycle;
    assert!(life.instances_using("local", other).is_empty()); // same cache, different model: deletable
    let root_mount =
        vec![json!({"source": "/data/hf", "dest": "/root/.cache/huggingface", "ro": true})];
    let ext = |name: &str, engine: &str, params: Value, mounts: Vec<Value>| {
        env.node.add_container(
            name,
            ContainerOpts {
                engine: Some(engine.into()),
                params: obj(params),
                mounts,
                ..ContainerOpts::default()
            },
        );
    };
    ext(
        "ext-by-id",
        "vllm",
        json!({"model": "Other/Small"}),
        root_mount.clone(),
    );
    ext(
        "ext-by-path",
        "sglang",
        json!({"model_path": "/root/.cache/huggingface/hub/models--Other--Small/snapshots/abc"}),
        root_mount.clone(),
    );
    ext(
        "ext-unrelated",
        "vllm",
        json!({"model": "Someone/Else"}),
        root_mount.clone(),
    );
    ext("ext-unknown", "vllm", json!({}), root_mount); // no model argument: cautious
    ext(
        "ext-own-mount",
        "vllm",
        json!({}),
        vec![json!({"source": other, "dest": "/models", "ro": true})],
    );
    let mut using: Vec<String> = ["ext-by-id", "ext-by-path", "ext-unknown", "ext-own-mount"]
        .iter()
        .filter_map(|n| life.instance_for_container("local", n))
        .collect();
    using.sort();
    using.dedup();
    let mut got = life.instances_using("local", other);
    got.sort();
    assert!(got == using && !got.contains(&inst.id));
    let unrelated = life
        .instance_for_container("local", "ext-unrelated")
        .unwrap();
    assert!(!got.contains(&unrelated));
}

#[tokio::test]
async fn test_fit_fits_if_stop_names_the_resident_engine_on_that_host_only() {
    let mut env = FleetEnv::new();
    let inst = launch(&mut env, json!({"gpu_ids": [0]})).await;
    env.node.st().gpu_procs.insert(
        0,
        vec![json!({"pid": 1, "container": inst.container_name, "used_bytes": 80_i64 * (1 << 30)})],
    );
    env.node.set_gpu_free(GpuFree::Each(vec![16.0, 96.0]));
    let (report, _) = env
        .container()
        .lifecycle
        .preflight_full(
            &request(json!({"gpu_ids": [0], "name": "second"})),
            None,
            None,
        )
        .await
        .unwrap();
    let fit = report.fit.unwrap();
    assert!(fit.verdict == "wont_fit" && fit.fits_if_stop == [inst.name]);
}

// ---- the API ----------------------------------------------------------------------------------------------------------------------------
#[tokio::test]
async fn test_api_create_list_get_filtered_by_host() {
    let mut env = FleetEnv::new();
    seed(&env, "local");
    env.add_host("spark", "Spark", HelloOpts::default());
    seed(&env, "spark");
    let a = env
        .post(&format!("{P}/instances"))
        .json(&json!({"engine": "fake", "repo_id": QWEN, "params": {"max_model_len": 4096}}))
        .send()
        .await;
    let b = env
        .post(&format!("{P}/instances"))
        .json(&json!({"engine": "fake", "repo_id": QWEN, "host_id": "spark", "params": {"max_model_len": 4096}}))
        .send()
        .await;
    assert!(
        a.status == 201 && b.status == 201,
        "{} {}",
        a.text(),
        b.text()
    );
    let (a, b) = (a.json(), b.json());
    assert!(a["host_id"] == "local" && b["host_id"] == "spark" && b["host_label"] == "Spark");
    let all = env.get(&format!("{P}/instances")).send().await.json();
    let mut ids: Vec<Value> = all["items"]
        .as_array()
        .unwrap()
        .iter()
        .map(|i| i["id"].clone())
        .collect();
    ids.sort_by_key(ToString::to_string);
    let mut want = vec![a["id"].clone(), b["id"].clone()];
    want.sort_by_key(ToString::to_string);
    assert_eq!(ids, want);
    let only = env
        .get(&format!("{P}/instances?host=spark"))
        .send()
        .await
        .json();
    assert_eq!(
        only["items"]
            .as_array()
            .unwrap()
            .iter()
            .map(|i| i["id"].clone())
            .collect::<Vec<_>>(),
        [b["id"].clone()]
    );
    assert_eq!(
        env.get(&format!("{P}/instances?host=Bad%20Host!"))
            .send()
            .await
            .status,
        422
    );
    let nope = env
        .post(&format!("{P}/instances"))
        .json(&json!({"engine": "fake", "repo_id": QWEN, "host_id": "nope"}))
        .send()
        .await;
    assert_eq!(nope.status, 404);
}

#[tokio::test]
async fn test_api_preflight_and_unknown_host_problem_codes() {
    let mut env = FleetEnv::new();
    seed(&env, "local");
    let r = env
        .post(&format!("{P}/instances/preflight"))
        .json(&json!({"engine": "fake", "repo_id": QWEN, "host_id": "nope"}))
        .send()
        .await;
    assert!(r.status == 404 && r.code() == "unknown_host");
    env.node.set_state("down", None);
    let pf = env
        .post(&format!("{P}/instances/preflight"))
        .json(&json!({"engine": "fake", "repo_id": QWEN}))
        .send()
        .await
        .json();
    assert!(pf["ok"] == json!(false) && pf["checks"][0]["code"] == "host_down");
}

/// Security control: "copy as command" never holds a secret value and uses the node's own rendering.
#[tokio::test]
async fn test_api_command_snippets_hold_no_secret_and_use_the_nodes_rendering() {
    let mut env = FleetEnv::new();
    seed(&env, "local");
    let inst = env
        .post(&format!("{P}/instances"))
        .json(&json!({"engine": "fake", "repo_id": QWEN, "params": {"max_model_len": 4096, "api_key": SECRET}}))
        .send()
        .await
        .json();
    let cmd = env
        .get(&format!(
            "{P}/instances/{}/command",
            inst["id"].as_str().unwrap()
        ))
        .send()
        .await;
    assert_eq!(cmd.status, 200, "{}", cmd.text());
    let body = cmd.json();
    let run = body["docker_run"].as_str().unwrap();
    assert!(run.starts_with("docker run -d --name ec-") && run.contains("--cap-drop ALL"));
    assert!(
        run.contains("-e FAKE_API_KEY")
            && !cmd.text().contains(SECRET)
            && !cmd.text().contains("s3cr3t")
    );
    let compose = body["compose_yaml"].as_str().unwrap();
    assert!(compose.contains("${FAKE_API_KEY}") && compose.contains("GPU-uuid-0"));
    assert!(
        compose.contains("/data/hf/hub:/root/.cache/huggingface/hub:ro")
            && body["engine_cli"].as_str().unwrap().contains("--model")
    );
    let render = env.node.ops("docker.render")[0]["spec"].clone();
    assert!(!render.to_string().contains(SECRET));
    assert_eq!(
        render["secret_env"]
            .as_object()
            .unwrap()
            .keys()
            .collect::<Vec<_>>(),
        ["FAKE_API_KEY"]
    );
}

/// Security control: logs (one-shot and followed) are scrubbed of the instance's secret values and token shapes.
#[tokio::test]
async fn test_api_logs_follow_stream_and_scrub() {
    let mut env = FleetEnv::new();
    seed(&env, "local");
    let inst = env
        .post(&format!("{P}/instances"))
        .json(&json!({"engine": "fake", "repo_id": QWEN, "params": {"max_model_len": 4096, "api_key": SECRET}}))
        .send()
        .await
        .json();
    let id = inst["id"].as_str().unwrap();
    set_logs(
        &env.node,
        inst["container_name"].as_str().unwrap(),
        &format!("args: api_key={SECRET} token hf_abcdefghijkl1234\nsecond line\n"),
    );
    let logs = env
        .get(&format!("{P}/instances/{id}/logs"))
        .send()
        .await
        .text();
    assert!(
        !logs.contains(SECRET)
            && !logs.contains("hf_abcdef")
            && logs.contains("[redacted]")
            && logs.contains("second line")
    );
    let once = env
        .get(&format!("{P}/instances/{id}/logs/stream?once=true"))
        .send()
        .await
        .text();
    assert!(once.contains("event: log") && !once.contains(SECRET));
    let live = env
        .get(&format!("{P}/instances/{id}/logs/stream"))
        .send()
        .await
        .text();
    assert!(live.matches("event: log").count() == 2 && !live.contains(SECRET));
}

#[tokio::test]
async fn test_node_errors_become_problems_with_the_hosts_label_and_the_nodes_code() {
    let mut env = FleetEnv::new();
    seed(&env, "local");
    env.node.fail(
        "docker.run",
        NodeError::new("local", "docker_error", "boom"),
    );
    let r = env
        .post(&format!("{P}/instances"))
        .json(&json!({"engine": "fake", "repo_id": QWEN}))
        .send()
        .await;
    assert!(r.status == 502 && r.code() == "docker_error" && r.json()["detail"] == "testbox: boom");
    assert!(
        r.header("content-type")
            .unwrap()
            .starts_with("application/problem+json")
    );
}

#[tokio::test]
async fn test_unserved_node_error_is_still_a_problem_through_the_global_handler() {
    let mut env = FleetEnv::new();
    env.node.set_state("down", Some("connection refused"));
    let r = env.get(&format!("{P}/hardware")).send().await;
    assert!(
        r.status == 503
            && r.code() == "node_unavailable"
            && r.json()["detail"]
                .as_str()
                .unwrap()
                .contains("connection refused")
    );
    assert!(
        r.header("content-type")
            .unwrap()
            .starts_with("application/problem+json")
    );
}

/// The fake enforces what the real node's whitelist cleaning would do: a typo is a failure, not a silent drop.
#[tokio::test]
async fn test_fake_node_itself_rejects_dropped_arguments() {
    use ec_hub::domain::fleet::NodeClient;
    let env = FleetEnv::new();
    let e = env
        .node
        .call(
            "docker.stop",
            Some(obj(json!({"name": "x", "typo_field": 1}))),
            std::time::Duration::from_secs(5),
        )
        .await
        .unwrap_err();
    assert_eq!(e.code, "bad_args");
    env.node.set_mode("monitor");
    let e2 = env
        .node
        .call(
            "docker.rm",
            Some(obj(json!({"name": "x"}))),
            std::time::Duration::from_secs(5),
        )
        .await
        .unwrap_err();
    assert_eq!(e2.code, "monitor_only");
}

/// Security control: adoption only takes ids that look like ours and survives a clash.
#[tokio::test]
async fn test_adoption_only_takes_ids_that_look_like_ours_and_survives_a_clash() {
    let mut env = FleetEnv::new();
    let base = json!({"engine-console.engine": "fake", "engine-console.model": "a/b", "engine-console.port": "18010",
                      "engine-console.role": "engine"});
    let with = |iid: &str, name: &str| {
        let mut l = base.clone();
        l["engine-console.instance"] = json!(iid);
        l["engine-console.name"] = json!(name);
        l
    };
    // ids that would shadow derived instances in the API, or are simply not ours
    for (i, bad) in ["ext-evil", "scr-evil", "evil", "not_inst_1"]
        .iter()
        .enumerate()
    {
        add_managed(
            &env.node,
            &format!("ec-bad{i}"),
            with(bad, &format!("n{i}")),
            "fake/engine:1.0",
        );
    }
    let c = env.container();
    assert_eq!(c.lifecycle.adopt(None).await.unwrap(), 0);
    // a container that claims the id of an instance that already exists does not abort the rest
    let existing = launch(&mut env, json!({})).await;
    add_managed(
        &env.node,
        "ec-clash",
        with(&existing.id, "clash"),
        "fake/engine:1.0",
    );
    add_managed(
        &env.node,
        "ec-fine",
        with("inst_fine", "fine"),
        "fake/engine:1.0",
    );
    assert_eq!(c.lifecycle.adopt(None).await.unwrap(), 1);
    assert_eq!(c.lifecycle.get("inst_fine").unwrap().host_id, "local");
    assert_eq!(
        c.lifecycle.get(&existing.id).unwrap().container_name,
        existing.container_name
    );
}

#[tokio::test]
async fn test_a_script_job_that_never_ends_is_failed_at_the_upper_bound() {
    let mut env = FleetEnv::new();
    env.node
        .set_policy("scripts", json!([fakefleet::big_model_script()]));
    let gate = fakefleet::Gate::new(); // never released
    env.node.script_run(
        "big-model",
        "up",
        fakefleet::ScriptRun {
            gate: Some(gate),
            ..fakefleet::ScriptRun::lines(&[("out", "started")])
        },
    );
    let c = env.container();
    c.scripts.set_job_max_s(0.05);
    let job = c
        .scripts
        .run("local", "big-model", "up", None)
        .await
        .unwrap();
    c.scripts.wait(&job.id).await;
    let done = c.scripts.job(&job.id).unwrap();
    assert!(done.state == "failed" && done.output_tail.contains("[timeout]"));
    assert!(
        env.node.streams().last().unwrap().is_cancelled() && c.scripts.busy("local").is_empty()
    );
}

#[tokio::test]
async fn test_two_requests_to_start_the_same_instance_launch_it_once() {
    let mut env = FleetEnv::new();
    let inst = launch(&mut env, json!({})).await;
    let c = env.container();
    c.lifecycle.stop(&inst.id).await.unwrap();
    let n = run_specs(&env.node).len();
    let (r1, r2) = tokio::join!(c.lifecycle.start(&inst.id), c.lifecycle.start(&inst.id));
    let results = [r1, r2];
    assert_eq!(results.iter().filter(|r| r.is_ok()).count(), 1);
    assert_eq!(
        results
            .iter()
            .filter(|r| r.as_ref().is_err_and(|e| e.status() == 409))
            .count(),
        1
    );
    assert_eq!(run_specs(&env.node).len(), n + 1); // one launch, not two
}

#[tokio::test]
async fn test_a_stop_whose_request_went_away_does_not_leave_the_instance_stopping() {
    let mut env = FleetEnv::new();
    let inst = launch(&mut env, json!({})).await;
    let c = env.container();
    c.lifecycle.sync_links().await; // the first look at a host adopts what runs there
    exec(
        &c,
        "UPDATE instances SET state='stopping' WHERE id=?",
        &inst.id,
    );
    c.lifecycle.tick().await.unwrap();
    assert_eq!(c.lifecycle.get(&inst.id).unwrap().state, "stopping"); // the container still runs: a stop may be under way
    let cn = inst.container_name.clone().unwrap();
    update_container(&env.node, &cn, json!({"state": "exited", "exit_code": 0}));
    update_container(
        &env.node,
        &format!("{cn}-gw"),
        json!({"state": "exited", "exit_code": 0}),
    );
    c.lifecycle.tick().await.unwrap();
    assert_eq!(c.lifecycle.get(&inst.id).unwrap().state, "stopped");
}

#[tokio::test]
async fn test_restart_waits_for_the_hosts_next_samples_before_the_fit_check() {
    let mut env = FleetEnv::with(|c| c.restart_settle_s = 0.05);
    let inst = launch(&mut env, json!({})).await;
    let t0 = std::time::Instant::now();
    let again = env.container().lifecycle.restart(&inst.id).await.unwrap();
    assert!(again.state == "starting" && t0.elapsed() >= std::time::Duration::from_millis(50));
}

#[tokio::test]
async fn test_probe_reads_through_the_nodes_http_request() {
    // the readiness probe is an http.request through the host's node (never a direct connection)
    let mut env = FleetEnv::new();
    env.engine_up();
    let inst = launch(&mut env, json!({})).await;
    let c = env.container();
    c.lifecycle.tick().await.unwrap();
    assert_eq!(c.lifecycle.get(&inst.id).unwrap().state, "ready");
    let r = c
        .nodehttp
        .request(
            "local",
            inst.port.unwrap(),
            "GET",
            "/health",
            HttpOpts::default(),
        )
        .await
        .unwrap();
    assert_eq!(r.status, 200);
}
