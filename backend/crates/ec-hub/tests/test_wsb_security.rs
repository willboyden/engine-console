//! Fleet security controls, end to end: the hub's NodeLink driving the real Rust `ecnode` (backend/target/debug/ecnode,
//! built by `cargo test --workspace`) with a monitor-mode policy. A monitor-mode node must refuse every operation that
//! would change the machine, the hub must relay that refusal as `monitor_only` (409 over the API) without retrying
//! or rewriting it, and monitor operations must still work. The node gets an empty PATH: it can find neither docker
//! nor nvidia-smi, so nothing on this machine is touched. (No Python counterpart: PORTING.md §12 asks
//! for an explicit test of every security control.)

#![allow(
    clippy::needless_pass_by_value,
    clippy::expect_used,
    clippy::panic,
    clippy::unwrap_used,
    clippy::too_many_lines
)]

use std::os::unix::fs::PermissionsExt as _;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ec_hub::domain::fleet::{HostConfig, NodeClient};
use ec_hub::services::common::EventBus;
use ec_hub::services::fleet::{LinkLike, LinkTiming, NodeLink};
use ec_hub::services::nodeerr::status_of;
use serde_json::{Value, json};

/// target/debug/ecnode, next to this test's target/debug/deps directory.
fn ecnode() -> PathBuf {
    let exe = std::env::current_exe().unwrap();
    let p = exe
        .parent()
        .and_then(|d| d.parent())
        .unwrap()
        .join("ecnode");
    assert!(
        p.is_file(),
        "{} is missing: build it first (cargo build -p ecnode, or run cargo test --workspace)",
        p.display()
    );
    p
}

fn obj(v: Value) -> ec_hub::domain::errors::Obj {
    v.as_object().cloned().unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_a_monitor_mode_node_refuses_every_op_that_would_change_the_machine() {
    let d = tempfile::tempdir().unwrap();
    std::fs::set_permissions(d.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
    let (home, bin, models) = (
        d.path().join("home"),
        d.path().join("empty-bin"),
        d.path().join("models"),
    );
    for p in [&home, &bin, &models] {
        std::fs::create_dir_all(p).unwrap();
    }
    let policy = d.path().join("policy.json");
    std::fs::write(
        &policy,
        // hermetic (PORTING.md §A2): no discovery probes (the defaults include 8000, 11434 and 30000) and no docker
        json!({"version": 1, "mode": "monitor", "engine_images": ["vllm/vllm-openai:*"],
        "discovery_ports": [], "docker": {"binary": "/nonexistent/docker"},
        "model_roots": [{"id": "m", "path": models.display().to_string(), "kind": "plain"}]})
        .to_string(),
    )
    .unwrap();
    std::fs::set_permissions(&policy, std::fs::Permissions::from_mode(0o600)).unwrap();
    let env = vec![
        ("HOME".to_string(), home.display().to_string()),
        ("PATH".to_string(), bin.display().to_string()),
        // the Rust node finds its tools only in ECNODE_TOOL_PATH (or the system PATH without it, where the real
        // nvidia-smi and ip live): an empty directory, so no tool at all
        ("ECNODE_TOOL_PATH".to_string(), bin.display().to_string()),
        ("LANG".to_string(), "C.UTF-8".to_string()),
        // the node's GPUs come from nvidia-smi among its tools (none here), never the real NVML
        ("ECNODE_NO_NVML".to_string(), "1".to_string()),
    ];
    let host = HostConfig {
        id: "local".into(),
        label: "This machine".into(),
        ssh: None,
        port: None,
        ssh_config: false,
        enabled: true,
    };
    let argv = vec![
        ecnode().display().to_string(),
        "--stdio".into(),
        "--policy".into(),
        policy.display().to_string(),
    ];
    let link = Arc::new(NodeLink::new(
        host,
        NodeLink::fixed_argv(argv),
        Some(env),
        Arc::new(EventBus::new()),
        LinkTiming::default(),
        None,
        Vec::new(),
    ));
    link.start().await;
    let end = Instant::now() + Duration::from_secs(20);
    while link.link().state != "up" {
        assert!(
            Instant::now() < end,
            "the node never came up: {:?}",
            link.link()
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let hello = link.latest("hello").unwrap();
    assert_eq!(hello["policy"]["mode"], "monitor");

    let t = Duration::from_secs(10);
    let spec = json!({"name": "ec-x", "image": "vllm/vllm-openai:v1", "argv": [], "network_alias": "ec-x"});
    let refused: Vec<(&str, Value)> = vec![
        ("docker.run", json!({"spec": spec})),
        ("docker.stop", json!({"name": "ec-x"})),
        ("docker.start", json!({"name": "ec-x"})),
        ("docker.rm", json!({"name": "ec-x"})),
        ("docker.pull", json!({"image": "vllm/vllm-openai:v1"})),
        (
            "docker.external",
            json!({"name": "other", "action": "stop"}),
        ),
        ("net.ensure_internal", json!({})),
        (
            "gateway.start",
            json!({"name": "ec-x-gw", "instance": "inst_x", "host_port": 18000, "engine_host": "ec-x", "engine_port": 8000}),
        ),
        (
            "fs.symlink",
            json!({"root": "m", "link": "a", "target": "b"}),
        ),
        (
            "fs.put_small",
            json!({"root": "m", "rel": "a", "content_b64": "eA=="}),
        ),
        ("fs.delete", json!({"root": "m", "rel": "a"})),
    ];
    for (op, args) in refused {
        let e = link.call(op, Some(obj(args)), t).await.unwrap_err();
        assert_eq!(e.code, "monitor_only", "{op}: {e}");
        assert!(
            e.message.contains("monitor mode") && e.message.contains(op),
            "{op}: {}",
            e.message
        );
        assert_eq!(status_of(&e.code), 409);
    }
    // the streaming ones refuse the same way, before any upload
    for (op, args) in [
        ("fs.write", json!({"root": "m", "rel": "a", "size": 1})),
        ("scripts.run", json!({"name": "s", "action": "up"})),
    ] {
        let mut s = link.open(op, Some(obj(args)));
        while s.next().await.is_some() {}
        assert_eq!(s.result().await.unwrap_err().code, "monitor_only", "{op}");
    }
    assert!(!models.join("a").exists());
    // monitor operations still work
    let info = link.call("node.info", None, t).await.unwrap();
    assert_eq!(info["policy"]["mode"], "monitor");
    // the monitor gate lets read-only ops through (what the handler then answers is the node's business)
    for (op, args) in [
        ("fs.stat", json!({"root": "m", "paths": ["a"]})),
        ("docker.images", json!({})),
        ("models.list", json!({})),
        ("docker.inspect", json!({"name": "ec-x"})),
    ] {
        if let Err(e) = link.call(op, Some(obj(args)), t).await {
            assert_ne!(e.code, "monitor_only", "{op}: {e}");
        }
    }
    assert_eq!(link.inflight(), 0);
    link.stop().await;
    assert_eq!(link.link().state, "down");
}

/// The hub never sends an operation the protocol does not define, whatever a caller asks for.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_the_hub_refuses_an_op_outside_the_protocol_before_sending_it() {
    let host = HostConfig {
        id: "local".into(),
        label: "This machine".into(),
        ssh: None,
        port: None,
        ssh_config: false,
        enabled: true,
    };
    let link = NodeLink::new(
        host,
        NodeLink::fixed_argv(vec!["/nonexistent".into()]),
        Some(Vec::new()),
        Arc::new(EventBus::new()),
        LinkTiming::default(),
        None,
        Vec::new(),
    );
    for op in ["shell", "docker.exec", "fs.chmod", ""] {
        let e = link
            .call(op, None, Duration::from_secs(1))
            .await
            .unwrap_err();
        assert_eq!(e.code, "unknown_op", "{op}");
        assert!(
            e.message
                .contains("is not an operation of the node protocol")
        );
    }
}
