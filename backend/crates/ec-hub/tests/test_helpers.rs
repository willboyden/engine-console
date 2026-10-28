//! Port of backend/tests/core/test_helpers.py, the nodeview rows. The docker.py and discovery.py rows (snippets,
//! gateway name, text hygiene, generic metrics) are with the lifecycle tests (backend/TEST-MAP.md).

#[path = "common/kit.rs"]
mod kit;

use std::sync::Arc;

use ec_hub::domain::errors::Obj;
use ec_hub::services::nodeview as nv;
use kit::fakefleet::{ContainerOpts, EndpointOpts, ports};
use kit::{FakeNode, HelloOpts};
use serde_json::{Value, json};

/// `FakeNode(HostConfig(id="x", label="X"))`
fn node() -> Arc<FakeNode> {
    FakeNode::new(FakeNode::host_config("x", "X", None), HelloOpts::default())
}

fn obj(v: Value) -> Obj {
    match v {
        Value::Object(o) => o,
        _ => Obj::new(),
    }
}

/// Edit the node's hello in place (Python's `node.hello[...] = ...` / `node.policy.update(...)`).
fn edit_hello(n: &FakeNode, f: impl FnOnce(&mut Obj)) {
    f(&mut n.st().hello);
}

fn edit_policy(n: &FakeNode, f: impl FnOnce(&mut Obj)) {
    edit_hello(n, |h| {
        if let Some(Value::Object(p)) = h.get_mut("policy") {
            f(p);
        }
    });
}

#[test]
fn test_the_policys_image_rule_is_a_literal_repository_never_a_prefix() {
    let a64 = format!("vllm/vllm-openai@sha256:{}", "a".repeat(64));
    let upper64 = format!("vllm/vllm-openai@sha256:{}", "A".repeat(64));
    let cases: &[(&str, &[&str], bool)] = &[
        ("vllm/vllm-openai:v0.23.0", &["vllm/vllm-openai:*"], true),
        (
            "vllm/vllm-openai:v0.23.0",
            &["vllm/vllm-openai:v0.23.0"],
            true,
        ),
        (
            "vllm/vllm-openai:v0.24.0",
            &["vllm/vllm-openai:v0.23.0"],
            false,
        ),
        ("vllm/vllm-openai:evil/x", &["vllm/vllm-openai:*"], false),
        ("vllm/vllm-openai-evil:1", &["vllm/vllm-openai:*"], false),
        ("other/vllm/vllm-openai:1", &["vllm/vllm-openai:*"], false),
        ("vllm/vllm-openai:1:2", &["vllm/vllm-openai:*"], false),
        ("vllm/vllm-openai:", &["vllm/vllm-openai:*"], false),
        (&a64, &["vllm/vllm-openai@sha256:*"], true),
        (&upper64, &["vllm/vllm-openai@sha256:*"], false),
        (
            "vllm/vllm-openai@sha256:abc",
            &["vllm/vllm-openai@sha256:*"],
            false,
        ),
        ("vllm/vllm-openai:1", &["vllm/vllm-openai@sha256:*"], false),
        ("a/b:1", &["*:*"], false),
        ("a/b:1", &["a/*:*"], false),
        ("localhost:5000/eng:1", &["localhost:5000/eng:*"], true),
        ("a/b:1", &[], false),
        ("nginx:1", &["nginx@sha256:*"], false),
    ];
    for (i, (image, patterns, ok)) in cases.iter().enumerate() {
        let patterns: Vec<String> = patterns.iter().map(|p| (*p).to_string()).collect();
        assert_eq!(
            nv::image_allowed(image, &patterns),
            *ok,
            "patterns{i}: {image} {patterns:?}"
        );
    }
}

#[test]
fn test_policy_readers_tolerate_a_node_that_said_nothing_or_nonsense() {
    let n = node();
    assert_eq!(
        (
            nv::mode(&*n),
            nv::arch(&*n).as_str(),
            nv::port_range(&*n, (1, 2))
        ),
        ("manage", "x86_64", (18000, 18099))
    );
    edit_policy(&n, |p| {
        p.insert("mode".into(), json!("weird"));
        p.insert("port_range".into(), json!([5, 3]));
        p.insert("external_control".into(), Value::Null);
    });
    assert_eq!(
        (
            nv::mode(&*n),
            nv::port_range(&*n, (1, 2)),
            nv::external_control(&*n).as_str()
        ),
        ("monitor", (1, 2), "none")
    );
    edit_policy(&n, |p| {
        p.insert("port_range".into(), json!([70000, 70001]));
    });
    assert_eq!(nv::port_range(&*n, (1, 2)), (1, 2));
    edit_hello(&n, |h| {
        if let Some(Value::Object(host)) = h.get_mut("host") {
            host.insert("arch".into(), Value::Null);
        }
    });
    assert_eq!(nv::arch(&*n), "x86_64");
}

#[test]
fn test_host_ports_only_count_what_is_reachable_on_loopback() {
    let c = obj(json!({"ports": [
        {"host_ip": "0.0.0.0", "host_port": 1}, {"host_ip": "127.0.0.1", "host_port": 2},
        {"host_ip": "10.0.0.5", "host_port": 3}, {"host_ip": "", "host_port": 4}, {"host_ip": "::", "host_port": 5},
        {"host_ip": "0.0.0.0", "host_port": 1}, "junk", {"host_ip": "0.0.0.0", "host_port": "7"}]}));
    assert_eq!(nv::host_ports(&c), vec![1, 2, 4, 5]);
}

#[test]
fn test_ports_in_use_combines_containers_and_endpoints() {
    let n = node();
    n.add_container(
        "a",
        ContainerOpts {
            host_ports: ports(18000, 8000, "0.0.0.0"),
            ..ContainerOpts::default()
        },
    );
    n.add_endpoint(18005, EndpointOpts::default());
    assert_eq!(nv::ports_in_use(&*n), vec![18000, 18005]);
}

#[test]
fn test_numbers_from_untrusted_data() {
    let v = |j: Value| j;
    assert_eq!(nv::num(Some(&v(json!(true)))), None);
    assert_eq!(nv::num(Some(&v(json!("1")))), None);
    assert_eq!(nv::num(None), None); // JSON has no NaN: a missing value stands in for float("nan")
    assert_eq!(nv::num(Some(&v(json!(3)))), Some(3.0));
    assert_eq!(
        nv::gib(Some(&v(json!(2_i64 * 1024 * 1024 * 1024)))),
        Some(2.0)
    );
    assert_eq!(nv::gib(None), None);
    assert!(nv::ok_status(Some(&json!(200))) && nv::ok_status(Some(&json!(299))));
    assert!(
        !nv::ok_status(Some(&json!(300)))
            && !nv::ok_status(Some(&json!(true)))
            && !nv::ok_status(None)
    );
    assert!(nv::auth_status(Some(&json!(401))) && nv::auth_status(Some(&json!(403))));
    assert!(!nv::auth_status(Some(&json!(404))));
}
