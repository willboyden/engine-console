//! The hub's policy validator (ec-engines' policy_check.rs) against the node's real one: for every case the
//! hub's `check_policy_text(text)` and `ecnode::policy::parse(&json, None, None)` must agree on Ok/Err, with the SAME
//! error string on Err and the same outline on Ok. The size cap and JSON syntax errors are compared with the node's
//! `load_with` on a private temp file (its "policy file <path> ...; running in monitor mode" wording around the
//! hub's "the policy ..."). Paths in the cases do not exist on a normal machine, so the node's realpath of a root is
//! its normpath and its answer does not depend on the machine running the test.
#![allow(clippy::panic, clippy::expect_used)] // test helpers outside #[test] fns

use std::os::unix::fs::PermissionsExt;

use ec_hub::services::policy_check::{
    ACTION_KEYS, DOCKER_KEYS, MAX_POLICY_BYTES, PolicyOutline, ROOT_KEYS, SCRIPT_KEYS, TOP_KEYS,
    check_policy_text,
};
use ecnode::policy::{self, LoadOptions, Policy};
use serde_json::{Value, json};

const MONITOR_EXAMPLE: &str = include_str!("../../../../node/examples/policy.monitor.json");
const MANAGE_EXAMPLE: &str = include_str!("../../../../node/examples/policy.manage.json");
const DIGEST: &str = "62ff2089abf5a9ed33bd232895bef5e22f7bb4b200675cec49a5ebc48e3d4ac8";

/// Every JSON type that a field might wrongly be.
fn wrong_types() -> Vec<Value> {
    vec![
        Value::Null,
        json!(true),
        json!(1),
        json!(1.5),
        json!("x"),
        json!([]),
        json!({}),
    ]
}

/// A valid manage-mode policy that sets every key, so removing or retyping any one of them is a real change.
fn base() -> Value {
    json!({
        "_comment": "differential base",
        "version": 1,
        "mode": "manage",
        "docker": {"binary": "/usr/bin/docker", "context": "rootless"},
        "engine_images": ["vllm/vllm-openai:*", "ghcr.io/ggml-org/llama.cpp@sha256:*", "lmsysorg/sglang:v0.5.0"],
        "gateway_image": format!("nginx@sha256:{DIGEST}"),
        "engine_network": "engine-console-engines",
        "port_range": [18000, 18099],
        "discovery_ports": [8000, 30000],
        "model_roots": [
            {"id": "hf", "path": "/data/hf/hub", "kind": "hf-hub", "writable": true, "label": "HF cache"},
            {"id": "gguf", "path": "/mnt/models-x/gguf", "kind": "gguf", "writable": false, "label": "GGUF"}
        ],
        "scripts": [
            {"name": "big-model", "label": "Big model", "engine": "sglang", "description": "Starts first.",
             "containers": ["myproj-sglang-*"], "status_action": "status", "timeout_s": 900,
             "actions": [
                {"name": "up", "label": "Start", "argv": ["/opt/lab/model.sh", "up"], "args": [], "confirm": true,
                 "danger": false},
                {"name": "status", "argv": ["/opt/lab/model.sh", "status"]},
                {"name": "switch", "argv": ["/opt/lab/use-model.sh"], "args": ["model-a", "model-b"]}
             ]}
        ],
        "allow_pull": true,
        "allow_delete": false,
        "allow_ipc_host": true,
        "external_control": "stop_start",
        "max_shm_gib": 64
    })
}

/// `base` with the value at JSON pointer `ptr` (which must exist) replaced.
fn set(b: &Value, ptr: &str, v: Value) -> Value {
    let mut out = b.clone();
    *out.pointer_mut(ptr)
        .unwrap_or_else(|| panic!("no {ptr} in the base")) = v;
    out
}

/// `base` with `key` put into (or replaced in) the object at `parent`.
fn put(b: &Value, parent: &str, key: &str, v: Value) -> Value {
    let mut out = b.clone();
    out.pointer_mut(parent)
        .and_then(Value::as_object_mut)
        .unwrap_or_else(|| panic!("no object at {parent:?}"))
        .insert(key.to_string(), v);
    out
}

/// `base` without `key` in the object at `parent`.
fn without(b: &Value, parent: &str, key: &str) -> Value {
    let mut out = b.clone();
    out.pointer_mut(parent)
        .and_then(Value::as_object_mut)
        .unwrap_or_else(|| panic!("no object at {parent:?}"))
        .shift_remove(key);
    out
}

/// One minimal valid root / script / action, for the "too many" and corner cases.
fn root(id: &str, path: &str) -> Value {
    json!({"id": id, "path": path})
}

fn action(name: &str) -> Value {
    json!({"name": name, "argv": ["/opt/lab/x.sh", name]})
}

fn outline_of(p: &Policy) -> PolicyOutline {
    PolicyOutline {
        mode: p.mode.clone(),
        engine_images: p.engine_images.len(),
        model_roots: p.model_roots.len(),
        writable_roots: p.model_roots.iter().filter(|r| r.writable).count(),
        scripts: p.scripts.len(),
        allow_pull: p.allow_pull,
        allow_delete: p.allow_delete,
        allow_ipc_host: p.allow_ipc_host,
        external_control: p.external_control.clone(),
    }
}

/// The node's error for a policy FILE holding `text` (the size cap and JSON decoding live in `load_with`).
fn node_file_error(text: &str) -> Option<String> {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("policy.json");
    std::fs::write(&path, text).expect("write policy");
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).expect("chmod");
    let p = path.to_str().expect("utf-8 temp path");
    let home = dir.path().to_str().expect("utf-8 temp dir");
    let opts = LoadOptions::new(Some(p), Some(home), "/nonexistent/ecnode/policy.json");
    let (pol, _) = policy::load_with(&opts);
    pol.error
        .map(|e| e.replace(&format!("policy file {p}"), "policy file <path>"))
}

#[derive(Default)]
struct Run {
    cases: usize,
    ok: usize,
    err: usize,
    failures: Vec<String>,
}

impl Run {
    /// JSON text the node can decode: hub vs `parse(&value, None, None)`.
    fn text(&mut self, label: &str, text: &str) {
        self.cases += 1;
        let mine = check_policy_text(text);
        let value: Value = match serde_json::from_str(text) {
            Ok(v) => v,
            Err(e) => {
                self.failures.push(format!(
                    "{label}: the case is not JSON ({e}); use file_text"
                ));
                return;
            }
        };
        let node = policy::parse(&value, None, None).map_err(|e| e.0);
        match (mine, node) {
            (Ok(o), Ok(p)) => {
                self.ok += 1;
                if o != outline_of(&p) {
                    self.failures.push(format!(
                        "{label}: outlines differ\n  hub:  {o:?}\n  node: {:?}",
                        outline_of(&p)
                    ));
                }
            }
            (Err(a), Err(b)) => {
                self.err += 1;
                if a != b {
                    self.failures.push(format!(
                        "{label}: messages differ\n  hub:  {a}\n  node: {b}"
                    ));
                }
            }
            (a, b) => self.failures.push(format!(
                "{label}: hub {a:?}\n  node {:?}",
                b.map(|p| outline_of(&p))
            )),
        }
    }

    fn value(&mut self, label: &str, v: &Value) {
        let text = serde_json::to_string(v).expect("serialise case");
        self.text(label, &text);
    }

    /// Text refused before validation (size cap, JSON syntax): hub vs the node loading it from a file.
    fn file_text(&mut self, label: &str, text: &str) {
        self.cases += 1;
        let node = node_file_error(text);
        match check_policy_text(text) {
            Err(m) => {
                self.err += 1;
                let Some(why) = m.strip_prefix("the policy ") else {
                    self.failures
                        .push(format!("{label}: hub message {m:?} lacks 'the policy '"));
                    return;
                };
                let want = format!("policy file <path> {why}; running in monitor mode");
                if node.as_deref() != Some(want.as_str()) {
                    self.failures
                        .push(format!("{label}:\n  hub:  {want}\n  node: {node:?}"));
                }
            }
            Ok(o) => self
                .failures
                .push(format!("{label}: hub accepted ({o:?}), node said {node:?}")),
        }
    }

    fn finish(self, min_ok: usize, min_err: usize) -> usize {
        assert!(
            self.failures.is_empty(),
            "{} of {} cases disagree:\n{}",
            self.failures.len(),
            self.cases,
            self.failures.join("\n")
        );
        assert!(self.ok >= min_ok, "only {} accepted cases", self.ok);
        assert!(self.err >= min_err, "only {} refused cases", self.err);
        self.cases
    }
}

#[test]
fn shipped_examples_and_e2e_shape_agree() {
    let mut run = Run::default();
    run.text("examples/policy.monitor.json", MONITOR_EXAMPLE);
    run.text("examples/policy.manage.json", MANAGE_EXAMPLE);
    // frontend/e2e/run.mjs monitorPolicy(): monitor mode, no docker, no probes, two seeded roots
    run.value(
        "e2e monitor policy",
        &json!({
            "_comment": "e2e: monitor mode, no docker, no discovery probes",
            "version": 1, "mode": "monitor", "docker": {"binary": "/nonexistent/e2e/docker"}, "discovery_ports": [],
            "model_roots": [
                {"id": "e2e-hf", "path": "/data/e2e/models/hf/hub", "kind": "hf-hub", "label": "e2e Hugging Face cache"},
                {"id": "e2e-plain", "path": "/data/e2e/models/plain", "kind": "plain", "label": "e2e checkpoints"}
            ]
        }),
    );
    run.value("empty object", &json!({}));
    run.value("base", &base());
    let n = run.finish(5, 0);
    println!("examples: {n} cases");

    // and the outline says what the files say
    let m = check_policy_text(MANAGE_EXAMPLE).expect("manage example");
    assert_eq!(
        m,
        PolicyOutline {
            mode: "manage".into(),
            engine_images: 3,
            model_roots: 3,
            writable_roots: 2,
            scripts: 1,
            allow_pull: false,
            allow_delete: true,
            allow_ipc_host: true,
            external_control: "none".into(),
        }
    );
}

#[test]
fn every_key_removed_or_wrongly_typed_agrees() {
    let b = base();
    let mut run = Run::default();
    // (object pointer, its keys in the base)
    let levels = [
        ("", TOP_KEYS),
        ("/docker", DOCKER_KEYS),
        ("/model_roots/0", ROOT_KEYS),
        ("/scripts/0", SCRIPT_KEYS),
        ("/scripts/0/actions/0", ACTION_KEYS),
    ];
    for (parent, keys) in levels {
        for key in keys {
            run.value(
                &format!("{parent}/{key} removed"),
                &without(&b, parent, key),
            );
            for t in wrong_types() {
                run.value(&format!("{parent}/{key} = {t}"), &put(&b, parent, key, t));
            }
        }
        for unknown in ["bogus", "Mode", "ID", "_comment", "_", "_x", " version", ""] {
            run.value(
                &format!("{parent} + unknown {unknown:?}"),
                &put(&b, parent, unknown, json!(1)),
            );
        }
    }
    // a non-object where an object belongs
    for ptr in ["/model_roots/0", "/scripts/0", "/scripts/0/actions/0"] {
        for t in wrong_types() {
            run.value(&format!("{ptr} = {t}"), &set(&b, ptr, t));
        }
    }
    for t in wrong_types() {
        run.text(&format!("top level {t}"), &t.to_string());
    }
    run.text("top level string", "\"policy\"");
    run.text("top level list of object", "[{\"mode\": \"manage\"}]");
    let n = run.finish(20, 200);
    println!("removed/retyped: {n} cases");
}

#[test]
fn top_level_values_agree() {
    let b = base();
    let mut run = Run::default();
    let cases: Vec<(&str, Vec<Value>)> = vec![
        (
            "version",
            vec![
                json!(1),
                json!(2),
                json!("1"),
                json!(1.0),
                json!(true),
                json!(false),
                json!(0),
                json!(-1),
                json!(1.000_000_1),
                Value::Null,
            ],
        ),
        (
            "mode",
            vec![
                json!("monitor"),
                json!("manage"),
                json!("Manage"),
                json!(""),
                json!(" manage"),
                json!(["manage"]),
            ],
        ),
        (
            "external_control",
            vec![
                json!("none"),
                json!("stop_start"),
                json!("STOP_START"),
                json!("stop-start"),
                json!(""),
            ],
        ),
        (
            "max_shm_gib",
            vec![
                json!(0),
                json!(0.0),
                json!(-1),
                json!(0.001),
                json!(1),
                json!(4096),
                json!(4096.0),
                json!(4096.5),
                json!(4097),
                json!(1e3),
                json!("1"),
                json!(true),
                json!(u64::MAX),
                json!(i64::MIN),
            ],
        ),
        (
            "engine_network",
            vec![
                json!("net"),
                json!("a.b_c-d"),
                json!("-net"),
                json!(".net"),
                json!("a b"),
                json!("a/b"),
                json!("a".repeat(64)),
                json!("a".repeat(65)),
                json!(""),
                json!("net\u{0}"),
                json!("réseau"),
            ],
        ),
    ];
    for (key, values) in cases {
        for v in values {
            run.value(&format!("{key} = {v}"), &put(&b, "", key, v));
        }
    }
    for key in ["allow_pull", "allow_delete", "allow_ipc_host"] {
        for v in [
            json!(true),
            json!(false),
            json!("true"),
            json!(1),
            json!(0),
            Value::Null,
        ] {
            run.value(&format!("{key} = {v}"), &put(&b, "", key, v));
        }
    }
    // duplicate keys in the text: the decoder keeps the last
    run.text("duplicate mode", r#"{"mode": "manage", "mode": "bogus"}"#);
    run.text("duplicate mode 2", r#"{"mode": "bogus", "mode": "manage"}"#);
    // unknown keys: the first in document order is the one named
    run.text("two unknown keys", r#"{"zeta": 1, "alpha": 2}"#);
    let n = run.finish(20, 30);
    println!("top-level values: {n} cases");
}

#[test]
fn docker_images_and_ports_agree() {
    let b = base();
    let mut run = Run::default();
    for v in [
        json!(false),
        json!(0),
        json!(""),
        json!([]),
        json!({}),
        json!("docker"),
        json!(["docker"]),
        json!(1),
        json!(true),
        json!({"binary": "docker"}),
        json!({"context": null}),
        json!({"binary": null, "context": null}),
    ] {
        run.value(&format!("docker = {v}"), &put(&b, "", "docker", v));
    }
    for v in [
        json!("docker"),
        json!("/usr/bin/docker"),
        json!("/usr/local/bin/docker"),
        json!("bin/docker"),
        json!("podman"),
        json!("/usr/../bin/docker"),
        json!("/usr/bin/../docker"),
        json!("/usr/bin/./docker"),
        json!("/usr/bin/.."),
        json!("/"),
        json!("//usr/bin/docker"),
        json!("/a b/docker"),
        json!("/opt/docker+cli"),
        json!(format!("/{}", "a".repeat(255))),
        json!(format!("/{}", "a".repeat(256))),
        json!("/usr/bin/döcker"),
        json!(""),
    ] {
        run.value(
            &format!("docker.binary = {v}"),
            &put(&b, "/docker", "binary", v),
        );
    }
    for v in [
        json!("rootless"),
        json!("default"),
        json!("a+b"),
        json!("a.b_c-d"),
        json!("-x"),
        json!("+x"),
        json!("a/b"),
        json!("a b"),
        json!("a".repeat(64)),
        json!("a".repeat(65)),
        json!(""),
    ] {
        run.value(
            &format!("docker.context = {v}"),
            &put(&b, "/docker", "context", v),
        );
    }

    let digest = format!("sha256:{DIGEST}");
    let patterns = vec![
        json!("vllm/vllm-openai:*"),
        json!("vllm/vllm-openai@sha256:*"),
        json!("nvcr.io/nvidia/tensorrt-llm/release:*"),
        json!("localhost:5000/team/repo:*"),
        json!("registry.example.com/team/repo:v1.2"),
        json!("ollama/ollama"),
        json!("ollama/ollama:latest"),
        json!(format!("ollama/ollama@{digest}")),
        json!(format!("ollama/ollama:0.5@{digest}")),
        json!("ollama/ollama:Tag_1.2-rc"),
        json!("ollama/ollama:-tag"),
        json!("ollama/ollama:.tag"),
        json!(format!("ollama/ollama:{}", "t".repeat(128))),
        json!(format!("ollama/ollama:{}", "t".repeat(129))),
        json!("Ollama/ollama:*"),
        json!("ollama/Ollama:latest"),
        json!("REGISTRY.example.com/ollama:*"),
        json!("ollama/*"),
        json!("ollama/oll*:*"),
        json!("*"),
        json!("*/ollama:*"),
        json!("ollama/ollama:*@sha256:*"),
        json!("ollama/ollama:latest*"),
        json!("ollama/ollama:*-cuda"),
        json!("ollama/ollama@sha256:abc"),
        json!(format!("ollama/ollama@sha256:{}", DIGEST.to_uppercase())),
        json!("ollama/ollama@md5:*"),
        json!("ollama//ollama:*"),
        json!("ollama/ollama/:*"),
        json!("a__b/c--d/e.f_g:*"),
        json!("a___b:*"),
        json!("a..b:*"),
        json!(format!("{}:t", "a".repeat(254))),
        json!(format!("{}:t", "a".repeat(255))),
        json!(""),
        json!(" ollama/ollama:*"),
        json!(1),
        Value::Null,
        json!(["vllm/vllm-openai:*"]),
    ];
    for p in &patterns {
        run.value(
            &format!("engine_images = [{p}]"),
            &put(&b, "", "engine_images", json!(["vllm/vllm-openai:*", p])),
        );
    }
    let many: Vec<Value> = (0..64).map(|i| json!(format!("repo{i}/x:*"))).collect();
    run.value(
        "64 engine_images",
        &put(&b, "", "engine_images", json!(many)),
    );
    let too_many: Vec<Value> = (0..65).map(|i| json!(format!("repo{i}/x:*"))).collect();
    run.value(
        "65 engine_images",
        &put(&b, "", "engine_images", json!(too_many)),
    );
    for g in [
        json!(format!("nginx@{digest}")),
        json!("nginx:1.27"),
        json!("nginx"),
        json!("nginx:*"),
        json!("nginx@sha256:*"),
        json!("Nginx:1"),
        json!(format!("{}:t", "a".repeat(255))),
        json!(""),
    ] {
        run.value(
            &format!("gateway_image = {g}"),
            &put(&b, "", "gateway_image", g),
        );
    }

    for pr in [
        json!([1023, 2000]),
        json!([1024, 1024]),
        json!([1024, 2024]),
        json!([1024, 2025]),
        json!([18000, 17999]),
        json!([18000, 19000]),
        json!([18000, 19001]),
        json!([64535, 65535]),
        json!([65000, 65535]),
        json!([65000, 65536]),
        json!([65535, 65535]),
        json!([65536, 65536]),
        json!([0, 10]),
        json!([-1, 10]),
        json!([1024.0, 2000]),
        json!([18000, 18099.5]),
        json!([18000, 18099.0]),
        json!([18000, 18099, 18100]),
        json!([18000]),
        json!([]),
        json!(["18000", 18099]),
        json!([true, 18099]),
        json!([18000, null]),
        json!([u64::MAX, 1]),
        json!("18000-18099"),
        json!({"0": 18000, "1": 18099}),
    ] {
        run.value(
            &format!("port_range = {pr}"),
            &put(&b, "", "port_range", pr),
        );
    }
    for dp in [
        json!([8000, 8000]),
        json!([8000, 30000, 8000]),
        json!([0]),
        json!([1]),
        json!([65535]),
        json!([65536]),
        json!([-1]),
        json!([1.5]),
        json!([8000.0]),
        json!(["80"]),
        json!([true]),
        json!([null]),
        json!([]),
        json!((1..=128).collect::<Vec<u32>>()),
        json!((1..=129).collect::<Vec<u32>>()),
        json!(vec![8000; 128]),
        json!(vec![8000; 129]),
    ] {
        run.value(
            &format!("discovery_ports = {dp}"),
            &put(&b, "", "discovery_ports", dp),
        );
    }
    let n = run.finish(40, 60);
    println!("docker/images/ports: {n} cases");
}

#[test]
fn model_roots_agree() {
    let b = base();
    let mut run = Run::default();
    let r0 = "/model_roots/0";
    for id in [
        json!("hf"),
        json!("h"),
        json!("0"),
        json!("a_b-c"),
        json!("HF"),
        json!("-hf"),
        json!("_hf"),
        json!("a.b"),
        json!("a b"),
        json!("a".repeat(32)),
        json!("a".repeat(33)),
        json!("gguf"),
        json!(""),
        json!("é"),
    ] {
        run.value(&format!("root id = {id}"), &put(&b, r0, "id", id));
    }
    for k in policy::KINDS {
        run.value(&format!("root kind = {k}"), &put(&b, r0, "kind", json!(k)));
    }
    for k in [
        json!("bogus"),
        json!("HF-HUB"),
        json!(""),
        json!("a".repeat(17)),
        json!("a".repeat(16)),
    ] {
        run.value(&format!("root kind = {k}"), &put(&b, r0, "kind", k));
    }
    let mut paths: Vec<String> = vec![
        "/data/hf".into(),
        "/data/hf/".into(),
        "/data//hf".into(),
        "///data/hf".into(),
        "//data/hf".into(),
        "data/hf".into(),
        "./data".into(),
        "~/models".into(),
        String::new(),
        "/data/hf:/x".into(),
        "/data/a,b".into(),
        "/data/a\\b".into(),
        "/data/a\nb".into(),
        "/data/a\rb".into(),
        "/data/a\0b".into(),
        "/data/./hf".into(),
        "/data/../hf".into(),
        "/data/hf/.".into(),
        "/data/hf/..".into(),
        "/data/.hf".into(),
        "/data/..hf".into(),
        "/data/hf...".into(),
        "/data/modèles".into(),
        "/var/lib/ollama".into(),
        "/srv/models".into(),
        "/etc/ecnode".into(),
        "/etc/ecnode/".into(),
        "/etc/ecnode/models".into(),
        "/etc/ecnode//models".into(),
        "//etc/ecnode".into(),
        "//etc/ecnode/models".into(),
        "///etc/ecnode".into(),
        "/etc/ecnodes".into(),
        "/etc/ecnode-x".into(),
        "/etc/ecnod".into(),
        "/etc/x".into(),
        "/etc/".into(),
        "//etc".into(),
        "//".into(),
        "///".into(),
        "//data".into(),
        "//mnt".into(),
        "/mnt/".into(),
        "/var/lib/".into(),
        "/usr//local".into(),
        format!("/data/{}", "a".repeat(506)),
        format!("/data/{}", "a".repeat(507)),
        format!("/data/{}", "é".repeat(506)),
        format!("/data/{}", "é".repeat(507)),
    ];
    paths.extend(policy::FORBIDDEN_ROOTS.iter().map(|p| (*p).to_string()));
    paths.extend(policy::FORBIDDEN_ROOTS.iter().map(|p| format!("{p}/")));
    paths.extend(
        policy::FORBIDDEN_ROOTS
            .iter()
            .map(|p| format!("{p}/models-x")),
    );
    for p in &paths {
        run.value(
            &format!("root path = {p:?}"),
            &put(&b, r0, "path", json!(p)),
        );
    }
    for v in [json!(1), json!(["/data/hf"]), json!("relative")] {
        run.value(&format!("root path = {v}"), &put(&b, r0, "path", v));
    }
    for w in [json!(true), json!(false), json!("true"), json!(1), json!(0)] {
        run.value(&format!("root writable = {w}"), &put(&b, r0, "writable", w));
    }
    for l in [
        json!("HF"),
        json!("a".repeat(64)),
        json!("a".repeat(65)),
        json!("é".repeat(64)),
        json!("é".repeat(65)),
        json!(""),
        json!("a\u{0}b"),
    ] {
        run.value(&format!("root label = {l}"), &put(&b, r0, "label", l));
    }
    // duplicate ids, the second refused (also after a different id)
    run.value(
        "duplicate root id",
        &put(
            &b,
            "",
            "model_roots",
            json!([
                root("a", "/data/a"),
                root("b", "/data/b"),
                root("a", "/data/c")
            ]),
        ),
    );
    run.value(
        "same path twice",
        &put(
            &b,
            "",
            "model_roots",
            json!([root("a", "/data/a"), root("b", "/data/a")]),
        ),
    );
    run.value(
        "missing id and path",
        &put(&b, "", "model_roots", json!([{}])),
    );
    run.value("only id", &put(&b, "", "model_roots", json!([{"id": "a"}])));
    run.value(
        "only path",
        &put(&b, "", "model_roots", json!([{"path": "/data/a"}])),
    );
    run.value(
        "id null",
        &put(
            &b,
            "",
            "model_roots",
            json!([{"id": null, "path": "/data/a"}]),
        ),
    );
    for n in [31, 32, 33] {
        let roots: Vec<Value> = (0..n)
            .map(|i| root(&format!("r{i}"), &format!("/data/m/r{i}")))
            .collect();
        run.value(
            &format!("{n} roots"),
            &put(&b, "", "model_roots", json!(roots)),
        );
    }
    let writable: Vec<Value> = (0..5)
        .map(|i| json!({"id": format!("w{i}"), "path": format!("/data/w{i}"), "writable": i % 2 == 0}))
        .collect();
    run.value(
        "writable count",
        &put(&b, "", "model_roots", json!(writable)),
    );
    let n = run.finish(40, 100);
    println!("model roots: {n} cases");
}

#[test]
fn scripts_agree() {
    let b = base();
    let mut run = Run::default();
    let s0 = "/scripts/0";
    let a0 = "/scripts/0/actions/0";
    for name in [
        json!("big-model"),
        json!("a"),
        json!("Big"),
        json!("-x"),
        json!("a.b"),
        json!("a".repeat(32)),
        json!("a".repeat(33)),
        json!(""),
    ] {
        run.value(
            &format!("script name = {name}"),
            &put(&b, s0, "name", name.clone()),
        );
        run.value(&format!("action name = {name}"), &put(&b, a0, "name", name));
    }
    for c in [
        json!(["myproj-*"]),
        json!([""]),
        json!(["a".repeat(128)]),
        json!(["a".repeat(129)]),
        json!(["é".repeat(128)]),
        json!(["a\u{0}"]),
        json!([1]),
        json!([null]),
        json!(["ok", 2]),
        json!(vec!["c"; 16]),
        json!(vec!["c"; 17]),
    ] {
        run.value(&format!("containers = {c}"), &put(&b, s0, "containers", c));
    }
    for argv in [
        json!(["/opt/lab/x.sh"]),
        json!(["opt/lab/x.sh"]),
        json!(["x.sh", "up"]),
        json!(["/opt/lab/x\n.sh"]),
        json!(["/opt/lab/x.sh", "a\nb"]),
        json!(["/opt/lab/x.sh", ""]),
        json!(["/opt/lab/x.sh", 1]),
        json!(["/opt/lab/x.sh", null]),
        json!(["/opt/lab/x.sh", "a\u{0}"]),
        json!([]),
        json!(["/"]),
        json!([format!("/{}", "a".repeat(4095))]),
        json!([format!("/{}", "a".repeat(4096))]),
        json!(["/opt/lab/x.sh", "é".repeat(4096)]),
        json!(["/opt/lab/x.sh", "é".repeat(4097)]),
        json!(vec!["/opt/lab/x.sh"; 64]),
        json!(vec!["/opt/lab/x.sh"; 65]),
        json!("/opt/lab/x.sh"),
    ] {
        run.value(
            &format!(
                "argv = {}",
                argv.to_string().chars().take(80).collect::<String>()
            ),
            &put(&b, a0, "argv", argv),
        );
    }
    for args in [
        json!(["model-a"]),
        json!([""]),
        json!(["a".repeat(64)]),
        json!(["a".repeat(65)]),
        json!(["é".repeat(64)]),
        json!(["a\u{0}"]),
        json!([1]),
        json!([null]),
        json!(vec!["a"; 32]),
        json!(vec!["a"; 33]),
        json!("model-a"),
    ] {
        run.value(&format!("args = {args}"), &put(&b, a0, "args", args));
    }
    for (key, limit) in [("label", 64usize), ("engine", 32), ("description", 500)] {
        for v in [
            json!("x".repeat(limit)),
            json!("x".repeat(limit + 1)),
            json!("é".repeat(limit)),
            json!("é".repeat(limit + 1)),
            json!(""),
            json!("a\u{0}"),
        ] {
            run.value(
                &format!(
                    "script {key} = {} chars",
                    v.as_str().map_or(0, |s| s.chars().count())
                ),
                &put(&b, s0, key, v.clone()),
            );
            if key == "label" {
                run.value(&format!("action label = {v}"), &put(&b, a0, key, v));
            }
        }
    }
    for st in [
        json!("status"),
        json!("up"),
        json!("missing"),
        json!("Status"),
        json!("a".repeat(32)),
        json!("a".repeat(33)),
        json!(""),
    ] {
        run.value(
            &format!("status_action = {st}"),
            &put(&b, s0, "status_action", st),
        );
    }
    run.value(
        "status_action without actions",
        &put(
            &without(&b, s0, "actions"),
            s0,
            "status_action",
            json!("status"),
        ),
    );
    for t in [
        json!(0),
        json!(1),
        json!(900),
        json!(86400),
        json!(86401),
        json!(1.5),
        json!(900.0),
        json!(-1),
        json!("900"),
        json!(u64::MAX),
    ] {
        run.value(&format!("timeout_s = {t}"), &put(&b, s0, "timeout_s", t));
    }
    for flag in ["confirm", "danger"] {
        for v in [json!(true), json!(false), json!("yes"), json!(1)] {
            run.value(&format!("action {flag} = {v}"), &put(&b, a0, flag, v));
        }
    }
    // duplicates, counts and ordering of the checks
    run.value(
        "duplicate script names",
        &put(
            &b,
            "",
            "scripts",
            json!([{"name": "a"}, {"name": "b"}, {"name": "a"}]),
        ),
    );
    run.value(
        "duplicate action names",
        &put(
            &b,
            s0,
            "actions",
            json!([action("up"), action("down"), action("up")]),
        ),
    );
    run.value(
        "action not object",
        &put(&b, s0, "actions", json!([action("up"), "down"])),
    );
    run.value(
        "bad container and bad action: containers first",
        &put(
            &put(&b, s0, "containers", json!([""])),
            s0,
            "actions",
            json!([1]),
        ),
    );
    run.value(
        "bad status_action and bad timeout: status first",
        &put(
            &put(&b, s0, "status_action", json!("nope")),
            s0,
            "timeout_s",
            json!(0),
        ),
    );
    run.value(
        "bad timeout and bad label: timeout first",
        &put(&put(&b, s0, "timeout_s", json!(0)), s0, "label", json!("")),
    );
    run.value(
        "script missing name",
        &put(&b, "", "scripts", json!([{"actions": [action("up")]}])),
    );
    run.value(
        "minimal script",
        &put(&b, "", "scripts", json!([{"name": "s"}])),
    );
    run.value(
        "action missing argv",
        &put(&b, s0, "actions", json!([{"name": "up"}])),
    );
    run.value(
        "action missing name",
        &put(&b, s0, "actions", json!([{"argv": ["/x"]}])),
    );
    for n in [16, 17] {
        let acts: Vec<Value> = (0..n).map(|i| action(&format!("a{i}"))).collect();
        run.value(
            &format!("{n} actions"),
            &put(&b, s0, "actions", json!(acts)),
        );
    }
    for n in [32, 33] {
        let scripts: Vec<Value> = (0..n).map(|i| json!({"name": format!("s{i}")})).collect();
        run.value(
            &format!("{n} scripts"),
            &put(&b, "", "scripts", json!(scripts)),
        );
    }
    let n = run.finish(30, 80);
    println!("scripts: {n} cases");
}

#[test]
fn check_order_across_sections_agrees() {
    // Several faults at once: the node reports the first in its order, and so must the hub.
    let b = base();
    let mut run = Run::default();
    let faults: Vec<(&str, &str, Value)> = vec![
        ("", "version", json!(2)),
        ("", "mode", json!("x")),
        ("/docker", "binary", json!("rel")),
        ("/docker", "context", json!("-x")),
        ("", "engine_images", json!(["*"])),
        ("", "gateway_image", json!("nginx:*")),
        ("", "engine_network", json!("-n")),
        ("", "port_range", json!([1, 2])),
        ("", "discovery_ports", json!([0])),
        ("/model_roots/0", "path", json!("/etc")),
        ("/scripts/0", "timeout_s", json!(0)),
        ("", "external_control", json!("x")),
        ("", "max_shm_gib", json!(0)),
        ("", "allow_pull", json!(1)),
        ("", "allow_delete", json!(1)),
        ("", "allow_ipc_host", json!(1)),
    ];
    for (i, first) in faults.iter().enumerate() {
        for later in &faults[i + 1..] {
            let v = put(
                &put(&b, first.0, first.1, first.2.clone()),
                later.0,
                later.1,
                later.2.clone(),
            );
            run.value(
                &format!("{}/{} then {}/{}", first.0, first.1, later.0, later.1),
                &v,
            );
        }
    }
    let n = run.finish(0, 100);
    println!("check order: {n} cases");
}

#[test]
fn size_cap_and_json_syntax_agree_with_the_node_loader() {
    assert_eq!(
        Some(MAX_POLICY_BYTES),
        usize::try_from(policy::MAX_POLICY_BYTES).ok(),
        "the cap is the node's"
    );
    let mut run = Run::default();
    // exactly at the cap: decoded and validated (and the node's loader accepts it too)
    let pad = |n: usize| {
        format!(
            "{{\"_comment\": \"{}\", \"mode\": \"manage\"}}",
            "x".repeat(n)
        )
    };
    let overhead = pad(0).len();
    let at_cap = pad(MAX_POLICY_BYTES - overhead);
    assert_eq!(at_cap.len(), MAX_POLICY_BYTES);
    run.text("exactly at the cap", &at_cap);
    assert_eq!(
        node_file_error(&at_cap),
        None,
        "the node accepts a file at the cap"
    );
    run.file_text(
        "one byte over the cap",
        &pad(MAX_POLICY_BYTES - overhead + 1),
    );
    run.file_text(
        "over the cap and not JSON",
        &"x".repeat(MAX_POLICY_BYTES + 1),
    );
    // the cap counts bytes, not characters
    let chars = (MAX_POLICY_BYTES - overhead) / 2 + 1;
    let wide = format!(
        "{{\"_comment\": \"{}\", \"mode\": \"manage\"}}",
        "é".repeat(chars)
    );
    assert!(wide.chars().count() < MAX_POLICY_BYTES && wide.len() > MAX_POLICY_BYTES);
    run.file_text("over the cap in bytes, under in characters", &wide);

    for bad in [
        "",
        " ",
        "\n\n",
        "{",
        "}",
        "[1, 2",
        "{\"mode\" \"manage\"}",
        "{\"mode\": \"manage\",}",
        "{\"mode\": \"manage\" \"x\": 1}",
        "{'mode': 'manage'}",
        "{mode: 1}",
        "{\"mode\": \"manage\"} trailing",
        "{} {}",
        "1 2",
        "{\"mode\": \"unterminated",
        "{\"mode\": \"tab\there\"}",
        "{\"mode\": \"new\nline\"}",
        "{\"mode\": \"\\q\"}",
        "{\"mode\": \"\\u12\"}",
        "{\"max_shm_gib\": NaN}",
        "{\"max_shm_gib\": Infinity}",
        "{\"max_shm_gib\": -Infinity}",
        "NaN",
        "Infinity",
        "{\"max_shm_gib\": 1e999}",
        "{\"max_shm_gib\": 01}",
        "{\"max_shm_gib\": .5}",
        "{\"max_shm_gib\": +1}",
        "{\"mode\": True}",
        "{\"mode\": None}",
        "nul",
        "tru",
        "\u{feff}{}",
        "{\n  \"mode\": \"manage\",\n  \"version\": ,\n}",
        "{\n  \"é\": \"ü\",\n  \"x\": é\n}",
        "// comment\n{}",
        "{\"a\": [1, 2,]}",
        "{\"a\": {\"b\": 1}",
    ] {
        run.file_text(&format!("json {bad:?}"), bad);
    }
    let n = run.finish(1, 40);
    println!("size cap / JSON syntax: {n} cases");
}

#[test]
fn key_lists_are_the_nodes() {
    // The node names its allowed keys, sorted, in the unknown-key error; the hub's lists must give the same names.
    let b = base();
    let levels = [
        ("", "policy", TOP_KEYS),
        ("/docker", "policy.docker", DOCKER_KEYS),
        ("/model_roots/0", "policy.model_roots[0]", ROOT_KEYS),
        ("/scripts/0", "policy.scripts[0]", SCRIPT_KEYS),
        (
            "/scripts/0/actions/0",
            "policy.scripts[0].actions[0]",
            ACTION_KEYS,
        ),
    ];
    for (parent, where_, keys) in levels {
        let v = put(&b, parent, "zzz_unknown", json!(1));
        let node = policy::parse(&v, None, None).map(|_| ()).map_err(|e| e.0);
        let mut sorted = keys.to_vec();
        sorted.sort_unstable();
        let want = format!(
            "{where_}: unknown key 'zzz_unknown' (allowed: {})",
            sorted.join(", ")
        );
        assert_eq!(node, Err(want), "{where_}");
        // and no list carries a key twice
        let mut dedup = sorted.clone();
        dedup.dedup();
        assert_eq!(dedup, sorted, "{where_} lists a key twice");
    }
}
