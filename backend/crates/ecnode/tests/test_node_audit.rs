//! Security audit of the node (Rust-only, tightenings beyond Python): one section per finding, and each test fails
//! without its fix. Findings whose code has private seams are tested next to it (`test_audit_*` unit tests). Every
//! probe here is a scripted answer: no socket is ever opened to a real port.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    unreachable_pub
)]

#[path = "common/wsd_agent.rs"]
mod wsd_agent;
#[path = "common/wsd_kit.rs"]
mod wsd_kit;

use std::collections::BTreeSet;
use std::os::unix::fs::symlink;
use std::path::Path;
use std::sync::{Arc, Mutex};

use ec_proto::{Obj, POLICY_DENIED};
use ecnode::agent::Agent;
use ecnode::discovery::{self, PortProbe, Probe};
use ecnode::errors::HandlerError;
use ecnode::fsops::{WriteLocks, fs_put_small, fs_read, fs_symlink, fs_write_with};
use ecnode::policy::{LoadOptions, Policy, PolicyStore, Root, monotonic, parse};
use serde_json::{Value, json};
use wsd_agent::World;
use wsd_kit::{C, FakeStream, container, manage_policy, obj, store_for, write_policy};

// ── F1: no usable policy, no probing ─────────────────────────────────────────────────────────────────────────

/// A scripted prober that answers like vLLM on EVERY port it is asked about (so anything probed would become an
/// endpoint) and records the ports; it opens no socket.
fn vllm_everywhere(tried: &Arc<Mutex<Vec<u16>>>) -> discovery::Prober {
    let t = Arc::clone(tried);
    Arc::new(move |port| {
        t.lock().unwrap().push(port);
        PortProbe {
            port,
            models: Probe {
                status: Some(200),
                body: br#"{"object": "list", "data": [{"id": "m"}]}"#.to_vec(),
            },
            health: Probe {
                status: Some(200),
                body: Vec::new(),
            },
            metrics: Probe {
                status: Some(200),
                body: b"vllm:num_requests_running 0\n".to_vec(),
            },
        }
    })
}

fn agent_probing(w: &World, store: PolicyStore, tried: &Arc<Mutex<Vec<u16>>>) -> Agent {
    let mut parts = w.parts(w.fb.env());
    parts.prober = vllm_everywhere(tried);
    parts.quiet_prober = vllm_everywhere(tried);
    parts.metrics_get = Arc::new(|_, _, _| Probe::default());
    Agent::with_parts(store, parts)
}

/// A store for an explicit policy file (whatever is at `path`).
fn store_at(w: &World, path: &Path) -> PolicyStore {
    let opts = LoadOptions::new(
        Some(&path.to_string_lossy()),
        Some(&w.path().join("home").to_string_lossy()),
        &w.path()
            .join("no-etc")
            .join("policy.json")
            .to_string_lossy(),
    );
    PolicyStore::with_options(opts, 0.0, monotonic())
}

/// A valid manage policy with NO `discovery_ports` key (Python's defaults apply).
fn policy_with_default_ports(w: &World) -> Value {
    let mut p = manage_policy(&w.machine, json!({}));
    p.as_object_mut().unwrap().remove("discovery_ports");
    p
}

/// A host-network vLLM (its ports are guessed from argv / the engine default 8000) and a vLLM publishing
/// 127.0.0.1:18001.
fn engines() -> Value {
    json!([
        container(
            "hostnet-vllm",
            &"7".repeat(64),
            "vllm/vllm-openai:v1",
            &["vllm", "serve", "m", "--port", "8000"],
            C {
                network: Some("host"),
                ..C::default()
            }
        ),
        container(
            "published-vllm",
            &"8".repeat(64),
            "vllm/vllm-openai:v1",
            &["vllm", "serve", "m"],
            C {
                ports: Some(json!({"8000/tcp": [{"HostIp": "127.0.0.1", "HostPort": "18001"}]})),
                ..C::default()
            }
        ),
    ])
}

/// Every way the node ends up without a usable policy.
fn unusable_policies(w: &World) -> Vec<(&'static str, PolicyStore)> {
    let dir = w.path().join("pol");
    let good = policy_with_default_ports(w);
    let invalid = write_policy(&dir.join("invalid.json"), &json!({}), 0o644);
    std::fs::write(&invalid, "{\"version\": 1, \"mode\": ").unwrap();
    let real = write_policy(&dir.join("real.json"), &good, 0o644);
    symlink(&real, dir.join("link.json")).unwrap();
    let writable = write_policy(&dir.join("writable.json"), &good, 0o666);
    let typo = write_policy(
        &dir.join("typo.json"),
        &json!({"version": 1, "mode": "manage", "discovery_portz": [8000]}),
        0o644,
    );
    vec![
        ("missing", store_for(w.path(), None)),
        ("invalid JSON", store_at(w, &invalid)),
        ("symlink", store_at(w, &dir.join("link.json"))),
        ("group/world-writable", store_at(w, &writable)),
        ("schema error", store_at(w, &typo)),
    ]
}

#[test]
fn test_audit_f1_no_usable_policy_probes_no_port() {
    for case in 0..5 {
        let w = World::new();
        w.fb.update(json!({"containers": [engines()[0].clone()]}));
        let (what, store) = unusable_policies(&w).swap_remove(case);
        let tried: Arc<Mutex<Vec<u16>>> = Arc::default();
        let a = agent_probing(&w, store, &tried);
        let hello = a.hello(false);
        assert!(hello["policy"]["error"].is_string(), "{what}: {hello:?}");
        assert_eq!(hello["policy"]["discovery_ports"], json!([]), "{what}");
        let inv = a.refresh();
        // Python probed 8000, 8080, 11434 and 30000 (and the host-network engine's 8000) every round
        assert_eq!(*tried.lock().unwrap(), Vec::<u16>::new(), "{what}");
        assert_eq!(inv.msg["endpoints"], json!([]), "{what}");
        // the container is still listed (monitor mode shows what runs)
        assert_eq!(inv.msg["containers"][0]["name"], "hostnet-vllm", "{what}");
        assert!(a.allowed_ports(true).is_empty(), "{what}");
    }
}

#[test]
fn test_audit_f1_a_valid_policy_without_discovery_ports_keeps_the_defaults() {
    let w = World::new();
    w.fb.update(json!({"containers": [engines()[0].clone()]}));
    let tried: Arc<Mutex<Vec<u16>>> = Arc::default();
    let a = agent_probing(
        &w,
        store_for(w.path(), Some(&policy_with_default_ports(&w))),
        &tried,
    );
    assert_eq!(a.policy().error, None);
    let inv = a.refresh();
    let probed: BTreeSet<u16> = tried.lock().unwrap().iter().copied().collect();
    assert_eq!(probed, BTreeSet::from([8000, 8080, 11434, 30000]));
    let ports: BTreeSet<u16> = a.allowed_ports(false).into_iter().collect();
    assert_eq!(ports, BTreeSet::from([8000, 8080, 11434, 30000]));
    assert_eq!(inv.msg["endpoints"][0]["container"], "hostnet-vllm");
}

#[test]
fn test_audit_f1_without_a_policy_only_published_ports_are_reachable() {
    // a container that publishes a port is still probed and reachable; a guessed or bare port never is
    let w = World::new();
    w.fb.update(json!({"containers": engines()}));
    let tried: Arc<Mutex<Vec<u16>>> = Arc::default();
    let a = agent_probing(&w, store_for(w.path(), None), &tried);
    let inv = a.refresh();
    assert_eq!(*tried.lock().unwrap(), vec![18001]);
    let eps: Vec<(Value, Value)> = inv.msg["endpoints"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| (e["port"].clone(), e["container"].clone()))
        .collect();
    assert_eq!(eps, vec![(json!(18001), json!("published-vllm"))]);
    assert_eq!(
        a.allowed_ports(true).into_iter().collect::<Vec<_>>(),
        vec![18001]
    );

    // an inventory built while the policy was valid holds bare and guessed ports; once the policy is refused they
    // are no longer reachable, even before the next inventory round
    let w = World::new();
    w.fb.update(json!({"containers": engines()}));
    let path = write_policy(
        &w.path().join("pol").join("policy.json"),
        &policy_with_default_ports(&w),
        0o644,
    );
    let tried: Arc<Mutex<Vec<u16>>> = Arc::default();
    let a = agent_probing(&w, store_at(&w, &path), &tried);
    a.refresh();
    let before: BTreeSet<u16> = a.allowed_ports(false).into_iter().collect();
    assert_eq!(before, BTreeSet::from([8000, 8080, 11434, 18001, 30000]));
    std::fs::write(&path, "{ not json").unwrap();
    let after: BTreeSet<u16> = a.allowed_ports(false).into_iter().collect();
    assert!(a.policy().error.is_some());
    assert_eq!(after, BTreeSet::from([18001]));
}

// ── F2: model roots and writes stay clear of the node's and the user's control files ──────────────────────────

#[test]
fn test_audit_f2_model_roots_stay_clear_of_the_nodes_and_users_files() {
    let tmp = tempfile::tempdir().unwrap();
    let t = tmp.path();
    let home = t.join("users").join("someone");
    for d in [
        ".ssh",
        ".config/ecnode",
        ".local/state/ecnode",
        "models",
        ".cache/huggingface/hub",
    ] {
        std::fs::create_dir_all(home.join(d)).unwrap();
    }
    std::fs::create_dir_all(t.join("pol")).unwrap();
    std::fs::create_dir_all(t.join("data").join("hf")).unwrap();
    symlink(home.join(".ssh"), t.join("data").join("keys")).unwrap();
    symlink(&home, t.join("data").join("me")).unwrap();
    let policy_path = t.join("pol").join("policy.json");
    let s = |p: &Path| p.to_string_lossy().into_owned();
    let check = |root: &Path, writable: bool| {
        parse(
            &json!({"model_roots": [{"id": "r", "path": s(root), "writable": writable}]}),
            Some(&s(&policy_path)),
            Some(&s(&home)),
        )
    };
    let refused = [
        home.join(".ssh"),
        home.join(".ssh").join("old"),
        home.join(".config"),
        home.join(".config").join("ecnode"),
        home.join(".config").join("ecnode").join("x"),
        home.join(".local"),
        home.join(".local").join("state"),
        home.join(".local").join("state").join("ecnode"),
        home.join(".local")
            .join("state")
            .join("ecnode")
            .join("instances"),
        t.join("users"), // an ancestor of the home directory: it holds every rc file
        t.to_path_buf(), // holds the home directory and the policy file
        t.join("pol"),
        t.join("pol").join("models"),
        Path::new("/etc/ecnode").to_path_buf(),
        Path::new("/etc/ecnode/models").to_path_buf(),
        t.join("data").join("keys"), // a symlink to ~/.ssh
        t.join("data").join("me"),   // a symlink to the home directory
    ];
    for root in &refused {
        for writable in [false, true] {
            let e = check(root, writable)
                .err()
                .unwrap_or_else(|| panic!("{} (writable {writable}) was accepted", root.display()));
            assert!(
                e.0.starts_with("policy.model_roots[0].path: "),
                "{}: {}",
                root.display(),
                e.0
            );
        }
    }
    let e = check(&home.join(".config"), true).unwrap_err();
    assert!(
        e.0.contains("overlaps the node's user policy directory"),
        "{}",
        e.0
    );
    let e = check(&t.join("users"), true).unwrap_err();
    assert!(e.0.contains("holds the home directory"), "{}", e.0);
    // ordinary roots, including ones inside the home directory, are fine
    for root in [
        t.join("data").join("hf"),
        home.join("models"),
        home.join(".cache").join("huggingface").join("hub"),
    ] {
        check(&root, true).unwrap_or_else(|e| panic!("{}: {}", root.display(), e.0));
    }
}

fn writable_root(root: &Path) -> Policy {
    Policy {
        path: Some("/x/policy.json".into()),
        mode: "manage".into(),
        model_roots: vec![Root {
            writable: true,
            ..Root::new("w", &root.to_string_lossy())
        }],
        ..Policy::default()
    }
}

fn denied(r: Result<Obj, HandlerError>, what: &str) {
    match r {
        Err(HandlerError::Op(e)) => assert_eq!(e.code, POLICY_DENIED, "{what}: {e:?}"),
        other => panic!("{what}: expected policy_denied, got {other:?}"),
    }
}

fn put(policy: &Policy, rel: &str) -> Result<Obj, HandlerError> {
    fs_put_small(
        policy,
        &obj(json!({"root": "w", "rel": rel, "content_b64": "eA=="})),
    )
}

fn write(policy: &Policy, rel: &str) -> Result<Obj, HandlerError> {
    let stream = FakeStream::new(vec![b"abc".to_vec()]);
    fs_write_with(
        policy,
        &obj(json!({"root": "w", "rel": rel, "size": 3})),
        &stream,
        &WriteLocks::new(),
        &|_| Ok((1 << 30, 4096)),
    )
}

fn link(policy: &Policy, at: &str, target: &str) -> Result<Obj, HandlerError> {
    fs_symlink(
        policy,
        &obj(json!({"root": "w", "link": at, "target": target})),
    )
}

#[test]
fn test_audit_f2_writes_refuse_login_scripts_keys_and_credential_names() {
    // a writable root that happens to be someone's home-like directory (e.g. a shared /data/<user>)
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("root");
    std::fs::create_dir_all(root.join("blobs")).unwrap();
    std::fs::write(root.join("blobs").join("b1"), "x").unwrap();
    let p = writable_root(&root);
    for rel in [
        ".bashrc",
        ".profile",
        ".zshrc",
        ".BASH_PROFILE",
        ".pgpass",
        ".gitconfig",
        "repo/.git/hooks/post-checkout",
        ".netrc",
        ".ssh/authorized_keys",
        "sub/.config/ecnode/policy.json",
        ".local/state/ecnode/x",
        "api_token.json",
    ] {
        denied(put(&p, rel), &format!("put_small {rel}"));
        denied(write(&p, rel), &format!("write {rel}"));
        denied(link(&p, rel, "blobs/b1"), &format!("symlink {rel}"));
    }
    // nothing was created on the way: no file, no directory
    let mut left: Vec<String> = std::fs::read_dir(&root)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    left.sort();
    assert_eq!(left, vec!["blobs"]);
}

#[test]
fn test_audit_f2_writes_refuse_a_protected_directory_reached_through_a_link() {
    // no segment of the path names it: a symlink inside the root leads into a protected directory inside the root
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("root");
    std::fs::create_dir_all(root.join(".ssh")).unwrap();
    std::fs::create_dir_all(root.join("a").join(".config")).unwrap();
    symlink(".ssh", root.join("alias")).unwrap();
    symlink("a/.config", root.join("cfg")).unwrap();
    let p = writable_root(&root);
    denied(put(&p, "alias/config"), "put_small through a link");
    denied(write(&p, "alias/rc"), "write through a link");
    denied(
        write(&p, "cfg/new/dir/file"),
        "write through a link, new dirs",
    );
    denied(link(&p, "alias/x", "../a"), "symlink through a link");
    assert!(
        std::fs::read_dir(root.join(".ssh"))
            .unwrap()
            .next()
            .is_none()
    );
    assert!(
        std::fs::read_dir(root.join("a").join(".config"))
            .unwrap()
            .next()
            .is_none()
    );
}

#[test]
fn test_audit_f2_model_writes_still_work() {
    // what a hub writes for an HF download or a GGUF copy is unaffected
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("root");
    std::fs::create_dir_all(&root).unwrap();
    let p = writable_root(&root);
    let repo = "models--org--name";
    write(&p, &format!("{repo}/blobs/0123abcd")).unwrap();
    link(
        &p,
        &format!("{repo}/snapshots/rev1/tokenizer.json"),
        "../../blobs/0123abcd",
    )
    .unwrap();
    put(&p, &format!("{repo}/refs/main")).unwrap();
    write(&p, "gguf/model-Q4_K_M.gguf").unwrap();
    write(&p, "plain/special_tokens_map.json").unwrap();
    write(&p, "plain/.gitattributes").unwrap();
    assert_eq!(
        std::fs::read_link(root.join(repo).join("snapshots/rev1/tokenizer.json")).unwrap(),
        Path::new("../../blobs/0123abcd")
    );
    let stream = FakeStream::empty();
    fs_read(
        &p,
        &obj(json!({"root": "w", "rel": format!("{repo}/blobs/0123abcd")})),
        &stream,
    )
    .unwrap();
    assert_eq!(stream.data(), b"abc");
    assert_eq!(
        std::fs::read(root.join(repo).join("refs/main")).unwrap(),
        b"x"
    );
}

// ── F3: a bare port needs a real /v1/models listing ──────────────────────────────────────────────────────────

fn probe(status: i64, body: &[u8]) -> Probe {
    Probe {
        status: Some(status),
        body: body.to_vec(),
    }
}

#[test]
fn test_audit_f3_a_bare_port_needs_a_models_listing() {
    let listing = || probe(200, br#"{"object": "list", "data": [{"id": "m"}]}"#);
    let prober: discovery::Prober = Arc::new(move |port| {
        let none = Probe::default;
        let (models, health, metrics) = match port {
            // any page whose /metrics merely mentions an engine (a Grafana, a proxy, a scraped exporter)
            9001 => (
                none(),
                none(),
                probe(200, b"# scraped: vllm:num_requests_running 3\n"),
            ),
            9002 => (
                probe(200, b"<!doctype html>"),
                probe(200, b"ok"),
                probe(200, b"dynamo_frontend_requests_total 1\n"),
            ),
            9003 => (
                probe(404, b""),
                none(),
                probe(200, b"sglang:num_running_reqs 0\n"),
            ),
            // a real engine: listing + metrics naming it
            9004 => (
                listing(),
                none(),
                probe(200, b"vllm:num_requests_running 0\n"),
            ),
            // an OpenAI-compatible server: listing + health
            9005 => (listing(), probe(200, b""), none()),
            // a native Ollama: a listing, no /health, no /metrics
            11434 => (
                probe(200, br#"{"object": "list", "data": []}"#),
                none(),
                none(),
            ),
            9006 => (
                probe(
                    200,
                    br#"{"data": [{"id": "llama3:8b", "owned_by": "library"}]}"#,
                ),
                none(),
                none(),
            ),
            // an engine with an API key: 401 + /metrics naming it
            9007 => (probe(401, b""), none(), probe(200, b"vllm:x 1\n")),
            _ => (none(), none(), none()),
        };
        PortProbe {
            port,
            models,
            health,
            metrics,
        }
    });
    let eps = discovery::build_endpoints(
        &[],
        &[9001, 9002, 9003, 9004, 9005, 11434, 9006, 9007],
        &prober,
        2,
        &prober,
    );
    let got: Vec<(Value, Value)> = eps
        .iter()
        .map(|e| (e["port"].clone(), e["engine"].clone()))
        .collect();
    assert_eq!(
        got,
        vec![
            (json!(9004), json!("vllm")),
            (json!(9005), json!("openai_compatible")),
            (json!(11434), json!("ollama")),
            (json!(9006), json!("ollama")),
            (json!(9007), json!("vllm")),
        ]
    );
}
