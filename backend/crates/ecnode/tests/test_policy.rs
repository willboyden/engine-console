//! node/tests/test_policy.py: the node policy: lookup order, refused files (-> monitor mode with a reason), schema,
//! image patterns, reloads.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::needless_pass_by_value,
    clippy::double_ended_iterator_last,
    clippy::redundant_clone,
    unreachable_pub
)]

#[path = "common/wsd_kit.rs"]
mod wsd_kit;

use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, UNIX_EPOCH};

use ecnode::policy::{
    self, DEFAULT_DISCOVERY, LoadOptions, MAX_POLICY_BYTES, Policy, PolicyStore, image_matches,
    parse, valid_image_pattern,
};
use serde_json::{Value, json};
use wsd_kit::write_policy;

fn good() -> Value {
    json!({"version": 1, "mode": "manage", "engine_images": ["vllm/vllm-openai:*"], "discovery_ports": [],
           "model_roots": [{"id": "hf", "path": "/data/hf", "kind": "hf-home", "writable": true}]})
}

fn with(base: Value, change: Value) -> Value {
    let mut b = base;
    for (k, v) in change.as_object().unwrap() {
        b[k] = v.clone();
    }
    b
}

fn opts(tmp: &Path, path: Option<&Path>) -> LoadOptions {
    LoadOptions::new(
        path.map(|p| p.to_str().unwrap()),
        Some(tmp.join("home").to_str().unwrap()),
        tmp.join("etc").join("policy.json").to_str().unwrap(),
    )
}

fn load(tmp: &Path, path: Option<&Path>) -> Policy {
    policy::load_with(&opts(tmp, path)).0
}

fn error(p: &Policy) -> String {
    p.error.clone().unwrap_or_default()
}

// ── lookup order ─────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn test_no_file_means_monitor_mode_with_a_reason() {
    let tmp = tempfile::tempdir().unwrap();
    let p = load(tmp.path(), None);
    assert!(p.mode == "monitor" && p.path.is_none() && error(&p).contains("no policy file"));
}

#[test]
fn test_etc_file_wins_over_the_user_file() {
    let tmp = tempfile::tempdir().unwrap();
    let t = tmp.path();
    let user = write_policy(
        &t.join("home")
            .join(".config")
            .join("ecnode")
            .join("policy.json"),
        &with(good(), json!({"mode": "monitor"})),
        0o644,
    );
    assert_eq!(load(t, None).path.as_deref(), Some(user.to_str().unwrap()));
    let etc = write_policy(&t.join("etc").join("policy.json"), &good(), 0o644);
    // the etc location must be root-owned: a user-owned file there is refused (monitor mode, with the reason)
    let mut o = opts(t, None);
    o.etc_path = etc.to_str().unwrap().into();
    o.system_dirs = vec![format!("{}/", t.join("etc").display())];
    let p = policy::load_with(&o).0;
    assert_eq!(p.path.as_deref(), Some(etc.to_str().unwrap()));
    assert!(
        p.mode == "monitor" && error(&p).contains("owned by root"),
        "{:?}",
        p.error
    );
    // outside the system dirs the same file is fine
    let p = load(t, None);
    assert_eq!(p.path.as_deref(), Some(etc.to_str().unwrap()));
    assert_eq!(p.mode, "manage");
}

#[test]
fn test_explicit_path_is_used_and_missing_is_reported() {
    let tmp = tempfile::tempdir().unwrap();
    let p = load(tmp.path(), Some(&tmp.path().join("missing.json")));
    assert!(p.mode == "monitor" && error(&p).contains("does not exist"));
    let g = write_policy(&tmp.path().join("x").join("p.json"), &good(), 0o644);
    assert_eq!(load(tmp.path(), Some(&g)).mode, "manage");
}

// ── refused files ────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn test_group_or_world_writable_file_is_refused() {
    for mode in [0o664, 0o646, 0o666] {
        let tmp = tempfile::tempdir().unwrap();
        let path = write_policy(&tmp.path().join("x").join("p.json"), &good(), mode);
        let p = load(tmp.path(), Some(&path));
        assert!(
            p.mode == "monitor" && error(&p).contains("writable"),
            "{mode:o}: {:?}",
            p.error
        );
    }
}

#[test]
fn test_symlinked_policy_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let real = write_policy(&tmp.path().join("x").join("real.json"), &good(), 0o644);
    let link = tmp.path().join("x").join("link.json");
    std::os::unix::fs::symlink(&real, &link).unwrap();
    let p = load(tmp.path(), Some(&link));
    assert!(p.mode == "monitor" && error(&p).contains("symlink"));
}

#[test]
fn test_policy_in_a_world_writable_directory_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let d = tmp.path().join("open");
    let path = write_policy(&d.join("p.json"), &good(), 0o644);
    std::fs::set_permissions(&d, std::fs::Permissions::from_mode(0o777)).unwrap();
    let p = load(tmp.path(), Some(&path));
    std::fs::set_permissions(&d, std::fs::Permissions::from_mode(0o700)).unwrap();
    assert!(
        p.mode == "monitor" && error(&p).contains("others can write"),
        "{:?}",
        p.error
    );
}

#[test]
fn test_group_writable_directory_is_fine_for_a_user_policy_but_not_under_etc() {
    let tmp = tempfile::tempdir().unwrap();
    let d = tmp.path().join("grp");
    let path = write_policy(&d.join("p.json"), &good(), 0o644);
    std::fs::set_permissions(&d, std::fs::Permissions::from_mode(0o775)).unwrap();
    assert_eq!(load(tmp.path(), Some(&path)).mode, "manage");
    let mut o = opts(tmp.path(), Some(&path));
    o.system_dirs = vec![format!("{}/", d.display())];
    o.uid = Some(nix::unistd::getuid().as_raw());
    // (user-owned under a system dir is refused for ownership first)
    assert_eq!(policy::load_with(&o).0.mode, "monitor");
}

#[test]
fn test_fifo_is_refused_without_hanging() {
    let tmp = tempfile::tempdir().unwrap();
    let fifo = tmp.path().join("x").join("fifo.json");
    std::fs::create_dir(tmp.path().join("x")).unwrap();
    nix::unistd::mkfifo(&fifo, nix::sys::stat::Mode::from_bits_truncate(0o644)).unwrap();
    let p = load(tmp.path(), Some(&fifo));
    assert!(
        p.mode == "monitor" && error(&p).contains("regular file"),
        "{:?}",
        p.error
    );
}

#[test]
fn test_bad_content_means_monitor_mode_with_the_reason() {
    for (text, why) in [
        ("{not json", "not valid JSON"),
        (r#"{"mode": "manage", "allow_delte": true}"#, "unknown key"),
        (r#"{"mode": "admin"}"#, "monitor\" or \"manage"),
        (r#"{"version": 2}"#, "version"),
        (r#"{"max_shm_gib": NaN}"#, "not valid JSON"),
        ("[1, 2]", "one JSON object"),
    ] {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("x").join("p.json");
        std::fs::create_dir(tmp.path().join("x")).unwrap();
        std::fs::set_permissions(tmp.path().join("x"), std::fs::Permissions::from_mode(0o700))
            .unwrap();
        std::fs::write(&path, text).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        let p = load(tmp.path(), Some(&path));
        assert!(
            p.mode == "monitor" && error(&p).contains(why),
            "{text}: {:?}",
            p.error
        );
        // nothing from a refused file is used
        assert!(p.model_roots.is_empty() && p.scripts.is_empty());
    }
}

#[test]
fn test_oversized_file_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let pad = "x".repeat(usize::try_from(MAX_POLICY_BYTES).unwrap() + 10);
    let path = write_policy(
        &tmp.path().join("x").join("p.json"),
        &with(good(), json!({"_pad": pad})),
        0o644,
    );
    assert!(error(&load(tmp.path(), Some(&path))).contains("larger than"));
}

// ── schema ───────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn test_schema_errors_name_the_field() {
    let cases = [
        (
            json!({"model_roots": [{"id": "x", "path": "relative"}]}),
            "absolute path",
        ),
        (
            json!({"model_roots": [{"id": "x", "path": "/"}]}),
            "too broad",
        ),
        (
            json!({"model_roots": [{"id": "x", "path": "/etc"}]}),
            "too broad",
        ),
        (
            json!({"model_roots": [{"id": "x", "path": "/data/../etc"}]}),
            "'..'",
        ),
        (
            json!({"model_roots": [{"id": "x", "path": "/data/a:b"}]}),
            "':'",
        ),
        (
            json!({"model_roots": [{"id": "Bad Id", "path": "/data/x"}]}),
            "not a valid value",
        ),
        (
            json!({"model_roots": [{"id": "x", "path": "/data/x"}, {"id": "x", "path": "/data/y"}]}),
            "used twice",
        ),
        (
            json!({"model_roots": [{"id": "x", "path": "/data/x", "kind": "zip"}]}),
            "kind must be",
        ),
        (json!({"engine_images": ["*"]}), "literal repository"),
        (json!({"engine_images": ["vllm/*:*"]}), "literal repository"),
        (json!({"gateway_image": "nginx:*"}), "exact image reference"),
        (json!({"port_range": [18100, 18000]}), "port_range[1]"),
        (json!({"port_range": [80, 90]}), "port_range[0]"),
        (json!({"discovery_ports": [0]}), "discovery_ports[0]"),
        (json!({"docker": {"binary": "../docker"}}), "absolute path"),
        (
            json!({"docker": {"context": "-H tcp://x"}}),
            "not a valid value",
        ),
        (json!({"external_control": "all"}), "external_control"),
        (json!({"max_shm_gib": 0}), "max_shm_gib"),
        (json!({"allow_pull": "yes"}), "true or false"),
        (
            json!({"scripts": [{"name": "s", "actions": [{"name": "up", "argv": ["model.sh", "up"]}]}]}),
            "absolute path",
        ),
        (
            json!({"scripts": [{"name": "s", "actions": [{"name": "up", "argv": []}]}]}),
            "1-64 non-empty",
        ),
        (
            json!({"scripts": [{"name": "s", "status_action": "nope", "actions": [{"name": "up", "argv": ["/bin/true"]}]}]}),
            "status_action",
        ),
    ];
    for (change, fragment) in cases {
        // the policy file's directory is no model root's (audit F2): "/x" would put every root inside "/"
        let e = parse(
            &with(good(), change.clone()),
            Some("/x/policy.json"),
            Some("/home/someone"),
        )
        .unwrap_err();
        assert!(e.0.contains(fragment), "{change}: {}", e.0);
    }
}

#[test]
fn test_home_directory_itself_is_not_a_root() {
    assert!(
        parse(
            &json!({"model_roots": [{"id": "h", "path": "/home/someone"}]}),
            None,
            Some("/home/someone")
        )
        .is_err()
    );
    parse(
        &json!({"model_roots": [{"id": "h", "path": "/home/someone/models"}]}),
        None,
        Some("/home/someone"),
    )
    .unwrap();
}

#[test]
fn test_comment_keys_and_defaults() {
    let p = parse(
        &json!({"_comment": "hi", "mode": "manage"}),
        Some("/x"),
        None,
    )
    .unwrap();
    assert!(p.manage() && p.engine_images.is_empty() && !p.allow_ipc_host && !p.allow_delete);
    assert_eq!(p.discovery_ports, DEFAULT_DISCOVERY);
    assert_eq!(p.port_range, (18000, 18099));
    assert!(
        parse(&json!({"discovery_ports": []}), None, None)
            .unwrap()
            .discovery_ports
            .is_empty()
    );
}

#[test]
fn test_summary_never_contains_script_argv() {
    let p = parse(
        &with(
            good(),
            json!({"scripts": [{"name": "big", "containers": ["lab-sglang-*"], "status_action": "status",
                    "actions": [{"name": "status", "argv": ["/opt/lab/secret-path.sh", "status"]},
                                {"name": "switch", "argv": ["/opt/lab/use.sh"], "args": ["a", "b"]}]}]}),
        ),
        Some("/x/policy.json"),
        None,
    )
    .unwrap();
    let s = Value::Object(p.summary()).to_string();
    assert!(!s.contains("secret-path") && !s.contains("/opt/lab"));
    assert_eq!(
        p.summary()["scripts"][0]["actions"][1]["args"],
        json!(["a", "b"])
    );
}

// ── image patterns ───────────────────────────────────────────────────────────────────────────────────────
#[test]
fn test_image_patterns_compare_the_repository_literally() {
    let d64a = format!("vllm/vllm-openai:x@sha256:{}", "a".repeat(64));
    let d64b = format!("lmsysorg/sglang@sha256:{}", "b".repeat(64));
    let cases: Vec<(&str, &str, bool)> = vec![
        ("vllm/vllm-openai:*", "vllm/vllm-openai:v0.11.0", true),
        ("vllm/vllm-openai:*", "vllm/vllm-openai:latest", true),
        ("vllm/vllm-openai:*", "evil/vllm/vllm-openai:x", false),
        ("vllm/vllm-openai:*", "vllm/vllm-openai-evil:x", false),
        ("vllm/vllm-openai:*", "docker.io/vllm/vllm-openai:x", false),
        ("vllm/vllm-openai:*", "vllm/vllm-openai", false),
        ("vllm/vllm-openai:*", &d64a, false),
        ("vllm/vllm-openai:*", "vllm/vllm-openai:x y", false),
        ("vllm/vllm-openai:*", "vllm/vllm-openai:-x", false),
        ("vllm/vllm-openai:*", "--help", false),
        ("lmsysorg/sglang@sha256:*", &d64b, true),
        ("lmsysorg/sglang@sha256:*", "lmsysorg/sglang:latest", false),
        (
            "lmsysorg/sglang@sha256:*",
            "lmsysorg/sglang@sha256:short",
            false,
        ),
        (
            "nvcr.io/nvidia/tensorrt-llm/release:*",
            "nvcr.io/nvidia/tensorrt-llm/release:1.0",
            true,
        ),
        (
            "nvcr.io/nvidia/tensorrt-llm/release:*",
            "nvcr.io/nvidia/tensorrt-llm/release-x:1.0",
            false,
        ),
        ("localhost:5000/eng:*", "localhost:5000/eng:t1", true),
        (
            "ghcr.io/ggml-org/llama.cpp:server-cuda",
            "ghcr.io/ggml-org/llama.cpp:server-cuda",
            true,
        ),
        (
            "ghcr.io/ggml-org/llama.cpp:server-cuda",
            "ghcr.io/ggml-org/llama.cpp:server",
            false,
        ),
    ];
    for (pattern, image, ok) in cases {
        assert!(valid_image_pattern(pattern), "{pattern}");
        assert_eq!(image_matches(pattern, image), ok, "{pattern} {image}");
    }
}

// ── reloads ──────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn test_store_reloads_an_edited_file() {
    let tmp = tempfile::tempdir().unwrap();
    let path = write_policy(&tmp.path().join("x").join("p.json"), &good(), 0o644);
    let clock = Arc::new(Mutex::new(0.0_f64));
    let c = Arc::clone(&clock);
    let mut o = opts(tmp.path(), Some(&path));
    o.etc_path = tmp
        .path()
        .join("etc")
        .join("p.json")
        .to_str()
        .unwrap()
        .into();
    let store = PolicyStore::with_options(o, 2.0, Arc::new(move || *c.lock().unwrap()));
    assert!(store.current().manage());
    write_policy(&path, &with(good(), json!({"mode": "monitor"})), 0o644);
    // a different mtime even within the same clock tick
    std::fs::File::options()
        .write(true)
        .open(&path)
        .unwrap()
        .set_modified(UNIX_EPOCH + Duration::from_nanos(1))
        .unwrap();
    assert!(store.current().manage()); // not re-checked yet
    *clock.lock().unwrap() = 5.0;
    assert_eq!(store.current().mode, "monitor");
}

#[test]
fn test_shipped_examples_are_valid_policies() {
    let ex = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../node/examples");
    for (name, mode) in [
        ("policy.monitor.json", "monitor"),
        ("policy.manage.json", "manage"),
    ] {
        let data: Value =
            serde_json::from_str(&std::fs::read_to_string(ex.join(name)).unwrap()).unwrap();
        let p = parse(&data, Some(name), Some("/home/someone")).unwrap();
        assert!(p.mode == mode && !p.model_roots.is_empty(), "{name}");
    }
}

#[test]
fn test_security_policy_store_from_policy_never_reads_a_file() {
    // Rust-only: a fixed store (tests, in-memory tools) never re-reads anything.
    let store = PolicyStore::from_policy(Policy::default());
    assert_eq!(store.current().mode, "monitor");
    assert!(store.reload().discovery_ports.is_empty());
}
