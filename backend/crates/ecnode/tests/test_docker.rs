//! node/tests/test_docker.py: docker.* ops against the fake docker CLI: run-spec rules, argv hardening, secrets,
//! ownership, gateway.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::needless_pass_by_value,
    clippy::double_ended_iterator_last,
    clippy::redundant_clone,
    unreachable_pub
)]

#[path = "common/wsd_agent.rs"]
mod wsd_agent;
#[path = "common/wsd_kit.rs"]
mod wsd_kit;

use std::collections::BTreeSet;
use std::os::unix::fs::PermissionsExt;
use std::sync::Arc;

use ec_proto::{
    BAD_ARGS, CONFLICT, DOCKER_ERROR, NOT_FOUND, NOT_OWNED, POLICY_DENIED, PORT_IN_USE,
};
use ecnode::agent::Agent;
use ecnode::docker::effective_context;
use ecnode::gpus::Gpus;
use ecnode::util::{CmdResult, EnvMap, ProcRunner, Runner};
use serde_json::{Value, json};
use wsd_agent::{World, call, err, ok};
use wsd_kit::{
    ARGV_SECRET, CID_EXT, ENV_VALUE, FakeStream, GATEWAY_IMAGE, GPU_A, GPU_B, cid_engine,
    manage_policy,
};

const SECRET: &str = "hub-secret-value-for-the-engine";

fn spec(over: Value) -> Value {
    let mut s = json!({
        "name": "ec-test-1", "instance": "inst_9", "image": "vllm/vllm-openai:v0.11.0",
        "argv": ["--model", "org/m", "--port", "8000"], "container_port": 8000, "gpu_uuids": [GPU_A, GPU_B],
        "labels": {"engine-console.role": "engine", "engine-console.instance": "inst_9",
                   "engine-console.engine": "vllm", "engine-console.gpu_ids": "0,1"},
        "env": {"HF_HUB_OFFLINE": "1"}, "secret_env": {"VLLM_API_KEY": SECRET},
        "mounts": [{"root": "hf", "rel": "", "dest": "/root/.cache/huggingface", "ro": true}],
        "volumes": [{"name": "ec-vllm-cache", "dest": "/root/.cache/vllm"}], "shm_size": "16g", "ipc_host": true,
        "files": {"/etc/engine/chat.jinja": "{{ messages }}"},
    });
    for (k, v) in over.as_object().unwrap() {
        s[k] = v.clone();
    }
    s
}

/// the `agent` fixture: manage policy, GPUs learned from the fake nvidia-smi
fn agent(w: &World, over: Value) -> Agent {
    let a = w.agent(Some(&manage_policy(&w.machine, over)));
    a.hello(false);
    a
}

fn argv_of(r: &ec_proto::Obj) -> Vec<String> {
    r["argv"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_str().unwrap().to_string())
        .collect()
}

fn mode_of(p: &std::path::Path) -> u32 {
    std::fs::metadata(p).unwrap().permissions().mode() & 0o7777
}

#[test]
fn test_render_shows_the_hardened_argv_without_secret_values() {
    let w = World::new();
    let a = agent(&w, json!({}));
    let argv = argv_of(&ok(&a, "docker.render", json!({"spec": spec(json!({}))})));
    assert_eq!(argv[..4], ["docker", "--context", "testctx", "run"]);
    let joined = argv.join(" ");
    let hf = w.machine.data.join("hf");
    for frag in [
        "--cap-drop ALL".to_string(),
        "--security-opt no-new-privileges:true".into(),
        "--pids-limit 4096".into(),
        "--restart no".into(),
        "--pull never".into(),
        "--network engine-console-engines".into(),
        "--network-alias ec-test-1".into(),
        format!("--gpus \"device={GPU_A},{GPU_B}\""),
        "--ipc host".into(),
        "--shm-size 16g".into(),
        "--label engine-console=1".into(),
        "--label engine-console.instance=inst_9".into(),
        "-e HF_HUB_OFFLINE=1".into(),
        format!("-v {}:/root/.cache/huggingface:ro", hf.display()),
        "-v ec-vllm-cache:/root/.cache/vllm".into(),
    ] {
        assert!(joined.contains(&frag), "{frag}");
    }
    let hf_off = argv.iter().position(|x| x == "HF_HUB_OFFLINE=1").unwrap();
    let e = argv.iter().skip(hf_off).position(|x| x == "-e").unwrap() + hf_off;
    assert_eq!(argv[e + 1], "VLLM_API_KEY");
    assert!(!joined.contains(SECRET));
    assert_eq!(
        argv[argv.len() - 5..],
        [
            "vllm/vllm-openai:v0.11.0",
            "--model",
            "org/m",
            "--port",
            "8000"
        ]
    );
    let files = w.state().join("instances").join("inst_9").join("files");
    assert!(joined.contains(&format!(
        "{}/0-chat.jinja:/etc/engine/chat.jinja:ro",
        files.display()
    )));
    assert!(!files.exists()); // render changes nothing
    assert!(
        !w.fb
            .argvs()
            .iter()
            .any(|c| c.get(2).map(String::as_str) == Some("run"))
    );
}

#[test]
fn test_run_passes_secrets_only_through_the_child_environment() {
    let w = World::new();
    let a = agent(&w, json!({}));
    let res = ok(&a, "docker.run", json!({"spec": spec(json!({}))}));
    assert_eq!(res["id"].as_str().unwrap().len(), 64);
    let runs: Vec<Value> =
        w.fb.calls()
            .into_iter()
            .filter(|c| {
                c["argv"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .take(3)
                    .any(|x| x == "run")
            })
            .collect();
    assert_eq!(runs.len(), 1);
    assert!(!runs[0]["argv"].to_string().contains(SECRET));
    assert_eq!(runs[0]["env"], json!({"VLLM_API_KEY": SECRET})); // docker read it from its own environment
    assert_eq!(
        w.fb.state()["networks"]["engine-console-engines"],
        json!({"internal": true})
    );
    let fdir = w.state().join("instances").join("inst_9").join("files");
    assert_eq!(
        std::fs::read_to_string(fdir.join("0-chat.jinja")).unwrap(),
        "{{ messages }}"
    );
    for d in [
        w.state(),
        fdir.parent().unwrap().to_path_buf(),
        fdir.clone(),
    ] {
        assert_eq!(mode_of(&d), 0o700, "{}", d.display());
    }
    assert_eq!(mode_of(&fdir.join("0-chat.jinja")), 0o644);
}

// 2026-10-04 review, L1: the docker CLI's environment is built from scratch, plus docker.run's secret values.
const CHILD_NAMES: &[&str] = &[
    "PATH",
    "HOME",
    "LANG",
    "XDG_RUNTIME_DIR",
    "DOCKER_HOST",
    "DOCKER_CONTEXT",
];

fn names(v: &Value) -> BTreeSet<String> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_str().unwrap().to_string())
        .collect()
}

fn child_names(extra: &[&str]) -> BTreeSet<String> {
    CHILD_NAMES
        .iter()
        .chain(extra)
        .map(|s| (*s).to_string())
        .collect()
}

#[test]
fn test_docker_children_get_an_explicit_environment() {
    let w = World::new();
    let t = w.path().display().to_string();
    let mut env: EnvMap = w.fb.env();
    for (k, v) in [
        ("LD_PRELOAD", format!("{t}/no-such-evil.so")),
        ("LD_LIBRARY_PATH", t.clone()),
        ("DOCKER_CONFIG", format!("{t}/evil-docker-config")),
        ("DOCKER_CERT_PATH", t.clone()),
        (
            "OTEL_EXPORTER_OTLP_ENDPOINT",
            "http://198.51.100.7:4318".into(),
        ),
        ("OTEL_SERVICE_NAME", "x".into()),
        ("PYTHONPATH", format!("{t}/nowhere")),
        ("GODEBUG", "http2debug=2".into()),
        ("LC_ALL", "C".into()),
        ("HTTPS_PROXY", "http://198.51.100.7:3128".into()),
        ("CANARY_SECRET", "must-not-reach-docker".into()),
        ("HOME", t.clone()),
        ("XDG_RUNTIME_DIR", format!("{t}/run")),
        ("DOCKER_HOST", format!("unix://{t}/run/docker.sock")),
        ("DOCKER_CONTEXT", "testctx".into()),
    ] {
        env.insert(k.into(), v);
    }
    let a = w.agent_with(
        Some(&manage_policy(&w.machine, json!({}))),
        w.parts(env.clone()),
    );
    a.hello(false);
    let _ = std::fs::remove_file(&w.fb.docker_log); // only docker.run's calls below
    ok(&a, "docker.run", json!({"spec": spec(json!({}))}));
    let calls = w.fb.calls();
    let is_run = |c: &Value| {
        c["argv"]
            .as_array()
            .unwrap()
            .iter()
            .take(3)
            .any(|x| x == "run")
    };
    let runs: Vec<&Value> = calls.iter().filter(|c| is_run(c)).collect();
    assert!(runs.len() == 1 && calls.len() > 1);
    assert_eq!(names(&runs[0]["env_names"]), child_names(&["VLLM_API_KEY"])); // exactly these, nothing inherited
    assert_eq!(runs[0]["env"], json!({"VLLM_API_KEY": SECRET})); // docker still reads -e NAME
    for c in calls.iter().filter(|c| !is_run(c)) {
        assert_eq!(names(&c["env_names"]), child_names(&[]), "{}", c["argv"]); // network inspect/create, ...
    }
    let _ = std::fs::remove_file(w.fb.smi_log());
    let runner: Arc<dyn Runner> = Arc::new(ProcRunner::new(env));
    assert!(!Gpus::new(false, runner).static_info().0.is_empty()); // nvidia-smi: same rule
    for line in std::fs::read_to_string(w.fb.smi_log()).unwrap().lines() {
        let v: Value = serde_json::from_str(line).unwrap();
        assert_eq!(names(&v["env_names"]), child_names(&[]));
    }
}

#[test]
fn test_run_spec_rules() {
    let cases: Vec<(Value, &str, &str)> = vec![
        (json!({"name": "my-engine"}), BAD_ARGS, "spec.name"),
        (json!({"name": "ec-UPPER"}), BAD_ARGS, "spec.name"),
        (json!({"instance": "bad id!"}), BAD_ARGS, "spec.instance"),
        (
            json!({"image": "evil/vllm/vllm-openai:x"}),
            POLICY_DENIED,
            "not allowed",
        ),
        (
            json!({"image": "vllm/vllm-openai-evil:x"}),
            POLICY_DENIED,
            "not allowed",
        ),
        (
            json!({"image": "--privileged"}),
            BAD_ARGS,
            "image reference",
        ),
        (json!({"container_port": 0}), BAD_ARGS, "container_port"),
        (
            json!({"gpu_uuids": ["GPU-not-here"]}),
            BAD_ARGS,
            "not a GPU of this machine",
        ),
        (json!({"network_alias": "x"}), BAD_ARGS, "network_alias"),
        (
            json!({"labels": {"com.evil": "1"}}),
            BAD_ARGS,
            "engine-console.",
        ),
        (
            json!({"labels": {"engine-console.role": "gateway"}}),
            BAD_ARGS,
            "must be 'engine'",
        ),
        (
            json!({"labels": {"engine-console.x": "a\nb"}}),
            BAD_ARGS,
            "control character",
        ),
        (json!({"env": {"1BAD": "x"}}), BAD_ARGS, "env name"),
        (json!({"env": {"HF_TOKEN": "x"}}), BAD_ARGS, "secret_env"),
        (
            json!({"secret_env": {"LD_PRELOAD": "/tmp/x.so"}}),
            POLICY_DENIED,
            "docker CLI",
        ),
        (
            json!({"secret_env": {"DOCKER_HOST": "tcp://x"}}),
            POLICY_DENIED,
            "docker CLI",
        ),
        (
            json!({"secret_env": {"PATH": "/tmp"}}),
            POLICY_DENIED,
            "docker CLI",
        ),
        (
            json!({"secret_env": {"HF_HUB_OFFLINE": "1"}}),
            BAD_ARGS,
            "does not look like a secret",
        ),
        (
            json!({"secret_env": {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://198.51.100.7:4318"}}),
            POLICY_DENIED,
            "docker CLI",
        ),
        (
            json!({"secret_env": {"OTEL_EXPORTER_OTLP_CLIENT_KEY": "/x.key"}}),
            POLICY_DENIED,
            "docker CLI",
        ),
        (
            json!({"secret_env": {"GRPC_DEFAULT_SSL_ROOTS_FILE_PATH": "/x"}}),
            POLICY_DENIED,
            "docker CLI",
        ),
        (
            json!({"secret_env": {"SSLKEYLOGFILE": "/x"}}),
            POLICY_DENIED,
            "docker CLI",
        ),
        (
            json!({"secret_env": {"LOCPATH": "/x"}}),
            POLICY_DENIED,
            "docker CLI",
        ),
        (
            json!({"mounts": [{"root": "nope", "rel": "", "dest": "/m"}]}),
            NOT_FOUND,
            "model root",
        ),
        (
            json!({"mounts": [{"root": "hf", "rel": "../..", "dest": "/m"}]}),
            BAD_ARGS,
            "'..'",
        ),
        (
            json!({"mounts": [{"root": "hf", "rel": "", "dest": "relative"}]}),
            BAD_ARGS,
            "absolute",
        ),
        (
            json!({"mounts": [{"root": "hf", "rel": "", "dest": "/m:rw,z"}]}),
            BAD_ARGS,
            "':'",
        ),
        (
            json!({"mounts": [{"root": "hf", "rel": "", "dest": "/a/../etc"}]}),
            BAD_ARGS,
            "'..'",
        ),
        (
            json!({"mounts": [{"root": "hf", "rel": "", "dest": "/"}]}),
            BAD_ARGS,
            "not '/'",
        ),
        (
            json!({"volumes": [{"name": "data", "dest": "/d"}]}),
            BAD_ARGS,
            "ec-<name>",
        ),
        (
            json!({"volumes": [{"name": "ec-x", "dest": "/root/.cache/huggingface"}]}),
            BAD_ARGS,
            "used twice",
        ),
        (json!({"shm_size": "16G"}), BAD_ARGS, "shm_size"),
        (json!({"shm_size": "65g"}), POLICY_DENIED, "max_shm_gib"),
        (
            json!({"files": {"relative.txt": "x"}}),
            BAD_ARGS,
            "absolute",
        ),
    ];
    let w = World::new();
    let a = agent(&w, json!({}));
    for (over, code, fragment) in cases {
        let (c, m) = err(&a, "docker.run", json!({"spec": spec(over.clone())}));
        assert!(c == code && m.contains(fragment), "{over}: {c} {m}");
    }
}

#[test]
fn test_mount_symlink_escape_and_writable_rules() {
    let w = World::new();
    let a = agent(&w, json!({}));
    std::os::unix::fs::symlink(w.path(), w.machine.data.join("hf").join("esc")).unwrap();
    let (c, _) = err(
        &a,
        "docker.render",
        json!({"spec": spec(json!({"mounts": [{"root": "hf", "rel": "esc", "dest": "/m"}]}))}),
    );
    assert_eq!(c, POLICY_DENIED);
    std::fs::create_dir(w.machine.data.join("ro").join("m")).unwrap();
    // audit H3: a read-write mount is refused even on a writable root (a container could plant links in it)
    for (root, rel) in [("ro", "m"), ("hf", "")] {
        let (c, m) = err(
            &a,
            "docker.render",
            json!({"spec": spec(json!({"mounts": [{"root": root, "rel": rel, "dest": "/m", "ro": false}]}))}),
        );
        assert!(c == POLICY_DENIED && m.contains("read-only"), "{root}");
    }
    let argv = argv_of(&ok(
        &a,
        "docker.render",
        json!({"spec": spec(json!({"mounts": [{"root": "ro", "rel": "m", "dest": "/models", "ro": true},
                                              {"root": "hf", "rel": "", "dest": "/hf"}]}))}),
    ));
    assert!(argv.contains(&format!(
        "{}:/models:ro",
        w.machine.data.join("ro").join("m").display()
    )));
    assert!(argv.contains(&format!("{}:/hf:ro", w.machine.data.join("hf").display())));
    assert!(!argv.iter().any(|x| x.ends_with(":/hf")));
}

#[test]
fn test_empty_rel_is_the_root_itself_for_mounts_only() {
    let w = World::new();
    let a = agent(&w, json!({}));
    let argv = argv_of(&ok(
        &a,
        "docker.render",
        json!({"spec": spec(json!({"mounts": [{"root": "hf", "rel": "", "dest": "/hf"}]}))}),
    ));
    assert!(argv.contains(&format!("{}:/hf:ro", w.machine.data.join("hf").display())));
    for rel in [".", "./", "x/../"] {
        let (c, _) = err(
            &a,
            "docker.render",
            json!({"spec": spec(json!({"mounts": [{"root": "hf", "rel": rel, "dest": "/hf"}]}))}),
        );
        assert_eq!(c, BAD_ARGS, "{rel}");
    }
}

#[test]
fn test_docker_context_falls_back_to_rootless_only_when_default_is_unusable() {
    let w = World::new();
    let runner = ProcRunner::new(w.fb.env());
    w.fb.update(
        json!({"context": "default", "contexts": ["default", "rootless", "desktop-linux"]}),
    );
    let nosock = w.path().join("no-such.sock").to_string_lossy().into_owned();
    let empty = EnvMap::new();
    let one =
        |k: &str, v: &str| -> EnvMap { [(k.to_string(), v.to_string())].into_iter().collect() };
    assert_eq!(
        effective_context("docker", Some("lab"), &runner, &empty, &nosock).as_deref(),
        Some("lab")
    ); // policy wins
    assert_eq!(
        effective_context("docker", None, &runner, &empty, &nosock).as_deref(),
        Some("rootless")
    );
    assert_eq!(
        effective_context(
            "docker",
            None,
            &runner,
            &one("DOCKER_HOST", "unix:///x"),
            &nosock
        ),
        None
    );
    assert_eq!(
        effective_context(
            "docker",
            None,
            &runner,
            &one("DOCKER_CONTEXT", "x"),
            &nosock
        ),
        None
    );
    let usable = w.path().join("docker.sock");
    std::fs::write(&usable, "").unwrap();
    std::fs::set_permissions(&usable, std::fs::Permissions::from_mode(0o600)).unwrap();
    // a default socket this user may use: keep "default"
    assert_eq!(
        effective_context("docker", None, &runner, &empty, usable.to_str().unwrap()),
        None
    );
    w.fb.update(json!({"context": "lab-remote"})); // a current context chosen in the CLI config
    assert_eq!(
        effective_context("docker", None, &runner, &empty, &nosock),
        None
    );
    w.fb.update(json!({"context": "default", "contexts": ["default"]})); // no rootless context to fall back to
    assert_eq!(
        effective_context("docker", None, &runner, &empty, &nosock),
        None
    );
}

#[test]
fn test_ipc_host_needs_the_policy() {
    let w = World::new();
    let a = agent(&w, json!({"allow_ipc_host": false}));
    assert_eq!(
        err(&a, "docker.render", json!({"spec": spec(json!({}))})).0,
        POLICY_DENIED
    );
    let argv = argv_of(&ok(
        &a,
        "docker.render",
        json!({"spec": spec(json!({"ipc_host": false}))}),
    ));
    assert_eq!(
        argv[argv.iter().position(|x| x == "--ipc").unwrap() + 1],
        "private"
    );
}

#[test]
fn test_existing_container_name_is_a_conflict_and_missing_image_not_found() {
    let w = World::new();
    let a = agent(&w, json!({}));
    assert_eq!(
        err(
            &a,
            "docker.run",
            json!({"spec": spec(json!({"name": "ec-owned-1"}))})
        )
        .0,
        CONFLICT
    );
    assert_eq!(
        err(
            &a,
            "docker.run",
            json!({"spec": spec(json!({"image": "vllm/vllm-openai:v9"}))})
        )
        .0,
        NOT_FOUND
    );
}

#[test]
fn test_engine_network_that_is_not_internal_is_refused() {
    let w = World::new();
    let a = agent(&w, json!({}));
    w.fb.update(json!({"networks": {"engine-console-engines": {"internal": false}}}));
    assert_eq!(err(&a, "net.ensure_internal", json!({})).0, CONFLICT);
    assert_eq!(
        err(&a, "docker.run", json!({"spec": spec(json!({}))})).0,
        CONFLICT
    );
    w.fb.update(json!({"networks": {}}));
    assert_eq!(
        Value::Object(ok(&a, "net.ensure_internal", json!({}))),
        json!({"network": "engine-console-engines"})
    );
}

// ── ownership ────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn test_only_console_owned_containers() {
    let w = World::new();
    let a = agent(&w, json!({}));
    for op in ["docker.stop", "docker.start", "docker.rm", "docker.logs"] {
        assert_eq!(
            err(&a, op, json!({"name": "other-engine"})).0,
            NOT_OWNED,
            "{op}"
        );
        assert_eq!(err(&a, op, json!({"name": "no-such"})).0, NOT_FOUND, "{op}");
        assert_eq!(err(&a, op, json!({"name": "--all"})).0, BAD_ARGS, "{op}");
    }
}

#[test]
fn test_role_and_instance_must_match_exactly() {
    let w = World::new();
    let a = agent(&w, json!({}));
    assert_eq!(
        err(
            &a,
            "docker.stop",
            json!({"name": "ec-owned-1", "role": "gateway"})
        )
        .0,
        NOT_OWNED
    );
    assert_eq!(
        err(
            &a,
            "docker.stop",
            json!({"name": "ec-owned-1", "instance": "inst_2"})
        )
        .0,
        NOT_OWNED
    );
    assert!(
        ok(
            &a,
            "docker.stop",
            json!({"name": "ec-owned-1", "instance": "inst_1", "timeout_s": 5})
        )
        .is_empty()
    );
    let stop =
        w.fb.argvs()
            .into_iter()
            .filter(|c| c.iter().any(|x| x == "stop"))
            .last()
            .unwrap();
    assert_eq!(
        stop[stop.len() - 3..],
        ["-t".to_string(), "5".into(), cid_engine()]
    ); // the verified id, not the name
}

#[test]
fn test_rm_removes_the_instance_files() {
    let w = World::new();
    let a = agent(&w, json!({}));
    ok(&a, "docker.run", json!({"spec": spec(json!({}))}));
    assert!(w.state().join("instances").join("inst_9").exists());
    ok(
        &a,
        "docker.rm",
        json!({"name": "ec-test-1", "instance": "inst_9"}),
    );
    assert!(!w.state().join("instances").join("inst_9").exists());
}

#[test]
fn test_logs_are_scrubbed_and_tail_clamped() {
    let w = World::new();
    let a = agent(&w, json!({}));
    let res = ok(
        &a,
        "docker.logs",
        json!({"name": "ec-owned-1", "tail": 999_999}),
    );
    let text = res["text"].as_str().unwrap();
    assert!(
        text.contains("INFO ready") && !text.contains(ARGV_SECRET) && res["truncated"] == false
    );
    let logs =
        w.fb.argvs()
            .into_iter()
            .filter(|c| c.iter().any(|x| x == "logs"))
            .last()
            .unwrap();
    assert_eq!(
        logs[logs.len() - 3..],
        ["--tail".to_string(), "5000".into(), cid_engine()]
    );
}

#[test]
fn test_inspect_is_sanitized() {
    let w = World::new();
    let a = agent(&w, json!({}));
    let info = ok(&a, "docker.inspect", json!({"name": "other-engine"}))["info"].clone();
    let blob = info.to_string();
    assert!(!blob.contains(ENV_VALUE) && !blob.contains("topsecretvalue"));
    assert_eq!(info["engine"], "sglang");
    assert_eq!(info["managed"], false);
    assert_eq!(info["compose_project"], "lab");
}

// ── images ───────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn test_images_and_presence() {
    let w = World::new();
    let a = agent(&w, json!({}));
    let imgs = ok(&a, "docker.images", json!({}));
    let refs: BTreeSet<&str> = imgs["images"]
        .as_array()
        .unwrap()
        .iter()
        .map(|i| i["ref"].as_str().unwrap())
        .collect();
    assert_eq!(
        refs,
        BTreeSet::from(["vllm/vllm-openai:v0.11.0", GATEWAY_IMAGE])
    );
    assert_eq!(
        ok(
            &a,
            "docker.image_present",
            json!({"image": "vllm/vllm-openai:v0.11.0"})
        )["present"],
        true
    );
    assert_eq!(
        Value::Object(ok(
            &a,
            "docker.image_present",
            json!({"image": "vllm/vllm-openai:nope"})
        )),
        json!({"present": false, "id": null})
    );
    assert_eq!(
        err(&a, "docker.image_present", json!({"image": "-x"})).0,
        BAD_ARGS
    );
}

#[test]
fn test_pull_rules_and_progress() {
    let w = World::new();
    let a = agent(&w, json!({"allow_pull": false}));
    assert_eq!(
        err(&a, "docker.pull", json!({"image": "vllm/vllm-openai:v1"})).0,
        POLICY_DENIED
    );
    let w = World::new();
    let a = agent(&w, json!({}));
    assert_eq!(
        err(
            &a,
            "docker.pull",
            json!({"image": "evil/vllm/vllm-openai:x"})
        )
        .0,
        POLICY_DENIED
    );
    let s = FakeStream::empty();
    let res = call(
        &a,
        "docker.pull",
        json!({"image": "vllm/vllm-openai:v1"}),
        Some(&s),
    )
    .unwrap();
    assert_eq!(res["id"], format!("sha256:{}", "ef".repeat(32)));
    assert!(
        s.events()
            .iter()
            .any(|e| e["line"].as_str().unwrap().contains("Downloaded"))
    );
}

// ── engines the console did not create ──────────────────────────────────────────────────────────────────
#[test]
fn test_external_control_rules() {
    let w = World::new();
    let a = agent(&w, json!({"external_control": "none"}));
    assert_eq!(
        err(
            &a,
            "docker.external",
            json!({"name": "other-engine", "action": "stop"})
        )
        .0,
        POLICY_DENIED
    );
    let scripts = json!([{"name": "sg", "containers": ["other-*"], "actions": [{"name": "up", "argv": ["/bin/true"]}]}]);
    let w = World::new();
    let a = agent(&w, json!({"scripts": scripts}));
    let (c, m) = err(
        &a,
        "docker.external",
        json!({"name": "other-engine", "action": "stop"}),
    );
    assert!(c == POLICY_DENIED && m.contains("script"));
    let w = World::new();
    let a = agent(&w, json!({}));
    assert_eq!(
        err(
            &a,
            "docker.external",
            json!({"name": "ec-owned-1", "action": "stop"})
        )
        .0,
        POLICY_DENIED
    );
    assert_eq!(
        err(
            &a,
            "docker.external",
            json!({"name": "grafana", "action": "stop"})
        )
        .0,
        POLICY_DENIED
    );
    assert_eq!(
        err(
            &a,
            "docker.external",
            json!({"name": "other-engine", "action": "kill"})
        )
        .0,
        BAD_ARGS
    );
    assert!(
        ok(
            &a,
            "docker.external",
            json!({"name": "other-engine", "action": "stop"})
        )
        .is_empty()
    );
    let stop =
        w.fb.argvs()
            .into_iter()
            .filter(|c| c.iter().any(|x| x == "stop"))
            .last()
            .unwrap();
    assert_eq!(stop.last().unwrap(), CID_EXT);
}

// ── gateway ──────────────────────────────────────────────────────────────────────────────────────────────
fn gw(over: Value) -> Value {
    let mut g = json!({"name": "ec-owned-1-gw", "instance": "inst_1", "host_port": 18005, "engine_host": "ec-owned-1",
                       "engine_port": 8000});
    for (k, v) in over.as_object().unwrap() {
        g[k] = v.clone();
    }
    g
}

fn port_free_is(a: &mut Agent, free: bool) {
    a.port_free = Arc::new(move |_| free);
}

#[test]
fn test_gateway_rules() {
    let w = World::new();
    let mut a = agent(&w, json!({}));
    port_free_is(&mut a, true);
    assert_eq!(
        err(&a, "gateway.start", gw(json!({"host_port": 18100}))).0,
        POLICY_DENIED
    );
    assert_eq!(
        err(&a, "gateway.start", gw(json!({"host_port": 8080}))).0,
        POLICY_DENIED
    );
    assert_eq!(
        err(
            &a,
            "gateway.start",
            gw(json!({"engine_host": "other-engine"}))
        )
        .0,
        NOT_OWNED
    );
    assert_eq!(
        err(&a, "gateway.start", gw(json!({"engine_host": "Bad;Host"}))).0,
        BAD_ARGS
    );
    assert_eq!(
        err(&a, "gateway.start", gw(json!({"name": "gw"}))).0,
        BAD_ARGS
    );
    port_free_is(&mut a, false);
    assert_eq!(err(&a, "gateway.start", gw(json!({}))).0, PORT_IN_USE);
}

#[test]
fn test_gateway_without_an_image_in_the_policy() {
    let w = World::new();
    let mut a = agent(&w, json!({"gateway_image": null}));
    port_free_is(&mut a, true);
    assert_eq!(err(&a, "gateway.start", gw(json!({}))).0, POLICY_DENIED);
}

#[test]
fn test_gateway_start_creates_connects_and_starts() {
    let w = World::new();
    let mut a = agent(&w, json!({}));
    port_free_is(&mut a, true);
    assert_eq!(
        ok(&a, "gateway.start", gw(json!({})))["id"]
            .as_str()
            .unwrap()
            .len(),
        64
    );
    let argvs = w.fb.argvs();
    let create = argvs
        .iter()
        .find(|a| a.iter().take(3).any(|x| x == "create"))
        .unwrap();
    let joined = create.join(" ");
    for frag in [
        "--network bridge",
        "--read-only",
        "--cap-drop ALL",
        "--user 101:101",
        "--pull never",
        "-p 127.0.0.1:18005:8080",
        "--label engine-console.role=gateway",
        "--label engine-console.instance=inst_1",
        GATEWAY_IMAGE,
    ] {
        assert!(joined.contains(frag), "{frag}");
    }
    let conf = create
        .iter()
        .find(|x| x.starts_with("NGINX_CONF="))
        .unwrap();
    assert!(conf.contains("daemon off;") && conf.contains("proxy_pass ec-owned-1:8000;"));
    assert!(argvs.iter().any(|a| a.get(2..5)
        == Some(&[
            "network".into(),
            "connect".into(),
            "engine-console-engines".into()
        ])));
    assert!(
        argvs
            .iter()
            .any(|a| a.get(2..4) == Some(&["start".into(), "ec-owned-1-gw".into()]))
    );
}

#[test]
fn test_gateway_is_removed_when_it_cannot_start() {
    let w = World::new();
    // docker.ok("network", "connect", ...) fails ("boom"); everything else reaches the fake docker
    let real = ProcRunner::new(w.fb.env());
    let failing = move |argv: &[String], timeout: f64, env: Option<&EnvMap>| -> CmdResult {
        if argv
            .windows(2)
            .any(|p| p[0] == "network" && p[1] == "connect")
        {
            return CmdResult::new(1, "", "boom");
        }
        real.run(argv, timeout, env)
    };
    let mut parts = w.parts(w.fb.env());
    parts.runner = Arc::new(failing);
    let mut a = w.agent_with(Some(&manage_policy(&w.machine, json!({}))), parts);
    a.hello(false);
    port_free_is(&mut a, true);
    assert_eq!(err(&a, "gateway.start", gw(json!({}))).0, DOCKER_ERROR);
    assert!(
        w.fb.argvs()
            .iter()
            .any(|c| c.get(2..5) == Some(&["rm".into(), "-f".into(), "ec-owned-1-gw".into()]))
    );
}

#[test]
fn test_ulimits_are_allowlisted_validated_and_rendered() {
    let w = World::new();
    let a = agent(&w, json!({}));
    let argv = argv_of(&ok(
        &a,
        "docker.render",
        json!({"spec": spec(json!({"ulimits": {"stack": "67108864", "nofile": "65536:65536"}}))}),
    ));
    let joined = argv.join(" ");
    assert!(
        joined.contains("--ulimit stack=67108864")
            && joined.contains("--ulimit nofile=65536:65536")
    );
    assert_eq!(
        err(
            &a,
            "docker.render",
            json!({"spec": spec(json!({"ulimits": {"memlock": "-1"}}))})
        )
        .0,
        BAD_ARGS
    );
    assert_eq!(
        err(
            &a,
            "docker.render",
            json!({"spec": spec(json!({"ulimits": {"stack": "lots"}}))})
        )
        .0,
        BAD_ARGS
    );
    assert!(
        !argv_of(&ok(&a, "docker.render", json!({"spec": spec(json!({}))})))
            .join(" ")
            .contains("--ulimit")
    );
}

/// security review 2026-10-04 M5: an HF_HOME root holds `token` and `stored_tokens` next to `hub/`.
#[test]
fn test_m5_a_mount_that_would_show_the_container_a_token_file_is_refused() {
    let w = World::new();
    let a = agent(&w, json!({}));
    let hf = w.machine.data.join("hf");
    std::fs::create_dir_all(hf.join("hub").join("models--a--b").join("blobs")).unwrap();
    let rendered = argv_of(&ok(&a, "docker.render", json!({"spec": spec(json!({}))}))); // a cache without tokens mounts
    assert!(rendered.contains(&format!("{}:/root/.cache/huggingface:ro", hf.display())));
    std::fs::write(hf.join("token"), "x").unwrap();
    let (c, m) = err(&a, "docker.render", json!({"spec": spec(json!({}))}));
    assert!(c == POLICY_DENIED && m.contains("'token'"), "{m}");
    std::fs::remove_file(hf.join("token")).unwrap();
    std::fs::create_dir(hf.join("sub")).unwrap();
    std::fs::write(hf.join("sub").join("stored_tokens"), "x").unwrap(); // one level down is checked too
    let (c, m) = err(&a, "docker.render", json!({"spec": spec(json!({}))}));
    assert!(c == POLICY_DENIED && m.contains("sub/stored_tokens"), "{m}");
    // the hub/ folder itself (an hf-hub root on .../hub) still mounts
    let argv = argv_of(&ok(
        &a,
        "docker.render",
        json!({"spec": spec(json!({"mounts": [{"root": "hf", "rel": "hub", "dest": "/root/.cache/huggingface/hub"}]}))}),
    ));
    assert!(argv.contains(&format!(
        "{}:/root/.cache/huggingface/hub:ro",
        hf.join("hub").display()
    )));
}

/// security re-audit 2026-10-04: `a/b/token` passed the old two-level scan; model trees must not trip it, and a
/// source too big to check is refused rather than mounted unchecked.
#[test]
fn test_m5_the_secret_scan_reaches_deeper_and_skips_model_trees() {
    let w = World::new();
    let mut a = agent(&w, json!({}));
    let hf = w.machine.data.join("hf");
    let snap = hf
        .join("hub")
        .join("models--a--b")
        .join("snapshots")
        .join("abc");
    std::fs::create_dir_all(&snap).unwrap();
    for name in [
        "tokenizer.json",
        "special_tokens_map.json",
        "added_tokens.json",
    ] {
        std::fs::write(snap.join(name), "{}").unwrap();
    }
    std::fs::write(snap.join("token"), "x").unwrap(); // inside a model tree: weights and configs only, never walked
    ok(&a, "docker.render", json!({"spec": spec(json!({}))}));
    std::fs::create_dir_all(hf.join("creds").join("hf")).unwrap();
    std::fs::write(hf.join("creds").join("hf").join("hf_token.txt"), "x").unwrap(); // three levels down
    let (c, m) = err(&a, "docker.render", json!({"spec": spec(json!({}))}));
    assert!(
        c == POLICY_DENIED && m.contains("creds/hf/hf_token.txt"),
        "{m}"
    );
    std::fs::remove_file(hf.join("creds").join("hf").join("hf_token.txt")).unwrap();
    ok(&a, "docker.render", json!({"spec": spec(json!({}))}));
    a.secret_scan_max_entries = 3;
    let (c, m) = err(&a, "docker.render", json!({"spec": spec(json!({}))}));
    assert!(c == POLICY_DENIED && m.contains("too many"), "{m}");
}
