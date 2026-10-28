//! The cli.py half of node/tests/test_cli.py: `ecnode check` and `ecnode policy-template` (cli.py). The
//! `__main__.py` half is tests/test_cli.rs.
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
use std::path::Path;
use std::sync::Arc;

use ecnode::agent::{Agent, VERSION};
use ecnode::cli::{candidate_roots, check_to, detect_context, policy_template_to};
use ecnode::gpus::Gpus;
use ecnode::policy::{LoadOptions, PolicyStore, parse};
use ecnode::util::{EnvMap, ProcRunner, Runner};
use serde_json::{Value, json};
use wsd_agent::World;
use wsd_kit::{ARGV_SECRET, ENV_VALUE, GATEWAY_IMAGE, manage_policy, write_policy};

fn run_check(a: &Agent) -> (i32, String) {
    let mut out = Vec::new();
    let code = check_to(a, &mut out);
    (code, String::from_utf8(out).unwrap())
}

#[test]
fn test_check_prints_a_readable_summary_without_secrets() {
    let w = World::new();
    let a = w.agent(Some(&manage_policy(&w.machine, json!({"mode": "monitor"}))));
    let (code, text) = run_check(&a);
    assert_eq!(code, 0, "{text}");
    for frag in [
        format!("ecnode {VERSION}"),
        "Test Linux 1.0".into(),
        "GPUs     2 via nvidia-smi".into(),
        "NVIDIA Test GPU".into(),
        "docker   29.0.0 · context testctx · rootless yes · nvidia runtime yes".into(),
        "mode MONITOR".into(),
        format!(
            "hf         hf-home  {}",
            w.machine.data.join("hf").display()
        ),
        "Data Drive 8TB".into(),
        "from 10.0.0.9:/export/models".into(),
        "running  3 containers (2 engines, 1 console-owned)".into(),
    ] {
        assert!(text.contains(&frag), "{frag}\n{text}");
    }
    assert!(!text.contains(ENV_VALUE) && !text.contains(ARGV_SECRET));
    // the header names this implementation; there is no Python on a Rust node (PORTING.md DV5, DV24)
    let first = text.lines().next().unwrap_or_default();
    assert!(
        first.starts_with(&format!("ecnode {VERSION} (rust) · protocol 1 · node id ")),
        "{first}"
    );
    assert!(!first.contains("Python"), "{first}");
}

#[test]
fn test_check_flags_a_refused_policy() {
    let w = World::new();
    let path = write_policy(
        &w.path().join("p").join("policy.json"),
        &manage_policy(&w.machine, json!({})),
        0o666,
    );
    let opts = LoadOptions::new(
        Some(path.to_str().unwrap()),
        Some(w.path().to_str().unwrap()),
        w.path().join("none").to_str().unwrap(),
    );
    let store = PolicyStore::with_options(opts, 2.0, ecnode::policy::monotonic());
    let a = Agent::with_parts(store, w.parts(w.fb.env()));
    let (code, text) = run_check(&a);
    assert_eq!(code, 1);
    assert!(
        text.contains("! policy file") && text.contains("writable"),
        "{text}"
    );
    // Python probed the default ports here (8000, 8080, 11434, 30000); a node without a usable policy probes none
    // (security audit F1, a deliberate deviation)
    let tried: BTreeSet<u16> = w.prober.tried().into_iter().collect();
    assert_eq!(tried, BTreeSet::new());
}

#[test]
fn test_candidate_roots_point_inside_caches_and_ollama_homes() {
    let tmp = tempfile::tempdir().unwrap();
    let home = tmp.path().join("home");
    std::fs::create_dir_all(home.join(".cache").join("huggingface").join("hub")).unwrap();
    std::fs::create_dir_all(home.join(".ollama").join("models").join("manifests")).unwrap();
    std::fs::create_dir(home.join("models")).unwrap();
    // only directories below this test's home exist for it (never the real /models, /data/...)
    let h = home.clone();
    let isdir = move |p: &str| Path::new(p).starts_with(&h) && Path::new(p).is_dir();
    let roots = candidate_roots(&EnvMap::new(), Some(home.to_str().unwrap()), &isdir);
    let got: Vec<(String, String, String)> = roots
        .iter()
        .map(|r| {
            (
                r["id"].as_str().unwrap().into(),
                r["kind"].as_str().unwrap().into(),
                r["path"].as_str().unwrap().into(),
            )
        })
        .collect();
    let s = |p: &Path| p.to_string_lossy().into_owned();
    assert_eq!(
        got,
        vec![
            (
                "hf".into(),
                "hf-hub".into(),
                s(&home.join(".cache").join("huggingface").join("hub"))
            ),
            (
                "ollama".into(),
                "ollama".into(),
                s(&home.join(".ollama").join("models"))
            ),
            ("models".into(), "auto".into(), s(&home.join("models"))),
        ]
    );
    assert!(roots.iter().all(|r| r["writable"] == false));
}

fn template(w: &World, home: &Path) -> Value {
    let runner: Arc<dyn Runner> = Arc::new(ProcRunner::new(w.fb.env()));
    let gpus = Gpus::new(false, Arc::clone(&runner));
    let mut out = Vec::new();
    assert_eq!(
        policy_template_to(
            &mut out,
            &runner,
            &gpus,
            &EnvMap::new(),
            Some(home.to_str().unwrap())
        ),
        0
    );
    serde_json::from_slice(&out).unwrap()
}

#[test]
fn test_policy_template_is_a_valid_monitor_policy() {
    let w = World::new();
    let home = w.path().join("tmplhome");
    std::fs::create_dir_all(home.join(".cache").join("huggingface").join("hub")).unwrap();
    let data = template(&w, &home);
    let p = parse(&data, None, Some(home.to_str().unwrap())).unwrap();
    assert_eq!(p.mode, "monitor");
    assert_eq!(p.docker_context.as_deref(), Some("testctx"));
    assert_eq!(p.gateway_image.as_deref(), Some(GATEWAY_IMAGE));
    let ids: Vec<&str> = p.model_roots.iter().map(|r| r.id.as_str()).collect();
    assert_eq!(ids, vec!["hf"]);
    assert_eq!(data["_detected_gpus"].as_array().unwrap().len(), 2);
    assert!(!w.path().join("tmplhome").join("policy.json").exists()); // it writes nothing
}

#[test]
fn test_template_and_examples_prefer_digest_pins() {
    // 2026-10-04 review, L7: no exact reference pinned by a movable tag, no placeholder digest, and a note saying why.
    let w = World::new();
    let mut docs: Vec<(String, Value)> = vec![("template".into(), template(&w, w.path()))];
    let ex = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../node/examples");
    let mut names: Vec<String> = std::fs::read_dir(&ex)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().into_owned()))
        .filter(|n| n.starts_with("policy.") && n.ends_with(".json"))
        .collect();
    names.sort();
    for n in names {
        docs.push((
            n.clone(),
            serde_json::from_str(&std::fs::read_to_string(ex.join(&n)).unwrap()).unwrap(),
        ));
    }
    assert_eq!(docs.len(), 3);
    for (name, data) in &docs {
        parse(data, None, Some(w.path().to_str().unwrap())).unwrap();
        assert!(
            data["_engine_images"]
                .as_str()
                .unwrap()
                .contains("@sha256:"),
            "{name}"
        );
        let mut refs: Vec<Value> = data["engine_images"].as_array().unwrap().clone();
        refs.push(data["gateway_image"].clone());
        for r in refs {
            let Some(r) = r.as_str() else { continue };
            if r.ends_with(":*") || r.ends_with("@sha256:*") {
                continue; // a pattern, not a pin
            }
            let digest = r.split_once("@sha256:").map_or("", |(_, d)| d);
            assert_eq!(digest.len(), 64, "{name} {r}");
            assert!(name == "template" || digest != "0".repeat(64), "{name} {r}"); // the template's is the fake docker's
        }
        assert!(
            !data["engine_images"]
                .as_array()
                .unwrap()
                .contains(&json!("ghcr.io/ggml-org/llama.cpp:*")),
            "{name}"
        );
    }
}

#[test]
fn test_template_prefers_an_existing_rootless_context() {
    let w = World::new();
    let runner = ProcRunner::new(w.fb.env());
    w.fb.update(json!({"context": "default", "contexts": ["default", "rootless"]}));
    assert_eq!(detect_context(&runner).as_deref(), Some("rootless"));
    w.fb.update(json!({"context": "default", "contexts": ["default"]}));
    assert_eq!(detect_context(&runner), None);
    w.fb.update(
        json!({"context": "lab-remote", "contexts": ["default", "lab-remote", "rootless"]}),
    );
    assert_eq!(detect_context(&runner).as_deref(), Some("lab-remote"));
}
