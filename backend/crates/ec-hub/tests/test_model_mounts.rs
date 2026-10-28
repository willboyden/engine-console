//! Port of backend/tests/core/test_model_mounts.py, the lifecycle half: what each model format mounts into the
//! engine's container through a real launch, and the launch context the adapter receives. The pure-planning half
//! (layouts, mounts, refs, display names, env split, labels) is in services/runspec.rs's unit tests.

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

use std::sync::{Arc, Mutex};

use ec_hub::adapters::base::{
    Compat, EngineAdapter, FloatMap, Hardware, LaunchContext, LaunchSpec, ModelInfo, Obj, ParamSpec,
};
use ec_hub::domain::errors::Problem;
use ec_hub::domain::models::{Instance, InstanceCreate};
use ec_hub::domain::refs::ModelRef;
use fakefleet::{CopyOpts, FleetEnv};
use kit::{FakeAdapter, FakeMultiAdapter};
use serde_json::{Value, json};
use wse_kit::{codes, container, run_specs};

const HF: &str = "/root/.cache/huggingface";

fn gguf(rel: &str) -> ModelRef {
    gguf_in(rel, "gguf")
}

fn gguf_in(rel: &str, root: &str) -> ModelRef {
    ModelRef {
        format: "gguf".into(),
        repo_id: None,
        root_id: Some(root.into()),
        rel: Some(rel.into()),
        name: None,
    }
}

fn req(engine: &str, r: &ModelRef, extra: Value) -> InstanceCreate {
    let mut b = json!({"engine": engine, "model": r});
    for (k, v) in extra.as_object().cloned().unwrap_or_default() {
        b[k] = v;
    }
    serde_json::from_value(b).unwrap()
}

/// `start(env, ref, engine=..., **kw)`
async fn start(
    env: &mut FleetEnv,
    r: &ModelRef,
    engine: &str,
    extra: Value,
) -> Result<Instance, ec_hub::domain::errors::AppError> {
    env.container()
        .lifecycle
        .create(&req(engine, r, extra), None)
        .await
}

fn multi_env() -> FleetEnv {
    let mut env = FleetEnv::new();
    env.adapters.push(Arc::new(FakeMultiAdapter));
    env
}

/// A FakeMultiAdapter under another id whose launch spec is changed by `edit` (Python's subclasses in this file).
struct Variant {
    id: &'static str,
    edit: fn(LaunchSpec) -> LaunchSpec,
    seen: Arc<Mutex<Vec<LaunchContext>>>,
}

impl EngineAdapter for Variant {
    fn id(&self) -> &str {
        self.id
    }
    fn display_name(&self) -> &str {
        "Fake C++ Engine"
    }
    fn default_image(&self) -> &str {
        "fake/engine:1.0"
    }
    fn model_formats(&self) -> Vec<String> {
        FakeMultiAdapter.model_formats()
    }
    fn images_by_arch(&self) -> Vec<(String, String)> {
        FakeMultiAdapter.images_by_arch()
    }
    fn build_launch_ctx(&self, ctx: &LaunchContext) -> Result<LaunchSpec, Problem> {
        self.seen.lock().unwrap().push(ctx.clone());
        FakeMultiAdapter.build_launch_ctx(ctx).map(self.edit)
    }
    fn param_catalog(&self) -> Vec<ParamSpec> {
        FakeAdapter.param_catalog()
    }
    fn presets(&self) -> Vec<(String, Obj)> {
        FakeAdapter.presets()
    }
    fn validate(&self, params: &Obj) -> Vec<String> {
        FakeAdapter.validate(params)
    }
    fn build_launch(
        &self,
        m: &str,
        p: &Obj,
        hw: &Hardware,
        s: &str,
        h: &str,
    ) -> Result<LaunchSpec, Problem> {
        FakeAdapter.build_launch(m, p, hw, s, h)
    }
    fn compatibility(&self, m: &ModelInfo, p: &Obj, hw: &Hardware) -> Vec<Compat> {
        FakeAdapter.compatibility(m, p, hw)
    }
    fn memory_model(&self, m: &ModelInfo, p: &Obj) -> FloatMap {
        FakeAdapter.memory_model(m, p)
    }
    fn parse_metrics(&self, text: &str) -> FloatMap {
        FakeAdapter.parse_metrics(text)
    }
    fn parse_startup_log(&self, line: &str) -> Option<Obj> {
        FakeAdapter.parse_startup_log(line)
    }
}

fn variant(
    id: &'static str,
    edit: fn(LaunchSpec) -> LaunchSpec,
) -> (Arc<dyn EngineAdapter>, Arc<Mutex<Vec<LaunchContext>>>) {
    let seen = Arc::new(Mutex::new(Vec::new()));
    (
        Arc::new(Variant {
            id,
            edit,
            seen: Arc::clone(&seen),
        }),
        seen,
    )
}

// ---- through the lifecycle --------------------------------------------------------------------------------------------------------------
#[tokio::test]
async fn test_gguf_launch_mounts_the_directory_and_passes_the_file_as_model_path() {
    let mut env = multi_env();
    let r = gguf("unsloth/m/m-Q4.gguf");
    env.library.add(
        "local",
        &r,
        CopyOpts {
            size_bytes: 5 * (1 << 30),
            ..CopyOpts::new("gguf", "unsloth/m/m-Q4.gguf")
        },
    );
    let inst = start(&mut env, &r, "fakecpp", json!({})).await.unwrap();
    let spec = run_specs(&env.node)[0].clone();
    assert_eq!(
        spec["mounts"],
        json!([{"root": "gguf", "rel": "unsloth/m", "dest": "/models", "ro": true}])
    ); // no HF cache for gguf
    assert_eq!(
        spec["argv"].as_array().unwrap()[..4],
        [
            json!("--model"),
            json!("/models/m-Q4.gguf"),
            json!("--alias"),
            json!("m-Q4")
        ]
    );
    assert_eq!(spec["env"]["FAKE_FORMAT"], "gguf");
    assert!(inst.model == Some(r.clone()) && inst.repo_id == "m-Q4" && inst.name == "m-Q4");
    assert!(
        spec["labels"]["engine-console.format"] == "gguf"
            && spec["labels"]["engine-console.rel"] == "unsloth/m/m-Q4.gguf"
    );
    assert_eq!(
        inst.model_location.unwrap().path,
        "/data/gguf/unsloth/m/m-Q4.gguf"
    );
    let c = container(&env.node, inst.container_name.as_deref().unwrap()).unwrap();
    assert_eq!(c["mounts"][0]["source"], "/data/gguf/unsloth/m");
}

#[tokio::test]
async fn test_gguf_at_the_top_of_a_root_mounts_the_root_itself() {
    let mut env = multi_env();
    let r = gguf("top.gguf");
    env.library
        .add("local", &r, CopyOpts::new("gguf", "top.gguf"));
    start(&mut env, &r, "fakecpp", json!({})).await.unwrap();
    assert_eq!(
        run_specs(&env.node)[0]["mounts"],
        json!([{"root": "gguf", "rel": "", "dest": "/models", "ro": true}])
    );
}

#[tokio::test]
async fn test_plain_model_directory_launch() {
    let mut env = multi_env();
    let r = ModelRef {
        format: "plain".into(),
        repo_id: None,
        root_id: Some("gguf".into()),
        rel: Some("org/my-model".into()),
        name: None,
    };
    env.library
        .add("local", &r, CopyOpts::new("gguf", "org/my-model"));
    start(&mut env, &r, "fakecpp", json!({})).await.unwrap();
    let spec = run_specs(&env.node)[0].clone();
    assert_eq!(
        spec["mounts"],
        json!([{"root": "gguf", "rel": "org/my-model", "dest": "/models", "ro": true}])
    );
    assert_eq!(
        spec["argv"].as_array().unwrap()[..2],
        [json!("--model"), json!("/models")]
    );
}

/// The 2 parametrised ids of test_ollama_launch_mounts_the_stores_models_dir_read_only.
#[tokio::test]
async fn test_ollama_launch_mounts_the_stores_models_dir_read_only() {
    for (copy_rel, root_path, rel) in [
        (
            "manifests/registry.ollama.ai/library/llama3/8b",
            "/data/ollama/models",
            "",
        ),
        (
            "models/manifests/registry.ollama.ai/library/llama3/8b",
            "/data/ollama",
            "models",
        ),
    ] {
        let mut env = multi_env();
        let r = ModelRef {
            format: "ollama".into(),
            repo_id: None,
            root_id: None,
            rel: None,
            name: Some("llama3:8b".into()),
        };
        let mut roots = env.node.policy()["model_roots"].clone();
        roots[2]["path"] = json!(root_path);
        env.node.set_policy("model_roots", roots);
        env.library
            .add("local", &r, CopyOpts::new("ollama", copy_rel));
        let inst = start(&mut env, &r, "fakecpp", json!({})).await.unwrap();
        let spec = run_specs(&env.node)[0].clone();
        assert_eq!(
            spec["mounts"],
            json!([{"root": "ollama", "rel": rel, "dest": "/root/.ollama/models", "ro": true}])
        );
        assert_eq!(
            spec["argv"].as_array().unwrap()[..2],
            [json!("--model"), json!("llama3:8b")]
        );
        assert_eq!(inst.repo_id, "llama3:8b");
    }
}

#[tokio::test]
async fn test_a_named_ollama_model_must_be_in_the_library_but_a_bare_server_need_not() {
    let mut env = multi_env();
    let named = ModelRef {
        format: "ollama".into(),
        repo_id: None,
        root_id: None,
        rel: None,
        name: Some("llama3:8b".into()),
    };
    let e = start(&mut env, &named, "fakecpp", json!({}))
        .await
        .unwrap_err();
    assert!(codes(&e).contains_key("model_missing"));
    assert!(env.node.ops("docker.run").is_empty());
    let bare = ModelRef {
        format: "ollama".into(),
        repo_id: None,
        root_id: None,
        rel: None,
        name: None,
    };
    let inst = start(&mut env, &bare, "fakecpp", json!({})).await.unwrap();
    assert_eq!(inst.repo_id, "ollama");
    assert_eq!(
        run_specs(&env.node)[0]["mounts"],
        json!([{"root": "ollama", "rel": "models", "dest": "/root/.ollama/models", "ro": true}])
    );
}

#[tokio::test]
async fn test_an_hf_root_in_hub_layout_is_mounted_under_the_hub_folder() {
    let mut env = FleetEnv::new();
    let mut roots = env.node.policy()["model_roots"].clone();
    roots[0]["kind"] = json!("hf-hub");
    roots[0]["path"] = json!("/data/hub-cache");
    env.node.set_policy("model_roots", roots);
    let cfg = json!({"num_hidden_layers": 2, "hidden_size": 64, "num_attention_heads": 2, "num_key_value_heads": 1});
    env.library.add_hf(
        "local",
        "Qwen/Qwen3-32B",
        &cfg,
        65_530_000_000,
        "hf",
        Some("models--Qwen--Qwen3-32B"),
        None,
    );
    let r: InstanceCreate = serde_json::from_value(
        json!({"engine": "fake", "repo_id": "Qwen/Qwen3-32B", "params": {"max_model_len": 512}}),
    )
    .unwrap();
    let inst = env.container().lifecycle.create(&r, None).await.unwrap();
    assert_eq!(
        run_specs(&env.node)[0]["mounts"],
        json!([{"root": "hf", "rel": "", "dest": format!("{HF}/hub"), "ro": true}])
    );
    let c = container(&env.node, inst.container_name.as_deref().unwrap()).unwrap();
    assert_eq!(c["mounts"][0]["source"], "/data/hub-cache");
}

/// Security control (H3): the HF cache mount is read-only even on a writable root.
#[tokio::test]
async fn test_the_hf_mount_is_read_only_even_on_a_writable_root() {
    let mut env = FleetEnv::new();
    assert_eq!(env.node.policy()["model_roots"][0]["writable"], json!(true));
    wse_kit::seed(&env, "local");
    let r: InstanceCreate = serde_json::from_value(
        json!({"engine": "fake", "repo_id": "Qwen/Qwen3-32B", "params": {"max_model_len": 4096}}),
    )
    .unwrap();
    env.container().lifecycle.create(&r, None).await.unwrap();
    assert!(
        run_specs(&env.node)[0]["mounts"]
            .as_array()
            .unwrap()
            .iter()
            .all(|m| m["ro"] == json!(true))
    );
}

#[tokio::test]
async fn test_an_adapter_that_wants_the_hf_cache_next_to_a_gguf_gets_it_when_the_host_has_one() {
    let mut env = FleetEnv::new();
    let (a, _) = variant("wantscache", |mut l| {
        l.needs_hf_cache = true;
        l
    });
    env.adapters.push(a);
    let r = gguf("a/m.gguf");
    env.library
        .add("local", &r, CopyOpts::new("gguf", "a/m.gguf"));
    start(&mut env, &r, "wantscache", json!({})).await.unwrap();
    let node = Arc::clone(&env.node);
    let dests = move |i: usize| -> Vec<Value> {
        run_specs(&node)[i]["mounts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["dest"].clone())
            .collect()
    };
    assert_eq!(dests(0), [json!("/models"), json!(format!("{HF}/hub"))]);
    let roots: Vec<Value> = env.node.policy()["model_roots"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|r| r["kind"] != "hf-home")
        .cloned()
        .collect();
    env.node.set_policy("model_roots", Value::Array(roots));
    start(&mut env, &r, "wantscache", json!({"name": "two"}))
        .await
        .unwrap();
    assert_eq!(dests(1), [json!("/models")]); // no HF root on this host: just skip it
}

#[tokio::test]
async fn test_a_model_on_shared_storage_is_loaded_from_the_copy_the_library_finds_for_that_host() {
    let mut env = multi_env();
    let r = gguf_in("a/m.gguf", "shared");
    let mut roots = env.node.policy()["model_roots"]
        .as_array()
        .cloned()
        .unwrap();
    roots.push(json!({"id": "shared", "path": "/mnt/shared", "kind": "gguf", "writable": false, "label": "Shared"}));
    env.node.set_policy("model_roots", Value::Array(roots));
    env.library
        .add("local", &r, CopyOpts::new("shared", "a/m.gguf"));
    start(&mut env, &r, "fakecpp", json!({})).await.unwrap();
    assert_eq!(
        run_specs(&env.node)[0]["mounts"],
        json!([{"root": "shared", "rel": "a", "dest": "/models", "ro": true}])
    );
}

#[tokio::test]
async fn test_an_adapter_that_cannot_load_the_format_is_refused_at_build_time_too() {
    let mut env = multi_env();
    let r = gguf("a/m.gguf");
    env.library
        .add("local", &r, CopyOpts::new("gguf", "a/m.gguf"));
    let e = start(&mut env, &r, "fake", json!({})).await.unwrap_err();
    assert!(codes(&e).contains_key("format_unsupported"));
    assert!(env.node.ops("docker.run").is_empty());
}

/// Security control: hostile model refs are 422 before any check or node call.
#[tokio::test]
async fn test_hostile_model_refs_are_422_before_any_check_or_call() {
    let mut env = multi_env();
    let c = env.container();
    for r in [
        gguf("../../etc/passwd.gguf"),
        ModelRef {
            format: "plain".into(),
            repo_id: None,
            root_id: Some("x y".into()),
            rel: Some("a".into()),
            name: None,
        },
    ] {
        let e = c
            .lifecycle
            .preflight_full(&req("fakecpp", &r, json!({})), None, None)
            .await
            .unwrap_err();
        assert_eq!(e.status(), 422);
    }
    assert!(env.node.calls().is_empty());
}

#[tokio::test]
async fn test_adapters_are_asked_through_the_v2_entry_point_with_a_complete_context() {
    let mut env = FleetEnv::new();
    let (a, seen) = variant("spy", |l| l);
    env.adapters.push(a);
    let r = gguf("a/m.gguf");
    env.library
        .add("local", &r, CopyOpts::new("gguf", "a/m.gguf"));
    start(
        &mut env,
        &r,
        "spy",
        json!({"params": {"max_model_len": 4096}}),
    )
    .await
    .unwrap();
    let ctx = seen.lock().unwrap().last().cloned().unwrap();
    assert_eq!(
        (
            ctx.model_format.as_str(),
            ctx.model_path.as_deref(),
            ctx.served_name.as_str(),
            ctx.hf_cache_container_path.as_str()
        ),
        ("gguf", Some("/models/m.gguf"), "m", HF)
    );
    assert!(
        ctx.hw.host_id == "local"
            && ctx.hw.arch == "x86_64"
            && Value::Object(ctx.params) == json!({"max_model_len": 4096})
    );
}

#[tokio::test]
async fn test_generated_config_files_travel_in_the_run_spec_and_are_validated_against_the_nodes_limits()
 {
    let mut env = FleetEnv::new();
    let (with_files, _) = variant("withfiles", |mut l| {
        l.files = json!({"/etc/engine/models.yaml": "models: []\n"})
            .as_object()
            .cloned()
            .unwrap();
        l
    });
    let (too_many, _) = variant("toomany", |mut l| {
        l.files = (0..5).map(|i| (format!("/etc/f{i}"), json!("x"))).collect();
        l
    });
    env.adapters.push(with_files);
    env.adapters.push(too_many);
    let r = gguf("a/m.gguf");
    env.library
        .add("local", &r, CopyOpts::new("gguf", "a/m.gguf"));
    start(&mut env, &r, "withfiles", json!({})).await.unwrap();
    assert_eq!(
        run_specs(&env.node)[0]["files"],
        json!({"/etc/engine/models.yaml": "models: []\n"})
    );
    let e = start(&mut env, &r, "toomany", json!({"name": "two"}))
        .await
        .unwrap_err();
    assert!(codes(&e).contains_key("invalid_launch"));
    assert_eq!(run_specs(&env.node).len(), 1);
}
