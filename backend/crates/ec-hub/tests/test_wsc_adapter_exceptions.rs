//! Adapter exceptions in the fit service (PORTING.md §A5). Rust-native: Python catches an exception from
//! the fit estimate in two places, the lifecycle's `_resolve_gpus` ("an adapter's memory model may not cope with an
//! unknown model: use the request") and the `/models` listing ("one odd model must not blank the list"); both need
//! the exception back as a value, never as a panic. On `/fit` it is Python's catch-all 500.

#![allow(clippy::panic, clippy::expect_used, clippy::unwrap_used)]

#[path = "common/kit.rs"]
mod kit;

use std::sync::Arc;

use ec_hub::adapters::base::{
    Compat, EngineAdapter, FloatMap, Hardware, LaunchSpec, ModelInfo, Obj, ParamSpec,
};
use ec_hub::adapters::pycompat::{self, raise};
use ec_hub::domain::errors::{AppError, Problem};
use kit::{FakeAdapter, TestEnv};
use serde_json::json;

const P: &str = "/api/v1";

/// The fake engine, raising a ZeroDivisionError from the methods named in `raises`.
struct Raising {
    raises: &'static [&'static str],
}

impl Raising {
    fn check(&self, method: &str) {
        if self.raises.contains(&method) {
            raise("ZeroDivisionError", &format!("{method}: division by zero"));
        }
    }
}

impl EngineAdapter for Raising {
    fn id(&self) -> &str {
        "raising"
    }
    fn display_name(&self) -> &str {
        "Raising"
    }
    fn default_image(&self) -> &str {
        FakeAdapter.default_image()
    }
    fn param_catalog(&self) -> Vec<ParamSpec> {
        FakeAdapter.param_catalog()
    }
    fn presets(&self) -> Vec<(String, Obj)> {
        FakeAdapter.presets()
    }
    fn validate(&self, params: &Obj) -> Vec<String> {
        self.check("validate");
        FakeAdapter.validate(params)
    }
    fn build_launch(
        &self,
        model: &str,
        params: &Obj,
        hw: &Hardware,
        served_name: &str,
        hf_cache_container_path: &str,
    ) -> Result<LaunchSpec, Problem> {
        FakeAdapter.build_launch(model, params, hw, served_name, hf_cache_container_path)
    }
    fn compatibility(&self, model: &ModelInfo, params: &Obj, hw: &Hardware) -> Vec<Compat> {
        self.check("compatibility");
        FakeAdapter.compatibility(model, params, hw)
    }
    fn memory_model(&self, model: &ModelInfo, params: &Obj) -> FloatMap {
        self.check("memory_model");
        FakeAdapter.memory_model(model, params)
    }
    fn host_memory_gib(&self, model: &ModelInfo, params: &Obj, hw: &Hardware) -> FloatMap {
        self.check("host_memory_gib");
        FakeAdapter.host_memory_gib(model, params, hw)
    }
    fn parse_metrics(&self, prometheus_text: &str) -> FloatMap {
        FakeAdapter.parse_metrics(prometheus_text)
    }
    fn parse_startup_log(&self, line: &str) -> Option<Obj> {
        FakeAdapter.parse_startup_log(line)
    }
}

fn env_with(raises: &'static [&'static str]) -> TestEnv {
    let mut env = TestEnv::new();
    env.adapters = vec![Arc::new(FakeAdapter), Arc::new(Raising { raises })];
    env
}

fn model() -> ModelInfo {
    serde_json::from_value(json!({"repo_id": "Qwen/Qwen3-32B", "num_layers": 64, "hidden_size": 5120,
                                  "num_attention_heads": 64, "num_kv_heads": 8, "head_dim": 128,
                                  "max_position_embeddings": 40960, "weight_bytes": 65_000_000_000_i64}))
    .unwrap()
}

fn is_generic_500(e: &AppError) -> bool {
    matches!(e, AppError::Problem(p) if p.status == 500 && p.code == "internal_error" && p.detail == "unexpected server error")
}

/// `_resolve_gpus` catches any exception from `resolve_gpus`: the Rust caller needs an `Err` to fall back on.
#[tokio::test]
async fn test_resolve_gpus_returns_a_raising_memory_model_as_err() {
    pycompat::install_quiet_panic_hook();
    let mut env = env_with(&["memory_model"]);
    let fit = Arc::clone(&env.container().fit);
    let err = fit
        .resolve_gpus("raising", &model(), &Obj::new(), &[], "local")
        .unwrap_err();
    assert!(is_generic_500(&err), "{err:?}");
    // GPUs the request names never reach the adapter
    assert_eq!(
        fit.resolve_gpus("raising", &model(), &Obj::new(), &[1], "local")
            .unwrap(),
        [1]
    );
    // the other engine is unaffected
    assert!(
        fit.resolve_gpus("fake", &model(), &Obj::new(), &[], "local")
            .is_ok()
    );
}

/// The `/models` listing skips an engine whose estimate raises: assess_info must hand the exception back.
#[tokio::test]
async fn test_assess_info_returns_each_raising_adapter_method_as_err() {
    pycompat::install_quiet_panic_hook();
    for method in [
        &["validate"][..],
        &["memory_model"],
        &["compatibility"],
        &["host_memory_gib"],
    ] {
        let mut env = env_with(method);
        let fit = Arc::clone(&env.container().fit);
        let err = fit
            .assess_info("raising", &model(), &Obj::new(), &[0], 1, "local")
            .unwrap_err();
        assert!(is_generic_500(&err), "{method:?}: {err:?}");
        let ok = fit
            .assess_info("fake", &model(), &Obj::new(), &[0], 1, "local")
            .unwrap();
        assert!(!ok.verdict.is_empty());
    }
}

/// On `/fit` (an HTTP path) the exception is Python's catch-all 500, and the server carries on.
#[tokio::test]
async fn test_fit_with_a_raising_engine_is_the_generic_500() {
    pycompat::install_quiet_panic_hook();
    let mut env = env_with(&["memory_model"]);
    env.seed_hub_model("Qwen/Qwen3-32B", None);
    let r = env
        .post(&format!("{P}/fit"))
        .json(&json!({"engine": "raising", "repo_id": "Qwen/Qwen3-32B", "params": {}}))
        .send()
        .await;
    assert_eq!(r.status, 500, "{}", r.text());
    assert_eq!(r.code(), "internal_error");
    assert!(!r.text().contains("ZeroDivisionError"), "{}", r.text());
    let ok = env
        .post(&format!("{P}/fit"))
        .json(&json!({"engine": "fake", "repo_id": "Qwen/Qwen3-32B", "params": {}}))
        .send()
        .await;
    assert_eq!(ok.status, 200, "{}", ok.text());
}
