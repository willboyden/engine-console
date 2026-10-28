//! Port of backend/tests/core/test_learn_engines_api.py: GET /learn, and the engines API that merges learn-mode
//! explanations into the parameter catalogue. Python registers adapters at run time and monkeypatches the learn
//! content; here both are given when the app is built (`TestEnv.adapters`, `TestEnv.learn`).
#![allow(clippy::panic, clippy::expect_used, clippy::unwrap_used)] // test helpers outside #[test] fns

#[path = "common/kit.rs"]
mod kit;

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use ec_hub::adapters::base::{
    Compat, EngineAdapter, Explain, FloatMap, Hardware, LaunchContext, LaunchSpec, ModelInfo, Obj,
    ParamSpec,
};
use ec_hub::domain::errors::Problem;
use ec_hub::domain::learn::LearnBundle;
use kit::{FakeAdapter, FakeMultiAdapter, TestEnv};
use serde_json::{Value, json};

const P: &str = "/api/v1";

fn doc(section: &str, items: &Value) -> String {
    json!({"version": 1, section: items}).to_string()
}

fn ex(summary: &str, level: &str) -> Value {
    json!({"summary": summary, "details": "d", "when_to_change": "w", "risks": "r", "example": "e",
           "terms": ["kv_cache"], "level": level})
}

/// The `content` fixture: a small learn bundle on disk (real files, loaded by the real loader).
fn content(root: &Path) -> LearnBundle {
    std::fs::create_dir_all(root.join("params")).unwrap();
    std::fs::create_dir_all(root.join("presets")).unwrap();
    let w = |rel: &str, text: String| std::fs::write(root.join(rel), text).unwrap();
    w(
        "glossary.json",
        doc(
            "terms",
            &json!({"kv_cache": {"title": "KV cache", "short": "Memory of the conversation.",
                                                          "long": "", "see_also": []}}),
        ),
    );
    w(
        "metrics.json",
        doc(
            "metrics",
            &json!({"requests_running": ex("How many requests run right now.", "advanced")}),
        ),
    );
    w(
        "concepts.json",
        doc(
            "concepts",
            &json!({"state.ready": ex("The engine answers requests.", "basic")}),
        ),
    );
    w(
        "params/fake.json",
        doc(
            "params",
            &json!({"max_model_len": ex("The longest conversation.", "basic"),
                                                 "tp": ex("Cards one model is split across.", "advanced")}),
        ),
    );
    w(
        "params/fakecpp.json",
        doc(
            "params",
            &json!({"max_model_len": ex("The longest conversation (cpp).", "advanced")}),
        ),
    );
    w(
        "params/ghost.json",
        doc(
            "params",
            &json!({"x": ex("An engine that is not installed.", "advanced")}),
        ),
    );
    w(
        "presets/fake.json",
        doc(
            "presets",
            &json!({"balanced": ex("A bit of everything.", "basic")}),
        ),
    );
    w(
        "presets/ghost.json",
        doc("presets", &json!({"p": ex("not installed", "advanced")})),
    );
    ec_hub::learn::load_bundle_dir(root).unwrap()
}

/// An env serving the `content` bundle, with `adapters` registered.
fn env_with(adapters: Vec<Arc<dyn EngineAdapter>>) -> TestEnv {
    let mut env = TestEnv::new();
    env.learn = Some(content(&env.tmp.path().join("learn")));
    env.adapters = adapters;
    env
}

fn fake_and_cpp() -> Vec<Arc<dyn EngineAdapter>> {
    vec![Arc::new(FakeAdapter), Arc::new(FakeMultiAdapter)]
}

fn keys(v: &Value) -> Vec<&str> {
    let mut k: Vec<&str> = v.as_object().unwrap().keys().map(String::as_str).collect();
    k.sort_unstable();
    k
}

async fn by_id(env: &mut TestEnv, path: &str, id: &str) -> HashMap<String, Value> {
    let list = env.get(path).send().await.json();
    list.as_array()
        .unwrap()
        .iter()
        .map(|e| (e[id].as_str().unwrap().to_string(), e.clone()))
        .collect()
}

/// An adapter that is `inner` except for its catalogue or its own preset explanations (Python patches the instance
/// or subclasses it).
struct Tweaked {
    inner: Arc<dyn EngineAdapter>,
    catalog: Option<Vec<ParamSpec>>,
    info: Option<Vec<(String, Explain)>>,
}

impl EngineAdapter for Tweaked {
    fn id(&self) -> &str {
        self.inner.id()
    }
    fn display_name(&self) -> &str {
        self.inner.display_name()
    }
    fn default_image(&self) -> &str {
        self.inner.default_image()
    }
    fn default_port(&self) -> i64 {
        self.inner.default_port()
    }
    fn model_formats(&self) -> Vec<String> {
        self.inner.model_formats()
    }
    fn images_by_arch(&self) -> Vec<(String, String)> {
        self.inner.images_by_arch()
    }
    fn multi_model(&self) -> bool {
        self.inner.multi_model()
    }
    fn ported(&self) -> bool {
        self.inner.ported()
    }
    fn metric_prefixes(&self) -> Option<Vec<String>> {
        self.inner.metric_prefixes()
    }
    fn image_for(&self, arch: &str) -> String {
        self.inner.image_for(arch)
    }
    fn build_launch_ctx(&self, ctx: &LaunchContext) -> Result<LaunchSpec, Problem> {
        self.inner.build_launch_ctx(ctx)
    }
    fn preset_info(&self) -> Vec<(String, Explain)> {
        self.info
            .clone()
            .unwrap_or_else(|| self.inner.preset_info())
    }
    fn param_catalog(&self) -> Vec<ParamSpec> {
        self.catalog
            .clone()
            .unwrap_or_else(|| self.inner.param_catalog())
    }
    fn presets(&self) -> Vec<(String, Obj)> {
        self.inner.presets()
    }
    fn validate(&self, params: &Obj) -> Vec<String> {
        self.inner.validate(params)
    }
    fn build_launch(
        &self,
        m: &str,
        p: &Obj,
        hw: &Hardware,
        s: &str,
        h: &str,
    ) -> Result<LaunchSpec, Problem> {
        self.inner.build_launch(m, p, hw, s, h)
    }
    fn compatibility(&self, m: &ModelInfo, p: &Obj, hw: &Hardware) -> Vec<Compat> {
        self.inner.compatibility(m, p, hw)
    }
    fn memory_model(&self, m: &ModelInfo, p: &Obj) -> FloatMap {
        self.inner.memory_model(m, p)
    }
    fn host_memory_gib(&self, m: &ModelInfo, p: &Obj, hw: &Hardware) -> FloatMap {
        self.inner.host_memory_gib(m, p, hw)
    }
    fn parse_metrics(&self, text: &str) -> FloatMap {
        self.inner.parse_metrics(text)
    }
    fn parse_startup_log(&self, line: &str) -> Option<Obj> {
        self.inner.parse_startup_log(line)
    }
}

#[tokio::test]
async fn test_learn_serves_the_bundle_filtered_to_registered_engines() {
    let mut env = env_with(vec![Arc::new(FakeAdapter)]);
    let body = env.get(&format!("{P}/learn")).send().await.json();
    assert_eq!(body["version"], 1);
    assert_eq!(
        (keys(&body["glossary"]), keys(&body["metrics"])),
        (vec!["kv_cache"], vec!["requests_running"])
    );
    assert_eq!(keys(&body["concepts"]), ["state.ready"]);
    // `ghost` and `fakecpp` are not registered
    assert_eq!(
        (keys(&body["params"]), keys(&body["presets"])),
        (vec!["fake"], vec!["fake"])
    );
    assert_eq!(
        body["params"]["fake"]["max_model_len"]["summary"],
        "The longest conversation."
    );
    let mut env = env_with(fake_and_cpp());
    let body = env.get(&format!("{P}/learn")).send().await.json();
    assert_eq!(keys(&body["params"]), ["fake", "fakecpp"]);
    assert!(!body.to_string().contains("ghost"));
    // the documented shape
    let b: LearnBundle = serde_json::from_value(body).unwrap();
    for section in [&b.metrics, &b.concepts] {
        for v in section.values() {
            serde_json::from_value::<Explain>(v.clone()).unwrap();
        }
    }
    for engine in b.params.values().chain(b.presets.values()) {
        for v in engine.as_object().unwrap().values() {
            serde_json::from_value::<Explain>(v.clone()).unwrap();
        }
    }
}

#[tokio::test]
async fn test_learn_with_no_packaged_content_is_an_empty_valid_bundle() {
    let mut env = TestEnv::new();
    env.learn = Some(ec_hub::learn::empty_bundle());
    let body = env.get(&format!("{P}/learn")).send().await.json();
    assert_eq!(
        body,
        serde_json::to_value(ec_hub::learn::empty_bundle()).unwrap()
    );
    assert_eq!(
        body,
        json!({"version": 1, "glossary": {}, "metrics": {}, "concepts": {}, "params": {}, "presets": {}})
    );
}

#[tokio::test]
async fn test_params_carry_their_explain_and_a_level() {
    let mut env = env_with(vec![Arc::new(FakeAdapter)]);
    let by_key = by_id(&mut env, &format!("{P}/engines/fake/params"), "key").await;
    assert_eq!(
        by_key["max_model_len"]["explain"]["summary"],
        "The longest conversation."
    );
    assert_eq!(by_key["tp"]["explain"]["level"], "advanced");
    assert!(by_key["kv_dtype"]["explain"].is_null()); // no explanation yet: null, not invented
    assert!(
        by_key
            .values()
            .all(|p| p["level"] == "basic" || p["level"] == "advanced")
    );
    assert_eq!(
        (
            by_key["max_model_len"]["level"].clone(),
            by_key["trust_remote_code"]["level"].clone()
        ),
        (json!("basic"), json!("basic"))
    );
}

#[tokio::test]
async fn test_a_level_the_adapter_set_is_kept_and_advanced_params_default_to_advanced() {
    let mut catalog = FakeAdapter.param_catalog();
    catalog[0].level = Some("expert".into());
    catalog[1].advanced = true;
    let tweaked = Tweaked {
        inner: Arc::new(FakeAdapter),
        catalog: Some(catalog),
        info: None,
    };
    let mut env = env_with(vec![Arc::new(tweaked)]);
    let by_key = by_id(&mut env, &format!("{P}/engines/fake/params"), "key").await;
    assert_eq!(
        (
            by_key["max_model_len"]["level"].clone(),
            by_key["tp"]["level"].clone()
        ),
        (json!("expert"), json!("advanced"))
    );
}

#[tokio::test]
async fn test_the_catalogue_itself_is_not_mutated_by_serving_it() {
    let mut env = env_with(vec![Arc::new(FakeAdapter)]);
    env.get(&format!("{P}/engines/fake/params")).send().await;
    let c = env.container();
    let fake = c.adapters.get("fake").unwrap();
    assert!(
        fake.param_catalog()
            .iter()
            .all(|p| p.explain.is_none() && p.level.is_none())
    );
}

#[tokio::test]
async fn test_unknown_engine_is_a_404_problem() {
    let mut env = TestEnv::new();
    let r = env.get(&format!("{P}/engines/nope/params")).send().await;
    assert_eq!((r.status, r.code().as_str()), (404, "unknown_engine"));
}

#[tokio::test]
async fn test_engines_list_gains_formats_images_by_arch_multi_model_and_preset_info() {
    let mut env = env_with(fake_and_cpp());
    let eng = by_id(&mut env, &format!("{P}/engines"), "id").await;
    let (fake, cpp) = (&eng["fake"], &eng["fakecpp"]);
    assert_eq!(
        (
            fake["model_formats"].clone(),
            fake["images_by_arch"].clone()
        ),
        (json!(["hf"]), json!({}))
    );
    assert_eq!(fake["multi_model"], false);
    assert_eq!(
        cpp["model_formats"],
        json!(["gguf", "hf", "ollama", "plain"])
    );
    assert_eq!(
        cpp["images_by_arch"],
        json!({"aarch64": "fake/engine:1.0-arm64"})
    );
    let names: Vec<&str> = fake["presets"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, ["balanced", "long-context"]);
    assert_eq!(
        fake["presets_info"]["balanced"]["summary"],
        "A bit of everything."
    ); // from the learn content
    assert_eq!(
        (fake["default_image"].clone(), fake["display_name"].clone()),
        (json!("fake/engine:1.0"), json!("Fake Engine"))
    );
}

#[tokio::test]
async fn test_preset_info_falls_back_to_the_adapters_own_when_learn_has_none() {
    let own = Explain {
        summary: "from the adapter".into(),
        details: String::new(),
        when_to_change: String::new(),
        risks: String::new(),
        example: String::new(),
        terms: Vec::new(),
        level: "advanced".into(),
    };
    let with_info = Tweaked {
        inner: Arc::new(FakeMultiAdapter),
        catalog: None,
        info: Some(vec![("balanced".into(), own.clone())]),
    };
    let mut env = env_with(vec![Arc::new(FakeAdapter), Arc::new(with_info)]);
    let eng = by_id(&mut env, &format!("{P}/engines"), "id").await;
    assert_eq!(
        eng["fakecpp"]["presets_info"],
        json!({"balanced": serde_json::to_value(&own).unwrap()})
    );
    assert_eq!(
        eng["fake"]["presets_info"]["balanced"]["summary"],
        "A bit of everything."
    );
}

#[tokio::test]
async fn test_default_image_shows_the_settings_pin() {
    let mut env = TestEnv::new();
    env.req("PUT", &format!("{P}/settings"))
        .json(&json!({"image_pins": {"fake": "fake/engine:2.0"}}))
        .send()
        .await;
    let eng = by_id(&mut env, &format!("{P}/engines"), "id").await;
    assert_eq!(eng["fake"]["default_image"], "fake/engine:2.0");
}
