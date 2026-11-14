//! Engine Console's engine adapters in the browser (ADR 0023): `ec-engines` behind a small JSON-in, JSON-out API
//! that `frontend/js/engines-wasm.js` loads. The Launch page's command preview, env list and files, the parameter
//! form's validation, the image-policy badge and the Config files page's checks all come from the same Rust code the
//! hub runs. Everything here is advisory: the hub validates again on preflight and launch.
//!
//! Every export but [`image_allowed`] and [`api_version`] answers a JSON envelope:
//! `{"ok": <value>}`, `{"problem": {"status", "code", "detail", ...extra}}` (what the hub would answer with that
//! status), or `{"raised": {"kind", "msg"}}` (a Python exception the adapter raised, which the hub reports as a 500).
//! On wasm32 a panicking adapter cannot be caught (panic = abort): the module traps, and the loader discards the
//! instance. Natively (the tests) the adapters' `try_*` forms turn the same panic into `raised`.

use std::sync::{Arc, LazyLock};

use ec_engines::adapters::base::{Hardware, LaunchContext, ModelInfo};
use ec_engines::adapters::pycompat::PyExc;
use ec_engines::adapters::{AdapterRegistry, EngineAdapter, load_default_adapters};
use ec_engines::errors::{Obj, Problem, unprocessable};
use ec_engines::served;
use serde_json::{Value, json};
use wasm_bindgen::prelude::wasm_bindgen;

/// The version of this JSON API; the loader refuses a module of another version (a stale frontend/pkg/).
pub const API_VERSION: u32 = 1;

static REGISTRY: LazyLock<AdapterRegistry> = LazyLock::new(load_default_adapters);

fn ok(v: Value) -> String {
    let mut o = Obj::new();
    o.insert("ok".into(), v);
    Value::Object(o).to_string()
}

fn problem(p: &Problem) -> String {
    let mut o = Obj::new();
    o.insert("status".into(), Value::from(p.status));
    o.insert("code".into(), Value::String(p.code.clone()));
    o.insert("detail".into(), Value::String(p.detail.clone()));
    for (k, v) in &p.extra {
        o.insert(k.clone(), v.clone());
    }
    json!({ "problem": o }).to_string()
}

/// A raised exception as the hub's launch build treats it: a ValueError is a 422 `invalid_params` with its message;
/// any other kind is reported as raised.
fn raised(e: &PyExc) -> String {
    if e.kind == "ValueError" {
        return problem(&unprocessable(e.msg.clone()).code("invalid_params"));
    }
    json!({ "raised": { "kind": e.kind, "msg": e.msg } }).to_string()
}

fn adapter(engine: &str) -> Result<Arc<dyn EngineAdapter>, String> {
    REGISTRY.get(engine).map_err(|p| problem(&p))
}

fn parse<T: serde::de::DeserializeOwned>(what: &str, text: &str) -> Result<T, String> {
    serde_json::from_str(text).map_err(|e| problem(&unprocessable(format!("{what}: {e}"))))
}

fn both(r: Result<String, String>) -> String {
    r.unwrap_or_else(|e| e)
}

/// The version of this JSON API.
#[wasm_bindgen]
pub fn api_version() -> u32 {
    API_VERSION
}

/// The workspace version the module was built from (shown next to the preview).
#[wasm_bindgen]
pub fn build_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// `GET /engines` as the hub answers it without image pins or learn files: each engine's own default image and
/// preset explanations.
#[wasm_bindgen]
pub fn engines() -> String {
    let list: Vec<Value> = REGISTRY
        .list()
        .iter()
        .map(|a| {
            served::engine_entry(
                a.as_ref(),
                a.default_image(),
                &served::preset_info_json(a.as_ref()),
            )
        })
        .collect();
    ok(Value::Array(list))
}

/// `GET /engines/{engine}/params` without the learn bundle's explanations.
#[wasm_bindgen]
pub fn catalog(engine: &str) -> String {
    both(adapter(engine).map(|a| {
        ok(serde_json::to_value(served::served_catalog(a.as_ref())).unwrap_or(Value::Null))
    }))
}

/// The engine's named presets, `[{name, params}]` in the adapter's order.
#[wasm_bindgen]
pub fn presets(engine: &str) -> String {
    both(adapter(engine).map(|a| {
        ok(Value::Array(
            a.presets()
                .into_iter()
                .map(|(n, p)| json!({ "name": n, "params": p }))
                .collect(),
        ))
    }))
}

/// The adapter's validation of `params` (a JSON object): the error messages the hub's preflight would list
/// (unknown keys, types and ranges, and the cross-field rules such as tool choice needing a tool-call parser).
#[wasm_bindgen]
pub fn validate(engine: &str, params: &str) -> String {
    both((|| {
        let a = adapter(engine)?;
        let p: Obj = parse("params", params)?;
        Ok(match a.try_validate(&p) {
            Ok(errors) => ok(json!(errors)),
            Err(e) => raised(&e),
        })
    })())
}

/// The engine command the hub would build for one launch: `ctx` is a LaunchContext (model, model_format,
/// model_path, served_name, hf_cache_container_path, params, hw). The answer is the LaunchSpec (image, argv, env,
/// files, ...) or the hub's refusal. The `docker run` around it is rendered by the node and is not part of this.
#[wasm_bindgen]
pub fn render(engine: &str, ctx: &str) -> String {
    both((|| {
        let a = adapter(engine)?;
        let c: LaunchContext = parse("launch context", ctx)?;
        Ok(match a.try_build_launch_ctx(&c) {
            Ok(Ok(spec)) => ok(serde_json::to_value(spec).unwrap_or(Value::Null)),
            Ok(Err(p)) => problem(&p),
            Err(e) => raised(&e),
        })
    })())
}

/// The adapter's compatibility rules for a model (`{model: ModelInfo, params, hw: Hardware}`): `[{level, code,
/// message}]`.
#[wasm_bindgen]
pub fn compatibility(engine: &str, input: &str) -> String {
    both((|| {
        let a = adapter(engine)?;
        let v: Value = parse("input", input)?;
        let model: ModelInfo = parse(
            "model",
            &v.get("model").cloned().unwrap_or(Value::Null).to_string(),
        )?;
        let hw: Hardware = parse(
            "hw",
            &v.get("hw").cloned().unwrap_or(Value::Null).to_string(),
        )?;
        let params: Obj = match v.get("params") {
            None | Some(Value::Null) => Obj::new(),
            Some(p) => parse("params", &p.to_string())?,
        };
        Ok(match a.try_compatibility(&model, &params, &hw) {
            Ok(list) => ok(serde_json::to_value(list).unwrap_or(Value::Null)),
            Err(e) => raised(&e),
        })
    })())
}

/// The node policy's `engine_images` rule (ec_proto::image, the node's own code): does `image` match one of
/// `patterns` (a JSON array of strings)? Anything that is not such an array matches nothing.
#[wasm_bindgen]
pub fn image_allowed(patterns: &str, image: &str) -> bool {
    match serde_json::from_str::<Vec<String>>(patterns) {
        Ok(p) => ec_proto::image::image_allowed(&p, image),
        Err(_) => false,
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    fn hw(n: usize) -> Value {
        let ids: Vec<usize> = (0..n).collect();
        json!({
            "gpu_ids": ids, "gpu_uuids": ids.iter().map(|i| format!("GPU-{i}")).collect::<Vec<_>>(),
            "gpu_names": vec!["NVIDIA RTX PRO 6000"; n], "gpu_total_gib": vec![96.0; n], "gpu_free_gib": vec![90.0; n],
            "compute_capability": "12.0", "host_ram_gib": 256.0,
        })
    }

    fn ctx(model: &str, params: &Value) -> String {
        json!({
            "model": model, "served_name": model, "hf_cache_container_path": "/root/.cache/huggingface",
            "params": params, "hw": hw(1),
        })
        .to_string()
    }

    fn env(answer: &str) -> Value {
        serde_json::from_str(answer).unwrap()
    }

    /// The old JS preview dropped a false bool; the hub renders vLLM's negatable ones as `--no-<flag>`.
    #[test]
    fn test_vllm_bool_false_renders_no_flag() {
        let r = env(&render(
            "vllm",
            &ctx(
                "org/m",
                &json!({"enable_prefix_caching": false, "enforce_eager": true}),
            ),
        ));
        let argv: Vec<String> = serde_json::from_value(r["ok"]["argv"].clone()).unwrap();
        assert!(
            argv.contains(&"--no-enable-prefix-caching".to_string()),
            "{argv:?}"
        );
        assert!(argv.contains(&"--enforce-eager".to_string()), "{argv:?}");
        assert!(
            argv.windows(2)
                .any(|w| w == ["--served-model-name", "org/m"]),
            "{argv:?}"
        );
        // the env the hub adds (telemetry off), which the old preview never showed
        assert!(
            r["ok"]["env"].get("VLLM_NO_USAGE_STATS").is_some(),
            "{}",
            r["ok"]["env"]
        );
    }

    /// TensorRT-LLM's `yaml:` parameters go into the extra options file, never onto the command line.
    #[test]
    fn test_trtllm_yaml_keys_go_to_the_file_not_argv() {
        let r = env(&render(
            "trtllm",
            &ctx("org/m", &json!({"kv_cache_dtype": "fp8"})),
        ));
        let spec = &r["ok"];
        let argv = spec["argv"].to_string();
        assert!(
            !argv.contains("yaml:") && !argv.contains("kv_cache_config"),
            "{argv}"
        );
        let files = spec["files"].as_object().unwrap();
        let yaml = files["/config/extra-llm-api-options.yaml"]
            .as_str()
            .unwrap();
        assert!(
            yaml.contains("kv_cache_config:") && yaml.contains("dtype: fp8"),
            "{yaml}"
        );
        assert!(argv.contains("extra-llm-api-options"), "{argv}");
    }

    #[test]
    fn test_validate_gives_the_cross_field_rules() {
        let r = env(&validate("vllm", r#"{"enable_auto_tool_choice": true}"#));
        let errs = r["ok"].as_array().unwrap();
        assert!(
            errs.iter()
                .any(|e| e.as_str().unwrap().contains("requires tool_call_parser")),
            "{errs:?}"
        );
        let r = env(&validate(
            "vllm",
            r#"{"enable_auto_tool_choice": true, "tool_call_parser": "hermes"}"#,
        ));
        assert_eq!(r["ok"], json!([]));
        let r = env(&validate("vllm", r#"{"no_such_key": 1}"#));
        assert_eq!(r["ok"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn test_problems_and_bad_input() {
        let r = env(&catalog("nope"));
        assert_eq!(r["problem"]["status"], 404);
        assert_eq!(r["problem"]["code"], "unknown_engine");
        let r = env(&render("vllm", "{not json"));
        assert_eq!(r["problem"]["status"], 422);
        let r = env(&validate("vllm", "[1]"));
        assert_eq!(r["problem"]["code"], "invalid_params");
    }

    /// What traps the module on wasm32 (an exception the adapter raises: here TensorRT-LLM converting a GiB figure
    /// too large for an integer) comes back as `raised` natively; frontend/tests drive the same input through the
    /// .wasm to prove the loader recovers from the trap.
    #[test]
    fn test_a_raised_exception_is_reported() {
        let r = env(&render(
            "trtllm",
            &ctx("org/m", &json!({"kv_cache_max_gib": 1e308})),
        ));
        assert_eq!(r["raised"]["kind"], "OverflowError", "{r}");
    }

    #[test]
    fn test_engines_catalogs_and_presets_match_the_served_shape() {
        let r = env(&engines());
        let ids: Vec<&str> = r["ok"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e["id"].as_str().unwrap())
            .collect();
        assert_eq!(ids, ["vllm", "sglang", "trtllm", "llamacpp", "ollama"]);
        for id in ids {
            let c = env(&catalog(id));
            let specs = c["ok"].as_array().unwrap();
            assert!(!specs.is_empty());
            assert!(
                specs
                    .iter()
                    .all(|s| matches!(s["level"].as_str(), Some("basic" | "advanced" | "expert")))
            );
            assert!(env(&presets(id))["ok"].is_array());
        }
    }

    #[test]
    fn test_compatibility_answers_a_list() {
        let input = json!({"model": {"repo_id": "org/m", "quantization": "fp8"}, "params": {}, "hw": hw(1)});
        let r = env(&compatibility("vllm", &input.to_string()));
        assert!(r["ok"].is_array(), "{r}");
    }

    #[test]
    fn test_image_allowed_is_the_node_rule() {
        let d = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        assert!(image_allowed(
            r#"["vllm/vllm-openai:*"]"#,
            "vllm/vllm-openai:v1"
        ));
        assert!(!image_allowed(
            r#"["repo/x:*"]"#,
            &format!("repo/x:v1@sha256:{d}")
        ));
        assert!(!image_allowed("not json", "vllm/vllm-openai:v1"));
        assert_eq!(api_version(), API_VERSION);
    }
}
