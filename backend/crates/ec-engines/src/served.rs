//! What the hub serves about each engine (`GET /engines`, `GET /engines/{engine}/params`), less the parts that
//! depend on the hub's state (its image pins and the learn bundle's explanations). The hub's handlers and the
//! browser's module (ec-engines-wasm) build the same JSON from here.

use serde_json::{Map, Value, json};

use crate::adapters::EngineAdapter;
use crate::adapters::base::ParamSpec;

/// The parameter catalog as served: each spec's `level` filled in from `advanced` when the adapter left it empty.
/// `explain` is left as the adapter wrote it (the hub merges the learn bundle's explanations in).
pub fn served_catalog(a: &dyn EngineAdapter) -> Vec<ParamSpec> {
    a.param_catalog()
        .into_iter()
        .map(|mut p| {
            if p.level.as_deref().is_none_or(str::is_empty) {
                p.level = Some(if p.advanced { "advanced" } else { "basic" }.into());
            }
            p
        })
        .collect()
}

/// The adapter's own preset explanations (`preset_info()`), name -> Explain, as a JSON object.
pub fn preset_info_json(a: &dyn EngineAdapter) -> Value {
    Value::Object(
        a.preset_info()
            .into_iter()
            .map(|(k, e)| (k, serde_json::to_value(e).unwrap_or(Value::Null)))
            .collect(),
    )
}

/// One entry of `GET /engines`: `default_image` is the image the caller pins (else the adapter's), `presets_info` the
/// explanations the caller chose (the learn bundle's, else [`preset_info_json`]).
pub fn engine_entry(a: &dyn EngineAdapter, default_image: &str, presets_info: &Value) -> Value {
    let presets: Vec<Value> = a
        .presets()
        .into_iter()
        .map(|(n, p)| json!({"name": n, "params": p}))
        .collect();
    let mut formats = a.model_formats();
    formats.sort();
    let images: Map<String, Value> = a
        .images_by_arch()
        .into_iter()
        .map(|(k, v)| (k, Value::String(v)))
        .collect();
    json!({
        "id": a.id(),
        "display_name": a.display_name(),
        "default_image": default_image,
        "presets": presets,
        "model_formats": formats,
        "images_by_arch": images,
        "multi_model": a.multi_model(),
        "presets_info": presets_info,
    })
}
