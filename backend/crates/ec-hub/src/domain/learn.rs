//! Learn mode shapes (port of domain/learn.py). Content: backend/src/engine_console/learn/*.json.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// One glossary entry, e.g. key "kv_cache".
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Term {
    pub title: String,
    pub short: String,
    #[serde(default)]
    pub long: String,
    #[serde(default)]
    pub see_also: Vec<String>,
}

/// GET /api/v1/learn. Maps keep the JSON files' order (serde_json preserve_order), as Python dicts do.
/// `metrics`/`concepts` values are `Explain` objects and `params`/`presets` are engine -> key -> `Explain`; they are
/// kept as validated JSON here (learn/mod.rs checks every entry against `Explain`) so the dump is byte-for-byte
/// the pydantic one, defaults included.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LearnBundle {
    pub version: i64,
    pub glossary: Map<String, Value>,
    pub metrics: Map<String, Value>,
    pub concepts: Map<String, Value>,
    pub params: Map<String, Value>,
    pub presets: Map<String, Value>,
}
