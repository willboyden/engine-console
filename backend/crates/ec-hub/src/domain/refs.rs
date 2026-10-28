//! Small contract-v2 shapes shared by domain::models and domain::fleet (port of domain/refs.py).

use serde::{Deserialize, Serialize};

/// What to load, on the host chosen for the instance. Exactly the fields of its format are used.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelRef {
    /// "hf" | "gguf" | "ollama" | "plain"
    #[serde(default = "hf")]
    pub format: String,
    #[serde(default)]
    pub repo_id: Option<String>,
    #[serde(default)]
    pub root_id: Option<String>,
    #[serde(default)]
    pub rel: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
}

fn hf() -> String {
    "hf".to_string()
}

impl ModelRef {
    /// `ModelRef(format="hf", repo_id=...)`
    pub fn hf(repo_id: &str) -> Self {
        Self {
            format: "hf".into(),
            repo_id: Some(repo_id.into()),
            root_id: None,
            rel: None,
            name: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelLocation {
    pub host_id: String,
    pub root_id: String,
    #[serde(default)]
    pub root_label: Option<String>,
    pub path: String,
    #[serde(default)]
    pub drive: Option<String>,
    #[serde(default)]
    pub fstype: Option<String>,
    #[serde(default)]
    pub remote: bool,
    #[serde(default)]
    pub shared_from: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InstanceVram {
    pub gpu_index: i64,
    pub gpu_uuid: String,
    #[serde(default)]
    pub used_gib: Option<f64>,
}

/// What the UI may offer for one instance; `reason` says why something is not offered.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct InstanceControls {
    #[serde(default)]
    pub start: bool,
    #[serde(default)]
    pub stop: bool,
    #[serde(default)]
    pub restart: bool,
    #[serde(default)]
    pub remove: bool,
    #[serde(default)]
    pub logs: bool,
    #[serde(default)]
    pub configure: bool,
    #[serde(default)]
    pub reason: Option<String>,
}
