//! Adapter registry (port of adapters/__init__.py).
//!
//! The registry lists the five engines in Python's order (vllm, sglang, trtllm, llamacpp, ollama). Each adapter
//! module exposes `pub fn adapter() -> Arc<dyn EngineAdapter>`.

pub mod base;
pub mod llamacpp;
pub mod llamacpp_params;
pub mod llamacpp_parse;
pub mod ollama;
pub mod ollama_params;
pub mod pycompat;
pub mod sglang;
pub mod sglang_params;
pub mod sglang_parsers;
pub mod stub;
pub mod trtllm;
pub mod trtllm_params;
pub mod trtllm_parse;
pub mod vllm;
pub mod vllm_params;
pub mod vllm_parse;

use std::sync::Arc;

pub use base::EngineAdapter;

use crate::domain::errors::{Problem, not_found};

#[derive(Clone, Default)]
pub struct AdapterRegistry {
    by_id: Vec<Arc<dyn EngineAdapter>>,
}

impl std::fmt::Debug for AdapterRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list()
            .entries(self.by_id.iter().map(|a| a.id().to_string()))
            .finish()
    }
}

impl AdapterRegistry {
    pub fn new(adapters: impl IntoIterator<Item = Arc<dyn EngineAdapter>>) -> Self {
        let mut r = Self::default();
        for a in adapters {
            r.register(a);
        }
        r
    }

    /// Register (a later adapter with the same id replaces the earlier one, keeping its position, as a dict does).
    pub fn register(&mut self, adapter: Arc<dyn EngineAdapter>) {
        if let Some(slot) = self.by_id.iter_mut().find(|a| a.id() == adapter.id()) {
            *slot = adapter;
        } else {
            self.by_id.push(adapter);
        }
    }

    /// 404 `unknown_engine` "unknown engine: <id>"
    pub fn get(&self, engine_id: &str) -> Result<Arc<dyn EngineAdapter>, Problem> {
        self.by_id
            .iter()
            .find(|a| a.id() == engine_id)
            .cloned()
            .ok_or_else(|| not_found(format!("unknown engine: {engine_id}")).code("unknown_engine"))
    }

    pub fn list(&self) -> Vec<Arc<dyn EngineAdapter>> {
        self.by_id.clone()
    }

    pub fn contains(&self, engine_id: &str) -> bool {
        self.by_id.iter().any(|a| a.id() == engine_id)
    }
}

/// `load_default_adapters()`: the five engines, in Python's DEFAULT_MODULES order.
pub fn load_default_adapters() -> AdapterRegistry {
    AdapterRegistry::new([
        vllm::adapter(),
        sglang::adapter(),
        trtllm::adapter(),
        llamacpp::adapter(),
        ollama::adapter(),
    ])
}
