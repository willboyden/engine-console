//! The slice of the library the runtime services need (port of services/libport.py).
//! `LibraryService` implements it; `NullLibrary` stands in until the composition root wires the real one.

use async_trait::async_trait;

use crate::adapters::base::ModelInfo;
use crate::domain::errors::AppResult;
use crate::domain::fleet::{LibraryCopy, RootInfo};
use crate::domain::refs::ModelRef;

#[async_trait]
pub trait LibraryPort: Send + Sync {
    /// The copy of `ref` on `host_id`'s own roots (`own`) or loadable from it.
    fn find(&self, host_id: &str, r: &ModelRef) -> Option<LibraryCopy>;
    /// Where a running instance's model lives (its host's copy).
    fn locate(&self, host_id: &str, r: &ModelRef) -> Option<LibraryCopy>;
    async fn model_info(&self, host_id: &str, r: &ModelRef) -> AppResult<Option<ModelInfo>>;
    fn hf_root(&self, host_id: &str) -> Option<RootInfo>;
    fn ollama_root(&self, host_id: &str) -> Option<RootInfo>;
    fn copies(&self, host_id: &str) -> Vec<LibraryCopy>;
}

/// No library: nothing is found.
#[derive(Debug, Default, Clone, Copy)]
pub struct NullLibrary;

#[async_trait]
impl LibraryPort for NullLibrary {
    fn find(&self, _host_id: &str, _r: &ModelRef) -> Option<LibraryCopy> {
        None
    }
    fn locate(&self, _host_id: &str, _r: &ModelRef) -> Option<LibraryCopy> {
        None
    }
    async fn model_info(&self, _host_id: &str, _r: &ModelRef) -> AppResult<Option<ModelInfo>> {
        Ok(None)
    }
    fn hf_root(&self, _host_id: &str) -> Option<RootInfo> {
        None
    }
    fn ollama_root(&self, _host_id: &str) -> Option<RootInfo> {
        None
    }
    fn copies(&self, _host_id: &str) -> Vec<LibraryCopy> {
        Vec::new()
    }
}
