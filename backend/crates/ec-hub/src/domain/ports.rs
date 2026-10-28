//! Ports for side effects the hub itself performs (port of domain/ports.py).
//! The node ports (FleetPort, NodeClient, NodeStream) are in domain::fleet. GpuProbe and CommandRunner of the
//! Python file are not needed by the hub any more (contract v2: nodes do that); they are not ported.

use async_trait::async_trait;
use serde_json::{Map, Value};

use crate::domain::errors::AppResult;

#[derive(Debug, Clone, PartialEq)]
pub struct RepoFile {
    pub path: String,
    pub size: i64,
    /// LFS oid; None for plain git blobs (size-checked only)
    pub sha256: Option<String>,
}

/// Raw facts from the hub about one repo revision (already parsed, no HTTP objects).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct HubModel {
    pub repo_id: String,
    pub sha: Option<String>,
    pub gated: bool,
    pub license: Option<String>,
    pub pipeline_tag: Option<String>,
    pub library_name: Option<String>,
    pub tags: Vec<String>,
    pub files: Vec<RepoFile>,
    /// dtype -> param count, in the API's order
    pub safetensors_params: Vec<(String, i64)>,
    pub safetensors_total: Option<i64>,
    pub config: Option<Map<String, Value>>,
    pub downloads: Option<i64>,
    pub likes: Option<i64>,
}

/// One file body being downloaded from the hub.
#[async_trait]
pub trait ByteStream: Send {
    /// Content length of the whole file when known.
    fn total(&self) -> Option<i64>;
    /// Byte offset the stream actually starts at (0 if the server ignored Range).
    fn offset(&self) -> i64;
    /// Next chunk; None at the end.
    async fn chunk(&mut self) -> AppResult<Option<bytes::Bytes>>;
    async fn close(&mut self);
}

/// The Hugging Face client (services/hf_http.rs implements it; tests use fakes).
#[async_trait]
pub trait HubClient: Send + Sync {
    #[allow(clippy::too_many_arguments)]
    async fn search(
        &self,
        q: &str,
        task: Option<&str>,
        library: Option<&str>,
        quant: Option<&str>,
        sort: &str,
        limit: i64,
        token_ok: bool,
    ) -> AppResult<Vec<HubModel>>;
    async fn model(&self, repo_id: &str, revision: Option<&str>) -> AppResult<HubModel>;
    async fn card(&self, repo_id: &str, revision: Option<&str>) -> AppResult<String>;
    async fn open_file(
        &self,
        repo_id: &str,
        revision: &str,
        path: &str,
        start: i64,
    ) -> AppResult<Box<dyn ByteStream>>;
}
