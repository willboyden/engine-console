//! API/domain data shapes (port of domain/models.py). Additive changes only: the API serves them.
//!
//! Serialisation rules that keep the JSON identical to pydantic's `model_dump()`: every field is always present
//! (`None` -> `null`, never skipped), `float` fields are `f64` (pydantic turns an int given to a float field into a
//! float, and serde_json writes `5.0`), `int` fields are `i64`, `dict[str, Any]` is `Obj`, `Any` is `Value`, and a
//! `Literal[...]` is a `String` whose values the request validator checks (api/validate.rs).

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::adapters::base::Compat;
use crate::domain::errors::Obj;
use crate::domain::refs::{InstanceControls, InstanceVram, ModelLocation, ModelRef};

pub const HOST_ID_PATTERN: &str = r"^[a-z0-9][a-z0-9-]{0,31}$";
pub const INSTANCE_NAME_PATTERN: &str = r"^[A-Za-z0-9][A-Za-z0-9 _.\-]{0,63}$";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Page<T> {
    pub items: Vec<T>,
    #[serde(default)]
    pub next_cursor: Option<String>,
}

impl<T> Page<T> {
    pub fn new(items: Vec<T>, next_cursor: Option<String>) -> Self {
        Self { items, next_cursor }
    }
}

// ---- hardware -------------------------------------------------------------------------------
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GpuStat {
    pub index: i64,
    pub uuid: String,
    pub name: String,
    pub total_gib: f64,
    pub free_gib: f64,
    pub used_gib: f64,
    #[serde(default)]
    pub util_pct: Option<f64>,
    #[serde(default)]
    pub temp_c: Option<f64>,
    #[serde(default)]
    pub power_w: Option<f64>,
    #[serde(default)]
    pub fan_pct: Option<f64>,
    #[serde(default)]
    pub compute_capability: Option<String>,
}

/// Host RAM/swap from /proc/meminfo, in GiB. used = total - available.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HostMemInfo {
    pub total_gib: f64,
    pub used_gib: f64,
    pub available_gib: f64,
    pub free_gib: f64,
    pub cached_gib: f64,
    pub shmem_gib: f64,
    pub swap_total_gib: f64,
    pub swap_used_gib: f64,
    pub updated_at: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HostAlert {
    /// "warn" | "crit"
    pub level: String,
    pub code: String,
    pub message: String,
}

/// Host RAM held by one container (cgroup v2 `file` includes shmem, so cache = file - shmem).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HostMemory {
    pub total_gib: f64,
    pub anon_gib: f64,
    pub cache_gib: f64,
    pub shmem_gib: f64,
    pub kernel_gib: f64,
    /// "cgroup" | "rss"
    pub source: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HostRamFit {
    pub needed_gib: f64,
    pub available_gib: Option<f64>,
    pub total_gib: Option<f64>,
    /// "ok" | "tight" | "wont_fit" | "unknown"
    pub verdict: String,
    pub breakdown: indexmap_f64::Breakdown,
    #[serde(default)]
    pub notes: Vec<String>,
}

/// `dict[str, float]` with insertion order (serde_json's preserve_order map of f64 values).
pub mod indexmap_f64 {
    pub type Breakdown = serde_json::Map<String, serde_json::Value>;
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HardwareReport {
    pub gpus: Vec<GpuStat>,
    pub host_ram_gib: f64,
    pub host_ram_free_gib: f64,
    /// "nvml" | "nvidia-smi" | "none"
    pub source: String,
    #[serde(default)]
    pub memory: Option<HostMemInfo>,
    #[serde(default)]
    pub alerts: Vec<HostAlert>,
    #[serde(default = "local")]
    pub host_id: String,
}

pub fn local() -> String {
    "local".to_string()
}

// ---- fit ------------------------------------------------------------------------------------
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GpuFit {
    pub gpu_id: i64,
    pub uuid: String,
    pub name: String,
    pub weights_gib: f64,
    pub kv_cache_gib: f64,
    pub activations_gib: f64,
    pub cuda_graphs_gib: f64,
    pub overhead_gib: f64,
    pub total_gib: f64,
    pub budget_gib: f64,
    pub vram_total_gib: f64,
    pub free_gib: f64,
    pub utilization_pct: f64,
    /// "fits" | "tight" | "wont_fit" | "unknown"
    pub verdict: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FitReport {
    pub verdict: String,
    /// "high" | "medium" | "low"
    pub confidence: String,
    pub tp: i64,
    pub tp_required: Option<i64>,
    pub concurrency: i64,
    pub max_len: Option<i64>,
    #[serde(default)]
    pub kv_bytes_per_token: Option<f64>,
    pub per_gpu: Vec<GpuFit>,
    pub max_context_at_current_concurrency: Option<i64>,
    pub max_concurrency_at_current_context: Option<i64>,
    #[serde(default)]
    pub fits_if_stop: Vec<String>,
    #[serde(default)]
    pub notes: Vec<String>,
    #[serde(default)]
    pub compat: Vec<Compat>,
    #[serde(default)]
    pub host_ram: Option<HostRamFit>,
}

/// A running instance's VRAM on given GPUs, so the estimator can say "fits if you stop X".
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResidentUse {
    pub name: String,
    pub gpu_ids: Vec<i64>,
    pub gib_per_gpu: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FitRequest {
    pub engine: String,
    #[serde(default)]
    pub repo_id: Option<String>,
    #[serde(default)]
    pub revision: Option<String>,
    #[serde(default)]
    pub params: Obj,
    #[serde(default)]
    pub gpu_ids: Vec<i64>,
    #[serde(default = "one")]
    pub concurrency: i64,
    #[serde(default = "local")]
    pub host_id: String,
    #[serde(default)]
    pub model: Option<ModelRef>,
}

fn one() -> i64 {
    1
}

// ---- downloads ------------------------------------------------------------------------------
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Download {
    pub id: String,
    pub repo_id: String,
    #[serde(default)]
    pub revision: Option<String>,
    #[serde(default)]
    pub commit_sha: Option<String>,
    #[serde(default)]
    pub allow_patterns: Option<Vec<String>>,
    /// "queued" | "running" | "paused" | "completed" | "failed" | "cancelled"
    pub state: String,
    #[serde(default)]
    pub total_bytes: i64,
    #[serde(default)]
    pub done_bytes: i64,
    #[serde(default)]
    pub speed_bps: f64,
    #[serde(default)]
    pub eta_s: Option<f64>,
    #[serde(default)]
    pub files_total: i64,
    #[serde(default)]
    pub files_done: i64,
    #[serde(default)]
    pub current_file: Option<String>,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub error_code: Option<String>,
    pub created_at: f64,
    pub updated_at: f64,
    #[serde(default = "local")]
    pub host_id: String,
    #[serde(default)]
    pub root_id: Option<String>,
    /// "hf" | "copy"
    #[serde(default = "hf")]
    pub kind: String,
    #[serde(default)]
    pub source_copy_id: Option<String>,
}

fn hf() -> String {
    "hf".to_string()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DownloadRequest {
    pub repo_id: String,
    #[serde(default)]
    pub revision: Option<String>,
    #[serde(default)]
    pub allow_patterns: Option<Vec<String>>,
    #[serde(default = "local")]
    pub host_id: String,
    #[serde(default)]
    pub root_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LocalModel {
    pub repo_id: String,
    pub size_bytes: i64,
    pub revisions: Vec<String>,
    pub last_used: Option<f64>,
    pub path: String,
    #[serde(default)]
    pub engines_that_fit: Vec<String>,
    #[serde(default = "local")]
    pub host_id: String,
}

// ---- instances ------------------------------------------------------------------------------
pub const INSTANCE_STATES: &[&str] = &[
    "stopped",
    "starting",
    "loading",
    "ready",
    "stopping",
    "failed",
    "auth_required",
    "unreachable",
];

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InstanceCreate {
    pub engine: String,
    #[serde(default)]
    pub repo_id: Option<String>,
    #[serde(default)]
    pub params: Obj,
    #[serde(default)]
    pub gpu_ids: Vec<i64>,
    #[serde(default)]
    pub profile_id: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub ttl_idle_s: Option<i64>,
    #[serde(default = "local")]
    pub host_id: String,
    #[serde(default)]
    pub model: Option<ModelRef>,
    #[serde(default)]
    pub image: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Instance {
    pub id: String,
    pub name: String,
    pub engine: String,
    pub repo_id: String,
    pub params: Obj,
    pub gpu_ids: Vec<i64>,
    pub gpu_uuids: Vec<String>,
    pub port: Option<i64>,
    pub container_name: Option<String>,
    #[serde(default)]
    pub container_port: Option<i64>,
    #[serde(default)]
    pub internal_endpoint: Option<String>,
    pub image: Option<String>,
    pub state: String,
    #[serde(default)]
    pub phase: Option<String>,
    #[serde(default)]
    pub progress_pct: Option<f64>,
    #[serde(default)]
    pub pinned: bool,
    #[serde(default)]
    pub ttl_idle_s: Option<i64>,
    #[serde(default)]
    pub profile_id: Option<String>,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub last_logs: Option<String>,
    #[serde(default)]
    pub fit: Option<Obj>,
    pub created_at: f64,
    #[serde(default)]
    pub started_at: Option<f64>,
    #[serde(default)]
    pub last_request_at: Option<f64>,
    #[serde(default)]
    pub uptime_s: Option<f64>,
    #[serde(default = "yes")]
    pub managed: bool,
    /// "console" | "external"
    #[serde(default = "console")]
    pub source: String,
    #[serde(default)]
    pub endpoint: Option<String>,
    #[serde(default)]
    pub served_models: Vec<String>,
    #[serde(default)]
    pub state_reason: Option<String>,
    #[serde(default)]
    pub history_since: Option<f64>,
    #[serde(default)]
    pub host_memory: Option<HostMemory>,
    #[serde(default = "local")]
    pub host_id: String,
    #[serde(default)]
    pub host_label: Option<String>,
    /// "managed" | "script" | "external"
    #[serde(default = "managed")]
    pub management: String,
    #[serde(default)]
    pub script: Option<String>,
    #[serde(default)]
    pub vram: Vec<InstanceVram>,
    #[serde(default)]
    pub model: Option<ModelRef>,
    #[serde(default)]
    pub model_location: Option<ModelLocation>,
    #[serde(default)]
    pub controls: InstanceControls,
}

fn yes() -> bool {
    true
}
fn console() -> String {
    "console".to_string()
}
fn managed() -> String {
    "managed".to_string()
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct InstancePatch {
    #[serde(default)]
    pub pinned: Option<bool>,
    #[serde(default)]
    pub ttl_idle_s: Option<i64>,
    #[serde(default)]
    pub name: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PreflightCheck {
    pub code: String,
    /// "ok" | "warn" | "block"
    pub level: String,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PreflightReport {
    pub ok: bool,
    #[serde(default)]
    pub gpu_ids: Vec<i64>,
    pub checks: Vec<PreflightCheck>,
    #[serde(default)]
    pub fit: Option<FitReport>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CommandSnippets {
    pub docker_run: String,
    pub compose_yaml: String,
    pub engine_cli: String,
}

// ---- profiles -------------------------------------------------------------------------------
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProfileIn {
    pub name: String,
    pub engine: String,
    #[serde(default)]
    pub repo_id: Option<String>,
    #[serde(default)]
    pub preset: Option<String>,
    #[serde(default)]
    pub params: Obj,
    #[serde(default)]
    pub description: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Profile {
    pub id: String,
    pub name: String,
    pub engine: String,
    pub repo_id: Option<String>,
    pub params: Obj,
    pub description: String,
    pub created_at: f64,
    pub updated_at: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProfileDiff {
    pub a: String,
    pub b: String,
    pub changed: Obj,
    pub only_a: Obj,
    pub only_b: Obj,
}

// ---- metrics / bench ------------------------------------------------------------------------
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricPoint {
    pub t: f64,
    /// `dict[str, float]`
    pub values: Obj,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchRequest {
    pub instance_id: String,
    /// `str | dict[str, Any]`, default "quick"
    #[serde(default = "quick")]
    pub suite: Value,
    #[serde(default)]
    pub confirm_external: bool,
}

fn quick() -> Value {
    Value::String("quick".into())
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchRun {
    pub id: String,
    pub instance_id: String,
    #[serde(default)]
    pub engine: Option<String>,
    #[serde(default)]
    pub repo_id: Option<String>,
    #[serde(default)]
    pub profile_id: Option<String>,
    #[serde(default)]
    pub params: Obj,
    pub suite: Obj,
    /// "running" | "completed" | "failed"
    pub state: String,
    #[serde(default)]
    pub results: Obj,
    #[serde(default)]
    pub error: Option<String>,
    pub created_at: f64,
    #[serde(default)]
    pub finished_at: Option<f64>,
}

// ---- chat / arena / usage -------------------------------------------------------------------
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConversationIn {
    #[serde(default = "new_chat")]
    pub title: String,
    #[serde(default)]
    pub system_prompt: Option<String>,
    #[serde(default)]
    pub instance_id: Option<String>,
}

fn new_chat() -> String {
    "New chat".to_string()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Conversation {
    pub id: String,
    pub title: String,
    pub system_prompt: Option<String>,
    pub instance_id: Option<String>,
    pub created_at: f64,
    pub updated_at: f64,
    #[serde(default)]
    pub messages: Option<Vec<Obj>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PromptIn {
    pub title: String,
    pub content: String,
    #[serde(default)]
    pub tags: Vec<String>,
}

/// `Prompt(PromptIn)`: the PromptIn fields first, then its own.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Prompt {
    pub title: String,
    pub content: String,
    #[serde(default)]
    pub tags: Vec<String>,
    pub id: String,
    pub created_at: f64,
    pub updated_at: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArenaMatchIn {
    pub prompt: String,
    pub instance_a: String,
    pub instance_b: String,
    #[serde(default)]
    pub system_prompt: Option<String>,
    #[serde(default = "yes")]
    pub blind: bool,
    #[serde(default = "max_tokens")]
    pub max_tokens: i64,
}

fn max_tokens() -> i64 {
    512
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArenaVote {
    /// "a" | "b" | "tie"
    pub winner: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UsageRow {
    pub key: String,
    pub requests: i64,
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
    pub avg_latency_ms: f64,
}

// ---- keys / settings ------------------------------------------------------------------------
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KeyIn {
    pub name: String,
    /// "admin" | "viewer"
    #[serde(default = "viewer")]
    pub role: String,
}

fn viewer() -> String {
    "viewer".to_string()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KeyInfo {
    pub id: String,
    pub name: String,
    pub role: String,
    pub prefix: String,
    pub created_at: f64,
    pub last_used_at: Option<f64>,
    /// only ever populated once, in the create response
    #[serde(default)]
    pub secret: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct SettingsPatch {
    #[serde(default)]
    pub default_gpu_ids: Option<Vec<i64>>,
    #[serde(default)]
    pub idle_ttl_s: Option<i64>,
    /// `dict[str, str]`, in insertion order
    #[serde(default)]
    pub image_pins: Option<Obj>,
    /// `dict[str, dict[str, str]]`
    #[serde(default)]
    pub host_image_pins: Option<Obj>,
}
