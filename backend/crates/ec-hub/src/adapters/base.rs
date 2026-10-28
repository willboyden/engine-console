//! Engine port: the single seam between the engine-agnostic core and a concrete engine (port of
//! adapters/base.py). Each engine adapter implements `EngineAdapter`.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::adapters::pycompat::{PyResult, catch};
use crate::domain::errors::Problem;

pub type Obj = Map<String, Value>;

/// An ordered `dict[str, float]` that may hold NaN (host_memory_gib uses NaN for "needed but unknown"), which a
/// serde_json map cannot.
pub type FloatMap = Vec<(String, f64)>;

/// Look a key up in a [`FloatMap`].
pub fn fget(m: &FloatMap, key: &str) -> Option<f64> {
    m.iter().find(|(k, _)| k == key).map(|(_, v)| *v)
}

/// Plain-language help for one parameter, metric or concept ("learn mode").
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Explain {
    pub summary: String,
    #[serde(default)]
    pub details: String,
    #[serde(default)]
    pub when_to_change: String,
    #[serde(default)]
    pub risks: String,
    #[serde(default)]
    pub example: String,
    #[serde(default)]
    pub terms: Vec<String>,
    /// "basic" | "advanced" | "expert"
    #[serde(default = "advanced")]
    pub level: String,
}

fn advanced() -> String {
    "advanced".to_string()
}

/// One tunable engine parameter, described enough to render a form + validate + emit a CLI flag.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParamSpec {
    pub key: String,
    pub flag: String,
    pub label: String,
    pub help: String,
    /// "int" | "float" | "bool" | "enum" | "string" | "string_list" | "json"
    #[serde(rename = "type")]
    pub type_: String,
    /// "memory" | "parallelism" | "context" | "quantization" | "scheduling" | "tools_reasoning" | "performance" |
    /// "speculative" | "network" | "advanced"
    pub group: String,
    #[serde(default)]
    pub default: Value,
    #[serde(default)]
    pub choices: Option<Vec<String>>,
    #[serde(default)]
    pub min: Option<f64>,
    #[serde(default)]
    pub max: Option<f64>,
    #[serde(default)]
    pub advanced: bool,
    #[serde(default)]
    pub affects_memory: bool,
    #[serde(default = "yes")]
    pub requires_restart: bool,
    /// "flag" | "value"
    #[serde(default = "flag")]
    pub bool_style: String,
    #[serde(default)]
    pub docs_url: Option<String>,
    #[serde(default)]
    pub explain: Option<Explain>,
    #[serde(default)]
    pub level: Option<String>,
    #[serde(default)]
    pub unit: Option<String>,
}

fn yes() -> bool {
    true
}
fn flag() -> String {
    "flag".to_string()
}

/// Facts about a model needed for fit estimation + compatibility rules.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ModelInfo {
    pub repo_id: String,
    #[serde(default)]
    pub revision: Option<String>,
    #[serde(default)]
    pub architectures: Vec<String>,
    #[serde(default)]
    pub model_type: Option<String>,
    #[serde(default)]
    pub num_params: Option<i64>,
    #[serde(default)]
    pub num_active_params: Option<i64>,
    #[serde(default)]
    pub is_moe: bool,
    #[serde(default)]
    pub weight_bytes: Option<i64>,
    #[serde(default)]
    pub quantization: Option<String>,
    #[serde(default)]
    pub dtype: Option<String>,
    #[serde(default)]
    pub hidden_size: Option<i64>,
    #[serde(default)]
    pub num_layers: Option<i64>,
    #[serde(default)]
    pub num_attention_heads: Option<i64>,
    #[serde(default)]
    pub num_kv_heads: Option<i64>,
    #[serde(default)]
    pub head_dim: Option<i64>,
    #[serde(default)]
    pub max_position_embeddings: Option<i64>,
    #[serde(default)]
    pub sliding_window: Option<i64>,
    #[serde(default)]
    pub kv_lora_rank: Option<i64>,
    #[serde(default)]
    pub gated: bool,
    #[serde(default)]
    pub license: Option<String>,
    #[serde(default)]
    pub pipeline_tag: Option<String>,
    #[serde(default)]
    pub raw_config: Obj,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Hardware {
    pub gpu_ids: Vec<i64>,
    pub gpu_uuids: Vec<String>,
    pub gpu_names: Vec<String>,
    pub gpu_total_gib: Vec<f64>,
    pub gpu_free_gib: Vec<f64>,
    /// "12.0"
    pub compute_capability: String,
    pub host_ram_gib: f64,
    #[serde(default = "local")]
    pub host_id: String,
    /// "x86_64" | "aarch64"
    #[serde(default = "x86")]
    pub arch: String,
    #[serde(default)]
    pub unified_memory: bool,
}

fn local() -> String {
    "local".to_string()
}
fn x86() -> String {
    "x86_64".to_string()
}

/// What the lifecycle service needs to start an engine container. Produced by the adapter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LaunchSpec {
    pub image: String,
    pub argv: Vec<String>,
    #[serde(default)]
    pub env: Map<String, Value>,
    pub container_port: i64,
    pub health_path: String,
    pub metrics_path: String,
    #[serde(default = "models_path")]
    pub models_path: String,
    #[serde(default = "shm")]
    pub shm_size: String,
    #[serde(default = "yes")]
    pub ipc_host: bool,
    /// container path -> content
    #[serde(default)]
    pub files: Map<String, Value>,
    #[serde(default = "yes")]
    pub needs_hf_cache: bool,
    /// "stack" / "nofile" only
    #[serde(default)]
    pub ulimits: Map<String, Value>,
}

fn models_path() -> String {
    "/v1/models".to_string()
}
fn shm() -> String {
    "16g".to_string()
}

/// Everything an adapter needs to build one launch (see adapters/base.py LaunchContext for each format's rules).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LaunchContext {
    pub model: String,
    /// "hf" | "gguf" | "ollama" | "plain"
    #[serde(default = "hf")]
    pub model_format: String,
    #[serde(default)]
    pub model_path: Option<String>,
    pub served_name: String,
    pub hf_cache_container_path: String,
    #[serde(default)]
    pub params: Obj,
    pub hw: Hardware,
}

fn hf() -> String {
    "hf".to_string()
}

pub const MODEL_MOUNT: &str = "/models";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Compat {
    /// "ok" | "warn" | "block"
    pub level: String,
    pub code: String,
    pub message: String,
}

/// `EngineAdapter`. Sync, like the Python ABC. Errors from `build_launch` are problems (422 `invalid_params` etc.).
pub trait EngineAdapter: Send + Sync {
    fn id(&self) -> &str;
    fn display_name(&self) -> &str;
    /// pinned tag (never :latest)
    fn default_image(&self) -> &str;
    fn default_port(&self) -> i64 {
        8000
    }
    /// ModelFormat values this engine loads
    fn model_formats(&self) -> Vec<String> {
        vec!["hf".to_string()]
    }
    /// e.g. {"aarch64": "<arm64 image>"}; default_image otherwise (ordered)
    fn images_by_arch(&self) -> Vec<(String, String)> {
        Vec::new()
    }
    /// one server, many models loaded on demand (Ollama)
    fn multi_model(&self) -> bool {
        false
    }
    /// False only for the placeholder adapter (adapters/stub.rs): the routes that would serve its empty catalogs
    /// answer 501 not_ported instead. A real adapter keeps the default.
    fn ported(&self) -> bool {
        true
    }
    /// Metric-name prefixes the node keeps (`metric_prefixes` attribute). None: not declared.
    fn metric_prefixes(&self) -> Option<Vec<String>> {
        None
    }
    /// The pinned default image for a host architecture ("x86_64" | "aarch64").
    fn image_for(&self, arch: &str) -> String {
        self.images_by_arch()
            .into_iter()
            .find(|(a, _)| a == arch)
            .map_or_else(|| self.default_image().to_string(), |(_, i)| i)
    }
    /// Contract v2 entry point; the default bridges to `build_launch` like the Python base class.
    fn build_launch_ctx(&self, ctx: &LaunchContext) -> Result<LaunchSpec, Problem> {
        self.build_launch(
            &ctx.model,
            &ctx.params,
            &ctx.hw,
            &ctx.served_name,
            &ctx.hf_cache_container_path,
        )
    }
    /// Optional: a plain-language description per preset name (name -> Explain), ordered.
    fn preset_info(&self) -> Vec<(String, Explain)> {
        Vec::new()
    }
    fn param_catalog(&self) -> Vec<ParamSpec>;
    /// Named param bundles, ordered: 'balanced', 'max-throughput', ...
    fn presets(&self) -> Vec<(String, Obj)>;
    /// Human-readable errors (empty = valid). Unknown keys are errors.
    fn validate(&self, params: &Obj) -> Vec<String>;
    fn build_launch(
        &self,
        model: &str,
        params: &Obj,
        hw: &Hardware,
        served_name: &str,
        hf_cache_container_path: &str,
    ) -> Result<LaunchSpec, Problem>;
    fn compatibility(&self, model: &ModelInfo, params: &Obj, hw: &Hardware) -> Vec<Compat>;
    /// Engine-specific knobs for the shared fit estimator (mem_fraction, kv_bytes_per_elem, overhead_gib, max_len,
    /// max_seqs, tp, cuda_graph_gib), all optional.
    fn memory_model(&self, model: &ModelInfo, params: &Obj) -> FloatMap;
    /// Host RAM the engine pins on top of its baseline, GiB per component; NaN = needed but unknown.
    fn host_memory_gib(&self, _model: &ModelInfo, _params: &Obj, _hw: &Hardware) -> FloatMap {
        Vec::new()
    }
    /// Engine Prometheus text -> canonical keys (docs/ARCHITECTURE.md §Metrics).
    fn parse_metrics(&self, prometheus_text: &str) -> FloatMap;
    /// Structured progress from one startup log line ({phase, pct?, kv_cache_tokens?}).
    fn parse_startup_log(&self, line: &str) -> Option<Obj>;

    // ── the fallible forms (PORTING.md §A5) ─────────────────────────────────────────────────────────────────
    // Every caller outside an HTTP handler (metrics poller, lifecycle supervisor and preflight, bench, spawned tasks)
    // uses these: a Python exception the adapter raises comes back as `Err` instead of unwinding through the caller.
    // The defaults run the plain method under `pycompat::catch`; an adapter may implement them natively (returning
    // `Err(pycompat::exc(..))`) and write the plain method as `pycompat::or_raise(self.try_x(..))`.

    /// `validate`, with a raised exception as `Err`.
    fn try_validate(&self, params: &Obj) -> PyResult<Vec<String>> {
        catch(|| self.validate(params))
    }
    /// `build_launch_ctx`, with a raised exception as the outer `Err` (a refusal stays the inner `Err(Problem)`).
    fn try_build_launch_ctx(&self, ctx: &LaunchContext) -> PyResult<Result<LaunchSpec, Problem>> {
        catch(|| self.build_launch_ctx(ctx))
    }
    /// `compatibility`, with a raised exception as `Err`.
    fn try_compatibility(
        &self,
        model: &ModelInfo,
        params: &Obj,
        hw: &Hardware,
    ) -> PyResult<Vec<Compat>> {
        catch(|| self.compatibility(model, params, hw))
    }
    /// `memory_model`, with a raised exception as `Err` (preflight then falls back to the requested GPUs).
    fn try_memory_model(&self, model: &ModelInfo, params: &Obj) -> PyResult<FloatMap> {
        catch(|| self.memory_model(model, params))
    }
    /// `host_memory_gib`, with a raised exception as `Err`.
    fn try_host_memory_gib(
        &self,
        model: &ModelInfo,
        params: &Obj,
        hw: &Hardware,
    ) -> PyResult<FloatMap> {
        catch(|| self.host_memory_gib(model, params, hw))
    }
    /// `parse_metrics`, with a raised exception as `Err` (the poller keeps polling).
    fn try_parse_metrics(&self, prometheus_text: &str) -> PyResult<FloatMap> {
        catch(|| self.parse_metrics(prometheus_text))
    }
    /// `parse_startup_log`, with a raised exception as `Err`.
    fn try_parse_startup_log(&self, line: &str) -> PyResult<Option<Obj>> {
        catch(|| self.parse_startup_log(line))
    }
}
