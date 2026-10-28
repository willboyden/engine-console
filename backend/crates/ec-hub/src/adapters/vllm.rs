//! Port of backend/src/engine_console/adapters/vllm.py: the vLLM OpenAI server
//! (pinned image v0.27.1).
//!
//! Launch conventions (the LaunchSpec has no field for them, so the lifecycle service adds them): the image entrypoint
//! is already `vllm serve`, so `argv` starts with `--model`; `--ipc=host`/`--shm-size`, GPU pinning by UUID, the model
//! mounts and the engine network are docker flags. HF_TOKEN never enters the spec (the core injects it by name);
//! `api_key` travels in env (VLLM_API_KEY), never argv. Keys whose catalog flag is `env:NAME` become container env vars.
//! VLLM_SERVER_DEV_MODE is never set (it exposes an unauthenticated /collective_rpc).
//!
//! Model formats: `hf` (a repo id, the HF cache mounted read-only) and `plain` (a directory mounted at `/models`).
//!
//! Python semantics (see `adapters::pycompat`): the checks run on the JSON values the API received, with Python's
//! truthiness, `str()`, numeric tower and messages. Where the Python adapter raises an exception it does not catch
//! (only on values `validate` would never pass, or a GPU reporting 0 GiB total), this port panics, which the HTTP layer
//! answers with the same generic 500.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};
use std::time::Instant;

use regex::Regex;
use serde_json::{Map, Value};

use super::base::{
    Compat, EngineAdapter, FloatMap, Hardware, LaunchContext, LaunchSpec, MODEL_MOUNT, ModelInfo,
    Obj, ParamSpec,
};
use super::pycompat::{self as py, Anchor, JsonDoc, Num, compile};
use super::vllm_params::{
    ATTENTION_KV_DTYPES, ATTENTION_NOT_ON_SM120, DEFAULT_IMAGE, NARGS_KEYS, NEGATABLE_KEYS,
    build_catalog,
};
use super::vllm_parse::{Scrape, parse_log_line, scrape_to_canonical};
use crate::domain::errors::{Problem, unprocessable};

const GIB: f64 = 1_073_741_824.0;
/// measured once, see `memory_model`
const OUTSIDE_POOL_GIB: f64 = 2.5;
/// not measured
const NCCL_PER_EXTRA_RANK_GIB: f64 = 0.5;
/// Approximate bytes per KV element (scales and metadata not included): a planning figure for the fit estimator.
const KV_BYTES: &[(&str, f64)] = &[
    ("fp8", 1.0),
    ("fp8_e4m3", 1.0),
    ("fp8_e5m2", 1.0),
    ("fp8_inc", 1.0),
    ("fp8_ds_mla", 1.0),
    ("fp8_per_token_head", 1.0),
    ("int8_per_token_head", 1.0),
    ("int4_per_token_head", 0.5),
    ("nvfp4", 0.5625),
    ("turboquant_3bit_nc", 0.4),
    ("turboquant_4bit_nc", 0.5),
    ("turboquant_k3v4_nc", 0.45),
    ("turboquant_k8v4", 0.75),
];
/// Parameters removed from the catalog, with the reason a saved profile that still carries them should read.
pub const RETIRED: &[(&str, &str)] = &[
    (
        "image",
        "'image' is no longer a parameter: engine images are pinned in Settings, not per instance",
    ),
    (
        "nvfp4_gemm_backend",
        "'nvfp4_gemm_backend' is gone: VLLM_NVFP4_GEMM_BACKEND does not exist in vLLM 0.27.1; use linear_backend \
         (flashinfer_b12x, flashinfer_cutlass, marlin, ...)",
    ),
    (
        "flashinfer_moe_backend",
        "'flashinfer_moe_backend' is gone: VLLM_FLASHINFER_MOE_BACKEND does not exist in vLLM 0.27.1; use \
         moe_backend (flashinfer_b12x, flashinfer_cutlass, marlin, ...)",
    ),
];
/// Architecture / model-type fragments of hybrid models that keep a per-sequence Mamba or linear-attention state.
const HYBRID_MARKERS: &[&str] = &[
    "nemotronh",
    "nemotron_h",
    "mamba",
    "jamba",
    "zamba",
    "bamba",
    "falconh1",
    "falcon_h1",
    "qwen3next",
    "qwen3_next",
    "qwen3_5",
    "qwen3.5",
    "granitemoehybrid",
    "plamo2",
];
const NON_REASONING_MARKERS: &[&str] = &["coder", "instruct-2507"];

/// A repo-name pattern of the dialect tables (all `re.IGNORECASE`).
pub enum Pattern {
    Re(Regex),
    /// `qwen3(?![._\d])|qwen2\.?5|qwq`: the regex crate has no look-ahead, so the first branch is checked by hand.
    QwenJson,
}

impl Pattern {
    fn re(p: &str) -> Self {
        Self::Re(compile(p, true, Anchor::Search))
    }

    pub fn search(&self, s: &str) -> bool {
        match self {
            Self::Re(r) => r.is_match(s),
            Self::QwenJson => {
                static QWEN3: LazyLock<Regex> =
                    LazyLock::new(|| compile("qwen3", true, Anchor::Search));
                static REST: LazyLock<Regex> =
                    LazyLock::new(|| compile(r"qwen2\.?5|qwq", true, Anchor::Search));
                QWEN3.find_iter(s).any(|m| {
                    s[m.end()..]
                        .chars()
                        .next()
                        .is_none_or(|c| c != '.' && c != '_' && py::decimal(c).is_none())
                }) || REST.is_match(s)
            }
        }
    }
}

/// repo-name regex -> tool-call parsers that match the model's dialect (first = recommended).
pub static TOOL_DIALECTS: LazyLock<Vec<(Pattern, &'static [&'static str])>> = LazyLock::new(|| {
    vec![
        (Pattern::re(r"gpt-oss"), &["openai"][..]),
        (
            Pattern::re(r"qwen3[._-]?(coder|next)|qwen3[._][5-9]"),
            &["qwen3_xml", "qwen3_coder"],
        ),
        (Pattern::re(r"nemotron"), &["qwen3_coder", "qwen3_xml"]),
        (Pattern::QwenJson, &["hermes"]),
        (Pattern::re(r"hermes"), &["hermes"]),
        (
            Pattern::re(r"llama[-_]?4"),
            &["llama4_pythonic", "llama4_json"],
        ),
        (Pattern::re(r"llama[-_]?3"), &["llama3_json", "pythonic"]),
        (Pattern::re(r"mistral|ministral|devstral"), &["mistral"]),
        (Pattern::re(r"deepseek[-_]?v?3\.?2"), &["deepseek_v32"]),
        (Pattern::re(r"deepseek[-_]?v?3\.?1"), &["deepseek_v31"]),
        (Pattern::re(r"deepseek[-_]?v?4"), &["deepseek_v4"]),
        (Pattern::re(r"deepseek[-_]?(v3|r1)"), &["deepseek_v3"]),
        (Pattern::re(r"glm[-_]?(4\.[567]|5)"), &["glm47", "glm45"]),
        (Pattern::re(r"kimi[-_]?k2"), &["kimi_k2"]),
        (Pattern::re(r"kimi[-_]?k3"), &["kimi_k3"]),
        (Pattern::re(r"minimax[-_]?m2"), &["minimax_m2"]),
        (Pattern::re(r"minimax[-_]?m3"), &["minimax_m3"]),
        (Pattern::re(r"laguna"), &["poolside_v1"]),
        (Pattern::re(r"granite[-_]?4"), &["granite4"]),
        (Pattern::re(r"gemma[-_]?4"), &["gemma4"]),
    ]
});

/// repo-name regex -> reasoning parser.
pub static REASONING_HINTS: LazyLock<Vec<(Pattern, &'static str)>> = LazyLock::new(|| {
    vec![
        (Pattern::re(r"gpt-oss"), "openai_gptoss"),
        (Pattern::re(r"deepseek-r1|qwq"), "deepseek_r1"),
        (Pattern::re(r"qwen3"), "qwen3"),
        (Pattern::re(r"nemotron"), "nemotron_v3"),
        (Pattern::re(r"glm[-_]?(4\.[567]|5)"), "glm45"),
        (Pattern::re(r"kimi-k2-thinking"), "kimi_k2"),
        (Pattern::re(r"laguna"), "poolside_v1"),
        (Pattern::re(r"minimax[-_]?m2"), "minimax_m2_append_think"),
        (Pattern::re(r"olmo-3"), "olmo3"),
        (Pattern::re(r"gemma[-_]?4"), "gemma4"),
    ]
});

static IMAGE_VERSION: LazyLock<Regex> =
    LazyLock::new(|| compile(r":v?(\d+)\.(\d+)\.(\d+)", false, Anchor::Search));
static CC_MAJOR: LazyLock<Regex> = LazyLock::new(|| compile(r"^(\d+)\.\d+$", false, Anchor::Start));
static CC_FULL: LazyLock<Regex> = LazyLock::new(|| compile(r"\d+\.\d+", false, Anchor::Full));

/// `_image_version`: (major, minor, patch) of a pinned tag.
pub fn image_version(image: &str) -> Option<(String, String, String)> {
    let m = IMAGE_VERSION.captures(image)?;
    let n = |i: usize| py::int_str(&m[i]).map(|b| b.0);
    Some((n(1)?, n(2)?, n(3)?))
}

/// `_cc_major`: the major compute capability, or None.
pub fn cc_major(hw: &Hardware) -> Option<Num> {
    let cc = py::strip(&hw.compute_capability);
    let m = CC_MAJOR.captures(cc)?;
    py::int_str(&m[1]).map(|b| b.small().map_or(Num::Big(b), Num::I))
}

pub fn tool_dialect(repo: &str) -> Option<&'static [&'static str]> {
    TOOL_DIALECTS
        .iter()
        .find(|(p, _)| p.search(repo))
        .map(|(_, a)| *a)
}

pub fn reasoning_hint(repo: &str) -> Option<&'static str> {
    REASONING_HINTS
        .iter()
        .find(|(p, _)| p.search(repo))
        .map(|(_, h)| *h)
}

type Clock = Arc<dyn Fn() -> f64 + Send + Sync>;
/// rate state per served-model-name set: (t, prompt_total, generation_total)
type RateState = HashMap<Vec<String>, (f64, Option<f64>, Option<f64>)>;

pub struct VllmAdapter {
    catalog: Vec<ParamSpec>,
    clock: Clock,
    prev: Mutex<RateState>,
}

impl Default for VllmAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl VllmAdapter {
    /// `VllmAdapter()`: the clock is `time.monotonic`.
    pub fn new() -> Self {
        static START: LazyLock<Instant> = LazyLock::new(Instant::now);
        Self::with_clock(Arc::new(|| START.elapsed().as_secs_f64()))
    }

    /// `VllmAdapter(clock=...)`.
    pub fn with_clock(clock: Clock) -> Self {
        Self {
            catalog: build_catalog(),
            clock,
            prev: Mutex::new(HashMap::new()),
        }
    }

    fn spec(&self, key: &str) -> Option<&ParamSpec> {
        self.catalog.iter().find(|p| p.key == key)
    }

    /// `_check_value`: the errors for one non-None value. Err is the message of a ValueError Python raises (an
    /// `isdigit()` superscript given to `int()`).
    fn check_value(spec: &ParamSpec, val: &Value) -> Result<Vec<String>, String> {
        let (k, t) = (spec.key.as_str(), spec.type_.as_str());
        let one = |m: String| vec![m];
        Ok(match t {
            "bool" => {
                if val.is_boolean() {
                    vec![]
                } else {
                    one(format!("{k}: expected true/false"))
                }
            }
            "int" | "float" => {
                let n = match (val, Num::of(val)) {
                    (Value::Number(_), Some(n)) if n.as_f64().is_finite() => n,
                    _ => return Ok(one(format!("{k}: expected a number"))),
                };
                if t == "int" && matches!(n, Num::F(f) if f.fract() != 0.0) {
                    return Ok(one(format!("{k}: expected an integer")));
                }
                if let Some(min) = spec.min
                    && n.lt(&Num::F(min))
                {
                    return Ok(one(format!("{k}: must be >= {}", py::fmt_g(min))));
                }
                if let Some(max) = spec.max
                    && n.gt(&Num::F(max))
                {
                    return Ok(one(format!("{k}: must be <= {}", py::fmt_g(max))));
                }
                vec![]
            }
            "enum" => {
                let choices = spec.choices.clone().unwrap_or_default();
                let s = py::str_of(val);
                if choices.contains(&s) {
                    vec![]
                } else {
                    one(format!("{k}: '{s}' is not one of {}", choices.join(", ")))
                }
            }
            "string" => match val {
                Value::String(s) if !py::strip(s).is_empty() => {
                    if s.starts_with('-') && !spec.flag.starts_with("env:") {
                        one(format!(
                            "{k}: a value must not start with '-' (it would be read as another flag)"
                        ))
                    } else {
                        vec![]
                    }
                }
                _ => one(format!("{k}: expected a non-empty string")),
            },
            "string_list" => {
                let items = match val {
                    Value::Array(a) if !a.is_empty() => a,
                    _ => return Ok(one(format!("{k}: expected a non-empty list of strings"))),
                };
                let ok = items.iter().all(|x| match x {
                    Value::String(s) => !py::strip(s).is_empty(),
                    Value::Number(n) => !n.is_f64(),
                    _ => false,
                });
                if !ok {
                    return Ok(one(format!("{k}: expected a non-empty list of strings")));
                }
                if k == "cudagraph_capture_sizes" {
                    // `all(str(x).isdigit() and int(x) > 0 for x in val)`
                    for x in items {
                        let s = py::str_of(x);
                        if !py::isdigit(&s) {
                            return Ok(one(format!("{k}: entries must be positive integers")));
                        }
                        // isdigit() but not decimal (a superscript): Python's int() raises ValueError here
                        let Some(n) = py::int_str(&s) else {
                            return Err(format!(
                                "invalid literal for int() with base 10: {}",
                                ec_proto::pyfmt::repr_str(&s)
                            ));
                        };
                        if n.is_zero() || n.is_negative() {
                            return Ok(one(format!("{k}: entries must be positive integers")));
                        }
                    }
                }
                if items.iter().any(|x| py::str_of(x).starts_with('-')) {
                    return Ok(one(format!(
                        "{k}: entries must not start with '-' (they would be read as another flag)"
                    )));
                }
                vec![]
            }
            "json" => {
                if let Value::String(s) = val {
                    match JsonDoc::loads(s) {
                        Err(e) => one(format!("{k}: invalid JSON ({})", e.0)),
                        Ok(doc) if !doc.is_container() => {
                            one(format!("{k}: expected a JSON object"))
                        }
                        Ok(_) => vec![],
                    }
                } else if val.is_object() || val.is_array() {
                    vec![]
                } else {
                    one(format!("{k}: expected a JSON object"))
                }
            }
            _ => vec![],
        })
    }

    /// `validate`; Err is the message of a ValueError Python raises (build_launch_ctx turns it into a 422).
    fn validate_inner(&self, params: &Obj) -> Result<Vec<String>, String> {
        let mut errs: Vec<String> = Vec::new();
        for (key, val) in params {
            let Some(spec) = self.spec(key) else {
                errs.push(RETIRED.iter().find(|(k, _)| k == key).map_or_else(
                    || format!("unknown parameter '{key}'"),
                    |(_, m)| (*m).to_string(),
                ));
                continue;
            };
            if val.is_null() {
                continue;
            }
            errs.extend(Self::check_value(spec, val)?);
        }
        if !errs.is_empty() {
            return Ok(errs);
        }
        let g = |k: &str| params.get(k);
        if py::truthy_opt(g("enable_auto_tool_choice")) && !py::truthy_opt(g("tool_call_parser")) {
            errs.push(
                "enable_auto_tool_choice requires tool_call_parser (vLLM refuses to start otherwise)".into(),
            );
        }
        if py::truthy_opt(g("lora_modules")) && !py::truthy_opt(g("enable_lora")) {
            errs.push("lora_modules requires enable_lora=true".into());
        }
        // vLLM merges the spec_* shortcuts into speculative_config and only refuses a key given in both places.
        let has = |k: &str| -> Option<bool> {
            match g("speculative_config") {
                Some(Value::String(s)) => JsonDoc::loads(s)
                    .ok()
                    .filter(JsonDoc::is_dict)
                    .map(|d| d.has_key(k)),
                Some(Value::Object(m)) => Some(m.contains_key(k)),
                _ => None,
            }
        };
        for (key, short) in [
            ("method", "spec_method"),
            ("model", "spec_model"),
            ("num_speculative_tokens", "spec_tokens"),
        ] {
            if !py::is_none(g(short)) && has(key) == Some(true) {
                errs.push(format!(
                    "{short} and speculative_config['{key}'] are mutually exclusive (vLLM refuses to start)"
                ));
            }
        }
        let (cpu_loras, loras) = (g("max_cpu_loras"), g("max_loras"));
        if py::truthy_opt(cpu_loras)
            && py::truthy_opt(loras)
            && Num::need(cpu_loras, "<").lt(&Num::need(loras, "<"))
        {
            errs.push("max_cpu_loras must be >= max_loras".into());
        }
        if s_opt(g("allowed_local_media_path")) == Some("/") {
            errs.push("allowed_local_media_path must not be '/': any prompt could read any container file".into());
        }
        Ok(errs)
    }

    /// `_render`: CLI tokens for one param. bool -> --x / --no-x (only when the flag is negatable).
    fn render(spec: &ParamSpec, val: &Value) -> Vec<String> {
        let flag = spec.flag.clone();
        match spec.type_.as_str() {
            "bool" => {
                if py::truthy(val) {
                    vec![flag]
                } else if NEGATABLE_KEYS.contains(&spec.key.as_str()) {
                    vec![format!("--no-{}", flag.get(2..).unwrap_or(""))]
                } else {
                    vec![]
                }
            }
            "string_list" => {
                let items: Vec<String> = val
                    .as_array()
                    .map(|a| a.iter().map(py::str_of).collect())
                    .unwrap_or_default();
                if NARGS_KEYS.contains(&spec.key.as_str()) {
                    let mut out = vec![flag];
                    out.extend(items);
                    out
                } else {
                    items.into_iter().flat_map(|x| [flag.clone(), x]).collect()
                }
            }
            "json" => {
                let text = match val {
                    Value::String(s) => JsonDoc::loads(s)
                        .unwrap_or_else(|e| py::raise("JSONDecodeError", e.0))
                        .dumps_compact(),
                    other => crate::pyjson::dumps_compact(other),
                };
                vec![flag, text]
            }
            "float" => vec![flag, py::fmt_g(py::float_or_raise(Some(val)))],
            "int" => vec![flag, py::int_or_raise(Some(val)).to_string()],
            _ => vec![flag, py::str_of(val)],
        }
    }
}

fn compat(level: &str, code: &str, message: impl Into<String>) -> Compat {
    Compat {
        level: level.into(),
        code: code.into(),
        message: message.into(),
    }
}

/// `min(values)` of a non-empty float list, as Python picks it (a later item replaces the current one only when it is
/// smaller, so a NaN in front stays).
pub(crate) fn py_min(values: &[f64]) -> f64 {
    let mut m = values.first().copied().unwrap_or(f64::NAN);
    for &x in values.iter().skip(1) {
        if x < m {
            m = x;
        }
    }
    m
}

/// `str(ValidationError)` of `LaunchSpec(argv=...)` when argv holds non-strings (pydantic 2.13: no int -> str coercion).
pub(crate) fn launchspec_argv_error(bad: &[(usize, &Value)]) -> String {
    let n = bad.len();
    let mut out = format!(
        "{n} validation error{} for LaunchSpec",
        if n == 1 { "" } else { "s" }
    );
    for (i, v) in bad {
        out.push_str(&format!(
            "\nargv.{i}\n  Input should be a valid string [type=string_type, input_value={}, input_type={}]\n    For \
             further information visit https://errors.pydantic.dev/2.13/v/string_type",
            ec_proto::pyfmt::repr_value(v),
            py::type_name(Some(v))
        ));
    }
    out
}

fn s_opt(v: Option<&Value>) -> Option<&str> {
    v.and_then(Value::as_str)
}

impl EngineAdapter for VllmAdapter {
    fn id(&self) -> &str {
        "vllm"
    }
    fn display_name(&self) -> &str {
        "vLLM"
    }
    fn default_image(&self) -> &str {
        DEFAULT_IMAGE
    }
    fn default_port(&self) -> i64 {
        8000
    }
    fn model_formats(&self) -> Vec<String> {
        vec!["hf".into(), "plain".into()]
    }
    fn metric_prefixes(&self) -> Option<Vec<String>> {
        Some(vec!["vllm:".into()])
    }

    fn param_catalog(&self) -> Vec<ParamSpec> {
        self.catalog.clone()
    }

    fn presets(&self) -> Vec<(String, Obj)> {
        let o = |v: Value| match v {
            Value::Object(m) => m,
            _ => Map::new(),
        };
        vec![
            // vLLM's own defaults plus a 64K window; safe on any model whose native length allows it.
            (
                "balanced".into(),
                o(
                    serde_json::json!({"gpu_memory_utilization": 0.90, "max_model_len": 65536,
                    "enable_prefix_caching": true, "performance_mode": "balanced"}),
                ),
            ),
            (
                "max-throughput".into(),
                o(
                    serde_json::json!({"gpu_memory_utilization": 0.92, "max_model_len": 32768,
                    "max_num_seqs": 512, "max_num_batched_tokens": 16384,
                    "enable_prefix_caching": true, "enable_chunked_prefill": true,
                    "async_scheduling": true, "performance_mode": "throughput"}),
                ),
            ),
            (
                "low-latency".into(),
                o(
                    serde_json::json!({"gpu_memory_utilization": 0.90, "max_model_len": 16384,
                    "max_num_seqs": 8, "max_num_batched_tokens": 2048,
                    "enable_prefix_caching": true, "async_scheduling": true,
                    "performance_mode": "interactivity", "stream_interval": 1}),
                ),
            ),
            // fp8 KV doubles token capacity; few sequences because each one is big. Attention backend left on auto:
            // on sm_120 forcing FLASH_ATTN with fp8 KV is refused (see compatibility()).
            (
                "long-context".into(),
                o(
                    serde_json::json!({"gpu_memory_utilization": 0.92, "max_model_len": 262144,
                    "kv_cache_dtype": "fp8", "max_num_seqs": 8,
                    "max_num_batched_tokens": 8192, "enable_prefix_caching": true,
                    "enable_chunked_prefill": true}),
                ),
            ),
            // hermes suits Hermes-family and Qwen2.5-style models; change the parser for other families.
            (
                "tool-agent".into(),
                o(
                    serde_json::json!({"gpu_memory_utilization": 0.90, "max_model_len": 65536, "max_num_seqs": 32,
                    "enable_prefix_caching": true, "enable_auto_tool_choice": true,
                    "tool_call_parser": "hermes"}),
                ),
            ),
            // qwen3 + qwen3_xml is the usual pairing for Qwen3.x reasoning models.
            (
                "reasoning".into(),
                o(
                    serde_json::json!({"gpu_memory_utilization": 0.90, "max_model_len": 131072, "max_num_seqs": 16,
                    "enable_prefix_caching": true, "enable_chunked_prefill": true,
                    "reasoning_parser": "qwen3", "enable_auto_tool_choice": true,
                    "tool_call_parser": "qwen3_xml"}),
                ),
            ),
        ]
    }

    fn validate(&self, params: &Obj) -> Vec<String> {
        self.validate_inner(params)
            .unwrap_or_else(|m| py::raise("ValueError", &m))
    }

    fn build_launch(
        &self,
        model: &str,
        params: &Obj,
        hw: &Hardware,
        served_name: &str,
        hf_cache_container_path: &str,
    ) -> Result<LaunchSpec, Problem> {
        self.build_launch_ctx(&LaunchContext {
            model: model.into(),
            model_format: "hf".into(),
            model_path: None,
            served_name: served_name.into(),
            hf_cache_container_path: hf_cache_container_path.into(),
            params: params.clone(),
            hw: hw.clone(),
        })
    }

    /// Err is Python's ValueError, already as the core raises it: 422 `invalid_params` with `str(e)`.
    fn build_launch_ctx(&self, ctx: &LaunchContext) -> Result<LaunchSpec, Problem> {
        let mut formats = self.model_formats();
        formats.sort();
        if !formats.contains(&ctx.model_format) {
            return Err(unprocessable(format!(
                "vLLM cannot load {} models (it loads: {})",
                ctx.model_format,
                formats.join(", ")
            )));
        }
        let errs = self.validate_inner(&ctx.params).map_err(unprocessable)?;
        if !errs.is_empty() {
            return Err(unprocessable(errs.join("; ")));
        }
        let plain = ctx.model_format == "plain";
        let model_arg = if plain {
            ctx.model_path
                .clone()
                .filter(|p| !p.is_empty())
                .unwrap_or_else(|| MODEL_MOUNT.to_string())
        } else {
            ctx.model.clone()
        };
        if model_arg.is_empty() || model_arg.starts_with('-') {
            return Err(unprocessable("invalid model identifier"));
        }
        let mut argv: Vec<String> = vec![
            "--model".into(),
            model_arg,
            "--host".into(),
            "0.0.0.0".into(),
            "--port".into(),
            self.default_port().to_string(),
        ];
        // Telemetry off (deny-by-default egress): vLLM honours VLLM_NO_USAGE_STATS and DO_NOT_TRACK.
        let mut env = Map::new();
        let set = |env: &mut Map<String, Value>, k: &str, v: &str| {
            env.insert(k.into(), Value::String(v.into()));
        };
        set(&mut env, "VLLM_NO_USAGE_STATS", "1");
        set(&mut env, "DO_NOT_TRACK", "1");
        set(&mut env, "HF_HUB_DISABLE_TELEMETRY", "1");
        // JIT kernels compile for the architecture list given here; the host's real capability avoids building others.
        let cc = py::strip(&ctx.hw.compute_capability);
        if CC_FULL.is_match(cc) {
            set(&mut env, "TORCH_CUDA_ARCH_LIST", cc);
        }
        if plain {
            set(&mut env, "HF_HUB_OFFLINE", "1"); // a local directory: never wait on the hub
        } else {
            set(&mut env, "HF_HOME", &ctx.hf_cache_container_path);
        }
        let params = &ctx.params;
        // `[ctx.served_name] + [n for n in params.get("served_model_name") or [] if n != ctx.served_name]`: an int
        // entry (validate allows it) stays an int, and pydantic then refuses the LaunchSpec (a ValueError).
        let mut extra: Vec<(usize, &Value)> = Vec::new();
        argv.push("--served-model-name".into());
        argv.push(ctx.served_name.clone());
        if let Some(v) = params.get("served_model_name")
            && py::truthy(v)
        {
            for n in v.as_array().map_or(&[][..], Vec::as_slice) {
                match n {
                    Value::String(s) if *s == ctx.served_name => {}
                    Value::String(s) => argv.push(s.clone()),
                    other => {
                        extra.push((argv.len(), other));
                        argv.push(py::str_of(other));
                    }
                }
            }
        }
        if !extra.is_empty() {
            return Err(unprocessable(launchspec_argv_error(&extra)));
        }
        for spec in &self.catalog {
            let Some(val) = params.get(&spec.key).filter(|v| !v.is_null()) else {
                continue;
            };
            if spec.key == "served_model_name" || (spec.key == "revision" && plain) {
                continue; // a directory has no hub revision
            }
            if let Some(name) = spec.flag.strip_prefix("env:") {
                let v = if spec.type_ == "bool" {
                    if py::truthy(val) {
                        "1".to_string()
                    } else {
                        "0".to_string()
                    }
                } else {
                    py::str_of(val)
                };
                set(&mut env, name, &v);
            } else {
                argv.extend(Self::render(spec, val));
            }
        }
        Ok(LaunchSpec {
            image: DEFAULT_IMAGE.into(),
            argv,
            env,
            container_port: self.default_port(),
            health_path: "/health".into(),
            metrics_path: "/metrics".into(),
            models_path: "/v1/models".into(),
            shm_size: "16g".into(),
            ipc_host: true,
            files: Map::new(),
            needs_hf_cache: !plain,
            ulimits: Map::new(),
        })
    }

    #[allow(clippy::too_many_lines)]
    fn compatibility(&self, model: &ModelInfo, params: &Obj, hw: &Hardware) -> Vec<Compat> {
        let mut out: Vec<Compat> = Vec::new();
        let mut add = |lvl: &str, code: &str, msg: String| out.push(compat(lvl, code, msg));
        let g = |k: &str| params.get(k).filter(|v| !v.is_null());
        let repo = model.repo_id.to_lowercase();
        let mut blob_parts = vec![model.model_type.clone().unwrap_or_default()];
        blob_parts.extend(model.architectures.iter().cloned());
        let arch_blob = blob_parts.join(" ").to_lowercase();
        let quant = model
            .quantization
            .clone()
            .unwrap_or_default()
            .to_lowercase();
        let cc = cc_major(hw);
        let cc_is = |n: i128| cc.as_ref().is_some_and(|c| c.same(&Num::I(n)));
        let is_nvfp4 = quant.contains("nvfp4")
            || (quant.contains("fp4") && !quant.contains("mx"))
            || repo.contains("nvfp4");
        let is_mxfp4 = quant.contains("mxfp4") || repo.contains("mxfp4");
        let is_gpt_oss =
            repo.contains("gpt-oss") || model.architectures.iter().any(|a| a.contains("GptOss"));
        let moe_backend = g("moe_backend");
        let moe_is = |s: &str| moe_backend.is_some_and(|v| py::py_eq(v, &Value::String(s.into())));
        let tp = py::or_num(g("tensor_parallel_size"), Num::I(1));
        let pp = py::or_num(g("pipeline_parallel_size"), Num::I(1));
        let dp = py::or_num(g("data_parallel_size"), Num::I(1));
        let ver = image_version(DEFAULT_IMAGE);
        let one = Num::I(1);

        // --- quantization versus GPU generation
        if cc_is(12) && is_nvfp4 && model.is_moe && (moe_backend.is_none() || moe_is("auto")) {
            add(
                "warn",
                "nvfp4_moe_sm120",
                "NVFP4 MoE on sm_120 can fall back to a slow Marlin kernel in vLLM. Use the FP8 build, run it on \
                 SGLang, or time moe_backend flashinfer_b12x (the sm_12x kernel in the vLLM 0.27.1 help), cutlass and \
                 marlin."
                    .into(),
            );
        }
        if cc_is(12) && is_mxfp4 && model.is_moe && !is_gpt_oss {
            add(
                "block",
                "mxfp4_moe_sm120",
                "MXFP4 MoE asserts on sm_120 except gpt-oss. Use an FP8 build of this model."
                    .into(),
            );
        }
        if is_gpt_oss {
            add(
                "warn",
                "gpt_oss_prefer_ollama",
                "gpt-oss ships native MXFP4 weights: leave quantization unset. If vLLM does not start on your GPU \
                 generation, run the model with llama.cpp or Ollama instead."
                    .into(),
            );
        }
        if model.gated {
            add(
                "warn",
                "gated_needs_hf_token",
                "Gated model: set HF_TOKEN (Settings) and accept the license on the model page before downloading or \
                 starting."
                    .into(),
            );
        }
        // --- parallel sizes
        let world = tp.mul(&pp).mul(&dp);
        let n_gpus = Num::I(i128::try_from(hw.gpu_ids.len()).unwrap_or(i128::MAX));
        if world.gt(&n_gpus) {
            add(
                "block",
                "tp_exceeds_gpus",
                format!(
                    "tensor*pipeline*data parallel = {world} but only {} GPU(s) selected. Lower the parallel sizes or \
                     select more GPUs.",
                    hw.gpu_ids.len()
                ),
            );
        } else if tp.gt(&one) {
            add(
                "warn",
                "tp_over_pcie",
                "Tensor parallelism all-reduces on every layer; without NVLink that traffic crosses PCIe (workstation \
                 cards usually have no NVLink). Prefer one model per card unless it does not fit."
                    .into(),
            );
        }
        let (heads, kv) = (model.num_attention_heads, model.num_kv_heads);
        let heads_n = heads.filter(|h| *h != 0).map(|h| Num::I(i128::from(h)));
        let kv_n = kv.filter(|h| *h != 0).map(|h| Num::I(i128::from(h)));
        if tp.gt(&one) && heads_n.as_ref().is_some_and(|h| h.rem(&tp).truthy()) {
            add(
                "block",
                "tp_heads_indivisible",
                format!(
                    "{} attention heads are not divisible by tensor_parallel_size={tp}. Pick a divisor.",
                    heads.unwrap_or(0)
                ),
            );
        } else if tp.gt(&one)
            && let Some(k) = kv_n.as_ref()
            && k.rem(&tp).truthy()
            && tp.rem(k).truthy()
        {
            add(
                "block",
                "tp_kv_heads_indivisible",
                format!(
                    "{} KV heads and tensor_parallel_size={tp}: one must divide the other.",
                    kv.unwrap_or(0)
                ),
            );
        }
        // --- KV cache dtype and attention backend
        let (kvd, backend) = (g("kv_cache_dtype"), g("attention_backend"));
        // `backend in <dict>` / `in <frozenset>` hashes it: a list or dict raises TypeError
        let unhashable = || {
            if backend.is_some_and(|b| b.is_array() || b.is_object()) {
                py::raise(
                    "TypeError",
                    &format!("unhashable type: '{}'", py::type_name(backend)),
                );
            }
        };
        if py::truthy_opt(kvd) {
            unhashable();
        }
        let backend_s = backend.map(py::str_of);
        let kv_table = backend
            .and_then(Value::as_str)
            .and_then(|b| ATTENTION_KV_DTYPES.iter().find(|(n, _)| *n == b))
            .map(|(_, d)| *d);
        if py::truthy_opt(kvd)
            && let Some(dtypes) = kv_table
        {
            let kvs = py::str_of(kvd.unwrap_or(&Value::Null));
            let b = backend_s.unwrap_or_default();
            if cc_is(12)
                && kvs.starts_with("fp8")
                && (b == "FLASH_ATTN" || b == "FLASH_ATTN_DIFFKV")
            {
                add(
                    "block",
                    "kv_fp8_flash_attn",
                    "On sm_120 FLASH_ATTN runs FlashAttention 2, which cannot use an fp8 KV cache; vLLM stops with \
                     'Selected backend FLASH_ATTN is not valid for this configuration'. Set attention_backend to \
                     FLASHINFER (or leave it unset), or set kv_cache_dtype to auto."
                        .into(),
                );
            } else if !dtypes.contains(&kvs.as_str()) {
                let mut sorted: Vec<&str> = dtypes.to_vec();
                sorted.sort_unstable();
                add(
                    "block",
                    "kv_dtype_backend_mismatch",
                    format!(
                        "attention_backend {b} does not support kv_cache_dtype {kvs} (it supports: {}). Pick another \
                         backend or dtype.",
                        sorted.join(", ")
                    ),
                );
            }
        }
        if py::truthy_opt(kvd)
            && py::str_of(kvd.unwrap_or(&Value::Null)).starts_with("fp8")
            && (quant.is_empty() || quant == "none")
        {
            add(
                "warn",
                "kv_fp8_accuracy",
                "fp8 KV cache trades accuracy for capacity; vLLM logs 'Using uncalibrated q_scale 1.0 and/or \
                 prob_scale 1.0 with fp8 attention' when the checkpoint ships no scales. Compare outputs on your task \
                 before adopting it."
                    .into(),
            );
        }
        if cc_is(12) {
            unhashable();
        }
        if cc_is(12)
            && let Some(b) = backend.and_then(Value::as_str)
            && ATTENTION_NOT_ON_SM120.contains(&b)
        {
            add(
                "block",
                "mla_backend_arch",
                format!(
                    "{b} does not support compute capability 12.0 in vLLM 0.27.1. Use TRITON_MLA (or \
                     FLASHINFER_MLA_SPARSE_SM120 for sparse-MLA models), or leave attention_backend unset."
                ),
            );
        }
        // --- kernel backends
        let linear = g("linear_backend");
        let linear_is = |s: &str| linear.is_some_and(|v| py::py_eq(v, &Value::String(s.into())));
        if moe_is("aiter")
            || moe_is("flydsl")
            || linear_is("aiter")
            || linear_is("xpu")
            || linear_is("xpu_woq")
            || (moe_is("hpc") && cc.is_some() && !cc_is(9))
        {
            add(
                "block",
                "kernel_backend_not_for_gpu",
                "The chosen moe_backend or linear_backend is not built for this GPU: aiter and flydsl are AMD-only, \
                 xpu is Intel-only and hpc is FP8 on Hopper only (vLLM 0.27.1 help). Use auto, cutlass, marlin or \
                 flashinfer_b12x."
                    .into(),
            );
        }
        if moe_is("deep_gemm") && quant != "fp8" && quant != "fp8_block" {
            add(
                "warn",
                "deep_gemm_fp8_only",
                "The deep_gemm MoE backend only runs FP8 block-quantized checkpoints (vLLM 0.27.1 help). Use an FP8 \
                 build, or leave moe_backend on auto."
                    .into(),
            );
        }
        if py::truthy_opt(g("enable_expert_parallel")) && !model.is_moe {
            add(
                "warn",
                "ep_dense_model",
                "enable_expert_parallel has no effect on a dense model.".into(),
            );
        }
        let hash = g("prefix_caching_hash_algo");
        let hash_is = |s: &str| hash.is_some_and(|v| py::py_eq(v, &Value::String(s.into())));
        if hash_is("xxhash") || hash_is("xxhash_cbor") {
            add(
                "warn",
                "prefix_hash_xxhash_missing",
                "The xxhash prefix-cache hash needs the optional xxhash package, which the vLLM 0.27.1 image does not \
                 include; start-up fails unless your image has it. Use sha256 (the default) or sha256_cbor."
                    .into(),
            );
        }
        let q = g("quantization");
        let model_q = model.quantization.clone().unwrap_or_default();
        if py::truthy_opt(q) && !model_q.is_empty() {
            let qs = py::str_of(q.unwrap_or(&Value::Null));
            if !quant.contains(&qs) && !qs.contains(&quant) && model_q.to_lowercase() != "none" {
                add(
                    "block",
                    "quant_override_prequantized",
                    format!(
                        "The checkpoint is already {model_q}; forcing quantization={qs} misloads it. Unset \
                         quantization and let vLLM auto-detect."
                    ),
                );
            }
        }
        if !py::truthy_opt(q)
            && model_q.is_empty()
            && let Some(wb) = model.weight_bytes.filter(|w| *w != 0)
            && !hw.gpu_total_gib.is_empty()
        {
            let min_total = py_min(&hw.gpu_total_gib);
            if (wb as f64) / GIB > 0.9 * min_total {
                add(
                    "warn",
                    "fp8_recommended",
                    "Unquantized weights nearly fill a GPU. Set quantization=fp8 (the usual choice) or pick an \
                     FP8/NVFP4-dense build."
                        .into(),
                );
            }
        }
        // --- context length
        let pos = model.max_position_embeddings.filter(|p| *p != 0);
        let mml = g("max_model_len").filter(|v| py::truthy(v));
        if let (Some(mml), Some(pos)) = (mml, pos)
            && Num::need(Some(mml), ">").gt(&Num::I(i128::from(pos)))
        {
            if py::truthy_opt(g("hf_overrides")) {
                add(
                    "warn",
                    "yarn_short_context_quality",
                    "Context extended past the trained length via hf_overrides (static YaRN scales all positions): \
                     short-prompt quality can drop. Measure before adopting."
                        .into(),
                );
            } else if !py::truthy_opt(g("allow_long_max_model_len")) {
                add(
                    "block",
                    "max_len_exceeds_model",
                    format!(
                        "max_model_len {} exceeds the model's {pos}. Extend with a YaRN hf_overrides block or lower \
                         max_model_len.",
                        py::str_of(mml)
                    ),
                );
            }
        }
        // --- memory share (image source: request_memory in v1/worker/utils.py)
        let util = py::or_num(g("gpu_memory_utilization"), Num::F(0.92));
        if !py::truthy_opt(g("kv_cache_memory_bytes")) {
            for ((gid, total), free) in hw
                .gpu_ids
                .iter()
                .zip(&hw.gpu_total_gib)
                .zip(&hw.gpu_free_gib)
            {
                let need = util.mul(&Num::F(*total)).as_f64();
                if need > free + 0.5 {
                    if *total == 0.0 {
                        py::raise("ZeroDivisionError", "float division by zero");
                    }
                    let lower = f64::max(0.05, (free - 1.0) / total);
                    add(
                        "block",
                        "insufficient_free_vram",
                        format!(
                            "GPU {gid}: gpu_memory_utilization {} needs {} GiB free but only {} GiB is. Stop whatever \
                             else uses the card or lower it to {}.",
                            py::fmt_g(util.as_f64()),
                            py::fixed(need, 1),
                            py::fixed(*free, 1),
                            py::fixed(lower, 2)
                        ),
                    );
                    break;
                }
            }
        }
        if util.gt(&Num::F(0.95)) {
            add(
                "warn",
                "util_very_high",
                "gpu_memory_utilization > 0.95 leaves little headroom for CUDA graphs and activation spikes; OOM at \
                 startup is likely."
                    .into(),
            );
        }
        if HYBRID_MARKERS.iter().any(|m| arch_blob.contains(m))
            && !py::truthy_opt(g("max_num_seqs"))
            && !py::truthy_opt(g("enforce_eager"))
        {
            add(
                "warn",
                "hybrid_max_num_seqs",
                "Hybrid (Mamba-style) models need one cache block per running sequence. With the default max_num_seqs \
                 (1024 on cards of 70 GiB or more) vLLM can stop at CUDA graph capture with 'max_num_seqs (N) exceeds \
                 available Mamba cache blocks (M)', whatever the memory share. Set max_num_seqs (for example 256) or \
                 raise gpu_memory_utilization."
                    .into(),
            );
        }
        if py::truthy_opt(g("enforce_eager")) {
            add(
                "warn",
                "eager_slow",
                "enforce_eager disables CUDA graphs and noticeably slows decode.".into(),
            );
        }
        if py::truthy_opt(g("cpu_offload_gb")) {
            add(
                "warn",
                "cpu_offload_slow",
                "CPU weight offload streams weights over PCIe every step; expect a large slowdown."
                    .into(),
            );
        }
        if py::truthy_opt(g("trust_remote_code")) {
            add(
                "warn",
                "trust_remote_code",
                "trust_remote_code runs Python from the model repo. Only enable for repos you have reviewed.".into(),
            );
        }
        if py::truthy_opt(g("enable_sleep_mode")) {
            add(
                "warn",
                "sleep_mode_dev_router",
                "The /sleep endpoints are registered only when VLLM_SERVER_DEV_MODE is set, which also exposes \
                 unauthenticated /collective_rpc; the console never sets it."
                    .into(),
            );
        }
        // --- tool and reasoning parsers
        let agentic = matches!(
            model.pipeline_tag.as_deref(),
            None | Some("text-generation" | "image-text-to-text" | "any-to-any")
        );
        let tool = g("tool_call_parser");
        let allowed = tool_dialect(&repo);
        if agentic
            && !py::truthy_opt(tool)
            && let Some(a) = allowed
        {
            add(
                "warn",
                "tool_parser_missing",
                format!(
                    "No tool-call parser set; agent/tool use will not work. For this model family set \
                     enable_auto_tool_choice=true and tool_call_parser={}.",
                    a[0]
                ),
            );
        }
        if py::truthy_opt(tool)
            && let Some(a) = allowed
            && !tool.and_then(Value::as_str).is_some_and(|t| a.contains(&t))
        {
            add(
                "warn",
                "tool_parser_dialect_mismatch",
                format!(
                    "tool_call_parser={} does not match this model family's tool-call dialect; use {}. A mismatch \
                     leaves raw tool markup in the message content.",
                    py::str_of(tool.unwrap_or(&Value::Null)),
                    a.join(" or ")
                ),
            );
        }
        if py::truthy_opt(tool) && !py::truthy_opt(g("enable_auto_tool_choice")) {
            add(
                "warn",
                "tool_parser_inactive",
                "tool_call_parser has no effect without enable_auto_tool_choice=true.".into(),
            );
        }
        if agentic
            && !py::truthy_opt(g("reasoning_parser"))
            && !NON_REASONING_MARKERS.iter().any(|m| repo.contains(m))
            && let Some(hint) = reasoning_hint(&repo)
        {
            add(
                "warn",
                "reasoning_parser_missing",
                format!(
                    "This family emits reasoning; without reasoning_parser={hint} the chain-of-thought leaks into \
                     message.content."
                ),
            );
        }
        // --- image pin versus model family (recalled from the model card, not checked in the image)
        if repo.contains("omni")
            && repo.contains("nemotron")
            && let Some((a, b, c)) = &ver
            && (a.as_str(), b.as_str(), c.as_str()) != ("0", "20", "0")
        {
            add(
                "warn",
                "image_pin_nemotron_omni",
                format!(
                    "NVIDIA's model card names vLLM 0.20.0 for Nemotron 3 Nano Omni; this console pins vLLM {a}.{b}.{c}, \
                     so the model may fail to load. If it does, pin vllm/vllm-openai:v0.20.0 in Settings."
                ),
            );
        }
        out
    }

    fn memory_model(&self, model: &ModelInfo, params: &Obj) -> FloatMap {
        let g = |k: &str| params.get(k);
        let tp = py::int_or_raise(Some(py::or_value(
            g("tensor_parallel_size"),
            &Value::from(1),
        )));
        let tp_f = tp.as_f64();
        let kv_key = py::str_of(py::or_value(g("kv_cache_dtype"), &Value::from("auto")));
        let mut out: FloatMap = vec![
            (
                "mem_fraction".into(),
                py::float_or_raise(Some(py::or_value(
                    g("gpu_memory_utilization"),
                    &Value::from(0.92),
                ))),
            ),
            (
                "kv_bytes_per_elem".into(),
                KV_BYTES
                    .iter()
                    .find(|(k, _)| *k == kv_key)
                    .map_or(2.0, |(_, b)| *b),
            ),
            // Memory the process holds on top of its gpu_memory_utilization share (CUDA context, NCCL buffers): one
            // measurement (vLLM 0.27.1, one GPU, enforce_eager, 0.5B model, share 0.15 of a 95.6 GiB card) showed
            // 16.86 GiB held against a 14.34 GiB share, so about 2.5 GiB; +0.5 GiB per extra TP rank is a guess.
            ("overhead_gib".into(), 0.0),
            (
                "outside_share_gib".into(),
                OUTSIDE_POOL_GIB + NCCL_PER_EXTRA_RANK_GIB * (tp_f - 1.0),
            ),
            // 0.27.1 worker/utils.py request_memory: refuses to start unless share x card total is free.
            ("claims_share".into(), 1.0),
            // vLLM picks 1024 on GPUs of 70 GiB or more and 256 below that; hardware is not known here.
            (
                "max_seqs".into(),
                py::float_or_raise(Some(py::or_value(g("max_num_seqs"), &Value::from(256)))),
            ),
            ("tp".into(), tp_f),
            // measured 0.59-0.89 GiB CUDA-graph pool, inside the share on 0.27.1
            (
                "cuda_graph_gib".into(),
                if py::truthy_opt(g("enforce_eager")) {
                    0.0
                } else {
                    0.75
                },
            ),
        ];
        let pos = model.max_position_embeddings.map(Value::from);
        let max_len = match g("max_model_len") {
            Some(v) if py::truthy(v) => Some(v.clone()),
            _ => pos,
        };
        if let Some(m) = max_len.filter(py::truthy) {
            out.push(("max_len".into(), py::float_or_raise(Some(&m))));
        }
        if let Some(kvb) = g("kv_cache_memory_bytes").filter(|v| py::truthy(v)) {
            // vLLM ignores gpu_memory_utilization when this is set; the extra key lets the estimator use it.
            out.push((
                "kv_cache_fixed_gib".into(),
                py::float_or_raise(Some(kvb)) / GIB,
            ));
        }
        out
    }

    fn host_memory_gib(&self, _model: &ModelInfo, params: &Obj, _hw: &Hardware) -> FloatMap {
        let g = |k: &str| params.get(k);
        let one = Value::from(1);
        let gpus = py::int_or_raise(Some(py::or_value(g("tensor_parallel_size"), &one))).mul(
            &py::int_or_raise(Some(py::or_value(g("pipeline_parallel_size"), &one))),
        );
        let mut out: FloatMap = Vec::new();
        if py::truthy_opt(g("cpu_offload_gb")) {
            let per_gpu =
                py::float_or_raise(Some(py::or_value(g("cpu_offload_gb"), &Value::from(0))));
            out.push((
                "cpu_weight_offload".into(),
                Num::F(per_gpu).mul(&gpus).as_f64(),
            ));
        }
        if py::truthy_opt(g("kv_offloading_size")) {
            out.push((
                "kv_offload_buffer".into(),
                py::float_or_raise(Some(py::or_value(g("kv_offloading_size"), &Value::from(0)))),
            ));
        }
        out
    }

    fn parse_metrics(&self, prometheus_text: &str) -> FloatMap {
        let scrape = Scrape::new(prometheus_text);
        let Some(mut out) = scrape_to_canonical(&scrape) else {
            return Vec::new();
        };
        // Counters carry no rate in one scrape, so keep the previous sample per served-model-name set.
        let key = scrape.model_names();
        let now = (self.clock)();
        let get = |out: &FloatMap, k: &str| out.iter().find(|(x, _)| x == k).map(|(_, v)| *v);
        let (prompt, gen_) = (
            get(&out, "prompt_tokens_total"),
            get(&out, "generation_tokens_total"),
        );
        let mut prev_map = self
            .prev
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(&(t0, p0, g0)) = prev_map.get(&key) {
            let dt = now - t0;
            if dt >= 0.2 {
                if let (Some(p), Some(p0)) = (prompt, p0)
                    && p >= p0
                {
                    out.push(("prompt_tps".into(), (p - p0) / dt));
                }
                if let (Some(gn), Some(g0)) = (gen_, g0)
                    && gn >= g0
                {
                    out.push(("generation_tps".into(), (gn - g0) / dt));
                }
            }
        }
        if prompt.is_some() || gen_.is_some() {
            prev_map.insert(key, (now, prompt, gen_));
        }
        out
    }

    fn parse_startup_log(&self, line: &str) -> Option<Obj> {
        parse_log_line(line)
    }
}

pub fn adapter() -> Arc<dyn EngineAdapter> {
    Arc::new(VllmAdapter::new())
}
