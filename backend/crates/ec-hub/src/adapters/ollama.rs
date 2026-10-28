//! Python: backend/src/engine_console/adapters/ollama.py.
//!
//! Ollama adapter: one `ollama serve` that loads models from a store on demand (`multi_model`). The core mounts the
//! host's Ollama models directory read-only at `MODELS_DIR` and passes `ctx.model` = `name:tag` (or "" for a bare
//! server); only the "ollama" format is loadable. Always set (`RESERVED_ENV`): OLLAMA_HOST, OLLAMA_MODELS,
//! OLLAMA_NO_CLOUD=1, OLLAMA_NOPRUNE=1. Ollama has no Prometheus endpoint: `metrics_path` is `/api/version` (a cheap
//! always-200 call the node allows) and `parse_metrics` returns nothing. Fit model: the KV cache covers
//! `context_length x num_parallel` tokens per loaded model, on one GPU unless `sched_spread`.

use std::sync::{Arc, LazyLock};

use regex::Regex;
use serde_json::{Map, Value, json};

use super::base::{
    Compat, EngineAdapter, FloatMap, Hardware, LaunchContext, LaunchSpec, ModelInfo, Obj, ParamSpec,
};
use super::ollama_params::{
    CONTAINER_PORT, DEFAULT_IMAGE, MODELS_DIR, OMIT_WHEN_FALSE, build_catalog, kv_bytes,
};
use super::trtllm_parse::obj as pairs;
use super::trtllm_parse::py::{self, Int, Num, PyResult, exc, or_raise};
use crate::domain::errors::{Problem, unprocessable};

const GIB: f64 = 1_073_741_824.0;
/// Ollama's default batch size (recalled); only feeds the activation heuristic
const DEFAULT_BATCH: f64 = 512.0;

static NAME_RE: LazyLock<Regex> =
    LazyLock::new(|| py::pyre(r"^[A-Za-z0-9][A-Za-z0-9._:/+\-]{0,254}$"));
static DURATION_RE: LazyLock<Regex> =
    LazyLock::new(|| py::pyre(r"^-?(\d+(\.\d+)?(ns|us|µs|ms|s|m|h))+$"));
static SECONDS_RE: LazyLock<Regex> = LazyLock::new(|| py::pyre(r"^-?\d+$"));
static LIBRARY_RE: LazyLock<Regex> = LazyLock::new(|| py::pyre(r"^[a-z0-9_]+$"));
static CTRL_RE: LazyLock<Regex> = LazyLock::new(|| py::pyre(r"[\x00-\x1f\x7f]"));
static SPACE_COMMA_RE: LazyLock<Regex> = LazyLock::new(|| py::pyre(r"[\s,]"));

/// `default_context`: context tokens Ollama picks when none is set, by total GPU memory (below 24 GiB 4k, from 24 GiB
/// 32k, from 48 GiB 256k; docs.ollama.com/context-length).
pub fn default_context(total_gib: f64) -> i64 {
    if total_gib >= 48.0 {
        262_144
    } else if total_gib >= 24.0 {
        32_768
    } else {
        4096
    }
}

/// `_num(params, key, default)`.
fn num(params: &Obj, key: &str, default: f64) -> f64 {
    params.get(key).and_then(Num::of).map_or(default, Num::f)
}

/// `int(x)` of a finite float, exact.
fn int(x: f64) -> Int {
    Int::from_f64(x).unwrap_or_else(Int::zero)
}

/// `max(1, int(x))`.
fn at_least_one(x: f64) -> Int {
    let v = int(x);
    if v.cmp_i128(1) == std::cmp::Ordering::Less {
        Int::from_i128(1)
    } else {
        v
    }
}

/// `float(i)` (OverflowError past 1.8e308, as in Python).
fn fl(i: &Int) -> PyResult<f64> {
    i.to_f64()
        .ok_or_else(|| exc("OverflowError", "int too large to convert to float"))
}

pub struct OllamaAdapter {
    catalog: Vec<ParamSpec>,
}

impl Default for OllamaAdapter {
    fn default() -> Self {
        Self::new()
    }
}

fn obj(v: Value) -> Obj {
    match v {
        Value::Object(m) => m,
        _ => Obj::new(),
    }
}

impl OllamaAdapter {
    pub fn new() -> Self {
        Self {
            catalog: build_catalog(),
        }
    }

    fn by_key(&self, key: &str) -> Option<&ParamSpec> {
        self.catalog.iter().find(|p| p.key == key)
    }

    fn check_value(spec: &ParamSpec, val: &Value) -> Vec<String> {
        let k = spec.key.as_str();
        match spec.type_.as_str() {
            "bool" => {
                return if val.is_boolean() {
                    vec![]
                } else {
                    vec![format!("{k}: expected true/false")]
                };
            }
            "int" => {
                let Some(n) = Num::of(val).filter(|n| n.is_finite() && n.is_integral()) else {
                    return vec![format!("{k}: expected an integer")];
                };
                if let Some(min) = spec.min
                    && n.lt(min)
                {
                    return vec![format!("{k}: must be >= {}", ec_proto::pyfmt::fmt_g(min))];
                }
                return vec![];
            }
            "enum" => {
                let choices = spec.choices.clone().unwrap_or_default();
                let s = py::str_of(val);
                return if choices.contains(&s) {
                    vec![]
                } else {
                    vec![format!("{k}: '{s}' is not one of {}", choices.join(", "))]
                };
            }
            "string_list" => {
                let ok = val.as_array().is_some_and(|a| {
                    !a.is_empty()
                        && a.iter().all(|x| {
                            x.as_str().is_some_and(|s| {
                                !s.is_empty() && !SPACE_COMMA_RE.is_match(s) && !CTRL_RE.is_match(s)
                            })
                        })
                });
                return if ok {
                    vec![]
                } else {
                    vec![format!(
                        "{k}: expected a non-empty list of origins without spaces or commas"
                    )]
                };
            }
            _ => {}
        }
        // strings (and numbers for the duration params, which accept a number of seconds)
        if k == "keep_alive" || k == "load_timeout" {
            let text = match val {
                Value::String(s) => py::strip(s).to_string(),
                Value::Number(n) if !n.is_f64() => py::str_of(val),
                _ => String::new(),
            };
            let mut ok = SECONDS_RE.is_match(&text) || DURATION_RE.is_match(&text);
            if k == "load_timeout" && text.starts_with('-') {
                ok = false;
            }
            return if ok {
                vec![]
            } else {
                vec![format!(
                    "{k}: expected a duration such as 5m or 24h, or a number of seconds"
                )]
            };
        }
        let Some(s) = val
            .as_str()
            .filter(|s| !py::strip(s).is_empty() && !CTRL_RE.is_match(s))
        else {
            return vec![format!("{k}: expected a non-empty string")];
        };
        if k == "llm_library" && !LIBRARY_RE.is_match(s) {
            return vec![format!("{k}: expected a library name such as cuda_v13")];
        }
        vec![]
    }

    /// `_render`: the environment value of one parameter.
    fn render(spec: &ParamSpec, val: &Value) -> String {
        match spec.type_.as_str() {
            "bool" => if py::truthy(Some(val)) { "1" } else { "0" }.into(),
            "string_list" => val
                .as_array()
                .map(|a| a.iter().map(py::str_of).collect::<Vec<_>>().join(","))
                .unwrap_or_default(),
            _ => py::strip(&py::str_of(val)).to_string(),
        }
    }

    /// `build_launch_ctx`: ValueError for what the user must fix (the lifecycle's 422 `invalid_params`).
    pub fn py_build_launch_ctx(&self, ctx: &LaunchContext) -> PyResult<LaunchSpec> {
        self.build(ctx).map_err(|m| exc("ValueError", m))
    }

    fn build(&self, ctx: &LaunchContext) -> Result<LaunchSpec, String> {
        if ctx.model_format != "ollama" {
            return Err(format!(
                "Ollama loads models from its own store (format 'ollama'); importing a GGUF file or a Hugging Face repo ({}) into a managed Ollama is not supported yet",
                ctx.model_format
            ));
        }
        if !ctx.model.is_empty() && !NAME_RE.is_match(&ctx.model) {
            return Err("model must be an Ollama name such as llama3.1:8b".into());
        }
        let errs = self.validate(&ctx.params);
        if !errs.is_empty() {
            return Err(errs.join("; "));
        }
        let mut env = Map::new();
        env.insert(
            "OLLAMA_HOST".into(),
            format!("0.0.0.0:{CONTAINER_PORT}").into(),
        );
        env.insert("OLLAMA_MODELS".into(), MODELS_DIR.into());
        env.insert("OLLAMA_NO_CLOUD".into(), "1".into());
        env.insert("OLLAMA_NOPRUNE".into(), "1".into());
        for spec in &self.catalog {
            let Some(val) = py::get(&ctx.params, &spec.key) else {
                continue;
            };
            if py::is_bool(Some(val), false) && OMIT_WHEN_FALSE.contains(&spec.key.as_str()) {
                continue;
            }
            let name = spec.flag.strip_prefix("env:").unwrap_or(&spec.flag);
            env.insert(name.into(), Self::render(spec, val).into());
        }
        Ok(LaunchSpec {
            image: DEFAULT_IMAGE.into(),
            argv: vec!["serve".into()],
            env,
            container_port: CONTAINER_PORT,
            health_path: "/api/version".into(),
            metrics_path: "/api/version".into(),
            models_path: "/v1/models".into(),
            shm_size: "1g".into(),
            ipc_host: false,
            files: Map::new(),
            needs_hf_cache: false,
            ulimits: Map::new(),
        })
    }

    /// `preload_request`: (path, JSON body) of a POST that loads `model` without generating text. Optional helper for
    /// the core (not part of the locked port).
    pub fn preload_request(&self, model: &str, _params: &Obj) -> Result<(String, Value), String> {
        if !NAME_RE.is_match(model) {
            return Err("model must be an Ollama name such as llama3.1:8b".into());
        }
        Ok((
            "/api/generate".into(),
            json!({"model": model, "stream": false}),
        ))
    }
}

pub fn adapter() -> Arc<dyn EngineAdapter> {
    Arc::new(OllamaAdapter::new())
}

impl EngineAdapter for OllamaAdapter {
    fn id(&self) -> &str {
        "ollama"
    }
    fn display_name(&self) -> &str {
        "Ollama"
    }
    fn default_image(&self) -> &str {
        DEFAULT_IMAGE
    }
    fn default_port(&self) -> i64 {
        CONTAINER_PORT
    }
    fn model_formats(&self) -> Vec<String> {
        vec!["ollama".into()]
    }
    fn multi_model(&self) -> bool {
        true
    }
    fn metric_prefixes(&self) -> Option<Vec<String>> {
        Some(Vec::new())
    }

    fn param_catalog(&self) -> Vec<ParamSpec> {
        self.catalog.clone()
    }

    fn presets(&self) -> Vec<(String, Obj)> {
        vec![
            (
                "balanced".into(),
                obj(json!({"context_length": 8192, "keep_alive": "30m"})),
            ),
            (
                "max-throughput".into(),
                obj(json!({"num_parallel": 4, "context_length": 8192, "flash_attention": true})),
            ),
            (
                "low-latency".into(),
                obj(
                    json!({"num_parallel": 1, "context_length": 8192, "keep_alive": "-1", "flash_attention": true}),
                ),
            ),
            (
                "long-context".into(),
                obj(
                    json!({"context_length": 131072, "num_parallel": 1, "flash_attention": true,
                           "kv_cache_type": "q8_0"}),
                ),
            ),
            (
                "low-memory".into(),
                obj(
                    json!({"context_length": 4096, "kv_cache_type": "q8_0", "flash_attention": true,
                           "keep_alive": "1m", "max_loaded_models": 1}),
                ),
            ),
            (
                "many-models".into(),
                obj(json!({"max_loaded_models": 3, "keep_alive": "10m", "context_length": 8192})),
            ),
            (
                "big-model-spread".into(),
                obj(json!({"sched_spread": true, "context_length": 16384})),
            ),
        ]
    }

    fn validate(&self, params: &Obj) -> Vec<String> {
        or_raise(self.try_validate(params))
    }

    /// A ValueError is the user's to fix: 422 `invalid_params` with its text, as the Python lifecycle answers.
    fn build_launch_ctx(&self, ctx: &LaunchContext) -> Result<LaunchSpec, Problem> {
        or_raise(self.try_build_launch_ctx(ctx))
    }

    /// Python's `build_launch` ([`Self::py_build_launch_ctx`]): a ValueError is the inner refusal (422), any other
    /// exception the outer `Err` (PORTING.md §A5).
    fn try_build_launch_ctx(&self, ctx: &LaunchContext) -> PyResult<Result<LaunchSpec, Problem>> {
        match self.py_build_launch_ctx(ctx) {
            Ok(spec) => Ok(Ok(spec)),
            Err(e) if e.kind == "ValueError" => Ok(Err(unprocessable(e.msg))),
            Err(e) => Err(e),
        }
    }

    /// v1 bridge: `model` is an Ollama name:tag (or empty for a bare server).
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
            model_format: "ollama".into(),
            model_path: None,
            served_name: served_name.into(),
            hf_cache_container_path: hf_cache_container_path.into(),
            params: params.clone(),
            hw: hw.clone(),
        })
    }

    fn compatibility(&self, model: &ModelInfo, params: &Obj, hw: &Hardware) -> Vec<Compat> {
        or_raise(self.try_compatibility(model, params, hw))
    }

    fn memory_model(&self, model: &ModelInfo, params: &Obj) -> FloatMap {
        or_raise(self.try_memory_model(model, params))
    }

    // host_memory_gib: the default (empty) is right; the CPU share of a partly offloaded model is chosen by Ollama's
    // scheduler at load time and cannot be derived from the parameters.

    fn parse_metrics(&self, prometheus_text: &str) -> FloatMap {
        or_raise(self.try_parse_metrics(prometheus_text))
    }

    fn parse_startup_log(&self, line: &str) -> Option<Obj> {
        or_raise(self.try_parse_startup_log(line))
    }

    // ── the fallible forms, native (PORTING.md §A5) ────────────────────────────────────────────────────────────
    /// `validate` (never raises).
    fn try_validate(&self, params: &Obj) -> PyResult<Vec<String>> {
        let mut errs = Vec::new();
        for (key, val) in params {
            let Some(spec) = self.by_key(key) else {
                errs.push(format!("unknown parameter '{key}'"));
                continue;
            };
            if val.is_null() {
                continue;
            }
            errs.extend(Self::check_value(spec, val));
        }
        Ok(errs)
    }

    #[allow(clippy::cast_precision_loss)]
    fn try_compatibility(
        &self,
        model: &ModelInfo,
        params: &Obj,
        hw: &Hardware,
    ) -> PyResult<Vec<Compat>> {
        let mut out: Vec<Compat> = Vec::new();
        let mut add = |level: &str, code: &str, message: String| {
            out.push(Compat {
                level: level.into(),
                code: code.into(),
                message,
            });
        };
        let g = |k: &str| py::get(params, k);
        let ctx = int(num(params, "context_length", 0.0));
        let par = at_least_one(num(params, "num_parallel", 1.0));
        let total: f64 = hw.gpu_total_gib.iter().fold(0.0, |a, b| a + b);
        let free: f64 = hw.gpu_free_gib.iter().fold(0.0, |a, b| a + b);
        let kv_name = if py::truthy(g("kv_cache_type")) {
            py::str_of(g("kv_cache_type").unwrap_or(&Value::Null))
        } else {
            "f16".into()
        };
        let kv_tok = match (model.num_layers, model.num_kv_heads, model.head_dim) {
            (Some(l), Some(h), Some(d)) if l != 0 && h != 0 && d != 0 => {
                Some(2.0 * l as f64 * h as f64 * d as f64 * kv_bytes(&kv_name).unwrap_or(2.0))
            }
            _ => None,
        };
        if hw.compute_capability == "12.0" {
            add(
                "ok",
                "blackwell_cuda_build",
                "This GPU is Blackwell (compute capability 12.0): the pinned Ollama image ships CUDA 12 and 13 runners and the CUDA 12 runner carries sm_120a kernels, so no extra settings are needed.".into(),
            );
        }
        let has_gpus = !hw.gpu_total_gib.is_empty() && total != 0.0;
        if ctx.is_zero() && has_gpus {
            let chosen = default_context(total);
            if chosen >= 32_768 {
                let cost = match kv_tok {
                    Some(k) => format!(
                        ", about {:.1} GiB of KV cache per loaded model",
                        fl(&Int::from_i128(i128::from(chosen)).times(&par))? * k / GIB
                    ),
                    None => String::new(),
                };
                add(
                    "warn",
                    "default_context_by_vram",
                    format!(
                        "No context_length is set, so Ollama picks {chosen} tokens for {total:.0} GiB of GPU memory{cost}. Set context_length to what you need (for example 32768) to keep memory for the model itself."
                    ),
                );
            }
        } else if let Some(k) = kv_tok
            && !ctx.is_zero()
            && free != 0.0
            && !hw.gpu_free_gib.is_empty()
            && fl(&ctx.times(&par))? * k / GIB > 0.9 * free
        {
            add(
                "warn",
                "kv_exceeds_vram",
                format!(
                    "A context of {ctx} tokens x {par} request(s) needs about {:.1} GiB of KV cache and only {free:.1} GiB of GPU memory is free. Lower context_length or num_parallel, or set kv_cache_type to q8_0.",
                    fl(&ctx.times(&par))? * k / GIB
                ),
            );
        }
        if par.cmp_i128(1) == std::cmp::Ordering::Greater {
            add(
                "ok",
                "parallel_multiplies_context",
                format!(
                    "num_parallel={par}: Ollama multiplies the context by the number of parallel requests, so every loaded model holds that many tokens of KV cache."
                ),
            );
        }
        if let Some(t) = g("kv_cache_type")
            .and_then(Value::as_str)
            .filter(|t| *t == "q8_0" || *t == "q4_0")
            && py::is_bool(g("flash_attention"), false)
        {
            add(
                "warn",
                "kv_quant_needs_flash_attn",
                format!(
                    "kv_cache_type={t} only takes effect with flash attention, and flash_attention is off. Turn flash_attention on or leave it unset."
                ),
            );
        }
        let ka =
            g("keep_alive").map_or_else(String::new, |v| py::strip(&py::str_of(v)).to_string());
        if ka.starts_with('-') {
            add(
                "warn",
                "keep_alive_forever",
                "A negative keep_alive keeps the model in memory until the server stops, so that GPU memory is not available to other engines. Use a duration such as 30m unless the model must always be hot.".into(),
            );
        }
        let lib = if py::truthy(g("llm_library")) {
            py::str_of(g("llm_library").unwrap_or(&Value::Null))
        } else {
            String::new()
        };
        if lib.starts_with("cpu") {
            add(
                "warn",
                "cpu_offload_slow",
                "llm_library selects the CPU: models run without the GPU, which is far slower."
                    .into(),
            );
        }
        Ok(out)
    }

    #[allow(clippy::cast_precision_loss)]
    fn try_memory_model(&self, _model: &ModelInfo, params: &Obj) -> PyResult<FloatMap> {
        let g = |k: &str| params.get(k);
        let par = at_least_one(num(params, "num_parallel", 1.0));
        let ctx = int(num(params, "context_length", 0.0));
        let kv_name = if py::truthy(g("kv_cache_type")) {
            py::str_of(g("kv_cache_type").unwrap_or(&Value::Null))
        } else {
            "f16".into()
        };
        // Ollama places layers by free memory; there is no share to configure
        let mut out: FloatMap = vec![
            ("mem_fraction".into(), 1.0),
            (
                "kv_bytes_per_elem".into(),
                kv_bytes(&kv_name).unwrap_or(2.0),
            ),
            (
                "overhead_gib".into(),
                1.0 + num(params, "gpu_overhead", 0.0) / GIB,
            ),
            ("cuda_graph_gib".into(), 0.0),
            ("max_batched_tokens".into(), DEFAULT_BATCH),
            ("max_seqs".into(), fl(&par)?),
        ];
        if ctx.cmp_i128(0) == std::cmp::Ordering::Greater {
            // the KV cache covers every parallel slot (Ollama FAQ)
            out.push(("max_len".into(), fl(&ctx.times(&par))?));
        }
        if !py::truthy(g("sched_spread")) {
            // one GPU unless the model needs more or spreading is forced
            out.push(("tp".into(), 1.0));
        }
        Ok(out)
    }

    /// `host_memory_gib`: the base class's `{}`.
    fn try_host_memory_gib(
        &self,
        _model: &ModelInfo,
        _params: &Obj,
        _hw: &Hardware,
    ) -> PyResult<FloatMap> {
        Ok(FloatMap::new())
    }

    /// Ollama exports no Prometheus metrics.
    fn try_parse_metrics(&self, _prometheus_text: &str) -> PyResult<FloatMap> {
        Ok(FloatMap::new())
    }

    fn try_parse_startup_log(&self, line: &str) -> PyResult<Option<Obj>> {
        parse_log_line(line)
    }
}

// ------------------------------------------------------------------ log lines
macro_rules! rx {
    ($name:ident, $p:expr) => {
        static $name: LazyLock<Regex> = LazyLock::new(|| py::pyre($p));
    };
}
rx!(MSG, r#"\bmsg=(?:"((?:[^"\\]|\\.)*)"|(\S+))"#);
rx!(LEVEL, r"\blevel=(\w+)");
rx!(LISTENING, r"Listening on (\S+) \(version ([^)]+)\)");
rx!(DEFAULT_CTX, r"default_num_ctx=(\d+)");
rx!(OFFLOADED, r"offloaded (\d+)/(\d+) layers to GPU");
rx!(STARTED, r"runner started in ([\d.]+) seconds");
rx!(SIZE, r"^([\d.]+)\s*(B|KiB|MiB|GiB|TiB)$");
static FIELDS: LazyLock<Vec<(&'static str, Regex)>> = LazyLock::new(|| {
    ["id", "library", "total", "available", "total_vram"]
        .into_iter()
        .map(|n| (n, py::pyre(&format!(r#"\b{n}=(?:"([^"]*)"|(\S+))"#))))
        .collect()
});
const ERRORS: [(&str, &str); 4] = [
    ("read-only file system", "models_dir_read_only"),
    ("address already in use", "port_in_use"),
    ("cudaMalloc failed", "cuda_oom"),
    ("out of memory", "cuda_oom"),
];

fn field(line: &str, name: &str) -> Option<String> {
    let re = &FIELDS.iter().find(|(n, _)| *n == name)?.1;
    let m = re.captures(line)?;
    m.get(1)
        .or_else(|| m.get(2))
        .map(|x| x.as_str().to_string())
}

/// `_gib`: a size such as "95.0 GiB" in GiB, rounded to 2 places. Err where Python's float() raises (the whole line
/// then yields nothing, D-A2-3).
fn gib(text: Option<&str>) -> PyResult<Option<f64>> {
    let Some(t) = text.filter(|t| !t.is_empty()) else {
        return Ok(None);
    };
    let Some(m) = SIZE.captures(py::strip(t)) else {
        return Ok(None);
    };
    let v = py::float_of_value(&Value::from(m.get(1).map_or("", |x| x.as_str())))?;
    let unit = match m.get(2).map_or("", |x| x.as_str()) {
        "B" => 1.0 / GIB,
        "KiB" => 1.0 / 1_048_576.0,
        "MiB" => 1.0 / 1024.0,
        "GiB" => 1.0,
        _ => 1024.0,
    };
    Ok(Some(py::round_to(v * unit, 2)))
}

fn opt(v: Option<f64>) -> Value {
    v.map_or(Value::Null, Value::from)
}

fn opt_s(v: Option<String>) -> Value {
    v.map_or(Value::Null, Value::from)
}

/// `parse_log_line`: structured progress for one Ollama log line (`time=... level=... source=... msg="..."`), or
/// None.
pub fn parse_log_line(line: &str) -> PyResult<Option<Map<String, Value>>> {
    let stripped = py::strip(line);
    if stripped.is_empty() {
        return Ok(None);
    }
    let m = MSG.captures(stripped);
    let text: String = match &m {
        Some(c) => c
            .get(1)
            .or_else(|| c.get(2))
            .map_or_else(String::new, |x| x.as_str().to_string()),
        None => stripped.to_string(),
    };
    let message = || Value::from(py::tail(stripped, 300));
    for (needle, code) in ERRORS {
        if stripped.contains(needle) {
            return Ok(Some(pairs(vec![
                ("error", code.into()),
                ("message", message()),
            ])));
        }
    }
    if let Some(lm) = LISTENING.captures(&text) {
        return Ok(Some(pairs(vec![
            ("phase", "ready".into()),
            ("pct", 100.0.into()),
            ("listening", lm.get(1).map_or("", |x| x.as_str()).into()),
            ("version", lm.get(2).map_or("", |x| x.as_str()).into()),
        ])));
    }
    if text == "inference compute" {
        let library = field(stripped, "library");
        let gpu = pairs(vec![
            ("id", opt_s(field(stripped, "id"))),
            ("library", opt_s(library.clone())),
            ("total_gib", opt(gib(field(stripped, "total").as_deref())?)),
            (
                "available_gib",
                opt(gib(field(stripped, "available").as_deref())?),
            ),
        ]);
        let mut out = pairs(vec![("gpu", Value::Object(gpu))]);
        if library.as_deref() == Some("cpu") {
            // no GPU was found: models will run on the CPU
            out.insert("warning".into(), "cpu_only".into());
        }
        return Ok(Some(out));
    }
    if text == "vram-based default context"
        && let Some(dm) = DEFAULT_CTX.captures(stripped)
    {
        return Ok(Some(pairs(vec![
            (
                "default_num_ctx",
                py::int_of(dm.get(1).map_or("", |x| x.as_str()))?,
            ),
            (
                "total_vram_gib",
                opt(gib(field(stripped, "total_vram").as_deref())?),
            ),
        ])));
    }
    if let Some(om) = OFFLOADED.captures(&text) {
        return Ok(Some(pairs(vec![
            ("phase", "loading_weights".into()),
            (
                "gpu_layers",
                py::int_of(om.get(1).map_or("", |x| x.as_str()))?,
            ),
            (
                "total_layers",
                py::int_of(om.get(2).map_or("", |x| x.as_str()))?,
            ),
        ])));
    }
    if let Some(sm) = STARTED.captures(&text) {
        let secs = py::float_of_value(&Value::from(sm.get(1).map_or("", |x| x.as_str())))?;
        return Ok(Some(pairs(vec![("model_load_seconds", secs.into())])));
    }
    if text.contains("Ollama cloud disabled") {
        return Ok(Some(pairs(vec![(
            "cloud_disabled",
            text.trim_end_matches(py::isspace).ends_with("true").into(),
        )])));
    }
    if LEVEL
        .captures(stripped)
        .is_some_and(|l| l.get(1).map(|x| x.as_str()) == Some("ERROR"))
        || stripped.starts_with("Error:")
    {
        return Ok(Some(pairs(vec![
            ("error", "engine_error".into()),
            ("message", message()),
        ])));
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_context_tiers() {
        assert_eq!(default_context(23.9), 4096);
        assert_eq!(default_context(24.0), 32_768);
        assert_eq!(default_context(48.0), 262_144);
    }

    #[test]
    fn log_lines() {
        let got = parse_log_line(
            r#"time=x level=INFO msg="vram-based default context" total_vram="512 MiB" default_num_ctx=4096"#,
        );
        assert_eq!(
            got.map(|o| o.map(Value::Object)),
            Ok(Some(
                json!({"default_num_ctx": 4096, "total_vram_gib": 0.5})
            ))
        );
    }
}
