//! Python: backend/src/engine_console/adapters/llamacpp.py.
//!
//! llama.cpp adapter: `llama-server` serving ONE GGUF model. The image entrypoint is `/app/llama-server`, so `argv`
//! starts with `-m`. Always on: `--host 0.0.0.0 --port 8080 --alias <served>[,extras] --metrics --offline`, and
//! `--cors-origins localhost` unless the user sets `cors_origins` (the engine default reflects ANY origin, measured).
//! The API key travels as `LLAMA_API_KEY` in the environment, never on argv; a custom chat template goes through
//! `LaunchSpec.files`. Fit model: `max_len` is the total context; partial offload is reported through
//! `weights_gpu_fraction` / `kv_gpu_fraction`; host RAM through `host_memory_gib` (NaN = needed but unknowable).

use std::sync::{Arc, LazyLock};

use regex::Regex;
use serde_json::{Map, Value, json};

use super::base::{
    Compat, EngineAdapter, FloatMap, Hardware, LaunchContext, LaunchSpec, MODEL_MOUNT, ModelInfo,
    Obj, ParamSpec,
};
use super::llamacpp_params::{
    CHAT_TEMPLATE_PATH, CHAT_TEMPLATES, CONTAINER_PORT, DEFAULT_IMAGE, LAYER_KEYS,
    MAX_TEMPLATE_BYTES, NEGATABLE_KEYS, PATH_KEYS, build_catalog, cache_type_bytes,
};
use super::llamacpp_parse::{parse_log_line, scrape_to_canonical};
use super::trtllm_parse::py::{self, Exact, Int, Num, PyResult, PyVal, exc, or_raise};
use crate::domain::errors::{Problem, unprocessable};

const GIB: f64 = 1_073_741_824.0;
const NO_QUANT_V: [&str; 3] = ["f16", "bf16", "f32"];
const DRAFT_MODEL_TYPES: [&str; 3] = ["draft-simple", "draft-eagle3", "draft-dflash"];
const EXPERT_COUNT_KEYS: [&str; 4] = [
    "expert_count",
    "num_experts",
    "n_routed_experts",
    "num_local_experts",
];
const EXPERT_USED_KEYS: [&str; 3] = [
    "expert_used_count",
    "num_experts_per_tok",
    "n_experts_per_tok",
];
const MIN_SLOT_TOKENS: i128 = 2048;
/// the pinned build's default --fit-target
const FIT_MARGIN_MIB: f64 = 1024.0;
/// the pinned build's default --cache-ram
const CACHE_RAM_MIB: i128 = 8192;

static NAME_RE: LazyLock<Regex> = LazyLock::new(|| py::pyre(r"^[A-Za-z0-9][A-Za-z0-9._:/+\-]*$"));
static TENSOR_SPLIT_RE: LazyLock<Regex> =
    LazyLock::new(|| py::pyre(r"^\d+(\.\d+)?(,\d+(\.\d+)?)*$"));
static DEVICE_RE: LazyLock<Regex> =
    LazyLock::new(|| py::pyre(r"^(none|[A-Za-z0-9_.\-]+(,[A-Za-z0-9_.\-]+)*)$"));
static ORIGINS_RE: LazyLock<Regex> = LazyLock::new(|| py::pyre(r"^[A-Za-z0-9*:/.,_\-\[\]]+$"));
static CTRL_RE: LazyLock<Regex> = LazyLock::new(|| py::pyre(r"[\x00-\x1f\x7f]"));
static SPACE_RE: LazyLock<Regex> = LazyLock::new(|| py::pyre(r"\s"));
static LORA_SCALED_RE: LazyLock<Regex> = LazyLock::new(|| py::pyre(r"^.+:-?\d+(\.\d+)?$"));

/// `_model_path_error`: a .gguf file under the model mount (no '..', no control characters, no leading '-').
fn model_path_error(path: Option<&str>, what: &str) -> Option<String> {
    let path = path.filter(|p| !p.is_empty())?;
    if CTRL_RE.is_match(path)
        || !path.starts_with(&format!("{MODEL_MOUNT}/"))
        || path.starts_with('-')
    {
        return Some(format!("{what} must be a file inside {MODEL_MOUNT}/"));
    }
    if path.split('/').any(|p| p == "..") {
        return Some(format!("{what} must not contain '..'"));
    }
    if !path.to_lowercase().ends_with(".gguf") {
        return Some(format!("{what} must be a .gguf file"));
    }
    None
}

fn path_error(path: Option<&str>, what: &str) -> Option<String> {
    if path.is_none_or(str::is_empty) {
        return Some(format!("{what} is required"));
    }
    model_path_error(path, what)
}

/// `_num(params, key, default)`: a numeric param, or `default` when it is unset (bools are not numbers here).
fn num(params: &Obj, key: &str, default: f64) -> f64 {
    params.get(key).and_then(Num::of).map_or(default, Num::f)
}

/// `_layer_count`: an explicit layer count, or None for unset / 'auto' / 'all' (`int()` raises ValueError on a
/// non-decimal digit that passed `isdigit()`, such as '²').
fn layer_count(v: Option<&Value>) -> PyResult<Option<Int>> {
    Ok(match v {
        Some(n @ Value::Number(x)) if !x.is_f64() => Num::of(n).and_then(Int::of),
        Some(Value::String(s)) if py::isdigit(s) => Int::parse_digits(&py::int_digits(s)?),
        _ => None,
    })
}

/// `int(v or 0)`.
fn int_or_zero(v: Option<&Value>) -> PyResult<Int> {
    match v {
        Some(x) if py::truthy(Some(x)) => py::int_of_value(x),
        _ => Ok(Int::zero()),
    }
}

/// Python's `v < k` / `v > k` of a JSON value and an int: the ordering, or TypeError for a non-number.
fn cmp_value(v: &Value, k: i128) -> PyResult<std::cmp::Ordering> {
    match v {
        Value::Bool(b) => Ok(i128::from(*b).cmp(&k)),
        Value::Number(_) => Num::of(v)
            .and_then(|n| Exact::of(n).cmp_i128(k))
            .ok_or_else(|| exc("TypeError", "comparison with NaN")),
        other => Err(exc(
            "TypeError",
            format!(
                "'<' not supported between instances of '{}' and 'int'",
                py::type_name(other)
            ),
        )),
    }
}

/// `ctx // par` of an int and a JSON value.
fn floordiv_value(ctx: &Int, par: &Value) -> PyResult<Exact> {
    let zero = || exc("ZeroDivisionError", "integer division or modulo by zero");
    match par {
        Value::Bool(b) => ctx.floordiv(i128::from(*b)).map(Exact::I).ok_or_else(zero),
        Value::Number(_) => {
            let n = Num::of(par).unwrap_or(Num::Int(0));
            Exact::I(ctx.clone()).floordiv(n).ok_or_else(|| match n {
                Num::Float(_) => exc("ZeroDivisionError", "float floor division by zero"),
                Num::Int(_) => zero(),
            })
        }
        other => Err(exc(
            "TypeError",
            format!(
                "unsupported operand type(s) for //: 'int' and '{}'",
                py::type_name(other)
            ),
        )),
    }
}

/// `float(x)` of each comma-separated piece of `str(value)`.
fn split_floats(v: &Value) -> PyResult<Vec<f64>> {
    py::str_of(v)
        .split(',')
        .map(|x| py::float_of_value(&Value::from(x)))
        .collect()
}

/// `float(i)` (OverflowError past 1.8e308 in Python).
fn fl(i: &Int) -> f64 {
    i.to_f64().unwrap_or(f64::INFINITY)
}

/// `_kv_bytes_per_elem`: mean bytes per KV element over the K and V halves.
fn kv_bytes_per_elem(params: &Obj) -> f64 {
    let half = |key: &str| {
        let name = if py::truthy(params.get(key)) {
            py::str_of(params.get(key).unwrap_or(&Value::Null))
        } else {
            "f16".into()
        };
        cache_type_bytes(&name).unwrap_or(2.0)
    };
    (half("cache_type_k") + half("cache_type_v")) / 2.0
}

/// `_kv_bytes_per_token`: K and V of every layer for one token; None when the model facts are missing.
#[allow(clippy::cast_precision_loss)]
fn kv_bytes_per_token(model: &ModelInfo, params: &Obj) -> Option<f64> {
    let l = model.num_layers.filter(|v| *v != 0)?;
    let h = model.num_kv_heads.filter(|v| *v != 0)?;
    let d = model.head_dim.filter(|v| *v != 0)?;
    Some(2.0 * l as f64 * h as f64 * d as f64 * kv_bytes_per_elem(params))
}

/// `_offload`: (share of the weights on the GPU, share of the KV cache on the GPU); the output layer counts as one
/// extra offloadable layer.
#[allow(clippy::cast_precision_loss)]
fn offload(params: &Obj, model: &ModelInfo) -> PyResult<(f64, f64)> {
    if py::is_str(params.get("device"), "none") {
        return Ok((0.0, 0.0));
    }
    let (mut weights, mut kv) = (1.0, 1.0);
    if let Some(n) = layer_count(params.get("n_gpu_layers"))? {
        if n.is_zero() {
            weights = 0.0;
            kv = 0.0;
        } else if let Some(layers) = model.num_layers.filter(|l| *l != 0) {
            let layers = i128::from(layers);
            // min(n, k) / k: int / int true division
            let share = |k: i128| match n.to_i128() {
                Some(v) if v < k => Int::from_i128(v).truediv(k).unwrap_or(1.0),
                _ => 1.0,
            };
            weights = share(layers + 1);
            kv = share(layers);
        }
    }
    if py::is_bool(params.get("kv_offload"), false) {
        kv = 0.0;
    }
    Ok((weights, kv))
}

fn raw_int(model: &ModelInfo, names: &[&str]) -> Option<i64> {
    model.raw_config.iter().find_map(|(key, val)| {
        let base = key.rsplit('.').next().unwrap_or(key);
        (names.contains(&base))
            .then(|| val.as_i64().filter(|v| *v > 0 && !val.is_f64()))
            .flatten()
    })
}

/// `_expert_share`: share of the weight bytes in expert tensors; None when any fact is unknown (never guessed).
#[allow(clippy::cast_precision_loss)]
fn expert_share(model: &ModelInfo) -> Option<f64> {
    let total = model.num_params.filter(|v| *v != 0)?;
    let active = model.num_active_params.filter(|v| *v != 0)?;
    let experts = raw_int(model, &EXPERT_COUNT_KEYS)?;
    let used = raw_int(model, &EXPERT_USED_KEYS)?;
    if used >= experts || active >= total {
        return None;
    }
    let share = (total - active) as f64 / (total as f64 * (1.0 - used as f64 / experts as f64));
    let share = if 0.0 > share { 0.0 } else { share };
    Some(if 1.0 < share { 1.0 } else { share })
}

/// `_moe_cpu_layers`: fraction of the layers whose expert weights are forced to the CPU: 0.0 when none, NaN when
/// unknowable.
#[allow(clippy::cast_precision_loss)]
fn moe_cpu_layers(params: &Obj, model: &ModelInfo) -> PyResult<f64> {
    if py::truthy(params.get("cpu_moe")) {
        return Ok(1.0);
    }
    let n = params.get("n_cpu_moe");
    if py::truthy(n) {
        // min(int(n) / num_layers, 1.0) if num_layers else nan: the condition is evaluated first
        return Ok(match model.num_layers.filter(|l| *l != 0) {
            Some(l) => {
                let x = int_or_zero(n)?
                    .truediv(i128::from(l))
                    .unwrap_or(f64::INFINITY);
                if 1.0 < x { 1.0 } else { x }
            }
            None => f64::NAN,
        });
    }
    let ot = if py::truthy(params.get("override_tensor")) {
        py::str_of(params.get("override_tensor").unwrap_or(&Value::Null))
    } else {
        String::new()
    };
    if ot.to_uppercase().contains("=CPU") {
        return Ok(if ot.contains("exps") { 1.0 } else { f64::NAN });
    }
    Ok(0.0)
}

pub struct LlamaCppAdapter {
    catalog: Vec<ParamSpec>,
}

impl Default for LlamaCppAdapter {
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

impl LlamaCppAdapter {
    pub fn new() -> Self {
        Self {
            catalog: build_catalog(),
        }
    }

    fn by_key(&self, key: &str) -> Option<&ParamSpec> {
        self.catalog.iter().find(|p| p.key == key)
    }

    fn check_value(spec: &ParamSpec, val: &Value) -> PyResult<Vec<String>> {
        let k = spec.key.as_str();
        if spec.type_ == "json" && !LAYER_KEYS.contains(&k) {
            let obj = match val {
                Value::String(s) => match py::json_loads(s) {
                    Ok(o) => o,
                    Err(e) if e.kind == "JSONDecodeError" => {
                        return Ok(vec![format!("{k}: invalid JSON ({})", e.msg)]);
                    }
                    Err(e) => return Err(e),
                },
                other => PyVal::from_json(other),
            };
            return Ok(if matches!(obj, PyVal::Dict(_)) {
                vec![]
            } else {
                vec![format!("{k}: expected a JSON object")]
            });
        }
        Ok(Self::check_plain(spec, val))
    }

    fn check_plain(spec: &ParamSpec, val: &Value) -> Vec<String> {
        let k = spec.key.as_str();
        if LAYER_KEYS.contains(&k) {
            let ok = match val {
                Value::Number(n) if !n.is_f64() => Num::of(val).is_some_and(|n| !n.lt(0.0)),
                Value::String(s) => s == "auto" || s == "all" || py::isdigit(s),
                _ => false,
            };
            return if ok {
                vec![]
            } else {
                vec![format!("{k}: expected a layer count, 'auto' or 'all'")]
            };
        }
        match spec.type_.as_str() {
            "bool" => {
                if val.is_boolean() {
                    vec![]
                } else {
                    vec![format!("{k}: expected true/false")]
                }
            }
            t @ ("int" | "float") => {
                let Some(n) = Num::of(val).filter(|n| n.is_finite()) else {
                    return vec![format!("{k}: expected a number")];
                };
                if t == "int" && !n.is_integral() {
                    return vec![format!("{k}: expected an integer")];
                }
                if let Some(min) = spec.min
                    && n.lt(min)
                {
                    return vec![format!("{k}: must be >= {}", ec_proto::pyfmt::fmt_g(min))];
                }
                if let Some(max) = spec.max
                    && n.gt(max)
                {
                    return vec![format!("{k}: must be <= {}", ec_proto::pyfmt::fmt_g(max))];
                }
                vec![]
            }
            "enum" => {
                let choices = spec.choices.clone().unwrap_or_default();
                let s = py::str_of(val);
                if choices.contains(&s) {
                    vec![]
                } else {
                    vec![format!("{k}: '{s}' is not one of {}", choices.join(", "))]
                }
            }
            "string" => Self::check_string(k, val),
            "string_list" => {
                let ok = val.as_array().is_some_and(|a| {
                    !a.is_empty()
                        && a.iter()
                            .all(|x| x.as_str().is_some_and(|s| NAME_RE.is_match(s)))
                });
                if ok {
                    vec![]
                } else {
                    vec![format!(
                        "{k}: expected a non-empty list of names (letters, digits and . _ : / + -)"
                    )]
                }
            }
            _ => vec![],
        }
    }

    fn check_string(k: &str, val: &Value) -> Vec<String> {
        let Some(s) = val.as_str().filter(|s| !py::strip(s).is_empty()) else {
            return vec![format!("{k}: expected a non-empty string")];
        };
        if k == "chat_template_jinja" {
            if s.contains('\0') {
                return vec![format!("{k}: must not contain NUL characters")];
            }
            return if s.len() <= MAX_TEMPLATE_BYTES {
                vec![]
            } else {
                vec![format!("{k}: template is over {MAX_TEMPLATE_BYTES} bytes")]
            };
        }
        if k == "chat_template" {
            if s.contains('\0') || s.chars().count() > 4000 {
                return vec![format!(
                    "{k}: too long for the command line (over 4000 characters): use chat_template_jinja"
                )];
            }
            return vec![];
        }
        if CTRL_RE.is_match(s) {
            return vec![format!("{k}: must not contain control characters")];
        }
        if k == "api_key" {
            return if !SPACE_RE.is_match(s) && s.chars().count() <= 512 {
                vec![]
            } else {
                vec![format!("{k}: no whitespace, at most 512 characters")]
            };
        }
        if s.starts_with('-') {
            return vec![format!(
                "{k}: must not start with '-' (it would be read as a flag)"
            )];
        }
        if PATH_KEYS.contains(&k) {
            let items: Vec<&str> = if k == "lora" || k == "lora_scaled" {
                s.split(',').collect()
            } else {
                vec![s]
            };
            for item in items {
                let path = if k == "lora_scaled" {
                    item.rsplit_once(':').map_or(item, |(p, _)| p)
                } else {
                    item
                };
                if let Some(err) = path_error(Some(path), k) {
                    return vec![err];
                }
                if k == "lora_scaled" && !LORA_SCALED_RE.is_match(item) {
                    return vec![format!("{k}: entries look like /models/<file>:<scale>")];
                }
            }
            return vec![];
        }
        let fail = |re: &Regex, msg: &str| {
            if re.is_match(s) {
                vec![]
            } else {
                vec![format!("{k}: {msg}")]
            }
        };
        match k {
            "tensor_split" => fail(&TENSOR_SPLIT_RE, "expected proportions such as 3,1"),
            "device" => fail(
                &DEVICE_RE,
                "expected device names such as CUDA0,CUDA1, or none",
            ),
            "cors_origins" => fail(
                &ORIGINS_RE,
                "expected origins such as localhost or https://example.org",
            ),
            "override_tensor" => {
                if s.split(',').all(|part| part.contains('=')) {
                    vec![]
                } else {
                    vec![format!("{k}: expected pattern=buffer entries")]
                }
            }
            "override_kv" => {
                if s.split(',').all(|p| p.contains('=') && p.contains(':')) {
                    vec![]
                } else {
                    vec![format!("{k}: expected KEY=TYPE:VALUE entries")]
                }
            }
            _ => vec![],
        }
    }

    /// `_render`: CLI tokens for one param. bool -> --x, or --no-x when the flag has a negation (false is otherwise
    /// omitted).
    fn render(spec: &ParamSpec, val: &Value) -> Vec<String> {
        let flag = spec.flag.clone();
        match spec.type_.as_str() {
            "bool" => {
                if py::truthy(Some(val)) {
                    vec![flag]
                } else if NEGATABLE_KEYS.contains(&spec.key.as_str()) {
                    vec![format!("--no-{}", &flag[2..])]
                } else {
                    vec![]
                }
            }
            "json" => {
                let obj = match val {
                    Value::String(s) => or_raise(py::json_loads(s)),
                    other => PyVal::from_json(other),
                };
                vec![flag, obj.dumps_compact()]
            }
            "float" => vec![
                flag,
                ec_proto::pyfmt::fmt_g(Num::of(val).map_or(0.0, Num::f)),
            ],
            "int" => vec![
                flag,
                Num::of(val).map_or_else(|| py::str_of(val), Num::int_str),
            ],
            _ => vec![flag, py::str_of(val)],
        }
    }

    /// `build_launch_ctx`: ValueError for what the user must fix (the lifecycle's 422 `invalid_params`).
    pub fn py_build_launch_ctx(&self, ctx: &LaunchContext) -> PyResult<LaunchSpec> {
        let value_error = |m: String| exc("ValueError", m);
        if ctx.model_format != "gguf" {
            return Err(value_error(format!(
                "llama.cpp loads GGUF files only, not {} models",
                ctx.model_format
            )));
        }
        if let Some(err) = path_error(ctx.model_path.as_deref(), "model_path") {
            return Err(value_error(err));
        }
        let errs = self.try_validate(&ctx.params)?;
        if !errs.is_empty() {
            return Err(value_error(errs.join("; ")));
        }
        let served = py::strip(&ctx.served_name);
        if served.is_empty()
            || served.contains(',')
            || served.starts_with('-')
            || CTRL_RE.is_match(served)
        {
            return Err(value_error(
                "served name must be non-empty, without commas or control characters".into(),
            ));
        }
        let params = &ctx.params;
        let mut names = vec![served.to_string()];
        if py::truthy(params.get("served_model_name"))
            && let Some(extra) = params.get("served_model_name").and_then(Value::as_array)
        {
            names.extend(extra.iter().map(py::str_of).filter(|n| n != served));
        }
        let mut argv: Vec<String> = vec![
            "-m".into(),
            ctx.model_path.clone().unwrap_or_default(),
            "--host".into(),
            "0.0.0.0".into(),
            "--port".into(),
            CONTAINER_PORT.to_string(),
            "--alias".into(),
            names.join(","),
            "--metrics".into(),
            "--offline".into(),
        ];
        let mut env = Map::new();
        let mut files = Map::new();
        let custom_template = py::truthy(params.get("chat_template_jinja"))
            || (py::truthy(params.get("chat_template"))
                && !params
                    .get("chat_template")
                    .and_then(Value::as_str)
                    .is_some_and(|t| CHAT_TEMPLATES.contains(&t)));
        if py::get(params, "jinja").is_none() && custom_template {
            // the help: a custom template is accepted only with --jinja set before it
            argv.push("--jinja".into());
        }
        for spec in &self.catalog {
            let Some(val) = py::get(params, &spec.key) else {
                continue;
            };
            if spec.key == "served_model_name" {
                continue;
            }
            if let Some(name) = spec.flag.strip_prefix("env:") {
                env.insert(name.into(), py::str_of(val).into());
            } else if spec.key == "chat_template_jinja" {
                files.insert(CHAT_TEMPLATE_PATH.into(), py::str_of(val).into());
                argv.push(spec.flag.clone());
                argv.push(CHAT_TEMPLATE_PATH.into());
            } else {
                argv.extend(Self::render(spec, val));
            }
        }
        if py::get(params, "cors_origins").is_none() {
            argv.push("--cors-origins".into());
            argv.push("localhost".into());
        }
        Ok(LaunchSpec {
            image: DEFAULT_IMAGE.into(),
            argv,
            env,
            container_port: CONTAINER_PORT,
            health_path: "/health".into(),
            metrics_path: "/metrics".into(),
            models_path: "/v1/models".into(),
            shm_size: "1g".into(),
            ipc_host: false,
            files,
            needs_hf_cache: false,
            ulimits: Map::new(),
        })
    }
}

pub fn adapter() -> Arc<dyn EngineAdapter> {
    Arc::new(LlamaCppAdapter::new())
}

impl EngineAdapter for LlamaCppAdapter {
    fn id(&self) -> &str {
        "llamacpp"
    }
    fn display_name(&self) -> &str {
        "llama.cpp"
    }
    fn default_image(&self) -> &str {
        DEFAULT_IMAGE
    }
    fn default_port(&self) -> i64 {
        CONTAINER_PORT
    }
    fn model_formats(&self) -> Vec<String> {
        vec!["gguf".into()]
    }
    fn metric_prefixes(&self) -> Option<Vec<String>> {
        Some(vec!["llamacpp:".into()])
    }

    fn param_catalog(&self) -> Vec<ParamSpec> {
        self.catalog.clone()
    }

    fn presets(&self) -> Vec<(String, Obj)> {
        // n_gpu_layers is left on 'auto' in every preset except the MoE one: with the engine's own fit on, a model that
        // does not fit is offloaded partly instead of failing to start.
        vec![
            (
                "balanced".into(),
                obj(json!({"ctx_size": 16384, "parallel": 1, "flash_attn": "on"})),
            ),
            (
                "max-throughput".into(),
                obj(
                    json!({"parallel": 8, "ctx_size": 65536, "kv_unified": true, "cont_batching": true,
                           "ubatch_size": 1024, "flash_attn": "on"}),
                ),
            ),
            (
                "low-latency".into(),
                obj(
                    json!({"parallel": 1, "ctx_size": 8192, "flash_attn": "on", "cache_reuse": 256}),
                ),
            ),
            (
                "long-context".into(),
                obj(
                    json!({"ctx_size": 131072, "parallel": 1, "cache_type_k": "q8_0", "cache_type_v": "q8_0",
                           "flash_attn": "on"}),
                ),
            ),
            (
                "tool-agent".into(),
                obj(
                    json!({"jinja": true, "ctx_size": 65536, "parallel": 2, "flash_attn": "on", "cache_reuse": 256}),
                ),
            ),
            (
                "reasoning".into(),
                obj(
                    json!({"jinja": true, "reasoning": "on", "reasoning_format": "deepseek", "ctx_size": 32768,
                           "parallel": 1, "flash_attn": "on"}),
                ),
            ),
            (
                "big-model-offload".into(),
                obj(
                    json!({"fit": "on", "ctx_size": 8192, "parallel": 1, "cache_type_k": "q8_0",
                           "cache_type_v": "q8_0", "flash_attn": "on"}),
                ),
            ),
            (
                "moe-experts-on-cpu".into(),
                obj(
                    json!({"cpu_moe": true, "n_gpu_layers": "all", "ctx_size": 32768, "parallel": 1,
                           "flash_attn": "on"}),
                ),
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

    /// v1 bridge: llama.cpp serves a mounted GGUF, so `model` must already be a path under the model mount.
    fn build_launch(
        &self,
        model: &str,
        params: &Obj,
        hw: &Hardware,
        served_name: &str,
        hf_cache_container_path: &str,
    ) -> Result<LaunchSpec, Problem> {
        let path = model
            .starts_with(&format!("{MODEL_MOUNT}/"))
            .then(|| model.to_string());
        self.build_launch_ctx(&LaunchContext {
            model: model.into(),
            model_format: "gguf".into(),
            model_path: path,
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

    fn host_memory_gib(&self, model: &ModelInfo, params: &Obj, hw: &Hardware) -> FloatMap {
        or_raise(self.try_host_memory_gib(model, params, hw))
    }

    fn parse_metrics(&self, prometheus_text: &str) -> FloatMap {
        or_raise(self.try_parse_metrics(prometheus_text))
    }

    fn parse_startup_log(&self, line: &str) -> Option<Obj> {
        or_raise(self.try_parse_startup_log(line))
    }

    // ── the fallible forms, native (PORTING.md §A5) ────────────────────────────────────────────────────────────
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
            errs.extend(Self::check_value(spec, val)?);
        }
        if !errs.is_empty() {
            return Ok(errs);
        }
        if py::truthy(params.get("chat_template")) && py::truthy(params.get("chat_template_jinja"))
        {
            errs.push(
                "chat_template and chat_template_jinja are alternatives: set only one".into(),
            );
        }
        Ok(errs)
    }

    #[allow(clippy::too_many_lines, clippy::cast_precision_loss)]
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
        let n_gpu = hw.gpu_ids.len();
        let ctx = int_or_zero(g("ctx_size"))?;
        let split = g("split_mode");
        let split_is = |s: &str| py::is_str(split, s);
        let native = model.max_position_embeddings.filter(|n| *n != 0);
        let kv_tok = kv_bytes_per_token(model, params);
        let (w_frac, kv_frac) = offload(params, model)?;

        // Blackwell: the pinned image carries sm_120a kernels (CUDA 12.8).
        if hw.compute_capability == "12.0" {
            add(
                "ok",
                "blackwell_cuda_build",
                "This GPU is Blackwell (compute capability 12.0): the pinned llama.cpp image is built with CUDA 12.8 and carries sm_120a kernels, so no extra settings are needed.".into(),
            );
        }
        // The engine's own fit.
        let fit_on = !py::is_str(g("fit"), "off");
        let mut blockers: Vec<String> = Vec::new();
        if py::truthy(g("tensor_split")) {
            blockers.push("tensor_split".into());
        }
        if py::truthy(g("override_tensor")) {
            blockers.push("override_tensor".into());
        }
        if split_is("row") || split_is("tensor") {
            blockers.push(format!(
                "split_mode={}",
                py::str_of(split.unwrap_or(&Value::Null))
            ));
        }
        let mut unset: Vec<&str> = Vec::new();
        if ctx.is_zero() {
            unset.push("ctx_size");
        }
        if g("n_gpu_layers").is_none() || py::is_str(g("n_gpu_layers"), "auto") {
            unset.push("n_gpu_layers");
        }
        if fit_on && blockers.is_empty() && !unset.is_empty() {
            let margin =
                Int::from_f64(num(params, "fit_target", FIT_MARGIN_MIB)).unwrap_or_else(Int::zero);
            add(
                "ok",
                "fit_auto_adjust",
                format!(
                    "llama-server's own fit is on: it will choose {} so the model fits the free GPU memory, keeping {margin} MiB free per GPU. The start-up log says what it picked.",
                    unset.join(" and ")
                ),
            );
        } else if (!fit_on || !blockers.is_empty()) && ctx.is_zero() {
            let why = if fit_on {
                format!("{} turns the automatic fit off", blockers.join(", "))
            } else {
                "fit is off".into()
            };
            let size = match (native, kv_tok) {
                (Some(n), Some(k)) if k != 0.0 => format!(
                    " ({n} tokens needs about {:.1} GiB of KV cache)",
                    n as f64 * k * kv_frac / GIB
                ),
                _ => String::new(),
            };
            add(
                "warn",
                "fit_auto_adjust",
                format!(
                    "Nothing will shrink the context for you ({why}), so the engine allocates the model's full trained context{size}. Set ctx_size to a value that fits."
                ),
            );
        }
        // KV cache type needs flash attention.
        if let Some(v) = g("cache_type_v")
            && !v.as_str().is_some_and(|s| NO_QUANT_V.contains(&s))
            && py::is_str(g("flash_attn"), "off")
        {
            add(
                "block",
                "kv_quant_needs_flash_attn",
                format!(
                    "cache_type_v={} needs flash attention: llama-server refuses to start with 'V cache quantization requires flash_attn'. Set flash_attn to on (or auto), or use f16 for the values.",
                    py::str_of(v)
                ),
            );
        }
        // Multi-GPU placement.
        if let Some(ts) = g("tensor_split").filter(|v| py::truthy(Some(v))) {
            let used = split_floats(ts)?.into_iter().filter(|f| *f > 0.0).count();
            if n_gpu > 0 && used > n_gpu {
                add(
                    "warn",
                    "tensor_split_mismatch",
                    format!(
                        "tensor_split names {used} GPUs but only {n_gpu} {} selected. Give one proportion per selected GPU.",
                        if n_gpu == 1 { "is" } else { "are" }
                    ),
                );
            }
        }
        if let Some(mg) = g("main_gpu") {
            let idx = Int::from_f64(num(params, "main_gpu", 0.0)).unwrap_or_else(Int::zero);
            if n_gpu > 0
                && idx.cmp_i128(i128::try_from(n_gpu).unwrap_or(i128::MAX))
                    != std::cmp::Ordering::Less
            {
                add(
                    "block",
                    "main_gpu_out_of_range",
                    format!(
                        "main_gpu={} but the container sees {n_gpu} GPU(s), numbered from 0. Pick a lower index.",
                        py::str_of(mg)
                    ),
                );
            }
        }
        if split_is("none") && n_gpu > 1 {
            add(
                "warn",
                "split_mode_none_multi_gpu",
                format!(
                    "split_mode=none uses one GPU, yet {n_gpu} are selected: the others stay idle. Select one GPU, or use the layer split to share the model."
                ),
            );
        }
        if (split_is("row") || split_is("tensor")) && n_gpu > 1 {
            add(
                "warn",
                "tp_over_pcie",
                "Row and tensor splits exchange data between the GPUs for every layer. Without NVLink that goes over PCIe and can be slower than the default layer split; prefer one model per card unless it does not fit.".into(),
            );
        }
        if split_is("tensor") {
            let block = py::is_str(g("flash_attn"), "off");
            add(
                if block { "block" } else { "warn" },
                "split_mode_tensor_experimental",
                format!(
                    "The tensor split is marked EXPERIMENTAL in the help, needs flash attention, and turns the automatic fit off. {}",
                    if block {
                        "Set flash_attn to on or auto."
                    } else {
                        "Prefer the layer split."
                    }
                ),
            );
        }
        // CPU offload.
        let n = layer_count(g("n_gpu_layers"))?;
        if py::is_str(g("device"), "none") || n.as_ref().is_some_and(Int::is_zero) {
            add(
                "warn",
                "cpu_offload_slow",
                "No layers are offloaded: the whole model runs on the CPU, which is far slower than a GPU.".into(),
            );
        } else if let (true, Some(n), Some(layers)) =
            (w_frac < 1.0, n, model.num_layers.filter(|l| *l != 0))
        {
            add(
                "warn",
                "cpu_offload_slow",
                format!(
                    "n_gpu_layers={n} of {} layers: the rest run on the CPU and every token waits for them. Expect a large slowdown.",
                    i128::from(layers) + 1
                ),
            );
        }
        // Context beyond what the model was trained for, and context per slot.
        let par = g("parallel");
        // par in (None, -1)
        let unified = py::is_bool(g("kv_unified"), true)
            || par.is_none_or(|p| Num::of(p).is_some_and(|n| n.num_eq(Num::Int(-1))));
        let whole = Exact::I(ctx.clone());
        let per_slot = match par {
            // unified or not par or par < 2
            Some(p)
                if !unified
                    && py::truthy(Some(p))
                    && cmp_value(p, 2)? != std::cmp::Ordering::Less =>
            {
                floordiv_value(&ctx, p)?
            }
            _ => whole,
        };
        if let Some(nat) = native
            && !ctx.is_zero()
            && per_slot.cmp_i128(i128::from(nat)) == Some(std::cmp::Ordering::Greater)
        {
            if g("rope_scaling")
                .and_then(Value::as_str)
                .is_some_and(|r| r == "yarn" || r == "linear")
                || py::truthy(g("rope_scale"))
            {
                add(
                    "warn",
                    "context_beyond_native",
                    format!(
                        "A {}-token context is past the {nat} the model was trained on; rope scaling stretches it, but short-prompt quality can drop. Measure on your own prompts.",
                        per_slot.py_str()
                    ),
                );
            } else {
                let factor = per_slot
                    .truediv(i128::from(nat))
                    .and_then(|q| Int::from_f64(q.ceil()))
                    .unwrap_or_else(Int::zero);
                add(
                    "warn",
                    "context_beyond_native",
                    format!(
                        "A {}-token context is past the {nat} the model was trained on and quality degrades without scaling. Set rope_scaling=yarn with rope_scale={factor} and yarn_orig_ctx={nat}, or lower ctx_size.",
                        per_slot.py_str()
                    ),
                );
            }
        }
        // ctx and par and par > 1 and kv_unified is not True and ctx // par < 2048
        let small_slot = match par {
            Some(p) if !ctx.is_zero() && py::truthy(Some(p)) => {
                if cmp_value(p, 1)? == std::cmp::Ordering::Greater
                    && !py::is_bool(g("kv_unified"), true)
                {
                    let each = floordiv_value(&ctx, p)?;
                    (each.cmp_i128(MIN_SLOT_TOKENS) == Some(std::cmp::Ordering::Less))
                        .then_some((p, each))
                } else {
                    None
                }
            }
            _ => None,
        };
        if let Some((p, each)) = small_slot {
            add(
                "warn",
                "ctx_per_slot_small",
                format!(
                    "ctx_size {ctx} over {} slots leaves {} tokens for each conversation, so ordinary prompts are cut off. Raise ctx_size, lower parallel, or turn on kv_unified so slots share the budget.",
                    py::str_of(p),
                    each.py_str()
                ),
            );
        }
        // KV cache against free VRAM.
        if let Some(k) = kv_tok.filter(|k| *k != 0.0)
            && !ctx.is_zero()
            && !hw.gpu_free_gib.is_empty()
        {
            let free = if split_is("none") {
                hw.gpu_free_gib[0]
            } else {
                hw.gpu_free_gib.iter().fold(0.0, |a, b| a + b)
            };
            let kv_gib = fl(&ctx) * k * kv_frac / GIB;
            if kv_gib > 0.9 * free {
                add(
                    "warn",
                    "kv_exceeds_vram",
                    format!(
                        "At ctx_size {ctx} the KV cache alone needs about {kv_gib:.1} GiB and only {free:.1} GiB of GPU memory is free. Lower ctx_size, set cache_type_k and cache_type_v to q8_0 (about half the size), or set kv_offload=false (slow)."
                    ),
                );
            }
        }
        // Authentication, templates, speculation, MoE placement.
        if py::truthy(g("api_key")) {
            add(
                "warn",
                "api_key_blocks_metrics",
                "With an API key set, llama-server still answers /health and /v1/models but returns 401 for /metrics, so live metrics for this instance are lost. The engine is on an isolated network behind a loopback gateway; leave the key unset unless something else needs it.".into(),
            );
        }
        if py::is_bool(g("jinja"), false) {
            add(
                "warn",
                "jinja_disabled",
                "Without the Jinja engine tool calls and reasoning output are not parsed, and most modern chat templates do not render. Turn jinja back on unless you use one of the built-in simple templates.".into(),
            );
        }
        let stype_v = g("spec_type");
        if let Some(v @ (Value::Array(_) | Value::Object(_))) = stype_v {
            // `stype in frozenset(...)` hashes it
            return Err(exc(
                "TypeError",
                format!("unhashable type: '{}'", py::type_name(v)),
            ));
        }
        let stype = stype_v.and_then(Value::as_str);
        if let Some(t) = stype.filter(|t| DRAFT_MODEL_TYPES.contains(t))
            && !py::truthy(g("spec_draft_model"))
        {
            add(
                "warn",
                "spec_type_needs_model",
                format!(
                    "spec_type={t} needs a draft model file: set spec_draft_model (a GGUF inside the model folder)."
                ),
            );
        } else if stype == Some("draft-mtp") {
            add(
                "warn",
                "spec_type_needs_model",
                "spec_type=draft-mtp needs a GGUF that contains the model's own MTP layers; llama-server stops with 'context type MTP requested but model doesn't contain MTP layers' otherwise.".into(),
            );
        }
        let moe = moe_cpu_layers(params, model)?;
        if moe != 0.0 {
            // NaN (unknowable) also counts as "some"
            if model.is_moe {
                let hint = if g("load_mode").is_none() || py::is_str(g("load_mode"), "mmap") {
                    " The engine logs a hint to turn memory-mapping off (load_mode none) when tensors are overridden to the CPU."
                } else {
                    ""
                };
                add(
                    "ok",
                    "moe_experts_on_cpu",
                    format!(
                        "Expert weights stay in host RAM, usually most of an MoE model's size, while attention and shared layers run on the GPU. Speed then depends on RAM bandwidth.{hint}"
                    ),
                );
            } else {
                add(
                    "warn",
                    "moe_experts_on_cpu",
                    "cpu_moe / n_cpu_moe / override_tensor only move mixture-of-experts weights, and this model is not one.".into(),
                );
            }
        }
        Ok(out)
    }

    fn try_memory_model(&self, model: &ModelInfo, params: &Obj) -> PyResult<FloatMap> {
        let fit_margin_gib = num(params, "fit_target", FIT_MARGIN_MIB) / 1024.0;
        let overhead = if py::is_str(params.get("fit"), "off") {
            1.0
        } else if fit_margin_gib > 1.0 {
            fit_margin_gib
        } else {
            1.0
        };
        // no VRAM share to configure: llama.cpp takes what it needs, margin below
        let mut out: FloatMap = vec![
            ("mem_fraction".into(), 1.0),
            ("kv_bytes_per_elem".into(), kv_bytes_per_elem(params)),
            ("overhead_gib".into(), overhead),
            ("cuda_graph_gib".into(), 0.0),
            (
                "max_batched_tokens".into(),
                num(params, "ubatch_size", 512.0),
            ),
        ];
        // float(int(x)) is trunc(x) for a finite float
        let ctx = num(params, "ctx_size", 0.0).trunc();
        if ctx > 0.0 {
            // the KV buffer is sized for this many tokens in total, whatever the slot count
            out.push(("max_len".into(), ctx));
        }
        if num(params, "parallel", 0.0) > 0.0 {
            out.push(("max_seqs".into(), num(params, "parallel", 0.0)));
        }
        let tensor_split: Vec<f64> = match py::get(params, "tensor_split") {
            Some(ts) if py::truthy(Some(ts)) => split_floats(ts)?,
            _ => Vec::new(),
        };
        if py::is_str(params.get("split_mode"), "none") {
            out.push(("tp".into(), 1.0));
        } else if !tensor_split.is_empty() {
            let used = tensor_split.iter().filter(|x| **x > 0.0).count().max(1);
            #[allow(clippy::cast_precision_loss)]
            out.push(("tp".into(), used as f64));
        }
        let (w_frac, kv_frac) = offload(params, model)?;
        if w_frac < 1.0 {
            out.push(("weights_gpu_fraction".into(), w_frac));
        }
        if kv_frac < 1.0 {
            out.push(("kv_gpu_fraction".into(), kv_frac));
        }
        Ok(out)
    }

    /// Host RAM on top of the engine baseline (weights that stay on the CPU count although a memory-mapped file is
    /// reclaimable page cache); NaN means needed but unknowable.
    #[allow(clippy::cast_precision_loss)]
    fn try_host_memory_gib(
        &self,
        model: &ModelInfo,
        params: &Obj,
        _hw: &Hardware,
    ) -> PyResult<FloatMap> {
        let mut out = FloatMap::new();
        let (w_frac, kv_frac) = offload(params, model)?;
        let weights_gib = model
            .weight_bytes
            .filter(|w| *w != 0)
            .map(|w| w as f64 / GIB);
        if w_frac < 1.0 {
            out.push((
                "cpu_resident_weights".into(),
                weights_gib.map_or(f64::NAN, |w| w * (1.0 - w_frac)),
            ));
        }
        let moe = moe_cpu_layers(params, model)?;
        if moe != 0.0 {
            let v = match (weights_gib, expert_share(model)) {
                (Some(w), Some(share)) if !moe.is_nan() => w * share * moe * w_frac,
                _ => f64::NAN,
            };
            // experts of layers that are on the CPU anyway are already in cpu_resident_weights
            out.push(("moe_expert_weights_on_cpu".into(), v));
        }
        if kv_frac < 1.0 {
            let ctx = int_or_zero(py::get(params, "ctx_size"))?;
            let v = match kv_bytes_per_token(model, params) {
                Some(k) if k != 0.0 && !ctx.is_zero() => fl(&ctx) * k * (1.0 - kv_frac) / GIB,
                _ => f64::NAN,
            };
            out.push(("kv_cache_on_cpu".into(), v));
        }
        let cache_mib = match py::get(params, "cache_ram") {
            None => Int::from_i128(CACHE_RAM_MIB),
            Some(v) => py::int_of_value(v)?,
        };
        match cache_mib.cmp_i128(0) {
            // no limit: it grows with use
            std::cmp::Ordering::Less => out.push(("prompt_cache".into(), f64::NAN)),
            // a ceiling, reached only when many distinct prompts were cached
            std::cmp::Ordering::Greater => out.push((
                "prompt_cache".into(),
                cache_mib.truediv(1024).unwrap_or(f64::INFINITY),
            )),
            std::cmp::Ordering::Equal => {}
        }
        Ok(out)
    }

    /// `parse_metrics` (never raises: a bad sample value is skipped).
    fn try_parse_metrics(&self, prometheus_text: &str) -> PyResult<FloatMap> {
        Ok(scrape_to_canonical(prometheus_text))
    }

    fn try_parse_startup_log(&self, line: &str) -> PyResult<Option<Obj>> {
        parse_log_line(line)
    }
}
