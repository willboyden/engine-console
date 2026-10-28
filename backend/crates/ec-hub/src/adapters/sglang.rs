//! Port of backend/src/engine_console/adapters/sglang.py: the SGLang engine (image
//! pinned to `lmsysorg/sglang:v0.5.20-cu130`, see `sglang_params.rs`).
//!
//! Container needs the LaunchSpec cannot express (the core adds them): shm_size, `ipc: host` for TP>1, GPU pinning by
//! UUID, the engine network, the model mounts. Model formats: `hf` and `plain` (a directory mounted at `/models`). The
//! launch command is `python3 -m sglang.launch_server`; every catalog flag is also accepted by `sglang serve`.
//!
//! Python semantics (see `adapters::pycompat`): a parameter given as JSON null is Python's None, and SGLang's adapter
//! computes with some of them unguarded (`g("ep_size", 1) > 1`, `float(params.get("tp_size", 1))`): where Python raises
//! there, this port panics and the HTTP layer answers the same generic 500.

use std::sync::Arc;
use std::sync::LazyLock;

use regex::Regex;
use serde_json::{Map, Value};

use super::base::{
    Compat, EngineAdapter, FloatMap, Hardware, LaunchContext, LaunchSpec, MODEL_MOUNT, ModelInfo,
    Obj, ParamSpec,
};
use super::pycompat::{self as py, Anchor, JsonDoc, Num, compile};
use super::sglang_params::{DEFAULT_IMAGE, build_catalog};
use super::sglang_parsers;
use super::vllm::{Pattern, py_min};
use crate::domain::errors::{Problem, unprocessable};

const PORT: i64 = 30000;
/// inside the container; core publishes to 127.0.0.1
const BIND_ALL: &str = "0.0.0.0";
/// Backends refused or unbuilt on sm_120: fa3 asserts SM 8.x/9.x in attention_registry.py.
const HOPPER_ONLY: &[&str] = &["fa3", "flashmla"];
/// Backends written for datacenter Blackwell (sm_100) or untested on sm_120; allowed, but unconfirmed.
const DC_BLACKWELL: &[&str] = &["trtllm_mla", "cutedsl_mla", "tokenspeed_mla", "fa4"];
/// Parameters removed from the catalog, with the reason a saved profile that still carries them should read.
pub const RETIRED: &[(&str, &str)] = &[(
    "api_key",
    "'api_key' is not offered: SGLang 0.5.20 accepts its API key only as a command-line argument, and the console \
     never puts a secret on a command line. Put an authenticating proxy in front of the engine",
)];

/// model family regex -> tool parsers that match its dialect (first = recommended), searched in
/// `"<repo_id> <model_type>"` with `re.IGNORECASE`.
pub static TOOL_DIALECTS: LazyLock<Vec<(Pattern, &'static [&'static str])>> = LazyLock::new(|| {
    let re = |p: &str| Pattern::Re(compile(p, true, Anchor::Search));
    vec![
        (
            re(r"qwen3[._-]?(coder|next)|qwen3[._][5-9]"),
            &["qwen3_coder"][..],
        ),
        (Pattern::QwenJson, &["qwen25", "qwen", "hermes"]),
        (re(r"gpt-oss"), &["gpt-oss"]),
        (re(r"deepseek[-_]?v?3\.?2"), &["deepseekv32"]),
        (re(r"deepseek[-_]?v?3\.?1"), &["deepseekv31"]),
        (re(r"deepseek[-_]?v?4"), &["deepseekv4"]),
        (re(r"deepseek[-_]?(v3|r1)"), &["deepseekv3"]),
        (re(r"glm[-_]?(4\.7|5)"), &["glm47"]),
        (re(r"glm[-_]?4\.?[56]"), &["glm45", "glm"]),
        (re(r"kimi[-_]?k2"), &["kimi_k2"]),
        (re(r"llama[-_]?3"), &["llama3", "pythonic"]),
        (re(r"hermes"), &["hermes"]),
        (re(r"mistral|ministral|devstral"), &["mistral"]),
    ]
});

static GPTOSS: LazyLock<Regex> = LazyLock::new(|| compile(r"gpt[-_]?oss", true, Anchor::Search));
static TOOL_FAMILIES: LazyLock<Regex> = LazyLock::new(|| {
    compile(
        r"qwen3|llama|glm|kimi|mistral|hermes|deepseek",
        true,
        Anchor::Search,
    )
});
static CC_FULL: LazyLock<Regex> = LazyLock::new(|| compile(r"\d+\.\d+", false, Anchor::Full));

pub struct SglangAdapter {
    catalog: Vec<ParamSpec>,
}

impl Default for SglangAdapter {
    fn default() -> Self {
        Self::new()
    }
}

/// Python's `max(values)` of a non-empty float list (a later item replaces the current one only when larger).
fn py_max(values: &[f64]) -> f64 {
    let mut m = values.first().copied().unwrap_or(f64::NAN);
    for &x in values.iter().skip(1) {
        if x > m {
            m = x;
        }
    }
    m
}

fn compat(level: &str, code: &str, message: impl Into<String>) -> Compat {
    Compat {
        level: level.into(),
        code: code.into(),
        message: message.into(),
    }
}

fn is_str(v: Option<&Value>, s: &str) -> bool {
    v.is_some_and(|x| py::py_eq(x, &Value::String(s.into())))
}

impl SglangAdapter {
    pub fn new() -> Self {
        Self {
            catalog: build_catalog(),
        }
    }

    fn spec(&self, key: &str) -> Option<&ParamSpec> {
        self.catalog.iter().find(|p| p.key == key)
    }

    /// `_check_value`: the errors for one non-None value.
    fn check_value(spec: &ParamSpec, v: &Value) -> Vec<String> {
        let (k, t) = (spec.key.as_str(), spec.type_.as_str());
        let one = |m: String| vec![m];
        match t {
            "bool" => {
                if v.is_boolean() {
                    vec![]
                } else {
                    one(format!("{k}: expected boolean"))
                }
            }
            "int" | "float" => {
                let (Value::Number(_), Some(n)) = (v, Num::of(v)) else {
                    return one(format!("{k}: expected {t}"));
                };
                if t == "int" && matches!(n, Num::F(f) if f.fract() != 0.0 || !f.is_finite()) {
                    return one(format!("{k}: expected {t}"));
                }
                if let Some(min) = spec.min
                    && n.lt(&Num::F(min))
                {
                    return one(format!("{k}: {n} is below minimum {}", py::fmt_g(min)));
                }
                if let Some(max) = spec.max
                    && n.gt(&Num::F(max))
                {
                    return one(format!("{k}: {n} is above maximum {}", py::fmt_g(max)));
                }
                vec![]
            }
            "enum" => {
                let choices = spec.choices.clone().unwrap_or_default();
                let ok = matches!(v, Value::String(s) if choices.is_empty() || choices.contains(s));
                if ok {
                    vec![]
                } else {
                    let repr = ec_proto::pyfmt::repr_value(&Value::Array(
                        choices.iter().map(|c| Value::String(c.clone())).collect(),
                    ));
                    let shown = if spec.choices.is_some() {
                        repr
                    } else {
                        "None".to_string()
                    };
                    one(format!("{k}: '{}' not one of {shown}", py::str_of(v)))
                }
            }
            "string" => match v {
                Value::String(s) if !s.is_empty() => {
                    if s.starts_with('-') || s.contains('\0') || s.contains('\n') {
                        one(format!(
                            "{k}: value must not start with '-' or contain control characters"
                        ))
                    } else {
                        vec![]
                    }
                }
                _ => one(format!("{k}: expected non-empty string")),
            },
            "string_list" => {
                let ok = matches!(v, Value::Array(a) if !a.is_empty() && a.iter().all(|x| matches!(x,
                    Value::String(s) if !s.is_empty() && !s.starts_with('-'))));
                if ok {
                    vec![]
                } else {
                    one(format!("{k}: expected non-empty list of strings"))
                }
            }
            "json" => match v {
                Value::Object(_) | Value::Array(_) => vec![],
                Value::String(s) => {
                    if JsonDoc::loads(s).is_ok() {
                        vec![]
                    } else {
                        one(format!("{k}: invalid JSON"))
                    }
                }
                _ => one(format!("{k}: expected JSON object or string")),
            },
            _ => vec![],
        }
    }
}

impl EngineAdapter for SglangAdapter {
    fn id(&self) -> &str {
        "sglang"
    }
    fn display_name(&self) -> &str {
        "SGLang"
    }
    fn default_image(&self) -> &str {
        DEFAULT_IMAGE
    }
    fn default_port(&self) -> i64 {
        PORT
    }
    fn model_formats(&self) -> Vec<String> {
        vec!["hf".into(), "plain".into()]
    }
    fn metric_prefixes(&self) -> Option<Vec<String>> {
        Some(vec!["sglang:".into()])
    }

    fn param_catalog(&self) -> Vec<ParamSpec> {
        self.catalog.clone()
    }

    fn presets(&self) -> Vec<(String, Obj)> {
        let o = |v: Value| match v {
            Value::Object(m) => m,
            _ => Map::new(),
        };
        // Values are generic; parser presets follow a typical Qwen3 recipe (qwen3_coder + qwen3).
        vec![
            (
                "balanced".into(),
                o(
                    serde_json::json!({"mem_fraction_static": 0.85, "chunked_prefill_size": 8192,
                    "schedule_policy": "lpm", "max_running_requests": 32}),
                ),
            ),
            (
                "max-throughput".into(),
                o(
                    serde_json::json!({"mem_fraction_static": 0.88, "chunked_prefill_size": 16384,
                    "max_prefill_tokens": 32768, "schedule_policy": "lpm",
                    "enable_mixed_chunk": true, "cuda_graph_max_bs_decode": 256,
                    "stream_interval": 8}),
                ),
            ),
            (
                "low-latency".into(),
                o(
                    serde_json::json!({"mem_fraction_static": 0.80, "chunked_prefill_size": 2048,
                    "schedule_policy": "fcfs", "max_running_requests": 8,
                    "stream_interval": 1, "schedule_conservativeness": 1.0}),
                ),
            ),
            (
                "long-context".into(),
                o(
                    serde_json::json!({"mem_fraction_static": 0.85, "chunked_prefill_size": 4096,
                    "kv_cache_dtype": "fp8_e4m3", "max_running_requests": 4,
                    "schedule_policy": "lpm"}),
                ),
            ),
            (
                "tool-agent".into(),
                o(
                    serde_json::json!({"mem_fraction_static": 0.85, "chunked_prefill_size": 8192,
                    "schedule_policy": "lpm", "tool_call_parser": "qwen3_coder",
                    "reasoning_parser": "qwen3", "enable_cache_report": true,
                    "max_running_requests": 16}),
                ),
            ),
            (
                "reasoning".into(),
                o(
                    serde_json::json!({"mem_fraction_static": 0.85, "chunked_prefill_size": 8192,
                    "reasoning_parser": "qwen3", "schedule_policy": "lpm",
                    "watchdog_timeout": 1800}),
                ),
            ),
        ]
    }

    fn validate(&self, params: &Obj) -> Vec<String> {
        let mut errs: Vec<String> = Vec::new();
        for (k, v) in params {
            let Some(spec) = self.spec(k) else {
                errs.push(RETIRED.iter().find(|(r, _)| r == k).map_or_else(
                    || format!("unknown parameter '{k}'"),
                    |(_, m)| (*m).to_string(),
                ));
                continue;
            };
            if v.is_null() {
                continue;
            }
            errs.extend(Self::check_value(spec, v));
        }
        if !errs.is_empty() {
            return errs;
        }
        let g = |k: &str| params.get(k);
        let one = Value::from(1);
        let algo = g("speculative_algorithm");
        if (is_str(algo, "EAGLE") || is_str(algo, "EAGLE3") || is_str(algo, "STANDALONE"))
            && !py::truthy_opt(g("speculative_draft_model_path"))
        {
            errs.push(format!(
                "speculative_algorithm={} requires speculative_draft_model_path",
                py::str_of(algo.unwrap_or(&Value::Null))
            ));
        }
        if py::is_none(algo)
            && [
                "speculative_draft_model_path",
                "speculative_num_steps",
                "speculative_eagle_topk",
                "speculative_num_draft_tokens",
            ]
            .iter()
            .any(|k| py::truthy_opt(g(k)))
        {
            errs.push("speculative_* options are set but speculative_algorithm is not".into());
        }
        if py::truthy_opt(g("enable_dp_attention"))
            && !py::py_eq(
                py::get_or(params, "dp_size", &one),
                py::get_or(params, "tp_size", &one),
            )
        {
            errs.push("enable_dp_attention requires dp_size == tp_size".into());
        }
        let (tp, ep) = (
            py::get_or(params, "tp_size", &one),
            py::get_or(params, "ep_size", &one),
        );
        let ep_n = Num::need(Some(ep), ">");
        if ep_n.gt(&Num::I(1)) && !Num::need(Some(tp), "%").rem(&ep_n).same(&Num::I(0)) {
            errs.push(format!(
                "ep_size ({}) must divide tp_size ({})",
                py::str_of(ep),
                py::str_of(tp)
            ));
        }
        if py::truthy_opt(g("enable_hierarchical_cache"))
            && py::truthy_opt(g("disable_radix_cache"))
        {
            errs.push("enable_hierarchical_cache requires the radix cache (disable_radix_cache must be off)".into());
        }
        if py::truthy_opt(g("enable_trace")) && !py::truthy_opt(g("otlp_traces_endpoint")) {
            errs.push("enable_trace requires otlp_traces_endpoint".into());
        }
        if py::truthy_opt(g("hicache_size")) && !py::truthy_opt(g("enable_hierarchical_cache")) {
            errs.push("hicache_* options need enable_hierarchical_cache".into());
        }
        errs
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
                "SGLang cannot load {} models (it loads: {})",
                ctx.model_format,
                formats.join(", ")
            )));
        }
        let errs = self.validate(&ctx.params);
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
        let params = &ctx.params;
        let served = match params.get("served_model_name") {
            Some(v) if py::truthy(v) => py::str_of(v),
            _ => ctx.served_name.clone(),
        };
        let mut argv: Vec<String> = [
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            &model_arg,
            "--host",
            BIND_ALL,
            "--port",
            &PORT.to_string(),
            "--served-model-name",
            &served,
            // forced: the console's metrics/dashboards depend on /metrics
            "--enable-metrics",
        ]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
        let mut extra_env: Vec<(String, String)> = Vec::new();
        for spec in &self.catalog {
            if spec.key == "enable_metrics" || spec.key == "served_model_name" {
                continue;
            }
            if spec.key == "revision" && plain {
                continue; // a directory has no hub revision
            }
            let Some(v) = params.get(&spec.key).filter(|v| !v.is_null()) else {
                continue;
            };
            if let Some(name) = spec.flag.strip_prefix("env:") {
                let val = if spec.type_ == "bool" {
                    if py::truthy(v) {
                        "1".to_string()
                    } else {
                        "0".to_string()
                    }
                } else {
                    py::str_of(v)
                };
                extra_env.push((name.to_string(), val));
            } else {
                match spec.type_.as_str() {
                    "bool" => {
                        if py::truthy(v) {
                            argv.push(spec.flag.clone());
                        }
                    }
                    "string_list" => {
                        argv.push(spec.flag.clone());
                        argv.extend(
                            v.as_array()
                                .map_or(&[][..], Vec::as_slice)
                                .iter()
                                .map(py::str_of),
                        );
                    }
                    "json" => {
                        argv.push(spec.flag.clone());
                        argv.push(match v {
                            Value::String(s) => s.clone(),
                            other => crate::pyjson::dumps_compact(other),
                        });
                    }
                    "float" => {
                        argv.push(spec.flag.clone());
                        argv.push(py::repr(py::float_or_raise(Some(v))));
                    }
                    _ => {
                        argv.push(spec.flag.clone());
                        argv.push(py::str_of(v));
                    }
                }
            }
        }
        // Telemetry off (deny-by-default egress). A plain model is local: never wait on the hub.
        let mut env = Map::new();
        env.insert("HF_HUB_DISABLE_TELEMETRY".into(), "1".into());
        if plain {
            env.insert("HF_HUB_OFFLINE".into(), "1".into());
        } else {
            env.insert(
                "HF_HOME".into(),
                Value::String(ctx.hf_cache_container_path.clone()),
            );
        }
        for (k, v) in extra_env {
            env.insert(k, Value::String(v));
        }
        Ok(LaunchSpec {
            image: DEFAULT_IMAGE.into(),
            argv,
            env,
            container_port: PORT,
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
        let g = |k: &str| params.get(k);
        let one = Value::from(1);
        let (tp_v, dp_v, pp_v) = (
            py::get_or(params, "tp_size", &one),
            py::get_or(params, "dp_size", &one),
            py::get_or(params, "pp_size", &one),
        );
        let tp = Num::need(Some(tp_v), "*");
        let pp = Num::need(Some(pp_v), "*");
        let tp_pp = tp.mul(&pp);
        let world = if py::truthy_opt(g("enable_dp_attention")) {
            tp_pp.mul(&Num::I(1))
        } else {
            tp_pp.mul(&Num::need(Some(dp_v), "*"))
        };
        let n_gpus = hw.gpu_ids.len();
        if world.gt(&Num::I(i128::try_from(n_gpus).unwrap_or(i128::MAX))) {
            out.push(compat(
                "block",
                "gpus_insufficient",
                format!(
                    "tp*pp*dp needs {world} GPUs but only {n_gpus} selected; lower tp_size/dp_size or select more GPUs."
                ),
            ));
        }
        // SGLang stops at start-up when the selected GPUs' free memory differs by more than 10 %.
        let free = &hw.gpu_free_gib;
        if tp.gt(&Num::I(1)) && free.len() > 1 {
            let (lo, hi) = (py_min(free), py_max(free));
            if hi > 0.0 && lo < 0.9 * hi {
                out.push(compat(
                    "block",
                    "tp_memory_unbalanced",
                    format!(
                        "The selected GPUs have {} and {} GiB free; SGLang stops with 'The memory capacity is \
                         unbalanced' when they differ by more than 10 %. Free the busier card, stop what shares it, or \
                         pick cards with similar free memory.",
                        py::fixed(lo, 1),
                        py::fixed(hi, 1)
                    ),
                ));
            }
        }
        let frac = g("mem_fraction_static").filter(|v| !v.is_null());
        if let Some(f) = frac
            && Num::need(Some(f), ">").gt(&Num::F(0.92))
        {
            out.push(compat(
                "warn",
                "mem_fraction_high",
                format!(
                    "mem_fraction_static={} leaves too little headroom for activations, CUDA graphs and kernel \
                     scratch: SGLang's own automatic choice normally keeps at least 10 GiB free on cards over 60 GB. \
                     Out-of-memory during a long prefill is likely. Keep it at 0.92 or below unless the card is \
                     dedicated and you tested it.",
                    py::str_of(f)
                ),
            ));
        }
        let quant: Option<String> = model
            .quantization
            .as_deref()
            .map(str::to_lowercase)
            .filter(|q| !q.is_empty());
        let q_arg = g("quantization").filter(|v| !v.is_null());
        let name = format!(
            "{} {}",
            model.repo_id,
            model.model_type.clone().unwrap_or_default()
        );
        let is_gptoss = GPTOSS.is_match(&name);
        let cc = hw.compute_capability.as_str();
        if quant.as_deref() == Some("nvfp4")
            || is_str(q_arg, "modelopt_fp4")
            || is_str(q_arg, "nvfp4_online")
        {
            out.push(compat(
                "ok",
                "nvfp4_ok_sglang",
                format!(
                    "NVFP4 {}",
                    if model.is_moe {
                        "MoE: SGLang is the recommended engine on sm_120 (vLLM can fall back to a slow Marlin kernel)."
                    } else {
                        "dense: supported on sm_120."
                    }
                ),
            ));
        }
        if quant.as_deref() == Some("mxfp4") || is_str(q_arg, "mxfp4") {
            if model.is_moe && !is_gptoss && cc.starts_with("12") {
                out.push(compat(
                    "block",
                    "mxfp4_moe_sm120",
                    "MXFP4 MoE asserts on sm_120 except gpt-oss. Use an FP8 checkpoint (or quantization=fp8).",
                ));
            } else if is_gptoss {
                out.push(compat(
                    "warn",
                    "mxfp4_gptoss_engine",
                    "gpt-oss ships native MXFP4 weights; the SGLang path for them on sm_120 is untested here. If it \
                     fails, run the model with llama.cpp or Ollama.",
                ));
            }
        }
        if is_str(q_arg, "fp8") && quant.as_deref().is_some_and(|q| q != "fp8") {
            out.push(compat(
                "warn",
                "quant_override_prequantized",
                format!(
                    "quantization=fp8 on a pre-quantized ({}) checkpoint misloads; unset quantization so SGLang \
                     auto-detects its config.",
                    quant.clone().unwrap_or_default()
                ),
            ));
        }
        if is_str(q_arg, "modelopt_fp4") && quant.as_deref().is_some_and(|q| q != "nvfp4") {
            out.push(compat(
                "warn",
                "modelopt_fp4_mismatch",
                "modelopt_fp4 needs an NVFP4 ModelOpt checkpoint.",
            ));
        }
        if py::truthy_opt(q_arg) && !q_arg.is_some_and(Value::is_string) {
            py::raise(
                "AttributeError",
                &format!(
                    "'{}' object has no attribute 'startswith'",
                    py::type_name(q_arg)
                ),
            );
        }
        if py::truthy_opt(q_arg)
            && q_arg
                .and_then(Value::as_str)
                .is_some_and(|q| q.starts_with("modelopt"))
            && quant.is_none()
        {
            out.push(compat(
                "warn",
                "modelopt_unquantized",
                "ModelOpt quantization expects a pre-quantized checkpoint; for a bf16 model use quantization=fp8.",
            ));
        }
        // --- attention backends versus GPU generation
        let mut backends: Vec<String> = Vec::new();
        for k in [
            "attention_backend",
            "prefill_attention_backend",
            "decode_attention_backend",
        ] {
            if let Some(v) = g(k).filter(|v| py::truthy(v)) {
                if v.is_array() || v.is_object() {
                    py::raise(
                        "TypeError",
                        &format!("unhashable type: '{}'", py::type_name(Some(v))),
                    );
                }
                let s = py::str_of(v);
                if v.is_string() && !backends.contains(&s) {
                    backends.push(s);
                }
            }
        }
        if cc.starts_with("12") {
            let mut hopper: Vec<&str> = HOPPER_ONLY
                .iter()
                .copied()
                .filter(|b| backends.iter().any(|x| x == b))
                .collect();
            hopper.sort_unstable();
            if !hopper.is_empty() {
                let extra = if hopper.contains(&"fa3") {
                    " SGLang stops with 'FlashAttention v3 Backend requires SM>=80 and SM<=90'."
                } else {
                    ""
                };
                out.push(compat(
                    "block",
                    "attention_backend_hopper_only",
                    format!(
                        "attention_backend {} is built for Hopper (sm_90), not sm_120.{extra} Use flashinfer or triton.",
                        hopper.join(", ")
                    ),
                ));
            }
            let mut dc: Vec<&str> = DC_BLACKWELL
                .iter()
                .copied()
                .filter(|b| backends.iter().any(|x| x == b))
                .collect();
            dc.sort_unstable();
            if !dc.is_empty() {
                out.push(compat(
                    "warn",
                    "attention_backend_dc_blackwell",
                    format!(
                        "attention_backend {} targets datacenter Blackwell (sm_100); unconfirmed on sm_120. Prefer \
                         flashinfer or triton.",
                        dc.join(", ")
                    ),
                ));
            }
            if (is_str(g("attention_backend"), "trtllm_mha")
                || is_str(g("prefill_attention_backend"), "trtllm_mha"))
                && is_str(g("kv_cache_dtype"), "fp8_e4m3")
            {
                out.push(compat(
                    "block",
                    "trtllm_mha_fp8_kv_sm120",
                    "On sm_120 the trtllm_mha prefill kernel does not support an fp8_e4m3 KV cache (SGLang stops at \
                     start-up). Use kv_cache_dtype auto, or another prefill attention backend such as flashinfer.",
                ));
            }
        }
        // --- hybrid-model kernels (image source: arg_groups/attention_hook.py, handle_linear_attn_backend)
        let cc_major = if CC_FULL.is_match(py::strip(cc)) {
            py::int_str(cc.split('.').next().unwrap_or(""))
                .map(|b| b.small().map_or(Num::Big(b), Num::I))
        } else {
            None
        };
        let lin = py::or_value(
            g("linear_attn_decode_backend"),
            g("linear_attn_backend").unwrap_or(&Value::Null),
        );
        if cc_major.as_ref().is_some_and(|c| !c.lt(&Num::I(10)))
            && py::py_eq(lin, &Value::from("flashinfer"))
        {
            let ssm = g("mamba_ssm_dtype").filter(|v| !v.is_null());
            match ssm {
                Some(s) if !py::py_eq(s, &Value::from("bfloat16")) => out.push(compat(
                    "block",
                    "linear_attn_flashinfer_needs_bf16_state",
                    format!(
                        "The FlashInfer linear-attention decode kernel needs mamba_ssm_dtype bfloat16 on this GPU \
                         generation (SGLang stops with '--linear-attn-decode-backend flashinfer on SM100+ requires \
                         --mamba-ssm-dtype bfloat16'; sm_120 counts); you set {}. Set it to bfloat16 or use the triton \
                         decode backend.",
                        py::str_of(s)
                    ),
                )),
                None => out.push(compat(
                    "warn",
                    "linear_attn_flashinfer_needs_bf16_state",
                    "The FlashInfer linear-attention decode kernel needs mamba_ssm_dtype bfloat16 on this GPU \
                     generation; SGLang stops otherwise. Set mamba_ssm_dtype to bfloat16 unless the model's own \
                     configuration already says so.",
                )),
                _ => {}
            }
        }
        // --- tool-call dialect
        let tool = g("tool_call_parser");
        if py::truthy_opt(tool) && !is_str(tool, "auto") {
            if let Some((_, allowed)) = TOOL_DIALECTS.iter().find(|(p, _)| p.search(&name))
                && !tool
                    .and_then(Value::as_str)
                    .is_some_and(|t| allowed.contains(&t))
            {
                out.push(compat(
                    "warn",
                    "tool_parser_dialect_mismatch",
                    format!(
                        "tool_call_parser={} does not match this model's dialect; use {}. Qwen3.x/Coder emit XML \
                         (<function=..>) which needs qwen3_coder; qwen expects JSON in <tool_call>.",
                        py::str_of(tool.unwrap_or(&Value::Null)),
                        allowed.join(" / ")
                    ),
                ));
            }
        } else if !py::truthy_opt(tool) && TOOL_FAMILIES.is_match(&name) {
            out.push(compat(
                "warn",
                "no_tool_parser",
                "No tool_call_parser: agentic clients get raw text in content and an empty tool_calls, then stall. \
                 Set one if you serve tool-using clients.",
            ));
        }
        if py::truthy_opt(g("enable_dp_attention")) {
            if !py::py_eq(dp_v, tp_v) {
                out.push(compat(
                    "block",
                    "dp_attention_dp_ne_tp",
                    "enable_dp_attention requires dp_size == tp_size.",
                ));
            }
            if !model.is_moe && model.kv_lora_rank.is_none_or(|r| r == 0) {
                out.push(compat(
                    "warn",
                    "dp_attention_unsupported_arch",
                    "DP attention is supported for DeepSeek-style (MLA) and Qwen MoE models; this model looks dense.",
                ));
            }
        }
        if Num::need(Some(py::get_or(params, "ep_size", &one)), ">").gt(&Num::I(1)) && !model.is_moe
        {
            out.push(compat(
                "warn",
                "ep_dense_model",
                "ep_size > 1 has no effect on a dense model.",
            ));
        }
        let (ctx, native) = (g("context_length"), model.max_position_embeddings);
        if let Some(c) = ctx.filter(|v| py::truthy(v))
            && let Some(n) = native.filter(|n| *n != 0)
            && Num::need(Some(c), ">").gt(&Num::I(i128::from(n)))
        {
            let c = py::str_of(c);
            if py::truthy_opt(g("allow_longer_context_len")) {
                out.push(compat(
                    "warn",
                    "context_beyond_native",
                    format!(
                        "context_length {c} > native {n}: SGLang accepts it with a warning that 'This may lead to \
                         incorrect model outputs or CUDA errors'. Configure rope scaling (for example YaRN in \
                         json_model_override_args) and measure quality before relying on it."
                    ),
                ));
            } else {
                out.push(compat(
                    "warn",
                    "context_beyond_native",
                    format!(
                        "context_length {c} > native {n}: SGLang stops at start-up with 'User-specified \
                         context_length ... is greater than the derived context_length' unless you turn on \
                         allow_longer_context_len and configure rope scaling (for example YaRN in \
                         json_model_override_args), or lower context_length."
                    ),
                ));
            }
        }
        if g("chunked_prefill_size").is_some_and(|v| py::py_eq(v, &Value::from(-1)))
            && py::truthy_opt(g("enable_mixed_chunk"))
        {
            out.push(compat(
                "warn",
                "mixed_chunk_needs_chunked_prefill",
                "enable_mixed_chunk needs chunked prefill; chunked_prefill_size=-1 disables it.",
            ));
        }
        let kvd = g("kv_cache_dtype");
        if is_str(kvd, "fp4_e2m1") || is_str(kvd, "nvfp4") || is_str(kvd, "fp4_mx_block16") {
            out.push(compat(
                "warn",
                "kv_fp4_experimental",
                "FP4 KV cache types are new in SGLang and need CUDA 12.8 or newer; output quality is unvalidated here.",
            ));
        }
        if py::truthy_opt(g("enable_torch_compile")) {
            out.push(compat(
                "warn",
                "torch_compile_slow_start",
                "torch.compile is experimental and adds minutes to startup.",
            ));
        }
        if py::truthy_opt(g("cpu_offload_gb")) {
            out.push(compat(
                "warn",
                "cpu_offload_slow",
                "CPU weight offload streams over PCIe every step; expect a large slowdown.",
            ));
        }
        if matches!(g("enable_metrics"), Some(Value::Bool(false))) {
            out.push(compat(
                "warn",
                "metrics_forced",
                "enable_metrics=false is ignored: the console needs /metrics.",
            ));
        }
        if py::truthy_opt(g("log_requests")) {
            out.push(compat(
                "warn",
                "log_requests_leaks",
                "log_requests writes prompts and outputs into container logs.",
            ));
        }
        if py::truthy_opt(g("trust_remote_code")) {
            out.push(compat(
                "warn",
                "trust_remote_code",
                "trust_remote_code executes model-repo Python code in the container.",
            ));
        }
        if model.gated {
            out.push(compat(
                "warn",
                "gated_model",
                "Gated model: accept the license on Hugging Face and provide HF_TOKEN.",
            ));
        }
        out
    }

    /// SGLang semantics (0.5.20, kv_cache_configurator.py): the static pool is what is free when the engine starts
    /// times mem_fraction_static and holds weights plus the KV pool only, so overhead/cuda_graph are 0 here and
    /// `nonstatic_reserve_frac` says how much is left for activations and graphs. 0.85 is a conservative planning value.
    fn memory_model(&self, model: &ModelInfo, params: &Obj) -> FloatMap {
        let kv = params.get("kv_cache_dtype");
        let kv_bytes = if is_str(kv, "fp8_e5m2") || is_str(kv, "fp8_e4m3") {
            1.0
        } else if is_str(kv, "fp4_e2m1") || is_str(kv, "nvfp4") || is_str(kv, "fp4_mx_block16") {
            0.5
        } else {
            2.0
        };
        let frac = params.get("mem_fraction_static").filter(|v| !v.is_null());
        let mem_fraction = frac.map_or(0.85, |f| py::float_or_raise(Some(f)));
        let one = Value::from(1);
        let mut out: FloatMap = vec![
            ("mem_fraction".into(), mem_fraction),
            (
                "mem_fraction_is_default".into(),
                if frac.is_some() { 0.0 } else { 1.0 },
            ),
            ("kv_bytes_per_elem".into(), kv_bytes),
            ("overhead_gib".into(), 0.0),
            ("cuda_graph_gib".into(), 0.0),
            (
                "tp".into(),
                py::float_or_raise(Some(py::get_or(params, "tp_size", &one))),
            ),
            // 0.5.20 kv_cache_configurator._profile_available_bytes: KV = fraction x free at launch - weights.
            ("share_of_free".into(), 1.0),
        ];
        out.push(("nonstatic_reserve_frac".into(), 1.0 - mem_fraction));
        let pos = model.max_position_embeddings.map(Value::from);
        let ctx = match params.get("context_length") {
            Some(v) if py::truthy(v) => Some(v.clone()),
            _ => pos,
        };
        if let Some(c) = ctx.filter(py::truthy) {
            out.push(("max_len".into(), py::float_or_raise(Some(&c))));
        }
        if let Some(m) = params.get("max_running_requests").filter(|v| py::truthy(v)) {
            out.push(("max_seqs".into(), py::float_or_raise(Some(m))));
        }
        out
    }

    fn host_memory_gib(&self, model: &ModelInfo, params: &Obj, hw: &Hardware) -> FloatMap {
        let g = |k: &str| params.get(k);
        let tp = py::int_or_raise(Some(py::or_value(g("tp_size"), &Value::from(1))));
        let mut out: FloatMap = Vec::new();
        if py::truthy_opt(g("cpu_offload_gb")) {
            // assumed per TP rank (like vLLM)
            let gb = py::float_or_raise(Some(py::or_value(g("cpu_offload_gb"), &Value::from(0))));
            out.push(("cpu_weight_offload".into(), Num::F(gb).mul(&tp).as_f64()));
        }
        if py::truthy_opt(g("enable_hierarchical_cache")) {
            let key = "hicache_host_pool (assumed per TP rank)".to_string();
            if py::truthy_opt(g("hicache_size")) {
                // GB; overrides hicache_ratio
                let gb = py::float_or_raise(Some(py::or_value(g("hicache_size"), &Value::from(0))));
                out.push((key, Num::F(gb).mul(&tp).as_f64()));
            } else {
                // ratio is relative to the DEVICE KV pool: static budget minus this rank's weights
                let frac = py::float_or_raise(Some(py::or_value(
                    g("mem_fraction_static"),
                    &Value::from(0.85),
                )));
                match model.weight_bytes.filter(|w| *w != 0) {
                    Some(wb) if !hw.gpu_total_gib.is_empty() => {
                        let tp_f = tp.as_f64();
                        let x =
                            frac * py_min(&hw.gpu_total_gib) - (wb as f64) / 1_073_741_824.0 / tp_f;
                        let pool = if 0.0 > x { 0.0 } else { x };
                        let ratio = py::float_or_raise(Some(py::or_value(
                            g("hicache_ratio"),
                            &Value::from(2.0),
                        )));
                        out.push((key, ratio * pool * tp_f));
                    }
                    // device pool unknown: never guess
                    _ => out.push((key, f64::NAN)),
                }
            }
        }
        out
    }

    fn parse_metrics(&self, prometheus_text: &str) -> FloatMap {
        sglang_parsers::parse_metrics(prometheus_text)
    }

    fn parse_startup_log(&self, line: &str) -> Option<Obj> {
        sglang_parsers::parse_startup_log(line)
    }
}

pub fn adapter() -> Arc<dyn EngineAdapter> {
    Arc::new(SglangAdapter::new())
}
