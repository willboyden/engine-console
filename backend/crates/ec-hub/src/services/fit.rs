//! Python: backend/src/engine_console/services/fit.py (tests/core/test_fit.py,
//! test_fit_host_ram.py, test_fit_offload.py; docs/FIT-ESTIMATOR.md).
//!
//! The pure fit estimator: `estimate_fit(model, mem, hw, concurrency, resident, compat, hw_all) -> FitReport` and
//! `estimate_host_ram(breakdown, host) -> HostRamFit`. Internal: only services/fitting.rs calls it.
//!
//! Everything is per GPU, in GiB. Never fabricate: a missing input degrades `confidence` and, when the KV cache
//! cannot be computed at all, the verdict becomes `unknown` unless weights alone already prove `wont_fit`. The float
//! arithmetic follows Python's operation order exactly (the results are compared against Python goldens), and every
//! place where Python raises (a division by zero, `int()` of a non-finite float) is an error here too.

use std::collections::BTreeSet;

use serde_json::{Map, Value};

use crate::adapters::base::{Compat, FloatMap, Hardware, ModelInfo};
use crate::domain::errors::{Problem, internal};
use crate::domain::models::{FitReport, GpuFit, HostMemInfo, HostRamFit, ResidentUse};
use crate::services::gguf_info::py;
use crate::services::hf::py_round;

pub const GIB: f64 = 1_073_741_824.0;
/// total <= 90 % of budget -> fits
pub const FITS_MAX: f64 = 0.90;
/// 90 %..100 % -> tight; above -> wont_fit
pub const TIGHT_MAX: f64 = 1.00;
pub const DEFAULT_MEM_FRACTION: f64 = 0.90;
/// CUDA context + NCCL buffers, per GPU
pub const DEFAULT_OVERHEAD_GIB: f64 = 1.0;
pub const DEFAULT_GRAPH_GIB: f64 = 1.0;
pub const DEFAULT_BATCHED_TOKENS: f64 = 8192.0;
/// bytes-per-hidden-element multiplier for prefill activations (rough)
pub const ACT_FACTOR: f64 = 16.0;
pub const ACT_MIN_GIB: f64 = 0.5;
/// when hidden_size is unknown
pub const ACT_DEFAULT_GIB: f64 = 1.5;
/// DeepSeek MLA decoupled-rope dim when absent from config
pub const DEFAULT_QK_ROPE: i64 = 64;
/// heuristic: python + CUDA runtime anon memory (~3 GiB measured on a running SGLang container)
pub const ENGINE_PROCESS_GIB: f64 = 4.0;
pub const HOST_OK: f64 = 0.80;

/// bytes per parameter, only used when the on-disk safetensors size is unknown (value, its Python repr)
const BYTES_PER_PARAM: &[(&str, f64, &str)] = &[
    ("fp32", 4.0, "4"),
    ("float32", 4.0, "4"),
    ("bf16", 2.0, "2"),
    ("bfloat16", 2.0, "2"),
    ("fp16", 2.0, "2"),
    ("float16", 2.0, "2"),
    ("fp8", 1.0, "1"),
    ("int8", 1.0, "1"),
    ("nvfp4", 0.5, "0.5"),
    ("mxfp4", 0.5, "0.5"),
    ("fp4", 0.5, "0.5"),
    ("awq", 0.5, "0.5"),
    ("gptq", 0.5, "0.5"),
    ("int4", 0.5, "0.5"),
];

/// `dict[str, float]` built from an adapter's memory model (a later duplicate key wins, as in a dict).
#[derive(Debug, Clone, Default)]
pub struct Mem(Vec<(String, f64)>);

impl Mem {
    pub fn from_map(m: &FloatMap) -> Self {
        let mut out: Vec<(String, f64)> = Vec::new();
        for (k, v) in m {
            match out.iter_mut().find(|(kk, _)| kk == k) {
                Some(slot) => slot.1 = *v,
                None => out.push((k.clone(), *v)),
            }
        }
        Self(out)
    }
    pub fn get(&self, k: &str) -> Option<f64> {
        self.0.iter().find(|(kk, _)| kk == k).map(|(_, v)| *v)
    }
    pub fn has(&self, k: &str) -> bool {
        self.0.iter().any(|(kk, _)| kk == k)
    }
    /// `bool(x)` of a float: NaN is true, only 0.0 is false.
    fn truthy(&self, k: &str) -> Option<f64> {
        self.get(k).filter(|v| *v != 0.0)
    }
}

fn bad() -> Problem {
    internal("unexpected server error")
}

/// `a / b` for floats: Python raises ZeroDivisionError (a 500) where IEEE gives inf/NaN.
fn div(a: f64, b: f64) -> Result<f64, Problem> {
    if b == 0.0 { Err(bad()) } else { Ok(a / b) }
}

/// `int(x)` of a float: truncation; NaN and infinities raise in Python.
fn int_f(x: f64) -> Result<i64, Problem> {
    if !x.is_finite() {
        return Err(bad());
    }
    #[allow(clippy::cast_possible_truncation)]
    Ok(x.trunc() as i64)
}

/// `a // b` for floats (CPython's float_floor_div).
fn floordiv_f(a: f64, b: f64) -> Result<f64, Problem> {
    if b == 0.0 {
        return Err(bad());
    }
    let m = a % b;
    let mut d = (a - m) / b;
    if m != 0.0 && ((b < 0.0) != (m < 0.0)) {
        d -= 1.0;
    }
    if d != 0.0 {
        let mut f = d.floor();
        if d - f > 0.5 {
            f += 1.0;
        }
        Ok(f)
    } else {
        Ok((0.0f64).copysign(a / b))
    }
}

/// `f"{n:,}"` for an int.
fn thousands(n: i64) -> String {
    let s = n.unsigned_abs().to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    if n < 0 { format!("-{out}") } else { out }
}

fn g(x: f64) -> String {
    ec_proto::pyfmt::fmt_g(x)
}

/// `f"{x:.0%}"`
fn pct0(x: f64) -> String {
    format!("{:.0}%", x * 100.0)
}

fn cfg_of(model: &ModelInfo) -> &Map<String, Value> {
    match model.raw_config.get("text_config") {
        Some(Value::Object(inner)) => inner,
        _ => &model.raw_config,
    }
}

/// Return (bytes, caveat). Caveat is set when the number is derived rather than measured.
fn weights_bytes(model: &ModelInfo) -> (Option<f64>, Option<String>) {
    if let Some(w) = model.weight_bytes.filter(|w| *w != 0) {
        #[allow(clippy::cast_precision_loss)]
        return (Some(w as f64), None);
    }
    if let Some(n) = model.num_params.filter(|n| *n != 0) {
        let key = model
            .quantization
            .as_deref()
            .filter(|q| !q.is_empty())
            .or_else(|| model.dtype.as_deref().filter(|d| !d.is_empty()))
            .unwrap_or("")
            .to_lowercase();
        if let Some((_, bpp, repr)) = BYTES_PER_PARAM.iter().find(|(k, _, _)| *k == key) {
            #[allow(clippy::cast_precision_loss)]
            return (
                Some(n as f64 * bpp),
                Some(format!(
                    "weights estimated from {} params at {repr} B/param",
                    thousands(n)
                )),
            );
        }
    }
    (None, None)
}

/// (full_layers, sliding_layers, window, note).
fn layer_split(model: &ModelInfo) -> (i64, i64, Option<i64>, Option<String>) {
    let layers = model.num_layers.unwrap_or(0);
    let window = model.sliding_window;
    let cfg = cfg_of(model);
    if window.is_none_or(|w| w == 0)
        || cfg.get("use_sliding_window") == Some(&Value::Bool(false))
        || layers <= 0
    {
        return (layers, 0, None, None);
    }
    if let Some(Value::Array(types)) = cfg.get("layer_types")
        && i64::try_from(types.len()).ok() == Some(layers)
    {
        let n_sl = types
            .iter()
            .filter(|t| t.as_str() == Some("sliding_attention"))
            .count();
        let n_sl = i64::try_from(n_sl).unwrap_or(0);
        return (layers - n_sl, n_sl, window, None);
    }
    if let Some(p) = cfg.get("sliding_window_pattern").and_then(py::as_int)
        && p > 1
    {
        let n_full = py::floordiv(layers, p).unwrap_or(0);
        return (n_full, layers - n_full, window, None);
    }
    (
        0,
        layers,
        window,
        Some("sliding_window set without a layer pattern: assumed every layer is windowed".into()),
    )
}

/// Per-GPU KV cache sizing for one (model, kv dtype, tp).
#[derive(Debug, Clone)]
pub struct Kv {
    per_layer: f64,
    n_full: i64,
    n_sl: i64,
    window: Option<i64>,
}

impl Kv {
    /// `_Kv(per_layer_token_bytes, n_full, n_sl, window)`
    pub fn new(per_layer: f64, n_full: i64, n_sl: i64, window: Option<i64>) -> Self {
        Self {
            per_layer,
            n_full,
            n_sl,
            window,
        }
    }

    #[allow(clippy::cast_precision_loss)]
    pub fn bytes_for(&self, ctx: i64, seqs: i64) -> f64 {
        let eff_sl = match self.window.filter(|w| *w != 0) {
            Some(w) => ctx.min(w),
            None => ctx,
        };
        let tokens =
            i128::from(self.n_full) * i128::from(ctx) + i128::from(self.n_sl) * i128::from(eff_sl);
        self.per_layer * (tokens as f64) * (seqs as f64)
    }

    #[allow(clippy::cast_precision_loss)]
    pub fn bytes_per_token_full(&self) -> f64 {
        self.per_layer * ((self.n_full + self.n_sl) as f64)
    }

    /// Largest per-sequence context that fits `seqs` sequences in the pool (closed form).
    #[allow(clippy::cast_precision_loss)]
    pub fn max_ctx(&self, pool_bytes: f64, seqs: i64) -> Result<Option<i64>, Problem> {
        if pool_bytes <= 0.0 {
            return Ok(Some(0));
        }
        let t = div(pool_bytes, self.per_layer * seqs as f64)?;
        let window = self.window.filter(|w| *w != 0);
        let Some(window) = window.filter(|_| self.n_sl != 0) else {
            return if self.n_full != 0 {
                Ok(Some(int_f(div(t, self.n_full as f64)?)?))
            } else {
                Ok(None)
            };
        };
        let all = self.n_full + self.n_sl;
        if t <= (i128::from(all) * i128::from(window)) as f64 {
            return Ok(Some(int_f(div(t, all as f64)?)?));
        }
        if self.n_full != 0 {
            let sl = (i128::from(self.n_sl) * i128::from(window)) as f64;
            Ok(Some(int_f(div(t - sl, self.n_full as f64)?)?))
        } else {
            Ok(None)
        }
    }
}

/// Build the KV sizing. Returns (kv|None, notes, lowered) where lowered means confidence -> low.
fn kv_model(
    model: &ModelInfo,
    tp: i64,
    kv_bytes: f64,
) -> Result<(Option<Kv>, Vec<String>, bool), Problem> {
    let mut notes = Vec::new();
    let Some(layers) = model.num_layers.filter(|l| *l != 0) else {
        return Ok((
            None,
            vec!["num_layers missing from config: KV cache cannot be computed".into()],
            true,
        ));
    };
    let _ = layers;
    let (n_full, n_sl, window, wnote) = layer_split(model);
    if let Some(n) = wnote {
        notes.push(n);
    }
    if let Some(rank) = model.kv_lora_rank.filter(|r| *r != 0) {
        let rope = match cfg_of(model).get("qk_rope_head_dim") {
            None => DEFAULT_QK_ROPE,
            Some(v) => py_int_value(v)?,
        };
        #[allow(clippy::cast_precision_loss)]
        let per_layer = (i128::from(rank) + i128::from(rope)) as f64 * kv_bytes;
        notes.push("MLA latent KV cache is replicated on every TP rank".into());
        return Ok((
            Some(Kv {
                per_layer,
                n_full,
                n_sl,
                window,
            }),
            notes,
            false,
        ));
    }
    let heads = model.num_attention_heads;
    let mut lowered = false;
    let kv_heads = match model.num_kv_heads {
        Some(k) => k,
        None => {
            let Some(h) = heads.filter(|h| *h != 0) else {
                notes.push("attention head counts missing: KV cache cannot be computed".into());
                return Ok((None, notes, true));
            };
            lowered = true;
            notes.push("num_key_value_heads missing: assumed MHA (upper bound on KV size)".into());
            h
        }
    };
    let head_dim = model.head_dim.filter(|d| *d != 0).or_else(|| {
        match (
            model.hidden_size.filter(|x| *x != 0),
            heads.filter(|x| *x != 0),
        ) {
            (Some(hs), Some(h)) => py::floordiv(hs, h),
            _ => None,
        }
    });
    let Some(head_dim) = head_dim.filter(|d| *d != 0) else {
        notes.push("head_dim unknown: KV cache cannot be computed".into());
        return Ok((None, notes, true));
    };
    #[allow(clippy::cast_precision_loss)]
    let per_gpu_heads = ((kv_heads as f64 / tp.max(1) as f64).ceil()).max(1.0);
    if tp > kv_heads {
        notes.push(format!(
            "tp={tp} exceeds {kv_heads} KV heads: KV heads are replicated across ranks"
        ));
    }
    // 2 * per_gpu_heads * head_dim (exact ints in Python) * kv_bytes
    #[allow(clippy::cast_precision_loss)]
    let ints = 2.0 * per_gpu_heads * head_dim as f64;
    Ok((
        Some(Kv {
            per_layer: ints * kv_bytes,
            n_full,
            n_sl,
            window,
        }),
        notes,
        lowered,
    ))
}

/// `int(v)` of a JSON value (int, float truncated, integer string); anything else raises in Python.
fn py_int_value(v: &Value) -> Result<i64, Problem> {
    match v {
        Value::Bool(b) => Ok(i64::from(*b)),
        Value::Number(n) => match n.as_i64() {
            Some(i) if !n.is_f64() => Ok(i),
            _ => int_f(n.as_f64().ok_or_else(bad)?),
        },
        Value::String(s) => crate::services::store::py_int(s).ok_or_else(bad),
        _ => Err(bad()),
    }
}

fn activations_gib(model: &ModelInfo, mem: &Mem, tp: i64) -> Result<(f64, bool), Problem> {
    if let Some(a) = mem.get("activations_gib") {
        return Ok((a, false));
    }
    let Some(hidden) = model.hidden_size.filter(|h| *h != 0) else {
        return Ok((ACT_DEFAULT_GIB, true));
    };
    let tokens = mem
        .get("max_batched_tokens")
        .unwrap_or(DEFAULT_BATCHED_TOKENS);
    #[allow(clippy::cast_precision_loss)]
    let x = div(
        div(tokens * hidden as f64 * 2.0 * ACT_FACTOR, tp as f64)?,
        GIB,
    )?;
    // max(ACT_MIN_GIB, x): the first argument stays unless x is greater
    Ok((if x > ACT_MIN_GIB { x } else { ACT_MIN_GIB }, false))
}

/// `_verdict(total, budget)`
pub fn verdict_of(total: f64, budget: f64) -> &'static str {
    if budget <= 0.0 {
        return "wont_fit";
    }
    let ratio = total / budget;
    if ratio <= FITS_MAX {
        "fits"
    } else if ratio <= TIGHT_MAX {
        "tight"
    } else {
        "wont_fit"
    }
}

const CONF: [&str; 3] = ["high", "medium", "low"];

fn rank(v: &str) -> u8 {
    match v {
        "fits" => 0,
        "tight" => 1,
        "unknown" => 2,
        _ => 3,
    }
}

fn worst(verdicts: &[&'static str]) -> &'static str {
    let mut best: Option<&'static str> = None;
    for v in verdicts {
        if best.is_none_or(|b| rank(v) > rank(b)) {
            best = Some(v);
        }
    }
    best.unwrap_or("unknown")
}

/// Share (0..1) of the weights or of the KV cache that an engine keeps on the GPU (1.0 when not reported).
pub fn gpu_fraction(mem: &Mem, key: &str) -> f64 {
    match mem.get(key) {
        Some(f) if f.is_finite() => f.clamp(0.0, 1.0),
        _ => 1.0,
    }
}

#[allow(clippy::too_many_arguments)]
fn tp_required(
    model: &ModelInfo,
    mem: &Mem,
    hw: &Hardware,
    weights: Option<f64>,
    max_len: Option<i64>,
    concurrency: i64,
    kv_bytes: f64,
    fraction: f64,
) -> Result<Option<i64>, Problem> {
    let Some(weights) = weights else {
        return Ok(None);
    };
    if hw.gpu_total_gib.is_empty() {
        return Ok(None);
    }
    let min_total = hw
        .gpu_total_gib
        .iter()
        .copied()
        .fold(f64::INFINITY, |a, b| if b < a { b } else { a });
    let budget = min_total * fraction;
    let (w_frac, kv_frac) = (
        gpu_fraction(mem, "weights_gpu_fraction"),
        gpu_fraction(mem, "kv_gpu_fraction"),
    );
    for tp in [1i64, 2, 4, 8] {
        if usize::try_from(tp).unwrap_or(usize::MAX) > hw.gpu_ids.len() {
            break;
        }
        if let Some(h) = model.num_attention_heads.filter(|h| *h != 0)
            && h % tp != 0
        {
            continue;
        }
        let (act, _) = activations_gib(model, mem, tp)?;
        #[allow(clippy::cast_precision_loss)]
        let fixed = weights * w_frac / tp as f64 / GIB
            + act
            + mem.get("cuda_graph_gib").unwrap_or(DEFAULT_GRAPH_GIB)
            + mem.get("overhead_gib").unwrap_or(DEFAULT_OVERHEAD_GIB);
        let mut kv_gib = 0.0;
        if let Some(ml) = max_len.filter(|m| *m != 0) {
            let (kv, _, _) = kv_model(model, tp, kv_bytes)?;
            if let Some(kv) = kv {
                kv_gib = kv.bytes_for(ml, concurrency) * kv_frac / GIB;
            }
        }
        if fixed + kv_gib <= budget {
            return Ok(Some(tp));
        }
    }
    Ok(None)
}

/// `estimate_fit(model, mem, hw, concurrency=, resident=, compat=, hw_all=)`.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
pub fn estimate_fit(
    model: &ModelInfo,
    mem: &Mem,
    hw: &Hardware,
    concurrency: i64,
    resident: &[ResidentUse],
    compat: &[Compat],
    hw_all: Option<&Hardware>,
) -> Result<FitReport, Problem> {
    let mut notes: Vec<String> = Vec::new();
    let mut conf_rank = 0usize;
    let lower = |r: &mut usize, to: usize| *r = (*r).max(to);

    let n_gpu = i64::try_from(hw.gpu_ids.len()).unwrap_or(i64::MAX);
    let tp = match mem.truthy("tp") {
        Some(t) => int_f(t)?,
        None => {
            if n_gpu != 0 {
                n_gpu
            } else {
                1
            }
        }
    }
    .max(1);
    let fraction = mem.get("mem_fraction").unwrap_or(DEFAULT_MEM_FRACTION);
    if !mem.has("mem_fraction") {
        notes.push(format!(
            "engine gave no mem_fraction: assumed {}",
            crate::pyjson::float_repr(DEFAULT_MEM_FRACTION)
        ));
    }
    let kv_bytes = mem.get("kv_bytes_per_elem").unwrap_or(2.0);

    let (weights_total, wcaveat) = weights_bytes(model);
    if let Some(c) = wcaveat {
        notes.push(c);
        lower(&mut conf_rank, 1);
    }

    let mut max_len = match mem.truthy("max_len") {
        Some(m) => Some(int_f(m)?),
        None => None,
    };
    let native = model.max_position_embeddings.filter(|m| *m != 0);
    if max_len.is_none()
        && let Some(n) = native
    {
        max_len = Some(n);
        notes.push(format!("max_len not set: using the model's native {n}"));
        lower(&mut conf_rank, 1);
    }
    if let (Some(ml), Some(n)) = (max_len.filter(|m| *m != 0), native)
        && ml > n
    {
        notes.push(format!(
            "max_len {ml} exceeds native {n} (needs rope scaling)"
        ));
    }

    let compat_list = compat.to_vec();
    if n_gpu == 0 {
        notes.push("no GPU selected or detected".into());
        return Ok(FitReport {
            verdict: "unknown".into(),
            confidence: "low".into(),
            tp,
            tp_required: None,
            concurrency,
            max_len,
            kv_bytes_per_token: None,
            per_gpu: Vec::new(),
            max_context_at_current_concurrency: None,
            max_concurrency_at_current_context: None,
            fits_if_stop: Vec::new(),
            notes,
            compat: compat_list,
            host_ram: None,
        });
    }

    let (mut kv, kv_notes, kv_low) = kv_model(model, tp, kv_bytes)?;
    notes.extend(kv_notes.iter().cloned());
    if kv_low {
        lower(&mut conf_rank, 2);
    }
    if kv.is_some() && kv_notes.iter().any(|n| n.contains("assumed every layer")) {
        lower(&mut conf_rank, 1);
    }
    if kv.is_some() && max_len.is_none() {
        kv = None;
        notes.push("context length unknown: KV cache cannot be sized".into());
        lower(&mut conf_rank, 2);
    }

    let (act_est, act_default) = activations_gib(model, mem, tp)?;
    // SGLang-style engines carve activations out of the non-static share, so the static budget (weights + KV) is
    // not charged for them; the adapter signals that with nonstatic_reserve_frac.
    let charge_act = !mem.has("nonstatic_reserve_frac");
    let act = if charge_act { act_est } else { 0.0 };
    if act_default {
        notes.push("hidden_size missing: activation memory is a default guess".into());
        lower(&mut conf_rank, 1);
    }
    let graphs = mem.get("cuda_graph_gib").unwrap_or(DEFAULT_GRAPH_GIB);
    let overhead = mem.get("overhead_gib").unwrap_or(DEFAULT_OVERHEAD_GIB);
    let outside = mem.get("outside_share_gib").unwrap_or(0.0);
    let of_free = mem.truthy("share_of_free").is_some();
    let claims = mem.truthy("claims_share").is_some();
    let (w_frac, kv_frac) = (
        gpu_fraction(mem, "weights_gpu_fraction"),
        gpu_fraction(mem, "kv_gpu_fraction"),
    );
    if w_frac < 1.0 {
        notes.push(format!(
            "{} of the weights stay in host RAM (partial GPU offload): it fits more, but every token is slower",
            pct0(1.0 - w_frac)
        ));
    }
    if kv_frac < 1.0 {
        notes.push(format!(
            "{} of the KV cache stays in host RAM",
            pct0(1.0 - kv_frac)
        ));
    }
    #[allow(clippy::cast_precision_loss)]
    let weights_gpu = weights_total.map(|w| w * w_frac / tp as f64 / GIB);
    let mut kv_gib = match (&kv, max_len.filter(|m| *m != 0)) {
        (Some(kv), Some(ml)) => Some(kv.bytes_for(ml, concurrency) * kv_frac / GIB),
        _ => None,
    };
    let fixed_kv = mem.truthy("kv_cache_fixed_gib");
    if let Some(f) = fixed_kv {
        kv_gib = Some(f);
        notes.push(format!(
            "KV pool fixed by engine settings at {} GiB per GPU",
            g(f)
        ));
    }

    if tp > n_gpu {
        notes.push(format!("tp={tp} needs {tp} GPUs but only {n_gpu} selected"));
    }

    let used = usize::try_from(tp.min(n_gpu)).unwrap_or(0);
    let mut per_gpu: Vec<GpuFit> = Vec::new();
    let mut verdicts: Vec<&'static str> = Vec::new();
    let mut freeable: BTreeSet<String> = BTreeSet::new();
    let mut known_pool: Vec<f64> = Vec::new();
    if weights_gpu.is_none() {
        notes.push("weights size unknown (no safetensors sizes or parameter count)".into());
        lower(&mut conf_rank, 2);
        verdicts.push("unknown");
    }
    for i in 0..used {
        let gpu_id = hw.gpu_ids[i];
        let total_vram = hw.gpu_total_gib.get(i).copied().ok_or_else(bad)?;
        let free = hw.gpu_free_gib.get(i).copied().ok_or_else(bad)?;
        let uuid = hw.gpu_uuids.get(i).cloned().ok_or_else(bad)?;
        let name = hw.gpu_names.get(i).cloned().ok_or_else(bad)?;
        let budget = (if of_free { free } else { total_vram }) * fraction;
        let here: Vec<&ResidentUse> = resident
            .iter()
            .filter(|r| r.gpu_ids.contains(&gpu_id))
            .collect();
        let here_sum = here.iter().fold(0.0, |a, r| a + r.gib_per_gpu);
        if of_free && free < 0.98 * total_vram {
            notes.push(format!(
                "GPU {gpu_id}: the share is {} of the {free:.1} GiB free at launch, not of the {total_vram:.1} GiB card",
                g(fraction)
            ));
        }
        let Some(wg) = weights_gpu else {
            per_gpu.push(GpuFit {
                gpu_id,
                uuid,
                name,
                weights_gib: 0.0,
                kv_cache_gib: 0.0,
                activations_gib: py_round(act, 3),
                cuda_graphs_gib: graphs,
                overhead_gib: overhead,
                total_gib: 0.0,
                budget_gib: py_round(budget, 3),
                vram_total_gib: total_vram,
                free_gib: free,
                utilization_pct: 0.0,
                verdict: "unknown".into(),
            });
            continue;
        };
        let fixed = wg + act + graphs + overhead;
        let (total, mut v) = match kv_gib {
            None => (
                fixed,
                if fixed > budget {
                    "wont_fit"
                } else {
                    "unknown"
                },
            ),
            Some(k) => {
                let total = fixed + k;
                known_pool.push(budget - fixed);
                (total, verdict_of(total, budget))
            }
        };
        // What must be free to launch: the whole share for an engine that claims it at start, else what this
        // configuration uses; plus what the engine holds outside its share.
        let physical = (if claims {
            budget
        } else {
            total + if charge_act { 0.0 } else { act_est }
        }) + outside;
        let base = if of_free { free } else { total_vram };
        if let Some(reserve) = mem.get("nonstatic_reserve_frac")
            && act_est + graphs + overhead > reserve * base
        {
            notes.push(format!(
                "GPU {gpu_id}: estimated activations ({act_est:.1} GiB) exceed the non-static reserve ({:.1} GiB); \
                 lower the static fraction",
                reserve * base
            ));
        }
        if (v == "fits" || v == "tight") && physical > free {
            v = "wont_fit";
            let why = if claims {
                format!(
                    " (the engine needs its whole {} share free at start)",
                    g(fraction)
                )
            } else {
                String::new()
            };
            if !here.is_empty() && physical <= free + here_sum {
                freeable.extend(here.iter().map(|r| r.name.clone()));
                notes.push(format!(
                    "GPU {gpu_id}: needs {physical:.1} GiB{why} but only {free:.1} GiB is free (other engines are resident)"
                ));
            } else {
                notes.push(format!(
                    "GPU {gpu_id}: needs {physical:.1} GiB{why} but only {free:.1} GiB is free"
                ));
            }
        } else if of_free
            && v == "wont_fit"
            && kv_gib.is_some()
            && !here.is_empty()
            && matches!(
                verdict_of(total, (free + here_sum) * fraction),
                "fits" | "tight"
            )
        {
            // a share of free memory shrinks with every resident engine; stopping them makes it fit
            freeable.extend(here.iter().map(|r| r.name.clone()));
            notes.push(format!(
                "GPU {gpu_id}: {} of the {free:.1} GiB free is only {budget:.1} GiB (other engines are resident)",
                g(fraction)
            ));
        }
        per_gpu.push(GpuFit {
            gpu_id,
            uuid,
            name,
            weights_gib: py_round(wg, 3),
            kv_cache_gib: py_round(kv_gib.unwrap_or(0.0), 3),
            activations_gib: py_round(act, 3),
            cuda_graphs_gib: graphs,
            overhead_gib: overhead,
            total_gib: py_round(total, 3),
            budget_gib: py_round(budget, 3),
            vram_total_gib: total_vram,
            free_gib: free,
            utilization_pct: if budget > 0.0 {
                py_round(total / budget * 100.0, 1)
            } else {
                0.0
            },
            verdict: v.into(),
        });
        verdicts.push(v);
    }

    let mut verdict = worst(&verdicts);
    if tp > n_gpu {
        verdict = "wont_fit";
    }
    let confidence = if kv_gib.is_none() && verdict == "unknown" {
        "low"
    } else if verdict == "wont_fit" && kv_gib.is_none() {
        "medium" // a lower bound already exceeds the budget
    } else {
        CONF[conf_rank.min(2)]
    };

    let mut max_ctx: Option<i64> = None;
    let mut max_conc: Option<i64> = None;
    if let (Some(kv), Some(ml)) = (&kv, max_len.filter(|m| *m != 0))
        && !known_pool.is_empty()
        && kv_frac > 0.0
    {
        // The GPU pool only has to hold the GPU-resident share of the cache, so it stretches by 1 / kv_frac.
        let base = match fixed_kv {
            Some(f) => f,
            None => known_pool
                .iter()
                .copied()
                .fold(f64::INFINITY, |a, b| if b < a { b } else { a }),
        };
        let pool = div(base * GIB, if fixed_kv.is_some() { 1.0 } else { kv_frac })?;
        max_ctx = kv.max_ctx(pool, concurrency)?;
        if let (Some(mc), Some(n)) = (max_ctx, native) {
            max_ctx = Some(mc.min(n));
        }
        let one = kv.bytes_for(ml, 1);
        max_conc = if one > 0.0 {
            Some(int_f(floordiv_f(pool, one)?)?.max(0))
        } else {
            None
        };
    }

    let tpr = tp_required(
        model,
        mem,
        hw_all.unwrap_or(hw),
        weights_total,
        max_len,
        concurrency,
        kv_bytes,
        fraction,
    )?;
    if compat_list.iter().any(|c| c.level == "block") && (verdict == "fits" || verdict == "tight") {
        notes.push("engine compatibility rules block this configuration".into());
    }

    Ok(FitReport {
        verdict: verdict.into(),
        confidence: confidence.into(),
        tp,
        tp_required: tpr,
        concurrency,
        max_len,
        kv_bytes_per_token: kv.as_ref().map(Kv::bytes_per_token_full),
        per_gpu,
        max_context_at_current_concurrency: max_ctx,
        max_concurrency_at_current_context: max_conc,
        fits_if_stop: freeable.into_iter().collect(),
        notes,
        compat: compat_list,
        host_ram: None,
    })
}

/// Host-RAM verdict from an adapter's `host_memory_gib()` plus a small engine-process baseline. Only unreclaimable
/// memory counts against MemAvailable; a NaN component means the adapter could not compute it: verdict `unknown`.
pub fn estimate_host_ram(breakdown: &FloatMap, host: Option<&HostMemInfo>) -> HostRamFit {
    let dict = Mem::from_map(breakdown);
    let mut unknown: Vec<String> = dict
        .0
        .iter()
        .filter(|(_, v)| v.is_nan())
        .map(|(k, _)| k.clone())
        .collect();
    unknown.sort();
    let mut known: Vec<(String, f64)> = dict
        .0
        .iter()
        .filter(|(_, v)| !v.is_nan())
        .map(|(k, v)| (k.clone(), py_round(*v, 3)))
        .collect();
    match known.iter_mut().find(|(k, _)| k == "engine_process") {
        Some(slot) => slot.1 = ENGINE_PROCESS_GIB,
        None => known.push(("engine_process".into(), ENGINE_PROCESS_GIB)),
    }
    let needed = py_round(known.iter().fold(0.0, |a, (_, v)| a + v), 3);
    let mut bd = Map::new();
    for (k, v) in &known {
        bd.insert(k.clone(), Value::from(*v));
    }
    let mut notes = vec![
        "Weight loading streams through the page cache; only the unreclaimable part (anon, shared memory, pinned \
         buffers) counts against available RAM, so the model's file size is not added here."
            .to_string(),
        format!(
            "engine_process is a {} GiB heuristic (python + CUDA runtime), not a measurement.",
            g(ENGINE_PROCESS_GIB)
        ),
    ];
    if !unknown.is_empty() {
        notes.push(format!(
            "could not be computed (model or hardware facts missing): {}; needed is a lower bound",
            unknown.join(", ")
        ));
    }
    let Some(host) = host else {
        notes.push("host memory could not be read".into());
        return HostRamFit {
            needed_gib: needed,
            available_gib: None,
            total_gib: None,
            verdict: "unknown".into(),
            breakdown: bd,
            notes,
        };
    };
    let verdict = if unknown.is_empty() {
        let ratio = if host.available_gib > 0.0 {
            needed / host.available_gib
        } else {
            f64::INFINITY
        };
        if ratio <= HOST_OK {
            "ok"
        } else if ratio <= 1.0 {
            "tight"
        } else {
            "wont_fit"
        }
    } else {
        "unknown"
    };
    if host.shmem_gib > 0.25 * host.total_gib {
        notes.push(format!(
            "{:.1} GiB of host RAM is already shared memory and cannot be reclaimed",
            host.shmem_gib
        ));
    }
    HostRamFit {
        needed_gib: needed,
        available_gib: Some(py_round(host.available_gib, 3)),
        total_gib: Some(py_round(host.total_gib, 3)),
        verdict: verdict.into(),
        breakdown: bd,
        notes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_python_number_helpers() {
        assert_eq!(thousands(32_000_000_000), "32,000,000,000");
        assert_eq!(thousands(999), "999");
        assert_eq!(thousands(-1234), "-1,234");
        assert_eq!(pct0(0.25), "25%");
        assert_eq!(floordiv_f(7.0, 2.0).ok(), Some(3.0));
        assert_eq!(floordiv_f(-7.0, 2.0).ok(), Some(-4.0));
        assert!(div(1.0, 0.0).is_err());
        assert!(int_f(f64::NAN).is_err());
    }
}
