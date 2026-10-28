//! Python: backend/src/engine_console/adapters/llamacpp_parse.py.
//!
//! Pure parsers for llama-server Prometheus text and log lines (no I/O, no engine imports, no state). `/metrics`
//! (switched on by `--metrics`) carries no labels; this build exports no KV-cache metric (older builds'
//! `kv_cache_usage_ratio` is read when present). Log patterns match message text only and ignore the
//! `0.00.007.493 I srv    load_model:` prefix.

use std::sync::LazyLock;

use regex::Regex;
use serde_json::{Map, Value};

use super::base::FloatMap;
use super::trtllm_parse::obj;
use super::trtllm_parse::py::{self, Int, PyResult};

pub const PREFIX: &str = "llamacpp:";

fn re(p: &str) -> Regex {
    py::pyre(p)
}

static LINE: LazyLock<Regex> =
    LazyLock::new(|| re(r"^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{(.*)\})?\s+(\S+)(?:\s+\S+)?\s*$"));

/// `parse_samples`: (name without the prefix, value) per sample line; comments, labels and malformed lines are
/// dropped, NaN values are kept out.
pub fn parse_samples(text: &str) -> Vec<(String, f64)> {
    let mut out = Vec::new();
    for raw in crate::services::common::py_splitlines(text) {
        let line = py::strip(raw);
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some(m) = LINE.captures(line) else {
            continue;
        };
        let name = m.get(1).map_or("", |x| x.as_str());
        let Some(rest) = name.strip_prefix(PREFIX) else {
            continue;
        };
        let Some(value) = py::float(m.get(3).map_or("", |x| x.as_str())) else {
            continue;
        };
        if !value.is_nan() {
            out.push((rest.to_string(), value));
        }
    }
    out
}

/// `scrape_to_canonical`: the canonical keys llama-server can supply; absent keys are omitted.
pub fn scrape_to_canonical(text: &str) -> FloatMap {
    // more than one series (router mode, forks): add them up
    let mut sums: Vec<(String, f64)> = Vec::new();
    for (name, value) in parse_samples(text) {
        if let Some(slot) = sums.iter_mut().find(|(n, _)| *n == name) {
            slot.1 += value;
        } else {
            sums.push((name, value));
        }
    }
    let get = |n: &str| -> Option<f64> {
        // counters carry `_total` on the wire; a build that drops (or adds) it is read the same way
        let alt = match n.strip_suffix("_total") {
            Some(base) => base.to_string(),
            None => format!("{n}_total"),
        };
        for cand in [n, alt.as_str()] {
            if let Some((_, v)) = sums.iter().find(|(k, _)| k == cand) {
                return Some(*v);
            }
        }
        None
    };
    let mut out = FloatMap::new();
    let mut put = |key: &str, val: Option<f64>, scale: f64| {
        if let Some(v) = val {
            out.push((key.to_string(), v * scale));
        }
    };
    put("requests_running", get("requests_processing"), 1.0);
    put("requests_waiting", get("requests_deferred"), 1.0);
    put("prompt_tokens_total", get("prompt_tokens_total"), 1.0);
    put(
        "generation_tokens_total",
        get("tokens_predicted_total"),
        1.0,
    );
    put("prompt_tps", get("prompt_tokens_seconds"), 1.0);
    put("generation_tps", get("predicted_tokens_seconds"), 1.0);
    put("kv_cache_usage_pct", get("kv_cache_usage_ratio"), 100.0); // not exported by the pinned build
    out
}

// ------------------------------------------------------------------ log lines
macro_rules! rx {
    ($name:ident, $p:expr) => {
        static $name: LazyLock<Regex> = LazyLock::new(|| re($p));
    };
}
rx!(ANSI, r"\x1b\[[0-9;]*m");
rx!(LOG_PREFIX, r"^\s*\d+\.\d{2}\.\d{3}\.\d{3}\s+[A-Z]\s+");
rx!(DOTS, r"^\.{10,}$");
rx!(LOADING, r"\bloading model '");
rx!(
    META,
    r"loaded meta data with (\d+) key-value pairs and (\d+) tensors from (\S+)"
);
rx!(OFFLOADED, r"offloaded (\d+)/(\d+) layers to GPU");
rx!(MODEL_BUF, r"(\S+)\s+model buffer size\s*=\s*([\d.]+) MiB");
rx!(
    KV_TOTAL,
    r"size\s*=\s*([\d.]+) MiB \(\s*(\d+) cells,\s*(\d+) layers"
);
rx!(KV_BUF, r"(\S+)\s+KV buffer size\s*=\s*([\d.]+) MiB");
rx!(N_CTX, r"\bn_ctx\s+=\s+(\d+)");
rx!(N_CTX_TRAIN, r"\bn_ctx_train\s+=\s+(\d+)");
rx!(
    SLOTS,
    r"initializing(?: slots)?, n_slots = (\d+)(?:, n_ctx_slot = (\d+))?"
);
rx!(
    AUTO_PARALLEL,
    r"n_parallel is set to auto, using n_parallel = (\d+) and kv_unified = (true|false)"
);
rx!(FIT_CTX, r"context size reduced from (\d+) to (\d+)");
rx!(
    FIT_PROJECTED,
    r"projected to use (\d+) MiB of device memory vs\. (\d+) MiB of free device memory"
);
rx!(
    FIT_FAILED,
    r"(?:failed to fit params to free device memory|encountered an error while trying to fit params to free device memory): (.*)"
);
rx!(CUDA_DEVICES, r"found (\d+) CUDA devices");
rx!(
    CUDA_DEVICE,
    r"Device (\d+): (.+?), compute capability (\d+\.\d+), VMM: (\w+), VRAM: (\d+) MiB"
);
rx!(BUILD, r"build (\d+) \(([0-9a-f]+)\) with ");
rx!(LISTENING, r"\blistening on (\S+)");
rx!(READY, r"(?:^|\s)model loaded\s*$|all slots are idle");
rx!(ALLOC_CUDA, r"failed to allocate CUDA\d* buffer");

/// (needle, error code), most specific first: the first needle found in a line decides.
const ERRORS: [(&str, &str); 12] = [
    ("failed to open GGUF file", "model_file_missing"),
    ("cudaMalloc failed", "cuda_oom"),
    ("out of memory", "cuda_oom"),
    (
        "V cache quantization requires flash_attn",
        "kv_quant_needs_flash_attn",
    ),
    (
        "SPLIT_MODE_TENSOR requires flash_attn",
        "split_tensor_needs_flash_attn",
    ),
    ("couldn't bind HTTP server socket", "port_in_use"),
    ("unsupported model architecture", "unsupported_architecture"),
    ("failed to create llama_context", "context_create_failed"),
    ("exiting due to HTTP server error", "http_server_failed"),
    ("failed to initialize HTTP server", "http_server_failed"),
    ("failed to load model", "model_load_failed"),
    ("exiting due to model loading error", "model_load_failed"),
];

fn g(m: &regex::Captures<'_>, i: usize) -> String {
    m.get(i)
        .map_or_else(String::new, |x| x.as_str().to_string())
}

fn int(m: &regex::Captures<'_>, i: usize) -> PyResult<Value> {
    py::int_of(&g(m, i))
}

/// `round(float(group) / 1024, 3)` (ValueError on a malformed number such as `1.2.3`, as in Python).
fn mib_to_gib(m: &regex::Captures<'_>, i: usize) -> PyResult<Value> {
    let f = py::float_of_value(&Value::from(g(m, i)))?;
    Ok(Value::from(py::round_to(f / 1024.0, 3)))
}

/// `parse_log_line`: structured progress for one llama-server log line, or None. A line yields the first thing it
/// matches, errors first.
pub fn parse_log_line(line: &str) -> PyResult<Option<Map<String, Value>>> {
    let no_ansi = ANSI.replace_all(line, "");
    let no_prefix = LOG_PREFIX.replace(&no_ansi, "");
    let text = py::strip(&no_prefix);
    if text.is_empty() {
        return Ok(None);
    }
    let message = || Value::from(py::tail(text, 300));
    for (needle, code) in ERRORS {
        if text.contains(needle) {
            return Ok(Some(obj(vec![
                ("error", code.into()),
                ("message", message()),
            ])));
        }
    }
    if ALLOC_CUDA.is_match(text) {
        return Ok(Some(obj(vec![
            ("error", "cuda_oom".into()),
            ("message", message()),
        ])));
    }
    if DOTS.is_match(text) {
        #[allow(clippy::cast_precision_loss)]
        let pct = text.chars().count().min(100) as f64;
        return Ok(Some(obj(vec![
            ("phase", "loading_weights".into()),
            ("pct", pct.into()),
        ])));
    }
    if LOADING.is_match(text) {
        return Ok(Some(obj(vec![
            ("phase", "loading_weights".into()),
            ("pct", 0.0.into()),
        ])));
    }
    if let Some(m) = META.captures(text) {
        return Ok(Some(obj(vec![
            ("phase", "loading_weights".into()),
            ("pct", 0.0.into()),
            ("tensors", int(&m, 2)?),
        ])));
    }
    if text.contains("loading model tensors") {
        return Ok(Some(obj(vec![
            ("phase", "loading_weights".into()),
            ("pct", 0.0.into()),
        ])));
    }
    if let Some(m) = OFFLOADED.captures(text) {
        let mut out = obj(vec![
            ("phase", "loading_weights".into()),
            ("gpu_layers", int(&m, 1)?),
            ("total_layers", int(&m, 2)?),
        ]);
        if py::int_digits(&g(&m, 1))? == "0" {
            // the GPU was not used at all: no device visible, or --n-gpu-layers 0
            out.insert("warning".into(), "cpu_only".into());
        }
        return Ok(Some(out));
    }
    if let Some(m) = MODEL_BUF.captures(text) {
        return Ok(Some(obj(vec![
            ("buffer", g(&m, 1).into()),
            ("model_buffer_gib", mib_to_gib(&m, 2)?),
        ])));
    }
    if let Some(m) = KV_TOTAL.captures(text) {
        return Ok(Some(obj(vec![
            ("kv_cache_gib", mib_to_gib(&m, 1)?),
            ("kv_cache_tokens", int(&m, 2)?),
        ])));
    }
    if let Some(m) = KV_BUF.captures(text) {
        return Ok(Some(obj(vec![(
            "kv_buffer",
            Value::Object(obj(vec![
                ("device", g(&m, 1).into()),
                ("gib", mib_to_gib(&m, 2)?),
            ])),
        )])));
    }
    if let Some(m) = AUTO_PARALLEL.captures(text) {
        return Ok(Some(obj(vec![
            ("n_slots", int(&m, 1)?),
            ("kv_unified", (g(&m, 2) == "true").into()),
        ])));
    }
    if let Some(m) = SLOTS.captures(text) {
        let mut slots = obj(vec![("n_slots", int(&m, 1)?)]);
        if !g(&m, 2).is_empty() {
            slots.insert("n_ctx_slot".into(), int(&m, 2)?);
        }
        return Ok(Some(slots));
    }
    if let Some(m) = N_CTX.captures(text) {
        return Ok(Some(obj(vec![("n_ctx", int(&m, 1)?)])));
    }
    if let Some(m) = N_CTX_TRAIN.captures(text) {
        return Ok(Some(obj(vec![("n_ctx_train", int(&m, 1)?)])));
    }
    if text.contains("warming up the model") {
        return Ok(Some(obj(vec![("phase", "warming_up".into())])));
    }
    if let Some(m) = FIT_CTX.captures(text) {
        return Ok(Some(obj(vec![(
            "fit_ctx_reduced",
            Value::Object(obj(vec![("from", int(&m, 1)?), ("to", int(&m, 2)?)])),
        )])));
    }
    if let Some(m) = FIT_PROJECTED.captures(text) {
        return Ok(Some(obj(vec![
            ("fit_projected_mib", int(&m, 1)?),
            ("fit_free_mib", int(&m, 2)?),
        ])));
    }
    if let Some(m) = FIT_FAILED.captures(text) {
        return Ok(Some(obj(vec![
            ("warning", "fit_failed".into()),
            ("message", py::tail(&g(&m, 1), 300).into()),
        ])));
    }
    if let Some(m) = CUDA_DEVICE.captures(text) {
        // round(int(group) / 1024, 2): int / int true division
        let vram = Int::parse_digits(&py::int_digits(&g(&m, 5))?)
            .and_then(|v| v.truediv(1024))
            .unwrap_or(f64::INFINITY);
        return Ok(Some(obj(vec![(
            "gpu",
            Value::Object(obj(vec![
                ("index", int(&m, 1)?),
                ("name", g(&m, 2).into()),
                ("compute_capability", g(&m, 3).into()),
                ("vram_gib", py::round_to(vram, 2).into()),
            ])),
        )])));
    }
    if let Some(m) = CUDA_DEVICES.captures(text) {
        return Ok(Some(obj(vec![("gpus_found", int(&m, 1)?)])));
    }
    if text.contains("failed to mlock") {
        return Ok(Some(obj(vec![
            ("warning", "mlock_failed".into()),
            ("message", message()),
        ])));
    }
    if text.contains("entering sleeping state") {
        return Ok(Some(obj(vec![("state", "sleeping".into())])));
    }
    if text.contains("exiting sleeping state") {
        return Ok(Some(obj(vec![("state", "awake".into())])));
    }
    if let Some(m) = BUILD.captures(text) {
        return Ok(Some(obj(vec![
            ("build", g(&m, 1).into()),
            ("commit", g(&m, 2).into()),
        ])));
    }
    if READY.is_match(text) {
        return Ok(Some(obj(vec![
            ("phase", "ready".into()),
            ("pct", 100.0.into()),
        ])));
    }
    if let Some(m) = LISTENING.captures(text) {
        return Ok(Some(obj(vec![("listening", g(&m, 1).into())])));
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samples_and_canonical_keys() {
        let s = parse_samples(
            "# HELP x\nllamacpp:a 1.5\nbad line\nllamacpp:b 2 1712345\nllamacpp:c NaN\n",
        );
        assert_eq!(s, vec![("a".to_string(), 1.5), ("b".to_string(), 2.0)]);
        let m = scrape_to_canonical(
            "llamacpp:kv_cache_usage_ratio 0.25\nllamacpp:tokens_predicted 10\n",
        );
        assert_eq!(
            m,
            vec![
                ("generation_tokens_total".to_string(), 10.0),
                ("kv_cache_usage_pct".to_string(), 25.0)
            ]
        );
    }

    #[test]
    fn log_lines() {
        assert_eq!(
            parse_log_line("main: model loaded").map(|o| o.map(Value::Object)),
            Ok(Some(serde_json::json!({"phase": "ready", "pct": 100.0})))
        );
        assert_eq!(parse_log_line("   "), Ok(None));
        assert_eq!(
            parse_log_line("CUDA0 model buffer size = 1.2.3 MiB").map_err(|e| e.kind),
            Err("ValueError".to_string())
        );
    }
}
