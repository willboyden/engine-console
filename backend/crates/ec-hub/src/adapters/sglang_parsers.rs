//! Port of backend/src/engine_console/adapters/sglang_parsers.py: pure parsers for
//! SGLang output (Prometheus text -> canonical metrics, startup log lines).
//!
//! Metric names come from `sglang/srt/observability/metrics_collector.py` of the pinned image. Names are normalised
//! (prefix `sglang:`/`sglang_`, `_total` suffix stripped) so a rename between those spellings is tolerated; unknown
//! metrics are ignored. Histogram buckets are lifetime totals, so the quantiles are lifetime figures.
//!
//! Log patterns come from the f-strings in the pinned image's source; no line was captured from a live 0.5.20 engine.
//! Where Python raises while converting a captured number (`[\d.]+` matching `1.2.3`), this returns None.

use std::collections::HashMap;
use std::sync::LazyLock;

use regex::Regex;
use serde_json::Value;

use super::base::{FloatMap, Obj};
use super::pycompat::{self as py, Anchor, FloatDict, compile};
use super::vllm_parse::{Raised, float_v, int_v, obj};

fn search(pat: &str) -> Regex {
    compile(pat, false, Anchor::Search)
}

static SAMPLE: LazyLock<Regex> = LazyLock::new(|| {
    compile(
        r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+(\S+)(?:\s+-?\d+)?\s*$",
        false,
        Anchor::Start,
    )
});
static LE: LazyLock<Regex> = LazyLock::new(|| search(r#"(?:^|,)\s*le="([^"]*)""#));
static STEM_PREFIX: LazyLock<Regex> = LazyLock::new(|| search(r"^sglang[:_]"));

/// `_stem`: drop the `sglang:`/`sglang_` prefix and a `_total` suffix.
pub fn stem(name: &str) -> String {
    let n = STEM_PREFIX.replace(name, "");
    n.strip_suffix("_total").unwrap_or(&n).to_string()
}

/// `_num`: `float(s)`, None for an invalid text or NaN.
pub fn num(s: &str) -> Option<f64> {
    py::float_str(s).filter(|v| !v.is_nan())
}

/// Prometheus-style quantile from cumulative (le, count) pairs, linear interpolation in-bucket.
pub fn histogram_quantile(q: f64, buckets: &[(f64, f64)]) -> Option<f64> {
    let b = py::sorted_by_float_key(buckets);
    let &(_, total) = b.last()?;
    if total <= 0.0 {
        return None;
    }
    let target = q * total;
    let (mut prev_le, mut prev_c) = (0.0_f64, 0.0_f64);
    for &(le, c) in &b {
        if c >= target {
            if le.is_infinite() {
                return if prev_le > 0.0 { Some(prev_le) } else { None };
            }
            #[allow(clippy::float_cmp)]
            if c == prev_c {
                return Some(le);
            }
            return Some(prev_le + (le - prev_le) * (target - prev_c) / (c - prev_c));
        }
        prev_le = le;
        prev_c = c;
    }
    None
}

/// Scalar samples by stem (in first-seen order) and histogram cumulative buckets by stem (summed over label sets).
#[derive(Debug, Default, Clone)]
pub struct Parsed {
    pub scalars: Vec<(String, Vec<f64>)>,
    pub hists: Vec<(String, FloatDict)>,
}

impl Parsed {
    fn scalar(&self, name: &str) -> Option<&[f64]> {
        self.scalars
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_slice())
    }
}

/// `parse_prometheus`: (scalar samples by stem, histogram cumulative buckets by stem summed over label sets).
pub fn parse_prometheus(text: &str) -> Parsed {
    let mut out = Parsed::default();
    let mut sidx: HashMap<String, usize> = HashMap::new();
    let mut hidx: HashMap<String, usize> = HashMap::new();
    for raw in crate::services::common::py_splitlines(text) {
        let line = py::strip(raw);
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some(m) = SAMPLE.captures(line) else {
            continue;
        };
        let name = &m[1];
        let labels = m.get(2).map_or("", |g| g.as_str());
        let Some(val) = num(&m[3]) else {
            continue;
        };
        if let Some(base) = name.strip_suffix("_bucket") {
            if let Some(le) = LE.captures(labels) {
                let le_v = if &le[1] == "+Inf" || &le[1] == "Inf" {
                    Some(f64::INFINITY)
                } else {
                    num(&le[1])
                };
                if let Some(le_v) = le_v {
                    let key = stem(base);
                    let i = *hidx.entry(key.clone()).or_insert_with(|| {
                        out.hists.push((key, FloatDict::default()));
                        out.hists.len() - 1
                    });
                    out.hists[i].1.add(le_v, val);
                }
            }
            continue;
        }
        if name.ends_with("_sum") || name.ends_with("_count") || name.ends_with("_created") {
            continue;
        }
        let key = stem(name);
        let i = *sidx.entry(key.clone()).or_insert_with(|| {
            out.scalars.push((key, Vec::new()));
            out.scalars.len() - 1
        });
        out.scalars[i].1.push(val);
    }
    out
}

fn first<'a>(p: &'a Parsed, names: &[&str]) -> Option<&'a [f64]> {
    names.iter().find_map(|n| p.scalar(n))
}

/// Prometheus text -> canonical metrics.
pub fn parse_metrics(text: &str) -> FloatMap {
    let parsed = parse_prometheus(text);
    let mut out: FloatMap = Vec::new();
    fn set(out: &mut FloatMap, key: &str, v: f64) {
        if let Some(slot) = out.iter_mut().find(|(k, _)| k == key) {
            slot.1 = v;
        } else {
            out.push((key.to_string(), v));
        }
    }
    let total = |out: &mut FloatMap, key: &str, names: &[&str]| {
        if let Some(v) = first(&parsed, names).filter(|v| !v.is_empty()) {
            set(out, key, py::sum(v));
        }
    };
    let mean_pct = |out: &mut FloatMap, key: &str, names: &[&str]| {
        if let Some(v) = first(&parsed, names).filter(|v| !v.is_empty()) {
            let m = py::sum(v) / v.len() as f64;
            // sglang gauges are 0-1 fractions
            set(out, key, py::round(if m <= 1.0 { m * 100.0 } else { m }, 4));
        }
    };
    // By default only the scheduler of tensor-parallel rank 0 reports these gauges, so summing over label sets adds
    // data-parallel groups, not duplicates.
    total(&mut out, "requests_running", &["num_running_reqs"]);
    total(&mut out, "requests_waiting", &["num_queue_reqs"]);
    mean_pct(
        &mut out,
        "kv_cache_usage_pct",
        &["token_usage", "full_token_usage"],
    );
    if !out.iter().any(|(k, _)| k == "kv_cache_usage_pct") {
        // Under tensor parallelism every rank reports max_total_num_tokens but only rank 0 kv_used_tokens: compare means.
        let (used, cap) = (
            first(&parsed, &["kv_used_tokens"]),
            first(&parsed, &["max_total_num_tokens"]),
        );
        if let (Some(used), Some(cap)) = (
            used.filter(|v| !v.is_empty()),
            cap.filter(|v| !v.is_empty()),
        ) && py::sum(cap) > 0.0
        {
            let u = py::sum(used) / used.len() as f64;
            let c = py::sum(cap) / cap.len() as f64;
            set(&mut out, "kv_cache_usage_pct", py::round(u / c * 100.0, 4));
        }
    }
    mean_pct(&mut out, "prefix_cache_hit_pct", &["cache_hit_rate"]);
    total(&mut out, "prompt_tokens_total", &["prompt_tokens"]);
    total(&mut out, "generation_tokens_total", &["generation_tokens"]);
    total(&mut out, "generation_tps", &["gen_throughput"]);
    // `num_retracted_requests_total` is the cumulative counter; `num_retracted_reqs` is a per-interval gauge and is
    // deliberately not used (a gauge would break the "total" contract of this key).
    total(&mut out, "preemptions_total", &["num_retracted_requests"]);
    mean_pct(&mut out, "spec_decode_accept_pct", &["spec_accept_rate"]);
    if let Some(v) = first(&parsed, &["spec_accept_length"]).filter(|v| !v.is_empty()) {
        // extra (non-canonical) key: mean accepted tokens per verify step
        set(&mut out, "spec_accept_length", py::sum(v) / v.len() as f64);
    }
    for (key, hist, q) in [
        ("ttft_p50_s", "time_to_first_token_seconds", 0.5),
        ("ttft_p95_s", "time_to_first_token_seconds", 0.95),
        ("itl_p50_s", "inter_token_latency_seconds", 0.5),
        ("e2e_p50_s", "e2e_request_latency_seconds", 0.5),
        ("e2e_p95_s", "e2e_request_latency_seconds", 0.95),
    ] {
        if let Some((_, h)) = parsed.hists.iter().find(|(k, _)| k == hist)
            && let Some(r) = histogram_quantile(q, h.items())
        {
            set(&mut out, key, r);
        }
    }
    out
}

// ── startup log ─────────────────────────────────────────────────────────────────────────────────────────
static ANSI: LazyLock<Regex> = LazyLock::new(|| search(r"\x1b\[[0-9;]*[A-Za-z]"));
static TQDM: LazyLock<Regex> = LazyLock::new(|| search(r"(\d+)%\|[^|]*\|\s*(\d+)/(\d+)"));
static TQDM_TXT: LazyLock<Regex> =
    LazyLock::new(|| search(r"(\d+)%\s+Completed\s*\|\s*(\d+)/(\d+)"));
static OOM: LazyLock<Regex> = LazyLock::new(|| search(r"CUDA out of memory|OutOfMemoryError"));
static RAISE_FRAC: LazyLock<Regex> =
    LazyLock::new(|| search(r"Raise --mem-fraction-static above ([\d.]+)"));
static GRAPH_FAILED: LazyLock<Regex> =
    LazyLock::new(|| search(r"Capture (?:\w+ )*?(?:cuda|CUDA) graph failed"));
static CAPTURING: LazyLock<Regex> =
    LazyLock::new(|| search(r"Capturing (batches|num tokens) \((?:bs|num_tokens)=(\d+)[^)]*\)"));
static GRAPH_BEGIN: LazyLock<Regex> =
    LazyLock::new(|| search(r"Capture (?:\w+ )*?(?:cuda|CUDA) graph begin"));
static GRAPH_END_MEM: LazyLock<Regex> =
    LazyLock::new(|| search(r"Capture (?:\w+ )*?(?:cuda|CUDA) graph end.*?mem usage=([\d.]+) GB"));
static GRAPH_END: LazyLock<Regex> =
    LazyLock::new(|| search(r"Capture (?:\w+ )*?(?:cuda|CUDA) graph end"));
static SHARDS_WORD: LazyLock<Regex> =
    LazyLock::new(|| compile(r"shards|checkpoint", true, Anchor::Search));
static LOADING_WORD: LazyLock<Regex> = LazyLock::new(|| search(r"Loading|loading"));
static LOAD_END: LazyLock<Regex> =
    LazyLock::new(|| search(r"Load weight end\..*?mem usage=([\d.]+) GB"));
static ELAPSED: LazyLock<Regex> = LazyLock::new(|| search(r"elapsed=([\d.]+) s"));
static KV_ALLOC: LazyLock<Regex> =
    LazyLock::new(|| search(r"KV Cache (?:is allocated|VA upper bound)\..*?#tokens: (\d+)"));
static MAX_TOTAL: LazyLock<Regex> = LazyLock::new(|| search(r"max_total_num_tokens=(\d+)"));
static MAX_RUNNING: LazyLock<Regex> = LazyLock::new(|| search(r"max_running_requests=(\d+)"));
static CONTEXT_LEN: LazyLock<Regex> = LazyLock::new(|| search(r"context_len=(\d+)"));
static COMPILING: LazyLock<Regex> = LazyLock::new(|| search(r"torch\.compile|Compiling|Compile "));

/// Map one engine log line to structured progress, or None. Keys: `phase` in {loading_weights, allocating_kv,
/// capturing_graphs, compiling, ready, failed}, `pct`, `kv_cache_tokens`, `weights_gib`, `cuda_graph_gib`, and for a
/// failed phase `error` in {oom, memory_unbalanced, mem_fraction_too_low, attention_backend_unsupported,
/// cuda_graph_capture_failed}.
pub fn parse_startup_log(line: &str) -> Option<Obj> {
    parse_inner(line).unwrap_or(None)
}

fn parse_inner(line: &str) -> Result<Option<Obj>, Raised> {
    let stripped = ANSI.replace_all(line, "");
    let mut s: &str = &stripped;
    if s.contains('\r') {
        // tqdm redraws: keep the last frame
        let last = s.rsplit('\r').next().unwrap_or("");
        if !last.is_empty() {
            s = last;
        }
    }
    let s = py::strip(s);
    if s.is_empty() {
        return Ok(None);
    }
    let phase = |p: &str| ("phase", Value::String(p.to_string()));
    let failed = |e: &str| vec![phase("failed"), ("error", Value::String(e.to_string()))];
    if s.contains("The server is fired up and ready to roll") {
        return Ok(Some(obj(vec![phase("ready"), ("pct", Value::from(100.0))])));
    }
    if OOM.is_match(s) {
        return Ok(Some(obj(failed("oom"))));
    }
    // Failure messages read from the pinned image's source; none was provoked on a live engine.
    if s.contains("The memory capacity is unbalanced") {
        return Ok(Some(obj(failed("memory_unbalanced"))));
    }
    if let Some(m) = RAISE_FRAC.captures(s) {
        let mut d = failed("mem_fraction_too_low");
        d.push(("suggested_mem_fraction_static", float_v(&m[1])?));
        return Ok(Some(obj(d)));
    }
    if s.contains("FlashAttention v3 Backend requires SM") {
        return Ok(Some(obj(failed("attention_backend_unsupported"))));
    }
    if GRAPH_FAILED.is_match(s) {
        return Ok(Some(obj(failed("cuda_graph_capture_failed"))));
    }
    // Decode graphs show "Capturing batches (bs=N ...)", prefill graphs "Capturing num tokens (num_tokens=N ...)".
    if let Some(m) = CAPTURING.captures(s) {
        let key = if &m[1] == "batches" {
            "batch_size"
        } else {
            "num_tokens"
        };
        let mut d = obj(vec![phase("capturing_graphs"), (key, int_v(&m[2])?)]);
        if let Some(t) = TQDM.captures(s) {
            d.insert("pct".into(), float_v(&t[1])?);
        }
        return Ok(Some(d));
    }
    if GRAPH_BEGIN.is_match(s) {
        return Ok(Some(obj(vec![
            phase("capturing_graphs"),
            ("pct", Value::from(0.0)),
        ])));
    }
    if let Some(m) = GRAPH_END_MEM.captures(s) {
        return Ok(Some(obj(vec![
            phase("capturing_graphs"),
            ("pct", Value::from(100.0)),
            ("cuda_graph_gib", float_v(&m[1])?),
        ])));
    }
    if GRAPH_END.is_match(s) {
        return Ok(Some(obj(vec![
            phase("capturing_graphs"),
            ("pct", Value::from(100.0)),
        ])));
    }
    if SHARDS_WORD.is_match(s)
        && LOADING_WORD.is_match(s)
        && let Some(t) = TQDM_TXT.captures(s).or_else(|| TQDM.captures(s))
    {
        return Ok(Some(obj(vec![
            phase("loading_weights"),
            ("pct", float_v(&t[1])?),
        ])));
    }
    if let Some(m) = LOAD_END.captures(s) {
        let mut d = obj(vec![
            phase("loading_weights"),
            ("pct", Value::from(100.0)),
            ("weights_gib", float_v(&m[1])?),
        ]);
        if let Some(e) = ELAPSED.captures(s) {
            d.insert("load_seconds".into(), float_v(&e[1])?);
        }
        return Ok(Some(d));
    }
    if s.contains("Load weight begin") {
        return Ok(Some(obj(vec![
            phase("loading_weights"),
            ("pct", Value::from(0.0)),
        ])));
    }
    if let Some(m) = KV_ALLOC.captures(s) {
        return Ok(Some(obj(vec![
            phase("allocating_kv"),
            ("kv_cache_tokens", int_v(&m[1])?),
        ])));
    }
    if let Some(m) = MAX_TOTAL.captures(s) {
        let mut d = obj(vec![
            phase("allocating_kv"),
            ("kv_cache_tokens", int_v(&m[1])?),
        ]);
        for (key, rx) in [
            ("max_running_requests", &*MAX_RUNNING),
            ("context_len", &*CONTEXT_LEN),
        ] {
            if let Some(k) = rx.captures(s) {
                d.insert(key.into(), int_v(&k[1])?);
            }
        }
        return Ok(Some(d));
    }
    if COMPILING.is_match(s) {
        return Ok(Some(obj(vec![phase("compiling")])));
    }
    Ok(None)
}
