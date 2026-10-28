//! Port of backend/src/engine_console/adapters/vllm_parse.py: pure parsers for vLLM
//! Prometheus text and container log lines (no I/O).
//!
//! Metric names were confirmed against a live v0.27.1 `/metrics` scrape and the v0.27.1 image's metric registry (see
//! the Python module). prometheus_client appends `_total` to counters, so lookups try `<name>` then `<name>_total`.
//!
//! Python raises in two places that its callers catch (the metrics listener, the supervisor tick): a histogram bucket
//! whose `le` label is not a number makes `parse_metrics` raise, and a log line whose captured number does not convert
//! (`1.2.3`, a 400-digit integer) makes `parse_log_line` raise. Here the first gives `None` from
//! [`scrape_to_canonical`] (the adapter then reports nothing) and the second `None` (no progress from that line).

/// The shared Python-compat module, under the name the vLLM/SGLang code and its tests use (PORTING.md §A5).
pub use super::pycompat as py;

use std::collections::HashMap;
use std::sync::LazyLock;

use regex::Regex;
use serde_json::{Map, Number, Value};

use self::py::{Anchor, BigInt, FloatDict, compile};
use super::base::{FloatMap, Obj};

/// Label set of one sample, as the Python dict built from `findall` (a repeated name keeps its first position and
/// its last value).
pub type Labels = Vec<(String, String)>;

static LINE: LazyLock<Regex> = LazyLock::new(|| {
    compile(
        r"^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{(.*)\})?\s+(\S+)(?:\s+\S+)?\s*$",
        false,
        Anchor::Start,
    )
});
static LABEL: LazyLock<Regex> = LazyLock::new(|| {
    compile(
        r#"([A-Za-z_][A-Za-z0-9_]*)="((?:[^"\\]|\\.)*)""#,
        false,
        Anchor::Search,
    )
});

fn label_get<'a>(labels: &'a Labels, key: &str) -> Option<&'a str> {
    labels
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.as_str())
}

/// Parse Prometheus exposition text into (name, labels, value); skips comments and bad lines.
pub fn parse_samples(text: &str) -> Vec<(String, Labels, f64)> {
    let mut out = Vec::new();
    for raw in crate::services::common::py_splitlines(text) {
        let line = py::strip(raw);
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some(m) = LINE.captures(line) else {
            continue;
        };
        let Some(value) = py::float_str(&m[3]) else {
            continue;
        };
        let mut labels: Labels = Vec::new();
        for c in LABEL.captures_iter(m.get(2).map_or("", |g| g.as_str())) {
            let k = c[1].to_string();
            let v = c[2].replace("\\\"", "\"").replace("\\\\", "\\");
            if let Some(slot) = labels.iter_mut().find(|(x, _)| *x == k) {
                slot.1 = v;
            } else {
                labels.push((k, v));
            }
        }
        out.push((m[1].to_string(), labels, value));
    }
    out
}

/// Indexed view of one scrape, tolerant to `_total` and legacy names.
#[derive(Debug, Default, Clone)]
pub struct Scrape {
    names: Vec<String>,
    by_name: HashMap<String, Vec<(Labels, f64)>>,
}

impl Scrape {
    pub fn new(text: &str) -> Self {
        let mut s = Self::default();
        for (name, labels, value) in parse_samples(text) {
            if name.starts_with("vllm:") {
                if !s.by_name.contains_key(&name) {
                    s.names.push(name.clone());
                }
                s.by_name.entry(name).or_default().push((labels, value));
            }
        }
        s
    }

    fn rows(&self, names: &[&str], suffix: &str) -> &[(Labels, f64)] {
        for base in names {
            for cand in [format!("{base}{suffix}"), format!("{base}_total{suffix}")] {
                if let Some(rows) = self.by_name.get(&cand) {
                    return rows;
                }
            }
        }
        &[]
    }

    fn values(rows: &[(Labels, f64)]) -> Vec<f64> {
        rows.iter().map(|(_, v)| *v).collect()
    }

    /// Sum over all label sets (engines / DP ranks) of the first metric name that exists.
    pub fn total(&self, names: &[&str]) -> Option<f64> {
        let rows = self.rows(names, "");
        if rows.is_empty() {
            return None;
        }
        Some(py::sum(&Self::values(rows)))
    }

    pub fn mean(&self, names: &[&str]) -> Option<f64> {
        let rows = self.rows(names, "");
        if rows.is_empty() {
            return None;
        }
        Some(py::sum(&Self::values(rows)) / rows.len() as f64)
    }

    /// Merged cumulative buckets [(le, count)] sorted by le, and the total count. `Err` where Python raises (an `le`
    /// that `float()` refuses).
    #[allow(clippy::type_complexity)]
    pub fn histogram(&self, names: &[&str]) -> Result<Option<(Vec<(f64, f64)>, f64)>, Raised> {
        let rows = self.rows(names, "_bucket");
        if rows.is_empty() {
            return Ok(None);
        }
        let mut merged = FloatDict::default();
        for (labels, v) in rows {
            let Some(le) = label_get(labels, "le") else {
                continue;
            };
            let key = if le == "+Inf" || le == "Inf" {
                f64::INFINITY
            } else {
                py::float_str(le).ok_or(Raised)?
            };
            merged.add(key, *v);
        }
        let buckets: Vec<(f64, f64)> = py::sorted_by_float_key(merged.items());
        Ok(buckets.last().map(|&(_, total)| (buckets.clone(), total)))
    }

    /// The `model_name` labels seen, sorted (the adapter's rate-state key).
    pub fn model_names(&self) -> Vec<String> {
        let mut names: Vec<String> = Vec::new();
        for n in &self.names {
            for (lab, _) in self.by_name.get(n).map_or(&[][..], Vec::as_slice) {
                if let Some(m) = label_get(lab, "model_name")
                    && !names.iter().any(|x| x == m)
                {
                    names.push(m.to_string());
                }
            }
        }
        names.sort();
        names
    }
}

/// Prometheus-style quantile with linear interpolation inside the bucket (lifetime totals; a quantile in the +Inf
/// bucket returns the highest finite bound).
pub fn histogram_quantile(q: f64, buckets: &[(f64, f64)]) -> Option<f64> {
    let &(_, total) = buckets.last()?;
    if total <= 0.0 {
        return None;
    }
    let rank = q * total;
    let (mut prev_le, mut prev_count) = (0.0_f64, 0.0_f64);
    for &(le, count) in buckets {
        if count >= rank {
            if le.is_infinite() {
                return buckets
                    .iter()
                    .rev()
                    .map(|(b, _)| *b)
                    .find(|b| !b.is_infinite());
            }
            let span = count - prev_count;
            if span <= 0.0 {
                return Some(le);
            }
            return Some(prev_le + (le - prev_le) * ((rank - prev_count) / span));
        }
        prev_le = le;
        prev_count = count;
    }
    None
}

fn put(out: &mut FloatMap, key: &str, val: Option<f64>) {
    if let Some(v) = val
        && !v.is_nan()
    {
        if let Some(slot) = out.iter_mut().find(|(k, _)| k == key) {
            slot.1 = v;
        } else {
            out.push((key.to_string(), v));
        }
    }
}

/// Instantaneous canonical keys (the rates `*_tps` are added by the adapter, which keeps state). `None` where Python
/// raises.
pub fn scrape_to_canonical(s: &Scrape) -> Option<FloatMap> {
    let mut out: FloatMap = Vec::new();
    put(
        &mut out,
        "requests_running",
        s.total(&["vllm:num_requests_running"]),
    );
    put(
        &mut out,
        "requests_waiting",
        s.total(&["vllm:num_requests_waiting"]),
    );
    // 0-1 fraction on every version we know (gpu_cache_usage_perc is the pre-rename spelling).
    let kv = s.mean(&["vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"]);
    put(&mut out, "kv_cache_usage_pct", kv.map(|k| k * 100.0));

    let hits = s.total(&["vllm:prefix_cache_hits", "vllm:gpu_prefix_cache_hits"]);
    let queries = s.total(&["vllm:prefix_cache_queries", "vllm:gpu_prefix_cache_queries"]);
    match (hits, queries) {
        (Some(h), Some(q)) if q != 0.0 => {
            put(&mut out, "prefix_cache_hit_pct", Some(100.0 * h / q));
        }
        (_, None) => {
            let legacy = s.mean(&[
                "vllm:gpu_prefix_cache_hit_rate",
                "vllm:cpu_prefix_cache_hit_rate",
            ]);
            put(&mut out, "prefix_cache_hit_pct", legacy.map(|l| l * 100.0));
        }
        _ => {}
    }

    put(
        &mut out,
        "prompt_tokens_total",
        s.total(&["vllm:prompt_tokens"]),
    );
    put(
        &mut out,
        "generation_tokens_total",
        s.total(&["vllm:generation_tokens"]),
    );
    put(
        &mut out,
        "preemptions_total",
        s.total(&["vllm:num_preemptions"]),
    );

    let quantiles: [(&str, f64, &[&str]); 5] = [
        ("ttft_p50_s", 0.5, &["vllm:time_to_first_token_seconds"]),
        ("ttft_p95_s", 0.95, &["vllm:time_to_first_token_seconds"]),
        // request_time_per_output_token_seconds is a different (per-request mean) series: not a fallback.
        (
            "itl_p50_s",
            0.5,
            &[
                "vllm:inter_token_latency_seconds",
                "vllm:time_per_output_token_seconds",
            ],
        ),
        ("e2e_p50_s", 0.5, &["vllm:e2e_request_latency_seconds"]),
        ("e2e_p95_s", 0.95, &["vllm:e2e_request_latency_seconds"]),
    ];
    for (key, q, names) in quantiles {
        if let Some((buckets, _)) = s.histogram(names).ok()? {
            put(&mut out, key, histogram_quantile(q, &buckets));
        }
    }

    let accepted = s.total(&["vllm:spec_decode_num_accepted_tokens"]);
    let drafted = s.total(&["vllm:spec_decode_num_draft_tokens"]);
    match (accepted, drafted) {
        (Some(a), Some(d)) if d != 0.0 => {
            put(&mut out, "spec_decode_accept_pct", Some(100.0 * a / d));
        }
        (_, None) => {
            let legacy = s.mean(&["vllm:spec_decode_draft_acceptance_rate"]);
            put(
                &mut out,
                "spec_decode_accept_pct",
                legacy.map(|l| l * 100.0),
            );
        }
        _ => {}
    }
    Some(out)
}

// ── log lines ───────────────────────────────────────────────────────────────────────────────────────────
fn search(pat: &str) -> Regex {
    compile(pat, false, Anchor::Search)
}

static ANSI: LazyLock<Regex> = LazyLock::new(|| search(r"\x1b\[[0-9;]*m"));
static KV_TOKENS: LazyLock<Regex> = LazyLock::new(|| search(r"GPU KV cache size: ([\d,]+) tokens"));
static MAX_CONC: LazyLock<Regex> =
    LazyLock::new(|| search(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x"));
static KV_AVAIL: LazyLock<Regex> =
    LazyLock::new(|| search(r"Available KV cache memory: ([\d.]+) GiB"));
static KV_RESERVED: LazyLock<Regex> =
    LazyLock::new(|| search(r"reserved ([\d.]+) GiB memory for KV Cache"));
static LOADED: LazyLock<Regex> =
    LazyLock::new(|| search(r"Model loading took ([\d.]+) GiB(?: memory)? and ([\d.]+) seconds"));
static SHARDS: LazyLock<Regex> = LazyLock::new(|| {
    search(r"Loading (?:safetensors checkpoint|pt checkpoint|checkpoint)[^:]*shards:\s+(\d+)%")
});
static CUDA_GRAPH: LazyLock<Regex> =
    LazyLock::new(|| search(r"Capturing CUDA graphs[^|]*?:\s*(\d+)%"));
static GRAPH_DONE: LazyLock<Regex> =
    LazyLock::new(|| search(r"Graph capturing finished in ([\d.]+) secs?, took ([\d.]+) GiB"));
static INIT_ENGINE: LazyLock<Regex> = LazyLock::new(|| {
    search(r"init engine \(profile, create kv cache, warmup model\) took ([\d.]+) s")
});
// Format string in vllm/v1/metrics/loggers.py (v0.27.1): "Deferred" and "Preemptions" appear only when non-zero.
static STATS: LazyLock<Regex> = LazyLock::new(|| {
    search(concat!(
        r"Avg prompt throughput: ([\d.]+) tokens/s, Avg generation throughput: ([\d.]+) tokens/s, ",
        r"Running: (\d+) reqs?, Waiting: (\d+) reqs?(?:, Deferred: (\d+) reqs?)?(?:, Preemptions: (\d+))?, ",
        r"GPU KV cache usage: ([\d.]+)%(?:, Prefix cache hit rate: ([\d.]+)%)?"
    ))
});
static EST_MAX_LEN: LazyLock<Regex> =
    LazyLock::new(|| search(r"estimated maximum model length is (\d+)"));
static MAMBA_SEQS: LazyLock<Regex> = LazyLock::new(|| {
    search(r"max_num_seqs \((\d+)\) exceeds available Mamba cache blocks \((\d+)\)")
});
static BAD_BACKEND: LazyLock<Regex> =
    LazyLock::new(|| search(r"Selected backend (\w+) is not valid for this configuration"));

/// Python raised (a ValueError or OverflowError its caller catches) while converting a captured number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Raised;

/// `float(s)` as a JSON number (ValueError -> Raised; a non-finite result cannot be carried in JSON -> Raised).
pub(crate) fn float_v(s: &str) -> Result<Value, Raised> {
    py::float_str(s)
        .and_then(Number::from_f64)
        .map(Value::Number)
        .ok_or(Raised)
}

/// `int(s)` of a `\d+` capture.
pub(crate) fn int_v(s: &str) -> Result<Value, Raised> {
    py::int_str(s).and_then(|b| b.to_value()).ok_or(Raised)
}

/// `int(_num(s))`: `float(s.replace(",", ""))` then `int()`.
fn int_num_v(s: &str) -> Result<Value, Raised> {
    let f = py::float_str(&s.replace(',', "")).ok_or(Raised)?;
    BigInt::from_f64(f).and_then(|b| b.to_value()).ok_or(Raised)
}

pub(crate) fn obj(pairs: Vec<(&str, Value)>) -> Obj {
    let mut m = Map::new();
    for (k, v) in pairs {
        m.insert(k.to_string(), v);
    }
    m
}

/// Map one engine log line to structured progress, or None when it carries none (or where Python raises).
///
/// Keys: `phase` in {loading_weights, compiling, capturing_graphs, warming_up, ready}, `pct`, `kv_cache_tokens`,
/// `kv_cache_gib`, `max_concurrency`, `weights_gib`, `graph_gib`, `error`, plus `stats` for the periodic throughput
/// line. A single tqdm line may hold many progress redraws; the last wins. Failure lines return `error` with one of
/// cuda_oom, kv_cache_too_small, max_num_seqs_too_high, attention_backend_invalid, no_gpu, insufficient_free_vram.
pub fn parse_log_line(line: &str) -> Option<Obj> {
    parse_log_line_inner(line).unwrap_or(None)
}

fn parse_log_line_inner(line: &str) -> Result<Option<Obj>, Raised> {
    let line = ANSI.replace_all(line, "");
    let line: &str = &line;
    let phase = |p: &str| ("phase", Value::String(p.to_string()));
    if line.contains("Application startup complete") {
        return Ok(Some(obj(vec![phase("ready"), ("pct", Value::from(100.0))])));
    }
    if let Some(m) = KV_TOKENS.captures(line) {
        return Ok(Some(obj(vec![("kv_cache_tokens", int_num_v(&m[1])?)])));
    }
    if let Some(m) = MAX_CONC.captures(line) {
        return Ok(Some(obj(vec![
            ("max_concurrency", float_v(&m[2])?),
            ("max_model_len", int_num_v(&m[1])?),
        ])));
    }
    if let Some(m) = KV_AVAIL.captures(line) {
        return Ok(Some(obj(vec![("kv_cache_gib", float_v(&m[1])?)])));
    }
    if let Some(m) = KV_RESERVED.captures(line) {
        return Ok(Some(obj(vec![("kv_cache_gib", float_v(&m[1])?)])));
    }
    if let Some(m) = LOADED.captures(line) {
        return Ok(Some(obj(vec![
            phase("loading_weights"),
            ("pct", Value::from(100.0)),
            ("weights_gib", float_v(&m[1])?),
            ("load_seconds", float_v(&m[2])?),
        ])));
    }
    if line.contains("Starting to load model") {
        return Ok(Some(obj(vec![
            phase("loading_weights"),
            ("pct", Value::from(0.0)),
        ])));
    }
    if let Some(m) = SHARDS.captures_iter(line).last() {
        return Ok(Some(obj(vec![
            phase("loading_weights"),
            ("pct", float_v(&m[1])?),
        ])));
    }
    if line.contains("Capturing CUDA graphs") {
        let mut out = obj(vec![phase("capturing_graphs")]);
        if let Some(m) = CUDA_GRAPH.captures_iter(line).last() {
            out.insert("pct".into(), float_v(&m[1])?);
        }
        return Ok(Some(out));
    }
    if let Some(m) = GRAPH_DONE.captures(line) {
        return Ok(Some(obj(vec![
            phase("capturing_graphs"),
            ("pct", Value::from(100.0)),
            ("graph_gib", float_v(&m[2])?),
        ])));
    }
    if let Some(m) = INIT_ENGINE.captures(line) {
        return Ok(Some(obj(vec![
            phase("warming_up"),
            ("init_seconds", float_v(&m[1])?),
        ])));
    }
    if line.contains("torch.compile took")
        || line.contains("Dynamo bytecode transform")
        || line.contains("Compiling a graph for")
        || line.contains("Directly load AOT compilation")
    {
        return Ok(Some(obj(vec![phase("compiling")])));
    }
    if let Some(m) = STATS.captures(line) {
        let mut st = obj(vec![
            ("prompt_tps", float_v(&m[1])?),
            ("generation_tps", float_v(&m[2])?),
            ("running", int_v(&m[3])?),
            ("waiting", int_v(&m[4])?),
            ("kv_cache_pct", float_v(&m[7])?),
        ]);
        if let Some(g) = m.get(5) {
            st.insert("deferred".into(), int_v(g.as_str())?);
        }
        if let Some(g) = m.get(6) {
            st.insert("preemptions".into(), int_v(g.as_str())?);
        }
        if let Some(g) = m.get(8) {
            st.insert("prefix_hit_pct".into(), float_v(g.as_str())?);
        }
        return Ok(Some(obj(vec![("stats", Value::Object(st))])));
    }
    let tail = || Value::String(py::tail_chars(py::strip(line), 300).to_string());
    let error = |e: &str| ("error", Value::String(e.to_string()));
    if line.contains("OutOfMemoryError") || line.contains("CUDA out of memory") {
        return Ok(Some(obj(vec![error("cuda_oom"), ("message", tail())])));
    }
    // Failure messages read from the pinned image's source; none was provoked on a live engine.
    if let Some(m) = EST_MAX_LEN.captures(line) {
        return Ok(Some(obj(vec![
            error("kv_cache_too_small"),
            ("suggested_max_model_len", int_v(&m[1])?),
            ("message", tail()),
        ])));
    }
    if line.contains("is larger than the available KV cache memory")
        || line.contains("larger than the maximum number of tokens that can be stored in KV cache")
        || line.contains("No available memory for the cache blocks")
    {
        return Ok(Some(obj(vec![
            error("kv_cache_too_small"),
            ("message", tail()),
        ])));
    }
    if let Some(m) = MAMBA_SEQS.captures(line) {
        return Ok(Some(obj(vec![
            error("max_num_seqs_too_high"),
            ("max_num_seqs_limit", int_v(&m[2])?),
            ("message", tail()),
        ])));
    }
    if let Some(m) = BAD_BACKEND.captures(line) {
        return Ok(Some(obj(vec![
            error("attention_backend_invalid"),
            ("backend", Value::String(m[1].to_string())),
            ("message", tail()),
        ])));
    }
    if line.contains("No CUDA runtime is found") || line.contains("Failed to infer device type") {
        return Ok(Some(obj(vec![error("no_gpu"), ("message", tail())])));
    }
    if line.contains("Free memory on device")
        && line.contains("is less than desired GPU memory utilization")
    {
        return Ok(Some(obj(vec![
            error("insufficient_free_vram"),
            ("message", tail()),
        ])));
    }
    Ok(None)
}
