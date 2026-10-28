//! Python: backend/src/engine_console/services/hf.py (tests/data/test_hf_gguf.py).
//!
//! HF service: search, model detail, the pure ModelInfo builder used by the fit estimator, and the GGUF quant grouping
//! the download picker needs (contract v2, docs/ARCHITECTURE.md 10.6). The API shapes `SearchHit`, `GgufGroup` and
//! `ModelDetail` are API shapes (the routes return them), and so are the `HfService` methods other modules call.

use crate::sync::Recover;
use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};

use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::adapters::base::ModelInfo;
use crate::domain::errors::{AppResult, internal};
use crate::domain::models::Page;
use crate::domain::ports::{HubClient, HubModel, RepoFile};
use crate::services::gguf_info::py;
use crate::services::store::paginate;

/// safetensors dtype -> bytes per parameter
const ST_BYTES: &[(&str, i64)] = &[
    ("F64", 8),
    ("F32", 4),
    ("BF16", 2),
    ("F16", 2),
    ("F8_E4M3", 1),
    ("F8_E5M2", 1),
    ("I8", 1),
    ("U8", 1),
    ("I16", 2),
    ("I32", 4),
    ("I64", 8),
];
const KNOWN_QUANT_TAGS: &[&str] = &["nvfp4", "fp8", "mxfp4", "awq", "gptq", "int4", "int8"];

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchHit {
    pub repo_id: String,
    pub downloads: Option<i64>,
    pub likes: Option<i64>,
    pub pipeline_tag: Option<String>,
    pub library_name: Option<String>,
    pub gated: bool,
    pub tags: Vec<String>,
    pub num_params: Option<i64>,
    pub approx_size_gib: Option<f64>,
    pub quantization: Option<String>,
}

/// The .gguf files of one quantisation of a repo: one file, or every shard of a split set.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GgufGroup {
    pub quant: String,
    pub files: Vec<String>,
    pub total_bytes: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelDetail {
    pub info: ModelInfo,
    pub card: String,
    /// [{path, size, sha256}]
    pub files: Vec<Value>,
    pub sha: Option<String>,
    #[serde(default)]
    pub gguf_groups: Vec<GgufGroup>,
}

/// GET /hf/search parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchParams {
    pub q: String,
    pub task: Option<String>,
    pub library: Option<String>,
    pub quant: Option<String>,
    pub max_gib: Option<f64>,
    pub gated: Option<bool>,
    pub sort: String,
    pub limit: i64,
    pub cursor: Option<String>,
}

impl Default for SearchParams {
    fn default() -> Self {
        Self {
            q: String::new(),
            task: None,
            library: None,
            quant: None,
            max_gib: None,
            gated: None,
            sort: "downloads".into(),
            limit: 20,
            cursor: None,
        }
    }
}

/// Python's `round(x, n)` (correctly rounded on the binary value, ties to even: what `{:.n}` does too).
pub fn py_round(x: f64, n: usize) -> f64 {
    if !x.is_finite() {
        return x;
    }
    format!("{x:.n$}").parse().unwrap_or(x)
}

fn st_bytes(dtype: &str) -> i64 {
    ST_BYTES
        .iter()
        .find(|(d, _)| *d == dtype)
        .map_or(2, |(_, b)| *b)
}

/// `sum(n * _ST_BYTES.get(dt, 2) for dt, n in params.items())` (saturating: Python ints do not overflow).
fn safetensors_bytes(params: &[(String, i64)]) -> i64 {
    params.iter().fold(0i64, |acc, (dt, n)| {
        acc.saturating_add(n.saturating_mul(st_bytes(dt)))
    })
}

fn num_eq(v: Option<&Value>, x: f64) -> bool {
    matches!(v, Some(Value::Number(n)) if n.as_f64() == Some(x))
}

fn str_lower(v: Option<&Value>, default: &str) -> String {
    match v {
        None => default.to_lowercase(),
        Some(v) => ec_proto::pyfmt::str_value(v).to_lowercase(),
    }
}

/// `detect_quantization(cfg, repo_id, tags)`: the quantisation method a config declares, else a known name in the
/// repo id or its tags.
pub fn detect_quantization(
    cfg: &Map<String, Value>,
    repo_id: &str,
    tags: &[String],
) -> Option<String> {
    if let Some(Value::Object(q)) = cfg.get("quantization_config") {
        let method = str_lower(q.get("quant_method"), "");
        if method == "compressed-tensors" {
            // a falsy or non-object `config_groups` (Python would raise on a truthy non-dict) reads as empty
            if let Some(Value::Object(groups)) = q.get("config_groups") {
                for grp in groups.values() {
                    let w = match grp {
                        Value::Object(g) => match g.get("weights") {
                            Some(Value::Object(w)) => w.clone(),
                            _ => Map::new(),
                        },
                        _ => Map::new(),
                    };
                    let bits = w.get("num_bits");
                    let typ = str_lower(w.get("type"), "");
                    if num_eq(bits, 4.0) && typ == "float" {
                        return Some("nvfp4".into());
                    }
                    if num_eq(bits, 8.0) && typ == "float" {
                        return Some("fp8".into());
                    }
                    if num_eq(bits, 4.0) {
                        return Some("int4".into());
                    }
                    if num_eq(bits, 8.0) {
                        return Some("int8".into());
                    }
                }
            }
            return Some("compressed-tensors".into());
        }
        if method == "modelopt" || method == "nvfp4" {
            let algo = str_lower(q.get("quant_algo"), "");
            return Some(if algo.contains("fp4") {
                "nvfp4".into()
            } else if algo.contains("fp8") {
                "fp8".into()
            } else {
                method
            });
        }
        if method == "bitsandbytes" {
            let four = q.get("load_in_4bit").is_some_and(py::truthy);
            return Some(if four { "bnb4" } else { "bnb8" }.into());
        }
        if !method.is_empty() {
            return Some(method);
        }
    }
    let mut hay = repo_id.to_lowercase();
    for t in tags {
        hay.push(' ');
        hay.push_str(&t.to_lowercase());
    }
    KNOWN_QUANT_TAGS
        .iter()
        .find(|k| hay.contains(*k))
        .map(|k| (*k).to_string())
}

/// `build_model_info(hub, revision)`: the facts the fit estimator needs, from a repo's config.json and listing.
/// A config whose `model_type` is not a string fails like pydantic does in Python (500 internal error).
pub fn build_model_info(hub: &HubModel, revision: Option<&str>) -> AppResult<ModelInfo> {
    let raw = hub.config.clone().unwrap_or_default();
    let cfg = match raw.get("text_config") {
        Some(Value::Object(t)) => t.clone(),
        _ => raw.clone(),
    };
    let geti = |keys: &[&str]| -> Option<i64> {
        for src in [&cfg, &raw] {
            for k in keys {
                if let Some(v) = src.get(*k).and_then(py::as_int) {
                    return Some(v);
                }
            }
        }
        None
    };
    let experts = geti(&["num_experts", "n_routed_experts", "num_local_experts"]);
    let hidden = geti(&["hidden_size", "n_embd", "d_model"]);
    let heads = geti(&["num_attention_heads", "n_head"]);
    let mla = geti(&["kv_lora_rank"]);
    let mut head_dim = geti(&["head_dim"]);
    if head_dim.is_none()
        && let (Some(h), Some(n)) = (hidden.filter(|h| *h != 0), heads.filter(|n| *n != 0))
        && mla.is_none_or(|m| m == 0)
    {
        head_dim = py::floordiv(h, n);
    }
    let st_files: Vec<i64> = hub
        .files
        .iter()
        .filter(|f| f.path.ends_with(".safetensors"))
        .map(|f| f.size)
        .collect();
    let mut weight_bytes =
        (!st_files.is_empty()).then(|| st_files.iter().fold(0i64, |a, b| a.saturating_add(*b)));
    if weight_bytes.is_none() && !hub.safetensors_params.is_empty() {
        weight_bytes = Some(safetensors_bytes(&hub.safetensors_params));
    }
    let architectures = match raw.get("architectures") {
        Some(Value::Array(a)) => a
            .iter()
            .filter_map(|x| x.as_str().map(str::to_string))
            .collect(),
        _ => Vec::new(),
    };
    let dtype = py::or_chain(&[
        cfg.get("torch_dtype"),
        cfg.get("dtype"),
        raw.get("torch_dtype"),
        raw.get("dtype"),
    ])
    .and_then(Value::as_str)
    .map(str::to_string);
    let model_type = match raw.get("model_type") {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) => Some(s.clone()),
        // pydantic refuses a non-string `model_type` (ValidationError -> 500 in Python)
        Some(_) => return Err(internal("unexpected server error").into()),
    };
    Ok(ModelInfo {
        repo_id: hub.repo_id.clone(),
        revision: revision.map(str::to_string),
        architectures,
        model_type,
        num_params: hub.safetensors_total,
        num_active_params: None,
        is_moe: experts.is_some_and(|e| e > 1),
        weight_bytes,
        quantization: detect_quantization(&raw, &hub.repo_id, &hub.tags),
        dtype,
        hidden_size: hidden,
        num_layers: geti(&["num_hidden_layers", "n_layer"]),
        num_attention_heads: heads,
        num_kv_heads: geti(&[
            "num_key_value_heads",
            "num_kv_heads",
            "multi_query_group_num",
        ]),
        head_dim,
        max_position_embeddings: geti(&["max_position_embeddings", "n_positions"]),
        sliding_window: geti(&["sliding_window"]),
        kv_lora_rank: mla,
        gated: hub.gated,
        license: hub.license.clone(),
        pipeline_tag: hub.pipeline_tag.clone(),
        raw_config: raw,
    })
}

// ---- GGUF quantisation names ----------------------------------------------------------------------------------
// llama.cpp's quantisation names, plus the two prefixes repos put in front of them: Unsloth's dynamic quants
// ("UD-Q4_K_XL") and imatrix quants ("i1-Q4_K_M"). Matched case-insensitively and only as a whole token, so
// "Q4_K_M" is found in "model-Q4_K_M-00001-of-00003.gguf" and in "model.q4_k_m.gguf" but not in a word.
static GGUF_SPLIT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)-\d{5}-of-\d{5}\z").unwrap_or_else(|_| unreachable!()));
/// Python: `(?<![A-Za-z0-9])(TOKEN)(?![A-Za-z0-9])`. The negative lookahead becomes "a non-alphanumeric character or
/// the end" (consumed: a token never starts with one, so no match is lost) and the lookbehind is checked by hand
/// in [`gguf_tokens`]; the alternation order and greediness are Python's, so the same token wins.
static GGUF_QUANT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)((?:UD-|i1-)?(?:IQ\d_(?:XXS|XS|S|M|NL|XL)|TQ\d_0|Q\d_K(?:_(?:XXS|XS|S|M|L|XL))?|Q\d_[01]|BF16|F16|F32|MXFP4(?:_MOE)?|NVFP4))(?:[^A-Za-z0-9]|\z)",
    )
    .unwrap_or_else(|_| unreachable!())
});

fn is_ascii_alnum(c: char) -> bool {
    c.is_ascii_alphanumeric()
}

/// Every quant token of `s`, in order (Python's `finditer` with the lookarounds).
fn gguf_tokens(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut pos = 0;
    while pos <= s.len() {
        let Some(c) = GGUF_QUANT.captures_at(s, pos) else {
            break;
        };
        let (Some(whole), Some(tok)) = (c.get(0), c.get(1)) else {
            break;
        };
        let before = s[..whole.start()].chars().next_back();
        if before.is_some_and(is_ascii_alnum) {
            // the lookbehind fails here: Python moves on one character
            pos = whole.start() + s[whole.start()..].chars().next().map_or(1, char::len_utf8);
            continue;
        }
        out.push(tok.as_str().to_string());
        pos = tok.end();
        if tok.end() == whole.end() && tok.start() == tok.end() {
            pos += 1;
        }
    }
    out
}

/// The path without ".gguf" and without a split-set suffix ("-00001-of-00003").
pub fn gguf_stem(path: &str) -> String {
    let p: String = if py::ends_with_ci(path, ".gguf") {
        let n = path.chars().count();
        path.chars().take(n - 5).collect()
    } else {
        path.to_string()
    };
    let (text, nl) = py::dollar(&p);
    match GGUF_SPLIT.find(text) {
        Some(m) => format!("{}{nl}", &text[..m.start()]),
        None => p,
    }
}

/// The quantisation token of a .gguf repo path ("unsloth/x/Q4_K_M/Model-Q4_K_M-00001-of-00002.gguf" -> "Q4_K_M").
/// The last token wins and directories count too; a multimodal projector ("mmproj-F16.gguf") is its own group; a
/// file with no recognisable token is labelled with its own name.
pub fn gguf_quant(path: &str) -> String {
    let stem = gguf_stem(path);
    let base = stem.rsplit('/').next().unwrap_or(&stem).to_string();
    let tok = gguf_tokens(&stem).pop().map(|raw| {
        let rest: String = raw.chars().skip(3).collect();
        if py::starts_with_ci(&raw, "ud-") {
            format!("UD-{}", rest.to_uppercase())
        } else if py::starts_with_ci(&raw, "i1-") {
            format!("i1-{}", rest.to_uppercase())
        } else {
            raw.to_uppercase()
        }
    });
    if py::starts_with_ci(&base, "mmproj") {
        return match tok {
            Some(t) => format!("mmproj-{t}"),
            None => "mmproj".into(),
        };
    }
    tok.unwrap_or(base)
}

/// Group (path, size) pairs of .gguf files by quantisation; smallest group first (the order a picker wants).
pub fn group_gguf<'a>(files: impl IntoIterator<Item = (&'a str, i64)>) -> Vec<GgufGroup> {
    let mut order: Vec<String> = Vec::new();
    let mut by_quant: HashMap<String, Vec<(String, i64)>> = HashMap::new();
    for (path, size) in files {
        if py::ends_with_ci(path, ".gguf") {
            let q = gguf_quant(path);
            if !by_quant.contains_key(&q) {
                order.push(q.clone());
            }
            by_quant
                .entry(q)
                .or_default()
                .push((path.to_string(), size));
        }
    }
    let mut groups: Vec<GgufGroup> = order
        .into_iter()
        .map(|q| {
            let items = by_quant.remove(&q).unwrap_or_default();
            let mut files: Vec<String> = items.iter().map(|(p, _)| p.clone()).collect();
            files.sort();
            let total = items.iter().fold(0i64, |a, (_, s)| a.saturating_add(*s));
            GgufGroup {
                quant: q,
                files,
                total_bytes: total,
            }
        })
        .collect();
    groups.sort_by(|a, b| {
        a.total_bytes
            .cmp(&b.total_bytes)
            .then_with(|| a.quant.cmp(&b.quant))
    });
    groups
}

pub struct HfService {
    hub: Arc<dyn HubClient>,
    ttl: Duration,
    cache: Mutex<HashMap<(String, String), (Instant, HubModel)>>,
}

impl std::fmt::Debug for HfService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("HfService")
    }
}

impl HfService {
    pub fn new(hub: Arc<dyn HubClient>) -> Self {
        Self::with_ttl(hub, Duration::from_secs(300))
    }

    /// `HfService(hub, cache_ttl_s=...)` (0: no caching, as the data tests use).
    pub fn with_ttl(hub: Arc<dyn HubClient>, ttl: Duration) -> Self {
        Self {
            hub,
            ttl,
            cache: Mutex::new(HashMap::new()),
        }
    }

    async fn hub_model(&self, repo_id: &str, revision: Option<&str>) -> AppResult<HubModel> {
        let key = (repo_id.to_string(), revision.unwrap_or("main").to_string());
        // the cache survives a poisoned lock (crate::sync): a panic elsewhere never turns caching off
        if let Some((t, m)) = self.cache.guard().get(&key)
            && t.elapsed() < self.ttl
        {
            return Ok(m.clone());
        }
        let m = self.hub.model(repo_id, revision).await?;
        self.cache.guard().insert(key, (Instant::now(), m.clone()));
        Ok(m)
    }

    pub async fn model_info(&self, repo_id: &str, revision: Option<&str>) -> AppResult<ModelInfo> {
        build_model_info(&self.hub_model(repo_id, revision).await?, revision)
    }

    pub async fn detail(&self, repo_id: &str, revision: Option<&str>) -> AppResult<ModelDetail> {
        let hub = self.hub_model(repo_id, revision).await?;
        let card = self.hub.card(repo_id, revision).await?;
        Ok(ModelDetail {
            info: build_model_info(&hub, revision)?,
            card,
            sha: hub.sha.clone(),
            files: hub
                .files
                .iter()
                .map(|f| json!({"path": f.path, "size": f.size, "sha256": f.sha256}))
                .collect(),
            gguf_groups: Self::gguf_groups(&hub.files),
        })
    }

    /// The quantisations a GGUF repo offers, split sets kept together: [{quant, files, total_bytes}].
    pub fn gguf_groups(files: &[RepoFile]) -> Vec<GgufGroup> {
        group_gguf(files.iter().map(|f| (f.path.as_str(), f.size)))
    }

    /// The same from `{path, size}` objects (entries that are not a str path and an int size are skipped).
    pub fn gguf_groups_of_dicts(files: &[Value]) -> Vec<GgufGroup> {
        group_gguf(files.iter().filter_map(|f| {
            let path = f.get("path")?.as_str()?;
            let size = py::as_int(f.get("size")?)?;
            Some((path, size))
        }))
    }

    pub async fn list_files(&self, repo_id: &str, revision: Option<&str>) -> AppResult<HubModel> {
        self.hub_model(repo_id, revision).await
    }

    pub async fn search(&self, params: &SearchParams) -> AppResult<Page<SearchHit>> {
        let (limit, offset) = paginate(params.limit, params.cursor.as_deref());
        // HF has no offset param: over-fetch and slice; filters that HF can't do run client-side.
        let fetch = (offset + limit + 1).min(100);
        let raw = self
            .hub
            .search(
                &params.q,
                params.task.as_deref(),
                params.library.as_deref(),
                params.quant.as_deref(),
                &params.sort,
                fetch,
                true,
            )
            .await?;
        let mut hits = Vec::new();
        for m in raw {
            #[allow(clippy::cast_precision_loss)]
            let size = (!m.safetensors_params.is_empty())
                .then(|| safetensors_bytes(&m.safetensors_params) as f64 / 1_073_741_824.0);
            if let (Some(max), Some(s)) = (params.max_gib, size)
                && s > max
            {
                continue;
            }
            if params.gated.is_some_and(|g| g != m.gated) {
                continue;
            }
            hits.push(SearchHit {
                repo_id: m.repo_id.clone(),
                downloads: m.downloads,
                likes: m.likes,
                pipeline_tag: m.pipeline_tag.clone(),
                library_name: m.library_name.clone(),
                gated: m.gated,
                tags: m.tags.iter().take(40).cloned().collect(),
                num_params: m.safetensors_total,
                approx_size_gib: size.map(|s| py_round(s, 2)),
                quantization: detect_quantization(&Map::new(), &m.repo_id, &m.tags),
            });
        }
        let (off, lim) = (
            usize::try_from(offset).unwrap_or(usize::MAX),
            usize::try_from(limit).unwrap_or(0),
        );
        let more = hits.len() > off.saturating_add(lim);
        let page: Vec<SearchHit> = hits.into_iter().skip(off).take(lim).collect();
        Ok(Page::new(page, more.then(|| (offset + limit).to_string())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gguf_tokens_honour_lookarounds() {
        assert_eq!(gguf_quant("Q4_K_MX.gguf"), "Q4_K");
        assert_eq!(gguf_quant("aQ4_K_M.gguf"), "aQ4_K_M");
        assert_eq!(gguf_quant("x-Q4_K_M-Q8_0.gguf"), "Q8_0");
        assert_eq!(gguf_stem("m-00001-of-00002\n"), "m\n");
        assert_eq!(py_round(0.125, 2), 0.12);
        assert_eq!(py_round(2.675, 2), 2.67);
    }
}
