//! Python: backend/src/engine_console/services/gguf_info.py.
//!
//! GGUF and Ollama facts -> ModelInfo (contract v2, docs/ARCHITECTURE.md 10.5: `model_info` for gguf and ollama
//! copies), quant names, split-shard names. The hub never opens a model file: a node's scanner reads a GGUF header
//! (or an Ollama config blob) and hands over an entry with a few fixed fields plus a bounded, untrusted `meta` object.
//! llama.cpp's own key names are accepted flat ("llama.block_count") or nested ({"llama": {"block_count": 32}}), as
//! are the short names of the node's header reader. Anything not found stays None, so the fit estimator reports low
//! confidence instead of a made-up number.
//!
//! `py` holds the small Python-compatibility helpers the library and data modules share (str.strip, posixpath, floor
//! division, `re`'s `$`), each documented with the Python behaviour it reproduces.

use std::sync::LazyLock;

use regex::Regex;
use serde_json::{Map, Value};

use crate::adapters::base::ModelInfo;

/// llama.cpp's `llama_ftype` values as stored in `general.file_type` (the node's header reader carries the same
/// table); an unknown value gives None.
pub const FILE_TYPES: &[(i64, &str)] = &[
    (0, "F32"),
    (1, "F16"),
    (2, "Q4_0"),
    (3, "Q4_1"),
    (7, "Q8_0"),
    (8, "Q5_0"),
    (9, "Q5_1"),
    (10, "Q2_K"),
    (11, "Q3_K_S"),
    (12, "Q3_K_M"),
    (13, "Q3_K_L"),
    (14, "Q4_K_S"),
    (15, "Q4_K_M"),
    (16, "Q5_K_S"),
    (17, "Q5_K_M"),
    (18, "Q6_K"),
    (19, "IQ2_XXS"),
    (20, "IQ2_XS"),
    (21, "Q2_K_S"),
    (22, "IQ3_XS"),
    (23, "IQ3_XXS"),
    (24, "IQ1_S"),
    (25, "IQ4_NL"),
    (26, "IQ3_S"),
    (27, "IQ3_M"),
    (28, "IQ2_S"),
    (29, "IQ2_M"),
    (30, "IQ4_XS"),
    (31, "IQ1_M"),
    (32, "BF16"),
    (36, "TQ1_0"),
    (37, "TQ2_0"),
    (38, "MXFP4_MOE"),
];

/// `-(\d{5})-of-(\d{5})$`, case-insensitive (`$` handled by [`py::dollar`]).
static SHARD_SUFFIX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)-(\d{5})-of-(\d{5})\z").unwrap_or_else(|_| unreachable!()));
static SIZE_LABEL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\A\s*(\d+(?:\.\d+)?)\s*([KMBT])\s*\z").unwrap_or_else(|_| unreachable!())
});
/// raw_config keeps a handful of scalar keys, never tokenizer vocabularies
const RAW_KEYS: usize = 64;

/// Python-compatibility helpers shared by the library and data modules.
pub mod py {
    use serde_json::Value;

    /// `str.isspace()` for one character: Unicode White_Space plus the separators U+001C..U+001F, which Python
    /// counts as whitespace and Rust's `char::is_whitespace` does not.
    pub fn is_space(c: char) -> bool {
        c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
    }

    /// `str.strip()`
    pub fn strip(s: &str) -> &str {
        s.trim_matches(is_space)
    }

    /// `s[:n]` in characters.
    pub fn take(s: &str, n: usize) -> String {
        s.chars().take(n).collect()
    }

    /// `str.split()` (no argument): runs of whitespace, no empty strings.
    pub fn split_ws(s: &str) -> Vec<&str> {
        s.split(is_space).filter(|w| !w.is_empty()).collect()
    }

    /// Python's truthiness of a JSON value (`None`, `False`, `0`, `0.0`, `""`, `[]`, `{}` are false).
    pub fn truthy(v: &Value) -> bool {
        ec_proto::sanitize::truthy(v)
    }

    /// `a or b or ...` over optional JSON values (a missing key is `None`): the first truthy one, else the last.
    pub fn or_chain<'a>(vals: &[Option<&'a Value>]) -> Option<&'a Value> {
        let mut last = None;
        for v in vals {
            last = *v;
            if v.is_some_and(truthy) {
                return *v;
            }
        }
        last
    }

    /// `a // b` for ints (floor division); None for a zero divisor or an overflow.
    pub fn floordiv(a: i64, b: i64) -> Option<i64> {
        let q = a.checked_div(b)?;
        if a.checked_rem(b)? != 0 && ((a < 0) != (b < 0)) {
            Some(q - 1)
        } else {
            Some(q)
        }
    }

    /// `isinstance(v, int) and not isinstance(v, bool)` as an i64 (JSON integers that fit).
    pub fn as_int(v: &Value) -> Option<i64> {
        match v {
            Value::Number(n) if !n.is_f64() => n.as_i64(),
            _ => None,
        }
    }

    /// `isinstance(v, (int, float)) and not isinstance(v, bool)` as f64.
    pub fn as_num(v: &Value) -> Option<f64> {
        match v {
            Value::Number(n) => n.as_f64(),
            _ => None,
        }
    }

    /// The text a pattern ending in Python's `$` sees: `$` also matches just before one trailing "\n", so a match
    /// anchored at the end is tried on the text without that newline. Returns (text to match, the newline kept).
    pub fn dollar(s: &str) -> (&str, &str) {
        match s.strip_suffix('\n') {
            Some(head) => (head, "\n"),
            None => (s, ""),
        }
    }

    /// `posixpath.normpath`
    pub fn normpath(path: &str) -> String {
        if path.is_empty() {
            return ".".to_string();
        }
        let initial = if path.starts_with('/') {
            if path.starts_with("//") && !path.starts_with("///") {
                2
            } else {
                1
            }
        } else {
            0
        };
        let mut out: Vec<&str> = Vec::new();
        for comp in path.split('/') {
            if comp.is_empty() || comp == "." {
                continue;
            }
            if comp != ".." || (initial == 0 && out.is_empty()) || out.last() == Some(&"..") {
                out.push(comp);
            } else if !out.is_empty() {
                out.pop();
            }
        }
        let joined = out.join("/");
        let s = format!("{}{joined}", "/".repeat(initial));
        if s.is_empty() { ".".to_string() } else { s }
    }

    /// `posixpath.join(a, b)`
    pub fn join(a: &str, b: &str) -> String {
        if b.starts_with('/') {
            b.to_string()
        } else if a.is_empty() || a.ends_with('/') {
            format!("{a}{b}")
        } else {
            format!("{a}/{b}")
        }
    }

    /// `posixpath.relpath(path, start)` for two paths that are both absolute or both relative (the current
    /// directory cancels out in the second case, as it does in Python).
    pub fn relpath(path: &str, start: &str) -> String {
        let a = normpath(path);
        let s = normpath(start);
        let sl: Vec<&str> = s
            .split('/')
            .filter(|x| !x.is_empty() && *x != ".")
            .collect();
        let pl: Vec<&str> = a
            .split('/')
            .filter(|x| !x.is_empty() && *x != ".")
            .collect();
        let common = sl.iter().zip(pl.iter()).take_while(|(x, y)| x == y).count();
        let mut parts: Vec<&str> = vec![".."; sl.len() - common];
        parts.extend_from_slice(&pl[common..]);
        if parts.is_empty() {
            ".".to_string()
        } else {
            parts.join("/")
        }
    }

    /// `posixpath.dirname`
    pub fn dirname(p: &str) -> String {
        let i = p.rfind('/').map_or(0, |i| i + 1);
        let head = &p[..i];
        if !head.is_empty() && head.chars().any(|c| c != '/') {
            head.trim_end_matches('/').to_string()
        } else {
            head.to_string()
        }
    }

    /// `posixpath.basename`
    pub fn basename(p: &str) -> &str {
        p.rfind('/').map_or(p, |i| &p[i + 1..])
    }

    /// `s.lower().endswith(suffix)` for an ASCII `suffix` (only ASCII letters lower-case to ASCII letters).
    pub fn ends_with_ci(s: &str, suffix: &str) -> bool {
        let n = suffix.chars().count();
        let tail: Vec<char> = s.chars().rev().take(n).collect();
        tail.len() == n
            && tail
                .iter()
                .rev()
                .zip(suffix.chars())
                .all(|(a, b)| a.is_ascii() && a.to_ascii_lowercase() == b)
    }

    /// `s.lower().startswith(prefix)` for an ASCII `prefix`.
    pub fn starts_with_ci(s: &str, prefix: &str) -> bool {
        let n = prefix.chars().count();
        let head: Vec<char> = s.chars().take(n).collect();
        head.len() == n
            && head
                .iter()
                .zip(prefix.chars())
                .all(|(a, b)| a.is_ascii() && a.to_ascii_lowercase() == b)
    }
}

/// `quant_from_file_type`
pub fn quant_from_file_type(value: &Value) -> Option<String> {
    let v = py::as_int(value)?;
    FILE_TYPES
        .iter()
        .find(|(k, _)| *k == v)
        .map(|(_, q)| (*q).to_string())
}

/// The spelling ModelInfo.quantization and the adapters' compatibility rules use: lower case, no vendor prefix,
/// "mxfp4_moe" -> "mxfp4" (it is the same format).
pub fn normalize_quant(quant: Option<&str>) -> Option<String> {
    let quant = quant.filter(|q| !q.is_empty())?;
    let mut q = py::strip(quant).to_lowercase();
    for prefix in ["ud-", "i1-"] {
        if let Some(rest) = q.strip_prefix(prefix) {
            q = rest.to_string();
        }
    }
    if q == "mxfp4_moe" {
        return Some("mxfp4".into());
    }
    (!q.is_empty()).then_some(q)
}

/// "8.0B" / "70B" / "335M" -> a parameter count (Ollama's `model_type` and GGUF's `general.size_label`).
pub fn parse_size_label(label: &Value) -> Option<i64> {
    let s = label.as_str()?;
    let c = SIZE_LABEL.captures(s)?;
    let num: f64 = ascii_number(c.get(1)?.as_str())?;
    let unit = match c.get(2)?.as_str().to_ascii_uppercase().as_str() {
        "K" => 1e3,
        "M" => 1e6,
        "B" => 1e9,
        _ => 1e12,
    };
    #[allow(clippy::cast_possible_truncation)]
    Some((num * unit) as i64)
}

/// `float()` of a `\d+(\.\d+)?` match; None for non-ASCII digits (Python reads every Unicode decimal digit; a size
/// label written in another script is not supported here).
fn ascii_number(s: &str) -> Option<f64> {
    if s.chars().all(|c| c.is_ascii_digit() || c == '.') {
        s.parse().ok()
    } else {
        None
    }
}

/// Every file of a split set from its first shard ("m-00001-of-00003.gguf" -> the three names); a plain file, or a
/// name that is not the first shard, gives itself.
pub fn shard_names(first: &str, shards: Option<i64>) -> Vec<String> {
    let (directory, name) = match first.rfind('/') {
        Some(i) => (&first[..i], &first[i + 1..]),
        None => ("", first),
    };
    let stem = if py::ends_with_ci(name, ".gguf") {
        let n = name.chars().count();
        name.chars().take(n - 5).collect::<String>()
    } else {
        name.to_string()
    };
    let (text, _) = py::dollar(&stem);
    let Some(c) = SHARD_SUFFIX.captures(text) else {
        return vec![first.to_string()];
    };
    let (Some(idx), Some(tot), Some(whole)) = (
        c.get(1).and_then(|m| m.as_str().parse::<i64>().ok()),
        c.get(2).and_then(|m| m.as_str().parse::<i64>().ok()),
        c.get(0),
    ) else {
        return vec![first.to_string()];
    };
    if idx != 1 || !(2..=999).contains(&tot) || shards.is_some_and(|s| s != tot) {
        return vec![first.to_string()];
    }
    let head = &text[..whole.start()];
    let prefix = if directory.is_empty() {
        String::new()
    } else {
        format!("{directory}/")
    };
    (1..=tot)
        .map(|i| format!("{prefix}{head}-{i:05}-of-{tot:05}.gguf"))
        .collect()
}

// ---- reading untrusted metadata ------------------------------------------------------------------------------
/// meta["a.b.c"], meta["a"]["b.c"], meta["a"]["b"]["c"] ...: whichever way the node nested it. A present full key
/// answers even when its value is null (as Python's `return meta[head]`); a nested null does not.
fn lookup<'a>(meta: &'a Map<String, Value>, parts: &[&str]) -> Option<&'a Value> {
    for i in (1..=parts.len()).rev() {
        let head = parts[..i].join(".");
        if let Some(v) = meta.get(&head) {
            if i == parts.len() {
                return Some(v);
            }
            if let Value::Object(sub) = v
                && let Some(found) = lookup(sub, &parts[i..])
                && !found.is_null()
            {
                return Some(found);
            }
        }
    }
    None
}

fn get<'a>(meta: &'a Map<String, Value>, key: &str) -> Option<&'a Value> {
    if key.is_empty() {
        return None;
    }
    let parts: Vec<&str> = key.split('.').collect();
    lookup(meta, &parts)
}

/// `_int`: positive ints, or positive integral floats; never bools.
fn int_(v: Option<&Value>) -> Option<i64> {
    match v? {
        Value::Number(n) => {
            if !n.is_f64() {
                return n.as_i64().filter(|i| *i > 0);
            }
            let f = n.as_f64()?;
            if f.is_finite() && f.fract() == 0.0 && f > 0.0 {
                #[allow(clippy::cast_possible_truncation)]
                Some(f as i64)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Per-layer arrays exist (hybrid models: zero for a layer without attention). The largest entry is the safe answer
/// for a memory estimate.
fn int_or_max(v: Option<&Value>) -> Option<i64> {
    if let Some(Value::Array(a)) = v {
        return a.iter().take(512).filter_map(|x| int_(Some(x))).max();
    }
    int_(v)
}

fn str_(v: Option<&Value>) -> Option<String> {
    let s = v?.as_str()?;
    let t = py::strip(s);
    (!t.is_empty()).then(|| py::take(t, 128))
}

/// The first key whose value is not None.
fn first<'a>(meta: &'a Map<String, Value>, keys: &[String]) -> Option<&'a Value> {
    keys.iter()
        .filter_map(|k| get(meta, k))
        .find(|v| !v.is_null())
}

fn strs(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| (*s).to_string()).collect()
}

/// A ModelInfo for a gguf or ollama library entry. `size` is the bytes of every file of the model (weights).
pub fn model_info_from_entry(
    name: &str,
    entry: &Map<String, Value>,
    fmt: &str,
    size: Option<i64>,
) -> ModelInfo {
    let empty = Map::new();
    let meta = match entry.get("meta") {
        Some(Value::Object(m)) => m,
        _ => &empty,
    };
    let arch = str_(entry.get("arch"))
        .or_else(|| str_(first(meta, &strs(&["arch", "general.architecture"]))))
        .or_else(|| {
            if fmt == "ollama" {
                str_(get(meta, "model_family"))
            } else {
                None
            }
        });
    let keys = |suffix: &str, short: &[&str]| -> Vec<String> {
        let mut out = Vec::new();
        if let Some(a) = &arch {
            out.push(format!("{a}.{suffix}"));
        }
        out.extend(strs(short));
        out
    };
    let layers = int_(first(
        meta,
        &keys(
            "block_count",
            &["block_count", "n_layer", "num_hidden_layers"],
        ),
    ));
    let heads = int_or_max(first(
        meta,
        &keys(
            "attention.head_count",
            &[
                "head_count",
                "attention.head_count",
                "n_head",
                "num_attention_heads",
            ],
        ),
    ));
    let mut kv_heads = int_or_max(first(
        meta,
        &keys(
            "attention.head_count_kv",
            &[
                "head_count_kv",
                "attention.head_count_kv",
                "n_head_kv",
                "num_key_value_heads",
            ],
        ),
    ));
    if kv_heads.is_none() && heads.is_some() {
        kv_heads = heads; // llama.cpp's own default when the key is absent: multi-head attention
    }
    let embedding = int_(first(
        meta,
        &keys(
            "embedding_length",
            &["embedding_length", "n_embd", "hidden_size"],
        ),
    ));
    let key_len = int_(first(
        meta,
        &keys(
            "attention.key_length",
            &["key_length", "attention.key_length", "head_dim"],
        ),
    ));
    let head_dim = key_len.or_else(|| match (embedding, heads) {
        (Some(e), Some(h)) => py::floordiv(e, h),
        _ => None,
    });
    let experts = int_(first(
        meta,
        &keys("expert_count", &["expert_count", "num_experts"]),
    ));
    let context = int_(entry.get("context")).or_else(|| {
        int_(first(
            meta,
            &keys(
                "context_length",
                &["context_length", "context", "max_position_embeddings"],
            ),
        ))
    });
    let params = int_(entry.get("params"))
        .or_else(|| int_(first(meta, &strs(&["params", "general.parameter_count"]))))
        .or_else(|| {
            first(
                meta,
                &strs(&["general.size_label", "size_label", "model_type"]),
            )
            .and_then(parse_size_label)
        });
    let file_type = first(meta, &strs(&["general.file_type", "file_type"]));
    let quant = str_(entry.get("quant"))
        .or_else(|| str_(get(meta, "quant")))
        .or_else(|| file_type.and_then(quant_from_file_type))
        .or_else(|| str_(file_type));
    let quantization = normalize_quant(quant.as_deref());
    let weight_bytes = match size {
        Some(s) => int_(Some(&Value::from(s))),
        None => int_(entry.get("size")),
    };
    let mut raw_config = Map::new();
    for (k, v) in meta.iter().take(512) {
        if raw_config.len() >= RAW_KEYS {
            break;
        }
        if k.starts_with("tokenizer") {
            continue;
        }
        match v {
            Value::String(s) => {
                raw_config.insert(k.clone(), Value::String(py::take(s, 200)));
            }
            Value::Number(_) | Value::Bool(_) => {
                raw_config.insert(k.clone(), v.clone());
            }
            _ => {}
        }
    }
    let sliding_window = int_(first(
        meta,
        &keys("attention.sliding_window", &["sliding_window"]),
    ));
    let kv_lora_rank = int_(first(
        meta,
        &keys("attention.kv_lora_rank", &["kv_lora_rank"]),
    ));
    let dtype = match quantization.as_deref() {
        Some("f32") => Some("float32".to_string()),
        Some("f16") => Some("float16".to_string()),
        Some("bf16") => Some("bfloat16".to_string()),
        _ => None,
    };
    ModelInfo {
        repo_id: name.to_string(),
        revision: str_(entry.get("revision")),
        architectures: arch.clone().into_iter().collect(),
        model_type: arch,
        num_params: params,
        num_active_params: None,
        is_moe: experts.is_some_and(|e| e > 1),
        weight_bytes,
        quantization,
        dtype,
        hidden_size: embedding,
        num_layers: layers,
        num_attention_heads: heads,
        num_kv_heads: kv_heads,
        head_dim,
        max_position_embeddings: context,
        sliding_window,
        kv_lora_rank,
        gated: false,
        license: None,
        pipeline_tag: None,
        raw_config,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_py_helpers_match_python() {
        assert_eq!(py::normpath("/a//b/./c/../d"), "/a/b/d");
        assert_eq!(py::normpath("//x/y"), "//x/y");
        assert_eq!(py::normpath("///x"), "/x");
        assert_eq!(py::normpath("../a/.."), "..");
        assert_eq!(py::normpath(""), ".");
        assert_eq!(
            py::relpath("r/blobs/x", "r/snapshots/c/sub"),
            "../../../blobs/x"
        );
        assert_eq!(py::relpath("/data/hf", "/data/hf"), ".");
        assert_eq!(py::relpath("/mnt/alpha/hf/x", "/mnt/alpha"), "hf/x");
        assert_eq!(py::dirname("a/b/c"), "a/b");
        assert_eq!(py::dirname("c"), "");
        assert_eq!(py::dirname("/c"), "/");
        assert_eq!(py::floordiv(-7, 2), Some(-4));
        assert_eq!(py::floordiv(-7, -2), Some(3));
        assert_eq!(py::floordiv(7, 0), None);
        assert_eq!(py::strip("\u{1c} x \n"), "x");
        assert!(py::ends_with_ci("A.GGUF", ".gguf"));
        assert!(!py::ends_with_ci("gguf", ".gguf"));
    }

    #[test]
    fn test_shard_names_and_size_labels() {
        assert_eq!(
            shard_names("d/m-00001-of-00003.gguf", None),
            vec![
                "d/m-00001-of-00003.gguf",
                "d/m-00002-of-00003.gguf",
                "d/m-00003-of-00003.gguf"
            ]
        );
        assert_eq!(
            shard_names("m-00002-of-00003.gguf", None),
            vec!["m-00002-of-00003.gguf"]
        );
        assert_eq!(
            shard_names("m-00001-of-00003.gguf", Some(2)),
            vec!["m-00001-of-00003.gguf"]
        );
        assert_eq!(shard_names("m.gguf", None), vec!["m.gguf"]);
        assert_eq!(parse_size_label(&json!("8.0B")), Some(8_000_000_000));
        assert_eq!(parse_size_label(&json!(" 335m ")), Some(335_000_000));
        assert_eq!(parse_size_label(&json!("x")), None);
        assert_eq!(normalize_quant(Some("UD-Q4_K_XL")), Some("q4_k_xl".into()));
        assert_eq!(normalize_quant(Some("MXFP4_MOE")), Some("mxfp4".into()));
        assert_eq!(normalize_quant(Some("  ")), None);
    }

    #[test]
    fn test_model_info_from_flat_and_nested_meta() {
        let e = json!({"arch": "llama", "meta": {"llama.block_count": 32, "llama": {"attention": {"head_count": 32,
            "head_count_kv": [0, 8, 8]}}, "embedding_length": 4096, "general.file_type": 15,
            "tokenizer.ggml.tokens": "x", "note": "y"}});
        let m = model_info_from_entry("m", e.as_object().expect("obj"), "gguf", Some(10));
        assert_eq!(m.num_layers, Some(32));
        assert_eq!(m.num_attention_heads, Some(32));
        assert_eq!(m.num_kv_heads, Some(8));
        assert_eq!(m.head_dim, Some(128));
        assert_eq!(m.quantization.as_deref(), Some("q4_k_m"));
        assert_eq!(m.weight_bytes, Some(10));
        assert!(!m.raw_config.contains_key("tokenizer.ggml.tokens"));
    }
}
