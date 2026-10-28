//! Python: node/src/ecnode/models.py.
//!
//! Model scanner (docs/FLEET-PROTOCOL.md §6.1): what models each policy root holds, in the background.
//!
//! Root kinds: hf-home (<root>/hub/models--org--name), hf-hub (<root>/models--org--name), gguf (.gguf files, split
//! sets grouped), plain (directories with config.json + weights), ollama (manifests -> layers -> blobs), other (one
//! entry per top-level directory) and auto (detected per root). Bounded per root: at most 50 000 directory entries
//! and 120 s. Metadata files are read only when their realpath stays inside the root, through O_NOFOLLOW opens, with
//! size caps; names that look like keys or secrets are never opened (an Ollama home keeps its own ssh-style keys).
//!
//! File names that are not UTF-8 are shown with U+FFFD in entries (Python would carry surrogate escapes that the
//! frame encoder then refuses); the files themselves are always reached through their real paths.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::fs::{FileType, Metadata};
use std::io::Read;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, LazyLock, Mutex, MutexGuard};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use ec_proto::{IO_ERROR, NOT_FOUND, Obj, clean_json};
use serde_json::{Value, json};

use crate::confine::{inside, open_inside, open_root, read_inside, split_rel};
use crate::errors::{HandlerError, OpError};
use crate::gguf;
use crate::policy::{Policy, Root};
use crate::util::{Guarded, clean_text, is_secret_name, py};

pub type PolicyFn = Arc<dyn Fn() -> Arc<Policy> + Send + Sync>;
pub type OnChange = Arc<dyn Fn(i64) + Send + Sync>;
pub type Clock = Arc<dyn Fn() -> f64 + Send + Sync>;

pub const MAX_ENTRIES: i64 = 50_000;
pub const MAX_SECONDS: f64 = 120.0;
pub const HUNG_GRACE_S: f64 = 30.0;
pub const RESCAN_S: f64 = 600.0;
pub const CHECK_S: f64 = 30.0;
pub const FRESH_S: f64 = 10.0;
pub const CONFIG_MAX: usize = 512 * 1024;
pub const MANIFEST_MAX: usize = 256 * 1024;
pub const SAFETENSORS_HEADER_MAX: u64 = 16 * 1024 * 1024;
/// the largest real shards hold a few thousand tensors
pub const SAFETENSORS_MAX_TENSORS: usize = 200_000;
pub const SAFETENSORS_MAX_DIMS: usize = 32;
pub const PARAMS_MAX: u128 = 1 << 53;
pub const MAX_SAFETENSORS_FILES: usize = 64;
pub const PAGE_BYTES: usize = 768 * 1024;
pub const WEIGHT_EXT: &[&str] = &[".safetensors", ".bin", ".pt", ".pth"];
pub const BIG_FILE: u64 = 64 * 1024 * 1024;
const NOT_WEIGHTS: &[&str] = &[
    ".json",
    ".md",
    ".txt",
    ".jinja",
    ".py",
    ".model",
    ".tiktoken",
    ".gitattributes",
];

/// proto MODEL_ENTRY, in its order.
pub const MODEL_ENTRY: &[&str] = &[
    "root", "rel", "format", "name", "revision", "size", "files", "mtime", "atime", "quant",
    "params", "arch", "context", "shards", "meta", "warnings",
];
const META_KEYS: &[&str] = &[
    "architectures",
    "model_type",
    "torch_dtype",
    "dtype",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "max_position_embeddings",
    "vocab_size",
    "rope_scaling",
    "rope_theta",
    "sliding_window",
    "num_experts",
    "num_local_experts",
    "n_routed_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
    "kv_lora_rank",
    "q_lora_rank",
    "qk_rope_head_dim",
    "v_head_dim",
    "tie_word_embeddings",
    "layer_types",
];
const QUANT_KEYS: &[&str] = &[
    "quant_method",
    "bits",
    "group_size",
    "quant_algo",
    "format",
    "weight_block_size",
    "fmt",
    "load_in_4bit",
    "load_in_8bit",
    "kv_cache_quant_algo",
];
const SUBCONFIGS: &[&str] = &["text_config", "llm_config", "language_config"];

fn re(p: &str) -> regex::Regex {
    regex::Regex::new(p).unwrap_or_else(|_| std::process::abort())
}

fn fre(p: &str) -> fancy_regex::Regex {
    fancy_regex::Regex::new(p).unwrap_or_else(|_| std::process::abort())
}

// Python `$` (no MULTILINE) also matches before one final newline: `\n?\z` here.
static SPLIT: LazyLock<regex::Regex> =
    LazyLock::new(|| re(r"(?i)\A(?P<base>.+)-(?P<idx>[0-9]{5})-of-(?P<n>[0-9]{5})\.gguf\n?\z"));
static QUANT_IN_NAME: LazyLock<fancy_regex::Regex> = LazyLock::new(|| {
    fre(
        r"(?i)(?:^|[-_.])((?:UD-)?(?:I?Q[1-8](?:_[0-9A-Z]{1,4}){0,3}|BF16|F16|F32|FP16|FP8|MXFP4(?:_MOE)?|TQ[12]_0))(?=\n?\z|[-_.])",
    )
});
static HF_QUANT_IN_NAME: LazyLock<fancy_regex::Regex> = LazyLock::new(|| {
    fre(
        r"(?i)(?:^|[-_.])(NVFP4|MXFP4|FP8|AWQ|GPTQ|INT4|INT8|W4A16|W8A8|W4A4|BNB-4BIT|GGUF)(?=\n?\z|[-_.])",
    )
});
static REV: LazyLock<regex::Regex> = LazyLock::new(|| re(r"\A[0-9a-f]{40}\z"));
static DIGEST: LazyLock<regex::Regex> = LazyLock::new(|| re(r"\Asha256[:-]([0-9a-f]{64})\z"));
static PARAM_SIZE: LazyLock<regex::Regex> =
    LazyLock::new(|| re(r"(?i)\A\s*([0-9]+(?:\.[0-9]+)?)\s*([KMBT])\s*\z"));

// ── clocks ───────────────────────────────────────────────────────────────────────────────────────────────
static EPOCH: LazyLock<Instant> = LazyLock::new(Instant::now);

/// `time.monotonic()`
pub fn mono() -> f64 {
    EPOCH.elapsed().as_secs_f64()
}

/// `time.time()`
pub fn wall() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64())
}

// ── budgets ──────────────────────────────────────────────────────────────────────────────────────────────
/// The per-root scan budget ran out; the entries found so far are still used.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BudgetExceeded(pub String);

pub struct Budget {
    left: i64,
    limit: i64,
    deadline: f64,
    seconds: f64,
    clock: Clock,
}

impl Budget {
    pub fn new(entries: i64, seconds: f64) -> Self {
        Self::with_clock(entries, seconds, Arc::new(mono))
    }

    pub fn with_clock(entries: i64, seconds: f64, clock: Clock) -> Self {
        let deadline = clock() + seconds;
        Self {
            left: entries,
            limit: entries,
            deadline,
            seconds,
            clock,
        }
    }

    pub fn tick(&mut self, n: i64) -> Result<(), BudgetExceeded> {
        self.left -= n;
        if self.left < 0 {
            return Err(BudgetExceeded(format!(
                "stopped after {} directory entries (scan budget)",
                self.limit
            )));
        }
        if (self.clock)() > self.deadline {
            return Err(BudgetExceeded(format!(
                "stopped after {} s (scan budget)",
                ec_proto::pyfmt::fmt_g(self.seconds)
            )));
        }
        Ok(())
    }
}

impl Default for Budget {
    fn default() -> Self {
        Self::new(MAX_ENTRIES, MAX_SECONDS)
    }
}

type Scan<T> = Result<T, BudgetExceeded>;

// ── small helpers ────────────────────────────────────────────────────────────────────────────────────────
/// An empty MODEL_ENTRY with root, rel, format, name and warnings set (`_entry`).
fn entry(root: &str, rel: &str, fmt: &str, name: &str) -> Obj {
    let mut e = Obj::new();
    for k in MODEL_ENTRY {
        e.insert((*k).to_string(), Value::Null);
    }
    e.insert("root".into(), json!(root));
    e.insert("rel".into(), json!(rel));
    e.insert("format".into(), json!(fmt));
    e.insert("name".into(), json!(clean_text(name, 256)));
    e.insert("warnings".into(), json!([]));
    e
}

fn set(e: &mut Obj, k: &str, v: Value) {
    e.insert(k.to_string(), v);
}

fn opt_str(v: Option<String>) -> Value {
    v.map_or(Value::Null, Value::String)
}

fn opt_f(v: Option<f64>) -> Value {
    v.map_or(Value::Null, |f| json!(f))
}

fn opt_i(v: Option<i64>) -> Value {
    v.map_or(Value::Null, |i| json!(i))
}

/// st_mtime / st_atime as Python computes them (`sec + nsec * 1e-9`)
#[allow(clippy::cast_precision_loss)]
fn mtime_of(m: &Metadata) -> f64 {
    m.mtime() as f64 + m.mtime_nsec() as f64 * 1e-9
}

#[allow(clippy::cast_precision_loss)]
fn atime_of(m: &Metadata) -> f64 {
    m.atime() as f64 + m.atime_nsec() as f64 * 1e-9
}

fn size_of(m: &Metadata) -> i64 {
    i64::try_from(m.size()).unwrap_or(i64::MAX)
}

/// One directory entry (os.DirEntry).
#[derive(Debug, Clone)]
struct Ent {
    /// the name as text (U+FFFD for bytes that are not UTF-8)
    name: String,
    path: PathBuf,
    ft: Option<FileType>,
}

impl Ent {
    /// `is_dir(follow_symlinks=False)`
    fn is_dir(&self) -> bool {
        self.ft.is_some_and(|t| t.is_dir())
    }

    /// `is_file(follow_symlinks=False)`
    fn is_file(&self) -> bool {
        self.ft.is_some_and(|t| t.is_file())
    }

    fn is_symlink(&self) -> bool {
        self.ft.is_some_and(|t| t.is_symlink())
    }
}

fn list_dir(path: &Path, budget: Option<&RefCell<Budget>>) -> Scan<Vec<Ent>> {
    let Ok(rd) = std::fs::read_dir(path) else {
        return Ok(Vec::new());
    };
    let mut out = Vec::new();
    for d in rd {
        let Ok(d) = d else {
            return Ok(Vec::new()); // an OSError while iterating: Python returns nothing
        };
        if let Some(b) = budget {
            b.borrow_mut().tick(1)?;
        }
        out.push(Ent {
            name: d.file_name().to_string_lossy().into_owned(),
            path: d.path(),
            ft: d.file_type().ok(),
        });
    }
    Ok(out)
}

fn name_bytes(e: &Ent) -> &[u8] {
    e.path.file_name().map_or(&[][..], |n| n.as_bytes())
}

/// `_scandir`: the entries of a directory sorted by name (none when it cannot be read), each one ticking the
/// budget.
fn scandir(path: &Path, budget: &RefCell<Budget>) -> Scan<Vec<Ent>> {
    let mut out = list_dir(path, Some(budget))?;
    out.sort_by(|a, b| name_bytes(a).cmp(name_bytes(b)));
    Ok(out)
}

fn lstat(p: &Path) -> Option<Metadata> {
    std::fs::symlink_metadata(p).ok()
}

/// `_json`: bytes -> a JSON value, None when it does not parse.
fn json_of(data: Option<Vec<u8>>) -> Option<Value> {
    let d = data?;
    serde_json::from_str(&String::from_utf8_lossy(&d)).ok()
}

fn obj_of(v: Option<Value>) -> Option<Obj> {
    match v {
        Some(Value::Object(o)) => Some(o),
        _ => None,
    }
}

fn subset(cfg: &Obj, depth: usize) -> Obj {
    let mut out = Obj::new();
    for k in META_KEYS {
        if let Some(v) = cfg.get(*k) {
            out.insert((*k).to_string(), v.clone());
        }
    }
    if let Some(Value::Object(qc)) = cfg.get("quantization_config") {
        let mut q = Obj::new();
        for k in QUANT_KEYS {
            if let Some(v) = qc.get(*k) {
                q.insert((*k).to_string(), v.clone());
            }
        }
        out.insert("quantization_config".into(), Value::Object(q));
    }
    if depth == 0 {
        for sub in SUBCONFIGS {
            if let Some(Value::Object(inner)) = cfg.get(*sub) {
                out.insert((*sub).to_string(), Value::Object(subset(inner, 1)));
            }
        }
    }
    match clean_json(&Value::Object(out)) {
        Value::Object(o) => o,
        _ => Obj::new(),
    }
}

fn cfg_get<'a>(cfg: &'a Obj, key: &str) -> Option<&'a Value> {
    if let Some(v) = cfg.get(key) {
        return Some(v);
    }
    for sub in SUBCONFIGS {
        if let Some(Value::Object(inner)) = cfg.get(*sub)
            && let Some(v) = inner.get(key)
        {
            return Some(v);
        }
    }
    None
}

/// `str(v or "")`
fn str_or_empty(v: Option<&Value>) -> String {
    match v {
        Some(x) if py::truthy(x) => py::str_of(x),
        _ => String::new(),
    }
}

/// `isinstance(v, int)` (bools included), as `str()` prints it.
fn py_int(v: Option<&Value>) -> Option<String> {
    match v {
        Some(Value::Number(n)) if !n.is_f64() => Some(n.to_string()),
        Some(Value::Bool(b)) => Some(if *b { "True" } else { "False" }.to_string()),
        _ => None,
    }
}

/// fp8 / nvfp4 / awq-4bit / gptq-4bit / bnb-4bit / w4a16 ... from quantization_config, else from the name, else
/// the stored dtype (bf16, fp16, fp32).
pub fn quant_of_config(cfg: Option<&Obj>, name: &str) -> Option<String> {
    let mut qc = cfg
        .and_then(|c| c.get("quantization_config"))
        .and_then(Value::as_object);
    if qc.is_none()
        && let Some(c) = cfg
    {
        qc = cfg_get(c, "quantization_config").and_then(Value::as_object);
    }
    if let Some(qc) = qc {
        let method = str_or_empty(qc.get("quant_method")).to_lowercase();
        let bits = py_int(qc.get("bits"));
        if method == "fp8" || method == "fbgemm_fp8" {
            return Some("fp8".into());
        }
        if (method == "awq" || method == "gptq")
            && let Some(b) = bits
        {
            return Some(format!("{method}-{b}bit"));
        }
        if method == "bitsandbytes" {
            let four = qc.get("load_in_4bit").is_some_and(py::truthy);
            return Some(if four { "bnb-4bit" } else { "bnb-8bit" }.into());
        }
        if method == "modelopt" {
            let algo = str_or_empty(qc.get("quant_algo")).to_lowercase();
            return Some(if algo.is_empty() {
                "modelopt".into()
            } else {
                algo
            });
        }
        if method == "compressed-tensors" {
            let fmt = str_or_empty(qc.get("format")).to_lowercase();
            if fmt.contains("nvfp4") {
                return Some("nvfp4".into());
            }
            if fmt.contains("float") || fmt.contains("fp8") {
                return Some("fp8".into());
            }
            if let Some(Value::Object(groups)) = qc.get("config_groups") {
                for g in groups.values() {
                    let w = g
                        .as_object()
                        .and_then(|g| g.get("weights"))
                        .and_then(Value::as_object);
                    if let Some(w) = w
                        && let Some(nb) = py_int(w.get("num_bits"))
                    {
                        let fp = w.get("type") == Some(&json!("float"));
                        return Some(if fp {
                            format!("fp{nb}")
                        } else {
                            format!("w{nb}a16")
                        });
                    }
                }
            }
            return Some("compressed-tensors".into());
        }
        if !method.is_empty() {
            return Some(clean_text(&method, 32));
        }
    }
    if let Ok(Some(c)) = HF_QUANT_IN_NAME.captures(name)
        && let Some(m) = c.get(1)
    {
        return Some(m.as_str().to_lowercase());
    }
    let dtype = cfg.and_then(|c| {
        let t = cfg_get(c, "torch_dtype");
        if t.is_some_and(py::truthy) {
            t
        } else {
            cfg_get(c, "dtype")
        }
    });
    let s = dtype.map_or_else(|| "None".to_string(), py::str_of);
    match s.as_str() {
        "bfloat16" => Some("bf16".into()),
        "float16" => Some("fp16".into()),
        "float32" => Some("fp32".into()),
        _ => None,
    }
}

fn arch_ctx(cfg: Option<&Obj>) -> (Option<String>, Option<i64>) {
    let Some(cfg) = cfg else {
        return (None, None);
    };
    let first = cfg
        .get("architectures")
        .and_then(Value::as_array)
        .and_then(|a| a.first())
        .filter(|v| v.is_string());
    let arch = first.or_else(|| cfg.get("model_type"));
    let ctx = cfg_get(cfg, "max_position_embeddings")
        .filter(|v| v.is_i64() || v.is_u64())
        .and_then(Value::as_i64)
        .filter(|c| *c > 0 && *c < (1 << 31));
    (arch.and_then(Value::as_str).map(|a| clean_text(a, 64)), ctx)
}

/// A dimension as Python's `isinstance(dim, int) and dim >= 0` sees it (bools are 0/1).
fn dim_of(v: &Value) -> Option<u128> {
    match v {
        Value::Bool(b) => Some(u128::from(*b)),
        Value::Number(n) if !n.is_f64() => n.as_u64().map(u128::from),
        _ => None,
    }
}

/// Total parameters: the index's metadata.total_parameters, else the sum of tensor element counts read from each
/// file's header (bounded), which counts packed 4-bit weights wrongly, so callers skip it for those.
pub fn safetensors_params(
    root_real: &Path,
    files: &[PathBuf],
    index: Option<&Value>,
) -> Option<i64> {
    if let Some(Value::Object(index)) = index
        && let Some(Value::Object(meta)) = index.get("metadata")
        && let Some(Value::Number(tp)) = meta.get("total_parameters")
        && !tp.is_f64()
        && let Some(tp) = tp.as_i64()
        && tp > 0
    {
        return Some(tp);
    }
    if files.is_empty() || files.len() > MAX_SAFETENSORS_FILES {
        return None;
    }
    let mut total: u128 = 0;
    for path in files {
        let fd = open_inside(root_real, path)?;
        let mut f = std::fs::File::from(fd);
        let mut head = [0u8; 8];
        let mut got = 0;
        while got < 8 {
            match f.read(&mut head[got..]) {
                Ok(0) => break,
                Ok(n) => got += n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(_) => return None,
            }
        }
        if got != 8 {
            return None;
        }
        let n = u64::from_le_bytes(head);
        if n == 0 || n > SAFETENSORS_HEADER_MAX {
            return None;
        }
        let mut buf = vec![0u8; usize::try_from(n).ok()?];
        f.read_exact(&mut buf).ok()?;
        let Some(Value::Object(hdr)) = json_of(Some(buf)) else {
            return None;
        };
        if hdr.len() > SAFETENSORS_MAX_TENSORS {
            return None;
        }
        for (k, t) in &hdr {
            let shape = t
                .as_object()
                .and_then(|t| t.get("shape"))
                .and_then(Value::as_array);
            let Some(shape) = shape.filter(|_| k != "__metadata__") else {
                continue;
            };
            // bounded: a hostile header with millions of dims would otherwise cost minutes of CPU per file
            if shape.len() > SAFETENSORS_MAX_DIMS {
                return None;
            }
            let mut count: u128 = 1;
            for dim in shape {
                count = count.checked_mul(dim_of(dim)?)?;
                if count > PARAMS_MAX {
                    return None;
                }
            }
            total += count;
            if total > PARAMS_MAX {
                return None;
            }
        }
    }
    if total > 0 {
        i64::try_from(total).ok()
    } else {
        None
    }
}

/// (display name without the split suffix and quant token, quant)
pub fn gguf_name_quant(filename: &str, header: Option<&Obj>) -> (String, Option<String>) {
    let mut stem = if filename.to_lowercase().ends_with(".gguf") {
        let n = filename.chars().count();
        filename
            .chars()
            .take(n.saturating_sub(5))
            .collect::<String>()
    } else {
        filename.to_string()
    };
    if let Some(m) = SPLIT.captures(filename)
        && let Some(base) = m.name("base")
    {
        stem = base.as_str().to_string();
    }
    let mut quant: Option<String> = header
        .and_then(|h| h.get("quant"))
        .and_then(Value::as_str)
        .map(str::to_string);
    if let Ok(Some(qm)) = QUANT_IN_NAME.captures(&stem)
        && let (Some(whole), Some(g1)) = (qm.get(0), qm.get(1))
    {
        let token = g1.as_str().to_uppercase();
        if quant.as_deref().is_none_or(str::is_empty) {
            quant = Some(token.clone());
        }
        let q = quant
            .clone()
            .unwrap_or_else(|| "None".into())
            .to_uppercase();
        if token == q || token.replace("UD-", "") == q {
            let cut = format!("{}{}", &stem[..whole.start()], &stem[whole.end()..]);
            let cut = py::strip_chars(&cut, "-_. ");
            if !cut.is_empty() {
                stem = cut.to_string();
            }
        }
    }
    (stem, quant)
}

// ── per-kind scanners ────────────────────────────────────────────────────────────────────────────────────
struct Ctx {
    root: Root,
    real: PathBuf,
    budget: RefCell<Budget>,
}

impl Ctx {
    fn at(&self, rel: &str) -> PathBuf {
        if rel.is_empty() {
            self.real.clone()
        } else {
            py::join(&self.real, &[rel])
        }
    }
}

/// (rel, size, mtime, atime) and the file's path
#[derive(Debug, Clone)]
struct FileRec {
    rel: String,
    size: i64,
    mtime: f64,
    atime: f64,
    path: PathBuf,
}

fn hidden_or_secret(name: &str) -> bool {
    name.starts_with('.') || is_secret_name(name)
}

/// `_walk`: (rel of entry, entry) below ctx.real/rel, depth-first, no symlinked dirs, no hidden or secret names.
fn walk(
    ctx: &Ctx,
    rel: &str,
    depth: usize,
    max_depth: usize,
    f: &mut dyn FnMut(&str, &Ent),
) -> Scan<()> {
    for d in scandir(&ctx.at(rel), &ctx.budget)? {
        if hidden_or_secret(&d.name) {
            continue;
        }
        let r = if rel.is_empty() {
            d.name.clone()
        } else {
            format!("{rel}/{}", d.name)
        };
        f(&r, &d);
        if depth < max_depth && d.is_dir() {
            walk(ctx, &r, depth + 1, max_depth, f)?;
        }
    }
    Ok(())
}

fn fname_of(rel: &str) -> &str {
    rel.rsplit('/').next().unwrap_or(rel)
}

/// Group .gguf files into split sets (same dir, base and count) and singles.
fn gguf_groups(files: Vec<FileRec>) -> Vec<Vec<FileRec>> {
    let mut sets: Vec<((String, String, String), Vec<FileRec>)> = Vec::new();
    let mut singles: Vec<Vec<FileRec>> = Vec::new();
    for f in files {
        let (d, fname) = match f.rel.rsplit_once('/') {
            Some((d, n)) => (d.to_string(), n.to_string()),
            None => (String::new(), f.rel.clone()),
        };
        if let Some(m) = SPLIT.captures(&fname) {
            let key = (
                d,
                m.name("base").map_or("", |x| x.as_str()).to_string(),
                m.name("n").map_or("", |x| x.as_str()).to_string(),
            );
            match sets.iter_mut().find(|(k, _)| *k == key) {
                Some((_, v)) => v.push(f),
                None => sets.push((key, vec![f])),
            }
        } else {
            singles.push(vec![f]);
        }
    }
    for (_, mut v) in sets {
        v.sort_by(|a, b| {
            a.rel
                .cmp(&b.rel)
                .then(a.size.cmp(&b.size))
                .then(a.mtime.total_cmp(&b.mtime))
                .then(a.atime.total_cmp(&b.atime))
        });
        singles.push(v);
    }
    singles
}

/// GGUF facts of a file whose realpath is inside the root (never anything else).
pub fn gguf_header(root_real: &Path, path: &Path) -> Option<Obj> {
    let fd = open_inside(root_real, path)?;
    gguf::read_file(path, Some(fd), gguf::MAX_HEADER_BYTES, true)
}

fn hget(h: Option<&Obj>, k: &str) -> Value {
    h.and_then(|h| h.get(k)).cloned().unwrap_or(Value::Null)
}

fn max_f(it: impl Iterator<Item = f64>) -> f64 {
    it.fold(f64::NEG_INFINITY, f64::max)
}

fn gguf_entry(ctx: &Ctx, group: &[FileRec], name: Option<&str>, revision: Option<&str>) -> Obj {
    let first = &group[0];
    let fname = fname_of(&first.rel);
    let header = gguf_header(&ctx.real, &first.path);
    let h = header.as_ref();
    let mut warnings: Vec<String> = Vec::new();
    let m = SPLIT.captures(fname);
    let mut shards: Option<i64> = None;
    let mut params: Option<i64> = h.and_then(|h| h.get("params")).and_then(Value::as_i64);
    if let Some(m) = &m {
        let n: i64 = m
            .name("n")
            .and_then(|x| x.as_str().parse().ok())
            .unwrap_or(0);
        let idx: i64 = m
            .name("idx")
            .and_then(|x| x.as_str().parse().ok())
            .unwrap_or(0);
        shards = Some(n);
        if i64::try_from(group.len()).unwrap_or(0) != n {
            warnings.push(format!("split set has {} of {n} files", group.len()));
        }
        if idx != 1 {
            warnings.push("the first file of the split set is missing".into());
        }
        // every shard lists its own tensors: add them up for the parameter count
        for f in &group[1..] {
            let h2 = gguf_header(&ctx.real, &f.path);
            let p2 = h2
                .as_ref()
                .and_then(|h| h.get("params"))
                .and_then(Value::as_i64);
            params = match (params, p2) {
                (Some(a), Some(b)) => a.checked_add(b),
                _ => None,
            };
        }
    }
    match h {
        None => warnings.push("could not read the GGUF header".into()),
        Some(h) if !h.get("complete").is_some_and(py::truthy) => {
            warnings.push("GGUF header only partly read".into());
        }
        Some(_) => {}
    }
    let (disp, quant) = gguf_name_quant(fname, h);
    let mut meta = Obj::new();
    if let Some(h) = h {
        for k in [
            "version",
            "name",
            "size_label",
            "block_count",
            "embedding_length",
            "head_count",
            "head_count_kv",
            "expert_count",
            "expert_used_count",
            "file_type",
        ] {
            if let Some(v) = h.get(k).filter(|v| !v.is_null()) {
                meta.insert(k.to_string(), v.clone());
            }
        }
    }
    if fname.to_lowercase().contains("mmproj")
        || h.is_some_and(|h| h.get("arch") == Some(&json!("clip")))
    {
        meta.insert("role".into(), json!("mmproj"));
    }
    if m.is_some() {
        let files: Vec<Value> = group
            .iter()
            .take(64)
            .map(|f| json!(fname_of(&f.rel)))
            .collect();
        meta.insert("files".into(), Value::Array(files));
    }
    let mut e = entry(&ctx.root.id, &first.rel, "gguf", name.unwrap_or(&disp));
    set(&mut e, "revision", opt_str(revision.map(str::to_string)));
    set(
        &mut e,
        "size",
        json!(group.iter().map(|f| f.size).sum::<i64>()),
    );
    set(&mut e, "files", json!(group.len()));
    set(&mut e, "mtime", json!(max_f(group.iter().map(|f| f.mtime))));
    set(&mut e, "atime", json!(max_f(group.iter().map(|f| f.atime))));
    set(&mut e, "quant", opt_str(quant));
    set(&mut e, "params", opt_i(params));
    set(&mut e, "arch", hget(h, "arch"));
    set(&mut e, "context", hget(h, "context"));
    set(&mut e, "shards", opt_i(shards));
    set(&mut e, "meta", clean_json(&Value::Object(meta)));
    set(&mut e, "warnings", json!(warnings));
    e
}

fn scan_gguf(ctx: &Ctx, sink: &mut Vec<Obj>, base: &str, max_depth: usize) -> Scan<()> {
    let mut files: Vec<FileRec> = Vec::new();
    let res = walk(ctx, base, 1, max_depth, &mut |rel, d| {
        if !d.name.to_lowercase().ends_with(".gguf") {
            return;
        }
        let Ok(st) = std::fs::metadata(&d.path) else {
            return;
        };
        if !st.is_file() {
            return;
        }
        if d.is_symlink() && !inside(&py::realpath(&d.path), &ctx.real) {
            return;
        }
        files.push(FileRec {
            rel: rel.to_string(),
            size: size_of(&st),
            mtime: mtime_of(&st),
            atime: atime_of(&st),
            path: d.path.clone(),
        });
    });
    // the budget may have run out: still list what was found
    for g in gguf_groups(files) {
        sink.push(gguf_entry(ctx, &g, None, None));
    }
    res
}

/// (bytes, files, newest mtime, newest atime, complete) of a directory tree (no symlinks followed).
fn dir_size(ctx: &Ctx, path: &Path, max_depth: usize) -> (i64, i64, f64, f64, bool) {
    let (mut total, mut count) = (0i64, 0i64);
    let (mut mtime, mut atime) = (0.0f64, 0.0f64);
    let mut stack: Vec<(PathBuf, usize)> = vec![(path.to_path_buf(), 0)];
    while let Some((p, depth)) = stack.pop() {
        let Ok(ents) = scandir(&p, &ctx.budget) else {
            return (total, count, mtime, atime, false);
        };
        for d in ents {
            if is_secret_name(&d.name) {
                continue;
            }
            let Some(st) = lstat(&d.path) else { continue };
            if st.is_dir() && depth < max_depth {
                stack.push((d.path.clone(), depth + 1));
            } else if st.file_type().is_file() {
                total += size_of(&st);
                count += 1;
                mtime = mtime.max(mtime_of(&st));
                atime = atime.max(atime_of(&st));
            }
        }
    }
    (total, count, mtime, atime, true)
}

/// `[n for n in os.listdir(path) if not is_secret_name(n)][:10000]`
fn names(path: &Path) -> Vec<String> {
    match list_dir(path, None) {
        Ok(v) => v
            .into_iter()
            .map(|e| e.name)
            .filter(|n| !is_secret_name(n))
            .take(10_000)
            .collect(),
        Err(_) => Vec::new(),
    }
}

fn has_weight_ext(n: &str) -> bool {
    WEIGHT_EXT.iter().any(|x| n.ends_with(x))
}

/// Large regular files that are not a known weight format (a custom checkpoint format, e.g. `*.ftw`).
fn other_weights(path: &Path, names: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    for n in names.iter().take(2000) {
        if has_weight_ext(n) || NOT_WEIGHTS.iter().any(|x| n.ends_with(x)) {
            continue;
        }
        let Some(st) = lstat(&py::join(path, &[n])) else {
            continue;
        };
        if st.file_type().is_file() && st.size() >= BIG_FILE {
            out.push(n.clone());
        }
    }
    out
}

/// config.json plus weights (weights in an unknown format still count, with a warning on the entry).
fn is_model_dir(path: &Path) -> bool {
    let names = names(path);
    if !names.iter().any(|n| n == "config.json") {
        return false;
    }
    names.iter().any(|n| has_weight_ext(n)) || !other_weights(path, &names).is_empty()
}

fn unpacked(quant: Option<&str>) -> bool {
    matches!(quant, None | Some("fp8" | "bf16" | "fp16" | "fp32"))
}

fn count_or_none(n: usize) -> Value {
    if n > 0 { json!(n) } else { Value::Null }
}

fn plain_entry(ctx: &Ctx, rel: &str, path: &Path) -> Obj {
    let cfg = obj_of(json_of(read_inside(
        &ctx.real,
        &py::join(path, &["config.json"]),
        CONFIG_MAX,
    )));
    let (size, count, mtime, atime, complete) = dir_size(ctx, path, 8);
    let name = if rel.is_empty() {
        py::basename(&ctx.real).to_string_lossy().into_owned()
    } else {
        fname_of(rel).to_string()
    };
    let quant = quant_of_config(cfg.as_ref(), &name);
    let (arch, context) = arch_ctx(cfg.as_ref());
    let all = names(path);
    let mut st_files: Vec<PathBuf> = all
        .iter()
        .filter(|f| f.ends_with(".safetensors"))
        .map(|f| py::join(path, &[f]))
        .collect();
    st_files.sort_by(|a, b| a.as_os_str().as_bytes().cmp(b.as_os_str().as_bytes()));
    let index = json_of(read_inside(
        &ctx.real,
        &py::join(path, &["model.safetensors.index.json"]),
        8 << 20,
    ));
    let params = if unpacked(quant.as_deref()) {
        safetensors_params(&ctx.real, &st_files, index.as_ref())
    } else {
        None
    };
    let mut warnings: Vec<String> = if complete {
        Vec::new()
    } else {
        vec!["size is a lower bound: the scan budget ran out inside this directory".into()]
    };
    let names = names(path);
    if !names.iter().any(|n| has_weight_ext(n)) {
        let mut exts: Vec<String> = other_weights(path, &names)
            .iter()
            .map(|n| {
                let e = py::splitext_ext(n);
                if e.is_empty() {
                    n.clone()
                } else {
                    e.to_string()
                }
            })
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        exts.sort();
        exts.truncate(4);
        warnings.push(format!(
            "the weights are in a format the common engines do not load ({}); listed so its disk space is visible",
            exts.join(", ")
        ));
    }
    let mut e = entry(&ctx.root.id, rel, "plain", &name);
    set(&mut e, "size", json!(size));
    set(&mut e, "files", json!(count));
    set(&mut e, "mtime", opt_f(Some(mtime).filter(|m| *m != 0.0)));
    set(&mut e, "atime", opt_f(Some(atime).filter(|a| *a != 0.0)));
    set(&mut e, "quant", opt_str(quant));
    set(&mut e, "params", opt_i(params));
    set(&mut e, "arch", opt_str(arch));
    set(&mut e, "context", opt_i(context));
    set(&mut e, "shards", count_or_none(st_files.len()));
    set(
        &mut e,
        "meta",
        match &cfg {
            Some(c) if !c.is_empty() => Value::Object(subset(c, 0)),
            _ => Value::Null,
        },
    );
    set(&mut e, "warnings", json!(warnings));
    e
}

fn scan_plain(ctx: &Ctx, sink: &mut Vec<Obj>, max_depth: usize) -> Scan<()> {
    if is_model_dir(&ctx.real) {
        sink.push(plain_entry(ctx, "", &ctx.real));
        return Ok(());
    }
    let mut stack: Vec<(String, usize)> = vec![(String::new(), 0)];
    while let Some((rel, depth)) = stack.pop() {
        for d in scandir(&ctx.at(&rel), &ctx.budget)? {
            if hidden_or_secret(&d.name) || !d.is_dir() {
                continue;
            }
            let r = if rel.is_empty() {
                d.name.clone()
            } else {
                format!("{rel}/{}", d.name)
            };
            if is_model_dir(&d.path) {
                sink.push(plain_entry(ctx, &r, &d.path));
            } else if depth + 1 < max_depth {
                stack.push((r, depth + 1));
            }
        }
    }
    Ok(())
}

fn scan_other(ctx: &Ctx, sink: &mut Vec<Obj>) -> Scan<()> {
    for d in scandir(&ctx.real, &ctx.budget)? {
        if hidden_or_secret(&d.name) || !d.is_dir() {
            continue;
        }
        let (size, count, mtime, atime, complete) = dir_size(ctx, &d.path, 32);
        let mut e = entry(&ctx.root.id, &d.name, "other", &d.name);
        set(&mut e, "size", json!(size));
        set(&mut e, "files", json!(count));
        set(&mut e, "mtime", opt_f(Some(mtime).filter(|m| *m != 0.0)));
        set(&mut e, "atime", opt_f(Some(atime).filter(|a| *a != 0.0)));
        let w: Vec<&str> = if complete {
            Vec::new()
        } else {
            vec!["size is a lower bound: the scan budget ran out"]
        };
        set(&mut e, "warnings", json!(w));
        sink.push(e);
        if !complete {
            return Err(BudgetExceeded("the scan budget ran out".into()));
        }
    }
    Ok(())
}

// ── Hugging Face cache ───────────────────────────────────────────────────────────────────────────────────
fn scan_hf(ctx: &Ctx, sink: &mut Vec<Obj>, base: &str) -> Scan<()> {
    for d in scandir(&ctx.at(base), &ctx.budget)? {
        if d.name.starts_with("models--") && d.is_dir() {
            let rel = if base.is_empty() {
                d.name.clone()
            } else {
                format!("{base}/{}", d.name)
            };
            let entries = hf_repo(ctx, &d.path, &rel, &d.name)?;
            sink.extend(entries);
        }
    }
    Ok(())
}

/// (revision to show, warnings): refs/main when its snapshot exists, else the newest snapshot.
fn hf_snapshot(ctx: &Ctx, repo: &Path) -> Scan<(Option<String>, Vec<String>)> {
    let mut warnings: Vec<String> = Vec::new();
    let raw = read_inside(&ctx.real, &py::join(repo, &["refs", "main"]), 256).unwrap_or_default();
    let ascii: String = raw
        .iter()
        .map(|b| {
            if b.is_ascii() {
                char::from(*b)
            } else {
                '\u{fffd}'
            }
        })
        .collect();
    let r = py::strip(&ascii).to_string();
    let snaps: Vec<Ent> = scandir(&py::join(repo, &["snapshots"]), &ctx.budget)?
        .into_iter()
        .filter(Ent::is_dir)
        .collect();
    if !r.is_empty() && REV.is_match(&r) && snaps.iter().any(|d| d.name == r) {
        if snaps.len() > 1 {
            warnings.push(format!("{} revisions on disk", snaps.len()));
        }
        return Ok((Some(r), warnings));
    }
    if snaps.is_empty() {
        return Ok((
            None,
            vec!["no snapshot: the download has not finished (or was removed)".into()],
        ));
    }
    // max(snaps, key=mtime): the first of the newest
    let mut newest = &snaps[0];
    let mut best = lstat(&newest.path).map_or(0.0, |m| mtime_of(&m));
    for d in &snaps[1..] {
        let t = lstat(&d.path).map_or(0.0, |m| mtime_of(&m));
        if t > best {
            best = t;
            newest = d;
        }
    }
    warnings.push("refs/main is missing or points nowhere: showing the newest snapshot".into());
    Ok((Some(newest.name.clone()), warnings))
}

fn hf_repo(ctx: &Ctx, repo: &Path, rel: &str, dirname: &str) -> Scan<Vec<Obj>> {
    let repo_id = dirname["models--".len()..].replace("--", "/");
    let (rev, mut warnings) = hf_snapshot(ctx, repo)?;
    let blobs_dir = py::join(repo, &["blobs"]);
    let mut blobs: HashMap<String, Metadata> = HashMap::new();
    let mut incomplete = 0;
    for d in scandir(&blobs_dir, &ctx.budget)? {
        let Some(bst) = lstat(&d.path) else { continue };
        if !bst.file_type().is_file() {
            continue;
        }
        if d.name.ends_with(".incomplete") {
            incomplete += 1;
        } else {
            blobs.insert(d.name.clone(), bst);
        }
    }
    if incomplete > 0 {
        warnings.push(format!("{incomplete} file(s) still downloading"));
    }
    let disk: i64 = blobs.values().map(size_of).sum();
    let Some(rev) = rev else {
        let mut e = entry(&ctx.root.id, rel, "hf", &repo_id);
        set(&mut e, "size", json!(disk));
        set(&mut e, "files", json!(0));
        set(&mut e, "warnings", json!(warnings));
        set(&mut e, "meta", json!({"disk_bytes": disk}));
        return Ok(vec![e]);
    };
    let snap = py::join(repo, &["snapshots", rev.as_str()]);
    let mut files: Vec<FileRec> = Vec::new();
    let mut used: HashSet<String> = HashSet::new();
    let mut size: i64 = 0;
    let mut stack: Vec<(String, usize)> = vec![(String::new(), 0)];
    while let Some((sub, depth)) = stack.pop() {
        let dir = if sub.is_empty() {
            snap.clone()
        } else {
            py::join(&snap, &[&sub])
        };
        for d in scandir(&dir, &ctx.budget)? {
            let r = if sub.is_empty() {
                d.name.clone()
            } else {
                format!("{sub}/{}", d.name)
            };
            if is_secret_name(&d.name) {
                continue;
            }
            if d.is_dir() {
                if depth < 6 {
                    stack.push((r, depth + 1));
                }
                continue;
            }
            let mut st: Option<Metadata> = None;
            let mut blob: Option<String> = None;
            if d.is_symlink() {
                let Ok(link) = std::fs::read_link(&d.path) else {
                    continue;
                };
                let target = py::normpath(&py::join(&py::dirname(&d.path), &[link.as_os_str()]));
                let tname = py::basename(&target).to_string_lossy().into_owned();
                if py::dirname(&target).as_os_str().as_bytes() == blobs_dir.as_os_str().as_bytes()
                    && blobs.contains_key(&tname)
                {
                    st = blobs.get(&tname).cloned();
                    blob = Some(tname);
                } else {
                    let real = py::realpath(&d.path);
                    if inside(&real, &ctx.real) {
                        st = lstat(&real);
                    }
                }
            } else {
                st = lstat(&d.path);
            }
            let Some(st) = st.filter(|s| s.file_type().is_file()) else {
                if warnings.len() < 8 {
                    warnings.push(format!("{} points to a missing blob", py::take(&r, 120)));
                }
                continue;
            };
            if blob.as_ref().is_none_or(|b| !used.contains(b)) {
                size += size_of(&st);
                if let Some(b) = blob {
                    used.insert(b);
                }
            }
            files.push(FileRec {
                rel: r,
                size: size_of(&st),
                mtime: mtime_of(&st),
                atime: atime_of(&st),
                path: d.path.clone(),
            });
        }
    }
    let ggufs: Vec<FileRec> = files
        .iter()
        .filter(|f| f.rel.to_lowercase().ends_with(".gguf"))
        .cloned()
        .collect();
    let has_weights = files.iter().any(|f| has_weight_ext(&f.rel));
    if !ggufs.is_empty() && !has_weights {
        let n = files.len();
        return Ok(vec![hf_gguf_entry(
            ctx,
            rel,
            &repo_id,
            &rev,
            ggufs,
            (size, n),
            warnings,
        )]);
    }
    let cfg = obj_of(json_of(read_inside(
        &ctx.real,
        &py::join(&snap, &["config.json"]),
        CONFIG_MAX,
    )));
    if cfg.is_none() {
        warnings.push("no config.json in the snapshot".into());
    }
    let quant = quant_of_config(cfg.as_ref(), &repo_id);
    let (arch, context) = arch_ctx(cfg.as_ref());
    let mut st_files: Vec<PathBuf> = files
        .iter()
        .filter(|f| f.rel.ends_with(".safetensors") && !f.rel.contains('/'))
        .map(|f| py::join(&snap, &[&f.rel]))
        .collect();
    st_files.sort_by(|a, b| a.as_os_str().as_bytes().cmp(b.as_os_str().as_bytes()));
    let index = json_of(read_inside(
        &ctx.real,
        &py::join(&snap, &["model.safetensors.index.json"]),
        8 << 20,
    ));
    let params = if unpacked(quant.as_deref()) {
        safetensors_params(&ctx.real, &st_files, index.as_ref())
    } else {
        None
    };
    let mut meta: Obj = match &cfg {
        Some(c) if !c.is_empty() => subset(c, 0),
        _ => Obj::new(),
    };
    if disk != size {
        meta.insert("disk_bytes".into(), json!(disk));
    }
    let mut e = entry(&ctx.root.id, rel, "hf", &repo_id);
    set(&mut e, "revision", json!(rev));
    set(&mut e, "size", json!(size));
    set(&mut e, "files", json!(files.len()));
    let mt = (!files.is_empty()).then(|| max_f(files.iter().map(|f| f.mtime)));
    let at = (!files.is_empty()).then(|| max_f(files.iter().map(|f| f.atime)));
    set(&mut e, "mtime", opt_f(mt));
    set(&mut e, "atime", opt_f(at));
    set(&mut e, "quant", opt_str(quant));
    set(&mut e, "params", opt_i(params));
    set(&mut e, "arch", opt_str(arch));
    set(&mut e, "context", opt_i(context));
    set(&mut e, "shards", count_or_none(st_files.len()));
    set(
        &mut e,
        "meta",
        if meta.is_empty() {
            Value::Null
        } else {
            clean_json(&Value::Object(meta))
        },
    );
    warnings.truncate(8);
    set(&mut e, "warnings", json!(warnings));
    Ok(vec![e])
}

/// One entry for a GGUF-only HF repo: rel is the repo directory like every HF entry (the hub mounts the cache);
/// each quant file (or split set) is listed in meta.gguf_files with its path below the repo directory.
/// `totals` = (size, files) of the whole snapshot.
fn hf_gguf_entry(
    ctx: &Ctx,
    rel: &str,
    repo_id: &str,
    rev: &str,
    ggufs: Vec<FileRec>,
    totals: (i64, usize),
    mut warnings: Vec<String>,
) -> Obj {
    let mtime = max_f(ggufs.iter().map(|f| f.mtime));
    let atime = max_f(ggufs.iter().map(|f| f.atime));
    let groups = gguf_groups(
        ggufs
            .into_iter()
            .map(|f| FileRec {
                rel: format!("snapshots/{rev}/{}", f.rel),
                ..f
            })
            .collect(),
    );
    let mut listed: Vec<Obj> = Vec::new();
    let mut first: Option<Obj> = None;
    for g in groups.iter().take(32) {
        let fname = fname_of(&g[0].rel);
        let header = gguf_header(&ctx.real, &g[0].path);
        if first.is_none() {
            first.clone_from(&header);
        }
        let (_name, quant) = gguf_name_quant(fname, header.as_ref());
        let m = SPLIT.captures(fname);
        let shards: Option<i64> = m.and_then(|m| m.name("n").and_then(|x| x.as_str().parse().ok()));
        let role = fname.to_lowercase().contains("mmproj").then_some("mmproj");
        let mut o = Obj::new();
        o.insert("rel".into(), json!(g[0].rel));
        o.insert("quant".into(), opt_str(quant));
        o.insert("size".into(), json!(g.iter().map(|f| f.size).sum::<i64>()));
        o.insert("files".into(), json!(g.len()));
        o.insert("shards".into(), opt_i(shards));
        o.insert("role".into(), role.map_or(Value::Null, |r| json!(r)));
        listed.push(o);
    }
    let quants: Vec<Value> = listed
        .iter()
        .filter(|x| x.get("role") == Some(&Value::Null))
        .map(|x| x.get("quant").cloned().unwrap_or(Value::Null))
        .collect();
    if groups.len() > 32 {
        warnings.push(format!(
            "{} GGUF files or sets: only the first 32 are listed",
            groups.len()
        ));
    }
    let f = first.as_ref();
    let one = quants.len() == 1;
    let mut e = entry(&ctx.root.id, rel, "gguf", repo_id);
    set(&mut e, "revision", json!(rev));
    set(&mut e, "size", json!(totals.0));
    set(&mut e, "files", json!(totals.1));
    set(&mut e, "mtime", json!(mtime));
    set(&mut e, "atime", json!(atime));
    set(
        &mut e,
        "quant",
        if one { quants[0].clone() } else { Value::Null },
    );
    set(
        &mut e,
        "params",
        if one { hget(f, "params") } else { Value::Null },
    );
    set(&mut e, "arch", hget(f, "arch"));
    set(&mut e, "context", hget(f, "context"));
    let listed: Vec<Value> = listed.into_iter().map(Value::Object).collect();
    set(
        &mut e,
        "meta",
        clean_json(&json!({"hf_repo": repo_id, "gguf_files": listed})),
    );
    warnings.truncate(8);
    set(&mut e, "warnings", json!(warnings));
    e
}

// ── Ollama ───────────────────────────────────────────────────────────────────────────────────────────────
fn scan_ollama(ctx: &Ctx, sink: &mut Vec<Obj>, base: &str) -> Scan<()> {
    let models = ctx.at(base);
    let manifests = py::join(&models, &["manifests"]);
    let mut stack: Vec<(PathBuf, Vec<String>)> = vec![(manifests, Vec::new())];
    let mut seen: Vec<(PathBuf, Vec<String>)> = Vec::new();
    let mut res: Scan<()> = Ok(());
    while let Some((path, parts)) = stack.pop() {
        let ents = match scandir(&path, &ctx.budget) {
            Ok(e) => e,
            Err(e) => {
                res = Err(e);
                break;
            }
        };
        for d in ents {
            if hidden_or_secret(&d.name) {
                continue;
            }
            let mut p = parts.clone();
            p.push(d.name.clone());
            if d.is_dir() {
                if parts.len() < 5 {
                    stack.push((d.path.clone(), p));
                }
            } else if d.is_file() && parts.len() >= 2 {
                // <registry>/<namespace>/<model>/<tag>
                seen.push((d.path.clone(), p));
            }
        }
    }
    // finally: what was found is listed even when the budget ran out
    seen.sort_by(|a, b| {
        a.0.as_os_str()
            .as_bytes()
            .cmp(b.0.as_os_str().as_bytes())
            .then_with(|| a.1.cmp(&b.1))
    });
    for (path, parts) in &seen {
        if let Some(e) = ollama_entry(ctx, &models, path, parts) {
            sink.push(e);
        }
    }
    res
}

/// name:tag from a manifest path below manifests/: library models short, others with their namespace.
pub fn ollama_name(parts: &[String]) -> String {
    let n = parts.len();
    let (registry, namespace, model, tag) = if n >= 4 {
        (
            parts[0].clone(),
            parts[1..n - 2].join("/"),
            parts[n - 2].clone(),
            parts[n - 1].clone(),
        )
    } else {
        let k = n.saturating_sub(3);
        let g = |i: usize| parts.get(k + i).cloned().unwrap_or_default();
        (String::new(), g(0), g(1), g(2))
    };
    let default_registry = registry == "registry.ollama.ai" || registry.is_empty();
    if default_registry && namespace == "library" {
        format!("{model}:{tag}")
    } else if default_registry {
        format!("{namespace}/{model}:{tag}")
    } else {
        format!("{registry}/{namespace}/{model}:{tag}")
    }
}

fn params_from_label(label: Option<&Value>) -> Option<i64> {
    let s = label?.as_str()?;
    let m = PARAM_SIZE.captures(s)?;
    let num: f64 = m.get(1)?.as_str().parse().ok()?;
    let mult = match m.get(2)?.as_str().to_ascii_uppercase().as_str() {
        "K" => 1e3,
        "M" => 1e6,
        "B" => 1e9,
        "T" => 1e12,
        _ => return None,
    };
    let v = (num * mult).trunc();
    if v.is_finite() && v.abs() < 9.2e18 {
        #[allow(clippy::cast_possible_truncation)]
        Some(v as i64)
    } else {
        None
    }
}

fn ollama_entry(ctx: &Ctx, models: &Path, path: &Path, parts: &[String]) -> Option<Obj> {
    let Some(Value::Object(doc)) = json_of(read_inside(&ctx.real, path, MANIFEST_MAX)) else {
        return None;
    };
    let empty = Vec::new();
    let layers = doc
        .get("layers")
        .and_then(Value::as_array)
        .unwrap_or(&empty);
    let blobs = py::join(models, &["blobs"]);
    let mut size: i64 = 0;
    let mut warnings: Vec<String> = Vec::new();
    let mut model_blob: Option<PathBuf> = None;
    let mut atime: Option<f64> = None;
    for layer in layers.iter().take(64) {
        let Value::Object(layer) = layer else {
            continue;
        };
        let digest = str_or_empty(layer.get("digest"));
        let Some(m) = DIGEST.captures(&digest) else {
            continue;
        };
        let hex = m.get(1).map_or("", |x| x.as_str()).to_string();
        let blob = py::join(&blobs, &[format!("sha256-{hex}")]);
        let st = lstat(&blob);
        match st.as_ref().filter(|s| s.file_type().is_file()) {
            Some(s) => size += size_of(s),
            None => {
                size += match layer.get("size") {
                    Some(Value::Bool(true)) => 1,
                    Some(Value::Number(n)) if !n.is_f64() => {
                        n.as_i64().filter(|v| *v > 0).unwrap_or(0)
                    }
                    _ => 0,
                };
                warnings.push(format!("layer blob sha256-{}... is missing", &hex[..12]));
            }
        }
        let media = layer.get("mediaType").map_or_else(String::new, py::str_of);
        if media.ends_with(".model") {
            model_blob = Some(blob);
            atime = st.as_ref().map(atime_of);
        }
    }
    let cfg_digest = match doc.get("config") {
        Some(Value::Object(c)) => DIGEST
            .captures(&str_or_empty(c.get("digest")))
            .and_then(|m| m.get(1).map(|x| x.as_str().to_string())),
        _ => None,
    };
    let facts: Obj = cfg_digest
        .and_then(|d| {
            obj_of(json_of(read_inside(
                &ctx.real,
                &py::join(&blobs, &[format!("sha256-{d}")]),
                64 * 1024,
            )))
        })
        .unwrap_or_default();
    let header = model_blob.as_ref().and_then(|b| gguf_header(&ctx.real, b));
    let h = header.as_ref();
    let rel = py::path_str(&py::relpath(path, &ctx.real));
    let st = lstat(path);
    let mut params = h.and_then(|h| h.get("params")).and_then(Value::as_i64);
    if params.is_none_or(|p| p == 0) {
        params = params_from_label(facts.get("model_type"));
    }
    let mut meta = Obj::new();
    for k in [
        "model_format",
        "model_family",
        "model_families",
        "model_type",
        "file_type",
    ] {
        if let Some(v) = facts.get(k) {
            meta.insert(k.to_string(), v.clone());
        }
    }
    if let Some(h) = h {
        for k in [
            "block_count",
            "embedding_length",
            "head_count",
            "head_count_kv",
            "expert_count",
            "expert_used_count",
        ] {
            if let Some(v) = h.get(k).filter(|v| !v.is_null()) {
                meta.insert(k.to_string(), v.clone());
            }
        }
    }
    let rev = model_blob
        .as_ref()
        .map(|b| py::basename(b).to_string_lossy().into_owned())
        .and_then(|n| {
            DIGEST
                .captures(&n)
                .and_then(|m| m.get(1).map(|x| x.as_str()[..12].to_string()))
        });
    let quant = match facts.get("file_type") {
        Some(Value::String(s)) => Value::String(clean_text(s, 32)),
        _ => hget(h, "quant"),
    };
    let arch = match facts.get("model_family") {
        Some(Value::String(s)) => Value::String(clean_text(s, 64)),
        _ => hget(h, "arch"),
    };
    let mut e = entry(&ctx.root.id, &rel, "ollama", &ollama_name(parts));
    set(&mut e, "revision", opt_str(rev));
    set(&mut e, "size", json!(size));
    set(&mut e, "files", json!(layers.len()));
    set(&mut e, "mtime", opt_f(st.as_ref().map(mtime_of)));
    set(&mut e, "atime", opt_f(atime));
    set(&mut e, "quant", quant);
    set(&mut e, "params", opt_i(params));
    set(&mut e, "arch", arch);
    set(&mut e, "context", hget(h, "context"));
    let meta = clean_json(&Value::Object(meta));
    set(
        &mut e,
        "meta",
        if py::truthy(&meta) { meta } else { Value::Null },
    );
    warnings.truncate(8);
    set(&mut e, "warnings", json!(warnings));
    Some(e)
}

// ── a whole root ─────────────────────────────────────────────────────────────────────────────────────────
/// [(kind, base rel)] for an `auto` root.
fn detect(real: &Path, budget: &RefCell<Budget>) -> Scan<Vec<(&'static str, &'static str)>> {
    if py::join(real, &["hub"]).is_dir() {
        return Ok(vec![("hf-home", "hub")]);
    }
    if scandir(real, budget)?
        .iter()
        .any(|d| d.name.starts_with("models--"))
    {
        return Ok(vec![("hf-hub", "")]);
    }
    if py::join(real, &["models", "manifests"]).is_dir() {
        return Ok(vec![("ollama", "models")]);
    }
    if py::join(real, &["manifests"]).is_dir() {
        return Ok(vec![("ollama", "")]);
    }
    Ok(vec![("plain", ""), ("gguf", "")])
}

/// Why scan_root produced no result (Python: FileNotFoundError, another OSError, or a BudgetExceeded raised
/// outside the per-kind scanners, which the background scanner treats as "no answer").
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScanError {
    Missing,
    Os(i32),
    Budget(String),
}

pub type ScanResult = Result<(Vec<Obj>, Option<String>), ScanError>;

/// (every model entry of one root, why the scan stopped early or None).
pub fn scan_root(root: &Root, budget: Budget) -> ScanResult {
    let real = py::realpath(Path::new(&root.path));
    if !real.is_dir() {
        return Err(ScanError::Missing);
    }
    let ctx = Ctx {
        root: root.clone(),
        real,
        budget: RefCell::new(budget),
    };
    let kinds: Vec<(String, String)> = match root.kind.as_str() {
        "auto" => detect(&ctx.real, &ctx.budget)
            .map_err(|e| ScanError::Budget(e.0))?
            .into_iter()
            .map(|(k, b)| (k.to_string(), b.to_string()))
            .collect(),
        "hf-home" => vec![("hf-home".into(), "hub".into())],
        "ollama" => {
            let base = if py::join(&ctx.real, &["models", "manifests"]).is_dir() {
                "models"
            } else {
                ""
            };
            vec![("ollama".into(), base.into())]
        }
        k => vec![(k.to_string(), String::new())],
    };
    let mut found: Vec<Obj> = Vec::new();
    for (kind, base) in &kinds {
        let r = match kind.as_str() {
            "hf-home" | "hf-hub" => scan_hf(&ctx, &mut found, base),
            "gguf" => scan_gguf(&ctx, &mut found, "", 6),
            "plain" => scan_plain(&ctx, &mut found, 3),
            "ollama" => scan_ollama(&ctx, &mut found, base),
            _ => scan_other(&ctx, &mut found),
        };
        if let Err(BudgetExceeded(why)) = r {
            return Ok((dedupe(found), Some(why)));
        }
    }
    Ok((dedupe(found), None))
}

fn dedupe(entries: Vec<Obj>) -> Vec<Obj> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut out = Vec::new();
    for e in entries {
        let key = e.get("rel").map_or_else(|| "None".to_string(), py::str_of);
        if seen.insert(key) {
            out.push(e);
        }
    }
    out
}

/// models.config: the bounded config.json of a model (an HF repo dir, a plain dir, or a config.json path); for a
/// .gguf file, the facts from its header.
pub fn model_config(
    policy: &Policy,
    root_id: &Value,
    rel: Option<&str>,
) -> Result<Obj, HandlerError> {
    let h = open_root(policy, root_id, false)?;
    let parts = split_rel(rel, "path")?;
    let rel = rel.unwrap_or("");
    let path = h.path(&parts);
    let last = parts.last().map_or("", String::as_str);
    let mut out = Obj::new();
    if last.to_lowercase().ends_with(".gguf") {
        let Some(header) = gguf_header(&h.real, &path) else {
            return Err(OpError::new(
                NOT_FOUND,
                format!(
                    "{} is not a readable GGUF file in model root {}",
                    py::repr_take(rel, 300),
                    py::repr(&h.root.id)
                ),
            )
            .into());
        };
        out.insert("config".into(), clean_json(&json!({"gguf": header})));
        return Ok(out);
    }
    let candidate: Option<PathBuf> = if last == "config.json" {
        Some(path)
    } else if last.starts_with("models--") {
        let ctx = Ctx {
            root: h.root.clone(),
            real: h.real.clone(),
            budget: RefCell::new(Budget::new(5000, 10.0)),
        };
        let (rev, _w) =
            hf_snapshot(&ctx, &path).map_err(|_| HandlerError::internal("BudgetExceeded"))?;
        rev.map(|r| py::join(&path, &["snapshots", r.as_str(), "config.json"]))
    } else {
        Some(py::join(&path, &["config.json"]))
    };
    let data = candidate.and_then(|c| read_inside(&h.real, &c, CONFIG_MAX));
    let Some(data) = data else {
        return Err(OpError::new(
            NOT_FOUND,
            format!(
                "there is no readable config.json for {} in model root {}",
                py::repr_take(rel, 300),
                py::repr(&h.root.id)
            ),
        )
        .into());
    };
    let Some(cfg @ Value::Object(_)) = json_of(Some(data)) else {
        return Err(OpError::new(
            IO_ERROR,
            format!(
                "the config.json of {} is not a JSON object",
                py::repr_take(rel, 300)
            ),
        )
        .into());
    };
    out.insert("config".into(), clean_json(&cfg));
    Ok(out)
}

// ── the background scanner ───────────────────────────────────────────────────────────────────────────────
/// Entries without access times: reading a model must not count as a library change.
fn comparable(entries: &[Obj]) -> Vec<Obj> {
    entries
        .iter()
        .map(|e| {
            let mut c = e.clone();
            c.remove("atime");
            c
        })
        .collect()
}

type Sig = Vec<(Vec<u8>, i128)>;

/// Cheap change detector: mtimes of the directories that change when a model appears or disappears.
fn signature(root: &Root, limit: usize) -> Sig {
    let real = py::realpath(Path::new(&root.path));
    let mut sig: Sig = Vec::new();
    let mut stack: Vec<(PathBuf, usize)> = vec![(real, 0)];
    while sig.len() < limit {
        let Some((path, depth)) = stack.pop() else {
            break;
        };
        let Some(st) = lstat(&path) else { continue };
        sig.push((
            path.as_os_str().as_bytes().to_vec(),
            i128::from(st.mtime()) * 1_000_000_000 + i128::from(st.mtime_nsec()),
        ));
        if depth >= 3 {
            continue;
        }
        let Ok(rd) = std::fs::read_dir(&path) else {
            continue;
        };
        for d in rd {
            let Ok(d) = d else { break };
            if sig.len() + stack.len() >= limit {
                break;
            }
            let name = d.file_name().to_string_lossy().into_owned();
            if d.file_type().is_ok_and(|t| t.is_dir()) && !hidden_or_secret(&name) {
                stack.push((d.path(), depth + 1));
            }
        }
    }
    sig.sort();
    sig
}

pub type ScanRootFn = Arc<dyn Fn(&Root, Budget) -> ScanResult + Send + Sync>;

/// Scanner knobs (Python constructor arguments plus the module constants tests patch).
#[derive(Clone)]
pub struct ScannerOptions {
    pub entries: i64,
    pub seconds: f64,
    pub rescan_s: f64,
    pub check_s: f64,
    pub hung_grace_s: f64,
    /// what scans one root (models.scan_root; tests replace it)
    pub scan_root: Option<ScanRootFn>,
}

impl std::fmt::Debug for ScannerOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScannerOptions")
            .field("entries", &self.entries)
            .field("seconds", &self.seconds)
            .field("rescan_s", &self.rescan_s)
            .field("check_s", &self.check_s)
            .field("hung_grace_s", &self.hung_grace_s)
            .field("scan_root", &self.scan_root.is_some())
            .finish()
    }
}

impl Default for ScannerOptions {
    fn default() -> Self {
        Self {
            entries: MAX_ENTRIES,
            seconds: MAX_SECONDS,
            rescan_s: RESCAN_S,
            check_s: CHECK_S,
            hung_grace_s: HUNG_GRACE_S,
            scan_root: None,
        }
    }
}

struct State {
    by_root: HashMap<String, Vec<Obj>>,
    scans: HashMap<String, Obj>,
    snapshot: Arc<Vec<Obj>>,
    generation: i64,
    assigned: i64,
    pending: bool,
    pending_all: bool,
    pending_roots: HashSet<String>,
    notify: bool,
    running: bool,
    done_at: f64,
    next_full: f64,
}

/// A scan_root outcome as `_scan_one`'s work() returns it: (status, entries, error).
type Work = (&'static str, Vec<Obj>, Option<String>);

/// Scans every policy root in the background and keeps the latest result as numbered generations.
///
/// The first scan runs at start. After that a root is rescanned when its directory signature changes (checked
/// every CHECK_S), every RESCAN_S anyway, and when the hub asks (models.scan). A scan that finds no difference
/// publishes nothing, except a requested one: the hub waits for the generation models.scan promised.
pub struct Scanner {
    policy_fn: PolicyFn,
    on_change: OnChange,
    opts: ScannerOptions,
    state: Mutex<State>,
    cv: Condvar,
    stop: AtomicBool,
    sigs: Mutex<HashMap<String, Sig>>,
    guard: Guarded,
}

impl Scanner {
    pub fn new(policy_fn: PolicyFn, on_change: OnChange, opts: ScannerOptions) -> Arc<Self> {
        Arc::new(Self {
            policy_fn,
            on_change,
            opts,
            state: Mutex::new(State {
                by_root: HashMap::new(),
                scans: HashMap::new(),
                snapshot: Arc::new(Vec::new()),
                generation: 0,
                assigned: 0,
                pending: false,
                pending_all: false,
                pending_roots: HashSet::new(),
                notify: false,
                running: false,
                done_at: -1e18,
                next_full: 0.0,
            }),
            cv: Condvar::new(),
            stop: AtomicBool::new(false),
            sigs: Mutex::new(HashMap::new()),
            guard: Guarded::new(),
        })
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        crate::util::lock(&self.state)
    }

    fn sigs(&self) -> MutexGuard<'_, HashMap<String, Sig>> {
        crate::util::lock(&self.sigs)
    }

    // ── control ──────────────────────────────────────────────────────────────────────────────────────────
    pub fn start(self: &Arc<Self>) {
        self.request(None, true);
        let me = Arc::clone(self);
        let _ = std::thread::Builder::new()
            .name("ecnode-models".into())
            .spawn(move || me.run_loop());
    }

    pub fn stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
        let _g = self.lock();
        self.cv.notify_all();
    }

    /// Ask for a rescan; returns the generation that scan will publish (or the current one when the last scan
    /// finished less than FRESH_S ago and nothing was forced).
    pub fn request(&self, roots: Option<Vec<String>>, force: bool) -> i64 {
        let mut st = self.lock();
        let idle = !st.running && !st.pending;
        let roots = roots.filter(|r| !r.is_empty());
        if !force && roots.is_none() && idle && st.generation != 0 && mono() - st.done_at < FRESH_S
        {
            return st.generation;
        }
        st.pending = true;
        st.notify = true;
        match roots {
            Some(r) => st.pending_roots.extend(r),
            None => st.pending_all = true,
        }
        self.cv.notify_all();
        st.assigned + 1
    }

    pub fn generation(&self) -> i64 {
        self.lock().generation
    }

    /// Queue what the loop queues every RESCAN_S: a full rescan nobody asked for, published only when it finds a
    /// change (tests set `_pending` and `_pending_all` for this in Python).
    pub fn queue_full_rescan(&self) {
        let mut st = self.lock();
        st.pending = true;
        st.pending_all = true;
    }

    // ── reading ──────────────────────────────────────────────────────────────────────────────────────────
    /// models.list: one page of the latest generation (cursor = offset; next_cursor None at the end).
    pub fn page(&self, cursor: Option<i64>, limit: Option<i64>) -> Obj {
        let policy = (self.policy_fn)();
        let (snap, generation, scans) = {
            let st = self.lock();
            (Arc::clone(&st.snapshot), st.generation, st.scans.clone())
        };
        let start = usize::try_from(cursor.unwrap_or(0).max(0)).unwrap_or(usize::MAX);
        let limit = match limit {
            None | Some(0) => 200,
            Some(l) => l,
        };
        let n = usize::try_from(limit.clamp(1, 200)).unwrap_or(200);
        let mut page: Vec<Value> = Vec::new();
        let mut used = 0usize;
        for e in snap.iter().skip(start).take(n) {
            let v = Value::Object(e.clone());
            let size = py::dumps(&v).len();
            if !page.is_empty() && used + size > PAGE_BYTES {
                break;
            }
            page.push(v);
            used += size;
        }
        let end = start.saturating_add(page.len());
        let roots: Vec<Value> = policy
            .model_roots
            .iter()
            .map(|r| {
                Value::Object(scans.get(&r.id).cloned().unwrap_or_else(|| {
                    let mut o = Obj::new();
                    o.insert("root".into(), json!(r.id));
                    o.insert("scanned_at".into(), Value::Null);
                    o.insert("complete".into(), json!(false));
                    o.insert("entries".into(), json!(0));
                    o.insert("error".into(), Value::Null);
                    o.insert("duration_s".into(), Value::Null);
                    o
                }))
            })
            .collect();
        let mut out = Obj::new();
        out.insert("generation".into(), json!(generation));
        out.insert("entries".into(), Value::Array(page));
        out.insert(
            "next_cursor".into(),
            if end < snap.len() {
                json!(end)
            } else {
                Value::Null
            },
        );
        out.insert("roots".into(), Value::Array(roots));
        out
    }

    /// (entries, total bytes) of the latest generation.
    pub fn count(&self) -> (usize, i64) {
        let st = self.lock();
        let total = st
            .snapshot
            .iter()
            .map(|e| e.get("size").and_then(Value::as_i64).unwrap_or(0))
            .sum();
        (st.snapshot.len(), total)
    }

    // ── the loop ─────────────────────────────────────────────────────────────────────────────────────────
    fn run_loop(&self) {
        while !self.stop.load(Ordering::SeqCst) {
            let pending = {
                let mut st = self.lock();
                if !st.pending {
                    st = crate::util::sync::wait_timeout(
                        &self.cv,
                        st,
                        Duration::from_secs_f64(self.opts.check_s.max(0.0)),
                    );
                }
                if self.stop.load(Ordering::SeqCst) {
                    return;
                }
                st.pending
            };
            // one round that panics is reported and costs that round, never the scanner (node hardening)
            let round = crate::util::panics::contain("model scanner", || self.round(pending));
            if round.is_none() {
                self.lock().running = false;
            }
        }
    }

    /// One pass of the loop: queue the roots whose signature changed (or a full rescan when due), then scan.
    fn round(&self, pending: bool) {
        if !pending {
            let policy = (self.policy_fn)();
            let stale: Vec<String> = policy
                .model_roots
                .iter()
                .filter(|r| self.changed(r))
                .map(|r| r.id.clone())
                .collect();
            let full = mono() >= self.lock().next_full;
            if stale.is_empty() && !full {
                return;
            }
            let mut st = self.lock();
            if !st.pending {
                st.pending = true;
                if full {
                    st.pending_all = true;
                }
                st.pending_roots.extend(stale);
            }
        }
        self.run_once();
    }

    /// Run the pending scan now (the loop calls this; tests may too). Returns the published generation.
    pub fn run_once(&self) -> Option<i64> {
        let (roots, notify, generation) = {
            let mut st = self.lock();
            if !st.pending {
                return None;
            }
            let roots = if st.pending_all {
                None
            } else {
                Some(st.pending_roots.clone())
            };
            let notify = st.notify;
            st.pending = false;
            st.pending_all = false;
            st.notify = false;
            st.pending_roots = HashSet::new();
            st.assigned += 1;
            st.running = true;
            (roots, notify, st.assigned)
        };
        let wanted = |id: &str| roots.as_ref().is_none_or(|r| r.contains(id));
        let policy = (self.policy_fn)();
        let results: Vec<(String, Option<Vec<Obj>>, Obj)> = policy
            .model_roots
            .iter()
            .filter(|r| wanted(&r.id))
            .map(|r| {
                let (entries, scan) = self.scan_one(r);
                (r.id.clone(), entries, scan)
            })
            .collect();
        let publish = {
            let mut st = self.lock();
            st.running = false;
            for (rid, entries, scan) in results {
                if let Some(e) = entries {
                    st.by_root.insert(rid.clone(), e);
                }
                st.scans.insert(rid, scan);
            }
            let ids: Vec<String> = policy.model_roots.iter().map(|r| r.id.clone()).collect();
            st.by_root.retain(|k, _| ids.contains(k));
            st.scans.retain(|k, _| ids.contains(k));
            let mut new: Vec<Obj> = ids
                .iter()
                .flat_map(|rid| st.by_root.get(rid).cloned().unwrap_or_default())
                .collect();
            new.sort_by_key(|e| {
                (
                    e.get("root").map_or_else(|| "None".into(), py::str_of),
                    e.get("rel").map_or_else(|| "None".into(), py::str_of),
                )
            });
            let changed = comparable(&new) != comparable(&st.snapshot);
            let publish = changed || notify;
            if publish {
                st.generation = generation;
                st.snapshot = Arc::new(new);
            }
            st.done_at = mono();
            if roots.is_none() {
                st.next_full = mono() + self.opts.rescan_s;
            }
            publish
        };
        for r in &policy.model_roots {
            if wanted(&r.id) {
                let root = r.clone();
                let sig = self
                    .guard
                    .call(
                        &format!("sig:{}", r.id),
                        move || signature(&root, 3000),
                        Duration::from_secs(10),
                    )
                    .unwrap_or_default();
                self.sigs().insert(r.id.clone(), sig);
            }
        }
        if publish {
            (self.on_change)(generation);
            Some(generation)
        } else {
            None
        }
    }

    fn changed(&self, root: &Root) -> bool {
        let r = root.clone();
        let Some(sig) = self.guard.call(
            &format!("sig:{}", root.id),
            move || signature(&r, 3000),
            Duration::from_secs(10),
        ) else {
            return false;
        };
        self.sigs().get(&root.id).is_some_and(|old| *old != sig)
    }

    fn scan_one(&self, root: &Root) -> (Option<Vec<Obj>>, Obj) {
        let started = wall();
        let t0 = mono();
        let (entries, seconds) = (self.opts.entries, self.opts.seconds);
        let hook = self.opts.scan_root.clone();
        let r = root.clone();
        let work = move || -> Option<Work> {
            let budget = Budget::new(entries, seconds);
            let res = match &hook {
                Some(h) => h(&r, budget),
                None => scan_root(&r, budget),
            };
            match res {
                Ok((found, err)) => Some(("ok", found, err)),
                Err(ScanError::Missing) => Some(("missing", Vec::new(), None)),
                Err(ScanError::Os(n)) => Some((
                    "error",
                    Vec::new(),
                    Some(format!("cannot read the root: {}", py::strerror(n))),
                )),
                // an exception out of work(): Guarded gives "no answer"
                Err(ScanError::Budget(_)) => None,
            }
        };
        let res = self
            .guard
            .call(
                &format!("scan:{}", root.id),
                work,
                Duration::from_secs_f64((seconds + self.opts.hung_grace_s).max(0.0)),
            )
            .flatten();
        let took = py::round(mono() - t0, 3);
        let prev = self.lock().by_root.get(&root.id).map_or(0, Vec::len);
        let scan = |complete: bool, n: usize, err: Option<String>| {
            let mut o = Obj::new();
            o.insert("root".into(), json!(root.id));
            o.insert("scanned_at".into(), json!(started));
            o.insert("complete".into(), json!(complete));
            o.insert("entries".into(), json!(n));
            o.insert("duration_s".into(), json!(took));
            o.insert("error".into(), opt_str(err));
            o
        };
        match res {
            None => (
                None,
                scan(
                    false,
                    prev,
                    Some(
                        "the filesystem did not answer in time (a hung network mount?); showing the previous scan"
                            .into(),
                    ),
                ),
            ),
            Some(("missing", _, _)) => (
                Some(Vec::new()),
                scan(
                    false,
                    0,
                    Some(format!("{} does not exist on this machine (not mounted?)", root.path)),
                ),
            ),
            Some(("error", _, err)) => (Some(Vec::new()), scan(false, 0, err)),
            Some((_, found, err)) => {
                let n = found.len();
                (Some(found), scan(err.is_none(), n, err))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_and_quant_from_file_names() {
        assert_eq!(
            gguf_name_quant("Qwen3-8B-Q4_K_M.gguf", None),
            ("Qwen3-8B".into(), Some("Q4_K_M".into()))
        );
        assert_eq!(
            gguf_name_quant("Big-UD-Q2_K_XL-00001-of-00003.gguf", None),
            ("Big".into(), Some("UD-Q2_K_XL".into()))
        );
        assert_eq!(gguf_name_quant("plain.gguf", None), ("plain".into(), None));
    }

    #[test]
    fn ollama_names() {
        let p = |a: &[&str]| a.iter().map(|s| (*s).to_string()).collect::<Vec<_>>();
        assert_eq!(
            ollama_name(&p(&["registry.ollama.ai", "library", "llama3", "8b"])),
            "llama3:8b"
        );
        assert_eq!(
            ollama_name(&p(&["hf.co", "org", "repo", "Q8_0"])),
            "hf.co/org/repo:Q8_0"
        );
        assert_eq!(
            ollama_name(&p(&["registry.ollama.ai", "me", "m", "t"])),
            "me/m:t"
        );
        assert_eq!(ollama_name(&p(&["library", "m", "t"])), "m:t");
    }

    #[test]
    fn quant_from_configs() {
        let q = |v: Value| quant_of_config(v.as_object(), "x");
        assert_eq!(
            q(json!({"quantization_config": {"quant_method": "awq", "bits": 4}})),
            Some("awq-4bit".into())
        );
        assert_eq!(
            q(json!({"quantization_config": {"quant_method": "gptq", "bits": true}})),
            Some("gptq-Truebit".into())
        );
        assert_eq!(
            q(json!({"quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"}})),
            Some("nvfp4".into())
        );
        assert_eq!(
            q(
                json!({"quantization_config": {"quant_method": "compressed-tensors",
                     "config_groups": {"g": {"weights": {"num_bits": 4, "type": "int"}}}}})
            ),
            Some("w4a16".into())
        );
        assert_eq!(
            q(json!({"text_config": {"torch_dtype": "bfloat16"}})),
            Some("bf16".into())
        );
        assert_eq!(
            quant_of_config(None, "Org/M-FP8-Dynamic"),
            Some("fp8".into())
        );
        assert_eq!(params_from_label(Some(&json!("8.0B"))), Some(8_000_000_000));
        assert_eq!(
            params_from_label(Some(&json!(" 70b "))),
            Some(70_000_000_000)
        );
    }
}
