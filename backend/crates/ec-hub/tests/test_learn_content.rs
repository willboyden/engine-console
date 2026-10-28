//! Port of backend/tests/learn/test_learn_content.py: the learn-mode content gate (docs/ARCHITECTURE.md section 10.7,
//! learn/README.md), core rows. Every information button in the UI shows text from the learn files; this fails the
//! build when the UI would show nothing, something unreadable, or something that must not be public. Offline: it reads
//! JSON, the docs and the frontend's key list; it never touches docker, the network or a GPU.
//!
//! Not here (backend/TEST-MAP.md): the `check` concept family, stale concepts and the code-scan rows need the emitters
//! of the lifecycle service and the adapters (tests/test_learn_check_codes.rs); the per-engine parameter and preset
//! coverage needs the real adapter catalogs. Python reads closed value lists from `Literal` types; where Rust has the
//! list as a const it is used, otherwise the list is copied below with its Python source named.

#![allow(clippy::panic, clippy::expect_used, clippy::print_stderr)]

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use ec_hub::adapters::base::Explain;
use ec_hub::adapters::load_default_adapters;
use ec_hub::domain::fleet::{LIBRARY_FORMATS, LINK_STATES};
use ec_hub::domain::learn::{LearnBundle, Term};
use ec_hub::domain::models::INSTANCE_STATES;
use ec_hub::services::common::py_splitlines;
use ec_hub::services::hostmem::ALERT_CODES;
use regex::Regex;
use serde::de::{DeserializeSeed, Deserializer, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Value};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn learn_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("learn")
}

fn re(p: &str) -> Regex {
    Regex::new(p).unwrap_or_else(|e| panic!("{p}: {e}"))
}

// ---- what must exist ------------------------------------------------------------------------------------------------
/// The glossary keys engine specialists and the UI refer to by name. Deleting or renaming one breaks them.
#[rustfmt::skip]
const REQUIRED_TERMS: &[&str] = &[
    "model", "weights", "parameters", "inference_engine", "serving", "gpu", "vram", "unified_memory", "host_ram",
    "shared_memory", "page_cache", "swap", "cgroup", "container", "docker_image", "image_tag", "gateway",
    "internal_network", "port", "endpoint", "openai_compatible_api", "token", "tokenizer", "context_window",
    "prompt", "prefill", "decode", "ttft", "itl", "throughput", "latency", "batch", "continuous_batching",
    "concurrency", "kv_cache", "paged_attention", "prefix_caching", "chunked_prefill", "quantization", "bf16",
    "fp16", "fp8", "nvfp4", "mxfp4", "int8", "int4", "awq", "gptq", "gguf", "gguf_quant_types", "safetensors",
    "hugging_face_hub", "hf_cache", "repo_id", "revision", "gated_model", "hf_token", "moe", "active_parameters",
    "dense_model", "tensor_parallelism", "pipeline_parallelism", "expert_parallelism", "data_parallelism", "nccl",
    "nvlink", "pcie", "peer_to_peer", "cuda", "cuda_graphs", "torch_compile", "compute_capability", "blackwell",
    "attention_backend", "flash_attention", "speculative_decoding", "draft_model", "mtp", "rope_scaling", "yarn",
    "sliding_window_attention", "gqa", "mla", "lora", "tool_calling", "reasoning_model", "chat_template",
    "trust_remote_code", "served_model_name", "cpu_offload", "kv_cache_offload", "gpu_layers_offload",
    "tensor_split", "mmap", "mlock", "cpu_threads", "numa", "fit_estimate", "preflight", "prometheus_metrics",
    "sampling", "temperature", "max_tokens", "embedding_model", "vision_language_model", "multimodal", "benchmark",
    "idle_ttl", "health_check", "warmup", "oom", "node_agent", "host", "nfs", "zfs", "model_root", "monitor_mode",
    "manage_mode", "restricted_ssh_key", "egress_proxy", "elo_rating", "drive", "mount", "docker_rootless",
    "management_class", "lifecycle_script",
];

/// ARCHITECTURE section 5 (engine metrics) and the host / instance keys of sections 8 and 10.9.
#[rustfmt::skip]
const CANONICAL_METRICS: &[&str] = &[
    "requests_running", "requests_waiting", "kv_cache_usage_pct", "prefix_cache_hit_pct", "prompt_tokens_total",
    "generation_tokens_total", "ttft_p50_s", "ttft_p95_s", "itl_p50_s", "e2e_p50_s", "e2e_p95_s", "prompt_tps",
    "generation_tps", "preemptions_total", "spec_decode_accept_pct",
];
#[rustfmt::skip]
const HOST_METRICS: &[&str] = &[
    "host_ram_gib", "host_ram_anon_gib", "host_ram_cache_gib", "host_ram_shmem_gib", "ram_used_gib",
    "ram_available_gib", "shmem_gib", "swap_used_gib", "vram_used_gib", "gpu_util_pct", "gpu_temp_c", "gpu_power_w",
    "gpu_power_limit_w", "gpu_fan_pct", "cpu_util_pct", "load1", "disk_free_bytes",
    "disk_free_gib", // the frontend asks for this spelling (frontend/js/learn-keys.js); same meaning, other unit
];

/// Concept families (ARCHITECTURE 10.7).
#[rustfmt::skip]
const FAMILIES: &[&str] = &[
    "state", "link", "management", "verdict", "confidence", "check", "format", "root", "alert", "download", "mode",
    "policy", "page", "field",
];
const V2_ALERT_CODES: &[&str] = &["gpu_hot", "vram_full"]; // raised from live GPU stats, not by services/hostmem.rs
const ROOT_KINDS: &[&str] = &[
    "auto", "hf-home", "hf-hub", "plain", "gguf", "ollama", "other",
]; // FLEET-PROTOCOL 6.1
const MODES: &[&str] = &["monitor", "manage"];
#[rustfmt::skip]
const POLICY_KEYS: &[&str] = &["engine_images", "allow_pull", "allow_delete", "external_control", "scripts"];
#[rustfmt::skip]
const PAGES: &[&str] = &[
    "dashboard", "hosts", "host", "library", "discover", "downloads", "launch", "wizard", "instances", "instance",
    "metrics", "bench", "chat", "arena", "usage", "settings", "learn",
];
#[rustfmt::skip]
const FIELDS: &[&str] = &[
    "host", "gpus", "root", "image", "engine", "model", "preset", "concurrency", "name", "ttl", "pin", "profile",
    "revision", "allow_patterns", "quant", "target_host", "beginner_mode",
];
// Closed value lists Rust's domain keeps only in doc comments (copied from the Python Literal types they mirror).
const MANAGEMENT: &[&str] = &["managed", "script", "external"]; // domain/refs.py Management
const VERDICTS: &[&str] = &["fits", "tight", "wont_fit", "unknown"]; // domain/models.py Verdict
const HOST_RAM_VERDICTS: &[&str] = &["ok", "tight", "wont_fit", "unknown"]; // domain/models.py HostRamFit.verdict
const CONFIDENCES: &[&str] = &["high", "medium", "low"]; // domain/models.py Confidence
const DOWNLOAD_STATES: &[&str] = &[
    "queued",
    "running",
    "paused",
    "completed",
    "failed",
    "cancelled",
]; // DownloadState
const DOWNLOAD_KINDS: &[&str] = &["hf", "copy"]; // domain/models.py Download.kind

// ---- limits (mirrors learn/README.md; test_limits_match_readme keeps the two in step) --------------------------------
#[rustfmt::skip]
const EXPLAIN_LIMITS: &[(&str, usize)] =
    &[("summary", 160), ("details", 900), ("when_to_change", 400), ("risks", 400), ("example", 200)];
const TERM_LIMITS: &[(&str, usize)] = &[("short", 160), ("long", 1500)];
const SINGLE_LINE_FIELDS: &[&str] = &["title", "short", "summary", "example"];
const ACTION_FAMILIES: &[&str] = &["state", "link", "check", "alert", "verdict", "download"]; // must say what to do
const TERM_FIELDS: &[&str] = &["long", "see_also", "short", "title"];
const EXPLAIN_FIELDS: &[&str] = &[
    "details",
    "example",
    "level",
    "risks",
    "summary",
    "terms",
    "when_to_change",
];

// ---- key shapes -----------------------------------------------------------------------------------------------------
const SNAKE: &str = r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*";
const NAME: &str = r"[a-z][a-z0-9]*(?:[_-][a-z0-9]+)*";
static KEY_SHAPES: LazyLock<BTreeMap<&'static str, Regex>> = LazyLock::new(|| {
    let full = |p: String| re(&format!("^(?:{p})$"));
    BTreeMap::from([
        ("terms", full(SNAKE.into())),
        ("metrics", full(SNAKE.into())),
        ("params", full(SNAKE.into())),
        ("presets", full(NAME.into())),
        ("concepts", full(format!(r"[a-z]+\.{NAME}"))),
    ])
});
const SECTION_OF_FILE: &[(&str, &str)] = &[
    ("glossary.json", "terms"),
    ("metrics.json", "metrics"),
    ("concepts.json", "concepts"),
];
const STRICT_FILES: &[&str] = &["glossary.json", "metrics.json", "concepts.json"]; // jargon links enforced here

// ---- wording rules --------------------------------------------------------------------------------------------------
static HYGIENE_RULES: LazyLock<Vec<(&'static str, Regex)>> = LazyLock::new(|| {
    vec![
        (
            "placeholder text",
            re(r"(?i)\b(?:TODO|TBD|FIXME|XXX)\b|lorem ipsum|\?\?\?"),
        ),
        (
            "vague 'may cause issues' wording: name the failure",
            re(
                r"(?i)\b(?:may|might|could|can) (?:cause|lead to|result in) (?:some |various |other )?(?:issues|problems)\b",
            ),
        ),
        (
            "promotional wording",
            re(
                r"(?i)\b(?:blazing(?:ly)?|revolutionary|seamless(?:ly)?|cutting[- ]edge|state[- ]of[- ]the[- ]art|best[- ]in[- ]class|world[- ]class|game[- ]chang\w+|supercharg\w+|turbocharg\w+|effortless(?:ly)?|unleash\w*)\b",
            ),
        ),
        (
            "emoji or pictograph",
            re(r"[\x{1F000}-\x{1FAFF}\x{2600}-\x{26FF}\x{2700}-\x{27BF}]"),
        ),
    ]
});

/// `(?<=[.!?])\s+(?=[A-Z0-9`'"(])`: a whitespace run after sentence punctuation, before what starts a sentence.
fn sentence_break(text: &str) -> bool {
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i].is_whitespace() {
            let start = i;
            while i < chars.len() && chars[i].is_whitespace() {
                i += 1;
            }
            let before = start.checked_sub(1).map(|j| chars[j]);
            let after = chars.get(i).copied();
            if matches!(before, Some('.' | '!' | '?'))
                && after.is_some_and(|c| {
                    c.is_ascii_uppercase() || c.is_ascii_digit() || "`'\"(".contains(c)
                })
            {
                return true;
            }
        } else {
            i += 1;
        }
    }
    false
}

/// Jargon a beginner cannot be expected to know: if the text uses it, the glossary key must be in `terms` /
/// `see_also` (README rule 7). The third member is Python's `(?<!chunked )` lookbehind (the regex crate has none).
#[rustfmt::skip]
const JARGON: &[(&str, &str, Option<&str>)] = &[
    (r"\bKV[ -]?cache", "kv_cache", None), (r"\bVRAM\b", "vram", None),
    (r"\bunified memory\b", "unified_memory", None), (r"\bNVLink\b", "nvlink", None), (r"\bPCIe\b", "pcie", None),
    (r"\bpeer-to-peer\b|\bP2P\b", "peer_to_peer", None), (r"\bNCCL\b", "nccl", None),
    (r"\btensor[ -]parallel", "tensor_parallelism", None), (r"\bpipeline[ -]parallel", "pipeline_parallelism", None),
    (r"\bexpert[ -]parallel", "expert_parallelism", None), (r"\bdata[ -]parallel", "data_parallelism", None),
    (r"\bquantiz|\bquantis", "quantization", None), (r"\bFP8\b", "fp8", None), (r"\bNVFP4\b", "nvfp4", None),
    (r"\bMXFP4\b", "mxfp4", None), (r"\bBF16\b|\bbfloat16\b", "bf16", None), (r"\bFP16\b", "fp16", None),
    (r"\bINT8\b", "int8", None), (r"\bINT4\b", "int4", None), (r"\bAWQ\b", "awq", None), (r"\bGPTQ\b", "gptq", None),
    (r"\bGGUF\b", "gguf", None), (r"\bsafetensors\b", "safetensors", None),
    (r"\bMoE\b|\bmixture[ -]of[ -]experts\b", "moe", None), (r"\bactive parameters\b", "active_parameters", None),
    (r"\bprefill\b", "prefill", Some("chunked ")), (r"\bchunked prefill\b", "chunked_prefill", None),
    (r"\bdecode\b", "decode", None), (r"\bTTFT\b|\btime to first token\b", "ttft", None),
    (r"\bITL\b|\binter-token latency\b", "itl", None), (r"\bp(?:50|95|99)\b|\bpercentiles?\b", "percentile", None),
    (r"\bcgroups?\b", "cgroup", None), (r"\bpage cache\b", "page_cache", None), (r"\bswap\b", "swap", None),
    (r"\bshared memory\b", "shared_memory", None), (r"\bCUDA graphs?\b", "cuda_graphs", None),
    (r"\btorch\.compile\b", "torch_compile", None), (r"\bYaRN\b", "yarn", None), (r"\bRoPE\b", "rope_scaling", None),
    (r"\bLoRA\b", "lora", None), (r"\bprefix[ -]cach", "prefix_caching", None),
    (r"\bpaged attention\b", "paged_attention", None), (r"\bcontinuous batching\b", "continuous_batching", None),
    (r"\bspeculative decoding\b", "speculative_decoding", None), (r"\bdraft model\b", "draft_model", None),
    (r"\bMTP\b|\bmulti-token prediction\b", "mtp", None), (r"\bNUMA\b", "numa", None), (r"\bmmap\b", "mmap", None),
    (r"\bmlock\b", "mlock", None), (r"\bcompute capability\b|\bsm_\d+", "compute_capability", None),
    (r"\bBlackwell\b", "blackwell", None), (r"\bGQA\b|\bgrouped-query\b", "gqa", None),
    (r"\bMLA\b|\blatent attention\b", "mla", None), (r"\bsliding[ -]window\b", "sliding_window_attention", None),
    (r"\bFlash-?Attention\b", "flash_attention", None), (r"\battention backend\b", "attention_backend", None),
    (r"\btool[ -]call", "tool_calling", None), (r"\breasoning model", "reasoning_model", None),
    (r"\bchat template\b", "chat_template", None), (r"\btrust_remote_code\b", "trust_remote_code", None),
    (r"\bserved model name\b", "served_model_name", None), (r"\bCPU offload", "cpu_offload", None),
    (r"\bOOM\b|\bout of memory\b", "oom", None), (r"\bnode agent\b|\becnode\b", "node_agent", None),
    (r"\bNFS\b", "nfs", None), (r"\bZFS\b", "zfs", None), (r"\begress proxy\b", "egress_proxy", None),
    (r"\bElo\b", "elo_rating", None), (r"\bmonitor mode\b", "monitor_mode", None),
    (r"\bmanage mode\b", "manage_mode", None), (r"\brootless\b", "docker_rootless", None),
];
static JARGON_RE: LazyLock<Vec<(Regex, &'static str, Option<&'static str>)>> =
    LazyLock::new(|| {
        JARGON
            .iter()
            .map(|(p, k, not_after)| (re(&format!("(?i){p}")), *k, *not_after))
            .collect()
    });

// ---- publish safety -------------------------------------------------------------------------------------------------
// Written in halves so a plain-text search of the repository for home paths finds real leaks only, not this lint.
const HOME_PREFIXES: &[&str] = &[concat!("/ho", "me/"), concat!("/Us", "ers/")];
static HOME_PATH: LazyLock<Regex> = LazyLock::new(|| {
    let alts: Vec<String> = HOME_PREFIXES.iter().map(|p| regex::escape(p)).collect();
    re(&format!(r"(?i){}|[A-Za-z]:\\Users\\", alts.join("|")))
});
static EMAIL: LazyLock<Regex> =
    LazyLock::new(|| re(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)+"));
/// 192.0.2.0/24, 198.51.100.0/24, 203.0.113.0/24, 10.0.0.0/8 as (network, prefix length).
const DOCUMENTATION_NETS: &[([u8; 4], u32)] = &[
    ([192, 0, 2, 0], 24),
    ([198, 51, 100, 0], 24),
    ([203, 0, 113, 0], 24),
    ([10, 0, 0, 0], 8),
];
static SECRET_SHAPES: LazyLock<Vec<(&'static str, Regex)>> = LazyLock::new(|| {
    vec![
        ("Hugging Face token", re(r"\bhf_[A-Za-z0-9]{20,}")),
        ("API key", re(r"\bsk-[A-Za-z0-9_\-]{20,}")),
        (
            "GitHub token",
            re(r"\bgh[pousr]_[A-Za-z0-9]{20,}|\bgithub_pat_[A-Za-z0-9_]{20,}"),
        ),
        ("AWS access key", re(r"\bAKIA[0-9A-Z]{16}\b")),
        ("Slack token", re(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}")),
        (
            "private key block",
            re(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
        ),
        (
            "ssh public key",
            re(r"\bssh-(?:ed25519|rsa|dss)\s+AAAA[0-9A-Za-z+/=]{20,}"),
        ),
    ]
});
#[rustfmt::skip]
const COMMON_NAMES: &[&str] = &[
    "localhost", "ubuntu", "debian", "runner", "github", "docker", "builder", "worker", "vagrant", "jenkins", "gitlab",
    "codespace", "vscode", "devcontainer", "default", "administrator", "ec2-user",
];

// ---- the bundle, typed --------------------------------------------------------------------------------------------
/// `LearnBundle` with its entries as `Term` / `Explain` (the Rust bundle keeps them as validated JSON).
#[derive(Clone, Default)]
struct Bundle {
    glossary: Vec<(String, Term)>,
    metrics: Vec<(String, Explain)>,
    concepts: Vec<(String, Explain)>,
    params: Vec<(String, Vec<(String, Explain)>)>,
    presets: Vec<(String, Vec<(String, Explain)>)>,
}

fn typed<T: serde::de::DeserializeOwned>(m: &Map<String, Value>) -> Vec<(String, T)> {
    m.iter()
        .map(|(k, v)| {
            (
                k.clone(),
                serde_json::from_value(v.clone()).unwrap_or_else(|e| panic!("{k}: {e}")),
            )
        })
        .collect()
}

fn per_engine(m: &Map<String, Value>) -> Vec<(String, Vec<(String, Explain)>)> {
    m.iter()
        .map(|(engine, v)| {
            (
                engine.clone(),
                typed(v.as_object().expect("an engine's entries")),
            )
        })
        .collect()
}

impl From<&LearnBundle> for Bundle {
    fn from(b: &LearnBundle) -> Self {
        Self {
            glossary: typed(&b.glossary),
            metrics: typed(&b.metrics),
            concepts: typed(&b.concepts),
            params: per_engine(&b.params),
            presets: per_engine(&b.presets),
        }
    }
}

fn bundle() -> Bundle {
    Bundle::from(&*ec_hub::learn::bundle().expect("the packaged learn bundle loads"))
}

fn has<T>(entries: &[(String, T)], key: &str) -> bool {
    entries.iter().any(|(k, _)| k == key)
}

fn ex(summary: &str) -> Explain {
    Explain {
        summary: summary.into(),
        details: String::new(),
        when_to_change: String::new(),
        risks: String::new(),
        example: String::new(),
        terms: Vec::new(),
        level: "advanced".into(),
    }
}

fn explain_field<'a>(e: &'a Explain, field: &str) -> &'a str {
    match field {
        "summary" => &e.summary,
        "details" => &e.details,
        "when_to_change" => &e.when_to_change,
        "risks" => &e.risks,
        "example" => &e.example,
        other => panic!("no Explain field {other}"),
    }
}

fn term_field<'a>(t: &'a Term, field: &str) -> &'a str {
    match field {
        "title" => &t.title,
        "short" => &t.short,
        "long" => &t.long,
        other => panic!("no Term field {other}"),
    }
}

fn len(s: &str) -> usize {
    s.chars().count()
}

// ---- helpers --------------------------------------------------------------------------------------------------------
fn report(title: &str, problems: &[String]) -> String {
    let limit = 250;
    let shown = &problems[..problems.len().min(limit)];
    let more = if problems.len() > limit {
        format!("\n  ... and {} more", problems.len() - limit)
    } else {
        String::new()
    };
    format!(
        "{title} ({}):\n  - {}{more}",
        problems.len(),
        shown.join("\n  - ")
    )
}

/// Fail with just the list of offenders when there are any.
fn expect_none(title: &str, problems: &[String]) {
    assert!(problems.is_empty(), "{}", report(title, problems));
}

/// `json.loads` that also reports duplicate object keys (serde_json, like the stock parser, keeps the last one).
fn strict_json(text: &str) -> Result<(Value, Vec<String>), serde_json::Error> {
    struct Strict<'a>(&'a RefCell<Vec<String>>);

    impl<'de> DeserializeSeed<'de> for Strict<'_> {
        type Value = Value;
        fn deserialize<D: Deserializer<'de>>(self, d: D) -> Result<Value, D::Error> {
            d.deserialize_any(self)
        }
    }

    impl<'de> Visitor<'de> for Strict<'_> {
        type Value = Value;
        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("JSON")
        }
        fn visit_bool<E>(self, v: bool) -> Result<Value, E> {
            Ok(Value::Bool(v))
        }
        fn visit_i64<E>(self, v: i64) -> Result<Value, E> {
            Ok(Value::from(v))
        }
        fn visit_u64<E>(self, v: u64) -> Result<Value, E> {
            Ok(Value::from(v))
        }
        fn visit_f64<E>(self, v: f64) -> Result<Value, E> {
            Ok(Value::from(v))
        }
        fn visit_str<E>(self, v: &str) -> Result<Value, E> {
            Ok(Value::String(v.to_string()))
        }
        fn visit_string<E>(self, v: String) -> Result<Value, E> {
            Ok(Value::String(v))
        }
        fn visit_unit<E>(self) -> Result<Value, E> {
            Ok(Value::Null)
        }
        fn visit_none<E>(self) -> Result<Value, E> {
            Ok(Value::Null)
        }
        fn visit_seq<A: SeqAccess<'de>>(self, mut a: A) -> Result<Value, A::Error> {
            let mut out = Vec::new();
            while let Some(v) = a.next_element_seed(Strict(self.0))? {
                out.push(v);
            }
            Ok(Value::Array(out))
        }
        fn visit_map<A: MapAccess<'de>>(self, mut a: A) -> Result<Value, A::Error> {
            let mut out = Map::new();
            while let Some(k) = a.next_key::<String>()? {
                let v = a.next_value_seed(Strict(self.0))?;
                if out.contains_key(&k) {
                    self.0.borrow_mut().push(k.clone());
                }
                out.insert(k, v); // like dict(pairs): the last value, at the first one's place
            }
            Ok(Value::Object(out))
        }
    }

    let dups = RefCell::new(Vec::new());
    let mut de = serde_json::Deserializer::from_str(text);
    let doc = Strict(&dups).deserialize(&mut de)?;
    de.end()?;
    Ok((doc, dups.into_inner()))
}

/// Every file under `root`, sorted by path.
fn walk(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d)
            .unwrap_or_else(|e| panic!("{}: {e}", d.display()))
            .flatten()
        {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

fn rel_of(p: &Path, root: &Path) -> String {
    p.strip_prefix(root)
        .map(|r| r.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// Every JSON file under the learn directory with the section it must contain ("" = not an allowed file).
fn learn_files(root: &Path) -> Vec<(PathBuf, String)> {
    walk(root)
        .into_iter()
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .map(|p| {
            let rel = rel_of(&p, root);
            let section = SECTION_OF_FILE.iter().find(|(f, _)| *f == rel).map_or_else(
                || {
                    if rel.starts_with("params/") {
                        "params"
                    } else if rel.starts_with("presets/") {
                        "presets"
                    } else {
                        ""
                    }
                },
                |(_, s)| *s,
            );
            (p, section.to_string())
        })
        .collect()
}

/// (file label, key, Explain) for every explanation in the bundle except glossary terms.
fn explain_items(b: &Bundle) -> Vec<(String, String, Explain)> {
    let mut items: Vec<(String, String, Explain)> = b
        .metrics
        .iter()
        .map(|(k, v)| ("metrics.json".to_string(), k.clone(), v.clone()))
        .collect();
    items.extend(
        b.concepts
            .iter()
            .map(|(k, v)| ("concepts.json".to_string(), k.clone(), v.clone())),
    );
    for (engine, params) in &b.params {
        items.extend(
            params
                .iter()
                .map(|(k, v)| (format!("params/{engine}.json"), k.clone(), v.clone())),
        );
    }
    for (engine, presets) in &b.presets {
        items.extend(
            presets
                .iter()
                .map(|(k, v)| (format!("presets/{engine}.json"), k.clone(), v.clone())),
        );
    }
    items
}

/// family -> [(value, where that value comes from)], every family but `check` (its codes come from the
/// lifecycle service's and the adapters' emitters).
fn required_concepts() -> BTreeMap<&'static str, Vec<(String, &'static str)>> {
    let of = |vals: &[&str], origin: &'static str| -> Vec<(String, &'static str)> {
        vals.iter().map(|v| ((*v).to_string(), origin)).collect()
    };
    let mut verdict = of(VERDICTS, "domain/models.py Verdict / HostRamFit");
    verdict.extend(of(
        HOST_RAM_VERDICTS,
        "domain/models.py Verdict / HostRamFit",
    ));
    let mut alert = of(V2_ALERT_CODES, "dashboard");
    alert.extend(of(ALERT_CODES, "services/hostmem.rs ALERT_CODES"));
    let mut download = of(DOWNLOAD_STATES, "domain/models.py DownloadState");
    download.extend(
        DOWNLOAD_KINDS
            .iter()
            .map(|k| (format!("kind_{k}"), "domain/models.py Download.kind")),
    );
    BTreeMap::from([
        (
            "state",
            of(INSTANCE_STATES, "domain/models.rs INSTANCE_STATES"),
        ),
        ("link", of(LINK_STATES, "domain/fleet.rs LINK_STATES")),
        ("management", of(MANAGEMENT, "domain/refs.py Management")),
        ("verdict", verdict),
        ("confidence", of(CONFIDENCES, "domain/models.py Confidence")),
        (
            "format",
            of(LIBRARY_FORMATS, "domain/fleet.rs LIBRARY_FORMATS"),
        ),
        ("root", of(ROOT_KINDS, "docs/FLEET-PROTOCOL.md 6.1")),
        ("alert", alert),
        ("download", download),
        ("mode", of(MODES, "docs/FLEET-PROTOCOL.md 6.2")),
        ("policy", of(POLICY_KEYS, "docs/FLEET-PROTOCOL.md 6")),
        ("page", of(PAGES, "ARCHITECTURE 10.7")),
        ("field", of(FIELDS, "ARCHITECTURE 10.7")),
    ])
}

/// (concept family -> ids, metric keys)
type UiKeys = (Vec<(String, Vec<String>)>, Vec<String>);

/// The concept and metric keys the frontend says it asks for, read from frontend/js/learn-keys.js (None outside
/// the repository).
fn ui_requested_keys() -> Option<UiKeys> {
    let path = repo_root().join("frontend/js/learn-keys.js");
    let text = std::fs::read_to_string(path).ok()?;
    let block = re(r"(?s)CONCEPT_FAMILIES\s*=\s*\{(.*?)\n\};").captures(&text);
    let metrics = re(r"(?s)METRIC_KEYS\s*=\s*\[(.*?)\];").captures(&text);
    let (Some(block), Some(metrics)) = (block, metrics) else {
        return Some((Vec::new(), Vec::new()));
    };
    let quoted = re(r"'([^']+)'");
    let mut families: Vec<(String, Vec<String>)> = Vec::new();
    for c in re(r"(?s)(\w+):\s*\[(.*?)\]").captures_iter(&block[1]) {
        let ids: Vec<String> = quoted
            .captures_iter(&c[2])
            .map(|q| q[1].to_string())
            .collect();
        families.retain(|(n, _)| n != &c[1]); // a dict: the last one wins
        families.push((c[1].to_string(), ids));
    }
    let keys = re(r"'([a-z0-9_]+)'")
        .captures_iter(&metrics[1])
        .map(|q| q[1].to_string())
        .collect();
    Some((families, keys))
}

// ---- the checks (pure functions: the real tests call them on the real bundle, the self-tests on small fakes) ----------
/// Shape of one learn file: JSON, no duplicate keys, {"version": 1, "<section>": {...}}, sorted, known fields.
fn check_file_text(rel: &str, section: &str, text: &str) -> Vec<String> {
    if section.is_empty() {
        return vec![format!(
            "{rel}: not an allowed learn file (glossary.json, metrics.json, concepts.json, params/<engine>.json, \
             presets/<engine>.json)"
        )];
    }
    let (doc, duplicates) = match strict_json(text) {
        Ok(d) => d,
        Err(e) => return vec![format!("{rel}: invalid JSON ({e})")],
    };
    let mut problems: Vec<String> = duplicates
        .iter()
        .map(|k| format!("{rel}: duplicate key '{k}' (the second silently replaces the first)"))
        .collect();
    let top_ok = doc
        .as_object()
        .is_some_and(|o| o.len() == 2 && o.contains_key("version") && o.contains_key(section));
    if !top_ok {
        problems.push(format!(
            r#"{rel}: top level must be exactly {{"version": 1, "{section}": {{...}}}}"#
        ));
        return problems;
    }
    if !matches!(&doc["version"], Value::Number(n) if !n.is_f64() && n.as_i64() == Some(1)) {
        problems.push(format!("{rel}: version must be the number 1"));
    }
    let Some(body) = doc[section].as_object() else {
        problems.push(format!("{rel}: '{section}' must be an object"));
        return problems;
    };
    let keys: Vec<&String> = body.keys().collect();
    let mut sorted = keys.clone();
    sorted.sort();
    if let Some((first, _)) = keys.iter().zip(&sorted).find(|(k, s)| k != s) {
        problems.push(format!(
            "{rel}: keys are not sorted (first out of place: '{first}')"
        ));
    }
    let allowed = if section == "terms" {
        TERM_FIELDS
    } else {
        EXPLAIN_FIELDS
    };
    for (key, entry) in body {
        if !KEY_SHAPES[section].is_match(key) {
            problems.push(format!("{rel}: malformed key '{key}'"));
        }
        if section == "concepts"
            && let Some((family, _)) = key.split_once('.')
            && !FAMILIES.contains(&family)
        {
            problems.push(format!(
                "{rel}: '{key}' has an unknown concept family (known: {})",
                FAMILIES.join(", ")
            ));
        }
        let Some(entry) = entry.as_object() else {
            problems.push(format!("{rel}: '{key}' must be an object"));
            continue;
        };
        let unknown: BTreeSet<&String> = entry
            .keys()
            .filter(|f| !allowed.contains(&f.as_str()))
            .collect();
        for field in unknown {
            problems.push(format!(
                "{rel}: '{key}' has unknown field '{field}' (allowed: {})",
                allowed.join(", ")
            ));
        }
    }
    problems
}

fn check_limits(b: &Bundle) -> Vec<String> {
    let mut problems = Vec::new();
    for (key, t) in &b.glossary {
        if t.title.trim().is_empty() {
            problems.push(format!("glossary.json: '{key}' has an empty title"));
        }
        if t.short.trim().is_empty() {
            problems.push(format!("glossary.json: '{key}' has an empty short"));
        }
        for (field, limit) in TERM_LIMITS {
            let n = len(term_field(t, field));
            if n > *limit {
                problems.push(format!(
                    "glossary.json: '{key}'.{field} is {n} chars (limit {limit})"
                ));
            }
        }
    }
    for (file, key, e) in explain_items(b) {
        if e.summary.trim().is_empty() {
            problems.push(format!("{file}: '{key}' has an empty summary"));
        }
        for (field, limit) in EXPLAIN_LIMITS {
            let n = len(explain_field(&e, field));
            if n > *limit {
                problems.push(format!(
                    "{file}: '{key}'.{field} is {n} chars (limit {limit})"
                ));
            }
        }
    }
    problems
}

/// Fields a reader of our own files must always find (params and presets only need a summary: advised below).
fn check_required_text(b: &Bundle) -> Vec<String> {
    let mut problems = Vec::new();
    for (key, t) in &b.glossary {
        if t.long.trim().is_empty() {
            problems.push(format!("glossary.json: '{key}' has no long text"));
        } else if len(&t.long) <= len(&t.short) {
            problems.push(format!(
                "glossary.json: '{key}'.long is not longer than its short"
            ));
        }
    }
    for (file, key, e) in explain_items(b) {
        let mut need: Vec<&str> = match file.as_str() {
            "metrics.json" => vec!["summary", "details", "when_to_change", "risks", "example"],
            "concepts.json" => vec!["summary", "details"],
            _ => vec![],
        };
        if file == "concepts.json" && ACTION_FAMILIES.contains(&key.split('.').next().unwrap_or(""))
        {
            need.push("when_to_change");
        }
        for f in need {
            if explain_field(&e, f).trim().is_empty() {
                problems.push(format!("{file}: '{key}' has no {f}"));
            }
        }
    }
    problems
}

/// (file, key, field, text) for every string a reader sees.
fn all_text(b: &Bundle) -> Vec<(String, String, &'static str, String)> {
    let mut out = Vec::new();
    for (key, t) in &b.glossary {
        for field in ["title", "short", "long"] {
            out.push((
                "glossary.json".to_string(),
                key.clone(),
                field,
                term_field(t, field).to_string(),
            ));
        }
    }
    for (file, key, e) in explain_items(b) {
        for field in ["summary", "details", "when_to_change", "risks", "example"] {
            out.push((
                file.clone(),
                key.clone(),
                field,
                explain_field(&e, field).to_string(),
            ));
        }
    }
    out
}

fn check_hygiene(b: &Bundle) -> Vec<String> {
    let mut problems = Vec::new();
    for (file, key, field, text) in all_text(b) {
        let place = format!("{file}: '{key}'.{field}");
        if text.is_empty() {
            continue;
        }
        if text != text.trim() {
            problems.push(format!("{place}: leading or trailing whitespace"));
        }
        if text.contains("  ")
            || text.contains('\t')
            || text.chars().any(|c| (c as u32) < 32 && c != '\n')
        {
            problems.push(format!("{place}: double space, tab or control character"));
        }
        if SINGLE_LINE_FIELDS.contains(&field) && text.contains('\n') {
            problems.push(format!("{place}: must be one line"));
        }
        for (label, rx) in HYGIENE_RULES.iter() {
            if let Some(m) = rx.find(&text) {
                problems.push(format!("{place}: {label} ('{}')", m.as_str()));
            }
        }
        if field == "summary" || field == "short" {
            if !text.ends_with('.') {
                problems.push(format!(
                    "{place}: must be a sentence ending with a full stop"
                ));
            }
            if sentence_break(&text) {
                problems.push(format!(
                    "{place}: must be ONE sentence (README: 'One plain sentence')"
                ));
            }
        }
    }
    problems
}

fn has_duplicates(v: &[String]) -> bool {
    v.iter().collect::<BTreeSet<_>>().len() != v.len()
}

/// Every `terms` / `see_also` key exists in the glossary; no duplicates; no self links; unique titles.
fn check_links(b: &Bundle) -> Vec<String> {
    let mut problems = Vec::new();
    for (key, t) in &b.glossary {
        for k in t.see_also.iter().filter(|k| !has(&b.glossary, k)) {
            problems.push(format!(
                "glossary.json: '{key}'.see_also -> unknown term '{k}'"
            ));
        }
        if t.see_also.contains(key) {
            problems.push(format!("glossary.json: '{key}'.see_also links to itself"));
        }
        if has_duplicates(&t.see_also) {
            problems.push(format!("glossary.json: '{key}'.see_also has duplicates"));
        }
    }
    for (file, key, e) in explain_items(b) {
        for k in e.terms.iter().filter(|k| !has(&b.glossary, k)) {
            problems.push(format!("{file}: '{key}'.terms -> unknown term '{k}'"));
        }
        if has_duplicates(&e.terms) {
            problems.push(format!("{file}: '{key}'.terms has duplicates"));
        }
    }
    let mut titles: Vec<(String, &String)> = Vec::new();
    for (key, t) in &b.glossary {
        let folded = t.title.to_lowercase();
        match titles.iter().find(|(f, _)| *f == folded) {
            Some((_, other)) if *other != key => {
                problems.push(format!(
                    "glossary.json: '{key}' and '{other}' have the same title '{}'",
                    t.title
                ));
            }
            Some(_) => {}
            None => titles.push((folded, key)),
        }
    }
    problems
}

fn jargon(text: &str) -> BTreeSet<&'static str> {
    JARGON_RE
        .iter()
        .filter(|(rx, _, not_after)| {
            rx.find_iter(text)
                .any(|m| not_after.is_none_or(|p| !text[..m.start()].to_lowercase().ends_with(p)))
        })
        .map(|(_, key, _)| *key)
        .collect()
}

/// README rule 7: a glossary word used in the text is listed in `terms` (or `see_also`). Returns (problems in strict
/// files, advice for the others).
fn check_jargon(b: &Bundle, strict: &[&str]) -> (Vec<String>, Vec<String>) {
    let mut hard = Vec::new();
    let mut soft = Vec::new();
    for (key, t) in &b.glossary {
        let missing = jargon(&format!("{} {}", t.short, t.long))
            .into_iter()
            .filter(|m| !t.see_also.iter().any(|s| s == m) && m != key);
        if strict.contains(&"glossary.json") {
            hard.extend(missing.map(|m| {
                format!("glossary.json: '{key}' uses '{m}' but does not list it in see_also")
            }));
        }
    }
    for (file, key, e) in explain_items(b) {
        let text = [
            &e.summary,
            &e.details,
            &e.when_to_change,
            &e.risks,
            &e.example,
        ]
        .map(String::as_str)
        .join(" ");
        let missing = jargon(&text)
            .into_iter()
            .filter(|m| !e.terms.iter().any(|t| t == m))
            .map(|m| format!("{file}: '{key}' uses '{m}' but does not list it in terms"));
        if strict.contains(&file.as_str()) {
            hard.extend(missing);
        } else {
            soft.extend(missing);
        }
    }
    (hard, soft)
}

/// Names that identify whoever runs the tests (account, home directory, host). Read at run time so the public
/// repository never has to contain them. Short and generic names are skipped to avoid false alarms.
fn private_markers() -> Vec<String> {
    let mut names: BTreeSet<String> = BTreeSet::new();
    if let Some(home) = std::env::var_os("HOME")
        && let Some(n) = Path::new(&home).file_name()
    {
        names.insert(n.to_string_lossy().into_owned());
    }
    if let Ok(h) = nix::unistd::gethostname() {
        names.insert(
            h.to_string_lossy()
                .split('.')
                .next()
                .unwrap_or("")
                .to_string(),
        );
    }
    let user = ["LOGNAME", "USER", "LNAME", "USERNAME"]
        .iter()
        .find_map(|v| std::env::var(v).ok().filter(|s| !s.is_empty()))
        .or_else(|| {
            nix::unistd::User::from_uid(nix::unistd::getuid())
                .ok()
                .flatten()
                .map(|u| u.name)
        });
    names.extend(user);
    names
        .into_iter()
        .map(|n| n.to_lowercase())
        .filter(|n| len(n) >= 6 && !COMMON_NAMES.contains(&n.as_str()))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn is_word(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// `(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?!\w)` (`re.finditer`): the dotted quads of a line.
fn ipv4_candidates(line: &str) -> Vec<&str> {
    let chars: Vec<(usize, char)> = line.char_indices().collect();
    let digits = |mut i: usize| -> usize {
        let start = i;
        while i < chars.len() && i - start < 3 && chars[i].1.is_ascii_digit() {
            i += 1;
        }
        i - start
    };
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let before_ok = i == 0 || !(is_word(chars[i - 1].1) || chars[i - 1].1 == '.');
        let mut j = i;
        let mut ok = before_ok;
        for part in 0..4 {
            if !ok {
                break;
            }
            let n = digits(j);
            if n == 0 {
                ok = false;
                break;
            }
            j += n;
            if part < 3 {
                if j < chars.len() && chars[j].1 == '.' {
                    j += 1;
                } else {
                    ok = false;
                }
            }
        }
        if ok && !(j < chars.len() && is_word(chars[j].1)) {
            let end = chars.get(j).map_or(line.len(), |c| c.0);
            out.push(&line[chars[i].0..end]);
            i = j;
        } else {
            i += 1;
        }
    }
    out
}

fn in_documentation_net(ip: std::net::Ipv4Addr) -> bool {
    DOCUMENTATION_NETS.iter().any(|(net, bits)| {
        let mask = u32::MAX << (32 - bits);
        u32::from(ip) & mask == u32::from(std::net::Ipv4Addr::from(*net)) & mask
    })
}

/// `(?<![A-Za-z0-9])<marker>(?![A-Za-z0-9])`, case-insensitive.
fn has_marker(line: &str, rx: &Regex) -> bool {
    let mut at = 0;
    while let Some(m) = rx.find_at(line, at) {
        let before = line[..m.start()].chars().next_back();
        let after = line[m.end()..].chars().next();
        if !before.is_some_and(|c| c.is_ascii_alphanumeric())
            && !after.is_some_and(|c| c.is_ascii_alphanumeric())
        {
            return true;
        }
        at = m.start() + line[m.start()..].chars().next().map_or(1, char::len_utf8);
    }
    false
}

/// file label -> text. Reports file:line and the kind of problem, never the private value itself.
fn scan_publish_safety(files: &[(String, String)], markers: Option<&[String]>) -> Vec<String> {
    let mut problems = Vec::new();
    let markers: Vec<String> = markers.map_or_else(private_markers, <[String]>::to_vec);
    let marker_rx: Vec<Regex> = markers
        .iter()
        .map(|m| re(&format!("(?i){}", regex::escape(m))))
        .collect();
    for (label, text) in files {
        for (n, line) in py_splitlines(text).into_iter().enumerate() {
            let place = format!("{label}:{}", n + 1);
            if HOME_PATH.is_match(line) {
                problems.push(format!("{place}: absolute home-directory path"));
            }
            if EMAIL.is_match(line) {
                problems.push(format!("{place}: e-mail address"));
            }
            for cand in ipv4_candidates(line) {
                // not an address (e.g. 999.1.1.1, or a leading zero) -> skipped, as ipaddress.ip_address refuses it
                if let Ok(ip) = cand.parse::<std::net::Ipv4Addr>()
                    && !in_documentation_net(ip)
                {
                    problems.push(format!(
                        "{place}: IP address outside the documentation ranges (use 192.0.2.x, 198.51.100.x, \
                         203.0.113.x or 10.x.x.x)"
                    ));
                }
            }
            for (kind, rx) in SECRET_SHAPES.iter() {
                if rx.is_match(line) {
                    problems.push(format!("{place}: looks like a {kind}"));
                }
            }
            if marker_rx.iter().any(|rx| has_marker(line, rx)) {
                problems.push(format!(
                    "{place}: the name of the account or machine running the tests"
                ));
            }
        }
    }
    problems
}

fn wrap(keys: &[String]) -> String {
    format!("    {}", keys.join(", "))
}

// ---- loading and shape ----------------------------------------------------------------------------------------------
#[test]
fn test_bundle_loads() {
    let missing: Vec<&str> = SECTION_OF_FILE
        .iter()
        .map(|(f, _)| *f)
        .filter(|f| !learn_dir().join(f).is_file())
        .collect();
    assert!(
        missing.is_empty(),
        "missing learn file(s): {}",
        missing.join(", ")
    );
    let b = ec_hub::learn::bundle().expect("bundle");
    assert_eq!(b.version, 1);
    assert!(!b.glossary.is_empty() && !b.metrics.is_empty() && !b.concepts.is_empty());
}

#[test]
fn test_files_are_well_formed() {
    let root = learn_dir();
    let mut problems = Vec::new();
    for (path, section) in learn_files(&root) {
        let rel = rel_of(&path, &root);
        let raw = std::fs::read(&path).expect("read");
        if raw.starts_with(b"\xef\xbb\xbf") {
            problems.push(format!("{rel}: starts with a byte-order mark"));
        }
        match String::from_utf8(raw) {
            Ok(text) => problems.extend(check_file_text(&rel, &section, &text)),
            Err(e) => problems.push(format!("{rel}: not UTF-8 ({e})")),
        }
    }
    expect_none("learn files that do not follow learn/README.md", &problems);
}

#[test]
fn test_limits_match_readme() {
    let text = std::fs::read_to_string(learn_dir().join("README.md")).expect("README");
    let explain: BTreeMap<String, usize> =
        re(r"(?m)^\|\s*`(summary|details|when_to_change|risks|example)`\s*\|\s*(\d+)\s*chars")
            .captures_iter(&text)
            .map(|m| (m[1].to_string(), m[2].parse().expect("int")))
            .collect();
    let term: BTreeMap<String, usize> = re(r"`(short|long)`\s*\((\d+)\s*chars\)")
        .captures_iter(&text)
        .map(|m| (m[1].to_string(), m[2].parse().expect("int")))
        .collect();
    let want = |l: &[(&str, usize)]| {
        l.iter()
            .map(|(k, v)| ((*k).to_string(), *v))
            .collect::<BTreeMap<_, _>>()
    };
    assert_eq!(
        explain,
        want(EXPLAIN_LIMITS),
        "README Explain limits differ from this test's"
    );
    assert_eq!(
        term,
        want(TERM_LIMITS),
        "README Term limits differ from this test's"
    );
}

#[test]
fn test_limits() {
    expect_none(
        "text over the limits of learn/README.md or empty",
        &check_limits(&bundle()),
    );
}

#[test]
fn test_required_text() {
    expect_none(
        "fields a reader of glossary / metrics / concepts must find",
        &check_required_text(&bundle()),
    );
}

#[test]
fn test_hygiene() {
    expect_none("wording and formatting problems", &check_hygiene(&bundle()));
}

#[test]
fn test_links_resolve() {
    expect_none(
        "broken or duplicate glossary links",
        &check_links(&bundle()),
    );
}

#[test]
fn test_jargon_is_linked() {
    let (hard, soft) = check_jargon(&bundle(), STRICT_FILES);
    if !soft.is_empty() {
        eprintln!(
            "{}",
            report(
                "per-engine text uses glossary words without listing them in terms (advice)",
                &soft
            )
        );
    }
    expect_none(
        "glossary / metrics / concepts use jargon without linking it",
        &hard,
    );
}

#[test]
fn test_advice_engine_text_is_not_thin() {
    // Advice only: a parameter or preset popover with just a summary is thin (README shows all fields filled).
    let items = explain_items(&bundle());
    let mut thin: Vec<String> = items
        .iter()
        .filter(|(f, _, e)| {
            (f.starts_with("params/") || f.starts_with("presets/")) && e.details.trim().is_empty()
        })
        .map(|(f, k, _)| format!("{f}: '{k}' has no details"))
        .collect();
    thin.extend(
        items
            .iter()
            .filter(|(f, _, e)| {
                f.starts_with("params/")
                    && !e.details.trim().is_empty()
                    && e.when_to_change.trim().is_empty()
            })
            .map(|(f, k, _)| format!("{f}: '{k}' has no when_to_change")),
    );
    if !thin.is_empty() {
        eprintln!("{}", report("thin per-engine explanations (advice)", &thin));
    }
}

// ---- coverage: glossary, metrics, concepts --------------------------------------------------------------------------
#[test]
fn test_required_glossary_terms() {
    let b = bundle();
    let missing: Vec<String> = REQUIRED_TERMS
        .iter()
        .filter(|k| !has(&b.glossary, k))
        .map(|k| (*k).to_string())
        .collect();
    assert!(
        missing.is_empty(),
        "{} required glossary term(s) missing (engine specialists link to them):\n{}",
        missing.len(),
        wrap(&missing)
    );
    assert_eq!(
        REQUIRED_TERMS.iter().collect::<BTreeSet<_>>().len(),
        REQUIRED_TERMS.len(),
        "a duplicate"
    );
}

#[test]
fn test_metric_coverage() {
    let b = bundle();
    let missing: Vec<String> = CANONICAL_METRICS
        .iter()
        .chain(HOST_METRICS)
        .filter(|k| !has(&b.metrics, k))
        .map(|k| (*k).to_string())
        .collect();
    assert!(
        missing.is_empty(),
        "{} metric(s) without an explanation in metrics.json:\n{}",
        missing.len(),
        wrap(&missing)
    );
}

#[test]
fn test_canonical_metrics_match_architecture_doc() {
    let text = std::fs::read_to_string(repo_root().join("docs/ARCHITECTURE.md"))
        .expect("docs/ARCHITECTURE.md");
    let m = re(r"(?sm)^## 5\. Canonical metric keys[^\n]*\n(.*?)^## ")
        .captures(&text)
        .expect("ARCHITECTURE.md has no '## 5. Canonical metric keys' section any more: update this test");
    let listed: BTreeSet<&str> = re(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
        .find_iter(&m[1])
        .map(|x| x.as_str())
        .collect();
    let want: BTreeSet<&str> = CANONICAL_METRICS.iter().copied().collect();
    assert_eq!(
        listed, want,
        "ARCHITECTURE.md section 5 and this test disagree"
    );
}

#[test]
fn test_ui_requested_keys_are_explained() {
    // Contract with the frontend: every key listed in frontend/js/learn-keys.js has an explanation.
    let (families, metrics) = ui_requested_keys().expect("frontend/js/learn-keys.js");
    let n: usize = families.iter().map(|(_, ids)| ids.len()).sum();
    assert!(
        n >= 40 && metrics.len() >= 15,
        "could not read the key lists from frontend/js/learn-keys.js any more: update ui_requested_keys()"
    );
    let b = bundle();
    let mut missing: Vec<String> = families
        .iter()
        .flat_map(|(fam, ids)| ids.iter().map(move |i| format!("{fam}.{i}")))
        .filter(|k| !has(&b.concepts, k))
        .map(|k| format!("concept {k}"))
        .collect();
    missing.extend(
        metrics
            .iter()
            .filter(|k| !has(&b.metrics, k) && !has(&b.concepts, k))
            .map(|k| format!("metric {k}")),
    );
    expect_none(
        "keys the UI asks for (frontend/js/learn-keys.js) that have no explanation",
        &missing,
    );
}

#[test]
fn test_concept_family_coverage() {
    // one Rust test for the parametrised ids (every family but `check`, in tests/test_learn_check_codes.rs)
    let b = bundle();
    let mut problems = Vec::new();
    for (family, required) in required_concepts() {
        let mut missing: Vec<String> = required
            .iter()
            .filter(|(v, _)| !has(&b.concepts, &format!("{family}.{v}")))
            .map(|(v, w)| format!("{family}.{v}   (from {w})"))
            .collect();
        missing.sort();
        missing.dedup();
        problems.extend(missing);
    }
    assert_eq!(required_concepts().len(), FAMILIES.len() - 1); // all but `check`
    expect_none("concepts.json has no explanation for these keys", &problems);
}

#[test]
fn test_root_kinds_match_protocol_doc() {
    let text = std::fs::read_to_string(repo_root().join("docs/FLEET-PROTOCOL.md"))
        .expect("docs/FLEET-PROTOCOL.md");
    let m = re(r"(?sm)^### 6\.1 Root kinds\n(.*?)^### ")
        .captures(&text)
        .expect("FLEET-PROTOCOL.md has no '### 6.1 Root kinds' section any more: update this test");
    let kinds: BTreeSet<&str> = re(r"(?m)^\|\s*`([a-z-]+)`\s*\|")
        .captures_iter(&m[1])
        .map(|c| c.get(1).expect("kind").as_str())
        .collect();
    assert_eq!(
        kinds,
        ROOT_KINDS.iter().copied().collect::<BTreeSet<_>>(),
        "root kinds in the protocol doc differ"
    );
}

// ---- coverage: engine parameters and presets ------------------------------------------------------------------------
#[test]
fn test_adapters_registered() {
    // Python's registry skips a module that fails to import; Rust's adapters are compiled in, so this checks that
    // every DEFAULT_MODULES engine is registered.
    let ids: BTreeSet<String> = load_default_adapters()
        .list()
        .iter()
        .map(|a| a.id().to_string())
        .collect();
    let absent: Vec<&str> = ["vllm", "sglang", "trtllm", "llamacpp", "ollama"]
        .into_iter()
        .filter(|m| !ids.contains(*m))
        .collect();
    assert!(absent.is_empty(), "adapter(s) not registered: {absent:?}");
    assert!(
        ids.contains("vllm") && ids.contains("sglang"),
        "registered adapters: {ids:?}"
    );
}

// ---- publish safety -------------------------------------------------------------------------------------------------
#[test]
fn test_publish_safety() {
    // Public repository: scan every learn file, README included, as raw text.
    let root = learn_dir();
    let files: Vec<(String, String)> = walk(&root)
        .into_iter()
        .filter(|p| p.extension().is_some_and(|x| x == "json" || x == "md"))
        .map(|p| {
            (
                rel_of(&p, &root),
                std::fs::read_to_string(&p).expect("utf-8"),
            )
        })
        .collect();
    assert!(files.len() >= 13, "{} files", files.len());
    expect_none(
        "content that must not be published",
        &scan_publish_safety(&files, None),
    );
}

// ---- the gate tests itself ------------------------------------------------------------------------------------------
/// A tiny valid bundle; tests replace its sections.
fn small() -> Bundle {
    let term = Term {
        title: "KV cache".into(),
        short: "The notes an engine keeps.".into(),
        long: "The notes an engine keeps while it works.".into(),
        see_also: Vec::new(),
    };
    Bundle {
        glossary: vec![("kv_cache".into(), term)],
        metrics: vec![(
            "load1".into(),
            Explain {
                details: "d".into(),
                when_to_change: "w".into(),
                risks: "r".into(),
                example: "e".into(),
                ..ex("The load.")
            },
        )],
        concepts: vec![(
            "state.ready".into(),
            Explain {
                details: "d".into(),
                when_to_change: "w".into(),
                ..ex("Ready.")
            },
        )],
        ..Bundle::default()
    }
}

fn with_metric(e: Explain) -> Bundle {
    Bundle {
        metrics: vec![("load1".into(), e)],
        ..small()
    }
}

fn with_concept(key: &str, e: Explain) -> Bundle {
    Bundle {
        concepts: vec![(key.into(), e)],
        ..small()
    }
}

#[test]
fn test_gate_flags_limit_and_text_problems() {
    let bad = Explain {
        details: "y".repeat(901),
        when_to_change: " padded ".into(),
        risks: "  ".into(),
        example: "z".repeat(201),
        ..ex(&"x".repeat(161))
    };
    let problems = check_limits(&with_metric(bad));
    let fields: BTreeSet<&str> = problems
        .iter()
        .filter(|p| p.contains("chars"))
        .filter_map(|p| p.split("'.").nth(1)?.split_whitespace().next())
        .collect();
    assert!(
        ["summary", "details", "example"]
            .iter()
            .all(|f| fields.contains(f)),
        "{problems:?}"
    );
    let two = Explain {
        details: "Do not wait; it may cause issues. TODO".into(),
        ..ex("One thing. Another thing.")
    };
    let hygiene = check_hygiene(&with_metric(two)).join(" | ");
    assert!(
        hygiene.contains("ONE sentence")
            && hygiene.contains("vague")
            && hygiene.contains("placeholder"),
        "{hygiene}"
    );
    assert!(
        check_hygiene(&with_metric(ex("No stop")))
            .join(" | ")
            .contains("full stop")
    );
    assert!(
        check_hygiene(&with_metric(ex("A seamless fix.")))
            .join(" | ")
            .contains("promotional")
    );
}

#[test]
fn test_gate_flags_missing_required_text() {
    let thin = ex("Only a summary.");
    assert_eq!(check_required_text(&with_metric(thin.clone())).len(), 4); // details, when, risks, example
    assert_eq!(
        check_required_text(&with_concept("state.ready", thin.clone())),
        [
            "concepts.json: 'state.ready' has no details",
            "concepts.json: 'state.ready' has no when_to_change"
        ]
    );
    assert_eq!(
        check_required_text(&with_concept("page.chat", thin)),
        ["concepts.json: 'page.chat' has no details"]
    );
}

#[test]
fn test_gate_flags_dangling_links_and_unlinked_jargon() {
    let linked = Explain {
        details: "d".into(),
        terms: vec!["nonsense".into(), "nonsense".into()],
        ..ex("See the VRAM.")
    };
    let problems = check_links(&with_concept("state.ready", linked.clone()));
    assert!(
        problems
            .iter()
            .any(|p| p.contains("unknown term 'nonsense'"))
            && problems.iter().any(|p| p.contains("duplicates"))
    );
    let glossary = vec![(
        "kv_cache".to_string(),
        Term {
            title: "KV cache".into(),
            short: "s.".into(),
            long: "Longer text.".into(),
            see_also: vec!["kv_cache".into(), "gone".into()],
        },
    )];
    let problems = check_links(&Bundle {
        glossary,
        ..small()
    });
    assert!(
        problems.iter().any(|p| p.contains("itself"))
            && problems.iter().any(|p| p.contains("unknown term 'gone'"))
    );
    let (hard, soft) = check_jargon(
        &Bundle {
            metrics: Vec::new(),
            ..with_concept("state.ready", linked)
        },
        STRICT_FILES,
    );
    assert_eq!(
        hard,
        ["concepts.json: 'state.ready' uses 'vram' but does not list it in terms"]
    );
    assert!(soft.is_empty());
    let (hard, soft) = check_jargon(
        &Bundle {
            concepts: Vec::new(),
            metrics: Vec::new(),
            ..small()
        },
        &[],
    );
    assert!(hard.is_empty() && soft.is_empty());
    let params = vec![(
        "vllm".to_string(),
        vec![("gpu_memory_utilization".to_string(), ex("Uses VRAM."))],
    )];
    let (hard, soft) = check_jargon(
        &Bundle {
            params,
            ..Bundle::default()
        },
        STRICT_FILES,
    );
    assert!(hard.is_empty());
    assert_eq!(
        soft,
        ["params/vllm.json: 'gpu_memory_utilization' uses 'vram' but does not list it in terms"]
    );
}

#[test]
fn test_gate_flags_malformed_files() {
    let ok = r#"{"version": 1, "metrics": {"a_b": {"summary": "S."}, "c": {"summary": "S."}}}"#;
    assert!(check_file_text("metrics.json", "metrics", ok).is_empty());
    assert!(!check_file_text("x.json", "", ok).is_empty());
    assert!(
        check_file_text("metrics.json", "metrics", "{")
            .iter()
            .any(|p| p.contains("invalid JSON"))
    );
    let any = |text: &str, section: &str, needle: &str| {
        let file = if section == "concepts" {
            "concepts.json"
        } else {
            "metrics.json"
        };
        check_file_text(file, section, text)
            .iter()
            .any(|p| p.contains(needle))
    };
    assert!(any(
        r#"{"version": 1, "metrics": {"a": {"summary": "S."}, "a": {"summary": "T."}}}"#,
        "metrics",
        "duplicate key 'a'"
    ));
    assert!(any(
        r#"{"version": 1, "metrics": {"b": {"summary": "S."}, "a": {"summary": "S."}}}"#,
        "metrics",
        "not sorted"
    ));
    assert!(any(
        r#"{"version": 1, "metrics": {"a": {"summary": "S.", "when_to_chnage": "x"}}}"#,
        "metrics",
        "unknown field 'when_to_chnage'"
    ));
    assert!(
        !check_file_text("metrics.json", "metrics", r#"{"version": 2, "terms": {}}"#).is_empty()
    );
    assert!(any(
        r#"{"version": 1, "metrics": {"Bad Key": {"summary": "S."}}}"#,
        "metrics",
        "malformed key"
    ));
    assert!(any(
        r#"{"version": 1, "concepts": {"colour.red": {"summary": "S."}}}"#,
        "concepts",
        "unknown concept family"
    ));
    assert!(
        check_file_text(
            "presets/vllm.json",
            "presets",
            r#"{"version": 1, "presets": {"max-throughput": {"summary": "S."}}}"#
        )
        .is_empty()
    );
}

#[test]
fn test_gate_flags_unsafe_content() {
    let clean = vec![(
        "a.json".to_string(),
        "Use 192.0.2.7 or 10.1.2.3, and ~/.cache/huggingface. Version 0.5.14. Token: [set, 37 chars].".to_string(),
    )];
    assert!(scan_publish_safety(&clean, Some(&[])).is_empty());
    let leaky = [
        format!(r#"path "{}someone/models""#, HOME_PREFIXES[0]),
        "mail me at someone@example.com".to_string(),
        format!("server at {} and 127.0.0.{}", ["8"; 4].join("."), 1), // a public and a loopback address
        format!("token hf_{}", "A".repeat(34)),
        format!("key sk-{}", "b".repeat(30)),
        "owner is Exampleperson here".to_string(),
        concat!("-----BEGIN OPENSSH ", "PRIVATE KEY-----").to_string(),
    ]
    .join("\n");
    let found = scan_publish_safety(
        &[("a.json".to_string(), leaky)],
        Some(&["exampleperson".to_string()]),
    );
    assert!(
        found.len() == 8 && found.iter().all(|p| p.starts_with("a.json:")),
        "{found:?}"
    );
    assert!(
        !found
            .iter()
            .any(|p| p.to_lowercase().contains("exampleperson")),
        "findings must not echo the private value"
    );
    let lines: Vec<&str> = found.iter().filter_map(|p| p.split(':').nth(1)).collect();
    assert_eq!(lines, ["1", "2", "3", "3", "4", "5", "6", "7"]);
}
