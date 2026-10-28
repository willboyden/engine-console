//! Shared helpers of the vLLM and SGLang adapter tests. Include with
//! `#[path = "common/ws_a1_kit.rs"] mod ws_a1_kit;` (or `../common/` from a subdirectory module).

#![allow(dead_code, clippy::panic, clippy::expect_used, unreachable_pub)]

use std::path::{Path, PathBuf};
use std::sync::Once;

use ec_hub::adapters::base::{FloatMap, Hardware, ModelInfo, Obj};
use serde_json::{Map, Value};

pub fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/ws_a1")
}

/// A committed fixture of the pinned images (`fixtures/<engine>/<name>`).
pub fn fixture(engine: &str, name: &str) -> String {
    let p = data_dir().join("fixtures").join(engine).join(name);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

pub fn fixture_json(engine: &str, name: &str) -> Value {
    serde_json::from_str(&fixture(engine, name)).expect("fixture json")
}

/// The learn content the hub ships (crates/ec-hub/learn, embedded at build time).
pub fn learn_file(folder: &str, engine: &str) -> Value {
    let p = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("learn")
        .join(folder)
        .join(format!("{engine}.json"));
    serde_json::from_str(
        &std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display())),
    )
    .expect("learn json")
}

/// `set(doc["flags"])`: the flag names of a fixture whose "flags" is a list or an object.
pub fn flag_names(v: &Value) -> std::collections::BTreeSet<String> {
    match v {
        Value::Array(a) => a
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_string)
            .collect(),
        Value::Object(o) => o.keys().cloned().collect(),
        _ => std::collections::BTreeSet::new(),
    }
}

pub fn obj(v: Value) -> Obj {
    match v {
        Value::Object(m) => m,
        other => panic!("not an object: {other}"),
    }
}

pub fn get(m: &FloatMap, key: &str) -> Option<f64> {
    m.iter().find(|(k, _)| k == key).map(|(_, v)| *v)
}

pub fn keys(m: &FloatMap) -> Vec<&str> {
    m.iter().map(|(k, _)| k.as_str()).collect()
}

/// pytest.approx default: relative 1e-6.
pub fn approx(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-6 * b.abs().max(1e-12)
}

/// The vLLM tests' `hw` fixture: two RTX PRO 6000 Blackwell cards.
pub fn hw_vllm() -> Hardware {
    Hardware {
        gpu_ids: vec![0, 1],
        gpu_uuids: vec!["GPU-aaaa".into(), "GPU-bbbb".into()],
        gpu_names: vec!["RTX PRO 6000 Blackwell".into(); 2],
        gpu_total_gib: vec![95.6, 95.6],
        gpu_free_gib: vec![94.0, 94.0],
        compute_capability: "12.0".into(),
        host_ram_gib: 246.0,
        host_id: "local".into(),
        arch: "x86_64".into(),
        unified_memory: false,
    }
}

/// The SGLang tests' `hw(n, cc, free)`.
pub fn hw_n(n: usize, cc: &str, free: Option<Vec<f64>>) -> Hardware {
    Hardware {
        gpu_ids: (0..n as i64).collect(),
        gpu_uuids: (0..n).map(|i| format!("GPU-{i}")).collect(),
        gpu_names: vec!["RTX PRO 6000".into(); n],
        gpu_total_gib: vec![95.0; n],
        gpu_free_gib: free.unwrap_or_else(|| vec![94.0; n]),
        compute_capability: cc.into(),
        host_ram_gib: 246.0,
        host_id: "local".into(),
        arch: "x86_64".into(),
        unified_memory: false,
    }
}

pub fn model(repo: &str) -> ModelInfo {
    ModelInfo {
        repo_id: repo.into(),
        ..ModelInfo::default()
    }
}

/// The vLLM tests' `dense` fixture.
pub fn dense() -> ModelInfo {
    ModelInfo {
        repo_id: "example-org/chat-model-36B".into(),
        architectures: vec!["LlamaForCausalLM".into()],
        num_params: Some(36_000_000_000),
        weight_bytes: Some(72 * 1024 * 1024 * 1024),
        num_attention_heads: Some(64),
        num_kv_heads: Some(8),
        max_position_embeddings: Some(131_072),
        pipeline_tag: Some("text-generation".into()),
        ..ModelInfo::default()
    }
}

static QUIET: Once = Once::new();

thread_local! {
    /// Set while `raises` runs its closure: only then is a Python-exception panic expected (and kept quiet).
    static EXPECTING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Run `f`, expecting the adapter to raise a Python exception (a panic from `py::raise`). The panic message is kept out
/// of the test output; any other panic still prints.
pub fn raises<R>(f: impl FnOnce() -> R) -> bool {
    QUIET.call_once(|| {
        let default = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let msg = info
                .payload()
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| {
                    info.payload()
                        .downcast_ref::<&str>()
                        .map(|s| (*s).to_string())
                })
                .unwrap_or_default();
            let python = [
                "TypeError:",
                "ValueError:",
                "ZeroDivisionError:",
                "AttributeError:",
                "RecursionError:",
                "OverflowError:",
                "JSONDecodeError:",
            ];
            if !(EXPECTING.with(std::cell::Cell::get) && python.iter().any(|p| msg.starts_with(p)))
            {
                default(info);
            }
        }));
    });
    EXPECTING.with(|e| e.set(true));
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).is_err();
    EXPECTING.with(|e| e.set(false));
    r
}

/// A golden float: a number, or {"__float__": "nan"|"inf"|"-inf"}, or a repr string.
pub fn golden_float(v: &Value) -> f64 {
    match v {
        Value::Number(n) => n.as_f64().expect("f64"),
        Value::String(s) => ec_hub::adapters::vllm_parse::py::float_str(s).expect("repr"),
        Value::Object(o) => match o.get("__float__").and_then(Value::as_str) {
            Some("nan") => f64::NAN,
            Some("inf") => f64::INFINITY,
            Some("-inf") => f64::NEG_INFINITY,
            _ => panic!("bad float {v}"),
        },
        _ => panic!("bad float {v}"),
    }
}

/// A FloatMap against its golden `[[key, repr], ...]`: same keys in the same order, same bits (NaN == NaN).
pub fn floatmap_matches(got: &FloatMap, golden: &Value) -> Result<(), String> {
    let want = golden.as_array().ok_or("golden not a list")?;
    let w: Vec<(String, f64)> = want
        .iter()
        .map(|p| {
            let k = p[0].as_str().unwrap_or_default().to_string();
            (k, golden_float(&p[1]))
        })
        .collect();
    let same = got.len() == w.len()
        && got.iter().zip(&w).all(|((ka, va), (kb, vb))| {
            ka == kb && (va.to_bits() == vb.to_bits() || (va.is_nan() && vb.is_nan()))
        });
    if same {
        Ok(())
    } else {
        Err(format!("got {got:?}\nwant {w:?}"))
    }
}

/// Undo the generator's `enc()`: {"__float__": ...} cannot be a JSON number, so callers that need it as a value get
/// null (only model/hardware fields ever carry finite floats in the vectors).
pub fn plain(v: &Value) -> Value {
    match v {
        Value::Object(o) if o.contains_key("__float__") => Value::Null,
        Value::Object(o) => Value::Object(
            o.iter()
                .map(|(k, x)| (k.clone(), plain(x)))
                .collect::<Map<_, _>>(),
        ),
        Value::Array(a) => Value::Array(a.iter().map(plain).collect()),
        other => other.clone(),
    }
}
