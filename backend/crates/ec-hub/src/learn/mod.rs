//! Learn-mode content (port of learn/__init__.py).
//!
//! The JSON files stay where the Python package keeps them (backend/src/engine_console/learn/, the single source
//! during the port; PORTING.md D6) and are embedded at build time. Every entry is validated into `Term`/`Explain`
//! and dumped back, so the bundle carries pydantic's defaults and drops unknown keys exactly like `model_dump()`.
//! A malformed file fails the first call (and the tests) instead of showing a half-empty popover.

use std::path::Path;
use std::sync::{Arc, OnceLock};

use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Map, Value};

use crate::adapters::base::Explain;
use crate::domain::learn::{LearnBundle, Term};

macro_rules! learn_file {
    ($path:literal) => {
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/learn/", $path))
    };
}

/// (engine id, params file, presets file), in `sorted(glob("*.json"))` order.
const ENGINES: &[(&str, &str, &str)] = &[
    (
        "llamacpp",
        learn_file!("params/llamacpp.json"),
        learn_file!("presets/llamacpp.json"),
    ),
    (
        "ollama",
        learn_file!("params/ollama.json"),
        learn_file!("presets/ollama.json"),
    ),
    (
        "sglang",
        learn_file!("params/sglang.json"),
        learn_file!("presets/sglang.json"),
    ),
    (
        "trtllm",
        learn_file!("params/trtllm.json"),
        learn_file!("presets/trtllm.json"),
    ),
    (
        "vllm",
        learn_file!("params/vllm.json"),
        learn_file!("presets/vllm.json"),
    ),
];

fn section(text: &str, name: &str, section: &str) -> Result<Map<String, Value>, String> {
    let doc: Value = serde_json::from_str(text).map_err(|e| format!("{name}: {e}"))?;
    match (doc.get("version"), doc.get(section)) {
        (Some(v), Some(Value::Object(m))) if v == &Value::from(1) => Ok(m.clone()),
        _ => Err(format!(
            "{name}: expected {{\"version\": 1, \"{section}\": {{...}}}}"
        )),
    }
}

/// Validate every value of a section as `T` and dump it back (defaults filled, unknown keys dropped).
fn typed<T: DeserializeOwned + Serialize>(
    m: Map<String, Value>,
    name: &str,
) -> Result<Map<String, Value>, String> {
    let mut out = Map::new();
    for (k, v) in m {
        let t: T = serde_json::from_value(v).map_err(|e| format!("{name}: {k}: {e}"))?;
        let dumped = serde_json::to_value(t).map_err(|e| format!("{name}: {k}: {e}"))?;
        out.insert(k, dumped);
    }
    Ok(out)
}

pub fn load_bundle() -> Result<LearnBundle, String> {
    let glossary = typed::<Term>(
        section(learn_file!("glossary.json"), "glossary.json", "terms")?,
        "glossary.json",
    )?;
    let metrics = typed::<Explain>(
        section(learn_file!("metrics.json"), "metrics.json", "metrics")?,
        "metrics.json",
    )?;
    let concepts = typed::<Explain>(
        section(learn_file!("concepts.json"), "concepts.json", "concepts")?,
        "concepts.json",
    )?;
    let mut params = Map::new();
    let mut presets = Map::new();
    for (engine, p, s) in ENGINES {
        let pn = format!("{engine}.json");
        params.insert(
            (*engine).into(),
            Value::Object(typed::<Explain>(section(p, &pn, "params")?, &pn)?),
        );
        presets.insert(
            (*engine).into(),
            Value::Object(typed::<Explain>(section(s, &pn, "presets")?, &pn)?),
        );
    }
    Ok(LearnBundle {
        version: 1,
        glossary,
        metrics,
        concepts,
        params,
        presets,
    })
}

/// The packaged content, loaded once. An invalid file is an error every time it is asked for.
pub fn bundle() -> Result<Arc<LearnBundle>, String> {
    static B: OnceLock<Result<Arc<LearnBundle>, String>> = OnceLock::new();
    B.get_or_init(|| load_bundle().map(Arc::new)).clone()
}

/// `LearnBundle()`: version 1, every section empty.
pub fn empty_bundle() -> LearnBundle {
    LearnBundle {
        version: 1,
        glossary: Map::new(),
        metrics: Map::new(),
        concepts: Map::new(),
        params: Map::new(),
        presets: Map::new(),
    }
}

/// The largest learn file read from a directory (the packaged ones are tens of KiB).
pub const MAX_LEARN_FILE: usize = 4 * 1024 * 1024;

/// `_read`: a missing file (or anything but a regular file: `Path.is_file()`) is an empty section; anything else must
/// be `{"version": 1, "<section>": {...}}`. The read itself is non-blocking, re-checked on the descriptor and capped,
/// so a file swapped for a FIFO after the check is an error, never a hang (hardening).
fn read_section(path: &Path, sect: &str) -> Result<Map<String, Value>, String> {
    if !path.is_file() {
        return Ok(Map::new());
    }
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let text = crate::fsutil::read_text(path, crate::fsutil::Follow::Yes, MAX_LEARN_FILE)
        .map_err(|e| format!("{name}: {e}"))?;
    section(&text, &name, sect)
}

/// `_per_engine`: every `<engine>.json` of a folder (sorted by name), keyed by its stem.
fn per_engine(folder: &Path, sect: &str) -> Result<Map<String, Value>, String> {
    let mut out = Map::new();
    if !folder.is_dir() {
        return Ok(out);
    }
    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(folder)
        .map_err(|e| e.to_string())?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    files.sort();
    for p in files {
        let stem = p
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let name = p
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        out.insert(
            stem,
            Value::Object(typed::<Explain>(read_section(&p, sect)?, &name)?),
        );
    }
    Ok(out)
}

/// `load_bundle(root)`: the learn content of a directory laid out like the packaged one (tests, and the place a
/// cutover moves the files to).
pub fn load_bundle_dir(root: &Path) -> Result<LearnBundle, String> {
    Ok(LearnBundle {
        version: 1,
        glossary: typed::<Term>(
            read_section(&root.join("glossary.json"), "terms")?,
            "glossary.json",
        )?,
        metrics: typed::<Explain>(
            read_section(&root.join("metrics.json"), "metrics")?,
            "metrics.json",
        )?,
        concepts: typed::<Explain>(
            read_section(&root.join("concepts.json"), "concepts")?,
            "concepts.json",
        )?,
        params: per_engine(&root.join("params"), "params")?,
        presets: per_engine(&root.join("presets"), "presets")?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A FIFO in a bundle directory is no file (Python's `is_file()`): an empty section at once, never a hang; an
    /// oversized file is an error (hardening finding 4).
    #[test]
    fn test_a_bundle_dir_with_a_fifo_or_a_huge_file_never_hangs() {
        let d = tempfile::tempdir().unwrap();
        std::fs::create_dir(d.path().join("params")).unwrap();
        let fifo = |p: &Path| {
            nix::unistd::mkfifo(p, nix::sys::stat::Mode::from_bits_truncate(0o600)).unwrap();
        };
        fifo(&d.path().join("params/vllm.json"));
        fifo(&d.path().join("glossary.json"));
        let t = std::time::Instant::now();
        let b = load_bundle_dir(d.path()).unwrap();
        assert!(b.glossary.is_empty() && b.params["vllm"] == Value::Object(Map::new()));
        assert!(t.elapsed() < std::time::Duration::from_secs(2));
        let big = format!(
            "{{\"version\": 1, \"metrics\": {{}}, \"pad\": \"{}\"}}",
            "x".repeat(MAX_LEARN_FILE)
        );
        std::fs::write(d.path().join("metrics.json"), big).unwrap();
        assert!(
            load_bundle_dir(d.path())
                .unwrap_err()
                .contains("metrics.json")
        );
    }

    #[test]
    fn test_bundle_loads_and_has_every_engine() {
        let b = bundle().unwrap();
        let engines: Vec<&String> = b.params.keys().collect();
        assert_eq!(engines, ["llamacpp", "ollama", "sglang", "trtllm", "vllm"]);
        assert!(!b.glossary.is_empty() && !b.metrics.is_empty() && !b.concepts.is_empty());
        let any = b.metrics.values().next().unwrap();
        for k in [
            "summary",
            "details",
            "when_to_change",
            "risks",
            "example",
            "terms",
            "level",
        ] {
            assert!(any.get(k).is_some(), "{k}");
        }
    }
}
