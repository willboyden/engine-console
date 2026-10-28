//! The lifecycle rows of backend/tests/learn/test_learn_content.py: the `check` concept family, whose codes come from
//! the emitters in the lifecycle service and the adapters, and the closed-family stale check that needs them. Python
//! scans its package with `ast` for `PreflightCheck(code=...)`, `Compat(code=...)` and `add(level, "code", ...)`; here
//! a scanner reads the hub's Rust sources for the same emitters (`pc("code", ...)`, `PreflightCheck { code: "..." }`,
//! `Compat { code: "..." }`, `add(level, "code", ...)`, `compat(level, "code", ...)`) and for raised problem codes
//! (`.code("...")`, `Problem::new(status, "code", ...)`).
//!
//! Closed value lists Python reads from `Literal` types are copied here (with their source) where Rust has no const
//! (the same lists as tests/test_learn_content.rs).

#![allow(
    clippy::redundant_clone,
    clippy::cloned_ref_to_slice_refs,
    clippy::uninlined_format_args,
    clippy::manual_let_else,
    clippy::expect_used,
    clippy::panic,
    clippy::unwrap_used,
    clippy::too_many_arguments,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    clippy::too_many_lines,
    clippy::needless_pass_by_value,
    clippy::type_complexity
)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use ec_hub::domain::fleet::{LIBRARY_FORMATS, LINK_STATES};
use ec_hub::domain::models::INSTANCE_STATES;
use ec_hub::services::hostmem::ALERT_CODES;
use regex::Regex;

/// Preflight codes named by the contract (ARCHITECTURE 10.4 / 10.11).
const V2_CHECK_CODES: &[&str] = &[
    "host_down",
    "monitor_only",
    "image_not_allowed",
    "image_missing",
    "model_missing",
    "format_unsupported",
    "port_in_use",
];
/// Raised by the lifecycle's launch build as a problem and shown by preflight as a check with the problem's code.
const DYNAMIC_CHECK_CODES: &[&str] = &["secret_in_argv", "invalid_launch"];
const V2_ALERT_CODES: &[&str] = &["gpu_hot", "vram_full"];
const ROOT_KINDS: &[&str] = &[
    "auto", "hf-home", "hf-hub", "plain", "gguf", "ollama", "other",
];
const MODES: &[&str] = &["monitor", "manage"];
const MANAGEMENT: &[&str] = &["managed", "script", "external"]; // domain/refs.py Management
const VERDICTS: &[&str] = &["fits", "tight", "wont_fit", "unknown"]; // domain/models.py Verdict
const HOST_RAM_VERDICTS: &[&str] = &["ok", "tight", "wont_fit", "unknown"]; // HostRamFit.verdict
const CONFIDENCES: &[&str] = &["high", "medium", "low"]; // domain/models.py Confidence
const DOWNLOAD_STATES: &[&str] = &[
    "queued",
    "running",
    "paused",
    "completed",
    "failed",
    "cancelled",
];
const DOWNLOAD_KINDS: &[&str] = &["hf", "copy"];

fn src_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src")
}

fn rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).collect();
    entries.sort();
    for p in entries {
        if p.is_dir() {
            rs_files(&p, out);
        } else if p.extension().is_some_and(|e| e == "rs") {
            out.push(p);
        }
    }
}

fn line_of(text: &str, at: usize) -> usize {
    text[..at].matches('\n').count() + 1
}

/// The text of the `{ ... }` that starts at the first `{` at or after `from` (balanced; strings not special-cased:
/// struct literals of emitters hold no braces in their strings).
fn braced(text: &str, from: usize) -> &str {
    let Some(open) = text[from..].find('{').map(|i| from + i) else {
        return "";
    };
    let mut depth = 0usize;
    for (i, c) in text[open..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return &text[open..=open + i];
                }
            }
            _ => {}
        }
    }
    &text[open..]
}

/// `emitted_codes(constructors, helpers, root)`: code -> ["rel/path.rs:line", ...] for every emitter with a literal
/// code. A struct literal `Name { .. code: "x" .. }` of a constructor counts, and so does a helper call whose second
/// argument is a literal code and whose first is a level literal or any expression (a computed level).
fn emitted_codes(
    constructors: &[&str],
    helpers: &[&str],
    root: &Path,
) -> BTreeMap<String, Vec<String>> {
    let mut files = Vec::new();
    rs_files(root, &mut files);
    let mut found: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let code_field = Regex::new(r#"\bcode:\s*"([A-Za-z0-9_]+)""#).unwrap();
    for f in files {
        let rel = f
            .strip_prefix(root)
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");
        let text = std::fs::read_to_string(&f).unwrap();
        let mut hits: Vec<(usize, String)> = Vec::new();
        for name in constructors {
            let re = Regex::new(&format!(r"\b{name}\s*\{{")).unwrap();
            for m in re.find_iter(&text) {
                if let Some(c) = code_field.captures(braced(&text, m.start())) {
                    hits.push((m.start(), c[1].to_string()));
                }
            }
        }
        for name in helpers {
            let re = Regex::new(&format!(
                // the level: a literal, a name, or an inline `if c { "block" } else { "warn" }` (the llama.cpp adapter)
                r#"\b{name}\(\s*("ok"|"warn"|"block"|[A-Za-z_][A-Za-z0-9_.]*|if [^{{}}(),]+\{{\s*"[a-z]+"\s*\}}\s*else\s*\{{\s*"[a-z]+"\s*\}})\s*,\s*"([A-Za-z0-9_]+)""#
            ))
            .unwrap();
            // the level may be computed (`add(level, "code", ...)`): any non-literal first argument counts too
            for c in re.captures_iter(&text) {
                hits.push((c.get(0).unwrap().start(), c[2].to_string()));
            }
        }
        hits.sort();
        for (at, code) in hits {
            found
                .entry(code)
                .or_default()
                .push(format!("{rel}:{}", line_of(&text, at)));
        }
    }
    found
}

/// Every preflight / compatibility code the hub emits today, with where.
fn check_codes() -> BTreeMap<String, Vec<String>> {
    // `compat(level, code, msg)`: the adapters' Rust spelling of Python's `Compat(level=..., code=...)`
    let mut out = emitted_codes(
        &["PreflightCheck", "Compat"],
        &["add", "compat"],
        &src_root(),
    );
    // the lifecycle's own helper `pc(code, level, message)` (Python writes `PreflightCheck(code=..., ...)`)
    let pc = Regex::new(r#"\bpc\(\s*"([A-Za-z0-9_]+)""#).unwrap();
    let mut files = Vec::new();
    rs_files(&src_root(), &mut files);
    for f in files {
        let rel = f
            .strip_prefix(src_root())
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");
        let text = std::fs::read_to_string(&f).unwrap();
        for c in pc.captures_iter(&text) {
            out.entry(c[1].to_string()).or_default().push(format!(
                "{rel}:{}",
                line_of(&text, c.get(0).unwrap().start())
            ));
        }
    }
    out
}

/// Codes of problems raised with a literal code (`.code("x")` or `Problem::new(status, "x", ..)`).
fn raised_codes() -> BTreeMap<String, Vec<String>> {
    let mut files = Vec::new();
    rs_files(&src_root(), &mut files);
    let res = [
        Regex::new(r#"\.code\(\s*"([A-Za-z0-9_]+)"\s*\)"#).unwrap(),
        Regex::new(r#"Problem::new\(\s*\d+\s*,\s*"([A-Za-z0-9_]+)""#).unwrap(),
    ];
    let mut out: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for f in files {
        let rel = f
            .strip_prefix(src_root())
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");
        let text = std::fs::read_to_string(&f).unwrap();
        for re in &res {
            for c in re.captures_iter(&text) {
                out.entry(c[1].to_string()).or_default().push(format!(
                    "{rel}:{}",
                    line_of(&text, c.get(0).unwrap().start())
                ));
            }
        }
    }
    out
}

fn concepts() -> Vec<String> {
    ec_hub::learn::bundle()
        .expect("bundle")
        .concepts
        .keys()
        .cloned()
        .collect()
}

fn check_required() -> BTreeMap<String, String> {
    let mut req: BTreeMap<String, String> = BTreeMap::new();
    for c in V2_CHECK_CODES {
        req.insert((*c).into(), "contract v2 (ARCHITECTURE 10.4, 10.11)".into());
    }
    for c in DYNAMIC_CHECK_CODES {
        req.insert(
            (*c).into(),
            "raised dynamically by the launch build (services/lifecycle.rs)".into(),
        );
    }
    for (c, w) in check_codes() {
        req.insert(c, w.iter().take(2).cloned().collect::<Vec<_>>().join(", ")); // a found literal wins
    }
    req
}

fn expect_none(what: &str, items: &[String]) {
    assert!(items.is_empty(), "{what}:\n  {}", items.join("\n  "));
}

/// Port of test_concept_family_coverage[check]: every check code the code emits (and the contract's) is explained.
#[test]
fn test_concept_family_coverage_check() {
    let have = concepts();
    let missing: Vec<String> = check_required()
        .iter()
        .filter(|(v, _)| !have.contains(&format!("check.{v}")))
        .map(|(v, w)| format!("check.{v}   (from {w})"))
        .collect();
    expect_none(
        "concepts.json has no explanation for these 'check.*' keys",
        &missing,
    );
}

/// Port of test_no_stale_concepts: closed families list exactly what exists now.
#[test]
fn test_no_stale_concepts() {
    let of = |v: &[&str]| v.iter().map(|s| (*s).to_string()).collect::<Vec<_>>();
    let mut required: BTreeMap<&str, Vec<String>> = BTreeMap::new();
    required.insert("state", of(INSTANCE_STATES));
    required.insert("link", of(LINK_STATES));
    required.insert("management", of(MANAGEMENT));
    required.insert("verdict", [of(VERDICTS), of(HOST_RAM_VERDICTS)].concat());
    required.insert("confidence", of(CONFIDENCES));
    required.insert("format", of(LIBRARY_FORMATS));
    required.insert("root", of(ROOT_KINDS));
    required.insert(
        "download",
        [
            of(DOWNLOAD_STATES),
            DOWNLOAD_KINDS.iter().map(|k| format!("kind_{k}")).collect(),
        ]
        .concat(),
    );
    required.insert("mode", of(MODES));
    required.insert("check", check_required().into_keys().collect());
    required.insert("alert", [of(V2_ALERT_CODES), of(ALERT_CODES)].concat());
    let stale: Vec<String> = concepts()
        .into_iter()
        .filter(|key| {
            let (family, value) = key.split_once('.').unwrap_or((key.as_str(), ""));
            required
                .get(family)
                .is_some_and(|vals| !vals.iter().any(|v| v == value))
        })
        .collect();
    expect_none(
        "concepts.json explains values that no longer exist (remove or rename)",
        &stale,
    );
}

/// Port of test_code_scan_finds_the_known_emitters (with the adapters' anchors).
#[test]
fn test_code_scan_finds_the_known_emitters() {
    let codes = check_codes();
    let anchors = [
        "invalid_params",
        "fit",
        "gpu_shared",
        "nvfp4_moe_sm120",
        "gpus_insufficient",
    ];
    let lost: Vec<String> = anchors
        .iter()
        .filter(|c| !codes.contains_key(**c))
        .map(|c| (*c).to_string())
        .collect();
    expect_none("the code scan no longer finds these emitters", &lost);
    assert!(
        ["host_ram_low", "shmem_high", "swap_heavy"]
            .iter()
            .all(|c| ALERT_CODES.contains(c))
    );
}

/// The lifecycle half of the row above: the scan finds the lifecycle service's preflight emitters.
#[test]
fn test_code_scan_finds_the_lifecycle_emitters() {
    let codes = check_codes();
    for anchor in [
        "invalid_params",
        "fit",
        "gpu_shared",
        "host_down",
        "no_free_port",
        "trust_remote_code",
    ] {
        let at = codes
            .get(anchor)
            .unwrap_or_else(|| panic!("the scan lost {anchor}"));
        assert!(
            at.iter().any(|w| w.starts_with("services/lifecycle.rs:")),
            "{anchor}: {at:?}"
        );
    }
    assert!(
        ["host_ram_low", "shmem_high", "swap_heavy"]
            .iter()
            .all(|c| ALERT_CODES.contains(c))
    );
}

/// Port of test_dynamic_check_codes_are_still_raised: each dynamic check code is still raised with a literal code.
#[test]
fn test_dynamic_check_codes_are_still_raised() {
    let raised = raised_codes();
    let gone: Vec<String> = DYNAMIC_CHECK_CODES
        .iter()
        .filter(|c| !raised.contains_key(**c))
        .map(|c| (*c).to_string())
        .collect();
    expect_none(
        "DYNAMIC_CHECK_CODES names codes that no problem raises any more",
        &gone,
    );
    assert!(
        raised["secret_in_argv"]
            .iter()
            .any(|w| w.starts_with("services/lifecycle.rs:"))
    );
}

/// Port of test_gate_scans_code_for_emitted_codes: the scanner's own test, on a Rust fixture shaped like Python's.
#[test]
fn test_gate_scans_code_for_emitted_codes() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("m.rs"),
        "fn f() {\n\
         \x20   let a = Compat { level: \"warn\".into(),\n\
         \x20                    code: \"multi_line\".into(), message: \"x\".into() };\n\
         \x20   let b = PreflightCheck { code: \"plain\".into(), level: \"ok\".into(), message: \"y\".into() };\n\
         \x20   add(\"block\", \"via_helper\", \"z\");\n\
         \x20   add(level, code, msg);\n\
         \x20   other(code = \"ignored\");\n\
         \x20   PreflightCheck { code: e.code.clone(), level: \"block\".into(), message: \"dynamic\".into() };\n\
         }\n",
    )
    .unwrap();
    let found = emitted_codes(&["PreflightCheck", "Compat"], &["add"], dir.path());
    let want: BTreeMap<String, Vec<String>> = [
        ("multi_line", "m.rs:2"),
        ("plain", "m.rs:4"),
        ("via_helper", "m.rs:5"),
    ]
    .iter()
    .map(|(c, w)| ((*c).to_string(), vec![(*w).to_string()]))
    .collect();
    assert_eq!(found, want);
}
