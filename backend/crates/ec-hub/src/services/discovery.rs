//! Python: backend/src/engine_console/services/discovery.py.
//!
//! Pure helpers for engines the console did not create (contract v2). Finding them is the node's job: it lists
//! running containers, classifies them, redacts their argv, probes their loopback endpoints and pushes all of that in
//! its `inventory` message. What stays on the hub is text hygiene for names that end up in instance ids, and the
//! generic Prometheus map for engines that have no adapter. Everything a node says is untrusted: length-capped,
//! control characters stripped, never executed.

use std::sync::LazyLock;

use regex::Regex;

/// external instance ids: `ext-<host>-<slug>`
pub const EXT: &str = "ext-";
/// script instance ids: `scr-<host>-<script name>`
pub const SCRIPT: &str = "scr-";

/// generic Prometheus -> canonical keys, for engines without an adapter (missing keys are simply omitted)
pub const GENERIC_METRICS: &[(&str, &[(&str, &str)])] = &[(
    "llamacpp",
    &[
        ("llamacpp:requests_processing", "requests_running"),
        ("llamacpp:requests_deferred", "requests_waiting"),
        ("llamacpp:prompt_tokens_total", "prompt_tokens_total"),
        ("llamacpp:tokens_predicted_total", "generation_tokens_total"),
    ],
)];

static CTRL: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[\x00-\x1f\x7f]").unwrap_or_else(|_| unreachable!()));
static NOT_SLUG: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[^a-z0-9]+").unwrap_or_else(|_| unreachable!()));
static NOT_DISPLAY: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[^A-Za-z0-9 _.:\-]").unwrap_or_else(|_| unreachable!()));

/// `s[:n]` in characters.
pub(crate) fn take(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

/// `clean_text(s, limit)`: control characters removed, then at most `limit` characters.
pub fn clean_text(s: &str, limit: usize) -> String {
    take(&CTRL.replace_all(s, ""), limit)
}

/// `clean_text(s)` with Python's default limit (200).
pub fn clean_text200(s: &str) -> String {
    clean_text(s, 200)
}

/// `slug(s)`: lower case, runs of anything but `[a-z0-9]` become one `-`, trimmed, at most 48, else "unknown".
pub fn slug(s: &str) -> String {
    let lower = s.to_lowercase();
    let out = take(NOT_SLUG.replace_all(&lower, "-").trim_matches('-'), 48);
    if out.is_empty() {
        "unknown".into()
    } else {
        out
    }
}

/// `display_name(s)`: control characters dropped, then everything but `[A-Za-z0-9 _.:-]`, trimmed of " .-", at
/// most 64, else "external".
pub fn display_name(s: &str) -> String {
    let no_ctrl = CTRL.replace_all(s, "");
    let kept = NOT_DISPLAY.replace_all(&no_ctrl, "");
    let out = take(kept.trim_matches([' ', '.', '-']), 64);
    if out.is_empty() {
        "external".into()
    } else {
        out
    }
}

/// Python's `float(token)` (what a Prometheus value may look like); `None` where Python raises ValueError.
pub(crate) fn py_float(tok: &str) -> Option<f64> {
    let t = tok.trim_matches(py_is_space);
    let lower = t.to_ascii_lowercase();
    let (sign, body) = match lower.strip_prefix('-') {
        Some(r) => (-1.0, r),
        None => (1.0, lower.strip_prefix('+').unwrap_or(&lower)),
    };
    match body {
        "inf" | "infinity" => return Some(sign * f64::INFINITY),
        "nan" => return Some(f64::NAN),
        _ => {}
    }
    let b = body.as_bytes();
    if b.is_empty()
        || !b[0].is_ascii_digit() && b[0] != b'.'
        || !b
            .iter()
            .all(|c| c.is_ascii_digit() || matches!(c, b'.' | b'e' | b'+' | b'-' | b'_'))
    {
        return None;
    }
    // an underscore only between two digits
    for (i, c) in b.iter().enumerate() {
        if *c == b'_'
            && !(i > 0 && i + 1 < b.len() && b[i - 1].is_ascii_digit() && b[i + 1].is_ascii_digit())
        {
            return None;
        }
    }
    let cleaned: String = body.chars().filter(|c| *c != '_').collect();
    cleaned.parse::<f64>().ok().map(|v| sign * v)
}

/// `str.isspace()` for one character (what `str.split()` splits on).
pub(crate) fn py_is_space(c: char) -> bool {
    c.is_whitespace() || matches!(c, '\u{1c}'..='\u{1f}')
}

/// Map whatever canonical keys an adapter-less engine exposes; everything else is ignored. Ordered like Python's
/// dict (the first assignment fixes the position, later ones update the value).
pub fn parse_generic_metrics(engine: &str, text: &str) -> Vec<(String, f64)> {
    let Some((_, mapping)) = GENERIC_METRICS.iter().find(|(e, _)| *e == engine) else {
        return Vec::new();
    };
    let mut out: Vec<(String, f64)> = Vec::new();
    for line in crate::services::common::py_splitlines(text) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (name, rest) = line.split_once(' ').unwrap_or((line, ""));
        let name = name.split('{').next().unwrap_or(name);
        let Some((_, key)) = mapping.iter().find(|(n, _)| *n == name) else {
            continue;
        };
        // `float(rest.split()[0])`
        let Some(first) = rest.split(py_is_space).find(|t| !t.is_empty()) else {
            continue;
        };
        let Some(v) = py_float(first) else { continue };
        if let Some(slot) = out.iter_mut().find(|(k, _)| k == key) {
            slot.1 = v;
        } else {
            out.push(((*key).to_string(), v));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // Port of tests/core/test_helpers.py::test_text_hygiene_helpers.
    #[test]
    fn test_text_hygiene_helpers() {
        assert_eq!((EXT, SCRIPT), ("ext-", "scr-"));
        assert_eq!(slug("My Engine!"), "my-engine");
        assert_eq!(slug("../../etc"), "etc");
        assert_eq!(slug(""), "unknown");
        assert_eq!(slug(&"x".repeat(99)).len(), 48);
        assert_eq!(display_name("a\x00b\x1b[0m c"), "ab0m c");
        assert_eq!(display_name("!!!"), "external");
        assert_eq!(display_name(&"x".repeat(99)).chars().count(), 64);
        assert_eq!(clean_text("a\nb\x00c", 5), "abc");
        assert_eq!(clean_text200(&"x".repeat(300)), "x".repeat(200));
    }

    // Port of tests/core/test_helpers.py::test_generic_metrics_map_only_what_it_knows_and_survives_garbage.
    #[test]
    fn test_generic_metrics_map_only_what_it_knows_and_survives_garbage() {
        let text = "# HELP x\nllamacpp:requests_processing 3\nllamacpp:requests_deferred{a=\"b\"} 1\n\
                    llamacpp:tokens_predicted_total 9.5\nother 5\nllamacpp:prompt_tokens_total notanumber\n\n";
        assert_eq!(
            parse_generic_metrics("llamacpp", text),
            vec![
                ("requests_running".to_string(), 3.0),
                ("requests_waiting".to_string(), 1.0),
                ("generation_tokens_total".to_string(), 9.5)
            ]
        );
        assert!(parse_generic_metrics("unknown-engine", text).is_empty());
        assert!(parse_generic_metrics("llamacpp", "").is_empty());
    }

    #[test]
    fn test_py_float_accepts_what_python_float_accepts() {
        for (s, v) in [
            ("3", 3.0),
            ("9.5", 9.5),
            ("1e3", 1000.0),
            ("-2", -2.0),
            (".5", 0.5),
            ("5.", 5.0),
            ("1_000", 1000.0),
            ("+1.5E2", 150.0),
        ] {
            assert_eq!(py_float(s), Some(v), "{s}");
        }
        assert!(py_float("+Inf").is_some_and(f64::is_infinite));
        assert!(py_float("-infinity").is_some_and(|v| v.is_infinite() && v < 0.0));
        assert!(py_float("NaN").is_some_and(f64::is_nan));
        for bad in [
            "notanumber",
            "",
            "1__0",
            "_1",
            "0x10",
            "1e",
            "--1",
            "+-1",
            "1_",
            "e5",
        ] {
            assert_eq!(py_float(bad), None, "{bad}");
        }
    }

    proptest::proptest! {
        // whatever a node sends, the hub-side names stay inside their alphabets and caps
        #[test]
        fn prop_slug_display_name_and_clean_text_are_safe(s in "\\PC{0,200}") {
            let sl = slug(&s);
            proptest::prop_assert!(sl.len() <= 48);
            proptest::prop_assert!(sl.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-'));
            proptest::prop_assert!(!sl.starts_with('-') && !sl.contains("--"));
            let d = display_name(&s);
            proptest::prop_assert!(d.chars().count() <= 64);
            proptest::prop_assert!(d.chars().all(|c| c.is_ascii_alphanumeric() || " _.:-".contains(c)));
            let c = clean_text(&s, 50);
            proptest::prop_assert!(c.chars().count() <= 50 && !c.chars().any(|ch| (ch as u32) < 0x20 || ch == '\x7f'));
        }
    }
}
